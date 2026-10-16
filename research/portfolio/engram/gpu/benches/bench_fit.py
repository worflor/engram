"""
bench_fit.py

bake-off: GPU (wgpu) fit_pair kernel vs numpy fit_all.

synthesizes a realistic batch — N articles, each segmented into a
handful of blocks of length L — then fits every (segment, pair) on
both backends and compares.

correctness:
    max |K_gpu - K_cpu|, max |G_gpu - G_cpu| relative to |K_cpu|
    (f32 vs f64: ~1e-4 relative is the ceiling we can hope for)

timing:
    wall clock per backend, warmup passes ignored.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import numpy as np

from engram_codec import fit_all, _to_complex, SEED_COUNT, MIN_BLOCK, MAX_BLOCK
from engram.gpu.engine import GpuEngine, pack_batch, gpu_available


# ── synthetic batch generator ─────────────────────────────────────────

def synth_segment(T: int, D: int, seed: int) -> np.ndarray:
    """
    make a [T, D] float32 trajectory that's physically plausible:
    D/2 independent damped complex oscillators with known K, G, plus noise.
    the solver should recover K, G to within numerical precision.
    """
    rng = np.random.default_rng(seed)
    P = D // 2
    freqs = rng.uniform(0.1, 2.0, size=P)
    damps = rng.uniform(0.90, 0.995, size=P)
    K = 2.0 * damps * np.cos(freqs)
    G = damps * damps

    z = np.zeros((T, P), dtype=np.complex128)
    z[0] = rng.standard_normal(P) + 1j * rng.standard_normal(P)
    z[1] = rng.standard_normal(P) + 1j * rng.standard_normal(P)
    for n in range(2, T):
        z[n] = K * z[n - 1] - G * z[n - 2]
    # add a touch of noise so RMS is non-zero
    z = z + 0.01 * (rng.standard_normal((T, P)) + 1j * rng.standard_normal((T, P)))

    out = np.empty((T, D), dtype=np.float32)
    out[:, 0::2] = z.real.astype(np.float32)
    out[:, 1::2] = z.imag.astype(np.float32)
    return out


def synth_batch(n_articles: int, dim: int, seed: int = 0):
    """
    emulate what _ingest_worker produces per article: a handful of
    variable-length segments between MIN_BLOCK+1 and MAX_BLOCK timesteps.
    """
    rng = np.random.default_rng(seed)
    segments = []
    for i in range(n_articles):
        n_segs_here = int(rng.integers(3, 8))
        for j in range(n_segs_here):
            T = int(rng.integers(MIN_BLOCK + 4, MAX_BLOCK + 1))
            segments.append(synth_segment(T, dim, seed + 1000 * i + j))
    return segments


# ── reference CPU path ────────────────────────────────────────────────

def cpu_fit(segments):
    """run fit_all over each segment; return stacked K, G, rms."""
    P = segments[0].shape[1] // 2
    out_K = np.empty((len(segments), P), dtype=np.complex128)
    out_G = np.empty((len(segments), P), dtype=np.complex128)
    out_rms = np.empty((len(segments), P), dtype=np.float32)
    for i, s in enumerate(segments):
        z_c = _to_complex(s.astype(np.float32))
        K, G, _, rms = fit_all(z_c)
        out_K[i] = K
        out_G[i] = G
        out_rms[i] = rms
    return out_K, out_G, out_rms


# ── GPU path ──────────────────────────────────────────────────────────

def gpu_fit(engine: GpuEngine, segments):
    traj, segs = pack_batch(segments)
    K, G, rms = engine.fit_batch(traj, segs)
    P = int(segs[0, 2]) // 2
    n = len(segments)
    return K.reshape(n, P), G.reshape(n, P), rms.reshape(n, P)


# ── main ──────────────────────────────────────────────────────────────

def report_diff(name, gpu, cpu):
    mag = np.maximum(np.abs(cpu).max(), 1.0)
    adiff = np.abs(gpu.astype(np.complex128) - cpu)
    rdiff = adiff / np.maximum(np.abs(cpu), 1e-6)
    print(f"  {name}: abs.max={adiff.max():.3e}  rel.max={rdiff.max():.3e}  "
          f"rel.median={np.median(rdiff):.3e}")


def run():
    if not gpu_available():
        print("wgpu not available. exiting.")
        return

    engine = GpuEngine()
    print(f"adapter: {engine.describe()}")
    print()

    DIM = 300  # match alexandria's GloVe dim

    for n_articles in (8, 32, 64, 128, 256):
        segments = synth_batch(n_articles, DIM, seed=42)
        total_segs = len(segments)
        total_floats = sum(s.size for s in segments)
        print(f"-- n_articles={n_articles}  segments={total_segs}  "
              f"data={total_floats * 4 / 1e6:.2f} MB --")

        # warmup
        _ = gpu_fit(engine, segments)
        _ = cpu_fit(segments[:2])

        # CPU
        t0 = time.perf_counter()
        K_cpu, G_cpu, rms_cpu = cpu_fit(segments)
        t_cpu = time.perf_counter() - t0

        # GPU
        t0 = time.perf_counter()
        K_gpu, G_gpu, rms_gpu = gpu_fit(engine, segments)
        t_gpu = time.perf_counter() - t0

        speedup = t_cpu / max(t_gpu, 1e-9)
        print(f"  CPU fit_all : {t_cpu * 1000:8.2f} ms")
        print(f"  GPU fit_pair: {t_gpu * 1000:8.2f} ms   ({speedup:.2f}x)")
        report_diff("K  ", K_gpu, K_cpu)
        report_diff("G  ", G_gpu, G_cpu)
        report_diff("rms", rms_gpu.astype(np.complex128), rms_cpu.astype(np.complex128))
        print()


if __name__ == "__main__":
    run()
