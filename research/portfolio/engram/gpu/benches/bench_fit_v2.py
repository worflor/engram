"""
bench_fit_v2.py

v1 (shared-memory reduction, 2-pass rms, pair-per-workgroup) vs.
v2 (pair-per-thread, coalesced reads, single-pass algebraic rms) vs.
cpu fit_all. same synthetic data, same seeds. reports timing and
numerical agreement for both gpu paths so regressions in either are
visible.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import numpy as np

from engram_codec import fit_all, _to_complex, MIN_BLOCK, MAX_BLOCK
from engram.gpu.engine import GpuEngine, pack_batch, gpu_available


def synth_segment(T: int, D: int, seed: int) -> np.ndarray:
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
    z = z + 0.01 * (rng.standard_normal((T, P)) + 1j * rng.standard_normal((T, P)))

    out = np.empty((T, D), dtype=np.float32)
    out[:, 0::2] = z.real.astype(np.float32)
    out[:, 1::2] = z.imag.astype(np.float32)
    return out


def synth_batch(n_articles: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    segments = []
    for i in range(n_articles):
        n_segs_here = int(rng.integers(3, 8))
        for j in range(n_segs_here):
            T = int(rng.integers(MIN_BLOCK + 4, MAX_BLOCK + 1))
            segments.append(synth_segment(T, dim, seed + 1000 * i + j))
    return segments


def cpu_fit(segments):
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


def gpu_fit(engine: GpuEngine, segments):
    traj, segs = pack_batch(segments)
    K, G, rms = engine.fit_batch(traj, segs)
    P = int(segs[0, 2]) // 2
    n = len(segments)
    return K.reshape(n, P), G.reshape(n, P), rms.reshape(n, P)


def timed_gpu(engine, segments, iters=3):
    # warmup
    _ = gpu_fit(engine, segments)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        out = gpu_fit(engine, segments)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return out, best


def rel_err(gpu, cpu):
    adiff = np.abs(gpu.astype(np.complex128) - cpu)
    rdiff = adiff / np.maximum(np.abs(cpu), 1e-6)
    return float(adiff.max()), float(rdiff.max()), float(np.median(rdiff))


def run():
    if not gpu_available():
        print("wgpu not available.")
        return

    eng_v1 = GpuEngine(shader="fit_pair.wgsl")
    eng_v3 = GpuEngine(shader="fit_pair_v3.wgsl")
    eng_v4 = GpuEngine(shader="fit_pair_v4.wgsl")
    print(f"adapter: {eng_v1.describe()}")
    print()

    DIM = 300

    header = f"{'n_art':>6} {'segs':>6} {'cpu':>8} {'v1':>8} {'v3':>8} {'v4':>8} {'v1x':>6} {'v3x':>6} {'v4x':>6}"
    print(header)
    print("-" * len(header))

    for n_articles in (32, 128, 256, 512):
        segments = synth_batch(n_articles, DIM, seed=42)
        total_segs = len(segments)

        _ = cpu_fit(segments[:2])
        t0 = time.perf_counter()
        K_cpu, G_cpu, rms_cpu = cpu_fit(segments)
        t_cpu = time.perf_counter() - t0

        (K1, G1, rms1), t_v1 = timed_gpu(eng_v1, segments)
        (K3, G3, rms3), t_v3 = timed_gpu(eng_v3, segments)
        (K4, G4, rms4), t_v4 = timed_gpu(eng_v4, segments)

        print(f"{n_articles:>6} {total_segs:>6} "
              f"{t_cpu*1e3:>7.1f} {t_v1*1e3:>7.1f} {t_v3*1e3:>7.1f} {t_v4*1e3:>7.1f} "
              f"{t_cpu/t_v1:>5.2f}x {t_cpu/t_v3:>5.2f}x {t_cpu/t_v4:>5.2f}x")

        if n_articles == 256:
            print()
            for name, K, G, rms in [("v1", K1, G1, rms1),
                                     ("v3", K3, G3, rms3),
                                     ("v4", K4, G4, rms4)]:
                print(f"  {name} accuracy vs cpu (f64):")
                ak, rk, mk = rel_err(K, K_cpu)
                print(f"    K   abs.max={ak:.3e}  rel.max={rk:.3e}  rel.med={mk:.3e}")
                ag, rg, mg = rel_err(G, G_cpu)
                print(f"    G   abs.max={ag:.3e}  rel.max={rg:.3e}  rel.med={mg:.3e}")
                ar, rr, mr = rel_err(rms.astype(np.complex128), rms_cpu.astype(np.complex128))
                print(f"    rms abs.max={ar:.3e}  rel.max={rr:.3e}  rel.med={mr:.3e}")
            print()


if __name__ == "__main__":
    run()
