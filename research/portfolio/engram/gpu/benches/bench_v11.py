"""bench_v11.py — final shootout including DF arithmetic."""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import numpy as np
from engram_codec import fit_all, _to_complex, MIN_BLOCK, MAX_BLOCK
from engram.gpu.engine import GpuEngine, pack_batch


def synth_segment(T, D, seed):
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


def synth_batch(n_articles, dim, seed=0):
    rng = np.random.default_rng(seed)
    segments = []
    for i in range(n_articles):
        for j in range(int(rng.integers(3, 8))):
            T = int(rng.integers(MIN_BLOCK + 4, MAX_BLOCK + 1))
            segments.append(synth_segment(T, dim, seed + 1000 * i + j))
    return segments


def run():
    eng_v3 = GpuEngine(shader="fit_pair_v3.wgsl")
    eng_v5 = GpuEngine(shader="fit_pair_v5_noresid.wgsl")
    eng_v11 = GpuEngine(shader="fit_pair_v11_df.wgsl")
    print(f"adapter: {eng_v3.describe()}")
    print()

    print(f"{'n_art':>6} {'segs':>6} {'cpu':>8} {'v3':>8} {'v5':>8} {'v11':>8} "
          f"{'v3x':>6} {'v5x':>6} {'v11x':>6}")
    print("-" * 80)

    for n_art in (32, 128, 256, 512):
        segments = synth_batch(n_art, 300, seed=42)
        n_segs = len(segments)
        P = segments[0].shape[1] // 2

        K_cpu = np.empty((n_segs, P), dtype=np.complex128)
        G_cpu = np.empty((n_segs, P), dtype=np.complex128)
        rms_cpu = np.empty((n_segs, P), dtype=np.float32)
        t0 = time.perf_counter()
        for i, s in enumerate(segments):
            z_c = _to_complex(s.astype(np.float32))
            K, G, _, rms = fit_all(z_c)
            K_cpu[i] = K; G_cpu[i] = G; rms_cpu[i] = rms
        t_cpu = time.perf_counter() - t0

        def timed(eng):
            for _ in range(3):
                traj, segs = pack_batch(segments)
                _ = eng.fit_batch(traj, segs)
            best = float("inf")
            for _ in range(5):
                traj, segs = pack_batch(segments)
                t0 = time.perf_counter()
                out = eng.fit_batch(traj, segs)
                dt = time.perf_counter() - t0
                if dt < best: best = dt
            return out, best

        (K3, G3, rms3), t3 = timed(eng_v3)
        (K5, G5, rms5), t5 = timed(eng_v5)
        (K11, G11, rms11), t11 = timed(eng_v11)

        print(f"{n_art:>6} {n_segs:>6} "
              f"{t_cpu*1e3:>7.1f} {t3*1e3:>7.1f} {t5*1e3:>7.1f} {t11*1e3:>7.1f} "
              f"{t_cpu/t3:>5.2f}x {t_cpu/t5:>5.2f}x {t_cpu/t11:>5.2f}x")

        if n_art == 256:
            K3 = K3.reshape(n_segs, P); G3 = G3.reshape(n_segs, P); rms3 = rms3.reshape(n_segs, P)
            K11 = K11.reshape(n_segs, P); G11 = G11.reshape(n_segs, P); rms11 = rms11.reshape(n_segs, P)
            def err(a, b):
                denom = np.maximum(np.abs(b), 1e-6)
                return float(np.median(np.abs(a.astype(np.complex128) - b) / denom))
            def errmax(a, b):
                denom = np.maximum(np.abs(b), 1e-6)
                return float(np.max(np.abs(a.astype(np.complex128) - b) / denom))
            print(f"\n  accuracy vs cpu f64:")
            for name, K, G, rms in [("v3", K3, G3, rms3), ("v11", K11, G11, rms11)]:
                rcomp = rms.astype(np.complex128); rref = rms_cpu.astype(np.complex128)
                print(f"    {name:>4}: K med={err(K,K_cpu):.2e} max={errmax(K,K_cpu):.2e}  "
                      f"G med={err(G,G_cpu):.2e} max={errmax(G,G_cpu):.2e}  "
                      f"rms med={err(rcomp,rref):.2e} max={errmax(rcomp,rref):.2e}")
            print()


if __name__ == "__main__":
    run()
