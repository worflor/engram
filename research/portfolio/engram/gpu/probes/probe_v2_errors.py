"""
probe_v2_errors.py — characterize the precision catastrophe in v2.

v2's algebraic rms identity had median rel error 1.5e-4 (fine) but max
0.28 (catastrophic). before committing to DF emulation or hybrid
detection, find out WHICH pairs fail and how many. maybe most are fine.
"""

from __future__ import annotations

import sys
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
        n_segs_here = int(rng.integers(3, 8))
        for j in range(n_segs_here):
            T = int(rng.integers(MIN_BLOCK + 4, MAX_BLOCK + 1))
            segments.append(synth_segment(T, dim, seed + 1000 * i + j))
    return segments


def run():
    segments = synth_batch(256, 300, seed=42)
    n_segs = len(segments)
    P = segments[0].shape[1] // 2

    # cpu reference
    K_cpu = np.empty((n_segs, P), dtype=np.complex128)
    G_cpu = np.empty((n_segs, P), dtype=np.complex128)
    rms_cpu = np.empty((n_segs, P), dtype=np.float32)
    for i, s in enumerate(segments):
        z_c = _to_complex(s.astype(np.float32))
        K, G, _, rms = fit_all(z_c)
        K_cpu[i] = K
        G_cpu[i] = G
        rms_cpu[i] = rms

    eng_v2 = GpuEngine(shader="fit_pair_v2.wgsl")
    eng_v3 = GpuEngine(shader="fit_pair_v3.wgsl")

    traj, segs = pack_batch(segments)
    K2, G2, rms2 = eng_v2.fit_batch(traj, segs)
    K3, G3, rms3 = eng_v3.fit_batch(traj, segs)
    rms2 = rms2.reshape(n_segs, P)
    rms3 = rms3.reshape(n_segs, P)

    # per-pair relative rms error
    denom = np.maximum(rms_cpu.astype(np.float64), 1e-10)
    err2 = np.abs(rms2.astype(np.float64) - rms_cpu.astype(np.float64)) / denom
    err3 = np.abs(rms3.astype(np.float64) - rms_cpu.astype(np.float64)) / denom

    total = err2.size
    for thresh in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        n2 = int((err2 > thresh).sum())
        n3 = int((err3 > thresh).sum())
        print(f"  rel err > {thresh:>6.0e}:  v2={n2:>6} ({n2/total*100:5.2f}%)   v3={n3:>6} ({n3/total*100:5.2f}%)")

    print()
    print(f"total pairs: {total}")
    print(f"v2: max={err2.max():.3e}  99th={np.percentile(err2, 99):.3e}  median={np.median(err2):.3e}")
    print(f"v3: max={err3.max():.3e}  99th={np.percentile(err3, 99):.3e}  median={np.median(err3):.3e}")

    # profile: what characterizes the catastrophic-error pairs?
    bad_mask = err2 > 1e-2  # >1% rel error
    if bad_mask.any():
        print(f"\n{bad_mask.sum()} pairs with v2 rel err > 1%:")
        # what's special about them?
        print(f"  their cpu rms distribution: min={rms_cpu[bad_mask].min():.3e}  "
              f"med={np.median(rms_cpu[bad_mask]):.3e}  "
              f"max={rms_cpu[bad_mask].max():.3e}")
        print(f"  all-pairs cpu rms:           min={rms_cpu.min():.3e}  "
              f"med={np.median(rms_cpu):.3e}  "
              f"max={rms_cpu.max():.3e}")
        # correlate with |K|: are catastrophic cases where K is large?
        K_mag = np.abs(K_cpu)
        print(f"  bad pairs |K|:     min={K_mag[bad_mask].min():.3f}  "
              f"med={np.median(K_mag[bad_mask]):.3f}  "
              f"max={K_mag[bad_mask].max():.3f}")
        print(f"  all-pairs |K|:     min={K_mag.min():.3f}  "
              f"med={np.median(K_mag):.3f}  "
              f"max={K_mag.max():.3f}")


if __name__ == "__main__":
    run()
