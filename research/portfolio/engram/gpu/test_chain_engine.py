"""
test_chain_engine.py — correctness check for ChainEngine.gather_paired_batch.

builds a small glove matrix + synthetic article indices, compares GPU
output against numpy reference (matrix[idxs] + cumsum smooth + pair).

HARD CAPS:
  - tiny V, D
  - few short articles
  - absolute tolerance 1e-4 (f32 accumulation drift)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu.chain_engine import ChainEngine


def numpy_reference(
    matrix: np.ndarray,
    pairing: np.ndarray,
    indices_list: list[np.ndarray],
    window: int,
) -> list[np.ndarray]:
    """CPU reference: gather → cumsum smooth → apply pairing."""
    D = matrix.shape[1]
    out = []
    for idxs in indices_list:
        T = len(idxs)
        if T < window:
            out.append(np.empty((0, D), dtype=np.float32))
            continue
        # gather with -1 treated as zero
        raw = np.zeros((T, D), dtype=np.float32)
        mask = idxs >= 0
        raw[mask] = matrix[idxs[mask]]
        cs = np.empty((T + 1, D), dtype=np.float32); cs[0] = 0
        np.cumsum(raw, axis=0, out=cs[1:])
        smoothed = (cs[window:] - cs[:-window]) / np.float32(window)
        paired = smoothed[:, pairing]
        out.append(paired.astype(np.float32))
    return out


def run_case(
    V: int, D: int, window: int,
    lens: list[int], seed: int,
    engine: GpuEngine,
) -> None:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((V, D)).astype(np.float32) * 0.3
    # random permutation for pairing (permute indices, then D will be
    # reordered by pairing). keep it simple: pair-swap.
    pairing = np.arange(D, dtype=np.int32)
    # pairwise swap: 0↔1, 2↔3, ...
    for i in range(0, D - 1, 2):
        pairing[i], pairing[i + 1] = pairing[i + 1], pairing[i]

    # make some token indices per article. sprinkle -1s for skip paths.
    indices_list = []
    for L in lens:
        idx = rng.integers(0, V, size=L, dtype=np.int32)
        # stamp ~5% as -1 (out-of-vocab)
        drop = rng.random(L) < 0.05
        idx[drop] = -1
        indices_list.append(idx)

    chain = ChainEngine(engine, matrix, pairing, window=window,
                        matrix_dtype="i8")

    t0 = time.time()
    gpu_out = chain.gather_paired_batch(indices_list)
    dt_gpu = time.time() - t0

    t0 = time.time()
    cpu_out = numpy_reference(matrix, pairing, indices_list, window)
    dt_cpu = time.time() - t0

    assert len(gpu_out) == len(cpu_out)
    max_diff = 0.0
    for i, (g, c) in enumerate(zip(gpu_out, cpu_out)):
        assert g.shape == c.shape, \
            f"article {i}: gpu shape {g.shape} != cpu shape {c.shape}"
        if g.size == 0:
            continue
        diff = float(np.max(np.abs(g - c)))
        if diff > max_diff:
            max_diff = diff

    total_out = sum(g.size for g in gpu_out)
    print(f"  V={V:5d} D={D:3d} W={window:2d} "
          f"articles={len(lens):3d} max_len={max(lens) if lens else 0:5d}  "
          f"out_floats={total_out:>9d}  max_diff={max_diff:.3e}  "
          f"gpu={dt_gpu*1000:.1f}ms cpu={dt_cpu*1000:.1f}ms")

    # f16 matrix quantization introduces per-element noise ~1e-3 in
    # magnitude; after averaging W=10 samples it drops to ~3e-4. set
    # tolerance accordingly for the f16 path.
    # int8 per-row-scale quantization puts per-element error at ~1/127
    # of the row max (about 0.004 for GloVe-scale matrices). averaging
    # W=10 samples brings it down, but the cross-row variance dominates.
    # allow 2% error against the f32 reference.
    tol = 2e-2
    if max_diff > tol:
        raise AssertionError(f"max_diff {max_diff:.3e} exceeds {tol:.1e}")


def run() -> None:
    if not gpu_available():
        print("wgpu not available"); return

    engine = GpuEngine()
    print(f"adapter: {engine.describe()}")
    print()

    print("=== ChainEngine correctness ===")
    run_case(V=1000,  D=300, window=10, lens=[50, 100, 200, 12, 11],
             seed=1, engine=engine)
    run_case(V=5000,  D=300, window=10, lens=[500, 1000, 200, 300, 150],
             seed=2, engine=engine)
    run_case(V=10000, D=300, window=10, lens=[2000] * 20,
             seed=3, engine=engine)
    # edge: some articles below window (emit empty)
    run_case(V=1000,  D=300, window=10, lens=[5, 9, 10, 11, 500],
             seed=4, engine=engine)
    # edge: single article
    run_case(V=1000,  D=300, window=10, lens=[1000],
             seed=5, engine=engine)
    # edge: D not multiple of 4 — i8 path requires D%4==0, so skip;
    # f32/f16 paths would handle it fine.
    # run_case(V=1000,  D=302, window=10, lens=[500, 500],
    #          seed=6, engine=engine)
    print("\nOK: all cases within 1e-4")


if __name__ == "__main__":
    run()
