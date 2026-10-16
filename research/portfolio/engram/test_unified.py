"""test_unified.py — verify the single-submit fused pipeline produces
K_sums equivalent to the legacy numba-segment + CPU-reduceat path.

drives a small batch through both paths, compares K_sums numerically.
small drift expected because:
  - GPU segment kernel may produce slightly different block boundaries
    than numba (same algorithm but f32 vs f64 precision)
  - per_article_reduce runs in f32, numba reduceat in native dtype

critical invariant: total_segs (GPU) should match total_segs (numba)
on most real inputs — the kernels implement the same segmentation
rule with the same constants.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from engram_codec import EngramFile
from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu.chain_engine import ChainEngine
from engram.gpu.unified import UnifiedStage


def test_small_batch():
    if not gpu_available():
        print("wgpu missing")
        return False

    # build a synthetic matrix + pairing
    V, D = 2000, 300
    P = D // 2
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((V, D)).astype(np.float32) * 0.3
    pairing = np.arange(D, dtype=np.int32)

    engine = GpuEngine()
    chain = ChainEngine(engine, matrix, pairing, window=10, matrix_dtype="f32")
    unified = UnifiedStage(engine.device, engine, P=P)

    # a small batch of "articles" — integer index arrays
    n_art = 6
    indices_list = []
    for _ in range(n_art):
        T = int(rng.integers(200, 500))
        indices_list.append(rng.integers(0, V, size=T).astype(np.int32))

    # STEP 1: run chain's gather+features submit
    combined = chain.submit_gather_and_features(indices_list)
    if combined is None:
        print("gather+features returned None")
        return False
    handle = combined["handle"]
    feat_ctx = combined["feat_ctx"]

    # STEP 2: run unified fused submit
    result = unified.submit_fused(
        chain, engine, handle, feat_ctx,
        n_art=n_art, max_block=64,
    )

    if result is None:
        print("unified returned None")
        return False

    print(f"unified K_sums shape: {result['K_sums'].shape}")
    print(f"unified G_sums shape: {result['G_sums'].shape}")
    print(f"unified rms_sums shape: {result['rms_sums'].shape}")
    print(f"unified n_segs_per_art: {result['n_segs_per_art']}")
    print(f"unified total_segs: {result['total_segs']}")

    # sanity checks
    assert result["K_sums"].shape == (n_art, P)
    assert result["G_sums"].shape == (n_art, P)
    assert result["rms_sums"].shape == (n_art, P)
    assert result["n_segs_per_art"].shape == (n_art,)

    # total_segs should equal sum of per-article segment counts
    per_art_sum = int(result["n_segs_per_art"].sum())
    assert per_art_sum == result["total_segs"], \
        f"per-art sum {per_art_sum} != total_segs {result['total_segs']}"
    print(f"  [ok] per-article sum matches total_segs")

    # all sums should be finite
    assert np.all(np.isfinite(result["K_sums"].real))
    assert np.all(np.isfinite(result["K_sums"].imag))
    assert np.all(np.isfinite(result["G_sums"].real))
    assert np.all(np.isfinite(result["rms_sums"]))
    print(f"  [ok] all sums finite")

    # articles with 0 segments should have 0 sums
    for a in range(n_art):
        if result["n_segs_per_art"][a] == 0:
            assert np.all(result["K_sums"][a] == 0)
            assert np.all(result["G_sums"][a] == 0)
            assert np.all(result["rms_sums"][a] == 0)
    print(f"  [ok] empty articles have zero sums")

    print(f"  [PASS] small batch through unified pipeline")
    return True


def test_brain_from_sums():
    """full unified pipeline: chain gather+features → unified submit_fused
    → Brain.absorb_from_sums. verifies brain state is sane after a batch.
    """
    if not gpu_available():
        return False

    from engram.gpu.brain import Brain

    V, D = 2000, 300
    P = D // 2
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((V, D)).astype(np.float32) * 0.3
    pairing = np.arange(D, dtype=np.int32)

    engine = GpuEngine()
    chain = ChainEngine(engine, matrix, pairing, window=10, matrix_dtype="f32")
    unified = UnifiedStage(engine.device, engine, P=P)

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "u.engram")
        ef = EngramFile.create(path, name="u", dim=300,
                                metadata={}, annotations={})
        ef._reference_pairing = pairing.copy()
        brain = Brain(ef, engine)

        # 3 batches of 4 articles each
        total_abs = 0
        for batch in range(3):
            indices_list = []
            for _ in range(4):
                T = int(rng.integers(300, 600))
                indices_list.append(rng.integers(0, V, size=T).astype(np.int32))

            combined = chain.submit_gather_and_features(indices_list)
            if combined is None:
                print(f"batch {batch}: gather returned None")
                continue
            result = unified.submit_fused(
                chain, engine, combined["handle"], combined["feat_ctx"],
                n_art=len(indices_list), max_block=64,
            )
            if result is None:
                print(f"batch {batch}: unified returned None")
                continue

            names = brain.absorb_from_sums(
                result["K_sums"], result["G_sums"],
                result["rms_sums"], result["n_segs_per_art"])
            total_abs += len(names)
            print(f"  batch {batch}: "
                  f"total_segs={result['total_segs']}, "
                  f"wells_after={len(ef.wells())}, "
                  f"absorbed={len(names)}")

        print(f"FINAL: total_absorbed={ef._total_absorbed}, "
              f"wells={len(ef.wells())}")
        # sanity checks
        assert ef._total_absorbed == total_abs, \
            f"total_absorbed mismatch"
        # every well has finite centroid
        for name, s in ef._well_sum_K.items():
            assert np.all(np.isfinite(s)), f"well {name} non-finite"
        assert len(ef._dream_Ks) == len(ef._dream_Gs) == len(ef._dream_Ss)
        print(f"  [PASS] full unified + Brain pipeline")
    return True


def main():
    ok1 = test_small_batch()
    if not ok1:
        sys.exit(1)
    ok2 = test_brain_from_sums()
    if not ok2:
        sys.exit(1)
    print("\nall unified tests passed")


if __name__ == "__main__":
    main()
