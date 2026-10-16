"""test_gpu_absorb.py — verify Brain produces equivalent engram
state to the CPU absorb_flat path, then benchmark at scale.

three tests:
  1. equivalence: build two brains, drive one through absorb_flat and
     one through Brain with the same observations, compare state.
  2. continuation: Brain must handle growing brains across many
     batches, including new well creation.
  3. benchmark: preseeded 375-well brain, 200 batches of 48 articles,
     measure absorb-per-batch on both paths.
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engram_codec import Engram, EngramFile, WELL_ESTABLISHED
from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu.brain import Brain


def preseed_brain(path: str, n_wells: int, P: int, seed: int = 42):
    """synthesize n_wells established wells with spread-out centroids."""
    ef = EngramFile.create(path, name="bench", dim=P * 2,
                           metadata={}, annotations={})
    rng = np.random.default_rng(seed)
    for i in range(n_wells):
        name = f"well_{i}"
        real = rng.standard_normal(P) * 0.6
        imag = rng.standard_normal(P) * 0.6
        count = WELL_ESTABLISHED + 5 + int(rng.integers(0, 20))
        K_sum = (real + 1j * imag).astype(np.complex128) * count
        ef._well_sum_K[name] = K_sum
        ef._well_count[name] = count
    ef._next_well_id = n_wells
    ef._invalidate_caches()
    return ef


def absorb_stacks(brain, K, G, R, bdy):
    """drive a per-segment stack batch through Brain.absorb_from_sums.

    Brain's only absorb API consumes per-article sums (what the unified
    GPU pipeline's per_article_reduce kernel produces). this harness
    generates per-segment K/G/rms stacks with boundaries as a
    convenience, so we reduce them here before handing to Brain. the
    reduction matches what per_article_reduce.wgsl computes on GPU.
    """
    bdy = np.asarray(bdy, dtype=np.int64)
    starts = bdy[:-1]
    n_segs = np.diff(bdy).astype(np.int64)
    # np.add.reduceat would choke on empty-tail articles where start == len;
    # filter to non-empty, then scatter back.
    n_art = len(starts)
    nonempty = n_segs > 0
    K_sums = np.zeros((n_art, K.shape[1]), dtype=np.complex64)
    G_sums = np.zeros((n_art, G.shape[1]), dtype=np.complex64)
    R_sums = np.zeros((n_art, R.shape[1]), dtype=np.float32)
    if int(nonempty.sum()) > 0:
        starts_ne = starts[nonempty].astype(np.int64)
        # np.add.reduceat with sorted starts gives per-segment-range sums
        K_sums[nonempty] = np.add.reduceat(K, starts_ne, axis=0).astype(np.complex64)
        G_sums[nonempty] = np.add.reduceat(G, starts_ne, axis=0).astype(np.complex64)
        R_sums[nonempty] = np.add.reduceat(R, starts_ne, axis=0).astype(np.float32)
    return brain.absorb_from_sums(K_sums, G_sums, R_sums, n_segs)


def gen_batch(n_art: int, segs_per: int, P: int, rng):
    """yield a batch: K_stack, G_stack, rms_stack, boundaries."""
    K = (rng.standard_normal((n_art * segs_per, P))
         + 1j * rng.standard_normal((n_art * segs_per, P))
         ).astype(np.complex64) * 0.3
    G = (rng.standard_normal((n_art * segs_per, P))
         + 1j * rng.standard_normal((n_art * segs_per, P))
         ).astype(np.complex64) * 0.3
    R = rng.standard_normal((n_art * segs_per, P)).astype(np.float32) * 0.1
    boundaries = np.arange(0, n_art * segs_per + 1, segs_per, dtype=np.int32)
    return K, G, R, boundaries


def test_equivalence():
    """two sub-tests:
      1a. TIGHT observations (scale=0.3) — tests routing correctness
          on a well-dominated case. counts must match exactly because
          each article clearly belongs to one preseeded well and fp
          precision can't flip the argmin.
      1b. WIDE observations (scale=2.0) — forces new-well spawning on
          BOTH paths. asserts the well SET matches exactly, which is
          what the in-batch rewire pass guarantees. counts can drift
          due to fp-induced argmin flips among close wells, bounded.
    """
    print("=" * 60)
    print("TEST 1a: equivalence — tight observations (argmin unambiguous)")
    print("=" * 60)
    P = 150

    def _drive(scale, n_batches=12, n_art=8, segs=10, spawn_expected=False):
        with tempfile.TemporaryDirectory() as td:
            p1 = str(Path(td) / "cpu.engram")
            p2 = str(Path(td) / "gpu.engram")
            ef_cpu = preseed_brain(p1, 20, P, seed=42)
            ef_gpu = preseed_brain(p2, 20, P, seed=42)
            engine = GpuEngine()
            abs_gpu = Brain(ef_gpu, engine)

            rng = np.random.default_rng(123)
            for _ in range(n_batches):
                K = (rng.standard_normal((n_art * segs, P))
                     + 1j * rng.standard_normal((n_art * segs, P))
                     ).astype(np.complex64) * scale
                G = (rng.standard_normal((n_art * segs, P))
                     + 1j * rng.standard_normal((n_art * segs, P))
                     ).astype(np.complex64) * scale
                R = rng.standard_normal((n_art * segs, P)).astype(np.float32) * 0.1
                bdy = np.arange(0, n_art * segs + 1, segs, dtype=np.int32)

                for a in range(len(bdy) - 1):
                    s, e = int(bdy[a]), int(bdy[a + 1])
                    ef_cpu.absorb_flat(K[s:e], G[s:e], R[s:e])
                absorb_stacks(abs_gpu, K, G, R, bdy)

            return ef_cpu, ef_gpu

    # ── 1a. tight observations → exact counts expected ────────────
    ef_cpu, ef_gpu = _drive(scale=0.3)
    cpu_names = sorted(ef_cpu.wells().keys())
    gpu_names = sorted(ef_gpu.wells().keys())
    assert ef_cpu._total_absorbed == ef_gpu._total_absorbed
    assert cpu_names == gpu_names, \
        f"tight: well set drifted. cpu-only={[n for n in cpu_names if n not in gpu_names][:5]}"
    max_count_diff = max(
        abs(ef_cpu._well_count[n] - ef_gpu._well_count[n])
        for n in cpu_names)
    print(f"  wells: {len(cpu_names)}  max count diff: {max_count_diff}")
    assert max_count_diff == 0, \
        f"tight: counts diverged: {max_count_diff}"
    print("  [OK — bit-exact routing at tight spread]")
    print()

    # ── 1b. empty brain + wide obs → exercises bootstrap + rewire ─
    print("=" * 60)
    print("TEST 1b: equivalence — empty-brain bootstrap + new-well rewire")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        p1 = str(Path(td) / "cpu.engram")
        p2 = str(Path(td) / "gpu.engram")
        ef_cpu = EngramFile.create(p1, name="bench", dim=P * 2,
                                    metadata={}, annotations={})
        ef_gpu = EngramFile.create(p2, name="bench", dim=P * 2,
                                    metadata={}, annotations={})
        engine = GpuEngine()
        abs_gpu = Brain(ef_gpu, engine)
        rng = np.random.default_rng(777)

        for _ in range(8):
            K = (rng.standard_normal((40, P))
                 + 1j * rng.standard_normal((40, P))
                 ).astype(np.complex64) * 1.5
            G = (rng.standard_normal((40, P))
                 + 1j * rng.standard_normal((40, P))
                 ).astype(np.complex64) * 1.5
            R = rng.standard_normal((40, P)).astype(np.float32) * 0.1
            bdy = np.arange(0, 41, 10, dtype=np.int32)

            for a in range(len(bdy) - 1):
                s, e = int(bdy[a]), int(bdy[a + 1])
                ef_cpu.absorb_flat(K[s:e], G[s:e], R[s:e])
            absorb_stacks(abs_gpu, K, G, R, bdy)

    cpu_names = sorted(ef_cpu.wells().keys())
    gpu_names = sorted(ef_gpu.wells().keys())
    cpu_only = [n for n in cpu_names if n not in gpu_names]
    gpu_only = [n for n in gpu_names if n not in cpu_names]
    print(f"  wells: cpu={len(cpu_names)} gpu={len(gpu_names)}")
    print(f"  cpu-only: {cpu_only[:5]}  gpu-only: {gpu_only[:5]}")
    assert ef_cpu._total_absorbed == ef_gpu._total_absorbed
    # well SET must match exactly because bootstrap transitions to
    # GPU path mid-batch, and rewire handles in-batch new-well peers.
    assert cpu_names == gpu_names, \
        f"bootstrap+rewire: well SET drifted. cpu_only={cpu_only[:10]} gpu_only={gpu_only[:10]}"
    print("  [OK — set exact across bootstrap + rewire + GPU path]")
    print()


def test_continuation():
    print("=" * 60)
    print("TEST 2: growing brain across many batches (new wells)")
    print("=" * 60)
    P = 150
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "grow.engram")
        ef = EngramFile.create(path, name="grow", dim=P * 2,
                               metadata={}, annotations={})
        engine = GpuEngine()
        abs_gpu = Brain(ef, engine)

        # drive many batches with wide random observations — should
        # spawn new wells frequently
        for b in range(30):
            K, G, R, bdy = gen_batch(4, 8, P, rng)
            K = K * 3.0  # larger spread → escape velocity more often
            names = absorb_stacks(abs_gpu, K, G, R, bdy)
            assert all(n != "unknown" for n in names)
        print(f"  after 30 batches: wells={len(ef.wells())}  "
              f"total_absorbed={ef._total_absorbed}")
        assert len(ef.wells()) >= 2, "should have spawned at least 2 wells"
        assert ef._total_absorbed == 30 * 4
        print("  [OK]")
    print()


def test_benchmark():
    print("=" * 60)
    print("TEST 3: benchmark vs CPU absorb_flat at 375 wells")
    print("=" * 60)
    P = 150
    n_wells = 375
    rng = np.random.default_rng(999)
    n_batches = 50
    n_art = 48
    segs_per = 50

    # pregenerate ALL batches up front so the timed loop measures
    # only absorb cost, not rng / allocation overhead.
    batches = [gen_batch(n_art, segs_per, P, rng) for _ in range(n_batches + 3)]

    # CPU baseline
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "cpu.engram")
        ef_cpu = preseed_brain(p, n_wells, P, seed=42)
        # warmup
        for K, G, R, bdy in batches[:3]:
            for a in range(len(bdy) - 1):
                s, e = int(bdy[a]), int(bdy[a + 1])
                ef_cpu.absorb_flat(K[s:e], G[s:e], R[s:e])

        t0 = time.perf_counter()
        for K, G, R, bdy in batches[3:]:
            for a in range(len(bdy) - 1):
                s, e = int(bdy[a]), int(bdy[a + 1])
                ef_cpu.absorb_flat(K[s:e], G[s:e], R[s:e])
        cpu_dt = time.perf_counter() - t0
        cpu_per_batch_ms = cpu_dt / n_batches * 1000.0
        print(f"  CPU absorb_flat : {cpu_per_batch_ms:.1f} ms/batch "
              f"({n_batches * n_art / cpu_dt:.1f} art/s)")

    # GPU
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "gpu.engram")
        ef_gpu = preseed_brain(p, n_wells, P, seed=42)
        engine = GpuEngine()
        abs_gpu = Brain(ef_gpu, engine)
        # warmup
        for K, G, R, bdy in batches[:3]:
            absorb_stacks(abs_gpu, K, G, R, bdy)

        # accumulated per-stage timings across the whole run
        stage_totals = {
            "upload": 0.0, "gap": 0.0, "compute": 0.0,
            "readback": 0.0, "apply": 0.0,
        }

        t0 = time.perf_counter()
        for K, G, R, bdy in batches[3:]:
            absorb_stacks(abs_gpu, K, G, R, bdy)
            for k in stage_totals:
                stage_totals[k] += abs_gpu.last_stage_ms[k]
        gpu_dt = time.perf_counter() - t0
        gpu_per_batch_ms = gpu_dt / n_batches * 1000.0
        print(f"  Brain.absorb_from_sums: {gpu_per_batch_ms:.1f} ms/batch "
              f"({n_batches * n_art / gpu_dt:.1f} art/s)")
        print(f"  stage breakdown (avg ms/batch):")
        for k, tot in sorted(stage_totals.items(),
                              key=lambda x: -x[1]):
            avg = tot / n_batches
            pct = avg / gpu_per_batch_ms * 100.0
            print(f"    {k:10s} {avg:6.2f} ms  ({pct:5.1f}%)")

    speedup = cpu_per_batch_ms / gpu_per_batch_ms
    print(f"  speedup: {speedup:.1f}x")
    print()


def main():
    if not gpu_available():
        print("wgpu missing")
        return
    test_equivalence()
    test_continuation()
    test_benchmark()
    print("all tests passed")


if __name__ == "__main__":
    main()
