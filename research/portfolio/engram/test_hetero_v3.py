"""
test_hetero_v3.py — bring-up for the 4-stage pipeline with ChainEngine.

mirrors test_hetero_v2.py but uses idx_fn + ChainEngine instead of
full embed_fn. the CPU workers emit only indices; the GloVe matrix
lives on the GPU.

hard caps: tiny synthetic corpus, V=50k D=300, batch=32, 4 workers,
timeouts on every queue.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engram_codec import Engram, EngramFile, derive_pairing
from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu.chain_engine import ChainEngine
from engram.hetero_v3 import ingest_hetero_v3


def make_fake_idx(vocab: int = 50000):
    """returns a closure that tokenises synthetic 'tokN' words and returns
    only int32 indices — NO matrix gather happens on the CPU side."""
    import re
    word_to_idx = {f"tok{i}": i for i in range(vocab)}

    def idx(text: str) -> np.ndarray:
        toks = re.findall(r"tok\d+", text)
        if not toks:
            return np.empty(0, dtype=np.int32)
        return np.fromiter(
            (word_to_idx[t] for t in toks if t in word_to_idx),
            dtype=np.int32, count=-1,
        )

    return idx


def synth_articles(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        n_words = int(rng.integers(200, 800))
        text = " ".join(f"tok{int(x)}" for x in
                        rng.integers(0, 50000, size=n_words))
        out.append((f"article_{i:04d}", text))
    return out


def run() -> None:
    if not gpu_available():
        print("wgpu not available"); return

    engine = GpuEngine()
    print(f"adapter: {engine.describe()}")

    rng = np.random.default_rng(1234)
    V, D = 50000, 300
    matrix = (rng.standard_normal((V, D)) * 0.3).astype(np.float32)

    # derive pairing from a sample trajectory so downstream packets use
    # the same permutation that ChainEngine applies on GPU.
    print("deriving pairing from sample...")
    sample_idxs = rng.integers(0, V, size=400, dtype=np.int32)
    sample_raw = matrix[sample_idxs]
    T = sample_raw.shape[0]; W = 10
    cs = np.empty((T + 1, D), dtype=np.float32); cs[0] = 0
    np.cumsum(sample_raw, axis=0, out=cs[1:])
    sample_smooth = (cs[W:] - cs[:-W]) / np.float32(W)
    pairing = derive_pairing(sample_smooth)
    print(f"  pairing shape={pairing.shape}")

    print("building ChainEngine...")
    t0 = time.time()
    chain = ChainEngine(engine, matrix, pairing, window=10)
    print(f"  chain ready in {time.time()-t0:.2f}s "
          f"(uploaded {matrix.nbytes/1e6:.0f} MB)")

    idx_fn = make_fake_idx()

    with tempfile.TemporaryDirectory() as td:
        brain = str(Path(td) / "_v3.engram")
        ef = EngramFile.create(brain, name="v3_test", dim=D,
                                metadata={}, annotations={})
        codec = Engram(dim=D)

        # warmup (also exercises first batch in each kernel)
        warm = synth_articles(4, seed=0)
        t0 = time.time()
        ingest_hetero_v3(
            ef=ef, codec=codec, articles=warm,
            idx_fn=idx_fn, chain=chain,
            label="warmup",
            engine=engine,
            batch_size=4, batch_timeout_ms=100,
            n_cpu_workers=2, progress_every=999999,
            telemetry_interval_s=999.0,
        )
        print(f"warmup: {time.time() - t0:.2f}s\n")

        arts = synth_articles(200, seed=1)
        print(f"running {len(arts)} articles through v3 pipeline...")
        t0 = time.time()
        n = ingest_hetero_v3(
            ef=ef, codec=codec, articles=arts,
            idx_fn=idx_fn, chain=chain,
            label="main",
            engine=engine,
            batch_size=32, batch_timeout_ms=100,
            n_cpu_workers=4, progress_every=50,
            telemetry_interval_s=3.0,
        )
        dt = time.time() - t0
        print()
        print(f"absorbed {n} in {dt:.2f}s = {n/dt:.1f} articles/s")
        print(f"wells: {len(ef.wells())}")

        ef.save(brain)
        ef2 = EngramFile.load(brain)
        assert ef2._total_absorbed == ef._total_absorbed
        print("save/load round-trip: OK")


if __name__ == "__main__":
    run()
