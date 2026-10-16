"""
test_hetero_ingest.py — safe bring-up test for the heterogeneous
pipeline. small synthetic corpus first. every stage has explicit
timeouts so a hang surfaces fast rather than locking the machine.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import numpy as np

from engram_codec import Engram, EngramFile
from engram.gpu.engine import GpuEngine, gpu_available
from engram.hetero_ingest import ingest_hetero


def make_fake_embed(dim=300, vocab_size=50000):
    rng = np.random.default_rng(1234)
    table = (rng.standard_normal((vocab_size, dim)) * 0.3).astype(np.float32)
    word_to_idx = {f"tok{i}": i for i in range(vocab_size)}

    import re
    def embed(text: str) -> np.ndarray:
        tokens = re.findall(r"tok\d+", text)
        if len(tokens) < 10:
            return np.empty((0, dim), dtype=np.float32)
        idxs = [word_to_idx[t] for t in tokens if t in word_to_idx]
        if len(idxs) < 10:
            return np.empty((0, dim), dtype=np.float32)
        raw = table[np.asarray(idxs, dtype=np.int32)]
        W = 10
        T, D = raw.shape
        cs = np.empty((T + 1, D), dtype=np.float32)
        cs[0] = 0
        np.cumsum(raw, axis=0, out=cs[1:])
        return (cs[W:] - cs[:-W]) / np.float32(W)
    return embed


def synth_articles(n, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        n_words = int(rng.integers(200, 800))
        text = " ".join(f"tok{int(x)}" for x in rng.integers(0, 50000, size=n_words))
        out.append((f"article_{i:04d}", text))
    return out


def run(n_articles=50, n_workers=4):
    if not gpu_available():
        print("wgpu not available"); return

    print(f"=== SAFE BRING-UP: {n_articles} synthetic articles, "
          f"{n_workers} CPU workers ===")
    print()

    engine = GpuEngine()
    print(f"adapter: {engine.describe()}")
    print()

    embed = make_fake_embed()
    articles = synth_articles(n_articles)

    with tempfile.TemporaryDirectory() as td:
        brain_path = str(Path(td) / "_hetero_test.engram")
        ef = EngramFile.create(brain_path, name="hetero_test", dim=300,
                                metadata={}, annotations={})
        codec = Engram(dim=300)

        # warmup JIT — first dispatch is slow
        warm_articles = articles[:4]
        t0 = time.perf_counter()
        ingest_hetero(
            ef=ef, codec=codec, articles=warm_articles,
            embed_fn=embed, label="warmup",
            engine=engine, batch_size=4, n_workers=2,
            checkpoint_every=9999, progress_every=999999,
        )
        warm_t = time.perf_counter() - t0
        print(f"warmup {len(warm_articles)} articles in {warm_t:.2f}s")
        print()

        # main run
        t0 = time.perf_counter()
        n = ingest_hetero(
            ef=ef, codec=codec, articles=articles[4:],
            embed_fn=embed, label="main",
            engine=engine,
            batch_size=32, n_workers=n_workers,
            checkpoint_every=9999, progress_every=50,
        )
        dt = time.perf_counter() - t0
        print()
        print(f"main run: {n}/{n_articles-4} absorbed in {dt:.2f}s "
              f"= {n/dt:.0f} articles/s")

        # save/load sanity
        ef.save(brain_path)
        ef2 = EngramFile.load(brain_path)
        assert ef2._total_absorbed == ef._total_absorbed
        print(f"save/load round-trip: OK  (wells={len(ef.wells())})")


if __name__ == "__main__":
    run(n_articles=50, n_workers=4)
