"""
bench_ingest.py

measure end-to-end ingest throughput with and without the threaded
CPU-prep pipeline. synthetic articles, a synthetic embed function
simulating GloVe's dict-lookup cost.

this is the only bench that measures what the user actually cares
about in production: articles/second, wall clock.
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import numpy as np

from engram_codec import Engram, EngramFile
from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu_ingest import BatchedGpuIngest, ingest_gpu


def synth_article(seed: int) -> tuple[str, str]:
    rng = np.random.default_rng(seed)
    n_words = int(rng.integers(400, 1000))
    text = " ".join(f"tok{int(x)}" for x in rng.integers(0, 20000, size=n_words))
    return f"article_{seed:05d}", text


def make_fake_embed(dim: int = 300, vocab_size: int = 50000):
    """simulates text_to_trajectory with a matrix-backed dict."""
    rng = np.random.default_rng(1234)
    table = rng.standard_normal((vocab_size, dim)).astype(np.float32) * 0.3
    word_to_idx = {f"tok{i}": i for i in range(vocab_size)}

    import re
    def embed(text: str) -> np.ndarray:
        tokens = re.findall(r"tok\d+", text)
        if len(tokens) < 4:
            return np.empty((0, dim), dtype=np.float32)
        idxs = [word_to_idx[t] for t in tokens if t in word_to_idx]
        vecs = table[idxs]
        window = 10
        if vecs.shape[0] < window:
            return vecs
        T, D = vecs.shape
        cumsum = np.vstack([np.zeros((1, D), dtype=np.float32), np.cumsum(vecs, axis=0)])
        out = np.empty((T - window + 1, D), dtype=np.float32)
        for i in range(T - window + 1):
            out[i] = (cumsum[i + window] - cumsum[i]) / window
        return out
    return embed


def run():
    if not gpu_available():
        print("wgpu not available")
        return

    engine = GpuEngine()
    embed = make_fake_embed()
    print(f"adapter: {engine.describe()}")
    print()

    N = 800
    articles = [synth_article(i) for i in range(N)]
    codec = Engram(dim=300)

    for n_workers in (1, 2, 4, 8):
        with tempfile.TemporaryDirectory() as td:
            brain_path = str(Path(td) / "_bench.engram")
            ef = EngramFile.create(brain_path, name="bench", dim=300,
                                    metadata={}, annotations={})

            t0 = time.perf_counter()
            ingest_gpu(
                ef=ef, codec=codec, articles=iter(articles),
                embed_fn=embed, label=f"nw={n_workers}",
                engine=engine, batch_size=128,
                checkpoint_every=99999,
                checkpoint_path=None,
                n_workers=n_workers,
            )
            dt = time.perf_counter() - t0
            rate = ef._total_absorbed / dt
            print(f"  n_workers={n_workers}  ->{ef._total_absorbed} absorbed in {dt:.2f}s = {rate:.0f}/s")

    # synchronous path for comparison
    print()
    print("synchronous (queue-per-article) path:")
    with tempfile.TemporaryDirectory() as td:
        brain_path = str(Path(td) / "_bench.engram")
        ef = EngramFile.create(brain_path, name="bench", dim=300,
                                metadata={}, annotations={})
        accum = BatchedGpuIngest(ef=ef, codec=codec, engine=engine,
                                  batch_size=128, progress_every=999999)
        t0 = time.perf_counter()
        for title, text in articles:
            accum.queue(title, text, embed)
        accum.flush()
        dt = time.perf_counter() - t0
        rate = ef._total_absorbed / dt
        print(f"  sync        ->{ef._total_absorbed} absorbed in {dt:.2f}s = {rate:.0f}/s")


if __name__ == "__main__":
    run()
