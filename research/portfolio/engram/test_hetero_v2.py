"""
test_hetero_v2.py — bring-up for the 4-stage pipeline.

safety-first: small synthetic corpus, caps everything, times out
via shell.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engram_codec import Engram, EngramFile
from engram.gpu.engine import GpuEngine, gpu_available
from engram.hetero_v2 import ingest_hetero_v2


def make_fake_embed(dim=300, vocab=50000):
    rng = np.random.default_rng(1234)
    table = (rng.standard_normal((vocab, dim)) * 0.3).astype(np.float32)
    word_to_idx = {f"tok{i}": i for i in range(vocab)}
    import re
    def embed(text: str) -> np.ndarray:
        tokens = re.findall(r"tok\d+", text)
        if len(tokens) < 10:
            return np.empty((0, dim), dtype=np.float32)
        idxs = [word_to_idx[t] for t in tokens if t in word_to_idx]
        if len(idxs) < 10:
            return np.empty((0, dim), dtype=np.float32)
        raw = table[np.asarray(idxs, dtype=np.int32)]
        T, D = raw.shape
        W = 10
        cs = np.empty((T + 1, D), dtype=np.float32); cs[0] = 0
        np.cumsum(raw, axis=0, out=cs[1:])
        return (cs[W:] - cs[:-W]) / np.float32(W)
    return embed


def synth_articles(n, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        n_words = int(rng.integers(200, 800))
        text = " ".join(f"tok{int(x)}" for x in
                        rng.integers(0, 50000, size=n_words))
        out.append((f"article_{i:04d}", text))
    return out


def run():
    if not gpu_available():
        print("wgpu not available"); return

    engine = GpuEngine()
    print(f"adapter: {engine.describe()}")
    embed = make_fake_embed()

    with tempfile.TemporaryDirectory() as td:
        brain = str(Path(td) / "_v2.engram")
        ef = EngramFile.create(brain, name="v2_test", dim=300,
                                metadata={}, annotations={})
        codec = Engram(dim=300)

        # warmup so JIT doesn't dominate the real timing
        warm = synth_articles(4, seed=0)
        t0 = time.time()
        ingest_hetero_v2(
            ef=ef, codec=codec, articles=warm, embed_fn=embed,
            label="warmup",
            engine=engine, batch_size=4, batch_timeout_ms=100,
            n_cpu_workers=2, progress_every=999999,
            telemetry_interval_s=999.0,  # suppress telemetry during warmup
        )
        print(f"warmup: {time.time() - t0:.2f}s\n")

        # main
        arts = synth_articles(200, seed=1)
        print(f"running {len(arts)} articles through 4-stage pipeline...")
        t0 = time.time()
        n = ingest_hetero_v2(
            ef=ef, codec=codec, articles=arts, embed_fn=embed,
            label="main",
            engine=engine, batch_size=32, batch_timeout_ms=100,
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
