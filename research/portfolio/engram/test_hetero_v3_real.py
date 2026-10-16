"""
test_hetero_v3_real.py — real enwiki with GPU-resident matrix.

mirrors test_hetero_v2_real.py so we can compare rates apples-to-apples.
100 articles, 4 CPU workers, batch=32, 100ms timeout.
"""

from __future__ import annotations

import sys
import re
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "mass"))

import numpy as np
import mwparserfromhell

from engram_codec import Engram, EngramFile, derive_pairing
from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu.chain_engine import ChainEngine
from engram.hetero_v3 import ingest_hetero_v3
from engram.mass_embed import make_mass_idx_fn
from engram.mass.test_real import read_first_articles, ENWIKI


def build_vocab_and_matrix(articles, dim: int = 300):
    all_toks = []
    for _, wt in articles:
        pt = mwparserfromhell.parse(wt).strip_code().lower()
        all_toks.extend(w for w in re.findall(r"[a-z]+", pt) if len(w) > 1)
    vocab = sorted(set(all_toks))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    rng = np.random.default_rng(1234)
    matrix = (rng.standard_normal((len(vocab), dim)) * 0.3).astype(np.float32)
    return word_to_idx, matrix


def derive_pairing_from_sample(matrix, idx_fn, articles, window: int = 10):
    """run MASS on one real article to get a representative trajectory,
    then derive pairing from it. ensures chain + packet pairing agree."""
    for _, wt in articles:
        idxs = idx_fn(wt)
        if idxs.shape[0] < max(64, window):
            continue
        raw = matrix[idxs]
        T, D = raw.shape
        cs = np.empty((T + 1, D), dtype=np.float32); cs[0] = 0
        np.cumsum(raw, axis=0, out=cs[1:])
        smoothed = (cs[window:] - cs[:-window]) / np.float32(window)
        return derive_pairing(smoothed)
    # fallback — sequential pairing if every article was too short
    return np.arange(matrix.shape[1], dtype=np.int32)


def run() -> None:
    if not Path(ENWIKI).exists():
        print("enwiki missing"); return
    if not gpu_available():
        print("wgpu missing"); return

    print("reading 1000 articles from real enwiki...")
    arts = read_first_articles(ENWIKI, n_articles=1000)
    total_bytes = sum(len(wt) for _, wt in arts)
    print(f"  {len(arts)} articles, {total_bytes/1024:.0f} KB wikitext")

    print("building vocab...")
    t0 = time.time()
    word_to_idx, matrix = build_vocab_and_matrix(arts)
    print(f"  vocab={len(word_to_idx)} matrix={matrix.nbytes/1e6:.0f} MB "
          f"in {time.time()-t0:.1f}s")

    idx_fn = make_mass_idx_fn(word_to_idx)

    print("deriving pairing from sample article...")
    t0 = time.time()
    pairing = derive_pairing_from_sample(matrix, idx_fn, arts)
    print(f"  pairing in {time.time()-t0:.2f}s")

    engine = GpuEngine()
    print(f"engine: {engine.describe()}")

    print("building ChainEngine (f16 matrix)...")
    t0 = time.time()
    chain = ChainEngine(engine, matrix, pairing, window=10,
                        matrix_dtype="f16")
    print(f"  chain ready in {time.time()-t0:.2f}s "
          f"(uploaded {chain.matrix_gpu.nbytes/1e6:.0f} MB f16)")

    with tempfile.TemporaryDirectory() as td:
        brain = str(Path(td) / "_v3_real.engram")
        ef = EngramFile.create(brain, name="v3_real", dim=300,
                                metadata={}, annotations={})
        codec = Engram(dim=300)

        print("\nwarmup (4 articles)...")
        t0 = time.time()
        ingest_hetero_v3(
            ef=ef, codec=codec, articles=arts[:4],
            idx_fn=idx_fn, chain=chain,
            engine=engine,
            label="warmup",
            batch_size=4, batch_timeout_ms=100, n_cpu_workers=2,
            progress_every=999999, telemetry_interval_s=999.0,
        )
        print(f"  warmup: {time.time()-t0:.2f}s")

        print("\nmain (96 articles, 4 workers)...")
        t0 = time.time()
        n = ingest_hetero_v3(
            ef=ef, codec=codec, articles=arts[4:],
            idx_fn=idx_fn, chain=chain,
            engine=engine,
            label="main",
            batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
            progress_every=30, telemetry_interval_s=3.0,
        )
        dt = time.time() - t0
        print()
        print(f"done: {n}/{len(arts)-4} absorbed in {dt:.2f}s "
              f"= {n/dt:.1f} articles/s")
        print(f"wells: {len(ef.wells())}")


if __name__ == "__main__":
    run()
