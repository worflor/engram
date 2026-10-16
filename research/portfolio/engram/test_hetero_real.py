"""
test_hetero_real.py — small real-enwiki integration test for the
MASS + hetero pipeline.

HARD CAPS (defensive):
  - max 40 articles (small; can't eat memory)
  - 60s wall clock ceiling (enforced by subprocess timeout from caller)
  - batch size 16 (keeps GPU work chunks small)
  - 4 CPU workers (matches typical core count)

this is the "two rats, tiny tunnel" test. if THIS hangs, we back off
before touching real production.
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

from engram_codec import Engram, EngramFile
from engram.gpu.engine import GpuEngine, gpu_available
from engram.hetero_ingest import ingest_hetero
from engram.mass_embed import make_mass_embed_fn
from engram.mass.test_real import read_first_articles, ENWIKI


def build_vocab_and_matrix(articles, dim=300):
    """build a fake glove-scale matrix from observed tokens only.
    fast and keeps memory bounded for this test."""
    all_toks = []
    for _, wt in articles:
        pt = mwparserfromhell.parse(wt).strip_code().lower()
        all_toks.extend(w for w in re.findall(r"[a-z]+", pt) if len(w) > 1)
    vocab = sorted(set(all_toks))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    rng = np.random.default_rng(1234)
    matrix = (rng.standard_normal((len(vocab), dim)) * 0.3).astype(np.float32)
    return word_to_idx, matrix


def run():
    if not Path(ENWIKI).exists():
        print("enwiki bz2 missing"); return
    if not gpu_available():
        print("wgpu not available"); return

    print("=== HETERO + MASS + REAL ENWIKI (small) ===")
    print()

    # hard cap article count
    articles = read_first_articles(ENWIKI, n_articles=40)
    total_bytes = sum(len(wt) for _, wt in articles)
    print(f"articles:   {len(articles)}  ({total_bytes/1024:.0f} KB wikitext)")

    t0 = time.time()
    word_to_idx, matrix = build_vocab_and_matrix(articles)
    print(f"vocab:      {len(word_to_idx)} words in {time.time()-t0:.1f}s")

    embed = make_mass_embed_fn(matrix, word_to_idx, window=10)
    engine = GpuEngine()
    print(f"engine:     {engine.describe()}")
    print()

    with tempfile.TemporaryDirectory() as td:
        brain_path = str(Path(td) / "_hetero_real.engram")
        ef = EngramFile.create(brain_path, name="hetero_real", dim=300,
                                metadata={}, annotations={})
        codec = Engram(dim=300)

        # warmup: 4 articles with tiny batch (numba JIT, MASS trie, fit_pair)
        print("warmup (4 articles, batch=4, 2 workers)...")
        t0 = time.time()
        ingest_hetero(
            ef=ef, codec=codec, articles=articles[:4],
            embed_fn=embed, label="warmup",
            engine=engine, batch_size=4, n_workers=2,
            checkpoint_every=9999, progress_every=999999,
        )
        print(f"  warmup: {time.time() - t0:.1f}s")
        print()

        # main: 36 articles
        print("main (36 articles, batch=16, 4 workers)...")
        t0 = time.time()
        n_absorbed = ingest_hetero(
            ef=ef, codec=codec, articles=articles[4:],
            embed_fn=embed, label="main",
            engine=engine, batch_size=16, n_workers=4,
            checkpoint_every=9999, progress_every=10,
        )
        dt = time.time() - t0
        print()
        print(f"done: {n_absorbed} absorbed in {dt:.2f}s "
              f"= {n_absorbed/dt:.1f} articles/s")
        print(f"wells: {len(ef.wells())}")

        # save/load
        ef.save(brain_path)
        ef2 = EngramFile.load(brain_path)
        assert ef2._total_absorbed == ef._total_absorbed
        print(f"save/load round-trip: OK")


if __name__ == "__main__":
    run()
