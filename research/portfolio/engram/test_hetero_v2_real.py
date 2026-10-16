"""
test_hetero_v2_real.py — 4-stage pipeline on real enwiki, small.

100 articles. 4 CPU workers. batch_size 32, batch_timeout 100ms.
prints live telemetry every 3 seconds so we can see where time goes.
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
from engram.hetero_v2 import ingest_hetero_v2
from engram.mass_embed import make_mass_embed_fn
from engram.mass.test_real import read_first_articles, ENWIKI


def build_vocab_and_matrix(articles, dim=300):
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
        print("enwiki missing"); return
    if not gpu_available():
        print("wgpu missing"); return

    print("reading 100 articles from real enwiki...")
    arts = read_first_articles(ENWIKI, n_articles=100)
    total_bytes = sum(len(wt) for _, wt in arts)
    print(f"  {len(arts)} articles, {total_bytes/1024:.0f} KB wikitext")

    print("building vocab + MASS embed...")
    t0 = time.time()
    word_to_idx, matrix = build_vocab_and_matrix(arts)
    embed = make_mass_embed_fn(matrix, word_to_idx, window=10)
    print(f"  vocab={len(word_to_idx)} in {time.time()-t0:.1f}s")

    engine = GpuEngine()
    with tempfile.TemporaryDirectory() as td:
        brain = str(Path(td) / "_v2_real.engram")
        ef = EngramFile.create(brain, name="v2_real", dim=300,
                                metadata={}, annotations={})
        codec = Engram(dim=300)

        # warmup
        print("\nwarmup (4 articles)...")
        t0 = time.time()
        ingest_hetero_v2(
            ef=ef, codec=codec, articles=arts[:4], embed_fn=embed,
            label="warmup", engine=engine,
            batch_size=4, batch_timeout_ms=100, n_cpu_workers=2,
            progress_every=999999, telemetry_interval_s=999.0,
        )
        print(f"  warmup: {time.time()-t0:.2f}s")

        # real run
        print("\nmain (96 articles, 4 workers)...")
        t0 = time.time()
        n = ingest_hetero_v2(
            ef=ef, codec=codec, articles=arts[4:], embed_fn=embed,
            label="main", engine=engine,
            batch_size=32, batch_timeout_ms=100, n_cpu_workers=4,
            progress_every=30, telemetry_interval_s=3.0,
        )
        dt = time.time() - t0
        print()
        print(f"done: {n}/{len(arts)-4} absorbed in {dt:.2f}s "
              f"= {n/dt:.1f} articles/s")
        print(f"wells: {len(ef.wells())}")


if __name__ == "__main__":
    run()
