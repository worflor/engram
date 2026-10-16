"""
bench_v3_realistic.py — measure v3 with a GloVe-scale vocab (400k words).

we don't have 400k real words lying around, so we synthesize a vocab
of that size that includes the corpus's actual tokens (so lookups
hit) plus padding words to stress the hash table to production scale.

this is the true test of whether the fused path's throughput
survives at GloVe scale.
"""

from __future__ import annotations

import sys
import time
import re
import string
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mwparserfromhell

from corpus import corpus
from v3_fused import FusedIngest


def build_vocab_at_scale(articles, target_size=400_000):
    """collect real tokens + pad with synthetic words to reach target."""
    toks = set()
    for a in articles:
        plaintext = mwparserfromhell.parse(a).strip_code().lower()
        toks.update(w for w in re.findall(r"[a-z]+", plaintext) if len(w) > 1)
    base = sorted(toks)
    # pad with deterministic pseudo-words to reach target_size
    needed = target_size - len(base)
    if needed > 0:
        letters = string.ascii_lowercase
        padded = []
        for i in range(needed):
            # 5-char pseudoword
            w = ""
            x = i
            for _ in range(5):
                w += letters[x % 26]
                x //= 26
            w += "zzz"  # suffix to avoid colliding with real words
            padded.append(w)
        vocab = base + padded
    else:
        vocab = base[:target_size]
    return {w: i for i, w in enumerate(vocab)}


def run():
    N = 200
    articles = corpus(N, seed=42, size_hint=5000)
    total_bytes = sum(len(a) for a in articles)
    print(f"corpus: {N} articles, {total_bytes/1024:.0f} KB")
    print()

    for vocab_size in (1_000, 10_000, 100_000, 400_000):
        print(f"building vocab of size {vocab_size:,}...", end=" ", flush=True)
        t0 = time.perf_counter()
        vocab = build_vocab_at_scale(articles, target_size=vocab_size)
        print(f"{time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        fused = FusedIngest(vocab)
        build_time = time.perf_counter() - t0
        print(f"  table build: {build_time*1000:.0f} ms   "
              f"size: vocab_buf={fused.vocab_buf.nbytes/1024:.0f}KB  "
              f"table={fused.table.nbytes/1024/1024:.1f}MB")

        # warmup
        for a in articles[:5]:
            fused.extract(a)

        t0 = time.perf_counter()
        outs = [fused.extract(a) for a in articles]
        dt = time.perf_counter() - t0
        total_idxs = sum(len(o) for o in outs)
        print(f"  extract:     {dt*1000:.1f} ms   "
              f"{total_bytes/dt/1e6:.1f} MB/s   "
              f"{total_idxs} indices")
        print()


if __name__ == "__main__":
    run()
