"""
bench_v3_parallel.py — does mass_fused release the GIL under numba?

if yes, throwing 4-8 threads at it multiplies throughput linearly
and this becomes the answer for bz2 phase ingest regardless of
archive format. threading parallelism is basically free speedup
once GIL is out of the way.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from corpus import corpus
from v3_fused import FusedIngest
from bench_v3_realistic import build_vocab_at_scale


def run():
    N = 800
    articles = corpus(N, seed=42, size_hint=5000)
    total_bytes = sum(len(a) for a in articles)
    print(f"corpus: {N} articles, {total_bytes/1024:.0f} KB")
    print()

    print("building 400k-word vocab...", end=" ", flush=True)
    t0 = time.perf_counter()
    vocab = build_vocab_at_scale(articles, target_size=400_000)
    fused = FusedIngest(vocab)
    print(f"{time.perf_counter() - t0:.1f}s")
    print()

    # warmup
    for a in articles[:5]:
        fused.extract(a)

    for n_workers in (1, 2, 4, 8, 16):
        # fresh warmup between configurations
        for a in articles[:2]:
            fused.extract(a)

        if n_workers == 1:
            t0 = time.perf_counter()
            outs = [fused.extract(a) for a in articles]
            dt = time.perf_counter() - t0
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                t0 = time.perf_counter()
                outs = list(pool.map(fused.extract, articles, chunksize=8))
                dt = time.perf_counter() - t0
        mbps = total_bytes / dt / 1e6
        ratio = (total_bytes / 1e6) / (dt)
        print(f"  n_workers={n_workers:>2}  {dt*1000:>7.1f} ms   "
              f"{mbps:>6.1f} MB/s   "
              f"{N/dt:>7.0f} articles/s")


if __name__ == "__main__":
    run()
