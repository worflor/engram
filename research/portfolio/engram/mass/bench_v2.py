"""bench_v2.py — add inline tokenization to the ladder.

the "current production" path is:
  mwparserfromhell.parse(text).strip_code()  →  re.findall(r"[a-z]+", lower)

the "v2 fused" path is:
  mass_tokenize(text)  →  list of word strings

we measure both and compare the TOKEN SEQUENCES, not just byte output.
"""

from __future__ import annotations
import sys, time, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mwparserfromhell
from corpus import corpus
from v2_tokenize import mass_tokenize


def current_tokens(text: str):
    plaintext = mwparserfromhell.parse(text).strip_code()
    return [w for w in re.findall(r"[a-z]+", plaintext.lower()) if len(w) > 1]


def v2_tokens(text: str):
    return mass_tokenize(text)


def run():
    N = 200
    articles = corpus(N, seed=42, size_hint=5000)
    total_bytes = sum(len(a) for a in articles)
    print(f"corpus: {N} articles, {total_bytes/1024:.0f} KB")
    print()

    # warmup
    for a in articles[:5]:
        current_tokens(a)
        v2_tokens(a)

    t0 = time.perf_counter()
    cur_all = [current_tokens(a) for a in articles]
    t_cur = time.perf_counter() - t0

    t0 = time.perf_counter()
    v2_all = [v2_tokens(a) for a in articles]
    t_v2 = time.perf_counter() - t0

    cur_tokens = sum(len(ts) for ts in cur_all)
    v2_tokens_n = sum(len(ts) for ts in v2_all)

    print(f"{'stage':<30s} {'ms':>10s} {'MB/s':>8s} {'tokens':>10s}")
    print(f"{'-'*60}")
    print(f"{'current (mwp+re.findall)':<30s} {t_cur*1000:>10.1f} "
          f"{total_bytes/t_cur/1e6:>8.1f} {cur_tokens:>10}")
    print(f"{'v2 fused numba':<30s} {t_v2*1000:>10.1f} "
          f"{total_bytes/t_v2/1e6:>8.1f} {v2_tokens_n:>10}")
    print(f"speedup: {t_cur/t_v2:.1f}x")
    print()

    # semantic equivalence at token level
    print("token-level equivalence (sample 0):")
    a = set(cur_all[0]); b = set(v2_all[0])
    if a | b:
        j = len(a & b) / len(a | b)
        print(f"  jaccard:        {j:.2%}")
        print(f"  |current only|: {len(a - b)} tokens")
        print(f"  |v2 only|:      {len(b - a)} tokens")
        print(f"  current sample: {list(a - b)[:5]}")
        print(f"  v2 sample:      {list(b - a)[:5]}")


if __name__ == "__main__":
    run()
