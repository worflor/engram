"""
bench_v3.py — full bytes -> vocab indices fused benchmark.

compares three paths that all produce an int32[] of GloVe indices:
  A. current production: mwparserfromhell + regex + dict lookup (via fast_embed style)
  B. v2 fused MASS (tokenize only) + python dict lookup
  C. v3 fully fused (MASS + hash-table lookup, single numba call)
"""

from __future__ import annotations

import sys
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mwparserfromhell

from corpus import corpus
from v2_tokenize import mass_tokenize
from v3_fused import FusedIngest


# build a fake GloVe-scale vocab (pulled from corpus's observed tokens
# so lookups actually hit — measures the fused path in its best case)
def build_vocab(articles, max_words=100000):
    toks = []
    for a in articles:
        plaintext = mwparserfromhell.parse(a).strip_code().lower()
        toks.extend(w for w in re.findall(r"[a-z]+", plaintext) if len(w) > 1)
    vocab = sorted(set(toks))[:max_words]
    return {w: i for i, w in enumerate(vocab)}


def run():
    N = 200
    articles = corpus(N, seed=42, size_hint=5000)
    total_bytes = sum(len(a) for a in articles)
    print(f"corpus: {N} articles, {total_bytes/1024:.0f} KB")
    print()

    print("building vocab from corpus...", end=" ", flush=True)
    t0 = time.perf_counter()
    word_to_idx = build_vocab(articles)
    print(f"{len(word_to_idx)} words in {time.perf_counter() - t0:.1f}s")

    # path A: current production
    def path_a(text):
        plaintext = mwparserfromhell.parse(text).strip_code()
        tokens = [w for w in re.findall(r"[a-z]+", plaintext.lower()) if len(w) > 1]
        idxs = [word_to_idx[t] for t in tokens if t in word_to_idx]
        return np.asarray(idxs, dtype=np.int32)

    # path B: v2 fused tokenize, python dict lookup
    def path_b(text):
        tokens = mass_tokenize(text)
        idxs = [word_to_idx[t] for t in tokens if t in word_to_idx]
        return np.asarray(idxs, dtype=np.int32)

    # path C: v3 fully fused
    print("building fused vocab table...", end=" ", flush=True)
    t0 = time.perf_counter()
    fused = FusedIngest(word_to_idx)
    print(f"done in {time.perf_counter() - t0:.1f}s")
    print(f"  vocab_buf: {fused.vocab_buf.nbytes/1024:.0f} KB")
    print(f"  table:     {fused.table.nbytes/1024:.0f} KB")
    print()

    def path_c(text):
        return fused.extract(text)

    # warmup all
    for a in articles[:5]:
        path_a(a); path_b(a); path_c(a)

    # measure
    t0 = time.perf_counter()
    out_a = [path_a(a) for a in articles]
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_b = [path_b(a) for a in articles]
    t_b = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_c = [path_c(a) for a in articles]
    t_c = time.perf_counter() - t0

    print(f"{'path':<40s} {'ms':>10s} {'MB/s':>8s} {'indices':>10s}")
    print("-" * 72)
    for name, t, out in [
        ("A current (mwparser + re + dict)", t_a, out_a),
        ("B v2 fused MASS + python dict", t_b, out_b),
        ("C v3 FULLY fused (bytes -> indices)", t_c, out_c),
    ]:
        total_idxs = sum(len(o) for o in out)
        print(f"{name:<40s} {t*1000:>10.1f} {total_bytes/t/1e6:>8.1f} "
              f"{total_idxs:>10}")

    print()
    print(f"speedup  B over A: {t_a/t_b:.2f}x")
    print(f"speedup  C over A: {t_a/t_c:.2f}x")
    print(f"speedup  C over B: {t_b/t_c:.2f}x")

    # semantic equivalence — do A and C agree on the multiset of indices?
    print()
    print("semantic equivalence (sample 0 — multiset jaccard of indices):")
    for i in range(3):
        sa = set(out_a[i].tolist())
        sc = set(out_c[i].tolist())
        if sa | sc:
            j = len(sa & sc) / len(sa | sc)
            print(f"  article {i}: A={len(out_a[i])} C={len(out_c[i])} "
                  f"jaccard={j:.2%}")


if __name__ == "__main__":
    run()
