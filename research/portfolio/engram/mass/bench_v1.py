"""bench_v1.py — add numba MASS to the ladder."""

from __future__ import annotations
import sys, time, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mwparserfromhell
from corpus import corpus
from v0_python import mass_strip as mass_v0
from v1_numba import mass_strip as mass_v1


def run():
    N = 200
    articles = corpus(N, seed=42, size_hint=5000)
    total_bytes = sum(len(a) for a in articles)
    print(f"corpus: {N} articles, {total_bytes/1024:.0f} KB")
    print()

    # warmup (numba JIT compiles on first call — first article included)
    for a in articles[:5]:
        mwparserfromhell.parse(a).strip_code()
        mass_v0(a)
        mass_v1(a)

    runs = []

    t0 = time.perf_counter()
    mwp_outputs = [mwparserfromhell.parse(a).strip_code() for a in articles]
    runs.append(("mwparserfromhell", time.perf_counter() - t0, mwp_outputs))

    t0 = time.perf_counter()
    v0_outputs = [mass_v0(a) for a in articles]
    runs.append(("v0 python MASS", time.perf_counter() - t0, v0_outputs))

    t0 = time.perf_counter()
    v1_outputs = [mass_v1(a) for a in articles]
    runs.append(("v1 numba MASS", time.perf_counter() - t0, v1_outputs))

    base_t = runs[0][1]
    for name, t, out in runs:
        bytes_out = sum(len(o) for o in out)
        mbps = total_bytes / t / 1e6
        print(f"{name:<22s}  {t*1000:>8.1f} ms   "
              f"{mbps:>6.1f} MB/s   "
              f"out={bytes_out/1024:.0f} KB   "
              f"speedup {base_t/t:>5.2f}x")

    # semantic equivalence check: v1 vs v0 on prose words
    def words(s):
        return set(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", s))

    print()
    print("semantic equivalence (jaccard of 3+char words, sample 0):")
    s0_mwp = words(runs[0][2][0])
    s0_v0 = words(runs[1][2][0])
    s0_v1 = words(runs[2][2][0])
    for (a_name, a_set), (b_name, b_set) in [
        (("mwp", s0_mwp), ("v0", s0_v0)),
        (("mwp", s0_mwp), ("v1", s0_v1)),
        (("v0", s0_v0), ("v1", s0_v1)),
    ]:
        if a_set | b_set:
            j = len(a_set & b_set) / len(a_set | b_set)
            print(f"  {a_name} vs {b_name}: {j:.2%}")

    # byte-for-byte agreement between v0 and v1 (should be near-identical)
    exact_match = sum(1 for a, b in zip(runs[1][2], runs[2][2]) if a == b)
    print(f"\nv0 == v1 bytewise: {exact_match}/{N}")


if __name__ == "__main__":
    run()
