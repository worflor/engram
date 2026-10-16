"""bench_v0.py — baseline measurement: mwparserfromhell vs python MASS.

we are NOT yet trying to be fast. just measuring semantic equivalence
and the starting point for the speedup ladder.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mwparserfromhell

from corpus import corpus
from v0_python import mass_strip


def run():
    N = 200
    articles = corpus(N, seed=42, size_hint=5000)
    total_bytes = sum(len(a) for a in articles)
    print(f"corpus: {N} articles, {total_bytes/1024:.0f} KB total, "
          f"avg {total_bytes/N:.0f} B/article")
    print()

    # warmup both
    for a in articles[:5]:
        mwparserfromhell.parse(a).strip_code()
        mass_strip(a)

    # mwparserfromhell
    t0 = time.perf_counter()
    mwp_outputs = [mwparserfromhell.parse(a).strip_code() for a in articles]
    t_mwp = time.perf_counter() - t0

    # v0 python
    t0 = time.perf_counter()
    v0_outputs = [mass_strip(a) for a in articles]
    t_v0 = time.perf_counter() - t0

    mwp_total = sum(len(o) for o in mwp_outputs)
    v0_total = sum(len(o) for o in v0_outputs)

    print(f"mwparserfromhell:  {t_mwp*1000:>8.1f} ms   "
          f"{total_bytes/t_mwp/1e6:>6.1f} MB/s   "
          f"output={mwp_total/1024:.0f} KB")
    print(f"v0 python MASS:    {t_v0*1000:>8.1f} ms   "
          f"{total_bytes/t_v0/1e6:>6.1f} MB/s   "
          f"output={v0_total/1024:.0f} KB")
    print(f"speedup: {t_mwp/t_v0:.1f}x")
    print()

    # semantic drift: check how much they disagree on prose content
    # (we don't expect bit-identity, since MASS strips differently — we
    # expect roughly similar output lengths and >80% shared words)
    import re
    def words(s):
        return set(w for w in re.findall(r"[a-zA-Z]{2,}", s.lower()))

    mwp_words = words(mwp_outputs[0])
    v0_words = words(v0_outputs[0])
    if mwp_words:
        jaccard = len(mwp_words & v0_words) / len(mwp_words | v0_words)
        print(f"word overlap (sample 0 jaccard): {jaccard:.2%}")

    # find first meaningful divergence for debugging
    for idx, (m, v) in enumerate(zip(mwp_outputs[:3], v0_outputs[:3])):
        print(f"\n--- article {idx} ---")
        print(f"  mwp len={len(m)}  v0 len={len(v)}")
        # show first 150 chars of each
        print(f"  mwp[:150] = {m[:150]!r}")
        print(f"  v0 [:150] = {v[:150]!r}")


if __name__ == "__main__":
    run()
