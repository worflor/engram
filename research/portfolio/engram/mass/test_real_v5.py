"""test_real_v5.py — add batch mode to the ladder."""

from __future__ import annotations

import sys
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mwparserfromhell

from test_real import read_first_articles, ENWIKI
from v4_tables import FusedIngestV4
from v5_batch import FusedIngestBatch


def run():
    if not Path(ENWIKI).exists():
        print("enwiki bz2 missing, skipping"); return

    print("streaming real enwiki...")
    articles = read_first_articles(ENWIKI, n_articles=80)
    total_bytes = sum(len(w) for _, w in articles)
    texts = [wt for _, wt in articles]
    print(f"  {len(articles)} articles, {total_bytes/1024:.0f} KB")
    print()

    print("building vocab...", end=" ", flush=True)
    t0 = time.perf_counter()
    all_tokens = []
    for _, wt in articles:
        pt = mwparserfromhell.parse(wt).strip_code().lower()
        all_tokens.extend(w for w in re.findall(r"[a-z]+", pt) if len(w) > 1)
    word_to_idx = {w: i for i, w in enumerate(sorted(set(all_tokens)))}
    print(f"{len(word_to_idx)} words in {time.perf_counter() - t0:.1f}s")

    fv4 = FusedIngestV4(word_to_idx)
    fv5 = FusedIngestBatch(word_to_idx)

    # warmup
    for _, wt in articles[:3]:
        fv4.extract(wt)
    fv5.extract_batch(texts[:3])

    def current_path(wt):
        pt = mwparserfromhell.parse(wt).strip_code()
        toks = [w for w in re.findall(r"[a-z]+", pt.lower()) if len(w) > 1]
        return np.array([word_to_idx[t] for t in toks if t in word_to_idx],
                        dtype=np.int32)

    t0 = time.perf_counter()
    cur_out = [current_path(wt) for _, wt in articles]
    t_cur = time.perf_counter() - t0

    t0 = time.perf_counter()
    v4_out = [fv4.extract(wt) for _, wt in articles]
    t_v4 = time.perf_counter() - t0

    t0 = time.perf_counter()
    v5_out = fv5.extract_batch(texts)
    t_v5 = time.perf_counter() - t0

    print()
    print(f"{'path':<20s} {'ms':>8s} {'MB/s':>7s} {'tokens':>10s}  {'mean jac':>8s}")
    print("-" * 65)
    for name, t, out in [
        ("current", t_cur, cur_out),
        ("v4 per-article", t_v4, v4_out),
        ("v5 batch", t_v5, v5_out),
    ]:
        toks = sum(len(o) for o in out)
        js = []
        for c, vo in zip(cur_out, out):
            a = set(c.tolist()); b = set(vo.tolist())
            if a | b:
                js.append(len(a & b) / len(a | b))
        jac = np.mean(js) if js else 1.0
        print(f"{name:<20s} {t*1000:>8.1f} {total_bytes/t/1e6:>7.1f} "
              f"{toks:>10} {jac:>7.1%}")

    print()
    print(f"v4 over current: {t_cur/t_v4:.1f}x")
    print(f"v5 over current: {t_cur/t_v5:.1f}x")
    print(f"v5 over v4:      {t_v4/t_v5:.2f}x")

    # sanity: v4 and v5 should produce IDENTICAL results
    identical = sum(1 for a, b in zip(v4_out, v5_out)
                    if len(a) == len(b) and (a == b).all())
    print(f"\nv4 == v5 bytewise: {identical}/{len(articles)}")


if __name__ == "__main__":
    run()
