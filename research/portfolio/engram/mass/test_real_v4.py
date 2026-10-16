"""test_real_v4.py — same as test_real.py but with v4 grammar."""

from __future__ import annotations

import sys
import time
import bz2
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mwparserfromhell

from test_real import read_first_articles, ENWIKI
from v3_fused import FusedIngest
from v4_tables import FusedIngestV4


def run():
    if not Path(ENWIKI).exists():
        print("enwiki bz2 missing, skipping"); return

    print("streaming real enwiki...")
    articles = read_first_articles(ENWIKI, n_articles=40)
    total_bytes = sum(len(w) for _, w in articles)
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

    fv3 = FusedIngest(word_to_idx)
    fv4 = FusedIngestV4(word_to_idx)

    for _, wt in articles[:3]:
        fv3.extract(wt); fv4.extract(wt)

    def current_path(wt):
        pt = mwparserfromhell.parse(wt).strip_code()
        toks = [w for w in re.findall(r"[a-z]+", pt.lower()) if len(w) > 1]
        return np.array([word_to_idx[t] for t in toks if t in word_to_idx],
                        dtype=np.int32)

    t0 = time.perf_counter()
    cur_out = [current_path(wt) for _, wt in articles]
    t_cur = time.perf_counter() - t0

    t0 = time.perf_counter()
    v3_out = [fv3.extract(wt) for _, wt in articles]
    t_v3 = time.perf_counter() - t0

    t0 = time.perf_counter()
    v4_out = [fv4.extract(wt) for _, wt in articles]
    t_v4 = time.perf_counter() - t0

    print()
    print(f"{'path':<20s} {'ms':>8s} {'MB/s':>7s} {'tokens':>10s}  {'mean jac':>8s}")
    print("-" * 65)
    for name, t, out in [
        ("current", t_cur, cur_out),
        ("v3",      t_v3, v3_out),
        ("v4",      t_v4, v4_out),
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

    # detailed v4 jaccard breakdown
    v4_js = []
    for c, vo in zip(cur_out, v4_out):
        a = set(c.tolist()); b = set(vo.tolist())
        if a | b:
            v4_js.append(len(a & b) / len(a | b))
    vj = np.array(v4_js)
    print()
    print(f"v4 jaccard distribution:")
    print(f"  mean={vj.mean():.2%}  median={np.median(vj):.2%}  "
          f"min={vj.min():.2%}  p25={np.percentile(vj,25):.2%}")

    # speedup
    print()
    print(f"v3 over current: {t_cur/t_v3:.1f}x")
    print(f"v4 over current: {t_cur/t_v4:.1f}x")
    print(f"v4 over v3:      {t_v3/t_v4:.2f}x (slower because more rules)")

    # worst case with v4
    worst = int(np.argmin(vj))
    print()
    print(f"v4 worst article: {articles[worst][0]!r}  j={vj[worst]:.2%}")
    rev = {i: w for w, i in word_to_idx.items()}
    only_cur = list(set(cur_out[worst].tolist()) - set(v4_out[worst].tolist()))[:10]
    only_v4  = list(set(v4_out[worst].tolist()) - set(cur_out[worst].tolist()))[:10]
    print(f"  only-current: {[rev[i] for i in only_cur]}")
    print(f"  only-v4:      {[rev[i] for i in only_v4]}")


if __name__ == "__main__":
    run()
