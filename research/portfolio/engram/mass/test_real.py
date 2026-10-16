"""
test_real.py — run v3 MASS on a small slice of actual enwiki.

no disk explosion: we open the existing bz2 file, stream a few MB of
decompressed text, extract the first N articles, test MASS against
mwparserfromhell. measures the ONE thing that matters: does our
state machine produce semantically equivalent output on real
wikipedia markup, not our synthetic corpus.
"""

from __future__ import annotations

import sys
import time
import bz2
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mwparserfromhell
import lxml.etree as ET

from v3_fused import FusedIngest


ENWIKI = str(Path.home() / "Downloads" / "enwiki-20260301-pages-articles-multistream.xml.bz2")


def read_first_articles(path: str, n_articles: int = 50, max_bytes: int = 20_000_000):
    """
    stream bz2 → XML → yield (title, wikitext) pairs until we have
    n_articles or we've read max_bytes. hard cap on read so we don't
    eat RAM.
    """
    articles = []
    with bz2.open(path, "rb") as f:
        # bz2 streams; we ask iterparse to read as needed.
        # to avoid reading the whole file, we wrap with a size-limiting
        # reader. but lxml's iterparse doesn't easily accept a read
        # limit, so we just break out when we have enough.
        for event, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "page":
                title_el = None
                text_el = None
                for child in elem.iter():
                    if child.tag.endswith("title") or child.tag == "title":
                        title_el = child
                    if child.tag.endswith("text") or child.tag == "text":
                        text_el = child
                if title_el is not None and text_el is not None:
                    title = title_el.text or ""
                    wikitext = text_el.text or ""
                    if (wikitext.startswith("#REDIRECT") or ":" in title
                            or len(wikitext) < 1000):
                        elem.clear()
                        continue
                    articles.append((title, wikitext))
                    if len(articles) >= n_articles:
                        break
                elem.clear()
    return articles


def run():
    if not Path(ENWIKI).exists():
        print(f"enwiki bz2 not found at {ENWIKI}; skipping")
        return

    print("streaming real enwiki...")
    t0 = time.perf_counter()
    articles = read_first_articles(ENWIKI, n_articles=40)
    read_t = time.perf_counter() - t0
    total_bytes = sum(len(w) for _, w in articles)
    print(f"  {len(articles)} articles, {total_bytes/1024:.0f} KB of wikitext "
          f"in {read_t:.1f}s")
    print()

    # show first article title + preview
    print(f"first article: {articles[0][0]!r}")
    print(f"  wikitext[:200]: {articles[0][1][:200]!r}")
    print()

    # build vocab from actual mwparser output
    print("building vocab from mwparser output...", end=" ", flush=True)
    t0 = time.perf_counter()
    all_tokens = []
    for _, wt in articles:
        plaintext = mwparserfromhell.parse(wt).strip_code().lower()
        all_tokens.extend(w for w in re.findall(r"[a-z]+", plaintext) if len(w) > 1)
    vocab_words = sorted(set(all_tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab_words)}
    print(f"{len(word_to_idx)} words in {time.perf_counter() - t0:.1f}s")

    fused = FusedIngest(word_to_idx)

    # warmup
    for _, wt in articles[:3]:
        fused.extract(wt)

    # measure: current path vs v3
    def current_path(wt):
        pt = mwparserfromhell.parse(wt).strip_code()
        toks = [w for w in re.findall(r"[a-z]+", pt.lower()) if len(w) > 1]
        idxs = np.array([word_to_idx[t] for t in toks if t in word_to_idx],
                        dtype=np.int32)
        return idxs

    t0 = time.perf_counter()
    cur_out = [current_path(wt) for _, wt in articles]
    t_cur = time.perf_counter() - t0

    t0 = time.perf_counter()
    v3_out = [fused.extract(wt) for _, wt in articles]
    t_v3 = time.perf_counter() - t0

    print()
    print(f"{'path':<30s} {'ms':>8s} {'MB/s':>8s} {'tokens':>10s}")
    print("-" * 60)
    print(f"{'current (mwp + re + dict)':<30s} {t_cur*1000:>8.1f} "
          f"{total_bytes/t_cur/1e6:>8.1f} {sum(len(o) for o in cur_out):>10}")
    print(f"{'v3 fused':<30s} {t_v3*1000:>8.1f} "
          f"{total_bytes/t_v3/1e6:>8.1f} {sum(len(o) for o in v3_out):>10}")
    print(f"speedup: {t_cur/t_v3:.1f}x")
    print()

    # semantic comparison
    jaccards = []
    for i, ((title, _), cur, v3) in enumerate(zip(articles, cur_out, v3_out)):
        a = set(cur.tolist()); b = set(v3.tolist())
        if a | b:
            j = len(a & b) / len(a | b)
            jaccards.append(j)
    js = np.array(jaccards)
    print(f"jaccard of index sets (per article, n={len(js)}):")
    print(f"  mean:   {js.mean():.2%}")
    print(f"  median: {np.median(js):.2%}")
    print(f"  min:    {js.min():.2%}")
    print(f"  p25:    {np.percentile(js, 25):.2%}")

    # token count ratio
    cur_n = np.array([len(o) for o in cur_out])
    v3_n  = np.array([len(o) for o in v3_out])
    ratio = v3_n / np.maximum(cur_n, 1)
    print()
    print(f"token count ratio v3/current (per article):")
    print(f"  mean:   {ratio.mean():.2%}")
    print(f"  median: {np.median(ratio):.2%}")
    print(f"  min:    {ratio.min():.2%}   max: {ratio.max():.2%}")

    # show worst-case article
    worst = int(np.argmin(js))
    print()
    print(f"worst jaccard article: {articles[worst][0]!r}  j={js[worst]:.2%}")
    cur_set = set(cur_out[worst].tolist())
    v3_set = set(v3_out[worst].tolist())
    rev = {i: w for w, i in word_to_idx.items()}
    only_cur = list(cur_set - v3_set)[:10]
    only_v3  = list(v3_set - cur_set)[:10]
    print(f"  only-current sample: {[rev[i] for i in only_cur]}")
    print(f"  only-v3 sample:      {[rev[i] for i in only_v3]}")


if __name__ == "__main__":
    run()
