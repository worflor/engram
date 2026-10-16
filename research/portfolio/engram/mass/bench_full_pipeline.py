"""
bench_full_pipeline.py — the ACTUAL end-to-end test.

measure:
  A. current production path: bz2 → xml → strip_code → re.findall → dict → matrix[idx]
  B. mass-backed path:         bz2 → xml → v4_mass(raw_wikitext) → matrix[idx]

both produce a float32 trajectory ready for downstream GPU. A and B are
compared on wall-clock time and on how closely their trajectories agree
(cosine similarity between per-article mean vectors — a proxy for whether
the semantic content is preserved).
"""

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


def build_vocab_and_matrix(articles, dim=300):
    """fake GloVe: deterministic random vectors for observed words."""
    all_toks = []
    for _, wt in articles:
        pt = mwparserfromhell.parse(wt).strip_code().lower()
        all_toks.extend(w for w in re.findall(r"[a-z]+", pt) if len(w) > 1)
    vocab = sorted(set(all_toks))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    rng = np.random.default_rng(1234)
    matrix = rng.standard_normal((len(vocab), dim)).astype(np.float32) * 0.3
    return word_to_idx, matrix


def current_embed(wikitext, word_to_idx, matrix, window=10):
    """production path: strip → re → dict → matrix[] → smooth."""
    pt = mwparserfromhell.parse(wikitext).strip_code()
    toks = [w for w in re.findall(r"[a-z]+", pt.lower()) if len(w) > 1]
    idxs = [word_to_idx[t] for t in toks if t in word_to_idx]
    if len(idxs) < max(4, window):
        return np.empty((0, matrix.shape[1]), dtype=np.float32)
    raw = matrix[np.asarray(idxs, dtype=np.int32)]
    T, D = raw.shape
    if T < window:
        return raw
    cumsum = np.empty((T + 1, D), dtype=np.float32)
    cumsum[0] = 0
    np.cumsum(raw, axis=0, out=cumsum[1:])
    return (cumsum[window:] - cumsum[:-window]) / np.float32(window)


def mass_embed(wikitext, fused: FusedIngestV4, matrix, window=10):
    """mass path: v4 → matrix[] → smooth. no strip_code, no re, no dict."""
    idxs = fused.extract(wikitext)
    if len(idxs) < max(4, window):
        return np.empty((0, matrix.shape[1]), dtype=np.float32)
    raw = matrix[idxs]
    T, D = raw.shape
    if T < window:
        return raw
    cumsum = np.empty((T + 1, D), dtype=np.float32)
    cumsum[0] = 0
    np.cumsum(raw, axis=0, out=cumsum[1:])
    return (cumsum[window:] - cumsum[:-window]) / np.float32(window)


def run():
    if not Path(ENWIKI).exists():
        print("enwiki bz2 missing, skipping"); return

    print("streaming real enwiki (80 articles)...")
    articles = read_first_articles(ENWIKI, n_articles=80)
    total_bytes = sum(len(wt) for _, wt in articles)
    print(f"  {total_bytes/1024:.0f} KB of wikitext")
    print()

    print("building fake glove from observed vocab...", end=" ", flush=True)
    t0 = time.perf_counter()
    word_to_idx, matrix = build_vocab_and_matrix(articles)
    print(f"{len(word_to_idx)} words in {time.perf_counter() - t0:.1f}s")

    fused = FusedIngestV4(word_to_idx)

    # warmup mass + current
    for _, wt in articles[:3]:
        current_embed(wt, word_to_idx, matrix)
        mass_embed(wt, fused, matrix)

    # measure
    t0 = time.perf_counter()
    cur_trajs = [current_embed(wt, word_to_idx, matrix) for _, wt in articles]
    t_cur = time.perf_counter() - t0

    t0 = time.perf_counter()
    mass_trajs = [mass_embed(wt, fused, matrix) for _, wt in articles]
    t_mass = time.perf_counter() - t0

    print()
    print(f"{'path':<25s} {'ms':>8s} {'MB/s':>7s} {'total_T':>10s}")
    print("-" * 55)
    t_cur_toks = sum(t.shape[0] for t in cur_trajs)
    t_mass_toks = sum(t.shape[0] for t in mass_trajs)
    print(f"{'current production':<25s} {t_cur*1000:>8.1f} "
          f"{total_bytes/t_cur/1e6:>7.1f} {t_cur_toks:>10}")
    print(f"{'v4 mass end-to-end':<25s} {t_mass*1000:>8.1f} "
          f"{total_bytes/t_mass/1e6:>7.1f} {t_mass_toks:>10}")
    print()
    print(f"end-to-end speedup: {t_cur/t_mass:.1f}x")

    # trajectory equivalence: compare per-article MEAN VECTOR cosine sim
    # (rough proxy for "did we preserve semantic content")
    cos_sims = []
    for a, b in zip(cur_trajs, mass_trajs):
        if a.shape[0] == 0 or b.shape[0] == 0:
            continue
        ma = a.mean(axis=0)
        mb = b.mean(axis=0)
        denom = (np.linalg.norm(ma) * np.linalg.norm(mb)) + 1e-10
        cos_sims.append(float(np.dot(ma, mb) / denom))
    cs = np.array(cos_sims)
    print()
    print(f"trajectory mean-vector cosine similarity (per article):")
    print(f"  mean:    {cs.mean():.3f}")
    print(f"  median:  {np.median(cs):.3f}")
    print(f"  min:     {cs.min():.3f}")
    print(f"  p10:     {np.percentile(cs, 10):.3f}")

    # trajectory length comparison
    print()
    lens_cur = np.array([t.shape[0] for t in cur_trajs])
    lens_mass = np.array([t.shape[0] for t in mass_trajs])
    ratio = lens_mass / np.maximum(lens_cur, 1)
    print(f"trajectory length ratio mass/current:")
    print(f"  mean:   {ratio.mean():.2%}")
    print(f"  median: {np.median(ratio):.2%}")


if __name__ == "__main__":
    run()
