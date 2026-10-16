"""
test_codec_equivalence.py — the DOWNSTREAM test.

the user's brain absorbs K centroids, not raw trajectories. if MASS
produces trajectories that are semantically equivalent (high cosine),
do they also produce EQUIVALENT K centroids after going through the
engram codec's derive_pairing + segment + fit_all pipeline?

this is the real acceptance gate. if K centroids match, MASS is a
bit-for-bit drop-in for brain training.
"""

from __future__ import annotations

import sys
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import mwparserfromhell

from test_real import read_first_articles, ENWIKI
from v4_tables import FusedIngestV4
from engram_codec import (
    derive_pairing, apply_pairing, derive_block_sizes, segment,
    fit_all, _to_complex, SEED_COUNT, MIN_BLOCK,
)


def build_vocab_and_matrix(articles, dim=300):
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
    pt = mwparserfromhell.parse(wikitext).strip_code()
    toks = [w for w in re.findall(r"[a-z]+", pt.lower()) if len(w) > 1]
    idxs = [word_to_idx[t] for t in toks if t in word_to_idx]
    if len(idxs) < max(4, window):
        return np.empty((0, matrix.shape[1]), dtype=np.float32)
    raw = matrix[np.asarray(idxs, dtype=np.int32)]
    T, D = raw.shape
    if T < window:
        return raw
    cs = np.empty((T + 1, D), dtype=np.float32); cs[0] = 0
    np.cumsum(raw, axis=0, out=cs[1:])
    return (cs[window:] - cs[:-window]) / np.float32(window)


def mass_embed(wikitext, fused, matrix, window=10):
    idxs = fused.extract(wikitext)
    if len(idxs) < max(4, window):
        return np.empty((0, matrix.shape[1]), dtype=np.float32)
    raw = matrix[idxs]
    T, D = raw.shape
    if T < window:
        return raw
    cs = np.empty((T + 1, D), dtype=np.float32); cs[0] = 0
    np.cumsum(raw, axis=0, out=cs[1:])
    return (cs[window:] - cs[:-window]) / np.float32(window)


def encode_to_k_centroids(traj: np.ndarray, pairing: np.ndarray) -> np.ndarray | None:
    """replicate what BatchedGpuIngest does to turn a trajectory into
    K centroids. returns the mean K across segments, or None if the
    trajectory is too short."""
    if traj.shape[0] <= SEED_COUNT:
        return None
    w = traj.astype(np.float32)
    wp = apply_pairing(w, pairing)
    macro_size, _ = derive_block_sizes(wp)
    segs = segment(wp, macro_size)
    if not segs:
        return None

    all_Ks = []
    for ss, se in segs:
        L = se - ss
        if L < MIN_BLOCK + 1:
            continue
        seed1 = wp[ss - 1] if ss >= 1 else wp[0]
        seed2 = wp[ss - 2] if ss >= 2 else (wp[0] if ss <= 1 else seed1)
        data = wp[ss:se]
        z_full = np.vstack([seed2[np.newaxis], seed1[np.newaxis], data])
        z_c = _to_complex(z_full)
        K, G, _, _ = fit_all(z_c)
        all_Ks.append(K)
    if not all_Ks:
        return None
    # mean K across segments == what absorb() calls obs_centroid
    return np.mean(np.stack(all_Ks), axis=0)


def run():
    if not Path(ENWIKI).exists():
        print("enwiki bz2 missing"); return

    print("streaming real enwiki (60 articles)...")
    articles = read_first_articles(ENWIKI, n_articles=60)
    print(f"  {len(articles)} articles")
    print()

    print("building fake glove...", end=" ", flush=True)
    t0 = time.perf_counter()
    word_to_idx, matrix = build_vocab_and_matrix(articles)
    print(f"{len(word_to_idx)} words in {time.perf_counter() - t0:.1f}s")
    fused = FusedIngestV4(word_to_idx)

    # warmup
    for _, wt in articles[:2]:
        current_embed(wt, word_to_idx, matrix)
        mass_embed(wt, fused, matrix)

    # produce trajectories via both paths
    cur_trajs = [current_embed(wt, word_to_idx, matrix) for _, wt in articles]
    mass_trajs = [mass_embed(wt, fused, matrix) for _, wt in articles]

    # derive reference pairing from the first current trajectory (same as
    # BatchedGpuIngest does)
    ref_pairing = None
    for t in cur_trajs:
        if t.shape[0] >= 10:
            ref_pairing = derive_pairing(t)
            break
    assert ref_pairing is not None

    # now run the codec on both sets of trajectories
    cur_K = []
    mass_K = []
    for a, b in zip(cur_trajs, mass_trajs):
        ka = encode_to_k_centroids(a, ref_pairing)
        kb = encode_to_k_centroids(b, ref_pairing)
        if ka is not None and kb is not None:
            cur_K.append(ka)
            mass_K.append(kb)

    print(f"  encoded {len(cur_K)} articles both ways")
    print()

    # compare K centroids — this is the ACTUAL signal that goes into
    # absorb(). if these agree, MASS is a drop-in replacement.
    cur_stack = np.stack(cur_K)  # [N, P] complex
    mass_stack = np.stack(mass_K)

    # per-article K-centroid cosine similarity
    cs = []
    for ka, kb in zip(cur_K, mass_K):
        denom = (np.linalg.norm(ka) * np.linalg.norm(kb)) + 1e-10
        # complex dot product → real part is cosine of real components
        cs.append(float(np.real(np.vdot(ka, kb)) / denom))
    cs = np.array(cs)
    print(f"K-centroid cosine sim (per article):")
    print(f"  mean:    {cs.mean():.4f}")
    print(f"  median:  {np.median(cs):.4f}")
    print(f"  min:     {cs.min():.4f}")
    print(f"  p10:     {np.percentile(cs, 10):.4f}")
    print(f"  p25:     {np.percentile(cs, 25):.4f}")

    # magnitude ratio
    cur_mag = np.linalg.norm(cur_stack, axis=1)
    mass_mag = np.linalg.norm(mass_stack, axis=1)
    ratio = mass_mag / np.maximum(cur_mag, 1e-10)
    print()
    print(f"K-centroid magnitude ratio mass/current:")
    print(f"  mean:    {ratio.mean():.3f}")
    print(f"  median:  {np.median(ratio):.3f}")

    # worst article
    worst = int(np.argmin(cs))
    print()
    print(f"worst article: {articles[worst][0]!r}  cos={cs[worst]:.3f}")
    print(f"  cur traj len: {cur_trajs[worst].shape}")
    print(f"  mass traj len: {mass_trajs[worst].shape}")


if __name__ == "__main__":
    run()
