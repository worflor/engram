"""
engram_glove_test.py

Engram on raw word trajectories. no model. no inference. no normalization.

GloVe maps each word to a fixed point in R^300. a sentence is a sequence
of those points. the trajectory dynamics come from how the vocabulary
sequences, not from a neural network smoothing them.

scientific text oscillates between technical terms and common words.
narrative drifts through related words. code repeats keyword patterns.
the eigenvalues should capture these differences because nothing is
normalizing them away.
"""

from __future__ import annotations

import sys
import os
import re
import time
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, spectral_summary, eigenvalues_all
from engram_babel_test import fetch_article, cosine, ARTICLES, LANGS

GLOVE_PATH = Path(os.path.dirname(__file__)) / ".." / "cache" / "_glove_cache" / "glove.6B.300d.txt"
CACHE_DIR = Path(os.path.dirname(__file__)) / ".." / "cache" / "_babel_cache"


# ── GloVe loader ──────────────────────────────────────────────────────

def load_glove(path: str = str(GLOVE_PATH), dim: int = 300) -> dict[str, np.ndarray]:
    """load GloVe embeddings. caches the parsed dict as a pickle."""
    cache = Path(str(path) + ".pkl")
    if cache.exists():
        print(f"loading GloVe from cache...", end=" ", flush=True)
        t0 = time.time()
        with open(str(cache), "rb") as f:
            glove = pickle.load(f)
        print(f"{len(glove)} words in {time.time() - t0:.1f}s")
        return glove

    print(f"parsing GloVe ({path})...", end=" ", flush=True)
    t0 = time.time()
    glove: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if vec.shape[0] == dim:
                glove[word] = vec
    print(f"{len(glove)} words in {time.time() - t0:.1f}s")

    print("caching pickle...", end=" ", flush=True)
    with open(str(cache), "wb") as f:
        pickle.dump(glove, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")

    return glove


# ── text to trajectory ────────────────────────────────────────────────

def tokenize_simple(text: str) -> list[str]:
    """lowercase, split on non-alpha, drop short tokens."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def text_to_trajectory(
    text: str,
    glove: dict[str, np.ndarray],
    window: int = 1,
) -> np.ndarray:
    """
    convert text to a trajectory in R^300.

    window=1 (default): one vector per word. raw vocabulary dynamics.
    window>1: sliding average over N words. captures phrase-level
              semantics from static embeddings. kills per-word jitter,
              preserves topic drift. the bass, not the treble.

    the window size controls the semantic resolution:
        3-5:   phrase-level (noun phrases, verb phrases)
        10-20: clause-level (subject-verb-object)
        50+:   sentence-level (very smooth, topic-only)
    """
    tokens = tokenize_simple(text)
    vecs = [glove[t] for t in tokens if t in glove]
    if len(vecs) < max(4, window):
        return np.empty((0, 300), dtype=np.float32)

    raw = np.stack(vecs)

    if window <= 1:
        return raw

    # sliding window average: each point is the mean of its neighborhood
    T, D = raw.shape
    smoothed = np.empty((T - window + 1, D), dtype=np.float32)
    # cumulative sum for O(1) per-window averaging
    cumsum = np.vstack([np.zeros((1, D), dtype=np.float32), np.cumsum(raw, axis=0)])
    for i in range(T - window + 1):
        smoothed[i] = (cumsum[i + window] - cumsum[i]) / window

    return smoothed


# ── eigenvalue histogram fingerprint ──────────────────────────────────

def eigenvalue_fingerprint(packet) -> np.ndarray:
    """spectral distribution fingerprint (histogram of |lambda| and freq)."""
    mag_edges = np.array([0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.5, 1e6])
    freq_edges = np.array([0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, np.pi + 0.01])

    all_mags, all_freqs = [], []
    for blk in packet.blocks:
        if blk.macro_K is not None and len(blk.macro_K) > 0:
            l1, l2 = eigenvalues_all(blk.macro_K, blk.macro_G)
            all_mags.extend(np.abs(l1).tolist())
            all_mags.extend(np.abs(l2).tolist())
            all_freqs.extend(np.abs(np.angle(l1)).tolist())
            all_freqs.extend(np.abs(np.angle(l2)).tolist())

    if not all_mags:
        return np.zeros(len(mag_edges) - 1 + len(freq_edges) - 1, dtype=np.float32)

    mag_hist, _ = np.histogram(all_mags, bins=mag_edges)
    freq_hist, _ = np.histogram(all_freqs, bins=freq_edges)
    mag_p = mag_hist.astype(np.float32) / max(mag_hist.sum(), 1)
    freq_p = freq_hist.astype(np.float32) / max(freq_hist.sum(), 1)
    return np.concatenate([mag_p, freq_p])


# ── the test ──────────────────────────────────────────────────────────

HAND_TEXTS = {
    "physics": "The gravitational force between two bodies is proportional to the product of their masses divided by the square of the distance between them. This fundamental law governs the motion of planets and the trajectories of satellites in orbit around the earth.",
    "cooking": "Dice the onion and saute in butter until translucent. Add garlic and stir for thirty seconds. Pour in the broth and bring to a boil then reduce heat and simmer for twenty minutes. Season with salt and pepper to taste.",
    "code": "def fibonacci sequence number return zero if number is less than or equal to zero return one if number equals one return fibonacci of number minus one plus fibonacci of number minus two",
    "poetry": "the fog comes on little cat feet it sits looking over harbor and city on silent haunches and then moves on",
    "legal": "The party of the first part hereinafter referred to as the licensor grants to the party of the second part hereinafter referred to as the licensee a nonexclusive royalty free license to use reproduce and distribute the software.",
}


def run():
    print("=" * 70)
    print("  ENGRAM GLOVE TEST")
    print("  raw word trajectories. no model. no inference. no normalization.")
    print("=" * 70)
    print()

    glove = load_glove()
    codec = Engram(dim=300)

    # ── hand-crafted genre test ───────────────────────────────────
    print("--- genre test (hand-crafted texts) ---\n")

    genre_fps = {}
    for genre, text in HAND_TEXTS.items():
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] < 4:
            print(f"  {genre}: too few words in GloVe")
            continue

        pkt = codec.encode(traj)
        s = summary(pkt)
        fp = eigenvalue_fingerprint(pkt)
        genre_fps[genre] = fp

        spec = spectral_summary(pkt.blocks[0].macro_K, pkt.blocks[0].macro_G) if pkt.blocks else {}
        print(f"  {genre:12s}: {traj.shape[0]} words, {s['blocks']} blocks, capture={s['capture']}%")
        if spec:
            print(f"               freq={spec['freq_mean']:.3f} mag={spec['mag_mean']:.3f} dec={spec['n_decaying']} sus={spec['n_sustained']} div={spec['n_divergent']}")

    print()
    if len(genre_fps) >= 2:
        genres = sorted(genre_fps.keys())
        print(f"  {'':12s}", end="")
        for g in genres:
            print(f"{g:>10s}", end="")
        print()
        for g1 in genres:
            print(f"  {g1:12s}", end="")
            for g2 in genres:
                if g1 == g2:
                    print(f"{'---':>10s}", end="")
                else:
                    print(f"{cosine(genre_fps[g1], genre_fps[g2]):10.4f}", end="")
            print()

    # ── Wikipedia cross-lingual test ──────────────────────────────
    print("\n--- Wikipedia cross-lingual (GloVe, English only for now) ---\n")

    wiki_fps = {}
    for art in ARTICLES:
        topic = art["topic"]
        title = art["en"]

        text = fetch_article(title, "en")
        if len(text) < 100:
            continue

        traj = text_to_trajectory(text, glove)
        if traj.shape[0] < 10:
            print(f"  {topic}: too few GloVe words ({traj.shape[0]})")
            continue

        pkt = codec.encode(traj)
        s = summary(pkt)
        fp = eigenvalue_fingerprint(pkt)
        wiki_fps[topic] = fp

        print(f"  {topic:24s}: {traj.shape[0]} words, {s['blocks']} blocks, capture={s['capture']}%")

    print()
    if len(wiki_fps) >= 2:
        topics = sorted(wiki_fps.keys())
        pairs = []
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                sc = cosine(wiki_fps[topics[i]], wiki_fps[topics[j]])
                pairs.append((sc, topics[i], topics[j]))
        pairs.sort()
        print("  most DIFFERENT topic pairs (lowest cosine):")
        for sc, t1, t2 in pairs[:5]:
            print(f"    {t1:20s} ~ {t2:20s}: {sc:.4f}")
        print("  most SIMILAR topic pairs:")
        for sc, t1, t2 in pairs[-5:]:
            print(f"    {t1:20s} ~ {t2:20s}: {sc:.4f}")

        mean_sim = np.mean([p[0] for p in pairs])
        std_sim = np.std([p[0] for p in pairs])
        print(f"\n  mean cosine: {mean_sim:.4f} +/- {std_sim:.4f}")
        if std_sim > 0.05:
            print(f"  there IS variance. different topics have different eigenvalue distributions.")
        else:
            print(f"  low variance. eigenvalue distributions are still too similar.")

    # ── dream from Wikipedia ──────────────────────────────────────
    print("\n--- dream ---")
    all_pkts = []
    for art in ARTICLES:
        text = fetch_article(art["en"], "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 10:
            all_pkts.append(codec.encode(traj))

    if len(all_pkts) >= 3:
        gen = codec.dream(all_pkts, n_blocks=4, samples_per_block=8)
        if gen is not None:
            print(f"  dreamed {gen.shape[0]} word-vectors from {len(all_pkts)} articles")
            print(f"  range: [{gen.min():.4f}, {gen.max():.4f}]")

            # find nearest GloVe word for each dreamed vector
            # (just the first 5, as a sanity check)
            all_words = list(glove.keys())[:50000]
            all_vecs = np.stack([glove[w] for w in all_words])
            print("  nearest words for first 5 dreamed vectors:")
            for i in range(min(5, gen.shape[0])):
                dists = np.linalg.norm(all_vecs - gen[i, :300], axis=1)
                nearest_idx = np.argmin(dists)
                print(f"    [{i}] {all_words[nearest_idx]} (dist={dists[nearest_idx]:.3f})")


if __name__ == "__main__":
    run()
