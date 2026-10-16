"""
engram_category_test.py

do eigenvalue fingerprints cluster by category?

5 categories × 10 articles = 50 Wikipedia articles.
GloVe word trajectories. no model. no normalization.

if the spectral distributions cluster by category, the codec has
discovered the dynamics of different knowledge structures from raw
word sequences alone.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, eigenvalues_all
from engram_glove_test import load_glove, text_to_trajectory, tokenize_simple
from engram_babel_test import fetch_article, cosine

CATEGORIES = {
    "physics": [
        "General relativity", "Quantum mechanics", "Thermodynamics",
        "Electromagnetism", "Special relativity", "Classical mechanics",
        "Particle physics", "Nuclear physics", "Fluid dynamics", "Optics",
    ],
    "music": [
        "Jazz", "Classical music", "Blues", "Rock music", "Hip hop music",
        "Electronic music", "Opera", "Symphony", "Musical instrument", "Rhythm",
    ],
    "biology": [
        "Photosynthesis", "DNA", "Evolution", "Cell (biology)", "Genetics",
        "Ecology", "Virus", "Protein", "Neuroscience", "Immune system",
    ],
    "history": [
        "Roman Empire", "World War II", "Ancient Egypt", "French Revolution",
        "Industrial Revolution", "Cold War", "Renaissance", "Byzantine Empire",
        "Mongol Empire", "Age of Enlightenment",
    ],
    "computing": [
        "Python (programming language)", "Algorithm", "Computer science",
        "Artificial intelligence", "Machine learning", "Database",
        "Operating system", "Internet", "Cryptography", "Computer network",
    ],
}


def eigenvalue_fingerprint(packet) -> np.ndarray:
    """histogram of |lambda| and freq across all blocks."""
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

    n_mag = len(mag_edges) - 1
    n_freq = len(freq_edges) - 1
    if not all_mags:
        return np.zeros(n_mag + n_freq, dtype=np.float32)

    mag_hist, _ = np.histogram(all_mags, bins=mag_edges)
    freq_hist, _ = np.histogram(all_freqs, bins=freq_edges)
    mag_p = mag_hist.astype(np.float32) / max(mag_hist.sum(), 1)
    freq_p = freq_hist.astype(np.float32) / max(freq_hist.sum(), 1)
    return np.concatenate([mag_p, freq_p])


def run():
    print("=" * 70)
    print("  ENGRAM CATEGORY CLUSTERING TEST")
    print("  5 categories x 10 articles = 50 Wikipedia articles")
    print("  GloVe word trajectories. do eigenvalues cluster by category?")
    print("=" * 70)
    print()

    glove = load_glove()
    codec = Engram(dim=300)

    # encode all articles
    results: dict[str, list[tuple[str, np.ndarray, dict]]] = defaultdict(list)
    all_fps: list[tuple[str, str, np.ndarray]] = []

    for cat, titles in CATEGORIES.items():
        print(f"\n  [{cat}]")
        for title in titles:
            print(f"    {title}...", end=" ", flush=True)
            try:
                text = fetch_article(title, "en")
            except Exception as e:
                print(f"FETCH ERROR: {e}")
                continue

            traj = text_to_trajectory(text, glove)
            if traj.shape[0] < 20:
                print(f"too few words ({traj.shape[0]})")
                continue

            pkt = codec.encode(traj)
            s = summary(pkt)
            fp = eigenvalue_fingerprint(pkt)

            results[cat].append((title, fp, s))
            all_fps.append((cat, title, fp))
            print(f"{traj.shape[0]} words, {s['blocks']} blocks, capture={s['capture']}%")

    print()

    # ── within-category vs between-category similarity ────────────
    print("=" * 70)
    print("  CLUSTERING ANALYSIS")
    print("=" * 70)
    print()

    within_scores = []
    between_scores = []

    cats = sorted(results.keys())
    for ci, cat in enumerate(cats):
        fps_cat = [fp for _, fp, _ in results[cat]]
        # within-category pairs
        for i in range(len(fps_cat)):
            for j in range(i + 1, len(fps_cat)):
                within_scores.append(cosine(fps_cat[i], fps_cat[j]))

        # between-category pairs (sample to keep it manageable)
        for cj_name in cats[ci + 1:]:
            fps_other = [fp for _, fp, _ in results[cj_name]]
            for fi in fps_cat:
                for fj in fps_other:
                    between_scores.append(cosine(fi, fj))

    if within_scores and between_scores:
        mean_within = np.mean(within_scores)
        mean_between = np.mean(between_scores)
        std_within = np.std(within_scores)
        std_between = np.std(between_scores)

        print(f"  within-category cosine:   {mean_within:.4f} +/- {std_within:.4f}  (n={len(within_scores)})")
        print(f"  between-category cosine:  {mean_between:.4f} +/- {std_between:.4f}  (n={len(between_scores)})")
        print(f"  separation:               {mean_within - mean_between:+.4f}")
        print()

        if mean_within > mean_between + 0.02:
            print(f"  categories CLUSTER in eigenvalue space.")
            print(f"  the dynamics of physics are different from the dynamics of music.")
        elif mean_within > mean_between:
            print(f"  weak clustering signal.")
        else:
            print(f"  no clustering detected.")
    print()

    # per-category centroid and inter-centroid distances
    print("  category centroids (cosine distance matrix):")
    centroids = {}
    for cat in cats:
        fps_cat = [fp for _, fp, _ in results[cat]]
        if fps_cat:
            centroids[cat] = np.mean(fps_cat, axis=0)

    if len(centroids) >= 2:
        cnames = sorted(centroids.keys())
        print(f"  {'':12s}", end="")
        for c in cnames:
            print(f"{c:>12s}", end="")
        print()
        for ci in cnames:
            print(f"  {ci:12s}", end="")
            for cj in cnames:
                if ci == cj:
                    print(f"{'---':>12s}", end="")
                else:
                    print(f"{cosine(centroids[ci], centroids[cj]):12.4f}", end="")
            print()

    # per-category capture rates
    print(f"\n  mean energy capture by category:")
    for cat in cats:
        caps = [s["capture"] for _, _, s in results[cat]]
        if caps:
            print(f"    {cat:12s}: {np.mean(caps):.1f}% +/- {np.std(caps):.1f}%")

    # per-category block sizes
    print(f"\n  mean macro block size by category:")
    for cat in cats:
        blocks = [s["macro_block"] for _, _, s in results[cat]]
        if blocks:
            print(f"    {cat:12s}: {np.mean(blocks):.1f} +/- {np.std(blocks):.1f}")


if __name__ == "__main__":
    run()
