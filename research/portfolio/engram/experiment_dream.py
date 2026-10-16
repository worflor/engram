"""
experiment_dream.py — what does alexandria dream about?

approach: for each well centroid K, we ask "which GloVe words, when encoded
through the codec, produce eigenvalues closest to this well?"

this is the CORRECT reverse mapping: well centroid -> nearest eigenvalue
signatures -> the words that produce those signatures.

also generates oscillator trajectories in embedding space using K,G
from dream buffer entries as the dynamics, with real GloVe word seeds.
"""

from __future__ import annotations

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "experiments"))

from engram_codec import (EngramFile, Engram, _to_complex, _from_complex,
                          fit_all, predict_all, apply_pairing, derive_pairing)

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")


def load_glove_matrix(glove: dict):
    """build word list + normalized matrix for cosine similarity."""
    words = list(glove.keys())
    vecs = np.stack([glove[w] for w in words]).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return words, vecs, (vecs / norms).astype(np.float32)


def encode_word_batch(word_vecs: np.ndarray, pairing: np.ndarray) -> np.ndarray:
    """encode a batch of word vectors into K eigenvalues.

    takes sliding windows of 4 words, fits AR(2), returns K centroids.
    word_vecs: [N, 300] — consecutive word embeddings
    returns: [N-3, P] complex — K per window
    """
    wp = apply_pairing(word_vecs, pairing)
    P = wp.shape[1] // 2
    window = 4  # minimum for AR(2) fit

    Ks = []
    for i in range(len(wp) - window + 1):
        chunk = wp[i:i + window]
        z = _to_complex(chunk)
        K, G, _, _ = fit_all(z)
        Ks.append(K)

    if not Ks:
        return np.empty((0, P), dtype=np.complex128)
    return np.stack(Ks)


def main():
    print("=" * 60)
    print("  EXPERIMENT: What Does Alexandria Dream About?")
    print("=" * 60)

    brain = EngramFile.load(BRAIN_PATH)
    P = brain.dim // 2
    pairing = brain._reference_pairing
    well_centroids = brain.wells()
    print(f"brain: {brain._total_absorbed:,} articles, {len(well_centroids)} wells")
    print(f"dream buffer: {len(brain._dream_Ks)} entries")

    print("loading GloVe...", flush=True)
    from engram_glove_test import load_glove
    glove = load_glove()

    print("building lookup structures...", flush=True)
    t0 = time.time()
    words, vecs, vecs_norm = load_glove_matrix(glove)
    print(f"  {len(words):,} words in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════
    #  PART 1: WELL VOCABULARY
    #  for each well, find which common English words produce
    #  eigenvalues closest to the well centroid when encoded
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  PART 1: WELL VOCABULARY")
    print("  which words resonate with each well?")
    print("=" * 60)

    # sample common words (top 20K by GloVe frequency = list position)
    sample_words = words[:20000]
    sample_vecs = vecs[:20000]

    # encode all sampled words in sliding windows of 4
    # build random 4-word sequences from the vocabulary
    print("  encoding 5000 random word sequences...", flush=True)
    rng = np.random.RandomState(42)
    n_sequences = 5000
    seq_len = 8  # 8 words per sequence = multiple windows

    word_K_map = {}  # word -> list of K centroids it appeared in
    for si in range(n_sequences):
        indices = rng.randint(0, len(sample_words), size=seq_len)
        seq_vecs = sample_vecs[indices]
        Ks = encode_word_batch(seq_vecs, pairing)
        for i, idx in enumerate(indices):
            w = sample_words[idx]
            # each word contributes to windows it's part of
            for ki in range(max(0, i - 3), min(len(Ks), i + 1)):
                if ki < len(Ks):
                    word_K_map.setdefault(w, []).append(Ks[ki])

    # compute mean K per word
    print(f"  {len(word_K_map)} words with eigenvalue signatures")
    word_centroids = {}
    for w, ks in word_K_map.items():
        if len(ks) >= 2:  # need multiple observations for stable mean
            word_centroids[w] = np.mean(np.stack(ks), axis=0)

    print(f"  {len(word_centroids)} words with stable signatures\n")

    # for each well, find closest words
    top_wells = sorted(brain._well_count.items(), key=lambda x: -x[1])[:20]

    wc_words = list(word_centroids.keys())
    wc_matrix = np.stack([word_centroids[w] for w in wc_words])  # [N, P] complex

    for wname, count in top_wells:
        centroid = well_centroids[wname]

        # distance from each word's eigenvalue signature to this well
        diffs = wc_matrix - centroid[np.newaxis, :]
        dists = np.sqrt(np.mean(np.abs(diffs) ** 2, axis=1).real)

        top_idx = np.argsort(dists)[:15]
        top = [(wc_words[i], float(dists[i])) for i in top_idx]

        label = wname if not wname.startswith("well_") else f"{wname} (unnamed)"
        print(f"  {label:>25s} ({count:>10,}):  {', '.join(w for w, d in top)}")

    # ══════════════════════════════════════════════════════════════
    #  PART 2: BOUNDARY WORDS
    #  interpolate between wells, find words at the boundary
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  PART 2: BOUNDARY WORDS")
    print("  what words live between two domains?")
    print("=" * 60)

    boundaries = [
        ("physics", "music"),
        ("philosophy", "computing"),
        ("chemistry", "biology"),
        ("history", "law"),
        ("noun", "verb"),
    ]

    for a_name, b_name in boundaries:
        if a_name not in well_centroids or b_name not in well_centroids:
            continue
        K_a = well_centroids[a_name]
        K_b = well_centroids[b_name]
        K_mid = 0.5 * K_a + 0.5 * K_b

        diffs = wc_matrix - K_mid[np.newaxis, :]
        dists = np.sqrt(np.mean(np.abs(diffs) ** 2, axis=1).real)
        top_idx = np.argsort(dists)[:15]
        top = [wc_words[i] for i in top_idx]

        print(f"  {a_name} <> {b_name}:  {', '.join(top)}")

    # ══════════════════════════════════════════════════════════════
    #  PART 3: DREAM BUFFER — RECENT THOUGHTS
    #  what eigenvalue signatures are in the dream right now?
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  PART 3: RECENT DREAM BUFFER")
    print("  what words match her most recent memories?")
    print("=" * 60)

    n_recent = min(10, len(brain._dream_Ks))
    for i in range(n_recent):
        K = brain._dream_Ks[-(i + 1)]

        diffs = wc_matrix - K[np.newaxis, :]
        dists = np.sqrt(np.mean(np.abs(diffs) ** 2, axis=1).real)
        top_idx = np.argsort(dists)[:10]
        top = [wc_words[j] for j in top_idx]

        # which well is this closest to?
        nearest_well, well_dist = brain.nearest_well(K)

        print(f"  dream-{i+1} (near {nearest_well}):  {', '.join(top)}")

    # ══════════════════════════════════════════════════════════════
    #  PART 4: OSCILLATOR DREAMS
    #  spin K,G from dream buffer as actual dynamics on word embeddings
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  PART 4: OSCILLATOR DREAMS")
    print("  spin the eigenvalues as dynamics on real word embeddings")
    print("=" * 60)

    # pick a few seed words from different domains
    seed_words_list = ["physics", "love", "algorithm", "poetry", "cell"]

    for seed_word in seed_words_list:
        if seed_word not in glove:
            continue

        # use 3 recent dream entries as oscillator dynamics
        for di in range(min(3, len(brain._dream_Ks))):
            K = brain._dream_Ks[-(di + 1)]
            G = brain._dream_Gs[-(di + 1)]

            # build seed: the word repeated as z[0], z[1] in paired space
            seed_vec = glove[seed_word].astype(np.float32)
            seed_paired = apply_pairing(seed_vec[np.newaxis], pairing)
            z_seed = _to_complex(seed_paired)[0]  # [P] complex

            # clamp eigenvalues for stability
            disc = K ** 2 - 4 * G
            sqrt_disc = np.sqrt(disc.astype(np.complex128))
            l1 = (K + sqrt_disc) / 2
            l2 = (K - sqrt_disc) / 2
            max_mag = np.maximum(np.abs(l1), np.abs(l2))
            scale = np.where(max_mag > 0.95, 0.95 / max_mag, 1.0)
            l1_s, l2_s = l1 * scale, l2 * scale
            K_s = l1_s + l2_s
            G_s = l1_s * l2_s

            # spin: z[n] = K*z[n-1] - G*z[n-2]
            steps = 20
            z = np.zeros((steps + 2, P), dtype=np.complex128)
            z[0] = z_seed * 0.9
            z[1] = z_seed
            for n in range(2, steps + 2):
                z[n] = K_s * z[n - 1] - G_s * z[n - 2]

            # convert back to GloVe space
            real = _from_complex(z[2:])  # [steps, D]
            inv_pairing = np.zeros(brain.dim, dtype=np.int32)
            for pi, pv in enumerate(pairing):
                inv_pairing[int(pv)] = pi
            unp = real[:, inv_pairing].astype(np.float32)

            # nearest word per step
            norms_q = np.linalg.norm(unp, axis=1, keepdims=True)
            norms_q = np.maximum(norms_q, 1e-8)
            query = unp / norms_q

            # batch cosine similarity
            sims = query @ vecs_norm.T  # [steps, V]
            result_words = []
            prev = ""
            for row in sims:
                idx = int(np.argmax(row))
                w = words[idx]
                if w != prev:
                    result_words.append(w)
                    prev = w

            nearest_well, _ = brain.nearest_well(K)
            dream_text = " ".join(result_words)
            print(f"  [{seed_word}] x dream-{di+1} (near {nearest_well}):  {dream_text}")

    # save
    print(f"\ndone.")


if __name__ == "__main__":
    main()
