"""
engram_loop.py

the closed loop. dream generates. the generation is encoded.
the encoding feeds back into dream. repeat.

what happens when the codec observes its own output?
do the eigenvalues converge to a fixed point?
diverge to noise? oscillate between attractors?
find something nobody put there?

no hypothesis. just observation.
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, spectral_summary, eigenvalues_all
from engram_glove_test import load_glove, text_to_trajectory, tokenize_simple
from engram_babel_test import fetch_article


def nearest_words(vec: np.ndarray, glove: dict, top_k: int = 3, vocab_subset: list = None) -> list[tuple[str, float]]:
    """find nearest GloVe words to a vector."""
    words = vocab_subset or list(glove.keys())[:50000]
    vecs = np.stack([glove[w] for w in words if w in glove])
    valid_words = [w for w in words if w in glove]
    dists = np.linalg.norm(vecs - vec[:300], axis=1)
    idx = np.argsort(dists)[:top_k]
    return [(valid_words[i], float(dists[i])) for i in idx]


def run():
    print("=" * 70)
    print("  ENGRAM CLOSED LOOP")
    print("  dream → encode → dream → encode → ...")
    print("  watching what happens.")
    print("=" * 70)
    print()

    glove = load_glove()
    codec = Engram(dim=300)

    # seed: encode 10 Wikipedia articles as the initial memory pool
    print("--- seeding with 10 Wikipedia articles ---\n")
    seed_titles = [
        "Photosynthesis", "Jazz", "Chess", "Volcano", "DNA",
        "Roman Empire", "Violin", "General relativity",
        "Python (programming language)", "Moon",
    ]

    packets = []
    for title in seed_titles:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            pkt = codec.encode(traj)
            packets.append(pkt)
            s = summary(pkt)
            print(f"  {title:30s}: {traj.shape[0]:6d} words, block={s['macro_block']}, capture={s['capture']}%")

    print(f"\n  pool: {len(packets)} packets")
    print()

    # the loop
    print("--- the loop ---\n")
    n_iterations = 12
    history = []

    # track a vocabulary of common words to look up nearest neighbors
    common_words = [w for w in list(glove.keys())[:30000]]

    for iteration in range(n_iterations):
        # dream: generate a novel trajectory from the current pool
        gen = codec.dream(packets, n_blocks=6, samples_per_block=12)

        if gen is None or gen.shape[0] < 4:
            print(f"  iteration {iteration}: dream returned nothing. stopping.")
            break

        # encode the dream's output
        dream_pkt = codec.encode(gen)
        dream_s = summary(dream_pkt)

        # spectral analysis of the dream
        all_mags, all_freqs = [], []
        for blk in dream_pkt.blocks:
            if blk.macro_K is not None and len(blk.macro_K) > 0:
                l1, l2 = eigenvalues_all(blk.macro_K, blk.macro_G)
                all_mags.extend(np.abs(l1).tolist())
                all_mags.extend(np.abs(l2).tolist())
                all_freqs.extend(np.abs(np.angle(l1)).tolist())
                all_freqs.extend(np.abs(np.angle(l2)).tolist())

        mag_mean = float(np.mean(all_mags)) if all_mags else 0.0
        freq_mean = float(np.mean(all_freqs)) if all_freqs else 0.0
        energy = float(np.sum(gen ** 2))

        # find nearest words to the dream's centroid and a few samples
        centroid = gen.mean(axis=0)
        nearest = nearest_words(centroid, glove, top_k=5, vocab_subset=common_words)
        word_str = ", ".join(f"{w}" for w, d in nearest)

        # sample a few individual vectors
        sample_words = []
        for si in [0, gen.shape[0] // 2, gen.shape[0] - 1]:
            nw = nearest_words(gen[si], glove, top_k=1, vocab_subset=common_words)
            if nw:
                sample_words.append(nw[0][0])

        record = {
            "iteration": iteration,
            "samples": gen.shape[0],
            "blocks": dream_s["blocks"],
            "macro_block": dream_s["macro_block"],
            "capture": dream_s["capture"],
            "energy": energy,
            "mag_mean": mag_mean,
            "freq_mean": freq_mean,
            "centroid_words": [w for w, _ in nearest],
            "sample_words": sample_words,
        }
        history.append(record)

        print(f"  [{iteration:2d}] {gen.shape[0]:3d} samples, block={dream_s['macro_block']:3d}, "
              f"capture={dream_s['capture']:5.1f}%, energy={energy:8.2f}, "
              f"|λ|={mag_mean:.4f}, freq={freq_mean:.3f}")
        print(f"       centroid: {word_str}")
        print(f"       samples: {' → '.join(sample_words)}")

        # feed the dream back into the pool
        packets.append(dream_pkt)

    print()

    # analysis: did the eigenvalues converge, diverge, or oscillate?
    print("=" * 70)
    print("  LOOP ANALYSIS")
    print("=" * 70)
    print()

    if len(history) >= 3:
        energies = [h["energy"] for h in history]
        mags = [h["mag_mean"] for h in history]
        freqs = [h["freq_mean"] for h in history]
        captures = [h["capture"] for h in history]
        blocks = [h["macro_block"] for h in history]

        print(f"  energy:   {energies[0]:.2f} → {energies[-1]:.2f} (ratio: {energies[-1] / max(energies[0], 1e-10):.3f})")
        print(f"  |lambda|: {mags[0]:.4f} → {mags[-1]:.4f}")
        print(f"  freq:     {freqs[0]:.3f} → {freqs[-1]:.3f}")
        print(f"  capture:  {captures[0]:.1f}% → {captures[-1]:.1f}%")
        print(f"  block:    {blocks[0]} → {blocks[-1]}")
        print()

        # convergence test: is the last third more stable than the first third?
        n = len(history)
        third = max(1, n // 3)
        first_var = np.std(energies[:third])
        last_var = np.std(energies[-third:])
        print(f"  energy variance: first third={first_var:.4f}, last third={last_var:.4f}")

        if last_var < first_var * 0.5:
            print(f"  converging. the eigenvalues found a basin.")
        elif last_var > first_var * 2.0:
            print(f"  diverging. the meta-oscillator is unstable.")
        else:
            print(f"  neither converging nor diverging. wandering.")

        # word drift: do the centroid words change or stabilize?
        print()
        print(f"  centroid word evolution:")
        for h in history:
            print(f"    [{h['iteration']:2d}] {', '.join(h['centroid_words'][:3])}")


if __name__ == "__main__":
    run()
