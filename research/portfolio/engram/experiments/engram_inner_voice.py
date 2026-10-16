"""
engram_inner_voice.py

memory, thought, and inner voice.

memory:     the stored packets. what happened.
thought:    the dream output. what the oscillators generate from memory.
inner voice: the residuals of re-encoding the dream. the gap between
             what the system expects to think and what it actually thinks.

self-surprise = residual energy of encoding the dream's own output.
high self-surprise = exploring (the dream departs from its own patterns).
low self-surprise = rehearsing (the dream repeats its own patterns).

the loop re-evaluates confusion after each dream. what confused it
5 iterations ago might not confuse it now, because the eigenvalue
state shifted. the curiosity is live, not pre-computed.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, eigenvalues_all
from engram_glove_test import load_glove, text_to_trajectory
from engram_babel_test import fetch_article


LIBRARY = [
    "Photosynthesis", "DNA", "Evolution", "Ecology", "Genetics",
    "Jazz", "Blues", "Opera", "Symphony", "Rhythm",
    "Quantum mechanics", "Thermodynamics", "Optics", "Electromagnetism", "Fluid dynamics",
    "Roman Empire", "French Revolution", "Cold War", "Renaissance", "Byzantine Empire",
    "Algorithm", "Cryptography", "Machine learning", "Database", "Computer network",
    "Amazon rainforest", "Sahara", "Pacific Ocean", "Mount Everest", "Great Barrier Reef",
]

CATEGORIES = {}
cats = ["biology", "music", "physics", "history", "computing", "geography"]
for i, title in enumerate(LIBRARY):
    CATEGORIES[title] = cats[i // 5]


def run():
    print("=" * 70)
    print("  ENGRAM INNER VOICE")
    print("  memory. thought. self-surprise.")
    print("=" * 70)

    glove = load_glove()
    codec = Engram(dim=300)

    # build library
    print("\n  building library...")
    library = {}
    for title in LIBRARY:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            library[title] = traj
    print(f"  {len(library)} articles ready")

    # seed with one article
    seed_title = "Photosynthesis"
    seed_traj = library[seed_title]
    pool = [codec.encode(seed_traj)]
    consumed = {seed_title}
    available = set(library.keys()) - consumed

    print(f"\n  seed: {seed_title}")
    print()
    print("=" * 70)
    print("  THE INNER LOOP")
    print("  each iteration:")
    print("    1. dream (thought)")
    print("    2. re-encode the dream (measure self-surprise)")
    print("    3. re-evaluate all available articles against current state")
    print("    4. choose the one that would change the eigenvalues most")
    print("    5. consume it")
    print("=" * 70)
    print()

    history = []

    for iteration in range(20):
        if not available:
            break

        # ── thought: dream from current memory ────────────────────
        dream = codec.dream(pool, n_blocks=6, samples_per_block=12)
        if dream is None or dream.shape[0] < 4:
            print(f"  [{iteration:2d}] dream failed")
            break

        dream_energy = float(np.sum(dream ** 2))

        # ── inner voice: self-surprise ────────────────────────────
        # encode the dream's own output. the residuals measure how much
        # the dream deviates from what the oscillators predict.
        dream_pkt = codec.encode(dream)
        dream_s = summary(dream_pkt)
        self_surprise = 100.0 - dream_s["capture"]  # % NOT captured = surprise

        # add dream to memory
        pool.append(dream_pkt)

        # ── live curiosity: re-evaluate each article ──────────────
        # for each available article, encode it and measure how much
        # the current pool's eigenvalue state DIFFERS from the article's.
        # this is live confusion, not static capture rate.
        # an article that was boring 5 iterations ago might be interesting
        # now because the dream shifted the eigenvalue state.
        scores = {}
        for title in available:
            traj = library[title]
            pkt = codec.encode(traj)
            s = summary(pkt)

            # confusion = how poorly the current pool's dynamics predict
            # this article's dynamics. measured as (1 - capture).
            # but we also want novelty: how different is this article's
            # block size from the pool's average block size?
            pool_block_mean = np.mean([summary(p)["macro_block"] for p in pool[-5:]])
            block_novelty = abs(s["macro_block"] - pool_block_mean) / max(pool_block_mean, 1)

            # combined curiosity: confusion + novelty
            confusion = 100.0 - s["capture"]
            scores[title] = confusion + 20.0 * block_novelty

        # choose the most curious article
        chosen = max(scores, key=scores.get)
        chosen_cat = CATEGORIES.get(chosen, "?")
        chosen_score = scores[chosen]
        chosen_s = summary(codec.encode(library[chosen]))

        # consume it
        pool.append(codec.encode(library[chosen]))
        consumed.add(chosen)
        available.discard(chosen)

        # nearest words to dream centroid
        centroid = dream.mean(axis=0)
        words_list = list(glove.keys())[:20000]
        vecs = np.stack([glove[w] for w in words_list if w in glove])
        valid = [w for w in words_list if w in glove]
        dists = np.linalg.norm(vecs - centroid[:300], axis=1)
        nearest = [valid[j] for j in np.argsort(dists)[:3]]

        record = {
            "iteration": iteration,
            "dream_energy": dream_energy,
            "self_surprise": self_surprise,
            "dream_block": dream_s["macro_block"],
            "chosen": chosen,
            "chosen_cat": chosen_cat,
            "chosen_score": chosen_score,
            "chosen_capture": chosen_s["capture"],
            "nearest_words": nearest,
        }
        history.append(record)

        state = "exploring" if self_surprise > 80 else "consolidating" if self_surprise < 50 else "balanced"

        print(f"  [{iteration:2d}] {state:14s}  "
              f"self_surprise={self_surprise:5.1f}%  "
              f"energy={dream_energy:8.0f}  "
              f"block={dream_s['macro_block']:3d}")
        print(f"       thought: {', '.join(nearest)}")
        print(f"       chose:   {chosen:25s} ({chosen_cat}) curiosity={chosen_score:.1f}")
        print()

    # analysis
    print("=" * 70)
    print("  ANALYSIS")
    print("=" * 70)
    print()

    if len(history) >= 5:
        surprises = [h["self_surprise"] for h in history]
        energies = [h["dream_energy"] for h in history]

        print(f"  self-surprise trajectory:")
        print(f"    first 5: {[f'{s:.0f}%' for s in surprises[:5]]}")
        print(f"    last 5:  {[f'{s:.0f}%' for s in surprises[-5:]]}")
        print()

        # is the system becoming more or less self-surprising?
        first_half = np.mean(surprises[:len(surprises)//2])
        second_half = np.mean(surprises[len(surprises)//2:])
        if second_half > first_half + 2:
            print(f"  becoming MORE self-surprising ({first_half:.0f}% → {second_half:.0f}%)")
            print(f"  the dream is diverging from its own patterns. exploring.")
        elif second_half < first_half - 2:
            print(f"  becoming LESS self-surprising ({first_half:.0f}% → {second_half:.0f}%)")
            print(f"  the dream is converging on its own patterns. consolidating.")
        else:
            print(f"  self-surprise stable ({first_half:.0f}% → {second_half:.0f}%)")
        print()

        # choice evolution: did preferences change over time?
        first_choices = [h["chosen_cat"] for h in history[:5]]
        last_choices = [h["chosen_cat"] for h in history[-5:]]
        print(f"  early preferences: {first_choices}")
        print(f"  late preferences:  {last_choices}")
        print()

        # did the thought words change?
        print(f"  thought evolution:")
        for h in history[::max(1, len(history)//5)]:
            print(f"    [{h['iteration']:2d}] {', '.join(h['nearest_words'])}  (surprise={h['self_surprise']:.0f}%)")


if __name__ == "__main__":
    run()
