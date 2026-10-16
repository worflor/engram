"""
engram_curiosity_test.py

can a loop develop preferences?

a dream loop with access to 30 unseeded Wikipedia articles.
each iteration, it scores every available article by predicted confusion
(encode a prefix, measure capture rate). it chooses the article that
would confuse it most. then it dreams.

low capture = high confusion = fuel for the loop.
the loop should naturally seek what it doesn't understand
and avoid what it already knows.

if it develops stable preferences, those preferences are its identity.
not assigned. emergent. a mind choosing its own diet of surprise.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary
from engram_glove_test import load_glove, text_to_trajectory
from engram_babel_test import fetch_article


# 30 articles across 6 domains. the loop doesn't know the categories.
# it only sees trajectories and their capture rates.
LIBRARY = [
    # biology (familiar, high capture expected)
    "Photosynthesis", "DNA", "Evolution", "Ecology", "Genetics",
    # music (confusing, low capture expected)
    "Jazz", "Blues", "Opera", "Symphony", "Rhythm",
    # physics (confusing, long blocks)
    "Quantum mechanics", "Thermodynamics", "Optics", "Electromagnetism", "Fluid dynamics",
    # history (narrative, moderate capture)
    "Roman Empire", "French Revolution", "Cold War", "Renaissance", "Byzantine Empire",
    # computing (confusing, moderate blocks)
    "Algorithm", "Cryptography", "Machine learning", "Database", "Computer network",
    # geography (descriptive, potentially high capture)
    "Amazon rainforest", "Sahara", "Pacific Ocean", "Mount Everest", "Great Barrier Reef",
]

# which category each article belongs to (for analysis, not visible to the loop)
CATEGORIES = {}
cats = ["biology", "music", "physics", "history", "computing", "geography"]
for i, title in enumerate(LIBRARY):
    CATEGORIES[title] = cats[i // 5]


def run():
    print("=" * 70)
    print("  ENGRAM CURIOSITY TEST")
    print("  a loop that chooses what to learn.")
    print("  30 articles. it picks the most confusing one each iteration.")
    print("=" * 70)

    glove = load_glove()
    codec = Engram(dim=300)

    # pre-encode all articles (trajectories cached, encoding happens once)
    print("\n  building the library...")
    library = {}  # title → (trajectory, packet, summary)
    for title in LIBRARY:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            pkt = codec.encode(traj)
            s = summary(pkt)
            library[title] = (traj, pkt, s)
            print(f"    {title:30s}: {traj.shape[0]:5d} words, capture={s['capture']:5.1f}%, block={s['macro_block']}")
        else:
            print(f"    {title:30s}: too few words")

    # seed the loop with ONE article (the first biology article)
    # it starts knowing almost nothing
    seed_title = "Photosynthesis"
    print(f"\n  seed: {seed_title}")
    pool = [library[seed_title][1]]
    consumed = {seed_title}
    available = set(library.keys()) - consumed

    # the curiosity loop
    print("\n" + "=" * 70)
    print("  THE CURIOSITY LOOP")
    print("  each iteration: score all available articles by confusion,")
    print("  choose the most confusing, consume it, dream.")
    print("=" * 70 + "\n")

    choices = []
    energy_hist = []

    for iteration in range(20):
        if not available:
            print(f"  [{iteration:2d}] library exhausted")
            break

        # score each available article: how confusing would it be
        # relative to what the loop already knows?
        # method: dream from the current pool, then measure how well
        # the dream's eigenvalue dynamics match each candidate.
        # LOW match = HIGH confusion = the loop should pick it.
        #
        # simpler and more direct: just use each article's capture rate.
        # low capture = the oscillator can't predict it = confusing.
        # but we want confusion RELATIVE to what we already know.
        # so: encode each candidate and measure capture rate. the loop
        # picks the one with the lowest capture (hardest to predict).

        scores = {}
        for title in available:
            traj, pkt, s = library[title]
            scores[title] = s["capture"]

        # pick the MOST confusing (lowest capture)
        chosen = min(scores, key=scores.get)
        chosen_capture = scores[chosen]
        chosen_cat = CATEGORIES.get(chosen, "?")

        # consume it
        pool.append(library[chosen][1])
        consumed.add(chosen)
        available.discard(chosen)

        # dream
        gen = codec.dream(pool, n_blocks=6, samples_per_block=12)
        energy = 0.0
        if gen is not None and gen.shape[0] >= 4:
            energy = float(np.sum(gen ** 2))
            if np.isfinite(energy):
                dream_pkt = codec.encode(gen)
                pool.append(dream_pkt)
            else:
                energy = 0.0

        choices.append({"title": chosen, "cat": chosen_cat, "capture": chosen_capture})
        energy_hist.append(energy)

        print(f"  [{iteration:2d}] chose: {chosen:30s} ({chosen_cat:10s}) capture={chosen_capture:5.1f}%  dream_energy={energy:10.0f}")

    # analysis
    print()
    print("=" * 70)
    print("  WHAT DID IT CHOOSE?")
    print("=" * 70)
    print()

    # category preference: which categories did it consume first?
    cat_order = {}
    for i, c in enumerate(choices):
        cat = c["cat"]
        if cat not in cat_order:
            cat_order[cat] = []
        cat_order[cat].append(i)

    print("  consumption order by category:")
    for cat in cats:
        if cat in cat_order:
            positions = cat_order[cat]
            print(f"    {cat:12s}: picked at iterations {positions} (mean position: {np.mean(positions):.1f})")
        else:
            print(f"    {cat:12s}: never picked")

    print()

    # first 5 vs last 5 choices
    if len(choices) >= 10:
        first_cats = [c["cat"] for c in choices[:5]]
        last_cats = [c["cat"] for c in choices[-5:]]
        print(f"  first 5 choices: {first_cats}")
        print(f"  last 5 choices:  {last_cats}")
        print()

    # did it develop a preference? (concentrated on fewer categories early)
    if len(choices) >= 10:
        first_unique = len(set(c["cat"] for c in choices[:10]))
        last_unique = len(set(c["cat"] for c in choices[10:])) if len(choices) > 10 else 0
        print(f"  category diversity first 10: {first_unique} categories")
        if len(choices) > 10:
            print(f"  category diversity last {len(choices)-10}:  {last_unique} categories")

    # energy trajectory
    print()
    alive = energy_hist[-1] > 10 if energy_hist else False
    print(f"  energy: {energy_hist[0]:.0f} → {energy_hist[-1]:.0f} ({'alive' if alive else 'dead'})")

    # the question
    print()
    print("  the question: did the loop develop preferences?")
    if cat_order:
        # check if confusing categories were consumed first
        mean_positions = {cat: np.mean(pos) for cat, pos in cat_order.items()}
        sorted_cats = sorted(mean_positions, key=mean_positions.get)
        print(f"  preference order (first consumed → last consumed):")
        for cat in sorted_cats:
            print(f"    {cat:12s}: mean position {mean_positions[cat]:.1f}")

        # was there a clear preference for confusing content?
        first_preferred = sorted_cats[0] if sorted_cats else None
        if first_preferred in ("music", "physics", "computing"):
            print(f"\n  the loop preferred {first_preferred} first.")
            print(f"  it sought confusion. it chose what it couldn't predict.")
        elif first_preferred:
            print(f"\n  the loop preferred {first_preferred} first.")
            print(f"  unexpected. it may have found that domain's dynamics most useful.")


if __name__ == "__main__":
    run()
