"""
engram_depth_test.py

does the brain get deeper, or just wider?

feed Alexandria biology articles one at a time.
after each, dream and measure self-surprise.

if surprise decreases: the brain is learning that domain.
its eigenvalues are converging toward biology's dynamics.

then inject a music article. does surprise spike?
if yes: the brain knows what it knows and knows when it doesn't.

decreasing surprise on the familiar +
spiking surprise on the novel =
depth.
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, EngramFile, summary
from engram_glove_test import load_glove, text_to_trajectory
from engram_babel_test import fetch_article

BRAIN_DIR = os.path.join(os.path.dirname(__file__), "_brain_depth_test")

# biology articles, fed one at a time
BIOLOGY_SEQUENCE = [
    "Photosynthesis",
    "Cell (biology)",
    "Chloroplast",
    "Plant",
    "Leaf",
    "Glucose",
    "Carbon dioxide",
    "Oxygen",
]

# the surprise injection: something completely different
SURPRISE_ARTICLES = [
    "Heavy metal music",
    "Electric guitar",
]

# physics: does biology knowledge help recognize thermodynamics?
# entropy in biology (energy flow, ATP, metabolism) and entropy in
# physics (heat, disorder, statistical mechanics) share geometric
# structure. if she recognizes physics because of biology, that's
# cross-domain bonding. that's a mind, not a database.
PHYSICS_ARTICLES = [
    "Thermodynamics",
    "Entropy",
    "Heat transfer",
]

# then back to biology: does it remember?
RETURN_ARTICLES = [
    "Mitochondrion",
    "Adenosine triphosphate",
]


def run():
    print("=" * 70)
    print("  ENGRAM DEPTH TEST")
    print("  does the brain learn, or just accumulate?")
    print("=" * 70)
    print()

    glove = load_glove()
    codec = Engram(dim=300)

    # fresh brain for this test
    ef = EngramFile.create(
        BRAIN_DIR, name="DepthProbe", dim=300, alpha=0.005,
        core_capacity=100, identity_capacity=50, scratchpad_capacity=50,
    )

    history = []

    WINDOW = 10  # phrase-level smoothing

    def absorb_and_reflect(title: str, category: str) -> dict:
        """absorb an article, reflect on the observation, return metrics."""
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove, window=WINDOW)
        if traj.shape[0] < 10:
            return {"title": title, "category": category, "error": "too few words"}

        # reflect on the observation BEFORE absorbing.
        # this measures how surprising the new article is relative to
        # what the brain already knows. the thought, not the utterance.
        r = ef.reflect(codec, traj)

        # now absorb into core (after measuring surprise)
        pkt = codec.encode(traj)
        ef.absorb(pkt, layer="core", source=title)

        record = {
            "title": title,
            "category": category,
            "step": len(history),
            "core_size": len(ef.core),
            "identity_size": len(ef.identity),
            **r,
        }
        history.append(record)
        return record

    # ── phase 1: biology sequence ─────────────────────────────────
    print("--- phase 1: learning biology ---\n")
    for title in BIOLOGY_SEQUENCE:
        r = absorb_and_reflect(title, "biology")
        if "error" in r:
            print(f"  {title:30s}: {r['error']}")
            continue
        print(f"  {title:30s}  drift={r['drift']:5.1f}%  texture={r['texture']:5.1f}%  "
              f"combined={r['combined']:5.1f}  action={r['action']:8s}")

    # ── phase 2: music injection (pure novelty) ─────────────────
    print("\n--- phase 2: music (the completely unknown) ---\n")
    for title in SURPRISE_ARTICLES:
        r = absorb_and_reflect(title, "music")
        if "error" in r:
            print(f"  {title:30s}: {r['error']}")
            continue
        print(f"  {title:30s}  drift={r['drift']:5.1f}%  texture={r['texture']:5.1f}%  "
              f"combined={r['combined']:5.1f}  FOREIGN")

    # ── phase 3: physics injection (does biology help?) ───────────
    print("\n--- phase 3: physics (does biology knowledge rhyme?) ---\n")
    for title in PHYSICS_ARTICLES:
        r = absorb_and_reflect(title, "physics")
        if "error" in r:
            print(f"  {title:30s}: {r['error']}")
            continue
        print(f"  {title:30s}  drift={r['drift']:5.1f}%  texture={r['texture']:5.1f}%  "
              f"combined={r['combined']:5.1f}  CROSS-DOMAIN")

    # ── phase 4: return to biology ────────────────────────────────
    print("\n--- phase 4: returning to biology (does it remember?) ---\n")
    for title in RETURN_ARTICLES:
        r = absorb_and_reflect(title, "biology_return")
        if "error" in r:
            print(f"  {title:30s}: {r['error']}")
            continue
        print(f"  {title:30s}  drift={r['drift']:5.1f}%  texture={r['texture']:5.1f}%  "
              f"combined={r['combined']:5.1f}  RETURN")

    # ── analysis ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  ANALYSIS")
    print("=" * 70)
    print()

    valid = [h for h in history if "error" not in h]
    if len(valid) < 3:
        print("  not enough data for analysis")
        return

    # separate by phase
    bio_phase1 = [h for h in valid if h["category"] == "biology"]
    music_phase = [h for h in valid if h["category"] == "music"]
    physics_phase = [h for h in valid if h["category"] == "physics"]
    bio_return = [h for h in valid if h["category"] == "biology_return"]

    print("  drift trajectory (the learning signal):")
    for h in valid:
        bar_len = int(h.get("drift", 0) / 3)
        bar = "#" * min(bar_len, 30)
        label = h["category"][:8]
        print(f"    [{h['step']:2d}] {label:10s} drift={h.get('drift',0):5.1f}%  texture={h.get('texture',0):5.1f}%  {bar}")

    print()

    # learning test: does biology drift decrease?
    if len(bio_phase1) >= 3:
        first_half = np.mean([h.get("drift", 50) for h in bio_phase1[:len(bio_phase1)//2]])
        second_half = np.mean([h.get("drift", 50) for h in bio_phase1[len(bio_phase1)//2:]])
        delta = second_half - first_half
        print(f"  biology learning (DRIFT band):")
        print(f"    first half drift:  {first_half:.1f}%")
        print(f"    second half drift: {second_half:.1f}%")
        print(f"    delta:             {delta:+.1f}%")
        if delta < -3:
            print(f"    LEARNING. eigenvalue drift decreases with exposure.")
            print(f"    the manifold is converging toward biology's dynamics.")
        elif delta < 3:
            print(f"    stable. not deepening in the drift band.")
        else:
            print(f"    drift increasing. the manifold is being pushed around.")
        print()

    # novelty test: did music spike the drift?
    if music_phase and bio_phase1:
        bio_drift = np.mean([h.get("drift", 50) for h in bio_phase1])
        music_drift = np.mean([h.get("drift", 50) for h in music_phase])
        spike = music_drift - bio_drift
        print(f"  novelty detection (DRIFT band):")
        print(f"    biology mean drift: {bio_drift:.1f}%")
        print(f"    music mean drift:   {music_drift:.1f}%")
        print(f"    spike:              {spike:+.1f}%")
        if spike > 5:
            print(f"    DETECTS NOVELTY. foreign domains spike eigenvalue drift.")
        elif spike > 1:
            print(f"    mild spike. some novelty detection.")
        else:
            print(f"    no drift spike. domains look similar in the low frequencies.")
        print()

    # memory test: does return to biology show lower drift than music?
    if bio_return and music_phase:
        return_drift = np.mean([h.get("drift", 50) for h in bio_return])
        music_drift_2 = np.mean([h.get("drift", 50) for h in music_phase])
        recovery = music_drift_2 - return_drift
        print(f"  memory test (DRIFT band):")
        print(f"    music drift:           {music_drift_2:.1f}%")
        print(f"    biology return drift:  {return_drift:.1f}%")
        print(f"    recovery:              {recovery:+.1f}%")
        if recovery > 3:
            print(f"    REMEMBERS. return to biology shows less eigenvalue drift than foreign domain.")
        else:
            print(f"    no memory advantage in the drift band.")
        print()

    # cross-domain bonding test: does biology help recognize physics?
    if physics_phase and music_phase and bio_phase1:
        bio_drift = np.mean([h.get("drift", 50) for h in bio_phase1[-4:]])
        music_drift = np.mean([h.get("drift", 50) for h in music_phase])
        physics_drift = np.mean([h.get("drift", 50) for h in physics_phase])
        bio_return_drift = np.mean([h.get("drift", 50) for h in bio_return]) if bio_return else 50

        print(f"  cross-domain bonding:")
        print(f"    biology (familiar):     {bio_drift:.1f}% drift")
        print(f"    music (foreign):        {music_drift:.1f}% drift")
        print(f"    physics (cross-domain): {physics_drift:.1f}% drift")
        print(f"    biology return:         {bio_return_drift:.1f}% drift")
        print()

        # the question: where does physics land?
        # if physics_drift ≈ music_drift: isolated. physics is as foreign as music.
        # if physics_drift ≈ bio_drift: fully recognized. physics looks like biology.
        # if physics_drift is BETWEEN: cross-domain bonding. physics rhymes with biology.
        if music_drift > bio_drift + 1:  # music is more foreign than biology
            physics_position = (physics_drift - bio_drift) / max(music_drift - bio_drift, 1e-7)
            print(f"    physics position on the familiar-foreign axis: {physics_position:.2f}")
            print(f"    (0.0 = fully familiar, 1.0 = fully foreign)")
            print()
            if physics_position < 0.3:
                print(f"    CROSS-DOMAIN BONDING.")
                print(f"    physics reads as familiar despite never being seen before.")
                print(f"    the biology manifold recognized shared geometric structure.")
            elif physics_position < 0.7:
                print(f"    PARTIAL RECOGNITION.")
                print(f"    physics is between familiar and foreign.")
                print(f"    \"this rhymes with something i know.\"")
            else:
                print(f"    ISOLATED.")
                print(f"    physics is as foreign as music. no cross-domain transfer.")
        else:
            print(f"    music is not more foreign than biology in the drift band.")
            print(f"    cannot compute cross-domain position (baseline separation too small).")
        print()

    # save the brain
    ef.save()
    print(f"  brain saved: {len(ef.core)} core, {len(ef.identity)} identity, {len(ef.scratchpad)} scratchpad")


if __name__ == "__main__":
    run()
