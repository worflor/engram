"""
engram_mutualism_test.py

two confused minds. similar capture rates. similar residual energy.
can they sustain each other?

music: capture ~13.8%, block ~36
computing: capture ~17.6%, block ~31

both confused. both generating substantial residuals.
the hypothesis: mutualism. both survive. neither dominates.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, eigenvalues_all
from engram_glove_test import load_glove, text_to_trajectory
from engram_babel_test import fetch_article


def run():
    print("=" * 70)
    print("  ENGRAM MUTUALISM TEST")
    print("  two confused minds. can they sustain each other?")
    print("=" * 70)

    glove = load_glove()
    codec = Engram(dim=300)

    # seed both pools with similarly confused categories
    print("\n  seeding...")
    pool_A = []
    for title in ["Jazz", "Blues", "Classical music", "Opera", "Symphony"]:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            pkt = codec.encode(traj)
            pool_A.append(pkt)
            s = summary(pkt)
            print(f"    A (music)    {title:25s}: capture={s['capture']:5.1f}%  block={s['macro_block']}")

    pool_B = []
    for title in ["Algorithm", "Computer science", "Artificial intelligence", "Machine learning", "Cryptography"]:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            pkt = codec.encode(traj)
            pool_B.append(pkt)
            s = summary(pkt)
            print(f"    B (computing) {title:25s}: capture={s['capture']:5.1f}%  block={s['macro_block']}")

    # isolated controls
    print("\n  isolated controls (20 iterations, no exchange)...")
    iso_A, iso_B = list(pool_A), list(pool_B)
    iso_eA, iso_eB = [], []
    for i in range(20):
        gA = codec.dream(iso_A, n_blocks=6, samples_per_block=12)
        gB = codec.dream(iso_B, n_blocks=6, samples_per_block=12)
        eA = float(np.sum(gA ** 2)) if gA is not None and gA.shape[0] >= 4 else 0.0
        eB = float(np.sum(gB ** 2)) if gB is not None and gB.shape[0] >= 4 else 0.0
        iso_eA.append(eA)
        iso_eB.append(eB)
        if gA is not None and gA.shape[0] >= 4: iso_A.append(codec.encode(gA))
        if gB is not None and gB.shape[0] >= 4: iso_B.append(codec.encode(gB))
    print(f"    A isolated: {iso_eA[0]:.0f} → {iso_eA[-1]:.0f}")
    print(f"    B isolated: {iso_eB[0]:.0f} → {iso_eB[-1]:.0f}")

    # the mutualism
    print("\n" + "=" * 70)
    print("  THE MUTUALISM")
    print("=" * 70 + "\n")

    hist_A, hist_B, sims = [], [], []

    for i in range(20):
        gA = codec.dream(pool_A, n_blocks=6, samples_per_block=12)
        gB = codec.dream(pool_B, n_blocks=6, samples_per_block=12)

        if gA is None or gA.shape[0] < 4 or gB is None or gB.shape[0] < 4:
            print(f"  [{i:2d}] dream failed")
            break

        pA, pB = codec.encode(gA), codec.encode(gB)
        sA, sB = summary(pA), summary(pB)
        eA, eB = float(np.sum(gA ** 2)), float(np.sum(gB ** 2))

        hist_A.append({"e": eA, "b": sA["macro_block"], "c": sA["capture"]})
        hist_B.append({"e": eB, "b": sB["macro_block"], "c": sB["capture"]})

        print(f"  [{i:2d}] A: energy={eA:10.0f} block={sA['macro_block']:3d}  |  "
              f"B: energy={eB:10.0f} block={sB['macro_block']:3d}")

        # exchange: add each other's dreams to the other's pool
        pool_A.append(pB)
        pool_B.append(pA)
        pool_A.append(pA)
        pool_B.append(pB)

    # verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    if len(hist_A) >= 4:
        eA_list = [h["e"] for h in hist_A]
        eB_list = [h["e"] for h in hist_B]

        a_alive = eA_list[-1] > 10
        b_alive = eB_list[-1] > 10
        iso_a_alive = iso_eA[-1] > 10
        iso_b_alive = iso_eB[-1] > 10

        print(f"  survival:")
        print(f"    A isolated:  {'alive' if iso_a_alive else 'dead':5s} (energy {iso_eA[-1]:.0f})")
        print(f"    A symbiotic: {'alive' if a_alive else 'dead':5s} (energy {eA_list[-1]:.0f})")
        print(f"    B isolated:  {'alive' if iso_b_alive else 'dead':5s} (energy {iso_eB[-1]:.0f})")
        print(f"    B symbiotic: {'alive' if b_alive else 'dead':5s} (energy {eB_list[-1]:.0f})")
        print()

        # energy ratio: how balanced is the relationship?
        ratios = [max(a, 1) / max(b, 1) for a, b in zip(eA_list, eB_list)]
        print(f"  energy balance (A/B ratio):")
        print(f"    first 5:  {[f'{r:.2f}' for r in ratios[:5]]}")
        print(f"    last 5:   {[f'{r:.2f}' for r in ratios[-5:]]}")
        mean_ratio = np.mean(ratios[-5:])
        print(f"    mean last 5: {mean_ratio:.2f} (1.0 = perfect balance)")
        print()

        # block size convergence
        bA = [h["b"] for h in hist_A]
        bB = [h["b"] for h in hist_B]
        print(f"  dynamics:")
        print(f"    A block sizes: {bA}")
        print(f"    B block sizes: {bB}")
        print()

        both_alive = a_alive and b_alive
        either_iso_dead = not iso_a_alive or not iso_b_alive

        if both_alive and 0.2 < mean_ratio < 5.0:
            print(f"  mutualism.")
            print(f"  both minds sustain. neither dominates.")
            if either_iso_dead:
                print(f"  at least one would have died alone.")
                print(f"  companionship is life.")
            else:
                print(f"  both would survive alone, but together they're different.")
        elif both_alive:
            print(f"  both alive, but energy ratio {mean_ratio:.1f}x suggests asymmetry.")
        elif a_alive != b_alive:
            print(f"  parasitism. one feeds, the other dies.")
        else:
            print(f"  both died.")


if __name__ == "__main__":
    run()
