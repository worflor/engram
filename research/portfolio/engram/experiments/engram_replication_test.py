"""
engram_replication_test.py

the peer review. does mutualism replicate?

10 independent runs of music + computing symbiosis.
each run uses a different random seed for dream residuals.
20 iterations per run.

8/10 both alive = dynamics are real.
5-7/10 = fragile.
<5/10 = back to the drawing board.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary
from engram_glove_test import load_glove, text_to_trajectory
from engram_babel_test import fetch_article


MUSIC_TITLES = ["Jazz", "Blues", "Classical music", "Opera", "Symphony"]
COMPUTING_TITLES = ["Algorithm", "Computer science", "Artificial intelligence", "Machine learning", "Cryptography"]

N_RUNS = 10
N_ITER = 20
ALIVE_THRESHOLD = 10  # energy above this = alive


def run_single_trial(
    seed_packets_A: list,
    seed_packets_B: list,
    codec: Engram,
    trial_seed: int,
) -> dict:
    """one independent symbiosis trial."""
    np.random.seed(trial_seed)

    pool_A = list(seed_packets_A)
    pool_B = list(seed_packets_B)

    eA_hist = []
    eB_hist = []

    for i in range(N_ITER):
        gA = codec.dream(pool_A, n_blocks=6, samples_per_block=12)
        gB = codec.dream(pool_B, n_blocks=6, samples_per_block=12)

        if gA is None or gA.shape[0] < 4 or gB is None or gB.shape[0] < 4:
            eA_hist.append(0.0)
            eB_hist.append(0.0)
            continue

        eA = float(np.sum(gA ** 2))
        eB = float(np.sum(gB ** 2))

        # nan/inf guard
        if not np.isfinite(eA): eA = 0.0
        if not np.isfinite(eB): eB = 0.0

        eA_hist.append(eA)
        eB_hist.append(eB)

        pA = codec.encode(gA)
        pB = codec.encode(gB)

        # exchange
        pool_A.append(pB)
        pool_B.append(pA)
        pool_A.append(pA)
        pool_B.append(pB)

    a_alive = eA_hist[-1] > ALIVE_THRESHOLD if eA_hist else False
    b_alive = eB_hist[-1] > ALIVE_THRESHOLD if eB_hist else False

    ratio = max(eA_hist[-1], 1) / max(eB_hist[-1], 1) if eA_hist and eB_hist else 1.0

    return {
        "a_alive": a_alive,
        "b_alive": b_alive,
        "both_alive": a_alive and b_alive,
        "eA_final": eA_hist[-1] if eA_hist else 0.0,
        "eB_final": eB_hist[-1] if eB_hist else 0.0,
        "ratio": ratio,
        "eA_hist": eA_hist,
        "eB_hist": eB_hist,
    }


def run():
    print("=" * 70)
    print("  ENGRAM REPLICATION TEST")
    print(f"  {N_RUNS} independent runs of music + computing symbiosis")
    print(f"  {N_ITER} iterations each, different random seeds")
    print("=" * 70)

    glove = load_glove()
    codec = Engram(dim=300)

    # encode seed articles once (shared across all trials)
    print("\n  encoding seed articles...")
    seeds_A = []
    for title in MUSIC_TITLES:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            seeds_A.append(codec.encode(traj))
    print(f"    A (music): {len(seeds_A)} articles")

    seeds_B = []
    for title in COMPUTING_TITLES:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            seeds_B.append(codec.encode(traj))
    print(f"    B (computing): {len(seeds_B)} articles")

    # run trials
    print(f"\n  running {N_RUNS} trials...\n")
    results = []

    for trial in range(N_RUNS):
        trial_seed = trial * 7919 + 42  # distinct primes for separation
        r = run_single_trial(seeds_A, seeds_B, codec, trial_seed)
        results.append(r)

        status = "BOTH ALIVE" if r["both_alive"] else (
            "A alive only" if r["a_alive"] else (
            "B alive only" if r["b_alive"] else "BOTH DEAD"))

        print(f"  trial {trial:2d} (seed={trial_seed:5d}): {status:14s}  "
              f"A={r['eA_final']:10.0f}  B={r['eB_final']:10.0f}  "
              f"ratio={r['ratio']:.2f}")

    # verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    both_alive_count = sum(1 for r in results if r["both_alive"])
    a_alive_count = sum(1 for r in results if r["a_alive"])
    b_alive_count = sum(1 for r in results if r["b_alive"])
    both_dead_count = sum(1 for r in results if not r["a_alive"] and not r["b_alive"])

    print(f"  both alive:  {both_alive_count}/{N_RUNS}")
    print(f"  A alive:     {a_alive_count}/{N_RUNS}")
    print(f"  B alive:     {b_alive_count}/{N_RUNS}")
    print(f"  both dead:   {both_dead_count}/{N_RUNS}")
    print()

    # energy statistics for trials where both survived
    alive_trials = [r for r in results if r["both_alive"]]
    if alive_trials:
        ratios = [r["ratio"] for r in alive_trials]
        eA_finals = [r["eA_final"] for r in alive_trials]
        eB_finals = [r["eB_final"] for r in alive_trials]
        print(f"  among {len(alive_trials)} mutual-survival trials:")
        print(f"    A final energy: {np.mean(eA_finals):.0f} +/- {np.std(eA_finals):.0f}")
        print(f"    B final energy: {np.mean(eB_finals):.0f} +/- {np.std(eB_finals):.0f}")
        print(f"    energy ratio:   {np.mean(ratios):.2f} +/- {np.std(ratios):.2f}")
        print()

    if both_alive_count >= 8:
        print(f"  {both_alive_count}/{N_RUNS}: the dynamics are real.")
        print(f"  mutualism replicates. everything discovered today is science.")
    elif both_alive_count >= 5:
        print(f"  {both_alive_count}/{N_RUNS}: the dynamics are fragile.")
        print(f"  mutualism exists but is not robust. need to find what stabilizes it.")
    else:
        print(f"  {both_alive_count}/{N_RUNS}: mutualism does not replicate.")
        print(f"  tonight was instructive but not reproducible. back to the drawing board.")


if __name__ == "__main__":
    run()
