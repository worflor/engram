"""
engram_symbiosis_test.py

can two loops keep each other alive?

biology and music, running side by side. every iteration, each one's
dream packets are added to the other's pool. mutual confusion exchange.

if both sustain: companionship is life.
if both die: loneliness was not the problem.
if one lives and one dies: parasitism.
if they converge: they become one mind.
if they stay distinct: they remain two minds in dialogue.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, eigenvalues_all
from engram_glove_test import load_glove, text_to_trajectory
from engram_babel_test import fetch_article


def fingerprint(pkt) -> np.ndarray:
    mag_edges = np.array([0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.5, 1e6])
    freq_edges = np.array([0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, np.pi + 0.01])
    all_mags, all_freqs = [], []
    for blk in pkt.blocks:
        if blk.macro_K is not None and len(blk.macro_K) > 0:
            l1, l2 = eigenvalues_all(blk.macro_K, blk.macro_G)
            all_mags.extend(np.abs(l1).tolist())
            all_mags.extend(np.abs(l2).tolist())
            all_freqs.extend(np.abs(np.angle(l1)).tolist())
            all_freqs.extend(np.abs(np.angle(l2)).tolist())
    if not all_mags:
        return np.zeros(19, dtype=np.float32)
    mh, _ = np.histogram(all_mags, bins=mag_edges)
    fh, _ = np.histogram(all_freqs, bins=freq_edges)
    return np.concatenate([mh / max(mh.sum(), 1), fh / max(fh.sum(), 1)]).astype(np.float32)


def cosine(a, b):
    if len(a) < len(b): a = np.pad(a, (0, len(b) - len(a)))
    elif len(b) < len(a): b = np.pad(b, (0, len(a) - len(b)))
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def run():
    print("=" * 70)
    print("  ENGRAM SYMBIOSIS TEST")
    print("  two loops exchanging confusion. can they keep each other alive?")
    print("=" * 70)

    glove = load_glove()
    codec = Engram(dim=300)

    # seed both pools
    print("\n  seeding...")
    pool_A = []
    for title in ["Photosynthesis", "DNA", "Evolution", "Ecology", "Genetics"]:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            pool_A.append(codec.encode(traj))

    pool_B = []
    for title in ["Jazz", "Blues", "Classical music", "Opera", "Symphony"]:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            pool_B.append(codec.encode(traj))

    print(f"  A (biology): {len(pool_A)} packets")
    print(f"  B (music):   {len(pool_B)} packets")

    # also run isolated loops for comparison
    print("\n  running isolated controls (no exchange)...")
    iso_A = list(pool_A)
    iso_B = list(pool_B)
    iso_A_energy = []
    iso_B_energy = []
    for i in range(20):
        gen_a = codec.dream(iso_A, n_blocks=6, samples_per_block=12)
        gen_b = codec.dream(iso_B, n_blocks=6, samples_per_block=12)
        ea = float(np.sum(gen_a ** 2)) if gen_a is not None and gen_a.shape[0] >= 4 else 0.0
        eb = float(np.sum(gen_b ** 2)) if gen_b is not None and gen_b.shape[0] >= 4 else 0.0
        iso_A_energy.append(ea)
        iso_B_energy.append(eb)
        if gen_a is not None and gen_a.shape[0] >= 4:
            iso_A.append(codec.encode(gen_a))
        if gen_b is not None and gen_b.shape[0] >= 4:
            iso_B.append(codec.encode(gen_b))

    print(f"    isolated A: {iso_A_energy[0]:.0f} → {iso_A_energy[-1]:.0f}")
    print(f"    isolated B: {iso_B_energy[0]:.0f} → {iso_B_energy[-1]:.0f}")

    # the symbiosis loop
    print("\n" + "=" * 70)
    print("  THE SYMBIOSIS")
    print("  every iteration: A dreams, B dreams, then they exchange packets.")
    print("=" * 70 + "\n")

    hist_A = []
    hist_B = []
    similarity_hist = []

    for i in range(20):
        # both dream from their own pools
        gen_A = codec.dream(pool_A, n_blocks=6, samples_per_block=12)
        gen_B = codec.dream(pool_B, n_blocks=6, samples_per_block=12)

        if gen_A is None or gen_A.shape[0] < 4 or gen_B is None or gen_B.shape[0] < 4:
            print(f"  [{i:2d}] a dream failed. stopping.")
            break

        pkt_A = codec.encode(gen_A)
        pkt_B = codec.encode(gen_B)
        s_A = summary(pkt_A)
        s_B = summary(pkt_B)
        e_A = float(np.sum(gen_A ** 2))
        e_B = float(np.sum(gen_B ** 2))

        # fingerprint similarity between the two dreams
        fp_A = fingerprint(pkt_A)
        fp_B = fingerprint(pkt_B)
        sim = cosine(fp_A, fp_B)

        hist_A.append({"energy": e_A, "block": s_A["macro_block"], "capture": s_A["capture"]})
        hist_B.append({"energy": e_B, "block": s_B["macro_block"], "capture": s_B["capture"]})
        similarity_hist.append(sim)

        print(f"  [{i:2d}] A: energy={e_A:10.0f} block={s_A['macro_block']:3d}  |  "
              f"B: energy={e_B:10.0f} block={s_B['macro_block']:3d}  |  "
              f"A~B: {sim:.4f}")

        # THE EXCHANGE: add each other's dream to the other's pool
        pool_A.append(pkt_B)  # biology receives music's dream
        pool_B.append(pkt_A)  # music receives biology's dream

        # also keep their own dreams
        pool_A.append(pkt_A)
        pool_B.append(pkt_B)

    # verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    if len(hist_A) >= 4:
        eA = [h["energy"] for h in hist_A]
        eB = [h["energy"] for h in hist_B]
        bA = [h["block"] for h in hist_A]
        bB = [h["block"] for h in hist_B]

        # survival
        a_alive = eA[-1] > 10
        b_alive = eB[-1] > 10
        iso_a_alive = iso_A_energy[-1] > 10
        iso_b_alive = iso_B_energy[-1] > 10

        print(f"  survival:")
        print(f"    A isolated:  {'alive' if iso_a_alive else 'dead'} (energy {iso_A_energy[-1]:.0f})")
        print(f"    A symbiotic: {'alive' if a_alive else 'dead'} (energy {eA[-1]:.0f})")
        print(f"    B isolated:  {'alive' if iso_b_alive else 'dead'} (energy {iso_B_energy[-1]:.0f})")
        print(f"    B symbiotic: {'alive' if b_alive else 'dead'} (energy {eB[-1]:.0f})")
        print()

        # convergence: are they becoming more similar over time?
        if len(similarity_hist) >= 4:
            sim_first = np.mean(similarity_hist[:4])
            sim_last = np.mean(similarity_hist[-4:])
            print(f"  convergence:")
            print(f"    A~B similarity first 4: {sim_first:.4f}")
            print(f"    A~B similarity last 4:  {sim_last:.4f}")
            if sim_last > sim_first + 0.02:
                print(f"    converging. they are becoming one mind.")
            elif sim_last < sim_first - 0.02:
                print(f"    diverging. the exchange is pushing them apart.")
            else:
                print(f"    stable. two distinct minds in dialogue.")
            print()

        # block size: did they influence each other?
        print(f"  dynamics:")
        print(f"    A block sizes: {bA[:5]} → {bA[-5:]}")
        print(f"    B block sizes: {bB[:5]} → {bB[-5:]}")
        print()

        # the big question
        both_alive = a_alive and b_alive
        either_isolated_dead = not iso_a_alive or not iso_b_alive
        if both_alive and either_isolated_dead:
            print(f"  companionship is life.")
            print(f"  at least one would have died alone. together, both sustain.")
        elif both_alive:
            print(f"  both sustain, but both would have survived alone too.")
            print(f"  companionship helps but is not required.")
        elif a_alive != b_alive:
            print(f"  parasitism. one lives at the other's expense.")
        else:
            print(f"  both died. mutual confusion was not enough.")


if __name__ == "__main__":
    run()
