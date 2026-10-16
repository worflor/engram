"""
engram_cpr_test.py

can confusion be transplanted?

biology is dying: high capture, low residuals, decaying energy.
music is alive: low capture, high residuals, sustained energy.

at iteration 5, inject music's packets into biology's pool.
does the borrowed confusion restart the loop?
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
    print("  ENGRAM CPR TEST")
    print("  can a dying loop be revived with borrowed confusion?")
    print("=" * 70)

    glove = load_glove()
    codec = Engram(dim=300)

    # seed biology
    print("\n  seeding biology...")
    bio_packets = []
    for title in ["Photosynthesis", "DNA", "Evolution", "Ecology", "Genetics"]:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            bio_packets.append(codec.encode(traj))
            print(f"    {title}: {traj.shape[0]} words")

    # seed music (the donor)
    print("\n  seeding music (the donor)...")
    music_packets = []
    for title in ["Jazz", "Blues", "Classical music", "Opera", "Symphony"]:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            music_packets.append(codec.encode(traj))
            print(f"    {title}: {traj.shape[0]} words")

    # also run music for 5 iterations to get warm donor packets
    print("\n  warming up donor (music dreams x5)...")
    donor_packets = list(music_packets)
    for i in range(5):
        gen = codec.dream(donor_packets, n_blocks=6, samples_per_block=12)
        if gen is not None and gen.shape[0] >= 4:
            donor_packets.append(codec.encode(gen))

    # the experiment
    print("\n" + "=" * 70)
    print("  THE EXPERIMENT")
    print("=" * 70)

    pool = list(bio_packets)
    history = []

    for i in range(12):
        # at iteration 5: inject music
        if i == 6:
            n_before = len(pool)
            pool.extend(donor_packets)
            print(f"\n  >>> INJECTION at iteration {i}: {len(donor_packets)} music packets added to pool ({n_before} → {len(pool)})")
            print()

        gen = codec.dream(pool, n_blocks=6, samples_per_block=12)
        if gen is None or gen.shape[0] < 4:
            print(f"  [{i:2d}] dream returned nothing")
            history.append({"iteration": i, "energy": 0, "block": 0, "capture": 0})
            continue

        pkt = codec.encode(gen)
        s = summary(pkt)
        energy = float(np.sum(gen ** 2))
        pool.append(pkt)

        # nearest words
        centroid = gen.mean(axis=0)
        words_list = list(glove.keys())[:30000]
        vecs = np.stack([glove[w] for w in words_list if w in glove])
        valid = [w for w in words_list if w in glove]
        dists = np.linalg.norm(vecs - centroid[:300], axis=1)
        nearest = [valid[j] for j in np.argsort(dists)[:3]]

        phase = "before" if i < 6 else "AFTER "
        print(f"  [{i:2d}] {phase}  energy={energy:10.1f}  block={s['macro_block']:3d}  capture={s['capture']:5.1f}%  {', '.join(nearest)}")

        history.append({
            "iteration": i, "energy": energy,
            "block": s["macro_block"], "capture": s["capture"],
            "words": nearest,
        })

    # verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    before = [h for h in history if h["iteration"] < 6 and h["energy"] > 0]
    after = [h for h in history if h["iteration"] >= 6 and h["energy"] > 0]

    if before and after:
        e_before = [h["energy"] for h in before]
        e_after = [h["energy"] for h in after]
        b_before = [h["block"] for h in before]
        b_after = [h["block"] for h in after]

        print(f"  before injection:")
        print(f"    energy: {e_before[0]:.0f} → {e_before[-1]:.0f} (trend: {'decaying' if e_before[-1] < e_before[0] else 'growing'})")
        print(f"    block sizes: {b_before}")
        print()
        print(f"  after injection:")
        print(f"    energy: {e_after[0]:.0f} → {e_after[-1]:.0f} (trend: {'decaying' if e_after[-1] < e_after[0] else 'growing'})")
        print(f"    block sizes: {b_after}")
        print()

        revived = e_after[-1] > e_before[-1]
        block_changed = np.mean(b_after) != np.mean(b_before)

        if revived:
            print(f"  the patient is alive.")
            print(f"  borrowed confusion restarted the loop.")
            if np.mean(b_after) < np.mean(b_before):
                print(f"  block size dropped: the loop adopted music's fast dynamics.")
            elif np.mean(b_after) > np.mean(b_before):
                print(f"  block size grew: the loop found a hybrid rhythm.")
        else:
            print(f"  the transplant did not revive the loop.")
            print(f"  biology's predictability absorbed the confusion.")


if __name__ == "__main__":
    run()
