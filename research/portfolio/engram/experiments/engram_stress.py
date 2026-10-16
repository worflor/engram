"""
engram_stress.py

build a real brain from real data. keep it on disk between runs.
stress the full EngramFile: absorb, reflect, evaporate, dream, save, load.

uses the GloVe word trajectories from the Wikipedia tests.
accumulates across runs so each execution adds more experience.
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

BRAIN_DIR = os.path.join(os.path.dirname(__file__), "_brain_alexandria")

KNOWLEDGE = [
    "Photosynthesis", "DNA", "Evolution", "Ecology", "Genetics",
    "Jazz", "Blues", "Classical music", "Opera", "Symphony",
    "Quantum mechanics", "Thermodynamics", "Optics",
    "Roman Empire", "French Revolution", "Cold War",
    "Algorithm", "Cryptography", "Machine learning",
    "Amazon rainforest", "Sahara", "Pacific Ocean",
]

INTERACTION_PROMPTS = [
    "The river cuts through the canyon over millions of years, carving shapes that echo the mathematics of erosion and fluid dynamics.",
    "When you listen to a fugue by Bach, each voice enters independently but the whole becomes something none of them could achieve alone.",
    "The difference between a good algorithm and a great one is not speed. It is the shape of the problem it refuses to solve.",
    "Every cell in your body carries the same DNA, yet a neuron looks nothing like a red blood cell. The instructions are identical. The reading is everything.",
    "The moon has no atmosphere. Sound cannot travel there. Yet astronauts reported a profound sense of silence that felt louder than anything on Earth.",
]

CATEGORIES = {}
cats = ["biology", "biology", "biology", "biology", "biology",
        "music", "music", "music", "music", "music",
        "physics", "physics", "physics",
        "history", "history", "history",
        "computing", "computing", "computing",
        "geography", "geography", "geography"]
for i, title in enumerate(KNOWLEDGE):
    CATEGORIES[title] = cats[i]


def run():
    print("=" * 70)
    print("  ENGRAM STRESS TEST")
    print("  building a persistent brain from real data")
    print("=" * 70)
    print()

    glove = load_glove()
    codec = Engram(dim=300)

    # load or create the brain
    if os.path.exists(os.path.join(BRAIN_DIR, "manifest.json")):
        print("  loading existing brain...")
        t0 = time.time()
        ef = EngramFile.load(BRAIN_DIR)
        print(f"  loaded in {time.time() - t0:.1f}s")
        print(f"  name: {ef.name}, tier: {ef.tier}")
        print(f"  core: {len(ef.core)}, identity: {len(ef.identity)}, scratchpad: {len(ef.scratchpad)}")
        print(f"  dream engine: {'yes' if ef.dream_meta_K is not None else 'no'}")
    else:
        print("  creating new brain: Alexandria")
        ef = EngramFile.create(
            BRAIN_DIR, name="Alexandria", dim=300, alpha=0.005,
            core_capacity=500, identity_capacity=200, scratchpad_capacity=100,
        )

    # ── phase 1: absorb knowledge into core ───────────────────────
    core_titles = set()
    for pkt, lc in zip(ef.core.packets, ef.core.lifecycles):
        core_titles.add(lc.source)

    new_knowledge = [t for t in KNOWLEDGE if t not in core_titles]
    if new_knowledge:
        print(f"\n--- absorbing {len(new_knowledge)} new knowledge articles into core ---\n")
        for title in new_knowledge:
            text = fetch_article(title, "en")
            traj = text_to_trajectory(text, glove)
            if traj.shape[0] < 20:
                print(f"  {title}: too few words")
                continue
            t0 = time.time()
            pkt = codec.encode(traj)
            s = summary(pkt)
            ef.absorb(pkt, layer="core", source=title)
            print(f"  {title:30s}: {traj.shape[0]:5d} words, {s['blocks']:4d} blocks, "
                  f"capture={s['capture']:5.1f}%, {time.time()-t0:.1f}s")
    else:
        print(f"\n  core already has all {len(KNOWLEDGE)} knowledge articles")

    # ── phase 2: absorb interaction trajectories into scratchpad ──
    print(f"\n--- absorbing {len(INTERACTION_PROMPTS)} interaction prompts into scratchpad ---\n")
    for prompt in INTERACTION_PROMPTS:
        traj = text_to_trajectory(prompt, glove)
        if traj.shape[0] < 4:
            continue
        pkt = codec.encode(traj)
        ef.absorb(pkt, layer="scratchpad", source="interaction")
        s = summary(pkt)
        words = prompt.split()[:6]
        print(f"  \"{' '.join(words)}...\": {traj.shape[0]} words, capture={s['capture']:.1f}%")

    # ── phase 3: crystallize dream engine ─────────────────────────
    print(f"\n--- crystallizing dream engine ---")
    t0 = time.time()
    ef.crystallize_dream(codec)
    print(f"  fitted meta-oscillator in {time.time()-t0:.1f}s")
    print(f"  tier: {ef.tier}")

    # ── phase 4: dream and reflect (10 iterations) ────────────────
    print(f"\n--- dream + reflect (10 iterations) ---\n")
    for i in range(10):
        dream = codec.dream(ef.all_packets, n_blocks=4, samples_per_block=8)
        if dream is None:
            print(f"  [{i:2d}] dream failed")
            continue

        r = ef.reflect(codec, dream)
        print(f"  [{i:2d}] surprise={r['surprise']:5.1f}%  relative={r['relative_surprise']:+6.1f}  "
              f"action={r['action']:8s}  energy={r['energy']:8.1f}  "
              f"identity={len(ef.identity)}  scratchpad={len(ef.scratchpad)}")

    # ── phase 5: evaporate ────────────────────────────────────────
    print(f"\n--- evaporation ---")
    before = len(ef.scratchpad)
    dropped = ef.evaporate()
    print(f"  scratchpad: {before} -> {len(ef.scratchpad)} ({dropped} evaporated)")

    # ── phase 6: consolidate ──────────────────────────────────────
    print(f"\n--- consolidation ---")
    for pool_name, pool in [("core", ef.core), ("identity", ef.identity), ("scratchpad", ef.scratchpad)]:
        before = len(pool)
        pruned = pool.consolidate()
        if pruned > 0:
            print(f"  {pool_name}: {before} -> {len(pool)} ({pruned} merged)")
        else:
            print(f"  {pool_name}: {len(pool)} packets (no duplicates)")

    # ── phase 7: save ─────────────────────────────────────────────
    print(f"\n--- saving ---")
    t0 = time.time()
    ef.save()
    save_time = time.time() - t0

    files = sorted(os.listdir(BRAIN_DIR))
    total_bytes = sum(os.path.getsize(os.path.join(BRAIN_DIR, f)) for f in files)
    print(f"  saved in {save_time:.1f}s ({total_bytes:,} bytes total)")
    for f in files:
        size = os.path.getsize(os.path.join(BRAIN_DIR, f))
        print(f"    {f:25s}: {size:>10,} bytes")

    # ── phase 8: reload and verify ────────────────────────────────
    print(f"\n--- reload verification ---")
    t0 = time.time()
    ef2 = EngramFile.load(BRAIN_DIR)
    load_time = time.time() - t0
    print(f"  loaded in {load_time:.1f}s")
    print(f"  name: {ef2.name}, tier: {ef2.tier}")
    print(f"  core: {len(ef2.core)}, identity: {len(ef2.identity)}, scratchpad: {len(ef2.scratchpad)}")
    print(f"  dream: {'yes' if ef2.dream_meta_K is not None else 'no'}")

    # verify decode
    decode_errors = 0
    for pool_name, pool in [("core", ef2.core), ("identity", ef2.identity), ("scratchpad", ef2.scratchpad)]:
        for pkt in pool.packets:
            try:
                dec = codec.decode(pkt)
                if dec.shape[1] != 300:
                    decode_errors += 1
            except Exception:
                decode_errors += 1
    print(f"  decode verification: {'PASS' if decode_errors == 0 else f'FAIL ({decode_errors} errors)'}")

    # fingerprint comparison
    fp1 = ef.fingerprint()
    fp2 = ef2.fingerprint()
    fp_match = np.allclose(fp1, fp2, atol=0.1)
    print(f"  fingerprint match: {'PASS' if fp_match else 'DRIFT (evaporation expected)'}")

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  BRAIN STATE: {ef2.name}")
    print(f"{'=' * 70}")
    print(f"  tier:        {ef2.tier}")
    print(f"  core:        {len(ef2.core):4d} packets  (permanent knowledge)")
    print(f"  identity:    {len(ef2.identity):4d} packets  (promoted dreams)")
    print(f"  scratchpad:  {len(ef2.scratchpad):4d} packets  (ephemeral memory)")
    print(f"  dream:       {'crystallized' if ef2.dream_meta_K is not None else 'none'}")
    print(f"  disk:        {total_bytes:,} bytes")

    sources = {}
    for pool in [ef2.core, ef2.identity, ef2.scratchpad]:
        for s in pool.memories_by_source():
            sources[s] = sources.get(s, 0) + pool.memories_by_source()[s]
    print(f"  sources:     {sources}")

    strongest_core = ef2.core.strongest(3)
    print(f"  strongest core: {[(i, f'{s:.3f}') for i, s in strongest_core]}")


if __name__ == "__main__":
    run()
