"""
engram_alexandria_session.py

one brain, one session, start to finish.

watch alexandria form: wells emerge, drift signals converge, dreams
find their affinity, the manifold takes shape article by article.

this is the session that chapter 20 will narrate.
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from engram_codec import Engram, EngramFile, summary, eigenvalues_all

# use the glove test's utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from engram_glove_test import load_glove, text_to_trajectory, tokenize_simple
from engram_babel_test import fetch_article

BRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'cache', '_alexandria_session.engram')

# articles grouped by domain, fed one at a time in interleaved order
# so the wells form gradually and the drift signal is visible
CURRICULUM = [
    # round 1: one from each domain
    ("Photosynthesis", "biology"),
    ("Jazz", "music"),
    ("Quantum mechanics", "physics"),
    ("Roman Empire", "history"),
    ("Algorithm", "computing"),

    # round 2: second from each domain (wells should start forming)
    ("DNA", "biology"),
    ("Blues", "music"),
    ("Thermodynamics", "physics"),
    ("French Revolution", "history"),
    ("Cryptography", "computing"),

    # round 3: deeper into each domain (drift should decrease within domains)
    ("Evolution", "biology"),
    ("Classical music", "music"),
    ("Optics", "physics"),
    ("Cold War", "history"),
    ("Machine learning", "computing"),

    # round 4: even deeper (wells should be well-established)
    ("Genetics", "biology"),
    ("Opera", "music"),
    ("Electromagnetism", "physics"),
    ("Renaissance", "history"),
    ("Database", "computing"),
]

WINDOW = 10  # phrase-level smoothing for learning signal


def run():
    print("=" * 70)
    print("  ALEXANDRIA: SESSION ONE")
    print("  one brain, one session, start to finish")
    print("=" * 70)
    print()

    glove = load_glove()
    codec = Engram(dim=300)

    # create or load
    if os.path.exists(BRAIN_PATH):
        print(f"  loading existing brain from {BRAIN_PATH}")
        ef = EngramFile.load(BRAIN_PATH)
        print(f"  permanent: {len(ef.permanent)}, ephemeral: {len(ef.ephemeral)}")
        print(f"  wells: {list(ef.wells().keys())}")
    else:
        print("  creating new brain: Alexandria")
        ef = EngramFile.create(BRAIN_PATH, name="Alexandria", dim=300, alpha=0.005)

    history = []
    t_session = time.time()

    print()
    print("=" * 70)
    print("  PHASE 1: ABSORBING KNOWLEDGE")
    print("  one article at a time, interleaved across domains")
    print("=" * 70)
    print()

    for i, (title, domain) in enumerate(CURRICULUM):
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove, window=WINDOW)

        if traj.shape[0] < 10:
            print(f"  [{i:2d}] {title:25s} ({domain:10s}): too few words")
            continue

        t0 = time.time()

        # measure BEFORE absorbing (how surprising is this to the current manifold?)
        m = ef.measure(codec, traj)
        encode_time = time.time() - t0

        # absorb into permanent pool
        if "packet" in m:
            ef.absorb(m["packet"], permanent=True, source=domain)

        # current well state
        wells = ef.wells()
        n_wells = len(wells)

        # inter-well distances (how far apart are the domains?)
        well_dists = {}
        well_names = sorted(wells.keys())
        for wi in range(len(well_names)):
            for wj in range(wi + 1, len(well_names)):
                na, nb = well_names[wi], well_names[wj]
                dist = float(np.sqrt(np.mean(np.abs(wells[na] - wells[nb]) ** 2)))
                well_dists[f"{na[:3]}-{nb[:3]}"] = round(dist, 4)

        record = {
            "step": i,
            "title": title,
            "domain": domain,
            "words": traj.shape[0],
            "drift": m.get("drift", 0),
            "texture": m.get("texture", 0),
            "capture": m.get("capture", 0),
            "block_size": m.get("block_size", 0),
            "nearest_well": m.get("nearest_well", "none"),
            "n_wells": n_wells,
            "n_permanent": len(ef.permanent),
            "encode_time": round(encode_time, 2),
        }
        history.append(record)

        nearest = m.get("nearest_well", "?")
        correct = "yes" if nearest == domain else ("new" if nearest == "none" else f"no ({nearest})")

        print(f"  [{i:2d}] {title:25s} ({domain:10s})  "
              f"drift={m.get('drift',0):5.1f}%  "
              f"nearest={nearest:10s} correct={correct:5s}  "
              f"wells={n_wells}  {encode_time:.1f}s")

    # ── well formation summary ────────────────────────────────────
    print()
    print("=" * 70)
    print("  WELL FORMATION")
    print("=" * 70)
    print()

    wells = ef.wells()
    gc = ef.global_centroid()
    if gc is not None:
        scale = float(np.sqrt(np.mean(np.abs(gc) ** 2))) + 1e-7
        print(f"  {len(wells)} wells formed:")
        for name in sorted(wells.keys()):
            count = sum(1 for lc in ef.permanent.lifecycles if lc.source == name)
            dist_from_global = float(np.sqrt(np.mean(np.abs(wells[name] - gc) ** 2))) / scale * 100
            print(f"    {name:12s}: {count:2d} packets, {dist_from_global:5.1f}% from global centroid")

        print()
        print("  inter-well distances:")
        well_names = sorted(wells.keys())
        # header
        print(f"    {'':12s}", end="")
        for n in well_names:
            print(f"{n[:8]:>10s}", end="")
        print()
        for na in well_names:
            print(f"    {na:12s}", end="")
            for nb in well_names:
                if na == nb:
                    print(f"{'---':>10s}", end="")
                else:
                    d = float(np.sqrt(np.mean(np.abs(wells[na] - wells[nb]) ** 2))) / scale * 100
                    print(f"{d:10.1f}", end="")
            print()

    # ── learning signal by domain ─────────────────────────────────
    print()
    print("=" * 70)
    print("  LEARNING SIGNAL (DRIFT BY DOMAIN)")
    print("=" * 70)
    print()

    domains = sorted(set(r["domain"] for r in history))
    for domain in domains:
        domain_records = [r for r in history if r["domain"] == domain]
        drifts = [r["drift"] for r in domain_records]
        if len(drifts) >= 2:
            first = drifts[0]
            last = drifts[-1]
            delta = last - first
            print(f"  {domain:12s}: {' → '.join(f'{d:.1f}%' for d in drifts)}  (delta: {delta:+.1f}%)")

    # ── nearest well accuracy ─────────────────────────────────────
    print()
    correct_count = sum(1 for r in history if r["nearest_well"] == r["domain"])
    new_count = sum(1 for r in history if r["nearest_well"] == "none")
    wrong_count = len(history) - correct_count - new_count
    print(f"  well matching: {correct_count} correct, {new_count} new (first encounter), {wrong_count} wrong")

    # ── phase 2: dream and reflect ────────────────────────────────
    print()
    print("=" * 70)
    print("  PHASE 2: DREAMING")
    print("  crystallize the meta-oscillator, dream, measure affinity")
    print("=" * 70)
    print()

    ef.crystallize_dream(codec)
    print(f"  meta-oscillator crystallized from {len(ef.all_packets)} packets")
    print()

    dream_affinities = []
    for di in range(10):
        dream = codec.dream(ef.all_packets, n_blocks=4, samples_per_block=8)
        if dream is None or dream.shape[0] < 4:
            print(f"  dream [{di}]: failed")
            continue

        energy = float(np.sum(dream ** 2))
        if not np.isfinite(energy):
            continue

        # which well does the dream land nearest?
        dream_pkt = codec.encode(dream)
        dream_Ks = [blk.macro_K for blk in dream_pkt.blocks
                    if blk.macro_K is not None and len(blk.macro_K) > 0]
        if dream_Ks:
            dream_centroid = np.mean(np.stack(dream_Ks), axis=0)
            well_name, well_dist = ef.nearest_well(dream_centroid)
        else:
            well_name = "none"

        dream_affinities.append(well_name)
        s = summary(dream_pkt)
        print(f"  dream [{di:2d}]: energy={energy:8.0f}  block={s['macro_block']:3d}  nearest_well={well_name}")

    if dream_affinities:
        print()
        affinity_counts = {}
        for w in dream_affinities:
            affinity_counts[w] = affinity_counts.get(w, 0) + 1
        print(f"  dream affinity distribution:")
        for w in sorted(affinity_counts, key=affinity_counts.get, reverse=True):
            bar = "#" * affinity_counts[w]
            print(f"    {w:12s}: {affinity_counts[w]:2d} {bar}")

    # ── phase 3: save ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SAVING")
    print("=" * 70)
    print()

    ef.save(BRAIN_PATH)
    size = os.path.getsize(BRAIN_PATH)
    elapsed = time.time() - t_session

    print(f"  saved: {BRAIN_PATH}")
    print(f"  size:  {size:,} bytes ({size/1024:.1f} KB)")
    print(f"  session time: {elapsed:.1f}s")

    # ── final state ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  ALEXANDRIA: SESSION STATE")
    print("=" * 70)
    print()

    print(f"  name:        {ef.name}")
    print(f"  born:        {ef.born}")
    print(f"  permanent:   {len(ef.permanent)} packets")
    print(f"  ephemeral:   {len(ef.ephemeral)} packets")
    print(f"  wells:       {len(ef.wells())}")
    for name in sorted(ef.wells().keys()):
        count = sum(1 for lc in ef.permanent.lifecycles if lc.source == name)
        print(f"               {name}: {count} packets")
    print(f"  dream:       {'crystallized' if ef.dream_meta_K is not None else 'none'}")
    fp = ef.fingerprint()
    print(f"  fingerprint: {fp.shape[0]} floats")
    print(f"  disk:        {size:,} bytes")


if __name__ == "__main__":
    run()
