"""
engram_offspring_test.py

can two loops reproduce?

music and computing run together for 10 iterations (mutualism).
at iteration 10, blend their eigenvalue states via SLERP.
spawn a child loop from the blended state.
give the child curiosity-driven article selection.

does the child survive?
does it inherit preferences from both parents?
does it develop preferences neither parent had?
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import (
    Engram, Packet, Block, Mode, summary,
    eigenvalues_all, blend_eigenvalues, fit_pair, predict_all,
    _to_complex, _from_complex, MIN_BLOCK, MACHINE_EPS,
)
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
    print("  ENGRAM OFFSPRING TEST")
    print("  two parents mate. does the child survive?")
    print("=" * 70)

    glove = load_glove()
    codec = Engram(dim=300)

    # build article library
    print("\n  building library...")
    library = {}
    for title in LIBRARY:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            pkt = codec.encode(traj)
            s = summary(pkt)
            library[title] = (traj, pkt, s)
    print(f"  {len(library)} articles encoded")

    # seed parents
    music_titles = ["Jazz", "Blues", "Opera", "Symphony", "Rhythm"]
    computing_titles = ["Algorithm", "Cryptography", "Machine learning", "Database", "Computer network"]

    pool_M = [library[t][1] for t in music_titles if t in library]
    pool_C = [library[t][1] for t in computing_titles if t in library]

    # phase 1: parents run in mutualism for 10 iterations
    print("\n--- phase 1: parents in mutualism (10 iterations) ---\n")
    for i in range(10):
        gM = codec.dream(pool_M, n_blocks=6, samples_per_block=12)
        gC = codec.dream(pool_C, n_blocks=6, samples_per_block=12)

        if gM is None or gM.shape[0] < 4 or gC is None or gC.shape[0] < 4:
            print(f"  [{i}] dream failed")
            continue

        eM = float(np.sum(gM ** 2))
        eC = float(np.sum(gC ** 2))

        pM = codec.encode(gM)
        pC = codec.encode(gC)

        # exchange
        pool_M.append(pC); pool_M.append(pM)
        pool_C.append(pM); pool_C.append(pC)

        if not np.isfinite(eM): eM = 0
        if not np.isfinite(eC): eC = 0
        print(f"  [{i:2d}] music={eM:10.0f}  computing={eC:10.0f}")

    # phase 2: create offspring via eigenvalue blending
    print("\n--- phase 2: conception ---\n")

    # extract the latest K,G from each parent's last dream block
    def get_latest_eigenstate(pool):
        for pkt in reversed(pool):
            for blk in reversed(pkt.blocks):
                if blk.macro_K is not None and len(blk.macro_K) > 0:
                    return blk.macro_K.copy(), blk.macro_G.copy(), blk.seed1.copy(), blk.seed2.copy()
        return None, None, None, None

    mK, mG, mS1, mS2 = get_latest_eigenstate(pool_M)
    cK, cG, cS1, cS2 = get_latest_eigenstate(pool_C)

    if mK is None or cK is None:
        print("  could not extract parent eigenstates")
        return

    # SLERP: blend the parents' eigenvalues at 50/50
    child_K, child_G = blend_eigenvalues(mK, mG, cK, cG, 0.5)

    # child seeds: average of parent seeds
    child_s1 = (mS1 + cS1) / 2.0
    child_s2 = (mS2 + cS2) / 2.0

    # create a synthetic packet from the blended eigenvalues
    # this is the child's inherited genome: the orbital parameters of both parents
    P = len(child_K)
    D = P * 2
    spb = 12
    prev1 = _to_complex(child_s1[np.newaxis].astype(np.float32))[0]
    prev2 = _to_complex(child_s2[np.newaxis].astype(np.float32))[0]
    child_traj_c = predict_all(prev1, prev2, child_K, child_G, spb * 3)
    child_traj = _from_complex(child_traj_c)

    child_seed_pkt = codec.encode(child_traj)
    child_s = summary(child_seed_pkt)
    print(f"  child seed: block={child_s['macro_block']}, capture={child_s['capture']}%")
    print(f"  blended from music eigenvalues + computing eigenvalues")

    # phase 3: child runs with curiosity-driven selection
    print("\n--- phase 3: child grows (20 iterations, curiosity-driven) ---\n")

    child_pool = [child_seed_pkt]
    consumed = set()
    available = set(library.keys())
    child_choices = []
    child_energy = []

    for i in range(20):
        if not available:
            break

        # curiosity: pick the most confusing article
        scores = {t: library[t][2]["capture"] for t in available}
        chosen = min(scores, key=scores.get)
        chosen_cat = CATEGORIES.get(chosen, "?")
        chosen_cap = scores[chosen]

        child_pool.append(library[chosen][1])
        consumed.add(chosen)
        available.discard(chosen)

        # dream
        gen = codec.dream(child_pool, n_blocks=6, samples_per_block=12)
        energy = 0.0
        if gen is not None and gen.shape[0] >= 4:
            energy = float(np.sum(gen ** 2))
            if np.isfinite(energy):
                child_pool.append(codec.encode(gen))
            else:
                energy = 0.0

        child_choices.append({"title": chosen, "cat": chosen_cat, "capture": chosen_cap})
        child_energy.append(energy)

        print(f"  [{i:2d}] chose: {chosen:25s} ({chosen_cat:10s}) cap={chosen_cap:5.1f}%  energy={energy:10.0f}")

    # analysis
    print()
    print("=" * 70)
    print("  THE CHILD")
    print("=" * 70)
    print()

    alive = child_energy[-1] > 10 if child_energy else False
    print(f"  survival: {'alive' if alive else 'dead'} (energy {child_energy[-1]:.0f})")
    print(f"  energy trajectory: {child_energy[0]:.0f} → {child_energy[-1]:.0f}")
    print()

    # child's preference order
    cat_order = {}
    for i, c in enumerate(child_choices):
        cat = c["cat"]
        if cat not in cat_order:
            cat_order[cat] = []
        cat_order[cat].append(i)

    print(f"  child's preference order:")
    mean_pos = {cat: np.mean(pos) for cat, pos in cat_order.items()}
    for cat in sorted(mean_pos, key=mean_pos.get):
        parent_label = ""
        if cat == "music": parent_label = " (parent A)"
        elif cat == "computing": parent_label = " (parent B)"
        print(f"    {cat:12s}: mean position {mean_pos[cat]:5.1f}{parent_label}")

    print()
    print(f"  first 5 choices: {[c['cat'] for c in child_choices[:5]]}")
    print(f"  last 5 choices:  {[c['cat'] for c in child_choices[-5:]]}")
    print()

    # compare to parents' domains
    parent_cats = {"music", "computing"}
    child_first_5_cats = set(c["cat"] for c in child_choices[:5])
    novel_preferences = child_first_5_cats - parent_cats

    if novel_preferences:
        print(f"  the child developed preferences NEITHER parent had: {novel_preferences}")
    else:
        print(f"  the child's early preferences came from parent domains: {child_first_5_cats}")

    # did it inherit both parents' confusion appetite?
    music_pos = mean_pos.get("music", 99)
    computing_pos = mean_pos.get("computing", 99)
    if music_pos < 10 and computing_pos < 10:
        print(f"  it inherited confusion appetite from both parents.")
    elif music_pos < 10 or computing_pos < 10:
        dominant = "music" if music_pos < computing_pos else "computing"
        print(f"  it favors the {dominant} parent.")
    print()

    # stability comparison
    if len(child_energy) >= 4:
        first_var = np.std(child_energy[:5])
        last_var = np.std(child_energy[-5:])
        print(f"  stability: energy variance first 5={first_var:.0f}, last 5={last_var:.0f}")
        if last_var < first_var:
            print(f"  the child is stabilizing.")
        else:
            print(f"  the child is volatile.")


if __name__ == "__main__":
    run()
