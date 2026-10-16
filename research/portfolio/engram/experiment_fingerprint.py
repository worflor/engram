"""
experiment_fingerprint.py

does the brain leave a detectable eigenvalue signature on the model's output?

takes the responses from experiment_model_size.json, encodes each one
through the codec, and compares eigenvalue signatures between:
  - with brain vs without brain
  - 0.6B vs 9B
  - different topics

if the brain shapes output dynamics, brain-guided responses should have
eigenvalue centroids CLOSER to the well centroids than unguided responses.
the brain's resonance should be detectable in the text it helped produce.
"""

from __future__ import annotations

import sys
import os
import json
import numpy as np
from pathlib import Path

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "experiments"))

from engram_codec import EngramFile, Engram, _to_complex, fit_all
from engram_glove_test import load_glove, text_to_trajectory

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")
EXPERIMENT_PATH = str(_HERE / "cache" / "_experiment_model_size.json")
WINDOW = 10


def encode_text(text: str, glove: dict, codec: Engram, pairing) -> dict:
    """encode a text response and return its eigenvalue signature."""
    traj = text_to_trajectory(text, glove, window=WINDOW)
    if traj.shape[0] < 4:
        return None

    pkt = codec.encode(traj, pairing=pairing)
    if not pkt.blocks:
        return None

    # collect all macro K values
    Ks = [blk.macro_K for blk in pkt.blocks
          if blk.macro_K is not None and len(blk.macro_K) > 0]
    if not Ks:
        return None

    centroid = np.mean(np.stack(Ks), axis=0)

    # eigenvalue magnitudes and frequencies
    magnitudes = np.abs(centroid)
    frequencies = np.angle(centroid)

    return {
        "centroid": centroid,
        "magnitudes": magnitudes,
        "frequencies": frequencies,
        "n_blocks": len(pkt.blocks),
        "capture": float(pkt.capture),
    }


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS distance between two complex vectors."""
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)))


def main():
    print("=" * 60)
    print("  EXPERIMENT: Brain Fingerprint Detection")
    print("=" * 60)

    # load brain + experiment data
    brain = EngramFile.load(BRAIN_PATH)
    codec = Engram(dim=brain.dim)
    pairing = brain._reference_pairing

    print(f"brain: {brain._total_absorbed:,} articles, {len(brain.wells())} wells")

    with open(EXPERIMENT_PATH, "r", encoding="utf-8") as f:
        experiments = json.load(f)

    print(f"experiment responses: {sum(len(e['turns']) for e in experiments)}")

    print("loading GloVe...", flush=True)
    glove = load_glove()
    print()

    # encode every response
    results = []
    for exp in experiments:
        model = exp["model"]
        has_brain = exp["brain"]
        topic = exp["topic"]
        model_label = "0.6B" if "0.6b" in model else "9B"

        for turn in exp["turns"]:
            response = turn["response"]
            if len(response) < 50:
                continue

            sig = encode_text(response, glove, codec, pairing)
            if sig is None:
                continue

            # distance from nearest well centroid
            nearest_name, nearest_dist = brain.nearest_well(sig["centroid"])

            # distance from global centroid
            gc = brain.global_centroid()
            gc_dist = distance(sig["centroid"], gc) if gc is not None else 0.0

            results.append({
                "model": model_label,
                "brain": has_brain,
                "topic": topic,
                "turn": turn["turn"],
                "nearest_well": nearest_name,
                "well_dist": nearest_dist,
                "global_dist": gc_dist,
                "capture": sig["capture"],
                "n_blocks": sig["n_blocks"],
                "mean_magnitude": float(np.mean(sig["magnitudes"])),
                "mean_frequency": float(np.mean(np.abs(sig["frequencies"]))),
            })

    # analyze: group by condition
    print(f"encoded {len(results)} responses\n")

    conditions = {}
    for r in results:
        key = f"{r['model']} {'+ brain' if r['brain'] else 'no brain'}"
        conditions.setdefault(key, []).append(r)

    print(f"{'condition':<20s}  {'well_dist':>9s}  {'global_dist':>11s}  {'capture':>7s}  {'magnitude':>9s}  {'frequency':>9s}")
    print("-" * 75)

    for cond_name in ["0.6B no brain", "0.6B + brain", "9B no brain", "9B + brain"]:
        entries = conditions.get(cond_name, [])
        if not entries:
            continue
        avg_well = np.mean([e["well_dist"] for e in entries])
        avg_gc = np.mean([e["global_dist"] for e in entries])
        avg_cap = np.mean([e["capture"] for e in entries])
        avg_mag = np.mean([e["mean_magnitude"] for e in entries])
        avg_freq = np.mean([e["mean_frequency"] for e in entries])
        print(f"{cond_name:<20s}  {avg_well:>9.4f}  {avg_gc:>11.4f}  {avg_cap:>6.1f}%  {avg_mag:>9.4f}  {avg_freq:>9.4f}")

    # per-topic breakdown: does the brain pull responses toward the right well?
    print(f"\n{'topic':<25s}  {'condition':<20s}  {'nearest_well':<15s}  {'well_dist':>9s}")
    print("-" * 75)

    by_topic = {}
    for r in results:
        by_topic.setdefault(r["topic"], []).append(r)

    for topic in sorted(by_topic.keys()):
        entries = by_topic[topic]
        for r in sorted(entries, key=lambda x: (x["model"], x["brain"])):
            cond = f"{r['model']} {'+ brain' if r['brain'] else 'no brain'}"
            print(f"{topic:<25s}  {cond:<20s}  {r['nearest_well']:<15s}  {r['well_dist']:>9.4f}")

    # the key question: are brain-guided responses closer to wells?
    print("\n" + "=" * 60)
    print("  KEY METRIC: Mean distance to nearest well")
    print("=" * 60)

    brain_on = [r["well_dist"] for r in results if r["brain"]]
    brain_off = [r["well_dist"] for r in results if not r["brain"]]

    if brain_on and brain_off:
        mean_on = np.mean(brain_on)
        mean_off = np.mean(brain_off)
        reduction = (1 - mean_on / mean_off) * 100

        print(f"  without brain: {mean_off:.4f}")
        print(f"  with brain:    {mean_on:.4f}")
        print(f"  reduction:     {reduction:.1f}%")
        print()
        if reduction > 0:
            print(f"  the brain pulls responses {reduction:.1f}% closer to known wells.")
            print(f"  the brain's resonance IS detectable in the text it helped produce.")
        else:
            print(f"  no detectable fingerprint. the brain's guidance doesn't")
            print(f"  imprint on the eigenvalue structure of the output text.")

    # save
    out_path = str(_HERE / "cache" / "_experiment_fingerprint.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nraw results saved: {out_path}")


if __name__ == "__main__":
    main()
