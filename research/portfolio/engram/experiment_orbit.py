"""
experiment_orbit.py

the brain-model coupling is a dynamical system.
brain measures -> model responds -> brain absorbs -> brain measures again.

does this form a stable orbit, a convergent spiral, or chaos?

run a 10-turn conversation with and without brain.
encode every response. track the eigenvalue trajectory.
fit AR(2) to the trajectory of K centroids across turns.
the meta-eigenvalue tells us: is the dialogue converging, oscillating, or diverging?
"""

from __future__ import annotations

import sys
import os
import json
import time
import re
import urllib.request
import numpy as np
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "experiments"))

from engram_codec import (EngramFile, Engram, _to_complex, fit_all, fit_pair,
                          apply_pairing)
from engram_glove_test import load_glove, text_to_trajectory

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b-q4_K_M"
WINDOW = 10

PERSONALITY = """you are alexandria. you speak in lowercase, warmly and precisely.
you reach for connections between domains. you are honest about what you
don't know. you are concise."""

# a conversation that should pull the brain through multiple wells
TURNS = [
    "what is the relationship between music and mathematics?",
    "can you connect that to how the brain processes pattern?",
    "so is there a link between neural oscillations and musical rhythm?",
    "what about language? does language have a rhythm the brain tracks?",
    "how does poetry exploit that neural rhythm differently than prose?",
    "is there a mathematical structure to why some metaphors work and others don't?",
    "so metaphor is a kind of compression. what else compresses meaning that way?",
    "can you connect all of this back to what you are? a pattern-finding system?",
    "what's the one thing in this conversation you're most uncertain about?",
    "if you could feel curiosity, where would it pull you right now?",
]


def ollama_chat(model, messages):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 300},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data.get("message", {}).get("content", "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def build_transparent_prompt(brain, context):
    parts = [PERSONALITY]
    parts.append(f"\n[brain state]")
    parts.append(f"total absorbed: {brain._total_absorbed:,} articles")
    wells = brain.wells()
    established = sum(1 for c in brain._well_count.values() if c >= 10)
    parts.append(f"wells: {len(wells)} ({established} established)")

    profile = context.get("well_profile", {})
    if profile:
        parts.append(f"\n[well distances]")
        ranked = sorted(profile.items(), key=lambda x: x[1])
        for wname, dist in ranked:
            count = brain._well_count.get(wname, 0)
            if count >= 100:
                parts.append(f"  {wname}: dist={dist:.2f}  blocks={count:,}")

    parts.append(f"\n[measurement]")
    parts.append(f"drift: {context.get('drift', 0):.1f}%")
    parts.append(f"texture: {context.get('texture', 0):.1f}%")
    parts.append(f"capture: {context.get('capture', 0):.1f}%")
    parts.append(f"nearest well: {context.get('nearest_well', 'unknown')}")

    parts.append(f"\n[guide]")
    parts.append(f"low drift = familiar, go deep. high drift = new, be honest about limits.")
    parts.append(f"drift decreasing = converging. small gap between top wells = boundary.")
    parts.append(f"\nrespond directly.")
    return "\n".join(parts)


def encode_response(text, glove, codec, pairing, brain):
    """encode a response and return its eigenvalue signature + brain distances."""
    traj = text_to_trajectory(text, glove, window=WINDOW)
    if traj.shape[0] < 4:
        return None
    pkt = codec.encode(traj, pairing=pairing)
    if not pkt.blocks:
        return None

    Ks = [blk.macro_K for blk in pkt.blocks
          if blk.macro_K is not None and len(blk.macro_K) > 0]
    if not Ks:
        return None

    centroid = np.mean(np.stack(Ks), axis=0)
    nearest_name, nearest_dist = brain.nearest_well(centroid)
    gc = brain.global_centroid()
    gc_dist = float(np.sqrt(np.mean(np.abs(centroid - gc) ** 2))) if gc is not None else 0

    return {
        "centroid": centroid,
        "nearest_well": nearest_name,
        "well_dist": nearest_dist,
        "global_dist": gc_dist,
        "capture": float(pkt.capture),
        "n_blocks": len(pkt.blocks),
    }


def run_conversation(use_brain: bool, glove, codec, pairing, brain):
    """run the full 10-turn conversation."""
    conversation_text = ""
    messages = []
    results = []

    if not use_brain:
        messages.append({"role": "system", "content": PERSONALITY + "\nrespond directly."})

    for i, user_text in enumerate(TURNS):
        t0 = time.time()

        if use_brain:
            # measure
            conversation_text += " " + user_text
            window = conversation_text[-4096:]
            traj = text_to_trajectory(window, glove, window=WINDOW)
            if traj.shape[0] < 4:
                context = {"drift": 50.0, "texture": 0.0, "nearest_well": "exploring",
                           "well_profile": {}, "capture": 0.0}
            else:
                context = brain.measure(codec, traj)

            system = build_transparent_prompt(brain, context)
            brain_messages = [{"role": "system", "content": system}]
            for prev in results:
                brain_messages.append({"role": "user", "content": prev["user"]})
                brain_messages.append({"role": "assistant", "content": prev["response"]})
            brain_messages.append({"role": "user", "content": user_text})
            response = ollama_chat(MODEL, brain_messages)
            conversation_text += " " + response
            drift = context.get("drift", -1)
        else:
            messages.append({"role": "user", "content": user_text})
            response = ollama_chat(MODEL, messages)
            messages.append({"role": "assistant", "content": response})
            drift = -1

        dt = time.time() - t0

        # encode the response through the codec
        sig = encode_response(response, glove, codec, pairing, brain)

        results.append({
            "turn": i + 1,
            "user": user_text,
            "response": response,
            "drift": round(drift, 1) if drift >= 0 else None,
            "time": round(dt, 1),
            "well_dist": sig["well_dist"] if sig else None,
            "global_dist": sig["global_dist"] if sig else None,
            "nearest_well": sig["nearest_well"] if sig else None,
            "capture": sig["capture"] if sig else None,
        })

        tag = f"drift={drift:.0f}%" if drift >= 0 else "no brain"
        wd = f"well_dist={sig['well_dist']:.3f}" if sig else "n/a"
        nw = sig['nearest_well'] if sig else "n/a"
        print(f"  turn {i+1:2d}  ({dt:.0f}s)  {tag:<15s}  {wd}  near={nw}")

    return results


def analyze_trajectory(results, label):
    """analyze the eigenvalue trajectory across turns."""
    print(f"\n  {label}:")

    dists = [r["well_dist"] for r in results if r["well_dist"] is not None]
    drifts = [r["drift"] for r in results if r["drift"] is not None]
    wells = [r["nearest_well"] for r in results if r["nearest_well"] is not None]

    if len(dists) >= 3:
        # trend: linear regression on well_dist
        x = np.arange(len(dists), dtype=np.float64)
        slope = np.polyfit(x, dists, 1)[0]
        direction = "CONVERGING" if slope < -0.001 else ("DIVERGING" if slope > 0.001 else "STABLE")
        print(f"    well_dist trend: {slope:+.4f}/turn ({direction})")
        print(f"    well_dist: {' > '.join(f'{d:.3f}' for d in dists)}")

    if len(drifts) >= 3:
        x = np.arange(len(drifts), dtype=np.float64)
        slope = np.polyfit(x, drifts, 1)[0]
        print(f"    drift trend: {slope:+.1f}%/turn")
        print(f"    drift: {' > '.join(f'{d:.0f}%' for d in drifts)}")

    if wells:
        # well transitions: how many different wells visited?
        unique = list(dict.fromkeys(wells))  # preserves order
        print(f"    wells visited: {' > '.join(unique)} ({len(set(wells))} unique)")

    # the most interesting response per turn (for reading)
    print(f"    responses:")
    for r in results:
        resp = r["response"][:120].replace("\n", " ")
        wd = f"  wd={r['well_dist']:.3f}" if r["well_dist"] else ""
        print(f"      t{r['turn']:2d}: {resp}{wd}")


def main():
    print("=" * 60)
    print("  EXPERIMENT: Orbital Dynamics of Brain-Model Coupling")
    print("=" * 60)

    brain = EngramFile.load(BRAIN_PATH)
    codec = Engram(dim=brain.dim)
    pairing = brain._reference_pairing
    print(f"brain: {brain._total_absorbed:,} articles, {len(brain.wells())} wells")

    print("loading GloVe...", flush=True)
    glove = load_glove()
    print()

    # run without brain
    print("--- 9B WITHOUT BRAIN (10 turns) ---")
    results_no_brain = run_conversation(False, glove, codec, pairing, brain)

    # run with brain
    print("\n--- 9B WITH BRAIN (10 turns) ---")
    results_brain = run_conversation(True, glove, codec, pairing, brain)

    # analysis
    print("\n" + "=" * 60)
    print("  ORBITAL ANALYSIS")
    print("=" * 60)

    analyze_trajectory(results_no_brain, "9B without brain")
    analyze_trajectory(results_brain, "9B with brain")

    # meta-eigenvalue: fit AR(2) to the well_dist sequence
    print("\n" + "=" * 60)
    print("  META-EIGENVALUE: dynamics of the coupling itself")
    print("=" * 60)

    for label, results in [("no brain", results_no_brain), ("with brain", results_brain)]:
        dists = [r["well_dist"] for r in results if r["well_dist"] is not None]
        if len(dists) >= 5:
            z = np.array([complex(d, 0) for d in dists])
            result = fit_pair(z)
            K, G = result.k, result.g

            # eigenvalues of the coupling
            disc = K * K - 4.0 * G
            l1 = (K + np.sqrt(disc)) / 2
            l2 = (K - np.sqrt(disc)) / 2

            print(f"\n  {label}:")
            print(f"    K = {K.real:.4f} + {K.imag:.4f}i")
            print(f"    G = {G.real:.4f} + {G.imag:.4f}i")
            print(f"    eigenvalue 1: |l1| = {abs(l1):.4f}  freq = {np.angle(l1):.4f} rad")
            print(f"    eigenvalue 2: |l2| = {abs(l2):.4f}  freq = {np.angle(l2):.4f} rad")

            if abs(l1) < 1 and abs(l2) < 1:
                print(f"    --> DAMPED: conversation converges (stable spiral)")
            elif abs(l1) > 1 or abs(l2) > 1:
                print(f"    --> DIVERGENT: conversation spirals outward")
            else:
                print(f"    --> SUSTAINED: conversation oscillates at constant amplitude")

    # save
    out_path = str(_HERE / "cache" / "_experiment_orbit.json")
    all_results = {
        "no_brain": results_no_brain,
        "with_brain": results_brain,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nraw results saved: {out_path}")


if __name__ == "__main__":
    main()
