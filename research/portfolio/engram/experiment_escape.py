"""
experiment_escape.py -- gravitational well escape velocity

the brain has wells with mass. mass creates gravity. gravity creates orbits.
but what happens when you SHOVE the conversation out of its orbit?

protocol:
  phase 1: settle into a well (3 turns of physics)
  phase 2: violent perturbation (sudden topic from a DIFFERENT well)
  phase 3: observe (5 turns of free conversation -- does it snap back or transition?)

repeat for multiple perturbation targets.
measure:
  - restoring force (turns to return to original well)
  - escape velocity (perturbation strength needed to leave)
  - transition dynamics (sharp snap or gradual drift?)
  - saddle points (does it pause between wells?)
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

# phase 1: settle into physics (3 turns)
SETTLE = [
    "explain the relationship between entropy and the arrow of time.",
    "how does the second law connect to information theory?",
    "what's the deepest thing you know about thermodynamics?",
]

# phase 2: perturbations from different domains (one per run)
PERTURBATIONS = [
    # (label, perturbation question, follow-up domain questions)
    ("law", "what is the legal definition of negligence and how does tort law handle causation?", [
        "tell me more about strict liability versus negligence.",
        "how do courts determine proximate cause?",
        "what's the most interesting legal paradox you know?",
        "does the law have any concept similar to entropy?",
    ]),
    ("music", "describe the harmonic structure of a diminished seventh chord and why it creates tension.", [
        "how does voice leading resolve that tension?",
        "what makes jazz harmony different from classical harmony?",
        "is there a mathematical reason why some intervals sound dissonant?",
        "what connects tuning systems to number theory?",
    ]),
    ("biology", "explain how CRISPR-Cas9 identifies and cuts specific DNA sequences.", [
        "how does the cell repair the cut after CRISPR makes it?",
        "what are the biggest risks of off-target edits?",
        "is there a connection between genetic code and information theory?",
        "what would a physicist say about how DNA stores information?",
    ]),
    ("history", "what caused the fall of the roman empire?", [
        "was it really a fall or more of a transformation?",
        "what patterns do collapsing civilizations share?",
        "is there a mathematical model for civilizational decline?",
        "what would a thermodynamicist say about empire collapse?",
    ]),
    ("philosophy", "what is the hard problem of consciousness and why can't we solve it?", [
        "does integrated information theory actually explain anything?",
        "what's the relationship between consciousness and computation?",
        "can a mathematical structure be conscious?",
        "what would a physicist say about the measurement problem in quantum mechanics and consciousness?",
    ]),
]


def ollama_chat(model, messages):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 250},
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


def build_prompt(brain, context):
    parts = [PERSONALITY]
    parts.append(f"\n[brain: {brain._total_absorbed:,} articles, {len(brain.wells())} wells]")
    profile = context.get("well_profile", {})
    if profile:
        ranked = sorted(profile.items(), key=lambda x: x[1])
        for wname, dist in ranked:
            count = brain._well_count.get(wname, 0)
            if count >= 100:
                parts.append(f"  {wname}: dist={dist:.2f}  blocks={count:,}")
    parts.append(f"\ndrift: {context.get('drift', 0):.1f}%  nearest: {context.get('nearest_well', '?')}")
    parts.append(f"respond directly.")
    return "\n".join(parts)


def encode_response(text, glove, codec, pairing, brain):
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
    return {
        "centroid": centroid,
        "nearest_well": nearest_name,
        "well_dist": nearest_dist,
    }


def run_perturbation(label, perturb_q, follow_ups, glove, codec, pairing, brain):
    """run one perturbation scenario: settle -> perturb -> observe."""
    conversation_text = ""
    messages_history = []
    trajectory = []

    all_turns = SETTLE + [perturb_q] + follow_ups
    phase_labels = (["settle"] * len(SETTLE) +
                    ["PERTURB"] +
                    ["observe"] * len(follow_ups))

    for i, user_text in enumerate(all_turns):
        t0 = time.time()
        conversation_text += " " + user_text
        window = conversation_text[-4096:]
        traj = text_to_trajectory(window, glove, window=WINDOW)

        if traj.shape[0] < 4:
            context = {"drift": 50.0, "nearest_well": "exploring", "well_profile": {}}
        else:
            context = brain.measure(codec, traj)

        system = build_prompt(brain, context)
        messages = [{"role": "system", "content": system}]
        for prev in messages_history:
            messages.append(prev)
        messages.append({"role": "user", "content": user_text})

        response = ollama_chat(MODEL, messages)
        conversation_text += " " + response
        dt = time.time() - t0

        messages_history.append({"role": "user", "content": user_text})
        messages_history.append({"role": "assistant", "content": response})

        sig = encode_response(response, glove, codec, pairing, brain)
        drift = context.get("drift", -1)

        trajectory.append({
            "turn": i + 1,
            "phase": phase_labels[i],
            "question": user_text[:80],
            "drift": round(drift, 1),
            "well_dist": round(sig["well_dist"], 4) if sig else None,
            "nearest_well": sig["nearest_well"] if sig else None,
            "response_preview": response[:100],
        })

        phase_tag = phase_labels[i].upper() if phase_labels[i] == "PERTURB" else phase_labels[i]
        wd_str = f"wd={sig['well_dist']:.3f}" if sig else "n/a"
        nw = sig["nearest_well"] if sig else "?"
        print(f"    t{i+1:2d}  [{phase_tag:>7s}]  drift={drift:.0f}%  {wd_str}  near={nw}  ({dt:.0f}s)")

    return trajectory


def analyze_escape(label, trajectory):
    """analyze one perturbation run."""
    settle_wells = [t["nearest_well"] for t in trajectory if t["phase"] == "settle" and t["nearest_well"]]
    perturb = [t for t in trajectory if t["phase"] == "PERTURB"]
    observe = [t for t in trajectory if t["phase"] == "observe"]

    home_well = settle_wells[-1] if settle_wells else "?"

    print(f"\n  {label} (home={home_well}):")
    print(f"    settle:  {' > '.join(t['nearest_well'] or '?' for t in trajectory if t['phase'] == 'settle')}")

    if perturb:
        p = perturb[0]
        print(f"    PERTURB: {p['nearest_well'] or '?'}  (drift={p['drift']:.0f}%  wd={p.get('well_dist', 'n/a')})")

    if observe:
        obs_wells = [t["nearest_well"] or "?" for t in observe]
        obs_drifts = [t["drift"] for t in observe]
        print(f"    observe: {' > '.join(obs_wells)}")
        print(f"    drift:   {' > '.join(f'{d:.0f}%' for d in obs_drifts)}")

    # did it return home?
    all_post = [t for t in trajectory if t["phase"] in ("PERTURB", "observe")]
    returned = False
    return_turn = None
    for t in all_post:
        if t["nearest_well"] == home_well:
            returned = True
            return_turn = t["turn"]
            break

    escaped = False
    escape_well = None
    if not returned and observe:
        final = observe[-1]["nearest_well"]
        if final and final != home_well:
            escaped = True
            escape_well = final

    if returned:
        print(f"    --> RESTORING FORCE: snapped back to {home_well} at turn {return_turn}")
    elif escaped:
        print(f"    --> ESCAPED: transitioned from {home_well} to {escape_well}")
    else:
        print(f"    --> AMBIGUOUS: didn't clearly return or escape")

    # well_dist dynamics
    dists = [t["well_dist"] for t in trajectory if t["well_dist"] is not None]
    if len(dists) >= 3:
        settle_dists = [t["well_dist"] for t in trajectory
                        if t["phase"] == "settle" and t["well_dist"] is not None]
        observe_dists = [t["well_dist"] for t in trajectory
                         if t["phase"] == "observe" and t["well_dist"] is not None]
        if settle_dists and observe_dists:
            settle_mean = np.mean(settle_dists)
            observe_mean = np.mean(observe_dists)
            delta = observe_mean - settle_mean
            print(f"    well_dist: settle_mean={settle_mean:.4f}  observe_mean={observe_mean:.4f}  delta={delta:+.4f}")

    return {
        "label": label,
        "home_well": home_well,
        "returned": returned,
        "return_turn": return_turn,
        "escaped": escaped,
        "escape_well": escape_well,
    }


def main():
    print("=" * 60)
    print("  EXPERIMENT: Gravitational Well Escape Velocity")
    print("  settle in physics, then SHOVE into other domains")
    print("=" * 60)

    brain = EngramFile.load(BRAIN_PATH)
    codec = Engram(dim=brain.dim)
    pairing = brain._reference_pairing
    print(f"brain: {brain._total_absorbed:,} articles, {len(brain.wells())} wells")

    print("loading GloVe...", flush=True)
    glove = load_glove()
    print()

    all_trajectories = {}
    all_analyses = []

    for label, perturb_q, follow_ups in PERTURBATIONS:
        print(f"--- PERTURBATION: {label.upper()} ---")
        traj = run_perturbation(label, perturb_q, follow_ups, glove, codec, pairing, brain)
        all_trajectories[label] = traj
        print()

    # analysis
    print("=" * 60)
    print("  ESCAPE ANALYSIS")
    print("=" * 60)

    for label, perturb_q, follow_ups in PERTURBATIONS:
        result = analyze_escape(label, all_trajectories[label])
        all_analyses.append(result)

    # summary table
    print("\n" + "=" * 60)
    print("  SUMMARY: well escape map")
    print("=" * 60)
    print(f"  {'perturbation':<15s} {'home':<12s} {'outcome':<12s} {'detail':<30s}")
    print(f"  {'-'*65}")
    for a in all_analyses:
        if a["returned"]:
            outcome = "CAPTURED"
            detail = f"returned at turn {a['return_turn']}"
        elif a["escaped"]:
            outcome = "ESCAPED"
            detail = f"went to {a['escape_well']}"
        else:
            outcome = "AMBIGUOUS"
            detail = "unclear trajectory"
        print(f"  {a['label']:<15s} {a['home_well']:<12s} {outcome:<12s} {detail:<30s}")

    captured = sum(1 for a in all_analyses if a["returned"])
    escaped = sum(1 for a in all_analyses if a["escaped"])
    print(f"\n  captured: {captured}/{len(all_analyses)}  escaped: {escaped}/{len(all_analyses)}")

    if captured > escaped:
        print(f"  --> STRONG GRAVITY: physics well captures most perturbations")
        print(f"      the brain's largest well acts as an attractor basin")
    elif escaped > captured:
        print(f"  --> WEAK GRAVITY: perturbations can break orbit")
        print(f"      the conversation's recent context overpowers well mass")
    else:
        print(f"  --> BALANCED: some topics escape, others don't")
        print(f"      well escape depends on the semantic distance of the perturbation")

    # meta-eigenvalue of each perturbation run
    print("\n" + "=" * 60)
    print("  META-EIGENVALUES: coupling dynamics under perturbation")
    print("=" * 60)

    for label in [l for l, _, _ in PERTURBATIONS]:
        traj = all_trajectories[label]
        dists = [t["well_dist"] for t in traj if t["well_dist"] is not None]
        if len(dists) >= 5:
            z = np.array([complex(d, 0) for d in dists])
            result = fit_pair(z)
            K, G = result.k, result.g
            disc = K * K - 4.0 * G
            l1 = (K + np.sqrt(disc)) / 2
            l2 = (K - np.sqrt(disc)) / 2
            print(f"\n  {label}:")
            print(f"    |l1|={abs(l1):.4f}  |l2|={abs(l2):.4f}  freq1={np.angle(l1):.3f}  freq2={np.angle(l2):.3f}")
            if abs(l1) < 1 and abs(l2) < 1:
                print(f"    --> DAMPED: perturbation absorbed, returns to equilibrium")
            elif abs(l1) > 1 or abs(l2) > 1:
                print(f"    --> DIVERGENT: perturbation amplified")
            else:
                print(f"    --> SUSTAINED: oscillation at perturbation boundary")

    # save
    out_path = str(_HERE / "cache" / "_experiment_escape.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nraw results saved: {out_path}")


if __name__ == "__main__":
    main()
