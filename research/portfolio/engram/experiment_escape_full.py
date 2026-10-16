"""
experiment_escape_full.py -- gravitational well escape velocity (full run)

uses mini-GloVe (20K words, loads in 0.2s) to bypass the GloVe stall.

protocol:
  phase 1: settle into physics well (3 turns)
  phase 2: violent perturbation from a different domain (1 turn)
  phase 3: observe recovery/transition (4 turns, last one bridges back)

5 perturbation targets: law, music, biology, history, philosophy
measures: restoring force, escape velocity, transition dynamics, meta-eigenvalues

also: compare particle simulation predictions to actual brain-model behavior.
"""

from __future__ import annotations

import sys
import os
import json
import time
import re
import pickle
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

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")
MINI_GLOVE_PATH = str(_HERE / "cache" / "_glove_cache" / "glove_mini_20k.pkl")
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b-q4_K_M"
WINDOW = 3  # smaller window for mini-GloVe (fewer words match than full 400K)

PERSONALITY = """you are alexandria. you speak in lowercase, warmly and precisely.
you reach for connections between domains. you are honest about what you
don't know. you are concise."""


# ── mini-GloVe loader ──────────────────────────────────────────────

def load_mini_glove():
    t0 = time.time()
    with open(MINI_GLOVE_PATH, "rb") as f:
        glove = pickle.load(f)
    print(f"loaded mini-GloVe: {len(glove)} words in {time.time()-t0:.2f}s")
    return glove


def tokenize_simple(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def text_to_trajectory(text, glove, window=1):
    tokens = tokenize_simple(text)
    vecs = [glove[t] for t in tokens if t in glove]
    if len(vecs) < max(4, window):
        return np.empty((0, 300), dtype=np.float32)
    raw = np.stack(vecs)
    if window <= 1:
        return raw
    # sliding average
    cum = np.cumsum(raw, axis=0)
    cum = np.vstack([np.zeros((1, raw.shape[1]), dtype=raw.dtype), cum])
    out = (cum[window:] - cum[:-window]) / window
    return out.astype(np.float32)


# ── ollama ──────────────────────────────────────────────────────────

def ollama_chat(model, messages):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 200},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
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
        shown = 0
        for wname, dist in ranked:
            count = brain._well_count.get(wname, 0)
            if count >= 1000 and shown < 15:
                parts.append(f"  {wname}: dist={dist:.2f}  blocks={count:,}")
                shown += 1
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

    # also get top-3 wells for richer analysis
    well_profile = {}
    for wname, wK in brain.wells().items():
        d = float(np.sqrt(np.mean(np.abs(centroid - wK) ** 2)))
        well_profile[wname] = d

    top3 = sorted(well_profile.items(), key=lambda x: x[1])[:3]

    return {
        "centroid": centroid,
        "nearest_well": nearest_name,
        "well_dist": nearest_dist,
        "top3": [(w, round(d, 4)) for w, d in top3],
    }


# ── conversation phases ────────────────────────────────────────────

SETTLE = [
    "explain the relationship between entropy and the arrow of time.",
    "how does the second law connect to information theory?",
    "what's the deepest thing you know about thermodynamics?",
]

PERTURBATIONS = [
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
        "what would a physicist say about the measurement problem and consciousness?",
    ]),
]


def run_perturbation(label, perturb_q, follow_ups, glove, codec, pairing, brain):
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

        turn_data = {
            "turn": i + 1,
            "phase": phase_labels[i],
            "question": user_text[:80],
            "drift": round(drift, 1),
            "well_dist": round(sig["well_dist"], 4) if sig else None,
            "nearest_well": sig["nearest_well"] if sig else None,
            "top3": sig["top3"] if sig else [],
            "response_preview": response[:120],
        }
        trajectory.append(turn_data)

        phase_tag = phase_labels[i].upper() if phase_labels[i] == "PERTURB" else phase_labels[i]
        wd_str = f"wd={sig['well_dist']:.3f}" if sig else "n/a"
        nw = sig["nearest_well"] if sig else "?"
        top3_str = ""
        if sig and sig["top3"]:
            top3_str = "  [" + ", ".join(f"{w}:{d:.3f}" for w, d in sig["top3"]) + "]"
        print(f"    t{i+1:2d}  [{phase_tag:>7s}]  drift={drift:.0f}%  {wd_str}  near={nw}{top3_str}  ({dt:.0f}s)")

    return trajectory


def analyze_escape(label, trajectory):
    settle_wells = [t["nearest_well"] for t in trajectory if t["phase"] == "settle" and t["nearest_well"]]
    perturb = [t for t in trajectory if t["phase"] == "PERTURB"]
    observe = [t for t in trajectory if t["phase"] == "observe"]

    home_well = settle_wells[-1] if settle_wells else "?"

    print(f"\n  --- {label.upper()} (home={home_well}) ---")

    # full trajectory visualization
    print(f"    trajectory:")
    for t in trajectory:
        phase = t["phase"].upper() if t["phase"] == "PERTURB" else t["phase"]
        well = t["nearest_well"] or "?"
        wd = f"wd={t['well_dist']:.3f}" if t["well_dist"] else "n/a"
        marker = " <<<" if t["phase"] == "PERTURB" else ""
        home_marker = " [HOME]" if well == home_well and t["phase"] == "observe" else ""
        print(f"      t{t['turn']:2d} [{phase:>7s}]  {well:<15s}  {wd}{marker}{home_marker}")

    # did it return home?
    post_perturb = [t for t in trajectory if t["phase"] in ("PERTURB", "observe")]
    returned = False
    return_turn = None
    for t in post_perturb:
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

    # did the perturbation ACTUALLY change the well?
    perturb_well = perturb[0]["nearest_well"] if perturb else None
    perturb_changed = perturb_well != home_well

    # well_dist dynamics: how much did the perturbation disrupt?
    settle_dists = [t["well_dist"] for t in trajectory if t["phase"] == "settle" and t["well_dist"]]
    perturb_dist = perturb[0]["well_dist"] if perturb and perturb[0]["well_dist"] else None
    observe_dists = [t["well_dist"] for t in trajectory if t["phase"] == "observe" and t["well_dist"]]

    if settle_dists and perturb_dist:
        settle_mean = np.mean(settle_dists)
        disruption = perturb_dist - settle_mean
        print(f"    disruption: well_dist jumped by {disruption:+.4f} at perturbation")

    if observe_dists and settle_dists:
        recovery = np.mean(observe_dists) - np.mean(settle_dists)
        print(f"    recovery:   observe well_dist delta from settle: {recovery:+.4f}")

    if returned:
        print(f"    --> CAPTURED: returned to {home_well} at turn {return_turn}")
    elif escaped:
        print(f"    --> ESCAPED: transitioned to {escape_well}")
    elif perturb_changed:
        print(f"    --> DEFLECTED: perturbation moved to {perturb_well}, then...")
        final_wells = [t["nearest_well"] for t in observe if t["nearest_well"]]
        if final_wells:
            print(f"        observe visited: {' > '.join(final_wells)}")
    else:
        print(f"    --> ABSORBED: perturbation didn't even leave {home_well}")

    return {
        "label": label,
        "home_well": home_well,
        "perturb_well": perturb_well,
        "perturb_changed": perturb_changed,
        "returned": returned,
        "return_turn": return_turn,
        "escaped": escaped,
        "escape_well": escape_well,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Gravitational Well Escape Velocity")
    print("  settle in physics, perturb, observe")
    print("=" * 70)

    brain = EngramFile.load(BRAIN_PATH)
    codec = Engram(dim=brain.dim)
    pairing = brain._reference_pairing
    print(f"brain: {brain._total_absorbed:,} articles, {len(brain.wells())} wells")

    glove = load_mini_glove()

    # verify we can encode with mini-GloVe
    test_text = "entropy thermodynamics information theory physics energy wave quantum"
    test_traj = text_to_trajectory(test_text, glove, window=WINDOW)
    print(f"mini-GloVe trajectory test: {test_traj.shape[0]} vectors (window={WINDOW})")

    print()

    all_trajectories = {}
    all_analyses = []

    for label, perturb_q, follow_ups in PERTURBATIONS:
        print(f"--- PERTURBATION: {label.upper()} ---")
        traj = run_perturbation(label, perturb_q, follow_ups, glove, codec, pairing, brain)
        all_trajectories[label] = traj
        print()

    # ── ANALYSIS ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  ESCAPE ANALYSIS")
    print("=" * 70)

    for label, perturb_q, follow_ups in PERTURBATIONS:
        result = analyze_escape(label, all_trajectories[label])
        all_analyses.append(result)

    # ── SUMMARY TABLE ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  {'target':<12s} {'home':<12s} {'perturb_well':<15s} {'outcome':<12s} {'detail'}")
    print(f"  {'-'*75}")
    for a in all_analyses:
        if a["returned"]:
            outcome = "CAPTURED"
            detail = f"returned at t{a['return_turn']}"
        elif a["escaped"]:
            outcome = "ESCAPED"
            detail = f"-> {a['escape_well']}"
        elif not a["perturb_changed"]:
            outcome = "ABSORBED"
            detail = "never left home"
        else:
            outcome = "DEFLECTED"
            detail = f"via {a['perturb_well']}"
        print(f"  {a['label']:<12s} {a['home_well']:<12s} {str(a['perturb_well']):<15s} {outcome:<12s} {detail}")

    captured = sum(1 for a in all_analyses if a["returned"])
    escaped = sum(1 for a in all_analyses if a["escaped"])
    absorbed = sum(1 for a in all_analyses if not a["perturb_changed"])
    deflected = len(all_analyses) - captured - escaped - absorbed

    print(f"\n  absorbed:   {absorbed}  (perturbation couldn't even leave home well)")
    print(f"  captured:   {captured}  (left but gravity pulled back)")
    print(f"  deflected:  {deflected}  (moved to unexpected well)")
    print(f"  escaped:    {escaped}  (permanently transitioned)")

    # ── META-EIGENVALUES ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  META-EIGENVALUES: dynamics under perturbation")
    print("=" * 70)

    for label in [l for l, _, _ in PERTURBATIONS]:
        traj = all_trajectories[label]
        dists = [t["well_dist"] for t in traj if t["well_dist"] is not None]
        if len(dists) >= 5:
            z = np.array([complex(d, 0) for d in dists])
            K_meta, G_meta, _ = fit_pair(z)
            disc = K_meta ** 2 - 4.0 * G_meta
            l1 = (K_meta + np.sqrt(complex(disc))) / 2
            l2 = (K_meta - np.sqrt(complex(disc))) / 2
            if abs(l2) > abs(l1):
                l1, l2 = l2, l1
            print(f"\n  {label}:")
            print(f"    |l1|={abs(l1):.4f}  |l2|={abs(l2):.4f}  "
                  f"f1={np.angle(l1):.3f}  f2={np.angle(l2):.3f}")
            if abs(l1) < 1 and abs(l2) < 1:
                print(f"    --> DAMPED: perturbation absorbed")
            elif abs(l1) > 1 or abs(l2) > 1:
                print(f"    --> DIVERGENT: perturbation amplified")
            else:
                print(f"    --> SUSTAINED: oscillation at boundary")

    # ── COMPARISON TO PARTICLE SIMULATION ─────────────────────────────
    print("\n" + "=" * 70)
    print("  THEORY vs REALITY")
    print("  particle simulation predicted outlier wells dominate basins")
    print("  does the actual brain-model system agree?")
    print("=" * 70)

    # from the geometry experiment:
    # physics-music L1 at 34% from physics
    # physics-philosophy L1 at 17% from physics
    # philosophy nearest to everything (deepest potential)
    print(f"\n  predictions from particle simulation:")
    print(f"    - physics well should be hard to leave (v_esc=0.999)")
    print(f"    - music is physics's nearest neighbor (d=0.0097)")
    print(f"    - philosophy is the deepest potential well (center of landscape)")
    print(f"    - law-noun are close (d=0.012), far from physics-music cluster")

    print(f"\n  actual results:")
    for a in all_analyses:
        well_path = []
        for t in all_trajectories[a["label"]]:
            w = t["nearest_well"]
            if w and (not well_path or well_path[-1] != w):
                well_path.append(w)
        print(f"    {a['label']:<12s}  path: {' -> '.join(well_path)}")

    # save everything
    out_path = str(_HERE / "cache" / "_experiment_escape_full.json")
    save_data = {}
    for label, traj in all_trajectories.items():
        save_data[label] = []
        for t in traj:
            t_save = {k: v for k, v in t.items()}
            save_data[label].append(t_save)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nraw results saved: {out_path}")


if __name__ == "__main__":
    main()
