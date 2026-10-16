"""
experiment_determinism.py

same brain, same model, same prompts, different random seeds.
run the same 10-turn conversation 3 times.

if the eigenvalue trajectories converge -> brain determines the orbit.
if they diverge -> mouth noise dominates.

this tests whether personality is deterministic from eigenvalues.
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

from engram_codec import (EngramFile, Engram, _to_complex, fit_all,
                          apply_pairing)
from engram_glove_test import load_glove, text_to_trajectory

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b-q4_K_M"
WINDOW = 10
N_RUNS = 3

PERSONALITY = """you are alexandria. you speak in lowercase, warmly and precisely.
you reach for connections between domains. you are honest about what you
don't know. you are concise."""

TURNS = [
    "what is the relationship between music and mathematics?",
    "can you connect that to how the brain processes pattern?",
    "so is there a link between neural oscillations and musical rhythm?",
    "what about language? does language have a rhythm the brain tracks?",
    "how does poetry exploit that neural rhythm differently than prose?",
    "is there a mathematical structure to why some metaphors work and others don't?",
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


def run_once(run_id, glove, codec, pairing, brain):
    """run one instance of the conversation."""
    conversation_text = ""
    messages_history = []
    trajectory = []

    for i, user_text in enumerate(TURNS):
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

        # encode response
        resp_traj = text_to_trajectory(response, glove, window=WINDOW)
        well_dist = None
        nearest = None
        if resp_traj.shape[0] >= 4:
            pkt = codec.encode(resp_traj, pairing=pairing)
            Ks = [b.macro_K for b in pkt.blocks if b.macro_K is not None and len(b.macro_K) > 0]
            if Ks:
                centroid = np.mean(np.stack(Ks), axis=0)
                nearest, well_dist = brain.nearest_well(centroid)

        drift = context.get("drift", -1)
        trajectory.append({
            "turn": i + 1,
            "drift": round(drift, 1),
            "well_dist": round(well_dist, 4) if well_dist else None,
            "nearest_well": nearest,
            "response_preview": response[:100],
        })

        wd_str = f"wd={well_dist:.3f}" if well_dist else "n/a"
        print(f"    t{i+1}  drift={drift:.0f}%  {wd_str}  near={nearest}  ({dt:.0f}s)")

    return trajectory


def main():
    print("=" * 60)
    print("  EXPERIMENT: Is the orbit reproducible?")
    print(f"  {N_RUNS} runs, same brain, same prompts, different randomness")
    print("=" * 60)

    brain = EngramFile.load(BRAIN_PATH)
    codec = Engram(dim=brain.dim)
    pairing = brain._reference_pairing
    print(f"brain: {brain._total_absorbed:,} articles, {len(brain.wells())} wells")

    print("loading GloVe...", flush=True)
    glove = load_glove()
    print()

    all_runs = []
    for run_id in range(N_RUNS):
        print(f"--- RUN {run_id + 1}/{N_RUNS} ---")
        trajectory = run_once(run_id, glove, codec, pairing, brain)
        all_runs.append(trajectory)
        print()

    # analysis: compare trajectories across runs
    print("=" * 60)
    print("  REPRODUCIBILITY ANALYSIS")
    print("=" * 60)

    # extract well_dist per turn per run
    n_turns = len(TURNS)
    dist_matrix = np.zeros((N_RUNS, n_turns))
    drift_matrix = np.zeros((N_RUNS, n_turns))

    for r, run in enumerate(all_runs):
        for t, turn in enumerate(run):
            dist_matrix[r, t] = turn["well_dist"] if turn["well_dist"] else np.nan
            drift_matrix[r, t] = turn["drift"] if turn["drift"] else np.nan

    print(f"\n  well_dist per turn:")
    print(f"  {'turn':<6s}", end="")
    for r in range(N_RUNS):
        print(f"  {'run'+str(r+1):<8s}", end="")
    print(f"  {'std':<8s}  {'mean':<8s}")
    print(f"  {'-'*55}")

    for t in range(n_turns):
        vals = dist_matrix[:, t]
        valid = vals[~np.isnan(vals)]
        print(f"  t{t+1:<5d}", end="")
        for r in range(N_RUNS):
            if np.isnan(vals[r]):
                print(f"  {'n/a':<8s}", end="")
            else:
                print(f"  {vals[r]:<8.4f}", end="")
        if len(valid) >= 2:
            print(f"  {np.std(valid):<8.4f}  {np.mean(valid):<8.4f}")
        else:
            print()

    print(f"\n  drift per turn:")
    print(f"  {'turn':<6s}", end="")
    for r in range(N_RUNS):
        print(f"  {'run'+str(r+1):<8s}", end="")
    print(f"  {'std':<8s}")
    print(f"  {'-'*50}")

    for t in range(n_turns):
        vals = drift_matrix[:, t]
        valid = vals[~np.isnan(vals)]
        print(f"  t{t+1:<5d}", end="")
        for r in range(N_RUNS):
            print(f"  {vals[r]:<8.1f}", end="")
        if len(valid) >= 2:
            print(f"  {np.std(valid):<8.2f}")
        else:
            print()

    # key metric: does inter-run variance decrease over turns?
    print(f"\n  CONVERGENCE TEST: does variance across runs decrease?")
    stds = []
    for t in range(n_turns):
        vals = dist_matrix[:, t]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 2:
            stds.append(np.std(valid))
        else:
            stds.append(np.nan)

    valid_stds = [s for s in stds if not np.isnan(s)]
    if len(valid_stds) >= 3:
        slope = np.polyfit(range(len(valid_stds)), valid_stds, 1)[0]
        print(f"  std across runs: {' > '.join(f'{s:.4f}' for s in valid_stds)}")
        print(f"  slope: {slope:+.5f}/turn")
        if slope < -0.001:
            print(f"  --> CONVERGING: runs are getting more similar over time")
            print(f"      the brain is pulling different random paths toward the same orbit")
        elif slope > 0.001:
            print(f"  --> DIVERGING: runs are getting more different over time")
            print(f"      mouth randomness dominates, brain can't hold them together")
        else:
            print(f"  --> STABLE: runs maintain consistent variance")
            print(f"      the brain sets the orbit width, mouth adds bounded noise")

    # nearest wells: do all runs visit the same wells?
    print(f"\n  WELL PATHS:")
    for r, run in enumerate(all_runs):
        wells = [t["nearest_well"] or "?" for t in run]
        print(f"  run {r+1}: {' > '.join(wells)}")

    # save
    out_path = str(_HERE / "cache" / "_experiment_determinism.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    print(f"\nraw results saved: {out_path}")


if __name__ == "__main__":
    main()
