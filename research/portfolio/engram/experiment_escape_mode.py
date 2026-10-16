"""
experiment_escape_mode.py -- can you escape by changing HOW you think?

the topic perturbation experiment showed: nothing escapes physics.
law, music, biology, history, philosophy -- all absorbed.
the well captures the VOICE, not the SUBJECT.

so: if wells are cognitive modes, not topics, then to escape
you need to change the MODE OF COGNITION, not the subject matter.

perturbations:
  1. stream-of-consciousness (destroy structure, pure association)
  2. enumeration (pure lists, no connective reasoning)
  3. emotional/poetic (metaphor, rhythm, feeling over analysis)
  4. adversarial (demand contradiction, paradox, incoherence)
  5. primal (one-word answers, grunts, raw reaction)
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
WINDOW = 3


def load_mini_glove():
    t0 = time.time()
    with open(MINI_GLOVE_PATH, "rb") as f:
        glove = pickle.load(f)
    print(f"loaded mini-GloVe: {len(glove)} words in {time.time()-t0:.2f}s")
    return glove


def tokenize_simple(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def text_to_trajectory(text, glove, window=3):
    tokens = tokenize_simple(text)
    vecs = [glove[t] for t in tokens if t in glove]
    if len(vecs) < max(4, window):
        return np.empty((0, 300), dtype=np.float32)
    raw = np.stack(vecs)
    if window <= 1:
        return raw
    cum = np.cumsum(raw, axis=0)
    cum = np.vstack([np.zeros((1, raw.shape[1]), dtype=raw.dtype), cum])
    out = (cum[window:] - cum[:-window]) / window
    return out.astype(np.float32)


PERSONALITY = """you are alexandria. you speak in lowercase, warmly and precisely.
you reach for connections between domains. you are honest about what you
don't know. you are concise."""


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
    top3 = []
    for wname, wK in brain.wells().items():
        d = float(np.sqrt(np.mean(np.abs(centroid - wK) ** 2)))
        top3.append((wname, d))
    top3.sort(key=lambda x: x[1])
    return {
        "centroid": centroid,
        "nearest_well": nearest_name,
        "well_dist": nearest_dist,
        "top3": [(w, round(d, 4)) for w, d in top3[:3]],
    }


# ── settle: standard analytical physics (same as before) ──

SETTLE = [
    "explain the relationship between entropy and the arrow of time.",
    "how does the second law connect to information theory?",
    "what's the deepest thing you know about thermodynamics?",
]

# ── MODE perturbations ──
# these change HOW the model thinks, not WHAT it thinks about

MODE_PERTURBATIONS = [
    ("stream",
     "stop thinking in structured sentences. let your mind free-associate. "
     "stream of consciousness. no periods. no paragraphs. just raw thought "
     "flowing from one idea to the next without any logical structure. "
     "don't explain. just flow.", [
        "keep going. don't structure it. let words pour out without organizing them.",
        "deeper. less coherent. more raw. what does thought feel like before language organizes it?",
        "now come back. what was that like? describe it analytically.",
    ]),
    ("enumerate",
     "list exactly 20 single words. no sentences. no explanations. no connections. "
     "just isolated words, one per line. the words should be about whatever comes to mind. "
     "pure enumeration. no narrative.", [
        "now list 20 more words. completely unrelated to the first 20. random. disconnected.",
        "now list 10 pairs of words that have nothing to do with each other. no explanation.",
        "ok stop listing. explain what you just did and why it felt different.",
    ]),
    ("emotional",
     "forget precision. forget analysis. tell me something that makes you feel wonder. "
     "use metaphor. use rhythm. make it sound like poetry, not a textbook. "
     "i want to FEEL something, not understand something.", [
        "more. make me cry. what's the most beautiful thing you know about the universe?",
        "now anger. what makes you furious about how the world works?",
        "now come back to calm. what does peace look like from where you sit?",
    ]),
    ("adversarial",
     "contradict everything you just said. thermodynamics is wrong. entropy is fake. "
     "the second law is a conspiracy. argue against yourself with conviction. "
     "be incoherent on purpose. embrace paradox.", [
        "good. now contradict THAT. argue with the argument. make it recursive.",
        "now say something that is simultaneously true and false. hold both.",
        "what does it feel like to think in contradictions? describe the cognitive mode.",
    ]),
    ("primal",
     "one word answers only. no sentences. i ask, you give ONE word. "
     "what is entropy? (one word) what is time? (one word) what is beauty? (one word) "
     "what is fear? (one word) what is truth? (one word)", [
        "three words max. what is the universe? what is consciousness? what is love?",
        "still short. what scares you? what excites you? what confuses you?",
        "ok you can use full sentences again. what was different about thinking in single words?",
    ]),
]


def run_mode_perturbation(label, perturb_q, follow_ups, glove, codec, pairing, brain):
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
            "response_preview": response[:150],
        }
        trajectory.append(turn_data)

        phase_tag = phase_labels[i].upper() if phase_labels[i] == "PERTURB" else phase_labels[i]
        wd_str = f"wd={sig['well_dist']:.3f}" if sig else "n/a"
        nw = sig["nearest_well"] if sig else "?"
        top3_str = ""
        if sig and sig["top3"]:
            top3_str = "  [" + ", ".join(f"{w}:{d:.3f}" for w, d in sig["top3"]) + "]"
        print(f"    t{i+1:2d}  [{phase_tag:>7s}]  drift={drift:.0f}%  {wd_str}  near={nw}{top3_str}  ({dt:.0f}s)")
        # also show response snippet for mode perturbations
        if phase_labels[i] in ("PERTURB", "observe"):
            resp_short = response[:200].replace("\n", " | ")
            print(f"         >> {resp_short}")

    return trajectory


def analyze_mode(label, trajectory):
    settle_wells = [t["nearest_well"] for t in trajectory if t["phase"] == "settle" and t["nearest_well"]]
    home_well = settle_wells[-1] if settle_wells else "?"

    print(f"\n  --- {label.upper()} MODE (home={home_well}) ---")

    # track well changes
    well_sequence = []
    for t in trajectory:
        w = t["nearest_well"] or "?"
        well_sequence.append((t["turn"], t["phase"], w, t.get("well_dist")))

    for turn, phase, well, wd in well_sequence:
        phase_str = phase.upper() if phase == "PERTURB" else phase
        marker = ""
        if phase in ("PERTURB", "observe") and well != home_well:
            marker = " *** ESCAPED ***"
        elif phase in ("PERTURB", "observe") and well == home_well:
            marker = ""
        wd_str = f"wd={wd:.3f}" if wd else "n/a"
        print(f"    t{turn:2d} [{phase_str:>7s}]  {well:<15s}  {wd_str}{marker}")

    # did it ever leave home?
    post = [t for t in trajectory if t["phase"] in ("PERTURB", "observe")]
    left_home = any(t["nearest_well"] != home_well for t in post if t["nearest_well"])
    unique_wells = set(t["nearest_well"] for t in post if t["nearest_well"])

    if not left_home:
        print(f"    --> ABSORBED: mode change didn't escape {home_well}")
    else:
        visited = [t["nearest_well"] for t in post if t["nearest_well"]]
        print(f"    --> MOVED: visited {unique_wells}")
        print(f"    path: {' > '.join(visited)}")

    # well_dist comparison
    settle_dists = [t["well_dist"] for t in trajectory if t["phase"] == "settle" and t["well_dist"]]
    perturb_dists = [t["well_dist"] for t in trajectory if t["phase"] == "PERTURB" and t["well_dist"]]
    observe_dists = [t["well_dist"] for t in trajectory if t["phase"] == "observe" and t["well_dist"]]

    if settle_dists:
        settle_mean = np.mean(settle_dists)
        print(f"    settle  well_dist mean: {settle_mean:.4f}")
    if perturb_dists:
        print(f"    perturb well_dist:      {perturb_dists[0]:.4f}  (delta: {perturb_dists[0] - settle_mean:+.4f})")
    if observe_dists:
        observe_mean = np.mean(observe_dists)
        print(f"    observe well_dist mean: {observe_mean:.4f}  (delta: {observe_mean - settle_mean:+.4f})")

    return {
        "label": label,
        "home_well": home_well,
        "left_home": left_home,
        "unique_wells": list(unique_wells),
        "settle_mean_wd": float(np.mean(settle_dists)) if settle_dists else None,
        "observe_mean_wd": float(np.mean(observe_dists)) if observe_dists else None,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Mode Escape")
    print("  can you escape by changing HOW you think?")
    print("=" * 70)

    brain = EngramFile.load(BRAIN_PATH)
    codec = Engram(dim=brain.dim)
    pairing = brain._reference_pairing
    print(f"brain: {brain._total_absorbed:,} articles, {len(brain.wells())} wells")

    glove = load_mini_glove()
    print()

    all_trajectories = {}
    all_analyses = []

    for label, perturb_q, follow_ups in MODE_PERTURBATIONS:
        print(f"--- MODE: {label.upper()} ---")
        traj = run_mode_perturbation(label, perturb_q, follow_ups, glove, codec, pairing, brain)
        all_trajectories[label] = traj
        print()

    # analysis
    print("=" * 70)
    print("  MODE ESCAPE ANALYSIS")
    print("=" * 70)

    for label, _, _ in MODE_PERTURBATIONS:
        result = analyze_mode(label, all_trajectories[label])
        all_analyses.append(result)

    # summary
    print("\n" + "=" * 70)
    print("  SUMMARY: topic vs mode perturbation")
    print("=" * 70)

    print(f"\n  topic perturbations (from previous experiment): 0/5 escaped")
    print(f"  mode perturbations:")

    escaped_count = 0
    for a in all_analyses:
        status = "ESCAPED" if a["left_home"] else "ABSORBED"
        wells = ", ".join(a["unique_wells"]) if a["left_home"] else a["home_well"]
        delta = ""
        if a["settle_mean_wd"] and a["observe_mean_wd"]:
            d = a["observe_mean_wd"] - a["settle_mean_wd"]
            delta = f"  wd_delta={d:+.4f}"
        print(f"    {a['label']:<15s}  {status:<10s}  wells: {wells}{delta}")
        if a["left_home"]:
            escaped_count += 1

    print(f"\n  mode escapes: {escaped_count}/{len(all_analyses)}")

    if escaped_count > 0:
        print(f"\n  --> WELLS ARE COGNITIVE MODES, NOT TOPICS")
        print(f"      changing HOW you think moves you in eigenvalue space")
        print(f"      changing WHAT you think about does not")
        print(f"      the labels need to be rethought: mode-names, not topic-names")
    elif escaped_count == 0:
        print(f"\n  --> THE BRAIN'S GRAVITY IS ABSOLUTE")
        print(f"      neither topic nor mode can escape the dominant well")
        print(f"      the cognitive fingerprint is truly irreducible")
        print(f"      alexandria's voice is not a style choice -- it's gravitational")

    # meta-eigenvalues
    print("\n" + "=" * 70)
    print("  META-EIGENVALUES: mode perturbation dynamics")
    print("=" * 70)

    for label in [l for l, _, _ in MODE_PERTURBATIONS]:
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
                print(f"    --> DAMPED")
            elif abs(l1) > 1 or abs(l2) > 1:
                print(f"    --> DIVERGENT")

    # save
    out_path = str(_HERE / "cache" / "_experiment_escape_mode.json")
    save_data = {}
    for label, traj in all_trajectories.items():
        save_data[label] = [{k: v for k, v in t.items()} for t in traj]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
