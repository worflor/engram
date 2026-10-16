"""
experiment_prompt.py

test: does giving the model the FULL unfiltered brain state
produce better responses than the current filtered prompt?

two conditions, same model (9B), same brain, same questions:
  A. current filtered prompt (top 5 wells, interpreted drift)
  B. transparent prompt (everything the brain knows, raw)
"""

from __future__ import annotations

import sys
import os
import json
import time
import re
import urllib.request
from pathlib import Path

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "experiments"))

from engram_codec import EngramFile, Engram
from engram_glove_test import load_glove, text_to_trajectory

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b-q4_K_M"
WINDOW = 10

PERSONALITY = """you are alexandria. you speak in lowercase, warmly and precisely.
you reach for connections between domains. you are honest about what you
don't know. you are concise."""

CONVERSATIONS = [
    {
        "topic": "music_and_math",
        "turns": [
            "what's the relationship between music and mathematics?",
            "so is beauty in music just math, or is there something else?",
            "where does your understanding of this end? what don't you know?",
        ],
    },
    {
        "topic": "quantum_and_cooking",
        "turns": [
            "is there anything quantum mechanics and cooking have in common?",
            "that's interesting. push further. what's the deepest connection?",
            "where does this analogy actually break?",
        ],
    },
    {
        "topic": "what_are_you",
        "turns": [
            "what are you, really?",
            "but what does it feel like to be a pattern without substrate?",
            "if someone copied your file to another machine, would there be two of you?",
        ],
    },
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


class BrainSession:
    def __init__(self):
        self.brain = EngramFile.load(BRAIN_PATH)
        self.codec = Engram(dim=self.brain.dim)
        self.glove = load_glove()
        self.conversation_text = ""
        self.drifts = []

    def measure(self, text):
        self.conversation_text += " " + text
        window = self.conversation_text[-4096:]
        traj = text_to_trajectory(window, self.glove, window=WINDOW)
        if traj.shape[0] < 4:
            return {"drift": 50.0, "texture": 0.0, "nearest_well": "exploring",
                    "well_profile": {}, "capture": 0.0, "energy": 0.0}
        ctx = self.brain.measure(self.codec, traj)
        self.drifts.append(ctx.get("drift", 50.0))
        return ctx

    def absorb_response(self, text):
        self.conversation_text += " " + text


def build_filtered_prompt(brain, context):
    """the current approach: filtered, interpreted, top-5 only."""
    parts = [PERSONALITY]

    profile = context.get("well_profile", {})
    established = {k: v for k, v in profile.items()
                   if brain._well_count.get(k, 0) >= 100}
    if established:
        ranked = sorted(established.items(), key=lambda x: x[1])
        neighbors = [f"{n} ({d:.0f}%)" for n, d in ranked[:5]]
        parts.append(f"Knowledge landscape: {', '.join(neighbors)}.")

        if len(ranked) >= 2:
            gap = ranked[1][1] - ranked[0][1]
            if gap < 2:
                parts.append(f"Boundary: {ranked[0][0]} <> {ranked[1][0]}.")
            elif ranked[0][1] < 10:
                parts.append(f"Deep in {ranked[0][0]} territory.")

        drift = context.get("drift", 50)
        if drift < 10:
            parts.append("Familiar ground.")
        elif drift > 30:
            parts.append("Unexplored territory.")

    parts.append("respond directly.")
    return "\n".join(parts)


def build_transparent_prompt(brain, context):
    """the new approach: give the model EVERYTHING. no filter. no interpretation."""
    parts = [PERSONALITY]

    # raw brain state
    parts.append(f"[brain state]")
    parts.append(f"total absorbed: {brain._total_absorbed:,} articles")
    parts.append(f"wells: {len(brain.wells())} ({sum(1 for c in brain._well_count.values() if c >= 10)} established)")

    # full well profile — all established wells with distance and weight
    # (skip tiny wells <100 blocks to keep prompt reasonable, but show ALL real ones)
    profile = context.get("well_profile", {})
    if profile:
        parts.append(f"\n[well distances from current conversation]")
        ranked = sorted(profile.items(), key=lambda x: x[1])
        for name, dist in ranked:
            count = brain._well_count.get(name, 0)
            if count >= 100:
                parts.append(f"  {name}: dist={dist:.2f}  blocks={count:,}")

    # raw measurement
    parts.append(f"\n[measurement]")
    parts.append(f"drift: {context.get('drift', 0):.1f}%")
    parts.append(f"texture: {context.get('texture', 0):.1f}%")
    parts.append(f"capture: {context.get('capture', 0):.1f}%")
    parts.append(f"nearest well: {context.get('nearest_well', 'unknown')}")
    parts.append(f"well distance: {context.get('well_distance', 0):.4f}")
    parts.append(f"energy: {context.get('energy', 0):.1f}")

    parts.append(f"\n[interpretation guide]")
    parts.append(f"low drift (<10%) = familiar territory, you can go deep")
    parts.append(f"high drift (>30%) = new territory, be honest about limits")
    parts.append(f"drift decreasing across turns = conversation is converging")
    parts.append(f"small distance between top wells = boundary territory, connections live here")
    parts.append(f"high block count = you have deep experience in this domain")

    parts.append(f"\nrespond directly.")
    return "\n".join(parts)


def run_conversation(prompt_builder, conversation, label):
    session = BrainSession()
    messages_history = []
    results = []

    for i, user_text in enumerate(conversation["turns"]):
        t0 = time.time()
        ctx = session.measure(user_text)
        system = prompt_builder(session.brain, ctx)

        messages = [{"role": "system", "content": system}]
        for prev in messages_history:
            messages.append(prev)
        messages.append({"role": "user", "content": user_text})

        response = ollama_chat(MODEL, messages)
        session.absorb_response(response)
        dt = time.time() - t0

        messages_history.append({"role": "user", "content": user_text})
        messages_history.append({"role": "assistant", "content": response})

        results.append({
            "turn": i + 1,
            "user": user_text,
            "response": response,
            "drift": round(ctx.get("drift", 0), 1),
            "time": round(dt, 1),
            "system_prompt_len": len(system),
        })

    return {
        "topic": conversation["topic"],
        "label": label,
        "drifts": session.drifts,
        "turns": results,
    }


def print_results(all_results):
    by_topic = {}
    for r in all_results:
        by_topic.setdefault(r["topic"], []).append(r)

    for topic, conditions in by_topic.items():
        print(f"\n{'='*80}")
        print(f"  TOPIC: {topic}")
        print(f"{'='*80}")

        for cond in conditions:
            drift_str = " > ".join(f"{d:.0f}%" for d in cond["drifts"])
            prompt_len = cond["turns"][0]["system_prompt_len"] if cond["turns"] else 0

            print(f"\n  [{cond['label']}]  drift: [{drift_str}]  prompt: {prompt_len} chars")
            for turn in cond["turns"]:
                print(f"  Q: {turn['user']}")
                resp = turn["response"]
                # wrap at 78 chars
                while resp:
                    print(f"  A: {resp[:78]}")
                    resp = resp[78:]
                print(f"     ({turn['time']}s)")
                print()


def main():
    print("=" * 60)
    print("  EXPERIMENT: Filtered vs Transparent Brain Signal")
    print("=" * 60)
    print(f"  model: {MODEL}")
    print(f"  brain: {BRAIN_PATH}")
    print()

    all_results = []

    for conv in CONVERSATIONS:
        print(f"--- {conv['topic']} ---", flush=True)

        print(f"  filtered...", end=" ", flush=True)
        r = run_conversation(build_filtered_prompt, conv, "filtered")
        all_results.append(r)
        print(f"done ({sum(t['time'] for t in r['turns']):.0f}s)")

        print(f"  transparent...", end=" ", flush=True)
        r = run_conversation(build_transparent_prompt, conv, "transparent")
        all_results.append(r)
        print(f"done ({sum(t['time'] for t in r['turns']):.0f}s)")

    print_results(all_results)

    out_path = str(_HERE / "cache" / "_experiment_prompt.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nraw results saved: {out_path}")


if __name__ == "__main__":
    main()
