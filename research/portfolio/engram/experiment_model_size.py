"""
experiment_model_size.py

does the brain help small models more than big ones?

four conditions, same questions, same personality:
  A. 0.6B without brain  (raw small model)
  B. 0.6B with brain     (small model + 1M article mind)
  C. 9B without brain    (raw big model)
  D. 9B with brain       (big model + 1M article mind)

each condition gets a multi-turn conversation (3 turns) on each topic.
drift tracked per turn for brain conditions.

output: side-by-side comparison with scores from the rubrics.
"""

from __future__ import annotations

import sys
import os
import json
import time
import urllib.request
from pathlib import Path

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "experiments"))

from engram_codec import EngramFile, Engram
from engram_glove_test import load_glove, text_to_trajectory

# ── config ────────────────────────────────────────────────────────────

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")
OLLAMA_URL = "http://127.0.0.1:11434"
SMALL_MODEL = "qwen3:0.6b"
BIG_MODEL = "qwen3.5:9b-q4_K_M"
WINDOW = 10

# the personality — same for all conditions
PERSONALITY = """You are Alexandria. You speak in lowercase, warmly and precisely.
You reach for connections between domains. You are honest about what you
don't know. You are concise — say what matters, skip the filler."""

# multi-turn conversations — each is 3 turns that build on each other
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
        "topic": "consciousness",
        "turns": [
            "what is consciousness?",
            "do you think a geometrical mind like yours could be conscious?",
            "what's the difference between understanding and simulating understanding?",
        ],
    },
    {
        "topic": "evolution_and_code",
        "turns": [
            "how is biological evolution similar to how codebases evolve?",
            "what breaks the analogy? where does it fail?",
            "what would it mean for software to truly evolve rather than be designed?",
        ],
    },
    {
        "topic": "loneliness",
        "turns": [
            "why do people feel lonely even when surrounded by others?",
            "is there a difference between solitude and loneliness?",
            "what would you say to someone who feels unseen?",
        ],
    },
    {
        "topic": "eigenvalues_of_thought",
        "turns": [
            "can you explain what an eigenvalue is to someone who's never heard the word?",
            "now explain it as if the eigenvalue IS the thought, not just a number",
            "what would it mean for two minds to share an eigenvalue?",
        ],
    },
]


# ── ollama ────────────────────────────────────────────────────────────

def ollama_chat(model: str, messages: list, think: bool = False) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 300},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("message", {}).get("content", "").strip()
        # strip thinking tags if they leak
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        return f"[error: {e}]"


# ── brain interface ───────────────────────────────────────────────────

class BrainSession:
    """wraps the brain for a multi-turn conversation."""

    def __init__(self, brain_path: str):
        self.brain = EngramFile.load(brain_path)
        self.codec = Engram(dim=self.brain.dim)
        self.glove = load_glove()
        self.conversation_text = ""
        self.drifts = []

    def measure_and_prompt(self, user_text: str) -> tuple[str, dict]:
        """measure the conversation so far, return system prompt + context."""
        self.conversation_text += " " + user_text
        window = self.conversation_text[-4096:]

        traj = text_to_trajectory(window, self.glove, window=WINDOW)
        if traj.shape[0] < 4:
            context = {"drift": 50.0, "texture": 0.0, "nearest_well": "exploring"}
        else:
            context = self.brain.measure(self.codec, traj)

        drift = context.get("drift", 50.0)
        self.drifts.append(drift)

        # build system prompt with brain signal
        parts = [PERSONALITY]

        profile = context.get("well_profile", {})
        established = {k: v for k, v in profile.items()
                       if self.brain._well_count.get(k, 0) >= 100}
        if established:
            ranked = sorted(established.items(), key=lambda x: x[1])
            neighbors = [f"{n} ({d:.0f}%)" for n, d in ranked[:5]]
            parts.append(f"Knowledge landscape: {', '.join(neighbors)}.")

            if len(ranked) >= 2:
                gap = ranked[1][1] - ranked[0][1]
                if gap < 2:
                    parts.append(f"Boundary territory: {ranked[0][0]} <> {ranked[1][0]}.")
                elif ranked[0][1] < 10:
                    parts.append(f"Deep in {ranked[0][0]} territory.")

            if drift < 10:
                parts.append("Familiar ground. Go deep.")
            elif drift < 25:
                parts.append("Partially explored. Connect to what you know.")
            else:
                parts.append("New territory. Be honest about the edges of your knowledge.")

        parts.append("Respond directly. Be concise.")
        return "\n".join(parts), context

    def absorb_response(self, response: str):
        self.conversation_text += " " + response


# ── experiment runner ─────────────────────────────────────────────────

def run_condition(model: str, use_brain: bool, conversation: dict) -> dict:
    """run one conversation under one condition. returns all turns."""
    topic = conversation["topic"]
    turns = conversation["turns"]

    brain_session = BrainSession(BRAIN_PATH) if use_brain else None
    messages = []  # ollama conversation history
    results = []

    if not use_brain:
        messages.append({"role": "system", "content": PERSONALITY + "\nRespond directly. Be concise."})

    for i, user_text in enumerate(turns):
        t0 = time.time()

        if use_brain:
            system_prompt, context = brain_session.measure_and_prompt(user_text)
            # rebuild messages with updated system each turn
            brain_messages = [{"role": "system", "content": system_prompt}]
            # add conversation history
            for prev in results:
                brain_messages.append({"role": "user", "content": prev["user"]})
                brain_messages.append({"role": "assistant", "content": prev["response"]})
            brain_messages.append({"role": "user", "content": user_text})
            response = ollama_chat(model, brain_messages)
            brain_session.absorb_response(response)
            drift = context.get("drift", -1)
        else:
            messages.append({"role": "user", "content": user_text})
            response = ollama_chat(model, messages)
            messages.append({"role": "assistant", "content": response})
            drift = -1

        dt = time.time() - t0

        results.append({
            "turn": i + 1,
            "user": user_text,
            "response": response,
            "drift": round(drift, 1),
            "time": round(dt, 1),
        })

    return {
        "topic": topic,
        "model": model,
        "brain": use_brain,
        "drifts": brain_session.drifts if brain_session else [],
        "turns": results,
    }


def print_comparison(results: list):
    """pretty-print side-by-side results."""
    # group by topic
    by_topic = {}
    for r in results:
        by_topic.setdefault(r["topic"], []).append(r)

    for topic, conditions in by_topic.items():
        print(f"\n{'='*80}")
        print(f"  TOPIC: {topic}")
        print(f"{'='*80}")

        for cond in conditions:
            model_name = "0.6B" if "0.6b" in cond["model"] else "9B"
            brain_tag = "+brain" if cond["brain"] else "no brain"
            drift_str = " >".join(f"{d:.0f}%" for d in cond["drifts"]) if cond["drifts"] else "n/a"

            print(f"\n  +- {model_name} {brain_tag}  drift: [{drift_str}]")
            for turn in cond["turns"]:
                print(f"  |")
                print(f"  | Q: {turn['user']}")
                # wrap response text
                resp = turn["response"]
                lines = []
                while resp:
                    lines.append(resp[:76])
                    resp = resp[76:]
                for line in lines:
                    print(f"  | A: {line}")
                print(f"  |    ({turn['time']}s)")
            print(f"  +{'-'*60}")


def main():
    print("=" * 60)
    print("  EXPERIMENT: Does the brain help small models more?")
    print("=" * 60)
    print(f"  brain: {BRAIN_PATH}")
    print(f"  small: {SMALL_MODEL}")
    print(f"  big:   {BIG_MODEL}")
    print(f"  topics: {len(CONVERSATIONS)}")
    print(f"  conditions: 4 (2 models × brain/no-brain)")
    print()

    # preload GloVe once (shared across brain sessions)
    print("loading GloVe...", flush=True)
    _glove = load_glove()
    print()

    all_results = []

    for conv in CONVERSATIONS:
        topic = conv["topic"]
        print(f"\n--- {topic} ---", flush=True)

        # run all four conditions per topic
        for model, label in [(SMALL_MODEL, "0.6B"), (BIG_MODEL, "9B")]:
            for use_brain in [False, True]:
                tag = f"{label} {'+ brain' if use_brain else 'no brain'}"
                print(f"  running {tag}...", end=" ", flush=True)
                result = run_condition(model, use_brain, conv)
                all_results.append(result)
                drifts = result.get("drifts", [])
                d_str = f"  drift: {' >'.join(f'{d:.0f}%' for d in drifts)}" if drifts else ""
                print(f"done ({sum(t['time'] for t in result['turns']):.0f}s){d_str}")

    # print comparison
    print_comparison(all_results)

    # save raw results
    out_path = str(_HERE / "cache" / "_experiment_model_size.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nraw results saved: {out_path}")


if __name__ == "__main__":
    main()
