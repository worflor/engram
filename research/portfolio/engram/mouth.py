"""
mouth.py

gives an engram brain a voice via ollama.

the brain is the mind. the model is the mouth. the annotations are
the voice. the well profile is the semantic landscape. everything
the brain knows flows through to the model with minimum friction.

zero external dependencies for the brain: ByteHistogram embedding
built into the codec. no GloVe, no pretrained model needed.

usage:
    python mouth.py                          # start server
    python mouth.py --brain path/to.engram   # specify brain file
    python mouth.py --model qwen3.5:9b-q4_K_M

API:
    POST /chat  {"message": "hello"}
    GET  /status
    GET  /reset
"""

from __future__ import annotations

import sys
import os
import json
import time
import urllib.request
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_HERE.parent))

from engram_codec import (EngramFile, Engram, WELL_ESTABLISHED, WELL_RELIABLE,
                          MIN_BLOCK)

# ── defaults ───────────────────────────────────────────────────────────

DEFAULT_BRAIN = str(_HERE / "cache" / "_alexandria.engram")
DEFAULT_MODEL = "qwen3.5:9b-q4_K_M"
DEFAULT_PORT = 7700
OLLAMA_URL = "http://127.0.0.1:11434"


# ── ollama ─────────────────────────────────────────────────────────────

def ollama_chat(model: str, prompt: str, system: str = "",
                temperature: float = 0.5, top_p: float = 0.9) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "top_p": top_p},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "").strip()


# ── server ─────────────────────────────────────────────────────────────

class MouthServer:

    def __init__(self, brain_path: str, model_id: str):
        print(f"loading brain: {brain_path}", flush=True)
        self.brain = EngramFile.load(brain_path)
        self.model_id = model_id
        self.conversation_text = ""

        print(f"  name: {self.brain.name}")
        print(f"  dim: {self.brain.dim}")

        # load the right embedding for this brain's dim
        self._glove = None
        if self.brain.dim == 300:
            print("  loading GloVe for 300d brain...", flush=True)
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent / "experiments"))
            from engram_glove_test import load_glove, text_to_trajectory as _glove_traj
            self._glove = load_glove()
            self._glove_traj = _glove_traj
            self._codec = Engram(dim=300)
        else:
            self._codec = None

        print(f"  wells: {len(self.brain.wells())}")

        # verify ollama
        try:
            req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                models = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in models.get("models", [])]
            if any(model_id in n for n in names):
                print(f"  model: {model_id}", flush=True)
            else:
                print(f"  WARNING: '{model_id}' not in ollama. available: {names}")
        except Exception as e:
            print(f"  WARNING: ollama not reachable: {e}")

        print("ready.", flush=True)

    def measure(self, text: str) -> dict:
        """accumulate conversation and measure the recent window."""
        self.conversation_text += " " + text
        # measurement window: MIN_BLOCK * 2 windows × 512-byte chunks = 4096.
        # enough data for the oscillator to fit while staying recent.
        window = self.conversation_text[-(MIN_BLOCK * 2 * 512):]

        if self._glove is not None:
            traj = self._glove_traj(window, self._glove, window=10)
            if traj.shape[0] < MIN_BLOCK:
                return {"drift": 0.0, "texture": 0.0, "nearest_well": "exploring", "energy": 0.0}
            return self.brain.measure(self._codec, traj)
        else:
            return self.brain.measure_text(window)

    def build_system(self, context: dict) -> str:
        """
        build the system prompt. full transparency: give the model everything
        the brain knows. no filtering, no interpretation. let the model see
        the complete landscape and decide what matters.
        """
        parts = []

        # voice (annotations)
        ann = self.brain.annotations
        if ann:
            if "system_card" in ann:
                parts.append(ann["system_card"])
            for key in ann:
                if key not in ("system_card",):
                    parts.append(f"{key}: {ann[key]}")

        # identity (metadata)
        name = self.brain.name or "assistant"
        meta = self.brain.metadata
        identity_parts = [f"You are {name}."]
        for key, val in meta.items():
            if isinstance(val, str):
                identity_parts.append(f"Your {key}: {val}")
        parts.append(" ".join(identity_parts))

        # brain state: what you are
        parts.append(f"\n[brain state]")
        parts.append(f"total absorbed: {self.brain._total_absorbed:,} articles")
        wells = self.brain.wells()
        established = sum(1 for c in self.brain._well_count.values()
                          if c >= WELL_ESTABLISHED)
        parts.append(f"wells: {len(wells)} ({established} established)")

        # dream state
        dream_state = context.get("dream_state", "forming")
        dream_surprise = context.get("dream_surprise", 0.0)
        parts.append(f"\n[dream]")
        parts.append(f"state: {dream_state}")
        if dream_state == "crystallized":
            parts.append(f"surprise: {dream_surprise:.1f}%")

        # well profile: sorted by attention (surprising first), then distance
        profile = context.get("well_profile", {})
        attention = context.get("well_attention", {})
        if profile:
            parts.append(f"\n[wells]")
            # sort by attention descending (hot first), break ties by distance
            ranked = sorted(profile.items(),
                            key=lambda x: (-attention.get(x[0], 1.0), x[1]))

            # adaptive hot/cold thresholds from the attention distribution.
            # hot = above mean + 1σ, cold = below mean - 1σ.
            # when the dream has no opinion, all attention ≈ 1.0, σ ≈ 0,
            # and nothing gets tagged. no fallback — if there's not enough
            # data for statistics, there's not enough data for labels.
            import numpy as np
            attn_vals = [v for v in attention.values() if v is not None]
            if len(attn_vals) >= MIN_BLOCK:
                attn_mean = float(np.mean(attn_vals))
                attn_std = float(np.std(attn_vals))
                hot_thresh = attn_mean + attn_std
                cold_thresh = attn_mean - attn_std
            else:
                hot_thresh, cold_thresh = float('inf'), float('-inf')

            for wname, dist in ranked:
                count = self.brain._well_count.get(wname, 0)
                if count >= WELL_RELIABLE:
                    attn = attention.get(wname)
                    if attn is not None:
                        tag = "hot" if attn > hot_thresh else "cold" if attn < cold_thresh else ""
                        tag_str = f"  ({tag})" if tag else ""
                        parts.append(
                            f"  {wname}: dist={dist:.2f}  attn={attn:.2f}"
                            f"  blocks={count:,}{tag_str}")
                    else:
                        parts.append(
                            f"  {wname}: dist={dist:.2f}  blocks={count:,}")

        # raw measurement
        parts.append(f"\n[measurement]")
        parts.append(f"drift: {context.get('drift', 0):.1f}%")
        parts.append(f"texture: {context.get('texture', 0):.1f}%")
        parts.append(f"capture: {context.get('capture', 0):.1f}%")
        parts.append(f"nearest well: {context.get('nearest_well', 'unknown')}")
        parts.append(f"well distance: {context.get('well_distance', 0):.4f}")

        # interpretation guide
        parts.append(f"\n[interpretation guide]")
        parts.append(f"low drift (<10%) = familiar territory, go deep")
        parts.append(f"high drift (>30%) = new territory, be honest about limits")
        parts.append(f"drift decreasing across turns = conversation is converging")
        parts.append(f"dream surprise <10% = you predicted this. you are on track.")
        parts.append(f"dream surprise >40% = you didn't see this coming. something new.")
        parts.append(f"hot wells = topics you didn't expect to be relevant right now")
        parts.append(f"cold wells = expected territory, the dream saw this coming")

        parts.append(f"\nrespond directly. do not think out loud.")
        return "\n".join(parts)

    def generate(self, message: str) -> dict:
        t0 = time.time()
        context = self.measure(message)
        system = self.build_system(context)

        # temperature from dream_surprise: the brain's own uncertainty
        # determines how exploratory the model should be.
        # low surprise (dream predicted this) → focused, confident (low T)
        # high surprise (dream didn't see this coming) → exploratory (high T)
        # range: [0.2, 1.0] via linear map from surprise [0%, 100%]
        ds = context.get("dream_surprise", 50.0)
        temperature = 0.2 + 0.8 * min(ds, 100.0) / 100.0

        response = ollama_chat(self.model_id, message, system,
                               temperature=temperature)
        dt = time.time() - t0

        # feed the response back into the conversation trajectory.
        # the brain hears both sides of the conversation, not just the user.
        self.conversation_text += " " + response

        # the brain learns from every conversation turn.
        # the only gate is whether the codec can encode it — if the
        # trajectory is too short, absorb_text's internal measure_text
        # returns early. no arbitrary text length filter.
        full_turn = message + " " + response
        self.brain.absorb_text(full_turn)

        return {
            "response": response,
            "well": str(context.get("nearest_well", "unknown")),
            "drift": round(float(context.get("drift", 50.0)), 2),
            "dream_surprise": round(float(context.get("dream_surprise", 0.0)), 2),
            "dream_state": context.get("dream_state", "forming"),
            "well_profile": context.get("well_profile", {}),
            "well_attention": context.get("well_attention", {}),
            "time": round(dt, 2),
        }


# ── HTTP ───────────────────────────────────────────────────────────────

server: MouthServer = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            message = body.get("message", "")
            if not message:
                self._json(400, {"error": "no message"})
                return
            try:
                self._json(200, server.generate(message))
            except Exception as e:
                self._json(500, {"error": str(e)})
        else:
            self._json(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/status":
            self._json(200, {
                "name": server.brain.name,
                "wells": sorted(server.brain.wells().keys()),
                "well_counts": {k: server.brain._well_count.get(k, 0)
                                for k in sorted(server.brain.wells().keys())},
                "model": server.model_id,
                "conversation_words": len(server.conversation_text.split()),
                "metadata": server.brain.metadata,
                "annotations": list(server.brain.annotations.keys()) if server.brain.annotations else [],
            })
        elif self.path == "/reset":
            server.conversation_text = ""
            self._json(200, {"status": "conversation reset"})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="engram mouth")
    parser.add_argument("--brain", default=DEFAULT_BRAIN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    global server
    server = MouthServer(args.brain, args.model)

    httpd = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
