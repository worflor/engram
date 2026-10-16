"""
test_nano.py — verify that sequential, scan, and Blelloch produce
IDENTICAL output on arbitrary byte sequences. this is the correctness
gate before we port to WGSL.
"""

from __future__ import annotations

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nano_sm import (
    build_transition_table, run_sequential, run_scan, run_blelloch,
)


def random_wikitext(seed: int, size: int = 500) -> bytes:
    """synth byte sequence with enough markup to exercise state transitions."""
    r = random.Random(seed)
    parts = []
    while sum(len(p) for p in parts) < size:
        choice = r.random()
        if choice < 0.6:
            parts.append(" ".join(
                "".join(r.choices("abcdefghijklmnopqrstuvwxyz", k=r.randint(2, 8)))
                for _ in range(r.randint(3, 10))
            ))
        elif choice < 0.75:
            parts.append("{{" + "".join(r.choices("abcdefghij|=0123", k=15)) + "}}")
        elif choice < 0.9:
            parts.append("[[" + "".join(r.choices("abcdefghij|", k=10)) + "]]")
        else:
            parts.append("<ref>" + "".join(r.choices("abcdefg", k=8)) + "</ref>")
    s = " ".join(parts)[:size]
    return s.encode("utf-8")


def test_one(text: bytes, tag: str = "") -> bool:
    T = build_transition_table()

    seq_states, seq_words = run_sequential(text, T)
    scn_states, scn_words = run_scan(text, T)
    blh_states, blh_words = run_blelloch(text, T)

    ok_states = (seq_states == scn_states).all() and (seq_states == blh_states).all()
    ok_words = (seq_words == scn_words) and (seq_words == blh_words)

    if not ok_states:
        print(f"  [{tag}] STATE MISMATCH")
        diff = np.where(seq_states != blh_states)[0]
        print(f"    first diff at position {diff[0] if len(diff) else 'none'}")
        print(f"    seq[:20] = {seq_states[:20]}")
        print(f"    blh[:20] = {blh_states[:20]}")
    if not ok_words:
        print(f"  [{tag}] WORD MISMATCH")
        print(f"    seq ({len(seq_words)}): {seq_words[:10]}")
        print(f"    blh ({len(blh_words)}): {blh_words[:10]}")
        # find first disagreement
        for k, (a, b) in enumerate(zip(seq_words, blh_words)):
            if a != b:
                print(f"    first diff at word {k}: seq={a} blh={b}")
                break
    return ok_states and ok_words


def run():
    T = build_transition_table()

    # handcrafted edge cases
    edge_cases = [
        b"",
        b"hello world",
        b"hello {{template}} world",
        b"aa [[link]] bb",
        b"aa [[file:x|caption text]] bb",
        b"multi {{t1}} and {{t2}} templates",
        b"nested {{outer {{inner}} still outer}} done",
        b"<ref>cite</ref> prose",
        b"a<b>c</b>d",
        b"unclosed {{ template no end",
        b"unclosed [[ link no end",
        b"{",
        b"{{",
        b"}}",
        b"[",
        b"[[",
        b"]]",
        b"<",
        b">",
    ]

    ok = 0
    bad = 0
    for i, tc in enumerate(edge_cases):
        if test_one(tc, f"edge_{i}"):
            ok += 1
        else:
            bad += 1
            print(f"  failed: {tc!r}")

    # fuzz
    print(f"edge cases: {ok}/{len(edge_cases)} passed")
    fuzz_ok = 0
    fuzz_bad = 0
    for seed in range(50):
        tc = random_wikitext(seed, size=random.Random(seed).randint(50, 2000))
        if test_one(tc, f"fuzz_{seed}"):
            fuzz_ok += 1
        else:
            fuzz_bad += 1
    print(f"fuzz:       {fuzz_ok}/{fuzz_ok + fuzz_bad} passed")

    if bad == 0 and fuzz_bad == 0:
        print("\nsequential == scan == blelloch — algorithm verified.")
    else:
        print(f"\n{bad + fuzz_bad} cases disagreed. fix before porting to WGSL.")


if __name__ == "__main__":
    run()
