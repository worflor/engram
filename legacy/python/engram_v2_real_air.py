"""
engram_v2_real_air.py

v2 real air test: 128 per-pair oscillators on actual Qwen generation.
"""

from __future__ import annotations
import sys, os, time, math
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from engram import soft_witness, ENGRAM_PROJ_DIM
from engram_v2 import (
    EngramCodecV2, Mode, N_PAIRS, PROJ_DIM,
    summary, byte_entropy, encode_wire, quantize_per_pair, _pack_varints,
)
from engram_real_air import load_qwen, build_scramble_matrix
from engram_generation_test import generate_with_hidden_states

PROMPTS = [
    ("engine explanation", "Explain how a four-stroke combustion engine works:"),
    ("story continuation", "Once upon a time, in a lighthouse on a rocky shore,"),
    ("code generation",    "def binary_search(arr, target):"),
    ("reasoning chain",    "If all roses are flowers and all flowers need water, then"),
    ("creative writing",   "The last human on Mars opened the airlock and saw"),
]


def run():
    print("=" * 70)
    print("  ENGRAM v2 REAL AIR: 128 PER-PAIR OSCILLATORS")
    print("=" * 70)
    print()

    model, tokenizer = load_qwen()
    S = build_scramble_matrix(model.config.hidden_size, PROJ_DIM)
    alpha = float(os.environ.get("ENGRAM_ALPHA", "0.005"))
    max_gen = int(os.environ.get("ENGRAM_MAX_GEN", "200"))
    codec = EngramCodecV2(alpha=alpha)

    print(f"alpha={alpha}, max_gen={max_gen}, oscillators={N_PAIRS}")
    print()

    results = []

    for name, prompt in PROMPTS:
        print(f"--- {name} ---")
        hidden, token_ids, text = generate_with_hidden_states(model, tokenizer, prompt, max_gen)
        T = hidden.shape[0]
        prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        gen_len = T - prompt_len
        print(f"  tokens: {T} (prompt={prompt_len}, gen={gen_len})")

        w_full = soft_witness(hidden.numpy(), S, alpha=alpha)

        for label, w in [("PROMPT", w_full[:prompt_len]), ("GENERATION", w_full[prompt_len:]), ("FULL", w_full)]:
            if w.shape[0] < 3:
                continue

            pkt = codec.encode_trajectory(w)
            s = summary(pkt)
            dec = codec.decode_trajectory(pkt)
            roundtrip = float(np.sqrt(np.mean((w - dec) ** 2)))

            # energy capture
            total_resid = sum(float(np.sum(b.residuals ** 2)) for b in pkt.blocks)
            total_signal = float(np.sum(w[2:] ** 2)) if w.shape[0] > 2 else 1e-8
            capture = max(0, (1.0 - total_resid / max(total_signal, 1e-8)) * 100)

            # wire size
            wire = encode_wire(pkt, codec.q_bits)
            raw_f32 = w.size * 4
            raw_i8 = w.shape[0] * PROJ_DIM

            # residual entropy
            resid_bytes = bytearray()
            for block in pkt.blocks:
                q, _ = quantize_per_pair(block.residuals, codec.q_bits)
                resid_bytes.extend(_pack_varints(q.ravel().astype(np.int32)))
            ent = byte_entropy(bytes(resid_bytes))
            ideal = max(1, int(math.ceil(len(resid_bytes) * ent / 8)))

            # velocity autocorrelation
            if w.shape[0] > 3:
                vel = np.diff(w, axis=0)
                vn = np.linalg.norm(vel, axis=1)
                vm = vn - vn.mean()
                ac = float(np.correlate(vm[:-1], vm[1:])[0] / (np.dot(vm, vm) + 1e-10)) if len(vm) > 1 else 0.0
            else:
                ac = 0.0

            # per-pair capture distribution
            pair_captures = []
            for pi in range(N_PAIRS):
                d0, d1 = pi*2, pi*2+1
                pr = sum(float(np.sum(b.residuals[:, d0:d1+1] ** 2)) for b in pkt.blocks)
                ps = sum(
                    float(np.sum(w[b.start_idx:b.start_idx+b.length, d0:d1+1] ** 2))
                    for b in pkt.blocks
                    if b.start_idx + b.length <= w.shape[0]
                )
                pair_captures.append(max(0, (1.0 - pr / max(ps, 1e-10)) * 100))
            pc_arr = np.array(pair_captures)

            print(f"  [{label}] {w.shape[0]} tok | modes={s['modes']}")
            print(f"    capture: {capture:.1f}% total | per-pair: mean={pc_arr.mean():.1f}%, median={np.median(pc_arr):.1f}%, max={pc_arr.max():.1f}%, >10%: {np.sum(pc_arr>10)}/128")
            print(f"    residual RMS: {s['residual_rms']:.4f} | autocorr: {ac:.4f}")
            print(f"    wire: {len(wire):,}B vs f32 {raw_f32:,}B ({raw_f32/max(len(wire),1):.1f}x) vs i8 {raw_i8:,}B ({raw_i8/max(len(wire),1):.2f}x)")
            print(f"    entropy: {ent:.2f} b/B | Logos est: {ideal:,}B ({len(resid_bytes)/max(ideal,1):.1f}x headroom)")

            if label == "GENERATION":
                logos_est = ideal + 2 * PROJ_DIM * 4 + len(pkt.blocks) * N_PAIRS * 16
                print(f"    LOGOS EST: ~{logos_est:,}B ({raw_f32/max(logos_est,1):.1f}x vs f32, {raw_i8/max(logos_est,1):.2f}x vs i8)")
                results.append({
                    "name": name, "gen": gen_len,
                    "capture": capture, "modes": s["modes"],
                    "resid_rms": s["residual_rms"],
                    "wire_ratio": raw_f32/max(len(wire),1),
                    "entropy": ent, "autocorr": ac,
                    "pair_mean_cap": float(pc_arr.mean()),
                    "pairs_above_10": int(np.sum(pc_arr > 10)),
                })
        print()

    # verdict
    print("=" * 70)
    print("  VERDICT: v2 PER-PAIR vs v1 5-FAMILY")
    print("=" * 70)
    print()

    if results:
        mean_cap = np.mean([r["capture"] for r in results])
        mean_pair_cap = np.mean([r["pair_mean_cap"] for r in results])
        mean_above = np.mean([r["pairs_above_10"] for r in results])
        mean_ratio = np.mean([r["wire_ratio"] for r in results])
        mean_ent = np.mean([r["entropy"] for r in results])
        mean_ac = np.mean([r["autocorr"] for r in results])
        mean_rms = np.mean([r["resid_rms"] for r in results])

        total_modes = {}
        for m in Mode:
            total_modes[m.name] = sum(r["modes"].get(m.name, 0) for r in results)
        total_blocks = sum(total_modes.values())

        print(f"  modes: {total_modes} ({total_blocks} blocks)")
        print(f"  energy capture: {mean_cap:.1f}% total (v1 was ~3%)")
        print(f"  per-pair capture: mean={mean_pair_cap:.1f}%, pairs>10%: {mean_above:.0f}/128")
        print(f"  residual RMS: {mean_rms:.4f} (v1 was ~0.49)")
        print(f"  compression: {mean_ratio:.1f}x vs f32 (v1 was ~3.0x)")
        print(f"  entropy: {mean_ent:.2f} b/B (v1 was ~6.72)")
        print(f"  velocity autocorr: {mean_ac:.4f} (v1 gen was ~0.10)")


if __name__ == "__main__":
    run()
