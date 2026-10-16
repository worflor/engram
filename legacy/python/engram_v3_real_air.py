"""v3 real air: hierarchical oscillators on Qwen generation."""

from __future__ import annotations
import sys, os, time, math
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram import soft_witness, ENGRAM_PROJ_DIM
from engram_v3 import EngramCodecV3, N_PAIRS, PROJ_DIM, v3_summary, byte_entropy
from engram_v2 import quantize_per_pair, _pack_varints
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
    print("  ENGRAM v3 REAL AIR: HIERARCHICAL OSCILLATORS")
    print("  macro (topic orbit) + micro (vocab jitter) = cascaded prediction")
    print("=" * 70)
    print()

    model, tokenizer = load_qwen()
    S = build_scramble_matrix(model.config.hidden_size, PROJ_DIM)
    alpha = float(os.environ.get("ENGRAM_ALPHA", "0.005"))
    max_gen = int(os.environ.get("ENGRAM_MAX_GEN", "200"))
    codec = EngramCodecV3(alpha=alpha)

    print(f"alpha={alpha}, max_gen={max_gen}")
    print()

    gen_results = []

    for name, prompt in PROMPTS:
        print(f"--- {name} ---")
        hidden, token_ids, text = generate_with_hidden_states(model, tokenizer, prompt, max_gen)
        T = hidden.shape[0]
        pl = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        gl = T - pl
        print(f"  {T} tokens (prompt={pl}, gen={gl})")

        w = soft_witness(hidden.numpy(), S, alpha=alpha)

        for label, ws in [("PROMPT", w[:pl]), ("GEN", w[pl:]), ("FULL", w)]:
            if ws.shape[0] < 3:
                continue

            pkt = codec.encode_trajectory(ws)
            s = v3_summary(pkt)
            dec = codec.decode_trajectory(pkt)
            rt_err = float(np.sqrt(np.mean((ws - dec) ** 2)))

            # total capture
            tr = sum(float(np.sum(b.residuals ** 2)) for b in pkt.blocks)
            ts = float(np.sum(ws[2:] ** 2)) if ws.shape[0] > 2 else 1e-8
            total_cap = max(0, (1.0 - tr / max(ts, 1e-8)) * 100)

            # residual entropy
            resid_bytes = bytearray()
            for blk in pkt.blocks:
                q, _ = quantize_per_pair(blk.residuals, 8)
                resid_bytes.extend(_pack_varints(q.ravel().astype(np.int32)))
            ent = byte_entropy(bytes(resid_bytes))
            ideal = max(1, int(math.ceil(len(resid_bytes) * ent / 8)))

            raw_f32 = ws.size * 4
            raw_i8 = ws.shape[0] * PROJ_DIM

            # velocity autocorr
            if ws.shape[0] > 3:
                vel = np.diff(ws, axis=0)
                vn = np.linalg.norm(vel, axis=1)
                vm = vn - vn.mean()
                ac = float(np.correlate(vm[:-1], vm[1:])[0] / (np.dot(vm, vm) + 1e-10))
            else:
                ac = 0.0

            print(f"  [{label:6s}] {ws.shape[0]:3d} tok | macro={s['mean_macro_capture']:5.1f}% micro={s['mean_micro_capture']:5.1f}% total={total_cap:5.1f}%")
            print(f"           resid_rms={s['residual_rms']:.4f} | ent={ent:.2f} b/B | autocorr={ac:.4f}")
            print(f"           blocks={s['n_blocks']} sub={s['n_sub_blocks']} | roundtrip={rt_err:.8f}")

            if label == "GEN":
                gen_results.append({
                    "name": name, "gen": gl,
                    "macro": s["mean_macro_capture"],
                    "micro": s["mean_micro_capture"],
                    "total": total_cap,
                    "resid_rms": s["residual_rms"],
                    "entropy": ent, "autocorr": ac,
                    "n_blocks": s["n_blocks"], "n_sub": s["n_sub_blocks"],
                })
        print()

    print("=" * 70)
    print("  VERDICT: v3 HIERARCHICAL vs v2 FLAT vs v1 5-FAMILY")
    print("=" * 70)
    print()
    if gen_results:
        mc = np.mean([r["macro"] for r in gen_results])
        uc = np.mean([r["micro"] for r in gen_results])
        tc = np.mean([r["total"] for r in gen_results])
        rms = np.mean([r["resid_rms"] for r in gen_results])
        ent = np.mean([r["entropy"] for r in gen_results])
        ac = np.mean([r["autocorr"] for r in gen_results])
        print(f"  GENERATION averages:")
        print(f"    macro capture:  {mc:.1f}%   (v2: 5.8%, v1: 3%)")
        print(f"    micro capture:  {uc:.1f}%   (of macro residuals)")
        print(f"    total capture:  {tc:.1f}%   (both levels)")
        print(f"    residual RMS:   {rms:.4f}  (v2: 0.481, v1: 0.489)")
        print(f"    entropy:        {ent:.2f} b/B (v2: 6.42, v1: 6.72)")
        print(f"    velocity autocorr: {ac:.4f}")

if __name__ == "__main__":
    run()
