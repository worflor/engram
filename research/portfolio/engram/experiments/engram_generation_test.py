"""
engram_generation_test.py

The real test: feed Qwen its own continuation, not human word salad.

Prompts are a human jerking the steering wheel. Generation is when the
model falls into orbit. This is where we find out if thoughts have momentum.

Also: measure the Shannon entropy of the residual byte stream to predict
what Logos would achieve if wired in. Logos approaches H0 on structured
data and beats it via inter-byte correlations (Bit1 mode).
"""

from __future__ import annotations

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from engram import (
    EngramCodec,
    EngramMode,
    ENGRAM_PROJ_DIM,
    OSCILLATOR_FAMILIES,
    soft_witness,
    engram_summary,
    engram_parameter_vector,
    encode_engram,
    quantize_residuals,
    pack_residuals,
)

from engram_real_air import (
    load_qwen,
    build_scramble_matrix,
)


# ── entropy measurement (what Logos would achieve) ────────────────────

def byte_entropy(data: bytes) -> float:
    """Shannon entropy H0 in bits per byte."""
    if len(data) == 0:
        return 0.0
    counts = np.zeros(256, dtype=np.int64)
    for b in data:
        counts[b] += 1
    probs = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


def ideal_size(data: bytes) -> int:
    """Ideal compressed size at Shannon entropy."""
    H = byte_entropy(data)
    return max(1, int(math.ceil(len(data) * H / 8)))


# ── generation with hidden state capture ──────────────────────────────

def generate_with_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
) -> tuple[torch.Tensor, list[int], str]:
    """
    Generate text token by token, capturing the last hidden state at each step.

    Returns (hidden_states [T_total, hidden_size], token_ids, generated_text).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    all_hidden = []
    all_token_ids = input_ids[0].tolist()

    # prefill: get hidden states for the prompt
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=True,
        )
    prompt_hidden = outputs.hidden_states[-1][0].detach().cpu()  # [prompt_len, hidden]
    all_hidden.append(prompt_hidden)
    past_kv = outputs.past_key_values

    # autoregressive generation
    next_token_id = int(outputs.logits[0, -1].argmax().item())
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for step in range(max_new_tokens):
        if next_token_id == eos_id:
            break

        all_token_ids.append(next_token_id)
        next_input = torch.tensor([[next_token_id]], dtype=torch.long)

        with torch.no_grad():
            outputs = model(
                input_ids=next_input,
                past_key_values=past_kv,
                output_hidden_states=True,
                use_cache=True,
            )

        step_hidden = outputs.hidden_states[-1][0].detach().cpu()  # [1, hidden]
        all_hidden.append(step_hidden)
        past_kv = outputs.past_key_values
        next_token_id = int(outputs.logits[0, -1].argmax().item())

    hidden_cat = torch.cat(all_hidden, dim=0)  # [T_total, hidden]
    text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    return hidden_cat, all_token_ids, text


# ── the experiment ────────────────────────────────────────────────────

GENERATION_PROMPTS = [
    ("engine explanation",
     "Explain how a four-stroke combustion engine works:"),
    ("story continuation",
     "Once upon a time, in a lighthouse on a rocky shore,"),
    ("code generation",
     "def binary_search(arr, target):"),
    ("reasoning chain",
     "If all roses are flowers and all flowers need water, then"),
    ("creative writing",
     "The last human on Mars opened the airlock and saw"),
]


def run_generation_experiment():
    print("=" * 70)
    print("  ENGRAM GENERATION TEST")
    print("  the model thinking, not human word salad")
    print("  + residual entropy analysis (what Logos would achieve)")
    print("=" * 70)
    print()

    model, tokenizer = load_qwen()
    hidden_size = model.config.hidden_size
    S = build_scramble_matrix(hidden_size, ENGRAM_PROJ_DIM)

    alpha = float(os.environ.get("ENGRAM_ALPHA", "0.005"))
    max_gen = int(os.environ.get("ENGRAM_MAX_GEN", "200"))
    codec = EngramCodec(alpha=alpha)

    print(f"alpha = {alpha}")
    print(f"max generation tokens = {max_gen}")
    print()

    all_results = []

    for name, prompt in GENERATION_PROMPTS:
        print(f"--- {name} ---")
        print(f"  prompt: {prompt}")

        # generate with hidden state capture
        t0 = time.time()
        hidden, token_ids, text = generate_with_hidden_states(
            model, tokenizer, prompt, max_new_tokens=max_gen,
        )
        gen_time = time.time() - t0
        T = hidden.shape[0]
        prompt_tokens = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        gen_tokens = T - prompt_tokens

        print(f"  total tokens: {T} (prompt={prompt_tokens}, generated={gen_tokens})")
        print(f"  generation time: {gen_time:.1f}s ({gen_tokens/max(gen_time,0.01):.1f} tok/s)")
        print(f"  output: {text[len(prompt):len(prompt)+120]}...")

        # soft witness projection
        witness_np = soft_witness(hidden.numpy(), S, alpha=alpha)

        # SEPARATE ANALYSIS: prompt vs generation
        w_prompt = witness_np[:prompt_tokens]
        w_gen = witness_np[prompt_tokens:]

        for label, w_seg in [("PROMPT", w_prompt), ("GENERATION", w_gen), ("FULL", witness_np)]:
            if w_seg.shape[0] < 3:
                print(f"  [{label}] too short ({w_seg.shape[0]} tokens)")
                continue

            # trajectory smoothness
            velocity = np.diff(w_seg, axis=0)
            vel_norm = np.linalg.norm(velocity, axis=1)
            mean_vel = float(vel_norm.mean())
            if len(vel_norm) > 2:
                vn = vel_norm - vel_norm.mean()
                autocorr = float(np.correlate(vn[:-1], vn[1:])[0] / (np.dot(vn, vn) + 1e-10))
            else:
                autocorr = 0.0

            # encode
            packet = codec.encode_trajectory(w_seg)
            summary = engram_summary(packet)

            # wire format
            wire = encode_engram(packet, codec.q_bits)
            raw_size = w_seg.size * 4
            bipolar_size = w_seg.shape[0] * ENGRAM_PROJ_DIM

            # residual byte stream entropy (predicts Logos compression)
            resid_bytes = bytearray()
            for block in packet.blocks:
                q = quantize_residuals(block.residuals, codec.q_bits)
                resid_bytes.extend(pack_residuals(q))

            resid_entropy = byte_entropy(bytes(resid_bytes))
            resid_ideal = ideal_size(bytes(resid_bytes))

            # per-family energy capture
            captures = []
            for fi, (fname, ds, de) in enumerate(OSCILLATOR_FAMILIES):
                fam_resid = sum(float(np.sum(b.residuals[:, ds:de] ** 2)) for b in packet.blocks)
                fam_total = sum(
                    float(np.sum(w_seg[b.start_idx:b.start_idx+b.length, ds:de] ** 2))
                    for b in packet.blocks
                    if b.start_idx + b.length <= w_seg.shape[0]
                )
                capture = (1.0 - fam_resid / fam_total) * 100 if fam_total > 1e-8 else 0.0
                captures.append((fname, de - ds, capture))

            best_capture = max(captures, key=lambda x: x[2])
            worst_capture = min(captures, key=lambda x: x[2])

            print(f"  [{label}] {w_seg.shape[0]} tokens")
            print(f"    velocity: mean={mean_vel:.4f}, autocorr(1)={autocorr:.4f}")
            print(f"    modes: {summary['mode_distribution']}")
            print(f"    residual RMS: {summary['residual_rms']:.4f}")
            print(f"    oscillator capture: best={best_capture[0]}({best_capture[2]:.1f}%), worst={worst_capture[0]}({worst_capture[2]:.1f}%)")
            print(f"    wire: {len(wire):,}B (raw {raw_size:,}B) = {raw_size/max(len(wire),1):.1f}x vs float32")
            print(f"    residual entropy: {resid_entropy:.2f} bits/byte (max 8.00)")
            print(f"    residual stream: {len(resid_bytes):,}B -> ideal {resid_ideal:,}B ({len(resid_bytes)/max(resid_ideal,1):.1f}x Logos headroom)")

            if label == "GENERATION" and w_seg.shape[0] >= 3:
                # wire with Logos estimate
                logos_est = resid_ideal + 2 * ENGRAM_PROJ_DIM * 4 + len(packet.blocks) * 20 * 4  # seeds + params
                logos_ratio_float = raw_size / max(logos_est, 1)
                logos_ratio_int8 = bipolar_size / max(logos_est, 1)
                print(f"    LOGOS ESTIMATE: ~{logos_est:,}B ({logos_ratio_float:.1f}x vs float32, {logos_ratio_int8:.2f}x vs int8)")

            if label == "GENERATION":
                all_results.append({
                    "name": name,
                    "gen_tokens": gen_tokens,
                    "autocorr": autocorr,
                    "residual_rms": summary["residual_rms"],
                    "modes": summary["mode_distribution"],
                    "wire_ratio_float": raw_size / max(len(wire), 1),
                    "resid_entropy": resid_entropy,
                    "best_capture": best_capture,
                    "worst_capture": worst_capture,
                    "logos_headroom": len(resid_bytes) / max(resid_ideal, 1),
                })

        print()

    # ── verdict ───────────────────────────────────────────────────────
    print("=" * 70)
    print("  VERDICT: GENERATION")
    print("=" * 70)
    print()

    if not all_results:
        print("  no generation results.")
        return

    total_modes = {}
    for m in EngramMode:
        total_modes[m.name] = sum(r["modes"].get(m.name, 0) for r in all_results)
    total_blocks = sum(total_modes.values())

    print(f"  Q1: Block modes on GENERATION (model thinking, not human input)")
    for mname, mcount in total_modes.items():
        pct = 100 * mcount / max(total_blocks, 1)
        print(f"      {mname:8s}: {mcount}/{total_blocks} ({pct:.0f}%)")
    orbital_pct = 100 * total_modes.get("ORBITAL", 0) / max(total_blocks, 1)
    if orbital_pct >= 80:
        print(f"      --> THE MODEL'S THOUGHTS ORBIT.")
    print()

    mean_autocorr = np.mean([r["autocorr"] for r in all_results])
    mean_resid = np.mean([r["residual_rms"] for r in all_results])
    print(f"  Q2: Does the MODEL's trajectory have momentum?")
    print(f"      velocity autocorrelation: {mean_autocorr:.4f}")
    print(f"      mean residual RMS: {mean_resid:.4f}")
    if mean_autocorr > 0.1:
        print(f"      --> YES. the model's latent space has physics. thoughts have inertia.")
    elif mean_autocorr > 0.0:
        print(f"      --> weak but positive. there is some momentum.")
    else:
        print(f"      --> no clear momentum in generation either.")
    print()

    mean_entropy = np.mean([r["resid_entropy"] for r in all_results])
    mean_headroom = np.mean([r["logos_headroom"] for r in all_results])
    mean_ratio = np.mean([r["wire_ratio_float"] for r in all_results])
    print(f"  Q3: Compression with Logos prediction")
    print(f"      current wire (no Logos): {mean_ratio:.1f}x vs float32")
    print(f"      residual entropy: {mean_entropy:.2f} bits/byte")
    print(f"      Logos headroom: {mean_headroom:.1f}x further compression available")
    print(f"      predicted with Logos: ~{mean_ratio * mean_headroom:.1f}x vs float32")
    if mean_entropy < 6.0:
        print(f"      --> RESIDUALS HAVE STRUCTURE. Logos will crush them.")
    elif mean_entropy < 7.5:
        print(f"      --> some structure. Logos helps but not dramatically.")
    else:
        print(f"      --> residuals are near-random. Logos won't help much.")
    print()

    # compare prompt vs generation momentum
    print(f"  PROMPT vs GENERATION comparison:")
    print(f"      (run with ENGRAM_MAX_GEN=500 for longer generation)")


if __name__ == "__main__":
    run_generation_experiment()
