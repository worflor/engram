"""
engram_real_air.py

The bird breathes real air.

Feeds actual Qwen 3.5 hidden states through the Engram codec and answers:

    Q1: What does the block mode distribution look like?
        ORBITAL/STATIC = the thought has gravity.
        100% RESET = the math failed. thoughts are too jagged.

    Q2: What is the compression ratio on real thought?
        3x+ without Logos = we've won.

    Q3: Does semantic trajectory have momentum?
        Low AR(2) residuals = yes. LLM latent space mimics physical dynamics.
        High residuals = no. Thoughts are not smooth.
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from engram import (
    EngramCodec,
    EngramMode,
    ENGRAM_PROJ_DIM,
    OSCILLATOR_FAMILIES,
    soft_witness,
    harden_witness,
    engram_summary,
    engram_parameter_vector,
    encode_engram,
)

# ── model loading ─────────────────────────────────────────────────────

def load_qwen():
    """Load Qwen 3.5 0.8B with hidden state extraction hook."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("TOWER_MODEL_ID", "Qwen/Qwen3.5-0.8B")
    print(f"loading {model_id} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  hidden_size={model.config.hidden_size}")
    print(f"  num_layers={model.config.num_hidden_layers}")
    print(f"  vocab_size={model.config.vocab_size}")

    return model, tokenizer


def extract_hidden_states(model, tokenizer, text: str) -> tuple[torch.Tensor, list[int]]:
    """
    Feed text through Qwen, extract last-layer hidden states.

    Returns (hidden_states [T, hidden_size], token_ids [T]).
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    token_ids = input_ids[0].tolist()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    # last hidden layer: outputs.hidden_states[-1] is [batch, seq, hidden]
    hidden = outputs.hidden_states[-1][0].detach().cpu()  # [T, hidden_size]
    return hidden, token_ids


def build_scramble_matrix(hidden_size: int, proj_dim: int = 256, seed: int = 42) -> np.ndarray:
    """
    Build a deterministic ±1 scramble matrix from Kizuna mask geometry.

    This replicates what hyper_cache._build_scramble_matrix_from_masks does
    but standalone, without requiring the full hyper_cache import chain.
    """
    # build parity table
    parity = np.zeros(65536, dtype=np.uint8)
    for i in range(1, 65536):
        parity[i] = parity[i >> 1] ^ (i & 1)

    # generate deterministic masks
    rng = np.random.RandomState(seed)
    masks = []
    seen = set()
    # start with Kizuna tier-1 masks
    tier1 = [
        0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080,
        0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, 0x8000,
        0x0101, 0x0202, 0x0404, 0x0808, 0x1010, 0x2020, 0x4040, 0x8080,
        0xFFFF,
    ]
    tier2 = [
        0x0003, 0x0005, 0x0006, 0x0009, 0x000A, 0x000C,
        0x0011, 0x0012, 0x0014, 0x0018, 0x0021, 0x0022,
        0x0024, 0x0028, 0x0030, 0x0041, 0x0042, 0x0044,
        0x0048, 0x0050, 0x0060, 0x0081, 0x0082, 0x0084,
        0x0088, 0x0090, 0x00A0, 0x00C0,
    ]
    for m in tier1 + tier2:
        if m not in seen:
            masks.append(m)
            seen.add(m)

    # fill remaining with deterministic random masks
    while len(masks) < proj_dim:
        m = rng.randint(1, 65536)
        if m not in seen and bin(m).count('1') >= 4:
            masks.append(m)
            seen.add(m)

    masks = np.array(masks[:proj_dim], dtype=np.int32)

    # feature indices: just use sequential for simplicity
    proj_input = min(hidden_size, 65535)
    feat_idx = np.arange(proj_input, dtype=np.int32) % 65536

    # build matrix: S[i,j] = (-1)^parity(mask[i] & feat[j])
    parity_vals = parity[(masks[:, None] & feat_idx[None, :]).astype(np.int32)]
    S = np.where(parity_vals, -1.0, 1.0).astype(np.float32)

    return S


# ── the experiment ────────────────────────────────────────────────────

CONVERSATIONS = [
    # structured explanation (should be smooth, orbital)
    (
        "car engine",
        "Explain step by step how a four-stroke internal combustion engine works. "
        "Cover the intake, compression, power, and exhaust strokes in detail. "
        "Explain the role of the crankshaft, pistons, valves, and spark plugs."
    ),
    # narrative (should have arcs, topic transitions)
    (
        "short story",
        "Write a short story about a lighthouse keeper who discovers a message "
        "in a bottle that changes everything. Include dialogue and description."
    ),
    # factual Q&A (should be dense, structured)
    (
        "science QA",
        "What is photosynthesis? How do plants convert sunlight into energy? "
        "What role do chloroplasts play? What is the chemical equation? "
        "How does this relate to the carbon cycle and climate change?"
    ),
    # code (should be highly structured, repetitive patterns)
    (
        "python code",
        "Write a Python function that implements binary search on a sorted list. "
        "Include docstrings, type hints, and edge case handling. "
        "Then write unit tests for it using pytest."
    ),
    # creative/abstract (should have big residuals, phase transitions)
    (
        "abstract",
        "What would happen if gravity suddenly worked in reverse for exactly "
        "one second every hour? How would civilization adapt? What would "
        "architecture look like? How would sports change? What about the oceans?"
    ),
]


def run_experiment():
    print("=" * 70)
    print("  ENGRAM REAL AIR TEST")
    print("  feeding actual LLM hidden states through the semantic trajectory codec")
    print("=" * 70)
    print()

    model, tokenizer = load_qwen()
    hidden_size = model.config.hidden_size
    S = build_scramble_matrix(hidden_size, ENGRAM_PROJ_DIM)

    # alpha calibrated from actual Qwen projection magnitudes:
    # raw S@h has median |value| ~90, so alpha=0.005 gives tanh(0.005*90) = tanh(0.45) ≈ 0.42
    # this puts 85% of values in the gradient zone where the oscillator can see momentum
    alpha = float(os.environ.get("ENGRAM_ALPHA", "0.005"))
    print(f"alpha = {alpha}")
    codec = EngramCodec(alpha=alpha)

    print(f"\nscramble matrix: [{ENGRAM_PROJ_DIM}, {S.shape[1]}]")
    print(f"projection: tanh(4.0 * S @ h_t)")
    print(f"oscillator families: {' | '.join(f'{n}({e-s}D)' for n, s, e in OSCILLATOR_FAMILIES)}")
    print()

    all_results = []

    for name, prompt in CONVERSATIONS:
        print(f"--- {name} ---")
        print(f"  prompt: {prompt[:80]}...")

        # extract hidden states
        t0 = time.time()
        hidden, token_ids = extract_hidden_states(model, tokenizer, prompt)
        extract_time = time.time() - t0
        T = hidden.shape[0]
        print(f"  tokens: {T}, hidden_size: {hidden.shape[1]}, extract: {extract_time:.2f}s")

        # soft witness projection
        witness_np = soft_witness(hidden.numpy(), S, alpha=alpha)
        print(f"  witness range: [{witness_np.min():.4f}, {witness_np.max():.4f}]")
        print(f"  witness mean |w|: {np.abs(witness_np).mean():.4f}")

        # how many dimensions are near saturation (|w| > 0.99)?
        saturated = np.sum(np.abs(witness_np) > 0.99) / witness_np.size * 100
        # how many are in the gradient zone (0.1 < |w| < 0.9)?
        gradient_zone = np.sum((np.abs(witness_np) > 0.1) & (np.abs(witness_np) < 0.9)) / witness_np.size * 100
        print(f"  saturation: {saturated:.1f}% near +/-1, {gradient_zone:.1f}% in gradient zone")

        # trajectory smoothness: autocorrelation of witness velocity
        if T > 2:
            velocity = np.diff(witness_np, axis=0)  # [T-1, 256]
            vel_norm = np.linalg.norm(velocity, axis=1)
            mean_vel = float(vel_norm.mean())
            std_vel = float(vel_norm.std())
            # autocorrelation of velocity norms (lag-1)
            if len(vel_norm) > 2:
                vn = vel_norm - vel_norm.mean()
                autocorr = float(np.correlate(vn[:-1], vn[1:])[0] / (np.dot(vn, vn) + 1e-10))
            else:
                autocorr = 0.0
            print(f"  velocity: mean={mean_vel:.4f}, std={std_vel:.4f}, autocorr(1)={autocorr:.4f}")
        else:
            mean_vel = std_vel = autocorr = 0.0

        # encode with Engram
        t0 = time.time()
        packet = codec.encode_trajectory(witness_np)
        encode_time = time.time() - t0
        summary = engram_summary(packet)

        print(f"  encode: {encode_time:.3f}s")
        print(f"  blocks: {summary['n_blocks']}")
        print(f"  modes: {summary['mode_distribution']}")
        print(f"  residual RMS: {summary['residual_rms']:.6f}")

        # decode and measure fidelity
        decoded = codec.decode_trajectory(packet)
        roundtrip_err = float(np.sqrt(np.mean((witness_np - decoded) ** 2)))
        print(f"  roundtrip RMS error: {roundtrip_err:.8f}")

        # wire format compression
        wire = encode_engram(packet, codec.q_bits)
        raw_size = witness_np.size * 4  # float32
        bipolar_size = T * ENGRAM_PROJ_DIM  # int8 (current HyperCache storage)
        ratio_vs_float = raw_size / max(len(wire), 1)
        ratio_vs_bipolar = bipolar_size / max(len(wire), 1)
        print(f"  raw float32: {raw_size:,} bytes")
        print(f"  bipolar int8: {bipolar_size:,} bytes")
        print(f"  engram wire: {len(wire):,} bytes")
        print(f"  compression vs float32: {ratio_vs_float:.1f}x")
        print(f"  compression vs int8: {ratio_vs_bipolar:.2f}x")

        # wire roundtrip fidelity
        wire_decoded = codec.decode_bytes(wire)
        wire_err = float(np.sqrt(np.mean((witness_np - wire_decoded) ** 2)))
        print(f"  wire roundtrip error: {wire_err:.6f}")

        # per-family analysis
        print(f"  per-family residual energy:")
        for fi, (fname, ds, de) in enumerate(OSCILLATOR_FAMILIES):
            family_resid = sum(
                float(np.sum(b.residuals[:, ds:de] ** 2))
                for b in packet.blocks
            )
            family_total = sum(
                float(np.sum(witness_np[b.start_idx:b.start_idx+b.length, ds:de] ** 2))
                for b in packet.blocks
                if b.start_idx + b.length <= T
            )
            if family_total > 1e-8:
                captured = 1.0 - family_resid / family_total
                print(f"    {fname:8s} ({de-ds:3d}D): {captured*100:.1f}% captured by oscillator")
            else:
                print(f"    {fname:8s} ({de-ds:3d}D): (no energy)")

        # parameter vector for handshake
        pv = engram_parameter_vector(packet)
        print(f"  parameter vector: {len(pv)} floats")

        print()

        all_results.append({
            "name": name,
            "tokens": T,
            "modes": summary["mode_distribution"],
            "residual_rms": summary["residual_rms"],
            "compression_vs_float": ratio_vs_float,
            "compression_vs_int8": ratio_vs_bipolar,
            "wire_error": wire_err,
            "velocity_autocorr": autocorr,
            "saturated_pct": saturated,
            "gradient_pct": gradient_zone,
        })

    # ── cross-conversation comparison ─────────────────────────────────
    print("=" * 70)
    print("  CROSS-CONVERSATION ORBITAL COMPARISON")
    print("  does K,G similarity correlate with semantic similarity?")
    print("=" * 70)
    print()

    # re-encode all conversations and compare parameter vectors
    param_vectors = []
    for name, prompt in CONVERSATIONS:
        hidden, _ = extract_hidden_states(model, tokenizer, prompt)
        w = soft_witness(hidden.numpy(), S, alpha=alpha)
        pkt = codec.encode_trajectory(w)
        param_vectors.append((name, engram_parameter_vector(pkt)))

    from engram import engram_parameter_distance

    print(f"{'':18s}", end="")
    for name, _ in param_vectors:
        print(f"{name:>14s}", end="")
    print()

    for i, (name_i, pv_i) in enumerate(param_vectors):
        print(f"{name_i:18s}", end="")
        for j, (name_j, pv_j) in enumerate(param_vectors):
            if i == j:
                print(f"{'---':>14s}", end="")
            else:
                # use cosine similarity instead of distance
                pi, pj = pv_i, pv_j
                if len(pi) < len(pj):
                    pi = np.pad(pi, (0, len(pj) - len(pi)))
                elif len(pj) < len(pi):
                    pj = np.pad(pj, (0, len(pi) - len(pj)))
                dot = float(np.dot(pi, pj))
                norm = float(np.linalg.norm(pi) * np.linalg.norm(pj))
                cos = dot / max(norm, 1e-8)
                print(f"{cos:14.4f}", end="")
        print()

    # ── summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    total_orbital = sum(r["modes"].get("ORBITAL", 0) for r in all_results)
    total_linear = sum(r["modes"].get("LINEAR", 0) for r in all_results)
    total_static = sum(r["modes"].get("STATIC", 0) for r in all_results)
    total_reset = sum(r["modes"].get("RESET", 0) for r in all_results)
    total_blocks = total_orbital + total_linear + total_static + total_reset

    print(f"  Q1: Block mode distribution")
    print(f"      ORBITAL: {total_orbital}/{total_blocks} ({100*total_orbital/max(total_blocks,1):.0f}%)")
    print(f"      LINEAR:  {total_linear}/{total_blocks} ({100*total_linear/max(total_blocks,1):.0f}%)")
    print(f"      STATIC:  {total_static}/{total_blocks} ({100*total_static/max(total_blocks,1):.0f}%)")
    print(f"      RESET:   {total_reset}/{total_blocks} ({100*total_reset/max(total_blocks,1):.0f}%)")
    if total_orbital + total_static > total_blocks * 0.5:
        print(f"      --> THE THOUGHT HAS GRAVITY.")
    elif total_reset > total_blocks * 0.5:
        print(f"      --> thoughts are too jagged. the oscillator failed.")
    else:
        print(f"      --> mixed results. the oscillator captures some structure.")
    print()

    mean_ratio = np.mean([r["compression_vs_float"] for r in all_results])
    mean_ratio_int8 = np.mean([r["compression_vs_int8"] for r in all_results])
    print(f"  Q2: Compression ratio on real thought")
    print(f"      vs float32: {mean_ratio:.1f}x average")
    print(f"      vs int8 (current HyperCache): {mean_ratio_int8:.2f}x average")
    if mean_ratio >= 3.0:
        print(f"      --> COMPRESSION WORKS. without Logos even hooked up.")
    elif mean_ratio >= 2.0:
        print(f"      --> decent compression. Logos will push it further.")
    else:
        print(f"      --> compression marginal. data is near-random in projection space.")
    print()

    mean_autocorr = np.mean([r["velocity_autocorr"] for r in all_results])
    mean_resid = np.mean([r["residual_rms"] for r in all_results])
    print(f"  Q3: Does semantic trajectory have momentum?")
    print(f"      velocity autocorrelation: {mean_autocorr:.4f}")
    print(f"      mean residual RMS: {mean_resid:.4f}")
    if mean_autocorr > 0.1:
        print(f"      --> YES. thoughts have inertia. the latent space has physics.")
    elif mean_autocorr > 0.0:
        print(f"      --> weak momentum. some structure, not fully smooth.")
    else:
        print(f"      --> no momentum detected. thoughts are memoryless.")


if __name__ == "__main__":
    run_experiment()
