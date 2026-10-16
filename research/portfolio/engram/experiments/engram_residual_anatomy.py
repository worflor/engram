"""
engram_residual_anatomy.py

Dissects the 73.6% uncaptured energy in Engram v3 residuals.

The oscillators capture continuous dynamics (orbits, jitter).
What's left? Two hypotheses:

    H1: Symbolic patterns — discrete vocabulary attractor hopping that
        repeats. The M-axis (exact match) and A-axis (structural class)
        from Logos would catch these.

    H2: Irreducible surprise — genuine noise floor. No predictor helps.

To test: implement lightweight versions of Logos' prediction axes
on the quantized residual byte stream:

    L-axis: bit-context model (255 binary contexts, like Logos)
    M-axis: exact-match byte predictor (sliding window)
    A-axis: nibble-class predictor (top 4 bits = byte class)

Measure conditional entropy under each predictor. The gap between
unconditional entropy H0 and conditional entropy H(byte|context)
IS the predictable structure that Logos would capture.
"""

from __future__ import annotations
import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from engram import soft_witness, ENGRAM_PROJ_DIM
from engram_v3 import EngramCodecV3, PROJ_DIM, v3_summary
from engram_v2 import quantize_per_pair, _pack_varints
from engram_real_air import load_qwen, build_scramble_matrix
from engram_generation_test import generate_with_hidden_states


# ── entropy tools ─────────────────────────────────────────────────────

def H0(data: bytes) -> float:
    """Unconditional byte entropy (bits/byte)."""
    if not data: return 0.0
    c = np.zeros(256, dtype=np.int64)
    for b in data: c[b] += 1
    p = c[c > 0] / len(data)
    return float(-np.sum(p * np.log2(p)))


# ── L-axis: bit-context model ────────────────────────────────────────
#
# same principle as Logos' 255-node binary tree.
# for each byte, predict bits MSB-first. context = partial byte decoded so far.
# count c0, c1 per context node (Laplace prior).
# conditional entropy = average -log2(p(bit|context)).

def bit_context_entropy(data: bytes) -> float:
    """
    Measure conditional entropy under Logos-style L-axis bit predictor.
    255 contexts × 2 counters = 510 uint32s.
    """
    if not data: return 0.0

    counts = np.ones((256, 2), dtype=np.float64)  # Laplace prior
    total_bits = 0.0
    n_bits = 0

    for byte_val in data:
        ctx = 1  # root of bit tree
        for bit_pos in range(7, -1, -1):
            bit = (byte_val >> bit_pos) & 1
            c0, c1 = counts[ctx, 0], counts[ctx, 1]
            total = c0 + c1
            p = c1 / total if bit == 1 else c0 / total
            total_bits += -math.log2(max(p, 1e-15))
            counts[ctx, bit] += 1
            ctx = (ctx << 1) | bit
            if ctx >= 256:
                break  # tree is full, stop

    return total_bits / max(len(data), 1)  # bits per byte


# ── M-axis: exact match byte predictor ────────────────────────────────
#
# for each position, look back in a window for exact byte matches.
# if the previous N bytes match a prior context, predict the next byte.
# measure: what fraction of bytes are correctly predicted?

def exact_match_analysis(data: bytes, ctx_len: int = 4, window: int = 1024) -> dict:
    """
    Measure exact-match predictability (M-axis proxy).
    Returns: hit_rate, conditional entropy, unique_contexts.
    """
    if len(data) < ctx_len + 1:
        return {"hit_rate": 0.0, "cond_entropy": 8.0, "unique_ctx": 0}

    # build context → next-byte histogram
    ctx_counts: dict[bytes, np.ndarray] = {}
    for i in range(ctx_len, len(data)):
        ctx = bytes(data[i - ctx_len:i])
        if ctx not in ctx_counts:
            ctx_counts[ctx] = np.zeros(256, dtype=np.int64)
        ctx_counts[ctx][data[i]] += 1

    # compute conditional entropy: H(byte | context)
    total_bits = 0.0
    total_count = 0
    hits = 0

    for i in range(ctx_len, len(data)):
        ctx = bytes(data[i - ctx_len:i])
        hist = ctx_counts.get(ctx)
        if hist is None:
            total_bits += 8.0
            total_count += 1
            continue

        total = hist.sum()
        if total < 2:  # not enough evidence
            total_bits += 8.0
            total_count += 1
            continue

        # predict: most likely byte under this context
        predicted = int(np.argmax(hist))
        if predicted == data[i]:
            hits += 1

        # conditional probability
        p = (hist[data[i]] + 1) / (total + 256)  # Laplace smoothed
        total_bits += -math.log2(max(p, 1e-15))
        total_count += 1

    return {
        "hit_rate": hits / max(total_count, 1),
        "cond_entropy": total_bits / max(total_count, 1),
        "unique_ctx": len(ctx_counts),
    }


# ── A-axis: nibble-class predictor ────────────────────────────────────
#
# byte class = top nibble (byte >> 4). 16 classes.
# predict bytes conditioned on the class of the previous byte.
# captures structural patterns like "after a small residual (class 0),
# another small residual is likely."

def nibble_class_analysis(data: bytes) -> dict:
    """
    Measure structural class predictability (A-axis proxy).
    """
    if len(data) < 2:
        return {"cond_entropy": 8.0, "class_entropy": 4.0}

    # 16 previous classes × 256 next-byte histograms
    class_counts = np.ones((16, 256), dtype=np.float64)  # Laplace

    total_bits = 0.0
    total_count = 0
    class_bits = 0.0

    for i in range(1, len(data)):
        prev_class = data[i - 1] >> 4
        cur_byte = data[i]
        cur_class = cur_byte >> 4

        # byte prediction conditioned on prev class
        hist = class_counts[prev_class]
        total = hist.sum()
        p = hist[cur_byte] / total
        total_bits += -math.log2(max(p, 1e-15))

        # class prediction (4-bit)
        class_hist = hist.reshape(16, 16).sum(axis=1)
        class_p = class_hist[cur_class] / class_hist.sum()
        class_bits += -math.log2(max(class_p, 1e-15))

        class_counts[prev_class, cur_byte] += 1
        total_count += 1

    return {
        "cond_entropy": total_bits / max(total_count, 1),
        "class_entropy": class_bits / max(total_count, 1),
    }


# ── order-1 byte entropy (simple inter-byte correlations) ────────────

def order1_entropy(data: bytes) -> float:
    """H(byte_n | byte_{n-1}). Measures pairwise predictability."""
    if len(data) < 2: return 8.0
    # 256 × 256 transition matrix
    trans = np.ones((256, 256), dtype=np.float64)  # Laplace
    for i in range(1, len(data)):
        trans[data[i-1], data[i]] += 1

    total_bits = 0.0
    for i in range(1, len(data)):
        row = trans[data[i-1]]
        p = row[data[i]] / row.sum()
        total_bits += -math.log2(max(p, 1e-15))
    return total_bits / (len(data) - 1)


# ── the anatomy ───────────────────────────────────────────────────────

PROMPTS = [
    ("engine", "Explain how a four-stroke combustion engine works:"),
    ("story",  "Once upon a time, in a lighthouse on a rocky shore,"),
    ("code",   "def binary_search(arr, target):"),
    ("reason", "If all roses are flowers and all flowers need water, then"),
    ("create", "The last human on Mars opened the airlock and saw"),
]

def run():
    print("=" * 70)
    print("  ENGRAM RESIDUAL ANATOMY")
    print("  what IS the 73.6% uncaptured energy?")
    print("  structured patterns? or irreducible noise?")
    print("=" * 70)
    print()

    model, tokenizer = load_qwen()
    S = build_scramble_matrix(model.config.hidden_size, PROJ_DIM)
    alpha = float(os.environ.get("ENGRAM_ALPHA", "0.005"))
    max_gen = int(os.environ.get("ENGRAM_MAX_GEN", "200"))
    codec = EngramCodecV3(alpha=alpha)

    all_h0 = []
    all_l = []
    all_o1 = []
    all_m = []
    all_a = []

    for name, prompt in PROMPTS:
        print(f"--- {name} ---")
        hidden, _, text = generate_with_hidden_states(model, tokenizer, prompt, max_gen)
        pl = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        w_gen = soft_witness(hidden.numpy(), S, alpha=alpha)[pl:]

        if w_gen.shape[0] < 4:
            print("  too short")
            continue

        # encode and extract residual byte stream
        pkt = codec.encode_trajectory(w_gen)
        s = v3_summary(pkt)

        resid_bytes = bytearray()
        for blk in pkt.blocks:
            q, _ = quantize_per_pair(blk.residuals, 8)
            resid_bytes.extend(_pack_varints(q.ravel().astype(np.int32)))
        resid = bytes(resid_bytes)

        print(f"  gen tokens: {w_gen.shape[0]}")
        print(f"  v3 capture: macro={s['mean_macro_capture']:.1f}% micro={s['mean_micro_capture']:.1f}%")
        print(f"  residual stream: {len(resid):,} bytes")
        print()

        # entropy layers
        h0 = H0(resid)
        l_ent = bit_context_entropy(resid)
        o1_ent = order1_entropy(resid)
        m_res = exact_match_analysis(resid, ctx_len=4, window=2048)
        a_res = nibble_class_analysis(resid)

        all_h0.append(h0)
        all_l.append(l_ent)
        all_o1.append(o1_ent)
        all_m.append(m_res["cond_entropy"])
        all_a.append(a_res["cond_entropy"])

        print(f"  ENTROPY LAYERS (bits/byte, lower = more predictable):")
        print(f"    H0 (unconditional):     {h0:.3f}")
        print(f"    Order-1 (prev byte):    {o1_ent:.3f}  ({h0-o1_ent:+.3f} gain)")
        print(f"    L-axis (bit context):   {l_ent:.3f}  ({h0-l_ent:+.3f} gain)")
        print(f"    A-axis (nibble class):  {a_res['cond_entropy']:.3f}  ({h0-a_res['cond_entropy']:+.3f} gain)")
        print(f"    M-axis (exact 4B ctx):  {m_res['cond_entropy']:.3f}  ({h0-m_res['cond_entropy']:+.3f} gain)")
        print(f"    M-axis hit rate:        {m_res['hit_rate']*100:.1f}%")
        print(f"    M-axis unique contexts: {m_res['unique_ctx']:,}")
        print(f"    A-axis class entropy:   {a_res['class_entropy']:.3f} bits/class (of 4.0 max)")
        print()

        # what fraction of the gap can each axis close?
        gap = 8.0 - h0  # total compressible bits
        l_gain = h0 - l_ent
        m_gain = h0 - m_res["cond_entropy"]
        a_gain = h0 - a_res["cond_entropy"]
        o1_gain = h0 - o1_ent
        print(f"  GAP ANALYSIS:")
        print(f"    total gap to random:    {gap:.3f} bits/byte")
        print(f"    L-axis closes:          {l_gain:.3f} ({l_gain/max(gap,0.001)*100:.0f}% of gap)")
        print(f"    Order-1 closes:         {o1_gain:.3f} ({o1_gain/max(gap,0.001)*100:.0f}% of gap)")
        print(f"    M-axis closes:          {m_gain:.3f} ({m_gain/max(gap,0.001)*100:.0f}% of gap)")
        print(f"    A-axis closes:          {a_gain:.3f} ({a_gain/max(gap,0.001)*100:.0f}% of gap)")
        best = max(l_gain, m_gain, a_gain)
        print(f"    best single axis:       {best:.3f} ({best/max(gap,0.001)*100:.0f}% of gap)")
        print()

    # verdict
    print("=" * 70)
    print("  VERDICT: ANATOMY OF THE UNCAPTURED 73.6%")
    print("=" * 70)
    print()
    if all_h0:
        mh0 = np.mean(all_h0)
        ml = np.mean(all_l)
        mo1 = np.mean(all_o1)
        mm = np.mean(all_m)
        ma = np.mean(all_a)
        gap = 8.0 - mh0

        print(f"  average across {len(all_h0)} conversations:")
        print(f"    H0 (unconditional):    {mh0:.3f} bits/byte")
        print(f"    L-axis (bit context):  {ml:.3f}  ({mh0-ml:+.3f})")
        print(f"    Order-1 (prev byte):   {mo1:.3f}  ({mh0-mo1:+.3f})")
        print(f"    M-axis (exact match):  {mm:.3f}  ({mh0-mm:+.3f})")
        print(f"    A-axis (class pred):   {ma:.3f}  ({mh0-ma:+.3f})")
        print()
        print(f"  total gap to random: {gap:.3f} bits/byte")
        print(f"  H0 already captures: {gap:.3f} of 8.0 ({gap/8*100:.0f}%)")
        print(f"  L-axis adds:         {mh0-ml:.3f} beyond H0")
        print(f"  M-axis adds:         {mh0-mm:.3f} beyond H0")
        print(f"  A-axis adds:         {mh0-ma:.3f} beyond H0")
        print()

        # best combined estimate (L + M are somewhat orthogonal)
        # conservative: take max of individual gains × 1.3 for combination
        best_individual = max(mh0 - ml, mh0 - mm, mh0 - ma)
        est_combined = min(best_individual * 1.5, gap)  # conservative combo
        final_ent = mh0 - est_combined
        print(f"  estimated Logos combined: ~{est_combined:.3f} bits/byte gain")
        print(f"  estimated final entropy:  ~{final_ent:.3f} bits/byte")
        print(f"  estimated compression vs raw residuals: {mh0/max(final_ent, 0.1):.2f}x")
        print()

        if best_individual > 0.3:
            print(f"  --> RESIDUALS HAVE SYMBOLIC STRUCTURE.")
            print(f"     creativity has grammar. Logos will find it.")
        elif best_individual > 0.1:
            print(f"  --> weak symbolic structure. some patterns.")
        else:
            print(f"  --> residuals are near-random. this is the entropy floor.")


if __name__ == "__main__":
    run()
