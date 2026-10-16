"""
engram_v3.py

Whisper Engram v3 — Hierarchical Oscillators

generation is a superposition of two dynamics:
    MACRO: the slow orbit. topic, genre, register. changes every 30-100 tokens.
    MICRO: the fast jitter. vocabulary attractors, token-level decisions.
           a drunkard's walk between peaks in vocab space.

v2 tried to capture both with one oscillator per pair per block.
it caught the macro (100% ORBITAL) but missed the micro (4-9% capture).

v3 separates them. two cascaded oscillators per pair:
    1. MACRO oscillator: fitted on the DECIMATED trajectory (every 4th token).
       captures topic momentum. slow K, slow G. one fit per block.
    2. MICRO oscillator: fitted on the RESIDUALS of the macro, at full resolution.
       captures vocabulary jitter. fast K, fast G. one fit per sub-block (8 tokens).

this is the same cascaded prediction pattern as the audio codec:
    audio:  harmonic (fundamental) → cascaded (overtones) → BCM (residuals)
    engram: macro (topic orbit) → micro (vocab jitter) → Logos (surprises)

the final residuals are the genuine surprises — moments where the model's
trajectory departed from BOTH the topic orbit AND the vocabulary pattern.

    the macro oscillator is the fundamental frequency of the thought.
    the micro oscillator is the harmonic series.
    the residuals are the noise floor.
"""

from __future__ import annotations

import struct
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


# ── constants ─────────────────────────────────────────────────────────

PROJ_DIM = 256
N_PAIRS = PROJ_DIM // 2
DEFAULT_ALPHA = 0.005

# macro: fitted on decimated trajectory
MACRO_DECIMATION = 4     # use every 4th token for macro fit
MACRO_MIN_BLOCK = 16     # minimum tokens in a macro block
MACRO_MAX_BLOCK = 64

# micro: fitted on residuals of macro at full rate
MICRO_BLOCK_SIZE = 8     # 8-token sub-blocks for micro oscillator

MAGIC = b"E3"
VERSION = 3


class Mode(IntEnum):
    CASCADED = 0   # macro + micro oscillators
    LINEAR   = 1   # constant velocity (both levels)
    STATIC   = 2   # hold
    RESET    = 3   # raw


# ── per-pair AR(2) fitting (from v2, kept lean) ───────────────────────

def _fit_pair(z: np.ndarray) -> Tuple[complex, complex, float]:
    """Fit AR(2) on a single complex time series. Returns (K, G, rms)."""
    T = z.shape[0]
    if T < 3:
        return 1.0 + 0j, 0.0 + 0j, 0.0

    A, B, tgt = z[1:-1], z[:-2], z[2:]
    sAA = float(np.sum(np.abs(A) ** 2))
    sBB = float(np.sum(np.abs(B) ** 2))
    sAB = complex(np.sum(A * np.conj(B)))
    sTA = complex(np.sum(tgt * np.conj(A)))
    sTB = complex(np.sum(tgt * np.conj(B)))

    ridge = 1e-4 * (sAA + sBB + 1.0)
    det = (sAA + ridge) * (sBB + ridge) - abs(sAB) ** 2
    if abs(det) < 1e-12:
        return 2.0 + 0j, 1.0 + 0j, _rms(z, 2.0 + 0j, 1.0 + 0j)

    K = (sTA * (sBB + ridge) - np.conj(sAB) * sTB) / det
    G = (sAB * sTA - (sAA + ridge) * sTB) / det
    return K, G, _rms(z, K, G)


def _rms(z, K, G):
    if z.shape[0] < 3: return 0.0
    err = z[2:] - (K * z[1:-1] - G * z[:-2])
    return float(np.sqrt(np.mean(np.abs(err) ** 2)))


def _predict_pair(s1: complex, s2: complex, K: complex, G: complex, length: int) -> np.ndarray:
    """Generate predicted complex trajectory."""
    out = np.empty(length, dtype=np.complex128)
    prev1, prev2 = s1, s2
    for n in range(length):
        out[n] = K * prev1 - G * prev2
        prev2, prev1 = prev1, out[n]
    return out


# ── vectorized all-pairs fitting ──────────────────────────────────────

def _to_complex(w: np.ndarray) -> np.ndarray:
    """[T, 256] float32 → [T, 128] complex128."""
    return w.reshape(w.shape[0], N_PAIRS, 2)[..., 0] + 1j * w.reshape(w.shape[0], N_PAIRS, 2)[..., 1]


def _from_complex(z: np.ndarray) -> np.ndarray:
    """[T, 128] complex128 → [T, 256] float32."""
    T = z.shape[0]
    out = np.empty((T, PROJ_DIM), dtype=np.float32)
    out[:, 0::2] = z.real.astype(np.float32)
    out[:, 1::2] = z.imag.astype(np.float32)
    return out


def fit_all_pairs(z_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit 128 independent AR(2) oscillators.
    z_c: [T, 128] complex
    Returns: K[128], G[128] complex arrays.
    """
    K = np.empty(N_PAIRS, dtype=np.complex128)
    G = np.empty(N_PAIRS, dtype=np.complex128)
    for pi in range(N_PAIRS):
        K[pi], G[pi], _ = _fit_pair(z_c[:, pi])
    return K, G


def predict_all_pairs(
    s1_c: np.ndarray, s2_c: np.ndarray,
    K: np.ndarray, G: np.ndarray,
    length: int,
) -> np.ndarray:
    """Predict [length, 128] complex trajectory from seeds + coefficients."""
    out = np.empty((length, N_PAIRS), dtype=np.complex128)
    prev1 = s1_c.copy()
    prev2 = s2_c.copy()
    for n in range(length):
        out[n] = K * prev1 - G * prev2
        prev2, prev1 = prev1, out[n]
    return out


# ── data structures ───────────────────────────────────────────────────

@dataclass(slots=True)
class CascadedBlock:
    """One block with macro + micro oscillator hierarchy."""
    mode: Mode
    start_idx: int
    length: int

    # macro oscillator (one K,G per pair, fitted on decimated trajectory)
    macro_K: np.ndarray    # [128] complex
    macro_G: np.ndarray    # [128] complex

    # micro oscillator (one K,G per pair per sub-block)
    micro_Ks: List[np.ndarray]  # list of [128] complex, one per sub-block
    micro_Gs: List[np.ndarray]

    # final residuals after both oscillators
    residuals: np.ndarray   # [length, 256] float32

    # seeds
    seed1: np.ndarray       # [256]
    seed2: np.ndarray       # [256]

    # per-pair quantization scales
    per_pair_scale: np.ndarray  # [128]

    # diagnostics
    macro_capture: float = 0.0
    micro_capture: float = 0.0


@dataclass(slots=True)
class Packet:
    proj_dim: int
    token_count: int
    alpha: float
    blocks: List[CascadedBlock]


# ── hierarchical encode ──────────────────────────────────────────────

def encode_block_cascaded(
    block_data: np.ndarray,    # [L, 256]
    seed1: np.ndarray,         # [256]
    seed2: np.ndarray,         # [256]
) -> CascadedBlock:
    """
    Cascaded oscillator encoding:

    1. MACRO: fit AR(2) on every-4th-token decimated trajectory.
       captures the slow topic orbit.

    2. Reconstruct macro prediction at full rate (interpolated).
       compute macro residuals.

    3. MICRO: fit AR(2) on 8-token sub-blocks of the macro residuals.
       captures vocabulary jitter patterns.

    4. Final residuals = macro_residuals - micro_prediction.
       these are the genuine surprises.
    """
    L = block_data.shape[0]

    if L < 3:
        return CascadedBlock(
            mode=Mode.RESET, start_idx=0, length=L,
            macro_K=np.zeros(N_PAIRS, dtype=np.complex128),
            macro_G=np.zeros(N_PAIRS, dtype=np.complex128),
            micro_Ks=[], micro_Gs=[],
            residuals=block_data.copy(),
            seed1=seed1.copy(), seed2=seed2.copy(),
            per_pair_scale=np.zeros(N_PAIRS, dtype=np.float32),
        )

    # ── MACRO: fit on full-rate trajectory ──────────────────────
    # captures the block-wide trend (topic orbit) across all L tokens.
    # same fit as v2, but the residuals then go to the micro level.
    z_full = np.vstack([seed2[np.newaxis], seed1[np.newaxis], block_data])
    z_c_full = _to_complex(z_full)

    macro_K, macro_G = fit_all_pairs(z_c_full)

    s2_c = _to_complex(seed2[np.newaxis])[0]
    s1_c = _to_complex(seed1[np.newaxis])[0]
    macro_pred_c = predict_all_pairs(s1_c, s2_c, macro_K, macro_G, L)
    macro_pred = _from_complex(macro_pred_c)

    # macro residuals
    macro_resid = block_data - macro_pred

    # macro energy capture
    signal_energy = float(np.sum(block_data ** 2))
    macro_resid_energy = float(np.sum(macro_resid ** 2))
    macro_capture = max(0.0, (1.0 - macro_resid_energy / max(signal_energy, 1e-10)) * 100)

    # ── MICRO: fit on sub-blocks of macro residuals ───────────────
    micro_Ks: List[np.ndarray] = []
    micro_Gs: List[np.ndarray] = []
    micro_pred_full = np.zeros_like(macro_resid)

    # sub-block segmentation
    n_sub = max(1, L // MICRO_BLOCK_SIZE)
    for si in range(n_sub):
        sub_start = si * MICRO_BLOCK_SIZE
        sub_end = min(sub_start + MICRO_BLOCK_SIZE, L)
        sub_len = sub_end - sub_start
        if sub_len < 3:
            micro_Ks.append(np.full(N_PAIRS, 1.0 + 0j))
            micro_Gs.append(np.full(N_PAIRS, 0.0 + 0j))
            continue

        # seeds for micro: last two samples before this sub-block
        if sub_start >= 2:
            m_seed2 = macro_resid[sub_start - 2]
            m_seed1 = macro_resid[sub_start - 1]
        elif sub_start == 1:
            m_seed2 = np.zeros(PROJ_DIM, dtype=np.float32)
            m_seed1 = macro_resid[0]
        else:
            m_seed2 = np.zeros(PROJ_DIM, dtype=np.float32)
            m_seed1 = np.zeros(PROJ_DIM, dtype=np.float32)

        # fit micro on this sub-block
        z_sub = np.vstack([m_seed2[np.newaxis], m_seed1[np.newaxis], macro_resid[sub_start:sub_end]])
        z_sub_c = _to_complex(z_sub)

        mK, mG = fit_all_pairs(z_sub_c)
        micro_Ks.append(mK)
        micro_Gs.append(mG)

        # predict micro sub-block
        ms2_c = _to_complex(m_seed2[np.newaxis])[0]
        ms1_c = _to_complex(m_seed1[np.newaxis])[0]
        micro_pred_c = predict_all_pairs(ms1_c, ms2_c, mK, mG, sub_len)
        micro_pred_full[sub_start:sub_end] = _from_complex(micro_pred_c)

    # final residuals: what's left after both oscillators
    final_resid = macro_resid - micro_pred_full

    # micro energy capture (relative to macro residuals)
    final_energy = float(np.sum(final_resid ** 2))
    micro_capture = max(0.0, (1.0 - final_energy / max(macro_resid_energy, 1e-10)) * 100)

    # per-pair quantization scales
    per_pair_scale = np.zeros(N_PAIRS, dtype=np.float32)
    for pi in range(N_PAIRS):
        d0, d1 = pi * 2, pi * 2 + 1
        per_pair_scale[pi] = float(np.abs(final_resid[:, d0:d1 + 1]).max())

    return CascadedBlock(
        mode=Mode.CASCADED,
        start_idx=0,  # filled in by caller
        length=L,
        macro_K=macro_K, macro_G=macro_G,
        micro_Ks=micro_Ks, micro_Gs=micro_Gs,
        residuals=final_resid,
        seed1=seed1.copy(), seed2=seed2.copy(),
        per_pair_scale=per_pair_scale,
        macro_capture=macro_capture,
        micro_capture=micro_capture,
    )


# ── hierarchical decode ──────────────────────────────────────────────

def decode_block_cascaded(block: CascadedBlock) -> np.ndarray:
    """Reconstruct trajectory from macro + micro + residuals."""
    L = block.length

    if block.mode == Mode.RESET:
        return block.residuals[:L].copy()

    # macro prediction
    s1_c = _to_complex(block.seed1[np.newaxis])[0]
    s2_c = _to_complex(block.seed2[np.newaxis])[0]
    macro_pred_c = predict_all_pairs(s1_c, s2_c, block.macro_K, block.macro_G, L)
    macro_pred = _from_complex(macro_pred_c)

    # compute macro residuals (we need them for micro seeds)
    # but we don't have the original — we reconstruct micro from stored params
    micro_pred = np.zeros((L, PROJ_DIM), dtype=np.float32)

    n_sub = len(block.micro_Ks)
    for si in range(n_sub):
        sub_start = si * MICRO_BLOCK_SIZE
        sub_end = min(sub_start + MICRO_BLOCK_SIZE, L)
        sub_len = sub_end - sub_start

        # micro seeds: derived from accumulated (micro_pred + residuals) of prior sub-blocks
        # this is the key insight: at decode time, we reconstruct macro_resid from
        # micro_prediction + final_residuals, then use that for seeding
        if sub_start >= 2:
            # macro_resid at these positions = micro_pred + final_resid
            m_seed2 = micro_pred[sub_start - 2] + block.residuals[sub_start - 2]
            m_seed1 = micro_pred[sub_start - 1] + block.residuals[sub_start - 1]
        elif sub_start == 1:
            m_seed2 = np.zeros(PROJ_DIM, dtype=np.float32)
            m_seed1 = micro_pred[0] + block.residuals[0]
        else:
            m_seed2 = np.zeros(PROJ_DIM, dtype=np.float32)
            m_seed1 = np.zeros(PROJ_DIM, dtype=np.float32)

        ms2_c = _to_complex(m_seed2[np.newaxis])[0]
        ms1_c = _to_complex(m_seed1[np.newaxis])[0]
        micro_pred_c = predict_all_pairs(ms1_c, ms2_c, block.micro_Ks[si], block.micro_Gs[si], sub_len)
        micro_pred[sub_start:sub_end] = _from_complex(micro_pred_c)

    # reconstruct: macro + micro + residuals
    return macro_pred + micro_pred + block.residuals


# ── block segmentation (prediction-error based) ──────────────────────

def segment_blocks(
    witness_path: np.ndarray,
    min_block: int = MACRO_MIN_BLOCK,
    max_block: int = MACRO_MAX_BLOCK,
    error_threshold: float = 2.0,
) -> List[Tuple[int, int]]:
    """Segment by linear prediction error spikes."""
    T = witness_path.shape[0]
    if T <= 2:
        return []

    err = np.zeros(T, dtype=np.float32)
    for t in range(2, T):
        pred = 2.0 * witness_path[t - 1] - witness_path[t - 2]
        err[t] = float(np.linalg.norm(witness_path[t] - pred))

    if T > 4:
        e_mean = float(np.mean(err[2:]))
        e_std = float(np.std(err[2:])) + 1e-8
    else:
        e_mean, e_std = 0.0, 1.0

    threshold = e_mean + error_threshold * e_std
    segments: List[Tuple[int, int]] = []
    block_start = 2

    for t in range(3, T):
        bl = t - block_start
        if bl >= max_block:
            segments.append((block_start, t))
            block_start = t
            continue
        if bl >= min_block and err[t] > threshold:
            segments.append((block_start, t))
            block_start = t

    if block_start < T:
        segments.append((block_start, T))

    if len(segments) > 1 and (segments[-1][1] - segments[-1][0]) < min_block:
        last = segments.pop()
        prev = segments.pop()
        segments.append((prev[0], last[1]))

    return segments


# ── the codec ─────────────────────────────────────────────────────────

class EngramCodecV3:
    """
    Whisper Engram v3: hierarchical oscillators.

    macro (topic orbit) → micro (vocab jitter) → residuals (surprises)

    like the audio codec separating fundamental from harmonics.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, max_block: int = MACRO_MAX_BLOCK):
        self.alpha = alpha
        self.max_block = max_block

    def encode_trajectory(self, witness_path: np.ndarray) -> Packet:
        T = witness_path.shape[0]
        w = witness_path.astype(np.float32)

        if T == 0:
            return Packet(PROJ_DIM, 0, self.alpha, [])
        if T <= 2:
            blk = CascadedBlock(
                mode=Mode.RESET, start_idx=0, length=T,
                macro_K=np.zeros(N_PAIRS, dtype=np.complex128),
                macro_G=np.zeros(N_PAIRS, dtype=np.complex128),
                micro_Ks=[], micro_Gs=[],
                residuals=w.copy(),
                seed1=w[-1].copy() if T >= 2 else w[0].copy(),
                seed2=w[0].copy(),
                per_pair_scale=np.zeros(N_PAIRS, dtype=np.float32),
            )
            return Packet(PROJ_DIM, T, self.alpha, [blk])

        segments = segment_blocks(w, max_block=self.max_block)
        blocks: List[CascadedBlock] = []

        for seg_start, seg_end in segments:
            block_data = w[seg_start:seg_end]
            seed2 = w[seg_start - 2].copy()
            seed1 = w[seg_start - 1].copy()

            blk = encode_block_cascaded(block_data, seed1, seed2)
            blk.start_idx = seg_start
            blocks.append(blk)

        return Packet(PROJ_DIM, T, self.alpha, blocks)

    def decode_trajectory(self, packet: Packet) -> np.ndarray:
        T = packet.token_count
        if T == 0:
            return np.empty((0, PROJ_DIM), dtype=np.float32)

        out = np.zeros((T, PROJ_DIM), dtype=np.float32)
        for blk in packet.blocks:
            if blk.mode == Mode.RESET and blk.start_idx == 0:
                L = min(blk.length, blk.residuals.shape[0])
                out[:L] = blk.residuals[:L]
                continue
            decoded = decode_block_cascaded(blk)
            out[blk.start_idx:blk.start_idx + blk.length] = decoded

        if len(packet.blocks) > 0 and packet.blocks[0].start_idx >= 2:
            out[0] = packet.blocks[0].seed2
            out[1] = packet.blocks[0].seed1

        return out


# ── diagnostics ───────────────────────────────────────────────────────

def v3_summary(packet: Packet) -> dict:
    mode_counts = {m.name: 0 for m in Mode}
    total_resid_sq = 0.0
    total_count = 0
    total_macro_cap = 0.0
    total_micro_cap = 0.0
    n_blocks = 0
    n_sub_blocks = 0

    for blk in packet.blocks:
        mode_counts[blk.mode.name] += 1
        total_resid_sq += float(np.sum(blk.residuals ** 2))
        total_count += blk.residuals.size
        total_macro_cap += blk.macro_capture
        total_micro_cap += blk.micro_capture
        n_blocks += 1
        n_sub_blocks += len(blk.micro_Ks)

    return {
        "token_count": packet.token_count,
        "n_blocks": n_blocks,
        "n_sub_blocks": n_sub_blocks,
        "modes": mode_counts,
        "residual_rms": round(float(np.sqrt(total_resid_sq / max(total_count, 1))), 6),
        "mean_macro_capture": round(total_macro_cap / max(n_blocks, 1), 1),
        "mean_micro_capture": round(total_micro_cap / max(n_blocks, 1), 1),
    }


def byte_entropy(data: bytes) -> float:
    if not data: return 0.0
    counts = np.zeros(256, dtype=np.int64)
    for b in data: counts[b] += 1
    probs = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


# ── self-test ─────────────────────────────────────────────────────────

def _self_test():
    print("=== Whisper Engram v3: Hierarchical Oscillators ===\n")
    print(f"macro: decimation={MACRO_DECIMATION}, block={MACRO_MIN_BLOCK}-{MACRO_MAX_BLOCK}")
    print(f"micro: sub-block={MICRO_BLOCK_SIZE}")
    print(f"oscillators: {N_PAIRS} pairs x 2 levels = {N_PAIRS * 2} oscillators")
    print()

    codec = EngramCodecV3()
    rng = np.random.RandomState(42)

    # test 1: constant
    print("--- constant ---")
    w = np.ones((50, PROJ_DIM), dtype=np.float32) * 0.3
    pkt = codec.encode_trajectory(w)
    dec = codec.decode_trajectory(pkt)
    err = float(np.sqrt(np.mean((w - dec) ** 2)))
    s = v3_summary(pkt)
    print(f"  roundtrip err: {err:.8f}")
    print(f"  macro capture: {s['mean_macro_capture']:.1f}%, micro: {s['mean_micro_capture']:.1f}%")

    # test 2: superposition of slow + fast oscillation
    print("\n--- superposition (slow orbit + fast jitter) ---")
    T = 120
    t = np.arange(T, dtype=np.float32)
    w2 = np.zeros((T, PROJ_DIM), dtype=np.float32)
    for pi in range(N_PAIRS):
        # slow orbit (macro): period ~40 tokens
        slow_freq = 0.05 + 0.005 * pi
        # fast jitter (micro): period ~6 tokens
        fast_freq = 0.4 + 0.03 * pi
        w2[:, pi*2]   = 0.3 * np.cos(slow_freq * t) + 0.1 * np.cos(fast_freq * t)
        w2[:, pi*2+1] = 0.3 * np.sin(slow_freq * t) + 0.1 * np.sin(fast_freq * t)
    w2 += 0.005 * rng.randn(T, PROJ_DIM).astype(np.float32)

    pkt2 = codec.encode_trajectory(w2)
    dec2 = codec.decode_trajectory(pkt2)
    err2 = float(np.sqrt(np.mean((w2 - dec2) ** 2)))
    s2 = v3_summary(pkt2)

    # total energy capture
    total_resid = sum(float(np.sum(b.residuals ** 2)) for b in pkt2.blocks)
    total_signal = float(np.sum(w2[2:] ** 2))
    total_capture = (1.0 - total_resid / total_signal) * 100

    print(f"  roundtrip err: {err2:.8f}")
    print(f"  blocks: {s2['n_blocks']}, sub-blocks: {s2['n_sub_blocks']}")
    print(f"  macro capture: {s2['mean_macro_capture']:.1f}%")
    print(f"  micro capture: {s2['mean_micro_capture']:.1f}% (of macro residuals)")
    print(f"  total capture: {total_capture:.1f}% (both levels combined)")
    print(f"  residual RMS: {s2['residual_rms']:.6f}")
    print(f"  modes: {s2['modes']}")

    # test 3: random (should not break)
    print("\n--- random ---")
    w3 = rng.randn(60, PROJ_DIM).astype(np.float32) * 0.3
    pkt3 = codec.encode_trajectory(w3)
    dec3 = codec.decode_trajectory(pkt3)
    err3 = float(np.sqrt(np.mean((w3 - dec3) ** 2)))
    s3 = v3_summary(pkt3)
    print(f"  roundtrip err: {err3:.8f}")
    print(f"  macro: {s3['mean_macro_capture']:.1f}%, micro: {s3['mean_micro_capture']:.1f}%")

    print("\n=== v3 self-test complete ===")


if __name__ == "__main__":
    _self_test()
