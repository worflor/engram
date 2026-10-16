"""
engram_v2.py

Whisper Engram v2 — Per-Pair Oscillators

v1 used 5 family oscillators (20 params/block). the 231D bond family
was a junk drawer — one K,G for 115 independent complex pairs. it
captured 1-3% of signal energy because it was fitting the average
momentum of an entire orchestra with one baton.

v2 uses 128 per-pair oscillators (512 params/block = 2KB). each pair
of dimensions gets its own K,G. every instrument gets its own conductor.

additionally:
    - block segmentation cuts on prediction error spikes, not velocity
    - per-pair adaptive quantization (no more uniform scaling)
    - TrajectorySource abstraction (the LLM is just one provider)

the codec still doesn't own the projection (that's v3). but with 128
oscillators, the prediction should capture dramatically more energy.
"""

from __future__ import annotations

import struct
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple, Protocol

import numpy as np


# ── constants ─────────────────────────────────────────────────────────

PROJ_DIM = 256
N_PAIRS = PROJ_DIM // 2     # 128 complex oscillator pairs
DEFAULT_ALPHA = 0.005        # calibrated for Qwen 3.5 projection scale
DEFAULT_MIN_BLOCK = 4
DEFAULT_MAX_BLOCK = 64
MAGIC = b"E2"
VERSION = 2


class Mode(IntEnum):
    ORBITAL = 0   # per-pair complex AR(2)
    LINEAR  = 1   # constant velocity: z[n] = 2·z[n-1] − z[n-2]
    STATIC  = 2   # topic hold: z[n] = z[n-1]
    RESET   = 3   # phase transition: raw seed


# ── trajectory source abstraction ─────────────────────────────────────

class TrajectorySource(Protocol):
    """Any source of high-dimensional temporal trajectories."""

    def project(self, raw_data: np.ndarray) -> np.ndarray:
        """Project raw data [T, D_raw] to trajectory space [T, PROJ_DIM]."""
        ...


class TanhProjection:
    """Soft witness: tanh(alpha * S @ h_t). Default for LLM hidden states."""

    def __init__(self, scramble_matrix: np.ndarray, alpha: float = DEFAULT_ALPHA):
        self.S = scramble_matrix.astype(np.float32)
        self.alpha = alpha

    def project(self, raw_data: np.ndarray) -> np.ndarray:
        h = raw_data[:, :self.S.shape[1]].astype(np.float32)
        return np.tanh(self.alpha * (h @ self.S.T)).astype(np.float32)

    @staticmethod
    def auto_alpha(scramble_matrix: np.ndarray, sample_hidden: np.ndarray) -> float:
        """Calibrate alpha from a sample of hidden states."""
        h = sample_hidden[:, :scramble_matrix.shape[1]].astype(np.float32)
        proj = h @ scramble_matrix.T
        median_abs = float(np.median(np.abs(proj)))
        if median_abs < 1e-6:
            return 1.0
        # target: tanh(alpha * median) ≈ 0.45 → alpha * median ≈ 0.485
        return 0.485 / median_abs


class IdentityProjection:
    """Pass-through for pre-projected data (audio features, sensor data, etc.)."""

    def project(self, raw_data: np.ndarray) -> np.ndarray:
        if raw_data.shape[1] != PROJ_DIM:
            raise ValueError(f"expected {PROJ_DIM} dims, got {raw_data.shape[1]}")
        return raw_data.astype(np.float32)


# ── data structures ───────────────────────────────────────────────────

@dataclass(slots=True)
class PairCoeffs:
    """Complex AR(2) coefficients for one dimension pair."""
    pair_idx: int     # which pair (0..127)
    kR: float = 0.0
    kI: float = 0.0
    gR: float = 0.0
    gI: float = 0.0


@dataclass(slots=True)
class Block:
    """One block of the Engram v2 coded trajectory."""
    mode: Mode
    start_idx: int
    length: int
    pair_coeffs: List[PairCoeffs]     # 128 per-pair K,G for ORBITAL
    residuals: np.ndarray             # [length, 256] float32
    seed1: np.ndarray                 # [256] float32
    seed2: np.ndarray                 # [256] float32
    per_pair_scale: np.ndarray        # [128] float32, quantization scale per pair
    fit_error: float = 0.0


@dataclass(slots=True)
class Packet:
    """Complete encoded trajectory."""
    proj_dim: int
    token_count: int
    alpha: float
    blocks: List[Block]


# ── per-pair AR(2) fitting ────────────────────────────────────────────

def fit_pair_ar2(z: np.ndarray) -> Tuple[complex, complex, float]:
    """
    Fit z[n] = K · z[n-1] − G · z[n-2] for a single complex pair.

    Parameters
    ----------
    z : [T] complex64

    Returns
    -------
    (K, G, rms_error)
    """
    T = z.shape[0]
    if T < 3:
        return complex(1.0, 0.0), complex(0.0, 0.0), 0.0

    target = z[2:]
    A = z[1:-1]
    B = z[:-2]

    sAA = float(np.real(np.sum(A * np.conj(A))))
    sBB = float(np.real(np.sum(B * np.conj(B))))
    sAB = complex(np.sum(A * np.conj(B)))
    sTA = complex(np.sum(target * np.conj(A)))
    sTB = complex(np.sum(target * np.conj(B)))

    # Tikhonov ridge
    ridge = 1e-4 * (sAA + sBB + 1.0)
    sAA_r = sAA + ridge
    sBB_r = sBB + ridge

    det = sAA_r * sBB_r - abs(sAB) ** 2
    if abs(det) < 1e-12:
        return complex(2.0, 0.0), complex(1.0, 0.0), _pair_rms(z, 2.0 + 0j, 1.0 + 0j)

    K = (sTA * sBB_r - np.conj(sAB) * sTB) / det
    G = (sAB * sTA - sAA_r * sTB) / det

    return K, G, _pair_rms(z, K, G)


def _pair_rms(z: np.ndarray, K: complex, G: complex) -> float:
    """RMS error for a single complex pair."""
    if z.shape[0] < 3:
        return 0.0
    pred = K * z[1:-1] - G * z[:-2]
    err = z[2:] - pred
    return float(np.sqrt(np.mean(np.abs(err) ** 2)))


def fit_all_pairs(z_full: np.ndarray) -> Tuple[List[PairCoeffs], float]:
    """
    Fit 128 independent complex AR(2) oscillators.

    Parameters
    ----------
    z_full : [T, 256] float32 (with 2 seed rows prepended)

    Returns
    -------
    (pair_coeffs, total_rms)
    """
    T = z_full.shape[0]
    # reshape to complex pairs: [T, 128]
    z_c = z_full.reshape(T, N_PAIRS, 2)
    z_c = z_c[..., 0] + 1j * z_c[..., 1]

    coeffs: List[PairCoeffs] = []
    total_err_sq = 0.0

    for pi in range(N_PAIRS):
        K, G, rms = fit_pair_ar2(z_c[:, pi])
        coeffs.append(PairCoeffs(pi, K.real, K.imag, G.real, G.imag))
        total_err_sq += rms * rms

    total_rms = float(np.sqrt(total_err_sq / N_PAIRS))
    return coeffs, total_rms


def linear_all_pairs_rms(z_full: np.ndarray) -> float:
    """RMS error if all pairs use linear prediction."""
    T = z_full.shape[0]
    if T < 3:
        return 0.0
    z_c = z_full.reshape(T, N_PAIRS, 2)
    z_c = z_c[..., 0] + 1j * z_c[..., 1]

    total = 0.0
    for pi in range(N_PAIRS):
        total += _pair_rms(z_c[:, pi], 2.0 + 0j, 1.0 + 0j) ** 2
    return float(np.sqrt(total / N_PAIRS))


# ── prediction ────────────────────────────────────────────────────────

def predict_block_v2(
    seed1: np.ndarray,
    seed2: np.ndarray,
    pair_coeffs: List[PairCoeffs],
    length: int,
    mode: Mode,
) -> np.ndarray:
    """Generate [length, 256] predicted trajectory."""
    out = np.empty((length, PROJ_DIM), dtype=np.float32)

    if mode == Mode.STATIC:
        for n in range(length):
            out[n] = seed1
        return out

    # convert seeds to complex pairs
    s1_c = seed1.reshape(N_PAIRS, 2)
    s1_c = s1_c[:, 0] + 1j * s1_c[:, 1]
    s2_c = seed2.reshape(N_PAIRS, 2)
    s2_c = s2_c[:, 0] + 1j * s2_c[:, 1]

    # vectorized: build K and G arrays
    K_arr = np.array([complex(pc.kR, pc.kI) for pc in pair_coeffs], dtype=np.complex128)
    G_arr = np.array([complex(pc.gR, pc.gI) for pc in pair_coeffs], dtype=np.complex128)

    prev1 = s1_c.astype(np.complex128)
    prev2 = s2_c.astype(np.complex128)

    for n in range(length):
        pred = K_arr * prev1 - G_arr * prev2
        out[n, 0::2] = pred.real.astype(np.float32)
        out[n, 1::2] = pred.imag.astype(np.float32)
        prev2 = prev1
        prev1 = pred

    return out


# ── per-pair adaptive quantization ────────────────────────────────────

def quantize_per_pair(residuals: np.ndarray, q_bits: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize residuals with per-pair scale factors.

    Returns (quantized [L, 256] int8, scales [128] float32).
    """
    L = residuals.shape[0]
    q = np.zeros((L, PROJ_DIM), dtype=np.int8)
    scales = np.zeros(N_PAIRS, dtype=np.float32)
    max_val = 2 ** (q_bits - 1) - 1

    for pi in range(N_PAIRS):
        d0, d1 = pi * 2, pi * 2 + 1
        pair_resid = residuals[:, d0:d1+1]  # [L, 2]
        max_abs = float(np.abs(pair_resid).max())
        if max_abs < 1e-10:
            scales[pi] = 0.0
            continue
        scales[pi] = max_abs
        scale = max_val / max_abs
        q[:, d0:d1+1] = np.clip(np.round(pair_resid * scale), -max_val, max_val).astype(np.int8)

    return q, scales


def dequantize_per_pair(quantized: np.ndarray, scales: np.ndarray, q_bits: int = 8) -> np.ndarray:
    """Inverse of quantize_per_pair."""
    L = quantized.shape[0]
    out = np.zeros((L, PROJ_DIM), dtype=np.float32)
    max_val = 2 ** (q_bits - 1) - 1

    for pi in range(N_PAIRS):
        d0, d1 = pi * 2, pi * 2 + 1
        if scales[pi] < 1e-10:
            continue
        out[:, d0:d1+1] = quantized[:, d0:d1+1].astype(np.float32) * (scales[pi] / max_val)

    return out


# ── block segmentation (prediction-error based) ──────────────────────

def segment_by_prediction_error(
    witness_path: np.ndarray,
    min_block: int = DEFAULT_MIN_BLOCK,
    max_block: int = DEFAULT_MAX_BLOCK,
    error_threshold: float = 2.0,
) -> List[Tuple[int, int]]:
    """
    Segment trajectory by AR(2) prediction error spikes.

    Instead of velocity magnitude (which doesn't correlate with
    semantic transitions), we use the oscillator's own failure signal:
    when the prediction suddenly gets much worse, cut the block.
    """
    T = witness_path.shape[0]
    if T <= 2:
        return []

    # compute per-token prediction error using linear predictor
    # (cheap proxy — actual AR(2) error needs fitted K,G)
    err = np.zeros(T, dtype=np.float32)
    for t in range(2, T):
        pred = 2.0 * witness_path[t - 1] - witness_path[t - 2]
        err[t] = float(np.linalg.norm(witness_path[t] - pred))

    # running statistics
    if T > 4:
        e_mean = float(np.mean(err[2:]))
        e_std = float(np.std(err[2:])) + 1e-8
    else:
        e_mean, e_std = 0.0, 1.0

    threshold = e_mean + error_threshold * e_std

    segments: List[Tuple[int, int]] = []
    block_start = 2

    for t in range(3, T):
        block_len = t - block_start
        if block_len >= max_block:
            segments.append((block_start, t))
            block_start = t
            continue
        if block_len >= min_block and err[t] > threshold:
            segments.append((block_start, t))
            block_start = t

    if block_start < T:
        segments.append((block_start, T))

    # merge tiny trailing blocks
    if len(segments) > 1 and (segments[-1][1] - segments[-1][0]) < min_block:
        last = segments.pop()
        prev = segments.pop()
        segments.append((prev[0], last[1]))

    return segments


# ── mode selection ────────────────────────────────────────────────────

def select_mode(
    block_data: np.ndarray,
    seed1: np.ndarray,
    seed2: np.ndarray,
) -> Tuple[Mode, List[PairCoeffs], float]:
    """Choose mode and fit per-pair coefficients."""
    L = block_data.shape[0]
    if L < 1:
        return Mode.RESET, [], 0.0

    # STATIC check
    if float(np.var(block_data, axis=0).mean()) < 1e-6:
        coeffs = [PairCoeffs(i, 1.0, 0.0, 0.0, 0.0) for i in range(N_PAIRS)]
        return Mode.STATIC, coeffs, 0.0

    z_full = np.vstack([seed2[np.newaxis], seed1[np.newaxis], block_data])

    # fit ORBITAL (128 independent oscillators)
    orb_coeffs, orb_rms = fit_all_pairs(z_full)
    lin_rms = linear_all_pairs_rms(z_full)

    if orb_rms < lin_rms:
        return Mode.ORBITAL, orb_coeffs, orb_rms
    else:
        lin_coeffs = [PairCoeffs(i, 2.0, 0.0, 1.0, 0.0) for i in range(N_PAIRS)]
        return Mode.LINEAR, lin_coeffs, lin_rms


# ── the codec ─────────────────────────────────────────────────────────

class EngramCodecV2:
    """
    Whisper Engram v2: per-pair oscillators.

    128 complex oscillators × 4 params = 512 params per block.
    Every instrument gets its own conductor.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        min_block: int = DEFAULT_MIN_BLOCK,
        max_block: int = DEFAULT_MAX_BLOCK,
        q_bits: int = 8,
    ):
        self.alpha = alpha
        self.min_block = min_block
        self.max_block = max_block
        self.q_bits = q_bits

    def encode_trajectory(self, witness_path: np.ndarray) -> Packet:
        T = witness_path.shape[0]
        w = witness_path.astype(np.float32)

        if T == 0:
            return Packet(PROJ_DIM, 0, self.alpha, [])

        if T <= 2:
            block = Block(
                mode=Mode.RESET, start_idx=0, length=T,
                pair_coeffs=[], residuals=w.copy(),
                seed1=w[-1].copy() if T >= 2 else w[0].copy(),
                seed2=w[0].copy(),
                per_pair_scale=np.zeros(N_PAIRS, dtype=np.float32),
            )
            return Packet(PROJ_DIM, T, self.alpha, [block])

        segments = segment_by_prediction_error(w, self.min_block, self.max_block)
        blocks: List[Block] = []

        for seg_start, seg_end in segments:
            block_data = w[seg_start:seg_end]
            L = block_data.shape[0]
            seed2 = w[seg_start - 2].copy()
            seed1 = w[seg_start - 1].copy()

            mode, coeffs, fit_err = select_mode(block_data, seed1, seed2)
            predicted = predict_block_v2(seed1, seed2, coeffs, L, mode)
            residuals = block_data - predicted

            # per-pair quantization
            q, scales = quantize_per_pair(residuals, self.q_bits)

            blocks.append(Block(
                mode=mode, start_idx=seg_start, length=L,
                pair_coeffs=coeffs, residuals=residuals,
                seed1=seed1, seed2=seed2,
                per_pair_scale=scales, fit_error=fit_err,
            ))

        return Packet(PROJ_DIM, T, self.alpha, blocks)

    def decode_trajectory(self, packet: Packet) -> np.ndarray:
        T = packet.token_count
        if T == 0:
            return np.empty((0, PROJ_DIM), dtype=np.float32)

        out = np.zeros((T, PROJ_DIM), dtype=np.float32)

        for block in packet.blocks:
            if block.mode == Mode.RESET and block.start_idx == 0:
                L = min(block.length, block.residuals.shape[0])
                out[:L] = block.residuals[:L]
                continue

            predicted = predict_block_v2(
                block.seed1, block.seed2,
                block.pair_coeffs, block.length, block.mode,
            )
            out[block.start_idx:block.start_idx + block.length] = predicted + block.residuals

        if len(packet.blocks) > 0 and packet.blocks[0].start_idx >= 2:
            out[0] = packet.blocks[0].seed2
            out[1] = packet.blocks[0].seed1

        return out


# ── zigzag + varint (from v1) ─────────────────────────────────────────

def _zigzag_enc(arr):
    a = arr.astype(np.int32)
    return ((a << 1) ^ (a >> 31)).astype(np.uint32)

def _zigzag_dec(arr):
    a = arr.astype(np.uint32)
    return ((a >> 1).astype(np.int32)) ^ (-(a & 1).astype(np.int32))

def _pack_varints(flat_int: np.ndarray) -> bytes:
    zz = _zigzag_enc(flat_int.ravel().astype(np.int32))
    out = bytearray()
    for v in zz:
        v = int(v) & 0xFFFFFFFF
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v & 0x7F)
    return bytes(out)

def _unpack_varints(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.int32)
    off = 0
    for i in range(count):
        result = 0; shift = 0
        while off < len(data):
            b = data[off]; off += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80): break
            shift += 7
        out[i] = _zigzag_dec(np.array([result & 0xFFFFFFFF], dtype=np.uint32))[0]
    return out


# ── wire format ───────────────────────────────────────────────────────

def encode_wire(packet: Packet, q_bits: int = 8) -> bytes:
    buf = bytearray()
    buf.extend(MAGIC)
    buf.append(VERSION)
    buf.append(0x01)
    buf.extend(struct.pack("<H", packet.proj_dim))
    buf.extend(struct.pack("<I", packet.token_count))
    buf.extend(struct.pack("<f", packet.alpha))
    buf.extend(struct.pack("<H", len(packet.blocks)))

    for bi, block in enumerate(packet.blocks):
        buf.append(int(block.mode))
        buf.extend(struct.pack("<H", block.length))

        has_seeds = (bi == 0) or (block.mode == Mode.RESET)
        buf.append(1 if has_seeds else 0)
        if has_seeds:
            buf.extend(block.seed1.astype(np.float32).tobytes())
            buf.extend(block.seed2.astype(np.float32).tobytes())

        if block.mode == Mode.ORBITAL:
            # 128 × 4 floats = 512 params
            for pc in block.pair_coeffs:
                buf.extend(struct.pack("<ffff", pc.kR, pc.kI, pc.gR, pc.gI))
        elif block.mode in (Mode.LINEAR, Mode.STATIC):
            pass  # implicit coefficients

        # per-pair scales
        buf.extend(block.per_pair_scale.astype(np.float32).tobytes())

        # quantized residuals
        q, _ = quantize_per_pair(block.residuals, q_bits)
        packed = _pack_varints(q.ravel().astype(np.int32))
        buf.extend(struct.pack("<I", len(packed)))
        buf.extend(packed)

    return bytes(buf)


def decode_wire(data: bytes) -> Packet:
    off = 0
    assert data[off:off+2] == MAGIC; off += 2
    version = data[off]; off += 1
    flags = data[off]; off += 1
    proj_dim, = struct.unpack_from("<H", data, off); off += 2
    token_count, = struct.unpack_from("<I", data, off); off += 4
    alpha, = struct.unpack_from("<f", data, off); off += 4
    n_blocks, = struct.unpack_from("<H", data, off); off += 2

    blocks = []
    prev_s1 = np.zeros(proj_dim, dtype=np.float32)
    prev_s2 = np.zeros(proj_dim, dtype=np.float32)
    cur_start = 2

    for bi in range(n_blocks):
        mode = Mode(data[off]); off += 1
        length, = struct.unpack_from("<H", data, off); off += 2

        has_seeds = data[off]; off += 1
        if has_seeds:
            seed1 = np.frombuffer(data[off:off+proj_dim*4], dtype=np.float32).copy(); off += proj_dim*4
            seed2 = np.frombuffer(data[off:off+proj_dim*4], dtype=np.float32).copy(); off += proj_dim*4
        else:
            seed1 = prev_s1.copy()
            seed2 = prev_s2.copy()

        pair_coeffs = []
        if mode == Mode.ORBITAL:
            for pi in range(N_PAIRS):
                kR, kI, gR, gI = struct.unpack_from("<ffff", data, off); off += 16
                pair_coeffs.append(PairCoeffs(pi, kR, kI, gR, gI))
        elif mode == Mode.LINEAR:
            pair_coeffs = [PairCoeffs(i, 2.0, 0.0, 1.0, 0.0) for i in range(N_PAIRS)]
        elif mode == Mode.STATIC:
            pair_coeffs = [PairCoeffs(i, 1.0, 0.0, 0.0, 0.0) for i in range(N_PAIRS)]

        scales = np.frombuffer(data[off:off+N_PAIRS*4], dtype=np.float32).copy(); off += N_PAIRS*4

        resid_len, = struct.unpack_from("<I", data, off); off += 4
        resid_data = data[off:off+resid_len]; off += resid_len
        q_flat = _unpack_varints(resid_data, length * proj_dim)
        q = q_flat.reshape(length, proj_dim).astype(np.int8)
        residuals = dequantize_per_pair(q, scales)

        block = Block(
            mode=mode, start_idx=cur_start, length=length,
            pair_coeffs=pair_coeffs, residuals=residuals,
            seed1=seed1, seed2=seed2, per_pair_scale=scales,
        )
        blocks.append(block)

        # propagate seeds
        if length >= 2:
            pred = predict_block_v2(seed1, seed2, pair_coeffs, length, mode)
            recon = pred + residuals
            prev_s1, prev_s2 = recon[-1].copy(), recon[-2].copy()
        elif length == 1:
            pred = predict_block_v2(seed1, seed2, pair_coeffs, 1, mode)
            prev_s2 = seed1.copy()
            prev_s1 = (pred[0] + residuals[0]).copy()
        else:
            prev_s1, prev_s2 = seed1.copy(), seed2.copy()

        cur_start += length

    return Packet(proj_dim, token_count, alpha, blocks)


# ── diagnostics ───────────────────────────────────────────────────────

def summary(packet: Packet) -> dict:
    mode_counts = {m.name: 0 for m in Mode}
    total_resid_energy = 0.0
    total_signal_energy = 0.0
    total_count = 0
    n_orbital_params = 0

    for block in packet.blocks:
        mode_counts[block.mode.name] += 1
        total_resid_energy += float(np.sum(block.residuals ** 2))
        total_count += block.residuals.size
        if block.mode == Mode.ORBITAL:
            n_orbital_params += len(block.pair_coeffs) * 4

    resid_rms = float(np.sqrt(total_resid_energy / max(total_count, 1)))

    return {
        "token_count": packet.token_count,
        "n_blocks": len(packet.blocks),
        "modes": mode_counts,
        "residual_rms": round(resid_rms, 6),
        "n_oscillators": N_PAIRS,
        "orbital_params_total": n_orbital_params,
    }


def byte_entropy(data: bytes) -> float:
    if len(data) == 0: return 0.0
    counts = np.zeros(256, dtype=np.int64)
    for b in data: counts[b] += 1
    probs = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


# ── self-test ─────────────────────────────────────────────────────────

def _self_test():
    print("=== Whisper Engram v2: Per-Pair Oscillators ===\n")
    print(f"oscillators: {N_PAIRS} pairs x 4 params = {N_PAIRS * 4} params/block\n")

    codec = EngramCodecV2(alpha=DEFAULT_ALPHA)
    rng = np.random.RandomState(42)

    # test 1: constant
    print("--- constant ---")
    w = np.ones((50, PROJ_DIM), dtype=np.float32) * 0.3
    pkt = codec.encode_trajectory(w)
    dec = codec.decode_trajectory(pkt)
    err = float(np.sqrt(np.mean((w - dec) ** 2)))
    print(f"  modes: {summary(pkt)['modes']}, err: {err:.8f}")
    assert err < 1e-5

    # test 2: multi-frequency (each pair has different dynamics)
    print("--- multi-frequency (per-pair) ---")
    w2 = np.zeros((80, PROJ_DIM), dtype=np.float32)
    t = np.arange(80, dtype=np.float32)
    for pi in range(N_PAIRS):
        theta1 = 0.1 + 0.02 * pi
        theta2 = 0.4 + 0.01 * pi
        w2[:, pi*2] = 0.3 * np.cos(theta1 * t) + 0.15 * np.cos(theta2 * t)
        w2[:, pi*2+1] = 0.3 * np.sin(theta1 * t) + 0.15 * np.sin(theta2 * t)
    w2 += 0.01 * rng.randn(80, PROJ_DIM).astype(np.float32)
    pkt2 = codec.encode_trajectory(w2)
    dec2 = codec.decode_trajectory(pkt2)
    err2 = float(np.sqrt(np.mean((w2 - dec2) ** 2)))
    s2 = summary(pkt2)
    print(f"  modes: {s2['modes']}, residual RMS: {s2['residual_rms']:.6f}, roundtrip err: {err2:.8f}")

    # energy capture
    total_resid = sum(float(np.sum(b.residuals ** 2)) for b in pkt2.blocks)
    total_signal = float(np.sum(w2[2:] ** 2))
    capture = (1.0 - total_resid / total_signal) * 100
    print(f"  oscillator energy capture: {capture:.1f}%")

    # wire format
    wire2 = encode_wire(pkt2, codec.q_bits)
    raw_size = w2.size * 4
    print(f"  wire: {len(wire2):,}B (raw {raw_size:,}B) = {raw_size/len(wire2):.1f}x")

    # wire roundtrip
    pkt2_rt = decode_wire(wire2)
    dec2_rt = codec.decode_trajectory(pkt2_rt)
    wire_err = float(np.sqrt(np.mean((w2 - dec2_rt) ** 2)))
    print(f"  wire roundtrip err: {wire_err:.6f}")

    # residual entropy
    resid_bytes = bytearray()
    for block in pkt2.blocks:
        q, _ = quantize_per_pair(block.residuals, codec.q_bits)
        resid_bytes.extend(_pack_varints(q.ravel().astype(np.int32)))
    ent = byte_entropy(bytes(resid_bytes))
    ideal = max(1, int(math.ceil(len(resid_bytes) * ent / 8)))
    print(f"  residual entropy: {ent:.2f} bits/byte, Logos ideal: {ideal:,}B ({len(resid_bytes)/ideal:.1f}x headroom)")
    print()

    # test 3: random (should not compress, but also not break)
    print("--- random ---")
    w3 = rng.randn(60, PROJ_DIM).astype(np.float32) * 0.3
    pkt3 = codec.encode_trajectory(w3)
    dec3 = codec.decode_trajectory(pkt3)
    err3 = float(np.sqrt(np.mean((w3 - dec3) ** 2)))
    print(f"  modes: {summary(pkt3)['modes']}, roundtrip err: {err3:.8f}")

    print()
    print("=== v2 self-test complete ===")


if __name__ == "__main__":
    _self_test()
