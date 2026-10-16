"""
engram.py

Whisper Engram — The Semantic Trajectory Codec by Woflo / MB

stores memory not as a file, but as initial conditions for a dynamical system.
recall is not loading — it is simulation.

the predictor is the same damped harmonic oscillator as Glyph:

    z[n] = K · z[n-1] − G · z[n-2]

but z is not a pen stroke in the complex plane. z is the state of a thought
in 256D Kizuna-projected witness space. the 256 dimensions are not independent.
they are structurally bound by the Kizuna mask geometry into 5 family subspaces:

    lower   (8D)  — intra-token structure, lower octonion
    upper   (8D)  — positional structure, upper octonion
    cross   (8D)  — coupling between lower and upper
    global  (1D)  — full-parity topic cohesion
    bond    (28D) — inter-token correlation dynamics (Tier-2 masks)

each family gets its own complex K, G — 20 real parameters per block.
this is the skeletal model: not every atom, not every finger, but every joint.

the witness projection uses tanh(α · S · h_t) instead of sign(S · h_t).
α is the cognitive temperature:
    high α → crystalline, binary, absolute (+1 or -1)
    low α  → fluid, continuous, capable of subtle gradients and momentum

── codec tower ─────────────────────────────────────────────────────────

    Engram (256D oscillator) → residuals → quantize → byte stream
                                                          ↓
                                                      (Logos-ready)

── block modes ─────────────────────────────────────────────────────────

    ORBITAL : complex AR(2) per family. wins for topic arcs, dialogue flow.
    LINEAR  : constant velocity extrapolation. wins for monotone continuation.
    STATIC  : topic hold. z[n] = z[n-1]. near-free when nothing changes.
    RESET   : phase transition. stores raw seed. the wormhole.

── the residuals ───────────────────────────────────────────────────────

    r[n] = w_actual[n] − w_predicted[n]

    zero residual    = the thought followed the orbit. logic. cliché. flow.
    large residual   = departure from orbit. metaphor. joke. insight.
    structured resid = a new orbit. the next block's K,G will capture it.

    the residuals are the spark of human intelligence, measured in bits.

── the duality ─────────────────────────────────────────────────────────

    Glyph fits the eigenvalues of the human hand.
    Engram fits the eigenvalues of a train of thought.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ── constants ─────────────────────────────────────────────────────────

ENGRAM_PROJ_DIM = 256
ENGRAM_DEFAULT_BLOCK_SIZE = 32
ENGRAM_MIN_BLOCK_SIZE = 4
ENGRAM_MAX_BLOCK_SIZE = 64
ENGRAM_DEFAULT_ALPHA = 4.0
ENGRAM_MAGIC = b"EG"
ENGRAM_VERSION = 1

# Q14 fixed-point scale (matches Glyph's coefficient encoding)
Q14 = 1 << 14  # 16384


# ── Kizuna family definitions ────────────────────────────────────────
#
# the 256 witness dimensions come from 256 masks selected by
# _select_mask_basis in hyper_cache.py. the first 53 are the
# KIZUNA_BOND_MASKS (25 Tier-1 + 28 Tier-2). the remaining 203
# are deterministic or token-spectrum-derived.
#
# we partition witness dims into families by their mask's structural
# affinity. dims 0..7 are the 8 lower singletons, 8..15 upper, etc.
# dims beyond 52 go into an "extended" bucket within the bond family.

FAMILY_NAMES = ("lower", "upper", "cross", "global", "bond")

# slices into the 256D witness vector for each family
# these match the KIZUNA_FAMILY_SLICES structure, extended to cover all 256D
FAMILY_RANGES: Tuple[Tuple[str, int, int], ...] = (
    ("lower",  0,   8),    # 8D
    ("upper",  8,  16),    # 8D
    ("cross", 16,  24),    # 8D
    ("global", 24, 25),    # 1D
    ("bond",  25, 53),     # 28D (Tier-2 masks)
    # dims 53..255 are extended masks — grouped with bond for prediction
)

# for the oscillator, we merge dims 25..255 into one "bond+extended" family
# this gives us 5 oscillators: lower(8D), upper(8D), cross(8D), global(1D), bond(203D)
OSCILLATOR_FAMILIES: Tuple[Tuple[str, int, int], ...] = (
    ("lower",  0,   8),
    ("upper",  8,  16),
    ("cross", 16,  24),
    ("global", 24,  25),
    ("bond",  25, 256),
)

N_FAMILIES = len(OSCILLATOR_FAMILIES)


# ── enums ─────────────────────────────────────────────────────────────

class EngramMode(IntEnum):
    ORBITAL = 0  # complex AR(2): z[n] = K·z[n-1] − G·z[n-2]
    LINEAR  = 1  # constant velocity: z[n] = 2·z[n-1] − z[n-2]
    STATIC  = 2  # topic hold: z[n] = z[n-1]
    RESET   = 3  # phase transition: raw seed stored


# ── data structures ───────────────────────────────────────────────────

@dataclass(slots=True)
class FamilyCoeffs:
    """Complex AR(2) coefficients for one Kizuna family subspace."""
    family_idx: int
    dim_start: int
    dim_end: int
    kR: float = 0.0
    kI: float = 0.0
    gR: float = 0.0
    gI: float = 0.0


@dataclass(slots=True)
class EngramBlock:
    """One block of the Engram-coded trajectory."""
    mode: EngramMode
    start_idx: int
    length: int
    family_coeffs: List[FamilyCoeffs]
    residuals: np.ndarray     # [length, 256] float32
    seed1: np.ndarray         # [256] float32, z[n-1] at block start
    seed2: np.ndarray         # [256] float32, z[n-2] at block start
    fit_error: float = 0.0    # RMS prediction error for diagnostics


@dataclass(slots=True)
class EngramPacket:
    """Complete encoded trajectory."""
    proj_dim: int
    token_count: int
    alpha: float
    blocks: List[EngramBlock]


# ── soft witness projection ───────────────────────────────────────────

def soft_witness(
    hidden_path: np.ndarray,
    scramble_matrix: np.ndarray,
    alpha: float = ENGRAM_DEFAULT_ALPHA,
) -> np.ndarray:
    """
    Soft witness: tanh(α · S · h_t).

    Parameters
    ----------
    hidden_path : [T, D] float32, raw hidden states
    scramble_matrix : [proj_dim, proj_input_dim] float32, ±1 matrix
    alpha : temperature. high = hard bipolar. low = smooth gradients.

    Returns
    -------
    [T, proj_dim] float32 in (-1, 1)
    """
    proj_input_dim = scramble_matrix.shape[1]
    h = hidden_path[:, :proj_input_dim].astype(np.float32)
    projected = h @ scramble_matrix.T  # [T, proj_dim]
    return np.tanh(alpha * projected).astype(np.float32)


def harden_witness(soft_w: np.ndarray) -> np.ndarray:
    """Convert soft witness to hard bipolar: sign(soft_w), with 0 → +1."""
    return np.where(soft_w >= 0, 1.0, -1.0).astype(np.float32)


# ── zigzag coding ─────────────────────────────────────────────────────

def zigzag_encode(v: int) -> int:
    """Map signed int to unsigned: 0→0, -1→1, 1→2, -2→3, ..."""
    return (v << 1) ^ (v >> 31)


def zigzag_decode(v: int) -> int:
    """Inverse of zigzag_encode."""
    return (v >> 1) ^ -(v & 1)


def zigzag_encode_array(arr: np.ndarray) -> np.ndarray:
    """Vectorized zigzag encode on int32 array."""
    a = arr.astype(np.int32)
    return ((a << 1) ^ (a >> 31)).astype(np.uint32)


def zigzag_decode_array(arr: np.ndarray) -> np.ndarray:
    """Vectorized zigzag decode on uint32 array."""
    a = arr.astype(np.uint32)
    return ((a >> 1).astype(np.int32)) ^ (-(a & 1).astype(np.int32))


# ── LEB128 varint coding ─────────────────────────────────────────────

def _encode_varint(v: int) -> bytes:
    """Encode unsigned int as LEB128."""
    v = v & 0xFFFFFFFF
    out = bytearray()
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def _decode_varint(data: bytes, offset: int) -> Tuple[int, int]:
    """Decode LEB128 varint. Returns (value, new_offset)."""
    result = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result & 0xFFFFFFFF, offset
        shift += 7
        if shift >= 35:
            break
    return result & 0xFFFFFFFF, offset


# ── quantization ──────────────────────────────────────────────────────

def quantize_residuals(residuals: np.ndarray, q_bits: int = 8) -> np.ndarray:
    """
    Quantize float32 residuals to int8.

    For residuals near {-2, -1, 0, 1, 2} (hard witness case),
    this is nearly lossless. For continuous residuals, we scale
    by (2^(q_bits-1) - 1) / max_abs and round.
    """
    max_abs = float(np.abs(residuals).max())
    if max_abs < 1e-8:
        return np.zeros_like(residuals, dtype=np.int8)
    scale = (2 ** (q_bits - 1) - 1) / max_abs
    return np.clip(np.round(residuals * scale), -127, 127).astype(np.int8)


def dequantize_residuals(
    quantized: np.ndarray,
    max_abs: float,
    q_bits: int = 8,
) -> np.ndarray:
    """Inverse of quantize_residuals."""
    if max_abs < 1e-8:
        return np.zeros(quantized.shape, dtype=np.float32)
    scale = max_abs / (2 ** (q_bits - 1) - 1)
    return quantized.astype(np.float32) * scale


# ── residual packing ──────────────────────────────────────────────────

def pack_residuals(quantized: np.ndarray) -> bytes:
    """Zigzag + LEB128 encode a flat int array."""
    flat = quantized.ravel().astype(np.int32)
    zz = zigzag_encode_array(flat)
    out = bytearray()
    for v in zz:
        out.extend(_encode_varint(int(v)))
    return bytes(out)


def unpack_residuals(data: bytes, count: int) -> np.ndarray:
    """LEB128 + zigzag decode to flat int32 array."""
    out = np.empty(count, dtype=np.int32)
    offset = 0
    for i in range(count):
        v, offset = _decode_varint(data, offset)
        out[i] = zigzag_decode(int(v))
    return out


# ── complex AR(2) fitting ─────────────────────────────────────────────
#
# generalized from Glyph's 2×2 Hermitian Cramer solver.
# for a D-dimensional family subspace, we pair dims into D//2 complex
# planes and fit one shared K, G across all planes simultaneously.
#
# the normal equation:
#   [sAA    sAB*] [K ]   [sTA*]
#   [sAB    sBB ] [-G] = [sTB*]
#
# where A = z[n-1], B = z[n-2], T = z[n], summed over all n and all
# complex dim pairs within the family.

def fit_complex_ar2(z: np.ndarray) -> Tuple[complex, complex, float]:
    """
    Fit z[n] = K · z[n-1] − G · z[n-2] via complex Cramer.

    Parameters
    ----------
    z : [T, D] float32 — family subspace trajectory (T >= 3)

    Returns
    -------
    (K, G, rms_error)
    """
    T, D = z.shape
    if T < 3:
        return complex(1.0, 0.0), complex(0.0, 0.0), 0.0

    # pair dimensions into complex planes
    n_pairs = D // 2
    has_odd = D % 2 == 1

    # complex-paired portion
    if n_pairs > 0:
        z_paired = z[:, :n_pairs * 2].reshape(T, n_pairs, 2)
        z_c = z_paired[..., 0] + 1j * z_paired[..., 1]  # [T, n_pairs]

        target = z_c[2:]      # z[n]
        A = z_c[1:-1]         # z[n-1]
        B = z_c[:-2]          # z[n-2]

        sAA = float(np.real(np.sum(A * np.conj(A))))
        sBB = float(np.real(np.sum(B * np.conj(B))))
        sAB = complex(np.sum(A * np.conj(B)))
        sTA = complex(np.sum(target * np.conj(A)))
        sTB = complex(np.sum(target * np.conj(B)))
    else:
        sAA = sBB = 0.0
        sAB = sTA = sTB = 0j

    # odd dimension: real-only contribution
    if has_odd:
        z_odd = z[:, -1].astype(np.float64)
        t_odd = z_odd[2:]
        a_odd = z_odd[1:-1]
        b_odd = z_odd[:-2]

        sAA += float(np.sum(a_odd * a_odd))
        sBB += float(np.sum(b_odd * b_odd))
        sAB += complex(float(np.sum(a_odd * b_odd)), 0.0)
        sTA += complex(float(np.sum(t_odd * a_odd)), 0.0)
        sTB += complex(float(np.sum(t_odd * b_odd)), 0.0)

    # Cramer solve with Tikhonov ridge regularization
    #
    # normal equations for z[n] = K·A − G·B:
    #   K·sAA − G·conj(sAB) = sTA
    #   −K·sAB + G·sBB      = −sTB
    #
    # Regularized: add λ·I to the Hermitian matrix (prevents det → 0
    # for single-frequency signals where sAA·sBB = |sAB|²).
    #
    ridge = 1e-4 * (sAA + sBB + 1.0)
    sAA_r = sAA + ridge
    sBB_r = sBB + ridge

    det = sAA_r * sBB_r - abs(sAB) ** 2
    if abs(det) < 1e-12:
        # degenerate: fall back to linear (K=2, G=1)
        return complex(2.0, 0.0), complex(1.0, 0.0), _ar2_rms(z, complex(2.0, 0.0), complex(1.0, 0.0))

    K = (sTA * sBB_r - np.conj(sAB) * sTB) / det
    G = (sAB * sTA - sAA_r * sTB) / det

    rms = _ar2_rms(z, K, G)
    return K, G, rms


def _ar2_rms(z: np.ndarray, K: complex, G: complex) -> float:
    """Compute RMS prediction error for AR(2) model."""
    T, D = z.shape
    if T < 3:
        return 0.0

    n_pairs = D // 2
    total_err = 0.0
    count = 0

    if n_pairs > 0:
        z_c = z[:, :n_pairs * 2].reshape(T, n_pairs, 2)
        z_c = z_c[..., 0] + 1j * z_c[..., 1]
        pred = K * z_c[1:-1] - G * z_c[:-2]
        err = z_c[2:] - pred
        total_err += float(np.sum(np.abs(err) ** 2))
        count += err.size

    if D % 2 == 1:
        z_odd = z[:, -1].astype(np.float64)
        pred_odd = K.real * z_odd[1:-1] - G.real * z_odd[:-2]
        err_odd = z_odd[2:] - pred_odd
        total_err += float(np.sum(err_odd ** 2))
        count += err_odd.size

    if count == 0:
        return 0.0
    return float(np.sqrt(total_err / count))


def _linear_rms(z: np.ndarray) -> float:
    """RMS error for linear predictor z[n] = 2·z[n-1] − z[n-2]."""
    return _ar2_rms(z, complex(2.0, 0.0), complex(1.0, 0.0))


def _static_rms(z: np.ndarray) -> float:
    """RMS error for static predictor z[n] = z[n-1]."""
    return _ar2_rms(z, complex(1.0, 0.0), complex(0.0, 0.0))


# ── prediction ────────────────────────────────────────────────────────

def predict_family(
    seed1: np.ndarray,   # [D] z[n-1]
    seed2: np.ndarray,   # [D] z[n-2]
    K: complex,
    G: complex,
    length: int,
) -> np.ndarray:
    """
    Generate predicted trajectory [length, D] from seeds + coefficients.
    Uses the complex AR(2) applied dimension-pair-wise.
    """
    D = seed1.shape[0]
    out = np.empty((length, D), dtype=np.float32)

    n_pairs = D // 2
    has_odd = D % 2 == 1

    # work in complex space
    if n_pairs > 0:
        s1_c = seed1[:n_pairs * 2].reshape(n_pairs, 2)
        s1_c = s1_c[:, 0] + 1j * s1_c[:, 1]
        s2_c = seed2[:n_pairs * 2].reshape(n_pairs, 2)
        s2_c = s2_c[:, 0] + 1j * s2_c[:, 1]

        prev1 = s1_c.copy()
        prev2 = s2_c.copy()

        for n in range(length):
            pred_c = K * prev1 - G * prev2
            out[n, :n_pairs * 2:2] = pred_c.real.astype(np.float32)
            out[n, 1:n_pairs * 2:2] = pred_c.imag.astype(np.float32)
            prev2 = prev1
            prev1 = pred_c

    # odd dim: real AR(2)
    if has_odd:
        prev1_r = float(seed1[-1])
        prev2_r = float(seed2[-1])
        for n in range(length):
            pred_r = K.real * prev1_r - G.real * prev2_r
            out[n, -1] = pred_r
            prev2_r = prev1_r
            prev1_r = pred_r

    return out


def predict_block(
    seed1: np.ndarray,               # [256]
    seed2: np.ndarray,               # [256]
    family_coeffs: List[FamilyCoeffs],
    length: int,
    mode: EngramMode,
) -> np.ndarray:
    """Generate full 256D predicted trajectory for a block."""
    predicted = np.empty((length, ENGRAM_PROJ_DIM), dtype=np.float32)

    if mode == EngramMode.STATIC:
        for n in range(length):
            predicted[n] = seed1
        return predicted

    for fc in family_coeffs:
        s1 = seed1[fc.dim_start:fc.dim_end]
        s2 = seed2[fc.dim_start:fc.dim_end]
        K = complex(fc.kR, fc.kI)
        G = complex(fc.gR, fc.gI)
        predicted[:, fc.dim_start:fc.dim_end] = predict_family(s1, s2, K, G, length)

    return predicted


# ── block segmentation ────────────────────────────────────────────────

def compute_transition_energy(
    witness_path: np.ndarray,
    window: int = 8,
) -> np.ndarray:
    """
    Compute per-token transition energy for phase-transition detection.
    Uses L2 norm of witness velocity (first difference) with smoothing.
    """
    T = witness_path.shape[0]
    if T < 2:
        return np.zeros(T, dtype=np.float32)

    velocity = np.diff(witness_path, axis=0)  # [T-1, 256]
    energy = np.linalg.norm(velocity, axis=1).astype(np.float32)  # [T-1]

    # pad to match T
    energy = np.concatenate([np.array([0.0], dtype=np.float32), energy])

    return energy


def segment_blocks(
    witness_path: np.ndarray,
    min_block: int = ENGRAM_MIN_BLOCK_SIZE,
    max_block: int = ENGRAM_MAX_BLOCK_SIZE,
    energy_threshold: float = 2.0,
) -> List[Tuple[int, int]]:
    """
    Adaptive block segmentation based on transition energy spikes.

    Returns list of (start, end) index pairs. Each block needs 2 seed
    tokens, so the first block starts at index 2 (indices 0,1 are seeds).
    """
    T = witness_path.shape[0]
    if T <= 2:
        return []

    energy = compute_transition_energy(witness_path)

    # running statistics for threshold
    segments: List[Tuple[int, int]] = []
    block_start = 2  # first 2 tokens are seeds

    # compute global energy stats for threshold
    if T > 4:
        e_slice = energy[2:]
        e_mean = float(np.mean(e_slice))
        e_std = float(np.std(e_slice)) + 1e-8
    else:
        e_mean = 0.0
        e_std = 1.0

    threshold = e_mean + energy_threshold * e_std

    for t in range(3, T):
        block_len = t - block_start

        # force split at max block size
        if block_len >= max_block:
            segments.append((block_start, t))
            block_start = t
            continue

        # split at energy spike (if block is large enough)
        if block_len >= min_block and energy[t] > threshold:
            segments.append((block_start, t))
            block_start = t

    # final block
    if block_start < T:
        segments.append((block_start, T))

    # merge tiny trailing blocks
    if len(segments) > 1 and (segments[-1][1] - segments[-1][0]) < min_block:
        last = segments.pop()
        prev = segments.pop()
        segments.append((prev[0], last[1]))

    return segments


# ── mode selection ────────────────────────────────────────────────────

def select_block_mode(
    block_data: np.ndarray,   # [L, 256]
    seed1: np.ndarray,        # [256]
    seed2: np.ndarray,        # [256]
) -> Tuple[EngramMode, List[FamilyCoeffs], float]:
    """
    Choose optimal mode and fit coefficients for a block.

    Returns (mode, family_coeffs, rms_error).
    """
    L = block_data.shape[0]

    if L < 1:
        return EngramMode.RESET, [], 0.0

    # check STATIC: variance across time
    temporal_var = float(np.var(block_data, axis=0).mean())
    if temporal_var < 1e-6:
        coeffs = [
            FamilyCoeffs(i, s, e, kR=1.0, kI=0.0, gR=0.0, gI=0.0)
            for i, (_, s, e) in enumerate(OSCILLATOR_FAMILIES)
        ]
        return EngramMode.STATIC, coeffs, 0.0

    # need seed context for fitting: prepend seeds
    z_full = np.vstack([seed2[np.newaxis], seed1[np.newaxis], block_data])

    # fit ORBITAL per family
    orbital_coeffs: List[FamilyCoeffs] = []
    orbital_total_err = 0.0
    linear_total_err = 0.0
    n_dims = 0

    for fi, (name, dim_start, dim_end) in enumerate(OSCILLATOR_FAMILIES):
        D = dim_end - dim_start
        z_family = z_full[:, dim_start:dim_end]

        K, G, orb_err = fit_complex_ar2(z_family)
        lin_err = _linear_rms(z_family)

        orbital_coeffs.append(FamilyCoeffs(
            family_idx=fi,
            dim_start=dim_start,
            dim_end=dim_end,
            kR=K.real,
            kI=K.imag,
            gR=G.real,
            gI=G.imag,
        ))

        orbital_total_err += orb_err * orb_err * D
        linear_total_err += lin_err * lin_err * D
        n_dims += D

    orbital_rms = float(np.sqrt(orbital_total_err / max(n_dims, 1)))
    linear_rms = float(np.sqrt(linear_total_err / max(n_dims, 1)))

    # LINEAR mode: K=2, G=1 for all families
    linear_coeffs = [
        FamilyCoeffs(i, s, e, kR=2.0, kI=0.0, gR=1.0, gI=0.0)
        for i, (_, s, e) in enumerate(OSCILLATOR_FAMILIES)
    ]

    # mode decision: orbital wins if it beats linear at all.
    # the 20-param overhead is cheap enough that even marginal wins pay off.
    if orbital_rms < linear_rms:
        return EngramMode.ORBITAL, orbital_coeffs, orbital_rms
    else:
        return EngramMode.LINEAR, linear_coeffs, linear_rms


# ── the codec ─────────────────────────────────────────────────────────

class EngramCodec:
    """
    Whisper Engram: the semantic trajectory codec.

    Compresses 256D witness trajectories into orbital parameters + residuals.
    Stores memory as initial conditions for a dynamical system.
    Recall is simulation, not loading.
    """

    def __init__(
        self,
        alpha: float = ENGRAM_DEFAULT_ALPHA,
        min_block: int = ENGRAM_MIN_BLOCK_SIZE,
        max_block: int = ENGRAM_MAX_BLOCK_SIZE,
        q_bits: int = 8,
    ):
        self.alpha = alpha
        self.min_block = min_block
        self.max_block = max_block
        self.q_bits = q_bits

    def encode_trajectory(
        self,
        witness_path: np.ndarray,
    ) -> EngramPacket:
        """
        Full encode pipeline.

        Parameters
        ----------
        witness_path : [T, 256] float32 — soft or hard witness path

        Returns
        -------
        EngramPacket with compressed blocks
        """
        T = witness_path.shape[0]
        w = witness_path.astype(np.float32)

        if T == 0:
            return EngramPacket(
                proj_dim=ENGRAM_PROJ_DIM,
                token_count=0,
                alpha=self.alpha,
                blocks=[],
            )

        if T <= 2:
            # too short for prediction — store as RESET
            seed2 = w[0] if T >= 1 else np.zeros(ENGRAM_PROJ_DIM, dtype=np.float32)
            seed1 = w[1] if T >= 2 else w[0]
            block = EngramBlock(
                mode=EngramMode.RESET,
                start_idx=0,
                length=T,
                family_coeffs=[],
                residuals=w.copy(),
                seed1=seed1.copy(),
                seed2=seed2.copy(),
            )
            return EngramPacket(
                proj_dim=ENGRAM_PROJ_DIM,
                token_count=T,
                alpha=self.alpha,
                blocks=[block],
            )

        # segment into blocks
        segments = segment_blocks(
            w,
            min_block=self.min_block,
            max_block=self.max_block,
        )

        blocks: List[EngramBlock] = []
        for seg_start, seg_end in segments:
            block_data = w[seg_start:seg_end]
            L = block_data.shape[0]

            # seeds: the two tokens before this block
            seed2 = w[seg_start - 2].copy()
            seed1 = w[seg_start - 1].copy()

            # select mode and fit
            mode, coeffs, fit_err = select_block_mode(block_data, seed1, seed2)

            # compute prediction
            predicted = predict_block(seed1, seed2, coeffs, L, mode)

            # residuals
            residuals = block_data - predicted

            blocks.append(EngramBlock(
                mode=mode,
                start_idx=seg_start,
                length=L,
                family_coeffs=coeffs,
                residuals=residuals,
                seed1=seed1,
                seed2=seed2,
                fit_error=fit_err,
            ))

        return EngramPacket(
            proj_dim=ENGRAM_PROJ_DIM,
            token_count=T,
            alpha=self.alpha,
            blocks=blocks,
        )

    def decode_trajectory(
        self,
        packet: EngramPacket,
    ) -> np.ndarray:
        """
        Full decode pipeline. Simulates the dynamical system.

        Returns [T, 256] float32 — reconstructed witness path.
        """
        T = packet.token_count
        if T == 0:
            return np.empty((0, ENGRAM_PROJ_DIM), dtype=np.float32)

        out = np.zeros((T, ENGRAM_PROJ_DIM), dtype=np.float32)

        for block in packet.blocks:
            if block.mode == EngramMode.RESET and block.start_idx == 0:
                # raw storage for very short trajectories
                L = min(block.length, block.residuals.shape[0])
                out[:L] = block.residuals[:L]
                continue

            # reconstruct: predicted + residual
            predicted = predict_block(
                block.seed1,
                block.seed2,
                block.family_coeffs,
                block.length,
                block.mode,
            )
            out[block.start_idx:block.start_idx + block.length] = (
                predicted + block.residuals
            )

        # fill in seeds (tokens 0 and 1)
        if len(packet.blocks) > 0 and packet.blocks[0].start_idx >= 2:
            out[0] = packet.blocks[0].seed2
            out[1] = packet.blocks[0].seed1

        return out

    # ── wire format ────────────────────────────────────────────────────

    def encode_bytes(self, witness_path: np.ndarray) -> bytes:
        """Encode trajectory to wire format."""
        packet = self.encode_trajectory(witness_path)
        return encode_engram(packet, self.q_bits)

    def decode_bytes(self, data: bytes) -> np.ndarray:
        """Decode trajectory from wire format."""
        packet = decode_engram(data)
        return self.decode_trajectory(packet)


# ── wire format serialization ─────────────────────────────────────────

def encode_engram(packet: EngramPacket, q_bits: int = 8) -> bytes:
    """
    Serialize EngramPacket to wire bytes.

    Header: EG [version:1] [flags:1] [proj_dim:2 LE] [token_count:4 LE]
            [alpha:4 float32 LE] [n_blocks:2 LE]

    Per block:
        [mode:1] [length:2 LE] [max_abs:4 float32 LE]
        [has_seeds:1]
        if has_seeds:
            [seed1: proj_dim * 4 bytes] [seed2: proj_dim * 4 bytes]
        if mode == ORBITAL:
            [n_families:1]
            per family: [idx:1][kR:4f][kI:4f][gR:4f][gI:4f]
        [residual_len:4 LE] [residual_data]
    """
    buf = bytearray()

    # header
    buf.extend(ENGRAM_MAGIC)
    buf.append(ENGRAM_VERSION)
    buf.append(0x01)  # flags: soft witness
    buf.extend(struct.pack("<H", packet.proj_dim))
    buf.extend(struct.pack("<I", packet.token_count))
    buf.extend(struct.pack("<f", packet.alpha))
    buf.extend(struct.pack("<H", len(packet.blocks)))

    for bi, block in enumerate(packet.blocks):
        buf.append(int(block.mode))
        buf.extend(struct.pack("<H", block.length))

        # max_abs for dequantization
        max_abs = float(np.abs(block.residuals).max()) if block.residuals.size > 0 else 0.0
        buf.extend(struct.pack("<f", max_abs))

        # seeds: only first block or RESET blocks
        has_seeds = (bi == 0) or (block.mode == EngramMode.RESET)
        buf.append(1 if has_seeds else 0)
        if has_seeds:
            buf.extend(block.seed1.astype(np.float32).tobytes())
            buf.extend(block.seed2.astype(np.float32).tobytes())

        # coefficients
        if block.mode == EngramMode.ORBITAL:
            buf.append(len(block.family_coeffs))
            for fc in block.family_coeffs:
                buf.append(fc.family_idx)
                buf.extend(struct.pack("<ffff", fc.kR, fc.kI, fc.gR, fc.gI))
        elif block.mode in (EngramMode.LINEAR, EngramMode.STATIC):
            buf.append(0)  # no explicit coeffs needed

        # residuals
        if block.mode == EngramMode.RESET and block.start_idx == 0:
            # raw data for tiny trajectories
            raw_bytes = block.residuals.astype(np.float32).tobytes()
            buf.extend(struct.pack("<I", len(raw_bytes)))
            buf.extend(raw_bytes)
        else:
            q = quantize_residuals(block.residuals, q_bits)
            packed = pack_residuals(q)
            buf.extend(struct.pack("<I", len(packed)))
            buf.extend(packed)

    return bytes(buf)


def decode_engram(data: bytes) -> EngramPacket:
    """Deserialize wire bytes to EngramPacket."""
    off = 0

    # header
    magic = data[off:off + 2]
    off += 2
    if magic != ENGRAM_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")

    version = data[off]; off += 1
    if version != ENGRAM_VERSION:
        raise ValueError(f"unsupported version: {version}")

    flags = data[off]; off += 1
    proj_dim, = struct.unpack_from("<H", data, off); off += 2
    token_count, = struct.unpack_from("<I", data, off); off += 4
    alpha, = struct.unpack_from("<f", data, off); off += 4
    n_blocks, = struct.unpack_from("<H", data, off); off += 2

    blocks: List[EngramBlock] = []
    prev_seed1 = np.zeros(proj_dim, dtype=np.float32)
    prev_seed2 = np.zeros(proj_dim, dtype=np.float32)
    cur_start = 2

    for bi in range(n_blocks):
        mode = EngramMode(data[off]); off += 1
        length, = struct.unpack_from("<H", data, off); off += 2
        max_abs, = struct.unpack_from("<f", data, off); off += 4

        has_seeds = data[off]; off += 1
        if has_seeds:
            seed1 = np.frombuffer(data[off:off + proj_dim * 4], dtype=np.float32).copy()
            off += proj_dim * 4
            seed2 = np.frombuffer(data[off:off + proj_dim * 4], dtype=np.float32).copy()
            off += proj_dim * 4
        else:
            seed1 = prev_seed1.copy()
            seed2 = prev_seed2.copy()

        # coefficients
        family_coeffs: List[FamilyCoeffs] = []
        if mode == EngramMode.ORBITAL:
            n_fam = data[off]; off += 1
            for _ in range(n_fam):
                fidx = data[off]; off += 1
                kR, kI, gR, gI = struct.unpack_from("<ffff", data, off); off += 16
                # look up dim range from OSCILLATOR_FAMILIES
                _, ds, de = OSCILLATOR_FAMILIES[fidx]
                family_coeffs.append(FamilyCoeffs(fidx, ds, de, kR, kI, gR, gI))
        elif mode in (EngramMode.LINEAR, EngramMode.STATIC):
            n_fam = data[off]; off += 1
            # reconstruct implicit coeffs
            if mode == EngramMode.LINEAR:
                family_coeffs = [
                    FamilyCoeffs(i, s, e, kR=2.0, kI=0.0, gR=1.0, gI=0.0)
                    for i, (_, s, e) in enumerate(OSCILLATOR_FAMILIES)
                ]
            else:
                family_coeffs = [
                    FamilyCoeffs(i, s, e, kR=1.0, kI=0.0, gR=0.0, gI=0.0)
                    for i, (_, s, e) in enumerate(OSCILLATOR_FAMILIES)
                ]

        # residuals
        resid_len, = struct.unpack_from("<I", data, off); off += 4
        resid_data = data[off:off + resid_len]; off += resid_len

        if mode == EngramMode.RESET and cur_start == 0:
            # raw float32 data
            residuals = np.frombuffer(resid_data, dtype=np.float32).reshape(-1, proj_dim).copy()
        else:
            count = length * proj_dim
            q_flat = unpack_residuals(resid_data, count)
            q = q_flat.reshape(length, proj_dim).astype(np.int8)
            residuals = dequantize_residuals(q, max_abs)

        block = EngramBlock(
            mode=mode,
            start_idx=cur_start,
            length=length,
            family_coeffs=family_coeffs,
            residuals=residuals,
            seed1=seed1,
            seed2=seed2,
        )
        blocks.append(block)

        # propagate seeds for next block
        if length >= 2:
            predicted = predict_block(seed1, seed2, family_coeffs, length, mode)
            reconstructed = predicted + residuals
            prev_seed1 = reconstructed[-1].copy()
            prev_seed2 = reconstructed[-2].copy()
        elif length == 1:
            predicted = predict_block(seed1, seed2, family_coeffs, 1, mode)
            prev_seed2 = seed1.copy()
            prev_seed1 = (predicted[0] + residuals[0]).copy()
        else:
            prev_seed1 = seed1.copy()
            prev_seed2 = seed2.copy()

        cur_start += length

    return EngramPacket(
        proj_dim=proj_dim,
        token_count=token_count,
        alpha=alpha,
        blocks=blocks,
    )


# ── handshake by parameters ──────────────────────────────────────────

def engram_parameter_vector(packet: EngramPacket) -> np.ndarray:
    """
    Extract flat parameter vector for fast trajectory matching.

    Each ORBITAL block contributes 5 × 4 = 20 floats (K,G per family).
    LINEAR/STATIC contribute a mode flag. RESET contributes seed hash.

    Matching K,G vectors asks: "are these thoughts moving the same way?"
    rather than "are these thoughts in the same place?"
    """
    params: List[float] = []
    for block in packet.blocks:
        if block.mode == EngramMode.ORBITAL:
            for fc in block.family_coeffs:
                params.extend([fc.kR, fc.kI, fc.gR, fc.gI])
        elif block.mode == EngramMode.LINEAR:
            params.extend([2.0, 0.0, 1.0, 0.0] * N_FAMILIES)
        elif block.mode == EngramMode.STATIC:
            params.extend([1.0, 0.0, 0.0, 0.0] * N_FAMILIES)
        else:  # RESET
            params.extend([0.0, 0.0, 0.0, 0.0] * N_FAMILIES)

    return np.asarray(params, dtype=np.float32)


def engram_parameter_distance(
    params_a: np.ndarray,
    params_b: np.ndarray,
) -> float:
    """
    Distance between two parameter vectors.

    When block counts differ, aligns using DTW-like padding.
    For equal-length vectors, this is L2 distance.
    """
    a = params_a.astype(np.float32)
    b = params_b.astype(np.float32)

    # pad shorter to match longer
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)), constant_values=0.0)
    elif len(b) < len(a):
        b = np.pad(b, (0, len(a) - len(b)), constant_values=0.0)

    return float(np.linalg.norm(a - b))


# ── diagnostics ───────────────────────────────────────────────────────

def engram_summary(packet: EngramPacket) -> dict:
    """Human-readable summary of an encoded trajectory."""
    mode_counts = {m.name: 0 for m in EngramMode}
    total_resid_energy = 0.0
    total_resid_count = 0

    for block in packet.blocks:
        mode_counts[block.mode.name] += 1
        total_resid_energy += float(np.sum(block.residuals ** 2))
        total_resid_count += block.residuals.size

    resid_rms = float(np.sqrt(total_resid_energy / max(total_resid_count, 1)))

    # raw witness size vs encoded size
    raw_bytes = packet.token_count * packet.proj_dim * 4  # float32
    n_orbital = mode_counts["ORBITAL"]
    n_other = sum(v for k, v in mode_counts.items() if k != "ORBITAL")
    param_bytes = n_orbital * N_FAMILIES * 16  # 4 floats × 4 bytes per family
    seed_bytes = 2 * packet.proj_dim * 4  # first block seeds

    return {
        "token_count": packet.token_count,
        "n_blocks": len(packet.blocks),
        "mode_distribution": mode_counts,
        "residual_rms": round(resid_rms, 6),
        "alpha": packet.alpha,
        "raw_witness_bytes": raw_bytes,
        "param_overhead_bytes": param_bytes + seed_bytes,
        "families": [name for name, _, _ in OSCILLATOR_FAMILIES],
    }


# ── self-test ─────────────────────────────────────────────────────────

def _self_test() -> None:
    """Quick round-trip smoke test."""
    print("=== Whisper Engram: Semantic Trajectory Codec ===\n")
    print("families: " + " · ".join(
        f"{name}({e - s}D)" for name, s, e in OSCILLATOR_FAMILIES
    ))
    print(f"total: {ENGRAM_PROJ_DIM}D witness space")
    print()

    codec = EngramCodec(alpha=4.0)
    rng = np.random.RandomState(42)

    # test 1: constant trajectory (should be STATIC)
    print("--- constant trajectory (topic hold) ---")
    w_const = np.ones((50, ENGRAM_PROJ_DIM), dtype=np.float32) * 0.7
    pkt = codec.encode_trajectory(w_const)
    dec = codec.decode_trajectory(pkt)
    err = float(np.sqrt(np.mean((w_const - dec) ** 2)))
    print(f"  {engram_summary(pkt)}")
    print(f"  roundtrip RMS error: {err:.6f}")
    assert err < 0.01, f"constant roundtrip failed: {err}"
    print()

    # test 2: smooth sinusoidal trajectory (should be ORBITAL)
    print("--- sinusoidal trajectory (orbital thought) ---")
    t = np.linspace(0, 4 * np.pi, 100)
    w_sin = np.zeros((100, ENGRAM_PROJ_DIM), dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        freq = 0.1 + 0.01 * d
        phase = 0.1 * d
        w_sin[:, d] = 0.5 * np.sin(freq * t + phase)
    pkt = codec.encode_trajectory(w_sin)
    dec = codec.decode_trajectory(pkt)
    err = float(np.sqrt(np.mean((w_sin - dec) ** 2)))
    summary = engram_summary(pkt)
    print(f"  modes: {summary['mode_distribution']}")
    print(f"  residual RMS: {summary['residual_rms']:.6f}")
    print(f"  roundtrip RMS error: {err:.6f}")
    print()

    # test 3: phase transition (structured → random → structured)
    print("--- phase transition (topic pivot) ---")
    w_phase = np.zeros((120, ENGRAM_PROJ_DIM), dtype=np.float32)
    # structured: smooth rotation
    for i in range(40):
        for d in range(ENGRAM_PROJ_DIM):
            w_phase[i, d] = 0.5 * np.sin(0.1 * i + 0.05 * d)
    # random: the creative leap
    w_phase[40:80] = rng.randn(40, ENGRAM_PROJ_DIM).astype(np.float32) * 0.3
    # structured again: new orbit
    for i in range(40):
        for d in range(ENGRAM_PROJ_DIM):
            w_phase[80 + i, d] = 0.5 * np.cos(0.15 * i + 0.03 * d)
    pkt = codec.encode_trajectory(w_phase)
    dec = codec.decode_trajectory(pkt)
    err = float(np.sqrt(np.mean((w_phase - dec) ** 2)))
    summary = engram_summary(pkt)
    print(f"  modes: {summary['mode_distribution']}")
    print(f"  blocks: {summary['n_blocks']}")
    print(f"  residual RMS: {summary['residual_rms']:.6f}")
    print(f"  roundtrip RMS error: {err:.6f}")
    print()

    # test 4: wire format round-trip
    print("--- wire format round-trip ---")
    wire = codec.encode_bytes(w_sin)
    dec_wire = codec.decode_bytes(wire)
    wire_err = float(np.sqrt(np.mean((w_sin - dec_wire) ** 2)))
    raw_size = w_sin.size * 4
    comp_ratio = raw_size / max(len(wire), 1)
    print(f"  raw witness: {raw_size:,} bytes")
    print(f"  wire format: {len(wire):,} bytes")
    print(f"  compression ratio: {comp_ratio:.1f}×")
    print(f"  wire roundtrip RMS error: {wire_err:.6f}")
    print()

    # test 5: coherent rotation (all dims same frequency — pure orbital)
    print("--- coherent rotation (uniform frequency) ---")
    w_rot = np.zeros((100, ENGRAM_PROJ_DIM), dtype=np.float32)
    t_rot = np.linspace(0, 4 * np.pi, 100)
    for d in range(0, ENGRAM_PROJ_DIM - 1, 2):
        w_rot[:, d] = 0.5 * np.cos(0.3 * t_rot)
        w_rot[:, d + 1] = 0.5 * np.sin(0.3 * t_rot)
    pkt_rot = codec.encode_trajectory(w_rot)
    dec_rot = codec.decode_trajectory(pkt_rot)
    err_rot = float(np.sqrt(np.mean((w_rot - dec_rot) ** 2)))
    summary_rot = engram_summary(pkt_rot)
    print(f"  modes: {summary_rot['mode_distribution']}")
    print(f"  residual RMS: {summary_rot['residual_rms']:.6f}")
    print(f"  roundtrip RMS error: {err_rot:.6f}")
    wire_rot = codec.encode_bytes(w_rot)
    ratio_rot = (w_rot.size * 4) / max(len(wire_rot), 1)
    print(f"  wire: {len(wire_rot):,} bytes ({ratio_rot:.1f}x compression)")
    print()

    # test 6: handshake by parameters
    print("--- handshake: parameter distance ---")
    pkt_sin = codec.encode_trajectory(w_sin)
    pkt_phase = codec.encode_trajectory(w_phase)

    pkt_const = codec.encode_trajectory(w_const)
    pv_sin = engram_parameter_vector(pkt_sin)
    pv_phase = engram_parameter_vector(pkt_phase)
    pv_const = engram_parameter_vector(pkt_const)

    d_self = engram_parameter_distance(pv_sin, pv_sin)
    d_diff = engram_parameter_distance(pv_sin, pv_phase)
    d_static = engram_parameter_distance(pv_sin, pv_const)
    print(f"  sin<->sin:   {d_self:.4f} (should be 0)")
    print(f"  sin<->phase: {d_diff:.4f}")
    print(f"  sin<->const: {d_static:.4f}")
    print()

    print("=== the duality ===")
    print()
    print("Glyph fits the eigenvalues of the human hand.")
    print("Engram fits the eigenvalues of a train of thought.")
    print()
    print("the residuals are not noise. they are creativity, measured in bits.")


if __name__ == "__main__":
    _self_test()
