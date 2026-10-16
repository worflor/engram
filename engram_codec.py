"""Engram functional reference by Woflo / MB.

Paired complex AR(2) trajectories, K/G signatures, statistical memory,
parameter-history generation, and versioned replay. Fitting uses float64
accumulation; signal samples remain float32. The native Rust kernels share
the reference packet and analysis contracts.

Memory contains sufficient statistics, routing anchors, a transition graph
and bounded parameter histories. Its size also depends on the number of
wells. Metadata and annotations are supplied descriptions, separate from
measured geometry. Host applications own embeddings and decisions.
"""

from __future__ import annotations

import os
import struct
import math
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Protocol

import numpy as np


def _real_array(values, dtype=np.float64):
    """Convert finite real samples without silently discarding complex parts."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(
            "real-valued samples required; encode complex coordinates as real pairs"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(
            "finite real-valued samples required in the requested precision"
        )
    return result


# optional JIT acceleration. if numba is available, the segmentation
# inner loop runs as compiled machine code. if not, pure numpy fallback.
try:
    from numba import njit as _njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def _njit(*args, **kwargs):
        def _passthrough(fn):
            return fn

        return _passthrough


# ── constants ─────────────────────────────────────────────────────────
#
# derived from the AR(2) predictor structure:

AR_ORDER = 2  # paired second-order autoregressive predictor
SEED_COUNT = AR_ORDER  # two initial conditions: z[n-1] and z[n-2]
MIN_BLOCK = SEED_COUNT + 2  # 2 seeds + 2 observations = minimum for a fit
MACHINE_EPS = np.finfo(np.float32).eps  # ~1.19e-7, the silence threshold

# wire format identifiers:

WIRE_MAGIC = b"EN"
__version__ = "0.4.0"

WIRE_VERSION = 4  # explicit reference packet state
WIRE_VERSION_COMPACT = 4  # statistics-only flag selects the compact payload

# Operational defaults. Some use predictor dimensions; routing and capture
# thresholds are explicit heuristics, not mathematical consequences of AR(2).

# Well maturity thresholds govern routing policy. Their values do not certify
# centroid accuracy or convergence of the observed distribution.
WELL_ESTABLISHED = MIN_BLOCK * AR_ORDER + AR_ORDER  # 10
WELL_RELIABLE = WELL_ESTABLISHED**2  # 100

# Scale multiplier for segmentation, generation guards and outlier thresholds.
# Its use does not establish that observations have a Gaussian distribution.
SIGMA_THRESHOLD = 3.0

# Initial anchored wells created before nearest-anchor routing engages.
# The escape rule can create further wells; this is not a global capacity cap.
BOOTSTRAP_WELLS = 32

# Escape multiplier on typical_gap. Novelty routing compares distance with
# typical_gap * ESCAPE_ENERGY * sqrt(min(depth_ratio, 1.0)). The parameter is
# a configurable clustering policy, not a physical energy or stability law.
ESCAPE_ENERGY = 1.0

# Skip micro refinement above the declared macro capture threshold. Residuals
# below that gate may still carry task-relevant signal.
MICRO_SKIP_CAPTURE = 100.0 - AR_ORDER  # 98.0

# quantization depth and the block ceiling it implies.
# 8-bit quantization = 128 distinct levels per half-axis.
# max block size = max distinguishable samples = 2^(QUANT_BITS-1).
QUANT_BITS = 8
MAX_BLOCK = 1 << (QUANT_BITS - 1)  # 128


# ── block modes ───────────────────────────────────────────────────────


class Mode(IntEnum):
    CASCADED = 0  # macro + micro oscillators (the full hierarchical predictor)
    LINEAR = 1  # K=2, G=1: constant velocity. the degenerate straight line.
    STATIC = 2  # K=1, G=0: hold. the trajectory is at rest.
    RAW = 3  # passthrough. not enough samples for prediction.


# ── trajectory source ──────────────────────────────────────────────────
#
# the codec does not know what generated the trajectory. a source protocol
# provides the projection from raw domain data to [T, D] float32.


class TrajectorySource(Protocol):
    """any source of temporal trajectories in R^D."""

    def project(self, raw: np.ndarray) -> np.ndarray: ...


class TanhProjection:
    """
    Project rows as tanh(α · S · x) in a fixed real coordinate system.

    Calibration sets α = atanh(0.5) / max(median(|S·x|), MACHINE_EPS).
    At the calibration median this targets magnitude 0.5, except when the
    numerical floor is active. It does not bound the fraction of saturated
    values or preserve an AR(2) recurrence through the nonlinear transform.
    Fit α on representative training data and retain it with the matrix;
    changing either changes the coordinates of accumulated memory.
    """

    def __init__(self, matrix: np.ndarray, alpha: Optional[float] = None):
        self.S = _real_array(matrix, dtype=np.float32).copy()
        if (
            self.S.ndim != 2
            or not self.S.size
            or self.S.shape[0] % 2
            or not np.isfinite(self.S).all()
            or self.S.size > 16 * 1024 * 1024
            or (alpha is not None and (not math.isfinite(alpha) or alpha < 0))
        ):
            raise ValueError(
                "finite even-output projection matrix and nonnegative alpha required"
            )
        self._alpha = alpha

    def _input(self, raw):
        raw = _real_array(raw, dtype=np.float32)
        if (
            raw.ndim != 2
            or raw.shape[1] < self.S.shape[1]
            or not np.isfinite(raw).all()
            or raw.size > 16 * 1024 * 1024
            or len(raw) * self.S.shape[0] > 16 * 1024 * 1024
        ):
            raise ValueError(
                "finite bounded projection input with enough coordinates required"
            )
        return raw[:, : self.S.shape[1]]

    @property
    def alpha(self) -> float:
        if self._alpha is None:
            raise RuntimeError(
                "alpha not calibrated — call calibrate() or project() first"
            )
        return self._alpha

    def calibrate(self, sample: np.ndarray) -> float:
        """derive α from median projection magnitude of a representative sample."""
        h = self._input(sample)
        if not len(h):
            raise ValueError("calibration needs a nonempty sample")
        proj = h.astype(np.float64) @ self.S.astype(np.float64).T
        median_abs = float(np.median(np.abs(proj)))
        # Map the untransformed median magnitude to 0.5 unless the floor binds.
        # This supplies no distribution-wide bound on nonlinear saturation.
        self._alpha = math.atanh(0.5) / max(median_abs, float(MACHINE_EPS))
        return self._alpha

    def project(self, raw: np.ndarray) -> np.ndarray:
        h = self._input(raw)
        if not len(h):
            return np.empty((0, self.S.shape[0]), np.float32)
        proj = h.astype(np.float64) @ self.S.astype(np.float64).T
        if self._alpha is None:
            self.calibrate(raw)
        with np.errstate(over="ignore"):
            return np.tanh(self._alpha * proj).astype(np.float32)


class Identity:
    """passthrough for pre-projected data (embeddings, MFCCs, etc.)."""

    def project(self, raw: np.ndarray) -> np.ndarray:
        return _real_array(raw, dtype=np.float32).copy()


class ByteHistogram:
    """Normalized 256-bin byte frequencies over complete non-overlapping windows.

    No model or dictionary is required. These are distributional input
    coordinates; downstream task meaning depends on the host's evaluation.
    """

    def __init__(self, chunk_size: int = 512):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        self.dim = 256

    def text_to_trajectory(self, text: str) -> np.ndarray:
        """
        convert text to a [T, 256] trajectory of byte frequency histograms.

        chunk size is fixed at self.chunk_size. short text that can't
        fill at least 4 windows returns an empty trajectory. this is by
        design: the histogram needs enough bytes for stable statistics.
        for chat-length text, accumulate multiple messages before measuring.
        """
        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        cs = self.chunk_size
        n = (len(data) // cs) * cs
        if n < cs * MIN_BLOCK:
            return np.empty((0, 256), dtype=np.float32)
        chunks = data[:n].reshape(-1, cs)
        T = chunks.shape[0]
        hist = np.zeros((T, 256), dtype=np.float32)
        for t in range(T):
            np.add.at(hist[t], chunks[t], 1.0)
            hist[t] /= cs
        return hist

    def bytes_to_trajectory(self, data: bytes) -> np.ndarray:
        """convert raw bytes to a [T, 256] trajectory. works on any byte stream."""
        arr = np.frombuffer(data, dtype=np.uint8)
        cs = self.chunk_size
        n = (len(arr) // cs) * cs
        if n < cs * MIN_BLOCK:
            return np.empty((0, 256), dtype=np.float32)
        chunks = arr[:n].reshape(-1, cs)
        T = chunks.shape[0]
        hist = np.zeros((T, 256), dtype=np.float32)
        for t in range(T):
            np.add.at(hist[t], chunks[t], 1.0)
            hist[t] /= cs
        return hist


# ── eigenvalue extraction ─────────────────────────────────────────────
#
# The roots of λ² − Kλ + G describe characteristic modes of each recurrence.
# They are local fitted dynamics, not a Fourier decomposition of the signal.


def eigenvalues(K: complex, G: complex) -> Tuple[complex, complex]:
    """
    solve λ² − K·λ + G = 0 for the two eigenvalues.

    |λ| < 1 gives a decaying mode; a repeated unit root may grow polynomially.
    The larger quadratic numerator is evaluated directly and its partner is
    recovered from the root product G, avoiding subtractive cancellation.
    """
    first, second = eigenvalues_all(np.asarray(K), np.asarray(G))
    if first.ndim != 0:
        raise ValueError("scalar oscillator coefficients required")
    return complex(first), complex(second)


def eigenvalues_all(
    K_arr: np.ndarray, G_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """eigenvalues for P pairs. returns (λ₁[P], λ₂[P]) complex arrays."""
    k, g = np.broadcast_arrays(
        np.asarray(K_arr, dtype=complex), np.asarray(G_arr, dtype=complex)
    )
    if not np.isfinite(k).all() or not np.isfinite(g).all():
        raise ValueError("finite oscillator coefficients required")
    # Component scales avoid overflowing abs(complex) and K**2. Sequential
    # divisions avoid forming scale**2, which need not be representable.
    scale = np.maximum(
        np.maximum(abs(k.real), abs(k.imag)),
        np.sqrt(np.maximum(abs(g.real), abs(g.imag))),
    )
    # Binary scaling is exact for representable powers of two and preserves
    # simple dyadic coefficient cases without extra normalization rounding.
    safe_scale = np.where(
        scale == 0, 1.0, np.ldexp(np.ones_like(scale), np.frexp(scale)[1] - 1)
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        normalized_k = k / safe_scale
        scaled_g = g / safe_scale
        normalized_g = scaled_g / safe_scale
        discriminant = np.sqrt(normalized_k * normalized_k - 4 * normalized_g)
        plus = (normalized_k + discriminant) / 2
        minus = (normalized_k - discriminant) / 2
        plus_is_larger = abs(plus) >= abs(minus)
        numerator = np.where(plus_is_larger, plus, minus)
        large = numerator * safe_scale
        small = np.divide(
            scaled_g, numerator, out=np.zeros_like(numerator), where=numerator != 0
        )
    first, second = (
        np.where(plus_is_larger, large, small),
        np.where(plus_is_larger, small, large),
    )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise FloatingPointError("oscillator roots exceeded finite range")
    return first, second


def spectral_summary(
    K_arr: np.ndarray, G_arr: np.ndarray, tolerance: float = 1e-6
) -> dict:
    """
    eigenvalue statistics for a set of oscillator pairs.
    frequencies (arg), magnitudes (abs), and stability classification.
    """
    K_arr, G_arr = np.asarray(K_arr, dtype=complex), np.asarray(G_arr, dtype=complex)
    if (
        K_arr.ndim != 1
        or K_arr.shape != G_arr.shape
        or not np.isfinite(K_arr).all()
        or not np.isfinite(G_arr).all()
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "finite matching coefficient vectors and nonnegative tolerance required"
        )
    l1, l2 = eigenvalues_all(K_arr, G_arr)
    with np.errstate(over="ignore", invalid="ignore"):
        mags = np.concatenate([np.hypot(l1.real, l1.imag), np.hypot(l2.real, l2.imag)])
    if not np.isfinite(mags).all():
        raise FloatingPointError("spectral radius exceeded finite range")
    magnitude_scale = float(np.max(mags, initial=0))
    normalized_magnitudes = mags / magnitude_scale if magnitude_scale else mags
    freqs = np.concatenate([np.angle(l1), np.angle(l2)])
    return {
        "n_pairs": len(K_arr),
        "freq_mean": float(np.mean(np.abs(freqs))) if len(freqs) else 0.0,
        "freq_std": float(np.std(np.abs(freqs))) if len(freqs) else 0.0,
        "mag_mean": float(np.mean(normalized_magnitudes)) * magnitude_scale
        if len(mags)
        else 0.0,
        "mag_std": float(np.std(normalized_magnitudes)) * magnitude_scale
        if len(mags)
        else 0.0,
        "n_decaying": int(np.sum(mags < 1.0 - tolerance)),
        "n_sustained": int(
            np.sum((mags >= 1.0 - tolerance) & (mags <= 1.0 + tolerance))
        ),
        "n_divergent": int(np.sum(mags > 1.0 + tolerance)),
    }


# ── core AR(2) fitting ────────────────────────────────────────────────
#
# the only algorithm in the codec. everything else is scaffolding around
# this one operation: find the K and G that best predict the next sample
# from the previous two.
#
# for a complex time series z[0..T-1], minimize:
#     E = Σ |z[n] − K·z[n−1] + G·z[n−2]|²
#
# the 2×2 Hermitian normal equation (Cramer solution):
#     [sAA+r  −s̄AB] [K]   [sTA]
#     [−sAB   sBB+r] [G] = [sTB]     (r = Tikhonov ridge)
#
# for a pure sinusoid at frequency ω: K = 2cos(ω), G = 1.
# for a decaying spiral at rate r, frequency ω: K = 2r·cos(ω), G = r².
# for a circle: K = e^{iθ}, G = 0.
# for an ellipse: both eigenvalues of λ² − Kλ + G = 0 are needed.


def fit_pair(z: np.ndarray) -> Tuple[complex, complex, float]:
    """
    fit z[n] = K·z[n−1] − G·z[n−2] on a single complex time series.

    Ridge is max(MACHINE_EPS, 1e-4 * trace(H) / 2), with the same
    coefficient penalty as fit_all. This regularizes rank-deficient designs
    and biases coefficients toward zero; its scale is a numerical policy.

    returns (K, G, rms_error).
    """
    z = np.asarray(z, dtype=np.complex128)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("fit requires a finite complex vector")
    T = z.shape[0]
    if T < SEED_COUNT + 1:
        return 2.0 + 0j, 1.0 + 0j, 0.0

    A, B, tgt = z[1:-1], z[:-2], z[2:]

    sAA = float(np.real(np.sum(A * np.conj(A))))
    sBB = float(np.real(np.sum(B * np.conj(B))))
    sAB = complex(np.sum(A * np.conj(B)))
    sTA = complex(np.sum(tgt * np.conj(A)))
    sTB = complex(np.sum(tgt * np.conj(B)))

    # Trace-scaled coefficient penalty with an absolute numerical floor.
    trace = sAA + sBB
    rank = 2.0
    ridge = max(MACHINE_EPS, 1e-4 * trace / rank)

    sAA_r = sAA + ridge
    sBB_r = sBB + ridge
    det = sAA_r * sBB_r - abs(sAB) ** 2

    if abs(det) < MACHINE_EPS * trace * trace:
        return 2.0 + 0j, 1.0 + 0j, _rms(z, 2.0 + 0j, 1.0 + 0j)

    K = (sTA * sBB_r - np.conj(sAB) * sTB) / det
    G = (sAB * sTA - sAA_r * sTB) / det
    return K, G, _rms(z, K, G)


def _rms(z: np.ndarray, K: complex, G: complex) -> float:
    if z.shape[0] < SEED_COUNT + 1:
        return 0.0
    err = z[2:] - (K * z[1:-1] - G * z[:-2])
    return float(np.sqrt(np.mean(np.abs(err) ** 2)))


# ── dimension pairing ─────────────────────────────────────────────────
#
# Pairing selects the two-dimensional planes on which independent complex
# recurrences are fitted. Absolute temporal correlation supplies a greedy
# data-derived matching; it is not guaranteed optimal for a downstream task.
# A fixed host pairing is required for comparing or accumulating signatures.


def derive_pairing(w: np.ndarray) -> np.ndarray:
    """
    Choose a greedy complex pairing from absolute temporal correlation.

    computes absolute temporal correlation between all dimension pairs,
    then greedily matches the most correlated pair, removes both, repeats.
    returns pairing[D] where pairing[2k] and pairing[2k+1] are the
    original dimension indices for oscillator pair k.

    cost: O(D² · T) for correlation, O(D²) for greedy matching.
    The cost can be material for wide inputs; retain a fixed basis when possible.
    """
    T, D = w.shape
    P = D // 2

    if T < 2:
        # not enough data to correlate. fall back to sequential pairing.
        return np.arange(D, dtype=np.int32)

    # correlation magnitude between all dimension pairs.
    # subtract mean, normalize, compute abs(dot product).
    w = np.asarray(w, dtype=np.float64)
    centered = w - w.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    norms = np.where(norms < MACHINE_EPS, 1.0, norms)
    normed = centered / norms

    # abs correlation matrix (D×D). we want the strongest |correlation|
    # regardless of sign, because anti-correlated dims also form good
    # complex pairs (they rotate at 180° phase offset).
    corr = np.abs(normed.T @ normed / max(T - 1, 1))

    # zero the diagonal (don't pair a dimension with itself)
    np.fill_diagonal(corr, 0.0)

    # greedy matching: repeatedly pick the strongest remaining pair.
    # vectorized: mask used dims by zeroing their rows/cols in corr,
    # then argmax the full matrix each iteration. O(P × D) not O(P × D²).
    pairing = np.empty(D, dtype=np.int32)
    work = corr.copy()
    work[np.tril_indices(D)] = -np.inf
    idx = 0

    for _ in range(P):
        flat = work.argmax()
        best_i, best_j = divmod(int(flat), D)
        pairing[idx] = best_i
        pairing[idx + 1] = best_j
        # zero out both dims so they can't be picked again
        work[best_i, :] = -np.inf
        work[:, best_i] = -np.inf
        work[best_j, :] = -np.inf
        work[:, best_j] = -np.inf
        idx += 2

    return pairing


def apply_pairing(w: np.ndarray, pairing: np.ndarray) -> np.ndarray:
    """reorder dimensions according to pairing. [T, D] → [T, D]."""
    return w[:, pairing].astype(np.float32)


def undo_pairing(w: np.ndarray, pairing: np.ndarray) -> np.ndarray:
    """reverse the pairing reorder. [T, D] → [T, D]. vectorized via inverse permutation."""
    D = w.shape[1]
    inv = np.empty(D, dtype=np.int32)
    inv[pairing] = np.arange(D, dtype=np.int32)
    return w[:, inv]


def _to_complex(w: np.ndarray) -> np.ndarray:
    """[T, D] float → [T, D//2] complex. assumes paired ordering."""
    paired = w.reshape(w.shape[0], w.shape[1] // 2, 2)
    return paired[..., 0].astype(np.float64) + 1j * paired[..., 1].astype(np.float64)


def _from_complex(z: np.ndarray) -> np.ndarray:
    """[T, P] complex → [T, 2P] float."""
    out = np.empty((z.shape[0], z.shape[1] * 2), dtype=np.float32)
    out[:, 0::2] = z.real.astype(np.float32)
    out[:, 1::2] = z.imag.astype(np.float32)
    return out


def fit_all(z_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    fit P independent AR(2) oscillators in one vectorized pass.
    all pairs are fitted simultaneously with batch numpy operations.
    no Python loop over pairs. returns K[P], G[P], mean_rms.
    """
    z_c = np.asarray(z_c, dtype=np.complex128)
    if z_c.ndim != 2 or not np.isfinite(z_c).all():
        raise ValueError("fit_all requires a finite complex matrix")
    T, P = z_c.shape
    if T < SEED_COUNT + 1:
        return (
            np.full(P, 2.0 + 0j, dtype=np.complex128),
            np.ones(P, dtype=np.complex128),
            0.0,
            np.zeros(P, dtype=np.float32),
        )

    A = z_c[1:-1]  # [T-2, P]
    B = z_c[:-2]  # [T-2, P]
    tgt = z_c[2:]  # [T-2, P]

    # batch cross-terms: sum over time axis, keep per-pair.
    # some trajectories have extreme values that overflow float64 when
    # multiplied and summed. the degenerate check below catches these
    # and falls back to linear, so the overflow is harmless but noisy.
    with np.errstate(over="ignore", invalid="ignore"):
        sAA = np.real(np.sum(A * np.conj(A), axis=0))  # [P] real
        sBB = np.real(np.sum(B * np.conj(B), axis=0))  # [P] real
        sAB = np.sum(A * np.conj(B), axis=0)  # [P] complex
        sTA = np.sum(tgt * np.conj(A), axis=0)  # [P] complex
        sTB = np.sum(tgt * np.conj(B), axis=0)  # [P] complex

        # ridge per pair
        trace = sAA + sBB
        ridge = np.maximum(MACHINE_EPS, 1e-4 * trace / 2.0)

        sAA_r = sAA + ridge
        sBB_r = sBB + ridge
        det = sAA_r * sBB_r - np.abs(sAB) ** 2

        # solve: K and G per pair
        K = (sTA * sBB_r - np.conj(sAB) * sTB) / det
        G = (sAB * sTA - sAA_r * sTB) / det

        # handle degenerate pairs (det ≈ 0 or overflow → inf/nan)
        degen = ~np.isfinite(det) | (np.abs(det) < MACHINE_EPS * trace * trace)
        K = np.where(degen | ~np.isfinite(K), 2.0 + 0j, K)
        G = np.where(degen | ~np.isfinite(G), 1.0 + 0j, G)

        # batch RMS
        pred = K[np.newaxis, :] * A - G[np.newaxis, :] * B
        err = tgt - pred
        rms_per_pair = np.sqrt(np.mean(np.abs(err) ** 2, axis=0))  # [P]
        rms_per_pair = np.where(np.isfinite(rms_per_pair), rms_per_pair, 0.0)
        mean_rms = float(np.mean(rms_per_pair))

    return (
        K.astype(np.complex128),
        G.astype(np.complex128),
        mean_rms,
        rms_per_pair.astype(np.float32),
    )


def predict_all(
    s1: np.ndarray,
    s2: np.ndarray,
    K: np.ndarray,
    G: np.ndarray,
    length: int,
) -> np.ndarray:
    """spin up the oscillators: generate [length, P] complex from seeds + K,G.
    the inner loop is inherently sequential (each step depends on the previous)
    but the P pairs are independent so the per-step multiply is vectorized."""
    P = s1.shape[0]
    out = np.empty((length, P), dtype=np.complex128)
    prev1 = s1.astype(np.complex128)
    prev2 = s2.astype(np.complex128)
    K_c = K.astype(np.complex128)
    G_c = G.astype(np.complex128)
    for n in range(length):
        out[n] = K_c * prev1 - G_c * prev2
        prev2, prev1 = prev1, out[n]
    return out


# ── data-derived block sizing ─────────────────────────────────────────
#
# the block size is not a tuning parameter. it is a property of the data.
# the autocorrelation length of the trajectory velocity tells you how far
# the current dynamics persist before the signal changes character.
# a slow meditation has long autocorrelation (big blocks). a jazz solo
# has short autocorrelation (small blocks). the codec adapts automatically.


def _autocorrelation_length(w: np.ndarray, max_lag: int = MAX_BLOCK) -> int:
    """
    the natural block size: first zero-crossing of the velocity ACF.

    velocity = first difference of the trajectory. its autocorrelation
    measures how far the current rate of change persists. when the ACF
    crosses zero, the dynamics have decorrelated. that's where you cut.
    """
    T = w.shape[0]
    if T < 4:
        return T

    vel = np.diff(np.asarray(w, dtype=np.float64), axis=0)
    vel_norm = np.linalg.norm(vel, axis=1)

    if vel_norm.std() < MACHINE_EPS:
        return T

    vn = vel_norm - vel_norm.mean()
    var = float(np.dot(vn, vn))
    if var < MACHINE_EPS:
        return T

    max_lag = min(max_lag, len(vn) // 2)
    for lag in range(1, max_lag):
        acf = float(np.dot(vn[:-lag], vn[lag:])) / var
        if acf <= 0:
            return max(MIN_BLOCK, lag)

    return max(MIN_BLOCK, max_lag)


def derive_block_sizes(w: np.ndarray) -> Tuple[int, int]:
    """
    derive macro and micro block sizes from the trajectory itself.

    macro = velocity autocorrelation length (the slow timescale).
    micro = macro / 4 (the fast timescale is at least 4× shorter —
            the harmonic separation principle from audio coding).
    """
    macro = min(MAX_BLOCK, max(MIN_BLOCK, _autocorrelation_length(w)))
    # micro = macro / MIN_BLOCK: guarantees MIN_BLOCK micro sub-blocks
    # per macro block, enough for the micro oscillator to fit.
    micro = max(MIN_BLOCK, macro // MIN_BLOCK)
    return macro, micro


# ── block segmentation ────────────────────────────────────────────────
#
# the trajectory is cut at phase transitions: moments where the linear
# predictor (constant velocity) suddenly fails. the threshold is the
# 3-sigma rule from normal distribution theory (P > 99.7%), not a
# tuning knob. if the prediction error is Gaussian, a spike beyond
# 3σ happens less than 0.3% of the time by chance alone.


@_njit(cache=True)
def _compute_prediction_errors(w: np.ndarray) -> np.ndarray:
    """vectorized linear prediction error per timestep."""
    T = w.shape[0]
    err = np.zeros(T, dtype=np.float64)
    for t in range(2, T):
        s = 0.0
        for d in range(w.shape[1]):
            diff = w[t, d] - (2.0 * w[t - 1, d] - w[t - 2, d])
            s += diff * diff
        err[t] = s**0.5
    return err


def segment(
    w: np.ndarray,
    max_block: int,
) -> List[Tuple[int, int]]:
    """
    segment trajectory at prediction error spikes (3-sigma rule).

    also enforces max_block as an upper bound on block length, and
    merges trailing blocks that are too short for a valid fit.
    """
    T = w.shape[0]
    if T <= SEED_COUNT:
        return []

    values = np.asarray(w, dtype=np.float64)
    err = np.zeros(T, dtype=np.float64)
    err[SEED_COUNT:] = np.linalg.norm(
        values[2:] - 2 * values[1:-1] + values[:-2], axis=1
    )

    segments: List[Tuple[int, int]] = []
    block_start = SEED_COUNT

    if T > SEED_COUNT + 2:
        e = err[SEED_COUNT:]
        e_mean = float(np.mean(e))
        e_std = float(np.std(e)) + MACHINE_EPS
        threshold = e_mean + SIGMA_THRESHOLD * e_std
    else:
        threshold = float("inf")

    for t in range(SEED_COUNT + 1, T):
        bl = t - block_start
        if bl >= max_block:
            segments.append((block_start, t))
            block_start = t
            continue
        if bl >= MIN_BLOCK and err[t] > threshold:
            segments.append((block_start, t))
            block_start = t

    if block_start < T:
        segments.append((block_start, T))

    if len(segments) > 1 and (segments[-1][1] - segments[-1][0]) < MIN_BLOCK:
        last = segments.pop()
        prev = segments.pop()
        segments.append((prev[0], last[1]))

    return segments


# ── per-pair adaptive quantization ────────────────────────────────────
#
# each pair scales independently. a pair where the oscillator predicted
# perfectly (residual ≈ 0) gets the full int8 range for sub-epsilon
# precision. a pair with large residuals uses a coarse scale. the codec
# spends bits where the oscillator failed and saves bits where it succeeded.


def quantize(
    residuals: np.ndarray, bits: int = QUANT_BITS
) -> Tuple[np.ndarray, np.ndarray]:
    """per-pair adaptive quantization. fully vectorized, no Python loop."""
    L, D = residuals.shape
    P = D // 2
    max_val = (1 << (bits - 1)) - 1

    # reshape to [L, P, 2], compute max abs per pair
    paired = residuals.reshape(L, P, 2)
    scales = np.abs(paired).max(axis=(0, 2)).astype(np.float32)  # [P]

    # scale factors: max_val / scale, with zero-scale pairs getting 0
    with np.errstate(divide="ignore"):
        safe_scales = np.where(scales < MACHINE_EPS, 0.0, max_val / scales)  # [P]

    # broadcast multiply, clip, cast
    scaled = paired * safe_scales[np.newaxis, :, np.newaxis]  # [L, P, 2]
    q = np.clip(np.round(scaled), -max_val, max_val).astype(np.int8)

    return q.reshape(L, D), scales


def dequantize(q: np.ndarray, scales: np.ndarray, bits: int = 8) -> np.ndarray:
    """inverse of quantize. fully vectorized."""
    L, D = q.shape
    P = D // 2
    max_val = (1 << (bits - 1)) - 1

    paired = q.reshape(L, P, 2).astype(np.float32)
    inv_scales = np.where(scales < MACHINE_EPS, 0.0, scales / max_val)  # [P]
    out = paired * inv_scales[np.newaxis, :, np.newaxis]
    return out.reshape(L, D)


# ── varint coding ─────────────────────────────────────────────────────
#
# zigzag maps signed → unsigned (0→0, −1→1, 1→2, −2→3, ...) so that
# small-magnitude residuals (the common case after oscillator prediction)
# produce small varints. LEB128 encodes each uint in 1-5 bytes.


def _zigzag(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.int32)
    return ((v << 1) ^ (v >> 31)).astype(np.uint32)


def _unzigzag(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.uint32)
    return ((v >> 1).astype(np.int32)) ^ (-(v & 1).astype(np.int32))


@_njit(cache=True)
def _pack_varints_numba(zz: np.ndarray) -> np.ndarray:
    """JIT varint encoder. returns uint8 array (pre-allocated worst case)."""
    n = zz.shape[0]
    buf = np.empty(n * 5, dtype=np.uint8)  # worst case: 5 bytes per varint
    pos = 0
    for i in range(n):
        v = int(zz[i]) & 0xFFFFFFFF
        while v >= 0x80:
            buf[pos] = (v & 0x7F) | 0x80
            pos += 1
            v >>= 7
        buf[pos] = v & 0x7F
        pos += 1
    return buf[:pos]


@_njit(cache=True)
def _unpack_varints_numba(data: np.ndarray, count: int) -> np.ndarray:
    """JIT varint decoder."""
    out = np.empty(count, dtype=np.int64)
    off = 0
    for i in range(count):
        result = np.int64(0)
        shift = 0
        while off < data.shape[0]:
            b = np.int64(data[off])
            off += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        # unzigzag inline
        v = result & 0xFFFFFFFF
        out[i] = np.int64(v >> 1) ^ -(np.int64(v) & 1)
    return out


def pack_varints(flat: np.ndarray) -> bytes:
    zz = _zigzag(flat.ravel().astype(np.int32))
    if _HAS_NUMBA:
        return bytes(_pack_varints_numba(zz))
    out = bytearray()
    for v in zz:
        v = int(v) & 0xFFFFFFFF
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v & 0x7F)
    return bytes(out)


def unpack_varints(data: bytes, count: int) -> np.ndarray:
    if _HAS_NUMBA:
        return _unpack_varints_numba(np.frombuffer(data, dtype=np.uint8), count).astype(
            np.int32
        )
    out = np.empty(count, dtype=np.int32)
    off = 0
    for i in range(count):
        result = shift = 0
        while off < len(data):
            b = data[off]
            off += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        out[i] = _unzigzag(np.array([result & 0xFFFFFFFF], dtype=np.uint32))[0]
    return out


# ── block data structure ──────────────────────────────────────────────
#
# a Block holds the orbital parameters and residuals for one segment of
# the trajectory. the decoder reconstructs the segment by spinning up
# the macro oscillators, then the micro oscillators on the macro residuals,
# then adding the final residuals. encode is analysis. decode is simulation.


@dataclass(slots=True)
class Block:
    mode: Mode
    start: int  # offset into the full trajectory
    length: int  # number of timesteps in this block
    macro_K: np.ndarray  # [P] complex — fundamental frequencies
    macro_G: np.ndarray  # [P] complex — fundamental damping
    micro_Ks: List[np.ndarray]  # list of [P] complex per sub-block — harmonics
    micro_Gs: List[np.ndarray]  # list of [P] complex per sub-block — harmonic damping
    residuals: np.ndarray  # [length, D] float32 — irreducible surprise
    seed1: np.ndarray  # [D] z[n−1] at block boundary
    seed2: np.ndarray  # [D] z[n−2] at block boundary
    scales: np.ndarray  # [P] per-pair quantization scale
    signal_energy: float = 0.0  # total signal energy in block (for SNR)
    residual_energy: float = 0.0  # total residual energy (for SNR)
    macro_capture: float = 0.0  # % signal energy in the fundamental
    micro_capture: float = 0.0  # % macro residual energy in the harmonics
    pair_rms: Optional[np.ndarray] = None  # [P] per-pair residual RMS (surprise scale)
    micro_size: int = 0  # zero selects the historical inferred partition


@dataclass(slots=True)
class Packet:
    """
    the encoded trajectory. orbital parameters + quantized residuals.
    this is the .engram file format: initial conditions for a dynamical system.
    to recall the trajectory, you don't load it. you simulate it.
    """

    dim: int  # D (must be even)
    n_pairs: int  # D // 2 complex oscillator pairs
    length: int  # total timesteps in the trajectory
    alpha: float  # projection temperature (0 = identity/pre-projected)
    macro_block: int  # data-derived macro block size (slow timescale)
    micro_block: int  # data-derived micro block size (fast timescale)
    pairing: np.ndarray  # [D] int32 — which dims form each complex pair
    blocks: List[Block]
    reconstructable: bool = True
    basis_known: bool = True

    def with_pairing(self, pairing):
        """Attach the host's original coordinate basis to a historical compact packet."""
        from dataclasses import replace

        if self.basis_known or self.reconstructable:
            raise ValueError("only a missing legacy compact basis can be attached")
        basis = np.asarray(pairing)
        if (
            basis.shape != (self.dim,)
            or not np.issubdtype(basis.dtype, np.integer)
            or not np.array_equal(np.sort(basis), np.arange(self.dim))
        ):
            raise ValueError("pairing must be a dimension permutation")
        return replace(self, pairing=basis.astype(np.int32), basis_known=True)

    @property
    def total_energy(self) -> float:
        return sum(b.signal_energy for b in self.blocks)

    @property
    def residual_energy_total(self) -> float:
        return sum(b.residual_energy for b in self.blocks)

    @property
    def capture(self) -> float:
        """total energy capture across all blocks, in percent."""
        te = float(self.total_energy)
        if te < MACHINE_EPS:
            return 0.0
        return round(max(0.0, (1.0 - float(self.residual_energy_total) / te) * 100), 4)

    @property
    def snr_db(self) -> float:
        """signal-to-residual ratio in dB."""
        re = self.residual_energy_total
        if re < MACHINE_EPS:
            return float("inf")
        return 10.0 * math.log10(max(self.total_energy, MACHINE_EPS) / re)


# ── hierarchical encode ──────────────────────────────────────────────
#
# cascaded prediction, same architecture as Harmonic audio:
#     fundamental → residuals → harmonics → residuals → entropy coder
#
# here:
#     macro oscillators → macro residuals → micro oscillators → final residuals
#
# the macro captures what persists across the block (the orbit).
# the micro captures what oscillates within the macro residuals (the jitter).
# what's left is genuine surprise: the irreducible information content.


def _raw_block(data, seed1, seed2):
    length, dim = data.shape
    energy = float(np.sum(np.asarray(data, dtype=np.float64) ** 2))
    return Block(
        Mode.RAW,
        0,
        length,
        np.zeros(dim // 2, complex),
        np.zeros(dim // 2, complex),
        [],
        [],
        data.copy(),
        seed1.copy(),
        seed2.copy(),
        np.zeros(dim // 2, dtype=np.float32),
        signal_energy=energy,
        residual_energy=energy,
    )


def encode_block(
    data: np.ndarray,  # [L, D]
    seed1: np.ndarray,  # [D] z[n−1]
    seed2: np.ndarray,  # [D] z[n−2]
    micro_size: int,
) -> Block:
    """
    cascaded encoding:
        1. macro AR(2) on full block → macro residuals
        2. micro AR(2) on sub-blocks of macro residuals → final residuals
    the residuals are what both oscillators could not predict.
    mode selection compares CASCADED vs LINEAR and picks the winner.
    """
    L, D = data.shape
    P = D // 2
    signal_energy = float(np.sum(data.astype(np.float64) ** 2))
    if L < SEED_COUNT + 1:
        return _raw_block(data, seed1, seed2)

    # STATIC: the trajectory is at rest. variance below machine precision.
    # costs zero residual bits — the decoder just holds the seed value.
    signal_scale = float(np.abs(data).max()) + MACHINE_EPS
    if float(np.max(np.abs(data - seed1))) <= MACHINE_EPS * signal_scale:
        return Block(
            mode=Mode.STATIC,
            start=0,
            length=L,
            macro_K=np.full(P, 1.0 + 0j),
            macro_G=np.zeros(P, dtype=np.complex128),
            micro_Ks=[],
            micro_Gs=[],
            residuals=np.zeros_like(data),
            seed1=seed1.copy(),
            seed2=seed2.copy(),
            scales=np.zeros(P, dtype=np.float32),
            signal_energy=signal_energy,
            residual_energy=0.0,
            macro_capture=100.0,
        )

    z_full = np.vstack([seed2[np.newaxis], seed1[np.newaxis], data])
    z_c = _to_complex(z_full)
    s1_c = _to_complex(seed1[np.newaxis])[0]
    s2_c = _to_complex(seed2[np.newaxis])[0]

    # ── try CASCADED: macro (fundamental) + micro (harmonics) ────
    macro_K, macro_G, _, _ = fit_all(z_c)
    macro_pred = _from_complex(predict_all(s1_c, s2_c, macro_K, macro_G, L))
    if not np.isfinite(macro_pred).all():
        return _raw_block(data, seed1, seed2)
    macro_resid = data - macro_pred
    if not np.isfinite(macro_resid).all():
        return _raw_block(data, seed1, seed2)
    macro_resid_energy = float(np.sum(macro_resid.astype(np.float64) ** 2))
    if not np.isfinite(macro_resid_energy):
        macro_resid_energy = (
            signal_energy  # divergent prediction, treat as zero capture
        )
    macro_capture = max(
        0.0, (1.0 - macro_resid_energy / max(signal_energy, MACHINE_EPS)) * 100
    )

    # micro: fit on sub-blocks of macro residuals
    # threshold gate (from audio codec): skip micro when macro residual
    # energy is too small. prevents fitting noise and divergent predictions.
    micro_Ks: List[np.ndarray] = []
    micro_Gs: List[np.ndarray] = []
    micro_pred = np.zeros_like(macro_resid)

    skip_micro = macro_capture > MICRO_SKIP_CAPTURE

    n_sub = (L + micro_size - 1) // micro_size
    if skip_micro:
        n_sub = 0
    for si in range(n_sub):
        ss = si * micro_size
        se = min(ss + micro_size, L)
        sl = se - ss

        if sl < SEED_COUNT + 1:
            micro_Ks.append(np.zeros(P, dtype=np.complex128))
            micro_Gs.append(np.zeros(P, dtype=np.complex128))
            continue

        if ss >= SEED_COUNT:
            ms2, ms1 = macro_resid[ss - 2], macro_resid[ss - 1]
        elif ss == 1:
            ms2, ms1 = np.zeros(D, dtype=np.float32), macro_resid[0]
        else:
            ms2 = ms1 = np.zeros(D, dtype=np.float32)

        z_sub = np.vstack([ms2[np.newaxis], ms1[np.newaxis], macro_resid[ss:se]])
        mK, mG, _, _ = fit_all(_to_complex(z_sub))
        micro_Ks.append(mK)
        micro_Gs.append(mG)

        ms1_c = _to_complex(ms1[np.newaxis])[0]
        ms2_c = _to_complex(ms2[np.newaxis])[0]
        micro_pred[ss:se] = _from_complex(predict_all(ms1_c, ms2_c, mK, mG, sl))
        if not np.isfinite(micro_pred[ss:se]).all():
            return _raw_block(data, seed1, seed2)

    cascaded_resid = macro_resid - micro_pred
    if not np.isfinite(cascaded_resid).all():
        return _raw_block(data, seed1, seed2)
    cascaded_energy = float(np.sum(cascaded_resid.astype(np.float64) ** 2))
    if not np.isfinite(cascaded_energy):
        cascaded_energy = macro_resid_energy
    micro_capture = max(
        0.0, (1.0 - cascaded_energy / max(macro_resid_energy, MACHINE_EPS)) * 100
    )

    # per-pair residual RMS: the surprise scale for each oscillator pair.
    # stored in the block for dream mode, so the dream can generate
    # surprise at the right magnitude without needing the full residuals.
    def _compute_pair_rms(resid: np.ndarray, P: int) -> np.ndarray:
        r2 = resid.reshape(resid.shape[0], P, 2) ** 2
        return np.sqrt(np.mean(r2, axis=(0, 2)))

    # ── try LINEAR: constant velocity, no parameters ─────────────
    linear_pred = _from_complex(
        predict_all(
            s1_c,
            s2_c,
            np.full(P, 2.0 + 0j),
            np.full(P, 1.0 + 0j),
            L,
        )
    )
    if not np.isfinite(linear_pred).all() or not np.isfinite(data - linear_pred).all():
        return _raw_block(data, seed1, seed2)
    linear_energy = float(np.sum((data.astype(np.float64) - linear_pred) ** 2))

    # ── mode selection: cascaded wins if it captures more energy ──
    if cascaded_energy < linear_energy:
        _, scales = quantize(cascaded_resid)
        return Block(
            mode=Mode.CASCADED,
            start=0,
            length=L,
            micro_size=micro_size,
            macro_K=macro_K,
            macro_G=macro_G,
            micro_Ks=micro_Ks,
            micro_Gs=micro_Gs,
            residuals=cascaded_resid,
            seed1=seed1.copy(),
            seed2=seed2.copy(),
            scales=scales,
            signal_energy=signal_energy,
            residual_energy=cascaded_energy,
            macro_capture=macro_capture,
            micro_capture=micro_capture,
            pair_rms=_compute_pair_rms(cascaded_resid, P),
        )
    else:
        linear_resid = data - linear_pred
        _, scales = quantize(linear_resid)
        return Block(
            mode=Mode.LINEAR,
            start=0,
            length=L,
            macro_K=np.full(P, 2.0 + 0j),
            macro_G=np.full(P, 1.0 + 0j),
            micro_Ks=[],
            micro_Gs=[],
            residuals=linear_resid,
            seed1=seed1.copy(),
            seed2=seed2.copy(),
            scales=scales,
            signal_energy=signal_energy,
            residual_energy=linear_energy,
            pair_rms=_compute_pair_rms(linear_resid, P),
        )


# ── hierarchical decode ──────────────────────────────────────────────
#
# decoding is simulation: spin up the oscillators from the stored seeds
# and eigenvalues, generate the predicted trajectory, add the residuals.
# the decoder does not need the original signal. it reconstructs the
# trajectory from its dynamical parameters and irreducible surprises.


def decode_block(block: Block) -> np.ndarray:
    """simulate: macro + micro + residuals → reconstructed trajectory."""
    L, D = block.length, block.residuals.shape[1]

    if block.mode == Mode.RAW:
        return block.residuals[:L].copy()

    if block.mode == Mode.STATIC:
        return np.tile(block.seed1, (L, 1))

    # spin up the fundamental oscillators
    s1_c = _to_complex(block.seed1[np.newaxis])[0]
    s2_c = _to_complex(block.seed2[np.newaxis])[0]
    macro = _from_complex(predict_all(s1_c, s2_c, block.macro_K, block.macro_G, L))

    if block.mode == Mode.LINEAR or not block.micro_Ks:
        return macro + block.residuals

    # spin up the harmonic oscillators on each sub-block
    micro = np.zeros((L, D), dtype=np.float32)
    micro_size = block.micro_size or max(MIN_BLOCK, L // max(len(block.micro_Ks), 1))

    for si, (mK, mG) in enumerate(zip(block.micro_Ks, block.micro_Gs)):
        ss = si * micro_size
        se = min(ss + micro_size, L)
        sl = se - ss
        if sl < 1:
            continue

        # micro seeds at decode time: reconstructed from (micro_pred + residuals)
        # of prior sub-blocks. this is causal — no future information needed.
        if ss >= SEED_COUNT:
            ms2 = micro[ss - 2] + block.residuals[ss - 2]
            ms1 = micro[ss - 1] + block.residuals[ss - 1]
        elif ss == 1:
            ms2 = np.zeros(D, dtype=np.float32)
            ms1 = micro[0] + block.residuals[0]
        else:
            ms2 = ms1 = np.zeros(D, dtype=np.float32)

        ms1_c = _to_complex(ms1[np.newaxis])[0]
        ms2_c = _to_complex(ms2[np.newaxis])[0]
        micro[ss:se] = _from_complex(predict_all(ms1_c, ms2_c, mK, mG, sl))

    return macro + micro + block.residuals


# ── wire format ───────────────────────────────────────────────────────
#
# the wire format is the packet serialized to bytes. it's designed for
# the same pattern as Glyph: serialize blocks, then the full byte stream
# goes through Logos for entropy coding. the format is self-describing
# so the decoder can reconstruct without external metadata.


class StreamEncoder:
    """
    incremental encoder. feed samples one at a time, receive blocks
    when the buffer fills or a phase transition is detected.
    """

    def __init__(self, dim: int, micro_size: int = MIN_BLOCK):
        if not isinstance(dim, int) or not 2 <= dim <= 4096 or dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.micro_size = micro_size
        self._buf: List[np.ndarray] = []
        self._max_block = MIN_BLOCK  # will grow after first block
        self._threshold = float("inf")
        self._err_sum = 0.0
        self._err_sq_sum = 0.0
        self._err_count = 0
        self._block_idx = 0
        self._emitted_samples = 0

    def push(self, sample: np.ndarray) -> Optional[Block]:
        """
        push one [D] sample. returns a Block when a block boundary is reached
        (buffer full or phase transition detected), otherwise None.
        """
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape != (self.dim,) or not np.isfinite(sample).all():
            raise ValueError(
                "sample must be a finite vector with the configured dimension"
            )
        self._buf.append(sample.copy())
        T = len(self._buf)

        # not enough samples yet
        if T < SEED_COUNT + MIN_BLOCK:
            return None

        # derive block size from accumulated data once we have enough
        if self._block_idx == 0 and T == SEED_COUNT + MIN_BLOCK:
            w = np.stack(self._buf)
            self._max_block = min(MAX_BLOCK, max(MIN_BLOCK, _autocorrelation_length(w)))
            self.micro_size = max(MIN_BLOCK, self._max_block // MIN_BLOCK)

        # linear prediction error for this sample (computed once, used for
        # both threshold tracking and phase transition detection)
        err = 0.0
        if T >= SEED_COUNT + 2:
            pred = 2.0 * self._buf[-2] - self._buf[-3]
            err = float(np.linalg.norm(self._buf[-1] - pred))
            self._err_sum += err
            self._err_sq_sum += err * err
            self._err_count += 1

            if self._err_count >= SEED_COUNT + 1:
                mean = self._err_sum / self._err_count
                var = self._err_sq_sum / self._err_count - mean * mean
                self._threshold = mean + SIGMA_THRESHOLD * (
                    max(var, 0.0) ** 0.5 + MACHINE_EPS
                )

        block_len = T - SEED_COUNT
        should_emit = block_len >= self._max_block

        # phase transition: the error we already computed exceeds the threshold
        if not should_emit and block_len >= MIN_BLOCK and err > self._threshold:
            should_emit = True

        if should_emit:
            return self._emit()

        return None

    def flush(self) -> Optional[Block]:
        """flush remaining samples as a final block."""
        if self._emitted_samples == 0 and 0 < len(self._buf) <= SEED_COUNT:
            data = np.stack(self._buf)
            block = encode_block(data, data[-1], data[0], self.micro_size)
            self._emitted_samples = len(data)
            self._buf = []
            return block
        if len(self._buf) > SEED_COUNT:
            return self._emit()
        return None

    def _emit(self) -> Block:
        """encode the buffer into a block and reset."""
        w = np.stack(self._buf)
        seed2, seed1 = w[0], w[1]
        data = w[SEED_COUNT:]
        blk = encode_block(data, seed1, seed2, self.micro_size)
        blk.start = self._emitted_samples + SEED_COUNT

        # keep the last two samples as seeds for the next block
        self._buf = [self._buf[-2].copy(), self._buf[-1].copy()]
        self._block_idx += 1
        self._emitted_samples += len(data)
        return blk


class StreamDecoder:
    """
    incremental decoder. feed blocks, receive reconstructed samples.
    """

    def __init__(self):
        self._samples: List[np.ndarray] = []

    def push(self, block: Block) -> np.ndarray:
        """
        decode one block, return the reconstructed [L, D] chunk.
        also appends to internal state for seed continuity.
        """
        from dataclasses import replace
        from engram_wire import validate

        d = len(block.seed1)
        validate(
            Packet(
                d,
                d // 2,
                block.length,
                0.0,
                4,
                4,
                np.arange(d),
                [replace(block, start=0)],
            ),
            require_full=True,
        )
        if not self._samples and block.start == 2:
            self._samples.extend([block.seed2.copy(), block.seed1.copy()])
        if block.start != len(self._samples):
            raise ValueError("stream blocks must be contiguous and ordered")
        decoded = decode_block(block)
        self._samples.extend(decoded)
        return decoded

    @property
    def trajectory(self) -> np.ndarray:
        """all decoded samples so far."""
        if not self._samples:
            return np.empty((0,), dtype=np.float32)
        return np.stack(self._samples)


# ── packet merging ────────────────────────────────────────────────────
#
# two packets from adjacent trajectory segments can be joined.
# the second packet's blocks are re-indexed to follow the first.


def merge(
    a: Packet, b: Packet, *, overlap: int = 0, overlap_tolerance: float = 0.0
) -> Packet:
    """Compose full packets without dropping initial rows or changing their basis.

    overlap=2 explicitly removes two shared boundary samples after comparing
    their reconstructed values using the caller's absolute tolerance.
    """
    import copy
    from engram_wire import validate

    validate(a, require_full=True)
    validate(b, require_full=True)
    if a.dim != b.dim or not np.array_equal(a.pairing, b.pairing):
        raise ValueError("merged packets must share dimension and coordinate pairing")
    if (
        overlap not in (0, 2)
        or not math.isfinite(overlap_tolerance)
        or overlap_tolerance < 0
    ):
        raise ValueError(
            "overlap must be zero or two with a finite nonnegative tolerance"
        )
    if not a.length:
        return copy.deepcopy(b)
    if not b.length:
        return copy.deepcopy(a)
    codec = Engram(a.dim)
    if overlap:
        if min(a.length, b.length) < 2 or not np.allclose(
            codec.decode(a)[-2:], codec.decode(b)[:2], atol=overlap_tolerance, rtol=0
        ):
            raise ValueError("overlapping samples do not agree")
    result = copy.deepcopy(a)
    offset = a.length - overlap
    blocks = copy.deepcopy(b.blocks)
    first = blocks[0]
    if first.start == 2 and not overlap:
        prefix = np.stack([first.seed2, first.seed1])
        bridge = _raw_block(prefix, prefix[-1], prefix[0])
        bridge.start = offset
        result.blocks.append(bridge)
    if first.start == 0 and overlap:
        # Historical/raw packets can include their initial conditions in a block.
        # Preserve their reconstructed suffix exactly rather than reseeding that model.
        suffix = apply_pairing(codec.decode(b)[2:], b.pairing)
        blocks = [] if not len(suffix) else [_raw_block(suffix, suffix[0], suffix[0])]
        if blocks:
            blocks[0].start = 2
    for block in blocks:
        block.start += offset
        result.blocks.append(block)
    result.length = a.length + b.length - overlap
    result.macro_block = max(a.macro_block, b.macro_block)
    result.micro_block = min(a.micro_block, b.micro_block)
    validate(result, require_full=True)
    return result


# ── eigenvalue interpolation ──────────────────────────────────────────
#
# blend between two eigenvalue states in polar form: interpolate
# magnitude logarithmically, angle by shortest arc. this preserves the
# spectral structure that a linear average of K,G would distort.


def blend_eigenvalues(
    K_a: np.ndarray,
    G_a: np.ndarray,
    K_b: np.ndarray,
    G_b: np.ndarray,
    t: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    interpolate between two eigenvalue states.

    t=0 → (K_a, G_a).  t=1 → (K_b, G_b).  t=0.5 → spectral midpoint.
    operates on eigenvalues (λ₁, λ₂), not on K,G directly, so frequency
    and decay rate interpolate smoothly without distortion.
    """
    arrays = [np.asarray(a, dtype=np.complex128) for a in (K_a, G_a, K_b, G_b)]
    if (
        not math.isfinite(t)
        or not 0 <= t <= 1
        or any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays)
    ):
        raise ValueError(
            "finite matching coefficients and interpolation in [0,1] required"
        )
    K_a, G_a, K_b, G_b = arrays
    if t == 0 or (np.array_equal(K_a, K_b) and np.array_equal(G_a, G_b)):
        return K_a.copy(), G_a.copy()
    if t == 1:
        return K_b.copy(), G_b.copy()
    l1_a, l2_a = eigenvalues_all(K_a, G_a)
    l1_b, l2_b = eigenvalues_all(K_b, G_b)

    def _interp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ma, mb = np.abs(a), np.abs(b)
        positive = (ma > 0) & (mb > 0)
        mag = (1 - t) * ma + t * mb
        mag[positive] = np.exp(
            (1 - t) * np.log(ma[positive]) + t * np.log(mb[positive])
        )
        ang_a, ang_b = np.angle(a), np.angle(b)
        delta = ang_b - ang_a
        delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)
        delta = np.where(delta < -np.pi, delta + 2 * np.pi, delta)
        return mag * np.exp(1j * (ang_a + t * delta))

    l1_t = _interp(l1_a, l1_b)
    l2_t = _interp(l2_a, l2_b)

    # K = λ₁ + λ₂,  G = λ₁ · λ₂
    return (l1_t + l2_t).astype(np.complex128), (l1_t * l2_t).astype(np.complex128)


# ── the codec ─────────────────────────────────────────────────────────


def _validated_trajectory(trajectory, dim, pairing=None):
    w = _real_array(trajectory, dtype=np.float32)
    if (
        w.ndim != 2
        or w.shape[1] != dim
        or not np.isfinite(w).all()
        or w.size > 16 * 1024 * 1024
    ):
        raise ValueError("trajectory must be a bounded finite [T, dim] array")
    if pairing is not None:
        pairing = np.asarray(pairing)
        if (
            pairing.shape != (dim,)
            or not np.issubdtype(pairing.dtype, np.integer)
            or not np.array_equal(np.sort(pairing), np.arange(dim))
        ):
            raise ValueError("pairing must be a permutation of the dimensions")
        pairing = pairing.astype(np.int32)
    return w, pairing


class Engram:
    """
    the Whisper Engram universal trajectory codec.

    input:  [T, D] float32. any signal that moves through R^D over time.
    output: Packet (orbital parameters + quantized residuals).

    Block sizes adapt to the data under the reference numerical policies.
    Dimension is positive and even; max_error is an optional host-selected
    full-wire reconstruction budget.
    """

    def __init__(
        self, dim: int = 256, alpha: float = 0.0, max_error: Optional[float] = None
    ):
        if not isinstance(dim, int) or not 2 <= dim <= 4096 or dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.n_pairs = dim // 2
        self.alpha = alpha
        if not math.isfinite(alpha) or (
            max_error is not None and (not math.isfinite(max_error) or max_error < 0)
        ):
            raise ValueError(
                "finite alpha and nonnegative finite error budget required"
            )
        self.max_error = max_error

    def encode(
        self, trajectory: np.ndarray, pairing: Optional[np.ndarray] = None
    ) -> Packet:
        """
        encode a trajectory into orbital parameters + residuals.

        pairing=None: derives the natural complex pairing from inter-dimension
            correlation. each trajectory gets its own coordinate system.
        pairing=array: uses the provided pairing. all encodes with the same
            pairing produce K values in the same coordinate system, so
            centroids can be meaningfully compared across articles.
        """
        w, pairing = _validated_trajectory(trajectory, self.dim, pairing)
        T, D = w.shape
        assert D == self.dim, f"expected dim={self.dim}, got {D}"
        seq_pairing = np.arange(D, dtype=np.int32)

        if T == 0:
            basis = seq_pairing if pairing is None else pairing
            return Packet(
                D, D // 2, 0, self.alpha, MIN_BLOCK, MIN_BLOCK, basis.copy(), []
            )
        if T <= SEED_COUNT:
            basis = seq_pairing if pairing is None else pairing
            paired = apply_pairing(w, basis)
            block = _raw_block(paired, paired[-1], paired[0])
            return Packet(
                D, D // 2, T, self.alpha, MIN_BLOCK, MIN_BLOCK, basis.copy(), [block]
            )

        # pairing: which dimensions form each complex oscillator pair.
        # provided pairing ensures all encodes share one coordinate system.
        # derived pairing finds co-moving dimensions per-trajectory.
        if pairing is None:
            pairing = derive_pairing(w)
        wp = apply_pairing(w, pairing)

        macro_size, micro_size = derive_block_sizes(wp)
        segments = segment(wp, macro_size)
        blocks: List[Block] = []

        for ss, se in segments:
            data = wp[ss:se]
            blk = encode_block(data, wp[ss - 1].copy(), wp[ss - 2].copy(), micro_size)
            blk.start = ss
            blocks.append(blk)

        packet = Packet(
            D, D // 2, T, self.alpha, macro_size, micro_size, pairing, blocks
        )
        if self.max_error is not None:
            try:
                reconstructed = self.decode(from_wire(to_wire(packet)))
                exceeds = (
                    np.max(np.abs(reconstructed.astype(np.float64) - w))
                    > self.max_error
                )
            except FloatingPointError:
                exceeds = True
            if exceeds:
                paired = apply_pairing(w, pairing)
                signal_energy = float(np.sum(w.astype(np.float64) ** 2))
                block = Block(
                    Mode.RAW,
                    0,
                    T,
                    np.zeros(D // 2, complex),
                    np.zeros(D // 2, complex),
                    [],
                    [],
                    paired,
                    paired[min(1, T - 1)].copy(),
                    paired[0].copy(),
                    np.zeros(D // 2, dtype=np.float32),
                    signal_energy=signal_energy,
                    residual_energy=signal_energy,
                )
                packet.blocks = [block]
        return packet

    def decode(self, packet: Packet) -> np.ndarray:
        """simulate the dynamical system, then undo the pairing to restore
        original dimension order."""
        from engram_wire import validate

        validate(packet, require_full=True)
        T = packet.length
        if T == 0:
            return np.empty((0, packet.dim), dtype=np.float32)

        out = np.zeros((T, packet.dim), dtype=np.float32)

        for blk in packet.blocks:
            if blk.mode == Mode.RAW and blk.start == 0:
                out[: blk.length] = blk.residuals[: blk.length]
                continue
            decoded = decode_block(blk)
            out[blk.start : blk.start + blk.length] = decoded

        # restore seed samples (the initial conditions of the dynamical system)
        if packet.blocks and packet.blocks[0].start >= SEED_COUNT:
            out[0] = packet.blocks[0].seed2
            out[1] = packet.blocks[0].seed1

        # undo the pairing to restore original dimension order
        if not np.isfinite(out).all():
            raise FloatingPointError("reconstruction exceeds float32 range")
        return undo_pairing(out, packet.pairing)

    # ── dream: the meta-oscillator ──────────────────────────────────
    #
    # the codec compresses trajectories into eigenvalues.
    # dream observes how those eigenvalues evolve across blocks and
    # memories, fits oscillators to the oscillator parameters, and
    # runs forward. generation is dream pointed at the future.
    # no neural network. no training. no backprop.
    # just fit_pair called on K,G histories — the same algorithm.

    def dream(
        self,
        packets: List[Packet],
        n_blocks: int = 8,
        samples_per_block: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[np.ndarray]:
        """
        the third mode: generate a novel trajectory from accumulated memories.

        observes how eigenvalues AND residual scales evolve across all
        stored packets, fits meta-oscillators on both, runs them forward,
        and renders the result as a new trajectory with generated surprise.

        the orbit comes from the K,G meta-oscillator (the predictable part).
        the surprise comes from Laplacian noise at the learned residual
        scale (the unpredictable part, with the right magnitude grammar).
        without the surprise, the loop converges to silence.
        """
        # extract K,G and per-pair residual scale histories.
        # pair_rms is precomputed during encoding and stored in the block,
        # so dream works on both full and compact (brain) packets.
        Ks: List[np.ndarray] = []
        Gs: List[np.ndarray] = []
        Ss: List[np.ndarray] = []  # per-pair residual scales
        last_blk: Optional[Block] = None
        last_packet = None
        from engram_wire import validate

        if (
            not isinstance(n_blocks, int)
            or n_blocks < 1
            or (
                samples_per_block is not None
                and (not isinstance(samples_per_block, int) or samples_per_block < 1)
            )
        ):
            raise ValueError("positive generation sizes required")
        basis = None
        P = self.dim // 2
        for pkt in packets:
            validate(pkt)
            if pkt.dim != self.dim or not pkt.basis_known:
                raise ValueError(
                    "generation requires matching dimensions and known pairings"
                )
            if basis is None:
                basis = pkt.pairing
            elif not np.array_equal(basis, pkt.pairing):
                raise ValueError("generation packets must share one pairing")
            for blk in pkt.blocks:
                if blk.mode in (Mode.CASCADED, Mode.LINEAR):
                    Ks.append(blk.macro_K)
                    Gs.append(blk.macro_G)
                    if blk.pair_rms is not None:
                        Ss.append(blk.pair_rms.astype(np.float64) + 0j)
                    elif blk.residuals.size > 0:
                        # fallback: compute from residuals (full format packets)
                        r2 = blk.residuals.reshape(blk.residuals.shape[0], P, 2) ** 2
                        Ss.append(np.sqrt(np.mean(r2, axis=(0, 2))) + 0j)
                    else:
                        Ss.append(np.zeros(P, dtype=np.complex128))
                    last_blk = blk
                    last_packet = pkt
        if len(Ks) < SEED_COUNT + 1 or last_blk is None:
            return None

        K_hist = np.stack(Ks)
        G_hist = np.stack(Gs)
        S_hist = np.stack(Ss)
        P = K_hist.shape[1]

        # fit meta-oscillators: AR(2) on K, G, and residual scale
        meta_KK = np.empty(P, dtype=np.complex128)
        meta_GK = np.empty(P, dtype=np.complex128)
        meta_KG = np.empty(P, dtype=np.complex128)
        meta_GG = np.empty(P, dtype=np.complex128)
        meta_KS = np.empty(P, dtype=np.complex128)
        meta_GS = np.empty(P, dtype=np.complex128)
        for p in range(P):
            meta_KK[p], meta_GK[p], _ = fit_pair(K_hist[:, p])
            meta_KG[p], meta_GG[p], _ = fit_pair(G_hist[:, p])
            meta_KS[p], meta_GS[p], _ = fit_pair(S_hist[:, p])

        # evolve K, G, and residual scale forward
        gen_K = predict_all(K_hist[-1], K_hist[-2], meta_KK, meta_GK, n_blocks)
        gen_G = predict_all(G_hist[-1], G_hist[-2], meta_KG, meta_GG, n_blocks)
        gen_S = predict_all(S_hist[-1], S_hist[-2], meta_KS, meta_GS, n_blocks)

        # constrain generated K,G to the observed manifold.
        # two constraints, both derived from the data:
        #
        # 1. eigenvalue magnitudes ≤ 1 (sustain or decay, never explode)
        # 2. K magnitude stays within the observed range (the dream stays
        #    on the manifold instead of drifting into uncharted eigenvalue space)
        obs_K_scale = float(np.mean(np.abs(K_hist))) + MACHINE_EPS
        for bi in range(n_blocks):
            # eigenvalue clamping
            l1, l2 = eigenvalues_all(gen_K[bi], gen_G[bi])
            mag1, mag2 = np.abs(l1), np.abs(l2)
            scale1 = np.where(mag1 > 1.0, 1.0 / (mag1 + MACHINE_EPS), 1.0)
            scale2 = np.where(mag2 > 1.0, 1.0 / (mag2 + MACHINE_EPS), 1.0)
            l1_c, l2_c = l1 * scale1, l2 * scale2
            gen_K[bi] = l1_c + l2_c
            gen_G[bi] = l1_c * l2_c

            # magnitude clamping: keep K within escape energy of observed
            gen_scale = float(np.mean(np.abs(gen_K[bi]))) + MACHINE_EPS
            if gen_scale > obs_K_scale * ESCAPE_ENERGY:
                gen_K[bi] *= obs_K_scale / gen_scale
                gen_G[bi] *= (obs_K_scale / gen_scale) ** 2

        # clamp generated residual scales to the observed range.
        # Bound the sampled residual scale by observations. This does not
        # imply conservation or a bound on the recurrent output amplitude.
        observed_max_scale = float(np.abs(S_hist.real).max()) + MACHINE_EPS

        # render: oscillator prediction + Laplacian surprise at learned scale
        spb = (
            samples_per_block
            if samples_per_block is not None
            else max(MIN_BLOCK, packets[-1].macro_block)
        )
        if spb * n_blocks * self.dim > 16 * 1024 * 1024:
            raise ValueError("generated trajectory exceeds value budget")
        prev1 = _to_complex(last_blk.seed1[np.newaxis].astype(np.float32))[0]
        prev2 = _to_complex(last_blk.seed2[np.newaxis].astype(np.float32))[0]
        if last_packet.reconstructable:
            ending = apply_pairing(self.decode(last_packet)[-2:], basis)
            if len(ending) == 2:
                prev2, prev1 = _to_complex(ending)

        chunks: List[np.ndarray] = []
        for bi in range(n_blocks):
            # the orbit (predictable)
            chunk_c = predict_all(prev1, prev2, gen_K[bi], gen_G[bi], spb)
            chunk = _from_complex(chunk_c)

            # the surprise (unpredictable, but with learned magnitude).
            # sampled from Laplacian (peaked at zero, exponential tails)
            # because the residual anatomy showed that's the true shape.
            # clamped to the observed maximum: no dream is more surprising
            # than the most surprising thing that actually happened.
            pair_scales = np.clip(np.abs(gen_S[bi].real), 0, observed_max_scale)
            for p in range(P):
                scale = max(float(pair_scales[p]), MACHINE_EPS)
                d0, d1 = p * 2, p * 2 + 1
                noise = (
                    rng.laplace(0, scale, (spb, 2))
                    if rng is not None
                    else np.random.laplace(0, scale, (spb, 2))
                ).astype(np.float32)
                chunk[:, d0 : d1 + 1] += noise

            chunks.append(chunk)
            if spb >= 2:
                prev1, prev2 = (
                    _to_complex(chunk[np.newaxis, -1])[0],
                    _to_complex(chunk[np.newaxis, -2])[0],
                )
            elif spb == 1:
                prev2 = prev1
                prev1 = _to_complex(chunk[np.newaxis, 0])[0]

        return undo_pairing(np.concatenate(chunks, axis=0), basis) if chunks else None

    def stream_encoder(self) -> StreamEncoder:
        """create a streaming encoder for incremental encoding."""
        return StreamEncoder(self.dim)

    def stream_decoder(self) -> StreamDecoder:
        """create a streaming decoder for incremental decoding."""
        return StreamDecoder()


# ── perception-action loop ────────────────────────────────────────────
#
# the codec is the nervous system. dream is the brain stem.
# but a brain in a jar dreams of nothing. it needs a world to push
# against: sense, predict, observe, measure surprise, update.
#
# the Loop class wires this cycle. a World protocol provides the
# external signal. the loop handles the rest:
#
#     predict (dream) → act (send to world) → sense (receive from world)
#     → surprise (encode observation, measure residuals) → remember
#     → predict → ...
#
# the World is anything that takes a trajectory and returns a trajectory.
# a conversation partner. a sensor stream. a game. another loop.


class World(Protocol):
    """anything the loop can interact with."""

    def respond(self, action: np.ndarray) -> np.ndarray:
        """
        receive the loop's dream output [T, D] and return an
        observation [T', D] from the world. T' can differ from T.
        the observation is what the loop will encode and learn from.
        """
        ...


class Loop:
    """
    the perception-action cycle. wires a codec to a world.

    each step:
        1. predict: dream from memory (the expectation)
        2. act: send the dream to the world
        3. sense: receive the world's response
        4. surprise: encode the observation, measure residual energy
        5. remember: add the encoded observation to the pool

    the surprise signal (residual energy of encoding the observation)
    is the primary learning signal. high surprise = the world did
    something the oscillators didn't predict. low surprise = the world
    confirmed the current eigenvalue state.
    """

    def __init__(
        self,
        codec: Engram,
        world: World,
        seed: Optional[Packet] = None,
        *,
        rng=None,
        capacity=64,
    ):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("loop capacity must be positive")
        self.codec = codec
        self.world = world
        self.rng = rng
        self.capacity = capacity
        self.pairing = seed.pairing.copy() if seed is not None else None
        if seed is not None:
            from engram_wire import validate

            validate(seed)
            if seed.dim != codec.dim or not seed.basis_known:
                raise ValueError("seed needs the codec dimension and a known basis")
        self.pool: List[Packet] = [seed] if seed is not None else []
        self.step_count = 0
        self.history: List[dict] = []

    def seed(self, trajectory: np.ndarray) -> None:
        """provide initial experience before the loop starts."""
        packet = self.codec.encode(trajectory, pairing=self.pairing)
        self.pairing = packet.pairing.copy()
        self.pool.append(packet)
        del self.pool[: -self.capacity]

    def step(self) -> dict:
        """
        one cycle of predict → act → sense → surprise → remember.
        returns a record of what happened.
        """
        # predict: dream from memory
        prediction = None
        if len(self.pool) >= 1:
            prediction = self.codec.dream(
                self.pool,
                n_blocks=MIN_BLOCK,
                samples_per_block=MIN_BLOCK * AR_ORDER,
                rng=self.rng,
            )

        # act: send prediction to the world (or silence if no prediction yet)
        if prediction is None or prediction.shape[0] < SEED_COUNT:
            action = np.zeros((MIN_BLOCK * AR_ORDER, self.codec.dim), dtype=np.float32)
        else:
            action = prediction

        # sense: receive the world's response
        observation, _ = _validated_trajectory(
            self.world.respond(action), self.codec.dim
        )

        if observation.shape[0] < MIN_BLOCK:
            return {
                "step": self.step_count,
                "surprise": 0.0,
                "energy": 0.0,
                "alive": False,
            }

        # surprise: encode the observation, measure what the oscillators couldn't predict
        obs_packet = self.codec.encode(observation, pairing=self.pairing)
        self.pairing = obs_packet.pairing.copy()
        obs_summary = summary(obs_packet)
        surprise = 100.0 - obs_summary["capture"]
        obs_energy = float(np.sum(observation**2))

        # also measure prediction error: how far was the dream from the observation?
        prediction_error = 0.0
        if prediction is not None:
            overlap = min(prediction.shape[0], observation.shape[0])
            if overlap > 0:
                prediction_error = float(
                    np.sqrt(
                        np.mean((prediction[:overlap] - observation[:overlap]) ** 2)
                    )
                )

        # remember: add the observation to the pool
        self.pool.append(obs_packet)
        del self.pool[: -self.capacity]

        record = {
            "step": self.step_count,
            "surprise": surprise,
            "prediction_error": prediction_error,
            "obs_energy": obs_energy,
            "obs_capture": obs_summary["capture"],
            "obs_block": obs_summary["macro_block"],
            "pool_size": len(self.pool),
            "alive": True,
        }
        self.history.append(record)
        self.step_count += 1
        return record

    def run(self, n_steps: int) -> List[dict]:
        """run the loop for n steps. returns history."""
        if not isinstance(n_steps, int) or n_steps < 0:
            raise ValueError("n_steps must be nonnegative")
        records = []
        for _ in range(n_steps):
            r = self.step()
            records.append(r)
            if not r["alive"]:
                break
        return records

    @property
    def surprise_trajectory(self) -> np.ndarray:
        """the surprise signal over time. the loop's emotional arc."""
        return np.array([h["surprise"] for h in self.history], dtype=np.float32)

    @property
    def prediction_error_trajectory(self) -> np.ndarray:
        """how wrong the dream was. the gap between expectation and reality."""
        return np.array([h["prediction_error"] for h in self.history], dtype=np.float32)


# ── built-in worlds ──────────────────────────────────────────────────
#
# simple worlds for testing. real worlds would be sensor streams,
# conversation partners, game environments, other loops.


class EchoWorld:
    """returns the action back with added noise. the simplest world."""

    def __init__(self, noise_scale: float = 0.1, *, rng=None):
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        self.noise_scale = noise_scale
        self.rng = rng

    def respond(self, action: np.ndarray) -> np.ndarray:
        return (
            action
            + (
                np.random.randn(*action.shape)
                if self.rng is None
                else self.rng.normal(size=action.shape)
            ).astype(np.float32)
            * self.noise_scale
        )


class MirrorWorld:
    """returns the action reversed in time. the world reflects."""

    def respond(self, action: np.ndarray) -> np.ndarray:
        return action[::-1].copy()


class PartnerLoop:
    """another loop as the world. mutual interaction."""

    def __init__(self, codec: Engram, *, rng=None, capacity=64):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("partner capacity must be positive")
        self.codec = codec
        self.rng, self.capacity = rng, capacity
        self.pairing = None
        self.pool: List[Packet] = []

    def seed(self, trajectory: np.ndarray) -> None:
        packet = self.codec.encode(trajectory, pairing=self.pairing)
        self.pairing = packet.pairing.copy()
        self.pool.append(packet)
        del self.pool[: -self.capacity]

    def respond(self, action: np.ndarray) -> np.ndarray:
        action, _ = _validated_trajectory(action, self.codec.dim)
        # encode the partner's action as our observation
        if action.shape[0] >= MIN_BLOCK:
            self.seed(action)
        # dream our response
        if len(self.pool) < 1:
            return action  # echo until we have memories
        response = self.codec.dream(
            self.pool,
            n_blocks=MIN_BLOCK,
            samples_per_block=MIN_BLOCK * AR_ORDER,
            rng=self.rng,
        )
        if response is None or response.shape[0] < SEED_COUNT:
            return action
        # remember our own dream
        self.seed(response)
        return response


class TrajectoryFeed:
    """feeds pre-computed trajectories one at a time. a curriculum."""

    def __init__(self, trajectories: List[np.ndarray]):
        self._feed = list(trajectories)
        self._idx = 0

    def respond(self, action: np.ndarray) -> np.ndarray:
        if self._idx < len(self._feed):
            obs = self._feed[self._idx]
            self._idx += 1
            return obs
        return action  # echo when feed is exhausted


# ── engram file (brain format) ─────────────────────────────────────────
#
# the brain stores sufficient statistics, not raw packets:
#
#     manifest.json    identity, physics, well index
#     wells.bin        per-well (sum_K, count) — the knowledge
#     dream.bin        circular buffer of K,G,S histories — the imagination
#     meta.bin         crystallized meta-oscillator (optional)
#
# packets are absorbed, statistics extracted, packet discarded.
# brain size is constant: ~5MB at dim=300 regardless of article count.
# wells emerge from the data. any new source tag creates a new well.


class EngramFile:
    """
    the .engram brain format. sufficient statistics, not raw packets.

    the brain stores what it learned, not what it saw. wells are running
    means of eigenvalue centroids per source tag. the dream buffer is a
    circular history of recent K,G,S blocks for dreaming. packets are
    absorbed, their statistics extracted, and then discarded.

    this means brain size is proportional to the number of distinct
    domains, not the number of articles. 25 domains = ~5MB brain
    whether you've absorbed 500 articles or 5 million.

    wells emerge from the data. any new source tag creates a new well.
    the brain doesn't know in advance what domains exist.

    the .engram file is a ZIP archive containing:
        manifest.json       identity, physics, well index
        wells.bin           per-well (sum_K, count) sufficient statistics
        dream.bin           circular buffer of recent K,G,S histories
        meta.bin            crystallized meta-oscillator (optional)
    """

    # dream buffer capacity scales with dimensionality.
    # the meta-oscillator needs SEED_COUNT+1 samples per pair to fit.
    # P = dim/2 pairs, each needs ≥ 3 samples for AR(2).
    # capacity = dim * multiplier gives (multiplier * 2) samples per pair.
    # at dim=300: capacity=1200, P=150, 8 samples/pair.
    DREAM_BUFFER_CAPACITY_PER_DIM = MIN_BLOCK  # 4: gives 2*MIN_BLOCK samples/pair

    def __init__(self, directory: str):
        self.path = Path(directory)
        self._save_path_available = bool(directory)

        # identity: the label on the jar. editable by the user.
        # name and born are first-class. everything else goes in the
        # metadata dict: pronouns, description, author, version.
        # only things a human would CHOOSE belong here. things a human
        # would FEEL (mood, curiosity, confusion) come from the geometry.
        self.name: str = ""
        self.born: str = ""
        self.metadata: dict = {}

        # annotations: the voice. freeform JSON. could be a one-line
        # personality note or a full system card. the brain doesn't
        # process this. the model that speaks for the brain does.
        # the app reads annotations and passes them as context to
        # whatever inference engine gives the brain its mouth.
        # any key, any structure, any depth. not our schema.
        self.annotations: dict = {}

        self.dim: int = 0
        self.alpha: float = 0.0

        # sufficient statistics: per-domain wells in TWO spaces.
        # K-space: oscillator parameters. used for well distances and dreaming.
        # raw-space: byte-histogram centroids. used for neuroplastic embedding.
        # when the brain measures new text, the raw-space distances are appended
        # to the trajectory. the oscillator sees the bytes AND the brain's
        # perspective on those bytes. as the brain lives more, the perspective
        # sharpens, the distances become more discriminative, and the K-space
        # wells refine with genuinely new information. this is neuroplasticity:
        # the embedding evolves with the brain's understanding.
        self._well_sum_K: dict[
            str, np.ndarray
        ] = {}  # K-space running sum (complex128[P])
        # G-space running sum: second AR(2) coefficient per well.
        # with K AND G we can compute actual eigenvalues from
        # λ² - Kλ + G = 0, meaning each well carries the full AR(2)
        # dynamical-system parameters — not just a centroid proxy.
        # this is what lets each well SIMULATE (spin up its own
        # oscillator), not just cluster.
        self._well_sum_G: dict[
            str, np.ndarray
        ] = {}  # G-space running sum (complex128[P])
        # per-pair residual RMS sum per well. gives the noise floor of
        # each mode — what's the typical surprise magnitude of
        # observations that routed here. useful for per-well dream
        # prediction uncertainty bands.
        self._well_sum_S: dict[
            str, np.ndarray
        ] = {}  # residual RMS running sum (float64[P])
        self._well_sum_raw: dict[
            str, np.ndarray
        ] = {}  # raw byte-histogram running sum (256d)
        self._well_count: dict[str, int] = {}  # observation count (shared)

        # ── fixed routing anchors (format v7) ──────────────────────
        # the running centroid (sum_K/count) is what each well "is" in
        # an aggregate sense: average of every K it absorbed. but for
        # ROUTING purposes that aggregate is the wrong thing — long-
        # corpus runs collapse it to the population mean by LLN, so a
        # single well ends up at the centroid of all data and absorbs
        # every subsequent article (it's by definition closest to the
        # cluster's center). routing anchors solve that by FREEZING
        # each well's routing position to the K, G, rms of the article
        # that spawned it. wells partition K-space by their bootstrap
        # positions; the running centroid is still tracked for
        # analysis, dream, and meta-osc, but isn't the routing target.
        self._well_anchor_K: dict[str, np.ndarray] = {}  # complex128[P]
        self._well_anchor_G: dict[str, np.ndarray] = {}  # complex128[P]
        self._well_anchor_S: dict[str, np.ndarray] = {}  # float64[P]

        # ── graph + per-well meta (format v6) ──────────────────────
        # transitions[A][B] = number of times an article routed to B
        # immediately after an article routed to A. turns the wells
        # from an isolated bag of attractors into a directed graph
        # of concepts, enabling concept-level prediction.
        self._well_transitions: dict[str, dict[str, int]] = {}

        # per-well rolling dream buffer: list of K-centroids from
        # articles that routed to THIS well. complements the global
        # dream buffer, which tracks the trajectory of ALL articles
        # regardless of destination. each well's buffer lets the
        # well fit its OWN meta-oscillator (meta_K, meta_G below),
        # so every well can predict its next incoming article's K
        # based on its own history — not the brain-wide average.
        self._well_dream_Ks: dict[str, list[np.ndarray]] = {}

        # per-well fitted meta-oscillator — AR(2) params over the
        # well's own dream trajectory. populated by finalize() or
        # lazily on first per-well prediction. None if the well's
        # dream history has fewer than SEED_COUNT+1 entries.
        self._well_meta_K: dict[str, np.ndarray] = {}
        self._well_meta_G: dict[str, np.ndarray] = {}

        # cap on per-well dream buffer length. memory scales as
        # n_wells × cap × P × 16 bytes (complex128). at 16 × AR_ORDER
        # = 32 entries, a 10k-well brain at P=150 uses ~750 MB just for
        # this — pushing python's footprint into territory where GC
        # pauses become visible. 4 × AR_ORDER = 8 entries gives ~190 MB
        # at 10k wells, which is fine, and still satisfies the per-well
        # meta-oscillator fit minimum (SEED_COUNT + 1 = 3). the per-well
        # AR(2) fit doesn't reliably beat the centroid baseline on
        # byte-text data anyway (see project_engram_dream_wells.md),
        # so giving it more history isn't earning its memory.
        self.WELL_DREAM_CAP = 4 * AR_ORDER
        self._total_absorbed: int = 0
        self._next_well_id: int = 0
        self._reference_pairing: Optional[np.ndarray] = None
        self._well_cache: Optional[dict] = None
        self._gc_cache: Optional[np.ndarray] = None
        self._nn_cache = None
        self._nn_names: List[str] = []
        self._nn_stack: Optional[np.ndarray] = None
        self._nn_mass: Optional[np.ndarray] = None

        # typical_gap caching — median pairwise nearest-neighbor
        # distance between established wells. invalidated only when
        # the established-well set size changes (a new well crosses
        # the WELL_ESTABLISHED threshold or a brand-new well appears).
        # the value drifts slowly enough with running-average centroid
        # motion that within a stable established set it's essentially
        # constant over thousands of observations.
        self._gap_cache: Optional[float] = None
        self._gap_cache_n_est: int = -1
        self._gap_median_count_cache: Optional[float] = None

        # dream buffer: circular history of recent K,G,S blocks
        self._dream_Ks: List[np.ndarray] = []
        self._dream_Gs: List[np.ndarray] = []
        self._dream_Ss: List[np.ndarray] = []  # pair_rms values

        # persisted meta-oscillator (crystallized from dream buffer)
        self.dream_meta_K: Optional[np.ndarray] = None
        self.dream_meta_G: Optional[np.ndarray] = None

    @staticmethod
    def create(
        directory: str,
        name: str = "",
        dim: int = 256,
        alpha: float = 0.005,
        metadata: Optional[dict] = None,
        annotations: Optional[dict] = None,
    ) -> "EngramFile":
        """
        create a new empty brain.

        name: what the brain is called. just a label.
        metadata: freeform identity dict. things the user CHOOSES:
            pronouns, description, author, version. not things the
            brain FEELS (mood, curiosity). those come from geometry.
        annotations: freeform voice dict. personality, system card, tone,
            behavioral notes. the speaking model reads this, not the brain.
        """
        Engram(dim=dim, alpha=alpha)
        if not math.isfinite(alpha):
            raise ValueError("memory alpha must be finite")
        import datetime

        ef = EngramFile(directory)
        ef.name = name
        ef.born = datetime.datetime.utcnow().isoformat() + "Z"
        ef.dim = dim
        ef.alpha = alpha
        if metadata:
            ef.metadata = metadata
        if annotations:
            ef.annotations = annotations
        return ef

    def absorb(self, packet: Packet, source: Optional[str] = None, **_kwargs) -> str:
        """Absorb predictive block statistics through the shared K/G/RMS update path."""
        from engram_wire import validate

        validate(packet)
        if not packet.basis_known:
            raise ValueError("attach the original coordinate basis before absorption")
        if packet.dim != self.dim:
            raise ValueError("packet dimension differs from memory")
        if self._reference_pairing is None:
            self._reference_pairing = packet.pairing.copy()
        elif not np.array_equal(packet.pairing, self._reference_pairing):
            raise ValueError("packet pairing differs from the memory coordinate basis")
        blocks = [b for b in packet.blocks if b.mode in (Mode.CASCADED, Mode.LINEAR)]
        p = self.dim // 2
        if not blocks:
            return self.absorb_flat(
                np.empty((0, p), complex),
                np.empty((0, p), complex),
                np.empty((0, p)),
                source,
            )
        return self.absorb_flat(
            np.stack([b.macro_K for b in blocks]),
            np.stack([b.macro_G for b in blocks]),
            np.stack(
                [b.pair_rms if b.pair_rms is not None else np.zeros(p) for b in blocks]
            ),
            source,
        )

    def absorb_flat(
        self,
        K_stack: np.ndarray,  # [n_blocks, P] complex (any precision)
        G_stack: np.ndarray,  # [n_blocks, P] complex
        rms_stack: np.ndarray,  # [n_blocks, P] float
        source: Optional[str] = None,
    ) -> str:
        """fast path for batched ingest: accepts already-stacked block
        features directly, no Block/Packet objects needed.

        semantically identical to absorb() for the codec's memory state
        (_well_sum_K, _well_count, _dream_Ks/Gs/Ss, _total_absorbed).
        the `packet.blocks` iteration is replaced by operating on the
        caller's numpy arrays, eliminating 500-to-16k Python dataclass
        constructions per batch that showed up as the dominant
        remaining cost after all other optimizations.
        """
        P = self.dim // 2
        K_stack = np.asarray(K_stack, dtype=np.complex128)
        G_stack = np.asarray(G_stack, dtype=np.complex128)
        rms_stack = np.asarray(rms_stack, dtype=np.float64)
        if (
            K_stack.ndim != 2
            or K_stack.shape[1] != P
            or G_stack.shape != K_stack.shape
            or rms_stack.shape != K_stack.shape
        ):
            raise ValueError("K/G/RMS stacks must share shape [blocks, pairs]")
        if not all(
            np.isfinite(a).all() for a in (K_stack, G_stack, rms_stack)
        ) or np.any(rms_stack < 0):
            raise ValueError("finite K/G and nonnegative RMS statistics required")
        if source is not None and (
            not isinstance(source, str) or len(source.encode("utf-8")) > 65535
        ):
            raise ValueError("invalid source label")

        if K_stack.shape[0] and self._reference_pairing is None:
            self._reference_pairing = np.arange(self.dim, dtype=np.int32)

        # empty article fast-out
        if K_stack is None or K_stack.shape[0] == 0:
            self._total_absorbed += 1
            return source or "unknown"

        # promote precision to match the codec's internal state. the
        # cost of this cast is ~O(n_blocks * P) and vectorized, much
        # cheaper than per-block Block() construction.
        if K_stack.dtype != np.complex128:
            K_stack = K_stack.astype(np.complex128)
        if G_stack.dtype != np.complex128:
            G_stack = G_stack.astype(np.complex128)
        if rms_stack.dtype != np.float64:
            rms_stack = rms_stack.astype(np.float64)

        obs_centroid = K_stack.mean(axis=0)
        n = int(K_stack.shape[0])
        # full routing feature: K, G, rms means. matches GPU Brain.
        obs_G_centroid = (
            G_stack.mean(axis=0)
            if G_stack is not None and G_stack.shape[0] > 0
            else None
        )
        obs_rms_centroid = (
            rms_stack.mean(axis=0)
            if rms_stack is not None and rms_stack.shape[0] > 0
            else None
        )

        # well routing — same logic as absorb(), centralized in the
        # _route_observation helper (vectorized + cached).
        well_name = self._route_observation(
            obs_centroid, source, obs_G=obs_G_centroid, obs_rms=obs_rms_centroid
        )

        # Accumulate all provided statistics; older imported wells may lack fields.
        is_new_well = well_name not in self._well_sum_K
        if is_new_well:
            self._well_sum_K[well_name] = np.zeros(P, dtype=np.complex128)
            self._well_count[well_name] = 0
        if well_name not in self._well_sum_G:
            self._well_sum_G[well_name] = np.zeros(P, dtype=np.complex128)
        if well_name not in self._well_sum_S:
            self._well_sum_S[well_name] = np.zeros(P, dtype=np.float64)
        if is_new_well:
            # pin routing anchor to THIS spawning observation's K/G/S
            # centroids, fixed for the well's lifetime. mirrors the
            # source GPU construction. Cross-backend equivalence requires
            # separate measured evidence, including its floating precision.
            self._well_anchor_K[well_name] = obs_centroid.astype(np.complex128).copy()
            if G_stack is not None and G_stack.shape[0] > 0:
                self._well_anchor_G[well_name] = (
                    G_stack.mean(axis=0).astype(np.complex128).copy()
                )
            else:
                self._well_anchor_G[well_name] = np.zeros(P, dtype=np.complex128)
            if rms_stack is not None and rms_stack.shape[0] > 0:
                self._well_anchor_S[well_name] = (
                    rms_stack.mean(axis=0).astype(np.float64).copy()
                )
            else:
                self._well_anchor_S[well_name] = np.zeros(P, dtype=np.float64)
        self._well_sum_K[well_name] += K_stack.sum(axis=0)
        if G_stack is not None:
            self._well_sum_G[well_name] += G_stack.sum(axis=0)
        if rms_stack is not None:
            self._well_sum_S[well_name] += rms_stack.sum(axis=0)
        self._well_count[well_name] += n

        # dream buffer — per-article centroids
        self._dream_Ks.append(obs_centroid)
        self._dream_Gs.append(G_stack.mean(axis=0))
        self._dream_Ss.append(rms_stack.mean(axis=0))

        # trim circular buffer
        cap = self.dim * self.DREAM_BUFFER_CAPACITY_PER_DIM
        if len(self._dream_Ks) > cap:
            excess = len(self._dream_Ks) - cap
            self._dream_Ks = self._dream_Ks[excess:]
            self._dream_Gs = self._dream_Gs[excess:]
            self._dream_Ss = self._dream_Ss[excess:]

        previous = getattr(self, "_last_well", None)
        if previous is not None:
            edges = self._well_transitions.setdefault(previous, {})
            edges[well_name] = edges.get(well_name, 0) + 1
        self._last_well = well_name
        history = self._well_dream_Ks.setdefault(well_name, [])
        history.append(obs_centroid.copy())
        del history[: -self.WELL_DREAM_CAP]
        self._well_meta_K.pop(well_name, None)
        self._well_meta_G.pop(well_name, None)
        self.dream_meta_K = self.dream_meta_G = None
        self._total_absorbed += 1
        self._invalidate_caches()
        return well_name

    def absorb_batch_flat(
        self, K_all, G_all, rms_all, boundaries, source=None
    ) -> list[str]:
        """Ordered batch updates with exactly the same routing/state as individual absorbs."""
        k, g, r = np.asarray(K_all), np.asarray(G_all), np.asarray(rms_all)
        bounds = np.asarray(boundaries)
        if (
            k.ndim != 2
            or k.shape != g.shape
            or k.shape != r.shape
            or bounds.ndim != 1
            or len(bounds) < 1
            or not np.issubdtype(bounds.dtype, np.integer)
        ):
            raise ValueError("invalid feature batch")
        if bounds[0] != 0 or bounds[-1] != len(k) or np.any(np.diff(bounds) < 0):
            raise ValueError("batch boundaries must monotonically partition all rows")
        if (
            k.shape[1] != self.dim // 2
            or not all(np.isfinite(a).all() for a in (k, g, r))
            or np.any(r < 0)
        ):
            raise ValueError("invalid feature values")
        return [
            self.absorb_flat(k[a:b], g[a:b], r[a:b], source)
            for a, b in zip(bounds[:-1], bounds[1:])
        ]

    # ── wells: per-domain eigenvalue centroids ─────────────────

    def wells(self) -> dict[str, np.ndarray]:
        """
        per-domain eigenvalue centroids. cached, O(1) per call.
        invalidated on absorb.
        """
        if self._well_cache is None:
            self._well_cache = {
                src: self._well_sum_K[src] / max(self._well_count[src], 1)
                for src in self._well_sum_K
            }
        return self._well_cache

    def global_centroid(self) -> Optional[np.ndarray]:
        """mass-weighted eigenvalue centroid across all wells.

        each well contributes proportional to its observation count,
        so the global centroid reflects where the brain's total mass is,
        not just the geometric center of well locations.
        """
        if self._gc_cache is not None:
            return self._gc_cache
        if not self._well_sum_K:
            return None
        total_sum = sum(self._well_sum_K.values())
        total_count = sum(self._well_count.values())
        if total_count < 1:
            return None
        self._gc_cache = total_sum / total_count
        return self._gc_cache

    def _invalidate_caches(self):
        """call after any absorb to force recomputation.

        the gap cache is NOT invalidated here — it's governed by its
        own staleness counter (see _route_observation). the gap value
        drifts only with running-average centroid motion over thousands
        of samples, so per-absorb invalidation is wasted work.
        """
        self._well_cache = None
        self._gc_cache = None
        self._nn_cache = None
        self._nn_mass = None

    @staticmethod
    def _compute_typical_gap(est_stack: np.ndarray) -> float:
        """median pairwise nearest-neighbor distance between established
        wells, computed in a single BLAS matmul instead of the O(W²)
        Python nested loop.

        for complex centroids a, b ∈ ℂ^P:
            ‖a − b‖² = ‖a‖² + ‖b‖² − 2·Re(a·b̄)
        the whole W×W matrix comes from one conjugate-transpose matmul.
        """
        W, P = est_stack.shape
        if W < 2:
            return 1.0  # caller won't use this path
        norms = (est_stack.real**2 + est_stack.imag**2).sum(axis=1)  # [W] real
        inner = est_stack @ est_stack.conj().T  # [W, W] complex
        d2 = (norms[:, None] + norms[None, :] - 2.0 * inner.real) / float(P)
        np.fill_diagonal(d2, np.inf)
        d2 = np.maximum(d2, 0.0)  # clamp fp drift below zero
        nn_dists = np.sqrt(d2.min(axis=1))
        return float(np.median(nn_dists))

    def _route_observation(
        self,
        obs_centroid: np.ndarray,
        source: Optional[str],
        obs_G: Optional[np.ndarray] = None,
        obs_rms: Optional[np.ndarray] = None,
    ) -> str:
        """decide which well absorbs this observation. replaces three
        identical ~25-line copies in absorb / absorb_flat /
        absorb_batch_flat, each of which paid an O(W²·P) Python loop
        to recompute typical_gap from scratch.

        caching: typical_gap and median_count over established wells
        are computed once, reused for GAP_CACHE_STALENESS subsequent
        absorbs, and invalidated when the number of established wells
        changes (which happens only when a new well crosses the
        WELL_ESTABLISHED threshold or a brand-new well appears).
        """
        if source is not None:
            return source
        # bootstrap: until BOOTSTRAP_WELLS distinct anchored wells
        # exist, spawn a new well for every observation. mirrors the
        # GPU Brain's bootstrap path so CPU absorb_flat and GPU absorb
        # produce the same well set on identical inputs.
        if len(self._well_sum_K) < BOOTSTRAP_WELLS:
            name = f"well_{self._next_well_id}"
            self._next_well_id += 1
            return name

        nearest, dist = self.nearest_well(obs_centroid, obs_G=obs_G, obs_rms=obs_rms)
        w = self.wells()
        established = {
            k: v for k, v in w.items() if self._well_count.get(k, 0) >= WELL_ESTABLISHED
        }
        n_est = len(established)
        if n_est < 2:
            return nearest

        # refresh gap cache when the established-well set size changes.
        # otherwise the value drifts only with centroid motion, which
        # is bounded by the running-average denominator — stable
        # enough that re-verification every batch is pure waste.
        if self._gap_cache is None or self._gap_cache_n_est != n_est:
            est_names = list(established.keys())
            est_stack = np.stack([established[n] for n in est_names])
            self._gap_cache = self._compute_typical_gap(est_stack)
            self._gap_median_count_cache = float(
                np.median([self._well_count[n] for n in est_names])
            )
            self._gap_cache_n_est = n_est

        typical_gap = self._gap_cache
        median_count = self._gap_median_count_cache
        nearest_count = self._well_count.get(nearest, 1)
        # match the GPU route shader: cap the depth ratio at 1.0 so
        # mature wells get the BASE escape mult, not runaway stickiness.
        ratio = min(np.log1p(nearest_count) / max(np.log1p(median_count), 1e-20), 1.0)
        escape_mult = ESCAPE_ENERGY * np.sqrt(ratio)

        if dist > typical_gap * escape_mult:
            name = f"well_{self._next_well_id}"
            self._next_well_id += 1
            return name
        return nearest

    def nearest_well(
        self,
        obs_centroid: np.ndarray,
        obs_G: Optional[np.ndarray] = None,
        obs_rms: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """find the well closest to an observation in routing-feature
        space.

        ROUTING USES FIXED ANCHORS, NOT RUNNING CENTROIDS. anchors are
        pinned at well creation; they never drift to the population
        mean as the well absorbs more articles. this prevents the
        "mega-well at centroid of all data" failure mode where running
        centroids drift to the cluster's geometric center and one well
        absorbs everything.

        when obs_G and obs_rms are provided, distance is computed over
        the full (K, G, rms) feature space — matching the GPU Brain's
        route shader exactly. when only obs_centroid is provided
        (legacy callers), falls back to K-only distance. wells loaded
        from format ≤ v6 (no anchor) fall back to their running
        centroid as the routing position. no mass weighting.
        """
        if not self._well_sum_K:
            return "unknown", 50.0

        if self._nn_cache is None:
            self._nn_names = list(self._well_sum_K.keys())
            K_anchors: list = []
            G_anchors: list = []
            S_anchors: list = []
            P = self.dim // 2
            for n in self._nn_names:
                anc_K = self._well_anchor_K.get(n)
                if anc_K is None:
                    cnt = max(int(self._well_count.get(n, 1)), 1)
                    anc_K = self._well_sum_K[n] / cnt
                K_anchors.append(anc_K)
                anc_G = self._well_anchor_G.get(n)
                if anc_G is None:
                    cnt = max(int(self._well_count.get(n, 1)), 1)
                    G_sum = self._well_sum_G.get(n)
                    anc_G = (
                        G_sum / cnt
                        if G_sum is not None
                        else np.zeros(P, dtype=np.complex128)
                    )
                G_anchors.append(anc_G)
                anc_S = self._well_anchor_S.get(n)
                if anc_S is None:
                    cnt = max(int(self._well_count.get(n, 1)), 1)
                    S_sum = self._well_sum_S.get(n)
                    anc_S = (
                        S_sum / cnt
                        if S_sum is not None
                        else np.zeros(P, dtype=np.float64)
                    )
                S_anchors.append(anc_S)
            self._nn_stack = np.stack(K_anchors)  # [N, P] complex
            self._nn_G_stack = np.stack(G_anchors)  # [N, P] complex
            self._nn_S_stack = np.stack(S_anchors)  # [N, P] real
            self._nn_cache = True

        # distance: K diff (re²+im²) per pair, plus G diff and rms diff
        # when provided. mean over P pairs, then sqrt for RMS distance.
        # matches the GPU route shader's per-pair sum, divided by P.
        K_diffs = self._nn_stack - obs_centroid[np.newaxis, :]
        K_d2_per_pair = K_diffs.real**2 + K_diffs.imag**2
        sum_sq = K_d2_per_pair.sum(axis=1)
        if obs_G is not None:
            G_diffs = self._nn_G_stack - obs_G[np.newaxis, :]
            sum_sq = sum_sq + (G_diffs.real**2 + G_diffs.imag**2).sum(axis=1)
        if obs_rms is not None:
            S_diffs = self._nn_S_stack - obs_rms[np.newaxis, :]
            sum_sq = sum_sq + (S_diffs**2).sum(axis=1)
        P = self.dim // 2
        raw_dists = np.sqrt(sum_sq.real / P)
        idx = int(np.argmin(raw_dists))
        return self._nn_names[idx], float(raw_dists[idx])

    # ── measure: separated from routing ────────────────────────

    def _raw_well_centroids(self) -> dict[str, np.ndarray]:
        """raw byte-histogram space well centroids."""
        return {
            src: self._well_sum_raw[src] / max(self._well_count.get(src, 1), 1)
            for src in self._well_sum_raw
            if src in self._well_count
        }

    def _augment_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """
        augment a byte histogram trajectory with raw-space well distances.

        for each window in the trajectory, compute its distance to every
        raw-space well centroid. append these distances as extra dimensions.
        the oscillator then sees the bytes AND the brain's perspective on
        those bytes.

        as the brain absorbs more, the raw centroids sharpen, the distances
        become more discriminative, and the augmented trajectory carries
        more semantic signal. this is neuroplasticity: the embedding
        evolves with the brain's understanding.

        if no wells exist yet, returns the trajectory unchanged.
        """
        raw_wells = self._raw_well_centroids()
        if not raw_wells:
            return traj

        well_names = sorted(raw_wells.keys())
        n_wells = len(well_names)
        centroids = np.stack([raw_wells[w] for w in well_names])  # [n_wells, 256]

        # compute per-window distance to each well centroid
        # traj is [T, 256], centroids is [n_wells, 256]
        # distances [T, n_wells]: RMS distance per window per well
        T = traj.shape[0]
        dists = np.empty((T, n_wells), dtype=np.float32)
        for i, c in enumerate(centroids):
            diff = traj - c[np.newaxis, :]  # [T, 256]
            dists[:, i] = np.sqrt(np.mean(diff**2, axis=1))

        # normalize distances to [0, 1] range
        d_max = dists.max()
        if d_max > 0:
            dists /= d_max

        # pad to even total dim if needed
        aug_dim = 256 + n_wells
        if aug_dim % 2 == 1:
            pad = np.zeros((T, 1), dtype=np.float32)
            return np.concatenate([traj, dists, pad], axis=1)
        return np.concatenate([traj, dists], axis=1)

    def measure_text(self, text: str, codec: Optional["Engram"] = None) -> dict:
        """Measure fixed 256-bin byte histograms in a persistent coordinate basis.

        Learned-coordinate refinement is explicit in engram_training; silently
        appending changing well distances would invalidate existing statistics.
        """
        if self.dim != 256:
            raise ValueError("text histogram memory needs dimension 256")
        if not hasattr(self, "_byte_hist"):
            self._byte_hist = ByteHistogram(chunk_size=512)

        traj = self._byte_hist.text_to_trajectory(text)
        if traj.shape[0] < MIN_BLOCK:
            return {"drift": 0.0, "texture": 0.0, "nearest_well": "none", "energy": 0.0}

        # store the raw trajectory mean for updating raw wells later
        self._last_raw_mean = np.mean(traj, axis=0)

        if codec is None:
            if not hasattr(self, "_codec") or self._codec.dim != 256:
                self._codec = Engram(dim=256)
            codec = self._codec

        return self.measure(codec, traj)

    def absorb_text(
        self, text: str, source: Optional[str] = None, codec: Optional["Engram"] = None
    ) -> str:
        """
        measure and absorb raw text in one call.

        also updates the raw-space well centroids so the neuroplastic
        embedding improves for future measurements.
        """
        m = self.measure_text(text, codec)

        # update raw-space well centroid
        well_name = None
        if "packet" in m:
            before = self._well_count.copy()
            well_name = self.absorb(m["packet"], source=source)
            added = self._well_count.get(well_name, 0) - before.get(well_name, 0)
            if added > 0:
                if well_name not in self._well_sum_raw:
                    self._well_sum_raw[well_name] = np.zeros(256, dtype=np.float64)
                self._well_sum_raw[well_name] += (
                    self._last_raw_mean.astype(np.float64) * added
                )

        return well_name or "unknown"

    def fast_absorb(self, trajectory: np.ndarray, source: Optional[str] = None) -> str:
        """Fit one shared-basis signature, without block segmentation or reconstruction."""
        w, _ = _validated_trajectory(trajectory, self.dim)
        if len(w) < 2:
            self._total_absorbed += 1
            return source or "unknown"
        if self._reference_pairing is None:
            self._reference_pairing = derive_pairing(w)
        signature = Signature.from_trajectory(w, pairing=self._reference_pairing)
        return self.absorb_flat(
            signature.k[None, :],
            signature.g[None, :],
            signature.pair_rms[None, :],
            source,
        )

    def measure(self, codec: "Engram", observation: np.ndarray) -> dict:
        """
        measure an observation's surprise in two bands without routing it.

        drift: eigenvalue centroid distance from the nearest well.
        texture: raw residual energy from encoding.
        nearest_well: which domain this observation is closest to.
        """
        if codec.dim != self.dim:
            raise ValueError(
                "codec and memory dimensions differ; supply a matching embedding"
            )
        if observation is not None:
            observation, _ = _validated_trajectory(observation, self.dim)
        if observation is None or observation.shape[0] < MIN_BLOCK:
            return {"drift": 0.0, "texture": 0.0, "nearest_well": "none", "energy": 0.0}

        # derive the reference pairing from the first trajectory the brain sees.
        # all subsequent encodes use this pairing, ensuring K values are in a
        # consistent coordinate system so well centroids are meaningful.
        if self._reference_pairing is None:
            self._reference_pairing = derive_pairing(observation.astype(np.float32))

        obs_pkt = codec.encode(observation, pairing=self._reference_pairing)
        s = summary(obs_pkt)
        energy = float(np.sum(observation**2))
        texture = 100.0 - s["capture"]

        obs_Ks = [
            blk.macro_K
            for blk in obs_pkt.blocks
            if blk.mode in (Mode.CASCADED, Mode.LINEAR)
        ]

        if not obs_Ks:
            return {
                "drift": 50.0,
                "texture": texture,
                "nearest_well": "none",
                "energy": energy,
                "capture": s["capture"],
                "block_size": s["macro_block"],
                "packet": obs_pkt,
            }

        obs_centroid = np.mean(np.stack(obs_Ks), axis=0)

        well_name, well_dist = self.nearest_well(obs_centroid)
        gc = self.global_centroid()
        if gc is not None:
            scale = float(np.sqrt(np.mean(np.abs(gc) ** 2))) + MACHINE_EPS
            drift = min(100.0, well_dist / scale * 100.0)
        else:
            scale = 1.0
            drift = 50.0

        # ── dream prior: what did the brain expect to see? ────────
        #
        # the meta-oscillator predicts the next eigenvalue centroid from
        # the trajectory of all previous centroids. the delta between
        # prediction and observation is dream surprise: the brain's own
        # measure of "i didn't see this coming."
        dream_surprise = 0.0
        dream_state = "forming"
        predicted_K = None

        # lazy crystallization: fit the meta-oscillator if we have
        # enough history but haven't fitted yet
        if self.dream_meta_K is None and len(self._dream_Ks) >= SEED_COUNT + 1:
            self.crystallize_dream(codec)

        if self.dream_meta_K is not None and len(self._dream_Ks) >= SEED_COUNT:
            s1 = self._dream_Ks[-1]
            s2 = self._dream_Ks[-2]
            raw_pred = predict_all(
                s1, s2, self.dream_meta_K, self.dream_meta_G, length=1
            )[0]
            # magnitude guard: if the meta-oscillator diverges, skip.
            # compare against the brain's own scale (global centroid),
            # not the observation — topic shifts can produce legitimately
            # different observation magnitudes.
            pred_mag = float(np.sqrt(np.mean(np.abs(raw_pred) ** 2).real))
            brain_mag = (
                float(np.sqrt(np.mean(np.abs(gc) ** 2).real)) if gc is not None else 1.0
            )
            if pred_mag < SIGMA_THRESHOLD * (brain_mag + MACHINE_EPS):
                predicted_K = raw_pred
                dream_delta = float(
                    np.sqrt(np.mean(np.abs(predicted_K - obs_centroid) ** 2).real)
                )
                dream_surprise = min(100.0, dream_delta / scale * 100.0)
                dream_state = "crystallized"
            else:
                dream_state = "diverged"

        # well-distance profile: vectorized, only established wells.
        w = self.wells()
        est_names = [n for n in w if self._well_count.get(n, 0) >= WELL_ESTABLISHED]
        well_profile = {}
        well_attention = {}
        if est_names:
            est_stack = np.stack([w[n] for n in est_names])  # [M, P]
            diffs = est_stack - obs_centroid[np.newaxis, :]
            dists = np.sqrt(np.mean(np.abs(diffs) ** 2, axis=1).real)
            well_profile = {
                n: round(float(d) / float(scale) * 100, 2)
                for n, d in zip(est_names, dists.tolist())
            }

            # ── attention: dream-surprise per well ────────────────
            #
            # wells the dream predicted nearby = expected = cold.
            # wells the dream thought were far but observation is near = hot.
            # attention[i] = dream_dist[i] / obs_dist[i].
            # >1 = surprising (the dream didn't predict we'd be here).
            # <1 = expected (the dream saw this coming).
            if predicted_K is not None:
                dream_diffs = est_stack - predicted_K[np.newaxis, :]
                dream_dists = np.sqrt(np.mean(np.abs(dream_diffs) ** 2, axis=1).real)
                for i, n in enumerate(est_names):
                    obs_d = float(dists[i]) + MACHINE_EPS
                    drm_d = float(dream_dists[i])
                    well_attention[n] = round(drm_d / obs_d, 3)

        return {
            "drift": round(float(drift), 2),
            "texture": round(float(texture), 2),
            "energy": round(float(energy), 2),
            "capture": float(s["capture"]),
            "block_size": int(s["macro_block"]),
            "nearest_well": str(well_name),
            "well_distance": round(float(well_dist), 6),
            "well_profile": well_profile,
            "well_attention": well_attention,
            "dream_surprise": round(float(dream_surprise), 2),
            "dream_state": dream_state,
            "packet": obs_pkt,
        }

    # ── transition graph + per-well meta API (format v6) ──────────

    def predict_next_well(self, current_well: str, k: int = 5) -> list:
        """predict the most likely next wells given the current one.

        returns list[(well_name, probability)] sorted by probability
        descending, up to k entries. probabilities are counts
        normalized by the total outgoing transitions from
        current_well. empty list if current_well has no recorded
        outgoing transitions.
        """
        out = self._well_transitions.get(current_well)
        if not out:
            return []
        total = sum(out.values())
        if total <= 0:
            return []
        ranked = sorted(out.items(), key=lambda kv: -kv[1])[:k]
        return [(w, c / total) for w, c in ranked]

    def transition_entropy(self, well: str) -> float:
        """Shannon entropy of outgoing transitions from `well`, in bits.

        a well that always routes to the same next well has entropy
        0 (fully predictable). a well that scatters uniformly to
        many targets has max entropy log2(n_targets). quantifies
        the well's PREDICTABILITY as a concept.
        """
        out = self._well_transitions.get(well)
        if not out:
            return 0.0
        total = sum(out.values())
        if total <= 0:
            return 0.0
        h = 0.0
        for c in out.values():
            if c <= 0:
                continue
            p = c / total
            h -= p * math.log2(p)
        return float(h)

    def top_transitions(self, limit: int = 10) -> list:
        """top edges in the graph by count. returns list of
        (src, dst, count) sorted descending. useful for a quick
        global view of "which transitions dominate the brain."
        """
        edges = []
        for a, targets in self._well_transitions.items():
            for b, c in targets.items():
                edges.append((a, b, int(c)))
        edges.sort(key=lambda t: -t[2])
        return edges[:limit]

    def per_well_meta(self, well: str):
        """fitted meta-oscillator (K, G) for this well, or None if
        the well doesn't have enough dream history (< SEED_COUNT+1
        entries) to fit. populated eagerly by finalize() or lazily
        on demand.
        """
        if well not in self._well_count:
            raise KeyError(well)
        history = self._well_dream_Ks.get(well, [])
        if well not in self._well_meta_K and len(history) >= SEED_COUNT + 1:
            k, g, _, _ = fit_all(np.stack(history))
            self._well_meta_K[well], self._well_meta_G[well] = k, g
        mK = self._well_meta_K.get(well)
        mG = self._well_meta_G.get(well)
        if mK is None or mG is None:
            return None
        return (mK, mG)

    def per_well_dream_predict(self, well: str) -> Optional[np.ndarray]:
        """predict the next K-centroid for this well using its
        OWN meta-oscillator fit to its OWN dream buffer. returns
        None if the well doesn't have a fitted meta-osc. unlike
        the brain-wide dream_meta, this prediction is local to the
        well — it's what this specific attractor expects to see
        next, not what the brain on average expects.
        """
        meta = self.per_well_meta(well)
        if meta is None:
            return None
        buf = self._well_dream_Ks.get(well)
        if not buf or len(buf) < SEED_COUNT:
            return None
        mK, mG = meta
        s1 = buf[-1]
        s2 = buf[-2]
        pred = predict_all(s1, s2, mK, mG, length=1)[0]
        return pred

    def measure_from_sums(
        self,
        K_sums: np.ndarray,  # [n_art, P] complex
        G_sums: np.ndarray,  # [n_art, P] complex
        rms_sums: np.ndarray,  # [n_art, P] float
        n_segs_per_art: np.ndarray,  # [n_art] int
    ) -> List[dict]:
        """batched measure() for the GPU unified-pipeline output.

        takes the per-article K/G/rms sums that per_article_reduce.wgsl
        already produces and returns the same bi-processor signal that
        single-item measure() returns — nearest_well, drift, well_profile,
        well_attention, dream_surprise, dream_state — but for a whole
        batch in one shot. no codec needed because the K centroids are
        already the observation's fingerprint; no raw trajectory
        available in this path means texture/capture/block_size/energy
        from the full encode are not emitted (they require the packet
        and aren't computable from sums alone).

        semantics match measure(): the returned dicts reflect the
        brain's state BEFORE these observations are absorbed, so call
        this before absorb_from_sums. dream_state is shared across the
        batch (the meta-oscillator at time-of-measurement is global).
        """
        K_sums, G_sums, rms_sums = (np.asarray(a) for a in (K_sums, G_sums, rms_sums))
        counts = np.asarray(n_segs_per_art)
        if (
            K_sums.ndim != 2
            or K_sums.shape[1] != self.dim // 2
            or G_sums.shape != K_sums.shape
            or rms_sums.shape != K_sums.shape
            or counts.shape != (K_sums.shape[0],)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts <= 0)
            or not all(np.isfinite(a).all() for a in [K_sums, G_sums, rms_sums])
            or np.any(rms_sums < 0)
        ):
            raise ValueError(
                "finite matching K/G/RMS sums and positive segment counts required"
            )
        n_art = int(K_sums.shape[0])
        if n_art == 0:
            return []

        # lazy crystallize the meta-oscillator (matches measure()'s
        # behavior). crystallize_dream takes a codec arg but doesn't
        # use it; inline the fit to avoid constructing a codec here.
        if self.dream_meta_K is None and len(self._dream_Ks) >= SEED_COUNT + 1:
            K_hist = np.stack(self._dream_Ks)
            Ph = K_hist.shape[1]
            self.dream_meta_K = np.empty(Ph, dtype=np.complex128)
            self.dream_meta_G = np.empty(Ph, dtype=np.complex128)
            for p in range(Ph):
                self.dream_meta_K[p], self.dream_meta_G[p], _ = fit_pair(K_hist[:, p])

        # per-article centroid = K_sum / n_segs
        n_safe = np.maximum(np.asarray(n_segs_per_art, dtype=np.float64), 1.0)
        centroids = (
            np.asarray(K_sums, dtype=np.complex128) / n_safe[:, None]
        )  # [n_art, P]

        gc = self.global_centroid()
        if gc is not None:
            scale = float(np.sqrt(np.mean(np.abs(gc) ** 2).real)) + MACHINE_EPS
        else:
            scale = 1.0

        # dream prediction — one shared prediction (state is global)
        predicted_K: Optional[np.ndarray] = None
        dream_state = "forming"
        if self.dream_meta_K is not None and len(self._dream_Ks) >= SEED_COUNT:
            s1 = self._dream_Ks[-1]
            s2 = self._dream_Ks[-2]
            raw_pred = predict_all(
                s1, s2, self.dream_meta_K, self.dream_meta_G, length=1
            )[0]
            pred_mag = float(np.sqrt(np.mean(np.abs(raw_pred) ** 2).real))
            brain_mag = (
                float(np.sqrt(np.mean(np.abs(gc) ** 2).real)) if gc is not None else 1.0
            )
            if pred_mag < SIGMA_THRESHOLD * (brain_mag + MACHINE_EPS):
                predicted_K = raw_pred
                dream_state = "crystallized"
            else:
                dream_state = "diverged"

        # vectorized dream_surprise across the batch
        if predicted_K is not None:
            dream_delta = np.sqrt(
                np.mean(np.abs(centroids - predicted_K[None, :]) ** 2, axis=1).real
            )  # [n_art]
            dream_surprises = np.minimum(100.0, dream_delta / scale * 100.0)
        else:
            dream_surprises = np.zeros(n_art, dtype=np.float64)

        # established-well stack for profile/attention (shared across batch)
        w = self.wells()
        est_names = [n for n in w if self._well_count.get(n, 0) >= WELL_ESTABLISHED]
        est_stack: Optional[np.ndarray] = None
        est_sigmas: Optional[np.ndarray] = None
        dream_dists_per_well: Optional[np.ndarray] = None
        if est_names:
            est_stack = np.stack([w[n] for n in est_names])  # [M, P]
            # per-well noise floor σ_w = sqrt(mean(S_w/count_w)) across
            # the P pairs. this is the well's own scale — the typical
            # residual magnitude of observations that routed there.
            # observations that are an actual signal for this well
            # sit within σ_w of its centroid; observations beyond are
            # genuinely surprising FROM THIS WELL'S PERSPECTIVE, even
            # if they'd route elsewhere.
            sigmas = np.empty(len(est_names), dtype=np.float64)
            for i, n in enumerate(est_names):
                cnt = max(int(self._well_count.get(n, 0)), 1)
                S_sum = self._well_sum_S.get(n)
                if S_sum is None or np.abs(S_sum).sum() < MACHINE_EPS:
                    # v4-loaded or pre-G-population well: no per-well
                    # noise info. use the brain's global scale as
                    # fallback so the per-well surprise is at least
                    # computable (just not locally adapted).
                    sigmas[i] = scale
                else:
                    sigmas[i] = max(
                        float(np.sqrt(np.mean(np.abs(S_sum) / cnt))),
                        MACHINE_EPS,
                    )
            est_sigmas = sigmas
            if predicted_K is not None:
                dream_diffs = predicted_K[None, :] - est_stack  # [M, P]
                dream_dists_per_well = np.sqrt(
                    np.mean(np.abs(dream_diffs) ** 2, axis=1).real
                )  # [M]

        results: List[dict] = []
        for a in range(n_art):
            obs_c = centroids[a]
            well_name, well_dist = self.nearest_well(obs_c)
            drift = (
                min(100.0, well_dist / scale * 100.0) if scale > MACHINE_EPS else 50.0
            )

            profile: dict = {}
            attention: dict = {}
            per_well_surprise: dict = {}
            if est_stack is not None:
                diffs = est_stack - obs_c[None, :]  # [M, P]
                dists = np.sqrt(np.mean(np.abs(diffs) ** 2, axis=1).real)  # [M]
                for i, n in enumerate(est_names):
                    profile[n] = round(float(dists[i]) / scale * 100.0, 2)
                if dream_dists_per_well is not None:
                    for i, n in enumerate(est_names):
                        obs_d = float(dists[i]) + MACHINE_EPS
                        attention[n] = round(float(dream_dists_per_well[i]) / obs_d, 3)
                # per-well surprise: each well asks "how surprising is
                # this observation relative to MY noise floor?" unlike
                # global dream_surprise (one meta-oscillator prediction
                # vs the observation), this gives a per-well view. the
                # routed well should always have a relatively low
                # per_well_surprise; other wells should have higher.
                if est_sigmas is not None:
                    for i, n in enumerate(est_names):
                        per_well_surprise[n] = round(
                            float(dists[i]) / float(est_sigmas[i]), 3
                        )

            results.append(
                {
                    "nearest_well": str(well_name),
                    "well_distance": round(float(well_dist), 6),
                    "drift": round(float(drift), 2),
                    "well_profile": profile,
                    "well_attention": attention,
                    "per_well_surprise": per_well_surprise,
                    "dream_surprise": round(float(dream_surprises[a]), 2),
                    "dream_state": dream_state,
                }
            )
        return results

    # ── dream ──────────────────────────────────────────────────

    def crystallize_dream(self, codec: "Engram") -> None:
        """fit the meta-oscillator from the dream buffer."""
        if codec.dim != self.dim:
            raise ValueError("codec dimension differs from memory")
        for well in self._well_count:
            self.per_well_meta(well)
        if len(self._dream_Ks) < SEED_COUNT + 1:
            return
        K_hist = np.stack(self._dream_Ks)
        P = K_hist.shape[1]
        self.dream_meta_K = np.empty(P, dtype=np.complex128)
        self.dream_meta_G = np.empty(P, dtype=np.complex128)
        for p in range(P):
            self.dream_meta_K[p], self.dream_meta_G[p], _ = fit_pair(K_hist[:, p])

    @property
    def dream_packets(self) -> List[Packet]:
        """
        synthesize lightweight packets from the dream buffer for Engram.dream().
        these are not stored packets — they're views into the buffer.
        """
        if not self._dream_Ks:
            return []
        P = len(self._dream_Ks[0])
        D = P * 2
        blocks = []
        for index, (K, G, S) in enumerate(
            zip(self._dream_Ks, self._dream_Gs, self._dream_Ss)
        ):
            blocks.append(
                Block(
                    mode=Mode.CASCADED,
                    start=index,
                    length=1,
                    macro_K=K,
                    macro_G=G,
                    micro_Ks=[],
                    micro_Gs=[],
                    residuals=np.empty((0, D), dtype=np.float32),
                    seed1=np.zeros(D, dtype=np.float32),
                    seed2=np.zeros(D, dtype=np.float32),
                    scales=np.zeros(P, dtype=np.float32),
                    pair_rms=S.astype(np.float32),
                )
            )
        # wrap all blocks in one packet
        pkt = Packet(
            dim=D,
            n_pairs=P,
            length=len(blocks),
            alpha=self.alpha,
            macro_block=1,
            micro_block=1,
            pairing=self._reference_pairing.copy()
            if self._reference_pairing is not None
            else np.arange(D, dtype=np.int32),
            blocks=blocks,
            reconstructable=False,
        )
        return [pkt]

    def fingerprint(self) -> np.ndarray:
        """spectral identity from dream buffer eigenvalue distributions."""
        # magnitude bins: concentrated around the stability boundary |λ|=1.
        # below 1: decaying oscillators (dying). above 1: divergent (cut by segmenter).
        # the narrow bands at 0.99-1.01 capture the sustained oscillators.
        mag_edges = np.array(
            [0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.5, 1e6]
        )
        # frequency bins: [0, π] radians per sample. low freq = slow drift,
        # high freq = fast oscillation. π is the Nyquist limit.
        freq_edges = np.array(
            [0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, np.pi + MACHINE_EPS]
        )
        all_mags, all_freqs = [], []
        for K, G in zip(self._dream_Ks, self._dream_Gs):
            l1, l2 = eigenvalues_all(K, G)
            all_mags.extend(np.abs(l1).tolist())
            all_mags.extend(np.abs(l2).tolist())
            all_freqs.extend(np.abs(np.angle(l1)).tolist())
            all_freqs.extend(np.abs(np.angle(l2)).tolist())
        if not all_mags:
            return np.zeros(22, dtype=np.float32)
        mh, _ = np.histogram(all_mags, bins=mag_edges)
        fh, _ = np.histogram(all_freqs, bins=freq_edges)
        stats = np.array(
            [
                float(self._total_absorbed),
                float(len(self._well_sum_K)),
                float(len(self._dream_Ks)),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                (mh / max(mh.sum(), 1)).astype(np.float32),
                (fh / max(fh.sum(), 1)).astype(np.float32),
                stats,
            ]
        )

    # ── persistence ───────────────────────────────────────────

    def save(self, filepath: Optional[str] = None) -> str:
        """
        write brain to a single .engram file.

        the brain stores sufficient statistics, not packets:
            wells.bin      per-well (sum_K, count) — the knowledge
            dream.bin      K,G,S circular buffer — the imagination
            meta.bin       crystallized meta-oscillator (optional)
            manifest.json  identity, physics, well index
        """
        import zipfile, io

        if not filepath and not self._save_path_available:
            raise ValueError("a memory imported from bytes needs an explicit save path")
        from engram_memory_io import validate_memory

        validate_memory(self)

        out_path = filepath or str(self.path)
        if not out_path.endswith(".engram"):
            out_path += ".engram"

        P = self.dim // 2

        # wells binary: [reference_pairing][n_wells][per well: name_len, name_utf8, sum_K, count]
        wells_buf = io.BytesIO()
        # reference pairing: the shared coordinate system for all encodes
        if self._reference_pairing is not None:
            wells_buf.write(self._reference_pairing.astype("<i2").tobytes())
        else:
            wells_buf.write(np.arange(self.dim, dtype=np.int16).tobytes())
        well_names = sorted(self._well_sum_K.keys())
        wells_buf.write(struct.pack("<I", len(well_names)))
        # v5 layout per well:
        #   name_len(u16), name_utf8,
        #   sum_K(complex128[P]), sum_G(complex128[P]), sum_S(float64[P]),
        #   sum_raw(float64[256]), count(u32)
        # v4 layout was sum_K, sum_raw, count (missing G and S). load()
        # detects version via manifest["version"] and parses accordingly.
        zeros_G = np.zeros(P, dtype=np.complex128)
        zeros_S = np.zeros(P, dtype=np.float64)
        # v7 layout per well (added 3 anchor blocks at the end):
        #   name_len(u16), name_utf8,
        #   sum_K(complex128[P]), sum_G(complex128[P]), sum_S(float64[P]),
        #   sum_raw(float64[256]), count(u32),
        #   anchor_K(complex128[P]), anchor_G(complex128[P]),
        #   anchor_S(float64[P])
        # if a well has no anchor (loaded from v6 brain and never re-
        # absorbed), we write the running centroid as the anchor — same
        # behavior as the v6→v7 fallback in _pack_well_row.
        for name in well_names:
            name_bytes = name.encode("utf-8")
            wells_buf.write(struct.pack("<H", len(name_bytes)))
            wells_buf.write(name_bytes)
            wells_buf.write(self._well_sum_K[name].astype("<c16").tobytes())
            sum_G = self._well_sum_G.get(name, zeros_G)
            wells_buf.write(sum_G.astype("<c16").tobytes())
            sum_S = self._well_sum_S.get(name, zeros_S)
            wells_buf.write(sum_S.astype("<f8").tobytes())
            raw = self._well_sum_raw.get(name, np.zeros(256, dtype=np.float64))
            wells_buf.write(raw.astype("<f8").tobytes())
            wells_buf.write(struct.pack("<I", self._well_count[name]))
            # anchor blocks. fall back to running centroid for v6
            # wells that never got re-absorbed under v7.
            cnt = max(int(self._well_count[name]), 1)
            anchor_K = self._well_anchor_K.get(
                name, self._well_sum_K[name].astype(np.complex128) / cnt
            )
            anchor_G = self._well_anchor_G.get(name, sum_G.astype(np.complex128) / cnt)
            anchor_S = self._well_anchor_S.get(name, sum_S.astype(np.float64) / cnt)
            wells_buf.write(anchor_K.astype("<c16").tobytes())
            wells_buf.write(anchor_G.astype("<c16").tobytes())
            wells_buf.write(anchor_S.astype("<f8").tobytes())

        # dream buffer binary: [n_blocks][per block: K, G, S as float32]
        dream_buf = io.BytesIO()
        n_dream = len(self._dream_Ks)
        dream_buf.write(struct.pack("<I", n_dream))
        for K, G, S in zip(self._dream_Ks, self._dream_Gs, self._dream_Ss):
            dream_buf.write(K.astype("<c16").tobytes())  # P * 16 bytes
            dream_buf.write(G.astype("<c16").tobytes())  # P * 16 bytes
            dream_buf.write(S.astype("<f8").tobytes())  # P * 8 bytes

        # meta-oscillator
        meta_data = None
        if self.dream_meta_K is not None:
            meta_buf = io.BytesIO()
            meta_buf.write(self.dream_meta_K.astype("<c16").tobytes())
            meta_buf.write(self.dream_meta_G.astype("<c16").tobytes())
            meta_data = meta_buf.getvalue()

        # ── v6 additions: transitions graph + per-well dream state ──
        # transitions: sparse dict[str, dict[str, int]]. zips small
        # enough to live in its own JSON file. serialize as a nested
        # mapping since well names are strings — more readable than
        # packing binary and comparable in size.
        transitions_data = None
        if self._well_transitions:
            transitions_data = json.dumps(
                {a: dict(bs) for a, bs in self._well_transitions.items()},
                separators=(",", ":"),
            )

        # per-well dream buffers + meta oscillators — binary, one file.
        # layout:
        #   u32 n_wells
        #   per well: u16 name_len, utf-8 name,
        #             u32 n_dream_entries,
        #             complex128[n_dream_entries, P]   (K buffer),
        #             u8 has_meta (1 = has meta_K/meta_G, 0 = no),
        #             complex128[P] meta_K (if has_meta),
        #             complex128[P] meta_G (if has_meta)
        well_dreams_buf = io.BytesIO()
        well_dream_names = sorted(self._well_dream_Ks.keys())
        well_dreams_buf.write(struct.pack("<I", len(well_dream_names)))
        for name in well_dream_names:
            name_bytes = name.encode("utf-8")
            buf = self._well_dream_Ks.get(name, [])
            well_dreams_buf.write(struct.pack("<H", len(name_bytes)))
            well_dreams_buf.write(name_bytes)
            well_dreams_buf.write(struct.pack("<I", len(buf)))
            for K in buf:
                well_dreams_buf.write(np.asarray(K, dtype=np.complex128).tobytes())
            has_meta = name in self._well_meta_K and name in self._well_meta_G
            well_dreams_buf.write(struct.pack("<B", 1 if has_meta else 0))
            if has_meta:
                well_dreams_buf.write(self._well_meta_K[name].astype("<c16").tobytes())
                well_dreams_buf.write(self._well_meta_G[name].astype("<c16").tobytes())
        well_dreams_data = well_dreams_buf.getvalue()

        manifest = {
            "format": "engram",
            # v7: per-well fixed routing anchors. solves the mega-well
            # collapse where running centroids drift to the population
            # mean over long ingests. v6 brains load with anchors
            # synthesized from running centroids on next absorb.
            "version": 8,
            "identity": {
                **(self.metadata if self.metadata else {}),
                "name": self.name,
                "born": self.born,
            },
            "physics": {
                "dim": self.dim,
                "pairs": P,
                "alpha": self.alpha,
            },
            "state": {
                "total_absorbed": self._total_absorbed,
                "next_well_id": self._next_well_id,
                "last_well": getattr(self, "_last_well", None),
                "pairing_initialized": self._reference_pairing is not None,
                "n_wells": len(well_names),
                "dream_buffer_size": n_dream,
                "dream_buffer_capacity": self.dim * self.DREAM_BUFFER_CAPACITY_PER_DIM,
                "has_meta": meta_data is not None,
            },
            "wells": {name: {"count": self._well_count[name]} for name in well_names},
        }

        # Preserve forward-compatible metadata and sections while replacing owned state.
        import copy

        original = copy.deepcopy(getattr(self, "_source_manifest", {}))
        for key in ("identity", "physics", "state"):
            manifest[key] = {**original.get(key, {}), **manifest[key]}
        manifest = {**original, **manifest}
        if getattr(self, "import_info", None):
            manifest["origin"] = self.import_info

        # atomic write: build the zip under a sibling temp name, then
        # os.replace() onto the target. on POSIX and NT this is an
        # atomic rename — power-loss / SIGKILL mid-write leaves either
        # the old file intact or the new file complete, never a
        # truncated half-written .engram.
        import time as _time

        tmp_path = f"{out_path}.tmp-{os.getpid()}-{int(_time.time_ns())}"
        # compression strategy: float binaries compress ~5-10% with
        # deflate but take 1-3 seconds of CPU per save at W=8k+ — that
        # blocks the absorb thread long enough to backpressure-crash
        # the GPU pipeline. STORED for the binaries (the bulk of the
        # data) keeps save under 500ms; DEFLATED for the small JSON
        # files where compression actually helps.
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
                zf.writestr(
                    "manifest.json",
                    json.dumps(manifest, indent=2),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
                zf.writestr("wells.bin", wells_buf.getvalue())
                zf.writestr("dream.bin", dream_buf.getvalue())
                if transitions_data is not None:
                    zf.writestr(
                        "transitions.json",
                        transitions_data,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                if well_dream_names:
                    zf.writestr("well_dreams.bin", well_dreams_data)
                if meta_data is not None:
                    zf.writestr("meta.bin", meta_data)
                if self.annotations:
                    zf.writestr(
                        "annotations.json",
                        json.dumps(self.annotations, indent=2),
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                for section, payload in getattr(self, "_extra_sections", {}).items():
                    if section not in {
                        "manifest.json",
                        "wells.bin",
                        "dream.bin",
                        "annotations.json",
                        "meta.bin",
                        "transitions.json",
                        "well_dreams.bin",
                    }:
                        zf.writestr(section, payload)
            os.replace(tmp_path, out_path)
        except BaseException:
            # cleanup partial temp file if anything went wrong
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

        return out_path

    @staticmethod
    def load(filepath) -> "EngramFile":
        """Load a bounded ZIP v4-v8, ENBR v1 or ENDB v2 memory snapshot."""
        from engram_memory_io import load_memory

        return load_memory(filepath)

    def copy(self) -> "EngramFile":
        """Independent state for experiments without mutating the original memory."""
        import copy

        return copy.deepcopy(self)


# ── diagnostics ───────────────────────────────────────────────────────
#
# the summary reports how much of the signal the oscillators captured
# (the physics) and how much is left in the residuals (the surprise).


def summary(packet: Packet) -> dict:
    modes = {m.name: 0 for m in Mode}
    resid_sq = count = 0.0
    macro_cap = micro_cap = 0.0
    n = 0

    for blk in packet.blocks:
        modes[blk.mode.name] += 1
        resid_sq += float(np.sum(blk.residuals**2))
        count += blk.residuals.size
        macro_cap += blk.macro_capture
        micro_cap += blk.micro_capture
        n += 1

    return {
        "tokens": packet.length,
        "dim": packet.dim,
        "blocks": len(packet.blocks),
        "macro_block": packet.macro_block,
        "micro_block": packet.micro_block,
        "modes": modes,
        "residual_rms": round(float(np.sqrt(resid_sq / max(count, 1))), 6),
        "macro_capture": round(macro_cap / max(n, 1), 1),
        "micro_capture": round(micro_cap / max(n, 1), 1),
        "capture": round(packet.capture, 1),
        "snr_db": round(packet.snr_db, 1) if packet.snr_db != float("inf") else "inf",
    }


def entropy(data: bytes) -> float:
    """Shannon entropy H0 in bits/byte. 8.0 = random. 0.0 = constant."""
    if not data:
        return 0.0
    c = np.zeros(256, dtype=np.int64)
    for b in data:
        c[b] += 1
    p = c[c > 0] / len(data)
    return float(-np.sum(p * np.log2(p)))


# ── self-test ─────────────────────────────────────────────────────────

from engram_wire import to_wire, from_wire
from engram_analysis import (
    Signature,
    Dynamics as Dynamics,
    dynamics as dynamics,
    fit_ar1 as fit_ar1,
    cosine as cosine,
)
from engram_embeddings import EmbeddingTable as EmbeddingTable


if __name__ == "__main__":
    print("the Whisper Engram Universal Trajectory Codec\n")

    codec = Engram(dim=256)
    rng = np.random.RandomState(42)

    # trajectory at rest. should produce STATIC blocks, zero residuals.
    w = np.ones((50, 256), dtype=np.float32) * 0.3
    pkt = codec.encode(w)
    dec = codec.decode(pkt)
    err = float(np.sqrt(np.mean((w - dec) ** 2)))
    s = summary(pkt)
    print(f"at rest:       {s['modes']}  err={err:.8f}")
    print(f"               capture={s['capture']}%  snr={s['snr_db']}dB")
    print(f"               block size: {s['macro_block']} (data-derived)")
    assert err < 1e-5

    # superposition of slow orbit + fast jitter. the canonical test:
    # can the codec separate the fundamental from the harmonics?
    T = 120
    t = np.arange(T, dtype=np.float32)
    w2 = np.zeros((T, 256), dtype=np.float32)
    for p in range(128):
        slow = 0.05 + 0.005 * p
        fast = 0.4 + 0.03 * p
        w2[:, p * 2] = 0.3 * np.cos(slow * t) + 0.1 * np.cos(fast * t)
        w2[:, p * 2 + 1] = 0.3 * np.sin(slow * t) + 0.1 * np.sin(fast * t)
    w2 += 0.005 * rng.randn(T, 256).astype(np.float32)

    pkt2 = codec.encode(w2)
    dec2 = codec.decode(pkt2)
    err2 = float(np.sqrt(np.mean((w2 - dec2) ** 2)))
    s2 = summary(pkt2)
    print(f"\nsuperposition: {s2['modes']}  err={err2:.8f}")
    print(
        f"               fundamental={s2['macro_capture']:.1f}%  harmonics={s2['micro_capture']:.1f}%  total={s2['capture']}%  snr={s2['snr_db']}dB"
    )
    print(f"               block: macro={s2['macro_block']} micro={s2['micro_block']}")
    assert err2 < 1e-4

    # eigenvalue introspection on first block
    blk = pkt2.blocks[0]
    spec = spectral_summary(blk.macro_K, blk.macro_G)
    print(
        f"               eigenvalues: {spec['n_decaying']} decaying, {spec['n_sustained']} sustained, {spec['n_divergent']} divergent"
    )
    print(
        f"               freq: mean={spec['freq_mean']:.3f} std={spec['freq_std']:.3f}  mag: mean={spec['mag_mean']:.3f}"
    )

    # random trajectory. no structure to capture.
    w3 = rng.randn(60, 256).astype(np.float32) * 0.3
    pkt3 = codec.encode(w3)
    dec3 = codec.decode(pkt3)
    err3 = float(np.sqrt(np.mean((w3 - dec3) ** 2)))
    s3 = summary(pkt3)
    print(f"\nnoise floor:   {s3['modes']}  err={err3:.8f}")
    print(f"               capture={s3['capture']}%  block: {s3['macro_block']}")
    assert err3 < 1e-4

    # wire format round-trip
    wire = to_wire(pkt2)
    pkt2_rt = from_wire(wire)
    dec2_rt = codec.decode(pkt2_rt)
    wire_err = float(np.sqrt(np.mean((w2 - dec2_rt) ** 2)))
    raw_size = w2.size * 4
    print(
        f"\nwire:          {len(wire):,} bytes (raw {raw_size:,}) = {raw_size / len(wire):.1f}x"
    )
    print(f"               roundtrip err={wire_err:.6f}")

    # streaming encode/decode round-trip
    enc = codec.stream_encoder()
    dec_stream = codec.stream_decoder()
    n_blocks_streamed = 0
    for i in range(w2.shape[0]):
        blk = enc.push(w2[i])
        if blk is not None:
            dec_stream.push(blk)
            n_blocks_streamed += 1
    final_blk = enc.flush()
    if final_blk is not None:
        dec_stream.push(final_blk)
        n_blocks_streamed += 1
    stream_out = dec_stream.trajectory
    # streaming doesn't include the two global seeds at the front,
    # so compare from where the blocks start
    if stream_out.shape[0] > 0:
        print(
            f"\nstreaming:     {n_blocks_streamed} blocks, {stream_out.shape[0]} samples decoded"
        )

    # dream: observe accumulated eigenvalue dynamics, generate a novel trajectory
    print("\n--- dream ---")
    packets = []
    for seed in range(5):
        rng_d = np.random.RandomState(seed)
        wd = np.zeros((60, 256), dtype=np.float32)
        td = np.arange(60, dtype=np.float32)
        for p in range(128):
            freq = 0.05 + 0.02 * p + 0.01 * seed
            wd[:, p * 2] = 0.3 * np.cos(freq * td) + 0.005 * rng_d.randn(60)
            wd[:, p * 2 + 1] = 0.3 * np.sin(freq * td) + 0.005 * rng_d.randn(60)
        packets.append(codec.encode(wd))

    gen = codec.dream(packets, n_blocks=8, samples_per_block=16)
    if gen is not None:
        print(f"               generated: {gen.shape[0]} samples from 8 dreamed blocks")
        print(f"               range: [{gen.min():.4f}, {gen.max():.4f}]")
        print(f"               energy: {np.sum(gen**2):.2f}")

        # blend: interpolate eigenvalue states between first and last packet
        K1, G1 = packets[0].blocks[0].macro_K, packets[0].blocks[0].macro_G
        K2, G2 = packets[-1].blocks[0].macro_K, packets[-1].blocks[0].macro_G
        K_mid, G_mid = blend_eigenvalues(K1, G1, K2, G2, 0.5)
        s_a = spectral_summary(K1, G1)
        s_b = spectral_summary(K2, G2)
        s_m = spectral_summary(K_mid, G_mid)
        print(
            f"               blend: freq_a={s_a['freq_mean']:.3f} freq_b={s_b['freq_mean']:.3f} freq_mid={s_m['freq_mean']:.3f}"
        )
    else:
        print("               not enough blocks for dream")

    print("\nthe wire carries eigenvalues and residuals.")
    print("the decoder reconstructs trajectories from orbital parameters")
    print("and irreducible surprise.")
    print("dream learns how the eigenvalues evolve.")
    print("generation is dream running forward.")
