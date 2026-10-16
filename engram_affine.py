"""Explicit affine AR(2) fits alongside Engram's homogeneous packet contract.

The intercept is unpenalized. Independently centered lag columns implement
the corresponding least-squares problem without treating a window mean as
the oscillator's equilibrium. This module neither changes V4 nor trains wells.
"""

from dataclasses import dataclass
import math

import numpy as np

from engram_analysis import Signature
from engram_codec import Engram, MACHINE_EPS, _real_array


def _readonly(value, dtype):
    result = (
        _real_array(value, dtype=dtype).copy()
        if np.dtype(dtype).kind == "f"
        else np.array(value, dtype=dtype, copy=True)
    )
    if not np.isfinite(result).all():
        raise ValueError("finite affine model state required")
    result.flags.writeable = False
    return result


def _complex(values, pairing):
    paired = values[:, pairing]
    return paired[:, ::2] + 1j * paired[:, 1::2]


def _coefficients(gram, rhs, relative_ridge):
    if not math.isfinite(relative_ridge) or relative_ridge < 0:
        raise ValueError("finite nonnegative relative ridge required")
    with np.errstate(over="ignore", invalid="ignore"):
        trace = np.trace(gram, axis1=1, axis2=2).real
        ridge = (
            np.maximum(float(MACHINE_EPS), relative_ridge * trace / 2)
            if relative_ridge
            else np.zeros(len(gram))
        )
        regularized = gram + ridge[:, None, None] * np.eye(2)
    if not np.isfinite(regularized).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("affine fit exceeded finite range")
    try:
        kg = np.linalg.solve(regularized, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "affine design is singular; use positive ridge or richer observations"
        ) from exc
    if not np.isfinite(kg).all():
        raise FloatingPointError("affine fit exceeded finite range")
    return kg


@dataclass(frozen=True)
class AffineOscillator:
    """Paired complex recurrence with a fitted origin and two causal seeds.

    signature contains centered-lag K/G and measured residual RMS. lag_means
    is [pair, target/previous/older]; seeds are in original channel order.
    Fits and forecasts retain float64 precision, independently of wire storage.
    """

    signature: Signature
    lag_means: np.ndarray
    seeds: np.ndarray

    def __post_init__(self):
        s = self.signature
        basis = np.asarray(s.pairing)
        d = len(basis)
        Engram(d)
        if (
            basis.dtype.kind not in "iu"
            or not np.array_equal(np.sort(basis), np.arange(d))
            or s.order != 2
            or type(s.samples) is not int
            or s.samples < 5
        ):
            raise ValueError(
                "an AR(2) signature with a matching basis and five samples is required"
            )
        k, g, rms = (
            _readonly(a, t)
            for a, t in [(s.k, complex), (s.g, complex), (s.pair_rms, float)]
        )
        if any(a.shape != (d // 2,) for a in (k, g, rms)) or np.any(rms < 0):
            raise ValueError("invalid affine signature shape or residual")
        means, seeds = _readonly(self.lag_means, complex), _readonly(self.seeds, float)
        if means.shape != (d // 2, 3) or seeds.shape != (2, d):
            raise ValueError("invalid affine lag means or seeds")
        object.__setattr__(
            self,
            "signature",
            Signature(k, g, rms, _readonly(basis, np.int32), s.samples, 2),
        )
        object.__setattr__(self, "lag_means", means)
        object.__setattr__(self, "seeds", seeds)

    @property
    def dim(self):
        return len(self.signature.pairing)

    @property
    def bias(self):
        """Explicit b in z[t] = K*z[t-1] - G*z[t-2] + b."""
        target, previous, older = self.lag_means.T
        with np.errstate(over="ignore", invalid="ignore"):
            result = target - self.signature.k * previous + self.signature.g * older
        if not np.isfinite(result).all():
            raise FloatingPointError("explicit affine bias exceeded finite range")
        result.flags.writeable = False
        return result

    @classmethod
    def fit(cls, trajectory, *, pairing=None, relative_ridge=1e-4):
        """Minimize squared residuals plus ridge*(|K|²+|G|²); b is free.

        Five rows allow three complex observations per pair. Identifiability
        still depends on the data. relative_ridge=0 requests unregularized
        fitting and rejects singular designs rather than inventing coefficients.
        """
        x = _real_array(trajectory)
        if x.ndim != 2 or len(x) < 5 or not np.isfinite(x).all():
            raise ValueError("at least five finite trajectory rows required")
        Engram(x.shape[1])
        d = x.shape[1]
        basis = np.arange(d, dtype=np.int32) if pairing is None else np.asarray(pairing)
        if (
            basis.shape != (d,)
            or basis.dtype.kind not in "iu"
            or not np.array_equal(np.sort(basis), np.arange(d))
        ):
            raise ValueError("pairing must be a complete integer permutation")
        if not math.isfinite(relative_ridge) or relative_ridge < 0:
            raise ValueError("finite nonnegative relative ridge required")
        z = _complex(x, basis)
        # These three means generally differ. Centering every column by one
        # whole-window mean does NOT eliminate an unpenalized intercept.
        with np.errstate(over="ignore", invalid="ignore"):
            joint = np.stack([z[2:], z[1:-1], z[:-2]], axis=-1)
            means = joint.mean(axis=0)
            centered = joint - means
            target = centered[..., 0]
            design = np.stack([centered[..., 1], -centered[..., 2]], axis=-1)
            gram = np.einsum("tpi,tpj->pij", design.conj(), design)
            rhs = np.einsum("tpi,tp->pi", design.conj(), target)
        if not np.isfinite(means).all():
            raise FloatingPointError("affine fit exceeded finite range")
        kg = _coefficients(gram, rhs, relative_ridge)
        with np.errstate(over="ignore", invalid="ignore"):
            errors = target - np.einsum("tpi,pi->tp", design, kg)
            rms = np.sqrt(np.mean(abs(errors) ** 2, axis=0))
        if not np.isfinite(kg).all() or not np.isfinite(rms).all():
            raise FloatingPointError("affine fit exceeded finite range")
        signature = Signature(kg[:, 0], kg[:, 1], rms, basis, len(x), 2)
        return cls(signature, means, x[-2:])

    @classmethod
    def from_history(cls, history, *, relative_ridge=1e-4):
        """Snapshot an existing WitnessHistory in O(D), including its decay.

        Uses joint lag means/scatter, not the history's common-centered fit.
        No old observation buffer, fabricated samples or live references are
        needed. WitnessHistory's coordinates use the identity pairing.
        """
        from engram_history import WitnessHistory

        if not isinstance(history, WitnessHistory) or history.samples < 5:
            raise ValueError("a WitnessHistory with at least five samples is required")
        cross = history.scatter
        gram = np.empty((history.dim // 2, 2, 2), complex)
        gram[:, 0, 0], gram[:, 1, 1] = cross[:, 1, 1], cross[:, 2, 2]
        gram[:, 0, 1], gram[:, 1, 0] = -cross[:, 2, 1], -cross[:, 1, 2]
        rhs = np.stack([cross[:, 0, 1], -cross[:, 0, 2]], axis=1)
        kg = _coefficients(gram, rhs, relative_ridge)
        coefficients = np.column_stack([np.ones(len(kg)), -kg[:, 0], kg[:, 1]])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            error = (
                np.einsum("pi,pij,pj->p", coefficients, cross, coefficients.conj()).real
                / history.joint_weight
            )
            rms = np.sqrt(np.maximum(error, 0))
        if not np.isfinite(rms).all():
            raise FloatingPointError("affine history fit exceeded finite range")
        seeds = np.empty((2, history.dim), float)
        seeds[:, ::2], seeds[:, 1::2] = (
            np.asarray(history.previous).real,
            np.asarray(history.previous).imag,
        )
        signature = Signature(
            kg[:, 0], kg[:, 1], rms, np.arange(history.dim), history.samples, 2
        )
        return cls(signature, history.joint_mean, seeds)

    def translated(self, offset):
        """Move the fitted trajectory's origin; keep K/G/residual unchanged."""
        delta = _real_array(offset)
        if delta.shape != (self.dim,) or not np.isfinite(delta).all():
            raise ValueError("finite matching translation required")
        dz = _complex(delta[None], self.signature.pairing)[0]
        with np.errstate(over="ignore", invalid="ignore"):
            means, seeds = self.lag_means + dz[:, None], self.seeds + delta
        return type(self)(self.signature, means, seeds)

    def forecast(self, steps, *, seeds=None):
        """Deterministic continuation; independent calls do not mutate the fit.

        Evaluates the anchored affine equation to avoid cancellation of a large
        absolute level. This fixes origin dependence, not unstable roots or
        arbitrary switching. Host-supplied seeds use the fitted coordinates.
        """
        if type(steps) is not int or steps < 0:
            raise ValueError("nonnegative integer forecast length required")
        initial = self.seeds if seeds is None else _real_array(seeds)
        if initial.shape != (2, self.dim) or not np.isfinite(initial).all():
            raise ValueError("two finite seed rows in original channel order required")
        older, previous = _complex(initial, self.signature.pairing)
        target_mean, previous_mean, older_mean = self.lag_means.T
        predicted = np.empty((steps, self.dim // 2), complex)
        for i in range(steps):
            with np.errstate(over="ignore", invalid="ignore"):
                value = (
                    target_mean
                    + self.signature.k * (previous - previous_mean)
                    - self.signature.g * (older - older_mean)
                )
            if not np.isfinite(value).all():
                raise FloatingPointError("affine forecast exceeded finite range")
            predicted[i] = value
            older, previous = previous, value
        result = np.empty((steps, self.dim), float)
        result[:, self.signature.pairing[::2]] = predicted.real
        result[:, self.signature.pairing[1::2]] = predicted.imag
        return result
