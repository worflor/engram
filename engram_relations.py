"""Context-indexed real multivariate moment summaries.

This module retains channel levels and centered cross-channel scatter. It does
not assign meaning to channel names, encode temporal order, or merge policies.
Storage is O(C D^2) for C observed contexts and D declared channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_CHECK_TOL = 1e-10


def _symmetric(values):
    # Preserve symmetric subnormal entries and avoid summing two large copies.
    return values + (values.T - values) * 0.5


def _schema(channels, representation):
    if not isinstance(channels, tuple) or not channels:
        raise ValueError("channels must be a nonempty tuple of unique names")
    if any(not isinstance(name, str) or not name for name in channels):
        raise ValueError("channel names must be nonempty strings")
    if len(set(channels)) != len(channels):
        raise ValueError("channel names must be unique")
    if not isinstance(representation, str) or not representation:
        raise ValueError("representation must be a nonempty string")


def _context_name(context):
    if not isinstance(context, str) or not context:
        raise ValueError("context must be a nonempty string")
    return context


def _readonly(values):
    array = np.ascontiguousarray(values, dtype=np.float64)
    # An immutable bytes owner prevents callers from re-enabling write access.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _validate_real_row(row, dim):
    raw = np.asarray(row)
    if raw.shape != (dim,):
        raise ValueError(f"row must have shape ({dim},)")
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TypeError("row must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("row values must be finite")
    return result


@dataclass(frozen=True)
class ContextMoments:
    """Immutable snapshot of one context's population moment summary."""

    count: int
    mean: np.ndarray
    scatter: np.ndarray
    covariance: np.ndarray
    variances: np.ndarray
    correlation: np.ndarray

    @classmethod
    def _create(cls, count, mean, scatter):
        mean = _readonly(mean)
        scatter = _readonly(scatter)
        covariance = _readonly(scatter / count)
        variances = _readonly(np.diag(covariance))
        dim = len(mean)
        correlation = np.zeros((dim, dim), dtype=np.float64)
        for i in range(dim):
            if variances[i] <= 0:
                continue
            for j in range(dim):
                if variances[j] > 0:
                    correlation[i, j] = (
                        covariance[i, j]
                        / math.sqrt(variances[i])
                        / math.sqrt(variances[j])
                    )
        correlation = np.clip((correlation + correlation.T) / 2, -1.0, 1.0)
        return cls(
            int(count),
            mean,
            scatter,
            covariance,
            variances,
            _readonly(correlation),
        )


class ContextRelations:
    """Streaming per-context multivariate means and centered scatter.

    `push` uses the stable centered Welford update. `merge` combines sufficient
    statistics with the parallel centered-moment identity. The context key and
    representation are caller-owned schema labels, not inferred semantics.
    """

    def __init__(self, channels: tuple[str, ...], *, representation: str):
        _schema(channels, representation)
        self._channels = tuple(channels)
        self._representation = representation
        self._states: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}

    @property
    def channels(self) -> tuple[str, ...]:
        return self._channels

    @property
    def representation(self) -> str:
        return self._representation

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(sorted(self._states))

    def push(self, context: str, row) -> None:
        key = _context_name(context)
        value = _validate_real_row(row, len(self.channels))
        old = self._states.get(key)
        if old is None:
            self._states[key] = (1, value.copy(), np.zeros((len(value), len(value))))
            return
        count, mean, scatter = old
        new_count = count + 1
        with np.errstate(over="ignore", invalid="ignore"):
            delta = value - mean
            updated_mean = mean + delta / new_count
            delta_after = value - updated_mean
            updated_scatter = scatter + np.outer(delta, delta_after)
            updated_scatter = _symmetric(updated_scatter)
        if not (np.isfinite(updated_mean).all() and np.isfinite(updated_scatter).all()):
            raise FloatingPointError("context moment update exceeded finite range")
        self._states[key] = (new_count, updated_mean, updated_scatter)

    def moments(self, context: str) -> ContextMoments | None:
        key = _context_name(context)
        state = self._states.get(key)
        if state is None:
            return None
        return ContextMoments._create(*state)

    def merge(self, other: ContextRelations) -> ContextRelations:
        """Return an independent parallel summary under an identical schema."""
        if not isinstance(other, ContextRelations):
            raise TypeError("other must be ContextRelations")
        if (self.channels, self.representation) != (
            other.channels,
            other.representation,
        ):
            raise ValueError("context relation schemas must match exactly")
        result = ContextRelations(self.channels, representation=self.representation)
        for key in sorted(set(self._states) | set(other._states)):
            left = self._states.get(key)
            right = other._states.get(key)
            if left is None:
                count, mean, scatter = right
                result._states[key] = (count, mean.copy(), scatter.copy())
                continue
            if right is None:
                count, mean, scatter = left
                result._states[key] = (count, mean.copy(), scatter.copy())
                continue
            na, mean_a, scatter_a = left
            nb, mean_b, scatter_b = right
            count = na + nb
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                fraction_b = nb / count
                delta = mean_b - mean_a
                mean = mean_a + delta * fraction_b
                smaller = min(na, nb)
                larger = max(na, nb)
                cross_weight = smaller * (larger / count)
                # Apply the weight before the outer product: delta^2 can
                # overflow although the weighted contribution is representable.
                weighted_delta = delta * math.sqrt(cross_weight)
                scatter = (
                    scatter_a + scatter_b + np.outer(weighted_delta, weighted_delta)
                )
                scatter = _symmetric(scatter)
            if not np.isfinite(mean).all() or not np.isfinite(scatter).all():
                raise FloatingPointError("context moment merge exceeded finite range")
            result._states[key] = (count, mean, scatter)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "channels": list(self.channels),
            "representation": self.representation,
            "contexts": {
                key: {
                    "count": count,
                    "mean": mean.tolist(),
                    "scatter": scatter.tolist(),
                }
                for key, (count, mean, scatter) in sorted(self._states.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextRelations:
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data.get("version") != 1
        ):
            raise ValueError("unsupported context relation checkpoint version")
        channels = data.get("channels")
        if isinstance(channels, list):
            channels = tuple(channels)
        result = cls(channels, representation=data.get("representation"))
        contexts = data.get("contexts")
        if not isinstance(contexts, dict):
            raise ValueError("contexts must be a mapping")
        dim = len(result.channels)
        restored = {}
        for raw_key, state in contexts.items():
            key = _context_name(raw_key)
            if not isinstance(state, dict):
                raise ValueError("context state must be a mapping")
            count = state.get("count")
            if type(count) is not int or count < 1:
                raise ValueError("context count must be a positive integer")
            try:
                raw_mean = np.asarray(state.get("mean"))
                raw_scatter = np.asarray(state.get("scatter"))
                if (
                    raw_mean.dtype.kind not in "iuf"
                    or raw_scatter.dtype.kind not in "iuf"
                ):
                    raise ValueError("moment arrays must contain real numeric values")
                with np.errstate(over="ignore", invalid="ignore"):
                    mean = np.asarray(raw_mean, dtype=np.float64)
                    scatter = np.asarray(raw_scatter, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("invalid context moment arrays") from error
            if mean.shape != (dim,) or scatter.shape != (dim, dim):
                raise ValueError("context moment arrays have the wrong shape")
            if not np.isfinite(mean).all() or not np.isfinite(scatter).all():
                raise ValueError("context moment arrays must be finite")
            magnitude = float(np.max(np.abs(scatter)))
            if count == 1:
                if magnitude != 0:
                    raise ValueError("a one-sample centered scatter must be zero")
                restored[key] = (count, mean.copy(), scatter.copy())
                continue
            # Work in scale-free coordinates for validation. This keeps the
            # PSD, rank and symmetry decisions invariant under changes of units,
            # and avoids overflowing the eigensolver on finite large matrices.
            normalized = scatter / magnitude if magnitude else scatter.copy()
            asymmetry = float(np.max(np.abs(normalized - normalized.T)))
            if asymmetry > _CHECK_TOL:
                raise ValueError("scatter must be symmetric")
            normalized = _symmetric(normalized)
            scatter = _symmetric(scatter)
            try:
                eigenvalues = np.linalg.eigvalsh(normalized)
                minimum_eigenvalue = float(eigenvalues[0])
            except np.linalg.LinAlgError as error:
                raise ValueError("scatter PSD check failed") from error
            if not np.isfinite(eigenvalues).all():
                raise ValueError("scatter PSD check exceeded finite range")
            tolerance = _CHECK_TOL
            if minimum_eigenvalue < -tolerance:
                raise ValueError("scatter must be positive semidefinite")
            # A centered scatter from n rows has rank at most n-1. Eigenvalues
            # beneath this roundoff threshold are treated as numerical zeros.
            positive_rank = int(np.count_nonzero(eigenvalues > tolerance))
            if positive_rank > count - 1:
                raise ValueError("scatter rank exceeds count minus one")
            restored[key] = (count, mean.copy(), scatter.copy())
        result._states = restored
        return result
