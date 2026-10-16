"""Ordered streaming AR(2) witness summaries and exact concatenation semantics."""

import copy
import math
import numpy as np
from engram_analysis import Signature
from engram_codec import MACHINE_EPS, _real_array


class WitnessHistory:
    """O(D) storage for an unlimited stream, using centered joint lag moments.

    decay=1 retains the whole episode's sufficient statistics. Lower decay
    gives an explicit exponential recency policy, without a fixed sample cap.
    Fits use the same ridge and K/G convention as the reference fit_all.
    """

    def __init__(self, dim, decay=1.0):
        if dim < 2 or dim % 2 or not math.isfinite(decay) or not 0 < decay <= 1:
            raise ValueError("positive even dimension and decay in (0,1] required")
        self.dim, self.decay = dim, float(decay)
        p = dim // 2
        self.samples = 0
        self.weight = 0.0
        self.joint_weight = 0.0
        self.mean = np.zeros(p, complex)
        self.joint_mean = np.zeros((p, 3), complex)
        self.scatter = np.zeros((p, 3, 3), complex)
        self.first = []
        self.previous = []

    def push(self, vector):
        x = _real_array(vector)
        if x.shape != (self.dim,) or not np.isfinite(x).all():
            raise ValueError("finite matching history vector required")
        z = x[::2] + 1j * x[1::2]
        if self.first is not None and len(self.first) < 2:
            self.first.append(z.copy())
        self.weight = self.weight * self.decay + 1
        self.mean += (z - self.mean) / self.weight
        if len(self.previous) == 2:
            joint = np.stack([z, self.previous[-1], self.previous[-2]], axis=1)
            self.joint_weight = self.joint_weight * self.decay + 1
            delta = joint - self.joint_mean
            self.joint_mean += delta / self.joint_weight
            self.scatter = (
                self.decay * self.scatter
                + delta[:, :, None] * (joint - self.joint_mean).conj()[:, None, :]
            )
        self.previous = (self.previous + [z.copy()])[-2:]
        self.samples += 1
        return self.signature()

    def concat(self, following):
        """Summary of this ordered stream followed by another, without replay.

        Internal triples are combined with at most two boundary triples. Older
        weights age by decay**len(following). Exact arithmetic is associative;
        floating grouping is tolerance-based. Neither operand is mutated.
        This composes fit statistics, not online well assignments or Q tables.
        """
        if not isinstance(following, WitnessHistory) or (self.dim, self.decay) != (
            following.dim,
            following.decay,
        ):
            raise ValueError(
                "concatenation requires matching history dimension and decay"
            )
        if not self.samples:
            return copy.deepcopy(following)
        if not following.samples:
            return copy.deepcopy(self)
        if following.first is None:
            raise ValueError(
                "following history lacks prefix endpoints from its legacy snapshot"
            )
        result = WitnessHistory(self.dim, self.decay)
        age = self.decay**following.samples
        wa, wb = self.weight * age, following.weight
        result.weight = wa + wb
        result.mean = (
            following.mean.copy()
            if wa == 0
            else self.mean + (following.mean - self.mean) * (wb / result.weight)
        )
        result.samples = self.samples + following.samples
        result.first = (
            None if self.first is None else (self.first + following.first)[:2]
        )
        result.first = (
            None if result.first is None else [z.copy() for z in result.first]
        )
        result.previous = [z.copy() for z in (self.previous + following.previous)[-2:]]

        def combine(weight, mean, scatter):
            if weight == 0:
                return
            old = result.joint_weight
            if old == 0:
                result.joint_weight, result.joint_mean, result.scatter = (
                    weight,
                    mean.copy(),
                    scatter.copy(),
                )
                return
            total = old + weight
            delta = mean - result.joint_mean
            result.scatter += (
                scatter
                + ((old / total) * weight)
                * delta[:, :, None]
                * delta.conj()[:, None, :]
            )
            result.joint_mean += delta * (weight / total)
            result.joint_weight = total

        combine(self.joint_weight * age, self.joint_mean, self.scatter * age)
        combine(following.joint_weight, following.joint_mean, following.scatter)
        zeros = np.zeros_like(result.scatter)
        if self.samples >= 2:
            joint = np.stack(
                [following.first[0], self.previous[-1], self.previous[-2]], axis=1
            )
            combine(self.decay ** (following.samples - 1), joint, zeros)
        if following.samples >= 2:
            joint = np.stack(
                [following.first[1], following.first[0], self.previous[-1]], axis=1
            )
            combine(self.decay ** (following.samples - 2), joint, zeros)
        return result

    def signature(self, *, center=True):
        if self.samples < 3:
            return None
        delta = self.joint_mean - (self.mean[:, None] if center else 0)
        cross = (
            self.scatter
            + self.joint_weight * delta[:, :, None] * delta.conj()[:, None, :]
        )
        aa, bb = cross[:, 1, 1].real, cross[:, 2, 2].real
        ab, ta, tb = cross[:, 1, 2], cross[:, 0, 1], cross[:, 0, 2]
        trace = aa + bb
        ridge = np.maximum(float(MACHINE_EPS), 1e-4 * trace / 2)
        ar, br = aa + ridge, bb + ridge
        det = ar * br - abs(ab) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            k = (ta * br - ab.conj() * tb) / det
            g = (ab * ta - ar * tb) / det
        bad = (~np.isfinite(det)) | (abs(det) < float(MACHINE_EPS) * trace**2)
        k = np.where(bad | ~np.isfinite(k), 2.0 + 0j, k)
        g = np.where(bad | ~np.isfinite(g), 1.0 + 0j, g)
        coefficients = np.stack([np.ones_like(k), -k, g], axis=1)
        error = (
            np.einsum("pi,pij,pj->p", coefficients, cross, coefficients.conj()).real
            / self.joint_weight
        )
        rms = np.sqrt(np.maximum(error, 0))
        if not all(np.isfinite(a).all() for a in [k, g, rms]):
            raise FloatingPointError("history fit exceeded finite range")
        return Signature(
            k, g, rms, np.arange(self.dim, dtype=np.int32), self.samples, 2
        )

    def snapshot(self):
        def complex_rows(a):
            return np.stack([a.real, a.imag], axis=-1).tolist()

        return {
            "version": 2,
            "samples": self.samples,
            "weight": self.weight,
            "joint_weight": self.joint_weight,
            "mean": complex_rows(self.mean),
            "joint_mean": complex_rows(self.joint_mean),
            "scatter": complex_rows(self.scatter),
            "previous": [complex_rows(z) for z in self.previous],
            "first": None
            if self.first is None
            else [complex_rows(z) for z in self.first],
        }

    @classmethod
    def restore(cls, dim, decay, data):
        history = cls(dim, decay)
        if data.get("version", 1) not in (1, 2):
            raise ValueError("unsupported witness history version")

        def complex_array(value, shape):
            a = np.asarray(value, float)
            if a.shape != shape + (2,) or not np.isfinite(a).all():
                raise ValueError("invalid saved history moments")
            return a[..., 0] + 1j * a[..., 1]

        p = dim // 2
        history.samples = data["samples"]
        history.weight = float(data["weight"])
        history.joint_weight = float(data["joint_weight"])
        if (
            type(history.samples) is not int
            or history.samples < 0
            or not np.isfinite([history.weight, history.joint_weight]).all()
            or history.weight < 0
            or history.joint_weight < 0
        ):
            raise ValueError("invalid history counters")
        history.mean = complex_array(data["mean"], (p,))
        history.joint_mean = complex_array(data["joint_mean"], (p, 3))
        history.scatter = complex_array(data["scatter"], (p, 3, 3))
        history.previous = [complex_array(z, (p,)) for z in data["previous"]]
        if len(history.previous) != min(history.samples, 2):
            raise ValueError("invalid history endpoints")
        first = data.get("first", data["previous"] if history.samples <= 2 else None)
        history.first = (
            None if first is None else [complex_array(z, (p,)) for z in first]
        )
        if history.first is not None and len(history.first) != min(history.samples, 2):
            raise ValueError("invalid history prefix endpoints")
        if history.samples <= 2 and (
            history.first is None
            or any(
                not np.array_equal(a, b)
                for a, b in zip(history.first, history.previous)
            )
        ):
            raise ValueError("short history endpoints disagree")
        return history
