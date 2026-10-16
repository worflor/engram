"""Shared oscillator diagnostics and trajectory signatures, independent of host policy."""

from dataclasses import dataclass
import math
import numpy as np


def fit_ar1(z):
    """Fit z[n] = K z[n-1]; the explicit short-observation path from WDYM."""
    z = np.asarray(z, dtype=np.complex128)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("AR(1) needs at least two finite complex samples")
    energy = float(np.vdot(z[:-1], z[:-1]).real)
    ridge = max(float(np.finfo(np.float32).eps), 1e-4 * energy)
    k = np.vdot(z[:-1], z[1:]) / (energy + ridge)
    rms = float(np.sqrt(np.mean(np.abs(z[1:] - k * z[:-1]) ** 2)))
    return complex(k), rms


@dataclass(frozen=True)
class Dynamics:
    roots: tuple[complex, complex]
    spectral_radius: float
    half_life_samples: float | None
    oscillation_period_samples: float | None
    classification: str
    repeated_root: bool


def dynamics(k, g, tolerance=1e-6):
    """Characteristic-root diagnostics; half-life describes an exponential envelope."""
    from engram_codec import eigenvalues

    if not np.isfinite([k, g]).all() or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("finite coefficients and nonnegative tolerance required")
    a, b = eigenvalues(complex(k), complex(g))
    radius = max(math.hypot(a.real, a.imag), math.hypot(b.real, b.imag))
    if not math.isfinite(radius):
        raise FloatingPointError("spectral radius exceeded finite range")
    classification = (
        "decaying"
        if radius < 1 - tolerance
        else ("growing" if radius > 1 + tolerance else "unit-radius")
    )
    half = -math.log(2) / math.log(radius) if 0 < radius < 1 else None
    period = None
    # A scalar real recurrence has a well-defined conjugate-pair frequency.
    if (
        abs(complex(k).imag) <= tolerance
        and abs(complex(g).imag) <= tolerance
        and complex(g).real > 0
        and abs(complex(k).real) / 2 < math.sqrt(complex(g).real)
    ):
        omega = abs(np.angle(a))
        if omega > 0:
            period = float(2 * math.pi / omega)
    scale = max(1.0, radius)
    repeated = abs(a / scale - b / scale) <= tolerance
    return Dynamics((a, b), radius, half, period, classification, repeated)


def cosine(a, b):
    a, b = np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("equal finite vectors required")
    # Scale each vector by its largest real/imaginary component before taking
    # norms or inner products. This preserves direction while avoiding the
    # overflow/underflow of a direct norm product for otherwise finite inputs.
    scale_a = float(np.max(np.maximum(np.abs(a.real), np.abs(a.imag)), initial=0))
    scale_b = float(np.max(np.maximum(np.abs(b.real), np.abs(b.imag)), initial=0))
    if scale_a == 0 or scale_b == 0:
        return 0.0
    scaled_a, scaled_b = a / scale_a, b / scale_b
    norm_a, norm_b = np.linalg.norm(scaled_a), np.linalg.norm(scaled_b)
    similarity = np.vdot(scaled_a, scaled_b).real / norm_a / norm_b
    return float(np.clip(similarity, -1, 1))


@dataclass(frozen=True)
class Signature:
    k: np.ndarray
    g: np.ndarray
    pair_rms: np.ndarray
    pairing: np.ndarray
    samples: int
    order: int

    @classmethod
    def from_trajectory(cls, trajectory, pairing=None):
        from engram_codec import (
            Engram,
            derive_pairing,
            apply_pairing,
            _to_complex,
            fit_all,
            _validated_trajectory,
            _real_array,
        )

        x = _real_array(trajectory, dtype=np.float32)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("signature needs at least two rows")
        # The codec boundary validates shape, finite values and supplied pairing.
        Engram(x.shape[1])
        x, pairing = _validated_trajectory(x, x.shape[1], pairing)
        basis = (
            derive_pairing(x)
            if pairing is None
            else np.asarray(pairing, dtype=np.int32)
        )
        z = _to_complex(apply_pairing(x, basis))
        if len(x) == 2:
            fits = [fit_ar1(z[:, i]) for i in range(z.shape[1])]
            k = np.array([f[0] for f in fits])
            g = np.zeros_like(k)
            rms = np.array([f[1] for f in fits])
            order = 1
        else:
            k, g, _, rms = fit_all(z)
            order = 2
        values = [np.array(a, copy=True) for a in (k, g, rms, basis)]
        for a in values:
            a.flags.writeable = False
        return cls(*values, len(x), order)

    def compare(self, other):
        if not np.array_equal(self.pairing, other.pairing):
            raise ValueError("signatures must use the same coordinate pairing")
        if self.order != other.order:
            raise ValueError(
                "AR(1) and AR(2) signatures need an explicit host comparison policy"
            )
        k, g = cosine(self.k, other.k), cosine(self.g, other.g)
        return {
            "k_cosine": k,
            "g_cosine": g,
            "joint_cosine": math.sqrt(max(0, k) * max(0, g)),
            "kg_rms_distance": float(
                np.sqrt(
                    np.mean(
                        np.abs(self.k - other.k) ** 2 + np.abs(self.g - other.g) ** 2
                    )
                )
            ),
        }
