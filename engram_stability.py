"""Normalized Schur coordinates for explicit frozen-oscillator generation views.

These coordinates do not replace observed K/G sufficient statistics. A stable
frozen blend is not a certificate for arbitrary switching, bounded transients
or noisy trajectories. A radius larger than one permits growing modes;
boundary modes of the chosen root disk are deliberately rejected.
"""

import math
import numpy as np


def _parameters(k, g, margin):
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError("Schur margin must be in [0,1)")
    k, g = np.asarray(k, complex), np.asarray(g, complex)
    if k.shape != g.shape or not np.isfinite(k).all() or not np.isfinite(g).all():
        raise ValueError("matching finite complex coefficient arrays required")
    return k, g


def _disk_radius(radius):
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("finite positive root-disk radius required")
    return float(radius)


def schur_coordinates(k, g, *, radius=1.0, margin=1e-8):
    """Return normalized kappa/g for roots strictly inside the requested disk.

    Normalize K/radius and G/radius². Returned g is dimensionless; retain the
    radius when reconstructing coefficients. Margin applies to these normalized
    reflection disks, not directly to root distance from the requested bound.
    """
    radius = _disk_radius(radius)
    k, g = _parameters(k, g, margin)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        k, g = k / radius, (g / radius) / radius
    if np.any(abs(g) >= 1 - margin):
        raise ValueError("oscillator is outside the strict Schur G disk/margin")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        kappa = (k - g * k.conj()) / (1 - abs(g) ** 2)
    if not np.isfinite(kappa).all() or np.any(abs(kappa) >= 1 - margin):
        raise ValueError(
            "oscillator is outside the strict Schur reflection disk/margin"
        )
    return kappa, g.copy()


def schur_coefficients(kappa, g, *, radius=1.0, margin=1e-8):
    """Map the product of strict reflection disks back to K/G coefficients."""
    radius = _disk_radius(radius)
    kappa, g = _parameters(kappa, g, margin)
    if np.any(abs(kappa) >= 1 - margin) or np.any(abs(g) >= 1 - margin):
        raise ValueError("strict reflection disks/margin required")
    with np.errstate(over="ignore", invalid="ignore"):
        k, product = (kappa + g * kappa.conj()) * radius, (g * radius) * radius
    if not np.isfinite(k).all() or not np.isfinite(product).all():
        raise FloatingPointError("scaled Schur coefficients exceeded finite range")
    try:
        schur_coordinates(k, product, radius=radius, margin=margin)
    except ValueError as exc:
        raise FloatingPointError(
            "scaled Schur coefficients lost their requested disk at finite precision"
        ) from exc
    return k, product


def blend_schur(k, g, weights, *, radius=1.0, margin=1e-8, dtype=np.complex128):
    """Convex generation blend of [node,pair] coefficients; check rounded output.

    Zero-weight nodes are excluded. Weights need not sum to one. Both Schur
    margins and roots are numerically checked after the requested complex64 or
    complex128 rounding, relative to radius. Participating inputs must already
    satisfy the bound; no input is clipped into the admissible region.
    """
    radius = _disk_radius(radius)
    k, g = _parameters(k, g, margin)
    weights = np.asarray(weights, float)
    precision = np.dtype(dtype)
    if (
        k.ndim != 2
        or not len(k)
        or k.shape[1] == 0
        or weights.shape != (len(k),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise ValueError("nonnegative nonzero weights and [node,pair] arrays required")
    if precision not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("coefficient precision must be complex64 or complex128")
    active = weights > 0
    weights = weights[active] / weights.max()
    weights /= weights.sum()
    reflection, products = schur_coordinates(
        k[active], g[active], radius=radius, margin=margin
    )
    result_k, result_g = schur_coefficients(
        weights @ reflection, weights @ products, radius=radius, margin=margin
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result_k, result_g = result_k.astype(precision), result_g.astype(precision)
    schur_coordinates(result_k, result_g, radius=radius, margin=margin)
    # Independent numeric root check on the actual rounded polynomial.
    kd, gd = (
        result_k.astype(complex) / radius,
        (result_g.astype(complex) / radius) / radius,
    )
    discriminant = np.sqrt(kd * kd - 4 * gd)
    root_radius = np.maximum(abs((kd + discriminant) / 2), abs((kd - discriminant) / 2))
    if not np.isfinite(root_radius).all() or np.any(root_radius >= 1):
        raise FloatingPointError("rounded Schur blend failed its strict root check")
    return result_k, result_g
