"""Real-coordinate boundaries and stable characteristic-root extraction."""

import numpy as np
import pytest

from engram_codec import (
    Engram,
    Identity,
    TanhProjection,
    eigenvalues,
    eigenvalues_all,
    spectral_summary,
)
from engram_analysis import dynamics, cosine
from engram_affine import AffineOscillator
from engram_history import WitnessHistory
from engram_npc import WitnessSchema
from engram_training import LookupSource, smooth_trajectory


@pytest.mark.parametrize("k", [1e8, 1e12, -1e12])
def test_unequal_characteristic_roots_preserve_the_small_mode(k):
    roots = eigenvalues(k, 1.0)
    assert min(abs(r) for r in roots) > 0
    np.testing.assert_allclose(roots[0] * roots[1], 1.0, rtol=1e-14)
    np.testing.assert_allclose(roots[0] + roots[1], k, rtol=1e-14)


@pytest.mark.parametrize(
    "k,g",
    [(1e308, 1e308), (1e308 + 1e308j, 1e308), (0.0, 5e-324), (1e-300, 0.0), (0.0, 0.0)],
)
def test_finite_extreme_roots_avoid_intermediate_overflow(k, g):
    roots = eigenvalues(k, g)
    assert np.isfinite(roots).all()
    if k == 1e308 and g == 1e308:
        assert min(abs(r) for r in roots) == 1.0
    if k == 0 and g > 0:
        assert all(r != 0 for r in roots)


def test_scalar_and_vector_roots_use_the_same_branch_convention():
    k = np.array([2.0, -2.0, 0.0, 1 + 2j, 1e12])
    g = np.array([1.0, 1.0, 0.0, 0.2 - 0.3j, 1.0])
    first, second = eigenvalues_all(k, g)
    for i in range(len(k)):
        np.testing.assert_array_equal([first[i], second[i]], eigenvalues(k[i], g[i]))
    # A separate companion eigensolver supplies the moderate-scale reference.
    for i in range(4):
        expected = np.linalg.eigvals([[k[i], -g[i]], [1, 0]])
        assert (
            min(
                max(abs(first[i] - a), abs(second[i] - b))
                for a, b in [expected, expected[::-1]]
            )
            < 1e-7
        )


@pytest.mark.parametrize("k,g", [(np.nan, 1), (1, np.inf)])
def test_nonfinite_characteristic_coefficients_reject(k, g):
    with pytest.raises(ValueError, match="finite"):
        eigenvalues(k, g)


@pytest.mark.parametrize(
    "entry",
    [
        lambda x: Engram(2).encode(x),
        lambda x: AffineOscillator.fit(x),
        lambda x: Identity().project(x),
        lambda x: TanhProjection(np.eye(2)).project(x),
        lambda x: smooth_trajectory(x),
        lambda x: LookupSource(x, lambda value: np.array([0, 1, 2])),
    ],
)
def test_real_trajectory_interfaces_reject_complex_arrays_without_discarding_parts(
    entry,
):
    with pytest.raises(ValueError, match="real"):
        entry(np.ones((8, 2), complex) * (1 + 1j))


def test_rejected_witness_does_not_advance_history():
    history = WitnessHistory(2)
    history.push([1, 2])
    before = history.snapshot()
    with pytest.raises(ValueError, match="real"):
        history.push(np.array([1 + 1j, 2]))
    assert history.snapshot() == before
    with pytest.raises(ValueError, match="real"):
        WitnessSchema(("x", "y")).vector(np.array([1 + 1j, 2]))


def test_affine_forecast_and_translation_reject_complex_host_coordinates():
    model = AffineOscillator.fit(np.ones((8, 2)))
    with pytest.raises(ValueError, match="real"):
        model.translated([1 + 1j, 0])
    with pytest.raises(ValueError, match="real"):
        model.forecast(2, seeds=np.ones((2, 2), complex))


def test_diagnostics_remain_finite_when_real_coefficient_squares_would_overflow():
    info = dynamics(1e308, 1e308)
    assert info.classification == "growing" and info.spectral_radius == 1e308
    assert info.oscillation_period_samples is None and not info.repeated_root
    summary = spectral_summary(np.array([1e308]), np.array([1e308]))
    assert summary["mag_mean"] == 5e307
    assert summary["mag_std"] == 5e307
    assert cosine([], []) == 0


def test_unrepresentable_spectral_magnitudes_reject_instead_of_emitting_infinity():
    for diagnose in [
        lambda: dynamics(1.7e308 + 1.7e308j, 1.0),
        lambda: spectral_summary(np.array([1.7e308 + 1.7e308j]), np.array([1.0])),
    ]:
        with pytest.raises(FloatingPointError, match="spectral radius"):
            diagnose()


def test_identity_projection_retains_independent_output_ownership():
    source = np.ones((4, 2), np.float32)
    projected = Identity().project(source)
    projected[:] = 0
    np.testing.assert_array_equal(source, np.ones((4, 2)))
