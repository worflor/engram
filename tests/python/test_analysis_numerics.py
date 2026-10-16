"""Scaled cosine and real-trajectory boundary regressions."""

import warnings

import numpy as np
import pytest

from engram_analysis import Signature, cosine


@pytest.mark.parametrize("magnitude", [1e308, 1e-300])
def test_cosine_handles_finite_huge_and_tiny_vectors(magnitude):
    a = np.array([magnitude, magnitude / 2, -magnitude])
    np.testing.assert_allclose(cosine(a, a), 1.0, atol=1e-15)
    np.testing.assert_allclose(cosine(a, -a), -1.0, atol=1e-15)


@pytest.mark.parametrize("magnitude", [1e308, 1e-300])
def test_complex_cosine_preserves_real_inner_product_at_extreme_scales(magnitude):
    a = np.array([magnitude + 1j * magnitude, -magnitude + 1j * (magnitude / 2)])
    np.testing.assert_allclose(cosine(a, a), 1.0, atol=1e-15)
    np.testing.assert_allclose(cosine(a, 1j * a), 0.0, atol=1e-15)
    np.testing.assert_allclose(cosine(a, -a), -1.0, atol=1e-15)


def test_cosine_preserves_zero_vector_and_ordinary_reference_values():
    assert cosine([0, 0], [1, 2]) == 0.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 2], [1, 2]) == pytest.approx(1.0)
    assert cosine([1, 2], [-1, -2]) == pytest.approx(-1.0)
    rng = np.random.default_rng(33)
    for _ in range(20):
        a = rng.normal(size=7) + 1j * rng.normal(size=7)
        b = rng.normal(size=7) + 1j * rng.normal(size=7)
        expected = np.vdot(a, b).real / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cosine(a, b) == pytest.approx(expected, abs=2e-16)


@pytest.mark.parametrize(
    "left,right",
    [
        ([1, 2], [[1, 2]]),
        ([1, 2], [1]),
        ([1, np.nan], [1, 2]),
        ([1, np.inf], [1, 2]),
    ],
)
def test_cosine_keeps_shape_and_finite_input_validation(left, right):
    with pytest.raises(ValueError):
        cosine(left, right)


def test_signature_rejects_complex_real_trajectory_without_cast_warning():
    trajectory = np.ones((6, 2), dtype=np.complex128) * (1 + 2j)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises((TypeError, ValueError)):
            Signature.from_trajectory(trajectory)
