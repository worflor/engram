"""Reflection-coordinate generation and independent rounded-root checks."""

import numpy as np
import pytest
from engram_stability import schur_coordinates, schur_coefficients, blend_schur
from engram_network import WellNetwork
from engram_analysis import Signature


def radius(k, g):
    return max(abs(np.roots([1, -complex(k), complex(g)])))


def test_audited_unstable_average_has_stable_reflection_blend():
    root = 0.995 + 0.01j
    k = np.array([[2 * root], [2 * root.conjugate()]])
    g = np.array([[root**2], [root.conjugate() ** 2]])
    assert radius(k.mean(), g.mean()) > 1
    bk, bg = blend_schur(k, g, [1, 1])
    assert radius(bk[0], bg[0]) < 1
    np.testing.assert_allclose(
        radius(bk[0], bg[0]), np.sqrt((root**2).real), atol=1e-12
    )


@pytest.mark.parametrize("precision", [np.complex64, np.complex128])
def test_random_convex_blends_against_polynomial_roots(precision):
    rng = np.random.default_rng(84)
    roots = rng.uniform(0, 0.98, (4, 250, 2)) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, (4, 250, 2))
    )
    k, g = roots.sum(axis=2), roots.prod(axis=2)
    reflection, products = schur_coordinates(k, g)
    rk, rg = schur_coefficients(reflection, products)
    np.testing.assert_allclose(rk, k, atol=1e-12)
    np.testing.assert_allclose(rg, g, atol=1e-12)
    output = blend_schur(k, g, rng.random(4), dtype=precision)
    assert all(radius(a, b) < 1 for a, b in zip(*output))
    assert output[0].dtype == precision


@pytest.mark.parametrize("k,g", [(2, 1), (1, 0), (2.2, 1.21), (0, 1)])
def test_boundary_and_unstable_modes_are_rejected(k, g):
    with pytest.raises(ValueError):
        schur_coordinates(k, g)


def test_generation_view_preserves_learned_memory_and_refuses_switching():
    root = 0.995 + 0.01j
    network = WellNetwork(2)
    for name, r in [("a", root), ("b", root.conjugate())]:
        network.learn_signature(
            Signature(
                np.array([2 * r]),
                np.array([r * r]),
                np.array([0.01]),
                np.arange(2),
                20,
                2,
            ),
            well=name,
        )
    before = {n: a.copy() for n, a in network.memory._well_sum_K.items()}
    counts = network.memory._well_count.copy()
    result = network.dream(
        np.array([[0.0, 0.0], [1.0, 0.0]]), blend={"a": 1, "b": 1}, steps=30
    )
    assert result.itinerary == () and result.original_radius < 1
    assert result.mixture == (("a", 0.5), ("b", 0.5))
    assert np.isfinite(result.trajectory).all()
    assert counts == network.memory._well_count
    for name, value in before.items():
        np.testing.assert_array_equal(value, network.memory._well_sum_K[name])
    with pytest.raises(ValueError):
        network.dream([[0, 0], [1, 0]], blend={"a": 1}, walk_every=5)
    with pytest.raises(ValueError):
        network.dream([[0, 0], [1, 0]], blend={"a": 1}, well="a")


def test_invalid_weights_margins_and_rounding_are_not_silently_repaired():
    for weights in [[0, 0], [-1, 1], [np.nan, 1]]:
        with pytest.raises(ValueError):
            blend_schur([[0.2], [0.3]], [[0.1], [0.2]], weights)
    with pytest.raises(ValueError):
        schur_coordinates(0.2, 0.1, margin=-1)
    with pytest.raises(ValueError):
        blend_schur([[0.2]], [[0.1]], [1], dtype=np.float32)
    k, g = blend_schur([[0.2], [3]], [[0.1], [2]], [1, 0])
    assert radius(k[0], g[0]) < 1


def test_float32_rounding_to_the_stability_boundary_is_rejected():
    k, g = [[1.49999999]], [[0.499999995]]
    bk, bg = blend_schur(k, g, [1], margin=0)
    assert radius(bk[0], bg[0]) < 1
    with pytest.raises((ValueError, FloatingPointError)):
        blend_schur(k, g, [1], margin=0, dtype=np.complex64)


@pytest.mark.parametrize("bound", [0.05, 0.5, 0.9, 1.0, 1.5])
@pytest.mark.parametrize("precision", [np.complex64, np.complex128])
def test_requested_root_disk_and_normalized_coefficients(bound, precision):
    roots = np.array([[0.6 + 0.2j, 0.3 - 0.1j], [-0.4 + 0.3j, 0.2 + 0.1j]])
    normalized_k, normalized_g = roots.sum(axis=1)[:, None], roots.prod(axis=1)[:, None]
    k, g = normalized_k * bound, normalized_g * bound * bound
    reflection, product = schur_coordinates(k, g, radius=bound)
    restored_k, restored_g = schur_coefficients(reflection, product, radius=bound)
    np.testing.assert_allclose(restored_k, k, atol=1e-14)
    np.testing.assert_allclose(restored_g, g, atol=1e-14)
    mixed_k, mixed_g = blend_schur(k, g, [1, 3], radius=bound, dtype=precision)
    assert radius(mixed_k[0], mixed_g[0]) < bound
    expected_k, expected_g = blend_schur(normalized_k, normalized_g, [1, 3])
    np.testing.assert_allclose(mixed_k, expected_k * bound, rtol=2e-6)
    np.testing.assert_allclose(mixed_g, expected_g * bound * bound, rtol=2e-6)


def test_radius_is_an_admissibility_contract_not_automatic_clipping():
    with pytest.raises(ValueError):
        blend_schur([[1.6]], [[0.64]], [1], radius=0.75)
    k, g = blend_schur([[1.0]], [[0.0]], [1], radius=1.2)
    assert (
        radius(k[0], g[0]) == 1
    )  # Larger disks explicitly permit boundary/growing modes.
    for invalid in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            schur_coordinates(0.2, 0.1, radius=invalid)
    for bound in [1e-200, 1e200]:
        k, g = blend_schur([[0.0]], [[0.0]], [1], radius=bound)
        assert k[0] == g[0] == 0
    with pytest.raises(FloatingPointError):
        schur_coefficients(0.9, 0.9, radius=1e-200)
    with pytest.raises(FloatingPointError):
        schur_coefficients(0.5, 0.5, radius=1e200)
    with pytest.raises((ValueError, FloatingPointError)):
        blend_schur([[5e199]], [[0.0]], [1], radius=1e200, dtype=np.complex64)


def test_scaled_well_generation_records_its_disk_and_preserves_statistics():
    network = WellNetwork(2)
    signature = Signature(
        np.array([0.7]), np.array([0.1]), np.zeros(1), np.arange(2), 20, 2
    )
    network.learn_signature(signature, well="motion")
    before = network.memory._well_sum_K["motion"].copy()
    dream = network.dream(
        [[0, 0], [1, 0]], blend={"motion": 1}, blend_radius=0.8, steps=12
    )
    assert dream.original_radius < 0.8 and dream.mixture_radius == 0.8
    assert dream.mixture == (("motion", 1.0),) and dream.itinerary == ()
    np.testing.assert_array_equal(before, network.memory._well_sum_K["motion"])
    with pytest.raises(ValueError):
        network.dream([[0, 0], [1, 0]], blend_radius=0.8)
    blend_schur([[0.749999995]], [[0.12499999875]], [1], radius=0.5, margin=0)
    with pytest.raises(ValueError):
        blend_schur(
            [[0.749999995]],
            [[0.12499999875]],
            [1],
            radius=0.5,
            margin=0,
            dtype=np.complex64,
        )
