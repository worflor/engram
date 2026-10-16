"""Independent affine least squares, coordinate symmetry and causal continuation."""

import numpy as np
import pytest

from engram_affine import AffineOscillator
from engram_analysis import Signature
from engram_npc import WitnessHistory


def test_unpenalized_intercept_matches_augmented_complex_least_squares():
    x = np.random.default_rng(12).normal(size=(60, 6)) + [2, 3, -5, 7, 1, -4]
    pairing = np.array([5, 0, 3, 1, 4, 2])
    model = AffineOscillator.fit(x, pairing=pairing, relative_ridge=0)
    paired = x[:, pairing]
    z = paired[:, ::2] + 1j * paired[:, 1::2]
    for p in range(3):
        design = np.column_stack([z[1:-1, p], -z[:-2, p], np.ones(len(z) - 2)])
        expected = np.linalg.lstsq(design, z[2:, p], rcond=None)[0]
        np.testing.assert_allclose(
            [model.signature.k[p], model.signature.g[p], model.bias[p]],
            expected,
            atol=1e-12,
        )
        error = z[2:, p] - design @ expected
        np.testing.assert_allclose(
            model.signature.pair_rms[p], np.sqrt(np.mean(abs(error) ** 2)), atol=1e-12
        )


def test_ridge_penalizes_coefficients_and_leaves_intercept_free():
    x = np.random.default_rng(72).normal(size=(30, 4)) + 100
    model = AffineOscillator.fit(x, relative_ridge=0.2)
    z = x[:, ::2] + 1j * x[:, 1::2]
    for p in range(2):
        design = np.column_stack([z[1:-1, p], -z[:-2, p], np.ones(len(z) - 2)])
        centered = design[:, :2] - design[:, :2].mean(axis=0)
        ridge = 0.2 * np.sum(abs(centered) ** 2) / 2
        augmented = np.vstack([design, np.diag([np.sqrt(ridge), np.sqrt(ridge), 0])])
        target = np.r_[z[2:, p], np.zeros(3)]
        expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
        np.testing.assert_allclose(
            [model.signature.k[p], model.signature.g[p], model.bias[p]],
            expected,
            atol=1e-10,
        )


def test_exact_rational_oscillator_with_shift_and_real_zero_companion():
    x = np.array([1, 0, -9 / 25, -27 / 125, 0, 243 / 3125])
    model = AffineOscillator.fit(
        np.column_stack([x[:5] + 7, np.zeros(5)]), relative_ridge=0
    )
    np.testing.assert_allclose(model.signature.k, 3 / 5, atol=1e-12)
    np.testing.assert_allclose(model.signature.g, 9 / 25, atol=1e-12)
    np.testing.assert_allclose(model.bias, 133 / 25, atol=1e-12)
    np.testing.assert_allclose(model.forecast(1), [[x[5] + 7, 0]], atol=1e-12)


@pytest.mark.parametrize("ridge", [0, 1e-4, 0.1])
def test_refitting_and_translating_preserve_dynamics_and_forecasts(ridge):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 4))
    shift = np.array([1e6, -1e6, 32, -5])
    base = AffineOscillator.fit(x, pairing=[3, 1, 0, 2], relative_ridge=ridge)
    refitted = AffineOscillator.fit(
        x + shift, pairing=[3, 1, 0, 2], relative_ridge=ridge
    )
    moved = base.translated(shift)
    np.testing.assert_allclose(base.signature.k, refitted.signature.k, atol=1e-10)
    np.testing.assert_allclose(base.signature.g, refitted.signature.g, atol=1e-10)
    np.testing.assert_allclose(
        base.signature.pair_rms, refitted.signature.pair_rms, atol=1e-10
    )
    np.testing.assert_allclose(
        refitted.forecast(40) - shift, base.forecast(40), atol=2e-9
    )
    np.testing.assert_allclose(moved.forecast(40) - shift, base.forecast(40), atol=2e-9)
    dz = shift[[3, 0]] + 1j * shift[[1, 2]]
    np.testing.assert_allclose(
        moved.bias - base.bias,
        dz * (1 - base.signature.k + base.signature.g),
        atol=1e-9,
    )


def test_constant_stream_and_seed_order_are_explicit():
    model = AffineOscillator.fit(np.tile([8.0, -3.0], (10, 1)))
    np.testing.assert_allclose(model.forecast(100), np.tile([8.0, -3.0], (100, 1)))
    assert model.forecast(0).shape == (0, 2)
    assert model.signature.k[0] == model.signature.g[0] == 0
    assert not model.seeds.flags.writeable and not model.signature.k.flags.writeable
    with pytest.raises(ValueError, match="singular"):
        AffineOscillator.fit(np.ones((10, 2)), relative_ridge=0)
    np.testing.assert_array_equal(model.forecast(4), model.forecast(4))


@pytest.mark.parametrize(
    "x", [np.ones((4, 2)), np.ones((10, 3)), np.full((10, 2), np.nan)]
)
def test_invalid_affine_observations_are_rejected(x):
    with pytest.raises(ValueError):
        AffineOscillator.fit(x)


def test_prediction_uses_supplied_seeds_and_original_channel_order():
    x = np.random.default_rng(42).normal(size=(50, 4))
    model = AffineOscillator.fit(x, pairing=[2, 0, 3, 1])
    seeds = x[:2]
    paired = seeds[:, model.signature.pairing]
    z = paired[:, ::2] + 1j * paired[:, 1::2]
    expected = model.signature.k * z[1] - model.signature.g * z[0] + model.bias
    out = model.forecast(1, seeds=seeds)[0, model.signature.pairing]
    np.testing.assert_allclose(out[::2] + 1j * out[1::2], expected, atol=1e-12)
    for invalid in [-1, 1.5, True]:
        with pytest.raises(ValueError):
            model.forecast(invalid)
    with pytest.raises(ValueError):
        model.translated([1, 2])


def test_growing_roots_are_explicit_and_numeric_overflow_is_rejected():
    def growing(k):
        signature = Signature(
            np.array([k]), np.array([0]), np.zeros(1), np.arange(2), 5, 2
        )
        return AffineOscillator(signature, np.zeros((1, 3)), np.array([[1, 0], [2, 0]]))

    np.testing.assert_allclose(growing(2).forecast(4)[:, 0], [4, 8, 16, 32])
    with pytest.raises(FloatingPointError):
        growing(1e308).forecast(3)


@pytest.mark.parametrize("decay", [1.0, 0.995, 0.5])
def test_affine_history_snapshot_matches_independent_weighted_problem(decay):
    x = np.random.default_rng(83).normal(size=(300, 4)) + [10, -7, 100, -50]
    history = WitnessHistory(4, decay)
    for row in x:
        history.push(row)
    model = AffineOscillator.from_history(history)
    z = x[:, ::2] + 1j * x[:, 1::2]
    weights = decay ** np.arange(len(x) - 3, -1, -1)
    for p in range(2):
        design = np.column_stack([z[1:-1, p], -z[:-2, p], np.ones(len(z) - 2)])
        centered = (
            design[:, :2]
            - np.sum(weights[:, None] * design[:, :2], axis=0) / weights.sum()
        )
        ridge = max(
            float(np.finfo(np.float32).eps),
            1e-4 * np.sum(weights[:, None] * abs(centered) ** 2) / 2,
        )
        augmented = np.vstack(
            [
                np.sqrt(weights[:, None]) * design,
                np.diag([np.sqrt(ridge), np.sqrt(ridge), 0]),
            ]
        )
        expected = np.linalg.lstsq(
            augmented, np.r_[np.sqrt(weights) * z[2:, p], np.zeros(3)], rcond=None
        )[0]
        np.testing.assert_allclose(
            [model.signature.k[p], model.signature.g[p], model.bias[p]],
            expected,
            atol=1e-9,
        )
        residual = z[2:, p] - design @ expected
        np.testing.assert_allclose(
            model.signature.pair_rms[p],
            np.sqrt(np.sum(weights * abs(residual) ** 2) / weights.sum()),
            atol=1e-10,
        )
    previous = model.forecast(20)
    history.push([0, 0, 0, 0])
    np.testing.assert_array_equal(model.forecast(20), previous)
