"""Focused exact-protocol and frozen-readout tests for recency decoding."""

import numpy as np
import pytest

from research.experiments import relational_memory as rm
from research.experiments import recency_decoding as rd


@pytest.mark.parametrize(
    "seed,label,shifted", [(7, 1, False), (19, -1, True), (307, 1, False)]
)
def test_bank_features_exactly_match_existing_relational_memory_protocol(
    seed, label, shifted
):
    rng = np.random.default_rng(seed)
    trajectory = rm.trajectory("recent_pulse", label, rng, shifted=shifted)
    expected = rm.representations(trajectory)["temporal_bank"]
    actual = rd.temporal_bank_features(trajectory)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (48,)


def test_rms_fast_slow_ratio_is_amplitude_invariant_for_exact_pulse_fixture():
    rng = np.random.default_rng(81)
    trajectory = rm.trajectory("recent_pulse", 1, rng)
    first = rd.temporal_bank_features(trajectory)
    second = rd.temporal_bank_features(trajectory * 1.7)
    ratio_first = rd.append_rms_ratio(first[None, :])[0, 48:]
    ratio_second = rd.append_rms_ratio(second[None, :])[0, 48:]
    np.testing.assert_allclose(ratio_first, ratio_second, rtol=2e-6, atol=2e-7)


def test_derived_view_adds_only_four_rms_ratios_and_no_history_metadata():
    rng = np.random.default_rng(211)
    trajectory = rm.trajectory("recent_pulse", -1, rng)
    bank = rd.temporal_bank_features(trajectory)
    augmented = rd.append_rms_ratio(bank[None, :])
    assert bank.shape == (48,)
    assert augmented.shape == (1, 52)
    np.testing.assert_array_equal(augmented[0, :48], bank)
    np.testing.assert_array_equal(bank[:8], trajectory[-1])


def test_linear_predictions_do_not_change_when_test_rows_are_appended():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(32, 6))
    labels = np.where(train[:, 0] > 0, 1, -1)
    validation = rng.normal(size=(16, 6))
    validation_labels = np.where(validation[:, 0] > 0, 1, -1)
    test = rng.normal(size=(9, 6))
    extra = rng.normal(size=(13, 6))
    first = rm.readout(train, labels, validation, validation_labels, test)[0]
    appended = rm.readout(
        train, labels, validation, validation_labels, np.vstack([test, extra])
    )[0][: len(test)]
    np.testing.assert_array_equal(first, appended)


def test_rbf_predictions_are_frozen_and_test_rows_do_not_set_scaling():
    rng = np.random.default_rng(419)
    train = rng.normal(size=(32, 5))
    labels = np.where(train[:, 0] + train[:, 1] > 0, 1, -1)
    validation = rng.normal(size=(16, 5))
    validation_labels = np.where(validation[:, 0] + validation[:, 1] > 0, 1, -1)
    test = rng.normal(size=(7, 5))
    extra = rng.normal(size=(11, 5)) * 1e6
    model = rd.fit_rbf(train, labels, validation, validation_labels)
    before = rd.predict_rbf(model, test)
    combined = rd.predict_rbf(model, np.vstack([test, extra]))[: len(test)]
    np.testing.assert_array_equal(before, combined)


def test_rbf_grid_uses_train_median_and_predeclared_candidates():
    train = np.array([[-1.0, 0], [-0.5, 1], [0.5, -1], [1, 0]])
    labels = np.array([-1, -1, 1, 1])
    validation = np.array([[-0.8, 0.2], [0.8, -0.2]])
    model = rd.fit_rbf(train, labels, validation, np.array([-1, 1]))
    assert model["gamma_factor"] in rd.RBF_GAMMA_FACTORS
    assert model["ridge"] in rd.RIDGES
    centered = (train - train.mean(axis=0)) / np.where(
        train.std(axis=0) < 1e-8, 1, train.std(axis=0)
    )
    distances = rd._sqdistances(centered, centered)[np.triu_indices(len(train), 1)]
    assert model["median_pairwise_squared_distance"] == np.median(
        distances[distances > 0]
    )
