"""Discriminating controls for the experimental relational representations."""

import numpy as np
import pytest

from research.experiments import relational_memory as rm
from research.experiments import relational_topology as rt


def paired(task):
    return [rm.trajectory(task, label, np.random.default_rng(7)) for label in [-1, 1]]


def test_phase_relation_is_lost_by_independent_fits_but_not_cross_moments():
    a, b = map(rm.representations, paired("phase_coupling"))
    np.testing.assert_allclose(a["ar2"], b["ar2"], atol=1e-12)
    np.testing.assert_allclose(a["temporal_bank"], b["temporal_bank"], atol=1e-12)
    assert np.linalg.norm(a["moments"] - b["moments"]) > 0.1


def test_conditional_parity_defeats_all_pooled_second_moments():
    a, b = map(rm.representations, paired("conditional_parity"))
    np.testing.assert_allclose(a["moments"], b["moments"], atol=1e-12)
    assert np.linalg.norm(a["role_fibers"] - b["role_fibers"]) > 0.1


def test_existing_exact_pulse_alias_and_bank_separation():
    x = np.zeros((6, 8))
    y = x.copy()
    x[3, 2], y[2, 2] = 1, 1
    a, b = rm.representations(x), rm.representations(y)
    np.testing.assert_allclose(a["ar2"], b["ar2"], atol=1e-12)
    assert np.linalg.norm(a["temporal_bank"] - b["temporal_bank"]) > 0.01


def test_float_path_features_match_independent_exact_tensor_coordinates():
    increments = ((1, 0), (0, 1), (0, 1), (1, 0))
    positions = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)])
    exact = rt.path_signature(increments)
    np.testing.assert_allclose(
        rm.path_features(positions), [float(v) for level in exact[1:] for v in level]
    )
    subdivided = np.vstack(
        [
            positions[0],
            *[
                point
                for a, b in zip(positions[:-1], positions[1:])
                for point in [(a + b) / 2, b]
            ],
        ]
    )
    np.testing.assert_allclose(
        rm.path_features(positions), rm.path_features(subdivided), atol=1e-12
    )


def test_readout_does_not_calibrate_on_other_test_rows():
    train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([-1, -1, 1, 1])
    p, ridge, _, _ = rm.readout(train, labels, train, labels, np.array([[0.5]]))
    q, other_ridge, _, _ = rm.readout(
        train, labels, train, labels, np.array([[0.5], [1e12]])
    )
    assert p[0] == q[0] == 1
    assert ridge == other_ridge


@pytest.mark.parametrize("task", rm.TASKS)
def test_generator_is_reproducible_finite_and_keeps_targets_external(task):
    for label in [-1, 1]:
        x = rm.trajectory(task, label, np.random.default_rng(91), shifted=True)
        np.testing.assert_array_equal(
            x, rm.trajectory(task, label, np.random.default_rng(91), shifted=True)
        )
        assert x.shape[1] == 8 and np.isfinite(x).all()
        if task != "current_control":
            np.testing.assert_array_equal(x[-1], np.zeros(8))


@pytest.mark.parametrize(
    "x", [np.zeros((2, 8)), np.zeros((3, 7)), np.full((3, 8), np.nan)]
)
def test_bad_research_trajectories_reject(x):
    with pytest.raises(ValueError):
        rm.representations(x)
