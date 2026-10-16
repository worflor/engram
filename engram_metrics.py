"""Discriminating trajectory evaluation, including the portfolio audit's DC control."""

import numpy as np


def forecast_metrics(actual, predicted, *, last_observation):
    """Compare held-out positions with prediction and a persistence baseline.

    Centered R² may be negative. A constant target has undefined R² (None),
    rather than turning an absent dynamic signal into a perfect score.
    """
    y, p, last = (
        np.asarray(a, dtype=np.float64) for a in [actual, predicted, last_observation]
    )
    if (
        y.ndim != 2
        or not y.size
        or p.shape != y.shape
        or last.shape != y.shape[1:]
        or not all(np.isfinite(a).all() for a in [y, p, last])
    ):
        raise ValueError("equal finite nonempty trajectories and one seed row required")
    error = float(np.mean((y - p) ** 2))
    variance = float(np.mean((y - y.mean(axis=0)) ** 2))
    energy = float(np.mean(y * y))
    return {
        "rmse": float(np.sqrt(error)),
        "persistence_rmse": float(np.sqrt(np.mean((y - last) ** 2))),
        "mean_baseline_rmse": float(np.sqrt(variance)),
        "centered_r2": 1 - error / variance if variance > 0 else None,
        "energy_capture": 1 - error / energy if energy > 0 else None,
    }


def permutation_test(left, right, *, rng, repeats=999):
    """Paired-identity correlation against a shuffled-right null (no rank pairing).

    Inputs must retain the ORIGINAL entity correspondence. Sorting each side
    independently destroys that correspondence and invalidates this experiment.
    """
    a, b = np.asarray(left, float), np.asarray(right, float)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or len(a) < 3
        or not isinstance(repeats, int)
        or repeats < 1
        or not np.isfinite([a, b]).all()
        or np.std(a) == 0
        or np.std(b) == 0
    ):
        raise ValueError(
            "nonconstant paired finite vectors and positive repeats required"
        )
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    observed = float(a @ b / denominator)
    null = np.array([a @ rng.permutation(b) / denominator for _ in range(repeats)])
    return {
        "correlation": observed,
        "null_mean": float(null.mean()),
        "p_value": float((1 + np.sum(abs(null) >= abs(observed))) / (repeats + 1)),
        "repeats": repeats,
    }
