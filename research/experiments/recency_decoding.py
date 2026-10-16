"""Readout-accessibility study for Engram's existing temporal bank.

This holds the recent-pulse generator, splits, histories, and bank features
fixed. It compares a linear head with standardized Gaussian RBF kernel ridge,
and adds an RMS fast/slow ratio derived from the same bank. Readouts are
validation-selected; test rows never set scaling or hyperparameters. This is an
authored task, not a general memory-capacity result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engram_history import WitnessHistory
from research.experiments import relational_memory as rm


SEEDS = (7, 19, 81, 211, 307, 419)
TRAINING = 96
VALIDATION = 48
EVALUATION = 96
RBF_GAMMA_FACTORS = (0.1, 1.0, 10.0)
RIDGES = (0.01, 1.0, 100.0)


def temporal_bank_features(trajectory) -> np.ndarray:
    """Return exactly the existing current + cumulative/fast AR(2) bank."""
    x = np.asarray(trajectory, dtype=float)
    if x.ndim != 2 or x.shape[1] != 8 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("finite eight-channel research trajectory required")
    signatures = []
    for decay in (1.0, 0.9):
        history = WitnessHistory(8, decay)
        for row in x:
            signature = history.push(row)
        signatures.append(
            np.r_[
                signature.k.real,
                signature.k.imag,
                signature.g.real,
                signature.g.imag,
                np.log1p(signature.pair_rms),
            ]
        )
    return np.concatenate([x[-1].copy(), *signatures])


def append_rms_ratio(bank_features) -> np.ndarray:
    """Append fast/slow residual-RMS ratios already recoverable from the bank."""
    features = np.asarray(bank_features, dtype=float)
    if features.ndim != 2 or features.shape[1] != 48 or not np.isfinite(features).all():
        raise ValueError("finite temporal-bank matrix with 48 columns required")
    slow = np.expm1(features[:, 24:28])
    fast = np.expm1(features[:, 44:48])
    ratio = np.divide(fast, slow, out=np.zeros_like(fast), where=slow != 0)
    if not np.isfinite(ratio).all():
        raise FloatingPointError("derived RMS ratio exceeded finite range")
    return np.concatenate([features, ratio], axis=1)


def _standardize(train, *others):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1
    return mean, scale, tuple((rows - mean) / scale for rows in others)


def _sqdistances(a, b):
    delta = a[:, None, :] - b[None, :, :]
    return np.maximum(np.einsum("ijk,ijk->ij", delta, delta), 0.0)


def fit_rbf(train, labels, validation, validation_labels):
    """Validation-select a Gaussian KRR readout using train-only scaling."""
    mean, scale, (a, v) = _standardize(train, train, validation)
    pairwise = _sqdistances(a, a)
    upper = pairwise[np.triu_indices(len(a), k=1)]
    nonzero = upper[upper > 0]
    median_distance = float(np.median(nonzero)) if len(nonzero) else 1.0
    if not np.isfinite(median_distance) or median_distance <= 0:
        median_distance = 1.0
    candidates = []
    for gamma_factor in RBF_GAMMA_FACTORS:
        gamma = gamma_factor / median_distance
        kernel_train = np.exp(-gamma * pairwise)
        kernel_validation = np.exp(-gamma * _sqdistances(v, a))
        for ridge in RIDGES:
            alpha = np.linalg.solve(kernel_train + ridge * np.eye(len(train)), labels)
            validation_scores = kernel_validation @ alpha
            accuracy = float(
                np.mean(np.where(validation_scores >= 0, 1, -1) == validation_labels)
            )
            candidates.append((accuracy, -ridge, -gamma_factor, gamma, ridge, alpha))
    best = max(candidates, key=lambda row: row[:3])
    model = {
        "mean": mean,
        "scale": scale,
        "train_scaled": a.copy(),
        "gamma": float(best[3]),
        "ridge": float(best[4]),
        "alpha": best[5].copy(),
        "gamma_factor": float(-best[2]),
        "median_pairwise_squared_distance": median_distance,
        "validation_accuracy": float(best[0]),
    }
    retained_bytes = int(
        model["mean"].nbytes
        + model["scale"].nbytes
        + model["train_scaled"].nbytes
        + model["alpha"].nbytes
        + 3 * np.dtype(np.float64).itemsize
    )
    model["readout_bytes"] = retained_bytes
    return model


def predict_rbf(model, rows) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    scaled = (rows - model["mean"]) / model["scale"]
    scores = np.exp(-model["gamma"] * _sqdistances(scaled, model["train_scaled"]))
    return np.where(scores @ model["alpha"] >= 0, 1, -1)


def _datasets_for_seed(seed: int):
    rng = np.random.default_rng(seed)
    feature_splits, labels_splits = [], []
    for count, shifted in (
        (TRAINING, False),
        (VALIDATION, False),
        (EVALUATION, False),
        (EVALUATION, True),
    ):
        labels = np.tile([-1, 1], count // 2)
        rng.shuffle(labels)
        rows = [
            temporal_bank_features(
                rm.trajectory("recent_pulse", int(label), rng, shifted=shifted)
            )
            for label in labels
        ]
        feature_splits.append(np.stack(rows))
        labels_splits.append(labels)
    return feature_splits, labels_splits


def _prediction_hash(predictions) -> str:
    canonical = np.asarray(predictions, dtype=np.int8).tobytes()
    return hashlib.sha256(canonical).hexdigest()


def campaign() -> dict:
    records = []
    for seed in SEEDS:
        splits, labels = _datasets_for_seed(seed)
        base = splits[0]
        augmented = [append_rms_ratio(rows) for rows in splits]
        heads = {}
        for name, data in (
            ("temporal_bank_linear", splits),
            ("rms_ratio_linear", augmented),
        ):
            prediction, ridge, validation_accuracy, readout_bytes = rm.readout(
                data[0], labels[0], data[1], labels[1], data[2]
            )
            # Re-evaluate the shifted test with the same selected linear head;
            # rm.readout is deterministic and the test argument cannot affect fit.
            shifted_prediction = rm.readout(
                data[0], labels[0], data[1], labels[1], data[3]
            )[0]
            heads[name] = {
                "predictions": (prediction, shifted_prediction),
                "ridge": ridge,
                "validation_accuracy": validation_accuracy,
                "readout_bytes": readout_bytes,
            }
        for name, data in (("temporal_bank_rbf", splits), ("rms_ratio_rbf", augmented)):
            model = fit_rbf(data[0], labels[0], data[1], labels[1])
            heads[name] = {
                "predictions": (
                    predict_rbf(model, data[2]),
                    predict_rbf(model, data[3]),
                ),
                "ridge": model["ridge"],
                "gamma_factor": model["gamma_factor"],
                "gamma": model["gamma"],
                "median_pairwise_squared_distance": model[
                    "median_pairwise_squared_distance"
                ],
                "validation_accuracy": model["validation_accuracy"],
                "readout_bytes": model["readout_bytes"],
            }
        for name, head in heads.items():
            for index, condition in ((0, "independent"), (1, "longer_and_rescaled")):
                predicted = head["predictions"][index]
                records.append(
                    {
                        "seed": seed,
                        "task": "recent_pulse",
                        "readout": name,
                        "condition": condition,
                        "accuracy": float(np.mean(predicted == labels[index + 2])),
                        "cases": EVALUATION,
                        "validation_accuracy": float(head["validation_accuracy"]),
                        "ridge": float(head["ridge"]),
                        "gamma_factor": head.get("gamma_factor"),
                        "gamma": head.get("gamma"),
                        "median_pairwise_squared_distance": head.get(
                            "median_pairwise_squared_distance"
                        ),
                        "feature_scalars": int(
                            augmented[0].shape[1]
                            if name.startswith("rms_ratio")
                            else base.shape[1]
                        ),
                        "feature_bytes": int(
                            (
                                augmented[0].shape[1]
                                if name.startswith("rms_ratio")
                                else base.shape[1]
                            )
                            * np.dtype(np.float64).itemsize
                        ),
                        "readout_bytes": int(head["readout_bytes"]),
                        "prediction_sha256": _prediction_hash(predicted),
                    }
                )

    aggregate = []
    for readout in (
        "temporal_bank_linear",
        "temporal_bank_rbf",
        "rms_ratio_linear",
        "rms_ratio_rbf",
    ):
        for condition in ("independent", "longer_and_rescaled"):
            selected = [
                row
                for row in records
                if row["readout"] == readout and row["condition"] == condition
            ]
            aggregate.append(
                {
                    "readout": readout,
                    "condition": condition,
                    "accuracy": float(np.mean([row["accuracy"] for row in selected])),
                    "minimum_seed_accuracy": min(row["accuracy"] for row in selected),
                    "cases": sum(row["cases"] for row in selected),
                    "feature_scalars": selected[0]["feature_scalars"],
                    "feature_bytes": selected[0]["feature_bytes"],
                }
            )
    sources = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in (
            "research/experiments/recency_decoding.py",
            "research/experiments/relational_memory.py",
            "engram_history.py",
            "engram_analysis.py",
            "engram_codec.py",
        )
    }
    return {
        "schema": "engram.recency-decoding/v1",
        "seeds": list(SEEDS),
        "task": "recent_pulse",
        "split_sizes": {
            "training": TRAINING,
            "validation": VALIDATION,
            "independent_test": EVALUATION,
            "longer_and_rescaled_test": EVALUATION,
        },
        "readout_protocol": {
            "linear": "relational_memory.readout; train-only standardization and validation-selected ridge from {0.01,1,100}.",
            "rbf": "Gaussian kernel ridge; train-only standardization, gamma=multiplier/median positive pairwise squared train distance; validation selects multipliers {0.1,1,10} and ridge {0.01,1,100}.",
            "rms_ratio": "Append expm1(fast log-RMS)/expm1(slow log-RMS) per complex pair; zero exactly where slow RMS is zero. No history or new samples are added.",
            "test_isolation": "Test rows are only passed to a frozen model for prediction; they do not set scaling, distance scale, gamma, ridge, or model coefficients.",
        },
        "sources": sources,
        "scope": "This compares readout accessibility for one authored recent-pulse generator. A gain shows accessibility for these tested readouts and splits; a failure does not prove information is absent from the retained feature.",
        "feature_scope": "All readouts receive the same current vector and existing cumulative/0.9 WitnessHistory signatures. RMS ratios are a deterministic view of those same signature values. No raw history, window length, sample count, or age token is added.",
        "aggregate": aggregate,
        "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/recency-decoding.json"
    )
    args = parser.parse_args()
    report = campaign()
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
