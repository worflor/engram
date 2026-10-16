"""Controlled relational-memory ablations; no claimed natural-language semantics.

The same linear outcome readout, training-only scaling and validation-selected
ridge are used for every representation. Generators declare their relations;
test streams are independently drawn and include amplitude/timing changes.
Fibers use an OBSERVED role channel, never a target or inferred hidden identity.
These are research sketches, not an extension of the supported checkpoint ABI.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engram_history import WitnessHistory

TASKS = (
    "phase_coupling",
    "event_order",
    "role_binding",
    "recent_pulse",
    "current_control",
    "conditional_parity",
)


def trajectory(task, label, rng, *, shifted=False):
    """Eight fixed named coordinates; labels are returned separately by caller."""
    n = int(rng.choice([32, 40, 48] if not shifted else [56, 64, 72]))
    x = np.zeros((n, 8))
    amplitude = rng.uniform(0.7, 1.3) if not shifted else rng.uniform(0.35, 1.8)
    t = np.arange(n) / n
    if task == "phase_coupling":
        z = amplitude * np.exp(
            1j * (2 * np.pi * rng.integers(2, 5) * t + rng.uniform(-np.pi, np.pi))
        )
        x[:, 2], x[:, 3] = z.real, z.imag
        x[:, 4], x[:, 5] = label * z.real, label * z.imag
    elif task == "event_order":
        a, b = (0.25, 0.65) if label > 0 else (0.65, 0.25)
        width = rng.uniform(0.035, 0.07)
        x[:, 2] = amplitude * np.exp(-(((t - a) / width) ** 2))
        x[:, 4] = amplitude * np.exp(-(((t - b) / width) ** 2))
    elif task in {"role_binding", "conditional_parity"}:
        events = [(r, c) for r in [-1, 1] for c in [-1, 1]]
        rng.shuffle(events)
        for indexes, (role, cue) in zip(np.array_split(np.arange(n), 4), events):
            x[indexes, 0] = role
            x[indexes, 2] = amplitude * cue
            x[indexes, 4] = (
                amplitude * label * role * (cue if task == "conditional_parity" else 1)
            )
        # Append, rather than replace an event: the four role/cue groups must
        # remain balanced for the exact pooled-moment collision.
        return np.vstack([x, np.zeros(8)])
    elif task == "recent_pulse":
        n = int(rng.integers(6, 13) if not shifted else rng.integers(13, 25))
        x = np.zeros((n, 8))
        x[-3 if label > 0 else -4, 2] = amplitude
    elif task == "current_control":
        x[:, 2] = rng.normal(0, 0.3, n)
        x[-1, 2] = label * amplitude
        return x
    else:
        raise ValueError(task)
    # Common endpoint removes the answer from the current-only baseline.
    x[-1] = 0
    return x


def moments(x):
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / len(x)
    return np.r_[
        x.mean(axis=0),
        np.diag(covariance) ** 0.5,
        covariance[np.triu_indices(x.shape[1])],
    ]


def correlations(x):
    """Explicit scale-invariant view; zero-variance pairs have value zero.

    This discards absolute levels/amplitudes. It never changes a stored Engram
    schema or makes an existing well's coordinates silently migrate.
    """
    centered = x - x.mean(axis=0)
    scatter = centered.T @ centered
    norm = np.sqrt(np.diag(scatter))
    denominator = norm[:, None] * norm[None, :]
    value = np.divide(
        scatter, denominator, out=np.zeros_like(scatter), where=denominator > 0
    )
    return value[np.triu_indices(x.shape[1])]


def path_features(path):
    """Level 1..3 signature of the piecewise-linear selected coordinate path.

    Segment tensors are exp(dx), concatenated by Chen's identity. No time
    channel: exact subdivision is invariant, durations are intentionally lost.
    """
    d = path.shape[1]
    levels = [np.ones(1), np.zeros(d), np.zeros(d * d), np.zeros(d * d * d)]
    for dx in np.diff(path, axis=0):
        segment = [
            np.ones(1),
            dx,
            np.kron(dx, dx) / 2,
            np.kron(np.kron(dx, dx), dx) / 6,
        ]
        levels = [
            sum(
                (np.kron(levels[j], segment[k - j]) for j in range(k + 1)),
                np.zeros(d**k),
            )
            for k in range(4)
        ]
    return np.concatenate(levels[1:])


def representations(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 8 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("finite eight-channel research trajectory required")
    current = x[-1]
    signatures = []
    for decay in [1.0, 0.9]:
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
    basic = moments(x)
    lag = (x[1:] - x[1:].mean(axis=0)).T @ (x[:-1] - x[:-1].mean(axis=0)) / (len(x) - 1)
    dx = np.diff(x, axis=0)
    relative = x[:-1] - x[0]
    area = (relative.T @ dx - dx.T @ relative) / 2
    fiber, normalized_fiber = [], []
    for role in [-1, 1]:
        rows = x[x[:, 0] == role, 1:]
        fiber.extend(
            np.r_[len(rows) / len(x), moments(rows)] if len(rows) else np.zeros(43)
        )
        normalized_fiber.extend(correlations(rows) if len(rows) else np.zeros(28))
    events = np.flatnonzero(x[:, 2] != 0)
    event_control = (
        [x[events[-1], 2], len(x) - 1 - events[-1]]
        if len(events)
        else [0.0, float(len(x))]
    )
    return {
        "current": current.copy(),
        "ar2": np.r_[current, signatures[0]],
        "temporal_bank": np.concatenate([current, *signatures]),
        "moments": np.r_[current, basic],
        "correlation_relations": np.r_[current, correlations(x)],
        "lag_relations": np.r_[current, basic, lag.ravel()],
        "oriented_area": np.r_[current, basic, area[np.triu_indices(8, 1)]],
        "path_level3": np.r_[
            current, x.mean(axis=0), x.std(axis=0), path_features(x[:, [2, 4]])
        ],
        "role_fibers": np.r_[current, basic, fiber],
        "normalized_role_fibers": np.r_[current, normalized_fiber],
        "last_event_control": np.r_[current, event_control],
    }


def readout(train, labels, validation, validation_labels, test):
    """Train-only units; same linear head and validation grid for all sketches."""
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-8] = 1

    def design(rows):
        return np.c_[np.ones(len(rows)), (rows - mean) / scale]

    a, v, b = map(design, [train, validation, test])
    candidates = []
    for ridge in [0.01, 1.0, 100.0]:
        penalty = np.eye(a.shape[1]) * ridge
        penalty[0, 0] = 0
        coef = np.linalg.solve(a.T @ a + penalty, a.T @ labels)
        score = np.mean(np.where(v @ coef >= 0, 1, -1) == validation_labels)
        candidates.append((score, -ridge, coef))
    best = max(candidates, key=lambda row: row[:2])
    return (
        np.where(b @ best[2] >= 0, 1, -1),
        -best[1],
        float(best[0]),
        int(best[2].nbytes + mean.nbytes + scale.nbytes),
    )


def campaign(*, seeds=(7, 19, 81), training=96, validation=48, evaluation=96):
    if any(n < 2 or n % 2 for n in [training, validation, evaluation]):
        raise ValueError("positive even split sizes required")
    rows = []
    observed = 0
    for seed in seeds:
        for task in TASKS:
            rng = np.random.default_rng(seed)
            split_features, split_labels = [], []
            feature_seconds = 0
            for count, shifted in [
                (training, False),
                (validation, False),
                (evaluation, False),
                (evaluation, True),
            ]:
                labels = np.tile([-1, 1], count // 2)
                rng.shuffle(labels)
                samples = []
                for label in labels:
                    x = trajectory(task, int(label), rng, shifted=shifted)
                    observed += len(x)
                    start = time.perf_counter()
                    samples.append(representations(x))
                    feature_seconds += time.perf_counter() - start
                split_features.append(
                    {
                        name: np.stack([sample[name] for sample in samples])
                        for name in samples[0]
                    }
                )
                split_labels.append(labels)
            for name in split_features[0]:
                for index, condition in [
                    (2, "independent"),
                    (3, "longer_and_rescaled"),
                ]:
                    predicted, ridge, val, size = readout(
                        split_features[0][name],
                        split_labels[0],
                        split_features[1][name],
                        split_labels[1],
                        split_features[index][name],
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "task": task,
                            "representation": name,
                            "condition": condition,
                            "accuracy": float(
                                np.mean(predicted == split_labels[index])
                            ),
                            "cases": evaluation,
                            "validation_accuracy": val,
                            "ridge": ridge,
                            "feature_scalars": int(split_features[0][name].shape[1]),
                            "feature_bytes": int(split_features[0][name].shape[1] * 8),
                            "readout_bytes": size,
                        }
                    )
            print(
                f"seed={seed} task={task} all-feature extraction={feature_seconds:.2f}s",
                flush=True,
            )
    aggregate = []
    for task in TASKS:
        for name in split_features[0]:
            for condition in ["independent", "longer_and_rescaled"]:
                selected = [
                    r
                    for r in rows
                    if (r["task"], r["representation"], r["condition"])
                    == (task, name, condition)
                ]
                aggregate.append(
                    {
                        "task": task,
                        "representation": name,
                        "condition": condition,
                        "accuracy": float(np.mean([r["accuracy"] for r in selected])),
                        "minimum_seed_accuracy": min(r["accuracy"] for r in selected),
                        "cases": sum(r["cases"] for r in selected),
                        "feature_bytes": selected[0]["feature_bytes"],
                    }
                )
    sources = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in [
            "research/experiments/relational_memory.py",
            "engram_history.py",
            "engram_analysis.py",
            "engram_codec.py",
        ]
    }
    return {
        "schema": "engram.relational-memory-study/v1",
        "seeds": list(seeds),
        "training_per_task_seed": training,
        "validation_per_task_seed": validation,
        "test_per_condition_task_seed": evaluation,
        "observed_rows": observed,
        "decision_evaluations": sum(r["cases"] for r in rows),
        "sources": sources,
        "scope": "Authored eight-channel relational tasks; one shared linear readout protocol. Not NPC policy learning, a natural-language benchmark, a Minecraft swarm, or a universal semantic representation.",
        "cost_scope": "Feature bytes count retained dense readout inputs; ingestion temporarily uses the complete research trajectory. This is not an O(D) streaming-memory or latency claim.",
        "aggregate": aggregate,
        "runs": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/relational-memory.json"
    )
    parser.add_argument("--training", type=int, default=96)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--evaluation", type=int, default=96)
    args = parser.parse_args()
    report = campaign(
        training=args.training, validation=args.validation, evaluation=args.evaluation
    )
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({k: report[k] for k in ["decision_evaluations", "observed_rows"]}))
