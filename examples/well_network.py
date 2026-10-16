"""Train an unlabeled oscillator-well network; evaluate untouched episodes.

python examples/well_network.py [--output report.json]
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from engram_network import WellNetwork
from engram_training import Trainer, TrainingItem
from engram_metrics import forecast_metrics

FAMILIES = ("slow orbit", "cross current", "rapid orbit")


def signal(family, rng, *, length=96, noise=0.003):
    """Two complex modes per pair, random phases/amplitudes and small frequency jitter."""
    omega = [0.16, 0.55, 1.12][family] + rng.normal(0, 0.001)
    t = np.arange(length)
    pairs = []
    for offset in [0.0, 0.08]:
        a = 0.997 * np.exp(1j * (omega + offset))
        b = 0.975 * np.exp(-1j * (omega + offset) * 0.8)
        ca, cb = rng.uniform(0.6, 1.4, 2) * np.exp(1j * rng.uniform(-np.pi, np.pi, 2))
        pairs.append(ca * a**t + cb * b**t)
    z = np.stack(pairs, axis=1)
    x = np.empty((length, 4), np.float32)
    x[:, ::2], x[:, 1::2] = z.real, z.imag
    return x + rng.normal(0, noise, x.shape).astype(np.float32)


def train(seed=7, episodes=20, noise=0.003, novelty_radius=0.15):
    rng = np.random.default_rng(seed)
    network = WellNetwork(4, novelty_radius=novelty_radius, temperature=0.1)
    assignments = defaultdict(Counter)
    training = []
    labels = {}
    for episode in range(episodes):
        for step, family in enumerate([0, 1, 2, 0, 1, 2]):
            key = f"episode-{episode}/step-{step}"
            trajectory = signal(family, rng, noise=noise)[:72]
            training.append(TrainingItem(key, trajectory, sequence_start=step == 0))
            labels[key] = family
    # Labels are withheld from learning. They are used only to name/evaluate the
    # unsupervised clusters after training has finished.
    for event in Trainer(network, source_id="two-mode-signals-v1").fit(training):
        assignments[event.well][labels[event.key]] += 1
    names = {well: counts.most_common(1)[0][0] for well, counts in assignments.items()}
    return network, names


def probe(
    network,
    names,
    *,
    seed=1007,
    family=0,
    noise=0.003,
    horizon=24,
    dream_noise=0.0,
    max_radius=None,
):
    rng = np.random.default_rng(seed)
    trajectory = signal(family, rng, length=72 + horizon, noise=noise)
    context, future = trajectory[:72], trajectory[72:]
    activations = network.activate(context)
    dreamed = network.dream(
        context,
        steps=horizon,
        noise=dream_noise,
        rng=np.random.default_rng(seed + 1),
        max_radius=max_radius,
    )
    return {
        "family": FAMILIES[family],
        "predicted_family": FAMILIES[names[activations[0].well]],
        "well": activations[0].well,
        "novel": activations[0].distance > network.novelty_radius,
        "activations": [asdict(a) for a in activations],
        "metrics": forecast_metrics(
            future, dreamed.trajectory, last_observation=context[-1]
        ),
        "original_radius": dreamed.original_radius,
        "stabilized": dreamed.stabilized,
        "context": context.tolist(),
        "actual": future.tolist(),
        "dream": dreamed.trajectory.tolist(),
    }


def experiment(*, seed=7, episodes=20, noise=0.003, novelty_radius=0.15):
    network, names = train(seed, episodes, noise, novelty_radius)
    correct, shuffled_correct, measurements, local_errors = 0, 0, [], []
    confusion = np.zeros((3, 3), int)
    for index in range(60):
        family = index % 3
        result = probe(
            network, names, seed=seed + 1000 + index, family=family, noise=noise
        )
        predicted = FAMILIES.index(result["predicted_family"])
        confusion[family, predicted] += 1
        correct += family == predicted
        measurements.append(result["metrics"])
        context = np.asarray(result["context"], np.float32)
        local = WellNetwork(4)
        local.learn(context)
        local_prediction = local.dream(context, steps=len(result["actual"]))
        local_errors.append(
            forecast_metrics(
                result["actual"],
                local_prediction.trajectory,
                last_observation=context[-1],
            )["rmse"]
        )
        shuffled = context[
            np.random.default_rng(seed + 5000 + index).permutation(len(context))
        ]
        shuffled_correct += names[network.activate(shuffled)[0].well] == family
    report = {
        "seed": seed,
        "episodes": episodes,
        "training_observations": episodes * 6,
        "held_out_observations": 60,
        "accuracy": correct / 60,
        "majority_baseline_accuracy": 1 / 3,
        "confusion": confusion.tolist(),
        "time_shuffled_accuracy": shuffled_correct / 60,
        "mean_forecast_rmse": float(np.mean([m["rmse"] for m in measurements])),
        "mean_local_fit_forecast_rmse": float(np.mean(local_errors)),
        "mean_persistence_rmse": float(
            np.mean([m["persistence_rmse"] for m in measurements])
        ),
        "wells": network.describe(),
        "family_names": {well: FAMILIES[label] for well, label in names.items()},
        "probe": probe(network, names, seed=seed + 3000, family=1, noise=noise),
    }
    return network, names, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, _, report = experiment()
    summary = {
        k: v for k, v in report.items() if k not in ["probe", "wells", "family_names"]
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
