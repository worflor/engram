"""Origin sensitivity and future error: synthetic controls and optional UCI HAR.

The real-data follow-up reuses a previously inspected benchmark. Its new task
is prefix-only forecasting, not a fresh confirmatory classification experiment.
No hyperparameter is selected from future targets. --real never downloads
unless --download is also supplied.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engram_affine import AffineOscillator
from engram_codec import fit_all
from npc_real_signals import dataset, schema_from, SHA256

METHODS = ("homogeneous", "window_centered", "affine", "persistence")


def forecasts(context, steps):
    z = context[:, ::2] + 1j * context[:, 1::2]
    outputs = {
        "affine": AffineOscillator.fit(context).forecast(steps),
        "persistence": np.repeat(context[-1:], steps, axis=0),
    }
    for name, origin in [
        ("homogeneous", np.zeros(z.shape[1])),
        ("window_centered", z.mean(axis=0)),
    ]:
        centered = z - origin
        k, g, _, _ = fit_all(centered)
        older, previous = centered[-2:]
        prediction = np.empty((steps, context.shape[1]))
        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(steps):
                value = k * previous - g * older
                uncentered = value + origin
                prediction[t, ::2] = uncentered.real
                prediction[t, 1::2] = uncentered.imag
                older, previous = previous, value
        outputs[name] = prediction
    return outputs


def errors(actual, predicted):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.sqrt(np.mean((actual - predicted) ** 2))
    return float(value) if np.isfinite(value) else None


def summarize(values):
    finite = [v for v in values if v is not None]
    return {
        "cases": len(values),
        "nonfinite": len(values) - len(finite),
        "mean_rmse_of_finite": float(np.mean(finite)),
        "median_rmse_of_finite": float(np.median(finite)),
        "p90_rmse_of_finite": float(np.quantile(finite, 0.9)),
        "maximum_rmse_of_finite": float(np.max(finite)),
    }


def synthetic(seed, cases):
    rng = np.random.default_rng(seed)
    results = {}
    for family in (
        "oscillations",
        "damped",
        "ramp",
        "white_noise",
        "random_walk",
        "frequency_change",
    ):
        all_errors = {
            name: {"original": [], "translated": [], "translation_defect": []}
            for name in METHODS
        }
        for _ in range(cases):
            t = np.arange(128)
            x = np.empty((128, 4))
            for p in range(2):
                omega = rng.uniform(0.08, 0.35)
                phase = rng.uniform(-np.pi, np.pi)
                rho = 0.975 if family == "damped" else 1
                z = rho**t * (
                    rng.uniform(0.4, 1.2) * np.exp(1j * (omega * t + phase))
                    + rng.uniform(0.2, 0.6) * np.exp(-1j * (omega * t - phase / 2))
                )
                if family == "frequency_change":
                    later = np.arange(32)
                    z[96:] = np.exp(1j * (omega * 95 + phase + 0.6 * (later + 1)))
                x[:, 2 * p], x[:, 2 * p + 1] = z.real, z.imag
            if family == "ramp":
                x = t[:, None] * rng.normal(0, 0.03, 4) + rng.normal(0, 1, 4)
            elif family == "white_noise":
                x = rng.normal(size=(128, 4))
            elif family == "random_walk":
                x = np.cumsum(rng.normal(0, 0.1, (128, 4)), axis=0)
            elif family != "frequency_change":
                x += rng.normal(0, 0.005, x.shape)
            offset = rng.uniform(-100, 100, 4)
            original = forecasts(x[:96], 32)
            translated = forecasts(x[:96] + offset, 32)
            for name in METHODS:
                row = all_errors[name]
                row["original"].append(errors(x[96:], original[name]))
                row["translated"].append(errors(x[96:] + offset, translated[name]))
                row["translation_defect"].append(
                    errors(original[name], translated[name] - offset)
                )
        results[family] = {
            name: {condition: summarize(values) for condition, values in row.items()}
            for name, row in all_errors.items()
        }
    return results


def real_data(args):
    (training, _, _), (testing, _, subjects) = dataset(args.cache, args.download)
    schema = schema_from(training)
    normalized = (testing - schema.offsets) / schema.scales
    # Pairing's zero companion is not a measured sensor or an evaluation target.
    padded = np.pad(normalized, ((0, 0), (0, 0), (0, 1)))
    measured = {name: [] for name in METHODS}
    for i, window in enumerate(padded):
        for name, prediction in forecasts(window[:96], 32).items():
            measured[name].append(errors(window[96:, :9], prediction[:, :9]))
        if (i + 1) % 500 == 0:
            print("Real windows", i + 1, flush=True)
    return {
        "dataset_sha256": SHA256,
        "scope": "Exploratory prefix forecasting on the previously inspected UCI HAR benchmark; no test-based parameter selection",
        "observed_frames": 96,
        "future_frames": 32,
        "measured_channels": 9,
        "normalization": "Training people only; frozen sensor means/deviations",
        "schema_identity": schema.identity,
        "methods": {
            name: {
                "all_windows": summarize(values),
                "subjects": {
                    str(s): summarize(
                        [v for v, subject in zip(values, subjects) if subject == s]
                    )
                    for s in sorted(set(subjects))
                },
                "rmse_sha256": hashlib.sha256(
                    json.dumps(values, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for name, values in measured.items()
        },
    }


def run(args):
    report = {
        "schema": "engram.affine-forecast/v1",
        "method": {
            "seed": args.seed,
            "cases_per_family": args.cases,
            "observed_frames": 96,
            "future_frames": 32,
            "relative_ridge": 1e-4,
            "translation": "Independent fixed per-channel offsets sampled uniformly from [-100,100]",
        },
        "synthetic": synthetic(args.seed, args.cases),
    }
    if args.real:
        report["real"] = real_data(args)
    report["runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "system": platform.system(),
    }
    report["sources"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in [
            "engram_affine.py",
            "engram_codec.py",
            "engram_npc.py",
            "engram_history.py",
            "research/experiments/affine_forecast.py",
            "research/experiments/npc_real_signals.py",
        ]
    }
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Saved", str(args.output), flush=True)
    for family, rows in report["synthetic"].items():
        print(
            family,
            {
                name: round(row["translated"]["median_rmse_of_finite"], 5)
                for name, row in rows.items()
            },
            flush=True,
        )
    if "real" in report:
        print(
            "Real median RMSE",
            {
                name: row["all_windows"]["median_rmse_of_finite"]
                for name, row in report["real"]["methods"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/uci-har")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/affine-forecast.json"
    )
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("positive synthetic case count required")
    run(args)
