"""Small offline study: model closure, coordinate changes and generated motion.

This is an investigation, not a new production projection or a benchmark claim.
Run from an installed checkout: python research/experiments/projection_audit.py
"""

from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import numpy as np
from engram_codec import TanhProjection, _to_complex, _from_complex, predict_all
from engram_analysis import Signature
from engram_metrics import forecast_metrics
from engram_network import WellNetwork

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from well_network import train, signal


def least_squares_future(context, steps, *, order=2, intercept=False):
    """Unregularized per-pair baseline to distinguish model mismatch from ridge bias."""
    z = _to_complex(np.asarray(context, np.float64))
    past = [row.copy() for row in z]
    coefficients = []
    for pair in range(z.shape[1]):
        design = np.column_stack(
            [z[order - lag : -lag, pair] for lag in range(1, order + 1)]
        )
        if intercept:
            design = np.column_stack([design, np.ones(len(design))])
        coefficients.append(np.linalg.lstsq(design, z[order:, pair], rcond=None)[0])
    coefficients = np.array(coefficients)
    for _ in range(steps):
        value = sum(
            coefficients[:, lag - 1] * past[-lag] for lag in range(1, order + 1)
        )
        if intercept:
            value += coefficients[:, -1]
        past.append(value)
    return _from_complex(np.array(past[-steps:]))


def inspect(x, *, context=96, horizon=64):
    source = x[:context].astype(np.float32)
    target = x[context : context + horizon].astype(np.float32)
    signature = Signature.from_trajectory(source, pairing=np.arange(x.shape[1]))
    z = _to_complex(source)
    forecast = _from_complex(
        predict_all(z[-1], z[-2], signature.k, signature.g, horizon)
    )
    full = _to_complex(x.astype(np.float32))
    teacher = _from_complex(
        signature.k * full[context - 1 : context + horizon - 1]
        - signature.g * full[context - 2 : context + horizon - 2]
    )
    metrics = forecast_metrics(target, forecast, last_observation=source[-1])
    target_std = float(np.sqrt(np.mean((target - target.mean(axis=0)) ** 2)))
    metrics["normalized_rollout_rmse"] = metrics["rmse"] / target_std
    metrics["one_step_rmse"] = float(np.sqrt(np.mean((target - teacher) ** 2)))
    metrics["least_squares_ar2_rmse"] = float(
        np.sqrt(np.mean((target - least_squares_future(source, horizon)) ** 2))
    )
    metrics["least_squares_ar4_rmse"] = float(
        np.sqrt(np.mean((target - least_squares_future(source, horizon, order=4)) ** 2))
    )
    metrics["affine_ar2_rmse"] = float(
        np.sqrt(
            np.mean(
                (target - least_squares_future(source, horizon, intercept=True)) ** 2
            )
        )
    )
    return {
        "metrics": metrics,
        "source": source.tolist(),
        "actual": target.tolist(),
        "forecast": forecast.tolist(),
    }


def transformations():
    t = np.arange(256)
    z = np.stack(
        [
            np.exp(0.17j * t) + 0.45 * np.exp(-0.31j * t),
            0.8 * np.exp(0.57j * t) + 0.6 * np.exp(-0.83j * t),
        ],
        axis=1,
    )
    x = _from_complex(z).astype(np.float32)
    rotated = _from_complex(z * (2 * np.exp(0.4j))).astype(np.float32)
    mixed = np.column_stack(
        [x[:, 0] + x[:, 2], x[:, 1] + x[:, 3], x[:, 0] - x[:, 2], x[:, 1] - x[:, 3]]
    ) / np.sqrt(2)
    shifted = x + np.array([5, -3, 4, -6], np.float32)
    cases = {
        "original": x,
        "within_pair_rotation_and_scale": rotated,
        "lossless_cross_pair_mixing": mixed,
        "translated": shifted,
        "translated_known_center_removed": shifted - np.array([5, -3, 4, -6]),
    }
    for name, alpha in [("tanh_weak", 0.1), ("tanh_auto", None), ("tanh_strong", 4.0)]:
        projector = TanhProjection(np.eye(4, dtype=np.float32), alpha=alpha)
        if alpha is None:
            projector.calibrate(x[:96])  # no calibration on the held-out future
        cases[name] = projector.project(x)
    return {name: inspect(value) for name, value in cases.items()}


def network_horizons():
    network, names = train()
    results = []
    for horizon in [24, 64, 128]:
        for offset in [0.0, 2.0]:
            measurements = []
            correct = 0
            for index in range(30):
                family = index % 3
                x = signal(
                    family, np.random.default_rng(1000 + index), length=72 + horizon
                )
                x += offset
                context, future = x[:72], x[72:]
                nearest = network.activate(context)[0].well
                correct += names[nearest] == family
                generated = network.dream(context, steps=horizon)
                metrics = forecast_metrics(
                    future, generated.trajectory, last_observation=context[-1]
                )
                metrics["affine_ar2_rmse"] = float(
                    np.sqrt(
                        np.mean(
                            (
                                future
                                - least_squares_future(context, horizon, intercept=True)
                            )
                            ** 2
                        )
                    )
                )
                measurements.append(metrics)
            results.append(
                {
                    "horizon": horizon,
                    "translation": offset,
                    "recognition": correct / 30,
                    "well_rmse": float(np.mean([m["rmse"] for m in measurements])),
                    "persistence_rmse": float(
                        np.mean([m["persistence_rmse"] for m in measurements])
                    ),
                    "affine_ar2_rmse": float(
                        np.mean([m["affine_ar2_rmse"] for m in measurements])
                    ),
                }
            )
    x = signal(2, np.random.default_rng(456))
    graph = []
    for walk in [None, 4, 8, 24]:
        generated = network.dream(
            x[:72], steps=24, walk_every=walk, rng=np.random.default_rng(42)
        )
        graph.append(
            {
                "walk_every": walk,
                "itinerary": list(generated.itinerary),
                "metrics": forecast_metrics(
                    x[72:], generated.trajectory, last_observation=x[71]
                ),
            }
        )
    return results, graph


def stable_operations():
    a, b = 199 / 200, 1 / 100
    root = a + 1j * b
    ka, ga = 2 * root, root * root
    kb, gb = ka.conjugate(), ga.conjugate()
    mean_k, mean_g = (ka + kb) / 2, (ga + gb) / 2

    # Exact-feature input is a supported boundary. Keep it separate from the
    # ridge-biased trajectory-fitting experiments above.
    def sig(k, g):
        return Signature(
            np.array([k]), np.array([g]), np.zeros(1), np.arange(2), 100, 2
        )

    network = WellNetwork(2)
    wa = network.learn_signature(sig(ka, ga))
    wb = network.learn_signature(sig(kb, gb))
    mean_roots = np.roots([1, -mean_k, mean_g])
    # Arbitrary switching is a separate issue from coefficient averaging.
    r = 0.9
    left = np.array([[2 * r, -r * r], [1, 0]], float)
    right = np.array([[-2 * r, -r * r], [1, 0]], float)
    state = np.array([1.0, 0.0])
    switched = []
    for i in range(24):
        state = (left if i % 2 == 0 else right) @ state
        switched.append(float(state[0]))
    transient = predict_all(
        np.array([1.0]), np.array([0.0]), np.array([1.98]), np.array([0.9801]), 200
    )
    return {
        "coefficient_average": {
            "individual_root_modulus": float(abs(root)),
            "feature_distance": float(np.sqrt(abs(ka - kb) ** 2 + abs(ga - gb) ** 2)),
            "automatic_same_well": wa == wb,
            "averaged_radius": float(max(abs(mean_roots))),
            "learned_well_radius": network.describe()[0]["radius"],
        },
        "switching": {
            "individual_radius": r,
            "two_step_radius": float(max(abs(np.linalg.eigvals(right @ left)))),
            "samples": switched,
            "peak_abs": max(abs(x) for x in switched),
        },
        "repeated_root_transient": {
            "radius": 0.99,
            "peak_abs": float(np.max(abs(transient))),
            "future_index_of_peak": int(np.argmax(abs(transient))),
        },
    }


def fitted_average_search():
    tested, found = 0, []
    for length in [16, 32, 64, 128, 512]:
        for real in [0.96, 0.98, 0.995, 0.999]:
            for imag in [0.005, 0.01, 0.03, 0.08, 0.15]:
                if abs(real + 1j * imag) >= 1:
                    continue
                for split in [0, 0.005, 0.02, 0.1]:
                    roots = np.array(
                        [real + 1j * imag, real - split + 1j * (imag + 0.03)]
                    )
                    if max(abs(roots)) >= 1:
                        continue
                    t = np.arange(length)
                    z = (
                        (roots[0] ** t + 0.5j * roots[1] ** t)
                        if split
                        else (1 + 0.05 * t) * (real + 1j * imag) ** t
                    )
                    x = np.column_stack([z.real, z.imag]).astype(np.float32)
                    s = Signature.from_trajectory(x, pairing=np.arange(2))
                    radius = max(abs(np.roots([1, -s.k[0], s.g[0]])))
                    meanradius = max(abs(np.roots([1, -s.k[0].real, s.g[0].real])))
                    distance = np.sqrt(
                        abs(s.k[0] - s.k[0].conjugate()) ** 2
                        + abs(s.g[0] - s.g[0].conjugate()) ** 2
                    )
                    tested += 1
                    if radius < 1 and meanradius > 1 and distance < 0.15:
                        found.append(
                            {
                                "length": length,
                                "real": real,
                                "imag": imag,
                                "split": split,
                                "fit_radius": float(radius),
                                "mean_radius": float(meanradius),
                                "distance": float(distance),
                            }
                        )
    return {
        "tested": tested,
        "counterexamples": found,
        "scope": "Bounded grid of ridge-fitted two-mode/generalized-mode signals and conjugates; absence is not a universal safety claim.",
    }


def plot(report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    choices = [
        ("original", "Original two-mode signal", None),
        ("lossless_cross_pair_mixing", "Lossless mixing across pairs", "ar4"),
        ("translated", "Same motion, translated", "affine"),
        ("tanh_auto", "Automatic tanh projection", None),
    ]
    for ax, (name, title, alternative) in zip(axes.flat, choices):
        case = report["transformations"][name]
        source, actual, forecast = [
            np.array(case[k]) for k in ["source", "actual", "forecast"]
        ]
        n = len(source)
        ax.plot(
            np.arange(n - 24, n), source[-24:, 0], color="#39809c", label="Observed"
        )
        ax.plot(
            np.arange(n, n + len(actual)),
            actual[:, 0],
            color="#348253",
            label="Actual future",
        )
        ax.plot(
            np.arange(n, n + len(forecast)),
            forecast[:, 0],
            color="#ce7d31",
            label="Engram AR(2)",
        )
        if alternative:
            baseline = least_squares_future(
                source,
                len(actual),
                order=4 if alternative == "ar4" else 2,
                intercept=alternative == "affine",
            )
            ax.plot(
                np.arange(n, n + len(actual)),
                baseline[:, 0],
                color="#8966b0",
                linestyle="--",
                label="AR(4) control"
                if alternative == "ar4"
                else "Offset-aware control",
            )
        ax.axvline(n - 0.5, color="#888888", linestyle=":", linewidth=1)
        ax.set_title(
            f"{title}\nNormalized rollout RMS {case['metrics']['normalized_rollout_rmse']:.3f}",
            loc="left",
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel("Coordinate 0 / transformed units")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "Engram: preserving information does not preserve a two-mode model\n96 observed rows · 64 held-out rows · same synthetic source",
        fontsize=14,
    )
    fig.savefig(ROOT / "docs/media/projection-audit.png", dpi=150)
    plt.close(fig)


def main(make_plot=False):
    transformed = transformations()
    horizons, graph = network_horizons()
    projector = TanhProjection(np.eye(4, dtype=np.float32))
    projector.project(np.zeros((4, 4), np.float32))
    after_zero = projector.project(np.full((4, 4), 0.001, np.float32))
    heavy_tail = np.repeat(
        np.array([0.0] * 490 + [1.0] * 20 + [1e6] * 490, dtype=np.float32)[:, None],
        4,
        axis=1,
    )
    heavy_projection = TanhProjection(np.eye(4, dtype=np.float32)).project(heavy_tail)
    report = {
        "schema": "engram-projection-audit-1",
        "scope": "Small synthetic counterexamples and baselines, not a corpus or semantic evaluation.",
        "transformations": transformed,
        "network_horizons": horizons,
        "graph_walk": graph,
        "stability_operations": stable_operations(),
        "fitted_average_search": fitted_average_search(),
        "zero_sample_calibration": {
            "alpha": projector.alpha,
            "subsequent_input": 0.001,
            "subsequent_output": float(after_zero[0, 0]),
        },
        "calibration_gradient_zone_counterexample": {
            "distribution": "49% zero, 2% one, 49% one million; all coordinates",
            "fraction_between_point_one_and_point_nine": float(
                np.mean((abs(heavy_projection) > 0.1) & (abs(heavy_projection) < 0.9))
            ),
            "scope": "Counterexample to the docstring's unqualified 85% gradient-zone assertion, not an error in tanh evaluation.",
        },
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "engram_codec.py",
                "engram_analysis.py",
                "engram_network.py",
                "examples/well_network.py",
                "research/experiments/projection_audit.py",
            ]
        },
    }
    assert transformed["original"]["metrics"]["least_squares_ar2_rmse"] < 1e-4
    assert (
        transformed["lossless_cross_pair_mixing"]["metrics"]["least_squares_ar2_rmse"]
        > 0.1
    )
    assert (
        transformed["lossless_cross_pair_mixing"]["metrics"]["least_squares_ar4_rmse"]
        < 1e-4
    )
    assert transformed["translated"]["metrics"]["affine_ar2_rmse"] < 1e-4
    assert report["stability_operations"]["coefficient_average"]["automatic_same_well"]
    assert report["stability_operations"]["coefficient_average"]["averaged_radius"] > 1
    path = ROOT / "research/projection-audit.json"
    serialized = re.sub(
        r"\[\s*([-\d][-\d.eE+,\s]*)\]",
        lambda m: "[" + " ".join(m.group(1).split()) + "]",
        json.dumps(report, indent=2),
    )
    path.write_text(serialized + "\n", encoding="utf-8", newline="\n")
    for name, case in transformed.items():
        m = case["metrics"]
        print(
            f"{name:36s} one-step={m['one_step_rmse']:.6f} rollout={m['rmse']:.6f} normalized={m['normalized_rollout_rmse']:.4f} "
            f"AR2-LS={m['least_squares_ar2_rmse']:.6f} AR4-LS={m['least_squares_ar4_rmse']:.6f} affine={m['affine_ar2_rmse']:.6f}"
        )
    print("horizons", json.dumps(horizons))
    print(
        "graph",
        json.dumps(
            [{**row, "metrics": {"rmse": row["metrics"]["rmse"]}} for row in graph]
        ),
    )
    print("operations", json.dumps(report["stability_operations"]))
    print("fitted_average_search", json.dumps(report["fitted_average_search"]))
    if make_plot:
        plot(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", action="store_true", help="Optional matplotlib figure"
    )
    main(parser.parse_args().plot)
