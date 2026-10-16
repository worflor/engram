"""Subject-held-out UCI HAR oscillator/readout experiment; explicit download only.

NumPy only. This measures a supervised geometric readout of Engram features,
not reward learning, forecasting, or the NPC's navigation policy. No corpus or
trained state is committed. Run from the repository root; see the research brief.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import urllib.request
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engram_codec import fit_all
from engram_npc import WitnessSchema, WitnessHistory

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
CHANNELS = tuple(
    f"{sensor}_{axis}"
    for sensor in ("body_acc", "body_gyro", "total_acc")
    for axis in "xyz"
)
RADII = (0.35, 0.6, 0.9, 1.2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(cache, download=False):
    archive = cache / "uci-har.zip"
    if not archive.exists():
        if not download:
            raise ValueError(
                "dataset absent; use --download to fetch the official archive"
            )
        cache.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(URL, temporary)
        if digest(temporary) != SHA256:
            raise ValueError("download differs from the pinned dataset")
        temporary.replace(archive)
    if digest(archive) != SHA256:
        raise ValueError("dataset SHA-256 differs")
    with zipfile.ZipFile(archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip"))) as z:
            splits = {}
            for split in ("train", "test"):
                prefix = f"UCI HAR Dataset/{split}/"
                signals = [
                    np.loadtxt(z.open(prefix + f"Inertial Signals/{name}_{split}.txt"))
                    for name in CHANNELS
                ]
                splits[split] = (
                    np.stack(signals, axis=-1),
                    np.loadtxt(z.open(prefix + f"y_{split}.txt"), dtype=int) - 1,
                    np.loadtxt(z.open(prefix + f"subject_{split}.txt"), dtype=int),
                )
    train, test = splits["train"], splits["test"]
    if len(train[0]) != 7352 or len(test[0]) != 2947 or set(train[2]) & set(test[2]):
        raise ValueError("official subject-separated split differs")
    for x, y, subjects in splits.values():
        if (
            x.shape != (len(y), 128, 9)
            or not np.isfinite(x).all()
            or set(y) != set(range(6))
        ):
            raise ValueError("unexpected finite signal/label layout")
    return train, test


def features(x, schema, *, shuffled=False):
    """Batch the canonical fitter over independent windows; never join windows."""
    raw = (x - schema.offsets) / schema.scales
    rng = np.random.default_rng(813)
    if shuffled:
        raw = np.stack([window[rng.permutation(len(window))] for window in raw])
    padded = np.pad(raw, ((0, 0), (0, 0), (0, 1)))
    dynamics = []
    for start in range(0, len(raw), 256):
        batch = padded[start : start + 256]
        centered = batch - batch.mean(axis=1, keepdims=True)
        z = centered[..., ::2] + 1j * centered[..., 1::2]
        k, g, _, rms = fit_all(z.transpose(1, 0, 2).reshape(128, -1))
        n = len(batch)
        k, g, rms = k.reshape(n, 5), g.reshape(n, 5), rms.reshape(n, 5)
        dynamics.append(
            np.concatenate([k.real, k.imag, g.real, g.imag, np.log1p(rms)], axis=1)
        )
    dynamic = np.concatenate(dynamics)
    moment = np.concatenate([raw.mean(axis=1), np.log1p(raw.std(axis=1))], axis=1)
    current = padded[:, -1]
    return {
        "current": current,
        "moments": moment,
        "oscillators": dynamic,
        "npc_context": np.concatenate([current, dynamic], axis=1),
        "moments_and_oscillators": np.concatenate([moment, dynamic], axis=1),
    }


class Readout:
    """Online radius wells and supervised class counts; no preset node budget.

    Geometry grows without labels. Labels annotate the selected well after
    routing. This is a deliberately simple research readout, not EngramNPC.
    """

    def __init__(self, radius):
        self.radius = radius

    def fit(self, x, y):
        self.offset = x.mean(axis=0)
        deviations = x.std(axis=0)
        self.scale = np.where(deviations > 1e-8, deviations, 1)
        normalized = (x - self.offset) / self.scale
        # Dataset-sized workspace is an allocation bound, not a clustering cap:
        # at most one well can be formed by each supplied observation.
        self.centers = np.empty_like(normalized)
        self.counts = np.zeros(len(x), int)
        self.labels = np.zeros((len(x), 6), int)
        self.size = 0
        for i in np.random.default_rng(7).permutation(len(x)):
            row = normalized[i]
            distances = np.mean((self.centers[: self.size] - row) ** 2, axis=1)
            nearest = int(np.argmin(distances)) if self.size else 0
            if not self.size or distances[nearest] > self.radius**2:
                nearest = self.size
                self.centers[nearest] = row
                self.size += 1
            self.counts[nearest] += 1
            self.centers[nearest] += (row - self.centers[nearest]) / self.counts[
                nearest
            ]
            self.labels[nearest, y[i]] += 1
        self.centers = self.centers[: self.size].copy()
        self.counts = self.counts[: self.size].copy()
        self.labels = self.labels[: self.size].copy()
        return self

    def predict(self, x):
        result = []
        for row in (x - self.offset) / self.scale:
            distances = np.mean((self.centers - row) ** 2, axis=1)
            result.append(int(self.labels[int(np.argmin(distances))].argmax()))
        return np.array(result)


def metrics(expected, predicted, subjects):
    confusion = np.zeros((6, 6), int)
    np.add.at(confusion, (expected, predicted), 1)
    recall = confusion.diagonal() / confusion.sum(axis=1)
    f1 = 2 * confusion.diagonal() / (confusion.sum(axis=0) + confusion.sum(axis=1))
    return {
        "accuracy": float(np.mean(expected == predicted)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion": confusion.tolist(),
        "subjects": {
            str(s): {
                "windows": int(np.sum(subjects == s)),
                "accuracy": float(
                    np.mean(expected[subjects == s] == predicted[subjects == s])
                ),
            }
            for s in sorted(set(subjects))
        },
    }


def schema_from(x):
    return WitnessSchema.calibrate(
        CHANNELS, x.reshape(-1, 9), representation="uci-har/50hz/inertial9-v1"
    )


def run(args):
    (train, y, subjects), (test, yt, st) = dataset(args.cache, args.download)
    # Decide the validation subjects before fitting or looking at test outcomes.
    validation_subjects = sorted(set(subjects))[::5]
    validation = np.isin(subjects, validation_subjects)
    fit = ~validation
    print("Training-only calibration and independent 128-frame fits", flush=True)
    inner_schema = schema_from(train[fit])
    final_schema = schema_from(train)
    inner_features = features(train, inner_schema)
    train_features = features(train, final_schema)
    test_features = features(test, final_schema)
    # Ensure the batch route is the same centered recurrence as the NPC stream.
    h = WitnessHistory(10)
    for row in train[0]:
        signature = h.push(final_schema.vector(row))
    np.testing.assert_allclose(
        train_features["oscillators"][0, :5], signature.k.real, atol=1e-8
    )
    np.testing.assert_allclose(
        train_features["oscillators"][0, 10:15], signature.g.real, atol=1e-8
    )
    # Time scrambling destroys order while preserving each window's moments.
    for collection, x, schema in [
        (inner_features, train, inner_schema),
        (train_features, train, final_schema),
        (test_features, test, final_schema),
    ]:
        shuffled = features(x, schema, shuffled=True)
        np.testing.assert_allclose(
            shuffled["moments"], collection["moments"], atol=1e-12
        )
        collection["shuffled_oscillators"] = shuffled["oscillators"]
    report = {
        "schema": "engram.real-signal-study/v1",
        "scope": "Supervised geometric readout, not NPC reward learning or general intelligence",
        "dataset": {
            "url": URL,
            "sha256": SHA256,
            "doi": "10.24432/C54S4K",
            "license": "CC BY 4.0",
            "train_windows": len(train),
            "test_windows": len(test),
            "frames_per_window": 128,
            "sampling_hz": 50,
            "train_subjects": sorted(map(int, set(subjects))),
            "test_subjects": sorted(map(int, set(st))),
            "validation_subjects": list(map(int, validation_subjects)),
            "labels": [
                "walking",
                "upstairs",
                "downstairs",
                "sitting",
                "standing",
                "laying",
            ],
        },
        "method": {
            "radius_candidates": RADII,
            "selection": "validation balanced accuracy; smaller radius breaks ties",
            "training_order_seed": 7,
            "shuffle_seed": 813,
            "coordinates": "training-only means and population deviations; fixed identity pairing and zero pad",
            "feature_metric": "training-only feature deviations, frozen before evaluation",
            "schema_identity": final_schema.identity,
            "test_use": "evaluation only; no radius/feature selection on test",
        },
        "results": {},
    }
    for name in train_features:
        candidates = []
        for radius in RADII:
            model = Readout(radius).fit(inner_features[name][fit], y[fit])
            score = metrics(
                y[validation],
                model.predict(inner_features[name][validation]),
                subjects[validation],
            )
            candidates.append(
                {
                    "radius": radius,
                    "balanced_accuracy": score["balanced_accuracy"],
                    "wells": model.size,
                }
            )
        chosen = max(
            candidates, key=lambda row: (row["balanced_accuracy"], -row["radius"])
        )["radius"]
        model = Readout(chosen).fit(train_features[name], y)
        predicted = model.predict(test_features[name])
        result = {
            "features": train_features[name].shape[1],
            "radius": chosen,
            "wells": model.size,
            "validation": candidates,
            "test": metrics(yt, predicted, st),
            "prediction_sha256": hashlib.sha256(
                predicted.astype("<i4").tobytes()
            ).hexdigest(),
        }
        report["results"][name] = result
        print(name, result["wells"], result["test"]["accuracy"], flush=True)
    report["runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "system": platform.system(),
    }
    report["sources"] = {
        p: digest(ROOT / p)
        for p in [
            "engram_npc.py",
            "engram_history.py",
            "engram_codec.py",
            "research/experiments/npc_real_signals.py",
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Saved " + str(args.output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/uci-har")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/npc/real-signals.json"
    )
    run(parser.parse_args())
