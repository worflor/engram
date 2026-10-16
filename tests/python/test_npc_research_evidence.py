"""Internal consistency of retained data/stress evidence; no corpus download."""

import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def receipt(name):
    report = json.loads((ROOT / "research/npc" / name).read_text(encoding="utf-8"))
    for path, expected in report["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    return report


def test_retained_subject_split_and_confusion_metrics():
    report = receipt("real-signals.json")
    data = report["dataset"]
    assert len(data["train_subjects"]) == 21 and len(data["test_subjects"]) == 9
    assert not set(data["train_subjects"]) & set(data["test_subjects"])
    assert set(data["validation_subjects"]) <= set(data["train_subjects"])
    assert data["train_windows"] == 7352 and data["test_windows"] == 2947
    for result in report["results"].values():
        test = result["test"]
        matrix = np.array(test["confusion"])
        assert matrix.shape == (6, 6) and (matrix >= 0).all()
        assert matrix.sum() == data["test_windows"]
        np.testing.assert_allclose(test["accuracy"], matrix.trace() / matrix.sum())
        np.testing.assert_allclose(
            test["balanced_accuracy"], (matrix.diagonal() / matrix.sum(axis=1)).mean()
        )
        np.testing.assert_allclose(
            test["macro_f1"],
            (2 * matrix.diagonal() / (matrix.sum(axis=0) + matrix.sum(axis=1))).mean(),
        )
        assert sum(row["windows"] for row in test["subjects"].values()) == matrix.sum()
        selection = max(
            result["validation"],
            key=lambda row: (row["balanced_accuracy"], -row["radius"]),
        )
        assert selection["radius"] == result["radius"]


def test_retained_deterministic_and_numerical_controls():
    report = receipt("stress.json")
    a, b = report["deterministic_replay"]
    assert a["python_hash_seed"] != b["python_hash_seed"]
    assert a["state_sha256"] == b["state_sha256"]
    assert a["decisions_sha256"] == b["decisions_sha256"]
    assert a["steps"] == a["real_updates"] == 1000
    for row in report["streaming_history"]:
        assert row["samples"] == 100000
        assert row["maximum_reference_error"] < 1e-6
        assert row["previous_rows"] == 2 and row["resident_array_bytes"] == 1088
    planning = report["observed_model_planning"]
    assert abs(planning["learned_value"] - planning["analytical_value"]) < 1e-10
    assert report["growth"]["stored_wells"] > 2048
    assert report["growth"]["available_actions"] == 4096
