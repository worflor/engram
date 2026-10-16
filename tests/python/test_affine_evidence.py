"""Check retained exact component findings and measured campaign identities."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_exact_translation_identity_and_design_ranks_are_retained():
    receipt = json.loads(
        (ROOT / "research/exact/affine-cases.json").read_text(encoding="utf-8")
    )
    studies = receipt["studies"]
    ranks = {}
    for row in studies:
        assert row["invocation"]["version"] == "1.0.28"
        for finding in row["answer"]["study"]["findings"]:
            if finding["asks"] == "rank over \u211a":
                ranks[row["case"]] = finding["value"]
    assert ranks == {
        "affine_design": 3,
        "homogeneous_design": 2,
        "homogeneous_augmented": 3,
    }
    verdicts = (
        receipt["checks"]["answer"]["verdicts"]
        + receipt["identity_check"]["answer"]["verdicts"]
    )
    assert len(verdicts) == 6 and all(v["verdict"] == "CONFIRMED" for v in verdicts)


def test_affine_campaign_sources_and_origin_equivariance():
    receipt = json.loads(
        (ROOT / "research/affine-forecast.json").read_text(encoding="utf-8")
    )
    for path, expected in receipt["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert (
        len(receipt["synthetic"]) == 6 and receipt["method"]["cases_per_family"] == 40
    )
    for methods in receipt["synthetic"].values():
        assert methods["affine"]["translation_defect"]["maximum_rmse_of_finite"] < 1e-8
        for row in methods.values():
            for condition in row.values():
                assert condition["cases"] == 40 and condition["nonfinite"] == 0
    for method in receipt["real"]["methods"].values():
        assert method["all_windows"]["cases"] == 2947
        assert method["all_windows"]["nonfinite"] == 0
        assert sum(s["cases"] for s in method["subjects"].values()) == 2947
