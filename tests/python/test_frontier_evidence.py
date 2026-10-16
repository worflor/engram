"""Independent binding of exact observer inputs, replay scope and law fixtures."""

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def exact_fit(bits, decay):
    mean = sum((F(bit) * decay ** (5 - i) for i, bit in enumerate(bits)), F(0)) / sum(
        (decay**i for i in range(6)), F(0)
    )
    a = [F(v) - mean for v in bits[1:-1]]
    b = [F(v) - mean for v in bits[:-2]]
    y = [F(v) - mean for v in bits[2:]]
    weights = [decay**i for i in range(3, -1, -1)]

    def dot(x, z):
        return sum((w * u * v for w, u, v in zip(weights, x, z)), F(0))

    aa, bb, ab, ay, by = dot(a, a), dot(b, b), dot(a, b), dot(a, y), dot(b, y)
    eps = F(1, 8388608)
    ridge = max(eps, (aa + bb) / 20000)
    det = (aa + ridge) * (bb + ridge) - ab * ab
    k, g = (
        (F(2), F(1))
        if not det or abs(det) < eps * (aa + bb) ** 2
        else ((ay * (bb + ridge) - ab * by) / det, (ab * ay - (aa + ridge) * by) / det)
    )
    residual = [t - k * u + g * v for t, u, v in zip(y, a, b)]
    return k, g, dot(residual, residual) / sum(weights)


def test_deployed_observer_replays_and_exact_label_binding():
    requests = json.loads(
        (ROOT / "research/exact/witness-observer-requests.json").read_text(
            encoding="utf-8"
        )
    )
    receipts = json.loads(
        (ROOT / "research/exact/witness-observer.json").read_text(encoding="utf-8")
    )
    for case in receipts["studies"]:
        expected = requests[case["case"]]
        assert case["request"] == expected["request"]
        query = json.loads(case["request"]["subject"].removeprefix("atlas "))["request"]
        findings = {f["asks"]: f["value"] for f in case["answer"]["study"]["findings"]}
        certified = findings["result"]
        assert (
            certified["source"] == query["source"]
            and certified["candidate"] == query["candidate"]
        )
        assert certified["contract"]["horizon"] == "all"
        assert case["verify"]["answer"]["kind"] == "valid"
        assert case["verify"]["answer"]["roots"] == findings["proof DAG"]["roots"]
        assert (
            case["invocation"]["version"]
            == case["verify"]["invocation"]["version"]
            == "1.0.28"
        )
        for bits, label in zip(query["source"]["states"], query["candidate"]):
            observation = (F(bits[-1]),) + tuple(
                v
                for decay in expected["decays"]
                for v in exact_fit(tuple(map(int, bits)), F(decay))
            )
            assert tuple(map(F, expected["observation_values"][label])) == observation
        result = certified["result"]
        assert len(set(result["partition"])) == 4
        assert result["sufficient"] == (case["case"] == "two_scales")
        if not result["sufficient"]:
            separator = result["separator"]
            left, right = separator["left"], separator["right"]
            assert query["candidate"][left] == query["candidate"][right]
            assert separator["actionWord"] == []
            assert query["source"]["goals"][left] != query["source"]["goals"][right]
        else:
            assert result["separator"] is None


def test_retained_composition_blending_and_exact_polynomial_fixture():
    receipt = json.loads(
        (ROOT / "research/history-frontiers.json").read_text(encoding="utf-8")
    )
    for path, expected in receipt["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert sum(receipt["exact_composition"].values()) == 8445
    assert receipt["float_composition"]["maximum_signature_error"] < 1e-6
    assert receipt["schur"]["accepted64"] == 10000
    assert receipt["schur"]["ordinary_unstable"] > 0
    for row in receipt["schur"]["radius_sweep"].values():
        assert row["cases"] == 64
        assert max(row["maximum_root_over_bound"].values()) < 1
    for record in receipt["temporal_policy"]:
        assert record["success_rate"] == (1.0 if len(record["decays"]) == 2 else 0.5)
    values = list(map(F, receipt["quintic"]["values"]))
    for i in range(6, len(values)):
        assert values[i] == sum(
            c * values[i - j - 1] for j, c in enumerate([6, -15, 20, -15, 6, -1])
        )
    assert (
        F(receipt["dwell_example"]["infinity_bound_at_43"])
        < 1
        < F(receipt["dwell_example"]["infinity_bound_at_42"])
    )
    exact = json.loads(
        (ROOT / "research/exact/history-composition-cases.json").read_text(
            encoding="utf-8"
        )
    )
    settled = [
        v for v in exact["checks"]["answer"]["verdicts"] if v["verdict"] == "CONFIRMED"
    ] + exact["route_pinned_checks"]["answer"]["verdicts"]
    assert len(settled) == 8 and all(v["verdict"] == "CONFIRMED" for v in settled)
