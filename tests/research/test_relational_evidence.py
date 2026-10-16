"""Bind retained research claims to their finite sources and native verdicts."""

import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from research.experiments.relational_minecraft import OutcomeTable

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name,field",
    [
        ("relational-memory.json", "sources"),
        ("relational-confirmation.json", "sources"),
        ("relational-topology.json", "source_sha256"),
        ("relational-links.json", "source_sha256"),
        ("relational-minecraft.json", "sources"),
    ],
)
def test_report_sources_match_executed_input_identities(name, field):
    report = json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))
    for path, expected in report[field].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path


def test_observer_table_and_goal_quotient_independently_match_sign_binding():
    receipt = json.loads(
        (ROOT / "research/exact/relational-observer.json").read_text(encoding="utf-8")
    )
    states = list(itertools.product([-1, 1], repeat=3))
    for entry in receipt["studies"]:
        assert entry["verify"]["answer"]["kind"] == "valid"
        findings = entry["answer"]["study"]["findings"]
        value = next(f["value"] for f in findings if f["asks"] == "result")
        source, outcome = value["source"], value["result"]
        assert source["states"] == [str(s) for s in states]
        assert source["goals"] == [[str(x * y * z)] for x, y, z in states]
        for i, action in enumerate(source["actions"]):
            assert action["next"] == [
                states.index(tuple(-v if k == i else v for k, v in enumerate(s)))
                for s in states
            ]
        if entry["case"] == "pair_products":
            assert value["candidate"] == [
                str((x * y, x * z, y * z)) for x, y, z in states
            ]
            assert not outcome["sufficient"]
            separator = outcome["separator"]
            assert (
                value["candidate"][separator["left"]]
                == value["candidate"][separator["right"]]
            )
            assert (
                source["goals"][separator["left"]]
                != source["goals"][separator["right"]]
            )
        else:
            assert outcome["sufficient"]
            assert value["candidate"] == [str(x * y * z) for x, y, z in states]
            assert len(set(outcome["partition"])) == 2


def test_native_outcome_binding_has_real_junit_and_fails_after_world_change():
    report = json.loads(
        (ROOT / "research/relational-minecraft.json").read_text(encoding="utf-8")
    )
    xml = ET.parse(ROOT / "research/exact/relational-minecraft.xml").getroot()
    assert len(list(xml.iter("testcase"))) == report["junit_cases"] > 0
    assert not list(xml.iter("failure")) and not list(xml.iter("error"))
    assert report["native_moves"] == 560
    assert report["native_requests"] == 784
    models = report["models"]
    assert models["context_fiber"]["translated_map_frozen"]["success_rate"] == 1
    assert models["action_only"]["translated_map_frozen"]["success_rate"] == 0.5
    assert models["inverted_binding"]["translated_map_frozen"]["success_rate"] == 0
    assert (
        models["context_fiber"]["reversed_map_before_relearning"]["success_rate"] == 0
    )
    assert models["context_fiber"]["reversed_map_after_relearning"]["success_rate"] == 1
    assert len(report["capture_sha256"]) == 64 and report["capture_bytes"] > 0


def test_outcome_table_preserves_contradiction_and_frozen_query_is_read_only():
    table = OutcomeTable("context_fiber")
    table.observe(-1, "north", 1.0)
    table.observe(-1, "north", -1.0)
    before = table.summary()
    assert before["cells"][0]["mean_native_terminal_reward"] == 0
    assert before["cells"][0]["count"] == 2
    table.choose(-1, ["north", "south"])
    assert table.summary() == before
    with pytest.raises(ValueError):
        table.observe(-1, "north", 0.25)
