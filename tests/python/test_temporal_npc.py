"""Goal-specific temporal distinctions and native checkpoint representation."""

import json
from pathlib import Path
import sys
import numpy as np
import pytest
from engram_codec import EngramFile
from engram_npc import EngramNPC, WitnessSchema

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from temporal_witness import study, encounter, HISTORIES


@pytest.mark.parametrize("seed", [7, 19, 81])
def test_multiscale_context_learns_the_declared_recency_goal(seed):
    _, cumulative = study((1.0,), seed=seed)
    agent, multiscale = study((1.0, 0.9), seed=seed)
    assert cumulative["success_rate"] == 0.5
    assert multiscale["success_rate"] == 1.0
    assert len(agent.histories) == 2 and agent.history is agent.histories[0]
    assert agent.history.samples == agent.histories[1].samples


def test_temporal_bank_checkpoint_resumes_all_scales_and_policy(tmp_path):
    agent, _ = study((1.0, 0.9, 0.5))
    path = tmp_path / "temporal.engram"
    agent.save(path)
    loaded = EngramNPC.load(path, schema=agent.schema)
    assert loaded.config == agent.config
    for a, b in zip(agent.histories, loaded.histories):
        assert a.snapshot() == b.snapshot()
    for values in HISTORIES:
        assert encounter(agent, values, train=False) == encounter(
            loaded, values, train=False
        )


def test_v1_checkpoint_restores_policy_without_fabricating_history_endpoints(tmp_path):
    agent = EngramNPC(WitnessSchema(("x", "y")))
    for i in range(7):
        agent.observe([i, 0])
    path = tmp_path / "old.engram"
    agent.save(path)
    memory = EngramFile.load(path)
    data = json.loads(memory._extra_sections["npc.json"])
    data["version"] = 1
    data["history"] = data.pop("histories")[0]
    data["history"].pop("first")
    data["history"].pop("version")
    data["config"]["history_decay"] = data["config"].pop("history_decays")[0]
    memory._extra_sections["npc.json"] = json.dumps(data).encode()
    memory.save(str(path))
    loaded = EngramNPC.load(path)
    assert loaded.history.first is None
    np.testing.assert_array_equal(
        loaded.history.signature().k, agent.history.signature().k
    )
    assert len(loaded.histories) == 1


@pytest.mark.parametrize("decays", [[], [1, 1], [0, 1], [1, 1.1]])
def test_invalid_temporal_scales_are_rejected(decays):
    with pytest.raises(ValueError):
        EngramNPC(WitnessSchema(("x", "y")), history_decays=decays)
