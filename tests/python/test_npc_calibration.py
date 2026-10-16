"""Training-only coordinates and conservation of saved observed experience."""

import json
import numpy as np
import pytest
from engram_codec import EngramFile
from engram_npc import EngramNPC, WitnessSchema


def test_stream_calibration_freezes_training_units_and_constant_channels(tmp_path):
    x = np.random.default_rng(31).normal(size=(1000, 3)) * [2, 100, 0] + [5, -30, 7]
    schema = WitnessSchema.calibrate(
        ("x", "y", "constant"), iter(x), representation="training-v1"
    )
    np.testing.assert_allclose(schema.offsets, x.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(schema.scales[:2], x.std(axis=0)[:2], atol=1e-12)
    assert schema.scales[2] == 1 and schema.dim == 4
    npc = EngramNPC(schema)
    for row in x[:10]:
        npc.observe(row)
    npc.save(tmp_path / "calibrated.engram")
    loaded = EngramNPC.load(tmp_path / "calibrated.engram", schema=schema)
    assert loaded.schema.identity == schema.identity
    np.testing.assert_array_equal(loaded.schema.vector(x[50]), schema.vector(x[50]))
    # Evaluation never re-estimates units from a changed population.
    npc.observe(x[0] + 100, learn=False)
    assert npc.schema.identity == schema.identity


def test_calibration_respects_host_noise_floors():
    schema = WitnessSchema.calibrate(
        ("x", "y"), [[1, 2], [1, 4]], representation="fixed", minimum_scales=(0.1, 2)
    )
    assert schema.offsets == (1, 3) and schema.scales == (0.1, 2)


def test_training_calibration_removes_positive_unit_changes():
    x = np.random.default_rng(12).normal(size=(100, 3))
    converted = x * [1000, 2, 0.01] + [20, -10, 5]
    a = WitnessSchema.calibrate(("a", "b", "c"), x, representation="base")
    b = WitnessSchema.calibrate(("a", "b", "c"), converted, representation="converted")
    np.testing.assert_allclose(
        [a.vector(row) for row in x], [b.vector(row) for row in converted], atol=1e-12
    )


def test_checkpoint_continues_observation_learning_and_planning_exactly(tmp_path):
    npc = EngramNPC(WitnessSchema(("x", "y")), seed=83)
    for i in range(3):
        npc.observe([i, 0])
    restored = None
    path = tmp_path / "continued.engram"
    for i in range(150):
        if i == 47:
            npc.save(path)
            restored = EngramNPC.load(path)
        actions = {"move", "wait", "look"}
        decision = npc.decide(actions)
        reward = 1 if decision.action == "look" else -0.1
        witness = [np.sin(i / 7), np.cos(i / 13)]
        npc.feedback(decision, witness, reward, next_actions=actions)
        npc.dream(3)
        if restored is not None:
            twin = restored.decide(reversed(sorted(actions)))
            assert (
                decision.action,
                decision.well,
                decision.value,
                decision.visits,
            ) == (twin.action, twin.well, twin.value, twin.visits)
            restored.feedback(twin, witness, reward, next_actions=actions)
            restored.dream(3)
    npc.save(path)
    expected = json.loads(npc.memory._extra_sections["npc.json"])
    restored.save(path)
    assert json.loads(restored.memory._extra_sections["npc.json"]) == expected


@pytest.mark.parametrize(
    "samples", [[], [[1, 2]], [[1, 2], [float("nan"), 2]], [[1], [2]]]
)
def test_invalid_calibration_is_rejected(samples):
    with pytest.raises(ValueError):
        WitnessSchema.calibrate(("x", "y"), samples, representation="fixed")


@pytest.mark.parametrize("mutation", ["inventory", "visits", "updates", "available"])
def test_checkpoint_reconciles_observed_experience(tmp_path, mutation):
    npc = EngramNPC(WitnessSchema(("x", "y")))
    for _ in range(3):
        npc.observe([0, 0])
    decision = npc.decide(["respond"])
    npc.feedback(decision, [0, 0], 1, terminal=True)
    path = tmp_path / "corrupt.engram"
    npc.save(path)
    memory = EngramFile.load(path)
    data = json.loads(memory._extra_sections["npc.json"])
    well = next(iter(data["wells"].values()))
    if mutation == "inventory":
        data["wells"].clear()
    elif mutation == "visits":
        well["actions"]["respond"]["visits"] += 1
    elif mutation == "updates":
        data["updates"] += 1
    else:
        well["available"] = [0]
    memory._extra_sections["npc.json"] = json.dumps(data).encode()
    memory.save(str(path))
    with pytest.raises(ValueError):
        EngramNPC.load(path)
