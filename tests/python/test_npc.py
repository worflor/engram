"""Experience learning, temporal ablation, replay and independent host contracts."""

import json
from pathlib import Path
import sys
import numpy as np
import pytest
from engram_codec import EngramFile, fit_all, _to_complex
from engram_npc import EngramNPC, WitnessSchema, WitnessHistory, DynamicsPrior

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from npc_experience import SCHEMA, study, episode, WitnessMaze


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_incremental_fit_matches_complete_centered_reference(dim):
    rng = np.random.default_rng(15)
    x = rng.normal(size=(150, dim))
    history = WitnessHistory(dim)
    for i, row in enumerate(x):
        fitted = history.push(row)
        if i < 2:
            assert fitted is None
        else:
            z = _to_complex(x[: i + 1] - x[: i + 1].mean(axis=0))
            k, g, _, rms = fit_all(z)
            np.testing.assert_allclose(fitted.k, k, atol=1e-8)
            np.testing.assert_allclose(fitted.g, g, atol=1e-8)
            np.testing.assert_allclose(fitted.pair_rms, rms, atol=1e-6)


def test_history_is_translation_invariant_with_constant_storage():
    x = np.random.default_rng(4).normal(size=(5000, 4))
    a, b = WitnessHistory(4), WitnessHistory(4)
    for row in x:
        a.push(row)
        b.push(row + np.array([100, -3, 5, 20]))
    np.testing.assert_allclose(a.signature().k, b.signature().k, atol=1e-9)
    assert a.samples == 5000 and len(a.previous) == 2 and a.scatter.shape == (2, 3, 3)


def test_decaying_joint_fit_matches_weighted_complete_sum():
    x = np.random.default_rng(3).normal(size=(50, 4))
    decay = 0.93
    h = WitnessHistory(4, decay)
    for row in x:
        h.push(row)
    z = _to_complex(x)
    weights = decay ** np.arange(len(x) - 1, -1, -1)
    mean = np.sum(z * weights[:, None], axis=0) / weights.sum()
    centered = z - mean
    w = decay ** np.arange(len(x) - 3, -1, -1)
    for pair in range(2):
        a, b, y = centered[1:-1, pair], centered[:-2, pair], centered[2:, pair]
        design = np.column_stack([a, -b])
        ridge = max(
            float(np.finfo(np.float32).eps),
            1e-4 * np.sum(w * (abs(a) ** 2 + abs(b) ** 2)) / 2,
        )
        result = np.linalg.solve(
            design.conj().T @ (w[:, None] * design) + ridge * np.eye(2),
            design.conj().T @ (w * y),
        )
        np.testing.assert_allclose(
            [h.signature().k[pair], h.signature().g[pair]], result, atol=1e-9
        )


@pytest.mark.parametrize("seed", [1, 7, 19])
def test_temporal_experience_learns_delayed_cue(seed):
    _, report = study(seed=seed, training=120, evaluation=20, planning=5)
    assert report["success_rate"] >= 0.9
    assert report["mean_steps"] <= 10


def test_disabling_fitted_history_exposes_state_aliasing():
    _, report = study(
        seed=7, training=120, evaluation=40, dynamics_weight=0, planning=5
    )
    assert report["success_rate"] == 0.5


def test_fitted_context_distinguishes_identical_junction_witnesses():
    agents = []
    for cue in [-1, 1]:
        agent = EngramNPC(SCHEMA)
        world = WitnessMaze()
        observation = world.reset(cue)
        for _ in range(3):
            agent.observe(observation["witness"])
        for _ in range(3):
            observation, _, _ = world.step("east")
            agent.observe(observation["witness"])
        agents.append(agent)
    np.testing.assert_array_equal(agents[0].current_vector, agents[1].current_vector)
    assert (
        agents[0]
        .history.signature()
        .compare(agents[1].history.signature())["kg_rms_distance"]
        > 0.05
    )


def test_response_changes_when_environment_changes():
    agent, _ = study(seed=7, training=120, evaluation=20, planning=5)
    environment = WitnessMaze()
    before = [
        episode(
            agent, environment, 1 if i % 2 else -1, train=False, reversed_goal=True
        )["success"]
        for i in range(20)
    ]
    for i in range(40):
        episode(agent, environment, 1 if i % 2 else -1, reversed_goal=True, planning=5)
    after = [
        episode(
            agent, environment, 1 if i % 2 else -1, train=False, reversed_goal=True
        )["success"]
        for i in range(20)
    ]
    assert sum(before) / 20 <= 0.1 and sum(after) / 20 >= 0.9


def test_tailored_checkpoint_restores_policy_rng_history_and_traces(tmp_path):
    agent, _ = study(seed=7, training=60, evaluation=10, planning=5)
    environment = WitnessMaze()
    observation = environment.reset(1)
    agent.begin_episode()
    for _ in range(3):
        agent.observe(observation["witness"])
    first = agent.decide(observation["actions"])
    next_observation, reward, terminal = environment.step(first.action)
    agent.feedback(
        first,
        next_observation["witness"],
        reward,
        next_actions=next_observation["actions"],
    )
    path = tmp_path / "npc.engram"
    agent.save(path)
    restored = EngramNPC.load(path, schema=SCHEMA)
    assert restored.memory._well_count == agent.memory._well_count
    assert restored.traces == agent.traces
    a = agent.decide(next_observation["actions"])
    b = restored.decide(next_observation["actions"])
    assert (a.action, a.well, a.value, a.visits) == (
        b.action,
        b.well,
        b.value,
        b.visits,
    )
    np.testing.assert_array_equal(
        restored.history.signature().k, agent.history.signature().k
    )
    with pytest.raises(ValueError):
        EngramNPC.load(
            path,
            schema=WitnessSchema(SCHEMA.channels, representation="another-environment"),
        )


def test_evaluation_and_planning_do_not_fabricate_observed_experience():
    agent, _ = study(seed=3, training=40, evaluation=10)
    counts = agent.memory._well_count.copy()
    updates = agent.updates
    visits = {
        n: {a: s.visits for a, s in w.actions.items()} for n, w in agent.wells.items()
    }
    agent.dream(100)
    assert counts == agent.memory._well_count and agent.updates == updates
    assert visits == {
        n: {a: s.visits for a, s in w.actions.items()} for n, w in agent.wells.items()
    }
    episode(agent, WitnessMaze(), 1, train=False)
    assert counts == agent.memory._well_count and agent.updates == updates


def test_decision_feedback_is_single_use_and_agent_specific(tmp_path):
    agents = [EngramNPC(WitnessSchema(("x", "y")), seed=7) for _ in range(2)]
    for agent in agents:
        for _ in range(3):
            agent.observe([0, 0])
    a, b = [agent.decide(["respond"]) for agent in agents]
    with pytest.raises(ValueError):
        agents[0].feedback(b, [0, 0], 1)
    with pytest.raises(ValueError):
        agents[0].save(tmp_path / "pending.engram")
    with pytest.raises(ValueError):
        agents[0].feedback(a, [np.nan, 0], 1)
    assert agents[0].pending == a
    agents[0].feedback(a, [0, 0], 1, terminal=True)
    with pytest.raises(ValueError):
        agents[0].feedback(a, [0, 0], 1)
    assert agents[1].pending == b


def test_odd_schemas_and_open_action_vocabulary():
    schema = WitnessSchema(("position", "temperature", "signal"), scales=(10, 2, 1))
    agent = EngramNPC(schema, novelty_radius=0)
    for i in range(150):
        agent.observe([i, 1, 2])
    assert agent.schema.dim == 4 and len(agent.wells) > 100
    actions = [f"host-action-{i}" for i in range(300)]
    result = agent.decide(actions)
    assert result.action in actions
    agent.feedback(result, [151, 1, 2], 1, next_actions=actions)
    assert len(agent.wells[agent.current_well].available) == 300


def test_generic_dynamics_seed_requires_correct_coordinates(tmp_path):
    memory = EngramFile.create("", dim=4)
    x = np.sin(np.arange(30)[:, None] * 0.2 + np.arange(4))
    memory.fast_absorb(x, source="seed")
    path = tmp_path / "seed.engram"
    memory.save(str(path))
    with pytest.raises(ValueError, match="no NPC experience"):
        EngramNPC.load(path)
    prior = DynamicsPrior.load(path, representation="sensor-vectors-v1")
    for row in x:
        profile = prior.observe(row, representation="sensor-vectors-v1")
    assert profile[0]["well"] == "seed" and profile[0]["distance"] < 1e-5
    with pytest.raises(ValueError):
        prior.observe(np.zeros(2), representation="sensor-vectors-v1")
    with pytest.raises(ValueError):
        prior.observe(np.zeros(4), representation="incompatible-vectors")


def test_frozen_alexandria_dynamics_prior_retains_missing_fields():
    root = Path(__file__).resolve().parents[2]
    prior = DynamicsPrior.load(
        root / "research/artifacts/alexandria.engram",
        representation="matching-300d-input-required",
    )
    with pytest.raises(ValueError):
        prior.observe(np.zeros(8), representation=prior.representation)
    assert prior.memory.dim == 300 and len(prior.memory.wells()) == 225
    for _ in range(3):
        profile = prior.observe(np.zeros(300), representation=prior.representation)
    assert all(row["statistics"] == "K" for row in profile)
    from engram_network import WellNetwork

    network = WellNetwork(
        300, memory=prior.memory.copy(), pairing=prior.memory._reference_pairing
    )
    with pytest.raises(ValueError, match="lacks measured G/RMS"):
        network.oscillator(next(iter(network.memory.wells())))


@pytest.mark.parametrize("bad", [[], [""], [1], ["ok", 1], "walk"])
def test_invalid_actions_are_rejected(bad):
    agent = EngramNPC(WitnessSchema(("x", "y")))
    agent.observe([0, 0])
    with pytest.raises(ValueError):
        agent.decide(bad)


def test_malformed_experience_state_is_rejected(tmp_path):
    agent = EngramNPC(WitnessSchema(("x", "y")))
    for _ in range(3):
        agent.observe([0, 0])
    path = tmp_path / "bad.engram"
    agent.save(path)
    memory = EngramFile.load(path)
    data = json.loads(memory._extra_sections["npc.json"])
    data["wells"][next(iter(data["wells"]))]["center"] = [float("nan")]
    memory._extra_sections["npc.json"] = json.dumps(data).encode()
    memory.save(str(path))
    with pytest.raises(ValueError):
        EngramNPC.load(path)
