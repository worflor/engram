"""Deterministic NPC stress campaign, without games, corpora or extra packages."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engram_npc import EngramNPC, WitnessHistory, WitnessSchema


def canonical_state(agent, path):
    agent.save(path)
    # Canonical policy/moments, without archive timestamps or ownership tokens.
    payload = agent.memory._extra_sections["npc.json"]
    return hashlib.sha256(
        json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def trajectory(step):
    return [np.sin(step * 0.07), np.cos(step * 0.13), (step % 17) / 17]


def replay(steps=1000):
    schema = WitnessSchema(("x", "y", "phase"), representation="stress-stream-v1")
    agent = EngramNPC(schema, seed=81, novelty_radius=0.25)
    for i in range(3):
        agent.observe(trajectory(i))
    decisions = []
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "checkpoint.engram"
        resumed = None
        for step in range(steps):
            if step == steps // 3:
                agent.save(path)
                resumed = EngramNPC.load(path, schema=schema)
            names = {f"response-{i}" for i in range(17 + step % 4)}
            decision = agent.decide(names)
            reward = (int(decision.action.split("-")[1]) % 5 - 2) / 2
            terminal = step % 47 == 46
            agent.feedback(
                decision,
                trajectory(step + 3),
                reward,
                terminal=terminal,
                next_actions=names,
            )
            agent.dream(3)
            if resumed is not None:
                twin = resumed.decide(reversed(sorted(names)))
                assert (twin.action, twin.well, twin.value, twin.visits) == (
                    decision.action,
                    decision.well,
                    decision.value,
                    decision.visits,
                )
                resumed.feedback(
                    twin,
                    trajectory(step + 3),
                    reward,
                    terminal=terminal,
                    next_actions=names,
                )
                resumed.dream(3)
            if terminal:
                agent.begin_episode()
                for i in range(3):
                    agent.observe(trajectory(step + i))
                if resumed is not None:
                    resumed.begin_episode()
                    for i in range(3):
                        resumed.observe(trajectory(step + i))
            decisions.append(
                [decision.action, decision.well, decision.value, decision.visits]
            )
        state = canonical_state(agent, path)
        assert canonical_state(resumed, path) == state
    return {
        "steps": steps,
        "checkpoint_split": steps // 3,
        "wells": len(agent.wells),
        "state_sha256": state,
        "decisions_sha256": hashlib.sha256(
            json.dumps(decisions, separators=(",", ":")).encode()
        ).hexdigest(),
        "real_updates": agent.updates,
        "planning_updates": agent.planning_updates,
    }


def histories(samples):
    rng = np.random.default_rng(981)
    x = rng.normal(size=(samples, 8)) + np.array([1e6, -1e6, 5, -3, 0, 0, 10, -9])
    results = []
    for decay in (1.0, 0.995, 0.5):
        history = WitnessHistory(8, decay)
        started = time.perf_counter()
        for row in x:
            history.push(row)
        fitted = history.signature()
        # Independent direct weighted normal equations; same centering/ridge.
        z = x[:, ::2] + 1j * x[:, 1::2]
        weights = decay ** np.arange(len(x) - 1, -1, -1)
        mean = np.sum(z * weights[:, None], axis=0) / weights.sum()
        centered = z - mean
        w = decay ** np.arange(len(x) - 3, -1, -1)
        coefficients, residuals = [], []
        for pair in range(4):
            design = np.column_stack([centered[1:-1, pair], -centered[:-2, pair]])
            target = centered[2:, pair]
            ridge = max(
                float(np.finfo(np.float32).eps),
                1e-4 * np.sum(w[:, None] * abs(design) ** 2) / 2,
            )
            kg = np.linalg.solve(
                design.conj().T @ (w[:, None] * design) + ridge * np.eye(2),
                design.conj().T @ (w * target),
            )
            coefficients.append(kg)
            residuals.append(
                np.sqrt(np.sum(w * abs(target - design @ kg) ** 2) / w.sum())
            )
        reference = np.array(coefficients)
        error = float(
            max(
                np.max(abs(fitted.k - reference[:, 0])),
                np.max(abs(fitted.g - reference[:, 1])),
                np.max(abs(fitted.pair_rms - residuals)),
            )
        )
        assert error < 1e-6
        resident = sum(
            a.nbytes
            for a in [
                history.mean,
                history.joint_mean,
                history.scatter,
                *history.previous,
                *(history.first or []),
            ]
        )
        restored = WitnessHistory.restore(
            8, decay, json.loads(json.dumps(history.snapshot()))
        )
        np.testing.assert_array_equal(restored.signature().k, fitted.k)
        results.append(
            {
                "samples": samples,
                "dimension": 8,
                "decay": decay,
                "maximum_reference_error": error,
                "resident_array_bytes": resident,
                "previous_rows": len(history.previous),
                "seconds": time.perf_counter() - started,
            }
        )
        print("History", decay, samples, error, flush=True)
    return results


def growth(wells, actions):
    agent = EngramNPC(WitnessSchema(("x", "y")), novelty_radius=0, seed=31)
    start = time.perf_counter()
    for i in range(wells + 2):
        agent.observe([i / 10, np.sin(i / 7)])
    assert len(agent.wells) == wells
    vocabulary = [f"action-{i}" for i in range(actions)]
    decision = agent.decide(vocabulary)
    agent.feedback(
        decision, [(wells + 2) / 10, 0], 1, terminal=True, next_actions=vocabulary
    )
    count = len(agent.wells)
    updates = agent.updates
    agent.dream(1000)
    assert len(agent.wells) == count and agent.updates == updates
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "grown.engram"
        agent.save(path)
        loaded = EngramNPC.load(path)
        assert len(loaded.wells) == count
        assert len(loaded.wells[loaded.current_well].available) == actions
        size = path.stat().st_size
    return {
        "requested_wells": wells,
        "stored_wells": count,
        "available_actions": actions,
        "real_updates": updates,
        "planning_updates": agent.planning_updates,
        "checkpoint_bytes": size,
        "seconds": time.perf_counter() - start,
    }


def bellman():
    agent = EngramNPC(WitnessSchema(("x", "y")), discount=0.95)
    for _ in range(3):
        agent.observe([0, 0])
    for _ in range(20):
        decision = agent.decide(["continue"])
        agent.feedback(decision, [0, 0], 1, next_actions=["continue"])
    visits = agent.updates
    observed = agent.memory._well_count.copy()
    agent.dream(4000)
    value = agent.responses()["continue"]["value"]
    assert abs(value - 20) < 1e-10
    assert agent.updates == visits and agent.memory._well_count == observed
    return {
        "analytical_value": 20,
        "learned_value": value,
        "observed_updates": visits,
        "planning_updates": agent.planning_updates,
    }


def main(args):
    results = []
    for seed in ("1", "987654"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        raw = subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--steps",
                str(args.steps),
            ],
            env=env,
            cwd=ROOT,
            text=True,
        )
        results.append({"python_hash_seed": seed, **json.loads(raw)})
    assert results[0]["state_sha256"] == results[1]["state_sha256"]
    assert results[0]["decisions_sha256"] == results[1]["decisions_sha256"]
    report = {
        "schema": "engram.npc-stress/v1",
        "scope": "Finite deterministic numerical and state-growth campaign; timings are observations, not performance guarantees",
        "deterministic_replay": results,
        "streaming_history": histories(args.samples),
        "growth": growth(args.wells, args.actions),
        "observed_model_planning": bellman(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
        "sources": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in [
                "engram_npc.py",
                "engram_history.py",
                "engram_codec.py",
                "research/experiments/npc_stress.py",
            ]
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Saved " + str(args.output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--wells", type=int, default=2048)
    parser.add_argument("--actions", type=int, default=4096)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/npc/stress.json"
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(replay(args.steps)))
    else:
        main(args)
