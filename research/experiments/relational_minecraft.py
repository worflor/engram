"""Probe whether simple relational memory helps on the native Minecraft T corridor.

This is a controlled outcome-table experiment, not an EngramNPC or Pets Plus
integration. Minecraft provides every witness and reward. The cue is retained
as an explicit host context key so the experiment tests context/action binding,
not whether the oscillator history discovers a latent context by itself.

The expensive GameTest is intentionally opt-in by running this module. It owns
only the Gradle process tree it starts and uses the shared Minecraft heavy lock.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engram_relations import ContextRelations

from tools.npc_lane import (  # noqa: E402
    MinecraftEnvironment,
    atomic_json,
    heavy_lock,
    read_json,
    short,
    source_hashes as lane_source_hashes,
)

BRANCHES = ("north", "south")
CONTEXTS = (-1, 1)
REPEATS_PER_CONTEXT_BRANCH = 2
EPISODES_PER_EVALUATION = 8
EAST_APPROACH_STEPS = 3
BRANCH_STEPS = 2
MODEL_NAMES = ("action_only", "context_fiber", "inverted_binding")


class OutcomeTable:
    """Small empirical mean-reward table with an explicit context-key rule."""

    def __init__(self, mode: str):
        if mode not in MODEL_NAMES:
            raise ValueError(f"unknown model mode: {mode}")
        self.mode = mode
        self.relations = ContextRelations(
            ("native_terminal_reward",),
            representation="minecraft-cue-action-outcomes-v1",
        )
        self.updates = 0

    def _training_key(self, cue: int, action: str) -> tuple[int, str] | str:
        if self.mode == "action_only":
            return action
        if self.mode == "inverted_binding":
            # Deliberate negative control: store under the opposite context,
            # while prediction uses the observed (unflipped) context.
            return (-cue, action)
        return (cue, action)

    def _prediction_key(self, cue: int, action: str) -> tuple[int, str] | str:
        if self.mode == "action_only":
            return action
        return (cue, action)

    def observe(self, cue: int, action: str, native_reward: float) -> None:
        if cue not in CONTEXTS or action not in BRANCHES:
            raise ValueError("cue/action outside the declared experiment vocabulary")
        if native_reward not in (-1.0, 1.0):
            raise ValueError("only native terminal rewards may train this table")
        key = json.dumps(self._training_key(cue, action), separators=(",", ":"))
        self.relations.push(key, [native_reward])
        self.updates += 1

    def choose(self, cue: int, available: list[str]) -> str:
        if cue not in CONTEXTS or not available:
            raise ValueError("prediction requires a native cue and legal action")
        # North is the declared deterministic tie break. This matters for the
        # pooled action-only model, whose balanced data should remain ambiguous.
        ordered = [action for action in BRANCHES if action in available]
        if not ordered:
            raise ValueError(f"no recognized branch actions in {available!r}")

        def empirical_mean(action: str) -> float:
            key = json.dumps(self._prediction_key(cue, action), separators=(",", ":"))
            item = self.relations.moments(key)
            return float(item.mean[0]) if item is not None else 0.0

        return max(ordered, key=empirical_mean)

    def summary(self) -> dict:
        rows = []
        for key in self.relations.contexts:
            moment = self.relations.moments(key)
            rows.append(
                {
                    "key": json.loads(key),
                    "count": moment.count,
                    "mean_native_terminal_reward": float(moment.mean[0]),
                }
            )
        return {"mode": self.mode, "updates": self.updates, "cells": rows}


def observed_cue(response: dict) -> int:
    """Read the cue from the actual reset witness; reject absent/ambiguous cues."""
    witness = response.get("witness")
    if not isinstance(witness, list) or len(witness) != 8:
        raise AssertionError(
            "native reset did not provide the expected 8-channel witness"
        )
    value = float(witness[1])
    if value not in (-1.0, 1.0):
        raise AssertionError(f"native reset cue channel was {value!r}")
    return int(value)


def native_episode(
    environment: MinecraftEnvironment,
    *,
    requested_cue: int,
    branch: str,
    reversed_goal: bool = False,
) -> dict:
    """Run the declared scripted approach/branch and return native terminal data."""
    if requested_cue not in CONTEXTS or branch not in BRANCHES:
        raise ValueError("invalid scripted encounter")

    reset = environment.reset(requested_cue, reversed_goal=reversed_goal)
    cue = observed_cue(reset)
    if cue != requested_cue:
        raise AssertionError("native cue marker did not match the reset request")

    for _ in range(EAST_APPROACH_STEPS):
        _, _, terminal = environment.step("east")
        if terminal:
            raise AssertionError("approach unexpectedly reached a terminal floor")

    junction = environment.sample()
    if not isinstance(junction.get("actions"), list):
        raise AssertionError("junction response omitted native legal actions")
    if float(junction["witness"][1]) != 0.0:
        raise AssertionError("cue should be hidden at the junction")
    if not (2.99 <= float(junction["position"][0]) <= 3.01):
        raise AssertionError("three east moves did not reach the native junction")
    if branch not in junction["actions"]:
        raise AssertionError(f"scripted branch {branch} is not natively legal")

    terminal_result = None
    for index in range(BRANCH_STEPS):
        result, _, terminal = environment.step(branch)
        if terminal:
            if index != BRANCH_STEPS - 1:
                raise AssertionError("endpoint was reached earlier than declared")
            terminal_result = result
            break
    if terminal_result is None:
        raise AssertionError("branch did not reach a native endpoint")

    reward = float(terminal_result["reward"])
    floor = str(terminal_result["floor"])
    if reward not in (-1.0, 1.0):
        raise AssertionError(f"unexpected native terminal feedback {reward!r}")
    if not any(name in floor for name in ("emerald_block", "magma_block")):
        raise AssertionError(f"terminal floor was not native emerald/magma: {floor}")
    return {
        "cue_context": cue,
        "branch_action": branch,
        "terminal_reward": reward,
        "success": reward > 0,
        "terminal_floor": floor,
        "terminal_health": float(terminal_result["witness"][7]),
        "reversed_goal": reversed_goal,
    }


def balanced_encounters(seed: int) -> list[tuple[int, str]]:
    pairs = [
        (cue, branch)
        for cue in CONTEXTS
        for branch in BRANCHES
        for _ in range(REPEATS_PER_CONTEXT_BRANCH)
    ]
    # A fixed, reproducible ordering avoids a systematic cue/action chronology.
    import random

    random.Random(seed).shuffle(pairs)
    return pairs


def collect_training(
    environment: MinecraftEnvironment,
    *,
    seed: int,
    reversed_goal: bool,
) -> tuple[list[dict], dict[str, OutcomeTable]]:
    samples = []
    tables = {name: OutcomeTable(name) for name in MODEL_NAMES}
    for cue, branch in balanced_encounters(seed):
        sample = native_episode(
            environment,
            requested_cue=cue,
            branch=branch,
            reversed_goal=reversed_goal,
        )
        samples.append(sample)
        for table in tables.values():
            table.observe(sample["cue_context"], branch, sample["terminal_reward"])
    if len(samples) != 8:
        raise AssertionError("the training design must collect eight native encounters")
    return samples, tables


def evaluate_model(
    environment: MinecraftEnvironment,
    table: OutcomeTable,
    *,
    reversed_goal: bool,
    offset: int,
    seed: int,
) -> dict:
    old_offset = environment.offset
    environment.offset = offset
    before_updates = table.updates
    samples = []
    try:
        schedule = [
            (cue, branch) for cue in CONTEXTS for branch in BRANCHES for _ in range(2)
        ]
        import random

        random.Random(seed).shuffle(schedule)
        for requested_cue, _ in schedule:
            reset = environment.reset(requested_cue, reversed_goal=reversed_goal)
            cue = observed_cue(reset)
            if cue != requested_cue:
                raise AssertionError("evaluation cue differed from the native marker")
            for _ in range(EAST_APPROACH_STEPS):
                _, _, terminal = environment.step("east")
                if terminal:
                    raise AssertionError("evaluation approach reached an endpoint")
            junction = environment.sample()
            if float(junction["witness"][1]) != 0.0:
                raise AssertionError(
                    "cue should be hidden from the current junction witness"
                )
            if not (2.99 <= float(junction["position"][0]) <= 3.01):
                raise AssertionError("evaluation did not reach the native junction")
            action = table.choose(cue, junction["actions"])
            if action not in junction["actions"]:
                raise AssertionError("model selected a non-native action")
            result = None
            for index in range(BRANCH_STEPS):
                result, _, terminal = environment.step(action)
                if terminal:
                    if index != BRANCH_STEPS - 1:
                        raise AssertionError(
                            "model branch reached an endpoint too early"
                        )
                    break
            if result is None or not result["terminal"]:
                raise AssertionError("model decision did not reach a native terminal")
            reward = float(result["reward"])
            floor = str(result["floor"])
            if reward not in (-1.0, 1.0) or not any(
                name in floor for name in ("emerald_block", "magma_block")
            ):
                raise AssertionError(
                    "evaluation outcome was not native terminal feedback"
                )
            samples.append(
                {
                    "cue_context": cue,
                    "chosen_action": action,
                    "terminal_reward": reward,
                    "success": reward > 0,
                    "terminal_floor": floor,
                    "terminal_health": float(result["witness"][7]),
                }
            )
        if table.updates != before_updates:
            raise AssertionError("frozen evaluation changed its outcome table")
    finally:
        environment.offset = old_offset
    return {
        "episodes": len(samples),
        "successes": sum(item["success"] for item in samples),
        "success_rate": sum(item["success"] for item in samples) / len(samples),
        "mean_native_terminal_reward": sum(item["terminal_reward"] for item in samples)
        / len(samples),
        "decisions": samples,
    }


def source_hashes() -> dict[str, str]:
    hashes = lane_source_hashes()
    path = Path(__file__).resolve()
    hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    hashes["engram_relations.py"] = hashlib.sha256(
        (ROOT / "engram_relations.py").read_bytes()
    ).hexdigest()
    return dict(sorted(hashes.items()))


def run_study(args, environment: MinecraftEnvironment, folder: Path) -> dict:
    report = {
        "schema": "engram.relational-minecraft/v1",
        "seed": args.seed,
        "universe": "native Minecraft 1.21.1 T-corridor GameTest; villager movement, collisions, wool cue, emerald/magma outcomes",
        "claim_scope": "empirical context/action outcome-table probe; not EngramNPC training, semantic inference, or Pets Plus integration",
        "context_binding": "cue is read from native reset witness and retained explicitly by the host until the cue-hidden junction",
        "feedback_source": "native terminal reward and floor returned by Minecraft; no cue-derived reward labels",
        "memory_implementation": "root engram_relations.ContextRelations; action/cue keys and scalar native reward moments",
        "training_design": {
            "encounters_per_map": 8,
            "balanced_context_action_pairs": 4,
            "repetitions_per_pair": 2,
            "approach": ["east"] * EAST_APPROACH_STEPS,
            "branch_continuation_steps": BRANCH_STEPS,
        },
        "evaluation_episodes_per_model_per_phase": EPISODES_PER_EVALUATION,
        "models": {},
    }

    default_samples, default_tables = collect_training(
        environment, seed=args.seed, reversed_goal=False
    )
    report["default_training"] = {
        "samples": default_samples,
        "tables": {name: table.summary() for name, table in default_tables.items()},
    }

    for name, table in default_tables.items():
        report["models"][name] = {
            "default_map_frozen": evaluate_model(
                environment,
                table,
                reversed_goal=False,
                offset=0,
                seed=args.seed + 101,
            ),
            "translated_map_frozen": evaluate_model(
                environment,
                table,
                reversed_goal=False,
                offset=24,
                seed=args.seed + 202,
            ),
            "reversed_map_before_relearning": evaluate_model(
                environment,
                table,
                reversed_goal=True,
                offset=0,
                seed=args.seed + 303,
            ),
        }

    reverse_samples, reverse_tables = collect_training(
        environment, seed=args.seed + 404, reversed_goal=True
    )
    report["reversed_relearning"] = {
        "native_encounters": reverse_samples,
        "tables": {name: table.summary() for name, table in reverse_tables.items()},
        "note": "fresh tables trained on eight new native feedback encounters; this measures relearning from the reversed map, not online forgetting",
    }
    for name, table in reverse_tables.items():
        report["models"][name]["reversed_map_after_relearning"] = evaluate_model(
            environment,
            table,
            reversed_goal=True,
            offset=0,
            seed=args.seed + 505,
        )

    expected_rates = {
        "default_map_frozen": {
            "action_only": 0.5,
            "context_fiber": 1.0,
            "inverted_binding": 0.0,
        },
        "translated_map_frozen": {
            "action_only": 0.5,
            "context_fiber": 1.0,
            "inverted_binding": 0.0,
        },
        "reversed_map_before_relearning": {
            "action_only": 0.5,
            "context_fiber": 0.0,
            "inverted_binding": 1.0,
        },
        "reversed_map_after_relearning": {
            "action_only": 0.5,
            "context_fiber": 1.0,
            "inverted_binding": 0.0,
        },
    }
    for phase, model_rates in expected_rates.items():
        for name, expected in model_rates.items():
            measured = report["models"][name][phase]["success_rate"]
            if measured != expected:
                raise AssertionError(
                    f"{name}/{phase}: expected {expected:.1%} from the balanced "
                    + f"fixture, observed {measured:.1%}"
                )

    report["native_requests"] = environment.requests
    report["native_moves"] = environment.native_moves
    report["semantic_trace_sha256"] = environment.trace_digest.hexdigest()
    atomic_json(folder / "study.json", report)
    return report


def terminate_owned_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def execute(args) -> dict:
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "tmp").mkdir(exist_ok=True)
    owner = uuid.uuid4().hex
    folder = work / "relational-runs" / owner
    folder.mkdir(parents=True)
    lane = folder / "lane"
    lane.mkdir()
    inputs = source_hashes()
    atomic_json(folder / "source.json", inputs)
    java = args.java
    if not java.is_file():
        raise RuntimeError(f"JDK 25 launcher not found: {java}")
    env = dict(os.environ)
    env["JAVA_TOOL_OPTIONS"] = (
        env.get("JAVA_TOOL_OPTIONS", "")
        + " -Djdk.net.unixdomain.tmpdir="
        + short(work / "tmp")
    ).strip()
    command = [
        str(java),
        "-cp",
        str(ROOT / "integrations/minecraft/gradle/wrapper/gradle-wrapper.jar"),
        "org.gradle.wrapper.GradleWrapperMain",
        "-p",
        str(ROOT / "integrations/minecraft"),
        "--no-daemon",
        "--console=plain",
        "--project-cache-dir",
        str(work / "project-cache"),
        "-Pengram.buildDir=" + str(work / "build"),
        "-Pengram.report=" + str(folder / "junit.xml"),
        "-Pengram.lane=" + str(lane),
        "-Pengram.session=" + owner,
        "-Pengram.runDir=" + str(folder / "game"),
        "runGametest",
    ]
    lock = args.heavy_lock
    print(f"Relational Minecraft run: {folder}", flush=True)
    process = None
    report = None
    with (
        heavy_lock(lock, owner),
        (folder / "gradle.log").open("w", encoding="utf-8") as log,
    ):
        options = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, **options
        )
        try:
            ready_path = lane / "ready.json"
            deadline = time.monotonic() + 300
            while not ready_path.is_file():
                if process.poll() is not None:
                    raise RuntimeError("Minecraft exited before becoming ready")
                if time.monotonic() > deadline:
                    raise TimeoutError("Minecraft readiness timeout")
                time.sleep(0.05)
            ready = read_json(ready_path)
            if ready.get("session") != owner:
                raise AssertionError("GameTest readiness session does not match")
            environment = MinecraftEnvironment(lane, owner, process)
            report = run_study(args, environment, folder)
            environment.request("finish")
            code = process.wait(timeout=1800)
            if code:
                raise RuntimeError(f"Gradle exited {code}")

            junit = ET.parse(folder / "junit.xml").getroot()
            cases = list(junit.iter("testcase"))
            if not cases or list(junit.iter("failure")) or list(junit.iter("error")):
                raise AssertionError("nonempty passing native JUnit receipt required")
            if source_hashes() != inputs:
                raise AssertionError(
                    "research, harness or game sources changed during run"
                )

            capture = folder / "native-capture.jsonl"
            shutil.copy2(lane / "native.jsonl", capture)
            exact_xml = ROOT / "research/exact/relational-minecraft.xml"
            exact_xml.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / "junit.xml", exact_xml)
            report.update(
                {
                    "junit_cases": len(cases),
                    "world_seed": ready["seed"],
                    "sources": inputs,
                    "capture": str(capture),
                    "run_directory": str(folder),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            atomic_json(folder / "study.json", report)
            # Keep the maintained receipt reduced: raw acknowledgements, logs,
            # GameTest output and build products remain on the external scratch disk.
            reduced = {
                key: value
                for key, value in report.items()
                if key
                not in {
                    "default_training",
                    "reversed_relearning",
                    "models",
                    "capture",
                    "run_directory",
                }
            }
            reduced["run_id"] = folder.name
            reduced["capture_sha256"] = hashlib.sha256(capture.read_bytes()).hexdigest()
            reduced["capture_bytes"] = capture.stat().st_size
            reduced["models"] = {
                name: {
                    phase: {
                        key: value
                        for key, value in outcome.items()
                        if key != "decisions"
                    }
                    for phase, outcome in phases.items()
                }
                for name, phases in report["models"].items()
            }
            reduced["default_training"] = {
                "encounters": len(report["default_training"]["samples"]),
                "tables": report["default_training"]["tables"],
            }
            reduced["reversed_relearning"] = {
                "encounters": len(report["reversed_relearning"]["native_encounters"]),
                "tables": report["reversed_relearning"]["tables"],
                "note": report["reversed_relearning"]["note"],
            }
            atomic_json(ROOT / "research/relational-minecraft.json", reduced)
            print(json.dumps(reduced, indent=2), flush=True)
            return report
        except BaseException:
            if process is not None:
                terminate_owned_tree(process)
            lines = (
                (folder / "gradle.log")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
            print("\n".join(lines[-80:]), flush=True)
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--heavy-lock",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--java",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


if __name__ == "__main__":
    execute(parse_args())
