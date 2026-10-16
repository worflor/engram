"""Real Minecraft experience-NPC lane: isolated world, native actor and file acknowledgements.

The Python reference makes decisions. Java supplies native observations and
applies those decisions on the server thread. No production server is contacted.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from engram_npc import EngramNPC
from npc_experience import SCHEMA, episode


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    deadline = time.monotonic() + 5
    while True:
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def read_json(path, deadline=None):
    deadline = time.monotonic() + 5 if deadline is None else deadline
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.005)


@contextmanager
def heavy_lock(path, owner):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            f"heavy Minecraft lock is held: {path}; no unrelated process will be stopped"
        ) from error
    marker = path / "owner.json"
    try:
        atomic_json(
            marker, {"owner": owner, "pid": os.getpid(), "purpose": "engram-npc"}
        )
        yield
    finally:
        if (
            not marker.is_file()
            or json.loads(marker.read_text(encoding="utf-8"))["owner"] != owner
        ):
            raise RuntimeError("lock ownership changed; refusing to release it")
        marker.unlink()
        path.rmdir()


def empty_structure():
    def text(s):
        raw = s.encode()
        return struct.pack(">H", len(raw)) + raw

    def tag(kind, name, payload):
        return bytes([kind]) + text(name) + payload

    body = tag(3, "DataVersion", struct.pack(">i", 3955))
    body += tag(
        9, "size", bytes([3]) + struct.pack(">i", 3) + struct.pack(">iii", 20, 16, 20)
    )
    body += tag(
        9,
        "palette",
        bytes([10])
        + struct.pack(">i", 1)
        + tag(8, "Name", text("minecraft:air"))
        + b"\0",
    )
    for name in ["blocks", "entities"]:
        body += tag(9, name, bytes([10]) + struct.pack(">i", 0))
    target = (
        ROOT
        / "integrations/minecraft/src/main/resources/data/engram_npc/structure/empty.nbt"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(bytes([10]) + text("") + body + b"\0", mtime=0))


def source_hashes():
    project = ROOT / "integrations/minecraft"
    paths = [
        p
        for p in project.rglob("*")
        if p.is_file()
        and not any(
            part in {"build", ".gradle", "run"} for part in p.relative_to(project).parts
        )
    ]
    paths += [
        ROOT / name
        for name in [
            "engram_npc.py",
            "engram_history.py",
            "engram_codec.py",
            "engram_analysis.py",
            "examples/npc_experience.py",
            "tools/npc_lane.py",
        ]
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def short(path):
    if os.name != "nt":
        return str(path)
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    return buffer.value if length else str(path)


class MinecraftEnvironment:
    def __init__(self, lane, session, process):
        self.lane, self.session, self.process = lane, session, process
        self.offset = 0
        self.requests = 0
        self.last_tick = -1
        self.native_moves = 0
        self.trace_digest = hashlib.sha256()

    def request(self, op, **arguments):
        identity = uuid.uuid4().hex
        atomic_json(
            self.lane / "request.json",
            {
                "schema": "engram.minecraft-request/v1",
                "id": identity,
                "session": self.session,
                "op": op,
                **arguments,
            },
        )
        response = self.lane / (identity + ".json")
        deadline = time.monotonic() + 30
        while not response.is_file():
            if self.process.poll() is not None:
                raise RuntimeError("Minecraft exited before acknowledging its action")
            if time.monotonic() > deadline:
                raise TimeoutError(f"world acknowledgement timed out: {op}")
            time.sleep(0.005)
        result = read_json(response, deadline)
        if (
            result.get("id") != identity
            or result.get("session") != self.session
            or "error" in result
        ):
            raise RuntimeError(f"lane rejected operation: {result}")
        if (
            result.get("schema") != "engram.minecraft-response/v1"
            or result.get("op") != op
        ):
            raise RuntimeError("native response contract differs")
        if result["game_tick"] <= self.last_tick:
            raise RuntimeError("world did not advance between lane observations")
        self.last_tick = result["game_tick"]
        self.requests += 1
        if op == "step":
            self.native_moves += 1
            if result["native_move_distance"] < 0.99:
                raise AssertionError(
                    "an advertised legal action did not move the native actor one cell"
                )
        semantic = {
            key: result[key]
            for key in ["op", "witness", "position", "reward", "terminal"]
            if key in result
        }
        self.trace_digest.update(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        )
        return result

    def reset(self, cue, *, reversed_goal=False):
        return self.request(
            "reset", cue=cue, reversed_goal=reversed_goal, offset=self.offset
        )

    def sample(self):
        return self.request("observe")

    def step(self, action):
        result = self.request("step", action=action)
        return result, result["reward"], result["terminal"]


def evaluate(agent, environment, count, *, reversed_goal=False):
    before = agent.updates
    results = [
        episode(
            agent,
            environment,
            1 if i % 2 else -1,
            train=False,
            reversed_goal=reversed_goal,
        )
        for i in range(count)
    ]
    assert before == agent.updates, "evaluation modified learned response values"
    return {
        "episodes": count,
        "success_rate": sum(r["success"] for r in results) / count,
        "mean_steps": sum(r["steps"] for r in results) / count,
        "example_trace": results[-1]["trace"],
    }


def controller(args, environment, run):
    blank = EngramNPC(SCHEMA, seed=args.seed)
    report = {
        "schema": "engram.minecraft-npc-study/v1",
        "seed": args.seed,
        "host": "native Minecraft 1.21.1 villager movement, collisions and block observations",
        "baseline": evaluate(blank, environment, args.evaluation),
        "initial_checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest()
        if args.checkpoint
        else None,
    }
    atomic_json(run / "partial.json", report)
    print(
        f"Minecraft empty-memory success: {report['baseline']['success_rate']:.1%}",
        flush=True,
    )
    agent = (
        EngramNPC.load(args.checkpoint, schema=SCHEMA)
        if args.checkpoint
        else EngramNPC(
            SCHEMA, seed=args.seed, novelty_radius=0.08, epsilon=0.1, exploration=0.25
        )
    )
    for i in range(args.training):
        episode(agent, environment, 1 if i % 2 else -1, planning=args.planning)
        if (i + 1) % 20 == 0:
            print(
                f"Minecraft training: {i + 1}/{args.training} episodes, {len(agent.wells)} wells",
                flush=True,
            )
    report["training_episodes"] = args.training
    report["learned"] = evaluate(agent, environment, args.evaluation)
    atomic_json(run / "partial.json", report)
    print(
        f"Minecraft learned success: {report['learned']['success_rate']:.1%}",
        flush=True,
    )
    checkpoint = run / "npc.engram"
    agent.save(checkpoint)
    restored = EngramNPC.load(checkpoint, schema=SCHEMA)
    environment.offset = 24
    report["restored_translated_world"] = evaluate(
        restored, environment, args.evaluation
    )
    atomic_json(run / "partial.json", report)
    print(
        f"Minecraft checkpoint/translated success: {report['restored_translated_world']['success_rate']:.1%}",
        flush=True,
    )
    environment.offset = 0
    if args.ablation:
        ablated = EngramNPC(
            SCHEMA,
            seed=args.seed,
            novelty_radius=0.08,
            epsilon=0.1,
            exploration=0.25,
            dynamics_weight=0,
        )
        for i in range(args.ablation):
            episode(ablated, environment, 1 if i % 2 else -1, planning=args.planning)
            if (i + 1) % 40 == 0:
                print(
                    f"Minecraft history ablation: {i + 1}/{args.ablation}", flush=True
                )
        report["without_fitted_history"] = evaluate(
            ablated, environment, args.evaluation
        )
        atomic_json(run / "partial.json", report)
        print(
            f"Minecraft history-disabled success: {report['without_fitted_history']['success_rate']:.1%}",
            flush=True,
        )
    if args.adaptation:
        report["before_environment_change"] = evaluate(
            restored, environment, args.evaluation, reversed_goal=True
        )
        for i in range(args.adaptation):
            episode(
                restored,
                environment,
                1 if i % 2 else -1,
                reversed_goal=True,
                planning=args.planning,
            )
        report["after_environment_change"] = evaluate(
            restored, environment, args.evaluation, reversed_goal=True
        )
    report.update(
        wells=len(agent.wells),
        real_feedback_updates=agent.updates,
        planning_updates=agent.planning_updates,
        native_requests=environment.requests,
        native_moves=environment.native_moves,
        semantic_trace_sha256=environment.trace_digest.hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    # Save failures too; a running JVM alone is never a successful study.
    atomic_json(run / "study.json", report)
    assert report["learned"]["success_rate"] >= 0.9, report["learned"]
    assert report["restored_translated_world"]["success_rate"] >= 0.9
    assert report["learned"]["success_rate"] > report["baseline"]["success_rate"] + 0.2
    if args.ablation:
        assert report["without_fitted_history"]["success_rate"] <= 0.6
    if args.adaptation:
        assert report["after_environment_change"]["success_rate"] >= 0.9
    return report


def run(args):
    empty_structure()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / "tmp"
    temporary.mkdir(exist_ok=True)
    owner = uuid.uuid4().hex
    folder = work / "runs" / owner
    folder.mkdir(parents=True)
    lane = folder / "lane"
    lane.mkdir()
    inputs = source_hashes()
    atomic_json(folder / "source.json", inputs)
    java = args.java or Path(shutil.which("java") or "")
    if not java.is_file():
        raise RuntimeError("a build JDK is required")
    env = dict(os.environ)
    env["JAVA_TOOL_OPTIONS"] = (
        env.get("JAVA_TOOL_OPTIONS", "")
        + " -Djdk.net.unixdomain.tmpdir="
        + short(temporary)
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
        "compileJava" if args.action == "build" else "runGametest",
    ]
    lock = args.heavy_lock or work / "heavy.lock"
    print("NPC lane run: " + str(folder), flush=True)
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
            if args.action == "run":
                deadline = time.monotonic() + 300
                while not (lane / "ready.json").is_file():
                    if process.poll() is not None:
                        raise RuntimeError("Minecraft exited before becoming ready")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Minecraft readiness timeout")
                    time.sleep(0.05)
                ready = read_json(lane / "ready.json")
                assert ready["session"] == owner
                environment = MinecraftEnvironment(lane, owner, process)
                report = controller(args, environment, folder)
                environment.request("finish")
            code = process.wait(timeout=1800)
            if code:
                raise RuntimeError(f"Gradle exited {code}")
            if args.action == "run":
                junit = ET.parse(folder / "junit.xml").getroot()
                cases = list(junit.iter("testcase"))
                assert (
                    cases
                    and not list(junit.iter("failure"))
                    and not list(junit.iter("error"))
                ), "actual nonempty JUnit verdict required"
                report["junit_cases"] = len(cases)
                report["world_seed"] = ready["seed"]
                report["sources"] = inputs
                report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                assert source_hashes() == inputs, (
                    "runtime sources changed during the study"
                )
                atomic_json(folder / "study.json", report)
                print(
                    json.dumps(
                        {k: v for k, v in report.items() if k not in ["sources"]},
                        indent=2,
                    ),
                    flush=True,
                )
            atomic_json(
                work / "latest.json", {"run": str(folder), "action": args.action}
            )
            print("Completed: " + str(folder), flush=True)
        except BaseException:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
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
            lines = (
                (folder / "gradle.log")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
            print(
                "\n".join([line for line in lines if not line.endswith(" [_]")][-80:]),
                flush=True,
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "run"])
    parser.add_argument("--work", type=Path, default=ROOT / ".smoke/minecraft")
    parser.add_argument(
        "--heavy-lock", type=Path, default=os.environ.get("MINECRAFT_HEAVY_LOCK")
    )
    parser.add_argument("--java", type=Path)
    parser.add_argument("--training", type=int, default=120)
    parser.add_argument("--evaluation", type=int, default=40)
    parser.add_argument("--ablation", type=int, default=120)
    parser.add_argument("--adaptation", type=int, default=40)
    parser.add_argument("--planning", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    if (
        min(
            arguments.training,
            arguments.ablation,
            arguments.adaptation,
            arguments.planning,
        )
        < 0
        or arguments.evaluation < 1
    ):
        parser.error(
            "nonnegative training/planning counts and positive evaluation count required"
        )
    if (
        arguments.training == 0
        and arguments.checkpoint is None
        and arguments.action == "run"
    ):
        parser.error("supply a tailored --checkpoint when skipping training")
    run(arguments)
