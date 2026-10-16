"""Seal reduced real-game receipts and a schematic of captured native behavior.

Usage: python tools/build_npc_evidence.py <full-study-run> <fresh-seeded-run>
Optional --plot requires matplotlib; no game world or working brain is copied.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from npc_lane import source_hashes
from engram_npc import WitnessHistory


def junction_witnesses(folder):
    rows = {}
    observations = []
    cue = None
    positions = []
    for line in (folder / "lane/native.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["op"] == "reset":
            cue = int(row["witness"][1])
            observations = []
            positions = []
        elif row["op"] in ("observe", "step"):
            observations.append(row["witness"])
            if row["op"] == "step":
                positions.append(row["position"])
            if positions == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]] and cue not in rows:
                history = WitnessHistory(8)
                for vector in observations:
                    signature = history.push(vector)

                def complex_rows(values):
                    return [[float(z.real), float(z.imag)] for z in values]

                rows[cue] = {
                    "earlier_cue": cue,
                    "native_observations": observations.copy(),
                    "junction_witness": row["witness"],
                    "K": complex_rows(signature.k),
                    "G": complex_rows(signature.g),
                }
    if (
        set(rows) != {-1, 1}
        or rows[-1]["junction_witness"] != rows[1]["junction_witness"]
    ):
        raise ValueError(
            "native stream did not witness the claimed current-observation alias"
        )
    return {
        "pairs": [rows[-1], rows[1]],
        "native_stream_sha256": hashlib.sha256(
            (folder / "lane/native.jsonl").read_bytes()
        ).hexdigest(),
    }


def read_run(folder):
    report = json.loads((folder / "study.json").read_text(encoding="utf-8"))
    junit = (folder / "junit.xml").read_bytes()
    root = ET.fromstring(junit)
    cases = list(root.iter("testcase"))
    if not cases or list(root.iter("failure")) or list(root.iter("error")):
        raise ValueError("a successful nonempty real JUnit report is required")
    if report.get("junit_cases") != len(cases) or report["sources"] != source_hashes():
        raise ValueError("run completion or runtime source identity differs")
    if (
        report["learned"]["success_rate"] < 0.9
        or report["restored_translated_world"]["success_rate"] < 0.9
    ):
        raise ValueError("behavioral acceptance did not pass")
    return (
        report,
        {"testcases": len(cases), "sha256": hashlib.sha256(junit).hexdigest()},
        junit,
    )


def render(report, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig = plt.figure(figsize=(12, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    for column, key, title in [
        (0, "before_environment_change", "Changed world: old experience"),
        (1, "after_environment_change", "After learning new consequences"),
    ]:
        ax = fig.add_subplot(grid[0, column])
        trace = report[key]["example_trace"]
        x = [0] + [row["position"][0] for row in trace]
        y = [0] + [row["position"][1] for row in trace]
        ax.plot(x, y, "o-", color="#487e63", markersize=5)
        ax.scatter(
            [x[-1]],
            [y[-1]],
            s=100,
            color="#448b5c" if trace[-1]["reward"] > 0 else "#bc624b",
            zorder=3,
        )
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.3, 3.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_xlabel("Native relative X")
        ax.set_ylabel("Native relative Z")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    ax = fig.add_subplot(grid[1, :])
    keys = [
        "baseline",
        "learned",
        "without_fitted_history",
        "restored_translated_world",
        "before_environment_change",
        "after_environment_change",
    ]
    labels = [
        "Empty memory",
        "Learned history",
        "History disabled",
        "Restored / moved arena",
        "Changed outcomes",
        "Relearned outcomes",
    ]
    values = [100 * report[key]["success_rate"] for key in keys]
    bars = ax.barh(
        labels,
        values,
        color=["#999999", "#487e63", "#999999", "#487e63", "#bc624b", "#487e63"],
    )
    ax.set_xlim(0, 113)
    ax.set_xlabel("Successful evaluation episodes (%)")
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(
            value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center"
        )
    fig.suptitle(
        "Engram NPC: learned experience in a real Minecraft world\nNative villager positions above; fixed evaluation, temporal ablation and adaptation below",
        fontsize=14,
    )
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main(args):
    first, junit_first, raw_first = read_run(args.first)
    second, junit_second, raw_second = read_run(args.second)
    if (
        first["without_fitted_history"]["success_rate"] > 0.6
        or first["after_environment_change"]["success_rate"] < 0.9
    ):
        raise ValueError("full study controls failed")
    if second["training_episodes"] != 0 or first["checkpoint_sha256"] != second.get(
        "initial_checkpoint_sha256"
    ):
        raise ValueError(
            "second world must consume the actual first learned checkpoint"
        )
    target = ROOT / "research/npc"
    target.mkdir(exist_ok=True)
    (target / "minecraft-first.xml").write_bytes(raw_first)
    (target / "minecraft-seeded.xml").write_bytes(raw_second)
    receipt = {
        "schema": "engram.npc-evidence/v1",
        "scope": "Finite controlled task in two fresh real JVM/world runs; no framebuffer, production world or general-game claim.",
        "first": first,
        "seeded_fresh_world": second,
        "junit": {"first": junit_first, "second": junit_second},
        "identical_current_witness_control": junction_witnesses(args.first),
    }
    serialized = re.sub(
        r"\[\s*([-\d][-\d.eE+,\s]*)\]",
        lambda m: "[" + " ".join(m.group(1).split()) + "]",
        json.dumps(receipt, indent=2),
    )
    (target / "minecraft.json").write_text(
        serialized + "\n", encoding="utf-8", newline="\n"
    )
    if args.plot:
        render(first, ROOT / "docs/media/npc-experience.png")
    print(
        "Sealed two successful native-game receipts; no world save or working memory copied"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--plot", action="store_true")
    main(parser.parse_args())
