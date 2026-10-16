"""Render the measured real-signal comparison. Optional dependency: matplotlib."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
receipt = json.loads(
    (ROOT / "research/npc/real-signals.json").read_text(encoding="utf-8")
)
results = receipt["results"]
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, (ax, classes) = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
keys = [
    "current",
    "moments",
    "oscillators",
    "npc_context",
    "moments_and_oscillators",
    "shuffled_oscillators",
]
labels = [
    "Current sample",
    "Window mean / deviation",
    "Oscillators",
    "NPC context features",
    "Moments + oscillators",
    "Time-scrambled oscillators",
]
values = [100 * results[k]["test"]["accuracy"] for k in keys]
bars = ax.barh(
    labels,
    values,
    color=["#888b91", "#888b91", "#437b70", "#437b70", "#437b70", "#b2745a"],
)
ax.invert_yaxis()
ax.set_xlim(0, 103)
ax.set_xlabel("Held-out window accuracy (%)")
ax.set_title("A shared geometric readout", loc="left")
for bar, value in zip(bars, values):
    ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center")
positions = np.arange(6)
for offset, key, color, label in [
    (-0.18, "moments", "#888b91", "Window moments"),
    (0.18, "oscillators", "#437b70", "Oscillators"),
]:
    confusion = np.array(results[key]["test"]["confusion"])
    recall = 100 * confusion.diagonal() / confusion.sum(axis=1)
    classes.bar(positions + offset, recall, width=0.36, color=color, label=label)
classes.set_xticks(
    positions,
    ["Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lay"],
    rotation=30,
    ha="right",
)
classes.set_ylim(0, 105)
classes.set_ylabel("Class recall (%)")
classes.set_title("Which activities remain distinguishable?", loc="left")
classes.legend(frameon=False)
fig.suptitle(
    "Engram on real human motion\n2,947 test windows from 9 held-out people; radius chosen on training subjects",
    fontsize=14,
)
fig.supxlabel(
    "Supervised readout of signal representations. This experiment does not evaluate the NPC action policy.",
    fontsize=9,
)
target = ROOT / "docs/media/npc-real-signals.png"
fig.savefig(target, dpi=150)
plt.close(fig)
print(target)
