"""Plot the measured recency task and the audited interpolation counterexample."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from engram_stability import blend_schur

ROOT = Path(__file__).resolve().parents[1]
receipt = json.loads(
    (ROOT / "research/history-frontiers.json").read_text(encoding="utf-8")
)
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, (history, policy, roots) = plt.subplots(
    1, 3, figsize=(13.6, 4.6), layout="constrained"
)
for shift, values, color, label in [
    (0, [0, 0, 0, 1, 0, 0], "#427f73", "A: recent"),
    (-0.12, [0, 0, 1, 0, 0, 0], "#b87857", "B: old"),
]:
    history.plot(
        np.arange(1, 7), np.array(values) + shift, "o-", color=color, label=label
    )
history.axvspan(3.5, 6.5, color="#92b5ad", alpha=0.15)
history.set_xticks(np.arange(1, 7))
history.set_xlabel("Observation")
history.set_ylabel("Pulse (B shifted down for display)")
history.set_title("Same cumulative fit; different goal", loc="left")
history.legend(frameon=False, fontsize=9)
selected = [r for r in receipt["temporal_policy"] if r["seed"] == 7]
policy.bar(
    ["Cumulative", "Two scales"],
    [100 * r["success_rate"] for r in selected],
    color=["#999b9d", "#427f73"],
)
policy.set_ylim(0, 110)
policy.set_ylabel("Frozen recency-task success (%)")
policy.set_title("240 encounters of actual feedback", loc="left")
for i, row in enumerate(selected):
    policy.text(
        i,
        100 * row["success_rate"] + 2,
        f"{100 * row['success_rate']:.0f}%",
        ha="center",
    )
r = 0.995 + 0.01j
k = np.array([[2 * r], [2 * r.conjugate()]])
g = np.array([[r * r], [r.conjugate() ** 2]])
for color, method, label in [
    ("#b87857", "ordinary", "K/G mean"),
    ("#427f73", "schur", "Reflection mean"),
]:
    if method == "ordinary":
        a, b = k.mean(), g.mean()
    else:
        a, b = (v[0] for v in blend_schur(k, g, [1, 1]))
    pair = np.roots([1, -a, b])
    roots.scatter(pair.real, pair.imag, color=color, s=65, label=label, zorder=3)
pair = np.array([r, r.conjugate()])
roots.scatter(
    pair.real,
    pair.imag,
    marker="x",
    color="#55585b",
    label="Original repeated roots",
    s=45,
)
y = np.linspace(-0.018, 0.018, 300)
roots.plot(np.sqrt(1 - y * y), y, "--", color="#888888", label="Unit-circle boundary")
roots.set_xlim(0.98, 1.01)
roots.set_ylim(-0.018, 0.018)
roots.set_aspect("equal", adjustable="box")
roots.set_xlabel("Root real part")
roots.set_ylabel("Root imaginary part")
roots.set_title("Change generation coordinates", loc="left")
roots.legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("What a compact memory retains — and how its motions compose", fontsize=15)
fig.supxlabel(
    "Measured controlled task and fixed-oscillator interpolation. Neither establishes arbitrary-goal or switching stability.",
    fontsize=9,
)
target = ROOT / "docs/media/history-composition.png"
fig.savefig(target, dpi=150)
plt.close(fig)
print(target)
