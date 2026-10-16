"""Render the actual affine-motion example and retained forecasting controls."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from affine_motion import demonstration

report, motion, offset, homogeneous, affine = demonstration()
receipt = json.loads(
    (ROOT / "research/affine-forecast.json").read_text(encoding="utf-8")
)
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, (path, controls) = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
path.plot(np.arange(96), motion[:96, 0], color="#8a8c90", label="Observed")
path.plot(
    np.arange(95, 128),
    motion[95:, 0],
    color="#333333",
    linewidth=2,
    label="Actual future",
)
path.plot(
    np.arange(96, 128),
    homogeneous[:, 0] - offset[0],
    "--",
    color="#b47459",
    label="Homogeneous fit",
)
path.plot(
    np.arange(96, 128), affine[:, 0] - offset[0], color="#437b70", label="Affine fit"
)
path.axvline(95, color="#aaaaaa", linewidth=1)
path.set_xlabel("Sample")
path.set_ylabel("X relative to the known display offset")
path.set_title("A translated orbit, fitted from its prefix", loc="left")
path.legend(frameon=False, fontsize=9)
positions = np.arange(6)
families = list(receipt["synthetic"])
for shift, method, color, label in [
    (-0.24, "homogeneous", "#b47459", "Homogeneous"),
    (0, "window_centered", "#8a8c90", "Window centered"),
    (0.24, "affine", "#437b70", "Affine"),
]:
    values = [
        receipt["synthetic"][family][method]["translated"]["median_rmse_of_finite"]
        for family in families
    ]
    controls.bar(positions + shift, values, width=0.24, color=color, label=label)
controls.set_xticks(
    positions,
    ["Orbit", "Damped", "Ramp", "Noise", "Random walk", "New frequency"],
    rotation=30,
    ha="right",
)
controls.set_ylabel("Median future RMSE (translated cases)")
controls.set_title("40 cases per family; 96 observed, 32 future", loc="left")
controls.legend(frameon=False, fontsize=9)
fig.suptitle("Separating motion from its origin", fontsize=16)
fig.supxlabel(
    "An explicit affine model. Forecasting accuracy remains dependent on the signal and horizon.",
    fontsize=9,
)
target = ROOT / "docs/media/affine-motion.png"
fig.savefig(target, dpi=150)
plt.close(fig)
print(target)
print(json.dumps(report))
