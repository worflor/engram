"""Render the measured relational ablations, retaining their failures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads(
        (ROOT / "research/relational-memory.json").read_text(encoding="utf-8")
    )
    methods = [
        ("current", "Current observation"),
        ("ar2", "Cumulative AR(2)"),
        ("temporal_bank", "Fast + slow AR(2)"),
        ("moments", "Cross-channel moments"),
        ("correlation_relations", "Normalized correlations"),
        ("lag_relations", "Lagged cross moments"),
        ("oriented_area", "Moments + oriented area"),
        ("path_level3", "Selected path, level 3"),
        ("role_fibers", "Role-conditioned moments"),
        ("normalized_role_fibers", "Role-conditioned correlations"),
        ("last_event_control", "Last-event age control"),
    ]
    tasks = [
        "phase_coupling",
        "event_order",
        "role_binding",
        "conditional_parity",
        "recent_pulse",
        "current_control",
    ]
    labels = [
        "Phase\ncoupling",
        "Event\norder",
        "Role\nbinding",
        "Three-way\nbinding",
        "Recent\npulse",
        "Current\ncontrol",
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, condition, title in zip(
        axes,
        ["independent", "longer_and_rescaled"],
        ["Independent held-out streams", "Longer streams + wider amplitude range"],
    ):
        data = np.array(
            [
                [
                    next(
                        row["accuracy"]
                        for row in report["aggregate"]
                        if (row["representation"], row["task"], row["condition"])
                        == (method, task, condition)
                    )
                    for task in tasks
                ]
                for method, _ in methods
            ]
        )
        im = ax.imshow(data, vmin=0.4, vmax=1, cmap="YlGnBu", aspect="auto")
        for i in range(len(methods)):
            for j in range(len(tasks)):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if data[i, j] > 0.78 else "#202a35",
                )
        ax.set_xticks(range(len(tasks)), labels)
        ax.set_yticks(range(len(methods)), [label for _, label in methods])
        ax.set_title(title, pad=15, fontweight="bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Which relationships survive compression?",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "Three seeds · balanced labels · train-only scaling · shared linear readout · authored tasks, not general semantic understanding",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.205, right=0.91, top=0.88, bottom=0.12, wspace=0.08)
    bar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.025)
    bar.set_label("Held-out accuracy")
    fig.savefig(ROOT / "docs/media/relational-memory.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
