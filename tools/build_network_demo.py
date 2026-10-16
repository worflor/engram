"""Generate the README figure and a reproducible, measured well-network receipt."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from well_network import experiment

SOURCES = [
    "examples/well_network.py",
    "engram_network.py",
    "engram_stability.py",
    "engram_analysis.py",
    "engram_training.py",
    "engram_metrics.py",
    "engram_codec.py",
    "tools/build_network_demo.py",
]


def fingerprint():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
    }


def build():
    _, _, report = experiment()
    receipt = {
        "schema": "engram-well-demo-1",
        "sources": fingerprint(),
        "scope": "Synthetic two-mode oscillators; 120 unlabeled training observations and 60 unseen phase/amplitude observations. No downloaded data.",
        "report": report,
    }
    (ROOT / "research/network-demo.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    probe = report["probe"]
    context, actual, dream = [
        np.asarray(probe[name])[:, 0] for name in ["context", "actual", "dream"]
    ]
    low, high = (
        min(context.min(), actual.min(), dream.min()),
        max(context.max(), actual.max(), dream.max()),
    )

    def polyline(values, start, color):
        points = " ".join(
            f"{380 + (start + i) / 95 * 600:.2f},{315 - (v - low) / (high - low) * 190:.2f}"
            for i, v in enumerate(values)
        )
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'

    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 420" role="img" aria-label="Measured Engram well network and held-out motion continuation">
<rect width="1040" height="420" rx="18" fill="#101817"/>
<style>text{font-family:system-ui,sans-serif;fill:#edf0df}.muted{fill:#acb9a9;font-size:13px}.title{font-family:Georgia,serif;font-size:30px}</style>
<text x="36" y="48" class="title">Learn motion. Remember routes. Dream forward.</text>
<text x="36" y="80" class="muted">120 observations · no labels during training · 60 held-out sequences</text>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#8fcddd"/></marker></defs>
<g stroke="#8fcddd" stroke-width="3" marker-end="url(#arrow)"><path d="M120 160L225 245"/><path d="M215 275L85 275"/><path d="M70 250L95 170"/></g>
<g fill="#243b2b" stroke="#c2ee87" stroke-width="2"><circle cx="105" cy="145" r="30"/><circle cx="245" cy="270" r="30"/><circle cx="55" cy="280" r="30"/></g>
<g text-anchor="middle" font-size="12"><text x="105" y="149">slow</text><text x="245" y="274">cross</text><text x="55" y="284">rapid</text></g>
<text x="380" y="110" class="muted">Coordinate 0 / observed → actual future + oscillator continuation</text>
<path d="M380 325H985" stroke="#34453c"/><path d="M835 120V330" stroke="#acb9a9" stroke-dasharray="4 5"/>
"""
    svg += (
        polyline(context, 0, "#8fcddd")
        + polyline(actual, 72, "#c2ee87")
        + polyline(dream, 72, "#f3ba88")
    )
    svg += (
        f'<text x="36" y="365" font-size="19">Held-out recognition: {report["accuracy"]:.0%} · shuffled time: {report["time_shuffled_accuracy"]:.0%}</text>'
        f'<text x="36" y="394" class="muted">Mean 24-step RMS: wells {report["mean_forecast_rmse"]:.3f} · local fit {report["mean_local_fit_forecast_rmse"]:.3f} · persistence {report["mean_persistence_rmse"]:.3f}. Synthetic task only.</text></svg>'
    )
    (ROOT / "docs/media/well-network.svg").write_text(svg, encoding="utf-8")
    print("Measured well-network figure and receipt rebuilt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        receipt = json.loads(
            (ROOT / "research/network-demo.json").read_text(encoding="utf-8")
        )
        if receipt["sources"] != fingerprint():
            raise SystemExit("Well-network receipt is stale; rebuild it")
        assert receipt["report"]["held_out_observations"] == 60
        print("Well-network receipt source identities checked")
    else:
        build()
