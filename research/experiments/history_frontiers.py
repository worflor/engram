"""Independent exact/float checks of history composition, observation and Schur views.

The finite observer source is a SIX-SAMPLE SLIDING-WINDOW system, not an
unbounded NPC brain. Its exact observation equivalence is supplied as a table;
Scryer replay certifies that declared finite source, not the table-generating
Python or all floating-point contexts. No downloads or external calls here.
"""

from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from engram_history import WitnessHistory
from engram_stability import blend_schur
from temporal_witness import study


def raw_summary(values, decay=F(1)):
    n = len(values)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    triples = [(values[i], values[i - 1], values[i - 2]) for i in range(2, n)]
    w = weights[2:]
    return {
        "n": n,
        "weight": sum(weights, F(0)),
        "sum": sum((a * b for a, b in zip(values, weights)), F(0)),
        "jw": sum(w, F(0)),
        "js": tuple(
            sum((q * t[i] for q, t in zip(w, triples)), F(0)) for i in range(3)
        ),
        "cross": tuple(
            tuple(
                sum((q * t[i] * t[j] for q, t in zip(w, triples)), F(0))
                for j in range(3)
            )
            for i in range(3)
        ),
        "first": tuple(values[:2]),
        "last": tuple(values[-2:]),
    }


def join(a, b, decay):
    out = {
        "n": a["n"] + b["n"],
        "first": (a["first"] + b["first"])[:2],
        "last": (a["last"] + b["last"])[-2:],
    }
    age = decay ** b["n"]
    out["weight"] = age * a["weight"] + b["weight"]
    out["sum"] = age * a["sum"] + b["sum"]
    out["jw"] = age * a["jw"] + b["jw"]
    sums = [age * a["js"][i] + b["js"][i] for i in range(3)]
    cross = [
        [age * a["cross"][i][j] + b["cross"][i][j] for j in range(3)] for i in range(3)
    ]
    boundary = []
    if a["n"] >= 2 and b["n"]:
        boundary.append((0, (b["first"][0], a["last"][-1], a["last"][-2])))
    if a["n"] and b["n"] >= 2:
        boundary.append((1, (b["first"][1], b["first"][0], a["last"][-1])))
    for index, triple in boundary:
        w = decay ** (b["n"] - 1 - index)
        out["jw"] += w
        for i in range(3):
            sums[i] += w * triple[i]
            for j in range(3):
                cross[i][j] += w * triple[i] * triple[j]
    out["js"] = tuple(sums)
    out["cross"] = tuple(tuple(row) for row in cross)
    return out


def exact_signature(values, decay):
    raw = raw_summary(values, decay)
    mean = raw["sum"] / raw["weight"]
    c = [
        [
            raw["cross"][i][j]
            - mean * (raw["js"][i] + raw["js"][j])
            + raw["jw"] * mean * mean
            for j in range(3)
        ]
        for i in range(3)
    ]
    aa, bb, ab, ta, tb = c[1][1], c[2][2], c[1][2], c[0][1], c[0][2]
    eps = F(1, 8388608)
    trace = aa + bb
    ridge = max(eps, trace / F(20000))
    ar, br = aa + ridge, bb + ridge
    det = ar * br - ab * ab
    k, g = (
        (F(2), F(1))
        if det == 0 or abs(det) < eps * trace * trace
        else ((ta * br - ab * tb) / det, (ab * ta - ar * tb) / det)
    )
    coeff = [F(1), -k, g]
    mse = (
        sum((coeff[i] * c[i][j] * coeff[j] for i in range(3) for j in range(3)), F(0))
        / raw["jw"]
    )
    return k, g, mse


def exact_composition():
    counts = {}
    for decay in [F(1), F(9, 10), F(1, 2)]:
        checked = 0
        for n in range(7):
            for values in itertools.product((0, 1), repeat=n):
                direct = raw_summary(values, decay)
                for i in range(n + 1):
                    for j in range(i, n + 1):
                        a, b, c = (
                            raw_summary(part, decay)
                            for part in [values[:i], values[i:j], values[j:]]
                        )
                        assert join(join(a, b, decay), c, decay) == direct
                        assert join(a, join(b, c, decay), decay) == direct
                        checked += 1
        counts[str(decay)] = checked
    return counts


def floating_composition():
    rng = np.random.default_rng(314)
    maximum = 0.0
    cases = 0
    for decay in [1.0, 0.995, 0.9, 0.5]:
        for n in [3, 10, 100, 1000]:
            x = rng.normal(size=(n, 4))
            direct = WitnessHistory(4, decay)
            for row in x:
                direct.push(row)
            for cuts in [(0, 1), (1, 2), (n // 3, 2 * n // 3)]:
                pieces = []
                for part in np.split(x, cuts):
                    h = WitnessHistory(4, decay)
                    for row in part:
                        h.push(row)
                    pieces.append(h)
                for merged in [
                    pieces[0].concat(pieces[1]).concat(pieces[2]),
                    pieces[0].concat(pieces[1].concat(pieces[2])),
                ]:
                    for name in ["k", "g", "pair_rms"]:
                        maximum = max(
                            maximum,
                            float(
                                np.max(
                                    abs(
                                        getattr(direct.signature(), name)
                                        - getattr(merged.signature(), name)
                                    )
                                )
                            ),
                        )
                    cases += 1
    assert maximum < 1e-6
    return {"cases": cases, "maximum_signature_error": maximum}


def observer_table(decays):
    states = list(itertools.product((0, 1), repeat=6))
    ids = {state: i for i, state in enumerate(states)}
    observations = [
        (state[-1],)
        + tuple(v for decay in decays for v in exact_signature(state, decay))
        for state in states
    ]
    classes = {}
    candidate = []
    for obs in observations:
        if obs not in classes:
            classes[obs] = f"observation-{len(classes)}"
        candidate.append(classes[obs])
    source = {
        "kind": "finite-table",
        "states": ["".join(map(str, s)) for s in states],
        "actions": [
            {
                "name": f"append-{bit}",
                "next": [ids[state[1:] + (bit,)] for state in states],
            }
            for bit in (0, 1)
        ],
        "goals": [["1" if any(state[-3:]) else "0"] for state in states],
    }
    return {"source": source, "candidate": candidate, "horizon": "all"}, {
        name: [str(v) for v in observation] for observation, name in classes.items()
    }


def schur_campaign(count=10000):
    rng = np.random.default_rng(928)
    unstable = 0
    rejected32 = 0
    maximum64 = maximum32 = 0.0
    for i in range(count):
        roots = rng.uniform(0.05, 0.999, (2, 2)) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, (2, 2))
        )
        if i % 2 == 0:
            roots[:, 1] = roots[:, 0]
        k, g = roots.sum(axis=1)[:, None], roots.prod(axis=1)[:, None]
        weights = rng.random(2)
        weights /= weights.sum()
        plaink, plaing = weights @ k, weights @ g
        unstable += int(max(abs(np.roots([1, -plaink[0], plaing[0]]))) >= 1)
        for dtype in [np.complex128, np.complex64]:
            try:
                bk, bg = blend_schur(k, g, weights, dtype=dtype)
            except (ValueError, FloatingPointError):
                if dtype == np.complex128:
                    raise
                rejected32 += 1
                continue
            radius = float(max(abs(np.roots([1, -complex(bk[0]), complex(bg[0])]))))
            assert radius < 1
            if dtype == np.complex128:
                maximum64 = max(maximum64, radius)
            else:
                maximum32 = max(maximum32, radius)
    radius_sweep = {}
    for bound in [0.05, 0.5, 0.9, 1.0, 1.5]:
        roots = rng.uniform(0.1, 0.9, (2, 64, 2)) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, (2, 64, 2))
        )
        scaled_k, scaled_g = (
            roots.sum(axis=2) * bound,
            roots.prod(axis=2) * bound * bound,
        )
        maxima = {}
        for dtype in [np.complex128, np.complex64]:
            bk, bg = blend_schur(scaled_k, scaled_g, [1, 3], radius=bound, dtype=dtype)
            normalized = [
                float(max(abs(np.roots([1, -complex(a), complex(b)])))) / bound
                for a, b in zip(bk, bg)
            ]
            assert max(normalized) < 1
            maxima[np.dtype(dtype).name] = max(normalized)
        radius_sweep[str(bound)] = {"cases": 64, "maximum_root_over_bound": maxima}
    return {
        "cases": count,
        "seed": 928,
        "ordinary_unstable": unstable,
        "accepted64": count,
        "rejected32": rejected32,
        "maximum_accepted_radius64": maximum64,
        "maximum_accepted_radius32": maximum32,
        "distribution": "Independent uniform radii .05..999 and phases; half of the polynomials have repeated roots; independent convex weights",
        "radius_sweep": radius_sweep,
    }


def rank(matrix):
    a = [list(map(F, row)) for row in matrix]
    row = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(row, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        scale = a[row][col]
        a[row] = [x / scale for x in a[row]]
        for i in range(row + 1, len(a)):
            scale = a[i][col]
            a[i] = [x - scale * y for x, y in zip(a[i], a[row])]
        row += 1
        if row == len(a):
            break
    return row


def polynomial_fixture():
    values = [
        10 * F(i, 16) ** 3 - 15 * F(i, 16) ** 4 + 6 * F(i, 16) ** 5 for i in range(16)
    ]
    difference = values
    for _ in range(6):
        difference = [b - a for a, b in zip(difference, difference[1:])]
    assert all(v == 0 for v in difference)
    coefficients = [6, -15, 20, -15, 6, -1]
    assert all(
        values[i] == sum(coefficients[j] * values[i - 1 - j] for j in range(6))
        for i in range(6, len(values))
    )
    hankel = [[values[i + j] for j in range(6)] for i in range(6)]
    design = [[values[i - 1], -values[i - 2]] for i in range(2, len(values))]
    augmented = [row + [values[i + 2]] for i, row in enumerate(design)]
    assert rank(hankel) == 6 and rank(design) == 2 and rank(augmented) == 3
    return {
        "law": "10t^3-15t^4+6t^5 at t=n/16, n=0..15; one fixed polynomial piece, not an execution of Vixen",
        "values": [str(v) for v in values],
        "hankel_rank": 6,
        "ar2_design_rank": 2,
        "ar2_augmented_rank": 3,
        "sixth_differences": [str(v) for v in difference],
    }


def main():
    a, b = (0, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0)
    assert raw_summary(a) == raw_summary(b)
    assert exact_signature(a, F(1)) == exact_signature(b, F(1))
    report = {
        "schema": "engram.history-frontiers/v1",
        "provenance": "Independent reconstruction from a user-supplied research excerpt; its linked packet was not supplied and its reported counts are not reused",
        "exact_composition": exact_composition(),
        "float_composition": floating_composition(),
        "pulse_counterexample": {
            "histories": [a, b],
            "goal_recent_pulse": [True, False],
            "exact_signature": [str(v) for v in exact_signature(a, F(1))],
        },
        "temporal_policy": [
            study(decays, seed=seed)[1]
            for seed in [7, 19, 81]
            for decays in [(1.0,), (1.0, 0.9)]
        ],
        "schur": schur_campaign(),
        "quintic": polynomial_fixture(),
    }
    requests = {}
    for name, decays in [("cumulative", [F(1)]), ("two_scales", [F(1), F(9, 10)])]:
        request, binding = observer_table(decays)
        requests[name] = {
            "request": {
                "subject": "atlas "
                + json.dumps(
                    {"op": "observer", "request": request}, separators=(",", ":")
                )
            },
            "observation_values": binding,
            "decays": [str(d) for d in decays],
        }
    (ROOT / "research/exact/witness-observer-requests.json").write_text(
        json.dumps(requests, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    report["dwell_example"] = {
        "K": ["9/5", "-9/5"],
        "G": "81/100",
        "minimum_block": 43,
        "infinity_bound_at_43": str(F(9, 10) ** 42 * F(404, 5)),
        "infinity_bound_at_42": str(F(9, 10) ** 41 * F(789, 10)),
        "scope": "Exact repeated-root pair only; deterministic homogeneous blocks, no noise or reset. Not a runtime graph certificate.",
    }
    assert (
        F(report["dwell_example"]["infinity_bound_at_43"])
        < 1
        < F(report["dwell_example"]["infinity_bound_at_42"])
    )
    report["sources"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in [
            "engram_history.py",
            "engram_npc.py",
            "engram_stability.py",
            "engram_network.py",
            "examples/temporal_witness.py",
            "research/experiments/history_frontiers.py",
        ]
    }
    (ROOT / "research/history-frontiers.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in [
                    "exact_composition",
                    "float_composition",
                    "temporal_policy",
                    "schur",
                    "dwell_example",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
