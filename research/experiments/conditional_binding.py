"""Adversarial three-way binding with actual streaming ContextRelations state.

Labels determine parity, not supplied context keys. Decoders are declared
algebraic probes, not newly trained NPC policies. Fresh sensor noise, role
imbalance, per-context affine nuisance and incorrect bindings test limits.
"""

from collections import Counter
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engram_relations import ContextRelations

ROLES = {-1: "stranger", 1: "owner"}
SCENARIOS = {
    "clean": {},
    "amplitude_range": {"scales": True},
    "context_offsets": {"offset": 3.0, "scales": True},
    "offsets_and_imbalance": {"offset": 3.0, "fraction_positive": 0.8, "scales": True},
    "rare_role": {"fraction_positive": 0.95},
    "sensor_noise_025": {"noise": 0.25},
    "sensor_noise_1": {"noise": 1.0},
    "sensor_noise_2": {"noise": 2.0},
    "binding_flips_010": {"flip": 0.1},
    "binding_flips_050": {"flip": 0.5},
    "binding_flips_100": {"flip": 1.0},
    "unseen_role_keys": {"unseen": True},
}


def exact_parity():
    """Every one/two-variable marginal agrees, while triple parity differs."""
    sources = {
        label: [(r, y, label * r * y) for r, y in itertools.product([-1, 1], repeat=2)]
        for label in [-1, 1]
    }
    marginals = []
    for width in [1, 2]:
        for indexes in itertools.combinations(range(3), width):
            hist = {
                label: Counter(tuple(row[i] for i in indexes) for row in rows)
                for label, rows in sources.items()
            }
            assert hist[-1] == hist[1]
            marginals.append(
                {
                    "coordinates": list(indexes),
                    "equal": True,
                    "probabilities": {
                        str(key): str(F(count, 4))
                        for key, count in sorted(hist[1].items())
                    },
                }
            )
    correlations = {}
    for label, rows in sources.items():
        correlations[str(label)] = {
            str(role): str(sum(F(y * z, 2) for r, y, z in rows if r == role))
            for role in [-1, 1]
        }
        assert all(r * y * z == label for r, y, z in rows)
    return {
        "conditional_sources": {str(k): v for k, v in sources.items()},
        "marginals": marginals,
        "conditional_yz": correlations,
        "target_information_bits": {
            "any_one_coordinate": 0,
            "any_two_coordinates": 0,
            "all_three_coordinates": 1,
        },
        "scope": "Uniform binary parity law. These exact information statements are not a chosen partial-information-decomposition estimator or a claim of novel synergy mathematics.",
    }


def encounter(
    label,
    rng,
    *,
    samples=128,
    fraction_positive=0.5,
    noise=0.0,
    flip=0.0,
    offset=0.0,
    scales=False,
    unseen=False,
):
    if samples < 4 or samples % 2:
        raise ValueError("positive even sample budget >=4 required")
    plus = max(2, min(samples - 2, int(round(samples * fraction_positive / 2)) * 2))
    role = np.r_[np.ones(plus), -np.ones(samples - plus)]
    y = np.concatenate(
        [rng.permutation(np.tile([-1.0, 1.0], n // 2)) for n in [plus, samples - plus]]
    )
    z = label * role * y
    rows = np.c_[y, z] + offset * role[:, None]
    rows += rng.normal(0, noise, rows.shape)
    if scales:
        # Strictly positive, independently sampled units in each role/channel.
        factors = np.exp(rng.uniform(np.log(1 / 8), np.log(8), (2, 2)))
        rows *= factors[(role > 0).astype(int)]
    observed_role = role.copy()
    observed_role[rng.random(samples) < flip] *= -1
    contexts = [
        ROLES[int(r)] if not unseen else "unseen-" + ROLES[int(r)]
        for r in observed_role
    ]
    indexes = rng.permutation(samples)
    return [contexts[i] for i in indexes], rows[indexes]


def summarize(contexts, rows):
    sketch = ContextRelations(
        ("cue", "consequence"), representation="conditional-binding-v1"
    )
    for context, row in zip(contexts, rows):
        sketch.push(context, row)
    return sketch


def conditional_score(sketch, *, normalized=True):
    left, right = sketch.moments("stranger"), sketch.moments("owner")
    if left is None or right is None:
        return None
    if normalized and (np.any(left.variances <= 0) or np.any(right.variances <= 0)):
        return None
    matrix = "correlation" if normalized else "covariance"
    return float((getattr(right, matrix)[0, 1] - getattr(left, matrix)[0, 1]) / 2)


def scores(contexts, rows, sketch):
    global_summary = ContextRelations(
        ("cue", "consequence"), representation="unconditioned-control-v1"
    )
    for row in rows:
        global_summary.push("all", row)
    r = np.array(
        [1 if c == "owner" else -1 if c == "stranger" else 0 for c in contexts]
    )
    centered = rows - rows.mean(axis=0)
    return {
        "pooled_correlation": float(global_summary.moments("all").correlation[0, 1]),
        "raw_triple_product": float(np.mean(r * rows[:, 0] * rows[:, 1])),
        "globally_centered_triple": float(np.mean(r * centered[:, 0] * centered[:, 1])),
        "conditional_covariance": conditional_score(sketch, normalized=False),
        "conditional_correlation": conditional_score(sketch),
    }


def level_loss_control():
    states = []
    for label in [-1, 1]:
        sketch = ContextRelations(
            ("cue", "consequence"), representation="level-control-v1"
        )
        for role, y in itertools.product([-1, 1], repeat=2):
            sketch.push(ROLES[role], [y, label * role])
        states.append(sketch)
    # Correlation projection genuinely aliases these two memories, but the full
    # moment object retains opposite means. This is stronger than failed training.
    for context in ROLES.values():
        np.testing.assert_array_equal(
            states[0].moments(context).correlation,
            states[1].moments(context).correlation,
        )
    means = [float(s.moments("owner").mean[1]) for s in states]
    assert means == [-1.0, 1.0]
    return {
        "correlation_views_equal": True,
        "owner_consequence_means": means,
        "interpretation": "Normalized projection loses constant levels; the full summary still contains the distinction.",
    }


def four_way_obstruction():
    """An exact limitation of FULL second moments inside a fixed role fiber."""
    states = {}
    refined_scores = {}
    for label in [-1, 1]:
        sketch = ContextRelations(("a", "b", "outcome"), representation="four-way-v1")
        refined = ContextRelations(
            ("b", "outcome"), representation="four-way-refined-v1"
        )
        exact = {}
        for role in [-1, 1]:
            rows = [
                (a, b, label * role * a * b)
                for a, b in itertools.product([-1, 1], repeat=2)
            ]
            # Exact per-role first and full second moments, independently of
            # floating accumulation or any readout. Both labels yield zero
            # means and identity population covariance.
            sums = tuple(sum(row[i] for row in rows) for i in range(3))
            cross = tuple(
                tuple(sum(row[i] * row[j] for row in rows) for j in range(3))
                for i in range(3)
            )
            exact[role] = (sums, cross)
            for a, b, z in rows:
                sketch.push(ROLES[role], [a, b, z])
                refined.push(f"{role}:{a}", [b, z])
        states[label] = exact
        score = (
            sum(
                role * a * refined.moments(f"{role}:{a}").correlation[0, 1]
                for role, a in itertools.product([-1, 1], repeat=2)
            )
            / 4
        )
        refined_scores[str(label)] = float(score)
        assert np.isclose(score, label)
    assert states[-1] == states[1]
    return {
        "fixed_role_full_first_second_moments_equal": True,
        "per_role_sums": list(states[1][-1][0]),
        "per_role_cross_products": [list(row) for row in states[1][-1][1]],
        "refined_context_scores": refined_scores,
        "scope": "Four-way parity is genuinely lost by fixed-role full second moments. Conditioning additionally on a retains the b/outcome relation. No generic automatic refinement rule is established.",
    }


def campaign(*, seeds=tuple(range(1001, 1009)), episodes=64):
    if episodes < 2 or episodes % 2:
        raise ValueError("even episode budget required")
    results = []
    max_merge_error = 0.0
    max_json_error = 0.0
    numeric_bytes = set()
    for seed in seeds:
        for scenario, options in SCENARIOS.items():
            rng = np.random.default_rng(seed)
            labels = np.tile([-1, 1], episodes // 2)
            rng.shuffle(labels)
            counts = {}
            for index, label in enumerate(labels):
                contexts, rows = encounter(int(label), rng, **options)
                sketch = summarize(contexts, rows)
                values = scores(contexts, rows, sketch)
                for method, score in values.items():
                    item = counts.setdefault(method, {"correct": 0, "covered": 0})
                    if score is not None:
                        item["covered"] += 1
                        # Declared tie break and numerical zero tolerance.
                        prediction = 1 if abs(score) < 1e-12 or score >= 0 else -1
                        item["correct"] += int(prediction == label)
                numeric_bytes.add(
                    sum(
                        sketch.moments(c).mean.nbytes + sketch.moments(c).scatter.nbytes
                        for c in sketch.contexts
                    )
                )
                if index == 0:
                    chunks = [
                        summarize(contexts[i : i + 16], rows[i : i + 16])
                        for i in range(0, len(rows), 16)
                    ]
                    merged = chunks[0]
                    for chunk in chunks[1:]:
                        merged = merged.merge(chunk)
                    restored = ContextRelations.from_dict(
                        json.loads(json.dumps(sketch.to_dict(), allow_nan=False))
                    )
                    for context in sketch.contexts:
                        original = sketch.moments(context)
                        other = merged.moments(context)
                        max_merge_error = max(
                            max_merge_error,
                            float(np.max(abs(original.covariance - other.covariance))),
                        )
                        max_json_error = max(
                            max_json_error,
                            float(
                                np.max(
                                    abs(
                                        original.scatter
                                        - restored.moments(context).scatter
                                    )
                                )
                            ),
                        )
            for method, item in counts.items():
                results.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "method": method,
                        "episodes": episodes,
                        "correct": item["correct"],
                        "covered": item["covered"],
                        "accuracy": item["correct"] / episodes,
                        "coverage": item["covered"] / episodes,
                    }
                )
            print(f"seed={seed} scenario={scenario}", flush=True)
    aggregate = []
    for scenario in SCENARIOS:
        for method in values:
            selected = [
                r
                for r in results
                if r["scenario"] == scenario and r["method"] == method
            ]
            n = sum(r["episodes"] for r in selected)
            aggregate.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "episodes": n,
                    "accuracy": sum(r["correct"] for r in selected) / n,
                    "coverage": sum(r["covered"] for r in selected) / n,
                    "minimum_seed_accuracy": min(r["accuracy"] for r in selected),
                }
            )
    sources = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in ["engram_relations.py", "research/experiments/conditional_binding.py"]
    }
    return {
        "schema": "engram.conditional-binding/v1",
        "seeds": list(seeds),
        "episodes_per_seed_scenario": episodes,
        "scenarios": SCENARIOS,
        "stream_samples": 128,
        "streams": len(seeds) * len(SCENARIOS) * episodes,
        "diagnostic_decisions": sum(r["episodes"] for r in results),
        "exact_parity": exact_parity(),
        "level_loss": level_loss_control(),
        "four_way_obstruction": four_way_obstruction(),
        "numeric_mean_scatter_bytes_per_stream": sorted(numeric_bytes),
        "storage_scope": "Only mean/scatter numeric payload, excluding keys, Python objects, integer counters and snapshots; there are two observed contexts in this campaign.",
        "maximum_chunk_merge_covariance_error": max_merge_error,
        "maximum_json_scatter_error": max_json_error,
        "scope": "Algebraic readouts of known authored relations, not trained policies, PID estimates, real sensor data, or calibrated uncertainty. Noisy identity can corrupt a relational summary.",
        "aggregate": aggregate,
        "runs": results,
        "sources": sources,
    }


if __name__ == "__main__":
    report = campaign()
    (ROOT / "research/conditional-binding.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "streams",
                    "diagnostic_decisions",
                    "maximum_chunk_merge_covariance_error",
                ]
            }
        )
    )
