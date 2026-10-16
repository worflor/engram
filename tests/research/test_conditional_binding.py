"""Independent parity and projection-loss controls for the streaming study."""

from collections import Counter
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from research.experiments import conditional_binding as cb

ROOT = Path(__file__).resolve().parents[2]


def test_all_pairwise_probability_laws_are_exactly_independent_of_parity():
    law = cb.exact_parity()
    sources = law["conditional_sources"]
    for width in [1, 2]:
        for indexes in itertools.combinations(range(3), width):
            distributions = []
            for label in [-1, 1]:
                counts = Counter(
                    tuple(row[i] for i in indexes) for row in sources[str(label)]
                )
                distributions.append(
                    {values: F(count, 4) for values, count in counts.items()}
                )
            assert distributions[0] == distributions[1]
            assert set(distributions[0].values()) == {F(1, 2**width)}
    for label, rows in sources.items():
        assert all(np.prod(row) == int(label) for row in rows)


def test_conditional_covariance_survives_affine_nuisance_and_role_imbalance():
    for label in [-1, 1]:
        contexts, rows = cb.encounter(
            label,
            np.random.default_rng(71),
            offset=3.0,
            scales=True,
            fraction_positive=0.8,
        )
        sketch = cb.summarize(contexts, rows)
        values = cb.scores(contexts, rows, sketch)
        assert np.sign(values["conditional_covariance"]) == label
        np.testing.assert_allclose(values["conditional_correlation"], label, atol=1e-12)
    # The declared nuisance can swamp global centering even without noise.
    assert values["globally_centered_triple"] < 0


def test_full_statistics_retain_constant_levels_lost_by_correlation_projection():
    result = cb.level_loss_control()
    assert result["correlation_views_equal"]
    assert result["owner_consequence_means"] == [-1.0, 1.0]


def test_even_full_role_conditioned_second_moments_lose_a_four_way_relation():
    result = cb.four_way_obstruction()
    assert result["fixed_role_full_first_second_moments_equal"]
    assert result["per_role_sums"] == [0, 0, 0]
    assert result["per_role_cross_products"] == [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    assert result["refined_context_scores"] == {"-1": -1.0, "1": 1.0}


def test_inverted_bindings_reverse_the_answer_and_unseen_roles_are_absent():
    for label in [-1, 1]:
        contexts, rows = cb.encounter(label, np.random.default_rng(31), flip=1.0)
        assert np.sign(cb.conditional_score(cb.summarize(contexts, rows))) == -label
        contexts, rows = cb.encounter(label, np.random.default_rng(31), unseen=True)
        assert cb.conditional_score(cb.summarize(contexts, rows)) is None


def test_sealed_streaming_study_matches_root_and_runner_sources():
    r = json.loads(
        (ROOT / "research/conditional-binding.json").read_text(encoding="utf-8")
    )
    for path, digest in r["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert r["maximum_json_scatter_error"] == 0
    assert r["maximum_chunk_merge_covariance_error"] < 1e-10
    assert r["numeric_mean_scatter_bytes_per_stream"] == [96]


def test_additional_scryer_components_are_confirmed_with_proper_scope():
    r = json.loads(
        (ROOT / "research/exact/relation-moments.json").read_text(encoding="utf-8")
    )
    assert [v["verdict"] for v in r["answer"]["verdicts"]] == ["CONFIRMED"] * 3
    assert "Not a whole-module certificate" in r["scope"]
