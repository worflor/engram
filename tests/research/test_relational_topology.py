"""Exact finite probes for relational path and braid features."""

from fractions import Fraction as F

import pytest

from research.experiments import relational_topology as rt


def test_chen_signature_distinguishes_equal_endpoint_and_area_at_level_three():
    result = rt.path_collision_probe()
    assert result["level1_equal"]
    assert result["level2_equal"]
    assert not result["level3_equal"]
    assert result["endpoint"] == ["2", "2"]
    assert result["signed_area"] == "0"


def test_path_signature_is_exact_and_uses_standard_segment_exponential():
    signature = rt.path_signature(((1, 0), (0, 1)))
    assert signature[1] == (F(1), F(1))
    assert signature[2] == (F(1, 2), F(1), F(0), F(1, 2))
    assert signature[3] == (
        F(1, 6),
        F(1, 2),
        F(0),
        F(1, 2),
        F(0),
        F(0),
        F(0),
        F(1, 6),
    )


def test_path_area_is_oriented_and_reverses_sign():
    clockwise = rt.path_signature(((1, 0), (0, 1), (-1, 0), (0, -1)))
    counterclockwise = rt.path_signature(((0, 1), (1, 0), (0, -1), (-1, 0)))
    assert rt.signed_area(clockwise) == F(1)
    assert rt.signed_area(counterclockwise) == F(-1)


def test_braid_relation_and_inverse_cancellation_are_exact():
    assert rt.braid_matrix((1, 2, 1)) == rt.braid_matrix((2, 1, 2))
    assert rt.braid_matrix((1, -1, 2, -2)) == rt.I


def test_selected_braids_share_permutation_and_unsigned_counts_but_not_matrix():
    positive, negative = (1, 2, 1), (-1, -2, -1)
    assert rt.permutation(positive) == rt.permutation(negative)
    assert rt.unsigned_counts(positive) == rt.unsigned_counts(negative)
    assert rt.braid_matrix(positive) != rt.braid_matrix(negative)
    assert (
        rt.braid_campaign(samples=4, seed=12)["recognition"][
            "exponent_sum_baseline_balanced_accuracy"
        ]
        == 1.0
    )


def test_ordered_braid_pair_controls_for_signed_counts_and_exponent_sum():
    first, second = (1, 1, 2, 2), (1, 2, 2, 1)
    assert rt.permutation(first) == rt.permutation(second) == (0, 1, 2)
    assert rt.exponent_sum(first) == rt.exponent_sum(second) == 4
    assert rt.signed_counts(first) == rt.signed_counts(second) == (2, 0, 2, 0)
    assert rt.braid_matrix(first) != rt.braid_matrix(second)
    result = rt.ordered_braid_campaign(samples=128, seed=39)
    assert result["matrix_prototype_recognition"]["accuracy"] == 1.0
    assert all(
        value == 128
        for value in result["paired_inverse_insertions"]["feature_equal_pairs"].values()
    )
    assert result["scope"].startswith("Exact recognition")


def test_equivalent_word_campaign_recognizes_rewrite_classes_exactly():
    result = rt.braid_campaign(samples=128, seed=91)["recognition"]
    assert result["equivalent_words_classified"] == result["equivalent_words_total"]
    assert result["same_permutation_collisions"] == 128
    assert result["same_unsigned_count_collisions"] == 128


def test_matrix_representation_is_not_faithful_and_reports_exponent_sum():
    word = (1, 2) * 6
    assert rt.braid_matrix(word) == rt.I
    assert rt.exponent_sum(word) == 12


@pytest.mark.parametrize("depth", [0, 4, True])
def test_path_signature_rejects_unsupported_depth(depth):
    with pytest.raises(ValueError):
        rt.path_signature((), depth=depth)
