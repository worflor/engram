"""Independent exact checks for the closed-polygon linking probe."""

from fractions import Fraction as F

import pytest

from research.experiments.relational_links import (
    LINK_A,
    LINK_B,
    _subdivide,
    _transform,
    build_report,
    linking_number,
)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _project(point):
    x, y, z = map(F, point)
    return (x + z / 3, y + z / 5), z


def _independent_crossing_signs(a, b):
    signs = []
    for a0raw, a1raw in zip(a, a[1:] + a[:1]):
        a0, h0 = _project(a0raw)
        a1, h1 = _project(a1raw)
        da = (a1[0] - a0[0], a1[1] - a0[1])
        for b0raw, b1raw in zip(b, b[1:] + b[:1]):
            b0, g0 = _project(b0raw)
            b1, g1 = _project(b1raw)
            db = (b1[0] - b0[0], b1[1] - b0[1])
            delta = (b0[0] - a0[0], b0[1] - a0[1])
            determinant = _cross(da, db)
            if determinant == 0:
                continue
            t = _cross(delta, db) / determinant
            s = _cross(delta, da) / determinant
            if 0 < t < 1 and 0 < s < 1:
                height_difference = h0 + t * (h1 - h0) - g0 - s * (g1 - g0)
                signs.append(
                    (1 if determinant > 0 else -1)
                    * (1 if height_difference > 0 else -1)
                )
    return signs


def test_exact_mixed_crossings_give_linking_number_minus_one():
    value, crossings = linking_number(LINK_A, LINK_B)
    assert value == -1
    assert len(crossings) == 2
    assert [row["sign"] for row in crossings] == [-1, -1]
    assert [row["sign"] for row in crossings] == _independent_crossing_signs(
        LINK_A, LINK_B
    )
    assert [(row["parameter_a"], row["parameter_b"]) for row in crossings] == [
        (F(9, 20), F(2, 3)),
        (F(11, 20), F(2, 3)),
    ]


def test_translation_reversal_and_exact_noncrossing_subdivision():
    assert linking_number(LINK_A, _transform(LINK_B, shift=(5, 0, 0)))[0] == 0
    assert linking_number(LINK_A, tuple(reversed(LINK_B)))[0] == 1
    subdivided = _subdivide(LINK_A, 0)
    assert linking_number(subdivided, LINK_B)[0] == -1


def test_64_positive_rational_scales_and_common_translations_preserve_link():
    for index in range(1, 65):
        scale = F(index, index + 3)
        shift = (F(index, 7), F(-index, 11), F(index % 5, 13))
        a = _transform(LINK_A, scale=scale, shift=shift)
        b = _transform(LINK_B, scale=scale, shift=shift)
        assert linking_number(a, b)[0] == -1


def test_projected_vertex_crossing_is_rejected():
    a = ((-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0))
    b = ((0, -2, 0), (3, 0, 1), (0, 3, 1), (-3, 0, 1))
    with pytest.raises(ValueError, match="vertex crossing"):
        linking_number(a, b)


def test_collinear_projected_edge_overlap_is_rejected():
    a = ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))
    b = ((1, 0, 0), (3, 0, 0), (3, -1, 0), (1, -1, 0))
    with pytest.raises(ValueError, match="collinear"):
        linking_number(a, b)


def test_zero_height_separation_at_proper_crossing_is_rejected():
    a = ((-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0))
    b = ((0, -3, 0), (3, 0, 0), (0, 3, 0), (-3, 0, 0))
    with pytest.raises(ValueError, match="zero height"):
        linking_number(a, b)


def test_nonexact_float_coordinates_are_rejected():
    with pytest.raises(TypeError, match="exact"):
        linking_number(((0.0, 0, 0), (1, 0, 0), (0, 1, 0)), LINK_B)


def test_report_records_scope_controls_and_rational_invariance_trials():
    report = build_report()
    assert report["linked_pair"]["linking_number"] == -1
    assert report["controls"]["common_positive_rational_scale_and_translation"] == {
        "trials": 64,
        "all_preserved": True,
    }
    assert "not a knot classifier" in report["scope"]
    assert "not evidence of semantic understanding" in report["scope"]
