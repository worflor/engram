"""Streaming, merge, and serialization checks for context relation moments."""

import copy
import json

import numpy as np
import pytest

from engram_relations import ContextRelations


def _stream(channels=("x", "y", "z"), representation="world-v1", rows=()):
    result = ContextRelations(channels, representation=representation)
    for context, row in rows:
        result.push(context, row)
    return result


def _arrays_equal(actual, expected, atol=1e-9):
    np.testing.assert_allclose(actual.mean, expected.mean, rtol=0, atol=atol)
    np.testing.assert_allclose(actual.scatter, expected.scatter, rtol=0, atol=atol)
    np.testing.assert_allclose(
        actual.covariance, expected.covariance, rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        actual.correlation, expected.correlation, rtol=0, atol=atol
    )


@pytest.mark.parametrize(
    "channels,representation",
    [
        ((), "r"),
        (("x", "x"), "r"),
        (("",), "r"),
        ((1,), "r"),
        (("x",), ""),
        (("x",), None),
        (["x"], "r"),
    ],
)
def test_schema_requires_unique_nonempty_channel_names_and_representation(
    channels, representation
):
    with pytest.raises((TypeError, ValueError)):
        ContextRelations(channels, representation=representation)


def test_streaming_centered_moments_match_independent_numpy_reference_with_offsets():
    rng = np.random.default_rng(912)
    rows = rng.normal(size=(1200, 4)) * [3.0, 0.2, 8.0, 1.5]
    rows += [1e9, -3e10, 7e8, 2e11]
    relations = ContextRelations(("a", "b", "c", "d"), representation="sensor-map")
    for row in rows:
        relations.push("safe", row)
    observed = relations.moments("safe")
    centered = rows - rows.mean(axis=0)
    reference_scatter = centered.T @ centered
    reference_covariance = reference_scatter / len(rows)
    assert observed.count == len(rows)
    np.testing.assert_allclose(observed.mean, rows.mean(axis=0), rtol=0, atol=3e-4)
    np.testing.assert_allclose(observed.scatter, reference_scatter, rtol=2e-7, atol=0.1)
    np.testing.assert_allclose(
        observed.covariance, reference_covariance, rtol=2e-7, atol=0.01
    )


def test_contexts_are_sorted_and_unknown_context_has_no_fabricated_summary():
    relations = _stream(
        channels=("x", "y"),
        rows=[("z", [1, 0]), ("a", [0, 1]), ("m", [2, 3])],
    )
    assert relations.contexts == ("a", "m", "z")
    assert relations.moments("missing") is None


def test_means_retain_constant_levels_even_when_correlations_are_zero():
    relations = _stream(
        channels=("constant", "moving", "also-constant"),
        rows=[("room", [7, 1, -4]), ("room", [7, 2, -4]), ("room", [7, 3, -4])],
    )
    summary = relations.moments("room")
    np.testing.assert_array_equal(summary.mean, [7, 2, -4])
    np.testing.assert_array_equal(summary.variances[[0, 2]], [0, 0])
    np.testing.assert_array_equal(summary.correlation[[0, 2], :], 0)
    np.testing.assert_array_equal(summary.correlation[:, [0, 2]], 0)
    assert summary.correlation[1, 1] == 1


def test_snapshot_arrays_are_copied_and_backed_by_immutable_storage():
    source = np.array([1.0, 2.0, 3.0])
    relations = _stream(rows=[("c", source)])
    source[:] = 100
    summary = relations.moments("c")
    np.testing.assert_array_equal(summary.mean, [1, 2, 3])
    for array in (
        summary.mean,
        summary.scatter,
        summary.covariance,
        summary.variances,
        summary.correlation,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_merge_matches_single_stream_for_multiple_contexts_and_is_independent():
    rng = np.random.default_rng(13)
    rows = [
        ("b" if i % 3 else "a", rng.normal(size=4) + [1e7, -2e8, 4, 0])
        for i in range(701)
    ]
    direct = _stream(channels=("w", "x", "y", "z"), rows=rows)
    left = _stream(channels=("w", "x", "y", "z"), rows=rows[:307])
    right = _stream(channels=("w", "x", "y", "z"), rows=rows[307:])
    merged = left.merge(right)
    assert merged.contexts == direct.contexts
    for context in direct.contexts:
        _arrays_equal(merged.moments(context), direct.moments(context), atol=2e-5)
    before_left = copy.deepcopy(left.to_dict())
    before_right = copy.deepcopy(right.to_dict())
    merged.push("a", [10, 20, 30, 40])
    assert left.to_dict() == before_left
    assert right.to_dict() == before_right


def test_empty_merge_is_identity_and_does_not_alias_summaries():
    populated = _stream(rows=[("c", [1, 2, 3]), ("c", [3, 2, 1])])
    empty = ContextRelations(
        populated.channels, representation=populated.representation
    )
    merged = empty.merge(populated)
    _arrays_equal(merged.moments("c"), populated.moments("c"))
    merged.push("c", [2, 2, 2])
    assert populated.moments("c").count == 2


def test_merge_requires_exact_channel_order_and_representation():
    base = _stream(channels=("x", "y"), representation="r", rows=[("c", [1, 2])])
    with pytest.raises(ValueError, match="schemas"):
        base.merge(_stream(channels=("y", "x"), representation="r"))
    with pytest.raises(ValueError, match="schemas"):
        base.merge(_stream(channels=("x", "y"), representation="r2"))
    with pytest.raises(TypeError):
        base.merge(object())


def test_version_one_json_roundtrip_and_mutable_input_do_not_alias():
    relations = _stream(
        channels=("x", "y"),
        representation="npc-witness-v2",
        rows=[("safe", [1, 5]), ("safe", [3, 7]), ("danger", [-4, 9])],
    )
    payload = relations.to_dict()
    restored = ContextRelations.from_dict(json.loads(json.dumps(payload)))
    assert restored.to_dict() == payload
    payload["contexts"]["safe"]["mean"][0] = 999
    assert restored.moments("safe").mean[0] == 2
    with pytest.raises(ValueError, match="version"):
        ContextRelations.from_dict({**relations.to_dict(), "version": 2})


@pytest.mark.parametrize(
    "bad_row,exception",
    [
        ([1, 2], ValueError),
        ([1, 2, 3, 4], ValueError),
        ([1, float("nan"), 3], ValueError),
        ([1, float("inf"), 3], ValueError),
        ([True, False, True], TypeError),
        ([1 + 2j, 2, 3], TypeError),
    ],
)
def test_invalid_push_is_rejected_without_changing_existing_state(bad_row, exception):
    relations = _stream(rows=[("c", [1, 2, 3]), ("c", [2, 4, 8])])
    before = relations.to_dict()
    with pytest.raises(exception):
        relations.push("c", bad_row)
    assert relations.to_dict() == before


def test_nonfinite_update_overflow_is_atomic():
    relations = _stream(channels=("x",), rows=[("c", [-1e308])])
    before = relations.to_dict()
    with pytest.raises(FloatingPointError):
        relations.push("c", [1e308])
    assert relations.to_dict() == before


def test_nonfinite_merge_overflow_raises_without_mutating_operands():
    left = _stream(channels=("x",), rows=[("c", [-1e308])])
    right = _stream(channels=("x",), rows=[("c", [1e308])])
    before_left, before_right = left.to_dict(), right.to_dict()
    with pytest.raises(FloatingPointError):
        left.merge(right)
    assert left.to_dict() == before_left
    assert right.to_dict() == before_right


def test_checkpoint_rejects_invalid_count_shape_nonfinite_and_missing_scatter():
    base = _stream(channels=("x", "y"), rows=[("c", [1, 2]), ("c", [2, 3])]).to_dict()
    mutations = []
    bad = copy.deepcopy(base)
    bad["contexts"]["c"]["count"] = True
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["contexts"]["c"]["mean"] = [1]
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["contexts"]["c"]["mean"][0] = float("nan")
    mutations.append(bad)
    bad = copy.deepcopy(base)
    del bad["contexts"]["c"]["scatter"]
    mutations.append(bad)
    for payload in mutations:
        with pytest.raises(ValueError):
            ContextRelations.from_dict(payload)


def test_checkpoint_rejects_material_asymmetry_and_negative_psd_direction():
    base = _stream(channels=("x", "y"), rows=[("c", [0, 0]), ("c", [1, 1])]).to_dict()
    asymmetric = copy.deepcopy(base)
    asymmetric["contexts"]["c"]["scatter"][0][1] += 1
    with pytest.raises(ValueError, match="symmetric"):
        ContextRelations.from_dict(asymmetric)
    indefinite = copy.deepcopy(base)
    indefinite["contexts"]["c"]["scatter"] = [[1, 2], [2, 1]]
    with pytest.raises(ValueError, match="semidefinite"):
        ContextRelations.from_dict(indefinite)


def test_checkpoint_requires_exact_integer_version_and_count_one_zero_scatter():
    base = _stream(channels=("x", "y"), rows=[("c", [1, 2])]).to_dict()
    boolean_version = copy.deepcopy(base)
    boolean_version["version"] = True
    with pytest.raises(ValueError, match="version"):
        ContextRelations.from_dict(boolean_version)
    nonzero = copy.deepcopy(base)
    nonzero["contexts"]["c"]["scatter"] = [[1e-5, 0], [0, 0]]
    with pytest.raises(ValueError, match="one-sample"):
        ContextRelations.from_dict(nonzero)
    roundoff = copy.deepcopy(base)
    roundoff["contexts"]["c"]["scatter"] = [[1e-12, 0], [0, 0]]
    with pytest.raises(ValueError, match="one-sample"):
        ContextRelations.from_dict(roundoff)


def test_checkpoint_rejects_rank_two_scatter_for_two_samples():
    base = _stream(channels=("x", "y"), rows=[("c", [0, 0]), ("c", [1, 1])]).to_dict()
    forged = copy.deepcopy(base)
    forged["contexts"]["c"]["scatter"] = [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="rank exceeds"):
        ContextRelations.from_dict(forged)


def test_longdouble_float64_cast_overflow_is_rejected_without_warning_or_mutation():
    huge = np.finfo(np.longdouble).max
    if not np.isfinite(huge) or huge <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64 range")
    relations = _stream(channels=("x",), rows=[("c", [3])])
    before = relations.to_dict()
    with pytest.raises(ValueError, match="finite"):
        relations.push("c", np.array([huge], dtype=np.longdouble))
    assert relations.to_dict() == before


def test_channel_and_representation_properties_cannot_be_rebound():
    relations = ContextRelations(("x", "y"), representation="fixed")
    with pytest.raises(AttributeError):
        relations.channels = ("other",)
    with pytest.raises(AttributeError):
        relations.representation = "other"


@pytest.mark.parametrize("context", ["", None, 1])
def test_context_labels_must_be_nonempty_strings(context):
    relations = ContextRelations(("x",), representation="r")
    with pytest.raises(ValueError):
        relations.push(context, [1])
    with pytest.raises(ValueError):
        relations.moments(context)


@pytest.mark.parametrize("bad_value", [1 + 2j, "1.25"])
def test_checkpoint_does_not_silently_cast_complex_or_textual_moments(bad_value):
    state = _stream(channels=("x",), rows=[("c", [1.0])]).to_dict()
    state["contexts"]["c"]["mean"] = [bad_value]
    with pytest.raises(ValueError):
        ContextRelations.from_dict(state)


@pytest.mark.parametrize("scale", [1e-250, 1.0, 1e300])
def test_checkpoint_symmetry_and_psd_checks_are_scale_invariant(scale):
    base = _stream(
        channels=("x", "y"), rows=[("c", [0, 0]), ("c", [1, 1]), ("c", [2, -1])]
    ).to_dict()
    base["contexts"]["c"]["scatter"] = (
        np.array([[2.0, 1.0], [1.0, 2.0]]) * scale
    ).tolist()
    restored = ContextRelations.from_dict(base)
    np.testing.assert_allclose(restored.moments("c").scatter / scale, [[2, 1], [1, 2]])
    base["contexts"]["c"]["scatter"] = (
        np.array([[1.0, 0.1], [0.0, 1.0]]) * scale
    ).tolist()
    with pytest.raises(ValueError, match="symmetric"):
        ContextRelations.from_dict(base)
    base["contexts"]["c"]["scatter"] = (
        np.array([[1.0, 2.0], [2.0, 1.0]]) * scale
    ).tolist()
    with pytest.raises(ValueError, match="semidefinite"):
        ContextRelations.from_dict(base)


def test_merge_does_not_overflow_before_applying_small_cross_weight():
    left = _stream(channels=("x",), rows=[("c", [-8e153])])
    right = _stream(channels=("x",), rows=[("c", [8e153])])
    result = left.merge(right).moments("c")
    assert np.isfinite(result.scatter).all()
    np.testing.assert_allclose(result.scatter / 1e308, [[1.28]], rtol=1e-14)
    direct = _stream(channels=("x",), rows=[("c", [-8e153]), ("c", [8e153])]).moments(
        "c"
    )
    np.testing.assert_allclose(result.covariance, direct.covariance)


def test_checkpoint_preserves_symmetric_subnormal_scatter():
    base = _stream(channels=("x",), rows=[("c", [0]), ("c", [1])]).to_dict()
    tiny = np.nextafter(0.0, 1.0)
    base["contexts"]["c"]["scatter"] = [[tiny]]
    assert ContextRelations.from_dict(base).moments("c").scatter[0, 0] == tiny
