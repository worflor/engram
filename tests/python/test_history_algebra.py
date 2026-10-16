"""Ordered summary composition, decayed boundaries, and legacy information loss."""

import copy
import itertools
import numpy as np
import pytest
from engram_history import WitnessHistory


def history(rows, decay=1.0, dim=4):
    result = WitnessHistory(dim, decay)
    for row in rows:
        result.push(row)
    return result


def equal(a, b, atol=2e-9):
    assert a.samples == b.samples
    np.testing.assert_allclose(
        [a.weight, a.joint_weight], [b.weight, b.joint_weight], atol=atol
    )
    for name in ["mean", "joint_mean", "scatter", "first", "previous"]:
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=atol)
    if a.samples >= 3:
        for name in ["k", "g", "pair_rms"]:
            np.testing.assert_allclose(
                getattr(a.signature(), name), getattr(b.signature(), name), atol=atol
            )


@pytest.mark.parametrize("decay", [1.0, 0.99, 0.9, 0.5])
def test_all_short_boundaries_and_both_associations(decay):
    rng = np.random.default_rng(78)
    for lengths in itertools.product(range(4), repeat=3):
        x = rng.normal(size=(sum(lengths), 4))
        a, b = np.cumsum(lengths)[:2]
        chunks = [history(part, decay) for part in [x[:a], x[a:b], x[b:]]]
        full = history(x, decay)
        left = chunks[0].concat(chunks[1]).concat(chunks[2])
        right = chunks[0].concat(chunks[1].concat(chunks[2]))
        equal(full, left)
        equal(full, right)


@pytest.mark.parametrize("decay", [1.0, 0.995, 0.9, 0.1])
def test_long_large_offset_and_exponentially_aged_chunks(decay):
    x = np.random.default_rng(41).normal(size=(1000, 4)) + [1e5, -1e5, 3, 7]
    direct = history(x, decay)
    pieces = [history(part, decay) for part in np.array_split(x, [1, 2, 100, 101, 700])]
    joined = history([], decay)
    for piece in pieces:
        joined = joined.concat(piece)
    equal(direct, joined, atol=2e-7)
    assert len(joined.first) == len(joined.previous) == 2


def test_concat_is_ordered_and_does_not_mutate_operands():
    a = history([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]])
    b = history([[4, 1, 0, 0], [0, 3, 0, 0], [2, 7, 0, 0]])
    before = copy.deepcopy(a.snapshot()), copy.deepcopy(b.snapshot())
    ab, ba = a.concat(b), b.concat(a)
    assert not np.allclose(ab.joint_mean, ba.joint_mean)
    ab.push([100, 0, 0, 0])
    assert before == (a.snapshot(), b.snapshot())
    with pytest.raises(ValueError):
        a.concat(WitnessHistory(2))
    with pytest.raises(ValueError):
        a.concat(WitnessHistory(4, 0.9))


def test_old_snapshot_cannot_invent_its_missing_prefix():
    original = history(np.arange(24).reshape(6, 4))
    state = original.snapshot()
    state.pop("first")
    state.pop("version")
    old = WitnessHistory.restore(4, 1, state)
    assert old.first is None
    np.testing.assert_array_equal(old.signature().k, original.signature().k)
    with pytest.raises(ValueError, match="prefix"):
        original.concat(old)
    assert old.concat(original).first is None
    assert WitnessHistory(4).concat(old).first is None
    short = history([[1, 2, 3, 4]])
    state = short.snapshot()
    state.pop("first")
    state.pop("version")
    equal(short, WitnessHistory.restore(4, 1, state))


def test_multiset_collision_and_recency_distinction():
    a = np.column_stack([[0, 0, 0, 1, 0, 0], np.zeros(6)])
    b = np.column_stack([[0, 0, 1, 0, 0, 0], np.zeros(6)])
    for name in ["k", "g", "pair_rms"]:
        np.testing.assert_allclose(
            getattr(history(a, dim=2).signature(), name),
            getattr(history(b, dim=2).signature(), name),
            atol=1e-14,
        )
    assert (
        history(a, 0.9, 2)
        .signature()
        .compare(history(b, 0.9, 2).signature())["kg_rms_distance"]
        > 0.05
    )
