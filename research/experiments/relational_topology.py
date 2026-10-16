"""Exact probes of ordered path and braid features for relational memory.

Research harness only; this is not an installed Engram feature or a claim that
topological invariants encode semantics. Run from the repository root with
``python research/experiments/relational_topology.py`` to write the canonical
JSON report. All mathematical checks use integers/Fractions and Python's stdlib.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _outer(left: tuple[F, ...], right: tuple[F, ...]) -> tuple[F, ...]:
    return tuple(a * b for a in left for b in right)


def path_signature(path: tuple[tuple[int, int], ...], depth: int = 3):
    """Exact Chen signature of a piecewise-linear 2D path, through ``depth``.

    Tensor coordinates are lexicographic words over x,y. Segment signatures
    are exp(v) truncated at ``depth``; path concatenation uses Chen's identity.
    The path is open or closed as supplied; no implicit closure is introduced.
    """
    if type(depth) is not int or not 1 <= depth <= 3:
        raise ValueError("depth must be an integer from 1 through 3")
    signature = [(F(1),)] + [
        tuple(F(0) for _ in range(2**k)) for k in range(1, depth + 1)
    ]
    for vector in path:
        if len(vector) != 2 or any(type(v) is not int for v in vector):
            raise ValueError("path steps must be integer 2D vectors")
        v = tuple(map(F, vector))
        segment = [(F(1),)]
        power = (F(1),)
        for k in range(1, depth + 1):
            power = _outer(power, v)
            segment.append(tuple(q / math_factorial(k) for q in power))
        combined = []
        for k in range(depth + 1):
            terms = [_outer(signature[i], segment[k - i]) for i in range(k + 1)]
            combined.append(
                tuple(sum((term[j] for term in terms), F(0)) for j in range(2**k))
            )
        signature = combined
    return tuple(signature)


def math_factorial(n: int) -> int:
    return 1 if n <= 1 else n * math_factorial(n - 1)


def signed_area(signature) -> F:
    """Antisymmetric level-2 coordinate: one half of xdy-y dx."""
    if len(signature) < 3:
        raise ValueError("area requires a level-2 signature")
    level2 = signature[2]
    return (level2[1] - level2[2]) / 2


def path_collision_probe() -> dict:
    # Both start at (0,0), end at (2,2), and have zero signed area. Their
    # ordered third-level coordinates differ, so endpoint+area is insufficient.
    first = ((1, 0), (0, 1), (0, 1), (1, 0))
    second = ((0, 1), (1, 0), (1, 0), (0, 1))
    a, b = path_signature(first), path_signature(second)
    endpoints = (a[1], b[1])
    areas = (signed_area(a), signed_area(b))
    assert endpoints[0] == endpoints[1] == (F(2), F(2))
    assert areas[0] == areas[1] == 0
    assert a[3] != b[3]
    return {
        "paths": [[[str(x), str(y)] for x, y in p] for p in (first, second)],
        "endpoint": [str(v) for v in endpoints[0]],
        "signed_area": str(areas[0]),
        "level1_equal": a[1] == b[1],
        "level2_equal": a[2] == b[2],
        "level3_equal": a[3] == b[3],
        "level3_first": [str(x) for x in a[3]],
        "level3_second": [str(x) for x in b[3]],
        "scope": "Open finite witness paths; no canonical closure, knot invariant, or semantic interpretation is asserted.",
    }


Matrix = tuple[tuple[int, int], tuple[int, int]]
I: Matrix = ((1, 0), (0, 1))
SIGMA1: Matrix = ((1, 1), (0, 1))
SIGMA2: Matrix = ((1, 0), (-1, 1))
SIGMA1_INV: Matrix = ((1, -1), (0, 1))
SIGMA2_INV: Matrix = ((1, 0), (1, 1))
GENERATORS = {1: SIGMA1, 2: SIGMA2, -1: SIGMA1_INV, -2: SIGMA2_INV}


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2))
        for i in range(2)
    )  # type: ignore[return-value]


def braid_matrix(word: tuple[int, ...] | list[int]) -> Matrix:
    result = I
    for generator in word:
        result = matrix_multiply(result, GENERATORS[generator])
    return result


def exponent_sum(word: tuple[int, ...] | list[int]) -> int:
    return sum(1 if g > 0 else -1 for g in word)


def permutation(word: tuple[int, ...] | list[int]) -> tuple[int, int, int]:
    # The Artin map B3 -> S3 forgets crossing sign.
    result = [0, 1, 2]
    for generator in word:
        i = abs(generator) - 1
        result[i], result[i + 1] = result[i + 1], result[i]
    return tuple(result)


def unsigned_counts(word: tuple[int, ...] | list[int]) -> tuple[int, int]:
    counts = Counter(abs(g) for g in word)
    return counts[1], counts[2]


def signed_counts(word: tuple[int, ...] | list[int]) -> tuple[int, int, int, int]:
    counts = Counter(word)
    return counts[1], counts[-1], counts[2], counts[-2]


def _rewrite_once(word: list[int], rng: random.Random, sign: int = 1) -> list[int]:
    out = list(word)
    if rng.random() < 0.55:
        # Insert an exact adjacent inverse pair at a uniformly chosen location.
        g = sign * rng.choice((-2, -1, 1, 2))
        i = rng.randrange(len(out) + 1)
        out[i:i] = [g, -g]
        return out
    # Apply ABA <-> BAB, including the inverse relation.
    triples = ([1, 2, 1], [2, 1, 2], [-1, -2, -1], [-2, -1, -2])
    matches = []
    for i in range(max(0, len(out) - 2)):
        for left, right in (
            (triples[0], triples[1]),
            (triples[1], triples[0]),
            (triples[2], triples[3]),
            (triples[3], triples[2]),
        ):
            if out[i : i + 3] == left:
                matches.append((i, right))
    matches.sort(key=lambda item: item[0])
    if matches:
        i, replacement = rng.choice(matches)
        out[i : i + 3] = replacement
    else:
        g = sign * rng.choice((-2, -1, 1, 2))
        i = rng.randrange(len(out) + 1)
        out[i:i] = [g, -g]
    return out


def equivalent_word(
    seed: tuple[int, ...], rng: random.Random, edits: int, sign: int = 1
) -> tuple[int, ...]:
    word = list(seed)
    for _ in range(edits):
        word = _rewrite_once(word, rng, sign)
    return tuple(word)


def braid_campaign(samples: int = 1000, seed: int = 260101) -> dict:
    """Recognize two exact braid elements after semantics-preserving rewrites.

    Both base words have the same S3 permutation and same unsigned generator
    counts. One maps to matrix M, the other M^-1. The campaign tests whether the
    stated 2x2 representation preserves this distinction under equivalent-word
    rewrites; it does not test whether that representation is faithful for B3.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    rng = random.Random(seed)
    positive = (1, 2, 1)
    negative = (-1, -2, -1)
    positive_matrix, negative_matrix = braid_matrix(positive), braid_matrix(negative)
    assert positive_matrix != negative_matrix
    assert permutation(positive) == permutation(negative)
    assert unsigned_counts(positive) == unsigned_counts(negative) == (2, 1)

    # Matrix prototypes are the only classifier: exact equality after reducing
    # through the supplied representation. Paired edit counts remove a trivial
    # class/length cue from the permutation and unsigned-count controls.
    correct = 0
    count_collisions = 0
    permutation_collisions = 0
    max_word_length = 0
    for _ in range(samples):
        edits = rng.randint(1, 24)
        # Mirror the same edit schedule structurally by generating a candidate
        # from the negative seed with an equal number of independent valid edits.
        state = rng.getstate()
        a = equivalent_word(positive, rng, edits)
        rng.setstate(state)
        b = equivalent_word(negative, rng, edits, sign=-1)
        assert braid_matrix(a) == positive_matrix
        assert braid_matrix(b) == negative_matrix
        assert permutation(a) == permutation(b)
        assert unsigned_counts(a) == unsigned_counts(b)
        count_collisions += unsigned_counts(a) == unsigned_counts(b)
        permutation_collisions += permutation(a) == permutation(b)
        for word, expected in ((a, 0), (b, 1)):
            observed = 0 if braid_matrix(word) == positive_matrix else 1
            correct += observed == expected
            max_word_length = max(max_word_length, len(word))

    aba, bab = (1, 2, 1), (2, 1, 2)
    cancel = (1, -1, 2, -2)
    ab = (1, 2)
    sixth_power = ab * 6
    assert braid_matrix(aba) == braid_matrix(bab)
    assert braid_matrix(cancel) == I
    assert braid_matrix(sixth_power) == I
    assert exponent_sum(sixth_power) == 12
    return {
        "matrix_sigma1": SIGMA1,
        "matrix_sigma2": SIGMA2,
        "checks": {
            "aba_equals_bab": braid_matrix(aba) == braid_matrix(bab),
            "adjacent_inverse_cancellation": braid_matrix(cancel) == I,
            "ab_power_6_is_identity": braid_matrix(sixth_power) == I,
            "ab_power_6_exponent_sum": exponent_sum(sixth_power),
        },
        "recognition": {
            "samples_per_class": samples,
            "seed": seed,
            "equivalent_words_classified": correct,
            "equivalent_words_total": 2 * samples,
            "accuracy": correct / (2 * samples),
            "max_word_length": max_word_length,
            "same_permutation_collisions": permutation_collisions,
            "same_unsigned_count_collisions": count_collisions,
            "paired_control_baselines": {
                "permutation_only_balanced_accuracy": 0.5,
                "unsigned_generator_counts_only_balanced_accuracy": 0.5,
                "basis": "Every positive/negative generated pair has identical control features and opposite labels; 0.5 is the exact balanced accuracy for any deterministic classifier using only that feature.",
            },
            "exponent_sum_baseline_balanced_accuracy": 1.0,
            "exponent_sum_baseline_note": "The positive and negative base words have exponent sums +3 and -3; this comparison alone is trivially solved by exponent sum.",
            "base_positive_matrix": positive_matrix,
            "base_negative_matrix": negative_matrix,
            "base_permutation": permutation(positive),
            "base_unsigned_counts": unsigned_counts(positive),
        },
        "scope": "This SL(2,Z) representation distinguishes the selected pair but is not faithful: (sigma1 sigma2)^6 maps to I despite exponent sum 12. Recognition is authored algebraic equivalence, not learned semantics.",
    }


def _insert_inverse_pairs(
    word: tuple[int, ...], rng: random.Random, pairs: int
) -> tuple[int, ...]:
    result = list(word)
    for _ in range(pairs):
        generator = rng.choice((-2, -1, 1, 2))
        at = rng.randrange(len(result) + 1)
        result[at:at] = [generator, -generator]
    return tuple(result)


def ordered_braid_campaign(samples: int = 1000, seed: int = 260102) -> dict:
    """Test word order after matching permutation, exponent, and signed counts.

    This is an authored exact distinction, not a learned semantic classifier:
    matrix prototypes label two base words, then exact inverse-pair insertions
    create held-out equivalent spellings. The raw ordered word itself is also a
    perfect classifier, so this experiment makes no claim that matrices beat a
    general ordered-word model.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    first, second = (1, 1, 2, 2), (1, 2, 2, 1)
    first_matrix, second_matrix = braid_matrix(first), braid_matrix(second)
    assert first_matrix != second_matrix
    assert permutation(first) == permutation(second) == (0, 1, 2)
    assert exponent_sum(first) == exponent_sum(second) == 4
    assert signed_counts(first) == signed_counts(second) == (2, 0, 2, 0)

    rng = random.Random(seed)
    correct = 0
    paired_equal_features = {
        "permutation": 0,
        "exponent_sum": 0,
        "signed_counts": 0,
        "unsigned_counts": 0,
    }
    maximum_length = 0
    for _ in range(samples):
        pairs = rng.randint(1, 24)
        state = rng.getstate()
        a = _insert_inverse_pairs(first, rng, pairs)
        rng.setstate(state)
        b = _insert_inverse_pairs(second, rng, pairs)
        assert braid_matrix(a) == first_matrix
        assert braid_matrix(b) == second_matrix
        paired_equal_features["permutation"] += permutation(a) == permutation(b)
        paired_equal_features["exponent_sum"] += exponent_sum(a) == exponent_sum(b)
        paired_equal_features["signed_counts"] += signed_counts(a) == signed_counts(b)
        paired_equal_features["unsigned_counts"] += unsigned_counts(
            a
        ) == unsigned_counts(b)
        for word, expected in ((a, 0), (b, 1)):
            observed = 0 if braid_matrix(word) == first_matrix else 1
            correct += observed == expected
            maximum_length = max(maximum_length, len(word))

    assert all(value == samples for value in paired_equal_features.values())
    return {
        "samples_per_class": samples,
        "seed": seed,
        "base_words": [first, second],
        "base_matrices": [first_matrix, second_matrix],
        "base_permutation": permutation(first),
        "base_exponent_sum": exponent_sum(first),
        "base_signed_counts": signed_counts(first),
        "paired_inverse_insertions": {
            "feature_equal_pairs": paired_equal_features,
            "maximum_word_length": maximum_length,
        },
        "matrix_prototype_recognition": {
            "correct": correct,
            "total": 2 * samples,
            "accuracy": correct / (2 * samples),
        },
        "control_baselines": {
            "permutation_only_balanced_accuracy": 0.5,
            "exponent_sum_only_balanced_accuracy": 0.5,
            "signed_generator_counts_only_balanced_accuracy": 0.5,
            "unsigned_generator_counts_only_balanced_accuracy": 0.5,
            "basis": "Every paired example has equal stated control features and opposite labels, so those controls cannot separate a pair.",
        },
        "scope": "Exact recognition of two authored B3 elements under free cancellation. The raw ordered word is itself sufficient; this does not show a matrix advantage over a general ordered-word model, nor that braids encode semantic relations.",
    }


def _json_safe(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def build_report() -> dict:
    report = {
        "schema": "engram.relational-topology/v1",
        "provenance": "Exact finite mathematical probes authored for relational-memory design; no Minecraft or Pet Plus runtime was executed.",
        "path_signature": {
            "definition": "Truncated Chen signature of piecewise-linear open paths in R^2, exact rational tensor coordinates, levels 1-3.",
            "collision_probe": path_collision_probe(),
        },
        "braid_representation": braid_campaign(),
        "order_controlled_braid_campaign": ordered_braid_campaign(),
        "knot_scope": "Out of scope: these finite witnesses are open paths; no closure convention is defined, so no knot score is computed.",
    }
    source_files = [
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "tests/research/test_relational_topology.py",
    ]
    report["source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in source_files
        if (ROOT / path).is_file()
    }
    return report


def main() -> None:
    report = _json_safe(build_report())
    destination = ROOT / "research/relational-topology.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
