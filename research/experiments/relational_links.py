"""Exact rational linking-number probe for two closed polygonal curves.

Research-only construction. This computes the classical linking number of two
disjoint, oriented, closed polygons from a generic projection. It is not a knot
classifier, does not apply to open witness histories, and makes no semantic
claim. All predicates and intersection parameters use ``fractions.Fraction``.
"""

from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

Point3 = tuple[F, F, F]
Point2 = tuple[F, F]


def _fraction_point(point) -> Point3:
    if len(point) != 3:
        raise ValueError("each polygon vertex must have three rational coordinates")
    if any(isinstance(value, float) or isinstance(value, bool) for value in point):
        raise TypeError("exact integer or Fraction coordinates are required")
    try:
        result = tuple(F(value) for value in point)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise TypeError("exact integer or Fraction coordinates are required") from error
    return result  # type: ignore[return-value]


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross2(a: Point2, b: Point2) -> F:
    return a[0] * b[1] - a[1] * b[0]


def _dot2(a: Point2, b: Point2) -> F:
    return a[0] * b[0] + a[1] * b[1]


def _project(point: Point3) -> tuple[Point2, F]:
    x, y, z = point
    return (x + z / 3, y + z / 5), z


def _segments(polygon: tuple[Point3, ...]):
    for index, start in enumerate(polygon):
        yield index, start, polygon[(index + 1) % len(polygon)]


def linking_number(polygon_a, polygon_b) -> tuple[int, list[dict]]:
    """Return oriented linking number and exact mixed-crossing witnesses.

    Projection is ``u=x+z/3, v=y+z/5`` with height ``w=z``. A proper
    projected crossing contributes
    ``sign(det(tangent_a,tangent_b))*sign(height_a-height_b)``; the sum divided
    by two is the linking number. Degenerate projection events are rejected.
    Polygons are implicitly closed by joining the final vertex to the first.
    """
    a = tuple(_fraction_point(point) for point in polygon_a)
    b = tuple(_fraction_point(point) for point in polygon_b)
    if len(a) < 3 or len(b) < 3:
        raise ValueError("each closed polygon needs at least three vertices")
    total = 0
    crossings = []
    for index_a, (point_a0, point_a1) in enumerate(zip(a, a[1:] + a[:1])):
        a0, wa0 = _project(point_a0)
        a1, wa1 = _project(point_a1)
        tangent_a = _sub(a1, a0)
        norm_a = _dot2(tangent_a, tangent_a)
        if norm_a == 0:
            raise ValueError("projection collapses a polygon edge")
        for index_b, (point_b0, point_b1) in enumerate(zip(b, b[1:] + b[:1])):
            b0, wb0 = _project(point_b0)
            b1, wb1 = _project(point_b1)
            tangent_b = _sub(b1, b0)
            norm_b = _dot2(tangent_b, tangent_b)
            if norm_b == 0:
                raise ValueError("projection collapses a polygon edge")
            delta = _sub(b0, a0)
            determinant = _cross2(tangent_a, tangent_b)
            if determinant == 0:
                if _cross2(delta, tangent_a) != 0:
                    continue
                # Parallel collinear segments: reject overlap or a single
                # projected touch, rather than inventing crossing signs.
                first = _dot2(delta, tangent_a) / norm_a
                second = first + _dot2(tangent_b, tangent_a) / norm_a
                overlap_lo = max(F(0), min(first, second))
                overlap_hi = min(F(1), max(first, second))
                if overlap_lo <= overlap_hi:
                    raise ValueError("collinear projected edges overlap or touch")
                continue
            t = _cross2(delta, tangent_b) / determinant
            s = _cross2(delta, tangent_a) / determinant
            if not (0 <= t <= 1 and 0 <= s <= 1):
                continue
            if t in (0, 1) or s in (0, 1):
                raise ValueError("projected vertex crossing is not generic")
            height_delta = (wa0 + t * (wa1 - wa0)) - (wb0 + s * (wb1 - wb0))
            if height_delta == 0:
                raise ValueError("projected crossing has zero height separation")
            sign = (1 if determinant > 0 else -1) * (1 if height_delta > 0 else -1)
            total += sign
            crossings.append(
                {
                    "edge_a": index_a,
                    "edge_b": index_b,
                    "parameter_a": t,
                    "parameter_b": s,
                    "determinant": determinant,
                    "height_a_minus_b": height_delta,
                    "sign": sign,
                }
            )
    if total % 2:
        raise ArithmeticError(
            "mixed crossing sum is odd; projection/sign convention failed"
        )
    return total // 2, crossings


LINK_A = (
    (-2, -2, 0),
    (2, -2, 0),
    (2, 2, 0),
    (-2, 2, 0),
)
LINK_B = (
    (1, 0, -1),
    (3, 0, -1),
    (3, 0, 1),
    (1, 0, 1),
)


def _transform(polygon, scale=F(1), shift=(F(0), F(0), F(0))):
    return tuple(
        tuple(
            scale * F(coordinate) + F(offset)
            for coordinate, offset in zip(point, shift)
        )
        for point in polygon
    )


def _subdivide(polygon, edge_index: int) -> tuple[Point3, ...]:
    points = tuple(_fraction_point(point) for point in polygon)
    start = points[edge_index]
    end = points[(edge_index + 1) % len(points)]
    midpoint = tuple((x + y) / 2 for x, y in zip(start, end))
    return points[: edge_index + 1] + (midpoint,) + points[edge_index + 1 :]


def _safe_crossing_row(crossing: dict) -> dict:
    return {
        key: str(value) if isinstance(value, F) else value
        for key, value in crossing.items()
    }


def build_report() -> dict:
    linked, crossings = linking_number(LINK_A, LINK_B)
    translated, translated_crossings = linking_number(
        LINK_A, _transform(LINK_B, shift=(5, 0, 0))
    )
    reversed_linking, reversed_crossings = linking_number(
        LINK_A, tuple(reversed(LINK_B))
    )
    subdivided, subdivided_crossings = linking_number(_subdivide(LINK_A, 0), LINK_B)
    invariance_trials = 64
    for index in range(1, invariance_trials + 1):
        scale = F(index, index + 3)
        shift = (F(index, 7), F(-index, 11), F(index % 5, 13))
        moved_a = _transform(LINK_A, scale=scale, shift=shift)
        moved_b = _transform(LINK_B, scale=scale, shift=shift)
        assert linking_number(moved_a, moved_b)[0] == linked
    assert abs(linked) == 1
    assert translated == 0 and not translated_crossings
    assert reversed_linking == -linked
    assert subdivided == linked
    assert len(crossings) == len(subdivided_crossings) == 2
    return {
        "schema": "engram.relational-links/v1",
        "provenance": "Exact rational polygon experiment; no game runtime or semantic inference was involved.",
        "projection": {
            "u": "x + z/3",
            "v": "y + z/5",
            "height": "z",
            "crossing_sign": "sign(det(tangent_a,tangent_b))*sign(height_a-height_b)",
        },
        "linked_pair": {
            "polygon_a": LINK_A,
            "polygon_b": LINK_B,
            "crossings": [_safe_crossing_row(row) for row in crossings],
            "signed_crossing_sum": sum(row["sign"] for row in crossings),
            "linking_number": linked,
        },
        "controls": {
            "translate_b_by_x_5": {
                "linking_number": translated,
                "proper_mixed_crossings": len(translated_crossings),
            },
            "reverse_b_orientation": {
                "linking_number": reversed_linking,
                "crossing_signs": [row["sign"] for row in reversed_crossings],
            },
            "subdivide_a_edge_0_at_midpoint": {
                "linking_number": subdivided,
                "proper_mixed_crossings": len(subdivided_crossings),
                "subdivision_is_not_at_crossing": True,
            },
            "common_positive_rational_scale_and_translation": {
                "trials": invariance_trials,
                "all_preserved": True,
            },
        },
        "rejection_contract": [
            "projected vertex crossings",
            "collinear projected edge overlaps or touches",
            "projected zero-length edges",
            "zero height separation at a projected mixed crossing",
        ],
        "scope": "Linking number applies to two disjoint oriented closed curves. It is not a knot classifier, does not apply to arbitrary open emotional histories without a declared closure, and is not evidence of semantic understanding.",
    }


def _json_safe(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def main() -> None:
    report = _json_safe(build_report())
    sources = [
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "tests/research/test_relational_links.py",
    ]
    report["source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in sources
        if (ROOT / path).is_file()
    }
    destination = ROOT / "research/relational-links.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
