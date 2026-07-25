r"""Exact target-row pivot charts for the ``n=6,d=3`` star system.

Fix an apex vertex.  The exact star factorization has one shared
``243 x 15`` matrix ``A`` for the three apex colors.  Restricting ``A`` to
the three constant residual colorings ``00000``, ``11111``, and ``22222``
gives a ``3 x 15`` matrix.  Its row of color ``a`` is supported only on
the five columns whose partner color is ``a``.

Consequently, among the ``binomial(15,3)=455`` three-column minors,
exactly ``5**3=125`` are structurally nonzero.  They are indexed by a
partner triple ``(v_0,v_1,v_2)`` and factor as

``sign * P_(v_0)(0000) * P_(v_1)(1111) * P_(v_2)(2222)``.

At a direct-GHZ solution the restricted matrix maps three star-weight
vectors to the three standard basis vectors, so it has rank three.
Thus the 125 determinant opens cover every solution in this fixed-apex
presentation.  This is a finite exact cover, not a solution of any open.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import combinations, permutations, product
import json
from math import comb
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.star_linearization import (
    QuadraticPolynomial,
    star_linearization,
)


N = 6
D = 3
TARGET_ROWS = (0, 121, 242)
STAR_PIVOT_CHART_SCHEMA = "krenn-n6-d3-star-pivot-chart-v1"
STAR_PIVOT_AUDIT_SCHEMA = "krenn-n6-d3-star-pivot-audit-v1"


class KrennStarPivotChartError(RuntimeError):
    """A target-row pivot chart or its exact replay changed."""


def _validated_apex(apex: int) -> int:
    if isinstance(apex, bool):
        raise KrennStarPivotChartError("the apex must be an integer")
    apex = int(apex)
    if not 0 <= apex < N:
        raise KrennStarPivotChartError(
            "the apex must be a K6 vertex"
        )
    return apex


def _permutation_sign(positions: Sequence[int]) -> int:
    positions = tuple(map(int, positions))
    if tuple(sorted(positions)) != tuple(range(len(positions))):
        raise KrennStarPivotChartError(
            "a determinant position tuple is not a permutation"
        )
    inversions = sum(
        positions[left] > positions[right]
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    )
    return -1 if inversions % 2 else 1


def _factor_fingerprint(
    factors: Sequence[QuadraticPolynomial],
) -> str:
    digest = hashlib.sha256()
    for color, factor in enumerate(factors):
        digest.update(f"{color}|".encode("ascii"))
        for monomial in factor:
            digest.update(
                f"{monomial[0]},{monomial[1]};".encode("ascii")
            )
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class StarPivotChart:
    """One structurally nonzero target-row minor."""

    apex: int
    partner_by_color: tuple[int, int, int]
    columns_by_color: tuple[int, int, int]
    minor_columns: tuple[int, int, int]
    row_column_positions: tuple[int, int, int]
    determinant_sign: int
    quadratic_factors: tuple[
        QuadraticPolynomial,
        QuadraticPolynomial,
        QuadraticPolynomial,
    ]
    schema: str = STAR_PIVOT_CHART_SCHEMA

    def __post_init__(self) -> None:
        apex = _validated_apex(self.apex)
        factorization = star_linearization(apex)
        remaining = factorization.remaining_vertices
        if (
            self.schema != STAR_PIVOT_CHART_SCHEMA
            or len(self.partner_by_color) != D
            or any(
                partner not in remaining
                for partner in self.partner_by_color
            )
            or len(set(self.columns_by_color)) != D
            or self.minor_columns
            != tuple(sorted(self.columns_by_color))
            or self.row_column_positions
            != tuple(
                self.minor_columns.index(column)
                for column in self.columns_by_color
            )
            or self.determinant_sign
            != _permutation_sign(self.row_column_positions)
            or any(len(factor) != 3 for factor in self.quadratic_factors)
        ):
            raise KrennStarPivotChartError(
                "a star pivot chart has malformed coordinates"
            )
        expected_columns = tuple(
            factorization.columns.index((partner, color))
            for color, partner in enumerate(self.partner_by_color)
        )
        expected_factors = tuple(
            factorization.entries[TARGET_ROWS[color]][column]
            for color, column in enumerate(expected_columns)
        )
        if (
            self.columns_by_color != expected_columns
            or self.quadratic_factors != expected_factors
        ):
            raise KrennStarPivotChartError(
                "a star pivot determinant failed factor replay"
            )

    @property
    def expanded_terms_before_collection(self) -> int:
        return 3**3

    def factor_fingerprint(self) -> str:
        return _factor_fingerprint(self.quadratic_factors)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "apex": self.apex,
            "partner_triple_by_color": list(self.partner_by_color),
            "columns_by_color": list(self.columns_by_color),
            "minor_columns_in_canonical_order": list(
                self.minor_columns
            ),
            "row_column_positions": list(
                self.row_column_positions
            ),
            "determinant_sign": self.determinant_sign,
            "factor_sha256": self.factor_fingerprint(),
            "factor_term_counts": [
                len(factor) for factor in self.quadratic_factors
            ],
            "expanded_terms_before_collection":
                self.expanded_terms_before_collection,
        }


@lru_cache(maxsize=N)
def star_pivot_charts(apex: int) -> tuple[StarPivotChart, ...]:
    """Return all 125 exact determinant opens for one apex."""

    apex = _validated_apex(apex)
    factorization = star_linearization(apex)
    charts = []
    for partners in product(
        factorization.remaining_vertices, repeat=D
    ):
        columns_by_color = tuple(
            factorization.columns.index((partner, color))
            for color, partner in enumerate(partners)
        )
        minor_columns = tuple(sorted(columns_by_color))
        positions = tuple(
            minor_columns.index(column)
            for column in columns_by_color
        )
        factors = tuple(
            factorization.entries[TARGET_ROWS[color]][column]
            for color, column in enumerate(columns_by_color)
        )
        charts.append(StarPivotChart(
            apex=apex,
            partner_by_color=partners,
            columns_by_color=columns_by_color,
            minor_columns=minor_columns,
            row_column_positions=positions,
            determinant_sign=_permutation_sign(positions),
            quadratic_factors=factors,
        ))
    result = tuple(charts)
    if (
        len(result) != 5**3
        or len({
            chart.partner_by_color for chart in result
        }) != len(result)
        or len({
            chart.minor_columns for chart in result
        }) != len(result)
    ):
        raise KrennStarPivotChartError(
            "the 125-open star pivot cover changed"
        )
    return result


def _minor_census(apex: int) -> tuple[int, int]:
    """Count zero and nonzero three-column minors by exact support."""

    factorization = star_linearization(_validated_apex(apex))
    structurally_nonzero = 0
    structurally_zero = 0
    chart_minors = {
        chart.minor_columns for chart in star_pivot_charts(apex)
    }
    for columns in combinations(range(15), 3):
        colors = tuple(
            factorization.columns[column][1]
            for column in columns
        )
        nonzero = set(colors) == set(range(D))
        if nonzero:
            structurally_nonzero += 1
            if columns not in chart_minors:
                raise KrennStarPivotChartError(
                    "a nonzero target-row minor is absent from the cover"
                )
        else:
            structurally_zero += 1
            if columns in chart_minors:
                raise KrennStarPivotChartError(
                    "a structurally zero target-row minor entered the cover"
                )
    return structurally_zero, structurally_nonzero


def _chart_cover_fingerprint(
    charts: Sequence[StarPivotChart],
) -> str:
    digest = hashlib.sha256()
    for chart in charts:
        digest.update(
            (
                f"{chart.apex}|{chart.partner_by_color}|"
                f"{chart.columns_by_color}|{chart.minor_columns}|"
                f"{chart.determinant_sign}|"
                f"{chart.factor_fingerprint()}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def _act_on_local_triple(
    triple: Sequence[int],
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, int, int]:
    """Apply ``S5 x S3`` to a local partner triple."""

    triple = tuple(map(int, triple))
    vertex_permutation = tuple(map(int, vertex_permutation))
    color_permutation = tuple(map(int, color_permutation))
    if (
        len(triple) != D
        or any(value not in range(5) for value in triple)
        or tuple(sorted(vertex_permutation)) != tuple(range(5))
        or tuple(sorted(color_permutation)) != tuple(range(D))
    ):
        raise KrennStarPivotChartError(
            "an orbit action coordinate is malformed"
        )
    result = [0] * D
    for old_color in range(D):
        result[color_permutation[old_color]] = (
            vertex_permutation[triple[old_color]]
        )
    return tuple(result)


def _equality_pattern(triple: Sequence[int]) -> str:
    multiplicities = sorted(
        (
            tuple(triple).count(value)
            for value in set(triple)
        ),
        reverse=True,
    )
    if multiplicities == [3]:
        return "all-same"
    if multiplicities == [2, 1]:
        return "exactly-two-same"
    if multiplicities == [1, 1, 1]:
        return "all-distinct"
    raise KrennStarPivotChartError(
        "a partner triple has an impossible equality pattern"
    )


@lru_cache(maxsize=1)
def star_pivot_orbit_audit() -> dict:
    """Enumerate the exact three ``S5 x S3`` equality-pattern orbits."""

    vertex_group = tuple(permutations(range(5)))
    color_group = tuple(permutations(range(D)))
    representatives = (
        ("all-same", (0, 0, 0)),
        ("exactly-two-same", (0, 0, 1)),
        ("all-distinct", (0, 1, 2)),
    )
    rows = []
    union = set()
    for pattern, representative in representatives:
        orbit = {
            _act_on_local_triple(
                representative,
                vertex_permutation,
                color_permutation,
            )
            for vertex_permutation in vertex_group
            for color_permutation in color_group
        }
        stabilizer = sum(
            _act_on_local_triple(
                representative,
                vertex_permutation,
                color_permutation,
            )
            == representative
            for vertex_permutation in vertex_group
            for color_permutation in color_group
        )
        if (
            any(_equality_pattern(triple) != pattern for triple in orbit)
            or len(orbit) * stabilizer
            != len(vertex_group) * len(color_group)
            or union.intersection(orbit)
        ):
            raise KrennStarPivotChartError(
                "the pivot-chart orbit classification failed"
            )
        union.update(orbit)
        rows.append({
            "equality_pattern": pattern,
            "representative_local_partner_triple": list(
                representative
            ),
            "orbit_size": len(orbit),
            "stabilizer_order": stabilizer,
        })
    universe = set(product(range(5), repeat=D))
    if union != universe:
        raise KrennStarPivotChartError(
            "the pivot-chart orbits do not partition all 125 triples"
        )
    return {
        "group": "S5 residual vertices x S3 colors",
        "group_order": len(vertex_group) * len(color_group),
        "universe_size": len(universe),
        "orbit_count": len(rows),
        "orbits": rows,
        "orbit_sizes_sum_to_125": (
            sum(row["orbit_size"] for row in rows) == 125
        ),
        "equality_pattern_is_complete_invariant": True,
    }


@lru_cache(maxsize=1)
def star_pivot_chart_audit() -> dict:
    """Return the strict six-apex pivot-cover audit."""

    apex_rows = []
    for apex in range(N):
        factorization = star_linearization(apex)
        nonzero_entries = 0
        coefficient_terms = 0
        for color, target_row in enumerate(TARGET_ROWS):
            for column, (_partner, partner_color) in enumerate(
                factorization.columns
            ):
                polynomial = factorization.entries[target_row][column]
                expected_nonzero = partner_color == color
                if bool(polynomial) != expected_nonzero:
                    raise KrennStarPivotChartError(
                        "the target-row color blocks changed"
                    )
                if polynomial:
                    nonzero_entries += 1
                    coefficient_terms += len(polynomial)
                    if len(polynomial) != 3:
                        raise KrennStarPivotChartError(
                            "a residual K4 factor lost a matching term"
                        )
        zero_minors, nonzero_minors = _minor_census(apex)
        charts = star_pivot_charts(apex)
        if (
            zero_minors + nonzero_minors != comb(15, 3)
            or nonzero_minors != len(charts)
            or len(charts) != 5**3
        ):
            raise KrennStarPivotChartError(
                "the target-row minor census changed"
            )
        apex_rows.append({
            "apex": apex,
            "residual_vertices": list(
                factorization.remaining_vertices
            ),
            "restricted_shape": [3, 15],
            "target_rows": list(TARGET_ROWS),
            "target_row_colorings": [
                [color] * 5 for color in range(D)
            ],
            "block_diagonal_by_partner_color": True,
            "nonzero_entries": nonzero_entries,
            "quadratic_coefficient_terms": coefficient_terms,
            "three_column_minors_total": comb(15, 3),
            "structurally_zero_minors": zero_minors,
            "structurally_nonzero_minors": nonzero_minors,
            "nonzero_minor_count_is_5_cubed": (
                nonzero_minors == 5**3
            ),
            "open_charts": [
                chart.to_dict() for chart in charts
            ],
            "open_chart_sha256": _chart_cover_fingerprint(charts),
        })
    payload = {
        "schema": STAR_PIVOT_AUDIT_SCHEMA,
        "parameters": {"n": N, "d": D},
        "apex_audits": apex_rows,
        "determinant_factorization": {
            "formula": (
                "sign * P_(v0)(0000) * P_(v1)(1111) "
                "* P_(v2)(2222)"
            ),
            "quadratic_matching_terms_per_factor": 3,
            "expanded_terms_before_collection": 27,
            "all_125_factorized_determinants_replayed_exactly": True,
        },
        "rank_and_cover": {
            "restricted_target_matrix_maximum_rank": 3,
            "direct_GHZ_membership_forces_restricted_rank": 3,
            "reason": (
                "restriction of the three star solves maps to the "
                "three independent standard basis vectors"
            ),
            "the_125_determinant_opens_cover_every_solution_per_apex":
                True,
            "all_six_apex_covers_replayed": True,
        },
        "orbit_classification": star_pivot_orbit_audit(),
        "claim_boundary": {
            "any_open_chart_solved": False,
            "finite_counterexample_found": False,
            "natural_chart_decided": False,
            "global_affine_membership_decided": False,
            "cover_is_a_nonexistence_proof": False,
        },
    }
    return json.loads(json.dumps(payload, allow_nan=False))


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            set(left) == set(right)
            and all(
                _strict_json_equal(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, list):
        return (
            len(left) == len(right)
            and all(
                _strict_json_equal(left_value, right_value)
                for left_value, right_value in zip(
                    left, right, strict=True
                )
            )
        )
    return left == right


def verify_star_pivot_chart_audit(payload: Mapping) -> dict:
    """Rebuild all six covers and reject any altered claim."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennStarPivotChartError(
            "the star pivot audit is not strict JSON"
        ) from error
    expected = star_pivot_chart_audit()
    if not _strict_json_equal(normalized, expected):
        raise KrennStarPivotChartError(
            "the star pivot audit failed exact replay"
        )
    return normalized


__all__ = [
    "KrennStarPivotChartError",
    "STAR_PIVOT_AUDIT_SCHEMA",
    "STAR_PIVOT_CHART_SCHEMA",
    "StarPivotChart",
    "star_pivot_chart_audit",
    "star_pivot_charts",
    "star_pivot_orbit_audit",
    "verify_star_pivot_chart_audit",
]
