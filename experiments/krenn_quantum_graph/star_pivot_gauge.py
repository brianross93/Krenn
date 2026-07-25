r"""Exact gauge normalization of the ``n=6,d=3`` star-pivot opens.

For a fixed apex ``r`` and residual partner ``v``, the pure residual
coefficient ``P_v(a^4)`` is a sum of the three perfect matchings on the
four vertices outside ``{r,v}``.  Every term has direct-GHZ color-gauge
character

``sum_(u != r,v) theta_(u,a) = -(theta_(r,a) + theta_(v,a))``.

The three factors in a nonzero star-pivot minor use different colors, so
their characters form a primitive rank-three quotient of the 15-dimensional
direct-GHZ gauge torus.  Every point of the pivot open is therefore
gauge-equivalent, without extracting roots, to a point where all three
factors are one.  This replaces saturation by three polynomial slice
equations for the purpose of deciding existence on that open.

The additional twelve coordinate characters recorded here are only a
lattice audit.  Their corresponding weights need not be nonzero on the
pivot open, so setting those weights to one would define a narrower
numerical subchart, not a global gauge slice.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.star_linearization import (
    star_linearization,
)
from experiments.krenn_quantum_graph.star_pivot_charts import (
    StarPivotChart,
    star_pivot_charts,
)
from experiments.krenn_quantum_graph.system import (
    canonical_variable_key,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.vertical_component import (
    COLOR_GAUGE_PARAMETERS,
    color_diagonal_exponent_matrix,
    coloring_character,
    monomial_character,
)


N = 6
D = 3
STAR_PIVOT_GAUGE_AUDIT_SCHEMA = (
    "krenn-n6-d3-star-pivot-gauge-audit-v1"
)

# Apex-zero representatives of the three S5 x S3 partner-triple orbits.
_BASE_REPRESENTATIVES = (
    ("all-same", (1, 1, 1)),
    ("exactly-two-same", (1, 1, 2)),
    ("all-distinct", (1, 2, 3)),
)

# One determinant-two complement for apex zero.  Vertex transpositions
# transport it to every other apex.  Every listed edge avoids the apex.
_BASE_COMPLEMENT_KEYS = (
    (2, 3, 0, 0),
    (1, 2, 0, 1),
    (1, 2, 0, 2),
    (1, 2, 1, 0),
    (1, 2, 2, 0),
    (1, 3, 0, 0),
    (1, 3, 0, 1),
    (1, 3, 0, 2),
    (1, 4, 0, 0),
    (1, 4, 0, 1),
    (1, 4, 0, 2),
    (1, 5, 0, 0),
)


class KrennStarPivotGaugeError(RuntimeError):
    """A star-pivot gauge identity or exact replay failed."""


def _dot(left: Sequence[int], right: Sequence[int]) -> int:
    left = tuple(map(int, left))
    right = tuple(map(int, right))
    if len(left) != len(right):
        raise KrennStarPivotGaugeError(
            "a gauge-character pairing changed dimension"
        )
    return sum(
        a * b for a, b in zip(left, right, strict=True)
    )


def _rank_over_q(matrix: Sequence[Sequence[int]]) -> int:
    rows = [
        [Fraction(value) for value in row]
        for row in matrix
    ]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennStarPivotGaugeError(
            "an exact character matrix is ragged"
        )
    rank = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        value = rows[rank][column]
        rows[rank] = [entry / value for entry in rows[rank]]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            multiplier = rows[row][column]
            rows[row] = [
                left - multiplier * right
                for left, right in zip(
                    rows[row], rows[rank], strict=True
                )
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _rank_mod_two(matrix: Sequence[Sequence[int]]) -> int:
    rows = [
        [int(value) & 1 for value in row]
        for row in matrix
    ]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennStarPivotGaugeError(
            "a mod-two character matrix is ragged"
        )
    rank = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for row in range(len(rows)):
            if row != rank and rows[row][column]:
                rows[row] = [
                    left ^ right
                    for left, right in zip(
                        rows[row], rows[rank], strict=True
                    )
                ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _determinant_over_z(
    matrix: Sequence[Sequence[int]],
) -> int:
    """Return a square integer determinant by Bareiss elimination."""

    rows = [list(map(int, row)) for row in matrix]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise KrennStarPivotGaugeError(
            "an exact determinant matrix is not square"
        )
    if not rows:
        return 1
    sign = 1
    previous = 1
    for column in range(size - 1):
        pivot_row = next(
            (
                row
                for row in range(column, size)
                if rows[row][column]
            ),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != column:
            rows[column], rows[pivot_row] = (
                rows[pivot_row],
                rows[column],
            )
            sign *= -1
        pivot = rows[column][column]
        for row in range(column + 1, size):
            for other_column in range(column + 1, size):
                numerator = (
                    rows[row][other_column] * pivot
                    - rows[row][column]
                    * rows[column][other_column]
                )
                if numerator % previous:
                    raise KrennStarPivotGaugeError(
                        "Bareiss exact division failed"
                    )
                rows[row][other_column] = numerator // previous
            rows[row][column] = 0
        previous = pivot
    return sign * rows[-1][-1]


def _parameter_key(column: int) -> tuple[int, int]:
    column = int(column)
    if not 0 <= column < COLOR_GAUGE_PARAMETERS:
        raise KrennStarPivotGaugeError(
            "a color-gauge parameter column is invalid"
        )
    return divmod(column, D)


def _unimodular_minor_columns(
    rows: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    selected: list[int] = []
    rank = 0
    for column in range(COLOR_GAUGE_PARAMETERS):
        trial = [
            [row[index] for index in (*selected, column)]
            for row in rows
        ]
        trial_rank = _rank_over_q(trial)
        if trial_rank > rank:
            selected.append(column)
            rank = trial_rank
            if rank == len(rows):
                break
    if len(selected) != len(rows):
        raise KrennStarPivotGaugeError(
            "the factor characters lost full row rank"
        )
    minor = [
        [row[column] for column in selected]
        for row in rows
    ]
    if abs(_determinant_over_z(minor)) != 1:
        raise KrennStarPivotGaugeError(
            "the factor-character quotient is not primitive"
        )
    return tuple(selected)


def _apex_transposition(apex: int) -> tuple[int, ...]:
    apex = int(apex)
    if not 0 <= apex < N:
        raise KrennStarPivotGaugeError(
            "the apex is outside K6"
        )
    permutation = list(range(N))
    permutation[0], permutation[apex] = (
        permutation[apex],
        permutation[0],
    )
    return tuple(permutation)


def _transported_representatives(
    apex: int,
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    permutation = _apex_transposition(apex)
    return tuple(
        (
            pattern,
            tuple(permutation[vertex] for vertex in partners),
        )
        for pattern, partners in _BASE_REPRESENTATIVES
    )


def _transported_complement_indices(
    apex: int,
) -> tuple[int, ...]:
    permutation = _apex_transposition(apex)
    indices = []
    for i, j, a, b in _BASE_COMPLEMENT_KEYS:
        key = canonical_variable_key(
            N,
            D,
            permutation[i],
            permutation[j],
            a,
            b,
        )
        indices.append(variable_index(N, D, *key))
    result = tuple(indices)
    if (
        len(result) != 12
        or len(set(result)) != 12
        or any(
            apex in variable_key(N, D, index)[:2]
            for index in result
        )
    ):
        raise KrennStarPivotGaugeError(
            "the transported complement meets the apex star"
        )
    return result


def _chart_by_partners(
    apex: int,
    partners: Sequence[int],
) -> StarPivotChart:
    partners = tuple(map(int, partners))
    chart = next(
        (
            candidate
            for candidate in star_pivot_charts(apex)
            if candidate.partner_by_color == partners
        ),
        None,
    )
    if chart is None:
        raise KrennStarPivotGaugeError(
            "an orbit representative is absent from the pivot cover"
        )
    return chart


def _factor_characters(
    chart: StarPivotChart,
) -> tuple[tuple[int, ...], ...]:
    exponent_matrix = color_diagonal_exponent_matrix()
    rows = []
    for color, (partner, polynomial) in enumerate(
        zip(
            chart.partner_by_color,
            chart.quadratic_factors,
            strict=True,
        )
    ):
        incident = exponent_matrix[
            variable_index(
                N,
                D,
                chart.apex,
                partner,
                color,
                color,
            )
        ]
        expected = tuple(-value for value in incident)
        term_characters = tuple(
            monomial_character(monomial)
            for monomial in polynomial
        )
        if (
            len(term_characters) != 3
            or any(
                character != expected
                for character in term_characters
            )
        ):
            raise KrennStarPivotGaugeError(
                "a pure residual factor lost its semi-invariant character"
            )
        rows.append(expected)
    return tuple(rows)


def _normalization_exponent(
    apex: int,
    partner: int,
    color: int,
    residual_vertex: int,
) -> tuple[int, ...]:
    if residual_vertex in (apex, partner):
        raise KrennStarPivotGaugeError(
            "a normalization vertex lies outside the residual K4"
        )
    full = [[0] * D for _ in range(N)]
    full[apex][color] = 1
    full[residual_vertex][color] = -1
    if any(
        sum(full[vertex][entry] for vertex in range(N))
        for entry in range(D)
    ):
        raise KrennStarPivotGaugeError(
            "an explicit normalization violates product-one gauge"
        )
    return tuple(
        full[vertex][entry]
        for vertex in range(N - 1)
        for entry in range(D)
    )


def _normalization_audit(
    chart: StarPivotChart,
    factor_characters: Sequence[Sequence[int]],
) -> dict:
    eligibility = []
    deterministic_exponents = []
    for color, partner in enumerate(chart.partner_by_color):
        eligible = tuple(
            vertex
            for vertex in range(N)
            if vertex not in (chart.apex, partner)
        )
        if len(eligible) != 4:
            raise KrennStarPivotGaugeError(
                "a residual K4 does not have four normalization choices"
            )
        pairings = []
        cross_pairings = []
        for residual_vertex in eligible:
            exponent = _normalization_exponent(
                chart.apex,
                partner,
                color,
                residual_vertex,
            )
            pairings.append(
                _dot(factor_characters[color], exponent)
            )
            cross_pairings.append([
                _dot(other, exponent)
                for other in factor_characters
            ])
            for target_color in range(D):
                if _dot(
                    coloring_character([target_color] * N),
                    exponent,
                ):
                    raise KrennStarPivotGaugeError(
                        "a normalization changes a pure GHZ target"
                    )
        if pairings != [-1] * 4:
            raise KrennStarPivotGaugeError(
                "an explicit normalization does not invert its factor"
            )
        expected_cross = [
            -1 if other_color == color else 0
            for other_color in range(D)
        ]
        if any(
            row != expected_cross for row in cross_pairings
        ):
            raise KrennStarPivotGaugeError(
                "factor normalizations are not color-independent"
            )
        deterministic_exponents.append(
            _normalization_exponent(
                chart.apex,
                partner,
                color,
                eligible[0],
            )
        )
        eligibility.append({
            "color": color,
            "partner": partner,
            "eligible_residual_vertices": list(eligible),
            "eligible_choice_count": len(eligible),
            "selected_factor_character_pairings": pairings,
            "all_selected_factor_character_pairings": cross_pairings,
        })
    action = [
        [
            _dot(character, exponent)
            for exponent in deterministic_exponents
        ]
        for character in factor_characters
    ]
    if action != [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]:
        raise KrennStarPivotGaugeError(
            "the three explicit normalizations lost their split action"
        )
    return {
        "choices_by_color": eligibility,
        "simultaneous_choice_count": 4**3,
        "deterministic_action_matrix_on_selected_factors": action,
        "action_determinant": _determinant_over_z(action),
        "normalization_formula": (
            "lambda_(u,a)=P_v(a^4)^-1, "
            "lambda_(apex,a)=P_v(a^4), all others 1"
        ),
        "global_product_one_preserved": True,
        "direct_GHZ_target_preserved": True,
        "three_colors_normalize_independently": True,
        "root_extraction_required": False,
    }


def _elimination_audit(chart: StarPivotChart) -> dict:
    factorization = star_linearization(chart.apex)
    indices = tuple(
        factorization.star_variable_blocks[apex_color][column]
        for apex_color in range(D)
        for column in chart.columns_by_color
    )
    if (
        len(indices) != 9
        or len(set(indices)) != 9
        or any(
            chart.apex not in variable_key(N, D, index)[:2]
            for index in indices
        )
    ):
        raise KrennStarPivotGaugeError(
            "the shared pivot columns do not select nine star weights"
        )
    return {
        "shared_matrix_columns": list(chart.columns_by_color),
        "apex_color_system_count": D,
        "star_weights_eliminable_after_factor_normalization": len(
            indices
        ),
        "star_weight_indices": list(indices),
        "star_weight_keys": [
            list(variable_key(N, D, index))
            for index in indices
        ],
        "normalized_restricted_minor_determinant":
            chart.determinant_sign,
        "polynomial_substitution_requires_division": False,
    }


def _representative_audit(
    chart: StarPivotChart,
    pattern: str,
    complement_indices: Sequence[int],
) -> dict:
    factor_characters = _factor_characters(chart)
    rank = _rank_over_q(factor_characters)
    minor_columns = _unimodular_minor_columns(factor_characters)
    minor = [
        [row[column] for column in minor_columns]
        for row in factor_characters
    ]
    complement_rows = [
        color_diagonal_exponent_matrix()[index]
        for index in complement_indices
    ]
    square = [*factor_characters, *complement_rows]
    determinant = _determinant_over_z(square)
    if determinant != 2:
        raise KrennStarPivotGaugeError(
            "the oriented nonstar complement lost determinant +2"
        )
    nonstar_indices = tuple(
        index
        for index in range(len(color_diagonal_exponent_matrix()))
        if chart.apex not in variable_key(N, D, index)[:2]
    )
    full_nonstar_rows = [
        *factor_characters,
        *(
            color_diagonal_exponent_matrix()[index]
            for index in nonstar_indices
        ),
    ]
    full_ambient_rows = [
        *factor_characters,
        *color_diagonal_exponent_matrix(),
    ]
    if (
        rank != 3
        or _rank_over_q(square) != COLOR_GAUGE_PARAMETERS
        or _rank_mod_two(full_nonstar_rows) != 14
        or _rank_mod_two(full_ambient_rows) != 14
    ):
        raise KrennStarPivotGaugeError(
            "a pivot-character lattice invariant changed"
        )
    return {
        "equality_pattern": pattern,
        "partner_triple_by_color": list(chart.partner_by_color),
        "factor_characters": [
            list(row) for row in factor_characters
        ],
        "factor_character_rank_over_Q": rank,
        "factor_character_unimodular_minor_columns": list(
            minor_columns
        ),
        "factor_character_unimodular_minor_parameter_keys": [
            list(_parameter_key(column))
            for column in minor_columns
        ],
        "factor_character_unimodular_minor_determinant":
            _determinant_over_z(minor),
        "factor_character_smith_invariant_factors": [1, 1, 1],
        "normalization": _normalization_audit(
            chart, factor_characters
        ),
        "star_elimination": _elimination_audit(chart),
        "nonstar_complement": {
            "weight_indices": list(complement_indices),
            "weight_keys": [
                list(variable_key(N, D, index))
                for index in complement_indices
            ],
            "avoids_all_apex_star_weights": True,
            "combined_rank_over_Q": _rank_over_q(square),
            "oriented_determinant": determinant,
            "smith_invariant_factors": [1] * 14 + [2],
            "factor_plus_all_nonstar_rank_mod_2": 14,
            "factor_plus_all_ambient_rank_mod_2": 14,
            "lattice_index_two_is_minimal": True,
            "complex_torus_map_surjective": True,
            "finite_kernel_order": 2,
            "base_field_root_extraction_may_be_required": True,
        },
    }


def _oriented_complement(
    apex: int,
    representative_charts: Sequence[StarPivotChart],
) -> tuple[int, ...]:
    indices = list(_transported_complement_indices(apex))
    exponent_matrix = color_diagonal_exponent_matrix()

    def determinants() -> tuple[int, ...]:
        return tuple(
            _determinant_over_z([
                *_factor_characters(chart),
                *(exponent_matrix[index] for index in indices),
            ])
            for chart in representative_charts
        )

    values = determinants()
    if values == (-2, -2, -2):
        indices[0], indices[1] = indices[1], indices[0]
        values = determinants()
    if values != (2, 2, 2):
        raise KrennStarPivotGaugeError(
            "one common complement does not orient all three orbits"
        )
    return tuple(indices)


def _full_semi_invariant_census(apex: int) -> dict:
    chart_count = 0
    factor_count = 0
    term_count = 0
    for chart in star_pivot_charts(apex):
        _factor_characters(chart)
        chart_count += 1
        factor_count += D
        term_count += sum(
            len(polynomial)
            for polynomial in chart.quadratic_factors
        )
    if (chart_count, factor_count, term_count) != (
        125,
        375,
        1_125,
    ):
        raise KrennStarPivotGaugeError(
            "the full semi-invariant census changed"
        )
    return {
        "pivot_charts_checked": chart_count,
        "pure_residual_factors_checked": factor_count,
        "quadratic_matching_terms_checked": term_count,
        "every_term_character_equals": (
            "-(theta_(apex,a) + theta_(partner,a))"
        ),
        "every_term_replayed_exactly": True,
    }


@lru_cache(maxsize=1)
def star_pivot_gauge_audit() -> dict:
    """Return the strict six-apex gauge-normalization audit."""

    apex_rows = []
    for apex in range(N):
        representative_rows = _transported_representatives(apex)
        representative_charts = tuple(
            _chart_by_partners(apex, partners)
            for _pattern, partners in representative_rows
        )
        complement = _oriented_complement(
            apex, representative_charts
        )
        apex_rows.append({
            "apex": apex,
            "full_pivot_cover_semi_invariant_census":
                _full_semi_invariant_census(apex),
            "transported_orbit_representatives": [
                _representative_audit(
                    chart,
                    pattern,
                    complement,
                )
                for (
                    (pattern, _partners),
                    chart,
                ) in zip(
                    representative_rows,
                    representative_charts,
                    strict=True,
                )
            ],
            "one_common_nonstar_complement_for_all_three_orbits":
                True,
        })
    payload = {
        "schema": STAR_PIVOT_GAUGE_AUDIT_SCHEMA,
        "parameters": {
            "n": N,
            "d": D,
            "direct_GHZ_color_gauge_dimension":
                COLOR_GAUGE_PARAMETERS,
        },
        "semi_invariant_formula": {
            "factor": "P_v(a^4)",
            "character": (
                "sum_(u != apex,v) theta_(u,a) "
                "= -(theta_(apex,a)+theta_(v,a))"
            ),
            "reason": (
                "each residual K4 perfect matching covers its four "
                "vertices exactly once"
            ),
        },
        "apex_audits": apex_rows,
        "existence_equivalence": {
            "field": "C",
            "pivot_open_nonempty_iff_three_factor_unit_slice_nonempty":
                True,
            "unit_slice_equations": [
                "P_(v0)(0000)-1=0",
                "P_(v1)(1111)-1=0",
                "P_(v2)(2222)-1=0",
            ],
            "normalization_uses_only_inverse_of_nonzero_factor":
                True,
            "normalization_requires_root_extraction": False,
            "normalization_requires_Rabinowitsch_variable": False,
            "normalization_requires_saturation": False,
            "pivot_star_elimination_is_polynomial_on_unit_slice":
                True,
        },
        "claim_boundary": {
            "three_factor_unit_slice_covers_every_pivot_open_orbit":
                True,
            "twelve_coordinate_anchors_are_guaranteed_nonzero":
                False,
            "fifteen_anchor_numerical_chart_covers_entire_pivot_open":
                False,
            "fifteen_anchor_chart_is_valid_only_on_its_nonzero_subopen":
                True,
            "full_pivot_open_is_claimed_to_be_affine_space": False,
            "pivot_chart_solved": False,
            "finite_counterexample_found": False,
            "global_affine_membership_decided": False,
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
                _strict_json_equal(a, b)
                for a, b in zip(left, right, strict=True)
            )
        )
    return left == right


def verify_star_pivot_gauge_audit(payload: Mapping) -> dict:
    """Recompute the audit and reject altered data or promoted claims."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennStarPivotGaugeError(
            "the star-pivot gauge audit is not strict JSON"
        ) from error
    expected = star_pivot_gauge_audit()
    if not _strict_json_equal(normalized, expected):
        raise KrennStarPivotGaugeError(
            "the star-pivot gauge audit failed exact replay"
        )
    return normalized


__all__ = [
    "KrennStarPivotGaugeError",
    "STAR_PIVOT_GAUGE_AUDIT_SCHEMA",
    "star_pivot_gauge_audit",
    "verify_star_pivot_gauge_audit",
]
