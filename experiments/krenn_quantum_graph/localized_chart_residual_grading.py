r"""Exact residual-character grading on the four natural repair charts.

The normalized natural seed chart carries a nine-dimensional algebraic
torus action.  A character ``chi_j in Z^9`` records how a chart variable
transforms.  It is not a tropical valuation.  A cocharacter ``lambda``
produces the valuation-lineality shift

    w_j -> w_j + <lambda, chi_j>.

Consequently the character lattice supplies a lineality space for any
eventual Gröbner or tropical fan; it does not itself enumerate that fan.

On a repair-monomial chart, the factors fixed to one have independent
characters.  This module quotients their character span integrally, assigns
the branch inverse its contragredient character, and checks every term of
all 730 generators.  The resulting blocks are exact input structure for a
graded F4/Buchberger computation.  They do not decide whether a chart ideal
is unit or proper.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import gcd
import hashlib
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_ideals import (
    strict_json_equal,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    residual_torus_characters,
)
from experiments.krenn_quantum_graph.localized_chart_monomial_atlas import (
    REPAIR_MONOMIAL_REPRESENTATIVES,
    repair_monomial_chart,
)


RESIDUAL_GRADING_SCHEMA = (
    "krenn-n6-d3-natural-repair-residual-grading-v1"
)
AMBIENT_CHARACTER_RANK = 9

EXPECTED_REPAIR_ROWS = {
    (11, 65): {
        "fixed_rank": 1,
        "residual_rank": 8,
        "variable_count": 129,
        "zero_character_variables": 5,
        "character_blocks": 558,
        "zero_character_generators": 7,
        "multiplicity_histogram": ((1, 415), (2, 126), (3, 8), (4, 8), (7, 1)),
        "character_sha256":
            "c5246fd9056a4f32d10eb68f8a9574ed1593346d87263318d5886cdeddcab50c",
    },
    (29, 47): {
        "fixed_rank": 1,
        "residual_rank": 8,
        "variable_count": 129,
        "zero_character_variables": 5,
        "character_blocks": 642,
        "zero_character_generators": 7,
        "multiplicity_histogram": ((1, 567), (2, 66), (3, 8), (7, 1)),
        "character_sha256":
            "c43c11b2443fc10439853b600f02bc8ae199cd814968878db1605f8c15de06ab",
    },
    (11, 55, 133): {
        "fixed_rank": 2,
        "residual_rank": 7,
        "variable_count": 128,
        "zero_character_variables": 8,
        "character_blocks": 432,
        "zero_character_generators": 11,
        "multiplicity_histogram": (
            (1, 243),
            (2, 132),
            (3, 28),
            (4, 20),
            (6, 8),
            (11, 1),
        ),
        "character_sha256":
            "4900679206ad139f515b18b97664bd8b5e13097ec400fbd064c24469cb7f7057",
    },
    (29, 55, 106): {
        "fixed_rank": 2,
        "residual_rank": 7,
        "variable_count": 128,
        "zero_character_variables": 8,
        "character_blocks": 576,
        "zero_character_generators": 11,
        "multiplicity_histogram": (
            (1, 491),
            (2, 36),
            (3, 36),
            (4, 12),
            (11, 1),
        ),
        "character_sha256":
            "3f2c95e2deba62e06696918e0e7733ec4d5848c44dc4193b5c57651b530a63ba",
    },
}

Character = tuple[int, ...]


class KrennResidualGradingError(RuntimeError):
    """An integral quotient, character, or claim-boundary replay failed."""


def _rank_over_q(rows: Sequence[Sequence[int]]) -> int:
    matrix = [list(map(Fraction, row)) for row in rows]
    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise KrennResidualGradingError("a character matrix is ragged")
    rank = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(rank, len(matrix))
                if matrix[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [
            value / pivot_value for value in matrix[rank]
        ]
        for row in range(len(matrix)):
            if row == rank or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                left - factor * right
                for left, right in zip(
                    matrix[row], matrix[rank], strict=True
                )
            ]
        rank += 1
    return rank


def _determinant(rows: Sequence[Sequence[int]]) -> int:
    matrix = [list(map(Fraction, row)) for row in rows]
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise KrennResidualGradingError(
            "an integral minor is not square"
        )
    determinant = Fraction(1)
    sign = 1
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if matrix[row][column]
            ),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            matrix[column], matrix[pivot] = (
                matrix[pivot],
                matrix[column],
            )
            sign *= -1
        pivot_value = matrix[column][column]
        determinant *= pivot_value
        for row in range(column + 1, size):
            if not matrix[row][column]:
                continue
            factor = matrix[row][column] / pivot_value
            for other in range(column + 1, size):
                matrix[row][other] -= factor * matrix[column][other]
    result = determinant * sign
    if result.denominator != 1:
        raise KrennResidualGradingError(
            "an integer matrix acquired a fractional determinant"
        )
    return result.numerator


def _smith_diagonal(rows: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Return invariant factors for the small full-row-rank matrices."""

    normalized = tuple(tuple(map(int, row)) for row in rows)
    rank = _rank_over_q(normalized)
    if rank != len(normalized):
        raise KrennResidualGradingError(
            "fixed repair characters are not independent"
        )
    if not normalized:
        return ()
    width = len(normalized[0])
    previous_divisor = 1
    diagonal = []
    for order in range(1, rank + 1):
        divisor = 0
        for selected_rows in combinations(range(rank), order):
            for selected_columns in combinations(range(width), order):
                minor = tuple(
                    tuple(
                        normalized[row][column]
                        for column in selected_columns
                    )
                    for row in selected_rows
                )
                divisor = gcd(divisor, abs(_determinant(minor)))
        if not divisor or divisor % previous_divisor:
            raise KrennResidualGradingError(
                "fixed-character determinantal divisors are invalid"
            )
        diagonal.append(divisor // previous_divisor)
        previous_divisor = divisor
    return tuple(diagonal)


def _quotient_one_unit_row(
    vector: Sequence[int],
    relation: Sequence[int],
    pivot: int,
) -> Character:
    vector = tuple(map(int, vector))
    relation = tuple(map(int, relation))
    pivot_value = relation[pivot]
    if abs(pivot_value) != 1:
        raise KrennResidualGradingError(
            "integral quotient elimination needs a unit pivot"
        )
    multiplier = vector[pivot] * pivot_value
    reduced = tuple(
        left - multiplier * right
        for left, right in zip(vector, relation, strict=True)
    )
    if reduced[pivot]:
        raise KrennResidualGradingError(
            "unit-pivot quotient elimination failed"
        )
    return tuple(
        value for index, value in enumerate(reduced) if index != pivot
    )


def _integral_quotient(
    fixed_characters: Sequence[Sequence[int]],
) -> dict:
    """Construct a split integral quotient by the fixed-character lattice."""

    fixed = tuple(tuple(map(int, row)) for row in fixed_characters)
    if (
        any(len(row) != AMBIENT_CHARACTER_RANK for row in fixed)
        or _rank_over_q(fixed) != len(fixed)
    ):
        raise KrennResidualGradingError(
            "fixed-character matrix has the wrong rank or width"
        )
    smith = _smith_diagonal(fixed)
    if smith != (1,) * len(fixed):
        raise KrennResidualGradingError(
            "fixed-character lattice is not saturated"
        )

    current_relations = list(fixed)
    images = [
        tuple(int(row == column) for column in range(AMBIENT_CHARACTER_RANK))
        for row in range(AMBIENT_CHARACTER_RANK)
    ]
    pivot_coordinates = []
    while current_relations:
        relation = current_relations.pop(0)
        pivot = next(
            (
                index
                for index, value in enumerate(relation)
                if abs(value) == 1
            ),
            None,
        )
        if pivot is None:
            raise KrennResidualGradingError(
                "the retained saturated quotient lacks a unit pivot"
            )
        pivot_coordinates.append(pivot)
        images = [
            _quotient_one_unit_row(image, relation, pivot)
            for image in images
        ]
        current_relations = [
            _quotient_one_unit_row(other, relation, pivot)
            for other in current_relations
        ]

    quotient_rank = AMBIENT_CHARACTER_RANK - len(fixed)
    projection = tuple(images)
    if (
        any(len(row) != quotient_rank for row in projection)
        or _rank_over_q(projection) != quotient_rank
    ):
        raise KrennResidualGradingError(
            "integral quotient projection lost rank"
        )
    zero = (0,) * quotient_rank
    if any(
        tuple(
            sum(
                character[source] * projection[source][target]
                for source in range(AMBIENT_CHARACTER_RANK)
            )
            for target in range(quotient_rank)
        ) != zero
        for character in fixed
    ):
        raise KrennResidualGradingError(
            "integral quotient did not kill every fixed character"
        )

    section_coordinates = None
    section_determinant = None
    for selected in combinations(
        range(AMBIENT_CHARACTER_RANK), quotient_rank
    ):
        determinant = _determinant(
            tuple(projection[index] for index in selected)
        )
        if abs(determinant) == 1:
            section_coordinates = selected
            section_determinant = determinant
            break
    if section_coordinates is None:
        raise KrennResidualGradingError(
            "integral quotient projection has no unimodular section minor"
        )

    return {
        "projection": projection,
        "rank": quotient_rank,
        "smith_diagonal": smith,
        "unit_pivot_coordinates_in_successive_quotients": tuple(
            pivot_coordinates
        ),
        "section_source_coordinates": section_coordinates,
        "section_minor_determinant": section_determinant,
    }


def _project_character(
    character: Sequence[int],
    projection: Sequence[Sequence[int]],
) -> Character:
    character = tuple(map(int, character))
    if (
        len(character) != AMBIENT_CHARACTER_RANK
        or len(projection) != AMBIENT_CHARACTER_RANK
    ):
        raise KrennResidualGradingError(
            "a character cannot be applied to the quotient projection"
        )
    quotient_rank = len(projection[0])
    return tuple(
        sum(
            character[source] * projection[source][target]
            for source in range(AMBIENT_CHARACTER_RANK)
        )
        for target in range(quotient_rank)
    )


def _monomial_character(
    monomial: Sequence[int],
    variable_characters: Sequence[Character],
) -> Character:
    rank = len(variable_characters[0])
    return tuple(
        sum(variable_characters[variable][coordinate] for variable in monomial)
        for coordinate in range(rank)
    )


def _character_digest(
    variable_characters: Sequence[Character],
    generator_characters: Sequence[Character],
) -> str:
    digest = hashlib.sha256()
    for label, characters in (
        ("variables", variable_characters),
        ("generators", generator_characters),
    ):
        digest.update(f"{label}\n".encode("ascii"))
        for character in characters:
            digest.update(",".join(map(str, character)).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def repair_residual_grading(
    representative_ambient_monomial: Sequence[int],
) -> dict:
    """Build and exactly replay one repair chart's residual grading."""

    representative = tuple(sorted(map(
        int, representative_ambient_monomial
    )))
    if representative not in REPAIR_MONOMIAL_REPRESENTATIVES:
        raise KrennResidualGradingError(
            "repair monomial is not one of the four exact representatives"
        )
    chart = repair_monomial_chart(representative)
    original_characters = residual_torus_characters(6)
    fixed_characters = tuple(
        original_characters[variable]
        for variable in chart.fixed_original_chart_variables
    )
    repair_characters = tuple(
        original_characters[
            chart.retained_original_chart_variables[variable]
        ]
        for variable in range(
            len(chart.retained_original_chart_variables)
        )
    ) + (
        tuple(
            -entry
            for entry in original_characters[
                chart.localized_original_chart_variable
            ]
        ),
    )

    full_repair_sum = tuple(
        sum(
            original_characters[variable][coordinate]
            for variable in (
                *chart.fixed_original_chart_variables,
                chart.localized_original_chart_variable,
            )
        )
        for coordinate in range(AMBIENT_CHARACTER_RANK)
    )
    if full_repair_sum != (0,) * AMBIENT_CHARACTER_RANK:
        raise KrennResidualGradingError(
            "a repair monomial is not residual-torus invariant"
        )

    quotient = _integral_quotient(fixed_characters)
    projection = quotient["projection"]
    variable_characters = tuple(
        _project_character(character, projection)
        for character in repair_characters
    )
    residual_rank = quotient["rank"]
    if (
        len(variable_characters) != chart.variable_count
        or _rank_over_q(variable_characters) != residual_rank
    ):
        raise KrennResidualGradingError(
            "repair variables do not span the residual character lattice"
        )

    generator_characters = []
    for polynomial in chart.generators:
        term_characters = {
            _monomial_character(monomial, variable_characters)
            for _coefficient, monomial in polynomial.terms
        }
        if len(term_characters) != 1:
            raise KrennResidualGradingError(
                "a repair-chart generator is not residual homogeneous"
            )
        generator_characters.append(next(iter(term_characters)))
    generator_characters = tuple(generator_characters)

    block_sizes = Counter(generator_characters)
    multiplicity_histogram = tuple(sorted(Counter(
        block_sizes.values()
    ).items()))
    character_sha256 = _character_digest(
        variable_characters, generator_characters
    )
    zero = (0,) * residual_rank
    expected = EXPECTED_REPAIR_ROWS[representative]
    observed = {
        "fixed_rank": len(fixed_characters),
        "residual_rank": residual_rank,
        "variable_count": chart.variable_count,
        "zero_character_variables": variable_characters.count(zero),
        "character_blocks": len(block_sizes),
        "zero_character_generators": block_sizes[zero],
        "multiplicity_histogram": multiplicity_histogram,
        "character_sha256": character_sha256,
    }
    if observed != expected:
        raise KrennResidualGradingError(
            "repair residual-grading census changed"
        )

    return {
        "representative_ambient_monomial": list(representative),
        "fixed_original_chart_variables": list(
            chart.fixed_original_chart_variables
        ),
        "localized_original_chart_variable":
            chart.localized_original_chart_variable,
        "repair_monomial_character_sum": list(full_repair_sum),
        "integral_character_quotient": {
            "ambient_rank": AMBIENT_CHARACTER_RANK,
            "fixed_factor_rank": len(fixed_characters),
            "residual_rank": residual_rank,
            "smith_diagonal": list(quotient["smith_diagonal"]),
            "projection_rows": [
                list(row) for row in projection
            ],
            "unit_pivot_coordinates_in_successive_quotients": list(
                quotient[
                    "unit_pivot_coordinates_in_successive_quotients"
                ]
            ),
            "section_source_coordinates": list(
                quotient["section_source_coordinates"]
            ),
            "section_minor_determinant":
                quotient["section_minor_determinant"],
            "fixed_character_lattice_saturated": True,
            "quotient_is_split_over_Z": True,
        },
        "variables": {
            "count": chart.variable_count,
            "zero_character_variables": variable_characters.count(zero),
            "branch_inverse_character_is_zero_after_quotient": (
                variable_characters[-1] == zero
            ),
        },
        "generators": {
            "count": len(chart.generators),
            "all_terms_residual_homogeneous": True,
            "character_blocks": len(block_sizes),
            "zero_character_generators": block_sizes[zero],
            "block_multiplicity_histogram": [
                {
                    "multiplicity": multiplicity,
                    "blocks": blocks,
                }
                for multiplicity, blocks in multiplicity_histogram
            ],
            "variable_and_generator_character_sha256":
                character_sha256,
        },
        "valuation_lineality": {
            "dimension": residual_rank,
            "map": (
                "lambda -> (<lambda, chi_j>)_j on repair variables"
            ),
            "initial_forms_constant_along_this_subspace": True,
            "quotienting_removes_gauge_redundancy": True,
            "tropical_valuation_vectors_enumerated": False,
            "tropical_or_grobner_cones_enumerated": False,
            "reason": (
                "characters label a torus action; valuation directions are "
                "cocharacters paired with them, and genuine cones live "
                "only after quotienting by this lineality"
            ),
        },
        "claim_boundary": {
            "chart_unit_or_proper_status_decided": False,
            "finite_affine_GHZ_membership_decided": False,
            "tropical_prevariety_computed": False,
            "tropical_variety_computed": False,
            "grading_is_a_tropical_decomposition": False,
        },
    }


def _build_residual_grading_audit() -> dict:
    rows = [
        repair_residual_grading(representative)
        for representative in REPAIR_MONOMIAL_REPRESENTATIVES
    ]
    return {
        "schema": RESIDUAL_GRADING_SCHEMA,
        "natural_seed": [0, 4, 8],
        "ambient_residual_character_lattice": "Z^9",
        "repair_charts": rows,
        "exact_checks": {
            "all_four_repair_representatives_covered": len(rows) == 4,
            "all_fixed_character_quotients_split_over_Z": all(
                row["integral_character_quotient"][
                    "quotient_is_split_over_Z"
                ]
                for row in rows
            ),
            "all_730_generator_sets_homogeneous": all(
                row["generators"]["count"] == 730
                and row["generators"][
                    "all_terms_residual_homogeneous"
                ]
                for row in rows
            ),
        },
        "interpretation": {
            "character_grading": (
                "exact algebraic-torus weights used to block polynomial "
                "linear algebra"
            ),
            "tropical_valuation": (
                "orders of a one-parameter or Puiseux family"
            ),
            "relationship": (
                "cocharacter pairing with character weights supplies "
                "lineality, not the remaining tropical cones"
            ),
        },
        "claim_boundary": {
            "chart_unit_or_proper_status_decided": False,
            "finite_counterexample_found": False,
            "global_nonexistence_proved": False,
            "tropical_cones_enumerated": False,
        },
    }


def residual_grading_audit() -> dict:
    """Return a fresh JSON-ready exact audit of all four repair charts."""

    return json.loads(json.dumps(_build_residual_grading_audit()))


def verify_residual_grading_audit(payload: Mapping) -> dict:
    """Rebuild the complete audit and reject any altered datum or claim."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennResidualGradingError(
            "residual-grading audit is not strict JSON"
        ) from error
    expected = _build_residual_grading_audit()
    if not strict_json_equal(normalized, expected):
        raise KrennResidualGradingError(
            "residual-grading audit failed exact replay"
        )
    return normalized


__all__ = [
    "KrennResidualGradingError",
    "REPAIR_MONOMIAL_REPRESENTATIVES",
    "RESIDUAL_GRADING_SCHEMA",
    "repair_residual_grading",
    "residual_grading_audit",
    "verify_residual_grading_audit",
]
