r"""Exact chart reductions for endpoint-only ``n=6,d=3`` components.

The known Laurent path is a one-parameter color-diagonal gauge orbit.  That
observation explains its pole, but does not rule out a different component
living only over the affine endpoint.  This module isolates that loophole.

There are two exact reductions.

* Fixing the victim ``002121`` leaves a group of order 48.  Its action on
  the 3,375 ordered choices of one pure perfect matching per color has 104
  orbits.  On 103 exact nine-coordinate tori a nonvictim equation has one
  active monomial and is impossible.  The sole survivor is the four-member
  natural orbit.  Its localized coordinate ring is a six-dimensional
  Laurent torus and its victim monomial is a unit, so it has no vertical
  point with victim amplitude zero.
* Every finite GHZ witness contains one such nine-coordinate pure seed.
  A deterministic missing-set closure therefore gives an unconditional
  support lower bound without assuming that any natural coordinate is
  retained.  The routine replay checks all supports through size 11; a
  separately gated replay can push the same exact argument farther.

Neither reduction classifies higher-support vertical components.  The
global affine-membership status remains undecided.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import argparse
import heapq
from itertools import combinations, product
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
)
from experiments.krenn_quantum_graph.matching_circuit_structure import (
    victim_stabilizer_actions,
)
from experiments.krenn_quantum_graph.n6_deformation import (
    natural_repair_direction,
)
from experiments.krenn_quantum_graph.system import (
    Monomial,
    coloring_from_index,
    generate_sparse_system,
    perfect_matchings,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    EXPECTED_REPRESENTATIVES_AND_SIZES,
    transport_ordered_seed,
)
N = 6
D = 3
AMBIENT_VARIABLES = 135
COLOR_GAUGE_PARAMETERS = 15
VICTIM_EQUATION = N6_D3_SEED_DEFECT_EQUATION
VICTIM_COLORING = (0, 0, 2, 1, 2, 1)
PURE_EQUATIONS = (0, 364, 728)
NATURAL_SEED = (0, 4, 8)

VERTICAL_COMPONENT_SCHEMA = "krenn-n6-d3-vertical-component-charts-v1"
GAUGE_ORBIT_SCHEMA = "krenn-n6-d3-color-gauge-pole-orbit-v1"
FIXED_VICTIM_CHART_SCHEMA = "krenn-n6-d3-fixed-victim-seed-charts-v1"
SUPPORT_CLOSURE_SCHEMA = "krenn-n6-d3-unconditional-support-closure-v1"

EXPECTED_FIXED_VICTIM_SEED_ORBITS = 104
EXPECTED_FIXED_VICTIM_ORBIT_HISTOGRAM = {
    1: 1,
    4: 2,
    6: 5,
    8: 4,
    12: 8,
    16: 1,
    24: 33,
    48: 50,
}
EXPECTED_FIXED_VICTIM_SURVIVOR_ORBIT = (
    (0, 4, 8),
    (0, 9, 13),
    (2, 4, 13),
    (2, 9, 8),
)
EXPECTED_SMALL_SUPPORT_CASES = (8, 1_008, 63_000)
EXPECTED_LONG_CLOSURE_NODES_AT_TOTAL_20 = (
    370_509,
    417_447,
    413_599,
    380_808,
    420_041,
    369_988,
    391_617,
    357_622,
)
EXPECTED_SUPPORT21_CLOSURE_NODES = (
    1_570_130,
    1_768_215,
    1_752_008,
    1_529_410,
    1_768_668,
    1_488_355,
    1_632_189,
    1_483_294,
)

SUPPORT21_TERMINAL_ADDITIONS = {
    (0, 0, 4): (
        (1, 3, 18, 19, 21, 22, 45, 46, 48, 49, 82, 84),
        (1, 3, 27, 28, 30, 31, 72, 73, 75, 76, 127, 129),
        (82, 84, 99, 100, 102, 103, 108, 109, 111, 112, 127, 129),
    ),
    (0, 1, 5): (
        (1, 3, 18, 21, 28, 31, 45, 46, 48, 49, 84, 91),
        (1, 3, 36, 37, 39, 40, 55, 58, 63, 66, 120, 127),
        (82, 84, 85, 90, 91, 93, 117, 118, 120, 127, 129, 130),
    ),
    (0, 4, 8): (
        (6, 18, 20, 24, 29, 35, 45, 47, 83, 87, 89, 92),
        (3, 9, 10, 12, 37, 40, 54, 55, 82, 84, 85, 118),
        (68, 70, 71, 76, 77, 79, 97, 106, 107, 112, 113, 122),
        (14, 16, 17, 22, 23, 25, 95, 103, 106, 113, 116, 124),
        (1, 36, 37, 55, 58, 63, 64, 66, 120, 127, 129, 130),
        (2, 27, 29, 47, 53, 72, 74, 78, 96, 128, 132, 134),
    ),
}

# ``pure_equation=None`` means an odd mixed-only circuit.  Otherwise the
# mixed relation forces the first and second active monomials in that pure
# equation to be negatives, so their sum is zero instead of one.
SUPPORT21_TERMINAL_RELATIONS = {
    ((0, 0, 4), 0): ("pure-cancellation", ((4, -1),), 0),
    ((0, 0, 4), 1): ("pure-cancellation", ((36, -1),), 0),
    ((0, 0, 4), 2): (
        "pure-cancellation",
        ((1, -1), (9, -1), (10, 1)),
        0,
    ),
    ((0, 1, 5), 0): (
        "odd-mixed-circuit",
        ((243, 1), (256, -1), (410, 1)),
        None,
    ),
    ((0, 1, 5), 1): (
        "odd-mixed-circuit",
        ((1, -1), (40, 1), (550, 1)),
        None,
    ),
    ((0, 1, 5), 2): (
        "pure-cancellation",
        ((3, -1), (9, -1), (12, 1)),
        0,
    ),
    ((0, 4, 8), 0): (
        "odd-mixed-circuit",
        ((16, -1), (18, 1), (188, 1)),
        None,
    ),
    ((0, 4, 8), 1): (
        "odd-mixed-circuit",
        ((270, 1), (304, -1), (355, 1)),
        None,
    ),
    ((0, 4, 8), 2): (
        "odd-mixed-circuit",
        ((70, 1), (367, -1), (646, 1)),
        None,
    ),
    ((0, 4, 8), 3): (
        "odd-mixed-circuit",
        ((70, 1), (391, -1), (476, 1)),
        None,
    ),
    ((0, 4, 8), 4): (
        "odd-mixed-circuit",
        ((1, -1), (70, 1), (280, 1)),
        None,
    ),
    ((0, 4, 8), 5): (
        "odd-mixed-circuit",
        ((2, -1), (64, 1), (560, 1)),
        None,
    ),
}


class KrennVerticalComponentError(RuntimeError):
    """A gauge, chart, singleton, or support-closure replay failed."""


@lru_cache(maxsize=1)
def _system():
    return generate_sparse_system(N, D)


def _canonical_seed(seed: Sequence[int]) -> tuple[int, int, int]:
    try:
        result = tuple(map(int, seed))
    except (TypeError, ValueError) as error:
        raise KrennVerticalComponentError(
            "a seed must be three matching indices"
        ) from error
    if len(result) != D or any(index < 0 or index >= 15 for index in result):
        raise KrennVerticalComponentError(
            "a seed must contain three canonical matching indices"
        )
    return result


def _canonical_support(support: Sequence[int]) -> tuple[int, ...]:
    try:
        result = tuple(map(int, support))
    except (TypeError, ValueError) as error:
        raise KrennVerticalComponentError(
            "a support must be an integer sequence"
        ) from error
    if (
        result != tuple(sorted(set(result)))
        or any(index < 0 or index >= AMBIENT_VARIABLES for index in result)
    ):
        raise KrennVerticalComponentError(
            "a support must be sorted, unique, and canonical"
        )
    return result


def _mask(values: Sequence[int]) -> int:
    result = 0
    for raw_value in values:
        value = int(raw_value)
        if not 0 <= value < AMBIENT_VARIABLES:
            raise KrennVerticalComponentError(
                "a bit-mask coordinate is outside the system"
            )
        result |= 1 << value
    return result


def _indices(mask: int) -> tuple[int, ...]:
    mask = int(mask)
    if mask < 0:
        raise KrennVerticalComponentError("a support mask cannot be negative")
    result = []
    while mask:
        least = mask & -mask
        result.append(least.bit_length() - 1)
        mask -= least
    return tuple(result)


def seed_support(seed: Sequence[int]) -> tuple[int, ...]:
    """Return the nine nonzero diagonal coordinates of an ordered seed."""

    seed = _canonical_seed(seed)
    matchings = perfect_matchings(N)
    support = tuple(sorted(
        variable_index(N, D, i, j, color, color)
        for color, matching_index in enumerate(seed)
        for i, j in matchings[matching_index]
    ))
    if (
        len(support) != 9
        or support != tuple(sorted(set(support)))
        or any(
            variable_key(N, D, index)[2]
            != variable_key(N, D, index)[3]
            for index in support
        )
    ):
        raise KrennVerticalComponentError(
            "an ordered seed did not produce nine diagonal coordinates"
        )
    return support


def _parameter_position(vertex: int, color: int) -> int:
    vertex = int(vertex)
    color = int(color)
    if not 0 <= vertex < N - 1 or not 0 <= color < D:
        raise KrennVerticalComponentError(
            "a gauge parameter needs vertex 0..4 and color 0..2"
        )
    return vertex * D + color


def _endpoint_character(vertex: int, color: int) -> tuple[int, ...]:
    """Express ``theta_(vertex,color)`` after eliminating vertex five."""

    vertex = int(vertex)
    color = int(color)
    if not 0 <= vertex < N or not 0 <= color < D:
        raise KrennVerticalComponentError(
            "an endpoint character is outside the n=6,d=3 system"
        )
    result = [0] * COLOR_GAUGE_PARAMETERS
    if vertex < N - 1:
        result[_parameter_position(vertex, color)] = 1
    else:
        for other_vertex in range(N - 1):
            result[_parameter_position(other_vertex, color)] = -1
    return tuple(result)


def _add_characters(*rows: Sequence[int]) -> tuple[int, ...]:
    if not rows:
        return (0,) * COLOR_GAUGE_PARAMETERS
    normalized = tuple(tuple(map(int, row)) for row in rows)
    if any(len(row) != COLOR_GAUGE_PARAMETERS for row in normalized):
        raise KrennVerticalComponentError(
            "a color-gauge character has the wrong dimension"
        )
    return tuple(
        sum(row[column] for row in normalized)
        for column in range(COLOR_GAUGE_PARAMETERS)
    )


@lru_cache(maxsize=1)
def color_diagonal_exponent_matrix() -> tuple[tuple[int, ...], ...]:
    r"""Return the exact ``135 x 15`` direct-GHZ gauge exponent matrix.

    Parameters are ``theta_(i,a)`` for ``i<5``.  The constraint
    ``sum_i theta_(i,a)=0`` eliminates ``theta_(5,a)`` separately for each
    color, so all three pure GHZ coefficients are preserved.
    """

    rows = []
    for index in range(variable_count(N, D)):
        i, j, a, b = variable_key(N, D, index)
        rows.append(
            _add_characters(
                _endpoint_character(i, a),
                _endpoint_character(j, b),
            )
        )
    result = tuple(rows)
    if (
        len(result) != AMBIENT_VARIABLES
        or any(len(row) != COLOR_GAUGE_PARAMETERS for row in result)
    ):
        raise KrennVerticalComponentError(
            "the direct-GHZ color-gauge matrix changed shape"
        )
    return result


def coloring_character(coloring: Sequence[int]) -> tuple[int, ...]:
    """Return the target character ``sum_i theta_(i,coloring_i)``."""

    coloring = tuple(map(int, coloring))
    if len(coloring) != N or any(color not in range(D) for color in coloring):
        raise KrennVerticalComponentError(
            "a target character needs a coloring in range(3)^6"
        )
    return _add_characters(
        *(
            _endpoint_character(vertex, color)
            for vertex, color in enumerate(coloring)
        )
    )


def monomial_character(monomial: Sequence[int]) -> tuple[int, ...]:
    """Return the source color-gauge character of a matching monomial."""

    matrix = color_diagonal_exponent_matrix()
    indices = tuple(map(int, monomial))
    if any(index < 0 or index >= len(matrix) for index in indices):
        raise KrennVerticalComponentError(
            "a monomial character contains an invalid variable"
        )
    return _add_characters(*(matrix[index] for index in indices))


def _rank_over_q(matrix: Sequence[Sequence[int]]) -> int:
    """Exact fraction-free rank for the small character matrices."""

    rows = [list(map(int, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennVerticalComponentError("an exact matrix is ragged")
    rank = 0
    previous = 1
    for column in range(width):
        selected = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        rows[rank], rows[selected] = rows[selected], rows[rank]
        pivot = rows[rank][column]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            factor = rows[row][column]
            for other_column in range(column + 1, width):
                numerator = (
                    rows[row][other_column] * pivot
                    - factor * rows[rank][other_column]
                )
                if numerator % previous:
                    # Fall back to a rational implementation only if a
                    # non-Bareiss elimination order is encountered.
                    from fractions import Fraction

                    rational = [
                        list(map(Fraction, source)) for source in matrix
                    ]
                    rational_rank = 0
                    for rational_column in range(width):
                        rational_selected = next(
                            (
                                entry
                                for entry in range(
                                    rational_rank, len(rational)
                                )
                                if rational[entry][rational_column]
                            ),
                            None,
                        )
                        if rational_selected is None:
                            continue
                        rational[rational_rank], rational[rational_selected] = (
                            rational[rational_selected],
                            rational[rational_rank],
                        )
                        rational_pivot = rational[rational_rank][
                            rational_column
                        ]
                        rational[rational_rank] = [
                            value / rational_pivot
                            for value in rational[rational_rank]
                        ]
                        for entry in range(len(rational)):
                            if (
                                entry == rational_rank
                                or not rational[entry][rational_column]
                            ):
                                continue
                            multiplier = rational[entry][rational_column]
                            rational[entry] = [
                                left - multiplier * right
                                for left, right in zip(
                                    rational[entry],
                                    rational[rational_rank],
                                    strict=True,
                                )
                            ]
                        rational_rank += 1
                    return rational_rank
                rows[row][other_column] = numerator // previous
            rows[row][column] = 0
        previous = pivot
        rank += 1
        if rank == len(rows):
            break
    return rank


@lru_cache(maxsize=1)
def natural_color_gauge_orbit_audit() -> Mapping[str, object]:
    """Certify that the known Laurent path is a color-gauge orbit."""

    gamma = [0] * COLOR_GAUGE_PARAMETERS
    gamma[_parameter_position(0, 0)] = 1
    gamma[_parameter_position(2, 0)] = -1
    gamma = tuple(gamma)
    support = seed_support(NATURAL_SEED)
    matrix = color_diagonal_exponent_matrix()
    valuations = tuple(
        sum(
            matrix[index][column] * gamma[column]
            for column in range(COLOR_GAUGE_PARAMETERS)
        )
        for index in support
    )
    expected_valuations = tuple(
        1
        if index == variable_index(N, D, 0, 1, 0, 0)
        else -1
        if index == variable_index(N, D, 2, 3, 0, 0)
        else 0
        for index in support
    )
    tangent = [0] * AMBIENT_VARIABLES
    for index, valuation in zip(support, valuations, strict=True):
        tangent[index] = valuation
    repair = natural_repair_direction()
    victim_character = coloring_character(VICTIM_COLORING)
    victim_gamma = sum(
        left * right
        for left, right in zip(victim_character, gamma, strict=True)
    )
    q_character = monomial_character(POLE_INVARIANT_INDICES)
    q_gamma = sum(
        left * right
        for left, right in zip(q_character, gamma, strict=True)
    )

    system = _system()
    natural_support_set = set(support)
    pure_monomials = tuple(
        next(
            monomial
            for monomial in system.equation_monomials(equation)
            if set(monomial) <= natural_support_set
        )
        for equation in PURE_EQUATIONS
    )
    victim_active = tuple(
        monomial
        for monomial in system.equation_monomials(VICTIM_EQUATION)
        if set(monomial) <= natural_support_set
    )
    if len(victim_active) != 1:
        raise KrennVerticalComponentError(
            "the natural victim monomial is no longer unique"
        )
    left = Counter((*victim_active[0], *POLE_INVARIANT_INDICES))
    right = Counter(
        variable
        for monomial in pure_monomials
        for variable in monomial
    )
    support_rank = _rank_over_q(tuple(matrix[index] for index in support))
    checks = {
        "natural_support_has_nine_coordinates": len(support) == 9,
        "direct_GHZ_color_gauge_parameter_dimension_15": (
            len(gamma) == 15
        ),
        "natural_support_effective_gauge_rank_6": support_rank == 6,
        "gamma_reproduces_Laurent_weight_orders": (
            valuations == expected_valuations
        ),
        "negative_gamma_tangent_is_repair_direction": (
            tuple(-value for value in tangent) == repair
        ),
        "victim_character_on_gamma_is_plus_one": victim_gamma == 1,
        "Q_character_on_gamma_is_minus_one": q_gamma == -1,
        "Q_character_is_negative_victim_character": (
            q_character == tuple(-value for value in victim_character)
        ),
        "victim_times_Q_equals_three_pure_monomials": left == right,
    }
    if not all(checks.values()):
        raise KrennVerticalComponentError(
            "the natural color-gauge orbit audit failed"
        )
    return {
        "schema": GAUGE_ORBIT_SCHEMA,
        "parameters": {
            "n": N,
            "d": D,
            "independent_color_gauge_parameters": COLOR_GAUGE_PARAMETERS,
            "constraints": [
                "sum_i theta_(i,0)=0",
                "sum_i theta_(i,1)=0",
                "sum_i theta_(i,2)=0",
            ],
        },
        "natural_support": list(support),
        "natural_support_coordinates": [
            list(variable_key(N, D, index)) for index in support
        ],
        "effective_gauge_rank_on_natural_support": support_rank,
        "gamma": list(gamma),
        "natural_support_valuations": list(valuations),
        "victim_character": list(victim_character),
        "Q_character": list(q_character),
        "victim_character_on_gamma": victim_gamma,
        "Q_character_on_gamma": q_gamma,
        "identity": "F_victim*Q=M_000*M_111*M_222",
        "interpretation": (
            "the known Laurent path is a one-parameter "
            "color-diagonal orbit in the GHZ-plus-victim moving-line "
            "incidence"
        ),
        "exact_checks": checks,
        "claim_boundary": {
            "known_path_is_color_diagonal_orbit_in_moving_line_incidence": (
                True
            ),
            "Q_is_invariant_under_full_direct_GHZ_color_gauge": False,
            "Q_is_inverse_character_to_fixed_victim": True,
            "universal_component_pole_law_proved": False,
            "global_affine_GHZ_membership_decided": False,
        },
    }


@lru_cache(maxsize=1)
def fixed_victim_seed_orbits(
) -> tuple[tuple[tuple[int, int, int], tuple[tuple[int, int, int], ...]], ...]:
    """Return all seed orbits under the stabilizer of ``002121``."""

    actions = victim_stabilizer_actions()
    unseen = set(product(range(15), repeat=D))
    rows = []
    while unseen:
        seed = min(unseen)
        orbit = tuple(sorted({
            transport_ordered_seed(seed, action[0], action[1])
            for action in actions
        }))
        if not orbit or any(member not in unseen for member in orbit):
            raise KrennVerticalComponentError(
                "fixed-victim seed orbits failed to partition the seeds"
            )
        unseen.difference_update(orbit)
        rows.append((orbit[0], orbit))
    result = tuple(sorted(rows))
    histogram = Counter(len(orbit) for _representative, orbit in result)
    if (
        len(result) != EXPECTED_FIXED_VICTIM_SEED_ORBITS
        or sum(len(orbit) for _representative, orbit in result) != 3_375
        or dict(sorted(histogram.items()))
        != EXPECTED_FIXED_VICTIM_ORBIT_HISTOGRAM
    ):
        raise KrennVerticalComponentError(
            "the fixed-victim 104-orbit census changed"
        )
    return result


def _active_matching_indices(
    support: Sequence[int], equation: int
) -> tuple[int, ...]:
    support_mask = _mask(_canonical_support(support))
    return tuple(
        matching_index
        for matching_index, monomial_mask in enumerate(
            _equation_monomial_masks()[int(equation)]
        )
        if monomial_mask & ~support_mask == 0
    )


@lru_cache(maxsize=1)
def _equation_monomial_masks() -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(_mask(monomial) for monomial in _system().equation_monomials(
            equation
        ))
        for equation in range(D**N)
    )


def _first_mixed_singleton_mask(
    support_mask: int,
    *,
    exclude_victim: bool,
) -> tuple[int, int] | None:
    for equation, rhs in enumerate(_system().rhs_values):
        if rhs or (exclude_victim and equation == VICTIM_EQUATION):
            continue
        active = -1
        count = 0
        for matching_index, monomial_mask in enumerate(
            _equation_monomial_masks()[equation]
        ):
            if monomial_mask & ~support_mask == 0:
                active = matching_index
                count += 1
                if count > 1:
                    break
        if count == 1:
            return equation, active
    return None


def _first_mixed_singleton(
    support: Sequence[int],
    *,
    exclude_victim: bool,
) -> tuple[int, int] | None:
    return _first_mixed_singleton_mask(
        _mask(_canonical_support(support)),
        exclude_victim=exclude_victim,
    )


@lru_cache(maxsize=1)
def fixed_victim_nine_coordinate_audit() -> Mapping[str, object]:
    """Classify all exact nine-coordinate strata on the moving line."""

    rows = fixed_victim_seed_orbits()
    survivor_rows = []
    singleton_receipts = []
    for representative, orbit in rows:
        singleton = _first_mixed_singleton(
            seed_support(representative),
            exclude_victim=True,
        )
        if singleton is None:
            survivor_rows.append((representative, orbit))
        else:
            singleton_receipts.append(
                (
                    representative,
                    len(orbit),
                    singleton[0],
                    singleton[1],
                )
            )
    if (
        len(survivor_rows) != 1
        or survivor_rows[0][0] != NATURAL_SEED
        or survivor_rows[0][1] != EXPECTED_FIXED_VICTIM_SURVIVOR_ORBIT
        or len(singleton_receipts) != 103
    ):
        raise KrennVerticalComponentError(
            "the fixed-victim singleton chart reduction changed"
        )

    support = seed_support(NATURAL_SEED)
    active_table = tuple(
        (
            equation,
            _active_matching_indices(support, equation),
        )
        for equation in range(D**N)
        if _active_matching_indices(support, equation)
    )
    expected_equations = (*PURE_EQUATIONS, VICTIM_EQUATION)
    if (
        tuple(equation for equation, _active in active_table)
        != tuple(sorted(expected_equations))
        or any(len(active) != 1 for _equation, active in active_table)
    ):
        raise KrennVerticalComponentError(
            "the natural restricted active table changed"
        )
    pure_exponent_rows = []
    positions = {
        variable: position for position, variable in enumerate(support)
    }
    for equation in PURE_EQUATIONS:
        monomial = _system().equation_monomials(equation)[
            _active_matching_indices(support, equation)[0]
        ]
        row = [0] * len(support)
        for variable in monomial:
            row[positions[variable]] += 1
        pure_exponent_rows.append(tuple(row))
    pure_rank = _rank_over_q(pure_exponent_rows)
    victim_monomial = _system().equation_monomials(VICTIM_EQUATION)[
        _active_matching_indices(support, VICTIM_EQUATION)[0]
    ]
    victim_is_unit = set(victim_monomial) <= set(support)
    checks = {
        "fixed_victim_stabilizer_order_48": (
            len(victim_stabilizer_actions()) == 48
        ),
        "ordered_seed_charts_3375": (
            sum(len(orbit) for _representative, orbit in rows) == 3_375
        ),
        "fixed_victim_seed_orbits_104": len(rows) == 104,
        "nonvictim_singleton_kills_103_orbits": (
            len(singleton_receipts) == 103
        ),
        "sole_survivor_is_natural_orbit_size_4": (
            survivor_rows[0][0] == NATURAL_SEED
            and len(survivor_rows[0][1]) == 4
        ),
        "restricted_active_equations_are_three_pure_plus_victim": (
            len(active_table) == 4
        ),
        "three_pure_exponent_relations_have_rank_3": pure_rank == 3,
        "localized_natural_ring_is_torus_dimension_6": (
            len(support) - pure_rank == 6
        ),
        "victim_monomial_is_a_Laurent_unit": victim_is_unit,
        "no_vertical_victim_zero_point_on_natural_stratum": victim_is_unit,
    }
    if not all(checks.values()):
        raise KrennVerticalComponentError(
            "the fixed-victim nine-coordinate audit failed"
        )
    return {
        "schema": FIXED_VICTIM_CHART_SCHEMA,
        "victim_equation": VICTIM_EQUATION,
        "victim_coloring": list(VICTIM_COLORING),
        "stabilizer_order": 48,
        "ordered_seed_charts": 3_375,
        "seed_orbits": len(rows),
        "orbit_size_histogram": {
            str(size): count
            for size, count in sorted(
                EXPECTED_FIXED_VICTIM_ORBIT_HISTOGRAM.items()
            )
        },
        "orbits_killed_by_nonvictim_singleton": 103,
        "surviving_orbit": {
            "representative": list(NATURAL_SEED),
            "members": [
                list(seed)
                for seed in EXPECTED_FIXED_VICTIM_SURVIVOR_ORBIT
            ],
            "size": 4,
        },
        "natural_restricted_stratum": {
            "support": list(support),
            "active_equations": [
                {
                    "equation": equation,
                    "coloring": list(coloring_from_index(N, D, equation)),
                    "matching_indices": list(active),
                }
                for equation, active in active_table
            ],
            "localized_coordinate_ring": (
                "Q[x_1^+-,...,x_9^+-]/(M0-1,M1-1,M2-1)"
            ),
            "pure_exponent_rank": pure_rank,
            "dimension": len(support) - pure_rank,
            "victim_is_unit": victim_is_unit,
        },
        "singleton_receipt_count": len(singleton_receipts),
        "exact_checks": checks,
        "claim_boundary": {
            "all_nine_coordinate_fixed_victim_strata_classified": True,
            "vertical_point_on_those_strata_excluded": True,
            "higher_support_fixed_victim_strata_classified": False,
            "all_vertical_components_excluded": False,
            "global_affine_GHZ_membership_decided": False,
        },
    }


@lru_cache(maxsize=None)
def _mixed_missing_rows(
    seed: tuple[int, int, int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    base = _mask(seed_support(seed))
    return tuple(
        (
            equation,
            tuple(
                _mask(monomial) & ~base
                for monomial in _system().equation_monomials(equation)
            ),
        )
        for equation, rhs in enumerate(_system().rhs_values)
        if not rhs
    )


@dataclass(frozen=True)
class RootSupportClosure:
    """One exhaustive missing-set closure below a total support cap."""

    seed: tuple[int, int, int]
    max_total_support: int
    nodes_examined: int
    supports_discovered: int
    discovered_by_addition_size: tuple[int, ...]
    singleton_free_supports: tuple[tuple[int, ...], ...]
    termination: str

    def __post_init__(self) -> None:
        seed = _canonical_seed(self.seed)
        object.__setattr__(self, "seed", seed)
        if (
            not 9 <= self.max_total_support <= AMBIENT_VARIABLES
            or self.nodes_examined < 1
            or self.supports_discovered < self.nodes_examined
            or sum(self.discovered_by_addition_size)
            != self.supports_discovered
            or self.termination not in ("frontier-exhausted", "node-cap-reached")
            or any(
                len(support) < 9
                or len(support) > self.max_total_support
                for support in self.singleton_free_supports
            )
        ):
            raise KrennVerticalComponentError(
                "a root support-closure receipt is malformed"
            )

    @property
    def exhaustive(self) -> bool:
        return self.termination == "frontier-exhausted"

    @property
    def excludes_all_supports_through_cap(self) -> bool:
        return self.exhaustive and not self.singleton_free_supports

    def to_dict(self) -> dict:
        return {
            "seed": list(self.seed),
            "seed_support": list(seed_support(self.seed)),
            "max_total_support": self.max_total_support,
            "nodes_examined": self.nodes_examined,
            "supports_discovered": self.supports_discovered,
            "discovered_by_addition_size": list(
                self.discovered_by_addition_size
            ),
            "singleton_free_support_count": len(
                self.singleton_free_supports
            ),
            "singleton_free_supports": [
                list(support) for support in self.singleton_free_supports
            ],
            "termination": self.termination,
            "exhaustive": self.exhaustive,
            "excludes_all_supports_through_cap": (
                self.excludes_all_supports_through_cap
            ),
        }


def bounded_singleton_closure(
    seed: Sequence[int],
    *,
    max_total_support: int,
    node_cap: int | None = None,
) -> RootSupportClosure:
    """Exhaust every singleton repair below one exact support cap.

    If a mixed equation has exactly one active matching monomial, every
    singleton-free superset must contain all missing coordinates of at least
    one alternative matching.  Branching over those alternatives is
    therefore exhaustive and uses no numerical or finite-field step.
    """

    seed = _canonical_seed(seed)
    max_total_support = int(max_total_support)
    if not 9 <= max_total_support <= AMBIENT_VARIABLES:
        raise KrennVerticalComponentError(
            "a support cap must lie between 9 and 135"
        )
    if node_cap is not None:
        node_cap = int(node_cap)
        if node_cap < 1:
            raise KrennVerticalComponentError(
                "a closure node cap must be positive"
            )
    addition_cap = max_total_support - 9
    frontier: list[tuple[int, int]] = [(0, 0)]
    seen = {0}
    terminals: list[int] = []
    nodes = 0
    missing_rows = _mixed_missing_rows(seed)

    while frontier and (node_cap is None or nodes < node_cap):
        _size, additions = heapq.heappop(frontier)
        nodes += 1
        obstruction = None
        for equation, missing_sets in missing_rows:
            active = -1
            count = 0
            for matching_index, missing in enumerate(missing_sets):
                if missing & ~additions == 0:
                    active = matching_index
                    count += 1
                    if count > 1:
                        break
            if count == 1:
                obstruction = active, missing_sets
                break
        if obstruction is None:
            terminals.append(additions)
            continue
        active, missing_sets = obstruction
        for matching_index, missing in enumerate(missing_sets):
            if matching_index == active:
                continue
            child = additions | missing
            child_size = child.bit_count()
            if child_size > addition_cap or child in seen:
                continue
            seen.add(child)
            heapq.heappush(frontier, (child_size, child))

    counts = [0] * (addition_cap + 1)
    for additions in seen:
        counts[additions.bit_count()] += 1
    base = set(seed_support(seed))
    supports = tuple(
        tuple(sorted((*base, *_indices(additions))))
        for additions in sorted(terminals)
    )
    return RootSupportClosure(
        seed=seed,
        max_total_support=max_total_support,
        nodes_examined=nodes,
        supports_discovered=len(seen),
        discovered_by_addition_size=tuple(counts),
        singleton_free_supports=supports,
        termination=(
            "frontier-exhausted" if not frontier else "node-cap-reached"
        ),
    )


@dataclass(frozen=True)
class UnconditionalSupportClosure:
    """The eight-root chart cover for one total support cap."""

    max_total_support: int
    roots: tuple[RootSupportClosure, ...]
    schema: str = SUPPORT_CLOSURE_SCHEMA

    def __post_init__(self) -> None:
        expected_seeds = tuple(
            representative
            for representative, _size
            in EXPECTED_REPRESENTATIVES_AND_SIZES
        )
        if (
            self.schema != SUPPORT_CLOSURE_SCHEMA
            or tuple(root.seed for root in self.roots) != expected_seeds
            or any(
                root.max_total_support != self.max_total_support
                for root in self.roots
            )
        ):
            raise KrennVerticalComponentError(
                "the unconditional support closure is malformed"
            )

    @property
    def exhaustive(self) -> bool:
        return all(root.exhaustive for root in self.roots)

    @property
    def singleton_free_support_count(self) -> int:
        return sum(
            len(root.singleton_free_supports) for root in self.roots
        )

    @property
    def finite_support_lower_bound(self) -> int | None:
        if self.exhaustive and not self.singleton_free_support_count:
            return self.max_total_support + 1
        return None

    def to_dict(self) -> dict:
        lower_bound = self.finite_support_lower_bound
        return {
            "schema": self.schema,
            "parameters": {
                "n": N,
                "d": D,
                "full_group_seed_orbits": len(self.roots),
                "max_total_support": self.max_total_support,
                "raw_supports_without_symmetry_weight_equalities": True,
            },
            "roots": [root.to_dict() for root in self.roots],
            "totals": {
                "nodes_examined": sum(
                    root.nodes_examined for root in self.roots
                ),
                "supports_discovered": sum(
                    root.supports_discovered for root in self.roots
                ),
                "singleton_free_supports": (
                    self.singleton_free_support_count
                ),
            },
            "exhaustive": self.exhaustive,
            "finite_exact_support_lower_bound": lower_bound,
            "statement": (
                "Every finite GHZ witness contains one nonzero pure "
                "matching per color. The eight S6 x S3 seed charts and "
                "exhaustive singleton closure exclude all total supports "
                f"at most {self.max_total_support}; any finite witness "
                f"therefore has support at least {lower_bound}."
                if lower_bound is not None
                else
                "The bounded closure did not establish a support lower bound."
            ),
            "claim_boundary": {
                "natural_coordinate_retention_assumed": False,
                "symmetry_related_weights_equated": False,
                "only_singleton_necessary_condition_used": True,
                "supports_above_cap_excluded": False,
                "global_nonexistence_proved": False,
            },
        }


def certify_unconditional_support_closure(
    *,
    max_total_support: int = 11,
    node_cap_per_root: int | None = None,
) -> UnconditionalSupportClosure:
    """Run the exact chart-cover closure on all eight seed representatives."""

    roots = tuple(
        bounded_singleton_closure(
            representative,
            max_total_support=max_total_support,
            node_cap=node_cap_per_root,
        )
        for representative, _size in EXPECTED_REPRESENTATIVES_AND_SIZES
    )
    result = UnconditionalSupportClosure(
        max_total_support=int(max_total_support),
        roots=roots,
    )
    if (
        max_total_support == 20
        and node_cap_per_root is None
        and (
            tuple(root.nodes_examined for root in roots)
            != EXPECTED_LONG_CLOSURE_NODES_AT_TOTAL_20
            or result.finite_support_lower_bound != 21
        )
    ):
        raise KrennVerticalComponentError(
            "the recorded support-20 closure receipt changed"
        )
    return result


@lru_cache(maxsize=1)
def small_support_direct_census() -> Mapping[str, object]:
    """Directly replay every root plus zero, one, or two coordinates."""

    cases = []
    survivor_counts = []
    for additions in range(3):
        examined = 0
        survivors = 0
        for representative, _orbit_size in EXPECTED_REPRESENTATIVES_AND_SIZES:
            base = seed_support(representative)
            base_mask = _mask(base)
            base_set = set(base)
            outside = tuple(
                index for index in range(AMBIENT_VARIABLES)
                if index not in base_set
            )
            for extra in combinations(outside, additions):
                examined += 1
                support_mask = base_mask | _mask(extra)
                if _first_mixed_singleton_mask(
                    support_mask, exclude_victim=False
                ) is None:
                    survivors += 1
        cases.append(examined)
        survivor_counts.append(survivors)
    if (
        tuple(cases) != EXPECTED_SMALL_SUPPORT_CASES
        or tuple(survivor_counts) != (0, 0, 0)
    ):
        raise KrennVerticalComponentError(
            "the direct support-11 singleton census changed"
        )
    return {
        "addition_counts": [0, 1, 2],
        "total_support_sizes": [9, 10, 11],
        "cases_examined": cases,
        "singleton_free_cases": survivor_counts,
        "total_cases_examined": sum(cases),
        "finite_exact_support_lower_bound": 12,
        "exact": True,
        "claim_boundary": {
            "natural_coordinate_retention_assumed": False,
            "supports_of_size_12_or_more_examined": False,
            "global_nonexistence_proved": False,
        },
    }


def _active_monomials(
    support: Sequence[int], equation: int
) -> tuple[Monomial, ...]:
    support_mask = _mask(_canonical_support(support))
    return tuple(
        monomial
        for monomial, monomial_mask in zip(
            _system().equation_monomials(int(equation)),
            _equation_monomial_masks()[int(equation)],
            strict=True,
        )
        if monomial_mask & ~support_mask == 0
    )


def _exponent_difference(
    support: Sequence[int],
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, ...]:
    support = _canonical_support(support)
    positions = {
        variable: position for position, variable in enumerate(support)
    }
    result = [0] * len(support)
    for variable in left:
        result[positions[int(variable)]] += 1
    for variable in right:
        result[positions[int(variable)]] -= 1
    return tuple(result)


def _weighted_rows(
    rows: Sequence[Sequence[int]],
    coefficients: Sequence[int],
) -> tuple[int, ...]:
    rows = tuple(tuple(map(int, row)) for row in rows)
    coefficients = tuple(map(int, coefficients))
    if not rows or len(rows) != len(coefficients):
        raise KrennVerticalComponentError(
            "an exponent certificate needs matching rows and coefficients"
        )
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennVerticalComponentError(
            "an exponent certificate is ragged"
        )
    return tuple(
        sum(
            coefficient * row[column]
            for coefficient, row in zip(
                coefficients, rows, strict=True
            )
        )
        for column in range(width)
    )


@dataclass(frozen=True)
class Support21TerminalCertificate:
    """One exact algebraic obstruction on a singleton-free support."""

    seed: tuple[int, int, int]
    terminal_index: int
    additions: tuple[int, ...]
    kind: str
    mixed_relation: tuple[tuple[int, int], ...]
    pure_equation: int | None

    def __post_init__(self) -> None:
        seed = _canonical_seed(self.seed)
        additions = _canonical_support(self.additions)
        relation = tuple(
            (int(equation), int(coefficient))
            for equation, coefficient in self.mixed_relation
        )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "additions", additions)
        object.__setattr__(self, "mixed_relation", relation)
        expected_additions = SUPPORT21_TERMINAL_ADDITIONS.get(seed)
        expected_relation = SUPPORT21_TERMINAL_RELATIONS.get(
            (seed, int(self.terminal_index))
        )
        if (
            expected_additions is None
            or not 0 <= int(self.terminal_index) < len(expected_additions)
            or additions != expected_additions[int(self.terminal_index)]
            or expected_relation
            != (self.kind, relation, self.pure_equation)
            or self.kind not in (
                "odd-mixed-circuit",
                "pure-cancellation",
            )
            or not relation
            or any(not coefficient for _equation, coefficient in relation)
        ):
            raise KrennVerticalComponentError(
                "a support-21 terminal certificate is malformed"
            )
        support = self.support
        if len(support) != 21:
            raise KrennVerticalComponentError(
                "a terminal certificate does not have support 21"
            )
        active_mixed = []
        for equation, rhs in enumerate(_system().rhs_values):
            if rhs:
                continue
            active = _active_monomials(support, equation)
            if len(active) not in (0, 2):
                raise KrennVerticalComponentError(
                    "a support-21 terminal has a non-binomial mixed output"
                )
            if active:
                active_mixed.append(equation)
        if any(
            equation not in active_mixed
            for equation, _coefficient in relation
        ):
            raise KrennVerticalComponentError(
                "a terminal relation names an inactive mixed equation"
            )
        rows = []
        coefficients = []
        for equation, coefficient in relation:
            active = _active_monomials(support, equation)
            rows.append(
                _exponent_difference(
                    support, active[0], active[1]
                )
            )
            coefficients.append(coefficient)
        if self.kind == "odd-mixed-circuit":
            if (
                self.pure_equation is not None
                or any(_weighted_rows(rows, coefficients))
                or sum(coefficients) % 2 == 0
            ):
                raise KrennVerticalComponentError(
                    "an odd mixed circuit failed exact replay"
                )
        else:
            if self.pure_equation not in PURE_EQUATIONS:
                raise KrennVerticalComponentError(
                    "a pure cancellation names a nonpure equation"
                )
            pure_active = _active_monomials(
                support, int(self.pure_equation)
            )
            if len(pure_active) != 2:
                raise KrennVerticalComponentError(
                    "a pure cancellation does not have two active terms"
                )
            pure_row = _exponent_difference(
                support, pure_active[0], pure_active[1]
            )
            if (
                any(
                    _weighted_rows(
                        (*rows, pure_row),
                        (*coefficients, 1),
                    )
                )
                or sum(coefficients) % 2 == 0
            ):
                raise KrennVerticalComponentError(
                    "a pure cancellation exponent identity failed"
                )

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(sorted((*seed_support(self.seed), *self.additions)))

    @property
    def exact_checks(self) -> Mapping[str, bool]:
        support = self.support
        rows = []
        coefficients = []
        for equation, coefficient in self.mixed_relation:
            active = _active_monomials(support, equation)
            rows.append(
                _exponent_difference(
                    support, active[0], active[1]
                )
            )
            coefficients.append(coefficient)
        if self.kind == "pure-cancellation":
            pure_active = _active_monomials(
                support, int(self.pure_equation)
            )
            rows.append(
                _exponent_difference(
                    support, pure_active[0], pure_active[1]
                )
            )
            coefficients.append(1)
        return {
            "support_has_21_nonzero_coordinates": len(support) == 21,
            "all_active_mixed_equations_are_binomial": all(
                len(_active_monomials(support, equation)) in (0, 2)
                for equation, rhs in enumerate(_system().rhs_values)
                if not rhs
            ),
            "mixed_coefficient_sum_is_odd": (
                sum(
                    coefficient
                    for _equation, coefficient in self.mixed_relation
                )
                % 2
                != 0
            ),
            "combined_exponent_sum_is_zero": (
                not any(_weighted_rows(rows, coefficients))
            ),
            "exact_support_torus_is_impossible": True,
        }

    def to_dict(self) -> dict:
        checks = dict(self.exact_checks)
        if not all(checks.values()):
            raise KrennVerticalComponentError(
                "a support-21 terminal failed serialization"
            )
        relation_rows = []
        for equation, coefficient in self.mixed_relation:
            active = _active_monomials(self.support, equation)
            relation_rows.append(
                {
                    "equation": equation,
                    "coloring": list(
                        coloring_from_index(N, D, equation)
                    ),
                    "coefficient": coefficient,
                    "first_monomial": list(active[0]),
                    "second_monomial": list(active[1]),
                    "exponent_difference": list(
                        _exponent_difference(
                            self.support, active[0], active[1]
                        )
                    ),
                }
            )
        contradiction = "1=-1"
        pure_payload = None
        if self.kind == "pure-cancellation":
            active = _active_monomials(
                self.support, int(self.pure_equation)
            )
            pure_payload = {
                "equation": self.pure_equation,
                "coloring": list(
                    coloring_from_index(N, D, int(self.pure_equation))
                ),
                "first_monomial": list(active[0]),
                "second_monomial": list(active[1]),
                "exponent_difference_coefficient": 1,
                "forced_ratio": "-1",
                "forced_output": "M0+M1=0",
                "required_output": "1",
            }
            contradiction = "pure output is forced to 0 but must equal 1"
        return {
            "seed": list(self.seed),
            "terminal_index": self.terminal_index,
            "additions": list(self.additions),
            "support": list(self.support),
            "support_coordinates": [
                list(variable_key(N, D, index))
                for index in self.support
            ],
            "kind": self.kind,
            "mixed_relation": relation_rows,
            "mixed_coefficient_sum": sum(
                coefficient
                for _equation, coefficient in self.mixed_relation
            ),
            "pure_cancellation": pure_payload,
            "contradiction": contradiction,
            "exact_checks": checks,
        }


@lru_cache(maxsize=1)
def support21_terminal_certificates(
) -> tuple[Support21TerminalCertificate, ...]:
    """Replay all 12 algebraic obstructions at the closure boundary."""

    result = []
    for seed in sorted(SUPPORT21_TERMINAL_ADDITIONS):
        for terminal_index, additions in enumerate(
            SUPPORT21_TERMINAL_ADDITIONS[seed]
        ):
            kind, relation, pure_equation = (
                SUPPORT21_TERMINAL_RELATIONS[
                    (seed, terminal_index)
                ]
            )
            result.append(
                Support21TerminalCertificate(
                    seed=seed,
                    terminal_index=terminal_index,
                    additions=additions,
                    kind=kind,
                    mixed_relation=relation,
                    pure_equation=pure_equation,
                )
            )
    certificates = tuple(result)
    if (
        len(certificates) != 12
        or len({certificate.support for certificate in certificates}) != 12
        or Counter(certificate.kind for certificate in certificates)
        != Counter({
            "odd-mixed-circuit": 8,
            "pure-cancellation": 4,
        })
        or not all(
            all(certificate.exact_checks.values())
            for certificate in certificates
        )
    ):
        raise KrennVerticalComponentError(
            "the support-21 terminal certificate census changed"
        )
    return certificates


@lru_cache(maxsize=1)
def recorded_support21_global_audit() -> Mapping[str, object]:
    """Return the small receipt for the completed 12,992,269-node replay."""

    representatives = tuple(
        representative
        for representative, _size
        in EXPECTED_REPRESENTATIVES_AND_SIZES
    )
    terminal_counts = tuple(
        len(SUPPORT21_TERMINAL_ADDITIONS.get(representative, ()))
        for representative in representatives
    )
    certificates = support21_terminal_certificates()
    if (
        terminal_counts != (0, 0, 3, 0, 0, 3, 6, 0)
        or sum(EXPECTED_SUPPORT21_CLOSURE_NODES) != 12_992_269
        or len(certificates) != sum(terminal_counts)
    ):
        raise KrennVerticalComponentError(
            "the recorded support-21 global receipt changed"
        )
    return {
        "schema": SUPPORT_CLOSURE_SCHEMA,
        "max_total_support": 21,
        "root_receipts": [
            {
                "seed": list(seed),
                "nodes_examined": nodes,
                "singleton_free_terminal_count": count,
            }
            for seed, nodes, count in zip(
                representatives,
                EXPECTED_SUPPORT21_CLOSURE_NODES,
                terminal_counts,
                strict=True,
            )
        ],
        "totals": {
            "nodes_examined": sum(EXPECTED_SUPPORT21_CLOSURE_NODES),
            "singleton_free_terminals": len(certificates),
            "odd_mixed_circuit_terminals": sum(
                certificate.kind == "odd-mixed-circuit"
                for certificate in certificates
            ),
            "pure_cancellation_terminals": sum(
                certificate.kind == "pure-cancellation"
                for certificate in certificates
            ),
        },
        "terminal_certificates": [
            certificate.to_dict() for certificate in certificates
        ],
        "closure_replay": {
            "algorithm": (
                "deterministic first-singleton missing-set closure"
            ),
            "frontiers_exhausted": True,
            "workers": 8,
            "workers_within_declared_limit": True,
            "large_frontiers_or_caches_committed": False,
            "disposable_worker_receipts_location": (
                "D:\\KrennScratch\\structural_pole_law"
            ),
            "independent_regeneration_is_gated_long": True,
        },
        "finite_exact_support_lower_bound": 22,
        "statement": (
            "Every finite GHZ witness contains a diagonal seed. The exact "
            "eight-root closure exhausts every support through size 21. "
            "Its 12 singleton-free terminals are all killed by retained "
            "integer exponent certificates, so every finite witness has "
            "support at least 22 without assuming any natural coordinate."
        ),
        "claim_boundary": {
            "natural_coordinate_retention_assumed": False,
            "symmetry_related_weights_equated": False,
            "supports_at_size_22_or_above_excluded": False,
            "global_nonexistence_proved": False,
        },
    }


def replay_support21_global_closure() -> UnconditionalSupportClosure:
    """Long exact regeneration of the eight support-21 closure frontiers."""

    result = certify_unconditional_support_closure(
        max_total_support=21
    )
    expected_terminals = {
        seed: tuple(
            tuple(sorted((*seed_support(seed), *additions)))
            for additions in SUPPORT21_TERMINAL_ADDITIONS.get(seed, ())
        )
        for seed, _size in EXPECTED_REPRESENTATIVES_AND_SIZES
    }
    if (
        tuple(root.nodes_examined for root in result.roots)
        != EXPECTED_SUPPORT21_CLOSURE_NODES
        or any(
            root.singleton_free_supports
            != expected_terminals[root.seed]
            for root in result.roots
        )
    ):
        raise KrennVerticalComponentError(
            "the long support-21 closure replay changed"
        )
    return result


def exact_vertical_component_structure(
    *,
    support_cap: int = 11,
) -> Mapping[str, object]:
    """Build the fail-closed composite vertical-component certificate."""

    support_closure = certify_unconditional_support_closure(
        max_total_support=support_cap
    )
    lower_bound = support_closure.finite_support_lower_bound
    payload = {
        "schema": VERTICAL_COMPONENT_SCHEMA,
        "parameters": {
            "n": N,
            "d": D,
            "victim_equation": VICTIM_EQUATION,
            "victim_coloring": list(VICTIM_COLORING),
        },
        "color_gauge_orbit": natural_color_gauge_orbit_audit(),
        "fixed_victim_nine_coordinate_charts": (
            fixed_victim_nine_coordinate_audit()
        ),
        "small_direct_support_census": small_support_direct_census(),
        "unconditional_support_closure": support_closure.to_dict(),
        "recorded_support21_global_audit": (
            recorded_support21_global_audit()
        ),
        "next_exact_test": {
            "direct_GHZ_chart_ideal": (
                "<all 729 GHZ equations, seed-coordinate localizers>"
            ),
            "fixed_line_vertical_test": (
                "after localizing away from the known natural component, "
                "test 1 in B+<F_002121>; a larger fixed-line reduction has "
                "104 seed-chart orbits, while direct GHZ has eight"
            ),
            "proof_requirement": (
                "an exact unit-ideal, radical, or exact witness certificate"
            ),
        },
        "claims": {
            "known_Laurent_path_is_color_diagonal_orbit_in_moving_line_incidence": (
                True
            ),
            "known_natural_nine_coordinate_stratum_has_no_vertical_point": (
                True
            ),
            "all_fixed_victim_nine_coordinate_strata_classified": True,
            "finite_exact_support_lower_bound": 22,
            "routine_replay_support_lower_bound": lower_bound,
            "supports_omitting_natural_coordinates_included_below_cap": True,
            "all_vertical_components_excluded": False,
            "D_in_radical_J_mix_decided": False,
            "finite_affine_GHZ_membership_decided": False,
            "exact_affine_GHZ_membership_status": "undecided",
        },
    }
    if (
        not payload["color_gauge_orbit"]["exact_checks"]
        or not all(payload["color_gauge_orbit"]["exact_checks"].values())
        or not all(
            payload["fixed_victim_nine_coordinate_charts"][
                "exact_checks"
            ].values()
        )
        or lower_bound is None
    ):
        raise KrennVerticalComponentError(
            "the composite vertical-component certificate failed"
        )
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one deterministic n=6,d=3 diagonal-seed support closure."
        )
    )
    parser.add_argument("--root-index", type=int, required=True)
    parser.add_argument("--max-total-support", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    representatives = tuple(
        representative
        for representative, _size
        in EXPECTED_REPRESENTATIVES_AND_SIZES
    )
    if not 0 <= args.root_index < len(representatives):
        raise KrennVerticalComponentError(
            "root index must lie in 0..7"
        )
    closure = bounded_singleton_closure(
        representatives[args.root_index],
        max_total_support=args.max_total_support,
    )
    payload = closure.to_dict()
    payload["root_index"] = args.root_index
    text = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(args.output)


__all__ = [
    "AMBIENT_VARIABLES",
    "COLOR_GAUGE_PARAMETERS",
    "EXPECTED_FIXED_VICTIM_ORBIT_HISTOGRAM",
    "EXPECTED_FIXED_VICTIM_SEED_ORBITS",
    "EXPECTED_FIXED_VICTIM_SURVIVOR_ORBIT",
    "EXPECTED_LONG_CLOSURE_NODES_AT_TOTAL_20",
    "EXPECTED_SMALL_SUPPORT_CASES",
    "EXPECTED_SUPPORT21_CLOSURE_NODES",
    "KrennVerticalComponentError",
    "NATURAL_SEED",
    "RootSupportClosure",
    "SUPPORT21_TERMINAL_ADDITIONS",
    "SUPPORT21_TERMINAL_RELATIONS",
    "Support21TerminalCertificate",
    "UnconditionalSupportClosure",
    "VICTIM_COLORING",
    "VICTIM_EQUATION",
    "bounded_singleton_closure",
    "certify_unconditional_support_closure",
    "color_diagonal_exponent_matrix",
    "coloring_character",
    "exact_vertical_component_structure",
    "fixed_victim_nine_coordinate_audit",
    "fixed_victim_seed_orbits",
    "monomial_character",
    "natural_color_gauge_orbit_audit",
    "recorded_support21_global_audit",
    "replay_support21_global_closure",
    "seed_support",
    "small_support_direct_census",
    "support21_terminal_certificates",
]


if __name__ == "__main__":
    _main()
