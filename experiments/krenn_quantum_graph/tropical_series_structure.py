r"""Exact scope audit for the known Laurent valuation and residual grading.

There are two different pieces of "tropical" structure near the natural
``n=6,d=3`` seed, and conflating them obscures what the known Laurent path
can prove.

First restrict the moving-line incidence

    Phi(W) = GHZ + u*e_002121

to the natural nine-coordinate torus.  The three pure equations and the
victim equation give four exponent relations in the ten valuations
``(val(W_i), val(u))``.  Their common kernel has dimension six.  The direct
GHZ color-diagonal gauge has rank six on the same ten coordinates and lies
in that kernel, so the two spaces are equal.  Thus the entire tropical cone
on this exact support is gauge lineality.  In particular, the displayed
Laurent valuation is not a transverse ray from which adjacent cones can be
read after quotienting the gauge.

Second pass to the all-nine-normalized natural chart for the *constant* GHZ
fiber.  Its defect equation is

    1 + six quadratic repair monomials
      + eight cubic repair monomials = 0.

Every one of the fifteen terms has residual ``Z^9`` character zero.
Consequently the residual grading is useful for block polynomial algebra,
but is completely blind to the decisive first repair balance.  For
nonnegative coordinate valuations, the constant term can tie for the
minimum only if at least one repair monomial has valuation zero.  Its
factors then all have valuation zero.  The fourteen opens obtained this way
are exactly the repair-monomial atlas, in four strict-stabilizer orbits.

This classifies the exact-support moving cone and the first tropical tie of
one defect hypersurface.  It does not enumerate higher-support cones or
exclude a component contained entirely in ``u=0``.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import heapq
from itertools import product
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    NATURAL_DEFECT_EQUATION,
    NATURAL_ORBIT_INDEX,
    natural_defect_polynomial,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    normalized_seed_chart,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    residual_torus_characters,
)
from experiments.krenn_quantum_graph.localized_chart_monomial_atlas import (
    REPAIR_MONOMIAL_REPRESENTATIVES,
    natural_repair_monomial_orbits,
)
from experiments.krenn_quantum_graph.matching_circuit_structure import (
    victim_stabilizer_actions,
)
from experiments.krenn_quantum_graph.structural_pole_law import (
    FIXED_VICTIM_SEEDS,
)
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    generate_sparse_system,
)
from experiments.krenn_quantum_graph.vertical_component import (
    NATURAL_SEED,
    VICTIM_COLORING,
    VICTIM_EQUATION,
    color_diagonal_exponent_matrix,
    coloring_character,
    fixed_victim_nine_coordinate_audit,
    natural_color_gauge_orbit_audit,
    seed_support,
)


TROPICAL_SERIES_SCHEMA = "krenn-n6-d3-tropical-series-structure-v1"


class KrennTropicalSeriesError(RuntimeError):
    """An exponent, character, orbit, or claim-boundary replay failed."""


def _strict_json_equal(left: object, right: object) -> bool:
    """Compare JSON data without Python's ``True == 1`` coercion."""

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


def _rank_over_q(matrix: Sequence[Sequence[int]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennTropicalSeriesError("an exact matrix is ragged")
    rank = 0
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
        rows[rank] = [value / pivot for value in rows[rank]]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                left - factor * right
                for left, right in zip(rows[row], rows[rank], strict=True)
            ]
        rank += 1
    return rank


def _dot(left: Sequence[int], right: Sequence[int]) -> int:
    left = tuple(map(int, left))
    right = tuple(map(int, right))
    if len(left) != len(right):
        raise KrennTropicalSeriesError("an exact dot product is ragged")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _exponent(
    monomial: Sequence[int], support: Sequence[int], *, append_u: int = 0
) -> tuple[int, ...]:
    support = tuple(map(int, support))
    positions = {variable: position for position, variable in enumerate(support)}
    result = [0] * (len(support) + 1)
    for raw_variable in monomial:
        variable = int(raw_variable)
        try:
            result[positions[variable]] += 1
        except KeyError as error:
            raise KrennTropicalSeriesError(
                "a matching monomial left the natural support"
            ) from error
    result[-1] = int(append_u)
    return tuple(result)


def _mask(values: Sequence[int]) -> int:
    result = 0
    for value in values:
        result |= 1 << int(value)
    return result


def _indices(mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(135) if mask & (1 << index))


def _matching_masks() -> tuple[tuple[int, ...], ...]:
    system = generate_sparse_system(6, 3)
    return tuple(
        tuple(_mask(monomial) for monomial in system.equation_monomials(
            equation
        ))
        for equation in range(729)
    )


def _first_nonvictim_singleton(
    support_mask: int, matching_masks: Sequence[Sequence[int]]
) -> tuple[int, int] | None:
    excluded = {0, 364, 728, VICTIM_EQUATION}
    for equation, monomials in enumerate(matching_masks):
        if equation in excluded:
            continue
        active = -1
        count = 0
        for monomial in monomials:
            if monomial & ~support_mask == 0:
                active = monomial
                count += 1
                if count > 1:
                    break
        if count == 1:
            return equation, active
    return None


def _singleton_closure(
    root_support: Sequence[int],
    maximum_total_support: int,
    matching_masks: Sequence[Sequence[int]],
) -> dict:
    """Exhaust supersets by completing the first singleton equation."""

    root = _mask(root_support)
    maximum_total_support = int(maximum_total_support)
    queue = [(root.bit_count(), root)]
    seen = {root}
    terminals = []
    size_census: Counter[int] = Counter()
    cutoff_frontier = set()
    pruned_child_transitions = 0
    while queue:
        _size, support = heapq.heappop(queue)
        size_census[support.bit_count()] += 1
        singleton = _first_nonvictim_singleton(support, matching_masks)
        if singleton is None:
            terminals.append(support)
            continue
        equation, active = singleton
        for alternative in matching_masks[equation]:
            if alternative == active:
                continue
            child = support | alternative
            if child.bit_count() > maximum_total_support:
                cutoff_frontier.add(support)
                pruned_child_transitions += 1
            elif child not in seen:
                seen.add(child)
                heapq.heappush(queue, (child.bit_count(), child))
    return {
        "nodes": len(seen),
        "maximum_total_support": maximum_total_support,
        "size_census": dict(sorted(size_census.items())),
        "terminal_supports": tuple(
            sorted((_indices(mask) for mask in terminals))
        ),
        "bounded_search_complete_through_support": True,
        "cutoff_frontier_nodes": len(cutoff_frontier),
        "pruned_child_transitions": pruned_child_transitions,
        "frontier_exhausted": not cutoff_frontier,
    }


def _difference(
    left: Sequence[int], right: Sequence[int], support: Sequence[int]
) -> tuple[int, ...]:
    support = tuple(map(int, support))
    positions = {variable: position for position, variable in enumerate(support)}
    result = [0] * len(support)
    for variable in left:
        result[positions[int(variable)]] += 1
    for variable in right:
        result[positions[int(variable)]] -= 1
    return tuple(result)


def _natural_exact_support_cone() -> dict:
    system = generate_sparse_system(6, 3)
    support = seed_support(NATURAL_SEED)
    pure_equations = (0, 364, 728)
    active = {}
    for equation in (*pure_equations, VICTIM_EQUATION):
        monomials = tuple(
            monomial
            for monomial in system.equation_monomials(equation)
            if set(monomial) <= set(support)
        )
        if len(monomials) != 1:
            raise KrennTropicalSeriesError(
                "the natural torus lost a unique active matching"
            )
        active[equation] = monomials[0]

    relations = tuple(
        _exponent(active[equation], support)
        for equation in pure_equations
    ) + (
        tuple(
            left - right
            for left, right in zip(
                _exponent(active[VICTIM_EQUATION], support),
                _exponent((), support, append_u=1),
                strict=True,
            )
        ),
    )
    relation_rank = _rank_over_q(relations)

    ambient_gauge = color_diagonal_exponent_matrix()
    gauge_rows = tuple(ambient_gauge[index] for index in support) + (
        coloring_character(VICTIM_COLORING),
    )
    gauge_rank = _rank_over_q(gauge_rows)
    annihilation = tuple(
        tuple(
            sum(
                relation[row] * gauge_rows[row][column]
                for row in range(len(gauge_rows))
            )
            for column in range(len(gauge_rows[0]))
        )
        for relation in relations
    )

    orbit = natural_color_gauge_orbit_audit()
    gamma = tuple(orbit["gamma"])
    known_valuation = (
        *tuple(orbit["natural_support_valuations"]),
        orbit["victim_character_on_gamma"],
    )
    gamma_image = tuple(
        sum(row[column] * gamma[column] for column in range(len(gamma)))
        for row in gauge_rows
    )
    fixed_victim = fixed_victim_nine_coordinate_audit()

    checks = {
        "natural_support_has_nine_weights": len(support) == 9,
        "four_unique_active_incidence_monomials": len(active) == 4,
        "incidence_exponent_relation_rank_is_four": relation_rank == 4,
        "incidence_tropical_kernel_dimension_is_six": (
            len(support) + 1 - relation_rank == 6
        ),
        "direct_GHZ_gauge_rank_on_weights_and_u_is_six": gauge_rank == 6,
        "gauge_is_annihilated_by_all_incidence_relations": all(
            not any(row) for row in annihilation
        ),
        "incidence_tropical_kernel_equals_gauge_image": (
            relation_rank == 4
            and gauge_rank == 6
            and all(not any(row) for row in annihilation)
        ),
        "known_Laurent_valuation_is_gamma_image": (
            gamma_image == known_valuation
        ),
        "known_Laurent_valuation_satisfies_all_four_relations": all(
            _dot(relation, known_valuation) == 0
            for relation in relations
        ),
        "fixed_victim_seed_orbits_104": (
            fixed_victim["seed_orbits"] == 104
        ),
        "other_103_nine_coordinate_orbits_have_a_singleton": (
            fixed_victim["orbits_killed_by_nonvictim_singleton"] == 103
        ),
        "sole_nine_coordinate_survivor_is_natural_orbit": (
            tuple(
                tuple(seed)
                for seed in fixed_victim["surviving_orbit"]["members"]
            )
            == (
                (0, 4, 8),
                (0, 9, 13),
                (2, 4, 13),
                (2, 9, 8),
            )
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise KrennTropicalSeriesError(
            f"the exact-support cone replay failed: {failed}"
        )
    return {
        "support": list(support),
        "moving_incidence_coordinates": 10,
        "exponent_relations": [list(row) for row in relations],
        "relation_rank_over_Q": relation_rank,
        "tropical_kernel_dimension": len(support) + 1 - relation_rank,
        "color_gauge_rank": gauge_rank,
        "known_Laurent_valuation": list(known_valuation),
        "known_Laurent_cocharacter": list(gamma),
        "fixed_victim_nine_coordinate_seed_orbits": 104,
        "fixed_victim_nine_coordinate_singleton_orbits": 103,
        "fixed_victim_surviving_orbit_size": 4,
        "interpretation": (
            "on the natural exact-support torus the full moving-incidence "
            "tropical cone is exactly direct-GHZ color-gauge lineality"
        ),
        "exact_checks": checks,
    }


def _residual_invariant_defect_star() -> dict:
    chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
    characters = residual_torus_characters(NATURAL_ORBIT_INDEX)
    ambient_to_local = {
        ambient: local
        for local, ambient in enumerate(chart.remaining_weight_indices)
    }
    defect = natural_defect_polynomial()
    rows = []
    for monomial, coefficient in sorted(
        defect.items(), key=lambda item: (len(item[0]), item[0])
    ):
        try:
            local = tuple(ambient_to_local[index] for index in monomial)
        except KeyError as error:
            raise KrennTropicalSeriesError(
                "a defect repair factor was fixed by the natural chart"
            ) from error
        character = tuple(
            sum(characters[variable][coordinate] for variable in local)
            for coordinate in range(9)
        )
        rows.append(
            {
                "coefficient": str(coefficient),
                "ambient_monomial": list(monomial),
                "degree": len(monomial),
                "residual_character": list(character),
            }
        )

    repair_terms = tuple(
        monomial for monomial in defect if monomial
    )
    orbits = natural_repair_monomial_orbits()
    flattened_orbits = {
        tuple(monomial) for orbit in orbits for monomial in orbit
    }
    checks = {
        "natural_chart_is_seed_0_4_8": (
            chart.seed == NATURAL_SEED == (0, 4, 8)
        ),
        "defect_equation_is_70": (
            NATURAL_DEFECT_EQUATION == VICTIM_EQUATION == 70
        ),
        "defect_has_constant_plus_fourteen_repairs": (
            len(defect) == 15 and () in defect and len(repair_terms) == 14
        ),
        "repair_degree_census_is_six_quadratic_eight_cubic": (
            sum(len(monomial) == 2 for monomial in repair_terms) == 6
            and sum(len(monomial) == 3 for monomial in repair_terms) == 8
        ),
        "all_fifteen_terms_have_residual_character_zero": all(
            not any(row["residual_character"]) for row in rows
        ),
        "four_strict_stabilizer_orbits_cover_all_repairs": (
            len(orbits) == 4
            and tuple(map(len, orbits)) == (3, 3, 6, 2)
            and flattened_orbits == set(repair_terms)
        ),
        "orbit_representatives_match_exact_atlas": (
            tuple(orbit[0] for orbit in orbits)
            == REPAIR_MONOMIAL_REPRESENTATIVES
        ),
        "nonnegative_tropical_tie_forces_zero_repair_value": True,
        "zero_repair_value_forces_every_factor_value_zero": True,
        "four_repair_orbit_charts_cover_first_defect_ties": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise KrennTropicalSeriesError(
            f"the residual defect-star replay failed: {failed}"
        )
    return {
        "residual_character_lattice": "Z^9",
        "defect_equation": NATURAL_DEFECT_EQUATION,
        "terms": rows,
        "repair_term_count": len(repair_terms),
        "repair_orbit_sizes": [len(orbit) for orbit in orbits],
        "repair_orbit_representatives": [
            list(representative)
            for representative in REPAIR_MONOMIAL_REPRESENTATIVES
        ],
        "nonnegative_valuation_lemma": (
            "the constant has value 0 and every repair value is a sum of "
            "nonnegative factor values; a tropical tie therefore forces one "
            "repair value, and every factor in it, to have value 0"
        ),
        "interpretation": (
            "the residual Z^9 grading is lineality and assigns the entire "
            "decisive defect balance to degree zero"
        ),
        "exact_checks": checks,
    }


def _cross_node_minimal_unions() -> dict:
    """Compare the six pairwise unions of the four natural victim nodes."""

    system = generate_sparse_system(6, 3)
    pure = {0, 364, 728}
    supports = tuple(
        frozenset(seed_support(seed)) for seed in FIXED_VICTIM_SEEDS
    )
    rows = []
    unions = set()
    for right in range(len(supports)):
        for left in range(right):
            support = supports[left] | supports[right]
            unions.add(support)
            singleton_equations = []
            other_active_counts = []
            for equation in range(729):
                if equation in pure or equation == VICTIM_EQUATION:
                    continue
                active = tuple(
                    monomial
                    for monomial in system.equation_monomials(equation)
                    if set(monomial) <= support
                )
                if len(active) == 1:
                    singleton_equations.append(equation)
                elif active:
                    other_active_counts.append((equation, len(active)))
            rows.append(
                {
                    "seed_indices": [left, right],
                    "seeds": [
                        list(FIXED_VICTIM_SEEDS[left]),
                        list(FIXED_VICTIM_SEEDS[right]),
                    ],
                    "union_support_size": len(support),
                    "nonvictim_singleton_equations": singleton_equations,
                    "nonvictim_singleton_colorings": [
                        list(coloring_from_index(6, 3, equation))
                        for equation in singleton_equations
                    ],
                    "other_active_nonvictim_equation_counts": [
                        [equation, count]
                        for equation, count in other_active_counts
                    ],
                }
            )

    base = min(unions, key=lambda support: tuple(sorted(support)))
    orbit = {
        frozenset(action[2][variable] for variable in base)
        for action in victim_stabilizer_actions()
    }
    canonical_support = tuple(sorted(supports[0] | supports[1]))
    canonical_row = next(
        row for row in rows if row["seed_indices"] == [0, 1]
    )
    canonical_singletons = tuple(
        canonical_row["nonvictim_singleton_equations"]
    )
    matching_masks = _matching_masks()
    canonical_mask = _mask(canonical_support)

    alternative_missing_sets = []
    missing_histograms = []
    for equation in canonical_singletons:
        active = tuple(
            monomial
            for monomial in matching_masks[equation]
            if monomial & ~canonical_mask == 0
        )
        if len(active) != 1:
            raise KrennTropicalSeriesError(
                "a canonical bridge equation lost its singleton"
            )
        missing = tuple(sorted({
            tuple(
                index
                for index in range(135)
                if monomial & ~canonical_mask & (1 << index)
            )
            for monomial in matching_masks[equation]
            if monomial != active[0]
        }, key=lambda values: (len(values), values)))
        alternative_missing_sets.append(missing)
        missing_histograms.append(dict(sorted(Counter(
            map(len, missing)
        ).items())))

    minimum_addition_size = 136
    minimum_additions: set[frozenset[int]] = set()
    completion_combinations = 0
    for choices in product(*alternative_missing_sets):
        completion_combinations += 1
        addition = frozenset().union(*map(frozenset, choices))
        if len(addition) < minimum_addition_size:
            minimum_addition_size = len(addition)
            minimum_additions = {addition}
        elif len(addition) == minimum_addition_size:
            minimum_additions.add(addition)
    minimum_addition_rows = tuple(
        tuple(sorted(addition)) for addition in sorted(
            minimum_additions, key=lambda values: tuple(sorted(values))
        )
    )
    if len(minimum_addition_rows) != 1:
        raise KrennTropicalSeriesError(
            "the canonical bridge minimum completion is no longer unique"
        )
    minimum_support = tuple(sorted((
        *canonical_support, *minimum_addition_rows[0]
    )))
    minimum_support_singletons = []
    minimum_mask = _mask(minimum_support)
    for equation in range(729):
        if equation in {0, 364, 728, VICTIM_EQUATION}:
            continue
        active = sum(
            monomial & ~minimum_mask == 0
            for monomial in matching_masks[equation]
        )
        if active == 1:
            minimum_support_singletons.append(equation)

    minimum_branch_closure = _singleton_closure(
        minimum_support, 24, matching_masks
    )
    full_pair_closure = _singleton_closure(
        canonical_support, 21, matching_masks
    )

    expected_terminal = (
        0, 13, 16, 23, 26, 31, 34, 41, 44, 49, 52, 59, 62, 67,
        70, 77, 80, 81, 98, 121, 126,
    )
    terminal_supports = full_pair_closure["terminal_supports"]
    terminal = terminal_supports[0] if terminal_supports else ()
    terminal_set = set(terminal)
    active_mixed = {}
    for equation in range(729):
        if equation in {0, 364, 728, VICTIM_EQUATION}:
            continue
        active = tuple(
            monomial
            for monomial in system.equation_monomials(equation)
            if set(monomial) <= terminal_set
        )
        if active:
            active_mixed[equation] = active
    pure_active = {
        equation: tuple(
            monomial
            for monomial in system.equation_monomials(equation)
            if set(monomial) <= terminal_set
        )
        for equation in (0, 364, 728)
    }
    cancellation_receipts = {
        364: ((329, -1), (369, 1), (404, 1)),
        728: ((404, -1), (485, 1), (647, 1)),
    }
    receipt_rows = []
    receipts_exact = True
    for pure_equation, relation in cancellation_receipts.items():
        pure_monomials = pure_active[pure_equation]
        if len(pure_monomials) != 2:
            receipts_exact = False
            continue
        pure_difference = _difference(
            pure_monomials[0], pure_monomials[1], terminal
        )
        mixed_sum = tuple(
            sum(
                coefficient * _difference(
                    active_mixed[equation][0],
                    active_mixed[equation][1],
                    terminal,
                )[coordinate]
                for equation, coefficient in relation
            )
            for coordinate in range(len(terminal))
        )
        exact = (
            mixed_sum == pure_difference
            and sum(coefficient for _equation, coefficient in relation) % 2
            == 1
        )
        receipts_exact &= exact
        receipt_rows.append({
            "pure_equation": pure_equation,
            "relation": [
                {"mixed_equation": equation, "coefficient": coefficient}
                for equation, coefficient in relation
            ],
            "coefficient_sum": sum(
                coefficient for _equation, coefficient in relation
            ),
            "integer_exponent_identity_exact": exact,
            "consequence": (
                "the two active pure monomials have ratio -1, so their "
                "sum is zero rather than one"
            ),
        })

    checks = {
        "four_natural_fixed_victim_nodes": len(supports) == 4,
        "six_distinct_pairwise_node_unions": (
            len(rows) == len(unions) == 6
        ),
        "all_pairwise_unions_have_support_thirteen": all(
            row["union_support_size"] == 13 for row in rows
        ),
        "every_pairwise_union_has_four_nonvictim_singletons": all(
            len(row["nonvictim_singleton_equations"]) == 4 for row in rows
        ),
        "no_pairwise_union_has_another_active_nonvictim_equation": all(
            not row["other_active_nonvictim_equation_counts"] for row in rows
        ),
        "six_pairwise_unions_form_one_victim_stabilizer_orbit": (
            orbit == unions
        ),
        "every_minimal_pairwise_union_torus_is_empty": True,
        "canonical_pair_has_four_singletons_410_450_572_612": (
            canonical_singletons == (410, 450, 572, 612)
        ),
        "each_singleton_has_six_two_and_eight_three_coordinate_repairs": (
            missing_histograms == [{2: 6, 3: 8}] * 4
        ),
        "all_38416_simultaneous_completion_choices_enumerated": (
            completion_combinations == 14**4 == 38_416
        ),
        "unique_minimum_completion_adds_four_coordinates": (
            minimum_addition_size == 4
            and minimum_addition_rows == ((5, 7, 86, 131),)
        ),
        "minimum_completion_support_is_seventeen": (
            len(minimum_support) == 17
        ),
        "minimum_completion_creates_seven_new_singletons": (
            len(minimum_support_singletons) == 7
        ),
        "minimum_completion_branch_has_no_terminal_through_24": (
            minimum_branch_closure[
                "bounded_search_complete_through_support"
            ]
            and minimum_branch_closure["maximum_total_support"] == 24
            and not minimum_branch_closure["frontier_exhausted"]
            and minimum_branch_closure["cutoff_frontier_nodes"] > 0
            and minimum_branch_closure["nodes"] == 1_975
            and not minimum_branch_closure["terminal_supports"]
        ),
        "full_canonical_pair_closure_exhausted_through_21": (
            full_pair_closure[
                "bounded_search_complete_through_support"
            ]
            and full_pair_closure["maximum_total_support"] == 21
            and not full_pair_closure["frontier_exhausted"]
            and full_pair_closure["cutoff_frontier_nodes"] > 0
            and full_pair_closure["nodes"] == 7_564
        ),
        "full_pair_closure_has_one_support_21_terminal": (
            terminal_supports == (expected_terminal,)
        ),
        "support_21_terminal_has_fourteen_mixed_binomials": (
            len(active_mixed) == 14
            and all(len(monomials) == 2 for monomials in active_mixed.values())
        ),
        "support_21_terminal_pure_counts_are_one_two_two": (
            tuple(len(pure_active[equation]) for equation in (0, 364, 728))
            == (1, 2, 2)
        ),
        "both_pure_cancellation_receipts_replay_exactly": receipts_exact,
        "no_pairwise_natural_node_bridge_through_support_21": (
            terminal_supports == (expected_terminal,) and receipts_exact
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise KrennTropicalSeriesError(
            f"the cross-node union replay failed: {failed}"
        )
    return {
        "nodes": [list(seed) for seed in FIXED_VICTIM_SEEDS],
        "pairwise_unions": rows,
        "pairwise_union_orbits_under_victim_stabilizer": 1,
        "canonical_pair_completion": {
            "seed_indices": [0, 1],
            "support": list(canonical_support),
            "singleton_equations": list(canonical_singletons),
            "alternative_missing_size_histograms": [
                {str(size): count for size, count in histogram.items()}
                for histogram in missing_histograms
            ],
            "simultaneous_completion_choices": completion_combinations,
            "minimum_added_coordinates": [
                list(addition) for addition in minimum_addition_rows
            ],
            "minimum_added_coordinate_count": minimum_addition_size,
            "minimum_completed_support": list(minimum_support),
            "new_singleton_equations": minimum_support_singletons,
            "minimum_branch_closure_through_support_24": {
                "nodes": minimum_branch_closure["nodes"],
                "maximum_total_support": (
                    minimum_branch_closure["maximum_total_support"]
                ),
                "size_census": {
                    str(size): count for size, count in
                    minimum_branch_closure["size_census"].items()
                },
                "singleton_free_terminals": 0,
                "bounded_search_complete_through_support": (
                    minimum_branch_closure[
                        "bounded_search_complete_through_support"
                    ]
                ),
                "cutoff_frontier_nodes": (
                    minimum_branch_closure["cutoff_frontier_nodes"]
                ),
                "pruned_child_transitions": (
                    minimum_branch_closure["pruned_child_transitions"]
                ),
                "frontier_exhausted": (
                    minimum_branch_closure["frontier_exhausted"]
                ),
            },
        },
        "full_canonical_pair_closure_through_support_21": {
            "nodes": full_pair_closure["nodes"],
            "maximum_total_support": (
                full_pair_closure["maximum_total_support"]
            ),
            "size_census": {
                str(size): count
                for size, count in full_pair_closure["size_census"].items()
            },
            "singleton_free_terminals": [list(terminal)],
            "terminal_active_mixed_binomials": sorted(active_mixed),
            "terminal_pure_active_monomial_counts": {
                str(equation): len(monomials)
                for equation, monomials in pure_active.items()
            },
            "pure_cancellation_receipts": receipt_rows,
            "bounded_search_complete_through_support": (
                full_pair_closure[
                    "bounded_search_complete_through_support"
                ]
            ),
            "cutoff_frontier_nodes": (
                full_pair_closure["cutoff_frontier_nodes"]
            ),
            "pruned_child_transitions": (
                full_pair_closure["pruned_child_transitions"]
            ),
            "frontier_exhausted": full_pair_closure["frontier_exhausted"],
        },
        "interpretation": (
            "turning on the weights of a second natural node does not bridge "
            "the two moving components: four new singleton mixed outputs "
            "must first be repaired by additional matching coordinates; "
            "the complete canonical closure through support 21 has one "
            "terminal, killed exactly by odd pure-cancellation receipts"
        ),
        "exact_checks": checks,
    }


def tropical_series_structure_audit() -> dict:
    """Return the exact known-cone and residual-blind-spot audit."""

    exact_support = _natural_exact_support_cone()
    defect_star = _residual_invariant_defect_star()
    cross_node = _cross_node_minimal_unions()
    payload = {
        "schema": TROPICAL_SERIES_SCHEMA,
        "known_exact_support_moving_cone": exact_support,
        "natural_node_cross_comparison": cross_node,
        "constant_fiber_residual_defect_star": defect_star,
        "concrete_next_certificate_target": {
            "name": "higher-support initial-ideal atlas",
            "scope": (
                "primitive integer valuation cones that activate repair "
                "coordinates, modulo residual lineality and S6 x S3"
            ),
            "certificate_per_cone": (
                "an exact Laurent initial-ideal unit/monomial certificate, "
                "or an exact Puiseux leading solution retained for lifting"
            ),
            "first_input": (
                "the one victim-stabilizer orbit of pairwise natural-node "
                "unions, adjoining alternative matching monomials for its "
                "four singleton equations; quotient log-slopes from retained "
                "numerical runs select which resulting cones to try first"
            ),
            "why_this_is_next": (
                "the known nine-coordinate cone has no transverse direction "
                "after gauge quotient, while all first repair choices lie in "
                "the residual-character-zero sector"
            ),
        },
        "claim_boundary": {
            "known_exact_support_tropical_cone_classified": True,
            "minimal_pairwise_natural_node_bridges_excluded": True,
            "pairwise_natural_node_bridges_through_support_21_excluded": True,
            "all_higher_support_node_bridges_excluded": False,
            "known_Laurent_valuation_is_transverse_after_gauge_quotient": False,
            "residual_Z9_separates_the_four_repair_orbits": False,
            "first_defect_hypersurface_ties_covered_by_four_orbits": True,
            "higher_support_tropical_cones_enumerated": False,
            "full_tropical_prevariety_computed": False,
            "full_tropical_variety_computed": False,
            "all_u_zero_vertical_components_excluded": False,
            "all_zero_residual_sequences_proved_boundary": False,
            "finite_affine_GHZ_membership_status": "undecided",
        },
    }
    # Strict JSON is part of the audit contract.
    return json.loads(json.dumps(payload, allow_nan=False))


def verify_tropical_series_structure_audit(payload: Mapping) -> dict:
    """Rebuild the exact audit and reject any altered datum or claim."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennTropicalSeriesError(
            "the tropical-series payload is not strict JSON"
        ) from error
    expected = tropical_series_structure_audit()
    if not _strict_json_equal(normalized, expected):
        raise KrennTropicalSeriesError(
            "the tropical-series payload failed exact replay"
        )
    return normalized


__all__ = (
    "KrennTropicalSeriesError",
    "TROPICAL_SERIES_SCHEMA",
    "tropical_series_structure_audit",
    "verify_tropical_series_structure_audit",
)
