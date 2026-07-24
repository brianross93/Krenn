"""Exact symmetry and matching-circuit structure of the natural pole.

This module separates three statements which are easy to conflate.

* The stabilizer of the victim coloring transports the known Laurent
  degeneration into four vertex-scalar-gauge-invariant pole charts.  If
  ``Q_0,...,Q_3`` are the resulting monomials and ``S=sum(Q_i)``, then every
  literal symmetry copy of the natural path satisfies ``t*S=1`` exactly.
  Under the larger color-diagonal gauge, each ``Q_i`` instead carries the
  inverse victim character, so the invariant quantity is ``t*Q_i``.
* The ten cubic ``K_{3,3}`` parity circuits span the complete integral
  kernel of the uncolored perfect-matching incidence map after Laurent
  localization.
* The tempting homogeneous identity

      F_v * S = D

  is not a global polynomial identity.  Its exact expansion has 56 spill
  terms, and the established exact ``k=1`` dual proves that no degree-six
  mixed-generator correction can turn this ansatz into an identity for
  ``D``.

The first statement is a symmetry-complete law on the known natural
branches, not on every component of the moving fiber.  The second is a
localized multiplicative statement and does not cover coordinate strata
where matching terms vanish.  The final audit therefore keeps the global
radical-membership and affine-GHZ questions explicitly undecided.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, combinations_with_replacement

from experiments.krenn_quantum_graph.border_image import (
    LAURENT_ONE,
    LAURENT_T,
    LaurentPolynomial,
    evaluate_laurent_system,
    natural_laurent_weight_entries,
)
from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
)
from experiments.krenn_quantum_graph.source_ideal import (
    _group_actions,
    _transport_coloring,
    _transport_monomial,
    d_monomials,
    exact_two_row_dual,
)
from experiments.krenn_quantum_graph.system import (
    Monomial,
    canonical_edges,
    coloring_from_index,
    generate_sparse_system,
    perfect_matchings,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
)


MATCHING_CIRCUIT_STRUCTURE_SCHEMA = (
    "krenn.n6_d3.matching_circuit_structure.v1"
)

N = 6
D = 3
VICTIM_COLORING = (0, 0, 2, 1, 2, 1)

EXPECTED_VICTIM_STABILIZER_ACTIONS = 48
EXPECTED_LITERAL_NATURAL_PATHS = 24
EXPECTED_Q_ORBIT: tuple[Monomial, ...] = (
    (13, 26, 67, 80, 81, 126),
    (13, 44, 62, 67, 99, 108),
    (26, 31, 49, 80, 99, 108),
    (31, 44, 49, 62, 81, 126),
)

EXPECTED_MATCHING_INCIDENCE_RANK = 10
EXPECTED_MATCHING_LATTICE_RANK = 5
EXPECTED_K33_CIRCUITS = 10
UNIMODULAR_CIRCUIT_ROWS = (0, 1, 2, 4, 5)
UNIMODULAR_MATCHING_COLUMNS = (0, 1, 3, 4, 7)
EXPECTED_UNIMODULAR_DETERMINANT = 1

EXPECTED_D_TERMS = 3_375
EXPECTED_NAIVE_TERMS = 60
EXPECTED_NAIVE_D_OVERLAP = 4
EXPECTED_NAIVE_SPILLS = 56
EXPECTED_NAIVE_SPILL_ORBITS = 4
EXPECTED_RESIDUAL_TERMS = 3_427
EXPECTED_RESIDUAL_COEFFICIENTS = {-1: 56, 1: 3_371}


class KrennMatchingCircuitStructureError(RuntimeError):
    """A symmetry, circuit-lattice, or global-ansatz replay failed."""


GroupAction = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]


def _victim_coloring() -> tuple[int, ...]:
    victim = coloring_from_index(
        N, D, N6_D3_SEED_DEFECT_EQUATION
    )
    if victim != VICTIM_COLORING:
        raise KrennMatchingCircuitStructureError(
            "the victim-coloring convention changed"
        )
    return victim


@lru_cache(maxsize=1)
def victim_stabilizer_actions() -> tuple[GroupAction, ...]:
    """Return the exact ``S6 x S3`` stabilizer of the victim coloring."""

    victim = _victim_coloring()
    actions = tuple(
        action
        for action in _group_actions()
        if _transport_coloring(
            victim,
            action[0],
            action[1],
        )
        == victim
    )
    if len(actions) != EXPECTED_VICTIM_STABILIZER_ACTIONS:
        raise KrennMatchingCircuitStructureError(
            "the victim stabilizer no longer has 48 actions"
        )
    return actions


@lru_cache(maxsize=1)
def pole_invariant_orbit() -> tuple[Monomial, ...]:
    """Return the four victim-stabilizer transforms of the pole monomial."""

    orbit = tuple(sorted({
        _transport_monomial(
            POLE_INVARIANT_INDICES,
            action[2],
        )
        for action in victim_stabilizer_actions()
    }))
    if orbit != EXPECTED_Q_ORBIT:
        raise KrennMatchingCircuitStructureError(
            "the four pole monomials changed"
        )
    return orbit


def _vertex_degrees(monomial: Sequence[int]) -> tuple[int, ...]:
    degrees = [0] * N
    for variable in monomial:
        left, right, _left_color, _right_color = variable_key(
            N, D, int(variable)
        )
        degrees[left] += 1
        degrees[right] += 1
    return tuple(degrees)


def _laurent_product(
    values: Mapping[int, LaurentPolynomial],
    monomial: Sequence[int],
) -> LaurentPolynomial:
    result = LAURENT_ONE
    for variable in monomial:
        value = values.get(int(variable))
        if value is None:
            return LaurentPolynomial()
        result *= value
    return result


def _transport_laurent_entries(
    action: GroupAction,
) -> tuple[tuple[int, LaurentPolynomial], ...]:
    variable_permutation = action[2]
    return tuple(sorted(
        (
            variable_permutation[index],
            value,
        )
        for index, value in natural_laurent_weight_entries()
    ))


def _path_key(
    entries: Sequence[tuple[int, LaurentPolynomial]],
) -> tuple[tuple[int, tuple[tuple[int, Fraction], ...]], ...]:
    return tuple(
        (index, value.terms)
        for index, value in entries
    )


def _expected_moving_output() -> tuple[LaurentPolynomial, ...]:
    system = generate_sparse_system(N, D)
    output = tuple(
        LaurentPolynomial.constant(value)
        + (
            LAURENT_T
            if equation == N6_D3_SEED_DEFECT_EQUATION
            else LaurentPolynomial()
        )
        for equation, value in enumerate(system.rhs_values)
    )
    return output


@lru_cache(maxsize=1)
def victim_symmetry_pole_audit() -> Mapping[str, object]:
    """Replay ``t*(Q_0+...+Q_3)=1`` on every natural symmetry path."""

    actions = victim_stabilizer_actions()
    q_orbit = pole_invariant_orbit()
    q_set = set(q_orbit)
    q_vertex_degrees = tuple(
        _vertex_degrees(monomial) for monomial in q_orbit
    )
    gauge_invariant = all(
        degrees == (2,) * N for degrees in q_vertex_degrees
    )

    stabilizer_preserves_sum = all(
        {
            _transport_monomial(monomial, action[2])
            for monomial in q_orbit
        }
        == q_set
        for action in actions
    )
    if not stabilizer_preserves_sum or not gauge_invariant:
        raise KrennMatchingCircuitStructureError(
            "the pole orbit lost a gauge or victim-stabilizer invariant"
        )

    path_actions: defaultdict[
        tuple[tuple[int, tuple[tuple[int, Fraction], ...]], ...],
        list[GroupAction],
    ] = defaultdict(list)
    path_entries = {}
    for action in actions:
        entries = _transport_laurent_entries(action)
        key = _path_key(entries)
        path_actions[key].append(action)
        path_entries[key] = entries

    if (
        len(path_entries) != EXPECTED_LITERAL_NATURAL_PATHS
        or Counter(map(len, path_actions.values())) != Counter({2: 24})
    ):
        raise KrennMatchingCircuitStructureError(
            "the literal natural-path orbit changed"
        )

    system = generate_sparse_system(N, D)
    expected_output = _expected_moving_output()
    active_q_counts: Counter[int] = Counter()
    patterns = set()
    all_outputs_exact = True
    all_pole_laws_exact = True
    for entries in path_entries.values():
        values = dict(entries)
        q_values = tuple(
            _laurent_product(values, monomial)
            for monomial in q_orbit
        )
        active = tuple(
            index
            for index, value in enumerate(q_values)
            if not value.is_zero
        )
        if len(active) != 1:
            raise KrennMatchingCircuitStructureError(
                "a natural path activates the wrong number of pole charts"
            )
        active_q_counts[active[0]] += 1
        pole_sum = sum(q_values, LaurentPolynomial())
        all_pole_laws_exact &= (
            q_values[active[0]]
            == LaurentPolynomial.monomial(-1)
            and pole_sum == LaurentPolynomial.monomial(-1)
            and LAURENT_T * pole_sum == LAURENT_ONE
        )
        patterns.add(tuple(
            value.to_expression() for value in q_values
        ))
        all_outputs_exact &= (
            evaluate_laurent_system(system, entries)
            == expected_output
        )

    if (
        active_q_counts != Counter({0: 6, 1: 6, 2: 6, 3: 6})
        or len(patterns) != 4
        or not all_outputs_exact
        or not all_pole_laws_exact
    ):
        raise KrennMatchingCircuitStructureError(
            "a transformed natural branch failed exact replay"
        )

    return {
        "victim_equation": N6_D3_SEED_DEFECT_EQUATION,
        "victim_coloring": list(_victim_coloring()),
        "stabilizer_actions": len(actions),
        "pole_monomials": [list(monomial) for monomial in q_orbit],
        "pole_monomial_variable_keys": [
            [list(variable_key(N, D, variable)) for variable in monomial]
            for monomial in q_orbit
        ],
        "pole_monomial_vertex_degrees": [
            list(degrees) for degrees in q_vertex_degrees
        ],
        "each_Q_is_product_one_vertex_gauge_invariant": gauge_invariant,
        "stabilizer_preserves_S": stabilizer_preserves_sum,
        "S_definition": "Q0 + Q1 + Q2 + Q3",
        "literal_natural_paths": len(path_entries),
        "actions_per_literal_path": 2,
        "literal_paths_per_active_Q": {
            str(index): active_q_counts[index]
            for index in range(len(q_orbit))
        },
        "cross_evaluation_patterns": [
            list(pattern) for pattern in sorted(patterns)
        ],
        "all_729_outputs_equal_GHZ_plus_t_victim": all_outputs_exact,
        "S_on_every_literal_path": "t^-1",
        "t_times_S_on_every_literal_path": "1",
        "local_branch_law_exact": all_pole_laws_exact,
        "scope": (
            "all victim-stabilizer copies of the known natural Laurent "
            "branch; no claim about other moving-fiber components"
        ),
    }


def _matching_incidence(
    matching_indices: Sequence[int],
) -> tuple[int, ...]:
    edges = canonical_edges(N)
    matchings = perfect_matchings(N)
    return tuple(
        sum(
            edge in matchings[int(index)]
            for index in matching_indices
        )
        for edge in edges
    )


def _rank_over_q(
    matrix: Sequence[Sequence[int]],
) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennMatchingCircuitStructureError(
            "exact rank needs a rectangular matrix"
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
        pivot_value = rows[rank][column]
        rows[rank] = [
            value / pivot_value for value in rows[rank]
        ]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                left - factor * right
                for left, right in zip(
                    rows[row], rows[rank], strict=True
                )
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _bareiss_determinant(
    matrix: Sequence[Sequence[int]],
) -> int:
    rows = [list(map(int, row)) for row in matrix]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise KrennMatchingCircuitStructureError(
            "Bareiss determinant needs a square matrix"
        )
    if size == 0:
        return 1
    sign = 1
    previous = 1
    for pivot_index in range(size - 1):
        pivot_row = next(
            (
                row
                for row in range(pivot_index, size)
                if rows[row][pivot_index]
            ),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            rows[pivot_index], rows[pivot_row] = (
                rows[pivot_row],
                rows[pivot_index],
            )
            sign *= -1
        pivot = rows[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = (
                    rows[row][column] * pivot
                    - rows[row][pivot_index]
                    * rows[pivot_index][column]
                )
                if numerator % previous:
                    raise KrennMatchingCircuitStructureError(
                        "Bareiss division was not exact"
                    )
                rows[row][column] = numerator // previous
        previous = pivot
    return sign * rows[-1][-1]


def _k33_bipartition(
    incidence: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    edges = canonical_edges(N)
    support = {
        edge
        for edge, value in zip(edges, incidence, strict=True)
        if value
    }
    if (
        len(incidence) != len(edges)
        or set(incidence) != {0, 1}
        or len(support) != 9
    ):
        raise KrennMatchingCircuitStructureError(
            "a cubic collision does not have nine simple edges"
        )
    for first in combinations(range(N), 3):
        if 0 not in first:
            continue
        second = tuple(
            vertex for vertex in range(N) if vertex not in first
        )
        crossing = {
            tuple(sorted((left, right)))
            for left in first
            for right in second
        }
        if crossing == support:
            return tuple(first), second
    raise KrennMatchingCircuitStructureError(
        "a cubic collision is not a K3,3"
    )


def _matching_parity(
    matching_index: int,
    bipartition: tuple[Sequence[int], Sequence[int]],
) -> int:
    left, right = bipartition
    right_position = {
        vertex: index for index, vertex in enumerate(right)
    }
    partners = {}
    for first, second in perfect_matchings(N)[matching_index]:
        partners[first] = second
        partners[second] = first
    if any(partners[vertex] not in right_position for vertex in left):
        raise KrennMatchingCircuitStructureError(
            "a K3,3 parity matching uses an internal edge"
        )
    permutation = tuple(
        right_position[partners[vertex]]
        for vertex in left
    )
    return sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    ) % 2


@lru_cache(maxsize=1)
def matching_circuit_lattice_audit() -> Mapping[str, object]:
    """Certify the complete integral matching-incidence lattice."""

    matchings = perfect_matchings(N)
    edges = canonical_edges(N)
    incidence_matrix = tuple(
        tuple(int(edge in matching) for edge in edges)
        for matching in matchings
    )
    incidence_rank = _rank_over_q(incidence_matrix)

    quadratic_fibers: defaultdict[
        tuple[int, ...],
        list[tuple[int, ...]],
    ] = defaultdict(list)
    for multiset in combinations_with_replacement(
        range(len(matchings)), 2
    ):
        quadratic_fibers[_matching_incidence(multiset)].append(multiset)

    cubic_fibers: defaultdict[
        tuple[int, ...],
        list[tuple[int, ...]],
    ] = defaultdict(list)
    for multiset in combinations_with_replacement(
        range(len(matchings)), 3
    ):
        cubic_fibers[_matching_incidence(multiset)].append(multiset)
    collision_items = tuple(
        (incidence, tuple(fiber))
        for incidence, fiber in sorted(cubic_fibers.items())
        if len(fiber) > 1
    )

    circuit_rows = []
    circuit_payload = []
    bipartitions = set()
    for incidence, fiber in collision_items:
        if (
            len(fiber) != 2
            or any(len(multiset) != 3 for multiset in fiber)
        ):
            raise KrennMatchingCircuitStructureError(
                "a cubic collision fiber changed shape"
            )
        bipartition = _k33_bipartition(incidence)
        bipartitions.add(bipartition)
        crossing = tuple(
            index
            for index in range(len(matchings))
            if all(
                (left in bipartition[0])
                != (right in bipartition[0])
                for left, right in matchings[index]
            )
        )
        parity_classes = tuple(
            tuple(
                index
                for index in crossing
                if _matching_parity(index, bipartition) == parity
            )
            for parity in (0, 1)
        )
        if (
            len(crossing) != 6
            or tuple(map(len, parity_classes)) != (3, 3)
            or set(fiber) != set(parity_classes)
        ):
            raise KrennMatchingCircuitStructureError(
                "a collision is not the two K3,3 parity classes"
            )
        relation = tuple(
            1
            if index in parity_classes[0]
            else -1
            if index in parity_classes[1]
            else 0
            for index in range(len(matchings))
        )
        image = tuple(
            sum(
                relation[row] * incidence_matrix[row][column]
                for row in range(len(matchings))
            )
            for column in range(len(edges))
        )
        if any(image):
            raise KrennMatchingCircuitStructureError(
                "a parity circuit left the incidence kernel"
            )
        circuit_rows.append(relation)
        circuit_payload.append({
            "bipartition": [
                list(bipartition[0]),
                list(bipartition[1]),
            ],
            "even_matching_indices": list(parity_classes[0]),
            "odd_matching_indices": list(parity_classes[1]),
            "relation_coefficients": list(relation),
        })

    circuit_rank = _rank_over_q(circuit_rows)
    minor = tuple(
        tuple(
            circuit_rows[row][column]
            for column in UNIMODULAR_MATCHING_COLUMNS
        )
        for row in UNIMODULAR_CIRCUIT_ROWS
    )
    determinant = _bareiss_determinant(minor)
    expected_bipartitions = {
        (
            tuple(first),
            tuple(
                vertex for vertex in range(N) if vertex not in first
            ),
        )
        for first in combinations(range(N), 3)
        if 0 in first
    }
    quadratic_collision_count = sum(
        len(fiber) > 1 for fiber in quadratic_fibers.values()
    )
    # The integer kernel of an integer matrix is saturated.  The circuit
    # rows lie in that kernel and have its full rank five.  A unit maximal
    # minor makes their row lattice saturated in Z^15 as well, so the two
    # rank-five lattices coincide, not merely their Q-spans.
    complete = (
        len(matchings) == 15
        and incidence_rank == EXPECTED_MATCHING_INCIDENCE_RANK
        and len(quadratic_fibers) == 120
        and quadratic_collision_count == 0
        and len(cubic_fibers) == 670
        and len(collision_items) == EXPECTED_K33_CIRCUITS
        and Counter(map(len, cubic_fibers.values()))
        == Counter({1: 660, 2: 10})
        and bipartitions == expected_bipartitions
        and circuit_rank == EXPECTED_MATCHING_LATTICE_RANK
        and determinant == EXPECTED_UNIMODULAR_DETERMINANT
    )
    if not complete:
        raise KrennMatchingCircuitStructureError(
            "the exact matching-circuit lattice audit failed"
        )

    return {
        "perfect_matchings": len(matchings),
        "matching_incidence_shape": [len(matchings), len(edges)],
        "matching_incidence_rank_over_Q": incidence_rank,
        "integral_kernel_rank": len(matchings) - incidence_rank,
        "quadratic_multisets": 120,
        "quadratic_collision_fibers": quadratic_collision_count,
        "cubic_multisets": 680,
        "cubic_incidence_sums": len(cubic_fibers),
        "cubic_collision_fibers": len(collision_items),
        "circuits": circuit_payload,
        "circuit_relation_rank_over_Q": circuit_rank,
        "unimodular_minor": {
            "circuit_rows": list(UNIMODULAR_CIRCUIT_ROWS),
            "matching_columns": list(UNIMODULAR_MATCHING_COLUMNS),
            "matrix": [list(row) for row in minor],
            "determinant": determinant,
        },
        "integral_kernel_generated_by_K3_3_circuits": True,
        "localized_laurent_relation_lattice_complete": True,
        "polynomial_toric_ideal_generation_claimed": False,
        "scope": (
            "multiplicative relations among the 15 matching terms of one "
            "fixed coloring after Laurent localization"
        ),
    }


@lru_cache(maxsize=None)
def _canonical_full_group_monomial(
    monomial: Monomial,
) -> Monomial:
    return min(
        _transport_monomial(monomial, action[2])
        for action in _group_actions()
    )


@lru_cache(maxsize=1)
def naive_global_identity_audit() -> Mapping[str, object]:
    """Refute the naive global polynomial identity ``F_v*S=D``."""

    system = generate_sparse_system(N, D)
    victim_terms = system.equation_monomials(
        N6_D3_SEED_DEFECT_EQUATION
    )
    q_orbit = pole_invariant_orbit()

    naive = Counter(
        tuple(sorted((*q_monomial, *victim_term)))
        for q_monomial in q_orbit
        for victim_term in victim_terms
    )
    d_expansion = Counter(d_monomials())
    overlap = set(naive) & set(d_expansion)
    spills = tuple(sorted(set(naive) - set(d_expansion)))
    spill_orbits = {
        _canonical_full_group_monomial(monomial)
        for monomial in spills
    }
    residual = d_expansion.copy()
    residual.subtract(naive)
    residual = Counter({
        monomial: coefficient
        for monomial, coefficient in residual.items()
        if coefficient
    })

    dual = exact_two_row_dual()
    k1_negative = dual.exact_Q_nonmembership_proved
    exact = (
        len(d_expansion) == EXPECTED_D_TERMS
        and len(naive) == EXPECTED_NAIVE_TERMS
        and Counter(naive.values()) == Counter({1: 60})
        and len(overlap) == EXPECTED_NAIVE_D_OVERLAP
        and len(spills) == EXPECTED_NAIVE_SPILLS
        and len(spill_orbits) == EXPECTED_NAIVE_SPILL_ORBITS
        and len(residual) == EXPECTED_RESIDUAL_TERMS
        and dict(sorted(Counter(residual.values()).items()))
        == EXPECTED_RESIDUAL_COEFFICIENTS
        and k1_negative
        and dual.lambda_transpose_b == -720
    )
    if not exact:
        raise KrennMatchingCircuitStructureError(
            "the naive global-identity audit changed"
        )

    return {
        "ansatz": "F_victim * (Q0 + Q1 + Q2 + Q3) = D",
        "ordinary_degree": 9,
        "D_expansion_terms": len(d_expansion),
        "F_victim_times_S_raw_terms": (
            len(victim_terms) * len(q_orbit)
        ),
        "F_victim_times_S_distinct_terms": len(naive),
        "terms_overlapping_D_support": len(overlap),
        "outside_D_support_spill_terms": len(spills),
        "spill_full_group_orbits": len(spill_orbits),
        "spill_full_group_orbit_representatives": [
            list(monomial) for monomial in sorted(spill_orbits)
        ],
        "residual_nonzero_terms": len(residual),
        "residual_coefficient_census": {
            str(coefficient): count
            for coefficient, count in sorted(
                Counter(residual.values()).items()
            )
        },
        "naive_global_polynomial_identity_holds": False,
        "naive_global_polynomial_identity_refuted_exactly": True,
        "F_victim_times_S_is_in_J_mix": True,
        "k1_exact_dual_lambda_transpose_b": dual.lambda_transpose_b,
        "D_not_in_J_mix_at_k1_replayed": k1_negative,
        "degree_six_mixed_multiplier_repair_exists": False,
        "set_theoretic_or_higher_power_relation_decided": False,
        "global_GHZ_nonexistence_proved": False,
        "claim_boundary": (
            "The local symmetry law cannot be promoted to the displayed "
            "degree-nine polynomial identity, even after adding arbitrary "
            "degree-six multiples of mixed generators.  This does not "
            "decide radical membership or exclude a higher-degree law."
        ),
    }


@lru_cache(maxsize=1)
def exact_matching_circuit_structure() -> Mapping[str, object]:
    """Return the complete fail-closed structural preflight."""

    symmetry = victim_symmetry_pole_audit()
    circuits = matching_circuit_lattice_audit()
    global_ansatz = naive_global_identity_audit()
    return {
        "schema": MATCHING_CIRCUIT_STRUCTURE_SCHEMA,
        "victim_symmetry_pole": symmetry,
        "matching_circuit_lattice": circuits,
        "naive_global_identity": global_ansatz,
        "claims": {
            "natural_symmetry_branches_obey_t_times_S_equals_1": True,
            "fixed_coloring_localized_matching_lattice_complete": True,
            "naive_degree_nine_global_identity_exists": False,
            "global_moving_fiber_pole_law_proved": False,
            "D_in_radical_J_mix_decided": False,
            "exact_affine_GHZ_membership_decided": False,
        },
    }


__all__ = [
    "EXPECTED_Q_ORBIT",
    "KrennMatchingCircuitStructureError",
    "MATCHING_CIRCUIT_STRUCTURE_SCHEMA",
    "VICTIM_COLORING",
    "exact_matching_circuit_structure",
    "matching_circuit_lattice_audit",
    "naive_global_identity_audit",
    "pole_invariant_orbit",
    "victim_stabilizer_actions",
    "victim_symmetry_pole_audit",
]
