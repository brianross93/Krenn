"""Exact matching-circuit audit behind the known support obstructions.

This module isolates a small combinatorial mechanism that occurs in several
``n=6,d=3`` computations.  It proves three deliberately limited statements.

* The 15 perfect matchings of ``K_6`` have no quadratic incidence collision.
* Their first incidence collisions occur for cubic multisets.  There are ten,
  one for each unordered ``3+3`` bipartition, and every collision is the two
  parity classes of the six perfect matchings of the resulting ``K_{3,3}``.
* The first recorded support-21 contradiction is one such cubic circuit.
  Its three binomial equations give an exact identity after setting source
  variables outside that 21-coordinate support to zero.  In the unrestricted
  source ring the same combination has 39 distinct spill terms.

The last statement is support-conditional.  The 39 omitted degree-three
generator terms are defined by the *source-coordinate support*, not by
``supp(D^2)``.  Nothing here decides ``D^2 in J_mix``, radical membership, or
global affine GHZ membership.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, combinations_with_replacement
import json
from math import gcd
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.support_extension import (
    size21_support_audits,
)
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    generate_sparse_system,
    perfect_matchings,
)


N = 6
D = 3
EDGE_COUNT = 15
PERFECT_MATCHING_COUNT = 15
QUADRATIC_MULTISET_COUNT = 120
CUBIC_MULTISET_COUNT = 680
CUBIC_INCIDENCE_SUM_COUNT = 670
K33_COLLISION_FIBER_COUNT = 10

MECHANISM_AUDIT_SCHEMA = "krenn.n6_d3.mechanism_audit.v1"

EXPECTED_LOCAL_SUPPORT = (
    0,
    6,
    13,
    18,
    20,
    24,
    26,
    29,
    35,
    45,
    47,
    67,
    80,
    81,
    83,
    87,
    89,
    92,
    98,
    121,
    126,
)
EXPECTED_LOCAL_EQUATIONS = (16, 18, 188)
EXPECTED_LOCAL_COLORINGS = (
    (0, 0, 0, 1, 2, 1),
    (0, 0, 0, 2, 0, 0),
    (0, 2, 0, 2, 2, 2),
)
EXPECTED_LOCAL_MATCHING_PAIRS = ((1, 9), (0, 6), (8, 11))
EXPECTED_LOCAL_SIGNS = (1, -1, -1)
EXPECTED_LOCAL_ACTIVE_MONOMIALS = (
    ((0, 92, 121), (29, 45, 121)),
    ((0, 83, 126), (20, 45, 126)),
    ((20, 80, 92), (29, 80, 83)),
)
EXPECTED_LOCAL_BIPARTITION = ((0, 2, 5), (1, 3, 4))


class KrennMechanismAuditError(ValueError):
    """An exact matching or local-support receipt failed replay."""


def _canonical_edges() -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(N), 2))


@lru_cache(maxsize=1)
def enumerate_k6_perfect_matchings(
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Enumerate the 15 perfect matchings without using stored fixtures."""

    def recurse(
        vertices: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        if not vertices:
            return ((),)
        first = vertices[0]
        result = []
        for position in range(1, len(vertices)):
            partner = vertices[position]
            remaining = vertices[1:position] + vertices[position + 1 :]
            for tail in recurse(remaining):
                result.append(((first, partner), *tail))
        return tuple(result)

    result = tuple(sorted(recurse(tuple(range(N)))))
    if (
        len(result) != PERFECT_MATCHING_COUNT
        or len(set(result)) != PERFECT_MATCHING_COUNT
        or any(
            len(matching) != 3
            or {vertex for edge in matching for vertex in edge}
            != set(range(N))
            for matching in result
        )
    ):
        raise KrennMechanismAuditError(
            "the recursive K6 perfect-matching enumeration failed"
        )
    return result


def _incidence_vector(
    matching_indices: Sequence[int],
) -> tuple[int, ...]:
    edges = _canonical_edges()
    matchings = enumerate_k6_perfect_matchings()
    return tuple(
        sum(edge in matchings[index] for index in matching_indices)
        for edge in edges
    )


def _incidence_fibers(
    degree: int,
) -> dict[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    if degree not in (2, 3):
        raise KrennMechanismAuditError(
            "only the audited quadratic and cubic degrees are supported"
        )
    fibers: defaultdict[tuple[int, ...], list[tuple[int, ...]]] = (
        defaultdict(list)
    )
    for matching_multiset in combinations_with_replacement(
        range(PERFECT_MATCHING_COUNT), degree
    ):
        fibers[_incidence_vector(matching_multiset)].append(
            matching_multiset
        )
    return {
        incidence: tuple(multisets)
        for incidence, multisets in sorted(fibers.items())
    }


def _canonical_fiber_payload(
    fibers: Mapping[
        tuple[int, ...], Sequence[Sequence[int]]
    ],
) -> list[dict]:
    return [
        {
            "incidence_sum": list(incidence),
            "matching_multisets": [
                list(multiset) for multiset in multisets
            ],
        }
        for incidence, multisets in sorted(fibers.items())
    ]


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _permutation_parity(permutation: Sequence[int]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return inversions % 2


def _k33_bipartition(
    incidence: Sequence[int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    edges = _canonical_edges()
    support_edges = {
        edge for edge, multiplicity in zip(edges, incidence, strict=True)
        if multiplicity
    }
    if (
        len(incidence) != EDGE_COUNT
        or set(incidence) != {0, 1}
        or len(support_edges) != 9
    ):
        raise KrennMechanismAuditError(
            "a cubic collision does not have a simple nine-edge support"
        )
    adjacency = {vertex: set() for vertex in range(N)}
    for left, right in support_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    colors = {0: 0}
    frontier = [0]
    while frontier:
        vertex = frontier.pop()
        for neighbor in adjacency[vertex]:
            proposed = 1 - colors[vertex]
            if neighbor in colors and colors[neighbor] != proposed:
                raise KrennMechanismAuditError(
                    "a cubic collision support is not bipartite"
                )
            if neighbor not in colors:
                colors[neighbor] = proposed
                frontier.append(neighbor)
    if len(colors) != N:
        raise KrennMechanismAuditError(
            "a cubic collision support is disconnected"
        )
    first = tuple(
        vertex for vertex in range(N) if colors[vertex] == colors[0]
    )
    second = tuple(vertex for vertex in range(N) if vertex not in first)
    complete_bipartite_edges = {
        tuple(sorted((left, right)))
        for left in first
        for right in second
    }
    if (
        len(first) != 3
        or len(second) != 3
        or support_edges != complete_bipartite_edges
    ):
        raise KrennMechanismAuditError(
            "a cubic collision support is not K3,3"
        )
    return first, second


def _matching_parity_across_bipartition(
    matching_index: int,
    bipartition: tuple[Sequence[int], Sequence[int]],
) -> int:
    left, right = bipartition
    right_positions = {
        vertex: position for position, vertex in enumerate(right)
    }
    matching = enumerate_k6_perfect_matchings()[matching_index]
    partners = {}
    for first, second in matching:
        partners[first] = second
        partners[second] = first
    if any(partners[vertex] not in right_positions for vertex in left):
        raise KrennMechanismAuditError(
            "a claimed K3,3 matching uses an internal edge"
        )
    permutation = tuple(
        right_positions[partners[vertex]] for vertex in left
    )
    return _permutation_parity(permutation)


def _switch_payload(
    first: int,
    second: int,
) -> dict:
    matchings = enumerate_k6_perfect_matchings()
    first_edges = set(matchings[first])
    second_edges = set(matchings[second])
    shared = sorted(first_edges & second_edges)
    cycle = sorted(first_edges ^ second_edges)
    difference = tuple(
        int(edge in first_edges) - int(edge in second_edges)
        for edge in _canonical_edges()
    )
    if (
        len(shared) != 1
        or len(cycle) != 4
        or Counter(vertex for edge in cycle for vertex in edge)
        != Counter(
            {
                vertex: 2
                for vertex in {
                    item for edge in cycle for item in edge
                }
            }
        )
        or Counter(difference) != Counter({0: 11, 1: 2, -1: 2})
    ):
        raise KrennMechanismAuditError(
            "a parity-class pairing is not a four-cycle switch"
        )
    return {
        "matching_indices": [first, second],
        "common_edge": list(shared[0]),
        "four_cycle_edges": [list(edge) for edge in cycle],
        "signed_incidence_difference": list(difference),
    }


def _collision_payload(
    incidence: tuple[int, ...],
    fiber: tuple[tuple[int, ...], tuple[int, ...]],
) -> dict:
    if (
        len(fiber) != 2
        or any(len(multiset) != 3 for multiset in fiber)
        or any(len(set(multiset)) != 3 for multiset in fiber)
        or set(fiber[0]) & set(fiber[1])
    ):
        raise KrennMechanismAuditError(
            "a cubic collision is not a disjoint 3-versus-3 circuit"
        )
    bipartition = _k33_bipartition(incidence)
    all_crossing_matchings = tuple(
        index
        for index in range(PERFECT_MATCHING_COUNT)
        if all(
            (left in bipartition[0]) != (right in bipartition[0])
            for left, right in enumerate_k6_perfect_matchings()[index]
        )
    )
    parity_classes = {
        parity: tuple(
            index
            for index in all_crossing_matchings
            if _matching_parity_across_bipartition(
                index, bipartition
            )
            == parity
        )
        for parity in (0, 1)
    }
    if (
        len(all_crossing_matchings) != 6
        or set(map(tuple, fiber))
        != {parity_classes[0], parity_classes[1]}
        or _incidence_vector(parity_classes[0])
        != _incidence_vector(parity_classes[1])
    ):
        raise KrennMechanismAuditError(
            "a collision fiber is not the two K3,3 parity classes"
        )

    # Sorted parity classes give a deterministic pairing.  Each pair differs
    # by one transposition, hence is a four-cycle switch with one common edge.
    switches = tuple(
        _switch_payload(first, second)
        for first, second in zip(
            parity_classes[0], parity_classes[1], strict=True
        )
    )
    switch_sum = tuple(
        sum(
            switch["signed_incidence_difference"][edge]
            for switch in switches
        )
        for edge in range(EDGE_COUNT)
    )
    relation_coefficients = [
        -1 if index in parity_classes[1] else 1
        if index in parity_classes[0] else 0
        for index in range(PERFECT_MATCHING_COUNT)
    ]
    nonzero_coefficients = [
        abs(value) for value in relation_coefficients if value
    ]
    proper_balanced_subrelation_exists = any(
        _incidence_vector(first_subset)
        == _incidence_vector(second_subset)
        for subset_size in (1, 2)
        for first_subset in combinations(
            parity_classes[0], subset_size
        )
        for second_subset in combinations(
            parity_classes[1], subset_size
        )
    )
    if (
        any(switch_sum)
        or gcd(*nonzero_coefficients) != 1
        or len(nonzero_coefficients) != 6
        or proper_balanced_subrelation_exists
    ):
        raise KrennMechanismAuditError(
            "the three-switch relation is not a primitive circuit"
        )
    return {
        "bipartition": [
            list(bipartition[0]),
            list(bipartition[1]),
        ],
        "incidence_sum": list(incidence),
        "incidence_edges": [
            list(edge)
            for edge, value in zip(
                _canonical_edges(), incidence, strict=True
            )
            if value
        ],
        "collision_fiber": [
            list(multiset) for multiset in fiber
        ],
        "even_parity_matching_indices": list(parity_classes[0]),
        "odd_parity_matching_indices": list(parity_classes[1]),
        "relation_coefficients_in_matching_order": relation_coefficients,
        "canonical_three_switches": list(switches),
        "signed_switch_sum": list(switch_sum),
        "proper_balanced_subrelation_exists": False,
        "primitive_circuit": True,
    }


def matching_collision_audit() -> dict:
    """Return the complete exact degree-two/degree-three incidence audit."""

    matchings = enumerate_k6_perfect_matchings()
    system_matchings = tuple(perfect_matchings(N))
    if matchings != system_matchings:
        raise KrennMechanismAuditError(
            "the independent and system perfect-matching orders disagree"
        )

    quadratic = _incidence_fibers(2)
    cubic = _incidence_fibers(3)
    quadratic_payload = _canonical_fiber_payload(quadratic)
    cubic_payload = _canonical_fiber_payload(cubic)
    collision_items = tuple(
        (incidence, fiber)
        for incidence, fiber in cubic.items()
        if len(fiber) > 1
    )
    collisions = tuple(
        _collision_payload(incidence, fiber)
        for incidence, fiber in collision_items
    )
    bipartitions = {
        tuple(map(tuple, collision["bipartition"]))
        for collision in collisions
    }
    expected_bipartitions = {
        (
            tuple(sorted(first)),
            tuple(vertex for vertex in range(N) if vertex not in first),
        )
        for first in combinations(range(N), 3)
        if 0 in first
    }
    if (
        sum(len(fiber) for fiber in quadratic.values())
        != QUADRATIC_MULTISET_COUNT
        or len(quadratic) != QUADRATIC_MULTISET_COUNT
        or set(map(len, quadratic.values())) != {1}
        or sum(len(fiber) for fiber in cubic.values())
        != CUBIC_MULTISET_COUNT
        or len(cubic) != CUBIC_INCIDENCE_SUM_COUNT
        or len(collisions) != K33_COLLISION_FIBER_COUNT
        or sorted(map(len, (fiber for _, fiber in collision_items)))
        != [2] * K33_COLLISION_FIBER_COUNT
        or Counter(map(len, cubic.values()))
        != Counter({1: 660, 2: 10})
        or bipartitions != expected_bipartitions
    ):
        raise KrennMechanismAuditError(
            "the matching-incidence collision census changed"
        )
    return {
        "vertices": N,
        "canonical_edges": [list(edge) for edge in _canonical_edges()],
        "perfect_matchings": [
            [list(edge) for edge in matching] for matching in matchings
        ],
        "perfect_matching_count": len(matchings),
        "independent_order_agrees_with_system": True,
        "quadratic": {
            "unordered_matching_multisets": QUADRATIC_MULTISET_COUNT,
            "distinct_incidence_sums": len(quadratic),
            "collision_fibers": 0,
            "all_fiber_sizes": [1],
            "complete_fiber_fingerprint_sha256": _fingerprint(
                quadratic_payload
            ),
        },
        "cubic": {
            "unordered_matching_multisets": CUBIC_MULTISET_COUNT,
            "distinct_incidence_sums": len(cubic),
            "fiber_size_census": {"1": 660, "2": 10},
            "collision_fibers": len(collisions),
            "complete_fiber_fingerprint_sha256": _fingerprint(
                cubic_payload
            ),
            "collisions": list(collisions),
        },
        "mechanism": (
            "The first matching-incidence relations are ten primitive "
            "cubic circuits, one per K3,3 bipartition.  Each side is one "
            "parity class of the six crossing matchings, and a canonical "
            "pairing realizes the circuit as three four-cycle switches."
        ),
    }


def _exponent_difference(
    support_positions: Mapping[int, int],
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, ...]:
    result = [0] * len(support_positions)
    for variable in left:
        result[support_positions[variable]] += 1
    for variable in right:
        result[support_positions[variable]] -= 1
    return tuple(result)


def _multiply(*monomials: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            variable
            for monomial in monomials
            for variable in monomial
        )
    )


def _nonzero_counter(counter: Counter) -> Counter:
    return Counter(
        {
            monomial: coefficient
            for monomial, coefficient in counter.items()
            if coefficient
        }
    )


def _support21_local_circuit(
    support_index: int,
    *,
    include_spill_terms: bool,
) -> dict:
    audits = size21_support_audits()
    if not 0 <= support_index < len(audits):
        raise KrennMechanismAuditError(
            "support-21 audit index is out of range"
        )
    audit = audits[support_index]
    system = generate_sparse_system(N, D)
    support = tuple(audit.support)
    support_set = set(support)
    positions = {
        variable: position for position, variable in enumerate(support)
    }
    equations = tuple(audit.odd_cycle.equations)
    matching_pairs = tuple(audit.odd_cycle.matching_pairs)
    signs = tuple(audit.odd_cycle.signs)

    active_rows = []
    outside_rows = []
    for equation, expected_pair in zip(
        equations, matching_pairs, strict=True
    ):
        equation_terms = system.equation_monomials(equation)
        active = tuple(
            (matching_index, monomial)
            for matching_index, monomial in enumerate(equation_terms)
            if all(variable in support_set for variable in monomial)
        )
        outside = tuple(
            (matching_index, monomial)
            for matching_index, monomial in enumerate(equation_terms)
            if matching_index not in {index for index, _ in active}
        )
        if (
            tuple(index for index, _ in active) != expected_pair
            or len(active) != 2
            or len(outside) != 13
            or any(
                all(variable in support_set for variable in monomial)
                for _, monomial in outside
            )
        ):
            raise KrennMechanismAuditError(
                "a support-21 equation did not restrict to its binomial"
            )
        active_rows.append(active)
        outside_rows.append(outside)

    exponent_rows = tuple(
        _exponent_difference(
            positions,
            active[0][1],
            active[1][1],
        )
        for active in active_rows
    )
    signed_exponent_sum = tuple(
        sum(
            signs[row] * exponent_rows[row][column]
            for row in range(3)
        )
        for column in range(len(support))
    )
    if any(signed_exponent_sum):
        raise KrennMechanismAuditError(
            "the local signed exponent relation does not vanish"
        )

    # Orient each binomial so prod(X_i) = prod(Y_i) is exactly the signed
    # exponent relation.  Since F_i|_S = X_i + Y_i, telescoping gives
    #
    #   X_2 X_3 F_1 - Y_1 X_3 F_2 + Y_1 Y_2 F_3
    #       = prod(X_i) + prod(Y_i) = 2 T
    #
    # in the quotient that kills variables outside the source support S.
    positive_factors = tuple(
        active[0][1] if sign == 1 else active[1][1]
        for sign, active in zip(signs, active_rows, strict=True)
    )
    negative_factors = tuple(
        active[1][1] if sign == 1 else active[0][1]
        for sign, active in zip(signs, active_rows, strict=True)
    )
    positive_product = _multiply(*positive_factors)
    negative_product = _multiply(*negative_factors)
    if positive_product != negative_product:
        raise KrennMechanismAuditError(
            "the local toric monomial identity failed"
        )
    target = positive_product
    multipliers = (
        _multiply(positive_factors[1], positive_factors[2]),
        _multiply(negative_factors[0], positive_factors[2]),
        _multiply(negative_factors[0], negative_factors[1]),
    )
    combination_coefficients = (1, -1, 1)

    restricted_expansion = Counter()
    full_expansion = Counter()
    spill_records = []
    original_outside_monomials = []
    for equation, active, outside, multiplier, coefficient in zip(
        equations,
        active_rows,
        outside_rows,
        multipliers,
        combination_coefficients,
        strict=True,
    ):
        active_indices = {index for index, _ in active}
        for matching_index, monomial in enumerate(
            system.equation_monomials(equation)
        ):
            product_monomial = _multiply(multiplier, monomial)
            full_expansion[product_monomial] += coefficient
            if matching_index in active_indices:
                restricted_expansion[product_monomial] += coefficient
            else:
                original_outside_monomials.append(monomial)
                spill_records.append(
                    {
                        "equation": equation,
                        "matching_index": matching_index,
                        "combination_coefficient": coefficient,
                        "generator_degree_three_monomial": list(monomial),
                        "lifted_degree_nine_monomial": list(
                            product_monomial
                        ),
                    }
                )

    restricted_nonzero = _nonzero_counter(restricted_expansion)
    full_nonzero = _nonzero_counter(full_expansion)
    spill_products = tuple(
        tuple(record["lifted_degree_nine_monomial"])
        for record in spill_records
    )
    original_outside_monomials = tuple(original_outside_monomials)
    if (
        restricted_nonzero != Counter({target: 2})
        or len(spill_records) != 39
        or len(set(original_outside_monomials)) != 39
        or len(set(spill_products)) != 39
        or target in spill_products
        or len(full_nonzero) != 40
        or full_nonzero[target] != 2
        or Counter(full_nonzero.values())
        != Counter({1: 26, -1: 13, 2: 1})
    ):
        raise KrennMechanismAuditError(
            "the support-conditional lift/spill census changed"
        )

    positive_matching_triple = tuple(
        pair[0] if sign == 1 else pair[1]
        for sign, pair in zip(signs, matching_pairs, strict=True)
    )
    negative_matching_triple = tuple(
        pair[1] if sign == 1 else pair[0]
        for sign, pair in zip(signs, matching_pairs, strict=True)
    )
    incidence = _incidence_vector(positive_matching_triple)
    if incidence != _incidence_vector(negative_matching_triple):
        raise KrennMechanismAuditError(
            "the local exponent cycle lost its matching circuit"
        )
    bipartition = _k33_bipartition(incidence)
    cubic_fiber = sorted(
        (tuple(sorted(positive_matching_triple)),
         tuple(sorted(negative_matching_triple)))
    )
    actual_cubic_fiber = _incidence_fibers(3)[incidence]
    if tuple(cubic_fiber) != actual_cubic_fiber:
        raise KrennMechanismAuditError(
            "the local circuit is not one of the ten cubic fibers"
        )

    payload = {
        "support_index": support_index,
        "source_coordinate_support": list(support),
        "source_coordinate_support_size": len(support),
        "equations": list(equations),
        "colorings": [
            list(coloring_from_index(N, D, equation))
            for equation in equations
        ],
        "matching_pairs": [
            list(pair) for pair in matching_pairs
        ],
        "signs": list(signs),
        "active_degree_three_monomials": [
            [list(monomial) for _, monomial in active]
            for active in active_rows
        ],
        "exponent_rows_in_support_order": [
            list(row) for row in exponent_rows
        ],
        "signed_exponent_sum": list(signed_exponent_sum),
        "positive_matching_triple": list(
            positive_matching_triple
        ),
        "negative_matching_triple": list(
            negative_matching_triple
        ),
        "K3_3_bipartition": [
            list(bipartition[0]),
            list(bipartition[1]),
        ],
        "cubic_matching_collision_fiber": [
            list(multiset) for multiset in actual_cubic_fiber
        ],
        "combination_coefficients": list(
            combination_coefficients
        ),
        "degree_six_multipliers": [
            list(multiplier) for multiplier in multipliers
        ],
        "degree_nine_target_monomial": list(target),
        "restricted_expansion": {
            "quotient": (
                "set every source variable outside the listed "
                "21-coordinate support to zero"
            ),
            "identity": (
                "Q0*F16 - Q1*F18 + Q2*F188 = 2*T "
                "in the source-support quotient"
                if support_index == 0
                else "Q0*F0 - Q1*F1 + Q2*F2 = 2*T, with F0,F1,F2 "
                "the three listed equations, in the source-support quotient"
            ),
            "nonzero_term_count": len(restricted_nonzero),
            "target_coefficient": restricted_nonzero[target],
            "replays_exactly": True,
        },
        "unrestricted_expansion": {
            "outside_source_support_generator_terms": len(
                original_outside_monomials
            ),
            "outside_terms_are_degree_three": all(
                len(monomial) == 3
                for monomial in original_outside_monomials
            ),
            "outside_degree_three_terms_pairwise_distinct": (
                len(set(original_outside_monomials))
                == len(original_outside_monomials)
            ),
            "lifted_degree_nine_spill_terms": len(spill_products),
            "lifted_spill_terms_pairwise_distinct": (
                len(set(spill_products)) == len(spill_products)
            ),
            "lifted_spills_avoid_local_target": (
                target not in spill_products
            ),
            "nonzero_full_expansion_terms": len(full_nonzero),
            "nonzero_coefficient_census": {
                "-1": 13,
                "1": 26,
                "2": 1,
            },
            "closes_to_two_times_target": False,
            "terminology": (
                "These 39 terms are outside a fixed source-coordinate "
                "support. They are not an audit of supp(D^2), and this "
                "is not a k=2 source-ideal computation."
            ),
        },
        "support_conditional_contradiction_over_Q": True,
        "global_source_ideal_identity": False,
    }
    if include_spill_terms:
        payload["unrestricted_expansion"]["spill_terms"] = spill_records
        payload["unrestricted_expansion"][
            "spill_fingerprint_sha256"
        ] = _fingerprint(spill_records)
    return payload


def local_support_mechanism_audit() -> dict:
    """Replay the first local identity and classify all six stored cycles."""

    first = _support21_local_circuit(
        0, include_spill_terms=True
    )
    if (
        tuple(first["source_coordinate_support"])
        != EXPECTED_LOCAL_SUPPORT
        or tuple(first["equations"]) != EXPECTED_LOCAL_EQUATIONS
        or tuple(map(tuple, first["colorings"]))
        != EXPECTED_LOCAL_COLORINGS
        or tuple(map(tuple, first["matching_pairs"]))
        != EXPECTED_LOCAL_MATCHING_PAIRS
        or tuple(first["signs"]) != EXPECTED_LOCAL_SIGNS
        or tuple(
            tuple(map(tuple, row))
            for row in first["active_degree_three_monomials"]
        )
        != EXPECTED_LOCAL_ACTIVE_MONOMIALS
        or tuple(map(tuple, first["K3_3_bipartition"]))
        != EXPECTED_LOCAL_BIPARTITION
    ):
        raise KrennMechanismAuditError(
            "the canonical support-21 local receipt changed"
        )
    compact = tuple(
        _support21_local_circuit(
            index, include_spill_terms=False
        )
        for index in range(len(size21_support_audits()))
    )
    if (
        len(compact) != 6
        or any(
            record["unrestricted_expansion"][
                "outside_source_support_generator_terms"
            ]
            != 39
            or not record[
                "support_conditional_contradiction_over_Q"
            ]
            or record["global_source_ideal_identity"]
            for record in compact
        )
    ):
        raise KrennMechanismAuditError(
            "the six support-21 mechanism census changed"
        )
    return {
        "canonical_first_support": first,
        "all_six_support21_cycles": {
            "cycle_count": len(compact),
            "all_are_K3_3_cubic_matching_circuits": True,
            "outside_source_support_generator_term_counts": [
                record["unrestricted_expansion"][
                    "outside_source_support_generator_terms"
                ]
                for record in compact
            ],
            "lifted_degree_nine_spill_term_counts": [
                record["unrestricted_expansion"][
                    "lifted_degree_nine_spill_terms"
                ]
                for record in compact
            ],
            "compact_receipts": list(compact),
        },
    }


def _build_exact_mechanism_summary() -> dict:
    matching = matching_collision_audit()
    local = local_support_mechanism_audit()
    return {
        "schema": MECHANISM_AUDIT_SCHEMA,
        "matching_incidence": matching,
        "support21_local_mechanism": local,
        "computations_performed": [
            "independent recursive enumeration of all 15 K6 matchings",
            "complete exact enumeration of 120 quadratic matching multisets",
            "complete exact enumeration of 680 cubic matching multisets",
            "exact K3,3/parity/three-switch classification of ten collisions",
            "exact replay of all six stored support-21 three-binomial cycles",
            "symbolic collection of the canonical restricted and full lift",
        ],
        "claims": {
            "quadratic_matching_incidence_collision_exists": False,
            "cubic_matching_collisions_are_exactly_ten_K3_3_circuits": True,
            "all_six_support21_cycles_use_a_K3_3_circuit": True,
            "canonical_support21_nonzero_solution_excluded_over_Q": True,
            "all_six_recorded_support21_nonzero_solutions_excluded_over_Q": (
                True
            ),
            "canonical_local_combination_is_global_source_ideal_identity": (
                False
            ),
            "D_not_in_J_mix_at_k1_reproved_here": False,
            "D_squared_in_J_mix_decided": False,
            "D_in_radical_J_mix_decided": False,
            "global_GHZ_nonexistence_proved": False,
            "exact_affine_GHZ_membership_decided": False,
        },
        "distinctions": {
            "prior_k1_result": (
                "D not in J_mix is an established separate result and is "
                "not reproved by this mechanism audit."
            ),
            "present_exact_result": (
                "Six particular exact nonzero source supports are excluded "
                "by support-conditional K3,3 cubic circuits."
            ),
            "radical_question": (
                "Whether some D^k lies in J_mix remains undecided."
            ),
            "global_question": (
                "Global affine GHZ membership remains undecided; known "
                "border membership is not affine membership."
            ),
        },
        "claim_boundary": (
            "The cubic K3,3 relation explains the three-binomial "
            "contradiction only after restricting to a fixed nonzero "
            "source-coordinate support.  The unrestricted lift retains "
            "39 distinct spill terms.  This does not decide D^2 in J_mix, "
            "radical membership, global GHZ nonexistence, or affine versus "
            "strict-border membership."
        ),
    }


@lru_cache(maxsize=1)
def _exact_summary_json() -> str:
    return json.dumps(
        _build_exact_mechanism_summary(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def exact_mechanism_summary() -> dict:
    """Return a fresh JSON-ready copy of the exact mechanism certificate."""

    return json.loads(_exact_summary_json())


def verify_mechanism_payload(payload: Mapping) -> dict:
    """Recompute the complete certificate and reject any altered payload."""

    if not isinstance(payload, Mapping):
        raise KrennMechanismAuditError(
            "the mechanism certificate must be a mapping"
        )
    try:
        candidate = json.loads(
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise KrennMechanismAuditError(
            "the mechanism certificate is not strict JSON"
        ) from exc
    expected = exact_mechanism_summary()
    if set(candidate) != set(expected):
        raise KrennMechanismAuditError(
            "the mechanism certificate top-level schema changed"
        )
    if candidate != expected:
        raise KrennMechanismAuditError(
            "the mechanism certificate failed exact semantic replay"
        )
    return {
        "matching_enumeration_replayed": True,
        "quadratic_uniqueness_replayed": True,
        "ten_K3_3_cubic_circuits_replayed": True,
        "local_three_equation_identity_replayed": True,
        "thirty_nine_spill_terms_replayed": True,
        "global_membership_decided": False,
    }
