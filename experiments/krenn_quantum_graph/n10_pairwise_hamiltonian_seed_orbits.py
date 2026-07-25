"""Exact pairwise-Hamiltonian matching-triple census for ``n=10,d=3``.

This is a deliberately bounded structural computation.  It does not
enumerate all ``945^3`` ordered triples.  It fixes one ordered Hamiltonian
pair of perfect matchings, enumerates its 148 common Hamiltonian partners,
and canonically quotients those partners by ``S_10 x S_3``.

The resulting ten orbit types are exhaustive only inside the locus where
every pair of selected monochromatic matchings forms a Hamilton 10-cycle.
Each selected triple has fifteen distinct diagonal source coordinates.
Every additional perfect matching contained in their cubic union gives a
unique mixed coloring and hence a singleton seed-supported equation.  This
forces an outside repair in a hypothetical solution but does not exclude
unrestricted repairs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, permutations
import json
from math import factorial, gcd, lcm
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.krenn_quantum_graph.system import (
    Matching,
    perfect_matchings,
    variable_count,
    variable_index,
)


N = 10
D = 3
MATCHING_COUNT = 945
FIXED_FIRST = 0
FIXED_SECOND = 124
FIXED_FIRST_HAMILTONIAN_PARTNERS = 384
FIXED_PAIR_COMMON_PARTNERS = 148
ORDERED_PAIRWISE_HAMILTONIAN_TRIPLES = 53_706_240
UNORDERED_COMPATIBILITY_TRIANGLES = 8_951_040
ORDERED_COLOR_S10_ORBITS = 24
FULL_S10_X_S3_ORBITS = 10
GROUP_ORDER = factorial(N) * factorial(D)

CENSUS_SCHEMA = "krenn-n10-pairwise-hamiltonian-seed-census-v1"
MANIFEST_SCHEMA = (
    "krenn-n10-pairwise-hamiltonian-seed-census-manifest-v1"
)

EXPECTED_CLASS_ROWS = (
    {
        "third": 248,
        "fixed_pair_class_size": 15,
        "internal_indices": (0, 1, 14, 66, 68, 124, 248),
        "rank": 6,
        "graph": (False, 2, 3, 3, 3),
        "circuit_histogram": {2: 1},
    },
    {
        "third": 250,
        "fixed_pair_class_size": 10,
        "internal_indices": (0, 7, 81, 124, 138, 250),
        "rank": 6,
        "graph": (False, 3, 0, 3, 3),
        "circuit_histogram": {},
    },
    {
        "third": 253,
        "fixed_pair_class_size": 15,
        "internal_indices": (0, 1, 97, 124, 138, 239, 253),
        "rank": 6,
        "graph": (False, 2, 1, 3, 6),
        "circuit_histogram": {3: 1},
    },
    {
        "third": 260,
        "fixed_pair_class_size": 30,
        "internal_indices": (0, 14, 35, 124, 225, 260),
        "rank": 6,
        "graph": (False, 2, 2, 3, 3),
        "circuit_histogram": {},
    },
    {
        "third": 266,
        "fixed_pair_class_size": 30,
        "internal_indices": (0, 14, 81, 83, 123, 124, 264, 266),
        "rank": 6,
        "graph": (False, 1, 4, 3, 6),
        "circuit_histogram": {2: 1, 3: 2},
    },
    {
        "third": 284,
        "fixed_pair_class_size": 10,
        "internal_indices": (0, 9, 124, 171, 239, 284),
        "rank": 5,
        "graph": (False, 2, 0, 3, 6),
        "circuit_histogram": {3: 1},
    },
    {
        "third": 443,
        "fixed_pair_class_size": 3,
        "internal_indices": (
            0,
            1,
            14,
            66,
            68,
            105,
            106,
            119,
            124,
            443,
            450,
            451,
            456,
        ),
        "rank": 7,
        "graph": (True, 0, 5, 4, 8),
        "circuit_histogram": {2: 10, 3: 25, 4: 30, 5: 17, 6: 5},
    },
    {
        "third": 448,
        "fixed_pair_class_size": 15,
        "internal_indices": (
            0,
            1,
            97,
            105,
            106,
            124,
            448,
            450,
            451,
        ),
        "rank": 6,
        "graph": (False, 0, 3, 4, 7),
        "circuit_histogram": {2: 3, 3: 3, 4: 3},
    },
    {
        "third": 485,
        "fixed_pair_class_size": 5,
        "internal_indices": (
            0,
            1,
            19,
            29,
            124,
            134,
            171,
            177,
            450,
            451,
            483,
            485,
        ),
        "rank": 7,
        "graph": (True, 0, 6, 4, 12),
        "circuit_histogram": {2: 3, 3: 18, 4: 12, 5: 24},
    },
    {
        "third": 492,
        "fixed_pair_class_size": 15,
        "internal_indices": (
            0,
            1,
            92,
            124,
            171,
            450,
            451,
            492,
        ),
        "rank": 6,
        "graph": (False, 0, 2, 4, 6),
        "circuit_histogram": {2: 1, 3: 1, 4: 2},
    },
)

CLAIM_BOUNDARY = {
    "all_n10_seed_triples_classified": False,
    "eqsystem_solutions_classified": False,
    "n10_boundary_membership_proved": False,
    "n10_existence_proved": False,
    "n10_nonexistence_proved": False,
    "pairwise_hamiltonian_seed_support_can_be_repaired": False,
    "pairwise_hamiltonian_seed_support_is_unrepairable": False,
    "singleton_victim_is_global_nonexistence_proof": False,
    "ten_orbit_types_persist_for_general_n": False,
    "weights_on_symmetry_related_coordinates_are_equal": False,
}

SOURCE_FILES = (
    "experiments/krenn_quantum_graph/n10_pairwise_hamiltonian_seed_orbits.py",
    "experiments/krenn_quantum_graph/system.py",
    "tests/test_krenn_n10_pairwise_hamiltonian_seed_orbits.py",
)


class KrennN10PairwiseHamiltonianError(ValueError):
    """The exact census or its round-trip verification failed."""


def _exact_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KrennN10PairwiseHamiltonianError(
            f"{label} must be an exact integer"
        )
    return value


@lru_cache(maxsize=1)
def _matchings() -> tuple[Matching, ...]:
    matchings = perfect_matchings(N)
    if len(matchings) != MATCHING_COUNT:
        raise KrennN10PairwiseHamiltonianError(
            "the K10 perfect-matching census changed"
        )
    if len(set(matchings)) != MATCHING_COUNT:
        raise KrennN10PairwiseHamiltonianError(
            "the K10 matching list is not injective"
        )
    return matchings


@lru_cache(maxsize=1)
def _matching_index() -> Mapping[Matching, int]:
    return {
        matching: index
        for index, matching in enumerate(_matchings())
    }


@lru_cache(maxsize=1)
def _mate_rows() -> tuple[tuple[int, ...], ...]:
    rows = []
    for matching in _matchings():
        mates = [-1] * N
        for first, second in matching:
            mates[first] = second
            mates[second] = first
        if any(mate < 0 for mate in mates):
            raise KrennN10PairwiseHamiltonianError(
                "a perfect matching left a vertex unmatched"
            )
        rows.append(tuple(mates))
    return tuple(rows)


def _matching_payload(index: int) -> list[list[int]]:
    index = _exact_integer(index, "matching index")
    if index < 0 or index >= MATCHING_COUNT:
        raise KrennN10PairwiseHamiltonianError(
            "matching index is outside range(945)"
        )
    return [list(edge) for edge in _matchings()[index]]


def _is_hamiltonian_union(first: int, second: int) -> bool:
    first = _exact_integer(first, "first matching index")
    second = _exact_integer(second, "second matching index")
    if (
        first < 0
        or second < 0
        or first >= MATCHING_COUNT
        or second >= MATCHING_COUNT
    ):
        raise KrennN10PairwiseHamiltonianError(
            "Hamiltonian test received an invalid matching index"
        )
    first_mates = _mate_rows()[first]
    second_mates = _mate_rows()[second]
    if any(
        first_mates[vertex] == second_mates[vertex]
        for vertex in range(N)
    ):
        return False
    seen = {0}
    queue = [0]
    for vertex in queue:
        for neighbor in (
            first_mates[vertex],
            second_mates[vertex],
        ):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == N


def _transport_matching(
    matching: Matching, vertex_permutation: Sequence[int]
) -> Matching:
    permutation = tuple(
        _exact_integer(value, "permutation entry")
        for value in vertex_permutation
    )
    if (
        len(permutation) != N
        or tuple(sorted(permutation)) != tuple(range(N))
    ):
        raise KrennN10PairwiseHamiltonianError(
            "a vertex action must be a permutation of range(10)"
        )
    return tuple(
        sorted(
            (
                min(permutation[first], permutation[second]),
                max(permutation[first], permutation[second]),
            )
            for first, second in matching
        )
    )


def _transport_index(
    matching_index: int, vertex_permutation: Sequence[int]
) -> int:
    return _matching_index()[
        _transport_matching(
            _matchings()[matching_index], vertex_permutation
        )
    ]


@lru_cache(maxsize=1)
def fixed_first_hamiltonian_partners() -> tuple[int, ...]:
    partners = tuple(
        index
        for index in range(MATCHING_COUNT)
        if _is_hamiltonian_union(FIXED_FIRST, index)
    )
    formula = 2 ** (N // 2 - 1) * factorial(N // 2 - 1)
    if (
        len(partners) != FIXED_FIRST_HAMILTONIAN_PARTNERS
        or len(partners) != formula
        or partners[0] != FIXED_SECOND
    ):
        raise KrennN10PairwiseHamiltonianError(
            "the fixed K10 Hamiltonian-partner census changed"
        )
    return partners


@lru_cache(maxsize=1)
def fixed_pair_common_partners() -> tuple[int, ...]:
    if FIXED_SECOND not in fixed_first_hamiltonian_partners():
        raise KrennN10PairwiseHamiltonianError(
            "the fixed second matching is not Hamilton-compatible"
        )
    partners = tuple(
        index
        for index in range(MATCHING_COUNT)
        if _is_hamiltonian_union(FIXED_FIRST, index)
        and _is_hamiltonian_union(FIXED_SECOND, index)
    )
    if len(partners) != FIXED_PAIR_COMMON_PARTNERS:
        raise KrennN10PairwiseHamiltonianError(
            "the fixed ordered-pair common-neighbor census changed"
        )
    return partners


@lru_cache(maxsize=None)
def _pair_normalizers(
    first: int, second: int
) -> tuple[tuple[int, ...], ...]:
    """Return the ten maps carrying an ordered Hamilton pair to (0,124)."""

    if not _is_hamiltonian_union(first, second):
        raise KrennN10PairwiseHamiltonianError(
            "only a Hamiltonian ordered pair can be normalized"
        )
    source_first = _mate_rows()[first]
    source_second = _mate_rows()[second]
    target_first = _mate_rows()[FIXED_FIRST]
    target_second = _mate_rows()[FIXED_SECOND]
    normalizers = []
    for root_image in range(N):
        permutation = [-1] * N
        permutation[0] = root_image
        queue = [0]
        for vertex in queue:
            for source_mates, target_mates in (
                (source_first, target_first),
                (source_second, target_second),
            ):
                source_neighbor = source_mates[vertex]
                target_neighbor = target_mates[
                    permutation[vertex]
                ]
                if permutation[source_neighbor] < 0:
                    permutation[source_neighbor] = target_neighbor
                    queue.append(source_neighbor)
                elif (
                    permutation[source_neighbor]
                    != target_neighbor
                ):
                    raise KrennN10PairwiseHamiltonianError(
                        "alternating-cycle propagation is inconsistent"
                    )
        checked = tuple(permutation)
        if (
            tuple(sorted(checked)) != tuple(range(N))
            or _transport_index(first, checked) != FIXED_FIRST
            or _transport_index(second, checked) != FIXED_SECOND
        ):
            raise KrennN10PairwiseHamiltonianError(
                "an ordered-pair normalizer failed exact transport"
            )
        normalizers.append(checked)
    result = tuple(normalizers)
    if len(result) != N or len(set(result)) != N:
        raise KrennN10PairwiseHamiltonianError(
            "an ordered Hamilton pair does not have ten normalizers"
        )
    return result


@lru_cache(maxsize=None)
def canonical_fixed_pair_third(third: int) -> int:
    """Canonicalize ``(0,124,third)`` under the full ``S_10 x S_3``."""

    third = _exact_integer(third, "third matching index")
    if third not in fixed_pair_common_partners():
        raise KrennN10PairwiseHamiltonianError(
            "the third matching is not a common Hamiltonian partner"
        )
    triple = (FIXED_FIRST, FIXED_SECOND, third)
    images = []
    for first_color, second_color, third_color in permutations(
        range(D)
    ):
        for vertex_permutation in _pair_normalizers(
            triple[first_color], triple[second_color]
        ):
            image = _transport_index(
                triple[third_color], vertex_permutation
            )
            if image not in fixed_pair_common_partners():
                raise KrennN10PairwiseHamiltonianError(
                    "a normalized color action left the fixed-pair slice"
                )
            images.append(image)
    if len(images) != factorial(D) * N:
        raise KrennN10PairwiseHamiltonianError(
            "the full color/normalizer image census changed"
        )
    return min(images)


@lru_cache(maxsize=1)
def _fixed_pair_classes() -> tuple[tuple[int, tuple[int, ...]], ...]:
    classes: dict[int, list[int]] = {}
    for third in fixed_pair_common_partners():
        key = canonical_fixed_pair_third(third)
        classes.setdefault(key, []).append(third)
    rows = tuple(
        (key, tuple(members))
        for key, members in sorted(classes.items())
    )
    expected = tuple(
        (
            row["third"],
            row["fixed_pair_class_size"],
        )
        for row in EXPECTED_CLASS_ROWS
    )
    observed = tuple(
        (key, len(members)) for key, members in rows
    )
    if (
        observed != expected
        or len(rows) != FULL_S10_X_S3_ORBITS
        or sum(len(members) for _key, members in rows)
        != FIXED_PAIR_COMMON_PARTNERS
        or any(
            canonical_fixed_pair_third(key) != key
            for key, _members in rows
        )
    ):
        raise KrennN10PairwiseHamiltonianError(
            "the ten full-symmetry classes changed"
        )
    return rows


def _fixed_pair_ordered_color_orbit_count() -> int:
    """Count orbits of the 148 thirds under the ordered-pair stabilizer."""

    candidates = fixed_pair_common_partners()
    candidate_set = set(candidates)
    seen: set[int] = set()
    orbit_count = 0
    for root in candidates:
        if root in seen:
            continue
        orbit_count += 1
        seen.add(root)
        queue = [root]
        for third in queue:
            for normalizer in _pair_normalizers(
                FIXED_FIRST, FIXED_SECOND
            ):
                image = _transport_index(third, normalizer)
                if image not in candidate_set:
                    raise KrennN10PairwiseHamiltonianError(
                        "the ordered-pair stabilizer left its slice"
                    )
                if image not in seen:
                    seen.add(image)
                    queue.append(image)
    if orbit_count != ORDERED_COLOR_S10_ORBITS:
        raise KrennN10PairwiseHamiltonianError(
            "the ordered-color S10 orbit count changed"
        )
    return orbit_count


@lru_cache(maxsize=1)
def _compatibility_graph_audit() -> dict[str, object]:
    """Build the full 945-vertex Hamilton-compatibility graph."""

    rows = [0] * MATCHING_COUNT
    for first in range(MATCHING_COUNT):
        for second in range(first + 1, MATCHING_COUNT):
            if _is_hamiltonian_union(first, second):
                rows[first] |= 1 << second
                rows[second] |= 1 << first
    degrees = tuple(row.bit_count() for row in rows)
    if set(degrees) != {FIXED_FIRST_HAMILTONIAN_PARTNERS}:
        raise KrennN10PairwiseHamiltonianError(
            "the Hamilton-compatibility graph is not 384-regular"
        )
    edge_count = sum(degrees) // 2
    common_neighbor_counts = set()
    for first, row in enumerate(rows):
        remaining = row >> (first + 1)
        offset = first + 1
        while remaining:
            low_bit = remaining & -remaining
            position = low_bit.bit_length() - 1
            second = offset + position
            common_neighbor_counts.add(
                (rows[first] & rows[second]).bit_count()
            )
            remaining ^= low_bit
    if common_neighbor_counts != {FIXED_PAIR_COMMON_PARTNERS}:
        raise KrennN10PairwiseHamiltonianError(
            "compatible ordered pairs do not have 148 common partners"
        )
    triangle_count = (
        edge_count * FIXED_PAIR_COMMON_PARTNERS // 3
    )
    if (
        edge_count != 181_440
        or triangle_count != UNORDERED_COMPATIBILITY_TRIANGLES
        or triangle_count * factorial(D)
        != ORDERED_PAIRWISE_HAMILTONIAN_TRIPLES
    ):
        raise KrennN10PairwiseHamiltonianError(
            "the independent compatibility-graph masses changed"
        )
    row_width = (MATCHING_COUNT + 7) // 8
    raw = b"".join(
        row.to_bytes(row_width, "little") for row in rows
    )
    return {
        "vertices": MATCHING_COUNT,
        "regular_degree": degrees[0],
        "edges": edge_count,
        "common_neighbors_per_edge": next(
            iter(common_neighbor_counts)
        ),
        "triangles": triangle_count,
        "raw_bit_rows_bytes": len(raw),
        "raw_bit_rows_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _rational_rank(vectors: Sequence[Sequence[int]]) -> int:
    matrix = [
        [Fraction(value) for value in vector]
        for vector in vectors
    ]
    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise KrennN10PairwiseHamiltonianError(
            "an incidence matrix is ragged"
        )
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
        matrix[rank], matrix[pivot] = (
            matrix[pivot],
            matrix[rank],
        )
        pivot_value = matrix[rank][column]
        matrix[rank] = [
            value / pivot_value for value in matrix[rank]
        ]
        for row in range(len(matrix)):
            if row == rank or not matrix[row][column]:
                continue
            coefficient = matrix[row][column]
            matrix[row] = [
                value - coefficient * pivot_entry
                for value, pivot_entry in zip(
                    matrix[row], matrix[rank], strict=True
                )
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _primitive_kernel_vector(
    vectors: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Return the primitive kernel of a minimally dependent column set."""

    column_count = len(vectors)
    coordinate_count = len(vectors[0])
    matrix = [
        [
            Fraction(vectors[column][coordinate])
            for column in range(column_count)
        ]
        for coordinate in range(coordinate_count)
    ]
    pivot_columns = []
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(pivot_row, coordinate_count)
                if matrix[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = (
            matrix[pivot], matrix[pivot_row]
        )
        pivot_value = matrix[pivot_row][column]
        matrix[pivot_row] = [
            value / pivot_value for value in matrix[pivot_row]
        ]
        for row in range(coordinate_count):
            if row == pivot_row or not matrix[row][column]:
                continue
            coefficient = matrix[row][column]
            matrix[row] = [
                value - coefficient * pivot_entry
                for value, pivot_entry in zip(
                    matrix[row], matrix[pivot_row], strict=True
                )
            ]
        pivot_columns.append(column)
        pivot_row += 1
    free_columns = tuple(
        column
        for column in range(column_count)
        if column not in pivot_columns
    )
    if len(free_columns) != 1:
        raise KrennN10PairwiseHamiltonianError(
            "a circuit does not have a one-dimensional kernel"
        )
    values = [Fraction(0) for _column in range(column_count)]
    values[free_columns[0]] = Fraction(1)
    for row, pivot_column in reversed(
        tuple(enumerate(pivot_columns))
    ):
        values[pivot_column] = -sum(
            matrix[row][column] * values[column]
            for column in free_columns
        )
    denominator = 1
    for value in values:
        denominator = lcm(denominator, value.denominator)
    integers = [int(value * denominator) for value in values]
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    integers = [value // divisor for value in integers]
    first_nonzero = next(value for value in integers if value)
    if first_nonzero < 0:
        integers = [-value for value in integers]
    result = tuple(integers)
    if (
        any(not value for value in result)
        or gcd(*tuple(abs(value) for value in result)) != 1
    ):
        raise KrennN10PairwiseHamiltonianError(
            "a primitive circuit vector failed normalization"
        )
    return result


def _circuits(
    internal_indices: Sequence[int],
    vectors: Sequence[Sequence[int]],
) -> tuple[dict[str, object], ...]:
    rank_cache: dict[tuple[int, ...], int] = {}

    def subset_rank(subset: tuple[int, ...]) -> int:
        if subset not in rank_cache:
            rank_cache[subset] = _rational_rank(
                tuple(vectors[index] for index in subset)
            )
        return rank_cache[subset]

    ambient_rank = _rational_rank(vectors)
    records = []
    for size in range(2, min(len(vectors), ambient_rank + 1) + 1):
        for subset in combinations(range(len(vectors)), size):
            if subset_rank(subset) != size - 1:
                continue
            if any(
                subset_rank(
                    tuple(
                        index
                        for index in subset
                        if index != removed
                    )
                )
                != size - 1
                for removed in subset
            ):
                continue
            selected_vectors = tuple(
                vectors[index] for index in subset
            )
            coefficients = _primitive_kernel_vector(
                selected_vectors
            )
            signed_sum = tuple(
                sum(
                    coefficient * vector[coordinate]
                    for coefficient, vector in zip(
                        coefficients,
                        selected_vectors,
                        strict=True,
                    )
                )
                for coordinate in range(len(selected_vectors[0]))
            )
            positive_degree = sum(
                coefficient
                for coefficient in coefficients
                if coefficient > 0
            )
            negative_degree = -sum(
                coefficient
                for coefficient in coefficients
                if coefficient < 0
            )
            if (
                any(signed_sum)
                or positive_degree != negative_degree
                or positive_degree <= 0
            ):
                raise KrennN10PairwiseHamiltonianError(
                    "an exact circuit relation failed replay"
                )
            records.append(
                {
                    "matching_indices": [
                        internal_indices[index]
                        for index in subset
                    ],
                    "primitive_coefficients": list(coefficients),
                    "support_size": size,
                    "binomial_degree": positive_degree,
                }
            )
    return tuple(records)


def _union_edges(seed: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                edge
                for matching_index in seed
                for edge in _matchings()[matching_index]
            }
        )
    )


def _internal_matching_indices(
    seed: Sequence[int],
) -> tuple[int, ...]:
    edges = set(_union_edges(seed))
    return tuple(
        index
        for index, matching in enumerate(_matchings())
        if set(matching) <= edges
    )


def _graph_profile(
    edges: Sequence[tuple[int, int]],
    internal_indices: Sequence[int],
) -> dict[str, object]:
    adjacency = [set() for _vertex in range(N)]
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    if any(len(neighbors) != D for neighbors in adjacency):
        raise KrennN10PairwiseHamiltonianError(
            "a pairwise-Hamiltonian union is not cubic"
        )
    colors: dict[int, int] = {}
    bipartite = True
    for root in range(N):
        if root in colors:
            continue
        colors[root] = 0
        queue = [root]
        for vertex in queue:
            for neighbor in adjacency[vertex]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[vertex]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    bipartite = False
    triangle_count = sum(
        int(
            second in adjacency[first]
            and third in adjacency[first]
            and third in adjacency[second]
        )
        for first, second, third in combinations(range(N), 3)
    )
    four_cycle_twice = 0
    for first, second in combinations(range(N), 2):
        common = len(adjacency[first] & adjacency[second])
        four_cycle_twice += common * (common - 1) // 2
    four_cycle_count = four_cycle_twice // 2
    girth = N + 1
    for root in range(N):
        distances = [-1] * N
        parents = [-1] * N
        distances[root] = 0
        queue = [root]
        for vertex in queue:
            for neighbor in adjacency[vertex]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[vertex] + 1
                    parents[neighbor] = vertex
                    queue.append(neighbor)
                elif parents[vertex] != neighbor:
                    girth = min(
                        girth,
                        distances[vertex]
                        + distances[neighbor]
                        + 1,
                    )
    hamilton_cycle_count = sum(
        _is_hamiltonian_union(first, second)
        for first, second in combinations(internal_indices, 2)
    )
    return {
        "connected": True,
        "cubic": True,
        "bipartite": bipartite,
        "triangle_count": triangle_count,
        "four_cycle_count": four_cycle_count,
        "girth": girth,
        "hamilton_10_cycle_count": hamilton_cycle_count,
    }


def _seed_support_and_victims(
    seed: tuple[int, int, int],
    internal_indices: Sequence[int],
) -> dict[str, object]:
    edge_color: dict[tuple[int, int], int] = {}
    support = []
    for color, matching_index in enumerate(seed):
        for first, second in _matchings()[matching_index]:
            edge = (first, second)
            if edge in edge_color:
                raise KrennN10PairwiseHamiltonianError(
                    "a pairwise-Hamiltonian seed reused an edge"
                )
            edge_color[edge] = color
            support.append(
                variable_index(
                    N, D, first, second, color, color
                )
            )
    if len(support) != 15 or len(set(support)) != 15:
        raise KrennN10PairwiseHamiltonianError(
            "a seed did not define fifteen source coordinates"
        )
    records = []
    colorings = set()
    mixed_profile_counts: Counter[str] = Counter()
    for matching_index in internal_indices:
        coloring = [-1] * N
        color_edge_counts = Counter()
        monomial = []
        for first, second in _matchings()[matching_index]:
            color = edge_color[(first, second)]
            coloring[first] = color
            coloring[second] = color
            color_edge_counts[color] += 1
            monomial.append(
                variable_index(
                    N, D, first, second, color, color
                )
            )
        checked_coloring = tuple(coloring)
        if checked_coloring in colorings:
            raise KrennN10PairwiseHamiltonianError(
                "two internal terms share a seed-supported coloring"
            )
        colorings.add(checked_coloring)
        monochromatic = len(set(checked_coloring)) == 1
        profile = "+".join(
            str(value)
            for value in sorted(
                color_edge_counts.values(), reverse=True
            )
        )
        if not monochromatic:
            mixed_profile_counts[profile] += 1
        records.append(
            {
                "matching_index": matching_index,
                "inherited_coloring": list(checked_coloring),
                "monochromatic": monochromatic,
                "seed_edge_color_count_profile": profile,
                "source_monomial_variable_indices": monomial,
            }
        )
    monochromatic_indices = tuple(
        record["matching_index"]
        for record in records
        if record["monochromatic"]
    )
    if monochromatic_indices != tuple(sorted(seed)):
        raise KrennN10PairwiseHamiltonianError(
            "the internal monochromatic terms changed"
        )
    return {
        "seed_source_variable_indices": sorted(support),
        "internal_colored_monomials": records,
        "singleton_mixed_victim_count": len(records) - D,
        "mixed_victim_edge_color_profiles": dict(
            sorted(mixed_profile_counts.items())
        ),
        "all_seed_supported_mixed_terms_are_singletons": True,
    }


def _class_payload(
    expected: Mapping[str, object],
    members: Sequence[int],
) -> dict[str, object]:
    third = _exact_integer(expected["third"], "class third")
    class_size = _exact_integer(
        expected["fixed_pair_class_size"],
        "fixed-pair class size",
    )
    if len(members) != class_size or min(members) != third:
        raise KrennN10PairwiseHamiltonianError(
            "a fixed-pair class failed its representative regression"
        )
    seed = (FIXED_FIRST, FIXED_SECOND, third)
    if not all(
        _is_hamiltonian_union(seed[first], seed[second])
        for first, second in combinations(range(D), 2)
    ):
        raise KrennN10PairwiseHamiltonianError(
            "a representative is not pairwise Hamiltonian"
        )
    edges = _union_edges(seed)
    internal_indices = _internal_matching_indices(seed)
    expected_internal = tuple(expected["internal_indices"])
    if len(edges) != 15 or internal_indices != expected_internal:
        raise KrennN10PairwiseHamiltonianError(
            "a representative's cubic union census changed"
        )
    edge_position = {
        edge: position for position, edge in enumerate(edges)
    }
    vectors = tuple(
        tuple(
            int(edge in _matchings()[matching_index])
            for edge in edges
        )
        for matching_index in internal_indices
    )
    rank = _rational_rank(vectors)
    circuits = _circuits(internal_indices, vectors)
    circuit_histogram = Counter(
        record["binomial_degree"] for record in circuits
    )
    expected_histogram = Counter(expected["circuit_histogram"])
    graph = _graph_profile(edges, internal_indices)
    expected_graph = tuple(expected["graph"])
    observed_graph = (
        graph["bipartite"],
        graph["triangle_count"],
        graph["four_cycle_count"],
        graph["girth"],
        graph["hamilton_10_cycle_count"],
    )
    if (
        rank != expected["rank"]
        or circuit_histogram != expected_histogram
        or observed_graph != expected_graph
        or set(edge_position) != set(edges)
    ):
        raise KrennN10PairwiseHamiltonianError(
            "a representative's graph or circuit invariants changed"
        )
    victim_payload = _seed_support_and_victims(
        seed, internal_indices
    )
    normalized_first_component_size = (
        FIXED_FIRST_HAMILTONIAN_PARTNERS * class_size
    )
    ordered_orbit_size = (
        MATCHING_COUNT * normalized_first_component_size
    )
    if GROUP_ORDER % ordered_orbit_size:
        raise KrennN10PairwiseHamiltonianError(
            "a full orbit size does not divide S10 x S3"
        )
    stabilizer_size = GROUP_ORDER // ordered_orbit_size
    if stabilizer_size != factorial(D) * N // class_size:
        raise KrennN10PairwiseHamiltonianError(
            "the direct and fiber stabilizer counts disagree"
        )
    return {
        "id": f"PH10-{third}",
        "id_semantics": {
            "prefix": "PH10",
            "prefix_meaning": (
                "n10 triple with all three pairwise unions Hamiltonian"
            ),
            "numeric_suffix_meaning": (
                "canonical third matching index in the fixed-pair slice"
            ),
            "ordinal_case_number": False,
        },
        "representative_matching_indices": list(seed),
        "representative_matchings": [
            _matching_payload(index) for index in seed
        ],
        "fixed_ordered_pair_class_size": class_size,
        "fixed_first_matching_component_size": (
            normalized_first_component_size
        ),
        "ordered_triple_orbit_size": ordered_orbit_size,
        "stabilizer_size_in_S10_x_S3": stabilizer_size,
        "physical_union_edges": [list(edge) for edge in edges],
        "physical_union_edge_count": len(edges),
        "physical_union_graph": graph,
        "internal_physical_matching_count": len(internal_indices),
        "internal_physical_matching_indices": list(
            internal_indices
        ),
        "colored_exponent_matrix": {
            "ambient_source_coordinates": variable_count(N, D),
            "restricted_seed_coordinates": len(edges),
            "columns": len(vectors),
            "rank_over_Q": rank,
            "nullity_over_Q": len(vectors) - rank,
        },
        "support_minimal_toric_circuits": list(circuits),
        "toric_circuit_count_by_binomial_degree": {
            str(degree): count
            for degree, count in sorted(circuit_histogram.items())
        },
        **victim_payload,
    }


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


@lru_cache(maxsize=1)
def _census_bytes() -> bytes:
    fixed_classes = _fixed_pair_classes()
    expected_by_third = {
        row["third"]: row for row in EXPECTED_CLASS_ROWS
    }
    class_payloads = tuple(
        _class_payload(expected_by_third[third], members)
        for third, members in fixed_classes
    )
    compatibility_graph = _compatibility_graph_audit()
    orbit_mass = sum(
        row["ordered_triple_orbit_size"]
        for row in class_payloads
    )
    graph_signatures = {
        (
            row["internal_physical_matching_count"],
            row["colored_exponent_matrix"]["rank_over_Q"],
            row["physical_union_graph"]["bipartite"],
            row["physical_union_graph"]["triangle_count"],
            row["physical_union_graph"]["four_cycle_count"],
            row["physical_union_graph"]["girth"],
        )
        for row in class_payloads
    }
    if (
        _fixed_pair_ordered_color_orbit_count()
        != ORDERED_COLOR_S10_ORBITS
        or orbit_mass != ORDERED_PAIRWISE_HAMILTONIAN_TRIPLES
        or len(graph_signatures) != FULL_S10_X_S3_ORBITS
    ):
        raise KrennN10PairwiseHamiltonianError(
            "the final orbit mass or graph separation changed"
        )
    payload = {
        "schema": CENSUS_SCHEMA,
        "parameters": {
            "n": N,
            "d": D,
            "source_variable_count": variable_count(N, D),
        },
        "algorithm": {
            "full_945_cubed_triple_enumeration_used": False,
            "fixed_first_matching_index": FIXED_FIRST,
            "fixed_second_matching_index": FIXED_SECOND,
            "fixed_ordered_pair_normalizers": N,
            "color_orderings_per_candidate": factorial(D),
            "canonical_images_per_candidate": N * factorial(D),
            "canonical_key": (
                "least transported third matching index after all "
                "six color orderings and ten ordered-pair normalizers"
            ),
            "equivalence": "S_10 x S_3",
        },
        "counts": {
            "perfect_matchings": MATCHING_COUNT,
            "hamiltonian_partners_of_fixed_matching": (
                len(fixed_first_hamiltonian_partners())
            ),
            "common_hamiltonian_partners_of_fixed_ordered_pair": (
                len(fixed_pair_common_partners())
            ),
            "ordered_pairwise_hamiltonian_triples": (
                ORDERED_PAIRWISE_HAMILTONIAN_TRIPLES
            ),
            "unordered_compatibility_graph_triangles": (
                UNORDERED_COMPATIBILITY_TRIANGLES
            ),
            "S10_orbits_with_colors_ordered": (
                ORDERED_COLOR_S10_ORBITS
            ),
            "S10_x_S3_orbits": len(class_payloads),
            "sum_of_ordered_orbit_sizes": orbit_mass,
        },
        "compatibility_graph_audit": compatibility_graph,
        "orbits": list(class_payloads),
        "exact_checks": {
            "matching_count_945": len(_matchings()) == 945,
            "fixed_partner_formula_384": (
                len(fixed_first_hamiltonian_partners()) == 384
            ),
            "fixed_pair_common_partners_148": (
                len(fixed_pair_common_partners()) == 148
            ),
            "compatibility_graph_is_384_regular": (
                compatibility_graph["regular_degree"] == 384
            ),
            "compatibility_graph_edges_181440": (
                compatibility_graph["edges"] == 181_440
            ),
            "compatibility_graph_triangles_8951040": (
                compatibility_graph["triangles"] == 8_951_040
            ),
            "ordered_color_orbits_24": (
                _fixed_pair_ordered_color_orbit_count() == 24
            ),
            "full_symmetry_orbits_10": (
                len(class_payloads) == 10
            ),
            "orbit_mass_53706240": (
                orbit_mass == 53_706_240
            ),
            "ten_graphs_separated_by_exact_invariants": (
                len(graph_signatures) == 10
            ),
            "every_orbit_has_a_singleton_mixed_victim": all(
                row["singleton_mixed_victim_count"] > 0
                for row in class_payloads
            ),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if not all(payload["exact_checks"].values()):
        raise KrennN10PairwiseHamiltonianError(
            "an exact census check is false"
        )
    return _canonical_json_bytes(payload)


def build_census_payload() -> dict[str, object]:
    """Recompute and return a fresh exact census payload."""

    return json.loads(_census_bytes().decode("utf-8"))


def _strict_json(path: Path) -> object:
    def pairs(values: Iterable[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise KrennN10PairwiseHamiltonianError(
                    f"{path.name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_float(value: str) -> object:
        raise KrennN10PairwiseHamiltonianError(
            f"{path.name} contains non-integer number {value}"
        )

    try:
        return json.loads(
            path.read_text("utf-8"),
            object_pairs_hook=pairs,
            parse_float=reject_float,
            parse_constant=reject_float,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KrennN10PairwiseHamiltonianError(
            f"could not read strict JSON from {path}"
        ) from error


def _file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _false_claims(payload: object) -> bool:
    return (
        type(payload) is dict
        and payload == CLAIM_BOUNDARY
        and all(value is False for value in payload.values())
    )


def write_bundle(directory: Path) -> None:
    """Write ``census.json`` and a hash manifest beside a prepared README."""

    directory = directory.resolve()
    readme = directory / "README.md"
    if not readme.is_file():
        raise KrennN10PairwiseHamiltonianError(
            "the result README must exist before writing the bundle"
        )
    census = directory / "census.json"
    census.write_bytes(_census_bytes())
    root = _repo_root()
    source_records = {
        filename: _file_record(root / filename)
        for filename in SOURCE_FILES
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "bundle_files": {
            "README.md": _file_record(readme),
            "census.json": _file_record(census),
        },
        "source_files": source_records,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (directory / "manifest.json").write_bytes(
        _canonical_json_bytes(manifest)
    )
    verify_bundle(directory)


def verify_bundle(directory: Path) -> None:
    """Fail closed unless a committed bundle exactly recomputes."""

    directory = directory.resolve()
    expected_inventory = {
        "README.md",
        "census.json",
        "manifest.json",
    }
    if (
        not directory.is_dir()
        or {path.name for path in directory.iterdir()}
        != expected_inventory
    ):
        raise KrennN10PairwiseHamiltonianError(
            "the result bundle inventory changed"
        )
    manifest = _strict_json(directory / "manifest.json")
    if (
        type(manifest) is not dict
        or set(manifest)
        != {
            "schema",
            "bundle_files",
            "source_files",
            "claim_boundary",
        }
        or manifest["schema"] != MANIFEST_SCHEMA
        or not _false_claims(manifest["claim_boundary"])
        or set(manifest["bundle_files"])
        != {"README.md", "census.json"}
        or set(manifest["source_files"]) != set(SOURCE_FILES)
    ):
        raise KrennN10PairwiseHamiltonianError(
            "the result manifest schema or claim boundary changed"
        )
    root = _repo_root()
    for filename, record in manifest["bundle_files"].items():
        if record != _file_record(directory / filename):
            raise KrennN10PairwiseHamiltonianError(
                f"bundle hash mismatch for {filename}"
            )
    for filename, record in manifest["source_files"].items():
        if record != _file_record(root / filename):
            raise KrennN10PairwiseHamiltonianError(
                f"source hash mismatch for {filename}"
            )
    census_path = directory / "census.json"
    census = _strict_json(census_path)
    if (
        type(census) is not dict
        or census.get("schema") != CENSUS_SCHEMA
        or not _false_claims(census.get("claim_boundary"))
        or census["claim_boundary"] != manifest["claim_boundary"]
        or _canonical_json_bytes(census) != census_path.read_bytes()
        or census_path.read_bytes() != _census_bytes()
    ):
        raise KrennN10PairwiseHamiltonianError(
            "the committed census does not exactly recompute"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
    )
    arguments = parser.parse_args(argv)
    if arguments.verify_only:
        verify_bundle(arguments.results_directory)
    else:
        write_bundle(arguments.results_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
