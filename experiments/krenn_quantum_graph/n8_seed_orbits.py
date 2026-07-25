"""Exact ``S_8 x S_3`` census for monochromatic matching triples.

A seed chooses one perfect matching of ``K_8`` for each of three colors.
The twelve corresponding diagonal source coordinates are nonzero in any
branch obtained by selecting one nonzero term from each monochromatic
equation.

This module performs the complete ordered-triple orbit census using only
nine involutory generators: seven adjacent vertex swaps and two adjacent
color swaps.  It also replays every mixed equation supported on the twelve
seed coordinates and records the two representatives for which every pair
of seed matchings forms a Hamilton cycle.

The ``H6`` representative has a primitive six-term relation in the *colored
source-coordinate* exponent lattice.  It is not identified with the
uncolored ``K3,3``-plus-common-edge circuit.  The ``H5`` representative's
five internal colored monomials are independent over ``Q``.

These are exact finite combinatorial statements.  A singleton on the seed
coordinates only forces a hypothetical complex solution to activate a
repair term; it is not by itself a nonexistence proof.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import json
from math import factorial, gcd
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.system import (
    Coloring,
    Matching,
    canonical_edges,
    coloring_index,
    perfect_matchings,
    variable_count,
    variable_index,
)


N = 8
D = 3
MATCHING_COUNT = 105
ORDERED_SEED_COUNT = MATCHING_COUNT**D
GENERATOR_COUNT = 7 + 2
GROUP_ORDER = factorial(N) * factorial(D)

N8_SEED_ORBIT_SCHEMA = "krenn-n8-seed-orbit-circuit-census-v1"
H5_REPRESENTATIVE = (0, 19, 38)
H6_REPRESENTATIVE = (0, 19, 43)

EXPECTED_REPRESENTATIVES_SIZES_AND_SINGLETONS = (
    ((0, 0, 0), 105, 78),
    ((0, 0, 1), 3_780, 42),
    ((0, 0, 4), 10_080, 24),
    ((0, 0, 16), 3_780, 22),
    ((0, 0, 19), 15_120, 14),
    ((0, 1, 2), 1_260, 24),
    ((0, 1, 3), 30_240, 21),
    ((0, 1, 5), 30_240, 12),
    ((0, 1, 15), 7_560, 22),
    ((0, 1, 17), 7_560, 12),
    ((0, 1, 18), 120_960, 11),
    ((0, 1, 20), 60_480, 6),
    ((0, 1, 52), 60_480, 10),
    ((0, 1, 58), 30_240, 6),
    ((0, 4, 8), 10_080, 9),
    ((0, 4, 13), 3_360, 15),
    ((0, 4, 16), 60_480, 10),
    ((0, 4, 17), 120_960, 5),
    ((0, 4, 21), 120_960, 5),
    ((0, 4, 23), 120_960, 4),
    ((0, 4, 27), 60_480, 3),
    ((0, 4, 28), 60_480, 7),
    ((0, 4, 29), 20_160, 9),
    ((0, 16, 32), 1_260, 6),
    ((0, 16, 35), 30_240, 2),
    ((0, 16, 52), 5_040, 6),
    ((0, 16, 53), 15_120, 4),
    ((0, 16, 55), 15_120, 6),
    ((0, 16, 56), 30_240, 4),
    (H5_REPRESENTATIVE, 60_480, 2),
    (H6_REPRESENTATIVE, 40_320, 3),
)


class KrennN8SeedOrbitError(ValueError):
    """The generator action, census, or exact payload failed replay."""


def _exact_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KrennN8SeedOrbitError(f"{label} must be an exact integer")
    return value


def _validate_seed(seed: Sequence[int]) -> tuple[int, int, int]:
    try:
        values = tuple(
            _exact_integer(value, "matching index") for value in seed
        )
    except TypeError as error:
        raise KrennN8SeedOrbitError(
            "a seed must be a finite sequence"
        ) from error
    if len(values) != D or any(
        value < 0 or value >= MATCHING_COUNT for value in values
    ):
        raise KrennN8SeedOrbitError(
            "an n=8 ternary seed needs three valid matching indices"
        )
    return values


def _canonical_matching(
    matching: Sequence[Sequence[int]],
) -> Matching:
    try:
        edges = tuple(
            sorted(
                (
                    (
                        min(
                            _exact_integer(edge[0], "edge endpoint"),
                            _exact_integer(edge[1], "edge endpoint"),
                        ),
                        max(
                            _exact_integer(edge[0], "edge endpoint"),
                            _exact_integer(edge[1], "edge endpoint"),
                        ),
                    )
                    for edge in matching
                )
            )
        )
    except (IndexError, TypeError) as error:
        raise KrennN8SeedOrbitError(
            "a transported matching contains a malformed edge"
        ) from error
    vertices = tuple(vertex for edge in edges for vertex in edge)
    if (
        len(edges) != N // 2
        or any(
            first == second
            or first < 0
            or second >= N
            for first, second in edges
        )
        or tuple(sorted(vertices)) != tuple(range(N))
    ):
        raise KrennN8SeedOrbitError(
            "transported edges do not form a perfect matching of K8"
        )
    return edges


@lru_cache(maxsize=1)
def _matching_index() -> Mapping[Matching, int]:
    matchings = perfect_matchings(N)
    if len(matchings) != MATCHING_COUNT:
        raise KrennN8SeedOrbitError(
            "the K8 perfect-matching census changed"
        )
    return {
        matching: index for index, matching in enumerate(matchings)
    }


def _transport_matching(
    matching: Matching, vertex_permutation: Sequence[int]
) -> Matching:
    permutation = tuple(vertex_permutation)
    if (
        len(permutation) != N
        or tuple(sorted(permutation)) != tuple(range(N))
    ):
        raise KrennN8SeedOrbitError(
            "a vertex action must be a permutation of range(8)"
        )
    return _canonical_matching(
        (
            (permutation[first], permutation[second])
            for first, second in matching
        )
    )


@lru_cache(maxsize=1)
def adjacent_vertex_action_rows() -> tuple[tuple[int, ...], ...]:
    """Return the seven adjacent-transposition actions on 105 matchings."""

    matchings = perfect_matchings(N)
    matching_index = _matching_index()
    rows = []
    for position in range(N - 1):
        permutation = list(range(N))
        permutation[position], permutation[position + 1] = (
            permutation[position + 1],
            permutation[position],
        )
        row = tuple(
            matching_index[
                _transport_matching(matching, permutation)
            ]
            for matching in matchings
        )
        if (
            set(row) != set(range(MATCHING_COUNT))
            or any(
                row[row[index]] != index
                for index in range(MATCHING_COUNT)
            )
        ):
            raise KrennN8SeedOrbitError(
                "an adjacent vertex action is not an involution"
            )
        rows.append(row)
    if len(rows) != N - 1:
        raise KrennN8SeedOrbitError(
            "the adjacent vertex generator census changed"
        )
    return tuple(rows)


def n8_seed_generator_neighbors(
    seed: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """Apply seven vertex and two color adjacent-transposition generators."""

    first, second, third = _validate_seed(seed)
    neighbors = [
        (row[first], row[second], row[third])
        for row in adjacent_vertex_action_rows()
    ]
    neighbors.extend(
        (
            (second, first, third),
            (first, third, second),
        )
    )
    if len(neighbors) != GENERATOR_COUNT:
        raise KrennN8SeedOrbitError(
            "the seed generator census changed"
        )
    return tuple(neighbors)


def _encode_seed(seed: tuple[int, int, int]) -> int:
    first, second, third = seed
    return (
        first * MATCHING_COUNT + second
    ) * MATCHING_COUNT + third


def _decode_seed(index: int) -> tuple[int, int, int]:
    index = _exact_integer(index, "encoded seed")
    if index < 0 or index >= ORDERED_SEED_COUNT:
        raise KrennN8SeedOrbitError(
            "an encoded seed is outside the ordered census"
        )
    first, remainder = divmod(
        index, MATCHING_COUNT * MATCHING_COUNT
    )
    second, third = divmod(remainder, MATCHING_COUNT)
    return first, second, third


@lru_cache(maxsize=1)
def _generator_orbit_data(
) -> tuple[
    tuple[tuple[tuple[int, int, int], int], ...],
    bytes,
]:
    """Traverse all ordered seeds by generator-action BFS."""

    vertex_rows = adjacent_vertex_action_rows()
    orbit_labels = bytearray(ORDERED_SEED_COUNT)
    orbit_rows = []
    for root in range(ORDERED_SEED_COUNT):
        if orbit_labels[root]:
            continue
        orbit_label = len(orbit_rows) + 1
        if orbit_label > 255:
            raise KrennN8SeedOrbitError(
                "the orbit label no longer fits the exact byte census"
            )
        orbit_labels[root] = orbit_label
        queue = [root]
        cursor = 0
        while cursor < len(queue):
            encoded = queue[cursor]
            cursor += 1
            first, second, third = _decode_seed(encoded)
            generated = [
                (
                    row[first] * MATCHING_COUNT + row[second]
                ) * MATCHING_COUNT + row[third]
                for row in vertex_rows
            ]
            generated.extend(
                (
                    (
                        second * MATCHING_COUNT + first
                    ) * MATCHING_COUNT + third,
                    (
                        first * MATCHING_COUNT + third
                    ) * MATCHING_COUNT + second,
                )
            )
            if len(generated) != GENERATOR_COUNT:
                raise KrennN8SeedOrbitError(
                    "the BFS generator census changed"
                )
            for neighbor in generated:
                if not orbit_labels[neighbor]:
                    orbit_labels[neighbor] = orbit_label
                    queue.append(neighbor)
        orbit_rows.append((_decode_seed(root), len(queue)))

    expected_rows = tuple(
        (representative, orbit_size)
        for representative, orbit_size, _victims
        in EXPECTED_REPRESENTATIVES_SIZES_AND_SINGLETONS
    )
    rows = tuple(orbit_rows)
    if (
        rows != expected_rows
        or len(rows) != 31
        or sum(size for _representative, size in rows)
        != ORDERED_SEED_COUNT
        or orbit_labels.count(0)
    ):
        raise KrennN8SeedOrbitError(
            "the deterministic 31-orbit BFS regression changed"
        )
    return rows, bytes(orbit_labels)


def canonical_n8_seed_representative(
    seed: Sequence[int],
) -> tuple[int, int, int]:
    """Return the least encoded seed in its generator-action orbit."""

    checked = _validate_seed(seed)
    rows, labels = _generator_orbit_data()
    orbit_index = labels[_encode_seed(checked)] - 1
    if orbit_index < 0 or orbit_index >= len(rows):
        raise KrennN8SeedOrbitError(
            "a seed has no generator-action orbit label"
        )
    return rows[orbit_index][0]


def _matching_payload(matching_index: int) -> list[list[int]]:
    matching_index = _exact_integer(
        matching_index, "matching payload index"
    )
    if matching_index < 0 or matching_index >= MATCHING_COUNT:
        raise KrennN8SeedOrbitError(
            "matching payload index is outside range(105)"
        )
    return [
        [first, second]
        for first, second in perfect_matchings(N)[matching_index]
    ]


def _monomial_variable_indices(
    matching_index: int, coloring: Coloring
) -> tuple[int, ...]:
    matching = perfect_matchings(N)[matching_index]
    return tuple(
        variable_index(
            N,
            D,
            first,
            second,
            coloring[first],
            coloring[second],
        )
        for first, second in matching
    )


def _seed_support(seed: tuple[int, int, int]) -> tuple[int, ...]:
    support = tuple(
        sorted(
            variable_index(
                N, D, first, second, color, color
            )
            for color, matching_index in enumerate(seed)
            for first, second in perfect_matchings(N)[matching_index]
        )
    )
    if len(support) != 12 or len(set(support)) != 12:
        raise KrennN8SeedOrbitError(
            "a matching triple did not force twelve source coordinates"
        )
    return support


def _singleton_mixed_victims(
    seed: tuple[int, int, int],
) -> tuple[dict[str, object], ...]:
    """Enumerate all active matching terms on the twelve seed coordinates."""

    edge_colors: dict[tuple[int, int], list[int]] = {}
    for color, matching_index in enumerate(seed):
        for edge in perfect_matchings(N)[matching_index]:
            edge_colors.setdefault(edge, []).append(color)

    terms_by_coloring: dict[
        Coloring, list[int]
    ] = {}
    for matching_index, matching in enumerate(perfect_matchings(N)):
        choices = tuple(
            tuple(edge_colors.get(edge, ())) for edge in matching
        )
        if any(not edge_choices for edge_choices in choices):
            continue
        for edge_colors_selected in product(*choices):
            coloring = [-1] * N
            for edge, color in zip(
                matching, edge_colors_selected, strict=True
            ):
                coloring[edge[0]] = color
                coloring[edge[1]] = color
            checked_coloring = tuple(coloring)
            if any(color < 0 for color in checked_coloring):
                raise KrennN8SeedOrbitError(
                    "an active seed term left a vertex uncolored"
                )
            terms_by_coloring.setdefault(
                checked_coloring, []
            ).append(matching_index)

    seed_support = set(_seed_support(seed))
    victims = []
    for coloring, active_matchings in sorted(
        terms_by_coloring.items(),
        key=lambda item: coloring_index(N, D, item[0]),
    ):
        if all(color == coloring[0] for color in coloring):
            continue
        if len(active_matchings) != 1:
            raise KrennN8SeedOrbitError(
                "an active mixed seed equation is not a singleton"
            )
        matching_index = active_matchings[0]
        monomial = _monomial_variable_indices(
            matching_index, coloring
        )
        if not set(monomial) <= seed_support:
            raise KrennN8SeedOrbitError(
                "a seed victim monomial escapes the seed support"
            )
        victims.append(
            {
                "coloring": list(coloring),
                "coloring_index": coloring_index(
                    N, D, coloring
                ),
                "matching_index": matching_index,
                "matching": _matching_payload(matching_index),
                "source_monomial_variable_indices": list(
                    monomial
                ),
            }
        )
    return tuple(victims)


def _physical_union(
    seed: tuple[int, int, int],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                edge
                for matching_index in seed
                for edge in perfect_matchings(N)[matching_index]
            }
        )
    )


def _pair_union_component_sizes(
    first_matching: int, second_matching: int
) -> tuple[int, ...]:
    adjacency = [set() for _vertex in range(N)]
    for matching_index in (first_matching, second_matching):
        for first, second in perfect_matchings(N)[matching_index]:
            adjacency[first].add(second)
            adjacency[second].add(first)
    seen = set()
    sizes = []
    for root in range(N):
        if root in seen:
            continue
        seen.add(root)
        queue = [root]
        cursor = 0
        while cursor < len(queue):
            vertex = queue[cursor]
            cursor += 1
            for neighbor in adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        sizes.append(len(queue))
    return tuple(sorted(sizes, reverse=True))


def _pair_profiles(
    seed: tuple[int, int, int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        _pair_union_component_sizes(seed[first], seed[second])
        for first, second in ((0, 1), (0, 2), (1, 2))
    )


def _internal_matching_indices(
    seed: tuple[int, int, int],
) -> tuple[int, ...]:
    union = set(_physical_union(seed))
    return tuple(
        matching_index
        for matching_index, matching in enumerate(
            perfect_matchings(N)
        )
        if set(matching) <= union
    )


def _graph_profile(
    edges: Sequence[tuple[int, int]],
) -> dict[str, object]:
    adjacency = [set() for _vertex in range(N)]
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    seen = {0}
    queue = [0]
    cursor = 0
    while cursor < len(queue):
        vertex = queue[cursor]
        cursor += 1
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    colors: dict[int, int] = {}
    bipartite = True
    for root in range(N):
        if root in colors:
            continue
        colors[root] = 0
        queue = [root]
        cursor = 0
        while cursor < len(queue):
            vertex = queue[cursor]
            cursor += 1
            for neighbor in adjacency[vertex]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[vertex]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    bipartite = False
    triangles = sum(
        int(
            second in adjacency[first]
            and third in adjacency[first]
            and third in adjacency[second]
        )
        for first, second, third in combinations(range(N), 3)
    )
    return {
        "connected": len(seen) == N,
        "bipartite": bipartite,
        "triangle_count": triangles,
    }


def _colored_internal_records(
    seed: tuple[int, int, int],
) -> tuple[dict[str, object], ...]:
    edge_color: dict[tuple[int, int], int] = {}
    for color, matching_index in enumerate(seed):
        for edge in perfect_matchings(N)[matching_index]:
            if edge in edge_color:
                raise KrennN8SeedOrbitError(
                    "a hard representative is not edge-disjoint by color"
                )
            edge_color[edge] = color

    records = []
    for matching_index in _internal_matching_indices(seed):
        coloring = [-1] * N
        monomial = []
        for first, second in perfect_matchings(N)[matching_index]:
            color = edge_color[(first, second)]
            coloring[first] = color
            coloring[second] = color
            monomial.append(
                variable_index(
                    N, D, first, second, color, color
                )
            )
        checked_coloring = tuple(coloring)
        records.append(
            {
                "matching_index": matching_index,
                "matching": _matching_payload(matching_index),
                "inherited_coloring": list(checked_coloring),
                "coloring_index": coloring_index(
                    N, D, checked_coloring
                ),
                "monochromatic": all(
                    color == checked_coloring[0]
                    for color in checked_coloring
                ),
                "source_monomial_variable_indices": monomial,
            }
        )
    return tuple(records)


def _exponent_vector(
    monomial: Sequence[int],
) -> tuple[int, ...]:
    dimension = variable_count(N, D)
    vector = [0] * dimension
    for raw_variable in monomial:
        variable = _exact_integer(
            raw_variable, "monomial variable"
        )
        if variable < 0 or variable >= dimension:
            raise KrennN8SeedOrbitError(
                "a monomial variable is outside the source"
            )
        vector[variable] += 1
    return tuple(vector)


def _physical_incidence_vector(
    matching_index: int,
) -> tuple[int, ...]:
    edge_position = {
        edge: position
        for position, edge in enumerate(canonical_edges(N))
    }
    vector = [0] * len(edge_position)
    for edge in perfect_matchings(N)[matching_index]:
        vector[edge_position[edge]] += 1
    return tuple(vector)


def _rational_rank(vectors: Sequence[Sequence[int]]) -> int:
    matrix = [
        [Fraction(value) for value in vector]
        for vector in vectors
    ]
    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise KrennN8SeedOrbitError(
            "an exponent matrix is ragged"
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


def _hard_case_payload(
    label: str, seed: tuple[int, int, int]
) -> dict[str, object]:
    records = _colored_internal_records(seed)
    vectors = tuple(
        _exponent_vector(
            record["source_monomial_variable_indices"]
        )
        for record in records
    )
    rank = _rational_rank(vectors)
    internal_indices = tuple(
        record["matching_index"] for record in records
    )
    seed_indices = set(seed)
    mixed_indices = tuple(
        matching_index
        for matching_index in internal_indices
        if matching_index not in seed_indices
    )
    victim_colorings = {
        tuple(victim["coloring"])
        for victim in _singleton_mixed_victims(seed)
    }
    if any(
        tuple(record["inherited_coloring"])
        not in victim_colorings
        for record in records
        if not record["monochromatic"]
    ):
        raise KrennN8SeedOrbitError(
            "an internal mixed hard-case term is not a singleton victim"
        )
    payload: dict[str, object] = {
        "label": label,
        "representative_matching_indices": list(seed),
        "pair_union_component_sizes": [
            list(profile) for profile in _pair_profiles(seed)
        ],
        "physical_union_edges": [
            list(edge) for edge in _physical_union(seed)
        ],
        "physical_union_edge_count": len(_physical_union(seed)),
        "physical_union_graph": _graph_profile(
            _physical_union(seed)
        ),
        "internal_matching_count": len(records),
        "internal_matching_indices": list(internal_indices),
        "internal_colored_monomials": list(records),
        "colored_exponent_matrix": {
            "ambient_source_coordinates": variable_count(N, D),
            "columns": len(vectors),
            "rank_over_Q": rank,
            "nullity_over_Q": len(vectors) - rank,
        },
    }
    if label == "H5":
        if (
            seed != H5_REPRESENTATIVE
            or internal_indices != (0, 1, 14, 19, 38)
            or len(mixed_indices) != 2
            or rank != 5
        ):
            raise KrennN8SeedOrbitError(
                "the exact H5 independence regression changed"
            )
        payload["internal_relation"] = {
            "exists_over_Q": False,
            "scope": (
                "the five colored monomial exponent columns internal "
                "to the H5 physical union only"
            ),
        }
        return payload

    if label != "H6" or seed != H6_REPRESENTATIVE:
        raise KrennN8SeedOrbitError("unknown hard representative")
    if (
        internal_indices != (0, 1, 19, 29, 33, 43)
        or mixed_indices != (1, 29, 33)
        or rank != 5
    ):
        raise KrennN8SeedOrbitError(
            "the exact H6 circuit regression changed"
        )
    coefficients = tuple(
        -1 if matching_index in seed_indices else 1
        for matching_index in internal_indices
    )
    physical_vectors = tuple(
        _physical_incidence_vector(matching_index)
        for matching_index in internal_indices
    )
    signed_physical_sum = tuple(
        sum(
            coefficient * vector[coordinate]
            for coefficient, vector in zip(
                coefficients, physical_vectors, strict=True
            )
        )
        for coordinate in range(len(canonical_edges(N)))
    )
    signed_sum = tuple(
        sum(
            coefficient * vector[coordinate]
            for coefficient, vector in zip(
                coefficients, vectors, strict=True
            )
        )
        for coordinate in range(variable_count(N, D))
    )
    seed_product = tuple(
        sum(
            vector[coordinate]
            for matching_index, vector in zip(
                internal_indices, vectors, strict=True
            )
            if matching_index in seed_indices
        )
        for coordinate in range(variable_count(N, D))
    )
    mixed_product = tuple(
        sum(
            vector[coordinate]
            for matching_index, vector in zip(
                internal_indices, vectors, strict=True
            )
            if matching_index not in seed_indices
        )
        for coordinate in range(variable_count(N, D))
    )
    every_five_independent = all(
        _rational_rank(
            tuple(vectors[index] for index in subset)
        )
        == 5
        for subset in combinations(range(6), 5)
    )
    coefficient_gcd = 0
    for coefficient in coefficients:
        coefficient_gcd = gcd(
            coefficient_gcd, abs(coefficient)
        )
    common_product = tuple(
        coordinate
        for coordinate, multiplicity in enumerate(seed_product)
        if multiplicity
    )
    if (
        any(signed_physical_sum)
        or any(signed_sum)
        or seed_product != mixed_product
        or common_product != _seed_support(seed)
        or any(
            multiplicity not in (0, 1)
            for multiplicity in seed_product
        )
        or not every_five_independent
        or coefficient_gcd != 1
        or _graph_profile(_physical_union(seed))
        != {
            "connected": True,
            "bipartite": False,
            "triangle_count": 1,
        }
    ):
        raise KrennN8SeedOrbitError(
            "the colored-coordinate H6 product circuit failed"
        )
    payload["colored_product_circuit"] = {
        "relation": [
            {
                "matching_index": matching_index,
                "coefficient": coefficient,
            }
            for matching_index, coefficient in zip(
                internal_indices, coefficients, strict=True
            )
        ],
        "seed_side_matching_indices": list(seed),
        "mixed_side_matching_indices": list(mixed_indices),
        "signed_physical_edge_incidence_sum_nonzero_entries": [],
        "signed_exponent_sum_nonzero_entries": [],
        "common_product_variable_indices": list(common_product),
        "each_common_product_coordinate_multiplicity": 1,
        "primitive_integer_relation": coefficient_gcd == 1,
        "all_proper_five_column_subsets_independent": (
            every_five_independent
        ),
        "embedded_uncolored_K3_3_plus_common_edge_claimed": False,
    }
    return payload


@lru_cache(maxsize=1)
def _build_census_payload() -> dict[str, object]:
    orbit_rows, labels = _generator_orbit_data()
    if len(labels) != ORDERED_SEED_COUNT:
        raise KrennN8SeedOrbitError(
            "the generator-action label census is incomplete"
        )
    expected_singletons = {
        representative: singleton_count
        for representative, _size, singleton_count
        in EXPECTED_REPRESENTATIVES_SIZES_AND_SINGLETONS
    }
    rows = []
    for orbit_index, (representative, orbit_size) in enumerate(
        orbit_rows
    ):
        victims = _singleton_mixed_victims(representative)
        if len(victims) != expected_singletons[representative]:
            raise KrennN8SeedOrbitError(
                "a singleton-victim regression changed"
            )
        internal_count = len(
            _internal_matching_indices(representative)
        )
        hard_case = (
            "H5"
            if representative == H5_REPRESENTATIVE
            else "H6"
            if representative == H6_REPRESENTATIVE
            else None
        )
        if GROUP_ORDER % orbit_size:
            raise KrennN8SeedOrbitError(
                "an orbit size does not divide S8 x S3"
            )
        rows.append(
            {
                "orbit_index": orbit_index,
                "representative_matching_indices": list(
                    representative
                ),
                "representative_matchings": [
                    _matching_payload(matching_index)
                    for matching_index in representative
                ],
                "ordered_seed_orbit_size": orbit_size,
                "stabilizer_size_in_S8_x_S3": (
                    GROUP_ORDER // orbit_size
                ),
                "seed_source_variable_indices": list(
                    _seed_support(representative)
                ),
                "pair_union_component_sizes": [
                    list(profile)
                    for profile in _pair_profiles(representative)
                ],
                "physical_union_edge_count": len(
                    _physical_union(representative)
                ),
                "internal_physical_matching_count": internal_count,
                "hard_case": hard_case,
                "singleton_mixed_victim_count": len(victims),
                "singleton_mixed_victims": list(victims),
                "all_seed_supported_mixed_terms_are_singletons": True,
            }
        )

    h5 = _hard_case_payload("H5", H5_REPRESENTATIVE)
    h6 = _hard_case_payload("H6", H6_REPRESENTATIVE)
    return {
        "schema": N8_SEED_ORBIT_SCHEMA,
        "parameters": {
            "n": N,
            "d": D,
            "source_variable_count": variable_count(N, D),
            "colors_ordered_before_quotient": True,
        },
        "action": {
            "group": "S_8 x S_3",
            "group_order": GROUP_ORDER,
            "algorithm": (
                "generator-action BFS over all ordered triples"
            ),
            "full_S8_action_table_built": False,
            "adjacent_vertex_swap_generators": [
                [position, position + 1]
                for position in range(N - 1)
            ],
            "adjacent_color_swap_generators": [[0, 1], [1, 2]],
            "generator_count": GENERATOR_COUNT,
            "canonicalization": (
                "least base-105 encoded ordered triple in each "
                "generator orbit"
            ),
        },
        "counts": {
            "perfect_matchings": MATCHING_COUNT,
            "ordered_three_matching_seeds": ORDERED_SEED_COUNT,
            "symmetry_orbits": len(rows),
            "sum_of_ordered_orbit_sizes": sum(
                row["ordered_seed_orbit_size"] for row in rows
            ),
            "orbits_with_singleton_mixed_victims": sum(
                bool(row["singleton_mixed_victim_count"])
                for row in rows
            ),
            "pairwise_hamiltonian_hard_orbits": sum(
                all(
                    profile == [N]
                    for profile in row[
                        "pair_union_component_sizes"
                    ]
                )
                for row in rows
            ),
        },
        "orbits": rows,
        "hard_cases": {"H5": h5, "H6": h6},
        "exact_checks": {
            "matching_census_105": MATCHING_COUNT
            == len(perfect_matchings(N)),
            "nine_involutory_generators": (
                len(adjacent_vertex_action_rows()) + 2
                == GENERATOR_COUNT
            ),
            "ordered_seed_census_1157625": (
                ORDERED_SEED_COUNT == 1_157_625
            ),
            "thirty_one_S8_x_S3_orbits": len(rows) == 31,
            "orbit_sizes_sum_to_1157625": (
                sum(
                    row["ordered_seed_orbit_size"]
                    for row in rows
                )
                == ORDERED_SEED_COUNT
            ),
            "every_orbit_has_a_singleton_mixed_victim": all(
                row["singleton_mixed_victim_count"] > 0
                for row in rows
            ),
            "exactly_two_pairwise_hamiltonian_hard_orbits": (
                sum(
                    all(
                        profile == [N]
                        for profile in row[
                            "pair_union_component_sizes"
                        ]
                    )
                    for row in rows
                )
                == 2
            ),
            "H5_has_five_independent_internal_columns": (
                h5["colored_exponent_matrix"]["rank_over_Q"] == 5
                and h5["colored_exponent_matrix"][
                    "nullity_over_Q"
                ]
                == 0
            ),
            "H6_has_verified_primitive_product_circuit": (
                h6["colored_exponent_matrix"]["rank_over_Q"] == 5
                and h6["colored_exponent_matrix"][
                    "nullity_over_Q"
                ]
                == 1
                and h6["colored_product_circuit"][
                    "primitive_integer_relation"
                ]
                and h6["colored_product_circuit"][
                    "all_proper_five_column_subsets_independent"
                ]
            ),
        },
        "claim_boundary": {
            "scope": (
                "finite matching-triple orbit and seed-support "
                "colored-exponent census only"
            ),
            "symmetry_weight_equalities": 0,
            "numerical_search_performed": False,
            "singleton_victim_alone_excludes_complex_weights": False,
            "H5_has_no_relation_in_the_full_source_system": False,
            "H6_circuit_closes_unrestricted_spill_terms": False,
            "H6_odd_relation_alone_proves_contradiction": False,
            "classical_pfaffian_or_plucker_identity_claimed": False,
            "blocker_boundary_escape_proved": False,
            "n8_nonexistence_proved": False,
        },
    }


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise KrennN8SeedOrbitError(
            "the seed-orbit payload is not strict JSON"
        ) from error


def n8_seed_orbit_census() -> dict[str, object]:
    """Return a fresh JSON-ready copy of the deterministic exact census."""

    return json.loads(_canonical_json_bytes(_build_census_payload()))


def verify_n8_seed_orbit_census(
    payload: Mapping,
) -> dict[str, object]:
    """Reject any payload differing from a complete native recomputation."""

    if not isinstance(payload, Mapping):
        raise KrennN8SeedOrbitError(
            "the seed-orbit payload must be a mapping"
        )
    encoded = _canonical_json_bytes(payload)
    expected = _canonical_json_bytes(_build_census_payload())
    if encoded != expected:
        raise KrennN8SeedOrbitError(
            "the n=8 seed-orbit payload failed exact replay"
        )
    return json.loads(encoded)


__all__ = [
    "EXPECTED_REPRESENTATIVES_SIZES_AND_SINGLETONS",
    "GENERATOR_COUNT",
    "GROUP_ORDER",
    "H5_REPRESENTATIVE",
    "H6_REPRESENTATIVE",
    "KrennN8SeedOrbitError",
    "MATCHING_COUNT",
    "N8_SEED_ORBIT_SCHEMA",
    "ORDERED_SEED_COUNT",
    "adjacent_vertex_action_rows",
    "canonical_n8_seed_representative",
    "n8_seed_generator_neighbors",
    "n8_seed_orbit_census",
    "verify_n8_seed_orbit_census",
]
