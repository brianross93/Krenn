"""Exact ``S_6 x S_3`` orbit indexing for three-matching ternary seeds.

A seed chooses one perfect matching of ``K_6`` for each of three colors.
This module indexes those choices under vertex and color transport.  It does
not identify any weight variables, search signs, or assert a no-go result.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement, permutations, product
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.system import (
    Matching,
    perfect_matchings,
)
from experiments.krenn_quantum_graph.transport import (
    validate_permutation,
)


SEED_ORBIT_SCHEMA = "krenn-quantum-graph-ternary-seed-orbits-v1"
EXPECTED_REPRESENTATIVES_AND_SIZES = (
    ((0, 0, 0), 15),
    ((0, 0, 1), 270),
    ((0, 0, 4), 360),
    ((0, 1, 2), 90),
    ((0, 1, 3), 1080),
    ((0, 1, 5), 1080),
    ((0, 4, 8), 360),
    ((0, 4, 13), 120),
)


class KrennSeedOrbitError(ValueError):
    """Seed transport, census construction, or payload replay failed."""


def _canonical_matching(matching: Sequence[Sequence[int]]) -> Matching:
    edges = tuple(
        sorted(
            (
                (min(map(int, edge)), max(map(int, edge)))
                for edge in matching
            )
        )
    )
    if (
        len(edges) != 3
        or len({vertex for edge in edges for vertex in edge}) != 6
    ):
        raise KrennSeedOrbitError(
            "transported edges do not form a perfect matching"
        )
    return edges


@lru_cache(maxsize=1)
def _matching_index() -> Mapping[Matching, int]:
    return {
        matching: index
        for index, matching in enumerate(perfect_matchings(6))
    }


def transport_matching(
    matching: Matching, vertex_permutation: Sequence[int]
) -> Matching:
    """Push one matching forward under an old-to-new vertex map."""

    vertex_permutation = validate_permutation(
        vertex_permutation, 6
    )
    return _canonical_matching(
        (
            (
                vertex_permutation[i],
                vertex_permutation[j],
            )
            for i, j in matching
        )
    )


@lru_cache(maxsize=1)
def vertex_action_table() -> tuple[tuple[int, ...], ...]:
    """The exact action of all 720 vertex permutations on 15 matchings."""

    matchings = perfect_matchings(6)
    index = _matching_index()
    rows = tuple(
        tuple(
            index[transport_matching(matching, permutation)]
            for matching in matchings
        )
        for permutation in permutations(range(6))
    )
    if len(rows) != 720 or any(
        set(row) != set(range(15)) for row in rows
    ):
        raise KrennSeedOrbitError(
            "vertex action is not 720 permutations of 15 matchings"
        )
    return rows


def _validate_seed(seed: Sequence[int]) -> tuple[int, int, int]:
    seed = tuple(map(int, seed))
    if len(seed) != 3 or any(index < 0 or index >= 15 for index in seed):
        raise KrennSeedOrbitError(
            "a ternary seed needs three valid matching indices"
        )
    return seed


def transport_ordered_seed(
    seed: Sequence[int],
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, int, int]:
    """Transport colors and matchings without identifying any weights."""

    seed = _validate_seed(seed)
    vertex_permutation = validate_permutation(
        vertex_permutation, 6
    )
    color_permutation = validate_permutation(
        color_permutation, 3
    )
    matching_index = _matching_index()
    matchings = perfect_matchings(6)
    result = [0, 0, 0]
    for old_color, source_index in enumerate(seed):
        target_color = color_permutation[old_color]
        result[target_color] = matching_index[
            transport_matching(
                matchings[source_index], vertex_permutation
            )
        ]
    return tuple(result)


def canonical_seed_representative(
    seed: Sequence[int],
) -> tuple[int, int, int]:
    """Return the least matching-index triple in its ``S_6 x S_3`` orbit."""

    seed = _validate_seed(seed)
    # The full S_3 action is exactly all reorderings of the three transported
    # matching indices, so sorting gives its least color representative.
    return min(
        tuple(sorted((row[seed[0]], row[seed[1]], row[seed[2]])))
        for row in vertex_action_table()
    )


def _explicit_ordered_orbit(
    representative: Sequence[int],
) -> set[tuple[int, int, int]]:
    representative = _validate_seed(representative)
    return {
        transport_ordered_seed(
            representative,
            vertex_permutation,
            color_permutation,
        )
        for vertex_permutation in permutations(range(6))
        for color_permutation in permutations(range(3))
    }


def _matching_payload(matching_index: int) -> list[list[int]]:
    return [
        [i, j] for i, j in perfect_matchings(6)[matching_index]
    ]


def _build_census_payload() -> dict:
    matchings = perfect_matchings(6)
    ordered_seeds = tuple(product(range(len(matchings)), repeat=3))
    color_unordered = tuple(
        combinations_with_replacement(range(len(matchings)), 3)
    )
    grouped: dict[tuple[int, int, int], int] = {}
    for seed in ordered_seeds:
        representative = canonical_seed_representative(seed)
        grouped[representative] = grouped.get(representative, 0) + 1
    rows = tuple(sorted(grouped.items()))
    if rows != EXPECTED_REPRESENTATIVES_AND_SIZES:
        raise KrennSeedOrbitError(
            "the deterministic eight-orbit regression changed"
        )

    orbit_rows = []
    for orbit_index, (representative, orbit_size) in enumerate(rows):
        explicit_orbit = _explicit_ordered_orbit(representative)
        if (
            len(explicit_orbit) != orbit_size
            or any(
                canonical_seed_representative(seed)
                != representative
                for seed in explicit_orbit
            )
        ):
            raise KrennSeedOrbitError(
                "an explicit transported orbit failed replay"
            )
        orbit_rows.append(
            {
                "orbit_index": orbit_index,
                "representative_matching_indices": list(
                    representative
                ),
                "representative_matchings": [
                    _matching_payload(index)
                    for index in representative
                ],
                "ordered_seed_orbit_size": orbit_size,
                "stabilizer_size_in_S6_x_S3": (
                    720 * 6 // orbit_size
                ),
            }
        )

    return {
        "schema": SEED_ORBIT_SCHEMA,
        "parameters": {
            "n": 6,
            "d": 3,
            "colors_ordered_before_quotient": True,
        },
        "action": {
            "group": "S_6 x S_3",
            "vertex_permutations": 720,
            "color_permutations": 6,
            "canonicalization": (
                "least matching-index triple after vertex transport "
                "and arbitrary color reordering"
            ),
        },
        "counts": {
            "perfect_matchings": len(matchings),
            "ordered_three_matching_seeds": len(ordered_seeds),
            "color_unordered_matching_multisets": len(
                color_unordered
            ),
            "symmetry_orbits": len(rows),
            "sum_of_ordered_orbit_sizes": sum(
                orbit_size for _representative, orbit_size in rows
            ),
        },
        "orbits": orbit_rows,
        "exact_checks": {
            "matching_census_15": len(matchings) == 15,
            "ordered_seed_census_3375": len(ordered_seeds) == 3375,
            "color_unordered_multiset_census_680": (
                len(color_unordered) == 680
            ),
            "eight_orbits": len(rows) == 8,
            "orbit_sizes_sum_to_3375": (
                sum(size for _representative, size in rows) == 3375
            ),
            "all_explicit_orbits_replayed": True,
        },
        "claim_boundary": {
            "orbit_indexing_only": True,
            "symmetry_weight_equalities": 0,
            "sign_search_performed": False,
            "no_go_claimed": False,
            "nonexistence_proved": False,
        },
    }


def ternary_seed_orbit_census() -> dict:
    """Return a fresh JSON-ready deterministic census."""

    return json.loads(json.dumps(_build_census_payload()))


def verify_ternary_seed_orbit_census(payload: Mapping) -> dict:
    """Recompute the full census and reject any payload corruption."""

    try:
        normalized = json.loads(
            json.dumps(payload, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise KrennSeedOrbitError(
            "seed-orbit payload is not JSON-ready"
        ) from error
    expected = _build_census_payload()
    if normalized != expected:
        raise KrennSeedOrbitError(
            "seed-orbit payload failed exact census replay"
        )
    return normalized
