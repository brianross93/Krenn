"""Exact toric first-shell gate for the ``n=8,d=3`` H5/H6 seeds.

The correct compactification for image-versus-border membership retains the
projective output direction: it is the closure of the graph of the matching
map (equivalently, a Rees/blow-up construction), not weight projectivization
alone.  This module does not construct that full compactification.  It
certifies one finite collection of its simplest candidate initial strata.

For each pairwise-Hamiltonian seed, the three monochromatic matching products
are normalized to order zero.  The target-preserving color-diagonal torus has
dimension 21; its restriction to the twelve seed coordinates has saturated
rank 9, so all twelve seed orders can be normalized to zero without root
extraction.

Each original mixed singleton has exactly twelve repairs using two new source
coordinates.  This module chooses one such repair for every original victim
and studies the *flat* two-level valuation with order zero on the resulting
support and positive order on every other source coordinate.  There are
``12^2=144`` H5 and ``12^3=1728`` H6 decorated choices, reduced by the seed
stabilizers to 43 and 304 strata.  Every stratum has a zero-target equation
whose initial form is one nonzero monomial.  Hence none can lie over the GHZ
direction in the graph compactification.

This excludes only the enumerated flat cost-two first shell.  It does not
classify different minimal layers, negative quotient directions, deeper
repairs, the other 29 seed orbits, the full saturated initial ideal, or
affine membership at ``n=8``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import permutations, product
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.krenn_quantum_graph.independent_verifier import (
    independent_perfect_matchings,
)
from experiments.krenn_quantum_graph.n8_seed_orbits import (
    H5_REPRESENTATIVE,
    H6_REPRESENTATIVE,
    n8_seed_orbit_census,
)
from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    coloring_from_index,
    coloring_index,
    perfect_matchings,
    variable_count,
    variable_index,
    variable_key,
)


N = 8
D = 3
SOURCE_VARIABLES = 252
COLORINGS = 6_561
MATCHING_COUNT = 105
GAUGE_DIMENSION = 21
FIRST_SHELL_SCHEMA = "krenn-n8-d3-toric-first-shell-v1"
FIRST_SHELL_MANIFEST_SCHEMA = (
    "krenn-n8-d3-toric-first-shell-manifest-v1"
)
DEFAULT_RESULTS_DIRECTORY = (
    Path("results")
    / "krenn_quantum_graph"
    / "n8_d3_toric_first_shell"
)
CERTIFICATE_FILE = "certificate.json"
README_FILE = "README.md"
MANIFEST_FILE = "manifest.json"
SOURCE_PATHS = (
    "experiments/krenn_quantum_graph/independent_verifier.py",
    "experiments/krenn_quantum_graph/n8_seed_orbits.py",
    "experiments/krenn_quantum_graph/n8_toric_first_shell.py",
    "experiments/krenn_quantum_graph/system.py",
    "tests/test_krenn_n8_toric_first_shell.py",
)

EXPECTED = {
    "H5": {
        "seed": H5_REPRESENTATIVE,
        "stabilizer": 4,
        "victims": 2,
        "raw": 144,
        "orbits": 43,
        "support_size": 16,
        "orbit_size_histogram": {1: 2, 2: 11, 4: 30},
        "singleton_histogram": {4: 8, 5: 20, 6: 75, 7: 13, 8: 20, 9: 8},
        "minimum_singletons": 4,
        "best_orbits": 3,
        "slice_determinant": -1,
    },
    "H6": {
        "seed": H6_REPRESENTATIVE,
        "stabilizer": 6,
        "victims": 3,
        "raw": 1_728,
        "orbits": 304,
        "support_size": 18,
        "orbit_size_histogram": {1: 2, 2: 5, 3: 22, 6: 275},
        "singleton_histogram": {
            6: 17,
            7: 78,
            8: 162,
            9: 212,
            10: 399,
            11: 234,
            12: 158,
            13: 180,
            14: 111,
            15: 96,
            16: 45,
            17: 27,
            20: 9,
        },
        "minimum_singletons": 6,
        "best_orbits": 4,
        "slice_determinant": 1,
    },
}


class KrennN8ToricFirstShellError(ValueError):
    """The toric first-shell census or its artifact failed exact replay."""


def _canonical_matching(
    matching: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    result = tuple(
        sorted(
            (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
            for edge in matching
        )
    )
    if (
        len(result) != N // 2
        or len({vertex for edge in result for vertex in edge}) != N
    ):
        raise KrennN8ToricFirstShellError(
            "a purported K8 perfect matching is malformed"
        )
    return result


@lru_cache(maxsize=1)
def _primary_matchings() -> tuple[tuple[tuple[int, int], ...], ...]:
    result = tuple(
        _canonical_matching(matching) for matching in perfect_matchings(N)
    )
    if len(result) != MATCHING_COUNT or len(set(result)) != MATCHING_COUNT:
        raise KrennN8ToricFirstShellError(
            "the primary K8 matching census changed"
        )
    return result


@lru_cache(maxsize=1)
def _independent_matchings() -> tuple[tuple[tuple[int, int], ...], ...]:
    result = tuple(
        _canonical_matching(matching)
        for matching in independent_perfect_matchings(N)
    )
    if (
        len(result) != MATCHING_COUNT
        or set(result) != set(_primary_matchings())
    ):
        raise KrennN8ToricFirstShellError(
            "the independent K8 matching census disagreed"
        )
    return result


def _rank_over_q(matrix: Sequence[Sequence[int | Fraction]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennN8ToricFirstShellError(
            "exact rank needs a rectangular matrix"
        )
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
            coefficient = rows[row][column]
            rows[row] = [
                value - coefficient * pivot_value
                for value, pivot_value in zip(
                    rows[row], rows[rank], strict=True
                )
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _determinant(matrix: Sequence[Sequence[int | Fraction]]) -> Fraction:
    rows = [list(map(Fraction, row)) for row in matrix]
    size = len(rows)
    if not size or any(len(row) != size for row in rows):
        raise KrennN8ToricFirstShellError(
            "exact determinant needs a nonempty square matrix"
        )
    result = Fraction(1)
    for column in range(size):
        selected = next(
            (
                row
                for row in range(column, size)
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            return Fraction(0)
        if selected != column:
            rows[column], rows[selected] = rows[selected], rows[column]
            result = -result
        pivot = rows[column][column]
        result *= pivot
        rows[column] = [value / pivot for value in rows[column]]
        for row in range(column + 1, size):
            coefficient = rows[row][column]
            if coefficient:
                rows[row] = [
                    value - coefficient * pivot_value
                    for value, pivot_value in zip(
                        rows[row], rows[column], strict=True
                    )
                ]
    return result


def _integer_matrix_product(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    left = tuple(tuple(map(int, row)) for row in left)
    right = tuple(tuple(map(int, row)) for row in right)
    if not left or not right or len(left[0]) != len(right):
        raise KrennN8ToricFirstShellError(
            "integer matrix-product dimensions differ"
        )
    width = len(right[0])
    if any(len(row) != width for row in right):
        raise KrennN8ToricFirstShellError(
            "the right integer matrix is ragged"
        )
    return tuple(
        tuple(
            sum(
                left[row][middle] * right[middle][column]
                for middle in range(len(right))
            )
            for column in range(width)
        )
        for row in range(len(left))
    )


@lru_cache(maxsize=1)
def target_preserving_gauge_matrix() -> tuple[tuple[int, ...], ...]:
    """Return the ``252 x 21`` GHZ-stabilizer cocharacter matrix."""

    if variable_count(N, D) != SOURCE_VARIABLES:
        raise KrennN8ToricFirstShellError(
            "the n=8,d=3 source dimension changed"
        )
    rows = []
    for index in range(SOURCE_VARIABLES):
        u, v, a, b = variable_key(N, D, index)
        row = []
        for color in range(D):
            for vertex in range(N - 1):
                row.append(
                    int(a == color and u == vertex)
                    - int(a == color and u == N - 1)
                    + int(b == color and v == vertex)
                    - int(b == color and v == N - 1)
                )
        rows.append(tuple(row))
    result = tuple(rows)
    if _rank_over_q(result) != GAUGE_DIMENSION:
        raise KrennN8ToricFirstShellError(
            "the target-preserving gauge stopped being faithful"
        )
    return result


def _hard_case_row(label: str) -> Mapping[str, object]:
    census = n8_seed_orbit_census()
    row = next(
        (
            candidate
            for candidate in census["orbits"]
            if candidate["hard_case"] == label
        ),
        None,
    )
    if row is None:
        raise KrennN8ToricFirstShellError(
            f"the {label} seed row is absent"
        )
    return row


def _greedy_minor(
    matrix: Sequence[Sequence[int]],
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    rows = tuple(tuple(map(int, row)) for row in matrix)
    columns: list[int] = []
    for column in range(len(rows[0])):
        candidate = (*columns, column)
        if (
            _rank_over_q(
                tuple(
                    tuple(row[index] for index in candidate)
                    for row in rows
                )
            )
            > len(columns)
        ):
            columns.append(column)
        if len(columns) == rank:
            break
    selected_rows: list[int] = []
    for row in range(len(rows)):
        candidate = (*selected_rows, row)
        if (
            _rank_over_q(
                tuple(
                    tuple(rows[index][column] for column in columns)
                    for index in candidate
                )
            )
            > len(selected_rows)
        ):
            selected_rows.append(row)
        if len(selected_rows) == rank:
            break
    minor = tuple(
        tuple(rows[row][column] for column in columns)
        for row in selected_rows
    )
    determinant = _determinant(minor)
    if determinant.denominator != 1:
        raise KrennN8ToricFirstShellError(
            "the gauge slice determinant is nonintegral"
        )
    return tuple(selected_rows), tuple(columns), determinant.numerator


def _seed_gauge_record(label: str) -> dict[str, object]:
    expected = EXPECTED[label]
    row = _hard_case_row(label)
    support = tuple(map(int, row["seed_source_variable_indices"]))
    gauge = target_preserving_gauge_matrix()
    restricted = tuple(gauge[index] for index in support)
    rank = _rank_over_q(restricted)
    minor_rows, minor_columns, determinant = _greedy_minor(
        restricted, 9
    )
    matching_rows = []
    for color, matching_index in enumerate(expected["seed"]):
        matching = _primary_matchings()[matching_index]
        monomial = tuple(
            variable_index(N, D, u, v, color, color)
            for u, v in matching
        )
        if not set(monomial).issubset(support):
            raise KrennN8ToricFirstShellError(
                "a seed matching product left its seed support"
            )
        character = tuple(
            sum(gauge[index][column] for index in monomial)
            for column in range(GAUGE_DIMENSION)
        )
        matching_rows.append(
            {
                "color": color,
                "matching_index": matching_index,
                "source_variable_indices": list(monomial),
                "gauge_character": list(character),
            }
        )
    if (
        rank != 9
        or determinant != expected["slice_determinant"]
        or any(any(record["gauge_character"]) for record in matching_rows)
    ):
        raise KrennN8ToricFirstShellError(
            f"the exact {label} seed gauge slice changed"
        )
    return {
        "seed_support": list(support),
        "full_target_preserving_torus_dimension": GAUGE_DIMENSION,
        "restriction_matrix_shape": [len(support), GAUGE_DIMENSION],
        "restriction_rank_over_Q": rank,
        "three_matching_product_orders_are_invariant": True,
        "normalizable_seed_order_subspace_dimension": len(support) - D,
        "residual_gauge_dimension_after_seed_normalization": (
            GAUGE_DIMENSION - rank
        ),
        "saturated_unimodular_slice": {
            "row_indices": list(minor_rows),
            "column_indices": list(minor_columns),
            "determinant": determinant,
            "root_extraction_required": False,
        },
        "monochromatic_matching_products": matching_rows,
    }


def _matching_incidence_rotor() -> dict[str, object]:
    edges = tuple(canonical_edges(N))
    matchings = _primary_matchings()
    independent = _independent_matchings()
    incidence = tuple(
        tuple(int(edge in matching) for matching in matchings)
        for edge in edges
    )
    rank = _rank_over_q(incidence)
    transpose = tuple(zip(*incidence, strict=True))
    gram = _integer_matrix_product(transpose, incidence)
    gram_squared = _integer_matrix_product(gram, gram)
    scale = 1_080
    projector_scaled = tuple(
        tuple(
            (scale if row == column else 0)
            - 78 * gram[row][column]
            + gram_squared[row][column]
            for column in range(MATCHING_COUNT)
        )
        for row in range(MATCHING_COUNT)
    )
    incidence_times_projector = _integer_matrix_product(
        incidence, projector_scaled
    )
    projector_squared = _integer_matrix_product(
        projector_scaled, projector_scaled
    )
    idempotent = all(
        projector_squared[row][column]
        == scale * projector_scaled[row][column]
        for row in range(MATCHING_COUNT)
        for column in range(MATCHING_COUNT)
    )
    matching_index = {
        matching: index for index, matching in enumerate(matchings)
    }
    generator_checks = []
    for vertex in range(N - 1):
        permutation = list(range(N))
        permutation[vertex], permutation[vertex + 1] = (
            permutation[vertex + 1],
            permutation[vertex],
        )
        action = tuple(
            matching_index[
                _canonical_matching(
                    tuple(
                        (permutation[u], permutation[v])
                        for u, v in matching
                    )
                )
            ]
            for matching in matchings
        )
        commutes = all(
            projector_scaled[action[row]][action[column]]
            == projector_scaled[row][column]
            for row in range(MATCHING_COUNT)
            for column in range(MATCHING_COUNT)
        )
        generator_checks.append(
            {
                "adjacent_vertex_swap": [vertex, vertex + 1],
                "commutes_with_projector": commutes,
            }
        )
    matrix_bytes = json.dumps(
        projector_scaled,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    kernel_rank = MATCHING_COUNT - rank
    checks = {
        "primary_and_independent_matching_sets_agree": (
            set(matchings) == set(independent)
        ),
        "incidence_shape_28_by_105": (
            len(incidence) == 28
            and all(len(row) == MATCHING_COUNT for row in incidence)
        ),
        "incidence_rank_21_over_Q": rank == 21,
        "projector_is_symmetric": all(
            projector_scaled[row][column]
            == projector_scaled[column][row]
            for row in range(MATCHING_COUNT)
            for column in range(MATCHING_COUNT)
        ),
        "incidence_times_projector_is_zero": not any(
            value
            for row in incidence_times_projector
            for value in row
        ),
        "scaled_projector_is_idempotent": idempotent,
        "scaled_projector_trace_is_1080_times_84": (
            sum(
                projector_scaled[index][index]
                for index in range(MATCHING_COUNT)
            )
            == scale * 84
        ),
        "all_adjacent_S8_generators_commute": all(
            record["commutes_with_projector"]
            for record in generator_checks
        ),
    }
    if kernel_rank != 84 or not all(checks.values()):
        raise KrennN8ToricFirstShellError(
            "the exact K8 matching rotor projector changed"
        )
    return {
        "fixed_coloring_matching_space_dimension": MATCHING_COUNT,
        "edge_matching_incidence_shape": [28, MATCHING_COUNT],
        "edge_matching_incidence_rank_over_Q": rank,
        "rotor_kernel_dimension": kernel_rank,
        "projector_formula": (
            "P=I-(13/180)G+(1/1080)G^2, G=E^T E"
        ),
        "integer_scale": scale,
        "scaled_projector_sha256": sha256(matrix_bytes).hexdigest(),
        "adjacent_generator_checks": generator_checks,
        "exact_checks": checks,
        "interpretation": {
            "all_apex_edge_marginals": "q=E*m",
            "rotor_coordinates": "h=P*m",
            "projective_pair": "[q:h]",
            "raw_all_apex_regrouping_is_injective": False,
            "rotor_retains_matching_interference_lost_by_marginals": True,
            "rotor_used_to_exclude_first_shell": False,
            "rotor_is_a_Rees_exceptional_coordinate": False,
            "rotor_or_displayed_circuits_form_a_tropical_basis": False,
            "rotor_proves_H5_H6_or_global_obstruction": False,
        },
    }


def _monomial_for_coloring(
    coloring: Sequence[int],
    matching: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    values = tuple(map(int, coloring))
    if len(values) != N or any(value < 0 or value >= D for value in values):
        raise KrennN8ToricFirstShellError(
            "an n=8 coloring is malformed"
        )
    return tuple(
        variable_index(N, D, u, v, values[u], values[v])
        for u, v in matching
    )


def _repair_options(
    label: str,
) -> tuple[tuple[dict[str, object], ...], ...]:
    row = _hard_case_row(label)
    seed_support = set(map(int, row["seed_source_variable_indices"]))
    result = []
    for victim in row["singleton_mixed_victims"]:
        coloring = tuple(map(int, victim["coloring"]))
        costs: Counter[int] = Counter()
        options = []
        for matching_index, matching in enumerate(_primary_matchings()):
            monomial = _monomial_for_coloring(coloring, matching)
            new_coordinates = tuple(
                sorted(set(monomial).difference(seed_support))
            )
            costs[len(new_coordinates)] += 1
            if len(new_coordinates) == 2:
                options.append(
                    {
                        "matching_index": matching_index,
                        "matching": [list(edge) for edge in matching],
                        "monomial_variable_indices": list(monomial),
                        "new_coordinate_indices": list(new_coordinates),
                    }
                )
        if dict(sorted(costs.items())) != {0: 1, 2: 12, 3: 32, 4: 60}:
            raise KrennN8ToricFirstShellError(
                f"a {label} victim repair-cost histogram changed"
            )
        result.append(tuple(options))
    if (
        len(result) != EXPECTED[label]["victims"]
        or any(len(options) != 12 for options in result)
    ):
        raise KrennN8ToricFirstShellError(
            f"the {label} minimal repair table changed"
        )
    return tuple(result)


def _transport_matching(
    matching: Sequence[tuple[int, int]],
    vertices: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    return _canonical_matching(
        tuple((vertices[u], vertices[v]) for u, v in matching)
    )


@lru_cache(maxsize=None)
def _seed_stabilizer(
    label: str,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    seed = tuple(EXPECTED[label]["seed"])
    matchings = _primary_matchings()
    matching_index = {
        matching: index for index, matching in enumerate(matchings)
    }
    stabilizer = []
    for vertices in permutations(range(N)):
        transported = tuple(
            matching_index[
                _transport_matching(matchings[index], vertices)
            ]
            for index in seed
        )
        for colors in permutations(range(D)):
            image = [None] * D
            for old_color, matching_index_value in enumerate(transported):
                image[colors[old_color]] = matching_index_value
            if tuple(image) == seed:
                stabilizer.append((tuple(vertices), tuple(colors)))
    result = tuple(stabilizer)
    if len(result) != EXPECTED[label]["stabilizer"]:
        raise KrennN8ToricFirstShellError(
            f"the exact {label} seed stabilizer changed"
        )
    return result


def _transport_variable(
    index: int,
    vertices: Sequence[int],
    colors: Sequence[int],
) -> int:
    u, v, a, b = variable_key(N, D, int(index))
    new_u, new_v = vertices[u], vertices[v]
    new_a, new_b = colors[a], colors[b]
    if new_u < new_v:
        return variable_index(N, D, new_u, new_v, new_a, new_b)
    return variable_index(N, D, new_v, new_u, new_b, new_a)


def _transport_coloring(
    coloring: Sequence[int],
    vertices: Sequence[int],
    colors: Sequence[int],
) -> tuple[int, ...]:
    result = [-1] * N
    for old_vertex, old_color in enumerate(coloring):
        result[vertices[old_vertex]] = colors[int(old_color)]
    return tuple(result)


def _decoration_key(
    decoration: Sequence[tuple[Sequence[int], Sequence[int]]],
    vertices: Sequence[int],
    colors: Sequence[int],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (
                _transport_coloring(coloring, vertices, colors),
                tuple(
                    sorted(
                        _transport_variable(index, vertices, colors)
                        for index in monomial
                    )
                ),
            )
            for coloring, monomial in decoration
        )
    )


def _canonical_decoration(
    label: str,
    decoration: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    return min(
        _decoration_key(decoration, vertices, colors)
        for vertices, colors in _seed_stabilizer(label)
    )


def _active_term_signature(
    support: Sequence[int],
    matching_enumerator: Sequence[Sequence[tuple[int, int]]],
) -> dict[int, tuple[tuple[tuple[tuple[int, int], ...], tuple[int, ...]], ...]]:
    support = tuple(sorted(map(int, support)))
    edge_variables: dict[
        tuple[int, int], list[tuple[int, int, int]]
    ] = defaultdict(list)
    for index in support:
        u, v, a, b = variable_key(N, D, index)
        edge_variables[(u, v)].append((index, a, b))
    terms: dict[
        int, list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]]
    ] = defaultdict(list)
    for raw_matching in matching_enumerator:
        matching = _canonical_matching(raw_matching)
        choices = tuple(edge_variables[edge] for edge in matching)
        if any(not choice for choice in choices):
            continue
        for selected in product(*choices):
            coloring = [-1] * N
            monomial = []
            for (u, v), (index, a, b) in zip(
                matching, selected, strict=True
            ):
                coloring[u], coloring[v] = a, b
                monomial.append(index)
            equation = coloring_index(N, D, coloring)
            terms[equation].append(
                (matching, tuple(sorted(monomial)))
            )
    return {
        equation: tuple(sorted(records))
        for equation, records in terms.items()
    }


def _support_initial_form_record(
    support: Sequence[int],
) -> dict[str, object]:
    support = tuple(sorted(map(int, support)))
    if len(set(support)) != len(support):
        raise KrennN8ToricFirstShellError(
            "a first-shell support repeats a coordinate"
        )
    primary = _active_term_signature(support, _primary_matchings())
    independent = _active_term_signature(
        support, _independent_matchings()
    )
    if primary != independent:
        raise KrennN8ToricFirstShellError(
            "the two matching enumerators disagree on an initial support"
        )
    targets = {
        coloring_index(N, D, (color,) * N) for color in range(D)
    }
    singletons = []
    for equation, records in sorted(primary.items()):
        if equation in targets or len(records) != 1:
            continue
        matching, monomial = records[0]
        singletons.append(
            {
                "equation": equation,
                "coloring": list(coloring_from_index(N, D, equation)),
                "matching": [list(edge) for edge in matching],
                "monomial_variable_indices": list(monomial),
            }
        )
    active_histogram = Counter(map(len, primary.values()))
    active_histogram[0] = COLORINGS - len(primary)
    return {
        "support": list(support),
        "support_size": len(support),
        "active_term_count_histogram_all_6561_equations": {
            str(key): value
            for key, value in sorted(active_histogram.items())
        },
        "zero_target_singleton_initial_forms": singletons,
        "zero_target_singleton_count": len(singletons),
        "all_6561_equations_accounted": (
            sum(active_histogram.values()) == COLORINGS
        ),
        "primary_independent_matching_enumerators_agree": True,
        "initial_stratum_can_map_to_projective_GHZ": not singletons,
    }


def _hard_case_first_shell(label: str) -> dict[str, object]:
    expected = EXPECTED[label]
    row = _hard_case_row(label)
    victims = tuple(row["singleton_mixed_victims"])
    repair_options = _repair_options(label)
    stabilizer = _seed_stabilizer(label)
    seed_support = set(map(int, row["seed_source_variable_indices"]))

    individual_groups: dict[object, int] = Counter()
    for victim, options in zip(victims, repair_options, strict=True):
        coloring = tuple(map(int, victim["coloring"]))
        for option in options:
            decoration = (
                (
                    coloring,
                    tuple(option["monomial_variable_indices"]),
                ),
            )
            individual_groups[
                _canonical_decoration(label, decoration)
            ] += 1
    if len(individual_groups) != 7:
        raise KrennN8ToricFirstShellError(
            f"the {label} individual repair orbit count changed"
        )

    grouped: dict[
        tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        list[tuple[dict[str, object], ...]],
    ] = defaultdict(list)
    raw_supports: set[tuple[int, ...]] = set()
    for options in product(*repair_options):
        decoration = tuple(
            (
                tuple(map(int, victim["coloring"])),
                tuple(map(int, option["monomial_variable_indices"])),
            )
            for victim, option in zip(victims, options, strict=True)
        )
        raw_supports.add(
            tuple(
                sorted(
                    seed_support.union(
                        *(
                            set(
                                map(
                                    int,
                                    option[
                                        "monomial_variable_indices"
                                    ],
                                )
                            )
                            for option in options
                        )
                    )
                )
            )
        )
        grouped[_canonical_decoration(label, decoration)].append(options)
    if (
        len(grouped) != expected["orbits"]
        or len(raw_supports) != expected["raw"]
    ):
        raise KrennN8ToricFirstShellError(
            f"the {label} simultaneous support/orbit count changed"
        )

    orbit_records = []
    raw_singleton_histogram: Counter[int] = Counter()
    orbit_size_histogram: Counter[int] = Counter()
    support_size_histogram: Counter[int] = Counter()
    for orbit_index, members in enumerate(
        sorted(
            grouped.values(),
            key=lambda group: min(
                tuple(option["matching_index"] for option in member)
                for member in group
            ),
        )
    ):
        representative = min(
            members,
            key=lambda member: tuple(
                option["matching_index"] for option in member
            ),
        )
        support = tuple(
            sorted(
                seed_support.union(
                    *(
                        set(map(int, option["monomial_variable_indices"]))
                        for option in representative
                    )
                )
            )
        )
        initial = _support_initial_form_record(support)
        orbit_size = len(members)
        singleton_count = initial["zero_target_singleton_count"]
        raw_singleton_histogram[singleton_count] += orbit_size
        orbit_size_histogram[orbit_size] += 1
        support_size_histogram[len(support)] += orbit_size
        orbit_records.append(
            {
                "orbit_index": orbit_index,
                "raw_decorated_orbit_size": orbit_size,
                "representative_repair_matching_indices": [
                    option["matching_index"] for option in representative
                ],
                "representative_repair_new_coordinate_indices": [
                    option["new_coordinate_indices"]
                    for option in representative
                ],
                "flat_valuation": {
                    "order_zero_coordinate_indices": list(support),
                    "all_other_source_coordinate_orders": "strictly-positive",
                    "common_weight_projectivization_only": False,
                },
                "initial_form_replay": initial,
                "exact_nonzero_support_stratum_excluded": (
                    singleton_count > 0
                ),
                "stratum_excluded_by_singleton_initial_monomial": (
                    singleton_count > 0
                ),
            }
        )
    best_orbits = sum(
        record["initial_form_replay"]["zero_target_singleton_count"]
        == expected["minimum_singletons"]
        for record in orbit_records
    )
    checks = {
        "raw_assignments_match_12_power_victims": (
            sum(len(values) for values in grouped.values())
            == expected["raw"]
            == 12 ** expected["victims"]
        ),
        "every_raw_decoration_has_a_distinct_exact_support": (
            len(raw_supports) == expected["raw"]
        ),
        "decorated_orbit_count": len(orbit_records)
        == expected["orbits"],
        "orbit_sizes_sum_to_raw_assignments": (
            sum(
                record["raw_decorated_orbit_size"]
                for record in orbit_records
            )
            == expected["raw"]
        ),
        "orbit_sizes_divide_seed_stabilizer": all(
            len(stabilizer) % record["raw_decorated_orbit_size"] == 0
            for record in orbit_records
        ),
        "all_supports_have_expected_size": (
            dict(support_size_histogram)
            == {expected["support_size"]: expected["raw"]}
        ),
        "orbit_size_histogram_exact": (
            dict(sorted(orbit_size_histogram.items()))
            == expected["orbit_size_histogram"]
        ),
        "singleton_histogram_exact": (
            dict(sorted(raw_singleton_histogram.items()))
            == expected["singleton_histogram"]
        ),
        "minimum_singletons_exact": (
            min(raw_singleton_histogram)
            == expected["minimum_singletons"]
        ),
        "best_orbit_count_exact": best_orbits
        == expected["best_orbits"],
        "every_flat_first_shell_stratum_excluded": all(
            record["stratum_excluded_by_singleton_initial_monomial"]
            for record in orbit_records
        ),
        "every_exact_nonzero_support_stratum_excluded": all(
            record["exact_nonzero_support_stratum_excluded"]
            for record in orbit_records
        ),
        "both_matching_enumerators_agree_on_every_orbit": all(
            record["initial_form_replay"][
                "primary_independent_matching_enumerators_agree"
            ]
            for record in orbit_records
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricFirstShellError(
            f"the {label} first-shell checks failed: {failed}"
        )
    return {
        "label": label,
        "representative_matching_indices": list(expected["seed"]),
        "seed_stabilizer_order": len(stabilizer),
        "original_singleton_victim_count": len(victims),
        "original_singleton_victims": list(victims),
        "repair_cost_histogram_per_original_victim": {
            "0": 1,
            "2": 12,
            "3": 32,
            "4": 60,
        },
        "individual_decorated_minimal_repairs": (
            len(victims) * 12
        ),
        "individual_repair_stabilizer_orbits": len(individual_groups),
        "simultaneous_raw_decorated_assignments": expected["raw"],
        "simultaneous_distinct_exact_supports": len(raw_supports),
        "simultaneous_decorated_stabilizer_orbits": len(orbit_records),
        "raw_support_size_histogram": {
            str(key): value
            for key, value in sorted(support_size_histogram.items())
        },
        "raw_singleton_spill_histogram": {
            str(key): value
            for key, value in sorted(raw_singleton_histogram.items())
        },
        "minimum_singleton_spills": min(raw_singleton_histogram),
        "orbits_attaining_minimum_singleton_spills": best_orbits,
        "decorated_orbit_size_histogram": {
            str(key): value
            for key, value in sorted(orbit_size_histogram.items())
        },
        "orbit_records": orbit_records,
        "exact_checks": checks,
        "consequence": (
            "every enumerated exact nonzero support and flat cost-two "
            "candidate initial stratum has a zero-target singleton monomial; "
            "the exact support has no torus solution and the flat "
            "candidate stratum cannot map to the projective GHZ "
            "output direction"
        ),
    }


@lru_cache(maxsize=1)
def _build_certificate() -> dict[str, object]:
    gauge = {
        label: _seed_gauge_record(label) for label in ("H5", "H6")
    }
    rotor = _matching_incidence_rotor()
    hard_cases = {
        label: _hard_case_first_shell(label)
        for label in ("H5", "H6")
    }
    total_orbits = sum(
        hard_cases[label]["simultaneous_decorated_stabilizer_orbits"]
        for label in hard_cases
    )
    total_raw = sum(
        hard_cases[label]["simultaneous_raw_decorated_assignments"]
        for label in hard_cases
    )
    if total_orbits != 347 or total_raw != 1_872:
        raise KrennN8ToricFirstShellError(
            "the combined first-shell census changed"
        )
    return {
        "schema": FIRST_SHELL_SCHEMA,
        "scope": {
            "parameters": {"n": N, "d": D},
            "source_variables": SOURCE_VARIABLES,
            "output_colorings": COLORINGS,
            "perfect_matchings_per_coloring": MATCHING_COUNT,
            "seed_orbits": ["H5", "H6"],
            "all_31_seed_orbits_covered": False,
            "floating_point_used": False,
            "random_seeds": [],
            "cpu_workers": 1,
        },
        "compactification_protocol": {
            "correct_border_object": (
                "closure of graph [w]->[T(w)] retaining projective "
                "output coordinates; equivalently Rees/blow-up data"
            ),
            "naive_weight_projectivization_is_sufficient": False,
            "equation_T_of_w_equals_h4_GHZ_at_h0_is_sufficient": False,
            "reason": (
                "h=0 alone records a base-locus point T(w)=0 but loses "
                "the limiting projective output direction"
            ),
            "interior_graph_point_means_affine_witness": True,
            "exceptional_graph_point_is_not_an_affine_witness": True,
            "exceptional_graph_point_proves_strict_border_membership": False,
            "full_Rees_algebra_constructed_here": False,
            "full_graph_or_Rees_fiber_computed": False,
            "gauge_quotient_constructed_here": False,
            "coarse_GIT_quotient_alone_decides_orbit_closure": False,
            "one_parameter_subgroup_orbit_incidence_must_be_retained": True,
        },
        "gauge_quotient": {
            "action": (
                "w_uv^ab -> lambda_(u,a) lambda_(v,b) w_uv^ab"
            ),
            "target_preserving_constraints": (
                "product_v lambda_(v,a)=1 for each color a"
            ),
            "cocharacter_dimension": GAUGE_DIMENSION,
            "common_Cstar_projectivization_dimension": 1,
            "Hopf_or_common_projectivization_removes_full_gauge": False,
            "seed_normalizations": gauge,
        },
        "fixed_coloring_matching_rotor": rotor,
        "flat_first_shell": {
            "definition": (
                "order zero on the twelve seed coordinates and the "
                "coordinates of one cost-two repair per original victim; "
                "strictly positive order on every other source coordinate"
            ),
            "total_raw_decorated_assignments": total_raw,
            "total_decorated_stabilizer_orbits": total_orbits,
            "hard_cases": hard_cases,
            "all_enumerated_exact_nonzero_supports_excluded": True,
            "all_enumerated_strata_excluded": True,
            "exclusion_mechanism": (
                "a zero-target generator has a one-monomial initial form"
            ),
        },
        "conclusion": {
            "classification": "exact-negative-flat-first-shell",
            "projective_GHZ_first_shell_graph_fiber_candidate_hit": False,
            "next_exact_object": (
                "different minimal layers, negative quotient directions, "
                "and deeper decorated valuation cones in a future "
                "graph/Rees compactification"
            ),
            "next_acceptance_test": (
                "saturated full initial ideal is proper and monomial-free; "
                "generator min-attainment alone is only a prevariety test"
            ),
        },
        "claim_boundary": {
            "enumerated_flat_cost2_H5_H6_strata_excluded": True,
            "enumerated_cost2_H5_H6_exact_supports_excluded": True,
            "all_H5_H6_valuation_cones_classified": False,
            "all_31_n8_seed_orbits_classified": False,
            "unequal_rate_cones_excluded": False,
            "deeper_repair_shells_excluded": False,
            "higher_cost_first_repairs_excluded": False,
            "full_saturated_initial_ideal_computed": False,
            "selected_circuits_claimed_to_be_tropical_basis": False,
            "raw_all_apex_regrouping_claimed_to_add_constraints": False,
            "naive_projective_or_Hopf_compactification_decides_image": False,
            "full_graph_or_Rees_fiber_computed": False,
            "gauge_quotient_constructed": False,
            "rotor_projector_used_to_exclude_first_shell": False,
            "rotor_projector_is_a_Rees_exceptional_coordinate": False,
            "rotor_projector_claimed_to_be_a_tropical_basis": False,
            "rotor_projector_proves_H5_H6_or_global_obstruction": False,
            "n8_projective_GHZ_border_membership_proved": False,
            "n8_strict_border_membership_proved": False,
            "n8_affine_GHZ_membership_decided": False,
            "n8_nonexistence_proved": False,
            "finite_counterexample_found": False,
        },
    }


def build_n8_toric_first_shell_certificate() -> dict[str, object]:
    """Return an isolated exact first-shell certificate."""

    return deepcopy(_build_certificate())


def verify_n8_toric_first_shell_certificate(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Recompute the complete finite certificate and reject any change."""

    if not isinstance(payload, Mapping):
        raise KrennN8ToricFirstShellError(
            "the first-shell certificate must be a mapping"
        )
    expected = build_n8_toric_first_shell_certificate()
    if dict(payload) != expected:
        raise KrennN8ToricFirstShellError(
            "the first-shell certificate differs from exact replay"
        )
    return expected


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _strict_json(path: Path) -> object:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise KrennN8ToricFirstShellError(
                    f"duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str):
        raise KrennN8ToricFirstShellError(
            f"nonfinite JSON constant {value!r}"
        )

    try:
        return json.loads(
            path.read_bytes().decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennN8ToricFirstShellError(
            f"could not decode {path.name}"
        ) from error


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _readme_text(certificate: Mapping[str, object]) -> str:
    flat = certificate["flat_first_shell"]
    h5 = flat["hard_cases"]["H5"]
    h6 = flat["hard_cases"]["H6"]
    return f"""# Exact `n=8,d=3` toric first-shell gate

This bundle is an exact first-shell preflight for a future gauge-quotiented
graph compactification of the matching map.  That compactification and
quotient are not constructed here, and this is not an `n=8` existence or
nonexistence proof.

## Geometry

A common Hopf/projective normalization removes only one complex scaling.  The
GHZ-preserving color-diagonal torus has dimension 21.  On either twelve-weight
seed its restriction has saturated rank 9, so the seed orders can be
normalized to zero without root extraction.

The correct border object retains the limiting output direction: the closure
of `[w] -> [T(w)]`, or corresponding Rees/blow-up data.  Merely setting
`T(w)=h^4 GHZ` and then `h=0` loses that direction.

For one fixed coloring, the K8 edge--matching incidence matrix has rank 21 in
the 105-dimensional matching-amplitude space.  Its exact equivariant rotor
projector has rank 84 and retains the sector invisible to all edge/apex
marginals.

## Exact first-shell result

| seed | raw decorated strata | stabilizer orbits | minimum singleton spills |
|---|---:|---:|---:|
| H5 | {h5["simultaneous_raw_decorated_assignments"]} | {h5["simultaneous_decorated_stabilizer_orbits"]} | {h5["minimum_singleton_spills"]} |
| H6 | {h6["simultaneous_raw_decorated_assignments"]} | {h6["simultaneous_decorated_stabilizer_orbits"]} | {h6["minimum_singleton_spills"]} |

All {flat["total_decorated_stabilizer_orbits"]} minimal flat strata have a
zero-target singleton initial monomial, so none can land over the projective
GHZ direction.  The same replay excludes all
{flat["total_raw_decorated_assignments"]} corresponding exact nonzero
supports, because their singleton monomial cannot vanish in the support
torus.

Different minimal layers, negative quotient directions, deeper repairs,
higher-cost first repairs, the other 29 seed orbits, and the full saturated
initial ideal remain open.

## Reproduce

```powershell
C:\\tmp\\Krenn-obstruction-venv\\Scripts\\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_first_shell --results-directory results\\krenn_quantum_graph\\n8_d3_toric_first_shell
```

```powershell
C:\\tmp\\Krenn-obstruction-venv\\Scripts\\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_first_shell --results-directory results\\krenn_quantum_graph\\n8_d3_toric_first_shell --verify-only
```
"""


def _expected_manifest(
    directory: Path,
    certificate: Mapping[str, object],
) -> dict[str, object]:
    root = _repo_root()
    return {
        "schema": FIRST_SHELL_MANIFEST_SCHEMA,
        "artifacts": [
            _file_record(directory / CERTIFICATE_FILE, relative_to=directory),
            _file_record(directory / README_FILE, relative_to=directory),
        ],
        "inputs": [
            _file_record(root / relative, relative_to=root)
            for relative in SOURCE_PATHS
        ],
        "claim_boundary": certificate["claim_boundary"],
        "scratch_data_required_for_verification": False,
    }


def write_n8_toric_first_shell_bundle(
    directory: Path | str = DEFAULT_RESULTS_DIRECTORY,
) -> dict[str, object]:
    """Write the compact certificate, explanation, and source manifest."""

    directory = Path(directory)
    if not directory.is_absolute():
        directory = _repo_root() / directory
    directory.mkdir(parents=True, exist_ok=True)
    expected_names = {CERTIFICATE_FILE, README_FILE, MANIFEST_FILE}
    unexpected = {
        path.name for path in directory.iterdir()
        if path.name not in expected_names
    }
    if unexpected:
        raise KrennN8ToricFirstShellError(
            f"unexpected first-shell artifact files: {sorted(unexpected)}"
        )
    certificate = build_n8_toric_first_shell_certificate()
    _write_bytes_atomic(
        directory / CERTIFICATE_FILE,
        _canonical_json_bytes(certificate),
    )
    _write_bytes_atomic(
        directory / README_FILE,
        _readme_text(certificate).encode("ascii"),
    )
    manifest = _expected_manifest(directory, certificate)
    _write_bytes_atomic(
        directory / MANIFEST_FILE,
        _canonical_json_bytes(manifest),
    )
    return certificate


def verify_n8_toric_first_shell_bundle(
    directory: Path | str = DEFAULT_RESULTS_DIRECTORY,
) -> dict[str, object]:
    """Verify inventory, hashes, sources, and every exact semantic datum."""

    directory = Path(directory)
    if not directory.is_absolute():
        directory = _repo_root() / directory
    if not directory.is_dir() or directory.is_symlink():
        raise KrennN8ToricFirstShellError(
            "the first-shell bundle directory is absent or linked"
        )
    expected_names = {CERTIFICATE_FILE, README_FILE, MANIFEST_FILE}
    names = {path.name for path in directory.iterdir()}
    if names != expected_names or any(
        path.is_symlink() or not path.is_file()
        for path in directory.iterdir()
    ):
        raise KrennN8ToricFirstShellError(
            "the first-shell bundle inventory changed"
        )
    certificate = _strict_json(directory / CERTIFICATE_FILE)
    if not isinstance(certificate, Mapping):
        raise KrennN8ToricFirstShellError(
            "the first-shell certificate is not a mapping"
        )
    verified = verify_n8_toric_first_shell_certificate(certificate)
    if (directory / README_FILE).read_bytes() != _readme_text(
        verified
    ).encode("ascii"):
        raise KrennN8ToricFirstShellError(
            "the first-shell README changed"
        )
    manifest = _strict_json(directory / MANIFEST_FILE)
    if manifest != _expected_manifest(directory, verified):
        raise KrennN8ToricFirstShellError(
            "the first-shell manifest or source ledger changed"
        )
    return verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact n=8,d=3 H5/H6 toric first-shell gate"
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        certificate = verify_n8_toric_first_shell_bundle(
            args.results_directory
        )
    else:
        certificate = write_n8_toric_first_shell_bundle(
            args.results_directory
        )
        verify_n8_toric_first_shell_bundle(args.results_directory)
    print(
        json.dumps(
            {
                "classification": certificate["conclusion"][
                    "classification"
                ],
                "decorated_orbits": certificate["flat_first_shell"][
                    "total_decorated_stabilizer_orbits"
                ],
                "n8_nonexistence_proved": certificate["claim_boundary"][
                    "n8_nonexistence_proved"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = (
    "DEFAULT_RESULTS_DIRECTORY",
    "KrennN8ToricFirstShellError",
    "build_n8_toric_first_shell_certificate",
    "target_preserving_gauge_matrix",
    "verify_n8_toric_first_shell_bundle",
    "verify_n8_toric_first_shell_certificate",
    "write_n8_toric_first_shell_bundle",
)


if __name__ == "__main__":
    raise SystemExit(main())
