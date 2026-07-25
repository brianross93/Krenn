r"""Exact mixed associated-graded and rotor gate for ``n=6,d=3``.

The known Laurent family is a one-parameter orbit of the full
color-diagonal GHZ stabilizer.  This module checks the mixed-color
cofactors, the complete differential of the matching map, and every higher
derivative order of the cubic map.  All of their apparent Laurent growth is
exactly the corresponding torus character.

The faithful chart-free object for one fixed coloring is the vector of its
15 perfect-matching amplitudes.  Simultaneous apex marginals retain only its
image under the 15 by 15 edge--matching incidence matrix.  That matrix has
rank 10 and a saturated five-dimensional kernel generated integrally by
primitive K3,3 parity circuits.  The missing summand is called the rotor
sector below.

This is a stopping and routing certificate.  It proves neither affine GHZ
membership nor nonmembership.  It says that more raw star, Jacobian,
Hessian, or cubic holonomy on the *known* Laurent orbit cannot distinguish
its boundary behavior, and identifies the projective matching-amplitude
data that a transverse search must retain.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TypeAlias

from experiments.krenn_quantum_graph.border_image import (
    LAURENT_ONE,
    LaurentPolynomial,
    natural_laurent_weight_entries,
)
from experiments.krenn_quantum_graph.fixtures import fixture_n4_d3
from experiments.krenn_quantum_graph.independent_verifier import (
    independent_perfect_matchings,
    verify_system_arrays,
)
from experiments.krenn_quantum_graph.multi_apex_holonomy import (
    laurent_matching_cofactor,
    laurent_specialize,
    laurent_star_coefficient_matrix,
    laurent_valuation,
    natural_stabilizer_exponents,
)
from experiments.krenn_quantum_graph.n6_deformation import (
    vertex_scalar_gauge_matrix,
)
from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    coloring_from_index,
    coloring_index,
    generate_sparse_system,
    perfect_matchings,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    n6_d3_seed_witness,
)


MIXED_GATE_SCHEMA = "krenn-n6-d3-mixed-associated-graded-gate-v1"
MIXED_MANIFEST_SCHEMA = (
    "krenn-n6-d3-mixed-associated-graded-manifest-v1"
)
DEFAULT_RESULTS_DIRECTORY = (
    Path("results")
    / "krenn_quantum_graph"
    / "n6_d3_mixed_associated_graded_gate"
)
SOURCE_PATHS = (
    "experiments/krenn_quantum_graph/border_image.py",
    "experiments/krenn_quantum_graph/fixtures.py",
    "experiments/krenn_quantum_graph/independent_verifier.py",
    "experiments/krenn_quantum_graph/mixed_associated_graded.py",
    "experiments/krenn_quantum_graph/multi_apex_holonomy.py",
    "experiments/krenn_quantum_graph/n6_deformation.py",
    "experiments/krenn_quantum_graph/source_ideal.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "experiments/krenn_quantum_graph/ternary_search.py",
    "experiments/krenn_quantum_graph/witness.py",
    "tests/test_krenn_mixed_associated_graded.py",
)
VICTIM_COLORING = (0, 0, 2, 1, 2, 1)
VICTIM_MATCHING_INDEX = 1
MINIMAL_REPAIR_MATCHING_INDICES = (0, 2, 4, 8, 9, 13)
ROTOR_PARTITIONS = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 4),
)
ROTOR_FREE_MATCHING_INDICES = (8, 10, 11, 13, 14)
INCIDENCE_MINOR_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
)
INCIDENCE_MINOR_MATCHINGS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 12)

MatchingEnumerator: TypeAlias = Callable[
    [int], Sequence[Sequence[tuple[int, int]]]
]
LaurentTable: TypeAlias = dict[
    tuple[int, tuple[int, ...]], LaurentPolynomial
]


class KrennMixedAssociatedGradedError(ValueError):
    """An exact mixed-layer replay or bundle failed."""


def _fraction_text(value: Fraction | int) -> str:
    value = Fraction(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _canonical_matching(
    matching: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((min(int(i), int(j)), max(int(i), int(j))) for i, j in matching)
    )


def _rank_over_q(matrix: Sequence[Sequence[int | Fraction]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennMixedAssociatedGradedError(
            "exact rank input is ragged"
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
        scale = rows[rank][column]
        rows[rank] = [entry / scale for entry in rows[rank]]
        for row in range(len(rows)):
            if row == rank:
                continue
            factor = rows[row][column]
            if factor:
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


def _determinant(
    matrix: Sequence[Sequence[int | Fraction]],
) -> Fraction:
    rows = [list(map(Fraction, row)) for row in matrix]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise KrennMixedAssociatedGradedError(
            "exact determinant input is not square"
        )
    determinant = Fraction(1)
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        value = rows[column][column]
        determinant *= value
        for row in range(column + 1, size):
            factor = rows[row][column] / value
            if factor:
                for target in range(column, size):
                    rows[row][target] -= factor * rows[column][target]
    return determinant


def _transpose(
    matrix: Sequence[Sequence[int | Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    if not matrix:
        return ()
    return tuple(
        tuple(Fraction(matrix[row][column]) for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


def _matrix_product(
    left: Sequence[Sequence[int | Fraction]],
    right: Sequence[Sequence[int | Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    if not left or not right or len(left[0]) != len(right):
        raise KrennMixedAssociatedGradedError(
            "exact matrix-product dimensions differ"
        )
    return tuple(
        tuple(
            sum(
                (
                    Fraction(left[row][middle])
                    * Fraction(right[middle][column])
                    for middle in range(len(right))
                ),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _shift_laurent(
    value: LaurentPolynomial, exponent: int
) -> LaurentPolynomial:
    return LaurentPolynomial(
        tuple(
            (power + int(exponent), coefficient)
            for power, coefficient in value.terms
        )
    )


def _constant_weights_at_one(
    entries: Mapping[int, LaurentPolynomial]
    | Sequence[tuple[int, LaurentPolynomial]],
) -> dict[int, LaurentPolynomial]:
    items = entries.items() if isinstance(entries, Mapping) else entries
    return {
        int(index): LaurentPolynomial.constant(
            laurent_specialize(value, 1)
        )
        for index, value in items
    }


def _weight_character(index: int, exponents: Sequence[int]) -> int:
    i, j, a, b = variable_key(6, 3, index)
    return int(exponents[3 * i + a] + exponents[3 * j + b])


def _coloring_character(
    coloring: Sequence[int], exponents: Sequence[int]
) -> int:
    return sum(
        int(exponents[3 * vertex + int(color)])
        for vertex, color in enumerate(coloring)
    )


def _matching_variables(
    coloring: Sequence[int],
    matching: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    return tuple(
        variable_index(
            6,
            3,
            i,
            j,
            int(coloring[i]),
            int(coloring[j]),
        )
        for i, j in matching
    )


def _derivative_table(
    order: int,
    weights: Mapping[int, LaurentPolynomial],
    matching_enumerator: MatchingEnumerator,
) -> LaurentTable:
    if order not in range(4):
        raise KrennMixedAssociatedGradedError(
            "the cubic map has derivative orders zero through three"
        )
    result: LaurentTable = {}
    matchings = matching_enumerator(6)
    for equation in range(3**6):
        coloring = coloring_from_index(6, 3, equation)
        for matching in matchings:
            variables = _matching_variables(coloring, matching)
            for selected_positions in combinations(range(3), order):
                selected = set(selected_positions)
                coefficient = LAURENT_ONE
                for position, variable in enumerate(variables):
                    if position not in selected:
                        coefficient *= weights.get(
                            variable, LaurentPolynomial()
                        )
                if coefficient.is_zero:
                    continue
                differentiated = tuple(
                    sorted(variables[position] for position in selected)
                )
                key = (equation, differentiated)
                result[key] = result.get(
                    key, LaurentPolynomial()
                ) + coefficient
    return {
        key: value for key, value in result.items() if not value.is_zero
    }


def _all_order_covariance() -> dict:
    path = dict(natural_laurent_weight_entries())
    base = _constant_weights_at_one(path)
    exponents = natural_stabilizer_exponents()
    records = {}
    expected_counts = (4, 162, 2_187, 10_935)
    expected_histograms = (
        {0: 3, 1: 1},
        {-1: 27, 0: 108, 1: 27},
        {-1: 243, 0: 1_701, 1: 243},
        {0: 10_935},
    )
    for order in range(4):
        primary = _derivative_table(order, path, perfect_matchings)
        independent = _derivative_table(
            order, path, independent_perfect_matchings
        )
        base_primary = _derivative_table(
            order, base, perfect_matchings
        )
        base_independent = _derivative_table(
            order, base, independent_perfect_matchings
        )
        if (
            primary != independent
            or base_primary != base_independent
            or set(primary) != set(base_primary)
        ):
            raise KrennMixedAssociatedGradedError(
                f"derivative order {order} independent replay disagreed"
            )
        for (equation, variables), value in primary.items():
            rho = _coloring_character(
                coloring_from_index(6, 3, equation), exponents
            )
            kappa = sum(
                _weight_character(variable, exponents)
                for variable in variables
            )
            if value != _shift_laurent(
                base_primary[(equation, variables)], rho - kappa
            ):
                raise KrennMixedAssociatedGradedError(
                    f"derivative order {order} failed torus covariance"
                )
        histogram = dict(
            sorted(
                Counter(
                    laurent_valuation(value)
                    for value in primary.values()
                ).items()
            )
        )
        if (
            len(primary) != expected_counts[order]
            or histogram != expected_histograms[order]
        ):
            raise KrennMixedAssociatedGradedError(
                f"derivative order {order} census changed"
            )
        records[str(order)] = {
            "unique_nonzero_unordered_coordinate_coefficients": len(
                primary
            ),
            "valuation_histogram": {
                str(key): value for key, value in histogram.items()
            },
            "primary_independent_exact_agreement": True,
            "every_coefficient_equals_base_times_character_monomial": True,
            "character_formula": (
                "rho(output coloring)-sum kappa(input coordinates)"
            ),
        }
    return records


def _cofactor_rows(
    n: int,
    weights: Mapping[int, LaurentPolynomial],
    matching_enumerator: MatchingEnumerator,
) -> tuple[tuple[tuple[int, int], tuple[int, ...], LaurentPolynomial], ...]:
    records = []
    for first, second in canonical_edges(n):
        remaining = tuple(
            vertex
            for vertex in range(n)
            if vertex not in (first, second)
        )
        for residual in product(range(3), repeat=n - 2):
            coloring = [0] * n
            for vertex, color in zip(remaining, residual, strict=True):
                coloring[vertex] = color
            value = laurent_matching_cofactor(
                n,
                3,
                weights,
                coloring,
                first,
                second,
                matching_enumerator=matching_enumerator,
            )
            if not value.is_zero:
                records.append(((first, second), residual, value))
    return tuple(records)


def _cofactor_and_star_gate() -> dict:
    n6_path = dict(natural_laurent_weight_entries())
    n6_primary = _cofactor_rows(6, n6_path, perfect_matchings)
    n6_independent = _cofactor_rows(
        6, n6_path, independent_perfect_matchings
    )
    if n6_primary != n6_independent:
        raise KrennMixedAssociatedGradedError(
            "independent n=6 cofactor replay disagreed"
        )
    k4_weights = {
        index: LaurentPolynomial.constant(value)
        for index, value in fixture_n4_d3().entries
    }
    k4_primary = _cofactor_rows(4, k4_weights, perfect_matchings)
    k4_independent = _cofactor_rows(
        4, k4_weights, independent_perfect_matchings
    )
    if k4_primary != k4_independent:
        raise KrennMixedAssociatedGradedError(
            "independent K4 cofactor replay disagreed"
        )

    mixed = tuple(
        record for record in n6_primary if len(set(record[1])) > 1
    )
    monochromatic = tuple(
        record for record in n6_primary if len(set(record[1])) == 1
    )
    if (
        len(n6_primary) != 18
        or len(mixed) != 9
        or len(monochromatic) != 9
        or len(k4_primary) != 6
        or any(len(set(record[1])) > 1 for record in k4_primary)
    ):
        raise KrennMixedAssociatedGradedError(
            "mixed/monochromatic cofactor census changed"
        )

    base = _constant_weights_at_one(n6_path)
    exponents = natural_stabilizer_exponents()
    star_records = []
    total_entries = 0
    total_nonzero = 0
    for root in range(6):
        path_matrix = laurent_star_coefficient_matrix(
            6, 3, n6_path, root
        )
        independent_matrix = laurent_star_coefficient_matrix(
            6,
            3,
            n6_path,
            root,
            matching_enumerator=independent_perfect_matchings,
        )
        base_matrix = laurent_star_coefficient_matrix(
            6, 3, base, root
        )
        if path_matrix != independent_matrix:
            raise KrennMixedAssociatedGradedError(
                "independent star-matrix replay disagreed"
            )
        others = tuple(vertex for vertex in range(6) if vertex != root)
        columns = tuple(
            (neighbor, color)
            for neighbor in others
            for color in range(3)
        )
        nonzero = 0
        for row_index, residual in enumerate(
            product(range(3), repeat=5)
        ):
            row_character = sum(
                exponents[3 * vertex + color]
                for vertex, color in zip(
                    others, residual, strict=True
                )
            )
            for column_index, (neighbor, color) in enumerate(columns):
                actual = path_matrix[row_index][column_index]
                expected = _shift_laurent(
                    base_matrix[row_index][column_index],
                    row_character - exponents[3 * neighbor + color],
                )
                if actual != expected:
                    raise KrennMixedAssociatedGradedError(
                        "star matrix failed exact diagonal factorization"
                    )
                nonzero += int(not actual.is_zero)
        entries = len(path_matrix) * len(path_matrix[0])
        total_entries += entries
        total_nonzero += nonzero
        star_records.append(
            {
                "root": root,
                "entries_checked": entries,
                "nonzero_entries": nonzero,
            }
        )
    if total_entries != 21_870 or total_nonzero != 108:
        raise KrennMixedAssociatedGradedError(
            "star covariance census changed"
        )

    def serialize(
        record: tuple[
            tuple[int, int], tuple[int, ...], LaurentPolynomial
        ]
    ) -> dict:
        edge, residual, value = record
        return {
            "deleted_pair": list(edge),
            "remaining_vertices": [
                vertex for vertex in range(6) if vertex not in edge
            ],
            "residual_colors": list(residual),
            "value": value.to_expression(),
            "valuation": laurent_valuation(value),
            "mixed": len(set(residual)) > 1,
        }

    valuation_histogram = Counter(
        laurent_valuation(record[2]) for record in n6_primary
    )
    mixed_histogram = Counter(
        laurent_valuation(record[2]) for record in mixed
    )
    return {
        "n6_laurent": {
            "coordinates_checked_per_enumerator": 15 * 3**4,
            "nonzero_count": 18,
            "monochromatic_count": 9,
            "mixed_count": 9,
            "valuation_histogram": {
                str(key): value
                for key, value in sorted(valuation_histogram.items())
            },
            "mixed_valuation_histogram": {
                str(key): value
                for key, value in sorted(mixed_histogram.items())
            },
            "nonzero_records": [serialize(record) for record in n6_primary],
            "primary_independent_exact_agreement": True,
        },
        "k4_exact_witness": {
            "coordinates_checked_per_enumerator": 6 * 3**2,
            "nonzero_count": 6,
            "monochromatic_count": 6,
            "mixed_count": 0,
            "primary_independent_exact_agreement": True,
        },
        "star_diagonal_factorization": {
            "formula": (
                "A_r(t)=diag(t^R(row))*A_r(1)"
                "*diag(t^(-x_(v,b)))"
            ),
            "roots": star_records,
            "entries_checked": total_entries,
            "nonzero_entries": total_nonzero,
            "primary_independent_exact_agreement": True,
        },
    }


def _jacobian_forest(covariance: Mapping) -> dict:
    del covariance
    path = dict(natural_laurent_weight_entries())
    base = _constant_weights_at_one(path)
    table = _derivative_table(1, base, perfect_matchings)
    row_neighbors: dict[int, set[int]] = defaultdict(set)
    column_neighbors: dict[int, set[int]] = defaultdict(set)
    for (equation, variables), value in table.items():
        if len(variables) != 1 or value != LAURENT_ONE:
            raise KrennMixedAssociatedGradedError(
                "natural Jacobian stopped being a unit support matrix"
            )
        variable = variables[0]
        row_neighbors[equation].add(variable)
        column_neighbors[variable].add(equation)
    active_rows = tuple(sorted(row_neighbors))
    active_columns = tuple(sorted(column_neighbors))
    matrix = tuple(
        tuple(
            int(column in row_neighbors[row])
            for column in active_columns
        )
        for row in active_rows
    )
    rank = _rank_over_q(matrix)

    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for row, columns in row_neighbors.items():
        for column in columns:
            left = ("r", row)
            right = ("c", column)
            adjacency[left].add(right)
            adjacency[right].add(left)
    unseen = set(adjacency)
    component_types: Counter[tuple[int, int, int]] = Counter()
    while unseen:
        start = min(unseen)
        queue = deque((start,))
        vertices = set()
        while queue:
            vertex = queue.popleft()
            if vertex in vertices:
                continue
            vertices.add(vertex)
            unseen.discard(vertex)
            queue.extend(adjacency[vertex].difference(vertices))
        rows = sum(kind == "r" for kind, _index in vertices)
        columns = len(vertices) - rows
        edges = sum(
            len(adjacency[vertex])
            for vertex in vertices
            if vertex[0] == "r"
        )
        component_types[(rows, columns, edges)] += 1
    component_count = sum(component_types.values())
    cycle_rank = len(table) - (
        len(active_rows) + len(active_columns)
    ) + component_count

    gauge = vertex_scalar_gauge_matrix()
    if _rank_over_q(gauge) != 5:
        raise KrennMixedAssociatedGradedError(
            "natural vertex gauge rank changed"
        )
    for row_index, row in enumerate(matrix):
        for direction in range(5):
            if sum(
                row[column_position]
                * gauge[active_columns[column_position]][direction]
                for column_position in range(len(active_columns))
            ):
                raise KrennMixedAssociatedGradedError(
                    f"Jacobian row {row_index} stopped killing gauge"
                )
    expected_types = {
        (1, 1, 1): 102,
        (2, 1, 2): 24,
        (4, 9, 12): 1,
    }
    if (
        len(table) != 162
        or len(active_rows) != 154
        or len(active_columns) != 135
        or rank != 130
        or component_count != 127
        or cycle_rank != 0
        or dict(component_types) != expected_types
    ):
        raise KrennMixedAssociatedGradedError(
            "natural Jacobian forest census changed"
        )
    return {
        "shape": [729, 135],
        "nonzero_entries": len(table),
        "active_rows": len(active_rows),
        "active_columns": len(active_columns),
        "row_degree_histogram": {
            str(key): value
            for key, value in sorted(
                Counter(map(len, row_neighbors.values())).items()
            )
        },
        "column_degree_histogram": {
            str(key): value
            for key, value in sorted(
                Counter(map(len, column_neighbors.values())).items()
            )
        },
        "component_count": component_count,
        "component_types": [
            {
                "rows": rows,
                "columns": columns,
                "edges": edges,
                "count": count,
            }
            for (rows, columns, edges), count in sorted(
                component_types.items()
            )
        ],
        "cycle_rank": cycle_rank,
        "rank_over_Q": rank,
        "nullity_over_Q": 135 - rank,
        "vertex_scalar_gauge_rank_over_Q": 5,
        "kernel_equals_vertex_scalar_gauge_over_Q": True,
        "first_order_holonomy_cycle_exists": False,
    }


def _edge_matching_incidence() -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[tuple[int, int], ...], ...],
]:
    edges = canonical_edges(6)
    matchings = tuple(map(_canonical_matching, perfect_matchings(6)))
    independent = tuple(
        map(_canonical_matching, independent_perfect_matchings(6))
    )
    if set(matchings) != set(independent) or len(matchings) != 15:
        raise KrennMixedAssociatedGradedError(
            "independent perfect-matching set disagreed"
        )
    matrix = tuple(
        tuple(int(edge in matching) for matching in matchings)
        for edge in edges
    )
    return matrix, edges, matchings


def _permutation_sign(values: Sequence[int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def _rotor_circuit(
    left: Sequence[int],
    matching_index: Mapping[tuple[tuple[int, int], ...], int],
) -> tuple[int, ...]:
    left = tuple(sorted(map(int, left)))
    right = tuple(vertex for vertex in range(6) if vertex not in left)
    vector = [0] * 15
    for assignment in permutations(range(3)):
        matching = _canonical_matching(
            tuple(
                (left[position], right[assignment[position]])
                for position in range(3)
            )
        )
        vector[matching_index[matching]] = _permutation_sign(assignment)
    return tuple(vector)


def _rotor_gate() -> dict:
    incidence, edges, matchings = _edge_matching_incidence()
    rank = _rank_over_q(incidence)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    minor = tuple(
        tuple(
            incidence[edge_index[edge]][matching]
            for matching in INCIDENCE_MINOR_MATCHINGS
        )
        for edge in INCIDENCE_MINOR_EDGES
    )
    minor_determinant = _determinant(minor)
    matching_index = {
        matching: index for index, matching in enumerate(matchings)
    }
    circuits = tuple(
        _rotor_circuit(partition, matching_index)
        for partition in ROTOR_PARTITIONS
    )
    for circuit in circuits:
        if any(
            sum(
                incidence[row][column] * circuit[column]
                for column in range(15)
            )
            for row in range(15)
        ):
            raise KrennMixedAssociatedGradedError(
                "a K3,3 circuit left the incidence kernel"
            )
    basis_minor = tuple(
        tuple(circuits[column][row] for column in range(5))
        for row in ROTOR_FREE_MATCHING_INDICES
    )
    basis_determinant = _determinant(basis_minor)

    transpose = _transpose(incidence)
    gram = _matrix_product(transpose, incidence)
    projector = tuple(
        tuple(
            Fraction(int(row == column))
            - gram[row][column] / 4
            + Fraction(1, 12)
            for column in range(15)
        )
        for row in range(15)
    )
    projector_squared = _matrix_product(projector, projector)
    incidence_projector = _matrix_product(incidence, projector)
    reconstruction = tuple(
        tuple(
            projector[row][column]
            + gram[row][column] / 4
            - Fraction(1, 12)
            for column in range(15)
        )
        for row in range(15)
    )
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(15))
        for row in range(15)
    )
    if (
        rank != 10
        or minor_determinant != -1
        or _rank_over_q(circuits) != 5
        or basis_determinant != -1
        or _rank_over_q(projector) != 5
        or projector_squared != projector
        or any(value for row in incidence_projector for value in row)
        or reconstruction != identity
    ):
        raise KrennMixedAssociatedGradedError(
            "edge-incidence rotor certificate changed"
        )

    equivariance_checks = 0
    for first in range(5):
        vertex_permutation = list(range(6))
        vertex_permutation[first], vertex_permutation[first + 1] = (
            vertex_permutation[first + 1],
            vertex_permutation[first],
        )
        for edge_position, edge in enumerate(edges):
            transported_edge = tuple(
                sorted(
                    (
                        vertex_permutation[edge[0]],
                        vertex_permutation[edge[1]],
                    )
                )
            )
            transported_row = edge_index[transported_edge]
            for matching_position, matching in enumerate(matchings):
                transported_matching = _canonical_matching(
                    tuple(
                        (
                            vertex_permutation[i],
                            vertex_permutation[j],
                        )
                        for i, j in matching
                    )
                )
                transported_column = matching_index[transported_matching]
                if (
                    incidence[edge_position][matching_position]
                    != incidence[transported_row][transported_column]
                ):
                    raise KrennMixedAssociatedGradedError(
                        "edge-incidence map lost S6 equivariance"
                    )
        equivariance_checks += 1

    first_circuit = circuits[0]
    circuit_support = tuple(
        index for index, value in enumerate(first_circuit) if value
    )
    restricted = tuple(
        tuple(row[column] for column in circuit_support)
        for row in incidence
    )
    positive = sum(first_circuit[index] == 1 for index in circuit_support)
    negative = sum(first_circuit[index] == -1 for index in circuit_support)
    if (
        _rank_over_q(restricted) != 5
        or positive != negative
        or positive != 3
    ):
        raise KrennMixedAssociatedGradedError(
            "primitive K3,3 parity corollary changed"
        )

    return {
        "edge_order": [list(edge) for edge in edges],
        "matching_order": [
            [list(edge) for edge in matching] for matching in matchings
        ],
        "incidence_matrix": [list(row) for row in incidence],
        "shape": [15, 15],
        "rank_over_Q": rank,
        "kernel_dimension_over_Q": 15 - rank,
        "unimodular_rank_minor": {
            "edge_rows": [list(edge) for edge in INCIDENCE_MINOR_EDGES],
            "matching_columns": list(INCIDENCE_MINOR_MATCHINGS),
            "determinant": _fraction_text(minor_determinant),
        },
        "primitive_K3_3_circuit_basis": {
            "partitions": [
                {
                    "left": list(left),
                    "right": [
                        vertex
                        for vertex in range(6)
                        if vertex not in left
                    ],
                }
                for left in ROTOR_PARTITIONS
            ],
            "vectors": [list(circuit) for circuit in circuits],
            "free_matching_coordinates": list(
                ROTOR_FREE_MATCHING_INDICES
            ),
            "basis_minor_determinant": _fraction_text(
                basis_determinant
            ),
            "integral_kernel_is_saturated": True,
            "integral_kernel_basis_certified": True,
        },
        "canonical_rotor_projector": {
            "formula": "P=I-(E^T E)/4+J/12",
            "matrix": [
                [_fraction_text(value) for value in row]
                for row in projector
            ],
            "rank_over_Q": 5,
            "idempotent": True,
            "image_equals_kernel_E": True,
            "reconstruction_formula": (
                "m=P*m+(E^T*q)/4-(sum(q))*1/36, q=E*m"
            ),
            "reconstruction_operator_is_identity": True,
            "mixed_equation_simplification": (
                "if sum(m)=0 then sum(q)=0 and "
                "m=P*m+(E^T*q)/4"
            ),
        },
        "S6_equivariance": {
            "adjacent_transposition_checks": equivariance_checks,
            "exact": True,
        },
        "representation_interpretation": {
            "matching_permutation_module": "[6]+[4,2]+[2,2,2]",
            "edge_permutation_module": "[6]+[5,1]+[4,2]",
            "kernel_label": "[2,2,2]",
            "kernel_name": "rotor sector",
            "dimension": 5,
        },
        "toric_holonomy": {
            "cross_multiplied_identity": (
                "product_even(m_M)=product_odd(m_M) "
                "for every primitive K3,3 parity circuit"
            ),
            "chart_independent": True,
            "full_color_diagonal_gauge_invariant": True,
            "division_on_zero_strata_used": False,
        },
        "parity_corollary": {
            "support": "the six matchings of one K3,3",
            "restricted_incidence_rank_over_Q": 5,
            "kernel": "m=alpha*C_parity",
            "toric_identity_after_substitution": (
                "alpha^3=(-alpha)^3"
            ),
            "characteristic_zero_consequence": "alpha=0",
            "nonzero_star_invisible_K3_3_rotor_realizable": False,
        },
    }


def _support_transport(
    support: Iterable[int],
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, ...]:
    result = []
    for index in support:
        i, j, a, b = variable_key(6, 3, int(index))
        result.append(
            variable_index(
                6,
                3,
                int(vertex_permutation[i]),
                int(vertex_permutation[j]),
                int(color_permutation[a]),
                int(color_permutation[b]),
            )
        )
    return tuple(sorted(result))


def _support_orbit_data(
    support: Iterable[int],
) -> tuple[tuple[int, ...], int]:
    images = {
        _support_transport(support, vertices, colors)
        for vertices in permutations(range(6))
        for colors in permutations(range(3))
    }
    return min(images), len(images)


def _matching_amplitude(
    coloring: Sequence[int],
    matching: Sequence[tuple[int, int]],
    weights: Mapping[int, LaurentPolynomial],
) -> LaurentPolynomial:
    result = LAURENT_ONE
    for variable in _matching_variables(coloring, matching):
        result *= weights.get(variable, LaurentPolynomial())
    return result


def _victim_projective_gate() -> dict:
    matchings = tuple(map(_canonical_matching, perfect_matchings(6)))
    independent = tuple(
        map(_canonical_matching, independent_perfect_matchings(6))
    )
    path = dict(natural_laurent_weight_entries())
    primary = {
        matching: _matching_amplitude(VICTIM_COLORING, matching, path)
        for matching in matchings
    }
    independent_values = {
        matching: _matching_amplitude(VICTIM_COLORING, matching, path)
        for matching in independent
    }
    if primary != independent_values:
        raise KrennMixedAssociatedGradedError(
            "independent victim matching amplitudes disagreed"
        )
    nonzero = tuple(
        (index, value)
        for index, matching in enumerate(matchings)
        if not (value := primary[matching]).is_zero
    )
    if (
        nonzero
        != (
            (
                VICTIM_MATCHING_INDEX,
                LaurentPolynomial.monomial(1),
            ),
        )
        or coloring_index(6, 3, VICTIM_COLORING)
        != N6_D3_SEED_DEFECT_EQUATION
    ):
        raise KrennMixedAssociatedGradedError(
            "known victim projective class changed"
        )

    natural_support = {
        index for index, _value in n6_d3_seed_witness().entries
    }
    matching_supports = tuple(
        set(_matching_variables(VICTIM_COLORING, matching))
        for matching in matchings
    )
    costs = tuple(
        len(support.difference(natural_support))
        for support in matching_supports
    )
    cost_histogram = Counter(costs)
    minimal = tuple(
        index for index, cost in enumerate(costs) if cost == 2
    )
    if (
        dict(cost_histogram) != {0: 1, 2: 6, 3: 8}
        or minimal != MINIMAL_REPAIR_MATCHING_INDICES
    ):
        raise KrennMixedAssociatedGradedError(
            "victim repair-cost census changed"
        )

    system = generate_sparse_system(6, 3)
    independent_system = verify_system_arrays(
        6,
        3,
        system.equation_offsets,
        system.monomial_variable_indices,
        system.rhs_values,
    )
    if not independent_system.exact:
        raise KrennMixedAssociatedGradedError(
            "independent branch-system replay failed"
        )
    reference_support = matching_supports[VICTIM_MATCHING_INDEX]
    branch_records = []
    orbit_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for matching_index_value in minimal:
        candidate_matching_support = matching_supports[
            matching_index_value
        ]
        support = tuple(
            sorted(natural_support.union(candidate_matching_support))
        )
        active_counts = []
        for equation in range(729):
            count = sum(
                set(monomial).issubset(support)
                for monomial in system.equation_monomials(equation)
            )
            active_counts.append(count)
        histogram = Counter(active_counts)
        singleton_equations = tuple(
            equation
            for equation, count in enumerate(active_counts)
            if count == 1
        )
        target_singletons = tuple(
            equation
            for equation in singleton_equations
            if system.rhs_values[equation]
        )
        spill_singletons = tuple(
            equation
            for equation in singleton_equations
            if not system.rhs_values[equation]
        )
        double_equations = tuple(
            equation
            for equation, count in enumerate(active_counts)
            if count == 2
        )
        if (
            len(support) != 11
            or dict(histogram) != {0: 723, 1: 5, 2: 1}
            or target_singletons != (0, 364, 728)
            or len(spill_singletons) != 2
            or double_equations != (N6_D3_SEED_DEFECT_EQUATION,)
        ):
            raise KrennMixedAssociatedGradedError(
                "a minimal victim branch incidence profile changed"
            )
        numerator = tuple(
            sorted(candidate_matching_support.difference(reference_support))
        )
        denominator = tuple(
            sorted(reference_support.difference(candidate_matching_support))
        )
        if len(numerator) != 2 or len(denominator) != 2:
            raise KrennMixedAssociatedGradedError(
                "minimal victim ratio stopped being two-by-two"
            )
        representative, orbit_size = _support_orbit_data(support)
        orbit_groups[representative].append(matching_index_value)
        branch_records.append(
            {
                "matching_index": matching_index_value,
                "matching": [
                    list(edge) for edge in matchings[matching_index_value]
                ],
                "new_coordinate_indices": list(
                    sorted(
                        candidate_matching_support.difference(
                            natural_support
                        )
                    )
                ),
                "new_coordinates": [
                    list(variable_key(6, 3, index))
                    for index in sorted(
                        candidate_matching_support.difference(
                            natural_support
                        )
                    )
                ],
                "ratio_numerator_indices": list(numerator),
                "ratio_denominator_indices": list(denominator),
                "ratio": (
                    "product(numerator)/product(denominator)"
                ),
                "initial_relation": "r_N=-1",
                "cross_multiplied_relation": "m_N+m_Mstar=0",
                "support": list(support),
                "support_size": len(support),
                "support_orbit_size": orbit_size,
                "active_monomial_count_histogram": {
                    str(key): value
                    for key, value in sorted(histogram.items())
                },
                "target_singleton_equations": list(target_singletons),
                "non_target_singleton_spills": list(spill_singletons),
                "non_target_singleton_spill_colorings": [
                    list(coloring_from_index(6, 3, equation))
                    for equation in spill_singletons
                ],
                "double_equation": N6_D3_SEED_DEFECT_EQUATION,
                "size11_branch_is_exact_GHZ_witness": False,
            }
        )
    grouped = sorted(sorted(values) for values in orbit_groups.values())
    if (
        grouped != [[0, 4, 8], [2, 9, 13]]
        or any(record["support_orbit_size"] != 1_080 for record in branch_records)
    ):
        raise KrennMixedAssociatedGradedError(
            "minimal victim support-orbit split changed"
        )

    return {
        "victim_coloring": list(VICTIM_COLORING),
        "victim_equation": N6_D3_SEED_DEFECT_EQUATION,
        "native_matching_order": [
            [list(edge) for edge in matching] for matching in matchings
        ],
        "known_path": {
            "only_nonzero_matching_index": VICTIM_MATCHING_INDEX,
            "matching": [
                list(edge) for edge in matchings[VICTIM_MATCHING_INDEX]
            ],
            "amplitude": "t",
            "projective_class_after_inverse_1PS": "e_Mstar",
            "exact_mixed_hyperplane": "sum_M m_M=0",
            "known_projective_point_lies_on_hyperplane": False,
            "all_transverse_ratios_positive_order_can_repair": False,
            "reason": (
                "in the Mstar chart the equation is "
                "1+sum_(N!=Mstar) r_N=0"
            ),
        },
        "repair_cost_histogram": {
            str(cost): cost_histogram.get(cost, 0)
            for cost in range(4)
        },
        "minimal_cost_two_matching_indices": list(minimal),
        "minimal_branch_support_orbits": grouped,
        "branch_records": branch_records,
        "consequence": (
            "within the Mstar chart a repairing arc must have at least "
            "one transverse ratio of nonpositive valuation; each of the "
            "six minimal cost-two single-matching equality branches "
            "creates two forced singleton spills and needs further repair"
        ),
    }


@lru_cache(maxsize=1)
def _build_mixed_associated_graded_gate() -> dict:
    cofactor = _cofactor_and_star_gate()
    covariance = _all_order_covariance()
    jacobian = _jacobian_forest(covariance)
    rotor = _rotor_gate()
    victim = _victim_projective_gate()
    claim_boundary = {
        "finite_exact_GHZ_witness_found": False,
        "affine_GHZ_membership_decided": False,
        "global_GHZ_nonmembership_proved": False,
        "strict_border_membership_reproved_here": False,
        (
            "tested_known_Laurent_orbit_raw_associated_graded_"
            "holonomy_closed_negatively"
        ): True,
        "all_possible_Laurent_or_formal_branches_classified": False,
        "rotor_sector_gives_global_exclusion_by_itself": False,
        "minimal_size11_projective_branches_are_exact_witnesses": False,
        "bounded_numerical_miss_is_proof": False,
    }
    return {
        "schema": MIXED_GATE_SCHEMA,
        "scope": {
            "parameters": {"n": 6, "d": 3},
            "benchmarks": [
                "K4_d3_exact_witness",
                "n6_d3_natural_Laurent_orbit",
            ],
            "cpu_workers": 1,
            "floating_point_used": False,
            "random_seeds": [],
            "matching_enumerators": ["primary", "independent"],
        },
        "mixed_cofactor_gate": cofactor,
        "all_derivative_orders_on_known_orbit": covariance,
        "natural_jacobian_support_forest": jacobian,
        "matching_amplitude_rotor": rotor,
        "victim_projective_departure": victim,
        "conclusion": {
            "classification": (
                "exact-negative-raw-associated-graded-preflight"
            ),
            "raw_star_Jacobian_Hessian_cubic_path_stopped": True,
            "smallest_faithful_fixed_coloring_state": (
                "(q=E*m, h=P_rot*m)"
            ),
            "next_finite_search": (
                "two symmetry-orbit representatives of the six minimal "
                "cost-two single-matching victim departures, retaining "
                "rotor coordinates and cross-multiplied K3,3 parity "
                "identities"
            ),
            "next_search_success_is_only_initial_form_candidate": True,
        },
        "claim_boundary": claim_boundary,
    }


def build_mixed_associated_graded_gate() -> dict:
    """Return an isolated copy of the exact cached certificate."""

    return deepcopy(_build_mixed_associated_graded_gate())


def verify_mixed_associated_graded_gate(payload: Mapping) -> dict:
    """Recompute every exact datum and reject any changed claim."""

    if not isinstance(payload, Mapping):
        raise KrennMixedAssociatedGradedError(
            "mixed associated-graded certificate must be a mapping"
        )
    expected = build_mixed_associated_graded_gate()
    if dict(payload) != expected:
        raise KrennMixedAssociatedGradedError(
            "mixed associated-graded certificate differs from exact replay"
        )
    return expected


def _canonical_json(payload: Mapping) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _readme_text(certificate: Mapping) -> str:
    jacobian = certificate["natural_jacobian_support_forest"]
    rotor = certificate["matching_amplitude_rotor"]
    victim = certificate["victim_projective_departure"]
    return f"""# Mixed associated-graded and rotor gate

This is an exact stopping and routing certificate for the known
`n=6,d=3` Laurent family.  It is not an affine membership or nonmembership
proof.

## What closes

All 1,215 deleted-pair cofactor coordinates were replayed with both
perfect-matching enumerators.  Exactly 18 are nonzero: nine monochromatic
and nine mixed.  The six `243 x 15` star matrices satisfy

```text
A_r(t) = diag(t^R(row)) A_r(1) diag(t^(-x_(v,b)))
```

entry by entry.  The same character telescope holds for all nonzero
coordinate coefficients of derivative orders 0, 1, 2, and 3 of the cubic
matching map.  Thus raw star, Jacobian, Hessian, and cubic contractions on
this particular orbit cannot reveal gauge-independent growth.

The Jacobian has {jacobian["nonzero_entries"]} nonzeros, rank
{jacobian["rank_over_Q"]}, nullity {jacobian["nullity_over_Q"]}, and its
nonzero bipartite support graph has cycle rank zero.  There is no hidden
first-order holonomy cycle.

## What survives

For one fixed coloring, let `m` be its 15 perfect-matching amplitudes and
let `q=E*m` be the 15 edge/apex marginals.  The exact incidence matrix has
rank {rotor["rank_over_Q"]} and a saturated rank-five integer kernel.  Five
primitive `K3,3` parity circuits form a `Z`-basis; both displayed basis
minors have determinant `-1`.

The canonical projector

```text
P_rot = I - (E^T E)/4 + J/12
```

has image `ker(E)`.  Hence `(q, P_rot*m)`, not `q` alone, is the smallest
faithful state in this decomposition.  Cross-multiplied parity binomials
are the genuine chart-independent toric holonomies.

## Concrete departure from the known branch

For victim `002121`, the known path has only matching
`{victim["known_path"]["matching"]}` active, with amplitude `t`.  Its
projective class is constant and is not on `sum_M m_M=0`.  Any exact repair
that stays in the `Mstar` chart must have at least one transverse ratio of
nonpositive valuation; it cannot converge projectively back to `e_Mstar`.

The exact matching repair-cost census is

```text
cost 0: {victim["repair_cost_histogram"]["0"]}
cost 1: {victim["repair_cost_histogram"]["1"]}
cost 2: {victim["repair_cost_histogram"]["2"]}
cost 3: {victim["repair_cost_histogram"]["3"]}
```

The six cost-two alternatives split into the two support-orbit classes
`[0,4,8]` and `[2,9,13]`.  Imposing the minimal relation `r_N=-1`
cancels the victim, but each size-11 branch creates exactly two non-target
singleton spills.  Those six branches are starting states, not witnesses.

## Reproduction

```powershell
Set-Location 'C:\\Users\\brssn\\Projects\\Krenn-counterexample-search'

C:\\tmp\\Krenn-obstruction-venv\\Scripts\\python.exe -B -m unittest -v `
  tests.test_krenn_mixed_associated_graded

C:\\tmp\\Krenn-obstruction-venv\\Scripts\\python.exe -B -m `
  experiments.krenn_quantum_graph.mixed_associated_graded `
  --results-directory `
  results\\krenn_quantum_graph\\n6_d3_mixed_associated_graded_gate `
  --verify-only
```

The computation is deterministic, serial, exact over `Q[t,t^-1]`, and
uses no floating point or cloud resources.
"""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        raise KrennMixedAssociatedGradedError(
            f"mixed gate manifest dependency is missing: {path}"
        )
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _expected_manifest(directory: Path, certificate: Mapping) -> dict:
    repository = Path(__file__).resolve().parents[2]
    return {
        "schema": MIXED_MANIFEST_SCHEMA,
        "bundle_files": {
            relative: _file_record(directory / relative)
            for relative in ("README.md", "certificate.json")
        },
        "source_files": {
            relative: _file_record(repository / relative)
            for relative in SOURCE_PATHS
        },
        "claim_boundary": certificate["claim_boundary"],
    }


def write_mixed_associated_graded_bundle(directory: Path) -> dict:
    """Write the exact certificate, README, and dependency manifest."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    certificate = build_mixed_associated_graded_gate()
    (directory / "certificate.json").write_text(
        _canonical_json(certificate), encoding="utf-8", newline="\n"
    )
    (directory / "README.md").write_text(
        _readme_text(certificate), encoding="utf-8", newline="\n"
    )
    manifest = _expected_manifest(directory, certificate)
    (directory / "manifest.json").write_text(
        _canonical_json(manifest), encoding="utf-8", newline="\n"
    )
    return manifest


def verify_mixed_associated_graded_bundle(directory: Path) -> dict:
    """Verify canonical content, source hashes, and all exact claims."""

    directory = Path(directory)
    try:
        certificate_text = (directory / "certificate.json").read_text(
            encoding="utf-8"
        )
        readme_text = (directory / "README.md").read_text(encoding="utf-8")
        manifest_text = (directory / "manifest.json").read_text(
            encoding="utf-8"
        )
        certificate = json.loads(certificate_text)
        manifest = json.loads(manifest_text)
    except (OSError, json.JSONDecodeError) as error:
        raise KrennMixedAssociatedGradedError(
            "mixed associated-graded bundle cannot be read"
        ) from error
    expected_certificate = verify_mixed_associated_graded_gate(certificate)
    if certificate_text != _canonical_json(expected_certificate):
        raise KrennMixedAssociatedGradedError(
            "mixed certificate serialization changed"
        )
    if readme_text != _readme_text(expected_certificate):
        raise KrennMixedAssociatedGradedError(
            "mixed gate README differs from exact regeneration"
        )
    expected_manifest = _expected_manifest(directory, expected_certificate)
    if (
        manifest != expected_manifest
        or manifest_text != _canonical_json(expected_manifest)
    ):
        raise KrennMixedAssociatedGradedError(
            "mixed gate manifest differs from exact regeneration"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the exact mixed associated-graded gate."
    )
    parser.add_argument(
        "--results-directory", type=Path, default=DEFAULT_RESULTS_DIRECTORY
    )
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.verify_only:
        verify_mixed_associated_graded_bundle(arguments.results_directory)
    else:
        write_mixed_associated_graded_bundle(arguments.results_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
