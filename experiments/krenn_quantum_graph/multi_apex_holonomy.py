r"""Exact gate for the first multi-apex holonomy proposal.

The simultaneous star identities suggest comparing the local right inverses
at different apex vertices.  This module tests the smallest versions of that
idea against the two mandatory benchmarks:

* the finite ``K4,d=3`` exception; and
* the exact ``n=6,d=3`` Laurent family approaching GHZ.

The outcome is a stopping result, not a nonexistence theorem.  The punctured
known Laurent family is exactly a one-parameter orbit of the full
color-diagonal GHZ stabilizer.  The chart-free cross-apex transition vanishes
on both sparse benchmarks, the deterministic scalar apex minors have
cancelling valuations, and the gauge-invariant monochromatic Pluecker terms
are identically one.

One first degree-nine mixed-equation repair is also tested exactly.  If

``C^a_rs = w_rs^(aa) H_(V\{r,s})(a,...,a)``

and ``A_a = H_V(a,...,a)``, the degree-nine proposal

``C^0 C^1 C^2 = A_0 A_1 A_2 I_6``

does not lie in the mixed-residual ideal.  The existing exact two-row dual
annihilates the whole degree-nine source map but pairs to 2160 with the trace
of the proposed identity.  This rules out that bounded ideal-certificate
ansatz only; radical membership and set-theoretic consequences remain open.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeAlias

from experiments.krenn_quantum_graph.border_image import (
    LAURENT_ONE,
    LAURENT_ZERO,
    LaurentPolynomial,
    natural_laurent_weight_entries,
)
from experiments.krenn_quantum_graph.fixtures import fixture_n4_d3
from experiments.krenn_quantum_graph.independent_verifier import (
    independent_perfect_matchings,
)
from experiments.krenn_quantum_graph.source_ideal import (
    certify_reynolds_reduction,
    exact_two_row_dual,
)
from experiments.krenn_quantum_graph.system import (
    Monomial,
    perfect_matchings,
    validate_parameters,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    n6_d3_seed_witness,
)


HOLONOMY_GATE_SCHEMA = "krenn-multi-apex-holonomy-gate-v1"
HOLONOMY_MANIFEST_SCHEMA = "krenn-multi-apex-holonomy-manifest-v1"
DEFAULT_RESULTS_DIRECTORY = (
    Path("results")
    / "krenn_quantum_graph"
    / "n6_d3_multi_apex_holonomy_gate"
)
MAX_LAURENT_DETERMINANT_SIZE = 15
SOURCE_PATHS = (
    "experiments/krenn_quantum_graph/border_image.py",
    "experiments/krenn_quantum_graph/fixtures.py",
    "experiments/krenn_quantum_graph/independent_verifier.py",
    "experiments/krenn_quantum_graph/multi_apex_holonomy.py",
    "experiments/krenn_quantum_graph/source_ideal.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/ternary_search.py",
    "experiments/krenn_quantum_graph/witness.py",
    "tests/test_krenn_multi_apex_holonomy.py",
)

LaurentMatrix: TypeAlias = tuple[tuple[LaurentPolynomial, ...], ...]
MatchingEnumerator: TypeAlias = Callable[
    [int], Sequence[Sequence[tuple[int, int]]]
]


class KrennMultiApexHolonomyError(ValueError):
    """A holonomy input, certificate, or bundle failed exact replay."""


def _exact_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KrennMultiApexHolonomyError(
            f"{label} must be an exact integer"
        )
    return value


def _as_laurent(
    value: LaurentPolynomial | int | Fraction,
) -> LaurentPolynomial:
    return (
        value
        if isinstance(value, LaurentPolynomial)
        else LaurentPolynomial.constant(value)
    )


def _fraction_text(value: Fraction | int) -> str:
    value = Fraction(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def laurent_specialize(
    value: LaurentPolynomial,
    point: Fraction | int,
) -> Fraction:
    """Evaluate a Laurent polynomial at an exact nonzero rational point."""

    if not isinstance(value, LaurentPolynomial):
        raise KrennMultiApexHolonomyError(
            "Laurent specialization needs a LaurentPolynomial"
        )
    point = Fraction(point)
    if not point:
        raise KrennMultiApexHolonomyError(
            "Laurent specialization point must be nonzero"
        )
    return sum(
        (
            coefficient * point**exponent
            for exponent, coefficient in value.terms
        ),
        Fraction(0),
    )


def laurent_valuation(value: LaurentPolynomial) -> int:
    """Return the lowest exponent of a nonzero Laurent polynomial."""

    if not isinstance(value, LaurentPolynomial) or value.is_zero:
        raise KrennMultiApexHolonomyError(
            "Laurent valuation needs a nonzero LaurentPolynomial"
        )
    return value.terms[0][0]


def _validated_weights(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial | int | Fraction],
) -> dict[int, LaurentPolynomial]:
    validate_parameters(n, d)
    result: dict[int, LaurentPolynomial] = {}
    for raw_index, raw_value in weights.items():
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise KrennMultiApexHolonomyError(
                "weight indices must be exact integers"
            )
        if not 0 <= raw_index < variable_count(n, d):
            raise KrennMultiApexHolonomyError(
                "a Laurent weight index is outside the system"
            )
        value = _as_laurent(raw_value)
        if not value.is_zero:
            result[raw_index] = value
    return result


def _weight(
    weights: Mapping[int, LaurentPolynomial],
    n: int,
    d: int,
    i: int,
    j: int,
    a: int,
    b: int,
) -> LaurentPolynomial:
    return weights.get(variable_index(n, d, i, j, a, b), LAURENT_ZERO)


def laurent_induced_matching_sum(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial | int | Fraction],
    coloring: Sequence[int],
    vertices: Sequence[int],
    *,
    matching_enumerator: MatchingEnumerator = perfect_matchings,
) -> LaurentPolynomial:
    """Evaluate one induced perfect-matching sum over ``Q[t,t^-1]``."""

    validate_parameters(n, d)
    checked_weights = _validated_weights(n, d, weights)
    checked_coloring = tuple(
        _exact_integer(color, "coloring entry") for color in coloring
    )
    checked_vertices = tuple(
        _exact_integer(vertex, "induced vertex") for vertex in vertices
    )
    if (
        len(checked_coloring) != n
        or any(color not in range(d) for color in checked_coloring)
        or len(checked_vertices) != len(set(checked_vertices))
        or any(vertex not in range(n) for vertex in checked_vertices)
        or len(checked_vertices) % 2
    ):
        raise KrennMultiApexHolonomyError(
            "invalid coloring or induced vertex set"
        )
    if not checked_vertices:
        return LAURENT_ONE
    total = LAURENT_ZERO
    for matching in matching_enumerator(len(checked_vertices)):
        term = LAURENT_ONE
        for local_i, local_j in matching:
            i = checked_vertices[local_i]
            j = checked_vertices[local_j]
            term *= _weight(
                checked_weights,
                n,
                d,
                i,
                j,
                checked_coloring[i],
                checked_coloring[j],
            )
        total += term
    return total


def laurent_matching_cofactor(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial | int | Fraction],
    coloring: Sequence[int],
    first: int,
    second: int,
    *,
    matching_enumerator: MatchingEnumerator = perfect_matchings,
) -> LaurentPolynomial:
    """Return the Laurent matching sum after deleting two vertices."""

    if (
        isinstance(first, bool)
        or isinstance(second, bool)
        or not isinstance(first, int)
        or not isinstance(second, int)
        or first == second
        or first not in range(n)
        or second not in range(n)
    ):
        raise KrennMultiApexHolonomyError(
            "a cofactor needs two distinct valid vertices"
        )
    return laurent_induced_matching_sum(
        n,
        d,
        weights,
        coloring,
        tuple(
            vertex
            for vertex in range(n)
            if vertex not in (first, second)
        ),
        matching_enumerator=matching_enumerator,
    )


def laurent_star_coefficient_matrix(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial | int | Fraction],
    root: int,
    *,
    matching_enumerator: MatchingEnumerator = perfect_matchings,
) -> LaurentMatrix:
    """Build the exact ``d^(n-1) x d(n-1)`` star coefficient matrix."""

    validate_parameters(n, d)
    root = _exact_integer(root, "star root")
    if root not in range(n):
        raise KrennMultiApexHolonomyError(
            "star root is outside range(n)"
        )
    checked_weights = _validated_weights(n, d, weights)
    others = tuple(vertex for vertex in range(n) if vertex != root)
    columns = tuple(
        (neighbor, color)
        for neighbor in others
        for color in range(d)
    )
    rows = []
    for residual in product(range(d), repeat=n - 1):
        residual_by_vertex = dict(zip(others, residual, strict=True))
        coloring = tuple(
            0 if vertex == root else residual_by_vertex[vertex]
            for vertex in range(n)
        )
        cofactors = {
            neighbor: laurent_matching_cofactor(
                n,
                d,
                checked_weights,
                coloring,
                root,
                neighbor,
                matching_enumerator=matching_enumerator,
            )
            for neighbor in others
        }
        rows.append(
            tuple(
                (
                    cofactors[neighbor]
                    if residual_by_vertex[neighbor] == color
                    else LAURENT_ZERO
                )
                for neighbor, color in columns
            )
        )
    return tuple(rows)


def _rank_over_q(
    matrix: Sequence[Sequence[Fraction | int]],
) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennMultiApexHolonomyError(
            "rational rank input is ragged"
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
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(rank + 1, len(rows)):
            factor = rows[row][column]
            if factor:
                rows[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(
                        rows[row], rows[rank], strict=True
                    )
                ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def lexicographic_independent_rows(
    matrix: LaurentMatrix,
    *,
    specialization: Fraction | int = 1,
) -> tuple[int, ...]:
    """Select the lexicographically first full-rank square row minor."""

    if not matrix or not matrix[0]:
        raise KrennMultiApexHolonomyError(
            "row selection needs a nonempty matrix"
        )
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise KrennMultiApexHolonomyError(
            "row selection matrix is ragged"
        )
    chosen: list[int] = []
    rational_rows: list[tuple[Fraction, ...]] = []
    rank = 0
    for index, row in enumerate(matrix):
        candidate = tuple(
            laurent_specialize(value, specialization) for value in row
        )
        candidate_rank = _rank_over_q((*rational_rows, candidate))
        if candidate_rank > rank:
            chosen.append(index)
            rational_rows.append(candidate)
            rank = candidate_rank
        if rank == width:
            return tuple(chosen)
    raise KrennMultiApexHolonomyError(
        "star matrix is not full column rank at the specialization"
    )


def laurent_determinant(
    matrix: LaurentMatrix,
) -> LaurentPolynomial:
    """Compute a capped square determinant without division."""

    size = len(matrix)
    if (
        not size
        or size > MAX_LAURENT_DETERMINANT_SIZE
        or any(len(row) != size for row in matrix)
    ):
        raise KrennMultiApexHolonomyError(
            "Laurent determinant needs a square matrix of size 1..15"
        )

    @lru_cache(maxsize=None)
    def recurse(mask: int) -> LaurentPolynomial:
        remaining = tuple(
            column for column in range(size) if mask & (1 << column)
        )
        if not remaining:
            return LAURENT_ONE
        row = size - len(remaining)
        total = LAURENT_ZERO
        for position, column in enumerate(remaining):
            term = matrix[row][column] * recurse(mask & ~(1 << column))
            total += term if position % 2 == 0 else -term
        return total

    return recurse((1 << size) - 1)


def _matrix_product(
    left: LaurentMatrix,
    right: LaurentMatrix,
) -> LaurentMatrix:
    if (
        not left
        or not right
        or len(left[0]) != len(right)
        or any(len(row) != len(left[0]) for row in left)
        or any(len(row) != len(right[0]) for row in right)
    ):
        raise KrennMultiApexHolonomyError(
            "Laurent matrix product has incompatible dimensions"
        )
    return tuple(
        tuple(
            sum(
                (
                    left[row][middle] * right[middle][column]
                    for middle in range(len(right))
                ),
                LAURENT_ZERO,
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _matrix_trace(matrix: LaurentMatrix) -> LaurentPolynomial:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise KrennMultiApexHolonomyError(
            "trace needs a nonempty square matrix"
        )
    return sum(
        (matrix[index][index] for index in range(len(matrix))),
        LAURENT_ZERO,
    )


def _constant_laurent_weights_from_fixture() -> dict[int, LaurentPolynomial]:
    return {
        index: LaurentPolynomial.constant(value)
        for index, value in fixture_n4_d3().entries
    }


def natural_stabilizer_exponents() -> tuple[int, ...]:
    """Return the exact 18-entry cocharacter generating the known path."""

    exponents = [0] * 18
    exponents[0 * 3 + 0] = 1
    exponents[2 * 3 + 0] = -1
    return tuple(exponents)


def certify_natural_stabilizer_orbit(
    weights: Mapping[int, LaurentPolynomial | int | Fraction],
) -> tuple[tuple[int, int, int, str], ...]:
    """Prove exact equality with the cocharacter orbit of the unit seed."""

    checked = _validated_weights(6, 3, weights)
    seed = {
        index: LaurentPolynomial.constant(value)
        for index, value in n6_d3_seed_witness().entries
    }
    if set(checked) != set(seed):
        raise KrennMultiApexHolonomyError(
            "Laurent path support differs from the natural seed"
        )
    exponents = natural_stabilizer_exponents()
    records = []
    for index, seed_value in sorted(seed.items()):
        character = _weight_cocharacter(6, 3, index, exponents)
        expected = LaurentPolynomial(
            tuple(
                (exponent + character, coefficient)
                for exponent, coefficient in seed_value.terms
            )
        )
        actual = checked[index]
        if actual != expected:
            raise KrennMultiApexHolonomyError(
                "Laurent path is not exactly the recorded stabilizer orbit"
            )
        records.append(
            (
                index,
                character,
                laurent_valuation(actual),
                actual.to_expression(),
            )
        )
    return tuple(records)


def _weight_cocharacter(
    n: int,
    d: int,
    index: int,
    exponents: Sequence[int],
) -> int:
    i, j, a, b = variable_key(n, d, index)
    return exponents[i * d + a] + exponents[j * d + b]


def _coloring_character(
    coloring: Sequence[int],
    exponents: Sequence[int],
    d: int,
) -> int:
    return sum(
        exponents[vertex * d + color]
        for vertex, color in enumerate(coloring)
    )


def _monochromatic_cofactor(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial],
    root: int,
    neighbor: int,
    color: int,
    *,
    matching_enumerator: MatchingEnumerator = perfect_matchings,
) -> LaurentPolynomial:
    return laurent_matching_cofactor(
        n,
        d,
        weights,
        (color,) * n,
        root,
        neighbor,
        matching_enumerator=matching_enumerator,
    )


def monochromatic_marginal_matrix(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial | int | Fraction],
    color: int,
    *,
    matching_enumerator: MatchingEnumerator = perfect_matchings,
) -> LaurentMatrix:
    """Return the exact invariant marginal matrix ``C^color``."""

    checked_weights = _validated_weights(n, d, weights)
    color = _exact_integer(color, "marginal color")
    if color not in range(d):
        raise KrennMultiApexHolonomyError(
            "marginal color is outside range(d)"
        )
    result = [
        [LAURENT_ZERO for _column in range(n)] for _row in range(n)
    ]
    for first in range(n):
        for second in range(first + 1, n):
            value = _weight(
                checked_weights,
                n,
                d,
                first,
                second,
                color,
                color,
            ) * _monochromatic_cofactor(
                n,
                d,
                checked_weights,
                first,
                second,
                color,
                matching_enumerator=matching_enumerator,
            )
            result[first][second] = value
            result[second][first] = value
    return tuple(tuple(row) for row in result)


def cross_apex_transition(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial | int | Fraction],
    target_root: int,
    source_root: int,
    *,
    matching_enumerator: MatchingEnumerator = perfect_matchings,
) -> LaurentMatrix:
    r"""Return the chart-free overlap ``T_(target<-source)=B_r Y_s``."""

    checked_weights = _validated_weights(n, d, weights)
    target_root = _exact_integer(target_root, "target root")
    source_root = _exact_integer(source_root, "source root")
    if (
        target_root not in range(n)
        or source_root not in range(n)
        or target_root == source_root
    ):
        raise KrennMultiApexHolonomyError(
            "cross-apex transition needs two distinct valid roots"
        )
    result = []
    for target_color in range(d):
        row = []
        for source_color in range(d):
            total = LAURENT_ZERO
            for vertex in range(n):
                if vertex in (target_root, source_root):
                    continue
                total += _monochromatic_cofactor(
                    n,
                    d,
                    checked_weights,
                    target_root,
                    vertex,
                    target_color,
                    matching_enumerator=matching_enumerator,
                ) * _weight(
                    checked_weights,
                    n,
                    d,
                    source_root,
                    vertex,
                    source_color,
                    target_color,
                )
            row.append(total)
        result.append(tuple(row))
    return tuple(result)


def _permutation_from_unit_matrix(matrix: LaurentMatrix) -> tuple[int, ...]:
    result = []
    for row in matrix:
        active = tuple(
            column for column, value in enumerate(row) if not value.is_zero
        )
        if len(active) != 1 or row[active[0]] != LAURENT_ONE:
            raise KrennMultiApexHolonomyError(
                "benchmark marginal product is not a unit permutation"
            )
        result.append(active[0])
    if len(set(result)) != len(result):
        raise KrennMultiApexHolonomyError(
            "benchmark marginal product is not bijective"
        )
    return tuple(result)


def _cycles(permutation: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    result = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle = []
        current = start
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = permutation[current]
        result.append(tuple(cycle))
    return tuple(result)


def abstract_flat_marginal_counterexample() -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[tuple[Fraction, ...], ...],
]:
    """Return exact rational n=6 marginal axioms with flat holonomy."""

    size = 6

    def shift(power: int) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(
                Fraction(int(column == (row + power) % size))
                for column in range(size)
            )
            for row in range(size)
        )

    def combine(
        terms: Sequence[
            tuple[Fraction | int, tuple[tuple[Fraction, ...], ...]]
        ],
    ) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(
                sum(
                    (
                        Fraction(coefficient) * matrix[row][column]
                        for coefficient, matrix in terms
                    ),
                    Fraction(0),
                )
                for column in range(size)
            )
            for row in range(size)
        )

    s1, s2, s3, s4, s5 = (shift(power) for power in range(1, 6))
    first = s3
    second = combine(((Fraction(1, 2), s2), (Fraction(1, 2), s4)))
    third = combine(((1, s1), (1, s5), (-1, s3)))
    return first, second, third


def _fraction_matrix_product(
    left: Sequence[Sequence[Fraction | int]],
    right: Sequence[Sequence[Fraction | int]],
) -> tuple[tuple[Fraction, ...], ...]:
    left = tuple(tuple(map(Fraction, row)) for row in left)
    right = tuple(tuple(map(Fraction, row)) for row in right)
    if not left or not right or len(left[0]) != len(right):
        raise KrennMultiApexHolonomyError(
            "rational matrix product has incompatible dimensions"
        )
    return tuple(
        tuple(
            sum(
                (
                    left[row][middle] * right[middle][column]
                    for middle in range(len(right))
                ),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _matching_permutation(
    matching: Sequence[tuple[int, int]],
    n: int,
) -> tuple[int, ...]:
    result = list(range(n))
    used: set[int] = set()
    for first, second in matching:
        if (
            first == second
            or first not in range(n)
            or second not in range(n)
            or first in used
            or second in used
        ):
            raise KrennMultiApexHolonomyError(
                "matching does not cover distinct vertices"
            )
        used.update((first, second))
        result[first] = second
        result[second] = first
    if used != set(range(n)):
        raise KrennMultiApexHolonomyError(
            "matching does not cover every vertex"
        )
    return tuple(result)


def trace_holonomy_coefficient(monomial: Monomial) -> int:
    r"""Return the coefficient in ``tr(C0 C1 C2)-6 A0 A1 A2``."""

    if len(monomial) != 9:
        raise KrennMultiApexHolonomyError(
            "trace-holonomy monomial must have degree nine"
        )
    by_color: list[list[tuple[int, int]]] = [[], [], []]
    for index in monomial:
        i, j, a, b = variable_key(6, 3, index)
        if a != b:
            raise KrennMultiApexHolonomyError(
                "trace-holonomy monomial must be monochromatic"
            )
        by_color[a].append((i, j))
    permutations_by_color = tuple(
        _matching_permutation(matching, 6) for matching in by_color
    )
    destination = tuple(
        permutations_by_color[2][
            permutations_by_color[1][permutations_by_color[0][vertex]]
        ]
        for vertex in range(6)
    )
    return sum(
        destination[vertex] == vertex for vertex in range(6)
    ) - 6


def _trace_coefficient_histogram() -> dict[int, int]:
    matchings = perfect_matchings(6)
    permutations_by_matching = tuple(
        _matching_permutation(matching, 6) for matching in matchings
    )
    histogram: Counter[int] = Counter()
    for first, second, third in product(range(15), repeat=3):
        permutations_for_triple = (
            permutations_by_matching[first],
            permutations_by_matching[second],
            permutations_by_matching[third],
        )
        destination = tuple(
            permutations_for_triple[2][
                permutations_for_triple[1][
                    permutations_for_triple[0][vertex]
                ]
            ]
            for vertex in range(6)
        )
        histogram[
            sum(destination[vertex] == vertex for vertex in range(6)) - 6
        ] += 1
    return dict(sorted(histogram.items()))


def _trace_invariance_checks() -> int:
    """Replay S6 x S3 invariance of the trace coefficient on all triples."""

    matchings = perfect_matchings(6)
    matching_sets = tuple(
        frozenset(tuple(sorted(edge)) for edge in matching)
        for matching in matchings
    )
    matching_index = {
        matching: index for index, matching in enumerate(matching_sets)
    }
    matching_permutations = tuple(
        _matching_permutation(matching, 6) for matching in matchings
    )
    vertex_generators = tuple(
        tuple(
            (
                vertex + 1
                if vertex == swap
                else swap
                if vertex == swap + 1
                else vertex
            )
            for vertex in range(6)
        )
        for swap in range(5)
    )
    color_generators = ((1, 0, 2), (0, 2, 1))
    checks = 0

    def coefficient(indices: Sequence[int]) -> int:
        p0, p1, p2 = (
            matching_permutations[index] for index in indices
        )
        return (
            sum(
                p2[p1[p0[vertex]]] == vertex for vertex in range(6)
            )
            - 6
        )

    for indices in product(range(15), repeat=3):
        expected = coefficient(indices)
        for vertex_permutation in vertex_generators:
            transported = []
            for index in indices:
                transported_matching = frozenset(
                    tuple(
                        sorted(
                            (
                                vertex_permutation[first],
                                vertex_permutation[second],
                            )
                        )
                    )
                    for first, second in matchings[index]
                )
                transported.append(matching_index[transported_matching])
            if coefficient(transported) != expected:
                raise KrennMultiApexHolonomyError(
                    "trace coefficient lost S6 invariance"
                )
            checks += 1
        for color_permutation in color_generators:
            transported = tuple(indices[color] for color in color_permutation)
            if coefficient(transported) != expected:
                raise KrennMultiApexHolonomyError(
                    "trace coefficient lost S3 invariance"
                )
            checks += 1
    return checks


def _matrix_row_sum_identity_checks() -> int:
    """Replay ``F*1=0`` on every matching triple and matrix row."""

    matching_permutations = tuple(
        _matching_permutation(matching, 6) for matching in perfect_matchings(6)
    )
    checks = 0
    for first, second, third in product(range(15), repeat=3):
        p0 = matching_permutations[first]
        p1 = matching_permutations[second]
        p2 = matching_permutations[third]
        for row in range(6):
            destination = p2[p1[p0[row]]]
            coefficient_sum = sum(
                int(destination == column) - int(row == column)
                for column in range(6)
            )
            if coefficient_sum:
                raise KrennMultiApexHolonomyError(
                    "degree-nine matrix row sum stopped vanishing"
                )
            checks += 1
    return checks


def _plucker_term(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial],
    root: int,
    neighbors: Sequence[int],
    *,
    matching_enumerator: MatchingEnumerator,
) -> tuple[LaurentPolynomial, LaurentPolynomial, LaurentPolynomial]:
    if d != 3 or len(neighbors) != 3:
        raise KrennMultiApexHolonomyError(
            "the bounded Pluecker gate is specific to d=3"
        )
    cofactors = tuple(
        _monochromatic_cofactor(
            n,
            d,
            weights,
            root,
            neighbors[color],
            color,
            matching_enumerator=matching_enumerator,
        )
        for color in range(3)
    )
    delta = cofactors[0] * cofactors[1] * cofactors[2]
    selected_star = tuple(
        tuple(
            _weight(
                weights,
                n,
                d,
                root,
                neighbors[neighbor_color],
                root_color,
                neighbor_color,
            )
            for root_color in range(3)
        )
        for neighbor_color in range(3)
    )
    determinant = laurent_determinant(selected_star)
    return delta, determinant, delta * determinant


@dataclass(frozen=True)
class ApexMinorRecord:
    root: int
    row_indices: tuple[int, ...]
    determinant: LaurentPolynomial

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "row_indices": list(self.row_indices),
            "determinant": self.determinant.to_expression(),
            "valuation": laurent_valuation(self.determinant),
        }


def _apex_minor_records(
    n: int,
    d: int,
    weights: Mapping[int, LaurentPolynomial],
    *,
    matching_enumerator: MatchingEnumerator,
) -> tuple[ApexMinorRecord, ...]:
    records = []
    for root in range(n):
        matrix = laurent_star_coefficient_matrix(
            n,
            d,
            weights,
            root,
            matching_enumerator=matching_enumerator,
        )
        row_indices = lexicographic_independent_rows(matrix)
        determinant = laurent_determinant(
            tuple(matrix[index] for index in row_indices)
        )
        records.append(ApexMinorRecord(root, row_indices, determinant))
    return tuple(records)


def _serialize_matrix(matrix: LaurentMatrix) -> list[list[str]]:
    return [
        [value.to_expression() for value in row] for row in matrix
    ]


@lru_cache(maxsize=1)
def _build_multi_apex_holonomy_gate() -> dict:
    """Build and internally replay the complete bounded exact gate."""

    k4_weights = _constant_laurent_weights_from_fixture()
    n6_weights = dict(natural_laurent_weight_entries())
    exponents = natural_stabilizer_exponents()
    if tuple(
        sum(exponents[vertex * 3 + color] for vertex in range(6))
        for color in range(3)
    ) != (0, 0, 0):
        raise KrennMultiApexHolonomyError(
            "known cocharacter left the GHZ stabilizer"
        )
    active_path_replay = certify_natural_stabilizer_orbit(n6_weights)
    victim = (0, 0, 2, 1, 2, 1)
    if (
        _coloring_character(victim, exponents, 3) != 1
        or any(
            _coloring_character((color,) * 6, exponents, 3)
            for color in range(3)
        )
    ):
        raise KrennMultiApexHolonomyError(
            "known target characters changed"
        )

    k4_primary = _apex_minor_records(
        4, 3, k4_weights, matching_enumerator=perfect_matchings
    )
    k4_independent = _apex_minor_records(
        4,
        3,
        k4_weights,
        matching_enumerator=independent_perfect_matchings,
    )
    n6_primary = _apex_minor_records(
        6, 3, n6_weights, matching_enumerator=perfect_matchings
    )
    n6_independent = _apex_minor_records(
        6,
        3,
        n6_weights,
        matching_enumerator=independent_perfect_matchings,
    )
    if k4_primary != k4_independent or n6_primary != n6_independent:
        raise KrennMultiApexHolonomyError(
            "independent apex-minor enumeration disagreed"
        )
    k4_minor_product = LAURENT_ONE
    for record in k4_primary:
        k4_minor_product *= record.determinant
    n6_minor_product = LAURENT_ONE
    for record in n6_primary:
        n6_minor_product *= record.determinant
    if k4_minor_product != LAURENT_ONE or n6_minor_product != LAURENT_ONE:
        raise KrennMultiApexHolonomyError(
            "deterministic apex-minor product stopped balancing"
        )

    transition_counts = {}
    for label, n, weights in (
        ("k4", 4, k4_weights),
        ("n6_laurent", 6, n6_weights),
    ):
        zero = 0
        total = 0
        for target_root in range(n):
            for source_root in range(n):
                if target_root == source_root:
                    continue
                primary = cross_apex_transition(
                    n,
                    3,
                    weights,
                    target_root,
                    source_root,
                    matching_enumerator=perfect_matchings,
                )
                independent = cross_apex_transition(
                    n,
                    3,
                    weights,
                    target_root,
                    source_root,
                    matching_enumerator=independent_perfect_matchings,
                )
                if primary != independent:
                    raise KrennMultiApexHolonomyError(
                        "independent cross-apex transition disagreed"
                    )
                total += 1
                zero += all(value.is_zero for row in primary for value in row)
        if zero != total:
            raise KrennMultiApexHolonomyError(
                "a sparse benchmark cross-apex transition became nonzero"
            )
        transition_counts[label] = {
            "ordered_off_diagonal_transitions": total,
            "identically_zero_transitions": zero,
        }

    plucker_records: dict[str, list[dict]] = {}
    for label, n, weights in (
        ("k4", 4, k4_weights),
        ("n6_laurent", 6, n6_weights),
    ):
        records = []
        for root in range(n):
            terms = []
            for neighbors in product(
                tuple(vertex for vertex in range(n) if vertex != root),
                repeat=3,
            ):
                primary = _plucker_term(
                    n,
                    3,
                    weights,
                    root,
                    neighbors,
                    matching_enumerator=perfect_matchings,
                )
                independent = _plucker_term(
                    n,
                    3,
                    weights,
                    root,
                    neighbors,
                    matching_enumerator=independent_perfect_matchings,
                )
                if primary != independent:
                    raise KrennMultiApexHolonomyError(
                        "independent Pluecker term disagreed"
                    )
                if not primary[2].is_zero:
                    terms.append((neighbors, *primary))
            if len(terms) != 1 or terms[0][3] != LAURENT_ONE:
                raise KrennMultiApexHolonomyError(
                    "benchmark lost its unique unit Pluecker term"
                )
            neighbors, delta, determinant, invariant = terms[0]
            records.append(
                {
                    "root": root,
                    "pivot_neighbors_by_color": list(neighbors),
                    "cofactor_product": delta.to_expression(),
                    "cofactor_product_valuation": laurent_valuation(delta),
                    "star_determinant": determinant.to_expression(),
                    "star_determinant_valuation": laurent_valuation(
                        determinant
                    ),
                    "invariant_pairing": invariant.to_expression(),
                }
            )
        plucker_records[label] = records

    marginal_records = {}
    for label, n, weights in (
        ("k4", 4, k4_weights),
        ("n6_laurent", 6, n6_weights),
    ):
        marginals = tuple(
            monochromatic_marginal_matrix(n, 3, weights, color)
            for color in range(3)
        )
        independent_marginals = tuple(
            monochromatic_marginal_matrix(
                n,
                3,
                weights,
                color,
                matching_enumerator=independent_perfect_matchings,
            )
            for color in range(3)
        )
        if marginals != independent_marginals:
            raise KrennMultiApexHolonomyError(
                "independent marginal enumeration disagreed"
            )
        holonomy = _matrix_product(
            _matrix_product(marginals[0], marginals[1]),
            marginals[2],
        )
        permutation = _permutation_from_unit_matrix(holonomy)
        marginal_records[label] = {
            "holonomy_matrix": _serialize_matrix(holonomy),
            "permutation": list(permutation),
            "cycles": [list(cycle) for cycle in _cycles(permutation)],
            "trace": _matrix_trace(holonomy).to_expression(),
            "determinant": laurent_determinant(holonomy).to_expression(),
        }
    if (
        marginal_records["k4"]["permutation"] != [0, 1, 2, 3]
        or marginal_records["n6_laurent"]["permutation"]
        != [2, 4, 1, 3, 0, 5]
    ):
        raise KrennMultiApexHolonomyError(
            "benchmark marginal holonomy changed"
        )

    flat_matrices = abstract_flat_marginal_counterexample()
    rational_identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(6))
        for row in range(6)
    )
    if _fraction_matrix_product(
        _fraction_matrix_product(flat_matrices[0], flat_matrices[1]),
        flat_matrices[2],
    ) != rational_identity:
        raise KrennMultiApexHolonomyError(
            "abstract n=6 flat marginal example stopped being flat"
        )
    for matrix in flat_matrices:
        if (
            matrix != tuple(tuple(row) for row in zip(*matrix, strict=True))
            or any(matrix[index][index] for index in range(6))
            or tuple(sum(row) for row in matrix) != (Fraction(1),) * 6
        ):
            raise KrennMultiApexHolonomyError(
                "abstract flat marginal axioms changed"
            )

    reynolds = certify_reynolds_reduction()
    dual = exact_two_row_dual()
    trace_coefficients = tuple(
        trace_holonomy_coefficient(key) for key in dual.row_keys
    )
    trace_rhs = tuple(
        coefficient * orbit_size
        for coefficient, orbit_size in zip(
            trace_coefficients,
            dual.row_orbit_sizes,
            strict=True,
        )
    )
    trace_pairing = sum(
        coefficient * rhs
        for coefficient, rhs in zip(
            dual.integer_lambda, trace_rhs, strict=True
        )
    )
    if (
        trace_coefficients != (-6, -4)
        or trace_rhs != (-2_160, -4_320)
        or trace_pairing != 2_160
        or not dual.lambda_transpose_B_zero
        or not reynolds.invariant_search_lossless_over_Q
    ):
        raise KrennMultiApexHolonomyError(
            "trace-holonomy exact dual replay changed"
        )
    histogram = _trace_coefficient_histogram()
    if histogram != {-6: 1_845, -4: 1_440, -2: 90}:
        raise KrennMultiApexHolonomyError(
            "trace-holonomy coefficient histogram changed"
        )
    invariance_checks = _trace_invariance_checks()
    if invariance_checks != 3_375 * 7:
        raise KrennMultiApexHolonomyError(
            "trace-holonomy invariance census changed"
        )
    row_sum_checks = _matrix_row_sum_identity_checks()
    if row_sum_checks != 3_375 * 6:
        raise KrennMultiApexHolonomyError(
            "degree-nine matrix row-sum census changed"
        )

    claim_boundary = {
        "affine_GHZ_membership_decided_by_this_gate": False,
        "counterexample_found": False,
        "cross_apex_zero_is_global_identity": False,
        "deterministic_minor_product_is_chart_independent_invariant": False,
        "full_GHZ_stabilizer_invariant_detects_known_path_divergence": False,
        "marginal_holonomy_alone_excludes_n6": False,
        "proposed_degree9_matrix_identity_in_J_mix": False,
        "proposed_degree9_matrix_identity_in_radical_J_mix_decided": False,
        "strict_border_membership_reproved_here": False,
        "trace_obstruction_is_characteristic_zero_exact": True,
    }
    return {
        "schema": HOLONOMY_GATE_SCHEMA,
        "scope": {
            "benchmarks": ["n4_d3_exact_fixture", "n6_d3_known_Laurent_path"],
            "cpu_workers": 1,
            "floating_point_used": False,
            "matching_enumerators": ["primary", "independent"],
            "random_seeds": [],
            "bounded_census": {
                "apex_minors": 10,
                "cross_apex_transitions": 42,
                "maximum_determinant_size": 15,
                "matching_triples": 3_375,
                "S6_x_S3_generator_checks": 23_625,
            },
        },
        "stabilizer_orbit": {
            "cocharacter_by_vertex_color": list(exponents),
            "color_sums": [0, 0, 0],
            "active_weight_index_character_valuation": [
                list(record) for record in active_path_replay
            ],
            "constant_GHZ_characters": [0, 0, 0],
            "victim_coloring": list(victim),
            "victim_equation": N6_D3_SEED_DEFECT_EQUATION,
            "victim_character": 1,
            "parameter_domain": "G_m (t != 0)",
            "known_path_is_exact_full_GHZ_stabilizer_orbit": True,
            (
                "regular_or_rational_full_stabilizer_invariants_are_"
                "constant_where_defined_on_t_nonzero"
            ): True,
        },
        "chart_free_transition": transition_counts,
        "deterministic_apex_minors": {
            "selection": (
                "lexicographically first independent rows at t=1; "
                "this is a deterministic diagnostic, not a chart-free invariant"
            ),
            "k4": [record.to_dict() for record in k4_primary],
            "k4_product": k4_minor_product.to_expression(),
            "n6_laurent": [record.to_dict() for record in n6_primary],
            "n6_valuation_sum": sum(
                laurent_valuation(record.determinant)
                for record in n6_primary
            ),
            "n6_product": n6_minor_product.to_expression(),
        },
        "monochromatic_plucker_pairing": plucker_records,
        "monochromatic_marginal_holonomy": marginal_records,
        "abstract_n6_flat_marginal_counterexample": {
            "matrices": [
                [
                    [_fraction_text(value) for value in row]
                    for row in matrix
                ]
                for matrix in flat_matrices
            ],
            "symmetric": True,
            "zero_diagonal": True,
            "row_stochastic": True,
            "ordered_product_is_identity": True,
            "meaning": (
                "the tested abstract axioms (symmetric, zero diagonal, "
                "row stochastic, flat ordered product) have exact n=6 "
                "solutions; shared Krenn-weight realizability is not claimed"
            ),
        },
        "degree9_mixed_ideal_gate": {
            "polynomial": "trace(C0*C1*C2)-6*A0*A1*A2",
            "ordinary_degree": 9,
            "fine_degree": [1] * 18,
            "coefficient_histogram": {
                str(key): value for key, value in histogram.items()
            },
            "S6_x_S3_invariance_checks": invariance_checks,
            "F_times_ones_zero_checks": row_sum_checks,
            "dual_row_keys": [list(key) for key in dual.row_keys],
            "dual_row_orbit_sizes": list(dual.row_orbit_sizes),
            "dual_integer_lambda": list(dual.integer_lambda),
            "dual_annihilates_full_compressed_source_map": (
                dual.lambda_transpose_B_zero
            ),
            "reynolds_group_order": reynolds.group_order,
            "reynolds_invariant_search_lossless_over_Q": (
                reynolds.invariant_search_lossless_over_Q
            ),
            "raw_source_map_annihilation_follows_by_reynolds_lift": True,
            "trace_coefficients_on_dual_rows": list(trace_coefficients),
            "trace_rhs_on_dual_rows": list(trace_rhs),
            "dual_pairing": trace_pairing,
            "trace_polynomial_in_J_mix_over_Q": False,
            "every_matrix_entry_in_J_mix_over_Q": False,
            "entry_corollary": (
                "S6 is transitive on diagonal and ordered off-diagonal "
                "entries; F*1=0, so membership of any entry would force "
                "membership of trace(F), contradicting the exact dual"
            ),
            "radical_membership_decided": False,
        },
        "conclusion": {
            "classification": "exact-negative-holonomy-preflight",
            "next_required_data": (
                "a continuation in this framework would need mixed-color "
                "cofactors or vanishing-order ratios beyond the tested "
                "monochromatic BY=I candidates"
            ),
            "stop_tested_holonomy_candidates": True,
        },
        "claim_boundary": claim_boundary,
    }


def build_multi_apex_holonomy_gate() -> dict:
    """Return an isolated copy of the cached exact certificate."""

    return deepcopy(_build_multi_apex_holonomy_gate())


def _canonical_json(payload: Mapping) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def verify_multi_apex_holonomy_gate(payload: Mapping) -> dict:
    """Recompute the exact gate and reject any changed claim or datum."""

    if not isinstance(payload, Mapping):
        raise KrennMultiApexHolonomyError(
            "holonomy certificate must be a mapping"
        )
    expected = build_multi_apex_holonomy_gate()
    if dict(payload) != expected:
        raise KrennMultiApexHolonomyError(
            "holonomy certificate differs from exact replay"
        )
    return expected


def _readme_text(certificate: Mapping) -> str:
    n6_minors = ", ".join(
        record["determinant"]
        for record in certificate["deterministic_apex_minors"]["n6_laurent"]
    )
    return f"""# Exact multi-apex holonomy gate

This bundle records a bounded stopping test for the first simultaneous-star
holonomy proposal.  It is an exact negative preflight, not a Krenn--Gu
existence or nonexistence theorem.

## Exact outcome

The punctured Laurent family (`t != 0`) is exactly the full color-diagonal
GHZ-stabilizer one-parameter orbit with

```text
x_(0,0) = +1
x_(2,0) = -1
all other x_(v,a) = 0.
```

It sends the natural seed to

```text
w_01^(00) = t
w_23^(00) = t^-1
```

and gives the victim coloring `002121` character `+1`.  Consequently every
regular or rational invariant of that full stabilizer is constant wherever
it is defined for `t != 0`.  The previously known pole monomial is not an
invariant of this larger group.

The chart-free overlaps `B_r Y_s`, `r != s`, vanish identically on both the
`K4,d=3` witness and the entire natural `n=6,d=3` Laurent family.  Their
support consists of one monochromatic perfect matching per color, and the
matching partner required by `B_r` cannot simultaneously be the partner
required by `Y_s`.

The deterministic full-rank star minors along the `n=6` path are

```text
{n6_minors}
```

Their valuations sum to zero and their product is exactly one.  These minors
use a declared lexicographic row chart; their product is a diagnostic, not a
claimed chart-independent invariant.

The gauge-invariant monochromatic Cauchy--Binet term at every apex is also
exactly one.  The marginal product `C^0 C^1 C^2` is `I_4` for `K4`; for the
natural `n=6` family it is the constant permutation

```text
[2, 4, 1, 3, 0, 5]
```

with cycles `(0 2 1 4)(3)(5)`, trace `2`, and determinant `-1`.
This separates the two sparse seeds but does not see Laurent divergence.
The certificate includes exact rational symmetric, zero-diagonal,
row-stochastic `6 x 6` matrices whose ordered product is the identity.  Thus
those four tested abstract marginal axioms alone have no six-vertex exclusion
power; realization by three blocks of one shared Krenn weighting is not
claimed.

## First mixed-equation repair

Let `A_a` be the all-color-`a` matching amplitude and

```text
C^a_rs = w_rs^(aa) H_(V minus {{r,s}})(a,...,a).
```

The homogeneous candidate tested here is

```text
F = C^0 C^1 C^2 - A_0 A_1 A_2 I_6.
```

It does not admit a degree-nine mixed-ideal identity.  For the invariant
trace `T=trace(F)`, the existing exact two-row dual annihilates the compressed
fine-graded source map and pairs to `2160`; Reynolds averaging over
`S6 x S3` lifts that obstruction losslessly to the raw source map over `Q`.
Therefore `T` is not in `J_mix` over `Q`.  Vertex symmetry and `F*1=0` then
imply that no individual entry `F_rs` lies in `J_mix`.

This does **not** decide membership in `radical(J_mix)`.  A higher-power
set-theoretic identity could still exist.

## Reproduction

```powershell
Set-Location 'C:\\Users\\brssn\\Projects\\Krenn-counterexample-search'

python -B -m unittest -v tests.test_krenn_multi_apex_holonomy

python -B -m experiments.krenn_quantum_graph.multi_apex_holonomy `
  --results-directory `
  results\\krenn_quantum_graph\\n6_d3_multi_apex_holonomy_gate `
  --verify-only
```

The computation is deterministic, exact over `Q[t,t^-1]`, serial, and uses
both native perfect-matching enumerators.  No floating point, cloud compute,
or large scratch cache is used.

## Claim boundary

The gate says to stop the tested cross-apex overlap, deterministic-minor,
monochromatic Pluecker/marginal-product, and degree-nine ideal candidates.
It neither finds a finite witness nor proves a new nonexistence result.
A continuation intended to distinguish this Laurent path in the same
framework would need mixed-color cofactor or vanishing-order data beyond the
`B_r X_r = I_3` normal form.
"""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        raise KrennMultiApexHolonomyError(
            f"holonomy manifest dependency is missing: {path}"
        )
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _expected_manifest(directory: Path, certificate: Mapping) -> dict:
    repository = Path(__file__).resolve().parents[2]
    return {
        "schema": HOLONOMY_MANIFEST_SCHEMA,
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


def write_multi_apex_holonomy_bundle(directory: Path) -> dict:
    """Write the small replayable certificate, README, and manifest."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    certificate = build_multi_apex_holonomy_gate()
    certificate_path = directory / "certificate.json"
    readme_path = directory / "README.md"
    certificate_path.write_text(
        _canonical_json(certificate), encoding="utf-8", newline="\n"
    )
    readme_path.write_text(
        _readme_text(certificate), encoding="utf-8", newline="\n"
    )

    manifest = _expected_manifest(directory, certificate)
    (directory / "manifest.json").write_text(
        _canonical_json(manifest), encoding="utf-8", newline="\n"
    )
    return manifest


def verify_multi_apex_holonomy_bundle(directory: Path) -> dict:
    """Verify hashes and exact content of a committed result bundle."""

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
        raise KrennMultiApexHolonomyError(
            "holonomy bundle cannot be read"
        ) from error
    expected_certificate = verify_multi_apex_holonomy_gate(certificate)
    if certificate_text != _canonical_json(expected_certificate):
        raise KrennMultiApexHolonomyError(
            "holonomy certificate serialization changed"
        )
    if readme_text != _readme_text(expected_certificate):
        raise KrennMultiApexHolonomyError(
            "holonomy README differs from exact regeneration"
        )
    expected_manifest = _expected_manifest(directory, expected_certificate)
    if manifest != expected_manifest or manifest_text != _canonical_json(
        expected_manifest
    ):
        raise KrennMultiApexHolonomyError(
            "holonomy manifest differs from exact regeneration"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the exact multi-apex holonomy gate."
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
    )
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.verify_only:
        verify_multi_apex_holonomy_bundle(arguments.results_directory)
    else:
        write_multi_apex_holonomy_bundle(arguments.results_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
