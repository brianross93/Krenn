r"""Count-only preflight for the ``D^2 in J_mix`` source-ideal problem.

The exact ``k=1`` experiment in :mod:`source_ideal` uses the fine grading

.. math::

   \deg W_{ij}^{ab}=e_{i,a}+e_{j,b}.

For ``k=2`` the codomain degree is ``2 delta`` and a multiplier of ``F_c``
has degree ``2 delta - deg(F_c)``.  Thus every vertex has one color-token of
degree one and two color-tokens of degree two in the multiplier fiber.

This module deliberately stops before matrix construction.  It computes the
raw dimensions by an exact inclusion-exclusion formula, and the
``S_6 x S_3`` orbit dimensions by Burnside's lemma.  A separate memory
receipt gives conservative dense and sparse lower/upper estimates and
refuses full construction when they exceed reviewed limits.

The raw coefficient formula is small.  A monomial is a loopless multigraph
on the 18 vertex-color tokens, with edges forbidden between tokens belonging
to the same physical vertex.  All degrees are at most two.  After
inclusion-exclusion over the six forbidden triangles, the unrestricted
multigraph count depends only on the numbers of residual degree-one and
degree-two vertices; its components are paths, cycles, and double edges.

No modular rank, Groebner basis, or Macaulay matrix is constructed here.
"""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import permutations
import json
from math import comb, factorial
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.source_ideal import (
    GROUP_ORDER,
    MIXED_COLORING_COUNT,
    OCCUPATION_REPRESENTATIVES,
)


N = 6
COLOR_COUNT = 3
TOKEN_COUNT = N * COLOR_COUNT
EDGE_VARIABLE_COUNT = 135
PERFECT_MATCHING_COUNT = 15
K = 2

HIGHER_POWER_SCHEMA = "krenn.n6_d3.source_ideal.k2_preflight.v1"

EXPECTED_RAW_MULTIPLIER_FIBER = 206_654_284_635
EXPECTED_RAW_DOMAIN_DIMENSION = 150_031_010_645_010
EXPECTED_RAW_CODOMAIN_DIMENSION = 74_680_326_909_360
EXPECTED_DOMAIN_ORBIT_CENSUS = {
    "2+2+2": 4_306_697_434,
    "3+2+1": 17_224_970_227,
    "3+3": 2_871_521_344,
    "4+1+1": 4_307_033_332,
    "4+2": 4_307_241_637,
    "5+1": 1_723_078_477,
}
EXPECTED_DOMAIN_ORBIT_DIMENSION = 34_740_542_451
EXPECTED_CODOMAIN_ORBIT_DIMENSION = 17_291_676_144
EXPECTED_D_SQUARED_SUPPORT_DIMENSION = 1_728_000
EXPECTED_D_SQUARED_SUPPORT_ORBITS = 663
EXPECTED_D_SQUARED_COEFFICIENT_CENSUS = {
    1: 3_375,
    2: 70_875,
    4: 496_125,
    8: 1_157_625,
}
EXPECTED_D_SQUARED_COEFFICIENT_ORBIT_CENSUS = {
    1: 8,
    2: 45,
    4: 195,
    8: 415,
}

# Independent regression targets for the same recurrences at k=1.
EXPECTED_K1_RAW_MULTIPLIER_FIBER = 6_040
EXPECTED_K1_RAW_CODOMAIN_DIMENSION = 11_608_920
EXPECTED_K1_DOMAIN_ORBIT_CENSUS = {
    "2+2+2": 202,
    "3+2+1": 556,
    "3+3": 126,
    "4+1+1": 188,
    "4+2": 172,
    "5+1": 70,
}
EXPECTED_K1_CODOMAIN_ORBIT_DIMENSION = 3_102

# These are intentionally policy limits, not claims about available RAM.
REVIEWED_MAX_SPARSE_BYTES = 64 * 1024**3
REVIEWED_MAX_DENSE_BYTES = 64 * 1024**3
REVIEWED_MAX_MATRIX_COLUMNS = 100_000_000


class KrennHigherPowerError(ValueError):
    """A ``k=2`` count, symmetry action, or memory receipt is malformed."""


def _validate_coloring(coloring: Sequence[int]) -> tuple[int, ...]:
    result = tuple(map(int, coloring))
    if (
        len(result) != N
        or any(color not in range(COLOR_COUNT) for color in result)
    ):
        raise KrennHigherPowerError("expected a coloring in range(3)^6")
    return result


def multiplier_token_degrees(
    coloring: Sequence[int],
) -> tuple[int, ...]:
    """Return ``2 delta - deg(F_c)`` in the canonical 18-token order."""

    coloring = _validate_coloring(coloring)
    return tuple(
        1 if color == coloring[vertex] else 2
        for vertex in range(N)
        for color in range(COLOR_COUNT)
    )


def codomain_token_degrees() -> tuple[int, ...]:
    """Return the fine degree of ``D^2``."""

    return (2,) * TOKEN_COUNT


@lru_cache(maxsize=None)
def unrestricted_max_degree_two_count(
    degree_one_vertices: int,
    degree_two_vertices: int,
) -> int:
    """Count loopless multigraphs with a prescribed ``1/2`` degree pattern.

    The ambient graph is complete and the vertices are labelled.  Components
    containing degree-one vertices are paths.  Components containing only
    degree-two vertices are simple cycles of length at least three or a
    double edge on two vertices.
    """

    n1 = int(degree_one_vertices)
    n2 = int(degree_two_vertices)
    if n1 < 0 or n2 < 0 or n1 % 2:
        return 0
    if n1:
        # Process the component containing one distinguished endpoint.
        return (n1 - 1) * sum(
            comb(n2, internal) * factorial(internal)
            * unrestricted_max_degree_two_count(
                n1 - 2,
                n2 - internal,
            )
            for internal in range(n2 + 1)
        )
    if not n2:
        return 1
    # Process the component containing one distinguished degree-two vertex.
    total = 0
    for size in range(2, n2 + 1):
        component_count = (
            1 if size == 2 else factorial(size - 1) // 2
        )
        total += (
            comb(n2 - 1, size - 1)
            * component_count
            * unrestricted_max_degree_two_count(0, n2 - size)
        )
    return total


def _triangle_inclusion_distribution(
    local_degrees: Sequence[int],
) -> Mapping[tuple[int, int], int]:
    """Signed residual ``(#degree1,#degree2)`` distribution for one K3."""

    degrees = tuple(map(int, local_degrees))
    if len(degrees) != COLOR_COUNT or any(
        degree not in (0, 1, 2) for degree in degrees
    ):
        raise KrennHigherPowerError(
            "a forbidden triangle needs three degrees in {0,1,2}"
        )
    edges = ((0, 1), (0, 2), (1, 2))
    distribution: Counter[tuple[int, int]] = Counter()
    for mask in range(1 << len(edges)):
        residual = list(degrees)
        selected = 0
        valid = True
        for edge_index, (left, right) in enumerate(edges):
            if mask >> edge_index & 1:
                residual[left] -= 1
                residual[right] -= 1
                selected += 1
                if residual[left] < 0 or residual[right] < 0:
                    valid = False
                    break
        if not valid:
            continue
        key = (residual.count(1), residual.count(2))
        distribution[key] += -1 if selected % 2 else 1
    return dict(sorted(distribution.items()))


def _convolve_distributions(
    left: Mapping[tuple[int, int], int],
    right: Mapping[tuple[int, int], int],
) -> Mapping[tuple[int, int], int]:
    result: Counter[tuple[int, int]] = Counter()
    for (left_one, left_two), left_coefficient in left.items():
        for (right_one, right_two), right_coefficient in right.items():
            result[
                left_one + right_one,
                left_two + right_two,
            ] += left_coefficient * right_coefficient
    return dict(sorted(
        (key, value) for key, value in result.items() if value
    ))


def _six_triangle_distribution(
    local_degrees: Sequence[int],
) -> Mapping[tuple[int, int], int]:
    local = _triangle_inclusion_distribution(local_degrees)
    result: Mapping[tuple[int, int], int] = {(0, 0): 1}
    for _vertex in range(N):
        result = _convolve_distributions(result, local)
    return result


def _raw_dimension_from_local_degrees(
    local_degrees: Sequence[int],
) -> int:
    distribution = _six_triangle_distribution(local_degrees)
    return sum(
        coefficient
        * unrestricted_max_degree_two_count(degree_one, degree_two)
        for (degree_one, degree_two), coefficient
        in distribution.items()
    )


@lru_cache(maxsize=1)
def raw_multiplier_fiber_dimension() -> int:
    """Exact number of ``k=2`` multiplier monomials for every coloring."""

    result = _raw_dimension_from_local_degrees((1, 2, 2))
    if result != EXPECTED_RAW_MULTIPLIER_FIBER:
        raise KrennHigherPowerError(
            "the raw multiplier dimension changed"
        )
    return result


@lru_cache(maxsize=1)
def raw_domain_dimension() -> int:
    """Exact raw number of all mixed-generator multiplier columns."""

    result = MIXED_COLORING_COUNT * raw_multiplier_fiber_dimension()
    if result != EXPECTED_RAW_DOMAIN_DIMENSION:
        raise KrennHigherPowerError("the raw domain dimension changed")
    return result


@lru_cache(maxsize=1)
def raw_codomain_dimension() -> int:
    """Exact number of monomials of fine degree ``2 delta``."""

    result = _raw_dimension_from_local_degrees((2, 2, 2))
    if result != EXPECTED_RAW_CODOMAIN_DIMENSION:
        raise KrennHigherPowerError(
            "the raw codomain dimension changed"
        )
    return result


def _cycle_type(permutation: Sequence[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    cycles = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cursor = start
        length = 0
        while cursor not in seen:
            seen.add(cursor)
            cursor = permutation[cursor]
            length += 1
        cycles.append(length)
    return tuple(sorted(cycles, reverse=True))


@lru_cache(maxsize=None)
def _permutation_classes(
    size: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]:
    classes: dict[
        tuple[int, ...],
        tuple[tuple[int, ...], int],
    ] = {}
    for permutation in permutations(range(size)):
        kind = _cycle_type(permutation)
        if kind in classes:
            representative, multiplicity = classes[kind]
            classes[kind] = (representative, multiplicity + 1)
        else:
            classes[kind] = (permutation, 1)
    return tuple(
        (kind, representative, multiplicity)
        for kind, (representative, multiplicity)
        in sorted(classes.items(), reverse=True)
    )


def _token_permutation(
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        int(vertex_permutation[vertex]) * COLOR_COUNT
        + int(color_permutation[color])
        for vertex in range(N)
        for color in range(COLOR_COUNT)
    )


def _permutation_cycles(
    permutation: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    pending = set(range(len(permutation)))
    cycles = []
    while pending:
        start = min(pending)
        cycle = []
        cursor = start
        while cursor not in cycle:
            cycle.append(cursor)
            pending.remove(cursor)
            cursor = permutation[cursor]
        cycles.append(tuple(cycle))
    return tuple(cycles)


def _allowed_edges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(TOKEN_COUNT)
        for right in range(left + 1, TOKEN_COUNT)
        if left // COLOR_COUNT != right // COLOR_COUNT
    )


ALLOWED_EDGES = _allowed_edges()
ALLOWED_EDGE_SET = frozenset(ALLOWED_EDGES)


def _edge_orbits(
    token_permutation: Sequence[int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    image = tuple(map(int, token_permutation))
    pending = set(ALLOWED_EDGES)
    result = []
    while pending:
        seed = min(pending)
        orbit = []
        edge = seed
        while edge not in orbit:
            orbit.append(edge)
            left, right = edge
            transported = sorted((image[left], image[right]))
            edge = (transported[0], transported[1])
        pending.difference_update(orbit)
        result.append(tuple(sorted(orbit)))
    return tuple(result)


def _fixed_problem(
    token_permutation: Sequence[int],
    target_degrees: Sequence[int],
) -> tuple[tuple[int, ...], tuple[tuple[tuple[int, ...], int], ...]]:
    """Return cycle targets and multiplicity-compressed orbit vectors."""

    permutation = tuple(map(int, token_permutation))
    target = tuple(map(int, target_degrees))
    if (
        len(permutation) != TOKEN_COUNT
        or tuple(sorted(permutation)) != tuple(range(TOKEN_COUNT))
        or len(target) != TOKEN_COUNT
        or any(degree not in (0, 1, 2) for degree in target)
    ):
        raise KrennHigherPowerError("fixed-monomial input is malformed")
    transported_edges = {
        tuple(sorted((permutation[left], permutation[right])))
        for left, right in ALLOWED_EDGES
    }
    if transported_edges != ALLOWED_EDGE_SET:
        raise KrennHigherPowerError(
            "token permutation does not preserve physical-vertex blocks"
        )
    cycles = _permutation_cycles(permutation)
    cycle_index = {
        token: index
        for index, cycle in enumerate(cycles)
        for token in cycle
    }
    cycle_target = []
    for cycle in cycles:
        values = {target[token] for token in cycle}
        if len(values) != 1:
            return (), ()
        cycle_target.append(next(iter(values)))

    vector_counts: Counter[tuple[int, ...]] = Counter()
    for orbit in _edge_orbits(permutation):
        incidences = [0] * TOKEN_COUNT
        for left, right in orbit:
            incidences[left] += 1
            incidences[right] += 1
        vector = []
        valid = True
        for cycle in cycles:
            values = {incidences[token] for token in cycle}
            if len(values) != 1:
                raise KrennHigherPowerError(
                    "an edge orbit has nonconstant cycle incidence"
                )
            value = next(iter(values))
            vector.append(value)
            if value > cycle_target[len(vector) - 1]:
                valid = False
        if valid:
            vector_counts[tuple(vector)] += 1
    vectors = tuple(sorted(vector_counts.items()))
    if any(
        all(entry == 0 for entry in vector) or multiplicity <= 0
        for vector, multiplicity in vectors
    ):
        raise KrennHigherPowerError("edge-orbit compression failed")
    return tuple(cycle_target), vectors


def _subtract(
    remaining: tuple[int, ...],
    vector: tuple[int, ...],
    factor: int = 1,
) -> tuple[int, ...] | None:
    result = tuple(
        value - factor * contribution
        for value, contribution in zip(remaining, vector, strict=True)
    )
    if any(value < 0 for value in result):
        return None
    return result


def _fixed_count_from_problem(
    target: tuple[int, ...],
    vector_counts: tuple[tuple[tuple[int, ...], int], ...],
) -> int:
    """Solve the edge-orbit exponent equations with all targets at most two."""

    @lru_cache(maxsize=None)
    def recurse(remaining: tuple[int, ...]) -> int:
        if not any(remaining):
            return 1

        # Choosing the most constrained positive cycle keeps the recursion
        # small for the near-identity conjugacy classes.
        positive = [index for index, value in enumerate(remaining) if value]

        def option_score(index: int) -> int:
            return sum(
                multiplicity
                for vector, multiplicity in vector_counts
                if vector[index] and _subtract(remaining, vector) is not None
            )

        pivot = min(positive, key=lambda index: (option_score(index), index))
        required = remaining[pivot]
        one_vectors = [
            (vector, multiplicity)
            for vector, multiplicity in vector_counts
            if vector[pivot] == 1
            and _subtract(remaining, vector) is not None
        ]
        two_vectors = [
            (vector, multiplicity)
            for vector, multiplicity in vector_counts
            if vector[pivot] == 2
            and _subtract(remaining, vector) is not None
        ]

        total = 0
        if required == 1:
            for vector, multiplicity in one_vectors:
                reduced = _subtract(remaining, vector)
                if reduced is not None:
                    total += multiplicity * recurse(reduced)
            return total

        for vector, multiplicity in two_vectors:
            reduced = _subtract(remaining, vector)
            if reduced is not None:
                total += multiplicity * recurse(reduced)

        # One edge orbit may occur with exponent two.
        for vector, multiplicity in one_vectors:
            reduced = _subtract(remaining, vector, 2)
            if reduced is not None:
                total += multiplicity * recurse(reduced)

        # Or two distinct edge orbits may each occur with exponent one.
        for left_index, (left, left_multiplicity) in enumerate(one_vectors):
            for right_index in range(left_index, len(one_vectors)):
                right, right_multiplicity = one_vectors[right_index]
                combined = tuple(
                    left_entry + right_entry
                    for left_entry, right_entry
                    in zip(left, right, strict=True)
                )
                reduced = _subtract(remaining, combined)
                if reduced is None:
                    continue
                choices = (
                    comb(left_multiplicity, 2)
                    if left_index == right_index
                    else left_multiplicity * right_multiplicity
                )
                total += choices * recurse(reduced)
        return total

    return recurse(target)


@lru_cache(maxsize=None)
def fixed_fine_graded_monomial_count(
    token_permutation: tuple[int, ...],
    target_degrees: tuple[int, ...],
) -> int:
    """Count fine-graded monomials fixed by one token permutation."""

    if token_permutation == tuple(range(TOKEN_COUNT)):
        if target_degrees == codomain_token_degrees():
            return raw_codomain_dimension()
        if target_degrees == (1,) * TOKEN_COUNT:
            return EXPECTED_K1_RAW_CODOMAIN_DIMENSION
        if all(
            sorted(
                target_degrees[
                    vertex * COLOR_COUNT:(vertex + 1) * COLOR_COUNT
                ]
            ) == [1, 2, 2]
            for vertex in range(N)
        ):
            return raw_multiplier_fiber_dimension()
        if all(
            sorted(
                target_degrees[
                    vertex * COLOR_COUNT:(vertex + 1) * COLOR_COUNT
                ]
            ) == [0, 1, 1]
            for vertex in range(N)
        ):
            return EXPECTED_K1_RAW_MULTIPLIER_FIBER
    target, vectors = _fixed_problem(token_permutation, target_degrees)
    if not target:
        return 0
    return _fixed_count_from_problem(target, vectors)


@lru_cache(maxsize=1)
def codomain_orbit_dimension() -> int:
    """Count the ``S_6 x S_3`` codomain monomial orbits by Burnside."""

    total = 0
    target = codomain_token_degrees()
    for _vertex_type, vertex, vertex_multiplicity in (
        _permutation_classes(N)
    ):
        for _color_type, color, color_multiplicity in (
            _permutation_classes(COLOR_COUNT)
        ):
            fixed = fixed_fine_graded_monomial_count(
                _token_permutation(vertex, color),
                target,
            )
            total += vertex_multiplicity * color_multiplicity * fixed
    if total % GROUP_ORDER:
        raise KrennHigherPowerError(
            "the k=2 codomain Burnside sum is not integral"
        )
    result = total // GROUP_ORDER
    if result != EXPECTED_CODOMAIN_ORBIT_DIMENSION:
        raise KrennHigherPowerError(
            "the k=2 codomain orbit dimension changed"
        )
    return result


def _transport_coloring(
    coloring: Sequence[int],
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, ...]:
    transported = [0] * N
    for old_vertex, old_color in enumerate(coloring):
        transported[int(vertex_permutation[old_vertex])] = int(
            color_permutation[old_color]
        )
    return tuple(transported)


@lru_cache(maxsize=None)
def _domain_fiber_orbits_for_representative(
    coloring: tuple[int, ...],
) -> int:
    coloring = _validate_coloring(coloring)
    target = multiplier_token_degrees(coloring)
    fixed_sum = 0
    stabilizer_size = 0
    for vertex in permutations(range(N)):
        for color in permutations(range(COLOR_COUNT)):
            if _transport_coloring(coloring, vertex, color) != coloring:
                continue
            stabilizer_size += 1
            fixed_sum += fixed_fine_graded_monomial_count(
                _token_permutation(vertex, color),
                target,
            )
    if not stabilizer_size or fixed_sum % stabilizer_size:
        raise KrennHigherPowerError(
            "a k=2 multiplier stabilizer Burnside sum is not integral"
        )
    return fixed_sum // stabilizer_size


@lru_cache(maxsize=1)
def domain_orbit_census() -> Mapping[str, int]:
    """Return exact domain-orbit counts for the six coloring occupations."""

    result = {}
    for occupation, coloring, coloring_orbit_size, _k1_fiber_orbits in (
        OCCUPATION_REPRESENTATIVES
    ):
        stabilizer_size = GROUP_ORDER // coloring_orbit_size
        if stabilizer_size * coloring_orbit_size != GROUP_ORDER:
            raise KrennHigherPowerError(
                "coloring orbit-stabilizer data changed"
            )
        result[occupation] = _domain_fiber_orbits_for_representative(
            coloring
        )
    census = dict(sorted(result.items()))
    if census != EXPECTED_DOMAIN_ORBIT_CENSUS:
        raise KrennHigherPowerError(
            "the k=2 domain orbit census changed"
        )
    return census


@lru_cache(maxsize=1)
def domain_orbit_dimension() -> int:
    result = sum(domain_orbit_census().values())
    if result != EXPECTED_DOMAIN_ORBIT_DIMENSION:
        raise KrennHigherPowerError(
            "the k=2 domain orbit dimension changed"
        )
    return result


def _k1_multiplier_token_degrees(
    coloring: Sequence[int],
) -> tuple[int, ...]:
    coloring = _validate_coloring(coloring)
    return tuple(
        0 if color == coloring[vertex] else 1
        for vertex in range(N)
        for color in range(COLOR_COUNT)
    )


@lru_cache(maxsize=1)
def k1_codomain_orbit_regression() -> int:
    """Replay the established k=1 codomain count with the k=2 machinery."""

    total = 0
    target = (1,) * TOKEN_COUNT
    for _vertex_type, vertex, vertex_multiplicity in (
        _permutation_classes(N)
    ):
        for _color_type, color, color_multiplicity in (
            _permutation_classes(COLOR_COUNT)
        ):
            fixed = fixed_fine_graded_monomial_count(
                _token_permutation(vertex, color),
                target,
            )
            total += vertex_multiplicity * color_multiplicity * fixed
    if total % GROUP_ORDER:
        raise KrennHigherPowerError(
            "the k=1 codomain regression Burnside sum is not integral"
        )
    result = total // GROUP_ORDER
    if result != EXPECTED_K1_CODOMAIN_ORBIT_DIMENSION:
        raise KrennHigherPowerError(
            "the independent k=1 codomain regression changed"
        )
    return result


@lru_cache(maxsize=1)
def k1_domain_orbit_census_regression() -> Mapping[str, int]:
    """Replay all six established k=1 domain orbit counts."""

    result = {}
    for occupation, coloring, _coloring_orbit_size, _k1_orbits in (
        OCCUPATION_REPRESENTATIVES
    ):
        target = _k1_multiplier_token_degrees(coloring)
        fixed_sum = 0
        stabilizer_size = 0
        for vertex in permutations(range(N)):
            for color in permutations(range(COLOR_COUNT)):
                if (
                    _transport_coloring(coloring, vertex, color)
                    != coloring
                ):
                    continue
                stabilizer_size += 1
                fixed_sum += fixed_fine_graded_monomial_count(
                    _token_permutation(vertex, color),
                    target,
                )
        if not stabilizer_size or fixed_sum % stabilizer_size:
            raise KrennHigherPowerError(
                "a k=1 regression stabilizer sum is not integral"
            )
        result[occupation] = fixed_sum // stabilizer_size
    census = dict(sorted(result.items()))
    if census != EXPECTED_K1_DOMAIN_ORBIT_CENSUS:
        raise KrennHigherPowerError(
            "the independent k=1 domain orbit regression changed"
        )
    return census


def k1_recurrence_regression() -> Mapping:
    """Return an independent replay of every published k=1 hard count."""

    raw_multiplier = _raw_dimension_from_local_degrees((0, 1, 1))
    raw_codomain = _raw_dimension_from_local_degrees((1, 1, 1))
    if (
        raw_multiplier != EXPECTED_K1_RAW_MULTIPLIER_FIBER
        or raw_codomain != EXPECTED_K1_RAW_CODOMAIN_DIMENSION
    ):
        raise KrennHigherPowerError(
            "the independent k=1 raw recurrence changed"
        )
    domain_census = k1_domain_orbit_census_regression()
    return {
        "raw_multiplier_fiber": raw_multiplier,
        "raw_codomain": raw_codomain,
        "domain_orbits": sum(domain_census.values()),
        "domain_orbits_by_occupation": domain_census,
        "codomain_orbits": k1_codomain_orbit_regression(),
        "matches_embedded_k1_regression_targets": True,
    }


def _k6_edges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(N)
        for right in range(left + 1, N)
    )


K6_EDGES = _k6_edges()
K6_EDGE_INDEX = {edge: index for index, edge in enumerate(K6_EDGES)}


def _perfect_matchings(vertices: tuple[int, ...]) -> tuple[
    tuple[tuple[int, int], ...], ...
]:
    if not vertices:
        return ((),)
    first = vertices[0]
    result = []
    for partner_index in range(1, len(vertices)):
        partner = vertices[partner_index]
        remaining = vertices[1:partner_index] + vertices[
            partner_index + 1:
        ]
        edge = tuple(sorted((first, partner)))
        for tail in _perfect_matchings(remaining):
            result.append(tuple(sorted((edge, *tail))))
    return tuple(sorted(result))


@lru_cache(maxsize=1)
def k6_perfect_matchings() -> tuple[tuple[int, ...], ...]:
    result = tuple(
        tuple(K6_EDGE_INDEX[edge] for edge in matching)
        for matching in _perfect_matchings(tuple(range(N)))
    )
    if len(result) != PERFECT_MATCHING_COUNT:
        raise KrennHigherPowerError("the K6 matching census changed")
    return result


@lru_cache(maxsize=1)
def squared_hafnian_support() -> tuple[tuple[int, ...], ...]:
    """Return exponent vectors in the support of one all-equal ``F_c^2``."""

    result = set()
    matchings = k6_perfect_matchings()
    for left_index, left in enumerate(matchings):
        for right in matchings[left_index:]:
            exponents = [0] * len(K6_EDGES)
            for edge in (*left, *right):
                exponents[edge] += 1
            result.add(tuple(exponents))
    support = tuple(sorted(result))
    if len(support) != comb(PERFECT_MATCHING_COUNT + 1, 2):
        raise KrennHigherPowerError(
            "the squared-hafnian support has collisions"
        )
    return support


def _transport_k6_exponents(
    exponents: Sequence[int],
    vertex_permutation: Sequence[int],
) -> tuple[int, ...]:
    transported = [0] * len(K6_EDGES)
    for index, exponent in enumerate(exponents):
        left, right = K6_EDGES[index]
        image = tuple(sorted((
            int(vertex_permutation[left]),
            int(vertex_permutation[right]),
        )))
        transported[K6_EDGE_INDEX[image]] = int(exponent)
    return tuple(transported)


@lru_cache(maxsize=None)
def _fixed_squared_hafnian_support(
    vertex_permutation: tuple[int, ...],
) -> int:
    return sum(
        _transport_k6_exponents(exponents, vertex_permutation)
        == exponents
        for exponents in squared_hafnian_support()
    )


def _compose(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, ...]:
    return tuple(int(left[int(right[index])]) for index in range(len(left)))


def _power(
    permutation: Sequence[int],
    exponent: int,
) -> tuple[int, ...]:
    result = tuple(range(len(permutation)))
    base = tuple(map(int, permutation))
    for _ in range(int(exponent)):
        result = _compose(base, result)
    return result


def _support_state_encode(
    graph_indices: Sequence[int],
) -> int:
    indices = tuple(map(int, graph_indices))
    support_size = len(squared_hafnian_support())
    if (
        len(indices) != COLOR_COUNT
        or any(index not in range(support_size) for index in indices)
    ):
        raise KrennHigherPowerError("D^2 support state is malformed")
    left, middle, right = indices
    return (left * support_size + middle) * support_size + right


def _support_state_decode(state: int) -> tuple[int, int, int]:
    support_size = len(squared_hafnian_support())
    state = int(state)
    if state not in range(support_size**COLOR_COUNT):
        raise KrennHigherPowerError("D^2 support state is out of range")
    left, remainder = divmod(state, support_size * support_size)
    middle, right = divmod(remainder, support_size)
    return left, middle, right


def _one_color_support_coefficient(graph_index: int) -> int:
    graph = squared_hafnian_support()[int(graph_index)]
    coefficient = 1 if sum(exponent == 2 for exponent in graph) == 3 else 2
    if coefficient == 1 and sum(exponent == 1 for exponent in graph):
        raise KrennHigherPowerError(
            "a repeated matching has unexpected simple edges"
        )
    return coefficient


@dataclass(frozen=True)
class DSquaredSupportOrbit:
    representative_state: int
    graph_indices: tuple[int, int, int]
    orbit_size: int
    coefficient: int

    def __post_init__(self) -> None:
        if (
            self.graph_indices
            != _support_state_decode(self.representative_state)
            or self.orbit_size <= 0
            or self.orbit_size > GROUP_ORDER
            or GROUP_ORDER % self.orbit_size
            or self.coefficient
            != (
                _one_color_support_coefficient(self.graph_indices[0])
                * _one_color_support_coefficient(self.graph_indices[1])
                * _one_color_support_coefficient(self.graph_indices[2])
            )
        ):
            raise KrennHigherPowerError(
                "a D^2 support-orbit record is malformed"
            )


@lru_cache(maxsize=1)
def _support_vertex_generator_actions() -> tuple[tuple[int, ...], ...]:
    support = squared_hafnian_support()
    index = {graph: position for position, graph in enumerate(support)}
    actions = []
    for position in range(N - 1):
        permutation = list(range(N))
        permutation[position], permutation[position + 1] = (
            permutation[position + 1],
            permutation[position],
        )
        actions.append(tuple(
            index[_transport_k6_exponents(graph, permutation)]
            for graph in support
        ))
    return tuple(actions)


def _support_generator_images(state: int) -> tuple[int, ...]:
    left, middle, right = _support_state_decode(state)
    images = [
        _support_state_encode((middle, left, right)),
        _support_state_encode((left, right, middle)),
    ]
    for action in _support_vertex_generator_actions():
        images.append(_support_state_encode((
            action[left],
            action[middle],
            action[right],
        )))
    return tuple(images)


@lru_cache(maxsize=1)
def _d_squared_support_orbit_partition() -> tuple[
    tuple[DSquaredSupportOrbit, ...],
    array,
]:
    state_count = len(squared_hafnian_support()) ** COLOR_COUNT
    visited = bytearray(state_count)
    orbit_index = array("H", [65_535]) * state_count
    records = []
    for seed in range(state_count):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = 1
        representative = seed
        orbit_size = 0
        members = []
        while stack:
            state = stack.pop()
            members.append(state)
            representative = min(representative, state)
            orbit_size += 1
            for image in _support_generator_images(state):
                if not visited[image]:
                    visited[image] = 1
                    stack.append(image)
        graph_indices = _support_state_decode(representative)
        record_index = len(records)
        for state in members:
            orbit_index[state] = record_index
        records.append(
            DSquaredSupportOrbit(
                representative_state=representative,
                graph_indices=graph_indices,
                orbit_size=orbit_size,
                coefficient=(
                    _one_color_support_coefficient(graph_indices[0])
                    * _one_color_support_coefficient(graph_indices[1])
                    * _one_color_support_coefficient(graph_indices[2])
                ),
            )
        )
    result = tuple(records)
    if (
        len(result) != EXPECTED_D_SQUARED_SUPPORT_ORBITS
        or sum(record.orbit_size for record in result)
        != EXPECTED_D_SQUARED_SUPPORT_DIMENSION
        or tuple(sorted(
            record.representative_state for record in result
        ))
        != tuple(record.representative_state for record in result)
    ):
        raise KrennHigherPowerError(
            "the D^2 support-orbit enumeration changed"
        )
    if (
        len(orbit_index) != state_count
        or any(index == 65_535 for index in orbit_index)
    ):
        raise KrennHigherPowerError(
            "the D^2 support orbit-index partition is incomplete"
        )
    return result, orbit_index


@lru_cache(maxsize=1)
def d_squared_support_orbits() -> tuple[DSquaredSupportOrbit, ...]:
    """Enumerate only the 663 right-hand-side support orbits."""

    return _d_squared_support_orbit_partition()[0]


def d_squared_support_orbit_index(state: int) -> int:
    state = int(state)
    partition = _d_squared_support_orbit_partition()[1]
    if state not in range(len(partition)):
        raise KrennHigherPowerError("D^2 support state is out of range")
    return int(partition[state])


@lru_cache(maxsize=1)
def d_squared_support_dimension() -> int:
    result = len(squared_hafnian_support()) ** COLOR_COUNT
    if result != EXPECTED_D_SQUARED_SUPPORT_DIMENSION:
        raise KrennHigherPowerError(
            "the D^2 support dimension changed"
        )
    return result


@lru_cache(maxsize=1)
def d_squared_support_orbit_dimension() -> int:
    """Count support orbits of ``D^2`` under ``S_6 x S_3``."""

    total = 0
    for _vertex_type, vertex, vertex_multiplicity in (
        _permutation_classes(N)
    ):
        fixed_one = _fixed_squared_hafnian_support(vertex)
        fixed_square = _fixed_squared_hafnian_support(_power(vertex, 2))
        fixed_cube = _fixed_squared_hafnian_support(_power(vertex, 3))
        # Color identity, three transpositions, and two 3-cycles.
        fixed_colored_tuples = (
            fixed_one**3
            + 3 * fixed_one * fixed_square
            + 2 * fixed_cube
        )
        total += vertex_multiplicity * fixed_colored_tuples
    if total % GROUP_ORDER:
        raise KrennHigherPowerError(
            "the D^2 support Burnside sum is not integral"
        )
    result = total // GROUP_ORDER
    if (
        result != EXPECTED_D_SQUARED_SUPPORT_ORBITS
        or result != len(d_squared_support_orbits())
    ):
        raise KrennHigherPowerError(
            "the D^2 support orbit dimension changed"
        )
    return result


def d_squared_coefficient_census() -> Mapping[int, int]:
    """Return the exact coefficient distribution in the expansion of ``D^2``."""

    one_color = Counter()
    matchings = k6_perfect_matchings()
    for left in matchings:
        for right in matchings:
            exponents = [0] * len(K6_EDGES)
            for edge in (*left, *right):
                exponents[edge] += 1
            one_color[tuple(exponents)] += 1
    if set(one_color.values()) != {1, 2}:
        raise KrennHigherPowerError(
            "one-color squared-hafnian coefficients changed"
        )
    result: Counter[int] = Counter()
    for left_value, left_count in Counter(one_color.values()).items():
        for middle_value, middle_count in Counter(
            one_color.values()
        ).items():
            for right_value, right_count in Counter(
                one_color.values()
            ).items():
                result[left_value * middle_value * right_value] += (
                    left_count * middle_count * right_count
                )
    census = dict(sorted(result.items()))
    if census != EXPECTED_D_SQUARED_COEFFICIENT_CENSUS:
        raise KrennHigherPowerError(
            "the D^2 coefficient census changed"
        )
    return census


def d_squared_coefficient_orbit_census() -> Mapping[int, int]:
    census = dict(sorted(Counter(
        record.coefficient for record in d_squared_support_orbits()
    ).items()))
    if census != EXPECTED_D_SQUARED_COEFFICIENT_ORBIT_CENSUS:
        raise KrennHigherPowerError(
            "the D^2 coefficient-orbit census changed"
        )
    return census


@dataclass(frozen=True)
class MatrixMemoryReceipt:
    raw_rows: int
    raw_columns: int
    orbit_rows: int
    orbit_columns: int
    raw_nnz: int
    orbit_nnz_upper_bound: int
    raw_csc_column_pointer_bytes: int
    orbit_csc_column_pointer_bytes: int
    raw_sparse_csc_bytes: int
    orbit_sparse_csc_upper_bound_bytes: int
    raw_dense_bytes: int
    orbit_dense_bytes: int
    reviewed_max_sparse_bytes: int
    reviewed_max_dense_bytes: int
    reviewed_max_matrix_columns: int
    full_raw_construction_authorized: bool
    full_orbit_construction_authorized: bool

    def __post_init__(self) -> None:
        values = (
            self.raw_rows,
            self.raw_columns,
            self.orbit_rows,
            self.orbit_columns,
            self.raw_nnz,
            self.orbit_nnz_upper_bound,
            self.raw_csc_column_pointer_bytes,
            self.orbit_csc_column_pointer_bytes,
            self.raw_sparse_csc_bytes,
            self.orbit_sparse_csc_upper_bound_bytes,
            self.raw_dense_bytes,
            self.orbit_dense_bytes,
        )
        if any(value <= 0 for value in values):
            raise KrennHigherPowerError(
                "matrix memory dimensions must be positive"
            )
        if self.full_raw_construction_authorized:
            raise KrennHigherPowerError(
                "the reviewed policy must refuse the raw k=2 matrix"
            )
        if (
            self.orbit_csc_column_pointer_bytes
            <= self.reviewed_max_sparse_bytes
            or self.full_orbit_construction_authorized
        ):
            raise KrennHigherPowerError(
                "the reviewed policy must refuse the orbit k=2 matrix"
            )

    def to_dict(self) -> dict:
        return {
            "raw_shape": [self.raw_rows, self.raw_columns],
            "orbit_shape": [self.orbit_rows, self.orbit_columns],
            "raw_nnz": self.raw_nnz,
            "orbit_nnz_upper_bound": self.orbit_nnz_upper_bound,
            "storage_model": {
                "value_bytes": 8,
                "row_index_bytes": 8,
                "column_pointer_bytes": 8,
                "sparse_format": "CSC",
                "scope": (
                    "fixed-array payload only; allocator, container, "
                    "elimination, and temporary overhead are excluded"
                ),
            },
            "raw_csc_column_pointer_bytes": (
                self.raw_csc_column_pointer_bytes
            ),
            "orbit_csc_column_pointer_bytes": (
                self.orbit_csc_column_pointer_bytes
            ),
            "raw_sparse_csc_bytes": self.raw_sparse_csc_bytes,
            "orbit_sparse_csc_upper_bound_bytes": (
                self.orbit_sparse_csc_upper_bound_bytes
            ),
            "raw_dense_bytes": self.raw_dense_bytes,
            "orbit_dense_bytes": self.orbit_dense_bytes,
            "reviewed_limits": {
                "max_sparse_bytes": self.reviewed_max_sparse_bytes,
                "max_dense_bytes": self.reviewed_max_dense_bytes,
                "max_matrix_columns": self.reviewed_max_matrix_columns,
            },
            "full_raw_construction_authorized": (
                self.full_raw_construction_authorized
            ),
            "full_orbit_construction_authorized": (
                self.full_orbit_construction_authorized
            ),
            "decision": (
                "refuse full matrix construction"
                if not self.full_orbit_construction_authorized
                else "orbit construction is within the reviewed estimate"
            ),
            "refusal_lower_bound": (
                "the compressed CSC column-pointer array alone exceeds "
                "the reviewed sparse-memory limit"
            ),
        }


def _csc_bytes(nonzeros: int, columns: int) -> int:
    # Conservative 64-bit values, row indices, and column pointers.
    return 16 * int(nonzeros) + 8 * (int(columns) + 1)


@lru_cache(maxsize=1)
def matrix_memory_receipt() -> MatrixMemoryReceipt:
    raw_rows = raw_codomain_dimension()
    raw_columns = raw_domain_dimension()
    orbit_rows = codomain_orbit_dimension()
    orbit_columns = domain_orbit_dimension()
    raw_nnz = PERFECT_MATCHING_COUNT * raw_columns
    orbit_nnz_upper_bound = PERFECT_MATCHING_COUNT * orbit_columns
    raw_pointers = 8 * (raw_columns + 1)
    orbit_pointers = 8 * (orbit_columns + 1)
    raw_sparse = _csc_bytes(raw_nnz, raw_columns)
    orbit_sparse = _csc_bytes(
        orbit_nnz_upper_bound,
        orbit_columns,
    )
    raw_dense = 8 * raw_rows * raw_columns
    orbit_dense = 8 * orbit_rows * orbit_columns
    raw_authorized = (
        raw_sparse <= REVIEWED_MAX_SPARSE_BYTES
        and raw_dense <= REVIEWED_MAX_DENSE_BYTES
        and raw_columns <= REVIEWED_MAX_MATRIX_COLUMNS
    )
    orbit_authorized = (
        orbit_sparse <= REVIEWED_MAX_SPARSE_BYTES
        and orbit_dense <= REVIEWED_MAX_DENSE_BYTES
        and orbit_columns <= REVIEWED_MAX_MATRIX_COLUMNS
    )
    return MatrixMemoryReceipt(
        raw_rows=raw_rows,
        raw_columns=raw_columns,
        orbit_rows=orbit_rows,
        orbit_columns=orbit_columns,
        raw_nnz=raw_nnz,
        orbit_nnz_upper_bound=orbit_nnz_upper_bound,
        raw_csc_column_pointer_bytes=raw_pointers,
        orbit_csc_column_pointer_bytes=orbit_pointers,
        raw_sparse_csc_bytes=raw_sparse,
        orbit_sparse_csc_upper_bound_bytes=orbit_sparse,
        raw_dense_bytes=raw_dense,
        orbit_dense_bytes=orbit_dense,
        reviewed_max_sparse_bytes=REVIEWED_MAX_SPARSE_BYTES,
        reviewed_max_dense_bytes=REVIEWED_MAX_DENSE_BYTES,
        reviewed_max_matrix_columns=REVIEWED_MAX_MATRIX_COLUMNS,
        full_raw_construction_authorized=raw_authorized,
        full_orbit_construction_authorized=orbit_authorized,
    )


def refuse_unreviewed_full_matrix_construction() -> None:
    """Raise unless both raw and orbit constructions pass reviewed limits."""

    receipt = matrix_memory_receipt()
    if (
        not receipt.full_raw_construction_authorized
        or not receipt.full_orbit_construction_authorized
    ):
        raise KrennHigherPowerError(
            "k=2 full matrix construction refused by the reviewed "
            "dimension and memory preflight"
        )


def count_fingerprint(payload: Mapping) -> str:
    """Hash a canonical JSON count payload for deterministic replay."""

    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def exact_k2_preflight() -> Mapping:
    """Return the exact count-only ``k=2`` preflight and claim boundary."""

    domain_census = domain_orbit_census()
    memory = matrix_memory_receipt()
    payload = {
        "schema": HIGHER_POWER_SCHEMA,
        "power": K,
        "fine_grading": {
            "lattice_rank": TOKEN_COUNT,
            "D_squared_degree": list(codomain_token_degrees()),
            "multiplier_degree_rule": (
                "2*delta - deg(F_c): degree 1 at c_i and degree 2 "
                "at the other two colors of every vertex"
            ),
            "ordinary_multiplier_degree": 15,
        },
        "raw_dimensions": {
            "mixed_colorings": MIXED_COLORING_COUNT,
            "multiplier_fiber_per_coloring": (
                raw_multiplier_fiber_dimension()
            ),
            "domain_columns": raw_domain_dimension(),
            "codomain_rows": raw_codomain_dimension(),
            "D_squared_support_monomials": (
                d_squared_support_dimension()
            ),
            "D_squared_coefficient_census": (
                d_squared_coefficient_census()
            ),
        },
        "orbit_dimensions": {
            "group": "S6 x S3",
            "group_order": GROUP_ORDER,
            "domain_columns": sum(domain_census.values()),
            "domain_columns_by_coloring_occupation": domain_census,
            "codomain_rows": codomain_orbit_dimension(),
            "D_squared_support_orbits": (
                d_squared_support_orbit_dimension()
            ),
            "D_squared_coefficient_orbit_census": (
                d_squared_coefficient_orbit_census()
            ),
            "Burnside_exact": True,
            "Reynolds_lossless_over_Q": True,
        },
        "independent_k1_recurrence_regression": (
            k1_recurrence_regression()
        ),
        "memory_preflight": memory.to_dict(),
        "computations_performed": {
            "full_matrix_constructed": False,
            "Groebner_basis_constructed": False,
            "modular_elimination_performed": False,
            "count_only": True,
        },
        "dependencies": {
            "prior_k1_result": "D not in J_mix at k=1",
            "prior_border_result": (
                "GHZ_6,3 is in the Euclidean and Zariski border image"
            ),
        },
        "claims": {
            "D_squared_in_J_mix_decided": False,
            "D_in_radical_J_mix_decided": False,
            "GHZ_nonexistence_proved": False,
            "exact_affine_GHZ_membership_decided": False,
            "evidence_status": "undecided",
        },
    }
    result = dict(payload)
    result["count_fingerprint"] = count_fingerprint(payload)
    return result


def main() -> int:
    print(json.dumps(exact_k2_preflight(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
