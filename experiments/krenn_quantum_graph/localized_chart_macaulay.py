r"""Symmetry- and torus-reduced bounded Nullstellensatz searches.

The nine seed weights fixed to one leave a nine-dimensional endpoint-color
torus.  Every normalized chart generator is homogeneous for its character
lattice.  Therefore the weight-zero part of a bounded Nullstellensatz
identity may be selected without loss.  Averaging over the finite ordered
seed stabilizer then shows that multiplier/generator pairs may be replaced
by their orbit sums, again without loss over characteristic zero.

This module constructs those exact orbit-sum columns.  Modular row reduction
is used only in a fail-closed way:

* a modular solution is discovery data, not a rational certificate;
* a modular miss is normally not a proof;
* if the source matrix has full column rank modulo ``p`` and adjoining the
  constant raises rank by one, the two nonzero minors also exist over Q.
  That proves only the stated bounded-degree nonmembership over Q.

No bounded miss is promoted to a decision about the full chart ideal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement, permutations
from math import gcd
from types import MappingProxyType
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_ideals import (
    CHART_VARIABLE_COUNT,
    MIXED_EQUATIONS,
    SparseChartPolynomial,
    normalized_seed_chart,
)
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    coloring_index,
    perfect_matchings,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    transport_ordered_seed,
)


BOUNDED_MACAULAY_SCHEMA = (
    "krenn-n6-d3-localized-chart-bounded-macaulay-v1"
)
EXPECTED_STABILIZER_SIZES = (288, 16, 12, 48, 4, 4, 12, 36)
EXPECTED_ORBIT_COLUMNS = {
    4: (11, 48, 24, 13, 58, 43, 18, 9),
    5: (90, 575, 383, 162, 1_044, 898, 311, 130),
    6: (670, 6_510, 6_049, 1_947, 17_234, 16_036, 5_318, 1_903),
}

Character = tuple[int, ...]
Monomial = tuple[int, ...]
CertificatePair = tuple[int, Monomial]


class KrennMacaulayError(RuntimeError):
    """A grading, symmetry, orbit, or modular rank replay failed."""


def _transport_coloring(
    coloring: Sequence[int],
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, ...]:
    coloring = tuple(map(int, coloring))
    result = [0] * 6
    for old_vertex, old_color in enumerate(coloring):
        result[vertex_permutation[old_vertex]] = color_permutation[old_color]
    return tuple(result)


@dataclass(frozen=True)
class OrderedSeedStabilizer:
    """Exact permutations induced by one ordered-seed stabilizer."""

    orbit_index: int
    vertex_color_elements: tuple[
        tuple[tuple[int, ...], tuple[int, ...]], ...
    ]
    chart_variable_permutations: tuple[tuple[int, ...], ...]
    generator_permutations: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        orbit_index = int(self.orbit_index)
        expected_size = EXPECTED_STABILIZER_SIZES[orbit_index]
        if (
            len(self.vertex_color_elements) != expected_size
            or len(self.chart_variable_permutations) != expected_size
            or len(self.generator_permutations) != expected_size
            or any(
                set(permutation) != set(range(CHART_VARIABLE_COUNT))
                for permutation in self.chart_variable_permutations
            )
            or any(
                set(permutation) != set(range(729))
                for permutation in self.generator_permutations
            )
        ):
            raise KrennMacaulayError(
                "ordered seed stabilizer permutation census changed"
            )

    @property
    def order(self) -> int:
        return len(self.vertex_color_elements)


def _transport_chart_variable(
    orbit_index: int,
    local_variable: int,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
    ambient_to_local: dict[int, int],
) -> int:
    local_variable = int(local_variable)
    if local_variable >= 126:
        return 126 + color_permutation[local_variable - 126]
    chart = normalized_seed_chart(orbit_index)
    ambient = chart.remaining_weight_indices[local_variable]
    i, j, a, b = variable_key(6, 3, ambient)
    transported = variable_index(
        6,
        3,
        vertex_permutation[i],
        vertex_permutation[j],
        color_permutation[a],
        color_permutation[b],
    )
    try:
        return ambient_to_local[transported]
    except KeyError as error:
        raise KrennMacaulayError(
            "a seed stabilizer moved a remaining weight into the fixed seed"
        ) from error


def _transport_generator(
    generator_index: int,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
    mixed_position: dict[int, int],
) -> int:
    generator_index = int(generator_index)
    if generator_index >= len(MIXED_EQUATIONS):
        color = generator_index - len(MIXED_EQUATIONS)
        return len(MIXED_EQUATIONS) + color_permutation[color]
    equation = MIXED_EQUATIONS[generator_index]
    transported_coloring = _transport_coloring(
        coloring_from_index(6, 3, equation),
        vertex_permutation,
        color_permutation,
    )
    transported_equation = coloring_index(6, 3, transported_coloring)
    try:
        return mixed_position[transported_equation]
    except KeyError as error:
        raise KrennMacaulayError(
            "a mixed equation transported to a pure equation"
        ) from error


def _transport_polynomial(
    polynomial: SparseChartPolynomial,
    variable_permutation: Sequence[int],
) -> SparseChartPolynomial:
    coefficients: dict[tuple[int, ...], int] = {}
    for coefficient, monomial in polynomial.terms:
        transported = tuple(
            sorted(variable_permutation[variable] for variable in monomial)
        )
        coefficients[transported] = (
            coefficients.get(transported, 0) + coefficient
        )
    return SparseChartPolynomial.from_mapping(coefficients)


@lru_cache(maxsize=8)
def ordered_seed_stabilizer(
    orbit_index: int,
) -> OrderedSeedStabilizer:
    """Construct and fully replay the stabilizer action on one chart."""

    orbit_index = int(orbit_index)
    chart = normalized_seed_chart(orbit_index)
    elements = tuple(
        (vertex_permutation, color_permutation)
        for vertex_permutation in permutations(range(6))
        for color_permutation in permutations(range(3))
        if transport_ordered_seed(
            chart.seed, vertex_permutation, color_permutation
        ) == chart.seed
    )
    if len(elements) != EXPECTED_STABILIZER_SIZES[orbit_index]:
        raise KrennMacaulayError(
            "ordered seed stabilizer order changed"
        )
    ambient_to_local = {
        ambient: local
        for local, ambient in enumerate(chart.remaining_weight_indices)
    }
    mixed_position = {
        equation: position
        for position, equation in enumerate(MIXED_EQUATIONS)
    }
    variable_permutations = []
    generator_permutations = []
    for vertex_permutation, color_permutation in elements:
        variable_permutations.append(tuple(
            _transport_chart_variable(
                orbit_index,
                variable,
                vertex_permutation,
                color_permutation,
                ambient_to_local,
            )
            for variable in range(CHART_VARIABLE_COUNT)
        ))
        generator_permutations.append(tuple(
            _transport_generator(
                generator,
                vertex_permutation,
                color_permutation,
                mixed_position,
            )
            for generator in range(729)
        ))
    result = OrderedSeedStabilizer(
        orbit_index=orbit_index,
        vertex_color_elements=elements,
        chart_variable_permutations=tuple(variable_permutations),
        generator_permutations=tuple(generator_permutations),
    )
    for variable_permutation, generator_permutation in zip(
        result.chart_variable_permutations,
        result.generator_permutations,
        strict=True,
    ):
        for generator, polynomial in enumerate(chart.generators):
            if _transport_polynomial(
                polynomial, variable_permutation
            ) != chart.generators[generator_permutation[generator]]:
                raise KrennMacaulayError(
                    "a stabilizer element failed full polynomial replay"
                )
    return result


@lru_cache(maxsize=8)
def residual_torus_characters(
    orbit_index: int,
) -> tuple[Character, ...]:
    """Return the Z^9 character of every normalized chart variable."""

    chart = normalized_seed_chart(int(orbit_index))
    endpoint_characters: dict[tuple[int, int], Character] = {}
    for color, matching_index in enumerate(chart.seed):
        matching = perfect_matchings(6)[matching_index]
        for edge_position, (left, right) in enumerate(matching):
            positive = [0] * 9
            positive[3 * color + edge_position] = 1
            negative = [0] * 9
            negative[3 * color + edge_position] = -1
            endpoint_characters[left, color] = tuple(positive)
            endpoint_characters[right, color] = tuple(negative)
    result = []
    for ambient in chart.remaining_weight_indices:
        i, j, a, b = variable_key(6, 3, ambient)
        result.append(tuple(
            left + right
            for left, right in zip(
                endpoint_characters[i, a],
                endpoint_characters[j, b],
                strict=True,
            )
        ))
    result.extend(((0,) * 9,) * 3)
    characters = tuple(result)
    if (
        len(characters) != CHART_VARIABLE_COUNT
        or len(set(characters[:126])) != 126
        or any(not any(character) for character in characters[:126])
        or any(any(character) for character in characters[126:])
    ):
        raise KrennMacaulayError(
            "residual torus character census changed"
        )
    return characters


def _monomial_character(
    monomial: Sequence[int],
    characters: Sequence[Character],
) -> Character:
    return tuple(
        sum(characters[variable][coordinate] for variable in monomial)
        for coordinate in range(9)
    )


@lru_cache(maxsize=8)
def generator_characters(
    orbit_index: int,
) -> tuple[Character, ...]:
    """Replay residual-torus homogeneity of all 729 generators."""

    chart = normalized_seed_chart(int(orbit_index))
    characters = residual_torus_characters(int(orbit_index))
    result = []
    for polynomial in chart.generators:
        weights = {
            _monomial_character(monomial, characters)
            for _coefficient, monomial in polynomial.terms
        }
        if len(weights) != 1:
            raise KrennMacaulayError(
                "a normalized generator is not torus homogeneous"
            )
        result.append(next(iter(weights)))
    return tuple(result)


def _multiplier_monomials_by_weight(
    orbit_index: int,
    maximum_degree: int,
) -> tuple[dict[Character, tuple[Monomial, ...]], ...]:
    characters = residual_torus_characters(int(orbit_index))
    result = []
    for degree in range(maximum_degree + 1):
        grouped: dict[Character, list[Monomial]] = {}
        for monomial in combinations_with_replacement(
            range(CHART_VARIABLE_COUNT), degree
        ):
            character = _monomial_character(monomial, characters)
            grouped.setdefault(character, []).append(monomial)
        result.append({
            character: tuple(monomials)
            for character, monomials in grouped.items()
        })
    return tuple(result)


@lru_cache(maxsize=24)
def bounded_certificate_pairs(
    orbit_index: int,
    total_degree_bound: int,
) -> tuple[CertificatePair, ...]:
    """Return every torus-compatible bounded multiplier/generator pair."""

    orbit_index = int(orbit_index)
    total_degree_bound = int(total_degree_bound)
    if total_degree_bound < 4:
        raise KrennMacaulayError(
            "the bounded chart search starts at total degree four"
        )
    chart = normalized_seed_chart(orbit_index)
    maximum_multiplier_degree = max(
        total_degree_bound - polynomial.degree
        for polynomial in chart.generators
    )
    grouped = _multiplier_monomials_by_weight(
        orbit_index, maximum_multiplier_degree
    )
    pairs = []
    for generator, (polynomial, character) in enumerate(zip(
        chart.generators,
        generator_characters(orbit_index),
        strict=True,
    )):
        target = tuple(-entry for entry in character)
        for degree in range(total_degree_bound - polynomial.degree + 1):
            pairs.extend(
                (generator, monomial)
                for monomial in grouped[degree].get(target, ())
            )
    return tuple(pairs)


def _transport_pair(
    pair: CertificatePair,
    variable_permutation: Sequence[int],
    generator_permutation: Sequence[int],
) -> CertificatePair:
    generator, monomial = pair
    return (
        generator_permutation[generator],
        tuple(sorted(
            variable_permutation[variable] for variable in monomial
        )),
    )


@lru_cache(maxsize=24)
def bounded_pair_orbits(
    orbit_index: int,
    total_degree_bound: int,
) -> tuple[tuple[CertificatePair, ...], ...]:
    """Return finite-stabilizer orbits of all allowed certificate pairs."""

    orbit_index = int(orbit_index)
    pairs = bounded_certificate_pairs(orbit_index, total_degree_bound)
    pair_set = set(pairs)
    symmetry = ordered_seed_stabilizer(orbit_index)
    seen: set[CertificatePair] = set()
    orbits = []
    for pair in pairs:
        if pair in seen:
            continue
        orbit = tuple(sorted({
            _transport_pair(
                pair, variable_permutation, generator_permutation
            )
            for variable_permutation, generator_permutation in zip(
                symmetry.chart_variable_permutations,
                symmetry.generator_permutations,
                strict=True,
            )
        }))
        if not set(orbit).issubset(pair_set):
            raise KrennMacaulayError(
                "stabilizer moved a bounded torus pair outside its space"
            )
        seen.update(orbit)
        orbits.append(orbit)
    if seen != pair_set:
        raise KrennMacaulayError(
            "bounded pair orbit partition is incomplete"
        )
    result = tuple(orbits)
    expected = EXPECTED_ORBIT_COLUMNS.get(total_degree_bound)
    if expected is not None and len(result) != expected[orbit_index]:
        raise KrennMacaulayError(
            "bounded orbit-column census changed"
        )
    return result


def _canonical_row_monomial(
    monomial: Monomial,
    variable_permutations: Sequence[Sequence[int]],
    cache: dict[Monomial, Monomial],
) -> Monomial:
    try:
        return cache[monomial]
    except KeyError:
        representative = min(
            tuple(sorted(permutation[variable] for variable in monomial))
            for permutation in variable_permutations
        )
        cache[monomial] = representative
        return representative


@lru_cache(maxsize=24)
def bounded_orbit_columns(
    orbit_index: int,
    total_degree_bound: int,
) -> tuple[Mapping[Monomial, int], ...]:
    """Build exact invariant orbit-sum Macaulay columns.

    Rows in the same monomial orbit are collapsed by summing.  On invariant
    polynomials this only multiplies each row by its nonzero orbit size over
    Q.  Modular callers must choose a prime coprime to the stabilizer order.
    """

    orbit_index = int(orbit_index)
    chart = normalized_seed_chart(orbit_index)
    symmetry = ordered_seed_stabilizer(orbit_index)
    row_cache: dict[Monomial, Monomial] = {}
    columns = []
    for pair_orbit in bounded_pair_orbits(
        orbit_index, total_degree_bound
    ):
        column: dict[Monomial, int] = {}
        # Row-orbit summation commutes with the stabilizer.  If P is the
        # polynomial for one pair, the distinct-pair orbit sum is the full
        # Reynolds sum divided by the pair stabilizer.  After collapsing a
        # monomial orbit, every transform of P contributes the same row sum.
        # Hence it is enough to collapse P once and multiply by the number
        # of distinct pairs in its orbit.
        generator, multiplier = pair_orbit[0]
        orbit_size = len(pair_orbit)
        for coefficient, generator_monomial in (
            chart.generators[generator].terms
        ):
            product_monomial = tuple(sorted(
                (*multiplier, *generator_monomial)
            ))
            row = _canonical_row_monomial(
                product_monomial,
                symmetry.chart_variable_permutations,
                row_cache,
            )
            total = (
                column.get(row, 0) + orbit_size * coefficient
            )
            if total:
                column[row] = total
            else:
                column.pop(row, None)
        if not column:
            raise KrennMacaulayError(
                "an invariant Macaulay orbit column vanished over Q"
            )
        columns.append(MappingProxyType(column))
    return tuple(columns)


def _primitive_relation(
    coefficients: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    coefficients = {
        column: int(coefficient)
        for column, coefficient in coefficients.items()
        if coefficient
    }
    if not coefficients:
        return ()
    common = 0
    for coefficient in coefficients.values():
        common = gcd(common, abs(coefficient))
    coefficients = {
        column: coefficient // common
        for column, coefficient in coefficients.items()
    }
    first_column = min(coefficients)
    if coefficients[first_column] < 0:
        coefficients = {
            column: -coefficient
            for column, coefficient in coefficients.items()
        }
    return tuple(sorted(coefficients.items()))


@lru_cache(maxsize=24)
def bounded_koszul_orbit_relations(
    orbit_index: int,
    total_degree_bound: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return exact averaged Koszul relations among orbit columns.

    If homogeneous generators ``g_i`` and ``g_j`` have opposite residual
    characters and their degrees fit the bound, then

        g_j * g_i - g_i * g_j = 0.

    Reynolds averaging expresses this identity in the invariant orbit-column
    basis.  The returned integer relations are exactly replayed against the
    expanded columns before they are accepted.
    """

    orbit_index = int(orbit_index)
    total_degree_bound = int(total_degree_bound)
    chart = normalized_seed_chart(orbit_index)
    symmetry = ordered_seed_stabilizer(orbit_index)
    pair_orbits = bounded_pair_orbits(
        orbit_index, total_degree_bound
    )
    pair_to_column = {
        pair: column
        for column, pair_orbit in enumerate(pair_orbits)
        for pair in pair_orbit
    }
    characters = generator_characters(orbit_index)
    relations = set()
    seen_generator_pairs: set[tuple[int, int]] = set()
    for left in range(len(chart.generators)):
        left_polynomial = chart.generators[left]
        for right in range(left + 1, len(chart.generators)):
            right_polynomial = chart.generators[right]
            if (
                left_polynomial.degree + right_polynomial.degree
                > total_degree_bound
                or characters[right]
                != tuple(-entry for entry in characters[left])
            ):
                continue
            generator_pair = (left, right)
            if generator_pair in seen_generator_pairs:
                continue
            generator_pair_orbit = {
                tuple(sorted((
                    permutation[left], permutation[right]
                )))
                for permutation in symmetry.generator_permutations
            }
            seen_generator_pairs.update(generator_pair_orbit)
            base_terms = tuple(
                ((left, monomial), coefficient)
                for coefficient, monomial in right_polynomial.terms
            ) + tuple(
                ((right, monomial), -coefficient)
                for coefficient, monomial in left_polynomial.terms
            )
            pair_coefficients: dict[CertificatePair, int] = {}
            for variable_permutation, generator_permutation in zip(
                symmetry.chart_variable_permutations,
                symmetry.generator_permutations,
                strict=True,
            ):
                for pair, coefficient in base_terms:
                    transported = _transport_pair(
                        pair,
                        variable_permutation,
                        generator_permutation,
                    )
                    pair_coefficients[transported] = (
                        pair_coefficients.get(transported, 0)
                        + coefficient
                    )
            touched_columns = {
                pair_to_column[pair]
                for pair, coefficient in pair_coefficients.items()
                if coefficient
            }
            relation = {}
            for column in touched_columns:
                values = {
                    pair_coefficients.get(pair, 0)
                    for pair in pair_orbits[column]
                }
                if len(values) != 1:
                    raise KrennMacaulayError(
                        "an averaged Koszul coefficient is not constant "
                        "on a certificate-pair orbit"
                    )
                coefficient = values.pop()
                if coefficient:
                    relation[column] = coefficient
            primitive = _primitive_relation(relation)
            if primitive:
                relations.add(primitive)
    result = tuple(sorted(relations))
    columns = bounded_orbit_columns(
        orbit_index, total_degree_bound
    )
    for relation in result:
        replay: dict[Monomial, int] = {}
        for column, relation_coefficient in relation:
            for monomial, column_coefficient in columns[column].items():
                total = (
                    replay.get(monomial, 0)
                    + relation_coefficient * column_coefficient
                )
                if total:
                    replay[monomial] = total
                else:
                    replay.pop(monomial, None)
        if replay:
            raise KrennMacaulayError(
                "an averaged Koszul relation failed exact replay"
            )
    return result


def _is_prime(value: int) -> bool:
    value = int(value)
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return value == divisor
        divisor += 1 if divisor == 2 else 2
    return True


def _add_sparse_column(
    raw_column: dict[int, int],
    basis: dict[int, dict[int, int]],
    prime: int,
) -> bool:
    vector = {
        row: coefficient % prime
        for row, coefficient in raw_column.items()
        if coefficient % prime
    }
    while vector:
        pivot = max(vector)
        coefficient = vector[pivot]
        prior = basis.get(pivot)
        if prior is None:
            inverse = pow(coefficient, -1, prime)
            basis[pivot] = {
                row: value * inverse % prime
                for row, value in vector.items()
                if value * inverse % prime
            }
            return True
        factor = coefficient
        for row, value in prior.items():
            updated = (vector.get(row, 0) - factor * value) % prime
            if updated:
                vector[row] = updated
            else:
                vector.pop(row, None)
    return False


def _sparse_vector_rank_mod_p(
    vectors: Sequence[dict[int, int]],
    prime: int,
) -> int:
    basis: dict[int, dict[int, int]] = {}
    return sum(
        int(_add_sparse_column(vector, basis, prime))
        for vector in vectors
    )


@dataclass(frozen=True)
class BoundedMacaulayResult:
    """Fail-closed result of one modular bounded certificate preflight."""

    orbit_index: int
    total_degree_bound: int
    prime: int
    stabilizer_order: int
    raw_torus_pairs: int
    invariant_columns: int
    invariant_rows: int
    nonzero_entries: int
    exact_koszul_relations: int
    independent_koszul_relation_rank_mod_p: int
    source_rank_upper_bound_over_q: int
    source_rank_mod_p: int
    augmented_rank_mod_p: int
    target_in_span_mod_p: bool
    bounded_nonmembership_over_q_certified: bool
    schema: str = BOUNDED_MACAULAY_SCHEMA

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "orbit_index": self.orbit_index,
            "seed": list(
                normalized_seed_chart(self.orbit_index).seed
            ),
            "total_degree_bound": self.total_degree_bound,
            "prime": self.prime,
            "stabilizer_order": self.stabilizer_order,
            "raw_torus_pairs": self.raw_torus_pairs,
            "invariant_columns": self.invariant_columns,
            "invariant_rows": self.invariant_rows,
            "nonzero_entries": self.nonzero_entries,
            "exact_koszul_relations": self.exact_koszul_relations,
            "independent_koszul_relation_rank_mod_p":
                self.independent_koszul_relation_rank_mod_p,
            "source_rank_upper_bound_over_q":
                self.source_rank_upper_bound_over_q,
            "source_rank_mod_p": self.source_rank_mod_p,
            "augmented_rank_mod_p": self.augmented_rank_mod_p,
            "target_in_span_mod_p": self.target_in_span_mod_p,
            "bounded_nonmembership_over_q_certified":
                self.bounded_nonmembership_over_q_certified,
            "exact_reason_if_certified": (
                "exact averaged Koszul relations give a source-rank upper "
                "bound over Q; modular source and augmented minors attain "
                "that bound and raise it for the constant"
                if self.bounded_nonmembership_over_q_certified
                else None
            ),
            "claim_boundary": {
                "full_chart_unit_ideal_decided": False,
                "finite_affine_GHZ_membership_decided": False,
                "modular_solution_is_rational_certificate": False,
                "bounded_miss_is_global_nonexistence_proof": False,
            },
        }


def modular_bounded_macaulay_preflight(
    orbit_index: int,
    total_degree_bound: int,
    *,
    prime: int = 1_009,
) -> BoundedMacaulayResult:
    """Reduce the invariant bounded Macaulay span over one prime."""

    orbit_index = int(orbit_index)
    total_degree_bound = int(total_degree_bound)
    prime = int(prime)
    if not _is_prime(prime):
        raise KrennMacaulayError("Macaulay modulus must be prime")
    symmetry = ordered_seed_stabilizer(orbit_index)
    if gcd(prime, symmetry.order) != 1:
        raise KrennMacaulayError(
            "Macaulay prime must be coprime to the stabilizer order"
        )
    columns = bounded_orbit_columns(orbit_index, total_degree_bound)
    relations = bounded_koszul_orbit_relations(
        orbit_index, total_degree_bound
    )
    relation_rank = _sparse_vector_rank_mod_p(
        (
            {column: coefficient for column, coefficient in relation}
            for relation in relations
        ),
        prime,
    )
    source_rank_upper_bound = len(columns) - relation_rank
    rows = tuple(sorted(
        {
            monomial
            for column in columns
            for monomial in column
        }.union({()}),
        key=lambda monomial: (len(monomial), monomial),
    ))
    row_index = {monomial: index for index, monomial in enumerate(rows)}
    basis: dict[int, dict[int, int]] = {}
    source_rank = 0
    for column in columns:
        indexed = {
            row_index[monomial]: coefficient
            for monomial, coefficient in column.items()
        }
        source_rank += int(_add_sparse_column(indexed, basis, prime))
    target_independent = _add_sparse_column(
        {row_index[()]: 1}, basis, prime
    )
    augmented_rank = source_rank + int(target_independent)
    certified = (
        source_rank == source_rank_upper_bound
        and target_independent
    )
    return BoundedMacaulayResult(
        orbit_index=orbit_index,
        total_degree_bound=total_degree_bound,
        prime=prime,
        stabilizer_order=symmetry.order,
        raw_torus_pairs=len(bounded_certificate_pairs(
            orbit_index, total_degree_bound
        )),
        invariant_columns=len(columns),
        invariant_rows=len(rows),
        nonzero_entries=sum(len(column) for column in columns),
        exact_koszul_relations=len(relations),
        independent_koszul_relation_rank_mod_p=relation_rank,
        source_rank_upper_bound_over_q=source_rank_upper_bound,
        source_rank_mod_p=source_rank,
        augmented_rank_mod_p=augmented_rank,
        target_in_span_mod_p=not target_independent,
        bounded_nonmembership_over_q_certified=certified,
    )
