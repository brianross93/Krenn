r"""Character-zero Macaulay systems for the sparse derivative slices.

The two sparse natural-chart slices append ``d-1`` to the 729 direct-GHZ
equations.  The derivative ``d`` is a primitive semi-invariant for the
residual ``Z^9`` torus.  Quotienting by its character makes ``d-1``
homogeneous and leaves a split ``Z^8`` grading.

If ``1`` belongs to one of these ideals, projecting a certificate to
character zero loses nothing.  Thus a generator of character ``beta`` only
needs multiplier monomials of character ``-beta``.  Averaging under the
order-two stabilizer of the selected derivative is also lossless over Q
and over odd finite fields.  It averages certificates; it never equates
symmetry-related variables.

This module constructs those exact bounded systems through certificate
degree six.  Exact averaged Koszul relations give a rational source-rank
upper bound.  A modular source minor attaining that bound, together with an
augmented minor one rank larger, proves bounded nonmembership over Q.

No bounded miss decides the full ideal.  A finite-field unit or proper ideal
also remains reconnaissance until an exact characteristic-zero certificate
is reconstructed and replayed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement
import json
from math import gcd
from types import MappingProxyType
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    DERIVATIVE_REPRESENTATIVES,
    NATURAL_ORBIT_INDEX,
    sparse_derivative_gauge_slice,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    SparseChartPolynomial,
    strict_json_equal,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    generator_characters,
    ordered_seed_stabilizer,
    residual_torus_characters,
)


GRADED_DERIVATIVE_MACAULAY_SCHEMA = (
    "krenn-n6-d3-sparse-derivative-graded-macaulay-v1"
)
GRADED_DERIVATIVE_MACAULAY_AUDIT_SCHEMA = (
    "krenn-n6-d3-sparse-derivative-graded-macaulay-audit-v1"
)
SUPPORTED_TOTAL_DEGREES = (4, 5, 6)
RETAINED_PRIMES = (31, 1_009)

EXPECTED_QUOTIENT_CENSUS = {
    11: {
        "derivative_character": (-1, 0, 0, 0, 0, 0, 0, 0, -1),
        "stabilizer_element_indices": (0, 4),
        "variable_blocks": 105,
        "zero_character_variables": 5,
        "variable_block_histogram": ((1, 84), (2, 20), (5, 1)),
        "generator_blocks": 558,
        "zero_character_generators": 7,
        "generator_block_histogram": (
            (1, 415),
            (2, 126),
            (3, 8),
            (4, 8),
            (7, 1),
        ),
    },
    29: {
        "derivative_character": (-1, 0, 0, 0, 0, 0, 0, 0, 1),
        "stabilizer_element_indices": (0, 8),
        "variable_blocks": 105,
        "zero_character_variables": 5,
        "variable_block_histogram": ((1, 84), (2, 20), (5, 1)),
        "generator_blocks": 642,
        "zero_character_generators": 7,
        "generator_block_histogram": (
            (1, 567),
            (2, 66),
            (3, 8),
            (7, 1),
        ),
    },
}

EXPECTED_BOUNDED_CENSUS = {
    (11, 4): (339, 187, 1_638, 2_054),
    (11, 5): (7_400, 3_794, 33_222, 49_278),
    (11, 6): (141_317, 71_078, 532_675, 981_006),
    (29, 4): (313, 180, 1_579, 1_964),
    (29, 5): (7_052, 3_676, 32_053, 47_380),
    (29, 6): (135_891, 68_731, 517_236, 944_817),
}

EXPECTED_KOSZUL_RELATIONS = {
    (11, 4): 0,
    (11, 5): 3,
    (11, 6): 126,
    (29, 4): 0,
    (29, 5): 3,
    (29, 6): 122,
}

RETAINED_DEGREE_SIX_RANKS = {
    11: {
        "raw_character_pairs": 141_317,
        "invariant_columns": 71_078,
        "invariant_rows": 532_675,
        "nonzero_entries": 981_006,
        "exact_koszul_relations": 126,
        "independent_relation_rank_mod_p": 126,
        "source_rank_upper_bound_over_q": 70_952,
        "source_rank_mod_p": 70_829,
        "augmented_rank_mod_p": 70_830,
    },
    29: {
        "raw_character_pairs": 135_891,
        "invariant_columns": 68_731,
        "invariant_rows": 517_236,
        "nonzero_entries": 944_817,
        "exact_koszul_relations": 122,
        "independent_relation_rank_mod_p": 122,
        "source_rank_upper_bound_over_q": 68_609,
        "source_rank_mod_p": 68_492,
        "augmented_rank_mod_p": 68_493,
    },
}

Character = tuple[int, ...]
Monomial = tuple[int, ...]
CertificatePair = tuple[int, Monomial]
SparseColumn = Mapping[Monomial, int]
Relation = tuple[tuple[int, int], ...]


class KrennGradedMacaulayError(RuntimeError):
    """A quotient, stabilizer, column, relation, or rank replay failed."""


def _validated_derivative(derivative_ambient_weight: int) -> int:
    value = int(derivative_ambient_weight)
    if value not in DERIVATIVE_REPRESENTATIVES:
        raise KrennGradedMacaulayError(
            "graded derivative weight must be 11 or 29"
        )
    return value


def _validated_degree(total_degree_bound: int) -> int:
    value = int(total_degree_bound)
    if value not in SUPPORTED_TOTAL_DEGREES:
        raise KrennGradedMacaulayError(
            "graded derivative Macaulay bounds are exactly 4, 5, and 6"
        )
    return value


def _project_character(
    character: Sequence[int],
    derivative_character: Sequence[int],
) -> Character:
    """Project ``Z^9`` to its split quotient by ``derivative_character``."""

    character = tuple(map(int, character))
    relation = tuple(map(int, derivative_character))
    if (
        len(character) != 9
        or len(relation) != 9
        or relation[0] != -1
        or any(relation[index] for index in range(1, 8))
        or abs(relation[8]) != 1
    ):
        raise KrennGradedMacaulayError(
            "the retained derivative quotient lost its unit-pivot form"
        )
    # Add character[0] times the relation, killing coordinate zero, and
    # retain coordinates 1,...,8.  The section on those eight coordinates
    # is the identity, so the quotient is split over Z.
    reduced = tuple(
        left + character[0] * right
        for left, right in zip(character, relation, strict=True)
    )
    if reduced[0]:
        raise KrennGradedMacaulayError(
            "the derivative-character quotient did not kill its pivot"
        )
    return reduced[1:]


def _monomial_character(
    monomial: Sequence[int],
    variable_characters: Sequence[Character],
) -> Character:
    return tuple(
        sum(variable_characters[variable][coordinate] for variable in monomial)
        for coordinate in range(8)
    )


@dataclass(frozen=True)
class DerivativeResidualGrading:
    """The exact split ``Z^8`` grading of one sparse derivative slice."""

    derivative_ambient_weight: int
    derivative_character: Character
    variable_characters: tuple[Character, ...]
    generator_characters: tuple[Character, ...]
    variable_blocks: int
    generator_blocks: int
    zero_character_variables: int
    zero_character_generators: int

    @property
    def residual_rank(self) -> int:
        return 8

    @property
    def smith_diagonal(self) -> tuple[int, ...]:
        return (1,)

    @property
    def projection_rows(self) -> tuple[Character, ...]:
        relation = self.derivative_character
        return tuple(
            _project_character(
                tuple(int(source == coordinate) for source in range(9)),
                relation,
            )
            for coordinate in range(9)
        )


@lru_cache(maxsize=2)
def derivative_residual_grading(
    derivative_ambient_weight: int,
) -> DerivativeResidualGrading:
    """Build and fully replay the split quotient grading."""

    ambient = _validated_derivative(derivative_ambient_weight)
    chart = sparse_derivative_gauge_slice(ambient)
    relation = tuple(chart.derivative_character)
    expected = EXPECTED_QUOTIENT_CENSUS[ambient]
    if relation != expected["derivative_character"]:
        raise KrennGradedMacaulayError(
            "the derivative character changed"
        )
    if _project_character(relation, relation) != (0,) * 8:
        raise KrennGradedMacaulayError(
            "the quotient did not kill the derivative character"
        )

    variables = tuple(
        _project_character(character, relation)
        for character in residual_torus_characters(NATURAL_ORBIT_INDEX)
    )
    generators = tuple(
        _project_character(character, relation)
        for character in generator_characters(NATURAL_ORBIT_INDEX)
    ) + ((0,) * 8,)
    if len(variables) != 129 or len(generators) != 730:
        raise KrennGradedMacaulayError(
            "the sparse derivative grading has the wrong dimensions"
        )

    for polynomial, generator_character in zip(
        chart.generators, generators, strict=True
    ):
        term_characters = {
            _monomial_character(monomial, variables)
            for _coefficient, monomial in polynomial.terms
        }
        if term_characters != {generator_character}:
            raise KrennGradedMacaulayError(
                "a sparse derivative generator is not quotient homogeneous"
            )

    variable_sizes = Counter(variables)
    generator_sizes = Counter(generators)
    observed = {
        "variable_blocks": len(variable_sizes),
        "zero_character_variables": variable_sizes[(0,) * 8],
        "variable_block_histogram": tuple(sorted(
            Counter(variable_sizes.values()).items()
        )),
        "generator_blocks": len(generator_sizes),
        "zero_character_generators": generator_sizes[(0,) * 8],
        "generator_block_histogram": tuple(sorted(
            Counter(generator_sizes.values()).items()
        )),
    }
    if any(observed[key] != expected[key] for key in observed):
        raise KrennGradedMacaulayError(
            "the sparse derivative character-block census changed"
        )
    return DerivativeResidualGrading(
        derivative_ambient_weight=ambient,
        derivative_character=relation,
        variable_characters=variables,
        generator_characters=generators,
        variable_blocks=observed["variable_blocks"],
        generator_blocks=observed["generator_blocks"],
        zero_character_variables=observed["zero_character_variables"],
        zero_character_generators=observed["zero_character_generators"],
    )


def _transport_polynomial(
    polynomial: SparseChartPolynomial,
    variable_permutation: Sequence[int],
) -> SparseChartPolynomial:
    coefficients: dict[Monomial, int] = {}
    for coefficient, monomial in polynomial.terms:
        transported = tuple(sorted(
            variable_permutation[variable] for variable in monomial
        ))
        total = coefficients.get(transported, 0) + coefficient
        if total:
            coefficients[transported] = total
        else:
            coefficients.pop(transported, None)
    return SparseChartPolynomial.from_mapping(coefficients)


@dataclass(frozen=True)
class DerivativeSliceStabilizer:
    """The order-two subgroup preserving one derivative gauge equation."""

    derivative_ambient_weight: int
    parent_element_indices: tuple[int, ...]
    variable_permutations: tuple[tuple[int, ...], ...]
    generator_permutations: tuple[tuple[int, ...], ...]

    @property
    def order(self) -> int:
        return len(self.variable_permutations)


@lru_cache(maxsize=2)
def derivative_slice_stabilizer(
    derivative_ambient_weight: int,
) -> DerivativeSliceStabilizer:
    """Return and exactly replay the subgroup fixing the selected slice."""

    ambient = _validated_derivative(derivative_ambient_weight)
    chart = sparse_derivative_gauge_slice(ambient)
    parent = ordered_seed_stabilizer(NATURAL_ORBIT_INDEX)
    local = chart.derivative_original_local_variable
    retained = []
    for index, (
        variable_permutation,
        natural_generator_permutation,
    ) in enumerate(zip(
        parent.chart_variable_permutations,
        parent.generator_permutations,
        strict=True,
    )):
        if variable_permutation[local] != local:
            continue
        generator_permutation = (
            *natural_generator_permutation,
            729,
        )
        if _transport_polynomial(
            chart.generators[-1], variable_permutation
        ) != chart.generators[-1]:
            continue
        retained.append((
            index,
            tuple(variable_permutation),
            tuple(generator_permutation),
        ))
    expected_indices = EXPECTED_QUOTIENT_CENSUS[ambient][
        "stabilizer_element_indices"
    ]
    if tuple(row[0] for row in retained) != expected_indices:
        raise KrennGradedMacaulayError(
            "the derivative-slice stabilizer changed"
        )
    result = DerivativeSliceStabilizer(
        derivative_ambient_weight=ambient,
        parent_element_indices=tuple(row[0] for row in retained),
        variable_permutations=tuple(row[1] for row in retained),
        generator_permutations=tuple(row[2] for row in retained),
    )
    if result.order != 2:
        raise KrennGradedMacaulayError(
            "the derivative-slice stabilizer is not order two"
        )
    for variable_permutation, generator_permutation in zip(
        result.variable_permutations,
        result.generator_permutations,
        strict=True,
    ):
        for generator, polynomial in enumerate(chart.generators):
            if _transport_polynomial(
                polynomial, variable_permutation
            ) != chart.generators[generator_permutation[generator]]:
                raise KrennGradedMacaulayError(
                    "the derivative stabilizer failed polynomial replay"
                )
    return result


@lru_cache(maxsize=2)
def _multiplier_monomials_through_degree_three(
    derivative_ambient_weight: int,
) -> tuple[Mapping[Character, tuple[Monomial, ...]], ...]:
    ambient = _validated_derivative(derivative_ambient_weight)
    characters = derivative_residual_grading(ambient).variable_characters
    result = []
    for degree in range(4):
        grouped: dict[Character, list[Monomial]] = defaultdict(list)
        for monomial in combinations_with_replacement(
            range(129), degree
        ):
            grouped[_monomial_character(monomial, characters)].append(
                monomial
            )
        result.append(MappingProxyType({
            character: tuple(monomials)
            for character, monomials in grouped.items()
        }))
    return tuple(result)


@lru_cache(maxsize=2)
def _zero_character_degree_four_monomials(
    derivative_ambient_weight: int,
) -> tuple[Monomial, ...]:
    """Enumerate only the degree-four block needed by the gauge generator."""

    ambient = _validated_derivative(derivative_ambient_weight)
    characters = derivative_residual_grading(ambient).variable_characters
    variables_by_character: dict[Character, list[int]] = defaultdict(list)
    for variable, character in enumerate(characters):
        variables_by_character[character].append(variable)

    result = []
    for first in range(129):
        for second in range(first, 129):
            for third in range(second, 129):
                target = tuple(
                    -(
                        characters[first][coordinate]
                        + characters[second][coordinate]
                        + characters[third][coordinate]
                    )
                    for coordinate in range(8)
                )
                result.extend(
                    (first, second, third, fourth)
                    for fourth in variables_by_character.get(target, ())
                    if fourth >= third
                )
    expected = {11: 13_353, 29: 13_253}[ambient]
    if len(result) != expected:
        raise KrennGradedMacaulayError(
            "the degree-four zero-character multiplier census changed"
        )
    return tuple(result)


def _multiplier_monomials(
    derivative_ambient_weight: int,
    degree: int,
    target: Character,
) -> tuple[Monomial, ...]:
    if degree <= 3:
        return _multiplier_monomials_through_degree_three(
            derivative_ambient_weight
        )[degree].get(target, ())
    if degree == 4 and target == (0,) * 8:
        return _zero_character_degree_four_monomials(
            derivative_ambient_weight
        )
    if degree == 4:
        return ()
    raise KrennGradedMacaulayError(
        "the retained graded multiplier enumeration stops at degree four"
    )


@lru_cache(maxsize=6)
def bounded_derivative_certificate_pairs(
    derivative_ambient_weight: int,
    total_degree_bound: int,
) -> tuple[CertificatePair, ...]:
    """Return every bounded character-compatible generator multiplier."""

    ambient = _validated_derivative(derivative_ambient_weight)
    bound = _validated_degree(total_degree_bound)
    chart = sparse_derivative_gauge_slice(ambient)
    characters = derivative_residual_grading(ambient).generator_characters
    pairs = []
    for generator, (polynomial, character) in enumerate(zip(
        chart.generators, characters, strict=True
    )):
        target = tuple(-entry for entry in character)
        for degree in range(bound - polynomial.degree + 1):
            pairs.extend(
                (generator, monomial)
                for monomial in _multiplier_monomials(
                    ambient, degree, target
                )
            )
    expected = EXPECTED_BOUNDED_CENSUS[(ambient, bound)][0]
    if len(pairs) != expected:
        raise KrennGradedMacaulayError(
            "the bounded derivative pair census changed"
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


@lru_cache(maxsize=6)
def bounded_derivative_pair_orbits(
    derivative_ambient_weight: int,
    total_degree_bound: int,
) -> tuple[tuple[CertificatePair, ...], ...]:
    """Partition the compatible pairs under the exact order-two action."""

    ambient = _validated_derivative(derivative_ambient_weight)
    bound = _validated_degree(total_degree_bound)
    pairs = bounded_derivative_certificate_pairs(ambient, bound)
    pair_set = set(pairs)
    symmetry = derivative_slice_stabilizer(ambient)
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
                symmetry.variable_permutations,
                symmetry.generator_permutations,
                strict=True,
            )
        }))
        if not set(orbit).issubset(pair_set):
            raise KrennGradedMacaulayError(
                "the derivative stabilizer moved a pair out of its space"
            )
        seen.update(orbit)
        orbits.append(orbit)
    if seen != pair_set:
        raise KrennGradedMacaulayError(
            "the derivative pair-orbit partition is incomplete"
        )
    result = tuple(orbits)
    expected = EXPECTED_BOUNDED_CENSUS[(ambient, bound)][1]
    if len(result) != expected:
        raise KrennGradedMacaulayError(
            "the bounded derivative orbit-column census changed"
        )
    return result


def _canonical_row(
    monomial: Monomial,
    variable_permutations: Sequence[Sequence[int]],
) -> Monomial:
    return min(
        tuple(sorted(
            permutation[variable] for variable in monomial
        ))
        for permutation in variable_permutations
    )


@lru_cache(maxsize=6)
def bounded_derivative_orbit_columns(
    derivative_ambient_weight: int,
    total_degree_bound: int,
) -> tuple[SparseColumn, ...]:
    """Construct exact order-two orbit-sum columns and row-orbit sums."""

    ambient = _validated_derivative(derivative_ambient_weight)
    bound = _validated_degree(total_degree_bound)
    chart = sparse_derivative_gauge_slice(ambient)
    symmetry = derivative_slice_stabilizer(ambient)
    columns = []
    for pair_orbit in bounded_derivative_pair_orbits(ambient, bound):
        column: dict[Monomial, int] = {}
        # Sum every distinct transformed pair directly.  Canonicalizing the
        # resulting monomials performs the exact row-orbit sum.
        for generator, multiplier in pair_orbit:
            for coefficient, generator_monomial in (
                chart.generators[generator].terms
            ):
                product = tuple(sorted(
                    (*multiplier, *generator_monomial)
                ))
                row = _canonical_row(
                    product, symmetry.variable_permutations
                )
                total = column.get(row, 0) + coefficient
                if total:
                    column[row] = total
                else:
                    column.pop(row, None)
        if not column:
            raise KrennGradedMacaulayError(
                "a derivative orbit column vanished over Q"
            )
        columns.append(MappingProxyType(column))
    result = tuple(columns)
    rows = {
        monomial
        for column in result
        for monomial in column
    }
    observed = (
        len(bounded_derivative_certificate_pairs(ambient, bound)),
        len(result),
        len(rows.union({()})),
        sum(len(column) for column in result),
    )
    if observed != EXPECTED_BOUNDED_CENSUS[(ambient, bound)]:
        raise KrennGradedMacaulayError(
            "the derivative orbit-matrix census changed"
        )
    return result


def _primitive_relation(
    coefficients: Mapping[int, int],
) -> Relation:
    cleaned = {
        int(column): int(coefficient)
        for column, coefficient in coefficients.items()
        if coefficient
    }
    if not cleaned:
        return ()
    common = 0
    for coefficient in cleaned.values():
        common = gcd(common, abs(coefficient))
    cleaned = {
        column: coefficient // common
        for column, coefficient in cleaned.items()
    }
    first = min(cleaned)
    if cleaned[first] < 0:
        cleaned = {
            column: -coefficient
            for column, coefficient in cleaned.items()
        }
    return tuple(sorted(cleaned.items()))


@lru_cache(maxsize=6)
def bounded_derivative_koszul_relations(
    derivative_ambient_weight: int,
    total_degree_bound: int,
) -> tuple[Relation, ...]:
    """Return exact averaged Koszul relations among the retained columns."""

    ambient = _validated_derivative(derivative_ambient_weight)
    bound = _validated_degree(total_degree_bound)
    if bound == 4:
        return ()
    chart = sparse_derivative_gauge_slice(ambient)
    grading = derivative_residual_grading(ambient)
    symmetry = derivative_slice_stabilizer(ambient)
    pair_orbits = bounded_derivative_pair_orbits(ambient, bound)
    pair_to_column = {
        pair: column
        for column, orbit in enumerate(pair_orbits)
        for pair in orbit
    }
    relations = set()
    for left in range(730):
        left_polynomial = chart.generators[left]
        left_character = grading.generator_characters[left]
        for right in range(left + 1, 730):
            right_polynomial = chart.generators[right]
            if (
                left_polynomial.degree + right_polynomial.degree > bound
                or grading.generator_characters[right]
                != tuple(-entry for entry in left_character)
            ):
                continue
            # g_right*g_left - g_left*g_right = 0, represented as
            # multiplier/generator pairs and then averaged over the subgroup.
            base_terms = tuple(
                ((left, monomial), coefficient)
                for coefficient, monomial in right_polynomial.terms
            ) + tuple(
                ((right, monomial), -coefficient)
                for coefficient, monomial in left_polynomial.terms
            )
            pair_coefficients: dict[CertificatePair, int] = {}
            for variable_permutation, generator_permutation in zip(
                symmetry.variable_permutations,
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
            touched = {
                pair_to_column[pair]
                for pair, coefficient in pair_coefficients.items()
                if coefficient
            }
            relation = {}
            for column in touched:
                values = {
                    pair_coefficients.get(pair, 0)
                    for pair in pair_orbits[column]
                }
                if len(values) != 1:
                    raise KrennGradedMacaulayError(
                        "an averaged derivative Koszul coefficient is not "
                        "constant on a pair orbit"
                    )
                coefficient = values.pop()
                if coefficient:
                    relation[column] = coefficient
            primitive = _primitive_relation(relation)
            if primitive:
                relations.add(primitive)

    result = tuple(sorted(relations))
    if len(result) != EXPECTED_KOSZUL_RELATIONS[(ambient, bound)]:
        raise KrennGradedMacaulayError(
            "the averaged derivative Koszul-relation census changed"
        )
    columns = bounded_derivative_orbit_columns(ambient, bound)
    for relation in result:
        replay: dict[Monomial, int] = {}
        for column, multiplier in relation:
            for monomial, coefficient in columns[column].items():
                total = (
                    replay.get(monomial, 0)
                    + multiplier * coefficient
                )
                if total:
                    replay[monomial] = total
                else:
                    replay.pop(monomial, None)
        if replay:
            raise KrennGradedMacaulayError(
                "an averaged derivative Koszul relation failed exact replay"
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
    raw_column: Mapping[int, int],
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


def _sparse_rank_mod_p(
    vectors: Sequence[Mapping[int, int]],
    prime: int,
) -> int:
    basis: dict[int, dict[int, int]] = {}
    return sum(
        int(_add_sparse_column(vector, basis, prime))
        for vector in vectors
    )


@dataclass(frozen=True)
class DerivativeBoundedMacaulayResult:
    """One fail-closed modular rank sandwich for an exact Q claim."""

    derivative_ambient_weight: int
    total_degree_bound: int
    prime: int
    raw_character_pairs: int
    invariant_columns: int
    invariant_rows: int
    nonzero_entries: int
    exact_koszul_relations: int
    independent_relation_rank_mod_p: int
    source_rank_upper_bound_over_q: int
    source_rank_mod_p: int
    augmented_rank_mod_p: int
    target_in_span_mod_p: bool
    bounded_nonmembership_over_q_certified: bool
    schema: str = GRADED_DERIVATIVE_MACAULAY_SCHEMA

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "derivative_ambient_weight":
                self.derivative_ambient_weight,
            "total_degree_bound": self.total_degree_bound,
            "prime": self.prime,
            "residual_character_rank": 8,
            "derivative_stabilizer_order": 2,
            "raw_character_pairs": self.raw_character_pairs,
            "invariant_columns": self.invariant_columns,
            "invariant_rows": self.invariant_rows,
            "nonzero_entries": self.nonzero_entries,
            "exact_koszul_relations": self.exact_koszul_relations,
            "independent_relation_rank_mod_p":
                self.independent_relation_rank_mod_p,
            "source_rank_upper_bound_over_q":
                self.source_rank_upper_bound_over_q,
            "source_rank_mod_p": self.source_rank_mod_p,
            "augmented_rank_mod_p": self.augmented_rank_mod_p,
            "target_in_span_mod_p": self.target_in_span_mod_p,
            "bounded_nonmembership_over_q_certified":
                self.bounded_nonmembership_over_q_certified,
            "exact_reason_if_certified": (
                "exact averaged Koszul relations have an independent "
                "minor modulo p, hence give a Q source-rank upper bound; "
                "source and augmented modular minors attain that bound "
                "and raise it for the constant"
                if self.bounded_nonmembership_over_q_certified
                else None
            ),
            "claim_boundary": {
                "full_slice_unit_or_proper_status_decided": False,
                "natural_chart_decided": False,
                "finite_affine_GHZ_membership_decided": False,
                "finite_field_unit_is_Q_unit_certificate": False,
                "bounded_miss_is_global_proof": False,
            },
        }


def modular_bounded_derivative_preflight(
    derivative_ambient_weight: int,
    total_degree_bound: int,
    *,
    prime: int = 1_009,
) -> DerivativeBoundedMacaulayResult:
    """Run one modular rank sandwich on the exact graded orbit system."""

    ambient = _validated_derivative(derivative_ambient_weight)
    bound = _validated_degree(total_degree_bound)
    prime = int(prime)
    if not _is_prime(prime):
        raise KrennGradedMacaulayError(
            "graded derivative modulus must be prime"
        )
    symmetry = derivative_slice_stabilizer(ambient)
    if gcd(prime, symmetry.order) != 1:
        raise KrennGradedMacaulayError(
            "graded derivative prime must be coprime to stabilizer order"
        )

    columns = bounded_derivative_orbit_columns(ambient, bound)
    relations = bounded_derivative_koszul_relations(ambient, bound)
    relation_rank = _sparse_rank_mod_p(
        tuple(
            {column: coefficient for column, coefficient in relation}
            for relation in relations
        ),
        prime,
    )
    source_rank_upper = len(columns) - relation_rank

    rows = tuple(sorted(
        {
            monomial
            for column in columns
            for monomial in column
        }.union({()}),
        key=lambda monomial: (len(monomial), monomial),
    ))
    row_index = {
        monomial: index for index, monomial in enumerate(rows)
    }
    basis: dict[int, dict[int, int]] = {}
    source_rank = 0
    for column in columns:
        indexed = {
            row_index[monomial]: coefficient
            for monomial, coefficient in column.items()
        }
        source_rank += int(_add_sparse_column(
            indexed, basis, prime
        ))
    target_independent = _add_sparse_column(
        {row_index[()]: 1}, basis, prime
    )
    augmented_rank = source_rank + int(target_independent)
    certified = (
        source_rank == source_rank_upper and target_independent
    )
    return DerivativeBoundedMacaulayResult(
        derivative_ambient_weight=ambient,
        total_degree_bound=bound,
        prime=prime,
        raw_character_pairs=len(
            bounded_derivative_certificate_pairs(ambient, bound)
        ),
        invariant_columns=len(columns),
        invariant_rows=len(rows),
        nonzero_entries=sum(len(column) for column in columns),
        exact_koszul_relations=len(relations),
        independent_relation_rank_mod_p=relation_rank,
        source_rank_upper_bound_over_q=source_rank_upper,
        source_rank_mod_p=source_rank,
        augmented_rank_mod_p=augmented_rank,
        target_in_span_mod_p=not target_independent,
        bounded_nonmembership_over_q_certified=certified,
    )


def _retained_degree_six_result(
    derivative_ambient_weight: int,
) -> DerivativeBoundedMacaulayResult:
    ambient = _validated_derivative(derivative_ambient_weight)
    row = RETAINED_DEGREE_SIX_RANKS[ambient]
    return DerivativeBoundedMacaulayResult(
        derivative_ambient_weight=ambient,
        total_degree_bound=6,
        prime=1_009,
        raw_character_pairs=row["raw_character_pairs"],
        invariant_columns=row["invariant_columns"],
        invariant_rows=row["invariant_rows"],
        nonzero_entries=row["nonzero_entries"],
        exact_koszul_relations=row["exact_koszul_relations"],
        independent_relation_rank_mod_p=(
            row["independent_relation_rank_mod_p"]
        ),
        source_rank_upper_bound_over_q=(
            row["source_rank_upper_bound_over_q"]
        ),
        source_rank_mod_p=row["source_rank_mod_p"],
        augmented_rank_mod_p=row["augmented_rank_mod_p"],
        target_in_span_mod_p=False,
        bounded_nonmembership_over_q_certified=False,
    )


def graded_derivative_macaulay_audit() -> dict:
    """Return retained exact degree-4/5 and modular degree-6 rows."""

    systems = []
    for ambient in DERIVATIVE_REPRESENTATIVES:
        grading = derivative_residual_grading(ambient)
        rows = [
            modular_bounded_derivative_preflight(
                ambient, degree, prime=1_009
            ).to_dict()
            for degree in (4, 5)
        ]
        rows.append(_retained_degree_six_result(ambient).to_dict())
        systems.append({
            "derivative_ambient_weight": ambient,
            "quotient_grading": {
                "ambient_rank": 9,
                "killed_rank": 1,
                "smith_diagonal": list(grading.smith_diagonal),
                "residual_rank": grading.residual_rank,
                "split_over_Z": True,
                "variable_blocks": grading.variable_blocks,
                "generator_blocks": grading.generator_blocks,
                "zero_character_variables":
                    grading.zero_character_variables,
                "zero_character_generators":
                    grading.zero_character_generators,
            },
            "bounded_rows": rows,
            "degree_six_unclosed_rank_gap": (
                rows[-1]["source_rank_upper_bound_over_q"]
                - rows[-1]["source_rank_mod_p"]
            ),
        })
    return json.loads(json.dumps({
        "schema": GRADED_DERIVATIVE_MACAULAY_AUDIT_SCHEMA,
        "systems": systems,
        "exact_conclusion": {
            "field": "Q",
            "certificate_character": [0] * 8,
            "stabilizer_averaging_order": 2,
            "symmetry_related_weights_equated": False,
            "no_nullstellensatz_identity_through_total_degree": 5,
            "reason": (
                "exact averaged Koszul relations give the Q source-rank "
                "upper bound, and modular source/augmented minors attain "
                "that bound and raise it by one"
            ),
        },
        "degree_six_status": {
            "field": "F_1009",
            "target_outside_modular_source_span": True,
            "rank_gaps": {"11": 123, "29": 117},
            "status": "reconnaissance-only",
        },
        "claim_boundary": {
            "degree_six_nonmembership_over_Q_certified": False,
            "full_slice_unit_or_proper_status_decided": False,
            "natural_chart_decided": False,
            "finite_affine_GHZ_membership_decided": False,
            "bounded_miss_is_global_proof": False,
        },
    }, allow_nan=False))


def verify_graded_derivative_macaulay_audit(
    payload: Mapping,
    *,
    full_degree_six_replay: bool = False,
) -> dict:
    """Strictly replay the aggregate audit and optionally both D6 ranks."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennGradedMacaulayError(
            "graded derivative audit is not strict JSON"
        ) from error
    expected = graded_derivative_macaulay_audit()
    if not strict_json_equal(normalized, expected):
        raise KrennGradedMacaulayError(
            "graded derivative audit failed exact replay"
        )
    if full_degree_six_replay:
        for system in normalized["systems"]:
            ambient = system["derivative_ambient_weight"]
            retained = system["bounded_rows"][-1]
            replay = modular_bounded_derivative_preflight(
                ambient, 6, prime=1_009
            ).to_dict()
            if not strict_json_equal(replay, retained):
                raise KrennGradedMacaulayError(
                    "graded derivative degree-six replay changed"
                )
    return normalized


__all__ = [
    "DerivativeBoundedMacaulayResult",
    "DerivativeResidualGrading",
    "DerivativeSliceStabilizer",
    "EXPECTED_BOUNDED_CENSUS",
    "EXPECTED_KOSZUL_RELATIONS",
    "EXPECTED_QUOTIENT_CENSUS",
    "GRADED_DERIVATIVE_MACAULAY_AUDIT_SCHEMA",
    "GRADED_DERIVATIVE_MACAULAY_SCHEMA",
    "KrennGradedMacaulayError",
    "RETAINED_DEGREE_SIX_RANKS",
    "RETAINED_PRIMES",
    "SUPPORTED_TOTAL_DEGREES",
    "bounded_derivative_certificate_pairs",
    "bounded_derivative_koszul_relations",
    "bounded_derivative_orbit_columns",
    "bounded_derivative_pair_orbits",
    "derivative_residual_grading",
    "derivative_slice_stabilizer",
    "graded_derivative_macaulay_audit",
    "modular_bounded_derivative_preflight",
    "verify_graded_derivative_macaulay_audit",
]
