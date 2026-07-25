r"""Exact derivative atlas for the natural localized Krenn chart.

The normalized natural chart ``(0,4,8)`` has one seed-defect equation,
equation 70.  It is multi-affine on a 12-variable repair core.  A retained
rational Nullstellensatz identity proves that the defect polynomial and its
12 partial derivatives generate the unit ideal.  Consequently its zero
hypersurface is smooth and the 12 derivative opens cover it exactly.

The strict ordered-seed stabilizer has two orbits on those derivatives, with
representatives ambient weights 11 and 29.  On a derivative open,

    f = x*d + h = 0,  d != 0,

introduce ``q*d-1`` and substitute ``x=-q*h`` in every other generator.
This gives two exact 129-variable affine ideals.  Unit ideals for both
exclude the entire natural chart; a proper ideal for either proves a finite
algebraic-complex witness exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from math import gcd
import re
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_ideals import (
    CHART_VARIABLE_COUNT,
    KrennLocalizedChartError,
    MIXED_EQUATIONS,
    SparseChartPolynomial,
    normalized_seed_chart,
    validate_singular_characteristic,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    generator_characters,
    ordered_seed_stabilizer,
    residual_torus_characters,
)


NATURAL_ORBIT_INDEX = 6
NATURAL_DEFECT_EQUATION = 70
NATURAL_CORE_AMBIENT_WEIGHTS = (
    11, 19, 29, 37, 47, 55, 65, 73, 88, 106, 113, 133,
)
EXPECTED_DERIVATIVE_ORBITS = (
    (11, 19, 65, 73, 88, 133),
    (29, 37, 47, 55, 106, 113),
)
DERIVATIVE_REPRESENTATIVES = (11, 29)

SMOOTHNESS_SCHEMA = "krenn-n6-d3-natural-defect-smoothness-v1"
DERIVATIVE_CHART_SCHEMA = (
    "krenn-n6-d3-natural-defect-eliminated-chart-v1"
)
SPARSE_DERIVATIVE_SLICE_SCHEMA = (
    "krenn-n6-d3-natural-defect-sparse-gauge-slice-v1"
)


class KrennDerivativeChartError(RuntimeError):
    """A smoothness certificate or derivative atlas replay failed."""


RATIONAL_SMOOTHNESS_MULTIPLIERS = (
    (
        "2*x19*x73*x88*x133+2/3*x11*x47*x113*x133"
        "+2*x29*x73*x88+1/2*x29*x47+1/2*x11*x65"
        "-x106*x113-x88*x133+1"
    ),
    (
        "-2*x11*x19*x73*x88*x133-2/3*x11^2*x47*x113*x133"
        "-3/2*x11*x29*x73*x88-x19*x37*x73*x88"
        "-1/6*x11*x37*x47*x113-1/6*x11*x19*x47*x133"
        "-1/2*x11*x29*x47-1/2*x11^2*x65"
        "-1/2*x29*x88*x106+1/2*x11*x106*x113"
        "+x11*x88*x133+x37*x88+x19*x106-3/2*x11"
    ),
    (
        "x19*x37*x73*x88*x113+1/6*x11*x37*x47*x113^2"
        "-x19^2*x73*x88*x133-1/3*x11*x19*x47*x113*x133"
        "-1/2*x29*x37*x55*x88-x19*x29*x73*x88"
        "-1/2*x19*x29*x47-1/2*x11*x19*x65"
        "+1/2*x37*x88*x113+1/2*x19*x106*x113"
        "+1/2*x29*x88+1/2*x11*x113-x19"
    ),
    (
        "-2*x19*x29*x73*x88*x133-2/3*x11*x29*x47*x113*x133"
        "-2*x29^2*x73*x88-1/2*x29^2*x47"
        "-1/2*x11*x29*x65+x29*x106*x113"
        "+1/2*x29*x88*x133+1/2*x37*x113-x29"
    ),
    (
        "-2*x19*x37*x73*x88*x133-2/3*x11*x37*x47*x113*x133"
        "+1/6*x11*x19*x47*x133^2-2*x29*x37*x73*x88"
        "-1/2*x29*x37*x47-1/2*x11*x37*x65"
        "+x37*x106*x113+x37*x88*x133"
        "-1/2*x19*x106*x133+1/2*x29*x106"
        "-1/2*x11*x133-x37"
    ),
    (
        "-x19*x73^2*x88-1/6*x11*x47*x73*x113"
        "-1/6*x11*x47*x55*x133-x73*x88^2*x133"
        "-1/2*x47*x106*x113-3/2*x73*x88-1/2*x47"
    ),
    (
        "x19*x73^2*x88*x113+1/6*x11*x47*x73*x113^2"
        "+x19*x55*x73*x88*x133"
        "+1/3*x11*x47*x55*x113*x133"
        "+1/2*x29*x55*x73*x88+1/2*x19*x65*x73*x88"
        "-1/2*x55*x106*x113-x55*x88*x133-1/2*x47*x113"
    ),
    (
        "-1/2*x29*x65*x73*x88+x73*x88*x113*x133"
        "-1/2*x73*x113"
    ),
    (
        "1/6*x11*x47*x55*x133^2+1/2*x37*x65*x73*x88"
        "+1/2*x73*x88*x133-1/2*x55*x106*x133"
        "-1/2*x65*x106-1/2*x47*x133"
    ),
    (
        "-x19*x73*x88*x106*x113-1/6*x11*x47*x106*x113^2"
        "+1/2*x29*x55*x88*x106-1/2*x11*x73*x88*x113"
        "+1/6*x11*x47*x113-1/2*x88*x106*x113-x88"
    ),
    (
        "-x19*x73*x88*x106*x133"
        "-1/3*x11*x47*x106*x113*x133"
        "-1/6*x11*x47*x88*x133^2-x29*x73*x88*x106"
        "-1/2*x11*x73*x88*x133+1/2*x106^2*x113"
        "+1/6*x11*x47*x133+3/2*x88*x106*x133-1/2*x106"
    ),
    (
        "-x19*x73*x88*x113*x133"
        "-1/6*x11*x47*x113^2*x133"
        "+1/2*x29*x55*x88*x133+1/2*x106*x113^2"
        "-1/2*x88*x113*x133+1/2*x113"
    ),
    (
        "-x19*x73*x88*x133^2"
        "-1/2*x11*x47*x113*x133^2"
        "-x29*x73*x88*x133+3/2*x106*x113*x133"
        "+x88*x133^2+x133"
    ),
)

AmbientPolynomial = dict[tuple[int, ...], Fraction]


def _clean(polynomial: AmbientPolynomial) -> AmbientPolynomial:
    return {
        monomial: coefficient
        for monomial, coefficient in polynomial.items()
        if coefficient
    }


def _add(
    *polynomials: AmbientPolynomial,
) -> AmbientPolynomial:
    result: AmbientPolynomial = {}
    for polynomial in polynomials:
        for monomial, coefficient in polynomial.items():
            result[monomial] = (
                result.get(monomial, Fraction(0)) + coefficient
            )
    return _clean(result)


def _scale(
    polynomial: AmbientPolynomial,
    coefficient: Fraction | int,
) -> AmbientPolynomial:
    coefficient = Fraction(coefficient)
    return _clean({
        monomial: coefficient * value
        for monomial, value in polynomial.items()
    })


def _multiply(
    *polynomials: AmbientPolynomial,
) -> AmbientPolynomial:
    result: AmbientPolynomial = {(): Fraction(1)}
    for polynomial in polynomials:
        product: AmbientPolynomial = {}
        for left, left_coefficient in result.items():
            for right, right_coefficient in polynomial.items():
                monomial = tuple(sorted((*left, *right)))
                product[monomial] = (
                    product.get(monomial, Fraction(0))
                    + left_coefficient * right_coefficient
                )
        result = _clean(product)
    return result


def _derivative(
    polynomial: AmbientPolynomial,
    variable: int,
) -> AmbientPolynomial:
    result: AmbientPolynomial = {}
    for monomial, coefficient in polynomial.items():
        multiplicity = monomial.count(variable)
        if not multiplicity:
            continue
        remaining = list(monomial)
        remaining.remove(variable)
        key = tuple(remaining)
        result[key] = (
            result.get(key, Fraction(0))
            + coefficient * multiplicity
        )
    return _clean(result)


_TERM_PATTERN = re.compile(r"[+-]?[^+-]+")
_FACTOR_PATTERN = re.compile(r"x(\d+)(?:\^(\d+))?")


def _parse_rational_polynomial(expression: str) -> AmbientPolynomial:
    expression = expression.replace(" ", "")
    result: AmbientPolynomial = {}
    for raw_term in _TERM_PATTERN.findall(expression):
        sign = -1 if raw_term.startswith("-") else 1
        term = raw_term.lstrip("+-")
        factors = term.split("*")
        coefficient = Fraction(sign)
        monomial = []
        for factor in factors:
            match = _FACTOR_PATTERN.fullmatch(factor)
            if match:
                variable = int(match.group(1))
                exponent = int(match.group(2) or "1")
                monomial.extend([variable] * exponent)
            else:
                try:
                    coefficient *= Fraction(factor)
                except (ValueError, ZeroDivisionError) as error:
                    raise KrennDerivativeChartError(
                        "a retained smoothness multiplier cannot be parsed"
                    ) from error
        key = tuple(sorted(monomial))
        result[key] = result.get(key, Fraction(0)) + coefficient
    return _clean(result)


def natural_defect_polynomial() -> AmbientPolynomial:
    """Return equation 70 in ambient-weight labels after seed normalization."""

    chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
    position = MIXED_EQUATIONS.index(NATURAL_DEFECT_EQUATION)
    polynomial = {}
    for coefficient, monomial in chart.mixed_generators[position].terms:
        ambient_monomial = tuple(sorted(
            chart.ambient_weight_index(variable)
            for variable in monomial
        ))
        polynomial[ambient_monomial] = Fraction(coefficient)
    if (
        set(variable for monomial in polynomial for variable in monomial)
        != set(NATURAL_CORE_AMBIENT_WEIGHTS)
        or len(polynomial) != 15
    ):
        raise KrennDerivativeChartError(
            "the natural defect polynomial changed"
        )
    return polynomial


def replay_smoothness_certificate() -> dict:
    """Verify the retained rational identity ``1 in <f, partials>``."""

    f = natural_defect_polynomial()
    derivatives = tuple(
        _derivative(f, variable)
        for variable in NATURAL_CORE_AMBIENT_WEIGHTS
    )
    multipliers = tuple(
        _parse_rational_polynomial(expression)
        for expression in RATIONAL_SMOOTHNESS_MULTIPLIERS
    )
    if len(multipliers) != 13:
        raise KrennDerivativeChartError(
            "smoothness certificate must have 13 multipliers"
        )
    replay = _add(*(
        _multiply(multiplier, generator)
        for multiplier, generator in zip(
            multipliers, (f, *derivatives), strict=True
        )
    ))
    if replay != {(): Fraction(1)}:
        raise KrennDerivativeChartError(
            "the rational smoothness identity failed exact replay"
        )
    digest = hashlib.sha256()
    for expression in RATIONAL_SMOOTHNESS_MULTIPLIERS:
        digest.update(expression.encode("ascii"))
        digest.update(b"\n")
    return {
        "field": "Q",
        "generators": 13,
        "multiplier_polynomials": len(multipliers),
        "identity": "sum_i multiplier_i * (f, partials)_i = 1",
        "exact_replay": True,
        "multiplier_sha256": digest.hexdigest(),
    }


def natural_derivative_orbits() -> tuple[tuple[int, ...], ...]:
    """Return the two exact strict-stabilizer orbits on the repair core."""

    chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
    symmetry = ordered_seed_stabilizer(NATURAL_ORBIT_INDEX)
    local_by_ambient = {
        ambient: local
        for local, ambient in enumerate(chart.remaining_weight_indices)
    }
    core = set(NATURAL_CORE_AMBIENT_WEIGHTS)
    unseen = set(core)
    orbits = []
    while unseen:
        start = min(unseen)
        local = local_by_ambient[start]
        orbit = {
            chart.ambient_weight_index(permutation[local])
            for permutation in symmetry.chart_variable_permutations
        }
        if not orbit.issubset(core):
            raise KrennDerivativeChartError(
                "the strict stabilizer moved a core variable outside the core"
            )
        orbits.append(tuple(sorted(orbit)))
        unseen.difference_update(orbit)
    result = tuple(sorted(orbits))
    if result != EXPECTED_DERIVATIVE_ORBITS:
        raise KrennDerivativeChartError(
            "natural derivative stabilizer orbits changed"
        )
    return result


def _local_polynomial_mapping(
    polynomial: SparseChartPolynomial,
) -> dict[tuple[int, ...], int]:
    return {
        monomial: coefficient
        for coefficient, monomial in polynomial.terms
    }


def _local_add(
    target: dict[tuple[int, ...], int],
    monomial: Sequence[int],
    coefficient: int,
) -> None:
    monomial = tuple(sorted(map(int, monomial)))
    total = target.get(monomial, 0) + int(coefficient)
    if total:
        target[monomial] = total
    else:
        target.pop(monomial, None)


def _local_multiply(
    left: Mapping[tuple[int, ...], int],
    right: Mapping[tuple[int, ...], int],
) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            _local_add(
                result,
                (*left_monomial, *right_monomial),
                left_coefficient * right_coefficient,
            )
    return result


def _extended_gcd(left: int, right: int) -> tuple[int, int, int]:
    """Return ``g,s,t`` with ``s*left+t*right=g=gcd(left,right)``."""

    left = int(left)
    right = int(right)
    old_remainder, remainder = abs(left), abs(right)
    old_left, current_left = 1, 0
    old_right, current_right = 0, 1
    while remainder:
        quotient = old_remainder // remainder
        old_remainder, remainder = (
            remainder,
            old_remainder - quotient * remainder,
        )
        old_left, current_left = (
            current_left,
            old_left - quotient * current_left,
        )
        old_right, current_right = (
            current_right,
            old_right - quotient * current_right,
        )
    return (
        old_remainder,
        old_left * (-1 if left < 0 else 1),
        old_right * (-1 if right < 0 else 1),
    )


def _primitive_bezout_cocharacter(
    character: Sequence[int],
) -> tuple[int, ...]:
    """Return an integral cocharacter pairing to one with a primitive weight."""

    coefficients: list[int] = []
    common_divisor = 0
    for raw_value in character:
        value = int(raw_value)
        new_divisor, old_multiplier, value_multiplier = _extended_gcd(
            common_divisor, value
        )
        coefficients = [
            old_multiplier * coefficient for coefficient in coefficients
        ]
        coefficients.append(value_multiplier)
        common_divisor = new_divisor
    if common_divisor != 1:
        raise KrennDerivativeChartError(
            "the selected derivative character is not primitive"
        )
    result = tuple(coefficients)
    if sum(
        int(value) * coefficient
        for value, coefficient in zip(character, result, strict=True)
    ) != 1:
        raise KrennDerivativeChartError(
            "the primitive derivative Bezout identity failed"
        )
    return result


def _local_monomial_character(
    monomial: Sequence[int],
    characters: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    return tuple(
        sum(
            int(characters[variable][coordinate])
            for variable in monomial
        )
        for coordinate in range(9)
    )


def _natural_defect_derivative(
    derivative_ambient_weight: int,
) -> tuple[int, SparseChartPolynomial]:
    """Return the natural-chart variable and exact defect derivative."""

    derivative_ambient_weight = int(derivative_ambient_weight)
    if derivative_ambient_weight not in DERIVATIVE_REPRESENTATIVES:
        raise KrennDerivativeChartError(
            "only strict derivative representatives 11 and 29 are built"
        )
    chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
    try:
        local_variable = chart.remaining_weight_indices.index(
            derivative_ambient_weight
        )
    except ValueError as error:
        raise KrennDerivativeChartError(
            "the derivative ambient weight is not in the natural chart"
        ) from error
    defect_position = chart.generator_labels.index(
        ("mixed", NATURAL_DEFECT_EQUATION)
    )
    derivative: dict[tuple[int, ...], int] = {}
    for coefficient, monomial in chart.generators[
        defect_position
    ].terms:
        multiplicity = monomial.count(local_variable)
        if not multiplicity:
            continue
        if multiplicity != 1:
            raise KrennDerivativeChartError(
                "the natural defect is not multi-affine"
            )
        reduced_monomial = list(monomial)
        reduced_monomial.remove(local_variable)
        _local_add(derivative, reduced_monomial, coefficient)
    if not derivative:
        raise KrennDerivativeChartError(
            "the selected natural defect derivative vanished"
        )
    return local_variable, SparseChartPolynomial.from_mapping(derivative)


@dataclass(frozen=True)
class SparseDerivativeGaugeSlice:
    """Natural equations plus ``d=1``, representing the open ``d!=0``."""

    derivative_ambient_weight: int
    derivative_original_local_variable: int
    derivative_character: tuple[int, ...]
    normalizing_cocharacter: tuple[int, ...]
    derivative_polynomial: SparseChartPolynomial
    generator_labels: tuple[tuple[str, int], ...]
    generators: tuple[SparseChartPolynomial, ...]
    schema: str = SPARSE_DERIVATIVE_SLICE_SCHEMA

    def __post_init__(self) -> None:
        chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
        expected_variable, expected_derivative = (
            _natural_defect_derivative(self.derivative_ambient_weight)
        )
        if (
            self.schema != SPARSE_DERIVATIVE_SLICE_SCHEMA
            or self.derivative_ambient_weight
            not in DERIVATIVE_REPRESENTATIVES
            or not 0 <= self.derivative_original_local_variable
            < CHART_VARIABLE_COUNT
            or chart.remaining_weight_indices[
                self.derivative_original_local_variable
            ] != self.derivative_ambient_weight
            or self.derivative_original_local_variable
            != expected_variable
            or self.derivative_polynomial != expected_derivative
            or len(self.derivative_character) != 9
            or len(self.normalizing_cocharacter) != 9
            or len(self.generator_labels) != 730
            or len(self.generators) != 730
            or self.generator_labels[:-1] != chart.generator_labels
            or self.generators[:-1] != chart.generators
            or self.generator_labels[-1]
            != ("derivative-gauge", self.derivative_ambient_weight)
        ):
            raise KrennDerivativeChartError(
                "a sparse derivative gauge slice failed metadata replay"
            )
        expected_gauge = {
            monomial: coefficient
            for coefficient, monomial in self.derivative_polynomial.terms
        }
        _local_add(expected_gauge, (), -1)
        if (
            self.generators[-1]
            != SparseChartPolynomial.from_mapping(expected_gauge)
            or self.term_count != 10_942
            or self.maximum_degree != 4
        ):
            raise KrennDerivativeChartError(
                "the sparse derivative gauge equations changed"
            )

        variable_characters = residual_torus_characters(
            NATURAL_ORBIT_INDEX
        )
        defect_position = chart.generator_labels.index(
            ("mixed", NATURAL_DEFECT_EQUATION)
        )
        defect_character = generator_characters(
            NATURAL_ORBIT_INDEX
        )[defect_position]
        expected_character = tuple(
            generator_component - variable_component
            for generator_component, variable_component in zip(
                defect_character,
                variable_characters[
                    self.derivative_original_local_variable
                ],
                strict=True,
            )
        )
        term_characters = {
            _local_monomial_character(monomial, variable_characters)
            for _coefficient, monomial in self.derivative_polynomial.terms
        }
        primitive_divisor = 0
        for component in self.derivative_character:
            primitive_divisor = gcd(primitive_divisor, abs(component))
        pairing = sum(
            character * cocharacter
            for character, cocharacter in zip(
                self.derivative_character,
                self.normalizing_cocharacter,
                strict=True,
            )
        )
        if (
            self.derivative_character != expected_character
            or term_characters != {self.derivative_character}
            or primitive_divisor != 1
            or pairing != 1
        ):
            raise KrennDerivativeChartError(
                "the derivative is not a primitively split semi-invariant"
            )

    @property
    def retained_original_chart_variables(self) -> tuple[int, ...]:
        return tuple(range(CHART_VARIABLE_COUNT))

    @property
    def variable_count(self) -> int:
        return CHART_VARIABLE_COUNT

    @property
    def term_count(self) -> int:
        return sum(polynomial.term_count for polynomial in self.generators)

    @property
    def maximum_degree(self) -> int:
        return max(polynomial.degree for polynomial in self.generators)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.derivative_ambient_weight}|"
                f"{self.derivative_original_local_variable}|"
                f"{self.derivative_character}|"
                f"{self.normalizing_cocharacter}\n"
            ).encode("ascii")
        )
        for label, polynomial in zip(
            self.generator_labels, self.generators, strict=True
        ):
            digest.update(f"{label[0]}:{label[1]}|".encode("ascii"))
            for coefficient, monomial in polynomial.terms:
                digest.update(
                    f"{coefficient}:{','.join(map(str, monomial))};".encode(
                        "ascii"
                    )
                )
            digest.update(b"\n")
        return digest.hexdigest()

    def summary(self) -> dict:
        return {
            "schema": self.schema,
            "derivative_ambient_weight":
                self.derivative_ambient_weight,
            "derivative_original_local_variable":
                self.derivative_original_local_variable,
            "retained_original_chart_variables": CHART_VARIABLE_COUNT,
            "variables": self.variable_count,
            "generators": len(self.generators),
            "sparse_terms": self.term_count,
            "maximum_degree": self.maximum_degree,
            "derivative_terms": self.derivative_polynomial.term_count,
            "residual_character": list(self.derivative_character),
            "character_gcd": 1,
            "normalizing_cocharacter": list(
                self.normalizing_cocharacter
            ),
            "character_cocharacter_pairing": 1,
            "natural_generators_homogeneous": True,
            "open_slice_equivalence": (
                "for every field-valued point with d!=0, the residual "
                "torus cocharacter scales d to 1; conversely d=1 implies "
                "d!=0"
            ),
            "sha256": self.fingerprint(),
            "claim_boundary": {
                "unit_ideal_decided": False,
                "proper_ideal_decided": False,
                "natural_chart_decided": False,
            },
        }


@dataclass(frozen=True)
class EliminatedDerivativeChart:
    """One derivative-localized natural chart after exact substitution."""

    eliminated_ambient_weight: int
    eliminated_original_local_variable: int
    retained_original_chart_variables: tuple[int, ...]
    generator_labels: tuple[tuple[str, int], ...]
    generators: tuple[SparseChartPolynomial, ...]
    schema: str = DERIVATIVE_CHART_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != DERIVATIVE_CHART_SCHEMA
            or self.eliminated_ambient_weight
            not in DERIVATIVE_REPRESENTATIVES
            or len(self.retained_original_chart_variables) != 128
            or self.eliminated_original_local_variable
            in self.retained_original_chart_variables
            or tuple(sorted((
                self.eliminated_original_local_variable,
                *self.retained_original_chart_variables,
            ))) != tuple(range(CHART_VARIABLE_COUNT))
            or len(self.generator_labels) != 729
            or len(self.generators) != 729
        ):
            raise KrennDerivativeChartError(
                "an eliminated derivative chart failed metadata replay"
            )

    @property
    def term_count(self) -> int:
        return sum(polynomial.term_count for polynomial in self.generators)

    @property
    def maximum_degree(self) -> int:
        return max(polynomial.degree for polynomial in self.generators)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.eliminated_ambient_weight}|"
                f"{self.eliminated_original_local_variable}|"
                f"{self.retained_original_chart_variables}\n"
            ).encode("ascii")
        )
        for label, polynomial in zip(
            self.generator_labels, self.generators, strict=True
        ):
            digest.update(f"{label[0]}:{label[1]}|".encode("ascii"))
            for coefficient, monomial in polynomial.terms:
                digest.update(
                    f"{coefficient}:{','.join(map(str, monomial))};".encode(
                        "ascii"
                    )
                )
            digest.update(b"\n")
        return digest.hexdigest()

    def summary(self) -> dict:
        return {
            "schema": self.schema,
            "eliminated_ambient_weight": self.eliminated_ambient_weight,
            "eliminated_original_local_variable":
                self.eliminated_original_local_variable,
            "retained_original_chart_variables": len(
                self.retained_original_chart_variables
            ),
            "derivative_inverse_variable": 128,
            "total_variables": 129,
            "total_generators": len(self.generators),
            "sparse_terms": self.term_count,
            "maximum_degree": self.maximum_degree,
            "recovery": (
                "eliminated weight = - derivative_inverse * constant_part"
            ),
            "sha256": self.fingerprint(),
        }


def _remap_mapping(
    polynomial: Mapping[tuple[int, ...], int],
    remap: Mapping[int, int],
) -> dict[tuple[int, ...], int]:
    result = {}
    for monomial, coefficient in polynomial.items():
        _local_add(
            result,
            tuple(remap[variable] for variable in monomial),
            coefficient,
        )
    return result


def eliminated_derivative_chart(
    eliminated_ambient_weight: int,
) -> EliminatedDerivativeChart:
    """Construct one of the two strict-symmetry derivative charts."""

    eliminated_ambient_weight = int(eliminated_ambient_weight)
    if eliminated_ambient_weight not in DERIVATIVE_REPRESENTATIVES:
        raise KrennDerivativeChartError(
            "only strict derivative representatives 11 and 29 are built"
        )
    chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
    try:
        eliminated = chart.remaining_weight_indices.index(
            eliminated_ambient_weight
        )
    except ValueError as error:
        raise KrennDerivativeChartError(
            "the eliminated ambient weight is not in the natural chart"
        ) from error
    retained = tuple(
        variable for variable in range(CHART_VARIABLE_COUNT)
        if variable != eliminated
    )
    remap = {
        original: reduced for reduced, original in enumerate(retained)
    }
    defect_position = MIXED_EQUATIONS.index(NATURAL_DEFECT_EQUATION)
    defect = _local_polynomial_mapping(
        chart.mixed_generators[defect_position]
    )
    derivative = {}
    constant_part = {}
    for monomial, coefficient in defect.items():
        if eliminated in monomial:
            reduced_monomial = list(monomial)
            reduced_monomial.remove(eliminated)
            _local_add(derivative, reduced_monomial, coefficient)
        else:
            _local_add(constant_part, monomial, coefficient)
    if not derivative or not constant_part:
        raise KrennDerivativeChartError(
            "the defect did not split as x*derivative+constant"
        )
    derivative_new = _remap_mapping(derivative, remap)
    constant_new = _remap_mapping(constant_part, remap)

    labels = []
    generators = []
    for label, polynomial in zip(
        chart.generator_labels, chart.generators, strict=True
    ):
        if label == ("mixed", NATURAL_DEFECT_EQUATION):
            continue
        raw = _local_polynomial_mapping(polynomial)
        without = {}
        coefficient_of_eliminated = {}
        for monomial, coefficient in raw.items():
            if eliminated in monomial:
                reduced_monomial = list(monomial)
                reduced_monomial.remove(eliminated)
                _local_add(
                    coefficient_of_eliminated,
                    reduced_monomial,
                    coefficient,
                )
            else:
                _local_add(without, monomial, coefficient)
        substituted = _remap_mapping(without, remap)
        if coefficient_of_eliminated:
            coefficient_new = _remap_mapping(
                coefficient_of_eliminated, remap
            )
            correction = _local_multiply(
                constant_new, coefficient_new
            )
            for monomial, coefficient in correction.items():
                _local_add(
                    substituted,
                    (*monomial, 128),
                    -coefficient,
                )
        labels.append(label)
        generators.append(SparseChartPolynomial.from_mapping(substituted))

    derivative_localizer = {}
    for monomial, coefficient in derivative_new.items():
        _local_add(
            derivative_localizer, (*monomial, 128), coefficient
        )
    _local_add(derivative_localizer, (), -1)
    labels.append(
        ("derivative-localizer", eliminated_ambient_weight)
    )
    generators.append(
        SparseChartPolynomial.from_mapping(derivative_localizer)
    )
    return EliminatedDerivativeChart(
        eliminated_ambient_weight=eliminated_ambient_weight,
        eliminated_original_local_variable=eliminated,
        retained_original_chart_variables=retained,
        generator_labels=tuple(labels),
        generators=tuple(generators),
    )


def sparse_derivative_gauge_slice(
    derivative_ambient_weight: int,
) -> SparseDerivativeGaugeSlice:
    """Build the exact low-degree slice equivalent to the derivative open."""

    chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
    local_variable, derivative = _natural_defect_derivative(
        derivative_ambient_weight
    )
    variable_characters = residual_torus_characters(NATURAL_ORBIT_INDEX)
    all_generator_characters = generator_characters(NATURAL_ORBIT_INDEX)
    defect_position = chart.generator_labels.index(
        ("mixed", NATURAL_DEFECT_EQUATION)
    )
    derivative_character = tuple(
        generator_component - variable_component
        for generator_component, variable_component in zip(
            all_generator_characters[defect_position],
            variable_characters[local_variable],
            strict=True,
        )
    )
    term_characters = {
        _local_monomial_character(monomial, variable_characters)
        for _coefficient, monomial in derivative.terms
    }
    if term_characters != {derivative_character}:
        raise KrennDerivativeChartError(
            "the selected defect derivative is not a semi-invariant"
        )
    cocharacter = _primitive_bezout_cocharacter(derivative_character)
    gauge_polynomial = {
        monomial: coefficient
        for coefficient, monomial in derivative.terms
    }
    _local_add(gauge_polynomial, (), -1)
    return SparseDerivativeGaugeSlice(
        derivative_ambient_weight=int(derivative_ambient_weight),
        derivative_original_local_variable=local_variable,
        derivative_character=derivative_character,
        normalizing_cocharacter=cocharacter,
        derivative_polynomial=derivative,
        generator_labels=(
            *chart.generator_labels,
            ("derivative-gauge", int(derivative_ambient_weight)),
        ),
        generators=(
            *chart.generators,
            SparseChartPolynomial.from_mapping(gauge_polynomial),
        ),
    )


def natural_derivative_atlas_audit() -> dict:
    """Return a fail-closed audit of smoothness and the two-chart reduction."""

    certificate = replay_smoothness_certificate()
    derivative_orbits = natural_derivative_orbits()
    charts = tuple(
        eliminated_derivative_chart(variable)
        for variable in DERIVATIVE_REPRESENTATIVES
    )
    sparse_slices = tuple(
        sparse_derivative_gauge_slice(variable)
        for variable in DERIVATIVE_REPRESENTATIVES
    )
    return json.loads(json.dumps({
        "schema": SMOOTHNESS_SCHEMA,
        "natural_chart_seed": [0, 4, 8],
        "defect_equation": NATURAL_DEFECT_EQUATION,
        "repair_core_ambient_weights": list(
            NATURAL_CORE_AMBIENT_WEIGHTS
        ),
        "smoothness_certificate": certificate,
        "conclusion": (
            "the twelve derivative opens cover the natural defect "
            "hypersurface exactly over characteristic zero"
        ),
        "strict_stabilizer": {
            "order": ordered_seed_stabilizer(
                NATURAL_ORBIT_INDEX
            ).order,
            "derivative_orbits": [
                list(orbit) for orbit in derivative_orbits
            ],
            "representatives": list(DERIVATIVE_REPRESENTATIVES),
            "symmetry_related_weights_equated": False,
        },
        "eliminated_charts": [chart.summary() for chart in charts],
        "sparse_gauge_slices": [
            chart.summary() for chart in sparse_slices
        ],
        "preferred_next_formulation": {
            "kind": "primitive residual-character gauge slice",
            "reason": (
                "retains the sparse degree-four natural chart and appends "
                "derivative-1 instead of creating degree-six substitution "
                "terms"
            ),
            "existence_equivalent_to_derivative_open": True,
        },
        "decision_rule": {
            "both_verified_unit": "excludes the full natural chart",
            "either_verified_proper": (
                "proves a finite point exists over algebraic closure of Q"
            ),
        },
        "claim_boundary": {
            "natural_chart_decided": False,
            "full_eight_chart_cover_decided": False,
            "unit_ideal_certificates_emitted": 0,
            "proper_ideal_certificates_emitted": 0,
        },
    }))


def _singular_polynomial_text(
    polynomial: SparseChartPolynomial,
    characteristic: int,
) -> str:
    pieces = []
    for raw_coefficient, monomial in polynomial.terms:
        coefficient = (
            raw_coefficient % characteristic
            if characteristic else raw_coefficient
        )
        factors = "*".join(f"x{variable}" for variable in monomial)
        if not factors:
            piece = str(coefficient)
        elif coefficient == 1:
            piece = factors
        elif coefficient == -1:
            piece = f"-{factors}"
        else:
            piece = f"{coefficient}*{factors}"
        pieces.append(piece)
    if characteristic:
        return "+".join(pieces)
    result = pieces[0]
    for piece in pieces[1:]:
        result += piece if piece.startswith("-") else f"+{piece}"
    return result


def singular_eliminated_derivative_script(
    eliminated_ambient_weight: int,
    *,
    characteristic: int = 31,
    algorithm: str = "std",
) -> str:
    """Export one exact derivative-eliminated chart to Singular."""

    try:
        characteristic = validate_singular_characteristic(characteristic)
    except KrennLocalizedChartError as error:
        raise KrennDerivativeChartError(
            "Singular characteristic must be zero or prime"
        ) from error
    if algorithm not in ("std", "slimgb"):
        raise KrennDerivativeChartError(
            "Singular algorithm must be std or slimgb"
        )
    chart = eliminated_derivative_chart(eliminated_ambient_weight)
    names = ",".join(f"x{index}" for index in range(129))
    generators = ",\n".join(
        _singular_polynomial_text(polynomial, characteristic)
        for polynomial in chart.generators
    )
    return "\n".join((
        "// Exact derivative-eliminated natural Krenn chart.",
        f"// schema={DERIVATIVE_CHART_SCHEMA}",
        (
            "// eliminated_ambient_weight="
            f"{chart.eliminated_ambient_weight}"
        ),
        "// x128 is the inverse of the selected defect derivative.",
        f"ring r={characteristic},({names}),dp;",
        f"ideal I={generators};",
        'print("KRENN_DERIVATIVE_PARSE_OK");',
        (
            'print("eliminated_ambient_weight='
            f'{chart.eliminated_ambient_weight}");'
        ),
        f'print("characteristic={characteristic}");',
        f'print("algorithm={algorithm}");',
        'print("variables="+string(nvars(basering)));',
        'print("generators="+string(size(I)));',
        "int started=timer;",
        f"ideal G={algorithm}(I);",
        "int elapsed=timer-started;",
        "poly unit_remainder=reduce(1,G);",
        "int is_unit=0;",
        "if (unit_remainder==0) { is_unit=1; }",
        'print("KRENN_DERIVATIVE_GROEBNER_DONE");',
        'print("timer_ticks="+string(elapsed));',
        'print("basis_size="+string(size(G)));',
        'print("unit_ideal="+string(is_unit));',
        "exit;",
        "",
    ))


def singular_sparse_derivative_gauge_script(
    derivative_ambient_weight: int,
    *,
    characteristic: int = 31,
    algorithm: str = "std",
) -> str:
    """Export the exact sparse ``d=1`` derivative slice to Singular."""

    try:
        characteristic = validate_singular_characteristic(characteristic)
    except KrennLocalizedChartError as error:
        raise KrennDerivativeChartError(
            "Singular characteristic must be zero or prime"
        ) from error
    if algorithm not in ("std", "slimgb"):
        raise KrennDerivativeChartError(
            "Singular algorithm must be std or slimgb"
        )
    chart = sparse_derivative_gauge_slice(derivative_ambient_weight)
    names = ",".join(
        f"x{index}" for index in range(chart.variable_count)
    )
    generators = ",\n".join(
        _singular_polynomial_text(polynomial, characteristic)
        for polynomial in chart.generators
    )
    return "\n".join((
        "// Exact sparse derivative-gauge natural Krenn chart.",
        f"// schema={SPARSE_DERIVATIVE_SLICE_SCHEMA}",
        (
            "// derivative_ambient_weight="
            f"{chart.derivative_ambient_weight}"
        ),
        (
            "// primitive_derivative_character="
            f"{chart.derivative_character}"
        ),
        "// the final generator is d-1; no variable was eliminated.",
        f"ring r={characteristic},({names}),dp;",
        f"ideal I={generators};",
        'print("KRENN_SPARSE_DERIVATIVE_PARSE_OK");',
        (
            'print("derivative_ambient_weight='
            f'{chart.derivative_ambient_weight}");'
        ),
        f'print("characteristic={characteristic}");',
        f'print("algorithm={algorithm}");',
        'print("variables="+string(nvars(basering)));',
        'print("generators="+string(size(I)));',
        "int started=timer;",
        f"ideal G={algorithm}(I);",
        "int elapsed=timer-started;",
        "poly unit_remainder=reduce(1,G);",
        "int is_unit=0;",
        "if (unit_remainder==0) { is_unit=1; }",
        'print("KRENN_SPARSE_DERIVATIVE_GROEBNER_DONE");',
        'print("timer_ticks="+string(elapsed));',
        'print("basis_size="+string(size(G)));',
        'print("unit_ideal="+string(is_unit));',
        "exit;",
        "",
    ))
