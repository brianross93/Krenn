"""Exact algebraic witnesses for the bounded Krenn counterexample campaign.

The existing :mod:`witness` and :mod:`tensor_map` modules intentionally work
over ``Q``.  A numerical counterexample search may instead suggest weights in
``Q(i)``, a small cyclotomic field, or another low-degree number field.  This
module provides a small, dependency-light exact layer for those candidates.

There are three deliberately separate stages:

* :class:`NumberField` and :class:`FieldElement` implement immutable exact
  arithmetic in a canonical power-basis presentation ``Q[theta]/(f)``.
* :class:`ExactCandidateWitness` is replayed by two full evaluators.  The
  primary evaluator uses the repository's sparse equation system, while the
  independent evaluator filters edge subsets for perfect matchings and
  rederives variable indices locally.
* :func:`reconstruct_complex_candidate` makes deterministic, explicitly
  bounded numerical recognition attempts.  Recognition never proves
  anything: only zero residuals from both exact evaluators certify a finite
  affine witness.

The field implementation is reviewed for degrees at most four.  Quadratic
and cubic irreducibility is decided exactly by the rational-root criterion.
Quartics are accepted only when they are a listed cyclotomic polynomial or
have an independently checkable irreducible reduction modulo a small prime.
This may reject some valid quartic fields, but never silently treats an
uncertified quotient with zero divisors as a number field.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import cmath
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import json
from math import gcd, isfinite, isqrt, lcm, pi
from numbers import Integral
from typing import Iterable, Mapping, Sequence

import numpy as np

from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    generate_sparse_system,
    variable_count,
    variable_key,
)


NUMBER_FIELD_SCHEMA = "krenn-counterexample-number-field-v1"
FIELD_ELEMENT_SCHEMA = "krenn-counterexample-field-element-v1"
EXACT_WITNESS_SCHEMA = "krenn-counterexample-exact-witness-v1"
EXACT_VERIFICATION_SCHEMA = (
    "krenn-counterexample-exact-verification-v1"
)
RECONSTRUCTION_SCHEMA = "krenn-counterexample-reconstruction-v1"
RECONSTRUCTION_BOUNDS_SCHEMA = (
    "krenn-counterexample-reconstruction-bounds-v1"
)

MAX_FIELD_DEGREE = 4
MAX_PRIMITIVE_POLYNOMIAL_HEIGHT = 1_000_000
_IRREDUCIBILITY_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
)


class KrennCounterexampleExactError(ValueError):
    """An exact field, witness, replay, or reconstruction is malformed."""


class KrennReconstructionError(KrennCounterexampleExactError):
    """A bounded numerical reconstruction request is malformed."""


def _exact_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise KrennCounterexampleExactError(
            f"{label} must be an exact integer"
        )
    return int(value)


def _exact_fraction(value, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise KrennCounterexampleExactError(
            f"{label} must be an exact integer or Fraction"
        )
    return Fraction(value)


def _fraction_payload(value: Fraction | int) -> dict:
    exact = Fraction(value)
    return {
        "numerator": exact.numerator,
        "denominator": exact.denominator,
    }


def _fraction_from_payload(payload: Mapping, label: str) -> Fraction:
    if not isinstance(payload, Mapping) or set(payload) != {
        "numerator",
        "denominator",
    }:
        raise KrennCounterexampleExactError(
            f"{label} is not a canonical rational payload"
        )
    numerator = _exact_int(payload["numerator"], f"{label} numerator")
    denominator = _exact_int(
        payload["denominator"], f"{label} denominator"
    )
    if denominator <= 0:
        raise KrennCounterexampleExactError(
            f"{label} denominator must be positive"
        )
    value = Fraction(numerator, denominator)
    if (
        value.numerator != numerator
        or value.denominator != denominator
    ):
        raise KrennCounterexampleExactError(
            f"{label} is not reduced to canonical form"
        )
    return value


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennCounterexampleExactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise KrennCounterexampleExactError(
        f"non-finite JSON constant {value!r} is forbidden"
    )


def _loads_json(text: str) -> Mapping:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise KrennCounterexampleExactError(
            "could not decode canonical exact JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise KrennCounterexampleExactError(
            "canonical exact JSON must contain an object"
        )
    return payload


def canonical_json_dumps(payload: Mapping) -> str:
    """Return deterministic JSON while rejecting NaN and infinity."""

    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise KrennCounterexampleExactError(
            "payload is not canonical finite JSON data"
        ) from error


def _poly_trim(
    coefficients: Sequence[Fraction | int],
) -> tuple[Fraction, ...]:
    values = [Fraction(value) for value in coefficients]
    while values and values[-1] == 0:
        values.pop()
    return tuple(values)


def _poly_add(
    left: Sequence[Fraction | int],
    right: Sequence[Fraction | int],
) -> tuple[Fraction, ...]:
    width = max(len(left), len(right))
    result = [Fraction(0)] * width
    for index in range(width):
        result[index] = (
            Fraction(left[index]) if index < len(left) else Fraction(0)
        ) + (
            Fraction(right[index])
            if index < len(right)
            else Fraction(0)
        )
    return _poly_trim(result)


def _poly_sub(
    left: Sequence[Fraction | int],
    right: Sequence[Fraction | int],
) -> tuple[Fraction, ...]:
    return _poly_add(left, tuple(-Fraction(value) for value in right))


def _poly_mul(
    left: Sequence[Fraction | int],
    right: Sequence[Fraction | int],
) -> tuple[Fraction, ...]:
    left = _poly_trim(left)
    right = _poly_trim(right)
    if not left or not right:
        return ()
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += (
                left_value * right_value
            )
    return _poly_trim(result)


def _poly_divmod(
    numerator: Sequence[Fraction | int],
    denominator: Sequence[Fraction | int],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    numerator_values = list(_poly_trim(numerator))
    denominator_values = _poly_trim(denominator)
    if not denominator_values:
        raise ZeroDivisionError("polynomial division by zero")
    if len(numerator_values) < len(denominator_values):
        return (), tuple(numerator_values)
    quotient = [Fraction(0)] * (
        len(numerator_values) - len(denominator_values) + 1
    )
    denominator_lead = denominator_values[-1]
    while (
        numerator_values
        and len(numerator_values) >= len(denominator_values)
    ):
        offset = len(numerator_values) - len(denominator_values)
        factor = numerator_values[-1] / denominator_lead
        quotient[offset] += factor
        for index, value in enumerate(denominator_values):
            numerator_values[offset + index] -= factor * value
        while numerator_values and numerator_values[-1] == 0:
            numerator_values.pop()
    return _poly_trim(quotient), _poly_trim(numerator_values)


def _poly_mod(
    numerator: Sequence[Fraction | int],
    modulus: Sequence[Fraction | int],
) -> tuple[Fraction, ...]:
    return _poly_divmod(numerator, modulus)[1]


def _poly_extended_gcd(
    left: Sequence[Fraction | int],
    right: Sequence[Fraction | int],
) -> tuple[
    tuple[Fraction, ...],
    tuple[Fraction, ...],
    tuple[Fraction, ...],
]:
    old_remainder = _poly_trim(left)
    remainder = _poly_trim(right)
    old_left_coefficient: tuple[Fraction, ...] = (Fraction(1),)
    left_coefficient: tuple[Fraction, ...] = ()
    old_right_coefficient: tuple[Fraction, ...] = ()
    right_coefficient: tuple[Fraction, ...] = (Fraction(1),)
    while remainder:
        quotient, next_remainder = _poly_divmod(
            old_remainder, remainder
        )
        old_remainder, remainder = remainder, next_remainder
        old_left_coefficient, left_coefficient = (
            left_coefficient,
            _poly_sub(
                old_left_coefficient,
                _poly_mul(quotient, left_coefficient),
            ),
        )
        old_right_coefficient, right_coefficient = (
            right_coefficient,
            _poly_sub(
                old_right_coefficient,
                _poly_mul(quotient, right_coefficient),
            ),
        )
    if not old_remainder:
        return (), (), ()
    scale = old_remainder[-1]
    return (
        tuple(value / scale for value in old_remainder),
        tuple(value / scale for value in old_left_coefficient),
        tuple(value / scale for value in old_right_coefficient),
    )


def _poly_evaluate_exact(
    coefficients: Sequence[Fraction | int], value: Fraction
) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * value + Fraction(coefficient)
    return result


def _poly_evaluate_complex(
    coefficients: Sequence[Fraction | int], value: complex
) -> complex:
    result = 0j
    for coefficient in reversed(coefficients):
        result = result * value + float(Fraction(coefficient))
    return result


def _positive_divisors(value: int) -> tuple[int, ...]:
    value = abs(int(value))
    if value == 0:
        return (0,)
    small = []
    large = []
    for divisor in range(1, isqrt(value) + 1):
        if value % divisor:
            continue
        small.append(divisor)
        quotient = value // divisor
        if quotient != divisor:
            large.append(quotient)
    return tuple((*small, *reversed(large)))


def _primitive_integer_polynomial(
    polynomial: Sequence[Fraction],
) -> tuple[int, ...]:
    denominator_lcm = 1
    for coefficient in polynomial:
        denominator_lcm = lcm(
            denominator_lcm, coefficient.denominator
        )
    integers = [
        coefficient.numerator
        * (denominator_lcm // coefficient.denominator)
        for coefficient in polynomial
    ]
    common = 0
    for value in integers:
        common = gcd(common, abs(value))
    if common:
        integers = [value // common for value in integers]
    if integers[-1] < 0:
        integers = [-value for value in integers]
    if max(map(abs, integers), default=0) > (
        MAX_PRIMITIVE_POLYNOMIAL_HEIGHT
    ):
        raise KrennCounterexampleExactError(
            "primitive minimal-polynomial height exceeds the reviewed bound"
        )
    return tuple(integers)


def _has_rational_root(polynomial: Sequence[Fraction]) -> bool:
    primitive = _primitive_integer_polynomial(polynomial)
    if primitive[0] == 0:
        return True
    numerators = _positive_divisors(primitive[0])
    denominators = _positive_divisors(primitive[-1])
    for numerator in numerators:
        for denominator in denominators:
            for sign in (-1, 1):
                candidate = Fraction(sign * numerator, denominator)
                if _poly_evaluate_exact(polynomial, candidate) == 0:
                    return True
    return False


def _fraction_mod_prime(value: Fraction, prime: int) -> int | None:
    denominator = value.denominator % prime
    if denominator == 0:
        return None
    return value.numerator * pow(denominator, -1, prime) % prime


def _poly_mod_prime_divides(
    polynomial: Sequence[int],
    divisor: Sequence[int],
    prime: int,
) -> bool:
    remainder = [int(value) % prime for value in polynomial]
    divisor = tuple(int(value) % prime for value in divisor)
    while remainder and remainder[-1] == 0:
        remainder.pop()
    if not divisor or divisor[-1] == 0:
        return False
    while len(remainder) >= len(divisor):
        offset = len(remainder) - len(divisor)
        factor = remainder[-1] * pow(divisor[-1], -1, prime) % prime
        for index, value in enumerate(divisor):
            remainder[offset + index] = (
                remainder[offset + index] - factor * value
            ) % prime
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return not remainder


def _irreducible_mod_prime(
    polynomial: Sequence[Fraction], prime: int
) -> bool:
    reduced = []
    for coefficient in polynomial:
        value = _fraction_mod_prime(coefficient, prime)
        if value is None:
            return False
        reduced.append(value)
    degree = len(reduced) - 1
    if reduced[-1] % prime == 0:
        return False
    for value in range(prime):
        total = 0
        for coefficient in reversed(reduced):
            total = (total * value + coefficient) % prime
        if total == 0:
            return False
    if degree <= 3:
        return True
    if degree != 4:
        return False
    for linear in range(prime):
        for constant in range(prime):
            if _poly_mod_prime_divides(
                reduced, (constant, linear, 1), prime
            ):
                return False
    return True


_KNOWN_IRREDUCIBLE_QUARTICS = {
    (Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    (Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    (Fraction(1), Fraction(-1), Fraction(1), Fraction(-1), Fraction(1)),
    (Fraction(1), Fraction(0), Fraction(-1), Fraction(0), Fraction(1)),
}


def _irreducibility_certificate(
    polynomial: tuple[Fraction, ...],
) -> str:
    degree = len(polynomial) - 1
    if degree == 1:
        return "canonical-rational-field"
    if degree in (2, 3):
        if _has_rational_root(polynomial):
            raise KrennCounterexampleExactError(
                "minimal polynomial is reducible over Q"
            )
        return "exact-rational-root-criterion"
    if degree != 4:
        raise KrennCounterexampleExactError(
            "number-field degree exceeds the reviewed bound"
        )
    if _has_rational_root(polynomial):
        raise KrennCounterexampleExactError(
            "quartic minimal polynomial has a rational root"
        )
    if polynomial in _KNOWN_IRREDUCIBLE_QUARTICS:
        return "reviewed-cyclotomic-quartic"
    for prime in _IRREDUCIBILITY_PRIMES:
        if _irreducible_mod_prime(polynomial, prime):
            return f"irreducible-reduction-mod-{prime}"
    raise KrennCounterexampleExactError(
        "quartic irreducibility was not certified within the reviewed bound"
    )


@dataclass(frozen=True)
class NumberField:
    """A canonical power-basis presentation ``Q[theta]/(f(theta))``.

    ``minimal_polynomial`` is stored in constant-to-leading order and is
    normalized to monic form.  Every degree-one presentation is canonicalized
    to ``f=x``, so the rational field has exactly one representation here.
    """

    minimal_polynomial: tuple[Fraction, ...]
    schema: str = NUMBER_FIELD_SCHEMA
    _certificate: str = dataclass_field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.schema != NUMBER_FIELD_SCHEMA:
            raise KrennCounterexampleExactError(
                "number-field schema changed"
            )
        try:
            polynomial = tuple(
                _exact_fraction(value, "minimal-polynomial coefficient")
                for value in self.minimal_polynomial
            )
        except TypeError as error:
            raise KrennCounterexampleExactError(
                "minimal polynomial must be a finite exact sequence"
            ) from error
        polynomial = _poly_trim(polynomial)
        if len(polynomial) < 2:
            raise KrennCounterexampleExactError(
                "minimal polynomial must have positive degree"
            )
        lead = polynomial[-1]
        polynomial = tuple(value / lead for value in polynomial)
        degree = len(polynomial) - 1
        if not 1 <= degree <= MAX_FIELD_DEGREE:
            raise KrennCounterexampleExactError(
                "number-field degree must lie between one and four"
            )
        if degree == 1:
            polynomial = (Fraction(0), Fraction(1))
        certificate = _irreducibility_certificate(polynomial)
        object.__setattr__(self, "minimal_polynomial", polynomial)
        object.__setattr__(self, "_certificate", certificate)

    @classmethod
    def rational(cls) -> "NumberField":
        return cls((Fraction(0), Fraction(1)))

    @property
    def degree(self) -> int:
        return len(self.minimal_polynomial) - 1

    @property
    def irreducibility_certificate(self) -> str:
        return self._certificate

    @property
    def zero(self) -> "FieldElement":
        return FieldElement(self, ())

    @property
    def one(self) -> "FieldElement":
        return FieldElement(self, (Fraction(1),))

    @property
    def generator(self) -> "FieldElement":
        return FieldElement(self, (Fraction(0), Fraction(1)))

    def coerce(
        self,
        value: "FieldElement | Fraction | int | Sequence[Fraction | int]",
    ) -> "FieldElement":
        if isinstance(value, FieldElement):
            if value.field != self:
                raise KrennCounterexampleExactError(
                    "cannot mix elements from different field presentations"
                )
            return value
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return FieldElement(self, tuple(value))
        return FieldElement(
            self, (_exact_fraction(value, "field scalar"),)
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "kind": "number-field",
            "degree": self.degree,
            "generator": "theta",
            "basis": [
                "1",
                *(
                    "theta" if exponent == 1 else f"theta^{exponent}"
                    for exponent in range(1, self.degree)
                ),
            ],
            "coefficient_order": "constant-to-leading",
            "minimal_polynomial": [
                _fraction_payload(value)
                for value in self.minimal_polynomial
            ],
            "irreducibility_certificate": self._certificate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "NumberField":
        expected_keys = {
            "schema",
            "kind",
            "degree",
            "generator",
            "basis",
            "coefficient_order",
            "minimal_polynomial",
            "irreducibility_certificate",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_keys:
            raise KrennCounterexampleExactError(
                "number-field payload key set changed"
            )
        if (
            payload["schema"] != NUMBER_FIELD_SCHEMA
            or payload["kind"] != "number-field"
            or payload["generator"] != "theta"
            or payload["coefficient_order"] != "constant-to-leading"
        ):
            raise KrennCounterexampleExactError(
                "number-field payload identity changed"
            )
        polynomial_payload = payload["minimal_polynomial"]
        if not isinstance(polynomial_payload, Sequence):
            raise KrennCounterexampleExactError(
                "minimal polynomial payload must be a sequence"
            )
        result = cls(
            tuple(
                _fraction_from_payload(
                    row, f"minimal-polynomial coefficient {index}"
                )
                for index, row in enumerate(polynomial_payload)
            )
        )
        if payload != result.to_dict():
            raise KrennCounterexampleExactError(
                "number-field payload is not canonical"
            )
        return result

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "NumberField":
        return cls.from_dict(_loads_json(text))


@dataclass(frozen=True)
class FieldElement:
    """An immutable element in the fixed power basis of ``field``."""

    field: NumberField
    coefficients: tuple[Fraction, ...] = ()
    schema: str = FIELD_ELEMENT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.field, NumberField):
            raise KrennCounterexampleExactError(
                "a field element requires a canonical NumberField"
            )
        if self.schema != FIELD_ELEMENT_SCHEMA:
            raise KrennCounterexampleExactError(
                "field-element schema changed"
            )
        try:
            raw = tuple(
                _exact_fraction(value, "field-element coefficient")
                for value in self.coefficients
            )
        except TypeError as error:
            raise KrennCounterexampleExactError(
                "field-element coefficients must be an exact sequence"
            ) from error
        reduced = _poly_mod(raw, self.field.minimal_polynomial)
        padded = tuple(
            (
                reduced[index]
                if index < len(reduced)
                else Fraction(0)
            )
            for index in range(self.field.degree)
        )
        object.__setattr__(self, "coefficients", padded)

    @property
    def is_zero(self) -> bool:
        return not any(self.coefficients)

    @property
    def is_rational(self) -> bool:
        return not any(self.coefficients[1:])

    def as_fraction(self) -> Fraction:
        if not self.is_rational:
            raise KrennCounterexampleExactError(
                "a non-rational field element has no Fraction value"
            )
        return self.coefficients[0]

    def approximate(self, generator_value: complex) -> complex:
        if not (
            isfinite(float(complex(generator_value).real))
            and isfinite(float(complex(generator_value).imag))
        ):
            raise KrennCounterexampleExactError(
                "field embedding must be a finite complex value"
            )
        result = 0j
        for coefficient in reversed(self.coefficients):
            result = result * generator_value + float(coefficient)
        return result

    def __eq__(self, other) -> bool:
        if isinstance(other, FieldElement):
            if self.field == other.field:
                return self.coefficients == other.coefficients
            return (
                self.is_rational
                and other.is_rational
                and self.as_fraction() == other.as_fraction()
            )
        if isinstance(other, bool) or not isinstance(
            other, (Integral, Fraction)
        ):
            return NotImplemented
        return self.is_rational and self.as_fraction() == Fraction(other)

    def __hash__(self) -> int:
        if self.is_rational:
            return hash(self.as_fraction())
        return hash((self.field, self.coefficients))

    def _coerce(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        return self.field.coerce(other)

    def __add__(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        right = self._coerce(other)
        return FieldElement(
            self.field,
            tuple(
                left + value
                for left, value in zip(
                    self.coefficients,
                    right.coefficients,
                    strict=True,
                )
            ),
        )

    def __radd__(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        return self + other

    def __neg__(self) -> "FieldElement":
        return FieldElement(
            self.field, tuple(-value for value in self.coefficients)
        )

    def __sub__(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        return self + (-self._coerce(other))

    def __rsub__(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        return self._coerce(other) - self

    def __mul__(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        right = self._coerce(other)
        return FieldElement(
            self.field,
            _poly_mul(self.coefficients, right.coefficients),
        )

    def __rmul__(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        return self * other

    def inverse(self) -> "FieldElement":
        if self.is_zero:
            raise ZeroDivisionError("zero has no inverse in a number field")
        gcd_polynomial, coefficient, _other = _poly_extended_gcd(
            self.coefficients, self.field.minimal_polynomial
        )
        if gcd_polynomial != (Fraction(1),):
            raise KrennCounterexampleExactError(
                "field inversion failed its exact Bezout replay"
            )
        result = FieldElement(self.field, coefficient)
        if self * result != self.field.one:
            raise KrennCounterexampleExactError(
                "field inverse failed exact multiplication replay"
            )
        return result

    def __truediv__(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        return self * self._coerce(other).inverse()

    def __rtruediv__(
        self, other: "FieldElement | Fraction | int"
    ) -> "FieldElement":
        return self._coerce(other) / self

    def __pow__(self, exponent: int) -> "FieldElement":
        exponent = _exact_int(exponent, "field exponent")
        if exponent < 0:
            return self.inverse() ** (-exponent)
        result = self.field.one
        factor = self
        power = exponent
        while power:
            if power & 1:
                result *= factor
            factor *= factor
            power >>= 1
        return result

    def coefficient_payload(self) -> list[dict]:
        return [_fraction_payload(value) for value in self.coefficients]

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "field": self.field.to_dict(),
            "coefficient_order": "power-basis-low-to-high",
            "coefficients": self.coefficient_payload(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "FieldElement":
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema",
            "field",
            "coefficient_order",
            "coefficients",
        }:
            raise KrennCounterexampleExactError(
                "field-element payload key set changed"
            )
        if (
            payload["schema"] != FIELD_ELEMENT_SCHEMA
            or payload["coefficient_order"]
            != "power-basis-low-to-high"
        ):
            raise KrennCounterexampleExactError(
                "field-element payload identity changed"
            )
        field = NumberField.from_dict(payload["field"])
        coefficient_payload = payload["coefficients"]
        if not isinstance(coefficient_payload, Sequence):
            raise KrennCounterexampleExactError(
                "field-element coefficients must be a sequence"
            )
        result = cls(
            field,
            tuple(
                _fraction_from_payload(
                    row, f"field-element coefficient {index}"
                )
                for index, row in enumerate(coefficient_payload)
            ),
        )
        if payload != result.to_dict():
            raise KrennCounterexampleExactError(
                "field-element payload is not canonical"
            )
        return result

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "FieldElement":
        return cls.from_dict(_loads_json(text))


def rational_field() -> NumberField:
    """Return the unique degree-one presentation used for ``Q``."""

    return NumberField.rational()


def gaussian_rational_field() -> NumberField:
    """Return ``Q(i)`` with generator satisfying ``theta^2+1=0``."""

    return NumberField((1, 0, 1))


_CYCLOTOMIC_POLYNOMIALS: Mapping[int, tuple[int, ...]] = {
    3: (1, 1, 1),
    4: (1, 0, 1),
    5: (1, 1, 1, 1, 1),
    6: (1, -1, 1),
    8: (1, 0, 0, 0, 1),
    10: (1, -1, 1, -1, 1),
    12: (1, 0, -1, 0, 1),
}


def cyclotomic_field(order: int) -> NumberField:
    """Return a reviewed cyclotomic presentation of degree at most four."""

    order = _exact_int(order, "cyclotomic order")
    try:
        polynomial = _CYCLOTOMIC_POLYNOMIALS[order]
    except KeyError as error:
        raise KrennCounterexampleExactError(
            "cyclotomic order is outside the reviewed degree-four list"
        ) from error
    return NumberField(polynomial)


@dataclass(frozen=True)
class ExactCandidateWitness:
    """A canonical sparse finite witness over one structural number field."""

    n: int
    d: int
    field: NumberField
    entries: tuple[tuple[int, FieldElement], ...]
    schema: str = EXACT_WITNESS_SCHEMA

    def __post_init__(self) -> None:
        n = _exact_int(self.n, "witness factor count")
        d = _exact_int(self.d, "witness local dimension")
        if n < 2 or n % 2 or d < 1:
            raise KrennCounterexampleExactError(
                "exact witness needs even n>=2 and d>=1"
            )
        if not isinstance(self.field, NumberField):
            raise KrennCounterexampleExactError(
                "exact witness requires one canonical number field"
            )
        if self.schema != EXACT_WITNESS_SCHEMA:
            raise KrennCounterexampleExactError(
                "exact-witness schema changed"
            )
        try:
            normalized = tuple(
                (
                    _exact_int(index, "exact witness index"),
                    self.field.coerce(value),
                )
                for index, value in self.entries
            )
        except (TypeError, ValueError) as error:
            raise KrennCounterexampleExactError(
                "exact witness entries must be index-value pairs"
            ) from error
        indices = tuple(index for index, _value in normalized)
        if (
            indices != tuple(sorted(indices))
            or len(indices) != len(set(indices))
        ):
            raise KrennCounterexampleExactError(
                "exact witness indices must be unique and increasing"
            )
        ambient = variable_count(n, d)
        if any(index < 0 or index >= ambient for index in indices):
            raise KrennCounterexampleExactError(
                "exact witness index is outside the ambient system"
            )
        if any(value.is_zero for _index, value in normalized):
            raise KrennCounterexampleExactError(
                "zero exact entries must be omitted"
            )
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "entries", normalized)

    @classmethod
    def from_index_values(
        cls,
        n: int,
        d: int,
        field: NumberField,
        values: (
            Mapping[int, FieldElement | Fraction | int]
            | Iterable[
                tuple[int, FieldElement | Fraction | int]
            ]
        ),
    ) -> "ExactCandidateWitness":
        items = values.items() if isinstance(values, Mapping) else values
        combined: dict[int, FieldElement] = {}
        for raw_index, raw_value in items:
            index = _exact_int(raw_index, "exact witness index")
            if index in combined:
                raise KrennCounterexampleExactError(
                    f"duplicate exact witness index {index}"
                )
            value = field.coerce(raw_value)
            if not value.is_zero:
                combined[index] = value
        return cls(n, d, field, tuple(sorted(combined.items())))

    @classmethod
    def from_dense(
        cls,
        n: int,
        d: int,
        field: NumberField,
        values: Sequence[FieldElement | Fraction | int],
    ) -> "ExactCandidateWitness":
        values = tuple(values)
        if len(values) != variable_count(n, d):
            raise KrennCounterexampleExactError(
                "dense exact witness has the wrong length"
            )
        return cls.from_index_values(
            n, d, field, enumerate(values)
        )

    @property
    def support_size(self) -> int:
        return len(self.entries)

    def as_dict(self) -> dict[int, FieldElement]:
        return dict(self.entries)

    def dense_values(self) -> tuple[FieldElement, ...]:
        values = self.as_dict()
        return tuple(
            values.get(index, self.field.zero)
            for index in range(variable_count(self.n, self.d))
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "n": self.n,
            "d": self.d,
            "field": self.field.to_dict(),
            "omitted_coordinates": "zero",
            "support_size": self.support_size,
            "entries": [
                {
                    "variable_index": index,
                    "coordinate": list(
                        variable_key(self.n, self.d, index)
                    ),
                    "coefficients": value.coefficient_payload(),
                }
                for index, value in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ExactCandidateWitness":
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema",
            "n",
            "d",
            "field",
            "omitted_coordinates",
            "support_size",
            "entries",
        }:
            raise KrennCounterexampleExactError(
                "exact-witness payload key set changed"
            )
        if (
            payload["schema"] != EXACT_WITNESS_SCHEMA
            or payload["omitted_coordinates"] != "zero"
        ):
            raise KrennCounterexampleExactError(
                "exact-witness payload identity changed"
            )
        n = _exact_int(payload["n"], "witness factor count")
        d = _exact_int(payload["d"], "witness local dimension")
        field = NumberField.from_dict(payload["field"])
        raw_entries = payload["entries"]
        if not isinstance(raw_entries, Sequence):
            raise KrennCounterexampleExactError(
                "exact-witness entries must be a sequence"
            )
        entries = []
        for row_index, row in enumerate(raw_entries):
            if not isinstance(row, Mapping) or set(row) != {
                "variable_index",
                "coordinate",
                "coefficients",
            }:
                raise KrennCounterexampleExactError(
                    "exact-witness row key set changed"
                )
            index = _exact_int(
                row["variable_index"],
                f"exact-witness row {row_index} index",
            )
            try:
                coordinate = tuple(
                    _exact_int(value, "witness coordinate")
                    for value in row["coordinate"]
                )
            except TypeError as error:
                raise KrennCounterexampleExactError(
                    "exact-witness coordinate must be a sequence"
                ) from error
            if coordinate != variable_key(n, d, index):
                raise KrennCounterexampleExactError(
                    "exact-witness coordinate does not match its index"
                )
            coefficient_payload = row["coefficients"]
            if not isinstance(coefficient_payload, Sequence):
                raise KrennCounterexampleExactError(
                    "exact-witness coefficients must be a sequence"
                )
            value = FieldElement(
                field,
                tuple(
                    _fraction_from_payload(
                        coefficient,
                        (
                            f"exact-witness row {row_index} "
                            f"coefficient {coefficient_index}"
                        ),
                    )
                    for coefficient_index, coefficient in enumerate(
                        coefficient_payload
                    )
                ),
            )
            entries.append((index, value))
        result = cls(n, d, field, tuple(entries))
        if (
            _exact_int(
                payload["support_size"], "exact-witness support size"
            )
            != result.support_size
            or payload != result.to_dict()
        ):
            raise KrennCounterexampleExactError(
                "exact-witness payload is not canonical"
            )
        return result

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExactCandidateWitness":
        return cls.from_dict(_loads_json(text))


@lru_cache(maxsize=None)
def _primary_system(n: int, d: int) -> SparsePolynomialSystem:
    return generate_sparse_system(n, d)


def evaluate_primary(
    witness: ExactCandidateWitness,
) -> tuple[FieldElement, ...]:
    """Evaluate with the primary generated sparse monomial system."""

    if not isinstance(witness, ExactCandidateWitness):
        raise KrennCounterexampleExactError(
            "primary exact replay requires an ExactCandidateWitness"
    )
    system = _primary_system(witness.n, witness.d)
    values = witness.as_dict()
    zero = witness.field.zero
    one = witness.field.one
    output = []
    for equation in range(system.equation_count):
        total = zero
        for monomial in system.equation_monomials(equation):
            term = one
            for variable in monomial:
                term *= values.get(variable, zero)
                if term.is_zero:
                    break
            total += term
        output.append(total)
    return tuple(output)


@lru_cache(maxsize=None)
def _independent_perfect_matchings(
    n: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Filter edge subsets, without calling the primary matching enumerator."""

    n = _exact_int(n, "independent factor count")
    if n < 2 or n % 2:
        raise KrennCounterexampleExactError(
            "independent matching replay needs even n>=2"
        )
    edges = tuple(combinations(range(n), 2))
    matchings = []
    for candidate in combinations(edges, n // 2):
        covered = tuple(vertex for edge in candidate for vertex in edge)
        if len(set(covered)) == n:
            matchings.append(candidate)
    expected = 1
    for odd in range(n - 1, 0, -2):
        expected *= odd
    if len(matchings) != expected:
        raise KrennCounterexampleExactError(
            "independent perfect-matching census failed"
        )
    return tuple(matchings)


def _independent_variable_index(
    n: int, d: int, i: int, j: int, a: int, b: int
) -> int:
    if not (0 <= i < j < n and 0 <= a < d and 0 <= b < d):
        raise KrennCounterexampleExactError(
            "independent replay received a noncanonical variable"
        )
    edge_index = i * (2 * n - i - 1) // 2 + (j - i - 1)
    return (edge_index * d + a) * d + b


def evaluate_independent(
    witness: ExactCandidateWitness,
) -> tuple[FieldElement, ...]:
    """Evaluate by independent matchings and a locally rederived index law."""

    if not isinstance(witness, ExactCandidateWitness):
        raise KrennCounterexampleExactError(
            "independent exact replay requires an ExactCandidateWitness"
    )
    values = witness.as_dict()
    matchings = _independent_perfect_matchings(witness.n)
    zero = witness.field.zero
    one = witness.field.one
    output = []
    for coloring in product(range(witness.d), repeat=witness.n):
        total = zero
        for matching in matchings:
            term = one
            for i, j in matching:
                index = _independent_variable_index(
                    witness.n,
                    witness.d,
                    i,
                    j,
                    coloring[i],
                    coloring[j],
                )
                term *= values.get(index, zero)
                if term.is_zero:
                    break
            total += term
        output.append(total)
    return tuple(output)


def _canonical_target_values(
    n: int, d: int, field: NumberField
) -> tuple[FieldElement, ...]:
    zero = field.zero
    one = field.one
    return tuple(
        (
            one
            if all(color == coloring[0] for color in coloring[1:])
            else zero
        )
        for coloring in product(range(d), repeat=n)
    )


def _residuals(
    output: Sequence[FieldElement],
    target: Sequence[FieldElement],
) -> tuple[FieldElement, ...]:
    if len(output) != len(target):
        raise KrennCounterexampleExactError(
            "exact output and target dimensions differ"
        )
    return tuple(
        value - expected
        for value, expected in zip(output, target, strict=True)
    )


@dataclass(frozen=True)
class ExactVerificationReport:
    """Content-derived two-enumerator exact verification of one witness."""

    witness: ExactCandidateWitness
    schema: str = EXACT_VERIFICATION_SCHEMA
    primary_residuals: tuple[FieldElement, ...] = dataclass_field(
        init=False
    )
    independent_residuals: tuple[FieldElement, ...] = dataclass_field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.witness, ExactCandidateWitness):
            raise KrennCounterexampleExactError(
                "exact report requires a canonical candidate witness"
            )
        if self.schema != EXACT_VERIFICATION_SCHEMA:
            raise KrennCounterexampleExactError(
                "exact-verification schema changed"
            )
        target = _canonical_target_values(
            self.witness.n, self.witness.d, self.witness.field
        )
        primary = _residuals(
            evaluate_primary(self.witness), target
        )
        independent = _residuals(
            evaluate_independent(self.witness), target
        )
        object.__setattr__(self, "primary_residuals", primary)
        object.__setattr__(
            self, "independent_residuals", independent
        )

    @property
    def equation_count(self) -> int:
        return self.witness.d**self.witness.n

    @property
    def enumerators_agree(self) -> bool:
        return self.primary_residuals == self.independent_residuals

    @property
    def primary_nonzero_equations(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, residual in enumerate(self.primary_residuals)
            if not residual.is_zero
        )

    @property
    def independent_nonzero_equations(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, residual in enumerate(
                self.independent_residuals
            )
            if not residual.is_zero
        )

    @property
    def primary_exact(self) -> bool:
        return (
            len(self.primary_residuals) == self.equation_count
            and not self.primary_nonzero_equations
        )

    @property
    def independent_exact(self) -> bool:
        return (
            len(self.independent_residuals) == self.equation_count
            and not self.independent_nonzero_equations
        )

    @property
    def exact(self) -> bool:
        return (
            self.enumerators_agree
            and self.primary_exact
            and self.independent_exact
        )

    def to_dict(self) -> dict:
        checks = {
            "primary_replayed_every_equation": (
                len(self.primary_residuals) == self.equation_count
            ),
            "independent_replayed_every_equation": (
                len(self.independent_residuals)
                == self.equation_count
            ),
            "primary_and_independent_residuals_agree": (
                self.enumerators_agree
            ),
            "all_primary_residuals_are_exactly_zero": (
                self.primary_exact
            ),
            "all_independent_residuals_are_exactly_zero": (
                self.independent_exact
            ),
        }
        return {
            "schema": self.schema,
            "witness": self.witness.to_dict(),
            "parameters": {
                "n": self.witness.n,
                "d": self.witness.d,
                "variables": variable_count(
                    self.witness.n, self.witness.d
                ),
                "equations": self.equation_count,
                "support_size": self.witness.support_size,
            },
            "primary_residuals": [
                residual.coefficient_payload()
                for residual in self.primary_residuals
            ],
            "independent_residuals": [
                residual.coefficient_payload()
                for residual in self.independent_residuals
            ],
            "primary_nonzero_equations": list(
                self.primary_nonzero_equations
            ),
            "independent_nonzero_equations": list(
                self.independent_nonzero_equations
            ),
            "exact_checks": checks,
            "exact": self.exact,
            "claim_boundary": {
                "finite_exact_number_field_weights": True,
                "all_equations_replayed_by_primary_enumerator": True,
                "all_equations_replayed_by_independent_enumerator": True,
                "f31_used_as_characteristic_zero_evidence": False,
                "numerical_closeness_used_as_exact_evidence": False,
                "exact_affine_image_membership_certified": self.exact,
                "complex_affine_image_membership_certified": self.exact,
                "nonexistence_proved": False,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ExactVerificationReport":
        if not isinstance(payload, Mapping):
            raise KrennCounterexampleExactError(
                "exact-verification payload must be an object"
            )
        try:
            witness_payload = payload["witness"]
        except KeyError as error:
            raise KrennCounterexampleExactError(
                "exact-verification payload omits its witness"
            ) from error
        result = cls(ExactCandidateWitness.from_dict(witness_payload))
        if payload != result.to_dict():
            raise KrennCounterexampleExactError(
                "exact-verification payload failed semantic replay"
            )
        return result

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ExactVerificationReport":
        return cls.from_dict(_loads_json(text))


def verify_exact_candidate(
    witness: ExactCandidateWitness,
) -> ExactVerificationReport:
    """Replay a candidate without promoting nonzero residuals to a claim."""

    return ExactVerificationReport(witness)


def require_exact_candidate(
    witness: ExactCandidateWitness,
) -> ExactVerificationReport:
    """Return the exact report, raising unless both enumerators certify it."""

    report = verify_exact_candidate(witness)
    if not report.exact:
        raise KrennCounterexampleExactError(
            "candidate is not an exact finite witness in all equations"
        )
    return report


@dataclass(frozen=True)
class ReconstructionBounds:
    """Finite deterministic bounds for numerical algebraic recognition."""

    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-9
    zero_tolerance: float = 1e-12
    max_denominator: int = 32
    max_numerator: int = 256
    coefficient_grid_denominator: int = 6
    coefficient_grid_numerator: int = 8
    cyclotomic_orders: tuple[int, ...] = (3, 4, 5, 6, 8, 10, 12)
    max_degree: int = 4
    polynomial_height: int = 3
    max_polynomial_checks: int = 2_000
    max_generator_candidates: int = 8
    max_field_attempts: int = 64
    schema: str = RECONSTRUCTION_BOUNDS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECONSTRUCTION_BOUNDS_SCHEMA:
            raise KrennReconstructionError(
                "reconstruction-bounds schema changed"
            )
        for name in (
            "absolute_tolerance",
            "relative_tolerance",
            "zero_tolerance",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise KrennReconstructionError(
                    f"{name} must be finite and nonnegative"
                )
            object.__setattr__(self, name, value)
        if self.absolute_tolerance == self.relative_tolerance == 0:
            raise KrennReconstructionError(
                "at least one reconstruction tolerance must be positive"
            )
        integer_bounds = {
            "max_denominator": self.max_denominator,
            "max_numerator": self.max_numerator,
            "coefficient_grid_denominator": (
                self.coefficient_grid_denominator
            ),
            "coefficient_grid_numerator": (
                self.coefficient_grid_numerator
            ),
            "polynomial_height": self.polynomial_height,
            "max_polynomial_checks": self.max_polynomial_checks,
            "max_generator_candidates": self.max_generator_candidates,
            "max_field_attempts": self.max_field_attempts,
        }
        for name, raw_value in integer_bounds.items():
            value = _exact_int(raw_value, name)
            if value < 1:
                raise KrennReconstructionError(
                    f"{name} must be positive"
                )
            object.__setattr__(self, name, value)
        degree = _exact_int(self.max_degree, "max degree")
        if not 1 <= degree <= MAX_FIELD_DEGREE:
            raise KrennReconstructionError(
                "max degree must lie between one and four"
            )
        object.__setattr__(self, "max_degree", degree)
        orders = tuple(
            _exact_int(order, "cyclotomic order")
            for order in self.cyclotomic_orders
        )
        if (
            orders != tuple(sorted(set(orders)))
            or any(order not in _CYCLOTOMIC_POLYNOMIALS for order in orders)
            or any(cyclotomic_field(order).degree > degree for order in orders)
        ):
            raise KrennReconstructionError(
                "cyclotomic orders must be unique, sorted, reviewed, "
                "and within max_degree"
            )
        object.__setattr__(self, "cyclotomic_orders", orders)

    def allowed_error(self, value: complex) -> float:
        return self.absolute_tolerance + (
            self.relative_tolerance * abs(value)
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "zero_tolerance": self.zero_tolerance,
            "max_denominator": self.max_denominator,
            "max_numerator": self.max_numerator,
            "coefficient_grid_denominator": (
                self.coefficient_grid_denominator
            ),
            "coefficient_grid_numerator": (
                self.coefficient_grid_numerator
            ),
            "cyclotomic_orders": list(self.cyclotomic_orders),
            "max_degree": self.max_degree,
            "polynomial_height": self.polynomial_height,
            "max_polynomial_checks": self.max_polynomial_checks,
            "max_generator_candidates": self.max_generator_candidates,
            "max_field_attempts": self.max_field_attempts,
        }


def _bounded_fraction(
    value: float,
    *,
    max_denominator: int,
    max_numerator: int,
    tolerance: float,
) -> Fraction | None:
    if not isfinite(float(value)):
        return None
    candidate = Fraction(float(value)).limit_denominator(max_denominator)
    if (
        abs(candidate.numerator) > max_numerator
        or abs(float(candidate) - value) > tolerance
    ):
        return None
    return candidate


@lru_cache(maxsize=None)
def _fraction_grid(
    max_denominator: int, max_numerator: int
) -> tuple[Fraction, ...]:
    values = {
        Fraction(numerator, denominator)
        for denominator in range(1, max_denominator + 1)
        for numerator in range(
            -max_numerator, max_numerator + 1
        )
    }
    return tuple(sorted(values))


def _coefficient_tuple_key(
    coefficients: Sequence[Fraction],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (value.numerator, value.denominator)
        for value in coefficients
    )


def _bounded_power_basis_recognition(
    value: complex,
    field: NumberField,
    generator_value: complex,
    bounds: ReconstructionBounds,
) -> tuple[FieldElement, float] | None:
    tolerance = bounds.allowed_error(value)
    if field.degree == 1:
        if abs(value.imag) > tolerance:
            return None
        rational = _bounded_fraction(
            value.real,
            max_denominator=bounds.max_denominator,
            max_numerator=bounds.max_numerator,
            tolerance=tolerance,
        )
        if rational is None:
            return None
        element = field.coerce(rational)
        error = abs(element.approximate(generator_value) - value)
        return (element, error) if error <= tolerance else None

    if field.degree == 2 and abs(generator_value.imag) > tolerance:
        second_float = value.imag / generator_value.imag
        constant_float = (
            value.real - second_float * generator_value.real
        )
        constant = _bounded_fraction(
            constant_float,
            max_denominator=bounds.max_denominator,
            max_numerator=bounds.max_numerator,
            tolerance=tolerance * (1 + abs(constant_float)),
        )
        second = _bounded_fraction(
            second_float,
            max_denominator=bounds.max_denominator,
            max_numerator=bounds.max_numerator,
            tolerance=tolerance * (1 + abs(second_float)),
        )
        if constant is None or second is None:
            return None
        element = FieldElement(field, (constant, second))
        error = abs(element.approximate(generator_value) - value)
        return (element, error) if error <= tolerance else None

    grid = _fraction_grid(
        bounds.coefficient_grid_denominator,
        bounds.coefficient_grid_numerator,
    )
    basis = tuple(
        generator_value**exponent
        for exponent in range(field.degree)
    )
    split = field.degree // 2
    left_rows = []
    for coefficients in product(grid, repeat=split):
        approximate = sum(
            float(coefficient) * basis[index]
            for index, coefficient in enumerate(coefficients)
        )
        left_rows.append((approximate, coefficients))
    right_rows = []
    for coefficients in product(
        grid, repeat=field.degree - split
    ):
        approximate = sum(
            float(coefficient) * basis[split + offset]
            for offset, coefficient in enumerate(coefficients)
        )
        right_rows.append((approximate, coefficients))
    right_rows.sort(
        key=lambda row: (
            row[0].real,
            row[0].imag,
            _coefficient_tuple_key(row[1]),
        )
    )
    right_reals = tuple(row[0].real for row in right_rows)
    best: tuple[
        float, tuple[tuple[int, int], ...], FieldElement
    ] | None = None
    for left_approximate, left_coefficients in left_rows:
        target = value - left_approximate
        start = bisect_left(right_reals, target.real - tolerance)
        stop = bisect_right(right_reals, target.real + tolerance)
        for right_approximate, right_coefficients in right_rows[
            start:stop
        ]:
            error = abs(
                left_approximate + right_approximate - value
            )
            if error > tolerance:
                continue
            coefficients = (
                *left_coefficients,
                *right_coefficients,
            )
            element = FieldElement(field, coefficients)
            replay_error = abs(
                element.approximate(generator_value) - value
            )
            key = (
                replay_error,
                _coefficient_tuple_key(element.coefficients),
                element,
            )
            if best is None or key[:2] < best[:2]:
                best = key
    return None if best is None else (best[2], best[0])


def _complex_payload(value: complex | None) -> dict | None:
    if value is None:
        return None
    return {"real": float(value.real), "imag": float(value.imag)}


@dataclass(frozen=True)
class ReconstructionAttempt:
    """One bounded embedding-recognition attempt and its exact replay."""

    method: str
    label: str
    field: NumberField | None
    generator_embedding: complex | None
    recognized: bool
    max_embedding_error: float | None
    candidate: ExactCandidateWitness | None
    verification: ExactVerificationReport | None
    reason: str

    def __post_init__(self) -> None:
        if not self.method or not self.label or not self.reason:
            raise KrennReconstructionError(
                "reconstruction attempt labels cannot be empty"
            )
        if self.generator_embedding is not None and not (
            isfinite(float(self.generator_embedding.real))
            and isfinite(float(self.generator_embedding.imag))
        ):
            raise KrennReconstructionError(
                "reconstruction embedding must be finite"
            )
        if self.max_embedding_error is not None and (
            not isfinite(float(self.max_embedding_error))
            or self.max_embedding_error < 0
        ):
            raise KrennReconstructionError(
                "reconstruction error must be finite and nonnegative"
            )
        if self.recognized != (self.candidate is not None):
            raise KrennReconstructionError(
                "recognized status must follow from candidate presence"
            )
        if self.candidate is not None and (
            self.field != self.candidate.field
            or self.verification is None
            or self.verification.witness != self.candidate
        ):
            raise KrennReconstructionError(
                "recognized candidate lacks its content-derived exact replay"
            )
        if self.candidate is None and self.verification is not None:
            raise KrennReconstructionError(
                "unrecognized attempt cannot carry an exact verification"
            )

    @property
    def exact(self) -> bool:
        return (
            self.verification is not None
            and self.verification.exact
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "label": self.label,
            "field": self.field.to_dict() if self.field else None,
            "generator_embedding": _complex_payload(
                self.generator_embedding
            ),
            "recognized": self.recognized,
            "max_embedding_error": self.max_embedding_error,
            "candidate": (
                self.candidate.to_dict()
                if self.candidate is not None
                else None
            ),
            "verification": (
                self.verification.to_dict()
                if self.verification is not None
                else None
            ),
            "exact": self.exact,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReconstructionReport:
    """Deterministic bounded reconstruction ledger with no miss-as-proof."""

    n: int
    d: int
    requested_support: tuple[int, ...]
    bounds: ReconstructionBounds
    attempts: tuple[ReconstructionAttempt, ...]
    polynomial_checks: int
    preserve_support: bool
    schema: str = RECONSTRUCTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECONSTRUCTION_SCHEMA:
            raise KrennReconstructionError(
                "reconstruction-report schema changed"
            )
        n = _exact_int(self.n, "reconstruction factor count")
        d = _exact_int(self.d, "reconstruction local dimension")
        if n < 2 or n % 2 or d < 1:
            raise KrennReconstructionError(
                "reconstruction needs even n>=2 and d>=1"
            )
        support = tuple(
            _exact_int(index, "reconstruction support index")
            for index in self.requested_support
        )
        if (
            support != tuple(sorted(set(support)))
            or any(
                index < 0 or index >= variable_count(n, d)
                for index in support
            )
        ):
            raise KrennReconstructionError(
                "reconstruction support is not canonical"
            )
        if not isinstance(self.bounds, ReconstructionBounds):
            raise KrennReconstructionError(
                "reconstruction report lacks canonical bounds"
            )
        if len(self.attempts) > self.bounds.max_field_attempts + 1:
            raise KrennReconstructionError(
                "reconstruction attempt ledger exceeds its bound"
            )
        checks = _exact_int(
            self.polynomial_checks, "polynomial check count"
        )
        if not 0 <= checks <= self.bounds.max_polynomial_checks:
            raise KrennReconstructionError(
                "polynomial check count exceeds its bound"
            )
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "requested_support", support)
        object.__setattr__(self, "polynomial_checks", checks)

    @property
    def exact_attempts(self) -> tuple[ReconstructionAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.exact)

    @property
    def exact_candidate(self) -> ExactCandidateWitness | None:
        return (
            self.exact_attempts[0].candidate
            if self.exact_attempts
            else None
        )

    @property
    def exact(self) -> bool:
        return self.exact_candidate is not None

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "parameters": {
                "n": self.n,
                "d": self.d,
                "ambient_variables": variable_count(self.n, self.d),
                "requested_support": list(self.requested_support),
                "preserve_support": self.preserve_support,
            },
            "bounds": self.bounds.to_dict(),
            "polynomial_checks": self.polynomial_checks,
            "attempts": [
                attempt.to_dict() for attempt in self.attempts
            ],
            "exact_candidate": (
                self.exact_candidate.to_dict()
                if self.exact_candidate is not None
                else None
            ),
            "exact": self.exact,
            "claim_boundary": {
                "bounded_numerical_recognition_only": True,
                "reconstruction_miss_is_nonexistence_proof": False,
                "approximate_fit_is_exact_witness": False,
                "exact_candidate_requires_two_exact_enumerators": True,
                "exact_affine_image_membership_certified": self.exact,
                "complex_affine_image_membership_certified": self.exact,
                "nonexistence_proved": False,
            },
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())


def _attempt_embedding(
    values: np.ndarray,
    support: tuple[int, ...],
    *,
    n: int,
    d: int,
    field: NumberField,
    generator_embedding: complex,
    bounds: ReconstructionBounds,
    preserve_support: bool,
    method: str,
    label: str,
) -> ReconstructionAttempt:
    polynomial_error = abs(
        _poly_evaluate_complex(
            field.minimal_polynomial, generator_embedding
        )
    )
    polynomial_scale = 1 + sum(
        abs(float(coefficient))
        * abs(generator_embedding) ** exponent
        for exponent, coefficient in enumerate(
            field.minimal_polynomial
        )
    )
    if polynomial_error > (
        10 * bounds.absolute_tolerance
        + 10 * bounds.relative_tolerance * polynomial_scale
    ):
        return ReconstructionAttempt(
            method=method,
            label=label,
            field=field,
            generator_embedding=generator_embedding,
            recognized=False,
            max_embedding_error=None,
            candidate=None,
            verification=None,
            reason="generator embedding failed its bounded polynomial check",
        )
    entries = []
    errors = []
    for index in support:
        recognized = _bounded_power_basis_recognition(
            complex(values[index]),
            field,
            generator_embedding,
            bounds,
        )
        if recognized is None:
            return ReconstructionAttempt(
                method=method,
                label=label,
                field=field,
                generator_embedding=generator_embedding,
                recognized=False,
                max_embedding_error=None,
                candidate=None,
                verification=None,
                reason=(
                    f"coordinate {index} had no coefficient vector "
                    "within the declared bounds"
                ),
            )
        element, error = recognized
        if preserve_support and element.is_zero:
            return ReconstructionAttempt(
                method=method,
                label=label,
                field=field,
                generator_embedding=generator_embedding,
                recognized=False,
                max_embedding_error=None,
                candidate=None,
                verification=None,
                reason=(
                    f"coordinate {index} reconstructed to zero while "
                    "support preservation was required"
                ),
            )
        if not element.is_zero:
            entries.append((index, element))
        errors.append(error)
    candidate = ExactCandidateWitness.from_index_values(
        n, d, field, entries
    )
    verification = verify_exact_candidate(candidate)
    return ReconstructionAttempt(
        method=method,
        label=label,
        field=field,
        generator_embedding=generator_embedding,
        recognized=True,
        max_embedding_error=max(errors, default=0.0),
        candidate=candidate,
        verification=verification,
        reason=(
            "recognized coefficients were evaluated in exact arithmetic "
            "by both enumerators"
        ),
    )


def _primitive_embedding_exponents(order: int) -> tuple[int, ...]:
    return tuple(
        exponent
        for exponent in range(1, order)
        if gcd(exponent, order) == 1
    )


def _candidate_generator_indices(
    values: np.ndarray,
    support: tuple[int, ...],
    bounds: ReconstructionBounds,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            support,
            key=lambda index: (
                abs(complex(values[index]).imag)
                <= bounds.zero_tolerance,
                index,
            ),
        )[: bounds.max_generator_candidates]
    )


def _bounded_polynomial_candidates(
    bounds: ReconstructionBounds,
) -> Iterable[tuple[int, ...]]:
    height = bounds.polynomial_height
    for degree in range(2, bounds.max_degree + 1):
        for lower in product(
            range(-height, height + 1), repeat=degree
        ):
            if lower[0] == 0:
                continue
            yield (*lower, 1)


def reconstruct_complex_candidate(
    values: Sequence[complex] | np.ndarray,
    *,
    n: int = 6,
    d: int = 3,
    support: Sequence[int] | None = None,
    bounds: ReconstructionBounds | None = None,
    preserve_support: bool = True,
) -> ReconstructionReport:
    """Try deterministic bounded exact reconstructions of one complex vector.

    The attempt order is ``Q``, Gaussian rationals, reviewed cyclotomic
    embeddings, then a generator-led bounded search over monic integer
    polynomials of degree at most four.  A proposal is considered exact only
    if :class:`ExactVerificationReport` certifies every equation twice.
    """

    n = _exact_int(n, "reconstruction factor count")
    d = _exact_int(d, "reconstruction local dimension")
    if n < 2 or n % 2 or d < 1:
        raise KrennReconstructionError(
            "reconstruction needs even n>=2 and d>=1"
        )
    bounds = ReconstructionBounds() if bounds is None else bounds
    if not isinstance(bounds, ReconstructionBounds):
        raise KrennReconstructionError(
            "bounds must be a ReconstructionBounds instance"
        )
    if not isinstance(preserve_support, bool):
        raise KrennReconstructionError(
            "preserve_support must be a Boolean"
        )
    try:
        vector = np.asarray(values, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise KrennReconstructionError(
            "complex candidate could not be converted to an array"
        ) from error
    ambient = variable_count(n, d)
    if vector.shape != (ambient,):
        raise KrennReconstructionError(
            f"complex candidate must have shape ({ambient},)"
        )
    if not (
        np.all(np.isfinite(vector.real))
        and np.all(np.isfinite(vector.imag))
    ):
        raise KrennReconstructionError(
            "complex candidate contains NaN or infinity"
        )
    if support is None:
        canonical_support = tuple(
            index
            for index, value in enumerate(vector)
            if abs(value) > bounds.zero_tolerance
        )
    else:
        canonical_support = tuple(
            _exact_int(index, "reconstruction support index")
            for index in support
        )
        if (
            canonical_support
            != tuple(sorted(set(canonical_support)))
            or any(
                index < 0 or index >= ambient
                for index in canonical_support
            )
        ):
            raise KrennReconstructionError(
                "provided reconstruction support is not canonical"
            )
        support_set = set(canonical_support)
        outside = [
            index
            for index, value in enumerate(vector)
            if index not in support_set
            and abs(value) > bounds.zero_tolerance
        ]
        if outside:
            raise KrennReconstructionError(
                "complex vector is nonzero outside the declared support"
            )
        if preserve_support and any(
            abs(vector[index]) <= bounds.zero_tolerance
            for index in canonical_support
        ):
            raise KrennReconstructionError(
                "declared support contains a numerically zero coordinate"
            )

    attempts: list[ReconstructionAttempt] = []
    attempted_presentations: set[
        tuple[tuple[Fraction, ...], int]
    ] = set()

    def add_attempt(
        field: NumberField,
        embedding: complex,
        method: str,
        label: str,
        embedding_id: int,
    ) -> bool:
        if len(attempts) >= bounds.max_field_attempts:
            return False
        key = (field.minimal_polynomial, embedding_id)
        if key in attempted_presentations:
            return True
        attempted_presentations.add(key)
        attempt = _attempt_embedding(
            vector,
            canonical_support,
            n=n,
            d=d,
            field=field,
            generator_embedding=embedding,
            bounds=bounds,
            preserve_support=preserve_support,
            method=method,
            label=label,
        )
        attempts.append(attempt)
        return not attempt.exact

    should_continue = add_attempt(
        rational_field(),
        0j,
        "rational",
        "Q",
        0,
    )
    if should_continue:
        should_continue = add_attempt(
            gaussian_rational_field(),
            1j,
            "gaussian-rational",
            "Q(i), positive-i embedding",
            1,
        )

    if should_continue:
        for order in bounds.cyclotomic_orders:
            field = cyclotomic_field(order)
            for exponent in _primitive_embedding_exponents(order):
                embedding = cmath.exp(
                    2j * pi * exponent / order
                )
                should_continue = add_attempt(
                    field,
                    embedding,
                    "cyclotomic",
                    f"Q(zeta_{order}), embedding exponent {exponent}",
                    order * 100 + exponent,
                )
                if not should_continue:
                    break
            if not should_continue:
                break

    polynomial_checks = 0
    generic_attempt_count = 0
    if should_continue and len(attempts) < bounds.max_field_attempts:
        generator_indices = _candidate_generator_indices(
            vector, canonical_support, bounds
        )
        for raw_polynomial in _bounded_polynomial_candidates(bounds):
            if polynomial_checks >= bounds.max_polynomial_checks:
                break
            polynomial_checks += 1
            for generator_index in generator_indices:
                generator_value = complex(vector[generator_index])
                error = abs(
                    _poly_evaluate_complex(
                        raw_polynomial, generator_value
                    )
                )
                scale = 1 + sum(
                    abs(coefficient)
                    * abs(generator_value) ** exponent
                    for exponent, coefficient in enumerate(
                        raw_polynomial
                    )
                )
                if error > (
                    bounds.absolute_tolerance
                    + bounds.relative_tolerance * scale
                ):
                    continue
                try:
                    field = NumberField(raw_polynomial)
                except KrennCounterexampleExactError:
                    continue
                should_continue = add_attempt(
                    field,
                    generator_value,
                    "bounded-low-degree-polynomial",
                    (
                        f"generator coordinate {generator_index}, "
                        f"polynomial {raw_polynomial}"
                    ),
                    -(polynomial_checks * ambient + generator_index + 1),
                )
                generic_attempt_count += 1
                if (
                    not should_continue
                    or len(attempts) >= bounds.max_field_attempts
                ):
                    break
            if (
                not should_continue
                or len(attempts) >= bounds.max_field_attempts
            ):
                break
        if should_continue and generic_attempt_count == 0:
            attempts.append(
                ReconstructionAttempt(
                    method="bounded-low-degree-polynomial",
                    label="bounded generic field search",
                    field=None,
                    generator_embedding=None,
                    recognized=False,
                    max_embedding_error=None,
                    candidate=None,
                    verification=None,
                    reason=(
                        f"no irreducible generator relation was found "
                        f"in {polynomial_checks} polynomial checks"
                    ),
                )
            )

    return ReconstructionReport(
        n=n,
        d=d,
        requested_support=canonical_support,
        bounds=bounds,
        attempts=tuple(attempts),
        polynomial_checks=polynomial_checks,
        preserve_support=bool(preserve_support),
    )


# Short aliases for callers that use the campaign terminology.
AlgebraicSparseWitness = ExactCandidateWitness
ExactFieldVerificationReport = ExactVerificationReport
reconstruct_candidate = reconstruct_complex_candidate
reconstruct_from_complex = reconstruct_complex_candidate
verify_exact = verify_exact_candidate
verify_candidate_exact = verify_exact_candidate


__all__ = (
    "AlgebraicSparseWitness",
    "EXACT_VERIFICATION_SCHEMA",
    "EXACT_WITNESS_SCHEMA",
    "ExactCandidateWitness",
    "ExactFieldVerificationReport",
    "ExactVerificationReport",
    "FIELD_ELEMENT_SCHEMA",
    "FieldElement",
    "KrennCounterexampleExactError",
    "KrennReconstructionError",
    "MAX_FIELD_DEGREE",
    "NUMBER_FIELD_SCHEMA",
    "NumberField",
    "RECONSTRUCTION_BOUNDS_SCHEMA",
    "RECONSTRUCTION_SCHEMA",
    "ReconstructionAttempt",
    "ReconstructionBounds",
    "ReconstructionReport",
    "canonical_json_dumps",
    "cyclotomic_field",
    "evaluate_independent",
    "evaluate_primary",
    "gaussian_rational_field",
    "rational_field",
    "reconstruct_candidate",
    "reconstruct_complex_candidate",
    "reconstruct_from_complex",
    "require_exact_candidate",
    "verify_candidate_exact",
    "verify_exact",
    "verify_exact_candidate",
)
