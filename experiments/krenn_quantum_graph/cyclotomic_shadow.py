"""Exact ``Q(omega)`` witness for the ``n=6,d=3`` equal-``g`` shadow.

Write ``omega^2 + omega + 1 = 0`` and put

    X = g_0^2,  Y = g_1^2,  Z = g_2^2.

Three diagonal quadratic forms

    A = X + Y + Z,
    B = X + omega*Y + omega^2*Z,
    C = X + omega^2*Y + omega*Z

satisfy

    (A^3 + B^3 + C^3 + 6*A*B*C) / 9 = X^3 + Y^3 + Z^3.

Placing scalar multiples of ``A``, ``B``, and ``C`` on the three perfect
matchings of a triangular prism realizes the left-hand side as a six-vertex
hafnian.  This gives an exact point in the equal-``g`` GHZ shadow fiber.

It is deliberately not represented as a full tensor witness.  The equal-``g``
shadow aggregates all colorings with the same occupation, and those
aggregated cancellations do not make the individual mixed coefficients zero.
In particular, coloring ``002121`` remains nonzero.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from experiments.krenn_quantum_graph.hafnian_identities import (
    EqualGShadowSystem,
    generate_equal_g_shadow_system,
)
from experiments.krenn_quantum_graph.system import perfect_matchings
from experiments.krenn_quantum_graph.targets import canonical_ghz_target


ExactRational = int | Fraction

N = 6
D = 3
VICTIM_COLORING = (0, 0, 2, 1, 2, 1)


class KrennCyclotomicShadowError(ValueError):
    """A cyclotomic scalar or the explicit shadow witness is malformed."""


@dataclass(frozen=True)
class QOmega:
    """An exact element ``a + b*omega`` of ``Q[omega]/(omega^2+omega+1)``."""

    constant: Fraction = Fraction(0)
    omega: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        try:
            constant = Fraction(self.constant)
            omega = Fraction(self.omega)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise KrennCyclotomicShadowError(
                "Q(omega) coefficients must be exact rationals"
            ) from error
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "omega", omega)

    @classmethod
    def coerce(cls, value: QOmega | ExactRational) -> QOmega:
        if isinstance(value, cls):
            return value
        try:
            return cls(Fraction(value), Fraction(0))
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise KrennCyclotomicShadowError(
                "value is not an exact Q(omega) scalar"
            ) from error

    @property
    def is_zero(self) -> bool:
        return self.constant == 0 and self.omega == 0

    @property
    def norm(self) -> Fraction:
        """Return ``(a+b*omega)(a+b*omega^2)``."""

        return (
            self.constant * self.constant
            - self.constant * self.omega
            + self.omega * self.omega
        )

    def conjugate(self) -> QOmega:
        """Apply the nontrivial automorphism ``omega -> omega^2``."""

        return QOmega(
            self.constant - self.omega,
            -self.omega,
        )

    def inverse(self) -> QOmega:
        if self.is_zero:
            raise ZeroDivisionError("zero has no inverse in Q(omega)")
        return self.conjugate() / self.norm

    def __add__(self, other: QOmega | ExactRational) -> QOmega:
        right = QOmega.coerce(other)
        return QOmega(
            self.constant + right.constant,
            self.omega + right.omega,
        )

    def __radd__(self, other: QOmega | ExactRational) -> QOmega:
        return self + other

    def __neg__(self) -> QOmega:
        return QOmega(-self.constant, -self.omega)

    def __sub__(self, other: QOmega | ExactRational) -> QOmega:
        return self + (-QOmega.coerce(other))

    def __rsub__(self, other: QOmega | ExactRational) -> QOmega:
        return QOmega.coerce(other) - self

    def __mul__(self, other: QOmega | ExactRational) -> QOmega:
        right = QOmega.coerce(other)
        # omega^2 = -1 - omega.
        return QOmega(
            self.constant * right.constant
            - self.omega * right.omega,
            self.constant * right.omega
            + self.omega * right.constant
            - self.omega * right.omega,
        )

    def __rmul__(self, other: QOmega | ExactRational) -> QOmega:
        return self * other

    def __truediv__(self, other: QOmega | ExactRational) -> QOmega:
        right = QOmega.coerce(other)
        if right.is_zero:
            raise ZeroDivisionError("division by zero in Q(omega)")
        conjugate = right.conjugate()
        numerator = self * conjugate
        return QOmega(
            numerator.constant / right.norm,
            numerator.omega / right.norm,
        )

    def __rtruediv__(self, other: QOmega | ExactRational) -> QOmega:
        return QOmega.coerce(other) / self

    def __pow__(self, exponent: int) -> QOmega:
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise TypeError("Q(omega) exponent must be an integer")
        if exponent < 0:
            return self.inverse() ** (-exponent)
        result = QOMEGA_ONE
        factor = self
        power = exponent
        while power:
            if power & 1:
                result *= factor
            factor *= factor
            power >>= 1
        return result


QOMEGA_ZERO = QOmega(0, 0)
QOMEGA_ONE = QOmega(1, 0)
OMEGA = QOmega(0, 1)
OMEGA_SQUARED = OMEGA * OMEGA


# Each nonzero edge carries a scalar multiple of one of the three forms below.
# Entries are the coefficients of X, Y, Z respectively.
PRISM_FORMS = {
    "A": (QOMEGA_ONE, QOMEGA_ONE, QOMEGA_ONE),
    "B": (QOMEGA_ONE, OMEGA, OMEGA_SQUARED),
    "C": (QOMEGA_ONE, OMEGA_SQUARED, OMEGA),
}

# The first, second, and third groups are the perfect matchings
# 01|23|45, 02|14|35, and 03|15|24.  Their only additional supported
# perfect matching is the rainbow matching 01|24|35.
PRISM_EDGE_DATA = (
    ((0, 1), "A", Fraction(2, 3)),
    ((2, 3), "A", Fraction(1, 6)),
    ((4, 5), "A", Fraction(1)),
    ((0, 2), "B", Fraction(1, 9)),
    ((1, 4), "B", Fraction(1)),
    ((3, 5), "B", Fraction(1)),
    ((0, 3), "C", Fraction(1, 9)),
    ((1, 5), "C", Fraction(1)),
    ((2, 4), "C", Fraction(1)),
)


def prism_equal_g_shadow_values() -> tuple[QOmega, ...]:
    """Return the explicit 90-coordinate ``Q(omega)`` shadow witness."""

    system = generate_equal_g_shadow_system(N, D)
    variable_index = {
        key: index for index, key in enumerate(system.variable_keys)
    }
    values = [QOMEGA_ZERO] * system.variable_count
    for (i, j), form_name, scalar in PRISM_EDGE_DATA:
        for color, coefficient in enumerate(PRISM_FORMS[form_name]):
            values[variable_index[(i, j, color, color)]] = (
                scalar * coefficient
            )
    return tuple(values)


def evaluate_cyclotomic_shadow(
    system: EqualGShadowSystem,
    values: Sequence[QOmega | ExactRational],
) -> tuple[QOmega, ...]:
    """Evaluate every equal-``g`` equation using exact pair arithmetic."""

    if not isinstance(system, EqualGShadowSystem):
        raise KrennCyclotomicShadowError(
            "a canonical equal-g shadow system is required"
        )
    if len(values) != system.variable_count:
        raise KrennCyclotomicShadowError(
            "cyclotomic shadow vector has the wrong length"
        )
    vector = tuple(QOmega.coerce(value) for value in values)
    result = []
    for equation in system.equation_terms:
        total = QOMEGA_ZERO
        for monomial in equation:
            term = QOMEGA_ONE
            for variable in monomial:
                term *= vector[variable]
            total += term
        result.append(total)
    return tuple(result)


def full_tensor_coloring_amplitude(
    coloring: Sequence[int],
    values: Sequence[QOmega | ExactRational] | None = None,
) -> QOmega:
    """Evaluate one full tensor coefficient of the diagonal lift.

    Off-diagonal full edge weights are zero.  Diagonal full weights agree
    exactly with the corresponding shadow coordinates, so this lift is
    unambiguous.
    """

    normalized = tuple(map(int, coloring))
    if len(normalized) != N or any(
        color < 0 or color >= D for color in normalized
    ):
        raise KrennCyclotomicShadowError(
            "coloring is not an element of range(3)^6"
        )
    vector = tuple(
        QOmega.coerce(value)
        for value in (
            prism_equal_g_shadow_values() if values is None else values
        )
    )
    system = generate_equal_g_shadow_system(N, D)
    if len(vector) != system.variable_count:
        raise KrennCyclotomicShadowError(
            "cyclotomic shadow vector has the wrong length"
        )
    by_key = dict(zip(system.variable_keys, vector, strict=True))
    total = QOMEGA_ZERO
    for matching in perfect_matchings(N):
        term = QOMEGA_ONE
        for i, j in matching:
            if normalized[i] != normalized[j]:
                term = QOMEGA_ZERO
                break
            color = normalized[i]
            term *= by_key[(i, j, color, color)]
        total += term
    return total


@dataclass(frozen=True)
class CyclotomicShadowCertificate:
    """Exact replay of the shadow witness and its full-tensor claim boundary."""

    values: tuple[QOmega, ...]
    equation_occupations: tuple[tuple[int, ...], ...]
    shadow_coefficients: tuple[QOmega, ...]
    target_coefficients: tuple[QOmega, ...]
    all_equal_amplitudes: tuple[QOmega, ...]
    victim_coloring: tuple[int, ...]
    victim_amplitude: QOmega

    @property
    def residual_coefficients(self) -> tuple[QOmega, ...]:
        return tuple(
            output - target
            for output, target in zip(
                self.shadow_coefficients,
                self.target_coefficients,
                strict=True,
            )
        )

    @property
    def all_28_shadow_equations_satisfied(self) -> bool:
        return (
            len(self.residual_coefficients) == 28
            and all(value.is_zero for value in self.residual_coefficients)
        )

    @property
    def full_tensor_victim_nonzero(self) -> bool:
        return not self.victim_amplitude.is_zero

    @property
    def full_tensor_solution_claimed(self) -> bool:
        return False


def certify_cyclotomic_shadow_witness() -> CyclotomicShadowCertificate:
    """Build and fail-closed replay the compact ``Q(omega)`` construction."""

    system = generate_equal_g_shadow_system(N, D)
    values = prism_equal_g_shadow_values()
    shadow = evaluate_cyclotomic_shadow(system, values)
    target = tuple(
        QOmega.coerce(value)
        for value in system.target_coefficients(
            canonical_ghz_target(N, D)
        )
    )
    certificate = CyclotomicShadowCertificate(
        values=values,
        equation_occupations=system.equation_occupations,
        shadow_coefficients=shadow,
        target_coefficients=target,
        all_equal_amplitudes=tuple(
            full_tensor_coloring_amplitude((color,) * N, values)
            for color in range(D)
        ),
        victim_coloring=VICTIM_COLORING,
        victim_amplitude=full_tensor_coloring_amplitude(
            VICTIM_COLORING, values
        ),
    )
    if not certificate.all_28_shadow_equations_satisfied:
        raise KrennCyclotomicShadowError(
            "explicit Q(omega) values failed an equal-g shadow equation"
        )
    if certificate.all_equal_amplitudes != (QOMEGA_ONE,) * D:
        raise KrennCyclotomicShadowError(
            "diagonal lift failed an all-equal tensor coefficient"
        )
    expected_victim = Fraction(2, 3) * OMEGA_SQUARED
    if certificate.victim_amplitude != expected_victim:
        raise KrennCyclotomicShadowError(
            "full-tensor victim coefficient changed"
        )
    if not certificate.full_tensor_victim_nonzero:
        raise KrennCyclotomicShadowError(
            "shadow-only witness unexpectedly became a full tensor witness"
        )
    return certificate
