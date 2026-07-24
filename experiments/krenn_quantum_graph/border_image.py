r"""Exact Laurent degeneration into the ``n=6,d=3`` GHZ border image.

The natural nine-slot seed has one mixed coefficient.  A reciprocal
rescaling of two color-zero weights gives the Laurent family

``W[0,1,0,0]=t`` and ``W[2,3,0,0]=t^-1``,

with the other seven natural seed weights equal to one.  This module replays
all 729 perfect-matching coefficients symbolically over ``Q[t,t^-1]`` and
certifies

.. math::

    \Phi(W(t)) = \operatorname{GHZ}_{6,3}
                 + t\,e_{(0,0,2,1,2,1)}.

The input weights have a pole at ``t=0``, while the output has a polynomial
extension whose specialization there is GHZ.  This proves membership in the
Zariski closure of the image over ``Q`` and hence over ``C``.  It does not
provide a finite weight vector at ``t=0`` and does not decide exact affine
image membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    VariableKey,
    coloring_from_index,
    generate_sparse_system,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    N6_D3_SEED_FACTORS,
)


BORDER_IMAGE_SCHEMA = "krenn-n6-d3-laurent-border-certificate-v1"


class KrennBorderImageError(ValueError):
    """A Laurent polynomial, weight family, or replay is malformed."""


@dataclass(frozen=True)
class LaurentPolynomial:
    """A canonical finite Laurent polynomial over ``Q`` in one variable."""

    terms: tuple[tuple[int, Fraction], ...] = ()

    def __post_init__(self) -> None:
        combined: dict[int, Fraction] = {}
        try:
            for raw_exponent, raw_coefficient in self.terms:
                exponent = int(raw_exponent)
                coefficient = Fraction(raw_coefficient)
                if coefficient:
                    combined[exponent] = (
                        combined.get(exponent, Fraction(0)) + coefficient
                    )
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise KrennBorderImageError(
                "Laurent terms must be exact exponent-coefficient pairs"
            ) from error
        normalized = tuple(
            (exponent, coefficient)
            for exponent, coefficient in sorted(combined.items())
            if coefficient
        )
        object.__setattr__(self, "terms", normalized)

    @classmethod
    def constant(cls, value: int | Fraction) -> LaurentPolynomial:
        coefficient = Fraction(value)
        return cls(()) if not coefficient else cls(((0, coefficient),))

    @classmethod
    def monomial(
        cls, exponent: int, coefficient: int | Fraction = 1
    ) -> LaurentPolynomial:
        coefficient = Fraction(coefficient)
        return (
            cls(())
            if not coefficient
            else cls(((int(exponent), coefficient),))
        )

    def __add__(
        self, other: LaurentPolynomial | int | Fraction
    ) -> LaurentPolynomial:
        other = _as_laurent(other)
        return LaurentPolynomial((*self.terms, *other.terms))

    def __radd__(
        self, other: LaurentPolynomial | int | Fraction
    ) -> LaurentPolynomial:
        return self + other

    def __neg__(self) -> LaurentPolynomial:
        return LaurentPolynomial(
            tuple(
                (exponent, -coefficient)
                for exponent, coefficient in self.terms
            )
        )

    def __sub__(
        self, other: LaurentPolynomial | int | Fraction
    ) -> LaurentPolynomial:
        return self + (-_as_laurent(other))

    def __rsub__(
        self, other: LaurentPolynomial | int | Fraction
    ) -> LaurentPolynomial:
        return _as_laurent(other) - self

    def __mul__(
        self, other: LaurentPolynomial | int | Fraction
    ) -> LaurentPolynomial:
        other = _as_laurent(other)
        return LaurentPolynomial(
            tuple(
                (
                    left_exponent + right_exponent,
                    left_coefficient * right_coefficient,
                )
                for left_exponent, left_coefficient in self.terms
                for right_exponent, right_coefficient in other.terms
            )
        )

    def __rmul__(
        self, other: LaurentPolynomial | int | Fraction
    ) -> LaurentPolynomial:
        return self * other

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def is_polynomial(self) -> bool:
        return all(exponent >= 0 for exponent, _coefficient in self.terms)

    @property
    def has_pole_at_zero(self) -> bool:
        return any(exponent < 0 for exponent, _coefficient in self.terms)

    def coefficient(self, exponent: int) -> Fraction:
        exponent = int(exponent)
        return next(
            (
                coefficient
                for candidate, coefficient in self.terms
                if candidate == exponent
            ),
            Fraction(0),
        )

    def specialize_zero(self) -> Fraction:
        """Specialize a polynomial at zero, rejecting a Laurent pole."""

        if not self.is_polynomial:
            raise KrennBorderImageError(
                "a Laurent polynomial with a pole cannot specialize at zero"
            )
        return self.coefficient(0)

    def to_expression(self, variable: str = "t") -> str:
        if not variable or not variable.isidentifier():
            raise KrennBorderImageError(
                "the Laurent display variable must be an identifier"
            )
        if self.is_zero:
            return "0"
        pieces = []
        for exponent, coefficient in self.terms:
            if exponent == 0:
                monomial = "1"
            elif exponent == 1:
                monomial = variable
            else:
                monomial = f"{variable}^{exponent}"
            if monomial == "1":
                pieces.append(str(coefficient))
            elif coefficient == 1:
                pieces.append(monomial)
            elif coefficient == -1:
                pieces.append(f"-{monomial}")
            else:
                pieces.append(f"{coefficient}*{monomial}")
        return " + ".join(pieces).replace("+ -", "- ")


def _as_laurent(
    value: LaurentPolynomial | int | Fraction,
) -> LaurentPolynomial:
    return (
        value
        if isinstance(value, LaurentPolynomial)
        else LaurentPolynomial.constant(value)
    )


LAURENT_ZERO = LaurentPolynomial()
LAURENT_ONE = LaurentPolynomial.constant(1)
LAURENT_T = LaurentPolynomial.monomial(1)
LAURENT_T_INVERSE = LaurentPolynomial.monomial(-1)


def natural_laurent_weight_entries(
) -> tuple[tuple[int, LaurentPolynomial], ...]:
    """Return the exact nine-slot Laurent family in variable-index order."""

    coordinates: dict[VariableKey, LaurentPolynomial] = {}
    for color, matching in enumerate(N6_D3_SEED_FACTORS):
        for i, j in matching:
            coordinates[(i, j, color, color)] = LAURENT_ONE
    coordinates[(0, 1, 0, 0)] = LAURENT_T
    coordinates[(2, 3, 0, 0)] = LAURENT_T_INVERSE
    entries = tuple(
        sorted(
            (
                variable_index(6, 3, *key),
                value,
            )
            for key, value in coordinates.items()
        )
    )
    if len(entries) != 9:
        raise KrennBorderImageError(
            "the natural Laurent family must have exactly nine slots"
        )
    return entries


def evaluate_laurent_system(
    system: SparsePolynomialSystem,
    entries: Mapping[int, LaurentPolynomial]
    | Sequence[tuple[int, LaurentPolynomial]],
) -> tuple[LaurentPolynomial, ...]:
    """Replay every matching monomial over ``Q[t,t^-1]``."""

    raw_items = entries.items() if isinstance(entries, Mapping) else entries
    values: dict[int, LaurentPolynomial] = {}
    seen: set[int] = set()
    try:
        for raw_index, raw_value in raw_items:
            index = int(raw_index)
            if index in seen:
                raise KrennBorderImageError(
                    f"duplicate Laurent weight index {index}"
                )
            seen.add(index)
            if not 0 <= index < system.variable_count:
                raise KrennBorderImageError(
                    "a Laurent weight index is outside the system"
                )
            if not isinstance(raw_value, LaurentPolynomial):
                raise KrennBorderImageError(
                    "Laurent weights must use LaurentPolynomial values"
                )
            if not raw_value.is_zero:
                values[index] = raw_value
    except TypeError as error:
        raise KrennBorderImageError(
            "Laurent entries must be an index-value mapping or sequence"
        ) from error

    output = []
    for equation in range(system.equation_count):
        total = LAURENT_ZERO
        for monomial in system.equation_monomials(equation):
            term = LAURENT_ONE
            for variable in monomial:
                term *= values.get(variable, LAURENT_ZERO)
            total += term
        output.append(total)
    return tuple(output)


@dataclass(frozen=True)
class LaurentBorderCertificate:
    """Full coefficient replay and fail-closed border-image claim record."""

    weight_entries: tuple[tuple[int, LaurentPolynomial], ...]
    output_coefficients: tuple[LaurentPolynomial, ...]
    expected_coefficients: tuple[LaurentPolynomial, ...]
    specialized_coefficients_at_zero: tuple[Fraction, ...]
    target_coefficients: tuple[Fraction, ...]
    defect_equation: int
    defect_coloring: tuple[int, ...]
    exact_checks: Mapping[str, bool]
    schema: str = BORDER_IMAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BORDER_IMAGE_SCHEMA:
            raise KrennBorderImageError(
                "Laurent border certificate schema changed"
            )
        if (
            len(self.output_coefficients) != 3**6
            or len(self.expected_coefficients) != 3**6
            or len(self.specialized_coefficients_at_zero) != 3**6
            or len(self.target_coefficients) != 3**6
        ):
            raise KrennBorderImageError(
                "the Laurent certificate must contain all 729 coefficients"
            )
        if self.defect_equation != N6_D3_SEED_DEFECT_EQUATION or (
            self.defect_coloring
            != coloring_from_index(
                6, 3, N6_D3_SEED_DEFECT_EQUATION
            )
        ):
            raise KrennBorderImageError(
                "the Laurent certificate defect convention changed"
            )
        canonical_system = generate_sparse_system(6, 3)
        canonical_target = tuple(
            Fraction(value) for value in canonical_system.rhs_values
        )
        canonical_expected = tuple(
            LaurentPolynomial.constant(target_value)
            + (
                LAURENT_T
                if equation == N6_D3_SEED_DEFECT_EQUATION
                else LAURENT_ZERO
            )
            for equation, target_value in enumerate(canonical_target)
        )
        if (
            self.weight_entries != natural_laurent_weight_entries()
            or self.target_coefficients != canonical_target
            or self.expected_coefficients != canonical_expected
            or self.output_coefficients != self.expected_coefficients
            or not all(
                coefficient.is_polynomial
                for coefficient in self.output_coefficients
            )
            or self.specialized_coefficients_at_zero
            != tuple(
                coefficient.specialize_zero()
                for coefficient in self.output_coefficients
            )
            or self.specialized_coefficients_at_zero
            != self.target_coefficients
        ):
            raise KrennBorderImageError(
                "Laurent border certificate content was not canonical"
            )
        if not self.exact:
            failed = [
                name for name, passed in self.exact_checks.items() if not passed
            ]
            raise KrennBorderImageError(
                f"Laurent border certificate failed: {failed}"
            )

    @property
    def exact(self) -> bool:
        return bool(self.exact_checks) and all(self.exact_checks.values())

    @property
    def border_image_membership_proved(self) -> bool:
        return self.exact

    @property
    def exact_affine_image_membership_proved(self) -> bool:
        return False

    @property
    def finite_weight_witness_at_zero(self) -> bool:
        return False

    def to_dict(self) -> dict:
        nonzero_output = [
            {
                "equation": equation,
                "coloring": list(coloring_from_index(6, 3, equation)),
                "coefficient": coefficient.to_expression(),
            }
            for equation, coefficient in enumerate(self.output_coefficients)
            if not coefficient.is_zero
        ]
        return {
            "schema": self.schema,
            "parameters": {
                "n": 6,
                "d": 3,
                "parameter": "t",
                "input_coefficient_ring": "Q[t,t^-1]",
                "output_extension_ring": "Q[t]",
            },
            "weight_family": [
                {
                    "variable_index": index,
                    "coordinate": list(variable_key(6, 3, index)),
                    "value": value.to_expression(),
                }
                for index, value in self.weight_entries
            ],
            "full_tensor_replay": {
                "coefficient_count": len(self.output_coefficients),
                "matching_terms_checked": 3**6 * 15,
                "nonzero_coefficients": nonzero_output,
                "identity": (
                    "Phi(W(t)) = GHZ_6,3 + t*e_(0,0,2,1,2,1)"
                ),
                "defect_equation": self.defect_equation,
                "defect_coloring": list(self.defect_coloring),
            },
            "symbolic_specialization": {
                "input_has_pole_at_t_zero": True,
                "output_is_regular_at_t_zero": True,
                "output_at_t_zero_equals_GHZ": True,
            },
            "exact_checks": dict(self.exact_checks),
            "claim_boundary": {
                "Q_defined_Laurent_degeneration": True,
                "GHZ_in_Zariski_closure_over_Q": True,
                "GHZ_in_Zariski_closure_over_C": True,
                "border_image_membership_proved": True,
                "finite_weight_witness_at_t_zero": False,
                "exact_affine_image_membership_proved": False,
                "exact_affine_image_membership_status": "undecided",
            },
        }


def certify_n6_d3_laurent_border() -> LaurentBorderCertificate:
    """Certify the natural full-tensor border degeneration exactly."""

    system = generate_sparse_system(6, 3)
    entries = natural_laurent_weight_entries()
    output = evaluate_laurent_system(system, entries)
    target = tuple(Fraction(value) for value in system.rhs_values)
    expected = tuple(
        LaurentPolynomial.constant(target_value)
        + (
            LAURENT_T
            if equation == N6_D3_SEED_DEFECT_EQUATION
            else LAURENT_ZERO
        )
        for equation, target_value in enumerate(target)
    )
    output_regular = all(value.is_polynomial for value in output)
    specialized = (
        tuple(value.specialize_zero() for value in output)
        if output_regular
        else ()
    )
    entry_values = dict(entries)
    pole_index = variable_index(6, 3, 2, 3, 0, 0)
    checks = {
        "canonical_n6_d3_variable_count_135": (
            system.variable_count == variable_count(6, 3) == 135
        ),
        "all_729_coefficients_replayed": len(output) == 729,
        "all_10935_matching_terms_replayed": (
            system.monomial_count == 729 * 15 == 10_935
        ),
        "natural_family_has_exactly_nine_nonzero_slots": len(entries) == 9,
        "unique_input_pole_is_W_2_3_0_0": (
            tuple(
                index
                for index, value in entries
                if value.has_pole_at_zero
            )
            == (pole_index,)
            and entry_values[pole_index] == LAURENT_T_INVERSE
        ),
        "full_tensor_identity_over_Q_t_t_inverse": output == expected,
        "all_negative_exponents_cancel_from_output": output_regular,
        "symbolic_output_specialization_at_zero_exists": (
            len(specialized) == 729
        ),
        "symbolic_output_specialization_equals_GHZ": (
            specialized == target
        ),
        "sole_nontarget_coefficient_is_t_at_002121": (
            output[N6_D3_SEED_DEFECT_EQUATION]
            - LaurentPolynomial.constant(
                target[N6_D3_SEED_DEFECT_EQUATION]
            )
            == LAURENT_T
            and coloring_from_index(
                6, 3, N6_D3_SEED_DEFECT_EQUATION
            )
            == (0, 0, 2, 1, 2, 1)
        ),
        "input_family_not_finite_at_t_zero": any(
            value.has_pole_at_zero for _index, value in entries
        ),
    }
    return LaurentBorderCertificate(
        weight_entries=entries,
        output_coefficients=output,
        expected_coefficients=expected,
        specialized_coefficients_at_zero=specialized,
        target_coefficients=target,
        defect_equation=N6_D3_SEED_DEFECT_EQUATION,
        defect_coloring=coloring_from_index(
            6, 3, N6_D3_SEED_DEFECT_EQUATION
        ),
        exact_checks=checks,
    )


__all__ = (
    "BORDER_IMAGE_SCHEMA",
    "KrennBorderImageError",
    "LAURENT_ONE",
    "LAURENT_T",
    "LAURENT_T_INVERSE",
    "LAURENT_ZERO",
    "LaurentBorderCertificate",
    "LaurentPolynomial",
    "certify_n6_d3_laurent_border",
    "evaluate_laurent_system",
    "natural_laurent_weight_entries",
)
