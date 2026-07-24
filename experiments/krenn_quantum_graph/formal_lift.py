r"""Exact formal lifting of the natural ``n=6,d=3`` repair direction.

The natural nine-slot point ``W_0`` satisfies

.. math::

    \Phi(W_0)=\operatorname{GHZ}_{6,3}+e_{70},

where equation 70 is the coloring ``(0,0,2,1,2,1)``.  The full Jacobian
has only the five vertex-scalar gauge directions in its kernel, and the
known repair tangent maps to ``-e_70``.

This module lifts that tangent along the moving-target equation

.. math::

    \Phi(W(s))=\operatorname{GHZ}_{6,3}+(1-s)e_{70}.

Writing ``a=W[0,1,0,0]`` and ``b=W[2,3,0,0]``, the exact lift is

.. math::

    a(s)=1-s,\qquad b(s)=(1-s)^{-1}.

For every positive truncation order ``N``, the polynomial approximation

.. math::

    W^{(N)}=W_0-s e_a+(s+s^2+\cdots+s^N)e_b

obeys the full 729-equation identity

.. math::

    \Phi(W^{(N)})
      =\operatorname{GHZ}_{6,3}+(1-s)e_{70}-s^{N+1}e_0.

Thus the repair lifts over ``Q[[s]]`` with no cokernel obstruction at any
order.  It does *not* give a finite affine witness at ``s=1``.  A degree-six
vertex-gauge-invariant input monomial equals ``(1-s)^{-1}`` on this branch,
so its pole cannot be removed by the certified vertex gauge.

The exact full-kernel certificate and a five-by-five determinant ``-2`` also
show that the Jacobian is injective on the declared 130-dimensional linear
gauge slice.  Hence the displayed moving-target coefficients are recursively
unique inside that fixed local slice.  No global branch uniqueness is
claimed.

Only small exact polynomial arithmetic is used; no computer-algebra package
is required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import permutations

from experiments.krenn_quantum_graph.n6_deformation import (
    N6NaturalDeformationCertificate,
    certify_n6_natural_deformation,
    natural_seed_jacobian,
    vertex_scalar_gauge_matrix,
)
from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    coloring_from_index,
    generate_sparse_system,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    n6_d3_seed_witness,
)


FORMAL_LIFT_SCHEMA = "krenn-n6-d3-natural-formal-lift-v1"
FORMAL_TRUNCATION_SCHEMA = "krenn-n6-d3-formal-truncation-v1"
GAUGE_SLICE_INJECTIVITY_SCHEMA = (
    "krenn-n6-d3-formal-gauge-slice-injectivity-v1"
)
DEFAULT_TRUNCATION_ORDER = 4

ALL_ZERO_EQUATION = 0
ALL_ONE_EQUATION = 364
ALL_TWO_EQUATION = 728

REPAIR_DECREASE_INDEX = variable_index(6, 3, 0, 1, 0, 0)
REPAIR_INCREASE_INDEX = variable_index(6, 3, 2, 3, 0, 0)

GAUGE_SLICE_INDICES = (
    variable_index(6, 3, 0, 2, 1, 1),
    variable_index(6, 3, 0, 3, 2, 2),
    variable_index(6, 3, 1, 4, 1, 1),
    variable_index(6, 3, 1, 5, 2, 2),
    variable_index(6, 3, 4, 5, 0, 0),
)

POLE_INVARIANT_INDICES = (
    variable_index(6, 3, 0, 2, 1, 1),
    variable_index(6, 3, 0, 3, 2, 2),
    variable_index(6, 3, 1, 4, 1, 1),
    variable_index(6, 3, 1, 5, 2, 2),
    variable_index(6, 3, 2, 3, 0, 0),
    variable_index(6, 3, 4, 5, 0, 0),
)

NATURAL_SUPPORT_MONOMIALS = (
    (
        ALL_ZERO_EQUATION,
        (
            variable_index(6, 3, 0, 1, 0, 0),
            variable_index(6, 3, 2, 3, 0, 0),
            variable_index(6, 3, 4, 5, 0, 0),
        ),
    ),
    (
        N6_D3_SEED_DEFECT_EQUATION,
        (
            variable_index(6, 3, 0, 1, 0, 0),
            variable_index(6, 3, 2, 4, 2, 2),
            variable_index(6, 3, 3, 5, 1, 1),
        ),
    ),
    (
        ALL_ONE_EQUATION,
        (
            variable_index(6, 3, 0, 2, 1, 1),
            variable_index(6, 3, 1, 4, 1, 1),
            variable_index(6, 3, 3, 5, 1, 1),
        ),
    ),
    (
        ALL_TWO_EQUATION,
        (
            variable_index(6, 3, 0, 3, 2, 2),
            variable_index(6, 3, 1, 5, 2, 2),
            variable_index(6, 3, 2, 4, 2, 2),
        ),
    ),
)

ExactScalar = int | Fraction
ExactVector = tuple[Fraction, ...]
IntegerMatrix = tuple[tuple[int, ...], ...]


class KrennFormalLiftError(RuntimeError):
    """The formal lift, derivative, gauge, or pole replay failed."""


def _as_fraction(value: ExactScalar) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise KrennFormalLiftError(
            "formal-lift coefficients must be exact integers or Fractions"
        )
    return Fraction(value)


def _validate_order(order: int) -> int:
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise KrennFormalLiftError(
            "a formal truncation order must be a positive integer"
        )
    return order


@dataclass(frozen=True)
class ExactPolynomial:
    """A canonical polynomial over ``Q`` in the variable ``s``."""

    terms: tuple[tuple[int, Fraction], ...] = ()

    def __post_init__(self) -> None:
        combined: dict[int, Fraction] = {}
        try:
            raw_terms = tuple(self.terms)
            for raw_exponent, raw_coefficient in raw_terms:
                if (
                    isinstance(raw_exponent, bool)
                    or not isinstance(raw_exponent, int)
                    or raw_exponent < 0
                ):
                    raise KrennFormalLiftError(
                        "polynomial exponents must be nonnegative integers"
                    )
                coefficient = _as_fraction(raw_coefficient)
                combined[raw_exponent] = (
                    combined.get(raw_exponent, Fraction(0))
                    + coefficient
                )
        except (TypeError, ValueError) as error:
            raise KrennFormalLiftError(
                "polynomial terms must be exponent-coefficient pairs"
            ) from error
        object.__setattr__(
            self,
            "terms",
            tuple(
                (exponent, coefficient)
                for exponent, coefficient in sorted(combined.items())
                if coefficient
            ),
        )

    @classmethod
    def constant(cls, value: ExactScalar) -> ExactPolynomial:
        coefficient = _as_fraction(value)
        return cls(()) if not coefficient else cls(((0, coefficient),))

    @classmethod
    def monomial(
        cls, exponent: int, coefficient: ExactScalar = 1
    ) -> ExactPolynomial:
        if (
            isinstance(exponent, bool)
            or not isinstance(exponent, int)
            or exponent < 0
        ):
            raise KrennFormalLiftError(
                "a polynomial monomial needs a nonnegative exponent"
            )
        exact = _as_fraction(coefficient)
        return cls(()) if not exact else cls(((exponent, exact),))

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def degree(self) -> int:
        return -1 if self.is_zero else self.terms[-1][0]

    def __bool__(self) -> bool:
        return not self.is_zero

    def coefficient(self, exponent: int) -> Fraction:
        if (
            isinstance(exponent, bool)
            or not isinstance(exponent, int)
            or exponent < 0
        ):
            raise KrennFormalLiftError(
                "a coefficient lookup needs a nonnegative exponent"
            )
        return next(
            (
                coefficient
                for term_exponent, coefficient in self.terms
                if term_exponent == exponent
            ),
            Fraction(0),
        )

    def __add__(
        self, other: ExactPolynomial | ExactScalar
    ) -> ExactPolynomial:
        other = _as_polynomial(other)
        return ExactPolynomial((*self.terms, *other.terms))

    def __radd__(
        self, other: ExactPolynomial | ExactScalar
    ) -> ExactPolynomial:
        return self + other

    def __neg__(self) -> ExactPolynomial:
        return ExactPolynomial(
            tuple(
                (exponent, -coefficient)
                for exponent, coefficient in self.terms
            )
        )

    def __sub__(
        self, other: ExactPolynomial | ExactScalar
    ) -> ExactPolynomial:
        return self + (-_as_polynomial(other))

    def __rsub__(
        self, other: ExactPolynomial | ExactScalar
    ) -> ExactPolynomial:
        return _as_polynomial(other) - self

    def __mul__(
        self, other: ExactPolynomial | ExactScalar
    ) -> ExactPolynomial:
        other = _as_polynomial(other)
        return ExactPolynomial(
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
        self, other: ExactPolynomial | ExactScalar
    ) -> ExactPolynomial:
        return self * other

    def __pow__(self, exponent: int) -> ExactPolynomial:
        if (
            isinstance(exponent, bool)
            or not isinstance(exponent, int)
            or exponent < 0
        ):
            raise KrennFormalLiftError(
                "a polynomial power needs a nonnegative integer"
            )
        result = POLYNOMIAL_ONE
        factor = self
        remaining = exponent
        while remaining:
            if remaining % 2:
                result *= factor
            factor *= factor
            remaining //= 2
        return result

    def specialize(self, value: ExactScalar) -> Fraction:
        point = _as_fraction(value)
        return sum(
            coefficient * point**exponent
            for exponent, coefficient in self.terms
        )

    def to_expression(self, variable: str = "s") -> str:
        if not isinstance(variable, str) or not variable.isidentifier():
            raise KrennFormalLiftError(
                "the polynomial display variable must be an identifier"
            )
        if self.is_zero:
            return "0"
        pieces: list[str] = []
        for exponent, coefficient in self.terms:
            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(coefficient)
            if exponent == 0:
                body = _fraction_text(magnitude)
            else:
                power = variable if exponent == 1 else f"{variable}^{exponent}"
                body = (
                    power
                    if magnitude == 1
                    else f"{_fraction_text(magnitude)}*{power}"
                )
            if not pieces:
                pieces.append(body if sign == "+" else f"-{body}")
            else:
                pieces.append(f" {sign} {body}")
        return "".join(pieces)

    def to_dict(self) -> dict:
        return {
            "variable": "s",
            "expression": self.to_expression(),
            "terms": [
                {
                    "exponent": exponent,
                    "coefficient": _fraction_text(coefficient),
                }
                for exponent, coefficient in self.terms
            ],
        }


def _as_polynomial(
    value: ExactPolynomial | ExactScalar,
) -> ExactPolynomial:
    if isinstance(value, ExactPolynomial):
        return value
    return ExactPolynomial.constant(value)


def _fraction_text(value: Fraction) -> str:
    value = Fraction(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


POLYNOMIAL_ZERO = ExactPolynomial()
POLYNOMIAL_ONE = ExactPolynomial.constant(1)
POLYNOMIAL_S = ExactPolynomial.monomial(1)


@lru_cache(maxsize=1)
def _system() -> SparsePolynomialSystem:
    return generate_sparse_system(6, 3)


@lru_cache(maxsize=1)
def _jacobian() -> tuple[tuple[int, ...], ...]:
    return natural_seed_jacobian()


@lru_cache(maxsize=1)
def _gauge_matrix() -> tuple[tuple[int, ...], ...]:
    return vertex_scalar_gauge_matrix()


@lru_cache(maxsize=1)
def _deformation_certificate() -> N6NaturalDeformationCertificate:
    """Return the exact source certificate for ``ker(J)=gauge``."""

    return certify_n6_natural_deformation()


def geometric_polynomial(order: int) -> ExactPolynomial:
    """Return ``1+s+...+s^order`` exactly."""

    order = _validate_order(order)
    return ExactPolynomial(
        tuple((exponent, Fraction(1)) for exponent in range(order + 1))
    )


def formal_truncation_weight_entries(
    order: int,
) -> tuple[tuple[int, ExactPolynomial], ...]:
    """Return the nine nonzero entries of ``W^(order)(s)``."""

    order = _validate_order(order)
    values = {
        index: ExactPolynomial.constant(value)
        for index, value in n6_d3_seed_witness().entries
    }
    values[REPAIR_DECREASE_INDEX] = POLYNOMIAL_ONE - POLYNOMIAL_S
    values[REPAIR_INCREASE_INDEX] = geometric_polynomial(order)
    result = tuple(sorted(values.items()))
    if len(result) != 9:
        raise KrennFormalLiftError(
            "the formal truncation must retain nine nonzero slots"
        )
    return result


def evaluate_polynomial_system(
    system: SparsePolynomialSystem,
    entries: Mapping[int, ExactPolynomial]
    | Sequence[tuple[int, ExactPolynomial]],
) -> tuple[ExactPolynomial, ...]:
    """Evaluate every matching equation over the exact ring ``Q[s]``."""

    if not isinstance(system, SparsePolynomialSystem):
        raise KrennFormalLiftError(
            "formal polynomial replay needs a sparse polynomial system"
        )
    raw_items = entries.items() if isinstance(entries, Mapping) else entries
    values: dict[int, ExactPolynomial] = {}
    seen: set[int] = set()
    try:
        for raw_index, raw_value in raw_items:
            index = int(raw_index)
            if index in seen:
                raise KrennFormalLiftError(
                    f"duplicate formal weight index {index}"
                )
            seen.add(index)
            if not 0 <= index < system.variable_count:
                raise KrennFormalLiftError(
                    "a formal weight index is outside the system"
                )
            if not isinstance(raw_value, ExactPolynomial):
                raise KrennFormalLiftError(
                    "formal weights must use ExactPolynomial values"
                )
            if not raw_value.is_zero:
                values[index] = raw_value
    except (TypeError, ValueError) as error:
        raise KrennFormalLiftError(
            "formal entries must be an index-polynomial mapping or sequence"
        ) from error

    output: list[ExactPolynomial] = []
    for equation in range(system.equation_count):
        total = POLYNOMIAL_ZERO
        for monomial in system.equation_monomials(equation):
            if any(variable not in values for variable in monomial):
                continue
            term = POLYNOMIAL_ONE
            for variable in monomial:
                term *= values[variable]
            total += term
        output.append(total)
    return tuple(output)


def expected_truncation_output(
    order: int,
) -> tuple[ExactPolynomial, ...]:
    r"""Return ``GHZ+(1-s)e_70-s^(order+1)e_0``."""

    order = _validate_order(order)
    output = [
        ExactPolynomial.constant(value) for value in _system().rhs_values
    ]
    output[N6_D3_SEED_DEFECT_EQUATION] += (
        POLYNOMIAL_ONE - POLYNOMIAL_S
    )
    output[ALL_ZERO_EQUATION] -= ExactPolynomial.monomial(order + 1)
    return tuple(output)


def natural_support_surviving_monomials(
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return all matching monomials supported by the natural nine slots."""

    system = _system()
    support = {
        index for index, _value in n6_d3_seed_witness().entries
    }
    return tuple(
        (equation, monomial)
        for equation in range(system.equation_count)
        for monomial in system.equation_monomials(equation)
        if all(variable in support for variable in monomial)
    )


@dataclass(frozen=True)
class FormalTruncationReplay:
    """Full exact replay of one polynomial formal truncation."""

    order: int
    weight_entries: tuple[tuple[int, ExactPolynomial], ...]
    output_coefficients: tuple[ExactPolynomial, ...]
    expected_coefficients: tuple[ExactPolynomial, ...]
    schema: str = FORMAL_TRUNCATION_SCHEMA

    def __post_init__(self) -> None:
        order = _validate_order(self.order)
        if self.schema != FORMAL_TRUNCATION_SCHEMA:
            raise KrennFormalLiftError(
                "the formal truncation schema changed"
            )
        canonical_entries = formal_truncation_weight_entries(order)
        if self.weight_entries != canonical_entries:
            raise KrennFormalLiftError(
                "formal truncation weight entries are not canonical"
            )
        replayed = evaluate_polynomial_system(_system(), self.weight_entries)
        expected = expected_truncation_output(order)
        if (
            len(self.output_coefficients) != 729
            or len(self.expected_coefficients) != 729
            or self.output_coefficients != replayed
            or self.expected_coefficients != expected
            or replayed != expected
        ):
            raise KrennFormalLiftError(
                "the 729-equation formal truncation identity failed"
            )

    @property
    def exact(self) -> bool:
        return self.output_coefficients == self.expected_coefficients

    @property
    def residual_from_moving_target(
        self,
    ) -> tuple[ExactPolynomial, ...]:
        moving_target = [
            ExactPolynomial.constant(value)
            for value in _system().rhs_values
        ]
        moving_target[N6_D3_SEED_DEFECT_EQUATION] += (
            POLYNOMIAL_ONE - POLYNOMIAL_S
        )
        return tuple(
            output - target
            for output, target in zip(
                self.output_coefficients, moving_target
            )
        )

    def to_dict(self) -> dict:
        nonzero_output = [
            {
                "equation": equation,
                "coloring": list(coloring_from_index(6, 3, equation)),
                "coefficient": coefficient.to_expression(),
            }
            for equation, coefficient in enumerate(
                self.output_coefficients
            )
            if not coefficient.is_zero
        ]
        residual = [
            {
                "equation": equation,
                "coloring": list(coloring_from_index(6, 3, equation)),
                "coefficient": coefficient.to_expression(),
            }
            for equation, coefficient in enumerate(
                self.residual_from_moving_target
            )
            if not coefficient.is_zero
        ]
        return {
            "schema": self.schema,
            "order": self.order,
            "input_family": [
                {
                    "index": index,
                    "coordinate": list(variable_key(6, 3, index)),
                    "polynomial": polynomial.to_expression(),
                }
                for index, polynomial in self.weight_entries
            ],
            "full_tensor_equation_count": len(self.output_coefficients),
            "identity": (
                "Phi(W^(N)(s)) = GHZ_6,3 + (1-s)*e_70 "
                "- s^(N+1)*e_0"
            ),
            "nonzero_output_coefficients": nonzero_output,
            "moving_target_residual": residual,
            "exact": self.exact,
        }


def replay_formal_truncation(order: int) -> FormalTruncationReplay:
    """Construct and verify the exact order-``N`` identity."""

    order = _validate_order(order)
    entries = formal_truncation_weight_entries(order)
    output = evaluate_polynomial_system(_system(), entries)
    return FormalTruncationReplay(
        order=order,
        weight_entries=entries,
        output_coefficients=output,
        expected_coefficients=expected_truncation_output(order),
    )


def _exact_dense_direction(
    entries: Mapping[int, ExactScalar],
) -> ExactVector:
    result = [Fraction(0)] * variable_count(6, 3)
    for index, raw_value in entries.items():
        index = int(index)
        if not 0 <= index < len(result):
            raise KrennFormalLiftError(
                "a formal direction index is outside the system"
            )
        result[index] += _as_fraction(raw_value)
    return tuple(result)


def first_order_repair_direction() -> ExactVector:
    """Return ``v_1=-e_a+e_b``."""

    return _exact_dense_direction(
        {
            REPAIR_DECREASE_INDEX: -1,
            REPAIR_INCREASE_INDEX: 1,
        }
    )


def recurrent_higher_order_direction() -> ExactVector:
    """Return the common coefficient ``v_k=e_b`` for every ``k>=2``."""

    return _exact_dense_direction({REPAIR_INCREASE_INDEX: 1})


def _validate_direction(direction: Sequence[ExactScalar]) -> ExactVector:
    try:
        result = tuple(_as_fraction(value) for value in direction)
    except TypeError as error:
        raise KrennFormalLiftError(
            "a derivative direction must be an exact dense sequence"
        ) from error
    if len(result) != variable_count(6, 3):
        raise KrennFormalLiftError(
            "a derivative direction must have 135 coordinates"
        )
    return result


def jacobian_action(direction: Sequence[ExactScalar]) -> ExactVector:
    """Apply the exact natural-seed Jacobian to a dense direction."""

    direction = _validate_direction(direction)
    return tuple(
        sum(
            Fraction(value) * direction[column]
            for column, value in enumerate(row)
        )
        for row in _jacobian()
    )


def hessian_action(
    left: Sequence[ExactScalar],
    right: Sequence[ExactScalar],
) -> ExactVector:
    """Return ``D^2 Phi(W_0)[left,right]`` exactly."""

    left = _validate_direction(left)
    right = _validate_direction(right)
    values = dict(n6_d3_seed_witness().entries)
    output: list[Fraction] = []
    for equation in range(_system().equation_count):
        total = Fraction(0)
        for monomial in _system().equation_monomials(equation):
            for left_position in range(len(monomial)):
                left_value = left[monomial[left_position]]
                if not left_value:
                    continue
                for right_position in range(len(monomial)):
                    if right_position == left_position:
                        continue
                    right_value = right[monomial[right_position]]
                    if not right_value:
                        continue
                    term = left_value * right_value
                    for position, variable in enumerate(monomial):
                        if position not in (left_position, right_position):
                            term *= values.get(variable, Fraction(0))
                    total += term
        output.append(total)
    return tuple(output)


def third_derivative_action(
    first: Sequence[ExactScalar],
    second: Sequence[ExactScalar],
    third: Sequence[ExactScalar],
) -> ExactVector:
    """Return the constant trilinear derivative ``D^3 Phi``."""

    directions = (
        _validate_direction(first),
        _validate_direction(second),
        _validate_direction(third),
    )
    output: list[Fraction] = []
    for equation in range(_system().equation_count):
        total = Fraction(0)
        for monomial in _system().equation_monomials(equation):
            for positions in permutations(range(len(monomial)), 3):
                term = Fraction(1)
                for direction, position in zip(directions, positions):
                    term *= direction[monomial[position]]
                total += term
        output.append(total)
    return tuple(output)


def _output_basis(equation: int, value: ExactScalar = 1) -> ExactVector:
    equation = int(equation)
    if not 0 <= equation < 729:
        raise KrennFormalLiftError(
            "an output-basis equation is outside the system"
        )
    result = [Fraction(0)] * 729
    result[equation] = _as_fraction(value)
    return tuple(result)


def _scale_vector(vector: ExactVector, scalar: ExactScalar) -> ExactVector:
    exact = _as_fraction(scalar)
    return tuple(exact * value for value in vector)


def _sparse_vector(vector: Sequence[ExactScalar]) -> list[dict]:
    exact = tuple(_as_fraction(value) for value in vector)
    return [
        {
            "index": index,
            "value": _fraction_text(value),
        }
        for index, value in enumerate(exact)
        if value
    ]


def gauge_slice_matrix() -> IntegerMatrix:
    """Return the five gauge rows fixed by the local rational slice."""

    gauge = _gauge_matrix()
    return tuple(gauge[index] for index in GAUGE_SLICE_INDICES)


def determinant_over_q(
    matrix: Sequence[Sequence[ExactScalar]],
) -> Fraction:
    """Compute a small square determinant by exact elimination."""

    rows = [
        [_as_fraction(value) for value in row] for row in matrix
    ]
    size = len(rows)
    if not size or any(len(row) != size for row in rows):
        raise KrennFormalLiftError(
            "an exact determinant needs a nonempty square matrix"
        )
    determinant = Fraction(1)
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
            determinant = -determinant
        pivot = rows[column][column]
        determinant *= pivot
        for entry in range(column, size):
            rows[column][entry] /= pivot
        for row in range(column + 1, size):
            factor = rows[row][column]
            if factor:
                for entry in range(column, size):
                    rows[row][entry] -= factor * rows[column][entry]
    return determinant


@dataclass(frozen=True)
class GaugeSliceInjectivityCertificate:
    r"""Exact proof that ``J`` is injective on the fixed linear slice.

    Let ``G`` be the five-column vertex-gauge tangent matrix and

    ``S = {v in Q^135 : v_i=0 for i in GAUGE_SLICE_INDICES}``.

    The source deformation certificate proves ``ker(J)=im(G)``.  Restricting
    the five gauge columns to the five fixed coordinates gives the matrix
    stored here.  Its determinant is ``-2``, so ``im(G)`` meets ``S`` only
    at zero.  Consequently ``ker(J|_S)=0`` and the ``729 x 130`` restricted
    Jacobian has rank 130 over ``Q``.
    """

    fixed_indices: tuple[int, ...]
    gauge_restriction_matrix: IntegerMatrix
    gauge_restriction_determinant: Fraction
    ambient_dimension: int
    slice_dimension: int
    full_jacobian_rank_over_q: int
    full_jacobian_nullity_over_q: int
    vertex_gauge_dimension: int
    kernel_slice_intersection_dimension: int
    restricted_jacobian_rank_over_q: int
    restricted_jacobian_nullity_over_q: int
    schema: str = GAUGE_SLICE_INJECTIVITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GAUGE_SLICE_INJECTIVITY_SCHEMA:
            raise KrennFormalLiftError(
                "the gauge-slice injectivity schema changed"
            )
        failed = [
            name for name, passed in self.exact_checks.items() if not passed
        ]
        if failed:
            raise KrennFormalLiftError(
                "the gauge-slice injectivity certificate failed: "
                f"{failed}"
            )

    @property
    def exact_checks(self) -> dict[str, bool]:
        source = _deformation_certificate()
        source_checks = source.exact_checks
        source_kernel_exact = (
            source.exact
            and source_checks.get(
                "vertex_gauge_rank_5_over_Q", False
            )
            and source_checks.get(
                "full_rank_sandwich_proves_rank_130_over_Q", False
            )
            and source_checks.get(
                "full_kernel_equals_gauge_over_Q", False
            )
        )
        fixed_indices_exact = (
            self.fixed_indices == GAUGE_SLICE_INDICES
            and len(self.fixed_indices) == 5
            and len(set(self.fixed_indices)) == 5
            and all(
                0 <= index < variable_count(6, 3)
                for index in self.fixed_indices
            )
        )
        matrix_exact = (
            self.gauge_restriction_matrix
            == gauge_slice_matrix()
            and len(self.gauge_restriction_matrix) == 5
            and all(
                len(row) == 5
                for row in self.gauge_restriction_matrix
            )
        )
        determinant_exact = (
            matrix_exact
            and self.gauge_restriction_determinant
            == determinant_over_q(self.gauge_restriction_matrix)
            == -2
        )
        dimensions_exact = (
            self.ambient_dimension == variable_count(6, 3) == 135
            and self.slice_dimension
            == self.ambient_dimension - len(self.fixed_indices)
            == 130
            and self.full_jacobian_rank_over_q
            == source.full_minor.size
            == 130
            and self.full_jacobian_nullity_over_q
            == self.ambient_dimension
            - self.full_jacobian_rank_over_q
            == 5
            and self.vertex_gauge_dimension == 5
        )
        trivial_intersection = (
            source_kernel_exact
            and fixed_indices_exact
            and determinant_exact
            and self.kernel_slice_intersection_dimension == 0
        )
        restricted_injective = (
            trivial_intersection
            and self.restricted_jacobian_nullity_over_q
            == self.kernel_slice_intersection_dimension
            == 0
            and self.restricted_jacobian_rank_over_q
            == self.slice_dimension
            - self.restricted_jacobian_nullity_over_q
            == 130
        )
        return {
            "source_deformation_certificate_is_exact": source.exact,
            "source_full_kernel_is_exactly_vertex_gauge": (
                source_kernel_exact
            ),
            "declared_slice_fixes_five_distinct_coordinates": (
                fixed_indices_exact
            ),
            "declared_slice_has_dimension_130_over_Q": dimensions_exact,
            "gauge_restriction_matrix_replayed_exactly": matrix_exact,
            "gauge_restriction_determinant_is_minus_2": (
                determinant_exact
            ),
            "kernel_meets_declared_slice_only_at_zero": (
                trivial_intersection
            ),
            "restricted_jacobian_has_rank_130_nullity_0_over_Q": (
                restricted_injective
            ),
        }

    @property
    def exact(self) -> bool:
        return all(self.exact_checks.values())

    @property
    def restricted_jacobian_injective_over_q(self) -> bool:
        return self.exact and self.restricted_jacobian_nullity_over_q == 0

    def to_dict(self) -> dict:
        checks = self.exact_checks
        exact = all(checks.values())
        return {
            "schema": self.schema,
            "source_full_jacobian_certificate": {
                "schema": _deformation_certificate().schema,
                "ambient_dimension": self.ambient_dimension,
                "rank_over_Q": self.full_jacobian_rank_over_q,
                "nullity_over_Q": self.full_jacobian_nullity_over_q,
                "kernel_equals_vertex_scalar_gauge_over_Q": checks[
                    "source_full_kernel_is_exactly_vertex_gauge"
                ],
                "vertex_gauge_dimension": self.vertex_gauge_dimension,
            },
            "fixed_linear_slice": {
                "definition": (
                    "v_i=0 for each listed fixed input coordinate"
                ),
                "fixed_indices": list(self.fixed_indices),
                "fixed_coordinates": [
                    list(variable_key(6, 3, index))
                    for index in self.fixed_indices
                ],
                "ambient_dimension": self.ambient_dimension,
                "codimension": len(self.fixed_indices),
                "dimension_over_Q": self.slice_dimension,
            },
            "kernel_intersection_proof": {
                "gauge_restriction_matrix": [
                    list(row)
                    for row in self.gauge_restriction_matrix
                ],
                "determinant_over_Q": _fraction_text(
                    self.gauge_restriction_determinant
                ),
                "kernel_slice_intersection_dimension_over_Q": (
                    self.kernel_slice_intersection_dimension
                ),
                "reason": (
                    "ker(J)=im(G), and the fixed-coordinate restriction "
                    "of G is invertible because its determinant is -2"
                ),
            },
            "restricted_jacobian": {
                "shape": [729, self.slice_dimension],
                "rank_over_Q": self.restricted_jacobian_rank_over_q,
                "nullity_over_Q": (
                    self.restricted_jacobian_nullity_over_q
                ),
                "injective_over_Q": (
                    self.restricted_jacobian_injective_over_q
                ),
            },
            "exact_checks": checks,
            "claim_boundary": {
                "proved_at_natural_seed": exact,
                "proved_for_declared_fixed_linear_slice": exact,
                "global_nonlinear_slice_claimed": False,
                "global_branch_uniqueness_claimed": False,
            },
        }


def certify_gauge_slice_injectivity(
) -> GaugeSliceInjectivityCertificate:
    """Certify the exact rank-130 Jacobian lock on the fixed slice."""

    source = _deformation_certificate()
    matrix = gauge_slice_matrix()
    ambient_dimension = variable_count(6, 3)
    slice_dimension = ambient_dimension - len(GAUGE_SLICE_INDICES)
    full_rank = source.full_minor.size
    return GaugeSliceInjectivityCertificate(
        fixed_indices=GAUGE_SLICE_INDICES,
        gauge_restriction_matrix=matrix,
        gauge_restriction_determinant=determinant_over_q(matrix),
        ambient_dimension=ambient_dimension,
        slice_dimension=slice_dimension,
        full_jacobian_rank_over_q=full_rank,
        full_jacobian_nullity_over_q=ambient_dimension - full_rank,
        vertex_gauge_dimension=len(matrix[0]),
        kernel_slice_intersection_dimension=0,
        restricted_jacobian_rank_over_q=slice_dimension,
        restricted_jacobian_nullity_over_q=0,
    )


def pole_invariant_vertex_degrees() -> tuple[int, ...]:
    """Return the six vertex degrees of the invariant monomial."""

    degrees = [0] * 6
    for index in POLE_INVARIANT_INDICES:
        i, j, _a, _b = variable_key(6, 3, index)
        degrees[i] += 1
        degrees[j] += 1
    return tuple(degrees)


def pole_invariant_truncation(order: int) -> ExactPolynomial:
    """Evaluate the gauge-invariant monomial on ``W^(order)``."""

    values = dict(formal_truncation_weight_entries(order))
    result = POLYNOMIAL_ONE
    for index in POLE_INVARIANT_INDICES:
        try:
            result *= values[index]
        except KeyError as error:
            raise KrennFormalLiftError(
                "the pole invariant left the natural support"
            ) from error
    return result


def _support_table() -> tuple[tuple[int, tuple[int, ...]], ...]:
    return natural_support_surviving_monomials()


@dataclass(frozen=True)
class FormalLiftCertificate:
    """Exact all-order local lift and gauge-invariant pole certificate."""

    truncation: FormalTruncationReplay
    jacobian_decrease_column: ExactVector
    jacobian_increase_column: ExactVector
    first_order_image: ExactVector
    half_second_derivative: ExactVector
    mixed_third_order_derivative: ExactVector
    cubic_first_order_derivative: ExactVector
    gauge_matrix: IntegerMatrix
    gauge_determinant: Fraction
    slice_injectivity: GaugeSliceInjectivityCertificate
    invariant_vertex_degrees: tuple[int, ...]
    invariant_truncation: ExactPolynomial
    support_monomials: tuple[tuple[int, tuple[int, ...]], ...]
    schema: str = FORMAL_LIFT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_LIFT_SCHEMA:
            raise KrennFormalLiftError(
                "the formal-lift certificate schema changed"
            )
        failed = [
            name for name, passed in self.exact_checks.items() if not passed
        ]
        if failed:
            raise KrennFormalLiftError(
                f"the formal-lift certificate failed: {failed}"
            )

    @property
    def exact_checks(self) -> dict[str, bool]:
        first = first_order_repair_direction()
        higher = recurrent_higher_order_direction()
        zero_basis = _output_basis(ALL_ZERO_EQUATION)
        defect_basis = _output_basis(
            N6_D3_SEED_DEFECT_EQUATION
        )
        denominator = POLYNOMIAL_ONE - POLYNOMIAL_S
        return {
            "full_729_equation_truncation_identity": (
                self.truncation.exact
                and self.truncation.output_coefficients
                == expected_truncation_output(self.truncation.order)
            ),
            "natural_support_has_exactly_four_surviving_monomials": (
                self.support_monomials
                == NATURAL_SUPPORT_MONOMIALS
                == _support_table()
            ),
            "jacobian_decrease_column_is_e0_plus_e70": (
                self.jacobian_decrease_column
                == jacobian_action(
                    _exact_dense_direction(
                        {REPAIR_DECREASE_INDEX: 1}
                    )
                )
                == tuple(
                    left + right
                    for left, right in zip(zero_basis, defect_basis)
                )
            ),
            "jacobian_increase_column_is_e0": (
                self.jacobian_increase_column
                == jacobian_action(higher)
                == zero_basis
            ),
            "first_order_repair_maps_to_minus_e70": (
                self.first_order_image
                == jacobian_action(first)
                == _scale_vector(defect_basis, -1)
            ),
            "half_hessian_is_minus_e0": (
                self.half_second_derivative
                == _scale_vector(hessian_action(first, first), Fraction(1, 2))
                == _scale_vector(zero_basis, -1)
            ),
            "third_order_mixed_hessian_is_minus_e0": (
                self.mixed_third_order_derivative
                == hessian_action(first, higher)
                == _scale_vector(zero_basis, -1)
            ),
            "third_derivative_on_first_order_direction_is_zero": (
                self.cubic_first_order_derivative
                == third_derivative_action(first, first, first)
                == _output_basis(ALL_ZERO_EQUATION, 0)
            ),
            "five_coordinate_gauge_slice_has_determinant_minus_2": (
                self.gauge_matrix
                == gauge_slice_matrix()
                and self.gauge_determinant
                == determinant_over_q(self.gauge_matrix)
                == -2
            ),
            "restricted_jacobian_is_injective_rank_130_on_slice": (
                self.slice_injectivity.exact
                and self.slice_injectivity
                == certify_gauge_slice_injectivity()
                and self.slice_injectivity.gauge_restriction_matrix
                == self.gauge_matrix
                and self.slice_injectivity.gauge_restriction_determinant
                == self.gauge_determinant
                and self.slice_injectivity.slice_dimension == 130
                and self.slice_injectivity
                .restricted_jacobian_rank_over_q
                == 130
                and self.slice_injectivity
                .restricted_jacobian_nullity_over_q
                == 0
            ),
            "formal_directions_lie_in_declared_gauge_slice": all(
                first[index] == 0 and higher[index] == 0
                for index in GAUGE_SLICE_INDICES
            ),
            "moving_target_coefficients_recursively_unique_in_slice": (
                self.slice_injectivity
                .restricted_jacobian_injective_over_q
                and all(
                    first[index] == 0 and higher[index] == 0
                    for index in GAUGE_SLICE_INDICES
                )
                and self.truncation.exact
            ),
            "pole_monomial_has_vertex_degree_two": (
                self.invariant_vertex_degrees
                == pole_invariant_vertex_degrees()
                == (2, 2, 2, 2, 2, 2)
            ),
            "pole_invariant_truncates_to_geometric_series": (
                self.invariant_truncation
                == pole_invariant_truncation(self.truncation.order)
                == geometric_polynomial(self.truncation.order)
            ),
            "geometric_identity_has_denominator_zero_at_s_one": (
                denominator * self.invariant_truncation
                == (
                    POLYNOMIAL_ONE
                    - ExactPolynomial.monomial(
                        self.truncation.order + 1
                    )
                )
                and denominator.specialize(1) == 0
            ),
        }

    @property
    def exact(self) -> bool:
        return all(self.exact_checks.values())

    def to_dict(self) -> dict:
        checks = self.exact_checks
        exact = all(checks.values())
        order = self.truncation.order
        return {
            "schema": self.schema,
            "parameters": {
                "n": 6,
                "d": 3,
                "formal_variable": "s",
                "truncation_order": order,
                "all_zero_equation": ALL_ZERO_EQUATION,
                "defect_equation": N6_D3_SEED_DEFECT_EQUATION,
                "defect_coloring": list(
                    coloring_from_index(
                        6, 3, N6_D3_SEED_DEFECT_EQUATION
                    )
                ),
            },
            "moving_target": {
                "equation": (
                    "Phi(W(s)) = GHZ_6,3 + (1-s)*e_70"
                ),
                "constant_GHZ_fiber": False,
                "seed_occurs_at_s": 0,
                "GHZ_endpoint_would_occur_at_s": 1,
            },
            "truncation_replay": self.truncation.to_dict(),
            "derivative_replay": {
                "decrease_coordinate": {
                    "index": REPAIR_DECREASE_INDEX,
                    "coordinate": list(
                        variable_key(
                            6, 3, REPAIR_DECREASE_INDEX
                        )
                    ),
                    "jacobian_column": _sparse_vector(
                        self.jacobian_decrease_column
                    ),
                },
                "increase_coordinate": {
                    "index": REPAIR_INCREASE_INDEX,
                    "coordinate": list(
                        variable_key(
                            6, 3, REPAIR_INCREASE_INDEX
                        )
                    ),
                    "jacobian_column": _sparse_vector(
                        self.jacobian_increase_column
                    ),
                },
                "v_1": _sparse_vector(first_order_repair_direction()),
                "v_k_for_every_k_at_least_2": _sparse_vector(
                    recurrent_higher_order_direction()
                ),
                "J_v_1": _sparse_vector(self.first_order_image),
                "one_half_D2_v1_v1": _sparse_vector(
                    self.half_second_derivative
                ),
                "D2_v1_v2": _sparse_vector(
                    self.mixed_third_order_derivative
                ),
                "D3_v1_v1_v1": _sparse_vector(
                    self.cubic_first_order_derivative
                ),
                "cokernel_obstruction_class": "zero at every order",
                "exact_preimage_of_each_higher_order_rhs": (
                    "e_81, since J*e_81=e_0"
                ),
            },
            "gauge_quotient": {
                "fixed_indices": list(GAUGE_SLICE_INDICES),
                "fixed_coordinates": [
                    list(variable_key(6, 3, index))
                    for index in GAUGE_SLICE_INDICES
                ],
                "gauge_exponent_matrix": [
                    list(row) for row in self.gauge_matrix
                ],
                "determinant_over_Q": _fraction_text(
                    self.gauge_determinant
                ),
                "local_rational_slice_certified": exact,
                "formal_coefficients_are_in_this_slice": checks[
                    "formal_directions_lie_in_declared_gauge_slice"
                ],
                "restricted_jacobian_injectivity": (
                    self.slice_injectivity.to_dict()
                ),
            },
            "recursive_uniqueness": {
                "coefficient_equation": (
                    "At order k, J|_S * v_k equals a right-hand side "
                    "determined by the moving target and v_1,...,v_(k-1)"
                ),
                "linear_operator": (
                    "natural-seed Jacobian restricted to the declared "
                    "130-dimensional fixed linear slice S"
                ),
                "restricted_jacobian_rank_over_Q": (
                    self.slice_injectivity
                    .restricted_jacobian_rank_over_q
                ),
                "restricted_jacobian_nullity_over_Q": (
                    self.slice_injectivity
                    .restricted_jacobian_nullity_over_q
                ),
                "displayed_coefficients_lie_in_slice": checks[
                    "formal_directions_lie_in_declared_gauge_slice"
                ],
                "moving_target_formal_lift_unique_in_fixed_slice": checks[
                    "moving_target_coefficients_recursively_unique_in_slice"
                ],
                "scope": (
                    "recursive uniqueness for this moving-target lift "
                    "inside this fixed local linear slice at W_0"
                ),
                "global_branch_uniqueness_claimed": False,
                "uniqueness_in_other_gauge_slices_claimed": False,
            },
            "gauge_invariant_pole": {
                "input_indices": list(POLE_INVARIANT_INDICES),
                "input_coordinates": [
                    list(variable_key(6, 3, index))
                    for index in POLE_INVARIANT_INDICES
                ],
                "vertex_degrees": list(self.invariant_vertex_degrees),
                "gauge_multiplier": "(product_i lambda_i)^2 = 1",
                "order_N_value": (
                    self.invariant_truncation.to_expression()
                ),
                "formal_value": "1/(1-s)",
                "denominator": "1-s",
                "denominator_at_s_1": 0,
                "pole_survives_vertex_gauge": exact,
            },
            "exact_checks": checks,
            "claim_boundary": {
                "first_order_repair_lifts_to_every_finite_order": exact,
                "moving_target_has_full_Q_power_series_lift": exact,
                "moving_target_formal_lift_unique_in_declared_linear_slice": (
                    checks[
                        "moving_target_coefficients_recursively_unique_in_slice"
                    ]
                ),
                "global_formal_branch_uniqueness_proved": False,
                "formal_uniqueness_in_other_gauge_slices_proved": False,
                "cokernel_obstruction_found_along_this_branch": False,
                "formal_arc_lies_in_constant_GHZ_fiber": False,
                "power_series_specialization_at_s_1_is_defined": False,
                "finite_polynomial_lift_in_certified_gauge_slice": False,
                "finite_affine_GHZ_witness_produced": False,
                "exact_affine_GHZ_image_membership_proved": False,
                "exact_affine_GHZ_image_membership_status": "undecided",
            },
        }


def certify_n6_d3_formal_lift(
    truncation_order: int = DEFAULT_TRUNCATION_ORDER,
) -> FormalLiftCertificate:
    """Build the exact local lift, derivative, gauge, and pole certificate."""

    order = _validate_order(truncation_order)
    truncation = replay_formal_truncation(order)
    first = first_order_repair_direction()
    higher = recurrent_higher_order_direction()
    jacobian_decrease = jacobian_action(
        _exact_dense_direction({REPAIR_DECREASE_INDEX: 1})
    )
    jacobian_increase = jacobian_action(higher)
    slice_injectivity = certify_gauge_slice_injectivity()
    return FormalLiftCertificate(
        truncation=truncation,
        jacobian_decrease_column=jacobian_decrease,
        jacobian_increase_column=jacobian_increase,
        first_order_image=jacobian_action(first),
        half_second_derivative=_scale_vector(
            hessian_action(first, first), Fraction(1, 2)
        ),
        mixed_third_order_derivative=hessian_action(first, higher),
        cubic_first_order_derivative=third_derivative_action(
            first, first, first
        ),
        gauge_matrix=gauge_slice_matrix(),
        gauge_determinant=determinant_over_q(gauge_slice_matrix()),
        slice_injectivity=slice_injectivity,
        invariant_vertex_degrees=pole_invariant_vertex_degrees(),
        invariant_truncation=pole_invariant_truncation(order),
        support_monomials=_support_table(),
    )


__all__ = (
    "ALL_ZERO_EQUATION",
    "DEFAULT_TRUNCATION_ORDER",
    "ExactPolynomial",
    "FORMAL_LIFT_SCHEMA",
    "FORMAL_TRUNCATION_SCHEMA",
    "FormalLiftCertificate",
    "FormalTruncationReplay",
    "GAUGE_SLICE_INDICES",
    "GAUGE_SLICE_INJECTIVITY_SCHEMA",
    "GaugeSliceInjectivityCertificate",
    "KrennFormalLiftError",
    "NATURAL_SUPPORT_MONOMIALS",
    "POLE_INVARIANT_INDICES",
    "POLYNOMIAL_ONE",
    "POLYNOMIAL_S",
    "POLYNOMIAL_ZERO",
    "REPAIR_DECREASE_INDEX",
    "REPAIR_INCREASE_INDEX",
    "certify_n6_d3_formal_lift",
    "certify_gauge_slice_injectivity",
    "determinant_over_q",
    "evaluate_polynomial_system",
    "expected_truncation_output",
    "first_order_repair_direction",
    "formal_truncation_weight_entries",
    "gauge_slice_matrix",
    "geometric_polynomial",
    "hessian_action",
    "jacobian_action",
    "natural_support_surviving_monomials",
    "pole_invariant_truncation",
    "pole_invariant_vertex_degrees",
    "recurrent_higher_order_direction",
    "replay_formal_truncation",
    "third_derivative_action",
)
