r"""Exact front end and verifier for the degree-24 ``n=8`` blocker chart.

The three-vertex Tutte barrier leaves five outside vertices.  Its blocker
edges form ``K_5``, so after normalizing one color coordinate at every
outside vertex there are ten bi-affine equations in ten variables.  The
generic projective intersection number is 24.

This module deliberately separates three logically different operations:

* deterministic construction of one dense rational specialization;
* export to a separately bounded Singular Gröbner-basis computation; and
* native exact verification of a returned finite-dimensional quotient
  representation.

A nonzero unital representation by commuting multiplication matrices on
which all ten equations vanish proves that the specialized affine ideal is
proper.  Over ``Q`` this proves an affine point exists over an algebraic
closure.  A representation over ``F_p`` proves only the modular statement.
Neither conclusion constrains a hypothetical ``n=8`` Krenn solution because
the deterministic specialization does not satisfy or impose the EqSystem.

The verifier does not trust a backend claim that the representation is the
full quotient.  Full-quotient or radical-membership claims require additional
Gröbner or ideal-membership certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Mapping, Sequence, TypeAlias

from experiments.krenn_quantum_graph.perfect_matching_blocker_ideals import (
    AffinePolynomial,
    PerfectMatchingBlockerType,
    TutteBarrierAffineChart,
    build_tutte_barrier_affine_chart,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


DEFAULT_SPECIALIZATION_SEED = 80320260725
DEFAULT_COEFFICIENT_HEIGHT = 127
DEFAULT_MODULUS = 31
MAXIMUM_QUOTIENT_DIMENSION = 64
MAXIMUM_EXPONENT = 1_000_000
MAXIMUM_TRANSCRIPT_BYTES = 8_000_000
MAXIMUM_BACKEND_POLYNOMIAL_TERMS = 128
MAXIMUM_BACKEND_FACTORS_PER_TERM = 80
MAXIMUM_COEFFICIENT_CHARACTERS = 4_096
PARSE_MARKER = "KRENN_N8_K5_BLOCKER_PARSE_OK"
COMPLETION_MARKER = "KRENN_N8_K5_BLOCKER_QUOTIENT_DONE"
SCHEMA = "krenn-n8-k5-blocker-quotient-reconnaissance-v1"
TRANSCRIPT_SCHEMA = "krenn-n8-k5-blocker-singular-transcript-v1"

Exponent: TypeAlias = tuple[int, ...]
FieldScalar: TypeAlias = int | Fraction
Matrix: TypeAlias = tuple[tuple[FieldScalar, ...], ...]
GeneralPolynomial: TypeAlias = tuple[tuple[Exponent, Fraction], ...]


class KrennBlockerQuotientError(ValueError):
    """A quotient input, backend transcript, or exact replay is malformed."""


def _exact_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KrennBlockerQuotientError(
            f"{label} must be an exact integer"
        )
    return value


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _characteristic(value: int) -> int:
    value = _exact_integer(value, "field characteristic")
    if value != 0 and not _is_prime(value):
        raise KrennBlockerQuotientError(
            "field characteristic must be zero or prime"
        )
    return value


def _field_scalar(
    value: FieldScalar,
    characteristic: int,
    label: str,
) -> FieldScalar:
    if characteristic == 0:
        if isinstance(value, bool) or not isinstance(
            value, (int, Fraction)
        ):
            raise KrennBlockerQuotientError(
                f"{label} must be an exact rational"
            )
        return Fraction(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise KrennBlockerQuotientError(
            f"{label} must be an exact modular integer"
        )
    return value % characteristic


def _exact_fraction(value: object, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(
        value, (int, Fraction)
    ):
        raise KrennBlockerQuotientError(
            f"{label} must be an exact integer or rational"
        )
    return Fraction(value)


def _fraction_in_field(
    value: int | Fraction,
    characteristic: int,
) -> FieldScalar:
    value = _exact_fraction(value, "field coefficient")
    if characteristic == 0:
        return value
    denominator = value.denominator % characteristic
    if denominator == 0:
        raise KrennBlockerQuotientError(
            "a rational denominator vanishes in the selected field"
        )
    return (
        value.numerator
        * pow(denominator, -1, characteristic)
    ) % characteristic


def _add(
    left: FieldScalar,
    right: FieldScalar,
    characteristic: int,
) -> FieldScalar:
    value = left + right
    return value if characteristic == 0 else value % characteristic


def _multiply(
    left: FieldScalar,
    right: FieldScalar,
    characteristic: int,
) -> FieldScalar:
    value = left * right
    return value if characteristic == 0 else value % characteristic


def _zero(characteristic: int) -> FieldScalar:
    return Fraction(0) if characteristic == 0 else 0


def _one(characteristic: int) -> FieldScalar:
    return Fraction(1) if characteristic == 0 else 1


def _zero_matrix(
    dimension: int,
    characteristic: int,
) -> list[list[FieldScalar]]:
    return [
        [_zero(characteristic) for _column in range(dimension)]
        for _row in range(dimension)
    ]


def _identity_matrix(
    dimension: int,
    characteristic: int,
) -> Matrix:
    return tuple(
        tuple(
            _one(characteristic)
            if row == column
            else _zero(characteristic)
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _matrix_add(
    left: Matrix,
    right: Matrix,
    characteristic: int,
) -> Matrix:
    return tuple(
        tuple(
            _add(left[row][column], right[row][column], characteristic)
            for column in range(len(left))
        )
        for row in range(len(left))
    )


def _matrix_scale(
    scalar: FieldScalar,
    matrix: Matrix,
    characteristic: int,
) -> Matrix:
    return tuple(
        tuple(
            _multiply(scalar, value, characteristic)
            for value in row
        )
        for row in matrix
    )


def _matrix_multiply(
    left: Matrix,
    right: Matrix,
    characteristic: int,
) -> Matrix:
    dimension = len(left)
    output = _zero_matrix(dimension, characteristic)
    for row in range(dimension):
        for middle in range(dimension):
            coefficient = left[row][middle]
            if not coefficient:
                continue
            for column in range(dimension):
                if right[middle][column]:
                    output[row][column] = _add(
                        output[row][column],
                        _multiply(
                            coefficient,
                            right[middle][column],
                            characteristic,
                        ),
                        characteristic,
                    )
    return tuple(tuple(row) for row in output)


def _matrix_vector(
    matrix: Matrix,
    vector: Sequence[FieldScalar],
    characteristic: int,
) -> tuple[FieldScalar, ...]:
    result = []
    for row in matrix:
        total = _zero(characteristic)
        for coefficient, value in zip(row, vector, strict=True):
            total = _add(
                total,
                _multiply(coefficient, value, characteristic),
                characteristic,
            )
        result.append(total)
    return tuple(result)


def _matrix_power(
    matrix: Matrix,
    exponent: int,
    characteristic: int,
) -> Matrix:
    exponent = _exact_integer(exponent, "matrix exponent")
    if exponent < 0:
        raise KrennBlockerQuotientError(
            "matrix exponent must be nonnegative"
        )
    result = _identity_matrix(len(matrix), characteristic)
    factor = matrix
    while exponent:
        if exponent & 1:
            result = _matrix_multiply(
                result, factor, characteristic
            )
        exponent //= 2
        if exponent:
            factor = _matrix_multiply(
                factor, factor, characteristic
            )
    return result


def _matrix_is_zero(matrix: Matrix) -> bool:
    return all(not value for row in matrix for value in row)


def _specialization_digest(seed: int, label: str) -> bytes:
    return hashlib.sha256(
        f"krenn-n8-k5-v2:{seed}:{label}".encode("ascii")
    ).digest()


def _deterministic_distinct_coefficients(
    slots: Sequence[tuple[int, int, int, int]],
    *,
    seed: int,
    height: int,
) -> dict[tuple[int, int, int, int], int]:
    """Assign distinct nonzero values, also nonzero modulo the pilot prime."""

    candidates = tuple(
        value
        for value in range(-height, height + 1)
        if value and value % DEFAULT_MODULUS
    )
    if len(candidates) < len(slots):
        raise KrennBlockerQuotientError(
            "coefficient height leaves too few distinct values "
            "that remain nonzero modulo the pilot prime"
        )
    ordered_slots = sorted(
        slots,
        key=lambda slot: _specialization_digest(
            seed, "slot:" + ":".join(map(str, slot))
        ),
    )
    ordered_values = sorted(
        candidates,
        key=lambda value: _specialization_digest(
            seed, f"value:{value}"
        ),
    )[: len(slots)]
    return dict(zip(ordered_slots, ordered_values, strict=True))


def deterministic_k5_blocker_chart(
    *,
    seed: int = DEFAULT_SPECIALIZATION_SEED,
    color: int = 0,
    coefficient_height: int = DEFAULT_COEFFICIENT_HEIGHT,
) -> TutteBarrierAffineChart:
    """Build one dense, distinctly weighted rational ``K_5`` chart."""

    seed = _exact_integer(seed, "specialization seed")
    color = _exact_integer(color, "normalized color")
    coefficient_height = _exact_integer(
        coefficient_height, "coefficient height"
    )
    if seed < 0 or seed >= 2**63:
        raise KrennBlockerQuotientError(
            "specialization seed must lie in range(2^63)"
        )
    if color not in range(3):
        raise KrennBlockerQuotientError(
            "normalized color must lie in range(3)"
        )
    if not 1 <= coefficient_height <= 1_000:
        raise KrennBlockerQuotientError(
            "coefficient height must lie in range(1,1001)"
        )
    blocker_type = PerfectMatchingBlockerType(
        8, 3, (1, 1, 1, 1, 1)
    )
    slots = tuple(
        (edge[0], edge[1], first_color, second_color)
        for edge in blocker_type.blocker_edges
        for first_color in range(3)
        for second_color in range(3)
    )
    coordinates = _deterministic_distinct_coefficients(
        slots, seed=seed, height=coefficient_height
    )
    witness = SparseWitness.from_coordinates(8, 3, coordinates)
    return build_tutte_barrier_affine_chart(
        witness, blocker_type, color
    )


def chart_payload(
    chart: TutteBarrierAffineChart,
) -> dict[str, object]:
    """Return the canonical exact input payload used for hashing."""

    return {
        "n": chart.n,
        "d": chart.d,
        "color": chart.color,
        "barrier_vertices": list(chart.barrier_vertices),
        "components": [list(component) for component in chart.components],
        "blocker_edges": [list(edge) for edge in chart.blocker_edges],
        "variable_keys": [list(key) for key in chart.variable_keys],
        "equations": [
            [
                {
                    "monomial": list(monomial),
                    "coefficient": [
                        coefficient.numerator,
                        coefficient.denominator,
                    ],
                }
                for monomial, coefficient in equation
            ]
            for equation in chart.equations
        ],
    }


def chart_sha256(chart: TutteBarrierAffineChart) -> str:
    payload = json.dumps(
        chart_payload(chart),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _singular_affine_polynomial(
    equation: AffinePolynomial,
    characteristic: int,
) -> str:
    pieces: list[str] = []
    for monomial, coefficient in equation:
        value = _fraction_in_field(coefficient, characteristic)
        if not value:
            continue
        negative = characteristic == 0 and value < 0
        magnitude = -value if negative else value
        if characteristic:
            scalar = str(magnitude)
        else:
            rational = Fraction(magnitude)
            scalar = (
                str(rational.numerator)
                if rational.denominator == 1
                else f"({rational.numerator}/{rational.denominator})"
            )
        variables = [f"x{index}" for index in monomial]
        term = "*".join((scalar, *variables))
        if not pieces:
            pieces.append(f"-{term}" if negative else term)
        else:
            pieces.append(("-" if negative else "+") + term)
    return "".join(pieces) if pieces else "0"


def singular_blocker_quotient_script(
    chart: TutteBarrierAffineChart,
    *,
    characteristic: int = DEFAULT_MODULUS,
    algorithm: str = "std",
) -> str:
    """Export a deterministic affine quotient probe to Singular."""

    characteristic = _characteristic(characteristic)
    if algorithm not in {"std", "slimgb"}:
        raise KrennBlockerQuotientError(
            "Singular algorithm must be 'std' or 'slimgb'"
        )
    if (
        chart.n != 8
        or chart.d != 3
        or len(chart.barrier_vertices) != 3
        or tuple(len(component) for component in chart.components)
        != (1, 1, 1, 1, 1)
        or chart.variable_count != 10
        or chart.equation_count != 10
    ):
        raise KrennBlockerQuotientError(
            "quotient export requires the square n=8 K5 blocker chart"
        )
    names = ",".join(
        f"x{index}" for index in range(chart.variable_count)
    )
    equations = ",\n  ".join(
        _singular_affine_polynomial(equation, characteristic)
        for equation in chart.equations
    )
    digest = chart_sha256(chart)
    variables = ",".join(
        f"x{index}" for index in range(chart.variable_count)
    )
    return "\n".join(
        (
            "// Deterministic n=8 K5 blocker quotient reconnaissance.",
            "// The normalized affine chart uses no saturation variable.",
            f"// chart_sha256={digest}",
            "option(redSB);",
            f"ring r={characteristic},({names}),dp;",
            f"ideal I=\n  {equations};",
            f"ideal V={variables};",
            f'print("{PARSE_MARKER}");',
            f'print("transcript_schema={TRANSCRIPT_SCHEMA}");',
            f'print("chart_sha256={digest}");',
            f'print("characteristic={characteristic}");',
            f'print("algorithm={algorithm}");',
            f'print("variable_count={chart.variable_count}");',
            "int started=timer;",
            f"ideal G={algorithm}(I);",
            "int elapsed=timer-started;",
            "int isunit=(size(G)==1 && G[1]==1);",
            "int krulldim=dim(G);",
            'print("timer_ticks="+string(elapsed));',
            'print("basis_size="+string(size(G)));',
            'print("unit_ideal="+string(isunit));',
            'print("krull_dimension="+string(krulldim));',
            "if (isunit==0 && krulldim==0)",
            "{",
            "  int quotientdim=vdim(G);",
            "  ideal B=kbase(G);",
            '  print("quotient_dimension="+string(quotientdim));',
            '  print("standard_basis_size="+string(size(B)));',
            "  int i;",
            "  int j;",
            "  for (j=1;j<=size(B);j++)",
            "  {",
            '    print("standard_basis["'
            '+string(j-1)+"]="+string(B[j]));',
            "  }",
            "  for (i=1;i<=size(V);i++)",
            "  {",
            "    for (j=1;j<=size(B);j++)",
            "    {",
            '      print("normal_form["'
            '+string(i-1)+","+string(j-1)+"]="'
            "+string(reduce(V[i]*B[j],G)));",
            "    }",
            "  }",
            "}",
            f'print("{COMPLETION_MARKER}");',
            "",
        )
    )


def _normalize_exponent(
    exponent: Sequence[int],
    variable_count: int,
) -> Exponent:
    values = tuple(
        _exact_integer(value, "monomial exponent")
        for value in exponent
    )
    if len(values) != variable_count or any(
        value < 0 or value > MAXIMUM_EXPONENT for value in values
    ):
        raise KrennBlockerQuotientError(
            "a monomial exponent vector has the wrong shape"
        )
    return values


@dataclass(frozen=True)
class QuotientRepresentation:
    """A finite exact cyclic representation of a polynomial quotient."""

    characteristic: int
    variable_count: int
    chart_sha256: str
    basis_monomials: tuple[Exponent, ...]
    multiplication_matrices: tuple[Matrix, ...]
    backend_reported_quotient_dimension: int

    def __post_init__(self) -> None:
        characteristic = _characteristic(self.characteristic)
        variable_count = _exact_integer(
            self.variable_count, "representation variable count"
        )
        reported = _exact_integer(
            self.backend_reported_quotient_dimension,
            "backend quotient dimension",
        )
        if variable_count < 1:
            raise KrennBlockerQuotientError(
                "representation variable count must be positive"
            )
        basis = tuple(
            _normalize_exponent(monomial, variable_count)
            for monomial in self.basis_monomials
        )
        dimension = len(basis)
        if (
            not isinstance(self.chart_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.chart_sha256) is None
        ):
            raise KrennBlockerQuotientError(
                "representation chart hash must be lowercase SHA-256"
            )
        if (
            not 1 <= dimension <= MAXIMUM_QUOTIENT_DIMENSION
            or len(set(basis)) != dimension
            or reported != dimension
            or (0,) * variable_count not in basis
            or any(sum(monomial) > dimension - 1 for monomial in basis)
        ):
            raise KrennBlockerQuotientError(
                "invalid finite quotient basis or dimension"
            )
        matrices = []
        if len(self.multiplication_matrices) != variable_count:
            raise KrennBlockerQuotientError(
                "one multiplication matrix is required per variable"
            )
        for matrix in self.multiplication_matrices:
            if len(matrix) != dimension or any(
                len(row) != dimension for row in matrix
            ):
                raise KrennBlockerQuotientError(
                    "a multiplication matrix has the wrong shape"
                )
            matrices.append(
                tuple(
                    tuple(
                        _field_scalar(
                            value,
                            characteristic,
                            "multiplication-matrix entry",
                        )
                        for value in row
                    )
                    for row in matrix
                )
            )
        object.__setattr__(self, "characteristic", characteristic)
        object.__setattr__(self, "variable_count", variable_count)
        object.__setattr__(self, "basis_monomials", basis)
        object.__setattr__(
            self, "multiplication_matrices", tuple(matrices)
        )
        object.__setattr__(
            self, "backend_reported_quotient_dimension", reported
        )

    @property
    def dimension(self) -> int:
        return len(self.basis_monomials)

    def to_dict(self) -> dict[str, object]:
        def scalar_payload(value: FieldScalar) -> object:
            if self.characteristic:
                return int(value)
            rational = Fraction(value)
            return [rational.numerator, rational.denominator]

        return {
            "characteristic": self.characteristic,
            "variable_count": self.variable_count,
            "chart_sha256": self.chart_sha256,
            "basis_monomials": [
                list(monomial) for monomial in self.basis_monomials
            ],
            "multiplication_matrices": [
                [
                    [scalar_payload(value) for value in row]
                    for row in matrix
                ]
                for matrix in self.multiplication_matrices
            ],
            "backend_reported_quotient_dimension": (
                self.backend_reported_quotient_dimension
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "QuotientRepresentation":
        if not isinstance(payload, Mapping) or set(payload) != {
            "characteristic",
            "variable_count",
            "chart_sha256",
            "basis_monomials",
            "multiplication_matrices",
            "backend_reported_quotient_dimension",
        }:
            raise KrennBlockerQuotientError(
                "quotient representation has the wrong schema"
            )
        characteristic = _characteristic(payload["characteristic"])

        def parse_scalar(value: object) -> FieldScalar:
            if characteristic:
                return _exact_integer(value, "modular matrix entry")
            if (
                not isinstance(value, list)
                or len(value) != 2
            ):
                raise KrennBlockerQuotientError(
                    "rational matrix entry must be [numerator,denominator]"
                )
            numerator = _exact_integer(
                value[0], "rational numerator"
            )
            denominator = _exact_integer(
                value[1], "rational denominator"
            )
            if denominator == 0:
                raise KrennBlockerQuotientError(
                    "rational denominator must be nonzero"
                )
            return Fraction(numerator, denominator)

        try:
            basis = tuple(
                tuple(monomial)
                for monomial in payload["basis_monomials"]
            )
            matrices = tuple(
                tuple(
                    tuple(parse_scalar(value) for value in row)
                    for row in matrix
                )
                for matrix in payload["multiplication_matrices"]
            )
        except TypeError as error:
            raise KrennBlockerQuotientError(
                "quotient representation arrays are malformed"
            ) from error
        return cls(
            characteristic=characteristic,
            variable_count=payload["variable_count"],
            chart_sha256=payload["chart_sha256"],
            basis_monomials=basis,
            multiplication_matrices=matrices,
            backend_reported_quotient_dimension=payload[
                "backend_reported_quotient_dimension"
            ],
        )


def _affine_to_general(
    equation: AffinePolynomial,
    variable_count: int,
) -> GeneralPolynomial:
    terms = []
    for monomial, coefficient in equation:
        exponent = [0] * variable_count
        for variable in monomial:
            exponent[variable] += 1
        terms.append((tuple(exponent), Fraction(coefficient)))
    return tuple(terms)


def multiplication_matrix_for_polynomial(
    representation: QuotientRepresentation,
    polynomial: GeneralPolynomial,
) -> Matrix:
    """Evaluate a general exact polynomial at commuting matrices."""

    dimension = representation.dimension
    characteristic = representation.characteristic
    try:
        raw_terms = tuple(polynomial)
    except TypeError as error:
        raise KrennBlockerQuotientError(
            "a general polynomial must be a finite term sequence"
        ) from error
    normalized_terms = []
    seen_exponents = set()
    for raw_term in raw_terms:
        try:
            raw_exponent, raw_coefficient = raw_term
        except (TypeError, ValueError) as error:
            raise KrennBlockerQuotientError(
                "a general polynomial term must be exponent-coefficient"
            ) from error
        exponent = _normalize_exponent(
            raw_exponent, representation.variable_count
        )
        coefficient = _exact_fraction(
            raw_coefficient, "general polynomial coefficient"
        )
        if not coefficient or exponent in seen_exponents:
            raise KrennBlockerQuotientError(
                "general polynomial terms must be nonzero and unique"
            )
        seen_exponents.add(exponent)
        normalized_terms.append((exponent, coefficient))
    result = tuple(
        tuple(row)
        for row in _zero_matrix(dimension, characteristic)
    )
    for exponent, coefficient in normalized_terms:
        coefficient = _fraction_in_field(
            coefficient, characteristic
        )
        term = _identity_matrix(dimension, characteristic)
        for variable, power in enumerate(exponent):
            if power:
                term = _matrix_multiply(
                    term,
                    _matrix_power(
                        representation.multiplication_matrices[
                            variable
                        ],
                        power,
                        characteristic,
                    ),
                    characteristic,
                )
        result = _matrix_add(
            result,
            _matrix_scale(coefficient, term, characteristic),
            characteristic,
        )
    return result


def multiplication_matrix_is_nilpotent(
    representation: QuotientRepresentation,
    matrix: Matrix,
) -> bool:
    """Test nilpotence using the exact dimension bound."""

    dimension = representation.dimension
    if len(matrix) != dimension or any(
        len(row) != dimension for row in matrix
    ):
        raise KrennBlockerQuotientError(
            "nilpotence matrix has the wrong shape"
        )
    normalized = tuple(
        tuple(
            _field_scalar(
                value,
                representation.characteristic,
                "nilpotence-matrix entry",
            )
            for value in row
        )
        for row in matrix
    )
    return _matrix_is_zero(
        _matrix_power(
            normalized,
            dimension,
            representation.characteristic,
        )
    )


def verify_quotient_representation(
    chart: TutteBarrierAffineChart,
    representation: QuotientRepresentation,
) -> dict[str, object]:
    """Independently replay one nonzero cyclic quotient representation."""

    if chart.variable_count != representation.variable_count:
        raise KrennBlockerQuotientError(
            "chart and quotient variable counts differ"
        )
    if chart_sha256(chart) != representation.chart_sha256:
        raise KrennBlockerQuotientError(
            "chart and quotient representation hashes differ"
        )
    characteristic = representation.characteristic
    matrices = representation.multiplication_matrices
    commutator_checks = 0
    for first in range(len(matrices)):
        for second in range(first):
            commutator_checks += 1
            if _matrix_multiply(
                matrices[first], matrices[second], characteristic
            ) != _matrix_multiply(
                matrices[second], matrices[first], characteristic
            ):
                raise KrennBlockerQuotientError(
                    "multiplication matrices do not commute"
                )
    equation_checks = 0
    for equation in chart.equations:
        equation_checks += 1
        evaluated = multiplication_matrix_for_polynomial(
            representation,
            _affine_to_general(
                equation, representation.variable_count
            ),
        )
        if not _matrix_is_zero(evaluated):
            raise KrennBlockerQuotientError(
                "a blocker equation does not vanish on the matrices"
            )
    dimension = representation.dimension
    constant_index = representation.basis_monomials.index(
        (0,) * representation.variable_count
    )
    cyclic_checks = 0
    for basis_index, monomial in enumerate(
        representation.basis_monomials
    ):
        vector = tuple(
            _one(characteristic)
            if index == constant_index
            else _zero(characteristic)
            for index in range(dimension)
        )
        for variable, power in enumerate(monomial):
            if power:
                vector = _matrix_vector(
                    _matrix_power(
                        matrices[variable], power, characteristic
                    ),
                    vector,
                    characteristic,
                )
        expected = tuple(
            _one(characteristic)
            if index == basis_index
            else _zero(characteristic)
            for index in range(dimension)
        )
        cyclic_checks += 1
        if vector != expected:
            raise KrennBlockerQuotientError(
                "basis monomials are inconsistent with matrix action"
            )
    identity = _identity_matrix(dimension, characteristic)
    if _matrix_is_zero(identity):
        raise KrennBlockerQuotientError(
            "the quotient representation is not unital and nonzero"
        )
    return {
        "schema": SCHEMA,
        "chart_sha256": chart_sha256(chart),
        "field": (
            "Q" if characteristic == 0 else f"F_{characteristic}"
        ),
        "representation_dimension": dimension,
        "commutator_checks": commutator_checks,
        "equation_matrix_checks": equation_checks,
        "cyclic_basis_checks": cyclic_checks,
        "exact_replay": True,
        "claim_boundary": {
            "nonzero_unital_representation_proves_"
            "specialized_affine_ideal_proper_over_field": True,
            "proves_complex_escape_for_this_specialization": (
                characteristic == 0
            ),
            "backend_full_quotient_claim_independently_verified": False,
            "eqsystem_constraints_imposed": False,
            "symbolic_90_parameter_quotient_built": False,
            "generic_degree_implies_all_specializations": False,
            "n8_boundary_escape_proved": False,
            "n8_nonexistence_proved": False,
        },
    }


_BASIS_PATTERN = re.compile(
    r"^standard_basis\[(\d+)\]=(.*)$"
)
_NORMAL_FORM_PATTERN = re.compile(
    r"^normal_form\[(\d+),(\d+)\]=(.*)$"
)
_VARIABLE_PATTERN = re.compile(r"^x(\d+)(?:\^(\d+))?$")


def _parse_backend_polynomial(
    expression: str,
    *,
    variable_count: int,
    characteristic: int,
) -> dict[Exponent, FieldScalar]:
    expression = expression.replace(" ", "")
    if len(expression.encode("utf-8")) > MAXIMUM_TRANSCRIPT_BYTES:
        raise KrennBlockerQuotientError(
            "Singular polynomial exceeds the parser byte limit"
        )
    if expression == "0":
        return {}
    if not expression or any(
        character in expression for character in "()"
    ):
        raise KrennBlockerQuotientError(
            "unsupported Singular polynomial syntax"
        )
    chunks = re.findall(r"[+-]?[^+-]+", expression)
    if (
        not chunks
        or len(chunks) > MAXIMUM_BACKEND_POLYNOMIAL_TERMS
        or "".join(chunks) != expression
    ):
        raise KrennBlockerQuotientError(
            "could not tokenize a Singular polynomial"
        )
    result: dict[Exponent, FieldScalar] = {}
    for chunk in chunks:
        sign = 1
        if chunk[0] == "+":
            chunk = chunk[1:]
        elif chunk[0] == "-":
            sign = -1
            chunk = chunk[1:]
        if not chunk:
            raise KrennBlockerQuotientError(
                "empty Singular polynomial term"
            )
        coefficient = Fraction(sign)
        exponent = [0] * variable_count
        factors = chunk.split("*")
        if len(factors) > MAXIMUM_BACKEND_FACTORS_PER_TERM:
            raise KrennBlockerQuotientError(
                "Singular polynomial term has too many factors"
            )
        for factor in factors:
            if len(factor) > MAXIMUM_COEFFICIENT_CHARACTERS:
                raise KrennBlockerQuotientError(
                    "Singular polynomial factor is too long"
                )
            variable_match = _VARIABLE_PATTERN.fullmatch(factor)
            if variable_match:
                variable = int(variable_match.group(1))
                power_text = variable_match.group(2) or "1"
                if len(power_text) > len(str(MAXIMUM_EXPONENT)):
                    raise KrennBlockerQuotientError(
                        "Singular monomial exponent is too large"
                    )
                power = int(power_text)
                if variable >= variable_count:
                    raise KrennBlockerQuotientError(
                        "Singular monomial is outside the chart"
                    )
                if (
                    power < 1
                    or power > MAXIMUM_EXPONENT
                    or exponent[variable] + power > MAXIMUM_EXPONENT
                ):
                    raise KrennBlockerQuotientError(
                        "Singular monomial exponent is too large"
                    )
                exponent[variable] += power
                continue
            try:
                coefficient *= Fraction(factor)
            except (ValueError, ZeroDivisionError) as error:
                raise KrennBlockerQuotientError(
                    "invalid Singular coefficient"
                ) from error
        key = tuple(exponent)
        value = _fraction_in_field(coefficient, characteristic)
        result[key] = _add(
            result.get(key, _zero(characteristic)),
            value,
            characteristic,
        )
        if not result[key]:
            del result[key]
    return result


def _unique_summary_value(
    lines: Sequence[str],
    key: str,
) -> str:
    prefix = f"{key}="
    matches = [
        line[len(prefix):]
        for line in lines
        if line.startswith(prefix)
    ]
    if len(matches) != 1:
        raise KrennBlockerQuotientError(
            f"Singular output needs exactly one {key} field"
        )
    return matches[0]


@dataclass(frozen=True)
class SingularQuotientOutcome:
    """Strictly parsed status from one completed Singular transcript."""

    characteristic: int
    algorithm: str
    chart_sha256: str
    timer_ticks: int
    groebner_basis_size: int
    unit_ideal: bool
    krull_dimension: int
    backend_quotient_dimension: int | None
    representation: QuotientRepresentation | None

    @property
    def backend_reports_proper_ideal(self) -> bool:
        return not self.unit_ideal


def parse_singular_blocker_output(
    chart: TutteBarrierAffineChart,
    output: str,
    *,
    characteristic: int,
    algorithm: str,
) -> SingularQuotientOutcome:
    """Parse and exactly replay a completed quotient transcript."""

    characteristic = _characteristic(characteristic)
    if algorithm not in {"std", "slimgb"}:
        raise KrennBlockerQuotientError(
            "Singular algorithm must be 'std' or 'slimgb'"
        )
    if not isinstance(output, str):
        raise KrennBlockerQuotientError(
            "Singular output must be text"
        )
    if len(output.encode("utf-8")) > MAXIMUM_TRANSCRIPT_BYTES:
        raise KrennBlockerQuotientError(
            "Singular output exceeds the transcript byte limit"
        )
    lines = tuple(
        line.strip() for line in output.splitlines() if line.strip()
    )
    if (
        lines.count(PARSE_MARKER) != 1
        or lines.count(COMPLETION_MARKER) != 1
    ):
        raise KrennBlockerQuotientError(
            "Singular output markers are missing or duplicated"
        )
    parse_index = lines.index(PARSE_MARKER)
    completion_index = lines.index(COMPLETION_MARKER)
    expected_metadata = (
        f"transcript_schema={TRANSCRIPT_SCHEMA}",
        f"chart_sha256={chart_sha256(chart)}",
        f"characteristic={characteristic}",
        f"algorithm={algorithm}",
        f"variable_count={chart.variable_count}",
    )
    if (
        completion_index <= parse_index
        or lines[parse_index + 1 : parse_index + 6]
        != expected_metadata
        or _unique_summary_value(lines, "transcript_schema")
        != TRANSCRIPT_SCHEMA
        or _unique_summary_value(lines, "chart_sha256")
        != chart_sha256(chart)
        or _unique_summary_value(lines, "characteristic")
        != str(characteristic)
        or _unique_summary_value(lines, "algorithm") != algorithm
        or _unique_summary_value(lines, "variable_count")
        != str(chart.variable_count)
    ):
        raise KrennBlockerQuotientError(
            "Singular transcript metadata is absent, reordered, or changed"
        )
    try:
        timer_ticks = int(_unique_summary_value(
            lines, "timer_ticks"
        ))
        basis_size = int(_unique_summary_value(
            lines, "basis_size"
        ))
        unit_integer = int(_unique_summary_value(
            lines, "unit_ideal"
        ))
        krull_dimension = int(_unique_summary_value(
            lines, "krull_dimension"
        ))
    except ValueError as error:
        raise KrennBlockerQuotientError(
            "Singular summary fields must be integers"
        ) from error
    if (
        timer_ticks < 0
        or basis_size < 1
        or unit_integer not in (0, 1)
    ):
        raise KrennBlockerQuotientError(
            "Singular summary fields are out of range"
        )
    unit_ideal = bool(unit_integer)
    basis_rows: dict[int, str] = {}
    normal_rows: dict[tuple[int, int], str] = {}
    for line in lines:
        basis_match = _BASIS_PATTERN.fullmatch(line)
        if basis_match:
            index = int(basis_match.group(1))
            if index in basis_rows:
                raise KrennBlockerQuotientError(
                    "duplicate standard-basis output row"
                )
            basis_rows[index] = basis_match.group(2)
            continue
        normal_match = _NORMAL_FORM_PATTERN.fullmatch(line)
        if normal_match:
            key = (
                int(normal_match.group(1)),
                int(normal_match.group(2)),
            )
            if key in normal_rows:
                raise KrennBlockerQuotientError(
                    "duplicate normal-form output row"
                )
            normal_rows[key] = normal_match.group(3)
    quotient_fields = [
        line for line in lines
        if line.startswith("quotient_dimension=")
    ]
    standard_size_fields = [
        line for line in lines
        if line.startswith("standard_basis_size=")
    ]
    if unit_ideal or krull_dimension != 0:
        if (
            quotient_fields
            or standard_size_fields
            or basis_rows
            or normal_rows
        ):
            raise KrennBlockerQuotientError(
                "non-finite transcript contains quotient matrix data"
            )
        return SingularQuotientOutcome(
            characteristic=characteristic,
            algorithm=algorithm,
            chart_sha256=chart_sha256(chart),
            timer_ticks=timer_ticks,
            groebner_basis_size=basis_size,
            unit_ideal=unit_ideal,
            krull_dimension=krull_dimension,
            backend_quotient_dimension=None,
            representation=None,
        )
    try:
        quotient_dimension = int(_unique_summary_value(
            lines, "quotient_dimension"
        ))
        standard_size = int(_unique_summary_value(
            lines, "standard_basis_size"
        ))
    except ValueError as error:
        raise KrennBlockerQuotientError(
            "finite quotient dimensions must be integers"
        ) from error
    if (
        not 1 <= quotient_dimension <= MAXIMUM_QUOTIENT_DIMENSION
        or standard_size != quotient_dimension
        or set(basis_rows) != set(range(quotient_dimension))
        or set(normal_rows) != {
            (variable, column)
            for variable in range(chart.variable_count)
            for column in range(quotient_dimension)
        }
    ):
        raise KrennBlockerQuotientError(
            "finite quotient output has an incomplete census"
        )
    basis_monomials = []
    for index in range(quotient_dimension):
        polynomial = _parse_backend_polynomial(
            basis_rows[index],
            variable_count=chart.variable_count,
            characteristic=characteristic,
        )
        if len(polynomial) != 1:
            raise KrennBlockerQuotientError(
                "standard quotient basis must consist of monomials"
            )
        (monomial, coefficient), = polynomial.items()
        if coefficient != _one(characteristic):
            raise KrennBlockerQuotientError(
                "standard quotient monomials must be monic"
            )
        basis_monomials.append(monomial)
    if len(set(basis_monomials)) != quotient_dimension:
        raise KrennBlockerQuotientError(
            "standard quotient monomials must be unique"
        )
    if any(
        sum(monomial) > quotient_dimension - 1
        for monomial in basis_monomials
    ):
        raise KrennBlockerQuotientError(
            "standard quotient monomial exceeds the order-ideal degree bound"
        )
    position = {
        monomial: index
        for index, monomial in enumerate(basis_monomials)
    }
    matrices = []
    for variable in range(chart.variable_count):
        matrix = _zero_matrix(quotient_dimension, characteristic)
        for column in range(quotient_dimension):
            polynomial = _parse_backend_polynomial(
                normal_rows[(variable, column)],
                variable_count=chart.variable_count,
                characteristic=characteristic,
            )
            for monomial, coefficient in polynomial.items():
                if monomial not in position:
                    raise KrennBlockerQuotientError(
                        "a normal form lies outside the reported basis"
                    )
                matrix[position[monomial]][column] = coefficient
        matrices.append(tuple(tuple(row) for row in matrix))
    representation = QuotientRepresentation(
        characteristic=characteristic,
        variable_count=chart.variable_count,
        chart_sha256=chart_sha256(chart),
        basis_monomials=tuple(basis_monomials),
        multiplication_matrices=tuple(matrices),
        backend_reported_quotient_dimension=quotient_dimension,
    )
    verify_quotient_representation(chart, representation)
    return SingularQuotientOutcome(
        characteristic=characteristic,
        algorithm=algorithm,
        chart_sha256=chart_sha256(chart),
        timer_ticks=timer_ticks,
        groebner_basis_size=basis_size,
        unit_ideal=False,
        krull_dimension=0,
        backend_quotient_dimension=quotient_dimension,
        representation=representation,
    )


__all__ = [
    "COMPLETION_MARKER",
    "DEFAULT_COEFFICIENT_HEIGHT",
    "DEFAULT_MODULUS",
    "DEFAULT_SPECIALIZATION_SEED",
    "KrennBlockerQuotientError",
    "MAXIMUM_QUOTIENT_DIMENSION",
    "MAXIMUM_TRANSCRIPT_BYTES",
    "PARSE_MARKER",
    "QuotientRepresentation",
    "SCHEMA",
    "SingularQuotientOutcome",
    "TRANSCRIPT_SCHEMA",
    "chart_payload",
    "chart_sha256",
    "deterministic_k5_blocker_chart",
    "multiplication_matrix_for_polynomial",
    "multiplication_matrix_is_nilpotent",
    "parse_singular_blocker_output",
    "singular_blocker_quotient_script",
    "verify_quotient_representation",
]
