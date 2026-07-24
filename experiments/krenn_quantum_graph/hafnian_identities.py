"""Exact weighted-hafnian contractions of the perfect-matching tensor map.

For local covectors ``g_i`` the defining equations recombine as

    sum_c Phi(W)_c prod_i g_i[c_i] = haf(B),

where ``B_ij = g_i^T W_ij g_j``.  Taking every ``g_i`` equal retains only
the color-symmetric part of each edge matrix and gives a homogeneous
degree-``n`` polynomial in one color vector ``g``.

This module implements both identities over ``Q``, the coefficient system of
the equal-``g`` shadow, deterministic modular Jacobian certificates, and a
Singular-script exporter.  A modular rank certificate is used only to prove a
specific integer minor nonzero in characteristic zero; it is not presented as
a finite-field proof about complex image membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import comb
from typing import Mapping, Sequence, TypeAlias

from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    perfect_matchings,
    validate_parameters,
    variable_index,
)
from experiments.krenn_quantum_graph.targets import ColoringTarget
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.witness import (
    SparseWitness,
    fraction_mod,
)


ExactScalar: TypeAlias = int | Fraction
Occupation: TypeAlias = tuple[int, ...]
ShadowVariable: TypeAlias = tuple[int, int, int, int]
ShadowMonomial: TypeAlias = tuple[int, ...]
ExactVector: TypeAlias = tuple[Fraction, ...]
ExactMatrix: TypeAlias = tuple[ExactVector, ...]

HAFNIAN_IDENTITY_SCHEMA = "krenn-weighted-hafnian-identity-v1"
EQUAL_G_SHADOW_SCHEMA = "krenn-equal-g-hafnian-shadow-v1"


class KrennHafnianIdentityError(ValueError):
    """A contraction, shadow system, or modular certificate is malformed."""


def _require_prime_modulus(modulus: int) -> int:
    if (
        isinstance(modulus, bool)
        or not isinstance(modulus, int)
        or modulus < 2
    ):
        raise KrennHafnianIdentityError(
            "modulus must be a prime integer"
        )
    divisor = 2
    while divisor * divisor <= modulus:
        if modulus % divisor == 0:
            raise KrennHafnianIdentityError(
                "modulus must be prime"
            )
        divisor += 1 if divisor == 2 else 2
    return modulus


def _fraction_vector(
    values: Sequence[ExactScalar],
    expected: int,
    label: str,
) -> ExactVector:
    try:
        result = tuple(Fraction(value) for value in values)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise KrennHafnianIdentityError(
            f"{label} must contain exact rational values"
        ) from error
    if len(result) != expected:
        raise KrennHafnianIdentityError(
            f"{label} must have length {expected}"
        )
    return result


def _local_covectors(
    n: int,
    d: int,
    covectors: Sequence[Sequence[ExactScalar]],
) -> tuple[ExactVector, ...]:
    if len(covectors) != n:
        raise KrennHafnianIdentityError(
            "one local covector is required at every vertex"
        )
    return tuple(
        _fraction_vector(covector, d, f"covector {vertex}")
        for vertex, covector in enumerate(covectors)
    )


def hafnian(matrix: Sequence[Sequence[ExactScalar]]) -> Fraction:
    """Return the exact hafnian of an even symmetric zero-diagonal matrix."""

    try:
        rows = tuple(tuple(Fraction(value) for value in row) for row in matrix)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise KrennHafnianIdentityError(
            "hafnian entries must be exact rational values"
        ) from error
    n = len(rows)
    if n < 2 or n % 2 or any(len(row) != n for row in rows):
        raise KrennHafnianIdentityError(
            "hafnian input must be a nonempty even square matrix"
        )
    if any(rows[i][i] for i in range(n)):
        raise KrennHafnianIdentityError(
            "hafnian input must have zero diagonal"
        )
    if any(rows[i][j] != rows[j][i] for i in range(n) for j in range(n)):
        raise KrennHafnianIdentityError(
            "hafnian input must be symmetric"
        )
    return sum(
        (
            _matching_product(rows, matching)
            for matching in perfect_matchings(n)
        ),
        Fraction(0),
    )


def _matching_product(
    matrix: ExactMatrix,
    matching: Sequence[tuple[int, int]],
) -> Fraction:
    result = Fraction(1)
    for i, j in matching:
        result *= matrix[i][j]
    return result


def contracted_edge_matrix(
    witness: SparseWitness,
    covectors: Sequence[Sequence[ExactScalar]],
) -> ExactMatrix:
    """Return ``B_ij = g_i^T W_ij g_j`` with ``B_ji=B_ij``."""

    n, d = validate_parameters(witness.n, witness.d)
    local = _local_covectors(n, d, covectors)
    values = witness.as_dict()
    matrix = [[Fraction(0) for _j in range(n)] for _i in range(n)]
    for i, j in canonical_edges(n):
        total = Fraction(0)
        for a in range(d):
            for b in range(d):
                total += (
                    local[i][a]
                    * values.get(variable_index(n, d, i, j, a, b), 0)
                    * local[j][b]
                )
        matrix[i][j] = total
        matrix[j][i] = total
    return tuple(tuple(row) for row in matrix)


def tensor_contraction(
    target: ColoringTarget,
    covectors: Sequence[Sequence[ExactScalar]],
) -> Fraction:
    """Contract an exact target with ``g_0 tensor ... tensor g_(n-1)``."""

    local = _local_covectors(target.n, target.d, covectors)
    total = Fraction(0)
    for coloring, coefficient in target.entries:
        term = Fraction(coefficient)
        for vertex, color in enumerate(coloring):
            term *= local[vertex][color]
        total += term
    return total


@dataclass(frozen=True)
class HafnianContractionCertificate:
    """Exact replay of one weighted contraction, optionally against a target."""

    n: int
    d: int
    hafnian_value: Fraction
    tensor_map_value: Fraction
    target_value: Fraction | None

    @property
    def map_identity_exact(self) -> bool:
        return self.hafnian_value == self.tensor_map_value

    @property
    def target_residual(self) -> Fraction | None:
        if self.target_value is None:
            return None
        return self.tensor_map_value - self.target_value

    @property
    def target_contraction_satisfied(self) -> bool | None:
        residual = self.target_residual
        return None if residual is None else residual == 0


def certify_hafnian_contraction(
    witness: SparseWitness,
    covectors: Sequence[Sequence[ExactScalar]],
    *,
    target: ColoringTarget | None = None,
) -> HafnianContractionCertificate:
    """Certify the weighted hafnian identity coefficientwise over ``Q``."""

    if target is not None and (target.n, target.d) != (
        witness.n,
        witness.d,
    ):
        raise KrennHafnianIdentityError(
            "target dimensions do not match the witness"
        )
    output = MatchingTensorMap(witness.n, witness.d).evaluate_exact(witness)
    certificate = HafnianContractionCertificate(
        n=witness.n,
        d=witness.d,
        hafnian_value=hafnian(
            contracted_edge_matrix(witness, covectors)
        ),
        tensor_map_value=tensor_contraction(output, covectors),
        target_value=(
            tensor_contraction(target, covectors)
            if target is not None
            else None
        ),
    )
    if not certificate.map_identity_exact:
        raise KrennHafnianIdentityError(
            "weighted hafnian identity failed exact replay"
        )
    return certificate


def occupations(n: int, d: int) -> tuple[Occupation, ...]:
    """Return all color occupations in deterministic lexicographic order."""

    n, d = validate_parameters(n, d)
    return tuple(
        occupation
        for occupation in product(range(n + 1), repeat=d)
        if sum(occupation) == n
    )


@dataclass(frozen=True)
class EqualGShadowSystem:
    """Target-independent coefficient system of ``haf(g^T W_ij g)``."""

    n: int
    d: int
    variable_keys: tuple[ShadowVariable, ...]
    equation_occupations: tuple[Occupation, ...]
    equation_terms: tuple[tuple[ShadowMonomial, ...], ...]
    schema: str = EQUAL_G_SHADOW_SCHEMA

    def __post_init__(self) -> None:
        n, d = validate_parameters(self.n, self.d)
        if self.schema != EQUAL_G_SHADOW_SCHEMA:
            raise KrennHafnianIdentityError(
                "equal-g shadow schema changed"
            )
        expected_variables = comb(n, 2) * comb(d + 1, 2)
        expected_equations = comb(n + d - 1, d - 1)
        if len(self.variable_keys) != expected_variables:
            raise KrennHafnianIdentityError(
                "equal-g shadow variable census changed"
            )
        if (
            self.equation_occupations != occupations(n, d)
            or len(self.equation_terms) != expected_equations
        ):
            raise KrennHafnianIdentityError(
                "equal-g shadow equation census changed"
            )
        degree = n // 2
        if any(
            len(monomial) != degree
            or any(
                variable < 0 or variable >= expected_variables
                for variable in monomial
            )
            for equation in self.equation_terms
            for monomial in equation
        ):
            raise KrennHafnianIdentityError(
                "equal-g shadow has an invalid monomial"
            )

    @property
    def variable_count(self) -> int:
        return len(self.variable_keys)

    @property
    def antisymmetric_variable_count(self) -> int:
        return comb(self.n, 2) * comb(self.d, 2)

    @property
    def full_edge_variable_count(self) -> int:
        return comb(self.n, 2) * self.d * self.d

    @property
    def equation_count(self) -> int:
        return len(self.equation_occupations)

    @property
    def degree(self) -> int:
        return self.n // 2

    @property
    def monomial_count(self) -> int:
        return sum(map(len, self.equation_terms))

    def evaluate(
        self,
        values: Sequence[ExactScalar],
    ) -> tuple[Fraction, ...]:
        vector = _fraction_vector(
            values, self.variable_count, "equal-g shadow vector"
        )
        result = []
        for equation in self.equation_terms:
            total = Fraction(0)
            for monomial in equation:
                term = Fraction(1)
                for variable in monomial:
                    term *= vector[variable]
                total += term
            result.append(total)
        return tuple(result)

    def values_from_witness(
        self,
        witness: SparseWitness,
    ) -> tuple[Fraction, ...]:
        if (witness.n, witness.d) != (self.n, self.d):
            raise KrennHafnianIdentityError(
                "witness dimensions do not match the equal-g shadow"
            )
        values = witness.as_dict()
        result = []
        for i, j, a, b in self.variable_keys:
            coefficient = values.get(
                variable_index(self.n, self.d, i, j, a, b), 0
            )
            if a != b:
                coefficient += values.get(
                    variable_index(self.n, self.d, i, j, b, a), 0
                )
            result.append(Fraction(coefficient))
        return tuple(result)

    def target_coefficients(
        self,
        target: ColoringTarget,
    ) -> tuple[Fraction, ...]:
        if (target.n, target.d) != (self.n, self.d):
            raise KrennHafnianIdentityError(
                "target dimensions do not match the equal-g shadow"
            )
        index = {
            occupation: equation
            for equation, occupation in enumerate(
                self.equation_occupations
            )
        }
        result = [Fraction(0)] * self.equation_count
        for coloring, coefficient in target.entries:
            occupation = tuple(
                coloring.count(color) for color in range(self.d)
            )
            result[index[occupation]] += Fraction(coefficient)
        return tuple(result)

    def output_coefficients(
        self,
        witness: SparseWitness,
    ) -> tuple[Fraction, ...]:
        output = MatchingTensorMap(self.n, self.d).evaluate_exact(witness)
        return self.target_coefficients(output)

    def jacobian_mod(
        self,
        values: Sequence[int],
        modulus: int = 31,
    ) -> tuple[tuple[int, ...], ...]:
        if len(values) != self.variable_count:
            raise KrennHafnianIdentityError(
                "modular shadow vector has the wrong length"
            )
        vector = tuple(int(value) % modulus for value in values)
        rows = []
        for equation in self.equation_terms:
            row = [0] * self.variable_count
            for monomial in equation:
                for position, variable in enumerate(monomial):
                    term = 1
                    for other_position, other in enumerate(monomial):
                        if other_position != position:
                            term = term * vector[other] % modulus
                    row[variable] = (
                        row[variable] + term
                    ) % modulus
            rows.append(tuple(row))
        return tuple(rows)


@lru_cache(maxsize=None)
def generate_equal_g_shadow_system(
    n: int,
    d: int,
) -> EqualGShadowSystem:
    """Build the compact coefficient equations of the equal-``g`` hafnian."""

    n, d = validate_parameters(n, d)
    color_pairs = tuple(combinations_with_replacement(range(d), 2))
    variable_keys = tuple(
        (i, j, a, b)
        for i, j in canonical_edges(n)
        for a, b in color_pairs
    )
    variable_index_by_key = {
        key: index for index, key in enumerate(variable_keys)
    }
    equation_occupations = occupations(n, d)
    occupation_index = {
        occupation: index
        for index, occupation in enumerate(equation_occupations)
    }
    equation_terms: list[list[ShadowMonomial]] = [
        [] for _occupation in equation_occupations
    ]
    for matching in perfect_matchings(n):
        for choices in product(color_pairs, repeat=n // 2):
            exponent = [0] * d
            monomial = []
            for (i, j), (a, b) in zip(matching, choices):
                exponent[a] += 1
                exponent[b] += 1
                monomial.append(
                    variable_index_by_key[(i, j, a, b)]
                )
            equation_terms[occupation_index[tuple(exponent)]].append(
                tuple(monomial)
            )
    return EqualGShadowSystem(
        n=n,
        d=d,
        variable_keys=variable_keys,
        equation_occupations=equation_occupations,
        equation_terms=tuple(
            tuple(equation) for equation in equation_terms
        ),
    )


@dataclass(frozen=True)
class EqualGIdentityCertificate:
    """Coefficientwise exact comparison of the shadow and full tensor map."""

    shadow_coefficients: tuple[Fraction, ...]
    output_coefficients: tuple[Fraction, ...]
    target_coefficients: tuple[Fraction, ...] | None

    @property
    def map_identity_exact(self) -> bool:
        return self.shadow_coefficients == self.output_coefficients

    @property
    def residual_coefficients(
        self,
    ) -> tuple[Fraction, ...] | None:
        if self.target_coefficients is None:
            return None
        return tuple(
            output - target
            for output, target in zip(
                self.output_coefficients,
                self.target_coefficients,
            )
        )


def certify_equal_g_identity(
    witness: SparseWitness,
    *,
    target: ColoringTarget | None = None,
) -> EqualGIdentityCertificate:
    """Replay every occupation coefficient of the equal-``g`` identity."""

    system = generate_equal_g_shadow_system(witness.n, witness.d)
    if target is not None and (target.n, target.d) != (
        witness.n,
        witness.d,
    ):
        raise KrennHafnianIdentityError(
            "target dimensions do not match the witness"
        )
    certificate = EqualGIdentityCertificate(
        shadow_coefficients=system.evaluate(
            system.values_from_witness(witness)
        ),
        output_coefficients=system.output_coefficients(witness),
        target_coefficients=(
            system.target_coefficients(target)
            if target is not None
            else None
        ),
    )
    if not certificate.map_identity_exact:
        raise KrennHafnianIdentityError(
            "equal-g coefficient identity failed exact replay"
        )
    return certificate


def deterministic_modular_values(
    length: int,
    *,
    modulus: int = 31,
    seed: int = 603,
) -> tuple[int, ...]:
    """Return a stable LCG probe vector without using ambient RNG state."""

    if length < 1 or modulus < 2:
        raise KrennHafnianIdentityError(
            "modular probe needs positive length and modulus"
        )
    state = int(seed) & 0x7FFFFFFF
    values = []
    for _index in range(length):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        values.append(state % modulus)
    return tuple(values)


def _rank_and_pivots_mod(
    matrix: Sequence[Sequence[int]],
    modulus: int,
) -> tuple[int, tuple[int, ...]]:
    if not matrix:
        return 0, ()
    rows = [list(map(lambda value: int(value) % modulus, row)) for row in matrix]
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennHafnianIdentityError(
            "modular matrix must be rectangular"
        )
    rank = 0
    pivots = []
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column] % modulus
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        try:
            inverse = pow(rows[rank][column], -1, modulus)
        except ValueError as error:
            raise KrennHafnianIdentityError(
                "modulus is not a field for the selected pivot"
            ) from error
        rows[rank] = [
            value * inverse % modulus for value in rows[rank]
        ]
        for other in range(rank + 1, len(rows)):
            factor = rows[other][column] % modulus
            if factor:
                rows[other] = [
                    (left - factor * right) % modulus
                    for left, right in zip(rows[other], rows[rank])
                ]
        pivots.append(column)
        rank += 1
        if rank == len(rows):
            break
    return rank, tuple(pivots)


def _determinant_mod(
    matrix: Sequence[Sequence[int]],
    modulus: int,
) -> int:
    rows = [list(int(value) % modulus for value in row) for row in matrix]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise KrennHafnianIdentityError(
            "modular determinant needs a square matrix"
        )
    determinant = 1
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if rows[row][column] % modulus
            ),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        pivot_value = rows[column][column] % modulus
        determinant = determinant * pivot_value % modulus
        try:
            inverse = pow(pivot_value, -1, modulus)
        except ValueError as error:
            raise KrennHafnianIdentityError(
                "modulus is not a field for the selected pivot"
            ) from error
        for row in range(column + 1, size):
            factor = rows[row][column] * inverse % modulus
            if factor:
                for entry in range(column, size):
                    rows[row][entry] = (
                        rows[row][entry]
                        - factor * rows[column][entry]
                    ) % modulus
    return determinant % modulus


@dataclass(frozen=True)
class ShadowJacobianCertificate:
    """A nonzero modular minor witnessing characteristic-zero dominance."""

    n: int
    d: int
    modulus: int
    probe_seed: int
    rank: int
    equation_count: int
    variable_count: int
    pivot_columns: tuple[int, ...]
    pivot_minor_determinant_mod: int

    @property
    def full_row_rank(self) -> bool:
        return (
            self.rank == self.equation_count
            and len(self.pivot_columns) == self.equation_count
            and self.pivot_minor_determinant_mod != 0
        )

    @property
    def characteristic_zero_dominance_certified(self) -> bool:
        return self.full_row_rank

    @property
    def finite_field_membership_proved(self) -> bool:
        return False

    @property
    def complex_target_membership_proved(self) -> bool:
        return False


def certify_shadow_jacobian_rank(
    n: int,
    d: int,
    *,
    modulus: int = 31,
    probe_seed: int = 603,
) -> ShadowJacobianCertificate:
    """Exhibit one full-row-rank integer Jacobian minor when available."""

    modulus = _require_prime_modulus(modulus)
    system = generate_equal_g_shadow_system(n, d)
    values = deterministic_modular_values(
        system.variable_count,
        modulus=modulus,
        seed=probe_seed,
    )
    matrix = system.jacobian_mod(values, modulus)
    rank, pivots = _rank_and_pivots_mod(matrix, modulus)
    determinant = 0
    if rank == system.equation_count:
        minor = tuple(
            tuple(row[column] for column in pivots)
            for row in matrix
        )
        determinant = _determinant_mod(minor, modulus)
    return ShadowJacobianCertificate(
        n=n,
        d=d,
        modulus=modulus,
        probe_seed=probe_seed,
        rank=rank,
        equation_count=system.equation_count,
        variable_count=system.variable_count,
        pivot_columns=pivots,
        pivot_minor_determinant_mod=determinant,
    )


def singular_groebner_script(
    system: EqualGShadowSystem,
    target: ColoringTarget,
    *,
    modulus: int = 31,
) -> str:
    """Export the exact target fiber to Singular over ``F_modulus``.

    The exporter is deterministic and does not run a computer-algebra backend.
    It is suitable for a separately bounded Gröbner job with its own resource
    ledger.
    """

    modulus = _require_prime_modulus(modulus)
    rhs = system.target_coefficients(target)
    names = tuple(f"x{index}" for index in range(system.variable_count))
    equations = []
    for terms, target_value in zip(system.equation_terms, rhs):
        pieces = [
            "*".join(names[variable] for variable in monomial)
            for monomial in terms
        ]
        target_mod = fraction_mod(target_value, modulus)
        if target_mod:
            pieces.append(str((-target_mod) % modulus))
        equations.append("+".join(pieces) if pieces else "0")
    return "\n".join(
        (
            "// Deterministic equal-g Krenn target fiber.",
            f"ring r={modulus},({','.join(names)}),dp;",
            "ideal I=" + ",\n".join(equations) + ";",
            "ideal G=std(I);",
            'print("groebner_basis_size");',
            "print(size(G));",
            'print("unit_ideal");',
            "print(size(G)==1 && G[1]==1);",
            "",
        )
    )


def shadow_structure_summary(n: int, d: int) -> Mapping:
    """Return JSON-ready counts and the deterministic dominance certificate."""

    system = generate_equal_g_shadow_system(n, d)
    certificate = certify_shadow_jacobian_rank(n, d)
    return {
        "schema": EQUAL_G_SHADOW_SCHEMA,
        "parameters": {"n": n, "d": d},
        "counts": {
            "full_edge_variables": system.full_edge_variable_count,
            "symmetric_shadow_variables": system.variable_count,
            "antisymmetric_invisible_variables": (
                system.antisymmetric_variable_count
            ),
            "occupation_equations": system.equation_count,
            "degree": system.degree,
            "expanded_monomials": system.monomial_count,
        },
        "jacobian_probe": {
            "modulus": certificate.modulus,
            "seed": certificate.probe_seed,
            "rank": certificate.rank,
            "pivot_minor_determinant_mod": (
                certificate.pivot_minor_determinant_mod
            ),
            "full_row_rank": certificate.full_row_rank,
            "characteristic_zero_dominance_certified": (
                certificate.characteristic_zero_dominance_certified
            ),
            "complex_target_membership_proved": False,
        },
        "claim_boundary": {
            "equal_g_sees_only_W_ab_plus_W_ba_for_a_neq_b": True,
            "general_distinct_g_i_identity_retains_full_edge_matrices": True,
            "dominance_excludes_nonzero_universal_polynomial_invariants_on_shadow_closure": (
                certificate.full_row_rank
            ),
            "jacobian_dominance_alone_proves_exact_GHZ_fiber_nonempty": (
                False
            ),
            "full_tensor_image_membership_proved": False,
        },
    }
