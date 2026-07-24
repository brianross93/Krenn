"""Exact star-linear slices of the equal-``g`` hafnian shadow.

For ``n=6,d=3``, every perfect matching contains exactly one edge incident
to a fixed hub vertex.  After fixing the other ten edge quadratics, the 28
occupation coefficients are therefore jointly linear in the 30 symmetric
shadow variables on the five-edge hub star.

The deterministic seed-912 sign slice below has an invertible 28 by 28
integer minor with determinant ``2**54``.  Consequently, with the remaining
two star variables fixed to zero, it solves every rational occupation target
by exact linear algebra.  This is a target-generic surjective slice over
``Q``; it is stronger than a finite-field membership experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from numbers import Integral
from typing import Sequence, TypeAlias

from experiments.krenn_quantum_graph.hafnian_identities import (
    EqualGShadowSystem,
    deterministic_modular_values,
    generate_equal_g_shadow_system,
)
from experiments.krenn_quantum_graph.targets import canonical_ghz_target


ExactScalar: TypeAlias = int | Fraction
ExactVector: TypeAlias = tuple[Fraction, ...]
IntegerMatrix: TypeAlias = tuple[tuple[int, ...], ...]

SHADOW_STAR_SLICE_SCHEMA = "krenn-n6-d3-shadow-star-slice-v1"
SHADOW_STAR_SOLUTION_SCHEMA = "krenn-n6-d3-shadow-star-solution-v1"

SLICE_N = 6
SLICE_D = 3
SLICE_HUB_VERTEX = 0
SLICE_SIGN_SEED = 912
SLICE_SIGN_MODULUS = 31
SLICE_SIGN_THRESHOLD = 16
EXPECTED_STAR_VARIABLE_COUNT = 30
EXPECTED_NONSTAR_VARIABLE_COUNT = 60
EXPECTED_EQUATION_COUNT = 28
EXPECTED_PIVOT_DETERMINANT = 2**54
EXPECTED_PIVOT_DETERMINANT_MOD_31 = 16

EXPECTED_CANONICAL_GHZ_PIVOT_VALUES: ExactVector = tuple(
    Fraction(value)
    for value in (
        "3",
        "65/8",
        "13/4",
        "15/8",
        "31/8",
        "5/8",
        "27/32",
        "-1/32",
        "-5/32",
        "9/32",
        "-35/32",
        "-13/32",
        "-1/8",
        "13/8",
        "-3/8",
        "3/8",
        "3/8",
        "5/8",
        "-49/32",
        "91/32",
        "-29/32",
        "-83/32",
        "81/32",
        "19/32",
        "-1/2",
        "15/8",
        "25/8",
        "9/4",
    )
)


class KrennShadowSliceError(ValueError):
    """A star slice, exact target vector, or linear replay is malformed."""


def _exact_vector(
    values: Sequence[ExactScalar],
    expected_length: int,
    label: str,
) -> ExactVector:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise KrennShadowSliceError(
            f"{label} must be an exact rational sequence"
        ) from error
    if len(raw) != expected_length:
        raise KrennShadowSliceError(
            f"{label} must have length {expected_length}"
        )
    if any(
        not isinstance(value, (Integral, Fraction))
        for value in raw
    ):
        raise KrennShadowSliceError(
            f"{label} must contain only exact rational values"
        )
    return tuple(Fraction(value) for value in raw)


def _determinant_bareiss(matrix: IntegerMatrix) -> int:
    """Return an exact integer determinant by fraction-free elimination."""

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise KrennShadowSliceError(
            "the slice determinant requires a nonempty square matrix"
        )
    rows = [list(map(int, row)) for row in matrix]
    sign = 1
    previous_pivot = 1
    for column in range(size - 1):
        pivot_row = next(
            (
                row
                for row in range(column, size)
                if rows[row][column]
            ),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != column:
            rows[column], rows[pivot_row] = (
                rows[pivot_row],
                rows[column],
            )
            sign = -sign
        pivot = rows[column][column]
        for row in range(column + 1, size):
            for entry in range(column + 1, size):
                numerator = (
                    rows[row][entry] * pivot
                    - rows[row][column] * rows[column][entry]
                )
                quotient, remainder = divmod(
                    numerator, previous_pivot
                )
                if remainder:
                    raise KrennShadowSliceError(
                        "Bareiss determinant lost exact divisibility"
                    )
                rows[row][entry] = quotient
            rows[row][column] = 0
        previous_pivot = pivot
    return sign * rows[-1][-1]


def _solve_square_exact(
    matrix: IntegerMatrix,
    rhs: ExactVector,
) -> ExactVector:
    """Solve one nonsingular square integer system exactly over ``Q``."""

    size = len(matrix)
    if (
        size == 0
        or len(rhs) != size
        or any(len(row) != size for row in matrix)
    ):
        raise KrennShadowSliceError(
            "the exact linear solve has incompatible dimensions"
        )
    rows = [
        [Fraction(value) for value in row] + [rhs[index]]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot_row = next(
            (
                row
                for row in range(column, size)
                if rows[row][column]
            ),
            None,
        )
        if pivot_row is None:
            raise KrennShadowSliceError(
                "the certified slice minor became singular"
            )
        rows[column], rows[pivot_row] = (
            rows[pivot_row],
            rows[column],
        )
        pivot = rows[column][column]
        rows[column] = [
            value / pivot for value in rows[column]
        ]
        for row in range(size):
            if row == column or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                left - factor * right
                for left, right in zip(rows[row], rows[column])
            ]
    return tuple(row[-1] for row in rows)


def _seed_912_signs(length: int) -> tuple[int, ...]:
    residues = deterministic_modular_values(
        length,
        modulus=SLICE_SIGN_MODULUS,
        seed=SLICE_SIGN_SEED,
    )
    return tuple(
        -1 if residue < SLICE_SIGN_THRESHOLD else 1
        for residue in residues
    )


@dataclass(frozen=True)
class ShadowStarSliceSolution:
    """One exact target and its replayed seed-912 shadow preimage."""

    target_coefficients: ExactVector
    shadow_values: ExactVector
    output_coefficients: ExactVector
    pivot_values: ExactVector
    free_star_values: ExactVector
    schema: str = SHADOW_STAR_SOLUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHADOW_STAR_SOLUTION_SCHEMA:
            raise KrennShadowSliceError(
                "shadow star solution schema changed"
            )
        if (
            len(self.target_coefficients) != EXPECTED_EQUATION_COUNT
            or len(self.output_coefficients) != EXPECTED_EQUATION_COUNT
            or len(self.shadow_values)
            != EXPECTED_STAR_VARIABLE_COUNT
            + EXPECTED_NONSTAR_VARIABLE_COUNT
            or len(self.pivot_values) != EXPECTED_EQUATION_COUNT
            or len(self.free_star_values)
            != EXPECTED_STAR_VARIABLE_COUNT
            - EXPECTED_EQUATION_COUNT
        ):
            raise KrennShadowSliceError(
                "shadow star solution census changed"
            )
        if self.output_coefficients != self.target_coefficients:
            raise KrennShadowSliceError(
                "shadow star solution failed exact coefficient replay"
            )

    @property
    def exact(self) -> bool:
        return self.output_coefficients == self.target_coefficients


@dataclass(frozen=True)
class N6D3ShadowStarSlice:
    """Immutable exact certificate for the seed-912 hub-star slice."""

    star_variables: tuple[int, ...]
    nonstar_variables: tuple[int, ...]
    fixed_shadow_values: tuple[int, ...]
    coefficient_matrix: IntegerMatrix
    pivot_columns: tuple[int, ...]
    free_columns: tuple[int, ...]
    pivot_determinant: int
    hub_vertex: int = SLICE_HUB_VERTEX
    sign_seed: int = SLICE_SIGN_SEED
    sign_modulus: int = SLICE_SIGN_MODULUS
    schema: str = SHADOW_STAR_SLICE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHADOW_STAR_SLICE_SCHEMA:
            raise KrennShadowSliceError(
                "shadow star slice schema changed"
            )
        if (
            self.hub_vertex != SLICE_HUB_VERTEX
            or self.sign_seed != SLICE_SIGN_SEED
            or self.sign_modulus != SLICE_SIGN_MODULUS
        ):
            raise KrennShadowSliceError(
                "shadow star slice convention changed"
            )
        if (
            len(self.star_variables) != EXPECTED_STAR_VARIABLE_COUNT
            or len(self.nonstar_variables)
            != EXPECTED_NONSTAR_VARIABLE_COUNT
            or set(self.star_variables).intersection(
                self.nonstar_variables
            )
            or set(self.star_variables).union(self.nonstar_variables)
            != set(
                range(
                    EXPECTED_STAR_VARIABLE_COUNT
                    + EXPECTED_NONSTAR_VARIABLE_COUNT
                )
            )
        ):
            raise KrennShadowSliceError(
                "shadow star variable partition changed"
            )
        if (
            len(self.fixed_shadow_values)
            != EXPECTED_STAR_VARIABLE_COUNT
            + EXPECTED_NONSTAR_VARIABLE_COUNT
            or any(
                self.fixed_shadow_values[variable]
                for variable in self.star_variables
            )
            or any(
                self.fixed_shadow_values[variable] not in {-1, 1}
                for variable in self.nonstar_variables
            )
        ):
            raise KrennShadowSliceError(
                "seed-912 fixed shadow values changed"
            )
        if (
            len(self.coefficient_matrix) != EXPECTED_EQUATION_COUNT
            or any(
                len(row) != EXPECTED_STAR_VARIABLE_COUNT
                for row in self.coefficient_matrix
            )
            or len(self.pivot_columns) != EXPECTED_EQUATION_COUNT
            or len(self.free_columns)
            != EXPECTED_STAR_VARIABLE_COUNT
            - EXPECTED_EQUATION_COUNT
            or set(self.pivot_columns).union(self.free_columns)
            != set(range(EXPECTED_STAR_VARIABLE_COUNT))
        ):
            raise KrennShadowSliceError(
                "shadow star linear system census changed"
            )
        minor = self.pivot_matrix
        replayed_determinant = _determinant_bareiss(minor)
        if (
            self.pivot_determinant != EXPECTED_PIVOT_DETERMINANT
            or replayed_determinant != self.pivot_determinant
            or self.pivot_determinant % SLICE_SIGN_MODULUS
            != EXPECTED_PIVOT_DETERMINANT_MOD_31
        ):
            raise KrennShadowSliceError(
                "seed-912 star minor determinant changed"
            )

    @property
    def pivot_matrix(self) -> IntegerMatrix:
        return tuple(
            tuple(row[column] for column in self.pivot_columns)
            for row in self.coefficient_matrix
        )

    @property
    def pivot_determinant_mod_31(self) -> int:
        return self.pivot_determinant % SLICE_SIGN_MODULUS

    @property
    def target_generic_over_q(self) -> bool:
        return self.pivot_determinant != 0

    def solve(
        self,
        target_coefficients: Sequence[ExactScalar],
    ) -> ShadowStarSliceSolution:
        """Solve any rational occupation vector with both free slots zero."""

        target = _exact_vector(
            target_coefficients,
            EXPECTED_EQUATION_COUNT,
            "shadow occupation target",
        )
        pivot_values = _solve_square_exact(
            self.pivot_matrix, target
        )
        values = [
            Fraction(value) for value in self.fixed_shadow_values
        ]
        for column, value in zip(
            self.pivot_columns, pivot_values
        ):
            values[self.star_variables[column]] = value
        for column in self.free_columns:
            values[self.star_variables[column]] = Fraction(0)
        system = generate_equal_g_shadow_system(SLICE_N, SLICE_D)
        output = system.evaluate(values)
        return ShadowStarSliceSolution(
            target_coefficients=target,
            shadow_values=tuple(values),
            output_coefficients=output,
            pivot_values=pivot_values,
            free_star_values=tuple(
                values[self.star_variables[column]]
                for column in self.free_columns
            ),
        )


def _build_coefficient_matrix(
    system: EqualGShadowSystem,
    star_variables: tuple[int, ...],
    fixed_values: tuple[int, ...],
) -> IntegerMatrix:
    star_positions = {
        variable: column
        for column, variable in enumerate(star_variables)
    }
    rows = []
    for equation in system.equation_terms:
        row = [0] * len(star_variables)
        for monomial in equation:
            selected = tuple(
                variable
                for variable in monomial
                if variable in star_positions
            )
            if len(selected) != 1:
                raise KrennShadowSliceError(
                    "a shadow monomial does not use exactly one hub-star "
                    "variable"
                )
            star_variable = selected[0]
            coefficient = 1
            for variable in monomial:
                if variable != star_variable:
                    coefficient *= fixed_values[variable]
            row[star_positions[star_variable]] += coefficient
        rows.append(tuple(row))
    return tuple(rows)


@lru_cache(maxsize=1)
def generate_n6_d3_shadow_star_slice() -> N6D3ShadowStarSlice:
    """Build and replay the deterministic seed-912 target-generic slice."""

    system = generate_equal_g_shadow_system(SLICE_N, SLICE_D)
    star_variables = tuple(
        variable
        for variable, (i, j, _a, _b) in enumerate(
            system.variable_keys
        )
        if SLICE_HUB_VERTEX in {i, j}
    )
    nonstar_variables = tuple(
        variable
        for variable in range(system.variable_count)
        if variable not in set(star_variables)
    )
    if (
        len(star_variables) != EXPECTED_STAR_VARIABLE_COUNT
        or len(nonstar_variables) != EXPECTED_NONSTAR_VARIABLE_COUNT
    ):
        raise KrennShadowSliceError(
            "n=6,d=3 hub-star variable census changed"
        )
    signs = _seed_912_signs(len(nonstar_variables))
    fixed_values = [0] * system.variable_count
    for variable, sign in zip(nonstar_variables, signs):
        fixed_values[variable] = sign
    fixed = tuple(fixed_values)
    if any(system.evaluate(fixed)):
        raise KrennShadowSliceError(
            "the zero-star slice unexpectedly has a constant term"
        )
    coefficient_matrix = _build_coefficient_matrix(
        system, star_variables, fixed
    )
    pivot_columns = tuple(range(EXPECTED_EQUATION_COUNT))
    free_columns = tuple(
        range(
            EXPECTED_EQUATION_COUNT,
            EXPECTED_STAR_VARIABLE_COUNT,
        )
    )
    minor = tuple(
        tuple(row[column] for column in pivot_columns)
        for row in coefficient_matrix
    )
    determinant = _determinant_bareiss(minor)
    return N6D3ShadowStarSlice(
        star_variables=star_variables,
        nonstar_variables=nonstar_variables,
        fixed_shadow_values=fixed,
        coefficient_matrix=coefficient_matrix,
        pivot_columns=pivot_columns,
        free_columns=free_columns,
        pivot_determinant=determinant,
    )


def solve_n6_d3_shadow_occupations(
    target_coefficients: Sequence[ExactScalar],
) -> ShadowStarSliceSolution:
    """Solve an arbitrary rational vector in the 28 occupation coordinates."""

    return generate_n6_d3_shadow_star_slice().solve(
        target_coefficients
    )


@lru_cache(maxsize=1)
def canonical_ghz_shadow_star_solution() -> ShadowStarSliceSolution:
    """Return the exact seed-912 solution of the canonical GHZ shadow."""

    system = generate_equal_g_shadow_system(SLICE_N, SLICE_D)
    target = system.target_coefficients(
        canonical_ghz_target(SLICE_N, SLICE_D)
    )
    solution = solve_n6_d3_shadow_occupations(target)
    if (
        solution.pivot_values
        != EXPECTED_CANONICAL_GHZ_PIVOT_VALUES
        or any(solution.free_star_values)
    ):
        raise KrennShadowSliceError(
            "canonical GHZ seed-912 solution changed"
        )
    return solution
