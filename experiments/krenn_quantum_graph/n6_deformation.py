r"""Exact differential profile of the natural ``n=6,d=3`` near-miss.

This module studies the full ``729 x 135`` Jacobian at the natural nine-slot
seed.  It proves the characteristic-zero ranks by two-sided certificates:

* a concrete integer minor has nonzero determinant modulo 31, so that same
  integer determinant is nonzero over ``Q``; and
* explicit exact kernel directions give the matching upper bound.

For the full Jacobian, the kernel is exactly the five-dimensional
vertex-scalar gauge.  After deleting the sole defect equation, the kernel
gains one explicit direction

``delta(W[0,1,0,0])=-1`` and ``delta(W[2,3,0,0])=+1``.

The full Jacobian sends this direction to ``-e_70``, so it cancels the
near-miss residual to first order while preserving the other 728 equations.
The seed is not an exact solution, and this infinitesimal statement is not
presented as a finite witness.

Reduction modulo 31 is used only to exhibit nonzero reductions of concrete
integer minors.  No finite-field solution or field-to-``C`` transfer is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    generate_sparse_system,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    n6_d3_seed_witness,
)
from experiments.krenn_quantum_graph.witness import (
    SparseWitness,
    evaluate_exact,
)


N6_DEFORMATION_SCHEMA = "krenn-n6-d3-natural-deformation-v1"
MINOR_MODULUS = 31

IntegerVector = tuple[int, ...]
IntegerMatrix = tuple[IntegerVector, ...]


class KrennN6DeformationError(RuntimeError):
    """The natural-seed Jacobian, gauge, or minor replay failed."""


def natural_seed_jacobian() -> IntegerMatrix:
    """Return the full integer Jacobian at the natural nine-slot seed."""

    system = generate_sparse_system(6, 3)
    witness = n6_d3_seed_witness()
    values = witness.as_dict()
    rows = []
    for equation in range(system.equation_count):
        row = [0] * system.variable_count
        for monomial in system.equation_monomials(equation):
            for position, differentiated_variable in enumerate(monomial):
                derivative = Fraction(1)
                for factor_position, factor_variable in enumerate(monomial):
                    if factor_position != position:
                        derivative *= values.get(
                            factor_variable, Fraction(0)
                        )
                if derivative.denominator != 1:
                    raise KrennN6DeformationError(
                        "the integral natural seed produced a fractional "
                        "Jacobian entry"
                    )
                row[differentiated_variable] += derivative.numerator
        rows.append(tuple(row))
    result = tuple(rows)
    if (
        len(result) != 729
        or any(len(row) != 135 for row in result)
    ):
        raise KrennN6DeformationError(
            "the natural Jacobian shape is not 729 x 135"
        )
    return result


def _vertex_basis_edge_exponent(i: int, j: int, direction: int) -> int:
    """Exponent for ``epsilon_direction=e_direction-e_5`` on edge ``ij``."""

    return (
        int(i == direction)
        + int(j == direction)
        - int(i == 5)
        - int(j == 5)
    )


def vertex_scalar_gauge_matrix(
    witness: SparseWitness | None = None,
) -> IntegerMatrix:
    r"""Return the five vertex-scalar gauge tangents at ``witness``.

    The finite action is

    .. math::

        W_{ij}^{ab}\longmapsto\lambda_i\lambda_j W_{ij}^{ab},
        \qquad\prod_i\lambda_i=1.

    Directions ``0,...,4`` use exponent vectors ``e_r-e_5``.
    """

    witness = n6_d3_seed_witness() if witness is None else witness
    if (witness.n, witness.d) != (6, 3):
        raise KrennN6DeformationError(
            "the natural gauge certificate requires n=6,d=3"
        )
    matrix = [[0] * 5 for _ in range(variable_count(6, 3))]
    for index, value in witness.entries:
        if value.denominator != 1:
            raise KrennN6DeformationError(
                "the integer gauge certificate needs integral weights"
            )
        i, j, _a, _b = variable_key(6, 3, index)
        for direction in range(5):
            matrix[index][direction] = (
                value.numerator
                * _vertex_basis_edge_exponent(i, j, direction)
            )
    return tuple(tuple(row) for row in matrix)


def gauge_action_preserves_every_monomial(
    system: SparsePolynomialSystem | None = None,
) -> bool:
    """Replay the five zero weights on all 10,935 matching monomials."""

    system = generate_sparse_system(6, 3) if system is None else system
    if (system.n, system.d) != (6, 3):
        raise KrennN6DeformationError(
            "the n=6 gauge replay requires the n=6,d=3 system"
        )
    for monomial in system.monomial_variable_indices:
        for direction in range(5):
            weight = 0
            for variable in monomial:
                i, j, _a, _b = variable_key(6, 3, variable)
                weight += _vertex_basis_edge_exponent(
                    i, j, direction
                )
            if weight:
                return False
    return True


def natural_repair_direction() -> IntegerVector:
    """Return ``delta`` with ``J delta = -e_70``."""

    direction = [0] * variable_count(6, 3)
    direction[variable_index(6, 3, 0, 1, 0, 0)] = -1
    direction[variable_index(6, 3, 2, 3, 0, 0)] = 1
    return tuple(direction)


def _matrix_vector_product(
    matrix: Sequence[Sequence[int]],
    vector: Sequence[int],
) -> IntegerVector:
    vector = tuple(map(int, vector))
    if any(len(row) != len(vector) for row in matrix):
        raise KrennN6DeformationError(
            "integer matrix-vector dimensions differ"
        )
    return tuple(
        sum(int(value) * vector[column] for column, value in enumerate(row))
        for row in matrix
    )


def _matrix_product(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> IntegerMatrix:
    left = tuple(tuple(map(int, row)) for row in left)
    right = tuple(tuple(map(int, row)) for row in right)
    if not left or not right or len(left[0]) != len(right):
        raise KrennN6DeformationError(
            "integer matrix-product dimensions differ"
        )
    width = len(right[0])
    if any(len(row) != width for row in right):
        raise KrennN6DeformationError(
            "the right integer matrix is ragged"
        )
    return tuple(
        tuple(
            sum(
                left[row][middle] * right[middle][column]
                for middle in range(len(right))
            )
            for column in range(width)
        )
        for row in range(len(left))
    )


def _rank_over_q(
    matrix: Sequence[Sequence[int]],
) -> int:
    """Exact rational rank for the small kernel-direction matrices."""

    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennN6DeformationError("the rational matrix is ragged")
    rank = 0
    for column in range(width):
        selected = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        rows[rank], rows[selected] = rows[selected], rows[rank]
        pivot = rows[rank][column]
        rows[rank] = [value / pivot for value in rows[rank]]
        for row in range(rank + 1, len(rows)):
            factor = rows[row][column]
            if factor:
                rows[row] = [
                    left - factor * right
                    for left, right in zip(rows[row], rows[rank])
                ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _determinant_mod(
    matrix: Sequence[Sequence[int]], modulus: int
) -> int:
    rows = [
        [int(value) % modulus for value in row] for row in matrix
    ]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise KrennN6DeformationError(
            "a modular determinant requires a square matrix"
        )
    determinant = 1
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
            return 0
        if selected != column:
            rows[column], rows[selected] = (
                rows[selected],
                rows[column],
            )
            determinant = -determinant
        pivot = rows[column][column] % modulus
        determinant = determinant * pivot % modulus
        inverse = pow(pivot, -1, modulus)
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
class IntegerMinorCertificate:
    """A nonzero modular reduction of one concrete integer minor."""

    matrix_shape: tuple[int, int]
    modulus: int
    row_indices: tuple[int, ...]
    column_indices: tuple[int, ...]
    determinant_mod: int

    def __post_init__(self) -> None:
        size = len(self.row_indices)
        if (
            self.modulus != MINOR_MODULUS
            or size != len(self.column_indices)
            or size < 1
            or not 0 <= self.determinant_mod < self.modulus
            or self.determinant_mod == 0
            or len(set(self.row_indices)) != size
            or len(set(self.column_indices)) != size
            or any(
                row < 0 or row >= self.matrix_shape[0]
                for row in self.row_indices
            )
            or any(
                column < 0 or column >= self.matrix_shape[1]
                for column in self.column_indices
            )
        ):
            raise KrennN6DeformationError(
                "the integer-minor certificate is malformed"
            )

    @property
    def size(self) -> int:
        return len(self.row_indices)

    @property
    def rank_lower_bound_over_q(self) -> int:
        return self.size

    def to_dict(self) -> dict:
        return {
            "matrix_shape": list(self.matrix_shape),
            "modulus": self.modulus,
            "minor_size": self.size,
            "row_indices": list(self.row_indices),
            "column_indices": list(self.column_indices),
            "determinant_mod_31": self.determinant_mod,
            "proof_role": (
                "The selected matrix entries are integers. A nonzero "
                "determinant reduction modulo 31 proves the same integer "
                "determinant is nonzero over Q."
            ),
            "finite_field_solution_claim": False,
            "finite_field_solution_to_C_transfer_used": False,
        }


def _nonzero_modular_minor(
    matrix: Sequence[Sequence[int]],
    *,
    modulus: int = MINOR_MODULUS,
) -> IntegerMinorCertificate:
    """Find a deterministic maximal-rank minor and replay its determinant."""

    original = tuple(tuple(map(int, row)) for row in matrix)
    if not original:
        raise KrennN6DeformationError(
            "a modular minor requires a nonempty matrix"
        )
    width = len(original[0])
    if not width or any(len(row) != width for row in original):
        raise KrennN6DeformationError(
            "a modular minor requires a rectangular matrix"
        )
    labels = list(range(len(original)))
    work = [
        [value % modulus for value in row] for row in original
    ]
    rank = 0
    selected_rows: list[int] = []
    selected_columns: list[int] = []
    for column in range(width):
        selected = next(
            (
                row
                for row in range(rank, len(work))
                if work[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        work[rank], work[selected] = work[selected], work[rank]
        labels[rank], labels[selected] = labels[selected], labels[rank]
        inverse = pow(work[rank][column], -1, modulus)
        for entry in range(column, width):
            work[rank][entry] = (
                work[rank][entry] * inverse % modulus
            )
        for row in range(rank + 1, len(work)):
            factor = work[row][column]
            if factor:
                for entry in range(column, width):
                    work[row][entry] = (
                        work[row][entry]
                        - factor * work[rank][entry]
                    ) % modulus
        selected_rows.append(labels[rank])
        selected_columns.append(column)
        rank += 1
        if rank == len(work):
            break

    minor = tuple(
        tuple(
            original[row][column]
            for column in selected_columns
        )
        for row in selected_rows
    )
    determinant = _determinant_mod(minor, modulus)
    if not determinant:
        raise KrennN6DeformationError(
            "the selected modular pivot minor is singular"
        )
    return IntegerMinorCertificate(
        matrix_shape=(len(original), width),
        modulus=modulus,
        row_indices=tuple(selected_rows),
        column_indices=tuple(selected_columns),
        determinant_mod=determinant,
    )


@dataclass(frozen=True)
class N6NaturalDeformationCertificate:
    """Exact rank sandwiches for the full and defect-deleted Jacobians."""

    full_minor: IntegerMinorCertificate
    defect_deleted_minor: IntegerMinorCertificate
    exact_checks: Mapping[str, bool]
    schema: str = N6_DEFORMATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != N6_DEFORMATION_SCHEMA:
            raise KrennN6DeformationError(
                "the n=6 deformation schema changed"
            )
        if not self.exact:
            failed = [
                name for name, passed in self.exact_checks.items() if not passed
            ]
            raise KrennN6DeformationError(
                f"the n=6 deformation certificate failed: {failed}"
            )

    @property
    def exact(self) -> bool:
        return bool(self.exact_checks) and all(self.exact_checks.values())

    def to_dict(self) -> dict:
        system = generate_sparse_system(6, 3)
        evaluation = evaluate_exact(
            system, n6_d3_seed_witness()
        )
        return {
            "schema": self.schema,
            "parameters": {
                "n": 6,
                "d": 3,
                "defect_equation": N6_D3_SEED_DEFECT_EQUATION,
                "prime": MINOR_MODULUS,
            },
            "point": {
                "name": "natural-nine-slot-near-miss",
                "support_size": 9,
                "exact_GHZ_solution": evaluation.satisfied,
                "nonzero_residual_count": (
                    evaluation.nonzero_residual_count
                ),
                "defect_residual": evaluation.residuals[
                    N6_D3_SEED_DEFECT_EQUATION
                ].numerator,
            },
            "full_jacobian": {
                "shape": [729, 135],
                "rank_over_Q": 130,
                "nullity_over_Q": 5,
                "lower_bound_certificate": self.full_minor.to_dict(),
                "upper_bound_certificate": (
                    "five independent exact vertex-scalar gauge "
                    "directions killed by the Jacobian"
                ),
                "kernel_equals_vertex_scalar_gauge_over_Q": True,
            },
            "defect_deleted_jacobian": {
                "removed_equation": N6_D3_SEED_DEFECT_EQUATION,
                "shape": [728, 135],
                "row_order": (
                    "increasing original equation order with equation 70 "
                    "omitted"
                ),
                "rank_over_Q": 129,
                "nullity_over_Q": 6,
                "lower_bound_certificate": (
                    self.defect_deleted_minor.to_dict()
                ),
                "upper_bound_certificate": (
                    "the five gauge directions plus the independent "
                    "repair direction lie in the deleted-row kernel"
                ),
                "kernel_equals_gauge_plus_repair_over_Q": True,
            },
            "vertex_scalar_gauge": {
                "shape": [135, 5],
                "rank_over_Q": 5,
                "basis": [
                    {
                        "positive_vertex": vertex,
                        "negative_vertex": 5,
                    }
                    for vertex in range(5)
                ],
                "finite_action": (
                    "W_ij^ab -> lambda_i*lambda_j*W_ij^ab "
                    "with product_i(lambda_i)=1"
                ),
                "preserves_every_matching_monomial": True,
            },
            "repair_direction": {
                "nonzero_coordinates": [
                    {
                        "coordinate": [0, 1, 0, 0],
                        "value": -1,
                    },
                    {
                        "coordinate": [2, 3, 0, 0],
                        "value": 1,
                    },
                ],
                "full_jacobian_image": "-e_70",
                "preserves_other_728_equations_to_first_order": True,
                "finite_exact_solution_proved": False,
            },
            "modular_integer_minor_role": {
                "statement": (
                    "Nonzero reductions of concrete integer minors give "
                    "lower bounds over Q. Exact kernel directions give the "
                    "upper bounds."
                ),
                "finite_field_solution_claim": False,
                "finite_field_solution_to_C_transfer_used": False,
                "proof_over_C": (
                    "The exact rational rank statements persist by scalar "
                    "extension after they have been proved over Q."
                ),
            },
            "exact_checks": dict(self.exact_checks),
            "claim_boundary": {
                "natural_seed_is_exact_GHZ_solution": False,
                "first_order_repair_is_finite_solution": False,
                "full_Jacobian_kernel_certified": True,
                "defect_deleted_kernel_certified": True,
                "finite_field_membership_proved": False,
                "exact_GHZ_image_membership_decided_here": False,
            },
        }


def certify_n6_natural_deformation(
) -> N6NaturalDeformationCertificate:
    """Build and verify both exact characteristic-zero rank sandwiches."""

    system = generate_sparse_system(6, 3)
    witness = n6_d3_seed_witness()
    jacobian = natural_seed_jacobian()
    gauge = vertex_scalar_gauge_matrix(witness)
    direction = natural_repair_direction()
    defect = N6_D3_SEED_DEFECT_EQUATION
    expected_direction_image = tuple(
        -int(equation == defect) for equation in range(729)
    )
    direction_image = _matrix_vector_product(jacobian, direction)
    jacobian_times_gauge = _matrix_product(jacobian, gauge)
    gauge_rank = _rank_over_q(gauge)
    gauge_plus_direction = tuple(
        tuple((*row, direction[index]))
        for index, row in enumerate(gauge)
    )
    gauge_plus_direction_rank = _rank_over_q(
        gauge_plus_direction
    )

    kept_equations = tuple(
        equation for equation in range(729) if equation != defect
    )
    defect_deleted = tuple(
        jacobian[equation] for equation in kept_equations
    )
    deleted_times_gauge = _matrix_product(defect_deleted, gauge)
    deleted_times_direction = _matrix_vector_product(
        defect_deleted, direction
    )

    full_minor = _nonzero_modular_minor(jacobian)
    deleted_minor = _nonzero_modular_minor(defect_deleted)

    checks = {
        "natural_seed_support_size_9": witness.support_size == 9,
        "full_jacobian_shape_729_by_135": (
            len(jacobian) == 729
            and all(len(row) == 135 for row in jacobian)
        ),
        "vertex_gauge_shape_135_by_5": (
            len(gauge) == 135
            and all(len(row) == 5 for row in gauge)
        ),
        "vertex_gauge_rank_5_over_Q": gauge_rank == 5,
        "gauge_action_preserves_all_10935_monomials": (
            system.monomial_count == 10_935
            and gauge_action_preserves_every_monomial(system)
        ),
        "full_jacobian_kills_vertex_gauge_exactly": all(
            value == 0
            for row in jacobian_times_gauge
            for value in row
        ),
        "full_integer_minor_size_130_nonzero_mod_31": (
            full_minor.size == 130
            and full_minor.determinant_mod != 0
        ),
        "full_rank_sandwich_proves_rank_130_over_Q": (
            full_minor.rank_lower_bound_over_q == 130
            and 135 - gauge_rank == 130
        ),
        "full_kernel_equals_gauge_over_Q": (
            gauge_rank == 135 - 130
        ),
        "repair_direction_has_two_declared_coordinates": (
            tuple(
                (index, value)
                for index, value in enumerate(direction)
                if value
            )
            == (
                (variable_index(6, 3, 0, 1, 0, 0), -1),
                (variable_index(6, 3, 2, 3, 0, 0), 1),
            )
        ),
        "full_jacobian_times_repair_is_minus_e70": (
            direction_image == expected_direction_image
        ),
        "gauge_plus_repair_rank_6_over_Q": (
            gauge_plus_direction_rank == 6
        ),
        "defect_deleted_jacobian_shape_728_by_135": (
            len(defect_deleted) == 728
            and all(len(row) == 135 for row in defect_deleted)
        ),
        "defect_deleted_jacobian_kills_gauge": all(
            value == 0
            for row in deleted_times_gauge
            for value in row
        ),
        "defect_deleted_jacobian_kills_repair": all(
            value == 0 for value in deleted_times_direction
        ),
        "defect_deleted_integer_minor_size_129_nonzero_mod_31": (
            deleted_minor.size == 129
            and deleted_minor.determinant_mod != 0
        ),
        "deleted_rank_sandwich_proves_rank_129_over_Q": (
            deleted_minor.rank_lower_bound_over_q == 129
            and 135 - gauge_plus_direction_rank == 129
        ),
        "deleted_kernel_equals_gauge_plus_repair_over_Q": (
            gauge_plus_direction_rank == 135 - 129
        ),
    }
    return N6NaturalDeformationCertificate(
        full_minor=full_minor,
        defect_deleted_minor=deleted_minor,
        exact_checks=checks,
    )


__all__ = (
    "IntegerMinorCertificate",
    "KrennN6DeformationError",
    "MINOR_MODULUS",
    "N6_DEFORMATION_SCHEMA",
    "N6NaturalDeformationCertificate",
    "certify_n6_natural_deformation",
    "gauge_action_preserves_every_monomial",
    "natural_repair_direction",
    "natural_seed_jacobian",
    "vertex_scalar_gauge_matrix",
)
