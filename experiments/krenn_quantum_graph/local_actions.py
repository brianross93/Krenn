r"""Exact local linear actions for the Krenn perfect-matching tensor map.

For one exact matrix

``A_i : Q^d -> Q^e``

at every vertex, the edge and target actions are

.. math::

    W'_{ij} = A_i W_{ij} A_j^T,\qquad
    T' = (A_0\otimes\cdots\otimes A_{n-1})T.

The perfect-matching map is equivariant for this action:

.. math::

    \Phi_e(W') =
    (A_0\otimes\cdots\otimes A_{n-1})\Phi_d(W).

Rectangular projections and invertible square local changes of basis use the
same primitive.  The implementation is exact over ``Q`` and keeps the
canonical endpoint convention ``i < j``: ``A_i`` always acts on the first
color slot and ``A_j`` on the second.

Equivariance transports witnesses forward.  By itself it is not a target
nonimage theorem: a backward no-go argument additionally needs an exact,
independently checkable proof that the transformed target is outside the
smaller perfect-matching image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from numbers import Integral
from typing import Iterable, Sequence, TypeAlias

from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    variable_index,
)
from experiments.krenn_quantum_graph.targets import ColoringTarget
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.witness import SparseWitness


ExactScalar: TypeAlias = int | Fraction
ExactMatrix: TypeAlias = tuple[tuple[Fraction, ...], ...]

LOCAL_ACTION_SCHEMA = "krenn-exact-local-action-v1"
LOCAL_EQUIVARIANCE_SCHEMA = "krenn-local-equivariance-certificate-v1"


class KrennLocalActionError(ValueError):
    """A local action, composition, or equivariance replay is malformed."""


def _exact_positive_integer(value, label: str) -> int:
    if not isinstance(value, Integral):
        raise KrennLocalActionError(f"{label} must be an exact integer")
    result = int(value)
    if result < 1:
        raise KrennLocalActionError(f"{label} must be positive")
    return result


def _exact_fraction(value, label: str) -> Fraction:
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    raise KrennLocalActionError(
        f"{label} must be an exact integer or Fraction"
    )


def _matrix_rank(matrix: ExactMatrix) -> int:
    rows = [list(row) for row in matrix]
    pivot_row = 0
    column_count = len(rows[0])
    for column in range(column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        if selected != pivot_row:
            rows[pivot_row], rows[selected] = (
                rows[selected],
                rows[pivot_row],
            )
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [entry / pivot for entry in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(
                    rows[row], rows[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


@dataclass(frozen=True)
class ExactLocalAction:
    """One common-shape exact matrix at every tensor factor."""

    n: int
    input_dimension: int
    output_dimension: int
    matrices: tuple[ExactMatrix, ...]
    schema: str = LOCAL_ACTION_SCHEMA

    def __post_init__(self) -> None:
        n = _exact_positive_integer(self.n, "vertex count")
        input_dimension = _exact_positive_integer(
            self.input_dimension, "input local dimension"
        )
        output_dimension = _exact_positive_integer(
            self.output_dimension, "output local dimension"
        )
        if n < 2 or n % 2:
            raise KrennLocalActionError(
                "the Krenn vertex count must be even and at least 2"
            )
        if self.schema != LOCAL_ACTION_SCHEMA:
            raise KrennLocalActionError("local-action schema changed")
        try:
            raw_matrices = tuple(self.matrices)
        except TypeError as error:
            raise KrennLocalActionError(
                "local matrices must be an iterable"
            ) from error
        if len(raw_matrices) != n:
            raise KrennLocalActionError(
                "there must be exactly one local matrix per vertex"
            )

        canonical: list[ExactMatrix] = []
        for vertex, raw_matrix in enumerate(raw_matrices):
            try:
                raw_rows = tuple(raw_matrix)
            except TypeError as error:
                raise KrennLocalActionError(
                    f"local matrix {vertex} must contain rows"
                ) from error
            if len(raw_rows) != output_dimension:
                raise KrennLocalActionError(
                    "every local matrix must have output_dimension rows"
                )
            rows: list[tuple[Fraction, ...]] = []
            for row_number, raw_row in enumerate(raw_rows):
                try:
                    entries = tuple(raw_row)
                except TypeError as error:
                    raise KrennLocalActionError(
                        f"row {row_number} of local matrix {vertex} "
                        "must be iterable"
                    ) from error
                if len(entries) != input_dimension:
                    raise KrennLocalActionError(
                        "every local-matrix row must have "
                        "input_dimension entries"
                    )
                rows.append(
                    tuple(
                        _exact_fraction(
                            entry,
                            (
                                f"local matrix {vertex} entry "
                                f"({row_number},{column})"
                            ),
                        )
                        for column, entry in enumerate(entries)
                    )
                )
            canonical.append(tuple(rows))

        object.__setattr__(self, "n", n)
        object.__setattr__(
            self, "input_dimension", input_dimension
        )
        object.__setattr__(
            self, "output_dimension", output_dimension
        )
        object.__setattr__(self, "matrices", tuple(canonical))

    @classmethod
    def from_matrices(
        cls,
        matrices: Iterable[Sequence[Sequence[ExactScalar]]],
    ) -> "ExactLocalAction":
        """Infer ``n``, ``d``, and ``e`` from nonempty local matrices."""

        try:
            rows_by_vertex = tuple(
                tuple(tuple(row) for row in matrix)
                for matrix in matrices
            )
        except TypeError as error:
            raise KrennLocalActionError(
                "local matrices must be a nested iterable"
            ) from error
        if not rows_by_vertex:
            raise KrennLocalActionError(
                "there must be at least one local matrix"
            )
        if not rows_by_vertex[0] or not rows_by_vertex[0][0]:
            raise KrennLocalActionError(
                "local matrices must have positive dimensions"
            )
        return cls(
            n=len(rows_by_vertex),
            input_dimension=len(rows_by_vertex[0][0]),
            output_dimension=len(rows_by_vertex[0]),
            matrices=rows_by_vertex,
        )

    @classmethod
    def identity(cls, n: int, d: int) -> "ExactLocalAction":
        """Return the identity at every vertex."""

        n = _exact_positive_integer(n, "vertex count")
        d = _exact_positive_integer(d, "local dimension")
        identity = tuple(
            tuple(Fraction(row == column) for column in range(d))
            for row in range(d)
        )
        return cls(
            n=n,
            input_dimension=d,
            output_dimension=d,
            matrices=(identity,) * n,
        )

    @property
    def square(self) -> bool:
        return self.input_dimension == self.output_dimension

    @property
    def local_gl(self) -> bool:
        """Whether every square local matrix is invertible over ``Q``."""

        return self.square and all(
            _matrix_rank(matrix) == self.input_dimension
            for matrix in self.matrices
        )

    @property
    def rectangular(self) -> bool:
        return not self.square


def compose_local_actions(
    after: ExactLocalAction,
    before: ExactLocalAction,
) -> ExactLocalAction:
    """Return ``after o before`` vertex by vertex."""

    if not isinstance(after, ExactLocalAction) or not isinstance(
        before, ExactLocalAction
    ):
        raise KrennLocalActionError(
            "composition requires two exact local actions"
        )
    if after.n != before.n:
        raise KrennLocalActionError(
            "composed local actions need the same vertex count"
        )
    if before.output_dimension != after.input_dimension:
        raise KrennLocalActionError(
            "composed local-action dimensions do not meet"
        )
    matrices = tuple(
        tuple(
            tuple(
                sum(
                    (
                        after_matrix[row][middle]
                        * before_matrix[middle][column]
                        for middle in range(after.input_dimension)
                    ),
                    Fraction(0),
                )
                for column in range(before.input_dimension)
            )
            for row in range(after.output_dimension)
        )
        for after_matrix, before_matrix in zip(
            after.matrices, before.matrices, strict=True
        )
    )
    return ExactLocalAction(
        n=before.n,
        input_dimension=before.input_dimension,
        output_dimension=after.output_dimension,
        matrices=matrices,
    )


def apply_local_action_to_witness(
    witness: SparseWitness,
    action: ExactLocalAction,
) -> SparseWitness:
    """Apply ``W'_{ij}=A_i W_{ij} A_j^T`` on canonical edges."""

    if not isinstance(witness, SparseWitness):
        raise KrennLocalActionError(
            "the local edge action requires a SparseWitness"
        )
    if not isinstance(action, ExactLocalAction):
        raise KrennLocalActionError(
            "the local edge action requires an ExactLocalAction"
        )
    if (witness.n, witness.d) != (
        action.n,
        action.input_dimension,
    ):
        raise KrennLocalActionError(
            "witness dimensions do not match the local-action source"
        )

    output_dimension = action.output_dimension
    source_blocks = {
        edge: [
            [Fraction(0) for _b in range(witness.d)]
            for _a in range(witness.d)
        ]
        for edge in canonical_edges(witness.n)
    }
    for (i, j, a, b), value in witness.coordinate_entries():
        source_blocks[(i, j)][a][b] = Fraction(value)

    transformed: dict[int, Fraction] = {}
    for i, j in canonical_edges(witness.n):
        left = action.matrices[i]
        right = action.matrices[j]
        block = source_blocks[(i, j)]
        for alpha in range(output_dimension):
            for beta in range(output_dimension):
                value = sum(
                    (
                        left[alpha][a]
                        * block[a][b]
                        * right[beta][b]
                        for a in range(witness.d)
                        for b in range(witness.d)
                    ),
                    Fraction(0),
                )
                if value:
                    transformed[
                        variable_index(
                            witness.n,
                            output_dimension,
                            i,
                            j,
                            alpha,
                            beta,
                        )
                    ] = value
    return SparseWitness.from_index_values(
        witness.n, output_dimension, transformed
    )


def apply_local_action_to_target(
    target: ColoringTarget,
    action: ExactLocalAction,
) -> ColoringTarget:
    """Apply the tensor product of the local matrices to ``target``."""

    if not isinstance(target, ColoringTarget):
        raise KrennLocalActionError(
            "the local tensor action requires a ColoringTarget"
        )
    if not isinstance(action, ExactLocalAction):
        raise KrennLocalActionError(
            "the local tensor action requires an ExactLocalAction"
        )
    if (target.n, target.d) != (
        action.n,
        action.input_dimension,
    ):
        raise KrennLocalActionError(
            "target dimensions do not match the local-action source"
        )

    coefficients: list[Fraction] = []
    for output_coloring in product(
        range(action.output_dimension), repeat=action.n
    ):
        coefficient = sum(
            (
                Fraction(value)
                * _basis_transition_coefficient(
                    action, output_coloring, input_coloring
                )
                for input_coloring, value in target.entries
            ),
            Fraction(0),
        )
        coefficients.append(coefficient)
    return ColoringTarget.from_dense(
        action.n, action.output_dimension, coefficients
    )


def _basis_transition_coefficient(
    action: ExactLocalAction,
    output_coloring: Sequence[int],
    input_coloring: Sequence[int],
) -> Fraction:
    coefficient = Fraction(1)
    for vertex in range(action.n):
        coefficient *= action.matrices[vertex][
            output_coloring[vertex]
        ][input_coloring[vertex]]
        if not coefficient:
            break
    return coefficient


@dataclass(frozen=True)
class LocalEquivarianceCertificate:
    """A self-replaying exact certificate for one witness and target."""

    action: ExactLocalAction
    source_witness: SparseWitness
    source_target: ColoringTarget
    schema: str = LOCAL_EQUIVARIANCE_SCHEMA
    source_output: ColoringTarget = field(init=False)
    source_residual: ColoringTarget = field(init=False)
    transformed_witness: SparseWitness = field(init=False)
    transformed_target: ColoringTarget = field(init=False)
    transformed_output: ColoringTarget = field(init=False)
    transformed_residual: ColoringTarget = field(init=False)
    expected_transformed_output: ColoringTarget = field(init=False)
    expected_transformed_residual: ColoringTarget = field(init=False)
    map_equivariant: bool = field(init=False)
    residual_equivariant: bool = field(init=False)
    source_satisfied: bool = field(init=False)
    transformed_satisfied: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != LOCAL_EQUIVARIANCE_SCHEMA:
            raise KrennLocalActionError(
                "local-equivariance certificate schema changed"
            )
        if not isinstance(self.action, ExactLocalAction):
            raise KrennLocalActionError(
                "certificate action must be an ExactLocalAction"
            )
        if not isinstance(self.source_witness, SparseWitness):
            raise KrennLocalActionError(
                "certificate source must be a SparseWitness"
            )
        if not isinstance(self.source_target, ColoringTarget):
            raise KrennLocalActionError(
                "certificate target must be a ColoringTarget"
            )
        source_dimensions = (
            self.action.n,
            self.action.input_dimension,
        )
        if (self.source_witness.n, self.source_witness.d) != (
            source_dimensions
        ) or (self.source_target.n, self.source_target.d) != (
            source_dimensions
        ):
            raise KrennLocalActionError(
                "certificate source dimensions do not match its action"
            )

        source_comparison = MatchingTensorMap(
            *source_dimensions
        ).compare_exact(self.source_witness, self.source_target)
        transformed_witness = apply_local_action_to_witness(
            self.source_witness, self.action
        )
        transformed_target = apply_local_action_to_target(
            self.source_target, self.action
        )
        transformed_comparison = MatchingTensorMap(
            self.action.n, self.action.output_dimension
        ).compare_exact(transformed_witness, transformed_target)
        expected_output = apply_local_action_to_target(
            source_comparison.output, self.action
        )
        expected_residual = apply_local_action_to_target(
            source_comparison.residual, self.action
        )

        map_equivariant = (
            transformed_comparison.output == expected_output
        )
        residual_equivariant = (
            transformed_comparison.residual == expected_residual
        )
        source_satisfied = source_comparison.satisfied
        transformed_satisfied = transformed_comparison.satisfied

        object.__setattr__(
            self, "source_output", source_comparison.output
        )
        object.__setattr__(
            self, "source_residual", source_comparison.residual
        )
        object.__setattr__(
            self, "transformed_witness", transformed_witness
        )
        object.__setattr__(
            self, "transformed_target", transformed_target
        )
        object.__setattr__(
            self, "transformed_output", transformed_comparison.output
        )
        object.__setattr__(
            self,
            "transformed_residual",
            transformed_comparison.residual,
        )
        object.__setattr__(
            self, "expected_transformed_output", expected_output
        )
        object.__setattr__(
            self, "expected_transformed_residual", expected_residual
        )
        object.__setattr__(self, "map_equivariant", map_equivariant)
        object.__setattr__(
            self, "residual_equivariant", residual_equivariant
        )
        object.__setattr__(
            self, "source_satisfied", source_satisfied
        )
        object.__setattr__(
            self, "transformed_satisfied", transformed_satisfied
        )
        if not map_equivariant or not residual_equivariant:
            raise KrennLocalActionError(
                "the exact local-equivariance replay failed"
            )
        if source_satisfied and not transformed_satisfied:
            raise KrennLocalActionError(
                "an exact source solution failed forward transport"
            )

    @property
    def exact(self) -> bool:
        return (
            self.map_equivariant
            and self.residual_equivariant
            and (not self.source_satisfied or self.transformed_satisfied)
        )

    @property
    def forward_solution_transport_certified(self) -> bool:
        return (
            self.exact
            and self.source_satisfied
            and self.transformed_satisfied
        )

    @property
    def individual_target_nonimage_proved(self) -> bool:
        return False

    @property
    def no_go_propagation_proved(self) -> bool:
        return False

    @property
    def claim_boundary(self) -> str:
        return (
            "Exact equivariance transports finite witnesses forward.  "
            "It does not by itself prove target nonimage or a no-go "
            "statement.  Backward no-go propagation requires a separate "
            "exact nonimage certificate for the transformed target."
        )


def certify_local_equivariance(
    witness: SparseWitness,
    target: ColoringTarget,
    action: ExactLocalAction,
) -> LocalEquivarianceCertificate:
    """Replay ``Phi_e(A.W)=A.Phi_d(W)`` and residual transport over ``Q``."""

    return LocalEquivarianceCertificate(
        action=action,
        source_witness=witness,
        source_target=target,
    )


# Descriptive aliases for callers that prefer "transform" terminology.
transform_witness_local = apply_local_action_to_witness
transform_target_local = apply_local_action_to_target
