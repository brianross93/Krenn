"""Native exact-Q deformation and gauge quotient at ``n=4,d=3``."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np

from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    variable_key,
)
from experiments.krenn_quantum_graph.witness import (
    SparseWitness,
    fraction_mod,
    require_exact_witness,
)


DEFORMATION_SCHEMA = "krenn-quantum-graph-deformation-certificate-v2"


class KrennDeformationError(RuntimeError):
    """The native Jacobian, gauge, or quotient replay failed."""


def jacobian_matrix(
    system: SparsePolynomialSystem,
    witness: SparseWitness,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return the dense exact-Q Jacobian at a sparse witness."""

    if (system.n, system.d) != (witness.n, witness.d):
        raise KrennDeformationError(
            "Jacobian witness and system parameters differ"
        )
    values = witness.as_dict()
    rows = [
        [Fraction(0) for _ in range(system.variable_count)]
        for _ in range(system.equation_count)
    ]
    for equation in range(system.equation_count):
        row = rows[equation]
        for monomial in system.equation_monomials(equation):
            for position, differentiated_variable in enumerate(monomial):
                derivative = Fraction(1)
                for factor_position, factor_variable in enumerate(monomial):
                    if factor_position != position:
                        derivative *= values.get(
                            factor_variable, Fraction(0)
                        )
                row[differentiated_variable] += derivative
    return tuple(tuple(row) for row in rows)


def rref_over_q(
    matrix: Sequence[Sequence[Fraction | int]],
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[int, ...]]:
    """Deterministic reduced row echelon form over the rationals."""

    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return (), ()
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise KrennDeformationError("rational matrix is ragged")
    pivot_row = 0
    pivots: list[int] = []
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
        rows[pivot_row] = [
            value / pivot for value in rows[pivot_row]
        ]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    rows[row], rows[pivot_row], strict=True
                )
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(tuple(row) for row in rows), tuple(pivots)


def rank_over_q(
    matrix: Sequence[Sequence[Fraction | int]],
) -> int:
    return len(rref_over_q(matrix)[1])


def _multiply_over_q(
    left: Sequence[Sequence[Fraction | int]],
    right: Sequence[Sequence[Fraction | int]],
) -> tuple[tuple[Fraction, ...], ...]:
    left = tuple(tuple(map(Fraction, row)) for row in left)
    right = tuple(tuple(map(Fraction, row)) for row in right)
    if not left or not right or len(left[0]) != len(right):
        raise KrennDeformationError(
            "rational matrix product has incompatible shapes"
        )
    column_count = len(right[0])
    if any(len(row) != column_count for row in right):
        raise KrennDeformationError("rational matrix is ragged")
    return tuple(
        tuple(
            sum(
                left[row][middle] * right[middle][column]
                for middle in range(len(right))
            )
            for column in range(column_count)
        )
        for row in range(len(left))
    )


_GAUGE_EDGE_DATA: Mapping[tuple[int, int], tuple[int, int]] = {
    (0, 1): (0, 1),
    (2, 3): (0, -1),
    (0, 2): (1, 1),
    (1, 3): (1, -1),
    (0, 3): (2, 1),
    (1, 2): (2, -1),
}


def reciprocal_gauge_matrix(
    witness: SparseWitness,
) -> tuple[tuple[Fraction, ...], ...]:
    """Infinitesimal action of the three complementary-edge rescalings."""

    if (witness.n, witness.d) != (4, 3):
        raise KrennDeformationError(
            "the sharp reciprocal gauge fixture is n=4,d=3"
        )
    matrix = [
        [Fraction(0), Fraction(0), Fraction(0)]
        for _ in range(54)
    ]
    for index, value in witness.entries:
        i, j, _a, _b = variable_key(4, 3, index)
        direction, sign = _GAUGE_EDGE_DATA[(i, j)]
        matrix[index][direction] = sign * value
    return tuple(tuple(row) for row in matrix)


def gauge_action_preserves_all_monomials(
    system: SparsePolynomialSystem,
) -> bool:
    """Check that every term has weight zero for all three torus factors."""

    if system.n != 4:
        raise KrennDeformationError(
            "the complementary-edge gauge replay requires n=4"
        )
    for monomial in system.monomial_variable_indices:
        balances = [0, 0, 0]
        for index in monomial:
            i, j, _a, _b = variable_key(
                system.n, system.d, index
            )
            direction, sign = _GAUGE_EDGE_DATA[(i, j)]
            balances[direction] += sign
        if balances != [0, 0, 0]:
            return False
    return True


def rescale_complementary_edges(
    witness: SparseWitness,
    scales: Sequence[Fraction | int],
) -> SparseWitness:
    """Apply the exact three-torus action whose tangent is the gauge."""

    if (witness.n, witness.d) != (4, 3):
        raise KrennDeformationError(
            "complementary-edge rescaling is defined here for n=4,d=3"
        )
    scales = tuple(map(Fraction, scales))
    if len(scales) != 3 or any(scale == 0 for scale in scales):
        raise KrennDeformationError(
            "three nonzero reciprocal scales are required"
        )
    result = {}
    for index, value in witness.entries:
        i, j, _a, _b = variable_key(4, 3, index)
        direction, sign = _GAUGE_EDGE_DATA[(i, j)]
        factor = (
            scales[direction]
            if sign == 1
            else Fraction(1, 1) / scales[direction]
        )
        result[index] = value * factor
    return SparseWitness.from_index_values(4, 3, result)


def _matrix_mod(
    matrix: Sequence[Sequence[Fraction | int]], modulus: int = 31
) -> np.ndarray:
    return np.asarray(
        [
            [fraction_mod(value, modulus) for value in row]
            for row in matrix
        ],
        dtype=np.int64,
    )


def _rank_mod(matrix: np.ndarray, modulus: int = 31) -> int:
    """Deterministic Gaussian rank over the indicated prime field."""

    work = np.asarray(matrix, dtype=np.int64).copy() % modulus
    pivot_row = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            work[[pivot_row, selected]] = work[[selected, pivot_row]]
        inverse = pow(int(work[pivot_row, column]), -1, modulus)
        work[pivot_row] = work[pivot_row] * inverse % modulus
        factors = work[:, column].copy()
        factors[pivot_row] = 0
        work = (
            work - factors[:, np.newaxis] * work[pivot_row]
        ) % modulus
        pivot_row += 1
        if pivot_row == work.shape[0]:
            break
    return pivot_row


def _transpose_over_q(
    matrix: Sequence[Sequence[Fraction | int]],
) -> tuple[tuple[Fraction, ...], ...]:
    rows = tuple(tuple(map(Fraction, row)) for row in matrix)
    if not rows:
        return ()
    if any(len(row) != len(rows[0]) for row in rows):
        raise KrennDeformationError("rational matrix is ragged")
    return tuple(
        tuple(rows[row][column] for row in range(len(rows)))
        for column in range(len(rows[0]))
    )


def _inverse_over_q(
    matrix: Sequence[Sequence[Fraction | int]],
) -> tuple[tuple[Fraction, ...], ...]:
    rows = [list(map(Fraction, row)) for row in matrix]
    size = len(rows)
    if not size or any(len(row) != size for row in rows):
        raise KrennDeformationError(
            "rational inverse requires a nonempty square matrix"
        )
    augmented = [
        [
            *row,
            *(
                Fraction(int(column == identity_column))
                for identity_column in range(size)
            ),
        ]
        for column, row in enumerate(rows)
    ]
    for column in range(size):
        selected = next(
            (
                row
                for row in range(column, size)
                if augmented[row][column]
            ),
            None,
        )
        if selected is None:
            raise KrennDeformationError(
                "rational quotient pivot block is singular"
            )
        if selected != column:
            augmented[column], augmented[selected] = (
                augmented[selected],
                augmented[column],
            )
        pivot = augmented[column][column]
        augmented[column] = [
            value / pivot for value in augmented[column]
        ]
        for row in range(size):
            if row == column or not augmented[row][column]:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return tuple(tuple(row[size:]) for row in augmented)


@dataclass(frozen=True)
class RationalQuotientFrame:
    """An explicit split presentation of ``Q^m / image(gauge)``."""

    pivot_rows: tuple[int, ...]
    free_rows: tuple[int, ...]
    quotient_map: tuple[tuple[Fraction, ...], ...]
    section: tuple[tuple[Fraction, ...], ...]

    @property
    def ambient_dimension(self) -> int:
        return len(self.pivot_rows) + len(self.free_rows)

    @property
    def quotient_dimension(self) -> int:
        return len(self.free_rows)


def build_rational_quotient_frame(
    gauge: Sequence[Sequence[Fraction | int]],
) -> RationalQuotientFrame:
    """Construct a deterministic exact-Q quotient by a full-rank image."""

    gauge = tuple(tuple(map(Fraction, row)) for row in gauge)
    if not gauge or not gauge[0]:
        raise KrennDeformationError("gauge matrix must be nonempty")
    ambient_dimension = len(gauge)
    gauge_dimension = len(gauge[0])
    if any(len(row) != gauge_dimension for row in gauge):
        raise KrennDeformationError("gauge matrix is ragged")
    _reduced, pivot_rows = rref_over_q(
        _transpose_over_q(gauge)
    )
    if len(pivot_rows) != gauge_dimension:
        raise KrennDeformationError("gauge matrix is not injective")
    pivot_row_set = set(pivot_rows)
    free_rows = tuple(
        row
        for row in range(ambient_dimension)
        if row not in pivot_row_set
    )
    pivot_block = tuple(
        tuple(gauge[row][column] for column in range(gauge_dimension))
        for row in pivot_rows
    )
    pivot_inverse = _inverse_over_q(pivot_block)

    quotient_map = [
        [Fraction(0) for _ in range(ambient_dimension)]
        for _ in free_rows
    ]
    for quotient_row, ambient_row in enumerate(free_rows):
        quotient_map[quotient_row][ambient_row] = Fraction(1)
        for pivot_column, pivot_row in enumerate(pivot_rows):
            quotient_map[quotient_row][pivot_row] = -sum(
                gauge[ambient_row][gauge_column]
                * pivot_inverse[gauge_column][pivot_column]
                for gauge_column in range(gauge_dimension)
            )

    section = [
        [Fraction(0) for _ in free_rows]
        for _ in range(ambient_dimension)
    ]
    for quotient_column, ambient_row in enumerate(free_rows):
        section[ambient_row][quotient_column] = Fraction(1)
    return RationalQuotientFrame(
        pivot_rows=pivot_rows,
        free_rows=free_rows,
        quotient_map=tuple(tuple(row) for row in quotient_map),
        section=tuple(tuple(row) for row in section),
    )


def _is_zero_over_q(
    matrix: Sequence[Sequence[Fraction | int]],
) -> bool:
    return all(Fraction(value) == 0 for row in matrix for value in row)


def _is_identity_over_q(
    matrix: Sequence[Sequence[Fraction | int]],
) -> bool:
    rows = tuple(tuple(map(Fraction, row)) for row in matrix)
    return bool(rows) and len(rows) == len(rows[0]) and all(
        rows[row][column] == int(row == column)
        for row in range(len(rows))
        for column in range(len(rows))
    )


@dataclass(frozen=True)
class DeformationCertificate:
    payload: Mapping

    @property
    def exact(self) -> bool:
        checks = self.payload["exact_checks"]
        return all(bool(value) for value in checks.values())

    def to_dict(self) -> dict:
        return dict(self.payload)


def certify_n4_d3_deformation(
    system: SparsePolynomialSystem,
    witness: SparseWitness,
) -> DeformationCertificate:
    """Certify the sharp gauge quotient natively over ``Q``."""

    if (system.n, system.d) != (4, 3) or (
        witness.n,
        witness.d,
    ) != (4, 3):
        raise KrennDeformationError(
            "the sharp deformation regression requires n=4,d=3"
        )
    witness_q, witness_f31 = require_exact_witness(system, witness)
    global_gauge_action = gauge_action_preserves_all_monomials(system)
    jacobian_q = jacobian_matrix(system, witness)
    gauge_q = reciprocal_gauge_matrix(witness)
    jacobian_rank_q = rank_over_q(jacobian_q)
    gauge_rank_q = rank_over_q(gauge_q)
    jacobian_times_gauge_q = _multiply_over_q(
        jacobian_q, gauge_q
    )
    q_complex = all(
        value == 0
        for row in jacobian_times_gauge_q
        for value in row
    )
    kernel_dimension_q = system.variable_count - jacobian_rank_q
    kernel_equals_gauge_q = (
        q_complex
        and gauge_rank_q == kernel_dimension_q == 3
    )

    quotient_frame = build_rational_quotient_frame(gauge_q)
    quotient_kills_gauge = _multiply_over_q(
        quotient_frame.quotient_map, gauge_q
    )
    quotient_section_identity = _multiply_over_q(
        quotient_frame.quotient_map, quotient_frame.section
    )
    quotient_jacobian_q = _multiply_over_q(
        jacobian_q, quotient_frame.section
    )
    quotient_factorization = _multiply_over_q(
        quotient_jacobian_q, quotient_frame.quotient_map
    )
    quotient_factorization_exact = (
        quotient_factorization == jacobian_q
    )
    quotient_rank_q = rank_over_q(quotient_jacobian_q)
    quotient_nullity_q = (
        quotient_frame.quotient_dimension - quotient_rank_q
    )
    quotient_kernel_equals_gauge = (
        _is_zero_over_q(quotient_kills_gauge)
        and _is_identity_over_q(quotient_section_identity)
        and gauge_rank_q == 3
        and quotient_frame.ambient_dimension == 54
        and quotient_frame.quotient_dimension == 51
    )

    # This finite-field block is a supplemental arithmetic regression.  None
    # of the characteristic-zero proof checks below depend on it.
    jacobian_f31 = _matrix_mod(jacobian_q)
    gauge_f31 = _matrix_mod(gauge_q)
    jacobian_rank_f31 = _rank_mod(jacobian_f31)
    gauge_rank_f31 = _rank_mod(gauge_f31)
    f31_complex = not np.any(jacobian_f31 @ gauge_f31 % 31)
    kernel_dimension_f31 = (
        system.variable_count - jacobian_rank_f31
    )
    kernel_equals_gauge_f31 = (
        f31_complex
        and gauge_rank_f31 == kernel_dimension_f31 == 3
    )

    exact_checks = {
        "witness_exact_over_Q": witness_q.satisfied,
        "gauge_action_preserves_every_system_monomial": (
            global_gauge_action
        ),
        "jacobian_rank_51_over_Q": jacobian_rank_q == 51,
        "jacobian_nullity_3_over_Q": kernel_dimension_q == 3,
        "gauge_rank_3_over_Q": gauge_rank_q == 3,
        "jacobian_kills_gauge_over_Q": q_complex,
        "raw_kernel_equals_gauge_over_Q": kernel_equals_gauge_q,
        "native_quotient_map_kills_gauge_over_Q": (
            _is_zero_over_q(quotient_kills_gauge)
        ),
        "native_quotient_section_identity_over_Q": (
            _is_identity_over_q(quotient_section_identity)
        ),
        "native_quotient_kernel_equals_gauge_over_Q": (
            quotient_kernel_equals_gauge
        ),
        "native_quotient_jacobian_factorization_over_Q": (
            quotient_factorization_exact
        ),
        "native_quotient_jacobian_injective_over_Q": (
            quotient_rank_q == 51 and quotient_nullity_q == 0
        ),
    }
    f31_checks = {
        "witness_exact_over_F31": witness_f31.satisfied,
        "jacobian_rank_51_over_F31": jacobian_rank_f31 == 51,
        "jacobian_nullity_3_over_F31": kernel_dimension_f31 == 3,
        "gauge_rank_3_over_F31": gauge_rank_f31 == 3,
        "jacobian_kills_gauge_over_F31": f31_complex,
        "raw_kernel_equals_gauge_over_F31": (
            kernel_equals_gauge_f31
        ),
    }
    payload = {
        "schema": DEFORMATION_SCHEMA,
        "parameters": {"n": 4, "d": 3, "prime": 31},
        "jacobian": {
            "shape": [81, 54],
            "rank_over_Q": jacobian_rank_q,
            "nullity_over_Q": kernel_dimension_q,
        },
        "reciprocal_edge_rescaling_gauge": {
            "shape": [54, 3],
            "rank_over_Q": gauge_rank_q,
            "directions": [
                {
                    "positive_edge": [0, 1],
                    "negative_edge": [2, 3],
                },
                {
                    "positive_edge": [0, 2],
                    "negative_edge": [1, 3],
                },
                {
                    "positive_edge": [0, 3],
                    "negative_edge": [1, 2],
                },
            ],
            "action": (
                "all color slots on the positive edge scale by lambda; "
                "all slots on its complementary negative edge scale by "
                "lambda^-1"
            ),
        },
        "native_gauge_quotient_over_Q": {
            "construction": "deterministic exact pivot-row complement",
            "ambient_dimension": quotient_frame.ambient_dimension,
            "gauge_image_dimension": gauge_rank_q,
            "quotient_dimension": quotient_frame.quotient_dimension,
            "pivot_rows": list(quotient_frame.pivot_rows),
            "quotient_map_shape": [
                len(quotient_frame.quotient_map),
                len(quotient_frame.quotient_map[0]),
            ],
            "section_shape": [
                len(quotient_frame.section),
                len(quotient_frame.section[0]),
            ],
            "induced_jacobian_shape": list(
                (len(quotient_jacobian_q), len(quotient_jacobian_q[0]))
            ),
            "induced_jacobian_rank": quotient_rank_q,
            "induced_jacobian_nullity": quotient_nullity_q,
        },
        "optional_arithmetic_regression_F31": {
            "proof_role": (
                "supplemental arithmetic regression; not used in the "
                "characteristic-zero quotient proof"
            ),
            "jacobian_rank": jacobian_rank_f31,
            "jacobian_nullity": kernel_dimension_f31,
            "gauge_rank": gauge_rank_f31,
            "checks": f31_checks,
            "passed": all(f31_checks.values()),
        },
        "exact_checks": exact_checks,
        "claim_boundary": {
            "characteristic_zero": (
                "The native exact-Q split quotient proves that the three "
                "gauge directions are the full Jacobian kernel and that the "
                "induced 51-dimensional Jacobian is injective; this persists "
                "after scalar extension to C."
            ),
            "finite_field": (
                "The standalone F_31 calculation is an optional arithmetic "
                "regression and is not used as a proof over C."
            ),
        },
    }
    certificate = DeformationCertificate(payload)
    if not certificate.exact:
        failed = [
            key for key, value in exact_checks.items() if not value
        ]
        raise KrennDeformationError(
            f"deformation certificate failed: {failed}"
        )
    return certificate
