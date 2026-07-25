r"""Anchor-free variable projection for the ``n=6,d=3`` Krenn system.

This staged module is deliberately numerical and fail-closed.  It reuses
the repository's exact apex-star factorization, pivot charts, and
direct-GHZ color-gauge character matrix, but it does not promote a small
floating-point residual to an exact witness.

For apex zero, the 135 weights split into 90 nonstar variables ``U`` and
45 star variables ``Y``.  The exact equations are

``A(U) Y = E``,

where ``A`` is ``243 x 15`` and is shared by the three apex colors.  This
module provides:

* raw SVD and smooth ridge variable projection;
* an analytic pullback gradient through the quadratic entries of ``A``;
* the three exact pivot-orbit representatives;
* a root-free retraction to the three-factor slice ``p_0=p_1=p_2=1``;
* an anchor-free horizontal projection

  ``ker(dp) intersect (diag(U) H ker(C_p))^perp``;

* an exact recession audit that refuses residual real-torus norm balancing;
* the ``B Y = I_3`` chart normal-form diagnostic; and
* exact-character audits for the optional determinant-two anchor chart and
  the known pole quantity.

No raw ``U`` norm is a mathematical search cap.  Raw norms are diagnostics
and only non-finite/overflow guards are enforced.  Bounds on recovered
``Y`` or ``Z`` are coordinates on the selected pivot slice and are not
gauge-invariant theorems.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
import math
import os
from typing import Mapping, Sequence


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
)
from experiments.krenn_quantum_graph.star_linearization import (
    star_linearization,
)
from experiments.krenn_quantum_graph.star_pivot_charts import (
    star_pivot_charts,
)
from experiments.krenn_quantum_graph.star_pivot_gauge import (
    star_pivot_gauge_audit,
)
from experiments.krenn_quantum_graph.system import (
    generate_sparse_system,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    n6_d3_seed_witness,
)
from experiments.krenn_quantum_graph.vertical_component import (
    color_diagonal_exponent_matrix,
)


N = 6
D = 3
APEX = 0
AMBIENT_VARIABLES = 135
NONSTAR_VARIABLES = 90
STAR_COLUMNS = 15
TARGETS = 3
ROWS = 243
TARGET_ROWS = (0, 121, 242)
GAUGE_DIMENSION = 15


def _safe_l2(values: np.ndarray | Sequence[complex]) -> float:
    """Return a scale-safe Euclidean/Frobenius norm."""

    array = np.asarray(values)
    if np.iscomplexobj(array):
        components = np.concatenate((
            np.abs(array.real).ravel(),
            np.abs(array.imag).ravel(),
        ))
    else:
        components = np.abs(array).ravel()
    if not components.size:
        return 0.0
    maximum = float(np.max(components))
    if maximum == 0.0:
        return 0.0
    if not math.isfinite(maximum):
        return math.inf
    scaled_squared = float(
        np.sum((components / maximum) ** 2, dtype=np.float64)
    )
    result = maximum * math.sqrt(scaled_squared)
    return result if math.isfinite(result) else math.inf
VARIABLE_PROJECTION_SCHEMA = (
    "krenn-n6-d3-anchor-free-star-variable-projection-v1"
)

# This is a floating-point overflow guard, not a mathematical search bound.
NUMERICAL_OVERFLOW_GUARD = 1.0e150


class KrennStarVariableProjectionError(RuntimeError):
    """An exact identity or fail-closed numerical guard failed."""


class KrennResidualGaugeBalanceError(
    KrennStarVariableProjectionError
):
    """The residual real-torus orbit did not admit a certified balance."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _finite_complex_array(
    values: Sequence[complex],
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if (
        array.shape != shape
        or not np.all(np.isfinite(array.real))
        or not np.all(np.isfinite(array.imag))
        or (
            array.size
            and float(np.max(np.abs(array)))
            > NUMERICAL_OVERFLOW_GUARD
        )
    ):
        raise KrennStarVariableProjectionError(
            f"{label} is malformed, non-finite, or beyond the "
            "floating-point overflow guard"
        )
    return array


def _u_array(values: Sequence[complex]) -> np.ndarray:
    return _finite_complex_array(
        values, shape=(NONSTAR_VARIABLES,), label="nonstar vector"
    )


def _y_array(values: Sequence[Sequence[complex]]) -> np.ndarray:
    return _finite_complex_array(
        values, shape=(STAR_COLUMNS, TARGETS), label="star matrix"
    )


def _squared_norm(values: np.ndarray) -> float:
    return float(np.vdot(values, values).real)


def _matrix_rank_over_q(
    matrix: Sequence[Sequence[int | Fraction]],
) -> int:
    rows = [
        [Fraction(value) for value in row]
        for row in matrix
    ]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennStarVariableProjectionError(
            "an exact matrix is ragged"
        )
    rank = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        value = rows[rank][column]
        rows[rank] = [entry / value for entry in rows[rank]]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                left - factor * right
                for left, right in zip(
                    rows[row], rows[rank], strict=True
                )
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _determinant_over_q(
    matrix: Sequence[Sequence[int | Fraction]],
) -> Fraction:
    rows = [
        [Fraction(value) for value in row]
        for row in matrix
    ]
    size = len(rows)
    if not rows or any(len(row) != size for row in rows):
        raise KrennStarVariableProjectionError(
            "an exact determinant needs a nonempty square matrix"
        )
    determinant = Fraction(1)
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        value = rows[column][column]
        determinant *= value
        rows[column] = [entry / value for entry in rows[column]]
        for row in range(column + 1, size):
            if not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                left - factor * right
                for left, right in zip(
                    rows[row], rows[column], strict=True
                )
            ]
    return determinant


def _rational_nullspace(
    matrix: Sequence[Sequence[int | Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    """Return a row-major exact basis matrix for the right kernel."""

    rows = [
        [Fraction(value) for value in row]
        for row in matrix
    ]
    if not rows:
        raise KrennStarVariableProjectionError(
            "the kernel matrix must have at least one row"
        )
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennStarVariableProjectionError(
            "the kernel matrix is ragged"
        )
    rank = 0
    pivot_columns: list[int] = []
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        value = rows[rank][column]
        rows[rank] = [entry / value for entry in rows[rank]]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                left - factor * right
                for left, right in zip(
                    rows[row], rows[rank], strict=True
                )
            ]
        pivot_columns.append(column)
        rank += 1
        if rank == len(rows):
            break
    free_columns = tuple(
        column
        for column in range(width)
        if column not in set(pivot_columns)
    )
    columns = []
    for free in free_columns:
        vector = [Fraction(0)] * width
        vector[free] = Fraction(1)
        for pivot_row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = -rows[pivot_row][free]
        columns.append(tuple(vector))
    result = tuple(
        tuple(columns[column][row] for column in range(len(columns)))
        for row in range(width)
    )
    if (
        len(result) != width
        or any(len(row) != width - rank for row in result)
    ):
        raise KrennStarVariableProjectionError(
            "the exact kernel basis changed dimension"
        )
    for source_row in matrix:
        for basis_column in range(width - rank):
            if sum(
                Fraction(source_row[index])
                * result[index][basis_column]
                for index in range(width)
            ):
                raise KrennStarVariableProjectionError(
                    "the exact kernel basis failed replay"
                )
    return result


@lru_cache(maxsize=1)
def nonstar_global_indices() -> tuple[int, ...]:
    indices = tuple(
        index
        for index in range(AMBIENT_VARIABLES)
        if APEX not in variable_key(N, D, index)[:2]
    )
    if len(indices) != NONSTAR_VARIABLES:
        raise KrennStarVariableProjectionError(
            "the apex-zero nonstar census changed"
        )
    return indices


@lru_cache(maxsize=1)
def _global_to_local() -> dict[int, int]:
    return {
        global_index: local_index
        for local_index, global_index in enumerate(
            nonstar_global_indices()
        )
    }


@lru_cache(maxsize=1)
def nonstar_character_matrix() -> tuple[tuple[int, ...], ...]:
    full = color_diagonal_exponent_matrix()
    rows = tuple(
        tuple(map(int, full[index]))
        for index in nonstar_global_indices()
    )
    if (
        len(rows) != NONSTAR_VARIABLES
        or any(len(row) != GAUGE_DIMENSION for row in rows)
        or _matrix_rank_over_q(rows) != GAUGE_DIMENSION
    ):
        raise KrennStarVariableProjectionError(
            "the nonstar gauge-character matrix changed"
        )
    return rows


@dataclass(frozen=True)
class PivotRepresentative:
    orbit_index: int
    equality_pattern: str
    partner_by_color: tuple[int, int, int]
    orbit_size: int
    columns_by_color: tuple[int, int, int]
    pivot_terms_local: tuple[
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
    ]
    factor_characters: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    normalization_cocharacters: tuple[tuple[int, int, int], ...]
    normalization_vertices: tuple[int, int, int]

    def to_dict(self) -> dict:
        return {
            "orbit_index": self.orbit_index,
            "equality_pattern": self.equality_pattern,
            "partner_by_color": list(self.partner_by_color),
            "orbit_size": self.orbit_size,
            "columns_by_color": list(self.columns_by_color),
            "factor_characters": [
                list(row) for row in self.factor_characters
            ],
            "normalization_cocharacters": [
                list(row) for row in self.normalization_cocharacters
            ],
            "normalization_vertices": list(
                self.normalization_vertices
            ),
        }


@lru_cache(maxsize=1)
def pivot_representatives() -> tuple[PivotRepresentative, ...]:
    expected = (
        ("all-same", (1, 1, 1), 5),
        ("exactly-two-same", (1, 1, 2), 60),
        ("all-distinct", (1, 2, 3), 60),
    )
    factorization = star_linearization(APEX)
    charts = star_pivot_charts(APEX)
    full_characters = color_diagonal_exponent_matrix()
    global_to_local = _global_to_local()
    result = []
    for orbit_index, (pattern, partners, orbit_size) in enumerate(
        expected
    ):
        chart = next(
            (
                candidate
                for candidate in charts
                if candidate.partner_by_color == partners
            ),
            None,
        )
        if chart is None:
            raise KrennStarVariableProjectionError(
                "a pivot representative disappeared"
            )
        local_terms = tuple(
            tuple(
                tuple(global_to_local[index] for index in monomial)
                for monomial in polynomial
            )
            for polynomial in chart.quadratic_factors
        )
        factor_characters = []
        for polynomial in chart.quadratic_factors:
            term_characters = tuple(
                tuple(
                    sum(
                        full_characters[index][parameter]
                        for index in monomial
                    )
                    for parameter in range(GAUGE_DIMENSION)
                )
                for monomial in polynomial
            )
            if (
                len(term_characters) != 3
                or any(
                    row != term_characters[0]
                    for row in term_characters[1:]
                )
            ):
                raise KrennStarVariableProjectionError(
                    "a pivot factor lost its semi-invariant character"
                )
            factor_characters.append(term_characters[0])

        cocharacters = [
            [0] * TARGETS for _parameter in range(GAUGE_DIMENSION)
        ]
        selected_vertices = []
        for color, partner in enumerate(partners):
            residual_vertex = min(
                vertex
                for vertex in range(1, N)
                if vertex != partner and vertex < N - 1
            )
            selected_vertices.append(residual_vertex)
            cocharacters[APEX * D + color][color] = 1
            cocharacters[residual_vertex * D + color][color] = -1
        pairings = tuple(
            tuple(
                sum(
                    factor_characters[row][parameter]
                    * cocharacters[parameter][column]
                    for parameter in range(GAUGE_DIMENSION)
                )
                for column in range(TARGETS)
            )
            for row in range(TARGETS)
        )
        if pairings != (
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        ):
            raise KrennStarVariableProjectionError(
                "the root-free normalization cocharacters changed"
            )
        if chart.columns_by_color != tuple(
            factorization.columns.index((partners[color], color))
            for color in range(D)
        ):
            raise KrennStarVariableProjectionError(
                "a representative pivot column changed"
            )
        result.append(PivotRepresentative(
            orbit_index=orbit_index,
            equality_pattern=pattern,
            partner_by_color=partners,
            orbit_size=orbit_size,
            columns_by_color=chart.columns_by_color,
            pivot_terms_local=local_terms,
            factor_characters=tuple(factor_characters),
            normalization_cocharacters=tuple(
                tuple(row) for row in cocharacters
            ),
            normalization_vertices=tuple(selected_vertices),
        ))
    return tuple(result)


def pivot_representative(
    representative: int | PivotRepresentative,
) -> PivotRepresentative:
    if isinstance(representative, PivotRepresentative):
        expected = pivot_representatives()[representative.orbit_index]
        if representative != expected:
            raise KrennStarVariableProjectionError(
                "an altered pivot representative was supplied"
            )
        return representative
    if isinstance(representative, bool):
        raise KrennStarVariableProjectionError(
            "a pivot orbit index must be an integer"
        )
    index = int(representative)
    if not 0 <= index < len(pivot_representatives()):
        raise KrennStarVariableProjectionError(
            "a pivot orbit index is outside 0..2"
        )
    return pivot_representatives()[index]


def pivot_characters(
    representative: int | PivotRepresentative,
) -> tuple[tuple[int, ...], ...]:
    return pivot_representative(representative).factor_characters


@lru_cache(maxsize=3)
def _gauge_kernel_cached(
    orbit_index: int,
) -> tuple[tuple[Fraction, ...], ...]:
    representative = pivot_representative(orbit_index)
    result = _rational_nullspace(
        representative.factor_characters
    )
    if (
        len(result) != GAUGE_DIMENSION
        or any(len(row) != 12 for row in result)
    ):
        raise KrennStarVariableProjectionError(
            "the residual gauge kernel changed from 12 dimensions"
        )
    return result


def gauge_kernel(
    representative: int | PivotRepresentative,
) -> tuple[tuple[Fraction, ...], ...]:
    return _gauge_kernel_cached(
        pivot_representative(representative).orbit_index
    )


def normalization_cocharacters(
    representative: int | PivotRepresentative,
) -> tuple[tuple[int, int, int], ...]:
    return pivot_representative(
        representative
    ).normalization_cocharacters


@lru_cache(maxsize=1)
def _matrix_term_incidence() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    factorization = star_linearization(APEX)
    global_to_local = _global_to_local()
    rows: list[int] = []
    columns: list[int] = []
    left: list[int] = []
    right: list[int] = []
    for row, matrix_row in enumerate(factorization.entries):
        for column, polynomial in enumerate(matrix_row):
            for first, second in polynomial:
                rows.append(row)
                columns.append(column)
                left.append(global_to_local[first])
                right.append(global_to_local[second])
    if len(rows) != 3_645:
        raise KrennStarVariableProjectionError(
            "the star matrix term census changed"
        )
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
    )


@lru_cache(maxsize=1)
def _target_matrix() -> np.ndarray:
    result = np.zeros((ROWS, TARGETS), dtype=np.complex128)
    for color, row in enumerate(TARGET_ROWS):
        result[row, color] = 1.0
    result.setflags(write=False)
    return result


def targets() -> np.ndarray:
    """Return the three shared residual target vectors."""

    return _target_matrix().copy()


def evaluate_star_matrix(values: Sequence[complex]) -> np.ndarray:
    """Evaluate the exact quadratic ``243 x 15`` star matrix."""

    u = _u_array(values)
    rows, columns, left, right = _matrix_term_incidence()
    matrix = np.zeros((ROWS, STAR_COLUMNS), dtype=np.complex128)
    np.add.at(
        matrix,
        (rows, columns),
        u[left] * u[right],
    )
    if not np.all(np.isfinite(matrix)):
        raise KrennStarVariableProjectionError(
            "star-matrix evaluation overflowed"
        )
    return matrix


def pivot_values(
    values: Sequence[complex],
    representative: int | PivotRepresentative,
) -> np.ndarray:
    u = _u_array(values)
    rep = pivot_representative(representative)
    result = np.zeros(TARGETS, dtype=np.complex128)
    for color, polynomial in enumerate(rep.pivot_terms_local):
        result[color] = sum(
            u[left] * u[right] for left, right in polynomial
        )
    return result


def pivot_jacobian(
    values: Sequence[complex],
    representative: int | PivotRepresentative,
) -> np.ndarray:
    u = _u_array(values)
    rep = pivot_representative(representative)
    jacobian = np.zeros(
        (TARGETS, NONSTAR_VARIABLES), dtype=np.complex128
    )
    for color, polynomial in enumerate(rep.pivot_terms_local):
        for left, right in polynomial:
            jacobian[color, left] += u[right]
            jacobian[color, right] += u[left]
    return jacobian


def retract_pivots(
    values: Sequence[complex],
    representative: int | PivotRepresentative,
    *,
    minimum_abs: float = 1.0e-12,
    replay_tolerance: float = 2.0e-8,
) -> tuple[np.ndarray, dict]:
    """Retract to ``p_0=p_1=p_2=1`` using integer cocharacters.

    Only integer powers and inverses of the three nonzero pivot factors
    are used.  No root, logarithm, coordinate anchor, saturation, or
    Rabinowitsch variable enters this normalization.
    """

    u = _u_array(values)
    rep = pivot_representative(representative)
    if (
        not np.isfinite(minimum_abs)
        or minimum_abs <= 0
        or not np.isfinite(replay_tolerance)
        or replay_tolerance <= 0
    ):
        raise KrennStarVariableProjectionError(
            "pivot-retraction tolerances are invalid"
        )
    before = pivot_values(u, rep)
    if np.any(np.abs(before) < minimum_abs):
        raise KrennStarVariableProjectionError(
            "a proposed step approached the boundary of its pivot open"
        )
    character_matrix = np.asarray(
        nonstar_character_matrix(), dtype=np.int64
    )
    cocharacters = np.asarray(
        rep.normalization_cocharacters, dtype=np.int64
    )
    exponents = character_matrix @ cocharacters
    multiplier = np.ones(NONSTAR_VARIABLES, dtype=np.complex128)
    for color in range(TARGETS):
        multiplier *= np.power(before[color], exponents[:, color])
    result = u * multiplier
    result = _u_array(result)
    after = pivot_values(result, rep)
    error = float(np.max(np.abs(after - 1.0)))
    if error > replay_tolerance:
        raise KrennStarVariableProjectionError(
            "the root-free pivot retraction failed numerical replay"
        )
    difference = result - u
    diagnostics = {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "orbit_index": rep.orbit_index,
        "pivot_values_before": [
            [float(value.real), float(value.imag)]
            for value in before
        ],
        "pivot_values_after": [
            [float(value.real), float(value.imag)]
            for value in after
        ],
        "maximum_pivot_error_after": error,
        "minimum_pivot_abs_before": float(
            np.min(np.abs(before))
        ),
        "maximum_multiplier_abs": float(
            np.max(np.abs(multiplier))
        ),
        "minimum_nonzero_multiplier_abs": float(
            np.min(np.abs(multiplier))
        ),
        "raw_to_retracted_l2": _safe_l2(difference),
        "raw_to_retracted_linf": float(
            np.max(np.abs(difference))
        ),
        "root_extraction_used": False,
        "coordinate_anchor_used": False,
        "mathematical_raw_norm_cap_used": False,
        "numerical_nonproof": True,
    }
    return result, diagnostics


def initial_point(
    representative: int | PivotRepresentative,
    seed: int,
) -> np.ndarray:
    """Return one deterministic dense, pivot-normalized starting point."""

    rep = pivot_representative(representative)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise KrennStarVariableProjectionError(
            "the deterministic seed must be an integer"
        )
    rng = np.random.default_rng(seed)
    for _attempt in range(64):
        log_magnitudes = rng.uniform(
            -0.5, 0.5, size=NONSTAR_VARIABLES
        )
        phases = rng.uniform(
            -math.pi, math.pi, size=NONSTAR_VARIABLES
        )
        candidate = np.exp(log_magnitudes + 1j * phases)
        try:
            normalized, _diagnostics = retract_pivots(
                candidate, rep, minimum_abs=1.0e-9
            )
        except KrennStarVariableProjectionError:
            continue
        return normalized
    raise KrennStarVariableProjectionError(
        "deterministic initialization failed to enter the pivot open"
    )


def natural_repair_initial_point(
    representative: int | PivotRepresentative,
    seed: int,
    *,
    perturbation_scale: float = 1.0e-2,
) -> np.ndarray:
    """Activate every missing nonstar coordinate around the natural seed.

    The natural support lies in the all-distinct pivot orbit.  Keeping its
    six nonstar coordinates exact while perturbing all 84 zero nonstar
    coordinates points the numerical search away from the known formal
    branch without assuming any repair weights are equal.
    """

    rep = pivot_representative(representative)
    if rep.orbit_index != 2:
        raise KrennStarVariableProjectionError(
            "the natural seed lies only in the all-distinct pivot orbit"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise KrennStarVariableProjectionError(
            "the deterministic seed must be an integer"
        )
    try:
        scale = float(perturbation_scale)
    except (TypeError, ValueError) as error:
        raise KrennStarVariableProjectionError(
            "the repair perturbation scale must be positive and finite"
        ) from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise KrennStarVariableProjectionError(
            "the repair perturbation scale must be positive and finite"
        )
    full_weights = np.zeros(
        AMBIENT_VARIABLES, dtype=np.complex128
    )
    for index, value in n6_d3_seed_witness().entries:
        full_weights[int(index)] = complex(value)
    candidate = full_weights[
        np.asarray(nonstar_global_indices(), dtype=np.int64)
    ].copy()
    repair_mask = candidate == 0.0
    if int(np.count_nonzero(~repair_mask)) != 6:
        raise KrennStarVariableProjectionError(
            "the natural nonstar support changed"
        )
    seed_sequence = np.random.SeedSequence(
        [seed, rep.orbit_index, 0x4E415455]
    )
    generator = np.random.default_rng(seed_sequence)
    noise = (
        generator.normal(size=NONSTAR_VARIABLES)
        + 1j * generator.normal(size=NONSTAR_VARIABLES)
    ) / math.sqrt(2.0)
    candidate[repair_mask] = scale * noise[repair_mask]
    if np.any(candidate[repair_mask] == 0.0):
        raise KrennStarVariableProjectionError(
            "a deterministic repair perturbation was exactly zero"
        )
    normalized, _diagnostics = retract_pivots(
        candidate, rep, minimum_abs=1.0e-9
    )
    return normalized


def _svd_rank(
    singular_values: np.ndarray,
    rows: int,
    columns: int,
    rank_rtol: float | None,
) -> tuple[np.ndarray, int, float]:
    if singular_values.ndim != 1:
        raise KrennStarVariableProjectionError(
            "an SVD singular-value vector is malformed"
        )
    leading = (
        float(singular_values[0]) if singular_values.size else 0.0
    )
    if rank_rtol is None:
        tolerance = float(
            max(rows, columns)
            * np.finfo(np.float64).eps
            * max(leading, 1.0)
        )
    else:
        if not np.isfinite(rank_rtol) or rank_rtol <= 0:
            raise KrennStarVariableProjectionError(
                "an SVD rank tolerance is invalid"
            )
        tolerance = float(
            float(rank_rtol) * max(leading, 1.0)
        )
    active = singular_values > tolerance
    return active, int(np.count_nonzero(active)), tolerance


def pullback_gradient(
    values: Sequence[complex],
    y_values: Sequence[Sequence[complex]],
    residual_values: Sequence[Sequence[complex]],
) -> np.ndarray:
    r"""Pull a matrix least-squares gradient back to the 90 ``U`` values.

    The returned complex vector ``g`` uses the convention

    ``df = 2 Re(vdot(g, dU))``.
    """

    u = _u_array(values)
    y = _y_array(y_values)
    residual = _finite_complex_array(
        residual_values,
        shape=(ROWS, TARGETS),
        label="variable-projection residual",
    )
    matrix_gradient = residual @ y.conj().T
    rows, columns, left, right = _matrix_term_incidence()
    coefficients = np.conj(matrix_gradient[rows, columns])
    holomorphic_coefficients = np.zeros(
        NONSTAR_VARIABLES, dtype=np.complex128
    )
    np.add.at(
        holomorphic_coefficients,
        left,
        coefficients * u[right],
    )
    np.add.at(
        holomorphic_coefficients,
        right,
        coefficients * u[left],
    )
    # If s_j is the holomorphic coefficient in
    # df=2 Re(sum_j s_j dU_j), then g_j=conj(s_j) gives the stated vdot
    # convention.
    result = np.conj(holomorphic_coefficients)
    if not np.all(np.isfinite(result)):
        raise KrennStarVariableProjectionError(
            "the analytic pullback gradient overflowed"
        )
    return result


@dataclass(frozen=True)
class ProjectionResult:
    mode: str
    objective: float
    residual_squared: float
    regularization_penalty: float
    y: np.ndarray
    residual: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int
    rank_tolerance: float
    gradient: np.ndarray | None
    smooth_gradient_certified_for_this_evaluation: bool
    ridge_mu: float | None

    def summary(self) -> dict:
        y_abs = np.abs(self.y)
        residual_abs = np.abs(self.residual)
        smallest_active = (
            float(self.singular_values[self.numerical_rank - 1])
            if self.numerical_rank
            else 0.0
        )
        condition = (
            float(self.singular_values[0] / smallest_active)
            if self.numerical_rank and smallest_active
            else None
        )
        return {
            "schema": VARIABLE_PROJECTION_SCHEMA,
            "mode": self.mode,
            "objective": self.objective,
            "residual_squared": self.residual_squared,
            "residual_l2": math.sqrt(self.residual_squared),
            "residual_max_abs": float(np.max(residual_abs)),
            "regularization_penalty":
                self.regularization_penalty,
            "ridge_mu": self.ridge_mu,
            "numerical_rank": self.numerical_rank,
            "rank_tolerance": self.rank_tolerance,
            "singular_values": [
                float(value) for value in self.singular_values
            ],
            "active_condition_number": condition,
            "active_condition_number_is_finite": (
                condition is not None and math.isfinite(condition)
            ),
            "y_l2": _safe_l2(self.y),
            "y_linf": float(np.max(y_abs)),
            "y_column_l2": [
                _safe_l2(self.y[:, color])
                for color in range(TARGETS)
            ],
            "smooth_gradient_certified_for_this_evaluation":
                self.smooth_gradient_certified_for_this_evaluation,
            "rank_change_or_threshold_is_exact_proof": False,
            "numerical_nonproof": True,
        }


def raw_variable_projection(
    values: Sequence[complex],
    *,
    rank_rtol: float | None = None,
) -> ProjectionResult:
    """Return the raw Moore-Penrose span-distance diagnostic.

    A gradient is returned only when all 15 columns are separated from
    the numerical rank threshold by a factor of at least 100.  At a rank
    drop the objective remains a diagnostic but is not used as a smooth
    optimization objective.
    """

    u = _u_array(values)
    matrix = evaluate_star_matrix(u)
    left, singular_values, right_h = np.linalg.svd(
        matrix, full_matrices=False
    )
    active, rank, tolerance = _svd_rank(
        singular_values, ROWS, STAR_COLUMNS, rank_rtol
    )
    coefficients = left.conj().T @ _target_matrix()
    scaled = np.zeros_like(coefficients)
    if rank:
        scaled[active] = (
            coefficients[active]
            / singular_values[active, np.newaxis]
        )
    y = right_h.conj().T @ scaled
    residual = matrix @ y - _target_matrix()
    residual_squared = _squared_norm(residual)
    smooth = bool(
        rank == STAR_COLUMNS
        and singular_values[-1] > 100.0 * tolerance
    )
    gradient = (
        pullback_gradient(u, y, residual) if smooth else None
    )
    return ProjectionResult(
        mode="raw-svd-span-distance",
        objective=residual_squared,
        residual_squared=residual_squared,
        regularization_penalty=0.0,
        y=y,
        residual=residual,
        singular_values=singular_values,
        numerical_rank=rank,
        rank_tolerance=tolerance,
        gradient=gradient,
        smooth_gradient_certified_for_this_evaluation=smooth,
        ridge_mu=None,
    )


def ridge_variable_projection(
    values: Sequence[complex],
    ridge_mu: float,
    *,
    rank_rtol: float | None = None,
) -> ProjectionResult:
    """Return the smooth ridge variable-projection value and gradient."""

    u = _u_array(values)
    if not np.isfinite(ridge_mu) or ridge_mu <= 0:
        raise KrennStarVariableProjectionError(
            "ridge_mu must be finite and strictly positive"
        )
    mu = float(ridge_mu)
    matrix = evaluate_star_matrix(u)
    left, singular_values, right_h = np.linalg.svd(
        matrix, full_matrices=False
    )
    active, rank, tolerance = _svd_rank(
        singular_values, ROWS, STAR_COLUMNS, rank_rtol
    )
    coefficients = left.conj().T @ _target_matrix()
    filter_values = singular_values / (
        singular_values**2 + mu
    )
    y = right_h.conj().T @ (
        filter_values[:, np.newaxis] * coefficients
    )
    residual = matrix @ y - _target_matrix()
    residual_squared = _squared_norm(residual)
    penalty = mu * _squared_norm(y)
    objective = residual_squared + penalty
    gradient = pullback_gradient(u, y, residual)
    return ProjectionResult(
        mode="smooth-ridge-variable-projection",
        objective=objective,
        residual_squared=residual_squared,
        regularization_penalty=penalty,
        y=y,
        residual=residual,
        singular_values=singular_values,
        numerical_rank=rank,
        rank_tolerance=tolerance,
        gradient=gradient,
        smooth_gradient_certified_for_this_evaluation=True,
        ridge_mu=mu,
    )


def assert_rank_stable(
    previous_rank: int,
    current_rank: int,
    *,
    label: str = "numerical matrix",
) -> None:
    """Fail closed if an optimizer attempts to cross a rank threshold."""

    if (
        isinstance(previous_rank, bool)
        or isinstance(current_rank, bool)
        or int(previous_rank) != int(current_rank)
    ):
        raise KrennStarVariableProjectionError(
            f"{label} changed numerical rank; restart or use the "
            "smooth ridge objective and do not infer an exact claim"
        )


@dataclass(frozen=True)
class HorizontalResult:
    projected_gradient: np.ndarray
    direction: np.ndarray
    constraint_singular_values: np.ndarray
    constraint_rank: int
    constraint_rank_tolerance: float
    tangent_residual_l2: float
    residual_vertical_inner_product_l2: float
    predicted_directional_derivative: float
    full_gauge_tangent_rank: int
    residual_gauge_tangent_rank: int
    jp_times_t_minus_cp_l2: float

    def summary(self) -> dict:
        return {
            "schema": VARIABLE_PROJECTION_SCHEMA,
            "constraint_rank": self.constraint_rank,
            "generic_constraint_rank": 15,
            "constraint_rank_drop": self.constraint_rank < 15,
            "constraint_rank_tolerance":
                self.constraint_rank_tolerance,
            "constraint_singular_values": [
                float(value)
                for value in self.constraint_singular_values
            ],
            "full_gauge_tangent_rank":
                self.full_gauge_tangent_rank,
            "residual_gauge_tangent_rank":
                self.residual_gauge_tangent_rank,
            "tangent_residual_l2": self.tangent_residual_l2,
            "residual_vertical_inner_product_l2":
                self.residual_vertical_inner_product_l2,
            "jp_times_t_minus_cp_l2":
                self.jp_times_t_minus_cp_l2,
            "projected_gradient_l2":
                _safe_l2(self.projected_gradient),
            "predicted_directional_derivative":
                self.predicted_directional_derivative,
            "coordinate_anchor_used": False,
            "rank_drop_is_exact_proof": False,
            "numerical_nonproof": True,
        }


def _numerical_rank_from_matrix(
    matrix: np.ndarray,
    rank_rtol: float | None,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    left, singular_values, right_h = np.linalg.svd(
        matrix, full_matrices=False
    )
    active, rank, tolerance = _svd_rank(
        singular_values,
        matrix.shape[0],
        matrix.shape[1],
        rank_rtol,
    )
    return singular_values, rank, tolerance, right_h[active]


def horizontal_projection(
    values: Sequence[complex],
    gradient_values: Sequence[complex],
    representative: int | PivotRepresentative,
    *,
    rank_rtol: float | None = None,
    pivot_tolerance: float = 2.0e-8,
) -> HorizontalResult:
    r"""Project a ridge gradient to the anchor-free pivot-slice horizontal.

    If ``g`` uses ``df=2 Re(vdot(g,dU))``, the returned direction is
    ``-P_H g`` and has predicted derivative ``-2 ||P_H g||^2``.
    """

    u = _u_array(values)
    gradient = _finite_complex_array(
        gradient_values,
        shape=(NONSTAR_VARIABLES,),
        label="variable-projection gradient",
    )
    rep = pivot_representative(representative)
    pivots = pivot_values(u, rep)
    if float(np.max(np.abs(pivots - 1.0))) > pivot_tolerance:
        raise KrennStarVariableProjectionError(
            "horizontal projection requires the p=1 slice"
        )
    h_matrix = np.asarray(
        nonstar_character_matrix(), dtype=np.complex128
    )
    tangent = u[:, np.newaxis] * h_matrix
    c_p = np.asarray(
        rep.factor_characters, dtype=np.complex128
    )
    jacobian = pivot_jacobian(u, rep)
    relation_error = _safe_l2(
        jacobian @ tangent
        - np.diag(pivots) @ c_p
    )
    kernel = np.asarray(
        [
            [complex(value) for value in row]
            for row in gauge_kernel(rep)
        ],
        dtype=np.complex128,
    )
    residual_vertical = tangent @ kernel
    constraints = np.vstack((
        jacobian,
        residual_vertical.conj().T,
    ))
    singular_values, rank, tolerance, active_right_rows = (
        _numerical_rank_from_matrix(constraints, rank_rtol)
    )
    if rank:
        row_component = active_right_rows.conj().T @ (
            active_right_rows @ gradient
        )
    else:
        row_component = np.zeros_like(gradient)
    projected = gradient - row_component
    direction = -projected
    tangent_residual = _safe_l2(jacobian @ direction)
    vertical_residual = _safe_l2(
        residual_vertical.conj().T @ direction
    )
    _, full_rank, _, _ = _numerical_rank_from_matrix(
        tangent, rank_rtol
    )
    _, residual_rank, _, _ = _numerical_rank_from_matrix(
        residual_vertical, rank_rtol
    )
    predicted = float(
        2.0 * np.vdot(gradient, direction).real
    )
    expected = -2.0 * _squared_norm(projected)
    if not math.isclose(
        predicted,
        expected,
        rel_tol=2.0e-10,
        abs_tol=2.0e-12,
    ):
        raise KrennStarVariableProjectionError(
            "the horizontal descent identity failed"
        )
    return HorizontalResult(
        projected_gradient=projected,
        direction=direction,
        constraint_singular_values=singular_values,
        constraint_rank=rank,
        constraint_rank_tolerance=tolerance,
        tangent_residual_l2=tangent_residual,
        residual_vertical_inner_product_l2=vertical_residual,
        predicted_directional_derivative=predicted,
        full_gauge_tangent_rank=full_rank,
        residual_gauge_tangent_rank=residual_rank,
        jp_times_t_minus_cp_l2=relation_error,
    )


def _logsumexp_balance_state(
    log_weight_squared: np.ndarray,
    action: np.ndarray,
    eta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    logits = log_weight_squared + 2.0 * (action @ eta)
    maximum = float(np.max(logits))
    exponentials = np.exp(logits - maximum)
    total = float(np.sum(exponentials))
    probabilities = exponentials / total
    value = maximum + math.log(total)
    moment = action.T @ probabilities
    gradient = 2.0 * moment
    centered = action - probabilities @ action
    hessian = 4.0 * (
        action.T
        @ (probabilities[:, np.newaxis] * centered)
    )
    hessian = 0.5 * (hessian + hessian.T)
    return value, gradient, hessian, moment


@lru_cache(maxsize=3)
def _residual_recession_certificates_cached(
    orbit_index: int,
) -> tuple[dict, ...]:
    """Return exact recession directions of the residual pivot gauge.

    For target color ``a``, put ``theta_(apex,a)=1`` and
    ``theta_(pivot_partner,a)=-1``.  The selected pivot factor has
    character ``-(theta_apex+theta_partner)``, so this lies in
    ``ker(C_p)``.  On nonstar weights it has exponent ``-1`` exactly on
    coordinates incident to the selected colored partner and zero
    elsewhere.  Thus the residual real-torus L2 norm has an unattained
    infimum whenever any such coordinate is nonzero.
    """

    rep = pivot_representative(orbit_index)
    h_matrix = nonstar_character_matrix()
    rows = []
    for color, partner in enumerate(rep.partner_by_color):
        if partner >= N - 1:
            raise KrennStarVariableProjectionError(
                "the staged pivot representatives unexpectedly use "
                "the eliminated gauge vertex"
            )
        cocharacter = [0] * GAUGE_DIMENSION
        cocharacter[APEX * D + color] = 1
        cocharacter[partner * D + color] = -1
        factor_pairing = tuple(
            sum(
                rep.factor_characters[row][parameter]
                * cocharacter[parameter]
                for parameter in range(GAUGE_DIMENSION)
            )
            for row in range(TARGETS)
        )
        exponents = tuple(
            sum(
                h_matrix[local][parameter]
                * cocharacter[parameter]
                for parameter in range(GAUGE_DIMENSION)
            )
            for local in range(NONSTAR_VARIABLES)
        )
        if (
            factor_pairing != (0, 0, 0)
            or any(value not in (-1, 0) for value in exponents)
            or -1 not in exponents
        ):
            raise KrennStarVariableProjectionError(
                "an exact residual-gauge recession certificate changed"
            )
        affected_local = tuple(
            index for index, value in enumerate(exponents)
            if value == -1
        )
        rows.append({
            "color": color,
            "pivot_partner": partner,
            "cocharacter": cocharacter,
            "factor_character_pairing": list(factor_pairing),
            "nonstar_exponents_are_only_minus_one_or_zero": True,
            "positive_nonstar_exponent_count": 0,
            "negative_nonstar_exponent_count":
                len(affected_local),
            "affected_local_indices": list(affected_local),
            "affected_global_indices": [
                nonstar_global_indices()[index]
                for index in affected_local
            ],
        })
    return tuple(rows)


def residual_recession_audit(
    representative: int | PivotRepresentative,
) -> dict:
    rep = pivot_representative(representative)
    certificates = _residual_recession_certificates_cached(
        rep.orbit_index
    )
    return {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "orbit_index": rep.orbit_index,
        "certificates": json.loads(json.dumps(certificates)),
        "exact_statement": (
            "each q_a lies in ker(C_p), has no positive U exponent, "
            "and strictly decreases the L2 norm whenever one affected "
            "coordinate is nonzero"
        ),
        "generic_dense_residual_L2_orbit_has_finite_minimizer":
            False,
        "residual_L2_balancing_is_a_valid_mandatory_retraction":
            False,
        "horizontal_projection_remains_available": True,
    }


def balance_residual_gauge(
    values: Sequence[complex],
    representative: int | PivotRepresentative,
    *,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 80,
    maximum_log_correction: float = 200.0,
    strict: bool = True,
) -> tuple[np.ndarray, dict]:
    r"""Audit a proposed minimum-L2 residual real-torus representative.

    On ``p=1`` let ``A_res = H_U K``.  This minimizes

    ``log sum_i |U_i|^2 exp(2 (A_res eta)_i)``

    over real ``eta`` by damped Newton/SVD only after exact recession
    tests.  Every dense pivot chart activates a certified recession, so
    this function normally fails closed and must not be used as a
    retained-run canonicalizer.  Exact zero coordinates are omitted from
    the logarithm and remain zero.  No selected coordinate is assumed
    nonzero.
    """

    u = _u_array(values)
    rep = pivot_representative(representative)
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or maximum_iterations < 1
        or not np.isfinite(maximum_log_correction)
        or maximum_log_correction <= 0
    ):
        raise KrennStarVariableProjectionError(
            "residual-gauge balance controls are invalid"
        )
    pivots_before = pivot_values(u, rep)
    if float(np.max(np.abs(pivots_before - 1.0))) > 2.0e-8:
        raise KrennStarVariableProjectionError(
            "residual balancing requires the p=1 slice"
        )
    h_matrix = np.asarray(
        nonstar_character_matrix(), dtype=np.float64
    )
    kernel = np.asarray(
        [
            [float(value) for value in row]
            for row in gauge_kernel(rep)
        ],
        dtype=np.float64,
    )
    residual_action = h_matrix @ kernel
    support = np.abs(u) > 0.0
    if not np.any(support):
        raise KrennStarVariableProjectionError(
            "the residual-gauge balance support is empty"
        )
    recession_rows = _residual_recession_certificates_cached(
        rep.orbit_index
    )
    active_recession = []
    for row in recession_rows:
        active_count = sum(
            bool(support[index])
            for index in row["affected_local_indices"]
        )
        if active_count:
            active_recession.append({
                "color": row["color"],
                "pivot_partner": row["pivot_partner"],
                "active_affected_coordinate_count": active_count,
                "cocharacter": row["cocharacter"],
            })
    if active_recession:
        diagnostics = {
            "schema": VARIABLE_PROJECTION_SCHEMA,
            "orbit_index": rep.orbit_index,
            "method": (
                "convex residual real-torus log-norm balance "
                "refused by exact recession certificate"
            ),
            "support_size": int(np.count_nonzero(support)),
            "zero_coordinate_count": int(np.count_nonzero(~support)),
            "converged": False,
            "accepted_for_optimization": False,
            "exact_recession_certificate_active": True,
            "active_recession_directions": active_recession,
            "suspected_no_finite_minimizer_or_recession": False,
            "no_finite_minimizer_certified_for_this_support": True,
            "coordinate_anchor_used": False,
            "zero_coordinates_excluded_from_chart": False,
            "mathematical_raw_norm_cap_used": False,
            "horizontal_projection_remains_available": True,
            "numerical_nonproof": True,
        }
        if strict:
            raise KrennResidualGaugeBalanceError(
                "residual L2 balancing has an exact recession on this "
                "support",
                diagnostics,
            )
        return u.copy(), diagnostics
    action = residual_action[support]
    log_weight_squared = 2.0 * np.log(np.abs(u[support]))
    eta = np.zeros(action.shape[1], dtype=np.float64)
    initial_value, _initial_gradient, _, initial_moment = (
        _logsumexp_balance_state(
            log_weight_squared, action, eta
        )
    )
    value = initial_value
    moment = initial_moment
    hessian_rank = 0
    converged = False
    line_search_failures = 0
    iterations = 0
    for iteration in range(maximum_iterations):
        iterations = iteration
        value, gradient, hessian, moment = (
            _logsumexp_balance_state(
                log_weight_squared, action, eta
            )
        )
        moment_norm = _safe_l2(moment)
        if moment_norm <= tolerance:
            converged = True
            break
        singular_values, hessian_rank, hessian_tolerance, _vh = (
            _numerical_rank_from_matrix(hessian, None)
        )
        if hessian_rank:
            # hessian is Hermitian real.  Its SVD right vectors diagonalize
            # it up to harmless signs; lstsq is clearer and deterministic.
            step = -np.linalg.pinv(
                hessian, rcond=(
                    hessian_tolerance
                    / max(float(singular_values[0]), 1.0)
                )
            ) @ gradient
        else:
            step = -gradient
        directional = float(np.dot(gradient, step))
        if not np.isfinite(directional) or directional >= 0:
            step = -gradient
            directional = -float(np.dot(gradient, gradient))
        step_norm = _safe_l2(step)
        if step_norm > 8.0:
            step *= 8.0 / step_norm
            directional = float(np.dot(gradient, step))
        accepted = False
        alpha = 1.0
        for _line_search in range(40):
            trial_eta = eta + alpha * step
            trial_value, _, _, _ = _logsumexp_balance_state(
                log_weight_squared, action, trial_eta
            )
            if trial_value <= value + 1.0e-4 * alpha * directional:
                eta = trial_eta
                value = trial_value
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            line_search_failures += 1
            break
        if _safe_l2(eta) > maximum_log_correction:
            break
    final_value, _final_gradient, final_hessian, final_moment = (
        _logsumexp_balance_state(
            log_weight_squared, action, eta
        )
    )
    final_moment_norm = _safe_l2(final_moment)
    if final_moment_norm <= tolerance:
        converged = True
    final_singular, hessian_rank, _, _ = (
        _numerical_rank_from_matrix(final_hessian, None)
    )
    action_singular, action_rank, _, _ = (
        _numerical_rank_from_matrix(action, None)
    )
    log_scales = residual_action @ eta
    suspected_recession = bool(
        not converged
        and (
            _safe_l2(eta)
            >= 0.95 * maximum_log_correction
            or (
                final_value < initial_value - 20.0
                and final_moment_norm
                > max(tolerance, 0.1 * _safe_l2(initial_moment))
            )
        )
    )
    finite_scale = bool(
        np.all(np.isfinite(log_scales))
        and float(np.max(np.abs(log_scales))) < 700.0
    )
    if finite_scale:
        balanced = u * np.exp(log_scales)
        balanced = _u_array(balanced)
        pivots_after = pivot_values(balanced, rep)
        pivot_error = float(np.max(np.abs(pivots_after - 1.0)))
    else:
        balanced = u.copy()
        pivots_after = pivots_before.copy()
        pivot_error = math.inf
    accepted_for_optimization = bool(
        converged
        and finite_scale
        and pivot_error <= 5.0e-8
    )
    difference = balanced - u
    diagnostics = {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "orbit_index": rep.orbit_index,
        "method": (
            "damped Newton/SVD on log sum |U_i|^2 exp(2 A_i eta)"
        ),
        "support_size": int(np.count_nonzero(support)),
        "zero_coordinate_count": int(np.count_nonzero(~support)),
        "residual_action_shape": [
            int(residual_action.shape[0]),
            int(residual_action.shape[1]),
        ],
        "support_action_rank": action_rank,
        "support_action_singular_values": [
            float(value) for value in action_singular
        ],
        "final_hessian_rank": hessian_rank,
        "final_hessian_singular_values": [
            float(value) for value in final_singular
        ],
        "initial_log_norm_objective": float(initial_value),
        "final_log_norm_objective": float(final_value),
        "initial_moment_norm": _safe_l2(initial_moment),
        "final_moment_norm": final_moment_norm,
        "tolerance": float(tolerance),
        "iterations": iterations,
        "maximum_iterations": maximum_iterations,
        "line_search_failures": line_search_failures,
        "eta_l2": _safe_l2(eta),
        "eta": [float(value) for value in eta],
        "maximum_abs_log_scale": float(
            np.max(np.abs(log_scales))
        ),
        "raw_to_balanced_l2": _safe_l2(difference),
        "raw_to_balanced_linf": float(
            np.max(np.abs(difference))
        ),
        "maximum_pivot_error_after": pivot_error,
        "converged": converged,
        "finite_scale": finite_scale,
        "suspected_no_finite_minimizer_or_recession":
            suspected_recession,
        "accepted_for_optimization": accepted_for_optimization,
        "coordinate_anchor_used": False,
        "zero_coordinates_excluded_from_chart": False,
        "mathematical_raw_norm_cap_used": False,
        "numerical_nonproof": True,
    }
    if strict and not accepted_for_optimization:
        raise KrennResidualGaugeBalanceError(
            "residual real-torus balancing failed closed",
            diagnostics,
        )
    return balanced, diagnostics


@dataclass(frozen=True)
class NormalFormResult:
    objective: float
    residual_squared: float
    regularization_penalty: float
    z: np.ndarray
    y: np.ndarray
    full_residual: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int
    rank_tolerance: float
    ridge_mu: float

    def summary(self) -> dict:
        return {
            "schema": VARIABLE_PROJECTION_SCHEMA,
            "mode": "BY=I_3 chart normal form",
            "objective": self.objective,
            "residual_squared": self.residual_squared,
            "residual_l2": math.sqrt(self.residual_squared),
            "residual_max_abs": float(
                np.max(np.abs(self.full_residual))
            ),
            "regularization_penalty":
                self.regularization_penalty,
            "ridge_mu": self.ridge_mu,
            "numerical_rank": self.numerical_rank,
            "rank_tolerance": self.rank_tolerance,
            "singular_values": [
                float(value) for value in self.singular_values
            ],
            "z_l2": _safe_l2(self.z),
            "z_linf": float(np.max(np.abs(self.z))),
            "y_l2": _safe_l2(self.y),
            "y_linf": float(np.max(np.abs(self.y))),
            "target_rows_enforced_exactly_in_the_parametrization":
                True,
            "y_or_z_bound_would_be_pivot_slice_dependent": True,
            "numerical_nonproof": True,
        }


def normal_form_diagnostic(
    values: Sequence[complex],
    representative: int | PivotRepresentative,
    *,
    ridge_mu: float = 1.0e-12,
    rank_rtol: float | None = None,
) -> NormalFormResult:
    """Solve the chart-normal-form ``240 x 12`` shared linear problem."""

    u = _u_array(values)
    rep = pivot_representative(representative)
    if not np.isfinite(ridge_mu) or ridge_mu <= 0:
        raise KrennStarVariableProjectionError(
            "normal-form ridge_mu must be strictly positive"
        )
    pivots = pivot_values(u, rep)
    if float(np.max(np.abs(pivots - 1.0))) > 2.0e-8:
        raise KrennStarVariableProjectionError(
            "the normal form requires the unit-pivot slice"
        )
    matrix = evaluate_star_matrix(u)
    pivot_columns = rep.columns_by_color
    free_columns = tuple(
        column
        for column in range(STAR_COLUMNS)
        if column not in set(pivot_columns)
    )
    b_matrix = matrix[np.asarray(TARGET_ROWS)]
    pivot_block = b_matrix[:, pivot_columns]
    if float(np.max(np.abs(pivot_block - np.eye(3)))) > 2.0e-8:
        raise KrennStarVariableProjectionError(
            "the selected target-row pivot block is not the identity"
        )
    c_matrix = b_matrix[:, free_columns]
    non_target_rows = np.asarray(
        [
            row for row in range(ROWS)
            if row not in set(TARGET_ROWS)
        ],
        dtype=np.int64,
    )
    a_pivot = matrix[non_target_rows][:, pivot_columns]
    a_free = matrix[non_target_rows][:, free_columns]
    reduced = a_free - a_pivot @ c_matrix
    rhs = -a_pivot
    left, singular_values, right_h = np.linalg.svd(
        reduced, full_matrices=False
    )
    active, rank, tolerance = _svd_rank(
        singular_values, 240, 12, rank_rtol
    )
    coefficients = left.conj().T @ rhs
    mu = float(ridge_mu)
    filter_values = singular_values / (
        singular_values**2 + mu
    )
    z = right_h.conj().T @ (
        filter_values[:, np.newaxis] * coefficients
    )
    y = np.zeros((STAR_COLUMNS, TARGETS), dtype=np.complex128)
    y[np.asarray(free_columns)] = z
    y[np.asarray(pivot_columns)] = np.eye(3) - c_matrix @ z
    full_residual = matrix @ y - _target_matrix()
    target_error = float(
        np.max(np.abs(full_residual[np.asarray(TARGET_ROWS)]))
    )
    if target_error > 5.0e-8:
        raise KrennStarVariableProjectionError(
            "the BY=I normal form failed target-row replay"
        )
    residual_squared = _squared_norm(full_residual)
    penalty = mu * _squared_norm(z)
    return NormalFormResult(
        objective=residual_squared + penalty,
        residual_squared=residual_squared,
        regularization_penalty=penalty,
        z=z,
        y=y,
        full_residual=full_residual,
        singular_values=singular_values,
        numerical_rank=rank,
        rank_tolerance=tolerance,
        ridge_mu=mu,
    )


def reconstruct_full_weights(
    values: Sequence[complex],
    y_values: Sequence[Sequence[complex]],
) -> np.ndarray:
    """Reconstruct all 135 weights from apex-zero ``U`` and ``Y``."""

    u = _u_array(values)
    y = _y_array(y_values)
    weights = np.zeros(AMBIENT_VARIABLES, dtype=np.complex128)
    weights[np.asarray(nonstar_global_indices(), dtype=np.int64)] = u
    factorization = star_linearization(APEX)
    for apex_color in range(D):
        for column, global_index in enumerate(
            factorization.star_variable_blocks[apex_color]
        ):
            weights[global_index] = y[column, apex_color]
    if np.count_nonzero(
        [
            APEX in variable_key(N, D, index)[:2]
            for index in range(AMBIENT_VARIABLES)
        ]
    ) != 45:
        raise KrennStarVariableProjectionError(
            "the reconstructed star census changed"
        )
    return _finite_complex_array(
        weights,
        shape=(AMBIENT_VARIABLES,),
        label="reconstructed ambient weights",
    )


@lru_cache(maxsize=1)
def _primary_monomial_array() -> np.ndarray:
    system = generate_sparse_system(N, D)
    result = np.asarray(
        system.monomial_variable_indices, dtype=np.int64
    ).reshape(729, 15, 3)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def _independent_matchings() -> tuple[
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...
]:
    edges = tuple(combinations(range(N), 2))
    rows = []
    for candidate in combinations(edges, 3):
        covered = tuple(vertex for edge in candidate for vertex in edge)
        if len(set(covered)) == N:
            rows.append(candidate)
    if len(rows) != 15:
        raise KrennStarVariableProjectionError(
            "the independent perfect-matching census changed"
        )
    return tuple(rows)


def _independent_variable_index(
    left: int,
    right: int,
    left_color: int,
    right_color: int,
) -> int:
    if left > right:
        left, right = right, left
        left_color, right_color = right_color, left_color
    edge_index = left * (2 * N - left - 1) // 2 + (
        right - left - 1
    )
    return (edge_index * D + left_color) * D + right_color


def dual_full_residual(
    values: Sequence[complex],
    y_values: Sequence[Sequence[complex]],
    *,
    numerical_zero_tolerance: float = 1.0e-10,
) -> dict:
    """Replay all 729 equations with primary and independent enumerators."""

    if (
        not np.isfinite(numerical_zero_tolerance)
        or numerical_zero_tolerance <= 0
    ):
        raise KrennStarVariableProjectionError(
            "the numerical-zero tolerance is invalid"
        )
    weights = reconstruct_full_weights(values, y_values)
    monomials = _primary_monomial_array()
    primary_outputs = np.prod(weights[monomials], axis=2).sum(axis=1)
    independent_outputs = []
    for coloring in product(range(D), repeat=N):
        value = 0.0 + 0.0j
        for matching in _independent_matchings():
            term = 1.0 + 0.0j
            for left, right in matching:
                term *= weights[_independent_variable_index(
                    left,
                    right,
                    coloring[left],
                    coloring[right],
                )]
            value += term
        independent_outputs.append(value)
    independent_outputs = np.asarray(
        independent_outputs, dtype=np.complex128
    )
    target = np.asarray(
        generate_sparse_system(N, D).rhs_values,
        dtype=np.complex128,
    )
    primary_residual = primary_outputs - target
    independent_residual = independent_outputs - target
    agreement = primary_outputs - independent_outputs
    maximum_agreement = float(np.max(np.abs(agreement)))
    agreement_tolerance = 2.0e-10 * max(
        1.0,
        float(np.max(np.abs(primary_outputs))),
        float(np.max(np.abs(independent_outputs))),
    )
    enumerators_agree = maximum_agreement <= agreement_tolerance
    if not enumerators_agree:
        raise KrennStarVariableProjectionError(
            "the primary and independent numerical enumerators disagree"
        )
    primary_max = float(np.max(np.abs(primary_residual)))
    independent_max = float(np.max(np.abs(independent_residual)))
    return {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "primary": {
            "residual_l2": _safe_l2(primary_residual),
            "residual_max_abs": primary_max,
        },
        "independent": {
            "residual_l2": _safe_l2(independent_residual),
            "residual_max_abs": independent_max,
        },
        "primary_independent_output_agreement_max_abs":
            maximum_agreement,
        "maximum_output_difference": maximum_agreement,
        "primary_independent_agreement_tolerance":
            agreement_tolerance,
        "all_729_replayed": (
            primary_residual.shape == (729,)
            and independent_residual.shape == (729,)
        ),
        "all_729_equations_replayed": (
            primary_residual.shape == (729,)
            and independent_residual.shape == (729,)
        ),
        "primary_and_independent_outputs_agree":
            enumerators_agree,
        "numerical_zero_tolerance":
            float(numerical_zero_tolerance),
        "numerical_zero_both_enumerators": bool(
            primary_max <= numerical_zero_tolerance
            and independent_max <= numerical_zero_tolerance
        ),
        "exact_verification_completed": False,
        "numerical_nonproof": True,
    }


@lru_cache(maxsize=3)
def _anchored_complement_cached(orbit_index: int) -> dict:
    rep = pivot_representative(orbit_index)
    audit = star_pivot_gauge_audit()
    apex_row = next(
        row for row in audit["apex_audits"]
        if row["apex"] == APEX
    )
    row = next(
        item
        for item in apex_row["transported_orbit_representatives"]
        if item["equality_pattern"] == rep.equality_pattern
    )
    indices = tuple(
        map(int, row["nonstar_complement"]["weight_indices"])
    )
    if len(indices) != 12 or any(
        index not in _global_to_local() for index in indices
    ):
        raise KrennStarVariableProjectionError(
            "the determinant-two complement is malformed"
        )
    full_characters = color_diagonal_exponent_matrix()
    square = (
        *rep.factor_characters,
        *(tuple(map(int, full_characters[index])) for index in indices),
    )
    determinant = _determinant_over_q(square)
    rank = _matrix_rank_over_q(square)
    if (
        abs(determinant) != 2
        or rank != GAUGE_DIMENSION
        or row["nonstar_complement"]["smith_invariant_factors"]
        != [1] * 14 + [2]
    ):
        raise KrennStarVariableProjectionError(
            "the anchor diagnostic failed determinant-two/SNF replay"
        )
    return {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "orbit_index": rep.orbit_index,
        "weight_indices": list(indices),
        "weight_keys": [
            list(variable_key(N, D, index)) for index in indices
        ],
        "exact_rank_over_Q": rank,
        "exact_determinant": int(determinant),
        "absolute_determinant_required": 2,
        "smith_invariant_factors": [1] * 14 + [2],
        "complex_torus_map_surjective": True,
        "finite_kernel_order": 2,
        "diagnostic_only": True,
        "coordinate_anchors_are_guaranteed_nonzero": False,
        "covers_solutions_with_a_zero_anchor": False,
        "used_by_anchor_free_primary_search": False,
        "numerical_nonproof": True,
    }


def anchored_complement_diagnostic(
    representative: int | PivotRepresentative,
) -> dict:
    return json.loads(json.dumps(
        _anchored_complement_cached(
            pivot_representative(representative).orbit_index
        ),
        allow_nan=False,
    ))


@lru_cache(maxsize=1)
def pole_quantity_character() -> tuple[int, ...]:
    full = color_diagonal_exponent_matrix()
    character = tuple(
        sum(full[index][parameter] for index in POLE_INVARIANT_INDICES)
        for parameter in range(GAUGE_DIMENSION)
    )
    expected = (
        -1, 1, 0,
        -1, 1, 0,
        0, 1, -1,
        0, 0, 0,
        0, 1, -1,
    )
    if character != expected:
        raise KrennStarVariableProjectionError(
            "the known pole quantity's color-gauge character changed"
        )
    return character


def path_diagnostics(
    values: Sequence[complex],
    representative: int | PivotRepresentative,
    y_values: Sequence[Sequence[complex]] | None = None,
) -> dict:
    """Return raw norms and the chart-dependent known pole quantity.

    The six-factor quantity is invariant under the older vertex-scalar
    gauge but is *not* invariant under the 15-dimensional direct-GHZ
    color gauge used here.  It is logged only as a chart/path-dependent
    degeneration diagnostic and is never a mathematical cap.
    """

    u = _u_array(values)
    rep = pivot_representative(representative)
    character = pole_quantity_character()
    augmented_rank = _matrix_rank_over_q((
        *rep.factor_characters,
        character,
    ))
    if (
        _matrix_rank_over_q(rep.factor_characters) != 3
        or augmented_rank != 4
    ):
        raise KrennStarVariableProjectionError(
            "the pole quantity's nontrivial residual character changed"
        )
    quantity_payload: dict[str, object]
    if y_values is None:
        quantity_payload = {
            "available": False,
            "reason": (
                "the six-factor quantity includes two apex-star "
                "weights and requires recovered Y"
            ),
        }
    else:
        weights = reconstruct_full_weights(u, y_values)
        factors = weights[
            np.asarray(POLE_INVARIANT_INDICES, dtype=np.int64)
        ]
        factor_magnitudes = np.abs(factors)
        exact_zero = bool(np.any(factor_magnitudes == 0.0))
        if exact_zero:
            quantity_payload = {
                "available": True,
                "value": [0.0, 0.0],
                "abs": 0.0,
                "log_abs": None,
                "log_abs_defined": False,
                "exact_zero_in_float64": True,
                "finite": True,
                "float64_value_available": True,
                "computed_without_product_overflow": True,
            }
        else:
            log_magnitude = float(
                np.sum(np.log(factor_magnitudes))
            )
            unit_phase = complex(
                np.prod(factors / factor_magnitudes)
            )
            minimum_log = math.log(
                float(np.nextafter(0.0, 1.0))
            )
            maximum_log = math.log(
                float(np.finfo(np.float64).max)
            )
            representable = bool(
                minimum_log <= log_magnitude <= maximum_log
            )
            if representable:
                magnitude = math.exp(log_magnitude)
                quantity = unit_phase * magnitude
                value = [
                    float(quantity.real),
                    float(quantity.imag),
                ]
                magnitude_value: float | None = magnitude
            else:
                value = None
                magnitude_value = None
            quantity_payload = {
                "available": True,
                "value": value,
                "abs": magnitude_value,
                "log_abs": log_magnitude,
                "log_abs_defined": True,
                "exact_zero_in_float64": False,
                "finite": representable,
                "float64_value_available": representable,
                "computed_without_product_overflow": True,
            }
    return {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "orbit_index": rep.orbit_index,
        "u_l2": _safe_l2(u),
        "u_linf": float(np.max(np.abs(u))),
        "u_min_nonzero_abs": float(
            np.min(np.abs(u[np.abs(u) > 0]))
        ) if np.any(np.abs(u) > 0) else 0.0,
        "pivot_values": [
            [float(value.real), float(value.imag)]
            for value in pivot_values(u, rep)
        ],
        "pole_quantity": quantity_payload,
        "pole_quantity_exact_color_gauge_character":
            list(character),
        "pivot_character_rank_over_Q": 3,
        "pivot_plus_pole_character_rank_over_Q": augmented_rank,
        "pole_quantity_full_direct_GHZ_color_gauge_invariant":
            False,
        "pole_quantity_vertex_scalar_invariant_only": True,
        "chart_and_gauge_path_dependent_diagnostic": True,
        "valid_as_full_gauge_invariant_mathematical_cap": False,
        "raw_u_norm_is_a_mathematical_search_cap": False,
        "numerical_nonproof": True,
    }


def detect_two_cycle(
    current: Sequence[complex],
    previous: Sequence[complex],
    two_back: Sequence[complex],
    *,
    relative_tolerance: float = 1.0e-10,
) -> dict:
    """Instrument a possible retraction/balance two-cycle."""

    current_array = _u_array(current)
    previous_array = _u_array(previous)
    two_back_array = _u_array(two_back)
    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance <= 0
    ):
        raise KrennStarVariableProjectionError(
            "the two-cycle tolerance is invalid"
        )
    one_step = _safe_l2(current_array - previous_array)
    two_step = _safe_l2(current_array - two_back_array)
    scale = max(1.0, _safe_l2(current_array))
    detected = bool(
        two_step <= relative_tolerance * scale
        and one_step > 10.0 * relative_tolerance * scale
    )
    return {
        "one_step_l2": one_step,
        "two_step_l2": two_step,
        "relative_tolerance": float(relative_tolerance),
        "two_cycle_detected": detected,
        "cycle_is_exact_proof": False,
        "numerical_nonproof": True,
    }


def variable_projection_diagnostics(
    values: Sequence[complex],
    representative: int | PivotRepresentative,
    *,
    ridge_mu: float = 1.0e-8,
) -> dict:
    """Return a serializable, explicitly nonproof diagnostic snapshot."""

    u = _u_array(values)
    rep = pivot_representative(representative)
    raw = raw_variable_projection(u)
    ridge = ridge_variable_projection(u, ridge_mu)
    normal = normal_form_diagnostic(
        u, rep, ridge_mu=max(ridge_mu, 1.0e-14)
    )
    horizontal = horizontal_projection(
        u, ridge.gradient, rep
    )
    return {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "orbit": rep.to_dict(),
        "raw": raw.summary(),
        "ridge": ridge.summary(),
        "normal_form": normal.summary(),
        "horizontal": horizontal.summary(),
        "path": path_diagnostics(u, rep, ridge.y),
        "anchor_cross_check":
            anchored_complement_diagnostic(rep),
        "claim_boundary": {
            "finite_counterexample_found": False,
            "exact_reconstruction_completed": False,
            "global_affine_membership_decided": False,
            "bounded_miss_is_a_proof": False,
            "numerical_residual_is_a_proof": False,
            "raw_u_norm_is_a_mathematical_cap": False,
        },
    }


@lru_cache(maxsize=1)
def core_audit() -> dict:
    """Return an exact-structure fingerprint and fail-closed ledger."""

    reps = pivot_representatives()
    payload = {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "parameters": {
            "n": N,
            "d": D,
            "apex": APEX,
            "nonstar_variables": NONSTAR_VARIABLES,
            "star_variables": 45,
            "shared_matrix_shape": [ROWS, STAR_COLUMNS],
            "targets": TARGETS,
            "direct_GHZ_color_gauge_dimension": GAUGE_DIMENSION,
        },
        "exact_inputs": {
            "star_linearization_sha256":
                star_linearization(APEX).fingerprint(),
            "nonstar_character_rank_over_Q":
                _matrix_rank_over_q(nonstar_character_matrix()),
            "pivot_orbits": [rep.to_dict() for rep in reps],
            "pivot_orbit_sizes_sum_to_125":
                sum(rep.orbit_size for rep in reps) == 125,
            "gauge_kernel_dimensions": [
                len(gauge_kernel(rep)[0]) for rep in reps
            ],
            "normalization_character_pairing": "-I_3",
            "root_extraction_required": False,
            "coordinate_anchor_required": False,
        },
        "numerical_methods": {
            "raw_span_distance": "rank-revealing SVD diagnostic",
            "smooth_objective": "ridge variable projection",
            "gradient": "exact quadratic-incidence pullback",
            "horizontal": (
                "ker(dp) intersect "
                "(diag(U) H ker(C_p)) orthogonal complement"
            ),
            "residual_real_torus_balance": (
                "convex log-norm damped Newton/SVD, refused whenever "
                "the exact built-in recession is active"
            ),
            "residual_L2_balance_has_generic_exact_recession": True,
            "residual_L2_balance_is_mandatory": False,
            "raw_u_clipping": False,
            "raw_u_mathematical_cap": False,
            "floating_point_overflow_guard_only": True,
        },
        "known_pole_quantity": {
            "exact_color_gauge_character":
                list(pole_quantity_character()),
            "full_color_gauge_invariant": False,
            "chart_path_dependent_diagnostic_only": True,
        },
        "claim_boundary": {
            "all_outputs_numerical_nonproof": True,
            "rank_threshold_is_exact_rank": False,
            "small_residual_is_exact_witness": False,
            "bounded_or_timed_miss_is_nonexistence_proof": False,
            "global_affine_membership_decided": False,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return json.loads(json.dumps(payload, allow_nan=False))


def source_fingerprints() -> dict:
    """Return deterministic exact-input fingerprints for checkpoints."""

    audit = core_audit()
    return {
        "schema": VARIABLE_PROJECTION_SCHEMA,
        "core_audit_sha256": audit["sha256"],
        "star_linearization_sha256": audit["exact_inputs"][
            "star_linearization_sha256"
        ],
        "nonstar_character_rank_over_Q": audit["exact_inputs"][
            "nonstar_character_rank_over_Q"
        ],
        "pivot_orbit_count": len(audit["exact_inputs"]["pivot_orbits"]),
    }


__all__ = [
    "AMBIENT_VARIABLES",
    "APEX",
    "D",
    "GAUGE_DIMENSION",
    "KrennResidualGaugeBalanceError",
    "KrennStarVariableProjectionError",
    "N",
    "NONSTAR_VARIABLES",
    "NormalFormResult",
    "PivotRepresentative",
    "ProjectionResult",
    "ROWS",
    "STAR_COLUMNS",
    "TARGET_ROWS",
    "VARIABLE_PROJECTION_SCHEMA",
    "anchored_complement_diagnostic",
    "assert_rank_stable",
    "balance_residual_gauge",
    "core_audit",
    "detect_two_cycle",
    "dual_full_residual",
    "evaluate_star_matrix",
    "gauge_kernel",
    "horizontal_projection",
    "initial_point",
    "natural_repair_initial_point",
    "nonstar_character_matrix",
    "nonstar_global_indices",
    "normal_form_diagnostic",
    "normalization_cocharacters",
    "path_diagnostics",
    "pivot_characters",
    "pivot_jacobian",
    "pivot_representative",
    "pivot_representatives",
    "pivot_values",
    "pole_quantity_character",
    "pullback_gradient",
    "raw_variable_projection",
    "reconstruct_full_weights",
    "residual_recession_audit",
    "retract_pivots",
    "ridge_variable_projection",
    "source_fingerprints",
    "targets",
    "variable_projection_diagnostics",
]
