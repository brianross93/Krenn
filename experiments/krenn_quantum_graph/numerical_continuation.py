"""Bounded complex continuation for the native ``n=6,d=3`` equations.

This module is deliberately numerical and fail-closed.  It evaluates the
original 729 perfect-matching equations, supplies their analytic complex
Jacobian, fixes the target-preserving color-diagonal gauge seen by each
declared support, and runs deterministic norm-bounded
Levenberg--Marquardt searches.

The known Laurent family is included as a control:

``Phi(W(t)) = GHZ + t*e_002121`` with ``W[0]=t`` and ``W[81]=t^-1``.

A small residual, a completed continuation, or a bounded-search miss is not
an exact witness or a nonexistence proof.  Every public result and checkpoint
records that boundary explicitly.  Exact reconstruction and verification
belong to a separate layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# NumPy's Windows OpenBLAS wheel may otherwise use every available core in
# each campaign worker.  These variables must be set before importing NumPy.
NUMERICAL_BLAS_THREAD_LIMIT = 1
_BLAS_THREAD_ENVIRONMENT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_variable in _BLAS_THREAD_ENVIRONMENT_VARIABLES:
    os.environ[_thread_variable] = str(NUMERICAL_BLAS_THREAD_LIMIT)

import numpy as np

from experiments.krenn_quantum_graph.formal_lift import (
    GAUGE_SLICE_INDICES,
    POLE_INVARIANT_INDICES,
    REPAIR_DECREASE_INDEX,
    REPAIR_INCREASE_INDEX,
)
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    generate_sparse_system,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    n6_d3_seed_witness,
)


N = 6
D = 3
AMBIENT_VARIABLES = 135
EQUATION_COUNT = 729
MATCHINGS_PER_EQUATION = 15
DEGREE = 3

NUMERICAL_CONTINUATION_SCHEMA = (
    "krenn-n6-d3-bounded-complex-continuation-v2"
)
GAUGE_CHART_SCHEMA = "krenn-n6-d3-numerical-gauge-chart-v2"
LM_PLAN_SCHEMA = "krenn-n6-d3-bounded-complex-lm-plan-v1"
LM_RESULT_SCHEMA = "krenn-n6-d3-bounded-complex-lm-result-v1"
CHECKPOINT_SCHEMA = "krenn-n6-d3-complex-lm-checkpoint-v2"
CONTINUATION_SCHEMA = "krenn-n6-d3-complex-continuation-result-v2"
JOB_SCHEMA = "krenn-n6-d3-numerical-job-v1"

VERTEX_SCALAR_GAUGE = "vertex-scalar"
GHZ_VICTIM_COLOR_DIAGONAL_GAUGE = (
    "ghz-victim-color-diagonal"
)
GHZ_COLOR_DIAGONAL_GAUGE = "ghz-color-diagonal"
_GAUGE_ACTION_GROUPS = {
    VERTEX_SCALAR_GAUGE,
    GHZ_VICTIM_COLOR_DIAGONAL_GAUGE,
    GHZ_COLOR_DIAGONAL_GAUGE,
}

DEFAULT_TARGET_SCHEDULE = (
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.0,
)
RESIDUAL_THRESHOLDS = (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)
CONSTANT_EQUATIONS = (0, 364, 728)
NATURAL_SUPPORT = tuple(index for index, _ in n6_d3_seed_witness().entries)
NATURAL_SUPPORT_SET = frozenset(NATURAL_SUPPORT)
NATURAL_GAUGE_ANCHORS = tuple(map(int, GAUGE_SLICE_INDICES))
KNOWN_Q_INDICES = tuple(map(int, POLE_INVARIANT_INDICES))

_TERMINAL_FAILURE_STATUSES = {
    "nonfinite-evaluation",
    "linear-algebra-failure",
    "invalid-step",
}
_LM_STATUSES = {
    "numerical-residual-tolerance",
    "iteration-cap-reached",
    "evaluation-cap-reached",
    "gradient-stalled",
    "step-stalled",
    "norm-bound-stalled",
    "nonfinite-evaluation",
    "linear-algebra-failure",
    "invalid-step",
    "checkpoint-paused",
}


class KrennNumericalContinuationError(ValueError):
    """A numerical plan, support, checkpoint, or replay is malformed."""


def _canonical_support(support: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(map(int, support))
    except TypeError as error:
        raise KrennNumericalContinuationError(
            "support must be an iterable of variable indices"
        ) from error
    if (
        values != tuple(sorted(values))
        or len(values) != len(set(values))
        or any(index < 0 or index >= AMBIENT_VARIABLES for index in values)
    ):
        raise KrennNumericalContinuationError(
            "support must contain unique increasing n=6,d=3 indices"
        )
    if not values:
        raise KrennNumericalContinuationError(
            "a numerical support cannot be empty"
        )
    return values


def _complex_pair(value: complex) -> list[float | None]:
    value = complex(value)
    return [_finite_or_none(value.real), _finite_or_none(value.imag)]


def _complex_from_pair(value: Sequence[float]) -> complex:
    try:
        real, imaginary = value
        result = complex(float(real), float(imaginary))
    except (TypeError, ValueError) as error:
        raise KrennNumericalContinuationError(
            "a complex JSON value must be a real-imaginary pair"
        ) from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise KrennNumericalContinuationError(
            "checkpoint complex values must be finite"
        )
    return result


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _json_bytes(value: Mapping) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise KrennNumericalContinuationError(
            "a numerical payload is not canonical JSON"
        ) from error
    return text.encode("utf-8")


def _payload_sha256(value: Mapping) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex_vector_sha256(values: Sequence[complex] | np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.complex128)
    if array.shape != (AMBIENT_VARIABLES,):
        raise KrennNumericalContinuationError(
            "numerical initialization hash needs 135 complex weights"
        )
    digest = hashlib.sha256()
    digest.update(b"krenn-n6-d3-complex-initialization-v1\0")
    digest.update(array.view(np.float64).tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _system():
    system = generate_sparse_system(N, D)
    if (
        system.variable_count != AMBIENT_VARIABLES
        or system.equation_count != EQUATION_COUNT
        or system.matching_count != MATCHINGS_PER_EQUATION
        or system.degree != DEGREE
    ):
        raise KrennNumericalContinuationError(
            "the canonical n=6,d=3 system dimensions changed"
        )
    return system


@lru_cache(maxsize=1)
def monomial_index_array() -> np.ndarray:
    """Return the read-only ``(10935,3)`` perfect-matching index table."""

    array = np.asarray(
        _system().monomial_variable_indices, dtype=np.intp
    )
    if array.shape != (
        EQUATION_COUNT * MATCHINGS_PER_EQUATION,
        DEGREE,
    ):
        raise KrennNumericalContinuationError(
            "the perfect-matching monomial array changed shape"
        )
    array.setflags(write=False)
    return array


@lru_cache(maxsize=1)
def equation_index_array() -> np.ndarray:
    """Return the equation index belonging to each flattened monomial."""

    result = np.repeat(
        np.arange(EQUATION_COUNT, dtype=np.intp),
        MATCHINGS_PER_EQUATION,
    )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def canonical_rhs_array() -> np.ndarray:
    """Return the read-only complex canonical-GHZ target."""

    result = np.asarray(_system().rhs_values, dtype=np.complex128)
    if result.shape != (EQUATION_COUNT,):
        raise KrennNumericalContinuationError(
            "the n=6,d=3 target vector changed shape"
        )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def system_fingerprint() -> str:
    """Hash the exact system structure used by numerical checkpoints."""

    digest = hashlib.sha256()
    digest.update(b"krenn-n6-d3-numerical-system-v1\0")
    monomials = np.ascontiguousarray(
        monomial_index_array(), dtype=np.uint16
    )
    rhs = np.ascontiguousarray(
        canonical_rhs_array().real, dtype=np.uint8
    )
    digest.update(monomials.tobytes(order="C"))
    digest.update(rhs.tobytes(order="C"))
    return digest.hexdigest()


def _dense_weights(
    weights: Sequence[complex] | np.ndarray,
    support: Sequence[int] | None = None,
) -> np.ndarray:
    """Normalize a dense or support-ordered vector to 135 complex entries."""

    array = np.asarray(weights, dtype=np.complex128)
    if array.ndim != 1:
        raise KrennNumericalContinuationError(
            "complex weights must be a one-dimensional vector"
        )
    if array.shape == (AMBIENT_VARIABLES,):
        result = array.copy()
        if support is not None:
            canonical = _canonical_support(support)
            outside = np.ones(AMBIENT_VARIABLES, dtype=bool)
            outside[np.asarray(canonical, dtype=np.intp)] = False
            result[outside] = 0.0
        return result
    if support is None:
        raise KrennNumericalContinuationError(
            "a support-ordered weight vector needs an explicit support"
        )
    canonical = _canonical_support(support)
    if array.shape != (len(canonical),):
        raise KrennNumericalContinuationError(
            "support-ordered weights have the wrong length"
        )
    result = np.zeros(AMBIENT_VARIABLES, dtype=np.complex128)
    result[np.asarray(canonical, dtype=np.intp)] = array
    return result


def target_array(target_amplitude: complex = 0.0j) -> np.ndarray:
    """Return ``GHZ + target_amplitude*e_002121``."""

    amplitude = complex(target_amplitude)
    if not (
        math.isfinite(amplitude.real)
        and math.isfinite(amplitude.imag)
    ):
        raise KrennNumericalContinuationError(
            "continuation target amplitude must be finite"
        )
    result = canonical_rhs_array().copy()
    result[N6_D3_SEED_DEFECT_EQUATION] += amplitude
    return result


def complex_output(
    weights: Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Evaluate all 729 perfect-matching coefficients vectorially."""

    dense = _dense_weights(weights)
    with np.errstate(over="ignore", invalid="ignore"):
        factors = dense[monomial_index_array()]
        terms = factors[:, 0] * factors[:, 1] * factors[:, 2]
        output = terms.reshape(
            EQUATION_COUNT, MATCHINGS_PER_EQUATION
        ).sum(axis=1)
    return np.asarray(output, dtype=np.complex128)


def complex_residual(
    weights: Sequence[complex] | np.ndarray,
    target_amplitude: complex = 0.0j,
) -> np.ndarray:
    """Evaluate the complete numerical residual in equation order."""

    return complex_output(weights) - target_array(target_amplitude)


def analytic_jacobian(
    weights: Sequence[complex] | np.ndarray,
    columns: Sequence[int] | None = None,
) -> np.ndarray:
    """Return the analytic holomorphic Jacobian for all 729 equations.

    ``columns`` controls which ambient variables are differentiated.  The
    returned matrix always has all 729 equation rows.
    """

    dense = _dense_weights(weights)
    if columns is None:
        canonical_columns = tuple(range(AMBIENT_VARIABLES))
    else:
        raw = tuple(map(int, columns))
        if (
            raw != tuple(sorted(raw))
            or len(raw) != len(set(raw))
            or any(
                index < 0 or index >= AMBIENT_VARIABLES
                for index in raw
            )
        ):
            raise KrennNumericalContinuationError(
                "Jacobian columns must be unique increasing ambient indices"
            )
        canonical_columns = raw
    jacobian = np.zeros(
        (EQUATION_COUNT, len(canonical_columns)),
        dtype=np.complex128,
    )
    if not canonical_columns:
        return jacobian

    ambient_to_local = np.full(
        AMBIENT_VARIABLES, -1, dtype=np.intp
    )
    ambient_to_local[
        np.asarray(canonical_columns, dtype=np.intp)
    ] = np.arange(len(canonical_columns), dtype=np.intp)
    monomials = monomial_index_array()
    equations = equation_index_array()
    with np.errstate(over="ignore", invalid="ignore"):
        factors = dense[monomials]
        derivatives = (
            factors[:, 1] * factors[:, 2],
            factors[:, 0] * factors[:, 2],
            factors[:, 0] * factors[:, 1],
        )
        for position in range(DEGREE):
            local = ambient_to_local[monomials[:, position]]
            active = local >= 0
            np.add.at(
                jacobian,
                (equations[active], local[active]),
                derivatives[position][active],
            )
    return jacobian


def residual_and_jacobian(
    weights: Sequence[complex] | np.ndarray,
    columns: Sequence[int] | None = None,
    target_amplitude: complex = 0.0j,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full residual and selected-column analytic Jacobian."""

    dense = _dense_weights(weights)
    return (
        complex_residual(dense, target_amplitude),
        analytic_jacobian(dense, columns),
    )


@lru_cache(maxsize=1)
def gauge_exponent_matrix() -> np.ndarray:
    """Return the exact ``135 x 5`` vertex-scalar exponent matrix.

    This legacy subgroup rescales a variable on edge ``ij`` by
    ``lambda_i lambda_j``, with ``prod_i lambda_i = 1``.  Numerical search
    uses the larger target stabilizers returned by
    :func:`color_diagonal_ghz_gauge_exponent_matrix` and
    :func:`color_diagonal_moving_target_gauge_exponent_matrix`.
    """

    result = np.empty((AMBIENT_VARIABLES, 5), dtype=np.int8)
    for index in range(AMBIENT_VARIABLES):
        i, j, _a, _b = variable_key(N, D, index)
        result[index] = tuple(
            int(i == direction)
            + int(j == direction)
            - int(i == 5)
            - int(j == 5)
            for direction in range(5)
        )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def color_diagonal_ghz_gauge_exponent_matrix() -> np.ndarray:
    r"""Return the exact ``135 x 15`` color-diagonal GHZ stabilizer.

    For ``lambda_{i,a}`` satisfying ``prod_i lambda_{i,a}=1`` separately
    for each color ``a``, the action

    ``w_{ij}^{ab} -> lambda_{i,a} lambda_{j,b} w_{ij}^{ab}``

    preserves the direct GHZ fiber.  Five independent vertex exponents are
    used for each of the three colors.
    """

    result = np.zeros((AMBIENT_VARIABLES, 15), dtype=np.int8)
    for index in range(AMBIENT_VARIABLES):
        i, j, a, b = variable_key(N, D, index)
        for color in range(D):
            offset = 5 * color
            for direction in range(5):
                result[index, offset + direction] = (
                    int(a == color)
                    * (int(i == direction) - int(i == 5))
                    + int(b == color)
                    * (int(j == direction) - int(j == 5))
                )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def victim_gauge_character() -> np.ndarray:
    """Return the 15 color-diagonal exponents of coloring ``002121``."""

    victim = coloring_from_index(
        N, D, N6_D3_SEED_DEFECT_EQUATION
    )
    result = np.zeros(15, dtype=np.int8)
    for vertex, color in enumerate(victim):
        offset = 5 * color
        for direction in range(5):
            result[offset + direction] += (
                int(vertex == direction) - int(vertex == 5)
            )
    if not np.any(result):
        raise KrennNumericalContinuationError(
            "the victim gauge character unexpectedly vanished"
        )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def _moving_target_gauge_parameter_basis() -> np.ndarray:
    """Return an integer ``15 x 14`` basis killing the victim character."""

    character = victim_gauge_character().astype(np.int16)
    pivot = int(np.flatnonzero(character)[0])
    pivot_value = int(character[pivot])
    if abs(pivot_value) != 1:
        raise KrennNumericalContinuationError(
            "the victim gauge character lacks a primitive pivot"
        )
    columns: list[np.ndarray] = []
    for index in range(len(character)):
        if index == pivot:
            continue
        column = np.zeros(len(character), dtype=np.int16)
        column[index] = pivot_value
        column[pivot] = -int(character[index])
        columns.append(column)
    result = np.column_stack(columns)
    if (
        result.shape != (15, 14)
        or np.any(character @ result)
        or np.linalg.matrix_rank(result.astype(np.float64)) != 14
    ):
        raise KrennNumericalContinuationError(
            "the moving-target gauge basis failed replay"
        )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def color_diagonal_moving_target_gauge_exponent_matrix() -> np.ndarray:
    """Return the fixed ``135 x 14`` GHZ+victim stabilizer.

    The matrix is the color-diagonal GHZ action restricted to the kernel of
    the victim character.  It therefore preserves every target
    ``GHZ+t*e_002121`` and can be held fixed through the whole continuation,
    including its final direct-GHZ step.
    """

    result = (
        color_diagonal_ghz_gauge_exponent_matrix().astype(np.int16)
        @ _moving_target_gauge_parameter_basis()
    )
    if result.shape != (AMBIENT_VARIABLES, 14):
        raise KrennNumericalContinuationError(
            "the moving-target gauge matrix changed shape"
        )
    result = np.asarray(result, dtype=np.int16)
    result.setflags(write=False)
    return result


def _gauge_exponents(action_group: str) -> np.ndarray:
    if action_group == VERTEX_SCALAR_GAUGE:
        return gauge_exponent_matrix()
    if action_group == GHZ_VICTIM_COLOR_DIAGONAL_GAUGE:
        return color_diagonal_moving_target_gauge_exponent_matrix()
    if action_group == GHZ_COLOR_DIAGONAL_GAUGE:
        return color_diagonal_ghz_gauge_exponent_matrix()
    raise KrennNumericalContinuationError(
        "unknown numerical gauge action group"
    )


@lru_cache(maxsize=1)
def known_q_color_gauge_character() -> np.ndarray:
    """Return the full color-diagonal character of the known pole monomial."""

    result = np.asarray(
        color_diagonal_ghz_gauge_exponent_matrix()[
            np.asarray(KNOWN_Q_INDICES, dtype=np.intp)
        ].sum(axis=0),
        dtype=np.int16,
    )
    if not np.array_equal(result, -victim_gauge_character()):
        raise KrennNumericalContinuationError(
            "known Q is not the inverse victim gauge character"
        )
    result.setflags(write=False)
    return result


def _rank_over_q(rows: Sequence[Sequence[int]]) -> int:
    """Compute the rank of a tiny integer matrix exactly over ``Q``."""

    work = [list(map(Fraction, row)) for row in rows]
    if not work:
        return 0
    width = len(work[0])
    if any(len(row) != width for row in work):
        raise KrennNumericalContinuationError(
            "gauge-rank input rows have inconsistent widths"
        )
    pivot_row = 0
    for column in range(width):
        selected = next(
            (
                row
                for row in range(pivot_row, len(work))
                if work[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        work[pivot_row], work[selected] = (
            work[selected],
            work[pivot_row],
        )
        pivot = work[pivot_row][column]
        work[pivot_row] = [
            value / pivot for value in work[pivot_row]
        ]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            multiplier = work[row][column]
            work[row] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(
                    work[row], work[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def _deterministic_anchor_basis(
    support: Sequence[int],
    matrix: np.ndarray,
    *,
    preferred: Sequence[int] = (),
) -> tuple[int, ...]:
    """Choose the lexicographic greedy basis of the effective gauge rows."""

    anchors: list[int] = []
    rows: list[tuple[int, ...]] = []
    rank = 0
    support_set = set(support)
    preferred = tuple(
        index for index in preferred if index in support_set
    )
    preferred_set = set(preferred)
    order = (*preferred, *(
        index for index in support if index not in preferred_set
    ))
    for index in order:
        row = tuple(map(int, matrix[index]))
        candidate_rank = _rank_over_q((*rows, row))
        if candidate_rank > rank:
            anchors.append(index)
            rows.append(row)
            rank = candidate_rank
        if rank == matrix.shape[1]:
            break
    expected = _rank_over_q(
        tuple(tuple(map(int, matrix[index])) for index in support)
    )
    if rank != expected:
        raise KrennNumericalContinuationError(
            "deterministic gauge anchors failed maximal-rank replay"
        )
    return tuple(sorted(anchors))


@dataclass(frozen=True)
class GaugeChart:
    """A support-local fixed chart for a declared target stabilizer."""

    support: tuple[int, ...]
    anchors: tuple[int, ...]
    free_indices: tuple[int, ...]
    rank: int
    kind: str = "support-adaptive"
    action_group: str = GHZ_COLOR_DIAGONAL_GAUGE
    anchor_value: complex = 1.0 + 0.0j
    schema: str = GAUGE_CHART_SCHEMA

    def __post_init__(self) -> None:
        support = _canonical_support(self.support)
        anchors = tuple(map(int, self.anchors))
        free = tuple(map(int, self.free_indices))
        rank = int(self.rank)
        value = complex(self.anchor_value)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "free_indices", free)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "anchor_value", value)
        if self.schema != GAUGE_CHART_SCHEMA:
            raise KrennNumericalContinuationError(
                "numerical gauge-chart schema changed"
            )
        if self.kind not in {
            "support-adaptive",
            "natural-certified",
            "natural-extended",
        }:
            raise KrennNumericalContinuationError(
                "unknown numerical gauge-chart kind"
            )
        if self.action_group not in _GAUGE_ACTION_GROUPS:
            raise KrennNumericalContinuationError(
                "unknown numerical gauge-chart action group"
            )
        if (
            anchors != tuple(sorted(anchors))
            or len(anchors) != len(set(anchors))
            or not set(anchors).issubset(support)
            or free
            != tuple(index for index in support if index not in anchors)
            or rank != len(anchors)
            or not 0
            <= rank
            <= _gauge_exponents(self.action_group).shape[1]
            or value != 1.0 + 0.0j
        ):
            raise KrennNumericalContinuationError(
                "numerical gauge-chart contents are not canonical"
            )
        matrix = _gauge_exponents(self.action_group)
        support_rank = _rank_over_q(
            tuple(tuple(map(int, matrix[index])) for index in support)
        )
        anchor_rank = _rank_over_q(
            tuple(tuple(map(int, matrix[index])) for index in anchors)
        )
        if support_rank != rank or anchor_rank != rank:
            raise KrennNumericalContinuationError(
                "gauge anchors do not span the effective support gauge"
            )
        if self.kind == "natural-certified" and (
            anchors != NATURAL_GAUGE_ANCHORS
            or rank != 5
            or self.action_group
            != GHZ_VICTIM_COLOR_DIAGONAL_GAUGE
        ):
            raise KrennNumericalContinuationError(
                "the certified natural gauge chart changed"
            )
        if self.kind == "natural-extended" and (
            not set(NATURAL_GAUGE_ANCHORS).issubset(anchors)
            or not set(NATURAL_GAUGE_ANCHORS).issubset(support)
        ):
            raise KrennNumericalContinuationError(
                "the extended natural gauge chart lost its certified anchors"
            )

    def to_dict(self) -> dict:
        matrix = _gauge_exponents(self.action_group)
        return {
            "schema": self.schema,
            "kind": self.kind,
            "gauge_action_group": self.action_group,
            "ambient_gauge_parameter_dimension": matrix.shape[1],
            "preserves_direct_GHZ_target": True,
            "preserves_nonzero_victim_target": (
                self.action_group
                != GHZ_COLOR_DIAGONAL_GAUGE
            ),
            "support": list(self.support),
            "support_size": len(self.support),
            "effective_gauge_action_rank": self.rank,
            "anchors": list(self.anchors),
            "anchor_variable_keys": [
                list(variable_key(N, D, index)) for index in self.anchors
            ],
            "anchor_exponent_rows": [
                list(map(int, matrix[index])) for index in self.anchors
            ],
            "anchor_value": _complex_pair(self.anchor_value),
            "free_indices": list(self.free_indices),
            "free_complex_dimension": len(self.free_indices),
            "symmetry_weight_equalities": 0,
        }


def support_gauge_chart(
    support: Iterable[int],
    *,
    prefer_natural: bool = False,
    action_group: str = GHZ_COLOR_DIAGONAL_GAUGE,
) -> GaugeChart:
    """Construct a deterministic maximal-rank chart for ``support``."""

    canonical = _canonical_support(support)
    matrix = _gauge_exponents(action_group)
    support_rank = _rank_over_q(
        tuple(
            tuple(map(int, matrix[index]))
            for index in canonical
        )
    )
    use_natural = (
        prefer_natural
        and set(NATURAL_GAUGE_ANCHORS).issubset(canonical)
    )
    anchors = _deterministic_anchor_basis(
        canonical,
        matrix,
        preferred=NATURAL_GAUGE_ANCHORS if use_natural else (),
    )
    kind = "support-adaptive"
    if use_natural:
        kind = (
            "natural-certified"
            if (
                action_group
                == GHZ_VICTIM_COLOR_DIAGONAL_GAUGE
                and anchors == NATURAL_GAUGE_ANCHORS
                and support_rank == 5
            )
            else "natural-extended"
        )
    return GaugeChart(
        support=canonical,
        anchors=anchors,
        free_indices=tuple(
            index for index in canonical if index not in anchors
        ),
        rank=len(anchors),
        kind=kind,
        action_group=action_group,
    )


def natural_gauge_chart(
    support: Iterable[int] = tuple(range(AMBIENT_VARIABLES)),
    *,
    action_group: str = GHZ_VICTIM_COLOR_DIAGONAL_GAUGE,
) -> GaugeChart:
    """Return a target-aware chart retaining the five natural anchors."""

    canonical = _canonical_support(support)
    if not set(NATURAL_GAUGE_ANCHORS).issubset(canonical):
        raise KrennNumericalContinuationError(
            "the natural gauge chart needs all five certified anchors"
        )
    return support_gauge_chart(
        canonical,
        prefer_natural=True,
        action_group=action_group,
    )


@dataclass(frozen=True)
class ProjectionReport:
    """Diagnostics for one hard support/norm projection."""

    before_l2: float | None
    before_linf: float | None
    after_l2: float | None
    after_linf: float | None
    linf_clipped_coordinates: int
    l2_scaled: bool
    support_zeroed_coordinates: int
    anchor_resets: int
    on_l2_boundary: bool
    on_linf_boundary: bool

    @property
    def changed(self) -> bool:
        return bool(
            self.linf_clipped_coordinates
            or self.l2_scaled
            or self.support_zeroed_coordinates
            or self.anchor_resets
        )

    def to_dict(self) -> dict:
        return {
            "before_l2": self.before_l2,
            "before_linf": self.before_linf,
            "after_l2": self.after_l2,
            "after_linf": self.after_linf,
            "linf_clipped_coordinates": self.linf_clipped_coordinates,
            "l2_scaled": self.l2_scaled,
            "support_zeroed_coordinates": (
                self.support_zeroed_coordinates
            ),
            "anchor_resets": self.anchor_resets,
            "on_l2_boundary": self.on_l2_boundary,
            "on_linf_boundary": self.on_linf_boundary,
            "changed": self.changed,
        }


def project_hard_bounds(
    weights: Sequence[complex] | np.ndarray,
    chart: GaugeChart,
    *,
    l2_bound: float,
    linf_bound: float,
) -> tuple[np.ndarray, ProjectionReport]:
    """Project into the declared chart and hard ``L2``/``Linf`` bounds.

    The operation first clips each free complex coordinate to its closed disk
    and then radially scales all free coordinates to the remaining L2 budget.
    It is deterministic and guarantees feasibility; no optimal-projection
    claim is needed by the solver.
    """

    if not isinstance(chart, GaugeChart):
        raise KrennNumericalContinuationError(
            "hard projection requires a canonical GaugeChart"
        )
    l2_bound = float(l2_bound)
    linf_bound = float(linf_bound)
    if (
        not math.isfinite(l2_bound)
        or not math.isfinite(linf_bound)
        or l2_bound <= 0.0
        or linf_bound < 1.0
        or l2_bound * l2_bound + 1.0e-14 < chart.rank
    ):
        raise KrennNumericalContinuationError(
            "hard bounds cannot contain the fixed gauge anchors"
        )
    dense = _dense_weights(weights)
    before_abs = np.abs(dense)
    before_l2 = float(np.linalg.norm(dense))
    before_linf = float(before_abs.max(initial=0.0))

    support_mask = np.zeros(AMBIENT_VARIABLES, dtype=bool)
    support_mask[np.asarray(chart.support, dtype=np.intp)] = True
    support_zeroed = int(np.count_nonzero(dense[~support_mask]))
    dense[~support_mask] = 0.0

    anchor_array = np.asarray(chart.anchors, dtype=np.intp)
    anchor_resets = int(
        np.count_nonzero(dense[anchor_array] != chart.anchor_value)
    )
    dense[anchor_array] = chart.anchor_value

    free = np.asarray(chart.free_indices, dtype=np.intp)
    free_values = dense[free]
    magnitudes = np.abs(free_values)
    clipped = magnitudes > linf_bound
    clipped_count = int(np.count_nonzero(clipped))
    if clipped_count:
        free_values = free_values.copy()
        free_values[clipped] *= (
            linf_bound / magnitudes[clipped]
        )
        dense[free] = free_values

    free_l2 = float(np.linalg.norm(dense[free]))
    available_squared = max(
        0.0,
        l2_bound * l2_bound
        - chart.rank * abs(chart.anchor_value) ** 2,
    )
    available = math.sqrt(available_squared)
    l2_scaled = free_l2 > available and free_l2 > 0.0
    if l2_scaled:
        dense[free] *= available / free_l2

    after_abs = np.abs(dense)
    after_l2 = float(np.linalg.norm(dense))
    after_linf = float(after_abs.max(initial=0.0))
    tolerance = 64.0 * np.finfo(np.float64).eps
    if (
        after_l2 > l2_bound * (1.0 + tolerance)
        or after_linf > linf_bound * (1.0 + tolerance)
        or np.count_nonzero(dense[~support_mask])
        or np.any(dense[anchor_array] != chart.anchor_value)
    ):
        raise KrennNumericalContinuationError(
            "hard projection failed its postcondition"
        )
    return dense, ProjectionReport(
        before_l2=_finite_or_none(before_l2),
        before_linf=_finite_or_none(before_linf),
        after_l2=_finite_or_none(after_l2),
        after_linf=_finite_or_none(after_linf),
        linf_clipped_coordinates=clipped_count,
        l2_scaled=l2_scaled,
        support_zeroed_coordinates=support_zeroed,
        anchor_resets=anchor_resets,
        on_l2_boundary=bool(
            after_l2 >= l2_bound * (1.0 - 1.0e-12)
        ),
        on_linf_boundary=bool(
            after_linf >= linf_bound * (1.0 - 1.0e-12)
        ),
    )


@dataclass(frozen=True)
class ResidualMetrics:
    """Complete 729-equation floating residual accounting."""

    finite: bool
    nonfinite_count: int
    l2_squared: float | None
    l2: float | None
    rms: float | None
    linf: float | None
    constant_linf: float | None
    mixed_linf: float | None
    counts_above_thresholds: tuple[tuple[float, int], ...]
    top_residuals: tuple[
        tuple[int, tuple[int, ...], float | None, complex], ...
    ]

    @classmethod
    def from_residual(
        cls,
        residual: Sequence[complex] | np.ndarray,
        *,
        top_count: int = 8,
    ) -> "ResidualMetrics":
        values = np.asarray(residual, dtype=np.complex128)
        if values.shape != (EQUATION_COUNT,):
            raise KrennNumericalContinuationError(
                "residual metrics require all 729 equations"
            )
        finite_mask = np.isfinite(values.real) & np.isfinite(values.imag)
        nonfinite = int(values.size - np.count_nonzero(finite_mask))
        magnitudes = np.abs(values)
        finite_magnitudes = magnitudes[finite_mask]
        if nonfinite:
            squared = l2 = rms = linf = None
        else:
            squared_value = math.fsum(
                float(value.real) * float(value.real)
                + float(value.imag) * float(value.imag)
                for value in values
            )
            squared = _finite_or_none(squared_value)
            l2_value = math.sqrt(squared_value)
            l2 = _finite_or_none(l2_value)
            rms = _finite_or_none(
                l2_value / math.sqrt(EQUATION_COUNT)
            )
            linf = _finite_or_none(
                float(finite_magnitudes.max(initial=0.0))
            )
        constant_values = magnitudes[
            np.asarray(CONSTANT_EQUATIONS, dtype=np.intp)
        ]
        mixed_mask = np.ones(EQUATION_COUNT, dtype=bool)
        mixed_mask[np.asarray(CONSTANT_EQUATIONS, dtype=np.intp)] = False
        mixed_values = magnitudes[mixed_mask]
        constant_linf = (
            _finite_or_none(float(constant_values.max(initial=0.0)))
            if np.all(np.isfinite(constant_values))
            else None
        )
        mixed_linf = (
            _finite_or_none(float(mixed_values.max(initial=0.0)))
            if np.all(np.isfinite(mixed_values))
            else None
        )
        counts = tuple(
            (
                threshold,
                int(np.count_nonzero(magnitudes > threshold)),
            )
            for threshold in RESIDUAL_THRESHOLDS
        )
        ranking = sorted(
            range(EQUATION_COUNT),
            key=lambda equation: (
                -(
                    float(magnitudes[equation])
                    if math.isfinite(float(magnitudes[equation]))
                    else math.inf
                ),
                equation,
            ),
        )[: max(0, int(top_count))]
        top = tuple(
            (
                equation,
                coloring_from_index(N, D, equation),
                _finite_or_none(float(magnitudes[equation])),
                complex(values[equation]),
            )
            for equation in ranking
        )
        return cls(
            finite=not nonfinite,
            nonfinite_count=nonfinite,
            l2_squared=squared,
            l2=l2,
            rms=rms,
            linf=linf,
            constant_linf=constant_linf,
            mixed_linf=mixed_linf,
            counts_above_thresholds=counts,
            top_residuals=top,
        )

    def to_dict(self) -> dict:
        return {
            "all_729_equations_accounted": True,
            "finite": self.finite,
            "nonfinite_count": self.nonfinite_count,
            "l2_squared": self.l2_squared,
            "l2": self.l2,
            "rms": self.rms,
            "linf": self.linf,
            "constant_equation_linf": self.constant_linf,
            "mixed_equation_linf": self.mixed_linf,
            "counts_above_thresholds": {
                format(threshold, ".0e"): count
                for threshold, count in self.counts_above_thresholds
            },
            "top_residuals": [
                {
                    "equation": equation,
                    "coloring": list(coloring),
                    "absolute_value": absolute_value,
                    "value": _complex_pair(value),
                }
                for equation, coloring, absolute_value, value
                in self.top_residuals
            ],
        }


@dataclass(frozen=True)
class WeightMetrics:
    """Finite-chart norm and effective-support diagnostics."""

    finite: bool
    nonfinite_count: int
    support_size: int
    l2: float | None
    linf: float | None
    min_supported_magnitude: float | None
    median_supported_magnitude: float | None
    effective_support_counts: tuple[tuple[float, int], ...]
    l2_bound_fraction: float | None
    linf_bound_fraction: float | None
    anchor_error_linf: float | None
    on_hard_boundary: bool

    @classmethod
    def from_weights(
        cls,
        weights: Sequence[complex] | np.ndarray,
        chart: GaugeChart,
        *,
        l2_bound: float,
        linf_bound: float,
    ) -> "WeightMetrics":
        dense = _dense_weights(weights)
        supported = dense[np.asarray(chart.support, dtype=np.intp)]
        finite_mask = (
            np.isfinite(supported.real) & np.isfinite(supported.imag)
        )
        nonfinite = int(
            supported.size - np.count_nonzero(finite_mask)
        )
        magnitudes = np.abs(supported)
        if nonfinite:
            l2 = linf = minimum = median = None
            l2_fraction = linf_fraction = None
            boundary = True
        else:
            l2_value = float(np.linalg.norm(supported))
            linf_value = float(magnitudes.max(initial=0.0))
            l2 = _finite_or_none(l2_value)
            linf = _finite_or_none(linf_value)
            minimum = _finite_or_none(float(magnitudes.min()))
            median = _finite_or_none(float(np.median(magnitudes)))
            l2_fraction = _finite_or_none(l2_value / l2_bound)
            linf_fraction = _finite_or_none(
                linf_value / linf_bound
            )
            boundary = bool(
                l2_value >= l2_bound * (1.0 - 1.0e-10)
                or linf_value >= linf_bound * (1.0 - 1.0e-10)
            )
        anchors = dense[np.asarray(chart.anchors, dtype=np.intp)]
        anchor_error = (
            _finite_or_none(
                float(
                    np.abs(anchors - chart.anchor_value).max(
                        initial=0.0
                    )
                )
            )
            if np.all(np.isfinite(anchors))
            else None
        )
        return cls(
            finite=not nonfinite,
            nonfinite_count=nonfinite,
            support_size=len(chart.support),
            l2=l2,
            linf=linf,
            min_supported_magnitude=minimum,
            median_supported_magnitude=median,
            effective_support_counts=tuple(
                (
                    threshold,
                    int(np.count_nonzero(magnitudes > threshold)),
                )
                for threshold in RESIDUAL_THRESHOLDS
            ),
            l2_bound_fraction=l2_fraction,
            linf_bound_fraction=linf_fraction,
            anchor_error_linf=anchor_error,
            on_hard_boundary=boundary,
        )

    def to_dict(self) -> dict:
        return {
            "finite": self.finite,
            "weights_remain_finite": self.finite,
            "nonfinite_count": self.nonfinite_count,
            "declared_support_size": self.support_size,
            "l2": self.l2,
            "linf": self.linf,
            "min_supported_magnitude": self.min_supported_magnitude,
            "median_supported_magnitude": (
                self.median_supported_magnitude
            ),
            "effective_support_counts": {
                format(threshold, ".0e"): count
                for threshold, count in self.effective_support_counts
            },
            "l2_bound_fraction": self.l2_bound_fraction,
            "linf_bound_fraction": self.linf_bound_fraction,
            "anchor_error_linf": self.anchor_error_linf,
            "on_hard_boundary": self.on_hard_boundary,
        }


@dataclass(frozen=True)
class InvariantMetrics:
    """Gauge-invariant perfect-matching and known-pole diagnostics."""

    finite: bool
    nonfinite_matching_products: int
    nonzero_matching_products: int
    matching_product_min_nonzero: float | None
    matching_product_max: float | None
    matching_product_log_span: float | None
    constant_matching_products: tuple[tuple[complex, ...], ...]
    known_q_defined_on_support: bool
    known_q_value: complex | None
    known_q_absolute_value: float | None
    known_q_log_absolute_value: float | None
    known_q_phase: float | None

    @classmethod
    def from_weights(
        cls,
        weights: Sequence[complex] | np.ndarray,
        support: Sequence[int],
    ) -> "InvariantMetrics":
        dense = _dense_weights(weights)
        monomials = monomial_index_array()
        with np.errstate(over="ignore", invalid="ignore"):
            factors = dense[monomials]
            products = (
                factors[:, 0] * factors[:, 1] * factors[:, 2]
            )
        product_finite = (
            np.isfinite(products.real) & np.isfinite(products.imag)
        )
        nonfinite = int(
            products.size - np.count_nonzero(product_finite)
        )
        magnitudes = np.abs(products)
        nonzero_mask = product_finite & (magnitudes > 0.0)
        nonzero = int(np.count_nonzero(nonzero_mask))
        if nonzero:
            minimum_value = float(magnitudes[nonzero_mask].min())
            maximum_value = float(magnitudes[nonzero_mask].max())
            log_span_value = math.log(maximum_value) - math.log(
                minimum_value
            )
            minimum = _finite_or_none(minimum_value)
            maximum = _finite_or_none(maximum_value)
            log_span = _finite_or_none(log_span_value)
        else:
            minimum = maximum = log_span = None
        constant_products = tuple(
            tuple(
                complex(value)
                for value in products[
                    equation
                    * MATCHINGS_PER_EQUATION : (
                        equation + 1
                    )
                    * MATCHINGS_PER_EQUATION
                ]
            )
            for equation in CONSTANT_EQUATIONS
        )
        q_defined = set(KNOWN_Q_INDICES).issubset(support)
        q_value: complex | None
        if q_defined:
            with np.errstate(over="ignore", invalid="ignore"):
                q_value = complex(
                    np.prod(
                        dense[
                            np.asarray(
                                KNOWN_Q_INDICES, dtype=np.intp
                            )
                        ]
                    )
                )
        else:
            q_value = None
        if q_value is not None and (
            math.isfinite(q_value.real)
            and math.isfinite(q_value.imag)
        ):
            q_absolute_value = abs(q_value)
            q_absolute = _finite_or_none(q_absolute_value)
            q_log = (
                _finite_or_none(math.log(q_absolute_value))
                if q_absolute_value > 0.0
                else None
            )
            q_phase = _finite_or_none(math.atan2(
                q_value.imag, q_value.real
            ))
        else:
            q_absolute = q_log = q_phase = None
        q_finite = q_value is None or (
            math.isfinite(q_value.real)
            and math.isfinite(q_value.imag)
        )
        return cls(
            finite=not nonfinite and q_finite,
            nonfinite_matching_products=nonfinite,
            nonzero_matching_products=nonzero,
            matching_product_min_nonzero=minimum,
            matching_product_max=maximum,
            matching_product_log_span=log_span,
            constant_matching_products=constant_products,
            known_q_defined_on_support=q_defined,
            known_q_value=q_value,
            known_q_absolute_value=q_absolute,
            known_q_log_absolute_value=q_log,
            known_q_phase=q_phase,
        )

    def to_dict(self) -> dict:
        return {
            "finite": self.finite,
            "all_matching_products_are_vertex_gauge_invariant": True,
            "matching_product_count": (
                EQUATION_COUNT * MATCHINGS_PER_EQUATION
            ),
            "nonfinite_matching_products": (
                self.nonfinite_matching_products
            ),
            "nonzero_matching_products": self.nonzero_matching_products,
            "matching_product_min_nonzero": (
                self.matching_product_min_nonzero
            ),
            "matching_product_max": self.matching_product_max,
            "matching_product_log_span": (
                self.matching_product_log_span
            ),
            "constant_matching_products": [
                [_complex_pair(value) for value in row]
                for row in self.constant_matching_products
            ],
            "known_Q": {
                "indices": list(KNOWN_Q_INDICES),
                "defined_on_support": (
                    self.known_q_defined_on_support
                ),
                "value": (
                    _complex_pair(self.known_q_value)
                    if self.known_q_value is not None
                    else None
                ),
                "absolute_value": self.known_q_absolute_value,
                "log_absolute_value": (
                    self.known_q_log_absolute_value
                ),
                "phase": self.known_q_phase,
                "known_Laurent_value": "t^-1",
                "color_diagonal_character": list(
                    map(int, known_q_color_gauge_character())
                ),
                "vertex_scalar_gauge_invariant": True,
                "moving_target_color_diagonal_gauge_invariant": True,
                "direct_GHZ_full_color_diagonal_gauge_invariant": False,
                "unqualified_gauge_invariance_claimed": False,
            },
        }


@dataclass(frozen=True)
class LMPlan:
    """Explicit deterministic safety bounds for one complex LM solve."""

    max_iterations: int = 120
    max_evaluations: int = 900
    l2_bound: float = 16.0
    linf_bound: float = 8.0
    initial_damping: float = 1.0e-3
    damping_increase: float = 8.0
    damping_decrease: float = 0.35
    initial_trust_radius: float = 2.0
    minimum_trust_radius: float = 1.0e-10
    maximum_trust_radius: float = 8.0
    backtracking_steps: int = 10
    acceptance_ratio: float = 1.0e-4
    residual_tolerance: float = 1.0e-10
    gradient_tolerance: float = 1.0e-12
    step_tolerance: float = 1.0e-12
    checkpoint_interval: int = 10
    schema: str = LM_PLAN_SCHEMA

    def __post_init__(self) -> None:
        integer_fields = {
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "backtracking_steps": self.backtracking_steps,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, raw_value in integer_fields.items():
            object.__setattr__(self, name, int(raw_value))
        float_fields = (
            "l2_bound",
            "linf_bound",
            "initial_damping",
            "damping_increase",
            "damping_decrease",
            "initial_trust_radius",
            "minimum_trust_radius",
            "maximum_trust_radius",
            "acceptance_ratio",
            "residual_tolerance",
            "gradient_tolerance",
            "step_tolerance",
        )
        for name in float_fields:
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.schema != LM_PLAN_SCHEMA:
            raise KrennNumericalContinuationError(
                "complex LM plan schema changed"
            )
        if not 1 <= self.max_iterations <= 10_000:
            raise KrennNumericalContinuationError(
                "LM iteration cap is outside the safety bound"
            )
        if not 2 <= self.max_evaluations <= 100_000:
            raise KrennNumericalContinuationError(
                "LM evaluation cap is outside the safety bound"
            )
        if not 0 <= self.backtracking_steps <= 30:
            raise KrennNumericalContinuationError(
                "LM backtracking cap is outside the safety bound"
            )
        if not 0 <= self.checkpoint_interval <= self.max_iterations:
            raise KrennNumericalContinuationError(
                "checkpoint interval is outside the iteration cap"
            )
        positive = (
            self.l2_bound,
            self.linf_bound,
            self.initial_damping,
            self.initial_trust_radius,
            self.minimum_trust_radius,
            self.maximum_trust_radius,
            self.residual_tolerance,
            self.gradient_tolerance,
            self.step_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise KrennNumericalContinuationError(
                "LM positive controls must be finite"
            )
        if (
            self.linf_bound < 1.0
            or not self.damping_increase > 1.0
            or not 0.0 < self.damping_decrease < 1.0
            or not 0.0 <= self.acceptance_ratio < 1.0
            or self.minimum_trust_radius
            > self.initial_trust_radius
            or self.initial_trust_radius
            > self.maximum_trust_radius
        ):
            raise KrennNumericalContinuationError(
                "LM damping, trust, or acceptance controls are inconsistent"
            )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "l2_bound": self.l2_bound,
            "linf_bound": self.linf_bound,
            "initial_damping": self.initial_damping,
            "damping_increase": self.damping_increase,
            "damping_decrease": self.damping_decrease,
            "initial_trust_radius": self.initial_trust_radius,
            "minimum_trust_radius": self.minimum_trust_radius,
            "maximum_trust_radius": self.maximum_trust_radius,
            "backtracking_steps": self.backtracking_steps,
            "acceptance_ratio": self.acceptance_ratio,
            "residual_tolerance": self.residual_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
            "step_tolerance": self.step_tolerance,
            "checkpoint_interval": self.checkpoint_interval,
            "complex_weights": True,
            "hard_l2_control": True,
            "hard_linf_control": True,
            "symmetry_weight_equalities": 0,
        }


@dataclass(frozen=True)
class LMIterationRecord:
    iteration: int
    evaluations: int
    objective: float | None
    residual_linf: float | None
    gradient_linf: float | None
    step_norm: float | None
    damping: float
    trust_radius: float
    accepted: bool
    ratio: float | None
    backtracking_trials: int
    projection_changed: bool
    hard_boundary_active: bool

    def __post_init__(self) -> None:
        if (
            self.iteration < 1
            or self.evaluations < 1
            or not math.isfinite(self.damping)
            or self.damping <= 0.0
            or not math.isfinite(self.trust_radius)
            or self.trust_radius <= 0.0
            or self.backtracking_trials < 0
        ):
            raise KrennNumericalContinuationError(
                "LM iteration trace record is malformed"
            )

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "objective": self.objective,
            "residual_linf": self.residual_linf,
            "gradient_linf": self.gradient_linf,
            "step_norm": self.step_norm,
            "damping": self.damping,
            "trust_radius": self.trust_radius,
            "accepted": self.accepted,
            "ratio": self.ratio,
            "backtracking_trials": self.backtracking_trials,
            "projection_changed": self.projection_changed,
            "hard_boundary_active": self.hard_boundary_active,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "LMIterationRecord":
        expected = {
            "iteration",
            "evaluations",
            "objective",
            "residual_linf",
            "gradient_linf",
            "step_norm",
            "damping",
            "trust_radius",
            "accepted",
            "ratio",
            "backtracking_trials",
            "projection_changed",
            "hard_boundary_active",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise KrennNumericalContinuationError(
                "LM iteration trace payload changed"
            )

        def optional_float(name: str) -> float | None:
            value = payload[name]
            if value is None:
                return None
            result = float(value)
            if not math.isfinite(result):
                raise KrennNumericalContinuationError(
                    "LM iteration trace contains a nonfinite JSON number"
                )
            return result

        try:
            result = cls(
                iteration=int(payload["iteration"]),
                evaluations=int(payload["evaluations"]),
                objective=optional_float("objective"),
                residual_linf=optional_float("residual_linf"),
                gradient_linf=optional_float("gradient_linf"),
                step_norm=optional_float("step_norm"),
                damping=float(payload["damping"]),
                trust_radius=float(payload["trust_radius"]),
                accepted=bool(payload["accepted"]),
                ratio=optional_float("ratio"),
                backtracking_trials=int(
                    payload["backtracking_trials"]
                ),
                projection_changed=bool(
                    payload["projection_changed"]
                ),
                hard_boundary_active=bool(
                    payload["hard_boundary_active"]
                ),
            )
        except (TypeError, ValueError) as error:
            raise KrennNumericalContinuationError(
                "could not decode LM iteration trace"
            ) from error
        if result.to_dict() != dict(payload):
            raise KrennNumericalContinuationError(
                "LM iteration trace payload was not canonical"
            )
        return result


@dataclass(frozen=True)
class NumericalSolveResult:
    """One bounded numerical solve; never an exact certificate."""

    run_id: str
    support: tuple[int, ...]
    chart: GaugeChart
    plan: LMPlan
    target_amplitude: complex
    status: str
    iterations: int
    evaluations: int
    accepted_steps: int
    rejected_steps: int
    weights: tuple[complex, ...]
    residual_metrics: ResidualMetrics
    weight_metrics: WeightMetrics
    invariant_metrics: InvariantMetrics
    trace: tuple[LMIterationRecord, ...]
    projection_count: int
    schema: str = LM_RESULT_SCHEMA
    exact_solution_certified: bool = False
    nonexistence_proved: bool = False

    def __post_init__(self) -> None:
        support = _canonical_support(self.support)
        weights = tuple(map(complex, self.weights))
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self, "target_amplitude", complex(self.target_amplitude)
        )
        if self.schema != LM_RESULT_SCHEMA:
            raise KrennNumericalContinuationError(
                "complex LM result schema changed"
            )
        if self.status not in _LM_STATUSES:
            raise KrennNumericalContinuationError(
                "unknown complex LM termination status"
            )
        if (
            len(weights) != AMBIENT_VARIABLES
            or self.chart.support != support
            or self.exact_solution_certified
            or self.nonexistence_proved
            or self.iterations < 0
            or self.iterations > self.plan.max_iterations
            or self.evaluations < 1
            or self.evaluations > self.plan.max_evaluations
            or self.accepted_steps < 0
            or self.rejected_steps < 0
            or self.accepted_steps + self.rejected_steps
            != self.iterations
            or len(self.trace) != self.iterations
            or self.projection_count < 0
            or any(
                record.iteration != index
                for index, record in enumerate(self.trace, start=1)
            )
        ):
            raise KrennNumericalContinuationError(
                "complex LM result failed its bounded claim contract"
            )

    @property
    def numerical_zero(self) -> bool:
        return bool(
            self.residual_metrics.finite
            and self.residual_metrics.linf is not None
            and self.residual_metrics.linf
            <= self.plan.residual_tolerance
        )

    @property
    def weights_remain_finite(self) -> bool:
        return self.weight_metrics.finite

    @property
    def interior_bounded_candidate(self) -> bool:
        return bool(
            self.weights_remain_finite
            and not self.weight_metrics.on_hard_boundary
        )

    def dense_weights(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.complex128)

    def to_dict(self, *, include_trace: bool = True) -> dict:
        weight_entries = [
            {
                "index": index,
                "variable_key": list(variable_key(N, D, index)),
                "value": _complex_pair(self.weights[index]),
            }
            for index in self.support
        ]
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "parameters": {
                "n": N,
                "d": D,
                "ambient_variables": AMBIENT_VARIABLES,
                "equations": EQUATION_COUNT,
            },
            "support": list(self.support),
            "support_size": len(self.support),
            "gauge_chart": self.chart.to_dict(),
            "plan": self.plan.to_dict(),
            "target": {
                "canonical_GHZ_plus_victim_amplitude": (
                    _complex_pair(self.target_amplitude)
                ),
                "victim_equation": N6_D3_SEED_DEFECT_EQUATION,
                "is_direct_GHZ_target": self.target_amplitude == 0.0j,
            },
            "status": self.status,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "projection_count": self.projection_count,
            "configured_blas_thread_limit_per_process": (
                NUMERICAL_BLAS_THREAD_LIMIT
            ),
            "weights": weight_entries,
            "residual_metrics": self.residual_metrics.to_dict(),
            "weight_metrics": self.weight_metrics.to_dict(),
            "gauge_invariant_metrics": self.invariant_metrics.to_dict(),
            "trace": (
                [record.to_dict() for record in self.trace]
                if include_trace
                else None
            ),
            "numerical_zero_under_declared_tolerance": (
                self.numerical_zero
            ),
            "weights_remain_finite": self.weights_remain_finite,
            "interior_bounded_candidate": (
                self.interior_bounded_candidate
            ),
            "exact_solution_certified": False,
            "claim_boundary": {
                "bounded_deterministic_numerical_search_only": True,
                "all_729_residuals_evaluated": True,
                "fixed_gauge_chart": True,
                "hard_l2_and_linf_controls": True,
                "numerical_zero_is_exact_witness": False,
                "exact_verification_performed": False,
                "bounded_miss_proves_nonexistence": False,
                "finite_field_transfer_used": False,
                "nonexistence_proved": False,
            },
        }


@dataclass(frozen=True)
class InitializerRecord:
    """A deterministic bounded complex initialization."""

    kind: str
    support: tuple[int, ...]
    chart: GaugeChart
    weights: tuple[complex, ...]
    seed_words: tuple[int, ...]
    activation_indices: tuple[int, ...]
    activation_amplitude: float
    laurent_t: float | None
    projection: ProjectionReport

    def dense_weights(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.complex128)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "support": list(self.support),
            "gauge_chart": self.chart.to_dict(),
            "seed_words": list(self.seed_words),
            "activation_indices": list(self.activation_indices),
            "activation_amplitude": self.activation_amplitude,
            "laurent_t": self.laurent_t,
            "projection": self.projection.to_dict(),
            "finite_weights": bool(
                np.all(np.isfinite(self.dense_weights()))
            ),
            "exact_solution_claimed": False,
        }


def _seed_sequence(
    seed: int | Sequence[int] | np.random.SeedSequence,
) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed
    try:
        return np.random.SeedSequence(seed)
    except (TypeError, ValueError) as error:
        raise KrennNumericalContinuationError(
            "initializer seed is not accepted by NumPy SeedSequence"
        ) from error


def _seed_words(sequence: np.random.SeedSequence) -> tuple[int, ...]:
    return tuple(
        map(int, sequence.generate_state(4, dtype=np.uint32))
    )


def natural_laurent_initializer(
    t: float,
    *,
    support: Iterable[int] = NATURAL_SUPPORT,
    chart: GaugeChart | None = None,
    l2_bound: float = 16.0,
    linf_bound: float = 8.0,
) -> InitializerRecord:
    """Return the bounded projection of the exact known Laurent control."""

    t = float(t)
    canonical = _canonical_support(support)
    if (
        not math.isfinite(t)
        or t == 0.0
        or not NATURAL_SUPPORT_SET.issubset(canonical)
    ):
        raise KrennNumericalContinuationError(
            "Laurent initialization needs nonzero finite t and natural support"
        )
    chart = (
        natural_gauge_chart(canonical)
        if chart is None
        else chart
    )
    if (
        chart.support != canonical
        or chart.action_group
        != GHZ_VICTIM_COLOR_DIAGONAL_GAUGE
        or not set(NATURAL_GAUGE_ANCHORS).issubset(chart.anchors)
    ):
        raise KrennNumericalContinuationError(
            "Laurent initialization requires the moving-target natural chart"
        )
    weights = np.zeros(AMBIENT_VARIABLES, dtype=np.complex128)
    for index, value in n6_d3_seed_witness().entries:
        weights[index] = complex(value)
    weights[REPAIR_DECREASE_INDEX] = complex(t)
    weights[REPAIR_INCREASE_INDEX] = complex(1.0 / t)
    projected, report = project_hard_bounds(
        weights,
        chart,
        l2_bound=l2_bound,
        linf_bound=linf_bound,
    )
    return InitializerRecord(
        kind="known-laurent-control",
        support=canonical,
        chart=chart,
        weights=tuple(map(complex, projected)),
        seed_words=(),
        activation_indices=(),
        activation_amplitude=0.0,
        laurent_t=t,
        projection=report,
    )


def transverse_activated_initializer(
    support: Iterable[int],
    seed: int | Sequence[int] | np.random.SeedSequence,
    *,
    activation_amplitude: float = 0.1,
    laurent_t: float = 1.0,
    activation_indices: Iterable[int] | None = None,
    chart: GaugeChart | None = None,
    l2_bound: float = 16.0,
    linf_bound: float = 8.0,
) -> InitializerRecord:
    """Perturb the Laurent control in declared repair coordinates.

    The activation phases are independent.  Symmetry-related coordinates are
    never tied.  Final optimization does not constrain activated weights to
    remain nonzero.
    """

    canonical = _canonical_support(support)
    if not NATURAL_SUPPORT_SET.issubset(canonical):
        raise KrennNumericalContinuationError(
            "transverse Laurent activation needs all natural coordinates"
        )
    amplitude = float(activation_amplitude)
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise KrennNumericalContinuationError(
            "activation amplitude must be positive and finite"
        )
    chart = (
        natural_gauge_chart(canonical)
        if chart is None
        else chart
    )
    base = natural_laurent_initializer(
        laurent_t,
        support=canonical,
        chart=chart,
        l2_bound=max(l2_bound, 2.0 * abs(1.0 / laurent_t)),
        linf_bound=max(linf_bound, 2.0 * abs(1.0 / laurent_t)),
    ).dense_weights()
    if activation_indices is None:
        activated = tuple(
            index
            for index in canonical
            if index not in NATURAL_SUPPORT_SET
            and index not in chart.anchors
        )
    else:
        activated = tuple(sorted(map(int, activation_indices)))
        if (
            len(activated) != len(set(activated))
            or not set(activated).issubset(canonical)
            or set(activated).intersection(chart.anchors)
        ):
            raise KrennNumericalContinuationError(
                "activation indices must be distinct free support coordinates"
            )
    if not activated:
        raise KrennNumericalContinuationError(
            "transverse initializer has no repair coordinate to activate"
        )
    sequence = _seed_sequence(seed)
    words = _seed_words(sequence)
    generator = np.random.default_rng(sequence)
    phases = generator.uniform(0.0, 2.0 * math.pi, len(activated))
    base[np.asarray(activated, dtype=np.intp)] = amplitude * np.exp(
        1.0j * phases
    )
    projected, report = project_hard_bounds(
        base,
        chart,
        l2_bound=l2_bound,
        linf_bound=linf_bound,
    )
    return InitializerRecord(
        kind="transverse-laurent-activation",
        support=canonical,
        chart=chart,
        weights=tuple(map(complex, projected)),
        seed_words=words,
        activation_indices=activated,
        activation_amplitude=amplitude,
        laurent_t=float(laurent_t),
        projection=report,
    )


def random_bounded_initializer(
    support: Iterable[int],
    seed: int | Sequence[int] | np.random.SeedSequence,
    *,
    chart: GaugeChart | None = None,
    scale: float = 0.5,
    l2_bound: float = 16.0,
    linf_bound: float = 8.0,
) -> InitializerRecord:
    """Return an iid complex deterministic start in a fixed support chart."""

    canonical = _canonical_support(support)
    chart = support_gauge_chart(canonical) if chart is None else chart
    if chart.support != canonical:
        raise KrennNumericalContinuationError(
            "random initializer chart belongs to another support"
        )
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise KrennNumericalContinuationError(
            "random initializer scale must be positive and finite"
        )
    sequence = _seed_sequence(seed)
    words = _seed_words(sequence)
    generator = np.random.default_rng(sequence)
    weights = np.zeros(AMBIENT_VARIABLES, dtype=np.complex128)
    free = np.asarray(chart.free_indices, dtype=np.intp)
    weights[free] = scale * (
        generator.standard_normal(len(free))
        + 1.0j * generator.standard_normal(len(free))
    ) / math.sqrt(2.0)
    weights[np.asarray(chart.anchors, dtype=np.intp)] = (
        chart.anchor_value
    )
    projected, report = project_hard_bounds(
        weights,
        chart,
        l2_bound=l2_bound,
        linf_bound=linf_bound,
    )
    return InitializerRecord(
        kind="random-bounded",
        support=canonical,
        chart=chart,
        weights=tuple(map(complex, projected)),
        seed_words=words,
        activation_indices=chart.free_indices,
        activation_amplitude=scale,
        laurent_t=None,
        projection=report,
    )


def projective_victim_repair_initializer(
    support: Iterable[int],
    seed: int | Sequence[int] | np.random.SeedSequence,
    *,
    numerator_indices: Sequence[int],
    denominator_indices: Sequence[int],
    expected_active_victim_matching_indices: Sequence[int],
    repair_scale: float = 0.1,
    chart: GaugeChart | None = None,
    l2_bound: float = 8.0,
    linf_bound: float = 8.0,
) -> InitializerRecord:
    """Initialize a direct-GHZ search on one projective victim branch.

    The two selected victim matching amplitudes have ratio

    ``r_N = prod(numerator_indices) / prod(denominator_indices)``.

    Gauge anchors are fixed before any randomization.  All remaining repair
    coordinates are assigned independent deterministic phases, after which
    one *free* relation coordinate is solved for so that ``r_N = -1``.
    Hard bounds are then checked without permitting clipping or radial
    rescaling, either of which could silently destroy the relation.

    This is an initialization condition only.  The subsequent direct-GHZ
    solve is free to leave the projective branch.
    """

    canonical = _canonical_support(support)
    if not NATURAL_SUPPORT_SET.issubset(canonical):
        raise KrennNumericalContinuationError(
            "projective victim initialization needs all natural coordinates"
        )
    numerator = tuple(map(int, numerator_indices))
    denominator = tuple(map(int, denominator_indices))
    expected_active = tuple(
        sorted(map(int, expected_active_victim_matching_indices))
    )
    relation_indices = (*numerator, *denominator)
    if (
        len(numerator) != 2
        or len(denominator) != 2
        or len(set(relation_indices)) != 4
        or not set(relation_indices).issubset(canonical)
        or set(numerator).intersection(NATURAL_SUPPORT_SET)
    ):
        raise KrennNumericalContinuationError(
            "projective victim relation must be a two-by-two branch "
            "inside the declared support"
        )
    if (
        len(expected_active) != 2
        or len(set(expected_active)) != 2
        or any(index < 0 or index >= MATCHINGS_PER_EQUATION for index in expected_active)
    ):
        raise KrennNumericalContinuationError(
            "projective victim initialization needs two distinguished "
            "active victim matchings"
        )
    support_set = set(canonical)
    active_victim = tuple(
        matching
        for matching, monomial in enumerate(
            _system().equation_monomials(
                N6_D3_SEED_DEFECT_EQUATION
            )
        )
        if set(monomial).issubset(support_set)
    )
    if active_victim != expected_active:
        raise KrennNumericalContinuationError(
            "projective support activates an unexpected victim matching; "
            "a multinomial initializer is required"
        )
    scale = float(repair_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise KrennNumericalContinuationError(
            "projective repair scale must be positive and finite"
        )
    chart = (
        natural_gauge_chart(
            canonical,
            action_group=GHZ_COLOR_DIAGONAL_GAUGE,
        )
        if chart is None
        else chart
    )
    if (
        chart.support != canonical
        or chart.action_group != GHZ_COLOR_DIAGONAL_GAUGE
        or not set(NATURAL_GAUGE_ANCHORS).issubset(chart.anchors)
    ):
        raise KrennNumericalContinuationError(
            "projective victim initialization requires a direct-GHZ "
            "natural gauge chart"
        )

    free_set = set(chart.free_indices)
    pivot = next(
        (index for index in numerator if index in free_set),
        next(
            (
                index
                for index in denominator
                if index in free_set
            ),
            None,
        ),
    )
    if pivot is None:
        raise KrennNumericalContinuationError(
            "projective victim relation has no free coordinate"
        )

    sequence = _seed_sequence(seed)
    words = _seed_words(sequence)
    generator = np.random.default_rng(sequence)
    weights = np.zeros(AMBIENT_VARIABLES, dtype=np.complex128)
    for index, value in n6_d3_seed_witness().entries:
        weights[index] = complex(value)
    anchor_array = np.asarray(chart.anchors, dtype=np.intp)
    weights[anchor_array] = chart.anchor_value

    relation_set = set(relation_indices)
    randomized_repairs = tuple(
        index
        for index in chart.free_indices
        if index not in NATURAL_SUPPORT_SET
        and index not in relation_set
    )
    random_phases = generator.uniform(
        0.0, 2.0 * math.pi, len(randomized_repairs)
    )
    if randomized_repairs:
        weights[
            np.asarray(randomized_repairs, dtype=np.intp)
        ] = scale * np.exp(1.0j * random_phases)

    free_relation = tuple(
        index
        for index in relation_indices
        if index in free_set and index != pivot
    )
    relation_phases = generator.uniform(
        0.0, 2.0 * math.pi, len(free_relation)
    )
    for index, phase in zip(
        free_relation, relation_phases, strict=True
    ):
        if index not in NATURAL_SUPPORT_SET:
            weights[index] = np.exp(1.0j * phase)

    def product(indices: Sequence[int]) -> complex:
        result = 1.0 + 0.0j
        for index in indices:
            result *= complex(weights[index])
        return result

    if pivot in numerator:
        other = tuple(index for index in numerator if index != pivot)
        divisor = product(other)
        target_product = -product(denominator)
    else:
        other = tuple(index for index in denominator if index != pivot)
        divisor = product(other)
        target_product = -product(numerator)
    if divisor == 0.0j:
        raise KrennNumericalContinuationError(
            "projective victim relation divisor vanished"
        )
    weights[pivot] = target_product / divisor

    projected, report = project_hard_bounds(
        weights,
        chart,
        l2_bound=l2_bound,
        linf_bound=linf_bound,
    )
    if report.changed:
        raise KrennNumericalContinuationError(
            "projective victim initialization exceeded hard bounds; "
            "resample or enlarge the declared bounds"
        )
    numerator_product = product(numerator)
    denominator_product = product(denominator)
    relation_error = numerator_product + denominator_product
    relation_scale = max(
        1.0, abs(numerator_product), abs(denominator_product)
    )
    if (
        denominator_product == 0.0j
        or abs(relation_error)
        > 128.0 * np.finfo(np.float64).eps * relation_scale
    ):
        raise KrennNumericalContinuationError(
            "projective victim relation failed numerical replay"
        )
    victim_amplitude = complex_output(projected)[
        N6_D3_SEED_DEFECT_EQUATION
    ]
    if abs(victim_amplitude) > (
        256.0 * np.finfo(np.float64).eps * relation_scale
    ):
        raise KrennNumericalContinuationError(
            "projective victim initializer did not reach the full victim "
            "hyperplane"
        )

    activation_indices = tuple(
        index
        for index in chart.free_indices
        if index not in NATURAL_SUPPORT_SET
    )
    return InitializerRecord(
        kind="projective-victim-repair",
        support=canonical,
        chart=chart,
        weights=tuple(map(complex, projected)),
        seed_words=words,
        activation_indices=activation_indices,
        activation_amplitude=scale,
        laurent_t=None,
        projection=report,
    )


def _validate_scratch_prefix(prefix: Path | str) -> Path:
    raw = Path(prefix)
    if not raw.is_absolute():
        raise KrennNumericalContinuationError(
            "checkpoint prefix must be an absolute non-OneDrive path"
        )
    resolved = raw.resolve(strict=False)
    if "onedrive" in str(resolved).casefold() or any(
        part.casefold() == "onedrive" for part in resolved.parts
    ):
        raise KrennNumericalContinuationError(
            "numerical checkpoints must not use OneDrive"
        )
    if resolved.suffix.casefold() in {".json", ".npz"}:
        resolved = resolved.with_suffix("")
    return resolved


def _checkpoint_paths(prefix: Path | str) -> tuple[Path, Path]:
    canonical = _validate_scratch_prefix(prefix)
    return canonical.with_suffix(".json"), canonical.with_suffix(".npz")


@dataclass(frozen=True)
class LMCheckpoint:
    """Round-trippable metadata for a numerical optimizer state."""

    run_id: str
    problem_sha256: str
    plan_sha256: str
    support: tuple[int, ...]
    gauge_anchors: tuple[int, ...]
    target_amplitude: complex
    iteration: int
    evaluations: int
    damping: float
    trust_radius: float
    status: str
    objective: float | None
    best_objective: float | None
    accepted_steps: int
    rejected_steps: int
    projection_count: int
    trace: tuple[LMIterationRecord, ...]
    npz_file: str
    npz_sha256: str
    schema: str = CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        trace = tuple(self.trace)
        object.__setattr__(self, "trace", trace)
        if self.schema != CHECKPOINT_SCHEMA:
            raise KrennNumericalContinuationError(
                "complex LM checkpoint schema changed"
            )
        if self.status not in _LM_STATUSES | {"running"}:
            raise KrennNumericalContinuationError(
                "unknown checkpoint optimizer status"
            )
        _canonical_support(self.support)
        if (
            not self.run_id
            or len(self.problem_sha256) != 64
            or len(self.plan_sha256) != 64
            or len(self.npz_sha256) != 64
            or not self.npz_file.endswith(".npz")
            or Path(self.npz_file).name != self.npz_file
            or self.iteration < 0
            or self.evaluations < 1
            or self.accepted_steps < 0
            or self.rejected_steps < 0
            or self.accepted_steps + self.rejected_steps
            != self.iteration
            or self.projection_count < 0
            or len(trace) != self.iteration
            or any(
                record.iteration != index
                for index, record in enumerate(trace, start=1)
            )
            or not math.isfinite(self.damping)
            or not math.isfinite(self.trust_radius)
        ):
            raise KrennNumericalContinuationError(
                "complex LM checkpoint metadata is malformed"
            )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "system_fingerprint": system_fingerprint(),
            "problem_sha256": self.problem_sha256,
            "plan_sha256": self.plan_sha256,
            "support": list(self.support),
            "gauge_anchors": list(self.gauge_anchors),
            "target_amplitude": _complex_pair(self.target_amplitude),
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "damping": self.damping,
            "trust_radius": self.trust_radius,
            "status": self.status,
            "objective": self.objective,
            "best_objective": self.best_objective,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "projection_count": self.projection_count,
            "trace": [record.to_dict() for record in self.trace],
            "npz_file": self.npz_file,
            "npz_sha256": self.npz_sha256,
            "claim_boundary": {
                "numerical_state_only": True,
                "exact_solution_certified": False,
                "nonexistence_proved": False,
            },
        }


@dataclass(frozen=True)
class LoadedLMCheckpoint:
    metadata: LMCheckpoint
    weights: np.ndarray
    best_weights: np.ndarray
    current_residual: np.ndarray
    objective_history: np.ndarray


def _problem_payload(
    run_id: str,
    support: Sequence[int],
    chart: GaugeChart,
    plan: LMPlan,
    target_amplitude: complex,
    initialization_sha256: str,
) -> dict:
    return {
        "run_id": str(run_id),
        "system_fingerprint": system_fingerprint(),
        "support": list(support),
        "chart": chart.to_dict(),
        "plan": plan.to_dict(),
        "target_amplitude": _complex_pair(target_amplitude),
        "initialization_sha256": initialization_sha256,
    }


def _write_checkpoint(
    prefix: Path | str,
    *,
    run_id: str,
    support: Sequence[int],
    chart: GaugeChart,
    plan: LMPlan,
    target_amplitude: complex,
    initialization_sha256: str,
    iteration: int,
    evaluations: int,
    damping: float,
    trust_radius: float,
    status: str,
    objective: float,
    best_objective: float,
    accepted_steps: int,
    rejected_steps: int,
    projection_count: int,
    trace: Sequence[LMIterationRecord],
    weights: np.ndarray,
    best_weights: np.ndarray,
    current_residual: np.ndarray,
    objective_history: Sequence[float],
) -> LMCheckpoint:
    json_path, legacy_npz_path = _checkpoint_paths(prefix)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = legacy_npz_path.with_name(
        f".{legacy_npz_path.name}.{os.getpid()}.{iteration}.tmp"
    )
    temporary_json = json_path.with_name(
        f".{json_path.name}.{os.getpid()}.tmp"
    )
    arrays = {
        "weights": np.asarray(weights, dtype=np.complex128),
        "best_weights": np.asarray(
            best_weights, dtype=np.complex128
        ),
        "current_residual": np.asarray(
            current_residual, dtype=np.complex128
        ),
        "objective_history": np.asarray(
            objective_history, dtype=np.float64
        ),
    }
    try:
        with temporary_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        npz_digest = _file_sha256(temporary_npz)
        npz_path = legacy_npz_path.with_name(
            f"{legacy_npz_path.stem}.g{int(iteration):06d}."
            f"{npz_digest[:16]}.npz"
        )
        problem = _problem_payload(
            run_id,
            support,
            chart,
            plan,
            target_amplitude,
            initialization_sha256,
        )
        checkpoint = LMCheckpoint(
            run_id=str(run_id),
            problem_sha256=_payload_sha256(problem),
            plan_sha256=_payload_sha256(plan.to_dict()),
            support=tuple(support),
            gauge_anchors=chart.anchors,
            target_amplitude=complex(target_amplitude),
            iteration=int(iteration),
            evaluations=int(evaluations),
            damping=float(damping),
            trust_radius=float(trust_radius),
            status=str(status),
            objective=_finite_or_none(objective),
            best_objective=_finite_or_none(best_objective),
            accepted_steps=int(accepted_steps),
            rejected_steps=int(rejected_steps),
            projection_count=int(projection_count),
            trace=tuple(trace),
            npz_file=npz_path.name,
            npz_sha256=npz_digest,
        )
        temporary_json.write_bytes(_json_bytes(checkpoint.to_dict()))
        # Publish the immutable generation before atomically moving the JSON
        # pointer.  A crash leaves either the old complete pair or an
        # unreferenced generation, never a torn referenced pair.
        os.replace(temporary_npz, npz_path)
        os.replace(temporary_json, json_path)
    finally:
        for temporary in (temporary_npz, temporary_json):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return checkpoint


def load_lm_checkpoint(prefix: Path | str) -> LoadedLMCheckpoint:
    """Load and structurally verify one atomic JSON/NPZ checkpoint pair."""

    json_path, _legacy_npz_path = _checkpoint_paths(prefix)
    if not json_path.is_file() or json_path.is_symlink():
        raise KrennNumericalContinuationError(
            "checkpoint JSON pointer is absent or linked"
        )
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        checkpoint = LMCheckpoint(
            run_id=str(payload["run_id"]),
            problem_sha256=str(payload["problem_sha256"]),
            plan_sha256=str(payload["plan_sha256"]),
            support=tuple(payload["support"]),
            gauge_anchors=tuple(payload["gauge_anchors"]),
            target_amplitude=_complex_from_pair(
                payload["target_amplitude"]
            ),
            iteration=int(payload["iteration"]),
            evaluations=int(payload["evaluations"]),
            damping=float(payload["damping"]),
            trust_radius=float(payload["trust_radius"]),
            status=str(payload["status"]),
            objective=(
                None
                if payload["objective"] is None
                else float(payload["objective"])
            ),
            best_objective=(
                None
                if payload["best_objective"] is None
                else float(payload["best_objective"])
            ),
            accepted_steps=int(payload["accepted_steps"]),
            rejected_steps=int(payload["rejected_steps"]),
            projection_count=int(payload["projection_count"]),
            trace=tuple(
                LMIterationRecord.from_dict(record)
                for record in payload["trace"]
            ),
            npz_file=str(payload["npz_file"]),
            npz_sha256=str(payload["npz_sha256"]),
            schema=str(payload["schema"]),
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise KrennNumericalContinuationError(
            "could not decode complex LM checkpoint metadata"
        ) from error
    if payload != checkpoint.to_dict():
        raise KrennNumericalContinuationError(
            "complex LM checkpoint metadata was not canonical"
        )
    npz_path = json_path.parent / checkpoint.npz_file
    if (
        not npz_path.is_file()
        or npz_path.is_symlink()
        or npz_path.parent.resolve(strict=False)
        != json_path.parent.resolve(strict=False)
        or _file_sha256(npz_path) != checkpoint.npz_sha256
    ):
        raise KrennNumericalContinuationError(
            "complex LM checkpoint NPZ digest changed"
        )
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "weights",
                "best_weights",
                "current_residual",
                "objective_history",
            }:
                raise KrennNumericalContinuationError(
                    "complex LM checkpoint array inventory changed"
                )
            weights = np.asarray(archive["weights"]).copy()
            best = np.asarray(archive["best_weights"]).copy()
            residual = np.asarray(
                archive["current_residual"]
            ).copy()
            history = np.asarray(
                archive["objective_history"]
            ).copy()
    except (OSError, ValueError) as error:
        raise KrennNumericalContinuationError(
            "could not load complex LM checkpoint arrays"
        ) from error
    if (
        weights.dtype != np.dtype(np.complex128)
        or best.dtype != np.dtype(np.complex128)
        or residual.dtype != np.dtype(np.complex128)
        or history.dtype != np.dtype(np.float64)
        or weights.shape != (AMBIENT_VARIABLES,)
        or best.shape != (AMBIENT_VARIABLES,)
        or residual.shape != (EQUATION_COUNT,)
        or history.ndim != 1
        or len(history) != checkpoint.iteration + 1
    ):
        raise KrennNumericalContinuationError(
            "complex LM checkpoint array shapes or dtypes changed"
        )
    return LoadedLMCheckpoint(
        metadata=checkpoint,
        weights=weights,
        best_weights=best,
        current_residual=residual,
        objective_history=history,
    )


def _objective(residual: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.vdot(residual, residual).real)


def _finite_complex_array(values: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(values.real))
        and np.all(np.isfinite(values.imag))
    )


def bounded_complex_lm(
    initial_weights: Sequence[complex] | np.ndarray | InitializerRecord,
    support: Iterable[int],
    *,
    chart: GaugeChart | None = None,
    plan: LMPlan | None = None,
    target_amplitude: complex = 0.0j,
    run_id: str = "bounded-complex-lm",
    checkpoint_prefix: Path | str | None = None,
    resume: bool = False,
    work_iteration_cap: int | None = None,
) -> NumericalSolveResult:
    """Run deterministic hard-bounded complex Levenberg--Marquardt.

    ``work_iteration_cap`` is an optional per-invocation pause control.  It is
    not part of the mathematical plan and permits exact checkpoint/resume
    tests without changing the plan hash.
    """

    canonical = _canonical_support(support)
    plan = LMPlan() if plan is None else plan
    if not isinstance(plan, LMPlan):
        raise KrennNumericalContinuationError(
            "bounded complex LM requires a canonical LMPlan"
        )
    amplitude = complex(target_amplitude)
    target_array(amplitude)
    chart = (
        support_gauge_chart(
            canonical,
            action_group=(
                GHZ_COLOR_DIAGONAL_GAUGE
                if amplitude == 0.0j
                else GHZ_VICTIM_COLOR_DIAGONAL_GAUGE
            ),
        )
        if chart is None
        else chart
    )
    if chart.support != canonical:
        raise KrennNumericalContinuationError(
            "complex LM gauge chart belongs to another support"
        )
    if plan.l2_bound * plan.l2_bound + 1.0e-14 < chart.rank:
        raise KrennNumericalContinuationError(
            "LM L2 bound cannot contain the fixed chart anchors"
        )
    if not str(run_id):
        raise KrennNumericalContinuationError(
            "complex LM run id cannot be empty"
        )
    if work_iteration_cap is not None:
        work_iteration_cap = int(work_iteration_cap)
        if not 1 <= work_iteration_cap <= plan.max_iterations:
            raise KrennNumericalContinuationError(
                "per-invocation work cap is outside the LM iteration cap"
            )
    if isinstance(initial_weights, InitializerRecord):
        if initial_weights.support != canonical:
            raise KrennNumericalContinuationError(
                "initializer belongs to another numerical support"
            )
        requested_initial = _dense_weights(
            initial_weights.dense_weights(), canonical
        )
    else:
        requested_initial = _dense_weights(
            initial_weights, canonical
        )
    initialization_sha256 = _complex_vector_sha256(
        requested_initial
    )

    problem = _problem_payload(
        str(run_id),
        canonical,
        chart,
        plan,
        amplitude,
        initialization_sha256,
    )
    problem_digest = _payload_sha256(problem)
    trace: list[LMIterationRecord] = []
    accepted_steps = 0
    rejected_steps = 0
    projection_count = 0
    invocation_start_iteration = 0
    restored_terminal_status: str | None = None

    if resume:
        if checkpoint_prefix is None:
            raise KrennNumericalContinuationError(
                "checkpoint resume needs a checkpoint prefix"
            )
        loaded = load_lm_checkpoint(checkpoint_prefix)
        metadata = loaded.metadata
        if (
            metadata.problem_sha256 != problem_digest
            or metadata.plan_sha256
            != _payload_sha256(plan.to_dict())
            or metadata.run_id != str(run_id)
            or metadata.support != canonical
            or metadata.gauge_anchors != chart.anchors
            or metadata.target_amplitude != amplitude
        ):
            raise KrennNumericalContinuationError(
                "checkpoint does not match the requested numerical problem"
            )
        weights = loaded.weights
        best_weights = loaded.best_weights
        residual = complex_residual(weights, amplitude)
        if not np.allclose(
            residual,
            loaded.current_residual,
            rtol=1.0e-13,
            atol=1.0e-13,
            equal_nan=True,
        ):
            raise KrennNumericalContinuationError(
                "checkpoint residual failed deterministic replay"
            )
        objective = _objective(residual)
        objective_matches = (
            metadata.objective is None
            and not math.isfinite(objective)
        ) or (
            metadata.objective is not None
            and math.isfinite(objective)
            and math.isclose(
                objective,
                metadata.objective,
                rel_tol=1.0e-13,
                abs_tol=1.0e-15,
            )
        )
        if not objective_matches:
            raise KrennNumericalContinuationError(
                "checkpoint objective failed deterministic replay"
            )
        best_residual = complex_residual(best_weights, amplitude)
        best_objective = _objective(best_residual)
        best_objective_matches = (
            metadata.best_objective is None
            and not math.isfinite(best_objective)
        ) or (
            metadata.best_objective is not None
            and math.isfinite(best_objective)
            and math.isclose(
                best_objective,
                metadata.best_objective,
                rel_tol=1.0e-13,
                abs_tol=1.0e-15,
            )
        )
        if not best_objective_matches:
            raise KrennNumericalContinuationError(
                "checkpoint best objective failed deterministic replay"
            )
        iteration = metadata.iteration
        evaluations = metadata.evaluations
        damping = metadata.damping
        trust_radius = metadata.trust_radius
        objective_history = list(map(float, loaded.objective_history))
        accepted_steps = metadata.accepted_steps
        rejected_steps = metadata.rejected_steps
        projection_count = metadata.projection_count
        trace = list(metadata.trace)
        if (
            iteration > plan.max_iterations
            or evaluations > plan.max_evaluations
        ):
            raise KrennNumericalContinuationError(
                "checkpoint exceeded the requested LM safety caps"
            )
        if metadata.status not in {
            "running",
            "checkpoint-paused",
        }:
            restored_terminal_status = metadata.status
        invocation_start_iteration = iteration
    else:
        weights, projection = project_hard_bounds(
            requested_initial,
            chart,
            l2_bound=plan.l2_bound,
            linf_bound=plan.linf_bound,
        )
        projection_count += int(projection.changed)
        residual = complex_residual(weights, amplitude)
        evaluations = 1
        objective = _objective(residual)
        best_weights = weights.copy()
        best_residual = residual.copy()
        best_objective = objective
        iteration = 0
        damping = plan.initial_damping
        trust_radius = plan.initial_trust_radius
        objective_history = [objective]

    status: str | None = restored_terminal_status
    if (
        status is None
        and (
            not _finite_complex_array(residual)
            or not math.isfinite(objective)
        )
    ):
        status = "nonfinite-evaluation"

    while status is None:
        metrics_now = ResidualMetrics.from_residual(residual)
        if (
            metrics_now.linf is not None
            and metrics_now.linf <= plan.residual_tolerance
        ):
            status = "numerical-residual-tolerance"
            break
        if iteration >= plan.max_iterations:
            status = "iteration-cap-reached"
            break
        if evaluations >= plan.max_evaluations:
            status = "evaluation-cap-reached"
            break
        if (
            work_iteration_cap is not None
            and iteration - invocation_start_iteration
            >= work_iteration_cap
        ):
            status = "checkpoint-paused"
            break

        jacobian = analytic_jacobian(weights, chart.free_indices)
        evaluations += 1
        if not _finite_complex_array(jacobian):
            status = "nonfinite-evaluation"
            break
        gradient = jacobian.conj().T @ residual
        gradient_linf = float(
            np.abs(gradient).max(initial=0.0)
        )
        if not math.isfinite(gradient_linf):
            status = "nonfinite-evaluation"
            break
        if gradient_linf <= plan.gradient_tolerance:
            status = "gradient-stalled"
            break

        column_norms = np.linalg.norm(jacobian, axis=0)
        scaling = np.maximum(column_norms, 1.0)
        augmented = np.vstack(
            (
                jacobian,
                math.sqrt(damping)
                * np.diag(scaling.astype(np.complex128)),
            )
        )
        right = np.concatenate(
            (
                -residual,
                np.zeros(len(chart.free_indices), dtype=np.complex128),
            )
        )
        try:
            step = np.linalg.lstsq(
                augmented, right, rcond=None
            )[0]
        except np.linalg.LinAlgError:
            status = "linear-algebra-failure"
            break
        if not _finite_complex_array(step):
            status = "invalid-step"
            break
        step_norm = float(np.linalg.norm(step))
        if not math.isfinite(step_norm):
            status = "invalid-step"
            break
        if step_norm <= plan.step_tolerance:
            status = "step-stalled"
            break
        if step_norm > trust_radius:
            step *= trust_radius / step_norm
            step_norm = trust_radius

        accepted = False
        selected_ratio: float | None = None
        selected_step_norm: float | None = None
        selected_projection: ProjectionReport | None = None
        trials = 0
        evaluation_exhausted = False
        for backtracking in range(plan.backtracking_steps + 1):
            if evaluations >= plan.max_evaluations:
                evaluation_exhausted = True
                break
            trials += 1
            multiplier = 0.5**backtracking
            candidate = weights.copy()
            candidate[np.asarray(chart.free_indices, dtype=np.intp)] += (
                multiplier * step
            )
            candidate, projection = project_hard_bounds(
                candidate,
                chart,
                l2_bound=plan.l2_bound,
                linf_bound=plan.linf_bound,
            )
            actual_step = (
                candidate[
                    np.asarray(chart.free_indices, dtype=np.intp)
                ]
                - weights[
                    np.asarray(chart.free_indices, dtype=np.intp)
                ]
            )
            actual_step_norm = float(np.linalg.norm(actual_step))
            if actual_step_norm <= plan.step_tolerance:
                selected_projection = projection
                selected_step_norm = actual_step_norm
                continue
            candidate_residual = complex_residual(
                candidate, amplitude
            )
            evaluations += 1
            if not _finite_complex_array(candidate_residual):
                selected_projection = projection
                selected_step_norm = actual_step_norm
                continue
            candidate_objective = _objective(candidate_residual)
            predicted_residual = residual + jacobian @ actual_step
            predicted_objective = _objective(predicted_residual)
            actual_reduction = objective - candidate_objective
            predicted_reduction = objective - predicted_objective
            ratio = (
                actual_reduction / predicted_reduction
                if predicted_reduction > 0.0
                else -math.inf
            )
            selected_ratio = _finite_or_none(ratio)
            selected_projection = projection
            selected_step_norm = actual_step_norm
            if (
                actual_reduction > 0.0
                and ratio >= plan.acceptance_ratio
            ):
                weights = candidate
                residual = candidate_residual
                objective = candidate_objective
                accepted = True
                projection_count += int(projection.changed)
                if objective < best_objective:
                    best_objective = objective
                    best_weights = weights.copy()
                    best_residual = residual.copy()
                break

        iteration += 1
        if accepted:
            accepted_steps += 1
            if selected_ratio is not None and selected_ratio > 0.75:
                trust_radius = min(
                    plan.maximum_trust_radius,
                    max(trust_radius, 2.0 * (selected_step_norm or 0.0)),
                )
                damping = max(
                    np.finfo(np.float64).tiny,
                    damping * plan.damping_decrease,
                )
            elif selected_ratio is not None and selected_ratio < 0.25:
                trust_radius = max(
                    plan.minimum_trust_radius,
                    0.5 * trust_radius,
                )
                damping *= plan.damping_increase
        else:
            rejected_steps += 1
            trust_radius = max(
                plan.minimum_trust_radius,
                0.5 * trust_radius,
            )
            damping *= plan.damping_increase

        current_metrics = ResidualMetrics.from_residual(residual)
        boundary_active = bool(
            selected_projection
            and (
                selected_projection.on_l2_boundary
                or selected_projection.on_linf_boundary
            )
        )
        trace.append(
            LMIterationRecord(
                iteration=iteration,
                evaluations=evaluations,
                objective=_finite_or_none(objective),
                residual_linf=current_metrics.linf,
                gradient_linf=_finite_or_none(gradient_linf),
                step_norm=(
                    _finite_or_none(selected_step_norm)
                    if selected_step_norm is not None
                    else None
                ),
                damping=float(damping),
                trust_radius=float(trust_radius),
                accepted=accepted,
                ratio=selected_ratio,
                backtracking_trials=trials,
                projection_changed=bool(
                    selected_projection
                    and selected_projection.changed
                ),
                hard_boundary_active=boundary_active,
            )
        )
        objective_history.append(objective)

        if evaluation_exhausted:
            status = "evaluation-cap-reached"
        elif (
            not accepted
            and trust_radius <= plan.minimum_trust_radius
            * (1.0 + 1.0e-12)
        ):
            status = (
                "norm-bound-stalled"
                if boundary_active
                else "step-stalled"
            )

        checkpoint_due = bool(
            checkpoint_prefix is not None
            and plan.checkpoint_interval
            and iteration % plan.checkpoint_interval == 0
        )
        if checkpoint_due:
            _write_checkpoint(
                checkpoint_prefix,
                run_id=str(run_id),
                support=canonical,
                chart=chart,
                plan=plan,
                target_amplitude=amplitude,
                initialization_sha256=initialization_sha256,
                iteration=iteration,
                evaluations=evaluations,
                damping=damping,
                trust_radius=trust_radius,
                status=status or "running",
                objective=objective,
                best_objective=best_objective,
                accepted_steps=accepted_steps,
                rejected_steps=rejected_steps,
                projection_count=projection_count,
                trace=trace,
                weights=weights,
                best_weights=best_weights,
                current_residual=residual,
                objective_history=objective_history,
            )

    if status is None:
        raise KrennNumericalContinuationError(
            "complex LM exited without a fail-closed status"
        )
    if checkpoint_prefix is not None and restored_terminal_status is None:
        _write_checkpoint(
            checkpoint_prefix,
            run_id=str(run_id),
            support=canonical,
            chart=chart,
            plan=plan,
            target_amplitude=amplitude,
            initialization_sha256=initialization_sha256,
            iteration=iteration,
            evaluations=evaluations,
            damping=damping,
            trust_radius=trust_radius,
            status=status,
            objective=objective,
            best_objective=best_objective,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            projection_count=projection_count,
            trace=trace,
            weights=weights,
            best_weights=best_weights,
            current_residual=residual,
            objective_history=objective_history,
        )

    residual_metrics = ResidualMetrics.from_residual(best_residual)
    weight_metrics = WeightMetrics.from_weights(
        best_weights,
        chart,
        l2_bound=plan.l2_bound,
        linf_bound=plan.linf_bound,
    )
    invariants = InvariantMetrics.from_weights(
        best_weights, canonical
    )
    return NumericalSolveResult(
        run_id=str(run_id),
        support=canonical,
        chart=chart,
        plan=plan,
        target_amplitude=amplitude,
        status=status,
        iterations=iteration,
        evaluations=evaluations,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        weights=tuple(map(complex, best_weights)),
        residual_metrics=residual_metrics,
        weight_metrics=weight_metrics,
        invariant_metrics=invariants,
        trace=tuple(trace),
        projection_count=projection_count,
    )


def solve_direct_ghz(
    initial_weights: Sequence[complex] | np.ndarray | InitializerRecord,
    support: Iterable[int],
    *,
    chart: GaugeChart | None = None,
    plan: LMPlan | None = None,
    run_id: str = "direct-ghz",
    checkpoint_prefix: Path | str | None = None,
    resume: bool = False,
) -> NumericalSolveResult:
    """Run one direct bounded solve against the exact GHZ target."""

    return bounded_complex_lm(
        initial_weights,
        support,
        chart=chart,
        plan=plan,
        target_amplitude=0.0j,
        run_id=run_id,
        checkpoint_prefix=checkpoint_prefix,
        resume=resume,
    )


@dataclass(frozen=True)
class ContinuationPlan:
    """A deterministic victim-amplitude homotopy ending at direct GHZ."""

    target_schedule: tuple[float, ...] = DEFAULT_TARGET_SCHEDULE
    lm_plan: LMPlan = field(default_factory=LMPlan)
    direct_final_polish: bool = True
    continue_after_nonconvergence: bool = False

    def __post_init__(self) -> None:
        schedule = tuple(map(float, self.target_schedule))
        object.__setattr__(self, "target_schedule", schedule)
        if not schedule or any(
            not math.isfinite(value) or value < 0.0
            for value in schedule
        ):
            raise KrennNumericalContinuationError(
                "continuation target schedule must be finite and nonnegative"
            )
        if any(
            schedule[index + 1] >= schedule[index]
            for index in range(len(schedule) - 1)
        ):
            raise KrennNumericalContinuationError(
                "continuation target schedule must strictly decrease"
            )
        if schedule[-1] != 0.0:
            raise KrennNumericalContinuationError(
                "continuation schedule must end with direct GHZ amplitude 0"
            )
        if not isinstance(self.lm_plan, LMPlan):
            raise KrennNumericalContinuationError(
                "continuation requires a canonical LMPlan"
            )
        if (
            not isinstance(self.direct_final_polish, bool)
            or not isinstance(
                self.continue_after_nonconvergence, bool
            )
        ):
            raise KrennNumericalContinuationError(
                "continuation policy flags must be Boolean"
            )

    def to_dict(self) -> dict:
        return {
            "target_schedule": list(self.target_schedule),
            "victim_equation": N6_D3_SEED_DEFECT_EQUATION,
            "lm_plan": self.lm_plan.to_dict(),
            "direct_final_polish": self.direct_final_polish,
            "continue_after_nonconvergence": (
                self.continue_after_nonconvergence
            ),
            "known_branch_Q_value": "t^-1",
            "bounded_numerical_continuation_only": True,
        }


@dataclass(frozen=True)
class ContinuationResult:
    """A sequence of bounded numerical solves along the target schedule."""

    run_id: str
    support: tuple[int, ...]
    chart: GaugeChart
    plan: ContinuationPlan
    steps: tuple[NumericalSolveResult, ...]
    status: str
    schema: str = CONTINUATION_SCHEMA
    exact_solution_certified: bool = False
    nonexistence_proved: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema != CONTINUATION_SCHEMA
            or self.status
            not in {
                "schedule-converged",
                "schedule-explored-with-nonconverged-steps",
                "step-nonconverged",
                "numerical-failure",
                "checkpoint-paused",
            }
            or self.exact_solution_certified
            or self.nonexistence_proved
            or any(step.support != self.support for step in self.steps)
        ):
            raise KrennNumericalContinuationError(
                "continuation result changed its fail-closed contract"
            )
        converged = tuple(
            step.status == "numerical-residual-tolerance"
            for step in self.steps
        )
        if (
            self.status == "schedule-converged"
            and (
                len(self.steps) < len(self.plan.target_schedule)
                or not all(converged)
                or self.steps[-1].target_amplitude != 0.0j
            )
        ):
            raise KrennNumericalContinuationError(
                "a converged continuation contains an unconverged step"
            )
        if (
            self.status
            == "schedule-explored-with-nonconverged-steps"
            and (
                not self.plan.continue_after_nonconvergence
                or all(converged)
            )
        ):
            raise KrennNumericalContinuationError(
                "exploratory continuation status is inconsistent"
            )

    @property
    def final_step(self) -> NumericalSolveResult | None:
        return self.steps[-1] if self.steps else None

    @property
    def weights_remain_finite(self) -> bool:
        return bool(
            self.steps
            and all(step.weights_remain_finite for step in self.steps)
        )

    @property
    def numerical_zero_at_ghz(self) -> bool:
        final = self.final_step
        return bool(
            final is not None
            and final.target_amplitude == 0.0j
            and final.numerical_zero
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "support": list(self.support),
            "gauge_chart": self.chart.to_dict(),
            "plan": self.plan.to_dict(),
            "status": self.status,
            "completed_target_amplitudes": [
                _complex_pair(step.target_amplitude)
                for step in self.steps
            ],
            "steps": [
                step.to_dict(include_trace=False) for step in self.steps
            ],
            "weights_remain_finite": self.weights_remain_finite,
            "numerical_zero_at_direct_GHZ": (
                self.numerical_zero_at_ghz
            ),
            "exact_solution_certified": False,
            "claim_boundary": {
                "continuation_is_numerical_only": True,
                "known_Laurent_pole_is_not_a_finite_witness": True,
                "numerical_zero_requires_exact_reconstruction": True,
                "bounded_miss_proves_nonexistence": False,
                "nonexistence_proved": False,
            },
        }


def run_continuation(
    initial_weights: Sequence[complex] | np.ndarray | InitializerRecord,
    support: Iterable[int],
    *,
    chart: GaugeChart | None = None,
    plan: ContinuationPlan | None = None,
    run_id: str = "continuation",
    checkpoint_directory: Path | str | None = None,
    resume: bool = False,
) -> ContinuationResult:
    """Correct successively against ``GHZ+t*e_002121`` down to ``t=0``."""

    canonical = _canonical_support(support)
    plan = ContinuationPlan() if plan is None else plan
    if not isinstance(plan, ContinuationPlan):
        raise KrennNumericalContinuationError(
            "run_continuation requires a canonical ContinuationPlan"
        )
    if isinstance(initial_weights, InitializerRecord):
        chart = initial_weights.chart if chart is None else chart
        current = initial_weights.dense_weights()
    else:
        chart = (
            support_gauge_chart(
                canonical,
                action_group=(
                    GHZ_VICTIM_COLOR_DIAGONAL_GAUGE
                ),
            )
            if chart is None
            else chart
        )
        current = _dense_weights(initial_weights, canonical)
    if chart.support != canonical:
        raise KrennNumericalContinuationError(
            "continuation gauge chart belongs to another support"
        )
    if chart.action_group not in {
        VERTEX_SCALAR_GAUGE,
        GHZ_VICTIM_COLOR_DIAGONAL_GAUGE,
    }:
        raise KrennNumericalContinuationError(
            "continuation chart does not preserve the victim target"
        )
    if resume and checkpoint_directory is None:
        raise KrennNumericalContinuationError(
            "continuation resume needs a checkpoint directory"
        )
    checkpoint_root: Path | None = None
    if checkpoint_directory is not None:
        checkpoint_root = _validate_scratch_prefix(
            Path(checkpoint_directory) / "checkpoint"
        ).parent
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    steps: list[NumericalSolveResult] = []
    aggregate_status: str | None = None
    saw_nonconvergence = False
    for step_index, target in enumerate(plan.target_schedule):
        prefix = (
            checkpoint_root
            / f"{run_id}-step-{step_index:03d}"
            if checkpoint_root is not None
            else None
        )
        resume_step = bool(
            resume
            and prefix is not None
            and _checkpoint_paths(prefix)[0].is_file()
        )
        result = bounded_complex_lm(
            current,
            canonical,
            chart=chart,
            plan=plan.lm_plan,
            target_amplitude=complex(target),
            run_id=f"{run_id}-step-{step_index:03d}",
            checkpoint_prefix=prefix,
            resume=resume_step,
        )
        steps.append(result)
        current = result.dense_weights()
        if result.status == "numerical-residual-tolerance":
            continue
        if result.status == "checkpoint-paused":
            aggregate_status = "checkpoint-paused"
            break
        if result.status in _TERMINAL_FAILURE_STATUSES:
            aggregate_status = "numerical-failure"
            break
        saw_nonconvergence = True
        if not plan.continue_after_nonconvergence:
            aggregate_status = "step-nonconverged"
            break

    if (
        aggregate_status is None
        and plan.direct_final_polish
        and steps
        and steps[-1].target_amplitude == 0.0j
        and len(steps) >= len(plan.target_schedule)
    ):
        polish_index = len(steps)
        prefix = (
            checkpoint_root / f"{run_id}-direct-polish"
            if checkpoint_root is not None
            else None
        )
        polished = solve_direct_ghz(
            current,
            canonical,
            chart=chart,
            plan=plan.lm_plan,
            run_id=f"{run_id}-direct-polish-{polish_index:03d}",
            checkpoint_prefix=prefix,
            resume=bool(
                resume
                and prefix is not None
                and _checkpoint_paths(prefix)[0].is_file()
            ),
        )
        steps.append(polished)
        if polished.status == "checkpoint-paused":
            aggregate_status = "checkpoint-paused"
        elif polished.status in _TERMINAL_FAILURE_STATUSES:
            aggregate_status = "numerical-failure"
        elif polished.status != "numerical-residual-tolerance":
            saw_nonconvergence = True
            aggregate_status = (
                "schedule-explored-with-nonconverged-steps"
                if plan.continue_after_nonconvergence
                else "step-nonconverged"
            )
    if aggregate_status is None:
        aggregate_status = (
            "schedule-explored-with-nonconverged-steps"
            if saw_nonconvergence
            else "schedule-converged"
        )
    return ContinuationResult(
        run_id=str(run_id),
        support=canonical,
        chart=chart,
        plan=plan,
        steps=tuple(steps),
        status=aggregate_status,
    )


@dataclass(frozen=True)
class NumericalJobSpec:
    """A deterministic multistart job independent of worker scheduling."""

    run_id: str
    support: tuple[int, ...]
    master_seed: int
    spawn_key: tuple[int, ...]
    seed_words: tuple[int, ...]
    initializer_kind: str
    activation_amplitude: float
    laurent_t: float
    dense: bool
    use_continuation: bool
    schema: str = JOB_SCHEMA

    def __post_init__(self) -> None:
        support = _canonical_support(self.support)
        object.__setattr__(self, "support", support)
        if self.schema != JOB_SCHEMA:
            raise KrennNumericalContinuationError(
                "numerical job schema changed"
            )
        if self.initializer_kind not in {
            "random-bounded",
            "transverse-laurent-activation",
        }:
            raise KrennNumericalContinuationError(
                "unknown numerical initializer kind"
            )
        if (
            not self.run_id
            or not math.isfinite(float(self.activation_amplitude))
            or float(self.activation_amplitude) <= 0.0
            or not math.isfinite(float(self.laurent_t))
            or float(self.laurent_t) == 0.0
            or self.dense
            != (support == tuple(range(AMBIENT_VARIABLES)))
        ):
            raise KrennNumericalContinuationError(
                "numerical job specification is malformed"
            )
        replay = np.random.SeedSequence(
            int(self.master_seed),
            spawn_key=tuple(map(int, self.spawn_key)),
        )
        if _seed_words(replay) != tuple(map(int, self.seed_words)):
            raise KrennNumericalContinuationError(
                "numerical job SeedSequence receipt failed replay"
            )
        if (
            self.initializer_kind
            == "transverse-laurent-activation"
            and not NATURAL_SUPPORT_SET.issubset(support)
        ):
            raise KrennNumericalContinuationError(
                "a transverse Laurent job omitted a natural coordinate"
            )

    def seed_sequence(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(
            int(self.master_seed),
            spawn_key=self.spawn_key,
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "support": list(self.support),
            "support_size": len(self.support),
            "master_seed": self.master_seed,
            "spawn_key": list(self.spawn_key),
            "seed_words_uint32": list(self.seed_words),
            "initializer_kind": self.initializer_kind,
            "activation_amplitude": self.activation_amplitude,
            "laurent_t": self.laurent_t,
            "dense_all_135_variables": self.dense,
            "use_continuation": self.use_continuation,
            "symmetry_weight_equalities": 0,
            "exact_solution_claimed": False,
        }


def deterministic_job_specs(
    supports: Sequence[Sequence[int]],
    *,
    master_seed: int = 60_320_260_724,
    starts_per_support: int = 4,
    include_dense: bool = True,
    dense_starts: int = 4,
    activation_amplitudes: Sequence[float] = (0.03, 0.1, 0.3, 1.0),
) -> tuple[NumericalJobSpec, ...]:
    """Pre-assign SeedSequence children before any parallel scheduling."""

    canonical_supports = tuple(
        _canonical_support(support) for support in supports
    )
    starts_per_support = int(starts_per_support)
    dense_starts = int(dense_starts)
    amplitudes = tuple(map(float, activation_amplitudes))
    if (
        not 1 <= starts_per_support <= 1_000
        or not 0 <= dense_starts <= 1_000
        or not amplitudes
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in amplitudes
        )
    ):
        raise KrennNumericalContinuationError(
            "multistart counts or activation amplitudes are invalid"
        )
    rows: list[tuple[str, tuple[int, ...], str, float, bool, bool]] = []
    for support_index, support in enumerate(canonical_supports):
        has_laurent = NATURAL_SUPPORT_SET.issubset(support)
        for start in range(starts_per_support):
            initializer = (
                "transverse-laurent-activation"
                if has_laurent and start % 2 == 0
                else "random-bounded"
            )
            rows.append(
                (
                    f"sparse-{support_index:04d}-{start:04d}",
                    support,
                    initializer,
                    amplitudes[start % len(amplitudes)],
                    False,
                    has_laurent,
                )
            )
    if include_dense:
        dense_support = tuple(range(AMBIENT_VARIABLES))
        for start in range(dense_starts):
            initializer = (
                "transverse-laurent-activation"
                if start % 2 == 0
                else "random-bounded"
            )
            rows.append(
                (
                    f"dense-{start:04d}",
                    dense_support,
                    initializer,
                    amplitudes[start % len(amplitudes)],
                    True,
                    initializer == "transverse-laurent-activation",
                )
            )
    root = np.random.SeedSequence(int(master_seed))
    children = root.spawn(len(rows))
    return tuple(
        NumericalJobSpec(
            run_id=run_id,
            support=support,
            master_seed=int(master_seed),
            spawn_key=child.spawn_key,
            seed_words=_seed_words(child),
            initializer_kind=initializer,
            activation_amplitude=amplitude,
            laurent_t=1.0,
            dense=dense,
            use_continuation=continuation,
        )
        for (
            run_id,
            support,
            initializer,
            amplitude,
            dense,
            continuation,
        ), child in zip(rows, children, strict=True)
    )


def dense_support_job_specs(
    *,
    master_seed: int = 60_320_260_724,
    starts: int = 8,
    activation_amplitudes: Sequence[float] = (0.03, 0.1, 0.3, 1.0),
) -> tuple[NumericalJobSpec, ...]:
    """Return deterministic all-135-variable comparison jobs."""

    return tuple(
        spec
        for spec in deterministic_job_specs(
            (),
            master_seed=master_seed,
            starts_per_support=1,
            include_dense=True,
            dense_starts=starts,
            activation_amplitudes=activation_amplitudes,
        )
        if spec.dense
    )


def run_numerical_job(
    spec: NumericalJobSpec,
    *,
    lm_plan: LMPlan | None = None,
    continuation_plan: ContinuationPlan | None = None,
    scratch_directory: Path | str | None = None,
    resume: bool = False,
) -> NumericalSolveResult | ContinuationResult:
    """Materialize and execute one deterministic job specification."""

    if not isinstance(spec, NumericalJobSpec):
        raise KrennNumericalContinuationError(
            "run_numerical_job requires a canonical job specification"
        )
    lm_plan = LMPlan() if lm_plan is None else lm_plan
    moving_target_job = bool(
        spec.use_continuation
        or spec.initializer_kind
        == "transverse-laurent-activation"
    )
    action_group = (
        GHZ_VICTIM_COLOR_DIAGONAL_GAUGE
        if moving_target_job
        else GHZ_COLOR_DIAGONAL_GAUGE
    )
    chart = (
        natural_gauge_chart(
            spec.support,
            action_group=action_group,
        )
        if (
            spec.dense
            or NATURAL_SUPPORT_SET.issubset(spec.support)
        )
        else support_gauge_chart(
            spec.support,
            action_group=action_group,
        )
    )
    if spec.initializer_kind == "transverse-laurent-activation":
        initializer = transverse_activated_initializer(
            spec.support,
            spec.seed_sequence(),
            activation_amplitude=spec.activation_amplitude,
            laurent_t=spec.laurent_t,
            chart=chart,
            l2_bound=lm_plan.l2_bound,
            linf_bound=lm_plan.linf_bound,
        )
    else:
        initializer = random_bounded_initializer(
            spec.support,
            spec.seed_sequence(),
            chart=chart,
            scale=spec.activation_amplitude,
            l2_bound=lm_plan.l2_bound,
            linf_bound=lm_plan.linf_bound,
        )
    checkpoint_root: Path | None = None
    if scratch_directory is not None:
        checkpoint_root = _validate_scratch_prefix(
            Path(scratch_directory) / spec.run_id
        )
    if resume and checkpoint_root is None:
        raise KrennNumericalContinuationError(
            "numerical job resume needs a scratch directory"
        )
    if spec.use_continuation:
        continuation_plan = (
            ContinuationPlan(lm_plan=lm_plan)
            if continuation_plan is None
            else continuation_plan
        )
        return run_continuation(
            initializer,
            spec.support,
            chart=chart,
            plan=continuation_plan,
            run_id=spec.run_id,
            checkpoint_directory=checkpoint_root,
            resume=resume,
        )
    direct_resume = bool(
        resume
        and checkpoint_root is not None
        and _checkpoint_paths(checkpoint_root)[0].is_file()
    )
    return solve_direct_ghz(
        initializer,
        spec.support,
        chart=chart,
        plan=lm_plan,
        run_id=spec.run_id,
        checkpoint_prefix=checkpoint_root,
        resume=direct_resume,
    )


__all__ = (
    "AMBIENT_VARIABLES",
    "CHECKPOINT_SCHEMA",
    "ContinuationPlan",
    "ContinuationResult",
    "DEFAULT_TARGET_SCHEDULE",
    "EQUATION_COUNT",
    "GHZ_COLOR_DIAGONAL_GAUGE",
    "GHZ_VICTIM_COLOR_DIAGONAL_GAUGE",
    "GaugeChart",
    "InitializerRecord",
    "InvariantMetrics",
    "JOB_SCHEMA",
    "KNOWN_Q_INDICES",
    "KrennNumericalContinuationError",
    "LMCheckpoint",
    "LMIterationRecord",
    "LMPlan",
    "LoadedLMCheckpoint",
    "NATURAL_GAUGE_ANCHORS",
    "NATURAL_SUPPORT",
    "NUMERICAL_BLAS_THREAD_LIMIT",
    "NUMERICAL_CONTINUATION_SCHEMA",
    "NumericalJobSpec",
    "NumericalSolveResult",
    "ProjectionReport",
    "ResidualMetrics",
    "WeightMetrics",
    "analytic_jacobian",
    "bounded_complex_lm",
    "canonical_rhs_array",
    "color_diagonal_ghz_gauge_exponent_matrix",
    "color_diagonal_moving_target_gauge_exponent_matrix",
    "complex_output",
    "complex_residual",
    "dense_support_job_specs",
    "deterministic_job_specs",
    "equation_index_array",
    "gauge_exponent_matrix",
    "known_q_color_gauge_character",
    "load_lm_checkpoint",
    "monomial_index_array",
    "natural_gauge_chart",
    "natural_laurent_initializer",
    "project_hard_bounds",
    "projective_victim_repair_initializer",
    "random_bounded_initializer",
    "residual_and_jacobian",
    "run_continuation",
    "run_numerical_job",
    "solve_direct_ghz",
    "support_gauge_chart",
    "system_fingerprint",
    "target_array",
    "transverse_activated_initializer",
    "victim_gauge_character",
    "VERTEX_SCALAR_GAUGE",
)
