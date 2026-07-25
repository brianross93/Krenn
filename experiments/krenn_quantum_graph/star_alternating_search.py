"""Bounded complex alternating-star search on an exact gauge chart.

For a fixed apex, the ``n=6,d=3`` perfect-matching tensor map is linear in
the 45 incident weights.  This module uses that factorization for a
deterministic numerical reconnaissance campaign, while closing the two
main loopholes in a naive alternating least-squares implementation:

* a rank-15 direct-GHZ color gauge is fixed by setting an exact,
  deterministically selected set of 15 anchor weights to one; and
* each update is a minimum-change correction to the current free star
  weights, with persistent global L2 and Linf bounds.

Every accepted apex update is replayed in all 729 equations by both the
primary sparse system and an independently enumerated perfect-matching
map.  Atomic checkpoints are written after every apex update and contain
enough fingerprints and state to reject stale or altered resumes.

This is numerical reconnaissance only.  A small residual is not an exact
counterexample, a bounded miss is not a nonexistence proof, and a trend
across increasing bounds is not a proof of border-only membership.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np

from experiments.krenn_quantum_graph.defect_mining import (
    NATURAL_REPRESENTATIVE,
    seed_witness,
)
from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
)
from experiments.krenn_quantum_graph.star_linearization import (
    star_linearization,
)
from experiments.krenn_quantum_graph.star_linearization_independent import (
    independent_star_linearization_audit,
)
from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    generate_sparse_system,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.vertical_component import (
    color_diagonal_exponent_matrix,
)


STAR_ALS_SCHEMA = "krenn-n6-d3-bounded-star-als-v2"
STAR_ALS_CHECKPOINT_SCHEMA = (
    "krenn-n6-d3-bounded-star-als-checkpoint-v2"
)
STAR_ALS_CAMPAIGN_SCHEMA = (
    "krenn-n6-d3-bounded-star-als-campaign-v2"
)
GAUGE_CHART_SCHEMA = "krenn-n6-d3-direct-ghz-color-gauge-chart-v1"
DEFAULT_SCRATCH_DIRECTORY = Path(
    r"D:\KrennScratch\counterexample_search\star_als"
)
DEFAULT_SEEDS = (2026072501, 2026072502, 2026072503)
DEFAULT_RADII = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
N = 6
D = 3
VARIABLES = 135
EQUATIONS = 729
GAUGE_DIMENSION = 15
GAUGE_ANCHOR_VALUE = 1.0 + 0.0j
# This exact full-rank basis was selected to balance the 30 endpoint-star
# incidences: no apex/color block contains more than two fixed variables.
# Keeping the tuple explicit makes the numerical chart stable across
# implementation refactors; ``_gauge_chart_internal`` independently
# certifies it against the exact exponent matrix on every fresh process.
BALANCED_GAUGE_ANCHORS = (
    1,
    46,
    48,
    53,
    72,
    79,
    81,
    85,
    89,
    108,
    112,
    116,
    126,
    130,
    134,
)
_REPLAY_RTOL = 2e-13
_REPLAY_ATOL = 2e-14


class KrennStarALSError(RuntimeError):
    """A gauge chart, alternating-star run, or checkpoint failed."""


@dataclass(frozen=True)
class StarALSConfig:
    """A deterministic trajectory and its two persistent global bounds.

    ``radius`` is retained as the campaign's historical name for the
    global Linf cap.  ``global_l2_radius=None`` means
    ``radius * sqrt(135)``; the effective value is always serialized in
    results and checkpoints.
    """

    radius: float
    maximum_sweeps: int
    seed: int
    initialization: str = "random"
    residual_tolerance: float = 1e-12
    maximum_seconds: float = 600.0
    global_l2_radius: float | None = None
    patience_apex_updates: int = 120
    minimum_relative_improvement: float = 1e-14
    schema: str = STAR_ALS_SCHEMA

    def __post_init__(self) -> None:
        effective_l2 = (
            float(self.radius) * math.sqrt(VARIABLES)
            if self.global_l2_radius is None
            else float(self.global_l2_radius)
        )
        if (
            self.schema != STAR_ALS_SCHEMA
            or isinstance(self.maximum_sweeps, bool)
            or not isinstance(self.maximum_sweeps, int)
            or self.maximum_sweeps < 1
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.initialization not in ("random", "natural-perturbed")
            or not np.isfinite(self.radius)
            or self.radius < 1.0
            or not np.isfinite(self.residual_tolerance)
            or self.residual_tolerance <= 0
            or not np.isfinite(self.maximum_seconds)
            or self.maximum_seconds <= 0
            or not np.isfinite(effective_l2)
            or effective_l2 < math.sqrt(GAUGE_DIMENSION)
            or isinstance(self.patience_apex_updates, bool)
            or not isinstance(self.patience_apex_updates, int)
            or self.patience_apex_updates < 1
            or not np.isfinite(self.minimum_relative_improvement)
            or self.minimum_relative_improvement < 0
        ):
            raise KrennStarALSError("an alternating-star config is invalid")

    @property
    def global_linf_cap(self) -> float:
        return float(self.radius)

    @property
    def effective_global_l2_cap(self) -> float:
        if self.global_l2_radius is None:
            return float(self.radius) * math.sqrt(VARIABLES)
        return float(self.global_l2_radius)

    def to_dict(self) -> dict:
        return asdict(self)

    def effective_bounds(self) -> dict:
        return {
            "global_linf_cap": self.global_linf_cap,
            "global_l2_cap": self.effective_global_l2_cap,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "config": self.to_dict(),
                "effective_bounds": self.effective_bounds(),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _complex_rows(values: Sequence[complex]) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def _complex_array(
    rows: Sequence[Sequence[float]], *, label: str
) -> np.ndarray:
    if (
        type(rows) not in (list, tuple)
        or len(rows) != VARIABLES
        or any(
            type(row) not in (list, tuple)
            or len(row) != 2
            or not all(np.isfinite(float(value)) for value in row)
            for row in rows
        )
    ):
        raise KrennStarALSError(f"{label} weight vector is malformed")
    return np.asarray(
        [complex(float(row[0]), float(row[1])) for row in rows],
        dtype=np.complex128,
    )


def _fraction_rank_pivots(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, tuple[int, ...]]:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise KrennStarALSError("the gauge exponent matrix is ragged")
    pivot_row = 0
    pivots: list[int] = []
    for column in range(len(rows[0])):
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
                left - factor * right
                for left, right in zip(
                    rows[row], rows[pivot_row], strict=True
                )
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row, tuple(pivots)


def _fraction_determinant(matrix: Sequence[Sequence[int]]) -> Fraction:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows or any(len(row) != len(rows) for row in rows):
        raise KrennStarALSError(
            "a gauge anchor determinant needs a square matrix"
        )
    determinant = Fraction(1)
    for column in range(len(rows)):
        selected = next(
            (
                row
                for row in range(column, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            return Fraction(0)
        if selected != column:
            rows[column], rows[selected] = (
                rows[selected],
                rows[column],
            )
            determinant = -determinant
        pivot = rows[column][column]
        determinant *= pivot
        rows[column] = [value / pivot for value in rows[column]]
        for row in range(column + 1, len(rows)):
            factor = rows[row][column]
            if factor:
                rows[row] = [
                    left - factor * right
                    for left, right in zip(
                        rows[row], rows[column], strict=True
                    )
                ]
    return determinant


@lru_cache(maxsize=1)
def _gauge_chart_internal() -> tuple[
    tuple[int, ...], tuple[tuple[int, ...], ...], int, str
]:
    exponent_rows = tuple(
        tuple(map(int, row))
        for row in color_diagonal_exponent_matrix()
    )
    if (
        len(exponent_rows) != VARIABLES
        or any(len(row) != GAUGE_DIMENSION for row in exponent_rows)
    ):
        raise KrennStarALSError(
            "the direct-GHZ gauge exponent matrix changed shape"
        )
    full_rank, _leftmost_pivots = _fraction_rank_pivots(
        tuple(zip(*exponent_rows))
    )
    anchors = BALANCED_GAUGE_ANCHORS
    anchor_rank, _anchor_pivots = _fraction_rank_pivots(
        tuple(zip(*(exponent_rows[index] for index in anchors)))
    )
    if (
        full_rank != GAUGE_DIMENSION
        or anchor_rank != GAUGE_DIMENSION
        or len(anchors) != GAUGE_DIMENSION
    ):
        raise KrennStarALSError(
            "the direct-GHZ gauge no longer has exact rank 15"
        )
    determinant = _fraction_determinant(
        tuple(exponent_rows[index] for index in anchors)
    )
    if determinant != -2:
        raise KrennStarALSError(
            "the balanced gauge anchor determinant changed from -2"
        )
    block_loads = {
        (vertex, color): 0
        for vertex in range(N)
        for color in range(D)
    }
    for index in anchors:
        left, right, left_color, right_color = variable_key(
            N, D, index
        )
        block_loads[(left, left_color)] += 1
        block_loads[(right, right_color)] += 1
    if max(block_loads.values()) > 2 or sum(block_loads.values()) != 30:
        raise KrennStarALSError(
            "the exact gauge anchors are no longer star-balanced"
        )
    digest_payload = {
        "schema": GAUGE_CHART_SCHEMA,
        "exponent_matrix": exponent_rows,
        "anchors": anchors,
        "anchor_minor_determinant": int(determinant),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return anchors, exponent_rows, int(determinant), fingerprint


def gauge_chart_audit() -> dict:
    """Return the exact, deterministic rank-15 gauge slice."""

    anchors, _matrix, determinant, fingerprint = (
        _gauge_chart_internal()
    )
    natural_support = {
        index
        for index, _value in seed_witness(
            NATURAL_REPRESENTATIVE
        ).entries
    }
    zero_on_natural_seed = tuple(
        index for index in anchors if index not in natural_support
    )
    if len(zero_on_natural_seed) != 13:
        raise KrennStarALSError(
            "the natural seed's position relative to the gauge chart changed"
        )
    return {
        "schema": GAUGE_CHART_SCHEMA,
        "gauge": "direct-GHZ color-diagonal",
        "exact_exponent_matrix_shape": [VARIABLES, GAUGE_DIMENSION],
        "exact_rank_over_Q": GAUGE_DIMENSION,
        "pivot_rule": (
            "fixed deterministic star-balanced row basis, exactly "
            "rank-checked against the 135x15 exponent matrix"
        ),
        "anchor_indices": list(anchors),
        "anchor_coordinates": [
            list(variable_key(N, D, index)) for index in anchors
        ],
        "anchor_values": [[1, 0] for _index in anchors],
        "anchor_minor_determinant": determinant,
        "maximum_anchors_in_any_apex_color_block": 2,
        "sha256": fingerprint,
        "domain": (
            "only the open chart where all 15 selected anchors are "
            "nonzero, gauge-fixed to one"
        ),
        "natural_seed_relation": {
            "anchors_nonzero_on_natural_seed": [
                index for index in anchors if index in natural_support
            ],
            "anchors_zero_on_natural_seed": list(zero_on_natural_seed),
            "known_natural_Laurent_branch_lies_in_this_open_chart": False,
            "natural_perturbed_initialization_is_a_chart_embedding_not_"
            "a_point_on_the_known_branch": True,
        },
        "claim_boundary": {
            "chart_covers_solutions_with_a_zero_anchor": False,
            "chart_search_is_global_affine_search": False,
        },
    }


def _gauge_anchor_indices() -> tuple[int, ...]:
    return _gauge_chart_internal()[0]


_SYSTEM = generate_sparse_system(N, D)
_MONOMIALS = np.asarray(
    _SYSTEM.monomial_variable_indices, dtype=np.int64
).reshape(EQUATIONS, 15, 3)
_TARGET = np.asarray(_SYSTEM.rhs_values, dtype=np.complex128)
_STAR_TARGET = np.zeros((243, 3), dtype=np.complex128)
_STAR_TARGET[0, 0] = 1
_STAR_TARGET[121, 1] = 1
_STAR_TARGET[242, 2] = 1


def _independent_variable_index(
    left: int,
    right: int,
    left_color: int,
    right_color: int,
) -> int:
    if left > right:
        left, right = right, left
        left_color, right_color = right_color, left_color
    if (
        not 0 <= left < right < N
        or not 0 <= left_color < D
        or not 0 <= right_color < D
    ):
        raise KrennStarALSError(
            "an independent variable coordinate is invalid"
        )
    edge_index = left * (2 * N - left - 1) // 2 + (
        right - left - 1
    )
    return (edge_index * D + left_color) * D + right_color


@lru_cache(maxsize=1)
def _independent_monomials() -> np.ndarray:
    vertices = tuple(range(N))
    edges = tuple(combinations(vertices, 2))
    matchings = tuple(
        candidate
        for candidate in combinations(edges, N // 2)
        if len({
            vertex for edge in candidate for vertex in edge
        }) == N
    )
    if len(matchings) != 15:
        raise KrennStarALSError(
            "the independent perfect-matching census changed"
        )
    monomials = []
    for coloring in product(range(D), repeat=N):
        monomials.append(tuple(
            tuple(
                _independent_variable_index(
                    left,
                    right,
                    coloring[left],
                    coloring[right],
                )
                for left, right in matching
            )
            for matching in matchings
        ))
    result = np.asarray(monomials, dtype=np.int64)
    if result.shape != (EQUATIONS, 15, 3):
        raise KrennStarALSError(
            "the independent tensor map changed shape"
        )
    result.setflags(write=False)
    return result


def tensor_outputs(weights: Sequence[complex]) -> np.ndarray:
    """Evaluate all equations from the primary sparse system."""

    array = np.asarray(weights, dtype=np.complex128)
    if array.shape != (VARIABLES,) or not np.all(np.isfinite(array)):
        raise KrennStarALSError(
            "tensor evaluation needs 135 finite complex weights"
        )
    return np.prod(array[_MONOMIALS], axis=2).sum(axis=1)


def independent_tensor_outputs(
    weights: Sequence[complex],
) -> np.ndarray:
    """Evaluate all equations from an independently built enumerator."""

    array = np.asarray(weights, dtype=np.complex128)
    if array.shape != (VARIABLES,) or not np.all(np.isfinite(array)):
        raise KrennStarALSError(
            "independent evaluation needs 135 finite complex weights"
        )
    monomials = _independent_monomials()
    return np.prod(array[monomials], axis=2).sum(axis=1)


def _residual_from_outputs(outputs: np.ndarray) -> dict:
    residual = outputs - _TARGET
    mixed_mask = np.ones(EQUATIONS, dtype=bool)
    mixed_mask[[0, 364, 728]] = False
    absolute = np.abs(residual)
    return {
        "l2": float(np.linalg.norm(residual)),
        "maximum": float(absolute.max()),
        "mixed_l2": float(np.linalg.norm(residual[mixed_mask])),
        "pure_residual_abs": [
            float(value) for value in absolute[[0, 364, 728]]
        ],
        "above_1e-6": int(np.count_nonzero(absolute > 1e-6)),
        "above_1e-10": int(np.count_nonzero(absolute > 1e-10)),
        "all_729_equations_evaluated": True,
    }


def residual_accounting(weights: Sequence[complex]) -> dict:
    return _residual_from_outputs(tensor_outputs(weights))


def independent_residual_accounting(
    weights: Sequence[complex],
) -> dict:
    return _residual_from_outputs(independent_tensor_outputs(weights))


def dual_residual_accounting(weights: Sequence[complex]) -> dict:
    """Replay all 729 equations by both enumerators and compare outputs."""

    primary_outputs = tensor_outputs(weights)
    independent_outputs = independent_tensor_outputs(weights)
    difference = np.abs(primary_outputs - independent_outputs)
    maximum_scale = max(
        1.0,
        float(np.abs(primary_outputs).max()),
        float(np.abs(independent_outputs).max()),
    )
    allowed = _REPLAY_ATOL + _REPLAY_RTOL * maximum_scale
    maximum_difference = float(difference.max())
    if maximum_difference > allowed:
        raise KrennStarALSError(
            "primary and independent 729-equation replays disagree"
        )
    return {
        "primary": _residual_from_outputs(primary_outputs),
        "independent": _residual_from_outputs(independent_outputs),
        "agreement": {
            "maximum_output_difference": maximum_difference,
            "allowed_output_difference": allowed,
            "all_729_outputs_agree": True,
        },
    }


def numerical_star_matrix(
    weights: Sequence[complex], apex: int
) -> np.ndarray:
    """Evaluate one exact quadratic star matrix numerically."""

    values = np.asarray(weights, dtype=np.complex128)
    if values.shape != (VARIABLES,) or not np.all(np.isfinite(values)):
        raise KrennStarALSError(
            "a numerical star matrix needs 135 finite weights"
        )
    exact = star_linearization(apex)
    matrix = np.zeros((243, 15), dtype=np.complex128)
    for row_index, row in enumerate(exact.entries):
        for column_index, polynomial in enumerate(row):
            if polynomial:
                matrix[row_index, column_index] = sum(
                    values[left] * values[right]
                    for left, right in polynomial
                )
    return matrix


def _minimum_norm_correction(
    matrix: np.ndarray, residual_rhs: np.ndarray
) -> tuple[np.ndarray, dict]:
    if matrix.shape[0] != 243 or residual_rhs.shape != (243,):
        raise KrennStarALSError(
            "a minimum-change star correction changed shape"
        )
    if not matrix.shape[1]:
        return np.zeros(0, dtype=np.complex128), {
            "numerical_rank": 0,
            "condition_number_on_numerical_image": None,
        }
    u, singular_values, vh = np.linalg.svd(
        matrix, full_matrices=False
    )
    if not len(singular_values) or not singular_values[0]:
        return np.zeros(matrix.shape[1], dtype=np.complex128), {
            "numerical_rank": 0,
            "condition_number_on_numerical_image": None,
        }
    threshold = (
        np.finfo(float).eps
        * max(matrix.shape)
        * singular_values[0]
    )
    active = singular_values > threshold
    coordinates = np.zeros_like(singular_values, dtype=np.complex128)
    projected = u.conj().T @ residual_rhs
    coordinates[active] = (
        projected[active] / singular_values[active]
    )
    correction = vh.conj().T @ coordinates
    rank = int(np.count_nonzero(active))
    condition = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank
        else None
    )
    return correction, {
        "numerical_rank": rank,
        "condition_number_on_numerical_image": condition,
    }


def minimum_change_star_correction(
    weights: Sequence[complex], apex: int
) -> tuple[np.ndarray, dict]:
    """Return the free-coordinate correction for one fixed apex.

    The correction minimizes the residual linearized in the current star,
    not the norm of a replacement star.  Gauge anchors have zero
    correction by construction.
    """

    values = np.asarray(weights, dtype=np.complex128)
    if values.shape != (VARIABLES,) or not np.all(np.isfinite(values)):
        raise KrennStarALSError(
            "a star correction needs 135 finite complex weights"
        )
    exact = star_linearization(apex)
    matrix = numerical_star_matrix(values, apex)
    anchor_set = set(_gauge_anchor_indices())
    delta = np.zeros(VARIABLES, dtype=np.complex128)
    color_rows = []
    for apex_color in range(D):
        block = np.asarray(
            exact.star_variable_blocks[apex_color], dtype=np.int64
        )
        free_columns = tuple(
            column
            for column, index in enumerate(block)
            if int(index) not in anchor_set
        )
        current_star = values[block]
        residual_rhs = (
            _STAR_TARGET[:, apex_color] - matrix @ current_star
        )
        correction, diagnostic = _minimum_norm_correction(
            matrix[:, free_columns], residual_rhs
        )
        if free_columns:
            global_free = block[np.asarray(free_columns, dtype=np.int64)]
            delta[global_free] = correction
        predicted_after = residual_rhs.copy()
        if free_columns:
            predicted_after -= matrix[:, free_columns] @ correction
        color_rows.append({
            "apex_color": apex_color,
            "free_columns": list(free_columns),
            "fixed_anchor_columns": [
                column
                for column in range(15)
                if column not in free_columns
            ],
            "free_variables": len(free_columns),
            "numerical_rank": diagnostic["numerical_rank"],
            "condition_number_on_numerical_image": diagnostic[
                "condition_number_on_numerical_image"
            ],
            "correction_l2": float(np.linalg.norm(correction)),
            "block_residual_l2_before": float(
                np.linalg.norm(residual_rhs)
            ),
            "untruncated_block_residual_l2_after": float(
                np.linalg.norm(predicted_after)
            ),
        })
    if np.any(delta[np.asarray(_gauge_anchor_indices(), dtype=np.int64)]):
        raise KrennStarALSError(
            "a star correction attempted to move a gauge anchor"
        )
    return delta, {
        "apex": int(apex),
        "minimum_change_not_origin_replacement": True,
        "free_star_variables": sum(
            row["free_variables"] for row in color_rows
        ),
        "correction_l2": float(np.linalg.norm(delta)),
        "by_apex_color": color_rows,
    }


def _within_caps(
    weights: np.ndarray, config: StarALSConfig
) -> bool:
    return bool(
        np.abs(weights).max() <= config.global_linf_cap
        and np.linalg.norm(weights)
        <= config.effective_global_l2_cap
    )


def _positive_boundary_root(
    start: complex, direction: complex, radius: float
) -> float:
    quadratic = float(abs(direction) ** 2)
    if quadratic == 0:
        return math.inf
    linear = float(2 * np.real(np.conj(start) * direction))
    constant = float(abs(start) ** 2 - radius**2)
    discriminant = max(0.0, linear * linear - 4 * quadratic * constant)
    root = (-linear + math.sqrt(discriminant)) / (2 * quadratic)
    return max(0.0, root)


def _maximum_feasible_alpha(
    weights: np.ndarray,
    delta: np.ndarray,
    config: StarALSConfig,
) -> tuple[float, list[str]]:
    """Find the deterministic maximal alpha in [0,1] allowed by both caps."""

    if not _within_caps(weights, config):
        raise KrennStarALSError(
            "the current trajectory already violates a global cap"
        )
    if delta.shape != (VARIABLES,) or not np.all(np.isfinite(delta)):
        raise KrennStarALSError("a proposed correction is malformed")
    alpha = 1.0
    limiting: list[str] = []
    changed = np.flatnonzero(delta)
    for index in changed:
        if abs(weights[index] + delta[index]) > config.global_linf_cap:
            bound = _positive_boundary_root(
                weights[index], delta[index], config.global_linf_cap
            )
            if bound < alpha:
                alpha = bound
                limiting = ["global-linf"]
            elif math.isclose(
                bound, alpha, rel_tol=8e-15, abs_tol=8e-15
            ):
                if "global-linf" not in limiting:
                    limiting.append("global-linf")

    quadratic = float(np.vdot(delta, delta).real)
    if quadratic:
        linear = float(2 * np.vdot(weights, delta).real)
        constant = float(
            np.vdot(weights, weights).real
            - config.effective_global_l2_cap**2
        )
        at_one = quadratic + linear + constant
        if at_one > 0:
            discriminant = max(
                0.0, linear * linear - 4 * quadratic * constant
            )
            bound = (
                -linear + math.sqrt(discriminant)
            ) / (2 * quadratic)
            bound = max(0.0, bound)
            if bound < alpha:
                alpha = bound
                limiting = ["global-l2"]
            elif math.isclose(
                bound, alpha, rel_tol=8e-15, abs_tol=8e-15
            ):
                if "global-l2" not in limiting:
                    limiting.append("global-l2")

    alpha = min(1.0, max(0.0, float(alpha)))
    proposal = weights + alpha * delta
    if not _within_caps(proposal, config):
        low = 0.0
        high = alpha
        for _iteration in range(100):
            middle = (low + high) / 2.0
            if _within_caps(weights + middle * delta, config):
                low = middle
            else:
                high = middle
        alpha = low
        proposal = weights + alpha * delta
    if 0.0 < alpha < 1.0:
        # Stay strictly inside a numerically computed boundary.  Rebuilding
        # the same complex vector later can otherwise differ by one ULP in
        # a BLAS norm reduction on Windows.
        alpha *= 1.0 - 32.0 * np.finfo(float).eps
        proposal = weights + alpha * delta
    if not _within_caps(proposal, config):
        raise KrennStarALSError(
            "the analytically truncated correction violates a cap"
        )
    return alpha, limiting


def _cross_ratio_diagnostics(weights: np.ndarray) -> dict:
    logs = []
    threshold = np.finfo(float).tiny
    for left, right in canonical_edges(N):
        for left_a in range(D):
            for left_b in range(left_a + 1, D):
                for right_a in range(D):
                    for right_b in range(right_a + 1, D):
                        diagonal = (
                            weights[variable_index(
                                N, D, left, right, left_a, right_a
                            )]
                            * weights[variable_index(
                                N, D, left, right, left_b, right_b
                            )]
                        )
                        off_diagonal = (
                            weights[variable_index(
                                N, D, left, right, left_a, right_b
                            )]
                            * weights[variable_index(
                                N, D, left, right, left_b, right_a
                            )]
                        )
                        if (
                            abs(diagonal) > threshold
                            and abs(off_diagonal) > threshold
                        ):
                            logs.append(float(
                                np.log(abs(diagonal / off_diagonal))
                            ))
    pole_product = np.prod(
        weights[np.asarray(POLE_INVARIANT_INDICES, dtype=np.int64)]
    )
    return {
        "defined_edge_color_cross_ratios": len(logs),
        "maximum_absolute_log_cross_ratio": (
            max(map(abs, logs)) if logs else None
        ),
        "known_vertex_scalar_pole_product_abs": float(abs(pole_product)),
    }


def _weight_accounting(
    weights: np.ndarray, config: StarALSConfig
) -> dict:
    absolute = np.abs(weights)
    anchors = np.asarray(_gauge_anchor_indices(), dtype=np.int64)
    return {
        "maximum_abs": float(absolute.max()),
        "l2": float(np.linalg.norm(weights)),
        "nonzero_above_1e-12": int(
            np.count_nonzero(absolute > 1e-12)
        ),
        "global_linf_cap": config.global_linf_cap,
        "global_l2_cap": config.effective_global_l2_cap,
        "within_both_global_caps": _within_caps(weights, config),
        "all_15_gauge_anchors_exactly_one": bool(
            np.all(weights[anchors] == GAUGE_ANCHOR_VALUE)
        ),
    }


def _initial_weights(config: StarALSConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    noise = (
        rng.standard_normal(VARIABLES)
        + 1j * rng.standard_normal(VARIABLES)
    )
    anchors = np.asarray(_gauge_anchor_indices(), dtype=np.int64)
    free_mask = np.ones(VARIABLES, dtype=bool)
    free_mask[anchors] = False
    result = np.zeros(VARIABLES, dtype=np.complex128)
    if config.initialization == "natural-perturbed":
        for index, value in seed_witness(
            NATURAL_REPRESENTATIVE
        ).entries:
            if free_mask[index]:
                result[index] = complex(value)
        result[free_mask] += 0.01 * noise[free_mask]
    else:
        result[free_mask] = noise[free_mask]
    result[anchors] = GAUGE_ANCHOR_VALUE

    free_maximum = float(np.abs(result[free_mask]).max())
    if free_maximum:
        target_linf = (
            config.global_linf_cap
            if config.initialization == "natural-perturbed"
            else 0.05 * config.global_linf_cap
        )
        linf_scale = min(1.0, target_linf / free_maximum)
        result[free_mask] *= linf_scale
    anchor_l2_squared = float(len(anchors))
    free_l2 = float(np.linalg.norm(result[free_mask]))
    available_squared = max(
        0.0,
        config.effective_global_l2_cap**2 - anchor_l2_squared,
    )
    if free_l2 and free_l2**2 > available_squared:
        result[free_mask] *= (
            math.sqrt(available_squared)
            / free_l2
            * float(np.nextafter(1.0, 0.0))
        )
    result[anchors] = GAUGE_ANCHOR_VALUE
    if not _within_caps(result, config):
        raise KrennStarALSError(
            "an initializer exceeded a persistent global bound"
        )
    return result


def _apex_schedule(
    config: StarALSConfig, sweep: int
) -> tuple[str, tuple[int, ...]]:
    if not 1 <= sweep <= config.maximum_sweeps:
        raise KrennStarALSError("an apex schedule has an invalid sweep")
    rotation = (config.seed + sweep - 1) % N
    forward = tuple(
        (rotation + offset) % N for offset in range(N)
    )
    if (config.seed + sweep) % 2:
        return "reverse", tuple(reversed(forward))
    return "forward", forward


def _source_fingerprints() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    filenames = (
        "star_alternating_search.py",
        "star_linearization.py",
        "star_linearization_independent.py",
        "system.py",
        "vertical_component.py",
    )
    result = {}
    for filename in filenames:
        path = directory / filename
        try:
            data = path.read_bytes()
        except OSError as error:
            raise KrennStarALSError(
                f"could not fingerprint source file {filename}"
            ) from error
        result[filename] = hashlib.sha256(data).hexdigest()
    return result


@lru_cache(maxsize=1)
def _star_fingerprints() -> tuple[dict, ...]:
    rows = []
    for apex in range(N):
        primary = star_linearization(apex).fingerprint()
        independent = independent_star_linearization_audit(apex)[
            "fingerprint"
        ]
        if primary != independent:
            raise KrennStarALSError(
                "primary and independent star fingerprints disagree"
            )
        rows.append({
            "apex": apex,
            "primary_sha256": primary,
            "independent_sha256": independent,
        })
    return tuple(rows)


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise KrennStarALSError(
            "refusing a non-regular star checkpoint target"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() and (
        temporary.is_symlink() or not temporary.is_file()
    ):
        raise KrennStarALSError(
            "refusing a non-regular star checkpoint temporary"
        )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for attempt in range(40):
        try:
            os.replace(temporary, path)
            break
        except PermissionError as error:
            # Windows readers, antivirus, and indexers can briefly retain a
            # handle to the destination.  Retrying the same atomic replace
            # preserves the checkpoint transaction.
            if attempt == 39:
                raise KrennStarALSError(
                    "an atomic checkpoint replace remained locked"
                ) from error
            time.sleep(min(0.005 * (attempt + 1), 0.1))


def _numbers_close(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return bool(np.isclose(
            left, right, rtol=5e-14, atol=5e-15
        ))
    if isinstance(left, dict):
        return (
            left.keys() == right.keys()
            and all(_numbers_close(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return (
            len(left) == len(right)
            and all(
                _numbers_close(a, b)
                for a, b in zip(left, right, strict=True)
            )
        )
    return left == right


def _assert_chart_and_caps(
    weights: np.ndarray,
    config: StarALSConfig,
    *,
    label: str,
) -> None:
    anchors = np.asarray(_gauge_anchor_indices(), dtype=np.int64)
    if not np.all(weights[anchors] == GAUGE_ANCHOR_VALUE):
        raise KrennStarALSError(
            f"{label} does not lie in the fixed gauge slice"
        )
    if not _within_caps(weights, config):
        raise KrennStarALSError(
            f"{label} violates a persistent global bound"
        )


def _cursor_from_updates(
    completed_updates: int, maximum_sweeps: int
) -> dict:
    if (
        type(completed_updates) is not int
        or not 0 <= completed_updates <= N * maximum_sweeps
    ):
        raise KrennStarALSError("a checkpoint update count is invalid")
    return {
        "next_sweep": completed_updates // N + 1,
        "next_position": completed_updates % N,
        "completed_sweeps": completed_updates // N,
        "completed_apex_updates": completed_updates,
    }


def _compact_residual(accounting: Mapping) -> dict:
    return {
        "primary_l2": accounting["primary"]["l2"],
        "primary_maximum": accounting["primary"]["maximum"],
        "independent_l2": accounting["independent"]["l2"],
        "independent_maximum": accounting["independent"]["maximum"],
        "maximum_output_difference": accounting["agreement"][
            "maximum_output_difference"
        ],
    }


def _validate_trace(
    trace: Sequence[Mapping],
    config: StarALSConfig,
    initial_residual: Mapping,
    current_residual: Mapping,
    cursor: Mapping,
) -> None:
    if type(trace) is not list:
        raise KrennStarALSError("a checkpoint trace is malformed")
    if len(trace) != cursor["completed_apex_updates"]:
        raise KrennStarALSError(
            "a checkpoint trace and cursor disagree"
        )
    previous = _compact_residual(initial_residual)
    previous_elapsed = 0.0
    expected_trace_keys = {
        "update_index",
        "sweep",
        "position_in_sweep",
        "direction",
        "schedule",
        "apex",
        "status",
        "accepted",
        "maximum_alpha_from_global_bounds",
        "accepted_alpha",
        "backtracks",
        "limiting_global_caps",
        "minimum_change_correction",
        "star_block_residual_l2_before",
        "star_block_residual_l2_after",
        "residual_before",
        "residual_after",
        "relative_primary_l2_improvement",
        "patience_count_after",
        "weights_after",
        "gauge_invariant_growth",
        "cumulative_elapsed_seconds",
    }
    allowed_update_statuses = {
        "zero-correction",
        "bound-stalled",
        "accepted-backtracked",
        "accepted-bound-truncated",
        "accepted-full-correction",
        "rejected-no-monotone-step",
    }
    for update_index, row in enumerate(trace, start=1):
        if type(row) is not dict:
            raise KrennStarALSError("a checkpoint trace row is malformed")
        sweep = (update_index - 1) // N + 1
        position = (update_index - 1) % N
        direction, schedule = _apex_schedule(config, sweep)
        if (
            set(row) != expected_trace_keys
            or row.get("update_index") != update_index
            or row.get("sweep") != sweep
            or row.get("position_in_sweep") != position
            or row.get("direction") != direction
            or row.get("schedule") != list(schedule)
            or row.get("apex") != schedule[position]
            or row.get("status") not in allowed_update_statuses
            or type(row.get("accepted")) is not bool
            or (
                row["accepted"]
                != str(row["status"]).startswith("accepted-")
            )
            or type(row.get("backtracks")) is not int
            or row["backtracks"] < 0
            or type(row.get("limiting_global_caps")) is not list
            or any(
                cap not in {"global-l2", "global-linf"}
                for cap in row["limiting_global_caps"]
            )
            or not np.isfinite(float(
                row.get("maximum_alpha_from_global_bounds")
            ))
            or not 0 <= row["maximum_alpha_from_global_bounds"] <= 1
            or not np.isfinite(float(row.get("accepted_alpha")))
            or not 0 <= row["accepted_alpha"] <= (
                row["maximum_alpha_from_global_bounds"] + 1e-15
            )
            or type(row.get("weights_after")) is not dict
            or not row["weights_after"].get(
                "within_both_global_caps", False
            )
            or not row["weights_after"].get(
                "all_15_gauge_anchors_exactly_one", False
            )
            or type(row.get("patience_count_after")) is not int
            or not 0 <= row["patience_count_after"] <= (
                config.patience_apex_updates
            )
            or not _numbers_close(row.get("residual_before"), previous)
            or type(row.get("residual_after")) is not dict
            or row["residual_after"]["primary_l2"]
            > row["residual_before"]["primary_l2"]
            or row["residual_after"]["independent_l2"]
            > row["residual_before"]["independent_l2"]
            or not np.isfinite(float(row.get("cumulative_elapsed_seconds")))
            or row["cumulative_elapsed_seconds"] < previous_elapsed
        ):
            raise KrennStarALSError(
                "a checkpoint trace failed deterministic replay"
            )
        previous = row["residual_after"]
        previous_elapsed = row["cumulative_elapsed_seconds"]
    if trace and not _numbers_close(
        trace[-1]["residual_after"],
        _compact_residual(current_residual),
    ):
        raise KrennStarALSError(
            "a checkpoint trace does not reach its current residual"
        )


def _checkpoint_payload(
    *,
    config: StarALSConfig,
    status: str,
    cursor: Mapping,
    weights: np.ndarray,
    best_weights: np.ndarray,
    initial_weights: np.ndarray,
    initial_residual: Mapping,
    current_residual: Mapping,
    best_residual: Mapping,
    trace: Sequence[Mapping],
    patience_count: int,
    cumulative_elapsed: float,
) -> dict:
    return {
        "schema": STAR_ALS_CHECKPOINT_SCHEMA,
        "config": config.to_dict(),
        "effective_bounds": config.effective_bounds(),
        "config_sha256": config.fingerprint(),
        "source_sha256": _source_fingerprints(),
        "star_sha256": list(_star_fingerprints()),
        "gauge_chart": gauge_chart_audit(),
        "state": {
            "status": status,
            "cursor": dict(cursor),
            "weights": _complex_rows(weights),
            "best_weights": _complex_rows(best_weights),
            "initial_weights": _complex_rows(initial_weights),
            "initial_residual": initial_residual,
            "current_residual": current_residual,
            "best_residual": best_residual,
            "trace": list(trace),
            "patience_count": patience_count,
            "cumulative_elapsed_seconds": float(cumulative_elapsed),
        },
        "claim_boundary": {
            "numerical_zero_is_exact_counterexample": False,
            "bounded_miss_is_nonexistence_proof": False,
            "open_gauge_chart_is_global_affine_space": False,
        },
    }


def _load_checkpoint(
    path: Path, config: StarALSConfig
) -> dict:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise KrennStarALSError(
            "a star checkpoint could not be loaded strictly"
        ) from error
    expected_top = {
        "schema",
        "config",
        "effective_bounds",
        "config_sha256",
        "source_sha256",
        "star_sha256",
        "gauge_chart",
        "state",
        "claim_boundary",
    }
    if (
        type(payload) is not dict
        or set(payload) != expected_top
        or payload.get("schema") != STAR_ALS_CHECKPOINT_SCHEMA
        or payload.get("config") != config.to_dict()
        or payload.get("effective_bounds") != config.effective_bounds()
        or payload.get("config_sha256") != config.fingerprint()
        or payload.get("source_sha256") != _source_fingerprints()
        or payload.get("star_sha256") != list(_star_fingerprints())
        or payload.get("gauge_chart") != gauge_chart_audit()
        or payload.get("claim_boundary") != {
            "numerical_zero_is_exact_counterexample": False,
            "bounded_miss_is_nonexistence_proof": False,
            "open_gauge_chart_is_global_affine_space": False,
        }
        or type(payload.get("state")) is not dict
    ):
        raise KrennStarALSError(
            "a star checkpoint does not match its deterministic run"
        )
    state = payload["state"]
    expected_state = {
        "status",
        "cursor",
        "weights",
        "best_weights",
        "initial_weights",
        "initial_residual",
        "current_residual",
        "best_residual",
        "trace",
        "patience_count",
        "cumulative_elapsed_seconds",
    }
    if (
        set(state) != expected_state
        or state["status"] not in {
            "running",
            "maximum-sweeps",
            "numerical-tolerance",
            "time-budget",
            "patience-exhausted",
        }
        or type(state["cursor"]) is not dict
        or type(state["patience_count"]) is not int
        or not 0 <= state["patience_count"]
        <= config.patience_apex_updates
        or not np.isfinite(float(state["cumulative_elapsed_seconds"]))
        or state["cumulative_elapsed_seconds"] < 0
    ):
        raise KrennStarALSError(
            "a star checkpoint state is malformed"
        )
    cursor = state["cursor"]
    if set(cursor) != {
        "next_sweep",
        "next_position",
        "completed_sweeps",
        "completed_apex_updates",
    }:
        raise KrennStarALSError("a checkpoint cursor is malformed")
    expected_cursor = _cursor_from_updates(
        cursor.get("completed_apex_updates"), config.maximum_sweeps
    )
    if cursor != expected_cursor:
        raise KrennStarALSError(
            "a checkpoint cursor is not the deterministic next cursor"
        )

    weights = _complex_array(state["weights"], label="current")
    best_weights = _complex_array(
        state["best_weights"], label="best"
    )
    initial_weights = _complex_array(
        state["initial_weights"], label="initial"
    )
    for label, values in (
        ("current weights", weights),
        ("best weights", best_weights),
        ("initial weights", initial_weights),
    ):
        _assert_chart_and_caps(values, config, label=label)
    replay_initial = dual_residual_accounting(initial_weights)
    replay_current = dual_residual_accounting(weights)
    replay_best = dual_residual_accounting(best_weights)
    if (
        not _numbers_close(
            state["initial_residual"], replay_initial
        )
        or not _numbers_close(
            state["current_residual"], replay_current
        )
        or not _numbers_close(state["best_residual"], replay_best)
        or state["best_residual"]["primary"]["l2"]
        > state["current_residual"]["primary"]["l2"]
    ):
        raise KrennStarALSError(
            "checkpoint weights failed full residual replay"
        )
    _validate_trace(
        state["trace"],
        config,
        state["initial_residual"],
        state["current_residual"],
        cursor,
    )
    primary_l2 = state["current_residual"]["primary"]["l2"]
    independent_l2 = state["current_residual"]["independent"]["l2"]
    at_tolerance = (
        primary_l2 <= config.residual_tolerance
        and independent_l2 <= config.residual_tolerance
    )
    out_of_patience = (
        state["patience_count"] >= config.patience_apex_updates
    )
    out_of_time = (
        state["cumulative_elapsed_seconds"] >= config.maximum_seconds
    )
    out_of_sweeps = (
        cursor["next_sweep"] > config.maximum_sweeps
    )
    terminal_consistent = (
        (
            state["status"] == "running"
            and not at_tolerance
            and not out_of_patience
            and not out_of_time
            and not out_of_sweeps
        )
        or (
            state["status"] == "maximum-sweeps"
            and not at_tolerance
            and not out_of_patience
            and not out_of_time
            and out_of_sweeps
        )
        or (
            state["status"] == "numerical-tolerance"
            and at_tolerance
        )
        or (
            state["status"] == "time-budget"
            and not at_tolerance
            and not out_of_patience
            and out_of_time
        )
        or (
            state["status"] == "patience-exhausted"
            and not at_tolerance
            and out_of_patience
        )
    )
    if not terminal_consistent:
        raise KrennStarALSError(
            "checkpoint status is inconsistent with its replayed state"
        )
    if state["trace"] and not np.isclose(
        state["trace"][-1]["cumulative_elapsed_seconds"],
        state["cumulative_elapsed_seconds"],
        rtol=0,
        atol=5e-12,
    ):
        raise KrennStarALSError(
            "checkpoint elapsed accounting does not reach its trace"
        )
    if state["trace"] and (
        state["trace"][-1]["patience_count_after"]
        != state["patience_count"]
    ):
        raise KrennStarALSError(
            "checkpoint patience accounting does not reach its trace"
        )
    return {
        **state,
        "weights_array": weights,
        "best_weights_array": best_weights,
        "initial_weights_array": initial_weights,
    }


def _star_block_residual_l2(
    matrix: np.ndarray, weights: np.ndarray, apex: int
) -> float:
    exact = star_linearization(apex)
    residuals = []
    for color in range(D):
        block = np.asarray(
            exact.star_variable_blocks[color], dtype=np.int64
        )
        residuals.append(
            matrix @ weights[block] - _STAR_TARGET[:, color]
        )
    return float(np.linalg.norm(np.column_stack(residuals)))


def _result_payload(
    *,
    config: StarALSConfig,
    status: str,
    cursor: Mapping,
    weights: np.ndarray,
    best_weights: np.ndarray,
    initial_residual: Mapping,
    current_residual: Mapping,
    best_residual: Mapping,
    trace: Sequence[Mapping],
    patience_count: int,
    cumulative_elapsed: float,
) -> dict:
    return {
        "schema": STAR_ALS_SCHEMA,
        "config": config.to_dict(),
        "effective_bounds": config.effective_bounds(),
        "config_sha256": config.fingerprint(),
        "source_sha256": _source_fingerprints(),
        "star_sha256": list(_star_fingerprints()),
        "gauge_chart": gauge_chart_audit(),
        "completed_sweeps": cursor["completed_sweeps"],
        "completed_apex_updates": cursor["completed_apex_updates"],
        "cursor": dict(cursor),
        "termination": status,
        "patience_count": patience_count,
        "cumulative_elapsed_seconds": float(cumulative_elapsed),
        "initial_residual": initial_residual,
        "final_residual": current_residual,
        "best_residual": best_residual,
        "weights": _complex_rows(weights),
        "best_weights": _complex_rows(best_weights),
        "trace": list(trace),
        "finite_bound": {
            **config.effective_bounds(),
            "final_weights": _weight_accounting(weights, config),
            "best_weights": _weight_accounting(best_weights, config),
            "all_updates_within_both_global_bounds": all(
                row["weights_after"]["within_both_global_caps"]
                for row in trace
            ),
        },
        "gauge_invariant_growth": _cross_ratio_diagnostics(weights),
        "independent_verification": {
            "all_729_equations_replayed": True,
            "primary_and_independent_outputs_agree": True,
            "maximum_output_difference": current_residual["agreement"][
                "maximum_output_difference"
            ],
        },
        "exact_verification": {
            "attempted": False,
            "status": "not-applicable-to-floating-point-trajectory",
        },
        "claim_boundary": {
            "numerical_zero_is_exact_counterexample": False,
            "bounded_miss_is_nonexistence_proof": False,
            "increasing_radius_trend_is_border_proof": False,
            "open_chart_covers_zero_anchor_solutions": False,
            "finite_affine_membership_status": "undecided",
        },
    }


def run_star_als(
    config: StarALSConfig,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run one deterministic, bounded, gauge-fixed star trajectory."""

    if not isinstance(config, StarALSConfig):
        raise KrennStarALSError("run_star_als needs a validated config")
    checkpoint = (
        None if checkpoint_path is None else Path(checkpoint_path)
    )
    if resume:
        if checkpoint is None or not checkpoint.is_file():
            raise KrennStarALSError(
                "resume needs an existing regular checkpoint"
            )
        loaded = _load_checkpoint(checkpoint, config)
        weights = loaded["weights_array"]
        best_weights = loaded["best_weights_array"]
        initial_weights = loaded["initial_weights_array"]
        initial_residual = loaded["initial_residual"]
        current_residual = loaded["current_residual"]
        best_residual = loaded["best_residual"]
        trace = loaded["trace"]
        patience_count = loaded["patience_count"]
        cumulative_before = float(
            loaded["cumulative_elapsed_seconds"]
        )
        cursor = loaded["cursor"]
        status = loaded["status"]
    else:
        if checkpoint is not None and checkpoint.exists():
            raise KrennStarALSError(
                "a checkpoint already exists; use resume"
            )
        weights = _initial_weights(config)
        initial_weights = weights.copy()
        initial_residual = dual_residual_accounting(weights)
        current_residual = initial_residual
        best_residual = initial_residual
        best_weights = weights.copy()
        trace: list[dict] = []
        patience_count = 0
        cumulative_before = 0.0
        cursor = _cursor_from_updates(0, config.maximum_sweeps)
        status = "running"

    if status != "running":
        return _result_payload(
            config=config,
            status=status,
            cursor=cursor,
            weights=weights,
            best_weights=best_weights,
            initial_residual=initial_residual,
            current_residual=current_residual,
            best_residual=best_residual,
            trace=trace,
            patience_count=patience_count,
            cumulative_elapsed=cumulative_before,
        )

    segment_start = time.monotonic()
    while cursor["next_sweep"] <= config.maximum_sweeps:
        sweep = cursor["next_sweep"]
        position = cursor["next_position"]
        direction, schedule = _apex_schedule(config, sweep)
        apex = schedule[position]
        update_index = cursor["completed_apex_updates"] + 1
        before = current_residual
        before_compact = _compact_residual(before)
        matrix = numerical_star_matrix(weights, apex)
        block_before = _star_block_residual_l2(
            matrix, weights, apex
        )
        if not np.isclose(
            block_before,
            before["primary"]["l2"],
            rtol=2e-13,
            atol=2e-14,
        ):
            raise KrennStarALSError(
                "the star block and full 729 residual disagree"
            )
        delta, correction_diagnostic = minimum_change_star_correction(
            weights, apex
        )
        alpha_bound, limiting_caps = _maximum_feasible_alpha(
            weights, delta, config
        )
        accepted_alpha = alpha_bound
        accepted = False
        candidate_residual = before
        proposal = weights
        backtracks = 0
        correction_norm = float(np.linalg.norm(delta))
        if correction_norm == 0:
            update_status = "zero-correction"
            accepted_alpha = 0.0
        elif alpha_bound == 0:
            update_status = "bound-stalled"
        else:
            while accepted_alpha > 0 and backtracks <= 48:
                candidate = weights + accepted_alpha * delta
                candidate[np.asarray(
                    _gauge_anchor_indices(), dtype=np.int64
                )] = GAUGE_ANCHOR_VALUE
                if not _within_caps(candidate, config):
                    raise KrennStarALSError(
                        "a truncated apex proposal violated a global cap"
                    )
                replay = dual_residual_accounting(candidate)
                if (
                    replay["primary"]["l2"]
                    <= before["primary"]["l2"]
                    and replay["independent"]["l2"]
                    <= before["independent"]["l2"]
                    and (
                        replay["primary"]["l2"]
                        < before["primary"]["l2"]
                        or replay["independent"]["l2"]
                        < before["independent"]["l2"]
                    )
                ):
                    proposal = candidate
                    candidate_residual = replay
                    accepted = True
                    break
                accepted_alpha *= 0.5
                backtracks += 1
            if accepted:
                if backtracks:
                    update_status = "accepted-backtracked"
                elif alpha_bound < 1.0:
                    update_status = "accepted-bound-truncated"
                else:
                    update_status = "accepted-full-correction"
            else:
                update_status = "rejected-no-monotone-step"
                accepted_alpha = 0.0

        if accepted:
            weights = proposal
            current_residual = candidate_residual
        _assert_chart_and_caps(weights, config, label="accepted weights")
        block_after = _star_block_residual_l2(matrix, weights, apex)
        after_compact = _compact_residual(current_residual)
        if (
            after_compact["primary_l2"]
            > before_compact["primary_l2"]
            or after_compact["independent_l2"]
            > before_compact["independent_l2"]
            or block_after > block_before * (1 + 2e-13) + 2e-14
            or not np.isclose(
                block_after,
                current_residual["primary"]["l2"],
                rtol=2e-13,
                atol=2e-14,
            )
        ):
            raise KrennStarALSError(
                "an accepted apex update was not residual-monotone"
            )

        relative_improvement = (
            before_compact["primary_l2"]
            - after_compact["primary_l2"]
        ) / max(before_compact["primary_l2"], np.finfo(float).tiny)
        if relative_improvement > config.minimum_relative_improvement:
            patience_count = 0
        else:
            patience_count += 1
        if (
            current_residual["primary"]["l2"]
            < best_residual["primary"]["l2"]
        ):
            best_residual = current_residual
            best_weights = weights.copy()

        cursor = _cursor_from_updates(
            update_index, config.maximum_sweeps
        )
        cumulative_elapsed = (
            cumulative_before + time.monotonic() - segment_start
        )
        trace_row = {
            "update_index": update_index,
            "sweep": sweep,
            "position_in_sweep": position,
            "direction": direction,
            "schedule": list(schedule),
            "apex": apex,
            "status": update_status,
            "accepted": accepted,
            "maximum_alpha_from_global_bounds": alpha_bound,
            "accepted_alpha": accepted_alpha,
            "backtracks": backtracks,
            "limiting_global_caps": limiting_caps,
            "minimum_change_correction": correction_diagnostic,
            "star_block_residual_l2_before": block_before,
            "star_block_residual_l2_after": block_after,
            "residual_before": before_compact,
            "residual_after": after_compact,
            "relative_primary_l2_improvement": relative_improvement,
            "patience_count_after": patience_count,
            "weights_after": _weight_accounting(weights, config),
            "gauge_invariant_growth": _cross_ratio_diagnostics(weights),
            "cumulative_elapsed_seconds": cumulative_elapsed,
        }
        trace.append(trace_row)
        if progress is not None:
            progress(trace_row)

        if (
            current_residual["primary"]["l2"]
            <= config.residual_tolerance
            and current_residual["independent"]["l2"]
            <= config.residual_tolerance
        ):
            status = "numerical-tolerance"
        elif patience_count >= config.patience_apex_updates:
            status = "patience-exhausted"
        elif cumulative_elapsed >= config.maximum_seconds:
            status = "time-budget"
        elif cursor["next_sweep"] > config.maximum_sweeps:
            status = "maximum-sweeps"
        else:
            status = "running"

        if checkpoint is not None:
            _write_json_atomic(
                checkpoint,
                _checkpoint_payload(
                    config=config,
                    status=status,
                    cursor=cursor,
                    weights=weights,
                    best_weights=best_weights,
                    initial_weights=initial_weights,
                    initial_residual=initial_residual,
                    current_residual=current_residual,
                    best_residual=best_residual,
                    trace=trace,
                    patience_count=patience_count,
                    cumulative_elapsed=cumulative_elapsed,
                ),
            )
        if status != "running":
            break

    cumulative_elapsed = (
        cumulative_before + time.monotonic() - segment_start
    )
    if trace:
        trace[-1]["cumulative_elapsed_seconds"] = cumulative_elapsed
        if checkpoint is not None:
            _write_json_atomic(
                checkpoint,
                _checkpoint_payload(
                    config=config,
                    status=status,
                    cursor=cursor,
                    weights=weights,
                    best_weights=best_weights,
                    initial_weights=initial_weights,
                    initial_residual=initial_residual,
                    current_residual=current_residual,
                    best_residual=best_residual,
                    trace=trace,
                    patience_count=patience_count,
                    cumulative_elapsed=cumulative_elapsed,
                ),
            )
    return _result_payload(
        config=config,
        status=status,
        cursor=cursor,
        weights=weights,
        best_weights=best_weights,
        initial_residual=initial_residual,
        current_residual=current_residual,
        best_residual=best_residual,
        trace=trace,
        patience_count=patience_count,
        cumulative_elapsed=cumulative_elapsed,
    )


def run_star_als_campaign(
    *,
    output_directory: Path = DEFAULT_SCRATCH_DIRECTORY,
    radii: Sequence[float] = DEFAULT_RADII,
    global_l2_radii: Sequence[float] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    maximum_sweeps: int = 200,
    maximum_seconds_per_run: float = 600.0,
    patience_apex_updates: int = 120,
    minimum_relative_improvement: float = 1e-14,
    initialization: str = "random",
    resume: bool = False,
) -> dict:
    """Run a deterministic sequential radius/seed campaign."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise KrennStarALSError(
            "the star campaign output must be a real directory"
        )
    normalized_radii = tuple(float(radius) for radius in radii)
    normalized_l2_radii = (
        (None,) * len(normalized_radii)
        if global_l2_radii is None
        else tuple(float(radius) for radius in global_l2_radii)
    )
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if (
        not normalized_radii
        or not normalized_seeds
        or len(set(normalized_radii)) != len(normalized_radii)
        or len(normalized_l2_radii) != len(normalized_radii)
        or len(set(normalized_seeds)) != len(normalized_seeds)
    ):
        raise KrennStarALSError(
            "campaign Linf/L2 caps and seeds must be nonempty, aligned, "
            "and unique"
        )
    results = []
    for radius, l2_radius in zip(
        normalized_radii, normalized_l2_radii, strict=True
    ):
        for seed in normalized_seeds:
            config = StarALSConfig(
                radius=radius,
                global_l2_radius=l2_radius,
                maximum_sweeps=maximum_sweeps,
                seed=seed,
                initialization=initialization,
                maximum_seconds=maximum_seconds_per_run,
                patience_apex_updates=patience_apex_updates,
                minimum_relative_improvement=minimum_relative_improvement,
            )
            radius_text = format(radius, ".12g").replace(".", "p")
            l2_text = (
                "default"
                if l2_radius is None
                else format(l2_radius, ".12g").replace(".", "p")
            )
            checkpoint = output_directory / (
                f"linf_{radius_text}_l2_{l2_text}_seed_{seed}"
                ".checkpoint.json"
            )
            should_resume = resume and checkpoint.is_file()
            result = run_star_als(
                config,
                checkpoint_path=checkpoint,
                resume=should_resume,
            )
            result_path = output_directory / (
                f"linf_{radius_text}_l2_{l2_text}_seed_{seed}"
                ".result.json"
            )
            _write_json_atomic(result_path, result)
            results.append({
                "radius": radius,
                "effective_bounds": config.effective_bounds(),
                "seed": seed,
                "result_path": result_path.name,
                "completed_sweeps": result["completed_sweeps"],
                "completed_apex_updates": result[
                    "completed_apex_updates"
                ],
                "termination": result["termination"],
                "elapsed_seconds": result[
                    "cumulative_elapsed_seconds"
                ],
                "final_residual_l2": result["final_residual"][
                    "primary"
                ]["l2"],
                "independent_final_residual_l2": result[
                    "final_residual"
                ]["independent"]["l2"],
                "maximum_weight_abs": result["finite_bound"][
                    "final_weights"
                ]["maximum_abs"],
                "weight_l2": result["finite_bound"][
                    "final_weights"
                ]["l2"],
            })
    best = min(results, key=lambda row: row["final_residual_l2"])
    manifest = {
        "schema": STAR_ALS_CAMPAIGN_SCHEMA,
        "output_directory": str(output_directory),
        "gauge_chart": gauge_chart_audit(),
        "radii_as_global_linf_caps": list(normalized_radii),
        "explicit_global_l2_caps": (
            None
            if global_l2_radii is None
            else list(normalized_l2_radii)
        ),
        "default_global_l2_cap_rule": "global_linf_cap*sqrt(135)",
        "seeds": list(normalized_seeds),
        "maximum_sweeps_per_run": maximum_sweeps,
        "maximum_seconds_per_run": maximum_seconds_per_run,
        "patience_apex_updates": patience_apex_updates,
        "minimum_relative_improvement": minimum_relative_improvement,
        "initialization": initialization,
        "workers": 1,
        "schedule": (
            "sequential overlapping apex updates; rotating, alternating "
            "forward/reverse sweeps"
        ),
        "randomness": "deterministic NumPy PCG64 seeds",
        "checkpointing": "atomic after every apex update",
        "results": results,
        "best_bounded_candidate": best,
        "claim_boundary": {
            "numerical_candidate_is_exact_witness": False,
            "bounded_campaign_miss_is_nonexistence_proof": False,
            "open_chart_search_is_global_affine_search": False,
            "finite_affine_membership_status": "undecided",
        },
    }
    _write_json_atomic(output_directory / "manifest.json", manifest)
    return manifest


def _comma_values(text: str, converter) -> tuple:
    try:
        result = tuple(converter(piece) for piece in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "comma-separated campaign values are invalid"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError(
            "comma-separated values cannot be empty"
        )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded deterministic complex star ALS on the fixed "
            "rank-15 direct-GHZ gauge chart."
        )
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_SCRATCH_DIRECTORY,
    )
    parser.add_argument(
        "--radii",
        default=",".join(map(str, DEFAULT_RADII)),
        help="comma-separated persistent global Linf caps",
    )
    parser.add_argument(
        "--global-l2-caps",
        default=None,
        help=(
            "optional comma-separated persistent global L2 caps, one "
            "for each Linf cap"
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_SEEDS)),
    )
    parser.add_argument("--maximum-sweeps", type=int, default=200)
    parser.add_argument(
        "--maximum-seconds-per-run", type=float, default=600.0
    )
    parser.add_argument(
        "--patience-apex-updates", type=int, default=120
    )
    parser.add_argument(
        "--minimum-relative-improvement", type=float, default=1e-14
    )
    parser.add_argument(
        "--initialization",
        choices=("random", "natural-perturbed"),
        default="random",
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    payload = run_star_als_campaign(
        output_directory=arguments.output_directory,
        radii=_comma_values(arguments.radii, float),
        global_l2_radii=(
            None
            if arguments.global_l2_caps is None
            else _comma_values(arguments.global_l2_caps, float)
        ),
        seeds=_comma_values(arguments.seeds, int),
        maximum_sweeps=arguments.maximum_sweeps,
        maximum_seconds_per_run=arguments.maximum_seconds_per_run,
        patience_apex_updates=arguments.patience_apex_updates,
        minimum_relative_improvement=(
            arguments.minimum_relative_improvement
        ),
        initialization=arguments.initialization,
        resume=arguments.resume,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
