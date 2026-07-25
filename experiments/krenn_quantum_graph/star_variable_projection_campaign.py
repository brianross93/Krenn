"""Deterministic bounded variable-projection campaign for Krenn ``n=6,d=3``.

The numerical core lives in :mod:`star_variable_projection`.  This module
owns only the optimizer and campaign mechanics:

* a hard Frobenius bound on the eliminated ``15 x 3`` star block;
* projected complex L-BFGS with a monotone Armijo line search;
* root-free retraction to one of the three exact ``P=1`` pivot slices;
* a pivot-slice star-block bound, loose overflow guards, atomic checkpoints,
  deterministic cold starts and warm radius continuation;
* an exact record that residual norm balancing is refused because every
  pivot representative has a certified recession direction; and
* a final replay of all 729 equations by the primary and independent
  enumerators supplied by the core.

No coordinate weight is fixed or assumed nonzero.  Raw ``U`` and ``Y``
norms are gauge-variant and are never presented as invariant bounds.  In
particular, the
12-coordinate determinant-two diagnostic complement is never used as a
search chart.

Every output is numerical reconnaissance.  A numerical zero is not an
exact witness, a bounded miss is not a nonexistence proof, and behavior
across increasing radii is not a proof of strict border membership.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
import operator
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np


CAMPAIGN_SCHEMA = "krenn-n6-d3-star-variable-projection-campaign-v1"
CONFIG_SCHEMA = "krenn-n6-d3-star-variable-projection-config-v1"
CHECKPOINT_SCHEMA = (
    "krenn-n6-d3-star-variable-projection-checkpoint-v1"
)
RESULT_SCHEMA = "krenn-n6-d3-star-variable-projection-result-v1"
PROJECTION_SCHEMA = (
    "krenn-n6-d3-bounded-star-variable-projection-evaluation-v1"
)
DEFAULT_CORE_MODULE = (
    "experiments.krenn_quantum_graph.star_variable_projection"
)
DEFAULT_SEEDS = tuple(range(2026072501, 2026072507))
DEFAULT_RADII = (2.0, 4.0, 8.0, 16.0)
MAXIMUM_WORKERS = 8
DEFAULT_WORKERS = 6
ORBIT_INDICES = (0, 1, 2)
U_VARIABLES = 90
Y_ROWS = 15
TARGET_COLUMNS = 3
_FLOAT_TINY = np.finfo(np.float64).tiny


class KrennVariableProjectionCampaignError(RuntimeError):
    """A campaign input, numerical invariant, or checkpoint failed."""


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a finite real number"
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a finite real number"
        ) from error
    if not math.isfinite(normalized):
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a finite real number"
        )
    return normalized


def _positive_float(value: Any, label: str) -> float:
    normalized = _finite_float(value, label)
    if normalized <= 0:
        raise KrennVariableProjectionCampaignError(
            f"{label} must be positive"
        )
    return normalized


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a positive integer"
        )
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a positive integer"
        ) from error
    if normalized <= 0:
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a positive integer"
        )
    return int(normalized)


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a nonnegative integer"
        )
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a nonnegative integer"
        ) from error
    if normalized < 0:
        raise KrennVariableProjectionCampaignError(
            f"{label} must be a nonnegative integer"
        )
    return int(normalized)


def validate_workers(value: Any) -> int:
    workers = _positive_integer(value, "workers")
    if workers > MAXIMUM_WORKERS:
        raise KrennVariableProjectionCampaignError(
            f"workers cannot exceed {MAXIMUM_WORKERS}"
        )
    return workers


@dataclass(frozen=True)
class RadiusCaps:
    """One pivot-slice star-block radius plus numerical overflow guards.

    ``y_frobenius`` is a control in the normalized ``P=1`` slice, not a
    direct-GHZ-gauge invariant.  The ``u_*`` values are deliberately huge,
    radius-dependent, enforced overflow cutoffs.  They scale as
    ``radius**(-2/3)`` to protect the cubic pullback diagnostics.  They are
    gauge-variant numerical domain restrictions, not mathematical bounds.
    """

    radius: float
    u_linf: float
    u_l2: float
    y_frobenius: float

    @classmethod
    def from_radius(cls, radius: float) -> "RadiusCaps":
        radius = _positive_float(radius, "radius")
        overflow_scale = 1.0e40 / (
            max(1.0, radius) ** (2.0 / 3.0)
        )
        return cls(
            radius=radius,
            # A is quadratic in U and its pullback gradient is cubic in
            # U and quadratic in the bounded Y radius.  This guard keeps
            # squared gradient diagnostics well below float64 overflow.
            # It is still enormous and is explicitly not a mathematical
            # search-region bound.
            u_linf=overflow_scale,
            u_l2=10.0 * overflow_scale,
            y_frobenius=radius,
        )

    def __post_init__(self) -> None:
        radius = _positive_float(self.radius, "radius")
        u_linf = _positive_float(self.u_linf, "u_linf")
        u_l2 = _positive_float(self.u_l2, "u_l2")
        y_frobenius = _positive_float(
            self.y_frobenius, "y_frobenius"
        )
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "u_linf", u_linf)
        object.__setattr__(self, "u_l2", u_l2)
        object.__setattr__(self, "y_frobenius", y_frobenius)


@dataclass(frozen=True)
class TrajectoryConfig:
    """One deterministic orbit/seed/radius trajectory."""

    orbit_index: int
    seed: int
    caps: RadiusCaps
    initialization: str = "cold"
    maximum_iterations: int = 2_000
    maximum_seconds: float = 600.0
    patience: int = 250
    residual_tolerance: float = 1e-12
    gradient_tolerance: float = 1e-12
    minimum_relative_improvement: float = 1e-14
    lbfgs_memory: int = 10
    armijo_constant: float = 1e-4
    backtrack_factor: float = 0.5
    maximum_backtracks: int = 60
    checkpoint_interval: int = 25
    minimum_pivot_abs: float = 1e-12
    rank_rtol: float = 1e-12
    natural_repair_scale: float = 1e-2
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise KrennVariableProjectionCampaignError(
                "the trajectory config schema changed"
            )
        orbit_index = _nonnegative_integer(
            self.orbit_index, "orbit_index"
        )
        if orbit_index not in ORBIT_INDICES:
            raise KrennVariableProjectionCampaignError(
                "orbit_index must be 0, 1, or 2"
            )
        seed = _nonnegative_integer(self.seed, "seed")
        if not isinstance(self.caps, RadiusCaps):
            raise KrennVariableProjectionCampaignError(
                "caps must be RadiusCaps"
            )
        if self.initialization not in (
            "cold",
            "warm",
            "natural-repair",
        ):
            raise KrennVariableProjectionCampaignError(
                "initialization must be cold, warm, or natural-repair"
            )
        for field_name in (
            "maximum_iterations",
            "patience",
            "lbfgs_memory",
            "maximum_backtracks",
            "checkpoint_interval",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name),
            )
        for field_name in (
            "maximum_seconds",
            "residual_tolerance",
            "gradient_tolerance",
            "minimum_relative_improvement",
            "minimum_pivot_abs",
            "rank_rtol",
            "natural_repair_scale",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_float(getattr(self, field_name), field_name),
            )
        armijo = _positive_float(
            self.armijo_constant, "armijo_constant"
        )
        backtrack = _positive_float(
            self.backtrack_factor, "backtrack_factor"
        )
        if armijo >= 1 or backtrack >= 1:
            raise KrennVariableProjectionCampaignError(
                "Armijo and backtrack constants must lie in (0,1)"
            )
        object.__setattr__(self, "armijo_constant", armijo)
        object.__setattr__(self, "backtrack_factor", backtrack)
        object.__setattr__(self, "orbit_index", orbit_index)
        object.__setattr__(self, "seed", seed)

    def to_dict(self) -> dict:
        payload = asdict(self)
        return json.loads(json.dumps(payload, allow_nan=False))

    def fingerprint(self) -> str:
        return _json_sha256(self.to_dict())


@dataclass
class ProjectionEvaluation:
    """One bounded and unbounded variable-projection evaluation."""

    phi: np.ndarray
    y: np.ndarray
    unconstrained_y: np.ndarray | None
    minimum_norm_y_frobenius: float | None
    minimum_norm_y_log10_frobenius: float | None
    minimum_norm_y_array_available: bool
    minimum_norm_y_overflowed_float64: bool
    minimum_norm_y_exact_zero: bool
    residual: np.ndarray
    objective: float
    target_distances_squared: tuple[float, ...]
    raw_objective: float
    raw_target_distances_squared: tuple[float, ...]
    y_cap_active: bool
    lagrange_multiplier: float | None
    lagrange_multiplier_log10: float | None
    lagrange_multiplier_float64_available: bool
    numerical_rank: int
    singular_values: np.ndarray
    condition_number: float
    ambient_gradient: np.ndarray
    horizontal_gradient: np.ndarray
    gradient_real_norm: float
    horizontal_diagnostics: dict
    path_diagnostics: dict

    def summary(self) -> dict:
        return {
            "schema": PROJECTION_SCHEMA,
            "objective": self.objective,
            "residual_l2": math.sqrt(max(self.objective, 0.0)),
            "target_distances_squared": list(
                self.target_distances_squared
            ),
            "raw_span_objective": self.raw_objective,
            "raw_span_residual_l2": math.sqrt(
                max(self.raw_objective, 0.0)
            ),
            "raw_target_distances_squared": list(
                self.raw_target_distances_squared
            ),
            "bounded_y_frobenius": _safe_l2(self.y),
            "minimum_norm_y_frobenius":
                self.minimum_norm_y_frobenius,
            "minimum_norm_y_frobenius_is_finite": (
                self.minimum_norm_y_frobenius is not None
            ),
            "minimum_norm_y_log10_frobenius":
                self.minimum_norm_y_log10_frobenius,
            "minimum_norm_y_array_available":
                self.minimum_norm_y_array_available,
            "minimum_norm_y_overflowed_float64":
                self.minimum_norm_y_overflowed_float64,
            "minimum_norm_y_exact_zero":
                self.minimum_norm_y_exact_zero,
            "bounded_y_maximum_abs": _maximum_abs(self.y),
            "minimum_norm_y_maximum_abs": (
                _maximum_abs(self.unconstrained_y)
                if self.unconstrained_y is not None
                else None
            ),
            "y_cap_active": self.y_cap_active,
            "lagrange_multiplier": self.lagrange_multiplier,
            "lagrange_multiplier_log10":
                self.lagrange_multiplier_log10,
            "lagrange_multiplier_float64_available":
                self.lagrange_multiplier_float64_available,
            "numerical_rank": self.numerical_rank,
            "singular_values": [
                float(value) for value in self.singular_values
            ],
            "condition_number": (
                self.condition_number
                if math.isfinite(self.condition_number)
                else None
            ),
            "condition_number_is_finite": math.isfinite(
                self.condition_number
            ),
            "horizontal_gradient_real_norm":
                self.gradient_real_norm,
            "horizontal_diagnostics":
                _jsonable(self.horizontal_diagnostics),
            "path_dependent_diagnostics":
                _jsonable(self.path_diagnostics),
        }


def _maximum_abs(values: np.ndarray) -> float:
    array = np.asarray(values)
    if not array.size:
        return 0.0
    return float(np.max(np.abs(array)))


def _safe_l2(values: np.ndarray) -> float:
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


def _complex_rows(values: np.ndarray) -> list:
    array = np.asarray(values, dtype=np.complex128)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def _complex_array(
    values: Any,
    *,
    shape: tuple[int, ...] | None = None,
    label: str = "complex array",
) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise KrennVariableProjectionCampaignError(
            f"{label} is not a real/imaginary encoding"
        ) from error
    if raw.ndim < 1 or raw.shape[-1] != 2:
        raise KrennVariableProjectionCampaignError(
            f"{label} is not a real/imaginary encoding"
        )
    result = raw[..., 0] + 1j * raw[..., 1]
    if shape is not None and result.shape != shape:
        raise KrennVariableProjectionCampaignError(
            f"{label} has shape {result.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(result)):
        raise KrennVariableProjectionCampaignError(
            f"{label} contains a nonfinite value"
        )
    return np.asarray(result, dtype=np.complex128)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return _complex_rows(value)
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise KrennVariableProjectionCampaignError(
                "a diagnostic contains a nonfinite float"
            )
        return result
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise KrennVariableProjectionCampaignError(
                "a diagnostic contains a nonfinite complex value"
            )
        return [float(value.real), float(value.imag)]
    if isinstance(value, float) and not math.isfinite(value):
        raise KrennVariableProjectionCampaignError(
            "a diagnostic contains a nonfinite float"
        )
    return value


def _json_sha256(payload: Mapping) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise KrennVariableProjectionCampaignError(
        f"core result is missing one of {names}"
    )


def _load_core(module_name: str = DEFAULT_CORE_MODULE) -> Any:
    if not isinstance(module_name, str) or not module_name:
        raise KrennVariableProjectionCampaignError(
            "core_module must be a nonempty import name"
        )
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise KrennVariableProjectionCampaignError(
            f"could not import numerical core {module_name!r}"
        ) from error


def _representative(core: Any, orbit_index: int) -> Any:
    representatives = tuple(core.pivot_representatives())
    if len(representatives) != 3:
        raise KrennVariableProjectionCampaignError(
            "the core did not expose exactly three pivot representatives"
        )
    try:
        representative = representatives[orbit_index]
    except IndexError as error:
        raise KrennVariableProjectionCampaignError(
            "the orbit representative is unavailable"
        ) from error
    exposed_index = None
    for name in ("index", "orbit_index"):
        if hasattr(representative, name):
            exposed_index = int(getattr(representative, name))
            break
        if isinstance(representative, Mapping) and name in representative:
            exposed_index = int(representative[name])
            break
    if exposed_index is not None and exposed_index != orbit_index:
        raise KrennVariableProjectionCampaignError(
            "pivot representatives are not in canonical orbit order"
        )
    return representative


def _targets(core: Any) -> np.ndarray:
    if hasattr(core, "targets"):
        target = np.asarray(core.targets(), dtype=np.complex128)
    else:
        target = np.zeros((243, 3), dtype=np.complex128)
        target[(0, 121, 242), (0, 1, 2)] = 1.0
    if target.ndim != 2 or target.shape[1] != TARGET_COLUMNS:
        raise KrennVariableProjectionCampaignError(
            "the target matrix must have three columns"
        )
    if not np.all(np.isfinite(target)):
        raise KrennVariableProjectionCampaignError(
            "the target matrix contains nonfinite values"
        )
    return target


def bounded_y_solve(
    phi: np.ndarray,
    target: np.ndarray,
    y_frobenius_cap: float,
    *,
    rank_rtol: float = 1e-12,
    maximum_bisections: int = 96,
) -> dict:
    """Solve ``min ||phi*y-target||`` under a Frobenius cap exactly.

    The active-cap solution is the SVD trust-region solution.  A common
    nonnegative Lagrange multiplier is found by deterministic log-scale
    bisection.  The minimum-norm unconstrained block is optional telemetry:
    its norm is still recorded logarithmically if its entries exceed the
    float64 range.
    """

    phi = np.asarray(phi, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    cap = _positive_float(y_frobenius_cap, "y_frobenius_cap")
    rank_rtol = _positive_float(rank_rtol, "rank_rtol")
    maximum_bisections = _positive_integer(
        maximum_bisections, "maximum_bisections"
    )
    if (
        phi.ndim != 2
        or target.ndim != 2
        or phi.shape[0] != target.shape[0]
        or not np.all(np.isfinite(phi))
        or not np.all(np.isfinite(target))
    ):
        raise KrennVariableProjectionCampaignError(
            "bounded_y_solve received malformed matrices"
        )
    left, singular_values, right_h = np.linalg.svd(
        phi, full_matrices=False
    )
    if not np.all(np.isfinite(singular_values)):
        raise KrennVariableProjectionCampaignError(
            "the bounded-Y SVD returned nonfinite singular values"
        )
    if singular_values.size:
        threshold = rank_rtol * singular_values[0]
        diagnostic_mask = singular_values > threshold
    else:
        diagnostic_mask = np.zeros(0, dtype=bool)
    rank = int(np.count_nonzero(diagnostic_mask))

    # The numerical-rank threshold is diagnostic only.  Dropping a small
    # but nonzero singular direction changes the bounded optimization
    # problem and invalidates its envelope gradient.  Every nonzero SVD
    # direction therefore participates in the trust-region solve.
    solve_mask = singular_values > 0.0
    solve_rank = int(np.count_nonzero(solve_mask))
    right = right_h.conj().T[:, solve_mask]
    left_coordinates = left[:, solve_mask].conj().T @ target
    selected_singulars = singular_values[solve_mask]

    def local_l2(values: np.ndarray) -> float:
        """Scale-safe L2 norm that preserves subnormal complex components."""

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

    def complex_polar(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mask, log magnitude, and phase without subnormal loss."""

        values = np.asarray(values, dtype=np.complex128)
        scale = np.maximum(np.abs(values.real), np.abs(values.imag))
        nonzero = scale > 0.0
        normalized_real = np.zeros_like(scale)
        normalized_imag = np.zeros_like(scale)
        np.divide(
            values.real,
            scale,
            out=normalized_real,
            where=nonzero,
        )
        np.divide(
            values.imag,
            scale,
            out=normalized_imag,
            where=nonzero,
        )
        normalized_magnitude = np.hypot(
            normalized_real, normalized_imag
        )
        log_magnitude = np.zeros_like(scale)
        log_magnitude[nonzero] = (
            np.log(scale[nonzero])
            + np.log(normalized_magnitude[nonzero])
        )
        phase = np.zeros_like(values)
        phase[nonzero] = (
            normalized_real[nonzero]
            / normalized_magnitude[nonzero]
            + 1j
            * normalized_imag[nonzero]
            / normalized_magnitude[nonzero]
        )
        return nonzero, log_magnitude, phase

    (
        nonzero_coordinates,
        coordinate_log_abs,
        coordinate_phases,
    ) = complex_polar(left_coordinates)
    unconstrained_exact_zero = not bool(
        np.any(nonzero_coordinates)
    )
    unconstrained_log_norm: float | None
    if solve_rank and not unconstrained_exact_zero:
        log_coordinate_abs = (
            coordinate_log_abs[nonzero_coordinates]
            - np.broadcast_to(
                np.log(selected_singulars)[:, None],
                left_coordinates.shape,
            )[nonzero_coordinates]
        )
        largest_log_coordinate = float(np.max(log_coordinate_abs))
        unconstrained_log_norm = (
            largest_log_coordinate
            + 0.5
            * math.log(
                float(
                    np.sum(
                        np.exp(
                            2.0
                            * (
                                log_coordinate_abs
                                - largest_log_coordinate
                            )
                        ),
                        dtype=np.float64,
                    )
                )
            )
        )
    elif solve_rank:
        unconstrained_log_norm = None
    else:
        unconstrained_log_norm = None

    maximum_float_log = math.log(np.finfo(np.float64).max)
    unconstrained: np.ndarray | None
    unconstrained_norm: float | None
    unconstrained_overflowed = bool(
        unconstrained_log_norm is not None
        and unconstrained_log_norm > maximum_float_log
    )
    if unconstrained_overflowed:
        unconstrained = None
        unconstrained_norm = None
    elif solve_rank:
        # Divide real and imaginary components separately.  NumPy's
        # complex/real division squares a subnormal denominator internally
        # and can overflow despite a representable quotient.
        denominator = selected_singulars[:, None]
        unconstrained_coordinates = (
            np.divide(left_coordinates.real, denominator)
            + 1j * np.divide(left_coordinates.imag, denominator)
        )
        unconstrained = right @ unconstrained_coordinates
        if not np.all(np.isfinite(unconstrained)):
            unconstrained = None
            unconstrained_norm = None
            unconstrained_overflowed = True
        else:
            candidate_norm = local_l2(unconstrained)
            if math.isfinite(candidate_norm):
                unconstrained_norm = candidate_norm
            else:
                unconstrained = None
                unconstrained_norm = None
                unconstrained_overflowed = True
    else:
        unconstrained = np.zeros(
            (phi.shape[1], target.shape[1]),
            dtype=np.complex128,
        )
        unconstrained_norm = 0.0

    log_singulars = (
        np.log(selected_singulars)
        if solve_rank
        else np.zeros(0, dtype=np.float64)
    )
    def active_spectral_state(
        log_multiplier: float,
        *,
        materialize: bool,
    ) -> tuple[np.ndarray | None, float]:
        """Return spectral coordinates and log norm without forming lambda."""

        row_log_factors = (
            log_singulars
            - np.logaddexp(
                2.0 * log_singulars,
                float(log_multiplier),
            )
        )
        log_amplitudes = (
            coordinate_log_abs[nonzero_coordinates]
            + np.broadcast_to(
                row_log_factors[:, None],
                left_coordinates.shape,
            )[nonzero_coordinates]
        )
        largest = float(np.max(log_amplitudes))
        log_norm = (
            largest
            + 0.5
            * math.log(
                float(
                    np.sum(
                        np.exp(2.0 * (log_amplitudes - largest)),
                        dtype=np.float64,
                    )
                )
            )
        )
        if not materialize:
            return None, log_norm
        coordinates = np.zeros_like(left_coordinates)
        coordinates[nonzero_coordinates] = (
            coordinate_phases[nonzero_coordinates]
            * np.exp(log_amplitudes)
        )
        if not np.all(np.isfinite(coordinates)):
            raise KrennVariableProjectionCampaignError(
                "the bounded-Y spectral coordinates exceeded "
                "numerical range"
            )
        return coordinates, log_norm

    if unconstrained_exact_zero or not solve_rank:
        cap_active = False
    else:
        cap_active = bool(
            unconstrained_log_norm is not None
            and unconstrained_log_norm
            > math.log(cap) + math.log1p(8.0e-15)
        )

    multiplier: float | None = 0.0
    log_multiplier: float | None = None
    multiplier_float64_available = True
    if cap_active:
        active_rows = np.any(nonzero_coordinates, axis=1)
        low_log = (
            2.0 * float(np.min(log_singulars[active_rows])) - 80.0
        )
        log_cap = math.log(cap)
        for _ in range(32):
            _, low_log_norm = active_spectral_state(
                low_log, materialize=False
            )
            if low_log_norm > log_cap:
                break
            low_log -= 80.0
        else:
            raise KrennVariableProjectionCampaignError(
                "the bounded-Y secular solve could not find its "
                "unconstrained bracket"
            )
        log_sigma_times_coordinate = (
            coordinate_log_abs[nonzero_coordinates]
            + np.broadcast_to(
                log_singulars[:, None],
                left_coordinates.shape,
            )[nonzero_coordinates]
        )
        largest_product_log = float(
            np.max(log_sigma_times_coordinate)
        )
        log_product_norm = (
            largest_product_log
            + 0.5
            * math.log(
                float(
                    np.sum(
                        np.exp(
                            2.0
                            * (
                                log_sigma_times_coordinate
                                - largest_product_log
                            )
                        ),
                        dtype=np.float64,
                    )
                )
            )
        )
        high_log = max(
            low_log + 1.0,
            log_product_norm - log_cap + math.log(2.0),
        )
        for _ in range(32):
            _, high_log_norm = active_spectral_state(
                high_log, materialize=False
            )
            if high_log_norm <= log_cap:
                break
            high_log += 80.0
        else:
            raise KrennVariableProjectionCampaignError(
                "the bounded-Y secular solve could not find its "
                "feasible bracket"
            )
        for _ in range(maximum_bisections):
            middle_log = (low_log + high_log) / 2.0
            _, middle_log_norm = active_spectral_state(
                middle_log, materialize=False
            )
            if middle_log_norm > log_cap:
                low_log = middle_log
            else:
                high_log = middle_log
        log_multiplier = high_log
        minimum_float_log = math.log(
            float(np.nextafter(np.float64(0.0), np.float64(1.0)))
        )
        if minimum_float_log <= high_log <= maximum_float_log:
            multiplier = math.exp(high_log)
        else:
            multiplier = None
            multiplier_float64_available = False

    if cap_active:
        if log_multiplier is None:
            raise KrennVariableProjectionCampaignError(
                "the active bounded-Y multiplier is unavailable"
            )
        bounded_coordinates, _bounded_log_norm = active_spectral_state(
            log_multiplier, materialize=True
        )
        if bounded_coordinates is None:
            raise KrennVariableProjectionCampaignError(
                "the active bounded-Y coordinates are unavailable"
        )
        bounded = right @ bounded_coordinates
        spectral_bounded_norm = local_l2(bounded_coordinates)
    elif unconstrained is not None:
        bounded = unconstrained.copy()
        spectral_bounded_norm = (
            0.0 if unconstrained_norm is None else unconstrained_norm
        )
    else:
        raise KrennVariableProjectionCampaignError(
            "an inactive cap requires a representable minimum-norm block"
        )
    bounded_norm = local_l2(bounded)
    if (
        not np.all(np.isfinite(bounded))
        or not math.isfinite(bounded_norm)
        or bounded_norm > cap * (1 + 2e-12) + 2e-14
        or not math.isclose(
            bounded_norm,
            spectral_bounded_norm,
            rel_tol=2e-12,
            abs_tol=2e-14,
        )
    ):
        raise KrennVariableProjectionCampaignError(
            "the bounded-Y solve violated its Frobenius cap"
        )
    if (
        cap_active
        and not math.isclose(
            spectral_bounded_norm,
            cap,
            rel_tol=2e-11,
            abs_tol=2e-13,
        )
    ):
        raise KrennVariableProjectionCampaignError(
            "the bounded-Y secular solve did not saturate its cap"
        )

    residual = phi @ bounded - target
    raw_projection = (
        left[:, solve_mask] @ left_coordinates
        if solve_rank
        else np.zeros_like(target)
    )
    raw_residual = raw_projection - target
    residual_norm = local_l2(residual)
    raw_residual_norm = local_l2(raw_residual)
    objective = residual_norm**2
    raw_objective = raw_residual_norm**2
    if log_multiplier is None:
        multiplier_times_bounded = np.zeros_like(bounded)
    else:
        (
            bounded_nonzero,
            bounded_log_abs,
            bounded_phases,
        ) = complex_polar(bounded)
        multiplier_times_bounded = np.zeros_like(bounded)
        scaled_logs = (
            bounded_log_abs[bounded_nonzero] + log_multiplier
        )
        if (
            scaled_logs.size
            and float(np.max(scaled_logs)) > maximum_float_log
        ):
            raise KrennVariableProjectionCampaignError(
                "the bounded-Y KKT multiplier term exceeded "
                "numerical range"
        )
        multiplier_times_bounded[bounded_nonzero] = (
            bounded_phases[bounded_nonzero]
            * np.exp(scaled_logs)
        )
    kkt_residual = (
        phi.conj().T @ residual + multiplier_times_bounded
    )
    kkt_residual_norm = local_l2(kkt_residual)
    leading_singular = (
        float(singular_values[0]) if singular_values.size else 0.0
    )
    kkt_scale = max(
        1.0,
        leading_singular * residual_norm,
        local_l2(multiplier_times_bounded),
    )
    if (
        not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(raw_residual))
        or not np.all(np.isfinite(kkt_residual))
        or not math.isfinite(objective)
        or not math.isfinite(raw_objective)
        or (
            multiplier is not None
            and not math.isfinite(multiplier)
        )
        or not math.isfinite(kkt_residual_norm)
        or not math.isfinite(kkt_scale)
        or kkt_residual_norm > 2e-10 * kkt_scale + 2e-13
        or objective < -1e-13
        or raw_objective < -1e-13
        or raw_objective > objective * (1 + 2e-11) + 2e-13
    ):
        raise KrennVariableProjectionCampaignError(
            "the bounded and unbounded projection objectives are inconsistent"
        )
    condition = (
        math.inf
        if rank == 0
        or rank < min(phi.shape)
        or singular_values[diagnostic_mask][-1] == 0
        else float(
            singular_values[diagnostic_mask][0]
            / singular_values[diagnostic_mask][-1]
        )
    )
    return {
        "y": bounded,
        "unconstrained_y": unconstrained,
        "minimum_norm_y_frobenius": unconstrained_norm,
        "minimum_norm_y_log10_frobenius": (
            unconstrained_log_norm / math.log(10.0)
            if unconstrained_log_norm is not None
            else None
        ),
        "minimum_norm_y_array_available": unconstrained is not None,
        "minimum_norm_y_overflowed_float64": unconstrained_overflowed,
        "minimum_norm_y_exact_zero": unconstrained_exact_zero,
        "residual": residual,
        "raw_residual": raw_residual,
        "objective": max(objective, 0.0),
        "raw_objective": max(raw_objective, 0.0),
        "target_distances_squared": tuple(
            local_l2(residual[:, column]) ** 2
            for column in range(target.shape[1])
        ),
        "raw_target_distances_squared": tuple(
            local_l2(raw_residual[:, column]) ** 2
            for column in range(target.shape[1])
        ),
        "cap_active": cap_active,
        "lagrange_multiplier": multiplier,
        "lagrange_multiplier_log10": (
            log_multiplier / math.log(10.0)
            if log_multiplier is not None
            else None
        ),
        "lagrange_multiplier_float64_available":
            multiplier_float64_available,
        "cap_saturation_abs_error": (
            abs(spectral_bounded_norm - cap) if cap_active else 0.0
        ),
        "kkt_residual_l2": kkt_residual_norm,
        "kkt_relative_residual": kkt_residual_norm / kkt_scale,
        "rank": rank,
        "singular_values": singular_values,
        "condition_number": condition,
    }


def _horizontal_project(
    core: Any,
    u: np.ndarray,
    vector: np.ndarray,
    representative: Any,
    rank_rtol: float,
) -> tuple[np.ndarray, dict]:
    result = core.horizontal_projection(
        u, vector, representative, rank_rtol=rank_rtol
    )
    projected = np.asarray(
        _field(
            result,
            "projected_gradient",
            "projected",
            "gradient",
        ),
        dtype=np.complex128,
    )
    if projected.shape != u.shape or not np.all(np.isfinite(projected)):
        raise KrennVariableProjectionCampaignError(
            "the horizontal projection returned a malformed vector"
        )
    if isinstance(result, Mapping):
        diagnostics = {
            key: value
            for key, value in result.items()
            if key not in (
                "projected_gradient",
                "projected",
                "gradient",
                "direction",
            )
        }
    elif hasattr(result, "summary"):
        diagnostics = result.summary()
    elif hasattr(result, "to_dict"):
        diagnostics = result.to_dict()
    else:
        diagnostics = {
            name: getattr(result, name)
            for name in (
                "pivot_jacobian_rank",
                "vertical_rank",
                "horizontal_rank",
                "pivot_tangency_error",
                "vertical_orthogonality_error",
            )
            if hasattr(result, name)
        }
    return projected, _jsonable(diagnostics)


def evaluate_projection(
    core: Any,
    u: np.ndarray,
    representative: Any,
    caps: RadiusCaps,
    *,
    rank_rtol: float,
) -> ProjectionEvaluation:
    """Evaluate ``min_{||Y||_F <= r} ||Phi(U)Y-E||_F^2``.

    The unconstrained span distance is retained only as telemetry.  The
    optimized objective and its gradient use the bounded ``Y`` solve.
    """

    u = np.asarray(u, dtype=np.complex128)
    if u.shape != (U_VARIABLES,) or not np.all(np.isfinite(u)):
        raise KrennVariableProjectionCampaignError(
            "the nonstar vector must contain 90 finite complex values"
        )
    phi = np.asarray(
        core.evaluate_star_matrix(u), dtype=np.complex128
    )
    target = _targets(core)
    if phi.shape != (target.shape[0], Y_ROWS):
        raise KrennVariableProjectionCampaignError(
            "the star matrix has the wrong shape"
        )
    solve = bounded_y_solve(
        phi,
        target,
        caps.y_frobenius,
        rank_rtol=rank_rtol,
    )
    ambient = np.asarray(
        core.pullback_gradient(
            u, solve["y"], solve["residual"]
        ),
        dtype=np.complex128,
    )
    if ambient.shape != u.shape or not np.all(np.isfinite(ambient)):
        raise KrennVariableProjectionCampaignError(
            "the core pullback returned a malformed gradient"
        )
    horizontal, horizontal_diagnostics = _horizontal_project(
        core, u, ambient, representative, rank_rtol
    )
    path_diagnostics = (
        _jsonable(
            core.path_diagnostics(u, representative, solve["y"])
        )
        if hasattr(core, "path_diagnostics")
        else {"supported": False}
    )
    return ProjectionEvaluation(
        phi=phi,
        y=solve["y"],
        unconstrained_y=solve["unconstrained_y"],
        minimum_norm_y_frobenius=solve[
            "minimum_norm_y_frobenius"
        ],
        minimum_norm_y_log10_frobenius=solve[
            "minimum_norm_y_log10_frobenius"
        ],
        minimum_norm_y_array_available=solve[
            "minimum_norm_y_array_available"
        ],
        minimum_norm_y_overflowed_float64=solve[
            "minimum_norm_y_overflowed_float64"
        ],
        minimum_norm_y_exact_zero=solve[
            "minimum_norm_y_exact_zero"
        ],
        residual=solve["residual"],
        objective=solve["objective"],
        target_distances_squared=solve[
            "target_distances_squared"
        ],
        raw_objective=solve["raw_objective"],
        raw_target_distances_squared=solve[
            "raw_target_distances_squared"
        ],
        y_cap_active=solve["cap_active"],
        lagrange_multiplier=solve["lagrange_multiplier"],
        lagrange_multiplier_log10=solve[
            "lagrange_multiplier_log10"
        ],
        lagrange_multiplier_float64_available=solve[
            "lagrange_multiplier_float64_available"
        ],
        numerical_rank=solve["rank"],
        singular_values=solve["singular_values"],
        condition_number=solve["condition_number"],
        ambient_gradient=ambient,
        horizontal_gradient=horizontal,
        gradient_real_norm=float(2.0 * _safe_l2(horizontal)),
        horizontal_diagnostics=horizontal_diagnostics,
        path_diagnostics=path_diagnostics,
    )


def _u_accounting(u: np.ndarray, caps: RadiusCaps) -> dict:
    maximum_abs = _maximum_abs(u)
    l2 = _safe_l2(u)
    return {
        "maximum_abs": maximum_abs,
        "l2": l2,
        "loose_overflow_linf_guard": caps.u_linf,
        "loose_overflow_l2_guard": caps.u_l2,
        "linf_overflow_guard_near": (
            maximum_abs >= caps.u_linf * (1 - 1e-10)
        ),
        "l2_overflow_guard_near": (
            l2 >= caps.u_l2 * (1 - 1e-10)
        ),
        "within_loose_overflow_guards": (
            maximum_abs <= caps.u_linf * (1 + 2e-13) + 2e-14
            and l2 <= caps.u_l2 * (1 + 2e-13) + 2e-14
        ),
        "overflow_cutoffs_are_enforced": True,
        "overflow_cutoffs_depend_on_y_radius": True,
        "mathematical_u_norm_cap_imposed": False,
        "raw_norm_is_direct_GHZ_gauge_invariant": False,
    }


def _within_finite_guards(u: np.ndarray, caps: RadiusCaps) -> bool:
    return bool(
        _u_accounting(u, caps)["within_loose_overflow_guards"]
    )


def _retract(
    core: Any,
    u: np.ndarray,
    representative: Any,
    minimum_pivot_abs: float,
) -> tuple[np.ndarray, dict]:
    raw = np.asarray(u, dtype=np.complex128)
    try:
        result = core.retract_pivots(
            raw,
            representative,
            minimum_abs=minimum_pivot_abs,
        )
    except Exception as error:
        raise KrennVariableProjectionCampaignError(
            "the root-free P=1 retraction failed"
        ) from error
    if isinstance(result, tuple) and len(result) == 2:
        normalized, diagnostics = result
    else:
        normalized, diagnostics = result, {}
    normalized = np.asarray(normalized, dtype=np.complex128)
    if normalized.shape != (U_VARIABLES,) or not np.all(
        np.isfinite(normalized)
    ):
        raise KrennVariableProjectionCampaignError(
            "the P=1 retraction returned malformed weights"
        )
    pivot_retraction_size = _safe_l2(normalized - raw)
    recession_audit = (
        _jsonable(core.residual_recession_audit(representative))
        if hasattr(core, "residual_recession_audit")
        else {"supported": False}
    )
    balance_diagnostics = {
        "attempted": False,
        "status": "refused-by-exact-recession-audit",
        "generic_finite_minimizer": False,
        "reason": (
            "a residual cocharacter preserves P=1 while weakly "
            "shrinking a nonstar incident block"
        ),
        "exact_recession_audit": recession_audit,
    }
    residual_balance_size = 0.0
    pivots = np.asarray(
        core.pivot_values(normalized, representative),
        dtype=np.complex128,
    )
    if pivots.shape != (3,) or not np.allclose(
        pivots,
        np.ones(3),
        rtol=3e-12,
        atol=3e-13,
    ):
        raise KrennVariableProjectionCampaignError(
            "the root-free retraction did not reach P=1"
        )
    path_diagnostics = (
        _jsonable(core.path_diagnostics(normalized, representative))
        if hasattr(core, "path_diagnostics")
        else {"supported": False}
    )
    return normalized, {
        **_jsonable(diagnostics),
        "maximum_pivot_unit_error": float(
            np.max(np.abs(pivots - 1))
        ),
        "pivot_retraction_l2_correction": pivot_retraction_size,
        "residual_balance_l2_correction": residual_balance_size,
        "total_raw_to_canonical_l2_correction":
            _safe_l2(normalized - raw),
        "residual_balance": balance_diagnostics,
        "path_dependent_diagnostics": path_diagnostics,
    }


def _core_fingerprints(core: Any) -> dict:
    if hasattr(core, "source_fingerprints"):
        result = _jsonable(core.source_fingerprints())
    elif hasattr(core, "audit"):
        result = {
            "core_audit_sha256": _json_sha256(
                _jsonable(core.audit())
            )
        }
    else:
        result = {
            "core_identity": getattr(
                core, "__name__", type(core).__qualname__
            )
        }
    module_file = getattr(core, "__file__", None)
    if module_file:
        core_path = Path(module_file).resolve()
        result["core_source"] = str(core_path)
        result["core_source_sha256"] = hashlib.sha256(
            core_path.read_bytes()
        ).hexdigest()
    source = Path(__file__).resolve()
    result["campaign_source_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    return result


def _real_inner(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.vdot(left, right).real)


def _lbfgs_direction(
    gradient: np.ndarray,
    history: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if not history:
        return -gradient
    q = gradient.copy()
    alphas: list[float] = []
    rhos: list[float] = []
    for step, change in reversed(history):
        curvature = _real_inner(change, step)
        if curvature <= 0:
            return -gradient
        rho = 1.0 / curvature
        alpha = rho * _real_inner(step, q)
        q = q - alpha * change
        alphas.append(alpha)
        rhos.append(rho)
    last_step, last_change = history[-1]
    denominator = _real_inner(last_change, last_change)
    scale = (
        _real_inner(last_step, last_change) / denominator
        if denominator > 0
        else 1.0
    )
    scale = min(max(scale, 1e-12), 1e12)
    result = scale * q
    for (step, change), alpha, rho in zip(
        history, reversed(alphas), reversed(rhos), strict=True
    ):
        beta = rho * _real_inner(change, result)
        result = result + step * (alpha - beta)
    return -result


def _transport_history(
    core: Any,
    u: np.ndarray,
    representative: Any,
    rank_rtol: float,
    history: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Project every retained secant pair into the current horizontal."""

    transported: list[tuple[np.ndarray, np.ndarray]] = []
    for step, change in history:
        projected_step, _ = _horizontal_project(
            core, u, step, representative, rank_rtol
        )
        projected_change, _ = _horizontal_project(
            core, u, change, representative, rank_rtol
        )
        curvature = _real_inner(projected_step, projected_change)
        scale = _safe_l2(projected_step) * _safe_l2(
            projected_change
        )
        if (
            math.isfinite(curvature)
            and math.isfinite(scale)
            and curvature > 1e-12 * max(scale, _FLOAT_TINY)
        ):
            transported.append((projected_step, projected_change))
    return transported


def _evaluation_signature(evaluation: ProjectionEvaluation) -> tuple:
    diagnostics = evaluation.horizontal_diagnostics
    rank_values = tuple(
        diagnostics.get(name)
        for name in (
            "constraint_rank",
            "full_gauge_tangent_rank",
            "residual_gauge_tangent_rank",
        )
    )
    return (
        evaluation.numerical_rank,
        evaluation.y_cap_active,
        rank_values,
    )


def _find_pole_quantity(value: Any) -> Any:
    """Find a chart-dependent Q diagnostic without assigning invariance."""

    if isinstance(value, Mapping):
        for key in (
            "pole_quantity",
            "pole_quantity_Q",
            "Q",
            "q_value",
        ):
            if key in value:
                return _jsonable(value[key])
        for item in value.values():
            found = _find_pole_quantity(item)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_pole_quantity(item)
            if found is not None:
                return found
    return None


def _cycle_signature(
    u: np.ndarray,
    evaluation: ProjectionEvaluation,
    retraction: Mapping,
) -> str:
    """Return a scale-relative, diagnostic-only state signature."""

    del evaluation, retraction
    scale = max(1.0, _maximum_abs(u))
    rounded = np.stack((u.real / scale, u.imag / scale), axis=-1)
    rounded = np.round(rounded, decimals=11)
    payload = {
        "scale_relative_retracted_horizontal_lift_u":
            rounded.tolist(),
    }
    return _json_sha256(payload)


def _initial_cold(
    core: Any,
    config: TrajectoryConfig,
    representative: Any,
) -> tuple[np.ndarray, dict]:
    if config.initialization == "natural-repair":
        if not hasattr(core, "natural_repair_initial_point"):
            raise KrennVariableProjectionCampaignError(
                "the core does not expose natural repair initialization"
            )
        candidates = (
            np.asarray(
                core.natural_repair_initial_point(
                    representative,
                    config.seed,
                    perturbation_scale=config.natural_repair_scale,
                ),
                dtype=np.complex128,
            )
            for _ in range(1)
        )
    elif hasattr(core, "initial_point"):
        candidates = (
            np.asarray(
                core.initial_point(representative, config.seed),
                dtype=np.complex128,
            )
            for _ in range(1)
        )
    else:
        seed_sequence = np.random.SeedSequence(
            [config.seed, config.orbit_index, 0x4B52454E]
        )
        generator = np.random.default_rng(seed_sequence)
        candidates = (
            (
                generator.normal(size=U_VARIABLES)
                + 1j * generator.normal(size=U_VARIABLES)
            )
            / math.sqrt(2.0)
            for _ in range(512)
        )
    attempts = 0
    for candidate in candidates:
        attempts += 1
        if candidate.shape != (U_VARIABLES,) or not np.all(
            np.isfinite(candidate)
        ):
            continue
        try:
            normalized, diagnostics = _retract(
                core,
                candidate,
                representative,
                config.minimum_pivot_abs,
            )
        except KrennVariableProjectionCampaignError:
            continue
        if _within_finite_guards(normalized, config.caps):
            return normalized, {
                "kind": config.initialization,
                "attempts": attempts,
                "natural_repair_scale": (
                    config.natural_repair_scale
                    if config.initialization == "natural-repair"
                    else None
                ),
                "retraction": diagnostics,
            }
    raise KrennVariableProjectionCampaignError(
        "no deterministic cold initialization fit the P=1 slice and caps"
    )


def _validate_warm(
    core: Any,
    config: TrajectoryConfig,
    representative: Any,
    warm_start: np.ndarray,
) -> tuple[np.ndarray, dict]:
    warm = np.asarray(warm_start, dtype=np.complex128)
    if warm.shape != (U_VARIABLES,) or not np.all(np.isfinite(warm)):
        raise KrennVariableProjectionCampaignError(
            "a warm start must contain 90 finite complex values"
        )
    normalized, diagnostics = _retract(
        core, warm, representative, config.minimum_pivot_abs
    )
    if not _within_finite_guards(normalized, config.caps):
        raise KrennVariableProjectionCampaignError(
            "the warm start violates a loose numerical overflow guard"
        )
    return normalized, {
        "kind": "warm",
        "attempts": 1,
        "retraction": diagnostics,
    }


def _history_to_json(
    history: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    return [
        {
            "step": _complex_rows(step),
            "gradient_change": _complex_rows(change),
        }
        for step, change in history
    ]


def _history_from_json(
    payload: Any,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(payload, list):
        raise KrennVariableProjectionCampaignError(
            "checkpoint L-BFGS history must be a list"
        )
    result = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != {
            "step",
            "gradient_change",
        }:
            raise KrennVariableProjectionCampaignError(
                "checkpoint L-BFGS history is malformed"
            )
        result.append((
            _complex_array(
                row["step"], shape=(U_VARIABLES,), label="history step"
            ),
            _complex_array(
                row["gradient_change"],
                shape=(U_VARIABLES,),
                label="history gradient change",
            ),
        ))
    return result


def _checkpoint_payload(
    *,
    config: TrajectoryConfig,
    fingerprints: Mapping,
    status: str,
    iteration: int,
    accepted_steps: int,
    patience_count: int,
    cumulative_elapsed: float,
    u: np.ndarray,
    best_u: np.ndarray,
    evaluation: ProjectionEvaluation,
    best_evaluation: ProjectionEvaluation,
    history: Sequence[tuple[np.ndarray, np.ndarray]],
    cycle_states: Sequence[np.ndarray],
    trace: Sequence[Mapping],
    initialization: Mapping,
) -> dict:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": config.to_dict(),
        "config_sha256": config.fingerprint(),
        "source_fingerprints": _jsonable(fingerprints),
        "status": status,
        "iteration": int(iteration),
        "accepted_steps": int(accepted_steps),
        "patience_count": int(patience_count),
        "cumulative_elapsed_seconds": float(cumulative_elapsed),
        "u": _complex_rows(u),
        "best_u": _complex_rows(best_u),
        "evaluation": evaluation.summary(),
        "best_evaluation": best_evaluation.summary(),
        "lbfgs_history": _history_to_json(history),
        "cycle_states": [
            _complex_rows(state) for state in cycle_states
        ],
        "trace": _jsonable(list(trace)),
        "initialization": _jsonable(initialization),
    }
    payload["payload_sha256"] = _json_sha256(payload)
    return payload


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    path = Path(path)
    if path.exists() and path.is_symlink():
        raise KrennVariableProjectionCampaignError(
            "refusing to replace a symlink checkpoint"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise KrennVariableProjectionCampaignError(
            "checkpoint parent must be a real directory"
        )
    normalized = _jsonable(payload)
    encoded = json.dumps(
        normalized,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        raise KrennVariableProjectionCampaignError(
            "an atomic checkpoint temporary path already exists"
        )
    try:
        temporary.write_text(encoded, encoding="ascii", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_checkpoint(
    path: Path,
    *,
    config: TrajectoryConfig,
    fingerprints: Mapping,
    core: Any,
    representative: Any,
) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise KrennVariableProjectionCampaignError(
            "resume checkpoint must be a regular file"
        )
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KrennVariableProjectionCampaignError(
            "resume checkpoint is unreadable strict JSON"
        ) from error
    if not isinstance(payload, dict):
        raise KrennVariableProjectionCampaignError(
            "resume checkpoint must be a JSON object"
        )
    supplied_hash = payload.pop("payload_sha256", None)
    if supplied_hash != _json_sha256(payload):
        raise KrennVariableProjectionCampaignError(
            "resume checkpoint payload hash changed"
        )
    payload["payload_sha256"] = supplied_hash
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("config_sha256") != config.fingerprint()
        or not _strict_json_equal(
            payload.get("config"), config.to_dict()
        )
        or not _strict_json_equal(
            payload.get("source_fingerprints"),
            _jsonable(fingerprints),
        )
    ):
        raise KrennVariableProjectionCampaignError(
            "resume checkpoint configuration or source changed"
        )
    for key in (
        "iteration",
        "accepted_steps",
        "patience_count",
    ):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise KrennVariableProjectionCampaignError(
                f"resume checkpoint {key} is invalid"
            )
    elapsed = _finite_float(
        payload.get("cumulative_elapsed_seconds"),
        "cumulative_elapsed_seconds",
    )
    if elapsed < 0:
        raise KrennVariableProjectionCampaignError(
            "checkpoint elapsed time cannot be negative"
        )
    u = _complex_array(
        payload.get("u"), shape=(U_VARIABLES,), label="checkpoint u"
    )
    best_u = _complex_array(
        payload.get("best_u"),
        shape=(U_VARIABLES,),
        label="checkpoint best_u",
    )
    for label, value in (("u", u), ("best_u", best_u)):
        pivots = np.asarray(
            core.pivot_values(value, representative)
        )
        if (
            not _within_finite_guards(value, config.caps)
            or pivots.shape != (3,)
            or not np.allclose(pivots, 1, rtol=3e-12, atol=3e-13)
        ):
            raise KrennVariableProjectionCampaignError(
                f"checkpoint {label} left the P=1 bounded slice"
            )
    evaluation = evaluate_projection(
        core,
        u,
        representative,
        config.caps,
        rank_rtol=config.rank_rtol,
    )
    best_evaluation = evaluate_projection(
        core,
        best_u,
        representative,
        config.caps,
        rank_rtol=config.rank_rtol,
    )
    for label, recomputed in (
        ("evaluation", evaluation),
        ("best_evaluation", best_evaluation),
    ):
        saved = payload.get(label)
        if (
            not isinstance(saved, dict)
            or not math.isclose(
                float(saved.get("objective", math.nan)),
                recomputed.objective,
                rel_tol=3e-12,
                abs_tol=3e-14,
            )
        ):
            raise KrennVariableProjectionCampaignError(
                f"checkpoint {label} failed objective replay"
            )
    history = _history_from_json(payload.get("lbfgs_history"))
    if len(history) > config.lbfgs_memory:
        raise KrennVariableProjectionCampaignError(
            "checkpoint L-BFGS history exceeds configured memory"
        )
    cycle_payload = payload.get("cycle_states")
    if (
        not isinstance(cycle_payload, list)
        or not 1 <= len(cycle_payload) <= 2
    ):
        raise KrennVariableProjectionCampaignError(
            "checkpoint cycle-state history is malformed"
        )
    cycle_states = [
        _complex_array(
            row,
            shape=(U_VARIABLES,),
            label="checkpoint cycle state",
        )
        for row in cycle_payload
    ]
    if not np.allclose(
        cycle_states[-1], u, rtol=2e-13, atol=2e-14
    ):
        raise KrennVariableProjectionCampaignError(
            "checkpoint cycle-state history does not reach current U"
        )
    trace = payload.get("trace")
    if not isinstance(trace, list):
        raise KrennVariableProjectionCampaignError(
            "checkpoint trace must be a list"
        )
    if payload["accepted_steps"] != len(trace):
        raise KrennVariableProjectionCampaignError(
            "checkpoint accepted-step count and trace differ"
        )
    return {
        **payload,
        "u_array": u,
        "best_u_array": best_u,
        "evaluation_object": evaluation,
        "best_evaluation_object": best_evaluation,
        "history_object": history,
        "cycle_states_object": cycle_states,
    }


def _replay_l2(replay: Mapping, name: str) -> float:
    """Read one enumerator's residual norm from current or legacy output."""

    nested = replay.get(name)
    if isinstance(nested, Mapping):
        raw = nested.get("residual_l2")
    else:
        raw = replay.get(f"{name}_l2")
    value = _finite_float(raw, f"{name} replay residual_l2")
    if value < 0:
        raise KrennVariableProjectionCampaignError(
            f"{name} replay residual_l2 must be nonnegative"
        )
    return value


def _validate_full_replay(
    replay: Mapping,
    evaluation: ProjectionEvaluation,
) -> dict:
    """Require both enumerators and the optimized objective to agree."""

    if not isinstance(replay, Mapping):
        raise KrennVariableProjectionCampaignError(
            "the final dual replay must be a mapping"
        )
    if (
        replay.get("all_729_replayed") is not True
        or replay.get("all_729_equations_replayed") is not True
        or replay.get(
            "primary_and_independent_outputs_agree"
        ) is not True
    ):
        raise KrennVariableProjectionCampaignError(
            "the final candidate did not receive an agreeing dual "
            "729-equation replay"
        )
    objective = _finite_float(
        evaluation.objective, "optimized bounded objective"
    )
    if objective < 0:
        raise KrennVariableProjectionCampaignError(
            "optimized bounded objective must be nonnegative"
        )
    expected_l2 = math.sqrt(objective)
    primary_l2 = _replay_l2(replay, "primary")
    independent_l2 = _replay_l2(replay, "independent")
    tolerance = {
        "relative": 5.0e-10,
        "absolute": 5.0e-12,
    }
    if not math.isclose(
        primary_l2,
        independent_l2,
        rel_tol=tolerance["relative"],
        abs_tol=tolerance["absolute"],
    ):
        raise KrennVariableProjectionCampaignError(
            "the two replay residual norms disagree"
        )
    if not math.isclose(
        primary_l2,
        expected_l2,
        rel_tol=tolerance["relative"],
        abs_tol=tolerance["absolute"],
    ):
        raise KrennVariableProjectionCampaignError(
            "the full-system replay residual does not match the "
            "optimized bounded objective"
        )
    return {
        "all_729_flags_required": True,
        "expected_l2_from_optimized_objective": expected_l2,
        "primary_l2": primary_l2,
        "independent_l2": independent_l2,
        "relative_tolerance": tolerance["relative"],
        "absolute_tolerance": tolerance["absolute"],
        "optimizer_and_full_system_agree": True,
    }


def _result_payload(
    *,
    core: Any,
    representative: Any,
    config: TrajectoryConfig,
    status: str,
    iteration: int,
    accepted_steps: int,
    patience_count: int,
    elapsed: float,
    best_u: np.ndarray,
    best_evaluation: ProjectionEvaluation,
    trace: Sequence[Mapping],
    initialization: Mapping,
    fingerprints: Mapping,
) -> dict:
    if hasattr(core, "dual_full_residual"):
        replay = _jsonable(
            core.dual_full_residual(
                best_u, best_evaluation.y
            )
        )
    else:
        replay = {
            "all_729_replayed": False,
            "all_729_equations_replayed": False,
            "reason": "core did not expose dual_full_residual",
        }
    replay["optimizer_full_system_cross_check"] = (
        _validate_full_replay(replay, best_evaluation)
    )
    if hasattr(core, "reconstruct_full_weights"):
        full_weights = np.asarray(
            core.reconstruct_full_weights(
                best_u, best_evaluation.y
            ),
            dtype=np.complex128,
        )
        if full_weights.shape != (135,) or not np.all(
            np.isfinite(full_weights)
        ):
            raise KrennVariableProjectionCampaignError(
                "full-weight reconstruction failed"
            )
        weight_accounting = {
            "maximum_abs": _maximum_abs(full_weights),
            "l2": _safe_l2(full_weights),
            "all_135_weights_finite": True,
        }
        encoded_weights = _complex_rows(full_weights)
    else:
        raise KrennVariableProjectionCampaignError(
            "core did not expose reconstruct_full_weights"
        )
    payload = {
        "schema": RESULT_SCHEMA,
        "config": config.to_dict(),
        "config_sha256": config.fingerprint(),
        "source_fingerprints": _jsonable(fingerprints),
        "termination": status,
        "iterations_attempted": iteration,
        "accepted_steps": accepted_steps,
        "patience_count": patience_count,
        "cumulative_elapsed_seconds": elapsed,
        "initialization": _jsonable(initialization),
        "best_u": _complex_rows(best_u),
        "best_bounded_y": _complex_rows(best_evaluation.y),
        "best_full_weights": encoded_weights,
        "best_evaluation": best_evaluation.summary(),
        "finite_accounting": {
            "u": _u_accounting(best_u, config.caps),
            "pivot_slice_y_frobenius_cap":
                config.caps.y_frobenius,
            "pivot_slice_y_cap_is_full_gauge_invariant": False,
            "raw_u_overflow_cutoffs_are_enforced": True,
            "raw_u_overflow_cutoffs_depend_on_y_radius": True,
            "raw_u_overflow_cutoffs_are_mathematical_bounds": False,
            "full_135_weight_norm_is_bounded_by_search_domain": False,
            "known_divergent_pole_excluded_by_search_domain": False,
            "full_weights": weight_accounting,
        },
        "independent_verification": replay,
        "trace": _jsonable(list(trace)),
        "claim_boundary": {
            "numerical_zero_is_exact_counterexample": False,
            "bounded_miss_is_nonexistence_proof": False,
            "increasing_radius_trend_is_border_proof": False,
            "hard_coordinate_anchors_used": False,
            "raw_u_norm_used_as_mathematical_cap": False,
            "raw_u_overflow_cutoff_enforced": True,
            "pole_quantity_Q_is_full_color_gauge_invariant": False,
            "pole_quantity_Q_used_as_cap": False,
            "bounded_y_is_only_a_P_equals_1_slice_control": True,
            "fixed_global_residual_gauge_slice_used": False,
            "nonzero_objective_comparable_across_other_lifts": False,
            "horizontal_projection_rules_out_accumulated_gauge_drift":
                False,
            "full_135_norm_bounded_search_completed": False,
            "known_divergent_pole_excluded_by_search_domain": False,
            "three_pivot_orbits_remain_an_exhaustive_open_cover": True,
            "numerical_two_cycle_is_exact_proof": False,
            "wall_clock_termination_is_bitwise_reproducible": False,
            "finite_affine_membership_status": "undecided",
            "exact_verification_attempted": False,
        },
    }
    payload["payload_sha256"] = _json_sha256(payload)
    return payload


def run_trajectory(
    config: TrajectoryConfig,
    *,
    checkpoint_path: Path,
    core: Any | None = None,
    core_module: str = DEFAULT_CORE_MODULE,
    warm_start: np.ndarray | None = None,
    resume: bool = False,
) -> dict:
    """Run one deterministic projected complex L-BFGS trajectory."""

    if not isinstance(config, TrajectoryConfig):
        raise KrennVariableProjectionCampaignError(
            "run_trajectory needs TrajectoryConfig"
        )
    core = _load_core(core_module) if core is None else core
    representative = _representative(core, config.orbit_index)
    fingerprints = _core_fingerprints(core)
    checkpoint_path = Path(checkpoint_path)

    if resume:
        loaded = _read_checkpoint(
            checkpoint_path,
            config=config,
            fingerprints=fingerprints,
            core=core,
            representative=representative,
        )
        if loaded["status"] != "running":
            return _result_payload(
                core=core,
                representative=representative,
                config=config,
                status=loaded["status"],
                iteration=loaded["iteration"],
                accepted_steps=loaded["accepted_steps"],
                patience_count=loaded["patience_count"],
                elapsed=loaded["cumulative_elapsed_seconds"],
                best_u=loaded["best_u_array"],
                best_evaluation=loaded["best_evaluation_object"],
                trace=loaded["trace"],
                initialization=loaded["initialization"],
                fingerprints=fingerprints,
            )
        u = loaded["u_array"]
        best_u = loaded["best_u_array"]
        evaluation = loaded["evaluation_object"]
        best_evaluation = loaded["best_evaluation_object"]
        history = loaded["history_object"]
        cycle_states = loaded["cycle_states_object"]
        trace = loaded["trace"]
        iteration = loaded["iteration"]
        accepted_steps = loaded["accepted_steps"]
        patience_count = loaded["patience_count"]
        cumulative_before = loaded["cumulative_elapsed_seconds"]
        initialization = loaded["initialization"]
    else:
        if checkpoint_path.exists():
            raise KrennVariableProjectionCampaignError(
                "checkpoint already exists; use resume"
            )
        if config.initialization == "warm":
            if warm_start is None:
                raise KrennVariableProjectionCampaignError(
                    "warm initialization requires warm_start"
                )
            u, initialization = _validate_warm(
                core, config, representative, warm_start
            )
        else:
            if warm_start is not None:
                raise KrennVariableProjectionCampaignError(
                    "non-warm initialization cannot receive warm_start"
                )
            u, initialization = _initial_cold(
                core, config, representative
            )
        evaluation = evaluate_projection(
            core,
            u,
            representative,
            config.caps,
            rank_rtol=config.rank_rtol,
        )
        best_u = u.copy()
        best_evaluation = evaluation
        history: list[tuple[np.ndarray, np.ndarray]] = []
        cycle_states = [u.copy()]
        trace: list[dict] = []
        iteration = 0
        accepted_steps = 0
        patience_count = 0
        cumulative_before = 0.0

    status = "running"
    started = time.monotonic()
    while status == "running":
        elapsed = cumulative_before + time.monotonic() - started
        if iteration >= config.maximum_iterations:
            status = "maximum-iterations"
            break
        if elapsed >= config.maximum_seconds:
            status = "time-budget"
            break
        if math.sqrt(evaluation.objective) <= config.residual_tolerance:
            status = "numerical-tolerance"
            break
        if evaluation.gradient_real_norm <= config.gradient_tolerance:
            status = "gradient-tolerance"
            break
        if patience_count >= config.patience:
            status = "patience-exhausted"
            break

        iteration += 1
        standard_gradient = 2.0 * evaluation.horizontal_gradient
        history = _transport_history(
            core,
            u,
            representative,
            config.rank_rtol,
            history,
        )
        direction = _lbfgs_direction(standard_gradient, history)
        direction, direction_diagnostics = _horizontal_project(
            core,
            u,
            direction,
            representative,
            config.rank_rtol,
        )
        slope = _real_inner(standard_gradient, direction)
        if slope >= -1e-16 * max(
            1.0,
            _safe_l2(standard_gradient)
            * _safe_l2(direction),
        ):
            history.clear()
            direction = -standard_gradient
            direction, direction_diagnostics = _horizontal_project(
                core,
                u,
                direction,
                representative,
                config.rank_rtol,
            )
            slope = _real_inner(standard_gradient, direction)
        if slope >= 0 or not math.isfinite(slope):
            status = "no-horizontal-descent-direction"
            break

        before = evaluation
        accepted = False
        step_length = 1.0
        backtracks = 0
        retraction_diagnostics: dict = {}
        candidate_evaluation = before
        candidate_u = u
        rejection_reason = None
        rejection_counts = {
            "pivot-retraction": 0,
            "finite-overflow-guard": 0,
            "bounded-variable-projection": 0,
            "armijo": 0,
            "invariant-cap": 0,
            "time-budget": 0,
        }
        while backtracks <= config.maximum_backtracks:
            if (
                cumulative_before + time.monotonic() - started
                >= config.maximum_seconds
            ):
                rejection_reason = "time-budget"
                rejection_counts[rejection_reason] += 1
                status = "time-budget-during-line-search"
                break
            raw = u + step_length * direction
            try:
                proposal, retraction_diagnostics = _retract(
                    core,
                    raw,
                    representative,
                    config.minimum_pivot_abs,
                )
            except KrennVariableProjectionCampaignError:
                rejection_reason = "pivot-retraction"
                rejection_counts[rejection_reason] += 1
            else:
                if not _within_finite_guards(proposal, config.caps):
                    rejection_reason = "finite-overflow-guard"
                    rejection_counts[rejection_reason] += 1
                else:
                    try:
                        proposal_evaluation = evaluate_projection(
                            core,
                            proposal,
                            representative,
                            config.caps,
                            rank_rtol=config.rank_rtol,
                        )
                    except (
                        KrennVariableProjectionCampaignError,
                        np.linalg.LinAlgError,
                    ):
                        rejection_reason = (
                            "bounded-variable-projection"
                        )
                        rejection_counts[rejection_reason] += 1
                        step_length *= config.backtrack_factor
                        backtracks += 1
                        continue
                    armijo_bound = (
                        before.objective
                        + config.armijo_constant
                        * step_length
                        * slope
                    )
                    if proposal_evaluation.objective <= armijo_bound:
                        accepted = True
                        candidate_u = proposal
                        candidate_evaluation = proposal_evaluation
                        break
                    rejection_reason = "armijo"
                    rejection_counts[rejection_reason] += 1
            step_length *= config.backtrack_factor
            backtracks += 1
        if not accepted:
            history.clear()
            patience_count += 1
            if backtracks > config.maximum_backtracks:
                status = "line-search-exhausted"
            elapsed = cumulative_before + time.monotonic() - started
            checkpoint = _checkpoint_payload(
                config=config,
                fingerprints=fingerprints,
                status=status,
                iteration=iteration,
                accepted_steps=accepted_steps,
                patience_count=patience_count,
                cumulative_elapsed=elapsed,
                u=u,
                best_u=best_u,
                evaluation=evaluation,
                best_evaluation=best_evaluation,
                history=history,
                cycle_states=cycle_states,
                trace=trace,
                initialization=initialization,
            )
            _write_json_atomic(checkpoint_path, checkpoint)
            continue

        previous_u = u
        previous_gradient = standard_gradient
        previous_signature = _evaluation_signature(before)
        u = candidate_u
        evaluation = candidate_evaluation
        accepted_steps += 1
        improvement = before.objective - evaluation.objective
        relative_improvement = improvement / max(
            before.objective, _FLOAT_TINY
        )
        if relative_improvement > config.minimum_relative_improvement:
            patience_count = 0
        else:
            patience_count += 1
        if evaluation.objective < best_evaluation.objective:
            best_u = u.copy()
            best_evaluation = evaluation

        new_gradient = 2.0 * evaluation.horizontal_gradient
        if _evaluation_signature(evaluation) != previous_signature:
            history.clear()
        else:
            step = u - previous_u
            step, _ = _horizontal_project(
                core,
                u,
                step,
                representative,
                config.rank_rtol,
            )
            transported_old, _ = _horizontal_project(
                core,
                u,
                previous_gradient,
                representative,
                config.rank_rtol,
            )
            change = new_gradient - transported_old
            curvature = _real_inner(step, change)
            scale = _safe_l2(step) * _safe_l2(change)
            if curvature > 1e-12 * max(scale, _FLOAT_TINY):
                history.append((step, change))
                history[:] = history[-config.lbfgs_memory :]
            else:
                history.clear()

        elapsed = cumulative_before + time.monotonic() - started
        trace_row = {
            "accepted_step": accepted_steps,
            "iteration": iteration,
            "step_length": step_length,
            "backtracks": backtracks,
            "rejection_before_acceptance": rejection_reason,
            "backtrack_rejection_counts": rejection_counts,
            "invariant_cap_enabled": False,
            "pole_quantity_Q_used_as_cap": False,
            "objective_before": before.objective,
            "objective_after": evaluation.objective,
            "relative_improvement": relative_improvement,
            "raw_span_objective_after": evaluation.raw_objective,
            "minimum_norm_y_frobenius_after":
                evaluation.minimum_norm_y_frobenius,
            "minimum_norm_y_log10_frobenius_after":
                evaluation.minimum_norm_y_log10_frobenius,
            "minimum_norm_y_array_available_after":
                evaluation.minimum_norm_y_array_available,
            "minimum_norm_y_overflowed_float64_after":
                evaluation.minimum_norm_y_overflowed_float64,
            "bounded_y_frobenius_after": _safe_l2(evaluation.y),
            "y_cap_active_after": evaluation.y_cap_active,
            "u_after": _u_accounting(u, config.caps),
            "horizontal_gradient_real_norm_after":
                evaluation.gradient_real_norm,
            "retraction": retraction_diagnostics,
            "path_dependent_diagnostics_after":
                evaluation.path_diagnostics,
            "direction_projection": direction_diagnostics,
            "cumulative_elapsed_seconds": elapsed,
        }
        state_signature = _cycle_signature(
            u, evaluation, retraction_diagnostics
        )
        if len(cycle_states) >= 2:
            if hasattr(core, "detect_two_cycle"):
                cycle_diagnostics = _jsonable(
                    core.detect_two_cycle(
                        u,
                        cycle_states[-1],
                        cycle_states[-2],
                    )
                )
            else:
                one_step_l2 = _safe_l2(u - cycle_states[-1])
                two_step_l2 = _safe_l2(u - cycle_states[-2])
                cycle_scale = max(1.0, _safe_l2(u))
                cycle_relative_tolerance = 1.0e-10
                cycle_diagnostics = {
                    "one_step_l2": one_step_l2,
                    "two_step_l2": two_step_l2,
                    "relative_tolerance":
                        cycle_relative_tolerance,
                    "two_cycle_detected": bool(
                        two_step_l2
                        <= cycle_relative_tolerance * cycle_scale
                        and one_step_l2
                        > 10.0
                        * cycle_relative_tolerance
                        * cycle_scale
                    ),
                    "cycle_is_exact_proof": False,
                    "numerical_nonproof": True,
                    "implementation": "campaign fallback",
                }
        else:
            cycle_diagnostics = {
                "available": False,
                "reason": "fewer than two prior accepted states",
                "two_cycle_detected": False,
                "cycle_is_exact_proof": False,
                "numerical_nonproof": True,
            }
        two_cycle = (
            cycle_diagnostics.get("two_cycle_detected") is True
        )
        previous_cap_states = [
            bool(row.get("y_cap_active_after"))
            for row in trace[-2:]
        ]
        cap_pattern = [
            *previous_cap_states,
            bool(evaluation.y_cap_active),
        ]
        alternating_cap = (
            len(cap_pattern) == 3
            and cap_pattern[0] == cap_pattern[2]
            and cap_pattern[0] != cap_pattern[1]
        )
        correction = float(
            retraction_diagnostics.get(
                "total_raw_to_canonical_l2_correction", 0.0
            )
        )
        trace_row[
            "canonicalization_correction_over_step_squared"
        ] = correction / max(step_length**2, _FLOAT_TINY)
        trace_row[
            "canonicalization_expected_second_order_for_horizontal_step"
        ] = True
        trace_row["state_signature"] = state_signature
        trace_row["numerical_two_cycle_diagnostic"] = (
            cycle_diagnostics
        )
        trace_row["numerical_two_cycle_detected"] = two_cycle
        trace_row["alternating_y_cap_pattern_detected"] = (
            alternating_cap
        )
        trace_row[
            "alternating_cap_with_nontrivial_retraction_detected"
        ] = alternating_cap and correction > 1e-10
        trace.append(trace_row)
        cycle_states.append(u.copy())
        cycle_states[:] = cycle_states[-2:]
        if two_cycle:
            status = "numerical-two-cycle"
        if (
            status != "running"
            or accepted_steps % config.checkpoint_interval == 0
        ):
            checkpoint = _checkpoint_payload(
                config=config,
                fingerprints=fingerprints,
                status=status,
                iteration=iteration,
                accepted_steps=accepted_steps,
                patience_count=patience_count,
                cumulative_elapsed=elapsed,
                u=u,
                best_u=best_u,
                evaluation=evaluation,
                best_evaluation=best_evaluation,
                history=history,
                cycle_states=cycle_states,
                trace=trace,
                initialization=initialization,
            )
            _write_json_atomic(checkpoint_path, checkpoint)

    elapsed = cumulative_before + time.monotonic() - started
    final_checkpoint = _checkpoint_payload(
        config=config,
        fingerprints=fingerprints,
        status=status,
        iteration=iteration,
        accepted_steps=accepted_steps,
        patience_count=patience_count,
        cumulative_elapsed=elapsed,
        u=u,
        best_u=best_u,
        evaluation=evaluation,
        best_evaluation=best_evaluation,
        history=history,
        cycle_states=cycle_states,
        trace=trace,
        initialization=initialization,
    )
    _write_json_atomic(checkpoint_path, final_checkpoint)
    return _result_payload(
        core=core,
        representative=representative,
        config=config,
        status=status,
        iteration=iteration,
        accepted_steps=accepted_steps,
        patience_count=patience_count,
        elapsed=elapsed,
        best_u=best_u,
        best_evaluation=best_evaluation,
        trace=trace,
        initialization=initialization,
        fingerprints=fingerprints,
    )


def _radius_text(radius: float) -> str:
    return format(float(radius), ".12g").replace(".", "p")


def _chain_worker(payload: Mapping) -> list[dict]:
    core_module = str(payload["core_module"])
    scratch = Path(payload["scratch"])
    orbit_index = int(payload["orbit_index"])
    seed = int(payload["seed"])
    radii = tuple(map(float, payload["radii"]))
    shared = dict(payload["shared_config"])
    chain_initialization = str(
        payload.get("chain_initialization", "cold")
    )
    if chain_initialization not in ("cold", "natural-repair"):
        raise KrennVariableProjectionCampaignError(
            "a chain initialization is invalid"
        )
    if chain_initialization == "natural-repair" and orbit_index != 2:
        raise KrennVariableProjectionCampaignError(
            "natural repair chains require the all-distinct orbit"
        )
    resume = bool(payload["resume"])
    results = []
    warm_u = None
    for radius_position, radius in enumerate(radii):
        caps = RadiusCaps.from_radius(radius)
        primary_config = TrajectoryConfig(
            orbit_index=orbit_index,
            seed=seed,
            caps=caps,
            initialization=chain_initialization,
            **shared,
        )
        stem = (
            f"orbit_{orbit_index}_seed_{seed}_radius_"
            f"{_radius_text(radius)}_{chain_initialization}"
        )
        primary_checkpoint = scratch / f"{stem}.checkpoint.json"
        primary_result = run_trajectory(
            primary_config,
            checkpoint_path=primary_checkpoint,
            core_module=core_module,
            resume=resume and primary_checkpoint.is_file(),
        )
        primary_path = scratch / f"{stem}.result.json"
        _write_json_atomic(primary_path, primary_result)
        primary_best = _complex_array(
            primary_result["best_u"],
            shape=(U_VARIABLES,),
            label="primary result best_u",
        )
        results.append({
            "orbit_index": orbit_index,
            "seed": seed,
            "radius": radius,
            "initialization": chain_initialization,
            "result_path": primary_path.name,
            "termination": primary_result["termination"],
            "smallest_recorded_lift_dependent_objective": primary_result[
                "best_evaluation"
            ]["objective"],
            "smallest_recorded_lift_dependent_raw_span_objective":
                primary_result["best_evaluation"]["raw_span_objective"],
            "objective_comparable_across_seeds_or_orbits": False,
            # Compatibility aliases for pre-audit result readers.
            "best_objective": primary_result[
                "best_evaluation"
            ]["objective"],
            "best_raw_span_objective": primary_result[
                "best_evaluation"
            ]["raw_span_objective"],
            "minimum_norm_y_frobenius": primary_result[
                "best_evaluation"
            ]["minimum_norm_y_frobenius"],
            "bounded_y_frobenius": primary_result[
                "best_evaluation"
            ]["bounded_y_frobenius"],
            "y_cap_active": primary_result[
                "best_evaluation"
            ]["y_cap_active"],
        })
        if radius_position == 0:
            warm_u = primary_best
            continue
        warm_config = TrajectoryConfig(
            orbit_index=orbit_index,
            seed=seed,
            caps=caps,
            initialization="warm",
            **shared,
        )
        stem = (
            f"orbit_{orbit_index}_seed_{seed}_radius_"
            f"{_radius_text(radius)}_"
            f"{'warm' if chain_initialization == 'cold' else 'natural-repair-warm'}"
        )
        warm_checkpoint = scratch / f"{stem}.checkpoint.json"
        warm_result = run_trajectory(
            warm_config,
            checkpoint_path=warm_checkpoint,
            core_module=core_module,
            warm_start=warm_u,
            resume=resume and warm_checkpoint.is_file(),
        )
        warm_path = scratch / f"{stem}.result.json"
        _write_json_atomic(warm_path, warm_result)
        warm_u = _complex_array(
            warm_result["best_u"],
            shape=(U_VARIABLES,),
            label="warm result best_u",
        )
        results.append({
            "orbit_index": orbit_index,
            "seed": seed,
            "radius": radius,
            "initialization": (
                "warm"
                if chain_initialization == "cold"
                else "natural-repair-warm"
            ),
            "result_path": warm_path.name,
            "termination": warm_result["termination"],
            "smallest_recorded_lift_dependent_objective": warm_result[
                "best_evaluation"
            ]["objective"],
            "smallest_recorded_lift_dependent_raw_span_objective":
                warm_result["best_evaluation"]["raw_span_objective"],
            "objective_comparable_across_seeds_or_orbits": False,
            # Compatibility aliases for pre-audit result readers.
            "best_objective": warm_result[
                "best_evaluation"
            ]["objective"],
            "best_raw_span_objective": warm_result[
                "best_evaluation"
            ]["raw_span_objective"],
            "minimum_norm_y_frobenius": warm_result[
                "best_evaluation"
            ]["minimum_norm_y_frobenius"],
            "bounded_y_frobenius": warm_result[
                "best_evaluation"
            ]["bounded_y_frobenius"],
            "y_cap_active": warm_result[
                "best_evaluation"
            ]["y_cap_active"],
        })
    return results


def _validated_scratch_root(path: Path) -> Path:
    path = Path(path).resolve()
    if "onedrive" in tuple(piece.lower() for piece in path.parts):
        raise KrennVariableProjectionCampaignError(
            "campaign scratch cannot be inside OneDrive"
        )
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise KrennVariableProjectionCampaignError(
            "campaign scratch root must be a real directory"
        )
    return path


def run_campaign(
    *,
    scratch_root: Path,
    core_module: str = DEFAULT_CORE_MODULE,
    radii: Sequence[float] = DEFAULT_RADII,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    initializations: Sequence[str] = ("cold",),
    workers: int = DEFAULT_WORKERS,
    maximum_iterations: int = 2_000,
    maximum_seconds_per_trajectory: float = 600.0,
    patience: int = 250,
    checkpoint_interval: int = 25,
    natural_repair_scale: float = 1.0e-2,
    resume: bool = False,
) -> dict:
    """Run cold starts plus warm continuation on all three pivot orbits."""

    scratch = _validated_scratch_root(scratch_root)
    workers = validate_workers(workers)
    normalized_radii = tuple(
        _positive_float(radius, "radius") for radius in radii
    )
    normalized_seeds = tuple(
        _nonnegative_integer(seed, "seed") for seed in seeds
    )
    normalized_initializations = tuple(
        str(value) for value in initializations
    )
    if (
        not normalized_radii
        or tuple(sorted(normalized_radii)) != normalized_radii
        or len(set(normalized_radii)) != len(normalized_radii)
        or not normalized_seeds
        or len(set(normalized_seeds)) != len(normalized_seeds)
        or any(seed < 0 for seed in normalized_seeds)
        or not normalized_initializations
        or len(set(normalized_initializations))
        != len(normalized_initializations)
        or any(
            value not in ("cold", "natural-repair")
            for value in normalized_initializations
        )
    ):
        raise KrennVariableProjectionCampaignError(
            "radii must be unique increasing positives, seeds unique "
            "nonnegative integers, and initializations unique members "
            "of {cold,natural-repair}"
        )
    shared = {
        "maximum_iterations": _positive_integer(
            maximum_iterations, "maximum_iterations"
        ),
        "maximum_seconds": _positive_float(
            maximum_seconds_per_trajectory,
            "maximum_seconds_per_trajectory",
        ),
        "patience": _positive_integer(patience, "patience"),
        "checkpoint_interval": _positive_integer(
            checkpoint_interval, "checkpoint_interval"
        ),
        "natural_repair_scale": _positive_float(
            natural_repair_scale, "natural_repair_scale"
        ),
    }
    jobs = [
        {
            "core_module": core_module,
            "scratch": str(scratch),
            "orbit_index": orbit,
            "seed": seed,
            "radii": list(normalized_radii),
            "shared_config": shared,
            "chain_initialization": initialization,
            "resume": bool(resume),
        }
        for initialization in normalized_initializations
        for orbit in ORBIT_INDICES
        if initialization == "cold" or orbit == 2
        for seed in normalized_seeds
    ]
    rows: list[dict] = []
    worker_errors: list[dict] = []
    if workers == 1:
        for job in jobs:
            try:
                rows.extend(_chain_worker(job))
            except Exception as error:
                worker_errors.append({
                    "orbit_index": job["orbit_index"],
                    "seed": job["seed"],
                    "chain_initialization":
                        job["chain_initialization"],
                    "error_type": type(error).__name__,
                    "message": str(error),
                })
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_chain_worker, job): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as error:
                    worker_errors.append({
                        "orbit_index": job["orbit_index"],
                        "seed": job["seed"],
                        "chain_initialization":
                            job["chain_initialization"],
                        "error_type": type(error).__name__,
                        "message": str(error),
                    })
    rows.sort(
        key=lambda row: (
            row["orbit_index"],
            row["seed"],
            row["radius"],
            row["initialization"],
        )
    )
    if worker_errors:
        failure_manifest = {
            "schema": CAMPAIGN_SCHEMA,
            "status": "incomplete-worker-failure",
            "scratch_root": str(scratch),
            "core_module": core_module,
            "radii": list(normalized_radii),
            "seeds": list(normalized_seeds),
            "initializations": list(normalized_initializations),
            "workers": workers,
            "completed_trajectory_rows": rows,
            "worker_errors": sorted(
                worker_errors,
                key=lambda row: (
                    row["orbit_index"],
                    row["seed"],
                    row["chain_initialization"],
                    row["error_type"],
                ),
            ),
            "claim_boundary": {
                "campaign_completed": False,
                "partial_rows_support_mathematical_claim": False,
                "finite_affine_membership_status": "undecided",
            },
        }
        _write_json_atomic(
            scratch / "failure_manifest.json",
            failure_manifest,
        )
        raise KrennVariableProjectionCampaignError(
            f"{len(worker_errors)} campaign worker chain(s) failed; "
            "see failure_manifest.json"
        )
    smallest_recorded = min(
        rows,
        key=lambda row: row[
            "smallest_recorded_lift_dependent_objective"
        ],
    )
    manifest = {
        "schema": CAMPAIGN_SCHEMA,
        "scratch_root": str(scratch),
        "core_module": core_module,
        "orbit_cover": {
            "fixed_apex": 0,
            "representative_count": 3,
            "orbit_indices": list(ORBIT_INDICES),
            "orbit_sizes": [5, 60, 60],
            "provenance": (
                "S5 residual vertices x S3 colors target-minor pivots"
            ),
            "legacy_eight_seed_charts_used": False,
            "hard_coordinate_anchors_used": False,
        },
        "gauge_control": {
            "selected_pivots_root_free_normalized_to_one": True,
            "residual_real_torus_balanced_after_each_retraction": False,
            "residual_norm_balancing_refused_by_exact_recession": True,
            "horizontal_residual_gauge_projection": True,
            "fixed_global_residual_gauge_slice": False,
            "cross_run_nonzero_objective_comparison_gauge_invariant":
                False,
            "horizontal_projection_prevents_accumulated_gauge_drift":
                False,
            "raw_u_norm_mathematical_cap": False,
            "raw_u_overflow_cutoff_enforced": True,
            "raw_u_overflow_cutoff_depends_on_y_radius": True,
            "raw_y_norm_full_gauge_invariant": False,
            "radius_controls": (
                "bounded star-block Frobenius norm plus enforced huge "
                "radius-dependent raw-U overflow cutoffs in each "
                "horizontal P=1 lift"
            ),
            "known_pole_quantity_Q_full_color_gauge_invariant": False,
            "known_pole_quantity_Q_used_as_cap": False,
            "full_135_weight_norm_is_bounded_by_search_domain": False,
            "known_divergent_pole_excluded_by_search_domain": False,
        },
        "radii": list(normalized_radii),
        "seeds": list(normalized_seeds),
        "initializations": list(normalized_initializations),
        "natural_repair_scale": shared["natural_repair_scale"],
        "workers": workers,
        "maximum_workers": MAXIMUM_WORKERS,
        "maximum_iterations_per_trajectory": shared[
            "maximum_iterations"
        ],
        "maximum_seconds_per_trajectory": shared["maximum_seconds"],
        "patience": shared["patience"],
        "checkpoint_interval": shared["checkpoint_interval"],
        "cold_start_at_every_radius":
            "cold" in normalized_initializations,
        "natural_repair_start_at_every_radius":
            "natural-repair" in normalized_initializations,
        "warm_continuation_after_first_radius": True,
        "deterministic_initialization_and_iteration_order": True,
        "wall_clock_termination_is_bitwise_reproducible": False,
        "trajectory_count": len(rows),
        "expected_trajectory_count": (
            len(jobs)
            * (2 * len(normalized_radii) - 1)
        ),
        "results": rows,
        "smallest_recorded_lift_dependent_candidate":
            smallest_recorded,
        "cross_run_nonzero_objectives_are_intrinsically_comparable":
            False,
        # Deprecated compatibility alias.  It is not a gauge-invariant
        # ordering of candidate quality.
        "best_p_equals_one_y_bounded_candidate": smallest_recorded,
        "claim_boundary": {
            "numerical_candidate_is_exact_witness": False,
            "bounded_campaign_miss_is_nonexistence_proof": False,
            "radius_trend_is_border_proof": False,
            "horizontal_slice_y_bound_is_global_gauge_invariant": False,
            "fixed_global_residual_gauge_slice_used": False,
            "cross_run_nonzero_objective_comparison_is_gauge_invariant":
                False,
            "horizontal_projection_rules_out_accumulated_gauge_drift":
                False,
            "raw_u_overflow_cutoffs_enforced": True,
            "full_135_norm_bounded_search_completed": False,
            "known_divergent_pole_excluded_by_search_domain": False,
            "numerical_two_cycle_is_exact_proof": False,
            "wall_clock_time_budget_is_reproducible_iteration_budget":
                False,
            "finite_affine_membership_status": "undecided",
        },
    }
    _write_json_atomic(scratch / "manifest.json", manifest)
    return manifest


def _comma_values(text: str, converter) -> tuple:
    try:
        values = tuple(converter(piece) for piece in text.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "comma-separated values are invalid"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError(
            "comma-separated values cannot be empty"
        )
    return values


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic bounded variable projection on the three "
            "exact n=6,d=3 star-pivot orbit representatives."
        )
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        required=True,
        help=(
            "explicit non-OneDrive scratch directory; e.g. "
            r"D:\KrennScratch\counterexample_search\variable_projection"
        ),
    )
    parser.add_argument("--core-module", default=DEFAULT_CORE_MODULE)
    parser.add_argument(
        "--radii", default=",".join(map(str, DEFAULT_RADII))
    )
    parser.add_argument(
        "--seeds", default=",".join(map(str, DEFAULT_SEEDS))
    )
    parser.add_argument(
        "--initializations",
        default="cold",
        help="comma-separated subset of cold,natural-repair",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS
    )
    parser.add_argument(
        "--maximum-iterations", type=int, default=2_000
    )
    parser.add_argument(
        "--maximum-seconds-per-trajectory",
        type=float,
        default=600.0,
    )
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=25
    )
    parser.add_argument(
        "--natural-repair-scale", type=float, default=1.0e-2
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    payload = run_campaign(
        scratch_root=arguments.scratch_root,
        core_module=arguments.core_module,
        radii=_comma_values(arguments.radii, float),
        seeds=_comma_values(arguments.seeds, int),
        initializations=_comma_values(
            arguments.initializations, str
        ),
        workers=arguments.workers,
        maximum_iterations=arguments.maximum_iterations,
        maximum_seconds_per_trajectory=(
            arguments.maximum_seconds_per_trajectory
        ),
        patience=arguments.patience,
        checkpoint_interval=arguments.checkpoint_interval,
        natural_repair_scale=arguments.natural_repair_scale,
        resume=arguments.resume,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "DEFAULT_CORE_MODULE",
    "DEFAULT_RADII",
    "DEFAULT_SEEDS",
    "DEFAULT_WORKERS",
    "KrennVariableProjectionCampaignError",
    "MAXIMUM_WORKERS",
    "ProjectionEvaluation",
    "RESULT_SCHEMA",
    "RadiusCaps",
    "TrajectoryConfig",
    "bounded_y_solve",
    "evaluate_projection",
    "run_campaign",
    "run_trajectory",
    "validate_workers",
]
