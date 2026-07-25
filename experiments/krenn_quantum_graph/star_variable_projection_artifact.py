"""Portable artifact for the star variable-projection campaign.

The scratch campaign may contain large terminal checkpoints and traces.
This module waits for its terminal ``manifest.json``, replays every result
against its matching checkpoint, freshly recomputes each best projection,
reconstructs all 135 weights, and invokes the primary/independent
729-equation verifier.  The committed two-file bundle retains compact rows
for every trajectory and embeds only deduplicated winners sufficient for a
fresh scratch-independent replay.

This is ``P=1`` pivot-slice reconnaissance with a Frobenius bound on the
recovered star block ``Y``.  It is not a globally gauge-invariant bound and
is not a bounded search in the norm of all 135 weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
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

from experiments.krenn_quantum_graph.star_variable_projection_campaign import (
    CAMPAIGN_SCHEMA,
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    DEFAULT_CORE_MODULE,
    KrennVariableProjectionCampaignError,
    PROJECTION_SCHEMA,
    RESULT_SCHEMA,
    RadiusCaps,
    TrajectoryConfig,
    evaluate_projection,
)


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "krenn-n6-d3-star-variable-projection-summary-v1"
MANIFEST_SCHEMA = (
    "krenn-n6-d3-star-variable-projection-summary-manifest-v1"
)
SUMMARY_FILE = "summary.json"
MANIFEST_FILE = "manifest.json"
ALL_FILES = (SUMMARY_FILE, MANIFEST_FILE)
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_WAIT_SECONDS = 7 * 24 * 60 * 60
POLL_SECONDS = 2.0
U_VARIABLES = 90
Y_SHAPE = (15, 3)
FULL_VARIABLES = 135
ORBIT_INDICES = (0, 1, 2)

PRODUCER_SOURCE = (
    "experiments/krenn_quantum_graph/"
    "star_variable_projection_artifact.py"
)
SOURCE_INPUTS = (
    "experiments/krenn_quantum_graph/formal_lift.py",
    "experiments/krenn_quantum_graph/star_linearization.py",
    "experiments/krenn_quantum_graph/star_pivot_charts.py",
    "experiments/krenn_quantum_graph/star_pivot_gauge.py",
    "experiments/krenn_quantum_graph/star_variable_projection.py",
    "experiments/krenn_quantum_graph/"
    "star_variable_projection_campaign.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/vertical_component.py",
)

_ARCHIVED_CAMPAIGN_BOUNDARY = {
    "numerical_candidate_is_exact_witness": False,
    "bounded_campaign_miss_is_nonexistence_proof": False,
    "radius_trend_is_border_proof": False,
    "horizontal_slice_y_bound_is_global_gauge_invariant": False,
    "full_135_norm_bounded_search_completed": False,
    "known_divergent_pole_excluded_by_search_domain": False,
    "numerical_two_cycle_is_exact_proof": False,
    "wall_clock_time_budget_is_reproducible_iteration_budget": False,
    "finite_affine_membership_status": "undecided",
}
_ARCHIVED_RESULT_BOUNDARY = {
    "numerical_zero_is_exact_counterexample": False,
    "bounded_miss_is_nonexistence_proof": False,
    "increasing_radius_trend_is_border_proof": False,
    "hard_coordinate_anchors_used": False,
    "raw_u_norm_used_as_mathematical_cap": False,
    "pole_quantity_Q_is_full_color_gauge_invariant": False,
    "pole_quantity_Q_used_as_cap": False,
    "bounded_y_is_only_a_P_equals_1_slice_control": True,
    "full_135_norm_bounded_search_completed": False,
    "known_divergent_pole_excluded_by_search_domain": False,
    "three_pivot_orbits_remain_an_exhaustive_open_cover": True,
    "numerical_two_cycle_is_exact_proof": False,
    "wall_clock_termination_is_bitwise_reproducible": False,
    "finite_affine_membership_status": "undecided",
    "exact_verification_attempted": False,
}
_SUMMARY_BOUNDARY = {
    "search_scope": (
        "three exact pivot-orbit P=1 slices with bounded recovered Y"
    ),
    "three_pivot_orbits_form_exhaustive_open_cover": True,
    "legacy_eight_seed_charts_used": False,
    "hard_coordinate_anchors_used": False,
    "raw_u_norm_used_as_mathematical_cap": False,
    "raw_u_overflow_cutoffs_enforced": True,
    "raw_u_overflow_cutoffs_depend_on_y_radius": True,
    "raw_u_overflow_cutoffs_are_mathematical_bounds": False,
    "bounded_y_is_full_color_gauge_invariant": False,
    "fixed_global_residual_gauge_slice_used": False,
    "cross_run_nonzero_objective_comparison_is_gauge_invariant": False,
    "horizontal_projection_rules_out_accumulated_gauge_drift": False,
    "full_135_norm_bounded_search_completed": False,
    "known_divergent_pole_excluded_by_search_domain": False,
    "pole_quantity_Q_is_full_color_gauge_invariant": False,
    "numerical_zero_is_exact_counterexample": False,
    "bounded_miss_is_nonexistence_proof": False,
    "radius_trend_is_border_proof": False,
    "numerical_two_cycle_is_exact_proof": False,
    "finite_affine_membership_status": "undecided",
}
_RUNTIME_FINGERPRINT_KEYS = {
    "campaign_source_sha256",
    "core_audit_sha256",
    "core_source",
    "core_source_sha256",
    "nonstar_character_rank_over_Q",
    "pivot_orbit_count",
    "schema",
    "star_linearization_sha256",
}
_RESULT_KEYS = {
    "schema",
    "config",
    "config_sha256",
    "source_fingerprints",
    "termination",
    "iterations_attempted",
    "accepted_steps",
    "patience_count",
    "cumulative_elapsed_seconds",
    "initialization",
    "best_u",
    "best_bounded_y",
    "best_full_weights",
    "best_evaluation",
    "finite_accounting",
    "independent_verification",
    "trace",
    "claim_boundary",
}
_CHECKPOINT_KEYS = {
    "schema",
    "config",
    "config_sha256",
    "source_fingerprints",
    "status",
    "iteration",
    "accepted_steps",
    "patience_count",
    "cumulative_elapsed_seconds",
    "u",
    "best_u",
    "evaluation",
    "best_evaluation",
    "lbfgs_history",
    "cycle_states",
    "trace",
    "initialization",
    "payload_sha256",
}
_MANIFEST_RESULT_KEYS = {
    "orbit_index",
    "seed",
    "radius",
    "initialization",
    "result_path",
    "termination",
    "best_objective",
    "best_raw_span_objective",
    "minimum_norm_y_frobenius",
    "bounded_y_frobenius",
    "y_cap_active",
}


class KrennVariableProjectionArtifactError(RuntimeError):
    """A scratch campaign or portable artifact failed replay."""


@dataclass(frozen=True)
class VerifiedVariableProjectionArtifact:
    directory: Path
    summary: Mapping
    manifest: Mapping
    checks: tuple[tuple[str, bool], ...]

    @property
    def valid(self) -> bool:
        return all(value for _name, value in self.checks)


def _canonical_json_bytes(payload) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (
        KrennVariableProjectionCampaignError,
        TypeError,
        ValueError,
    ) as error:
        raise KrennVariableProjectionArtifactError(
            "payload is not strict JSON"
        ) from error
    return (text + "\n").encode("ascii")


def _compact_sha256(payload) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise KrennVariableProjectionArtifactError(
            "payload cannot be fingerprinted"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennVariableProjectionArtifactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str):
    raise KrennVariableProjectionArtifactError(
        f"non-finite JSON constant {value!r}"
    )


def _load_json(path: Path, label: str) -> dict:
    try:
        if path.is_symlink() or not path.is_file():
            raise KrennVariableProjectionArtifactError(
                f"{label} is absent or linked"
            )
        if path.stat().st_size > MAX_JSON_BYTES:
            raise KrennVariableProjectionArtifactError(
                f"{label} exceeds its reviewed size bound"
            )
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except KrennVariableProjectionArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennVariableProjectionArtifactError(
            f"could not decode {label}"
        ) from error
    if type(payload) is not dict:
        raise KrennVariableProjectionArtifactError(
            f"{label} is not a JSON object"
        )
    if raw != _canonical_json_bytes(payload):
        raise KrennVariableProjectionArtifactError(
            f"{label} is not canonical round-trippable JSON"
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, label: str | None = None) -> dict:
    return {
        "path": path.name if label is None else label,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_record(label: str) -> dict:
    path = ROOT / label
    if path.is_symlink() or not path.is_file():
        raise KrennVariableProjectionArtifactError(
            f"source is absent or linked: {label}"
        )
    return {
        "path": label,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "hash_mode": "raw-bytes-v1",
    }


def _runtime_provenance(
    runtime_fingerprints: object,
    core: Any,
) -> dict:
    """Validate archived math hashes without rewriting runtime provenance."""

    if type(runtime_fingerprints) is not dict:
        raise KrennVariableProjectionArtifactError(
            "runtime source fingerprints are malformed"
        )
    current_core = core.source_fingerprints()
    if set(runtime_fingerprints) == _RUNTIME_FINGERPRINT_KEYS:
        if (
            any(
                runtime_fingerprints.get(key) != value
                for key, value in current_core.items()
            )
            or type(runtime_fingerprints["campaign_source_sha256"])
            is not str
            or len(runtime_fingerprints["campaign_source_sha256"]) != 64
            or any(
                piece not in "0123456789abcdef"
                for piece in runtime_fingerprints[
                    "campaign_source_sha256"
                ]
            )
        ):
            raise KrennVariableProjectionArtifactError(
                "archived runtime math fingerprints changed"
            )
        current_core_source = _source_record(
            "experiments/krenn_quantum_graph/"
            "star_variable_projection.py"
        )
        if (
            runtime_fingerprints["core_source_sha256"]
            != current_core_source["sha256"]
        ):
            raise KrennVariableProjectionArtifactError(
                "current core math source differs from archived runtime"
            )
        current_campaign_source = _source_record(
            "experiments/krenn_quantum_graph/"
            "star_variable_projection_campaign.py"
        )
        return {
            "mode": (
                "archived-runtime-provenance-plus-current-core-replay"
            ),
            "archived_runtime_fingerprints": runtime_fingerprints,
            "current_core_fingerprints": current_core,
            "current_core_source": current_core_source,
            "current_campaign_source": current_campaign_source,
            "archived_core_source_matches_current": True,
            "archived_campaign_source_matches_current": (
                runtime_fingerprints["campaign_source_sha256"]
                == current_campaign_source["sha256"]
            ),
            "current_campaign_source_used_to_generate_archived_results":
                False,
        }
    if runtime_fingerprints == current_core:
        # Deterministic test doubles and any future core-owned fixture.
        return {
            "mode": "core-owned-fingerprint-fixture",
            "archived_runtime_fingerprints": runtime_fingerprints,
            "current_core_fingerprints": current_core,
            "archived_core_source_matches_current": True,
            "archived_campaign_source_matches_current": None,
            "current_campaign_source_used_to_generate_archived_results":
                None,
        }
    raise KrennVariableProjectionArtifactError(
        "runtime fingerprints do not identify the replayed core"
    )


def _safe_local_file(directory: Path, label: object) -> Path:
    if type(label) is not str:
        raise KrennVariableProjectionArtifactError(
            "scratch path is not a string"
        )
    relative = Path(label)
    if (
        not label
        or relative.is_absolute()
        or bool(relative.drive)
        or len(relative.parts) != 1
        or label in (".", "..")
        or "/" in label
        or "\\" in label
        or ":" in label
    ):
        raise KrennVariableProjectionArtifactError(
            "scratch path escaped the campaign root"
        )
    path = directory / label
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True).parent != directory.resolve()
    ):
        raise KrennVariableProjectionArtifactError(
            f"scratch file is absent, linked, or escaped: {label}"
        )
    return path


def wait_for_terminal_manifest(
    scratch_root: Path,
    *,
    maximum_wait_seconds: float = 0.0,
    poll_seconds: float = POLL_SECONDS,
) -> Path:
    """Wait for ``manifest.json`` without inspecting other scratch files."""

    scratch_root = Path(scratch_root)
    if (
        scratch_root.is_symlink()
        or not scratch_root.is_dir()
        or isinstance(maximum_wait_seconds, bool)
        or not math.isfinite(float(maximum_wait_seconds))
        or not 0 <= maximum_wait_seconds <= MAX_WAIT_SECONDS
        or isinstance(poll_seconds, bool)
        or not math.isfinite(float(poll_seconds))
        or poll_seconds <= 0
    ):
        raise KrennVariableProjectionArtifactError(
            "scratch root or wait budget is invalid"
        )
    manifest = scratch_root / "manifest.json"
    deadline = time.monotonic() + float(maximum_wait_seconds)
    while not manifest.is_file() or manifest.is_symlink():
        if time.monotonic() >= deadline:
            raise KrennVariableProjectionArtifactError(
                "terminal manifest.json is not present"
            )
        time.sleep(min(float(poll_seconds), deadline - time.monotonic()))
    return manifest


def _numbers_close(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return bool(np.isclose(
            left, right, rtol=5e-11, atol=5e-13
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


def _complex_array(rows, shape: tuple[int, ...], label: str) -> np.ndarray:
    expected = math.prod(shape)
    if type(rows) is not list:
        raise KrennVariableProjectionArtifactError(
            f"{label} has the wrong length"
        )
    raw = np.asarray(rows, dtype=object)
    if raw.shape not in (shape + (2,), (expected, 2)):
        raise KrennVariableProjectionArtifactError(
            f"{label} has the wrong length"
        )
    for value in raw.reshape(-1):
        if (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
        ):
            raise KrennVariableProjectionArtifactError(
                f"{label} is not finite complex data"
            )
    pairs = np.asarray(raw, dtype=np.float64).reshape(expected, 2)
    values = pairs[:, 0] + 1j * pairs[:, 1]
    return np.asarray(values, dtype=np.complex128).reshape(shape)


def _complex_rows(values: np.ndarray) -> list[list[float]]:
    return [
        [float(value.real), float(value.imag)]
        for value in np.asarray(values).reshape(-1)
    ]


def _safe_l2(values: np.ndarray) -> float:
    array = np.asarray(values)
    components = np.concatenate((
        np.abs(array.real).reshape(-1),
        np.abs(array.imag).reshape(-1),
    ))
    maximum = float(np.max(components)) if components.size else 0.0
    if maximum == 0:
        return 0.0
    return maximum * math.sqrt(float(np.sum((components / maximum) ** 2)))


def _config(payload: Mapping) -> TrajectoryConfig:
    if type(payload) is not dict or payload.get("schema") != CONFIG_SCHEMA:
        raise KrennVariableProjectionArtifactError(
            "trajectory config schema changed"
        )
    fields = dict(payload)
    caps_payload = fields.pop("caps", None)
    if type(caps_payload) is not dict:
        raise KrennVariableProjectionArtifactError(
            "trajectory caps are malformed"
        )
    try:
        return TrajectoryConfig(
            caps=RadiusCaps(**caps_payload),
            **fields,
        )
    except (TypeError, ValueError) as error:
        raise KrennVariableProjectionArtifactError(
            "trajectory config could not be reconstructed"
        ) from error


def _representative(core: Any, orbit_index: int) -> Any:
    rows = tuple(core.pivot_representatives())
    matches = [
        row for row in rows
        if int(getattr(row, "orbit_index", getattr(row, "index", -1)))
        == orbit_index
    ]
    if len(matches) != 1:
        raise KrennVariableProjectionArtifactError(
            "core pivot representatives changed"
        )
    return matches[0]


def _trace_summary(trace: object) -> dict:
    if type(trace) is not list:
        raise KrennVariableProjectionArtifactError(
            "result trace is not a list"
        )
    rejection_counts: Counter[str] = Counter()
    cap_active = 0
    two_cycles = 0
    alternating = 0
    for row in trace:
        if type(row) is not dict:
            raise KrennVariableProjectionArtifactError(
                "result trace row is malformed"
            )
        cap_active += row.get("y_cap_active_after") is True
        two_cycles += row.get(
            "numerical_two_cycle_detected"
        ) is True
        alternating += row.get(
            "alternating_cap_with_nontrivial_retraction_detected"
        ) is True
        counts = row.get("backtrack_rejection_counts", {})
        if type(counts) is not dict:
            raise KrennVariableProjectionArtifactError(
                "trace backtrack accounting is malformed"
            )
        for key, value in counts.items():
            if type(key) is not str or type(value) is not int or value < 0:
                raise KrennVariableProjectionArtifactError(
                    "trace backtrack count is malformed"
                )
            rejection_counts[key] += value
    return {
        "accepted_trace_rows": len(trace),
        "y_cap_active_after_count": cap_active,
        "numerical_two_cycle_detected_count": two_cycles,
        "alternating_cap_with_nontrivial_retraction_count": alternating,
        "backtrack_rejection_counts": dict(
            sorted(rejection_counts.items())
        ),
    }


def _checkpoint_without_hash(checkpoint: Mapping) -> dict:
    return {
        key: value for key, value in checkpoint.items()
        if key != "payload_sha256"
    }


def _replay_result(
    scratch: Path,
    manifest_row: Mapping,
    core: Any,
    current_core_fingerprints: Mapping,
) -> tuple[dict, dict]:
    if (
        type(manifest_row) is not dict
        or set(manifest_row) != _MANIFEST_RESULT_KEYS
    ):
        raise KrennVariableProjectionArtifactError(
            "campaign result-row schema changed"
        )
    result_path = _safe_local_file(
        scratch, manifest_row["result_path"]
    )
    if not result_path.name.endswith(".result.json"):
        raise KrennVariableProjectionArtifactError(
            "result filename has no checkpoint peer"
        )
    checkpoint_path = _safe_local_file(
        scratch,
        result_path.name[:-len(".result.json")] + ".checkpoint.json",
    )
    result = _load_json(result_path, "trajectory result")
    checkpoint = _load_json(checkpoint_path, "trajectory checkpoint")
    if (
        set(result) != _RESULT_KEYS
        or result["schema"] != RESULT_SCHEMA
        or set(checkpoint) != _CHECKPOINT_KEYS
        or checkpoint["schema"] != CHECKPOINT_SCHEMA
        or checkpoint["payload_sha256"]
        != _compact_sha256(_checkpoint_without_hash(checkpoint))
    ):
        raise KrennVariableProjectionArtifactError(
            "result/checkpoint schema or payload hash changed"
        )
    config = _config(result["config"])
    if (
        result["config_sha256"] != config.fingerprint()
        or checkpoint["config"] != result["config"]
        or checkpoint["config_sha256"] != result["config_sha256"]
        or result["source_fingerprints"]
        != checkpoint["source_fingerprints"]
        or result["claim_boundary"] != _ARCHIVED_RESULT_BOUNDARY
        or config.orbit_index != manifest_row["orbit_index"]
        or config.seed != manifest_row["seed"]
        or config.caps.radius != manifest_row["radius"]
    ):
        raise KrennVariableProjectionArtifactError(
            "result configuration, source, or claim boundary changed"
        )
    provenance = _runtime_provenance(
        result["source_fingerprints"], core
    )
    if (
        provenance["current_core_fingerprints"]
        != current_core_fingerprints
    ):
        raise KrennVariableProjectionArtifactError(
            "current core fingerprint changed during replay"
        )
    result_initialization = manifest_row["initialization"]
    config_initialization = (
        "warm"
        if result_initialization in ("warm", "natural-repair-warm")
        else result_initialization
    )
    if config.initialization != config_initialization:
        raise KrennVariableProjectionArtifactError(
            "manifest and result initialization disagree"
        )
    overlap = {
        "status": result["termination"],
        "iteration": result["iterations_attempted"],
        "accepted_steps": result["accepted_steps"],
        "patience_count": result["patience_count"],
        "cumulative_elapsed_seconds": result[
            "cumulative_elapsed_seconds"
        ],
        "best_u": result["best_u"],
        "best_evaluation": result["best_evaluation"],
        "trace": result["trace"],
        "initialization": result["initialization"],
    }
    if any(checkpoint[key] != value for key, value in overlap.items()):
        raise KrennVariableProjectionArtifactError(
            "terminal result and checkpoint disagree"
        )
    if (
        type(result["accepted_steps"]) is not int
        or result["accepted_steps"] != len(result["trace"])
        or manifest_row["termination"] != result["termination"]
    ):
        raise KrennVariableProjectionArtifactError(
            "result trace or termination accounting changed"
        )
    representative = _representative(core, config.orbit_index)
    best_u = _complex_array(
        result["best_u"], (U_VARIABLES,), "best U"
    )
    pivots = np.asarray(
        core.pivot_values(best_u, representative),
        dtype=np.complex128,
    )
    if (
        pivots.shape != (3,)
        or not np.allclose(pivots, 1, rtol=3e-12, atol=3e-13)
        or not np.all(np.isfinite(best_u))
        or float(np.max(np.abs(best_u))) > config.caps.u_linf
        or _safe_l2(best_u) > config.caps.u_l2
    ):
        raise KrennVariableProjectionArtifactError(
            "best U left the P=1 finite-guard slice"
        )
    evaluation = evaluate_projection(
        core,
        best_u,
        representative,
        config.caps,
        rank_rtol=config.rank_rtol,
    )
    evaluation_summary = evaluation.summary()
    if not _numbers_close(
        result["best_evaluation"], evaluation_summary
    ):
        raise KrennVariableProjectionArtifactError(
            "best projection evaluation failed fresh replay"
        )
    bounded_y = _complex_array(
        result["best_bounded_y"], Y_SHAPE, "best bounded Y"
    )
    if not np.allclose(
        bounded_y, evaluation.y, rtol=5e-11, atol=5e-13
    ):
        raise KrennVariableProjectionArtifactError(
            "stored bounded Y failed fresh replay"
        )
    full_weights = np.asarray(
        core.reconstruct_full_weights(best_u, evaluation.y),
        dtype=np.complex128,
    )
    stored_full = _complex_array(
        result["best_full_weights"],
        (FULL_VARIABLES,),
        "best full weights",
    )
    if (
        full_weights.shape != (FULL_VARIABLES,)
        or not np.all(np.isfinite(full_weights))
        or not np.allclose(
            stored_full, full_weights, rtol=5e-11, atol=5e-13
        )
    ):
        raise KrennVariableProjectionArtifactError(
            "full 135-weight reconstruction failed replay"
        )
    replay = core.dual_full_residual(best_u, evaluation.y)
    if (
        replay.get("all_729_replayed") is not True
        or replay.get("primary_and_independent_outputs_agree") is not True
        or not _numbers_close(result["independent_verification"], replay)
    ):
        raise KrennVariableProjectionArtifactError(
            "primary/independent 729-equation replay failed"
        )
    evaluation_row = result["best_evaluation"]
    expected_manifest = {
        "orbit_index": config.orbit_index,
        "seed": config.seed,
        "radius": config.caps.radius,
        "initialization": result_initialization,
        "result_path": result_path.name,
        "termination": result["termination"],
        "best_objective": evaluation_row["objective"],
        "best_raw_span_objective": evaluation_row[
            "raw_span_objective"
        ],
        "minimum_norm_y_frobenius": evaluation_row[
            "minimum_norm_y_frobenius"
        ],
        "bounded_y_frobenius": evaluation_row[
            "bounded_y_frobenius"
        ],
        "y_cap_active": evaluation_row["y_cap_active"],
    }
    if not _numbers_close(manifest_row, expected_manifest):
        raise KrennVariableProjectionArtifactError(
            "manifest compact row failed result replay"
        )
    finite = result["finite_accounting"]
    full_accounting = finite.get("full_weights", {})
    if (
        finite.get("pivot_slice_y_frobenius_cap")
        != config.caps.y_frobenius
        or finite.get(
            "pivot_slice_y_cap_is_full_gauge_invariant"
        ) is not False
        or finite.get(
            "full_135_weight_norm_is_bounded_by_search_domain"
        ) is not False
        or finite.get(
            "known_divergent_pole_excluded_by_search_domain"
        ) is not False
        or full_accounting.get("all_135_weights_finite") is not True
        or not math.isclose(
            full_accounting.get("maximum_abs", math.nan),
            float(np.max(np.abs(full_weights))),
            rel_tol=5e-11,
            abs_tol=5e-13,
        )
        or not math.isclose(
            full_accounting.get("l2", math.nan),
            _safe_l2(full_weights),
            rel_tol=5e-11,
            abs_tol=5e-13,
        )
    ):
        raise KrennVariableProjectionArtifactError(
            "finite/path-boundary accounting changed"
        )
    candidate_id = (
        f"orbit-{config.orbit_index}_seed-{config.seed}_"
        f"radius-{format(config.caps.radius, '.12g')}_"
        f"{result_initialization}"
    )
    path_diag = evaluation_row["path_dependent_diagnostics"]
    compact = {
        "candidate_id": candidate_id,
        "orbit_index": config.orbit_index,
        "seed": config.seed,
        "radius": config.caps.radius,
        "initialization": result_initialization,
        "termination": result["termination"],
        "elapsed_seconds": result["cumulative_elapsed_seconds"],
        "iterations_attempted": result["iterations_attempted"],
        "accepted_steps": result["accepted_steps"],
        "patience_count": result["patience_count"],
        "objective": evaluation_row["objective"],
        "residual_l2": evaluation_row["residual_l2"],
        "target_distances_squared": evaluation_row[
            "target_distances_squared"
        ],
        "raw_span_objective": evaluation_row[
            "raw_span_objective"
        ],
        "raw_span_residual_l2": evaluation_row[
            "raw_span_residual_l2"
        ],
        "raw_target_distances_squared": evaluation_row[
            "raw_target_distances_squared"
        ],
        "bounded_y_frobenius": evaluation_row[
            "bounded_y_frobenius"
        ],
        "bounded_y_maximum_abs": evaluation_row[
            "bounded_y_maximum_abs"
        ],
        "minimum_norm_y_frobenius": evaluation_row[
            "minimum_norm_y_frobenius"
        ],
        "minimum_norm_y_log10_frobenius": evaluation_row[
            "minimum_norm_y_log10_frobenius"
        ],
        "minimum_norm_y_array_available": evaluation_row[
            "minimum_norm_y_array_available"
        ],
        "minimum_norm_y_overflowed_float64": evaluation_row[
            "minimum_norm_y_overflowed_float64"
        ],
        "y_cap_active": evaluation_row["y_cap_active"],
        "numerical_rank": evaluation_row["numerical_rank"],
        "condition_number": evaluation_row["condition_number"],
        "smallest_singular_value": evaluation_row[
            "singular_values"
        ][-1],
        "largest_singular_value": evaluation_row[
            "singular_values"
        ][0],
        "horizontal_gradient_real_norm": evaluation_row[
            "horizontal_gradient_real_norm"
        ],
        "horizontal_diagnostics": evaluation_row[
            "horizontal_diagnostics"
        ],
        "path_dependent_diagnostics": path_diag,
        "u_accounting": finite["u"],
        "full_weight_accounting": full_accounting,
        "independent_verification": replay,
        "trace_summary": _trace_summary(result["trace"]),
        "exact_verification": {
            "attempted": False,
            "status": "floating-point-reconnaissance-only",
        },
        "scratch_records": {
            "result": _record(result_path),
            "checkpoint": _record(checkpoint_path),
        },
        "config_sha256": result["config_sha256"],
        "runtime_source_fingerprints_sha256": _compact_sha256(
            result["source_fingerprints"]
        ),
    }
    candidate = {
        "candidate_id": candidate_id,
        "config": result["config"],
        "config_sha256": result["config_sha256"],
        "source_fingerprints": result["source_fingerprints"],
        "best_u": result["best_u"],
        "best_bounded_y": result["best_bounded_y"],
        "best_full_weights": result["best_full_weights"],
        "best_evaluation": result["best_evaluation"],
        "finite_accounting": result["finite_accounting"],
        "independent_verification": result[
            "independent_verification"
        ],
        "claim_boundary": result["claim_boundary"],
        "runtime_provenance": provenance,
        "source_result_record": _record(result_path),
    }
    return compact, candidate


def _input_ledger_digest(rows: Sequence[Mapping]) -> str:
    return _compact_sha256(list(rows))


def summarize_completed_campaign(
    scratch_root: Path,
    *,
    maximum_wait_seconds: float = 0.0,
) -> dict:
    """Replay a terminal campaign and build a portable summary payload."""

    manifest_path = wait_for_terminal_manifest(
        scratch_root,
        maximum_wait_seconds=maximum_wait_seconds,
    )
    scratch = manifest_path.parent.resolve(strict=True)
    campaign = _load_json(manifest_path, "terminal campaign manifest")
    if (
        campaign.get("schema") != CAMPAIGN_SCHEMA
        or campaign.get("scratch_root") != str(scratch)
        or campaign.get("core_module") != DEFAULT_CORE_MODULE
        or campaign.get("claim_boundary")
        != _ARCHIVED_CAMPAIGN_BOUNDARY
        or campaign.get("orbit_cover") != {
            "fixed_apex": 0,
            "representative_count": 3,
            "orbit_indices": [0, 1, 2],
            "orbit_sizes": [5, 60, 60],
            "provenance": (
                "S5 residual vertices x S3 colors target-minor pivots"
            ),
            "legacy_eight_seed_charts_used": False,
            "hard_coordinate_anchors_used": False,
        }
    ):
        raise KrennVariableProjectionArtifactError(
            "terminal campaign identity or claim boundary changed"
        )
    gauge = campaign.get("gauge_control")
    if (
        type(gauge) is not dict
        or gauge.get(
            "selected_pivots_root_free_normalized_to_one"
        ) is not True
        or gauge.get(
            "residual_norm_balancing_refused_by_exact_recession"
        ) is not True
        or gauge.get("horizontal_residual_gauge_projection") is not True
        or gauge.get("raw_u_norm_mathematical_cap") is not False
        or gauge.get("raw_y_norm_full_gauge_invariant") is not False
        or gauge.get(
            "full_135_weight_norm_is_bounded_by_search_domain"
        ) is not False
        or gauge.get(
            "known_divergent_pole_excluded_by_search_domain"
        ) is not False
    ):
        raise KrennVariableProjectionArtifactError(
            "campaign gauge-control boundary changed"
        )
    rows = campaign.get("results")
    if (
        type(rows) is not list
        or campaign.get("trajectory_count") != len(rows)
        or campaign.get("expected_trajectory_count") != len(rows)
        or not rows
    ):
        raise KrennVariableProjectionArtifactError(
            "terminal campaign trajectory count is incomplete"
        )
    core = importlib.import_module(DEFAULT_CORE_MODULE)
    source_fingerprints = core.source_fingerprints()
    compact_rows = []
    candidates = {}
    identities = set()
    input_ledger = [_record(manifest_path)]
    for row in rows:
        compact, candidate = _replay_result(
            scratch,
            row,
            core,
            source_fingerprints,
        )
        identity = (
            compact["orbit_index"],
            compact["seed"],
            compact["radius"],
            compact["initialization"],
        )
        if identity in identities:
            raise KrennVariableProjectionArtifactError(
                "campaign trajectory identity is duplicated"
            )
        identities.add(identity)
        compact_rows.append(compact)
        candidates[compact["candidate_id"]] = candidate
        input_ledger.extend(compact["scratch_records"].values())
    smallest_recorded = min(
        rows, key=lambda row: row["best_objective"]
    )
    if campaign.get(
        "best_p_equals_one_y_bounded_candidate"
    ) != smallest_recorded:
        raise KrennVariableProjectionArtifactError(
            "campaign smallest recorded lift-dependent row changed"
        )
    compact_rows.sort(
        key=lambda row: (
            row["orbit_index"],
            row["seed"],
            row["radius"],
            row["initialization"],
        )
    )
    smallest_bounded = min(
        compact_rows, key=lambda row: row["objective"]
    )
    smallest_raw = min(
        compact_rows, key=lambda row: row["raw_span_objective"]
    )
    per_cell = {}
    for key, cell_rows in _group_by_cell(compact_rows).items():
        per_cell[key] = min(
            cell_rows, key=lambda row: row["objective"]
        )
    selection_reasons: defaultdict[str, list[str]] = defaultdict(list)
    selection_reasons[smallest_bounded["candidate_id"]].append(
        "smallest-recorded-objective-in-deterministic-lifts"
    )
    selection_reasons[smallest_raw["candidate_id"]].append(
        "smallest-recorded-raw-span-objective-in-deterministic-lifts"
    )
    for key, winner in per_cell.items():
        selection_reasons[winner["candidate_id"]].append(
            f"orbit-radius-smallest-recorded-lift-dependent-row:{key}"
        )
    portable_candidates = []
    for candidate_id in sorted(selection_reasons):
        payload = dict(candidates[candidate_id])
        payload["selection_reasons"] = sorted(
            selection_reasons[candidate_id]
        )
        portable_candidates.append(payload)
    terminations = Counter(
        row["termination"] for row in compact_rows
    )
    cap_counts = Counter(
        format(row["radius"], ".12g")
        for row in compact_rows if row["y_cap_active"]
    )
    trend_rows = [
        {
            key: winner[key]
            for key in (
                "candidate_id",
                "orbit_index",
                "radius",
                "seed",
                "initialization",
                "objective",
                "residual_l2",
                "raw_span_objective",
                "raw_span_residual_l2",
                "bounded_y_frobenius",
                "minimum_norm_y_frobenius",
                "minimum_norm_y_log10_frobenius",
                "y_cap_active",
                "numerical_rank",
                "condition_number",
                "u_accounting",
                "full_weight_accounting",
                "path_dependent_diagnostics",
            )
        }
        for _key, winner in sorted(
            per_cell.items(),
            key=lambda item: (
                item[1]["orbit_index"], item[1]["radius"]
            ),
        )
    ]
    input_ledger = sorted(input_ledger, key=lambda row: row["path"])
    replay_command = [
        "python",
        "-B",
        "-m",
        "experiments.krenn_quantum_graph."
        "star_variable_projection_campaign",
        "--scratch-root",
        str(scratch),
        "--radii",
        ",".join(map(str, campaign["radii"])),
        "--seeds",
        ",".join(map(str, campaign["seeds"])),
        "--initializations",
        ",".join(campaign["initializations"]),
        "--workers",
        str(campaign["workers"]),
        "--maximum-iterations",
        str(campaign["maximum_iterations_per_trajectory"]),
        "--maximum-seconds-per-trajectory",
        str(campaign["maximum_seconds_per_trajectory"]),
        "--patience",
        str(campaign["patience"]),
        "--checkpoint-interval",
        str(campaign["checkpoint_interval"]),
        "--natural-repair-scale",
        str(campaign["natural_repair_scale"]),
        "--resume",
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "problem": {
            "n": 6,
            "d": 3,
            "apex": 0,
            "u_variables": U_VARIABLES,
            "star_variables": 45,
            "full_variables": FULL_VARIABLES,
            "equations_replayed_per_candidate": 729,
            "target": "GHZ_6,3",
        },
        "campaign": {
            "original_scratch_root": str(scratch),
            "core_module": campaign["core_module"],
            "radii": campaign["radii"],
            "seeds": campaign["seeds"],
            "initializations": campaign["initializations"],
            "workers": campaign["workers"],
            "maximum_workers": campaign["maximum_workers"],
            "maximum_iterations_per_trajectory": campaign[
                "maximum_iterations_per_trajectory"
            ],
            "maximum_seconds_per_trajectory": campaign[
                "maximum_seconds_per_trajectory"
            ],
            "patience": campaign["patience"],
            "checkpoint_interval": campaign["checkpoint_interval"],
            "natural_repair_scale": campaign[
                "natural_repair_scale"
            ],
            "trajectory_count": len(compact_rows),
            "expected_trajectory_count": campaign[
                "expected_trajectory_count"
            ],
            "canonical_replay_argv": replay_command,
            "wall_clock_termination_is_bitwise_reproducible": False,
            "fixed_global_residual_gauge_slice_used": False,
            "cross_run_nonzero_objective_comparison_is_gauge_invariant":
                False,
            "horizontal_projection_rules_out_accumulated_gauge_drift":
                False,
            "raw_u_overflow_cutoffs_enforced": True,
            "raw_u_overflow_cutoffs_depend_on_y_radius": True,
            "raw_u_overflow_cutoffs_are_mathematical_bounds": False,
        },
        "orbit_cover": campaign["orbit_cover"],
        "gauge_control": {
            **campaign["gauge_control"],
            "fixed_global_residual_gauge_slice": False,
            "cross_run_nonzero_objective_comparison_gauge_invariant":
                False,
            "horizontal_projection_prevents_accumulated_gauge_drift":
                False,
            "raw_u_overflow_cutoff_enforced": True,
            "raw_u_overflow_cutoff_depends_on_y_radius": True,
            "raw_u_overflow_cutoff_is_mathematical_bound": False,
        },
        "runtime_provenance": portable_candidates[0][
            "runtime_provenance"
        ],
        "trajectory_rows": compact_rows,
        "aggregate": {
            "termination_counts": dict(sorted(terminations.items())),
            "numerical_tolerance_count": sum(
                row["termination"] == "numerical-tolerance"
                for row in compact_rows
            ),
            "all_primary_independent_729_replays_agree": all(
                row["independent_verification"].get(
                    "primary_and_independent_outputs_agree"
                ) is True
                for row in compact_rows
            ),
            "all_135_weight_vectors_finite": all(
                row["full_weight_accounting"][
                    "all_135_weights_finite"
                ] is True
                for row in compact_rows
            ),
            "y_cap_active_counts_by_radius": dict(sorted(cap_counts.items())),
            "smallest_recorded_objective_in_deterministic_lifts_id":
                smallest_bounded[
                "candidate_id"
            ],
            "smallest_recorded_raw_span_objective_in_deterministic_lifts_id":
                smallest_raw[
                "candidate_id"
            ],
            "orbit_radius_recorded_minima": trend_rows,
            "cross_run_nonzero_objectives_are_intrinsically_comparable":
                False,
            "exact_verification": {
                "attempted_trajectories": 0,
                "finite_exact_witnesses_verified": 0,
            },
        },
        "portable_evidence": {
            "selected_candidates": portable_candidates,
            "selection_rule": (
                "deduplicated smallest recorded bounded/raw-span rows "
                "in deterministic lifts and every orbit-radius "
                "smallest recorded bounded row; these are not "
                "gauge-invariant rankings"
            ),
            "all_selected_candidates_freshly_replayable_without_scratch":
                True,
            "all_nonselected_trajectory_replays_are_generation_time_only":
                True,
            "verification_requires_original_scratch_directory": False,
        },
        "scratch_input_ledger": input_ledger,
        "scratch_input_ledger_sha256": _input_ledger_digest(
            input_ledger
        ),
        "source_ledger": [
            _source_record(label) for label in SOURCE_INPUTS
        ],
        "verification": {
            "all_terminal_result_checkpoint_pairs_cross_checked": True,
            "all_best_projections_freshly_evaluated_at_generation": True,
            "all_135_weights_reconstructed_at_generation": True,
            "all_729_equations_dually_replayed_at_generation": True,
            "portable_selected_candidate_count": len(
                portable_candidates
            ),
            "raw_traces_copied_into_portable_bundle": False,
        },
        "claim_boundary": dict(_SUMMARY_BOUNDARY),
    }


def _group_by_cell(
    rows: Sequence[Mapping],
) -> dict[str, list[Mapping]]:
    grouped: defaultdict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        key = (
            f"orbit-{row['orbit_index']}:"
            f"radius-{format(row['radius'], '.12g')}"
        )
        grouped[key].append(row)
    return dict(grouped)


def _verify_candidate(candidate: Mapping, core: Any) -> None:
    required = {
        "candidate_id",
        "config",
        "config_sha256",
        "source_fingerprints",
        "best_u",
        "best_bounded_y",
        "best_full_weights",
        "best_evaluation",
        "finite_accounting",
        "independent_verification",
        "claim_boundary",
        "runtime_provenance",
        "source_result_record",
        "selection_reasons",
    }
    if type(candidate) is not dict or set(candidate) != required:
        raise KrennVariableProjectionArtifactError(
            "portable candidate schema changed"
        )
    config = _config(candidate["config"])
    if (
        candidate["config_sha256"] != config.fingerprint()
        or candidate["claim_boundary"] != _ARCHIVED_RESULT_BOUNDARY
        or type(candidate["selection_reasons"]) is not list
        or not candidate["selection_reasons"]
    ):
        raise KrennVariableProjectionArtifactError(
            "portable candidate config/source/claims changed"
        )
    if candidate["runtime_provenance"] != _runtime_provenance(
        candidate["source_fingerprints"], core
    ):
        raise KrennVariableProjectionArtifactError(
            "portable runtime/current source provenance changed"
        )
    representative = _representative(core, config.orbit_index)
    u = _complex_array(candidate["best_u"], (U_VARIABLES,), "portable U")
    evaluation = evaluate_projection(
        core,
        u,
        representative,
        config.caps,
        rank_rtol=config.rank_rtol,
    )
    if not _numbers_close(
        candidate["best_evaluation"], evaluation.summary()
    ):
        raise KrennVariableProjectionArtifactError(
            "portable candidate projection failed replay"
        )
    y = _complex_array(
        candidate["best_bounded_y"], Y_SHAPE, "portable bounded Y"
    )
    if not np.allclose(y, evaluation.y, rtol=5e-11, atol=5e-13):
        raise KrennVariableProjectionArtifactError(
            "portable bounded Y failed replay"
        )
    full = np.asarray(
        core.reconstruct_full_weights(u, evaluation.y),
        dtype=np.complex128,
    )
    stored_full = _complex_array(
        candidate["best_full_weights"],
        (FULL_VARIABLES,),
        "portable full weights",
    )
    if not np.allclose(
        full, stored_full, rtol=5e-11, atol=5e-13
    ):
        raise KrennVariableProjectionArtifactError(
            "portable full weights failed reconstruction"
        )
    replay = core.dual_full_residual(u, evaluation.y)
    if (
        replay.get("all_729_replayed") is not True
        or replay.get("primary_and_independent_outputs_agree") is not True
        or not _numbers_close(
            candidate["independent_verification"], replay
        )
    ):
        raise KrennVariableProjectionArtifactError(
            "portable primary/independent 729 replay failed"
        )


def _verify_summary(summary: Mapping) -> None:
    if (
        type(summary) is not dict
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("claim_boundary") != _SUMMARY_BOUNDARY
        or summary.get("problem", {}).get("full_variables")
        != FULL_VARIABLES
        or summary.get("campaign", {}).get("core_module")
        != DEFAULT_CORE_MODULE
        or summary.get("orbit_cover", {}).get(
            "representative_count"
        ) != 3
        or summary.get("gauge_control", {}).get(
            "full_135_weight_norm_is_bounded_by_search_domain"
        ) is not False
        or summary.get("gauge_control", {}).get(
            "known_divergent_pole_excluded_by_search_domain"
        ) is not False
        or summary.get("gauge_control", {}).get(
            "raw_u_overflow_cutoff_enforced"
        ) is not True
        or summary.get("gauge_control", {}).get(
            "raw_u_overflow_cutoff_depends_on_y_radius"
        ) is not True
        or summary.get("gauge_control", {}).get(
            "cross_run_nonzero_objective_comparison_gauge_invariant"
        ) is not False
        or summary.get("source_ledger") != [
            _source_record(label) for label in SOURCE_INPUTS
        ]
    ):
        raise KrennVariableProjectionArtifactError(
            "portable summary static fields or claims changed"
        )
    ledger = summary.get("scratch_input_ledger")
    if (
        type(ledger) is not list
        or summary.get("scratch_input_ledger_sha256")
        != _input_ledger_digest(ledger)
    ):
        raise KrennVariableProjectionArtifactError(
            "scratch input ledger digest changed"
        )
    rows = summary.get("trajectory_rows")
    if (
        type(rows) is not list
        or len(rows) != summary["campaign"]["trajectory_count"]
        or len({
            (
                row["orbit_index"],
                row["seed"],
                row["radius"],
                row["initialization"],
            )
            for row in rows
        }) != len(rows)
    ):
        raise KrennVariableProjectionArtifactError(
            "portable compact trajectory inventory changed"
        )
    aggregate = summary.get("aggregate")
    terminations = Counter(row["termination"] for row in rows)
    smallest_bounded = min(rows, key=lambda row: row["objective"])
    smallest_raw = min(
        rows, key=lambda row: row["raw_span_objective"]
    )
    if (
        type(aggregate) is not dict
        or aggregate.get("termination_counts")
        != dict(sorted(terminations.items()))
        or aggregate.get(
            "smallest_recorded_objective_in_deterministic_lifts_id"
        ) != smallest_bounded["candidate_id"]
        or aggregate.get(
            "smallest_recorded_raw_span_objective_in_deterministic_lifts_id"
        ) != smallest_raw["candidate_id"]
        or aggregate.get(
            "cross_run_nonzero_objectives_are_intrinsically_comparable"
        ) is not False
        or aggregate.get(
            "all_primary_independent_729_replays_agree"
        ) is not True
        or aggregate.get("all_135_weight_vectors_finite") is not True
        or aggregate.get("exact_verification") != {
            "attempted_trajectories": 0,
            "finite_exact_witnesses_verified": 0,
        }
    ):
        raise KrennVariableProjectionArtifactError(
            "portable campaign aggregate changed"
        )
    evidence = summary.get("portable_evidence")
    if (
        type(evidence) is not dict
        or evidence.get(
            "all_selected_candidates_freshly_replayable_without_scratch"
        ) is not True
        or evidence.get(
            "all_nonselected_trajectory_replays_are_generation_time_only"
        ) is not True
        or evidence.get(
            "verification_requires_original_scratch_directory"
        ) is not False
        or type(evidence.get("selected_candidates")) is not list
        or not evidence["selected_candidates"]
    ):
        raise KrennVariableProjectionArtifactError(
            "portable evidence schema changed"
        )
    core = importlib.import_module(DEFAULT_CORE_MODULE)
    candidate_ids = set()
    reasons = []
    for candidate in evidence["selected_candidates"]:
        _verify_candidate(candidate, core)
        candidate_id = candidate["candidate_id"]
        if candidate_id in candidate_ids:
            raise KrennVariableProjectionArtifactError(
                "portable candidate is duplicated"
            )
        candidate_ids.add(candidate_id)
        reasons.extend(candidate["selection_reasons"])
    if (
        summary.get("runtime_provenance")
        != evidence["selected_candidates"][0]["runtime_provenance"]
        or any(
            candidate["runtime_provenance"]
            != summary["runtime_provenance"]
            for candidate in evidence["selected_candidates"]
        )
    ):
        raise KrennVariableProjectionArtifactError(
            "portable runtime provenance is inconsistent"
        )
    required_ids = {
        smallest_bounded["candidate_id"],
        smallest_raw["candidate_id"],
        *(
            min(cell, key=lambda row: row["objective"])[
                "candidate_id"
            ]
            for cell in _group_by_cell(rows).values()
        ),
    }
    if candidate_ids != required_ids:
        raise KrennVariableProjectionArtifactError(
            "portable candidate selection is incomplete"
        )


def generate_variable_projection_summary_bundle(
    scratch_root: Path,
    output_directory: Path,
    *,
    maximum_wait_seconds: float = 0.0,
) -> Path:
    """Replay scratch and write a portable two-file numerical bundle."""

    summary = summarize_completed_campaign(
        scratch_root,
        maximum_wait_seconds=maximum_wait_seconds,
    )
    output_directory = Path(output_directory)
    if output_directory.exists() and (
        output_directory.is_symlink()
        or not output_directory.is_dir()
    ):
        raise KrennVariableProjectionArtifactError(
            "artifact output must be a real directory"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(
        path.name not in ALL_FILES for path in output_directory.iterdir()
    ):
        raise KrennVariableProjectionArtifactError(
            "artifact output contains unreviewed files"
        )
    summary_path = output_directory / SUMMARY_FILE
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(output_directory / MANIFEST_FILE, {
        "schema": MANIFEST_SCHEMA,
        "artifact": _record(summary_path),
        "producer": _source_record(PRODUCER_SOURCE),
        "source_inputs": [
            _source_record(label) for label in SOURCE_INPUTS
        ],
        "scratch_input_file_count": len(
            summary["scratch_input_ledger"]
        ),
        "scratch_input_ledger_sha256": summary[
            "scratch_input_ledger_sha256"
        ],
        "portable_without_scratch": True,
        "verification_scope": (
            "selected candidates replay projection and both 729-equation "
            "enumerators; nonselected rows retain generation-time replay"
        ),
        "claim_boundary": dict(_SUMMARY_BOUNDARY),
    })
    return summary_path


def verify_variable_projection_summary_bundle(
    directory: Path,
) -> VerifiedVariableProjectionArtifact:
    """Verify the two-file bundle without reading campaign scratch."""

    directory = Path(directory)
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {path.name for path in directory.iterdir()} != set(ALL_FILES)
    ):
        raise KrennVariableProjectionArtifactError(
            "portable artifact inventory changed"
        )
    summary_path = directory / SUMMARY_FILE
    summary = _load_json(summary_path, "portable summary")
    manifest = _load_json(
        directory / MANIFEST_FILE, "portable manifest"
    )
    if (
        set(manifest) != {
            "schema",
            "artifact",
            "producer",
            "source_inputs",
            "scratch_input_file_count",
            "scratch_input_ledger_sha256",
            "portable_without_scratch",
            "verification_scope",
            "claim_boundary",
        }
        or manifest["schema"] != MANIFEST_SCHEMA
        or manifest["artifact"] != _record(summary_path)
        or manifest["producer"] != _source_record(PRODUCER_SOURCE)
        or manifest["source_inputs"] != [
            _source_record(label) for label in SOURCE_INPUTS
        ]
        or manifest["scratch_input_file_count"]
        != len(summary.get("scratch_input_ledger", []))
        or manifest["scratch_input_ledger_sha256"]
        != summary.get("scratch_input_ledger_sha256")
        or manifest["portable_without_scratch"] is not True
        or manifest["claim_boundary"] != _SUMMARY_BOUNDARY
    ):
        raise KrennVariableProjectionArtifactError(
            "portable manifest failed replay"
        )
    _verify_summary(summary)
    loaded = VerifiedVariableProjectionArtifact(
        directory=directory.resolve(),
        summary=summary,
        manifest=manifest,
        checks=(
            ("bundle_inventory_exact", True),
            ("bundle_hash_replayed", True),
            ("source_hashes_replayed", True),
            ("scratch_ledger_digest_replayed", True),
            ("compact_trajectory_aggregates_replayed", True),
            ("selected_projection_evaluations_replayed", True),
            ("selected_135_weight_reconstructions_replayed", True),
            ("selected_primary_independent_729_replays_agree", True),
            ("portable_without_scratch", True),
            ("claim_boundaries_fail_closed", True),
        ),
    )
    if not loaded.valid:
        raise KrennVariableProjectionArtifactError(
            "portable artifact checks failed"
        )
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify the portable star variable-projection "
            "campaign summary."
        )
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--from-scratch",
        type=Path,
        help="completed campaign scratch root; generation only",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help="wait only for terminal manifest.json before reading scratch",
    )
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        if arguments.from_scratch is None:
            parser.error("--from-scratch is required when generating")
        generate_variable_projection_summary_bundle(
            arguments.from_scratch,
            arguments.output_directory,
            maximum_wait_seconds=arguments.wait_seconds,
        )
    loaded = verify_variable_projection_summary_bundle(
        arguments.output_directory
    )
    print(json.dumps({
        "valid": loaded.valid,
        "trajectory_count": loaded.summary["campaign"][
            "trajectory_count"
        ],
        "portable_selected_candidate_count": loaded.summary[
            "verification"
        ]["portable_selected_candidate_count"],
        "smallest_recorded_objective_in_deterministic_lifts":
            loaded.summary["aggregate"][
                "smallest_recorded_objective_in_deterministic_lifts_id"
            ],
        "cross_run_nonzero_objective_comparison_is_gauge_invariant":
            False,
        "full_135_norm_bounded_search_completed": False,
        "finite_affine_membership_status": "undecided",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
