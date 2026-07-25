"""Fail-closed summary artifact for the bounded ``n=6,d=3`` star-ALS run.

The disposable campaign has eight group manifests: two initializations
times four explicit ``(Linf,L2)`` caps, with three fixed seeds in every
group.  Large results/checkpoints stay in scratch storage.  This module
replays them and emits only a small summary bundle.

Every final, best, and checkpoint-initial vector is evaluated in all 729
equations through the public primary/independent replay API.  The fixed
15-anchor gauge chart, both hard bounds, source hashes, exact star-matrix
fingerprints, result/checkpoint agreement, and claim boundaries are
checked fail-closed.

This is numerical reconnaissance on one open chart only.  It is not the
legacy eight-chart search or the three-pivot-orbit cover.  A numerical
zero is not an exact counterexample, and a bounded miss is not a proof.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np

from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
)
from experiments.krenn_quantum_graph.star_alternating_search import (
    DEFAULT_SEEDS,
    EQUATIONS,
    GAUGE_ANCHOR_VALUE,
    GAUGE_DIMENSION,
    N,
    STAR_ALS_CAMPAIGN_SCHEMA,
    STAR_ALS_CHECKPOINT_SCHEMA,
    STAR_ALS_SCHEMA,
    VARIABLES,
    KrennStarALSError,
    StarALSConfig,
    dual_residual_accounting,
    gauge_chart_audit,
)
from experiments.krenn_quantum_graph.star_linearization import (
    star_linearization,
)
from experiments.krenn_quantum_graph.star_linearization_independent import (
    independent_star_linearization_audit,
)
from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    variable_index,
)


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "krenn-n6-d3-star-als-summary-v1"
MANIFEST_SCHEMA = "krenn-n6-d3-star-als-summary-manifest-v1"
SUMMARY_FILE = "summary.json"
MANIFEST_FILE = "manifest.json"
ALL_FILES = (SUMMARY_FILE, MANIFEST_FILE)
EXPECTED_INITIALIZATIONS = ("natural-perturbed", "random")
EXPECTED_GROUPS = 8
EXPECTED_CAP_PAIRS = 4
EXPECTED_RUNS = 24
MAX_DECLARED_WORKERS = 16
MAX_GROUP_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_RUN_FILE_BYTES = 64 * 1024 * 1024
MAX_SUMMARY_BYTES = 4 * 1024 * 1024

PRODUCER_FILE = "experiments/krenn_quantum_graph/star_als_artifact.py"
NUMERICAL_SOURCE_FILES = (
    "experiments/krenn_quantum_graph/star_alternating_search.py",
    "experiments/krenn_quantum_graph/star_linearization.py",
    "experiments/krenn_quantum_graph/"
    "star_linearization_independent.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/vertical_component.py",
)

_GROUP_KEYS = {
    "schema", "output_directory", "gauge_chart",
    "radii_as_global_linf_caps", "explicit_global_l2_caps",
    "default_global_l2_cap_rule", "seeds",
    "maximum_sweeps_per_run", "maximum_seconds_per_run",
    "patience_apex_updates", "minimum_relative_improvement",
    "initialization", "workers", "schedule", "randomness",
    "checkpointing", "results", "best_bounded_candidate",
    "claim_boundary",
}
_GROUP_ROW_KEYS = {
    "radius", "effective_bounds", "seed", "result_path",
    "completed_sweeps", "completed_apex_updates", "termination",
    "elapsed_seconds", "final_residual_l2",
    "independent_final_residual_l2", "maximum_weight_abs",
    "weight_l2",
}
_RESULT_KEYS = {
    "schema", "config", "effective_bounds", "config_sha256",
    "source_sha256", "star_sha256", "gauge_chart",
    "completed_sweeps", "completed_apex_updates", "cursor",
    "termination", "patience_count", "cumulative_elapsed_seconds",
    "initial_residual", "final_residual", "best_residual",
    "weights", "best_weights", "trace", "finite_bound",
    "gauge_invariant_growth", "independent_verification",
    "exact_verification", "claim_boundary",
}
_CHECKPOINT_KEYS = {
    "schema", "config", "effective_bounds", "config_sha256",
    "source_sha256", "star_sha256", "gauge_chart", "state",
    "claim_boundary",
}
_STATE_KEYS = {
    "status", "cursor", "weights", "best_weights", "initial_weights",
    "initial_residual", "current_residual", "best_residual", "trace",
    "patience_count", "cumulative_elapsed_seconds",
}
_GROUP_BOUNDARY = {
    "numerical_candidate_is_exact_witness": False,
    "bounded_campaign_miss_is_nonexistence_proof": False,
    "open_chart_search_is_global_affine_search": False,
    "finite_affine_membership_status": "undecided",
}
_RESULT_BOUNDARY = {
    "numerical_zero_is_exact_counterexample": False,
    "bounded_miss_is_nonexistence_proof": False,
    "increasing_radius_trend_is_border_proof": False,
    "open_chart_covers_zero_anchor_solutions": False,
    "finite_affine_membership_status": "undecided",
}
_CHECKPOINT_BOUNDARY = {
    "numerical_zero_is_exact_counterexample": False,
    "bounded_miss_is_nonexistence_proof": False,
    "open_gauge_chart_is_global_affine_space": False,
}
_SUMMARY_BOUNDARY = {
    "search_scope": "one fixed 15-anchor open gauge chart only",
    "legacy_eight_charts_searched": False,
    "three_pivot_orbit_cover_searched": False,
    "all_affine_solutions_covered": False,
    "raw_weight_caps_are_full_color_gauge_invariant": False,
    "known_pole_product_is_full_color_gauge_invariant": False,
    "numerical_zero_is_exact_counterexample": False,
    "bounded_miss_is_nonexistence_proof": False,
    "increasing_cap_trend_is_border_proof": False,
    "finite_affine_membership_status": "undecided",
}


class KrennStarALSArtifactError(RuntimeError):
    """A scratch campaign or small summary failed replay."""


@dataclass(frozen=True)
class VerifiedStarALSArtifact:
    directory: Path
    campaign_root: Path
    summary: Mapping
    manifest: Mapping
    checks: tuple[tuple[str, bool], ...]

    @property
    def valid(self) -> bool:
        return all(value for _name, value in self.checks)


def _canonical_json_bytes(payload) -> bytes:
    try:
        text = json.dumps(
            payload, allow_nan=False, indent=2, sort_keys=True
        )
    except (TypeError, ValueError) as error:
        raise KrennStarALSArtifactError(
            "payload is not strict JSON"
        ) from error
    return (text + "\n").encode("utf-8")


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
            raise KrennStarALSArtifactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str):
    raise KrennStarALSArtifactError(
        f"non-finite JSON constant {value!r}"
    )


def _load_json(path: Path, maximum_bytes: int, label: str) -> dict:
    try:
        if path.is_symlink() or not path.is_file():
            raise KrennStarALSArtifactError(
                f"{label} is absent or linked"
            )
        if path.stat().st_size > maximum_bytes:
            raise KrennStarALSArtifactError(
                f"{label} exceeds its reviewed size bound"
            )
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except KrennStarALSArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennStarALSArtifactError(
            f"could not decode {label}"
        ) from error
    if type(payload) is not dict:
        raise KrennStarALSArtifactError(
            f"{label} is not a JSON object"
        )
    if raw != _canonical_json_bytes(payload):
        raise KrennStarALSArtifactError(
            f"{label} is not canonical round-trippable JSON"
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(label: str) -> dict:
    path = ROOT / label
    if path.is_symlink() or not path.is_file():
        raise KrennStarALSArtifactError(
            f"source input is absent or linked: {label}"
        )
    return {
        "path": label,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "hash_mode": "raw-bytes-v1",
    }


def _file_record(path: Path, root: Path) -> dict:
    path = path.resolve(strict=True)
    try:
        label = str(path.relative_to(root)).replace("\\", "/")
    except ValueError as error:
        raise KrennStarALSArtifactError(
            "an input escaped the campaign root"
        ) from error
    return {
        "path": label,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _safe_local_file(directory: Path, label: object) -> Path:
    if type(label) is not str:
        raise KrennStarALSArtifactError("run path is not a string")
    relative = Path(label)
    if (
        not label or relative.is_absolute() or relative.drive
        or len(relative.parts) != 1 or label in (".", "..")
        or "/" in label or "\\" in label or ":" in label
    ):
        raise KrennStarALSArtifactError(
            "run path escaped its group directory"
        )
    path = directory / label
    if (
        path.is_symlink() or not path.is_file()
        or path.resolve(strict=True).parent != directory.resolve()
    ):
        raise KrennStarALSArtifactError(
            f"run file is absent, linked, or escaped: {label}"
        )
    return path


def _numbers_close(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return bool(np.isclose(
            left, right, rtol=5e-13, atol=5e-14
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


def _complex_weights(rows, label: str) -> np.ndarray:
    if (
        type(rows) is not list or len(rows) != VARIABLES
        or any(
            type(row) is not list or len(row) != 2
            or type(row[0]) not in (int, float)
            or type(row[1]) not in (int, float)
            or not math.isfinite(float(row[0]))
            or not math.isfinite(float(row[1]))
            for row in rows
        )
    ):
        raise KrennStarALSArtifactError(
            f"{label} is not a finite 135-vector"
        )
    return np.asarray(
        [complex(float(real), float(imag)) for real, imag in rows],
        dtype=np.complex128,
    )


def _current_source_hashes() -> dict[str, str]:
    return {
        Path(record["path"]).name: record["sha256"]
        for record in (
            _source_record(label) for label in NUMERICAL_SOURCE_FILES
        )
    }


def _current_star_hashes() -> list[dict]:
    result = []
    for apex in range(N):
        primary = star_linearization(apex).fingerprint()
        independent = independent_star_linearization_audit(apex)[
            "fingerprint"
        ]
        if primary != independent:
            raise KrennStarALSArtifactError(
                "primary and independent star fingerprints disagree"
            )
        result.append({
            "apex": apex,
            "primary_sha256": primary,
            "independent_sha256": independent,
        })
    return result


def _growth(weights: np.ndarray) -> dict:
    logs = []
    tiny = np.finfo(float).tiny
    for left, right in canonical_edges(N):
        for left_a in range(3):
            for left_b in range(left_a + 1, 3):
                for right_a in range(3):
                    for right_b in range(right_a + 1, 3):
                        diagonal = (
                            weights[variable_index(
                                N, 3, left, right, left_a, right_a
                            )]
                            * weights[variable_index(
                                N, 3, left, right, left_b, right_b
                            )]
                        )
                        off_diagonal = (
                            weights[variable_index(
                                N, 3, left, right, left_a, right_b
                            )]
                            * weights[variable_index(
                                N, 3, left, right, left_b, right_a
                            )]
                        )
                        if abs(diagonal) > tiny and abs(off_diagonal) > tiny:
                            logs.append(float(np.log(
                                abs(diagonal / off_diagonal)
                            )))
    pole = np.prod(
        weights[np.asarray(POLE_INVARIANT_INDICES, dtype=np.int64)]
    )
    return {
        "defined_edge_color_cross_ratios": len(logs),
        "maximum_absolute_log_cross_ratio": (
            max(map(abs, logs)) if logs else None
        ),
        "known_vertex_scalar_pole_product_abs": float(abs(pole)),
    }


def _bounds(weights: np.ndarray, config: StarALSConfig, label: str) -> dict:
    anchors = np.asarray(
        gauge_chart_audit()["anchor_indices"], dtype=np.int64
    )
    if len(anchors) != GAUGE_DIMENSION or not np.all(
        weights[anchors] == GAUGE_ANCHOR_VALUE
    ):
        raise KrennStarALSArtifactError(
            f"{label} left the fixed 15-anchor gauge chart"
        )
    maximum = float(np.abs(weights).max())
    l2 = float(np.linalg.norm(weights))
    if (
        maximum > config.global_linf_cap
        or l2 > config.effective_global_l2_cap
    ):
        raise KrennStarALSArtifactError(
            f"{label} violates a persistent global cap"
        )
    return {
        "maximum_abs": maximum,
        "l2": l2,
        "linf_fraction_of_cap": maximum / config.global_linf_cap,
        "l2_fraction_of_cap": l2 / config.effective_global_l2_cap,
        "finite": True,
        "all_15_anchors_exactly_one": True,
        "within_both_global_caps": True,
    }


def _validate_residual(payload: Mapping, label: str) -> None:
    residual_keys = {
        "l2", "maximum", "mixed_l2", "pure_residual_abs",
        "above_1e-6", "above_1e-10", "all_729_equations_evaluated",
    }
    agreement_keys = {
        "maximum_output_difference", "allowed_output_difference",
        "all_729_outputs_agree",
    }
    if (
        type(payload) is not dict
        or set(payload) != {"primary", "independent", "agreement"}
    ):
        raise KrennStarALSArtifactError(
            f"{label} residual schema changed"
        )
    for enumerator in ("primary", "independent"):
        row = payload[enumerator]
        if (
            type(row) is not dict or set(row) != residual_keys
            or row["all_729_equations_evaluated"] is not True
            or type(row["pure_residual_abs"]) is not list
            or len(row["pure_residual_abs"]) != 3
        ):
            raise KrennStarALSArtifactError(
                f"{label} residual schema changed"
            )
    if (
        type(payload["agreement"]) is not dict
        or set(payload["agreement"]) != agreement_keys
        or payload["agreement"]["all_729_outputs_agree"] is not True
    ):
        raise KrennStarALSArtifactError(
            f"{label} agreement schema changed"
        )


def _replay(rows, recorded, config, label):
    weights = _complex_weights(rows, label)
    bounds = _bounds(weights, config, label)
    _validate_residual(recorded, label)
    try:
        replay = dual_residual_accounting(weights)
    except KrennStarALSError as error:
        raise KrennStarALSArtifactError(
            f"{label} failed dual 729-equation replay"
        ) from error
    if not _numbers_close(recorded, replay):
        raise KrennStarALSArtifactError(
            f"{label} residual accounting failed replay"
        )
    return replay, bounds, _growth(weights)


def _checkpoint_name(result_name: str) -> str:
    if not result_name.endswith(".result.json"):
        raise KrennStarALSArtifactError(
            "result has no deterministic checkpoint peer"
        )
    return result_name[:-12] + ".checkpoint.json"


def _expected_cursor(updates: int) -> dict:
    return {
        "next_sweep": updates // N + 1,
        "next_position": updates % N,
        "completed_sweeps": updates // N,
        "completed_apex_updates": updates,
    }


def _validate_run(directory, row, config, campaign_root):
    if type(row) is not dict or set(row) != _GROUP_ROW_KEYS:
        raise KrennStarALSArtifactError(
            "group result-row schema changed"
        )
    result_path = _safe_local_file(directory, row["result_path"])
    checkpoint_path = _safe_local_file(
        directory, _checkpoint_name(row["result_path"])
    )
    result = _load_json(
        result_path, MAX_RUN_FILE_BYTES, "star ALS result"
    )
    checkpoint = _load_json(
        checkpoint_path, MAX_RUN_FILE_BYTES, "star ALS checkpoint"
    )
    if set(result) != _RESULT_KEYS or result["schema"] != STAR_ALS_SCHEMA:
        raise KrennStarALSArtifactError("result schema changed")
    try:
        rebuilt = StarALSConfig(**result["config"])
    except (TypeError, KrennStarALSError) as error:
        raise KrennStarALSArtifactError(
            "result config failed strict construction"
        ) from error
    if (
        rebuilt != config
        or result["effective_bounds"] != config.effective_bounds()
        or result["config_sha256"] != config.fingerprint()
        or result["source_sha256"] != _current_source_hashes()
        or result["star_sha256"] != _current_star_hashes()
        or result["gauge_chart"] != gauge_chart_audit()
        or result["claim_boundary"] != _RESULT_BOUNDARY
        or result["exact_verification"] != {
            "attempted": False,
            "status": "not-applicable-to-floating-point-trajectory",
        }
    ):
        raise KrennStarALSArtifactError(
            "result config, fingerprint, or claim boundary changed"
        )
    updates = result["completed_apex_updates"]
    if (
        type(updates) is not int or not 0 <= updates <= N * config.maximum_sweeps
        or result["completed_sweeps"] != updates // N
        or result["cursor"] != _expected_cursor(updates)
        or result["termination"] not in {
            "maximum-sweeps", "numerical-tolerance", "time-budget",
            "patience-exhausted",
        }
        or type(result["trace"]) is not list
        or len(result["trace"]) != updates
        or type(result["patience_count"]) is not int
        or not 0 <= result["patience_count"] <= config.patience_apex_updates
        or not math.isfinite(float(result["cumulative_elapsed_seconds"]))
        or result["cumulative_elapsed_seconds"] < 0
    ):
        raise KrennStarALSArtifactError(
            "result cursor or termination is malformed"
        )
    final_replay, final_bounds, final_growth = _replay(
        result["weights"], result["final_residual"], config,
        "final weights"
    )
    best_replay, best_bounds, best_growth = _replay(
        result["best_weights"], result["best_residual"], config,
        "best weights"
    )
    if (
        best_replay["primary"]["l2"] > final_replay["primary"]["l2"] + 5e-14
        or not _numbers_close(
            result["gauge_invariant_growth"], final_growth
        )
        or result["independent_verification"] != {
            "all_729_equations_replayed": True,
            "primary_and_independent_outputs_agree": True,
            "maximum_output_difference": final_replay["agreement"][
                "maximum_output_difference"
            ],
        }
    ):
        raise KrennStarALSArtifactError(
            "best/final replay diagnostics are inconsistent"
        )
    finite = result["finite_bound"]
    if (
        type(finite) is not dict
        or set(finite) != {
            "global_linf_cap", "global_l2_cap", "final_weights",
            "best_weights", "all_updates_within_both_global_bounds",
        }
        or finite["all_updates_within_both_global_bounds"] is not True
        or finite["global_linf_cap"] != config.global_linf_cap
        or finite["global_l2_cap"] != config.effective_global_l2_cap
        or not _numbers_close(
            finite["final_weights"]["maximum_abs"],
            final_bounds["maximum_abs"],
        )
        or not _numbers_close(
            finite["final_weights"]["l2"], final_bounds["l2"]
        )
        or not _numbers_close(
            finite["best_weights"]["maximum_abs"],
            best_bounds["maximum_abs"],
        )
        or not _numbers_close(
            finite["best_weights"]["l2"], best_bounds["l2"]
        )
        or any(
            finite[which].get("within_both_global_caps") is not True
            or finite[which].get(
                "all_15_gauge_anchors_exactly_one"
            ) is not True
            for which in ("final_weights", "best_weights")
        )
    ):
        raise KrennStarALSArtifactError(
            "finite-bound accounting failed replay"
        )
    if (
        set(checkpoint) != _CHECKPOINT_KEYS
        or checkpoint["schema"] != STAR_ALS_CHECKPOINT_SCHEMA
        or checkpoint["config"] != result["config"]
        or checkpoint["effective_bounds"] != result["effective_bounds"]
        or checkpoint["config_sha256"] != result["config_sha256"]
        or checkpoint["source_sha256"] != result["source_sha256"]
        or checkpoint["star_sha256"] != result["star_sha256"]
        or checkpoint["gauge_chart"] != result["gauge_chart"]
        or checkpoint["claim_boundary"] != _CHECKPOINT_BOUNDARY
        or type(checkpoint["state"]) is not dict
        or set(checkpoint["state"]) != _STATE_KEYS
    ):
        raise KrennStarALSArtifactError(
            "terminal checkpoint schema or fingerprint changed"
        )
    state = checkpoint["state"]
    overlap = {
        "status": result["termination"],
        "cursor": result["cursor"],
        "weights": result["weights"],
        "best_weights": result["best_weights"],
        "initial_residual": result["initial_residual"],
        "current_residual": result["final_residual"],
        "best_residual": result["best_residual"],
        "trace": result["trace"],
        "patience_count": result["patience_count"],
        "cumulative_elapsed_seconds": result[
            "cumulative_elapsed_seconds"
        ],
    }
    if any(state[key] != value for key, value in overlap.items()):
        raise KrennStarALSArtifactError(
            "result and terminal checkpoint disagree"
        )
    _initial_replay, initial_bounds, _initial_growth = _replay(
        state["initial_weights"], state["initial_residual"], config,
        "initial weights"
    )
    expected_row = {
        "radius": config.radius,
        "effective_bounds": config.effective_bounds(),
        "seed": config.seed,
        "result_path": result_path.name,
        "completed_sweeps": result["completed_sweeps"],
        "completed_apex_updates": updates,
        "termination": result["termination"],
        "elapsed_seconds": result["cumulative_elapsed_seconds"],
        "final_residual_l2": final_replay["primary"]["l2"],
        "independent_final_residual_l2": final_replay[
            "independent"
        ]["l2"],
        "maximum_weight_abs": final_bounds["maximum_abs"],
        "weight_l2": final_bounds["l2"],
    }
    if not _numbers_close(row, expected_row):
        raise KrennStarALSArtifactError(
            "group manifest row failed result replay"
        )
    return {
        "seed": config.seed,
        "termination": result["termination"],
        "elapsed_seconds": result["cumulative_elapsed_seconds"],
        "completed_sweeps": result["completed_sweeps"],
        "completed_apex_updates": updates,
        "final_residual_l2": final_replay["primary"]["l2"],
        "final_residual_maximum": final_replay["primary"]["maximum"],
        "best_residual_l2": best_replay["primary"]["l2"],
        "best_residual_maximum": best_replay["primary"]["maximum"],
        "initial_bounds": initial_bounds,
        "final_bounds": final_bounds,
        "best_bounds": best_bounds,
        "final_growth": final_growth,
        "best_growth": best_growth,
        "numerical_tolerance_reached": (
            result["termination"] == "numerical-tolerance"
        ),
        "exact_verification": result["exact_verification"],
        "files": [
            _file_record(result_path, campaign_root),
            _file_record(checkpoint_path, campaign_root),
        ],
    }


def _command(directory, group, linf, l2, interpreter):
    arguments = [
        interpreter, "-B", "-m",
        "experiments.krenn_quantum_graph.star_alternating_search",
        "--output-directory", str(directory),
        "--radii", format(linf, ".17g"),
        "--global-l2-caps", format(l2, ".17g"),
        "--seeds", ",".join(map(str, DEFAULT_SEEDS)),
        "--maximum-sweeps", str(group["maximum_sweeps_per_run"]),
        "--maximum-seconds-per-run",
        format(group["maximum_seconds_per_run"], ".17g"),
        "--patience-apex-updates",
        str(group["patience_apex_updates"]),
        "--minimum-relative-improvement",
        format(group["minimum_relative_improvement"], ".17g"),
        "--initialization", group["initialization"], "--resume",
    ]
    return subprocess.list2cmdline(arguments)


def _validate_group(manifest_path, campaign_root, interpreter):
    group = _load_json(
        manifest_path, MAX_GROUP_MANIFEST_BYTES,
        "star ALS group manifest"
    )
    directory = manifest_path.parent.resolve(strict=True)
    if (
        set(group) != _GROUP_KEYS
        or group["schema"] != STAR_ALS_CAMPAIGN_SCHEMA
        or group["gauge_chart"] != gauge_chart_audit()
        or group["default_global_l2_cap_rule"]
        != "global_linf_cap*sqrt(135)"
        or group["seeds"] != list(DEFAULT_SEEDS)
        or group["initialization"] not in EXPECTED_INITIALIZATIONS
        or group["workers"] != 1
        or group["randomness"] != "deterministic NumPy PCG64 seeds"
        or group["checkpointing"] != "atomic after every apex update"
        or group["claim_boundary"] != _GROUP_BOUNDARY
        or type(group["radii_as_global_linf_caps"]) is not list
        or len(group["radii_as_global_linf_caps"]) != 1
        or type(group["explicit_global_l2_caps"]) is not list
        or len(group["explicit_global_l2_caps"]) != 1
        or type(group["results"]) is not list
        or len(group["results"]) != len(DEFAULT_SEEDS)
        or Path(group["output_directory"]).resolve() != directory
    ):
        raise KrennStarALSArtifactError(
            "group manifest failed strict validation"
        )
    linf = float(group["radii_as_global_linf_caps"][0])
    l2 = float(group["explicit_global_l2_caps"][0])
    runs = []
    seen = set()
    for row in group["results"]:
        seed = row.get("seed") if type(row) is dict else None
        if seed in seen or seed not in DEFAULT_SEEDS:
            raise KrennStarALSArtifactError(
                "group seed is duplicate or unexpected"
            )
        seen.add(seed)
        try:
            config = StarALSConfig(
                radius=linf,
                global_l2_radius=l2,
                maximum_sweeps=group["maximum_sweeps_per_run"],
                seed=seed,
                initialization=group["initialization"],
                maximum_seconds=group["maximum_seconds_per_run"],
                patience_apex_updates=group["patience_apex_updates"],
                minimum_relative_improvement=group[
                    "minimum_relative_improvement"
                ],
            )
        except (TypeError, KrennStarALSError) as error:
            raise KrennStarALSArtifactError(
                "group config is invalid"
            ) from error
        runs.append(
            _validate_run(directory, row, config, campaign_root)
        )
    if seen != set(DEFAULT_SEEDS):
        raise KrennStarALSArtifactError(
            "group does not contain all fixed seeds"
        )
    if group["best_bounded_candidate"] != min(
        group["results"], key=lambda row: row["final_residual_l2"]
    ):
        raise KrennStarALSArtifactError(
            "group best-candidate row is inconsistent"
        )
    best = min(runs, key=lambda row: row["best_residual_l2"])
    relative = str(
        manifest_path.resolve().relative_to(campaign_root)
    ).replace("\\", "/")
    return {
        "manifest_path": relative,
        "manifest_file": _file_record(manifest_path, campaign_root),
        "initialization": group["initialization"],
        "global_linf_cap": linf,
        "global_l2_cap": l2,
        "seeds": list(DEFAULT_SEEDS),
        "run_count": len(runs),
        "budget": {
            "maximum_sweeps_per_run": group[
                "maximum_sweeps_per_run"
            ],
            "maximum_seconds_per_run": group[
                "maximum_seconds_per_run"
            ],
            "patience_apex_updates": group["patience_apex_updates"],
            "minimum_relative_improvement": group[
                "minimum_relative_improvement"
            ],
        },
        "workers_recorded_by_group": 1,
        "canonical_replay_command": _command(
            directory, group, linf, l2, interpreter
        ),
        "runs": runs,
        "best_bounded_candidate": {
            key: best[key] for key in (
                "seed", "termination", "elapsed_seconds",
                "best_residual_l2", "best_residual_maximum",
                "best_bounds", "best_growth",
                "numerical_tolerance_reached", "exact_verification",
            )
        },
    }


def summarize_star_als_campaign(
    campaign_root: Path,
    *,
    parallel_group_workers: int = 8,
    interpreter: str = "python",
) -> dict:
    """Replay exactly eight group manifests and 24 bounded runs."""

    campaign_root = Path(campaign_root)
    if (
        campaign_root.is_symlink() or not campaign_root.is_dir()
        or type(parallel_group_workers) is not int
        or not 1 <= parallel_group_workers <= MAX_DECLARED_WORKERS
        or type(interpreter) is not str or not interpreter
    ):
        raise KrennStarALSArtifactError(
            "campaign root or resource declaration is invalid"
        )
    campaign_root = campaign_root.resolve(strict=True)
    manifests = sorted(
        path for path in campaign_root.rglob(MANIFEST_FILE)
        if path.parent != campaign_root
    )
    if (
        len(manifests) != EXPECTED_GROUPS
        or len({path.parent.resolve() for path in manifests})
        != EXPECTED_GROUPS
    ):
        raise KrennStarALSArtifactError(
            "campaign root must contain exactly eight group manifests"
        )
    groups = [
        _validate_group(path, campaign_root, interpreter)
        for path in manifests
    ]
    keys = {
        (
            group["initialization"], group["global_linf_cap"],
            group["global_l2_cap"],
        )
        for group in groups
    }
    caps = {
        initialization: {
            (group["global_linf_cap"], group["global_l2_cap"])
            for group in groups
            if group["initialization"] == initialization
        }
        for initialization in EXPECTED_INITIALIZATIONS
    }
    if (
        len(keys) != EXPECTED_GROUPS
        or any(len(rows) != EXPECTED_CAP_PAIRS for rows in caps.values())
        or caps[EXPECTED_INITIALIZATIONS[0]]
        != caps[EXPECTED_INITIALIZATIONS[1]]
        or len({
            _canonical_json_bytes(group["budget"]) for group in groups
        }) != 1
    ):
        raise KrennStarALSArtifactError(
            "group caps, initializations, or budgets disagree"
        )
    all_runs = [
        {
            **run,
            "initialization": group["initialization"],
            "global_linf_cap": group["global_linf_cap"],
            "global_l2_cap": group["global_l2_cap"],
        }
        for group in groups for run in group["runs"]
    ]
    if len(all_runs) != EXPECTED_RUNS:
        raise KrennStarALSArtifactError(
            "campaign does not contain exactly 24 runs"
        )
    inputs = []
    for group in groups:
        inputs.append(group["manifest_file"])
        for run in group["runs"]:
            inputs.extend(run["files"])
    if len(inputs) != 56 or len({
        row["path"] for row in inputs
    }) != 56:
        raise KrennStarALSArtifactError(
            "input inventory is not 8 manifests plus 48 run files"
        )
    groups = sorted(
        groups,
        key=lambda row: (
            row["initialization"], row["global_linf_cap"],
            row["global_l2_cap"],
        ),
    )
    public_groups = [{
        key: group[key] for key in (
            "manifest_path", "initialization", "global_linf_cap",
            "global_l2_cap", "seeds", "run_count", "budget",
            "workers_recorded_by_group", "canonical_replay_command",
            "best_bounded_candidate",
        )
    } for group in groups]
    best = min(all_runs, key=lambda row: row["best_residual_l2"])
    terminations = Counter(row["termination"] for row in all_runs)
    exact_statuses = Counter(
        row["exact_verification"]["status"] for row in all_runs
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "problem": {
            "n": N, "d": 3, "variables": VARIABLES,
            "equations_replayed_per_vector": EQUATIONS,
            "target": "GHZ_6,3",
        },
        "campaign": {
            "scratch_root_at_generation": str(campaign_root),
            "initializations": list(EXPECTED_INITIALIZATIONS),
            "cap_pairs": [
                {"global_linf_cap": linf, "global_l2_cap": l2}
                for linf, l2 in sorted(caps[EXPECTED_INITIALIZATIONS[0]])
            ],
            "fixed_seeds": list(DEFAULT_SEEDS),
            "groups": EXPECTED_GROUPS,
            "runs": EXPECTED_RUNS,
            "common_budget": groups[0]["budget"],
            "resources": {
                "workers_per_group_manifest": 1,
                "parallel_group_workers_declared": parallel_group_workers,
                "declared_worker_limit": MAX_DECLARED_WORKERS,
                "parallel_worker_count_is_caller_declared_not_"
                "machine_audited": True,
            },
            "commands": {
                "status": (
                    "canonical source-controlled replay commands; group "
                    "manifests do not attest original shell invocations"
                ),
                "interpreter": interpreter,
                "by_group": [
                    {
                        "manifest_path": row["manifest_path"],
                        "command": row["canonical_replay_command"],
                    }
                    for row in public_groups
                ],
            },
        },
        "gauge_chart": {
            "schema": gauge_chart_audit()["schema"],
            "anchor_indices": gauge_chart_audit()["anchor_indices"],
            "exact_rank_over_Q": gauge_chart_audit()["exact_rank_over_Q"],
            "anchor_minor_determinant": gauge_chart_audit()[
                "anchor_minor_determinant"
            ],
            "scope": "one fixed 15-anchor open chart",
        },
        "groups": public_groups,
        "aggregate": {
            "best_bounded_candidate": {
                key: best[key] for key in (
                    "initialization", "global_linf_cap", "global_l2_cap",
                    "seed", "termination", "elapsed_seconds",
                    "best_residual_l2", "best_residual_maximum",
                    "best_bounds", "best_growth",
                    "numerical_tolerance_reached", "exact_verification",
                )
            },
            "termination_counts": dict(sorted(terminations.items())),
            "numerical_tolerance_runs": sum(
                row["numerical_tolerance_reached"] for row in all_runs
            ),
            "all_recorded_weights_finite_and_bounded": all(
                row[which]["finite"]
                and row[which]["within_both_global_caps"]
                for row in all_runs
                for which in ("initial_bounds", "final_bounds", "best_bounds")
            ),
            "maximum_final_linf_fraction_of_cap": max(
                row["final_bounds"]["linf_fraction_of_cap"]
                for row in all_runs
            ),
            "maximum_final_l2_fraction_of_cap": max(
                row["final_bounds"]["l2_fraction_of_cap"]
                for row in all_runs
            ),
            "growth_diagnostics": {
                "maximum_final_absolute_log_cross_ratio": max((
                    row["final_growth"][
                        "maximum_absolute_log_cross_ratio"
                    ]
                    for row in all_runs
                    if row["final_growth"][
                        "maximum_absolute_log_cross_ratio"
                    ] is not None
                ), default=None),
                "maximum_best_absolute_log_cross_ratio": max((
                    row["best_growth"][
                        "maximum_absolute_log_cross_ratio"
                    ]
                    for row in all_runs
                    if row["best_growth"][
                        "maximum_absolute_log_cross_ratio"
                    ] is not None
                ), default=None),
                "maximum_final_known_pole_product_abs": max(
                    row["final_growth"][
                        "known_vertex_scalar_pole_product_abs"
                    ]
                    for row in all_runs
                ),
                "maximum_best_known_pole_product_abs": max(
                    row["best_growth"][
                        "known_vertex_scalar_pole_product_abs"
                    ]
                    for row in all_runs
                ),
            },
            "exact_verification": {
                "status_counts": dict(sorted(exact_statuses.items())),
                "attempted_runs": sum(
                    row["exact_verification"]["attempted"]
                    for row in all_runs
                ),
                "finite_exact_witnesses_verified": 0,
            },
        },
        "input_file_ledger": sorted(inputs, key=lambda row: row["path"]),
        "source_ledger": [
            _source_record(label) for label in NUMERICAL_SOURCE_FILES
        ],
        "verification": {
            "result_vectors_replayed": EXPECTED_RUNS * 2,
            "checkpoint_initial_vectors_replayed": EXPECTED_RUNS,
            "equations_per_vector": EQUATIONS,
            "primary_and_independent_enumerators_used": True,
            "terminal_result_checkpoint_pairs_cross_checked": EXPECTED_RUNS,
        },
        "claim_boundary": dict(_SUMMARY_BOUNDARY),
    }


def _artifact_record(path: Path) -> dict:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def generate_star_als_summary_bundle(
    campaign_root: Path,
    output_directory: Path,
    *,
    parallel_group_workers: int = 8,
    interpreter: str = "python",
) -> Path:
    summary = summarize_star_als_campaign(
        campaign_root,
        parallel_group_workers=parallel_group_workers,
        interpreter=interpreter,
    )
    output_directory = Path(output_directory)
    if output_directory.exists() and (
        output_directory.is_symlink() or not output_directory.is_dir()
    ):
        raise KrennStarALSArtifactError(
            "summary output must be a real directory"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(
        path.name not in ALL_FILES for path in output_directory.iterdir()
    ):
        raise KrennStarALSArtifactError(
            "summary output contains unreviewed files"
        )
    summary_path = output_directory / SUMMARY_FILE
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(output_directory / MANIFEST_FILE, {
        "schema": MANIFEST_SCHEMA,
        "artifact": _artifact_record(summary_path),
        "producer": _source_record(PRODUCER_FILE),
        "numerical_sources": [
            _source_record(label) for label in NUMERICAL_SOURCE_FILES
        ],
        "campaign_input_file_count": len(summary["input_file_ledger"]),
        "claim_boundary": dict(_SUMMARY_BOUNDARY),
    })
    return summary_path


def verify_star_als_summary_bundle(
    directory: Path,
    *,
    campaign_root: Path | None = None,
) -> VerifiedStarALSArtifact:
    directory = Path(directory)
    if (
        directory.is_symlink() or not directory.is_dir()
        or {path.name for path in directory.iterdir()} != set(ALL_FILES)
    ):
        raise KrennStarALSArtifactError(
            "summary bundle inventory changed"
        )
    manifest = _load_json(
        directory / MANIFEST_FILE, MAX_SUMMARY_BYTES,
        "star ALS summary manifest"
    )
    summary = _load_json(
        directory / SUMMARY_FILE, MAX_SUMMARY_BYTES,
        "star ALS summary"
    )
    if (
        set(manifest) != {
            "schema", "artifact", "producer", "numerical_sources",
            "campaign_input_file_count", "claim_boundary",
        }
        or manifest["schema"] != MANIFEST_SCHEMA
        or manifest["claim_boundary"] != _SUMMARY_BOUNDARY
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("claim_boundary") != _SUMMARY_BOUNDARY
    ):
        raise KrennStarALSArtifactError(
            "summary schema or claim boundary changed"
        )
    artifact = manifest["artifact"]
    summary_path = directory / SUMMARY_FILE
    if (
        type(artifact) is not dict
        or set(artifact) != {"path", "bytes", "sha256"}
        or artifact["path"] != SUMMARY_FILE
        or artifact["bytes"] != summary_path.stat().st_size
        or artifact["sha256"] != _sha256(summary_path)
    ):
        raise KrennStarALSArtifactError(
            "summary artifact hash replay failed"
        )
    sources = [
        _source_record(label) for label in NUMERICAL_SOURCE_FILES
    ]
    if (
        manifest["producer"] != _source_record(PRODUCER_FILE)
        or manifest["numerical_sources"] != sources
        or summary.get("source_ledger") != sources
        or manifest["campaign_input_file_count"]
        != len(summary.get("input_file_ledger", []))
    ):
        raise KrennStarALSArtifactError(
            "summary source or input ledger changed"
        )
    root = (
        Path(summary["campaign"]["scratch_root_at_generation"])
        if campaign_root is None else Path(campaign_root)
    )
    replay = summarize_star_als_campaign(
        root,
        parallel_group_workers=summary["campaign"]["resources"][
            "parallel_group_workers_declared"
        ],
        interpreter=summary["campaign"]["commands"]["interpreter"],
    )
    if _canonical_json_bytes(replay) != _canonical_json_bytes(summary):
        raise KrennStarALSArtifactError(
            "summary failed full campaign semantic replay"
        )
    loaded = VerifiedStarALSArtifact(
        directory=directory.resolve(),
        campaign_root=root.resolve(),
        summary=summary,
        manifest=manifest,
        checks=(
            ("bundle_inventory_exact", True),
            ("summary_hash_replayed", True),
            ("source_hashes_replayed", True),
            ("eight_group_manifests_replayed", True),
            ("twenty_four_results_replayed", True),
            ("twenty_four_checkpoints_cross_checked", True),
            ("all_final_best_initial_vectors_replayed", True),
            ("fixed_chart_and_bounds_replayed", True),
            ("claim_boundaries_fail_closed", True),
        ),
    )
    if not loaded.valid:
        raise KrennStarALSArtifactError("summary checks failed")
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate/verify the small bounded star-ALS summary."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--parallel-group-workers", type=int, default=8)
    parser.add_argument("--interpreter", default="python")
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        generate_star_als_summary_bundle(
            arguments.campaign_root,
            arguments.output_directory,
            parallel_group_workers=arguments.parallel_group_workers,
            interpreter=arguments.interpreter,
        )
    loaded = verify_star_als_summary_bundle(
        arguments.output_directory,
        campaign_root=arguments.campaign_root,
    )
    print(json.dumps({
        "valid": loaded.valid,
        "groups": loaded.summary["campaign"]["groups"],
        "runs": loaded.summary["campaign"]["runs"],
        "best_bounded_candidate": loaded.summary["aggregate"][
            "best_bounded_candidate"
        ],
        "all_recorded_weights_finite_and_bounded": loaded.summary[
            "aggregate"
        ]["all_recorded_weights_finite_and_bounded"],
        "finite_exact_witnesses_verified": loaded.summary["aggregate"][
            "exact_verification"
        ]["finite_exact_witnesses_verified"],
        "finite_affine_membership_status": loaded.summary[
            "claim_boundary"
        ]["finite_affine_membership_status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
