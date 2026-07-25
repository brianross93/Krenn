"""Provenance-checked continuation of one retained Krenn VP candidate.

This runner deliberately extends only a selected numerical basin.  It
starts from the retained best ``U`` vector, discards stale quasi-Newton
history, replays the current core/source fingerprints, and writes a new
checkpoint/result/manifest without upgrading any mathematical claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

from experiments.krenn_quantum_graph import (
    star_variable_projection_campaign as campaign,
)


CONTINUATION_SCHEMA = "krenn-n6-d3-vp-continuation-v1"


class KrennNumericalContinuationError(RuntimeError):
    """A retained result or continuation target failed validation."""


def _reject_constant(value: str) -> None:
    raise KrennNumericalContinuationError(
        f"nonfinite JSON constant {value!r} is forbidden"
    )


def _read_json(path: Path) -> dict:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise KrennNumericalContinuationError(
            "the parent result must be a regular file"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KrennNumericalContinuationError(
            "the parent result is not strict ASCII JSON"
        ) from error
    if not isinstance(payload, dict):
        raise KrennNumericalContinuationError(
            "the parent result must contain a JSON object"
        )
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _output_directory(path: Path) -> Path:
    resolved = Path(path).resolve()
    if "onedrive" in (piece.lower() for piece in resolved.parts):
        raise KrennNumericalContinuationError(
            "continuation output cannot be inside OneDrive"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise KrennNumericalContinuationError(
            "continuation output must be a real directory"
        )
    return resolved


def _warm_config(
    parent: dict,
    *,
    maximum_iterations: int,
    maximum_seconds: float,
    patience: int,
    checkpoint_interval: int,
) -> campaign.TrajectoryConfig:
    raw = parent.get("config")
    if not isinstance(raw, dict) or not isinstance(
        raw.get("caps"), dict
    ):
        raise KrennNumericalContinuationError(
            "the parent trajectory config is malformed"
        )
    caps = campaign.RadiusCaps(**raw["caps"])
    return campaign.TrajectoryConfig(
        orbit_index=raw["orbit_index"],
        seed=raw["seed"],
        caps=caps,
        initialization="warm",
        maximum_iterations=maximum_iterations,
        maximum_seconds=maximum_seconds,
        patience=patience,
        checkpoint_interval=checkpoint_interval,
        residual_tolerance=raw.get("residual_tolerance", 1.0e-12),
        gradient_tolerance=raw.get("gradient_tolerance", 1.0e-12),
        minimum_relative_improvement=raw.get(
            "minimum_relative_improvement", 1.0e-14
        ),
        lbfgs_memory=raw.get("lbfgs_memory", 10),
        armijo_constant=raw.get("armijo_constant", 1.0e-4),
        backtrack_factor=raw.get("backtrack_factor", 0.5),
        maximum_backtracks=raw.get("maximum_backtracks", 60),
        minimum_pivot_abs=raw.get("minimum_pivot_abs", 1.0e-12),
        rank_rtol=raw.get("rank_rtol", 1.0e-12),
        natural_repair_scale=raw.get(
            "natural_repair_scale", 1.0e-2
        ),
    )


def _validate_parent_candidate(
    parent: dict,
    *,
    core: Any,
    warm_u: np.ndarray,
) -> dict:
    """Replay the retained point and validate its config and self hash."""

    raw_config = parent.get("config")
    if not isinstance(raw_config, dict):
        raise KrennNumericalContinuationError(
            "the parent trajectory config is malformed"
        )
    if parent.get("config_sha256") != campaign._json_sha256(raw_config):
        raise KrennNumericalContinuationError(
            "the parent config hash changed"
        )
    supplied_hash = parent.get("payload_sha256")
    legacy_without_self_hash = supplied_hash is None
    if supplied_hash is not None:
        unsigned = dict(parent)
        unsigned.pop("payload_sha256", None)
        if supplied_hash != campaign._json_sha256(unsigned):
            raise KrennNumericalContinuationError(
                "the parent result payload hash changed"
            )
    try:
        representative = campaign._representative(
            core, int(raw_config["orbit_index"])
        )
        caps = campaign.RadiusCaps(**raw_config["caps"])
        evaluation = campaign.evaluate_projection(
            core,
            warm_u,
            representative,
            caps,
            rank_rtol=float(raw_config.get("rank_rtol", 1.0e-12)),
        )
        saved_objective = float(
            parent["best_evaluation"]["objective"]
        )
        saved_y = campaign._complex_array(
            parent["best_bounded_y"],
            shape=(campaign.Y_ROWS, campaign.TARGET_COLUMNS),
            label="parent best_bounded_y",
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        campaign.KrennVariableProjectionCampaignError,
    ) as error:
        raise KrennNumericalContinuationError(
            "the parent candidate cannot be replayed"
        ) from error
    if (
        not math.isfinite(saved_objective)
        or not math.isclose(
            evaluation.objective,
            saved_objective,
            rel_tol=5.0e-10,
            abs_tol=5.0e-12,
        )
    ):
        raise KrennNumericalContinuationError(
            "the parent best U does not reproduce its saved objective"
        )
    if not np.allclose(
        evaluation.y,
        saved_y,
        rtol=5.0e-10,
        atol=5.0e-12,
    ):
        raise KrennNumericalContinuationError(
            "the parent best U does not reproduce its bounded Y"
        )
    try:
        replay = core.dual_full_residual(warm_u, evaluation.y)
        cross_check = campaign._validate_full_replay(
            replay, evaluation
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        campaign.KrennVariableProjectionCampaignError,
    ) as error:
        raise KrennNumericalContinuationError(
            "the parent full-system replay failed"
        ) from error
    return {
        "config_sha256_verified": True,
        "payload_sha256_verified": supplied_hash is not None,
        "legacy_parent_without_self_hash":
            legacy_without_self_hash,
        "best_u_objective_recomputed": evaluation.objective,
        "best_bounded_y_recomputed": True,
        "full_system_replay": cross_check,
    }


def continue_result(
    *,
    parent_result_path: Path,
    output_directory: Path,
    label: str,
    maximum_iterations: int = 5_000,
    maximum_seconds: float = 600.0,
    patience: int = 5_000,
    checkpoint_interval: int = 50,
    core: Any | None = None,
) -> dict:
    """Continue one exact parent payload and emit a fail-closed manifest."""

    if not label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in label.lower()
    ):
        raise KrennNumericalContinuationError(
            "label must use only letters, digits, underscore, or hyphen"
        )
    parent_path = Path(parent_result_path).resolve()
    parent = _read_json(parent_path)
    if parent.get("schema") != campaign.RESULT_SCHEMA:
        raise KrennNumericalContinuationError(
            "the parent is not a variable-projection result"
        )
    core = (
        campaign._load_core(campaign.DEFAULT_CORE_MODULE)
        if core is None
        else core
    )
    current_fingerprints = campaign._core_fingerprints(core)
    if parent.get("source_fingerprints") != current_fingerprints:
        raise KrennNumericalContinuationError(
            "the parent source fingerprints do not match current code"
        )
    try:
        raw_u = np.asarray(parent["best_u"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise KrennNumericalContinuationError(
            "the parent best U encoding is malformed"
        ) from error
    if raw_u.shape != (campaign.U_VARIABLES, 2):
        raise KrennNumericalContinuationError(
            "the parent best U encoding has the wrong shape"
        )
    warm_u = raw_u[:, 0] + 1j * raw_u[:, 1]
    if not np.all(np.isfinite(warm_u)):
        raise KrennNumericalContinuationError(
            "the parent best U contains nonfinite values"
        )
    parent_validation = _validate_parent_candidate(
        parent,
        core=core,
        warm_u=warm_u,
    )
    config = _warm_config(
        parent,
        maximum_iterations=maximum_iterations,
        maximum_seconds=maximum_seconds,
        patience=patience,
        checkpoint_interval=checkpoint_interval,
    )
    output = _output_directory(output_directory)
    checkpoint_path = output / f"{label}.checkpoint.json"
    result_path = output / f"{label}.result.json"
    manifest_path = output / f"{label}.manifest.json"
    if any(
        path.exists()
        for path in (checkpoint_path, result_path, manifest_path)
    ):
        raise KrennNumericalContinuationError(
            "a continuation output already exists"
        )
    result = campaign.run_trajectory(
        config,
        checkpoint_path=checkpoint_path,
        core=core,
        warm_start=warm_u,
    )
    campaign._write_json_atomic(result_path, result)
    continuation_source = Path(__file__).resolve()
    manifest_fingerprints = {
        **current_fingerprints,
        "continuation_source": str(continuation_source),
        "continuation_source_sha256": _sha256(continuation_source),
    }
    manifest = {
        "schema": CONTINUATION_SCHEMA,
        "label": label,
        "parent_result": {
            "path": str(parent_path),
            "bytes": parent_path.stat().st_size,
            "sha256": _sha256(parent_path),
            "best_objective": parent["best_evaluation"]["objective"],
            "validation": parent_validation,
        },
        "config": config.to_dict(),
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
        },
        "result": {
            "path": result_path.name,
            "bytes": result_path.stat().st_size,
            "sha256": _sha256(result_path),
            "termination": result["termination"],
            "best_objective": result["best_evaluation"]["objective"],
        },
        "source_fingerprints": manifest_fingerprints,
        "claim_boundary": {
            "numerical_zero_is_exact_witness": False,
            "selected_basin_represents_all_basins": False,
            "continued_descent_is_border_or_membership_proof": False,
            "finite_affine_membership_status": "undecided",
        },
    }
    campaign._write_json_atomic(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Continue one retained Krenn variable-projection basin."
    )
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--maximum-iterations", type=int, default=5_000)
    parser.add_argument("--maximum-seconds", type=float, default=600.0)
    parser.add_argument("--patience", type=int, default=5_000)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    arguments = parser.parse_args(argv)
    manifest = continue_result(
        parent_result_path=arguments.parent_result,
        output_directory=arguments.output_directory,
        label=arguments.label,
        maximum_iterations=arguments.maximum_iterations,
        maximum_seconds=arguments.maximum_seconds,
        patience=arguments.patience,
        checkpoint_interval=arguments.checkpoint_interval,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CONTINUATION_SCHEMA",
    "KrennNumericalContinuationError",
    "continue_result",
]
