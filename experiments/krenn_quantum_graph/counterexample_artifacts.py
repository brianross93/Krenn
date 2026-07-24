"""Round-trippable artifacts for the bounded counterexample campaign.

Committed campaign bundles are intentionally compact: disposable optimizer
matrices and checkpoints remain outside the repository, while selected best
weights retain all 729 floating residuals for deterministic replay.  An exact
number-field witness is admitted only when both exact matching enumerators
certify every equation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence

import numpy as np

from experiments.krenn_quantum_graph.counterexample_exact import (
    ExactVerificationReport,
)
from experiments.krenn_quantum_graph.counterexample_search import (
    MAX_WORKERS,
    SupportCampaignResult,
    support_campaign_from_dict,
)
from experiments.krenn_quantum_graph.numerical_continuation import (
    AMBIENT_VARIABLES,
    ContinuationResult,
    GHZ_COLOR_DIAGONAL_GAUGE,
    GHZ_VICTIM_COLOR_DIAGONAL_GAUGE,
    InvariantMetrics,
    LMPlan,
    NATURAL_SUPPORT_SET,
    NumericalJobSpec,
    NumericalSolveResult,
    ResidualMetrics,
    WeightMetrics,
    complex_residual,
    natural_gauge_chart,
    support_gauge_chart,
    system_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_FILE = "support_search.json"
NUMERICAL_FILE = "numerical_search.json"
CERTIFICATE_FILE = "certificate.json"
MANIFEST_FILE = "manifest.json"
EXACT_WITNESS_FILE = "exact_witness.json"
EXACT_VERIFICATION_FILE = "exact_verification.json"

SUPPORT_SUMMARY_SCHEMA = "krenn-n6-d3-support-campaign-summary-v1"
NUMERICAL_SUMMARY_SCHEMA = "krenn-n6-d3-numerical-campaign-summary-v1"
CAMPAIGN_CERTIFICATE_SCHEMA = (
    "krenn-n6-d3-counterexample-campaign-certificate-v1"
)
CAMPAIGN_MANIFEST_SCHEMA = (
    "krenn-n6-d3-counterexample-campaign-manifest-v1"
)
BASELINE_COMMIT = "ab446724a61a293a3268c63b9e9e721c6a360561"
FEATURE_BRANCH = "codex/counterexample-search"
SCRATCH_ROOT = PureWindowsPath(r"D:\KrennScratch\counterexample_search")

_WORKER_THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_SUPPORT_ORIENTATION_POLICY = {
    "orbit_deduplication_key": (
        "lexicographically-least-S6xS3-support-image"
    ),
    "natural_laurent_stratum": (
        "preferred-natural-containing-orientation"
    ),
    "non_laurent_strata": "literal-canonical-orbit-representative",
    "symmetry_weight_equalities": 0,
}
_EXECUTION_KEYS = frozenset(
    {
        "branch",
        "baseline_origin_main",
        "master_seed",
        "workers",
        "threads_per_worker",
        "worker_thread_environment",
        "argv_receipt",
        "support_budgets",
        "support_orientation_policy",
        "numerical_budgets",
        "scratch_directory",
        "scratch_data_committed",
        "large_checkpoints_required_for_bundle_replay",
    }
)
_SUPPORT_BUDGET_KEYS = frozenset(
    {
        "orbit_node_cap",
        "discovered_support_cap_per_run",
        "retained_support_candidates",
        "retained_candidates_per_support_size",
        "greedy_pruned_omission_supports",
    }
)
_NUMERICAL_BUDGET_KEYS = frozenset(
    {
        "radii",
        "starts_per_sparse_support",
        "dense_starts_per_radius",
        "iterations_per_solve",
        "evaluations_per_solve",
        "continuation_schedule",
        "checkpoint_interval",
        "continue_after_nonconvergence",
        "resume_from_checkpoint",
    }
)

_LM_TERMINATION_STATUSES = frozenset(
    {
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
)
_CONTINUATION_GROWTH_KEYS = frozenset(
    {
        "target_amplitude",
        "residual_linf",
        "weight_l2",
        "weight_linf",
        "known_Q_absolute_value",
        "hard_boundary_active",
    }
)

REQUIRED_SOURCE_PATHS = (
    "experiments/krenn_quantum_graph/counterexample_artifacts.py",
    "experiments/krenn_quantum_graph/counterexample_campaign.py",
    "experiments/krenn_quantum_graph/counterexample_exact.py",
    "experiments/krenn_quantum_graph/counterexample_search.py",
    "experiments/krenn_quantum_graph/formal_lift.py",
    "experiments/krenn_quantum_graph/numerical_continuation.py",
    "experiments/krenn_quantum_graph/support_extension.py",
    "experiments/krenn_quantum_graph/support_search.py",
    "experiments/krenn_quantum_graph/system.py",
)


class KrennCounterexampleArtifactError(RuntimeError):
    """A campaign artifact, hash, inventory, or semantic replay failed."""


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennCounterexampleArtifactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str):
    raise KrennCounterexampleArtifactError(
        f"non-finite JSON constant {value!r} is forbidden"
    )


def _canonical_json_bytes(value: Mapping) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise KrennCounterexampleArtifactError(
            "campaign payload is not canonical finite JSON"
        ) from error


def _load_json(path: Path) -> Mapping:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.parent.resolve() != path.resolve().parent
    ):
        raise KrennCounterexampleArtifactError(
            f"campaign artifact is absent, linked, or unsafe: {path.name}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise KrennCounterexampleArtifactError(
            f"could not decode campaign artifact {path.name}"
        ) from error
    if not isinstance(payload, Mapping):
        raise KrennCounterexampleArtifactError(
            f"campaign artifact {path.name} is not a JSON object"
        )
    if path.read_bytes() != _canonical_json_bytes(payload):
        raise KrennCounterexampleArtifactError(
            f"campaign artifact {path.name} is not canonical JSON"
        )
    return payload


def _write_json_atomic(path: Path, value: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(value))
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict:
    resolved = path.resolve()
    root = relative_to.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise KrennCounterexampleArtifactError(
            "manifest path escapes its declared root"
        ) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise KrennCounterexampleArtifactError(
            "manifest input is absent or linked"
        )
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _complex_pair(value: complex) -> list[float]:
    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise KrennCounterexampleArtifactError(
            "selected numerical values must be finite"
        )
    return [float(value.real), float(value.imag)]


def _complex_from_pair(value) -> complex:
    try:
        real, imaginary = value
        result = complex(float(real), float(imaginary))
    except (TypeError, ValueError) as error:
        raise KrennCounterexampleArtifactError(
            "complex payload must be a real-imaginary pair"
        ) from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise KrennCounterexampleArtifactError(
            "complex payload must be finite"
        )
    return result


def _exact_nonnegative_integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise KrennCounterexampleArtifactError(
            f"{label} must be an exact integer"
        )
    result = int(value)
    if result < 0:
        raise KrennCounterexampleArtifactError(
            f"{label} must be nonnegative"
        )
    return result


def _finite_nonnegative_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise KrennCounterexampleArtifactError(
            f"{label} must be a finite real number"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise KrennCounterexampleArtifactError(
            f"{label} must be finite and nonnegative"
        )
    return result


def _positive_exact_integer(value, label: str) -> int:
    result = _exact_nonnegative_integer(value, label)
    if result == 0:
        raise KrennCounterexampleArtifactError(
            f"{label} must be positive"
        )
    return result


def _canonical_integer_argument(value: str, label: str) -> int:
    if not isinstance(value, str):
        raise KrennCounterexampleArtifactError(
            f"{label} argument must be text"
        )
    try:
        result = int(value)
    except ValueError as error:
        raise KrennCounterexampleArtifactError(
            f"{label} argument is not an integer"
        ) from error
    if str(result) != value:
        raise KrennCounterexampleArtifactError(
            f"{label} argument is not canonical"
        )
    return result


def _canonical_float_argument(value: str, label: str) -> float:
    if not isinstance(value, str):
        raise KrennCounterexampleArtifactError(
            f"{label} argument must be text"
        )
    try:
        result = float(value)
    except ValueError as error:
        raise KrennCounterexampleArtifactError(
            f"{label} argument is not a float"
        ) from error
    # Preserve the literal argv receipt, but replay its numerical meaning.
    # Ordinary spellings such as ``1e-8`` and ``1e-08`` are equivalent and
    # are cross-checked against the structured execution budget below.
    if not math.isfinite(result):
        raise KrennCounterexampleArtifactError(
            f"{label} argument is not finite"
        )
    return result


def _safe_windows_path(value, label: str) -> PureWindowsPath:
    if not isinstance(value, str) or not value:
        raise KrennCounterexampleArtifactError(
            f"{label} must be a nonempty path string"
        )
    result = PureWindowsPath(value)
    folded_parts = {part.casefold() for part in result.parts}
    if (
        not result.is_absolute()
        or ".." in result.parts
        or "onedrive" in folded_parts
        or "onedrive" in value.casefold()
    ):
        raise KrennCounterexampleArtifactError(
            f"{label} is not an absolute non-OneDrive path"
        )
    return result


def _parse_campaign_argv_receipt(payload) -> dict:
    if (
        not isinstance(payload, Sequence)
        or isinstance(payload, (str, bytes, bytearray))
        or not payload
        or any(not isinstance(value, str) for value in payload)
    ):
        raise KrennCounterexampleArtifactError(
            "campaign argv receipt must be a nonempty string sequence"
        )
    receipt = tuple(payload)
    cursor = 0

    def scalar(flag: str) -> str:
        nonlocal cursor
        if cursor + 1 >= len(receipt) or receipt[cursor] != flag:
            raise KrennCounterexampleArtifactError(
                f"campaign argv receipt expected {flag}"
            )
        value = receipt[cursor + 1]
        cursor += 2
        return value

    def vector(flag: str, following_flag: str) -> tuple[str, ...]:
        nonlocal cursor
        if cursor >= len(receipt) or receipt[cursor] != flag:
            raise KrennCounterexampleArtifactError(
                f"campaign argv receipt expected {flag}"
            )
        cursor += 1
        start = cursor
        while cursor < len(receipt) and receipt[cursor] != following_flag:
            if receipt[cursor].startswith("--"):
                raise KrennCounterexampleArtifactError(
                    f"campaign argv receipt changed after {flag}"
                )
            cursor += 1
        if cursor == start:
            raise KrennCounterexampleArtifactError(
                f"campaign argv receipt has no values for {flag}"
            )
        return receipt[start:cursor]

    scratch = scalar("--scratch-directory")
    results = scalar("--results-directory")
    workers = _canonical_integer_argument(
        scalar("--workers"), "workers"
    )
    orbit_nodes = _canonical_integer_argument(
        scalar("--orbit-nodes"), "orbit nodes"
    )
    support_state_cap = _canonical_integer_argument(
        scalar("--support-state-cap"), "support state cap"
    )
    retained_supports = _canonical_integer_argument(
        scalar("--retained-supports"), "retained supports"
    )
    omission_supports = _canonical_integer_argument(
        scalar("--omission-supports"), "omission supports"
    )
    starts_per_support = _canonical_integer_argument(
        scalar("--starts-per-support"), "starts per support"
    )
    dense_starts = _canonical_integer_argument(
        scalar("--dense-starts"), "dense starts"
    )
    radii = tuple(
        _canonical_float_argument(value, "radius")
        for value in vector("--radii", "--iterations")
    )
    iterations = _canonical_integer_argument(
        scalar("--iterations"), "iterations"
    )
    evaluations = _canonical_integer_argument(
        scalar("--evaluations"), "evaluations"
    )
    checkpoint_interval = _canonical_integer_argument(
        scalar("--checkpoint-interval"), "checkpoint interval"
    )
    continuation_schedule = tuple(
        _canonical_float_argument(value, "continuation target")
        for value in vector(
            "--continuation-schedule",
            "--best-per-radius-class",
        )
    )
    best_per_radius_class = _canonical_integer_argument(
        scalar("--best-per-radius-class"),
        "best candidates per radius and class",
    )
    reconstruction_trigger = _canonical_float_argument(
        scalar("--reconstruction-trigger"), "reconstruction trigger"
    )
    if cursor != len(receipt):
        raise KrennCounterexampleArtifactError(
            "campaign argv receipt contains trailing arguments"
        )
    return {
        "scratch_directory": scratch,
        "results_directory": results,
        "workers": workers,
        "orbit_node_cap": orbit_nodes,
        "discovered_support_cap_per_run": support_state_cap,
        "retained_support_candidates": retained_supports,
        "greedy_pruned_omission_supports": omission_supports,
        "starts_per_sparse_support": starts_per_support,
        "dense_starts_per_radius": dense_starts,
        "radii": radii,
        "iterations_per_solve": iterations,
        "evaluations_per_solve": evaluations,
        "checkpoint_interval": checkpoint_interval,
        "continuation_schedule": continuation_schedule,
        "best_per_radius_class": best_per_radius_class,
        "reconstruction_trigger": reconstruction_trigger,
    }


def _validate_execution_receipt(
    execution: Mapping,
    support: Mapping,
    numerical: Mapping,
) -> dict:
    if not isinstance(execution, Mapping) or set(execution) != (
        _EXECUTION_KEYS
    ):
        raise KrennCounterexampleArtifactError(
            "campaign execution receipt key set changed"
        )
    if (
        execution["branch"] != FEATURE_BRANCH
        or execution["baseline_origin_main"] != BASELINE_COMMIT
    ):
        raise KrennCounterexampleArtifactError(
            "campaign branch or baseline receipt changed"
        )
    master_seed = _exact_nonnegative_integer(
        execution["master_seed"], "campaign master seed"
    )
    workers = _positive_exact_integer(
        execution["workers"], "campaign worker count"
    )
    threads_per_worker = _positive_exact_integer(
        execution["threads_per_worker"], "threads per worker"
    )
    if (
        master_seed != numerical["master_seed"]
        or workers != numerical["worker_count"]
        or workers > MAX_WORKERS
        or threads_per_worker != 1
        or execution["worker_thread_environment"]
        != _WORKER_THREAD_ENVIRONMENT
    ):
        raise KrennCounterexampleArtifactError(
            "campaign seed, worker, or thread receipt is inconsistent"
        )
    if (
        execution["support_orientation_policy"]
        != _SUPPORT_ORIENTATION_POLICY
        or execution["scratch_data_committed"] is not False
        or execution[
            "large_checkpoints_required_for_bundle_replay"
        ]
        is not False
    ):
        raise KrennCounterexampleArtifactError(
            "campaign orientation or scratch claim boundary changed"
        )
    scratch = _safe_windows_path(
        execution["scratch_directory"], "campaign scratch directory"
    )
    if not scratch.is_relative_to(SCRATCH_ROOT):
        raise KrennCounterexampleArtifactError(
            "campaign scratch directory escaped the declared scratch root"
        )

    support_budgets = execution["support_budgets"]
    if (
        not isinstance(support_budgets, Mapping)
        or set(support_budgets) != _SUPPORT_BUDGET_KEYS
    ):
        raise KrennCounterexampleArtifactError(
            "campaign support-budget receipt changed"
        )
    orbit_nodes = _positive_exact_integer(
        support_budgets["orbit_node_cap"], "orbit node cap"
    )
    discovered_cap = _positive_exact_integer(
        support_budgets["discovered_support_cap_per_run"],
        "discovered-support cap",
    )
    retained_cap = _positive_exact_integer(
        support_budgets["retained_support_candidates"],
        "retained-support candidate cap",
    )
    retained_per_size = _positive_exact_integer(
        support_budgets["retained_candidates_per_support_size"],
        "retained per-size candidate cap",
    )
    omission_count = _positive_exact_integer(
        support_budgets["greedy_pruned_omission_supports"],
        "greedy-pruned omission count",
    )
    if (
        retained_per_size > retained_cap
        or omission_count > 45
    ):
        raise KrennCounterexampleArtifactError(
            "campaign support budgets are internally inconsistent"
        )
    plans = tuple(row["plan"] for row in support["runs"])
    if (
        not plans
        or any(
            plan["worker_count"] != workers
            or plan["deterministic_seed"] != master_seed
            or plan["discovered_support_cap"] != discovered_cap
            or _safe_windows_path(
                plan["scratch_directory"],
                "support-plan scratch directory",
            )
            != scratch
            for plan in plans
        )
        or not any(plan["node_cap"] == orbit_nodes for plan in plans)
        or not any(
            plan["candidate_cap"] == retained_cap for plan in plans
        )
        or not any(
            plan["per_support_size_candidate_quota"]
            == retained_per_size
            for plan in plans
        )
    ):
        raise KrennCounterexampleArtifactError(
            "campaign support budgets failed summary cross-replay"
        )
    pruned_runs = tuple(
        row
        for row in support["runs"]
        if row["plan"]["activation_width"] == 1
        and row["plan"]["omission_candidate_quota"]
        == row["plan"]["candidate_cap"]
    )
    if pruned_runs and not any(
        row["seed_count"] == omission_count for row in pruned_runs
    ):
        raise KrennCounterexampleArtifactError(
            "greedy-pruned omission budget failed summary cross-replay"
        )

    numerical_budgets = execution["numerical_budgets"]
    if (
        not isinstance(numerical_budgets, Mapping)
        or set(numerical_budgets) != _NUMERICAL_BUDGET_KEYS
    ):
        raise KrennCounterexampleArtifactError(
            "campaign numerical-budget receipt changed"
        )
    if (
        numerical_budgets["continue_after_nonconvergence"] is not True
        or numerical_budgets["resume_from_checkpoint"] is not True
    ):
        raise KrennCounterexampleArtifactError(
            "exploratory continuation and checkpoint-resume policies "
            "must both be enabled"
        )
    raw_radii = numerical_budgets["radii"]
    if (
        not isinstance(raw_radii, Sequence)
        or isinstance(raw_radii, (str, bytes, bytearray))
        or not raw_radii
    ):
        raise KrennCounterexampleArtifactError(
            "campaign radii receipt must be a nonempty sequence"
        )
    radii = tuple(
        _finite_nonnegative_number(value, "campaign radius")
        for value in raw_radii
    )
    if (
        len(radii) != len(set(radii))
        or any(radius < 1.0 for radius in radii)
    ):
        raise KrennCounterexampleArtifactError(
            "campaign radii must be unique and at least one"
        )
    summary_radii = tuple(
        float(row["l2_bound"]) for row in numerical["radii"]
    )
    if (
        any(
            float(row["linf_bound"]) != float(row["l2_bound"])
            for row in numerical["radii"]
        )
        or radii != summary_radii
    ):
        raise KrennCounterexampleArtifactError(
            "campaign radii failed numerical-summary cross-replay"
        )
    starts = _positive_exact_integer(
        numerical_budgets["starts_per_sparse_support"],
        "sparse starts per support",
    )
    dense_starts = _positive_exact_integer(
        numerical_budgets["dense_starts_per_radius"],
        "dense starts per radius",
    )
    iterations = _positive_exact_integer(
        numerical_budgets["iterations_per_solve"],
        "iterations per solve",
    )
    evaluations = _positive_exact_integer(
        numerical_budgets["evaluations_per_solve"],
        "evaluations per solve",
    )
    checkpoint_interval = _exact_nonnegative_integer(
        numerical_budgets["checkpoint_interval"],
        "checkpoint interval",
    )
    if (
        evaluations < 2
        or checkpoint_interval > iterations
        or starts > 1_000
        or dense_starts > 1_000
    ):
        raise KrennCounterexampleArtifactError(
            "campaign numerical integer budgets are inconsistent"
        )
    raw_schedule = numerical_budgets["continuation_schedule"]
    if (
        not isinstance(raw_schedule, Sequence)
        or isinstance(raw_schedule, (str, bytes, bytearray))
        or not raw_schedule
    ):
        raise KrennCounterexampleArtifactError(
            "continuation schedule receipt must be a nonempty sequence"
        )
    schedule = tuple(
        _finite_nonnegative_number(value, "continuation target")
        for value in raw_schedule
    )
    if (
        schedule[-1] != 0.0
        or any(
            schedule[index + 1] >= schedule[index]
            for index in range(len(schedule) - 1)
        )
    ):
        raise KrennCounterexampleArtifactError(
            "continuation schedule must strictly decrease to zero"
        )
    expected_jobs = len(radii) * (
        len(support["selected_supports_for_numerics"]) * starts
        + dense_starts
    )
    if numerical["jobs_scheduled"] != expected_jobs:
        raise KrennCounterexampleArtifactError(
            "scheduled numerical jobs disagree with execution budgets"
        )
    for record in numerical["best_candidates"]:
        plan = record["plan"]
        if (
            plan["max_iterations"] != iterations
            or plan["max_evaluations"] != evaluations
            or plan["checkpoint_interval"] != checkpoint_interval
        ):
            raise KrennCounterexampleArtifactError(
                "selected candidate plan disagrees with execution budgets"
            )
        target_receipt = tuple(
            float(row["target_amplitude"][0])
            for row in record["continuation_growth"]
        )
        expected_targets = (
            (*schedule, 0.0)
            if record["job"]["use_continuation"]
            else (0.0,)
        )
        if target_receipt != expected_targets:
            raise KrennCounterexampleArtifactError(
                "selected continuation disagrees with execution schedule"
            )

    argv = _parse_campaign_argv_receipt(execution["argv_receipt"])
    results_path = _safe_windows_path(
        argv["results_directory"], "campaign results directory"
    )
    results_root = PureWindowsPath(str(ROOT / "results"))
    if (
        _safe_windows_path(
            argv["scratch_directory"], "argv scratch directory"
        )
        != scratch
        or not results_path.is_relative_to(results_root)
        or argv["workers"] != workers
        or argv["orbit_node_cap"] != orbit_nodes
        or argv["discovered_support_cap_per_run"] != discovered_cap
        or argv["retained_support_candidates"] != retained_cap
        or argv["greedy_pruned_omission_supports"] != omission_count
        or argv["starts_per_sparse_support"] != starts
        or argv["dense_starts_per_radius"] != dense_starts
        or argv["radii"] != radii
        or argv["iterations_per_solve"] != iterations
        or argv["evaluations_per_solve"] != evaluations
        or argv["checkpoint_interval"] != checkpoint_interval
        or argv["continuation_schedule"] != schedule
        or not 1 <= argv["best_per_radius_class"] <= 1_000
        or argv["reconstruction_trigger"] <= 0.0
    ):
        raise KrennCounterexampleArtifactError(
            "campaign argv receipt failed structured cross-replay"
        )
    commands = numerical["command_lines"]
    if (
        not isinstance(commands, Sequence)
        or isinstance(commands, (str, bytes, bytearray))
        or len(commands) != 2
        or any(
            not isinstance(command, str)
            or "experiments.krenn_quantum_graph.counterexample_campaign"
            not in command
            for command in commands
        )
        or "--verify-only" in commands[0]
        or "--verify-only" not in commands[1]
    ):
        raise KrennCounterexampleArtifactError(
            "campaign command-line receipt changed"
        )
    return dict(execution)


def _job_spec_from_dict(payload: Mapping) -> NumericalJobSpec:
    try:
        result = NumericalJobSpec(
            run_id=str(payload["run_id"]),
            support=tuple(payload["support"]),
            master_seed=int(payload["master_seed"]),
            spawn_key=tuple(payload["spawn_key"]),
            seed_words=tuple(payload["seed_words_uint32"]),
            initializer_kind=str(payload["initializer_kind"]),
            activation_amplitude=float(
                payload["activation_amplitude"]
            ),
            laurent_t=float(payload["laurent_t"]),
            dense=bool(payload["dense_all_135_variables"]),
            use_continuation=bool(payload["use_continuation"]),
            schema=str(payload["schema"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise KrennCounterexampleArtifactError(
            "could not reconstruct deterministic numerical job"
        ) from error
    if result.to_dict() != dict(payload):
        raise KrennCounterexampleArtifactError(
            "numerical job payload failed deterministic replay"
        )
    return result


def _plan_from_dict(payload: Mapping) -> LMPlan:
    expected = {
        "schema",
        "max_iterations",
        "max_evaluations",
        "l2_bound",
        "linf_bound",
        "initial_damping",
        "damping_increase",
        "damping_decrease",
        "initial_trust_radius",
        "minimum_trust_radius",
        "maximum_trust_radius",
        "backtracking_steps",
        "acceptance_ratio",
        "residual_tolerance",
        "gradient_tolerance",
        "step_tolerance",
        "checkpoint_interval",
        "complex_weights",
        "hard_l2_control",
        "hard_linf_control",
        "symmetry_weight_equalities",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise KrennCounterexampleArtifactError(
            "LM plan payload key set changed"
        )
    if (
        payload["complex_weights"] is not True
        or payload["hard_l2_control"] is not True
        or payload["hard_linf_control"] is not True
        or payload["symmetry_weight_equalities"] != 0
    ):
        raise KrennCounterexampleArtifactError(
            "LM plan claim boundary changed"
        )
    kwargs = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "complex_weights",
            "hard_l2_control",
            "hard_linf_control",
            "symmetry_weight_equalities",
        }
    }
    try:
        result = LMPlan(**kwargs)
    except (TypeError, ValueError) as error:
        raise KrennCounterexampleArtifactError(
            "could not reconstruct bounded LM plan"
        ) from error
    if result.to_dict() != dict(payload):
        raise KrennCounterexampleArtifactError(
            "bounded LM plan failed semantic replay"
        )
    return result


def _selected_result(
    result: NumericalSolveResult | ContinuationResult,
) -> tuple[NumericalSolveResult, tuple[NumericalSolveResult, ...]]:
    if isinstance(result, NumericalSolveResult):
        return result, (result,)
    if not isinstance(result, ContinuationResult) or result.final_step is None:
        raise KrennCounterexampleArtifactError(
            "selected numerical campaign result has no final solve"
        )
    return result.final_step, result.steps


def numerical_candidate_record(
    result: NumericalSolveResult | ContinuationResult,
    job: NumericalJobSpec,
    *,
    reconstruction_status: str = "not-attempted",
) -> dict:
    """Create a compact, fully replayable record for one selected best run."""

    if reconstruction_status not in {
        "not-attempted",
        "attempted-no-exact-reconstruction",
        "exact-dual-verification-passed",
    }:
        raise KrennCounterexampleArtifactError(
            "unknown exact reconstruction status"
        )
    final, stages = _selected_result(result)
    if final.target_amplitude != 0.0j or final.support != job.support:
        raise KrennCounterexampleArtifactError(
            "selected campaign result is not a direct GHZ comparison"
        )
    weights = final.dense_weights()
    residual = complex_residual(weights)
    residual_metrics = ResidualMetrics.from_residual(residual)
    weight_metrics = WeightMetrics.from_weights(
        weights,
        final.chart,
        l2_bound=final.plan.l2_bound,
        linf_bound=final.plan.linf_bound,
    )
    invariants = InvariantMetrics.from_weights(weights, final.support)
    numerical_zero = bool(
        residual_metrics.linf is not None
        and residual_metrics.linf <= final.plan.residual_tolerance
    )
    if numerical_zero and not weight_metrics.on_hard_boundary:
        classification = "interior-numerical-zero-unverified"
    elif numerical_zero:
        classification = "boundary-numerical-zero-unverified"
    elif weight_metrics.on_hard_boundary:
        classification = "bounded-miss-with-active-norm-control"
    else:
        classification = "bounded-interior-miss"
    exact = reconstruction_status == "exact-dual-verification-passed"
    return {
        "run_id": job.run_id,
        "job": job.to_dict(),
        "solver_status": final.status,
        "solver_iterations": final.iterations,
        "solver_evaluations": final.evaluations,
        "plan": final.plan.to_dict(),
        "gauge_chart": final.chart.to_dict(),
        "support": list(final.support),
        "weights": [
            {
                "index": index,
                "value": _complex_pair(weights[index]),
            }
            for index in final.support
        ],
        "all_729_residuals": [
            _complex_pair(value) for value in residual
        ],
        "residual_metrics": residual_metrics.to_dict(),
        "weight_metrics": weight_metrics.to_dict(),
        "gauge_invariant_metrics": invariants.to_dict(),
        "continuation_growth": [
            {
                "target_amplitude": _complex_pair(
                    stage.target_amplitude
                ),
                "residual_linf": stage.residual_metrics.linf,
                "weight_l2": stage.weight_metrics.l2,
                "weight_linf": stage.weight_metrics.linf,
                "known_Q_absolute_value": (
                    stage.invariant_metrics.known_q_absolute_value
                ),
                "hard_boundary_active": (
                    stage.weight_metrics.on_hard_boundary
                ),
            }
            for stage in stages
        ],
        "classification": classification,
        "numerical_zero_under_declared_tolerance": numerical_zero,
        "weights_finite": weight_metrics.finite,
        "exact_reconstruction_status": reconstruction_status,
        "exact_dual_verification_passed": exact,
        "claim_boundary": {
            "all_729_floating_residuals_stored": True,
            "numerical_zero_is_exact_witness": False,
            "bounded_miss_proves_nonexistence": False,
            "exact_claim_requires_separate_dual_verification": True,
        },
    }


def verify_numerical_candidate_record(payload: Mapping) -> dict:
    """Replay every selected weight and all 729 stored residuals."""

    required = {
        "run_id",
        "job",
        "solver_status",
        "solver_iterations",
        "solver_evaluations",
        "plan",
        "gauge_chart",
        "support",
        "weights",
        "all_729_residuals",
        "residual_metrics",
        "weight_metrics",
        "gauge_invariant_metrics",
        "continuation_growth",
        "classification",
        "numerical_zero_under_declared_tolerance",
        "weights_finite",
        "exact_reconstruction_status",
        "exact_dual_verification_passed",
        "claim_boundary",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise KrennCounterexampleArtifactError(
            "numerical candidate record key set changed"
        )
    job = _job_spec_from_dict(payload["job"])
    if payload["run_id"] != job.run_id:
        raise KrennCounterexampleArtifactError(
            "numerical candidate run identity changed"
        )
    plan = _plan_from_dict(payload["plan"])
    status = payload["solver_status"]
    if (
        not isinstance(status, str)
        or status not in _LM_TERMINATION_STATUSES
    ):
        raise KrennCounterexampleArtifactError(
            "selected numerical solver status is unknown"
        )
    iterations = _exact_nonnegative_integer(
        payload["solver_iterations"], "selected solver iteration count"
    )
    evaluations = _exact_nonnegative_integer(
        payload["solver_evaluations"], "selected solver evaluation count"
    )
    if (
        iterations > plan.max_iterations
        or evaluations < 1
        or evaluations > plan.max_evaluations
        or (
            status == "iteration-cap-reached"
            and iterations != plan.max_iterations
        )
        or (
            status == "evaluation-cap-reached"
            and evaluations != plan.max_evaluations
        )
    ):
        raise KrennCounterexampleArtifactError(
            "selected solver accounting exceeds its declared plan"
        )
    support = tuple(map(int, payload["support"]))
    if support != job.support:
        raise KrennCounterexampleArtifactError(
            "numerical candidate support changed"
        )
    moving_target_job = bool(
        job.use_continuation
        or job.initializer_kind == "transverse-laurent-activation"
    )
    action_group = (
        GHZ_VICTIM_COLOR_DIAGONAL_GAUGE
        if moving_target_job
        else GHZ_COLOR_DIAGONAL_GAUGE
    )
    expected_chart = (
        natural_gauge_chart(support, action_group=action_group)
        if job.dense or NATURAL_SUPPORT_SET.issubset(support)
        else support_gauge_chart(support, action_group=action_group)
    )
    if payload["gauge_chart"] != expected_chart.to_dict():
        raise KrennCounterexampleArtifactError(
            "numerical candidate gauge chart failed replay"
        )
    weights = np.zeros(AMBIENT_VARIABLES, dtype=np.complex128)
    rows = payload["weights"]
    if not isinstance(rows, Sequence) or len(rows) != len(support):
        raise KrennCounterexampleArtifactError(
            "numerical candidate weight inventory changed"
        )
    for expected_index, row in zip(support, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {"index", "value"}:
            raise KrennCounterexampleArtifactError(
                "numerical candidate weight row changed"
            )
        if int(row["index"]) != expected_index:
            raise KrennCounterexampleArtifactError(
                "numerical candidate weight order changed"
            )
        weights[expected_index] = _complex_from_pair(row["value"])
    residual = complex_residual(weights)
    stored_residual = np.asarray(
        tuple(
            _complex_from_pair(value)
            for value in payload["all_729_residuals"]
        ),
        dtype=np.complex128,
    )
    if stored_residual.shape != (729,) or not np.array_equal(
        stored_residual, residual
    ):
        raise KrennCounterexampleArtifactError(
            "stored 729-equation residual vector failed replay"
        )
    residual_metrics = ResidualMetrics.from_residual(residual)
    weight_metrics = WeightMetrics.from_weights(
        weights,
        expected_chart,
        l2_bound=plan.l2_bound,
        linf_bound=plan.linf_bound,
    )
    invariants = InvariantMetrics.from_weights(weights, support)
    if (
        payload["residual_metrics"] != residual_metrics.to_dict()
        or payload["weight_metrics"] != weight_metrics.to_dict()
        or payload["gauge_invariant_metrics"] != invariants.to_dict()
    ):
        raise KrennCounterexampleArtifactError(
            "numerical candidate metrics failed semantic replay"
        )
    growth = payload["continuation_growth"]
    if (
        not isinstance(growth, Sequence)
        or isinstance(growth, (str, bytes, bytearray))
        or not growth
    ):
        raise KrennCounterexampleArtifactError(
            "continuation growth ledger must be a nonempty sequence"
        )
    target_values = []
    for row_index, row in enumerate(growth):
        if not isinstance(row, Mapping) or set(row) != (
            _CONTINUATION_GROWTH_KEYS
        ):
            raise KrennCounterexampleArtifactError(
                "continuation growth row key set changed"
            )
        target = _complex_from_pair(row["target_amplitude"])
        if target.imag != 0.0 or target.real < 0.0:
            raise KrennCounterexampleArtifactError(
                "continuation targets must be real and nonnegative"
            )
        target_values.append(target.real)
        _finite_nonnegative_number(
            row["residual_linf"],
            f"continuation row {row_index} residual",
        )
        row_l2 = _finite_nonnegative_number(
            row["weight_l2"],
            f"continuation row {row_index} l2 norm",
        )
        row_linf = _finite_nonnegative_number(
            row["weight_linf"],
            f"continuation row {row_index} linf norm",
        )
        if (
            row_l2 > plan.l2_bound * (1.0 + 1.0e-10)
            or row_linf > plan.linf_bound * (1.0 + 1.0e-10)
        ):
            raise KrennCounterexampleArtifactError(
                "continuation growth exceeded a declared hard norm bound"
            )
        expected_boundary = bool(
            row_l2 >= plan.l2_bound * (1.0 - 1.0e-10)
            or row_linf >= plan.linf_bound * (1.0 - 1.0e-10)
        )
        if (
            not isinstance(row["hard_boundary_active"], bool)
            or row["hard_boundary_active"] != expected_boundary
        ):
            raise KrennCounterexampleArtifactError(
                "continuation hard-boundary status failed replay"
            )
        known_q = row["known_Q_absolute_value"]
        if known_q is not None:
            _finite_nonnegative_number(
                known_q,
                f"continuation row {row_index} known-Q magnitude",
            )
    first_zero = next(
        (
            index
            for index, target in enumerate(target_values)
            if target == 0.0
        ),
        None,
    )
    if first_zero is None or any(
        target != 0.0 for target in target_values[first_zero:]
    ) or len(target_values) - first_zero > 2:
        raise KrennCounterexampleArtifactError(
            "continuation growth must end at the direct GHZ target"
        )
    if any(
        target_values[index + 1] >= target_values[index]
        for index in range(max(first_zero - 1, 0))
    ):
        raise KrennCounterexampleArtifactError(
            "positive continuation targets must strictly decrease"
        )
    if not job.use_continuation and len(growth) != 1:
        raise KrennCounterexampleArtifactError(
            "a direct numerical job cannot carry a continuation history"
        )
    expected_final_growth = {
        "target_amplitude": [0.0, 0.0],
        "residual_linf": residual_metrics.linf,
        "weight_l2": weight_metrics.l2,
        "weight_linf": weight_metrics.linf,
        "known_Q_absolute_value": invariants.known_q_absolute_value,
        "hard_boundary_active": weight_metrics.on_hard_boundary,
    }
    if growth[-1] != expected_final_growth:
        raise KrennCounterexampleArtifactError(
            "terminal continuation growth row failed final-state replay"
        )
    numerical_zero = bool(
        residual_metrics.linf is not None
        and residual_metrics.linf <= plan.residual_tolerance
    )
    if status == "numerical-residual-tolerance" and not numerical_zero:
        raise KrennCounterexampleArtifactError(
            "solver tolerance status disagrees with replayed residual"
        )
    if numerical_zero and not weight_metrics.on_hard_boundary:
        expected_classification = "interior-numerical-zero-unverified"
    elif numerical_zero:
        expected_classification = "boundary-numerical-zero-unverified"
    elif weight_metrics.on_hard_boundary:
        expected_classification = "bounded-miss-with-active-norm-control"
    else:
        expected_classification = "bounded-interior-miss"
    if payload["numerical_zero_under_declared_tolerance"] != numerical_zero:
        raise KrennCounterexampleArtifactError(
            "numerical-zero classification changed"
        )
    if (
        payload["classification"] != expected_classification
        or payload["weights_finite"] != weight_metrics.finite
    ):
        raise KrennCounterexampleArtifactError(
            "numerical finite/boundary classification changed"
        )
    status = payload["exact_reconstruction_status"]
    if status not in {
        "not-attempted",
        "attempted-no-exact-reconstruction",
        "exact-dual-verification-passed",
    }:
        raise KrennCounterexampleArtifactError(
            "exact reconstruction status changed"
        )
    if payload["exact_dual_verification_passed"] != (
        status == "exact-dual-verification-passed"
    ):
        raise KrennCounterexampleArtifactError(
            "exact reconstruction claim is not status-derived"
        )
    if payload["claim_boundary"] != {
        "all_729_floating_residuals_stored": True,
        "numerical_zero_is_exact_witness": False,
        "bounded_miss_proves_nonexistence": False,
        "exact_claim_requires_separate_dual_verification": True,
    }:
        raise KrennCounterexampleArtifactError(
            "numerical candidate claim boundary changed"
        )
    return dict(payload)


def build_support_summary(
    results: Sequence[SupportCampaignResult],
    *,
    selected_supports: Sequence[Sequence[int]],
) -> dict:
    runs = tuple(results)
    if not runs or any(
        not isinstance(result, SupportCampaignResult) for result in runs
    ):
        raise KrennCounterexampleArtifactError(
            "support summary needs canonical campaign results"
        )
    selected = tuple(tuple(map(int, support)) for support in selected_supports)
    if any(
        support != tuple(sorted(set(support)))
        or len(support) < 22
        or any(index < 0 or index >= AMBIENT_VARIABLES for index in support)
        for support in selected
    ):
        raise KrennCounterexampleArtifactError(
            "selected numerical supports are not canonical size-22+ strata"
        )
    available = {
        support
        for result in runs
        for candidate in result.candidates
        for support in (
            candidate.support,
            candidate.orbit_representative,
        )
    }
    if any(support not in available for support in selected):
        raise KrennCounterexampleArtifactError(
            "a selected numerical support is absent from support search"
        )
    return {
        "schema": SUPPORT_SUMMARY_SCHEMA,
        "runs": [result.to_dict() for result in runs],
        "selected_supports_for_numerics": [
            list(support) for support in selected
        ],
        "selected_support_sizes": [
            len(support) for support in selected
        ],
        "includes_selected_natural_omission": any(
            set(
                (
                    0,
                    13,
                    26,
                    67,
                    80,
                    81,
                    98,
                    121,
                    126,
                )
            ).difference(support)
            for support in selected
        ),
        "claim_boundary": {
            "support_generation_is_bounded": True,
            "singleton_closure_is_necessary_only": True,
            "symmetry_weight_equalities": 0,
            "support_miss_proves_nonexistence": False,
        },
    }


def verify_support_summary(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "runs",
        "selected_supports_for_numerics",
        "selected_support_sizes",
        "includes_selected_natural_omission",
        "claim_boundary",
    }:
        raise KrennCounterexampleArtifactError(
            "support summary key set changed"
        )
    if payload["schema"] != SUPPORT_SUMMARY_SCHEMA:
        raise KrennCounterexampleArtifactError(
            "support summary schema changed"
        )
    runs = tuple(
        support_campaign_from_dict(row) for row in payload["runs"]
    )
    selected = tuple(
        tuple(map(int, support))
        for support in payload["selected_supports_for_numerics"]
    )
    expected = build_support_summary(
        runs, selected_supports=selected
    )
    if dict(payload) != expected:
        raise KrennCounterexampleArtifactError(
            "support summary failed semantic replay"
        )
    return expected


def build_numerical_summary(
    *,
    master_seed: int,
    worker_count: int,
    radii: Sequence[tuple[float, float]],
    jobs_scheduled: int,
    jobs_completed: int,
    best_candidates: Sequence[Mapping],
    command_lines: Sequence[str],
) -> dict:
    worker_count = int(worker_count)
    if not 1 <= worker_count <= MAX_WORKERS:
        raise KrennCounterexampleArtifactError(
            "numerical worker count must be between 1 and 16"
        )
    radius_rows = tuple((float(a), float(b)) for a, b in radii)
    if not radius_rows or any(
        not math.isfinite(a)
        or not math.isfinite(b)
        or a <= 0
        or b <= 0
        for a, b in radius_rows
    ):
        raise KrennCounterexampleArtifactError(
            "numerical radii must be finite and positive"
        )
    scheduled = _exact_nonnegative_integer(
        jobs_scheduled, "scheduled numerical job count"
    )
    completed = _exact_nonnegative_integer(
        jobs_completed, "completed numerical job count"
    )
    if completed > scheduled:
        raise KrennCounterexampleArtifactError(
            "completed numerical jobs exceed scheduled jobs"
        )
    selected = [
        verify_numerical_candidate_record(record)
        for record in best_candidates
    ]
    if len(selected) > completed:
        raise KrennCounterexampleArtifactError(
            "selected candidates exceed completed numerical jobs"
        )
    radius_set = set(radius_rows)
    selected_keys = set()
    for record in selected:
        if record["job"]["master_seed"] != int(master_seed):
            raise KrennCounterexampleArtifactError(
                "selected numerical candidate uses another master seed"
            )
        candidate_radius = (
            record["plan"]["l2_bound"],
            record["plan"]["linf_bound"],
        )
        if candidate_radius not in radius_set:
            raise KrennCounterexampleArtifactError(
                "selected numerical candidate uses an undeclared radius"
            )
        key = (record["run_id"], *candidate_radius)
        if key in selected_keys:
            raise KrennCounterexampleArtifactError(
                "selected numerical candidate inventory contains a duplicate"
            )
        selected_keys.add(key)
    exact = any(
        record["exact_dual_verification_passed"]
        for record in selected
    )
    return {
        "schema": NUMERICAL_SUMMARY_SCHEMA,
        "system_fingerprint": system_fingerprint(),
        "master_seed": int(master_seed),
        "worker_count": worker_count,
        "radii": [
            {"l2_bound": a, "linf_bound": b}
            for a, b in radius_rows
        ],
        "jobs_scheduled": scheduled,
        "jobs_completed": completed,
        "selected_best_candidate_count": len(selected),
        "best_candidates": selected,
        "command_lines": list(map(str, command_lines)),
        "exact_candidate_found": exact,
        "claim_boundary": {
            "complex_multistart_is_bounded_and_deterministic": True,
            "all_large_checkpoints_are_disposable": True,
            "numerical_zero_is_not_a_proof": True,
            "bounded_search_miss_is_not_a_proof": True,
            "f31_solution_would_not_be_a_characteristic_zero_proof": True,
            "nonexistence_proved": False,
        },
    }


def verify_numerical_summary(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "system_fingerprint",
        "master_seed",
        "worker_count",
        "radii",
        "jobs_scheduled",
        "jobs_completed",
        "selected_best_candidate_count",
        "best_candidates",
        "command_lines",
        "exact_candidate_found",
        "claim_boundary",
    }:
        raise KrennCounterexampleArtifactError(
            "numerical summary key set changed"
        )
    if (
        payload["schema"] != NUMERICAL_SUMMARY_SCHEMA
        or payload["system_fingerprint"] != system_fingerprint()
    ):
        raise KrennCounterexampleArtifactError(
            "numerical summary identity changed"
        )
    radii = tuple(
        (row["l2_bound"], row["linf_bound"])
        for row in payload["radii"]
    )
    expected = build_numerical_summary(
        master_seed=payload["master_seed"],
        worker_count=payload["worker_count"],
        radii=radii,
        jobs_scheduled=payload["jobs_scheduled"],
        jobs_completed=payload["jobs_completed"],
        best_candidates=payload["best_candidates"],
        command_lines=payload["command_lines"],
    )
    if dict(payload) != expected:
        raise KrennCounterexampleArtifactError(
            "numerical summary failed semantic replay"
        )
    if not 0 <= expected["jobs_completed"] <= expected["jobs_scheduled"]:
        raise KrennCounterexampleArtifactError(
            "numerical job accounting is inconsistent"
        )
    return expected


def _certificate(
    support: Mapping,
    numerical: Mapping,
    exact_report: ExactVerificationReport | None,
    execution: Mapping,
) -> dict:
    exact = exact_report is not None
    if exact and (
        not exact_report.exact
        or (exact_report.witness.n, exact_report.witness.d) != (6, 3)
    ):
        raise KrennCounterexampleArtifactError(
            "exact bundle candidate is not a dual-verified n=6,d=3 witness"
        )
    if numerical["exact_candidate_found"] != exact:
        raise KrennCounterexampleArtifactError(
            "numerical and exact candidate statuses disagree"
        )
    allowed_supports = {
        tuple(support_row)
        for support_row in support["selected_supports_for_numerics"]
    }
    allowed_supports.add(tuple(range(AMBIENT_VARIABLES)))
    if any(
        tuple(record["support"]) not in allowed_supports
        for record in numerical["best_candidates"]
    ):
        raise KrennCounterexampleArtifactError(
            "selected numerical candidate was not authorized by support search"
        )
    numerical_complete = bool(
        numerical["jobs_scheduled"] > 0
        and numerical["jobs_completed"] == numerical["jobs_scheduled"]
    )
    execution_receipt = _validate_execution_receipt(
        execution, support, numerical
    )
    return {
        "schema": CAMPAIGN_CERTIFICATE_SCHEMA,
        "parameters": {
            "n": 6,
            "d": 3,
            "ambient_variables": 135,
            "equations": 729,
        },
        "baseline_commit": BASELINE_COMMIT,
        "execution": execution_receipt,
        "support_run_count": len(support["runs"]),
        "selected_support_count": len(
            support["selected_supports_for_numerics"]
        ),
        "numerical_jobs_scheduled": numerical["jobs_scheduled"],
        "numerical_jobs_completed": numerical["jobs_completed"],
        "selected_best_candidate_count": numerical[
            "selected_best_candidate_count"
        ],
        "exact_candidate_present": exact,
        "status": (
            "exact-finite-counterexample-dual-verified"
            if exact
            else (
                "bounded-search-complete-no-exact-candidate"
                if numerical_complete
                else "bounded-search-incomplete-no-exact-candidate"
            )
        ),
        "artifacts": {
            "support_search": SUPPORT_FILE,
            "numerical_search": NUMERICAL_FILE,
            "exact_witness": (
                EXACT_WITNESS_FILE if exact else None
            ),
            "exact_verification": (
                EXACT_VERIFICATION_FILE if exact else None
            ),
            "manifest": MANIFEST_FILE,
        },
        "claim_boundary": {
            "finite_counterexample_certified": exact,
            "all_729_exact_equations_verified_twice": exact,
            "numerical_zero_promoted_without_exact_reconstruction": False,
            "bounded_miss_proves_nonexistence": False,
            "f31_used_as_proof": False,
            "krenn_gu_conjecture_refuted": exact,
        },
    }


@dataclass(frozen=True)
class LoadedCounterexampleBundle:
    directory: Path
    support_summary: Mapping
    numerical_summary: Mapping
    certificate: Mapping
    exact_report: ExactVerificationReport | None

    @property
    def exact_counterexample(self) -> bool:
        return self.exact_report is not None and self.exact_report.exact


def write_counterexample_bundle(
    output_directory: Path | str,
    *,
    support_summary: Mapping,
    numerical_summary: Mapping,
    execution: Mapping,
    exact_report: ExactVerificationReport | None = None,
) -> Path:
    """Write a compact bundle and immediately replay every included claim."""

    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    support = verify_support_summary(support_summary)
    numerical = verify_numerical_summary(numerical_summary)
    certificate = _certificate(
        support, numerical, exact_report, execution
    )
    expected_names = {
        SUPPORT_FILE,
        NUMERICAL_FILE,
        CERTIFICATE_FILE,
        MANIFEST_FILE,
    }
    if exact_report is not None:
        expected_names.update(
            {EXACT_WITNESS_FILE, EXACT_VERIFICATION_FILE}
        )
    unexpected = {
        path.name
        for path in directory.iterdir()
        if path.name not in expected_names
    }
    if unexpected:
        raise KrennCounterexampleArtifactError(
            f"refusing bundle with unexpected files: {sorted(unexpected)}"
        )
    _write_json_atomic(directory / SUPPORT_FILE, support)
    _write_json_atomic(directory / NUMERICAL_FILE, numerical)
    if exact_report is not None:
        _write_json_atomic(
            directory / EXACT_WITNESS_FILE,
            exact_report.witness.to_dict(),
        )
        _write_json_atomic(
            directory / EXACT_VERIFICATION_FILE,
            exact_report.to_dict(),
        )
    _write_json_atomic(directory / CERTIFICATE_FILE, certificate)

    artifact_names = sorted(expected_names.difference((MANIFEST_FILE,)))
    inputs = tuple(ROOT / path for path in REQUIRED_SOURCE_PATHS)
    manifest = {
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "producer": _file_record(Path(__file__), relative_to=ROOT),
        "artifacts": [
            _file_record(directory / name, relative_to=directory)
            for name in artifact_names
        ],
        "inputs": [
            _file_record(path, relative_to=ROOT) for path in inputs
        ],
        "scratch_data_included": False,
        "scratch_data_required_for_verification": False,
    }
    _write_json_atomic(directory / MANIFEST_FILE, manifest)
    verify_counterexample_bundle(directory)
    return directory / CERTIFICATE_FILE


def verify_counterexample_bundle(
    output_directory: Path | str,
) -> LoadedCounterexampleBundle:
    """Verify hashes, exact inventory, semantics, and fail-closed status."""

    directory = Path(output_directory).resolve()
    manifest = _load_json(directory / MANIFEST_FILE)
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema",
        "producer",
        "artifacts",
        "inputs",
        "scratch_data_included",
        "scratch_data_required_for_verification",
    }:
        raise KrennCounterexampleArtifactError(
            "campaign manifest key set changed"
        )
    if (
        manifest["schema"] != CAMPAIGN_MANIFEST_SCHEMA
        or manifest["scratch_data_included"] is not False
        or manifest["scratch_data_required_for_verification"] is not False
    ):
        raise KrennCounterexampleArtifactError(
            "campaign manifest identity or scratch boundary changed"
        )
    artifact_records = manifest["artifacts"]
    if (
        not isinstance(artifact_records, Sequence)
        or isinstance(artifact_records, (str, bytes, bytearray))
        or any(
            not isinstance(record, Mapping)
            or set(record) != {"path", "bytes", "sha256"}
            or not isinstance(record.get("path"), str)
            for record in artifact_records
        )
    ):
        raise KrennCounterexampleArtifactError(
            "campaign artifact ledger is not a sequence"
        )
    artifact_names = tuple(record["path"] for record in artifact_records)
    bounded_names = tuple(
        sorted((CERTIFICATE_FILE, NUMERICAL_FILE, SUPPORT_FILE))
    )
    exact_names = tuple(
        sorted(
            (
                *bounded_names,
                EXACT_WITNESS_FILE,
                EXACT_VERIFICATION_FILE,
            )
        )
    )
    if artifact_names not in {bounded_names, exact_names}:
        raise KrennCounterexampleArtifactError(
            "campaign artifact ledger is not the unique sorted inventory"
        )
    present = {
        path.name for path in directory.iterdir()
    }
    expected_present = set(artifact_names).union((MANIFEST_FILE,))
    if present != expected_present:
        raise KrennCounterexampleArtifactError(
            "campaign artifact inventory changed"
        )
    for record in artifact_records:
        path = directory / record["path"]
        if record != _file_record(path, relative_to=directory):
            raise KrennCounterexampleArtifactError(
                f"campaign artifact hash changed: {path.name}"
            )
    expected_inputs = [
        _file_record(ROOT / path, relative_to=ROOT)
        for path in REQUIRED_SOURCE_PATHS
    ]
    if manifest["inputs"] != expected_inputs:
        raise KrennCounterexampleArtifactError(
            "campaign source ledger changed"
        )
    if manifest["producer"] != _file_record(
        Path(__file__), relative_to=ROOT
    ):
        raise KrennCounterexampleArtifactError(
            "campaign producer record changed"
        )

    support = verify_support_summary(_load_json(directory / SUPPORT_FILE))
    numerical = verify_numerical_summary(
        _load_json(directory / NUMERICAL_FILE)
    )
    certificate = _load_json(directory / CERTIFICATE_FILE)
    exact_report = None
    if EXACT_VERIFICATION_FILE in artifact_names:
        exact_payload = _load_json(directory / EXACT_VERIFICATION_FILE)
        exact_report = ExactVerificationReport.from_dict(exact_payload)
        witness_payload = _load_json(directory / EXACT_WITNESS_FILE)
        if witness_payload != exact_report.witness.to_dict():
            raise KrennCounterexampleArtifactError(
                "exact witness and verification payloads disagree"
            )
    elif EXACT_WITNESS_FILE in artifact_names:
        raise KrennCounterexampleArtifactError(
            "exact witness lacks its dual verification"
        )
    execution = certificate.get("execution")
    expected_certificate = _certificate(
        support, numerical, exact_report, execution
    )
    if certificate != expected_certificate:
        raise KrennCounterexampleArtifactError(
            "campaign certificate failed semantic replay"
        )
    return LoadedCounterexampleBundle(
        directory=directory,
        support_summary=support,
        numerical_summary=numerical,
        certificate=certificate,
        exact_report=exact_report,
    )


__all__ = (
    "CAMPAIGN_CERTIFICATE_SCHEMA",
    "CAMPAIGN_MANIFEST_SCHEMA",
    "KrennCounterexampleArtifactError",
    "LoadedCounterexampleBundle",
    "NUMERICAL_SUMMARY_SCHEMA",
    "SUPPORT_SUMMARY_SCHEMA",
    "build_numerical_summary",
    "build_support_summary",
    "numerical_candidate_record",
    "verify_counterexample_bundle",
    "verify_numerical_candidate_record",
    "verify_numerical_summary",
    "verify_support_summary",
    "write_counterexample_bundle",
)
