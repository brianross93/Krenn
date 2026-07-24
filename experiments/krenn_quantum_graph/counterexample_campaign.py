"""Command-line driver for the bounded ``n=6,d=3`` counterexample campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


WORKER_THREAD_LIMITS = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
for _thread_environment_name, _thread_environment_value in (
    WORKER_THREAD_LIMITS.items()
):
    # This executes before importing NumPy-bearing campaign modules and again
    # when Windows spawn imports this module inside each worker.
    os.environ[_thread_environment_name] = _thread_environment_value

from experiments.krenn_quantum_graph.counterexample_artifacts import (
    build_numerical_summary,
    build_support_summary,
    numerical_candidate_record,
    verify_counterexample_bundle,
    write_counterexample_bundle,
)
from experiments.krenn_quantum_graph.counterexample_exact import (
    ExactVerificationReport,
    ReconstructionBounds,
    reconstruct_complex_candidate,
)
from experiments.krenn_quantum_graph.counterexample_search import (
    AMBIENT_VARIABLES,
    MAX_DISCOVERED_SUPPORTS,
    MAX_WORKERS,
    SupportCampaignPlan,
    SupportCampaignResult,
    default_support_seeds,
    greedy_pruned_omission_seed,
    run_support_campaign,
)
from experiments.krenn_quantum_graph.numerical_continuation import (
    ContinuationPlan,
    ContinuationResult,
    GHZ_COLOR_DIAGONAL_GAUGE,
    LMPlan,
    NumericalJobSpec,
    NumericalSolveResult,
    deterministic_job_specs,
    natural_gauge_chart,
    run_numerical_job,
)
from experiments.krenn_quantum_graph.support_extension import (
    natural_support,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT
    / "results/krenn_quantum_graph/n6_d3_counterexample_search"
)
DEFAULT_SCRATCH = Path(r"D:\KrennScratch\counterexample_search")
MASTER_SEED = 60_320_260_724
BASELINE_COMMIT = "ab446724a61a293a3268c63b9e9e721c6a360561"
CAMPAIGN_LEDGER_SCHEMA = "krenn-n6-d3-disposable-campaign-ledger-v1"


class KrennCounterexampleCampaignError(RuntimeError):
    """A campaign budget, worker result, or output request is malformed."""


def _enforce_worker_thread_limits() -> dict[str, str]:
    for name, value in WORKER_THREAD_LIMITS.items():
        os.environ[name] = value
    return dict(WORKER_THREAD_LIMITS)


def _command_line(arguments: Sequence[str]) -> str:
    return subprocess.list2cmdline(tuple(map(str, arguments)))


def _radius_scratch_name(index: int, radius: float) -> str:
    value = format(float(radius), ".17g")
    return f"radius-{int(index):03d}-{value}"


def _campaign_argv_from_namespace(
    args: argparse.Namespace,
    radii: Sequence[float],
    schedule: Sequence[float],
) -> tuple[str, ...]:
    return (
        "--scratch-directory",
        str(args.scratch_directory),
        "--results-directory",
        str(args.results_directory),
        "--workers",
        str(args.workers),
        "--orbit-nodes",
        str(args.orbit_nodes),
        "--support-state-cap",
        str(args.support_state_cap),
        "--retained-supports",
        str(args.retained_supports),
        "--omission-supports",
        str(args.omission_supports),
        "--starts-per-support",
        str(args.starts_per_support),
        "--dense-starts",
        str(args.dense_starts),
        "--radii",
        *tuple(format(value, ".17g") for value in radii),
        "--iterations",
        str(args.iterations),
        "--evaluations",
        str(args.evaluations),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--continuation-schedule",
        *tuple(format(value, ".17g") for value in schedule),
        "--best-per-radius-class",
        str(args.best_per_radius_class),
        "--reconstruction-trigger",
        format(float(args.reconstruction_trigger), ".17g"),
    )


def _validated_campaign_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, tuple[float, ...], tuple[float, ...]]:
    try:
        scratch = Path(args.scratch_directory).resolve()
        results_directory = Path(args.results_directory).resolve()
        radii = tuple(map(float, args.radii))
        schedule = tuple(map(float, args.continuation_schedule))
        integer_controls = {
            "workers": int(args.workers),
            "orbit_nodes": int(args.orbit_nodes),
            "support_state_cap": int(args.support_state_cap),
            "retained_supports": int(args.retained_supports),
            "omission_supports": int(args.omission_supports),
            "starts_per_support": int(args.starts_per_support),
            "dense_starts": int(args.dense_starts),
            "iterations": int(args.iterations),
            "evaluations": int(args.evaluations),
            "checkpoint_interval": int(args.checkpoint_interval),
            "best_per_radius_class": int(args.best_per_radius_class),
        }
        reconstruction_trigger = float(args.reconstruction_trigger)
    except (AttributeError, TypeError, ValueError, OSError) as error:
        raise KrennCounterexampleCampaignError(
            "campaign arguments could not be normalized"
        ) from error

    if not 1 <= integer_controls["workers"] <= MAX_WORKERS:
        raise KrennCounterexampleCampaignError(
            "worker count must be between 1 and 16"
        )
    if not 1 <= integer_controls["orbit_nodes"] <= 1_000_000:
        raise KrennCounterexampleCampaignError(
            "orbit node cap is outside the safety bound"
        )
    if not (
        1
        <= integer_controls["support_state_cap"]
        <= MAX_DISCOVERED_SUPPORTS
    ):
        raise KrennCounterexampleCampaignError(
            "support state cap is outside the safety bound"
        )
    if not 1 <= integer_controls["retained_supports"] <= 10_000:
        raise KrennCounterexampleCampaignError(
            "retained support count is outside the safety bound"
        )
    if not 1 <= integer_controls["omission_supports"] <= 45:
        raise KrennCounterexampleCampaignError(
            "omission support count must be between 1 and 45"
        )
    if not 1 <= integer_controls["starts_per_support"] <= 1_000:
        raise KrennCounterexampleCampaignError(
            "sparse multistart count is outside the safety bound"
        )
    if not 1 <= integer_controls["dense_starts"] <= 1_000:
        raise KrennCounterexampleCampaignError(
            "dense comparison requires between 1 and 1000 starts"
        )
    if not 1 <= integer_controls["iterations"] <= 10_000:
        raise KrennCounterexampleCampaignError(
            "iteration cap is outside the safety bound"
        )
    if not 2 <= integer_controls["evaluations"] <= 100_000:
        raise KrennCounterexampleCampaignError(
            "evaluation cap is outside the safety bound"
        )
    if not (
        0
        <= integer_controls["checkpoint_interval"]
        <= integer_controls["iterations"]
    ):
        raise KrennCounterexampleCampaignError(
            "checkpoint interval is outside the iteration cap"
        )
    if not 1 <= integer_controls["best_per_radius_class"] <= 1_000:
        raise KrennCounterexampleCampaignError(
            "best-candidate retention count is outside the safety bound"
        )
    if (
        not math.isfinite(reconstruction_trigger)
        or reconstruction_trigger <= 0.0
    ):
        raise KrennCounterexampleCampaignError(
            "reconstruction trigger must be finite and positive"
        )
    dense_gauge_rank = natural_gauge_chart(
        tuple(range(AMBIENT_VARIABLES)),
        action_group=GHZ_COLOR_DIAGONAL_GAUGE,
    ).rank
    if (
        not radii
        or len(radii) != len(set(radii))
        or any(
            radii[index + 1] <= radii[index]
            for index in range(len(radii) - 1)
        )
        or any(
            not math.isfinite(radius)
            or radius < 1.0
            or radius * radius + 1.0e-14 < dense_gauge_rank
            for radius in radii
        )
    ):
        raise KrennCounterexampleCampaignError(
            "norm radii must be strictly increasing, finite, and large "
            "enough for the fixed dense gauge anchors"
        )
    if (
        len(schedule) < 2
        or schedule[0] <= 0.0
        or any(not math.isfinite(value) or value < 0.0 for value in schedule)
        or any(
            schedule[index + 1] >= schedule[index]
            for index in range(len(schedule) - 1)
        )
        or schedule[-1] != 0.0
    ):
        raise KrennCounterexampleCampaignError(
            "continuation schedule must strictly decrease to zero"
        )

    scratch_root = DEFAULT_SCRATCH.resolve()
    results_root = (ROOT / "results").resolve()
    if (
        "onedrive" in str(scratch).casefold()
        or not scratch.is_relative_to(scratch_root)
    ):
        raise KrennCounterexampleCampaignError(
            "campaign scratch must stay under "
            r"D:\KrennScratch\counterexample_search"
        )
    if (
        "onedrive" in str(results_directory).casefold()
        or not results_directory.is_relative_to(results_root)
    ):
        raise KrennCounterexampleCampaignError(
            "campaign results must stay under this repository's results tree"
        )
    for name, value in integer_controls.items():
        setattr(args, name, value)
    args.radii = radii
    args.continuation_schedule = schedule
    args.reconstruction_trigger = reconstruction_trigger
    return scratch, results_directory, radii, schedule


def _canonical_json_bytes(value: Mapping) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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


def _final_step(
    result: NumericalSolveResult | ContinuationResult,
) -> NumericalSolveResult:
    if isinstance(result, NumericalSolveResult):
        return result
    if isinstance(result, ContinuationResult) and result.final_step is not None:
        return result.final_step
    raise KrennCounterexampleCampaignError(
        "a numerical worker returned no final step"
    )


def _worker(
    task: tuple[
        NumericalJobSpec,
        LMPlan,
        tuple[float, ...],
        str,
    ],
) -> tuple[NumericalJobSpec, NumericalSolveResult | ContinuationResult]:
    spec, lm_plan, schedule, scratch = task
    _enforce_worker_thread_limits()
    continuation = ContinuationPlan(
        target_schedule=schedule,
        lm_plan=lm_plan,
        direct_final_polish=True,
        continue_after_nonconvergence=True,
    )
    result = run_numerical_job(
        spec,
        lm_plan=lm_plan,
        continuation_plan=continuation,
        scratch_directory=scratch,
        resume=True,
    )
    return spec, result


def _select_numerical_supports(
    support_results: Sequence[SupportCampaignResult],
) -> tuple[tuple[int, ...], ...]:
    selected: list[tuple[int, ...]] = []
    seen_orbits: set[tuple[int, ...]] = set()
    natural = set(natural_support())
    for result in support_results:
        for candidate in result.candidates:
            representative = candidate.orbit_representative
            if representative in seen_orbits:
                continue
            seen_orbits.add(representative)
            # The known Laurent initializer is expressed in the natural seed
            # orientation. Preserve that preferred orientation, while all
            # non-Laurent strata use the literal canonical representative.
            selected.append(
                candidate.support
                if natural.issubset(candidate.support)
                else representative
            )
    return tuple(selected)


def _support_campaigns(
    scratch: Path,
    *,
    worker_count: int,
    orbit_node_cap: int,
    discovered_support_cap: int,
    retained_candidate_cap: int,
    omission_seed_count: int,
) -> tuple[tuple[SupportCampaignResult, ...], tuple[tuple[int, ...], ...]]:
    if not 1 <= omission_seed_count <= 45:
        raise KrennCounterexampleCampaignError(
            "support campaign requires between 1 and 45 omission strata"
        )
    seeds = default_support_seeds(
        include_two_coordinate_omissions=False
    )
    retained_roots = tuple(
        seed for seed in seeds
        if seed.family == "retains-natural-size21"
    )
    retained_per_size_quota = max(
        1, math.ceil(retained_candidate_cap / 3)
    )
    retained_plan = SupportCampaignPlan(
        node_cap=max(2_000, retained_candidate_cap * 64),
        support_cap=24,
        candidate_cap=retained_candidate_cap,
        per_support_size_candidate_quota=retained_per_size_quota,
        discovered_support_cap=discovered_support_cap,
        omission_candidate_quota=0,
        activation_width=8,
        worker_count=worker_count,
        deterministic_seed=MASTER_SEED,
        scratch_directory=str(scratch),
        include_two_coordinate_omissions=False,
    )
    retained = run_support_campaign(
        retained_plan, seeds=retained_roots
    )

    orbit_roots = tuple(
        seed for seed in seeds
        if seed.family == "diagonal-seed-orbit"
    )
    orbit_plan = SupportCampaignPlan(
        node_cap=orbit_node_cap,
        support_cap=28,
        candidate_cap=8,
        per_support_size_candidate_quota=2,
        discovered_support_cap=discovered_support_cap,
        omission_candidate_quota=8,
        activation_width=8,
        worker_count=worker_count,
        deterministic_seed=MASTER_SEED,
        scratch_directory=str(scratch),
        include_two_coordinate_omissions=False,
    )
    orbit = run_support_campaign(orbit_plan, seeds=orbit_roots)

    natural = natural_support()
    omission_patterns = [
        (natural[index],)
        for index in range(min(omission_seed_count, len(natural)))
    ]
    if omission_seed_count > len(omission_patterns):
        for left in range(len(natural)):
            for right in range(left + 1, len(natural)):
                omission_patterns.append(
                    (natural[left], natural[right])
                )
                if len(omission_patterns) >= omission_seed_count:
                    break
            if len(omission_patterns) >= omission_seed_count:
                break
    pruned_roots = tuple(
        greedy_pruned_omission_seed(
            pattern,
            deterministic_seed=MASTER_SEED + index,
            name=f"greedy-pruned-omission-{index:02d}",
        )
        for index, pattern in enumerate(omission_patterns)
    )
    pruned_cap = max(map(lambda seed: len(seed.support), pruned_roots))
    pruned_plan = SupportCampaignPlan(
        node_cap=len(pruned_roots),
        support_cap=pruned_cap,
        candidate_cap=len(pruned_roots),
        per_support_size_candidate_quota=len(pruned_roots),
        discovered_support_cap=discovered_support_cap,
        omission_candidate_quota=len(pruned_roots),
        activation_width=1,
        worker_count=worker_count,
        deterministic_seed=MASTER_SEED,
        scratch_directory=str(scratch),
        include_two_coordinate_omissions=False,
    )
    pruned = run_support_campaign(
        pruned_plan, seeds=pruned_roots
    )
    selected = _select_numerical_supports((retained, orbit, pruned))
    return (retained, orbit, pruned), selected


def _result_key(
    pair: tuple[NumericalJobSpec, NumericalSolveResult | ContinuationResult],
) -> tuple:
    spec, result = pair
    final = _final_step(result)
    residual = final.residual_metrics.linf
    return (
        math.inf if residual is None else residual,
        final.weight_metrics.on_hard_boundary,
        (
            math.inf
            if final.weight_metrics.l2 is None
            else final.weight_metrics.l2
        ),
        spec.run_id,
    )


def _summary_row(
    radius: float,
    spec: NumericalJobSpec,
    result: NumericalSolveResult | ContinuationResult,
) -> dict:
    final = _final_step(result)
    return {
        "radius": radius,
        "run_id": spec.run_id,
        "dense": spec.dense,
        "support_size": len(spec.support),
        "initializer_kind": spec.initializer_kind,
        "use_continuation": spec.use_continuation,
        "status": final.status,
        "iterations": final.iterations,
        "evaluations": final.evaluations,
        "residual_linf": final.residual_metrics.linf,
        "residual_l2": final.residual_metrics.l2,
        "weights_finite": final.weight_metrics.finite,
        "weight_l2": final.weight_metrics.l2,
        "weight_linf": final.weight_metrics.linf,
        "hard_boundary_active": (
            final.weight_metrics.on_hard_boundary
        ),
        "known_Q_absolute_value": (
            final.invariant_metrics.known_q_absolute_value
        ),
        "numerical_zero": final.numerical_zero,
        "exact_solution_certified": False,
    }


def _select_best(
    rows: Sequence[
        tuple[float, NumericalJobSpec, NumericalSolveResult | ContinuationResult]
    ],
    *,
    per_radius_and_class: int,
) -> tuple[
    tuple[float, NumericalJobSpec, NumericalSolveResult | ContinuationResult],
    ...,
]:
    selected = []
    radii = sorted({radius for radius, _spec, _result in rows})
    natural = set(natural_support())
    for radius in radii:
        for candidate_class in (
            "natural-retaining-sparse",
            "natural-omitting-sparse",
            "dense",
        ):
            subset = [
                (spec, result)
                for candidate_radius, spec, result in rows
                if candidate_radius == radius
                and (
                    "dense"
                    if spec.dense
                    else "natural-retaining-sparse"
                    if natural.issubset(spec.support)
                    else "natural-omitting-sparse"
                )
                == candidate_class
            ]
            subset.sort(key=_result_key)
            selected.extend(
                (radius, spec, result)
                for spec, result in subset[:per_radius_and_class]
            )
    return tuple(selected)


def _reconstruct_if_triggered(
    result: NumericalSolveResult | ContinuationResult,
    *,
    trigger: float,
) -> tuple[str, ExactVerificationReport | None, Mapping | None]:
    final = _final_step(result)
    residual = final.residual_metrics.linf
    if (
        residual is None
        or residual > trigger
        or not final.weight_metrics.finite
    ):
        return "not-attempted", None, None
    report = reconstruct_complex_candidate(
        final.dense_weights(),
        n=6,
        d=3,
        support=final.support,
        bounds=ReconstructionBounds(
            absolute_tolerance=max(trigger, 1.0e-10),
            relative_tolerance=max(trigger, 1.0e-10),
            zero_tolerance=1.0e-10,
            max_denominator=1_024,
            max_numerator=4_096,
            coefficient_grid_denominator=8,
            coefficient_grid_numerator=12,
            max_polynomial_checks=4_000,
            max_field_attempts=96,
        ),
        preserve_support=False,
    )
    if report.exact and report.exact_attempts[0].verification is not None:
        return (
            "exact-dual-verification-passed",
            report.exact_attempts[0].verification,
            report.to_dict(),
        )
    return (
        "attempted-no-exact-reconstruction",
        None,
        report.to_dict(),
    )


def run_campaign(
    args: argparse.Namespace,
    *,
    argv_receipt: Sequence[str] | None = None,
) -> Path:
    scratch, results_directory, radii, schedule = (
        _validated_campaign_inputs(args)
    )
    effective_argv = (
        tuple(map(str, argv_receipt))
        if argv_receipt is not None
        else _campaign_argv_from_namespace(args, radii, schedule)
    )
    thread_limits = _enforce_worker_thread_limits()
    scratch.mkdir(parents=True, exist_ok=True)
    support_results, selected_supports = _support_campaigns(
        scratch,
        worker_count=args.workers,
        orbit_node_cap=args.orbit_nodes,
        discovered_support_cap=args.support_state_cap,
        retained_candidate_cap=args.retained_supports,
        omission_seed_count=args.omission_supports,
    )

    specs = deterministic_job_specs(
        selected_supports,
        master_seed=MASTER_SEED,
        starts_per_support=args.starts_per_support,
        include_dense=True,
        dense_starts=args.dense_starts,
        activation_amplitudes=(0.03, 0.1, 0.3, 1.0),
    )
    all_results: list[
        tuple[
            float,
            NumericalJobSpec,
            NumericalSolveResult | ContinuationResult,
        ]
    ] = []
    for radius_index, radius in enumerate(radii):
        plan = LMPlan(
            max_iterations=args.iterations,
            max_evaluations=args.evaluations,
            l2_bound=radius,
            linf_bound=radius,
            checkpoint_interval=args.checkpoint_interval,
        )
        radius_name = _radius_scratch_name(radius_index, radius)
        radius_scratch = scratch / radius_name
        tasks = tuple(
            (spec, plan, schedule, str(radius_scratch))
            for spec in specs
        )
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            completed = tuple(executor.map(_worker, tasks))
        all_results.extend(
            (radius, spec, result) for spec, result in completed
        )
        _write_json_atomic(
            scratch / f"{radius_name}-summary.json",
            {
                "schema": CAMPAIGN_LEDGER_SCHEMA,
                "radius": radius,
                "results": [
                    _summary_row(radius, spec, result)
                    for spec, result in completed
                ],
            },
        )

    selected = _select_best(
        all_results,
        per_radius_and_class=args.best_per_radius_class,
    )
    exact_report: ExactVerificationReport | None = None
    exact_reconstruction_ledgers = []
    best_records = []
    for radius, spec, result in selected:
        status, candidate_exact, reconstruction = (
            _reconstruct_if_triggered(
                result, trigger=args.reconstruction_trigger
            )
        )
        if reconstruction is not None:
            exact_reconstruction_ledgers.append(
                {
                    "radius": radius,
                    "run_id": spec.run_id,
                    "reconstruction": reconstruction,
                }
            )
        if candidate_exact is not None and exact_report is None:
            exact_report = candidate_exact
        best_records.append(
            numerical_candidate_record(
                result,
                spec,
                reconstruction_status=status,
            )
        )

    support_summary = build_support_summary(
        support_results,
        selected_supports=selected_supports,
    )
    invocation_arguments = (
        sys.executable,
        "-B",
        "-m",
        "experiments.krenn_quantum_graph.counterexample_campaign",
        *effective_argv,
    )
    invocation = _command_line(invocation_arguments)
    verify_arguments = (
        sys.executable,
        "-B",
        "-m",
        "experiments.krenn_quantum_graph.counterexample_campaign",
        "--verify-only",
        "--results-directory",
        str(results_directory),
    )
    numerical_summary = build_numerical_summary(
        master_seed=MASTER_SEED,
        worker_count=args.workers,
        radii=tuple((radius, radius) for radius in radii),
        jobs_scheduled=len(all_results),
        jobs_completed=len(all_results),
        best_candidates=best_records,
        command_lines=(
            invocation,
            _command_line(verify_arguments),
        ),
    )
    execution = {
        "branch": "codex/counterexample-search",
        "baseline_origin_main": BASELINE_COMMIT,
        "master_seed": MASTER_SEED,
        "workers": args.workers,
        "threads_per_worker": 1,
        "worker_thread_environment": thread_limits,
        "argv_receipt": (
            list(effective_argv)
        ),
        "support_budgets": {
            "orbit_node_cap": args.orbit_nodes,
            "discovered_support_cap_per_run": args.support_state_cap,
            "retained_support_candidates": args.retained_supports,
            "retained_candidates_per_support_size": max(
                1, math.ceil(args.retained_supports / 3)
            ),
            "greedy_pruned_omission_supports": args.omission_supports,
        },
        "support_orientation_policy": {
            "orbit_deduplication_key": (
                "lexicographically-least-S6xS3-support-image"
            ),
            "natural_laurent_stratum": (
                "preferred-natural-containing-orientation"
            ),
            "non_laurent_strata": "literal-canonical-orbit-representative",
            "symmetry_weight_equalities": 0,
        },
        "numerical_budgets": {
            "radii": list(radii),
            "starts_per_sparse_support": args.starts_per_support,
            "dense_starts_per_radius": args.dense_starts,
            "iterations_per_solve": args.iterations,
            "evaluations_per_solve": args.evaluations,
            "continuation_schedule": list(schedule),
            "checkpoint_interval": args.checkpoint_interval,
            "continue_after_nonconvergence": True,
            "resume_from_checkpoint": True,
        },
        "scratch_directory": str(scratch),
        "scratch_data_committed": False,
        "large_checkpoints_required_for_bundle_replay": False,
    }
    certificate_path = write_counterexample_bundle(
        results_directory,
        support_summary=support_summary,
        numerical_summary=numerical_summary,
        execution=execution,
        exact_report=exact_report,
    )
    _write_json_atomic(
        scratch / "campaign-ledger.json",
        {
            "schema": CAMPAIGN_LEDGER_SCHEMA,
            "execution": execution,
            "all_results": [
                _summary_row(radius, spec, result)
                for radius, spec, result in all_results
            ],
            "selected_run_ids": [
                {
                    "radius": radius,
                    "run_id": spec.run_id,
                }
                for radius, spec, _result in selected
            ],
            "exact_reconstruction_ledgers": (
                exact_reconstruction_ledgers
            ),
            "exact_candidate_found": exact_report is not None,
        },
    )
    return certificate_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the bounded deterministic n=6,d=3 "
            "Krenn counterexample campaign."
        )
    )
    parser.add_argument(
        "--scratch-directory",
        default=str(DEFAULT_SCRATCH),
    )
    parser.add_argument(
        "--results-directory",
        default=str(DEFAULT_RESULTS),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--orbit-nodes", type=int, default=40_000)
    parser.add_argument(
        "--support-state-cap", type=int, default=2_000_000
    )
    parser.add_argument("--retained-supports", type=int, default=6)
    parser.add_argument("--omission-supports", type=int, default=4)
    parser.add_argument("--starts-per-support", type=int, default=2)
    parser.add_argument("--dense-starts", type=int, default=4)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=(8.0, 16.0, 32.0),
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--evaluations", type=int, default=450)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--continuation-schedule",
        type=float,
        nargs="+",
        default=(1.0, 0.5, 0.25, 0.125, 0.0),
    )
    parser.add_argument(
        "--best-per-radius-class", type=int, default=2
    )
    parser.add_argument(
        "--reconstruction-trigger", type=float, default=1.0e-8
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    if args.verify_only:
        results_directory = Path(args.results_directory).resolve()
        if "onedrive" in str(results_directory).casefold():
            raise KrennCounterexampleCampaignError(
                "verified results cannot be read from OneDrive"
            )
        loaded = verify_counterexample_bundle(results_directory)
        print(
            json.dumps(
                {
                    "status": loaded.certificate["status"],
                    "exact_counterexample": loaded.exact_counterexample,
                    "directory": str(loaded.directory),
                },
                sort_keys=True,
            )
        )
        return 0
    certificate = run_campaign(args, argv_receipt=raw_argv)
    loaded = verify_counterexample_bundle(certificate.parent)
    print(
        json.dumps(
            {
                "certificate": str(certificate),
                "status": loaded.certificate["status"],
                "exact_counterexample": loaded.exact_counterexample,
                "jobs_completed": loaded.certificate[
                    "numerical_jobs_completed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
