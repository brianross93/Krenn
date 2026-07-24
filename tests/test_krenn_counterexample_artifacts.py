import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.counterexample_artifacts import (
    KrennCounterexampleArtifactError,
    build_numerical_summary,
    build_support_summary,
    numerical_candidate_record,
    verify_counterexample_bundle,
    verify_numerical_candidate_record,
    write_counterexample_bundle,
)
from experiments.krenn_quantum_graph.counterexample_exact import (
    ExactCandidateWitness,
    rational_field,
    verify_exact_candidate,
)
from experiments.krenn_quantum_graph.counterexample_search import (
    SupportCampaignPlan,
    SupportSeed,
    run_support_campaign,
)
from experiments.krenn_quantum_graph.numerical_continuation import (
    LMPlan,
    deterministic_job_specs,
    run_numerical_job,
)
from experiments.krenn_quantum_graph.support_extension import (
    size21_support_audits,
)
from experiments.krenn_quantum_graph.ternary_search import (
    n6_d3_seed_witness,
)


class KrennCounterexampleArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = SupportSeed(
            name="stored-size21",
            family="retains-natural-size21",
            support=size21_support_audits()[0].support,
        )
        support_plan = SupportCampaignPlan(
            node_cap=2,
            support_cap=22,
            candidate_cap=1,
            omission_candidate_quota=0,
            activation_width=1,
            worker_count=1,
            deterministic_seed=60320260724,
            scratch_directory=r"D:\KrennScratch\counterexample_search",
            include_two_coordinate_omissions=False,
        )
        cls.support_result = run_support_campaign(
            support_plan, seeds=(seed,)
        )
        cls.emitted_support = cls.support_result.candidates[0].support
        cls.support = cls.support_result.candidates[
            0
        ].orbit_representative
        jobs = deterministic_job_specs(
            (cls.support,),
            master_seed=60320260724,
            starts_per_support=2,
            include_dense=True,
            dense_starts=1,
        )
        cls.job = replace(jobs[1], use_continuation=False)
        cls.lm_plan = LMPlan(
            max_iterations=1,
            max_evaluations=20,
            l2_bound=8,
            linf_bound=8,
            checkpoint_interval=0,
        )
        cls.solve = run_numerical_job(cls.job, lm_plan=cls.lm_plan)
        cls.record = numerical_candidate_record(
            cls.solve,
            cls.job,
            reconstruction_status="not-attempted",
        )
        cls.support_summary = build_support_summary(
            (cls.support_result,),
            selected_supports=(cls.support,),
        )
        continuation_targets = tuple(
            row["target_amplitude"][0]
            for row in cls.record["continuation_growth"]
        )
        schedule = (
            continuation_targets[:-1]
            if len(continuation_targets) >= 2
            and continuation_targets[-2:] == (0.0, 0.0)
            else continuation_targets
        )
        results_directory = (
            Path(__file__).resolve().parents[1]
            / "results/krenn_quantum_graph/focused-artifact-test"
        )
        argv = (
            "--scratch-directory",
            support_plan.scratch_directory,
            "--results-directory",
            str(results_directory),
            "--workers",
            "1",
            "--orbit-nodes",
            str(support_plan.node_cap),
            "--support-state-cap",
            str(support_plan.discovered_support_cap),
            "--retained-supports",
            str(support_plan.candidate_cap),
            "--omission-supports",
            "1",
            "--starts-per-support",
            "2",
            "--dense-starts",
            "1",
            "--radii",
            "8",
            "--iterations",
            "1",
            "--evaluations",
            "20",
            "--checkpoint-interval",
            "0",
            "--continuation-schedule",
            *(format(value, ".17g") for value in schedule),
            "--best-per-radius-class",
            "1",
            "--reconstruction-trigger",
            "1e-8",
        )
        invocation = (
            "python -B -m "
            "experiments.krenn_quantum_graph.counterexample_campaign "
            + " ".join(argv)
        )
        verification = (
            "python -B -m "
            "experiments.krenn_quantum_graph.counterexample_campaign "
            f"--verify-only --results-directory {results_directory}"
        )
        cls.numerical_summary = build_numerical_summary(
            master_seed=60320260724,
            worker_count=1,
            radii=((8, 8),),
            jobs_scheduled=len(jobs),
            jobs_completed=len(jobs),
            best_candidates=(cls.record,),
            command_lines=(invocation, verification),
        )
        cls.execution = {
            "branch": "codex/counterexample-search",
            "baseline_origin_main": (
                "ab446724a61a293a3268c63b9e9e721c6a360561"
            ),
            "master_seed": 60320260724,
            "workers": 1,
            "threads_per_worker": 1,
            "worker_thread_environment": {
                "BLIS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
            "argv_receipt": list(argv),
            "support_budgets": {
                "orbit_node_cap": support_plan.node_cap,
                "discovered_support_cap_per_run": (
                    support_plan.discovered_support_cap
                ),
                "retained_support_candidates": (
                    support_plan.candidate_cap
                ),
                "retained_candidates_per_support_size": (
                    support_plan.per_support_size_candidate_quota
                ),
                "greedy_pruned_omission_supports": 1,
            },
            "support_orientation_policy": {
                "orbit_deduplication_key": (
                    "lexicographically-least-S6xS3-support-image"
                ),
                "natural_laurent_stratum": (
                    "preferred-natural-containing-orientation"
                ),
                "non_laurent_strata": (
                    "literal-canonical-orbit-representative"
                ),
                "symmetry_weight_equalities": 0,
            },
            "numerical_budgets": {
                "radii": [8.0],
                "starts_per_sparse_support": 2,
                "dense_starts_per_radius": 1,
                "iterations_per_solve": 1,
                "evaluations_per_solve": 20,
                "continuation_schedule": list(schedule),
                "checkpoint_interval": 0,
                "continue_after_nonconvergence": True,
                "resume_from_checkpoint": True,
            },
            "scratch_directory": support_plan.scratch_directory,
            "scratch_data_committed": False,
            "large_checkpoints_required_for_bundle_replay": False,
        }

    def test_selected_record_replays_all_729_residuals(self):
        replay = verify_numerical_candidate_record(self.record)
        self.assertEqual(replay, self.record)
        self.assertEqual(len(replay["all_729_residuals"]), 729)
        self.assertFalse(replay["exact_dual_verification_passed"])
        self.assertFalse(
            replay["claim_boundary"][
                "bounded_miss_proves_nonexistence"
            ]
        )

    def test_solver_and_continuation_tampering_fail_closed(self):
        mutations = (
            ("unknown solver status", "solver_status", "fabricated"),
            ("negative iterations", "solver_iterations", -1),
            ("zero evaluations", "solver_evaluations", 0),
        )
        for label, key, value in mutations:
            with self.subTest(label=label):
                payload = copy.deepcopy(self.record)
                payload[key] = value
                with self.assertRaises(KrennCounterexampleArtifactError):
                    verify_numerical_candidate_record(payload)

        payload = copy.deepcopy(self.record)
        payload["continuation_growth"] = [{"fabricated": True}]
        with self.assertRaises(KrennCounterexampleArtifactError):
            verify_numerical_candidate_record(payload)

        payload = copy.deepcopy(self.record)
        payload["continuation_growth"][-1]["residual_linf"] += 1.0
        with self.assertRaises(KrennCounterexampleArtifactError):
            verify_numerical_candidate_record(payload)

        payload = copy.deepcopy(self.record)
        payload["solver_status"] = "evaluation-cap-reached"
        with self.assertRaises(KrennCounterexampleArtifactError):
            verify_numerical_candidate_record(payload)

        payload = copy.deepcopy(self.record)
        payload["continuation_growth"].append(
            copy.deepcopy(payload["continuation_growth"][-1])
        )
        with self.assertRaises(KrennCounterexampleArtifactError):
            verify_numerical_candidate_record(payload)

    def test_bounded_miss_bundle_round_trips_without_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_counterexample_bundle(
                directory,
                support_summary=self.support_summary,
                numerical_summary=self.numerical_summary,
                execution=self.execution,
            )
            loaded = verify_counterexample_bundle(directory)
            self.assertFalse(loaded.exact_counterexample)
            self.assertEqual(
                loaded.certificate["status"],
                "bounded-search-complete-no-exact-candidate",
            )
            self.assertFalse(
                loaded.certificate["claim_boundary"][
                    "bounded_miss_proves_nonexistence"
                ]
            )
            self.assertFalse(
                loaded.certificate["claim_boundary"][
                    "krenn_gu_conjecture_refuted"
                ]
            )
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {
                    "support_search.json",
                    "numerical_search.json",
                    "certificate.json",
                    "manifest.json",
                },
            )

    def test_incomplete_or_empty_campaign_is_not_labeled_complete(self):
        incomplete = build_numerical_summary(
            master_seed=60320260724,
            worker_count=1,
            radii=((8, 8),),
            jobs_scheduled=3,
            jobs_completed=1,
            best_candidates=(self.record,),
            command_lines=self.numerical_summary["command_lines"],
        )
        empty = build_numerical_summary(
            master_seed=60320260724,
            worker_count=1,
            radii=((8, 8),),
            jobs_scheduled=0,
            jobs_completed=0,
            best_candidates=(),
            command_lines=self.numerical_summary["command_lines"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_counterexample_bundle(
                directory,
                support_summary=self.support_summary,
                numerical_summary=incomplete,
                execution=self.execution,
            )
            loaded = verify_counterexample_bundle(directory)
            self.assertEqual(
                loaded.certificate["status"],
                "bounded-search-incomplete-no-exact-candidate",
            )
            self.assertFalse(
                loaded.certificate["claim_boundary"][
                    "bounded_miss_proves_nonexistence"
                ]
            )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(KrennCounterexampleArtifactError):
                write_counterexample_bundle(
                    Path(temporary) / "bundle",
                    support_summary=self.support_summary,
                    numerical_summary=empty,
                    execution=self.execution,
                )

    def test_selected_candidate_must_match_summary_seed_and_radius(self):
        cases = (
            {
                "master_seed": 60320260725,
                "radii": ((8, 8),),
                "jobs_scheduled": 1,
                "jobs_completed": 1,
                "best_candidates": (self.record,),
            },
            {
                "master_seed": 60320260724,
                "radii": ((16, 16),),
                "jobs_scheduled": 1,
                "jobs_completed": 1,
                "best_candidates": (self.record,),
            },
            {
                "master_seed": 60320260724,
                "radii": ((8, 8),),
                "jobs_scheduled": 2,
                "jobs_completed": 2,
                "best_candidates": (self.record, self.record),
            },
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(KrennCounterexampleArtifactError):
                    build_numerical_summary(
                        worker_count=1,
                        command_lines=("python -m bounded-test",),
                        **case,
                    )

    def test_raw_and_refreshed_hash_corruption_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_counterexample_bundle(
                directory,
                support_summary=self.support_summary,
                numerical_summary=self.numerical_summary,
                execution=self.execution,
            )
            numerical_path = directory / "numerical_search.json"
            numerical_path.write_bytes(
                numerical_path.read_bytes() + b" "
            )
            with self.assertRaises(KrennCounterexampleArtifactError):
                verify_counterexample_bundle(directory)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_counterexample_bundle(
                directory,
                support_summary=self.support_summary,
                numerical_summary=self.numerical_summary,
                execution=self.execution,
            )
            numerical_path = directory / "numerical_search.json"
            numerical = json.loads(
                numerical_path.read_text(encoding="utf-8")
            )
            numerical["best_candidates"][0][
                "all_729_residuals"
            ][0][0] += 1.0
            encoded = (
                json.dumps(
                    numerical,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            numerical_path.write_bytes(encoded)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            record = next(
                row for row in manifest["artifacts"]
                if row["path"] == "numerical_search.json"
            )
            record["bytes"] = len(encoded)
            record["sha256"] = hashlib.sha256(encoded).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennCounterexampleArtifactError):
                verify_counterexample_bundle(directory)

    def test_manifest_inventory_must_be_exact_unique_and_sorted(self):
        for mutation in ("reordered", "duplicate", "extra"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary) / "bundle"
                    write_counterexample_bundle(
                        directory,
                        support_summary=self.support_summary,
                        numerical_summary=self.numerical_summary,
                        execution=self.execution,
                    )
                    manifest_path = directory / "manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    if mutation == "reordered":
                        manifest["artifacts"].reverse()
                    elif mutation == "duplicate":
                        manifest["artifacts"].append(
                            copy.deepcopy(manifest["artifacts"][0])
                        )
                    else:
                        extra_path = directory / "extra.json"
                        extra_path.write_bytes(b"{}\n")
                        manifest["artifacts"].append(
                            {
                                "path": extra_path.name,
                                "bytes": extra_path.stat().st_size,
                                "sha256": hashlib.sha256(
                                    extra_path.read_bytes()
                                ).hexdigest(),
                            }
                        )
                        manifest["artifacts"].sort(
                            key=lambda row: row["path"]
                        )
                    manifest_path.write_text(
                        json.dumps(
                            manifest,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(
                        KrennCounterexampleArtifactError
                    ):
                        verify_counterexample_bundle(directory)

    def test_execution_receipt_tampering_fails_before_write(self):
        cases = []
        for label, mutate in (
            (
                "baseline",
                lambda value: value.__setitem__(
                    "baseline_origin_main", "0" * 40
                ),
            ),
            (
                "worker",
                lambda value: value.__setitem__("workers", 2),
            ),
            (
                "scratch",
                lambda value: value.__setitem__(
                    "scratch_directory",
                    r"C:\Users\example\OneDrive\scratch",
                ),
            ),
            (
                "scratch-committed",
                lambda value: value.__setitem__(
                    "scratch_data_committed", True
                ),
            ),
            (
                "support-budget",
                lambda value: value["support_budgets"].__setitem__(
                    "orbit_node_cap", 3
                ),
            ),
            (
                "numerical-radius",
                lambda value: value["numerical_budgets"].__setitem__(
                    "radii", [16.0]
                ),
            ),
            (
                "stop-after-nonconvergence",
                lambda value: value["numerical_budgets"].__setitem__(
                    "continue_after_nonconvergence", False
                ),
            ),
            (
                "resume-disabled",
                lambda value: value["numerical_budgets"].__setitem__(
                    "resume_from_checkpoint", False
                ),
            ),
            (
                "argv",
                lambda value: value["argv_receipt"].__setitem__(
                    value["argv_receipt"].index("--workers") + 1,
                    "2",
                ),
            ),
        ):
            receipt = copy.deepcopy(self.execution)
            mutate(receipt)
            cases.append((label, receipt))
        for label, execution in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises(
                        KrennCounterexampleArtifactError
                    ):
                        write_counterexample_bundle(
                            Path(temporary) / "bundle",
                            support_summary=self.support_summary,
                            numerical_summary=self.numerical_summary,
                            execution=execution,
                        )

    def test_refreshed_execution_hash_still_fails_semantic_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_counterexample_bundle(
                directory,
                support_summary=self.support_summary,
                numerical_summary=self.numerical_summary,
                execution=self.execution,
            )
            certificate_path = directory / "certificate.json"
            certificate = json.loads(
                certificate_path.read_text(encoding="utf-8")
            )
            certificate["execution"]["workers"] = 2
            encoded = (
                json.dumps(
                    certificate,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            certificate_path.write_bytes(encoded)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            record = next(
                row
                for row in manifest["artifacts"]
                if row["path"] == "certificate.json"
            )
            record["bytes"] = len(encoded)
            record["sha256"] = hashlib.sha256(encoded).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennCounterexampleArtifactError):
                verify_counterexample_bundle(directory)

    def test_nonexact_candidate_cannot_enter_exact_inventory(self):
        field = rational_field()
        candidate = ExactCandidateWitness.from_index_values(
            6,
            3,
            field,
            n6_d3_seed_witness().entries,
        )
        report = verify_exact_candidate(candidate)
        self.assertFalse(report.exact)
        numerical = dict(self.numerical_summary)
        numerical["exact_candidate_found"] = True
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(KrennCounterexampleArtifactError):
                write_counterexample_bundle(
                    Path(temporary) / "bundle",
                    support_summary=self.support_summary,
                    numerical_summary=numerical,
                    execution=self.execution,
                    exact_report=report,
                )

    def test_selected_support_must_come_from_closure_result(self):
        with self.assertRaises(KrennCounterexampleArtifactError):
            build_support_summary(
                (self.support_result,),
                selected_supports=(tuple(range(22)),),
            )

    def test_selected_support_may_use_stored_orbit_representative(self):
        representative = self.support_result.candidates[
            0
        ].orbit_representative
        self.assertNotEqual(representative, self.emitted_support)
        summary = build_support_summary(
            (self.support_result,),
            selected_supports=(representative,),
        )
        self.assertEqual(
            summary["selected_supports_for_numerics"],
            [list(representative)],
        )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_counterexample_bundle(
                directory,
                support_summary=summary,
                numerical_summary=self.numerical_summary,
                execution=self.execution,
            )
            self.assertFalse(
                verify_counterexample_bundle(directory).exact_counterexample
            )


if __name__ == "__main__":
    unittest.main()
