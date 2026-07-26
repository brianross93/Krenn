from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.krenn_quantum_graph.numerical_continuation import (
    GHZ_COLOR_DIAGONAL_GAUGE,
    KrennNumericalContinuationError,
    LMPlan,
    complex_residual,
    natural_gauge_chart,
    projective_victim_repair_initializer,
    solve_direct_ghz,
)
from experiments.krenn_quantum_graph.projective_repair_campaign import (
    DEFAULT_RESULTS,
    EXISTING_SIZE22_SUPPORT_A,
    EXISTING_SIZE22_SUPPORT_B,
    KrennProjectiveRepairCampaignError,
    MASTER_SEED,
    active_victim_matchings,
    branch_orbit_hits,
    branch_spec,
    deterministic_jobs,
    existing_support_inventory,
    projective_branch_specs,
    run_n0_support_preflight,
    validate_projective_support,
    verify_bundle,
    verify_payload,
    victim_projective_audit,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
)


class ProjectiveRepairCampaignTests(unittest.TestCase):
    def test_exact_branch_specs_and_existing_support_mapping(self):
        n0, n2 = projective_branch_specs()
        self.assertEqual(n0.matching_index, 0)
        self.assertEqual(n0.numerator_indices, (88, 133))
        self.assertEqual(n0.denominator_indices, (98, 121))
        self.assertEqual(n2.matching_index, 2)
        self.assertEqual(n2.numerator_indices, (106, 113))
        self.assertEqual(n2.denominator_indices, (98, 121))

        inventory = existing_support_inventory()
        self.assertFalse(inventory["n0_size22_support_present"])
        self.assertEqual(
            tuple(
                row["branch_orbit_hit_counts"]
                for row in inventory["supports"]
            ),
            ({"0": 0, "2": 2}, {"0": 0, "2": 2}),
        )
        for support in (
            EXISTING_SIZE22_SUPPORT_A,
            EXISTING_SIZE22_SUPPORT_B,
        ):
            self.assertEqual(branch_orbit_hits(support, n0), ())
            self.assertEqual(len(branch_orbit_hits(support, n2)), 2)

        validation = validate_projective_support(
            EXISTING_SIZE22_SUPPORT_B, n2
        )
        self.assertTrue(
            validation["eligible_for_two_matching_initializer"]
        )
        self.assertEqual(
            validation["active_victim_matching_indices"], [1, 2]
        )
        self.assertFalse(validation["solution_certified"])
        self.assertFalse(validation["nonexistence_proved"])

    def test_initializer_is_direct_gauge_safe_and_projective(self):
        spec = branch_spec(2)
        chart = natural_gauge_chart(
            EXISTING_SIZE22_SUPPORT_B,
            action_group=GHZ_COLOR_DIAGONAL_GAUGE,
        )
        self.assertEqual(chart.rank, 10)
        self.assertTrue(
            set(spec.numerator_indices).intersection(
                chart.free_indices
            )
        )
        options = dict(
            numerator_indices=spec.numerator_indices,
            denominator_indices=spec.denominator_indices,
            expected_active_victim_matching_indices=(
                spec.expected_active_victim_matchings
            ),
            repair_scale=0.03,
            chart=chart,
            l2_bound=8.0,
            linf_bound=8.0,
        )
        first = projective_victim_repair_initializer(
            EXISTING_SIZE22_SUPPORT_B, MASTER_SEED, **options
        )
        replay = projective_victim_repair_initializer(
            EXISTING_SIZE22_SUPPORT_B, MASTER_SEED, **options
        )
        other = projective_victim_repair_initializer(
            EXISTING_SIZE22_SUPPORT_B, MASTER_SEED + 1, **options
        )
        np.testing.assert_array_equal(
            first.dense_weights(), replay.dense_weights()
        )
        self.assertFalse(
            np.array_equal(
                first.dense_weights(), other.dense_weights()
            )
        )
        self.assertEqual(first.kind, "projective-victim-repair")
        self.assertIsNone(first.laurent_t)
        self.assertEqual(
            first.chart.action_group, GHZ_COLOR_DIAGONAL_GAUGE
        )
        self.assertFalse(first.projection.changed)
        self.assertTrue(
            np.all(
                first.dense_weights()[
                    np.asarray(first.chart.anchors, dtype=np.intp)
                ]
                == 1.0
            )
        )
        self.assertAlmostEqual(
            abs(first.dense_weights()[106]), 1.0
        )
        self.assertAlmostEqual(
            abs(first.dense_weights()[113]), 1.0
        )

        audit = victim_projective_audit(
            first.dense_weights(), spec
        )
        self.assertEqual(
            audit["nonzero_matching_indices_at_1e-14"], [1, 2]
        )
        self.assertAlmostEqual(audit["r_N"][0], -1.0)
        self.assertAlmostEqual(audit["r_N"][1], 0.0)
        self.assertLess(
            abs(complex(*audit["cross_multiplied_relation_error"])),
            1.0e-14,
        )
        self.assertLess(
            abs(complex(*audit["full_victim_hyperplane_error"])),
            1.0e-14,
        )
        self.assertLess(
            abs(
                complex_residual(first.dense_weights())[
                    N6_D3_SEED_DEFECT_EQUATION
                ]
            ),
            1.0e-14,
        )

    def test_initializer_rejects_multinomial_and_post_relation_projection(self):
        spec = branch_spec(2)
        multinomial_support = tuple(
            sorted(
                set(EXISTING_SIZE22_SUPPORT_B).union(
                    branch_spec(0).branch_support
                )
            )
        )
        self.assertEqual(
            active_victim_matchings(multinomial_support), (0, 1, 2)
        )
        with self.assertRaisesRegex(
            KrennNumericalContinuationError,
            "multinomial initializer",
        ):
            projective_victim_repair_initializer(
                multinomial_support,
                MASTER_SEED,
                numerator_indices=spec.numerator_indices,
                denominator_indices=spec.denominator_indices,
                expected_active_victim_matching_indices=(1, 2),
            )

        chart = natural_gauge_chart(
            EXISTING_SIZE22_SUPPORT_B,
            action_group=GHZ_COLOR_DIAGONAL_GAUGE,
        )
        with self.assertRaisesRegex(
            KrennNumericalContinuationError,
            "exceeded hard bounds",
        ):
            projective_victim_repair_initializer(
                EXISTING_SIZE22_SUPPORT_B,
                MASTER_SEED,
                numerator_indices=spec.numerator_indices,
                denominator_indices=spec.denominator_indices,
                expected_active_victim_matching_indices=(1, 2),
                repair_scale=1.0,
                chart=chart,
                l2_bound=3.2,
                linf_bound=8.0,
            )

    def test_jobs_pair_radii_without_laurent_or_continuation(self):
        jobs = deterministic_jobs(
            {2: EXISTING_SIZE22_SUPPORT_B},
            starts_per_support=2,
            radii=(8.0, 32.0),
            repair_scales=(0.03, 0.1),
            master_seed=MASTER_SEED,
        )
        self.assertEqual(len(jobs), 4)
        self.assertEqual(
            jobs[0].seed_words,
            (1277533490, 4171084866, 41516467, 3461844625),
        )
        self.assertEqual(jobs[0].seed_words, jobs[1].seed_words)
        self.assertNotEqual(jobs[0].seed_words, jobs[2].seed_words)
        for job in jobs:
            payload = job.to_dict()
            self.assertEqual(payload["target"], "direct-GHZ")
            self.assertFalse(payload["known_laurent_initializer_used"])
            self.assertFalse(payload["continuation_used"])
            self.assertEqual(payload["symmetry_weight_equalities"], 0)

    def test_tiny_n0_preflight_and_solver_remain_fail_closed(self):
        preflight, supports = run_n0_support_preflight(
            node_cap=1,
            discovered_support_cap=1_000,
            candidate_scan_cap=1,
            worker_count=1,
            scratch_directory=(
                r"D:\KrennScratch\counterexample_search"
                r"\projective_test"
            ),
        )
        self.assertEqual(supports, ())
        self.assertEqual(preflight["eligible_support_count"], 0)
        self.assertFalse(
            preflight["bounded_miss_proves_n0_nonexistence"]
        )
        self.assertFalse(
            preflight["multinomial_candidates_are_excluded_globally"]
        )

        spec = branch_spec(2)
        chart = natural_gauge_chart(
            EXISTING_SIZE22_SUPPORT_B,
            action_group=GHZ_COLOR_DIAGONAL_GAUGE,
        )
        initializer = projective_victim_repair_initializer(
            EXISTING_SIZE22_SUPPORT_B,
            MASTER_SEED,
            numerator_indices=spec.numerator_indices,
            denominator_indices=spec.denominator_indices,
            expected_active_victim_matching_indices=(1, 2),
            chart=chart,
            l2_bound=8.0,
            linf_bound=8.0,
        )
        plan = LMPlan(
            max_iterations=1,
            max_evaluations=4,
            l2_bound=8.0,
            linf_bound=8.0,
            backtracking_steps=0,
            checkpoint_interval=0,
            residual_tolerance=1.0e-14,
            gradient_tolerance=1.0e-14,
            step_tolerance=1.0e-14,
        )
        result = solve_direct_ghz(
            initializer,
            EXISTING_SIZE22_SUPPORT_B,
            chart=chart,
            plan=plan,
            run_id="tiny-projective-fail-closed",
        )
        self.assertEqual(result.target_amplitude, 0.0j)
        self.assertFalse(result.exact_solution_certified)
        self.assertFalse(result.nonexistence_proved)
        self.assertLessEqual(result.iterations, 1)
        self.assertLessEqual(result.evaluations, 4)
        boundary = result.to_dict()["claim_boundary"]
        self.assertFalse(boundary["numerical_zero_is_exact_witness"])
        self.assertFalse(boundary["bounded_miss_proves_nonexistence"])

    def test_committed_bundle_and_refreshed_claims_fail_closed(self):
        for name in ("campaign.json", "manifest.json", "README.md"):
            self.assertNotIn(b"\r", (DEFAULT_RESULTS / name).read_bytes())
        payload = verify_bundle(DEFAULT_RESULTS)
        self.assertFalse(payload["exact_candidate_dual_verified"])
        self.assertEqual(
            payload["status"],
            "bounded-search-complete-no-exact-candidate",
        )
        with patch.object(
            sys,
            "executable",
            r"C:\DifferentPython\python.exe",
        ):
            self.assertEqual(verify_payload(deepcopy(payload)), payload)

        promoted = deepcopy(payload)
        promoted["claim_boundary"][
            "n0_support_22_nonexistence_proved"
        ] = True
        with self.assertRaisesRegex(
            KrennProjectiveRepairCampaignError,
            "claim boundary",
        ):
            verify_payload(promoted)

        forged_row_claim = deepcopy(payload)
        forged_row_claim["numerical_jobs"][0]["result"][
            "claim_boundary"
        ]["nonexistence_proved"] = True
        with self.assertRaisesRegex(
            KrennProjectiveRepairCampaignError,
            "metadata|boundary",
        ):
            verify_payload(forged_row_claim)

        forged_exact = deepcopy(payload)
        forged_exact["numerical_jobs"][0]["result"][
            "exact_solution_certified"
        ] = True
        with self.assertRaisesRegex(
            KrennProjectiveRepairCampaignError,
            "metadata",
        ):
            verify_payload(forged_exact)

        forged_reconstruction = deepcopy(payload)
        forged_reconstruction["numerical_jobs"][0][
            "reconstruction"
        ]["status"] = "exact-dual-verification-passed"
        with self.assertRaisesRegex(
            KrennProjectiveRepairCampaignError,
            "reconstruction",
        ):
            verify_payload(forged_reconstruction)

        forged_status = deepcopy(payload)
        forged_status["status"] = "GLOBAL-NONEXISTENCE-PROVED"
        with self.assertRaisesRegex(
            KrennProjectiveRepairCampaignError,
            "status",
        ):
            verify_payload(forged_status)

        forged_initializer = deepcopy(payload)
        forged_initializer["numerical_jobs"][0]["initializer"][
            "projective_audit"
        ]["r_N"] = [999.0, 0.0]
        with self.assertRaisesRegex(
            KrennProjectiveRepairCampaignError,
            "boundary",
        ):
            verify_payload(forged_initializer)

        forged_gauge = deepcopy(payload)
        forged_gauge["numerical_jobs"][0]["result"]["gauge_chart"][
            "gauge_action_group"
        ] = "ghz-victim-color-diagonal"
        with self.assertRaisesRegex(
            KrennProjectiveRepairCampaignError,
            "metadata",
        ):
            verify_payload(forged_gauge)

        forged_norm = deepcopy(payload)
        forged_norm["numerical_jobs"][0]["result"]["weight_metrics"][
            "l2"
        ] = 0.0
        with self.assertRaisesRegex(
            KrennProjectiveRepairCampaignError,
            "all-729",
        ):
            verify_payload(forged_norm)

        broken_accounting = deepcopy(payload)
        broken_accounting["n0_support_preflight"][
            "support_campaign"
        ]["nodes_examined"] += 1
        with self.assertRaisesRegex(
            (KrennProjectiveRepairCampaignError, ValueError),
            "accounting|support",
        ):
            verify_payload(broken_accounting)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory) / "bundle"
            shutil.copytree(DEFAULT_RESULTS, directory)
            campaign_path = directory / "campaign.json"
            manifest_path = directory / "manifest.json"
            copied = json.loads(
                campaign_path.read_text(encoding="utf-8")
            )
            copied["claim_boundary"][
                "multinomial_n0_branches_excluded"
            ] = True
            campaign_path.write_text(
                json.dumps(copied, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            campaign_bytes = campaign_path.read_bytes()
            for record in manifest["artifacts"]:
                if record["path"] == "campaign.json":
                    record["bytes"] = len(campaign_bytes)
                    record["sha256"] = sha256(
                        campaign_bytes
                    ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennProjectiveRepairCampaignError,
                "claim boundary",
            ):
                verify_bundle(directory)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory) / "bundle"
            shutil.copytree(DEFAULT_RESULTS, directory)
            campaign_path = directory / "campaign.json"
            manifest_path = directory / "manifest.json"
            campaign_text = campaign_path.read_text(encoding="utf-8")
            campaign_text = campaign_text.replace(
                '{\n  "branch_specs"',
                '{\n  "schema": "duplicate",\n  "branch_specs"',
                1,
            )
            campaign_path.write_text(campaign_text, encoding="utf-8")
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            campaign_bytes = campaign_path.read_bytes()
            for record in manifest["artifacts"]:
                if record["path"] == "campaign.json":
                    record["bytes"] = len(campaign_bytes)
                    record["sha256"] = sha256(
                        campaign_bytes
                    ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennProjectiveRepairCampaignError,
                "duplicate JSON key",
            ):
                verify_bundle(directory)


if __name__ == "__main__":
    unittest.main()
