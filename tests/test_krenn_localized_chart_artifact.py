import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.localized_chart_artifact import (
    DEFAULT_RESULT_DIRECTORY,
    KrennLocalizedArtifactError,
    verify_localized_chart_bundle,
    write_localized_chart_bundle,
)


class KrennLocalizedChartArtifactTest(unittest.TestCase):
    def test_committed_bundle_replays_and_keeps_global_status_undecided(self):
        manifest = verify_localized_chart_bundle(
            DEFAULT_RESULT_DIRECTORY
        )
        self.assertEqual(
            manifest["status"],
            "exact-localized-chart-membership-undecided",
        )
        self.assertEqual(
            manifest["headline"][
                "exact_bounded_certificate_degree_excluded"
            ],
            6,
        )
        self.assertFalse(
            manifest["claim_boundary"]["counterexample_found"]
        )
        self.assertFalse(
            manifest["claim_boundary"]["global_nonexistence_proved"]
        )

    def test_exact_star_reduction_and_failed_shortcut_are_retained(self):
        payload = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "star_linearization.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(payload["apex_factorizations"]), 6)
        for row in payload["apex_factorizations"]:
            self.assertEqual(row["rows_per_apex_color"], 243)
            self.assertEqual(row["columns_per_apex_color"], 15)
            self.assertTrue(row["all_729_equations_reconstructed"])
            self.assertEqual(
                row["primary_sha256"], row["independent_sha256"]
            )
        differencing = payload["differencing_review"]
        self.assertFalse(
            differencing[
                "proposed_single_column_differencing_lemma_valid"
            ]
        )
        self.assertEqual(
            differencing["signed_cross_monomials_from_other_four_columns"],
            24,
        )
        self.assertFalse(
            payload["claim_boundary"]["finite_counterexample_found"]
        )

    def test_redundant_certificate_pair_is_not_promoted(self):
        payload = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "certificate_routing.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(
            payload["exact_checks"][
                "proposed_two_conditions_are_equivalent_not_complementary"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"][
                "expert_proposed_pair_implies_unit_ideal"
            ]
        )
        self.assertTrue(
            payload["exact_checks"][
                "correct_cover_has_open_and_closed_branches"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"]["all_global_components_classified"]
        )

    def test_star_target_rows_give_three_orbit_pivot_cover(self):
        payload = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "star_pivot_charts.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(payload["apex_audits"]), 6)
        for row in payload["apex_audits"]:
            self.assertEqual(row["three_column_minors_total"], 455)
            self.assertEqual(row["structurally_zero_minors"], 330)
            self.assertEqual(row["structurally_nonzero_minors"], 125)
        self.assertEqual(
            [
                row["orbit_size"]
                for row in payload["orbit_classification"]["orbits"]
            ],
            [5, 60, 60],
        )
        self.assertTrue(
            payload["rank_and_cover"][
                "the_125_determinant_opens_cover_every_solution_per_apex"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"]["any_open_chart_solved"]
        )

    def test_pivot_opens_have_root_free_three_factor_gauge_slices(self):
        payload = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "star_pivot_gauge.json"
            ).read_text(encoding="utf-8")
        )
        equivalence = payload["existence_equivalence"]
        self.assertTrue(
            equivalence[
                "pivot_open_nonempty_iff_three_factor_unit_slice_nonempty"
            ]
        )
        self.assertFalse(
            equivalence["normalization_requires_root_extraction"]
        )
        self.assertFalse(
            equivalence["normalization_requires_Rabinowitsch_variable"]
        )
        self.assertFalse(
            equivalence["normalization_requires_saturation"]
        )
        for apex in payload["apex_audits"]:
            self.assertEqual(
                apex["full_pivot_cover_semi_invariant_census"][
                    "pivot_charts_checked"
                ],
                125,
            )
            for orbit in apex["transported_orbit_representatives"]:
                self.assertEqual(
                    orbit["factor_character_smith_invariant_factors"],
                    [1, 1, 1],
                )
                self.assertEqual(
                    orbit["star_elimination"][
                        "star_weights_eliminable_after_factor_normalization"
                    ],
                    9,
                )
        self.assertFalse(
            payload["claim_boundary"][
                "fifteen_anchor_numerical_chart_covers_entire_pivot_open"
            ]
        )

    def test_three_affine_slice_presentations_reconstruct_exactly(self):
        payload = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY
                / "star_pivot_affine_slices.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            payload["cover"]["S5_times_S3_orbit_count"], 3
        )
        self.assertTrue(
            payload["cover"]["every_solution_is_in_at_least_one_open"]
        )
        self.assertFalse(
            payload["gauge_normalization"][
                "requires_colon_or_saturation"
            ]
        )
        self.assertEqual(len(payload["presentations"]), 3)
        for row in payload["presentations"]:
            self.assertEqual(
                (
                    row["retained"]["ring_variable_count"],
                    row["retained"]["generator_count"],
                    row["retained"]["maximum_degree"],
                    row["retained"]["collected_term_count"],
                ),
                (135, 732, 3, 10_950),
            )
            self.assertEqual(
                (
                    row["eliminated"]["ring_variable_count"],
                    row["eliminated"]["generator_count"],
                    row["eliminated"]["maximum_degree"],
                    row["eliminated"]["collected_term_count"],
                ),
                (126, 723, 5, 35_292),
            )
            self.assertFalse(
                row["eliminated"]["symmetry_related_weights_identified"]
            )
        self.assertFalse(
            payload["claim_boundary"]["any_affine_slice_solved"]
        )

    def test_all_seven_generic_probes_have_retained_ten_minute_receipts(self):
        ledger = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "preflight_ledger.json"
            ).read_text(encoding="utf-8")
        )
        campaign = ledger["singular_engine"][
            "ten_minute_F31_reconnaissance"
        ]
        self.assertEqual(campaign["probe_count"], 7)
        self.assertEqual(
            campaign["budget_seconds_per_probe"], 600
        )
        self.assertEqual(
            ledger["host_policy"][
                "maximum_observed_concurrent_CAS_cpus"
            ],
            6,
        )
        for row in campaign["runs"]:
            self.assertEqual(row["budget_seconds"], 600)
            self.assertGreaterEqual(row["elapsed_seconds"], 600)
            self.assertEqual(row["return_code"], 124)
            self.assertEqual(row["status"], "timeout-after-parse")
            self.assertTrue(
                row["marker_receipt"]["parse_marker_observed"]
            )
            self.assertFalse(
                row["marker_receipt"]["completion_marker_observed"]
            )
            self.assertGreater(row["stdout"]["bytes"], 0)
            self.assertEqual(len(row["stdout"]["sha256"]), 64)
            self.assertEqual(len(row["receipt"]["sha256"]), 64)
            self.assertFalse(
                row["claim_boundary"]["timeout_is_a_chart_decision"]
            )

    def test_residual_grading_is_exact_but_not_promoted_to_a_decision(self):
        payload = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "residual_grading.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [
                row["integral_character_quotient"]["residual_rank"]
                for row in payload["repair_charts"]
            ],
            [8, 8, 7, 7],
        )
        self.assertTrue(
            payload["exact_checks"][
                "all_730_generator_sets_homogeneous"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"]["tropical_cones_enumerated"]
        )
        self.assertFalse(
            payload["claim_boundary"]["finite_counterexample_found"]
        )

    def test_sparse_derivative_slices_replace_the_degree_six_blowup(self):
        payload = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY
                / "natural_derivative_atlas.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(
            payload["preferred_next_formulation"][
                "existence_equivalent_to_derivative_open"
            ]
        )
        self.assertEqual(len(payload["sparse_gauge_slices"]), 2)
        for row in payload["sparse_gauge_slices"]:
            self.assertEqual(row["variables"], 129)
            self.assertEqual(row["generators"], 730)
            self.assertEqual(row["sparse_terms"], 10_942)
            self.assertEqual(row["maximum_degree"], 4)
            self.assertFalse(
                row["claim_boundary"]["natural_chart_decided"]
            )

    def test_character_blocked_and_leaf_free_payloads_fail_closed(self):
        graded = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY
                / "graded_derivative_macaulay.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            graded["exact_conclusion"][
                "no_nullstellensatz_identity_through_total_degree"
            ],
            5,
        )
        self.assertEqual(
            graded["degree_six_status"]["rank_gaps"],
            {"11": 123, "29": 117},
        )
        self.assertFalse(
            graded["claim_boundary"][
                "degree_six_nonmembership_over_Q_certified"
            ]
        )
        sparse = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "sparse_elimination.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [row["generators"]["character_blocks"]
             for row in sparse["systems"]],
            [558, 642],
        )
        self.assertTrue(
            all(
                row["triangular_rewrite"]["two_monic_pivots"]
                for row in sparse["systems"]
            )
        )
        leaf_free = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "leaf_free_saturation.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            leaf_free["claim_boundary"]["saturation_stages_completed"],
            0,
        )
        self.assertEqual(len(leaf_free["derivative_systems"]), 2)
        self.assertEqual(len(leaf_free["repair_systems"]), 4)

    def test_two_sparse_p1009_probes_used_full_ten_minute_budgets(self):
        ledger = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "preflight_ledger.json"
            ).read_text(encoding="utf-8")
        )
        campaign = ledger["singular_engine"][
            "ten_minute_F1009_sparse_two_pivot_reconnaissance"
        ]
        self.assertEqual(campaign["probe_count"], 2)
        self.assertEqual(campaign["budget_seconds_per_probe"], 600)
        for row in campaign["runs"]:
            self.assertGreaterEqual(row["elapsed_seconds"], 600)
            self.assertEqual(row["return_code"], 124)
            self.assertEqual(row["status"], "timeout-after-parse")
            self.assertTrue(
                row["marker_receipt"]["parse_marker_observed"]
            )
            self.assertFalse(
                row["marker_receipt"]["completion_marker_observed"]
            )
            self.assertFalse(
                row["claim_boundary"]["timeout_is_a_chart_decision"]
            )

    def test_two_leaf_free_A0_probes_used_full_ten_minute_budgets(self):
        ledger = json.loads(
            (
                DEFAULT_RESULT_DIRECTORY / "preflight_ledger.json"
            ).read_text(encoding="utf-8")
        )
        campaign = ledger["singular_engine"][
            "ten_minute_F31_leaf_free_A0_reconnaissance"
        ]
        self.assertEqual(campaign["probe_count"], 2)
        self.assertEqual(campaign["budget_seconds_per_probe"], 600)
        for row in campaign["runs"]:
            self.assertGreaterEqual(row["elapsed_seconds"], 600)
            self.assertEqual(row["return_code"], 124)
            self.assertEqual(row["status"], "timeout-after-parse")
            self.assertEqual(row["saturation_stages_completed"], 0)
            self.assertTrue(
                row["marker_receipt"]["parse_marker_observed"]
            )
            self.assertFalse(
                row["marker_receipt"]["completion_marker_observed"]
            )
            self.assertFalse(
                row["claim_boundary"][
                    "later_saturation_stages_completed"
                ]
            )

    def test_escalated_claim_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(DEFAULT_RESULT_DIRECTORY, copied)
            bounded_path = copied / "bounded_degree_six.json"
            bounded = json.loads(
                bounded_path.read_text(encoding="utf-8")
            )
            bounded["claim_boundary"][
                "finite_affine_GHZ_membership_status"
            ] = "nonexistent"
            bounded_path.write_text(
                json.dumps(bounded, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                verify_localized_chart_bundle(copied)

    def test_refreshed_hash_does_not_hide_bounded_claim_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(DEFAULT_RESULT_DIRECTORY, copied)
            bounded_path = copied / "bounded_degree_six.json"
            bounded = json.loads(
                bounded_path.read_text(encoding="utf-8")
            )
            bounded["proof_logic"] = "finite-field ranks prove everything"
            bounded_path.write_text(
                json.dumps(bounded, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_path = copied / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            for row in manifest["files"]:
                if row["path"] == "bounded_degree_six.json":
                    row["sha256"] = hashlib.sha256(
                        bounded_path.read_bytes()
                    ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                verify_localized_chart_bundle(copied)

    def test_refreshed_hash_does_not_promote_degree_six_reconnaissance(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(DEFAULT_RESULT_DIRECTORY, copied)
            graded_path = copied / "graded_derivative_macaulay.json"
            graded = json.loads(
                graded_path.read_text(encoding="utf-8")
            )
            graded["claim_boundary"][
                "degree_six_nonmembership_over_Q_certified"
            ] = True
            graded_path.write_text(
                json.dumps(graded, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_path = copied / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            for row in manifest["files"]:
                if row["path"] == "graded_derivative_macaulay.json":
                    row["sha256"] = hashlib.sha256(
                        graded_path.read_bytes()
                    ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                verify_localized_chart_bundle(copied)

    def test_refreshed_hash_does_not_promote_false_differencing_lemma(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(DEFAULT_RESULT_DIRECTORY, copied)
            star_path = copied / "star_linearization.json"
            star = json.loads(star_path.read_text(encoding="utf-8"))
            star["differencing_review"][
                "proposed_single_column_differencing_lemma_valid"
            ] = True
            star_path.write_text(
                json.dumps(star, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_path = copied / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            for row in manifest["files"]:
                if row["path"] == "star_linearization.json":
                    row["sha256"] = hashlib.sha256(
                        star_path.read_bytes()
                    ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                verify_localized_chart_bundle(copied)

    def test_manifest_claim_escalation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(DEFAULT_RESULT_DIRECTORY, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["claim_boundary"]["counterexample_found"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                verify_localized_chart_bundle(copied)

    def test_manifest_numeric_type_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(DEFAULT_RESULT_DIRECTORY, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["parameters"]["n"] = 6.0
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                verify_localized_chart_bundle(copied)

    def test_duplicate_json_key_is_rejected_when_last_value_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(DEFAULT_RESULT_DIRECTORY, copied)
            manifest_path = copied / "manifest.json"
            text = manifest_path.read_text(encoding="utf-8")
            expected = (
                '"status": '
                '"exact-localized-chart-membership-undecided"'
            )
            replacement = (
                '"status": "counterexample-found",\n'
                f'  {expected}'
            )
            self.assertIn(expected, text)
            manifest_path.write_text(
                text.replace(expected, replacement, 1),
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                verify_localized_chart_bundle(copied)

    def test_refreshed_hash_does_not_hide_boolean_integer_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(DEFAULT_RESULT_DIRECTORY, copied)
            bounded_path = copied / "bounded_degree_six.json"
            bounded = json.loads(
                bounded_path.read_text(encoding="utf-8")
            )
            bounded["charts"][0][
                "bounded_nonmembership_over_q_certified"
            ] = 1
            bounded_path.write_text(
                json.dumps(bounded, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_path = copied / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            for row in manifest["files"]:
                if row["path"] == "bounded_degree_six.json":
                    row["sha256"] = hashlib.sha256(
                        bounded_path.read_bytes()
                    ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                verify_localized_chart_bundle(copied)

    def test_generation_rejects_undeclared_output_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "stale.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaises(KrennLocalizedArtifactError):
                write_localized_chart_bundle(output)

    @unittest.skipUnless(
        os.environ.get("KRENN_RUN_LONG_TESTS") == "1",
        "set KRENN_RUN_LONG_TESTS=1 for all eight degree-six matrices",
    )
    def test_long_all_eight_degree_six_native_replay(self):
        verify_localized_chart_bundle(
            DEFAULT_RESULT_DIRECTORY,
            full_degree_six_replay=True,
        )


if __name__ == "__main__":
    unittest.main()
