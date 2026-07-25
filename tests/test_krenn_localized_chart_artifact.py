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
