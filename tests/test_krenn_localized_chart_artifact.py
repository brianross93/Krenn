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
