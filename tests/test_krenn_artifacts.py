from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.krenn_quantum_graph import independent_verifier
from experiments.krenn_quantum_graph.artifacts import (
    DEFAULT_RESULTS,
    KrennArtifactError,
    SYSTEM_FILE,
    system_arrays,
    verify_artifact_bundle,
    write_artifact_bundle,
)
from experiments.krenn_quantum_graph.deformation import (
    certify_n4_d3_deformation,
)
from experiments.krenn_quantum_graph.fixtures import fixture_n4_d3
from experiments.krenn_quantum_graph.generate_results import (
    BUNDLE_NAMES,
    generate_all,
    negative_benchmark_boundary,
    verify_all,
)
from experiments.krenn_quantum_graph.system import generate_sparse_system
from experiments.krenn_quantum_graph.targets import ColoringTarget


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_artifact_record(directory: Path, filename: str) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / filename
    for record in manifest["artifacts"]:
        if record["path"] == filename:
            record["bytes"] = path.stat().st_size
            record["sha256"] = _sha256(path)
            break
    else:
        raise AssertionError(f"missing manifest record for {filename}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_json_artifact(
    directory: Path, filename: str, payload
) -> None:
    (directory / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_artifact_record(directory, filename)


class KrennArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "krenn-results"
        cls.summary = generate_all(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_four_milestone_bundles_round_trip(self):
        self.assertEqual(set(self.summary), set(BUNDLE_NAMES))
        self.assertTrue(
            all(row["exact"] for row in self.summary.values())
        )
        self.assertEqual(
            self.summary["n4_d3_fixture"]["variables"], 54
        )
        self.assertEqual(
            self.summary["n6_d2_fixture"]["monomials"], 960
        )
        self.assertEqual(
            self.summary["n4_d4_negative_benchmark"]["equations"],
            256,
        )
        production = self.summary["n6_d4_production_system"]
        self.assertEqual(production["variables"], 240)
        self.assertEqual(production["equations"], 4096)
        self.assertEqual(production["monomials"], 61440)
        self.assertFalse(production["witness_present"])

    def test_negative_and_production_bundles_make_no_search_claim(self):
        negative = verify_artifact_bundle(
            self.output / "n4_d4_negative_benchmark"
        )
        production = verify_artifact_bundle(
            self.output / "n6_d4_production_system"
        )
        for bundle in (negative, production):
            self.assertFalse(bundle.boundary["search_performed"])
            self.assertFalse(
                bundle.boundary["no_solution_certificate_emitted"]
            )
            self.assertIsNone(bundle.witness)
        self.assertEqual(
            negative.certificate["status"],
            "known-negative-benchmark-not-reproved",
        )
        self.assertEqual(
            production.certificate["status"],
            "bounded-native-support-analysis-begun",
        )
        self.assertIn(
            "Bounded native support analysis has begun",
            production.boundary["statement"],
        )
        self.assertIn(
            "No complex weight search was performed",
            production.boundary["statement"],
        )
        self.assertIn(
            "no witness or nonexistence certificate",
            production.boundary["statement"],
        )

    def test_released_result_bundles_replay(self):
        released = verify_all(DEFAULT_RESULTS)
        self.assertEqual(set(released), set(BUNDLE_NAMES))
        self.assertTrue(all(row["exact"] for row in released.values()))

    def test_manifest_hash_detects_raw_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_artifact_bundle(
                generate_sparse_system(4, 3),
                directory,
                witness=fixture_n4_d3(),
            )
            certificate = directory / "certificate.json"
            certificate.write_text(
                certificate.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaises(
                independent_verifier.IndependentVerificationError
            ):
                independent_verifier.verify_artifact_bundle(directory)

    def test_semantic_corruption_fails_after_manifest_hash_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_artifact_bundle(
                generate_sparse_system(4, 3),
                directory,
            )
            archive_path = directory / SYSTEM_FILE
            with np.load(archive_path, allow_pickle=False) as archive:
                arrays = {
                    key: np.asarray(archive[key]).copy()
                    for key in archive.files
                }
            arrays["monomial_variable_indices"][0, 0] += 1
            with archive_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            _refresh_artifact_record(directory, SYSTEM_FILE)
            with self.assertRaisesRegex(
                independent_verifier.IndependentVerificationError,
                "monomial corruption",
            ):
                independent_verifier.verify_artifact_bundle(directory)
            with self.assertRaises(
                independent_verifier.IndependentVerificationError
            ):
                verify_artifact_bundle(directory)

    def test_duplicate_witness_row_fails_semantic_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_artifact_bundle(
                generate_sparse_system(4, 3),
                directory,
                witness=fixture_n4_d3(),
            )
            witness_path = directory / "witness.json"
            payload = json.loads(
                witness_path.read_text(encoding="utf-8")
            )
            payload["entries"].insert(1, dict(payload["entries"][0]))
            payload["support_size"] += 1
            witness_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _refresh_artifact_record(directory, "witness.json")
            with self.assertRaises(
                independent_verifier.IndependentVerificationError
            ):
                independent_verifier.verify_artifact_bundle(directory)

    def test_manifest_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_artifact_bundle(
                generate_sparse_system(4, 3),
                directory,
            )
            manifest_path = directory / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["artifacts"][0]["path"] = "../certificate.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(
                independent_verifier.IndependentVerificationError
            ):
                independent_verifier.verify_artifact_bundle(directory)

    def test_refreshed_hash_cannot_escalate_headline_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_artifact_bundle(
                generate_sparse_system(4, 3), directory
            )
            certificate = json.loads(
                (directory / "certificate.json").read_text(
                    encoding="utf-8"
                )
            )
            certificate["status"] = "NO_SOLUTION_PROVED"
            _rewrite_json_artifact(
                directory, "certificate.json", certificate
            )
            with self.assertRaisesRegex(
                independent_verifier.IndependentVerificationError,
                "headline certificate status",
            ):
                independent_verifier.verify_artifact_bundle(directory)
            with self.assertRaises(KrennArtifactError):
                write_artifact_bundle(
                    generate_sparse_system(4, 3),
                    Path(temporary) / "bad-status",
                    status="NO_SOLUTION_PROVED",
                )

    def test_refreshed_boundary_hash_cannot_claim_a_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_artifact_bundle(
                generate_sparse_system(4, 4),
                directory,
                boundary=negative_benchmark_boundary(),
            )
            boundary = json.loads(
                (directory / "claim_boundary.json").read_text(
                    encoding="utf-8"
                )
            )
            boundary["status"] = "NO_SOLUTION_PROVED"
            boundary["statement"] = "A proof is claimed."
            _rewrite_json_artifact(
                directory, "claim_boundary.json", boundary
            )
            with self.assertRaisesRegex(
                independent_verifier.IndependentVerificationError,
                "claim-boundary semantics",
            ):
                independent_verifier.verify_artifact_bundle(directory)

    def test_refreshed_deformation_hash_fails_independent_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            system = generate_sparse_system(4, 3)
            witness = fixture_n4_d3()
            write_artifact_bundle(
                system,
                directory,
                witness=witness,
                deformation=certify_n4_d3_deformation(
                    system, witness
                ),
            )
            deformation = json.loads(
                (
                    directory / "deformation_certificate.json"
                ).read_text(encoding="utf-8")
            )
            deformation["jacobian"]["rank_over_Q"] = 54
            _rewrite_json_artifact(
                directory,
                "deformation_certificate.json",
                deformation,
            )
            with self.assertRaisesRegex(
                independent_verifier.IndependentVerificationError,
                "deformation claims",
            ):
                independent_verifier.verify_artifact_bundle(directory)

    def test_required_source_ledger_and_uniqueness_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_artifact_bundle(
                generate_sparse_system(4, 3), directory
            )
            manifest_path = directory / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            input_labels = {
                record["path"] for record in manifest["inputs"]
            }
            self.assertIn(
                "experiments/krenn_quantum_graph/targets.py",
                input_labels,
            )
            self.assertIn(
                "experiments/krenn_quantum_graph/targets.py",
                independent_verifier.required_input_paths(
                    witness_present=False,
                    deformation_present=False,
                ),
            )
            manifest["inputs"] = [
                record
                for record in manifest["inputs"]
                if record["path"]
                != "experiments/krenn_quantum_graph/system.py"
            ]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                independent_verifier.IndependentVerificationError,
                "mandatory dependency",
            ):
                independent_verifier.verify_artifact_bundle(directory)

            write_artifact_bundle(
                generate_sparse_system(4, 3),
                Path(temporary) / "duplicate",
            )
            duplicate_manifest_path = (
                Path(temporary) / "duplicate" / "manifest.json"
            )
            duplicate = json.loads(
                duplicate_manifest_path.read_text(encoding="utf-8")
            )
            duplicate["artifacts"].append(
                dict(duplicate["artifacts"][0])
            )
            duplicate_manifest_path.write_text(
                json.dumps(duplicate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                independent_verifier.IndependentVerificationError,
                "unique and sorted",
            ):
                independent_verifier.verify_artifact_bundle(
                    Path(temporary) / "duplicate"
                )

    def test_deformation_manifest_is_native_and_standalone(self):
        directory = self.output / "n4_d3_fixture"
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        inputs = {record["path"] for record in manifest["inputs"]}
        self.assertIn(
            "experiments/krenn_quantum_graph/deformation.py",
            inputs,
        )
        self.assertFalse(
            any(
                path.startswith("experiments/section12_instantiation/")
                for path in inputs
            )
        )

    def test_artifact_parameters_are_bounded_before_enumeration(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_artifact_bundle(
                generate_sparse_system(4, 3), directory
            )
            certificate = json.loads(
                (directory / "certificate.json").read_text(
                    encoding="utf-8"
                )
            )
            certificate["parameters"] = {"n": 100, "d": 3}
            _rewrite_json_artifact(
                directory, "certificate.json", certificate
            )
            with self.assertRaisesRegex(
                independent_verifier.IndependentVerificationError,
                "bounded v1 milestone",
            ):
                independent_verifier.verify_artifact_bundle(directory)

    def test_v1_artifacts_reject_fractional_target_without_truncation(self):
        target = ColoringTarget.from_sparse(
            4, 2, {((0, 0, 0, 0)): Fraction(1, 2)}
        )
        system = generate_sparse_system(4, 2, target)
        self.assertEqual(
            system.rhs_values[0], Fraction(1, 2)
        )
        with mock.patch.object(
            np,
            "asarray",
            side_effect=AssertionError(
                "fractional target reached numpy conversion"
            ),
        ):
            with self.assertRaisesRegex(
                KrennArtifactError, "canonical GHZ target"
            ):
                system_arrays(system)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fractional-target"
            with self.assertRaisesRegex(
                KrennArtifactError, "distinct artifact schema"
            ):
                write_artifact_bundle(system, directory)
            self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
