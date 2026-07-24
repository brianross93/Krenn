from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from zipfile import ZipFile

import numpy as np

from experiments.krenn_quantum_graph.target_artifacts import (
    CERTIFICATE_FILE,
    MANIFEST_FILE,
    TARGET_FILE,
    TENSOR_MAP_ARCHIVE_KEYS,
    TENSOR_MAP_FILE,
    WITNESS_FILE,
    KrennTargetArtifactError,
    verify_target_artifact_bundle,
    write_target_artifact_bundle,
)
from experiments.krenn_quantum_graph.targets import ColoringTarget
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.witness import SparseWitness


def _write_json(path: Path, payload) -> None:
    path.write_bytes(
        (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_manifest_record(directory: Path, name: str) -> None:
    path = directory / name
    manifest_path = directory / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["artifacts"]:
        if record["path"] == name:
            record["bytes"] = path.stat().st_size
            record["sha256"] = _sha256(path)
            break
    else:
        raise AssertionError(f"manifest does not record {name}")
    _write_json(manifest_path, manifest)


class KrennTargetArtifactTest(unittest.TestCase):
    def _rational_problem(self):
        tensor_map = MatchingTensorMap(2, 2)
        witness = SparseWitness.from_coordinates(
            2,
            2,
            {
                (0, 1, 0, 0): Fraction(1, 2),
                (0, 1, 0, 1): -3,
                (0, 1, 1, 0): Fraction(7, 5),
            },
        )
        return tensor_map, tensor_map.evaluate_exact(witness), witness

    def test_rational_target_and_witness_round_trip_exactly(self):
        tensor_map, target, witness = self._rational_problem()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            certificate_path = write_target_artifact_bundle(
                tensor_map, target, directory, witness=witness
            )
            bundle = verify_target_artifact_bundle(directory)

            self.assertEqual(certificate_path, directory / CERTIFICATE_FILE)
            self.assertTrue(bundle.exact)
            self.assertEqual(bundle.target, target)
            self.assertEqual(bundle.witness, witness)
            self.assertEqual(
                bundle.target.coefficient((0, 0)), Fraction(1, 2)
            )
            self.assertEqual(
                bundle.target.coefficient((1, 0)), Fraction(7, 5)
            )
            self.assertEqual(
                bundle.certificate["status"],
                "exact-rational-affine-image-witness-certified",
            )
            claims = bundle.certificate["claims"]
            self.assertTrue(claims["witness_emitted"])
            self.assertTrue(
                claims[
                    "exact_rational_affine_image_membership_certified"
                ]
            )
            self.assertTrue(
                claims["complex_affine_image_membership_certified"]
            )
            self.assertFalse(
                claims["border_image_membership_certified"]
            )
            self.assertFalse(
                claims["nonexistence_certificate_emitted"]
            )

            with np.load(
                directory / TENSOR_MAP_FILE, allow_pickle=False
            ) as archive:
                self.assertEqual(
                    set(archive.files), TENSOR_MAP_ARCHIVE_KEYS
                )
                self.assertNotIn("rhs_values", archive.files)
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {
                    TENSOR_MAP_FILE,
                    TARGET_FILE,
                    WITNESS_FILE,
                    CERTIFICATE_FILE,
                    MANIFEST_FILE,
                },
            )
            manifest = json.loads(
                (directory / MANIFEST_FILE).read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(
                    len(record["sha256"]) == 64
                    for record in manifest["artifacts"]
                )
            )
            self.assertEqual(
                {
                    record["path"]
                    for record in manifest["inputs"]
                },
                {
                    "experiments/krenn_quantum_graph/"
                    "independent_verifier.py",
                    "experiments/krenn_quantum_graph/system.py",
                    "experiments/krenn_quantum_graph/targets.py",
                    "experiments/krenn_quantum_graph/tensor_map.py",
                    "experiments/krenn_quantum_graph/transport.py",
                    "experiments/krenn_quantum_graph/witness.py",
                },
            )

    def test_without_witness_bundle_is_structure_only_and_fail_closed(self):
        tensor_map = MatchingTensorMap(4, 2)
        target = ColoringTarget.from_sparse(
            4,
            2,
            {
                (0, 0, 1, 1): Fraction(5, 7),
                (1, 0, 1, 0): -2,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_target_artifact_bundle(
                tensor_map, target, directory
            )
            bundle = verify_target_artifact_bundle(directory)

            self.assertIsNone(bundle.witness)
            self.assertFalse((directory / WITNESS_FILE).exists())
            self.assertEqual(
                bundle.certificate["status"],
                "target-problem-structure-only-no-witness",
            )
            claims = bundle.certificate["claims"]
            for key in (
                "witness_emitted",
                "exact_rational_affine_image_membership_certified",
                "complex_affine_image_membership_certified",
                "projective_image_membership_certified",
                "local_diagonal_orbit_membership_certified",
                "border_image_membership_certified",
                "nonexistence_certificate_emitted",
            ):
                self.assertFalse(claims[key])
            self.assertIn("Without a witness", claims["statement"])

    def test_writer_rejects_nonmapping_witness_before_emitting_claims(self):
        tensor_map = MatchingTensorMap(2, 2)
        target = ColoringTarget.from_sparse(
            2, 2, {(0, 0): 1}
        )
        zero = SparseWitness.from_index_values(2, 2, ())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            with self.assertRaisesRegex(
                KrennTargetArtifactError, "does not map exactly"
            ):
                write_target_artifact_bundle(
                    tensor_map, target, directory, witness=zero
                )
            self.assertFalse(directory.exists())

    def test_raw_target_corruption_is_caught_by_sha256_manifest(self):
        tensor_map, target, witness = self._rational_problem()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_target_artifact_bundle(
                tensor_map, target, directory, witness=witness
            )
            payload = json.loads(
                (directory / TARGET_FILE).read_text(encoding="utf-8")
            )
            payload["entries"][0]["numerator"] += 1
            _write_json(directory / TARGET_FILE, payload)
            with self.assertRaisesRegex(
                KrennTargetArtifactError, "SHA256 manifest replay"
            ):
                verify_target_artifact_bundle(directory)

    def test_refreshed_target_hash_cannot_hide_semantic_corruption(self):
        tensor_map, target, witness = self._rational_problem()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_target_artifact_bundle(
                tensor_map, target, directory, witness=witness
            )
            payload = json.loads(
                (directory / TARGET_FILE).read_text(encoding="utf-8")
            )
            payload["entries"][0]["numerator"] += 2
            _write_json(directory / TARGET_FILE, payload)
            _refresh_manifest_record(directory, TARGET_FILE)
            with self.assertRaisesRegex(
                KrennTargetArtifactError,
                "witness failed primary exact target replay",
            ):
                verify_target_artifact_bundle(directory)

    def test_refreshed_map_hash_cannot_hide_matching_corruption(self):
        tensor_map, target, witness = self._rational_problem()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_target_artifact_bundle(
                tensor_map, target, directory, witness=witness
            )
            archive_path = directory / TENSOR_MAP_FILE
            with np.load(archive_path, allow_pickle=False) as archive:
                offsets = archive["equation_offsets"].copy()
                monomials = archive[
                    "monomial_variable_indices"
                ].copy()
            monomials[0, 0] = (
                int(monomials[0, 0]) + 1
            ) % tensor_map.variable_count
            with archive_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    equation_offsets=offsets,
                    monomial_variable_indices=monomials,
                )
            _refresh_manifest_record(directory, TENSOR_MAP_FILE)
            with self.assertRaisesRegex(
                KrennTargetArtifactError,
                "independent matching reconstruction",
            ):
                verify_target_artifact_bundle(directory)

    def test_duplicate_npz_member_is_rejected_after_hash_refresh(self):
        tensor_map, target, witness = self._rational_problem()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_target_artifact_bundle(
                tensor_map, target, directory, witness=witness
            )
            archive_path = directory / TENSOR_MAP_FILE
            with ZipFile(archive_path) as archive:
                duplicate = archive.read("equation_offsets.npy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with ZipFile(archive_path, "a") as archive:
                    archive.writestr(
                        "equation_offsets.npy", duplicate
                    )
            _refresh_manifest_record(directory, TENSOR_MAP_FILE)
            with self.assertRaisesRegex(
                KrennTargetArtifactError,
                "ZIP members are not unique",
            ):
                verify_target_artifact_bundle(directory)

    def test_unexpected_subdirectory_is_rejected(self):
        tensor_map = MatchingTensorMap(2, 2)
        target = ColoringTarget.from_sparse(2, 2, ())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_target_artifact_bundle(
                tensor_map, target, directory
            )
            (directory / "unrecorded").mkdir()
            with self.assertRaisesRegex(
                KrennTargetArtifactError, "directory inventory"
            ):
                verify_target_artifact_bundle(directory)

    def test_refreshed_certificate_hash_cannot_invent_image_claim(self):
        tensor_map = MatchingTensorMap(2, 2)
        target = ColoringTarget.from_sparse(
            2, 2, {(0, 1): Fraction(2, 3)}
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_target_artifact_bundle(
                tensor_map, target, directory
            )
            certificate_path = directory / CERTIFICATE_FILE
            certificate = json.loads(
                certificate_path.read_text(encoding="utf-8")
            )
            certificate["claims"][
                "complex_affine_image_membership_certified"
            ] = True
            _write_json(certificate_path, certificate)
            _refresh_manifest_record(directory, CERTIFICATE_FILE)
            with self.assertRaisesRegex(
                KrennTargetArtifactError,
                "certificate failed semantic replay",
            ):
                verify_target_artifact_bundle(directory)

    def test_manifest_paths_and_module_dependencies_fail_closed(self):
        tensor_map = MatchingTensorMap(2, 2)
        target = ColoringTarget.from_sparse(2, 2, ())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            write_target_artifact_bundle(
                tensor_map, target, directory
            )
            manifest_path = directory / MANIFEST_FILE
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["artifacts"][0]["path"] = "../certificate.json"
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                KrennTargetArtifactError,
                "manifest replay|inventory",
            ):
                verify_target_artifact_bundle(directory)

        import experiments.krenn_quantum_graph.target_artifacts as subject

        source = Path(subject.__file__).read_text(encoding="utf-8")
        self.assertNotIn("section12_instantiation", source)
        self.assertNotIn(
            "experiments.krenn_quantum_graph.artifacts", source
        )


if __name__ == "__main__":
    unittest.main()
