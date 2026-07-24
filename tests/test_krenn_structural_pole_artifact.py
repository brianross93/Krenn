import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.structural_pole_artifact import (
    ALL_FILES,
    ARTIFACT_FILES,
    CERTIFICATE_FILE,
    CERTIFICATE_SCHEMA,
    COMMITTED_BUNDLE_DIRECTORY,
    EXACT_CHECK_NAMES,
    KrennStructuralPoleArtifactError,
    MANIFEST_FILE,
    MATCHING_CIRCUIT_FILE,
    ROOT,
    SOURCE_INPUTS,
    STRUCTURAL_POLE_FILE,
    VERTICAL_COMPONENT_FILE,
    generate_structural_pole_bundle,
    verify_structural_pole_bundle,
)


def _canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload) -> None:
    path.write_bytes(_canonical_json_bytes(payload))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_artifact_record(directory: Path, filename: str) -> None:
    manifest_path = directory / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / filename
    for record in manifest["artifacts"]:
        if record["path"] == filename:
            record["bytes"] = path.stat().st_size
            record["sha256"] = _sha256(path)
            break
    else:
        raise AssertionError(f"missing artifact record {filename}")
    _write_json(manifest_path, manifest)


class KrennStructuralPoleArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "structural-pole"
        cls.certificate_path = generate_structural_pole_bundle(
            cls.output
        )
        cls.loaded = verify_structural_pole_bundle(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copied_bundle(self, name: str) -> Path:
        destination = Path(self.temporary.name) / name
        shutil.copytree(self.output, destination)
        return destination

    def test_bundle_round_trips_with_fail_closed_claims(self):
        self.assertEqual(
            self.certificate_path,
            self.output / CERTIFICATE_FILE,
        )
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            set(ALL_FILES),
        )
        self.assertTrue(self.loaded.exact)
        self.assertFalse(self.loaded.long_closure_replayed)
        self.assertEqual(
            tuple(name for name, _value in self.loaded.checks),
            EXACT_CHECK_NAMES,
        )
        self.assertTrue(
            all(value for _name, value in self.loaded.checks)
        )

        certificate = self.loaded.certificate
        self.assertEqual(certificate["schema"], CERTIFICATE_SCHEMA)
        exact = certificate["exact_results"]
        self.assertEqual(exact["finite_exact_support_lower_bound"], 22)
        self.assertTrue(exact["support_sizes_through_21_excluded"])
        self.assertFalse(
            exact["natural_coordinate_retention_assumed"]
        )
        self.assertEqual(exact["marked_unique_defect_seed_charts"], 360)
        self.assertFalse(
            exact["distinct_global_component_count_decided"]
        )

        weights = certificate["weight_status"]
        self.assertFalse(
            weights["known_Laurent_branch_weights_remain_finite"]
        )
        self.assertEqual(
            weights["full_color_diagonal_gauge_invariant_quantity"],
            "u*Q",
        )
        self.assertFalse(weights["finite_bounded_candidate_found"])

        candidate = certificate["exact_candidate_status"]
        self.assertFalse(candidate["finite_exact_witness_found"])
        self.assertFalse(candidate["witness_bundle_emitted"])
        boundary = certificate["claim_boundary"]
        self.assertFalse(
            boundary["support_22_or_larger_witness_excluded"]
        )
        self.assertFalse(
            boundary["global_finite_GHZ_nonexistence_proved"]
        )
        self.assertFalse(boundary["D_in_radical_J_mix_decided"])
        self.assertEqual(
            boundary["exact_affine_GHZ_membership_status"],
            "undecided",
        )

    def test_payloads_cross_check_the_exact_receipts(self):
        structural = self.loaded.structural_pole
        matching = self.loaded.matching_circuit
        vertical = self.loaded.vertical_component
        self.assertFalse(
            structural["seed_component_atlas"]["claim_boundary"][
                "distinct_global_component_count_decided"
            ]
        )
        self.assertTrue(
            matching["claims"][
                "fixed_coloring_localized_matching_lattice_complete"
            ]
        )
        support = vertical["recorded_support21_global_audit"]
        self.assertEqual(support["totals"], {
            "nodes_examined": 12_992_269,
            "odd_mixed_circuit_terminals": 8,
            "pure_cancellation_terminals": 4,
            "singleton_free_terminals": 12,
        })
        self.assertEqual(
            support["finite_exact_support_lower_bound"], 22
        )
        self.assertEqual(
            len(support["terminal_certificates"]), 12
        )

    def test_manifest_has_exact_artifact_and_source_ledgers(self):
        manifest = self.loaded.manifest
        self.assertEqual(
            [record["path"] for record in manifest["artifacts"]],
            list(sorted(ARTIFACT_FILES)),
        )
        self.assertEqual(
            [record["path"] for record in manifest["inputs"]],
            list(sorted(SOURCE_INPUTS)),
        )
        self.assertEqual(
            manifest["producer"]["path"],
            "experiments/krenn_quantum_graph/"
            "structural_pole_artifact.py",
        )
        for record in manifest["artifacts"]:
            path = self.output / record["path"]
            self.assertEqual(record["bytes"], path.stat().st_size)
            self.assertEqual(record["sha256"], _sha256(path))
        for record in (
            manifest["producer"],
            *manifest["inputs"],
        ):
            self.assertEqual(
                record["hash_mode"], "canonical-lf-text-v1"
            )
            source = ROOT / record["path"]
            canonical = source.read_bytes().replace(
                b"\r\n", b"\n"
            ).replace(b"\r", b"\n")
            self.assertEqual(record["canonical_bytes"], len(canonical))
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(canonical).hexdigest(),
            )

    def test_raw_corruption_fails_hash_replay(self):
        directory = self.copied_bundle("raw-corruption")
        path = directory / STRUCTURAL_POLE_FILE
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "hash replay"
        ):
            verify_structural_pole_bundle(directory)

    def test_refreshed_hash_cannot_strengthen_claims_or_lower_bound(self):
        claim = self.copied_bundle("claim-corruption")
        path = claim / CERTIFICATE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["claim_boundary"][
            "global_finite_GHZ_nonexistence_proved"
        ] = True
        _write_json(path, payload)
        _refresh_artifact_record(claim, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "semantic replay"
        ):
            verify_structural_pole_bundle(claim)

        bound = self.copied_bundle("bound-corruption")
        path = bound / CERTIFICATE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["exact_results"][
            "finite_exact_support_lower_bound"
        ] = 23
        _write_json(path, payload)
        _refresh_artifact_record(bound, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "semantic replay"
        ):
            verify_structural_pole_bundle(bound)

    def test_refreshed_hash_cannot_change_receipt_or_terminal_relation(self):
        receipt = self.copied_bundle("receipt-corruption")
        path = receipt / VERTICAL_COMPONENT_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["recorded_support21_global_audit"][
            "root_receipts"
        ][0]["nodes_examined"] += 1
        _write_json(path, payload)
        _refresh_artifact_record(receipt, VERTICAL_COMPONENT_FILE)
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "semantic replay"
        ):
            verify_structural_pole_bundle(receipt)

        relation = self.copied_bundle("relation-corruption")
        path = relation / VERTICAL_COMPONENT_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        certificate = payload[
            "recorded_support21_global_audit"
        ]["terminal_certificates"][0]
        certificate["mixed_relation"][0]["coefficient"] += 1
        _write_json(path, payload)
        _refresh_artifact_record(relation, VERTICAL_COMPONENT_FILE)
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "semantic replay"
        ):
            verify_structural_pole_bundle(relation)

    def test_missing_extra_and_path_escape_are_rejected(self):
        missing = self.copied_bundle("missing")
        (missing / MATCHING_CIRCUIT_FILE).unlink()
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "inventory"
        ):
            verify_structural_pole_bundle(missing)

        extra = self.copied_bundle("extra")
        (extra / "unrecorded.txt").write_text(
            "extra", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "inventory"
        ):
            verify_structural_pole_bundle(extra)

        escaped = self.copied_bundle("escaped")
        manifest_path = escaped / MANIFEST_FILE
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["artifacts"][0]["path"] = "../certificate.json"
        _write_json(manifest_path, manifest)
        with self.assertRaises(KrennStructuralPoleArtifactError):
            verify_structural_pole_bundle(escaped)

    def test_duplicate_nonfinite_and_noncanonical_json_are_rejected(self):
        duplicate = self.copied_bundle("duplicate")
        path = duplicate / CERTIFICATE_FILE
        raw = path.read_bytes()
        path.write_bytes(
            b'{\n  "schema": "duplicate",'
            + raw[1:]
        )
        _refresh_artifact_record(duplicate, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "duplicate JSON key"
        ):
            verify_structural_pole_bundle(duplicate)

        nonfinite = self.copied_bundle("nonfinite")
        path = nonfinite / CERTIFICATE_FILE
        raw = path.read_text(encoding="utf-8")
        raw = raw.replace(
            '"finite_exact_witness_found": false',
            '"finite_exact_witness_found": NaN',
            1,
        )
        path.write_text(raw, encoding="utf-8", newline="\n")
        _refresh_artifact_record(nonfinite, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "non-finite JSON"
        ):
            verify_structural_pole_bundle(nonfinite)

        noncanonical = self.copied_bundle("noncanonical")
        path = noncanonical / CERTIFICATE_FILE
        path.write_bytes(path.read_bytes() + b" ")
        _refresh_artifact_record(noncanonical, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError, "not canonical"
        ):
            verify_structural_pole_bundle(noncanonical)

    def test_source_ledger_tamper_is_rejected(self):
        directory = self.copied_bundle("source-ledger")
        manifest_path = directory / MANIFEST_FILE
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["inputs"][0]["sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            KrennStructuralPoleArtifactError,
            "source-ledger replay",
        ):
            verify_structural_pole_bundle(directory)

    @unittest.skipUnless(
        os.environ.get("KRENN_RUN_LONG_TESTS") == "1",
        "set KRENN_RUN_LONG_TESTS=1 for 12,992,269-node replay",
    )
    def test_long_verifier_replays_all_support21_roots(self):
        loaded = verify_structural_pole_bundle(
            self.output,
            replay_long_closure=True,
        )
        self.assertTrue(loaded.exact)
        self.assertTrue(loaded.long_closure_replayed)

    def test_committed_bundle_replays_if_it_already_exists(self):
        self.assertTrue(
            COMMITTED_BUNDLE_DIRECTORY.is_dir(),
            "the committed structural-pole bundle is required",
        )
        loaded = verify_structural_pole_bundle(
            COMMITTED_BUNDLE_DIRECTORY
        )
        self.assertTrue(loaded.exact)
        self.assertFalse(loaded.long_closure_replayed)


if __name__ == "__main__":
    unittest.main()
