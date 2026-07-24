import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.radical_obstruction_artifact import (
    ALL_FILES,
    ARTIFACT_FILES,
    CERTIFICATE_FILE,
    COMMITTED_BUNDLE_DIRECTORY,
    EXACT_CHECK_NAMES,
    KrennRadicalArtifactError,
    MANIFEST_FILE,
    RECONNAISSANCE_FILE,
    ROOT,
    SOURCE_INPUTS,
    generate_radical_obstruction_bundle,
    verify_radical_obstruction_bundle,
)


def _canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
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


class KrennRadicalObstructionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "radical-obstruction"
        cls.certificate_path = generate_radical_obstruction_bundle(
            cls.output
        )
        cls.loaded = verify_radical_obstruction_bundle(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copied_bundle(self, name: str) -> Path:
        destination = Path(self.temporary.name) / name
        shutil.copytree(self.output, destination)
        return destination

    def test_bundle_round_trips_with_independent_exact_checks(self):
        self.assertEqual(
            self.certificate_path,
            self.output / CERTIFICATE_FILE,
        )
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            set(ALL_FILES),
        )
        self.assertTrue(self.loaded.exact)
        self.assertEqual(
            tuple(name for name, _value in self.loaded.checks),
            EXACT_CHECK_NAMES,
        )
        self.assertTrue(
            all(value for _name, value in self.loaded.checks)
        )
        self.assertTrue(
            self.loaded.independent_checks[
                "switch_columns_replayed_symbolically"
            ]
        )
        self.assertTrue(
            self.loaded.independent_checks[
                "unimodular_minor_replayed"
            ]
        )
        self.assertTrue(
            self.loaded.independent_checks[
                "exact_contraction_minors_replayed"
            ]
        )
        self.assertTrue(
            self.loaded.independent_checks[
                "mechanism_ten_K3_3_cubic_circuits_replayed"
            ]
        )
        self.assertTrue(
            self.loaded.independent_checks[
                "mechanism_thirty_nine_spill_terms_replayed"
            ]
        )
        certificate = self.loaded.certificate
        preflight = certificate["k2_preflight"]
        support = certificate["support_obstruction"]
        mechanism = certificate["mechanism_audit"]
        self.assertFalse(
            preflight["memory_preflight"][
                "full_orbit_construction_authorized"
            ]
        )
        self.assertEqual(
            support["support_row_search"]["basis_determinant"],
            -1,
        )
        self.assertEqual(
            mechanism["matching_incidence"]["quadratic"][
                "distinct_incidence_sums"
            ],
            120,
        )
        self.assertEqual(
            mechanism["matching_incidence"]["cubic"][
                "collision_fibers"
            ],
            10,
        )
        claims = support["claims"]
        self.assertFalse(claims["D_squared_in_J_mix_decided"])
        self.assertFalse(claims["D_in_radical_J_mix_decided"])
        self.assertFalse(claims["global_GHZ_nonexistence_proved"])
        self.assertFalse(
            claims["exact_affine_GHZ_membership_decided"]
        )
        reconnaissance = self.loaded.reconnaissance
        self.assertEqual(
            reconnaissance["modular_rank_diagnostics"]["role"],
            "nonproof-reconnaissance",
        )
        self.assertFalse(
            reconnaissance["claims"][
                "D_squared_global_nonmembership_proved"
            ]
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
        for record in (
            manifest["producer"],
            *manifest["inputs"],
        ):
            source = ROOT / record["path"]
            canonical = source.read_bytes().replace(
                b"\r\n", b"\n"
            ).replace(b"\r", b"\n")
            self.assertEqual(
                record["canonical_bytes"], len(canonical)
            )
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(canonical).hexdigest(),
            )

    def test_raw_certificate_corruption_fails_hash_replay(self):
        directory = self.copied_bundle("raw-corruption")
        path = directory / CERTIFICATE_FILE
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "hash replay"
        ):
            verify_radical_obstruction_bundle(directory)

    def test_refreshed_hash_cannot_escalate_global_claims(self):
        directory = self.copied_bundle("claim-corruption")
        path = directory / CERTIFICATE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["support_obstruction"]["claims"][
            "global_GHZ_nonexistence_proved"
        ] = True
        _write_json(path, payload)
        _refresh_artifact_record(directory, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "semantic replay"
        ):
            verify_radical_obstruction_bundle(directory)

    def test_refreshed_hash_cannot_corrupt_symbolic_column(self):
        directory = self.copied_bundle("column-corruption")
        path = directory / CERTIFICATE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        search = payload["support_obstruction"][
            "support_row_search"
        ]
        search["basis_constraints"][0]["projected_entries"][0][1] += 1
        # Refresh the redundant determinant metadata too.  The producer
        # replay and the separate stdlib verifier still reject the column.
        search["basis_determinant"] = -1
        _write_json(path, payload)
        _refresh_artifact_record(directory, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "semantic replay"
        ):
            verify_radical_obstruction_bundle(directory)

    def test_refreshed_hash_cannot_corrupt_exact_determinant(self):
        directory = self.copied_bundle("determinant-corruption")
        path = directory / CERTIFICATE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["support_obstruction"]["support_row_search"][
            "basis_determinant"
        ] = 1
        _write_json(path, payload)
        _refresh_artifact_record(directory, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "semantic replay"
        ):
            verify_radical_obstruction_bundle(directory)

    def test_refreshed_hash_cannot_promote_reconnaissance(self):
        directory = self.copied_bundle(
            "reconnaissance-corruption"
        )
        path = directory / RECONNAISSANCE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["claims"][
            "D_squared_global_nonmembership_proved"
        ] = True
        _write_json(path, payload)
        _refresh_artifact_record(directory, RECONNAISSANCE_FILE)
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "semantic replay"
        ):
            verify_radical_obstruction_bundle(directory)

    def test_refreshed_hash_cannot_corrupt_mechanism_spill(self):
        directory = self.copied_bundle("mechanism-corruption")
        path = directory / CERTIFICATE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        spill = payload["mechanism_audit"][
            "support21_local_mechanism"
        ]["canonical_first_support"]["unrestricted_expansion"]
        spill["spill_terms"][0][
            "lifted_degree_nine_monomial"
        ][0] += 1
        spill["spill_fingerprint_sha256"] = hashlib.sha256(
            json.dumps(
                spill["spill_terms"],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        _write_json(path, payload)
        _refresh_artifact_record(directory, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "semantic replay"
        ):
            verify_radical_obstruction_bundle(directory)

    def test_inventory_path_escape_and_duplicate_json_are_rejected(self):
        missing = self.copied_bundle("missing")
        (missing / CERTIFICATE_FILE).unlink()
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "inventory"
        ):
            verify_radical_obstruction_bundle(missing)

        extra = self.copied_bundle("extra")
        (extra / "unrecorded.txt").write_text(
            "extra", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "inventory"
        ):
            verify_radical_obstruction_bundle(extra)

        escaped = self.copied_bundle("escaped")
        manifest_path = escaped / MANIFEST_FILE
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["artifacts"][0]["path"] = "../certificate.json"
        _write_json(manifest_path, manifest)
        with self.assertRaises(KrennRadicalArtifactError):
            verify_radical_obstruction_bundle(escaped)

        duplicate = self.copied_bundle("duplicate")
        certificate_path = duplicate / CERTIFICATE_FILE
        raw = certificate_path.read_text(encoding="utf-8")
        certificate_path.write_text(
            raw.replace(
                '{\n  "k2_preflight"',
                '{\n  "schema": "duplicate",\n  "k2_preflight"',
                1,
            ),
            encoding="utf-8",
        )
        _refresh_artifact_record(duplicate, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennRadicalArtifactError, "duplicate JSON key"
        ):
            verify_radical_obstruction_bundle(duplicate)

    def test_committed_bundle_replays_if_it_already_exists(self):
        if not COMMITTED_BUNDLE_DIRECTORY.exists():
            self.skipTest("no committed radical-obstruction bundle exists")
        loaded = verify_radical_obstruction_bundle(
            COMMITTED_BUNDLE_DIRECTORY
        )
        self.assertTrue(loaded.exact)


if __name__ == "__main__":
    unittest.main()
