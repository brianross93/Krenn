import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.source_ideal import (
    SOURCE_IDEAL_SCHEMA,
)
from experiments.krenn_quantum_graph.source_ideal_artifact import (
    ALL_FILES,
    ARTIFACT_FILES,
    CERTIFICATE_FILE,
    COMMITTED_BUNDLE_DIRECTORY,
    EXACT_CHECK_NAMES,
    KrennSourceIdealArtifactError,
    MANIFEST_FILE,
    ROOT,
    SOURCE_INPUTS,
    generate_source_ideal_bundle,
    verify_source_ideal_bundle,
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


class KrennSourceIdealArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "source-ideal"
        cls.certificate_path = generate_source_ideal_bundle(cls.output)
        cls.loaded = verify_source_ideal_bundle(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copied_bundle(self, name: str) -> Path:
        destination = Path(self.temporary.name) / name
        shutil.copytree(self.output, destination)
        return destination

    def test_bundle_round_trips_and_retains_the_exact_claim_boundary(self):
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
        certificate = self.loaded.certificate
        self.assertEqual(certificate["schema"], SOURCE_IDEAL_SCHEMA)
        claims = certificate["claims"]
        self.assertFalse(claims["D_in_J_mix_at_k1"])
        self.assertTrue(claims["exact_Q_nonmembership_proved"])
        self.assertFalse(
            claims["D_outside_radical_J_mix_proved"]
        )
        self.assertFalse(claims["GHZ_nonexistence_proved"])
        dual = certificate["exact_integer_dual"]
        self.assertEqual(dual["integer_lambda"], [1, -1])
        self.assertTrue(dual["lambda_transpose_B_zero"])
        self.assertEqual(dual["lambda_transpose_b"], -720)

    def test_manifest_has_exact_artifact_and_canonical_source_ledgers(self):
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
            "source_ideal_artifact.py",
        )
        certificate_record = manifest["artifacts"][0]
        certificate_path = self.output / CERTIFICATE_FILE
        self.assertEqual(
            certificate_record["bytes"],
            certificate_path.stat().st_size,
        )
        self.assertEqual(
            certificate_record["sha256"],
            _sha256(certificate_path),
        )
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

    def test_raw_certificate_corruption_fails_hash_replay(self):
        directory = self.copied_bundle("raw-corruption")
        path = directory / CERTIFICATE_FILE
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            KrennSourceIdealArtifactError, "hash replay"
        ):
            verify_source_ideal_bundle(directory)

    def test_refreshed_hash_cannot_invent_a_stronger_claim(self):
        directory = self.copied_bundle("claim-corruption")
        path = directory / CERTIFICATE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["claims"]["GHZ_nonexistence_proved"] = True
        _write_json(path, payload)
        _refresh_artifact_record(directory, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennSourceIdealArtifactError, "semantic replay"
        ):
            verify_source_ideal_bundle(directory)

    def test_refreshed_hash_cannot_corrupt_the_exact_dual(self):
        directory = self.copied_bundle("dual-corruption")
        path = directory / CERTIFICATE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["exact_integer_dual"]["lambda_transpose_b"] += 1
        _write_json(path, payload)
        _refresh_artifact_record(directory, CERTIFICATE_FILE)
        with self.assertRaisesRegex(
            KrennSourceIdealArtifactError, "semantic replay"
        ):
            verify_source_ideal_bundle(directory)

    def test_missing_extra_and_path_escape_are_rejected(self):
        missing = self.copied_bundle("missing")
        (missing / CERTIFICATE_FILE).unlink()
        with self.assertRaisesRegex(
            KrennSourceIdealArtifactError, "inventory"
        ):
            verify_source_ideal_bundle(missing)

        extra = self.copied_bundle("extra")
        (extra / "unrecorded.txt").write_text(
            "extra", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            KrennSourceIdealArtifactError, "inventory"
        ):
            verify_source_ideal_bundle(extra)

        escaped_artifact = self.copied_bundle("escaped-artifact")
        manifest_path = escaped_artifact / MANIFEST_FILE
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["artifacts"][0]["path"] = "../certificate.json"
        _write_json(manifest_path, manifest)
        with self.assertRaises(KrennSourceIdealArtifactError):
            verify_source_ideal_bundle(escaped_artifact)

        escaped_source = self.copied_bundle("escaped-source")
        manifest_path = escaped_source / MANIFEST_FILE
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["inputs"][0]["path"] = "../source_ideal.py"
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            KrennSourceIdealArtifactError, "unsafe|escaped"
        ):
            verify_source_ideal_bundle(escaped_source)

    def test_strict_manifest_and_certificate_schemas_reject_extra_keys(self):
        manifest_extra = self.copied_bundle("manifest-extra-key")
        manifest_path = manifest_extra / MANIFEST_FILE
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["unexpected"] = True
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            KrennSourceIdealArtifactError, "manifest schema"
        ):
            verify_source_ideal_bundle(manifest_extra)

        certificate_extra = self.copied_bundle(
            "certificate-extra-key"
        )
        certificate_path = certificate_extra / CERTIFICATE_FILE
        certificate = json.loads(
            certificate_path.read_text(encoding="utf-8")
        )
        certificate["unexpected"] = True
        _write_json(certificate_path, certificate)
        _refresh_artifact_record(
            certificate_extra, CERTIFICATE_FILE
        )
        with self.assertRaisesRegex(
            KrennSourceIdealArtifactError, "certificate schema"
        ):
            verify_source_ideal_bundle(certificate_extra)

    def test_committed_bundle_replays_if_it_already_exists(self):
        if not COMMITTED_BUNDLE_DIRECTORY.exists():
            self.skipTest("no committed source-ideal bundle exists")
        loaded = verify_source_ideal_bundle(
            COMMITTED_BUNDLE_DIRECTORY
        )
        self.assertTrue(loaded.exact)


if __name__ == "__main__":
    unittest.main()
