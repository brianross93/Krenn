import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.ternary_milestone_artifact import (
    ALL_FILES,
    KrennTernaryArtifactError,
    MANIFEST_FILE,
    ORBIT_FILE,
    SEARCH_FILE,
    SOURCE_INPUTS,
    default_search_plan,
    generate_ternary_milestone_bundle,
    verify_ternary_milestone_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class KrennTernaryMilestoneArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "ternary-milestone"
        generate_ternary_milestone_bundle(cls.output)
        cls.bundle = verify_ternary_milestone_bundle(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copied_bundle(self, name: str) -> Path:
        destination = Path(self.temporary.name) / name
        shutil.copytree(self.output, destination)
        return destination

    def test_default_bundle_round_trips_with_bounded_search(self):
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            set(ALL_FILES),
        )
        self.assertTrue(self.bundle.exact)
        self.assertEqual(
            self.bundle.result.plan, default_search_plan()
        )
        self.assertEqual(
            self.bundle.result.termination, "node-cap-reached"
        )
        self.assertEqual(
            self.bundle.result.nodes_examined_this_run, 32
        )
        self.assertEqual(
            self.bundle.result.checkpoint.total_nodes_examined, 32
        )
        self.assertEqual(
            self.bundle.result.checkpoint.best_candidate
            .nonzero_residual_count,
            1,
        )
        self.assertEqual(
            self.bundle.certificate["counts"]["seed_orbits"], 8
        )
        self.assertEqual(
            self.bundle.certificate["counts"]["ordered_seeds"],
            3375,
        )

    def test_certificate_retains_every_no_proof_boundary(self):
        boundary = self.bundle.certificate["claim_boundary"]
        self.assertTrue(boundary["orbit_indexing_only"])
        self.assertTrue(boundary["bounded_search_only"])
        self.assertEqual(boundary["symmetry_weight_equalities"], 0)
        self.assertFalse(boundary["search_exhaustive"])
        self.assertFalse(boundary["tree_certificate_complete"])
        self.assertFalse(boundary["exhaustive_nonexistence"])
        self.assertFalse(boundary["nonexistence_proved"])
        self.assertFalse(boundary["solution_certified"])
        self.assertFalse(boundary["proof_over_C"])
        self.assertIn("explored nodes", boundary["statement"])

    def test_manifest_has_exact_hash_and_source_ledgers(self):
        manifest = json.loads(
            (self.output / MANIFEST_FILE).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [row["path"] for row in manifest["artifacts"]],
            ["bounded_search.json", "certificate.json", "orbit_census.json"],
        )
        self.assertEqual(
            [row["path"] for row in manifest["inputs"]],
            sorted(SOURCE_INPUTS),
        )
        self.assertIn(
            "experiments/krenn_quantum_graph/targets.py",
            [row["path"] for row in manifest["inputs"]],
        )
        self.assertTrue(
            all(
                row["hash_mode"] == "canonical-lf-text-v1"
                for row in manifest["inputs"]
            )
        )

    def test_missing_targets_source_ledger_entry_is_rejected(self):
        directory = self.copied_bundle("missing-targets-source")
        manifest_path = directory / MANIFEST_FILE
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["inputs"] = [
            row
            for row in manifest["inputs"]
            if row["path"] != "experiments/krenn_quantum_graph/targets.py"
        ]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            KrennTernaryArtifactError, "source ledger changed"
        ):
            verify_ternary_milestone_bundle(directory)

    def test_raw_corruption_fails_manifest_replay(self):
        directory = self.copied_bundle("raw-corruption")
        certificate = directory / "certificate.json"
        certificate.write_text(
            certificate.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            KrennTernaryArtifactError, "hash replay"
        ):
            verify_ternary_milestone_bundle(directory)

    def test_refreshed_orbit_hash_still_fails_semantic_replay(self):
        directory = self.copied_bundle("orbit-corruption")
        path = directory / ORBIT_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["orbits"][0]["ordered_seed_orbit_size"] += 1
        payload["counts"]["sum_of_ordered_orbit_sizes"] += 1
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _refresh_artifact_record(directory, ORBIT_FILE)
        with self.assertRaisesRegex(
            KrennTernaryArtifactError, "orbit census"
        ):
            verify_ternary_milestone_bundle(directory)

    def test_refreshed_search_hash_fails_deterministic_replay(self):
        directory = self.copied_bundle("search-corruption")
        path = directory / SEARCH_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["result"]["checkpoint"][
            "total_nodes_examined"
        ] += 1
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _refresh_artifact_record(directory, SEARCH_FILE)
        with self.assertRaisesRegex(
            KrennTernaryArtifactError,
            "deterministic full replay",
        ):
            verify_ternary_milestone_bundle(directory)


if __name__ == "__main__":
    unittest.main()
