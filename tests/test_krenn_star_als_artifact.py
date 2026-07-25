import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.star_alternating_search import (
    DEFAULT_SEEDS,
    run_star_als_campaign,
)
from experiments.krenn_quantum_graph.star_als_artifact import (
    ALL_FILES,
    MANIFEST_FILE,
    SUMMARY_FILE,
    KrennStarALSArtifactError,
    generate_star_als_summary_bundle,
    summarize_star_als_campaign,
    verify_star_als_summary_bundle,
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


class KrennStarALSArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.campaign = cls.root / "scratch-campaign"
        cap_pairs = (
            (1.0, 4.0),
            (1.25, 5.0),
            (1.5, 6.0),
            (2.0, 7.0),
        )
        for initialization in ("random", "natural-perturbed"):
            for index, (linf, l2) in enumerate(cap_pairs):
                run_star_als_campaign(
                    output_directory=(
                        cls.campaign
                        / initialization
                        / f"cap-{index}"
                    ),
                    radii=(linf,),
                    global_l2_radii=(l2,),
                    seeds=DEFAULT_SEEDS,
                    maximum_sweeps=1,
                    maximum_seconds_per_run=60.0,
                    patience_apex_updates=1,
                    initialization=initialization,
                )
        cls.bundle = cls.root / "summary-bundle"
        cls.summary_path = generate_star_als_summary_bundle(
            cls.campaign,
            cls.bundle,
            parallel_group_workers=8,
            interpreter="python",
        )
        cls.loaded = verify_star_als_summary_bundle(
            cls.bundle,
            campaign_root=cls.campaign,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copied_campaign(self, label: str) -> Path:
        destination = self.root / label
        shutil.copytree(self.campaign, destination)
        for manifest_path in destination.rglob(MANIFEST_FILE):
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            payload["output_directory"] = str(manifest_path.parent)
            _write_json(manifest_path, payload)
        return destination

    def copied_bundle(self, label: str) -> Path:
        destination = self.root / label
        shutil.copytree(self.bundle, destination)
        return destination

    def test_summary_round_trips_and_replays_all_vectors(self):
        self.assertEqual(self.summary_path, self.bundle / SUMMARY_FILE)
        self.assertEqual(
            {path.name for path in self.bundle.iterdir()},
            set(ALL_FILES),
        )
        self.assertTrue(self.loaded.valid)
        summary = self.loaded.summary
        self.assertEqual(summary["campaign"]["groups"], 8)
        self.assertEqual(summary["campaign"]["runs"], 24)
        self.assertEqual(
            summary["campaign"]["fixed_seeds"], list(DEFAULT_SEEDS)
        )
        self.assertEqual(
            summary["verification"]["result_vectors_replayed"], 48
        )
        self.assertEqual(
            summary["verification"][
                "checkpoint_initial_vectors_replayed"
            ],
            24,
        )
        self.assertTrue(
            summary["aggregate"][
                "all_recorded_weights_finite_and_bounded"
            ]
        )
        self.assertEqual(
            summary["aggregate"]["exact_verification"][
                "attempted_runs"
            ],
            0,
        )

    def test_claim_boundaries_name_single_chart_and_nonproofs(self):
        boundary = self.loaded.summary["claim_boundary"]
        self.assertEqual(
            boundary["search_scope"],
            "one fixed 15-anchor open gauge chart only",
        )
        self.assertFalse(boundary["legacy_eight_charts_searched"])
        self.assertFalse(boundary["three_pivot_orbit_cover_searched"])
        self.assertFalse(boundary["all_affine_solutions_covered"])
        self.assertFalse(
            boundary["numerical_zero_is_exact_counterexample"]
        )
        self.assertFalse(
            boundary["bounded_miss_is_nonexistence_proof"]
        )
        self.assertEqual(
            boundary["finite_affine_membership_status"], "undecided"
        )

    def test_group_and_input_inventories_are_exact(self):
        summary = self.loaded.summary
        self.assertEqual(len(summary["groups"]), 8)
        self.assertEqual(len(summary["input_file_ledger"]), 56)
        self.assertEqual(
            len({
                (
                    row["initialization"],
                    row["global_linf_cap"],
                    row["global_l2_cap"],
                )
                for row in summary["groups"]
            }),
            8,
        )
        self.assertTrue(all(
            "--maximum-seconds-per-run 60"
            in row["canonical_replay_command"]
            for row in summary["groups"]
        ))

    def test_result_checkpoint_disagreement_is_rejected(self):
        campaign = self.copied_campaign("checkpoint-corruption")
        checkpoint = next(campaign.rglob("*.checkpoint.json"))
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        payload["state"]["weights"][0][0] += 0.25
        _write_json(checkpoint, payload)
        with self.assertRaisesRegex(
            KrennStarALSArtifactError, "checkpoint disagree"
        ):
            summarize_star_als_campaign(campaign)

    def test_paired_weight_corruption_is_rejected_by_chart_replay(self):
        campaign = self.copied_campaign("weight-corruption")
        result = next(campaign.rglob("*.result.json"))
        checkpoint = result.with_name(
            result.name.replace(".result.json", ".checkpoint.json")
        )
        result_payload = json.loads(
            result.read_text(encoding="utf-8")
        )
        checkpoint_payload = json.loads(
            checkpoint.read_text(encoding="utf-8")
        )
        anchor = result_payload["gauge_chart"]["anchor_indices"][0]
        result_payload["weights"][anchor] = [2.0, 0.0]
        checkpoint_payload["state"]["weights"][anchor] = [2.0, 0.0]
        _write_json(result, result_payload)
        _write_json(checkpoint, checkpoint_payload)
        with self.assertRaisesRegex(
            KrennStarALSArtifactError, "15-anchor"
        ):
            summarize_star_als_campaign(campaign)

    def test_missing_group_fails_closed(self):
        campaign = self.copied_campaign("missing-group")
        manifest = next(campaign.rglob(MANIFEST_FILE))
        shutil.rmtree(manifest.parent)
        with self.assertRaisesRegex(
            KrennStarALSArtifactError, "exactly eight"
        ):
            summarize_star_als_campaign(campaign)

    def test_refreshed_bundle_hash_cannot_strengthen_claim(self):
        bundle = self.copied_bundle("claim-corruption")
        summary_path = bundle / SUMMARY_FILE
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        summary["claim_boundary"][
            "bounded_miss_is_nonexistence_proof"
        ] = True
        _write_json(summary_path, summary)
        manifest_path = bundle / MANIFEST_FILE
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["artifact"]["bytes"] = summary_path.stat().st_size
        manifest["artifact"]["sha256"] = hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest()
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            KrennStarALSArtifactError, "claim boundary"
        ):
            verify_star_als_summary_bundle(
                bundle, campaign_root=self.campaign
            )

    def test_extra_bundle_file_is_rejected(self):
        bundle = self.copied_bundle("extra-bundle-file")
        (bundle / "large-checkpoint.json").write_text(
            "{}", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            KrennStarALSArtifactError, "inventory"
        ):
            verify_star_als_summary_bundle(
                bundle, campaign_root=self.campaign
            )


if __name__ == "__main__":
    unittest.main()
