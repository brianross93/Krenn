import json
from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph import (
    star_variable_projection_campaign as campaign,
)
from experiments.krenn_quantum_graph import (
    numerical_continuation as continuation_module,
)
from experiments.krenn_quantum_graph.numerical_continuation import (
    KrennNumericalContinuationError,
    continue_result,
)


class KrennNumericalContinuationTests(unittest.TestCase):
    def _parent(self, directory: Path) -> Path:
        checkpoint = directory / "parent.checkpoint.json"
        config = campaign.TrajectoryConfig(
            orbit_index=0,
            seed=2026072501,
            caps=campaign.RadiusCaps.from_radius(2),
            maximum_iterations=1,
            maximum_seconds=120,
            patience=3,
        )
        result = campaign.run_trajectory(
            config, checkpoint_path=checkpoint
        )
        path = directory / "parent.result.json"
        campaign._write_json_atomic(path, result)
        return path

    def test_real_parent_continues_with_fail_closed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self._parent(root)
            manifest = continue_result(
                parent_result_path=parent,
                output_directory=root / "continued",
                label="smoke",
                maximum_iterations=1,
                maximum_seconds=120,
                patience=3,
                checkpoint_interval=1,
            )
            self.assertEqual(
                manifest["result"]["termination"],
                "maximum-iterations",
            )
            self.assertTrue(
                (root / "continued" / "smoke.result.json").is_file()
            )
            self.assertFalse(
                manifest["claim_boundary"][
                    "numerical_zero_is_exact_witness"
                ]
            )
            self.assertEqual(
                manifest["claim_boundary"][
                    "finite_affine_membership_status"
                ],
                "undecided",
            )
            self.assertTrue(
                manifest["parent_result"]["validation"][
                    "config_sha256_verified"
                ]
            )
            self.assertTrue(
                manifest["parent_result"]["validation"][
                    "payload_sha256_verified"
                ]
            )
            self.assertIn(
                "continuation_source_sha256",
                manifest["source_fingerprints"],
            )
            source = Path(continuation_module.__file__).read_text(
                encoding="utf-8"
            )
            self.assertLess(
                source.index("os.environ[_thread_variable]"),
                source.index("import numpy as np"),
            )

    def test_stale_parent_source_fingerprint_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self._parent(root)
            original = json.loads(parent.read_text("ascii"))

            stale_source = json.loads(json.dumps(original))
            stale_source["source_fingerprints"][
                "campaign_source_sha256"
            ] = (
                "0" * 64
            )
            campaign._write_json_atomic(parent, stale_source)
            with self.assertRaises(KrennNumericalContinuationError):
                continue_result(
                    parent_result_path=parent,
                    output_directory=root / "continued",
                    label="stale",
                    maximum_iterations=1,
                    maximum_seconds=120,
                    patience=3,
                    checkpoint_interval=1,
                )

            stale_config = json.loads(json.dumps(original))
            stale_config["config"]["rank_rtol"] = 2.0e-12
            campaign._write_json_atomic(parent, stale_config)
            with self.assertRaises(KrennNumericalContinuationError):
                continue_result(
                    parent_result_path=parent,
                    output_directory=root / "continued-config",
                    label="stale_config",
                    maximum_iterations=1,
                    maximum_seconds=120,
                    patience=3,
                    checkpoint_interval=1,
                )

            stale_u = json.loads(json.dumps(original))
            stale_u.pop("payload_sha256")
            stale_u["best_u"][3][0] += 0.5
            campaign._write_json_atomic(parent, stale_u)
            with self.assertRaises(KrennNumericalContinuationError):
                continue_result(
                    parent_result_path=parent,
                    output_directory=root / "continued-u",
                    label="stale_u",
                    maximum_iterations=1,
                    maximum_seconds=120,
                    patience=3,
                    checkpoint_interval=1,
                )


if __name__ == "__main__":
    unittest.main()
