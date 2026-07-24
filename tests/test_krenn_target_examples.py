from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.generate_target_examples import (
    BUNDLE_NAMES,
    KrennTargetExampleError,
    generate_all,
    verify_all,
)
from experiments.krenn_quantum_graph.target_artifacts import (
    WITNESS_FILE,
    verify_target_artifact_bundle,
)
from experiments.krenn_quantum_graph.targets import (
    heralded_ghz_target,
    unnormalized_dicke_target,
    unnormalized_graph_state_target,
    unnormalized_w_target,
)


class KrennTargetExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "target-examples"
        cls.generated_summary = generate_all(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_collection_has_exact_deterministic_inventory(self):
        self.assertEqual(
            {entry.name for entry in self.output.iterdir()},
            set(BUNDLE_NAMES),
        )
        self.assertEqual(
            tuple(self.generated_summary), BUNDLE_NAMES
        )
        self.assertTrue(
            all(
                row["exact_artifact_replay"]
                for row in self.generated_summary.values()
            )
        )

    def test_arbitrary_rational_target_has_transparent_exact_witness(self):
        bundle = verify_target_artifact_bundle(
            self.output / BUNDLE_NAMES[0]
        )
        expected = {
            (0, 0): Fraction(1, 2),
            (0, 1): Fraction(-2, 3),
            (1, 0): Fraction(5, 7),
            (1, 1): Fraction(11, 5),
        }
        self.assertEqual(
            {
                coloring: bundle.target.coefficient(coloring)
                for coloring in expected
            },
            expected,
        )
        self.assertIsNotNone(bundle.witness)
        self.assertTrue(
            bundle.tensor_map.compare_exact(
                bundle.witness, bundle.target
            ).satisfied
        )
        self.assertEqual(bundle.witness.support_size, 4)
        claims = bundle.certificate["claims"]
        self.assertTrue(
            claims[
                "exact_rational_affine_image_membership_certified"
            ]
        )
        self.assertTrue(
            claims["complex_affine_image_membership_certified"]
        )
        for key in (
            "projective_image_membership_certified",
            "local_diagonal_orbit_membership_certified",
            "border_image_membership_certified",
            "nonexistence_certificate_emitted",
        ):
            self.assertFalse(claims[key])

    def test_named_targets_are_structure_only(self):
        expected = {
            BUNDLE_NAMES[1]: heralded_ghz_target(
                4, 3, 2, trigger_color=0
            ),
            BUNDLE_NAMES[2]: unnormalized_w_target(4),
            BUNDLE_NAMES[3]: unnormalized_dicke_target(4, 2),
            BUNDLE_NAMES[4]: unnormalized_graph_state_target(
                4, ((0, 1),)
            ),
        }
        self.assertEqual(
            [target.support_size for target in expected.values()],
            [3, 4, 6, 16],
        )
        for name, target in expected.items():
            with self.subTest(name=name):
                directory = self.output / name
                bundle = verify_target_artifact_bundle(directory)
                self.assertEqual(bundle.target, target)
                self.assertIsNone(bundle.witness)
                self.assertFalse((directory / WITNESS_FILE).exists())
                self.assertEqual(
                    bundle.certificate["status"],
                    "target-problem-structure-only-no-witness",
                )
                claims = bundle.certificate["claims"]
                for key, value in claims.items():
                    if key.endswith("_certified") or key in {
                        "witness_emitted",
                        "nonexistence_certificate_emitted",
                    }:
                        self.assertFalse(value, key)

    def test_verify_all_replays_the_same_semantic_summary(self):
        self.assertEqual(verify_all(self.output), self.generated_summary)
        self.assertTrue(
            self.generated_summary[BUNDLE_NAMES[0]][
                "affine_image_certified"
            ]
        )
        self.assertTrue(
            all(
                not self.generated_summary[name][
                    "affine_image_certified"
                ]
                for name in BUNDLE_NAMES[1:]
            )
        )

    def test_collection_inventory_fails_closed(self):
        unexpected = self.output / "unexpected.txt"
        unexpected.write_text(
            "not part of the release\n", encoding="utf-8"
        )
        try:
            with self.assertRaisesRegex(
                KrennTargetExampleError,
                "root inventory changed",
            ):
                verify_all(self.output)
        finally:
            unexpected.unlink()


if __name__ == "__main__":
    unittest.main()
