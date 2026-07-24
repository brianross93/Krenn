from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest

from experiments.krenn_quantum_graph.fixtures import fixture_n4_d3
from experiments.krenn_quantum_graph.local_actions import (
    ExactLocalAction,
    KrennLocalActionError,
    LocalEquivarianceCertificate,
    apply_local_action_to_target,
    apply_local_action_to_witness,
    certify_local_equivariance,
    compose_local_actions,
)
from experiments.krenn_quantum_graph.system import variable_index
from experiments.krenn_quantum_graph.targets import (
    ColoringTarget,
    canonical_ghz_target,
)
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.witness import SparseWitness


class KrennLocalActionTest(unittest.TestCase):
    def test_identity_and_composition_are_exact(self):
        witness = SparseWitness.from_coordinates(
            4,
            3,
            {
                (0, 1, 0, 2): Fraction(2, 3),
                (0, 2, 1, 0): -2,
                (1, 3, 2, 1): Fraction(5, 7),
                (2, 3, 1, 2): 3,
            },
        )
        target = MatchingTensorMap(4, 3).evaluate_exact(witness)
        identity = ExactLocalAction.identity(4, 3)
        self.assertTrue(identity.local_gl)
        self.assertFalse(identity.rectangular)
        self.assertEqual(
            apply_local_action_to_witness(witness, identity), witness
        )
        self.assertEqual(
            apply_local_action_to_target(target, identity), target
        )

        before = ExactLocalAction.from_matrices(
            (
                ((1, 0, 1), (0, 1, 0)),
                ((1, 1, 0), (0, 1, 1)),
                ((2, 0, 0), (0, 1, -1)),
                ((1, 0, 0), (1, 1, 1)),
            )
        )
        after = ExactLocalAction.from_matrices(
            (
                ((1, 2), (0, 1)),
                ((1, 0), (-1, 1)),
                ((2, 1), (1, 1)),
                ((1, -1), (1, 0)),
            )
        )
        composed = compose_local_actions(after, before)
        self.assertEqual(
            compose_local_actions(
                ExactLocalAction.identity(4, 2), before
            ),
            before,
        )
        self.assertEqual(
            compose_local_actions(
                before, ExactLocalAction.identity(4, 3)
            ),
            before,
        )
        self.assertEqual(
            apply_local_action_to_witness(
                apply_local_action_to_witness(witness, before),
                after,
            ),
            apply_local_action_to_witness(witness, composed),
        )
        self.assertEqual(
            apply_local_action_to_target(
                apply_local_action_to_target(target, before),
                after,
            ),
            apply_local_action_to_target(target, composed),
        )
        self.assertTrue(
            certify_local_equivariance(
                witness, target, composed
            ).exact
        )

    def test_rectangular_projection_sends_ghz3_fixture_to_ghz2(self):
        source_witness = fixture_n4_d3()
        source_target = canonical_ghz_target(4, 3)
        projection_matrix = ((1, 0, 0), (0, 1, 0))
        projection = ExactLocalAction.from_matrices(
            (projection_matrix,) * 4
        )
        certificate = certify_local_equivariance(
            source_witness, source_target, projection
        )

        self.assertTrue(projection.rectangular)
        self.assertFalse(projection.local_gl)
        self.assertTrue(certificate.exact)
        self.assertTrue(certificate.map_equivariant)
        self.assertTrue(certificate.residual_equivariant)
        self.assertTrue(certificate.source_satisfied)
        self.assertTrue(certificate.transformed_satisfied)
        self.assertEqual(
            certificate.transformed_target,
            canonical_ghz_target(4, 2),
        )
        self.assertEqual(certificate.transformed_witness.support_size, 4)
        self.assertEqual(
            MatchingTensorMap(4, 2).evaluate_exact(
                certificate.transformed_witness
            ),
            canonical_ghz_target(4, 2),
        )
        self.assertFalse(
            certificate.individual_target_nonimage_proved
        )
        self.assertFalse(certificate.no_go_propagation_proved)
        self.assertIn("does not by itself", certificate.claim_boundary)
        self.assertIn(
            "separate exact nonimage certificate",
            certificate.claim_boundary,
        )

    def test_asymmetric_local_gl_preserves_endpoint_color_slots(self):
        projection_matrix = ((1, 0, 0), (0, 1, 0))
        source_witness = apply_local_action_to_witness(
            fixture_n4_d3(),
            ExactLocalAction.from_matrices(
                (projection_matrix,) * 4
            ),
        )
        source_target = canonical_ghz_target(4, 2)
        action = ExactLocalAction.from_matrices(
            (
                ((1, 2), (0, 1)),
                ((1, 0), (3, 1)),
                ((1, -1), (1, 1)),
                ((2, 0), (1, 1)),
            )
        )
        transformed = apply_local_action_to_witness(
            source_witness, action
        )

        self.assertTrue(action.local_gl)
        # Edge (0,2) originally has only W[0,2,1,1]=1.  Its transformed
        # block is column_1(A_0) column_1(A_2)^T, so the first and second
        # endpoint slots genuinely use different local matrices.
        expected_block = ((-2, 2), (-1, 1))
        for alpha in range(2):
            for beta in range(2):
                self.assertEqual(
                    transformed.value(
                        variable_index(
                            4, 2, 0, 2, alpha, beta
                        )
                    ),
                    expected_block[alpha][beta],
                )

        certificate = certify_local_equivariance(
            source_witness, source_target, action
        )
        self.assertTrue(certificate.exact)
        self.assertTrue(certificate.source_satisfied)
        self.assertTrue(certificate.transformed_satisfied)
        self.assertEqual(
            certificate.transformed_output,
            certificate.expected_transformed_output,
        )

    def test_unsatisfied_residual_is_transported_exactly(self):
        witness = SparseWitness.from_coordinates(
            4,
            2,
            {
                (0, 1, 0, 1): 2,
                (0, 2, 1, 0): -1,
                (1, 3, 1, 1): 3,
                (2, 3, 0, 1): Fraction(1, 2),
            },
        )
        target = ColoringTarget.from_sparse(
            4,
            2,
            {
                (0, 0, 0, 0): 1,
                (0, 1, 1, 0): Fraction(2, 5),
                (1, 0, 1, 1): -3,
            },
        )
        action = ExactLocalAction.from_matrices(
            (
                ((1, 1),),
                ((2, -1),),
                ((0, 1),),
                ((1, 2),),
            )
        )
        certificate = certify_local_equivariance(
            witness, target, action
        )
        self.assertTrue(certificate.exact)
        self.assertFalse(certificate.source_satisfied)
        self.assertFalse(
            certificate.forward_solution_transport_certified
        )
        self.assertEqual(
            certificate.transformed_residual,
            apply_local_action_to_target(
                certificate.source_residual, action
            ),
        )

    def test_validation_corruption_and_dependencies_fail_closed(self):
        with self.assertRaises(KrennLocalActionError):
            ExactLocalAction.from_matrices(())
        with self.assertRaisesRegex(
            KrennLocalActionError, "even and at least 2"
        ):
            ExactLocalAction.from_matrices(
                (((1,),), ((1,),), ((1,),))
            )
        with self.assertRaisesRegex(
            KrennLocalActionError, "input_dimension entries"
        ):
            ExactLocalAction(
                n=4,
                input_dimension=2,
                output_dimension=1,
                matrices=(
                    ((1, 0),),
                    ((1, 0),),
                    ((1,),),
                    ((1, 0),),
                ),
            )
        with self.assertRaisesRegex(
            KrennLocalActionError, "integer or Fraction"
        ):
            ExactLocalAction.from_matrices(
                (
                    ((1.0,),),
                    ((1,),),
                    ((1,),),
                    ((1,),),
                )
            )
        with self.assertRaisesRegex(
            KrennLocalActionError, "one local matrix per vertex"
        ):
            ExactLocalAction(
                n=4,
                input_dimension=1,
                output_dimension=1,
                matrices=(((1,),),) * 2,
            )

        witness = fixture_n4_d3()
        action_d2 = ExactLocalAction.identity(4, 2)
        with self.assertRaisesRegex(
            KrennLocalActionError, "witness dimensions"
        ):
            apply_local_action_to_witness(witness, action_d2)
        with self.assertRaisesRegex(
            KrennLocalActionError, "target dimensions"
        ):
            apply_local_action_to_target(
                canonical_ghz_target(4, 3), action_d2
            )
        with self.assertRaisesRegex(
            KrennLocalActionError, "dimensions do not meet"
        ):
            compose_local_actions(
                ExactLocalAction.identity(4, 3),
                ExactLocalAction.identity(4, 2),
            )
        with self.assertRaisesRegex(
            KrennLocalActionError, "source dimensions"
        ):
            certify_local_equivariance(
                witness,
                canonical_ghz_target(4, 2),
                ExactLocalAction.identity(4, 3),
            )

        valid_certificate = certify_local_equivariance(
            witness,
            canonical_ghz_target(4, 3),
            ExactLocalAction.identity(4, 3),
        )
        with self.assertRaisesRegex(
            KrennLocalActionError, "schema changed"
        ):
            replace(valid_certificate, schema="corrupt")
        with self.assertRaisesRegex(
            KrennLocalActionError, "schema changed"
        ):
            LocalEquivarianceCertificate(
                action=ExactLocalAction.identity(4, 3),
                source_witness=witness,
                source_target=canonical_ghz_target(4, 3),
                schema="corrupt",
            )

        import experiments.krenn_quantum_graph.local_actions as subject

        source = Path(subject.__file__).read_text(encoding="utf-8")
        self.assertNotIn("section12_instantiation", source)
        self.assertNotIn("artifacts", source)
        self.assertNotIn("results", source)


if __name__ == "__main__":
    unittest.main()
