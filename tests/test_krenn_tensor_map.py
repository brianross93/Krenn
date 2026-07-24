from fractions import Fraction
from pathlib import Path
import unittest

from experiments.krenn_quantum_graph.targets import ColoringTarget
from experiments.krenn_quantum_graph.tensor_map import (
    CLAIM_MODE_RECORDS,
    ClaimMode,
    KrennTensorMapError,
    MatchingTensorMap,
    certify_map_equivariance,
    claim_mode_record,
    transport_target,
)
from experiments.krenn_quantum_graph.transport import transport_witness
from experiments.krenn_quantum_graph.witness import SparseWitness


class KrennMatchingTensorMapTest(unittest.TestCase):
    def test_n2_tensor_map_is_exactly_the_edge_tensor(self):
        tensor_map = MatchingTensorMap(2, 3)
        witness = SparseWitness.from_coordinates(
            2,
            3,
            {
                (0, 1, 0, 0): Fraction(2, 3),
                (0, 1, 0, 2): -4,
                (0, 1, 1, 0): 7,
                (0, 1, 2, 1): Fraction(-5, 2),
            },
        )
        output = tensor_map.evaluate_exact(witness)

        self.assertEqual(tensor_map.degree, 1)
        self.assertEqual(tensor_map.matching_count, 1)
        self.assertEqual(tensor_map.coefficient_count, 9)
        self.assertEqual(tensor_map.monomial_count, 9)
        self.assertEqual(tensor_map.equation_offsets, tuple(range(10)))
        self.assertEqual(output.coefficient((0, 0)), Fraction(2, 3))
        self.assertEqual(output.coefficient((0, 2)), -4)
        self.assertEqual(output.coefficient((1, 0)), 7)
        self.assertEqual(output.coefficient((2, 1)), Fraction(-5, 2))
        self.assertEqual(output.coefficient((2, 2)), 0)
        for coloring, value in output.entries:
            self.assertEqual(
                tensor_map.coefficient_exact(witness, coloring),
                Fraction(value),
            )

    def test_canonical_coefficient_formula_has_three_n4_matchings(self):
        tensor_map = MatchingTensorMap(4, 2)
        witness = SparseWitness.from_coordinates(
            4,
            2,
            {
                (0, 1, 0, 1): 2,
                (2, 3, 1, 0): 3,
                (0, 2, 0, 1): 5,
                (1, 3, 1, 0): 7,
                (0, 3, 0, 0): 11,
                (1, 2, 1, 1): 13,
            },
        )
        coloring = (0, 1, 1, 0)
        self.assertEqual(tensor_map.matching_count, 3)
        self.assertEqual(tensor_map.degree, 2)
        self.assertEqual(
            tensor_map.coefficient_exact(witness, coloring),
            2 * 3 + 5 * 7 + 11 * 13,
        )

    def test_exact_residual_and_real_rational_fidelity(self):
        tensor_map = MatchingTensorMap(2, 2)
        witness = SparseWitness.from_coordinates(
            2,
            2,
            {
                (0, 1, 0, 0): 1,
                (0, 1, 1, 1): 2,
            },
        )
        proportional = ColoringTarget.from_sparse(
            2,
            2,
            {
                (0, 0): Fraction(3, 2),
                (1, 1): 3,
            },
        )
        comparison = tensor_map.compare_exact(witness, proportional)
        self.assertFalse(comparison.satisfied)
        self.assertEqual(comparison.nonzero_residual_count, 2)
        self.assertEqual(
            comparison.residual,
            tensor_map.exact_residual(witness, proportional),
        )
        self.assertEqual(
            comparison.residual_norm_squared,
            Fraction(5, 4),
        )

        fidelity = tensor_map.exact_fidelity(witness, proportional)
        self.assertEqual(fidelity.value, 1)
        self.assertEqual(fidelity.inner_product, Fraction(15, 2))
        self.assertEqual(fidelity.output_norm_squared, 5)
        self.assertEqual(
            fidelity.target_norm_squared, Fraction(45, 4)
        )

        orthogonal = ColoringTarget.from_sparse(
            2, 2, {(0, 1): 1}
        )
        self.assertEqual(
            tensor_map.exact_fidelity(witness, orthogonal).value, 0
        )

    def test_fidelity_rejects_zero_output_and_zero_target(self):
        tensor_map = MatchingTensorMap(2, 2)
        zero_witness = SparseWitness.from_index_values(2, 2, ())
        nonzero_target = ColoringTarget.from_sparse(
            2, 2, {(0, 0): 1}
        )
        with self.assertRaisesRegex(
            KrennTensorMapError, "zero tensor-map output"
        ):
            tensor_map.exact_fidelity(zero_witness, nonzero_target)

        nonzero_witness = SparseWitness.from_coordinates(
            2, 2, {(0, 1, 0, 0): 1}
        )
        zero_target = ColoringTarget.from_sparse(2, 2, ())
        with self.assertRaisesRegex(
            KrennTensorMapError, "zero target"
        ):
            tensor_map.exact_fidelity(nonzero_witness, zero_target)

    def test_asymmetric_target_and_residual_transport_equivariance(self):
        tensor_map = MatchingTensorMap(4, 2)
        witness = SparseWitness.from_coordinates(
            4,
            2,
            {
                (0, 1, 0, 1): 2,
                (2, 3, 1, 0): 3,
                (0, 2, 1, 1): -5,
                (1, 3, 0, 1): 7,
                (0, 3, 1, 0): 11,
                (1, 2, 1, 1): Fraction(2, 3),
            },
        )
        target = ColoringTarget.from_sparse(
            4,
            2,
            {
                (0, 0, 0, 1): Fraction(2, 5),
                (1, 0, 1, 0): -9,
                (1, 1, 0, 0): Fraction(7, 3),
            },
        )
        vertex_permutation = (3, 1, 2, 0)
        color_permutation = (1, 0)
        transported_target = transport_target(
            target, vertex_permutation, color_permutation
        )
        self.assertNotEqual(transported_target, target)
        self.assertEqual(
            transported_target.coefficient((0, 1, 1, 1)),
            Fraction(2, 5),
        )

        certificate = certify_map_equivariance(
            tensor_map,
            witness,
            target,
            vertex_permutation,
            color_permutation,
        )
        self.assertTrue(certificate.exact)
        self.assertTrue(certificate.variable_bijection)
        self.assertTrue(certificate.coefficient_bijection)
        self.assertTrue(certificate.map_equivariant)
        self.assertTrue(certificate.residual_equivariant)

        transported_witness = transport_witness(
            witness, vertex_permutation, color_permutation
        )
        self.assertEqual(
            tensor_map.evaluate_exact(transported_witness),
            transport_target(
                tensor_map.evaluate_exact(witness),
                vertex_permutation,
                color_permutation,
            ),
        )

    def test_claim_modes_keep_border_and_fidelity_fail_closed(self):
        self.assertEqual(set(CLAIM_MODE_RECORDS), set(ClaimMode))
        affine = claim_mode_record(ClaimMode.AFFINE)
        border = claim_mode_record("border")
        fidelity = claim_mode_record(ClaimMode.FIDELITY)
        projective = claim_mode_record(ClaimMode.PROJECTIVE)
        local = claim_mode_record(ClaimMode.LOCAL_DIAGONAL)

        self.assertTrue(affine.implemented_here)
        self.assertTrue(affine.proves_affine_image_membership)
        self.assertFalse(projective.implemented_here)
        self.assertFalse(local.implemented_here)
        self.assertFalse(border.implemented_here)
        self.assertFalse(border.proves_border_image_membership)
        self.assertTrue(fidelity.implemented_here)
        self.assertTrue(fidelity.metric_only)
        self.assertFalse(fidelity.proves_affine_image_membership)
        self.assertIn(
            "not a border-image certificate",
            fidelity.claim_boundary,
        )
        with self.assertRaises(KrennTensorMapError):
            claim_mode_record("numerical-proof")

    def test_tensor_layer_has_no_artifact_or_section12_dependency(self):
        import experiments.krenn_quantum_graph.tensor_map as subject

        source = Path(subject.__file__).read_text(encoding="utf-8")
        self.assertNotIn("section12_instantiation", source)
        self.assertNotIn("artifacts", source)
        self.assertNotIn("generate_sparse_system", source)


if __name__ == "__main__":
    unittest.main()
