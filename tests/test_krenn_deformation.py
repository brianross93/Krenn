from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.deformation import (
    DEFORMATION_SCHEMA,
    build_rational_quotient_frame,
    certify_n4_d3_deformation,
    gauge_action_preserves_all_monomials,
    reciprocal_gauge_matrix,
    rescale_complementary_edges,
)
from experiments.krenn_quantum_graph.fixtures import fixture_n4_d3
from experiments.krenn_quantum_graph.system import generate_sparse_system
from experiments.krenn_quantum_graph.witness import (
    evaluate_exact,
    evaluate_mod,
)


class KrennDeformationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = generate_sparse_system(4, 3)
        cls.witness = fixture_n4_d3()
        cls.certificate = certify_n4_d3_deformation(
            cls.system, cls.witness
        )

    def test_raw_jacobian_has_sharp_rank_and_nullity(self):
        jacobian = self.certificate.payload["jacobian"]
        self.assertEqual(jacobian["shape"], [81, 54])
        self.assertEqual(jacobian["rank_over_Q"], 51)
        self.assertEqual(jacobian["nullity_over_Q"], 3)
        self.assertEqual(
            self.certificate.payload["schema"], DEFORMATION_SCHEMA
        )

    def test_three_raw_directions_are_exactly_reciprocal_gauge(self):
        self.assertTrue(
            gauge_action_preserves_all_monomials(self.system)
        )
        gauge = reciprocal_gauge_matrix(self.witness)
        nonzero_columns = {
            column: [
                (row, gauge[row][column])
                for row in range(54)
                if gauge[row][column]
            ]
            for column in range(3)
        }
        self.assertEqual(
            [len(nonzero_columns[column]) for column in range(3)],
            [2, 2, 2],
        )
        checks = self.certificate.payload["exact_checks"]
        self.assertTrue(checks["raw_kernel_equals_gauge_over_Q"])
        self.assertTrue(
            checks[
                "gauge_action_preserves_every_system_monomial"
            ]
        )

    def test_native_exact_q_quotient_removes_exactly_three(self):
        gauge = reciprocal_gauge_matrix(self.witness)
        frame = build_rational_quotient_frame(gauge)
        self.assertEqual(frame.ambient_dimension, 54)
        self.assertEqual(frame.quotient_dimension, 51)
        quotient = self.certificate.payload[
            "native_gauge_quotient_over_Q"
        ]
        self.assertEqual(quotient["ambient_dimension"], 54)
        self.assertEqual(quotient["gauge_image_dimension"], 3)
        self.assertEqual(quotient["quotient_dimension"], 51)
        self.assertEqual(quotient["quotient_map_shape"], [51, 54])
        self.assertEqual(quotient["section_shape"], [54, 51])
        self.assertEqual(quotient["induced_jacobian_rank"], 51)
        self.assertEqual(quotient["induced_jacobian_nullity"], 0)
        checks = self.certificate.payload["exact_checks"]
        for name in (
            "native_quotient_map_kills_gauge_over_Q",
            "native_quotient_section_identity_over_Q",
            "native_quotient_kernel_equals_gauge_over_Q",
            "native_quotient_jacobian_factorization_over_Q",
            "native_quotient_jacobian_injective_over_Q",
        ):
            self.assertTrue(checks[name])
        self.assertTrue(self.certificate.exact)

    def test_f31_is_labeled_as_optional_arithmetic_regression(self):
        regression = self.certificate.payload[
            "optional_arithmetic_regression_F31"
        ]
        self.assertIn("not used", regression["proof_role"])
        self.assertEqual(regression["jacobian_rank"], 51)
        self.assertEqual(regression["jacobian_nullity"], 3)
        self.assertEqual(regression["gauge_rank"], 3)
        self.assertTrue(regression["passed"])

    def test_reciprocal_action_gives_nonintegral_exact_witnesses(self):
        transformed = rescale_complementary_edges(
            self.witness,
            (Fraction(7, 3), Fraction(5, 2), Fraction(11, 4)),
        )
        self.assertFalse(transformed.integral)
        self.assertTrue(evaluate_exact(self.system, transformed).satisfied)
        self.assertTrue(evaluate_mod(self.system, transformed).satisfied)


if __name__ == "__main__":
    unittest.main()
