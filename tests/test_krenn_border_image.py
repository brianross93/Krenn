from fractions import Fraction
import json
import unittest

from experiments.krenn_quantum_graph.border_image import (
    BORDER_IMAGE_SCHEMA,
    KrennBorderImageError,
    LAURENT_ONE,
    LAURENT_T,
    LAURENT_T_INVERSE,
    LAURENT_ZERO,
    LaurentPolynomial,
    certify_n6_d3_laurent_border,
    evaluate_laurent_system,
    natural_laurent_weight_entries,
)
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    generate_sparse_system,
    variable_index,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
)


class KrennBorderImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = certify_n6_d3_laurent_border()

    def test_laurent_arithmetic_is_canonical_and_exact(self):
        polynomial = (
            LaurentPolynomial.monomial(-1, Fraction(2, 3))
            * LaurentPolynomial.monomial(2, Fraction(9, 4))
            + LaurentPolynomial.constant(5)
            - LaurentPolynomial.constant(2)
        )
        self.assertEqual(
            polynomial,
            LaurentPolynomial(
                ((0, 3), (1, Fraction(3, 2)))
            ),
        )
        self.assertEqual(polynomial.to_expression(), "3 + 3/2*t")
        self.assertTrue(polynomial.is_polynomial)
        self.assertEqual(polynomial.specialize_zero(), 3)
        with self.assertRaisesRegex(
            KrennBorderImageError, "pole"
        ):
            LAURENT_T_INVERSE.specialize_zero()

    def test_weight_family_has_the_declared_pole_and_eight_regular_slots(self):
        entries = dict(natural_laurent_weight_entries())
        self.assertEqual(len(entries), 9)
        self.assertEqual(
            entries[variable_index(6, 3, 0, 1, 0, 0)],
            LAURENT_T,
        )
        self.assertEqual(
            entries[variable_index(6, 3, 2, 3, 0, 0)],
            LAURENT_T_INVERSE,
        )
        self.assertEqual(
            sum(value == LAURENT_ONE for value in entries.values()),
            7,
        )
        self.assertEqual(
            [
                index
                for index, value in entries.items()
                if value.has_pole_at_zero
            ],
            [variable_index(6, 3, 2, 3, 0, 0)],
        )

    def test_all_729_outputs_replay_the_exact_border_identity(self):
        certificate = self.certificate
        self.assertEqual(certificate.schema, BORDER_IMAGE_SCHEMA)
        self.assertEqual(len(certificate.output_coefficients), 729)
        self.assertEqual(
            certificate.output_coefficients,
            certificate.expected_coefficients,
        )
        target = certificate.target_coefficients
        for equation, output in enumerate(
            certificate.output_coefficients
        ):
            expected = LaurentPolynomial.constant(target[equation])
            if equation == N6_D3_SEED_DEFECT_EQUATION:
                expected += LAURENT_T
            with self.subTest(equation=equation):
                self.assertEqual(output, expected)
        self.assertEqual(
            coloring_from_index(
                6, 3, N6_D3_SEED_DEFECT_EQUATION
            ),
            (0, 0, 2, 1, 2, 1),
        )

    def test_symbolic_output_specialization_is_exactly_ghz(self):
        certificate = self.certificate
        self.assertTrue(
            all(
                coefficient.is_polynomial
                for coefficient in certificate.output_coefficients
            )
        )
        self.assertEqual(
            certificate.specialized_coefficients_at_zero,
            certificate.target_coefficients,
        )
        self.assertTrue(certificate.exact)
        self.assertTrue(certificate.border_image_membership_proved)

    def test_claim_boundary_does_not_assert_a_finite_exact_witness(self):
        certificate = self.certificate
        self.assertFalse(certificate.finite_weight_witness_at_zero)
        self.assertFalse(
            certificate.exact_affine_image_membership_proved
        )
        payload = certificate.to_dict()
        boundary = payload["claim_boundary"]
        self.assertTrue(boundary["Q_defined_Laurent_degeneration"])
        self.assertTrue(boundary["GHZ_in_Zariski_closure_over_Q"])
        self.assertTrue(boundary["GHZ_in_Zariski_closure_over_C"])
        self.assertTrue(boundary["border_image_membership_proved"])
        self.assertFalse(boundary["finite_weight_witness_at_t_zero"])
        self.assertFalse(
            boundary["exact_affine_image_membership_proved"]
        )
        self.assertEqual(
            boundary["exact_affine_image_membership_status"],
            "undecided",
        )
        self.assertTrue(
            payload["symbolic_specialization"][
                "input_has_pole_at_t_zero"
            ]
        )
        self.assertTrue(
            payload["symbolic_specialization"][
                "output_at_t_zero_equals_GHZ"
            ]
        )
        json.dumps(payload, allow_nan=False)

    def test_compact_payload_records_only_three_constants_and_one_defect(self):
        nonzero = self.certificate.to_dict()["full_tensor_replay"][
            "nonzero_coefficients"
        ]
        self.assertEqual(
            [
                (row["coloring"], row["coefficient"])
                for row in nonzero
            ],
            [
                ([0, 0, 0, 0, 0, 0], "1"),
                ([0, 0, 2, 1, 2, 1], "t"),
                ([1, 1, 1, 1, 1, 1], "1"),
                ([2, 2, 2, 2, 2, 2], "1"),
            ],
        )

    def test_laurent_replay_rejects_duplicate_or_out_of_range_slots(self):
        system = generate_sparse_system(6, 3)
        index = variable_index(6, 3, 0, 1, 0, 0)
        with self.assertRaisesRegex(
            KrennBorderImageError, "duplicate"
        ):
            evaluate_laurent_system(
                system, ((index, LAURENT_ONE), (index, LAURENT_T))
            )
        with self.assertRaisesRegex(
            KrennBorderImageError, "duplicate"
        ):
            evaluate_laurent_system(
                system, ((index, LAURENT_ZERO), (index, LAURENT_T))
            )
        with self.assertRaisesRegex(
            KrennBorderImageError, "outside"
        ):
            evaluate_laurent_system(
                system, ((system.variable_count, LAURENT_ONE),)
            )


if __name__ == "__main__":
    unittest.main()
