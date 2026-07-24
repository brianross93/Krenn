import unittest
from fractions import Fraction

from experiments.krenn_quantum_graph.cyclotomic_shadow import (
    OMEGA,
    OMEGA_SQUARED,
    PRISM_EDGE_DATA,
    QOMEGA_ONE,
    QOMEGA_ZERO,
    VICTIM_COLORING,
    QOmega,
    certify_cyclotomic_shadow_witness,
    evaluate_cyclotomic_shadow,
    full_tensor_coloring_amplitude,
    prism_equal_g_shadow_values,
)
from experiments.krenn_quantum_graph.hafnian_identities import (
    generate_equal_g_shadow_system,
)


class KrennCyclotomicShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = generate_equal_g_shadow_system(6, 3)
        cls.values = prism_equal_g_shadow_values()
        cls.certificate = certify_cyclotomic_shadow_witness()

    def test_pair_arithmetic_reduces_mod_cyclotomic_relation(self):
        self.assertEqual(
            OMEGA_SQUARED + OMEGA + QOMEGA_ONE,
            QOMEGA_ZERO,
        )
        self.assertEqual(OMEGA**3, QOMEGA_ONE)
        self.assertEqual(
            QOmega(Fraction(2, 3), Fraction(-5, 7))
            * QOmega(Fraction(4, 5), Fraction(3, 2)),
            QOmega(Fraction(337, 210), Fraction(3, 2)),
        )
        value = QOmega(Fraction(7, 3), Fraction(-2, 5))
        self.assertEqual(value / value, QOMEGA_ONE)
        self.assertEqual(value.conjugate().conjugate(), value)

    def test_prism_witness_has_nine_edges_and_only_diagonal_slots(self):
        self.assertEqual(len(self.values), 90)
        self.assertEqual(len(PRISM_EDGE_DATA), 9)
        nonzero = [
            key
            for key, value in zip(
                self.system.variable_keys,
                self.values,
                strict=True,
            )
            if not value.is_zero
        ]
        self.assertEqual(len(nonzero), 27)
        self.assertTrue(all(a == b for _i, _j, a, b in nonzero))
        self.assertEqual(
            len({(i, j) for i, j, _a, _b in nonzero}),
            9,
        )

    def test_all_28_occupation_equations_match_ghz_exactly(self):
        replay = evaluate_cyclotomic_shadow(
            self.system, self.values
        )
        self.assertEqual(replay, self.certificate.shadow_coefficients)
        self.assertEqual(
            replay,
            self.certificate.target_coefficients,
        )
        self.assertEqual(len(replay), 28)
        self.assertTrue(
            self.certificate.all_28_shadow_equations_satisfied
        )
        nonzero = [
            (occupation, coefficient)
            for occupation, coefficient in zip(
                self.certificate.equation_occupations,
                replay,
                strict=True,
            )
            if not coefficient.is_zero
        ]
        self.assertEqual(
            nonzero,
            [
                ((0, 0, 6), QOMEGA_ONE),
                ((0, 6, 0), QOMEGA_ONE),
                ((6, 0, 0), QOMEGA_ONE),
            ],
        )

    def test_diagonal_lift_is_explicitly_shadow_only(self):
        self.assertEqual(
            self.certificate.all_equal_amplitudes,
            (QOMEGA_ONE, QOMEGA_ONE, QOMEGA_ONE),
        )
        expected = Fraction(2, 3) * OMEGA_SQUARED
        self.assertEqual(
            full_tensor_coloring_amplitude(VICTIM_COLORING),
            expected,
        )
        self.assertEqual(
            self.certificate.victim_amplitude,
            QOmega(Fraction(-2, 3), Fraction(-2, 3)),
        )
        self.assertTrue(
            self.certificate.full_tensor_victim_nonzero
        )
        self.assertFalse(
            self.certificate.full_tensor_solution_claimed
        )


if __name__ == "__main__":
    unittest.main()
