from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.hafnian_identities import (
    KrennHafnianIdentityError,
    certify_equal_g_identity,
    certify_hafnian_contraction,
    certify_shadow_jacobian_rank,
    generate_equal_g_shadow_system,
    shadow_structure_summary,
    singular_groebner_script,
)
from experiments.krenn_quantum_graph.targets import canonical_ghz_target
from experiments.krenn_quantum_graph.ternary_search import (
    n6_d3_seed_witness,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


class KrennHafnianIdentityTest(unittest.TestCase):
    def test_distinct_local_covectors_replay_full_asymmetric_edges(self):
        witness = SparseWitness.from_coordinates(
            4,
            3,
            {
                (0, 1, 0, 1): Fraction(2, 3),
                (0, 1, 1, 0): Fraction(-5, 7),
                (2, 3, 2, 1): Fraction(11, 5),
                (0, 2, 0, 2): 3,
                (1, 3, 1, 0): -2,
            },
        )
        covectors = (
            (1, 2, -1),
            (3, -2, 4),
            (Fraction(1, 2), 0, 5),
            (-1, Fraction(3, 5), 2),
        )
        certificate = certify_hafnian_contraction(
            witness, covectors
        )
        self.assertTrue(certificate.map_identity_exact)
        self.assertEqual(
            certificate.hafnian_value,
            certificate.tensor_map_value,
        )
        self.assertIsNone(certificate.target_value)

    def test_equal_g_forgets_exactly_the_antisymmetric_edge_part(self):
        left = SparseWitness.from_coordinates(
            2,
            2,
            {
                (0, 1, 0, 1): 2,
                (0, 1, 1, 0): 5,
            },
        )
        right = SparseWitness.from_coordinates(
            2,
            2,
            {
                (0, 1, 0, 1): 3,
                (0, 1, 1, 0): 4,
            },
        )
        left_equal = certify_equal_g_identity(left)
        right_equal = certify_equal_g_identity(right)
        self.assertTrue(left_equal.map_identity_exact)
        self.assertTrue(right_equal.map_identity_exact)
        self.assertEqual(
            left_equal.shadow_coefficients,
            right_equal.shadow_coefficients,
        )
        left_distinct = certify_hafnian_contraction(
            left, ((1, 0), (0, 1))
        )
        right_distinct = certify_hafnian_contraction(
            right, ((1, 0), (0, 1))
        )
        self.assertEqual(left_distinct.hafnian_value, 2)
        self.assertEqual(right_distinct.hafnian_value, 3)

    def test_natural_seed_total_sum_and_balanced_shadow_defect(self):
        witness = n6_d3_seed_witness()
        target = canonical_ghz_target(6, 3)
        total = certify_hafnian_contraction(
            witness,
            ((1, 1, 1),) * 6,
            target=target,
        )
        self.assertEqual(total.hafnian_value, 4)
        self.assertEqual(total.tensor_map_value, 4)
        self.assertEqual(total.target_value, 3)
        self.assertEqual(total.target_residual, 1)

        equal_g = certify_equal_g_identity(
            witness, target=target
        )
        system = generate_equal_g_shadow_system(6, 3)
        nonzero = [
            (occupation, residual)
            for occupation, residual in zip(
                system.equation_occupations,
                equal_g.residual_coefficients,
            )
            if residual
        ]
        self.assertEqual(nonzero, [((2, 2, 2), Fraction(1))])

    def test_equal_g_shadow_counts_and_dominance_certificates(self):
        summary_3 = shadow_structure_summary(6, 3)
        summary_4 = shadow_structure_summary(6, 4)
        self.assertEqual(
            summary_3["counts"],
            {
                "full_edge_variables": 135,
                "symmetric_shadow_variables": 90,
                "antisymmetric_invisible_variables": 45,
                "occupation_equations": 28,
                "degree": 3,
                "expanded_monomials": 3240,
            },
        )
        self.assertEqual(
            summary_4["counts"],
            {
                "full_edge_variables": 240,
                "symmetric_shadow_variables": 150,
                "antisymmetric_invisible_variables": 90,
                "occupation_equations": 84,
                "degree": 3,
                "expanded_monomials": 15000,
            },
        )
        self.assertEqual(
            summary_3["jacobian_probe"]["rank"], 28
        )
        self.assertEqual(
            summary_4["jacobian_probe"]["rank"], 84
        )
        self.assertNotEqual(
            summary_3["jacobian_probe"][
                "pivot_minor_determinant_mod"
            ],
            0,
        )
        self.assertNotEqual(
            summary_4["jacobian_probe"][
                "pivot_minor_determinant_mod"
            ],
            0,
        )
        self.assertTrue(
            summary_3["claim_boundary"][
                "dominance_excludes_nonzero_universal_"
                "polynomial_invariants_on_shadow_closure"
            ]
        )
        self.assertFalse(
            summary_4["claim_boundary"][
                "jacobian_dominance_alone_proves_exact_GHZ_fiber_nonempty"
            ]
        )

    def test_modular_groebner_export_is_deterministic_but_not_a_claim(self):
        system = generate_equal_g_shadow_system(6, 3)
        target = canonical_ghz_target(6, 3)
        first = singular_groebner_script(system, target)
        second = singular_groebner_script(system, target)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("// Deterministic"))
        self.assertIn("ring r=31,(x0,x1", first)
        self.assertIn("ideal G=std(I);", first)
        self.assertIn("unit_ideal", first)

    def test_validation_fails_closed(self):
        witness = SparseWitness.from_coordinates(
            2, 2, {(0, 1, 0, 0): 1}
        )
        with self.assertRaisesRegex(
            KrennHafnianIdentityError, "one local covector"
        ):
            certify_hafnian_contraction(witness, ((1, 0),))
        with self.assertRaisesRegex(
            KrennHafnianIdentityError, "length 2"
        ):
            certify_hafnian_contraction(
                witness, ((1,), (1, 0))
            )
        with self.assertRaisesRegex(
            KrennHafnianIdentityError, "dimensions"
        ):
            certify_equal_g_identity(
                witness, target=canonical_ghz_target(4, 2)
            )
        shadow = generate_equal_g_shadow_system(6, 3)
        with self.assertRaisesRegex(
            KrennHafnianIdentityError, "prime"
        ):
            singular_groebner_script(
                shadow, canonical_ghz_target(6, 3), modulus=15
            )
        with self.assertRaisesRegex(
            KrennHafnianIdentityError, "prime"
        ):
            certify_shadow_jacobian_rank(6, 3, modulus=15)


if __name__ == "__main__":
    unittest.main()
