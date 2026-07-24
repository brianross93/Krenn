from fractions import Fraction
import unittest
from unittest import mock

from experiments.krenn_quantum_graph import independent_verifier
from experiments.krenn_quantum_graph import system as primary
from experiments.krenn_quantum_graph.artifacts import system_arrays
from experiments.krenn_quantum_graph.deformation import (
    rescale_complementary_edges,
)
from experiments.krenn_quantum_graph.fixtures import (
    fixture_n4_d3,
    fixture_n6_d2,
)
from experiments.krenn_quantum_graph.witness import (
    KrennWitnessError,
    evaluate_exact,
    evaluate_mod,
    fraction_mod,
)


class KrennSparseSystemTest(unittest.TestCase):
    def test_deterministic_perfect_matching_census_and_order(self):
        self.assertEqual(
            primary.perfect_matchings(4),
            (
                ((0, 1), (2, 3)),
                ((0, 2), (1, 3)),
                ((0, 3), (1, 2)),
            ),
        )
        matchings = primary.perfect_matchings(6)
        self.assertEqual(len(matchings), 15)
        self.assertEqual(matchings, tuple(sorted(matchings)))
        self.assertEqual(matchings, primary.perfect_matchings(6))
        with self.assertRaises(primary.KrennSystemError):
            primary.perfect_matchings(5)

    def test_variable_index_is_canonical_and_round_trips(self):
        indices = set()
        for i in range(4):
            for j in range(i + 1, 4):
                for a in range(3):
                    for b in range(3):
                        index = primary.variable_index(
                            4, 3, i, j, a, b
                        )
                        indices.add(index)
                        self.assertEqual(
                            primary.variable_key(4, 3, index),
                            (i, j, a, b),
                        )
                        self.assertEqual(
                            primary.variable_index(
                                4, 3, j, i, b, a
                            ),
                            index,
                        )
        self.assertEqual(indices, set(range(54)))

    def test_n4_d3_fixture_exact_over_q_and_f31(self):
        system = primary.generate_sparse_system(4, 3)
        witness = fixture_n4_d3()
        self.assertEqual(
            (
                system.variable_count,
                system.equation_count,
                system.degree,
                system.matching_count,
                system.monomial_count,
            ),
            (54, 81, 2, 3, 243),
        )
        self.assertTrue(evaluate_exact(system, witness).satisfied)
        self.assertTrue(evaluate_mod(system, witness, 31).satisfied)

        rational = rescale_complementary_edges(
            witness,
            (Fraction(2), Fraction(3, 2), Fraction(5, 3)),
        )
        self.assertFalse(rational.integral)
        self.assertTrue(evaluate_exact(system, rational).satisfied)
        self.assertTrue(evaluate_mod(system, rational, 31).satisfied)

    def test_n6_d2_fixture_exact_and_structural_counts(self):
        system = primary.generate_sparse_system(6, 2)
        witness = fixture_n6_d2()
        self.assertEqual(
            (
                system.variable_count,
                system.equation_count,
                system.degree,
                system.matching_count,
                system.monomial_count,
            ),
            (60, 64, 3, 15, 960),
        )
        self.assertTrue(evaluate_exact(system, witness).satisfied)
        self.assertTrue(evaluate_mod(system, witness).satisfied)

    def test_negative_and_production_structural_targets(self):
        negative = primary.generate_sparse_system(4, 4)
        self.assertEqual(
            (
                negative.variable_count,
                negative.equation_count,
                negative.degree,
                negative.matching_count,
                negative.monomial_count,
            ),
            (96, 256, 2, 3, 768),
        )
        production = primary.generate_sparse_system(6, 4)
        self.assertEqual(
            (
                production.variable_count,
                production.equation_count,
                production.degree,
                production.matching_count,
                production.monomial_count,
            ),
            (240, 4096, 3, 15, 61440),
        )
        arrays = system_arrays(production)
        self.assertEqual(arrays["equation_offsets"].shape, (4097,))
        self.assertEqual(
            arrays["monomial_variable_indices"].shape, (61440, 3)
        )
        self.assertEqual(arrays["rhs_values"].shape, (4096,))
        self.assertEqual(
            str(arrays["monomial_variable_indices"].dtype), "uint16"
        )

    def test_independent_verifier_does_not_call_primary_enumerator(self):
        system = primary.generate_sparse_system(6, 2)
        arrays = system_arrays(system)
        with mock.patch.object(
            primary,
            "perfect_matchings",
            side_effect=AssertionError("primary enumerator was called"),
        ):
            report = independent_verifier.verify_system_arrays(
                6,
                2,
                arrays["equation_offsets"],
                arrays["monomial_variable_indices"],
                arrays["rhs_values"],
            )
        self.assertTrue(report.exact)
        self.assertEqual(report.matching_count, 15)

    def test_invalid_sparse_encodings_and_modular_denominators_fail(self):
        system = primary.generate_sparse_system(4, 3)
        offsets = list(system.equation_offsets)
        offsets[1] -= 1
        with self.assertRaises(primary.KrennSystemError):
            primary.SparsePolynomialSystem(
                4,
                3,
                tuple(offsets),
                system.monomial_variable_indices,
                system.rhs_values,
            )
        with self.assertRaises(KrennWitnessError):
            fraction_mod(Fraction(1, 31), 31)


if __name__ == "__main__":
    unittest.main()

