import copy
from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.star_linearization import (
    KrennStarLinearizationError,
    star_linearization,
    star_linearization_audit,
    verify_star_linearization_audit,
)
from experiments.krenn_quantum_graph.star_linearization_independent import (
    independent_star_linearization_audit,
)
from experiments.krenn_quantum_graph.system import (
    generate_sparse_system,
)
from experiments.krenn_quantum_graph.witness import (
    SparseWitness,
    evaluate_exact,
)


class KrennStarLinearizationTest(unittest.TestCase):
    def test_all_six_exact_matrix_censuses_and_independent_hashes(self):
        for apex in range(6):
            matrix = star_linearization(apex)
            independent = independent_star_linearization_audit(apex)
            self.assertEqual(len(matrix.row_colorings), 243)
            self.assertEqual(len(matrix.columns), 15)
            self.assertEqual(matrix.nonzero_entry_count, 1_215)
            self.assertEqual(matrix.coefficient_term_count, 3_645)
            self.assertEqual(matrix.target_rows, (0, 121, 242))
            self.assertEqual(
                matrix.fingerprint(), independent["fingerprint"]
            )
            self.assertTrue(
                independent["all_729_equations_reconstructed"]
            )

    def test_direct_exact_evaluation_equals_star_factorization(self):
        system = generate_sparse_system(6, 3)
        values = {
            index: Fraction((index * 17 + 5) % 23 - 11, 7)
            for index in range(135)
        }
        witness = SparseWitness.from_index_values(6, 3, values)
        direct = evaluate_exact(system, witness).equation_values
        for apex in range(6):
            matrix = star_linearization(apex)
            replay = []
            for apex_color in range(3):
                star_values = [
                    values[index]
                    for index in matrix.star_variable_blocks[apex_color]
                ]
                for row in range(243):
                    total = Fraction(0)
                    for column, polynomial in enumerate(
                        matrix.entries[row]
                    ):
                        coefficient = sum(
                            (
                                values[left] * values[right]
                                for left, right in polynomial
                            ),
                            Fraction(0),
                        )
                        total += star_values[column] * coefficient
                    replay.append(total)
            # Equation order groups by the apex color only for apex zero.
            if apex == 0:
                self.assertEqual(tuple(replay), direct)
            else:
                reordered = []
                for equation in range(729):
                    coloring = []
                    value = equation
                    for power in (243, 81, 27, 9, 3, 1):
                        digit, value = divmod(value, power)
                        coloring.append(digit)
                    remaining_row = 0
                    for vertex in matrix.remaining_vertices:
                        remaining_row = (
                            remaining_row * 3 + coloring[vertex]
                        )
                    reordered.append(
                        replay[coloring[apex] * 243 + remaining_row]
                    )
                self.assertEqual(tuple(reordered), direct)

    def test_differencing_shortcut_is_rejected_with_cross_terms(self):
        audit = star_linearization_audit()
        row = audit["differencing_review"]
        self.assertFalse(
            row["proposed_single_column_differencing_lemma_valid"]
        )
        self.assertEqual(
            row["signed_cross_monomials_from_other_four_columns"], 24
        )
        self.assertEqual(
            row["signed_monomials_in_full_row_difference"], 30
        )
        self.assertTrue(
            row["constant_color_case_tree_derived_from_the_lemma_is_void"]
        )
        self.assertFalse(row["all_constant_branch_is_forced"])
        self.assertFalse(row["all_constant_branch_used_downstream"])

    def test_audit_round_trip_and_claim_escalation_rejection(self):
        audit = star_linearization_audit()
        self.assertEqual(verify_star_linearization_audit(audit), audit)
        corrupted = copy.deepcopy(audit)
        corrupted["claim_boundary"]["finite_counterexample_found"] = True
        with self.assertRaises(KrennStarLinearizationError):
            verify_star_linearization_audit(corrupted)

    def test_invalid_coordinates_fail_closed(self):
        for apex in (-1, 6):
            with self.assertRaises(KrennStarLinearizationError):
                star_linearization(apex)
        matrix = star_linearization(0)
        with self.assertRaises(KrennStarLinearizationError):
            matrix.coefficient(243, 0)


if __name__ == "__main__":
    unittest.main()
