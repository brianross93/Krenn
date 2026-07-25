import copy
import unittest

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    DERIVATIVE_REPRESENTATIVES,
    NATURAL_DEFECT_EQUATION,
    sparse_derivative_gauge_slice,
)
from experiments.krenn_quantum_graph.localized_chart_sparse_elimination import (
    KrennSparseEliminationError,
    singular_triangular_derivative_script,
    sparse_derivative_quotient_grading,
    sparse_elimination_audit,
    triangular_sparse_derivative_system,
    verify_sparse_elimination_audit,
)


def add(target, monomial, coefficient):
    key = tuple(sorted(monomial))
    total = target.get(key, 0) + coefficient
    if total:
        target[key] = total
    else:
        target.pop(key, None)


class KrennLocalizedChartSparseEliminationTest(unittest.TestCase):
    def test_triangular_rewrite_preserves_the_ideal_exactly(self):
        for ambient in DERIVATIVE_REPRESENTATIVES:
            original = sparse_derivative_gauge_slice(ambient)
            transformed = triangular_sparse_derivative_system(ambient)
            position = original.generator_labels.index(
                ("mixed", NATURAL_DEFECT_EQUATION)
            )
            selected = original.derivative_original_local_variable
            replay = {
                monomial: coefficient
                for coefficient, monomial in transformed.generators[
                    position
                ].terms
            }
            for coefficient, monomial in original.generators[-1].terms:
                add(replay, (*monomial, selected), coefficient)
            self.assertEqual(
                replay,
                {
                    monomial: coefficient
                    for coefficient, monomial in original.generators[
                        position
                    ].terms
                },
            )
            for index, polynomial in enumerate(original.generators):
                if index != position:
                    self.assertEqual(
                        transformed.generators[index], polynomial
                    )
            self.assertEqual(transformed.term_count, 10_940)
            self.assertEqual(transformed.maximum_degree, 4)
            self.assertEqual(
                transformed.secondary_pivot_variable,
                {11: 62, 29: 44}[ambient],
            )
            self.assertEqual(
                tuple(
                    term
                    for term in transformed.generators[position].terms
                    if selected in term[1]
                ),
                ((1, (selected,)),),
            )
            secondary = transformed.secondary_pivot_variable
            self.assertEqual(
                tuple(
                    term
                    for term in transformed.generators[-1].terms
                    if secondary in term[1]
                ),
                ((1, (secondary,)),),
            )

    def test_both_split_quotient_gradings_replay_exactly(self):
        expected = {11: 558, 29: 642}
        for ambient in DERIVATIVE_REPRESENTATIVES:
            row = sparse_derivative_quotient_grading(ambient)
            self.assertEqual(row["quotient"]["smith_diagonal"], [1])
            self.assertEqual(row["quotient"]["residual_rank"], 8)
            self.assertTrue(row["quotient"]["split_over_Z"])
            self.assertEqual(
                row["variables"]["character_blocks"], 105
            )
            self.assertEqual(
                row["variables"]["zero_character_variables"], 5
            )
            self.assertEqual(
                row["generators"]["character_blocks"],
                expected[ambient],
            )
            self.assertEqual(
                row["generators"]["zero_character_generators"], 7
            )
            self.assertTrue(
                row["generators"][
                    "all_terms_quotient_homogeneous"
                ]
            )
            self.assertEqual(
                row["strict_derivative_stabilizer"]["order"], 2
            )
            self.assertFalse(
                row["claim_boundary"]["unit_or_proper_status_decided"]
            )

    def test_singular_dp_and_elimination_exports_are_deterministic(self):
        for ambient in DERIVATIVE_REPRESENTATIVES:
            dp = singular_triangular_derivative_script(
                ambient, characteristic=31, order="dp"
            )
            elimination = singular_triangular_derivative_script(
                ambient,
                characteristic=31,
                order="selected-elimination",
            )
            two_pivot = singular_triangular_derivative_script(
                ambient,
                characteristic=31,
                order="two-pivot-elimination",
            )
            self.assertEqual(
                dp,
                singular_triangular_derivative_script(
                    ambient, characteristic=31, order="dp"
                ),
            )
            self.assertIn(
                "KRENN_SPARSE_TRIANGULAR_DP_PARSE_OK", dp
            )
            self.assertIn(
                "KRENN_SPARSE_TRIANGULAR_SELECTED_ELIMINATION_PARSE_OK",
                elimination,
            )
            selected = triangular_sparse_derivative_system(
                ambient
            ).selected_original_local_variable
            self.assertIn(
                f"ring r=31,(x{selected},", elimination
            )
            self.assertIn(",(lp(1),dp(128));", elimination)
            secondary = triangular_sparse_derivative_system(
                ambient
            ).secondary_pivot_variable
            self.assertIn(
                f"ring r=31,(x{selected},x{secondary},",
                two_pivot,
            )
            self.assertIn(",(lp(2),dp(127));", two_pivot)
            self.assertIn("ideal G=std(I);", elimination)

    def test_export_validation_fails_closed(self):
        with self.assertRaises(KrennSparseEliminationError):
            singular_triangular_derivative_script(
                11, characteristic=4
            )
        with self.assertRaises(KrennSparseEliminationError):
            singular_triangular_derivative_script(
                11, order="not-an-order"
            )
        with self.assertRaises(KrennSparseEliminationError):
            singular_triangular_derivative_script(
                11, algorithm="not-an-algorithm"
            )

    def test_audit_verifier_rejects_claim_escalation(self):
        audit = sparse_elimination_audit()
        self.assertEqual(
            verify_sparse_elimination_audit(audit), audit
        )
        corrupted = copy.deepcopy(audit)
        corrupted["claim_boundary"]["natural_chart_decided"] = True
        with self.assertRaises(KrennSparseEliminationError):
            verify_sparse_elimination_audit(corrupted)


if __name__ == "__main__":
    unittest.main()
