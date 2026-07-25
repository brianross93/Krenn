import copy
import unittest

from experiments.krenn_quantum_graph.localized_chart_residual_grading import (
    KrennResidualGradingError,
    REPAIR_MONOMIAL_REPRESENTATIVES,
    repair_residual_grading,
    residual_grading_audit,
    verify_residual_grading_audit,
)


class KrennLocalizedChartResidualGradingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = residual_grading_audit()

    def test_all_four_integral_quotients_and_block_censuses(self):
        expected = {
            (11, 65): (1, 8, 129, 5, 558, 7),
            (29, 47): (1, 8, 129, 5, 642, 7),
            (11, 55, 133): (2, 7, 128, 8, 432, 11),
            (29, 55, 106): (2, 7, 128, 8, 576, 11),
        }
        self.assertEqual(
            tuple(
                tuple(row["representative_ambient_monomial"])
                for row in self.audit["repair_charts"]
            ),
            REPAIR_MONOMIAL_REPRESENTATIVES,
        )
        for row in self.audit["repair_charts"]:
            representative = tuple(
                row["representative_ambient_monomial"]
            )
            quotient = row["integral_character_quotient"]
            generators = row["generators"]
            observed = (
                quotient["fixed_factor_rank"],
                quotient["residual_rank"],
                row["variables"]["count"],
                row["variables"]["zero_character_variables"],
                generators["character_blocks"],
                generators["zero_character_generators"],
            )
            self.assertEqual(observed, expected[representative])
            self.assertEqual(
                quotient["smith_diagonal"],
                [1] * quotient["fixed_factor_rank"],
            )
            self.assertEqual(
                abs(quotient["section_minor_determinant"]), 1
            )
            self.assertTrue(
                quotient["fixed_character_lattice_saturated"]
            )
            self.assertTrue(quotient["quotient_is_split_over_Z"])
            self.assertTrue(
                row["variables"][
                    "branch_inverse_character_is_zero_after_quotient"
                ]
            )
            self.assertEqual(generators["count"], 730)
            self.assertTrue(
                generators["all_terms_residual_homogeneous"]
            )

    def test_character_lineality_is_not_promoted_to_tropical_cones(self):
        for row in self.audit["repair_charts"]:
            quotient = row["integral_character_quotient"]
            lineality = row["valuation_lineality"]
            boundary = row["claim_boundary"]
            self.assertEqual(
                lineality["dimension"], quotient["residual_rank"]
            )
            self.assertTrue(
                lineality["initial_forms_constant_along_this_subspace"]
            )
            self.assertFalse(
                lineality["tropical_valuation_vectors_enumerated"]
            )
            self.assertFalse(
                lineality["tropical_or_grobner_cones_enumerated"]
            )
            self.assertFalse(
                boundary["grading_is_a_tropical_decomposition"]
            )
            self.assertFalse(
                boundary["chart_unit_or_proper_status_decided"]
            )

    def test_direct_replay_is_deterministic(self):
        for representative in REPAIR_MONOMIAL_REPRESENTATIVES:
            first = repair_residual_grading(representative)
            second = repair_residual_grading(representative)
            self.assertEqual(first, second)
            digest = first["generators"][
                "variable_and_generator_character_sha256"
            ]
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_verifier_rejects_claim_escalation(self):
        self.assertEqual(
            verify_residual_grading_audit(self.audit), self.audit
        )
        corrupted = copy.deepcopy(self.audit)
        corrupted["claim_boundary"]["tropical_cones_enumerated"] = True
        with self.assertRaises(KrennResidualGradingError):
            verify_residual_grading_audit(corrupted)

    def test_unknown_repair_representative_is_rejected(self):
        with self.assertRaises(KrennResidualGradingError):
            repair_residual_grading((1, 2))


if __name__ == "__main__":
    unittest.main()
