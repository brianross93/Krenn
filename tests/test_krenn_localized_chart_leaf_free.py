import copy
import unittest

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    DERIVATIVE_REPRESENTATIVES,
)
from experiments.krenn_quantum_graph.localized_chart_leaf_free import (
    KrennLeafFreeError,
    derivative_leaf_free_system,
    leaf_free_saturation_audit,
    repair_leaf_free_system,
    singular_leaf_free_saturation_script,
    verify_leaf_free_saturation_audit,
)
from experiments.krenn_quantum_graph.localized_chart_monomial_atlas import (
    REPAIR_MONOMIAL_REPRESENTATIVES,
)


class KrennLocalizedChartLeafFreeTest(unittest.TestCase):
    def test_derivative_leaf_free_censuses_and_factors(self):
        for ambient in DERIVATIVE_REPRESENTATIVES:
            system = derivative_leaf_free_system(ambient)
            self.assertEqual(system.variable_count, 126)
            self.assertEqual(len(system.generators), 727)
            self.assertEqual(system.term_count, 10_894)
            self.assertEqual(system.maximum_degree, 3)
            self.assertEqual(
                system.saturation_factor_labels,
                ("A0", "A1", "A2"),
            )
            self.assertTrue(
                all(
                    factor.term_count == 15 and factor.degree == 3
                    for factor in system.saturation_factors
                )
            )
            self.assertTrue(
                all(
                    variable < system.variable_count
                    for polynomial in (
                        *system.generators,
                        *system.saturation_factors,
                    )
                    for _coefficient, monomial in polynomial.terms
                    for variable in monomial
                )
            )

    def test_repair_leaf_free_censuses_and_branch_factors(self):
        expected_variables = {
            (11, 65): 125,
            (29, 47): 125,
            (11, 55, 133): 124,
            (29, 55, 106): 124,
        }
        for representative in REPAIR_MONOMIAL_REPRESENTATIVES:
            system = repair_leaf_free_system(representative)
            self.assertEqual(
                system.variable_count,
                expected_variables[representative],
            )
            self.assertEqual(len(system.generators), 726)
            self.assertEqual(system.term_count, 10_890)
            self.assertEqual(system.maximum_degree, 3)
            self.assertEqual(
                system.saturation_factor_labels,
                ("y", "A0", "A1", "A2"),
            )
            self.assertEqual(
                system.saturation_factors[0].term_count, 1
            )
            self.assertEqual(system.saturation_factors[0].degree, 1)
            for factor in system.saturation_factors[1:]:
                self.assertEqual(factor.term_count, 15)
                self.assertEqual(factor.degree, 3)

    def test_all_mixed_generators_retain_fifteen_matching_terms(self):
        systems = [
            *(
                derivative_leaf_free_system(ambient)
                for ambient in DERIVATIVE_REPRESENTATIVES
            ),
            *(
                repair_leaf_free_system(representative)
                for representative in REPAIR_MONOMIAL_REPRESENTATIVES
            ),
        ]
        for system in systems:
            mixed = tuple(
                polynomial
                for label, polynomial in zip(
                    system.generator_labels,
                    system.generators,
                    strict=True,
                )
                if label[0] == "mixed"
            )
            self.assertEqual(len(mixed), 726)
            self.assertTrue(
                all(polynomial.term_count == 15 for polynomial in mixed)
            )

    def test_audit_is_fail_closed_and_round_trippable(self):
        audit = leaf_free_saturation_audit()
        self.assertEqual(
            verify_leaf_free_saturation_audit(audit), audit
        )
        self.assertEqual(
            audit["claim_boundary"]["saturation_stages_completed"], 0
        )
        self.assertTrue(
            audit["exact_equivalence"][
                "repair_localizers_are_u_times_y_minus_one"
            ]
        )
        corrupted = copy.deepcopy(audit)
        corrupted["claim_boundary"]["natural_chart_decided"] = True
        with self.assertRaises(KrennLeafFreeError):
            verify_leaf_free_saturation_audit(corrupted)

    def test_sequential_saturation_export_is_deterministic(self):
        initial = singular_leaf_free_saturation_script(
            (11,), characteristic=31, initial_stage_only=True
        )
        full = singular_leaf_free_saturation_script(
            (11,), characteristic=31, initial_stage_only=False
        )
        self.assertEqual(
            initial,
            singular_leaf_free_saturation_script(
                (11,),
                characteristic=31,
                initial_stage_only=True,
            ),
        )
        self.assertIn('LIB "elim.lib";', initial)
        self.assertIn("ideal S1=sat(S0,H0);", initial)
        self.assertNotIn("ideal S2=sat(S1,H1);", initial)
        self.assertIn("ideal S3=sat(S2,H2);", full)
        self.assertLess(
            initial.index("KRENN_LEAF_FREE_11_PARSE_OK"),
            initial.index("ideal S1=sat(S0,H0);"),
        )
        repair = singular_leaf_free_saturation_script(
            (11, 65), characteristic=31
        )
        self.assertIn("ideal S4=sat(S3,H3);", repair)
        with self.assertRaises(KrennLeafFreeError):
            singular_leaf_free_saturation_script(
                (11,), characteristic=4
            )


if __name__ == "__main__":
    unittest.main()
