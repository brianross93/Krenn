from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.support_search import (
    KrennSupportSearchError,
    N6_D4_SEED_DEFECTS,
    N6_D4_SEED_FACTORS,
    SupportClosurePlan,
    bounded_support_closure,
    color_partition_census,
    n6_d4_seed_coordinates,
    n6_d4_seed_support,
    n6_d4_seed_witness,
    singleton_obstructions,
    support_profile,
)
from experiments.krenn_quantum_graph.system import (
    coloring_index,
    generate_sparse_system,
    variable_key,
)
from experiments.krenn_quantum_graph.witness import evaluate_exact


class KrennNativeSupportSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = generate_sparse_system(6, 4)
        cls.support = n6_d4_seed_support()
        cls.profile = support_profile(cls.system, cls.support)

    def test_exact_twelve_weight_seed_has_only_four_defects(self):
        self.assertEqual(len(self.support), 12)
        expected_coordinates = {
            (i, j, color, color)
            for color, factor in enumerate(N6_D4_SEED_FACTORS)
            for i, j in factor
        }
        self.assertEqual(
            set(n6_d4_seed_coordinates()), expected_coordinates
        )
        self.assertEqual(
            {
                variable_key(6, 4, index) for index in self.support
            },
            expected_coordinates,
        )

        evaluation = evaluate_exact(
            self.system, n6_d4_seed_witness()
        )
        defects = tuple(
            index
            for index, residual in enumerate(evaluation.residuals)
            if residual
        )
        expected_defects = tuple(
            equation for equation, _coloring, _matching in N6_D4_SEED_DEFECTS
        )
        self.assertEqual(defects, expected_defects)
        self.assertTrue(
            all(
                evaluation.residuals[equation] == Fraction(1)
                for equation in defects
            )
        )
        self.assertEqual(
            sum(value == 0 for value in evaluation.equation_values),
            4088,
        )
        self.assertEqual(
            sum(value == 1 for value in evaluation.equation_values),
            8,
        )

    def test_support_profile_replays_feasible_matching_counts(self):
        constant_equations = tuple(
            coloring_index(6, 4, (color,) * 6)
            for color in range(4)
        )
        defect_equations = tuple(
            row[0] for row in N6_D4_SEED_DEFECTS
        )
        self.assertEqual(self.profile.support_size, 12)
        self.assertEqual(self.profile.active_equation_count, 8)
        self.assertEqual(
            tuple(
                index
                for index, count in enumerate(
                    self.profile.feasible_matching_counts
                )
                if count
            ),
            tuple(sorted((*constant_equations, *defect_equations))),
        )
        self.assertTrue(
            all(
                self.profile.feasible_matching_counts[equation] == 1
                for equation in (*constant_equations, *defect_equations)
            )
        )
        self.assertEqual(
            self.profile.singleton_nonconstant_equations,
            defect_equations,
        )
        self.assertEqual(
            self.profile.unit_weight_defect_equations,
            defect_equations,
        )
        self.assertFalse(
            self.profile.singleton_free_necessary_condition
        )
        self.assertFalse(self.profile.solution_certified)
        self.assertFalse(self.profile.proof_over_C)

    def test_four_singleton_obstructions_have_expected_matchings(self):
        obstructions = singleton_obstructions(
            self.system, self.support, profile=self.profile
        )
        self.assertEqual(len(obstructions), 4)
        for obstruction, expected in zip(
            obstructions, N6_D4_SEED_DEFECTS
        ):
            equation, coloring, matching = expected
            self.assertEqual(obstruction.equation, equation)
            self.assertEqual(obstruction.coloring, coloring)
            self.assertEqual(
                tuple(
                    variable_key(6, 4, variable)[:2]
                    for variable in obstruction.monomial
                ),
                matching,
            )
            self.assertEqual(
                tuple(
                    variable_key(6, 4, variable)[2:]
                    for variable in obstruction.monomial
                ),
                tuple(
                    (coloring[i], coloring[j]) for i, j in matching
                ),
            )

    def test_color_partition_census_is_complete(self):
        self.assertEqual(
            color_partition_census(6, 4),
            {
                (6,): 4,
                (5, 1): 72,
                (4, 2): 180,
                (4, 1, 1): 360,
                (3, 3): 120,
                (3, 2, 1): 1440,
                (3, 1, 1, 1): 480,
                (2, 2, 2): 360,
                (2, 2, 1, 1): 1080,
            },
        )
        self.assertEqual(
            sum(color_partition_census(6, 4).values()), 4096
        )

    def test_support_cap_fourteen_exhausts_first_closure_layer(self):
        plan = SupportClosurePlan(
            n=6,
            d=4,
            initial_support=self.support,
            node_cap=100,
            support_cap=14,
        )
        result = bounded_support_closure(self.system, plan)
        self.assertEqual(
            result.termination,
            "frontier-exhausted-under-support-cap",
        )
        # Six alternative matchings retain one active seed edge and add
        # exactly two coordinates.  All other alternatives exceed the cap.
        self.assertEqual(result.nodes_examined, 7)
        self.assertEqual(result.supports_discovered, 7)
        self.assertEqual(result.frontier_remaining, 0)
        self.assertEqual(result.singleton_free_supports, ())
        self.assertTrue(result.necessary_condition_only)
        self.assertFalse(result.solution_certified)
        self.assertFalse(result.proof_over_C)
        self.assertFalse(result.nonexistence_proved)
        self.assertEqual(result.symmetry_weight_equalities, 0)
        self.assertEqual(plan.symmetry_weight_equalities, 0)

    def test_node_bounded_search_is_deterministic(self):
        plan = SupportClosurePlan(
            n=6,
            d=4,
            initial_support=self.support,
            node_cap=3,
            support_cap=20,
        )
        first = bounded_support_closure(self.system, plan)
        second = bounded_support_closure(self.system, plan)
        self.assertEqual(first, second)
        self.assertEqual(first.termination, "node-cap-reached")
        self.assertEqual(first.nodes_examined, 3)
        self.assertLessEqual(
            max(
                map(len, first.singleton_free_supports),
                default=len(self.support),
            ),
            plan.support_cap,
        )
        self.assertFalse(first.proof_over_C)

    def test_plan_and_support_validation_fail_closed(self):
        with self.assertRaises(KrennSupportSearchError):
            SupportClosurePlan(
                n=6,
                d=4,
                initial_support=self.support,
                node_cap=0,
                support_cap=14,
            )
        with self.assertRaises(KrennSupportSearchError):
            support_profile(
                self.system, (*self.support, self.support[0])
            )


if __name__ == "__main__":
    unittest.main()
