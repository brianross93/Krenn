import unittest

from experiments.krenn_quantum_graph.system import (
    generate_sparse_system,
)
from experiments.krenn_quantum_graph.ternary_search import (
    KrennTernarySearchError,
    N6_D3_SEED_DEFECT_EQUATION,
    TernaryCandidate,
    TernaryCheckpoint,
    TernaryPartialAssignment,
    TernarySearchPlan,
    checkpoint_from_dict,
    deterministic_variable_order,
    evaluate_ternary_candidate,
    has_canonical_ghz_target,
    n6_d3_seed_values,
    partial_interval_feasible,
    replay_checkpoint,
    run_bounded_ternary_search,
)


class KrennTernarySearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = generate_sparse_system(6, 3)
        cls.seed = n6_d3_seed_values()

    def plan(self, *, nodes=7, frontier=400):
        return TernarySearchPlan(
            node_cap=nodes,
            time_cap_seconds=10.0,
            support_cap=12,
            frontier_cap=frontier,
        )

    def test_nine_weight_seed_is_checked_in_all_original_equations(self):
        candidate = evaluate_ternary_candidate(
            self.system, self.seed
        )
        self.assertEqual(candidate.support_size, 9)
        self.assertEqual(candidate.nonzero_residual_count, 1)
        self.assertEqual(candidate.residual_l1, 1)
        self.assertEqual(
            tuple(
                equation
                for equation, residual in enumerate(
                    candidate.residuals
                )
                if residual
            ),
            (N6_D3_SEED_DEFECT_EQUATION,),
        )
        self.assertEqual(
            candidate.residuals[N6_D3_SEED_DEFECT_EQUATION], 1
        )
        self.assertFalse(candidate.exact_solution)

    def test_variable_order_and_interval_pruning_are_deterministic(self):
        order = deterministic_variable_order(
            self.system, self.seed
        )
        self.assertEqual(len(order), 135)
        self.assertEqual(set(order), set(range(135)))
        self.assertEqual(
            order,
            deterministic_variable_order(self.system, self.seed),
        )
        defect_variables = {
            variable
            for monomial in self.system.equation_monomials(
                N6_D3_SEED_DEFECT_EQUATION
            )
            for variable in monomial
        }
        self.assertEqual(set(order[:15]), defect_variables)
        zero_repairs = tuple(
            sorted(
                variable
                for variable in defect_variables
                if self.seed[variable] == 0
            )
        )
        active_seed_slots = tuple(
            sorted(
                variable
                for variable in defect_variables
                if self.seed[variable] != 0
            )
        )
        self.assertEqual(order[:12], zero_repairs)
        self.assertEqual(order[12:15], active_seed_slots)
        self.assertTrue(
            partial_interval_feasible(
                self.system,
                TernaryPartialAssignment(()),
                order,
                support_cap=12,
            )
        )
        full_seed = TernaryPartialAssignment(
            tuple(self.seed[variable] for variable in order)
        )
        self.assertFalse(
            partial_interval_feasible(
                self.system, full_seed, order, support_cap=12
            )
        )

    def test_small_bounded_run_is_exact_reportable_and_no_claim(self):
        result = run_bounded_ternary_search(
            self.system, self.plan(nodes=7)
        )
        self.assertEqual(result.termination, "node-cap-reached")
        self.assertEqual(result.nodes_examined_this_run, 7)
        self.assertEqual(result.checkpoint.total_nodes_examined, 7)
        self.assertEqual(
            result.checkpoint.best_candidate,
            evaluate_ternary_candidate(
                self.system,
                result.checkpoint.best_candidate.values,
            ),
        )
        self.assertTrue(result.exact_original_equations_checked)
        self.assertFalse(result.solution_certified)
        self.assertFalse(result.proof_over_C)
        self.assertFalse(result.tree_certificate_complete)
        self.assertFalse(result.exhaustive_nonexistence)
        self.assertFalse(result.nonexistence_proved)
        self.assertEqual(result.symmetry_weight_equalities, 0)
        self.assertEqual(result.search_mode, "fixed-target")
        self.assertTrue(result.resumable)
        report = result.to_dict()
        self.assertEqual(report["termination"], "node-cap-reached")
        self.assertEqual(report["search_mode"], "fixed-target")
        self.assertEqual(report["plan"]["ambient_assignments"], "3^135")
        self.assertFalse(report["nonexistence_proved"])
        self.assertFalse(report["tree_certificate_complete"])

    def test_resume_matches_one_deterministic_run(self):
        first = run_bounded_ternary_search(
            self.system, self.plan(nodes=3)
        )
        resumed = run_bounded_ternary_search(
            self.system,
            self.plan(nodes=4),
            checkpoint=first.checkpoint,
        )
        single = run_bounded_ternary_search(
            self.system, self.plan(nodes=7)
        )
        self.assertEqual(
            resumed.checkpoint, single.checkpoint
        )
        self.assertEqual(
            resumed.checkpoint.total_nodes_examined, 7
        )
        self.assertEqual(resumed.nodes_examined_this_run, 4)

    def test_checkpoint_round_trip_and_exact_replay(self):
        result = run_bounded_ternary_search(
            self.system, self.plan(nodes=5)
        )
        decoded = checkpoint_from_dict(
            result.checkpoint.to_dict()
        )
        self.assertEqual(decoded, result.checkpoint)
        self.assertEqual(
            replay_checkpoint(
                self.system, self.plan(nodes=5), decoded
            ),
            decoded,
        )

        best = decoded.best_candidate
        fake_residuals = (0,) * len(best.residuals)
        corrupted = TernaryCheckpoint(
            variable_order=decoded.variable_order,
            preferred_values=decoded.preferred_values,
            frontier=decoded.frontier,
            best_candidate=TernaryCandidate(
                values=best.values,
                residuals=fake_residuals,
                support_size=best.support_size,
                nonzero_residual_count=0,
                residual_l1=0,
            ),
            total_nodes_examined=decoded.total_nodes_examined,
            support_cap=decoded.support_cap,
        )
        with self.assertRaisesRegex(
            KrennTernarySearchError, "failed exact replay"
        ):
            replay_checkpoint(
                self.system, self.plan(nodes=5), corrupted
            )

    def test_frontier_cap_stops_without_discarding_resume_node(self):
        result = run_bounded_ternary_search(
            self.system,
            self.plan(nodes=100, frontier=1),
        )
        self.assertEqual(result.termination, "frontier-cap-reached")
        self.assertEqual(result.nodes_examined_this_run, 1)
        self.assertEqual(
            result.checkpoint.frontier,
            (TernaryPartialAssignment(()),),
        )
        self.assertTrue(result.resumable)
        self.assertFalse(result.nonexistence_proved)

    def test_caps_and_problem_scope_fail_closed(self):
        with self.assertRaises(KrennTernarySearchError):
            TernarySearchPlan(
                node_cap=0,
                time_cap_seconds=10,
                support_cap=12,
                frontier_cap=100,
            )
        with self.assertRaisesRegex(
            KrennTernarySearchError, "fixed-target mode only"
        ):
            TernarySearchPlan(
                node_cap=1,
                time_cap_seconds=10,
                support_cap=12,
                frontier_cap=100,
                mode="ghz-orbit",
            )

    def test_tampered_non_ghz_target_fails_every_entry_point(self):
        class TamperedTarget:
            def __init__(self, source):
                self.n = source.n
                self.d = source.d
                self.rhs_values = list(source.rhs_values)
                self.rhs_values[0] = 0
                self.variable_count = source.variable_count
                self.equation_count = source.equation_count
                self.equation_monomials = source.equation_monomials

        tampered = TamperedTarget(self.system)
        self.assertFalse(has_canonical_ghz_target(tampered))
        plan = self.plan(nodes=2)
        with self.assertRaisesRegex(
            KrennTernarySearchError, "canonical GHZ RHS"
        ):
            evaluate_ternary_candidate(tampered, self.seed)
        with self.assertRaisesRegex(
            KrennTernarySearchError, "canonical GHZ RHS"
        ):
            run_bounded_ternary_search(tampered, plan)

        valid = run_bounded_ternary_search(self.system, plan)
        with self.assertRaisesRegex(
            KrennTernarySearchError, "canonical GHZ RHS"
        ):
            replay_checkpoint(
                tampered, plan, valid.checkpoint
            )
        with self.assertRaises(KrennTernarySearchError):
            TernarySearchPlan(
                node_cap=1,
                time_cap_seconds=0,
                support_cap=12,
                frontier_cap=100,
            )
        with self.assertRaises(KrennTernarySearchError):
            TernarySearchPlan(
                node_cap=1,
                time_cap_seconds=10,
                support_cap=8,
                frontier_cap=100,
            )


if __name__ == "__main__":
    unittest.main()
