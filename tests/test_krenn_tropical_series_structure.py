import copy
import json
import unittest
from pathlib import Path

from experiments.krenn_quantum_graph.tropical_series_structure import (
    KrennTropicalSeriesError,
    TROPICAL_SERIES_SCHEMA,
    tropical_series_structure_audit,
    verify_tropical_series_structure_audit,
)


class TropicalSeriesStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = tropical_series_structure_audit()

    def test_known_exact_support_cone_is_only_gauge_lineality(self):
        row = self.audit["known_exact_support_moving_cone"]
        self.assertEqual(row["relation_rank_over_Q"], 4)
        self.assertEqual(row["tropical_kernel_dimension"], 6)
        self.assertEqual(row["color_gauge_rank"], 6)
        self.assertTrue(all(row["exact_checks"].values()))
        self.assertEqual(
            row["known_Laurent_valuation"],
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 1],
        )

    def test_all_fixed_victim_nine_coordinate_nodes_are_classified(self):
        row = self.audit["known_exact_support_moving_cone"]
        self.assertEqual(row["fixed_victim_nine_coordinate_seed_orbits"], 104)
        self.assertEqual(
            row["fixed_victim_nine_coordinate_singleton_orbits"], 103
        )
        self.assertEqual(row["fixed_victim_surviving_orbit_size"], 4)

    def test_residual_grading_is_blind_to_the_defect_balance(self):
        row = self.audit["constant_fiber_residual_defect_star"]
        self.assertEqual(row["residual_character_lattice"], "Z^9")
        self.assertEqual(row["repair_term_count"], 14)
        self.assertEqual(row["repair_orbit_sizes"], [3, 3, 6, 2])
        self.assertTrue(all(row["exact_checks"].values()))
        self.assertTrue(
            all(
                term["residual_character"] == [0] * 9
                for term in row["terms"]
            )
        )

    def test_pairwise_natural_nodes_do_not_form_a_bridge(self):
        row = self.audit["natural_node_cross_comparison"]
        self.assertEqual(
            row["pairwise_union_orbits_under_victim_stabilizer"], 1
        )
        self.assertEqual(len(row["pairwise_unions"]), 6)
        self.assertTrue(all(row["exact_checks"].values()))
        for union in row["pairwise_unions"]:
            self.assertEqual(union["union_support_size"], 13)
            self.assertEqual(
                len(union["nonvictim_singleton_equations"]), 4
            )
            self.assertEqual(
                union["other_active_nonvictim_equation_counts"], []
            )
        completion = row["canonical_pair_completion"]
        self.assertEqual(completion["simultaneous_completion_choices"], 38_416)
        self.assertEqual(
            completion["minimum_added_coordinates"], [[5, 7, 86, 131]]
        )
        self.assertEqual(completion["minimum_added_coordinate_count"], 4)
        self.assertEqual(
            completion["minimum_branch_closure_through_support_24"]["nodes"],
            1_975,
        )
        minimum_closure = completion[
            "minimum_branch_closure_through_support_24"
        ]
        self.assertTrue(
            minimum_closure["bounded_search_complete_through_support"]
        )
        self.assertGreater(minimum_closure["cutoff_frontier_nodes"], 0)
        self.assertGreater(minimum_closure["pruned_child_transitions"], 0)
        self.assertFalse(minimum_closure["frontier_exhausted"])
        full = row["full_canonical_pair_closure_through_support_21"]
        self.assertEqual(full["nodes"], 7_564)
        self.assertEqual(len(full["singleton_free_terminals"]), 1)
        self.assertEqual(len(full["terminal_active_mixed_binomials"]), 14)
        self.assertTrue(
            full["bounded_search_complete_through_support"]
        )
        self.assertGreater(full["cutoff_frontier_nodes"], 0)
        self.assertGreater(full["pruned_child_transitions"], 0)
        self.assertFalse(full["frontier_exhausted"])
        self.assertTrue(
            all(
                receipt["integer_exponent_identity_exact"]
                for receipt in full["pure_cancellation_receipts"]
            )
        )

    def test_claims_remain_fail_closed(self):
        self.assertEqual(self.audit["schema"], TROPICAL_SERIES_SCHEMA)
        claims = self.audit["claim_boundary"]
        self.assertTrue(claims["known_exact_support_tropical_cone_classified"])
        self.assertFalse(
            claims[
                "known_Laurent_valuation_is_transverse_after_gauge_quotient"
            ]
        )
        self.assertTrue(
            claims["minimal_pairwise_natural_node_bridges_excluded"]
        )
        self.assertTrue(
            claims[
                "pairwise_natural_node_bridges_through_support_21_excluded"
            ]
        )
        self.assertFalse(claims["all_higher_support_node_bridges_excluded"])
        self.assertFalse(claims["higher_support_tropical_cones_enumerated"])
        self.assertFalse(claims["all_u_zero_vertical_components_excluded"])
        self.assertFalse(
            claims["all_zero_residual_sequences_proved_boundary"]
        )

    def test_round_trip_and_corruption_rejection(self):
        self.assertEqual(
            verify_tropical_series_structure_audit(self.audit), self.audit
        )
        corrupt = copy.deepcopy(self.audit)
        corrupt["claim_boundary"][
            "all_zero_residual_sequences_proved_boundary"
        ] = True
        with self.assertRaises(KrennTropicalSeriesError):
            verify_tropical_series_structure_audit(corrupt)
        corrupt_type = copy.deepcopy(self.audit)
        corrupt_type["claim_boundary"][
            "known_exact_support_tropical_cone_classified"
        ] = 1
        with self.assertRaises(KrennTropicalSeriesError):
            verify_tropical_series_structure_audit(corrupt_type)
        corrupt_number_type = copy.deepcopy(self.audit)
        corrupt_number_type["known_exact_support_moving_cone"][
            "relation_rank_over_Q"
        ] = 4.0
        with self.assertRaises(KrennTropicalSeriesError):
            verify_tropical_series_structure_audit(corrupt_number_type)

    def test_repository_audit_round_trips(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "krenn_quantum_graph"
            / "n6_d3_counterexample_search"
            / "tropical_series_structure.json"
        )
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            verify_tropical_series_structure_audit(stored), self.audit
        )


if __name__ == "__main__":
    unittest.main()
