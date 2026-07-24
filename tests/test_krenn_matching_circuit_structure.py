import unittest

from experiments.krenn_quantum_graph.matching_circuit_structure import (
    EXPECTED_Q_ORBIT,
    MATCHING_CIRCUIT_STRUCTURE_SCHEMA,
    VICTIM_COLORING,
    exact_matching_circuit_structure,
    matching_circuit_lattice_audit,
    naive_global_identity_audit,
    pole_invariant_orbit,
    victim_stabilizer_actions,
    victim_symmetry_pole_audit,
)


class VictimSymmetryPoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = victim_symmetry_pole_audit()

    def test_victim_stabilizer_and_q_orbit(self):
        self.assertEqual(len(victim_stabilizer_actions()), 48)
        self.assertEqual(
            tuple(pole_invariant_orbit()),
            EXPECTED_Q_ORBIT,
        )
        self.assertEqual(
            self.audit["victim_coloring"],
            list(VICTIM_COLORING),
        )
        self.assertEqual(self.audit["stabilizer_actions"], 48)
        self.assertEqual(len(self.audit["pole_monomials"]), 4)
        self.assertEqual(
            self.audit["pole_monomial_vertex_degrees"],
            [[2, 2, 2, 2, 2, 2]] * 4,
        )
        self.assertTrue(
            self.audit[
                "each_Q_is_product_one_vertex_gauge_invariant"
            ]
        )
        self.assertTrue(self.audit["stabilizer_preserves_S"])

    def test_all_literal_paths_replay_the_moving_target(self):
        self.assertEqual(self.audit["literal_natural_paths"], 24)
        self.assertEqual(self.audit["actions_per_literal_path"], 2)
        self.assertEqual(
            self.audit["literal_paths_per_active_Q"],
            {"0": 6, "1": 6, "2": 6, "3": 6},
        )
        self.assertEqual(
            len(self.audit["cross_evaluation_patterns"]),
            4,
        )
        self.assertTrue(
            self.audit[
                "all_729_outputs_equal_GHZ_plus_t_victim"
            ]
        )

    def test_symmetry_sum_has_exact_pole(self):
        self.assertEqual(
            self.audit["S_definition"],
            "Q0 + Q1 + Q2 + Q3",
        )
        self.assertEqual(
            self.audit["S_on_every_literal_path"],
            "t^-1",
        )
        self.assertEqual(
            self.audit["t_times_S_on_every_literal_path"],
            "1",
        )
        self.assertTrue(self.audit["local_branch_law_exact"])
        for pattern in self.audit["cross_evaluation_patterns"]:
            self.assertEqual(pattern.count("t^-1"), 1)
            self.assertEqual(pattern.count("0"), 3)


class MatchingCircuitLatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = matching_circuit_lattice_audit()

    def test_exact_collision_census(self):
        self.assertEqual(self.audit["perfect_matchings"], 15)
        self.assertEqual(
            self.audit["matching_incidence_shape"],
            [15, 15],
        )
        self.assertEqual(
            self.audit["matching_incidence_rank_over_Q"],
            10,
        )
        self.assertEqual(self.audit["integral_kernel_rank"], 5)
        self.assertEqual(self.audit["quadratic_multisets"], 120)
        self.assertEqual(
            self.audit["quadratic_collision_fibers"],
            0,
        )
        self.assertEqual(self.audit["cubic_multisets"], 680)
        self.assertEqual(self.audit["cubic_incidence_sums"], 670)
        self.assertEqual(
            self.audit["cubic_collision_fibers"],
            10,
        )

    def test_all_ten_relations_are_k33_parity_circuits(self):
        circuits = self.audit["circuits"]
        self.assertEqual(len(circuits), 10)
        self.assertEqual(
            {
                tuple(map(tuple, circuit["bipartition"]))
                for circuit in circuits
            },
            {
                (
                    tuple(first),
                    tuple(
                        vertex
                        for vertex in range(6)
                        if vertex not in first
                    ),
                )
                for first in (
                    (0, 1, 2),
                    (0, 1, 3),
                    (0, 1, 4),
                    (0, 1, 5),
                    (0, 2, 3),
                    (0, 2, 4),
                    (0, 2, 5),
                    (0, 3, 4),
                    (0, 3, 5),
                    (0, 4, 5),
                )
            },
        )
        for circuit in circuits:
            self.assertEqual(
                len(circuit["even_matching_indices"]),
                3,
            )
            self.assertEqual(
                len(circuit["odd_matching_indices"]),
                3,
            )
            self.assertEqual(
                circuit["relation_coefficients"].count(1),
                3,
            )
            self.assertEqual(
                circuit["relation_coefficients"].count(-1),
                3,
            )
            self.assertEqual(
                circuit["relation_coefficients"].count(0),
                9,
            )

    def test_unimodular_minor_certifies_integral_completeness(self):
        self.assertEqual(
            self.audit["circuit_relation_rank_over_Q"],
            5,
        )
        self.assertEqual(
            self.audit["unimodular_minor"]["circuit_rows"],
            [0, 1, 2, 4, 5],
        )
        self.assertEqual(
            self.audit["unimodular_minor"][
                "matching_columns"
            ],
            [0, 1, 3, 4, 7],
        )
        self.assertEqual(
            self.audit["unimodular_minor"]["determinant"],
            1,
        )
        self.assertTrue(
            self.audit[
                "integral_kernel_generated_by_K3_3_circuits"
            ]
        )
        self.assertTrue(
            self.audit[
                "localized_laurent_relation_lattice_complete"
            ]
        )
        self.assertFalse(
            self.audit["polynomial_toric_ideal_generation_claimed"]
        )


class NaiveGlobalIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = naive_global_identity_audit()

    def test_exact_spill_and_residual_census(self):
        self.assertEqual(self.audit["D_expansion_terms"], 3_375)
        self.assertEqual(
            self.audit["F_victim_times_S_raw_terms"],
            60,
        )
        self.assertEqual(
            self.audit["F_victim_times_S_distinct_terms"],
            60,
        )
        self.assertEqual(
            self.audit["terms_overlapping_D_support"],
            4,
        )
        self.assertEqual(
            self.audit["outside_D_support_spill_terms"],
            56,
        )
        self.assertEqual(self.audit["spill_full_group_orbits"], 4)
        self.assertEqual(
            len(
                self.audit[
                    "spill_full_group_orbit_representatives"
                ]
            ),
            4,
        )
        self.assertEqual(
            self.audit["residual_nonzero_terms"],
            3_427,
        )
        self.assertEqual(
            self.audit["residual_coefficient_census"],
            {"-1": 56, "1": 3_371},
        )

    def test_k1_dual_blocks_degree_nine_repair_only(self):
        self.assertFalse(
            self.audit["naive_global_polynomial_identity_holds"]
        )
        self.assertTrue(
            self.audit[
                "naive_global_polynomial_identity_refuted_exactly"
            ]
        )
        self.assertTrue(
            self.audit["F_victim_times_S_is_in_J_mix"]
        )
        self.assertEqual(
            self.audit["k1_exact_dual_lambda_transpose_b"],
            -720,
        )
        self.assertTrue(
            self.audit["D_not_in_J_mix_at_k1_replayed"]
        )
        self.assertFalse(
            self.audit["degree_six_mixed_multiplier_repair_exists"]
        )
        self.assertFalse(
            self.audit[
                "set_theoretic_or_higher_power_relation_decided"
            ]
        )
        self.assertFalse(
            self.audit["global_GHZ_nonexistence_proved"]
        )

    def test_composite_claims_remain_fail_closed(self):
        payload = exact_matching_circuit_structure()
        self.assertEqual(
            payload["schema"],
            MATCHING_CIRCUIT_STRUCTURE_SCHEMA,
        )
        claims = payload["claims"]
        self.assertTrue(
            claims[
                "natural_symmetry_branches_obey_t_times_S_equals_1"
            ]
        )
        self.assertTrue(
            claims[
                "fixed_coloring_localized_matching_lattice_complete"
            ]
        )
        self.assertFalse(
            claims["naive_degree_nine_global_identity_exists"]
        )
        self.assertFalse(
            claims["global_moving_fiber_pole_law_proved"]
        )
        self.assertFalse(claims["D_in_radical_J_mix_decided"])
        self.assertFalse(
            claims["exact_affine_GHZ_membership_decided"]
        )


if __name__ == "__main__":
    unittest.main()
