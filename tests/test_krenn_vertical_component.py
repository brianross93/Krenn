import json
import os
import unittest

from experiments.krenn_quantum_graph.vertical_component import (
    EXPECTED_FIXED_VICTIM_ORBIT_HISTOGRAM,
    EXPECTED_FIXED_VICTIM_SEED_ORBITS,
    EXPECTED_FIXED_VICTIM_SURVIVOR_ORBIT,
    EXPECTED_SMALL_SUPPORT_CASES,
    EXPECTED_SUPPORT21_CLOSURE_NODES,
    NATURAL_SEED,
    VICTIM_COLORING,
    bounded_singleton_closure,
    certify_unconditional_support_closure,
    color_diagonal_exponent_matrix,
    exact_vertical_component_structure,
    fixed_victim_nine_coordinate_audit,
    fixed_victim_seed_orbits,
    natural_color_gauge_orbit_audit,
    recorded_support21_global_audit,
    replay_support21_global_closure,
    small_support_direct_census,
    support21_terminal_certificates,
)


class KrennVerticalComponentTest(unittest.TestCase):
    def test_known_laurent_path_is_exact_color_gauge_orbit(self):
        audit = natural_color_gauge_orbit_audit()
        self.assertEqual(
            len(color_diagonal_exponent_matrix()), 135
        )
        self.assertTrue(
            all(len(row) == 15 for row in color_diagonal_exponent_matrix())
        )
        self.assertEqual(
            audit["interpretation"],
            "the known Laurent path is a one-parameter "
            "color-diagonal orbit in the GHZ-plus-victim moving-line "
            "incidence",
        )
        self.assertEqual(
            audit["effective_gauge_rank_on_natural_support"], 6
        )
        self.assertEqual(audit["victim_character_on_gamma"], 1)
        self.assertEqual(audit["Q_character_on_gamma"], -1)
        self.assertTrue(all(audit["exact_checks"].values()))
        self.assertFalse(
            audit["claim_boundary"][
                "Q_is_invariant_under_full_direct_GHZ_color_gauge"
            ]
        )

    def test_fixed_victim_has_104_seed_chart_orbits(self):
        rows = fixed_victim_seed_orbits()
        self.assertEqual(len(rows), EXPECTED_FIXED_VICTIM_SEED_ORBITS)
        self.assertEqual(
            sum(len(orbit) for _representative, orbit in rows), 3_375
        )
        histogram = {}
        for _representative, orbit in rows:
            histogram[len(orbit)] = histogram.get(len(orbit), 0) + 1
        self.assertEqual(histogram, EXPECTED_FIXED_VICTIM_ORBIT_HISTOGRAM)

    def test_only_natural_nine_coordinate_orbit_survives(self):
        audit = fixed_victim_nine_coordinate_audit()
        self.assertEqual(audit["victim_coloring"], list(VICTIM_COLORING))
        self.assertEqual(audit["seed_orbits"], 104)
        self.assertEqual(
            audit["orbits_killed_by_nonvictim_singleton"], 103
        )
        self.assertEqual(
            audit["surviving_orbit"]["representative"],
            list(NATURAL_SEED),
        )
        self.assertEqual(
            tuple(
                tuple(seed)
                for seed in audit["surviving_orbit"]["members"]
            ),
            EXPECTED_FIXED_VICTIM_SURVIVOR_ORBIT,
        )
        self.assertEqual(
            audit["natural_restricted_stratum"]["dimension"], 6
        )
        self.assertTrue(
            audit["natural_restricted_stratum"]["victim_is_unit"]
        )
        self.assertTrue(all(audit["exact_checks"].values()))
        self.assertFalse(
            audit["claim_boundary"]["all_vertical_components_excluded"]
        )

    def test_direct_small_support_census_is_unconditional(self):
        census = small_support_direct_census()
        self.assertEqual(
            tuple(census["cases_examined"]),
            EXPECTED_SMALL_SUPPORT_CASES,
        )
        self.assertEqual(census["singleton_free_cases"], [0, 0, 0])
        self.assertEqual(
            census["finite_exact_support_lower_bound"], 12
        )
        self.assertFalse(
            census["claim_boundary"]["natural_coordinate_retention_assumed"]
        )

    def test_routine_eight_root_closure_replays_support_11_bound(self):
        closure = certify_unconditional_support_closure(
            max_total_support=11
        )
        self.assertTrue(closure.exhaustive)
        self.assertEqual(closure.singleton_free_support_count, 0)
        self.assertEqual(closure.finite_support_lower_bound, 12)
        self.assertEqual(len(closure.roots), 8)
        self.assertTrue(
            all(root.excludes_all_supports_through_cap
                for root in closure.roots)
        )
        payload = closure.to_dict()
        self.assertFalse(
            payload["claim_boundary"][
                "natural_coordinate_retention_assumed"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"]["symmetry_related_weights_equated"]
        )

    def test_bounded_closure_fails_closed_on_node_cap(self):
        partial = bounded_singleton_closure(
            NATURAL_SEED,
            max_total_support=19,
            node_cap=5,
        )
        self.assertFalse(partial.exhaustive)
        self.assertFalse(partial.excludes_all_supports_through_cap)

    def test_support_21_terminals_all_have_exact_algebraic_obstructions(self):
        certificates = support21_terminal_certificates()
        self.assertEqual(len(certificates), 12)
        self.assertEqual(
            sum(certificate.kind == "odd-mixed-circuit"
                for certificate in certificates),
            8,
        )
        self.assertEqual(
            sum(certificate.kind == "pure-cancellation"
                for certificate in certificates),
            4,
        )
        self.assertTrue(
            all(
                all(certificate.exact_checks.values())
                for certificate in certificates
            )
        )
        audit = recorded_support21_global_audit()
        self.assertEqual(audit["totals"]["nodes_examined"], 12_992_269)
        self.assertEqual(
            audit["finite_exact_support_lower_bound"], 22
        )
        self.assertFalse(
            audit["claim_boundary"][
                "natural_coordinate_retention_assumed"
            ]
        )

    @unittest.skipUnless(
        os.environ.get("KRENN_RUN_LONG_TESTS") == "1",
        "set KRENN_RUN_LONG_TESTS=1 for the support-21 closure",
    )
    def test_long_unconditional_support_22_certificate(self):
        closure = replay_support21_global_closure()
        self.assertEqual(
            tuple(root.nodes_examined for root in closure.roots),
            EXPECTED_SUPPORT21_CLOSURE_NODES,
        )
        self.assertEqual(closure.singleton_free_support_count, 12)
        self.assertIsNone(closure.finite_support_lower_bound)

    def test_composite_payload_is_round_trippable_and_fail_closed(self):
        payload = exact_vertical_component_structure(support_cap=11)
        self.assertEqual(
            json.loads(json.dumps(payload, allow_nan=False)), payload
        )
        claims = payload["claims"]
        self.assertTrue(
            claims[
                "known_Laurent_path_is_color_diagonal_orbit_in_moving_line_incidence"
            ]
        )
        self.assertTrue(
            claims[
                "known_natural_nine_coordinate_stratum_has_no_vertical_point"
            ]
        )
        self.assertEqual(
            claims["finite_exact_support_lower_bound"], 22
        )
        self.assertEqual(
            claims["routine_replay_support_lower_bound"], 12
        )
        self.assertFalse(claims["all_vertical_components_excluded"])
        self.assertFalse(claims["D_in_radical_J_mix_decided"])
        self.assertFalse(
            claims["finite_affine_GHZ_membership_decided"]
        )
        self.assertEqual(
            claims["exact_affine_GHZ_membership_status"], "undecided"
        )


if __name__ == "__main__":
    unittest.main()
