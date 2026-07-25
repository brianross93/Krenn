import copy
import unittest

from experiments.krenn_quantum_graph.star_pivot_charts import (
    star_pivot_charts,
)
from experiments.krenn_quantum_graph.star_pivot_gauge import (
    KrennStarPivotGaugeError,
    STAR_PIVOT_GAUGE_AUDIT_SCHEMA,
    star_pivot_gauge_audit,
    verify_star_pivot_gauge_audit,
)
from experiments.krenn_quantum_graph.system import (
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.vertical_component import (
    color_diagonal_exponent_matrix,
    monomial_character,
)


class TestKrennStarPivotGauge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = star_pivot_gauge_audit()

    def test_every_term_of_every_pivot_factor_is_a_semi_invariant(self):
        self.assertEqual(
            self.audit["schema"],
            STAR_PIVOT_GAUGE_AUDIT_SCHEMA,
        )
        self.assertEqual(len(self.audit["apex_audits"]), 6)
        for apex_row in self.audit["apex_audits"]:
            census = apex_row[
                "full_pivot_cover_semi_invariant_census"
            ]
            self.assertEqual(census["pivot_charts_checked"], 125)
            self.assertEqual(
                census["pure_residual_factors_checked"], 375
            )
            self.assertEqual(
                census["quadratic_matching_terms_checked"], 1_125
            )
            self.assertTrue(census["every_term_replayed_exactly"])

        chart = next(
            chart
            for chart in star_pivot_charts(0)
            if chart.partner_by_color == (1, 2, 3)
        )
        exponent_matrix = color_diagonal_exponent_matrix()
        for color, (partner, polynomial) in enumerate(
            zip(
                chart.partner_by_color,
                chart.quadratic_factors,
                strict=True,
            )
        ):
            incident = exponent_matrix[
                variable_index(6, 3, 0, partner, color, color)
            ]
            expected = tuple(-value for value in incident)
            self.assertEqual(len(polynomial), 3)
            self.assertEqual(
                {monomial_character(term) for term in polynomial},
                {expected},
            )

    def test_three_factor_characters_are_primitive_rank_three(self):
        expected_patterns = {
            "all-same",
            "exactly-two-same",
            "all-distinct",
        }
        for apex_row in self.audit["apex_audits"]:
            representatives = apex_row[
                "transported_orbit_representatives"
            ]
            self.assertEqual(len(representatives), 3)
            self.assertEqual(
                {
                    row["equality_pattern"]
                    for row in representatives
                },
                expected_patterns,
            )
            for row in representatives:
                self.assertEqual(
                    row["factor_character_rank_over_Q"], 3
                )
                self.assertEqual(
                    row["factor_character_smith_invariant_factors"],
                    [1, 1, 1],
                )
                self.assertEqual(
                    abs(
                        row[
                            "factor_character_unimodular_minor_determinant"
                        ]
                    ),
                    1,
                )

    def test_explicit_normalization_has_four_choices_and_no_roots(self):
        expected_action = [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
        for apex_row in self.audit["apex_audits"]:
            for row in apex_row[
                "transported_orbit_representatives"
            ]:
                normalization = row["normalization"]
                self.assertEqual(
                    normalization["simultaneous_choice_count"], 64
                )
                self.assertEqual(
                    normalization[
                        "deterministic_action_matrix_on_selected_factors"
                    ],
                    expected_action,
                )
                self.assertEqual(
                    normalization["action_determinant"], -1
                )
                self.assertTrue(
                    normalization["global_product_one_preserved"]
                )
                self.assertTrue(
                    normalization["direct_GHZ_target_preserved"]
                )
                self.assertFalse(
                    normalization["root_extraction_required"]
                )
                for choice in normalization["choices_by_color"]:
                    self.assertEqual(
                        choice["eligible_choice_count"], 4
                    )
                    self.assertEqual(
                        len(choice["eligible_residual_vertices"]), 4
                    )
                    self.assertEqual(
                        choice[
                            "selected_factor_character_pairings"
                        ],
                        [-1, -1, -1, -1],
                    )

    def test_unit_factor_slice_eliminates_nine_star_weights(self):
        for apex_row in self.audit["apex_audits"]:
            apex = apex_row["apex"]
            for row in apex_row[
                "transported_orbit_representatives"
            ]:
                elimination = row["star_elimination"]
                self.assertEqual(
                    elimination[
                        "star_weights_eliminable_after_factor_normalization"
                    ],
                    9,
                )
                self.assertEqual(
                    len(set(elimination["star_weight_indices"])), 9
                )
                self.assertTrue(
                    all(
                        apex
                        in variable_key(6, 3, index)[:2]
                        for index in elimination[
                            "star_weight_indices"
                        ]
                    )
                )
                self.assertFalse(
                    elimination[
                        "polynomial_substitution_requires_division"
                    ]
                )

    def test_common_complement_avoids_the_entire_apex_star(self):
        expected_smith = [1] * 14 + [2]
        for apex_row in self.audit["apex_audits"]:
            apex = apex_row["apex"]
            representatives = apex_row[
                "transported_orbit_representatives"
            ]
            complements = [
                row["nonstar_complement"]
                for row in representatives
            ]
            self.assertEqual(
                len({
                    tuple(complement["weight_indices"])
                    for complement in complements
                }),
                1,
            )
            for complement in complements:
                self.assertEqual(
                    complement["combined_rank_over_Q"], 15
                )
                self.assertEqual(
                    complement["oriented_determinant"], 2
                )
                self.assertEqual(
                    complement["smith_invariant_factors"],
                    expected_smith,
                )
                self.assertEqual(
                    complement[
                        "factor_plus_all_nonstar_rank_mod_2"
                    ],
                    14,
                )
                self.assertTrue(
                    complement["lattice_index_two_is_minimal"]
                )
                self.assertTrue(
                    all(
                        apex not in variable_key(6, 3, index)[:2]
                        for index in complement["weight_indices"]
                    )
                )

    def test_claim_boundary_separates_global_slice_from_anchors(self):
        equivalence = self.audit["existence_equivalence"]
        self.assertTrue(
            equivalence[
                "pivot_open_nonempty_iff_three_factor_unit_slice_nonempty"
            ]
        )
        self.assertFalse(
            equivalence["normalization_requires_saturation"]
        )
        self.assertTrue(
            equivalence[
                "pivot_star_elimination_is_polynomial_on_unit_slice"
            ]
        )
        boundary = self.audit["claim_boundary"]
        self.assertTrue(
            boundary[
                "three_factor_unit_slice_covers_every_pivot_open_orbit"
            ]
        )
        self.assertFalse(
            boundary[
                "twelve_coordinate_anchors_are_guaranteed_nonzero"
            ]
        )
        self.assertFalse(
            boundary[
                "fifteen_anchor_numerical_chart_covers_entire_pivot_open"
            ]
        )
        self.assertFalse(boundary["pivot_chart_solved"])
        self.assertFalse(boundary["finite_counterexample_found"])
        self.assertFalse(boundary["global_affine_membership_decided"])

    def test_verifier_rejects_corruption_and_promoted_claims(self):
        self.assertEqual(
            verify_star_pivot_gauge_audit(self.audit),
            self.audit,
        )
        corrupted = copy.deepcopy(self.audit)
        corrupted["apex_audits"][0][
            "transported_orbit_representatives"
        ][0]["nonstar_complement"]["oriented_determinant"] = 1
        with self.assertRaises(KrennStarPivotGaugeError):
            verify_star_pivot_gauge_audit(corrupted)

        promoted = copy.deepcopy(self.audit)
        promoted["claim_boundary"]["finite_counterexample_found"] = True
        with self.assertRaises(KrennStarPivotGaugeError):
            verify_star_pivot_gauge_audit(promoted)


if __name__ == "__main__":
    unittest.main()
