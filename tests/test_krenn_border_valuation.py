import json
import unittest

from experiments.krenn_quantum_graph.border_valuation import (
    ACTIVE_COORDINATES,
    BORDER_VALUATION_SCHEMA,
    INVARIANT_MONOMIAL_Q_COORDINATES,
    certify_natural_border_valuation_obstruction,
    integer_farkas_vector,
    invariant_valuation_basis,
    natural_active_valuations,
    vertex_gauge_valuation_matrix,
)


class KrennBorderValuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = (
            certify_natural_border_valuation_obstruction()
        )

    def test_primal_valuation_problem_matches_the_laurent_family(self):
        certificate = self.certificate
        self.assertEqual(certificate.schema, BORDER_VALUATION_SCHEMA)
        self.assertEqual(certificate.active_coordinates, ACTIVE_COORDINATES)
        self.assertEqual(
            natural_active_valuations(),
            (1, -1, 0, 0, 0, 0, 0, 0, 0),
        )
        gauge = vertex_gauge_valuation_matrix()
        self.assertEqual(len(gauge), 9)
        self.assertTrue(all(len(row) == 5 for row in gauge))
        primal = certificate.to_dict()[
            "primal_integer_linear_program"
        ]
        self.assertFalse(primal["feasible_over_Z"])
        self.assertFalse(primal["feasible_over_Q"])
        self.assertEqual(len(primal["constraints"]), 9)

    def test_integer_farkas_vector_is_exact_contradiction(self):
        certificate = self.certificate
        farkas = integer_farkas_vector()
        self.assertEqual(
            farkas, (0, 1, 1, 1, 1, 0, 1, 1, 0)
        )
        self.assertTrue(all(value >= 0 for value in farkas))
        dual = certificate.to_dict()["integer_Farkas_certificate"]
        self.assertEqual(dual["dual_vector"], list(farkas))
        self.assertEqual(
            dual["dual_times_gauge_matrix"], [0, 0, 0, 0, 0]
        )
        self.assertEqual(dual["dual_times_initial_valuation"], -1)
        self.assertEqual(dual["summed_inequality"], "-1 >= 0")
        self.assertTrue(dual["contradiction"])
        self.assertEqual(
            certificate.exact_checks[
                "integer_Farkas_vertex_degrees_are_all_2"
            ],
            True,
        )

    def test_four_invariants_are_a_complete_gauge_quotient_basis(self):
        certificate = self.certificate
        invariants = invariant_valuation_basis()
        self.assertEqual(
            invariants,
            (
                (1, 1, 1, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 1, 1, 1, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 1, 1, 1),
                (0, 1, 1, 1, 1, 0, 1, 1, 0),
            ),
        )
        quotient = certificate.to_dict()[
            "complete_invariant_valuation_basis"
        ]
        self.assertEqual(quotient["ambient_dimension"], 9)
        self.assertEqual(quotient["gauge_orbit_dimension"], 5)
        self.assertEqual(quotient["quotient_dimension"], 4)
        self.assertEqual(quotient["rank_over_Q"], 4)
        self.assertEqual(quotient["values_on_path"], [0, 0, 0, -1])
        self.assertTrue(
            quotient["basis_annihilates_gauge_matrix"]
        )
        checks = certificate.exact_checks
        self.assertTrue(checks["sum_zero_vertex_gauge_rank_5_over_Q"])
        self.assertTrue(checks["invariant_basis_rank_4_over_Q"])
        self.assertTrue(
            checks["four_invariants_complete_the_9_minus_5_quotient"]
        )

    def test_invariant_monomial_Q_has_a_genuine_gauge_invariant_pole(self):
        q = self.certificate.to_dict()["invariant_monomial_Q"]
        self.assertEqual(
            q["coordinates"],
            [list(coordinate) for coordinate in INVARIANT_MONOMIAL_Q_COORDINATES],
        )
        self.assertEqual(q["vertex_degrees"], [2, 2, 2, 2, 2, 2])
        self.assertEqual(
            q["gauge_multiplier"], "(product_i lambda_i)^2 = 1"
        )
        self.assertEqual(q["value_on_path"], "t^-1")
        self.assertEqual(q["valuation_on_path"], -1)
        self.assertTrue(
            q[
                "regular_coordinates_would_force_nonnegative_valuation"
            ]
        )
        self.assertTrue(
            self.certificate.exact_checks[
                "invariant_Q_support_equals_Farkas_support"
            ]
        )

    def test_puiseux_and_positive_reparameterizations_remain_obstructed(self):
        scope = self.certificate.to_dict()["valuation_scope"]
        self.assertTrue(scope["Laurent_monomial_gauges_Z_valued"])
        self.assertTrue(scope["Puiseux_gauges_Q_valued"])
        self.assertTrue(
            scope["proof_uses_only_additivity_and_order_of_valuation"]
        )
        self.assertIn("-k", scope["ramified_reparameterization"])
        self.assertTrue(
            scope["positive_reparameterizations_remain_obstructed"]
        )

    def test_claim_boundary_is_specific_and_fail_closed(self):
        certificate = self.certificate
        boundary = certificate.to_dict()["claim_boundary"]
        self.assertTrue(boundary["specific_nine_support_path_certified"])
        self.assertTrue(
            boundary["full_product_one_vertex_scalar_gauge_certified"]
        )
        self.assertFalse(boundary["pole_removable_by_that_gauge"])
        for key in (
            "other_supports_or_paths_excluded",
            "support_changing_degenerations_excluded",
            "other_gauge_groups_excluded",
            "finite_exact_GHZ_witness_excluded",
            "affine_nonimage_proved",
        ):
            self.assertFalse(boundary[key])
        self.assertFalse(certificate.affine_nonimage_proved)
        self.assertTrue(
            certificate.this_family_vertex_gauge_regularization_impossible
        )
        self.assertFalse(
            boundary["border_membership_proved_by_this_obstruction"]
        )
        self.assertTrue(
            boundary["consistent_with_the_Laurent_border_certificate"]
        )
        self.assertTrue(certificate.exact)

    def test_certificate_is_deterministic_and_json_ready(self):
        replayed = certify_natural_border_valuation_obstruction()
        self.assertEqual(replayed, self.certificate)
        json.dumps(replayed.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
