import copy
import json
import unittest
from pathlib import Path

from experiments.krenn_quantum_graph.star_pivot_affine_slices import (
    star_pivot_affine_presentation,
)
from experiments.krenn_quantum_graph.star_pivot_compactification import (
    COX_VARIABLES,
    H_BY_COLOR,
    H_U,
    KrennStarPivotCompactificationError,
    boundary_orbits,
    star_only_boundary_orbits,
    star_pivot_block_presentation,
    star_pivot_compactification_audit,
    verify_star_pivot_compactification_audit,
)


class TestKrennStarPivotCompactification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = star_pivot_compactification_audit()

    def test_block_homogenization_round_trips_all_three_slices(self):
        for orbit in range(3):
            projective = star_pivot_block_presentation(orbit)
            affine = star_pivot_affine_presentation(
                orbit, "retained"
            )
            self.assertEqual(COX_VARIABLES, 139)
            self.assertEqual(len(projective.generators), 732)
            self.assertEqual(projective.term_count, 10_950)
            self.assertEqual(projective.maximum_ordinary_degree, 3)
            self.assertEqual(
                tuple(
                    generator.dehomogenized()
                    for generator in projective.generators
                ),
                affine.generators,
            )

    def test_exact_multidegree_histogram(self):
        expected = {
            (2, 0, 0, 0): 3,
            (2, 1, 0, 0): 243,
            (2, 0, 1, 0): 243,
            (2, 0, 0, 1): 243,
        }
        for orbit in range(3):
            self.assertEqual(
                star_pivot_block_presentation(
                    orbit
                ).multidegree_histogram,
                expected,
            )

    def test_boundary_specializations_have_exact_term_counts(self):
        for orbit in range(3):
            presentation = star_pivot_block_presentation(orbit)
            h_u_zero = sum(
                generator.term_count_after_zeroing((H_U,))
                for generator in presentation.generators
            )
            self.assertEqual(h_u_zero, 10_944)
            for color, homogenizer in enumerate(H_BY_COLOR):
                one_star_zero = sum(
                    generator.term_count_after_zeroing((homogenizer,))
                    for generator in presentation.generators
                )
                self.assertEqual(one_star_zero, 10_949)
                target_generator = (0, 364, 728)[color]
                self.assertEqual(
                    presentation.generators[
                        target_generator
                    ].term_count_after_zeroing((homogenizer,)),
                    15,
                )

    def test_boundary_orbits_respect_actual_pivot_stabilizers(self):
        self.assertEqual(
            [len(boundary_orbits(orbit)) for orbit in range(3)],
            [7, 11, 7],
        )
        self.assertEqual(
            [len(star_only_boundary_orbits(orbit)) for orbit in range(3)],
            [3, 5, 3],
        )
        for orbit in range(3):
            self.assertEqual(
                sum(
                    row["orbit_size"]
                    for row in boundary_orbits(orbit)
                ),
                15,
            )
            self.assertEqual(
                sum(
                    row["orbit_size"]
                    for row in star_only_boundary_orbits(orbit)
                ),
                7,
            )

    def test_audit_keeps_finite_and_boundary_claims_separate(self):
        decision = self.audit["decisive_exact_test"]
        claims = self.audit["claim_boundary"]
        cover = self.audit["exact_cover"]
        grading = self.audit["grading_and_symmetry"]
        self.assertTrue(
            cover["includes_zero_coordinate_and_support_strata"]
        )
        self.assertFalse(
            cover["requires_individual_weight_nonzero"]
        )
        self.assertFalse(
            decision["boundary_point_alone_is_a_decision"]
        )
        self.assertTrue(
            decision[
                "finite_open_H_nonzero_avoids_all_Cox_irrelevant_blocks"
            ]
        )
        self.assertEqual(grading["residual_character_rank"], 12)
        self.assertFalse(
            grading["rank_9_seed_chart_grading_is_the_same_grading"]
        )
        self.assertTrue(
            claims["Cox_homogeneous_presentation_constructed_exactly"]
        )
        self.assertFalse(
            claims["scheme_theoretic_projective_closure_computed"]
        )
        self.assertFalse(claims["compactification_constructed_exactly"])
        self.assertFalse(claims["Cox_irrelevant_saturation_computed"])
        self.assertFalse(claims["saturation_computed"])
        self.assertFalse(claims["finite_counterexample_found"])
        self.assertFalse(claims["global_nonexistence_proved"])

    def test_recommended_probe_is_exact_but_unrun(self):
        probe = self.audit["recommended_first_exact_probe"]
        self.assertEqual(
            probe["formulation"],
            "affine-U/projective-star retained slice",
        )
        self.assertIn(
            "Sat_B_Y0", probe["boundary_preflight_before_full_colon"]
        )
        self.assertIn(
            "Y0_j != 0", probe["boundary_preflight_before_full_colon"]
        )
        self.assertTrue(
            probe["boundary_Cox_irrelevant_saturation_required"]
        )
        self.assertFalse(
            probe[
                "raw_boundary_colon_without_irrelevant_saturation_is_decisive"
            ]
        )
        self.assertIn("K3=K2:h_2^infinity", probe["sequential_exact_test"])
        self.assertIn(
            "no finite witness", probe["conclusion_if_K3_is_unit"]
        )
        self.assertFalse(
            self.audit["claim_boundary"]["saturation_computed"]
        )

    def test_audit_is_strict_json_and_round_trips(self):
        encoded = json.dumps(self.audit, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(
            verify_star_pivot_compactification_audit(decoded),
            decoded,
        )

    def test_corruption_is_rejected(self):
        corrupted = copy.deepcopy(self.audit)
        corrupted["claim_boundary"]["global_nonexistence_proved"] = True
        with self.assertRaises(
            KrennStarPivotCompactificationError
        ):
            verify_star_pivot_compactification_audit(corrupted)
        corrupted_boundary = copy.deepcopy(self.audit)
        corrupted_boundary["recommended_first_exact_probe"][
            "raw_boundary_colon_without_irrelevant_saturation_is_decisive"
        ] = True
        with self.assertRaises(
            KrennStarPivotCompactificationError
        ):
            verify_star_pivot_compactification_audit(corrupted_boundary)

    def test_repository_audit_round_trips(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "krenn_quantum_graph"
            / "n6_d3_counterexample_search"
            / "star_pivot_compactification.json"
        )
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            verify_star_pivot_compactification_audit(stored), self.audit
        )


if __name__ == "__main__":
    unittest.main()
