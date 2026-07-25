import copy
from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.star_linearization import (
    star_linearization,
)
from experiments.krenn_quantum_graph.star_pivot_affine_slices import (
    KrennStarPivotAffineSliceError,
    PIVOT_ORBITS,
    reconstruct_nine_star_weights,
    singular_star_pivot_affine_slice_script,
    star_pivot_affine_presentation,
    star_pivot_affine_slice_audit,
    verify_star_pivot_affine_slice_audit,
)
from experiments.krenn_quantum_graph.system import generate_sparse_system


EXPECTED_ELIMINATED_HISTOGRAMS = (
    {
        "maximum_degree": {2: 3, 5: 720},
        "term_count": {4: 3, 48: 480, 51: 240},
        "degree_sets": {"0,2": 3, "2,3,5": 240, "3,5": 480},
    },
    {
        "maximum_degree": {2: 3, 3: 162, 5: 558},
        "term_count": {
            4: 3,
            15: 162,
            48: 264,
            51: 132,
            81: 54,
            84: 108,
        },
        "degree_sets": {
            "0,2": 3,
            "2,3,5": 240,
            "3": 162,
            "3,5": 318,
        },
    },
    {
        "maximum_degree": {2: 3, 3: 216, 5: 504},
        "term_count": {
            4: 3,
            15: 216,
            48: 210,
            51: 105,
            81: 54,
            84: 108,
            117: 27,
        },
        "degree_sets": {
            "0,2": 3,
            "2,3,5": 240,
            "3": 216,
            "3,5": 264,
        },
    },
)


def _system_residuals(values):
    system = generate_sparse_system(6, 3)
    residuals = []
    for equation in range(system.equation_count):
        total = Fraction(0)
        for monomial in system.equation_monomials(equation):
            term = Fraction(1)
            for variable in monomial:
                term *= values[variable]
            total += term
        residuals.append(total - system.rhs_values[equation])
    return tuple(residuals)


class KrennStarPivotAffineSlicesTest(unittest.TestCase):
    def test_three_orbits_are_a_complete_125_chart_partition(self):
        self.assertEqual(
            [
                (
                    orbit.equality_pattern,
                    orbit.partner_by_color,
                    orbit.orbit_size,
                    orbit.stabilizer_order,
                )
                for orbit in PIVOT_ORBITS
            ],
            [
                ("all-same", (1, 1, 1), 5, 144),
                ("exactly-two-same", (1, 1, 2), 60, 12),
                ("all-distinct", (1, 2, 3), 60, 12),
            ],
        )
        self.assertEqual(sum(orbit.orbit_size for orbit in PIVOT_ORBITS), 125)
        self.assertTrue(all(
            orbit.orbit_size * orbit.stabilizer_order == 720
            for orbit in PIVOT_ORBITS
        ))

    def test_retained_presentations_have_exact_cubic_census(self):
        for orbit in PIVOT_ORBITS:
            presentation = star_pivot_affine_presentation(
                orbit.index, "retained"
            )
            self.assertEqual(len(presentation.variable_indices), 135)
            self.assertEqual(len(presentation.generators), 732)
            self.assertEqual(presentation.maximum_degree, 3)
            self.assertEqual(presentation.term_count, 10_950)
            self.assertEqual(
                presentation.degree_term_census,
                {0: 6, 2: 9, 3: 10_935},
            )
            self.assertEqual(
                presentation.maximum_degree_histogram,
                {2: 3, 3: 729},
            )
            self.assertEqual(
                presentation.generator_term_count_histogram,
                {4: 3, 15: 726, 16: 3},
            )
            self.assertEqual(
                presentation.eliminated_variable_indices, ()
            )
            self.assertEqual(
                presentation.variable_indices, tuple(range(135))
            )

    def test_nine_weight_elimination_has_exact_orbit_censuses(self):
        for orbit, expected in zip(
            PIVOT_ORBITS,
            EXPECTED_ELIMINATED_HISTOGRAMS,
            strict=True,
        ):
            presentation = star_pivot_affine_presentation(
                orbit.index, "eliminated"
            )
            self.assertEqual(len(presentation.variable_indices), 126)
            self.assertEqual(
                len(presentation.eliminated_variable_indices), 9
            )
            self.assertEqual(len(presentation.generators), 723)
            self.assertEqual(presentation.maximum_degree, 5)
            self.assertEqual(presentation.term_count, 35_292)
            self.assertEqual(
                presentation.degree_term_census,
                {0: 3, 2: 729, 3: 8_640, 5: 25_920},
            )
            self.assertEqual(
                presentation.maximum_degree_histogram,
                expected["maximum_degree"],
            )
            self.assertEqual(
                presentation.generator_term_count_histogram,
                expected["term_count"],
            )
            self.assertEqual(
                presentation.degree_set_histogram,
                expected["degree_sets"],
            )
            self.assertEqual(
                set(presentation.variable_indices).intersection(
                    presentation.eliminated_variable_indices
                ),
                set(),
            )
            self.assertEqual(
                set(presentation.variable_indices).union(
                    presentation.eliminated_variable_indices
                ),
                set(range(135)),
            )

    def test_exact_reconstruction_replays_all_729_original_equations(self):
        system = generate_sparse_system(6, 3)
        factorization = star_linearization(0)
        restricted = {
            apex_color * 243 + target_row
            for apex_color in range(3)
            for target_row in (0, 121, 242)
        }
        for orbit in PIVOT_ORBITS:
            presentation = star_pivot_affine_presentation(
                orbit.index, "eliminated"
            )
            free_values = {
                variable: Fraction(
                    (variable * 17 + 11) % 19 - 9,
                    (variable * 7 + 5) % 13 + 1,
                )
                for variable in presentation.variable_indices
            }
            ambient = reconstruct_nine_star_weights(
                orbit.index, free_values
            )
            original_residuals = _system_residuals(ambient)
            eliminated_residuals = tuple(
                generator.evaluate(free_values)
                for generator in presentation.generators
            )
            self.assertEqual(
                eliminated_residuals[:720],
                tuple(
                    residual
                    for equation, residual in enumerate(
                        original_residuals
                    )
                    if equation not in restricted
                ),
            )

            normalization_residuals = eliminated_residuals[-3:]
            for apex_color in range(3):
                for residual_color, pivot_column in enumerate(
                    presentation.pivot_columns_by_color
                ):
                    pivot_variable = (
                        factorization.star_variable_blocks[
                            apex_color
                        ][pivot_column]
                    )
                    equation = (
                        apex_color * 243
                        + (0, 121, 242)[residual_color]
                    )
                    self.assertEqual(
                        original_residuals[equation],
                        ambient[pivot_variable]
                        * normalization_residuals[residual_color],
                    )
            self.assertEqual(len(system.rhs_values), 729)

    def test_independent_subset_filter_replay_and_claim_boundaries(self):
        audit = star_pivot_affine_slice_audit()
        self.assertEqual(audit["cover"]["open_triples_before_symmetry"], 125)
        self.assertEqual(
            audit["gauge_normalization"][
                "residual_vertex_parameters_that_change_one_factor"
            ],
            4,
        )
        self.assertFalse(
            audit["gauge_normalization"][
                "requires_Rabinowitsch_variable"
            ]
        )
        self.assertFalse(
            audit["gauge_normalization"][
                "requires_colon_or_saturation"
            ]
        )
        self.assertEqual(len(audit["independent_replay"]), 3)
        for row in audit["independent_replay"]:
            self.assertTrue(all(row["checks"].values()))
        self.assertFalse(
            audit["claim_boundary"]["any_affine_slice_solved"]
        )
        self.assertFalse(
            audit["claim_boundary"]["finite_counterexample_found"]
        )
        self.assertFalse(
            audit["claim_boundary"]["nonexistence_certificate_found"]
        )

    def test_audit_hashes_are_deterministic_and_corruption_fails(self):
        audit = star_pivot_affine_slice_audit()
        self.assertEqual(
            verify_star_pivot_affine_slice_audit(audit), audit
        )
        self.assertEqual(len(audit["sha256"]), 64)
        for row in audit["presentations"]:
            self.assertEqual(len(row["retained"]["sha256"]), 64)
            self.assertEqual(len(row["eliminated"]["sha256"]), 64)
            self.assertNotEqual(
                row["retained"]["sha256"],
                row["eliminated"]["sha256"],
            )

        corrupted = copy.deepcopy(audit)
        corrupted["presentations"][2]["eliminated"][
            "collected_term_count"
        ] -= 1
        with self.assertRaises(KrennStarPivotAffineSliceError):
            verify_star_pivot_affine_slice_audit(corrupted)
        escalated = copy.deepcopy(audit)
        escalated["claim_boundary"][
            "global_affine_membership_decided"
        ] = True
        with self.assertRaises(KrennStarPivotAffineSliceError):
            verify_star_pivot_affine_slice_audit(escalated)

    def test_singular_exports_are_deterministic_but_not_executed(self):
        retained = singular_star_pivot_affine_slice_script(
            1, "retained", characteristic=0, algorithm="slimgb"
        )
        self.assertEqual(
            retained,
            singular_star_pivot_affine_slice_script(
                1, "retained", characteristic=0, algorithm="slimgb"
            ),
        )
        self.assertIn("variables=135; generators=732; terms=10950", retained)
        self.assertIn("ring r=0,", retained)
        self.assertIn("ideal G=slimgb(I);", retained)
        self.assertIn(
            "No Rabinowitsch variable, localization, or saturation.",
            retained,
        )
        eliminated = singular_star_pivot_affine_slice_script(
            2, "eliminated", characteristic=31, algorithm="std"
        )
        self.assertIn(
            "variables=126; generators=723; terms=35292", eliminated
        )
        self.assertIn("ring r=31,", eliminated)
        self.assertIn("ideal G=std(I);", eliminated)

    def test_invalid_inputs_fail_closed(self):
        for orbit_index in (-1, 3, True, "not-an-orbit"):
            with self.assertRaises(KrennStarPivotAffineSliceError):
                star_pivot_affine_presentation(
                    orbit_index, "retained"
                )
        with self.assertRaises(KrennStarPivotAffineSliceError):
            star_pivot_affine_presentation(0, "unknown")
        for characteristic in (-1, 1, 4, True):
            with self.assertRaises(KrennStarPivotAffineSliceError):
                singular_star_pivot_affine_slice_script(
                    0, characteristic=characteristic
                )
        with self.assertRaises(KrennStarPivotAffineSliceError):
            singular_star_pivot_affine_slice_script(
                0, algorithm="mystery"
            )
        with self.assertRaises(KrennStarPivotAffineSliceError):
            reconstruct_nine_star_weights(0, [Fraction(0)] * 125)
        with self.assertRaises(KrennStarPivotAffineSliceError):
            reconstruct_nine_star_weights(0, {})


if __name__ == "__main__":
    unittest.main()
