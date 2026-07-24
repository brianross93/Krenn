import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.defect_mining import (
    CERTIFICATE_FILE,
    GROEBNER_D3_FILE,
    GROEBNER_D4_FILE,
    KrennDefectMiningError,
    MANIFEST_FILE,
    REPORT_FILE,
    SOURCE_INPUTS,
    build_defect_report,
    verify_defect_mining_bundle,
    write_defect_mining_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class KrennDefectMiningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_defect_report()

    def test_victim_is_named_exactly_and_polynomial_has_15_terms(self):
        near_miss = self.report["natural_near_miss"]
        self.assertEqual(near_miss["satisfied_equations"], 728)
        self.assertEqual(near_miss["total_equations"], 729)
        victim = near_miss["victim"]
        self.assertEqual(victim["equation"], 70)
        self.assertEqual(victim["coloring"], [0, 0, 2, 1, 2, 1])
        self.assertEqual(victim["occupation"], [2, 2, 2])
        self.assertEqual(victim["residual"], 1)
        self.assertEqual(
            victim["surviving_terms"],
            [
                {
                    "matching_index": 1,
                    "value": 1,
                    "variables": [
                        [0, 1, 0, 0],
                        [2, 4, 2, 2],
                        [3, 5, 1, 1],
                    ],
                }
            ],
        )
        polynomial = near_miss["exact_residual_polynomial"]
        self.assertEqual(len(polynomial["terms"]), 15)
        self.assertIn(
            "W[0,1,0,0]*W[2,4,2,2]*W[3,5,1,1]",
            polynomial["expanded_expression"],
        )
        self.assertEqual(
            polynomial["surviving_matching_indices"], [1]
        )
        self.assertEqual(polynomial["target_rhs"], 0)

    def test_all_eight_orbits_have_exact_support_defects(self):
        rows = self.report["orbit_representatives"]
        self.assertEqual(
            [
                row["representative_matching_indices"]
                for row in rows
            ],
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 4],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 5],
                [0, 4, 8],
                [0, 4, 13],
            ],
        )
        self.assertEqual(
            [row["nonzero_residual_count"] for row in rows],
            [24, 12, 6, 6, 5, 2, 1, 3],
        )
        self.assertEqual(
            sum(row["ordered_seed_orbit_size"] for row in rows),
            3375,
        )
        for row in rows:
            with self.subTest(
                representative=row[
                    "representative_matching_indices"
                ]
            ):
                self.assertEqual(
                    row["nonzero_residual_count"],
                    row["defect_formula_2s_plus_r"],
                )
                self.assertEqual(
                    row["four_plus_two_defects"],
                    2 * row["pairwise_shared_edge_sum"],
                )
                self.assertTrue(
                    all(
                        len(defect["surviving_terms"]) == 1
                        for defect in row["defects"]
                    )
                )
        fixed = self.report[
            "fixed_coordinate_evaluations_on_eight_representatives"
        ]
        self.assertEqual(
            fixed["R_40_coloring_001111"],
            [1, 1, 1, 1, 1, 1, 0, 0],
        )
        self.assertEqual(
            fixed["R_70_coloring_002121"],
            [0, 0, 0, 0, 0, 0, 1, 1],
        )
        self.assertEqual(fixed["common_defect_equations"], [])
        self.assertFalse(fixed["same_literal_defect_on_all_orbits"])
        census = self.report["seed_orbit_census_replay"]
        self.assertEqual(census["counts"]["symmetry_orbits"], 8)
        self.assertTrue(all(census["exact_checks"].values()))

    def test_hafnian_shadow_localizes_but_is_dominant(self):
        shadow = self.report["hafnian_shadow"]
        self.assertEqual(
            shadow["all_ones_contraction"],
            {
                "hafnian_of_edge_sums": 4,
                "sum_of_all_729_outputs": 4,
                "sum_of_GHZ_target": 3,
                "residual": 1,
            },
        )
        self.assertEqual(
            shadow["natural_equal_g_residual_coefficients"],
            [{"occupation": [2, 2, 2], "coefficient": 1}],
        )
        self.assertEqual(
            shadow["n6_d3_shadow"]["jacobian_probe"]["rank"], 28
        )
        self.assertEqual(
            shadow["n6_d4_shadow"]["jacobian_probe"]["rank"], 84
        )
        star = shadow["n6_d3_exact_star_linear_slice"]
        self.assertEqual(star["pivot_determinant_over_Z"], 2**54)
        self.assertEqual(star["pivot_determinant_mod_31"], 16)
        self.assertTrue(star["target_generic_over_Q"])
        self.assertTrue(star["canonical_GHZ_shadow_solution_exact"])
        self.assertEqual(len(star["canonical_GHZ_shadow_solution"]), 90)
        cyclotomic = shadow["n6_d3_sparse_cyclotomic_shadow"]
        self.assertTrue(
            cyclotomic["all_28_shadow_equations_satisfied"]
        )
        self.assertTrue(
            cyclotomic["full_tensor_victim"]["nonzero"]
        )
        self.assertFalse(cyclotomic["full_tensor_solution_claimed"])
        self.assertFalse(shadow["groebner_backend_run"])
        self.assertIn("surjecting", shadow["interpretation"])

    def test_laurent_curve_certifies_border_not_exact_membership(self):
        border = self.report[
            "full_tensor_laurent_border_certificate"
        ]
        self.assertEqual(
            border["full_tensor_replay"]["identity"],
            "Phi(W(t)) = GHZ_6,3 + t*e_(0,0,2,1,2,1)",
        )
        self.assertEqual(
            border["full_tensor_replay"]["nonzero_coefficients"],
            [
                {
                    "equation": 0,
                    "coloring": [0, 0, 0, 0, 0, 0],
                    "coefficient": "1",
                },
                {
                    "equation": 70,
                    "coloring": [0, 0, 2, 1, 2, 1],
                    "coefficient": "t",
                },
                {
                    "equation": 364,
                    "coloring": [1, 1, 1, 1, 1, 1],
                    "coefficient": "1",
                },
                {
                    "equation": 728,
                    "coloring": [2, 2, 2, 2, 2, 2],
                    "coefficient": "1",
                },
            ],
        )
        claims = border["claim_boundary"]
        self.assertTrue(claims["border_image_membership_proved"])
        self.assertFalse(
            claims["exact_affine_image_membership_proved"]
        )
        self.assertEqual(
            claims["exact_affine_image_membership_status"],
            "undecided",
        )

    def test_n6_deformation_rank_sandwich_is_recorded_exactly(self):
        deformation = self.report[
            "n6_natural_deformation_certificate"
        ]
        full = deformation["full_jacobian"]
        self.assertEqual(full["shape"], [729, 135])
        self.assertEqual(full["rank_over_Q"], 130)
        self.assertEqual(full["nullity_over_Q"], 5)
        self.assertTrue(
            full["kernel_equals_vertex_scalar_gauge_over_Q"]
        )
        deleted = deformation["defect_deleted_jacobian"]
        self.assertEqual(deleted["shape"], [728, 135])
        self.assertEqual(deleted["rank_over_Q"], 129)
        self.assertEqual(deleted["nullity_over_Q"], 6)
        self.assertEqual(
            deformation["repair_direction"]["full_jacobian_image"],
            "-e_70",
        )
        role = deformation["modular_integer_minor_role"]
        self.assertFalse(role["finite_field_solution_claim"])
        self.assertFalse(
            role["finite_field_solution_to_C_transfer_used"]
        )

    def test_border_regularization_is_formal_but_not_finite(self):
        analysis = self.report["border_regularization_analysis"]
        valuation = analysis[
            "vertex_gauge_valuation_obstruction"
        ]
        dual = valuation["integer_Farkas_certificate"]
        self.assertEqual(
            dual["dual_vector"], [0, 1, 1, 1, 1, 0, 1, 1, 0]
        )
        self.assertEqual(dual["dual_times_initial_valuation"], -1)
        self.assertTrue(dual["contradiction"])
        self.assertFalse(
            valuation["claim_boundary"]["affine_nonimage_proved"]
        )

        formal = analysis["all_order_formal_lift"]
        self.assertTrue(
            formal["claim_boundary"][
                "first_order_repair_lifts_to_every_finite_order"
            ]
        )
        self.assertFalse(
            formal["claim_boundary"][
                "power_series_specialization_at_s_1_is_defined"
            ]
        )
        self.assertTrue(
            formal["claim_boundary"][
                "moving_target_formal_lift_unique_in_declared_linear_slice"
            ]
        )
        self.assertFalse(
            formal["claim_boundary"][
                "global_formal_branch_uniqueness_proved"
            ]
        )
        self.assertEqual(
            formal["recursive_uniqueness"][
                "restricted_jacobian_rank_over_Q"
            ],
            130,
        )
        self.assertEqual(
            formal["recursive_uniqueness"][
                "restricted_jacobian_nullity_over_Q"
            ],
            0,
        )
        self.assertEqual(
            formal["gauge_quotient"]["determinant_over_Q"], "-2"
        )

        support = analysis["finite_support_extension"]
        closure = support["bounded_missing_set_closure"]
        self.assertTrue(closure["exhaustive_census_recomputed"])
        self.assertEqual(closure["termination"], "frontier-exhausted")
        self.assertEqual(closure["nodes_examined"], 1_632_189)
        self.assertEqual(
            closure["singleton_free_size21_support_count"], 6
        )
        support_boundary = support["claim_boundary"]
        self.assertTrue(
            support_boundary[
                "finite_exact_total_support_at_least_22_proved"
            ]
        )
        self.assertEqual(
            support_boundary[
                "finite_exact_total_support_lower_bound"
            ],
            22,
        )
        self.assertTrue(
            support_boundary["retains_all_nine_natural_slots"]
        )
        self.assertFalse(
            support_boundary[
                "supports_dropping_a_natural_slot_excluded"
            ]
        )
        self.assertFalse(
            support_boundary[
                "supports_of_total_size_22_or_more_excluded"
            ]
        )

    def test_mechanism_audit_rules_out_the_proposed_shortcuts(self):
        audit = self.report["mechanism_audit"]
        matching = audit["matching_linear_dependence"]
        self.assertEqual(matching["matching_edge_incidence_rank"], 10)
        self.assertEqual(matching["matching_edge_incidence_nullity"], 5)
        self.assertTrue(
            matching["sample_relation_replays_edgewise"]
        )
        self.assertFalse(
            matching["linear_matching_monomial_relation_found"]
        )

        sign = audit["sign"]
        self.assertEqual(sign["constant_residuals"], [0, 0, 0])
        self.assertEqual(sign["victim_residual"], -1)
        self.assertFalse(sign["plus_one_is_sign_forced"])

        representation = audit["representation"]
        self.assertEqual(
            representation[
                "S6_sign_multiplicity_in_matching_permutation_module"
            ],
            0,
        )
        self.assertEqual(
            representation["S6xS3_invariant_output_directions"], 7
        )
        self.assertFalse(
            representation["linear_output_invariant_exists"]
        )

        contraction = audit["n4_contraction"]
        self.assertEqual(contraction["retained_count"], 3)
        self.assertEqual(contraction["killed_count"], 12)
        self.assertTrue(
            all(
                row["contracted_target_is_in_Phi_4_3_image"]
                for row in contraction[
                    "pairs_retaining_the_seed_defect"
                ]
            )
        )
        self.assertEqual(
            contraction["dominance_audit"]["rank_mod_31"], 81
        )
        self.assertNotEqual(
            contraction["dominance_audit"][
                "pivot_minor_determinant_mod_31"
            ],
            0,
        )
        self.assertEqual(
            contraction["Phi_4_3_image_dimension_upper_bound"], 51
        )
        self.assertFalse(
            contraction[
                "universal_Bell_contraction_preserves_Phi_image"
            ]
        )

    def test_restricted_nullstellensatz_certificate_is_not_overclaimed(self):
        certificate = self.report[
            "fixed_support_nullstellensatz_certificate"
        ]
        self.assertEqual(len(certificate["constant_products"]), 3)
        self.assertEqual(len(certificate["defect_monomial_D"]), 3)
        self.assertEqual(
            len(certificate["complementary_monomial_Q"]), 6
        )
        self.assertTrue(
            certificate["certificate_replays_symbolically"]
        )
        self.assertTrue(
            certificate["content_identity_replays_symbolically"]
        )
        self.assertEqual(
            certificate["collected_identity_terms"],
            [{"monomial_variable_indices": [], "coefficient": 1}],
        )
        theorem = self.report["restricted_family_theorem"]
        self.assertFalse(
            theorem["signs_or_phases_can_cancel_the_forced_coefficient"]
        )
        self.assertFalse(theorem["full_135_variable_nonimage_theorem"])
        boundary = self.report["claim_boundary"]
        self.assertTrue(boundary["support_ansatz_no_go_certified"])
        self.assertFalse(boundary["GHZ_nonimage_proved"])
        self.assertTrue(
            boundary["GHZ_border_image_membership_proved"]
        )
        self.assertFalse(
            boundary["GHZ_exact_affine_image_membership_decided"]
        )
        self.assertTrue(
            boundary[
                "natural_support_extensions_through_21_excluded"
            ]
        )
        self.assertEqual(
            boundary[
                "natural_support_finite_exact_total_support_lower_bound"
            ],
            22,
        )
        self.assertFalse(
            boundary["supports_dropping_a_natural_slot_excluded"]
        )
        self.assertTrue(
            boundary["equal_g_GHZ_exact_fiber_nonempty_over_Q"]
        )
        self.assertFalse(boundary["candidate_full_image_invariant_found"])
        self.assertFalse(boundary["modular_Groebner_run_performed"])

    def test_bundle_round_trips_with_hash_and_source_ledgers(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "defect"
            write_defect_mining_bundle(output)
            bundle = verify_defect_mining_bundle(output)
            self.assertTrue(bundle.exact)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    REPORT_FILE,
                    CERTIFICATE_FILE,
                    GROEBNER_D3_FILE,
                    GROEBNER_D4_FILE,
                    MANIFEST_FILE,
                },
            )
            self.assertEqual(
                [row["path"] for row in bundle.manifest["inputs"]],
                list(SOURCE_INPUTS),
            )
            self.assertIn(
                "ring r=31,",
                (output / GROEBNER_D3_FILE).read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(
                "ideal G=std(I);",
                (output / GROEBNER_D4_FILE).read_text(
                    encoding="utf-8"
                ),
            )

    def test_corruption_fails_even_after_manifest_hash_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "defect"
            write_defect_mining_bundle(output)
            report_path = output / REPORT_FILE
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["natural_near_miss"]["victim"]["residual"] = 0
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_path = output / MANIFEST_FILE
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            for row in manifest["artifacts"]:
                if row["path"] == REPORT_FILE:
                    row["bytes"] = report_path.stat().st_size
                    row["sha256"] = _sha256(report_path)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennDefectMiningError,
                "semantic replay",
            ):
                verify_defect_mining_bundle(output)

    def test_new_modules_have_no_private_engine_dependency(self):
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "border_image.py",
            "border_valuation.py",
            "cyclotomic_shadow.py",
            "defect_mining.py",
            "formal_lift.py",
            "hafnian_identities.py",
            "n6_deformation.py",
            "shadow_slices.py",
            "support_extension.py",
        ):
            source = (
                root
                / "experiments"
                / "krenn_quantum_graph"
                / filename
            ).read_text(encoding="utf-8")
            self.assertNotIn("section12_instantiation", source)


if __name__ == "__main__":
    unittest.main()
