from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.mixed_associated_graded import (
    KrennMixedAssociatedGradedError,
    build_mixed_associated_graded_gate,
    verify_mixed_associated_graded_bundle,
    verify_mixed_associated_graded_gate,
    write_mixed_associated_graded_bundle,
)


RESULTS_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "krenn_quantum_graph"
    / "n6_d3_mixed_associated_graded_gate"
)


class KrennMixedAssociatedGradedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = build_mixed_associated_graded_gate()

    def test_mixed_cofactor_and_star_covariance_gate(self):
        gate = self.certificate["mixed_cofactor_gate"]
        n6 = gate["n6_laurent"]
        self.assertEqual(n6["coordinates_checked_per_enumerator"], 1215)
        self.assertEqual(n6["nonzero_count"], 18)
        self.assertEqual(n6["monochromatic_count"], 9)
        self.assertEqual(n6["mixed_count"], 9)
        self.assertEqual(
            n6["valuation_histogram"], {"-1": 3, "0": 12, "1": 3}
        )
        self.assertEqual(
            n6["mixed_valuation_histogram"],
            {"-1": 2, "0": 5, "1": 2},
        )
        self.assertEqual(
            gate["k4_exact_witness"]["mixed_count"], 0
        )
        star = gate["star_diagonal_factorization"]
        self.assertEqual(star["entries_checked"], 21_870)
        self.assertEqual(star["nonzero_entries"], 108)
        self.assertEqual(
            [record["nonzero_entries"] for record in star["roots"]],
            [18] * 6,
        )

    def test_every_derivative_order_is_pure_torus_character(self):
        orders = self.certificate[
            "all_derivative_orders_on_known_orbit"
        ]
        self.assertEqual(
            [
                orders[str(order)][
                    "unique_nonzero_unordered_coordinate_coefficients"
                ]
                for order in range(4)
            ],
            [4, 162, 2187, 10935],
        )
        self.assertEqual(
            orders["0"]["valuation_histogram"], {"0": 3, "1": 1}
        )
        self.assertEqual(
            orders["1"]["valuation_histogram"],
            {"-1": 27, "0": 108, "1": 27},
        )
        self.assertEqual(
            orders["2"]["valuation_histogram"],
            {"-1": 243, "0": 1701, "1": 243},
        )
        self.assertEqual(
            orders["3"]["valuation_histogram"], {"0": 10935}
        )
        self.assertTrue(
            all(
                record[
                    "every_coefficient_equals_base_times_character_monomial"
                ]
                for record in orders.values()
            )
        )

    def test_natural_jacobian_support_is_a_forest(self):
        gate = self.certificate["natural_jacobian_support_forest"]
        self.assertEqual(gate["shape"], [729, 135])
        self.assertEqual(gate["nonzero_entries"], 162)
        self.assertEqual(gate["active_rows"], 154)
        self.assertEqual(gate["active_columns"], 135)
        self.assertEqual(gate["row_degree_histogram"], {"1": 150, "3": 4})
        self.assertEqual(
            gate["column_degree_histogram"], {"1": 108, "2": 27}
        )
        self.assertEqual(gate["component_count"], 127)
        self.assertEqual(gate["cycle_rank"], 0)
        self.assertEqual(gate["rank_over_Q"], 130)
        self.assertEqual(gate["nullity_over_Q"], 5)
        self.assertTrue(
            gate["kernel_equals_vertex_scalar_gauge_over_Q"]
        )
        self.assertFalse(gate["first_order_holonomy_cycle_exists"])

    def test_edge_matching_incidence_has_exact_rotor_kernel(self):
        gate = self.certificate["matching_amplitude_rotor"]
        self.assertEqual(gate["shape"], [15, 15])
        self.assertEqual(gate["rank_over_Q"], 10)
        self.assertEqual(gate["kernel_dimension_over_Q"], 5)
        self.assertEqual(
            gate["unimodular_rank_minor"]["determinant"], "-1"
        )
        basis = gate["primitive_K3_3_circuit_basis"]
        self.assertEqual(len(basis["vectors"]), 5)
        self.assertEqual(basis["basis_minor_determinant"], "-1")
        self.assertTrue(basis["integral_kernel_is_saturated"])
        self.assertTrue(basis["integral_kernel_basis_certified"])
        for vector in basis["vectors"]:
            self.assertEqual(vector.count(1), 3)
            self.assertEqual(vector.count(-1), 3)
            self.assertEqual(vector.count(0), 9)
        projector = gate["canonical_rotor_projector"]
        self.assertEqual(projector["rank_over_Q"], 5)
        self.assertTrue(projector["idempotent"])
        self.assertTrue(projector["image_equals_kernel_E"])
        self.assertTrue(projector["reconstruction_operator_is_identity"])
        self.assertEqual(
            gate["S6_equivariance"]["adjacent_transposition_checks"], 5
        )

    def test_K3_3_parity_corollary_is_characteristic_zero_exact(self):
        gate = self.certificate["matching_amplitude_rotor"]
        parity = gate["parity_corollary"]
        self.assertEqual(parity["restricted_incidence_rank_over_Q"], 5)
        self.assertEqual(parity["kernel"], "m=alpha*C_parity")
        self.assertEqual(
            parity["toric_identity_after_substitution"],
            "alpha^3=(-alpha)^3",
        )
        self.assertEqual(
            parity["characteristic_zero_consequence"], "alpha=0"
        )
        self.assertFalse(
            parity["nonzero_star_invisible_K3_3_rotor_realizable"]
        )
        holonomy = gate["toric_holonomy"]
        self.assertTrue(holonomy["chart_independent"])
        self.assertTrue(
            holonomy["full_color_diagonal_gauge_invariant"]
        )
        self.assertFalse(holonomy["division_on_zero_strata_used"])

    def test_victim_cannot_remain_at_known_projective_point(self):
        gate = self.certificate["victim_projective_departure"]
        self.assertEqual(gate["victim_coloring"], [0, 0, 2, 1, 2, 1])
        self.assertEqual(gate["victim_equation"], 70)
        known = gate["known_path"]
        self.assertEqual(known["only_nonzero_matching_index"], 1)
        self.assertEqual(known["amplitude"], "t")
        self.assertFalse(
            known["known_projective_point_lies_on_hyperplane"]
        )
        self.assertFalse(
            known["all_transverse_ratios_positive_order_can_repair"]
        )
        self.assertEqual(
            gate["repair_cost_histogram"],
            {"0": 1, "1": 0, "2": 6, "3": 8},
        )
        self.assertEqual(
            gate["minimal_cost_two_matching_indices"],
            [0, 2, 4, 8, 9, 13],
        )
        self.assertEqual(
            gate["minimal_branch_support_orbits"],
            [[0, 4, 8], [2, 9, 13]],
        )
        self.assertEqual(len(gate["branch_records"]), 6)
        for record in gate["branch_records"]:
            self.assertEqual(record["initial_relation"], "r_N=-1")
            self.assertEqual(record["support_size"], 11)
            self.assertEqual(record["support_orbit_size"], 1080)
            self.assertEqual(
                record["active_monomial_count_histogram"],
                {"0": 723, "1": 5, "2": 1},
            )
            self.assertEqual(
                record["target_singleton_equations"], [0, 364, 728]
            )
            self.assertEqual(
                len(record["non_target_singleton_spills"]), 2
            )
            self.assertEqual(record["double_equation"], 70)
            self.assertFalse(
                record["size11_branch_is_exact_GHZ_witness"]
            )

    def test_certificate_and_bundle_round_trip_fail_closed(self):
        self.assertEqual(
            verify_mixed_associated_graded_gate(self.certificate),
            self.certificate,
        )
        mutable = build_mixed_associated_graded_gate()
        mutable["claim_boundary"]["affine_GHZ_membership_decided"] = True
        self.assertFalse(
            build_mixed_associated_graded_gate()["claim_boundary"][
                "affine_GHZ_membership_decided"
            ]
        )
        corrupted = deepcopy(self.certificate)
        corrupted["claim_boundary"]["affine_GHZ_membership_decided"] = True
        with self.assertRaisesRegex(
            KrennMixedAssociatedGradedError, "differs"
        ):
            verify_mixed_associated_graded_gate(corrupted)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            write_mixed_associated_graded_bundle(directory)
            verify_mixed_associated_graded_bundle(directory)

            readme_path = directory / "README.md"
            manifest_path = directory / "manifest.json"
            readme_path.write_text(
                readme_path.read_text(encoding="utf-8") + "\ncorruption\n",
                encoding="utf-8",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            readme_bytes = readme_path.read_bytes()
            manifest["bundle_files"]["README.md"] = {
                "bytes": len(readme_bytes),
                "sha256": sha256(readme_bytes).hexdigest(),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennMixedAssociatedGradedError, "README differs"
            ):
                verify_mixed_associated_graded_bundle(directory)

            write_mixed_associated_graded_bundle(directory)
            certificate_path = directory / "certificate.json"
            payload = json.loads(
                certificate_path.read_text(encoding="utf-8")
            )
            payload["matching_amplitude_rotor"]["rank_over_Q"] = 15
            certificate_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennMixedAssociatedGradedError, "differs"
            ):
                verify_mixed_associated_graded_bundle(directory)

    def test_committed_bundle_verifies(self):
        verify_mixed_associated_graded_bundle(RESULTS_DIRECTORY)


if __name__ == "__main__":
    unittest.main()
