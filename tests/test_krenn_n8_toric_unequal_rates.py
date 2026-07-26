from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import experiments.krenn_quantum_graph.n8_toric_unequal_rates as unequal


class KrennN8ToricUnequalRatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.certificate = (
            unequal.build_n8_toric_unequal_rate_certificate()
        )

    def test_global_gauge_index_and_zero_skeleton_quotients(self):
        certificate = self.certificate
        global_gauge = certificate["global_gauge_lattice_audit"]
        self.assertEqual(
            global_gauge["matrix_shape"],
            [252, 21],
        )
        self.assertEqual(
            global_gauge["ranks"],
            {"over_Q": 21, "mod_2": 20, "mod_3": 21},
        )
        self.assertEqual(global_gauge["saturation_index"], 2)
        self.assertEqual(
            global_gauge["index_two_minor"]["determinant"],
            -2,
        )
        self.assertTrue(all(global_gauge["exact_checks"].values()))

        self.assertEqual(
            certificate["scope"]["decorated_stabilizer_orbits"],
            347,
        )
        self.assertEqual(
            certificate["scope"]["raw_decorations_covered"],
            1_872,
        )
        self.assertEqual(
            certificate["scope"]["support_target_leading_charts"],
            541,
        )
        expected = {
            "H5": {
                "orbits": 43,
                "raw": 144,
                "charts": 57,
                "rank": 11,
                "determinants": {"-1": 24, "1": 19},
            },
            "H6": {
                "orbits": 304,
                "raw": 1_728,
                "charts": 484,
                "rank": 12,
                "determinants": {"-1": 158, "1": 146},
            },
        }
        for label, values in expected.items():
            record = certificate["hard_cases"][label]
            self.assertEqual(
                record["decorated_stabilizer_orbits"],
                values["orbits"],
            )
            self.assertEqual(
                record["raw_decorations_covered"],
                values["raw"],
            )
            self.assertEqual(
                record["target_leading_chart_count"],
                values["charts"],
            )
            self.assertEqual(
                record["restricted_gauge_rank"],
                values["rank"],
            )
            self.assertEqual(
                record["primitive_nonzero_skeleton_rational_quotient_rays"],
                0,
            )
            self.assertEqual(
                record["unimodular_minor_determinant_histogram"],
                values["determinants"],
            )
            self.assertTrue(all(record["exact_checks"].values()))

    def test_all_target_leading_charts_have_exact_Z_mod_4_quotient(self):
        total = 0
        q_determinants: dict[str, dict[int, int]] = {}
        for label in ("H5", "H6"):
            determinant_counts: dict[int, int] = {}
            for orbit in self.certificate["hard_cases"][label][
                "orbit_records"
            ]:
                audit = orbit["target_leading_chart_audit"]
                self.assertTrue(
                    audit[
                        "primary_independent_matching_enumerators_agree"
                    ]
                )
                self.assertTrue(
                    all(
                        record["exactly_these_two_support_terms"]
                        for record in audit[
                            "original_victim_two_term_audit"
                        ]
                    )
                )
                for chart in audit["target_leading_charts"]:
                    total += 1
                    quotient = chart["projective_integral_quotient"]
                    self.assertEqual(quotient["rank"], 0)
                    self.assertEqual(
                        quotient["torsion_invariant_factors"],
                        [4],
                    )
                    self.assertEqual(
                        quotient[
                            "rational_nonzero_primitive_ray_count"
                        ],
                        0,
                    )
                    q_map = chart["common_target_order_map"]
                    self.assertTrue(q_map["surjective_over_Z"])
                    determinant = q_map["unimodular_minor"][
                        "determinant"
                    ]
                    self.assertIn(determinant, (-1, 1))
                    determinant_counts[determinant] = (
                        determinant_counts.get(determinant, 0) + 1
                    )
                    self.assertTrue(all(chart["exact_checks"].values()))
            q_determinants[label] = determinant_counts
        self.assertEqual(total, 541)
        self.assertEqual(q_determinants["H5"], {-1: 20, 1: 37})
        self.assertEqual(q_determinants["H6"], {-1: 221, 1: 263})


    def test_independent_reconstruction_of_all_Z_mod_4_sequences(self):
        gauge = unequal.first_shell.target_preserving_gauge_matrix()
        chart_count = 0

        def dense_row(support, monomial):
            positions = {
                coordinate: position
                for position, coordinate in enumerate(support)
            }
            row = [0] * len(support)
            for coordinate in monomial:
                row[positions[coordinate]] += 1
            return tuple(row)

        def matrix_vector(matrix, vector):
            return tuple(
                sum(
                    coefficient * value
                    for coefficient, value in zip(
                        row, vector, strict=True
                    )
                )
                for row in matrix
            )

        for label in ("H5", "H6"):
            for orbit in self.certificate["hard_cases"][label][
                "orbit_records"
            ]:
                support = tuple(orbit["support"])
                restricted_gauge = tuple(
                    gauge[index] for index in support
                )
                gauge_minor_record = orbit[
                    "restricted_gauge_matrix"
                ]["unimodular_maximal_minor"]
                gauge_minor = tuple(
                    tuple(
                        restricted_gauge[row][column]
                        for column in gauge_minor_record[
                            "gauge_columns"
                        ]
                    )
                    for row in gauge_minor_record[
                        "support_row_positions"
                    ]
                )
                self.assertEqual(
                    unequal.first_shell._determinant(gauge_minor),
                    gauge_minor_record["determinant"],
                )
                self.assertEqual(
                    abs(gauge_minor_record["determinant"]),
                    1,
                )
                restricted_gauge_rank = (
                    unequal.first_shell._rank_over_q(restricted_gauge)
                )
                self.assertEqual(restricted_gauge_rank, len(gauge_minor))

                tie_rows = []
                for record in orbit["order_constraint_matrix"]["rows"][
                    3:
                ]:
                    row = [0] * len(support)
                    for entry in record["nonzero_coefficients"]:
                        row[entry["support_position"]] = entry[
                            "coefficient"
                        ]
                    tie_rows.append(tuple(row))

                for chart in orbit["target_leading_chart_audit"][
                    "target_leading_charts"
                ]:
                    chart_count += 1
                    target_rows = tuple(
                        dense_row(
                            support,
                            target[
                                "monomial_variable_indices"
                            ],
                        )
                        for target in chart[
                            "selected_target_monomials"
                        ]
                    )
                    projective = (
                        tuple(
                            second - first
                            for first, second in zip(
                                target_rows[0],
                                target_rows[1],
                                strict=True,
                            )
                        ),
                        tuple(
                            third - first
                            for first, third in zip(
                                target_rows[0],
                                target_rows[2],
                                strict=True,
                            )
                        ),
                        *tie_rows,
                    )
                    affine = (*target_rows, *tie_rows)
                    ones = (1,) * len(support)
                    self.assertEqual(
                        matrix_vector(projective, ones),
                        (0,) * len(projective),
                    )
                    self.assertEqual(sum(target_rows[0]), 4)
                    recovered = (
                        target_rows[0],
                        tuple(
                            first + difference
                            for first, difference in zip(
                                target_rows[0],
                                projective[0],
                                strict=True,
                            )
                        ),
                        tuple(
                            first + difference
                            for first, difference in zip(
                                target_rows[0],
                                projective[1],
                                strict=True,
                            )
                        ),
                        *projective[2:],
                    )
                    self.assertEqual(recovered, affine)
                    product = unequal.first_shell._integer_matrix_product(
                        affine, restricted_gauge
                    )
                    self.assertFalse(
                        any(value for row in product for value in row)
                    )
                    self.assertEqual(
                        unequal.first_shell._rank_over_q(affine)
                        + restricted_gauge_rank,
                        len(support),
                    )

                    q_system = (*projective, target_rows[0])
                    q_map = chart["common_target_order_map"]
                    witness = [0] * len(support)
                    for entry in q_map["q_equals_one_witness"]:
                        witness[entry["support_position"]] = entry[
                            "order"
                        ]
                    self.assertEqual(
                        matrix_vector(q_system, witness),
                        (0,) * len(projective) + (1,),
                    )
                    columns = q_map["unimodular_minor"][
                        "support_column_positions"
                    ]
                    q_minor = tuple(
                        tuple(row[column] for column in columns)
                        for row in q_system
                    )
                    self.assertEqual(
                        unequal.first_shell._determinant(q_minor),
                        q_map["unimodular_minor"]["determinant"],
                    )
                    self.assertIn(
                        q_map["unimodular_minor"]["determinant"],
                        (-1, 1),
                    )
                    exact_sequence = chart[
                        "projective_exact_sequence"
                    ]
                    self.assertEqual(
                        exact_sequence["q_of_common_scaling"],
                        4,
                    )
                    self.assertTrue(
                        exact_sequence["cokernel_is_Z_mod_4"]
                    )
        self.assertEqual(chart_count, 541)

    def test_bounded_depth_two_branch_and_symmetry_census(self):
        depth = self.certificate[
            "bounded_H5_depth_two_quotient_fan"
        ]
        self.assertEqual(depth["policy"]["raw_branch_count"], 20_736)
        self.assertEqual(
            depth["branch_support_size_and_quotient_dimension_histogram"],
            {
                "22,0": 284,
                "22,1": 4,
                "24,0": 19_588,
                "24,1": 852,
                "24,2": 8,
            },
        )
        self.assertEqual(
            depth["minimum_positive_quotient_support_size"],
            22,
        )
        self.assertEqual(
            depth["minimum_positive_fixed_chart_decorations"],
            4,
        )
        self.assertEqual(
            depth["complete_decoration_symmetry_classes"],
            2,
        )
        self.assertEqual(depth["oriented_quotient_ray_count"], 4)
        self.assertTrue(all(depth["exact_checks"].values()))

        expected = {
            "A": ([15, 15, 21, 35], [30, 30, 22, 33]),
            "B": ([15, 15, 22, 33], [30, 30, 21, 35]),
        }
        gauge = unequal.first_shell.target_preserving_gauge_matrix()
        for label, (representative, partner) in expected.items():
            record = depth["classes"][label]
            self.assertEqual(
                record["representative_secondary_matching_indices"],
                representative,
            )
            self.assertEqual(
                record["fixed_chart_partner_matching_indices"],
                partner,
            )
            self.assertEqual(record["support_size"], 22)
            self.assertEqual(record["tie_matrix_rank"], 5)
            self.assertEqual(record["residual_gauge_rank"], 4)
            self.assertEqual(record["quotient_dimension"], 1)
            self.assertEqual(
                record["complete_decoration_orbit_size"],
                4,
            )
            self.assertEqual(
                record["complete_decoration_stabilizer_order"],
                1,
            )
            self.assertFalse(
                record["oriented_rays_identified_by_symmetry"]
            )
            self.assertTrue(record["all_oriented_rays_excluded"])
            self.assertTrue(all(record["exact_checks"].values()))

            detector = {
                int(index): exponent
                for index, exponent in record[
                    "primitive_full_gauge_invariant_detector"
                ].items()
            }
            character = [
                sum(
                    exponent * gauge[index][column]
                    for index, exponent in detector.items()
                )
                for column in range(21)
            ]
            self.assertEqual(character, [0] * 21)
            self.assertEqual(
                record["detector_value_on_positive_ray"],
                1,
            )

    def test_four_oriented_rays_have_exact_projective_obstructions(self):
        depth = self.certificate[
            "bounded_H5_depth_two_quotient_fan"
        ]
        expected = {
            ("A", "+"): (14, {"-1": 7, "0": 7, "1": 2}),
            ("A", "-"): (10, {"-1": 3, "0": 7, "1": 6}),
            ("B", "+"): (12, {"-1": 4, "0": 8, "1": 3}),
            ("B", "-"): (12, {"-1": 4, "0": 8, "1": 3}),
        }
        matchings = unequal.first_shell._primary_matchings()
        for (label, orientation), (
            obstruction_count,
            histogram,
        ) in expected.items():
            ray = depth["classes"][label]["oriented_ray_replays"][
                orientation
            ]
            self.assertEqual(ray["target_minima"], [
                {
                    "color": 0,
                    "equation": 0,
                    "minimum_order": 0,
                    "minimum_term_count": 2,
                },
                {
                    "color": 1,
                    "equation": 3280,
                    "minimum_order": 0,
                    "minimum_term_count": 1,
                },
                {
                    "color": 2,
                    "equation": 6560,
                    "minimum_order": 0,
                    "minimum_term_count": 1,
                },
            ])
            self.assertEqual(
                ray[
                    "near_target_unique_mixed_minimum_order_histogram"
                ],
                histogram,
            )
            self.assertEqual(
                ray["obstructing_count"],
                obstruction_count,
            )
            self.assertTrue(
                ray[
                    "ray_excluded_from_projective_GHZ_initial_stratum"
                ]
            )
            self.assertTrue(all(ray["exact_checks"].values()))

            orders = [ray["outside_coordinate_guard_order"]] * 252
            for coordinate in ray["support_coordinate_orders"]:
                orders[coordinate["source_coordinate_index"]] = (
                    coordinate["order"]
                )
            witness = ray[
                "obstructing_unique_mixed_minima_at_order_at_most_target"
            ][0]
            coloring = witness["coloring"]
            term_orders = []
            term_monomials = []
            for matching in matchings:
                monomial = unequal.first_shell._monomial_for_coloring(
                    coloring, matching
                )
                term_orders.append(sum(orders[index] for index in monomial))
                term_monomials.append(tuple(sorted(monomial)))
            minimum = min(term_orders)
            minima = [
                monomial
                for order, monomial in zip(
                    term_orders, term_monomials, strict=True
                )
                if order == minimum
            ]
            self.assertLessEqual(minimum, 0)
            self.assertEqual(len(minima), 1)
            self.assertEqual(
                minima[0],
                tuple(witness["monomial_variable_indices"]),
            )

    def test_claim_boundaries_are_fail_closed(self):
        self.assertTrue(
            self.certificate["declared_family"][
                "no_outside_monomial_enters_any_recorded_flat_singleton_minimum"
            ]
        )
        claims = self.certificate["claim_boundary"]
        true_claims = {
            "all_347_balanced_minimal_skeleton_quotients_are_zero",
            "all_1872_raw_balanced_minimal_skeletons_are_covered",
            "flat_obstruction_exhaustive_in_declared_family",
            "all_541_support_target_leading_charts_have_zero_rational_quotient",
            "all_541_projective_integral_quotients_are_Z_mod_4",
            "all_original_victim_equations_are_exactly_two_term_on_support",
            "bounded_H5_parent_20736_depth_two_branches_enumerated",
            "bounded_H5_minimum_positive_support22_layer_classified",
            "bounded_H5_depth_two_four_oriented_rays_excluded",
        }
        self.assertTrue(true_claims.issubset(claims))
        for key, value in claims.items():
            self.assertEqual(value, key in true_claims)

    def test_certificate_corruption_is_rejected(self):
        verified = unequal.verify_n8_toric_unequal_rate_certificate(
            self.certificate
        )
        self.assertEqual(verified, self.certificate)

        corruptions = []
        quotient = deepcopy(self.certificate)
        quotient["hard_cases"]["H5"]["orbit_records"][0][
            "target_leading_chart_audit"
        ]["target_leading_charts"][0]["projective_integral_quotient"][
            "torsion_invariant_factors"
        ] = [2]
        corruptions.append(quotient)

        ray = deepcopy(self.certificate)
        ray["bounded_H5_depth_two_quotient_fan"]["classes"]["A"][
            "oriented_ray_replays"
        ]["+"]["obstructing_count"] = 0
        corruptions.append(ray)

        claim = deepcopy(self.certificate)
        claim["claim_boundary"]["n8_nonexistence_proved"] = True
        corruptions.append(claim)

        for payload in corruptions:
            with self.assertRaisesRegex(
                unequal.KrennN8ToricUnequalRateError,
                "differs from exact replay",
            ):
                unequal.verify_n8_toric_unequal_rate_certificate(
                    payload
                )

    def test_bundle_round_trip_inventory_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            written = unequal.write_n8_toric_unequal_rate_bundle(
                directory
            )
            self.assertEqual(
                unequal.verify_n8_toric_unequal_rate_bundle(directory),
                written,
            )
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {"certificate.json", "README.md", "manifest.json"},
            )
            readme_path = directory / "README.md"
            readme = readme_path.read_text(
                encoding="ascii"
            )
            self.assertIn("Projectively the integral quotient is `Z/4`", readme)
            self.assertIn("Bounded H5 depth-two fan", readme)
            self.assertIn("All four rays are exactly excluded", readme)
            self.assertIn("not a construction of the full GIT quotient", readme)

            readme_path.write_text(
                readme + "\ntampered README\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                unequal.KrennN8ToricUnequalRateError,
                "README changed",
            ):
                unequal.verify_n8_toric_unequal_rate_bundle(directory)

            unequal.write_n8_toric_unequal_rate_bundle(directory)

            manifest_path = directory / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="ascii")
            )
            manifest["claim_boundary"]["n8_nonexistence_proved"] = True
            manifest_path.write_bytes(
                unequal._canonical_json_bytes(manifest)
            )
            with self.assertRaisesRegex(
                unequal.KrennN8ToricUnequalRateError,
                "manifest or source ledger changed",
            ):
                unequal.verify_n8_toric_unequal_rate_bundle(directory)

            unequal.write_n8_toric_unequal_rate_bundle(directory)

            certificate_path = directory / "certificate.json"
            certificate_path.write_text(
                '{"schema":"first","schema":"second"}',
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                unequal.KrennN8ToricUnequalRateError,
                "duplicate JSON key",
            ):
                unequal.verify_n8_toric_unequal_rate_bundle(directory)

            unequal.write_n8_toric_unequal_rate_bundle(directory)
            (directory / "unexpected.cache").write_text(
                "disposable", encoding="ascii"
            )
            with self.assertRaisesRegex(
                unequal.KrennN8ToricUnequalRateError,
                "inventory changed",
            ):
                unequal.verify_n8_toric_unequal_rate_bundle(directory)

    def test_committed_bundle_verifies(self):
        directory = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "krenn_quantum_graph"
            / "n8_d3_toric_unequal_rates"
        )
        verified = unequal.verify_n8_toric_unequal_rate_bundle(
            directory
        )
        self.assertEqual(verified, self.certificate)


if __name__ == "__main__":
    unittest.main()
