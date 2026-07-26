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
        self.assertEqual(
            certificate["schema"],
            "krenn-n8-d3-toric-unequal-rates-v2",
        )
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
        family = depth["declared_family"]
        exact_support = family["exact_support_mode"]
        self.assertTrue(exact_support["support_equals_each_recorded_class_S"])
        self.assertTrue(exact_support["all_S_leading_coefficients_nonzero"])
        self.assertTrue(
            exact_support["outside_source_coordinates_zero_on_support_torus"]
        )
        extension = family["conditional_degeneration_extension"]
        self.assertTrue(extension["allowed"])
        self.assertTrue(
            extension["outside_source_coordinates_may_have_higher_order"]
        )
        self.assertIn(
            "strictly above every relevant active target",
            extension["required_condition"],
        )
        self.assertTrue(
            family[
                "common_active_target_monomial_order_normalized_to_zero"
            ]
        )
        self.assertFalse(family["outside_term_entry_cones_classified"])

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
            self.assertEqual(
                record["exclusion_basis"],
                "positive-singleton-tie-row-span-circuit",
            )
            self.assertTrue(
                record[
                    "entire_declared_quotient_line_and_all_residual_gauge_lifts_excluded"
                ]
            )
            self.assertEqual(
                record["positive_singleton_circuit"]["exclusion_scope"],
                "declared-exact-support-or-strictly-higher-outside-monomial-stratum",
            )
            self.assertFalse(
                record["positive_singleton_circuit"][
                    "outside_term_entry_strata_excluded"
                ]
            )
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

    def test_positive_singleton_circuits_and_target_orders_are_exact(self):
        depth = self.certificate[
            "bounded_H5_depth_two_quotient_fan"
        ]
        expected = {
            "A": {
                "coefficients": {
                    "571": 3,
                    "851": 2,
                    "2438": 1,
                    "2493": 1,
                    "4048": 1,
                    "5792": 1,
                },
                "weighted_row": [2, 2, 0, 3, 1, 2, 1, 3, 1, 1],
                "witness": [1, 3, 2, 0, 1, 3],
            },
            "B": {
                "coefficients": {"853": 1, "2430": 1, "6557": 1},
                "weighted_row": [1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
                "witness": [0, 1, 1, 0, 1, 0],
            },
        }
        target_counts = {0: 2, 3_280: 1, 6_560: 1}

        for label, values in expected.items():
            record = depth["classes"][label]
            support = tuple(record["support"])
            nonseed = tuple(record["nonseed_coordinates"])
            ties = tuple(tuple(row) for row in record["tie_matrix"])
            primary = unequal.first_shell._active_term_signature(
                support,
                unequal.first_shell._primary_matchings(),
            )
            independent = unequal.first_shell._active_term_signature(
                support,
                unequal.first_shell._independent_matchings(),
            )
            self.assertEqual(primary, independent)

            circuit = record["positive_singleton_circuit"]
            self.assertEqual(
                circuit["equation_coefficients"],
                values["coefficients"],
            )
            self.assertTrue(all(circuit["exact_checks"].values()))
            weighted = [0] * len(nonseed)
            for singleton in circuit["singleton_equations"]:
                equation = singleton["equation"]
                coefficient = singleton["positive_coefficient"]
                terms = primary[equation]
                self.assertEqual(len(terms), 1)
                matching = tuple(
                    tuple(edge) for edge in singleton["matching"]
                )
                monomial = tuple(
                    singleton["monomial_variable_indices"]
                )
                self.assertEqual(terms[0], (matching, monomial))
                incidence = [
                    sum(1 for value in monomial if value == coordinate)
                    for coordinate in nonseed
                ]
                self.assertEqual(
                    incidence,
                    singleton["nonseed_incidence_row"],
                )
                for position, value in enumerate(incidence):
                    weighted[position] += coefficient * value
            self.assertEqual(weighted, values["weighted_row"])
            self.assertEqual(
                circuit["weighted_nonseed_incidence_row"],
                values["weighted_row"],
            )
            self.assertEqual(
                circuit["tie_row_span_witness_coefficients"],
                values["witness"],
            )
            reconstructed = [
                sum(
                    coefficient * row[column]
                    for coefficient, row in zip(
                        values["witness"], ties, strict=True
                    )
                )
                for column in range(len(nonseed))
            ]
            self.assertEqual(reconstructed, weighted)
            self.assertEqual(
                unequal.first_shell._rank_over_q(ties),
                unequal.first_shell._rank_over_q(
                    (*ties, tuple(weighted))
                ),
            )

            target = circuit[
                "active_monochromatic_target_zero_order_audit"
            ]
            self.assertTrue(
                target[
                    "common_active_target_monomial_order_normalized_to_zero"
                ]
            )
            self.assertTrue(
                target["declared_scope_requires_no_outside_monomial_entry"]
            )
            self.assertIn(
                "target coordinate valuation is at least zero",
                target["consequence_in_declared_stratum"],
            )
            self.assertTrue(
                target[
                    "all_active_target_orders_identically_zero_on_tie_lattice"
                ]
            )
            self.assertEqual(len(target["support_terms"]), 4)
            observed_counts = {}
            for term in target["support_terms"]:
                equation = term["equation"]
                observed_counts[equation] = observed_counts.get(equation, 0) + 1
                matching = tuple(tuple(edge) for edge in term["matching"])
                monomial = tuple(term["monomial_variable_indices"])
                self.assertIn((matching, monomial), primary[equation])
                incidence = [
                    sum(1 for value in monomial if value == coordinate)
                    for coordinate in nonseed
                ]
                self.assertEqual(incidence, term["nonseed_incidence_row"])
                target_reconstruction = [
                    sum(
                        coefficient * row[column]
                        for coefficient, row in zip(
                            term["tie_row_span_witness_coefficients"],
                            ties,
                            strict=True,
                        )
                    )
                    for column in range(len(nonseed))
                ]
                self.assertEqual(target_reconstruction, incidence)
                self.assertEqual(
                    unequal.first_shell._rank_over_q(
                        (*ties, tuple(incidence))
                    ),
                    5,
                )
                self.assertTrue(
                    term["order_identically_zero_on_tie_lattice"]
                )
            self.assertEqual(observed_counts, target_counts)

    def test_four_canonical_lift_replays_are_diagnostic_only(self):
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
            ray = depth["classes"][label]["diagnostic_oriented_lift_replays"][
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
            self.assertEqual(
                ray["status"],
                "diagnostic-canonical-lift-replay",
            )
            self.assertTrue(ray["canonical_lift_diagnostic_only"])
            self.assertTrue(
                ray["chosen_lift_has_noncancellable_mixed_minimum"]
            )
            self.assertFalse(
                ray["quotient_ray_exclusion_claimed_from_this_replay"]
            )
            self.assertFalse(
                ray["absolute_orders_are_residual_gauge_invariant"]
            )
            self.assertIsNone(ray["exclusion_proof_basis"])
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
            "bounded_H5_support22_declared_target_leading_two_lines_excluded_by_positive_singleton_circuits",
            "bounded_H5_support22_declared_target_leading_all_residual_gauge_lifts_excluded",
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
            "diagnostic_oriented_lift_replays"
        ]["+"]["obstructing_count"] = 0
        corruptions.append(ray)

        circuit = deepcopy(self.certificate)
        circuit["bounded_H5_depth_two_quotient_fan"]["classes"]["A"][
            "positive_singleton_circuit"
        ]["equation_coefficients"]["571"] = 4
        corruptions.append(circuit)

        target = deepcopy(self.certificate)
        target["bounded_H5_depth_two_quotient_fan"]["classes"]["B"][
            "positive_singleton_circuit"
        ]["active_monochromatic_target_zero_order_audit"][
            "support_terms"
        ][0]["tie_row_span_witness_coefficients"][0] = 1
        corruptions.append(target)

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
            self.assertIn("positive singleton", readme)
            self.assertIn(
                "target polynomial valuation is therefore at least",
                readme,
            )
            self.assertIn("This excludes both quotient lines", readme)
            self.assertIn("exact support torus", readme)
            self.assertIn("not residual-gauge invariant", readme)
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
