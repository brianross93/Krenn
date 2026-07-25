import os
import copy
import unittest

from experiments.krenn_quantum_graph.localized_chart_graded_macaulay import (
    EXPECTED_BOUNDED_CENSUS,
    EXPECTED_KOSZUL_RELATIONS,
    EXPECTED_QUOTIENT_CENSUS,
    KrennGradedMacaulayError,
    bounded_derivative_certificate_pairs,
    bounded_derivative_koszul_relations,
    bounded_derivative_orbit_columns,
    bounded_derivative_pair_orbits,
    derivative_residual_grading,
    derivative_slice_stabilizer,
    graded_derivative_macaulay_audit,
    modular_bounded_derivative_preflight,
    verify_graded_derivative_macaulay_audit,
)


class KrennLocalizedChartGradedMacaulayTest(unittest.TestCase):
    def test_split_derivative_quotients_and_character_blocks(self):
        for ambient in (11, 29):
            grading = derivative_residual_grading(ambient)
            expected = EXPECTED_QUOTIENT_CENSUS[ambient]
            self.assertEqual(grading.residual_rank, 8)
            self.assertEqual(grading.smith_diagonal, (1,))
            self.assertEqual(
                grading.derivative_character,
                expected["derivative_character"],
            )
            self.assertEqual(
                grading.variable_blocks,
                expected["variable_blocks"],
            )
            self.assertEqual(
                grading.generator_blocks,
                expected["generator_blocks"],
            )
            self.assertEqual(
                grading.zero_character_variables,
                expected["zero_character_variables"],
            )
            self.assertEqual(
                grading.zero_character_generators,
                expected["zero_character_generators"],
            )
            # Coordinates 1,...,8 provide an explicit integral section.
            self.assertEqual(
                grading.projection_rows[1:],
                tuple(
                    tuple(int(left == right) for right in range(8))
                    for left in range(8)
                ),
            )
            relation_image = tuple(
                sum(
                    grading.derivative_character[source]
                    * grading.projection_rows[source][target]
                    for source in range(9)
                )
                for target in range(8)
            )
            self.assertEqual(relation_image, (0,) * 8)

    def test_order_two_actions_preserve_slices_without_equating_variables(
        self,
    ):
        for ambient in (11, 29):
            symmetry = derivative_slice_stabilizer(ambient)
            self.assertEqual(symmetry.order, 2)
            self.assertEqual(
                symmetry.parent_element_indices,
                EXPECTED_QUOTIENT_CENSUS[ambient][
                    "stabilizer_element_indices"
                ],
            )
            self.assertEqual(
                symmetry.variable_permutations[0],
                tuple(range(129)),
            )
            self.assertEqual(
                symmetry.generator_permutations[0],
                tuple(range(730)),
            )
            nonidentity = symmetry.variable_permutations[1]
            self.assertEqual(
                tuple(nonidentity[nonidentity[index]] for index in range(129)),
                tuple(range(129)),
            )
            self.assertNotEqual(nonidentity, tuple(range(129)))

    def test_degree_four_and_five_exact_matrix_censuses(self):
        for ambient in (11, 29):
            for degree in (4, 5):
                expected = EXPECTED_BOUNDED_CENSUS[(ambient, degree)]
                pairs = bounded_derivative_certificate_pairs(
                    ambient, degree
                )
                orbits = bounded_derivative_pair_orbits(
                    ambient, degree
                )
                columns = bounded_derivative_orbit_columns(
                    ambient, degree
                )
                rows = {
                    monomial
                    for column in columns
                    for monomial in column
                }.union({()})
                self.assertEqual(
                    (
                        len(pairs),
                        len(orbits),
                        len(rows),
                        sum(len(column) for column in columns),
                    ),
                    expected,
                )
                self.assertTrue(
                    all(len(orbit) in (1, 2) for orbit in orbits)
                )
                with self.assertRaises(TypeError):
                    columns[0][()] = 99

    def test_exact_koszul_relations_replay(self):
        for ambient in (11, 29):
            self.assertEqual(
                bounded_derivative_koszul_relations(ambient, 4),
                (),
            )
            columns = bounded_derivative_orbit_columns(ambient, 5)
            relations = bounded_derivative_koszul_relations(ambient, 5)
            self.assertEqual(
                len(relations),
                EXPECTED_KOSZUL_RELATIONS[(ambient, 5)],
            )
            for relation in relations:
                replay = {}
                for column, multiplier in relation:
                    for monomial, coefficient in columns[column].items():
                        total = (
                            replay.get(monomial, 0)
                            + multiplier * coefficient
                        )
                        if total:
                            replay[monomial] = total
                        else:
                            replay.pop(monomial, None)
                self.assertEqual(replay, {})

            tampered = list(relations[0])
            column, coefficient = tampered[0]
            tampered[0] = (column, coefficient + 1)
            replay = {}
            for column, multiplier in tampered:
                for monomial, value in columns[column].items():
                    total = (
                        replay.get(monomial, 0) + multiplier * value
                    )
                    if total:
                        replay[monomial] = total
                    else:
                        replay.pop(monomial, None)
            self.assertNotEqual(replay, {})

    def test_degree_four_and_five_rank_sandwiches_are_exact_over_q(self):
        expected_ranks = {
            (11, 4): (0, 187, 187, 188),
            (11, 5): (3, 3_791, 3_791, 3_792),
            (29, 4): (0, 180, 180, 181),
            (29, 5): (3, 3_673, 3_673, 3_674),
        }
        for ambient in (11, 29):
            for degree in (4, 5):
                for prime in (31, 1_009):
                    result = modular_bounded_derivative_preflight(
                        ambient, degree, prime=prime
                    )
                    self.assertEqual(
                        (
                            result.independent_relation_rank_mod_p,
                            result.source_rank_upper_bound_over_q,
                            result.source_rank_mod_p,
                            result.augmented_rank_mod_p,
                        ),
                        expected_ranks[(ambient, degree)],
                    )
                    self.assertFalse(result.target_in_span_mod_p)
                    self.assertTrue(
                        result.bounded_nonmembership_over_q_certified
                    )
                    payload = result.to_dict()
                    self.assertFalse(
                        payload["claim_boundary"][
                            "full_slice_unit_or_proper_status_decided"
                        ]
                    )
                    self.assertFalse(
                        payload["claim_boundary"][
                            "bounded_miss_is_global_proof"
                        ]
                    )

    def test_aggregate_audit_round_trips_and_keeps_degree_six_modular(self):
        audit = graded_derivative_macaulay_audit()
        self.assertEqual(
            verify_graded_derivative_macaulay_audit(audit), audit
        )
        self.assertEqual(
            audit["exact_conclusion"][
                "no_nullstellensatz_identity_through_total_degree"
            ],
            5,
        )
        self.assertEqual(
            audit["degree_six_status"]["rank_gaps"],
            {"11": 123, "29": 117},
        )
        for system in audit["systems"]:
            rows = system["bounded_rows"]
            self.assertTrue(
                all(
                    row["bounded_nonmembership_over_q_certified"]
                    for row in rows[:2]
                )
            )
            self.assertFalse(
                rows[2]["bounded_nonmembership_over_q_certified"]
            )
        corrupted = copy.deepcopy(audit)
        corrupted["claim_boundary"][
            "degree_six_nonmembership_over_Q_certified"
        ] = True
        with self.assertRaises(KrennGradedMacaulayError):
            verify_graded_derivative_macaulay_audit(corrupted)

    @unittest.skipUnless(
        os.environ.get("KRENN_RUN_LONG_GRADED_MACAULAY") == "1",
        "set KRENN_RUN_LONG_GRADED_MACAULAY=1 for degree-six matrices",
    )
    def test_degree_six_p1009_is_reconnaissance_only(self):
        expected = {
            11: (126, 70_952, 70_829, 70_830),
            29: (122, 68_609, 68_492, 68_493),
        }
        for ambient in (11, 29):
            result = modular_bounded_derivative_preflight(
                ambient, 6, prime=1_009
            )
            self.assertEqual(
                (
                    result.independent_relation_rank_mod_p,
                    result.source_rank_upper_bound_over_q,
                    result.source_rank_mod_p,
                    result.augmented_rank_mod_p,
                ),
                expected[ambient],
            )
            self.assertFalse(result.target_in_span_mod_p)
            self.assertFalse(
                result.bounded_nonmembership_over_q_certified
            )
            self.assertIsNone(
                result.to_dict()["exact_reason_if_certified"]
            )

    def test_invalid_inputs_fail_closed(self):
        for ambient in (0, 65):
            with self.assertRaises(KrennGradedMacaulayError):
                derivative_residual_grading(ambient)
        for degree in (3, 7):
            with self.assertRaises(KrennGradedMacaulayError):
                bounded_derivative_orbit_columns(11, degree)
        for prime in (2, 15):
            with self.assertRaises(KrennGradedMacaulayError):
                modular_bounded_derivative_preflight(
                    11, 4, prime=prime
                )


if __name__ == "__main__":
    unittest.main()
