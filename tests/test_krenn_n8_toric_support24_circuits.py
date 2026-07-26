from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import experiments.krenn_quantum_graph.n8_toric_first_shell as first_shell
import experiments.krenn_quantum_graph.n8_toric_support24_circuits as support24


class KrennN8ToricSupport24CircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.certificate = (
            support24.build_n8_toric_support24_circuit_certificate()
        )
        cls.layer = support24._enumerated_support24_layer()
        cls.groups = {
            tuple(
                group["representative"]["secondary_matching_indices"]
            ): group
            for group in cls.layer["groups"]
        }

    @staticmethod
    def _fractions(records):
        return tuple(
            Fraction(int(numerator), int(denominator))
            for numerator, denominator in records
        )

    def _assert_row_combination(
        self,
        tie_matrix,
        row,
        coefficient_records,
    ):
        coefficients = self._fractions(coefficient_records)
        self.assertEqual(len(coefficients), len(tie_matrix))
        reconstructed = tuple(
            sum(
                coefficients[tie] * int(tie_matrix[tie][column])
                for tie in range(len(tie_matrix))
            )
            for column in range(len(row))
        )
        self.assertEqual(reconstructed, tuple(map(Fraction, row)))

    def _assert_target_receipt(self, receipt, lattice):
        tie_matrix = tuple(
            tuple(map(int, row)) for row in lattice["tie_matrix"]
        )
        nonseed = tuple(map(int, lattice["nonseed"]))
        self.assertEqual(
            receipt["common_target_order_on_tie_kernel"], 0
        )
        self.assertTrue(
            receipt[
                "all_active_target_incidence_rows_lie_in_tie_row_span"
            ]
        )
        counts = [0, 0, 0]
        for term in receipt["active_target_terms"]:
            counts[int(term["color"])] += 1
            incidence = support24._incidence_row(
                nonseed, term["monomial_variable_indices"]
            )
            self.assertEqual(
                list(incidence),
                term["restricted_nonseed_incidence"],
            )
            self._assert_row_combination(
                tie_matrix,
                incidence,
                term["tie_row_combination_coefficients"],
            )
        self.assertEqual(
            counts, receipt["active_target_term_count_triple"]
        )

    def _assert_circuit_receipt(self, receipt, lattice):
        tie_matrix = tuple(
            tuple(map(int, row)) for row in lattice["tie_matrix"]
        )
        nonseed = tuple(map(int, lattice["nonseed"]))
        singleton_rows = []
        weights = []
        for term in receipt["terms"]:
            self.assertGreater(int(term["positive_coefficient"]), 0)
            incidence = support24._incidence_row(
                nonseed, term["monomial_variable_indices"]
            )
            self.assertEqual(
                list(incidence),
                term["restricted_nonseed_incidence"],
            )
            singleton_rows.append(incidence)
            weights.append(int(term["positive_coefficient"]))
        weighted = tuple(
            sum(
                weight * row[column]
                for weight, row in zip(
                    weights, singleton_rows, strict=True
                )
            )
            for column in range(len(nonseed))
        )
        self.assertEqual(
            list(weighted), receipt["weighted_nonseed_incidence"]
        )
        self._assert_row_combination(
            tie_matrix,
            weighted,
            receipt["tie_row_combination_coefficients"],
        )
        tie_rank = first_shell._rank_over_q(tie_matrix)
        augmented_rank = first_shell._rank_over_q(
            (*tie_matrix, *singleton_rows)
        )
        self.assertEqual(tie_rank, 6)
        self.assertEqual(
            augmented_rank, tie_rank + len(singleton_rows) - 1
        )
        self.assertEqual(receipt["tie_rank"], tie_rank)
        self.assertEqual(
            receipt["tie_plus_singleton_rows_rank"],
            augmented_rank,
        )
        self.assertTrue(all(receipt["exact_checks"].values()))

    def test_exact_branch_and_parent_C2_census(self):
        scope = self.certificate["scope"]
        census = self.certificate["branch_census"]
        symmetry = self.certificate["fixed_parent_C2"]
        self.assertEqual(scope["raw_branches_enumerated"], 20_736)
        self.assertEqual(
            scope["positive_support24_raw_branches"], 860
        )
        self.assertEqual(scope["parent_C2_classes"], 444)
        self.assertEqual(
            census["support_size_and_quotient_dimension_histogram"],
            {
                "22,0": 284,
                "22,1": 4,
                "24,0": 19_588,
                "24,1": 852,
                "24,2": 8,
            },
        )
        self.assertEqual(census["rank_one_raw_branches"], 852)
        self.assertEqual(census["rank_two_raw_branches"], 8)
        self.assertEqual(census["rank_one_distinct_supports"], 850)
        self.assertEqual(census["rank_two_distinct_supports"], 8)
        self.assertEqual(symmetry["order"], 2)
        self.assertEqual(
            symmetry["nonidentity_vertices"],
            [0, 1, 3, 2, 5, 4, 7, 6],
        )
        self.assertEqual(
            symmetry["nonidentity_colors"], [0, 2, 1]
        )
        self.assertEqual(
            symmetry["rank_one_orbit_size_histogram"],
            {"1": 28, "2": 412},
        )
        self.assertEqual(
            symmetry["rank_two_orbit_size_histogram"], {"2": 4}
        )
        self.assertTrue(
            all(self.certificate["exact_checks"].values())
        )
        declared = self.certificate["declared_family"]
        self.assertTrue(
            declared[
                "exact_source_support_equals_recorded_24_coordinates"
            ]
        )
        self.assertTrue(
            declared[
                "all_recorded_support_coordinates_have_nonzero_leading_coefficients"
            ]
        )
        self.assertTrue(
            declared[
                "outside_source_coordinates_are_zero_on_exact_support_torus"
            ]
        )
        self.assertTrue(
            declared[
                "common_active_target_monomial_order_normalized_to_zero"
            ]
        )
        self.assertTrue(declared["degree_four_base_change_allowed"])
        self.assertFalse(
            declared["outside_term_entry_cones_classified"]
        )

    def test_inventory_covers_exactly_the_444_representatives(self):
        inventory = support24._validate_inventory()
        classes = self.certificate["positive_circuit_theorem"][
            "classes"
        ]
        representatives = {
            tuple(record["representative_secondary_matching_indices"])
            for record in classes
        }
        self.assertEqual(set(inventory), representatives)
        self.assertEqual(
            sum(
                record["quotient_dimension"] == 1
                for record in inventory.values()
            ),
            440,
        )
        self.assertEqual(
            sum(
                record["quotient_dimension"] == 2
                for record in inventory.values()
            ),
            4,
        )
        self.assertEqual(
            self.certificate["inventory"]["sha256"],
            support24.sha256(
                support24.CIRCUIT_INVENTORY_PATH.read_bytes()
            ).hexdigest(),
        )

    def test_all_target_rows_and_positive_circuits_reconstruct(self):
        classes = self.certificate["positive_circuit_theorem"][
            "classes"
        ]
        inventory = support24._validate_inventory()
        self.assertEqual(len(classes), 444)
        for record in classes:
            representative_tuple = tuple(
                record["representative_secondary_matching_indices"]
            )
            group = self.groups[representative_tuple]
            representative = group["representative"]
            self.assertEqual(
                record["support_sha256"],
                support24._support_hash(
                    representative["lattice"]["support"]
                ),
            )
            self.assertEqual(
                record["nonseed_coordinates_sha256"],
                support24._json_sha256(
                    list(representative["lattice"]["nonseed"])
                ),
            )
            self.assertEqual(
                record["tie_matrix_sha256"],
                support24._json_sha256(
                    [
                        list(row)
                        for row in representative["lattice"][
                            "tie_matrix"
                        ]
                    ]
                ),
            )
            representative_targets = support24._target_receipt(
                representative
            )
            self._assert_target_receipt(
                representative_targets,
                representative["lattice"],
            )
            self.assertEqual(
                record["representative_target_order_receipt"],
                support24._compact_target_receipt(
                    representative_targets
                ),
            )
            representative_circuit = support24._circuit_receipt(
                representative,
                inventory[representative_tuple]["circuit"],
            )
            self._assert_circuit_receipt(
                representative_circuit,
                representative["lattice"],
            )
            self.assertEqual(
                record[
                    "representative_positive_singleton_circuit"
                ],
                support24._compact_circuit_receipt(
                    representative_circuit
                ),
            )

            transport = record["parent_C2_partner_transport"]
            partner_tuple = tuple(
                transport["partner_secondary_matching_indices"]
            )
            partner = next(
                member
                for member in group["members"]
                if member["secondary_matching_indices"]
                == partner_tuple
            )
            partner_targets = support24._target_receipt(partner)
            self._assert_target_receipt(
                partner_targets,
                partner["lattice"],
            )
            self.assertEqual(
                record["partner_target_order_receipt"],
                support24._compact_target_receipt(partner_targets),
            )
            full_transport = support24._transported_partner_receipt(
                representative,
                partner,
                representative_circuit,
                (
                    tuple(transport["vertices"]),
                    tuple(transport["colors"]),
                ),
            )
            self._assert_circuit_receipt(
                full_transport["transported_circuit"],
                partner["lattice"],
            )
            self.assertEqual(
                transport,
                support24._compact_partner_transport(full_transport),
            )
            self.assertTrue(
                transport[
                    "transported_equations_and_monomials_match_exactly"
                ]
            )
            self.assertTrue(all(record["exact_checks"].values()))

    def test_primary_and_independent_signatures_replay_directly(self):
        seen = {}
        for group in self.layer["groups"]:
            for member in group["members"]:
                support = tuple(member["lattice"]["support"])
                if support in seen:
                    continue
                primary = first_shell._active_term_signature(
                    support, first_shell._primary_matchings()
                )
                independent = first_shell._active_term_signature(
                    support, first_shell._independent_matchings()
                )
                self.assertEqual(primary, independent)
                seen[support] = support24._json_sha256(
                    support24._signature_payload(primary)
                )
        self.assertEqual(len(seen), 858)
        for record in self.certificate["positive_circuit_theorem"][
            "classes"
        ]:
            group = self.groups[
                tuple(record[
                    "representative_secondary_matching_indices"
                ])
            ]
            representative_support = tuple(
                group["representative"]["lattice"]["support"]
            )
            self.assertEqual(
                record["representative_target_order_receipt"][
                    "active_signature_sha256"
                ],
                seen[representative_support],
            )
            self.assertEqual(
                record["representative_positive_singleton_circuit"][
                    "active_signature_sha256"
                ],
                seen[representative_support],
            )
            partner_tuple = tuple(
                record["parent_C2_partner_transport"][
                    "partner_secondary_matching_indices"
                ]
            )
            partner = next(
                member
                for member in group["members"]
                if member["secondary_matching_indices"]
                == partner_tuple
            )
            partner_support = tuple(partner["lattice"]["support"])
            self.assertEqual(
                record["partner_target_order_receipt"][
                    "active_signature_sha256"
                ],
                seen[partner_support],
            )
            self.assertEqual(
                record["parent_C2_partner_transport"][
                    "transported_circuit"
                ]["active_signature_sha256"],
                seen[partner_support],
            )

    def test_parent_involution_transports_every_recorded_term(self):
        inventory = support24._validate_inventory()
        vertices = tuple(
            self.certificate["fixed_parent_C2"][
                "nonidentity_vertices"
            ]
        )
        colors = tuple(
            self.certificate["fixed_parent_C2"][
                "nonidentity_colors"
            ]
        )
        for record in self.certificate["positive_circuit_theorem"][
            "classes"
        ]:
            representative_tuple = tuple(
                record["representative_secondary_matching_indices"]
            )
            group = self.groups[representative_tuple]
            representative = group["representative"]
            representative_receipt = support24._circuit_receipt(
                representative,
                inventory[representative_tuple]["circuit"],
            )
            representative_terms = representative_receipt["terms"]
            partner_tuple = tuple(
                record["parent_C2_partner_transport"][
                    "partner_secondary_matching_indices"
                ]
            )
            partner = next(
                member
                for member in group["members"]
                if member["secondary_matching_indices"]
                == partner_tuple
            )
            full_transport = support24._transported_partner_receipt(
                representative,
                partner,
                representative_receipt,
                (vertices, colors),
            )
            partner_terms = {
                int(term["equation"]): term
                for term in full_transport["transported_circuit"][
                    "terms"
                ]
            }
            for term in representative_terms:
                coloring = first_shell.coloring_from_index(
                    8, 3, int(term["equation"])
                )
                equation = first_shell.coloring_index(
                    8,
                    3,
                    first_shell._transport_coloring(
                        coloring, vertices, colors
                    ),
                )
                transported = partner_terms[equation]
                expected_monomial = sorted(
                    first_shell._transport_variable(
                        index, vertices, colors
                    )
                    for index in term["monomial_variable_indices"]
                )
                expected_matching = first_shell._transport_matching(
                    tuple(tuple(edge) for edge in term["matching"]),
                    vertices,
                )
                self.assertEqual(
                    transported["positive_coefficient"],
                    term["positive_coefficient"],
                )
                self.assertEqual(
                    transported["monomial_variable_indices"],
                    expected_monomial,
                )
                self.assertEqual(
                    tuple(
                        tuple(edge)
                        for edge in transported["matching"]
                    ),
                    expected_matching,
                )

    def test_claim_boundary_is_fail_closed(self):
        boundary = self.certificate["claim_boundary"]
        self.assertTrue(
            boundary[
                "bounded_fixed_H5_parent_support24_positive_quotient_layer_excluded"
            ]
        )
        self.assertTrue(
            boundary[
                "all_support24_positive_quotient_branches_in_that_enumeration_excluded"
            ]
        )
        for claim in (
            "all_H5_depth_two_parents_excluded",
            "deeper_repair_closures_excluded",
            "outside_support_terms_at_leading_order_classified",
            "all_31_n8_seed_orbits_classified",
            "n8_projective_border_membership_proved",
            "n8_strict_border_membership_proved",
            "n8_affine_membership_proved",
            "n8_nonexistence_proved",
            "finite_counterexample_found",
        ):
            self.assertFalse(boundary[claim])
        self.assertFalse(
            self.certificate["rank_two_fan_diagnostic"]["included"]
        )
        self.assertFalse(
            self.certificate["rank_two_fan_diagnostic"][
                "used_for_exclusion"
            ]
        )

    def test_certificate_corruptions_are_rejected(self):
        corruptions = []

        coefficient = deepcopy(self.certificate)
        coefficient["positive_circuit_theorem"]["classes"][0][
            "representative_positive_singleton_circuit"
        ]["circuit"][0][1] += 1
        corruptions.append(coefficient)

        row_span = deepcopy(self.certificate)
        row_span["positive_circuit_theorem"]["classes"][0][
            "representative_positive_singleton_circuit"
        ]["tie_row_combination_coefficients"][0][0] += 1
        corruptions.append(row_span)

        target = deepcopy(self.certificate)
        target["positive_circuit_theorem"]["classes"][0][
            "representative_target_order_receipt"
        ]["active_target_terms_sha256"] = "0" * 64
        corruptions.append(target)

        partner = deepcopy(self.certificate)
        partner["positive_circuit_theorem"]["classes"][0][
            "parent_C2_partner_transport"
        ]["partner_secondary_matching_indices"][0] += 1
        corruptions.append(partner)

        claim = deepcopy(self.certificate)
        claim["claim_boundary"]["n8_nonexistence_proved"] = True
        corruptions.append(claim)

        inventory = deepcopy(self.certificate)
        inventory["inventory"]["sha256"] = "0" * 64
        corruptions.append(inventory)

        for payload in corruptions:
            with self.assertRaises(
                support24.KrennN8ToricSupport24CircuitError
            ):
                support24.verify_n8_toric_support24_circuit_certificate(
                    payload
                )
        with self.assertRaises(
            support24.KrennN8ToricSupport24CircuitError
        ):
            support24.verify_n8_toric_support24_circuit_certificate([])

    def test_bundle_round_trip_and_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "support24_bundle"
            written = support24.write_n8_toric_support24_circuit_bundle(
                directory
            )
            self.assertEqual(written, self.certificate)
            verified = support24.verify_n8_toric_support24_circuit_bundle(
                directory
            )
            self.assertEqual(verified, self.certificate)
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {"certificate.json", "README.md", "manifest.json"},
            )
            manifest = json.loads(
                (directory / "manifest.json").read_text("ascii")
            )
            self.assertEqual(
                {record["path"] for record in manifest["inputs"]},
                set(support24.SOURCE_PATHS),
            )
            self.assertFalse(
                manifest["scratch_data_required_for_verification"]
            )
            manifest["inputs"][0]["sha256"] = "0" * 64
            (directory / "manifest.json").write_bytes(
                support24._canonical_json_bytes(manifest)
            )
            with self.assertRaises(
                support24.KrennN8ToricSupport24CircuitError
            ):
                support24.verify_n8_toric_support24_circuit_bundle(
                    directory
                )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "support24_bundle"
            support24.write_n8_toric_support24_circuit_bundle(directory)
            (directory / "unexpected.txt").write_text(
                "not part of the exact bundle\n", encoding="ascii"
            )
            with self.assertRaises(
                support24.KrennN8ToricSupport24CircuitError
            ):
                support24.verify_n8_toric_support24_circuit_bundle(
                    directory
                )

    def test_readme_states_the_exact_boundary(self):
        readme = support24._readme_text(self.certificate)
        self.assertIn("860 raw branches", readme)
        self.assertIn("444 classes", readme)
        self.assertIn("not a global `n=8` nonexistence proof", readme)
        normalized = " ".join(readme.split())
        self.assertIn("positive-circuit theorem is exact", normalized)
        self.assertIn("basis-free", normalized)
        self.assertIn("outside-support", normalized)
        self.assertNotIn("proves n=8 nonexistence", readme.lower())


if __name__ == "__main__":
    unittest.main()
