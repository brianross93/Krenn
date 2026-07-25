import copy
import json
import unittest

from experiments.krenn_quantum_graph.n8_seed_orbits import (
    EXPECTED_REPRESENTATIVES_SIZES_AND_SINGLETONS,
    GENERATOR_COUNT,
    GROUP_ORDER,
    H5_REPRESENTATIVE,
    H6_REPRESENTATIVE,
    KrennN8SeedOrbitError,
    MATCHING_COUNT,
    N8_SEED_ORBIT_SCHEMA,
    ORDERED_SEED_COUNT,
    adjacent_vertex_action_rows,
    canonical_n8_seed_representative,
    n8_seed_generator_neighbors,
    n8_seed_orbit_census,
    verify_n8_seed_orbit_census,
)


class KrennN8SeedOrbitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = n8_seed_orbit_census()

    def test_generator_bfs_covers_exactly_thirty_one_orbits(self):
        self.assertEqual(self.payload["schema"], N8_SEED_ORBIT_SCHEMA)
        counts = self.payload["counts"]
        self.assertEqual(counts["perfect_matchings"], MATCHING_COUNT)
        self.assertEqual(
            counts["ordered_three_matching_seeds"],
            ORDERED_SEED_COUNT,
        )
        self.assertEqual(counts["symmetry_orbits"], 31)
        self.assertEqual(
            counts["sum_of_ordered_orbit_sizes"],
            ORDERED_SEED_COUNT,
        )
        self.assertEqual(
            counts["orbits_with_singleton_mixed_victims"], 31
        )
        self.assertEqual(
            counts["pairwise_hamiltonian_hard_orbits"], 2
        )
        action = self.payload["action"]
        self.assertEqual(action["group_order"], GROUP_ORDER)
        self.assertEqual(
            action["generator_count"], GENERATOR_COUNT
        )
        self.assertFalse(action["full_S8_action_table_built"])
        self.assertEqual(len(adjacent_vertex_action_rows()), 7)
        self.assertTrue(all(self.payload["exact_checks"].values()))

    def test_representatives_sizes_and_singletons_are_exact(self):
        observed = tuple(
            (
                tuple(row["representative_matching_indices"]),
                row["ordered_seed_orbit_size"],
                row["singleton_mixed_victim_count"],
            )
            for row in self.payload["orbits"]
        )
        self.assertEqual(
            observed,
            EXPECTED_REPRESENTATIVES_SIZES_AND_SINGLETONS,
        )
        self.assertEqual(
            [row["orbit_index"] for row in self.payload["orbits"]],
            list(range(31)),
        )
        self.assertEqual(
            sum(row[1] for row in observed), ORDERED_SEED_COUNT
        )
        for row in self.payload["orbits"]:
            with self.subTest(
                representative=row[
                    "representative_matching_indices"
                ]
            ):
                support = set(
                    row["seed_source_variable_indices"]
                )
                self.assertEqual(len(support), 12)
                self.assertTrue(
                    row[
                        "all_seed_supported_mixed_terms_are_singletons"
                    ]
                )
                self.assertEqual(
                    len(row["singleton_mixed_victims"]),
                    row["singleton_mixed_victim_count"],
                )
                for victim in row["singleton_mixed_victims"]:
                    coloring = victim["coloring"]
                    self.assertGreater(len(set(coloring)), 1)
                    self.assertTrue(
                        set(
                            victim[
                                "source_monomial_variable_indices"
                            ]
                        )
                        <= support
                    )

    def test_nine_generators_preserve_canonical_orbit_keys(self):
        samples = (
            (0, 0, 0),
            (104, 17, 63),
            H5_REPRESENTATIVE,
            H6_REPRESENTATIVE,
        )
        for seed in samples:
            representative = canonical_n8_seed_representative(seed)
            neighbors = n8_seed_generator_neighbors(seed)
            self.assertEqual(len(neighbors), GENERATOR_COUNT)
            for neighbor in neighbors:
                with self.subTest(seed=seed, neighbor=neighbor):
                    self.assertEqual(
                        canonical_n8_seed_representative(neighbor),
                        representative,
                    )

    def test_h5_is_the_independent_five_internal_monomial_case(self):
        h5 = self.payload["hard_cases"]["H5"]
        self.assertEqual(
            tuple(h5["representative_matching_indices"]),
            H5_REPRESENTATIVE,
        )
        self.assertEqual(
            h5["pair_union_component_sizes"],
            [[8], [8], [8]],
        )
        self.assertEqual(h5["physical_union_edge_count"], 12)
        self.assertEqual(h5["internal_matching_count"], 5)
        self.assertEqual(
            h5["internal_matching_indices"], [0, 1, 14, 19, 38]
        )
        matrix = h5["colored_exponent_matrix"]
        self.assertEqual(matrix["columns"], 5)
        self.assertEqual(matrix["rank_over_Q"], 5)
        self.assertEqual(matrix["nullity_over_Q"], 0)
        self.assertFalse(
            h5["internal_relation"]["exists_over_Q"]
        )
        self.assertEqual(
            sum(
                not record["monochromatic"]
                for record in h5["internal_colored_monomials"]
            ),
            2,
        )
        self.assertEqual(
            [
                record["inherited_coloring"]
                for record in h5["internal_colored_monomials"]
                if not record["monochromatic"]
            ],
            [
                [0, 0, 0, 0, 2, 1, 2, 1],
                [0, 0, 2, 1, 0, 0, 1, 2],
            ],
        )

    def test_h6_has_the_exact_primitive_colored_product_circuit(self):
        h6 = self.payload["hard_cases"]["H6"]
        self.assertEqual(
            tuple(h6["representative_matching_indices"]),
            H6_REPRESENTATIVE,
        )
        self.assertEqual(
            h6["pair_union_component_sizes"],
            [[8], [8], [8]],
        )
        self.assertEqual(h6["physical_union_edge_count"], 12)
        self.assertEqual(
            h6["physical_union_graph"],
            {
                "connected": True,
                "bipartite": False,
                "triangle_count": 1,
            },
        )
        self.assertEqual(h6["internal_matching_count"], 6)
        self.assertEqual(
            h6["internal_matching_indices"],
            [0, 1, 19, 29, 33, 43],
        )
        self.assertEqual(
            [
                record["inherited_coloring"]
                for record in h6["internal_colored_monomials"]
                if not record["monochromatic"]
            ],
            [
                [0, 0, 0, 0, 2, 1, 2, 1],
                [1, 2, 1, 1, 0, 0, 1, 2],
                [2, 1, 2, 2, 1, 2, 0, 0],
            ],
        )
        matrix = h6["colored_exponent_matrix"]
        self.assertEqual(matrix["rank_over_Q"], 5)
        self.assertEqual(matrix["nullity_over_Q"], 1)
        circuit = h6["colored_product_circuit"]
        self.assertEqual(
            circuit["seed_side_matching_indices"], [0, 19, 43]
        )
        self.assertEqual(
            circuit["mixed_side_matching_indices"], [1, 29, 33]
        )
        self.assertEqual(
            circuit["relation"],
            [
                {"matching_index": 0, "coefficient": -1},
                {"matching_index": 1, "coefficient": 1},
                {"matching_index": 19, "coefficient": -1},
                {"matching_index": 29, "coefficient": 1},
                {"matching_index": 33, "coefficient": 1},
                {"matching_index": 43, "coefficient": -1},
            ],
        )
        self.assertEqual(
            circuit[
                "signed_physical_edge_incidence_sum_nonzero_entries"
            ],
            [],
        )
        self.assertEqual(
            circuit["signed_exponent_sum_nonzero_entries"], []
        )
        self.assertEqual(
            set(circuit["common_product_variable_indices"]),
            set(
                next(
                    row["seed_source_variable_indices"]
                    for row in self.payload["orbits"]
                    if row["hard_case"] == "H6"
                )
            ),
        )
        self.assertTrue(circuit["primitive_integer_relation"])
        self.assertTrue(
            circuit["all_proper_five_column_subsets_independent"]
        )
        self.assertFalse(
            circuit[
                "embedded_uncolored_K3_3_plus_common_edge_claimed"
            ]
        )

    def test_payload_round_trips_and_claims_remain_fail_closed(self):
        round_trip = json.loads(json.dumps(self.payload))
        self.assertEqual(
            verify_n8_seed_orbit_census(round_trip), self.payload
        )
        boundary = self.payload["claim_boundary"]
        self.assertEqual(boundary["symmetry_weight_equalities"], 0)
        for key in (
            "numerical_search_performed",
            "singleton_victim_alone_excludes_complex_weights",
            "H5_has_no_relation_in_the_full_source_system",
            "H6_circuit_closes_unrestricted_spill_terms",
            "H6_odd_relation_alone_proves_contradiction",
            "classical_pfaffian_or_plucker_identity_claimed",
            "blocker_boundary_escape_proved",
            "n8_nonexistence_proved",
        ):
            self.assertFalse(boundary[key])

    def test_payload_corruption_and_public_input_errors_fail_closed(self):
        mutations = []

        changed_size = copy.deepcopy(self.payload)
        changed_size["orbits"][-1]["ordered_seed_orbit_size"] += 1
        mutations.append(changed_size)

        changed_victim = copy.deepcopy(self.payload)
        changed_victim["orbits"][-1][
            "singleton_mixed_victims"
        ][0]["matching_index"] += 1
        mutations.append(changed_victim)

        changed_circuit = copy.deepcopy(self.payload)
        changed_circuit["hard_cases"]["H6"][
            "colored_product_circuit"
        ]["relation"][0]["coefficient"] = 1
        mutations.append(changed_circuit)

        escalated = copy.deepcopy(self.payload)
        escalated["claim_boundary"]["n8_nonexistence_proved"] = True
        mutations.append(escalated)

        bool_for_integer = copy.deepcopy(self.payload)
        bool_for_integer["counts"]["perfect_matchings"] = True
        mutations.append(bool_for_integer)

        for mutation in mutations:
            with self.subTest(
                mutation=mutation["schema"]
            ):
                with self.assertRaises(KrennN8SeedOrbitError):
                    verify_n8_seed_orbit_census(mutation)

        for invalid in (
            (0, 0),
            (0, 0, 105),
            (0, 0, True),
            (0, 0, 1.0),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(KrennN8SeedOrbitError):
                    canonical_n8_seed_representative(invalid)
        with self.assertRaises(KrennN8SeedOrbitError):
            verify_n8_seed_orbit_census([])
        with self.assertRaises(KrennN8SeedOrbitError):
            verify_n8_seed_orbit_census(
                {"schema": float("nan")}
            )


if __name__ == "__main__":
    unittest.main()
