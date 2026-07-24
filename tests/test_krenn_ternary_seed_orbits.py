import copy
import json
from itertools import product
import unittest

from experiments.krenn_quantum_graph.system import perfect_matchings
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    EXPECTED_REPRESENTATIVES_AND_SIZES,
    KrennSeedOrbitError,
    canonical_seed_representative,
    ternary_seed_orbit_census,
    transport_ordered_seed,
    verify_ternary_seed_orbit_census,
    vertex_action_table,
)


class KrennTernarySeedOrbitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = ternary_seed_orbit_census()

    def test_exact_matching_seed_and_orbit_censuses(self):
        self.assertEqual(len(perfect_matchings(6)), 15)
        self.assertEqual(len(vertex_action_table()), 720)
        counts = self.payload["counts"]
        self.assertEqual(counts["perfect_matchings"], 15)
        self.assertEqual(
            counts["ordered_three_matching_seeds"], 3375
        )
        self.assertEqual(
            counts["color_unordered_matching_multisets"], 680
        )
        self.assertEqual(counts["symmetry_orbits"], 8)
        self.assertEqual(
            counts["sum_of_ordered_orbit_sizes"], 3375
        )
        self.assertEqual(
            len(tuple(product(range(15), repeat=3))), 3375
        )
        self.assertTrue(
            all(self.payload["exact_checks"].values())
        )

    def test_representatives_and_orbit_sizes_are_deterministic(self):
        observed = tuple(
            (
                tuple(row["representative_matching_indices"]),
                row["ordered_seed_orbit_size"],
            )
            for row in self.payload["orbits"]
        )
        self.assertEqual(
            observed, EXPECTED_REPRESENTATIVES_AND_SIZES
        )
        self.assertEqual(
            tuple(size for _representative, size in observed),
            (15, 270, 360, 90, 1080, 1080, 360, 120),
        )
        self.assertEqual(sum(size for _rep, size in observed), 3375)
        self.assertEqual(
            [row["orbit_index"] for row in self.payload["orbits"]],
            list(range(8)),
        )

    def test_vertex_and_color_transport_preserve_the_orbit_key(self):
        seed = (0, 4, 13)
        vertex = (5, 2, 4, 1, 3, 0)
        color = (2, 0, 1)
        transported = transport_ordered_seed(
            seed, vertex, color
        )
        self.assertNotEqual(transported, seed)
        self.assertEqual(
            canonical_seed_representative(transported),
            canonical_seed_representative(seed),
        )
        self.assertEqual(
            canonical_seed_representative((13, 0, 4)),
            canonical_seed_representative(seed),
        )

    def test_payload_is_json_ready_and_replays_exactly(self):
        round_trip = json.loads(json.dumps(self.payload))
        self.assertEqual(round_trip, self.payload)
        self.assertEqual(
            verify_ternary_seed_orbit_census(round_trip),
            self.payload,
        )
        boundary = self.payload["claim_boundary"]
        self.assertTrue(boundary["orbit_indexing_only"])
        self.assertEqual(boundary["symmetry_weight_equalities"], 0)
        self.assertFalse(boundary["sign_search_performed"])
        self.assertFalse(boundary["no_go_claimed"])
        self.assertFalse(boundary["nonexistence_proved"])

    def test_corrupted_orbit_census_fails_recomputation(self):
        corrupted = copy.deepcopy(self.payload)
        corrupted["orbits"][0]["ordered_seed_orbit_size"] += 1
        corrupted["counts"]["sum_of_ordered_orbit_sizes"] += 1
        with self.assertRaisesRegex(
            KrennSeedOrbitError, "exact census replay"
        ):
            verify_ternary_seed_orbit_census(corrupted)


if __name__ == "__main__":
    unittest.main()
