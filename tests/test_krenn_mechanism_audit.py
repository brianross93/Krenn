from copy import deepcopy
from hashlib import sha256
import json
import unittest

from experiments.krenn_quantum_graph.mechanism_audit import (
    CUBIC_INCIDENCE_SUM_COUNT,
    CUBIC_MULTISET_COUNT,
    K33_COLLISION_FIBER_COUNT,
    KrennMechanismAuditError,
    PERFECT_MATCHING_COUNT,
    QUADRATIC_MULTISET_COUNT,
    exact_mechanism_summary,
    local_support_mechanism_audit,
    matching_collision_audit,
    verify_mechanism_payload,
)


class KrennMechanismAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = exact_mechanism_summary()

    def test_matching_multiset_census_and_first_collisions(self):
        audit = matching_collision_audit()
        self.assertEqual(
            audit["perfect_matching_count"],
            PERFECT_MATCHING_COUNT,
        )
        self.assertTrue(
            audit["independent_order_agrees_with_system"]
        )
        quadratic = audit["quadratic"]
        self.assertEqual(
            quadratic["unordered_matching_multisets"],
            QUADRATIC_MULTISET_COUNT,
        )
        self.assertEqual(
            quadratic["distinct_incidence_sums"],
            QUADRATIC_MULTISET_COUNT,
        )
        self.assertEqual(quadratic["collision_fibers"], 0)
        cubic = audit["cubic"]
        self.assertEqual(
            cubic["unordered_matching_multisets"],
            CUBIC_MULTISET_COUNT,
        )
        self.assertEqual(
            cubic["distinct_incidence_sums"],
            CUBIC_INCIDENCE_SUM_COUNT,
        )
        self.assertEqual(
            cubic["fiber_size_census"], {"1": 660, "2": 10}
        )
        self.assertEqual(
            cubic["collision_fibers"],
            K33_COLLISION_FIBER_COUNT,
        )

    def test_every_cubic_collision_is_a_primitive_K33_circuit(self):
        collisions = self.payload["matching_incidence"]["cubic"][
            "collisions"
        ]
        self.assertEqual(len(collisions), 10)
        self.assertEqual(
            {
                tuple(collision["bipartition"][0])
                for collision in collisions
            },
            {
                (0, 1, 2),
                (0, 1, 3),
                (0, 1, 4),
                (0, 1, 5),
                (0, 2, 3),
                (0, 2, 4),
                (0, 2, 5),
                (0, 3, 4),
                (0, 3, 5),
                (0, 4, 5),
            },
        )
        for collision in collisions:
            self.assertEqual(
                len(collision["incidence_edges"]), 9
            )
            self.assertEqual(
                [len(side) for side in collision["collision_fiber"]],
                [3, 3],
            )
            self.assertEqual(
                len(collision["canonical_three_switches"]), 3
            )
            self.assertEqual(collision["signed_switch_sum"], [0] * 15)
            self.assertFalse(
                collision["proper_balanced_subrelation_exists"]
            )
            self.assertTrue(collision["primitive_circuit"])
            for switch in collision["canonical_three_switches"]:
                self.assertEqual(len(switch["common_edge"]), 2)
                self.assertEqual(len(switch["four_cycle_edges"]), 4)

    def test_canonical_local_identity_and_39_term_spill_replay(self):
        local = local_support_mechanism_audit()[
            "canonical_first_support"
        ]
        self.assertEqual(local["equations"], [16, 18, 188])
        self.assertEqual(
            local["matching_pairs"], [[1, 9], [0, 6], [8, 11]]
        )
        self.assertEqual(local["signs"], [1, -1, -1])
        self.assertEqual(
            local["K3_3_bipartition"],
            [[0, 2, 5], [1, 3, 4]],
        )
        self.assertEqual(local["signed_exponent_sum"], [0] * 21)
        restricted = local["restricted_expansion"]
        self.assertEqual(restricted["nonzero_term_count"], 1)
        self.assertEqual(restricted["target_coefficient"], 2)
        self.assertTrue(restricted["replays_exactly"])
        full = local["unrestricted_expansion"]
        self.assertEqual(
            full["outside_source_support_generator_terms"], 39
        )
        self.assertTrue(
            full["outside_degree_three_terms_pairwise_distinct"]
        )
        self.assertEqual(full["lifted_degree_nine_spill_terms"], 39)
        self.assertTrue(full["lifted_spill_terms_pairwise_distinct"])
        self.assertTrue(full["lifted_spills_avoid_local_target"])
        self.assertEqual(full["nonzero_full_expansion_terms"], 40)
        self.assertFalse(full["closes_to_two_times_target"])
        self.assertEqual(len(full["spill_terms"]), 39)

    def test_all_six_support_cycles_share_the_mechanism(self):
        census = self.payload["support21_local_mechanism"][
            "all_six_support21_cycles"
        ]
        self.assertEqual(census["cycle_count"], 6)
        self.assertTrue(
            census["all_are_K3_3_cubic_matching_circuits"]
        )
        self.assertEqual(
            census[
                "outside_source_support_generator_term_counts"
            ],
            [39] * 6,
        )
        self.assertEqual(
            census["lifted_degree_nine_spill_term_counts"],
            [39] * 6,
        )

    def test_claim_boundary_does_not_promote_local_obstruction(self):
        json.dumps(self.payload, allow_nan=False, sort_keys=True)
        claims = self.payload["claims"]
        self.assertFalse(
            claims[
                "canonical_local_combination_is_global_source_ideal_identity"
            ]
        )
        self.assertFalse(
            claims["D_not_in_J_mix_at_k1_reproved_here"]
        )
        self.assertFalse(claims["D_squared_in_J_mix_decided"])
        self.assertFalse(claims["D_in_radical_J_mix_decided"])
        self.assertFalse(claims["global_GHZ_nonexistence_proved"])
        self.assertFalse(
            claims["exact_affine_GHZ_membership_decided"]
        )
        self.assertIn(
            "fixed nonzero source-coordinate support",
            self.payload["claim_boundary"],
        )
        self.assertIn(
            "not reproved",
            self.payload["distinctions"]["prior_k1_result"],
        )

    def test_roundtrip_verifier_rejects_matching_corruption(self):
        candidate = deepcopy(self.payload)
        candidate["matching_incidence"]["perfect_matchings"][0][0] = [
            0,
            2,
        ]
        with self.assertRaisesRegex(
            KrennMechanismAuditError, "semantic replay"
        ):
            verify_mechanism_payload(candidate)

    def test_roundtrip_verifier_rejects_refreshed_spill_corruption(self):
        candidate = deepcopy(self.payload)
        full = candidate["support21_local_mechanism"][
            "canonical_first_support"
        ]["unrestricted_expansion"]
        full["spill_terms"][0]["lifted_degree_nine_monomial"][0] += 1
        # Refreshing the embedded receipt hash must not bypass recomputation.
        full["spill_fingerprint_sha256"] = sha256(
            json.dumps(
                full["spill_terms"],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(
            KrennMechanismAuditError, "semantic replay"
        ):
            verify_mechanism_payload(candidate)

    def test_roundtrip_verifier_rejects_promoted_claim(self):
        candidate = deepcopy(self.payload)
        candidate["claims"]["global_GHZ_nonexistence_proved"] = True
        with self.assertRaisesRegex(
            KrennMechanismAuditError, "semantic replay"
        ):
            verify_mechanism_payload(candidate)

    def test_exact_payload_verifies_after_json_roundtrip(self):
        candidate = json.loads(json.dumps(self.payload))
        checks = verify_mechanism_payload(candidate)
        self.assertTrue(
            checks["thirty_nine_spill_terms_replayed"]
        )
        self.assertFalse(checks["global_membership_decided"])


if __name__ == "__main__":
    unittest.main()
