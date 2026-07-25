import copy
import json
import unittest

from experiments.krenn_quantum_graph.certificate_routing import (
    CERTIFICATE_ROUTING_SCHEMA,
    KrennCertificateRoutingError,
    NATURAL_CHART_ORBIT_INDEX,
    certificate_routing_audit,
    verify_certificate_routing_audit,
)
from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    normalized_seed_chart,
)


class KrennCertificateRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = certificate_routing_audit()

    def test_rabinowitsch_pair_is_redundant_with_exact_counterexample(self):
        audit = self.audit
        self.assertEqual(audit["schema"], CERTIFICATE_ROUTING_SCHEMA)
        self.assertIn(
            "iff q^m in I",
            audit["rabinowitsch_equivalence"]["statement"],
        )
        example = audit["generic_counterexample"]
        self.assertEqual(example["ideal"], "I=(x)")
        self.assertEqual(example["q"], "x")
        self.assertEqual(example["power_membership"]["m"], 1)
        self.assertEqual(
            example["open_branch_certificate"]["identity"],
            "1 = z*x - (x*z-1)",
        )
        self.assertEqual(
            example["open_branch_certificate"][
                "replayed_left_side_terms"
            ],
            [{
                "coefficient": 1,
                "x_exponent": 0,
                "z_exponent": 0,
            }],
        )
        self.assertEqual(
            example["properness_certificate"]["image_of_I_generator"],
            {},
        )
        self.assertEqual(
            example["properness_certificate"]["image_of_one"],
            {"0": 1},
        )
        self.assertTrue(
            example["exact_checks"][
                "proposed_pair_holds_but_I_is_not_unit"
            ]
        )
        self.assertFalse(
            audit["claim_boundary"][
                "expert_proposed_pair_implies_unit_ideal"
            ]
        )

    def test_correct_route_covers_open_and_closed_loci(self):
        cover = self.audit["correct_open_closed_cover"]
        self.assertIn("D(q)", cover["geometric_cover"])
        self.assertIn("V(q)", cover["geometric_cover"])
        self.assertEqual(cover["closed_branch_empty"], "1 in I+(q)")
        self.assertEqual(
            cover["unit_ideal_equivalence"],
            "1 in I iff both branches are empty",
        )
        self.assertTrue(
            self.audit["claim_boundary"]["corrected_cover_logic_proved"]
        )
        self.assertFalse(
            self.audit["claim_boundary"][
                "new_unit_ideal_certificate_emitted"
            ]
        )

    def test_krenn_victim_and_known_component_pole_replay(self):
        replay = self.audit["krenn_replay"]
        victim = replay["victim_generator"]
        self.assertEqual(victim["symbol"], "u=F_002121")
        self.assertEqual(victim["equation"], 70)
        self.assertEqual(victim["coloring"], [0, 0, 2, 1, 2, 1])
        self.assertEqual(victim["generator_kind"], "mixed")
        self.assertEqual(victim["natural_chart_term_count"], 15)
        self.assertEqual(
            replay["known_Laurent_family"],
            {
                "u_pullback": "t",
                "Q_pullback": "t^-1",
                "u_times_Q_pullback": "1",
            },
        )
        self.assertTrue(
            replay["known_natural_component"]["component_wide"]
        )
        self.assertTrue(
            replay["claim_boundary"][
                "pole_law_asserted_only_for_known_natural_component"
            ]
        )
        self.assertFalse(
            replay["claim_boundary"][
                "all_global_incidence_components_classified"
            ]
        )

    def test_natural_chart_6_fixes_every_Q_factor_to_one(self):
        replay = self.audit["krenn_replay"]
        chart_payload = replay["natural_normalized_chart"]
        chart = normalized_seed_chart(NATURAL_CHART_ORBIT_INDEX)
        self.assertEqual(chart_payload["orbit_index"], 6)
        self.assertEqual(chart_payload["seed"], [0, 4, 8])
        self.assertEqual(
            tuple(chart_payload["fixed_Q_factor_indices"]),
            tuple(POLE_INVARIANT_INDICES),
        )
        self.assertTrue(
            set(POLE_INVARIANT_INDICES)
            <= set(chart.fixed_weight_indices)
        )
        self.assertEqual(chart_payload["Q_after_normalization"], "1")
        self.assertFalse(
            replay["claim_boundary"][
                "Q_routing_simplifies_natural_chart_6"
            ]
        )
        self.assertFalse(
            replay["claim_boundary"]["natural_chart_6_decided"]
        )

    def test_round_trip_is_deterministic_and_fails_closed(self):
        audit = self.audit
        self.assertEqual(certificate_routing_audit(), audit)
        self.assertEqual(verify_certificate_routing_audit(audit), audit)
        json.dumps(audit, allow_nan=False)

        escalated = copy.deepcopy(audit)
        escalated["claim_boundary"][
            "exact_affine_GHZ_membership_status"
        ] = "nonmember"
        with self.assertRaises(KrennCertificateRoutingError):
            verify_certificate_routing_audit(escalated)

        changed_q = copy.deepcopy(audit)
        changed_q["krenn_replay"]["Q"][
            "ambient_weight_indices"
        ][0] += 1
        with self.assertRaises(KrennCertificateRoutingError):
            verify_certificate_routing_audit(changed_q)

        type_corrupted = copy.deepcopy(audit)
        type_corrupted["claim_boundary"][
            "new_unit_ideal_certificate_emitted"
        ] = 0
        with self.assertRaises(KrennCertificateRoutingError):
            verify_certificate_routing_audit(type_corrupted)

        with self.assertRaises(KrennCertificateRoutingError):
            verify_certificate_routing_audit([])


if __name__ == "__main__":
    unittest.main()
