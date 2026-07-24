from dataclasses import replace
import json
import os
import unittest

from experiments.krenn_quantum_graph.support_extension import (
    EXPECTED_CAP20_SUPPORTS,
    EXPECTED_CAP21_SUPPORTS,
    EXPECTED_DISCOVERED_BY_ADDITION_SIZE,
    KrennSupportExtensionError,
    SIZE21_ADDITIONS,
    SupportClosureReplay,
    bounded_missing_set_closure,
    cached_exhaustive_cap21_replay,
    certify_support_extension,
    minimal_victim_repairs,
    natural_laurent_border_orders,
    natural_support,
    size21_support_audits,
    victim_missing_coordinate_histogram,
)
from experiments.krenn_quantum_graph.system import variable_key


class KrennSupportExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = certify_support_extension()

    def test_two_coordinates_are_the_exact_local_minimum(self):
        self.assertEqual(
            victim_missing_coordinate_histogram(),
            (1, 0, 6, 8),
        )
        repairs = minimal_victim_repairs()
        self.assertEqual(
            [repair.matching_index for repair in repairs],
            [0, 2, 4, 8, 9, 13],
        )
        self.assertEqual(
            [repair.added_coordinates for repair in repairs],
            [
                (88, 133),
                (106, 113),
                (11, 65),
                (19, 73),
                (29, 47),
                (37, 55),
            ],
        )
        self.assertEqual(
            [repair.forced_singleton_equations for repair in repairs],
            [
                (7, 63),
                (449, 643),
                (148, 286),
                (233, 565),
                (170, 558),
                (85, 279),
            ],
        )
        self.assertTrue(
            all(
                a != b
                for repair in repairs
                for _i, _j, a, b in repair.added_variable_keys
            )
        )

    def test_finite_valuation_repair_forces_order_zero_singletons(self):
        audits = self.certificate.finite_valuation_audits
        self.assertEqual(len(audits), 6)
        for audit in audits:
            self.assertEqual(
                audit.natural_constant_product_orders,
                (0, 0, 0),
            )
            self.assertTrue(
                audit.natural_coordinate_orders_forced_zero
            )
            self.assertEqual(audit.added_order_sum, 0)
            self.assertTrue(
                audit.added_coordinate_orders_forced_zero
            )
            self.assertEqual(audit.singleton_orders, (0, 0))
            self.assertFalse(audit.finite_regularization_possible)
            self.assertTrue(audit.laurent_poles_excluded)

    def test_natural_support_already_has_a_singular_border_path(self):
        support = natural_support()
        orders = dict(zip(support, natural_laurent_border_orders()))
        self.assertEqual(orders[0], 1)
        self.assertEqual(orders[81], -1)
        self.assertEqual(
            [order for variable, order in orders.items() if variable not in {0, 81}],
            [0] * 7,
        )

    def test_fast_bounded_closure_replays_early_exact_layers(self):
        replay = bounded_missing_set_closure(
            max_total_support=14,
            node_cap=1_000,
        )
        self.assertTrue(replay.exhaustive)
        self.assertTrue(replay.raw_supports_without_symmetry_quotient)
        self.assertEqual(replay.nodes_examined, 161)
        self.assertEqual(replay.supports_discovered, 161)
        self.assertEqual(
            replay.discovered_by_addition_size,
            EXPECTED_DISCOVERED_BY_ADDITION_SIZE[:6],
        )
        self.assertEqual(replay.singleton_free_supports, ())

    def test_six_size21_candidates_have_content_derived_odd_cycles(self):
        audits = size21_support_audits()
        self.assertEqual(len(audits), 6)
        self.assertEqual(
            [audit.added_coordinates for audit in audits],
            list(SIZE21_ADDITIONS),
        )
        for audit in audits:
            self.assertEqual(len(audit.support), 21)
            self.assertEqual(audit.mixed_equation_count, 18)
            self.assertEqual(
                set(audit.active_terms_per_mixed_equation),
                {2},
            )
            self.assertTrue(audit.singleton_free)
            self.assertTrue(
                audit.odd_cycle.exponent_relation_zero
            )
            self.assertEqual(audit.odd_cycle.rhs_product, -1)
            self.assertTrue(
                audit.odd_cycle
                .contradiction_over_characteristic_not_two
            )
            self.assertFalse(audit.finite_nonzero_solution_possible)

    def test_claim_boundary_is_only_a_support22_lower_bound(self):
        certificate = self.certificate
        self.assertEqual(certificate.minimum_local_additions, 2)
        self.assertEqual(
            certificate.recorded_cap20_supports,
            EXPECTED_CAP20_SUPPORTS,
        )
        self.assertEqual(
            certificate.recorded_cap21_supports,
            EXPECTED_CAP21_SUPPORTS,
        )
        self.assertEqual(
            certificate.recorded_discovered_by_addition_size,
            EXPECTED_DISCOVERED_BY_ADDITION_SIZE,
        )
        self.assertEqual(
            certificate.finite_exact_total_support_lower_bound,
            None,
        )
        self.assertFalse(certificate.exhaustive_census_recomputed)
        self.assertTrue(certificate.retains_all_natural_slots)
        self.assertEqual(certificate.symmetry_weight_equalities, 0)
        self.assertFalse(
            certificate.unrestricted_nonexistence_proved
        )
        boundary = certificate.to_dict()["claim_boundary"]
        self.assertFalse(
            boundary[
                "finite_exact_total_support_at_least_22_proved"
            ]
        )
        self.assertFalse(boundary["cap21_exhaustive_census_replayed"])
        self.assertIsNone(
            boundary["finite_exact_total_support_lower_bound"]
        )
        self.assertFalse(
            boundary["supports_of_total_size_22_or_more_excluded"]
        )
        self.assertFalse(
            boundary["supports_dropping_a_natural_slot_excluded"]
        )
        self.assertFalse(boundary["Laurent_or_border_paths_excluded"])

    def test_json_receipt_contains_all_six_content_cycles(self):
        payload = self.certificate.to_dict()
        json.dumps(payload, sort_keys=True)
        repairs = payload["minimum_local_repair"]["repairs"]
        self.assertEqual(
            payload["minimum_local_repair"][
                "victim_missing_coordinate_histogram_0_1_2_3"
            ],
            [1, 0, 6, 8],
        )
        supports = payload["bounded_missing_set_closure"][
            "size21_support_receipts"
        ]
        self.assertEqual(len(repairs), 6)
        self.assertEqual(len(supports), 6)
        for receipt in supports:
            cycle = receipt["odd_three_binomial_cycle"]
            self.assertEqual(len(cycle["equations"]), 3)
            self.assertEqual(len(cycle["matching_pairs"]), 3)
            self.assertEqual(len(cycle["exponent_rows_in_support_order"]), 3)
            self.assertEqual(cycle["signed_exponent_sum"], [0] * 21)
            self.assertEqual(cycle["rhs_product"], -1)
            self.assertTrue(
                cycle["contradiction_over_characteristic_not_two"]
            )

    def test_serialization_rejects_tampered_compact_receipts(self):
        with self.assertRaises(KrennSupportExtensionError):
            replace(
                self.certificate,
                finite_exact_total_support_lower_bound=21,
            ).to_dict()

        first = self.certificate.size21_audits[0]
        bad_cycle = replace(first.odd_cycle, rhs_product=1)
        bad_audit = replace(first, odd_cycle=bad_cycle)
        with self.assertRaises(KrennSupportExtensionError):
            replace(
                self.certificate,
                size21_audits=(
                    bad_audit,
                    *self.certificate.size21_audits[1:],
                ),
            ).to_dict()

    def test_serialization_rejects_forged_exhaustive_receipt(self):
        fake = SupportClosureReplay(
            max_total_support=21,
            node_cap=2_000_000,
            nodes_examined=1,
            supports_discovered=1,
            discovered_by_addition_size=(1,),
            singleton_free_supports=(),
            termination="frontier-exhausted",
        )
        with self.assertRaises(KrennSupportExtensionError):
            replace(
                self.certificate,
                exhaustive_replay=fake,
            ).to_dict()

    @unittest.skipUnless(
        os.environ.get("KRENN_LONG_TESTS") == "1",
        "set KRENN_LONG_TESTS=1 for the 1,632,189-support replay",
    )
    def test_long_cap21_census_recomputes_exactly(self):
        cached_exhaustive_cap21_replay.cache_clear()
        certificate = certify_support_extension(
            recompute_exhaustive_census=True
        )
        self.assertTrue(certificate.exhaustive_census_recomputed)
        payload = certificate.to_dict()
        boundary = payload["claim_boundary"]
        self.assertTrue(boundary["cap21_exhaustive_census_replayed"])
        self.assertTrue(
            boundary[
                "finite_exact_total_support_at_least_22_proved"
            ]
        )
        self.assertEqual(
            boundary["finite_exact_total_support_lower_bound"],
            22,
        )
        closure = payload["bounded_missing_set_closure"]
        self.assertEqual(
            closure["nodes_examined"],
            EXPECTED_CAP21_SUPPORTS,
        )
        self.assertEqual(
            closure["supports_discovered"],
            EXPECTED_CAP21_SUPPORTS,
        )
        self.assertEqual(
            closure["discovered_layer_sum"],
            EXPECTED_CAP21_SUPPORTS,
        )
        self.assertEqual(
            closure["discovered_by_addition_size"],
            list(EXPECTED_DISCOVERED_BY_ADDITION_SIZE),
        )
        self.assertEqual(
            cached_exhaustive_cap21_replay.cache_info().currsize,
            1,
        )
        second = certify_support_extension(
            recompute_exhaustive_census=True
        )
        self.assertIs(
            certificate.exhaustive_replay,
            second.exhaustive_replay,
        )


if __name__ == "__main__":
    unittest.main()
