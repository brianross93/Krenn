import json
from copy import deepcopy
import unittest
from dataclasses import replace

from experiments.krenn_quantum_graph.radical_obstruction import (
    EXPECTED_NONRAINBOW_RAW_MONOMIALS,
    EXPECTED_NONRAINBOW_SUPPORT_ORBITS,
    EXPECTED_RAINBOW_SUPPORT_ORBITS,
    EXPECTED_SWITCH_CONSTRAINT_RANK,
    EXPECTED_SWITCH_PREDECESSORS,
    EXPECTED_UNIQUE_SWITCH_CONSTRAINTS,
    KrennRadicalObstructionError,
    exact_radical_obstruction_summary,
    exact_support_functional_certificate,
    support_orbit_partition,
    switch_constraint_system,
    two_vertex_contraction_audit,
    verify_rainbow_singleton_constraints,
    verify_support_functional_certificate,
)
from experiments.krenn_quantum_graph.radical_obstruction_verifier import (
    KrennRadicalIndependentVerificationError,
    verify_support_obstruction_payload,
)


class KrennRadicalObstructionTests(unittest.TestCase):
    def test_support_partition_and_rainbow_singletons_replay(self):
        partition = support_orbit_partition()
        self.assertEqual(
            len(partition.rainbow_orbit_indices),
            EXPECTED_RAINBOW_SUPPORT_ORBITS,
        )
        self.assertEqual(
            len(partition.nonrainbow_orbit_indices),
            EXPECTED_NONRAINBOW_SUPPORT_ORBITS,
        )
        self.assertEqual(
            partition.nonrainbow_raw_monomials,
            EXPECTED_NONRAINBOW_RAW_MONOMIALS,
        )
        fingerprint = verify_rainbow_singleton_constraints()
        self.assertEqual(len(fingerprint), 64)

    def test_exact_nonrainbow_switch_minor_is_unimodular(self):
        system = switch_constraint_system()
        self.assertEqual(
            system.predecessor_count,
            EXPECTED_SWITCH_PREDECESSORS,
        )
        self.assertEqual(
            len(system.constraints),
            EXPECTED_UNIQUE_SWITCH_CONSTRAINTS,
        )
        certificate = verify_support_functional_certificate()
        self.assertIs(
            certificate,
            exact_support_functional_certificate(),
        )
        self.assertEqual(
            len(certificate.basis_constraints),
            EXPECTED_SWITCH_CONSTRAINT_RANK,
        )
        self.assertEqual(certificate.basis_determinant, -1)
        self.assertEqual(
            certificate.full_constraint_rank,
            EXPECTED_SWITCH_CONSTRAINT_RANK,
        )
        self.assertEqual(
            certificate.supported_invariant_annihilator_dimension,
            0,
        )
        self.assertFalse(certificate.D_squared_membership_decided)

    def test_exact_receipts_reject_corruption(self):
        certificate = exact_support_functional_certificate()
        with self.assertRaises(KrennRadicalObstructionError):
            replace(certificate, basis_determinant=0)
        with self.assertRaises(KrennRadicalObstructionError):
            replace(certificate, full_constraint_rank=30)
        first = certificate.basis_constraints[0]
        with self.assertRaises(KrennRadicalObstructionError):
            replace(first, multiplier=first.multiplier[:-1])
        corrupt = replace(
            first,
            entries=((first.entries[0][0], 9),),
        )
        with self.assertRaisesRegex(
            KrennRadicalObstructionError, "changed"
        ):
            verify_support_functional_certificate(replace(
                certificate,
                basis_constraints=(
                    corrupt,
                    *certificate.basis_constraints[1:],
                ),
            ))

    def test_summary_is_exact_but_does_not_claim_radical_membership(self):
        payload = exact_radical_obstruction_summary()
        json.dumps(payload, sort_keys=True)
        support_search = payload["support_row_search"]
        self.assertEqual(support_search["basis_determinant"], -1)
        self.assertEqual(
            support_search[
                "supported_invariant_annihilator_dimension"
            ],
            0,
        )
        self.assertIn(
            "off-support",
            support_search["claim_boundary"],
        )
        claims = payload["claims"]
        self.assertFalse(
            claims["support_limited_D_squared_separator_exists"]
        )
        self.assertFalse(claims["D_squared_in_J_mix_decided"])
        self.assertFalse(claims["D_in_radical_J_mix_decided"])
        self.assertFalse(claims["global_GHZ_nonexistence_proved"])
        self.assertFalse(
            claims["exact_affine_GHZ_membership_decided"]
        )
        self.assertEqual(
            payload["dependencies"]["prior_k1_result"],
            "D not in J_mix at k=1",
        )

    def test_two_vertex_contraction_has_exact_integer_minors(self):
        audit = two_vertex_contraction_audit()
        self.assertEqual(
            audit["Phi_4_3_affine_image_dimension_upper_bound"],
            51,
        )
        self.assertTrue(
            audit[
                "all_nonzero_contraction_strata_have_exact_rank_lower_bound_52"
            ]
        )
        self.assertFalse(
            audit[
                "universal_nonzero_bilinear_contraction_preserves_image"
            ]
        )
        self.assertEqual(
            [
                [
                    item["rank_mod_prime"]
                    for item in record[
                        "modular_rank_reconnaissance"
                    ]
                ]
                for record in audit["rank_strata"]
            ],
            [[71, 71, 71], [81, 81, 81], [81, 81, 81]],
        )
        for record in audit["rank_strata"]:
            exact = record["exact_dimension_obstruction"]
            self.assertEqual(exact["minor_size"], 52)
            self.assertEqual(exact["rank_over_Q_at_least"], 52)
            self.assertNotEqual(exact["integer_determinant"], 0)
            self.assertTrue(all(
                item["role"] == "nonproof-reconnaissance"
                for item in record[
                    "modular_rank_reconnaissance"
                ]
            ))

    def test_stdlib_only_independent_verifier_replays_symbolically(self):
        payload = exact_radical_obstruction_summary()
        checks = verify_support_obstruction_payload(payload)
        self.assertTrue(all(
            value is True
            for key, value in checks.items()
            if not key.endswith("_decided")
        ))
        self.assertFalse(
            checks["global_radical_membership_decided"]
        )
        self.assertFalse(
            checks["exact_affine_GHZ_membership_decided"]
        )

        corruption = deepcopy(payload)
        entry = corruption["support_row_search"][
            "basis_constraints"
        ][0]["projected_entries"][0]
        entry[1] += 1
        corruption["support_row_search"]["basis_determinant"] = -1
        with self.assertRaisesRegex(
            KrennRadicalIndependentVerificationError,
            "symbolic replay",
        ):
            verify_support_obstruction_payload(corruption)

        contraction_corruption = deepcopy(payload)
        exact_minor = contraction_corruption[
            "propagation_audits"
        ]["two_vertex_contraction"]["rank_strata"][0][
            "exact_dimension_obstruction"
        ]
        exact_minor["integer_determinant"] += 1
        with self.assertRaisesRegex(
            KrennRadicalIndependentVerificationError,
            "contraction minor failed replay",
        ):
            verify_support_obstruction_payload(
                contraction_corruption
            )


if __name__ == "__main__":
    unittest.main()
