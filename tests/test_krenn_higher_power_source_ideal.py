import json
import unittest
from dataclasses import replace

from experiments.krenn_quantum_graph.higher_power_source_ideal import (
    EXPECTED_CODOMAIN_ORBIT_DIMENSION,
    EXPECTED_D_SQUARED_COEFFICIENT_CENSUS,
    EXPECTED_D_SQUARED_COEFFICIENT_ORBIT_CENSUS,
    EXPECTED_D_SQUARED_SUPPORT_DIMENSION,
    EXPECTED_D_SQUARED_SUPPORT_ORBITS,
    EXPECTED_DOMAIN_ORBIT_CENSUS,
    EXPECTED_DOMAIN_ORBIT_DIMENSION,
    EXPECTED_K1_CODOMAIN_ORBIT_DIMENSION,
    EXPECTED_K1_DOMAIN_ORBIT_CENSUS,
    EXPECTED_K1_RAW_CODOMAIN_DIMENSION,
    EXPECTED_K1_RAW_MULTIPLIER_FIBER,
    EXPECTED_RAW_CODOMAIN_DIMENSION,
    EXPECTED_RAW_DOMAIN_DIMENSION,
    EXPECTED_RAW_MULTIPLIER_FIBER,
    KrennHigherPowerError,
    codomain_orbit_dimension,
    codomain_token_degrees,
    d_squared_coefficient_census,
    d_squared_coefficient_orbit_census,
    d_squared_support_dimension,
    d_squared_support_orbit_dimension,
    d_squared_support_orbit_index,
    d_squared_support_orbits,
    domain_orbit_census,
    domain_orbit_dimension,
    exact_k2_preflight,
    fixed_fine_graded_monomial_count,
    k1_recurrence_regression,
    matrix_memory_receipt,
    multiplier_token_degrees,
    raw_codomain_dimension,
    raw_domain_dimension,
    raw_multiplier_fiber_dimension,
    refuse_unreviewed_full_matrix_construction,
    squared_hafnian_support,
)


class KrennHigherPowerSourceIdealTests(unittest.TestCase):
    def test_exact_raw_dimensions_and_fine_degrees(self):
        self.assertEqual(codomain_token_degrees(), (2,) * 18)
        degrees = multiplier_token_degrees((0, 1, 2, 0, 1, 2))
        self.assertEqual(len(degrees), 18)
        self.assertEqual(sum(degrees), 30)
        for vertex in range(6):
            local = degrees[3 * vertex:3 * vertex + 3]
            self.assertEqual(sorted(local), [1, 2, 2])
        self.assertEqual(
            raw_multiplier_fiber_dimension(),
            EXPECTED_RAW_MULTIPLIER_FIBER,
        )
        self.assertEqual(
            raw_domain_dimension(),
            EXPECTED_RAW_DOMAIN_DIMENSION,
        )
        self.assertEqual(
            raw_codomain_dimension(),
            EXPECTED_RAW_CODOMAIN_DIMENSION,
        )
        invalid_action = list(range(18))
        invalid_action[0], invalid_action[3] = (
            invalid_action[3],
            invalid_action[0],
        )
        with self.assertRaisesRegex(
            KrennHigherPowerError, "physical-vertex blocks"
        ):
            fixed_fine_graded_monomial_count(
                tuple(invalid_action),
                codomain_token_degrees(),
            )

    def test_exact_burnside_dimensions(self):
        self.assertEqual(
            domain_orbit_census(),
            EXPECTED_DOMAIN_ORBIT_CENSUS,
        )
        self.assertEqual(
            domain_orbit_dimension(),
            EXPECTED_DOMAIN_ORBIT_DIMENSION,
        )
        self.assertEqual(
            codomain_orbit_dimension(),
            EXPECTED_CODOMAIN_ORBIT_DIMENSION,
        )

    def test_independent_k1_recurrence_regression(self):
        replay = k1_recurrence_regression()
        self.assertEqual(
            replay["raw_multiplier_fiber"],
            EXPECTED_K1_RAW_MULTIPLIER_FIBER,
        )
        self.assertEqual(
            replay["raw_codomain"],
            EXPECTED_K1_RAW_CODOMAIN_DIMENSION,
        )
        self.assertEqual(
            replay["domain_orbits_by_occupation"],
            EXPECTED_K1_DOMAIN_ORBIT_CENSUS,
        )
        self.assertEqual(replay["domain_orbits"], 1_314)
        self.assertEqual(
            replay["codomain_orbits"],
            EXPECTED_K1_CODOMAIN_ORBIT_DIMENSION,
        )
        self.assertTrue(
            replay["matches_embedded_k1_regression_targets"]
        )

    def test_D_squared_support_and_coefficients(self):
        self.assertEqual(len(squared_hafnian_support()), 120)
        self.assertEqual(
            d_squared_support_dimension(),
            EXPECTED_D_SQUARED_SUPPORT_DIMENSION,
        )
        records = d_squared_support_orbits()
        self.assertEqual(
            len(records), EXPECTED_D_SQUARED_SUPPORT_ORBITS
        )
        self.assertEqual(
            sum(record.orbit_size for record in records),
            EXPECTED_D_SQUARED_SUPPORT_DIMENSION,
        )
        self.assertEqual(
            d_squared_support_orbit_dimension(),
            EXPECTED_D_SQUARED_SUPPORT_ORBITS,
        )
        self.assertEqual(
            d_squared_coefficient_census(),
            EXPECTED_D_SQUARED_COEFFICIENT_CENSUS,
        )
        self.assertEqual(
            d_squared_coefficient_orbit_census(),
            EXPECTED_D_SQUARED_COEFFICIENT_ORBIT_CENSUS,
        )
        for orbit_index, record in enumerate(records):
            self.assertEqual(
                d_squared_support_orbit_index(
                    record.representative_state
                ),
                orbit_index,
            )

    def test_memory_preflight_refuses_raw_and_compressed_matrices(self):
        receipt = matrix_memory_receipt()
        self.assertEqual(
            (receipt.raw_rows, receipt.raw_columns),
            (
                EXPECTED_RAW_CODOMAIN_DIMENSION,
                EXPECTED_RAW_DOMAIN_DIMENSION,
            ),
        )
        self.assertEqual(
            (receipt.orbit_rows, receipt.orbit_columns),
            (
                EXPECTED_CODOMAIN_ORBIT_DIMENSION,
                EXPECTED_DOMAIN_ORBIT_DIMENSION,
            ),
        )
        self.assertEqual(
            receipt.raw_nnz,
            15 * EXPECTED_RAW_DOMAIN_DIMENSION,
        )
        self.assertEqual(
            receipt.orbit_nnz_upper_bound,
            15 * EXPECTED_DOMAIN_ORBIT_DIMENSION,
        )
        self.assertFalse(receipt.full_raw_construction_authorized)
        self.assertFalse(receipt.full_orbit_construction_authorized)
        self.assertGreater(
            receipt.orbit_sparse_csc_upper_bound_bytes,
            receipt.reviewed_max_sparse_bytes,
        )
        self.assertEqual(
            receipt.orbit_csc_column_pointer_bytes,
            277_924_339_616,
        )
        self.assertGreater(
            receipt.orbit_csc_column_pointer_bytes,
            receipt.reviewed_max_sparse_bytes,
        )
        with self.assertRaisesRegex(
            KrennHigherPowerError, "construction refused"
        ):
            refuse_unreviewed_full_matrix_construction()
        with self.assertRaises(KrennHigherPowerError):
            replace(
                receipt,
                full_raw_construction_authorized=True,
            )

    def test_summary_is_json_safe_and_retains_claim_boundary(self):
        payload = exact_k2_preflight()
        json.dumps(payload, sort_keys=True)
        self.assertEqual(len(payload["count_fingerprint"]), 64)
        self.assertFalse(
            payload["memory_preflight"][
                "full_orbit_construction_authorized"
            ]
        )
        claims = payload["claims"]
        self.assertFalse(claims["D_squared_in_J_mix_decided"])
        self.assertFalse(claims["D_in_radical_J_mix_decided"])
        self.assertFalse(claims["GHZ_nonexistence_proved"])
        self.assertFalse(
            claims["exact_affine_GHZ_membership_decided"]
        )
        self.assertEqual(
            payload["dependencies"]["prior_border_result"],
            (
                "GHZ_6,3 is in the Euclidean and Zariski "
                "border image"
            ),
        )


if __name__ == "__main__":
    unittest.main()
