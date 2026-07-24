import json
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from experiments.krenn_quantum_graph.source_ideal import (
    DEFAULT_MODULAR_PRIMES,
    DUAL_ROW_KEYS,
    EXPECTED_CODOMAIN_ORBITS,
    EXPECTED_COMPRESSED_FINGERPRINT,
    EXPECTED_COMPRESSED_NNZ,
    EXPECTED_D_MONOMIALS,
    EXPECTED_D_ORBITS,
    EXPECTED_DOMAIN_ORBITS,
    EXPECTED_DUAL_ROW_INDICES,
    EXPECTED_DUAL_ROW_ORBIT_SIZES,
    EXPECTED_INTEGER_DUAL_RHS,
    EXPECTED_MODULAR_AUGMENTED_RANK,
    EXPECTED_MODULAR_MATRIX_RANK,
    EXPECTED_MULTIPLIERS_PER_COLORING,
    EXPECTED_OCCUPATION_COLORINGS,
    EXPECTED_RAW_CODOMAIN_MONOMIALS,
    EXPECTED_RAW_DOMAIN_COLUMNS,
    KrennSourceIdealError,
    MULTIPLIER_GRAPH_TYPE_COUNTS,
    OCCUPATION_REPRESENTATIVES,
    build_compressed_system,
    certify_reynolds_reduction,
    codomain_orbit_count,
    compression_convention,
    d_monomials,
    d_orbit_records,
    domain_orbit_census,
    domain_orbits,
    enumerate_multiplier_monomials,
    exact_summary,
    exact_two_row_dual,
    mixed_occupation_census,
    monomial_fine_degree,
    monomial_mask,
    multiplier_count_by_inclusion_exclusion,
    multiplier_fine_degree,
    raw_codomain_monomial_count,
    raw_domain_column_count,
    run_modular_preflight,
    target_fine_degree,
    verify_exact_dual,
)


class KrennSourceIdealFastTests(unittest.TestCase):
    def test_complementary_fine_grading_enumerates_exactly_6040(self):
        for _occupation, coloring, _orbit_size, _fiber_orbits in (
            OCCUPATION_REPRESENTATIVES
        ):
            monomials = enumerate_multiplier_monomials(coloring)
            expected_degree = multiplier_fine_degree(coloring)
            self.assertEqual(
                len(monomials),
                EXPECTED_MULTIPLIERS_PER_COLORING,
            )
            self.assertEqual(tuple(sorted(monomials)), monomials)
            self.assertEqual(len(set(monomials)), len(monomials))
            self.assertEqual(
                len({monomial_mask(monomial) for monomial in monomials}),
                len(monomials),
            )
            self.assertTrue(
                all(len(monomial) == 6 for monomial in monomials)
            )
            self.assertTrue(
                all(
                    monomial_fine_degree(monomial) == expected_degree
                    for monomial in monomials
                )
            )
        self.assertEqual(
            multiplier_count_by_inclusion_exclusion(),
            EXPECTED_MULTIPLIERS_PER_COLORING,
        )
        self.assertEqual(
            sum(MULTIPLIER_GRAPH_TYPE_COUNTS.values()),
            EXPECTED_MULTIPLIERS_PER_COLORING,
        )

    def test_raw_domain_and_occupation_censuses(self):
        self.assertEqual(
            mixed_occupation_census(),
            EXPECTED_OCCUPATION_COLORINGS,
        )
        self.assertEqual(sum(EXPECTED_OCCUPATION_COLORINGS.values()), 726)
        self.assertEqual(
            raw_domain_column_count(),
            EXPECTED_RAW_DOMAIN_COLUMNS,
        )

    def test_exact_domain_orbit_compression(self):
        records = domain_orbits()
        self.assertEqual(len(records), EXPECTED_DOMAIN_ORBITS)
        self.assertEqual(
            sum(record.full_orbit_size for record in records),
            EXPECTED_RAW_DOMAIN_COLUMNS,
        )
        self.assertEqual(
            domain_orbit_census(),
            {
                "2+2+2": 202,
                "3+2+1": 556,
                "3+3": 126,
                "4+1+1": 188,
                "4+2": 172,
                "5+1": 70,
            },
        )

    def test_reynolds_averaging_is_lossless_over_Q(self):
        receipt = certify_reynolds_reduction()
        self.assertEqual(receipt.group_order, 4_320)
        self.assertEqual(receipt.action_count, 4_320)
        self.assertTrue(receipt.mixed_generator_set_stable)
        self.assertTrue(receipt.D_factor_set_stable)
        self.assertTrue(receipt.averaging_denominator_nonzero_over_Q)
        self.assertTrue(receipt.invariant_search_lossless_over_Q)

    def test_codomain_and_D_hard_censuses(self):
        self.assertEqual(
            raw_codomain_monomial_count(),
            EXPECTED_RAW_CODOMAIN_MONOMIALS,
        )
        self.assertEqual(
            codomain_orbit_count(),
            EXPECTED_CODOMAIN_ORBITS,
        )
        support = d_monomials()
        self.assertEqual(len(support), EXPECTED_D_MONOMIALS)
        self.assertTrue(
            all(
                monomial_fine_degree(monomial) == target_fine_degree()
                for monomial in support
            )
        )
        orbits = d_orbit_records()
        self.assertEqual(len(orbits), EXPECTED_D_ORBITS)
        self.assertEqual(
            sum(orbit_size for _key, orbit_size in orbits),
            EXPECTED_D_MONOMIALS,
        )

    def test_fast_exact_integer_dual_proves_k1_nonmembership(self):
        dual = exact_two_row_dual()
        self.assertEqual(dual.row_indices, EXPECTED_DUAL_ROW_INDICES)
        self.assertEqual(dual.row_keys, DUAL_ROW_KEYS)
        self.assertEqual(
            dual.row_orbit_sizes,
            EXPECTED_DUAL_ROW_ORBIT_SIZES,
        )
        self.assertEqual(dual.integer_lambda, (1, -1))
        self.assertTrue(dual.lambda_transpose_B_zero)
        self.assertEqual(
            dual.lambda_transpose_b,
            EXPECTED_INTEGER_DUAL_RHS,
        )
        self.assertEqual(dual.shared_column_index, 142)
        self.assertEqual(dual.shared_column_coefficient, 2_160)
        record = domain_orbits()[dual.shared_column_index]
        self.assertEqual(record.coloring, (0, 0, 0, 0, 1, 1))
        self.assertEqual(
            record.multiplier,
            (4, 17, 71, 85, 125, 126),
        )
        self.assertEqual(record.full_orbit_size, 2_160)
        self.assertTrue(dual.exact_Q_nonmembership_proved)
        self.assertIs(verify_exact_dual(), dual)

        payload = dual.to_dict()
        self.assertEqual(
            payload["normalized_Q_dual"],
            {
                "rows": [568, 550],
                "numerators": [1, -1],
                "denominator": 720,
                "transpose_b": 1,
            },
        )
        self.assertIn("radical", payload["claim_boundary"])
        self.assertEqual(
            [len(row) for row in payload["row_variable_keys"]],
            [9, 9],
        )

    def test_exact_receipts_reject_corruption(self):
        dual = exact_two_row_dual()
        corruptions = (
            {"row_indices": (550, 569)},
            {"row_keys": (dual.row_keys[1], dual.row_keys[0])},
            {"row_orbit_sizes": (360, 1_081)},
            {"shared_column_index": dual.shared_column_index + 1},
            {
                "shared_column_coefficient":
                dual.shared_column_coefficient + 1
            },
            {"integer_lambda": (1, 1)},
            {"lambda_transpose_b": -719},
        )
        for corruption in corruptions:
            with self.subTest(corruption=corruption):
                with self.assertRaises(KrennSourceIdealError):
                    replace(dual, **corruption)

        reynolds = certify_reynolds_reduction()
        with self.assertRaises(KrennSourceIdealError):
            replace(reynolds, mixed_generator_set_stable=False)
        with self.assertRaises(KrennSourceIdealError):
            replace(
                reynolds,
                averaging_denominator_nonzero_over_Q=False,
            )

    def test_summary_is_json_safe_and_claim_boundary_is_narrow(self):
        payload = exact_summary()
        dual = exact_two_row_dual()
        json.dumps(payload, sort_keys=True)
        claims = payload["claims"]
        convention = compression_convention()
        self.assertEqual(payload["compression_convention"], convention)
        self.assertEqual(
            payload["exact_integer_dual"]["compression_convention"],
            convention,
        )
        self.assertEqual(
            convention["codomain_row"],
            "sum of all raw equations in one codomain monomial orbit",
        )
        self.assertIn(
            "common coefficient",
            convention["domain_coordinate"],
        )
        self.assertEqual(
            claims["exact_Q_nonmembership_proved"],
            dual.exact_Q_nonmembership_proved,
        )
        self.assertEqual(
            claims["D_in_J_mix_at_k1"],
            not dual.exact_Q_nonmembership_proved,
        )
        self.assertEqual(
            claims["D_outside_radical_J_mix_proved"],
            dual.radical_nonmembership_proved,
        )
        self.assertEqual(
            claims["GHZ_nonexistence_proved"],
            dual.GHZ_nonexistence_proved,
        )
        self.assertTrue(
            payload["Reynolds_reduction"][
                "invariant_search_lossless_over_Q"
            ]
        )

    def test_modular_preflight_rejects_bad_prime_plans_before_work(self):
        with self.assertRaises(KrennSourceIdealError):
            run_modular_preflight((31,))
        with self.assertRaises(KrennSourceIdealError):
            run_modular_preflight((2, 31))
        with self.assertRaises(KrennSourceIdealError):
            run_modular_preflight((31, 31))


@unittest.skipUnless(
    os.environ.get("KRENN_LONG_TESTS") == "1",
    "set KRENN_LONG_TESTS=1 for full 3102x1314 construction and ranks",
)
class KrennSourceIdealLongTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compressed = build_compressed_system()

    def test_full_compressed_integer_system_and_dual(self):
        compressed = self.compressed
        self.assertEqual(
            compressed.shape,
            (EXPECTED_CODOMAIN_ORBITS, EXPECTED_DOMAIN_ORBITS),
        )
        self.assertEqual(compressed.nnz, EXPECTED_COMPRESSED_NNZ)
        self.assertEqual(
            sum(compressed.row_orbit_sizes),
            EXPECTED_RAW_CODOMAIN_MONOMIALS,
        )
        self.assertEqual(
            sum(value != 0 for value in compressed.rhs),
            EXPECTED_D_ORBITS,
        )
        self.assertEqual(
            compressed.fingerprint(),
            EXPECTED_COMPRESSED_FINGERPRINT,
        )
        for record, entries in zip(
            compressed.columns,
            compressed.column_entries,
            strict=True,
        ):
            self.assertEqual(
                sum(value for _row, value in entries),
                15 * record.full_orbit_size,
            )
        verify_exact_dual(compressed)
        corrupted_rhs = list(compressed.rhs)
        corrupted_rhs[EXPECTED_DUAL_ROW_INDICES[0]] = 0
        with self.assertRaises(KrennSourceIdealError):
            replace(compressed, rhs=tuple(corrupted_rhs))
        corrupted_columns = list(compressed.column_entries)
        first_column = list(corrupted_columns[0])
        row, value = first_column[0]
        first_column[0] = (row, value + 1)
        corrupted_columns[0] = tuple(first_column)
        with self.assertRaises(KrennSourceIdealError):
            replace(
                compressed,
                column_entries=tuple(corrupted_columns),
            )

    def test_modular_rank_preflight_and_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "source_ideal_modular.json"
            first = run_modular_preflight(
                DEFAULT_MODULAR_PRIMES,
                cache_path=cache,
            )
            self.assertFalse(first.loaded_from_cache)
            self.assertTrue(first.expected_rank_regression)
            self.assertEqual(
                [record.matrix_rank for record in first.records],
                [EXPECTED_MODULAR_MATRIX_RANK]
                * len(DEFAULT_MODULAR_PRIMES),
            )
            self.assertEqual(
                [record.augmented_rank for record in first.records],
                [EXPECTED_MODULAR_AUGMENTED_RANK]
                * len(DEFAULT_MODULAR_PRIMES),
            )
            second = run_modular_preflight(
                DEFAULT_MODULAR_PRIMES,
                cache_path=cache,
            )
            self.assertTrue(second.loaded_from_cache)
            self.assertEqual(
                [record.to_dict() for record in first.records],
                [record.to_dict() for record in second.records],
            )


if __name__ == "__main__":
    unittest.main()
