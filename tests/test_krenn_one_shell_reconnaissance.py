import json
from copy import deepcopy
import unittest

from experiments.krenn_quantum_graph.one_shell_reconnaissance import (
    EXPECTED_COLUMN_ORBIT_FINGERPRINT,
    EXPECTED_SPARSE_SIGNATURE_FINGERPRINT,
    KrennOneShellReconnaissanceError,
    one_shell_receipt_fingerprint,
    one_shell_reconnaissance_receipt,
    replay_one_shell_separator,
    verify_one_shell_reconnaissance_receipt,
)


def _refresh_fingerprint(payload):
    payload["receipt_fingerprint"] = (
        one_shell_receipt_fingerprint(payload)
    )


class KrennOneShellReconnaissanceTests(unittest.TestCase):
    def test_receipt_round_trips_with_bounded_claims(self):
        payload = one_shell_reconnaissance_receipt()
        round_tripped = json.loads(json.dumps(
            payload, sort_keys=True
        ))
        checks = verify_one_shell_reconnaissance_receipt(
            round_tripped
        )
        self.assertTrue(
            checks["exhaustive_within_declared_scope"]
        )
        self.assertFalse(checks["full_matrix_reconstructed"])

        scope = round_tripped["scope"]
        self.assertTrue(
            scope["exhaustive_within_declared_scope"]
        )
        self.assertFalse(scope["sampled"])
        self.assertFalse(scope["full_k2_domain_enumerated"])
        self.assertEqual(
            scope["counts"]["divisibility_tests"],
            7_220_070,
        )
        self.assertEqual(
            scope["counts"]["S6_x_S3_column_orbits"],
            4_493,
        )

        layout = round_tripped["deterministic_sparse_layout"]
        self.assertEqual(
            layout["column_orbit_fingerprint"],
            EXPECTED_COLUMN_ORBIT_FINGERPRINT,
        )
        self.assertEqual(
            layout["sparse_signature_fingerprint"],
            EXPECTED_SPARSE_SIGNATURE_FINGERPRINT,
        )
        self.assertFalse(
            layout["recomputed_in_normal_verification"]
        )

        records = round_tripped[
            "modular_rank_diagnostics"
        ]["records"]
        self.assertEqual(
            [
                (
                    record["prime"],
                    record["matrix_rank"],
                    record["augmented_rank"],
                )
                for record in records
            ],
            [
                (31, 4_493, 4_494),
                (1_009, 4_493, 4_494),
                (1_000_003, 4_493, 4_494),
            ],
        )
        self.assertTrue(all(
            record["role"] == "nonproof-reconnaissance"
            for record in records
        ))
        self.assertEqual(
            round_tripped["scratch_artifact_hashes"][
                "directory"
            ],
            "D:\\KrennScratch\\obstruction_certificate",
        )

        claims = round_tripped["claims"]
        self.assertTrue(
            claims["controlled_one_shell_nonmembership_proved"]
        )
        self.assertFalse(
            claims["D_squared_global_nonmembership_proved"]
        )
        self.assertFalse(
            claims["D_in_radical_J_mix_decided"]
        )
        self.assertFalse(
            claims["global_GHZ_nonexistence_proved"]
        )
        self.assertFalse(
            claims["exact_affine_GHZ_membership_decided"]
        )

    def test_exact_separator_and_global_escapes_replay(self):
        payload = one_shell_reconnaissance_receipt()
        replay = replay_one_shell_separator(payload)
        self.assertEqual(replay["support_orbit_size"], 360)
        self.assertEqual(replay["escape_orbit_size"], 1_080)
        self.assertEqual(replay["shell_column_hits"], [1, 3])
        self.assertEqual(
            replay["global_escape_column_hits"],
            [[0, 1], [0, 1]],
        )
        self.assertEqual(
            replay["local_incident_column_orbits"], 3
        )
        self.assertEqual(
            replay["included_touching_column_orbits"], 1
        )
        self.assertEqual(replay["integer_rhs_pairing"], 1_080)
        self.assertTrue(
            replay["exact_shell_separator_replayed"]
        )
        self.assertTrue(
            replay["explicit_global_escapes_replayed"]
        )

    def test_corrupted_rows_columns_and_counts_fail_after_rehash(self):
        original = one_shell_reconnaissance_receipt()

        corrupt_row = deepcopy(original)
        row = corrupt_row["exact_shell_separator"][
            "support_row"
        ]["variable_indices"]
        row[-1] = 133
        row.sort()
        _refresh_fingerprint(corrupt_row)
        with self.assertRaisesRegex(
            KrennOneShellReconnaissanceError,
            "rows, columns, counts",
        ):
            verify_one_shell_reconnaissance_receipt(corrupt_row)

        corrupt_column = deepcopy(original)
        multiplier = corrupt_column["exact_shell_separator"][
            "sole_included_touching_column"
        ]["multiplier_variable_indices"]
        multiplier[-1] = 133
        multiplier.sort()
        _refresh_fingerprint(corrupt_column)
        with self.assertRaisesRegex(
            KrennOneShellReconnaissanceError,
            "rows, columns, counts",
        ):
            verify_one_shell_reconnaissance_receipt(
                corrupt_column
            )

        corrupt_count = deepcopy(original)
        corrupt_count["scope"]["counts"][
            "raw_columns_hitting_support_representatives"
        ] += 1
        _refresh_fingerprint(corrupt_count)
        with self.assertRaisesRegex(
            KrennOneShellReconnaissanceError,
            "rows, columns, counts",
        ):
            verify_one_shell_reconnaissance_receipt(corrupt_count)

    def test_hash_only_corruption_is_rejected(self):
        payload = one_shell_reconnaissance_receipt()
        payload["receipt_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(
            KrennOneShellReconnaissanceError,
            "fingerprint",
        ):
            verify_one_shell_reconnaissance_receipt(payload)


if __name__ == "__main__":
    unittest.main()
