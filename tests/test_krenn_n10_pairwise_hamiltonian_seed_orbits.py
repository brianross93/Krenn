import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph import (
    n10_pairwise_hamiltonian_seed_orbits as census,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "results"
    / "krenn_quantum_graph"
    / "n10_d3_pairwise_hamiltonian_seed_orbits"
)


class KrennN10PairwiseHamiltonianSeedOrbitTest(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.payload = census.build_census_payload()

    def test_fixed_pair_reduction_is_exact_and_deterministic(self):
        self.assertEqual(
            len(census.fixed_first_hamiltonian_partners()),
            384,
        )
        self.assertEqual(
            census.fixed_first_hamiltonian_partners()[0],
            census.FIXED_SECOND,
        )
        self.assertEqual(
            len(census.fixed_pair_common_partners()),
            148,
        )
        self.assertEqual(
            len(
                census._pair_normalizers(
                    census.FIXED_FIRST,
                    census.FIXED_SECOND,
                )
            ),
            10,
        )
        self.assertEqual(
            census._fixed_pair_ordered_color_orbit_count(),
            24,
        )

    def test_ten_full_orbits_and_masses_reconcile(self):
        expected = [
            (248, 15, 5_443_200, 4),
            (250, 10, 3_628_800, 6),
            (253, 15, 5_443_200, 4),
            (260, 30, 10_886_400, 2),
            (266, 30, 10_886_400, 2),
            (284, 10, 3_628_800, 6),
            (443, 3, 1_088_640, 20),
            (448, 15, 5_443_200, 4),
            (485, 5, 1_814_400, 12),
            (492, 15, 5_443_200, 4),
        ]
        observed = [
            (
                row["representative_matching_indices"][2],
                row["fixed_ordered_pair_class_size"],
                row["ordered_triple_orbit_size"],
                row["stabilizer_size_in_S10_x_S3"],
            )
            for row in self.payload["orbits"]
        ]
        self.assertEqual(observed, expected)
        self.assertEqual(
            sum(row[1] for row in expected),
            148,
        )
        self.assertEqual(
            sum(row[2] for row in expected),
            53_706_240,
        )
        self.assertTrue(
            all(census.GROUP_ORDER % row[2] == 0 for row in expected)
        )

    def test_compatibility_graph_is_an_independent_mass_audit(self):
        audit = self.payload["compatibility_graph_audit"]
        self.assertEqual(
            {
                "vertices": audit["vertices"],
                "regular_degree": audit["regular_degree"],
                "edges": audit["edges"],
                "common_neighbors_per_edge": audit[
                    "common_neighbors_per_edge"
                ],
                "triangles": audit["triangles"],
                "raw_bit_rows_bytes": audit[
                    "raw_bit_rows_bytes"
                ],
            },
            {
                "vertices": 945,
                "regular_degree": 384,
                "edges": 181_440,
                "common_neighbors_per_edge": 148,
                "triangles": 8_951_040,
                "raw_bit_rows_bytes": 112_455,
            },
        )
        self.assertRegex(
            audit["raw_bit_rows_sha256"],
            r"\A[0-9a-f]{64}\Z",
        )
        self.assertEqual(
            audit["triangles"] * 6,
            self.payload["counts"][
                "ordered_pairwise_hamiltonian_triples"
            ],
        )

    def test_victims_ranks_and_circuits_replay_exactly(self):
        expected = {
            248: (7, 4, 6, 1, {"2": 1}),
            250: (6, 3, 6, 0, {}),
            253: (7, 4, 6, 1, {"3": 1}),
            260: (6, 3, 6, 0, {}),
            266: (8, 5, 6, 2, {"2": 1, "3": 2}),
            284: (6, 3, 5, 1, {"3": 1}),
            443: (
                13,
                10,
                7,
                6,
                {"2": 10, "3": 25, "4": 30, "5": 17, "6": 5},
            ),
            448: (9, 6, 6, 3, {"2": 3, "3": 3, "4": 3}),
            485: (
                12,
                9,
                7,
                5,
                {"2": 3, "3": 18, "4": 12, "5": 24},
            ),
            492: (8, 5, 6, 2, {"2": 1, "3": 1, "4": 2}),
        }
        observed = {}
        for row in self.payload["orbits"]:
            third = row["representative_matching_indices"][2]
            matrix = row["colored_exponent_matrix"]
            observed[third] = (
                row["internal_physical_matching_count"],
                row["singleton_mixed_victim_count"],
                matrix["rank_over_Q"],
                matrix["nullity_over_Q"],
                row["toric_circuit_count_by_binomial_degree"],
            )
            self.assertTrue(
                row["all_seed_supported_mixed_terms_are_singletons"]
            )
            for relation in row[
                "support_minimal_toric_circuits"
            ]:
                coefficients = relation[
                    "primitive_coefficients"
                ]
                self.assertEqual(
                    sum(value for value in coefficients if value > 0),
                    relation["binomial_degree"],
                )
                self.assertEqual(
                    -sum(value for value in coefficients if value < 0),
                    relation["binomial_degree"],
                )
        self.assertEqual(observed, expected)

    def test_claim_boundary_is_fail_closed(self):
        self.assertEqual(
            self.payload["claim_boundary"],
            census.CLAIM_BOUNDARY,
        )
        self.assertTrue(
            all(
                value is False
                for value in self.payload["claim_boundary"].values()
            )
        )
        self.assertTrue(
            all(self.payload["exact_checks"].values())
        )

    def test_committed_bundle_round_trips_and_corruption_fails(self):
        census.verify_bundle(BUNDLE)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            payload = json.loads(
                (copied / "census.json").read_text("utf-8")
            )
            payload["counts"]["S10_x_S3_orbits"] = 2
            (copied / "census.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = json.loads(
                (copied / "manifest.json").read_text("utf-8")
            )
            manifest["bundle_files"]["census.json"] = (
                census._file_record(copied / "census.json")
            )
            (copied / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(
                census.KrennN10PairwiseHamiltonianError
            ):
                census.verify_bundle(copied)

    def test_payload_mutation_does_not_poison_cached_recomputation(self):
        mutated = copy.deepcopy(self.payload)
        mutated["orbits"][0]["ordered_triple_orbit_size"] = 1
        fresh = census.build_census_payload()
        self.assertEqual(
            fresh["orbits"][0]["ordered_triple_orbit_size"],
            5_443_200,
        )


if __name__ == "__main__":
    unittest.main()
