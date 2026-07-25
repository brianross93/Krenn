import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph import n8_base_support_cnf as gate


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "results"
    / "krenn_quantum_graph"
    / "n8_d3_base_support_cnf_gate"
)


class KrennN8BaseSupportCNFTest(unittest.TestCase):
    def test_variable_blocks_and_exact_formula_counts(self):
        specification = gate.formula_specification()
        self.assertEqual(
            specification["variable_blocks"],
            {
                "edge": {"one_based_start": 1, "count": 252},
                "term": {"one_based_start": 253, "count": 688_905},
                "star": {
                    "one_based_start": 689_158,
                    "count": 168,
                },
                "pair": {
                    "one_based_start": 689_326,
                    "count": 1_464,
                },
                "full_column": {
                    "one_based_start": 690_790,
                    "count": 672,
                },
            },
        )
        self.assertEqual(
            specification["counts"]["variables"],
            691_461,
        )
        self.assertEqual(
            specification["counts"]["common_clauses"],
            4_141_782,
        )
        self.assertEqual(
            specification["counts"]["branch_clauses"],
            4_141_794,
        )
        self.assertEqual(
            specification["counts"]["branch_literals"],
            81_279_594,
        )
        self.assertEqual(
            sum(gate.FAMILY_CLAUSE_COUNTS.values()),
            gate.BRANCH_CLAUSE_COUNT,
        )
        self.assertEqual(
            sum(gate.FAMILY_LITERAL_COUNTS.values()),
            gate.BRANCH_LITERAL_COUNT,
        )

    def test_boolean_gate_clause_expansions_are_exact(self):
        self.assertEqual(
            [
                record.literals
                for record in gate._and_definition(
                    "test", 10, (1, -2, 3)
                )
            ],
            [
                (-10, 1),
                (-10, -2),
                (-10, 3),
                (10, -1, 2, -3),
            ],
        )
        self.assertEqual(
            [
                record.literals
                for record in gate._or_definition(
                    "test", 10, (1, 2, 3)
                )
            ],
            [
                (10, -1),
                (10, -2),
                (10, -3),
                (-10, 1, 2, 3),
            ],
        )
        self.assertEqual(
            [
                record.literals
                for record in gate._not_definition("test", 10, 2)
            ],
            [(-10, -2), (10, 2)],
        )

    def test_first_term_definition_uses_four_exact_edge_slots(self):
        records = []
        iterator = gate.iter_common_clause_records()
        for _index in range(5):
            records.append(next(iterator))
        term = gate.term_variable(0, 0)
        edges = gate._term_edge_variables(
            (0,) * gate.N,
            gate._primary_matchings()[0],
        )
        self.assertEqual(len(edges), 4)
        self.assertEqual(
            [record.family for record in records],
            ["term_definition"] * 5,
        )
        self.assertEqual(
            [record.literals for record in records],
            [
                (-term, edges[0]),
                (-term, edges[1]),
                (-term, edges[2]),
                (-term, edges[3]),
                (term, -edges[0], -edges[1], -edges[2], -edges[3]),
            ],
        )

    def test_all_31_branch_anchor_sets_are_distinct_and_exact(self):
        representatives = gate.branch_representatives()
        anchors = tuple(
            gate.branch_anchor_variables(index)
            for index in range(len(representatives))
        )
        self.assertEqual(len(representatives), 31)
        self.assertEqual(len(set(representatives)), 31)
        self.assertTrue(
            all(
                len(row) == 12
                and len(set(row)) == 12
                and tuple(sorted(row)) == row
                for row in anchors
            )
        )
        self.assertEqual(
            representatives[0],
            (0, 0, 0),
        )
        self.assertEqual(
            anchors[0],
            (1, 5, 9, 118, 122, 126, 199, 203, 207, 244, 248, 252),
        )

    def test_primary_and_independent_matching_tables_agree(self):
        specification = gate.formula_specification()
        self.assertEqual(len(gate._primary_matchings()), 105)
        self.assertEqual(len(gate._independent_matchings()), 105)
        self.assertEqual(
            set(gate._primary_matchings()),
            set(gate._independent_matchings()),
        )
        self.assertTrue(
            specification["matching_tables"]["sets_equal"]
        )
        for key in ("primary_sha256", "independent_sha256"):
            self.assertRegex(
                specification["matching_tables"][key],
                r"\A[0-9a-f]{64}\Z",
            )

    def test_claim_boundary_excludes_every_krenn_conclusion(self):
        self.assertTrue(
            all(value is False for value in gate.CLAIM_BOUNDARY.values())
        )
        self.assertEqual(
            set(gate.EXCLUDED_CUT_FAMILIES),
            {
                "closed_target_orbit",
                "equation_pattern_no_good",
                "exact_support_no_good",
                "generated_branch_tail",
                "laurent_identity_no_good",
            },
        )
        self.assertFalse(
            gate.CLAIM_BOUNDARY[
                "base_cnf_sat_model_is_complex_weight_witness"
            ]
        )

    def test_dense_all_edge_support_is_not_a_model_of_full_prefix(self):
        with self.assertRaisesRegex(
            gate.KrennN8BaseSupportCNFError,
            "full_column_assertions",
        ):
            gate.derive_assignment_and_semantic_report(
                range(1, gate.EDGE_VARIABLE_COUNT + 1),
                branch_index=0,
            )

    def test_pinned_solver_is_exact_when_available(self):
        if not gate.DEFAULT_SOLVER.is_file():
            self.skipTest("pinned local CaDiCaL is unavailable")
        record = gate._validate_solver(gate.DEFAULT_SOLVER)
        self.assertEqual(record["version"], "2.1.2")
        self.assertEqual(
            record["sha256"],
            gate.SOLVER_SHA256,
        )

    @unittest.skipUnless(
        os.environ.get("KRENN_RUN_LONG_TESTS") == "1",
        "set KRENN_RUN_LONG_TESTS=1 to stream the 4.14M-clause DIMACS",
    )
    def test_long_dimacs_generation_and_replay(self):
        certificate = json.loads(
            (BUNDLE / "certificate.json").read_text("utf-8")
        )
        branch_index = certificate["terminal_branch"]["branch_index"]
        active_edges = certificate["compact_model"][
            "active_edge_variables"
        ]
        assignment, _report = (
            gate.derive_assignment_and_semantic_report(
                active_edges,
                branch_index=branch_index,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "branch.cnf"
            receipt = gate.write_dimacs(
                path,
                branch_index=branch_index,
            )
            replay = gate.verify_dimacs_against_generator_and_model(
                path,
                branch_index=branch_index,
                assignment=assignment,
            )
            self.assertEqual(
                receipt["sha256"],
                replay["sha256"],
            )

    @unittest.skipUnless(
        (BUNDLE / "certificate.json").is_file(),
        "the deterministic SAT campaign has not emitted its bundle",
    )
    def test_committed_bundle_replays_and_corruption_fails(self):
        gate.verify_bundle(BUNDLE)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            certificate_path = copied / "certificate.json"
            certificate = json.loads(
                certificate_path.read_text("utf-8")
            )
            certificate["compact_model"]["active_edge_variables"].pop()
            certificate_path.write_bytes(
                gate._canonical_json_bytes(certificate)
            )
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest["bundle_files"]["certificate.json"] = (
                gate._file_record(certificate_path)
            )
            manifest_path.write_bytes(
                gate._canonical_json_bytes(manifest)
            )
            with self.assertRaises(
                gate.KrennN8BaseSupportCNFError
            ):
                gate.verify_bundle(copied)

    @unittest.skipUnless(
        (BUNDLE / "certificate.json").is_file(),
        "the deterministic SAT campaign has not emitted its bundle",
    )
    def test_refreshed_hash_typed_and_claim_corruptions_fail(self):
        corruptions = (
            (
                "representative bool aliases integer zero",
                ("terminal_branch", "representative_matching_indices", 0),
                False,
            ),
            (
                "CNF branch bool aliases integer zero",
                ("cnf_receipt", "branch_index"),
                False,
            ),
            (
                "solver seed bool aliases integer zero",
                ("solver_receipt", "seed"),
                False,
            ),
            (
                "attempted branch bool aliases integer zero",
                ("terminal_logic", "branches_attempted", 0),
                False,
            ),
            (
                "promoted Krenn claim",
                (
                    "claim_boundary",
                    "base_cnf_sat_model_is_complex_weight_witness",
                ),
                True,
            ),
            (
                "altered terminal conclusion",
                ("terminal_logic", "conclusion"),
                "Krenn membership decided.",
            ),
            (
                "forged solver hash",
                ("solver_receipt", "solver", "sha256"),
                "0" * 64,
            ),
        )
        for label, path, replacement in corruptions:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary:
                    copied = Path(temporary) / "bundle"
                    shutil.copytree(BUNDLE, copied)
                    certificate_path = copied / "certificate.json"
                    certificate = json.loads(
                        certificate_path.read_text("utf-8")
                    )
                    cursor = certificate
                    for key in path[:-1]:
                        cursor = cursor[key]
                    cursor[path[-1]] = replacement
                    certificate_path.write_bytes(
                        gate._canonical_json_bytes(certificate)
                    )
                    manifest_path = copied / "manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text("utf-8")
                    )
                    manifest["bundle_files"]["certificate.json"] = (
                        gate._file_record(certificate_path)
                    )
                    manifest_path.write_bytes(
                        gate._canonical_json_bytes(manifest)
                    )
                    with self.assertRaises(
                        gate.KrennN8BaseSupportCNFError
                    ):
                        gate.verify_bundle(copied)

    @unittest.skipUnless(
        (BUNDLE / "certificate.json").is_file(),
        "the deterministic SAT campaign has not emitted its bundle",
    )
    def test_malformed_manifest_submaps_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest["bundle_files"] = []
            manifest_path.write_bytes(
                gate._canonical_json_bytes(manifest)
            )
            with self.assertRaisesRegex(
                gate.KrennN8BaseSupportCNFError,
                "manifest schema or inventory",
            ):
                gate.verify_bundle(copied)


if __name__ == "__main__":
    unittest.main()
