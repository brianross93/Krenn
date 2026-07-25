from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.localized_chart_cas_runner import (
    ENGINE_BUDGET_SECONDS,
    KrennLocalizedCASRunnerError,
)
from experiments.krenn_quantum_graph.localized_chart_sparse_cas_runner import (
    CHARACTERISTIC,
    EXPECTED_INPUTS,
    MAXIMUM_WORKERS,
    SPARSE_PROBES,
    _script_bytes,
    prepare_sparse_inputs,
    run_sparse_probe_suite,
)


class KrennLocalizedChartSparseCASRunnerTest(unittest.TestCase):
    def test_two_exact_p1009_inputs_have_full_ten_minute_budgets(self):
        self.assertEqual(CHARACTERISTIC, 1_009)
        self.assertEqual(ENGINE_BUDGET_SECONDS, 600)
        self.assertEqual(len(SPARSE_PROBES), 2)
        self.assertEqual(
            [probe.characteristic for probe in SPARSE_PROBES],
            [1_009, 1_009],
        )
        self.assertEqual(
            [probe.algorithm for probe in SPARSE_PROBES],
            ["std", "std"],
        )
        for ambient, probe in zip((11, 29), SPARSE_PROBES, strict=True):
            self.assertEqual(
                len(_script_bytes(ambient)),
                EXPECTED_INPUTS[ambient]["bytes"],
            )
            self.assertIn(
                "TWO_PIVOT_ELIMINATION_PARSE_OK",
                probe.parse_marker,
            )
            text = _script_bytes(ambient).decode("utf-8")
            self.assertIn("(lp(2),dp(127));", text)
            self.assertIn("ideal G=std(I);", text)

    def test_generated_inputs_round_trip_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = prepare_sparse_inputs(root)
            second = prepare_sparse_inputs(root)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)
            changed = Path(first[0]["path"])
            changed.write_text("corrupted\n", encoding="utf-8")
            with self.assertRaises(KrennLocalizedCASRunnerError):
                prepare_sparse_inputs(root)

    def test_worker_validation_precedes_external_execution(self):
        for workers in (0, MAXIMUM_WORKERS + 1, True):
            with self.assertRaises(KrennLocalizedCASRunnerError):
                run_sparse_probe_suite(
                    Path("unused"), workers=workers
                )


if __name__ == "__main__":
    unittest.main()
