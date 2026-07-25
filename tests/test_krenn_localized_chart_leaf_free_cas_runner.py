from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.localized_chart_cas_runner import (
    ENGINE_BUDGET_SECONDS,
    KrennLocalizedCASRunnerError,
    classify_output,
)
from experiments.krenn_quantum_graph.localized_chart_leaf_free_cas_runner import (
    ALGORITHM,
    CHARACTERISTIC,
    EXPECTED_INPUTS,
    LEAF_FREE_PROBES,
    MAXIMUM_WORKERS,
    _script_bytes,
    prepare_leaf_free_inputs,
    run_leaf_free_probe_suite,
)


class KrennLocalizedChartLeafFreeCASRunnerTest(unittest.TestCase):
    def test_two_exact_f31_inputs_have_full_ten_minute_budgets(self):
        self.assertEqual(CHARACTERISTIC, 31)
        self.assertEqual(ALGORITHM, "sat-A0")
        self.assertEqual(ENGINE_BUDGET_SECONDS, 600)
        self.assertEqual(len(LEAF_FREE_PROBES), 2)
        for ambient, probe in zip(
            (11, 29), LEAF_FREE_PROBES, strict=True
        ):
            payload = _script_bytes(ambient)
            self.assertEqual(
                len(payload), EXPECTED_INPUTS[ambient]["bytes"]
            )
            text = payload.decode("utf-8")
            self.assertIn(
                f"KRENN_LEAF_FREE_{ambient}_PARSE_OK", text
            )
            self.assertIn("ideal S1=sat(S0,H0);", text)
            self.assertNotIn("ideal S2=sat(S1,H1);", text)
            self.assertEqual(probe.algorithm, "sat-A0")

    def test_generated_inputs_round_trip_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = prepare_leaf_free_inputs(root)
            second = prepare_leaf_free_inputs(root)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)
            changed = Path(first[0]["path"])
            changed.write_text("corrupted\n", encoding="utf-8")
            with self.assertRaises(KrennLocalizedCASRunnerError):
                prepare_leaf_free_inputs(root)

    def test_synthetic_completion_and_timeout_classify_fail_closed(self):
        probe = LEAF_FREE_PROBES[0]
        completed = "\n".join((
            probe.parse_marker,
            probe.completion_marker,
            "basis_size=7",
            "timer_ticks=9",
            "unit_ideal=0",
        ))
        outcome = classify_output(
            probe, return_code=0, stdout_text=completed
        )
        self.assertEqual(outcome["status"], "completed")
        self.assertFalse(
            outcome["claim_boundary"][
                "positive_characteristic_result_is_exact_Q_proof"
            ]
        )
        timed_out = classify_output(
            probe,
            return_code=124,
            stdout_text=probe.parse_marker,
        )
        self.assertEqual(
            timed_out["status"], "timeout-after-parse"
        )
        self.assertFalse(
            timed_out["claim_boundary"][
                "timeout_is_a_chart_decision"
            ]
        )

    def test_worker_validation_precedes_external_execution(self):
        for workers in (0, MAXIMUM_WORKERS + 1, True):
            with self.assertRaises(KrennLocalizedCASRunnerError):
                run_leaf_free_probe_suite(
                    Path("unused"), workers=workers
                )


if __name__ == "__main__":
    unittest.main()
