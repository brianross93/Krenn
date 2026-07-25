import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.krenn_quantum_graph.localized_chart_cas_runner import (
    CPUS_PER_PROBE,
    DOCKER_IMAGE_ID,
    ENGINE_BUDGET_SECONDS,
    KrennLocalizedCASRunnerError,
    MAXIMUM_WORKERS,
    MEMORY_GIB_PER_PROBE,
    OUTPUT_SUBDIRECTORY,
    PROBES,
    SingularProbe,
    classify_output,
    docker_argv,
    replay_probe_receipt,
    run_probe,
    run_probe_suite,
)


class KrennLocalizedChartCASRunnerTest(unittest.TestCase):
    def test_all_seven_probes_have_real_ten_minute_resource_caps(self):
        self.assertEqual(len(PROBES), 7)
        self.assertEqual(len({probe.key for probe in PROBES}), 7)
        self.assertEqual(ENGINE_BUDGET_SECONDS, 600)
        self.assertEqual(CPUS_PER_PROBE, 2)
        self.assertEqual(MEMORY_GIB_PER_PROBE, 8)
        self.assertLessEqual(
            MAXIMUM_WORKERS * CPUS_PER_PROBE, 12
        )
        for probe in PROBES:
            argv = docker_argv(probe)
            self.assertIn("none", argv)
            self.assertIn("600s", argv)
            self.assertIn("--rm", argv)
            self.assertIn("--cpus", argv)
            self.assertIn("--memory", argv)
            self.assertIn("--kill-after=15s", argv)
            self.assertIn(DOCKER_IMAGE_ID, argv)

    def test_marker_classification_is_fail_closed(self):
        probe = PROBES[0]
        timeout = classify_output(
            probe,
            return_code=124,
            stdout_text=f"{probe.parse_marker}\n",
        )
        self.assertEqual(timeout["status"], "timeout-after-parse")
        self.assertFalse(
            timeout["claim_boundary"]["timeout_is_a_chart_decision"]
        )

        completed = classify_output(
            probe,
            return_code=0,
            stdout_text=(
                f"{probe.parse_marker}\n"
                f"{probe.completion_marker}\n"
                "timer_ticks=12\nbasis_size=7\nunit_ideal=0\n"
            ),
        )
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(
            completed["selected_singular_output"]["unit_ideal"], "0"
        )
        self.assertFalse(
            completed["claim_boundary"][
                "positive_characteristic_result_is_exact_Q_proof"
            ]
        )
        incomplete = classify_output(
            probe,
            return_code=0,
            stdout_text=(
                f"{probe.parse_marker}\n"
                f"{probe.completion_marker}\n"
            ),
        )
        self.assertEqual(
            incomplete["status"], "invalid-completion-receipt"
        )
        duplicated = classify_output(
            probe,
            return_code=0,
            stdout_text=(
                f"{probe.parse_marker}\n{probe.parse_marker}\n"
                f"{probe.completion_marker}\n"
                "timer_ticks=12\nbasis_size=7\nunit_ideal=0\n"
            ),
        )
        self.assertEqual(duplicated["status"], "failed-before-parse")

        failed = classify_output(
            probe, return_code=1, stdout_text=""
        )
        self.assertEqual(failed["status"], "failed-before-parse")

    def test_receipt_round_trip_and_log_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            input_directory = scratch / "inputs"
            input_directory.mkdir()
            script_path = input_directory / "probe.sing"
            script_bytes = b'print("PARSE_OK");\n'
            script_path.write_bytes(script_bytes)
            probe = SingularProbe(
                key="synthetic",
                ideal="synthetic ideal",
                input_subdirectory="inputs",
                script_name=script_path.name,
                algorithm="std",
                characteristic=31,
                input_bytes=len(script_bytes),
                input_sha256=hashlib.sha256(
                    script_bytes
                ).hexdigest(),
                parse_marker="PARSE_OK",
                completion_marker="DONE",
            )
            output = scratch / OUTPUT_SUBDIRECTORY
            output.mkdir()

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(b"PARSE_OK\n")
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 124)

            with patch(
                "experiments.krenn_quantum_graph."
                "localized_chart_cas_runner.subprocess.run",
                side_effect=fake_run,
            ):
                receipt = run_probe(
                    probe,
                    scratch,
                    DOCKER_IMAGE_ID,
                    resume=False,
                )
            self.assertEqual(
                receipt["outcome"]["status"], "timeout-after-parse"
            )
            self.assertEqual(
                replay_probe_receipt(
                    probe, scratch, DOCKER_IMAGE_ID
                ),
                receipt,
            )
            (output / receipt["stdout"]["path"]).write_text(
                "PARSE_OK\ncorrupted\n", encoding="utf-8"
            )
            with self.assertRaises(KrennLocalizedCASRunnerError):
                replay_probe_receipt(
                    probe, scratch, DOCKER_IMAGE_ID
                )

    def test_worker_validation_precedes_external_execution(self):
        for workers in (0, MAXIMUM_WORKERS + 1, True):
            with self.assertRaises(KrennLocalizedCASRunnerError):
                run_probe_suite(Path("unused"), workers=workers)


if __name__ == "__main__":
    unittest.main()
