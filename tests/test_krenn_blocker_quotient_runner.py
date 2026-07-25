from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.krenn_quantum_graph.blocker_quotient_reconnaissance import (
    COMPLETION_MARKER,
    PARSE_MARKER,
    TRANSCRIPT_SCHEMA,
    chart_sha256,
    deterministic_k5_blocker_chart,
)
from experiments.krenn_quantum_graph.blocker_quotient_runner import (
    CPUS,
    DOCKER_IMAGE_ID,
    MAXIMUM_WORKERS,
    MODULAR_PROBE,
    PROBES,
    RATIONAL_PROBE,
    KrennBlockerQuotientRunnerError,
    BlockerQuotientProbe,
    docker_argv,
    prepare_probe_input,
    replay_probe,
    run_probe,
)


class KrennBlockerQuotientRunnerTest(unittest.TestCase):
    def test_predeclared_resource_plan_is_bounded_and_serial(self):
        self.assertEqual(PROBES, (MODULAR_PROBE, RATIONAL_PROBE))
        self.assertEqual(MAXIMUM_WORKERS, 1)
        self.assertEqual(CPUS, 2)
        self.assertLessEqual(CPUS * MAXIMUM_WORKERS, 12)
        self.assertEqual(
            (
                MODULAR_PROBE.characteristic,
                MODULAR_PROBE.engine_budget_seconds,
                MODULAR_PROBE.memory_gib,
            ),
            (31, 600, 4),
        )
        self.assertEqual(
            (
                RATIONAL_PROBE.characteristic,
                RATIONAL_PROBE.engine_budget_seconds,
                RATIONAL_PROBE.memory_gib,
            ),
            (0, 1800, 8),
        )

    def test_inputs_and_docker_argv_are_deterministic_and_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            first = prepare_probe_input(MODULAR_PROBE, scratch)
            second = prepare_probe_input(MODULAR_PROBE, scratch)
            self.assertEqual(first, second)
            self.assertEqual(first["bytes"], 2005)
            self.assertEqual(
                first["sha256"],
                "a991811ae7a2d21b6f6675b366ffec0bf0d8b564595011ea"
                "c2f4abf7c5c6f355",
            )
            argv = docker_argv(MODULAR_PROBE, scratch)
            self.assertIn(DOCKER_IMAGE_ID, argv)
            self.assertIn("none", argv)
            self.assertIn("600s", argv)
            self.assertIn("4g", argv)
            self.assertIn("--cpus", argv)
            self.assertNotIn("hodgepodge", " ".join(argv))

    def test_timeout_receipt_round_trips_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(
                    f"{PARSE_MARKER}\n".encode("ascii")
                )
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 124)

            with (
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_runner.inspect_engine",
                    return_value=DOCKER_IMAGE_ID,
                ),
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_runner.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                receipt = run_probe(MODULAR_PROBE, scratch)
            self.assertEqual(
                receipt["outcome"]["status"], "timeout-after-parse"
            )
            self.assertFalse(
                receipt["claim_boundary"]["timeout_or_miss_is_proof"]
            )
            self.assertEqual(
                replay_probe(MODULAR_PROBE, scratch), receipt
            )
            self.assertFalse(
                (scratch / "runs" / ".campaign.lock").exists()
            )
            stdout = (
                scratch
                / "runs"
                / receipt["stdout"]["path"]
            )
            stdout.write_text(
                f"{PARSE_MARKER}\ncorrupt\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                KrennBlockerQuotientRunnerError, "hash"
            ):
                replay_probe(MODULAR_PROBE, scratch)

    def test_completed_rational_receipt_binds_metadata_and_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            chart = deterministic_k5_blocker_chart()
            transcript = "\n".join(
                (
                    PARSE_MARKER,
                    f"transcript_schema={TRANSCRIPT_SCHEMA}",
                    f"chart_sha256={chart_sha256(chart)}",
                    "characteristic=0",
                    "algorithm=slimgb",
                    "variable_count=10",
                    "timer_ticks=3",
                    "basis_size=1",
                    "unit_ideal=1",
                    "krull_dimension=-1",
                    COMPLETION_MARKER,
                    "",
                )
            ).encode("ascii")

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(transcript)
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 0)

            with (
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_runner.inspect_engine",
                    return_value=DOCKER_IMAGE_ID,
                ),
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_runner.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                receipt = run_probe(RATIONAL_PROBE, scratch)
            self.assertEqual(
                receipt["outcome"]["status"], "completed-unit-ideal"
            )
            self.assertTrue(
                all(
                    value is False
                    for value in receipt["claim_boundary"].values()
                )
            )
            receipt_path = (
                scratch
                / "runs"
                / f"{RATIONAL_PROBE.key}.receipt.json"
            )
            payload = json.loads(receipt_path.read_text("utf-8"))
            payload["engine"]["image"] = "tampered"
            receipt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennBlockerQuotientRunnerError, "engine metadata"
            ):
                replay_probe(RATIONAL_PROBE, scratch)

    def test_existing_campaign_lock_blocks_a_second_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            runs = scratch / "runs"
            runs.mkdir(parents=True)
            (runs / ".campaign.lock").write_text(
                "another process\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                KrennBlockerQuotientRunnerError, "another quotient"
            ):
                run_probe(MODULAR_PROBE, scratch)

    def test_backend_stdout_must_be_strict_utf8(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(b"\xff")
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 0)

            with (
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_runner.inspect_engine",
                    return_value=DOCKER_IMAGE_ID,
                ),
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_runner.subprocess.run",
                    side_effect=fake_run,
                ),
                self.assertRaisesRegex(
                    KrennBlockerQuotientRunnerError, "strict UTF-8"
                ),
            ):
                run_probe(MODULAR_PROBE, scratch)
            self.assertFalse(
                (scratch / "runs" / ".campaign.lock").exists()
            )

    def test_probe_validation_fails_closed(self):
        invalid_rows = (
            {
                "key": "../escape",
                "characteristic": 31,
                "algorithm": "slimgb",
                "engine_budget_seconds": 600,
                "memory_gib": 4,
            },
            {
                "key": "bad_field",
                "characteristic": 1009,
                "algorithm": "slimgb",
                "engine_budget_seconds": 600,
                "memory_gib": 4,
            },
            {
                "key": "too_long",
                "characteristic": 0,
                "algorithm": "slimgb",
                "engine_budget_seconds": 1801,
                "memory_gib": 8,
            },
            {
                "key": "too_large",
                "characteristic": 0,
                "algorithm": "slimgb",
                "engine_budget_seconds": 1800,
                "memory_gib": 9,
            },
        )
        for row in invalid_rows:
            with self.subTest(key=row["key"]):
                with self.assertRaises(
                    KrennBlockerQuotientRunnerError
                ):
                    BlockerQuotientProbe(**row)


if __name__ == "__main__":
    unittest.main()
