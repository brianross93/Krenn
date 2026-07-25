import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.krenn_quantum_graph import (
    star_pivot_affine_cas_runner as affine_runner,
)
from experiments.krenn_quantum_graph.localized_chart_cas_runner import (
    CPUS_PER_PROBE,
    DOCKER_IMAGE_ID,
    ENGINE_BUDGET_SECONDS,
    KrennLocalizedCASRunnerError,
    MEMORY_GIB_PER_PROBE,
    OUTPUT_SUBDIRECTORY,
    classify_output,
    docker_argv,
    run_probe,
)
from experiments.krenn_quantum_graph.star_pivot_affine_cas_runner import (
    ALGORITHM,
    CHARACTERISTIC,
    CHECKPOINT_NAME,
    EXPECTED_GENERATOR_SOURCE_BYTES,
    EXPECTED_GENERATOR_SOURCE_SHA256,
    EXPECTED_INPUTS,
    EXPECTED_RETAINED_PRESENTATION_SHA256,
    GENERATOR_SOURCE_PATH,
    MAXIMUM_WORKERS,
    PRESENTATION_KIND,
    STAR_PIVOT_AFFINE_PROBES,
    prepare_star_pivot_affine_inputs,
    replay_checkpoint,
    run_star_pivot_affine_probe_suite,
    validate_generator_source,
)
from experiments.krenn_quantum_graph.star_pivot_affine_slices import (
    PIVOT_ORBITS,
    singular_star_pivot_affine_slice_script,
    star_pivot_affine_presentation,
)


class KrennStarPivotAffineCASRunnerTest(unittest.TestCase):
    def test_three_retained_orbit_probes_are_byte_exact(self):
        self.assertEqual(PRESENTATION_KIND, "retained")
        self.assertEqual(CHARACTERISTIC, 31)
        self.assertEqual(ALGORITHM, "slimgb")
        self.assertEqual(len(STAR_PIVOT_AFFINE_PROBES), 3)
        self.assertEqual(
            tuple(orbit.equality_pattern for orbit in PIVOT_ORBITS),
            ("all-same", "exactly-two-same", "all-distinct"),
        )
        self.assertEqual(
            len({probe.key for probe in STAR_PIVOT_AFFINE_PROBES}),
            3,
        )
        for probe, orbit in zip(
            STAR_PIVOT_AFFINE_PROBES, PIVOT_ORBITS, strict=True
        ):
            presentation = star_pivot_affine_presentation(
                orbit.index, PRESENTATION_KIND
            )
            self.assertEqual(presentation.maximum_degree, 3)
            self.assertEqual(
                presentation.fingerprint(),
                EXPECTED_RETAINED_PRESENTATION_SHA256[orbit.index],
            )
            script = singular_star_pivot_affine_slice_script(
                orbit.index,
                PRESENTATION_KIND,
                characteristic=CHARACTERISTIC,
                algorithm=ALGORITHM,
            ).encode("utf-8")
            self.assertEqual(len(script), EXPECTED_INPUTS[orbit.index]["bytes"])
            self.assertEqual(
                hashlib.sha256(script).hexdigest(),
                EXPECTED_INPUTS[orbit.index]["sha256"],
            )
            text = script.decode("utf-8")
            self.assertEqual(text.count(probe.parse_marker), 1)
            self.assertEqual(text.count(probe.completion_marker), 1)
            self.assertIn("No Rabinowitsch variable", text)
            self.assertNotIn("qz-1", text)
            self.assertNotIn("sat(", text)

    def test_source_fingerprint_is_pinned_and_corruption_fails(self):
        record = validate_generator_source()
        self.assertEqual(record["bytes"], EXPECTED_GENERATOR_SOURCE_BYTES)
        self.assertEqual(
            record["sha256"], EXPECTED_GENERATOR_SOURCE_SHA256
        )
        self.assertEqual(
            Path(record["path"]), GENERATOR_SOURCE_PATH
        )
        with tempfile.TemporaryDirectory() as temporary:
            source_copy = Path(temporary) / "source.py"
            source_copy.write_bytes(GENERATOR_SOURCE_PATH.read_bytes())
            self.assertEqual(
                validate_generator_source(source_copy)["sha256"],
                EXPECTED_GENERATOR_SOURCE_SHA256,
            )
            source_copy.write_bytes(
                source_copy.read_bytes() + b"\n# corruption\n"
            )
            with self.assertRaises(KrennLocalizedCASRunnerError):
                validate_generator_source(source_copy)

    def test_inputs_prepare_idempotently_and_refuse_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            first = prepare_star_pivot_affine_inputs(scratch)
            second = prepare_star_pivot_affine_inputs(scratch)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            for row, probe in zip(
                first, STAR_PIVOT_AFFINE_PROBES, strict=True
            ):
                self.assertEqual(row["bytes"], probe.input_bytes)
                self.assertEqual(row["sha256"], probe.input_sha256)
            path = Path(first[0]["path"])
            path.write_bytes(path.read_bytes() + b"// changed\n")
            with self.assertRaises(KrennLocalizedCASRunnerError):
                prepare_star_pivot_affine_inputs(scratch)

    def test_each_probe_has_real_bounded_offline_docker_limits(self):
        self.assertEqual(ENGINE_BUDGET_SECONDS, 600)
        self.assertEqual(CPUS_PER_PROBE, 2)
        self.assertEqual(MEMORY_GIB_PER_PROBE, 8)
        self.assertLessEqual(
            MAXIMUM_WORKERS * CPUS_PER_PROBE, 12
        )
        for probe in STAR_PIVOT_AFFINE_PROBES:
            argv = docker_argv(probe)
            self.assertIn("--pull", argv)
            self.assertIn("never", argv)
            self.assertIn("--network", argv)
            self.assertIn("none", argv)
            self.assertIn("--cpus", argv)
            self.assertIn("2", argv)
            self.assertIn("--memory", argv)
            self.assertIn("8g", argv)
            self.assertIn("600s", argv)
            self.assertIn("--kill-after=15s", argv)
            self.assertIn(DOCKER_IMAGE_ID, argv)

    def test_output_classification_keeps_modular_claims_fail_closed(self):
        probe = STAR_PIVOT_AFFINE_PROBES[0]
        completed = classify_output(
            probe,
            return_code=0,
            stdout_text=(
                f"{probe.parse_marker}\n"
                f"{probe.completion_marker}\n"
                "timer_ticks=9\nbasis_size=11\nunit_ideal=1\n"
            ),
        )
        self.assertEqual(completed["status"], "completed")
        self.assertFalse(
            completed["claim_boundary"][
                "positive_characteristic_result_is_exact_Q_proof"
            ]
        )
        timeout = classify_output(
            probe,
            return_code=124,
            stdout_text=f"{probe.parse_marker}\n",
        )
        self.assertEqual(timeout["status"], "timeout-after-parse")
        self.assertFalse(
            timeout["claim_boundary"]["timeout_is_a_chart_decision"]
        )
        duplicated = classify_output(
            probe,
            return_code=0,
            stdout_text=(
                f"{probe.parse_marker}\n{probe.parse_marker}\n"
                f"{probe.completion_marker}\n"
                "timer_ticks=9\nbasis_size=11\nunit_ideal=1\n"
            ),
        )
        self.assertEqual(duplicated["status"], "failed-before-parse")

    def test_atomic_checkpoint_replays_receipts_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            prepare_star_pivot_affine_inputs(scratch)
            output = scratch / OUTPUT_SUBDIRECTORY
            output.mkdir()
            probe = STAR_PIVOT_AFFINE_PROBES[0]

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(
                    (
                        f"{probe.parse_marker}\n"
                        f"{probe.completion_marker}\n"
                        "timer_ticks=7\n"
                        "basis_size=13\n"
                        "unit_ideal=0\n"
                    ).encode("utf-8")
                )
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 0)

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
            checkpoint = affine_runner._write_checkpoint(
                scratch, {probe.key: receipt}
            )
            self.assertEqual(
                len(checkpoint["completed_receipts"]), 1
            )
            self.assertEqual(
                replay_checkpoint(scratch), checkpoint
            )
            self.assertFalse(
                (output / f".{CHECKPOINT_NAME}.tmp").exists()
            )
            stdout_path = output / receipt["stdout"]["path"]
            stdout_path.write_text(
                stdout_path.read_text(encoding="utf-8")
                + "corrupted\n",
                encoding="utf-8",
            )
            with self.assertRaises(KrennLocalizedCASRunnerError):
                replay_checkpoint(scratch)

    def test_worker_validation_precedes_all_external_execution(self):
        for workers in (0, MAXIMUM_WORKERS + 1, True):
            with patch(
                "experiments.krenn_quantum_graph."
                "star_pivot_affine_cas_runner.inspect_docker_image"
            ) as inspect:
                with self.assertRaises(KrennLocalizedCASRunnerError):
                    run_star_pivot_affine_probe_suite(
                        Path("unused"), workers=workers
                    )
                inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
