import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.localized_chart_cas_runner import (
    CPUS_PER_PROBE,
    DOCKER_IMAGE,
    DOCKER_IMAGE_ID,
    ENGINE_BUDGET_SECONDS,
    MEMORY_GIB_PER_PROBE,
    RUNNER_SCHEMA,
    classify_output,
    docker_argv,
)
from experiments.krenn_quantum_graph.star_pivot_affine_cas_artifact import (
    ALL_FILES,
    BUNDLE_MANIFEST_FILE,
    SUMMARY_FILE,
    KrennStarPivotCASArtifactError,
    generate_star_pivot_cas_summary_bundle,
    summarize_completed_campaign,
    verify_star_pivot_cas_summary_bundle,
)
from experiments.krenn_quantum_graph.star_pivot_affine_cas_runner import (
    ALGORITHM,
    CHARACTERISTIC,
    CHECKPOINT_NAME,
    EXPECTED_RETAINED_PRESENTATION_SHA256,
    GENERATOR_SOURCE_PATH,
    INPUT_SUBDIRECTORY,
    MANIFEST_NAME,
    PRESENTATION_KIND,
    STAR_PIVOT_AFFINE_CHECKPOINT_SCHEMA,
    STAR_PIVOT_AFFINE_PROBES,
    STAR_PIVOT_AFFINE_RUNNER_SCHEMA,
)
from experiments.krenn_quantum_graph.star_pivot_affine_slices import (
    PIVOT_ORBITS,
    singular_star_pivot_affine_slice_script,
)


_RUNNER_BOUNDARY = {
    "positive_characteristic_result_is_exact_Q_proof": False,
    "timeout_is_a_slice_decision": False,
    "bounded_miss_is_a_nonexistence_proof": False,
    "raw_groebner_output_is_promoted_to_a_global_claim": False,
    "any_affine_slice_solved_over_Q_or_C": False,
    "finite_counterexample_found": False,
    "global_affine_membership_decided": False,
}


def _canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload) -> None:
    path.write_bytes(_canonical_json_bytes(payload))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_record(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _build_synthetic_campaign(parent: Path) -> Path:
    scratch = parent / "scratch"
    input_directory = scratch / INPUT_SUBDIRECTORY
    run_directory = scratch / "ten_minute_runs"
    input_directory.mkdir(parents=True)
    run_directory.mkdir()
    inputs = []
    for orbit, probe in zip(
        PIVOT_ORBITS, STAR_PIVOT_AFFINE_PROBES, strict=True
    ):
        payload = singular_star_pivot_affine_slice_script(
            orbit.index,
            PRESENTATION_KIND,
            characteristic=CHARACTERISTIC,
            algorithm=ALGORITHM,
        ).encode("utf-8")
        path = input_directory / probe.script_name
        path.write_bytes(payload)
        inputs.append({
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    starts = (
        "2026-07-25T06:00:00.000000+00:00",
        "2026-07-25T06:00:00.001000+00:00",
        "2026-07-25T06:10:01.000000+00:00",
    )
    elapsed = (600.10, 600.20, 600.30)
    receipts = []
    receipt_rows = []
    for probe, start, duration, input_row in zip(
        STAR_PIVOT_AFFINE_PROBES,
        starts,
        elapsed,
        inputs,
        strict=True,
    ):
        stdout_text = (
            "Synthetic Singular fixture\n"
            f"{probe.parse_marker}\n"
            "schema=krenn-n6-d3-star-pivot-affine-singular-v1\n"
            f"characteristic={CHARACTERISTIC}\n"
            f"algorithm={ALGORITHM}\n"
        )
        stdout_path = run_directory / f"{probe.key}.stdout.txt"
        stderr_path = run_directory / f"{probe.key}.stderr.txt"
        stdout_path.write_text(
            stdout_text, encoding="utf-8", newline="\n"
        )
        stderr_path.write_bytes(b"")
        outcome = classify_output(
            probe,
            return_code=124,
            stdout_text=stdout_text,
        )
        receipt = {
            "schema": RUNNER_SCHEMA,
            "probe": probe.to_dict(),
            "input_script": input_row,
            "docker": {
                "image": DOCKER_IMAGE,
                "image_id": DOCKER_IMAGE_ID,
                "argv": list(
                    docker_argv(probe, scratch, hardened=True)
                ),
                "network": "none",
            },
            "started_at_utc": start,
            "elapsed_seconds": duration,
            "stdout": _record(stdout_path),
            "stderr": _record(stderr_path),
            "outcome": outcome,
        }
        receipt_path = run_directory / f"{probe.key}.receipt.json"
        _write_json(receipt_path, receipt)
        row = {
            "key": probe.key,
            **_record(receipt_path),
            "elapsed_seconds": duration,
            "outcome": outcome,
        }
        receipts.append(receipt)
        receipt_rows.append(row)

    runner_path = Path(
        __import__(
            "experiments.krenn_quantum_graph."
            "star_pivot_affine_cas_runner",
            fromlist=["__file__"],
        ).__file__
    )
    generator_source = _source_record(Path(GENERATOR_SOURCE_PATH))
    runner_source = _source_record(runner_path)
    presentations = {
        str(key): value
        for key, value in EXPECTED_RETAINED_PRESENTATION_SHA256.items()
    }
    checkpoint = {
        "schema": STAR_PIVOT_AFFINE_CHECKPOINT_SCHEMA,
        "scratch_root": str(scratch),
        "generator_source": generator_source,
        "runner_source": runner_source,
        "presentation_kind": PRESENTATION_KIND,
        "characteristic": CHARACTERISTIC,
        "algorithm": ALGORITHM,
        "engine_budget_seconds_per_probe": ENGINE_BUDGET_SECONDS,
        "cpus_per_probe": CPUS_PER_PROBE,
        "memory_gib_per_probe": MEMORY_GIB_PER_PROBE,
        "docker_image": DOCKER_IMAGE,
        "docker_image_id": DOCKER_IMAGE_ID,
        "presentation_sha256": presentations,
        "inputs": inputs,
        "probes": [
            probe.to_dict() for probe in STAR_PIVOT_AFFINE_PROBES
        ],
        "random_seeds": [],
        "claim_boundary": _RUNNER_BOUNDARY,
        "completed_receipts": receipt_rows,
    }
    checkpoint_path = run_directory / CHECKPOINT_NAME
    _write_json(checkpoint_path, checkpoint)
    manifest = {
        "schema": STAR_PIVOT_AFFINE_RUNNER_SCHEMA,
        "scratch_root": str(scratch),
        "latest_invocation_workers": 2,
        "maximum_concurrent_probes_observed": 2,
        "maximum_concurrent_cpus": 4,
        "maximum_concurrent_memory_gib": 16,
        "generator_source": generator_source,
        "runner_source": runner_source,
        "presentation_kind": PRESENTATION_KIND,
        "eliminated_presentation_attempted": False,
        "characteristic": CHARACTERISTIC,
        "algorithm": ALGORITHM,
        "docker_image": DOCKER_IMAGE,
        "docker_image_id": DOCKER_IMAGE_ID,
        "engine_budget_seconds_per_probe": ENGINE_BUDGET_SECONDS,
        "cpus_per_probe": CPUS_PER_PROBE,
        "memory_gib_per_probe": MEMORY_GIB_PER_PROBE,
        "random_seeds": [],
        "presentation_sha256": presentations,
        "inputs": inputs,
        "receipts": receipt_rows,
        "checkpoint": {
            **_record(checkpoint_path),
            "completed_probe_count": 3,
        },
        "claim_boundary": _RUNNER_BOUNDARY,
    }
    _write_json(run_directory / MANIFEST_NAME, manifest)
    return run_directory


class KrennStarPivotCASArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.run_directory = _build_synthetic_campaign(
            cls.root / "base"
        )
        cls.bundle = cls.root / "bundle"
        cls.summary_path = generate_star_pivot_cas_summary_bundle(
            cls.run_directory, cls.bundle
        )
        cls.loaded = verify_star_pivot_cas_summary_bundle(
            cls.bundle
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def new_campaign(self, label: str) -> Path:
        return _build_synthetic_campaign(self.root / label)

    def copied_bundle(self, label: str) -> Path:
        destination = self.root / label
        shutil.copytree(self.bundle, destination)
        return destination

    def test_bundle_is_small_portable_and_round_trippable(self):
        self.assertEqual(self.summary_path, self.bundle / SUMMARY_FILE)
        self.assertEqual(
            {path.name for path in self.bundle.iterdir()},
            set(ALL_FILES),
        )
        self.assertTrue(self.loaded.valid)
        summary = self.loaded.summary
        self.assertTrue(
            summary["portable_evidence"][
                "self_contained_after_generation"
            ]
        )
        self.assertFalse(
            summary["portable_evidence"][
                "verification_requires_original_scratch_directory"
            ]
        )
        self.assertLess(self.summary_path.stat().st_size, 200_000)

    def test_three_parse_success_timeout_124_results_are_recorded(self):
        summary = self.loaded.summary
        aggregate = summary["aggregate"]
        self.assertEqual(aggregate["parse_success_count"], 3)
        self.assertEqual(
            aggregate["timeout_return_code_124_count"], 3
        )
        self.assertEqual(
            aggregate["standard_basis_completion_count"], 0
        )
        self.assertTrue(
            aggregate["all_elapsed_at_least_600_seconds"]
        )
        self.assertEqual(
            summary["campaign"]["characteristic"], 31
        )
        self.assertEqual(summary["campaign"]["algorithm"], "slimgb")
        self.assertEqual(
            summary["campaign"]["latest_invocation_workers"], 2
        )
        self.assertEqual(
            summary["campaign"]["resources"][
                "maximum_concurrent_cpus"
            ],
            4,
        )

    def test_claims_remain_fail_closed(self):
        boundary = self.loaded.summary["claim_boundary"]
        self.assertFalse(
            boundary["parse_success_is_standard_basis_completion"]
        )
        self.assertFalse(boundary["timeout_is_a_slice_decision"])
        self.assertFalse(
            boundary[
                "positive_characteristic_result_is_exact_Q_or_C_proof"
            ]
        )
        self.assertFalse(
            boundary["bounded_miss_is_a_nonexistence_proof"]
        )
        self.assertFalse(boundary["finite_counterexample_found"])
        self.assertFalse(
            boundary["global_affine_membership_decided"]
        )
        self.assertEqual(
            boundary["finite_affine_membership_status"], "undecided"
        )

    def test_stream_corruption_fails_hash_replay(self):
        run_directory = self.new_campaign("stream-corruption")
        stream = next(run_directory.glob("*.stdout.txt"))
        stream.write_text("corrupted\n", encoding="utf-8")
        with self.assertRaisesRegex(
            KrennStarPivotCASArtifactError, "stdout hash replay"
        ):
            summarize_completed_campaign(run_directory)

    def test_generated_input_corruption_is_rejected(self):
        run_directory = self.new_campaign("input-corruption")
        input_path = next(
            run_directory.parent.joinpath(
                INPUT_SUBDIRECTORY
            ).glob("*.sing")
        )
        input_path.write_bytes(input_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            KrennStarPivotCASArtifactError,
            "generated retained Singular input",
        ):
            summarize_completed_campaign(run_directory)

    def test_incomplete_checkpoint_is_rejected_even_with_fresh_hash(self):
        run_directory = self.new_campaign("checkpoint-corruption")
        checkpoint_path = run_directory / CHECKPOINT_NAME
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        checkpoint["completed_receipts"].pop()
        _write_json(checkpoint_path, checkpoint)
        manifest_path = run_directory / MANIFEST_NAME
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["checkpoint"] = {
            **_record(checkpoint_path),
            "completed_probe_count": 3,
        }
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            KrennStarPivotCASArtifactError,
            "receipt rows disagree",
        ):
            summarize_completed_campaign(run_directory)

    def test_refreshed_bundle_hash_cannot_escalate_claim(self):
        bundle = self.copied_bundle("claim-corruption")
        summary_path = bundle / SUMMARY_FILE
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        summary["claim_boundary"][
            "timeout_is_a_slice_decision"
        ] = True
        _write_json(summary_path, summary)
        manifest_path = bundle / BUNDLE_MANIFEST_FILE
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["artifact"] = _record(summary_path)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            KrennStarPivotCASArtifactError,
            "static evidence",
        ):
            verify_star_pivot_cas_summary_bundle(bundle)

    def test_extra_bundle_file_is_rejected(self):
        bundle = self.copied_bundle("extra-file")
        (bundle / "raw-stream.txt").write_text(
            "not portable inventory", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            KrennStarPivotCASArtifactError, "inventory"
        ):
            verify_star_pivot_cas_summary_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
