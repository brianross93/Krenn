"""Bounded Singular probes for the three star-pivot affine slices.

The default campaign exports the retained cubic presentation for each of
the three ``S5 x S3`` pivot-orbit representatives and runs it modulo 31.
Each engine invocation has a real ten-minute wall limit, two CPUs, eight
GiB of memory, no network, and the repository-pinned Singular image.

These runs are deliberately reconnaissance.  A timeout, a completed
positive-characteristic standard basis, or either value of ``unit_ideal``
does not decide the corresponding characteristic-zero slice.  In
particular, this module does not promote a raw Groebner computation to an
affine-membership or nonmembership claim.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_cas_runner import (
    CPUS_PER_PROBE,
    DOCKER_IMAGE,
    DOCKER_IMAGE_ID,
    ENGINE_BUDGET_SECONDS,
    KrennLocalizedCASRunnerError,
    MEMORY_GIB_PER_PROBE,
    OUTPUT_SUBDIRECTORY,
    SingularProbe,
    _assert_real_directory,
    _campaign_lock,
    _json_equal,
    _sha256,
    _strict_load,
    _write_json_atomic,
    inspect_docker_image,
    replay_probe_receipt,
    run_probe,
    validate_probe_input,
)
from experiments.krenn_quantum_graph.star_pivot_affine_slices import (
    PIVOT_ORBITS,
    singular_star_pivot_affine_slice_script,
    star_pivot_affine_presentation,
)


STAR_PIVOT_AFFINE_RUNNER_SCHEMA = (
    "krenn-n6-d3-star-pivot-affine-cas-suite-v1"
)
STAR_PIVOT_AFFINE_CHECKPOINT_SCHEMA = (
    "krenn-n6-d3-star-pivot-affine-cas-checkpoint-v1"
)
DEFAULT_STAR_PIVOT_AFFINE_SCRATCH_ROOT = Path(
    r"D:\KrennScratch\counterexample_search\star_pivot_affine_cas"
)
INPUT_SUBDIRECTORY = "retained_p31_inputs"
MANIFEST_NAME = "retained_p31_run_manifest.json"
CHECKPOINT_NAME = "retained_p31_checkpoint.json"
CHARACTERISTIC = 31
ALGORITHM = "slimgb"
PRESENTATION_KIND = "retained"
MAXIMUM_WORKERS = 3
GENERATOR_SOURCE_PATH = Path(__file__).with_name(
    "star_pivot_affine_slices.py"
)
EXPECTED_GENERATOR_SOURCE_BYTES = 39_840
EXPECTED_GENERATOR_SOURCE_SHA256 = (
    "74fdba0219ca8dc7d7424cc28153fc3bc2d7fb75ecd918af"
    "093efe141d5ae2e0"
)
EXPECTED_RETAINED_PRESENTATION_SHA256 = {
    0: "41c7aebc6263ed0d29c036566e357cd9caa2c6e3ac79f29ba6b7617c93ff9f18",
    1: "c50942b52b9fdac74aee50769545f51f3bf346599505f7791aa451c71cc9301f",
    2: "6b795fa5cdf087999d2fc6a8fc1fae12c5ab4a8dfe2a42126e046f1266c8eb3f",
}
EXPECTED_INPUTS = {
    0: {
        "bytes": 139_490,
        "sha256": (
            "90b4c5b8a86b90003269880131c2b3ef6d9ea875cea23deb"
            "78dee8911ed6d8f3"
        ),
    },
    1: {
        "bytes": 139_497,
        "sha256": (
            "4f8961c2b0f13fe6ce67645b1e0d06934ac98f50625425d"
            "cd14d866f6a2ed22f"
        ),
    },
    2: {
        "bytes": 139_491,
        "sha256": (
            "7c6f8889ca69075a7061958d94b1e1c7cc4b45cab529eee"
            "199dfdd3ea6f0c36a"
        ),
    },
}


def _script_name(orbit_index: int) -> str:
    return (
        f"star_pivot_orbit_{orbit_index}_retained_p31_slimgb.sing"
    )


def _script_bytes(orbit_index: int) -> bytes:
    return singular_star_pivot_affine_slice_script(
        orbit_index,
        PRESENTATION_KIND,
        characteristic=CHARACTERISTIC,
        algorithm=ALGORITHM,
    ).encode("utf-8")


def _build_probes() -> tuple[SingularProbe, ...]:
    probes = []
    for orbit in PIVOT_ORBITS:
        expected = EXPECTED_INPUTS[orbit.index]
        presentation = star_pivot_affine_presentation(
            orbit.index, PRESENTATION_KIND
        )
        script = _script_bytes(orbit.index)
        if (
            presentation.fingerprint()
            != EXPECTED_RETAINED_PRESENTATION_SHA256[orbit.index]
            or len(script) != expected["bytes"]
            or hashlib.sha256(script).hexdigest()
            != expected["sha256"]
        ):
            raise KrennLocalizedCASRunnerError(
                "a retained star-pivot CAS input fingerprint changed"
            )
        probes.append(SingularProbe(
            key=(
                f"star_pivot_orbit_{orbit.index}"
                "_retained_p31_slimgb"
            ),
            ideal=(
                "gauge-normalized star-pivot affine slice, "
                f"{orbit.equality_pattern} orbit, retained cubic form"
            ),
            input_subdirectory=INPUT_SUBDIRECTORY,
            script_name=_script_name(orbit.index),
            algorithm=ALGORITHM,
            characteristic=CHARACTERISTIC,
            input_bytes=expected["bytes"],
            input_sha256=expected["sha256"],
            parse_marker="KRENN_STAR_AFFINE_PARSE_OK",
            completion_marker="KRENN_STAR_AFFINE_GROEBNER_DONE",
        ))
    return tuple(probes)


STAR_PIVOT_AFFINE_PROBES = _build_probes()


def validate_generator_source(
    path: Path = GENERATOR_SOURCE_PATH,
) -> dict:
    """Replay the exact source fingerprint used to export the scripts."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise KrennLocalizedCASRunnerError(
            f"missing regular affine-slice generator source {path}"
        )
    size = path.stat().st_size
    digest = _sha256(path)
    if (
        size != EXPECTED_GENERATOR_SOURCE_BYTES
        or digest != EXPECTED_GENERATOR_SOURCE_SHA256
    ):
        raise KrennLocalizedCASRunnerError(
            "the affine-slice generator source fingerprint changed"
        )
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
    }


def _runner_source_record() -> dict:
    path = Path(__file__)
    if path.is_symlink() or not path.is_file():
        raise KrennLocalizedCASRunnerError(
            "the affine-slice CAS runner source is not a regular file"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise KrennLocalizedCASRunnerError(
            f"refusing non-regular affine-slice input target {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() and (
        temporary.is_symlink() or not temporary.is_file()
    ):
        raise KrennLocalizedCASRunnerError(
            f"refusing non-regular temporary target {temporary}"
        )
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def prepare_star_pivot_affine_inputs(
    scratch_root: Path = DEFAULT_STAR_PIVOT_AFFINE_SCRATCH_ROOT,
) -> tuple[dict, ...]:
    """Create or replay all three generated Singular inputs exactly."""

    validate_generator_source()
    scratch_root = Path(scratch_root)
    _assert_real_directory(scratch_root, create=False)
    input_directory = scratch_root / INPUT_SUBDIRECTORY
    _assert_real_directory(input_directory, create=True)
    rows = []
    for probe, orbit in zip(
        STAR_PIVOT_AFFINE_PROBES, PIVOT_ORBITS, strict=True
    ):
        path = input_directory / probe.script_name
        payload = _script_bytes(orbit.index)
        if path.exists():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != payload
            ):
                raise KrennLocalizedCASRunnerError(
                    f"retained affine-slice input changed: {path}"
                )
        else:
            _write_bytes_atomic(path, payload)
        rows.append(validate_probe_input(probe, scratch_root))
    return tuple(rows)


def _receipt_row(
    probe: SingularProbe,
    receipt: Mapping,
    output_directory: Path,
) -> dict:
    path = output_directory / f"{probe.key}.receipt.json"
    return {
        "key": probe.key,
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "elapsed_seconds": receipt["elapsed_seconds"],
        "outcome": receipt["outcome"],
    }


def _claim_boundary() -> dict:
    return {
        "positive_characteristic_result_is_exact_Q_proof": False,
        "timeout_is_a_slice_decision": False,
        "bounded_miss_is_a_nonexistence_proof": False,
        "raw_groebner_output_is_promoted_to_a_global_claim": False,
        "any_affine_slice_solved_over_Q_or_C": False,
        "finite_counterexample_found": False,
        "global_affine_membership_decided": False,
    }


def _static_checkpoint(
    scratch_root: Path,
) -> dict:
    return {
        "schema": STAR_PIVOT_AFFINE_CHECKPOINT_SCHEMA,
        "scratch_root": str(scratch_root),
        "generator_source": validate_generator_source(),
        "runner_source": _runner_source_record(),
        "presentation_kind": PRESENTATION_KIND,
        "characteristic": CHARACTERISTIC,
        "algorithm": ALGORITHM,
        "engine_budget_seconds_per_probe": ENGINE_BUDGET_SECONDS,
        "cpus_per_probe": CPUS_PER_PROBE,
        "memory_gib_per_probe": MEMORY_GIB_PER_PROBE,
        "docker_image": DOCKER_IMAGE,
        "docker_image_id": DOCKER_IMAGE_ID,
        "presentation_sha256": {
            str(key): value
            for key, value
            in EXPECTED_RETAINED_PRESENTATION_SHA256.items()
        },
        "inputs": [
            validate_probe_input(probe, scratch_root)
            for probe in STAR_PIVOT_AFFINE_PROBES
        ],
        "probes": [
            probe.to_dict() for probe in STAR_PIVOT_AFFINE_PROBES
        ],
        "random_seeds": [],
        "claim_boundary": _claim_boundary(),
    }


def _write_checkpoint(
    scratch_root: Path,
    receipts_by_key: Mapping[str, Mapping],
) -> dict:
    output_directory = Path(scratch_root) / OUTPUT_SUBDIRECTORY
    payload = {
        **_static_checkpoint(Path(scratch_root)),
        "completed_receipts": [
            _receipt_row(
                probe,
                receipts_by_key[probe.key],
                output_directory,
            )
            for probe in STAR_PIVOT_AFFINE_PROBES
            if probe.key in receipts_by_key
        ],
    }
    _write_json_atomic(output_directory / CHECKPOINT_NAME, payload)
    return payload


def replay_checkpoint(
    scratch_root: Path = DEFAULT_STAR_PIVOT_AFFINE_SCRATCH_ROOT,
) -> dict:
    """Validate a partial or completed checkpoint and all its receipts."""

    scratch_root = Path(scratch_root)
    output_directory = scratch_root / OUTPUT_SUBDIRECTORY
    payload = _strict_load(output_directory / CHECKPOINT_NAME)
    retained_static = {
        key: value
        for key, value in payload.items()
        if key != "completed_receipts"
    }
    if not _json_equal(
        retained_static, _static_checkpoint(scratch_root)
    ):
        raise KrennLocalizedCASRunnerError(
            "the retained affine-slice checkpoint metadata changed"
        )
    rows = payload.get("completed_receipts")
    if type(rows) is not list:
        raise KrennLocalizedCASRunnerError(
            "the affine-slice checkpoint lacks a receipt list"
        )
    probes = {probe.key: probe for probe in STAR_PIVOT_AFFINE_PROBES}
    if (
        len({row.get("key") for row in rows if type(row) is dict})
        != len(rows)
        or any(
            type(row) is not dict or row.get("key") not in probes
            for row in rows
        )
    ):
        raise KrennLocalizedCASRunnerError(
            "the affine-slice checkpoint receipt keys changed"
        )
    replayed = {}
    for row in rows:
        probe = probes[row["key"]]
        receipt = replay_probe_receipt(
            probe, scratch_root, DOCKER_IMAGE_ID
        )
        expected = _receipt_row(
            probe, receipt, output_directory
        )
        if not _json_equal(row, expected):
            raise KrennLocalizedCASRunnerError(
                f"checkpoint receipt replay failed for {probe.key}"
            )
        replayed[probe.key] = receipt
    return payload


def _maximum_overlap(receipts: Sequence[Mapping]) -> int:
    events = []
    for receipt in receipts:
        start = datetime.fromisoformat(receipt["started_at_utc"])
        end = start + timedelta(
            seconds=receipt["elapsed_seconds"]
        )
        events.extend(((start, 1), (end, -1)))
    active = 0
    maximum = 0
    for _timestamp, delta in sorted(
        events, key=lambda row: (row[0], row[1])
    ):
        active += delta
        maximum = max(maximum, active)
    return maximum


def run_star_pivot_affine_probe_suite(
    scratch_root: Path = DEFAULT_STAR_PIVOT_AFFINE_SCRATCH_ROOT,
    *,
    workers: int = 3,
    resume: bool = False,
) -> dict:
    """Run or replay the three locked retained-form 600-second probes."""

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAXIMUM_WORKERS
    ):
        raise KrennLocalizedCASRunnerError(
            f"workers must be in range(1,{MAXIMUM_WORKERS + 1})"
        )
    scratch_root = Path(scratch_root)
    prepare_star_pivot_affine_inputs(scratch_root)
    output_directory = scratch_root / OUTPUT_SUBDIRECTORY
    _assert_real_directory(output_directory, create=True)
    with _campaign_lock(output_directory):
        checkpoint_path = output_directory / CHECKPOINT_NAME
        if checkpoint_path.exists():
            if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
                raise KrennLocalizedCASRunnerError(
                    "the affine-slice checkpoint is not a regular file"
                )
            if not resume:
                raise KrennLocalizedCASRunnerError(
                    "a checkpoint already exists; use --resume"
                )
            replay_checkpoint(scratch_root)
        image_id = inspect_docker_image()
        receipts_by_key = {}
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for probe in STAR_PIVOT_AFFINE_PROBES:
                print(
                    f"START {probe.key} "
                    f"budget={ENGINE_BUDGET_SECONDS}s",
                    flush=True,
                )
                futures[executor.submit(
                    run_probe,
                    probe,
                    scratch_root,
                    image_id,
                    resume=resume,
                )] = probe
            for future in as_completed(futures):
                probe = futures[future]
                try:
                    receipt = future.result()
                except Exception as error:  # aggregate before failing
                    failures.append((probe.key, repr(error)))
                    print(f"FAILED {probe.key}: {error}", flush=True)
                else:
                    receipts_by_key[probe.key] = receipt
                    _write_checkpoint(
                        scratch_root, receipts_by_key
                    )
                    print(
                        f"DONE {probe.key} "
                        f"status={receipt['outcome']['status']} "
                        f"elapsed={receipt['elapsed_seconds']}",
                        flush=True,
                    )
        if failures:
            raise KrennLocalizedCASRunnerError(
                f"one or more star-pivot affine probes failed: {failures}"
            )
        receipts = tuple(
            receipts_by_key[probe.key]
            for probe in STAR_PIVOT_AFFINE_PROBES
        )
        checkpoint = _write_checkpoint(
            scratch_root, receipts_by_key
        )
        replay_checkpoint(scratch_root)
        maximum_active = _maximum_overlap(receipts)
        receipt_rows = [
            _receipt_row(probe, receipt, output_directory)
            for probe, receipt in zip(
                STAR_PIVOT_AFFINE_PROBES,
                receipts,
                strict=True,
            )
        ]
        checkpoint_path = output_directory / CHECKPOINT_NAME
        manifest = {
            "schema": STAR_PIVOT_AFFINE_RUNNER_SCHEMA,
            "scratch_root": str(scratch_root),
            "latest_invocation_workers": workers,
            "maximum_concurrent_probes_observed": maximum_active,
            "maximum_concurrent_cpus":
                maximum_active * CPUS_PER_PROBE,
            "maximum_concurrent_memory_gib":
                maximum_active * MEMORY_GIB_PER_PROBE,
            "generator_source": checkpoint["generator_source"],
            "runner_source": checkpoint["runner_source"],
            "presentation_kind": PRESENTATION_KIND,
            "eliminated_presentation_attempted": False,
            "characteristic": CHARACTERISTIC,
            "algorithm": ALGORITHM,
            "docker_image": DOCKER_IMAGE,
            "docker_image_id": image_id,
            "engine_budget_seconds_per_probe":
                ENGINE_BUDGET_SECONDS,
            "cpus_per_probe": CPUS_PER_PROBE,
            "memory_gib_per_probe": MEMORY_GIB_PER_PROBE,
            "random_seeds": [],
            "presentation_sha256":
                checkpoint["presentation_sha256"],
            "inputs": checkpoint["inputs"],
            "receipts": receipt_rows,
            "checkpoint": {
                "path": checkpoint_path.name,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
                "completed_probe_count": len(receipt_rows),
            },
            "claim_boundary": _claim_boundary(),
        }
        _write_json_atomic(
            output_directory / MANIFEST_NAME, manifest
        )
        return manifest


def _parse_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "workers must be an integer"
        ) from error
    if not 1 <= workers <= MAXIMUM_WORKERS:
        raise argparse.ArgumentTypeError(
            f"workers must be in range(1,{MAXIMUM_WORKERS + 1})"
        )
    return workers


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run three bounded retained-form star-pivot affine probes."
        )
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_STAR_PIVOT_AFFINE_SCRATCH_ROOT,
    )
    parser.add_argument("--workers", type=_parse_workers, default=3)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    payload = run_star_pivot_affine_probe_suite(
        arguments.scratch_root,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ALGORITHM",
    "CHARACTERISTIC",
    "CHECKPOINT_NAME",
    "DEFAULT_STAR_PIVOT_AFFINE_SCRATCH_ROOT",
    "EXPECTED_GENERATOR_SOURCE_BYTES",
    "EXPECTED_GENERATOR_SOURCE_SHA256",
    "EXPECTED_INPUTS",
    "EXPECTED_RETAINED_PRESENTATION_SHA256",
    "GENERATOR_SOURCE_PATH",
    "INPUT_SUBDIRECTORY",
    "MANIFEST_NAME",
    "MAXIMUM_WORKERS",
    "PRESENTATION_KIND",
    "STAR_PIVOT_AFFINE_CHECKPOINT_SCHEMA",
    "STAR_PIVOT_AFFINE_PROBES",
    "STAR_PIVOT_AFFINE_RUNNER_SCHEMA",
    "prepare_star_pivot_affine_inputs",
    "replay_checkpoint",
    "run_star_pivot_affine_probe_suite",
    "validate_generator_source",
]
