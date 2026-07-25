"""Bounded CAS runner for the two exact two-pivot derivative slices.

The inputs use the quotient-graded ``d=1`` systems with both monic pivots
placed in a two-variable elimination block.  Each finite-field computation
is reconnaissance only.  Completion never becomes a characteristic-zero
claim without an exact lift and replay.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from experiments.krenn_quantum_graph.localized_chart_cas_runner import (
    CPUS_PER_PROBE,
    DEFAULT_SCRATCH_ROOT,
    DOCKER_IMAGE,
    DOCKER_IMAGE_ID,
    ENGINE_BUDGET_SECONDS,
    KrennLocalizedCASRunnerError,
    MEMORY_GIB_PER_PROBE,
    OUTPUT_SUBDIRECTORY,
    RUNNER_SCHEMA,
    SingularProbe,
    _assert_real_directory,
    _campaign_lock,
    _sha256,
    _write_json_atomic,
    inspect_docker_image,
    run_probe,
    validate_probe_input,
)
from experiments.krenn_quantum_graph.localized_chart_sparse_elimination import (
    singular_triangular_derivative_script,
)


SPARSE_RUNNER_SCHEMA = (
    "krenn-n6-d3-sparse-two-pivot-cas-suite-v1"
)
INPUT_SUBDIRECTORY = "sparse_two_pivot_inputs"
MANIFEST_NAME = "sparse_two_pivot_run_manifest.json"
MAXIMUM_WORKERS = 2
CHARACTERISTIC = 1_009
ORDER = "two-pivot-elimination"
ALGORITHM = "std"
EXPECTED_INPUTS = {
    11: {
        "bytes": 128_711,
        "sha256": (
            "a0fe9990c2690eef36333dd25242b4f44b56335f32dd047f"
            "920c4bee257729c8"
        ),
    },
    29: {
        "bytes": 128_711,
        "sha256": (
            "3a2d5234e360fd06a1df6cc1e4a40add07166e81a956c99"
            "e9215102a28632751"
        ),
    },
}


def _script_name(ambient: int) -> str:
    return f"sparse_two_pivot_{ambient}_p1009_std.sing"


def _script_bytes(ambient: int) -> bytes:
    return singular_triangular_derivative_script(
        ambient,
        characteristic=CHARACTERISTIC,
        order=ORDER,
        algorithm=ALGORITHM,
    ).encode("utf-8")


def _build_probes() -> tuple[SingularProbe, ...]:
    rows = []
    marker = "KRENN_SPARSE_TRIANGULAR_TWO_PIVOT_ELIMINATION"
    for ambient in (11, 29):
        expected = EXPECTED_INPUTS[ambient]
        script = _script_bytes(ambient)
        digest = hashlib.sha256(script).hexdigest()
        if (
            len(script) != expected["bytes"]
            or digest != expected["sha256"]
        ):
            raise KrennLocalizedCASRunnerError(
                "a retained sparse two-pivot input fingerprint changed"
            )
        rows.append(SingularProbe(
            key=f"sparse_two_pivot_{ambient}_p1009_std",
            ideal=(
                "sparse derivative "
                f"{ambient}=1, triangular two-pivot form"
            ),
            input_subdirectory=INPUT_SUBDIRECTORY,
            script_name=_script_name(ambient),
            algorithm=ALGORITHM,
            characteristic=CHARACTERISTIC,
            input_bytes=expected["bytes"],
            input_sha256=expected["sha256"],
            parse_marker=f"{marker}_PARSE_OK",
            completion_marker=f"{marker}_GROEBNER_DONE",
        ))
    return tuple(rows)


SPARSE_PROBES = _build_probes()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise KrennLocalizedCASRunnerError(
            f"refusing non-regular sparse input target {path}"
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


def prepare_sparse_inputs(
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[dict, ...]:
    """Create or replay the two generated Singular inputs exactly."""

    scratch_root = Path(scratch_root)
    _assert_real_directory(scratch_root, create=False)
    input_directory = scratch_root / INPUT_SUBDIRECTORY
    _assert_real_directory(input_directory, create=True)
    rows = []
    for probe, ambient in zip(SPARSE_PROBES, (11, 29), strict=True):
        path = input_directory / probe.script_name
        payload = _script_bytes(ambient)
        if path.exists():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != payload
            ):
                raise KrennLocalizedCASRunnerError(
                    f"retained sparse input changed: {path}"
                )
        else:
            _write_bytes_atomic(path, payload)
        rows.append(validate_probe_input(probe, scratch_root))
    return tuple(rows)


def _maximum_overlap(receipts: Sequence[dict]) -> int:
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


def run_sparse_probe_suite(
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    *,
    workers: int = 2,
    resume: bool = False,
) -> dict:
    """Run or replay both locked 600-second sparse probes."""

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAXIMUM_WORKERS
    ):
        raise KrennLocalizedCASRunnerError(
            f"workers must be in range(1,{MAXIMUM_WORKERS + 1})"
        )
    scratch_root = Path(scratch_root)
    prepare_sparse_inputs(scratch_root)
    output_directory = scratch_root / OUTPUT_SUBDIRECTORY
    _assert_real_directory(output_directory, create=True)
    with _campaign_lock(output_directory):
        image_id = inspect_docker_image()
        receipts_by_key = {}
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_probe,
                    probe,
                    scratch_root,
                    image_id,
                    resume=resume,
                ): probe
                for probe in SPARSE_PROBES
            }
            for future in as_completed(futures):
                probe = futures[future]
                try:
                    receipt = future.result()
                except Exception as error:  # aggregate before failing
                    failures.append((probe.key, repr(error)))
                    print(f"FAILED {probe.key}: {error}", flush=True)
                else:
                    receipts_by_key[probe.key] = receipt
                    print(
                        f"DONE {probe.key} "
                        f"status={receipt['outcome']['status']} "
                        f"elapsed={receipt['elapsed_seconds']}",
                        flush=True,
                    )
        if failures:
            raise KrennLocalizedCASRunnerError(
                f"one or more sparse probes failed: {failures}"
            )
        receipts = tuple(
            receipts_by_key[probe.key] for probe in SPARSE_PROBES
        )
        maximum_active = _maximum_overlap(receipts)
        receipt_rows = []
        for probe, receipt in zip(
            SPARSE_PROBES, receipts, strict=True
        ):
            path = output_directory / f"{probe.key}.receipt.json"
            receipt_rows.append({
                "key": probe.key,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "elapsed_seconds": receipt["elapsed_seconds"],
                "outcome": receipt["outcome"],
            })
        manifest = {
            "schema": SPARSE_RUNNER_SCHEMA,
            "scratch_root": str(scratch_root),
            "workers": workers,
            "maximum_concurrent_probes_observed": maximum_active,
            "maximum_concurrent_cpus":
                maximum_active * CPUS_PER_PROBE,
            "maximum_concurrent_memory_gib":
                maximum_active * MEMORY_GIB_PER_PROBE,
            "docker_image": DOCKER_IMAGE,
            "docker_image_id": DOCKER_IMAGE_ID,
            "characteristic": CHARACTERISTIC,
            "algorithm": ALGORITHM,
            "order": ORDER,
            "engine_budget_seconds_per_probe":
                ENGINE_BUDGET_SECONDS,
            "random_seeds": [],
            "inputs": list(prepare_sparse_inputs(scratch_root)),
            "receipts": receipt_rows,
            "claim_boundary": {
                "finite_field_result_is_exact_Q_proof": False,
                "timeout_is_a_chart_decision": False,
                "unit_or_proper_status_over_Q": "undecided",
            },
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
            "Run the two bounded sparse two-pivot derivative probes."
        )
    )
    parser.add_argument(
        "--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT
    )
    parser.add_argument("--workers", type=_parse_workers, default=2)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    payload = run_sparse_probe_suite(
        arguments.scratch_root,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
