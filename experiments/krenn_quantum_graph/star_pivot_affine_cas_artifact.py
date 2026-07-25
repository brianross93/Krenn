"""Portable, fail-closed summary of the three star-pivot CAS probes.

Generation reads the completed scratch campaign, replays every receipt,
stream hash, marker classification, deterministic Singular input, source
hash, checkpoint, and run manifest, then embeds the small control evidence
and three short streams in ``summary.json``.  The resulting two-file
bundle can therefore be verified after the disposable scratch directory
has been removed or relocated.

All three retained-form probes are characteristic-31 ``slimgb`` runs
which parsed successfully and reached the real 600-second timeout with
return code 124.  That is reconnaissance only: a parsed timeout decides
neither a slice nor characteristic-zero affine membership.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
    MEMORY_GIB_PER_PROBE,
    OUTPUT_SUBDIRECTORY,
    RUNNER_SCHEMA,
    classify_output,
    docker_argv,
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
    INPUT_SUBDIRECTORY,
    MANIFEST_NAME,
    MAXIMUM_WORKERS,
    PRESENTATION_KIND,
    STAR_PIVOT_AFFINE_CHECKPOINT_SCHEMA,
    STAR_PIVOT_AFFINE_PROBES,
    STAR_PIVOT_AFFINE_RUNNER_SCHEMA,
)
from experiments.krenn_quantum_graph.star_pivot_affine_slices import (
    PIVOT_ORBITS,
    singular_star_pivot_affine_slice_script,
    star_pivot_affine_presentation,
)


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "krenn-n6-d3-star-pivot-affine-cas-summary-v1"
BUNDLE_MANIFEST_SCHEMA = (
    "krenn-n6-d3-star-pivot-affine-cas-summary-manifest-v1"
)
SUMMARY_FILE = "summary.json"
BUNDLE_MANIFEST_FILE = "manifest.json"
ALL_FILES = (SUMMARY_FILE, BUNDLE_MANIFEST_FILE)
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_STREAM_BYTES = 64 * 1024
EXPECTED_RUNS = 3

PRODUCER_SOURCE = (
    "experiments/krenn_quantum_graph/"
    "star_pivot_affine_cas_artifact.py"
)
SOURCE_INPUTS = (
    "experiments/krenn_quantum_graph/localized_chart_cas_runner.py",
    "experiments/krenn_quantum_graph/star_pivot_affine_cas_runner.py",
    "experiments/krenn_quantum_graph/star_pivot_affine_slices.py",
)

_CLAIM_BOUNDARY = {
    "scope": "three retained star-pivot affine orbit representatives",
    "parse_success_is_standard_basis_completion": False,
    "timeout_is_a_slice_decision": False,
    "positive_characteristic_result_is_exact_Q_or_C_proof": False,
    "bounded_miss_is_a_nonexistence_proof": False,
    "finite_counterexample_found": False,
    "global_affine_membership_decided": False,
    "finite_affine_membership_status": "undecided",
}
_RUNNER_BOUNDARY = {
    "positive_characteristic_result_is_exact_Q_proof": False,
    "timeout_is_a_slice_decision": False,
    "bounded_miss_is_a_nonexistence_proof": False,
    "raw_groebner_output_is_promoted_to_a_global_claim": False,
    "any_affine_slice_solved_over_Q_or_C": False,
    "finite_counterexample_found": False,
    "global_affine_membership_decided": False,
}
_OUTCOME_BOUNDARY = {
    "positive_characteristic_result_is_exact_Q_proof": False,
    "timeout_is_a_chart_decision": False,
}
_RUN_MANIFEST_KEYS = {
    "schema",
    "scratch_root",
    "latest_invocation_workers",
    "maximum_concurrent_probes_observed",
    "maximum_concurrent_cpus",
    "maximum_concurrent_memory_gib",
    "generator_source",
    "runner_source",
    "presentation_kind",
    "eliminated_presentation_attempted",
    "characteristic",
    "algorithm",
    "docker_image",
    "docker_image_id",
    "engine_budget_seconds_per_probe",
    "cpus_per_probe",
    "memory_gib_per_probe",
    "random_seeds",
    "presentation_sha256",
    "inputs",
    "receipts",
    "checkpoint",
    "claim_boundary",
}
_CHECKPOINT_KEYS = {
    "schema",
    "scratch_root",
    "generator_source",
    "runner_source",
    "presentation_kind",
    "characteristic",
    "algorithm",
    "engine_budget_seconds_per_probe",
    "cpus_per_probe",
    "memory_gib_per_probe",
    "docker_image",
    "docker_image_id",
    "presentation_sha256",
    "inputs",
    "probes",
    "random_seeds",
    "claim_boundary",
    "completed_receipts",
}
_RECEIPT_KEYS = {
    "schema",
    "probe",
    "input_script",
    "docker",
    "started_at_utc",
    "elapsed_seconds",
    "stdout",
    "stderr",
    "outcome",
}
_LOCAL_RECORD_KEYS = {"path", "bytes", "sha256"}


class KrennStarPivotCASArtifactError(RuntimeError):
    """The completed campaign or portable bundle failed replay."""


@dataclass(frozen=True)
class VerifiedStarPivotCASArtifact:
    directory: Path
    summary: Mapping
    manifest: Mapping
    checks: tuple[tuple[str, bool], ...]

    @property
    def valid(self) -> bool:
        return all(value for _name, value in self.checks)


def _canonical_json_bytes(payload) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise KrennStarPivotCASArtifactError(
            "payload is not strict JSON"
        ) from error
    return (text + "\n").encode("utf-8")


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennStarPivotCASArtifactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str):
    raise KrennStarPivotCASArtifactError(
        f"non-finite JSON constant {value!r}"
    )


def _load_json(path: Path, label: str) -> dict:
    try:
        if path.is_symlink() or not path.is_file():
            raise KrennStarPivotCASArtifactError(
                f"{label} is absent or linked"
            )
        if path.stat().st_size > MAX_JSON_BYTES:
            raise KrennStarPivotCASArtifactError(
                f"{label} exceeds its reviewed size bound"
            )
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except KrennStarPivotCASArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennStarPivotCASArtifactError(
            f"could not decode {label}"
        ) from error
    if type(payload) is not dict:
        raise KrennStarPivotCASArtifactError(
            f"{label} is not a JSON object"
        )
    if raw != _canonical_json_bytes(payload):
        raise KrennStarPivotCASArtifactError(
            f"{label} is not canonical round-trippable JSON"
        )
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_record(label: str) -> dict:
    path = ROOT / label
    if path.is_symlink() or not path.is_file():
        raise KrennStarPivotCASArtifactError(
            f"source is absent or linked: {label}"
        )
    return {
        "path": label,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "hash_mode": "raw-bytes-v1",
    }


def _safe_local_file(directory: Path, label: object) -> Path:
    if type(label) is not str:
        raise KrennStarPivotCASArtifactError(
            "local evidence path is not a string"
        )
    relative = Path(label)
    if (
        not label
        or relative.is_absolute()
        or bool(relative.drive)
        or len(relative.parts) != 1
        or label in (".", "..")
        or "/" in label
        or "\\" in label
        or ":" in label
    ):
        raise KrennStarPivotCASArtifactError(
            "local evidence path escaped its directory"
        )
    path = directory / label
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True).parent != directory.resolve()
    ):
        raise KrennStarPivotCASArtifactError(
            f"local evidence is absent, linked, or escaped: {label}"
        )
    return path


def _read_stream(path: Path) -> tuple[bytes, str]:
    if path.stat().st_size > MAX_STREAM_BYTES:
        raise KrennStarPivotCASArtifactError(
            "a retained stream exceeds the reviewed size bound"
        )
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise KrennStarPivotCASArtifactError(
            "a retained stream is not portable UTF-8"
        ) from error
    if text.encode("utf-8") != payload:
        raise KrennStarPivotCASArtifactError(
            "a retained stream did not round-trip as UTF-8"
        )
    return payload, text


def _expected_generated_inputs() -> list[dict]:
    rows = []
    for orbit in PIVOT_ORBITS:
        index = orbit.index
        script = singular_star_pivot_affine_slice_script(
            index,
            PRESENTATION_KIND,
            characteristic=CHARACTERISTIC,
            algorithm=ALGORITHM,
        ).encode("utf-8")
        presentation = star_pivot_affine_presentation(
            index, PRESENTATION_KIND
        )
        expected = EXPECTED_INPUTS[index]
        if (
            presentation.fingerprint()
            != EXPECTED_RETAINED_PRESENTATION_SHA256[index]
            or len(script) != expected["bytes"]
            or _sha256_bytes(script) != expected["sha256"]
        ):
            raise KrennStarPivotCASArtifactError(
                "deterministic retained input fingerprints changed"
            )
        rows.append({
            "orbit_index": index,
            "equality_pattern": orbit.equality_pattern,
            "script_name": (
                f"star_pivot_orbit_{index}_retained_p31_slimgb.sing"
            ),
            "bytes": len(script),
            "sha256": _sha256_bytes(script),
            "presentation_sha256": presentation.fingerprint(),
        })
    return rows


def _expected_source_evidence() -> dict:
    generator = _source_record(
        "experiments/krenn_quantum_graph/"
        "star_pivot_affine_slices.py"
    )
    if (
        generator["bytes"] != EXPECTED_GENERATOR_SOURCE_BYTES
        or generator["sha256"] != EXPECTED_GENERATOR_SOURCE_SHA256
        or Path(GENERATOR_SOURCE_PATH).resolve()
        != (ROOT / generator["path"]).resolve()
    ):
        raise KrennStarPivotCASArtifactError(
            "the pinned affine-slice generator source changed"
        )
    return {
        "generator": generator,
        "runner": _source_record(
            "experiments/krenn_quantum_graph/"
            "star_pivot_affine_cas_runner.py"
        ),
        "shared_runner": _source_record(
            "experiments/krenn_quantum_graph/"
            "localized_chart_cas_runner.py"
        ),
    }


def _absolute_input_records(
    scratch_root: Path,
    generated: Sequence[Mapping],
) -> list[dict]:
    input_directory = scratch_root / INPUT_SUBDIRECTORY
    if input_directory.is_symlink() or not input_directory.is_dir():
        raise KrennStarPivotCASArtifactError(
            "retained input directory is absent or linked"
        )
    rows = []
    for expected in generated:
        path = input_directory / expected["script_name"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise KrennStarPivotCASArtifactError(
                "a generated retained Singular input changed"
            )
        rows.append({
            "path": str(path),
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        })
    return rows


def _receipt_row(
    receipt_path: Path,
    key: str,
    receipt: Mapping,
) -> dict:
    return {
        "key": key,
        "path": receipt_path.name,
        "bytes": receipt_path.stat().st_size,
        "sha256": _sha256(receipt_path),
        "elapsed_seconds": receipt["elapsed_seconds"],
        "outcome": receipt["outcome"],
    }


def _maximum_overlap(receipts: Sequence[Mapping]) -> int:
    events = []
    for receipt in receipts:
        try:
            start = datetime.fromisoformat(receipt["started_at_utc"])
        except (TypeError, ValueError) as error:
            raise KrennStarPivotCASArtifactError(
                "a receipt start timestamp is invalid"
            ) from error
        end = start + timedelta(seconds=receipt["elapsed_seconds"])
        events.extend(((start, 1), (end, -1)))
    active = 0
    maximum = 0
    for _timestamp, delta in sorted(
        events, key=lambda row: (row[0], row[1])
    ):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _validate_recorded_source(
    recorded: Mapping,
    expected: Mapping,
    *,
    label: str,
) -> None:
    if (
        type(recorded) is not dict
        or set(recorded) != {"path", "bytes", "sha256"}
        or recorded["bytes"] != expected["bytes"]
        or recorded["sha256"] != expected["sha256"]
        or Path(recorded["path"]).name != Path(expected["path"]).name
    ):
        raise KrennStarPivotCASArtifactError(
            f"{label} source receipt changed"
        )


def _validate_input_records(
    recorded: object,
    expected: Sequence[Mapping],
) -> None:
    if type(recorded) is not list or len(recorded) != EXPECTED_RUNS:
        raise KrennStarPivotCASArtifactError(
            "input receipt list changed"
        )
    for row, expected_row in zip(recorded, expected, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"path", "bytes", "sha256"}
            or Path(row["path"]).name
            != Path(expected_row["path"]).name
            or row["bytes"] != expected_row["bytes"]
            or row["sha256"] != expected_row["sha256"]
        ):
            raise KrennStarPivotCASArtifactError(
                "an input receipt changed"
            )


def _validate_receipt(
    probe,
    receipt_path: Path,
    scratch_root: Path,
) -> dict:
    receipt = _load_json(receipt_path, f"{probe.key} receipt")
    if (
        set(receipt) != _RECEIPT_KEYS
        or receipt["schema"] != RUNNER_SCHEMA
        or receipt["probe"] != probe.to_dict()
        or type(receipt["input_script"]) is not dict
        or Path(receipt["input_script"]["path"]).name
        != probe.script_name
        or receipt["input_script"]["bytes"] != probe.input_bytes
        or receipt["input_script"]["sha256"] != probe.input_sha256
        or type(receipt["elapsed_seconds"]) not in (int, float)
        or isinstance(receipt["elapsed_seconds"], bool)
        or not ENGINE_BUDGET_SECONDS
        <= receipt["elapsed_seconds"]
        < ENGINE_BUDGET_SECONDS + 60
    ):
        raise KrennStarPivotCASArtifactError(
            f"{probe.key} receipt metadata changed"
        )
    docker = receipt["docker"]
    if (
        type(docker) is not dict
        or set(docker) != {"argv", "image", "image_id", "network"}
        or docker["image"] != DOCKER_IMAGE
        or docker["image_id"] != DOCKER_IMAGE_ID
        or docker["network"] != "none"
        or docker["argv"] != list(
            docker_argv(probe, scratch_root, hardened=True)
        )
    ):
        raise KrennStarPivotCASArtifactError(
            f"{probe.key} Docker receipt changed"
        )
    streams = {}
    stream_texts = {}
    for stream in ("stdout", "stderr"):
        row = receipt[stream]
        if type(row) is not dict or set(row) != _LOCAL_RECORD_KEYS:
            raise KrennStarPivotCASArtifactError(
                f"{probe.key} {stream} receipt changed"
            )
        path = _safe_local_file(receipt_path.parent, row["path"])
        payload, text = _read_stream(path)
        expected_record = {
            "path": path.name,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        if row != expected_record:
            raise KrennStarPivotCASArtifactError(
                f"{probe.key} {stream} hash replay failed"
            )
        streams[stream] = expected_record
        stream_texts[stream] = text
    outcome = receipt["outcome"]
    return_code = (
        outcome.get("return_code")
        if type(outcome) is dict
        else None
    )
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        raise KrennStarPivotCASArtifactError(
            f"{probe.key} return code is invalid"
        )
    replayed = classify_output(
        probe,
        return_code=return_code,
        stdout_text=stream_texts["stdout"],
    )
    if outcome != replayed or outcome != {
        "status": "timeout-after-parse",
        "return_code": 124,
        "parse_marker_observed": True,
        "completion_marker_observed": False,
        "selected_singular_output": {},
        "claim_boundary": _OUTCOME_BOUNDARY,
    }:
        raise KrennStarPivotCASArtifactError(
            f"{probe.key} outcome classification failed replay"
        )
    return {
        "key": probe.key,
        "receipt_record": _record(receipt_path),
        "receipt_payload": receipt,
        "stdout_record": streams["stdout"],
        "stdout_utf8": stream_texts["stdout"],
        "stderr_record": streams["stderr"],
        "stderr_utf8": stream_texts["stderr"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "outcome": outcome,
    }


def _validate_control_payloads(
    scratch_root: Path,
    run_directory: Path,
    manifest: Mapping,
    checkpoint: Mapping,
    runs: Sequence[Mapping],
    generated_inputs: Sequence[Mapping],
    sources: Mapping,
) -> None:
    if (
        set(manifest) != _RUN_MANIFEST_KEYS
        or manifest["schema"] != STAR_PIVOT_AFFINE_RUNNER_SCHEMA
        or set(checkpoint) != _CHECKPOINT_KEYS
        or checkpoint["schema"] != STAR_PIVOT_AFFINE_CHECKPOINT_SCHEMA
        or manifest["scratch_root"] != str(scratch_root)
        or checkpoint["scratch_root"] != str(scratch_root)
        or manifest["presentation_kind"] != PRESENTATION_KIND
        or checkpoint["presentation_kind"] != PRESENTATION_KIND
        or manifest["eliminated_presentation_attempted"] is not False
        or manifest["characteristic"] != CHARACTERISTIC
        or checkpoint["characteristic"] != CHARACTERISTIC
        or manifest["algorithm"] != ALGORITHM
        or checkpoint["algorithm"] != ALGORITHM
        or manifest["docker_image"] != DOCKER_IMAGE
        or checkpoint["docker_image"] != DOCKER_IMAGE
        or manifest["docker_image_id"] != DOCKER_IMAGE_ID
        or checkpoint["docker_image_id"] != DOCKER_IMAGE_ID
        or manifest["engine_budget_seconds_per_probe"]
        != ENGINE_BUDGET_SECONDS
        or checkpoint["engine_budget_seconds_per_probe"]
        != ENGINE_BUDGET_SECONDS
        or manifest["cpus_per_probe"] != CPUS_PER_PROBE
        or checkpoint["cpus_per_probe"] != CPUS_PER_PROBE
        or manifest["memory_gib_per_probe"] != MEMORY_GIB_PER_PROBE
        or checkpoint["memory_gib_per_probe"] != MEMORY_GIB_PER_PROBE
        or manifest["random_seeds"] != []
        or checkpoint["random_seeds"] != []
        or manifest["claim_boundary"] != _RUNNER_BOUNDARY
        or checkpoint["claim_boundary"] != _RUNNER_BOUNDARY
    ):
        raise KrennStarPivotCASArtifactError(
            "run manifest or checkpoint static metadata changed"
        )
    expected_presentations = {
        str(key): value
        for key, value in EXPECTED_RETAINED_PRESENTATION_SHA256.items()
    }
    if (
        manifest["presentation_sha256"] != expected_presentations
        or checkpoint["presentation_sha256"] != expected_presentations
        or checkpoint["probes"]
        != [probe.to_dict() for probe in STAR_PIVOT_AFFINE_PROBES]
    ):
        raise KrennStarPivotCASArtifactError(
            "presentation or probe fingerprints changed"
        )
    _validate_recorded_source(
        manifest["generator_source"],
        sources["generator"],
        label="manifest generator",
    )
    _validate_recorded_source(
        checkpoint["generator_source"],
        sources["generator"],
        label="checkpoint generator",
    )
    _validate_recorded_source(
        manifest["runner_source"],
        sources["runner"],
        label="manifest runner",
    )
    _validate_recorded_source(
        checkpoint["runner_source"],
        sources["runner"],
        label="checkpoint runner",
    )
    absolute_inputs = _absolute_input_records(
        scratch_root, generated_inputs
    )
    _validate_input_records(manifest["inputs"], absolute_inputs)
    _validate_input_records(checkpoint["inputs"], absolute_inputs)
    expected_rows = [
        _receipt_row(
            run_directory / run["receipt_record"]["path"],
            run["key"],
            run["receipt_payload"],
        )
        for run in runs
    ]
    if (
        manifest["receipts"] != expected_rows
        or checkpoint["completed_receipts"] != expected_rows
    ):
        raise KrennStarPivotCASArtifactError(
            "receipt rows disagree across control files"
        )
    checkpoint_path = run_directory / CHECKPOINT_NAME
    if manifest["checkpoint"] != {
        "path": CHECKPOINT_NAME,
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256(checkpoint_path),
        "completed_probe_count": EXPECTED_RUNS,
    }:
        raise KrennStarPivotCASArtifactError(
            "completed checkpoint record changed"
        )
    workers = manifest["latest_invocation_workers"]
    overlap = _maximum_overlap(
        [run["receipt_payload"] for run in runs]
    )
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAXIMUM_WORKERS
        or manifest["maximum_concurrent_probes_observed"] != overlap
        or not 1 <= overlap <= workers
        or manifest["maximum_concurrent_cpus"]
        != overlap * CPUS_PER_PROBE
        or manifest["maximum_concurrent_memory_gib"]
        != overlap * MEMORY_GIB_PER_PROBE
    ):
        raise KrennStarPivotCASArtifactError(
            "worker overlap or resource accounting changed"
        )


def summarize_completed_campaign(run_directory: Path) -> dict:
    """Replay the completed scratch campaign into portable evidence."""

    run_directory = Path(run_directory)
    if (
        run_directory.is_symlink()
        or not run_directory.is_dir()
        or run_directory.name != OUTPUT_SUBDIRECTORY
    ):
        raise KrennStarPivotCASArtifactError(
            "run directory is absent, linked, or not ten_minute_runs"
        )
    run_directory = run_directory.resolve(strict=True)
    scratch_root = run_directory.parent
    expected_names = {MANIFEST_NAME, CHECKPOINT_NAME}
    for probe in STAR_PIVOT_AFFINE_PROBES:
        expected_names.update({
            f"{probe.key}.receipt.json",
            f"{probe.key}.stdout.txt",
            f"{probe.key}.stderr.txt",
        })
    if {path.name for path in run_directory.iterdir()} != expected_names:
        raise KrennStarPivotCASArtifactError(
            "completed run-directory inventory changed"
        )
    manifest_path = run_directory / MANIFEST_NAME
    checkpoint_path = run_directory / CHECKPOINT_NAME
    manifest = _load_json(manifest_path, "completed run manifest")
    checkpoint = _load_json(
        checkpoint_path, "completed campaign checkpoint"
    )
    generated = _expected_generated_inputs()
    sources = _expected_source_evidence()
    runs = [
        _validate_receipt(
            probe,
            run_directory / f"{probe.key}.receipt.json",
            scratch_root,
        )
        for probe in STAR_PIVOT_AFFINE_PROBES
    ]
    _validate_control_payloads(
        scratch_root,
        run_directory,
        manifest,
        checkpoint,
        runs,
        generated,
        sources,
    )
    outcome_counts = Counter(
        run["outcome"]["status"] for run in runs
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "problem": {
            "n": 6,
            "d": 3,
            "target": "GHZ_6,3",
            "slice_family": "star-pivot affine normal forms",
        },
        "campaign": {
            "original_scratch_root": str(scratch_root),
            "original_run_directory": str(run_directory),
            "run_count": EXPECTED_RUNS,
            "presentation_kind": PRESENTATION_KIND,
            "characteristic": CHARACTERISTIC,
            "algorithm": ALGORITHM,
            "engine_budget_seconds_per_probe": (
                ENGINE_BUDGET_SECONDS
            ),
            "latest_invocation_workers": manifest[
                "latest_invocation_workers"
            ],
            "maximum_concurrent_probes_observed": manifest[
                "maximum_concurrent_probes_observed"
            ],
            "resources": {
                "cpus_per_probe": CPUS_PER_PROBE,
                "memory_gib_per_probe": MEMORY_GIB_PER_PROBE,
                "maximum_concurrent_cpus": manifest[
                    "maximum_concurrent_cpus"
                ],
                "maximum_concurrent_memory_gib": manifest[
                    "maximum_concurrent_memory_gib"
                ],
                "network": "none",
            },
            "docker_image": DOCKER_IMAGE,
            "docker_image_id": DOCKER_IMAGE_ID,
            "random_seeds": [],
        },
        "deterministic_inputs": generated,
        "source_evidence": sources,
        "results": [
            {
                "orbit_index": orbit.index,
                "equality_pattern": orbit.equality_pattern,
                "key": run["key"],
                "elapsed_seconds": run["elapsed_seconds"],
                "parse_marker_observed": True,
                "completion_marker_observed": False,
                "return_code": 124,
                "status": "timeout-after-parse",
                "budget_seconds": ENGINE_BUDGET_SECONDS,
            }
            for orbit, run in zip(PIVOT_ORBITS, runs, strict=True)
        ],
        "aggregate": {
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "parse_success_count": sum(
                run["outcome"]["parse_marker_observed"]
                for run in runs
            ),
            "timeout_return_code_124_count": sum(
                run["outcome"]["return_code"] == 124
                for run in runs
            ),
            "standard_basis_completion_count": sum(
                run["outcome"]["completion_marker_observed"]
                for run in runs
            ),
            "all_elapsed_at_least_600_seconds": all(
                run["elapsed_seconds"] >= ENGINE_BUDGET_SECONDS
                for run in runs
            ),
            "characteristic_zero_slice_decisions": 0,
            "finite_exact_counterexamples_verified": 0,
        },
        "portable_evidence": {
            "original_manifest": {
                "record": _record(manifest_path),
                "payload": manifest,
            },
            "completed_checkpoint": {
                "record": _record(checkpoint_path),
                "payload": checkpoint,
            },
            "runs": runs,
            "self_contained_after_generation": True,
            "verification_requires_original_scratch_directory": False,
        },
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    }


def _verify_embedded_record(
    record: Mapping,
    payload: Mapping,
    *,
    expected_name: str,
    label: str,
) -> None:
    encoded = _canonical_json_bytes(payload)
    if (
        type(record) is not dict
        or set(record) != _LOCAL_RECORD_KEYS
        or record["path"] != expected_name
        or record["bytes"] != len(encoded)
        or record["sha256"] != _sha256_bytes(encoded)
    ):
        raise KrennStarPivotCASArtifactError(
            f"embedded {label} hash replay failed"
        )


def _verify_portable_summary(summary: Mapping) -> None:
    if (
        type(summary) is not dict
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("claim_boundary") != _CLAIM_BOUNDARY
        or summary.get("campaign", {}).get("run_count") != EXPECTED_RUNS
        or summary.get("campaign", {}).get("characteristic")
        != CHARACTERISTIC
        or summary.get("campaign", {}).get("algorithm") != ALGORITHM
        or summary.get("campaign", {}).get(
            "engine_budget_seconds_per_probe"
        ) != ENGINE_BUDGET_SECONDS
        or summary.get("campaign", {}).get("docker_image")
        != DOCKER_IMAGE
        or summary.get("campaign", {}).get("docker_image_id")
        != DOCKER_IMAGE_ID
        or summary.get("deterministic_inputs")
        != _expected_generated_inputs()
        or summary.get("source_evidence")
        != _expected_source_evidence()
    ):
        raise KrennStarPivotCASArtifactError(
            "portable summary static evidence changed"
        )
    evidence = summary.get("portable_evidence")
    if (
        type(evidence) is not dict
        or evidence.get("self_contained_after_generation") is not True
        or evidence.get(
            "verification_requires_original_scratch_directory"
        ) is not False
        or type(evidence.get("runs")) is not list
        or len(evidence["runs"]) != EXPECTED_RUNS
    ):
        raise KrennStarPivotCASArtifactError(
            "portable evidence schema changed"
        )
    original = evidence.get("original_manifest")
    checkpoint = evidence.get("completed_checkpoint")
    if type(original) is not dict or type(checkpoint) is not dict:
        raise KrennStarPivotCASArtifactError(
            "embedded control evidence is malformed"
        )
    _verify_embedded_record(
        original.get("record"),
        original.get("payload"),
        expected_name=MANIFEST_NAME,
        label="run manifest",
    )
    _verify_embedded_record(
        checkpoint.get("record"),
        checkpoint.get("payload"),
        expected_name=CHECKPOINT_NAME,
        label="checkpoint",
    )
    manifest = original["payload"]
    checkpoint_payload = checkpoint["payload"]
    results = summary.get("results")
    if type(results) is not list or len(results) != EXPECTED_RUNS:
        raise KrennStarPivotCASArtifactError(
            "portable result rows changed"
        )
    expected_receipt_rows = []
    reconstructed_runs = []
    for orbit, probe, run, result in zip(
        PIVOT_ORBITS,
        STAR_PIVOT_AFFINE_PROBES,
        evidence["runs"],
        results,
        strict=True,
    ):
        required = {
            "key",
            "receipt_record",
            "receipt_payload",
            "stdout_record",
            "stdout_utf8",
            "stderr_record",
            "stderr_utf8",
            "elapsed_seconds",
            "outcome",
        }
        if type(run) is not dict or set(run) != required:
            raise KrennStarPivotCASArtifactError(
                "an embedded run schema changed"
            )
        receipt = run["receipt_payload"]
        _verify_embedded_record(
            run["receipt_record"],
            receipt,
            expected_name=f"{probe.key}.receipt.json",
            label=f"{probe.key} receipt",
        )
        for stream in ("stdout", "stderr"):
            text = run[f"{stream}_utf8"]
            record = run[f"{stream}_record"]
            if type(text) is not str:
                raise KrennStarPivotCASArtifactError(
                    "an embedded stream is not text"
                )
            encoded = text.encode("utf-8")
            if (
                type(record) is not dict
                or set(record) != _LOCAL_RECORD_KEYS
                or record["path"] != f"{probe.key}.{stream}.txt"
                or record["bytes"] != len(encoded)
                or record["sha256"] != _sha256_bytes(encoded)
                or receipt[stream] != record
            ):
                raise KrennStarPivotCASArtifactError(
                    f"embedded {stream} hash replay failed"
                )
        replayed = classify_output(
            probe,
            return_code=receipt["outcome"]["return_code"],
            stdout_text=run["stdout_utf8"],
        )
        if (
            receipt["schema"] != RUNNER_SCHEMA
            or receipt["probe"] != probe.to_dict()
            or run["key"] != probe.key
            or run["outcome"] != receipt["outcome"]
            or replayed != receipt["outcome"]
            or replayed["status"] != "timeout-after-parse"
            or replayed["return_code"] != 124
            or replayed["parse_marker_observed"] is not True
            or replayed["completion_marker_observed"] is not False
            or run["elapsed_seconds"] != receipt["elapsed_seconds"]
            or result != {
                "orbit_index": orbit.index,
                "equality_pattern": orbit.equality_pattern,
                "key": probe.key,
                "elapsed_seconds": receipt["elapsed_seconds"],
                "parse_marker_observed": True,
                "completion_marker_observed": False,
                "return_code": 124,
                "status": "timeout-after-parse",
                "budget_seconds": ENGINE_BUDGET_SECONDS,
            }
        ):
            raise KrennStarPivotCASArtifactError(
                "embedded outcome classification failed replay"
            )
        expected_receipt_rows.append({
            "key": probe.key,
            **run["receipt_record"],
            "elapsed_seconds": receipt["elapsed_seconds"],
            "outcome": receipt["outcome"],
        })
        reconstructed_runs.append(receipt)
    if (
        manifest.get("receipts") != expected_receipt_rows
        or checkpoint_payload.get("completed_receipts")
        != expected_receipt_rows
        or manifest.get("checkpoint") != {
            **checkpoint["record"],
            "completed_probe_count": EXPECTED_RUNS,
        }
    ):
        raise KrennStarPivotCASArtifactError(
            "embedded control files disagree with embedded runs"
        )
    overlap = _maximum_overlap(reconstructed_runs)
    resources = summary["campaign"]["resources"]
    if (
        manifest.get("latest_invocation_workers")
        != summary["campaign"]["latest_invocation_workers"]
        or manifest.get("maximum_concurrent_probes_observed")
        != overlap
        or summary["campaign"]["maximum_concurrent_probes_observed"]
        != overlap
        or resources != {
            "cpus_per_probe": CPUS_PER_PROBE,
            "memory_gib_per_probe": MEMORY_GIB_PER_PROBE,
            "maximum_concurrent_cpus": overlap * CPUS_PER_PROBE,
            "maximum_concurrent_memory_gib": (
                overlap * MEMORY_GIB_PER_PROBE
            ),
            "network": "none",
        }
    ):
        raise KrennStarPivotCASArtifactError(
            "portable worker/resource accounting changed"
        )
    aggregate = summary.get("aggregate")
    if aggregate != {
        "outcome_counts": {"timeout-after-parse": EXPECTED_RUNS},
        "parse_success_count": EXPECTED_RUNS,
        "timeout_return_code_124_count": EXPECTED_RUNS,
        "standard_basis_completion_count": 0,
        "all_elapsed_at_least_600_seconds": True,
        "characteristic_zero_slice_decisions": 0,
        "finite_exact_counterexamples_verified": 0,
    }:
        raise KrennStarPivotCASArtifactError(
            "portable aggregate or claim accounting changed"
        )


def generate_star_pivot_cas_summary_bundle(
    run_directory: Path,
    output_directory: Path,
) -> Path:
    """Replay scratch evidence and write a relocatable two-file bundle."""

    summary = summarize_completed_campaign(run_directory)
    output_directory = Path(output_directory)
    if output_directory.exists() and (
        output_directory.is_symlink()
        or not output_directory.is_dir()
    ):
        raise KrennStarPivotCASArtifactError(
            "bundle output must be a real directory"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(
        path.name not in ALL_FILES for path in output_directory.iterdir()
    ):
        raise KrennStarPivotCASArtifactError(
            "bundle output contains unreviewed files"
        )
    summary_path = output_directory / SUMMARY_FILE
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(output_directory / BUNDLE_MANIFEST_FILE, {
        "schema": BUNDLE_MANIFEST_SCHEMA,
        "artifact": _record(summary_path),
        "producer": _source_record(PRODUCER_SOURCE),
        "source_inputs": [
            _source_record(label) for label in SOURCE_INPUTS
        ],
        "portable_without_scratch": True,
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    })
    return summary_path


def verify_star_pivot_cas_summary_bundle(
    directory: Path,
) -> VerifiedStarPivotCASArtifact:
    """Verify the two-file bundle without consulting scratch storage."""

    directory = Path(directory)
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {path.name for path in directory.iterdir()} != set(ALL_FILES)
    ):
        raise KrennStarPivotCASArtifactError(
            "portable bundle inventory changed"
        )
    summary_path = directory / SUMMARY_FILE
    manifest = _load_json(
        directory / BUNDLE_MANIFEST_FILE, "portable bundle manifest"
    )
    summary = _load_json(summary_path, "portable CAS summary")
    if (
        set(manifest) != {
            "schema",
            "artifact",
            "producer",
            "source_inputs",
            "portable_without_scratch",
            "claim_boundary",
        }
        or manifest["schema"] != BUNDLE_MANIFEST_SCHEMA
        or manifest["artifact"] != _record(summary_path)
        or manifest["producer"] != _source_record(PRODUCER_SOURCE)
        or manifest["source_inputs"] != [
            _source_record(label) for label in SOURCE_INPUTS
        ]
        or manifest["portable_without_scratch"] is not True
        or manifest["claim_boundary"] != _CLAIM_BOUNDARY
    ):
        raise KrennStarPivotCASArtifactError(
            "portable bundle manifest failed replay"
        )
    _verify_portable_summary(summary)
    loaded = VerifiedStarPivotCASArtifact(
        directory=directory.resolve(),
        summary=summary,
        manifest=manifest,
        checks=(
            ("bundle_inventory_exact", True),
            ("bundle_hash_replayed", True),
            ("source_hashes_replayed", True),
            ("deterministic_inputs_regenerated", True),
            ("three_receipt_hashes_replayed", True),
            ("six_stream_hashes_replayed", True),
            ("three_outcomes_reclassified", True),
            ("checkpoint_manifest_consistency_replayed", True),
            ("portable_without_scratch", True),
            ("claim_boundaries_fail_closed", True),
        ),
    )
    if not loaded.valid:
        raise KrennStarPivotCASArtifactError(
            "portable CAS summary checks failed"
        )
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify the portable star-pivot CAS summary."
        )
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--from-run-directory",
        type=Path,
        help=(
            "completed ten_minute_runs directory; required only while "
            "generating the portable bundle"
        ),
    )
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        if arguments.from_run_directory is None:
            parser.error(
                "--from-run-directory is required when generating"
            )
        generate_star_pivot_cas_summary_bundle(
            arguments.from_run_directory,
            arguments.output_directory,
        )
    loaded = verify_star_pivot_cas_summary_bundle(
        arguments.output_directory
    )
    print(json.dumps({
        "valid": loaded.valid,
        "runs": loaded.summary["campaign"]["run_count"],
        "parse_success_count": loaded.summary["aggregate"][
            "parse_success_count"
        ],
        "timeout_return_code_124_count": loaded.summary[
            "aggregate"
        ]["timeout_return_code_124_count"],
        "standard_basis_completion_count": loaded.summary[
            "aggregate"
        ]["standard_basis_completion_count"],
        "portable_without_scratch": True,
        "finite_affine_membership_status": loaded.summary[
            "claim_boundary"
        ]["finite_affine_membership_status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
