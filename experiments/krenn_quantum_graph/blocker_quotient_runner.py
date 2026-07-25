"""Bounded local Singular runner for the ``n=8`` K5 quotient pilot.

The runner executes at most one backend process, with two CPUs, no network,
and explicit memory and wall-time limits.  Large transcripts and disposable
backend data stay under ``D:\\KrennScratch\\counterexample_search``.

Every completed finite quotient is parsed into multiplication matrices and
replayed by :mod:`blocker_quotient_reconnaissance`.  The modular probe is
reconnaissance only.  The rational probe proves a proper affine ideal only
for the deterministic coefficient specialization, which does not impose the
Krenn EqSystem.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.blocker_quotient_reconnaissance import (
    COMPLETION_MARKER,
    DEFAULT_SPECIALIZATION_SEED,
    MAXIMUM_TRANSCRIPT_BYTES,
    PARSE_MARKER,
    KrennBlockerQuotientError,
    chart_sha256,
    deterministic_k5_blocker_chart,
    parse_singular_blocker_output,
    singular_blocker_quotient_script,
    verify_quotient_representation,
)


RUN_SCHEMA = "krenn-n8-k5-blocker-quotient-run-v1"
DEFAULT_SCRATCH_ROOT = Path(
    r"D:\KrennScratch\counterexample_search\n8_k5_quotient_v2"
)
DOCKER_IMAGE = "krenn-n8-singular:ubuntu24.04-v1"
DOCKER_IMAGE_ID = (
    "sha256:f9d3378746fb922f73802e52117c8e67893c45af"
    "2497275e50b5a940b1a751df"
)
CPUS = 2
MAXIMUM_WORKERS = 1
HOST_GRACE_SECONDS = 60


class KrennBlockerQuotientRunnerError(RuntimeError):
    """A bounded CAS input, execution, or receipt failed closed."""


@dataclass(frozen=True)
class BlockerQuotientProbe:
    """One predeclared fixed-specialization Singular computation."""

    key: str
    characteristic: int
    algorithm: str
    engine_budget_seconds: int
    memory_gib: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not self.key
            or not all(
                character.islower()
                or character.isdigit()
                or character == "_"
                for character in self.key
            )
        ):
            raise KrennBlockerQuotientRunnerError(
                "probe key must be a lowercase safe identifier"
            )
        if (
            isinstance(self.characteristic, bool)
            or not isinstance(self.characteristic, int)
            or self.characteristic not in (0, 31)
            or self.algorithm != "slimgb"
            or isinstance(self.engine_budget_seconds, bool)
            or not isinstance(self.engine_budget_seconds, int)
            or not 1 <= self.engine_budget_seconds <= 1_800
            or isinstance(self.memory_gib, bool)
            or not isinstance(self.memory_gib, int)
            or not 1 <= self.memory_gib <= 8
        ):
            raise KrennBlockerQuotientRunnerError(
                "probe exceeds the reviewed field or resource plan"
            )


MODULAR_PROBE = BlockerQuotientProbe(
    key="seed_80320260725_p31_slimgb",
    characteristic=31,
    algorithm="slimgb",
    engine_budget_seconds=600,
    memory_gib=4,
)
RATIONAL_PROBE = BlockerQuotientProbe(
    key="seed_80320260725_q_slimgb",
    characteristic=0,
    algorithm="slimgb",
    engine_budget_seconds=1_800,
    memory_gib=8,
)
PROBES = (MODULAR_PROBE, RATIONAL_PROBE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise KrennBlockerQuotientRunnerError(
            f"{path.name} is not strict JSON"
        ) from error
    if type(payload) is not dict:
        raise KrennBlockerQuotientRunnerError(
            f"{path.name} must contain one JSON object"
        )
    return payload


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise KrennBlockerQuotientRunnerError(
            f"refusing non-regular output target {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise KrennBlockerQuotientRunnerError(
            f"refusing an existing temporary target {temporary}"
        )
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _real_directory(path: Path, *, create: bool) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise KrennBlockerQuotientRunnerError(
                f"{path} must be a real directory"
            )
    elif create:
        path.mkdir(parents=True)
    else:
        raise KrennBlockerQuotientRunnerError(
            f"missing directory {path}"
        )


def _probe_script(probe: BlockerQuotientProbe) -> str:
    chart = deterministic_k5_blocker_chart(
        seed=DEFAULT_SPECIALIZATION_SEED
    )
    return singular_blocker_quotient_script(
        chart,
        characteristic=probe.characteristic,
        algorithm=probe.algorithm,
    )


def _probe_payload(
    probe: BlockerQuotientProbe,
) -> dict[str, object]:
    return {
        "key": probe.key,
        "characteristic": probe.characteristic,
        "algorithm": probe.algorithm,
        "specialization_seed": DEFAULT_SPECIALIZATION_SEED,
        "chart_sha256": chart_sha256(
            deterministic_k5_blocker_chart()
        ),
        "engine_budget_seconds": probe.engine_budget_seconds,
        "cpus": CPUS,
        "memory_gib": probe.memory_gib,
    }


def _engine_payload(
    probe: BlockerQuotientProbe,
    scratch_root: Path,
) -> dict[str, object]:
    return {
        "image": DOCKER_IMAGE,
        "image_id": DOCKER_IMAGE_ID,
        "network": "none",
        "argv": list(docker_argv(probe, scratch_root)),
    }


def prepare_probe_input(
    probe: BlockerQuotientProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, object]:
    """Create or replay one deterministic Singular input exactly."""

    scratch_root = Path(scratch_root)
    _real_directory(scratch_root, create=True)
    input_directory = scratch_root / "inputs"
    _real_directory(input_directory, create=True)
    path = input_directory / f"{probe.key}.sing"
    payload = _probe_script(probe).encode("utf-8")
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise KrennBlockerQuotientRunnerError(
                f"retained probe input changed: {path}"
            )
    else:
        _write_bytes_atomic(path, payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def docker_argv(
    probe: BlockerQuotientProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[str, ...]:
    input_path = Path(
        prepare_probe_input(probe, scratch_root)["path"]
    )
    mount = f"{input_path.parent}:/work:ro"
    return (
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--cpus",
        str(CPUS),
        "--memory",
        f"{probe.memory_gib}g",
        "--entrypoint",
        "timeout",
        "-v",
        mount,
        DOCKER_IMAGE_ID,
        "--signal=TERM",
        "--kill-after=15s",
        f"{probe.engine_budget_seconds}s",
        "Singular",
        f"/work/{input_path.name}",
    )


def inspect_engine() -> str:
    try:
        completed = subprocess.run(
            (
                "docker",
                "image",
                "inspect",
                "--format={{.Id}}",
                DOCKER_IMAGE,
            ),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise KrennBlockerQuotientRunnerError(
            "Docker image inspection failed"
        ) from error
    image_id = completed.stdout.strip()
    if completed.returncode or image_id != DOCKER_IMAGE_ID:
        raise KrennBlockerQuotientRunnerError(
            "the Krenn Singular engine is unavailable or changed"
        )
    return image_id


def _stream_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_transcript(path: Path) -> str:
    if path.stat().st_size > MAXIMUM_TRANSCRIPT_BYTES:
        raise KrennBlockerQuotientRunnerError(
            "backend stdout exceeds the reviewed transcript limit"
        )
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise KrennBlockerQuotientRunnerError(
            "backend stdout is not strict UTF-8 text"
        ) from error


def _outcome_payload(
    probe: BlockerQuotientProbe,
    stdout_text: str,
    return_code: int,
) -> tuple[dict[str, object], object | None]:
    parse_count = stdout_text.splitlines().count(PARSE_MARKER)
    completion_count = stdout_text.splitlines().count(
        COMPLETION_MARKER
    )
    if return_code == 124 and parse_count == 1 and completion_count == 0:
        return (
            {
                "status": "timeout-after-parse",
                "return_code": return_code,
                "parse_marker_observed": True,
                "completion_marker_observed": False,
            },
            None,
        )
    if return_code != 0 or parse_count != 1 or completion_count != 1:
        return (
            {
                "status": "failed-or-invalid-transcript",
                "return_code": return_code,
                "parse_marker_observed": parse_count == 1,
                "completion_marker_observed": completion_count == 1,
            },
            None,
        )
    chart = deterministic_k5_blocker_chart()
    try:
        outcome = parse_singular_blocker_output(
            chart,
            stdout_text,
            characteristic=probe.characteristic,
            algorithm=probe.algorithm,
        )
    except KrennBlockerQuotientError as error:
        return (
            {
                "status": "completed-marker-but-replay-failed",
                "return_code": return_code,
                "parse_marker_observed": True,
                "completion_marker_observed": True,
                "replay_error": str(error),
            },
            None,
        )
    if outcome.unit_ideal:
        status = "completed-unit-ideal"
    elif outcome.representation is not None:
        status = "completed-finite-quotient"
    else:
        status = "completed-positive-dimensional"
    payload = {
        "status": status,
        "return_code": return_code,
        "parse_marker_observed": True,
        "completion_marker_observed": True,
        "timer_ticks": outcome.timer_ticks,
        "groebner_basis_size": outcome.groebner_basis_size,
        "unit_ideal": outcome.unit_ideal,
        "krull_dimension": outcome.krull_dimension,
        "backend_quotient_dimension": (
            outcome.backend_quotient_dimension
        ),
    }
    return payload, outcome


@contextmanager
def _exclusive_campaign_lock(scratch_root: Path):
    """Prevent concurrent backend jobs under one scratch campaign root."""

    _real_directory(scratch_root, create=True)
    output_directory = scratch_root / "runs"
    _real_directory(output_directory, create=True)
    lock_path = output_directory / ".campaign.lock"
    token = f"{os.getpid()}:{time.time_ns()}\n".encode("ascii")
    try:
        with lock_path.open("xb") as lock:
            lock.write(token)
            lock.flush()
            os.fsync(lock.fileno())
    except FileExistsError as error:
        raise KrennBlockerQuotientRunnerError(
            "another quotient backend job owns the campaign lock"
        ) from error
    try:
        yield
    finally:
        if (
            lock_path.is_symlink()
            or not lock_path.is_file()
            or lock_path.read_bytes() != token
        ):
            raise KrennBlockerQuotientRunnerError(
                "campaign lock changed while the backend was running"
            )
        lock_path.unlink()


def _run_probe_locked(
    probe: BlockerQuotientProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Run or replay one bounded backend probe."""

    scratch_root = Path(scratch_root)
    input_record = prepare_probe_input(probe, scratch_root)
    output_directory = scratch_root / "runs"
    _real_directory(output_directory, create=True)
    receipt_path = output_directory / f"{probe.key}.receipt.json"
    if resume and receipt_path.is_file() and not receipt_path.is_symlink():
        return replay_probe(probe, scratch_root)
    if receipt_path.exists():
        raise KrennBlockerQuotientRunnerError(
            f"receipt exists for {probe.key}; use resume"
        )
    stdout_path = output_directory / f"{probe.key}.stdout.txt"
    stderr_path = output_directory / f"{probe.key}.stderr.txt"
    if stdout_path.exists() or stderr_path.exists():
        raise KrennBlockerQuotientRunnerError(
            f"orphaned backend streams exist for {probe.key}"
        )
    image_id = inspect_engine()
    argv = docker_argv(probe, scratch_root)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with (
        stdout_path.open("xb") as stdout,
        stderr_path.open("xb") as stderr,
    ):
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=(
                    probe.engine_budget_seconds
                    + HOST_GRACE_SECONDS
                ),
            )
        except subprocess.TimeoutExpired as error:
            raise KrennBlockerQuotientRunnerError(
                "host timeout exceeded the engine limit and grace period"
            ) from error
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    elapsed = round(time.monotonic() - started, 6)
    stdout_text = _read_transcript(stdout_path)
    outcome_payload, outcome = _outcome_payload(
        probe, stdout_text, completed.returncode
    )
    representation_record = None
    verification = None
    if outcome is not None and outcome.representation is not None:
        representation_path = (
            output_directory / f"{probe.key}.representation.json"
        )
        representation_payload = outcome.representation.to_dict()
        _write_json_atomic(
            representation_path, representation_payload
        )
        representation_record = _stream_record(representation_path)
        verification = verify_quotient_representation(
            deterministic_k5_blocker_chart(),
            outcome.representation,
        )
    receipt = {
        "schema": RUN_SCHEMA,
        "probe": _probe_payload(probe),
        "input": input_record,
        "engine": _engine_payload(probe, scratch_root),
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed,
        "stdout": _stream_record(stdout_path),
        "stderr": _stream_record(stderr_path),
        "outcome": outcome_payload,
        "representation": representation_record,
        "verification": verification,
        "claim_boundary": {
            "eqsystem_constraints_imposed": False,
            "symbolic_90_parameter_quotient_built": False,
            "full_quotient_dimension_independently_certified": False,
            "unit_ideal_independently_certified": False,
            "modular_result_is_Q_or_C_proof": False,
            "modular_lift_proved": False,
            "rational_point_over_Q_proved": False,
            "quotient_reduced_proved": False,
            "quotient_radical_proved": False,
            "distinct_points_counted": False,
            "generic_degree_implies_all_specializations": False,
            "quotient_length_counts_distinct_points": False,
            "all_56_labeled_k5_blockers_checked": False,
            "degree24_k5_covers_degree30_type": False,
            "n8_boundary_escape_proved": False,
            "n8_nonexistence_proved": False,
            "timeout_or_miss_is_proof": False,
        },
    }
    _write_json_atomic(receipt_path, receipt)
    return replay_probe(probe, scratch_root)


def run_probe(
    probe: BlockerQuotientProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Run or replay one bounded backend probe under an exclusive lock."""

    scratch_root = Path(scratch_root)
    with _exclusive_campaign_lock(scratch_root):
        return _run_probe_locked(
            probe, scratch_root, resume=resume
        )


def replay_probe(
    probe: BlockerQuotientProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, object]:
    """Replay hashes, transcript parsing, and matrix verification."""

    scratch_root = Path(scratch_root)
    output_directory = scratch_root / "runs"
    receipt_path = output_directory / f"{probe.key}.receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise KrennBlockerQuotientRunnerError(
            f"missing regular receipt for {probe.key}"
        )
    receipt = _strict_json(receipt_path)
    if set(receipt) != {
        "schema",
        "probe",
        "input",
        "engine",
        "started_at_utc",
        "elapsed_seconds",
        "stdout",
        "stderr",
        "outcome",
        "representation",
        "verification",
        "claim_boundary",
    } or receipt["schema"] != RUN_SCHEMA:
        raise KrennBlockerQuotientRunnerError(
            "backend receipt schema changed"
        )
    if receipt["probe"] != _probe_payload(probe):
        raise KrennBlockerQuotientRunnerError(
            "backend probe metadata changed"
        )
    expected_input = prepare_probe_input(probe, scratch_root)
    if receipt["input"] != expected_input:
        raise KrennBlockerQuotientRunnerError(
            "backend input receipt changed"
        )
    if receipt["engine"] != _engine_payload(probe, scratch_root):
        raise KrennBlockerQuotientRunnerError(
            "backend engine metadata changed"
        )
    try:
        started_at = datetime.fromisoformat(
            receipt["started_at_utc"]
        )
    except (TypeError, ValueError) as error:
        raise KrennBlockerQuotientRunnerError(
            "invalid retained start time"
        ) from error
    elapsed = receipt["elapsed_seconds"]
    if (
        started_at.tzinfo is None
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
        or elapsed
        > probe.engine_budget_seconds + HOST_GRACE_SECONDS
    ):
        raise KrennBlockerQuotientRunnerError(
            "invalid retained timing metadata"
        )
    for stream in ("stdout", "stderr"):
        row = receipt[stream]
        if type(row) is not dict or set(row) != {
            "path", "bytes", "sha256"
        }:
            raise KrennBlockerQuotientRunnerError(
                f"invalid {stream} record"
            )
        if not isinstance(row["path"], str):
            raise KrennBlockerQuotientRunnerError(
                f"invalid {stream} path"
            )
        path = output_directory / row["path"]
        if (
            path.parent != output_directory
            or path.is_symlink()
            or not path.is_file()
            or row != _stream_record(path)
        ):
            raise KrennBlockerQuotientRunnerError(
                f"{stream} hash replay failed"
            )
    stdout_path = output_directory / receipt["stdout"]["path"]
    stdout_text = _read_transcript(stdout_path)
    if type(receipt["outcome"]) is not dict:
        raise KrennBlockerQuotientRunnerError(
            "invalid retained outcome"
        )
    return_code = receipt["outcome"].get("return_code")
    if isinstance(return_code, bool) or not isinstance(
        return_code, int
    ):
        raise KrennBlockerQuotientRunnerError(
            "invalid retained return code"
        )
    replayed_outcome, parsed = _outcome_payload(
        probe, stdout_text, return_code
    )
    if receipt["outcome"] != replayed_outcome:
        raise KrennBlockerQuotientRunnerError(
            "backend outcome replay changed"
        )
    representation_row = receipt["representation"]
    if parsed is not None and parsed.representation is not None:
        if (
            type(representation_row) is not dict
            or set(representation_row)
            != {"path", "bytes", "sha256"}
            or not isinstance(representation_row["path"], str)
        ):
            raise KrennBlockerQuotientRunnerError(
                "missing representation record"
            )
        representation_path = (
            output_directory / representation_row["path"]
        )
        if (
            representation_path.parent != output_directory
            or representation_path.is_symlink()
            or representation_row
            != _stream_record(representation_path)
        ):
            raise KrennBlockerQuotientRunnerError(
                "representation hash replay failed"
            )
        retained_representation = _strict_json(
            representation_path
        )
        if retained_representation != parsed.representation.to_dict():
            raise KrennBlockerQuotientRunnerError(
                "representation semantic replay failed"
            )
        verification = verify_quotient_representation(
            deterministic_k5_blocker_chart(),
            parsed.representation,
        )
        if receipt["verification"] != verification:
            raise KrennBlockerQuotientRunnerError(
                "matrix verification replay changed"
            )
    elif representation_row is not None or receipt["verification"] is not None:
        raise KrennBlockerQuotientRunnerError(
            "non-finite run retained a representation claim"
        )
    required_false = {
        "eqsystem_constraints_imposed",
        "symbolic_90_parameter_quotient_built",
        "full_quotient_dimension_independently_certified",
        "unit_ideal_independently_certified",
        "modular_result_is_Q_or_C_proof",
        "modular_lift_proved",
        "rational_point_over_Q_proved",
        "quotient_reduced_proved",
        "quotient_radical_proved",
        "distinct_points_counted",
        "generic_degree_implies_all_specializations",
        "quotient_length_counts_distinct_points",
        "all_56_labeled_k5_blockers_checked",
        "degree24_k5_covers_degree30_type",
        "n8_boundary_escape_proved",
        "n8_nonexistence_proved",
        "timeout_or_miss_is_proof",
    }
    if (
        type(receipt["claim_boundary"]) is not dict
        or set(receipt["claim_boundary"]) != required_false
        or any(
            value is not False
            for value in receipt["claim_boundary"].values()
        )
    ):
        raise KrennBlockerQuotientRunnerError(
            "backend receipt escalates a reconnaissance claim"
        )
    return receipt


def _probe_by_key(key: str) -> BlockerQuotientProbe:
    for probe in PROBES:
        if probe.key == key:
            return probe
    raise argparse.ArgumentTypeError("unknown quotient probe key")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run one bounded n=8 K5 quotient probe."
    )
    parser.add_argument(
        "probe", type=_probe_by_key, choices=PROBES
    )
    parser.add_argument(
        "--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    receipt = run_probe(
        arguments.probe,
        arguments.scratch_root,
        resume=arguments.resume,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
