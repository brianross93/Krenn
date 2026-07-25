"""Deterministic bounded CAS runner for the localized ``n=6,d=3`` charts.

The seven generic Singular probes are reconnaissance, not proof.  This
runner gives each one a real ten-minute engine budget, keeps Docker offline,
limits every container to two CPUs and eight GiB, and persists byte-exact
stdout, stderr, and replayable receipts under the designated scratch root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence


RUNNER_SCHEMA = "krenn-n6-d3-localized-chart-cas-run-v1"
SUITE_SCHEMA = "krenn-n6-d3-localized-chart-cas-suite-v1"
DEFAULT_SCRATCH_ROOT = Path(
    r"D:\KrennScratch\localized_chart_ideals"
)
OUTPUT_SUBDIRECTORY = "ten_minute_runs"
DOCKER_IMAGE = "hodgepodge-singular:ubuntu24.04"
DOCKER_IMAGE_ID = (
    "sha256:6613ac51738965fafd2ebb2839a02452811ac4e3"
    "e5416cfbbda0485ed1995e75"
)
ENGINE_BUDGET_SECONDS = 600
HOST_TIMEOUT_SECONDS = 660
CPUS_PER_PROBE = 2
MEMORY_GIB_PER_PROBE = 8
MAXIMUM_WORKERS = 6


class KrennLocalizedCASRunnerError(RuntimeError):
    """A CAS input, execution, or retained receipt failed closed."""


@dataclass(frozen=True)
class SingularProbe:
    """One fixed bounded Singular computation."""

    key: str
    ideal: str
    input_subdirectory: str
    script_name: str
    algorithm: str
    characteristic: int
    input_bytes: int
    input_sha256: str
    parse_marker: str
    completion_marker: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "ideal": self.ideal,
            "input_subdirectory": self.input_subdirectory,
            "script_name": self.script_name,
            "algorithm": self.algorithm,
            "characteristic": self.characteristic,
            "input_bytes": self.input_bytes,
            "input_sha256": self.input_sha256,
            "parse_marker": self.parse_marker,
            "completion_marker": self.completion_marker,
            "budget_seconds": ENGINE_BUDGET_SECONDS,
            "cpus": CPUS_PER_PROBE,
            "memory_gib": MEMORY_GIB_PER_PROBE,
        }


PROBES = (
    SingularProbe(
        key="natural_chart_p31_slimgb",
        ideal="natural unshifted chart",
        input_subdirectory="unshifted_slimgb",
        script_name="chart6_char31_slimgb.sing",
        algorithm="slimgb",
        characteristic=31,
        input_bytes=128_719,
        input_sha256=(
            "0fdc5574372dd447100b64829ef584a258069d8942bd895f"
            "8677e0c9b20c09b1"
        ),
        parse_marker="KRENN_CHART_PARSE_OK",
        completion_marker="KRENN_CHART_GROEBNER_DONE",
    ),
    SingularProbe(
        key="derivative_11_p31_std",
        ideal="derivative elimination ambient 11",
        input_subdirectory="derivative_atlas",
        script_name="natural_derivative_11_p31_std.sing",
        algorithm="std",
        characteristic=31,
        input_bytes=200_201,
        input_sha256=(
            "6c16496e96db48d427aa5e1b3e99f122e480a7918edb3ee"
            "04f9f95ed5ee1bbce"
        ),
        parse_marker="KRENN_DERIVATIVE_PARSE_OK",
        completion_marker="KRENN_DERIVATIVE_GROEBNER_DONE",
    ),
    SingularProbe(
        key="derivative_29_p31_std",
        ideal="derivative elimination ambient 29",
        input_subdirectory="derivative_atlas",
        script_name="natural_derivative_29_p31_std.sing",
        algorithm="std",
        characteristic=31,
        input_bytes=199_414,
        input_sha256=(
            "ea22026f710c7de477f495923ced9887a8b10c1a63efe6fc"
            "0c71153a0141f7bd"
        ),
        parse_marker="KRENN_DERIVATIVE_PARSE_OK",
        completion_marker="KRENN_DERIVATIVE_GROEBNER_DONE",
    ),
    SingularProbe(
        key="repair_11_65_p31_std",
        ideal="repair monomial 11,65",
        input_subdirectory="repair_monomial_atlas",
        script_name="repair_11_65_p31_std.sing",
        algorithm="std",
        characteristic=31,
        input_bytes=127_325,
        input_sha256=(
            "b3d4423d5b8bc0cebefd2505a683f1d0a43367c4e77c1b4"
            "f39be44961b0b6624"
        ),
        parse_marker="KRENN_REPAIR_CHART_PARSE_OK",
        completion_marker="KRENN_REPAIR_CHART_GROEBNER_DONE",
    ),
    SingularProbe(
        key="repair_29_47_p31_std",
        ideal="repair monomial 29,47",
        input_subdirectory="repair_monomial_atlas",
        script_name="repair_29_47_p31_std.sing",
        algorithm="std",
        characteristic=31,
        input_bytes=127_325,
        input_sha256=(
            "11414ca087bd8a8033ab24484fc019c7cbec9dcd74140b38"
            "8382c898fe13ab43"
        ),
        parse_marker="KRENN_REPAIR_CHART_PARSE_OK",
        completion_marker="KRENN_REPAIR_CHART_GROEBNER_DONE",
    ),
    SingularProbe(
        key="repair_11_55_133_p31_std",
        ideal="repair monomial 11,55,133",
        input_subdirectory="repair_monomial_atlas",
        script_name="repair_11_55_133_p31_std.sing",
        algorithm="std",
        characteristic=31,
        input_bytes=126_118,
        input_sha256=(
            "d59f13e043d04861e7e2e2a7ab46d5df5dace5feafba55"
            "aee986090e6d99c83e"
        ),
        parse_marker="KRENN_REPAIR_CHART_PARSE_OK",
        completion_marker="KRENN_REPAIR_CHART_GROEBNER_DONE",
    ),
    SingularProbe(
        key="repair_29_55_106_p31_std",
        ideal="repair monomial 29,55,106",
        input_subdirectory="repair_monomial_atlas",
        script_name="repair_29_55_106_p31_std.sing",
        algorithm="std",
        characteristic=31,
        input_bytes=126_115,
        input_sha256=(
            "e4c47a3486cde8a06376d9e0f21fb830822904251d2e721"
            "f839d1dfdc3ebb3ac"
        ),
        parse_marker="KRENN_REPAIR_CHART_PARSE_OK",
        completion_marker="KRENN_REPAIR_CHART_GROEBNER_DONE",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(payload) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_equal(left, right) -> bool:
    try:
        return _json_text(left) == _json_text(right)
    except (TypeError, ValueError):
        return False


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _strict_load(path: Path) -> dict:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise KrennLocalizedCASRunnerError(
            f"{path.name} is not unambiguous strict JSON"
        ) from error
    if type(payload) is not dict:
        raise KrennLocalizedCASRunnerError(
            f"{path.name} must contain one JSON object"
        )
    return payload


def _assert_real_directory(path: Path, *, create: bool) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise KrennLocalizedCASRunnerError(
                f"{path} must be a real directory"
            )
    elif create:
        path.mkdir(parents=True)
    else:
        raise KrennLocalizedCASRunnerError(
            f"required directory does not exist: {path}"
        )


def _assert_regular_target(path: Path) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise KrennLocalizedCASRunnerError(
            f"refusing non-regular output target {path}"
        )


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    _assert_regular_target(path)
    temporary = path.with_name(f".{path.name}.tmp")
    _assert_regular_target(temporary)
    with temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        output.write(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


@contextmanager
def _campaign_lock(output_directory: Path):
    lock_path = output_directory / ".campaign.lock"
    token = f"pid={os.getpid()}\n"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise KrennLocalizedCASRunnerError(
            f"another CAS campaign may be active: {lock_path}"
        ) from error
    try:
        os.write(descriptor, token.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        if (
            lock_path.is_file()
            and not lock_path.is_symlink()
            and lock_path.read_text(encoding="ascii") == token
        ):
            lock_path.unlink()


def probe_input_path(
    probe: SingularProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> Path:
    return Path(scratch_root) / probe.input_subdirectory / probe.script_name


def validate_probe_input(
    probe: SingularProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict:
    path = probe_input_path(probe, scratch_root)
    if path.is_symlink() or not path.is_file():
        raise KrennLocalizedCASRunnerError(
            f"missing regular Singular input {path}"
        )
    size = path.stat().st_size
    digest = _sha256(path)
    if size != probe.input_bytes or digest != probe.input_sha256:
        raise KrennLocalizedCASRunnerError(
            f"Singular input fingerprint changed for {probe.key}"
        )
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
    }


def docker_argv(
    probe: SingularProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    *,
    hardened: bool = True,
) -> tuple[str, ...]:
    mount = (
        f"{Path(scratch_root) / probe.input_subdirectory}:/work:ro"
    )
    prefix = (
        "docker",
        "run",
        "--rm",
    )
    if hardened:
        prefix += ("--pull", "never")
    prefix += (
        "--network",
        "none",
        "--cpus",
        str(CPUS_PER_PROBE),
        "--memory",
        f"{MEMORY_GIB_PER_PROBE}g",
        "--entrypoint",
        "timeout",
        "-v",
        mount,
        DOCKER_IMAGE_ID if hardened else DOCKER_IMAGE,
    )
    timeout_arguments = (
        "--signal=TERM",
        "--kill-after=15s",
        f"{ENGINE_BUDGET_SECONDS}s",
    ) if hardened else (f"{ENGINE_BUDGET_SECONDS}s",)
    return prefix + timeout_arguments + (
        "Singular",
        f"/work/{probe.script_name}",
    )


def classify_output(
    probe: SingularProbe,
    *,
    return_code: int,
    stdout_text: str,
) -> dict:
    """Classify markers and selected Singular summary lines fail closed."""

    lines = stdout_text.splitlines()
    parse_observed = lines.count(probe.parse_marker) == 1
    completion_observed = lines.count(probe.completion_marker) == 1
    values = {}
    for key in ("timer_ticks", "basis_size", "unit_ideal"):
        prefix = f"{key}="
        values[key] = [
            line[len(prefix):]
            for line in lines
            if line.startswith(prefix)
        ]
    valid_summary = (
        len(values["timer_ticks"]) == 1
        and values["timer_ticks"][0].isdigit()
        and len(values["basis_size"]) == 1
        and values["basis_size"][0].isdigit()
        and len(values["unit_ideal"]) == 1
        and values["unit_ideal"][0] in ("0", "1")
    )
    selected = (
        {key: entries[0] for key, entries in values.items()}
        if valid_summary else {}
    )
    if (
        return_code == 0
        and parse_observed
        and completion_observed
        and valid_summary
    ):
        status = "completed"
    elif return_code == 124 and parse_observed and not completion_observed:
        status = "timeout-after-parse"
    elif return_code == 137 and parse_observed:
        status = "forced-timeout-or-resource-termination-after-parse"
    elif return_code == 0 and parse_observed and completion_observed:
        status = "invalid-completion-receipt"
    elif not parse_observed:
        status = "failed-before-parse"
    elif completion_observed:
        status = "completion-marker-with-nonzero-exit"
    else:
        status = "failed-after-parse"
    return {
        "status": status,
        "return_code": return_code,
        "parse_marker_observed": parse_observed,
        "completion_marker_observed": completion_observed,
        "selected_singular_output": selected,
        "claim_boundary": {
            "positive_characteristic_result_is_exact_Q_proof": False,
            "timeout_is_a_chart_decision": False,
        },
    }


def _stream_record(path: Path) -> dict:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _receipt_static_fields(
    probe: SingularProbe,
    scratch_root: Path,
    image_id: str,
    *,
    hardened: bool = True,
) -> dict:
    return {
        "schema": RUNNER_SCHEMA,
        "probe": probe.to_dict(),
        "input_script": validate_probe_input(probe, scratch_root),
        "docker": {
            "image": DOCKER_IMAGE,
            "image_id": image_id,
            "argv": list(
                docker_argv(
                    probe, scratch_root, hardened=hardened
                )
            ),
            "network": "none",
        },
    }


def replay_probe_receipt(
    probe: SingularProbe,
    scratch_root: Path,
    image_id: str,
) -> dict:
    output_directory = Path(scratch_root) / OUTPUT_SUBDIRECTORY
    receipt_path = output_directory / f"{probe.key}.receipt.json"
    payload = _strict_load(receipt_path)
    expected_static = _receipt_static_fields(
        probe, Path(scratch_root), image_id, hardened=True
    )
    retained_static = {
        key: payload.get(key)
        for key in ("schema", "probe", "input_script", "docker")
    }
    legacy_static = _receipt_static_fields(
        probe, Path(scratch_root), image_id, hardened=False
    )
    if not (
        _json_equal(retained_static, expected_static)
        or _json_equal(retained_static, legacy_static)
    ):
        raise KrennLocalizedCASRunnerError(
            f"retained static receipt changed for {probe.key}"
        )
    for stream in ("stdout", "stderr"):
        row = payload.get(stream)
        if type(row) is not dict or set(row) != {
            "path", "bytes", "sha256"
        }:
            raise KrennLocalizedCASRunnerError(
                f"invalid {stream} receipt for {probe.key}"
            )
        path = output_directory / row["path"]
        if path.parent != output_directory or path.is_symlink():
            raise KrennLocalizedCASRunnerError(
                f"unsafe {stream} path for {probe.key}"
            )
        if not _json_equal(row, _stream_record(path)):
            raise KrennLocalizedCASRunnerError(
                f"{stream} hash replay failed for {probe.key}"
            )
    return_code = payload.get("outcome", {}).get("return_code")
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        raise KrennLocalizedCASRunnerError(
            f"invalid return code for {probe.key}"
        )
    stdout_path = output_directory / payload["stdout"]["path"]
    replayed_outcome = classify_output(
        probe,
        return_code=return_code,
        stdout_text=stdout_path.read_text(
            encoding="utf-8", errors="replace"
        ),
    )
    if not _json_equal(payload.get("outcome"), replayed_outcome):
        raise KrennLocalizedCASRunnerError(
            f"outcome replay failed for {probe.key}"
        )
    elapsed = payload.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
    ):
        raise KrennLocalizedCASRunnerError(
            f"invalid elapsed time for {probe.key}"
        )
    return payload


def run_probe(
    probe: SingularProbe,
    scratch_root: Path,
    image_id: str,
    *,
    resume: bool,
) -> dict:
    scratch_root = Path(scratch_root)
    output_directory = scratch_root / OUTPUT_SUBDIRECTORY
    receipt_path = output_directory / f"{probe.key}.receipt.json"
    if resume and receipt_path.is_file() and not receipt_path.is_symlink():
        return replay_probe_receipt(probe, scratch_root, image_id)
    if receipt_path.exists():
        raise KrennLocalizedCASRunnerError(
            f"receipt already exists for {probe.key}; use --resume"
        )
    static = _receipt_static_fields(probe, scratch_root, image_id)
    stdout_path = output_directory / f"{probe.key}.stdout.txt"
    stderr_path = output_directory / f"{probe.key}.stderr.txt"
    _assert_regular_target(stdout_path)
    _assert_regular_target(stderr_path)
    if stdout_path.exists() or stderr_path.exists():
        raise KrennLocalizedCASRunnerError(
            f"orphaned logs exist for {probe.key}; refusing to truncate"
        )
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    with (
        stdout_path.open("wb") as stdout_file,
        stderr_path.open("wb") as stderr_file,
    ):
        try:
            completed = subprocess.run(
                static["docker"]["argv"],
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                timeout=HOST_TIMEOUT_SECONDS,
            )
            stdout_file.flush()
            stderr_file.flush()
            os.fsync(stdout_file.fileno())
            os.fsync(stderr_file.fileno())
        except subprocess.TimeoutExpired as error:
            raise KrennLocalizedCASRunnerError(
                f"host timeout exceeded for {probe.key}"
            ) from error
    elapsed = round(time.monotonic() - started, 6)
    stdout_text = stdout_path.read_text(
        encoding="utf-8", errors="replace"
    )
    receipt = {
        **static,
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": elapsed,
        "stdout": _stream_record(stdout_path),
        "stderr": _stream_record(stderr_path),
        "outcome": classify_output(
            probe,
            return_code=completed.returncode,
            stdout_text=stdout_text,
        ),
    }
    _write_json_atomic(receipt_path, receipt)
    return replay_probe_receipt(probe, scratch_root, image_id)


def inspect_docker_image() -> str:
    try:
        completed = subprocess.run(
            (
                "docker",
                "image",
                "inspect",
                "--format={{.Id}}",
                DOCKER_IMAGE,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise KrennLocalizedCASRunnerError(
            "Docker image inspection failed"
        ) from error
    image_id = completed.stdout.strip()
    if completed.returncode or image_id != DOCKER_IMAGE_ID:
        raise KrennLocalizedCASRunnerError(
            "the retained Singular Docker image changed"
        )
    return image_id


def _run_probe_suite_unlocked(
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    *,
    workers: int = 3,
    resume: bool = False,
) -> dict:
    """Run or replay all seven probes and retain a suite manifest."""

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAXIMUM_WORKERS
    ):
        raise KrennLocalizedCASRunnerError(
            f"workers must be in range(1,{MAXIMUM_WORKERS + 1})"
        )
    scratch_root = Path(scratch_root)
    _assert_real_directory(scratch_root, create=False)
    for probe in PROBES:
        validate_probe_input(probe, scratch_root)
    output_directory = scratch_root / OUTPUT_SUBDIRECTORY
    _assert_real_directory(output_directory, create=True)
    image_id = inspect_docker_image()
    receipts_by_key = {}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for probe in PROBES:
            print(
                f"START {probe.key} budget={ENGINE_BUDGET_SECONDS}s",
                flush=True,
            )
            future = executor.submit(
                run_probe,
                probe,
                scratch_root,
                image_id,
                resume=resume,
            )
            futures[future] = probe
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
            f"one or more CAS probes failed: {failures}"
        )
    receipt_rows = []
    for probe in PROBES:
        receipt_path = (
            output_directory / f"{probe.key}.receipt.json"
        )
        receipt_rows.append({
            "key": probe.key,
            "path": receipt_path.name,
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256(receipt_path),
            "outcome": receipts_by_key[probe.key]["outcome"],
            "elapsed_seconds":
                receipts_by_key[probe.key]["elapsed_seconds"],
        })
    events = []
    for receipt in receipts_by_key.values():
        start = datetime.fromisoformat(receipt["started_at_utc"])
        end = start + timedelta(
            seconds=receipt["elapsed_seconds"]
        )
        events.extend(((start, 1), (end, -1)))
    active = 0
    maximum_active = 0
    for _, delta in sorted(events, key=lambda row: (row[0], row[1])):
        active += delta
        maximum_active = max(maximum_active, active)
    manifest = {
        "schema": SUITE_SCHEMA,
        "scratch_root": str(scratch_root),
        "latest_invocation_workers": workers,
        "maximum_concurrent_probes_observed": maximum_active,
        "maximum_concurrent_cpus":
            maximum_active * CPUS_PER_PROBE,
        "maximum_concurrent_memory_gib":
            maximum_active * MEMORY_GIB_PER_PROBE,
        "random_seeds": [],
        "docker_image": DOCKER_IMAGE,
        "docker_image_id": image_id,
        "engine_budget_seconds_per_probe": ENGINE_BUDGET_SECONDS,
        "receipts": receipt_rows,
        "claim_boundary": {
            "positive_characteristic_result_is_exact_Q_proof": False,
            "timeout_is_a_chart_decision": False,
        },
    }
    _write_json_atomic(
        output_directory / "ten_minute_run_manifest.json",
        manifest,
    )
    return manifest


def run_probe_suite(
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    *,
    workers: int = 3,
    resume: bool = False,
) -> dict:
    """Run one locked suite so concurrent invocations cannot oversubscribe."""

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAXIMUM_WORKERS
    ):
        raise KrennLocalizedCASRunnerError(
            f"workers must be in range(1,{MAXIMUM_WORKERS + 1})"
        )
    scratch_root = Path(scratch_root)
    _assert_real_directory(scratch_root, create=False)
    output_directory = scratch_root / OUTPUT_SUBDIRECTORY
    _assert_real_directory(output_directory, create=True)
    with _campaign_lock(output_directory):
        return _run_probe_suite_unlocked(
            scratch_root,
            workers=workers,
            resume=resume,
        )


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
        description="Run seven bounded ten-minute localized-chart probes."
    )
    parser.add_argument(
        "--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT
    )
    parser.add_argument("--workers", type=_parse_workers, default=3)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = run_probe_suite(
        arguments.scratch_root,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
