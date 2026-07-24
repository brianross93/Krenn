"""Replayable exact artifact for the ``n=6,d=3`` source-ideal result.

The bundle is intentionally small: ``certificate.json`` is the canonical
output of :func:`source_ideal.exact_summary`, and ``manifest.json`` records
the certificate plus the exact source inputs needed to reproduce it.
Verification replays both ledgers and recomputes the complete exact summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.source_ideal import (
    SOURCE_IDEAL_SCHEMA,
    exact_summary,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "krenn.n6_d3.source_ideal.artifact_manifest.v1"

CERTIFICATE_FILE = "certificate.json"
MANIFEST_FILE = "manifest.json"
ARTIFACT_FILES = (CERTIFICATE_FILE,)
ALL_FILES = (*ARTIFACT_FILES, MANIFEST_FILE)

SOURCE_INPUTS = (
    "experiments/krenn_quantum_graph/source_ideal.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
)
COMMITTED_BUNDLE_DIRECTORY = (
    ROOT / "results/krenn_quantum_graph/n6_d3_source_ideal_k1"
)
MAX_JSON_BYTES = 1024 * 1024

EXACT_CHECK_NAMES = (
    "artifact_inventory_exact",
    "artifact_hash_replayed",
    "producer_source_replayed",
    "input_source_ledger_replayed",
    "exact_summary_recomputed",
    "D_outside_J_mix_at_k1_replayed",
    "radical_nonmembership_not_claimed",
    "GHZ_nonexistence_not_claimed",
)


class KrennSourceIdealArtifactError(RuntimeError):
    """A source-ideal artifact failed integrity or semantic replay."""


@dataclass(frozen=True)
class LoadedSourceIdealArtifact:
    """A fully loaded source-ideal bundle and its replay checks."""

    directory: Path
    certificate: Mapping
    manifest: Mapping
    checks: tuple[tuple[str, bool], ...]

    @property
    def exact(self) -> bool:
        """Whether the complete, ordered check inventory passed."""

        return self.checks == tuple(
            (name, True) for name in EXACT_CHECK_NAMES
        )


def _canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _local_record(path: Path) -> dict:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _canonical_source_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(
        b"\r", b"\n"
    )


def _source_record(path: Path) -> dict:
    requested = Path(path)
    if requested.is_symlink():
        raise KrennSourceIdealArtifactError(
            "source input must not be a symbolic link"
        )
    path = requested.resolve()
    if not path.is_file():
        raise KrennSourceIdealArtifactError(
            "source input is absent"
        )
    try:
        label = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError as error:
        raise KrennSourceIdealArtifactError(
            "source input is outside the repository"
        ) from error
    payload = _canonical_source_bytes(path)
    return {
        "path": label,
        "canonical_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hash_mode": "canonical-lf-text-v1",
    }


def _safe_local_path(directory: Path, label: str) -> Path:
    if type(label) is not str:
        raise KrennSourceIdealArtifactError(
            "artifact path must be a string"
        )
    pure = Path(label)
    if (
        not label
        or label in (".", "..")
        or pure.is_absolute()
        or bool(pure.drive)
        or len(pure.parts) != 1
        or "/" in label
        or "\\" in label
        or ":" in label
    ):
        raise KrennSourceIdealArtifactError(
            "artifact path escaped its bundle"
        )
    path = directory / label
    if path.is_symlink() or not path.is_file():
        raise KrennSourceIdealArtifactError(
            f"artifact is absent or linked: {label}"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise KrennSourceIdealArtifactError(
            f"artifact could not be resolved: {label}"
        ) from error
    if resolved.parent != directory:
        raise KrennSourceIdealArtifactError(
            "artifact path escaped its bundle"
        )
    return resolved


def _verify_local_record(directory: Path, record: Mapping) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise KrennSourceIdealArtifactError(
            "artifact record schema changed"
        )
    if (
        type(record["path"]) is not str
        or type(record["bytes"]) is not int
        or record["bytes"] < 0
        or type(record["sha256"]) is not str
        or len(record["sha256"]) != 64
    ):
        raise KrennSourceIdealArtifactError(
            "artifact record types changed"
        )
    path = _safe_local_path(directory, record["path"])
    if (
        path.stat().st_size != record["bytes"]
        or _sha256(path) != record["sha256"]
    ):
        raise KrennSourceIdealArtifactError(
            f"artifact hash replay failed: {path.name}"
        )
    return path


def _safe_source_path(label: str) -> Path:
    if type(label) is not str:
        raise KrennSourceIdealArtifactError(
            "source-ledger path must be a string"
        )
    parts = label.split("/")
    if (
        not label
        or "\\" in label
        or any(part in ("", ".", "..") for part in parts)
        or Path(label).is_absolute()
        or bool(Path(label).drive)
    ):
        raise KrennSourceIdealArtifactError(
            "source-ledger path is unsafe"
        )
    candidate = ROOT
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise KrennSourceIdealArtifactError(
                "source-ledger input is linked"
            )
    if not candidate.is_file():
        raise KrennSourceIdealArtifactError(
            "source-ledger input is absent"
        )
    path = candidate.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise KrennSourceIdealArtifactError(
            "source-ledger path escaped the repository"
        ) from error
    return path


def _verify_source_record(record: Mapping) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "canonical_bytes",
        "sha256",
        "hash_mode",
    }:
        raise KrennSourceIdealArtifactError(
            "source-ledger record schema changed"
        )
    if (
        type(record["path"]) is not str
        or type(record["canonical_bytes"]) is not int
        or record["canonical_bytes"] < 0
        or type(record["sha256"]) is not str
        or len(record["sha256"]) != 64
        or record["hash_mode"] != "canonical-lf-text-v1"
    ):
        raise KrennSourceIdealArtifactError(
            "source-ledger record types changed"
        )
    path = _safe_source_path(record["path"])
    expected = _source_record(path)
    if _canonical_json_bytes(record) != _canonical_json_bytes(expected):
        raise KrennSourceIdealArtifactError(
            f"source-ledger replay failed: {record['path']}"
        )
    return path


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennSourceIdealArtifactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Mapping:
    try:
        if path.is_symlink() or not path.is_file():
            raise KrennSourceIdealArtifactError(
                f"{label} is absent or linked"
            )
        if path.stat().st_size > MAX_JSON_BYTES:
            raise KrennSourceIdealArtifactError(
                f"{label} exceeds the reviewed size bound"
            )
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except KrennSourceIdealArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennSourceIdealArtifactError(
            f"could not decode {label}"
        ) from error
    if not isinstance(payload, dict):
        raise KrennSourceIdealArtifactError(
            f"{label} is not a JSON object"
        )
    if raw != _canonical_json_bytes(payload):
        raise KrennSourceIdealArtifactError(
            f"{label} is not canonical round-trippable JSON"
        )
    return payload


def _certificate() -> dict:
    payload = dict(exact_summary())
    claims = payload.get("claims", {})
    if (
        payload.get("schema") != SOURCE_IDEAL_SCHEMA
        or claims.get("D_in_J_mix_at_k1") is not False
        or claims.get("exact_Q_nonmembership_proved") is not True
        or claims.get("D_outside_radical_J_mix_proved") is not False
        or claims.get("GHZ_nonexistence_proved") is not False
    ):
        raise KrennSourceIdealArtifactError(
            "exact source-ideal claim boundary changed"
        )
    return payload


def _manifest(directory: Path) -> dict:
    inputs = [
        _source_record(ROOT / label) for label in SOURCE_INPUTS
    ]
    return {
        "schema": MANIFEST_SCHEMA,
        "producer": _source_record(Path(__file__)),
        "artifacts": [
            _local_record(directory / label)
            for label in sorted(ARTIFACT_FILES)
        ],
        "inputs": sorted(inputs, key=lambda row: row["path"]),
    }


def generate_source_ideal_bundle(
    output_directory: Path | str,
) -> Path:
    """Write the two-file bundle atomically and replay it immediately."""

    requested = Path(output_directory)
    if requested.is_symlink():
        raise KrennSourceIdealArtifactError(
            "source-ideal output directory must not be linked"
        )
    directory = requested.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    entries = tuple(directory.iterdir())
    unexpected = {
        path.name for path in entries if path.name not in ALL_FILES
    }
    if unexpected:
        raise KrennSourceIdealArtifactError(
            "refusing a directory with unexpected files: "
            f"{sorted(unexpected)}"
        )
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise KrennSourceIdealArtifactError(
            "existing bundle entries must be regular local files"
        )

    _write_json_atomic(directory / CERTIFICATE_FILE, _certificate())
    _write_json_atomic(directory / MANIFEST_FILE, _manifest(directory))
    verify_source_ideal_bundle(directory)
    return directory / CERTIFICATE_FILE


def verify_source_ideal_bundle(
    output_directory: Path | str,
) -> LoadedSourceIdealArtifact:
    """Replay exact inventory, hashes, source provenance, and semantics."""

    requested = Path(output_directory)
    if requested.is_symlink() or not requested.is_dir():
        raise KrennSourceIdealArtifactError(
            "source-ideal bundle directory is absent or linked"
        )
    directory = requested.resolve()
    entries = tuple(directory.iterdir())
    if (
        {path.name for path in entries} != set(ALL_FILES)
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise KrennSourceIdealArtifactError(
            "source-ideal bundle file inventory changed"
        )

    manifest_path = _safe_local_path(directory, MANIFEST_FILE)
    manifest = _load_json(manifest_path, "source-ideal manifest")
    if set(manifest) != {
        "schema",
        "producer",
        "artifacts",
        "inputs",
    } or manifest.get("schema") != MANIFEST_SCHEMA:
        raise KrennSourceIdealArtifactError(
            "source-ideal manifest schema changed"
        )

    records = manifest.get("artifacts")
    if (
        not isinstance(records, list)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise KrennSourceIdealArtifactError(
            "source-ideal artifact ledger changed"
        )
    labels = tuple(record.get("path") for record in records)
    if labels != tuple(sorted(ARTIFACT_FILES)):
        raise KrennSourceIdealArtifactError(
            "source-ideal artifact ledger is not exact and sorted"
        )
    replayed = {
        label: _verify_local_record(directory, record)
        for label, record in zip(labels, records)
    }

    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise KrennSourceIdealArtifactError(
            "source-ideal producer record changed"
        )
    _verify_source_record(producer)
    expected_producer = _source_record(Path(__file__))
    if (
        _canonical_json_bytes(producer)
        != _canonical_json_bytes(expected_producer)
    ):
        raise KrennSourceIdealArtifactError(
            "source-ideal producer identity changed"
        )

    input_records = manifest.get("inputs")
    if (
        not isinstance(input_records, list)
        or any(not isinstance(record, dict) for record in input_records)
    ):
        raise KrennSourceIdealArtifactError(
            "source-ideal input ledger changed"
        )
    for record in input_records:
        _verify_source_record(record)
    expected_inputs = sorted(
        (
            _source_record(ROOT / label)
            for label in SOURCE_INPUTS
        ),
        key=lambda row: row["path"],
    )
    if (
        _canonical_json_bytes(input_records)
        != _canonical_json_bytes(expected_inputs)
    ):
        raise KrennSourceIdealArtifactError(
            "source-ideal input ledger changed"
        )

    certificate = _load_json(
        replayed[CERTIFICATE_FILE],
        "source-ideal certificate",
    )
    expected_certificate = _certificate()
    if set(certificate) != set(expected_certificate):
        raise KrennSourceIdealArtifactError(
            "source-ideal certificate schema changed"
        )
    if (
        _canonical_json_bytes(certificate)
        != _canonical_json_bytes(expected_certificate)
    ):
        raise KrennSourceIdealArtifactError(
            "source-ideal certificate failed semantic replay"
        )

    claims = certificate["claims"]
    checks = (
        ("artifact_inventory_exact", True),
        ("artifact_hash_replayed", True),
        ("producer_source_replayed", True),
        ("input_source_ledger_replayed", True),
        ("exact_summary_recomputed", True),
        (
            "D_outside_J_mix_at_k1_replayed",
            claims["D_in_J_mix_at_k1"] is False
            and claims["exact_Q_nonmembership_proved"] is True,
        ),
        (
            "radical_nonmembership_not_claimed",
            claims["D_outside_radical_J_mix_proved"] is False,
        ),
        (
            "GHZ_nonexistence_not_claimed",
            claims["GHZ_nonexistence_proved"] is False,
        ),
    )
    loaded = LoadedSourceIdealArtifact(
        directory=directory,
        certificate=certificate,
        manifest=manifest,
        checks=checks,
    )
    if not loaded.exact:
        raise KrennSourceIdealArtifactError(
            "source-ideal exact checks failed"
        )
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the exact k=1 source-ideal bundle."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        generate_source_ideal_bundle(arguments.output_directory)
    loaded = verify_source_ideal_bundle(arguments.output_directory)
    print(
        json.dumps(
            {
                "exact": loaded.exact,
                "D_in_J_mix_at_k1": loaded.certificate["claims"][
                    "D_in_J_mix_at_k1"
                ],
                "exact_Q_nonmembership_proved": loaded.certificate[
                    "claims"
                ]["exact_Q_nonmembership_proved"],
                "D_outside_radical_J_mix_proved": False,
                "GHZ_nonexistence_proved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
