"""Replayable exact artifact for the ``n=6,d=3`` radical preflight.

The committed bundle contains the exact ``k=2`` dimension/memory preflight,
the unimodular support-row obstruction, exact contraction-minor receipts,
the exact matching-circuit mechanism, and a separately labeled nonproof
one-shell reconnaissance receipt.  Verification replays canonical JSON,
hashes, the complete source ledger, producer semantics, and independent
symbolic verifiers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.higher_power_source_ideal import (
    exact_k2_preflight,
)
from experiments.krenn_quantum_graph.one_shell_reconnaissance import (
    KrennOneShellReconnaissanceError,
    one_shell_reconnaissance_receipt,
    verify_one_shell_reconnaissance_receipt,
)
from experiments.krenn_quantum_graph.mechanism_audit import (
    KrennMechanismAuditError,
    exact_mechanism_summary,
    verify_mechanism_payload,
)
from experiments.krenn_quantum_graph.radical_obstruction import (
    exact_radical_obstruction_summary,
)
from experiments.krenn_quantum_graph.radical_obstruction_verifier import (
    KrennRadicalIndependentVerificationError,
    verify_support_obstruction_payload,
)


ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE_SCHEMA = (
    "krenn.n6_d3.radical_obstruction.exact_bundle.v1"
)
MANIFEST_SCHEMA = (
    "krenn.n6_d3.radical_obstruction.artifact_manifest.v1"
)

CERTIFICATE_FILE = "certificate.json"
RECONNAISSANCE_FILE = "reconnaissance.json"
MANIFEST_FILE = "manifest.json"
ARTIFACT_FILES = (CERTIFICATE_FILE, RECONNAISSANCE_FILE)
ALL_FILES = (*ARTIFACT_FILES, MANIFEST_FILE)
MAX_JSON_BYTES = 4 * 1024 * 1024

SOURCE_INPUTS = (
    "experiments/krenn_quantum_graph/higher_power_source_ideal.py",
    "experiments/krenn_quantum_graph/mechanism_audit.py",
    "experiments/krenn_quantum_graph/one_shell_full_recompute.py",
    "experiments/krenn_quantum_graph/one_shell_reconnaissance.py",
    "experiments/krenn_quantum_graph/radical_obstruction.py",
    "experiments/krenn_quantum_graph/radical_obstruction_verifier.py",
    "experiments/krenn_quantum_graph/source_ideal.py",
    "experiments/krenn_quantum_graph/support_extension.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "experiments/krenn_quantum_graph/ternary_search.py",
    "experiments/krenn_quantum_graph/witness.py",
)
COMMITTED_BUNDLE_DIRECTORY = (
    ROOT / "results/krenn_quantum_graph/n6_d3_radical_obstruction"
)

EXACT_CHECK_NAMES = (
    "artifact_inventory_exact",
    "artifact_hash_replayed",
    "producer_source_replayed",
    "input_source_ledger_replayed",
    "exact_summaries_recomputed",
    "independent_support_orbit_census_replayed",
    "independent_rainbow_singletons_replayed",
    "independent_switch_columns_replayed_symbolically",
    "independent_unimodular_minor_replayed",
    "independent_exact_contraction_minors_replayed",
    "independent_contracted_GHZ_witness_replayed",
    "mechanism_quadratic_uniqueness_replayed",
    "mechanism_ten_K3_3_circuits_replayed",
    "mechanism_support21_spill_replayed",
    "one_shell_receipt_replayed",
    "one_shell_separator_and_global_escapes_replayed",
    "modular_reconnaissance_not_promoted",
    "full_matrix_construction_refused",
    "support_limited_separator_excluded",
    "radical_membership_not_claimed",
    "global_GHZ_nonexistence_not_claimed",
    "exact_affine_membership_not_claimed",
)


class KrennRadicalArtifactError(RuntimeError):
    """The radical-obstruction bundle failed integrity or semantic replay."""


@dataclass(frozen=True)
class LoadedRadicalArtifact:
    directory: Path
    certificate: Mapping
    reconnaissance: Mapping
    manifest: Mapping
    independent_checks: Mapping
    checks: tuple[tuple[str, bool], ...]

    @property
    def exact(self) -> bool:
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


def _canonical_source_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(
        b"\r", b"\n"
    )


def _source_record(path: Path) -> dict:
    requested = Path(path)
    if requested.is_symlink():
        raise KrennRadicalArtifactError(
            "source input must not be a symbolic link"
        )
    path = requested.resolve()
    if not path.is_file():
        raise KrennRadicalArtifactError("source input is absent")
    try:
        label = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError as error:
        raise KrennRadicalArtifactError(
            "source input is outside the repository"
        ) from error
    payload = _canonical_source_bytes(path)
    return {
        "path": label,
        "canonical_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hash_mode": "canonical-lf-text-v1",
    }


def _local_record(path: Path) -> dict:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _safe_local_path(directory: Path, label: str) -> Path:
    if type(label) is not str:
        raise KrennRadicalArtifactError(
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
        raise KrennRadicalArtifactError(
            "artifact path escaped its bundle"
        )
    path = directory / label
    if path.is_symlink() or not path.is_file():
        raise KrennRadicalArtifactError(
            f"artifact is absent or linked: {label}"
        )
    resolved = path.resolve(strict=True)
    if resolved.parent != directory:
        raise KrennRadicalArtifactError(
            "artifact path escaped its bundle"
        )
    return resolved


def _safe_source_path(label: str) -> Path:
    if type(label) is not str:
        raise KrennRadicalArtifactError(
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
        raise KrennRadicalArtifactError(
            "source-ledger path is unsafe"
        )
    candidate = ROOT
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise KrennRadicalArtifactError(
                "source-ledger input is linked"
            )
    if not candidate.is_file():
        raise KrennRadicalArtifactError(
            "source-ledger input is absent"
        )
    path = candidate.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise KrennRadicalArtifactError(
            "source-ledger path escaped the repository"
        ) from error
    return path


def _verify_local_record(directory: Path, record: Mapping) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise KrennRadicalArtifactError(
            "artifact record schema changed"
        )
    if (
        type(record["path"]) is not str
        or type(record["bytes"]) is not int
        or record["bytes"] < 0
        or type(record["sha256"]) is not str
        or len(record["sha256"]) != 64
    ):
        raise KrennRadicalArtifactError(
            "artifact record types changed"
        )
    path = _safe_local_path(directory, record["path"])
    if (
        path.stat().st_size != record["bytes"]
        or _sha256(path) != record["sha256"]
    ):
        raise KrennRadicalArtifactError(
            f"artifact hash replay failed: {path.name}"
        )
    return path


def _verify_source_record(record: Mapping) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "canonical_bytes",
        "sha256",
        "hash_mode",
    }:
        raise KrennRadicalArtifactError(
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
        raise KrennRadicalArtifactError(
            "source-ledger record types changed"
        )
    path = _safe_source_path(record["path"])
    if _canonical_json_bytes(record) != _canonical_json_bytes(
        _source_record(path)
    ):
        raise KrennRadicalArtifactError(
            f"source-ledger replay failed: {record['path']}"
        )
    return path


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennRadicalArtifactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Mapping:
    try:
        if path.is_symlink() or not path.is_file():
            raise KrennRadicalArtifactError(
                f"{label} is absent or linked"
            )
        if path.stat().st_size > MAX_JSON_BYTES:
            raise KrennRadicalArtifactError(
                f"{label} exceeds the reviewed size bound"
            )
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except KrennRadicalArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennRadicalArtifactError(
            f"could not decode {label}"
        ) from error
    if not isinstance(payload, dict):
        raise KrennRadicalArtifactError(
            f"{label} is not a JSON object"
        )
    if raw != _canonical_json_bytes(payload):
        raise KrennRadicalArtifactError(
            f"{label} is not canonical round-trippable JSON"
        )
    return payload


def _certificate() -> dict:
    preflight = dict(exact_k2_preflight())
    support = dict(exact_radical_obstruction_summary())
    mechanism = exact_mechanism_summary()
    claims = support["claims"]
    if (
        preflight["claims"]["D_squared_in_J_mix_decided"] is not False
        or preflight["claims"]["D_in_radical_J_mix_decided"] is not False
        or preflight["memory_preflight"][
            "full_orbit_construction_authorized"
        ] is not False
        or claims["D_squared_in_J_mix_decided"] is not False
        or claims["D_in_radical_J_mix_decided"] is not False
        or claims["global_GHZ_nonexistence_proved"] is not False
        or claims["exact_affine_GHZ_membership_decided"] is not False
        or mechanism["claims"]["D_squared_in_J_mix_decided"] is not False
        or mechanism["claims"]["D_in_radical_J_mix_decided"] is not False
        or mechanism["claims"][
            "global_GHZ_nonexistence_proved"
        ] is not False
    ):
        raise KrennRadicalArtifactError(
            "radical-obstruction claim boundary changed"
        )
    return {
        "schema": CERTIFICATE_SCHEMA,
        "k2_preflight": preflight,
        "mechanism_audit": mechanism,
        "support_obstruction": support,
    }


def _reconnaissance() -> dict:
    payload = one_shell_reconnaissance_receipt()
    claims = payload["claims"]
    if (
        payload["modular_rank_diagnostics"]["role"]
        != "nonproof-reconnaissance"
        or claims["D_squared_global_nonmembership_proved"] is not False
        or claims["D_squared_in_J_mix_decided"] is not False
        or claims["D_in_radical_J_mix_decided"] is not False
        or claims["global_GHZ_nonexistence_proved"] is not False
        or claims["exact_affine_GHZ_membership_decided"] is not False
    ):
        raise KrennRadicalArtifactError(
            "one-shell reconnaissance was promoted to a global claim"
        )
    return payload


def _manifest(directory: Path) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "producer": _source_record(Path(__file__)),
        "artifacts": [
            _local_record(directory / label)
            for label in sorted(ARTIFACT_FILES)
        ],
        "inputs": sorted(
            (
                _source_record(ROOT / label)
                for label in SOURCE_INPUTS
            ),
            key=lambda row: row["path"],
        ),
    }


def generate_radical_obstruction_bundle(
    output_directory: Path | str,
) -> Path:
    requested = Path(output_directory)
    if requested.is_symlink():
        raise KrennRadicalArtifactError(
            "radical output directory must not be linked"
        )
    directory = requested.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    entries = tuple(directory.iterdir())
    unexpected = {
        path.name for path in entries if path.name not in ALL_FILES
    }
    if unexpected:
        raise KrennRadicalArtifactError(
            "refusing a directory with unexpected files: "
            f"{sorted(unexpected)}"
        )
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise KrennRadicalArtifactError(
            "existing bundle entries must be regular local files"
        )
    _write_json_atomic(directory / CERTIFICATE_FILE, _certificate())
    _write_json_atomic(
        directory / RECONNAISSANCE_FILE,
        _reconnaissance(),
    )
    _write_json_atomic(directory / MANIFEST_FILE, _manifest(directory))
    verify_radical_obstruction_bundle(directory)
    return directory / CERTIFICATE_FILE


def verify_radical_obstruction_bundle(
    output_directory: Path | str,
) -> LoadedRadicalArtifact:
    requested = Path(output_directory)
    if requested.is_symlink() or not requested.is_dir():
        raise KrennRadicalArtifactError(
            "radical bundle directory is absent or linked"
        )
    directory = requested.resolve()
    entries = tuple(directory.iterdir())
    if (
        {path.name for path in entries} != set(ALL_FILES)
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise KrennRadicalArtifactError(
            "radical bundle file inventory changed"
        )

    manifest_path = _safe_local_path(directory, MANIFEST_FILE)
    manifest = _load_json(manifest_path, "radical manifest")
    if set(manifest) != {
        "schema",
        "producer",
        "artifacts",
        "inputs",
    } or manifest.get("schema") != MANIFEST_SCHEMA:
        raise KrennRadicalArtifactError(
            "radical manifest schema changed"
        )
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise KrennRadicalArtifactError(
            "radical artifact ledger changed"
        )
    labels = tuple(record.get("path") for record in records)
    if labels != tuple(sorted(ARTIFACT_FILES)):
        raise KrennRadicalArtifactError(
            "radical artifact ledger is not exact and sorted"
        )
    replayed = {
        label: _verify_local_record(directory, record)
        for label, record in zip(labels, records, strict=True)
    }

    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise KrennRadicalArtifactError(
            "radical producer record changed"
        )
    _verify_source_record(producer)
    if _canonical_json_bytes(producer) != _canonical_json_bytes(
        _source_record(Path(__file__))
    ):
        raise KrennRadicalArtifactError(
            "radical producer identity changed"
        )

    input_records = manifest.get("inputs")
    if not isinstance(input_records, list):
        raise KrennRadicalArtifactError(
            "radical input ledger changed"
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
    if _canonical_json_bytes(
        input_records
    ) != _canonical_json_bytes(expected_inputs):
        raise KrennRadicalArtifactError(
            "radical input ledger changed"
        )

    certificate = _load_json(
        replayed[CERTIFICATE_FILE],
        "radical certificate",
    )
    expected_certificate = _certificate()
    if set(certificate) != set(expected_certificate):
        raise KrennRadicalArtifactError(
            "radical certificate schema changed"
        )
    if _canonical_json_bytes(
        certificate
    ) != _canonical_json_bytes(expected_certificate):
        raise KrennRadicalArtifactError(
            "radical certificate failed semantic replay"
        )
    try:
        independent = verify_support_obstruction_payload(
            certificate["support_obstruction"]
        )
    except KrennRadicalIndependentVerificationError as error:
        raise KrennRadicalArtifactError(
            "independent symbolic replay failed"
        ) from error
    try:
        mechanism_checks = verify_mechanism_payload(
            certificate["mechanism_audit"]
        )
    except KrennMechanismAuditError as error:
        raise KrennRadicalArtifactError(
            "matching-mechanism replay failed"
        ) from error
    independent = dict(independent)
    independent.update({
        f"mechanism_{name}": value
        for name, value in mechanism_checks.items()
    })
    reconnaissance = _load_json(
        replayed[RECONNAISSANCE_FILE],
        "one-shell reconnaissance",
    )
    if _canonical_json_bytes(
        reconnaissance
    ) != _canonical_json_bytes(_reconnaissance()):
        raise KrennRadicalArtifactError(
            "one-shell reconnaissance failed semantic replay"
        )
    try:
        one_shell_checks = verify_one_shell_reconnaissance_receipt(
            reconnaissance
        )
    except KrennOneShellReconnaissanceError as error:
        raise KrennRadicalArtifactError(
            "one-shell exact local replay failed"
        ) from error

    preflight = certificate["k2_preflight"]
    claims = certificate["support_obstruction"]["claims"]
    checks = (
        ("artifact_inventory_exact", True),
        ("artifact_hash_replayed", True),
        ("producer_source_replayed", True),
        ("input_source_ledger_replayed", True),
        ("exact_summaries_recomputed", True),
        (
            "independent_support_orbit_census_replayed",
            independent["support_orbit_census_replayed"] is True,
        ),
        (
            "independent_rainbow_singletons_replayed",
            independent["rainbow_singletons_replayed"] is True,
        ),
        (
            "independent_switch_columns_replayed_symbolically",
            independent[
                "switch_columns_replayed_symbolically"
            ] is True,
        ),
        (
            "independent_unimodular_minor_replayed",
            independent["unimodular_minor_replayed"] is True,
        ),
        (
            "independent_exact_contraction_minors_replayed",
            independent["exact_contraction_minors_replayed"] is True,
        ),
        (
            "independent_contracted_GHZ_witness_replayed",
            independent["contracted_GHZ_witness_replayed"] is True,
        ),
        (
            "mechanism_quadratic_uniqueness_replayed",
            independent[
                "mechanism_quadratic_uniqueness_replayed"
            ] is True,
        ),
        (
            "mechanism_ten_K3_3_circuits_replayed",
            independent[
                "mechanism_ten_K3_3_cubic_circuits_replayed"
            ] is True,
        ),
        (
            "mechanism_support21_spill_replayed",
            independent[
                "mechanism_local_three_equation_identity_replayed"
            ] is True
            and independent[
                "mechanism_thirty_nine_spill_terms_replayed"
            ] is True,
        ),
        (
            "one_shell_receipt_replayed",
            one_shell_checks["receipt_fingerprint_valid"] is True,
        ),
        (
            "one_shell_separator_and_global_escapes_replayed",
            one_shell_checks[
                "exact_shell_separator_replayed"
            ] is True
            and one_shell_checks[
                "explicit_global_escapes_replayed"
            ] is True,
        ),
        (
            "modular_reconnaissance_not_promoted",
            reconnaissance["modular_rank_diagnostics"]["role"]
            == "nonproof-reconnaissance"
            and one_shell_checks[
                "D_squared_global_nonmembership_proved"
            ] is False
            and one_shell_checks[
                "D_in_radical_J_mix_decided"
            ] is False,
        ),
        (
            "full_matrix_construction_refused",
            preflight["memory_preflight"][
                "full_raw_construction_authorized"
            ] is False
            and preflight["memory_preflight"][
                "full_orbit_construction_authorized"
            ] is False,
        ),
        (
            "support_limited_separator_excluded",
            claims[
                "support_limited_D_squared_separator_exists"
            ] is False
            and independent[
                "support_limited_separator_excluded"
            ] is True,
        ),
        (
            "radical_membership_not_claimed",
            claims["D_in_radical_J_mix_decided"] is False,
        ),
        (
            "global_GHZ_nonexistence_not_claimed",
            claims["global_GHZ_nonexistence_proved"] is False,
        ),
        (
            "exact_affine_membership_not_claimed",
            claims["exact_affine_GHZ_membership_decided"] is False,
        ),
    )
    loaded = LoadedRadicalArtifact(
        directory=directory,
        certificate=certificate,
        reconnaissance=reconnaissance,
        manifest=manifest,
        independent_checks=independent,
        checks=checks,
    )
    if not loaded.exact:
        raise KrennRadicalArtifactError(
            "radical-obstruction exact checks failed"
        )
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify the exact k=2 radical-obstruction bundle."
        )
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        generate_radical_obstruction_bundle(
            arguments.output_directory
        )
    loaded = verify_radical_obstruction_bundle(
        arguments.output_directory
    )
    print(json.dumps({
        "exact": loaded.exact,
        "D_squared_in_J_mix_decided": False,
        "D_in_radical_J_mix_decided": False,
        "global_GHZ_nonexistence_proved": False,
        "exact_affine_GHZ_membership_decided": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
