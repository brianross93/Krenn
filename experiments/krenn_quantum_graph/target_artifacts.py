"""Exact rational target-problem artifacts for the Krenn tensor map.

This schema is deliberately separate from the original GHZ-system artifact
schema.  The compact NPZ contains only the target-independent matching map;
the exact target and optional finite rational witness are JSON artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np

from experiments.krenn_quantum_graph import independent_verifier
from experiments.krenn_quantum_graph.system import variable_key
from experiments.krenn_quantum_graph.targets import (
    BASIS_CONVENTION,
    TARGET_SCHEMA,
    ColoringTarget,
    KrennTargetError,
)
from experiments.krenn_quantum_graph.tensor_map import (
    TENSOR_MAP_SCHEMA,
    KrennTensorMapError,
    MatchingTensorMap,
)
from experiments.krenn_quantum_graph.witness import (
    KrennWitnessError,
    SparseWitness,
)


ROOT = Path(__file__).resolve().parents[2]

TARGET_PROBLEM_MANIFEST_SCHEMA = (
    "krenn-rational-target-problem-manifest-v1"
)
TARGET_PROBLEM_CERTIFICATE_SCHEMA = (
    "krenn-rational-target-problem-certificate-v1"
)
TARGET_PROBLEM_WITNESS_SCHEMA = (
    "krenn-rational-target-problem-witness-v1"
)

TENSOR_MAP_FILE = "tensor_map.npz"
TARGET_FILE = "target.json"
WITNESS_FILE = "witness.json"
CERTIFICATE_FILE = "certificate.json"
MANIFEST_FILE = "manifest.json"

TENSOR_MAP_ARCHIVE_KEYS = {
    "equation_offsets",
    "monomial_variable_indices",
}

# Bounds cover every current milestone, including n=6,d=4.  They also stop an
# untrusted archive from driving the independent subset-filtering enumerator
# into an uncontrolled reconstruction.
MAX_N = 6
MAX_D = 4
MAX_VARIABLE_COUNT = 240
MAX_COEFFICIENT_COUNT = 4_096
MAX_MONOMIAL_COUNT = 61_440
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024 * 1024

_SOURCE_INPUTS = (
    ROOT / "experiments/krenn_quantum_graph/independent_verifier.py",
    ROOT / "experiments/krenn_quantum_graph/system.py",
    ROOT / "experiments/krenn_quantum_graph/targets.py",
    ROOT / "experiments/krenn_quantum_graph/tensor_map.py",
    ROOT / "experiments/krenn_quantum_graph/transport.py",
    ROOT / "experiments/krenn_quantum_graph/witness.py",
)


class KrennTargetArtifactError(RuntimeError):
    """A rational target-problem bundle failed exact replay."""


@dataclass(frozen=True)
class LoadedTargetArtifactBundle:
    """Verified, round-tripped target problem and optional witness."""

    directory: Path
    tensor_map: MatchingTensorMap
    target: ColoringTarget
    witness: SparseWitness | None
    certificate: Mapping
    independent_report: (
        independent_verifier.IndependentSystemReport
    )

    @property
    def exact(self) -> bool:
        return self.independent_report.exact and all(
            bool(value)
            for value in self.certificate["exact_checks"].values()
        )


def _exact_json_integer(value, label: str) -> int:
    if type(value) is not int:
        raise KrennTargetArtifactError(
            f"{label} must be an exact JSON integer"
        )
    return value


def _validate_scope(n: int, d: int) -> tuple[int, int]:
    n = _exact_json_integer(n, "factor count")
    d = _exact_json_integer(d, "local dimension")
    if n < 2 or n % 2 or n > MAX_N:
        raise KrennTargetArtifactError(
            f"factor count must be even and between 2 and {MAX_N}"
        )
    if d < 1 or d > MAX_D:
        raise KrennTargetArtifactError(
            f"local dimension must be between 1 and {MAX_D}"
        )
    variable_total = n * (n - 1) // 2 * d * d
    coefficient_total = d**n
    matching_total = 1
    for odd in range(n - 1, 0, -2):
        matching_total *= odd
    monomial_total = coefficient_total * matching_total
    if (
        variable_total > MAX_VARIABLE_COUNT
        or coefficient_total > MAX_COEFFICIENT_COUNT
        or monomial_total > MAX_MONOMIAL_COUNT
    ):
        raise KrennTargetArtifactError(
            "target problem exceeds the reviewed artifact bounds"
        )
    return n, d


def _index_dtype(variable_total: int):
    if variable_total <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    if variable_total <= np.iinfo(np.uint32).max + 1:
        return np.uint32
    return np.uint64


def tensor_map_arrays(
    tensor_map: MatchingTensorMap,
) -> Mapping[str, np.ndarray]:
    """Return the compact map arrays; a target RHS is intentionally absent."""

    _validate_scope(tensor_map.n, tensor_map.d)
    return {
        "equation_offsets": np.asarray(
            tensor_map.equation_offsets, dtype=np.uint64
        ),
        "monomial_variable_indices": np.asarray(
            tensor_map.monomial_variable_indices,
            dtype=_index_dtype(tensor_map.variable_count),
        ),
    }


def target_witness_payload(witness: SparseWitness) -> Mapping:
    """Serialize a sparse rational witness in canonical index order."""

    return {
        "schema": TARGET_PROBLEM_WITNESS_SCHEMA,
        "n": witness.n,
        "d": witness.d,
        "omitted_coordinates": "zero",
        "support_size": witness.support_size,
        "entries": [
            {
                "variable_index": index,
                "coordinate": list(
                    variable_key(witness.n, witness.d, index)
                ),
                "numerator": value.numerator,
                "denominator": value.denominator,
            }
            for index, value in witness.entries
        ],
    }


def _canonical_json_bytes(value: Mapping) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json_bytes(value))
    temporary.replace(path)


def _write_npz_atomic(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _local_file_record(path: Path) -> Mapping:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_record(path: Path) -> Mapping:
    try:
        return independent_verifier.canonical_source_record(path)
    except independent_verifier.IndependentVerificationError as error:
        raise KrennTargetArtifactError(
            f"could not record source input {path}"
        ) from error


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _synthetic_ghz_rhs(n: int, d: int) -> np.ndarray:
    """Supply only the RHS expected by the existing structural verifier."""

    rhs = np.zeros(d**n, dtype=np.uint8)
    for color in range(d):
        index = 0
        for _position in range(n):
            index = index * d + color
        rhs[index] = 1
    return rhs


def _verify_structure_arrays(
    n: int,
    d: int,
    arrays: Mapping[str, np.ndarray],
) -> independent_verifier.IndependentSystemReport:
    if set(arrays) != TENSOR_MAP_ARCHIVE_KEYS:
        raise KrennTargetArtifactError(
            "tensor-map archive key set changed or contains an RHS"
        )
    try:
        return independent_verifier.verify_system_arrays(
            n,
            d,
            arrays["equation_offsets"],
            arrays["monomial_variable_indices"],
            _synthetic_ghz_rhs(n, d),
        )
    except independent_verifier.IndependentVerificationError as error:
        raise KrennTargetArtifactError(
            "tensor map failed independent matching reconstruction"
        ) from error


def _independent_witness_replay(
    arrays: Mapping[str, np.ndarray],
    entries: Sequence[tuple[int, Fraction]],
    target_dense: Sequence[Fraction],
) -> bool:
    """Replay Phi(W)=T directly from serialized arrays and rational rows."""

    offsets = arrays["equation_offsets"]
    monomials = arrays["monomial_variable_indices"]
    values = dict(entries)
    if len(target_dense) != len(offsets) - 1:
        return False
    for equation, expected in enumerate(target_dense):
        total = Fraction(0)
        start = int(offsets[equation])
        stop = int(offsets[equation + 1])
        for row in range(start, stop):
            term = Fraction(1)
            for raw_variable in monomials[row]:
                term *= values.get(int(raw_variable), Fraction(0))
            total += term
        if total != Fraction(expected):
            return False
    return True


def _artifact_inventory(witness_present: bool) -> Mapping:
    return {
        "tensor_map": TENSOR_MAP_FILE,
        "target": TARGET_FILE,
        "witness": WITNESS_FILE if witness_present else None,
        "certificate": CERTIFICATE_FILE,
        "manifest": MANIFEST_FILE,
    }


def _build_certificate(
    tensor_map: MatchingTensorMap,
    target: ColoringTarget,
    witness: SparseWitness | None,
    arrays: Mapping[str, np.ndarray],
) -> Mapping:
    report = _verify_structure_arrays(
        tensor_map.n, tensor_map.d, arrays
    )
    target_round_trip = (
        ColoringTarget.from_payload(target.to_payload()) == target
    )
    primary_structure = (
        tuple(map(int, arrays["equation_offsets"]))
        == tensor_map.equation_offsets
        and tuple(
            tuple(map(int, row))
            for row in arrays["monomial_variable_indices"]
        )
        == tensor_map.monomial_variable_indices
    )
    exact_checks = {
        "target_independent_archive_has_no_rhs": (
            set(arrays) == TENSOR_MAP_ARCHIVE_KEYS
        ),
        "primary_tensor_map_structure_replay": primary_structure,
        "independent_matching_reconstruction": report.exact,
        "target_json_round_trip_exact": target_round_trip,
        "independent_target_semantic_replay": True,
    }
    witness_present = witness is not None
    if witness is not None:
        primary_comparison = tensor_map.compare_exact(witness, target)
        target_dense = tuple(
            map(Fraction, target.dense_coefficients())
        )
        exact_checks.update(
            {
                "primary_exact_rational_witness_replay": (
                    primary_comparison.satisfied
                ),
                "independent_npz_witness_replay": (
                    _independent_witness_replay(
                        arrays, witness.entries, target_dense
                    )
                ),
            }
        )
    status = (
        "exact-rational-affine-image-witness-certified"
        if witness_present
        else "target-problem-structure-only-no-witness"
    )
    claims = {
        "witness_emitted": witness_present,
        "exact_rational_affine_image_membership_certified": (
            witness_present
        ),
        "complex_affine_image_membership_certified": witness_present,
        "projective_image_membership_certified": False,
        "local_diagonal_orbit_membership_certified": False,
        "border_image_membership_certified": False,
        "nonexistence_certificate_emitted": False,
        "statement": (
            "The emitted finite exact-rational witness satisfies "
            "Phi(W)=T coefficientwise, and therefore gives an affine image "
            "point over Q and C. No border-only or nonexistence claim is "
            "made."
            if witness_present
            else
            "This bundle records an exact rational target and the canonical "
            "matching map only. Without a witness it certifies no affine, "
            "projective, local-diagonal, border-image, or nonexistence "
            "claim."
        ),
    }
    return {
        "schema": TARGET_PROBLEM_CERTIFICATE_SCHEMA,
        "tensor_map_schema": TENSOR_MAP_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "status": status,
        "parameters": {"n": tensor_map.n, "d": tensor_map.d},
        "ordering": {
            "edges": "lexicographic canonical endpoints i<j",
            "endpoint_colors": "a major, b minor",
            "variables": "edge major, then a, then b",
            "colorings": BASIS_CONVENTION,
            "matchings": (
                "least unused vertex paired with partners ascending"
            ),
            "monomials": (
                "flattened by coloring then deterministic matching"
            ),
        },
        "counts": {
            "variables": tensor_map.variable_count,
            "coefficients": tensor_map.coefficient_count,
            "degree": tensor_map.degree,
            "matchings_per_coefficient": tensor_map.matching_count,
            "monomials": tensor_map.monomial_count,
            "target_support": target.support_size,
            "witness_support": (
                witness.support_size if witness is not None else 0
            ),
        },
        "compact_encoding": {
            "archive_keys": sorted(arrays),
            "equation_offsets_shape": list(
                arrays["equation_offsets"].shape
            ),
            "variable_index_terms_shape": list(
                arrays["monomial_variable_indices"].shape
            ),
            "variable_index_dtype": str(
                arrays["monomial_variable_indices"].dtype
            ),
            "structure_sha256": _array_digest(arrays),
        },
        "artifacts": _artifact_inventory(witness_present),
        "exact_checks": exact_checks,
        "claims": claims,
    }


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennTargetArtifactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Mapping:
    try:
        if path.is_symlink():
            raise KrennTargetArtifactError(
                f"{label} must not be a symbolic link"
            )
        if path.stat().st_size > MAX_JSON_BYTES:
            raise KrennTargetArtifactError(
                f"{label} exceeds the reviewed size bound"
            )
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except KrennTargetArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennTargetArtifactError(
            f"could not decode {label}"
        ) from error
    if not isinstance(payload, dict):
        raise KrennTargetArtifactError(
            f"{label} must contain a JSON object"
        )
    if raw != _canonical_json_bytes(payload):
        raise KrennTargetArtifactError(
            f"{label} is not canonical round-trippable JSON"
        )
    return payload


def _load_target(
    path: Path, n: int, d: int
) -> tuple[ColoringTarget, tuple[Fraction, ...]]:
    """Parse target semantics independently, then cross-check primary code."""

    payload = _load_json(path, "target artifact")
    if set(payload) != {
        "schema",
        "basis_convention",
        "n",
        "d",
        "entries",
    }:
        raise KrennTargetArtifactError(
            "target artifact schema changed"
        )
    if (
        payload.get("schema") != TARGET_SCHEMA
        or payload.get("basis_convention") != BASIS_CONVENTION
        or _exact_json_integer(payload.get("n"), "target n") != n
        or _exact_json_integer(payload.get("d"), "target d") != d
        or not isinstance(payload.get("entries"), list)
    ):
        raise KrennTargetArtifactError(
            "target artifact identity changed"
        )
    dense = [Fraction(0) for _ in range(d**n)]
    previous: tuple[int, ...] | None = None
    for row in payload["entries"]:
        if not isinstance(row, dict) or set(row) != {
            "coloring",
            "numerator",
            "denominator",
        }:
            raise KrennTargetArtifactError(
                "target coefficient row schema changed"
            )
        raw_coloring = row["coloring"]
        if (
            not isinstance(raw_coloring, list)
            or len(raw_coloring) != n
        ):
            raise KrennTargetArtifactError(
                "target coloring shape changed"
            )
        coloring = tuple(
            _exact_json_integer(color, "target color")
            for color in raw_coloring
        )
        if any(color < 0 or color >= d for color in coloring):
            raise KrennTargetArtifactError(
                "target color is outside range(d)"
            )
        if previous is not None and coloring <= previous:
            raise KrennTargetArtifactError(
                "target rows are not unique lexicographic entries"
            )
        previous = coloring
        numerator = _exact_json_integer(
            row["numerator"], "target numerator"
        )
        denominator = _exact_json_integer(
            row["denominator"], "target denominator"
        )
        if (
            numerator == 0
            or denominator <= 0
            or gcd(abs(numerator), denominator) != 1
        ):
            raise KrennTargetArtifactError(
                "target coefficient is not a reduced nonzero rational"
            )
        index = 0
        for color in coloring:
            index = index * d + color
        dense[index] = Fraction(numerator, denominator)
    try:
        target = ColoringTarget.from_payload(payload)
    except KrennTargetError as error:
        raise KrennTargetArtifactError(
            "target failed primary exact parsing"
        ) from error
    if (
        tuple(map(Fraction, target.dense_coefficients()))
        != tuple(dense)
        or target.to_payload() != payload
    ):
        raise KrennTargetArtifactError(
            "target failed independent semantic round trip"
        )
    return target, tuple(dense)


def _load_witness(
    path: Path, n: int, d: int
) -> tuple[SparseWitness, tuple[tuple[int, Fraction], ...]]:
    payload = _load_json(path, "witness artifact")
    if set(payload) != {
        "schema",
        "n",
        "d",
        "omitted_coordinates",
        "support_size",
        "entries",
    }:
        raise KrennTargetArtifactError(
            "target witness schema changed"
        )
    if (
        payload.get("schema") != TARGET_PROBLEM_WITNESS_SCHEMA
        or _exact_json_integer(payload.get("n"), "witness n") != n
        or _exact_json_integer(payload.get("d"), "witness d") != d
        or payload.get("omitted_coordinates") != "zero"
        or not isinstance(payload.get("entries"), list)
    ):
        raise KrennTargetArtifactError(
            "target witness identity changed"
        )
    variable_total = n * (n - 1) // 2 * d * d
    entries: list[tuple[int, Fraction]] = []
    previous = -1
    for row in payload["entries"]:
        if not isinstance(row, dict) or set(row) != {
            "variable_index",
            "coordinate",
            "numerator",
            "denominator",
        }:
            raise KrennTargetArtifactError(
                "target witness row schema changed"
            )
        index = _exact_json_integer(
            row["variable_index"], "witness variable index"
        )
        if index <= previous or index >= variable_total:
            raise KrennTargetArtifactError(
                "witness indices are not unique increasing in range"
            )
        previous = index
        raw_coordinate = row["coordinate"]
        if (
            not isinstance(raw_coordinate, list)
            or len(raw_coordinate) != 4
        ):
            raise KrennTargetArtifactError(
                "witness coordinate shape changed"
            )
        coordinate = tuple(
            _exact_json_integer(value, "witness coordinate")
            for value in raw_coordinate
        )
        if coordinate != variable_key(n, d, index):
            raise KrennTargetArtifactError(
                "witness coordinate does not match its canonical index"
            )
        numerator = _exact_json_integer(
            row["numerator"], "witness numerator"
        )
        denominator = _exact_json_integer(
            row["denominator"], "witness denominator"
        )
        if (
            numerator == 0
            or denominator <= 0
            or gcd(abs(numerator), denominator) != 1
        ):
            raise KrennTargetArtifactError(
                "witness coefficient is not a reduced nonzero rational"
            )
        entries.append((index, Fraction(numerator, denominator)))
    support_size = _exact_json_integer(
        payload.get("support_size"), "witness support size"
    )
    if support_size != len(entries):
        raise KrennTargetArtifactError(
            "witness support census changed"
        )
    try:
        witness = SparseWitness.from_index_values(n, d, entries)
    except KrennWitnessError as error:
        raise KrennTargetArtifactError(
            "witness failed primary exact parsing"
        ) from error
    if witness.entries != tuple(entries):
        raise KrennTargetArtifactError(
            "witness failed independent semantic round trip"
        )
    if target_witness_payload(witness) != payload:
        raise KrennTargetArtifactError(
            "witness JSON is not in canonical round-trip form"
        )
    return witness, tuple(entries)


def _load_tensor_map_archive(
    path: Path, n: int, d: int
) -> Mapping[str, np.ndarray]:
    try:
        if path.is_symlink():
            raise KrennTargetArtifactError(
                "tensor-map archive must not be a symbolic link"
            )
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise KrennTargetArtifactError(
                "tensor-map archive exceeds the reviewed size bound"
            )
        with ZipFile(path) as archive:
            member_names = [
                item.filename for item in archive.infolist()
            ]
            expected_members = {
                f"{key}.npy" for key in TENSOR_MAP_ARCHIVE_KEYS
            }
            if (
                len(member_names) != len(set(member_names))
                or set(member_names) != expected_members
            ):
                raise KrennTargetArtifactError(
                    "tensor-map ZIP members are not unique and exact"
                )
            if (
                sum(item.file_size for item in archive.infolist())
                > MAX_ARCHIVE_UNCOMPRESSED_BYTES
            ):
                raise KrennTargetArtifactError(
                    "tensor-map archive expands beyond the reviewed bound"
                )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != TENSOR_MAP_ARCHIVE_KEYS:
                raise KrennTargetArtifactError(
                    "tensor-map archive key set changed or contains an RHS"
                )
            offsets = np.asarray(archive["equation_offsets"])
            monomials = np.asarray(
                archive["monomial_variable_indices"]
            )
    except KrennTargetArtifactError:
        raise
    except (OSError, ValueError, BadZipFile) as error:
        raise KrennTargetArtifactError(
            "could not load tensor-map archive"
        ) from error
    expected_dtype = np.dtype(
        _index_dtype(n * (n - 1) // 2 * d * d)
    )
    if (
        offsets.dtype != np.dtype(np.uint64)
        or monomials.dtype != expected_dtype
    ):
        raise KrennTargetArtifactError(
            "tensor-map archive dtype contract changed"
        )
    return {
        "equation_offsets": offsets,
        "monomial_variable_indices": monomials,
    }


def _validate_certificate_identity(
    certificate: Mapping,
) -> tuple[int, int, bool]:
    if certificate.get("schema") != TARGET_PROBLEM_CERTIFICATE_SCHEMA:
        raise KrennTargetArtifactError(
            "target-problem certificate schema changed"
        )
    parameters = certificate.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != {"n", "d"}:
        raise KrennTargetArtifactError(
            "certificate parameter schema changed"
        )
    n, d = _validate_scope(
        parameters.get("n"), parameters.get("d")
    )
    artifacts = certificate.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "tensor_map",
        "target",
        "witness",
        "certificate",
        "manifest",
    }:
        raise KrennTargetArtifactError(
            "certificate artifact inventory changed"
        )
    witness_value = artifacts.get("witness")
    witness_present = witness_value is not None
    if artifacts != _artifact_inventory(witness_present):
        raise KrennTargetArtifactError(
            "certificate artifact names are not canonical"
        )
    return n, d, witness_present


def _verify_manifest(
    output_directory: Path,
    witness_present: bool,
) -> None:
    manifest = _load_json(
        output_directory / MANIFEST_FILE, "target-problem manifest"
    )
    if set(manifest) != {
        "schema",
        "producer",
        "artifacts",
        "inputs",
    } or manifest.get("schema") != TARGET_PROBLEM_MANIFEST_SCHEMA:
        raise KrennTargetArtifactError(
            "target-problem manifest schema changed"
        )
    if manifest.get("producer") != _source_record(Path(__file__)):
        raise KrennTargetArtifactError(
            "target-problem producer source record changed"
        )
    expected_inputs = [
        _source_record(path) for path in _SOURCE_INPUTS
    ]
    if manifest.get("inputs") != expected_inputs:
        raise KrennTargetArtifactError(
            "target-problem source input records changed"
        )
    inventory = _artifact_inventory(witness_present)
    artifact_names = sorted(
        name
        for key, name in inventory.items()
        if key != "manifest" and name is not None
    )
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != len(
        artifact_names
    ):
        raise KrennTargetArtifactError(
            "manifest artifact inventory changed"
        )
    for name, record in zip(artifact_names, records, strict=True):
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "bytes", "sha256"}
            or record.get("path") != name
            or record != _local_file_record(output_directory / name)
        ):
            raise KrennTargetArtifactError(
                f"SHA256 manifest replay failed for {name}"
            )
    expected_files = {
        name for name in inventory.values() if name is not None
    }
    actual_entries = {
        path.name for path in output_directory.iterdir()
    }
    if actual_entries != expected_files:
        raise KrennTargetArtifactError(
            "target-problem directory inventory changed"
        )
    if any(
        (output_directory / name).is_symlink()
        or not (output_directory / name).is_file()
        for name in expected_files
    ):
        raise KrennTargetArtifactError(
            "target-problem artifacts must be regular files"
        )


def write_target_artifact_bundle(
    tensor_map: MatchingTensorMap,
    target: ColoringTarget,
    output_directory: Path | str,
    *,
    witness: SparseWitness | None = None,
) -> Path:
    """Write and immediately replay one exact rational target problem."""

    n, d = _validate_scope(tensor_map.n, tensor_map.d)
    if (target.n, target.d) != (n, d):
        raise KrennTargetArtifactError(
            "target dimensions do not match the tensor map"
        )
    if witness is not None and (witness.n, witness.d) != (n, d):
        raise KrennTargetArtifactError(
            "witness dimensions do not match the tensor map"
        )
    arrays = tensor_map_arrays(tensor_map)
    _verify_structure_arrays(n, d, arrays)
    if (
        ColoringTarget.from_payload(target.to_payload()) != target
    ):
        raise KrennTargetArtifactError(
            "target did not round-trip exactly"
        )
    if witness is not None:
        try:
            comparison = tensor_map.compare_exact(witness, target)
        except KrennTensorMapError as error:
            raise KrennTargetArtifactError(
                "could not compare witness with target"
            ) from error
        if not comparison.satisfied or not _independent_witness_replay(
            arrays,
            witness.entries,
            tuple(map(Fraction, target.dense_coefficients())),
        ):
            raise KrennTargetArtifactError(
                "optional witness does not map exactly to the target"
            )

    raw_output_directory = Path(output_directory)
    if raw_output_directory.is_symlink():
        raise KrennTargetArtifactError(
            "target-problem output directory must not be a symbolic link"
        )
    output_directory = raw_output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    inventory = _artifact_inventory(witness is not None)
    expected_files = {
        name for name in inventory.values() if name is not None
    }
    unexpected = {
        path.name
        for path in output_directory.iterdir()
        if path.name not in expected_files
    }
    if unexpected:
        raise KrennTargetArtifactError(
            "refusing to overwrite a directory with unexpected files: "
            f"{sorted(unexpected)}"
        )

    _write_npz_atomic(output_directory / TENSOR_MAP_FILE, arrays)
    _write_json_atomic(output_directory / TARGET_FILE, target.to_payload())
    if witness is not None:
        _write_json_atomic(
            output_directory / WITNESS_FILE,
            target_witness_payload(witness),
        )
    certificate = _build_certificate(
        tensor_map, target, witness, arrays
    )
    _write_json_atomic(
        output_directory / CERTIFICATE_FILE, certificate
    )
    artifact_names = sorted(
        name
        for key, name in inventory.items()
        if key != "manifest" and name is not None
    )
    manifest = {
        "schema": TARGET_PROBLEM_MANIFEST_SCHEMA,
        "producer": _source_record(Path(__file__)),
        "artifacts": [
            _local_file_record(output_directory / name)
            for name in artifact_names
        ],
        "inputs": [
            _source_record(path) for path in _SOURCE_INPUTS
        ],
    }
    _write_json_atomic(output_directory / MANIFEST_FILE, manifest)
    verify_target_artifact_bundle(output_directory)
    return output_directory / CERTIFICATE_FILE


def verify_target_artifact_bundle(
    output_directory: Path | str,
) -> LoadedTargetArtifactBundle:
    """Replay provenance, matching structure, target, and optional witness."""

    raw_output_directory = Path(output_directory)
    if raw_output_directory.is_symlink():
        raise KrennTargetArtifactError(
            "target-problem bundle directory must not be a symbolic link"
        )
    output_directory = raw_output_directory.resolve()
    certificate = _load_json(
        output_directory / CERTIFICATE_FILE,
        "target-problem certificate",
    )
    n, d, witness_present = _validate_certificate_identity(certificate)
    _verify_manifest(output_directory, witness_present)

    arrays = _load_tensor_map_archive(
        output_directory / TENSOR_MAP_FILE, n, d
    )
    independent_report = _verify_structure_arrays(n, d, arrays)
    tensor_map = MatchingTensorMap(n, d)
    expected_arrays = tensor_map_arrays(tensor_map)
    if (
        not np.array_equal(
            arrays["equation_offsets"],
            expected_arrays["equation_offsets"],
        )
        or not np.array_equal(
            arrays["monomial_variable_indices"],
            expected_arrays["monomial_variable_indices"],
        )
    ):
        raise KrennTargetArtifactError(
            "tensor map failed primary structural reconstruction"
        )

    target, target_dense = _load_target(
        output_directory / TARGET_FILE, n, d
    )
    witness = None
    witness_entries: tuple[tuple[int, Fraction], ...] = ()
    if witness_present:
        witness, witness_entries = _load_witness(
            output_directory / WITNESS_FILE, n, d
        )
        if not tensor_map.compare_exact(witness, target).satisfied:
            raise KrennTargetArtifactError(
                "witness failed primary exact target replay"
            )
        if not _independent_witness_replay(
            arrays, witness_entries, target_dense
        ):
            raise KrennTargetArtifactError(
                "witness failed independent exact target replay"
            )

    expected_certificate = _build_certificate(
        tensor_map, target, witness, arrays
    )
    if certificate != expected_certificate:
        raise KrennTargetArtifactError(
            "target-problem certificate failed semantic replay"
        )
    bundle = LoadedTargetArtifactBundle(
        directory=output_directory,
        tensor_map=tensor_map,
        target=target,
        witness=witness,
        certificate=certificate,
        independent_report=independent_report,
    )
    if not bundle.exact:
        raise KrennTargetArtifactError(
            "target-problem bundle did not verify exactly"
        )
    return bundle
