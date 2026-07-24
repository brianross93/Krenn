"""Round-trippable compact Krenn system and witness artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.krenn_quantum_graph import independent_verifier
from experiments.krenn_quantum_graph.deformation import (
    DeformationCertificate,
    certify_n4_d3_deformation,
)
from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    generate_sparse_system,
    variable_key,
)
from experiments.krenn_quantum_graph.witness import (
    SparseWitness,
    WITNESS_SCHEMA,
    require_exact_witness,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/krenn_quantum_graph"

MANIFEST_SCHEMA = "krenn-quantum-graph-artifact-manifest-v1"
CERTIFICATE_SCHEMA = "krenn-quantum-graph-artifact-certificate-v1"
SYSTEM_FILE = "system.npz"
WITNESS_FILE = "witness.json"
DEFORMATION_FILE = "deformation_certificate.json"
BOUNDARY_FILE = "claim_boundary.json"
CERTIFICATE_FILE = "certificate.json"
MANIFEST_FILE = "manifest.json"


class KrennArtifactError(RuntimeError):
    """Artifact emission, integrity, or semantic replay failed."""


@dataclass(frozen=True)
class LoadedArtifactBundle:
    directory: Path
    system: SparsePolynomialSystem
    witness: SparseWitness | None
    deformation: Mapping | None
    boundary: Mapping | None
    certificate: Mapping
    independent_report: (
        independent_verifier.IndependentArtifactReport
    )

    @property
    def exact(self) -> bool:
        return self.independent_report.exact and all(
            bool(value)
            for value in self.certificate["exact_checks"].values()
        )


def _write_json_atomic(path: Path, value: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def _root_file_record(path: Path) -> Mapping:
    try:
        return independent_verifier.canonical_source_record(path)
    except independent_verifier.IndependentVerificationError as error:
        raise KrennArtifactError(
            f"could not record source input {path}"
        ) from error


def _require_v1_ghz_target(system: SparsePolynomialSystem) -> None:
    if not system.has_canonical_ghz_target:
        raise KrennArtifactError(
            "artifact schema v1 serializes only the canonical GHZ target; "
            "general rational targets require a distinct artifact schema"
        )


def _index_dtype(variable_total: int):
    if variable_total <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    if variable_total <= np.iinfo(np.uint32).max + 1:
        return np.uint32
    return np.uint64


def system_arrays(
    system: SparsePolynomialSystem,
) -> Mapping[str, np.ndarray]:
    _require_v1_ghz_target(system)
    return {
        "equation_offsets": np.asarray(
            system.equation_offsets, dtype=np.uint64
        ),
        "monomial_variable_indices": np.asarray(
            system.monomial_variable_indices,
            dtype=_index_dtype(system.variable_count),
        ),
        "rhs_values": np.asarray(
            system.rhs_values, dtype=np.uint8
        ),
    }


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


def witness_payload(witness: SparseWitness) -> Mapping:
    return {
        "schema": WITNESS_SCHEMA,
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


def _load_witness(path: Path, n: int, d: int) -> SparseWitness:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KrennArtifactError("could not decode witness artifact") from error
    if (
        payload.get("schema") != WITNESS_SCHEMA
        or int(payload.get("n", -1)) != n
        or int(payload.get("d", -1)) != d
        or payload.get("omitted_coordinates") != "zero"
    ):
        raise KrennArtifactError("witness artifact identity changed")
    entries = []
    previous = -1
    for row in payload.get("entries", ()):
        try:
            index = int(row["variable_index"])
            coordinate = tuple(map(int, row["coordinate"]))
            value = Fraction(
                int(row["numerator"]), int(row["denominator"])
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise KrennArtifactError("malformed witness row") from error
        if (
            index <= previous
            or coordinate != variable_key(n, d, index)
            or value == 0
            or value.denominator != int(row["denominator"])
        ):
            raise KrennArtifactError(
                "witness row failed canonical semantic replay"
            )
        previous = index
        entries.append((index, value))
    witness = SparseWitness.from_index_values(n, d, entries)
    if witness.support_size != int(payload.get("support_size", -1)):
        raise KrennArtifactError("witness support census changed")
    if payload != witness_payload(witness):
        raise KrennArtifactError(
            "witness JSON is not in canonical round-trip form"
        )
    return witness


def _artifact_inventory(
    witness: SparseWitness | None,
    deformation: Mapping | None,
    boundary: Mapping | None,
) -> Mapping[str, str | None]:
    return {
        "system": SYSTEM_FILE,
        "witness": WITNESS_FILE if witness is not None else None,
        "deformation": (
            DEFORMATION_FILE if deformation is not None else None
        ),
        "claim_boundary": (
            BOUNDARY_FILE if boundary is not None else None
        ),
        "certificate": CERTIFICATE_FILE,
        "manifest": MANIFEST_FILE,
    }


def _build_certificate(
    system: SparsePolynomialSystem,
    witness: SparseWitness | None,
    deformation: Mapping | None,
    boundary: Mapping | None,
) -> Mapping:
    arrays = system_arrays(system)
    independent_system = independent_verifier.verify_system_arrays(
        system.n,
        system.d,
        arrays["equation_offsets"],
        arrays["monomial_variable_indices"],
        arrays["rhs_values"],
    )
    exact_checks = {
        "primary_sparse_system_valid": True,
        "independent_sparse_system_replay": independent_system.exact,
    }
    if witness is not None:
        rational, finite = require_exact_witness(system, witness)
        independent_witness = (
            independent_verifier.verify_witness_entries(
                witness.n, witness.d, witness.entries
            )
        )
        exact_checks.update(
            {
                "witness_exact_over_Q": rational.satisfied,
                "witness_exact_over_F31": finite.satisfied,
                "independent_witness_replay": (
                    independent_witness.exact
                ),
            }
        )
    if deformation is not None:
        exact_checks["deformation_certificate_exact"] = all(
            bool(value)
            for value in deformation.get("exact_checks", {}).values()
        )
    if boundary is not None:
        exact_checks["claim_boundary_explicit"] = (
            dict(boundary)
            == independent_verifier.canonical_claim_boundary(
                system.n, system.d
            )
        )
    status = independent_verifier.expected_bundle_status(
        system.n,
        system.d,
        witness_present=witness is not None,
        deformation_present=deformation is not None,
        boundary=boundary,
    )
    claim_boundary = {
        "symmetry": (
            "Variables remain independent; S_n x S_d is used only for "
            "transport and verification, never to equate weights."
        ),
        "finite_field": (
            "An F_31 replay alone is not a proof over C."
        ),
        "witness": (
            "The exact-Q replay certifies the listed rational witness, hence "
            "the same point over C."
            if witness is not None
            else "No solution witness is asserted by this structural bundle."
        ),
        "search": (
            dict(boundary)
            if boundary is not None
            else {
                "search_performed": False,
                "no_solution_certificate_emitted": False,
            }
        ),
    }
    return {
        "schema": CERTIFICATE_SCHEMA,
        "system_schema": system.schema,
        "status": str(status),
        "parameters": {"n": system.n, "d": system.d},
        "ordering": {
            "edges": "lexicographic canonical endpoints i<j",
            "endpoint_colors": "a major, b minor",
            "variables": "edge major, then a, then b",
            "colorings": "lexicographic; last vertex changes fastest",
            "matchings": (
                "least unused vertex paired with partners ascending"
            ),
            "monomials": (
                "flattened by equation then deterministic matching"
            ),
        },
        "counts": {
            "variables": system.variable_count,
            "equations": system.equation_count,
            "degree": system.degree,
            "matchings_per_equation": system.matching_count,
            "monomials": system.monomial_count,
            "constant_rhs_equations": sum(system.rhs_values),
            "witness_support": (
                witness.support_size if witness is not None else 0
            ),
        },
        "compact_encoding": {
            "equation_offsets_shape": list(
                arrays["equation_offsets"].shape
            ),
            "variable_index_terms_shape": list(
                arrays["monomial_variable_indices"].shape
            ),
            "rhs_shape": list(arrays["rhs_values"].shape),
            "variable_index_dtype": str(
                arrays["monomial_variable_indices"].dtype
            ),
            "structure_sha256": _array_digest(arrays),
        },
        "artifacts": _artifact_inventory(
            witness, deformation, boundary
        ),
        "exact_checks": exact_checks,
        "claim_boundary": claim_boundary,
    }


def write_artifact_bundle(
    system: SparsePolynomialSystem,
    output_directory: Path | str,
    *,
    witness: SparseWitness | None = None,
    deformation: DeformationCertificate | Mapping | None = None,
    boundary: Mapping | None = None,
    status: str | None = None,
    input_paths: Sequence[Path | str] = (),
) -> Path:
    """Write a compact bundle, then replay it with both verifiers."""

    _require_v1_ghz_target(system)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if witness is not None and (
        system.n,
        system.d,
    ) != (witness.n, witness.d):
        raise KrennArtifactError(
            "artifact witness and system parameters differ"
        )
    if (
        system.n,
        system.d,
    ) not in independent_verifier.SUPPORTED_ARTIFACT_PARAMETERS:
        raise KrennArtifactError(
            "artifact schema v1 is bounded to the four milestone systems"
        )
    deformation_mapping = (
        deformation.to_dict()
        if isinstance(deformation, DeformationCertificate)
        else dict(deformation)
        if deformation is not None
        else None
    )
    boundary_mapping = dict(boundary) if boundary is not None else None
    derived_status = independent_verifier.expected_bundle_status(
        system.n,
        system.d,
        witness_present=witness is not None,
        deformation_present=deformation_mapping is not None,
        boundary=boundary_mapping,
    )
    if status is not None and status != derived_status:
        raise KrennArtifactError(
            "requested status does not follow from bundle contents"
        )
    inventory = _artifact_inventory(
        witness, deformation_mapping, boundary_mapping
    )
    expected_files = {
        name for name in inventory.values() if name is not None
    }
    unexpected = {
        path.name
        for path in output_directory.iterdir()
        if path.name not in expected_files
    }
    if unexpected:
        raise KrennArtifactError(
            f"refusing to overwrite bundle with unexpected files: "
            f"{sorted(unexpected)}"
        )

    arrays = system_arrays(system)
    _write_npz_atomic(output_directory / SYSTEM_FILE, arrays)
    if witness is not None:
        _write_json_atomic(
            output_directory / WITNESS_FILE, witness_payload(witness)
        )
    if deformation_mapping is not None:
        _write_json_atomic(
            output_directory / DEFORMATION_FILE,
            deformation_mapping,
        )
    if boundary_mapping is not None:
        _write_json_atomic(
            output_directory / BOUNDARY_FILE, boundary_mapping
        )
    certificate = _build_certificate(
        system,
        witness,
        deformation_mapping,
        boundary_mapping,
    )
    _write_json_atomic(
        output_directory / CERTIFICATE_FILE, certificate
    )

    default_inputs = [
        ROOT / "experiments/krenn_quantum_graph/system.py",
        ROOT / "experiments/krenn_quantum_graph/targets.py",
        ROOT / "experiments/krenn_quantum_graph/witness.py",
        ROOT
        / "experiments/krenn_quantum_graph/independent_verifier.py",
        ROOT
        / "experiments/krenn_quantum_graph/generate_results.py",
    ]
    if witness is not None:
        default_inputs.append(
            ROOT / "experiments/krenn_quantum_graph/fixtures.py"
        )
    if deformation_mapping is not None:
        default_inputs.append(
            ROOT / "experiments/krenn_quantum_graph/deformation.py"
        )
    resolved_inputs = tuple(
        sorted(
            dict.fromkeys(
                [
                    *(Path(path).resolve() for path in default_inputs),
                    *(Path(path).resolve() for path in input_paths),
                ]
            ),
            key=lambda path: str(path.relative_to(ROOT)).replace(
                "\\", "/"
            ),
        )
    )
    for path in resolved_inputs:
        if not path.is_file():
            raise KrennArtifactError(f"manifest input is absent: {path}")
    artifact_names = sorted(
        name
        for key, name in inventory.items()
        if key != "manifest" and name is not None
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "producer": _root_file_record(Path(__file__)),
        "artifacts": [
            _local_file_record(output_directory / name)
            for name in artifact_names
        ],
        "inputs": [
            _root_file_record(path) for path in resolved_inputs
        ],
    }
    _write_json_atomic(output_directory / MANIFEST_FILE, manifest)
    verify_artifact_bundle(output_directory)
    return output_directory / CERTIFICATE_FILE


def _load_system_archive(
    path: Path, n: int, d: int
) -> SparsePolynomialSystem:
    expected_index_dtype = np.dtype(
        _index_dtype(n * (n - 1) // 2 * d * d)
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != (
                independent_verifier.SYSTEM_ARCHIVE_KEYS
            ):
                raise KrennArtifactError(
                    "system archive key set changed"
                )
            offsets = np.asarray(archive["equation_offsets"])
            monomials = np.asarray(
                archive["monomial_variable_indices"]
            )
            rhs = np.asarray(archive["rhs_values"])
    except (OSError, ValueError) as error:
        raise KrennArtifactError("could not load system archive") from error
    if (
        offsets.dtype != np.dtype(np.uint64)
        or monomials.dtype != expected_index_dtype
        or rhs.dtype != np.dtype(np.uint8)
    ):
        raise KrennArtifactError("system archive dtype contract changed")
    return SparsePolynomialSystem(
        n=n,
        d=d,
        equation_offsets=tuple(map(int, offsets)),
        monomial_variable_indices=tuple(
            tuple(map(int, row)) for row in monomials
        ),
        rhs_values=tuple(map(int, rhs)),
    )


def verify_artifact_bundle(
    output_directory: Path | str,
) -> LoadedArtifactBundle:
    """Replay hashes and compare independent and primary reconstructions."""

    output_directory = Path(output_directory).resolve()
    try:
        certificate = json.loads(
            (output_directory / CERTIFICATE_FILE).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise KrennArtifactError(
            "could not decode Krenn artifact certificate"
        ) from error
    if certificate.get("schema") != CERTIFICATE_SCHEMA:
        raise KrennArtifactError("artifact certificate schema changed")
    independent_report = (
        independent_verifier.verify_artifact_bundle(output_directory)
    )
    parameters = certificate.get("parameters", {})
    n = int(parameters.get("n", -1))
    d = int(parameters.get("d", -1))
    system = _load_system_archive(
        output_directory / SYSTEM_FILE, n, d
    )
    primary_reconstruction = generate_sparse_system(n, d)
    if system != primary_reconstruction:
        raise KrennArtifactError(
            "system archive failed primary semantic reconstruction"
        )

    artifacts = certificate.get("artifacts", {})
    witness = None
    if artifacts.get("witness") is not None:
        witness = _load_witness(
            output_directory / str(artifacts["witness"]), n, d
        )
        require_exact_witness(system, witness)
    boundary = None
    if artifacts.get("claim_boundary") is not None:
        try:
            boundary = json.loads(
                (
                    output_directory
                    / str(artifacts["claim_boundary"])
                ).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise KrennArtifactError(
                "could not decode claim boundary artifact"
            ) from error
    deformation = None
    if artifacts.get("deformation") is not None:
        if witness is None:
            raise KrennArtifactError(
                "deformation artifact requires a witness"
            )
        try:
            deformation = json.loads(
                (
                    output_directory / str(artifacts["deformation"])
                ).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise KrennArtifactError(
                "could not decode deformation artifact"
            ) from error
        expected_deformation = certify_n4_d3_deformation(
            system, witness
        ).to_dict()
        if deformation != expected_deformation:
            raise KrennArtifactError(
                "deformation certificate failed semantic replay"
            )

    expected_certificate = _build_certificate(
        system,
        witness,
        deformation,
        boundary,
    )
    if certificate != expected_certificate:
        raise KrennArtifactError(
            "artifact certificate failed exact semantic replay"
        )
    bundle = LoadedArtifactBundle(
        directory=output_directory,
        system=system,
        witness=witness,
        deformation=deformation,
        boundary=boundary,
        certificate=certificate,
        independent_report=independent_report,
    )
    if not bundle.exact:
        raise KrennArtifactError("artifact bundle did not certify exactly")
    return bundle
