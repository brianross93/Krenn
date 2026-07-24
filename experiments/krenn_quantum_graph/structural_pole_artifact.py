"""Replayable exact bundle for the ``n=6,d=3`` structural pole analysis.

The bundle keeps the three exact structural payloads separate from a small
claim-boundary certificate.  Verification is fail-closed: it checks the
exact file inventory, byte hashes, canonical source ledgers, strict JSON,
and a fresh routine semantic reconstruction of every payload.

The completed support-21 closure had 12,992,269 exact search nodes.  Its
small deterministic receipts and terminal certificates are committed; a
full regeneration remains available through the gated long test and the
per-root command in :mod:`vertical_component`.  No numerical residual or
finite-field computation is promoted to an exact claim here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.matching_circuit_structure import (
    MATCHING_CIRCUIT_STRUCTURE_SCHEMA,
    exact_matching_circuit_structure,
)
from experiments.krenn_quantum_graph.structural_pole_law import (
    STRUCTURAL_POLE_SCHEMA,
    certify_structural_pole_law,
)
from experiments.krenn_quantum_graph.vertical_component import (
    EXPECTED_SUPPORT21_CLOSURE_NODES,
    VERTICAL_COMPONENT_SCHEMA,
    exact_vertical_component_structure,
    replay_support21_global_closure,
)


ROOT = Path(__file__).resolve().parents[2]

CERTIFICATE_SCHEMA = "krenn.n6_d3.structural_pole.certificate.v1"
MANIFEST_SCHEMA = "krenn.n6_d3.structural_pole.manifest.v1"

BASELINE_COMMIT = "ab446724a61a293a3268c63b9e9e721c6a360561"
COUNTEREXAMPLE_SEARCH_COMMIT = (
    "720f495810e1b9a403e7ea88acbe902f60f938c8"
)
OBSTRUCTION_CERTIFICATE_COMMIT = (
    "e304151c7da2e38ff19fe2f404157c976733d33d"
)
FEATURE_BRANCH = "codex/structural-pole-law"

CERTIFICATE_FILE = "certificate.json"
STRUCTURAL_POLE_FILE = "structural_pole_law.json"
MATCHING_CIRCUIT_FILE = "matching_circuit_structure.json"
VERTICAL_COMPONENT_FILE = "vertical_component.json"
MANIFEST_FILE = "manifest.json"

ARTIFACT_FILES = (
    CERTIFICATE_FILE,
    MATCHING_CIRCUIT_FILE,
    STRUCTURAL_POLE_FILE,
    VERTICAL_COMPONENT_FILE,
)
ALL_FILES = (*ARTIFACT_FILES, MANIFEST_FILE)

SOURCE_INPUTS = (
    "experiments/__init__.py",
    "experiments/krenn_quantum_graph/__init__.py",
    "experiments/krenn_quantum_graph/border_image.py",
    "experiments/krenn_quantum_graph/formal_lift.py",
    "experiments/krenn_quantum_graph/matching_circuit_structure.py",
    "experiments/krenn_quantum_graph/n6_deformation.py",
    "experiments/krenn_quantum_graph/source_ideal.py",
    "experiments/krenn_quantum_graph/structural_pole_law.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "experiments/krenn_quantum_graph/ternary_search.py",
    "experiments/krenn_quantum_graph/ternary_seed_orbits.py",
    "experiments/krenn_quantum_graph/transport.py",
    "experiments/krenn_quantum_graph/vertical_component.py",
    "experiments/krenn_quantum_graph/witness.py",
)

COMMITTED_BUNDLE_DIRECTORY = (
    ROOT / "results/krenn_quantum_graph/n6_d3_structural_pole_law"
)
MAX_JSON_BYTES = 4 * 1024 * 1024

EXACT_CHECK_NAMES = (
    "artifact_inventory_exact",
    "artifact_hashes_replayed",
    "producer_source_replayed",
    "input_source_ledger_replayed",
    "structural_pole_payload_recomputed",
    "matching_circuit_payload_recomputed",
    "vertical_component_payload_recomputed",
    "summary_certificate_recomputed",
    "natural_component_pole_law_replayed",
    "seed_charts_not_overcounted_as_global_components",
    "cross_payload_victim_fields_agree",
    "support21_recorded_receipt_validated",
    "support21_terminal_obstructions_recomputed",
    "recorded_support_lower_bound_22_validated",
    "long_replay_status_disclosed",
    "known_branch_unbounded_not_finite",
    "finite_exact_candidate_not_claimed",
    "global_nonexistence_not_claimed",
    "radical_membership_not_claimed",
)


class KrennStructuralPoleArtifactError(RuntimeError):
    """The structural-pole artifact failed integrity or semantic replay."""


@dataclass(frozen=True)
class LoadedStructuralPoleArtifact:
    """A fully verified structural-pole bundle."""

    directory: Path
    certificate: Mapping
    structural_pole: Mapping
    matching_circuit: Mapping
    vertical_component: Mapping
    manifest: Mapping
    checks: tuple[tuple[str, bool], ...]
    long_closure_replayed: bool = False

    @property
    def exact(self) -> bool:
        """Whether the complete ordered check inventory passed."""

        return self.checks == tuple(
            (name, True) for name in EXACT_CHECK_NAMES
        )


def _canonical_json_bytes(payload) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise KrennStructuralPoleArtifactError(
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
        raise KrennStructuralPoleArtifactError(
            "source input must not be a symbolic link"
        )
    resolved = requested.resolve()
    if not resolved.is_file():
        raise KrennStructuralPoleArtifactError("source input is absent")
    try:
        label = str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError as error:
        raise KrennStructuralPoleArtifactError(
            "source input is outside the repository"
        ) from error
    payload = _canonical_source_bytes(resolved)
    return {
        "path": label,
        "canonical_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hash_mode": "canonical-lf-text-v1",
    }


def _safe_local_path(directory: Path, label: str) -> Path:
    if type(label) is not str:
        raise KrennStructuralPoleArtifactError(
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
        raise KrennStructuralPoleArtifactError(
            "artifact path escaped its bundle"
        )
    path = directory / label
    if path.is_symlink() or not path.is_file():
        raise KrennStructuralPoleArtifactError(
            f"artifact is absent or linked: {label}"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise KrennStructuralPoleArtifactError(
            f"artifact could not be resolved: {label}"
        ) from error
    if resolved.parent != directory:
        raise KrennStructuralPoleArtifactError(
            "artifact path escaped its bundle"
        )
    return resolved


def _verify_local_record(directory: Path, record: Mapping) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise KrennStructuralPoleArtifactError(
            "artifact record schema changed"
        )
    if (
        type(record["path"]) is not str
        or type(record["bytes"]) is not int
        or record["bytes"] < 0
        or type(record["sha256"]) is not str
        or len(record["sha256"]) != 64
    ):
        raise KrennStructuralPoleArtifactError(
            "artifact record types changed"
        )
    path = _safe_local_path(directory, record["path"])
    if (
        path.stat().st_size != record["bytes"]
        or _sha256(path) != record["sha256"]
    ):
        raise KrennStructuralPoleArtifactError(
            f"artifact hash replay failed: {path.name}"
        )
    return path


def _safe_source_path(label: str) -> Path:
    if type(label) is not str:
        raise KrennStructuralPoleArtifactError(
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
        raise KrennStructuralPoleArtifactError(
            "source-ledger path is unsafe"
        )
    candidate = ROOT
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise KrennStructuralPoleArtifactError(
                "source-ledger input is linked"
            )
    if not candidate.is_file():
        raise KrennStructuralPoleArtifactError(
            "source-ledger input is absent"
        )
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise KrennStructuralPoleArtifactError(
            "source-ledger path escaped the repository"
        ) from error
    return resolved


def _verify_source_record(record: Mapping) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "canonical_bytes",
        "sha256",
        "hash_mode",
    }:
        raise KrennStructuralPoleArtifactError(
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
        raise KrennStructuralPoleArtifactError(
            "source-ledger record types changed"
        )
    path = _safe_source_path(record["path"])
    expected = _source_record(path)
    if _canonical_json_bytes(record) != _canonical_json_bytes(expected):
        raise KrennStructuralPoleArtifactError(
            f"source-ledger replay failed: {record['path']}"
        )
    return path


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennStructuralPoleArtifactError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str):
    raise KrennStructuralPoleArtifactError(
        f"non-finite JSON constant {value!r}"
    )


def _load_json(path: Path, label: str) -> Mapping:
    try:
        if path.is_symlink() or not path.is_file():
            raise KrennStructuralPoleArtifactError(
                f"{label} is absent or linked"
            )
        if path.stat().st_size > MAX_JSON_BYTES:
            raise KrennStructuralPoleArtifactError(
                f"{label} exceeds the reviewed size bound"
            )
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except KrennStructuralPoleArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennStructuralPoleArtifactError(
            f"could not decode {label}"
        ) from error
    if not isinstance(payload, dict):
        raise KrennStructuralPoleArtifactError(
            f"{label} is not a JSON object"
        )
    if raw != _canonical_json_bytes(payload):
        raise KrennStructuralPoleArtifactError(
            f"{label} is not canonical round-trippable JSON"
        )
    return payload


def _exact_payloads() -> dict[str, Mapping]:
    structural = certify_structural_pole_law().to_dict()
    matching = dict(exact_matching_circuit_structure())
    vertical = dict(exact_vertical_component_structure(support_cap=11))
    if (
        structural.get("schema") != STRUCTURAL_POLE_SCHEMA
        or matching.get("schema") != MATCHING_CIRCUIT_STRUCTURE_SCHEMA
        or vertical.get("schema") != VERTICAL_COMPONENT_SCHEMA
    ):
        raise KrennStructuralPoleArtifactError(
            "an exact structural payload schema changed"
        )
    return {
        STRUCTURAL_POLE_FILE: structural,
        MATCHING_CIRCUIT_FILE: matching,
        VERTICAL_COMPONENT_FILE: vertical,
    }


def _certificate(payloads: Mapping[str, Mapping]) -> dict:
    structural = payloads[STRUCTURAL_POLE_FILE]
    matching = payloads[MATCHING_CIRCUIT_FILE]
    vertical = payloads[VERTICAL_COMPONENT_FILE]

    structural_boundary = structural["claim_boundary"]
    atlas_boundary = structural["seed_component_atlas"][
        "claim_boundary"
    ]
    matching_claims = matching["claims"]
    vertical_claims = vertical["claims"]
    support_audit = vertical["recorded_support21_global_audit"]
    receipt = support_audit["totals"]

    if (
        structural_boundary[
            "natural_branch_relation_is_component_wide"
        ] is not True
        or structural_boundary[
            "unique_defect_seed_charts_certified"
        ] != 360
        or structural_boundary[
            "distinct_global_component_count_decided"
        ] is not False
        or atlas_boundary[
            "distinct_global_component_count_decided"
        ] is not False
        or matching_claims[
            "fixed_coloring_localized_matching_lattice_complete"
        ] is not True
        or matching_claims[
            "naive_degree_nine_global_identity_exists"
        ] is not False
        or vertical_claims[
            "known_Laurent_path_is_color_diagonal_orbit_in_moving_line_incidence"
        ] is not True
        or vertical_claims[
            "finite_exact_support_lower_bound"
        ] != 22
        or support_audit["finite_exact_support_lower_bound"] != 22
        or receipt["nodes_examined"] != 12_992_269
        or receipt["singleton_free_terminals"] != 12
        or receipt["odd_mixed_circuit_terminals"] != 8
        or receipt["pure_cancellation_terminals"] != 4
    ):
        raise KrennStructuralPoleArtifactError(
            "the exact structural claim boundary changed"
        )

    root_receipts = support_audit["root_receipts"]
    if tuple(
        row["nodes_examined"] for row in root_receipts
    ) != EXPECTED_SUPPORT21_CLOSURE_NODES:
        raise KrennStructuralPoleArtifactError(
            "the support-21 root receipt changed"
        )

    return {
        "schema": CERTIFICATE_SCHEMA,
        "problem": {
            "n": 6,
            "d": 3,
            "equation_count": 729,
            "variable_count": 135,
            "target": "GHZ_6,3",
            "victim_coloring": [0, 0, 2, 1, 2, 1],
        },
        "repository_provenance": {
            "baseline_origin_main": BASELINE_COMMIT,
            "feature_branch": FEATURE_BRANCH,
            "contextual_predecessor_commits_not_merged": [
                COUNTEREXAMPLE_SEARCH_COMMIT,
                OBSTRUCTION_CERTIFICATE_COMMIT,
            ],
        },
        "method": {
            "description": (
                "exact symmetry, gauge characters, matching-incidence "
                "lattices, singleton closure, and integer terminal "
                "relations"
            ),
            "arithmetic": "integer and rational exact arithmetic",
            "numerical_search_used_as_proof": False,
            "finite_field_search_used_as_proof": False,
            "symmetry_related_weights_equated": False,
            "random_seeds": [],
        },
        "exact_results": {
            "known_Laurent_path_is_color_diagonal_orbit_in_moving_line_incidence": (
                True
            ),
            "natural_local_component_obeys_u_times_Q_equals_1": True,
            "natural_local_component_has_no_finite_u_zero_endpoint": True,
            "marked_unique_defect_seed_charts": 360,
            "distinct_coordinate_supports_in_atlas": 360,
            "distinct_global_component_count_decided": False,
            "fixed_victim_nine_coordinate_seed_orbits_under_stabilizer": (
                104
            ),
            "fixed_victim_nine_coordinate_orbits_without_nonvictim_singleton": (
                1
            ),
            "localized_matching_lattice_rank": 5,
            "primitive_K3_3_parity_circuits": 10,
            "six_retained_coordinate_subspaces_excluded": True,
            "finite_exact_support_lower_bound": 22,
            "support_sizes_through_21_excluded": True,
            "natural_coordinate_retention_assumed": False,
        },
        "support_21_exhaustion": {
            "algorithm": (
                "deterministic first-singleton missing-set closure"
            ),
            "workers": 8,
            "declared_worker_limit": 16,
            "max_total_support": 21,
            "addition_cap_over_nine_coordinate_seed": 12,
            "node_cap_per_root": None,
            "root_nodes_examined": list(
                EXPECTED_SUPPORT21_CLOSURE_NODES
            ),
            "total_nodes_examined": 12_992_269,
            "singleton_free_terminals": 12,
            "terminal_obstructions": {
                "odd_mixed_integer_relations": 8,
                "pure_output_cancellations": 4,
            },
            "executed_disposable_command_template": (
                "python -B "
                "D:\\KrennScratch\\structural_pole_law\\"
                "closure_probe.py --index I --cap 12 --output "
                "D:\\KrennScratch\\structural_pole_law\\cap12_I.json"
            ),
            "executed_source_controlled_command_template": (
                "python -B -m "
                "experiments.krenn_quantum_graph.vertical_component "
                "--root-index I --max-total-support 21 --output "
                "D:\\KrennScratch\\structural_pole_law\\"
                "official_replay\\official21_I.json"
            ),
            "source_controlled_replay_completed": True,
            "source_controlled_receipt_directory": (
                "D:\\KrennScratch\\structural_pole_law\\official_replay"
            ),
            "independent_full_regeneration_test": (
                "PowerShell: $env:KRENN_RUN_LONG_TESTS='1'; "
                "python -B -m unittest "
                "tests.test_krenn_vertical_component; "
                "Remove-Item Env:KRENN_RUN_LONG_TESTS"
            ),
            "long_bundle_verification_command": (
                "python -B -m "
                "experiments.krenn_quantum_graph."
                "structural_pole_artifact "
                "results\\krenn_quantum_graph\\"
                "n6_d3_structural_pole_law --verify --long"
            ),
            "routine_bundle_verification_replays_full_closure": False,
            "long_bundle_verification_available": True,
            "routine_validation_scope": (
                "source-bound root counts plus exact recomputation of "
                "all 12 terminal lattice contradictions"
            ),
            "large_frontiers_or_caches_committed": False,
        },
        "weight_status": {
            "known_Laurent_branch_weights_remain_finite": False,
            "vertex_scalar_gauge_invariant_quantity": "Q",
            "full_color_diagonal_gauge_invariant_quantity": "u*Q",
            "Q_full_color_diagonal_character": (
                "inverse of the victim-output character"
            ),
            "known_natural_component_relation": "u*Q=1",
            "known_natural_component_behavior_as_u_tends_to_zero": (
                "Q tends to infinity"
            ),
            "finite_bounded_candidate_found": False,
        },
        "exact_candidate_status": {
            "finite_exact_witness_found": False,
            "candidate_field": None,
            "primary_729_equation_verification": "not applicable",
            "independent_729_equation_verification": "not applicable",
            "witness_bundle_emitted": False,
        },
        "claim_boundary": {
            "support_22_or_larger_witness_excluded": False,
            "all_vertical_components_excluded": False,
            "D_in_radical_J_mix_decided": False,
            "global_finite_GHZ_nonexistence_proved": False,
            "exact_affine_GHZ_membership_status": "undecided",
        },
        "next_exact_attempts": [
            (
                "compute the localized direct-GHZ ideal on each of the "
                "eight pure-seed orbit charts; unit ideals on the full "
                "cover or an exact point would decide affine membership"
            ),
            (
                "in the moving-line formulation, saturate the natural "
                "fixed-victim chart away from u*Q=1 and cover the other "
                "103 fixed-victim seed charts exactly"
            ),
            (
                "classify support-22 singleton-free terminals modulo "
                "S_6 x S_3 and test their integer exponent lattices"
            ),
        ],
    }


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


def generate_structural_pole_bundle(
    output_directory: Path | str,
) -> Path:
    """Write the exact five-file bundle and replay it immediately."""

    requested = Path(output_directory)
    if requested.is_symlink():
        raise KrennStructuralPoleArtifactError(
            "structural-pole output directory must not be linked"
        )
    directory = requested.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    entries = tuple(directory.iterdir())
    unexpected = {
        path.name for path in entries if path.name not in ALL_FILES
    }
    if unexpected:
        raise KrennStructuralPoleArtifactError(
            "refusing a directory with unexpected files: "
            f"{sorted(unexpected)}"
        )
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise KrennStructuralPoleArtifactError(
            "existing bundle entries must be regular local files"
        )

    payloads = _exact_payloads()
    payloads[CERTIFICATE_FILE] = _certificate(payloads)
    for label in sorted(ARTIFACT_FILES):
        _write_json_atomic(directory / label, payloads[label])
    _write_json_atomic(directory / MANIFEST_FILE, _manifest(directory))
    verify_structural_pole_bundle(directory)
    return directory / CERTIFICATE_FILE


def verify_structural_pole_bundle(
    output_directory: Path | str,
    *,
    replay_long_closure: bool = False,
) -> LoadedStructuralPoleArtifact:
    """Replay exact inventory, provenance, hashes, and all semantics."""

    requested = Path(output_directory)
    if requested.is_symlink() or not requested.is_dir():
        raise KrennStructuralPoleArtifactError(
            "structural-pole bundle directory is absent or linked"
        )
    directory = requested.resolve()
    entries = tuple(directory.iterdir())
    if (
        {path.name for path in entries} != set(ALL_FILES)
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise KrennStructuralPoleArtifactError(
            "structural-pole bundle file inventory changed"
        )

    manifest_path = _safe_local_path(directory, MANIFEST_FILE)
    manifest = _load_json(manifest_path, "structural-pole manifest")
    if set(manifest) != {
        "schema",
        "producer",
        "artifacts",
        "inputs",
    } or manifest.get("schema") != MANIFEST_SCHEMA:
        raise KrennStructuralPoleArtifactError(
            "structural-pole manifest schema changed"
        )

    records = manifest.get("artifacts")
    if (
        not isinstance(records, list)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise KrennStructuralPoleArtifactError(
            "structural-pole artifact ledger changed"
        )
    labels = tuple(record.get("path") for record in records)
    if labels != tuple(sorted(ARTIFACT_FILES)):
        raise KrennStructuralPoleArtifactError(
            "structural-pole artifact ledger is not exact and sorted"
        )
    replayed = {
        label: _verify_local_record(directory, record)
        for label, record in zip(labels, records, strict=True)
    }

    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise KrennStructuralPoleArtifactError(
            "structural-pole producer record changed"
        )
    _verify_source_record(producer)
    expected_producer = _source_record(Path(__file__))
    if (
        _canonical_json_bytes(producer)
        != _canonical_json_bytes(expected_producer)
    ):
        raise KrennStructuralPoleArtifactError(
            "structural-pole producer identity changed"
        )

    input_records = manifest.get("inputs")
    if (
        not isinstance(input_records, list)
        or any(not isinstance(record, dict) for record in input_records)
    ):
        raise KrennStructuralPoleArtifactError(
            "structural-pole input ledger changed"
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
        raise KrennStructuralPoleArtifactError(
            "structural-pole input ledger changed"
        )

    loaded_payloads = {
        label: _load_json(
            replayed[label],
            f"structural-pole {label}",
        )
        for label in ARTIFACT_FILES
    }
    expected_payloads = _exact_payloads()
    expected_payloads[CERTIFICATE_FILE] = _certificate(
        expected_payloads
    )
    for label in ARTIFACT_FILES:
        actual = loaded_payloads[label]
        expected = expected_payloads[label]
        if set(actual) != set(expected):
            raise KrennStructuralPoleArtifactError(
                f"{label} schema changed"
            )
        if _canonical_json_bytes(actual) != _canonical_json_bytes(
            expected
        ):
            raise KrennStructuralPoleArtifactError(
                f"{label} failed semantic replay"
            )

    certificate = loaded_payloads[CERTIFICATE_FILE]
    structural = loaded_payloads[STRUCTURAL_POLE_FILE]
    matching = loaded_payloads[MATCHING_CIRCUIT_FILE]
    vertical = loaded_payloads[VERTICAL_COMPONENT_FILE]
    exact_results = certificate["exact_results"]
    boundary = certificate["claim_boundary"]
    candidate = certificate["exact_candidate_status"]
    support = vertical["recorded_support21_global_audit"]
    victim_fields_agree = (
        structural["natural_component"]["parameters"][
            "victim_equation"
        ]
        == matching["victim_symmetry_pole"]["victim_equation"]
        == vertical["parameters"]["victim_equation"]
        == 70
        and structural["natural_component"]["parameters"][
            "victim_coloring"
        ]
        == matching["victim_symmetry_pole"]["victim_coloring"]
        == vertical["parameters"]["victim_coloring"]
        == [0, 0, 2, 1, 2, 1]
    )
    long_closure_replayed = False
    if replay_long_closure:
        long_closure = replay_support21_global_closure()
        long_closure_replayed = (
            tuple(
                root.nodes_examined for root in long_closure.roots
            )
            == EXPECTED_SUPPORT21_CLOSURE_NODES
            and sum(
                len(root.singleton_free_supports)
                for root in long_closure.roots
            )
            == 12
        )
        if not long_closure_replayed:
            raise KrennStructuralPoleArtifactError(
                "the full support-21 closure replay failed"
            )

    checks = (
        ("artifact_inventory_exact", True),
        ("artifact_hashes_replayed", True),
        ("producer_source_replayed", True),
        ("input_source_ledger_replayed", True),
        ("structural_pole_payload_recomputed", True),
        ("matching_circuit_payload_recomputed", True),
        ("vertical_component_payload_recomputed", True),
        ("summary_certificate_recomputed", True),
        (
            "natural_component_pole_law_replayed",
            exact_results[
                "natural_local_component_obeys_u_times_Q_equals_1"
            ] is True,
        ),
        (
            "seed_charts_not_overcounted_as_global_components",
            exact_results[
                "distinct_global_component_count_decided"
            ] is False,
        ),
        (
            "cross_payload_victim_fields_agree",
            victim_fields_agree,
        ),
        (
            "support21_recorded_receipt_validated",
            support["totals"]["nodes_examined"] == 12_992_269
            and tuple(
                row["nodes_examined"]
                for row in support["root_receipts"]
            ) == EXPECTED_SUPPORT21_CLOSURE_NODES,
        ),
        (
            "support21_terminal_obstructions_recomputed",
            support["totals"]["singleton_free_terminals"] == 12
            and support["totals"][
                "odd_mixed_circuit_terminals"
            ] == 8
            and support["totals"]["pure_cancellation_terminals"] == 4,
        ),
        (
            "recorded_support_lower_bound_22_validated",
            exact_results["finite_exact_support_lower_bound"] == 22
            and exact_results[
                "natural_coordinate_retention_assumed"
            ] is False,
        ),
        (
            "long_replay_status_disclosed",
            certificate["support_21_exhaustion"][
                "routine_bundle_verification_replays_full_closure"
            ] is False
            and certificate["support_21_exhaustion"][
                "long_bundle_verification_available"
            ] is True,
        ),
        (
            "known_branch_unbounded_not_finite",
            certificate["weight_status"][
                "known_Laurent_branch_weights_remain_finite"
            ] is False
            and certificate["weight_status"][
                "finite_bounded_candidate_found"
            ] is False,
        ),
        (
            "finite_exact_candidate_not_claimed",
            candidate["finite_exact_witness_found"] is False
            and candidate["witness_bundle_emitted"] is False,
        ),
        (
            "global_nonexistence_not_claimed",
            boundary[
                "global_finite_GHZ_nonexistence_proved"
            ] is False,
        ),
        (
            "radical_membership_not_claimed",
            boundary["D_in_radical_J_mix_decided"] is False
            and boundary[
                "exact_affine_GHZ_membership_status"
            ] == "undecided",
        ),
    )
    loaded = LoadedStructuralPoleArtifact(
        directory=directory,
        certificate=certificate,
        structural_pole=structural,
        matching_circuit=matching,
        vertical_component=vertical,
        manifest=manifest,
        checks=checks,
        long_closure_replayed=long_closure_replayed,
    )
    if not loaded.exact:
        raise KrennStructuralPoleArtifactError(
            "structural-pole exact checks failed"
        )
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the exact structural-pole bundle."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--long",
        action="store_true",
        help="recompute all 12,992,269 support-closure nodes",
    )
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        generate_structural_pole_bundle(arguments.output_directory)
    loaded = verify_structural_pole_bundle(
        arguments.output_directory,
        replay_long_closure=arguments.long,
    )
    print(json.dumps({
        "exact": loaded.exact,
        "finite_exact_support_lower_bound": (
            loaded.certificate["exact_results"][
                "finite_exact_support_lower_bound"
            ]
        ),
        "finite_exact_witness_found": False,
        "long_closure_replayed": loaded.long_closure_replayed,
        "global_finite_GHZ_nonexistence_proved": False,
        "exact_affine_GHZ_membership_status": "undecided",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
