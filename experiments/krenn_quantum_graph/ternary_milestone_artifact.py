"""Replayable bundle for the finite ``n=6,d=3`` ternary milestone."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.system import generate_sparse_system
from experiments.krenn_quantum_graph.ternary_search import (
    TERNARY_SEARCH_SCHEMA,
    TernarySearchPlan,
    TernarySearchResult,
    checkpoint_from_dict,
    replay_checkpoint,
    run_bounded_ternary_search,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    SEED_ORBIT_SCHEMA,
    ternary_seed_orbit_census,
    verify_ternary_seed_orbit_census,
)


ROOT = Path(__file__).resolve().parents[2]
BUNDLE_SCHEMA = "krenn-quantum-graph-ternary-milestone-bundle-v1"
SEARCH_RECORD_SCHEMA = "krenn-quantum-graph-bounded-search-record-v1"
CERTIFICATE_SCHEMA = "krenn-quantum-graph-ternary-certificate-v1"
MANIFEST_SCHEMA = "krenn-quantum-graph-ternary-manifest-v1"

ORBIT_FILE = "orbit_census.json"
SEARCH_FILE = "bounded_search.json"
CERTIFICATE_FILE = "certificate.json"
MANIFEST_FILE = "manifest.json"
ARTIFACT_FILES = (CERTIFICATE_FILE, ORBIT_FILE, SEARCH_FILE)
ALL_FILES = (*ARTIFACT_FILES, MANIFEST_FILE)

SOURCE_INPUTS = (
    "experiments/krenn_quantum_graph/targets.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/witness.py",
    "experiments/krenn_quantum_graph/transport.py",
    "experiments/krenn_quantum_graph/ternary_search.py",
    "experiments/krenn_quantum_graph/ternary_seed_orbits.py",
)


class KrennTernaryArtifactError(RuntimeError):
    """Ternary milestone emission, integrity, or replay failed."""


@dataclass(frozen=True)
class LoadedTernaryMilestone:
    directory: Path
    orbit_census: Mapping
    search_record: Mapping
    certificate: Mapping
    result: TernarySearchResult

    @property
    def exact(self) -> bool:
        return all(
            bool(value)
            for value in self.certificate["exact_checks"].values()
        )


def default_search_plan() -> TernarySearchPlan:
    return TernarySearchPlan(
        node_cap=32,
        time_cap_seconds=30.0,
        support_cap=12,
        frontier_cap=400,
    )


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    path = path.resolve()
    try:
        label = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError as error:
        raise KrennTernaryArtifactError(
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
    if (
        not label
        or Path(label).name != label
        or "/" in label
        or "\\" in label
    ):
        raise KrennTernaryArtifactError(
            "artifact path escaped its bundle"
        )
    path = directory / label
    if path.is_symlink() or not path.is_file():
        raise KrennTernaryArtifactError(
            f"artifact is absent or linked: {label}"
        )
    return path


def _verify_local_record(directory: Path, record: Mapping) -> Path:
    if set(record) != {"path", "bytes", "sha256"}:
        raise KrennTernaryArtifactError(
            "artifact record schema changed"
        )
    path = _safe_local_path(directory, str(record["path"]))
    if (
        path.stat().st_size != int(record["bytes"])
        or _sha256(path) != record["sha256"]
    ):
        raise KrennTernaryArtifactError(
            f"artifact hash replay failed: {path.name}"
        )
    return path


def _verify_source_record(record: Mapping) -> Path:
    if set(record) != {
        "path",
        "canonical_bytes",
        "sha256",
        "hash_mode",
    }:
        raise KrennTernaryArtifactError(
            "source-ledger record schema changed"
        )
    label = str(record["path"])
    pure = Path(label)
    if (
        not label
        or pure.is_absolute()
        or "\\" in label
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise KrennTernaryArtifactError(
            "source-ledger path is unsafe"
        )
    path = (ROOT / pure).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise KrennTernaryArtifactError(
            "source-ledger path escaped the repository"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise KrennTernaryArtifactError(
            "source-ledger input is absent or linked"
        )
    if dict(record) != _source_record(path):
        raise KrennTernaryArtifactError(
            f"source-ledger replay failed: {label}"
        )
    return path


def _search_record(result: TernarySearchResult) -> dict:
    return {
        "schema": SEARCH_RECORD_SCHEMA,
        "plan": result.plan.to_dict(),
        "result": result.to_dict(),
    }


def _plan_from_payload(payload: Mapping) -> TernarySearchPlan:
    try:
        plan = TernarySearchPlan(
            n=int(payload["n"]),
            d=int(payload["d"]),
            mode=str(payload["mode"]),
            schema=str(payload["schema"]),
            node_cap=int(payload["node_cap"]),
            time_cap_seconds=float(payload["time_cap_seconds"]),
            support_cap=int(payload["support_cap"]),
            frontier_cap=int(payload["frontier_cap"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise KrennTernaryArtifactError(
            "bounded-search plan is malformed"
        ) from error
    if plan.to_dict() != payload:
        raise KrennTernaryArtifactError(
            "bounded-search plan is not canonical"
        )
    return plan


def _result_from_search_record(
    system, payload: Mapping
) -> TernarySearchResult:
    if payload.get("schema") != SEARCH_RECORD_SCHEMA:
        raise KrennTernaryArtifactError(
            "bounded-search record schema changed"
        )
    plan = _plan_from_payload(payload.get("plan", {}))
    result_payload = payload.get("result", {})
    try:
        checkpoint = checkpoint_from_dict(
            result_payload["checkpoint"]
        )
        result = TernarySearchResult(
            plan=plan,
            checkpoint=checkpoint,
            termination=str(result_payload["termination"]),
            nodes_examined_this_run=int(
                result_payload["nodes_examined_this_run"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise KrennTernaryArtifactError(
            "bounded-search result is malformed"
        ) from error
    if result.to_dict() != result_payload:
        raise KrennTernaryArtifactError(
            "bounded-search result is not canonical"
        )
    replay_checkpoint(system, plan, checkpoint)
    replayed = run_bounded_ternary_search(system, plan)
    if replayed != result:
        raise KrennTernaryArtifactError(
            "bounded search failed deterministic full replay"
        )
    return result


def _build_certificate(
    orbit_census: Mapping,
    result: TernarySearchResult,
) -> dict:
    orbit_counts = orbit_census["counts"]
    candidate = result.checkpoint.best_candidate
    solution = candidate.exact_solution
    status = (
        "exact-integer-fixed-target-solution-certified"
        if solution
        else "bounded-fixed-target-ternary-search-no-zero-found"
    )
    return {
        "schema": CERTIFICATE_SCHEMA,
        "bundle_schema": BUNDLE_SCHEMA,
        "status": status,
        "parameters": {
            "n": 6,
            "d": 3,
            "domain": [-1, 0, 1],
            "search_mode": "fixed-target",
        },
        "counts": {
            "variables": 135,
            "equations": 729,
            "monomials": 10_935,
            "perfect_matchings": orbit_counts["perfect_matchings"],
            "ordered_seeds": orbit_counts[
                "ordered_three_matching_seeds"
            ],
            "color_unordered_multisets": orbit_counts[
                "color_unordered_matching_multisets"
            ],
            "seed_orbits": orbit_counts["symmetry_orbits"],
        },
        "bounded_search": {
            "termination": result.termination,
            "nodes_examined": result.nodes_examined_this_run,
            "total_nodes_examined": (
                result.checkpoint.total_nodes_examined
            ),
            "node_cap": result.plan.node_cap,
            "time_cap_seconds": result.plan.time_cap_seconds,
            "support_cap": result.plan.support_cap,
            "frontier_cap": result.plan.frontier_cap,
            "frontier_remaining": len(result.checkpoint.frontier),
            "best_support_size": candidate.support_size,
            "best_nonzero_residual_count": (
                candidate.nonzero_residual_count
            ),
            "best_residual_l1": candidate.residual_l1,
        },
        "artifacts": {
            "orbit_census": ORBIT_FILE,
            "bounded_search": SEARCH_FILE,
            "certificate": CERTIFICATE_FILE,
            "manifest": MANIFEST_FILE,
        },
        "exact_checks": {
            "orbit_census_recomputed": True,
            "eight_seed_orbits_replayed": (
                orbit_counts["symmetry_orbits"] == 8
            ),
            "ordered_seed_count_3375": (
                orbit_counts["ordered_three_matching_seeds"]
                == 3375
            ),
            "bounded_search_fully_replayed": True,
            "checkpoint_exactly_replayed": True,
            "best_candidate_original_equations_replayed": True,
        },
        "claim_boundary": {
            "orbit_indexing_only": True,
            "symmetry_weight_equalities": 0,
            "bounded_search_only": True,
            "search_exhaustive": False,
            "tree_certificate_complete": False,
            "exhaustive_nonexistence": False,
            "nonexistence_proved": False,
            "solution_certified": solution,
            "proof_over_C": solution,
            "statement": (
                "A bounded fixed-target ternary search was replayed. "
                "A capped miss proves only that no exact zero was found "
                "among the explored nodes."
                if not solution
                else (
                    "The listed integer assignment was replayed exactly "
                    "in all 729 equations and therefore embeds in C."
                )
            ),
        },
    }


def _manifest(directory: Path) -> dict:
    producer = _source_record(Path(__file__))
    inputs = [
        _source_record(ROOT / label) for label in SOURCE_INPUTS
    ]
    return {
        "schema": MANIFEST_SCHEMA,
        "producer": producer,
        "artifacts": [
            _local_record(directory / label)
            for label in sorted(ARTIFACT_FILES)
        ],
        "inputs": sorted(inputs, key=lambda row: row["path"]),
    }


def generate_ternary_milestone_bundle(
    output_directory: Path | str,
    *,
    plan: TernarySearchPlan | None = None,
) -> Path:
    """Generate all four files and immediately replay the bundle."""

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name
        for path in output_directory.iterdir()
        if path.name not in ALL_FILES
    }
    if unexpected:
        raise KrennTernaryArtifactError(
            f"refusing a directory with unexpected files: "
            f"{sorted(unexpected)}"
        )
    plan = default_search_plan() if plan is None else plan
    system = generate_sparse_system(6, 3)
    orbit_census = ternary_seed_orbit_census()
    result = run_bounded_ternary_search(system, plan)
    if result.termination == "time-cap-reached":
        raise KrennTernaryArtifactError(
            "generation hit the nondeterministic safety time cap"
        )
    search_record = _search_record(result)
    certificate = _build_certificate(orbit_census, result)
    _write_json_atomic(output_directory / ORBIT_FILE, orbit_census)
    _write_json_atomic(output_directory / SEARCH_FILE, search_record)
    _write_json_atomic(
        output_directory / CERTIFICATE_FILE, certificate
    )
    _write_json_atomic(
        output_directory / MANIFEST_FILE, _manifest(output_directory)
    )
    verify_ternary_milestone_bundle(output_directory)
    return output_directory / CERTIFICATE_FILE


def _load_json(path: Path, label: str) -> Mapping:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KrennTernaryArtifactError(
            f"could not decode {label}"
        ) from error
    if not isinstance(payload, dict):
        raise KrennTernaryArtifactError(
            f"{label} is not a JSON object"
        )
    return payload


def verify_ternary_milestone_bundle(
    output_directory: Path | str,
) -> LoadedTernaryMilestone:
    """Replay source/artifact hashes, all orbits, and the bounded search."""

    requested = Path(output_directory)
    if requested.is_symlink() or not requested.is_dir():
        raise KrennTernaryArtifactError(
            "ternary milestone directory is absent or linked"
        )
    directory = requested.resolve()
    if {path.name for path in directory.iterdir()} != set(ALL_FILES):
        raise KrennTernaryArtifactError(
            "ternary milestone file inventory changed"
        )
    manifest = _load_json(
        directory / MANIFEST_FILE, "manifest"
    )
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise KrennTernaryArtifactError(
            "ternary manifest schema changed"
        )
    records = tuple(manifest.get("artifacts", ()))
    labels = tuple(str(record.get("path", "")) for record in records)
    if labels != tuple(sorted(ARTIFACT_FILES)):
        raise KrennTernaryArtifactError(
            "artifact ledger is not exact and sorted"
        )
    replayed = {
        label: _verify_local_record(directory, record)
        for label, record in zip(labels, records)
    }
    expected_producer = _source_record(Path(__file__))
    if manifest.get("producer") != expected_producer:
        raise KrennTernaryArtifactError(
            "ternary producer identity changed"
        )
    _verify_source_record(expected_producer)
    input_records = tuple(manifest.get("inputs", ()))
    expected_inputs = tuple(
        sorted(
            (
                _source_record(ROOT / label)
                for label in SOURCE_INPUTS
            ),
            key=lambda row: row["path"],
        )
    )
    if input_records != expected_inputs:
        raise KrennTernaryArtifactError(
            "ternary source ledger changed"
        )
    for record in input_records:
        _verify_source_record(record)

    orbit_census = _load_json(
        replayed[ORBIT_FILE], "orbit census"
    )
    if orbit_census.get("schema") != SEED_ORBIT_SCHEMA:
        raise KrennTernaryArtifactError(
            "orbit-census schema changed"
        )
    try:
        verify_ternary_seed_orbit_census(orbit_census)
    except Exception as error:
        raise KrennTernaryArtifactError(
            "orbit census failed exact replay"
        ) from error

    search_record = _load_json(
        replayed[SEARCH_FILE], "bounded search"
    )
    system = generate_sparse_system(6, 3)
    result = _result_from_search_record(system, search_record)
    certificate = _load_json(
        replayed[CERTIFICATE_FILE], "certificate"
    )
    expected_certificate = _build_certificate(
        orbit_census, result
    )
    if certificate != expected_certificate:
        raise KrennTernaryArtifactError(
            "ternary certificate failed semantic replay"
        )
    bundle = LoadedTernaryMilestone(
        directory=directory,
        orbit_census=orbit_census,
        search_record=search_record,
        certificate=certificate,
        result=result,
    )
    if not bundle.exact:
        raise KrennTernaryArtifactError(
            "ternary milestone exact checks failed"
        )
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify a finite Krenn ternary bundle."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--node-cap", type=int, default=32)
    parser.add_argument("--time-cap-seconds", type=float, default=30.0)
    parser.add_argument("--support-cap", type=int, default=12)
    parser.add_argument("--frontier-cap", type=int, default=400)
    arguments = parser.parse_args(argv)
    if arguments.verify:
        bundle = verify_ternary_milestone_bundle(
            arguments.output_directory
        )
    else:
        plan = TernarySearchPlan(
            node_cap=arguments.node_cap,
            time_cap_seconds=arguments.time_cap_seconds,
            support_cap=arguments.support_cap,
            frontier_cap=arguments.frontier_cap,
        )
        generate_ternary_milestone_bundle(
            arguments.output_directory, plan=plan
        )
        bundle = verify_ternary_milestone_bundle(
            arguments.output_directory
        )
    print(
        json.dumps(
            {
                "exact": bundle.exact,
                "status": bundle.certificate["status"],
                "termination": bundle.result.termination,
                "nodes_examined": (
                    bundle.result.nodes_examined_this_run
                ),
                "seed_orbits": bundle.certificate["counts"][
                    "seed_orbits"
                ],
                "nonexistence_proved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
