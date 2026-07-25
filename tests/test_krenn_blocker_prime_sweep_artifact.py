from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.blocker_quotient_prime_sweep import (
    ALGORITHM,
    CLAIM_KEYS,
    CPUS,
    DEGREE24_SPEC,
    DEGREE30_SPEC,
    DOCKERFILE_PATH,
    DOCKER_IMAGE,
    DOCKER_IMAGE_ID,
    ENGINE_BUDGET_SECONDS,
    MAXIMUM_WORKERS,
    MEMORY_GIB,
    PROBES,
    RECEIPT_ARCHIVE_SCHEMA,
    RUN_SCHEMA,
    SINGULAR_VERSION,
    SPECS,
    SWEEP_PRIMES,
    SWEEP_SCHEMA,
    VERIFICATION_FALSE_KEYS,
    _baseline_provenance,
    _canonical_json_bytes,
    _probe_payload,
    chart_for_spec,
    claim_boundary,
    coefficient_audit,
)
from experiments.krenn_quantum_graph.blocker_quotient_reconnaissance import (
    TRANSCRIPT_SCHEMA,
    chart_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "results"
    / "krenn_quantum_graph"
    / "n8_d3_square_blocker_prime_sweep"
)
README = "README.md"
SWEEP = "sweep.json"
RECEIPTS = "receipts.json"
MANIFEST = "manifest.json"
BUNDLE_FILES = {README, SWEEP, RECEIPTS}
ALL_FILES = BUNDLE_FILES | {MANIFEST}
SOURCE_FILES = {
    "README.md",
    "experiments/krenn_quantum_graph/"
    "blocker_quotient_prime_sweep.py",
    "experiments/krenn_quantum_graph/"
    "blocker_quotient_reconnaissance.py",
    "experiments/krenn_quantum_graph/blocker_quotient_runner.py",
    "experiments/krenn_quantum_graph/n8_k5_singular.Dockerfile",
    "experiments/krenn_quantum_graph/"
    "perfect_matching_blocker_ideals.py",
    "experiments/krenn_quantum_graph/witness.py",
    "tests/test_krenn_blocker_quotient_prime_sweep.py",
    "tests/test_krenn_blocker_prime_sweep_artifact.py",
}
MANIFEST_SCHEMA = (
    "krenn-n8-d3-square-blocker-prime-sweep-manifest-v2"
)


class ArtifactError(RuntimeError):
    pass


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    if path.stat().st_size > 1024 * 1024:
        raise ArtifactError("small bundle JSON exceeded one MiB")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactError(f"nonfinite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ArtifactError("bundle JSON is malformed") from error
    if type(payload) is not dict:
        raise ArtifactError("bundle JSON must be one object")
    if path.read_bytes() != _canonical_json_bytes(payload):
        raise ArtifactError("bundle JSON is not canonical")
    return payload


def _record(path):
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_record(path, record):
    if (
        path.is_symlink()
        or not path.is_file()
        or type(record) is not dict
        or set(record) != {"bytes", "sha256"}
        or type(record["bytes"]) is not int
        or record["bytes"] < 0
        or type(record["sha256"]) is not str
        or len(record["sha256"]) != 64
        or record != _record(path)
    ):
        raise ArtifactError(f"hash replay failed for {path.name}")


def _false_claims(payload):
    return (
        type(payload) is dict
        and set(payload) == set(CLAIM_KEYS)
        and payload == claim_boundary()
        and all(value is False for value in payload.values())
    )


def _verify_bundle(directory):
    directory = directory.resolve()
    if (
        {
            path.name
            for path in directory.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        != ALL_FILES
        or any(
            path.is_symlink() or not path.is_file()
            for path in directory.iterdir()
        )
    ):
        raise ArtifactError("bundle inventory changed")

    manifest = _strict_json(directory / MANIFEST)
    if (
        set(manifest)
        != {
            "bundle_files",
            "claim_boundary",
            "schema",
            "source_files",
        }
        or manifest["schema"] != MANIFEST_SCHEMA
        or not _false_claims(manifest["claim_boundary"])
        or type(manifest["bundle_files"]) is not dict
        or set(manifest["bundle_files"]) != BUNDLE_FILES
        or type(manifest["source_files"]) is not dict
        or set(manifest["source_files"]) != SOURCE_FILES
    ):
        raise ArtifactError("manifest schema, inventory, or claims changed")
    for filename, record in manifest["bundle_files"].items():
        _verify_record(directory / filename, record)
    for filename, record in manifest["source_files"].items():
        _verify_record(ROOT / filename, record)

    sweep = _strict_json(directory / SWEEP)
    if (
        set(sweep)
        != {
            "baseline_provenance",
            "charts",
            "claim_boundary",
            "complete",
            "engine",
            "interpretation",
            "prime_policy",
            "runs",
            "schema",
            "specialization",
            "status",
        }
        or sweep["schema"] != SWEEP_SCHEMA
        or sweep["status"] != "complete-exact-modular-replay"
        or sweep["complete"] is not True
        or not _false_claims(sweep["claim_boundary"])
        or sweep["claim_boundary"] != manifest["claim_boundary"]
    ):
        raise ArtifactError("sweep schema, status, or claims changed")
    if sweep["specialization"] != {
        "seed": 80320260725,
        "coefficient_height": 127,
        "coefficient_slots_only": True,
        "within_each_chart_coefficients_pairwise_distinct_over_Z": True,
        "degree24_and_degree30_use_separate_fixed_specializations": True,
    }:
        raise ArtifactError("fixed specialization record changed")
    if sweep["prime_policy"] != {
        "selection_rule": (
            "first 20 primes strictly greater than "
            "2*coefficient_height=254"
        ),
        "ordered_primes": list(SWEEP_PRIMES),
        "count": 20,
        "minimum": 257,
        "maximum": 373,
        "nonadaptive": True,
    }:
        raise ArtifactError("prime policy changed")
    expected_engine = {
        "image": DOCKER_IMAGE,
        "image_id": DOCKER_IMAGE_ID,
        "singular_version": SINGULAR_VERSION,
        "dockerfile": {
            "repository_path": DOCKERFILE_PATH.relative_to(
                ROOT
            ).as_posix(),
            **_record(DOCKERFILE_PATH),
        },
        "algorithm": ALGORITHM,
        "network": "none",
        "cpus": CPUS,
        "maximum_workers": MAXIMUM_WORKERS,
        "memory_gib": MEMORY_GIB,
        "per_probe_engine_budget_seconds": ENGINE_BUDGET_SECONDS,
    }
    if sweep["engine"] != expected_engine:
        raise ArtifactError("engine provenance changed")
    if sweep["baseline_provenance"] != _baseline_provenance():
        raise ArtifactError("degree-24 baseline provenance changed")

    if type(sweep["charts"]) is not list or len(sweep["charts"]) != 2:
        raise ArtifactError("square blocker chart census changed")
    expected_dimensions = {
        DEGREE24_SPEC.key: 24,
        DEGREE30_SPEC.key: 30,
    }
    expected_basis_sizes = {
        DEGREE24_SPEC.key: 58,
        DEGREE30_SPEC.key: 83,
    }
    for spec, row in zip(SPECS, sweep["charts"], strict=True):
        chart = chart_for_spec(spec)
        expected = {
            "chart_key": spec.key,
            "blocker_type": {
                "barrier_size": spec.blocker_type.barrier_size,
                "odd_component_sizes": list(
                    spec.blocker_type.odd_component_sizes
                ),
                "labeled_orbit_size": (
                    spec.blocker_type.labeled_orbit_size
                ),
            },
            "projective_chow_degree": spec.projective_chow_degree,
            "chart_sha256": chart_sha256(chart),
            "variable_count": chart.variable_count,
            "equation_count": chart.equation_count,
            "coefficient_slot_count": spec.coefficient_slot_count,
            "completed_prime_count": 20,
            "verified_dimension_histogram": {
                str(expected_dimensions[spec.key]): 20
            },
        }
        if row != expected:
            raise ArtifactError(f"chart summary changed for {spec.key}")

    archive = _strict_json(directory / RECEIPTS)
    if (
        set(archive)
        != {
            "claim_boundary",
            "large_transcripts_or_matrices_in_repository",
            "ordered_probe_keys",
            "receipt_count",
            "receipts",
            "schema",
            "scratch_root",
        }
        or archive["schema"] != RECEIPT_ARCHIVE_SCHEMA
        or archive["ordered_probe_keys"]
        != [probe.key for probe in PROBES]
        or archive["receipt_count"] != len(PROBES)
        or archive["large_transcripts_or_matrices_in_repository"]
        is not False
        or archive["scratch_root"]
        != (
            r"D:\KrennScratch\counterexample_search"
            r"\n8_square_blocker_prime_sweep_v3"
        )
        or not _false_claims(archive["claim_boundary"])
        or archive["claim_boundary"] != sweep["claim_boundary"]
        or type(archive["receipts"]) is not list
        or len(archive["receipts"]) != len(PROBES)
        or type(sweep["runs"]) is not list
        or len(sweep["runs"]) != len(PROBES)
    ):
        raise ArtifactError("receipt archive schema or census changed")

    for probe, summary, archived in zip(
        PROBES,
        sweep["runs"],
        archive["receipts"],
        strict=True,
    ):
        if (
            type(archived) is not dict
            or set(archived)
            != {"bytes", "filename", "receipt", "sha256"}
            or archived["filename"]
            != f"{probe.key}.receipt.json"
            or type(archived["receipt"]) is not dict
        ):
            raise ArtifactError("archived receipt row changed")
        receipt_bytes = _canonical_json_bytes(archived["receipt"])
        if (
            archived["bytes"] != len(receipt_bytes)
            or archived["sha256"]
            != hashlib.sha256(receipt_bytes).hexdigest()
        ):
            raise ArtifactError("archived receipt hash changed")
        receipt = archived["receipt"]
        verification = receipt["verification"]
        dimension = expected_dimensions[probe.spec.key]
        variable_count = chart_for_spec(probe.spec).variable_count
        commutators = variable_count * (variable_count - 1) // 2
        if (
            receipt["schema"] != RUN_SCHEMA
            or receipt["probe"] != _probe_payload(probe)
            or not _false_claims(receipt["claim_boundary"])
            or receipt["outcome"]["status"]
            != "completed-finite-quotient"
            or receipt["outcome"][
                "backend_reported_quotient_dimension"
            ]
            != dimension
            or receipt["outcome"]["krull_dimension"] != 0
            or receipt["outcome"]["groebner_basis_size"]
            != expected_basis_sizes[probe.spec.key]
            or verification["schema"]
            != "krenn-n8-square-blocker-prime-verification-v2"
            or verification["chart_key"] != probe.spec.key
            or verification["chart_sha256"]
            != chart_sha256(chart_for_spec(probe.spec))
            or verification["field"] != f"F_{probe.characteristic}"
            or verification["representation_dimension"] != dimension
            or verification["commutator_checks"] != commutators
            or verification["equation_matrix_checks"]
            != variable_count
            or verification["cyclic_basis_checks"] != dimension
            or verification["exact_replay"] is not True
            or verification["legacy_wire_parser"][
                "transcript_schema"
            ]
            != TRANSCRIPT_SCHEMA
            or set(verification["claim_boundary"])
            != set(VERIFICATION_FALSE_KEYS)
            or any(
                value is not False
                for value in verification["claim_boundary"].values()
            )
        ):
            raise ArtifactError("receipt semantics changed")
        expected_summary = {
            "key": probe.key,
            "chart_key": probe.spec.key,
            "characteristic": probe.characteristic,
            "coefficient_audit": coefficient_audit(
                chart_for_spec(probe.spec), probe.characteristic
            ),
            "status": receipt["outcome"]["status"],
            "started_at_utc": receipt["started_at_utc"],
            "engine_subprocess_wall_seconds": (
                receipt["elapsed_seconds"]
            ),
            "timer_ticks": receipt["outcome"]["timer_ticks"],
            "groebner_basis_size": (
                receipt["outcome"]["groebner_basis_size"]
            ),
            "krull_dimension": receipt["outcome"]["krull_dimension"],
            "backend_reported_quotient_dimension": dimension,
            "verified_cyclic_module_dimension": dimension,
            "input": {
                "filename": Path(receipt["input"]["path"]).name,
                "bytes": receipt["input"]["bytes"],
                "sha256": receipt["input"]["sha256"],
            },
            "stdout": {
                "filename": receipt["stdout"]["path"],
                "bytes": receipt["stdout"]["bytes"],
                "sha256": receipt["stdout"]["sha256"],
            },
            "representation": {
                "filename": receipt["representation"]["path"],
                "bytes": receipt["representation"]["bytes"],
                "sha256": receipt["representation"]["sha256"],
            },
            "scratch_receipt": {
                "filename": archived["filename"],
                "bytes": archived["bytes"],
                "sha256": archived["sha256"],
            },
            "exact_matrix_replay": {
                "commutator_checks": commutators,
                "equation_matrix_checks": variable_count,
                "cyclic_basis_checks": dimension,
                "passed": True,
            },
        }
        if summary != expected_summary:
            raise ArtifactError("sweep summary and receipt diverged")
    return sweep


def _refresh_manifest_record(directory, filename):
    manifest_path = directory / MANIFEST
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["bundle_files"][filename] = _record(directory / filename)
    manifest_path.write_bytes(_canonical_json_bytes(manifest))


class KrennBlockerPrimeSweepArtifactTest(unittest.TestCase):
    def test_bundle_replays_exactly(self):
        sweep = _verify_bundle(BUNDLE)
        self.assertEqual(
            [chart["verified_dimension_histogram"] for chart in sweep["charts"]],
            [{"24": 20}, {"30": 20}],
        )

    def test_dimension_or_prime_reordering_fails_after_hash_refresh(self):
        for mutation in ("dimension", "prime-order"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    copied = Path(temporary) / "bundle"
                    shutil.copytree(BUNDLE, copied)
                    payload = _strict_json(copied / SWEEP)
                    if mutation == "dimension":
                        payload["runs"][0][
                            "verified_cyclic_module_dimension"
                        ] = 23
                    else:
                        payload["prime_policy"]["ordered_primes"][:2] = (
                            reversed(
                                payload["prime_policy"][
                                    "ordered_primes"
                                ][:2]
                            )
                        )
                    (copied / SWEEP).write_bytes(
                        _canonical_json_bytes(payload)
                    )
                    _refresh_manifest_record(copied, SWEEP)
                    with self.assertRaises(ArtifactError):
                        _verify_bundle(copied)

    def test_claim_promotion_or_removal_fails_after_hash_refresh(self):
        for mode in ("promote", "remove"):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary:
                    copied = Path(temporary) / "bundle"
                    shutil.copytree(BUNDLE, copied)
                    payload = _strict_json(copied / SWEEP)
                    key = "n8_nonexistence_proved"
                    if mode == "promote":
                        payload["claim_boundary"][key] = True
                    else:
                        del payload["claim_boundary"][key]
                    (copied / SWEEP).write_bytes(
                        _canonical_json_bytes(payload)
                    )
                    _refresh_manifest_record(copied, SWEEP)
                    with self.assertRaises(ArtifactError):
                        _verify_bundle(copied)

    def test_archived_receipt_forgery_fails_after_all_hashes_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            archive = _strict_json(copied / RECEIPTS)
            row = archive["receipts"][0]
            row["receipt"]["verification"]["representation_dimension"] = 23
            forged = _canonical_json_bytes(row["receipt"])
            row["bytes"] = len(forged)
            row["sha256"] = hashlib.sha256(forged).hexdigest()
            (copied / RECEIPTS).write_bytes(
                _canonical_json_bytes(archive)
            )
            _refresh_manifest_record(copied, RECEIPTS)
            with self.assertRaises(ArtifactError):
                _verify_bundle(copied)

    def test_extra_file_and_source_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            (copied / "extra.txt").write_text("extra", encoding="ascii")
            with self.assertRaisesRegex(ArtifactError, "inventory"):
                _verify_bundle(copied)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, copied)
            manifest = _strict_json(copied / MANIFEST)
            source = next(iter(manifest["source_files"]))
            manifest["source_files"][source]["sha256"] = "0" * 64
            (copied / MANIFEST).write_bytes(
                _canonical_json_bytes(manifest)
            )
            with self.assertRaises(ArtifactError):
                _verify_bundle(copied)


if __name__ == "__main__":
    unittest.main()
