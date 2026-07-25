import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.krenn_quantum_graph.blocker_quotient_reconnaissance import (
    DEFAULT_COEFFICIENT_HEIGHT,
    DEFAULT_SPECIALIZATION_SEED,
    chart_sha256,
    deterministic_k5_blocker_chart,
    singular_blocker_quotient_script,
)
from experiments.krenn_quantum_graph.blocker_quotient_runner import (
    CPUS,
    DEFAULT_SCRATCH_ROOT,
    DOCKER_IMAGE,
    DOCKER_IMAGE_ID,
    MODULAR_PROBE,
    RATIONAL_PROBE,
    RUN_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "results"
    / "krenn_quantum_graph"
    / "n8_d3_degree24_quotient_reconnaissance"
)
MANIFEST = "manifest.json"
RECONNAISSANCE = "reconnaissance.json"
README = "README.md"
RECEIPTS = {
    MODULAR_PROBE.key: f"{MODULAR_PROBE.key}.receipt.json",
    RATIONAL_PROBE.key: f"{RATIONAL_PROBE.key}.receipt.json",
}
BUNDLE_FILES = {README, RECONNAISSANCE, *RECEIPTS.values()}
ALL_FILES = BUNDLE_FILES | {MANIFEST}
SOURCE_FILES = {
    ".gitattributes",
    "experiments/krenn_quantum_graph/"
    "blocker_quotient_reconnaissance.py",
    "experiments/krenn_quantum_graph/blocker_quotient_runner.py",
    "experiments/krenn_quantum_graph/n8_k5_singular.Dockerfile",
    "experiments/krenn_quantum_graph/"
    "perfect_matching_blocker_ideals.py",
    "experiments/krenn_quantum_graph/witness.py",
}
MANIFEST_SCHEMA = (
    "krenn-n8-d3-degree24-quotient-artifact-manifest-v1"
)
RECONNAISSANCE_SCHEMA = (
    "krenn-n8-d3-degree24-quotient-reconnaissance-v1"
)
CLAIM_KEYS = {
    "all_56_labeled_k5_blockers_checked",
    "degree24_k5_covers_degree30_type",
    "distinct_points_counted",
    "eqsystem_constraints_imposed",
    "full_quotient_dimension_independently_certified",
    "generic_degree_implies_all_specializations",
    "modular_lift_proved",
    "modular_result_is_Q_or_C_proof",
    "n8_boundary_escape_proved",
    "n8_nonexistence_proved",
    "quotient_length_counts_distinct_points",
    "quotient_radical_proved",
    "quotient_reduced_proved",
    "rational_point_over_Q_proved",
    "symbolic_90_parameter_quotient_built",
    "timeout_or_miss_is_proof",
    "unit_ideal_independently_certified",
}
VERIFICATION_CLAIM_KEYS = {
    "backend_full_quotient_claim_independently_verified",
    "eqsystem_constraints_imposed",
    "generic_degree_implies_all_specializations",
    "n8_boundary_escape_proved",
    "n8_nonexistence_proved",
    "nonzero_unital_representation_proves_"
    "specialized_affine_ideal_proper_over_field",
    "proves_complex_escape_for_this_specialization",
    "symbolic_90_parameter_quotient_built",
}


class ArtifactError(RuntimeError):
    pass


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(payload):
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _strict_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    if path.stat().st_size > 1024 * 1024:
        raise ArtifactError("small bundle JSON exceeded its byte limit")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactError(f"nonfinite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ArtifactError("bundle JSON is malformed") from error
    if type(payload) is not dict:
        raise ArtifactError("bundle JSON must be an object")
    if path.read_bytes() != _canonical_json_bytes(payload):
        raise ArtifactError("bundle JSON is not canonical")
    return payload


def _refresh_manifest_record(directory, filename):
    path = directory / MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bundle_files"][filename] = {
        "bytes": (directory / filename).stat().st_size,
        "sha256": _sha256(directory / filename),
    }
    path.write_bytes(_canonical_json_bytes(payload))


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
        or path.stat().st_size != record["bytes"]
        or _sha256(path) != record["sha256"]
    ):
        raise ArtifactError(f"hash replay failed for {path.name}")


def _verify_bundle(directory):
    directory = directory.resolve()
    if {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    } != ALL_FILES or any(
        path.is_symlink() or not path.is_file()
        for path in directory.iterdir()
    ):
        raise ArtifactError("bundle inventory changed")
    manifest = _strict_json(directory / MANIFEST)
    if set(manifest) != {
        "bundle_files",
        "claim_boundary",
        "schema",
        "source_files",
    } or manifest["schema"] != MANIFEST_SCHEMA:
        raise ArtifactError("manifest schema changed")
    if (
        type(manifest["bundle_files"]) is not dict
        or set(manifest["bundle_files"]) != BUNDLE_FILES
    ):
        raise ArtifactError("manifest bundle inventory changed")
    for filename, record in manifest["bundle_files"].items():
        _verify_record(directory / filename, record)
    if (
        type(manifest["source_files"]) is not dict
        or set(manifest["source_files"]) != SOURCE_FILES
    ):
        raise ArtifactError("manifest source inventory changed")
    for filename, record in manifest["source_files"].items():
        path = ROOT / filename
        _verify_record(path, record)

    reconnaissance = _strict_json(directory / RECONNAISSANCE)
    if (
        set(reconnaissance)
        != {
            "chart",
            "claim_boundary",
            "exact_conclusions",
            "pilots",
            "schema",
            "scratch",
            "singular_engine",
            "status",
        }
        or reconnaissance["schema"] != RECONNAISSANCE_SCHEMA
        or reconnaissance["status"]
        != "fixed-specialization-cyclic-quotient-pipeline-validated"
    ):
        raise ArtifactError("reconnaissance schema or status changed")
    claims = reconnaissance["claim_boundary"]
    if (
        type(claims) is not dict
        or set(claims) != CLAIM_KEYS
        or claims != manifest["claim_boundary"]
        or any(value is not False for value in claims.values())
    ):
        raise ArtifactError("claim boundary was escalated")
    chart = deterministic_k5_blocker_chart()
    chart_record = reconnaissance["chart"]
    expected_chart = {
        "blocker_type": {
            "barrier_size": 3,
            "odd_component_sizes": [1, 1, 1, 1, 1],
            "outside_graph": "K5",
        },
        "chart_sha256": chart_sha256(chart),
        "coefficient_height": 127,
        "coefficient_slots": 90,
        "coefficient_values_distinct": 90,
        "colors": 3,
        "equation_count": 10,
        "every_coefficient_nonzero_mod_31": True,
        "every_coefficient_nonzero_over_Q": True,
        "normalized_color": 0,
        "specialization_seed": 80320260725,
        "variable_count": 10,
        "vertices": 8,
    }
    if (
        DEFAULT_SPECIALIZATION_SEED != 80320260725
        or DEFAULT_COEFFICIENT_HEIGHT != 127
        or chart_record != expected_chart
    ):
        raise ArtifactError("deterministic chart replay changed")
    coefficients = tuple(
        coefficient
        for equation in chart.equations
        for _monomial, coefficient in equation
    )
    if (
        len(coefficients) != 90
        or len(set(coefficients)) != 90
        or not all(coefficients)
        or not all(value.numerator % 31 for value in coefficients)
    ):
        raise ArtifactError("coefficient audit changed")
    dockerfile = (
        ROOT
        / "experiments"
        / "krenn_quantum_graph"
        / "n8_k5_singular.Dockerfile"
    )
    expected_engine = {
        "cpus": 2,
        "dockerfile_sha256": _sha256(dockerfile),
        "image": DOCKER_IMAGE,
        "image_id": DOCKER_IMAGE_ID,
        "maximum_workers": 1,
        "network": "none",
        "version": "4.3.2",
    }
    if reconnaissance["singular_engine"] != expected_engine:
        raise ArtifactError("engine provenance changed")
    if reconnaissance["scratch"] != {
        "large_matrices_committed_to_repository": False,
        "required_for_full_matrix_replay": True,
        "root": str(DEFAULT_SCRATCH_ROOT),
    }:
        raise ArtifactError("scratch boundary changed")

    if (
        type(reconnaissance["pilots"]) is not list
        or len(reconnaissance["pilots"]) != len(RECEIPTS)
        or any(type(row) is not dict for row in reconnaissance["pilots"])
    ):
        raise ArtifactError("pilot rows changed type")
    pilots = {row.get("key"): row for row in reconnaissance["pilots"]}
    if (
        len(pilots) != len(reconnaissance["pilots"])
        or set(pilots) != set(RECEIPTS)
    ):
        raise ArtifactError("pilot inventory changed")
    expected_run_data = {
        MODULAR_PROBE.key: {
            "engine_subprocess_wall_seconds": 0.324389,
            "basis_size": 56,
            "claim": "proper fixed specialized affine ideal over F_31 only",
            "dimension": 22,
            "field": "F_31",
            "representation_bytes": 62180,
            "representation_sha256": (
                "7cc7f0aa7ef6fbac20e1b4bc76121de8ef44b4713e6f773d"
                "872ee8d99fe0bb01"
            ),
            "receipt_sha256": (
                "0e414ea5358afbc56d7d50153b286911f3b104bb5a00cf62f"
                "3868297d99b0b76"
            ),
            "started_at_utc": "2026-07-25T18:26:40.468441+00:00",
            "stdout_bytes": 31943,
            "stdout_sha256": (
                "19cc8a5eed63f3c4ec62795ff64cc6e308ba79bc4f3c6a21a"
                "90c5fe390462801"
            ),
            "timer_ticks": 0,
        },
        RATIONAL_PROBE.key: {
            "engine_subprocess_wall_seconds": 73.255729,
            "basis_size": 58,
            "claim": (
                "proper fixed specialized affine ideal over Q; "
                "point over Qbar and C"
            ),
            "dimension": 24,
            "field": "Q",
            "representation_bytes": 5601312,
            "representation_sha256": (
                "a76032221e135bf72f2dc803fa005d7db2024e0936ef909f68"
                "171eb242f6dab9"
            ),
            "receipt_sha256": (
                "00885e3207460593ec7eef26867bb1e562864db058df045a15"
                "e810131f52c8ca"
            ),
            "started_at_utc": "2026-07-25T18:26:53.003398+00:00",
            "stdout_bytes": 5374192,
            "stdout_sha256": (
                "f14f090cd795223d562c9ecb1dcb832cb58560cfb3cd851da"
                "193153548fecbd8"
            ),
            "timer_ticks": 70,
        },
    }
    for probe in (MODULAR_PROBE, RATIONAL_PROBE):
        expected = expected_run_data[probe.key]
        pilot = pilots[probe.key]
        script = singular_blocker_quotient_script(
            chart,
            characteristic=probe.characteristic,
            algorithm=probe.algorithm,
        ).encode("utf-8")
        input_sha256 = hashlib.sha256(script).hexdigest()
        expected_pilot = {
            "algorithm": probe.algorithm,
            "engine_subprocess_wall_seconds": expected[
                "engine_subprocess_wall_seconds"
            ],
            "backend_quotient_dimension": expected["dimension"],
            "claim": expected["claim"],
            "engine_budget_seconds": probe.engine_budget_seconds,
            "exact_matrix_replay": {
                "commutator_checks": 45,
                "cyclic_basis_checks": expected["dimension"],
                "equation_matrix_checks": 10,
                "passed": True,
            },
            "field": expected["field"],
            "groebner_basis_size": expected["basis_size"],
            "input_sha256": input_sha256,
            "key": probe.key,
            "krull_dimension": 0,
            "memory_gib": probe.memory_gib,
            "receipt_file": RECEIPTS[probe.key],
            "representation_bytes": expected["representation_bytes"],
            "representation_sha256": expected[
                "representation_sha256"
            ],
            "scratch_receipt_sha256": expected["receipt_sha256"],
            "stdout_bytes": expected["stdout_bytes"],
            "stdout_sha256": expected["stdout_sha256"],
            "timer_ticks": expected["timer_ticks"],
        }
        if pilot != expected_pilot:
            raise ArtifactError("pilot semantic replay changed")
        receipt_path = directory / RECEIPTS[probe.key]
        receipt = _strict_json(receipt_path)
        mount = f"{DEFAULT_SCRATCH_ROOT / 'inputs'}:/work:ro"
        expected_argv = [
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
            f"/work/{probe.key}.sing",
        ]
        verification_claims = {
            key: False for key in VERIFICATION_CLAIM_KEYS
        }
        verification_claims[
            "nonzero_unital_representation_proves_"
            "specialized_affine_ideal_proper_over_field"
        ] = True
        verification_claims[
            "proves_complex_escape_for_this_specialization"
        ] = probe.characteristic == 0
        expected_receipt = {
            "claim_boundary": claims,
            "elapsed_seconds": expected[
                "engine_subprocess_wall_seconds"
            ],
            "engine": {
                "argv": expected_argv,
                "image": DOCKER_IMAGE,
                "image_id": DOCKER_IMAGE_ID,
                "network": "none",
            },
            "input": {
                "bytes": len(script),
                "path": str(
                    DEFAULT_SCRATCH_ROOT
                    / "inputs"
                    / f"{probe.key}.sing"
                ),
                "sha256": input_sha256,
            },
            "outcome": {
                "backend_quotient_dimension": expected["dimension"],
                "completion_marker_observed": True,
                "groebner_basis_size": expected["basis_size"],
                "krull_dimension": 0,
                "parse_marker_observed": True,
                "return_code": 0,
                "status": "completed-finite-quotient",
                "timer_ticks": expected["timer_ticks"],
                "unit_ideal": False,
            },
            "probe": {
                "algorithm": probe.algorithm,
                "characteristic": probe.characteristic,
                "chart_sha256": chart_sha256(chart),
                "cpus": CPUS,
                "engine_budget_seconds": probe.engine_budget_seconds,
                "key": probe.key,
                "memory_gib": probe.memory_gib,
                "specialization_seed": DEFAULT_SPECIALIZATION_SEED,
            },
            "representation": {
                "bytes": expected["representation_bytes"],
                "path": f"{probe.key}.representation.json",
                "sha256": expected["representation_sha256"],
            },
            "schema": RUN_SCHEMA,
            "started_at_utc": expected["started_at_utc"],
            "stderr": {
                "bytes": 0,
                "path": f"{probe.key}.stderr.txt",
                "sha256": (
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934"
                    "ca495991b7852b855"
                ),
            },
            "stdout": {
                "bytes": expected["stdout_bytes"],
                "path": f"{probe.key}.stdout.txt",
                "sha256": expected["stdout_sha256"],
            },
            "verification": {
                "chart_sha256": chart_sha256(chart),
                "claim_boundary": verification_claims,
                "commutator_checks": 45,
                "cyclic_basis_checks": expected["dimension"],
                "equation_matrix_checks": 10,
                "exact_replay": True,
                "field": expected["field"],
                "representation_dimension": expected["dimension"],
                "schema": (
                    "krenn-n8-k5-blocker-quotient-"
                    "reconnaissance-v1"
                ),
            },
        }
        if (
            _sha256(receipt_path) != expected["receipt_sha256"]
            or set(receipt.get("claim_boundary", {})) != CLAIM_KEYS
            or set(
                receipt.get("verification", {}).get(
                    "claim_boundary", {}
                )
            )
            != VERIFICATION_CLAIM_KEYS
            or receipt != expected_receipt
            or datetime.fromisoformat(
                receipt["started_at_utc"]
            ).utcoffset()
            is None
        ):
            raise ArtifactError("retained receipt replay changed")
    if reconnaissance["exact_conclusions"] != {
        "F31bar_point_for_fixed_specialization": True,
        "Qbar_and_complex_point_for_fixed_specialization": True,
        "f31_nonzero_unital_cyclic_quotient_dimension": 22,
        "q_nonzero_unital_cyclic_quotient_dimension": 24,
    }:
        raise ArtifactError("exact conclusion ledger changed")
    return reconnaissance


class KrennBlockerQuotientArtifactTest(unittest.TestCase):
    def copied_bundle(self, name):
        destination = Path(self.temporary.name) / name
        shutil.copytree(BUNDLE, destination)
        return destination

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary.cleanup()

    def test_committed_bundle_replays_exactly(self):
        reconnaissance = _verify_bundle(BUNDLE)
        self.assertEqual(
            reconnaissance["pilots"][0]["backend_quotient_dimension"],
            22,
        )
        self.assertEqual(
            reconnaissance["pilots"][1]["backend_quotient_dimension"],
            24,
        )

    def test_raw_corruption_fails_hash_replay(self):
        directory = self.copied_bundle("raw-corruption")
        path = directory / README
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ArtifactError, "hash replay"):
            _verify_bundle(directory)

    def test_refreshed_hash_cannot_escalate_a_claim(self):
        directory = self.copied_bundle("claim-escalation")
        path = directory / RECONNAISSANCE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["claim_boundary"]["n8_boundary_escape_proved"] = True
        path.write_bytes(_canonical_json_bytes(payload))
        _refresh_manifest_record(directory, RECONNAISSANCE)
        with self.assertRaisesRegex(ArtifactError, "claim boundary"):
            _verify_bundle(directory)

    def test_refreshed_hash_cannot_change_receipt_field(self):
        directory = self.copied_bundle("receipt-corruption")
        filename = RECEIPTS[RATIONAL_PROBE.key]
        path = directory / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["probe"]["characteristic"] = 31
        path.write_bytes(_canonical_json_bytes(payload))
        _refresh_manifest_record(directory, filename)
        with self.assertRaisesRegex(ArtifactError, "receipt replay"):
            _verify_bundle(directory)

    def test_refreshed_hash_cannot_change_engine_metadata(self):
        directory = self.copied_bundle("engine-corruption")
        filename = RECEIPTS[RATIONAL_PROBE.key]
        path = directory / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["engine"]["network"] = "bridge"
        path.write_bytes(_canonical_json_bytes(payload))
        _refresh_manifest_record(directory, filename)
        with self.assertRaisesRegex(ArtifactError, "receipt replay"):
            _verify_bundle(directory)

    def test_removed_claim_key_fails_even_if_all_hashes_refresh(self):
        directory = self.copied_bundle("removed-claim")
        key = "n8_nonexistence_proved"
        reconnaissance_path = directory / RECONNAISSANCE
        reconnaissance = json.loads(
            reconnaissance_path.read_text(encoding="utf-8")
        )
        del reconnaissance["claim_boundary"][key]
        reconnaissance_path.write_bytes(
            _canonical_json_bytes(reconnaissance)
        )
        _refresh_manifest_record(directory, RECONNAISSANCE)
        manifest_path = directory / MANIFEST
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        del manifest["claim_boundary"][key]
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        for filename in RECEIPTS.values():
            path = directory / filename
            receipt = json.loads(path.read_text(encoding="utf-8"))
            del receipt["claim_boundary"][key]
            path.write_bytes(_canonical_json_bytes(receipt))
            _refresh_manifest_record(directory, filename)
        with self.assertRaisesRegex(ArtifactError, "claim boundary"):
            _verify_bundle(directory)

    def test_duplicate_pilot_row_fails_after_hash_refresh(self):
        directory = self.copied_bundle("duplicate-pilot")
        path = directory / RECONNAISSANCE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pilots"].insert(
            0, json.loads(json.dumps(payload["pilots"][0]))
        )
        path.write_bytes(_canonical_json_bytes(payload))
        _refresh_manifest_record(directory, RECONNAISSANCE)
        with self.assertRaisesRegex(ArtifactError, "pilot rows"):
            _verify_bundle(directory)

    def test_missing_and_extra_files_fail_inventory(self):
        missing = self.copied_bundle("missing")
        (missing / README).unlink()
        with self.assertRaisesRegex(ArtifactError, "inventory"):
            _verify_bundle(missing)
        extra = self.copied_bundle("extra")
        (extra / "extra.txt").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "inventory"):
            _verify_bundle(extra)


if __name__ == "__main__":
    unittest.main()
