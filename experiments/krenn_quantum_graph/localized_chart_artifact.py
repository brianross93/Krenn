"""Round-trippable artifacts for the exact localized-chart campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    natural_derivative_atlas_audit,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    localized_chart_cover_audit,
    localized_chart_germ_audit,
    normalized_seed_chart,
    strict_json_equal,
    verify_localized_chart_cover_audit,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    BoundedMacaulayResult,
    EXPECTED_STABILIZER_SIZES,
    bounded_koszul_orbit_relations,
    modular_bounded_macaulay_preflight,
)
from experiments.krenn_quantum_graph.localized_chart_monomial_atlas import (
    natural_repair_monomial_atlas_audit,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_SCHEMA = "krenn-n6-d3-localized-chart-campaign-v1"
MANIFEST_SCHEMA = "krenn-n6-d3-localized-chart-manifest-v1"
DEFAULT_RESULT_DIRECTORY = (
    ROOT
    / "results"
    / "krenn_quantum_graph"
    / "n6_d3_localized_chart_ideals"
)
DATA_FILES = (
    "bounded_degree_six.json",
    "chart_cover.json",
    "germ_separation.json",
    "natural_derivative_atlas.json",
    "natural_repair_monomial_atlas.json",
    "preflight_ledger.json",
)
SOURCE_FILES = (
    "experiments/__init__.py",
    "experiments/krenn_quantum_graph/__init__.py",
    "experiments/krenn_quantum_graph/localized_chart_artifact.py",
    "experiments/krenn_quantum_graph/localized_chart_derivative.py",
    "experiments/krenn_quantum_graph/localized_chart_ideals.py",
    "experiments/krenn_quantum_graph/localized_chart_independent.py",
    "experiments/krenn_quantum_graph/localized_chart_macaulay.py",
    "experiments/krenn_quantum_graph/localized_chart_monomial_atlas.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "experiments/krenn_quantum_graph/ternary_seed_orbits.py",
    "experiments/krenn_quantum_graph/transport.py",
    "experiments/krenn_quantum_graph/witness.py",
)

EXPECTED_DEGREE_SIX_ROWS = (
    (82_869, 670, 2_504, 5_706, 15, 15, 655, 655, 656),
    (73_281, 6_510, 36_880, 76_475, 74, 74, 6_436, 6_436, 6_437),
    (68_487, 6_049, 45_931, 88_017, 24, 24, 6_025, 6_025, 6_026),
    (66_411, 1_947, 11_624, 22_868, 10, 10, 1_937, 1_937, 1_938),
    (
        66_538, 17_234, 133_117, 253_729, 72, 72,
        17_162, 17_162, 17_163,
    ),
    (
        63_103, 16_036, 128_168, 238_416, 35, 35,
        16_001, 16_001, 16_002,
    ),
    (61_154, 5_318, 42_683, 77_840, 8, 8, 5_310, 5_310, 5_311),
    (62_640, 1_903, 14_890, 27_164, 3, 3, 1_900, 1_900, 1_901),
)
EXPECTED_DEGREE_SIX_RELATION_HASHES = (
    "518572c45168ebcd9a2e884205837775aaba752de7a92b03f6e39cddf7ae995a",
    "0645fc13a9edc79344c315e6a6956b985a8b134a23220ca076bbbaa96489f0b8",
    "ac17a3d6339c77ed7d1f5f13374209fc8695bfbd895a77ad7cd56826182bd04c",
    "e679c46ea99e91ec18989a64cc0bce870519c01f7717d60fc1abd8f5112e85d8",
    "80fd7f29657a26691382285efd0491148c2d7f48a1cf8c2944a90a5fed9af275",
    "346d1ee7a190634c8b1b2eb2e0dadd2062c54f22a1ba89e1da67c630306e44fe",
    "e600eedafeb506f3b70c51aadb59a4009392d9969da4178103fcdf610b66434e",
    "e66c7126b99c2e2c107b5173337b35b32e7e83aedc6bea9935f45336d7b054e8",
)


class KrennLocalizedArtifactError(RuntimeError):
    """A localized-chart artifact failed strict replay."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_strict_json(path: Path) -> dict:
    """Load one unambiguous strict-JSON object from disk."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise KrennLocalizedArtifactError(
            f"{path.name} is not unambiguous strict JSON"
        ) from error
    if type(payload) is not dict:
        raise KrennLocalizedArtifactError(
            f"{path.name} must contain one JSON object"
        )
    return payload


def _relation_fingerprint(
    relations: tuple[tuple[tuple[int, int], ...], ...],
) -> str:
    digest = hashlib.sha256()
    for relation in relations:
        digest.update(
            ";".join(
                f"{column}:{coefficient}"
                for column, coefficient in relation
            ).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _expected_row_tuple(result: Mapping) -> tuple[int, ...]:
    return (
        int(result["raw_torus_pairs"]),
        int(result["invariant_columns"]),
        int(result["invariant_rows"]),
        int(result["nonzero_entries"]),
        int(result["exact_koszul_relations"]),
        int(result["independent_koszul_relation_rank_mod_p"]),
        int(result["source_rank_upper_bound_over_q"]),
        int(result["source_rank_mod_p"]),
        int(result["augmented_rank_mod_p"]),
    )


def _bounded_degree_six_envelope(rows: list[dict]) -> dict:
    return {
        "schema": ARTIFACT_SCHEMA,
        "method": {
            "total_certificate_degree_bound": 6,
            "residual_torus_weight_projection": True,
            "ordered_seed_stabilizer_reynolds_average": True,
            "coefficient_field_for_rank_minors": "F_1009",
            "prime_coprime_to_every_stabilizer_order": True,
            "exact_integer_koszul_relations": True,
        },
        "charts": rows,
        "exact_conclusion": (
            "over Q, none of the eight normalized chart ideals has a "
            "Nullstellensatz identity of total degree at most six"
        ),
        "proof_logic": (
            "independent exact Koszul relations bound each rational source "
            "rank above; a nonzero modular source minor attains that bound; "
            "a larger augmented modular minor puts 1 outside the Q-span"
        ),
        "claim_boundary": {
            "full_chart_unit_ideals_decided": 0,
            "proper_chart_ideals_decided": 0,
            "finite_affine_GHZ_membership_status": "undecided",
            "bounded_degree_miss_is_global_proof": False,
            "finite_field_result_alone_is_proof": False,
        },
    }


def retained_bounded_degree_six_audit() -> dict:
    """Rebuild the exact expected payload from retained rank data."""

    rows = []
    for orbit_index, retained in enumerate(EXPECTED_DEGREE_SIX_ROWS):
        (
            raw_pairs,
            columns,
            rows_count,
            nonzeros,
            relation_count,
            relation_rank,
            rank_upper,
            source_rank,
            augmented_rank,
        ) = retained
        result = BoundedMacaulayResult(
            orbit_index=orbit_index,
            total_degree_bound=6,
            prime=1_009,
            stabilizer_order=EXPECTED_STABILIZER_SIZES[orbit_index],
            raw_torus_pairs=raw_pairs,
            invariant_columns=columns,
            invariant_rows=rows_count,
            nonzero_entries=nonzeros,
            exact_koszul_relations=relation_count,
            independent_koszul_relation_rank_mod_p=relation_rank,
            source_rank_upper_bound_over_q=rank_upper,
            source_rank_mod_p=source_rank,
            augmented_rank_mod_p=augmented_rank,
            target_in_span_mod_p=False,
            bounded_nonmembership_over_q_certified=True,
        ).to_dict()
        if result["seed"] != list(
            normalized_seed_chart(orbit_index).seed
        ):
            raise KrennLocalizedArtifactError(
                "a retained degree-six seed label changed"
            )
        result["koszul_relation_sha256"] = (
            EXPECTED_DEGREE_SIX_RELATION_HASHES[orbit_index]
        )
        rows.append(result)
    return _bounded_degree_six_envelope(rows)


def bounded_degree_six_audit() -> dict:
    """Recompute all eight exact bounded-degree-six rank certificates."""

    rows = []
    for orbit_index in range(8):
        result = modular_bounded_macaulay_preflight(
            orbit_index, 6, prime=1_009
        ).to_dict()
        relations = bounded_koszul_orbit_relations(orbit_index, 6)
        result["koszul_relation_sha256"] = _relation_fingerprint(relations)
        if (
            _expected_row_tuple(result)
            != EXPECTED_DEGREE_SIX_ROWS[orbit_index]
            or not result["bounded_nonmembership_over_q_certified"]
            or result["target_in_span_mod_p"]
        ):
            raise KrennLocalizedArtifactError(
                "a degree-six bounded certificate regression changed"
            )
        rows.append(result)
    payload = _bounded_degree_six_envelope(rows)
    if not strict_json_equal(
        payload, retained_bounded_degree_six_audit()
    ):
        raise KrennLocalizedArtifactError(
            "recomputed degree-six payload differs from retained data"
        )
    return payload


def preflight_ledger() -> dict:
    """Record deterministic budgets and the bounded CAS outcomes."""

    return {
        "schema": ARTIFACT_SCHEMA,
        "host_policy": {
            "cpu_workers": 1,
            "maximum_allowed_workers": 16,
            "network": "disabled for every Docker run",
            "scratch_root": "D:\\KrennScratch\\localized_chart_ideals",
            "random_seeds": [],
        },
        "exact_macaulay": {
            "command": (
                "python -B -m experiments.krenn_quantum_graph."
                "localized_chart_artifact generate --output "
                "results/krenn_quantum_graph/"
                "n6_d3_localized_chart_ideals"
            ),
            "charts": 8,
            "total_degree_bound": 6,
            "prime": 1009,
            "deterministic": True,
            "result": (
                "exact bounded nonmembership through total degree six "
                "on all eight charts"
            ),
        },
        "singular_engine": {
            "docker_image": "hodgepodge-singular:ubuntu24.04",
            "image_id": (
                "sha256:6613ac51738965fafd2ebb2839a02452811ac4e3"
                "5416cfbbda0485ed1995e75"
            ),
            "version": "4.3.2",
            "coefficient_fields": ["F_31", "Q"],
            "runs": [
                {
                    "ideal": "natural unshifted chart",
                    "algorithm": "slimgb",
                    "characteristic": 31,
                    "budget_seconds": 60,
                    "cpus": 2,
                    "memory_gib": 8,
                    "command": (
                        "docker run --rm --network none --cpus 2 "
                        "--memory 8g --entrypoint timeout -v "
                        '"D:\\KrennScratch\\localized_chart_ideals\\'
                        'unshifted_slimgb:/work:ro" '
                        "hodgepodge-singular:ubuntu24.04 60s "
                        "Singular /work/chart6_char31_slimgb.sing"
                    ),
                    "input_script": {
                        "path": (
                            "D:\\KrennScratch\\localized_chart_ideals\\"
                            "unshifted_slimgb\\"
                            "chart6_char31_slimgb.sing"
                        ),
                        "bytes": 128_719,
                        "sha256": (
                            "0fdc5574372dd447100b64829ef584a258069d894"
                            "2bd895f8677e0c9b20c09b1"
                        ),
                    },
                    "marker_receipt": {
                        "parse_marker": "KRENN_CHART_PARSE_OK",
                        "parse_marker_observed_live": True,
                        "completion_marker":
                            "KRENN_CHART_GROEBNER_DONE",
                        "completion_marker_observed_live": False,
                        "stdout_bytes_retained": False,
                    },
                    "status": "timeout-after-parse",
                },
                {
                    "ideal": "derivative elimination ambient 11",
                    "algorithm": "std",
                    "characteristic": 31,
                    "budget_seconds": 60,
                    "cpus": 2,
                    "memory_gib": 8,
                    "command": (
                        "docker run --rm --network none --cpus 2 "
                        "--memory 8g --entrypoint timeout -v "
                        '"D:\\KrennScratch\\localized_chart_ideals\\'
                        'derivative_atlas:/work:ro" '
                        "hodgepodge-singular:ubuntu24.04 60s Singular "
                        "/work/natural_derivative_11_p31_std.sing"
                    ),
                    "input_script": {
                        "path": (
                            "D:\\KrennScratch\\localized_chart_ideals\\"
                            "derivative_atlas\\"
                            "natural_derivative_11_p31_std.sing"
                        ),
                        "bytes": 200_201,
                        "sha256": (
                            "6c16496e96db48d427aa5e1b3e99f122e480a791"
                            "8edb3ee04f9f95ed5ee1bbce"
                        ),
                    },
                    "marker_receipt": {
                        "parse_marker": "KRENN_DERIVATIVE_PARSE_OK",
                        "parse_marker_observed_live": True,
                        "completion_marker":
                            "KRENN_DERIVATIVE_GROEBNER_DONE",
                        "completion_marker_observed_live": False,
                        "stdout_bytes_retained": False,
                    },
                    "status": "timeout-after-parse",
                },
                {
                    "ideal": "derivative elimination ambient 29",
                    "algorithm": "std",
                    "characteristic": 31,
                    "budget_seconds": 30,
                    "cpus": 2,
                    "memory_gib": 8,
                    "command": (
                        "docker run --rm --network none --cpus 2 "
                        "--memory 8g --entrypoint timeout -v "
                        '"D:\\KrennScratch\\localized_chart_ideals\\'
                        'derivative_atlas:/work:ro" '
                        "hodgepodge-singular:ubuntu24.04 30s Singular "
                        "/work/natural_derivative_29_p31_std.sing"
                    ),
                    "input_script": {
                        "path": (
                            "D:\\KrennScratch\\localized_chart_ideals\\"
                            "derivative_atlas\\"
                            "natural_derivative_29_p31_std.sing"
                        ),
                        "bytes": 199_414,
                        "sha256": (
                            "ea22026f710c7de477f495923ced9887a8b10c1a6"
                            "3efe6fc0c71153a0141f7bd"
                        ),
                    },
                    "marker_receipt": {
                        "parse_marker": "KRENN_DERIVATIVE_PARSE_OK",
                        "parse_marker_observed_live": True,
                        "completion_marker":
                            "KRENN_DERIVATIVE_GROEBNER_DONE",
                        "completion_marker_observed_live": False,
                        "stdout_bytes_retained": False,
                    },
                    "status": "timeout-after-parse",
                },
                {
                    "ideal": "repair monomial 11,55,133",
                    "algorithm": "std",
                    "characteristic": 31,
                    "budget_seconds": 60,
                    "cpus": 2,
                    "memory_gib": 8,
                    "command": (
                        "docker run --rm --network none --cpus 2 "
                        "--memory 8g --entrypoint timeout -v "
                        '"D:\\KrennScratch\\localized_chart_ideals\\'
                        'repair_monomial_atlas:/work:ro" '
                        "hodgepodge-singular:ubuntu24.04 60s Singular "
                        "/work/repair_11_55_133_p31_std.sing"
                    ),
                    "input_script": {
                        "path": (
                            "D:\\KrennScratch\\localized_chart_ideals\\"
                            "repair_monomial_atlas\\"
                            "repair_11_55_133_p31_std.sing"
                        ),
                        "bytes": 126_118,
                        "sha256": (
                            "d59f13e043d04861e7e2e2a7ab46d5df5dace5f"
                            "eafba55aee986090e6d99c83e"
                        ),
                    },
                    "marker_receipt": {
                        "parse_marker": "KRENN_REPAIR_CHART_PARSE_OK",
                        "parse_marker_observed_live": True,
                        "completion_marker":
                            "KRENN_REPAIR_CHART_GROEBNER_DONE",
                        "completion_marker_observed_live": False,
                        "stdout_bytes_retained": False,
                    },
                    "status": "timeout-after-parse",
                },
                {
                    "ideal": "repair monomial 29,55,106",
                    "algorithm": "std",
                    "characteristic": 31,
                    "budget_seconds": 30,
                    "cpus": 2,
                    "memory_gib": 8,
                    "command": (
                        "docker run --rm --network none --cpus 2 "
                        "--memory 8g --entrypoint timeout -v "
                        '"D:\\KrennScratch\\localized_chart_ideals\\'
                        'repair_monomial_atlas:/work:ro" '
                        "hodgepodge-singular:ubuntu24.04 30s Singular "
                        "/work/repair_29_55_106_p31_std.sing"
                    ),
                    "input_script": {
                        "path": (
                            "D:\\KrennScratch\\localized_chart_ideals\\"
                            "repair_monomial_atlas\\"
                            "repair_29_55_106_p31_std.sing"
                        ),
                        "bytes": 126_115,
                        "sha256": (
                            "e4c47a3486cde8a06376d9e0f21fb83082290425"
                            "1d2e721f839d1dfdc3ebb3ac"
                        ),
                    },
                    "marker_receipt": {
                        "parse_marker": "KRENN_REPAIR_CHART_PARSE_OK",
                        "parse_marker_observed_live": True,
                        "completion_marker":
                            "KRENN_REPAIR_CHART_GROEBNER_DONE",
                        "completion_marker_observed_live": False,
                        "stdout_bytes_retained": False,
                    },
                    "status": "timeout-after-parse",
                },
                {
                    "ideal": "12-variable natural defect critical ideal",
                    "algorithm": "modStd exact final verification plus lift",
                    "characteristic": 0,
                    "budget_seconds": 120,
                    "cpus": 2,
                    "memory_gib": 4,
                    "command": (
                        "docker run --rm --network none --cpus 2 "
                        "--memory 4g --entrypoint timeout -v "
                        '"C:\\tmp:/work:ro" '
                        "hodgepodge-singular:ubuntu24.04 120s Singular "
                        "/work/krenn_natural_core_smooth.sing"
                    ),
                    "input_script": {
                        "path": "C:\\tmp\\krenn_natural_core_smooth.sing",
                        "bytes": 856,
                        "sha256": (
                            "7e609684df16f02f6d09c929f7d8abb061f47a54"
                            "440972b9313c503e1776b274"
                        ),
                    },
                    "selected_stdout": [
                        "unit=1",
                        "certificate_replay=1",
                        "certificate_rows=13",
                    ],
                    "native_exact_certificate_replayed": True,
                    "status": "unit-certificate-extracted-and-replayed",
                },
            ],
        },
        "claim_boundary": {
            "a_timeout_is_a_decision": False,
            "a_modular_basis_is_an_exact_chart_decision": False,
            "a_bounded_certificate_miss_is_a_global_decision": False,
            "unretained_timeout_stdout_is_a_certificate": False,
            "smooth_defect_hypersurface_decides_full_chart": False,
        },
    }


def _write_json(path: Path, payload: Mapping) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise KrennLocalizedArtifactError(
            f"refusing to overwrite non-regular artifact target {path.name}"
        )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _expected_manifest(output_directory: Path) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "parameters": {"n": 6, "d": 3},
        "status": "exact-localized-chart-membership-undecided",
        "files": [
            {"path": name, "sha256": _sha256(output_directory / name)}
            for name in DATA_FILES
        ],
        "source_ledger": [
            {"path": name, "sha256": _sha256(ROOT / name)}
            for name in SOURCE_FILES
        ],
        "headline": {
            "ordered_seed_charts": 3_375,
            "symmetry_representatives": 8,
            "exact_bounded_certificate_degree_excluded": 6,
            "finite_affine_GHZ_membership_status": "undecided",
        },
        "claim_boundary": {
            "counterexample_found": False,
            "global_nonexistence_proved": False,
            "border_membership_changed": False,
        },
    }


def write_localized_chart_bundle(
    output_directory: Path = DEFAULT_RESULT_DIRECTORY,
) -> dict:
    """Generate the source-controlled exact chart campaign bundle."""

    output_directory = Path(output_directory)
    expected_inventory = set(DATA_FILES).union({"manifest.json"})
    if output_directory.exists():
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise KrennLocalizedArtifactError(
                "artifact output path must be a real directory"
            )
    else:
        output_directory.mkdir(parents=True)
    existing = tuple(output_directory.iterdir())
    if (
        any(path.name not in expected_inventory for path in existing)
        or any(not path.is_file() or path.is_symlink() for path in existing)
    ):
        raise KrennLocalizedArtifactError(
            "artifact output directory contains an undeclared entry"
        )
    payloads = {
        "bounded_degree_six.json": bounded_degree_six_audit(),
        "chart_cover.json": localized_chart_cover_audit(),
        "germ_separation.json": localized_chart_germ_audit(),
        "natural_derivative_atlas.json":
            natural_derivative_atlas_audit(),
        "natural_repair_monomial_atlas.json":
            natural_repair_monomial_atlas_audit(),
        "preflight_ledger.json": preflight_ledger(),
    }
    for name, payload in payloads.items():
        _write_json(output_directory / name, payload)
    manifest = _expected_manifest(output_directory)
    _write_json(output_directory / "manifest.json", manifest)
    return manifest


def verify_localized_chart_bundle(
    output_directory: Path = DEFAULT_RESULT_DIRECTORY,
    *,
    full_degree_six_replay: bool = False,
) -> dict:
    """Verify inventory, hashes, exact audits, and bounded claims."""

    output_directory = Path(output_directory)
    expected_inventory = set(DATA_FILES).union({"manifest.json"})
    entries = tuple(output_directory.iterdir())
    actual_inventory = {path.name for path in entries}
    if (
        actual_inventory != expected_inventory
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise KrennLocalizedArtifactError(
            "localized chart bundle inventory changed"
        )
    manifest = _load_strict_json(output_directory / "manifest.json")
    expected_manifest = _expected_manifest(output_directory)
    if not strict_json_equal(manifest, expected_manifest):
        raise KrennLocalizedArtifactError(
            "localized chart manifest failed full exact replay"
        )
    cover = _load_strict_json(output_directory / "chart_cover.json")
    verify_localized_chart_cover_audit(cover)
    if not strict_json_equal(
        _load_strict_json(output_directory / "germ_separation.json"),
        localized_chart_germ_audit(),
    ):
        raise KrennLocalizedArtifactError(
            "germ separation audit failed exact replay"
        )
    if not strict_json_equal(
        _load_strict_json(
            output_directory / "natural_derivative_atlas.json"
        ),
        natural_derivative_atlas_audit(),
    ):
        raise KrennLocalizedArtifactError(
            "natural derivative atlas failed exact replay"
        )
    if not strict_json_equal(
        _load_strict_json(
            output_directory / "natural_repair_monomial_atlas.json"
        ),
        natural_repair_monomial_atlas_audit(),
    ):
        raise KrennLocalizedArtifactError(
            "natural repair-monomial atlas failed exact replay"
        )
    if not strict_json_equal(
        _load_strict_json(output_directory / "preflight_ledger.json"),
        preflight_ledger(),
    ):
        raise KrennLocalizedArtifactError(
            "preflight ledger failed exact replay"
        )
    bounded = _load_strict_json(
        output_directory / "bounded_degree_six.json"
    )
    expected_bounded = retained_bounded_degree_six_audit()
    if not strict_json_equal(bounded, expected_bounded):
        raise KrennLocalizedArtifactError(
            "bounded degree-six payload failed full exact replay"
        )
    rows = bounded["charts"]
    replay_indices = range(8) if full_degree_six_replay else (6,)
    for orbit_index in replay_indices:
        replay = modular_bounded_macaulay_preflight(
            orbit_index, 6, prime=1_009
        ).to_dict()
        if (
            _expected_row_tuple(replay)
            != EXPECTED_DEGREE_SIX_ROWS[orbit_index]
            or _relation_fingerprint(
                bounded_koszul_orbit_relations(orbit_index, 6)
            ) != rows[orbit_index]["koszul_relation_sha256"]
        ):
            raise KrennLocalizedArtifactError(
                "bounded degree-six native replay failed"
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or verify localized Krenn chart artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--output", type=Path, default=DEFAULT_RESULT_DIRECTORY
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "--output", type=Path, default=DEFAULT_RESULT_DIRECTORY
    )
    verify.add_argument("--full-degree-six-replay", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "generate":
        payload = write_localized_chart_bundle(arguments.output)
    else:
        payload = verify_localized_chart_bundle(
            arguments.output,
            full_degree_six_replay=arguments.full_degree_six_replay,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
