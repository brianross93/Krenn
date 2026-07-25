"""Round-trippable artifacts for the exact localized-chart campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from experiments.krenn_quantum_graph.certificate_routing import (
    KrennCertificateRoutingError,
    certificate_routing_audit,
    verify_certificate_routing_audit,
)
from experiments.krenn_quantum_graph.localized_chart_cas_runner import (
    DEFAULT_SCRATCH_ROOT,
    DOCKER_IMAGE,
    DOCKER_IMAGE_ID,
    ENGINE_BUDGET_SECONDS,
    OUTPUT_SUBDIRECTORY,
    PROBES,
    RUNNER_SCHEMA,
    SUITE_SCHEMA,
    docker_argv,
    probe_input_path,
)
from experiments.krenn_quantum_graph.localized_chart_derivative import (
    natural_derivative_atlas_audit,
)
from experiments.krenn_quantum_graph.localized_chart_graded_macaulay import (
    KrennGradedMacaulayError,
    graded_derivative_macaulay_audit,
    verify_graded_derivative_macaulay_audit,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    localized_chart_cover_audit,
    localized_chart_germ_audit,
    normalized_seed_chart,
    strict_json_equal,
    verify_localized_chart_cover_audit,
)
from experiments.krenn_quantum_graph.localized_chart_leaf_free import (
    KrennLeafFreeError,
    leaf_free_saturation_audit,
    verify_leaf_free_saturation_audit,
)
from experiments.krenn_quantum_graph.localized_chart_leaf_free_cas_runner import (
    LEAF_FREE_PROBES,
    LEAF_FREE_RUNNER_SCHEMA,
    MANIFEST_NAME as LEAF_FREE_MANIFEST_NAME,
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
from experiments.krenn_quantum_graph.localized_chart_residual_grading import (
    residual_grading_audit,
    verify_residual_grading_audit,
)
from experiments.krenn_quantum_graph.localized_chart_sparse_cas_runner import (
    MANIFEST_NAME as SPARSE_MANIFEST_NAME,
    SPARSE_PROBES,
    SPARSE_RUNNER_SCHEMA,
)
from experiments.krenn_quantum_graph.localized_chart_sparse_elimination import (
    KrennSparseEliminationError,
    sparse_elimination_audit,
    verify_sparse_elimination_audit,
)
from experiments.krenn_quantum_graph.star_linearization import (
    KrennStarLinearizationError,
    star_linearization_audit,
    verify_star_linearization_audit,
)
from experiments.krenn_quantum_graph.star_pivot_charts import (
    KrennStarPivotChartError,
    star_pivot_chart_audit,
    verify_star_pivot_chart_audit,
)
from experiments.krenn_quantum_graph.star_pivot_gauge import (
    KrennStarPivotGaugeError,
    star_pivot_gauge_audit,
    verify_star_pivot_gauge_audit,
)
from experiments.krenn_quantum_graph.star_pivot_affine_slices import (
    KrennStarPivotAffineSliceError,
    star_pivot_affine_slice_audit,
    verify_star_pivot_affine_slice_audit,
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
    "certificate_routing.json",
    "chart_cover.json",
    "germ_separation.json",
    "graded_derivative_macaulay.json",
    "leaf_free_saturation.json",
    "natural_derivative_atlas.json",
    "natural_repair_monomial_atlas.json",
    "preflight_ledger.json",
    "residual_grading.json",
    "sparse_elimination.json",
    "star_linearization.json",
    "star_pivot_charts.json",
    "star_pivot_affine_slices.json",
    "star_pivot_gauge.json",
)
SOURCE_FILES = (
    "experiments/__init__.py",
    "experiments/krenn_quantum_graph/__init__.py",
    "experiments/krenn_quantum_graph/certificate_routing.py",
    "experiments/krenn_quantum_graph/localized_chart_artifact.py",
    "experiments/krenn_quantum_graph/localized_chart_cas_runner.py",
    "experiments/krenn_quantum_graph/localized_chart_derivative.py",
    "experiments/krenn_quantum_graph/localized_chart_graded_macaulay.py",
    "experiments/krenn_quantum_graph/localized_chart_ideals.py",
    "experiments/krenn_quantum_graph/localized_chart_independent.py",
    "experiments/krenn_quantum_graph/localized_chart_leaf_free.py",
    "experiments/krenn_quantum_graph/localized_chart_leaf_free_cas_runner.py",
    "experiments/krenn_quantum_graph/localized_chart_macaulay.py",
    "experiments/krenn_quantum_graph/localized_chart_monomial_atlas.py",
    "experiments/krenn_quantum_graph/localized_chart_residual_grading.py",
    "experiments/krenn_quantum_graph/localized_chart_sparse_cas_runner.py",
    "experiments/krenn_quantum_graph/localized_chart_sparse_elimination.py",
    "experiments/krenn_quantum_graph/star_linearization.py",
    "experiments/krenn_quantum_graph/star_linearization_independent.py",
    "experiments/krenn_quantum_graph/star_pivot_charts.py",
    "experiments/krenn_quantum_graph/star_pivot_affine_slices.py",
    "experiments/krenn_quantum_graph/star_pivot_affine_cas_runner.py",
    "experiments/krenn_quantum_graph/star_pivot_gauge.py",
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
EMPTY_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
RETAINED_TEN_MINUTE_RESULTS = {
    "natural_chart_p31_slimgb": {
        "started_at_utc": "2026-07-25T02:12:04.865755+00:00",
        "elapsed_seconds": 600.622037,
        "stdout_bytes": 550,
        "stdout_sha256": (
            "5809e3b66b083ce0f0938d4fae0c4cb799cda19fb2f2a9b"
            "dcedacf6bd13b37bb"
        ),
        "receipt_bytes": 2_134,
        "receipt_sha256": (
            "e73c8fdfcee0b0901d78326d6f151fea11e2574e6b364de"
            "7d401e483e0eb414e"
        ),
        "hardened_command": False,
    },
    "derivative_11_p31_std": {
        "started_at_utc": "2026-07-25T02:12:04.865738+00:00",
        "elapsed_seconds": 600.563928,
        "stdout_bytes": 513,
        "stdout_sha256": (
            "d713aca1608371584c51f0c0576d07dd9fde3fc564366b28"
            "952037faa8b5c1b2"
        ),
        "receipt_bytes": 2_169,
        "receipt_sha256": (
            "ff2a249c96ce8272a0bef932fdee7df4c4207732835c416c"
            "8f2a93bf18f848fb"
        ),
        "hardened_command": False,
    },
    "derivative_29_p31_std": {
        "started_at_utc": "2026-07-25T02:12:04.865714+00:00",
        "elapsed_seconds": 600.587278,
        "stdout_bytes": 513,
        "stdout_sha256": (
            "147ac26cb492b03013d6c4cc0cb3fc5bc10778b3d0fe7562"
            "e8cdc3ea67626d35"
        ),
        "receipt_bytes": 2_169,
        "receipt_sha256": (
            "ddcfe27760040188fd06fe417e1efe1d0d6c6a0fcb55b354"
            "0bb7fd8f71926411"
        ),
        "hardened_command": False,
    },
    "repair_11_65_p31_std": {
        "started_at_utc": "2026-07-25T02:33:02.844489+00:00",
        "elapsed_seconds": 600.455603,
        "stdout_bytes": 507,
        "stdout_sha256": (
            "ac8844314393526f1692b20e1a3ee1a23788a954d23067a5"
            "fa42aa6637bbed96"
        ),
        "receipt_bytes": 2_266,
        "receipt_sha256": (
            "24d236c95e09c6cf6f496f3a9a3a3872f94c8572135ef7b"
            "78a77a66db7657cdc"
        ),
        "hardened_command": True,
    },
    "repair_29_47_p31_std": {
        "started_at_utc": "2026-07-25T02:33:02.846202+00:00",
        "elapsed_seconds": 600.451384,
        "stdout_bytes": 507,
        "stdout_sha256": (
            "28a1a8768fd71ddadfffb71b6788f10c342b2ba6ac7ca6a6"
            "a88dd8d7c06ee15f"
        ),
        "receipt_bytes": 2_266,
        "receipt_sha256": (
            "10301a96f9398556839a76faa2b467e85e8f050835addf4f"
            "27c1dfc1efce82ab"
        ),
        "hardened_command": True,
    },
    "repair_11_55_133_p31_std": {
        "started_at_utc": "2026-07-25T02:22:05.478353+00:00",
        "elapsed_seconds": 600.406723,
        "stdout_bytes": 511,
        "stdout_sha256": (
            "59c96d2c3669179f498dd120ddb1764bf8a90861bc563b4a"
            "df4ce66d87ca189c"
        ),
        "receipt_bytes": 2_174,
        "receipt_sha256": (
            "53ab8cd44d5aab2b5b2881cad2ed5d5de4c1dd5c23693d4"
            "374771e1eb11a3ad7"
        ),
        "hardened_command": False,
    },
    "repair_29_55_106_p31_std": {
        "started_at_utc": "2026-07-25T02:22:05.503417+00:00",
        "elapsed_seconds": 600.492321,
        "stdout_bytes": 511,
        "stdout_sha256": (
            "7eccf2f5b8219175f23984f051525a4869c1619c6ef109ac"
            "2828701a8069b46d"
        ),
        "receipt_bytes": 2_174,
        "receipt_sha256": (
            "60fe0819abd5f4eb6ad7f346b61ef7e7e76971e3e7d3cc2"
            "d7b3cb6157e8008e5"
        ),
        "hardened_command": False,
    },
}
RETAINED_SPARSE_TWO_PIVOT_RESULTS = {
    "sparse_two_pivot_11_p1009_std": {
        "started_at_utc": "2026-07-25T04:27:04.726306+00:00",
        "elapsed_seconds": 600.555479,
        "stdout_bytes": 578,
        "stdout_sha256": (
            "8786d40c64e038fccf56ad6e4f1b8076228cdedcf4ec828"
            "ab0a4691bd0f7bc2e"
        ),
        "receipt_bytes": 2_410,
        "receipt_sha256": (
            "70f79a1ff3f9dd3b7d1cb21c4551f529fa6771b70121154"
            "2f00b3c369b5ef613"
        ),
    },
    "sparse_two_pivot_29_p1009_std": {
        "started_at_utc": "2026-07-25T04:27:04.725728+00:00",
        "elapsed_seconds": 600.555902,
        "stdout_bytes": 578,
        "stdout_sha256": (
            "c4161ec01d65313df0631dc341848642c72a811f4b55f84"
            "b98f699f9b0bba448"
        ),
        "receipt_bytes": 2_410,
        "receipt_sha256": (
            "feaa7ce4eca6a5fe6b87e3c4ee33f00b80327622d89941d"
            "61db238247823691f"
        ),
    },
}
RETAINED_SPARSE_TWO_PIVOT_MANIFEST = {
    "bytes": 2_499,
    "sha256": (
        "4938cc2d40b2236596db33bb77ddac86f7e63cb21365b7cd"
        "82b7271e0a13d2b2"
    ),
}
RETAINED_LEAF_FREE_INITIAL_RESULTS = {
    "leaf_free_derivative_11_a0_p31_sat": {
        "started_at_utc": "2026-07-25T04:45:10.080168+00:00",
        "elapsed_seconds": 600.926601,
        "stdout_bytes": 1_286,
        "stdout_sha256": (
            "f408028a16f0cbd8cff7318445faa8c74aa2c66b0034acc4"
            "8ba58b1861a09671"
        ),
        "receipt_bytes": 2_398,
        "receipt_sha256": (
            "c10e14fe3a3add48ead97bff95e77f1fa86d45716f5739aa"
            "4b340a159261e11c"
        ),
    },
    "leaf_free_derivative_29_a0_p31_sat": {
        "started_at_utc": "2026-07-25T04:45:10.080231+00:00",
        "elapsed_seconds": 600.912834,
        "stdout_bytes": 1_286,
        "stdout_sha256": (
            "b0e9fe0e31c7cb304656872866d3e8aa379cc2db2c772795"
            "78881a606548ccc0"
        ),
        "receipt_bytes": 2_398,
        "receipt_sha256": (
            "6f95e2feb1a60e9d5fae34a3234459fb164fe967139f31d3"
            "cc7c83e821ff797a"
        ),
    },
}
RETAINED_LEAF_FREE_INITIAL_MANIFEST = {
    "bytes": 2_739,
    "sha256": (
        "2f7e5d4c8311504688a90130a87f146b1361835c30606c45"
        "1584c11044369e51"
    ),
}


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


def _superseded_short_preflight_ledger() -> dict:
    """Retain the original smoke-test receipts as historical provenance."""

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
                "e5416cfbbda0485ed1995e75"
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


def _retained_ten_minute_runs() -> list[dict]:
    if set(RETAINED_TEN_MINUTE_RESULTS) != {
        probe.key for probe in PROBES
    }:
        raise KrennLocalizedArtifactError(
            "the retained ten-minute CAS result keys changed"
        )
    rows = []
    output_root = DEFAULT_SCRATCH_ROOT / OUTPUT_SUBDIRECTORY
    for probe in PROBES:
        retained = RETAINED_TEN_MINUTE_RESULTS[probe.key]
        hardened = retained["hardened_command"]
        if (
            retained["elapsed_seconds"] < ENGINE_BUDGET_SECONDS
            or retained["elapsed_seconds"] >= 610
        ):
            raise KrennLocalizedArtifactError(
                "a retained ten-minute elapsed time changed"
            )
        rows.append({
            "runner_schema": RUNNER_SCHEMA,
            "key": probe.key,
            "ideal": probe.ideal,
            "algorithm": probe.algorithm,
            "characteristic": probe.characteristic,
            "budget_seconds": ENGINE_BUDGET_SECONDS,
            "cpus": 2,
            "memory_gib": 8,
            "command_profile": (
                "pinned-image-forced-kill"
                if hardened else "verified-image-tag"
            ),
            "command_argv": list(docker_argv(
                probe,
                DEFAULT_SCRATCH_ROOT,
                hardened=hardened,
            )),
            "input_script": {
                "path": str(probe_input_path(
                    probe, DEFAULT_SCRATCH_ROOT
                )),
                "bytes": probe.input_bytes,
                "sha256": probe.input_sha256,
            },
            "started_at_utc": retained["started_at_utc"],
            "elapsed_seconds": retained["elapsed_seconds"],
            "return_code": 124,
            "stdout": {
                "path": str(output_root / f"{probe.key}.stdout.txt"),
                "bytes": retained["stdout_bytes"],
                "sha256": retained["stdout_sha256"],
            },
            "stderr": {
                "path": str(output_root / f"{probe.key}.stderr.txt"),
                "bytes": 0,
                "sha256": EMPTY_SHA256,
            },
            "receipt": {
                "path": str(
                    output_root / f"{probe.key}.receipt.json"
                ),
                "bytes": retained["receipt_bytes"],
                "sha256": retained["receipt_sha256"],
            },
            "marker_receipt": {
                "parse_marker": probe.parse_marker,
                "parse_marker_observed": True,
                "completion_marker": probe.completion_marker,
                "completion_marker_observed": False,
            },
            "status": "timeout-after-parse",
            "claim_boundary": {
                "timeout_is_a_chart_decision": False,
                "F31_result_is_an_exact_Q_proof": False,
            },
        })
    return rows


def _retained_sparse_two_pivot_runs() -> list[dict]:
    if set(RETAINED_SPARSE_TWO_PIVOT_RESULTS) != {
        probe.key for probe in SPARSE_PROBES
    }:
        raise KrennLocalizedArtifactError(
            "the retained sparse two-pivot CAS keys changed"
        )
    rows = []
    output_root = DEFAULT_SCRATCH_ROOT / OUTPUT_SUBDIRECTORY
    for probe in SPARSE_PROBES:
        retained = RETAINED_SPARSE_TWO_PIVOT_RESULTS[probe.key]
        elapsed = retained["elapsed_seconds"]
        if elapsed < ENGINE_BUDGET_SECONDS or elapsed >= 610:
            raise KrennLocalizedArtifactError(
                "a retained sparse two-pivot elapsed time changed"
            )
        rows.append({
            "runner_schema": RUNNER_SCHEMA,
            "key": probe.key,
            "ideal": probe.ideal,
            "algorithm": probe.algorithm,
            "characteristic": probe.characteristic,
            "budget_seconds": ENGINE_BUDGET_SECONDS,
            "cpus": 2,
            "memory_gib": 8,
            "command_profile": "pinned-image-forced-kill",
            "command_argv": list(docker_argv(
                probe, DEFAULT_SCRATCH_ROOT, hardened=True
            )),
            "input_script": {
                "path": str(probe_input_path(
                    probe, DEFAULT_SCRATCH_ROOT
                )),
                "bytes": probe.input_bytes,
                "sha256": probe.input_sha256,
            },
            "started_at_utc": retained["started_at_utc"],
            "elapsed_seconds": elapsed,
            "return_code": 124,
            "stdout": {
                "path": str(output_root / f"{probe.key}.stdout.txt"),
                "bytes": retained["stdout_bytes"],
                "sha256": retained["stdout_sha256"],
            },
            "stderr": {
                "path": str(output_root / f"{probe.key}.stderr.txt"),
                "bytes": 0,
                "sha256": EMPTY_SHA256,
            },
            "receipt": {
                "path": str(output_root / f"{probe.key}.receipt.json"),
                "bytes": retained["receipt_bytes"],
                "sha256": retained["receipt_sha256"],
            },
            "marker_receipt": {
                "parse_marker": probe.parse_marker,
                "parse_marker_observed": True,
                "completion_marker": probe.completion_marker,
                "completion_marker_observed": False,
            },
            "status": "timeout-after-parse",
            "claim_boundary": {
                "timeout_is_a_chart_decision": False,
                "F1009_result_is_an_exact_Q_proof": False,
            },
        })
    return rows


def _retained_leaf_free_initial_runs() -> list[dict]:
    if set(RETAINED_LEAF_FREE_INITIAL_RESULTS) != {
        probe.key for probe in LEAF_FREE_PROBES
    }:
        raise KrennLocalizedArtifactError(
            "the retained leaf-free CAS keys changed"
        )
    rows = []
    output_root = DEFAULT_SCRATCH_ROOT / OUTPUT_SUBDIRECTORY
    for probe in LEAF_FREE_PROBES:
        retained = RETAINED_LEAF_FREE_INITIAL_RESULTS[probe.key]
        elapsed = retained["elapsed_seconds"]
        if elapsed < ENGINE_BUDGET_SECONDS or elapsed >= 610:
            raise KrennLocalizedArtifactError(
                "a retained leaf-free elapsed time changed"
            )
        rows.append({
            "runner_schema": RUNNER_SCHEMA,
            "key": probe.key,
            "ideal": probe.ideal,
            "algorithm": probe.algorithm,
            "characteristic": probe.characteristic,
            "saturation_factor": "A0",
            "saturation_stages_completed": 0,
            "budget_seconds": ENGINE_BUDGET_SECONDS,
            "cpus": 2,
            "memory_gib": 8,
            "command_profile": "pinned-image-forced-kill",
            "command_argv": list(docker_argv(
                probe, DEFAULT_SCRATCH_ROOT, hardened=True
            )),
            "input_script": {
                "path": str(probe_input_path(
                    probe, DEFAULT_SCRATCH_ROOT
                )),
                "bytes": probe.input_bytes,
                "sha256": probe.input_sha256,
            },
            "started_at_utc": retained["started_at_utc"],
            "elapsed_seconds": elapsed,
            "return_code": 124,
            "stdout": {
                "path": str(output_root / f"{probe.key}.stdout.txt"),
                "bytes": retained["stdout_bytes"],
                "sha256": retained["stdout_sha256"],
            },
            "stderr": {
                "path": str(output_root / f"{probe.key}.stderr.txt"),
                "bytes": 0,
                "sha256": EMPTY_SHA256,
            },
            "receipt": {
                "path": str(output_root / f"{probe.key}.receipt.json"),
                "bytes": retained["receipt_bytes"],
                "sha256": retained["receipt_sha256"],
            },
            "marker_receipt": {
                "parse_marker": probe.parse_marker,
                "parse_marker_observed": True,
                "completion_marker": probe.completion_marker,
                "completion_marker_observed": False,
            },
            "status": "timeout-after-parse",
            "claim_boundary": {
                "timeout_is_a_chart_decision": False,
                "F31_result_is_an_exact_Q_proof": False,
                "later_saturation_stages_completed": False,
            },
        })
    return rows


def preflight_ledger() -> dict:
    """Record the completed exact and ten-minute bounded computations."""

    historical = _superseded_short_preflight_ledger()
    return {
        "schema": ARTIFACT_SCHEMA,
        "host_policy": {
            "exact_macaulay_workers": 1,
            "maximum_allowed_workers": 16,
            "maximum_observed_concurrent_CAS_probes": 3,
            "maximum_observed_concurrent_CAS_cpus": 6,
            "maximum_observed_concurrent_CAS_memory_gib": 24,
            "network": "disabled for every Docker run",
            "scratch_root": str(DEFAULT_SCRATCH_ROOT),
            "random_seeds": [],
        },
        "exact_macaulay": historical["exact_macaulay"],
        "singular_engine": {
            "docker_image": DOCKER_IMAGE,
            "image_id": DOCKER_IMAGE_ID,
            "version": "4.3.2",
            "coefficient_fields": ["F_31", "F_1009", "Q"],
            "ten_minute_F31_reconnaissance": {
                "runner_schema": RUNNER_SCHEMA,
                "suite_schema": SUITE_SCHEMA,
                "probe_count": len(PROBES),
                "budget_seconds_per_probe": ENGINE_BUDGET_SECONDS,
                "aggregate_scratch_manifest": {
                    "path": str(
                        DEFAULT_SCRATCH_ROOT
                        / OUTPUT_SUBDIRECTORY
                        / "ten_minute_run_manifest.json"
                    ),
                    "bytes": 4_972,
                    "sha256": (
                        "9919270a4bfa7bcfa2c79bb0be102c809d87c27e"
                        "8feb6682492fe81def79f4b2"
                    ),
                },
                "execution_batches": [
                    {
                        "worker_limit": 3,
                        "maximum_cpus": 6,
                        "maximum_memory_gib": 24,
                        "new_probe_keys": [
                            "natural_chart_p31_slimgb",
                            "derivative_11_p31_std",
                            "derivative_29_p31_std",
                            "repair_11_55_133_p31_std",
                            "repair_29_55_106_p31_std",
                        ],
                    },
                    {
                        "worker_limit": 2,
                        "maximum_cpus": 4,
                        "maximum_memory_gib": 16,
                        "new_probe_keys": [
                            "repair_11_65_p31_std",
                            "repair_29_47_p31_std",
                        ],
                    },
                ],
                "runs": _retained_ten_minute_runs(),
                "outcome": (
                    "all seven probes parsed and timed out after the full "
                    "600-second engine budget without a completion marker"
                ),
            },
            "ten_minute_F1009_sparse_two_pivot_reconnaissance": {
                "runner_schema": SPARSE_RUNNER_SCHEMA,
                "probe_count": len(SPARSE_PROBES),
                "budget_seconds_per_probe": ENGINE_BUDGET_SECONDS,
                "algorithm": "std",
                "order": "two-pivot-elimination",
                "aggregate_scratch_manifest": {
                    "path": str(
                        DEFAULT_SCRATCH_ROOT
                        / OUTPUT_SUBDIRECTORY
                        / SPARSE_MANIFEST_NAME
                    ),
                    **RETAINED_SPARSE_TWO_PIVOT_MANIFEST,
                },
                "worker_limit": 2,
                "maximum_cpus": 4,
                "maximum_memory_gib": 16,
                "runs": _retained_sparse_two_pivot_runs(),
                "outcome": (
                    "both probes parsed and timed out after the full "
                    "600-second engine budget without a completion marker"
                ),
                "claim_boundary": {
                    "timeout_is_a_chart_decision": False,
                    "F1009_result_is_an_exact_Q_proof": False,
                },
            },
            "ten_minute_F31_leaf_free_A0_reconnaissance": {
                "runner_schema": LEAF_FREE_RUNNER_SCHEMA,
                "probe_count": len(LEAF_FREE_PROBES),
                "budget_seconds_per_probe": ENGINE_BUDGET_SECONDS,
                "algorithm": "sat-A0",
                "saturation_factor": "A0",
                "aggregate_scratch_manifest": {
                    "path": str(
                        DEFAULT_SCRATCH_ROOT
                        / OUTPUT_SUBDIRECTORY
                        / LEAF_FREE_MANIFEST_NAME
                    ),
                    **RETAINED_LEAF_FREE_INITIAL_MANIFEST,
                },
                "worker_limit": 2,
                "maximum_cpus": 4,
                "maximum_memory_gib": 16,
                "runs": _retained_leaf_free_initial_runs(),
                "outcome": (
                    "both A0 saturation probes parsed and timed out after "
                    "the full 600-second engine budget; no saturation "
                    "stage completed"
                ),
                "claim_boundary": {
                    "timeout_is_a_chart_decision": False,
                    "F31_result_is_an_exact_Q_proof": False,
                    "later_saturation_stages_completed": False,
                },
            },
            "superseded_smoke_preflights": {
                "purpose": "tractability smoke tests only",
                "runs": [
                    {
                        "ideal": row["ideal"],
                        "budget_seconds": row["budget_seconds"],
                        "status": row["status"],
                    }
                    for row in historical["singular_engine"]["runs"][:5]
                ],
            },
            "exact_Q_runs": [
                historical["singular_engine"]["runs"][5]
            ],
        },
        "claim_boundary": {
            "a_timeout_is_a_decision": False,
            "an_F31_result_is_an_exact_Q_decision": False,
            "a_bounded_certificate_miss_is_a_global_decision": False,
            "retained_timeout_logs_are_a_certificate": False,
            "smooth_defect_hypersurface_decides_full_chart": False,
            "finite_affine_GHZ_membership_status": "undecided",
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
            "repair_residual_character_ranks": [8, 8, 7, 7],
            "sparse_derivative_residual_character_rank": 8,
            "sparse_derivative_exact_Q_degree_excluded": 5,
            "sparse_two_pivot_p1009_timeouts": 2,
            "leaf_free_saturation_stages_completed": 0,
            "leaf_free_initial_A0_p31_timeouts": 2,
            "exact_star_factorizations": 6,
            "star_rows_per_apex_color": 243,
            "star_columns_per_apex_color": 15,
            "proposed_single_column_differencing_lemma_valid": False,
            "proposed_q_routed_certificate_pair_valid": False,
            "star_target_row_nonzero_pivot_charts": 125,
            "star_target_row_pivot_orbit_types": 3,
            "star_pivot_factors_gauge_normalizable_to_one": True,
            "star_weights_monic_after_pivot_normalization": 9,
            "star_pivot_affine_slice_variables": 135,
            "star_pivot_affine_slice_generators": 732,
            "star_pivot_affine_slice_orbit_types": 3,
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
        "certificate_routing.json": certificate_routing_audit(),
        "chart_cover.json": localized_chart_cover_audit(),
        "germ_separation.json": localized_chart_germ_audit(),
        "graded_derivative_macaulay.json":
            graded_derivative_macaulay_audit(),
        "leaf_free_saturation.json": leaf_free_saturation_audit(),
        "natural_derivative_atlas.json":
            natural_derivative_atlas_audit(),
        "natural_repair_monomial_atlas.json":
            natural_repair_monomial_atlas_audit(),
        "preflight_ledger.json": preflight_ledger(),
        "residual_grading.json": residual_grading_audit(),
        "sparse_elimination.json": sparse_elimination_audit(),
        "star_linearization.json": star_linearization_audit(),
        "star_pivot_charts.json": star_pivot_chart_audit(),
        "star_pivot_affine_slices.json":
            star_pivot_affine_slice_audit(),
        "star_pivot_gauge.json": star_pivot_gauge_audit(),
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
    try:
        verify_certificate_routing_audit(
            _load_strict_json(
                output_directory / "certificate_routing.json"
            )
        )
    except KrennCertificateRoutingError as error:
        raise KrennLocalizedArtifactError(
            "the certificate-routing correction failed exact replay"
        ) from error
    if not strict_json_equal(
        _load_strict_json(output_directory / "germ_separation.json"),
        localized_chart_germ_audit(),
    ):
        raise KrennLocalizedArtifactError(
            "germ separation audit failed exact replay"
        )
    try:
        verify_graded_derivative_macaulay_audit(
            _load_strict_json(
                output_directory / "graded_derivative_macaulay.json"
            ),
            full_degree_six_replay=full_degree_six_replay,
        )
        verify_leaf_free_saturation_audit(
            _load_strict_json(
                output_directory / "leaf_free_saturation.json"
            )
        )
    except (
        KrennGradedMacaulayError,
        KrennLeafFreeError,
    ) as error:
        raise KrennLocalizedArtifactError(
            "a derived exact chart audit failed replay"
        ) from error
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
    verify_residual_grading_audit(
        _load_strict_json(output_directory / "residual_grading.json")
    )
    try:
        verify_sparse_elimination_audit(
            _load_strict_json(
                output_directory / "sparse_elimination.json"
            )
        )
    except KrennSparseEliminationError as error:
        raise KrennLocalizedArtifactError(
            "the sparse elimination audit failed replay"
        ) from error
    try:
        verify_star_linearization_audit(
            _load_strict_json(
                output_directory / "star_linearization.json"
            )
        )
    except KrennStarLinearizationError as error:
        raise KrennLocalizedArtifactError(
            "the exact star-linearization audit failed replay"
        ) from error
    try:
        verify_star_pivot_chart_audit(
            _load_strict_json(
                output_directory / "star_pivot_charts.json"
            )
        )
    except KrennStarPivotChartError as error:
        raise KrennLocalizedArtifactError(
            "the exact star-pivot cover failed replay"
        ) from error
    try:
        verify_star_pivot_gauge_audit(
            _load_strict_json(
                output_directory / "star_pivot_gauge.json"
            )
        )
    except KrennStarPivotGaugeError as error:
        raise KrennLocalizedArtifactError(
            "the exact star-pivot gauge normalization failed replay"
        ) from error
    try:
        verify_star_pivot_affine_slice_audit(
            _load_strict_json(
                output_directory / "star_pivot_affine_slices.json"
            )
        )
    except KrennStarPivotAffineSliceError as error:
        raise KrennLocalizedArtifactError(
            "the exact star-pivot affine slices failed replay"
        ) from error
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
