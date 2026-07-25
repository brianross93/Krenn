import hashlib
import json
from pathlib import Path
import unittest

from experiments.krenn_quantum_graph.fixtures import (
    fixture_n4_d3,
    fixture_n6_d2,
)
from experiments.krenn_quantum_graph.multi_star_identities import (
    certify_multi_star_identities,
)
from experiments.krenn_quantum_graph.perfect_matching_blocker_ideals import (
    blocker_census,
    brute_force_orientation_count,
    perfect_matching_blocker_types,
    top_chow_coefficient,
)
from experiments.krenn_quantum_graph.targets import canonical_ghz_target
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.ternary_search import (
    n6_d3_seed_witness,
)
from experiments.krenn_quantum_graph.symmetry_structure import (
    k4_matching_action_audit,
    perfect_matching_module,
    qutrit_weight_space_decomposition,
)


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = (
    ROOT
    / "results/krenn_quantum_graph/n6_d3_external_nonimage_reference"
)
STRUCTURAL = (
    ROOT
    / "results/krenn_quantum_graph/n_ge_8_structural_program"
)
STRUCTURAL_BUNDLE_FILES = {
    "README.md",
    "blocker_orbits.json",
    "identity_regressions.json",
    "symmetry_census.json",
}
STRUCTURAL_SOURCE_FILES = {
    "experiments/krenn_quantum_graph/multi_star_identities.py",
    "experiments/krenn_quantum_graph/perfect_matching_blocker_ideals.py",
    "experiments/krenn_quantum_graph/symmetry_structure.py",
}


def _load_json(path):
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    raw = path.read_bytes()
    result = json.loads(
        raw.decode("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant {value}")
        ),
        object_pairs_hook=reject_duplicate_keys,
    )
    canonical = (
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise ValueError(f"{path} is not canonical pretty JSON")
    return result


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class KrennStructuralProgramArtifactTest(unittest.TestCase):
    def test_external_reference_is_static_and_fail_closed(self):
        self.assertTrue(EXTERNAL.is_dir())
        self.assertEqual(
            {path.name for path in EXTERNAL.iterdir()},
            {"manifest.json", "reference.json"},
        )
        self.assertTrue(
            all(
                path.is_file() and not path.is_symlink()
                for path in EXTERNAL.iterdir()
            )
        )
        reference = _load_json(EXTERNAL / "reference.json")
        manifest = _load_json(EXTERNAL / "manifest.json")
        self.assertEqual(
            reference,
            {
                "schema": (
                    "krenn-external-n6-d3-nonimage-reference-v1"
                ),
                "status": "external_formal_theorem_reference",
                "external_repository": {
                    "url": (
                        "https://github.com/algal/"
                        "krenn-gu-6x3-certificate"
                    ),
                    "observed_head": (
                        "c04696e515e0c02be140353fb52ea60c62e827b1"
                    ),
                    "proof_content_commit": (
                        "105ffbc50b0443433fc53c248272617cc022f3e2"
                    ),
                    "verification_record_commit": (
                        "c04696e515e0c02be140353fb52ea60c62e827b1"
                    ),
                    "license_file_present": False,
                    "external_material_vendored_here": False,
                },
                "formal_statement": {
                    "name": (
                        "KrennGuCertificate."
                        "eqSystem6_no_solution_d3"
                    ),
                    "text": (
                        "not exists W : "
                        "MonochromaticQuantumGraph.WeightsN 6 3 C, "
                        "MonochromaticQuantumGraph.EqSystemN 6 3 W"
                    ),
                    "scope": {
                        "vertices": 6,
                        "colors": 3,
                        "field": "complex",
                        "normalized_target": "canonical GHZ",
                        "unrestricted_weights": True,
                    },
                },
                "pinned_environment": {
                    "lean_toolchain": "leanprover/lean4:v4.27.0",
                    "formal_conjectures_commit": (
                        "e751934294a381afd2d5fc1124c5953c8e25f9fa"
                    ),
                    "mathlib_commit": (
                        "a3a10db0e9d66acbebf76c5e6a135066525ac900"
                    ),
                    "lake_manifest_git_object": (
                        "bab87de6cdb90226d8e4a4be94b815d6e3da0f8e"
                    ),
                },
                "external_authors_reported_verification": {
                    "date": "2026-07-24",
                    "full_build_jobs": 8421,
                    "wall_seconds": 1338.034,
                    "committed_hashes_passed": 50,
                    "committed_hashes_total": 50,
                    "optional_producer_tests_passed": 111,
                    "axiom_closure": [
                        "propext",
                        "Classical.choice",
                        "Lean.ofReduceBool",
                        "Lean.trustCompiler",
                        "Quot.sound",
                    ],
                    "sorryAx_present": False,
                    "trust_boundary": (
                        "compiler-trusting native_decide proof, "
                        "not kernel-only reduction"
                    ),
                },
                "local_static_audit": {
                    "committed_hashes_passed": 50,
                    "committed_hashes_total": 50,
                    "formal_statement_inspected": True,
                    "official_fifteen_matching_bridge_inspected": True,
                    "local_full_lean_build_completed": False,
                    "local_leancheck_run": False,
                    "note": (
                        "A redundant local replay was stopped at the "
                        "user's request. This is neither a failed proof "
                        "check nor a claimed local replay."
                    ),
                },
                "composition_with_this_repository": {
                    "internal_exact_identity": (
                        "Phi(W(t)) = GHZ_6_3 + t e_002121 for nonzero t"
                    ),
                    "internal_conclusion": (
                        "GHZ_6_3 lies in the affine-image closure"
                    ),
                    "external_conclusion": (
                        "GHZ_6_3 is not in the affine image"
                    ),
                    "combined_conclusion": (
                        "GHZ_6_3 lies in the strict border of the "
                        "affine image"
                    ),
                    "dependency": (
                        "The strict-border conclusion accepts the "
                        "pinned external theorem; the Laurent "
                        "certificate remains independently replayable "
                        "here."
                    ),
                },
                "claim_boundary": {
                    "settles_n6_d3": True,
                    "settles_general_krenn_gu": False,
                    "implies_canonical_n6_all_d_at_least_3_by_"
                    "color_restriction": True,
                    "next_unsettled_even_vertex_count": 8,
                    "local_rebuild_required_to_claim_independent_"
                    "local_lean_replay": True,
                },
            },
        )
        self.assertEqual(
            manifest,
            {
                "schema": (
                    "krenn-external-n6-d3-reference-manifest-v1"
                ),
                "files": {
                    "reference.json": _sha256(
                        EXTERNAL / "reference.json"
                    )
                },
                "claim_boundary": {
                    "external_proof_vendored": False,
                    "local_full_lean_replay_claimed": False,
                    "static_reference_only": True,
                },
            },
        )

    def test_blocker_artifact_replays_native_censuses(self):
        artifact = _load_json(STRUCTURAL / "blocker_orbits.json")
        expected_censuses = []
        for n in (4, 6, 8):
            native = blocker_census(n)
            rows = []
            for blocker_type in perfect_matching_blocker_types(n):
                row = {
                    "barrier_size": blocker_type.barrier_size,
                    "odd_component_sizes": list(
                        blocker_type.odd_component_sizes
                    ),
                    "blocker_edge_count": len(
                        blocker_type.blocker_edges
                    ),
                    "live_edge_count": len(blocker_type.live_edges),
                    "labeled_orbit_size": (
                        blocker_type.labeled_orbit_size
                    ),
                }
                if n == 8:
                    row.update(
                        {
                            "qutrit_chart_variables": (
                                blocker_type
                                .qutrit_chart_variable_count
                            ),
                            "qutrit_chart_is_square": (
                                blocker_type.qutrit_chart_is_square
                            ),
                        }
                    )
                if blocker_type.qutrit_chart_is_square and n == 8:
                    outside = tuple(
                        vertex
                        for component in blocker_type.components
                        for vertex in component
                    )
                    chow = top_chow_coefficient(
                        blocker_type.blocker_edges, outside
                    )
                    self.assertEqual(
                        chow,
                        brute_force_orientation_count(
                            blocker_type.blocker_edges, outside
                        ),
                    )
                    row["top_chow_coefficient"] = chow
                rows.append(row)
            expected_censuses.append(
                {
                    "n": n,
                    "orbit_type_count": native["orbit_type_count"],
                    "labeled_blocker_count": native[
                        "labeled_blocker_count"
                    ],
                    "types": rows,
                }
            )
        self.assertEqual(
            artifact,
            {
                "schema": "krenn-perfect-matching-blocker-orbits-v1",
                "classification": {
                    "ideal": (
                        "J_n = < product_(e in M) b_e : M is a "
                        "perfect matching of K_n >"
                    ),
                    "minimal_prime": (
                        "P_B = < b_e : e in B > for an "
                        "inclusion-minimal perfect-matching edge "
                        "blocker B"
                    ),
                    "complement_form": (
                        "K_s join (K_a1 disjoint_union ... "
                        "disjoint_union K_a_(s+2))"
                    ),
                    "conditions": (
                        "0 <= s <= n/2-1; all a_i positive odd; "
                        "sum a_i = n-s"
                    ),
                },
                "censuses": expected_censuses,
                "independent_exact_checks": {
                    "minimal_blocker_check": (
                        "primary perfect-matching enumeration"
                    ),
                    "chow_method_1": (
                        "truncated coefficient multiplication"
                    ),
                    "chow_method_2": (
                        "brute-force endpoint orientations"
                    ),
                    "n8_square_degrees_agree": True,
                },
                "claim_boundary": {
                    "classifies_scalar_zero_pattern_blockers": True,
                    "classifies_colored_weight_cancellations": False,
                    "positive_chow_degree_proves_projective_"
                    "intersection_nonempty": True,
                    "positive_chow_degree_proves_selected_"
                    "affine_color_chart_nonempty": False,
                    "missing_lemma": (
                        "boundary escape under simultaneous "
                        "EqSystem and multi-star constraints"
                    ),
                },
            },
        )

    def test_structural_manifest_has_exact_inventory_and_hashes(self):
        self.assertTrue(STRUCTURAL.is_dir())
        self.assertEqual(
            {path.name for path in STRUCTURAL.iterdir()},
            STRUCTURAL_BUNDLE_FILES | {"manifest.json"},
        )
        self.assertTrue(
            all(
                path.is_file() and not path.is_symlink()
                for path in STRUCTURAL.iterdir()
            )
        )
        manifest = _load_json(STRUCTURAL / "manifest.json")
        self.assertEqual(
            set(manifest),
            {
                "schema",
                "bundle_files",
                "source_files",
                "claim_boundary",
            },
        )
        self.assertEqual(
            set(manifest["bundle_files"]), STRUCTURAL_BUNDLE_FILES
        )
        self.assertEqual(
            set(manifest["source_files"]), STRUCTURAL_SOURCE_FILES
        )
        for relative, expected in manifest["bundle_files"].items():
            path = STRUCTURAL / relative
            self.assertEqual(path.resolve().parent, STRUCTURAL.resolve())
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(_sha256(path), expected)
        for relative, expected in manifest["source_files"].items():
            path = ROOT / relative
            self.assertTrue(
                path.resolve().is_relative_to(ROOT.resolve())
            )
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(_sha256(path), expected)
        self.assertFalse(
            manifest["claim_boundary"]["n8_boundary_escape_proved"]
        )
        self.assertFalse(
            manifest["claim_boundary"]["global_krenn_gu_proved"]
        )

    def test_multi_star_identity_regressions_replay_exactly(self):
        artifact = _load_json(STRUCTURAL / "identity_regressions.json")
        cases = []
        fixtures = (
            ("fixture_n4_d3", fixture_n4_d3()),
            ("fixture_n6_d2", fixture_n6_d2()),
            ("n6_d3_seed_witness", n6_d3_seed_witness()),
        )
        for fixture_name, witness in fixtures:
            certificate = certify_multi_star_identities(witness)
            tensor_map = MatchingTensorMap(witness.n, witness.d)
            residual = tensor_map.exact_residual(
                witness,
                canonical_ghz_target(witness.n, witness.d),
            )
            if residual.entries:
                self.assertEqual(
                    residual.entries,
                    (((0, 0, 2, 1, 2, 1), 1),),
                )
                output = "canonical_ghz_plus_e_002121"
            else:
                output = "canonical_ghz"
            cases.append(
                {
                    "fixture": fixture_name,
                    "n": witness.n,
                    "d": witness.d,
                    "output": output,
                    "certificate": {
                        "schema": certificate.schema,
                        "star_coefficient_checks": (
                            certificate.star_coefficient_checks
                        ),
                        "cofactor_recursion_checks": (
                            certificate.cofactor_recursion_checks
                        ),
                        "shared_edge_entry_checks": (
                            certificate.shared_edge_entry_checks
                        ),
                        "two_star_coefficient_checks": (
                            certificate.two_star_coefficient_checks
                        ),
                        "exact": certificate.exact,
                    },
                }
            )
        self.assertEqual(
            artifact,
            {
                "schema": "krenn-multi-star-identity-regressions-v1",
                "cases": cases,
                "claim_boundary": {
                    "universal_combinatorial_identities": True,
                    "adds_independent_equations_to_the_tensor_map": False,
                    "proves_n8_nonexistence": False,
                    "proves_induction_between_vertex_counts": False,
                },
            },
        )

    def test_symmetry_census_replays_exact_dimensions(self):
        artifact = _load_json(STRUCTURAL / "symmetry_census.json")
        audit = k4_matching_action_audit()
        matching_modules = []
        for n in (4, 6, 8):
            native = perfect_matching_module(n)
            matching_modules.append(
                {
                    "n": n,
                    "dimension": sum(
                        irrep.total_dimension for irrep in native
                    ),
                    "irreps": [
                        {
                            "partition": list(irrep.partition),
                            "dimension": irrep.dimension,
                        }
                        for irrep in native
                    ],
                }
            )
        qutrit_dimensions = []
        for n in (4, 6, 8):
            native = qutrit_weight_space_decomposition(n)
            qutrit_dimensions.append(
                {
                    "n": n,
                    "total": sum(
                        item.total_dimension for item in native
                    ),
                    "endpoint_symmetric": sum(
                        item.total_dimension
                        for item in native
                        if item.sector == "endpoint_symmetric"
                    ),
                    "endpoint_antisymmetric": sum(
                        item.total_dimension
                        for item in native
                        if item.sector == "endpoint_antisymmetric"
                    ),
                }
            )
        self.assertEqual(
            artifact,
            {
                "schema": "krenn-symmetry-structure-census-v1",
                "k4_matching_action": {
                    "vertex_group_order": audit["vertex_group_order"],
                    "perfect_matching_count": audit["matching_count"],
                    "image_order": audit["image_order"],
                    "kernel_order": audit["kernel_order"],
                    "kernel": audit["kernel"],
                    "conclusion": "S4 / V4 isomorphic to S3",
                },
                "perfect_matching_modules": matching_modules,
                "qutrit_weight_space": {
                    "formula": (
                        "([n]+[n-1,1]+[n-2,2]) tensor "
                        "(2 trivial + 2 standard) plus "
                        "([n-1,1]+[n-2,1,1]) tensor "
                        "(standard + sign)"
                    ),
                    "dimensions": qutrit_dimensions,
                },
                "claim_boundary": {
                    "exact_representation_census": True,
                    "proves_arbitrary_solution_is_symmetric": False,
                    "proves_an_unreachable_target_isotypic_"
                    "component": False,
                    "role": (
                        "organizes equivariant identities and orbit "
                        "types without equating symmetry-related "
                        "weights"
                    ),
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
