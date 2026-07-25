"""Tests for the portable star variable-projection campaign artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from experiments.krenn_quantum_graph import (
    star_variable_projection_artifact as artifact_module,
)
from experiments.krenn_quantum_graph import (
    star_variable_projection_campaign as campaign_module,
)
from experiments.krenn_quantum_graph.star_variable_projection_artifact import (
    KrennVariableProjectionArtifactError,
    generate_variable_projection_summary_bundle,
    summarize_completed_campaign,
    verify_variable_projection_summary_bundle,
    wait_for_terminal_manifest,
)
from experiments.krenn_quantum_graph.star_variable_projection_campaign import (
    CAMPAIGN_SCHEMA,
    MAXIMUM_WORKERS,
    RadiusCaps,
    TrajectoryConfig,
    evaluate_projection,
)


@dataclass(frozen=True)
class _FakeRepresentative:
    index: int


class _FakeCore:
    """Small deterministic core with production-compatible dimensions."""

    @staticmethod
    def pivot_representatives():
        return tuple(_FakeRepresentative(index) for index in range(3))

    @staticmethod
    def targets():
        target = np.zeros((243, 3), dtype=np.complex128)
        target[(15, 16, 17), (0, 1, 2)] = 1
        return target

    @staticmethod
    def initial_point(representative, seed):
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), int(representative.index), 991]
            )
        )
        u = 0.05 * (
            generator.normal(size=90)
            + 1j * generator.normal(size=90)
        )
        u[:3] = 1
        u[3:6] += 0.75
        return u

    @staticmethod
    def pivot_values(u, representative):
        del representative
        return np.asarray(u[:3], dtype=np.complex128)

    @staticmethod
    def retract_pivots(u, representative, minimum_abs=1e-12):
        del representative
        u = np.asarray(u, dtype=np.complex128).copy()
        pivots = u[:3].copy()
        if np.min(np.abs(pivots)) < minimum_abs:
            raise ValueError("pivot boundary")
        u[:3] = 1
        return u, {
            "root_free": True,
            "raw_pivot_minimum_abs": float(np.min(np.abs(pivots))),
        }

    @staticmethod
    def balance_residual_gauge(
        u, representative, tolerance=1e-10, maximum_iterations=80
    ):
        del representative, tolerance, maximum_iterations
        return np.asarray(u, dtype=np.complex128).copy(), {
            "status": "converged",
            "accepted_for_optimization": True,
            "moment_norm": 0.0,
            "tangent_rank": 0,
            "newton_iterations": 0,
            "suspected_recession": False,
        }

    @staticmethod
    def evaluate_star_matrix(u):
        u = np.asarray(u, dtype=np.complex128)
        matrix = np.zeros((243, 15), dtype=np.complex128)
        matrix[np.arange(15), np.arange(15)] = 1
        matrix[(15, 16, 17), (0, 1, 2)] = u[3:6]
        matrix[(18, 19, 20), (0, 1, 2)] = 0.1 * u[6:9]
        return matrix

    @staticmethod
    def pullback_gradient(u, y, residual):
        del u
        matrix_gradient = residual @ y.conj().T
        gradient = np.zeros(90, dtype=np.complex128)
        gradient[3:6] = matrix_gradient[
            (15, 16, 17), (0, 1, 2)
        ]
        gradient[6:9] = 0.1 * matrix_gradient[
            (18, 19, 20), (0, 1, 2)
        ]
        return gradient

    @staticmethod
    def horizontal_projection(
        u, gradient, representative, rank_rtol=None
    ):
        del u, representative, rank_rtol
        projected = np.asarray(
            gradient, dtype=np.complex128
        ).copy()
        projected[:3] = 0
        return {
            "projected_gradient": projected,
            "pivot_jacobian_rank": 3,
            "vertical_rank": 0,
            "horizontal_rank": 87,
            "pivot_tangency_error": 0.0,
            "vertical_orthogonality_error": 0.0,
        }

    @staticmethod
    def path_diagnostics(u, representative, y=None):
        del representative
        q = complex(u[3])
        if y is not None:
            q *= complex(y[0, 0])
        return {
            "pole_quantity_Q": [q.real, q.imag],
            "full_color_gauge_invariant": False,
            "chart_path_dependent": True,
            "q_character": [
                -1, 1, 0, -1, 1, 0, 0, 1, -1,
                0, 0, 0, 0, 1, -1,
            ],
            "pivot_character_rank": 3,
            "rank_after_adjoining_q_character": 4,
        }

    @staticmethod
    def reconstruct_full_weights(u, y):
        result = np.zeros(135, dtype=np.complex128)
        result[:90] = np.asarray(u)
        result[90:] = np.asarray(y).reshape(-1)
        return result

    @classmethod
    def dual_full_residual(cls, u, y):
        residual = cls.evaluate_star_matrix(u) @ y - cls.targets()
        l2 = float(np.linalg.norm(residual))
        return {
            "all_729_replayed": True,
            "all_729_equations_replayed": True,
            "primary_and_independent_outputs_agree": True,
            "primary": {"residual_l2": l2},
            "independent": {"residual_l2": l2},
            "primary_l2": l2,
            "independent_l2": l2,
            "maximum_output_difference": 0.0,
        }

    @staticmethod
    def source_fingerprints():
        return {
            "fake_core": "deterministic-test-double-v1",
            "exact_star_replay": True,
        }


def _canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _load(path: Path):
    return json.loads(path.read_text(encoding="ascii"))


def _write(path: Path, payload) -> None:
    path.write_bytes(_canonical_json_bytes(payload))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_artifact_record(directory: Path) -> None:
    summary_path = directory / artifact_module.SUMMARY_FILE
    manifest_path = directory / artifact_module.MANIFEST_FILE
    manifest = _load(manifest_path)
    manifest["artifact"] = {
        "path": summary_path.name,
        "bytes": summary_path.stat().st_size,
        "sha256": _sha256(summary_path),
    }
    _write(manifest_path, manifest)


class StarVariableProjectionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core_module_name = "_fake_krenn_vp_artifact_core"
        cls.original_core_module = artifact_module.DEFAULT_CORE_MODULE
        module = types.ModuleType(cls.core_module_name)
        fake = _FakeCore()
        for name in (
            "pivot_representatives",
            "targets",
            "initial_point",
            "pivot_values",
            "retract_pivots",
            "balance_residual_gauge",
            "evaluate_star_matrix",
            "pullback_gradient",
            "horizontal_projection",
            "path_diagnostics",
            "reconstruct_full_weights",
            "dual_full_residual",
            "source_fingerprints",
        ):
            setattr(module, name, getattr(fake, name))
        sys.modules[cls.core_module_name] = module
        artifact_module.DEFAULT_CORE_MODULE = cls.core_module_name
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="krenn-vp-artifact-"
        )
        cls.root = Path(cls.temporary.name)
        cls.scratch = cls.root / "scratch"
        cls.bundle = cls.root / "bundle"
        manifest = cls._write_fixture_campaign()
        if manifest["trajectory_count"] != 3:
            raise AssertionError("tiny campaign should have three orbits")
        generate_variable_projection_summary_bundle(
            cls.scratch, cls.bundle
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        artifact_module.DEFAULT_CORE_MODULE = cls.original_core_module
        sys.modules.pop(cls.core_module_name, None)

    @classmethod
    def _write_fixture_campaign(cls):
        cls.scratch.mkdir()
        core = sys.modules[cls.core_module_name]
        radius = 2.0
        seed = 2026072501
        rows = []
        for orbit_index in range(3):
            config = TrajectoryConfig(
                orbit_index=orbit_index,
                seed=seed,
                caps=RadiusCaps.from_radius(radius),
                initialization="cold",
                maximum_iterations=1,
                maximum_seconds=60,
                patience=3,
                checkpoint_interval=1,
            )
            representative = core.pivot_representatives()[orbit_index]
            u = core.initial_point(representative, seed)
            evaluation = evaluate_projection(
                core,
                u,
                representative,
                config.caps,
                rank_rtol=config.rank_rtol,
            )
            initialization = {
                "kind": "synthetic-unit-fixture",
                "orbit_index": orbit_index,
                "seed": seed,
            }
            result = campaign_module._result_payload(
                core=core,
                representative=representative,
                config=config,
                status="maximum-iterations",
                iteration=1,
                accepted_steps=0,
                patience_count=0,
                elapsed=0.25 + orbit_index,
                best_u=u,
                best_evaluation=evaluation,
                trace=[],
                initialization=initialization,
                fingerprints=core.source_fingerprints(),
            )
            # Model the completed pre-terminology-audit campaign files.
            result.pop("payload_sha256")
            result["claim_boundary"] = dict(
                artifact_module._ARCHIVED_RESULT_BOUNDARY
            )
            result["finite_accounting"].pop(
                "raw_u_overflow_cutoffs_are_enforced"
            )
            result["finite_accounting"].pop(
                "raw_u_overflow_cutoffs_depend_on_y_radius"
            )
            result["finite_accounting"].pop(
                "raw_u_overflow_cutoffs_are_mathematical_bounds"
            )
            result["independent_verification"].pop(
                "optimizer_full_system_cross_check"
            )
            checkpoint = campaign_module._checkpoint_payload(
                config=config,
                fingerprints=core.source_fingerprints(),
                status="maximum-iterations",
                iteration=1,
                accepted_steps=0,
                patience_count=0,
                cumulative_elapsed=0.25 + orbit_index,
                u=u,
                best_u=u,
                evaluation=evaluation,
                best_evaluation=evaluation,
                history=[],
                cycle_states=[u],
                trace=[],
                initialization=initialization,
            )
            stem = f"orbit{orbit_index}_seed{seed}_radius2_cold"
            result_name = stem + ".result.json"
            checkpoint_name = stem + ".checkpoint.json"
            _write(cls.scratch / result_name, result)
            _write(cls.scratch / checkpoint_name, checkpoint)
            summary = evaluation.summary()
            rows.append({
                "orbit_index": orbit_index,
                "seed": seed,
                "radius": radius,
                "initialization": "cold",
                "result_path": result_name,
                "termination": "maximum-iterations",
                "best_objective": summary["objective"],
                "best_raw_span_objective": summary[
                    "raw_span_objective"
                ],
                "minimum_norm_y_frobenius": summary[
                    "minimum_norm_y_frobenius"
                ],
                "bounded_y_frobenius": summary[
                    "bounded_y_frobenius"
                ],
                "y_cap_active": summary["y_cap_active"],
            })
        rows.sort(key=lambda row: row["orbit_index"])
        manifest = {
            "schema": CAMPAIGN_SCHEMA,
            "scratch_root": str(cls.scratch.resolve()),
            "core_module": cls.core_module_name,
            "orbit_cover": {
                "fixed_apex": 0,
                "representative_count": 3,
                "orbit_indices": [0, 1, 2],
                "orbit_sizes": [5, 60, 60],
                "provenance": (
                    "S5 residual vertices x S3 colors target-minor pivots"
                ),
                "legacy_eight_seed_charts_used": False,
                "hard_coordinate_anchors_used": False,
            },
            "gauge_control": {
                "selected_pivots_root_free_normalized_to_one": True,
                "residual_real_torus_balanced_after_each_retraction":
                    False,
                "residual_norm_balancing_refused_by_exact_recession":
                    True,
                "horizontal_residual_gauge_projection": True,
                "raw_u_norm_mathematical_cap": False,
                "raw_y_norm_full_gauge_invariant": False,
                "radius_controls": (
                    "bounded star-block Frobenius norm in the "
                    "horizontal P=1 lift"
                ),
                "known_pole_quantity_Q_full_color_gauge_invariant":
                    False,
                "known_pole_quantity_Q_used_as_cap": False,
                "full_135_weight_norm_is_bounded_by_search_domain":
                    False,
                "known_divergent_pole_excluded_by_search_domain": False,
            },
            "radii": [radius],
            "seeds": [seed],
            "initializations": ["cold"],
            "natural_repair_scale": 1e-2,
            "workers": 1,
            "maximum_workers": MAXIMUM_WORKERS,
            "maximum_iterations_per_trajectory": 1,
            "maximum_seconds_per_trajectory": 60.0,
            "patience": 3,
            "checkpoint_interval": 1,
            "trajectory_count": len(rows),
            "expected_trajectory_count": len(rows),
            "results": rows,
            "best_p_equals_one_y_bounded_candidate": min(
                rows, key=lambda row: row["best_objective"]
            ),
            "claim_boundary": dict(
                artifact_module._ARCHIVED_CAMPAIGN_BOUNDARY
            ),
        }
        _write(cls.scratch / "manifest.json", manifest)
        return manifest

    def _scratch_copy(self, name: str) -> Path:
        copied = self.root / name
        shutil.copytree(self.scratch, copied)
        manifest_path = copied / "manifest.json"
        manifest = _load(manifest_path)
        manifest["scratch_root"] = str(copied.resolve())
        _write(manifest_path, manifest)
        return copied

    def _bundle_copy(self, name: str) -> Path:
        copied = self.root / name
        shutil.copytree(self.bundle, copied)
        return copied

    def test_round_trip_is_portable_and_replays_selected_candidates(self):
        hidden = self.root / "scratch-hidden"
        self.scratch.rename(hidden)
        try:
            verified = verify_variable_projection_summary_bundle(
                self.bundle
            )
        finally:
            hidden.rename(self.scratch)
        self.assertTrue(verified.valid)
        self.assertEqual(
            {name for name, passed in verified.checks if passed},
            {
                "bundle_inventory_exact",
                "bundle_hash_replayed",
                "source_hashes_replayed",
                "scratch_ledger_digest_replayed",
                "compact_trajectory_aggregates_replayed",
                "selected_projection_evaluations_replayed",
                "selected_135_weight_reconstructions_replayed",
                "selected_primary_independent_729_replays_agree",
                "portable_without_scratch",
                "claim_boundaries_fail_closed",
            },
        )
        summary = verified.summary
        self.assertEqual(len(summary["trajectory_rows"]), 3)
        self.assertEqual(
            len(
                summary["portable_evidence"]["selected_candidates"]
            ),
            3,
        )
        self.assertNotIn(
            '"trace":',
            _canonical_json_bytes(summary).decode("ascii"),
        )
        self.assertFalse(
            summary["verification"][
                "raw_traces_copied_into_portable_bundle"
            ]
        )

    def test_claims_are_only_p_equals_one_y_bounded_reconnaissance(self):
        summary = _load(self.bundle / artifact_module.SUMMARY_FILE)
        claims = summary["claim_boundary"]
        self.assertIn("P=1", claims["search_scope"])
        self.assertFalse(
            claims["bounded_y_is_full_color_gauge_invariant"]
        )
        self.assertFalse(
            claims["full_135_norm_bounded_search_completed"]
        )
        self.assertFalse(
            claims["known_divergent_pole_excluded_by_search_domain"]
        )
        self.assertFalse(claims["numerical_zero_is_exact_counterexample"])
        self.assertFalse(claims["bounded_miss_is_nonexistence_proof"])
        self.assertEqual(
            claims["finite_affine_membership_status"], "undecided"
        )
        self.assertEqual(
            summary["aggregate"]["exact_verification"],
            {
                "attempted_trajectories": 0,
                "finite_exact_witnesses_verified": 0,
            },
        )

    def test_wait_refuses_to_read_partial_scratch(self):
        partial = self.root / "partial"
        partial.mkdir()
        (partial / "unrelated.partial").write_text(
            "not json", encoding="ascii"
        )
        with mock.patch.object(
            artifact_module,
            "_load_json",
            side_effect=AssertionError("partial file was read"),
        ):
            with self.assertRaisesRegex(
                KrennVariableProjectionArtifactError,
                "terminal manifest.json is not present",
            ):
                wait_for_terminal_manifest(
                    partial,
                    maximum_wait_seconds=0,
                )

    def test_scratch_result_corruption_is_rejected(self):
        copied = self._scratch_copy("scratch-corrupt-result")
        campaign = _load(copied / "manifest.json")
        result_path = copied / campaign["results"][0]["result_path"]
        result = _load(result_path)
        result["best_full_weights"][0][0] += 1.0
        _write(result_path, result)
        with self.assertRaisesRegex(
            KrennVariableProjectionArtifactError,
            "full 135-weight reconstruction failed replay",
        ):
            summarize_completed_campaign(copied)

    def test_checkpoint_payload_corruption_is_rejected(self):
        copied = self._scratch_copy("scratch-corrupt-checkpoint")
        campaign = _load(copied / "manifest.json")
        result_name = campaign["results"][0]["result_path"]
        checkpoint_name = result_name.replace(
            ".result.json", ".checkpoint.json"
        )
        checkpoint_path = copied / checkpoint_name
        checkpoint = _load(checkpoint_path)
        checkpoint["best_u"][3][0] += 1.0
        _write(checkpoint_path, checkpoint)
        with self.assertRaisesRegex(
            KrennVariableProjectionArtifactError,
            "result/checkpoint schema or payload hash changed",
        ):
            summarize_completed_campaign(copied)

    def test_refreshed_candidate_corruption_fails_fresh_replay(self):
        copied = self._bundle_copy("bundle-corrupt-candidate")
        summary_path = copied / artifact_module.SUMMARY_FILE
        summary = _load(summary_path)
        candidate = summary["portable_evidence"][
            "selected_candidates"
        ][0]
        candidate["best_u"][3][0] += 0.25
        _write(summary_path, summary)
        _refresh_artifact_record(copied)
        with self.assertRaises(
            KrennVariableProjectionArtifactError
        ):
            verify_variable_projection_summary_bundle(copied)

    def test_refreshed_claim_escalation_is_rejected(self):
        copied = self._bundle_copy("bundle-escalated-claim")
        summary_path = copied / artifact_module.SUMMARY_FILE
        summary = _load(summary_path)
        summary["claim_boundary"][
            "numerical_zero_is_exact_counterexample"
        ] = True
        _write(summary_path, summary)
        _refresh_artifact_record(copied)
        with self.assertRaisesRegex(
            KrennVariableProjectionArtifactError,
            "static fields or claims changed",
        ):
            verify_variable_projection_summary_bundle(copied)

    def test_extra_bundle_file_is_rejected(self):
        copied = self._bundle_copy("bundle-extra-file")
        (copied / "unreviewed.json").write_text(
            "{}\n", encoding="ascii"
        )
        with self.assertRaisesRegex(
            KrennVariableProjectionArtifactError,
            "inventory changed",
        ):
            verify_variable_projection_summary_bundle(copied)


if __name__ == "__main__":
    unittest.main()
