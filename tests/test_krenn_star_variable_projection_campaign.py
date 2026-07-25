"""Fast tests for the bounded star variable-projection campaign."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

try:
    from experiments.krenn_quantum_graph import (
        star_variable_projection as production_core,
    )
    from experiments.krenn_quantum_graph import (
        star_variable_projection_campaign as campaign_module,
    )
    from experiments.krenn_quantum_graph.star_variable_projection_campaign import (
        KrennVariableProjectionCampaignError,
        RadiusCaps,
        TrajectoryConfig,
        bounded_y_solve,
        run_campaign,
        run_trajectory,
        validate_workers,
    )
except ImportError:
    import star_variable_projection as production_core
    import star_variable_projection_campaign as campaign_module
    from star_variable_projection_campaign import (
        KrennVariableProjectionCampaignError,
        RadiusCaps,
        TrajectoryConfig,
        bounded_y_solve,
        run_campaign,
        run_trajectory,
        validate_workers,
    )


@dataclass(frozen=True)
class _FakeRepresentative:
    index: int


class _FakeCore:
    """Small deterministic core with production-compatible shapes."""

    __name__ = "fake_krenn_variable_projection_core"

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

    @classmethod
    def natural_repair_initial_point(
        cls, representative, seed, perturbation_scale=1e-2
    ):
        if representative.index != 2 or perturbation_scale <= 0:
            raise ValueError("natural repair requires orbit two")
        return cls.initial_point(representative, seed)

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
            # Legacy flat fields remain accepted by the replay reader.
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


class BoundedYTests(unittest.TestCase):
    def test_inactive_and_active_svd_trust_region(self):
        generator = np.random.default_rng(20260725)
        matrix = (
            generator.normal(size=(30, 7))
            + 1j * generator.normal(size=(30, 7))
        )
        target = (
            generator.normal(size=(30, 3))
            + 1j * generator.normal(size=(30, 3))
        )
        unconstrained = bounded_y_solve(
            matrix, target, 1e6, rank_rtol=1e-13
        )
        expected = np.linalg.lstsq(
            matrix, target, rcond=1e-13
        )[0]
        self.assertFalse(unconstrained["cap_active"])
        np.testing.assert_allclose(
            unconstrained["y"], expected, rtol=2e-12, atol=2e-12
        )

        cap = float(np.linalg.norm(expected)) / 3
        bounded = bounded_y_solve(
            matrix, target, cap, rank_rtol=1e-13
        )
        self.assertTrue(bounded["cap_active"])
        self.assertGreater(bounded["lagrange_multiplier"], 0)
        self.assertAlmostEqual(
            float(np.linalg.norm(bounded["y"])), cap, places=11
        )
        self.assertGreaterEqual(
            bounded["objective"], bounded["raw_objective"]
        )
        # KKT: A^*(A Y-E)+lambda Y = 0.
        kkt = (
            matrix.conj().T @ bounded["residual"]
            + bounded["lagrange_multiplier"] * bounded["y"]
        )
        self.assertLess(np.linalg.norm(kkt), 2e-9)

    def test_rank_deficient_solve_is_finite(self):
        matrix = np.zeros((20, 15), dtype=np.complex128)
        matrix[:, :3] = 1
        target = np.eye(20, 3, dtype=np.complex128)
        result = bounded_y_solve(matrix, target, 2.0)
        self.assertEqual(result["rank"], 1)
        self.assertTrue(np.isfinite(result["objective"]))
        self.assertTrue(np.all(np.isfinite(result["y"])))

    def test_bounded_solve_keeps_nonzero_below_rank_threshold(self):
        matrix = np.diag([1.0, 1.0e-8]).astype(np.complex128)
        target = np.asarray([[0.0], [1.0]], dtype=np.complex128)
        result = bounded_y_solve(
            matrix,
            target,
            1.0e4,
            rank_rtol=1.0e-6,
        )
        self.assertEqual(result["rank"], 1)
        self.assertTrue(result["cap_active"])
        self.assertGreater(abs(result["y"][1, 0]), 9.0e3)
        self.assertLess(result["objective"], 0.9999)
        kkt = (
            matrix.conj().T @ result["residual"]
            + result["lagrange_multiplier"] * result["y"]
        )
        self.assertLess(np.linalg.norm(kkt), 1.0e-11)

        # Envelope derivative of the bounded optimum with respect to Phi.
        direction = np.zeros_like(matrix)
        direction[1, 1] = 1.0
        step = 1.0e-11
        plus = bounded_y_solve(
            matrix + step * direction,
            target,
            1.0e4,
            rank_rtol=1.0e-6,
        )["objective"]
        minus = bounded_y_solve(
            matrix - step * direction,
            target,
            1.0e4,
            rank_rtol=1.0e-6,
        )["objective"]
        finite_difference = (plus - minus) / (2.0 * step)
        predicted = 2.0 * np.vdot(
            result["residual"],
            direction @ result["y"],
        ).real
        self.assertAlmostEqual(
            finite_difference,
            predicted,
            delta=2.0e-4 + 2.0e-7 * abs(predicted),
        )

    def test_subnormal_multiplier_and_unconstrained_overflow(self):
        smallest = float(
            np.nextafter(np.float64(0.0), np.float64(1.0))
        )
        matrix = np.asarray([[smallest]], dtype=np.complex128)
        target = np.asarray([[1.0]], dtype=np.complex128)
        result = bounded_y_solve(matrix, target, 16.0)

        self.assertTrue(result["cap_active"])
        self.assertAlmostEqual(
            float(np.linalg.norm(result["y"])), 16.0, delta=3.0e-10
        )
        self.assertIsNone(result["unconstrained_y"])
        self.assertIsNone(result["minimum_norm_y_frobenius"])
        self.assertFalse(result["minimum_norm_y_array_available"])
        self.assertTrue(result["minimum_norm_y_overflowed_float64"])
        self.assertGreater(
            result["minimum_norm_y_log10_frobenius"], 323.0
        )
        self.assertIsNone(result["lagrange_multiplier"])
        self.assertFalse(
            result["lagrange_multiplier_float64_available"]
        )
        self.assertLess(result["lagrange_multiplier_log10"], -323.0)
        self.assertLess(result["cap_saturation_abs_error"], 3.0e-10)
        self.assertTrue(np.isfinite(result["kkt_residual_l2"]))
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if not isinstance(value, np.ndarray)
            },
            allow_nan=False,
        )

    def test_sigma_squared_underflow_does_not_change_the_solve(self):
        matrix = np.diag([1.0, 1.0e-200]).astype(np.complex128)
        inactive_target = np.asarray(
            [[0.0], [1.0e-300]], dtype=np.complex128
        )
        inactive = bounded_y_solve(matrix, inactive_target, 2.0)
        self.assertFalse(inactive["cap_active"])
        self.assertAlmostEqual(
            abs(inactive["y"][1, 0]), 1.0e-100, delta=1.0e-114
        )
        self.assertEqual(inactive["lagrange_multiplier"], 0.0)
        self.assertTrue(
            inactive["lagrange_multiplier_float64_available"]
        )

        active_target = np.asarray(
            [[0.0], [1.0]], dtype=np.complex128
        )
        active = bounded_y_solve(matrix, active_target, 4.0)
        self.assertTrue(active["cap_active"])
        self.assertAlmostEqual(
            float(np.linalg.norm(active["y"])), 4.0, delta=1.0e-10
        )
        self.assertIsNotNone(active["lagrange_multiplier"])
        self.assertGreater(active["lagrange_multiplier"], 0.0)
        self.assertLess(active["kkt_relative_residual"], 1.0e-12)

    def test_subnormal_complex_magnitude_controls_cap_classification(self):
        smallest = float(
            np.nextafter(np.float64(0.0), np.float64(1.0))
        )
        matrix = np.asarray([[smallest]], dtype=np.complex128)
        target = np.asarray(
            [[smallest + 1j * smallest]], dtype=np.complex128
        )

        inactive = bounded_y_solve(matrix, target, 2.0)
        self.assertFalse(inactive["cap_active"])
        np.testing.assert_allclose(
            inactive["y"],
            np.asarray([[1.0 + 1.0j]]),
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        self.assertAlmostEqual(
            inactive["minimum_norm_y_frobenius"],
            math.sqrt(2.0),
            delta=2.0e-13,
        )

        cap = 1.2
        active = bounded_y_solve(matrix, target, cap)
        self.assertTrue(active["cap_active"])
        self.assertAlmostEqual(
            float(np.linalg.norm(active["y"])), cap, delta=3.0e-12
        )
        expected = cap / math.sqrt(2.0)
        self.assertAlmostEqual(
            active["y"][0, 0].real, expected, delta=3.0e-12
        )
        self.assertAlmostEqual(
            active["y"][0, 0].imag, expected, delta=3.0e-12
        )
        self.assertLess(active["cap_saturation_abs_error"], 3.0e-12)


class ConfigurationTests(unittest.TestCase):
    def test_worker_limit_and_radius_semantics(self):
        self.assertEqual(validate_workers(1), 1)
        self.assertEqual(validate_workers(8), 8)
        for invalid in (0, 9, True):
            with self.assertRaises(
                KrennVariableProjectionCampaignError
            ):
                validate_workers(invalid)
        caps = RadiusCaps.from_radius(4)
        self.assertEqual(caps.y_frobenius, 4)
        self.assertGreater(caps.u_linf, 1e35)
        self.assertGreater(caps.u_l2, 1e35)
        larger = RadiusCaps.from_radius(8)
        self.assertLess(larger.u_linf, caps.u_linf)
        self.assertLess(larger.u_l2, caps.u_l2)

    def test_config_defaults_to_ten_minutes(self):
        config = TrajectoryConfig(
            orbit_index=0,
            seed=1,
            caps=RadiusCaps.from_radius(2),
        )
        self.assertEqual(config.maximum_seconds, 600)
        self.assertEqual(config.lbfgs_memory, 10)
        for invalid_seed in (True, 1.0, 1.5):
            with self.assertRaises(
                KrennVariableProjectionCampaignError
            ):
                TrajectoryConfig(
                    orbit_index=0,
                    seed=invalid_seed,
                    caps=RadiusCaps.from_radius(2),
                )


class TrajectoryTests(unittest.TestCase):
    def _config(self, *, initialization="cold"):
        return TrajectoryConfig(
            orbit_index=1,
            seed=2026072501,
            caps=RadiusCaps.from_radius(2),
            initialization=initialization,
            maximum_iterations=4,
            maximum_seconds=60,
            patience=8,
            residual_tolerance=1e-15,
            gradient_tolerance=1e-15,
        )

    def test_trajectory_retracts_refuses_balance_and_replays(self):
        core = _FakeCore()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trajectory.json"
            result = run_trajectory(
                self._config(),
                checkpoint_path=checkpoint,
                core=core,
            )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(
                result["independent_verification"][
                    "all_729_replayed"
                ]
            )
            self.assertFalse(
                result["claim_boundary"][
                    "raw_u_norm_used_as_mathematical_cap"
                ]
            )
            self.assertFalse(
                result["claim_boundary"][
                    "pole_quantity_Q_is_full_color_gauge_invariant"
                ]
            )
            self.assertFalse(
                result["claim_boundary"]["pole_quantity_Q_used_as_cap"]
            )
            self.assertTrue(
                result["claim_boundary"][
                    "bounded_y_is_only_a_P_equals_1_slice_control"
                ]
            )
            if result["trace"]:
                row = result["trace"][0]
                if row["backtracks"] == 0:
                    self.assertIsNone(
                        row["rejection_before_acceptance"]
                    )
                self.assertIn("retraction", row)
                self.assertIn(
                    "residual_balance", row["retraction"]
                )
                self.assertFalse(
                    row["retraction"]["residual_balance"]["attempted"]
                )
                self.assertEqual(
                    row["retraction"]["residual_balance"]["status"],
                    "refused-by-exact-recession-audit",
                )
                self.assertFalse(row["invariant_cap_enabled"])
                self.assertEqual(
                    row["backtrack_rejection_counts"][
                        "invariant-cap"
                    ],
                    0,
                )
                self.assertIn(
                    "numerical_two_cycle_diagnostic", row
                )
                self.assertIn(
                    "numerical_two_cycle_detected", row
                )
                self.assertIn(
                    "alternating_cap_with_nontrivial_retraction_detected",
                    row,
                )
            payload = json.loads(checkpoint.read_text("ascii"))
            self.assertIn("payload_sha256", payload)
            self.assertGreaterEqual(len(payload["cycle_states"]), 1)
            self.assertLessEqual(len(payload["cycle_states"]), 2)
            checkpoint_u = np.asarray(payload["u"], dtype=np.float64)
            checkpoint_u = checkpoint_u[:, 0] + 1j * checkpoint_u[:, 1]
            last_cycle = np.asarray(
                payload["cycle_states"][-1], dtype=np.float64
            )
            last_cycle = last_cycle[:, 0] + 1j * last_cycle[:, 1]
            np.testing.assert_allclose(
                last_cycle,
                checkpoint_u,
                rtol=0,
                atol=0,
            )
            if result["trace"]:
                self.assertEqual(len(payload["cycle_states"]), 2)
                self.assertIn(
                    "numerical_two_cycle_diagnostic",
                    result["trace"][-1],
                )
                self.assertIn(
                    "numerical_two_cycle_detected",
                    result["trace"][-1],
                )

    def test_full_replay_must_match_optimized_objective(self):
        class MismatchedReplayCore(_FakeCore):
            @classmethod
            def dual_full_residual(cls, u, y):
                replay = super().dual_full_residual(u, y)
                replay["primary"]["residual_l2"] += 0.25
                replay["independent"]["residual_l2"] += 0.25
                return replay

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(
                KrennVariableProjectionCampaignError
            ):
                run_trajectory(
                    self._config(),
                    checkpoint_path=Path(directory) / "mismatch.json",
                    core=MismatchedReplayCore(),
                )

    def test_real_core_one_step_replays_and_checkpoints(self):
        config = TrajectoryConfig(
            orbit_index=0,
            seed=2026072501,
            caps=RadiusCaps.from_radius(2),
            maximum_iterations=1,
            maximum_seconds=120,
            patience=3,
            residual_tolerance=1.0e-30,
            gradient_tolerance=1.0e-30,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "real-core-one-step.json"
            result = run_trajectory(
                config,
                checkpoint_path=checkpoint,
                core=production_core,
            )
            self.assertEqual(result["iterations_attempted"], 1)
            replay = result["independent_verification"]
            self.assertTrue(replay["all_729_replayed"])
            self.assertTrue(replay["all_729_equations_replayed"])
            self.assertTrue(
                replay["primary_and_independent_outputs_agree"]
            )
            self.assertTrue(
                replay["optimizer_full_system_cross_check"][
                    "optimizer_and_full_system_agree"
                ]
            )
            self.assertFalse(
                result["initialization"]["retraction"][
                    "residual_balance"
                ]["attempted"]
            )
            self.assertIn(
                "core_source_sha256", result["source_fingerprints"]
            )
            payload = json.loads(checkpoint.read_text("ascii"))
            json.dumps(payload, sort_keys=True, allow_nan=False)
            self.assertGreaterEqual(len(payload["cycle_states"]), 1)
            self.assertLessEqual(len(payload["cycle_states"]), 2)

    def test_terminal_resume_and_corruption_rejection(self):
        core = _FakeCore()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trajectory.json"
            first = run_trajectory(
                self._config(),
                checkpoint_path=checkpoint,
                core=core,
            )
            resumed = run_trajectory(
                self._config(),
                checkpoint_path=checkpoint,
                core=core,
                resume=True,
            )
            self.assertEqual(
                first["termination"], resumed["termination"]
            )
            self.assertEqual(
                first["best_evaluation"]["objective"],
                resumed["best_evaluation"]["objective"],
            )
            payload = json.loads(checkpoint.read_text("ascii"))
            payload["iteration"] += 1
            checkpoint.write_text(
                json.dumps(payload), encoding="ascii"
            )
            with self.assertRaises(
                KrennVariableProjectionCampaignError
            ):
                run_trajectory(
                    self._config(),
                    checkpoint_path=checkpoint,
                    core=core,
                    resume=True,
                )


class CampaignTests(unittest.TestCase):
    def test_campaign_rejects_boolean_and_float_seeds_before_dispatch(self):
        for invalid_seed in (True, 1.0, 1.5):
            with self.subTest(seed=invalid_seed):
                with tempfile.TemporaryDirectory() as directory:
                    with mock.patch.object(
                        campaign_module,
                        "_chain_worker",
                        side_effect=AssertionError(
                            "invalid seed reached trajectory dispatch"
                        ),
                    ):
                        with self.assertRaises(
                            KrennVariableProjectionCampaignError
                        ):
                            run_campaign(
                                scratch_root=Path(directory),
                                radii=(2.0,),
                                seeds=(invalid_seed,),
                                workers=1,
                                maximum_iterations=1,
                                maximum_seconds_per_trajectory=60,
                                patience=2,
                            )

    def test_three_orbits_cold_and_warm_ladder(self):
        module_name = "_fake_krenn_vp_campaign_core"
        module = types.ModuleType(module_name)
        fake = _FakeCore()
        for name in (
            "pivot_representatives",
            "targets",
            "initial_point",
            "natural_repair_initial_point",
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
        sys.modules[module_name] = module
        try:
            with tempfile.TemporaryDirectory() as directory:
                manifest = run_campaign(
                    scratch_root=Path(directory),
                    core_module=module_name,
                    radii=(2.0, 4.0),
                    seeds=(2026072501,),
                    workers=1,
                    maximum_iterations=1,
                    maximum_seconds_per_trajectory=60,
                    patience=3,
                )
                self.assertEqual(
                    manifest["trajectory_count"], 9
                )
                self.assertEqual(
                    manifest["expected_trajectory_count"], 9
                )
                self.assertEqual(
                    {
                        row["orbit_index"]
                        for row in manifest["results"]
                    },
                    {0, 1, 2},
                )
                self.assertEqual(
                    sum(
                        row["initialization"] == "warm"
                        for row in manifest["results"]
                    ),
                    3,
                )
                self.assertFalse(
                    manifest["orbit_cover"][
                        "legacy_eight_seed_charts_used"
                    ]
                )
                self.assertFalse(
                    manifest["gauge_control"][
                        "known_pole_quantity_Q_used_as_cap"
                    ]
                )
                self.assertFalse(
                    manifest["gauge_control"][
                        "residual_real_torus_balanced_after_each_retraction"
                    ]
                )
                self.assertTrue(
                    manifest["gauge_control"][
                        "residual_norm_balancing_refused_by_exact_recession"
                    ]
                )
                self.assertIn(
                    "smallest_recorded_lift_dependent_candidate",
                    manifest,
                )
                self.assertIn(
                    "best_p_equals_one_y_bounded_candidate",
                    manifest,
                )
                self.assertFalse(
                    manifest[
                        "cross_run_nonzero_objectives_are_intrinsically_comparable"
                    ]
                )
                self.assertFalse(
                    manifest["gauge_control"][
                        "fixed_global_residual_gauge_slice"
                    ]
                )
                self.assertTrue(
                    manifest["gauge_control"][
                        "raw_u_overflow_cutoff_enforced"
                    ]
                )
            with tempfile.TemporaryDirectory() as directory:
                natural_manifest = run_campaign(
                    scratch_root=Path(directory),
                    core_module=module_name,
                    radii=(2.0,),
                    seeds=(2026072501,),
                    initializations=("natural-repair",),
                    workers=1,
                    maximum_iterations=1,
                    maximum_seconds_per_trajectory=60,
                    patience=3,
                )
                self.assertEqual(
                    natural_manifest["trajectory_count"], 1
                )
                self.assertEqual(
                    natural_manifest["results"][0]["orbit_index"], 2
                )
                self.assertEqual(
                    natural_manifest["results"][0]["initialization"],
                    "natural-repair",
                )
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
