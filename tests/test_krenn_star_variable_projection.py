"""Focused tests for the staged anchor-free variable-projection core."""

from __future__ import annotations

from fractions import Fraction
import json
import math
import unittest

import numpy as np

from experiments.krenn_quantum_graph import (
    star_variable_projection as vp,
)


class TestExactPivotGeometry(unittest.TestCase):
    def test_three_orbits_exact_kernel_and_root_free_cocharacters(self):
        representatives = vp.pivot_representatives()
        self.assertEqual(
            [
                (
                    rep.orbit_index,
                    rep.equality_pattern,
                    rep.partner_by_color,
                    rep.orbit_size,
                )
                for rep in representatives
            ],
            [
                (0, "all-same", (1, 1, 1), 5),
                (1, "exactly-two-same", (1, 1, 2), 60),
                (2, "all-distinct", (1, 2, 3), 60),
            ],
        )
        self.assertEqual(sum(rep.orbit_size for rep in representatives), 125)

        for rep in representatives:
            characters = rep.factor_characters
            kernel = vp.gauge_kernel(rep)
            self.assertEqual(len(kernel), 15)
            self.assertTrue(all(len(row) == 12 for row in kernel))
            for factor in range(3):
                for basis in range(12):
                    self.assertEqual(
                        sum(
                            Fraction(characters[factor][parameter])
                            * kernel[parameter][basis]
                            for parameter in range(15)
                        ),
                        0,
                    )
            cocharacters = rep.normalization_cocharacters
            pairing = tuple(
                tuple(
                    sum(
                        characters[row][parameter]
                        * cocharacters[parameter][column]
                        for parameter in range(15)
                    )
                    for column in range(3)
                )
                for row in range(3)
            )
            self.assertEqual(
                pairing,
                ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
            )

    def test_exact_residual_l2_recession_is_refused(self):
        for rep in vp.pivot_representatives():
            audit = vp.residual_recession_audit(rep)
            self.assertFalse(
                audit[
                    "generic_dense_residual_L2_orbit_has_finite_minimizer"
                ]
            )
            self.assertFalse(
                audit[
                    "residual_L2_balancing_is_a_valid_mandatory_retraction"
                ]
            )
            self.assertEqual(len(audit["certificates"]), 3)
            for row in audit["certificates"]:
                self.assertEqual(
                    row["factor_character_pairing"], [0, 0, 0]
                )
                self.assertEqual(
                    row["positive_nonstar_exponent_count"], 0
                )
                self.assertGreater(
                    row["negative_nonstar_exponent_count"], 0
                )

            u = vp.initial_point(rep, 2026072600 + rep.orbit_index)
            with self.assertRaises(vp.KrennResidualGaugeBalanceError) as caught:
                vp.balance_residual_gauge(u, rep)
            diagnostics = caught.exception.diagnostics
            self.assertFalse(diagnostics["accepted_for_optimization"])
            self.assertTrue(
                diagnostics[
                    "no_finite_minimizer_certified_for_this_support"
                ]
            )

    def test_recession_preserves_pivots_and_strictly_reduces_dense_l2(self):
        rep = vp.pivot_representative(2)
        u = vp.initial_point(rep, 1107)
        row = vp.residual_recession_audit(rep)["certificates"][0]
        exponents = np.zeros(vp.NONSTAR_VARIABLES)
        exponents[np.asarray(row["affected_local_indices"])] = -1.0
        moved = u * np.exp(exponents)
        np.testing.assert_allclose(
            vp.pivot_values(moved, rep),
            np.ones(3),
            rtol=0,
            atol=2e-12,
        )
        self.assertLess(np.linalg.norm(moved), np.linalg.norm(u))

        zeroed = u.copy()
        zeroed[np.asarray(row["affected_local_indices"])] = 0
        np.testing.assert_allclose(
            vp.pivot_values(zeroed, rep),
            np.ones(3),
            rtol=0,
            atol=2e-12,
        )
        unchanged = zeroed * np.exp(exponents)
        np.testing.assert_array_equal(unchanged, zeroed)
        _, diagnostics = vp.balance_residual_gauge(
            zeroed, rep, strict=False
        )
        self.assertGreater(diagnostics["zero_coordinate_count"], 0)
        # The other two exact recession directions remain active.
        self.assertFalse(diagnostics["accepted_for_optimization"])


class TestVariableProjection(unittest.TestCase):
    def setUp(self):
        self.rep = vp.pivot_representative(2)
        self.u = vp.initial_point(self.rep, 2026072501)

    def test_matrix_targets_raw_and_ridge_shapes(self):
        matrix = vp.evaluate_star_matrix(self.u)
        self.assertEqual(matrix.shape, (243, 15))
        self.assertEqual(vp.targets().shape, (243, 3))
        np.testing.assert_allclose(
            matrix[
                np.asarray(vp.TARGET_ROWS),
                np.asarray(self.rep.columns_by_color),
            ],
            np.ones(3),
            rtol=0,
            atol=2e-12,
        )

        raw = vp.raw_variable_projection(self.u)
        ridge = vp.ridge_variable_projection(self.u, 1e-5)
        self.assertGreaterEqual(raw.objective, 0)
        self.assertLessEqual(raw.objective, 3.0 + 1e-12)
        self.assertEqual(raw.y.shape, (15, 3))
        self.assertEqual(ridge.y.shape, (15, 3))
        self.assertEqual(ridge.gradient.shape, (90,))
        self.assertTrue(
            ridge.smooth_gradient_certified_for_this_evaluation
        )
        self.assertGreaterEqual(ridge.objective, ridge.residual_squared)
        self.assertEqual(len(ridge.singular_values), 15)
        self.assertTrue(ridge.summary()["numerical_nonproof"])

    def test_analytic_gradient_matches_real_and_imaginary_differences(self):
        mu = 1e-4
        result = vp.ridge_variable_projection(self.u, mu)
        step = 1e-6
        for index in (0, 17, 44, 89):
            basis = np.zeros(vp.NONSTAR_VARIABLES, dtype=np.complex128)
            basis[index] = 1
            real_fd = (
                vp.ridge_variable_projection(
                    self.u + step * basis, mu
                ).objective
                - vp.ridge_variable_projection(
                    self.u - step * basis, mu
                ).objective
            ) / (2 * step)
            imag_fd = (
                vp.ridge_variable_projection(
                    self.u + 1j * step * basis, mu
                ).objective
                - vp.ridge_variable_projection(
                    self.u - 1j * step * basis, mu
                ).objective
            ) / (2 * step)
            self.assertAlmostEqual(
                real_fd,
                2.0 * result.gradient[index].real,
                delta=3e-7 + 3e-4 * abs(real_fd),
            )
            self.assertAlmostEqual(
                imag_fd,
                2.0 * result.gradient[index].imag,
                delta=3e-7 + 3e-4 * abs(imag_fd),
            )

        rng = np.random.default_rng(991)
        direction = (
            rng.normal(size=vp.NONSTAR_VARIABLES)
            + 1j * rng.normal(size=vp.NONSTAR_VARIABLES)
        )
        direction /= np.linalg.norm(direction)
        directional_fd = (
            vp.ridge_variable_projection(
                self.u + step * direction, mu
            ).objective
            - vp.ridge_variable_projection(
                self.u - step * direction, mu
            ).objective
        ) / (2 * step)
        predicted = 2.0 * np.vdot(
            result.gradient, direction
        ).real
        self.assertAlmostEqual(
            directional_fd,
            predicted,
            delta=5e-7 + 5e-4 * abs(directional_fd),
        )

    def test_horizontal_projection_is_tangent_orthogonal_and_descent(self):
        mu = 1e-5
        result = vp.ridge_variable_projection(self.u, mu)
        horizontal = vp.horizontal_projection(
            self.u, result.gradient, self.rep
        )
        self.assertLess(horizontal.tangent_residual_l2, 1e-10)
        self.assertLess(
            horizontal.residual_vertical_inner_product_l2, 1e-10
        )
        self.assertLess(horizontal.jp_times_t_minus_cp_l2, 1e-10)
        self.assertLess(horizontal.predicted_directional_derivative, 0)
        self.assertEqual(horizontal.constraint_rank, 15)

        step = 1e-5
        plus = vp.ridge_variable_projection(
            self.u + step * horizontal.direction, mu
        ).objective
        minus = vp.ridge_variable_projection(
            self.u - step * horizontal.direction, mu
        ).objective
        finite_difference = (plus - minus) / (2 * step)
        self.assertAlmostEqual(
            finite_difference,
            horizontal.predicted_directional_derivative,
            delta=2e-7 + 2e-4 * abs(finite_difference),
        )

        trial = self.u + step * horizontal.direction
        retracted, diagnostics = vp.retract_pivots(trial, self.rep)
        np.testing.assert_allclose(
            vp.pivot_values(retracted, self.rep),
            np.ones(3),
            rtol=0,
            atol=2e-10,
        )
        self.assertLess(diagnostics["raw_to_retracted_l2"], 1e-8)
        self.assertLess(
            vp.ridge_variable_projection(retracted, mu).objective,
            result.objective,
        )

    def test_zero_coordinates_do_not_create_an_anchor_requirement(self):
        # For the all-same chart, every selected pivot factor excludes
        # partner vertex 1.  A weight incident to that partner can be zero
        # without leaving the pivot open.
        rep = vp.pivot_representative(0)
        u = vp.initial_point(rep, 8128)
        recession = vp.residual_recession_audit(rep)["certificates"][0]
        u[recession["affected_local_indices"][0]] = 0
        np.testing.assert_allclose(
            vp.pivot_values(u, rep),
            np.ones(3),
            rtol=0,
            atol=2e-12,
        )
        ridge = vp.ridge_variable_projection(u, 1e-5)
        horizontal = vp.horizontal_projection(u, ridge.gradient, rep)
        self.assertTrue(np.all(np.isfinite(horizontal.direction)))
        self.assertFalse(
            horizontal.summary()["coordinate_anchor_used"]
        )

    def test_numerical_two_cycle_helper_is_scale_relative(self):
        direction = np.zeros(vp.NONSTAR_VARIABLES, dtype=np.complex128)
        direction[17] = 1.0 + 0.25j
        previous = self.u + 1.0e-5 * direction
        current = self.u + 1.0e-12 * direction
        detected = vp.detect_two_cycle(
            current,
            previous,
            self.u,
            relative_tolerance=1.0e-10,
        )
        self.assertTrue(detected["two_cycle_detected"])
        self.assertFalse(detected["cycle_is_exact_proof"])
        self.assertTrue(detected["numerical_nonproof"])

        not_detected = vp.detect_two_cycle(
            self.u + 1.0e-5 * direction,
            previous,
            self.u,
            relative_tolerance=1.0e-10,
        )
        self.assertFalse(not_detected["two_cycle_detected"])


class TestReconstructionAndLedger(unittest.TestCase):
    def setUp(self):
        self.rep = vp.pivot_representative(1)
        self.u = vp.initial_point(self.rep, 30303)

    def test_normal_form_enforces_BY_identity(self):
        result = vp.normal_form_diagnostic(
            self.u, self.rep, ridge_mu=1e-6
        )
        self.assertEqual(result.z.shape, (12, 3))
        self.assertEqual(result.y.shape, (15, 3))
        self.assertLess(
            np.max(
                np.abs(
                    result.full_residual[
                        np.asarray(vp.TARGET_ROWS)
                    ]
                )
            ),
            1e-10,
        )
        self.assertTrue(
            result.summary()[
                "target_rows_enforced_exactly_in_the_parametrization"
            ]
        )
        self.assertTrue(
            result.summary()[
                "y_or_z_bound_would_be_pivot_slice_dependent"
            ]
        )

    def test_reconstruction_and_independent_729_replay(self):
        ridge = vp.ridge_variable_projection(self.u, 1e-6)
        weights = vp.reconstruct_full_weights(self.u, ridge.y)
        self.assertEqual(weights.shape, (135,))
        replay = vp.dual_full_residual(self.u, ridge.y)
        self.assertTrue(replay["all_729_replayed"])
        self.assertTrue(replay["all_729_equations_replayed"])
        self.assertEqual(
            replay["all_729_replayed"],
            replay["all_729_equations_replayed"],
        )
        self.assertTrue(
            replay["primary_and_independent_outputs_agree"]
        )
        self.assertLess(
            replay[
                "primary_independent_output_agreement_max_abs"
            ],
            1e-10,
        )
        self.assertAlmostEqual(
            replay["primary"]["residual_l2"],
            math.sqrt(ridge.residual_squared),
            delta=2e-12,
        )
        self.assertFalse(replay["exact_verification_completed"])
        self.assertTrue(replay["numerical_nonproof"])

    def test_anchor_cross_check_requires_det_two_and_is_not_primary(self):
        for rep in vp.pivot_representatives():
            audit = vp.anchored_complement_diagnostic(rep)
            self.assertEqual(abs(audit["exact_determinant"]), 2)
            self.assertEqual(
                audit["smith_invariant_factors"], [1] * 14 + [2]
            )
            self.assertTrue(audit["diagnostic_only"])
            self.assertFalse(
                audit["coordinate_anchors_are_guaranteed_nonzero"]
            )
        self.assertFalse(
            audit["used_by_anchor_free_primary_search"]
        )

    def test_natural_repair_initializer_activates_all_missing_u(self):
        representative = vp.pivot_representative(2)
        first = vp.natural_repair_initial_point(
            representative,
            2026072501,
            perturbation_scale=1.0e-2,
        )
        second = vp.natural_repair_initial_point(
            representative,
            2026072501,
            perturbation_scale=1.0e-2,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (vp.NONSTAR_VARIABLES,))
        self.assertEqual(np.count_nonzero(first), vp.NONSTAR_VARIABLES)
        np.testing.assert_allclose(
            vp.pivot_values(first, representative),
            np.ones(3),
            rtol=3e-12,
            atol=3e-13,
        )
        with self.assertRaises(vp.KrennStarVariableProjectionError):
            vp.natural_repair_initial_point(
                vp.pivot_representative(0),
                2026072501,
            )

    def test_pole_quantity_is_explicitly_not_a_full_gauge_invariant(self):
        ridge = vp.ridge_variable_projection(self.u, 1e-6)
        diagnostics = vp.path_diagnostics(
            self.u, self.rep, ridge.y
        )
        self.assertEqual(
            diagnostics["pole_quantity_exact_color_gauge_character"],
            [-1, 1, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1],
        )
        self.assertEqual(
            diagnostics["pivot_character_rank_over_Q"], 3
        )
        self.assertEqual(
            diagnostics[
                "pivot_plus_pole_character_rank_over_Q"
            ],
            4,
        )
        self.assertFalse(
            diagnostics[
                "pole_quantity_full_direct_GHZ_color_gauge_invariant"
            ]
        )
        self.assertFalse(
            diagnostics[
                "valid_as_full_gauge_invariant_mathematical_cap"
            ]
        )
        self.assertFalse(
            diagnostics["raw_u_norm_is_a_mathematical_search_cap"]
        )

    def test_zero_pole_quantity_is_strict_json(self):
        zero_y = np.zeros((vp.STAR_COLUMNS, vp.TARGETS), dtype=np.complex128)
        diagnostics = vp.path_diagnostics(
            self.u,
            self.rep,
            zero_y,
        )
        quantity = diagnostics["pole_quantity"]
        self.assertTrue(quantity["available"])
        self.assertEqual(quantity["abs"], 0.0)
        self.assertIsNone(quantity["log_abs"])
        self.assertFalse(quantity["log_abs_defined"])
        self.assertTrue(quantity["exact_zero_in_float64"])
        self.assertTrue(quantity["finite"])
        json.dumps(diagnostics, sort_keys=True, allow_nan=False)

    def test_audits_are_strict_json_and_claims_fail_closed(self):
        audit = vp.core_audit()
        json.dumps(audit, sort_keys=True, allow_nan=False)
        fingerprints = vp.source_fingerprints()
        self.assertEqual(
            fingerprints["core_audit_sha256"], audit["sha256"]
        )
        self.assertEqual(fingerprints["pivot_orbit_count"], 3)
        self.assertFalse(
            audit["numerical_methods"]["raw_u_clipping"]
        )
        self.assertTrue(
            audit["numerical_methods"][
                "residual_L2_balance_has_generic_exact_recession"
            ]
        )
        self.assertFalse(
            audit["known_pole_quantity"]["full_color_gauge_invariant"]
        )
        self.assertTrue(
            audit["claim_boundary"]["all_outputs_numerical_nonproof"]
        )
        snapshot = vp.variable_projection_diagnostics(
            self.u, self.rep, ridge_mu=1e-6
        )
        json.dumps(snapshot, sort_keys=True, allow_nan=False)
        self.assertFalse(
            snapshot["claim_boundary"][
                "finite_counterexample_found"
            ]
        )
        self.assertFalse(
            snapshot["claim_boundary"][
                "global_affine_membership_decided"
            ]
        )


if __name__ == "__main__":
    unittest.main()
