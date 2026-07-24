import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.krenn_quantum_graph.counterexample_search import (
    diagonal_seed_support,
)
from experiments.krenn_quantum_graph.numerical_continuation import (
    AMBIENT_VARIABLES,
    ContinuationPlan,
    EQUATION_COUNT,
    GHZ_COLOR_DIAGONAL_GAUGE,
    GHZ_VICTIM_COLOR_DIAGONAL_GAUGE,
    InvariantMetrics,
    KNOWN_Q_INDICES,
    KrennNumericalContinuationError,
    LMPlan,
    NATURAL_GAUGE_ANCHORS,
    NATURAL_SUPPORT,
    NUMERICAL_BLAS_THREAD_LIMIT,
    VERTEX_SCALAR_GAUGE,
    WeightMetrics,
    analytic_jacobian,
    bounded_complex_lm,
    color_diagonal_ghz_gauge_exponent_matrix,
    color_diagonal_moving_target_gauge_exponent_matrix,
    complex_output,
    complex_residual,
    deterministic_job_specs,
    known_q_color_gauge_character,
    load_lm_checkpoint,
    monomial_index_array,
    natural_gauge_chart,
    natural_laurent_initializer,
    random_bounded_initializer,
    run_continuation,
    support_gauge_chart,
    victim_gauge_character,
)
from experiments.krenn_quantum_graph.system import generate_sparse_system
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    n6_d3_seed_witness,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    EXPECTED_REPRESENTATIVES_AND_SIZES,
)
from experiments.krenn_quantum_graph.witness import evaluate_exact


class KrennNumericalContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = generate_sparse_system(6, 3)
        cls.seed_witness = n6_d3_seed_witness()
        cls.seed_weights = np.zeros(
            AMBIENT_VARIABLES, dtype=np.complex128
        )
        for index, value in cls.seed_witness.entries:
            cls.seed_weights[index] = complex(value)

    def test_natural_seed_has_exactly_one_of_729_residuals(self):
        residual = complex_residual(self.seed_weights)
        self.assertEqual(residual.shape, (EQUATION_COUNT,))
        self.assertEqual(
            int(np.count_nonzero(residual == 0.0j)),
            EQUATION_COUNT - 1,
        )
        self.assertEqual(
            tuple(map(int, np.flatnonzero(residual))),
            (N6_D3_SEED_DEFECT_EQUATION,),
        )
        self.assertEqual(
            residual[N6_D3_SEED_DEFECT_EQUATION],
            1.0 + 0.0j,
        )

    def test_vectorized_residual_matches_all_exact_equations(self):
        exact = evaluate_exact(self.system, self.seed_witness)
        expected = np.asarray(
            [complex(value) for value in exact.residuals],
            dtype=np.complex128,
        )
        vectorized = complex_residual(self.seed_weights)
        np.testing.assert_array_equal(vectorized, expected)
        np.testing.assert_array_equal(
            complex_output(self.seed_weights),
            expected
            + np.asarray(
                self.system.rhs_values, dtype=np.complex128
            ),
        )

    def test_analytic_jacobian_matches_centered_finite_differences(self):
        generator = np.random.default_rng(20260724)
        weights = 0.2 * (
            generator.standard_normal(AMBIENT_VARIABLES)
            + 1.0j * generator.standard_normal(AMBIENT_VARIABLES)
        )
        columns = (0, 11, 47, 81, 133)
        jacobian = analytic_jacobian(weights, columns)
        self.assertEqual(jacobian.shape, (EQUATION_COUNT, len(columns)))

        step = 1.0e-6
        for local, column in enumerate(columns):
            plus = weights.copy()
            minus = weights.copy()
            plus[column] += step
            minus[column] -= step
            difference = (
                complex_output(plus) - complex_output(minus)
            ) / (2.0 * step)
            with self.subTest(column=column):
                np.testing.assert_allclose(
                    jacobian[:, local],
                    difference,
                    rtol=2.0e-8,
                    atol=2.0e-9,
                )

    def test_target_aware_color_diagonal_gauge_ranks_and_invariance(self):
        full = color_diagonal_ghz_gauge_exponent_matrix()
        moving = (
            color_diagonal_moving_target_gauge_exponent_matrix()
        )
        self.assertEqual(full.shape, (AMBIENT_VARIABLES, 15))
        self.assertEqual(moving.shape, (AMBIENT_VARIABLES, 14))
        self.assertEqual(np.linalg.matrix_rank(full), 15)
        self.assertEqual(np.linalg.matrix_rank(moving), 14)

        monomials = monomial_index_array()
        for equation, matrix in (
            (0, full),
            (364, full),
            (728, full),
            (N6_D3_SEED_DEFECT_EQUATION, moving),
        ):
            rows = monomials[
                equation * 15 : (equation + 1) * 15
            ]
            exponent_sums = matrix[rows].sum(axis=1)
            with self.subTest(equation=equation):
                np.testing.assert_array_equal(
                    exponent_sums,
                    np.zeros_like(exponent_sums),
                )

        np.testing.assert_array_equal(
            known_q_color_gauge_character(),
            -victim_gauge_character(),
        )
        dense_moving = natural_gauge_chart()
        dense_direct = natural_gauge_chart(
            action_group=GHZ_COLOR_DIAGONAL_GAUGE
        )
        self.assertEqual(dense_moving.rank, 14)
        self.assertEqual(dense_direct.rank, 15)
        self.assertEqual(len(dense_moving.free_indices), 121)
        self.assertEqual(len(dense_direct.free_indices), 120)
        self.assertTrue(
            set(NATURAL_GAUGE_ANCHORS).issubset(
                dense_moving.anchors
            )
        )

        initializer = random_bounded_initializer(
            tuple(range(AMBIENT_VARIABLES)),
            12345,
            chart=dense_direct,
            scale=0.1,
            l2_bound=8.0,
            linf_bound=4.0,
        )
        weights = initializer.dense_weights()
        np.testing.assert_array_equal(
            weights[np.asarray(dense_direct.anchors)],
            np.ones(dense_direct.rank, dtype=np.complex128),
        )
        known_q = InvariantMetrics.from_weights(
            self.seed_weights, NATURAL_SUPPORT
        ).to_dict()["known_Q"]
        self.assertTrue(
            known_q[
                "moving_target_color_diagonal_gauge_invariant"
            ]
        )
        self.assertFalse(
            known_q[
                "direct_GHZ_full_color_diagonal_gauge_invariant"
            ]
        )
        self.assertFalse(
            known_q["unqualified_gauge_invariance_claimed"]
        )

    def test_blas_thread_limit_is_enforced_and_recorded(self):
        self.assertEqual(NUMERICAL_BLAS_THREAD_LIMIT, 1)
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            with self.subTest(name=name):
                self.assertEqual(os.environ.get(name), "1")

    def test_all_eight_diagonal_roots_have_expected_gauge_ranks(self):
        supports = tuple(
            diagonal_seed_support(representative)
            for representative, _orbit_size
            in EXPECTED_REPRESENTATIVES_AND_SIZES
        )
        charts = tuple(
            support_gauge_chart(
                support,
                action_group=VERTEX_SCALAR_GAUGE,
            )
            for support in supports
        )
        self.assertEqual(
            tuple(chart.rank for chart in charts),
            (2, 3, 4, 4, 4, 5, 5, 4),
        )
        for support, chart in zip(supports, charts, strict=True):
            self.assertEqual(len(chart.anchors), chart.rank)
            self.assertEqual(
                len(chart.free_indices), len(support) - chart.rank
            )
            self.assertEqual(
                set(chart.anchors).union(chart.free_indices),
                set(support),
            )
            self.assertEqual(
                chart.to_dict()["symmetry_weight_equalities"], 0
            )
            self.assertEqual(
                chart.action_group, VERTEX_SCALAR_GAUGE
            )

        direct_charts = tuple(
            support_gauge_chart(
                support,
                action_group=GHZ_COLOR_DIAGONAL_GAUGE,
            )
            for support in supports
        )
        self.assertEqual(
            tuple(chart.rank for chart in direct_charts),
            (6,) * len(supports),
        )

    def test_omitted_natural_anchor_uses_rank_adaptive_chart(self):
        omitted = NATURAL_GAUGE_ANCHORS[0]
        support = tuple(
            index
            for index in range(AMBIENT_VARIABLES)
            if index != omitted
        )
        chart = support_gauge_chart(support, prefer_natural=True)
        self.assertEqual(chart.kind, "support-adaptive")
        self.assertEqual(
            chart.action_group, GHZ_COLOR_DIAGONAL_GAUGE
        )
        self.assertEqual(chart.rank, 15)
        self.assertNotIn(omitted, chart.support)
        self.assertNotIn(omitted, chart.anchors)
        self.assertEqual(len(chart.free_indices), 119)
        with self.assertRaisesRegex(
            KrennNumericalContinuationError,
            "needs all five certified anchors",
        ):
            natural_gauge_chart(support)

    def test_laurent_weight_and_invariant_growth_hit_hard_bounds(self):
        observed_linf = []
        observed_q = []
        for t in (0.5, 0.25, 0.125):
            initializer = natural_laurent_initializer(
                t,
                l2_bound=32.0,
                linf_bound=16.0,
            )
            self.assertFalse(initializer.projection.changed)
            weights = initializer.dense_weights()
            residual = complex_residual(weights)
            self.assertEqual(
                tuple(map(int, np.flatnonzero(residual))),
                (N6_D3_SEED_DEFECT_EQUATION,),
            )
            self.assertAlmostEqual(
                abs(residual[N6_D3_SEED_DEFECT_EQUATION]), t
            )

            weight_metrics = WeightMetrics.from_weights(
                weights,
                initializer.chart,
                l2_bound=32.0,
                linf_bound=16.0,
            )
            invariants = InvariantMetrics.from_weights(
                weights, NATURAL_SUPPORT
            )
            self.assertTrue(
                set(KNOWN_Q_INDICES).issubset(NATURAL_SUPPORT)
            )
            self.assertTrue(invariants.known_q_defined_on_support)
            observed_linf.append(weight_metrics.linf)
            observed_q.append(invariants.known_q_absolute_value)

        self.assertEqual(observed_linf, [2.0, 4.0, 8.0])
        self.assertEqual(observed_q, [2.0, 4.0, 8.0])

        bounded = natural_laurent_initializer(
            0.01,
            l2_bound=4.0,
            linf_bound=2.0,
        )
        bounded_metrics = WeightMetrics.from_weights(
            bounded.dense_weights(),
            bounded.chart,
            l2_bound=4.0,
            linf_bound=2.0,
        )
        self.assertTrue(bounded.projection.changed)
        self.assertTrue(bounded.projection.on_linf_boundary)
        self.assertTrue(bounded_metrics.finite)
        self.assertTrue(bounded_metrics.on_hard_boundary)
        self.assertLessEqual(bounded_metrics.l2, 4.0)
        self.assertLessEqual(bounded_metrics.linf, 2.0)

    def test_job_specs_are_deterministic_and_dense_charts_are_target_aware(self):
        repair = next(
            index
            for index in range(AMBIENT_VARIABLES)
            if index not in set(NATURAL_SUPPORT)
        )
        sparse = tuple(sorted((*NATURAL_SUPPORT, repair)))
        options = dict(
            master_seed=60320260724,
            starts_per_support=2,
            include_dense=True,
            dense_starts=2,
            activation_amplitudes=(0.1, 0.3),
        )
        first = deterministic_job_specs((sparse,), **options)
        second = deterministic_job_specs((sparse,), **options)
        self.assertEqual(first, second)
        self.assertEqual(
            [spec.to_dict() for spec in first],
            [spec.to_dict() for spec in second],
        )
        self.assertEqual(len(first), 4)
        self.assertEqual(
            len({spec.seed_words for spec in first}), len(first)
        )

        dense = tuple(spec for spec in first if spec.dense)
        self.assertEqual(len(dense), 2)
        for spec in dense:
            action_group = (
                GHZ_VICTIM_COLOR_DIAGONAL_GAUGE
                if spec.use_continuation
                else GHZ_COLOR_DIAGONAL_GAUGE
            )
            chart = natural_gauge_chart(
                spec.support,
                action_group=action_group,
            )
            self.assertEqual(len(spec.support), 135)
            self.assertEqual(
                chart.rank, 14 if spec.use_continuation else 15
            )
            self.assertEqual(
                len(chart.free_indices),
                121 if spec.use_continuation else 120,
            )
            self.assertTrue(
                spec.to_dict()["dense_all_135_variables"]
            )

    def test_tiny_bounded_solver_retains_fail_closed_claims(self):
        chart = support_gauge_chart(NATURAL_SUPPORT)
        initializer = random_bounded_initializer(
            NATURAL_SUPPORT,
            424242,
            chart=chart,
            scale=0.2,
            l2_bound=8.0,
            linf_bound=4.0,
        )
        plan = LMPlan(
            max_iterations=1,
            max_evaluations=4,
            l2_bound=8.0,
            linf_bound=4.0,
            backtracking_steps=0,
            checkpoint_interval=0,
            residual_tolerance=1.0e-14,
            gradient_tolerance=1.0e-14,
            step_tolerance=1.0e-14,
        )
        result = bounded_complex_lm(
            initializer,
            NATURAL_SUPPORT,
            chart=chart,
            plan=plan,
            run_id="tiny-fail-closed",
        )
        self.assertLessEqual(result.iterations, plan.max_iterations)
        self.assertLessEqual(result.evaluations, plan.max_evaluations)
        self.assertTrue(result.weight_metrics.finite)
        self.assertLessEqual(result.weight_metrics.l2, plan.l2_bound)
        self.assertLessEqual(result.weight_metrics.linf, plan.linf_bound)
        self.assertFalse(result.exact_solution_certified)
        self.assertFalse(result.nonexistence_proved)

        payload = result.to_dict()
        self.assertEqual(
            payload[
                "configured_blas_thread_limit_per_process"
            ],
            1,
        )
        self.assertTrue(
            payload["residual_metrics"][
                "all_729_equations_accounted"
            ]
        )
        self.assertFalse(payload["exact_solution_certified"])
        self.assertFalse(
            payload["claim_boundary"][
                "numerical_zero_is_exact_witness"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"][
                "bounded_miss_proves_nonexistence"
            ]
        )

    def test_continuation_status_is_fail_closed_and_resume_is_deterministic(
        self,
    ):
        chart = natural_gauge_chart(NATURAL_SUPPORT)
        initializer = natural_laurent_initializer(
            1.0,
            chart=chart,
            l2_bound=8.0,
            linf_bound=4.0,
        )
        lm_plan = LMPlan(
            max_iterations=1,
            max_evaluations=4,
            l2_bound=8.0,
            linf_bound=4.0,
            backtracking_steps=0,
            checkpoint_interval=1,
            residual_tolerance=1.0e-30,
            gradient_tolerance=1.0e-30,
            step_tolerance=1.0e-30,
        )
        plan = ContinuationPlan(
            target_schedule=(1.0, 0.0),
            lm_plan=lm_plan,
            direct_final_polish=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = run_continuation(
                initializer,
                NATURAL_SUPPORT,
                chart=chart,
                plan=plan,
                run_id="fail-closed",
                checkpoint_directory=root,
            )
            self.assertEqual(first.status, "step-nonconverged")
            self.assertEqual(
                tuple(step.status for step in first.steps),
                (
                    "numerical-residual-tolerance",
                    "iteration-cap-reached",
                ),
            )
            self.assertFalse(first.numerical_zero_at_ghz)
            json_before = {
                path.name: path.read_bytes()
                for path in root.glob("*.json")
            }
            npz_before = tuple(
                sorted(path.name for path in root.glob("*.npz"))
            )
            replay = run_continuation(
                initializer,
                NATURAL_SUPPORT,
                chart=chart,
                plan=plan,
                run_id="fail-closed",
                checkpoint_directory=root,
                resume=True,
            )
            self.assertEqual(replay.to_dict(), first.to_dict())
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in root.glob("*.json")
                },
                json_before,
            )
            self.assertEqual(
                tuple(sorted(path.name for path in root.glob("*.npz"))),
                npz_before,
            )

        exploratory = run_continuation(
            initializer,
            NATURAL_SUPPORT,
            chart=chart,
            plan=ContinuationPlan(
                target_schedule=(1.0, 0.0),
                lm_plan=lm_plan,
                direct_final_polish=False,
                continue_after_nonconvergence=True,
            ),
            run_id="exploratory",
        )
        self.assertEqual(
            exploratory.status,
            "schedule-explored-with-nonconverged-steps",
        )

        repair = next(
            index
            for index in range(AMBIENT_VARIABLES)
            if index not in set(NATURAL_SUPPORT)
        )
        support = tuple(sorted((*NATURAL_SUPPORT, repair)))
        moving_chart = natural_gauge_chart(support)
        moving_initializer = random_bounded_initializer(
            support,
            9090,
            chart=moving_chart,
            scale=0.2,
            l2_bound=8.0,
            linf_bound=4.0,
        )
        mid_plan = ContinuationPlan(
            target_schedule=(1.0, 0.0),
            lm_plan=LMPlan(
                max_iterations=2,
                max_evaluations=10,
                l2_bound=8.0,
                linf_bound=4.0,
                backtracking_steps=0,
                checkpoint_interval=1,
                residual_tolerance=1.0e-30,
                gradient_tolerance=1.0e-30,
                step_tolerance=1.0e-30,
            ),
            direct_final_polish=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prefix = root / "mid-step-step-000"
            paused = bounded_complex_lm(
                moving_initializer,
                support,
                chart=moving_chart,
                plan=mid_plan.lm_plan,
                target_amplitude=1.0,
                run_id="mid-step-step-000",
                checkpoint_prefix=prefix,
                work_iteration_cap=1,
            )
            self.assertEqual(paused.status, "checkpoint-paused")
            resumed = run_continuation(
                moving_initializer,
                support,
                chart=moving_chart,
                plan=mid_plan,
                run_id="mid-step",
                checkpoint_directory=root,
                resume=True,
            )
            uninterrupted = run_continuation(
                moving_initializer,
                support,
                chart=moving_chart,
                plan=mid_plan,
                run_id="mid-step",
            )
            self.assertEqual(
                resumed.to_dict(), uninterrupted.to_dict()
            )
            self.assertEqual(resumed.steps[0].iterations, 2)
            self.assertEqual(len(resumed.steps[0].trace), 2)

    def test_checkpoint_round_trip_and_resume_matches_uninterrupted(self):
        repair = next(
            index
            for index in range(AMBIENT_VARIABLES)
            if index not in set(NATURAL_SUPPORT)
        )
        support = tuple(sorted((*NATURAL_SUPPORT, repair)))
        chart = support_gauge_chart(support)
        initializer = random_bounded_initializer(
            support,
            20260724,
            chart=chart,
            scale=0.35,
            l2_bound=8.0,
            linf_bound=4.0,
        )
        plan = LMPlan(
            max_iterations=3,
            max_evaluations=16,
            l2_bound=8.0,
            linf_bound=4.0,
            initial_trust_radius=0.25,
            backtracking_steps=0,
            acceptance_ratio=0.0,
            residual_tolerance=1.0e-30,
            gradient_tolerance=1.0e-30,
            step_tolerance=1.0e-30,
            checkpoint_interval=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary).resolve() / "lm-state"
            paused = bounded_complex_lm(
                initializer,
                support,
                chart=chart,
                plan=plan,
                run_id="checkpoint-resume",
                checkpoint_prefix=prefix,
                work_iteration_cap=1,
            )
            self.assertEqual(paused.status, "checkpoint-paused")
            self.assertTrue(prefix.with_suffix(".json").is_file())

            loaded = load_lm_checkpoint(prefix)
            referenced_npz = (
                prefix.parent / loaded.metadata.npz_file
            )
            self.assertTrue(referenced_npz.is_file())
            self.assertEqual(
                json.loads(
                    prefix.with_suffix(".json").read_text(
                        encoding="utf-8"
                    )
                ),
                loaded.metadata.to_dict(),
            )
            self.assertEqual(loaded.metadata.status, "checkpoint-paused")
            self.assertEqual(
                loaded.metadata.accepted_steps
                + loaded.metadata.rejected_steps,
                loaded.metadata.iteration,
            )
            self.assertEqual(
                len(loaded.metadata.trace),
                loaded.metadata.iteration,
            )
            self.assertEqual(
                loaded.weights.shape, (AMBIENT_VARIABLES,)
            )
            self.assertEqual(
                loaded.current_residual.shape, (EQUATION_COUNT,)
            )
            self.assertEqual(
                loaded.objective_history.shape,
                (loaded.metadata.iteration + 1,),
            )

            paused_again = bounded_complex_lm(
                initializer,
                support,
                chart=chart,
                plan=plan,
                run_id="checkpoint-resume",
                checkpoint_prefix=prefix,
                resume=True,
                work_iteration_cap=1,
            )
            self.assertEqual(
                paused_again.status, "checkpoint-paused"
            )
            self.assertEqual(paused_again.iterations, 2)

            resumed = bounded_complex_lm(
                initializer,
                support,
                chart=chart,
                plan=plan,
                run_id="checkpoint-resume",
                checkpoint_prefix=prefix,
                resume=True,
            )
            final_checkpoint = load_lm_checkpoint(prefix)
            self.assertEqual(
                final_checkpoint.metadata.status, resumed.status
            )
            np.testing.assert_array_equal(
                final_checkpoint.best_weights,
                resumed.dense_weights(),
            )
            self.assertEqual(
                final_checkpoint.metadata.accepted_steps,
                resumed.accepted_steps,
            )
            self.assertEqual(
                final_checkpoint.metadata.rejected_steps,
                resumed.rejected_steps,
            )
            self.assertEqual(
                final_checkpoint.metadata.projection_count,
                resumed.projection_count,
            )
            self.assertEqual(
                final_checkpoint.metadata.trace,
                resumed.trace,
            )

            terminal_json = prefix.with_suffix(".json").read_bytes()
            generations = tuple(
                sorted(path.name for path in prefix.parent.glob("*.npz"))
            )
            terminal_again = bounded_complex_lm(
                initializer,
                support,
                chart=chart,
                plan=plan,
                run_id="checkpoint-resume",
                checkpoint_prefix=prefix,
                resume=True,
            )
            self.assertEqual(
                terminal_again.to_dict(), resumed.to_dict()
            )
            self.assertEqual(
                prefix.with_suffix(".json").read_bytes(),
                terminal_json,
            )
            self.assertEqual(
                tuple(
                    sorted(
                        path.name
                        for path in prefix.parent.glob("*.npz")
                    )
                ),
                generations,
            )
            changed_initial = initializer.dense_weights()
            changed_initial[chart.free_indices[0]] += 0.125
            with self.assertRaisesRegex(
                KrennNumericalContinuationError,
                "does not match",
            ):
                bounded_complex_lm(
                    changed_initial,
                    support,
                    chart=chart,
                    plan=plan,
                    run_id="checkpoint-resume",
                    checkpoint_prefix=prefix,
                    resume=True,
                )

            uninterrupted = bounded_complex_lm(
                initializer,
                support,
                chart=chart,
                plan=plan,
                run_id="checkpoint-resume",
            )
        self.assertEqual(resumed.status, uninterrupted.status)
        self.assertEqual(resumed.iterations, uninterrupted.iterations)
        self.assertEqual(resumed.evaluations, uninterrupted.evaluations)
        np.testing.assert_array_equal(
            resumed.dense_weights(),
            uninterrupted.dense_weights(),
        )
        self.assertEqual(resumed.to_dict(), uninterrupted.to_dict())
        self.assertFalse(resumed.exact_solution_certified)
        self.assertFalse(resumed.nonexistence_proved)


if __name__ == "__main__":
    unittest.main()
