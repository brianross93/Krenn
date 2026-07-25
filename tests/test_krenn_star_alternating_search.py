import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.krenn_quantum_graph.star_alternating_search import (
    KrennStarALSError,
    StarALSConfig,
    _maximum_feasible_alpha,
    _within_caps,
    _write_json_atomic,
    dual_residual_accounting,
    gauge_chart_audit,
    independent_tensor_outputs,
    minimum_change_star_correction,
    numerical_star_matrix,
    run_star_als,
    run_star_als_campaign,
    tensor_outputs,
)


class KrennStarAlternatingSearchTest(unittest.TestCase):
    def test_exact_rank_15_direct_ghz_gauge_chart(self):
        chart = gauge_chart_audit()
        self.assertEqual(chart["exact_rank_over_Q"], 15)
        self.assertEqual(
            chart["anchor_indices"],
            [
                1,
                46,
                48,
                53,
                72,
                79,
                81,
                85,
                89,
                108,
                112,
                116,
                126,
                130,
                134,
            ],
        )
        self.assertEqual(chart["anchor_minor_determinant"], -2)
        self.assertEqual(
            chart["maximum_anchors_in_any_apex_color_block"], 2
        )
        self.assertEqual(
            len(
                chart["natural_seed_relation"][
                    "anchors_zero_on_natural_seed"
                ]
            ),
            13,
        )
        self.assertFalse(
            chart["natural_seed_relation"][
                "known_natural_Laurent_branch_lies_in_this_open_chart"
            ]
        )
        self.assertFalse(
            chart["claim_boundary"][
                "chart_covers_solutions_with_a_zero_anchor"
            ]
        )

    def test_minimum_change_correction_fixes_anchors(self):
        chart = gauge_chart_audit()
        anchors = np.asarray(chart["anchor_indices"], dtype=np.int64)
        weights = np.full(135, 0.1 + 0.05j, dtype=np.complex128)
        weights[anchors] = 1
        delta, diagnostic = minimum_change_star_correction(weights, 0)
        self.assertEqual(delta.shape, (135,))
        self.assertTrue(np.all(delta[anchors] == 0))
        self.assertTrue(
            diagnostic["minimum_change_not_origin_replacement"]
        )
        self.assertEqual(len(diagnostic["by_apex_color"]), 3)

    def test_trajectory_is_deterministic_monotone_gauge_fixed_and_bounded(
        self,
    ):
        config = StarALSConfig(
            radius=2.0,
            global_l2_radius=8.0,
            maximum_sweeps=2,
            seed=2026072501,
            maximum_seconds=30.0,
            patience_apex_updates=100,
        )
        first = run_star_als(config)
        second = run_star_als(config)
        self.assertEqual(first["weights"], second["weights"])
        self.assertEqual(first["trace"][0]["schedule"], second["trace"][0]["schedule"])
        self.assertEqual(len(first["trace"]), 12)
        self.assertEqual(first["termination"], "maximum-sweeps")
        self.assertNotEqual(
            first["trace"][0]["direction"],
            first["trace"][6]["direction"],
        )
        previous_primary = first["initial_residual"]["primary"]["l2"]
        previous_independent = first["initial_residual"]["independent"]["l2"]
        for row in first["trace"]:
            self.assertLessEqual(
                row["residual_after"]["primary_l2"], previous_primary
            )
            self.assertLessEqual(
                row["residual_after"]["independent_l2"],
                previous_independent,
            )
            self.assertTrue(
                row["weights_after"]["within_both_global_caps"]
            )
            self.assertTrue(
                row["weights_after"][
                    "all_15_gauge_anchors_exactly_one"
                ]
            )
            previous_primary = row["residual_after"]["primary_l2"]
            previous_independent = row["residual_after"][
                "independent_l2"
            ]
        self.assertFalse(
            first["claim_boundary"][
                "numerical_zero_is_exact_counterexample"
            ]
        )
        self.assertFalse(
            first["claim_boundary"][
                "open_chart_covers_zero_anchor_solutions"
            ]
        )

    def test_checkpoint_round_trip_replays_original_and_terminal_state(self):
        config = StarALSConfig(
            radius=1.0,
            maximum_sweeps=1,
            seed=2026072502,
            maximum_seconds=30.0,
            patience_apex_updates=100,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "run.json"
            first = run_star_als(
                config, checkpoint_path=checkpoint
            )
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["state"]["cursor"]["completed_apex_updates"], 6
            )
            self.assertEqual(len(payload["state"]["trace"]), 6)
            self.assertIn("source_sha256", payload)
            self.assertIn("star_sha256", payload)
            self.assertIn("initial_residual", payload["state"])
            self.assertIn("best_weights", payload["state"])

            resumed = run_star_als(
                config,
                checkpoint_path=checkpoint,
                resume=True,
            )
            self.assertEqual(first["weights"], resumed["weights"])
            self.assertEqual(
                first["initial_residual"], resumed["initial_residual"]
            )
            self.assertEqual(
                first["final_residual"], resumed["final_residual"]
            )

            payload["source_sha256"]["system.py"] = "0" * 64
            checkpoint.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(KrennStarALSError):
                run_star_als(
                    config,
                    checkpoint_path=checkpoint,
                    resume=True,
                )

            first = run_star_als(config)
            checkpoint.unlink()
            run_star_als(config, checkpoint_path=checkpoint)
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            payload["claim_boundary"][
                "numerical_zero_is_exact_counterexample"
            ] = True
            checkpoint.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(KrennStarALSError):
                run_star_als(
                    config,
                    checkpoint_path=checkpoint,
                    resume=True,
                )

    def test_independent_729_equation_replay_and_validation_fail_closed(
        self,
    ):
        values = np.ones(135, dtype=np.complex128)
        primary = tensor_outputs(values)
        independent = independent_tensor_outputs(values)
        self.assertTrue(np.array_equal(primary, independent))
        accounting = dual_residual_accounting(values)
        self.assertTrue(
            accounting["agreement"]["all_729_outputs_agree"]
        )
        self.assertEqual(numerical_star_matrix(values, 0).shape, (243, 15))
        with self.assertRaises(KrennStarALSError):
            numerical_star_matrix(values[:10], 0)
        with self.assertRaises(KrennStarALSError):
            StarALSConfig(radius=0, maximum_sweeps=1, seed=1)
        with self.assertRaises(KrennStarALSError):
            StarALSConfig(
                radius=1,
                global_l2_radius=1,
                maximum_sweeps=1,
                seed=1,
            )

    def test_campaign_accepts_explicit_paired_global_caps(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = run_star_als_campaign(
                output_directory=Path(temporary),
                radii=(4.0,),
                global_l2_radii=(8.0,),
                seeds=(2026072503,),
                maximum_sweeps=1,
                maximum_seconds_per_run=30.0,
            )
            self.assertEqual(payload["radii_as_global_linf_caps"], [4.0])
            self.assertEqual(payload["explicit_global_l2_caps"], [8.0])
            self.assertEqual(
                payload["results"][0]["effective_bounds"],
                {"global_linf_cap": 4.0, "global_l2_cap": 8.0},
            )
            self.assertIn(
                "linf_4_l2_8_seed_2026072503",
                payload["results"][0]["result_path"],
            )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(KrennStarALSError):
                run_star_als_campaign(
                    output_directory=Path(temporary),
                    radii=(4.0, 8.0),
                    global_l2_radii=(8.0,),
                    seeds=(2026072503,),
                    maximum_sweeps=1,
                    maximum_seconds_per_run=30.0,
                )

    def test_boundary_step_is_strictly_feasible_after_reconstruction(self):
        config = StarALSConfig(
            radius=8.0,
            global_l2_radius=8.0,
            maximum_sweeps=1,
            seed=1,
        )
        weights = np.zeros(135, dtype=np.complex128)
        anchors = np.asarray(
            gauge_chart_audit()["anchor_indices"], dtype=np.int64
        )
        weights[anchors] = 1
        free = next(index for index in range(135) if index not in anchors)
        weights[free] = 6
        delta = np.zeros(135, dtype=np.complex128)
        delta[free] = 10 + 3j
        alpha, limiting = _maximum_feasible_alpha(
            weights, delta, config
        )
        self.assertGreater(alpha, 0)
        self.assertLess(alpha, 1)
        self.assertIn("global-l2", limiting)
        proposal = weights + alpha * delta
        proposal[anchors] = 1
        self.assertTrue(_within_caps(proposal, config))
        self.assertLess(np.linalg.norm(proposal), 8.0)

    def test_atomic_checkpoint_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            real_replace = os.replace
            calls = 0

            def transient_lock(source, destination):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError("simulated Windows reader lock")
                return real_replace(source, destination)

            with mock.patch(
                "experiments.krenn_quantum_graph."
                "star_alternating_search.os.replace",
                side_effect=transient_lock,
            ):
                _write_json_atomic(path, {"value": 1})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"value": 1},
            )
            self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
