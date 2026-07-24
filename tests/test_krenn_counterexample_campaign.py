import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.krenn_quantum_graph.counterexample_campaign import (
    DEFAULT_RESULTS,
    DEFAULT_SCRATCH,
    KrennCounterexampleCampaignError,
    LMPlan,
    WORKER_THREAD_LIMITS,
    _campaign_argv_from_namespace,
    _command_line,
    _enforce_worker_thread_limits,
    _parser,
    _radius_scratch_name,
    _select_numerical_supports,
    _validated_campaign_inputs,
    _worker,
)
from experiments.krenn_quantum_graph.support_extension import natural_support


class KrennCounterexampleCampaignTests(unittest.TestCase):
    @staticmethod
    def args() -> argparse.Namespace:
        return _parser().parse_args([])

    def test_default_campaign_inputs_are_canonical_and_bounded(self):
        args = self.args()
        scratch, results, radii, schedule = _validated_campaign_inputs(args)
        self.assertEqual(scratch, DEFAULT_SCRATCH.resolve())
        self.assertEqual(results, DEFAULT_RESULTS.resolve())
        self.assertEqual(radii, (8.0, 16.0, 32.0))
        self.assertEqual(schedule[-1], 0.0)
        self.assertEqual(args.support_state_cap, 2_000_000)

    def test_rejects_zero_omissions_and_malformed_numeric_budgets(self):
        invalid = (
            ("omission_supports", 0),
            ("dense_starts", 0),
            ("workers", 17),
            ("support_state_cap", 5_000_001),
            ("checkpoint_interval", 51),
            ("best_per_radius_class", -1),
            ("radii", (8.0, 8.0)),
            ("radii", (16.0, 8.0)),
            ("radii", (math.sqrt(14.5), 8.0)),
            ("continuation_schedule", (1.0, 0.5, 0.5, 0.0)),
            ("continuation_schedule", (0.0,)),
        )
        for name, value in invalid:
            args = self.args()
            setattr(args, name, value)
            with self.subTest(name=name, value=value):
                with self.assertRaises(KrennCounterexampleCampaignError):
                    _validated_campaign_inputs(args)

    def test_rejects_scratch_and_results_outside_declared_roots(self):
        args = self.args()
        args.scratch_directory = str(Path.cwd() / "scratch")
        with self.assertRaises(KrennCounterexampleCampaignError):
            _validated_campaign_inputs(args)

        args = self.args()
        args.results_directory = str(DEFAULT_SCRATCH / "results")
        with self.assertRaises(KrennCounterexampleCampaignError):
            _validated_campaign_inputs(args)

        args = self.args()
        args.scratch_directory = (
            r"C:\Users\example\OneDrive\counterexample"
        )
        with self.assertRaises(KrennCounterexampleCampaignError):
            _validated_campaign_inputs(args)

    def test_worker_thread_limits_are_forced_to_one(self):
        receipt = _enforce_worker_thread_limits()
        self.assertEqual(receipt, WORKER_THREAD_LIMITS)
        self.assertTrue(receipt)
        self.assertTrue(
            all(os.environ[name] == "1" for name in receipt)
        )

    def test_round_trippable_argv_receipt_and_collision_free_radius_names(self):
        args = self.args()
        _scratch, _results, radii, schedule = (
            _validated_campaign_inputs(args)
        )
        receipt = _campaign_argv_from_namespace(args, radii, schedule)
        replay = _parser().parse_args(receipt)
        self.assertEqual(tuple(replay.radii), radii)
        self.assertEqual(tuple(replay.continuation_schedule), schedule)
        self.assertEqual(replay.support_state_cap, args.support_state_cap)

        command = _command_line(
            (
                r"C:\Program Files\Python\python.exe",
                "--results-directory",
                r"C:\A Path\results",
            )
        )
        self.assertIn('"C:\\Program Files\\Python\\python.exe"', command)
        self.assertIn('"C:\\A Path\\results"', command)
        self.assertNotEqual(
            _radius_scratch_name(0, 1.0000001),
            _radius_scratch_name(1, 1.0000002),
        )

    def test_selection_includes_every_campaign_and_deduplicates_orbits(self):
        natural = natural_support()
        natural_rep = tuple(range(22))
        orbit_rep = tuple(range(30, 52))
        pruned_rep = tuple(range(60, 82))
        retained = SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    support=natural + tuple(range(1, 14)),
                    orbit_representative=natural_rep,
                ),
            )
        )
        orbit = SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    support=tuple(range(31, 53)),
                    orbit_representative=orbit_rep,
                ),
                SimpleNamespace(
                    support=tuple(range(32, 54)),
                    orbit_representative=natural_rep,
                ),
            )
        )
        pruned = SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    support=tuple(range(61, 83)),
                    orbit_representative=pruned_rep,
                ),
            )
        )
        selected = _select_numerical_supports(
            (retained, orbit, pruned)
        )
        self.assertEqual(
            selected,
            (
                retained.candidates[0].support,
                orbit_rep,
                pruned_rep,
            ),
        )
        self.assertEqual(len(selected), len(set(selected)))

    def test_worker_enables_exploratory_progression_and_resume(self):
        sentinel = object()
        with patch(
            "experiments.krenn_quantum_graph.counterexample_campaign."
            "run_numerical_job",
            return_value=sentinel,
        ) as mocked:
            spec = object()
            lm_plan = LMPlan(
                max_iterations=1,
                max_evaluations=2,
                l2_bound=8.0,
                linf_bound=8.0,
                checkpoint_interval=0,
            )
            returned_spec, returned_result = _worker(
                (spec, lm_plan, (1.0, 0.0), str(DEFAULT_SCRATCH))
            )
        self.assertIs(returned_spec, spec)
        self.assertIs(returned_result, sentinel)
        keywords = mocked.call_args.kwargs
        self.assertTrue(keywords["resume"])
        self.assertTrue(
            keywords[
                "continuation_plan"
            ].continue_after_nonconvergence
        )


if __name__ == "__main__":
    unittest.main()
