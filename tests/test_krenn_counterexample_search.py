import unittest

from experiments.krenn_quantum_graph.counterexample_search import (
    KrennCounterexampleSearchError,
    SupportCampaignPlan,
    SupportSeed,
    canonical_support_representative,
    default_support_seeds,
    diagonal_seed_support,
    greedy_pruned_omission_seed,
    n6_d3_system,
    run_support_campaign,
    support_campaign_from_dict,
    support_orbit_size,
    transport_support,
)
from experiments.krenn_quantum_graph.support_extension import (
    natural_support,
    size21_support_audits,
)
from experiments.krenn_quantum_graph.support_search import (
    support_profile,
)
from experiments.krenn_quantum_graph.system import (
    perfect_matchings,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    EXPECTED_REPRESENTATIVES_AND_SIZES,
)


class KrennCounterexampleSupportSearchTests(unittest.TestCase):
    @staticmethod
    def plan(
        *,
        nodes=8,
        candidates=2,
        omission_quota=0,
        activation_width=2,
    ):
        return SupportCampaignPlan(
            node_cap=nodes,
            support_cap=22,
            candidate_cap=candidates,
            omission_candidate_quota=omission_quota,
            activation_width=activation_width,
            worker_count=1,
            deterministic_seed=60320260724,
            scratch_directory=r"D:\KrennScratch\counterexample_search",
            include_two_coordinate_omissions=False,
        )

    def test_all_eight_diagonal_seed_orbits_are_roots(self):
        seeds = default_support_seeds(
            include_two_coordinate_omissions=False
        )
        orbit_seeds = tuple(
            seed for seed in seeds
            if seed.family == "diagonal-seed-orbit"
        )
        self.assertEqual(len(orbit_seeds), 8)
        expected = tuple(
            diagonal_seed_support(representative)
            for representative, _size
            in EXPECTED_REPRESENTATIVES_AND_SIZES
        )
        self.assertEqual(
            tuple(seed.support for seed in orbit_seeds),
            expected,
        )
        self.assertEqual(
            expected[6],
            natural_support(),
        )
        self.assertEqual(len(perfect_matchings(6)), 15)
        self.assertTrue(
            any(
                set(natural_support()).difference(seed.support)
                for seed in orbit_seeds
            )
        )

    def test_arbitrary_support_orbit_representative_is_invariant(self):
        support = size21_support_audits()[0].support
        representative = canonical_support_representative(support)
        transported = transport_support(
            support,
            (1, 0, 2, 3, 4, 5),
            (2, 0, 1),
        )
        self.assertEqual(
            canonical_support_representative(transported),
            representative,
        )
        self.assertEqual(
            support_orbit_size(transported),
            support_orbit_size(support),
        )

    def test_tiny_size22_campaign_replays_singleton_closure(self):
        seed = SupportSeed(
            name="stored-size21",
            family="retains-natural-size21",
            support=size21_support_audits()[0].support,
        )
        result = run_support_campaign(
            self.plan(nodes=4, candidates=2),
            seeds=(seed,),
        )
        self.assertEqual(result.termination, "candidate-cap-reached")
        self.assertEqual(result.nodes_examined, 3)
        self.assertEqual(len(result.candidates), 2)
        for candidate in result.candidates:
            self.assertEqual(candidate.support_size, 22)
            profile = support_profile(
                n6_d3_system(),
                candidate.support,
            )
            self.assertTrue(
                profile.singleton_free_necessary_condition
            )
            self.assertFalse(candidate.omitted_natural_coordinates)
            self.assertEqual(candidate.symmetry_weight_equalities, 0)
        self.assertEqual(
            support_campaign_from_dict(result.to_dict()),
            result,
        )

    def test_per_size_quota_reaches_sizes_22_23_and_24(self):
        seed = SupportSeed(
            name="stored-size21",
            family="retains-natural-size21",
            support=size21_support_audits()[0].support,
        )
        plan = SupportCampaignPlan(
            node_cap=200,
            support_cap=24,
            candidate_cap=6,
            per_support_size_candidate_quota=2,
            discovered_support_cap=10_000,
            omission_candidate_quota=0,
            activation_width=2,
            worker_count=1,
            deterministic_seed=7,
            scratch_directory=(
                r"D:\KrennScratch\counterexample_search"
            ),
            include_two_coordinate_omissions=False,
        )
        result = run_support_campaign(plan, seeds=(seed,))
        self.assertEqual(
            tuple(candidate.support_size for candidate in result.candidates),
            (22, 22, 23, 23, 24, 24),
        )
        self.assertEqual(result.termination, "candidate-cap-reached")
        self.assertEqual(
            result.supports_discovered,
            result.nodes_examined + result.frontier_remaining,
        )

    def test_discovered_support_cap_is_an_explicit_termination(self):
        seed = SupportSeed(
            name="stored-size21",
            family="retains-natural-size21",
            support=size21_support_audits()[0].support,
        )
        plan = SupportCampaignPlan(
            node_cap=20,
            support_cap=22,
            candidate_cap=1,
            per_support_size_candidate_quota=1,
            discovered_support_cap=1,
            omission_candidate_quota=0,
            activation_width=2,
            worker_count=1,
            deterministic_seed=7,
            scratch_directory=(
                r"D:\KrennScratch\counterexample_search"
            ),
            include_two_coordinate_omissions=False,
        )
        result = run_support_campaign(plan, seeds=(seed,))
        self.assertEqual(
            result.termination, "discovered-support-cap-reached"
        )
        self.assertEqual(result.nodes_examined, 0)
        self.assertEqual(result.supports_discovered, 1)
        self.assertEqual(result.frontier_remaining, 1)

    def test_deterministic_seed_controls_activation_tie_breaking(self):
        seed = SupportSeed(
            name="stored-size21",
            family="retains-natural-size21",
            support=size21_support_audits()[0].support,
        )

        def activated_coordinate(deterministic_seed):
            plan = SupportCampaignPlan(
                node_cap=3,
                support_cap=22,
                candidate_cap=1,
                per_support_size_candidate_quota=1,
                discovered_support_cap=100,
                omission_candidate_quota=0,
                activation_width=1,
                worker_count=1,
                deterministic_seed=deterministic_seed,
                scratch_directory=(
                    r"D:\KrennScratch\counterexample_search"
                ),
                include_two_coordinate_omissions=False,
            )
            result = run_support_campaign(plan, seeds=(seed,))
            return tuple(
                sorted(
                    set(result.candidates[0].support).difference(
                        seed.support
                    )
                )
            )

        self.assertNotEqual(
            activated_coordinate(0),
            activated_coordinate(134),
        )

    def test_closure_branch_queue_has_exact_unique_state_accounting(self):
        seed = SupportSeed(
            name="diagonal-orbit-root",
            family="diagonal-seed-orbit",
            support=diagonal_seed_support(
                EXPECTED_REPRESENTATIVES_AND_SIZES[0][0]
            ),
        )
        plan = SupportCampaignPlan(
            node_cap=20,
            support_cap=28,
            candidate_cap=1,
            per_support_size_candidate_quota=1,
            discovered_support_cap=1_000,
            omission_candidate_quota=1,
            activation_width=2,
            worker_count=1,
            deterministic_seed=7,
            scratch_directory=(
                r"D:\KrennScratch\counterexample_search"
            ),
            include_two_coordinate_omissions=False,
        )
        result = run_support_campaign(plan, seeds=(seed,))
        self.assertEqual(result.termination, "node-cap-reached")
        self.assertEqual(
            result.supports_discovered,
            result.nodes_examined + result.frontier_remaining,
        )

    def test_transported_size21_root_searches_natural_omissions(self):
        transported = transport_support(
            size21_support_audits()[0].support,
            (0, 1, 2, 3, 4, 5),
            (1, 0, 2),
        )
        self.assertTrue(
            set(natural_support()).difference(transported)
        )
        seed = SupportSeed(
            name="transported-omission-root",
            family="omits-natural-by-symmetry",
            support=transported,
        )
        result = run_support_campaign(
            self.plan(
                nodes=4,
                candidates=2,
                omission_quota=2,
            ),
            seeds=(seed,),
        )
        self.assertTrue(result.includes_natural_omission)
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(
            all(
                candidate.omitted_natural_coordinates
                for candidate in result.candidates
            )
        )
        self.assertTrue(
            all(
                candidate.symmetry_weight_equalities == 0
                for candidate in result.candidates
            )
        )

    def test_dense_pruning_supplies_a_closed_omission_stratum(self):
        omitted = natural_support()[0]
        seed = greedy_pruned_omission_seed(
            (omitted,),
            deterministic_seed=60320260724,
            minimum_support=22,
        )
        self.assertNotIn(omitted, seed.support)
        self.assertGreaterEqual(len(seed.support), 22)
        self.assertLess(len(seed.support), 135)
        profile = support_profile(n6_d3_system(), seed.support)
        self.assertTrue(profile.singleton_free_necessary_condition)

    def test_caps_workers_and_scratch_location_fail_closed(self):
        common = dict(
            node_cap=1,
            support_cap=22,
            candidate_cap=1,
            omission_candidate_quota=0,
            activation_width=1,
            deterministic_seed=1,
            scratch_directory=r"D:\KrennScratch\counterexample_search",
        )
        with self.assertRaises(KrennCounterexampleSearchError):
            SupportCampaignPlan(worker_count=0, **common)
        with self.assertRaises(KrennCounterexampleSearchError):
            SupportCampaignPlan(worker_count=17, **common)
        with self.assertRaises(KrennCounterexampleSearchError):
            SupportCampaignPlan(
                worker_count=1,
                **{**common, "support_cap": 21},
            )
        with self.assertRaises(KrennCounterexampleSearchError):
            SupportCampaignPlan(
                worker_count=1,
                **{
                    **common,
                    "scratch_directory": (
                        r"C:\Users\example\OneDrive\counterexample"
                    ),
                },
            )
        with self.assertRaises(KrennCounterexampleSearchError):
            SupportCampaignPlan(
                worker_count=1,
                **{
                    **common,
                    "omission_candidate_quota": 2,
                },
            )

    def test_round_trip_rejects_semantic_tampering(self):
        seed = SupportSeed(
            name="stored-size21",
            family="retains-natural-size21",
            support=size21_support_audits()[0].support,
        )
        payload = run_support_campaign(
            self.plan(nodes=2, candidates=1),
            seeds=(seed,),
        ).to_dict()
        payload["candidates"][0]["feasible_matching_counts"][0] += 1
        with self.assertRaises(KrennCounterexampleSearchError):
            support_campaign_from_dict(payload)

    def test_round_trip_rejects_accounting_and_termination_tampering(self):
        seed = SupportSeed(
            name="stored-size21",
            family="retains-natural-size21",
            support=size21_support_audits()[0].support,
        )
        payload = run_support_campaign(
            self.plan(nodes=2, candidates=1),
            seeds=(seed,),
        ).to_dict()
        for field in (
            "nodes_examined",
            "supports_discovered",
            "frontier_remaining",
            "seed_count",
        ):
            changed = {
                **payload,
                field: -1,
            }
            with self.subTest(field=field):
                with self.assertRaises(KrennCounterexampleSearchError):
                    support_campaign_from_dict(changed)
        changed = {
            **payload,
            "termination": "frontier-exhausted-under-support-cap",
        }
        with self.assertRaises(KrennCounterexampleSearchError):
            support_campaign_from_dict(changed)


if __name__ == "__main__":
    unittest.main()
