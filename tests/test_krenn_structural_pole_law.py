import json
from collections import Counter
from dataclasses import replace
import unittest

from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
)
from experiments.krenn_quantum_graph.structural_pole_law import (
    FIXED_VICTIM_SEEDS,
    KrennStructuralPoleLawError,
    NATURAL_SEED,
    RETAINED_NATURAL_SUPPORTS,
    RETAINED_SUPPORT_ODD_RELATIONS,
    VICTIM_COLORING,
    VICTIM_EQUATION,
    certify_structural_pole_law,
    seed_support,
)
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    generate_sparse_system,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    n6_d3_seed_witness,
)


class KrennStructuralPoleLawTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = certify_structural_pole_law()

    def test_natural_branch_is_upgraded_to_its_unique_smooth_component(self):
        component = self.certificate.natural_component
        self.assertEqual(component.jacobian_rank_over_q, 130)
        self.assertEqual(component.incidence_tangent_dimension, 6)
        self.assertEqual(component.vertex_gauge_dimension, 5)
        self.assertEqual(component.parameterized_family_dimension, 6)
        self.assertEqual(
            component.q_indices, tuple(POLE_INVARIANT_INDICES)
        )
        self.assertTrue(all(dict(component.exact_checks).values()))

        payload = component.to_dict()
        self.assertEqual(
            payload["parameters"]["incidence_parameter"],
            "u=F_002121",
        )
        self.assertEqual(
            payload["parameters"]["formal_lift_parameter_alignment"],
            "u=1-s",
        )
        self.assertEqual(
            payload["pole_law"]["direct_parameter_identity"], "u*Q=1"
        )
        self.assertEqual(
            payload["pole_law"]["formal_parameter_identity"],
            "(1-s)*Q=1",
        )
        self.assertTrue(
            payload["component"][
                "unique_irreducible_component_through_unit_seed"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"][
                "all_global_incidence_components_classified"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"][
                "separate_finite_component_at_u_zero_excluded"
            ]
        )

    def test_natural_q_is_the_exact_complementary_seed_monomial(self):
        atlas = self.certificate.seed_component_atlas
        natural = next(
            component
            for component in atlas.components
            if component.seed == NATURAL_SEED
            and component.victim_equation == VICTIM_EQUATION
        )
        self.assertEqual(natural.support, seed_support(NATURAL_SEED))
        self.assertEqual(
            natural.support,
            tuple(index for index, _value in n6_d3_seed_witness().entries),
        )
        self.assertEqual(natural.q_indices, tuple(POLE_INVARIANT_INDICES))
        self.assertEqual(
            [variable_key(6, 3, index) for index in natural.q_indices],
            [
                (0, 2, 1, 1),
                (0, 3, 2, 2),
                (1, 4, 1, 1),
                (1, 5, 2, 2),
                (2, 3, 0, 0),
                (4, 5, 0, 0),
            ],
        )
        self.assertEqual(
            sorted((*natural.victim_monomial, *natural.q_indices)),
            sorted(
                variable
                for monomial in natural.pure_monomials
                for variable in monomial
            ),
        )

    def test_exact_component_atlas_has_90_times_4_distinct_entries(self):
        atlas = self.certificate.seed_component_atlas
        self.assertEqual(len(atlas.components), 360)
        self.assertEqual(
            len({component.support for component in atlas.components}),
            360,
        )
        victim_counts = Counter(
            component.victim_equation for component in atlas.components
        )
        self.assertEqual(len(victim_counts), 90)
        self.assertEqual(set(victim_counts.values()), {4})
        self.assertTrue(
            all(
                tuple(
                    component.victim_coloring.count(color)
                    for color in range(3)
                ) == (2, 2, 2)
                for component in atlas.components
            )
        )
        self.assertTrue(all(atlas.exact_checks.values()))
        self.assertTrue(
            atlas.exact_checks[
                "local_component_property_transports_by_symmetry"
            ]
        )

    def test_fixed_victim_has_exactly_the_four_declared_seeds(self):
        atlas = self.certificate.seed_component_atlas
        fixed = tuple(
            component.seed
            for component in atlas.components
            if component.victim_equation == VICTIM_EQUATION
        )
        self.assertEqual(
            coloring_from_index(6, 3, VICTIM_EQUATION),
            VICTIM_COLORING,
        )
        self.assertEqual(fixed, FIXED_VICTIM_SEEDS)
        self.assertEqual(
            fixed,
            (
                (0, 4, 8),
                (0, 9, 13),
                (2, 4, 13),
                (2, 9, 8),
            ),
        )
        payload = atlas.to_dict()
        self.assertEqual(
            payload["counts"],
            {
                "balanced_victim_colorings": 90,
                "constant_matching_candidates_per_victim": 27,
                "unique_defect_seed_charts_per_victim": 4,
                "ordered_seed_charts": 360,
                "distinct_coordinate_supports": 360,
            },
        )
        self.assertTrue(
            payload["claim_boundary"][
                "local_component_wide_pole_law_at_each_seed"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"][
                "distinct_global_component_count_decided"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"][
                "components_away_from_marked_seeds_excluded"
            ]
        )

    def test_every_seed_component_replays_only_four_output_monomials(self):
        system = generate_sparse_system(6, 3)
        for component in self.certificate.seed_component_atlas.components:
            with self.subTest(
                victim=component.victim_equation, seed=component.seed
            ):
                support = set(component.support)
                active = {
                    equation: tuple(
                        monomial
                        for monomial in system.equation_monomials(
                            equation
                        )
                        if set(monomial) <= support
                    )
                    for equation in range(3**6)
                }
                nonempty = {
                    equation: monomials
                    for equation, monomials in active.items()
                    if monomials
                }
                self.assertEqual(
                    set(nonempty),
                    {0, 364, 728, component.victim_equation},
                )
                self.assertTrue(
                    all(len(monomials) == 1
                        for monomials in nonempty.values())
                )
                self.assertTrue(all(component.exact_checks.values()))

    def test_six_literal_moving_target_tori_have_odd_circuits(self):
        audit = self.certificate.retained_support_parity
        self.assertEqual(
            tuple(certificate.support
                  for certificate in audit.moving_target_tori),
            RETAINED_NATURAL_SUPPORTS,
        )
        self.assertEqual(
            tuple(certificate.relation
                  for certificate in audit.moving_target_tori),
            RETAINED_SUPPORT_ODD_RELATIONS,
        )
        self.assertEqual(
            tuple(certificate.active_binomial_count
                  for certificate in audit.moving_target_tori),
            (25, 25, 33, 33, 41, 41),
        )
        for certificate in audit.moving_target_tori:
            with self.subTest(support=certificate.support):
                self.assertEqual(certificate.mode, "moving-target")
                self.assertTrue(all(certificate.exact_checks.values()))
                self.assertEqual(len(certificate.relation), 3)
                self.assertEqual(
                    sum(
                        coefficient
                        for _equation, coefficient
                        in certificate.relation
                    ) % 2,
                    1,
                )
                payload = certificate.to_dict()
                self.assertEqual(
                    payload["contradiction"],
                    "1=(-1)^(sum z_c)=-1",
                )
                self.assertFalse(
                    payload["claim_boundary"][
                        "zero_coordinate_boundary_included"
                    ]
                )

    def test_coordinate_boundaries_are_exhausted_without_overclaim(self):
        audit = self.certificate.retained_support_parity
        coordinate = audit.ghz_coordinate_subspaces
        self.assertEqual(
            tuple(len(entry.admissible_strata) for entry in coordinate),
            (2, 2, 4, 4, 8, 8),
        )
        self.assertEqual(
            tuple(
                tuple(
                    len(certificate.support)
                    for certificate in entry.admissible_strata
                )
                for entry in coordinate
            ),
            (
                (21, 22),
                (21, 22),
                (21, 22, 22, 23),
                (21, 22, 22, 23),
                (21, 22, 22, 22, 23, 23, 23, 24),
                (21, 22, 22, 22, 23, 23, 23, 24),
            ),
        )
        natural = set(seed_support(NATURAL_SEED))
        for entry in coordinate:
            with self.subTest(declared=entry.declared_support):
                self.assertTrue(all(entry.exact_checks.values()))
                self.assertTrue(
                    all(
                        natural <= set(certificate.support)
                        for certificate in entry.admissible_strata
                    )
                )
                self.assertTrue(
                    all(
                        certificate.mode == "direct-ghz"
                        and all(certificate.exact_checks.values())
                        for certificate in entry.admissible_strata
                    )
                )
                payload = entry.to_dict()
                self.assertTrue(
                    payload["claim_boundary"][
                        "zero_coordinate_boundary_exhausted_inside_subspace"
                    ]
                )
                self.assertFalse(
                    payload["claim_boundary"][
                        "supports_outside_declared_subspace_examined"
                    ]
                )
        self.assertTrue(all(audit.exact_checks.values()))
        self.assertFalse(
            audit.to_dict()["claim_boundary"]["global_nonexistence_proved"]
        )

    def test_all_payloads_are_strict_json_and_claims_remain_fail_closed(self):
        payload = self.certificate.to_dict()
        round_trip = json.loads(json.dumps(payload, allow_nan=False))
        self.assertEqual(round_trip, payload)
        boundary = payload["claim_boundary"]
        self.assertTrue(
            boundary["natural_branch_relation_is_component_wide"]
        )
        self.assertEqual(
            boundary["unique_defect_seed_charts_certified"], 360
        )
        self.assertFalse(
            boundary["distinct_global_component_count_decided"]
        )
        self.assertTrue(
            boundary["six_retained_coordinate_subspaces_excluded"]
        )
        self.assertFalse(boundary["all_incidence_components_classified"])
        self.assertFalse(
            boundary["global_finite_GHZ_nonexistence_proved"]
        )
        self.assertEqual(
            boundary["exact_affine_GHZ_membership_status"], "undecided"
        )

    def test_corrupted_component_and_toric_claims_are_rejected(self):
        natural = self.certificate.natural_component
        with self.assertRaises(KrennStructuralPoleLawError):
            replace(natural, jacobian_rank_over_q=129)
        with self.assertRaises(KrennStructuralPoleLawError):
            replace(
                natural,
                exact_checks=tuple(
                    (name, False if index == 0 else passed)
                    for index, (name, passed)
                    in enumerate(natural.exact_checks)
                ),
            )

        atlas = self.certificate.seed_component_atlas
        with self.assertRaises(KrennStructuralPoleLawError):
            replace(atlas, components=atlas.components[:-1])
        with self.assertRaises(KrennStructuralPoleLawError):
            replace(
                atlas.components[0],
                q_indices=atlas.components[0].q_indices[:-1],
            )

        parity = self.certificate.retained_support_parity
        first = parity.moving_target_tori[0]
        corrupted_relation = (
            (first.relation[0][0], first.relation[0][1] + 1),
            *first.relation[1:],
        )
        with self.assertRaises(KrennStructuralPoleLawError):
            replace(first, relation=corrupted_relation)
        coordinate = parity.ghz_coordinate_subspaces[0]
        with self.assertRaises(KrennStructuralPoleLawError):
            replace(
                coordinate,
                admissible_strata=coordinate.admissible_strata[:-1],
            )
        with self.assertRaises(KrennStructuralPoleLawError):
            replace(self.certificate, schema="corrupt")


if __name__ == "__main__":
    unittest.main()
