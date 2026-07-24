from dataclasses import replace
from fractions import Fraction
import json
import unittest

from experiments.krenn_quantum_graph.formal_lift import (
    ALL_ZERO_EQUATION,
    DEFAULT_TRUNCATION_ORDER,
    ExactPolynomial,
    FORMAL_LIFT_SCHEMA,
    GAUGE_SLICE_INDICES,
    GAUGE_SLICE_INJECTIVITY_SCHEMA,
    KrennFormalLiftError,
    NATURAL_SUPPORT_MONOMIALS,
    POLE_INVARIANT_INDICES,
    POLYNOMIAL_ONE,
    POLYNOMIAL_S,
    POLYNOMIAL_ZERO,
    REPAIR_DECREASE_INDEX,
    REPAIR_INCREASE_INDEX,
    certify_n6_d3_formal_lift,
    certify_gauge_slice_injectivity,
    determinant_over_q,
    evaluate_polynomial_system,
    first_order_repair_direction,
    gauge_slice_matrix,
    geometric_polynomial,
    natural_support_surviving_monomials,
    pole_invariant_truncation,
    pole_invariant_vertex_degrees,
    recurrent_higher_order_direction,
    replay_formal_truncation,
)
from experiments.krenn_quantum_graph.system import (
    generate_sparse_system,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
)


def sparse(vector):
    return [
        (index, value)
        for index, value in enumerate(vector)
        if value
    ]


class KrennFormalLiftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = certify_n6_d3_formal_lift(
            DEFAULT_TRUNCATION_ORDER
        )

    def test_small_polynomial_ring_is_canonical_and_exact(self):
        geometric = (
            POLYNOMIAL_ONE + POLYNOMIAL_S + POLYNOMIAL_S**2
        )
        product = (POLYNOMIAL_ONE - POLYNOMIAL_S) * geometric
        self.assertEqual(
            product,
            POLYNOMIAL_ONE - ExactPolynomial.monomial(3),
        )
        self.assertEqual(product.to_expression(), "1 - s^3")
        rational = (
            ExactPolynomial.monomial(2, Fraction(2, 3))
            + ExactPolynomial.monomial(2, Fraction(1, 3))
            - ExactPolynomial.monomial(2)
        )
        self.assertEqual(rational, POLYNOMIAL_ZERO)
        self.assertEqual(product.specialize(1), 0)
        with self.assertRaisesRegex(
            KrennFormalLiftError, "exact integers"
        ):
            ExactPolynomial.constant(0.5)
        with self.assertRaisesRegex(
            KrennFormalLiftError, "nonnegative"
        ):
            ExactPolynomial.monomial(-1)

    def test_general_truncations_replay_all_729_equations(self):
        for order in (1, 2, 5):
            with self.subTest(order=order):
                replay = replay_formal_truncation(order)
                self.assertTrue(replay.exact)
                self.assertEqual(len(replay.output_coefficients), 729)
                self.assertEqual(
                    replay.output_coefficients,
                    replay.expected_coefficients,
                )
                residual = sparse(
                    replay.residual_from_moving_target
                )
                self.assertEqual(
                    residual,
                    [
                        (
                            ALL_ZERO_EQUATION,
                            -ExactPolynomial.monomial(order + 1),
                        )
                    ],
                )
                self.assertEqual(
                    [
                        equation
                        for equation, coefficient in enumerate(
                            replay.output_coefficients
                        )
                        if not coefficient.is_zero
                    ],
                    [0, 70, 364, 728],
                )

    def test_natural_support_table_proves_the_general_identity(self):
        self.assertEqual(
            natural_support_surviving_monomials(),
            NATURAL_SUPPORT_MONOMIALS,
        )
        self.assertEqual(
            NATURAL_SUPPORT_MONOMIALS,
            (
                (0, (0, 81, 126)),
                (70, (0, 98, 121)),
                (364, (13, 67, 121)),
                (728, (26, 80, 98)),
            ),
        )

    def test_jacobian_hessian_and_cubic_orders_are_exact(self):
        certificate = self.certificate
        self.assertEqual(
            sparse(certificate.jacobian_decrease_column),
            [(0, Fraction(1)), (70, Fraction(1))],
        )
        self.assertEqual(
            sparse(certificate.jacobian_increase_column),
            [(0, Fraction(1))],
        )
        self.assertEqual(
            sparse(certificate.first_order_image),
            [(70, Fraction(-1))],
        )
        self.assertEqual(
            sparse(certificate.half_second_derivative),
            [(0, Fraction(-1))],
        )
        self.assertEqual(
            sparse(certificate.mixed_third_order_derivative),
            [(0, Fraction(-1))],
        )
        self.assertEqual(
            sparse(certificate.cubic_first_order_derivative),
            [],
        )
        self.assertEqual(
            sparse(first_order_repair_direction()),
            [
                (REPAIR_DECREASE_INDEX, Fraction(-1)),
                (REPAIR_INCREASE_INDEX, Fraction(1)),
            ],
        )
        self.assertEqual(
            sparse(recurrent_higher_order_direction()),
            [(REPAIR_INCREASE_INDEX, Fraction(1))],
        )

    def test_five_coordinate_slice_really_quotients_the_gauge(self):
        matrix = gauge_slice_matrix()
        self.assertEqual(
            matrix,
            (
                (1, 0, 1, 0, 0),
                (1, 0, 0, 1, 0),
                (0, 1, 0, 0, 1),
                (-1, 0, -1, -1, -1),
                (-1, -1, -1, -1, 0),
            ),
        )
        self.assertEqual(determinant_over_q(matrix), -2)
        first = first_order_repair_direction()
        higher = recurrent_higher_order_direction()
        self.assertTrue(
            all(
                first[index] == 0 and higher[index] == 0
                for index in GAUGE_SLICE_INDICES
            )
        )
        self.assertEqual(
            [
                variable_key(6, 3, index)
                for index in GAUGE_SLICE_INDICES
            ],
            [
                (0, 2, 1, 1),
                (0, 3, 2, 2),
                (1, 4, 1, 1),
                (1, 5, 2, 2),
                (4, 5, 0, 0),
            ],
        )

    def test_restricted_jacobian_is_exactly_injective_on_the_slice(self):
        lock = self.certificate.slice_injectivity
        self.assertEqual(lock.schema, GAUGE_SLICE_INJECTIVITY_SCHEMA)
        self.assertTrue(lock.exact)
        self.assertEqual(lock.ambient_dimension, 135)
        self.assertEqual(lock.slice_dimension, 130)
        self.assertEqual(lock.full_jacobian_rank_over_q, 130)
        self.assertEqual(lock.full_jacobian_nullity_over_q, 5)
        self.assertEqual(lock.vertex_gauge_dimension, 5)
        self.assertEqual(lock.kernel_slice_intersection_dimension, 0)
        self.assertEqual(lock.restricted_jacobian_rank_over_q, 130)
        self.assertEqual(lock.restricted_jacobian_nullity_over_q, 0)
        self.assertTrue(lock.restricted_jacobian_injective_over_q)
        self.assertEqual(lock.gauge_restriction_determinant, -2)
        self.assertTrue(
            lock.exact_checks[
                "source_full_kernel_is_exactly_vertex_gauge"
            ]
        )
        self.assertTrue(
            lock.exact_checks[
                "kernel_meets_declared_slice_only_at_zero"
            ]
        )
        payload = lock.to_dict()
        self.assertEqual(
            payload["restricted_jacobian"]["shape"], [729, 130]
        )
        self.assertEqual(
            payload["restricted_jacobian"]["rank_over_Q"], 130
        )
        self.assertEqual(
            payload["restricted_jacobian"]["nullity_over_Q"], 0
        )
        self.assertTrue(
            payload["restricted_jacobian"]["injective_over_Q"]
        )
        self.assertFalse(
            payload["claim_boundary"]["global_branch_uniqueness_claimed"]
        )

    def test_recursive_uniqueness_is_only_for_this_target_and_slice(self):
        payload = self.certificate.to_dict()
        uniqueness = payload["recursive_uniqueness"]
        self.assertEqual(
            uniqueness["restricted_jacobian_rank_over_Q"], 130
        )
        self.assertEqual(
            uniqueness["restricted_jacobian_nullity_over_Q"], 0
        )
        self.assertTrue(
            uniqueness[
                "moving_target_formal_lift_unique_in_fixed_slice"
            ]
        )
        self.assertIn("moving-target", uniqueness["scope"])
        self.assertIn("fixed local linear slice", uniqueness["scope"])
        self.assertFalse(uniqueness["global_branch_uniqueness_claimed"])
        self.assertFalse(
            uniqueness["uniqueness_in_other_gauge_slices_claimed"]
        )
        boundary = payload["claim_boundary"]
        self.assertTrue(
            boundary[
                "moving_target_formal_lift_unique_in_declared_linear_slice"
            ]
        )
        self.assertFalse(
            boundary["global_formal_branch_uniqueness_proved"]
        )
        self.assertFalse(
            boundary["formal_uniqueness_in_other_gauge_slices_proved"]
        )

    def test_degree_six_invariant_certifies_the_unremovable_pole(self):
        order = self.certificate.truncation.order
        invariant = pole_invariant_truncation(order)
        self.assertEqual(
            pole_invariant_vertex_degrees(),
            (2, 2, 2, 2, 2, 2),
        )
        self.assertEqual(invariant, geometric_polynomial(order))
        self.assertEqual(
            (POLYNOMIAL_ONE - POLYNOMIAL_S) * invariant,
            POLYNOMIAL_ONE
            - ExactPolynomial.monomial(order + 1),
        )
        self.assertEqual(
            tuple(POLE_INVARIANT_INDICES),
            (13, 26, 67, 80, 81, 126),
        )
        payload = self.certificate.to_dict()[
            "gauge_invariant_pole"
        ]
        self.assertEqual(payload["formal_value"], "1/(1-s)")
        self.assertEqual(payload["denominator_at_s_1"], 0)
        self.assertTrue(payload["pole_survives_vertex_gauge"])

    def test_claim_boundary_separates_formal_lift_from_affine_witness(self):
        certificate = self.certificate
        self.assertEqual(certificate.schema, FORMAL_LIFT_SCHEMA)
        self.assertTrue(certificate.exact)
        payload = certificate.to_dict()
        boundary = payload["claim_boundary"]
        self.assertTrue(
            boundary[
                "first_order_repair_lifts_to_every_finite_order"
            ]
        )
        self.assertTrue(
            boundary["moving_target_has_full_Q_power_series_lift"]
        )
        self.assertFalse(
            boundary["cokernel_obstruction_found_along_this_branch"]
        )
        self.assertFalse(
            boundary["formal_arc_lies_in_constant_GHZ_fiber"]
        )
        self.assertFalse(
            boundary["power_series_specialization_at_s_1_is_defined"]
        )
        self.assertFalse(
            boundary["finite_affine_GHZ_witness_produced"]
        )
        self.assertFalse(
            boundary["exact_affine_GHZ_image_membership_proved"]
        )
        self.assertEqual(
            boundary["exact_affine_GHZ_image_membership_status"],
            "undecided",
        )
        self.assertEqual(
            payload["parameters"]["defect_equation"],
            N6_D3_SEED_DEFECT_EQUATION,
        )
        json.dumps(payload, allow_nan=False)

    def test_replay_and_certificate_fail_closed_on_tampering(self):
        replay = self.certificate.truncation
        tampered = list(replay.output_coefficients)
        tampered[0] += POLYNOMIAL_ONE
        with self.assertRaisesRegex(
            KrennFormalLiftError, "729-equation"
        ):
            replace(replay, output_coefficients=tuple(tampered))

        system = generate_sparse_system(6, 3)
        with self.assertRaisesRegex(
            KrennFormalLiftError, "duplicate"
        ):
            evaluate_polynomial_system(
                system,
                (
                    (REPAIR_DECREASE_INDEX, POLYNOMIAL_ZERO),
                    (REPAIR_DECREASE_INDEX, POLYNOMIAL_ONE),
                ),
            )
        with self.assertRaisesRegex(
            KrennFormalLiftError, "outside"
        ):
            evaluate_polynomial_system(
                system,
                ((system.variable_count, POLYNOMIAL_ONE),),
            )
        lock = self.certificate.slice_injectivity
        with self.assertRaisesRegex(
            KrennFormalLiftError, "injectivity certificate failed"
        ):
            replace(lock, restricted_jacobian_rank_over_q=129)
        with self.assertRaisesRegex(
            KrennFormalLiftError, "injectivity certificate failed"
        ):
            replace(lock, gauge_restriction_determinant=0)

    def test_certificate_is_deterministic(self):
        replayed = certify_n6_d3_formal_lift(
            DEFAULT_TRUNCATION_ORDER
        )
        self.assertEqual(replayed, self.certificate)
        self.assertEqual(
            certify_gauge_slice_injectivity(),
            self.certificate.slice_injectivity,
        )
        self.assertEqual(
            replayed.exact_checks, self.certificate.exact_checks
        )


if __name__ == "__main__":
    unittest.main()
