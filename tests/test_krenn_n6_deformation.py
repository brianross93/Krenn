import json
import unittest

from experiments.krenn_quantum_graph.n6_deformation import (
    MINOR_MODULUS,
    N6_DEFORMATION_SCHEMA,
    certify_n6_natural_deformation,
    gauge_action_preserves_every_monomial,
    natural_repair_direction,
    natural_seed_jacobian,
    vertex_scalar_gauge_matrix,
)
from experiments.krenn_quantum_graph.system import variable_index
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
)


class KrennN6DeformationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = certify_n6_natural_deformation()

    def test_full_jacobian_rank_130_is_an_exact_rank_sandwich(self):
        certificate = self.certificate
        payload = certificate.to_dict()
        full = payload["full_jacobian"]
        self.assertEqual(certificate.schema, N6_DEFORMATION_SCHEMA)
        self.assertEqual(full["shape"], [729, 135])
        self.assertEqual(full["rank_over_Q"], 130)
        self.assertEqual(full["nullity_over_Q"], 5)
        self.assertTrue(
            full["kernel_equals_vertex_scalar_gauge_over_Q"]
        )
        minor = certificate.full_minor
        self.assertEqual(minor.matrix_shape, (729, 135))
        self.assertEqual(minor.modulus, MINOR_MODULUS)
        self.assertEqual(minor.size, 130)
        self.assertEqual(minor.determinant_mod, 30)
        self.assertEqual(len(set(minor.row_indices)), 130)
        self.assertEqual(len(set(minor.column_indices)), 130)

    def test_vertex_scalar_gauge_is_exactly_five_dimensional(self):
        gauge = vertex_scalar_gauge_matrix()
        self.assertEqual(len(gauge), 135)
        self.assertTrue(all(len(row) == 5 for row in gauge))
        self.assertTrue(gauge_action_preserves_every_monomial())
        checks = self.certificate.exact_checks
        self.assertTrue(checks["vertex_gauge_rank_5_over_Q"])
        self.assertTrue(
            checks["full_jacobian_kills_vertex_gauge_exactly"]
        )
        self.assertTrue(
            checks["full_kernel_equals_gauge_over_Q"]
        )

    def test_repair_direction_maps_exactly_to_minus_defect_basis(self):
        jacobian = natural_seed_jacobian()
        direction = natural_repair_direction()
        nonzero = [
            (index, value)
            for index, value in enumerate(direction)
            if value
        ]
        self.assertEqual(
            nonzero,
            [
                (variable_index(6, 3, 0, 1, 0, 0), -1),
                (variable_index(6, 3, 2, 3, 0, 0), 1),
            ],
        )
        image = tuple(
            sum(
                row[column] * direction[column]
                for column in range(135)
            )
            for row in jacobian
        )
        expected = tuple(
            -int(equation == N6_D3_SEED_DEFECT_EQUATION)
            for equation in range(729)
        )
        self.assertEqual(image, expected)
        self.assertEqual(
            [index for index, value in enumerate(image) if value],
            [N6_D3_SEED_DEFECT_EQUATION],
        )

    def test_deleting_equation_70_adds_exactly_the_repair_direction(self):
        certificate = self.certificate
        deleted = certificate.to_dict()["defect_deleted_jacobian"]
        self.assertEqual(
            deleted["removed_equation"],
            N6_D3_SEED_DEFECT_EQUATION,
        )
        self.assertEqual(deleted["shape"], [728, 135])
        self.assertEqual(deleted["rank_over_Q"], 129)
        self.assertEqual(deleted["nullity_over_Q"], 6)
        self.assertTrue(
            deleted["kernel_equals_gauge_plus_repair_over_Q"]
        )
        minor = certificate.defect_deleted_minor
        self.assertEqual(minor.matrix_shape, (728, 135))
        self.assertEqual(minor.size, 129)
        self.assertEqual(minor.determinant_mod, 1)
        checks = certificate.exact_checks
        self.assertTrue(
            checks["gauge_plus_repair_rank_6_over_Q"]
        )
        self.assertTrue(
            checks["defect_deleted_jacobian_kills_gauge"]
        )
        self.assertTrue(
            checks["defect_deleted_jacobian_kills_repair"]
        )

    def test_mod_31_is_only_a_nonzero_integer_minor_certificate(self):
        payload = self.certificate.to_dict()
        role = payload["modular_integer_minor_role"]
        self.assertFalse(role["finite_field_solution_claim"])
        self.assertFalse(
            role["finite_field_solution_to_C_transfer_used"]
        )
        for minor in (
            self.certificate.full_minor,
            self.certificate.defect_deleted_minor,
        ):
            record = minor.to_dict()
            self.assertIn("integer", record["proof_role"])
            self.assertFalse(record["finite_field_solution_claim"])
            self.assertFalse(
                record["finite_field_solution_to_C_transfer_used"]
            )
            self.assertNotEqual(record["determinant_mod_31"], 0)

    def test_claim_boundary_does_not_promote_first_order_repair(self):
        payload = self.certificate.to_dict()
        point = payload["point"]
        self.assertFalse(point["exact_GHZ_solution"])
        repair = payload["repair_direction"]
        self.assertTrue(
            repair["preserves_other_728_equations_to_first_order"]
        )
        self.assertFalse(repair["finite_exact_solution_proved"])
        boundary = payload["claim_boundary"]
        self.assertFalse(boundary["natural_seed_is_exact_GHZ_solution"])
        self.assertFalse(
            boundary["first_order_repair_is_finite_solution"]
        )
        self.assertFalse(boundary["finite_field_membership_proved"])
        self.assertFalse(
            boundary["exact_GHZ_image_membership_decided_here"]
        )
        self.assertTrue(self.certificate.exact)
        json.dumps(payload, allow_nan=False)

    def test_certificate_is_deterministic(self):
        replayed = certify_n6_natural_deformation()
        self.assertEqual(
            replayed.full_minor, self.certificate.full_minor
        )
        self.assertEqual(
            replayed.defect_deleted_minor,
            self.certificate.defect_deleted_minor,
        )
        self.assertEqual(
            replayed.exact_checks, self.certificate.exact_checks
        )


if __name__ == "__main__":
    unittest.main()
