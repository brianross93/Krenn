import unittest

from experiments.krenn_quantum_graph.symmetry_structure import (
    KrennSymmetryStructureError,
    ProductIrrep,
    SymmetricGroupIrrep,
    hook_length_dimension,
    k4_matching_action_audit,
    partitions,
    perfect_matching_module,
    qutrit_weight_space_decomposition,
)
from experiments.krenn_quantum_graph.system import (
    perfect_matching_count,
    variable_count,
)


class KrennSymmetryStructureTest(unittest.TestCase):
    def test_k4_exception_is_the_surjection_s4_to_s3(self):
        audit = k4_matching_action_audit()
        self.assertEqual(audit["vertex_group_order"], 24)
        self.assertEqual(audit["matching_count"], 3)
        self.assertEqual(audit["image_order"], 6)
        self.assertEqual(audit["kernel_order"], 4)
        self.assertTrue(audit["surjective_to_S3"])
        self.assertTrue(audit["kernel_is_klein_four"])

    def test_perfect_matching_module_decompositions(self):
        expected = {
            4: (((4,), 1), ((2, 2), 2)),
            6: (((6,), 1), ((4, 2), 9), ((2, 2, 2), 5)),
            8: (
                ((8,), 1),
                ((6, 2), 20),
                ((4, 4), 14),
                ((4, 2, 2), 56),
                ((2, 2, 2, 2), 14),
            ),
        }
        for n, expected_items in expected.items():
            with self.subTest(n=n):
                module = perfect_matching_module(n)
                self.assertEqual(
                    tuple(
                        (irrep.partition, irrep.dimension)
                        for irrep in module
                    ),
                    expected_items,
                )
                self.assertEqual(
                    sum(irrep.total_dimension for irrep in module),
                    perfect_matching_count(n),
                )

    def test_qutrit_weight_decomposition_reconciles_dimensions(self):
        for n in (4, 6, 8):
            with self.subTest(n=n):
                decomposition = qutrit_weight_space_decomposition(n)
                self.assertEqual(len(decomposition), 10)
                self.assertEqual(
                    sum(
                        constituent.total_dimension
                        for constituent in decomposition
                    ),
                    variable_count(n, 3),
                )
                symmetric = sum(
                    constituent.total_dimension
                    for constituent in decomposition
                    if constituent.sector == "endpoint_symmetric"
                )
                antisymmetric = sum(
                    constituent.total_dimension
                    for constituent in decomposition
                    if constituent.sector
                    == "endpoint_antisymmetric"
                )
                self.assertEqual(symmetric, 6 * n * (n - 1) // 2)
                self.assertEqual(
                    antisymmetric, 3 * n * (n - 1) // 2
                )

    def test_partition_and_hook_validation(self):
        self.assertEqual(
            partitions(4),
            ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)),
        )
        self.assertEqual(hook_length_dimension((4, 2)), 9)
        with self.assertRaisesRegex(
            KrennSymmetryStructureError, "partition total"
        ):
            partitions(-1)
        with self.assertRaisesRegex(
            KrennSymmetryStructureError, "Young diagram"
        ):
            hook_length_dimension((2, 3))
        with self.assertRaisesRegex(
            KrennSymmetryStructureError, "positive even"
        ):
            perfect_matching_module(5)
        with self.assertRaisesRegex(
            KrennSymmetryStructureError, "positive integer"
        ):
            SymmetricGroupIrrep((4,), 1.5)
        with self.assertRaisesRegex(
            KrennSymmetryStructureError, "product-irrep"
        ):
            ProductIrrep((4,), "trivial", True, "endpoint_symmetric")


if __name__ == "__main__":
    unittest.main()
