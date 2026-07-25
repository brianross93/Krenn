from dataclasses import replace
from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.hafnian_identities import (
    contracted_edge_matrix,
)
from experiments.krenn_quantum_graph.perfect_matching_blocker_ideals import (
    KrennPerfectMatchingBlockerError,
    PerfectMatchingBlockerType,
    blocker_census,
    brute_force_orientation_count,
    build_tutte_barrier_affine_chart,
    is_minimal_perfect_matching_blocker,
    perfect_matching_blocker_types,
    top_chow_coefficient,
)
from experiments.krenn_quantum_graph.system import canonical_edges
from experiments.krenn_quantum_graph.targets import canonical_ghz_target
from experiments.krenn_quantum_graph.hafnian_identities import (
    tensor_contraction,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


class KrennPerfectMatchingBlockerIdealTest(unittest.TestCase):
    def test_exact_orbit_and_labeled_censuses(self):
        expected = {
            4: (2, 8),
            6: (4, 91),
            8: (6, 1408),
        }
        for n, (orbit_count, labeled_count) in expected.items():
            with self.subTest(n=n):
                census = blocker_census(n)
                self.assertEqual(
                    census["orbit_type_count"], orbit_count
                )
                self.assertEqual(
                    census["labeled_blocker_count"],
                    labeled_count,
                )
                for blocker_type in perfect_matching_blocker_types(n):
                    self.assertTrue(
                        is_minimal_perfect_matching_blocker(
                            n, blocker_type.blocker_edges
                        )
                    )

    def test_n8_types_and_square_chow_degrees(self):
        types = perfect_matching_blocker_types(8)
        self.assertEqual(
            tuple(
                (
                    blocker_type.barrier_size,
                    blocker_type.odd_component_sizes,
                    len(blocker_type.blocker_edges),
                    blocker_type.labeled_orbit_size,
                )
                for blocker_type in types
            ),
            (
                (0, (7, 1), 7, 8),
                (0, (5, 3), 15, 56),
                (1, (5, 1, 1), 11, 168),
                (1, (3, 3, 1), 15, 560),
                (2, (3, 1, 1, 1), 12, 560),
                (3, (1, 1, 1, 1, 1), 10, 56),
            ),
        )
        square = tuple(
            blocker_type
            for blocker_type in types
            if blocker_type.qutrit_chart_is_square
        )
        self.assertEqual(
            tuple(
                (
                    blocker_type.barrier_size,
                    blocker_type.odd_component_sizes,
                )
                for blocker_type in square
            ),
            (
                (2, (3, 1, 1, 1)),
                (3, (1, 1, 1, 1, 1)),
            ),
        )
        for blocker_type, expected_degree in zip(
            square, (30, 24), strict=True
        ):
            outside = tuple(
                vertex
                for component in blocker_type.components
                for vertex in component
            )
            dynamic = top_chow_coefficient(
                blocker_type.blocker_edges, outside
            )
            brute_force = brute_force_orientation_count(
                blocker_type.blocker_edges, outside
            )
            self.assertEqual(dynamic, expected_degree)
            self.assertEqual(brute_force, expected_degree)

    def test_affine_chart_replays_contracted_bilinear_edges(self):
        coordinates = {}
        for edge_number, (first, second) in enumerate(
            canonical_edges(8)
        ):
            for first_color in range(3):
                for second_color in range(3):
                    numerator = (
                        5 * edge_number
                        + 3 * first_color
                        + second_color
                    ) % 11 - 5
                    if numerator:
                        coordinates[
                            (
                                first,
                                second,
                                first_color,
                                second_color,
                            )
                        ] = Fraction(
                            numerator,
                            1 + (
                                edge_number
                                + first_color
                                + second_color
                            )
                            % 5,
                        )
        witness = SparseWitness.from_coordinates(8, 3, coordinates)
        for blocker_type in perfect_matching_blocker_types(8):
            if not blocker_type.qutrit_chart_is_square:
                continue
            chart = build_tutte_barrier_affine_chart(
                witness, blocker_type, 1
            )
            values = tuple(
                Fraction((7 * index) % 13 - 6, index % 5 + 1)
                for index in range(chart.variable_count)
            )
            covectors = chart.product_covectors(values)
            contracted = contracted_edge_matrix(witness, covectors)
            self.assertEqual(
                chart.evaluate(values),
                tuple(
                    contracted[first][second]
                    for first, second in chart.blocker_edges
                ),
            )
            self.assertEqual(chart.ghz_contraction(values), 1)
            self.assertEqual(
                tensor_contraction(
                    canonical_ghz_target(8, 3), covectors
                ),
                1,
            )

    def test_vertex_relabeling_preserves_chart_census(self):
        witness = SparseWitness.from_coordinates(
            8,
            3,
            {
                (0, 7, 0, 1): 2,
                (1, 6, 2, 0): -3,
                (2, 5, 1, 1): Fraction(4, 5),
            },
        )
        blocker_type = PerfectMatchingBlockerType(
            8, 3, (1, 1, 1, 1, 1)
        )
        chart = build_tutte_barrier_affine_chart(
            witness,
            blocker_type,
            2,
            vertex_order=(7, 5, 3, 1, 6, 4, 2, 0),
        )
        self.assertEqual(chart.variable_count, 10)
        self.assertEqual(chart.equation_count, 10)
        self.assertTrue(chart.square)
        self.assertEqual(len(set(chart.blocker_edges)), 10)
        values = tuple(
            Fraction(index - 4, index % 3 + 1)
            for index in range(chart.variable_count)
        )
        contracted = contracted_edge_matrix(
            witness, chart.product_covectors(values)
        )
        self.assertEqual(
            chart.evaluate(values),
            tuple(
                contracted[first][second]
                for first, second in chart.blocker_edges
            ),
        )
        with self.assertRaisesRegex(
            KrennPerfectMatchingBlockerError, "blocker edges"
        ):
            replace(
                chart,
                blocker_edges=(
                    (5, 7),
                    *chart.blocker_edges[1:],
                ),
            )
        with self.assertRaisesRegex(
            KrennPerfectMatchingBlockerError, "polynomial"
        ):
            replace(
                chart,
                equations=(
                    (((chart.variable_count,), Fraction(1)),),
                    *chart.equations[1:],
                ),
            )
        with self.assertRaisesRegex(
            KrennPerfectMatchingBlockerError, "exact rationals"
        ):
            chart.evaluate(
                (0.5,) + (Fraction(0),) * (chart.variable_count - 1)
            )
        with self.assertRaisesRegex(
            KrennPerfectMatchingBlockerError, "exact integer"
        ):
            chart.product_covectors(
                (True,) + (Fraction(0),) * (chart.variable_count - 1)
            )

    def test_validation_fails_closed(self):
        with self.assertRaisesRegex(
            KrennPerfectMatchingBlockerError, "odd-partition"
        ):
            PerfectMatchingBlockerType(8, 2, (2, 2, 1, 1))
        with self.assertRaisesRegex(
            KrennPerfectMatchingBlockerError, "unique"
        ):
            is_minimal_perfect_matching_blocker(
                4, ((0, 1), (1, 0))
            )
        blocker_type = PerfectMatchingBlockerType(
            8, 3, (1, 1, 1, 1, 1)
        )
        witness = SparseWitness.from_coordinates(
            8, 3, {(0, 1, 0, 0): 1}
        )
        with self.assertRaisesRegex(
            KrennPerfectMatchingBlockerError, "permutation"
        ):
            build_tutte_barrier_affine_chart(
                witness,
                blocker_type,
                0,
                vertex_order=(0, 1, 2, 3, 4, 5, 6, 6),
            )


if __name__ == "__main__":
    unittest.main()
