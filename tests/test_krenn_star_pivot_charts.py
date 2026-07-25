import copy
from fractions import Fraction
from itertools import combinations
import unittest

from experiments.krenn_quantum_graph.star_linearization import (
    star_linearization,
)
from experiments.krenn_quantum_graph.star_pivot_charts import (
    KrennStarPivotChartError,
    star_pivot_chart_audit,
    star_pivot_charts,
    star_pivot_orbit_audit,
    verify_star_pivot_chart_audit,
)


def _evaluate_quadratic(polynomial, values):
    return sum(
        (
            values[left] * values[right]
            for left, right in polynomial
        ),
        Fraction(0),
    )


def _determinant_3_by_3(matrix):
    return (
        matrix[0][0] * matrix[1][1] * matrix[2][2]
        + matrix[0][1] * matrix[1][2] * matrix[2][0]
        + matrix[0][2] * matrix[1][0] * matrix[2][1]
        - matrix[0][2] * matrix[1][1] * matrix[2][0]
        - matrix[0][1] * matrix[1][0] * matrix[2][2]
        - matrix[0][0] * matrix[1][2] * matrix[2][1]
    )


class KrennStarPivotChartsTest(unittest.TestCase):
    def test_target_rows_are_exactly_three_partner_color_blocks(self):
        for apex in range(6):
            factorization = star_linearization(apex)
            for color, target_row in enumerate((0, 121, 242)):
                for column, (_partner, partner_color) in enumerate(
                    factorization.columns
                ):
                    polynomial = factorization.entries[
                        target_row
                    ][column]
                    self.assertEqual(
                        bool(polynomial), partner_color == color
                    )
                    if polynomial:
                        self.assertEqual(len(polynomial), 3)

    def test_exact_minor_census_and_factorizations(self):
        values = tuple(
            Fraction((index * 19 + 7) % 29 + 1, 11)
            for index in range(135)
        )
        for apex in range(6):
            factorization = star_linearization(apex)
            charts = star_pivot_charts(apex)
            self.assertEqual(len(charts), 125)
            self.assertEqual(
                len({
                    chart.minor_columns for chart in charts
                }),
                125,
            )
            nonzero_support_minors = sum(
                set(
                    factorization.columns[column][1]
                    for column in columns
                )
                == {0, 1, 2}
                for columns in combinations(range(15), 3)
            )
            self.assertEqual(nonzero_support_minors, 125)
            self.assertEqual(455 - nonzero_support_minors, 330)

            for chart in charts:
                numerical_minor = [
                    [
                        _evaluate_quadratic(
                            factorization.entries[target_row][column],
                            values,
                        )
                        for column in chart.minor_columns
                    ]
                    for target_row in (0, 121, 242)
                ]
                determinant = _determinant_3_by_3(
                    numerical_minor
                )
                factorized = chart.determinant_sign
                for factor in chart.quadratic_factors:
                    factorized *= _evaluate_quadratic(
                        factor, values
                    )
                self.assertEqual(determinant, factorized)

    def test_s5_times_s3_orbits_are_the_three_equality_patterns(self):
        audit = star_pivot_orbit_audit()
        self.assertEqual(audit["group_order"], 720)
        self.assertEqual(audit["universe_size"], 125)
        self.assertEqual(audit["orbit_count"], 3)
        self.assertEqual(
            [
                (
                    row["equality_pattern"],
                    row["representative_local_partner_triple"],
                    row["orbit_size"],
                    row["stabilizer_order"],
                )
                for row in audit["orbits"]
            ],
            [
                ("all-same", [0, 0, 0], 5, 144),
                ("exactly-two-same", [0, 0, 1], 60, 12),
                ("all-distinct", [0, 1, 2], 60, 12),
            ],
        )
        self.assertTrue(audit["orbit_sizes_sum_to_125"])

    def test_rank_cover_claims_and_boundaries_fail_closed(self):
        audit = star_pivot_chart_audit()
        for apex in audit["apex_audits"]:
            self.assertEqual(apex["restricted_shape"], [3, 15])
            self.assertEqual(
                apex["three_column_minors_total"], 455
            )
            self.assertEqual(
                apex["structurally_nonzero_minors"], 125
            )
            self.assertEqual(
                apex["structurally_zero_minors"], 330
            )
        self.assertEqual(
            audit["rank_and_cover"][
                "direct_GHZ_membership_forces_restricted_rank"
            ],
            3,
        )
        self.assertTrue(
            audit["rank_and_cover"][
                "the_125_determinant_opens_cover_every_solution_per_apex"
            ]
        )
        self.assertFalse(
            audit["claim_boundary"]["any_open_chart_solved"]
        )
        self.assertFalse(
            audit["claim_boundary"]["finite_counterexample_found"]
        )

    def test_strict_round_trip_rejects_corruption(self):
        audit = star_pivot_chart_audit()
        self.assertEqual(
            verify_star_pivot_chart_audit(audit), audit
        )
        corrupted = copy.deepcopy(audit)
        corrupted["orbit_classification"]["orbits"][1][
            "orbit_size"
        ] = 59
        with self.assertRaises(KrennStarPivotChartError):
            verify_star_pivot_chart_audit(corrupted)
        escalated = copy.deepcopy(audit)
        escalated["claim_boundary"][
            "global_affine_membership_decided"
        ] = True
        with self.assertRaises(KrennStarPivotChartError):
            verify_star_pivot_chart_audit(escalated)

    def test_invalid_apices_fail_closed(self):
        for apex in (-1, 6, True):
            with self.assertRaises(KrennStarPivotChartError):
                star_pivot_charts(apex)


if __name__ == "__main__":
    unittest.main()
