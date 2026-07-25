from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    DERIVATIVE_REPRESENTATIVES,
    KrennDerivativeChartError,
    NATURAL_DEFECT_EQUATION,
    eliminated_derivative_chart,
    natural_derivative_atlas_audit,
    natural_derivative_orbits,
    replay_smoothness_certificate,
    singular_eliminated_derivative_script,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    EXPECTED_EQUATION_ORBIT_COUNTS,
    MIXED_EQUATIONS,
    normalized_seed_chart,
    seed_gauge_split_audit,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    ordered_seed_stabilizer,
)
from experiments.krenn_quantum_graph.localized_chart_monomial_atlas import (
    EXPECTED_REPAIR_MONOMIAL_ORBITS,
    KrennMonomialAtlasError,
    REPAIR_MONOMIAL_REPRESENTATIVES,
    natural_repair_monomial_atlas_audit,
    natural_repair_monomial_orbits,
    repair_monomial_chart,
    singular_repair_monomial_script,
)


def evaluate(polynomial, values):
    total = 0
    for coefficient, monomial in polynomial.terms:
        term = coefficient
        for variable in monomial:
            term *= values[variable]
        total += term
    return total


class KrennLocalizedChartStructureTest(unittest.TestCase):
    def test_equation_orbit_counts_are_derived_from_each_stabilizer(self):
        counts = []
        for orbit_index in range(8):
            stabilizer = ordered_seed_stabilizer(orbit_index)
            seen = set()
            orbit_count = 0
            for generator in range(729):
                if generator in seen:
                    continue
                orbit = {
                    permutation[generator]
                    for permutation in stabilizer.generator_permutations
                }
                seen.update(orbit)
                orbit_count += 1
            self.assertEqual(seen, set(range(729)))
            counts.append(orbit_count)
        self.assertEqual(
            tuple(counts), EXPECTED_EQUATION_ORBIT_COUNTS
        )

    def test_specialized_singular_exporters_reject_composite_fields(self):
        with self.assertRaises(KrennDerivativeChartError):
            singular_eliminated_derivative_script(
                11, characteristic=4
            )
        with self.assertRaises(KrennMonomialAtlasError):
            singular_repair_monomial_script(
                (11, 65), characteristic=15
            )

    def test_all_nine_seed_gauge_characters_have_split_identity_minor(self):
        for orbit_index in range(8):
            audit = seed_gauge_split_audit(orbit_index)
            self.assertEqual(audit["split_rank"], 9)
            self.assertEqual(audit["smith_diagonal"], [1] * 9)
            self.assertEqual(
                audit["identity_minor"],
                [
                    [int(row == column) for column in range(9)]
                    for row in range(9)
                ],
            )

    def test_natural_defect_smoothness_has_exact_rational_identity(self):
        certificate = replay_smoothness_certificate()
        self.assertEqual(certificate["field"], "Q")
        self.assertEqual(certificate["generators"], 13)
        self.assertTrue(certificate["exact_replay"])
        self.assertEqual(
            natural_derivative_orbits(),
            (
                (11, 19, 65, 73, 88, 133),
                (29, 37, 47, 55, 106, 113),
            ),
        )
        audit = natural_derivative_atlas_audit()
        self.assertFalse(
            audit["claim_boundary"]["natural_chart_decided"]
        )
        self.assertEqual(
            audit["strict_stabilizer"]["representatives"], [11, 29]
        )

    def test_derivative_substitution_is_an_exact_polynomial_identity(self):
        original = normalized_seed_chart(6)
        defect_index = original.generator_labels.index(
            ("mixed", NATURAL_DEFECT_EQUATION)
        )
        values = tuple(
            Fraction((index % 9) - 4, (index % 5) + 1)
            for index in range(129)
        )
        for ambient in DERIVATIVE_REPRESENTATIVES:
            derivative = eliminated_derivative_chart(ambient)
            reduced_values = values
            original_values = [Fraction(0)] * 129
            for reduced, old in enumerate(
                derivative.retained_original_chart_variables
            ):
                original_values[old] = reduced_values[reduced]

            defect = original.generators[defect_index]
            eliminated = derivative.eliminated_original_local_variable
            constant_value = 0
            derivative_value = 0
            for coefficient, monomial in defect.terms:
                if eliminated in monomial:
                    term = coefficient
                    reduced_monomial = list(monomial)
                    reduced_monomial.remove(eliminated)
                    for variable in reduced_monomial:
                        term *= original_values[variable]
                    derivative_value += term
                else:
                    term = coefficient
                    for variable in monomial:
                        term *= original_values[variable]
                    constant_value += term
            q = reduced_values[128]
            original_values[eliminated] = -q * constant_value

            reduced_generator = 0
            for original_generator, label in zip(
                original.generators,
                original.generator_labels,
                strict=True,
            ):
                if label == ("mixed", NATURAL_DEFECT_EQUATION):
                    continue
                self.assertEqual(
                    evaluate(
                        derivative.generators[reduced_generator],
                        reduced_values,
                    ),
                    evaluate(original_generator, original_values),
                )
                reduced_generator += 1
            localizer_value = evaluate(
                derivative.generators[-1], reduced_values
            )
            self.assertEqual(
                localizer_value, q * derivative_value - 1
            )
            self.assertEqual(
                evaluate(defect, original_values),
                -constant_value * localizer_value,
            )

    def test_four_repair_monomial_charts_cover_without_zeroing_weights(self):
        self.assertEqual(
            natural_repair_monomial_orbits(),
            EXPECTED_REPAIR_MONOMIAL_ORBITS,
        )
        audit = natural_repair_monomial_atlas_audit()
        self.assertEqual(audit["repair_terms"]["total"], 14)
        self.assertEqual(len(audit["charts"]), 4)
        self.assertTrue(
            audit["residual_torus_gauge"][
                "all_unfixed_weights_remain_independent"
            ]
        )
        self.assertFalse(
            audit["strict_stabilizer"][
                "symmetry_related_weights_equated"
            ]
        )

    def test_repair_chart_substitution_and_localizer_replay_exactly(self):
        original = normalized_seed_chart(6)
        for representative in REPAIR_MONOMIAL_REPRESENTATIVES:
            repair = repair_monomial_chart(representative)
            reduced_values = tuple(
                Fraction((index % 7) - 3, (index % 4) + 1)
                for index in range(repair.variable_count)
            )
            original_values = [Fraction(1)] * 129
            for reduced, old in enumerate(
                repair.retained_original_chart_variables
            ):
                original_values[old] = reduced_values[reduced]
            for original_polynomial, reduced_polynomial in zip(
                original.generators,
                repair.generators[:729],
                strict=True,
            ):
                self.assertEqual(
                    evaluate(original_polynomial, original_values),
                    evaluate(reduced_polynomial, reduced_values),
                )
            localized_value = original_values[
                repair.localized_original_chart_variable
            ]
            inverse_value = reduced_values[
                repair.branch_inverse_variable
            ]
            self.assertEqual(
                evaluate(repair.generators[-1], reduced_values),
                inverse_value * localized_value - 1,
            )


if __name__ == "__main__":
    unittest.main()
