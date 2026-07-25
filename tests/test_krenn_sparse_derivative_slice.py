from fractions import Fraction
from math import gcd
import unittest

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    DERIVATIVE_REPRESENTATIVES,
    KrennDerivativeChartError,
    singular_sparse_derivative_gauge_script,
    sparse_derivative_gauge_slice,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    normalized_seed_chart,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    generator_characters,
    residual_torus_characters,
)


EXPECTED_AMBIENT_DERIVATIVES = {
    11: {(65,), (55, 133), (73, 113)},
    29: {(47,), (55, 106), (73, 88)},
}
EXPECTED_DERIVATIVE_CHARACTERS = {
    11: (-1, 0, 0, 0, 0, 0, 0, 0, -1),
    29: (-1, 0, 0, 0, 0, 0, 0, 0, 1),
}


def monomial_character(monomial, characters):
    return tuple(
        sum(characters[variable][coordinate] for variable in monomial)
        for coordinate in range(9)
    )


class KrennSparseDerivativeSliceTest(unittest.TestCase):
    def test_exact_sparse_shapes_and_ambient_derivatives(self):
        natural = normalized_seed_chart(6)
        for ambient in DERIVATIVE_REPRESENTATIVES:
            first = sparse_derivative_gauge_slice(ambient)
            second = sparse_derivative_gauge_slice(ambient)
            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertEqual(first.variable_count, 129)
            self.assertEqual(
                first.retained_original_chart_variables, tuple(range(129))
            )
            self.assertEqual(len(first.generators), 730)
            self.assertEqual(first.term_count, 10_942)
            self.assertEqual(first.maximum_degree, 4)
            self.assertEqual(first.derivative_polynomial.term_count, 3)
            self.assertEqual(first.generators[:-1], natural.generators)

            ambient_monomials = {
                tuple(sorted(
                    natural.remaining_weight_indices[variable]
                    for variable in monomial
                ))
                for _coefficient, monomial
                in first.derivative_polynomial.terms
            }
            self.assertEqual(
                ambient_monomials, EXPECTED_AMBIENT_DERIVATIVES[ambient]
            )
            expected_last = {
                monomial: coefficient
                for coefficient, monomial
                in first.derivative_polynomial.terms
            }
            expected_last[()] = -1
            self.assertEqual(
                {
                    monomial: coefficient
                    for coefficient, monomial in first.generators[-1].terms
                },
                expected_last,
            )

    def test_derivatives_are_primitive_semi_invariants(self):
        characters = residual_torus_characters(6)
        for ambient in DERIVATIVE_REPRESENTATIVES:
            chart = sparse_derivative_gauge_slice(ambient)
            self.assertEqual(
                chart.derivative_character,
                EXPECTED_DERIVATIVE_CHARACTERS[ambient],
            )
            term_characters = {
                monomial_character(monomial, characters)
                for _coefficient, monomial
                in chart.derivative_polynomial.terms
            }
            self.assertEqual(term_characters, {chart.derivative_character})
            divisor = 0
            for component in chart.derivative_character:
                divisor = gcd(divisor, abs(component))
            self.assertEqual(divisor, 1)
            self.assertEqual(
                sum(
                    character * cocharacter
                    for character, cocharacter in zip(
                        chart.derivative_character,
                        chart.normalizing_cocharacter,
                        strict=True,
                    )
                ),
                1,
            )

    def test_exact_residual_action_normalizes_nonzero_derivatives(self):
        characters = residual_torus_characters(6)
        homogeneous_generators = generator_characters(6)
        values = (Fraction(1),) * 129
        for ambient in DERIVATIVE_REPRESENTATIVES:
            chart = sparse_derivative_gauge_slice(ambient)
            derivative_value = chart.derivative_polynomial.evaluate(values)
            self.assertEqual(derivative_value, 3)
            parameter = 1 / derivative_value
            transformed = tuple(
                value * parameter ** sum(
                    character_component * cocharacter_component
                    for character_component, cocharacter_component in zip(
                        character,
                        chart.normalizing_cocharacter,
                        strict=True,
                    )
                )
                for value, character in zip(values, characters, strict=True)
            )
            self.assertEqual(
                chart.derivative_polynomial.evaluate(transformed),
                Fraction(1),
            )
            self.assertEqual(chart.generators[-1].evaluate(transformed), 0)
            for polynomial, generator_character in zip(
                chart.generators[:-1],
                homogeneous_generators,
                strict=True,
            ):
                exponent = sum(
                    character_component * cocharacter_component
                    for character_component, cocharacter_component in zip(
                        generator_character,
                        chart.normalizing_cocharacter,
                        strict=True,
                    )
                )
                self.assertEqual(
                    polynomial.evaluate(transformed),
                    parameter ** exponent * polynomial.evaluate(values),
                )
            summary = chart.summary()
            self.assertTrue(summary["natural_generators_homogeneous"])
            self.assertEqual(
                summary["character_cocharacter_pairing"], 1
            )
            self.assertFalse(
                summary["claim_boundary"]["natural_chart_decided"]
            )

    def test_sparse_singular_export_is_deterministic_and_fail_closed(self):
        first = singular_sparse_derivative_gauge_script(
            11, characteristic=31, algorithm="std"
        )
        second = singular_sparse_derivative_gauge_script(
            11, characteristic=31, algorithm="std"
        )
        self.assertEqual(first, second)
        self.assertIn("ring r=31,(x0,x1", first)
        self.assertIn("KRENN_SPARSE_DERIVATIVE_PARSE_OK", first)
        self.assertIn(
            'print("variables="+string(nvars(basering)))', first
        )
        self.assertIn(
            'print("generators="+string(size(I)))', first
        )
        self.assertIn("ideal G=std(I);", first)
        with self.assertRaises(KrennDerivativeChartError):
            singular_sparse_derivative_gauge_script(
                11, characteristic=4
            )
        with self.assertRaises(KrennDerivativeChartError):
            singular_sparse_derivative_gauge_script(
                11, algorithm="mystery"
            )
        with self.assertRaises(KrennDerivativeChartError):
            sparse_derivative_gauge_slice(19)


if __name__ == "__main__":
    unittest.main()
