from fractions import Fraction
from pathlib import Path
import unittest

from experiments.krenn_quantum_graph.hafnian_identities import (
    deterministic_modular_values,
    generate_equal_g_shadow_system,
)
from experiments.krenn_quantum_graph.shadow_slices import (
    EXPECTED_CANONICAL_GHZ_PIVOT_VALUES,
    EXPECTED_PIVOT_DETERMINANT,
    EXPECTED_PIVOT_DETERMINANT_MOD_31,
    KrennShadowSliceError,
    canonical_ghz_shadow_star_solution,
    generate_n6_d3_shadow_star_slice,
    solve_n6_d3_shadow_occupations,
)
from experiments.krenn_quantum_graph.targets import canonical_ghz_target


class KrennShadowSliceTest(unittest.TestCase):
    def test_seed_912_slice_is_exactly_star_linear(self):
        system = generate_equal_g_shadow_system(6, 3)
        star_slice = generate_n6_d3_shadow_star_slice()

        self.assertEqual(star_slice.star_variables, tuple(range(30)))
        self.assertEqual(
            star_slice.nonstar_variables, tuple(range(30, 90))
        )
        expected_signs = tuple(
            -1 if residue < 16 else 1
            for residue in deterministic_modular_values(
                60, modulus=31, seed=912
            )
        )
        self.assertEqual(
            tuple(
                star_slice.fixed_shadow_values[variable]
                for variable in star_slice.nonstar_variables
            ),
            expected_signs,
        )
        self.assertTrue(
            all(
                star_slice.fixed_shadow_values[variable] == 0
                for variable in star_slice.star_variables
            )
        )
        self.assertEqual(
            system.evaluate(star_slice.fixed_shadow_values),
            (Fraction(0),) * 28,
        )

        star = set(star_slice.star_variables)
        self.assertTrue(
            all(
                sum(variable in star for variable in monomial) == 1
                for equation in system.equation_terms
                for monomial in equation
            )
        )
        self.assertEqual(len(star_slice.coefficient_matrix), 28)
        self.assertTrue(
            all(
                len(row) == 30
                for row in star_slice.coefficient_matrix
            )
        )

    def test_seed_912_minor_is_nonzero_exactly_and_mod_31(self):
        star_slice = generate_n6_d3_shadow_star_slice()
        self.assertEqual(star_slice.pivot_columns, tuple(range(28)))
        self.assertEqual(star_slice.free_columns, (28, 29))
        self.assertEqual(
            star_slice.pivot_determinant,
            EXPECTED_PIVOT_DETERMINANT,
        )
        self.assertEqual(
            star_slice.pivot_determinant,
            2**54,
        )
        self.assertEqual(
            star_slice.pivot_determinant_mod_31,
            EXPECTED_PIVOT_DETERMINANT_MOD_31,
        )
        self.assertEqual(star_slice.pivot_determinant_mod_31, 16)
        self.assertTrue(star_slice.target_generic_over_q)

    def test_canonical_ghz_solution_replays_all_28_coefficients(self):
        system = generate_equal_g_shadow_system(6, 3)
        target = system.target_coefficients(
            canonical_ghz_target(6, 3)
        )
        solution = canonical_ghz_shadow_star_solution()

        self.assertTrue(solution.exact)
        self.assertEqual(solution.target_coefficients, target)
        self.assertEqual(solution.output_coefficients, target)
        self.assertEqual(
            system.evaluate(solution.shadow_values), target
        )
        self.assertEqual(
            solution.pivot_values,
            EXPECTED_CANONICAL_GHZ_PIVOT_VALUES,
        )
        self.assertEqual(
            tuple(
                solution.shadow_values[index]
                for index in range(28)
            ),
            EXPECTED_CANONICAL_GHZ_PIVOT_VALUES,
        )
        self.assertEqual(solution.free_star_values, (0, 0))
        self.assertEqual(solution.shadow_values[28:30], (0, 0))
        self.assertTrue(
            all(
                value.denominator in {1, 2, 4, 8, 32}
                for value in solution.pivot_values
            )
        )

    def test_target_generic_solver_accepts_arbitrary_rational_vector(self):
        target = tuple(
            Fraction((7 * index) % 17 - 8, index % 5 + 1)
            for index in range(28)
        )
        solution = solve_n6_d3_shadow_occupations(target)
        self.assertTrue(solution.exact)
        self.assertEqual(solution.target_coefficients, target)
        self.assertEqual(solution.output_coefficients, target)
        self.assertEqual(solution.free_star_values, (0, 0))

        zero = solve_n6_d3_shadow_occupations((0,) * 28)
        self.assertTrue(zero.exact)
        self.assertEqual(zero.pivot_values, (0,) * 28)
        self.assertEqual(zero.output_coefficients, (0,) * 28)

    def test_target_validation_fails_closed(self):
        with self.assertRaisesRegex(
            KrennShadowSliceError, "length 28"
        ):
            solve_n6_d3_shadow_occupations((0,) * 27)
        with self.assertRaisesRegex(
            KrennShadowSliceError, "exact rational"
        ):
            solve_n6_d3_shadow_occupations(
                (0,) * 27 + (0.5,)
            )

    def test_slice_module_has_no_private_engine_dependency(self):
        root = Path(__file__).resolve().parents[1]
        source = (
            root
            / "experiments"
            / "krenn_quantum_graph"
            / "shadow_slices.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("section12_instantiation", source)


if __name__ == "__main__":
    unittest.main()
