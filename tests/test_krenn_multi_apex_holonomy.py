from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.border_image import (
    LAURENT_ONE,
    LaurentPolynomial,
    natural_laurent_weight_entries,
)
from experiments.krenn_quantum_graph.fixtures import fixture_n4_d3
from experiments.krenn_quantum_graph.multi_apex_holonomy import (
    KrennMultiApexHolonomyError,
    abstract_flat_marginal_counterexample,
    build_multi_apex_holonomy_gate,
    cross_apex_transition,
    certify_natural_stabilizer_orbit,
    laurent_determinant,
    laurent_induced_matching_sum,
    laurent_specialize,
    lexicographic_independent_rows,
    natural_stabilizer_exponents,
    trace_holonomy_coefficient,
    verify_multi_apex_holonomy_bundle,
    verify_multi_apex_holonomy_gate,
    write_multi_apex_holonomy_bundle,
)
from experiments.krenn_quantum_graph.source_ideal import (
    exact_two_row_dual,
)
from experiments.krenn_quantum_graph.system import variable_key


RESULTS_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "krenn_quantum_graph"
    / "n6_d3_multi_apex_holonomy_gate"
)


def _fraction_matrix_product(left, right):
    return tuple(
        tuple(
            sum(
                left[row][middle] * right[middle][column]
                for middle in range(len(right))
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


class KrennMultiApexHolonomyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = build_multi_apex_holonomy_gate()

    def test_known_path_is_full_GHZ_stabilizer_orbit(self):
        record = self.certificate["stabilizer_orbit"]
        self.assertTrue(
            record["known_path_is_exact_full_GHZ_stabilizer_orbit"]
        )
        self.assertEqual(record["color_sums"], [0, 0, 0])
        self.assertEqual(record["victim_character"], 1)
        self.assertEqual(record["constant_GHZ_characters"], [0, 0, 0])
        self.assertEqual(record["parameter_domain"], "G_m (t != 0)")
        exponents = natural_stabilizer_exponents()
        self.assertEqual(exponents[0], 1)
        self.assertEqual(exponents[6], -1)
        self.assertEqual(sum(value != 0 for value in exponents), 2)

        entries = dict(natural_laurent_weight_entries())
        for index, character, valuation, expression in record[
            "active_weight_index_character_valuation"
        ]:
            i, j, a, b = variable_key(6, 3, index)
            self.assertEqual(
                character,
                exponents[3 * i + a] + exponents[3 * j + b],
            )
            self.assertEqual(character, valuation)
            self.assertEqual(entries[index].terms[0][0], valuation)
            self.assertEqual(entries[index].to_expression(), expression)

        corrupted = dict(entries)
        first = min(corrupted)
        corrupted[first] = corrupted[first] + LAURENT_ONE
        with self.assertRaisesRegex(
            KrennMultiApexHolonomyError, "not exactly"
        ):
            certify_natural_stabilizer_orbit(corrupted)

    def test_cross_apex_and_apex_minor_gate(self):
        transitions = self.certificate["chart_free_transition"]
        self.assertEqual(
            transitions["k4"],
            {
                "ordered_off_diagonal_transitions": 12,
                "identically_zero_transitions": 12,
            },
        )
        self.assertEqual(
            transitions["n6_laurent"],
            {
                "ordered_off_diagonal_transitions": 30,
                "identically_zero_transitions": 30,
            },
        )
        minors = self.certificate["deterministic_apex_minors"]
        self.assertEqual(
            [record["determinant"] for record in minors["k4"]],
            ["-1", "1", "-1", "1"],
        )
        self.assertEqual(
            [record["determinant"] for record in minors["n6_laurent"]],
            ["t^-9", "-t^-3", "-t^6", "-t^6", "-1", "1"],
        )
        self.assertEqual(minors["n6_valuation_sum"], 0)
        self.assertEqual(minors["n6_product"], "1")

        k4_weights = {
            index: LaurentPolynomial.constant(value)
            for index, value in fixture_n4_d3().entries
        }
        transition = cross_apex_transition(4, 3, k4_weights, 0, 1)
        self.assertTrue(
            all(value.is_zero for row in transition for value in row)
        )

    def test_unique_invariant_plucker_terms_and_marginal_products(self):
        plucker = self.certificate["monochromatic_plucker_pairing"]
        for label in ("k4", "n6_laurent"):
            for record in plucker[label]:
                self.assertEqual(record["invariant_pairing"], "1")
                self.assertEqual(
                    record["cofactor_product_valuation"]
                    + record["star_determinant_valuation"],
                    0,
                )

        marginal = self.certificate["monochromatic_marginal_holonomy"]
        self.assertEqual(marginal["k4"]["permutation"], [0, 1, 2, 3])
        self.assertEqual(marginal["k4"]["trace"], "4")
        self.assertEqual(marginal["k4"]["determinant"], "1")
        self.assertEqual(
            marginal["n6_laurent"]["permutation"],
            [2, 4, 1, 3, 0, 5],
        )
        self.assertEqual(
            marginal["n6_laurent"]["cycles"],
            [[0, 2, 1, 4], [3], [5]],
        )
        self.assertEqual(marginal["n6_laurent"]["trace"], "2")
        self.assertEqual(marginal["n6_laurent"]["determinant"], "-1")

    def test_abstract_flat_n6_marginals_have_no_exclusion_power(self):
        matrices = abstract_flat_marginal_counterexample()
        identity = tuple(
            tuple(Fraction(int(row == column)) for column in range(6))
            for row in range(6)
        )
        self.assertEqual(
            _fraction_matrix_product(
                _fraction_matrix_product(matrices[0], matrices[1]),
                matrices[2],
            ),
            identity,
        )
        for matrix in matrices:
            self.assertEqual(
                matrix,
                tuple(tuple(row) for row in zip(*matrix, strict=True)),
            )
            self.assertEqual(
                tuple(matrix[index][index] for index in range(6)),
                (Fraction(0),) * 6,
            )
            self.assertEqual(
                tuple(sum(row) for row in matrix),
                (Fraction(1),) * 6,
            )

    def test_degree9_trace_identity_has_exact_Q_dual_obstruction(self):
        gate = self.certificate["degree9_mixed_ideal_gate"]
        self.assertEqual(
            gate["coefficient_histogram"],
            {"-6": 1845, "-4": 1440, "-2": 90},
        )
        self.assertEqual(gate["S6_x_S3_invariance_checks"], 23_625)
        self.assertEqual(gate["F_times_ones_zero_checks"], 20_250)
        self.assertEqual(
            gate["trace_coefficients_on_dual_rows"], [-6, -4]
        )
        self.assertEqual(gate["trace_rhs_on_dual_rows"], [-2160, -4320])
        self.assertEqual(gate["dual_pairing"], 2160)
        self.assertTrue(gate["reynolds_invariant_search_lossless_over_Q"])
        self.assertTrue(
            gate["raw_source_map_annihilation_follows_by_reynolds_lift"]
        )
        self.assertFalse(gate["trace_polynomial_in_J_mix_over_Q"])
        self.assertFalse(gate["every_matrix_entry_in_J_mix_over_Q"])
        self.assertFalse(gate["radical_membership_decided"])

        dual = exact_two_row_dual()
        self.assertEqual(
            tuple(trace_holonomy_coefficient(key) for key in dual.row_keys),
            (-6, -4),
        )
        self.assertTrue(dual.lambda_transpose_B_zero)

    def test_small_exact_helpers_fail_closed(self):
        value = LaurentPolynomial(
            ((-2, Fraction(3, 2)), (1, Fraction(-4, 3)))
        )
        self.assertEqual(
            laurent_specialize(value, Fraction(2)),
            Fraction(3, 8) - Fraction(8, 3),
        )
        with self.assertRaisesRegex(
            KrennMultiApexHolonomyError, "nonzero"
        ):
            laurent_specialize(value, 0)
        with self.assertRaisesRegex(
            KrennMultiApexHolonomyError, "size 1..15"
        ):
            laurent_determinant(())
        with self.assertRaisesRegex(
            KrennMultiApexHolonomyError, "full column rank"
        ):
            lexicographic_independent_rows(
                (
                    (LAURENT_ONE, LAURENT_ONE),
                    (LAURENT_ONE, LAURENT_ONE),
                )
            )
        with self.assertRaisesRegex(
            KrennMultiApexHolonomyError, "invalid coloring"
        ):
            laurent_induced_matching_sum(
                4,
                3,
                {},
                (0, 0, 0),
                (0, 1),
            )
        with self.assertRaisesRegex(
            KrennMultiApexHolonomyError, "degree nine"
        ):
            trace_holonomy_coefficient((0,))

    def test_certificate_and_bundle_round_trip_fail_closed(self):
        self.assertEqual(
            verify_multi_apex_holonomy_gate(self.certificate),
            self.certificate,
        )
        mutable = build_multi_apex_holonomy_gate()
        mutable["claim_boundary"][
            "affine_GHZ_membership_decided_by_this_gate"
        ] = True
        self.assertFalse(
            build_multi_apex_holonomy_gate()["claim_boundary"][
                "affine_GHZ_membership_decided_by_this_gate"
            ]
        )
        corrupted = deepcopy(self.certificate)
        corrupted["claim_boundary"][
            "affine_GHZ_membership_decided_by_this_gate"
        ] = True
        with self.assertRaisesRegex(
            KrennMultiApexHolonomyError, "differs"
        ):
            verify_multi_apex_holonomy_gate(corrupted)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            write_multi_apex_holonomy_bundle(directory)
            verify_multi_apex_holonomy_bundle(directory)

            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_files"].pop(
                "experiments/krenn_quantum_graph/fixtures.py"
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennMultiApexHolonomyError, "manifest differs"
            ):
                verify_multi_apex_holonomy_bundle(directory)

            write_multi_apex_holonomy_bundle(directory)
            readme_path = directory / "README.md"
            readme_path.write_text(
                readme_path.read_text(encoding="utf-8") + "\ncorruption\n",
                encoding="utf-8",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            readme_bytes = readme_path.read_bytes()
            manifest["bundle_files"]["README.md"] = {
                "bytes": len(readme_bytes),
                "sha256": sha256(readme_bytes).hexdigest(),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennMultiApexHolonomyError, "README differs"
            ):
                verify_multi_apex_holonomy_bundle(directory)

            write_multi_apex_holonomy_bundle(directory)
            certificate_path = directory / "certificate.json"
            payload = json.loads(certificate_path.read_text(encoding="utf-8"))
            payload["degree9_mixed_ideal_gate"]["dual_pairing"] = 0
            certificate_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennMultiApexHolonomyError, "differs"
            ):
                verify_multi_apex_holonomy_bundle(directory)

    def test_committed_bundle_verifies(self):
        verify_multi_apex_holonomy_bundle(RESULTS_DIRECTORY)


if __name__ == "__main__":
    unittest.main()
