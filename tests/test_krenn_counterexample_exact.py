from fractions import Fraction
import unittest

import numpy as np

from experiments.krenn_quantum_graph.counterexample_exact import (
    ExactCandidateWitness,
    ExactVerificationReport,
    FieldElement,
    KrennCounterexampleExactError,
    NumberField,
    cyclotomic_field,
    gaussian_rational_field,
    rational_field,
    reconstruct_complex_candidate,
    verify_exact_candidate,
)
from experiments.krenn_quantum_graph.fixtures import (
    fixture_n6_d2,
)
from experiments.krenn_quantum_graph.system import (
    variable_count,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    n6_d3_seed_witness,
)


class KrennCounterexampleExactTests(unittest.TestCase):
    def test_number_fields_and_elements_round_trip_canonically(self):
        fields = (
            rational_field(),
            gaussian_rational_field(),
            cyclotomic_field(3),
            cyclotomic_field(5),
            NumberField((-2, 0, 1)),
        )
        for field in fields:
            with self.subTest(polynomial=field.minimal_polynomial):
                self.assertEqual(
                    NumberField.from_json(field.to_json()),
                    field,
                )
                element = field.coerce(
                    tuple(
                        Fraction(index + 1, index + 2)
                        for index in range(field.degree)
                    )
                )
                self.assertEqual(
                    FieldElement.from_json(element.to_json()),
                    element,
                )
                if not element.is_zero:
                    self.assertEqual(
                        element * element.inverse(),
                        field.one,
                    )
        gaussian = gaussian_rational_field()
        imaginary = gaussian.generator
        self.assertEqual(
            imaginary * imaginary,
            gaussian.coerce(-1),
        )

    def test_natural_seed_has_the_same_single_defect_in_each_field(self):
        source = n6_d3_seed_witness()
        for field in (
            rational_field(),
            gaussian_rational_field(),
            cyclotomic_field(3),
            cyclotomic_field(5),
        ):
            with self.subTest(field=field.minimal_polynomial):
                witness = ExactCandidateWitness.from_index_values(
                    6,
                    3,
                    field,
                    source.entries,
                )
                report = verify_exact_candidate(witness)
                self.assertFalse(report.exact)
                self.assertTrue(report.enumerators_agree)
                self.assertEqual(
                    report.primary_nonzero_equations,
                    (70,),
                )
                self.assertEqual(
                    report.independent_nonzero_equations,
                    (70,),
                )
                self.assertEqual(report.equation_count, 729)

    def test_gaussian_product_one_gauge_is_verified_twice(self):
        field = gaussian_rational_field()
        imaginary = field.generator
        vertex_scalars = (
            imaginary,
            -imaginary,
            field.one,
            field.one,
            field.one,
            field.one,
        )
        entries = []
        for index, value in fixture_n6_d2().entries:
            i, j, _a, _b = variable_key(6, 2, index)
            entries.append(
                (
                    index,
                    field.coerce(value)
                    * vertex_scalars[i]
                    * vertex_scalars[j],
                )
            )
        witness = ExactCandidateWitness.from_index_values(
            6, 2, field, entries
        )
        report = verify_exact_candidate(witness)
        self.assertTrue(report.exact)
        self.assertEqual(
            ExactCandidateWitness.from_json(witness.to_json()),
            witness,
        )
        self.assertEqual(
            ExactVerificationReport.from_json(report.to_json()),
            report,
        )

    def test_denominator_31_is_exact_without_an_f31_gate(self):
        field = rational_field()
        vertex_scalars = (
            field.coerce(31),
            field.coerce(Fraction(1, 31)),
            field.one,
            field.one,
            field.one,
            field.one,
        )
        entries = []
        for index, value in fixture_n6_d2().entries:
            i, j, _a, _b = variable_key(6, 2, index)
            entries.append(
                (
                    index,
                    field.coerce(value)
                    * vertex_scalars[i]
                    * vertex_scalars[j],
                )
            )
        witness = ExactCandidateWitness.from_index_values(
            6, 2, field, entries
        )
        self.assertTrue(
            any(
                coefficient.denominator == 31
                for _index, value in witness.entries
                for coefficient in value.coefficients
            )
        )
        report = verify_exact_candidate(witness)
        self.assertTrue(report.exact)
        self.assertFalse(
            report.to_dict()["claim_boundary"][
                "f31_used_as_characteristic_zero_evidence"
            ]
        )

    def test_bounded_reconstruction_finds_q_and_gaussian_q(self):
        rational_values = np.zeros(
            variable_count(6, 2), dtype=np.complex128
        )
        rational_support = tuple(
            index for index, _value in fixture_n6_d2().entries
        )
        for index, value in fixture_n6_d2().entries:
            rational_values[index] = complex(value)
        rational = reconstruct_complex_candidate(
            rational_values,
            n=6,
            d=2,
            support=rational_support,
        )
        self.assertTrue(rational.exact)
        self.assertEqual(rational.attempts[0].label, "Q")

        gaussian_values = rational_values.copy()
        gaussian_values[19] *= 1j
        gaussian_values[23] *= -1j
        gaussian = reconstruct_complex_candidate(
            gaussian_values,
            n=6,
            d=2,
            support=rational_support,
        )
        self.assertTrue(gaussian.exact)
        self.assertEqual(
            gaussian.exact_attempts[0].method,
            "gaussian-rational",
        )

    def test_duplicate_json_and_semantic_tampering_fail_closed(self):
        with self.assertRaises(KrennCounterexampleExactError):
            NumberField.from_json(
                '{"schema":"a","schema":"b"}'
            )
        witness = ExactCandidateWitness.from_index_values(
            6,
            3,
            rational_field(),
            n6_d3_seed_witness().entries,
        )
        payload = verify_exact_candidate(witness).to_dict()
        payload["primary_nonzero_equations"] = []
        with self.assertRaises(KrennCounterexampleExactError):
            ExactVerificationReport.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
