from dataclasses import replace
from fractions import Fraction
import hashlib
import unittest

from experiments.krenn_quantum_graph.blocker_quotient_reconnaissance import (
    COMPLETION_MARKER,
    DEFAULT_SPECIALIZATION_SEED,
    KrennBlockerQuotientError,
    PARSE_MARKER,
    TRANSCRIPT_SCHEMA,
    QuotientRepresentation,
    _parse_backend_polynomial,
    chart_sha256,
    deterministic_k5_blocker_chart,
    multiplication_matrix_for_polynomial,
    multiplication_matrix_is_nilpotent,
    parse_singular_blocker_output,
    singular_blocker_quotient_script,
    verify_quotient_representation,
)
from experiments.krenn_quantum_graph.perfect_matching_blocker_ideals import (
    PerfectMatchingBlockerType,
    build_tutte_barrier_affine_chart,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


def _linear_k5_chart():
    """Return the K5 chart whose ten equations are its ten variables."""

    blocker = PerfectMatchingBlockerType(
        8, 3, (1, 1, 1, 1, 1)
    )
    outside = tuple(
        vertex
        for component in blocker.components
        for vertex in component
    )
    coordinates = {}
    assigned = set()
    for first, second in blocker.blocker_edges:
        first_index = outside.index(first)
        second_index = outside.index(second)
        difference = (second_index - first_index) % 5
        if difference in (1, 2):
            tail, head, slot = first, second, difference
        else:
            tail, head = second, first
            slot = (first_index - second_index) % 5
        candidate_color = slot
        assigned.add((tail, candidate_color))
        if tail < head:
            coordinates[(tail, head, candidate_color, 0)] = 1
        else:
            coordinates[(head, tail, 0, candidate_color)] = 1
    expected = {
        (vertex, color)
        for vertex in outside
        for color in (1, 2)
    }
    if assigned != expected:
        raise AssertionError("regular tournament assignment changed")
    witness = SparseWitness.from_coordinates(8, 3, coordinates)
    chart = build_tutte_barrier_affine_chart(
        witness, blocker, 0
    )
    if {
        equation for equation in chart.equations
    } != {
        (((index,), Fraction(1)),)
        for index in range(chart.variable_count)
    }:
        raise AssertionError("linear K5 chart did not expose all variables")
    return chart


def _metadata_lines(chart, characteristic, algorithm):
    return [
        PARSE_MARKER,
        f"transcript_schema={TRANSCRIPT_SCHEMA}",
        f"chart_sha256={chart_sha256(chart)}",
        f"characteristic={characteristic}",
        f"algorithm={algorithm}",
        f"variable_count={chart.variable_count}",
    ]


def _dimension_one_transcript(
    chart, *, characteristic=0, algorithm="std"
):
    lines = [
        *_metadata_lines(chart, characteristic, algorithm),
        "timer_ticks=7",
        "basis_size=10",
        "unit_ideal=0",
        "krull_dimension=0",
        "quotient_dimension=1",
        "standard_basis_size=1",
        "standard_basis[0]=1",
    ]
    lines.extend(
        f"normal_form[{variable},0]=0"
        for variable in range(10)
    )
    lines.append(COMPLETION_MARKER)
    return "\n".join(lines) + "\n"


class KrennBlockerQuotientReconnaissanceTest(unittest.TestCase):
    def test_deterministic_dense_k5_input_and_script_hashes(self):
        chart = deterministic_k5_blocker_chart()
        self.assertEqual(DEFAULT_SPECIALIZATION_SEED, 80320260725)
        self.assertEqual(chart.barrier_vertices, (0, 1, 2))
        self.assertEqual(
            chart.components, ((3,), (4,), (5,), (6,), (7,))
        )
        self.assertEqual(chart.variable_count, 10)
        self.assertEqual(chart.equation_count, 10)
        self.assertEqual(sum(map(len, chart.equations)), 90)
        self.assertTrue(
            all(
                coefficient
                for equation in chart.equations
                for _monomial, coefficient in equation
            )
        )
        self.assertEqual(
            len(
                {
                    coefficient
                    for equation in chart.equations
                    for _monomial, coefficient in equation
                }
            ),
            90,
        )
        self.assertTrue(
            all(
                coefficient.numerator % 31
                for equation in chart.equations
                for _monomial, coefficient in equation
            )
        )
        self.assertEqual(
            chart_sha256(chart),
            "66a0e966a591ed1d28fd7cd98251f274940a746892a203603"
            "ff76e6e33595ab5",
        )
        expected = {
            (31, "std"): (
                "17cc9a2958fac0d6295fc0eb9983092fcec2244dbd858cb3c"
                "781d64ef4050a8a"
            ),
            (31, "slimgb"): (
                "a991811ae7a2d21b6f6675b366ffec0bf0d8b564595011ea"
                "c2f4abf7c5c6f355"
            ),
            (0, "std"): (
                "1e4c3c7bae13e585191298d3d8983f67bf06f18c8a17f1f"
                "7291aa0ee31c6699d"
            ),
            (0, "slimgb"): (
                "5b739a656839133d04dc954ca3d9ddc58ce03b960a824b04"
                "d0702ce8524944d6"
            ),
        }
        for (characteristic, algorithm), digest in expected.items():
            with self.subTest(
                characteristic=characteristic, algorithm=algorithm
            ):
                script = singular_blocker_quotient_script(
                    chart,
                    characteristic=characteristic,
                    algorithm=algorithm,
                )
                self.assertEqual(
                    hashlib.sha256(script.encode("utf-8")).hexdigest(),
                    digest,
                )
                self.assertIn(PARSE_MARKER, script)
                self.assertIn(COMPLETION_MARKER, script)
                self.assertNotIn("sat(", script)
                self.assertNotIn("Rabinowitsch", script)

    def test_exact_dimension_one_representation_round_trip(self):
        chart = _linear_k5_chart()
        outcome = parse_singular_blocker_output(
            chart,
            _dimension_one_transcript(chart),
            characteristic=0,
            algorithm="std",
        )
        self.assertFalse(outcome.unit_ideal)
        self.assertEqual(outcome.backend_quotient_dimension, 1)
        representation = outcome.representation
        self.assertIsNotNone(representation)
        report = verify_quotient_representation(
            chart, representation
        )
        self.assertEqual(report["field"], "Q")
        self.assertEqual(report["representation_dimension"], 1)
        self.assertTrue(
            report["claim_boundary"][
                "proves_complex_escape_for_this_specialization"
            ]
        )
        payload = representation.to_dict()
        self.assertEqual(
            QuotientRepresentation.from_dict(payload),
            representation,
        )
        identity = multiplication_matrix_for_polynomial(
            representation,
            (((0,) * 10, Fraction(1)),),
        )
        zero = multiplication_matrix_for_polynomial(
            representation, ()
        )
        self.assertFalse(
            multiplication_matrix_is_nilpotent(
                representation, identity
            )
        )
        self.assertTrue(
            multiplication_matrix_is_nilpotent(
                representation, zero
            )
        )

    def test_modular_representation_is_not_promoted_to_complex(self):
        chart = _linear_k5_chart()
        outcome = parse_singular_blocker_output(
            chart,
            _dimension_one_transcript(chart, characteristic=31),
            characteristic=31,
            algorithm="std",
        )
        report = verify_quotient_representation(
            chart, outcome.representation
        )
        self.assertEqual(report["field"], "F_31")
        self.assertFalse(
            report["claim_boundary"][
                "proves_complex_escape_for_this_specialization"
            ]
        )
        self.assertFalse(
            report["claim_boundary"][
                "eqsystem_constraints_imposed"
            ]
        )
        self.assertFalse(
            report["claim_boundary"]["n8_boundary_escape_proved"]
        )

    def test_unit_and_positive_dimensional_transcripts_stay_backend_only(
        self,
    ):
        chart = _linear_k5_chart()
        for unit, dimension in ((1, -1), (0, 1)):
            with self.subTest(unit=unit, dimension=dimension):
                transcript = "\n".join(
                    (
                        *_metadata_lines(chart, 31, "std"),
                        "timer_ticks=1",
                        "basis_size=1",
                        f"unit_ideal={unit}",
                        f"krull_dimension={dimension}",
                        COMPLETION_MARKER,
                        "",
                    )
                )
                outcome = parse_singular_blocker_output(
                    chart,
                    transcript,
                    characteristic=31,
                    algorithm="std",
                )
                self.assertEqual(outcome.unit_ideal, bool(unit))
                self.assertEqual(
                    outcome.backend_reports_proper_ideal, not bool(unit)
                )
                self.assertIsNone(outcome.representation)
                self.assertIsNone(
                    outcome.backend_quotient_dimension
                )

    def test_matrix_replay_rejects_equation_commutator_and_cyclic_tampering(
        self,
    ):
        chart = _linear_k5_chart()
        representation = parse_singular_blocker_output(
            chart,
            _dimension_one_transcript(chart, characteristic=31),
            characteristic=31,
            algorithm="std",
        ).representation
        bad_equation_matrices = list(
            representation.multiplication_matrices
        )
        bad_equation_matrices[0] = ((1,),)
        with self.assertRaisesRegex(
            KrennBlockerQuotientError, "equation"
        ):
            verify_quotient_representation(
                chart,
                replace(
                    representation,
                    multiplication_matrices=tuple(
                        bad_equation_matrices
                    ),
                ),
            )

        zero = ((0, 0), (0, 0))
        first = ((0, 1), (0, 0))
        second = ((0, 0), (1, 0))
        with self.assertRaisesRegex(
            KrennBlockerQuotientError, "commute"
        ):
            verify_quotient_representation(
                chart,
                QuotientRepresentation(
                    characteristic=31,
                    variable_count=10,
                    chart_sha256=chart_sha256(chart),
                    basis_monomials=(
                        (0,) * 10,
                        (1,) + (0,) * 9,
                    ),
                    multiplication_matrices=(
                        first,
                        second,
                        *(zero for _index in range(8)),
                    ),
                    backend_reported_quotient_dimension=2,
                ),
            )
        with self.assertRaisesRegex(
            KrennBlockerQuotientError, "basis monomials"
        ):
            verify_quotient_representation(
                chart,
                QuotientRepresentation(
                    characteristic=31,
                    variable_count=10,
                    chart_sha256=chart_sha256(chart),
                    basis_monomials=(
                        (0,) * 10,
                        (1,) + (0,) * 9,
                    ),
                    multiplication_matrices=tuple(
                        zero for _index in range(10)
                    ),
                    backend_reported_quotient_dimension=2,
                ),
            )

    def test_parser_and_public_validation_fail_closed(self):
        chart = _linear_k5_chart()
        transcript = _dimension_one_transcript(chart)
        mutations = (
            transcript.replace(PARSE_MARKER, "", 1),
            transcript.replace(
                PARSE_MARKER, f"{PARSE_MARKER}\n{PARSE_MARKER}", 1
            ),
            transcript.replace("normal_form[9,0]=0\n", ""),
            transcript.replace(
                "standard_basis[0]=1",
                "standard_basis[0]=x10",
            ),
            transcript.replace(
                "normal_form[0,0]=0",
                "normal_form[0,0]=x0",
            ),
            transcript.replace(
                "characteristic=0", "characteristic=31"
            ),
            transcript.replace(
                "algorithm=std", "algorithm=slimgb"
            ),
            transcript.replace(
                f"chart_sha256={chart_sha256(chart)}",
                "chart_sha256=" + "0" * 64,
            ),
            transcript.replace(
                "transcript_schema=" + TRANSCRIPT_SCHEMA,
                "transcript_schema=changed",
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=hashlib.sha256(
                mutation.encode()
            ).hexdigest()[:8]):
                with self.assertRaises(KrennBlockerQuotientError):
                    parse_singular_blocker_output(
                        chart,
                        mutation,
                        characteristic=0,
                        algorithm="std",
                    )
        for characteristic in (1, 4, True, 0.0):
            with self.subTest(characteristic=characteristic):
                with self.assertRaises(
                    KrennBlockerQuotientError
                ):
                    singular_blocker_quotient_script(
                        chart, characteristic=characteristic
                    )
        for seed in (-1, 2**63, True, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaises(
                    KrennBlockerQuotientError
                ):
                    deterministic_k5_blocker_chart(seed=seed)
        representation = parse_singular_blocker_output(
            chart,
            transcript,
            characteristic=0,
            algorithm="std",
        ).representation
        with self.assertRaisesRegex(
            KrennBlockerQuotientError, "exact"
        ):
            multiplication_matrix_for_polynomial(
                representation,
                (((0,) * 10, 0.5),),
            )
        with self.assertRaisesRegex(
            KrennBlockerQuotientError, "nonzero and unique"
        ):
            multiplication_matrix_for_polynomial(
                representation,
                (
                    ((0,) * 10, 1),
                    ((0,) * 10, 2),
                ),
            )
        payload = representation.to_dict()
        payload["multiplication_matrices"][0][0][0] = 0.5
        with self.assertRaises(KrennBlockerQuotientError):
            QuotientRepresentation.from_dict(payload)

    def test_realistic_polynomial_parser_and_exponent_guard(self):
        rational = _parse_backend_polynomial(
            "-1/9*x0+2*x1^2-x0*x3+31*x2-31*x2",
            variable_count=4,
            characteristic=0,
        )
        self.assertEqual(
            rational,
            {
                (1, 0, 0, 0): Fraction(-1, 9),
                (0, 2, 0, 0): Fraction(2),
                (1, 0, 0, 1): Fraction(-1),
            },
        )
        modular = _parse_backend_polynomial(
            "x0+30*x0+1/2*x1",
            variable_count=2,
            characteristic=31,
        )
        self.assertEqual(modular, {(0, 1): 16})
        with self.assertRaisesRegex(
            KrennBlockerQuotientError, "exponent"
        ):
            _parse_backend_polynomial(
                "x0^1000001",
                variable_count=1,
                characteristic=0,
            )


if __name__ == "__main__":
    unittest.main()
