import json
import hashlib
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.krenn_quantum_graph.blocker_quotient_prime_sweep import (
    ALGORITHM,
    CLAIM_KEYS,
    CPUS,
    DEGREE24_SPEC,
    DEGREE30_SPEC,
    ENGINE_BUDGET_SECONDS,
    MAXIMUM_WORKERS,
    MEMORY_GIB,
    PROBES,
    SPECS,
    SWEEP_PRIMES,
    KrennBlockerPrimeSweepError,
    PrimeProbe,
    _coefficient_vector,
    _baseline_provenance,
    _exclusive_sweep_lock,
    _summary_run,
    _sweep_verification_for_chart,
    chart_for_spec,
    claim_boundary,
    coefficient_audit,
    deterministic_degree30_blocker_chart,
    docker_argv,
    prepare_probe_input,
    probe_script,
    replay_probe,
    run_probe,
    run_sweep,
    verify_prime_policy,
)
from experiments.krenn_quantum_graph.blocker_quotient_reconnaissance import (
    COMPLETION_MARKER,
    DEFAULT_COEFFICIENT_HEIGHT,
    DEFAULT_SPECIALIZATION_SEED,
    PARSE_MARKER,
    TRANSCRIPT_SCHEMA,
    chart_sha256,
    parse_singular_blocker_output,
)
from experiments.krenn_quantum_graph.blocker_quotient_runner import (
    DOCKER_IMAGE_ID,
)
from experiments.krenn_quantum_graph.perfect_matching_blocker_ideals import (
    build_tutte_barrier_affine_chart,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


EXPECTED_PRIMES = (
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
    353,
    359,
    367,
    373,
)


def _unit_transcript(probe):
    chart = chart_for_spec(probe.spec)
    return "\n".join(
        (
            PARSE_MARKER,
            f"transcript_schema={TRANSCRIPT_SCHEMA}",
            f"chart_sha256={chart_sha256(chart)}",
            f"characteristic={probe.characteristic}",
            f"algorithm={ALGORITHM}",
            f"variable_count={chart.variable_count}",
            "timer_ticks=1",
            "basis_size=1",
            "unit_ideal=1",
            "krull_dimension=-1",
            COMPLETION_MARKER,
            "",
        )
    )


def _linear_degree30_chart():
    blocker = DEGREE30_SPEC.blocker_type
    edges = blocker.blocker_edges
    chosen_endpoints = None
    for mask in range(1 << len(edges)):
        counts = {vertex: 0 for vertex in range(2, 8)}
        endpoints = []
        for index, edge in enumerate(edges):
            endpoint = edge[(mask >> index) & 1]
            endpoints.append(endpoint)
            counts[endpoint] += 1
        if set(counts.values()) == {2}:
            chosen_endpoints = tuple(endpoints)
            break
    if chosen_endpoints is None:
        raise AssertionError("degree-30 orientation disappeared")
    incoming = {vertex: [] for vertex in range(2, 8)}
    for edge, endpoint in zip(edges, chosen_endpoints, strict=True):
        incoming[endpoint].append(edge)
    coordinates = {}
    for vertex, assigned_edges in incoming.items():
        for color, edge in zip(
            (1, 2), sorted(assigned_edges), strict=True
        ):
            first, second = edge
            if vertex == first:
                coordinates[(first, second, color, 0)] = 1
            else:
                coordinates[(first, second, 0, color)] = 1
    chart = build_tutte_barrier_affine_chart(
        SparseWitness.from_coordinates(8, 3, coordinates),
        blocker,
        0,
    )
    expected = {
        (((index,), 1),) for index in range(chart.variable_count)
    }
    if set(chart.equations) != expected:
        raise AssertionError("linear degree-30 chart changed")
    return chart


def _dimension_one_transcript(chart, prime=257):
    lines = [
        PARSE_MARKER,
        f"transcript_schema={TRANSCRIPT_SCHEMA}",
        f"chart_sha256={chart_sha256(chart)}",
        f"characteristic={prime}",
        f"algorithm={ALGORITHM}",
        f"variable_count={chart.variable_count}",
        "timer_ticks=1",
        "basis_size=1",
        "unit_ideal=0",
        "krull_dimension=0",
        "quotient_dimension=1",
        "standard_basis_size=1",
        "standard_basis[0]=1",
    ]
    lines.extend(
        f"normal_form[{variable},0]=0"
        for variable in range(chart.variable_count)
    )
    lines.append(COMPLETION_MARKER)
    return "\n".join(lines) + "\n"


class KrennBlockerQuotientPrimeSweepTest(unittest.TestCase):
    def test_prime_policy_is_exact_nonadaptive_and_coefficient_injective(
        self,
    ):
        self.assertEqual(SWEEP_PRIMES, EXPECTED_PRIMES)
        policy = verify_prime_policy()
        self.assertEqual(policy["ordered_primes"], list(EXPECTED_PRIMES))
        self.assertTrue(policy["nonadaptive"])
        self.assertEqual(
            SWEEP_PRIMES[0], 2 * DEFAULT_COEFFICIENT_HEIGHT + 3
        )
        for spec in SPECS:
            chart = chart_for_spec(spec)
            for prime in SWEEP_PRIMES:
                with self.subTest(chart=spec.key, prime=prime):
                    audit = coefficient_audit(chart, prime)
                    self.assertEqual(
                        audit["coefficient_count"],
                        spec.coefficient_slot_count,
                    )
                    self.assertEqual(audit["zero_residue_count"], 0)
                    self.assertEqual(
                        audit["distinct_residue_count"],
                        spec.coefficient_slot_count,
                    )
                    self.assertTrue(audit["coefficient_injective"])
        with self.assertRaisesRegex(
            KrennBlockerPrimeSweepError, "outside the sweep"
        ):
            coefficient_audit(chart_for_spec(DEGREE24_SPEC), 31)

    def test_p31_is_a_colliding_control_not_a_coefficient_injective_prime(
        self,
    ):
        for spec in SPECS:
            coefficients = _coefficient_vector(chart_for_spec(spec))
            integers = tuple(value.numerator for value in coefficients)
            self.assertTrue(
                all(
                    value.denominator == 1 and value.numerator % 31
                    for value in coefficients
                )
            )
            self.assertEqual(len(set(value % 31 for value in integers)), 30)
        provenance = _baseline_provenance()
        self.assertEqual(
            provenance["p31_control"][
                "verified_cyclic_module_dimension"
            ],
            22,
        )
        self.assertEqual(
            provenance["q_fixed_specialization"][
                "verified_cyclic_module_dimension"
            ],
            24,
        )
        self.assertEqual(
            provenance["p31_control"][
                "coefficient_residue_distinct_count"
            ],
            30,
        )

    def test_both_exact_square_charts_and_degree30_edges(self):
        self.assertEqual(DEFAULT_SPECIALIZATION_SEED, 80320260725)
        degree24 = chart_for_spec(DEGREE24_SPEC)
        degree30 = deterministic_degree30_blocker_chart()
        self.assertEqual(degree24.variable_count, 10)
        self.assertEqual(degree24.equation_count, 10)
        self.assertEqual(
            chart_sha256(degree24),
            "66a0e966a591ed1d28fd7cd98251f274940a746892a203603"
            "ff76e6e33595ab5",
        )
        self.assertEqual(degree30.barrier_vertices, (0, 1))
        self.assertEqual(
            degree30.components, ((2, 3, 4), (5,), (6,), (7,))
        )
        self.assertEqual(
            degree30.blocker_edges,
            (
                (2, 5),
                (2, 6),
                (2, 7),
                (3, 5),
                (3, 6),
                (3, 7),
                (4, 5),
                (4, 6),
                (4, 7),
                (5, 6),
                (5, 7),
                (6, 7),
            ),
        )
        self.assertEqual(degree30.variable_count, 12)
        self.assertEqual(degree30.equation_count, 12)
        self.assertEqual(tuple(map(len, degree30.equations)), (9,) * 12)
        coefficients = _coefficient_vector(degree30)
        self.assertEqual(len(coefficients), 108)
        self.assertEqual(len(set(coefficients)), 108)
        self.assertTrue(all(coefficient for coefficient in coefficients))
        self.assertEqual(
            chart_sha256(degree30),
            "cc020528fe57d893f874aef801e1f8d31adb235197266ef2"
            "1ec85eb2dc666d99",
        )

    def test_scripts_are_bound_to_chart_and_characteristic(self):
        for spec in SPECS:
            probe = PrimeProbe(spec, 257)
            script = probe_script(probe)
            chart = chart_for_spec(spec)
            self.assertIn(f"ring r=257,", script)
            self.assertIn(f"chart_sha256={chart_sha256(chart)}", script)
            self.assertIn(f"variable_count={chart.variable_count}", script)
            self.assertIn(PARSE_MARKER, script)
            self.assertIn(COMPLETION_MARKER, script)
            self.assertNotIn("sat(", script)
            self.assertNotIn("Rabinowitsch", script)
            transcript = _unit_transcript(probe)
            outcome = parse_singular_blocker_output(
                chart,
                transcript,
                characteristic=257,
                algorithm=ALGORITHM,
            )
            self.assertTrue(outcome.unit_ideal)
            with self.assertRaises(ValueError):
                parse_singular_blocker_output(
                    chart,
                    transcript,
                    characteristic=263,
                    algorithm=ALGORITHM,
                )

    def test_boundary_script_hashes_are_stable(self):
        expected = {
            (DEGREE24_SPEC.key, 257): (
                "3c1352bedf64d3b8287faf4c66141cea3e62c47d4ac66bb4"
                "372f5d333c6f422d"
            ),
            (DEGREE24_SPEC.key, 373): (
                "022a1aa2473427e4f21dd44a4af5684c7c94ebb572c7fa10"
                "0d70fadc3109eaf8"
            ),
            (DEGREE30_SPEC.key, 257): (
                "714591e7c17181140bc616abdde7c5a045c0d4d8585aa86c"
                "e6672d239a5e6f62"
            ),
            (DEGREE30_SPEC.key, 373): (
                "b450f0180bc0851b13e384621cd469465d7672c113deff1db"
                "2c63a8f014fa795"
            ),
        }
        for spec in SPECS:
            for prime in (257, 373):
                with self.subTest(chart=spec.key, prime=prime):
                    digest = hashlib.sha256(
                        probe_script(
                            PrimeProbe(spec, prime)
                        ).encode("utf-8")
                    ).hexdigest()
                    self.assertEqual(digest, expected[(spec.key, prime)])

    def test_finite_degree30_wire_replay_has_native_sweep_schema(self):
        chart = _linear_degree30_chart()
        probe = PrimeProbe(DEGREE30_SPEC, 257)
        outcome = parse_singular_blocker_output(
            chart,
            _dimension_one_transcript(chart),
            characteristic=257,
            algorithm=ALGORITHM,
        )
        report = _sweep_verification_for_chart(
            probe, chart, outcome.representation
        )
        self.assertEqual(
            report["schema"],
            "krenn-n8-square-blocker-prime-verification-v2",
        )
        self.assertEqual(report["variable_count"], 12)
        self.assertEqual(report["equation_count"], 12)
        self.assertEqual(report["commutator_checks"], 66)
        self.assertEqual(report["equation_matrix_checks"], 12)
        self.assertEqual(report["cyclic_basis_checks"], 1)
        self.assertTrue(report["exact_replay"])
        self.assertTrue(
            report["legacy_wire_parser"][
                "reused_for_wire_parsing_and_exact_matrix_arithmetic_only"
            ]
        )
        self.assertTrue(
            report["exact_conclusion"][
                "fixed_specialized_affine_ideal_proper_over_field"
            ]
        )
        self.assertTrue(
            all(
                value is False
                for value in report["claim_boundary"].values()
            )
        )

    def test_resource_plan_is_serial_offline_and_pinned(self):
        self.assertEqual(len(PROBES), 40)
        self.assertEqual(MAXIMUM_WORKERS, 1)
        self.assertEqual(CPUS, 2)
        self.assertLessEqual(CPUS * MAXIMUM_WORKERS, 12)
        self.assertEqual(MEMORY_GIB, 4)
        self.assertEqual(ENGINE_BUDGET_SECONDS, 600)
        self.assertEqual(
            tuple(probe.spec for probe in PROBES[:20]),
            (DEGREE24_SPEC,) * 20,
        )
        self.assertEqual(
            tuple(probe.spec for probe in PROBES[20:]),
            (DEGREE30_SPEC,) * 20,
        )
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            probe = PrimeProbe(DEGREE30_SPEC, 257)
            first = prepare_probe_input(probe, scratch)
            second = prepare_probe_input(probe, scratch)
            self.assertEqual(first, second)
            argv = docker_argv(probe, scratch)
            self.assertIn(DOCKER_IMAGE_ID, argv)
            self.assertIn("none", argv)
            self.assertIn("600s", argv)
            self.assertIn("4g", argv)
            self.assertIn("--cpus", argv)
            self.assertNotIn("hodgepodge", " ".join(argv).lower())

    def test_timeout_receipt_round_trips_and_stays_nonproof(self):
        probe = PrimeProbe(DEGREE30_SPEC, 257)
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(f"{PARSE_MARKER}\n".encode("ascii"))
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 124)

            with (
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_prime_sweep.inspect_engine",
                    return_value=DOCKER_IMAGE_ID,
                ),
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_prime_sweep.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                receipt = run_probe(probe, scratch)
            self.assertEqual(
                receipt["outcome"]["status"], "timeout-after-parse"
            )
            self.assertTrue(
                all(
                    value is False
                    for value in receipt["claim_boundary"].values()
                )
            )
            self.assertEqual(replay_probe(probe, scratch), receipt)
            self.assertFalse((scratch / "runs" / ".sweep.lock").exists())

    def test_backend_stdout_must_be_strict_utf8(self):
        probe = PrimeProbe(DEGREE24_SPEC, 257)
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(b"\xff")
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 0)

            with (
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_prime_sweep.inspect_engine",
                    return_value=DOCKER_IMAGE_ID,
                ),
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_prime_sweep.subprocess.run",
                    side_effect=fake_run,
                ),
                self.assertRaisesRegex(
                    KrennBlockerPrimeSweepError, "strict UTF-8"
                ),
            ):
                run_probe(probe, scratch)
            self.assertFalse((scratch / "runs" / ".sweep.lock").exists())

    def test_completed_backend_unit_report_is_not_promoted(self):
        probe = PrimeProbe(DEGREE24_SPEC, 257)
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            transcript = _unit_transcript(probe).encode("ascii")

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(transcript)
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 0)

            with (
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_prime_sweep.inspect_engine",
                    return_value=DOCKER_IMAGE_ID,
                ),
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_prime_sweep.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                receipt = run_probe(probe, scratch)
            self.assertEqual(
                receipt["outcome"]["status"], "completed-unit-ideal"
            )
            self.assertIsNone(receipt["representation"])
            self.assertIsNone(receipt["verification"])
            self.assertFalse(
                receipt["claim_boundary"][
                    "timeout_miss_or_backend_unit_report_is_proof"
                ]
            )
            receipt_path = (
                scratch / "runs" / f"{probe.key}.receipt.json"
            )
            payload = json.loads(receipt_path.read_text("utf-8"))
            payload["probe"]["characteristic"] = 263
            receipt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KrennBlockerPrimeSweepError, "metadata"
            ):
                replay_probe(probe, scratch)

    def test_summary_rejects_a_receipt_from_a_different_prime(self):
        first = PrimeProbe(DEGREE24_SPEC, 257)
        second = PrimeProbe(DEGREE24_SPEC, 263)
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)

            def fake_run(argv, **kwargs):
                characteristic = (
                    257 if "p0257" in " ".join(argv) else 263
                )
                probe = first if characteristic == 257 else second
                kwargs["stdout"].write(
                    _unit_transcript(probe).encode("ascii")
                )
                kwargs["stderr"].write(b"")
                return subprocess.CompletedProcess(argv, 0)

            with (
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_prime_sweep.inspect_engine",
                    return_value=DOCKER_IMAGE_ID,
                ),
                patch(
                    "experiments.krenn_quantum_graph."
                    "blocker_quotient_prime_sweep.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                first_receipt = run_probe(first, scratch)
                second_receipt = run_probe(second, scratch)
            _summary_run(first, first_receipt)
            with self.assertRaisesRegex(
                KrennBlockerPrimeSweepError, "paired probe"
            ):
                _summary_run(first, second_receipt)
            changed = deepcopy(first_receipt)
            changed["elapsed_seconds"] += 1
            with self.assertRaisesRegex(
                KrennBlockerPrimeSweepError, "provenance"
            ):
                _summary_run(first, changed)

    def test_batch_stops_after_first_nonfinite_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            failure = {
                "outcome": {"status": "timeout-after-parse"}
            }
            with patch(
                "experiments.krenn_quantum_graph."
                "blocker_quotient_prime_sweep._run_probe_locked",
                return_value=failure,
            ) as mocked:
                with self.assertRaisesRegex(
                    KrennBlockerPrimeSweepError,
                    "stopped after the first",
                ):
                    run_sweep(scratch)
            self.assertEqual(mocked.call_count, 1)
            progress = json.loads(
                (scratch / "progress.json").read_text("utf-8")
            )
            self.assertEqual(progress["completed_receipt_count"], 1)
            self.assertFalse(progress["all_receipts_present"])
            self.assertFalse(progress["complete"])
            self.assertFalse((scratch / "runs" / ".sweep.lock").exists())

    def test_existing_lock_and_invalid_probes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            runs = scratch / "runs"
            runs.mkdir(parents=True)
            (runs / ".sweep.lock").write_text(
                "another process\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                KrennBlockerPrimeSweepError, "another modular sweep"
            ):
                with _exclusive_sweep_lock(scratch):
                    self.fail("existing lock was ignored")
        for characteristic in (31, 0, 4, True, 257.0):
            with self.subTest(characteristic=characteristic):
                with self.assertRaises(KrennBlockerPrimeSweepError):
                    PrimeProbe(DEGREE24_SPEC, characteristic)
        for bad_prime in (257.0, True):
            with self.subTest(audit_prime=bad_prime):
                with self.assertRaises(KrennBlockerPrimeSweepError):
                    coefficient_audit(
                        chart_for_spec(DEGREE24_SPEC), bad_prime
                    )
        for projective_degree, slot_count in (
            (24.0, 90),
            (24, 90.0),
            (True, 90),
            (24, True),
        ):
            with self.subTest(
                projective_degree=projective_degree,
                slot_count=slot_count,
            ):
                with self.assertRaises(KrennBlockerPrimeSweepError):
                    type(DEGREE24_SPEC)(
                        key="invalid",
                        blocker_type=DEGREE24_SPEC.blocker_type,
                        projective_chow_degree=projective_degree,
                        coefficient_slot_count=slot_count,
                    )

    def test_claim_boundary_is_exhaustively_negative(self):
        claims = claim_boundary()
        self.assertGreaterEqual(len(CLAIM_KEYS), 20)
        self.assertEqual(set(claims), set(CLAIM_KEYS))
        self.assertTrue(
            all(value is False for value in claims.values())
        )
        for required in (
            "coefficient_injective_means_algebraically_good_prime",
            "flatness_or_good_reduction_proved",
            "generic_fiber_dimension_proved",
            "modular_result_is_Q_or_C_proof",
            "p31_is_proved_special_fiber",
            "n8_existence_proved",
            "n8_nonexistence_proved",
        ):
            self.assertIn(required, claims)


if __name__ == "__main__":
    unittest.main()
