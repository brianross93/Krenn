r"""Deterministic modular sweep for both square ``n=8,d=3`` blockers.

The structural blocker census has two square affine qutrit charts:

* barrier profile ``(2;3,1,1,1)``, with 12 equations and Chow degree 30;
* barrier profile ``(3;1,1,1,1,1)``, with 10 equations and Chow degree 24.

This module measures one fixed dense specialization of each chart at the
first twenty primes larger than twice the coefficient height.  Those primes
are *coefficient-injective*: the distinct signed integer coefficients remain
distinct and nonzero after reduction.  This arithmetic property is not a
good-reduction, flatness, or generic-fiber theorem.

Every finite transcript is parsed into multiplication matrices and replayed
exactly.  A successful replay proves only that the fixed specialized ideal
is proper over that finite field.  It does not identify the matrices with
the full coordinate ring and has no implication over ``Q`` or ``C``.

All backend transcripts and matrices are retained under
``D:\KrennScratch\counterexample_search``.  The runner is serial, offline,
bounded to two CPUs, and uses the pinned standalone Krenn Singular image.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.blocker_quotient_reconnaissance import (
    COMPLETION_MARKER,
    DEFAULT_COEFFICIENT_HEIGHT,
    DEFAULT_SPECIALIZATION_SEED,
    MAXIMUM_TRANSCRIPT_BYTES,
    PARSE_MARKER,
    QuotientRepresentation,
    TRANSCRIPT_SCHEMA,
    chart_sha256,
    deterministic_k5_blocker_chart,
    parse_singular_blocker_output,
    singular_blocker_quotient_script,
    verify_quotient_representation,
)
from experiments.krenn_quantum_graph.blocker_quotient_runner import (
    DOCKER_IMAGE,
    DOCKER_IMAGE_ID,
    inspect_engine,
)
from experiments.krenn_quantum_graph.perfect_matching_blocker_ideals import (
    PerfectMatchingBlockerType,
    TutteBarrierAffineChart,
    build_tutte_barrier_affine_chart,
    top_chow_coefficient,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


SWEEP_SCHEMA = "krenn-n8-square-blocker-prime-sweep-v2"
RUN_SCHEMA = "krenn-n8-square-blocker-prime-probe-v2"
PROGRESS_SCHEMA = "krenn-n8-square-blocker-prime-progress-v2"
RECEIPT_ARCHIVE_SCHEMA = (
    "krenn-n8-square-blocker-prime-receipt-archive-v2"
)
DEFAULT_SCRATCH_ROOT = Path(
    r"D:\KrennScratch\counterexample_search"
    r"\n8_square_blocker_prime_sweep_v3"
)
ALGORITHM = "slimgb"
CPUS = 2
MAXIMUM_WORKERS = 1
MEMORY_GIB = 4
ENGINE_BUDGET_SECONDS = 600
HOST_GRACE_SECONDS = 60
SINGULAR_VERSION = "4.3.2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = Path(__file__).with_name(
    "n8_k5_singular.Dockerfile"
)
BASELINE_BUNDLE = (
    REPOSITORY_ROOT
    / "results"
    / "krenn_quantum_graph"
    / "n8_d3_degree24_quotient_reconnaissance"
)
SWEEP_PRIMES = (
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

CLAIM_KEYS = (
    "all_labeled_blockers_checked",
    "backend_groebner_basis_independently_verified",
    "backend_quotient_dimension_is_independently_full_quotient",
    "coefficient_injective_means_algebraically_good_prime",
    "common_global_weight_specialization_checked",
    "degree24_k5_covers_degree30_type",
    "distinct_points_counted",
    "eqsystem_constraints_imposed",
    "flatness_or_good_reduction_proved",
    "generic_fiber_dimension_proved",
    "generic_prime_statement_proved",
    "hilbert_polynomial_stability_proved",
    "modular_lift_or_reconstruction_proved",
    "modular_result_is_Q_or_C_proof",
    "multiple_coefficient_specializations_checked",
    "n8_boundary_escape_proved",
    "n8_existence_proved",
    "n8_nonexistence_proved",
    "p31_is_proved_special_fiber",
    "quotient_radical_or_reduced_proved",
    "symbolic_108_parameter_quotient_built",
    "symbolic_90_parameter_quotient_built",
    "timeout_miss_or_backend_unit_report_is_proof",
    "universal_parameter_quotient_built",
    "verified_dimension_histogram_is_full_quotient_histogram",
)


def claim_boundary() -> dict[str, bool]:
    """Return the immutable-key, all-false sweep claim boundary."""

    return {key: False for key in CLAIM_KEYS}


VERIFICATION_FALSE_KEYS = (
    "backend_full_quotient_claim_independently_verified",
    "backend_groebner_basis_independently_verified",
    "eqsystem_constraints_imposed",
    "generic_fiber_dimension_proved",
    "modular_result_is_Q_or_C_proof",
    "n8_boundary_escape_proved",
    "n8_existence_proved",
    "n8_nonexistence_proved",
    "symbolic_108_parameter_quotient_built",
    "symbolic_90_parameter_quotient_built",
)


class KrennBlockerPrimeSweepError(RuntimeError):
    """A sweep input, backend transcript, or retained receipt failed closed."""


@dataclass(frozen=True)
class SquareBlockerSpec:
    """One exact square blocker chart used in the modular sweep."""

    key: str
    blocker_type: PerfectMatchingBlockerType
    projective_chow_degree: int
    coefficient_slot_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not self.key
            or any(
                not (
                    character.islower()
                    or character.isdigit()
                    or character == "_"
                )
                for character in self.key
            )
            or self.blocker_type.n != 8
            or not self.blocker_type.qutrit_chart_is_square
            or isinstance(self.projective_chow_degree, bool)
            or not isinstance(self.projective_chow_degree, int)
            or self.projective_chow_degree <= 0
            or isinstance(self.coefficient_slot_count, bool)
            or not isinstance(self.coefficient_slot_count, int)
            or self.coefficient_slot_count
            != 9 * len(self.blocker_type.blocker_edges)
        ):
            raise KrennBlockerPrimeSweepError(
                "invalid square blocker sweep specification"
            )
        outside = tuple(
            vertex
            for component in self.blocker_type.components
            for vertex in component
        )
        if (
            top_chow_coefficient(
                self.blocker_type.blocker_edges, outside
            )
            != self.projective_chow_degree
        ):
            raise KrennBlockerPrimeSweepError(
                "square blocker Chow degree changed"
            )


DEGREE24_SPEC = SquareBlockerSpec(
    key="degree24_b3_11111",
    blocker_type=PerfectMatchingBlockerType(
        8, 3, (1, 1, 1, 1, 1)
    ),
    projective_chow_degree=24,
    coefficient_slot_count=90,
)
DEGREE30_SPEC = SquareBlockerSpec(
    key="degree30_b2_3111",
    blocker_type=PerfectMatchingBlockerType(
        8, 2, (3, 1, 1, 1)
    ),
    projective_chow_degree=30,
    coefficient_slot_count=108,
)
SPECS = (DEGREE24_SPEC, DEGREE30_SPEC)


@dataclass(frozen=True)
class PrimeProbe:
    """One predeclared chart/prime computation."""

    spec: SquareBlockerSpec
    characteristic: int

    def __post_init__(self) -> None:
        if (
            self.spec not in SPECS
            or isinstance(self.characteristic, bool)
            or not isinstance(self.characteristic, int)
            or self.characteristic not in SWEEP_PRIMES
        ):
            raise KrennBlockerPrimeSweepError(
                "probe is outside the reviewed sweep"
            )

    @property
    def key(self) -> str:
        return f"{self.spec.key}_p{self.characteristic:04d}_{ALGORITHM}"


PROBES = tuple(
    PrimeProbe(spec, prime)
    for spec in SPECS
    for prime in SWEEP_PRIMES
)


def _is_prime(value: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _first_primes_above(lower_bound: int, count: int) -> tuple[int, ...]:
    result = []
    candidate = lower_bound + 1
    while len(result) < count:
        if _is_prime(candidate):
            result.append(candidate)
        candidate += 1
    return tuple(result)


def verify_prime_policy() -> dict[str, object]:
    """Replay the nonadaptive coefficient-injective prime policy."""

    expected = _first_primes_above(
        2 * DEFAULT_COEFFICIENT_HEIGHT, len(SWEEP_PRIMES)
    )
    if (
        SWEEP_PRIMES != expected
        or len(set(SWEEP_PRIMES)) != len(SWEEP_PRIMES)
        or any(not _is_prime(prime) for prime in SWEEP_PRIMES)
    ):
        raise KrennBlockerPrimeSweepError(
            "coefficient-injective prime policy changed"
        )
    return {
        "selection_rule": (
            "first 20 primes strictly greater than "
            "2*coefficient_height=254"
        ),
        "ordered_primes": list(SWEEP_PRIMES),
        "count": len(SWEEP_PRIMES),
        "minimum": SWEEP_PRIMES[0],
        "maximum": SWEEP_PRIMES[-1],
        "nonadaptive": True,
    }


def _specialization_digest(seed: int, label: str) -> bytes:
    return hashlib.sha256(
        f"krenn-n8-k5-v2:{seed}:{label}".encode("ascii")
    ).digest()


def _deterministic_distinct_coefficients(
    slots: Sequence[tuple[int, int, int, int]],
    *,
    seed: int,
    height: int,
) -> dict[tuple[int, int, int, int], int]:
    candidates = tuple(
        value
        for value in range(-height, height + 1)
        if value and value % 31
    )
    if len(candidates) < len(slots):
        raise KrennBlockerPrimeSweepError(
            "coefficient height leaves too few distinct values"
        )
    ordered_slots = sorted(
        slots,
        key=lambda slot: _specialization_digest(
            seed, "slot:" + ":".join(map(str, slot))
        ),
    )
    ordered_values = sorted(
        candidates,
        key=lambda value: _specialization_digest(
            seed, f"value:{value}"
        ),
    )[: len(slots)]
    return dict(zip(ordered_slots, ordered_values, strict=True))


def deterministic_degree30_blocker_chart(
    *,
    seed: int = DEFAULT_SPECIALIZATION_SEED,
    color: int = 0,
    coefficient_height: int = DEFAULT_COEFFICIENT_HEIGHT,
) -> TutteBarrierAffineChart:
    """Build the fixed dense 12-by-12 degree-30 blocker chart."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
        or isinstance(color, bool)
        or color not in range(3)
        or isinstance(coefficient_height, bool)
        or not isinstance(coefficient_height, int)
        or not 1 <= coefficient_height <= 1_000
    ):
        raise KrennBlockerPrimeSweepError(
            "invalid deterministic degree-30 chart parameters"
        )
    slots = tuple(
        (edge[0], edge[1], first_color, second_color)
        for edge in DEGREE30_SPEC.blocker_type.blocker_edges
        for first_color in range(3)
        for second_color in range(3)
    )
    coordinates = _deterministic_distinct_coefficients(
        slots, seed=seed, height=coefficient_height
    )
    witness = SparseWitness.from_coordinates(8, 3, coordinates)
    chart = build_tutte_barrier_affine_chart(
        witness, DEGREE30_SPEC.blocker_type, color
    )
    if (
        chart.variable_count != 12
        or chart.equation_count != 12
        or any(len(equation) != 9 for equation in chart.equations)
    ):
        raise KrennBlockerPrimeSweepError(
            "degree-30 affine chart shape changed"
        )
    return chart


def chart_for_spec(spec: SquareBlockerSpec) -> TutteBarrierAffineChart:
    if spec == DEGREE24_SPEC:
        return deterministic_k5_blocker_chart()
    if spec == DEGREE30_SPEC:
        return deterministic_degree30_blocker_chart()
    raise KrennBlockerPrimeSweepError("unknown blocker chart specification")


def _coefficient_vector(
    chart: TutteBarrierAffineChart,
) -> tuple[Fraction, ...]:
    return tuple(
        coefficient
        for equation in chart.equations
        for _monomial, coefficient in equation
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _compact_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def coefficient_audit(
    chart: TutteBarrierAffineChart, prime: int
) -> dict[str, object]:
    """Certify support preservation and coefficient injectivity modulo p."""

    if (
        isinstance(prime, bool)
        or not isinstance(prime, int)
        or prime not in SWEEP_PRIMES
    ):
        raise KrennBlockerPrimeSweepError(
            "coefficient audit prime is outside the sweep"
        )
    coefficients = _coefficient_vector(chart)
    if any(value.denominator != 1 for value in coefficients):
        raise KrennBlockerPrimeSweepError(
            "sweep specialization unexpectedly has denominators"
        )
    integers = tuple(value.numerator for value in coefficients)
    residues = tuple(value % prime for value in integers)
    if (
        len(set(integers)) != len(integers)
        or any(not value for value in integers)
        or any(not value for value in residues)
        or len(set(residues)) != len(residues)
    ):
        raise KrennBlockerPrimeSweepError(
            "selected prime is not coefficient-injective for this chart"
        )
    return {
        "coefficient_count": len(integers),
        "integer_vector_sha256": _compact_sha256(list(integers)),
        "zero_residue_count": residues.count(0),
        "distinct_residue_count": len(set(residues)),
        "residue_vector_sha256": _compact_sha256(list(residues)),
        "coefficient_injective": True,
    }


def _modular_scalar(value: Fraction, prime: int) -> int:
    denominator = value.denominator % prime
    if denominator == 0:
        raise KrennBlockerPrimeSweepError(
            "coefficient denominator vanished modulo the selected prime"
        )
    return value.numerator * pow(denominator, -1, prime) % prime


def _singular_polynomial(
    equation: Sequence[tuple[tuple[int, ...], Fraction]],
    prime: int,
) -> str:
    pieces = []
    for monomial, coefficient in equation:
        scalar = _modular_scalar(Fraction(coefficient), prime)
        if not scalar:
            continue
        factors = [str(scalar), *(f"x{index}" for index in monomial)]
        pieces.append("*".join(factors))
    return "+".join(pieces) if pieces else "0"


def _generic_square_script(
    chart: TutteBarrierAffineChart, prime: int
) -> str:
    names = ",".join(
        f"x{index}" for index in range(chart.variable_count)
    )
    equations = ",\n  ".join(
        _singular_polynomial(equation, prime)
        for equation in chart.equations
    )
    variables = ",".join(
        f"x{index}" for index in range(chart.variable_count)
    )
    digest = chart_sha256(chart)
    return "\n".join(
        (
            "// Deterministic n=8 square-blocker modular sweep.",
            "// The legacy transcript wire schema is reused by the parser.",
            f"// chart_sha256={digest}",
            "option(redSB);",
            f"ring r={prime},({names}),dp;",
            f"ideal I=\n  {equations};",
            f"ideal V={variables};",
            f'print("{PARSE_MARKER}");',
            f'print("transcript_schema={TRANSCRIPT_SCHEMA}");',
            f'print("chart_sha256={digest}");',
            f'print("characteristic={prime}");',
            f'print("algorithm={ALGORITHM}");',
            f'print("variable_count={chart.variable_count}");',
            "int started=timer;",
            f"ideal G={ALGORITHM}(I);",
            "int elapsed=timer-started;",
            "int isunit=(size(G)==1 && G[1]==1);",
            "int krulldim=dim(G);",
            'print("timer_ticks="+string(elapsed));',
            'print("basis_size="+string(size(G)));',
            'print("unit_ideal="+string(isunit));',
            'print("krull_dimension="+string(krulldim));',
            "if (isunit==0 && krulldim==0)",
            "{",
            "  int quotientdim=vdim(G);",
            "  ideal B=kbase(G);",
            '  print("quotient_dimension="+string(quotientdim));',
            '  print("standard_basis_size="+string(size(B)));',
            "  int i;",
            "  int j;",
            "  for (j=1;j<=size(B);j++)",
            "  {",
            '    print("standard_basis["'
            '+string(j-1)+"]="+string(B[j]));',
            "  }",
            "  for (i=1;i<=size(V);i++)",
            "  {",
            "    for (j=1;j<=size(B);j++)",
            "    {",
            '      print("normal_form["'
            '+string(i-1)+","+string(j-1)+"]="'
            "+string(reduce(V[i]*B[j],G)));",
            "    }",
            "  }",
            "}",
            f'print("{COMPLETION_MARKER}");',
            "",
        )
    )


def probe_script(probe: PrimeProbe) -> str:
    chart = chart_for_spec(probe.spec)
    coefficient_audit(chart, probe.characteristic)
    if probe.spec == DEGREE24_SPEC:
        return singular_blocker_quotient_script(
            chart,
            characteristic=probe.characteristic,
            algorithm=ALGORITHM,
        )
    return _generic_square_script(chart, probe.characteristic)


def _real_directory(path: Path, *, create: bool) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise KrennBlockerPrimeSweepError(
                f"{path} must be a real directory"
            )
    elif create:
        path.mkdir(parents=True)
    else:
        raise KrennBlockerPrimeSweepError(f"missing directory {path}")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise KrennBlockerPrimeSweepError(
            f"refusing non-regular output target {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise KrennBlockerPrimeSweepError(
            f"refusing existing temporary target {temporary}"
        )
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    _write_bytes_atomic(path, _canonical_json_bytes(payload))


def _strict_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise KrennBlockerPrimeSweepError(
            f"{path.name} is not strict JSON"
        ) from error
    if type(payload) is not dict:
        raise KrennBlockerPrimeSweepError(
            f"{path.name} must contain one JSON object"
        )
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_transcript(path: Path) -> str:
    if path.stat().st_size > MAXIMUM_TRANSCRIPT_BYTES:
        raise KrennBlockerPrimeSweepError(
            "backend stdout exceeds the frozen parser byte limit"
        )
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise KrennBlockerPrimeSweepError(
            "backend stdout is not strict UTF-8"
        ) from error


def prepare_probe_input(
    probe: PrimeProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, object]:
    scratch_root = Path(scratch_root)
    _real_directory(scratch_root, create=True)
    input_directory = scratch_root / "inputs"
    _real_directory(input_directory, create=True)
    path = input_directory / f"{probe.key}.sing"
    payload = probe_script(probe).encode("utf-8")
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise KrennBlockerPrimeSweepError(
                f"retained probe input changed: {path}"
            )
    else:
        _write_bytes_atomic(path, payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def docker_argv(
    probe: PrimeProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[str, ...]:
    input_path = Path(prepare_probe_input(probe, scratch_root)["path"])
    return (
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--cpus",
        str(CPUS),
        "--memory",
        f"{MEMORY_GIB}g",
        "--entrypoint",
        "timeout",
        "-v",
        f"{input_path.parent}:/work:ro",
        DOCKER_IMAGE_ID,
        "--signal=TERM",
        "--kill-after=15s",
        f"{ENGINE_BUDGET_SECONDS}s",
        "Singular",
        f"/work/{input_path.name}",
    )


def _probe_payload(probe: PrimeProbe) -> dict[str, object]:
    chart = chart_for_spec(probe.spec)
    return {
        "key": probe.key,
        "chart_key": probe.spec.key,
        "chart_sha256": chart_sha256(chart),
        "blocker_type": {
            "barrier_size": probe.spec.blocker_type.barrier_size,
            "odd_component_sizes": list(
                probe.spec.blocker_type.odd_component_sizes
            ),
        },
        "projective_chow_degree": (
            probe.spec.projective_chow_degree
        ),
        "characteristic": probe.characteristic,
        "algorithm": ALGORITHM,
        "specialization_seed": DEFAULT_SPECIALIZATION_SEED,
        "coefficient_height": DEFAULT_COEFFICIENT_HEIGHT,
        "coefficient_audit": coefficient_audit(
            chart, probe.characteristic
        ),
        "engine_budget_seconds": ENGINE_BUDGET_SECONDS,
        "cpus": CPUS,
        "memory_gib": MEMORY_GIB,
    }


def _engine_payload(
    probe: PrimeProbe, scratch_root: Path
) -> dict[str, object]:
    return {
        "image": DOCKER_IMAGE,
        "image_id": DOCKER_IMAGE_ID,
        "network": "none",
        "argv": list(docker_argv(probe, scratch_root)),
    }


@contextmanager
def _exclusive_sweep_lock(scratch_root: Path):
    _real_directory(scratch_root, create=True)
    runs = scratch_root / "runs"
    _real_directory(runs, create=True)
    lock_path = runs / ".sweep.lock"
    token = f"{os.getpid()}:{time.time_ns()}\n".encode("ascii")
    try:
        with lock_path.open("xb") as lock:
            lock.write(token)
            lock.flush()
            os.fsync(lock.fileno())
    except FileExistsError as error:
        raise KrennBlockerPrimeSweepError(
            "another modular sweep owns this scratch root"
        ) from error
    try:
        yield
    finally:
        if (
            lock_path.is_symlink()
            or not lock_path.is_file()
            or lock_path.read_bytes() != token
        ):
            raise KrennBlockerPrimeSweepError(
                "sweep lock changed while jobs were running"
            )
        lock_path.unlink()


def _parse_completed_probe(
    probe: PrimeProbe, stdout_text: str, return_code: int
):
    parse_count = stdout_text.splitlines().count(PARSE_MARKER)
    completion_count = stdout_text.splitlines().count(COMPLETION_MARKER)
    if return_code == 124 and parse_count == 1 and completion_count == 0:
        return (
            {
                "status": "timeout-after-parse",
                "return_code": return_code,
                "parse_marker_observed": True,
                "completion_marker_observed": False,
            },
            None,
        )
    if return_code != 0 or parse_count != 1 or completion_count != 1:
        return (
            {
                "status": "failed-or-invalid-transcript",
                "return_code": return_code,
                "parse_marker_observed": parse_count == 1,
                "completion_marker_observed": completion_count == 1,
            },
            None,
        )
    chart = chart_for_spec(probe.spec)
    try:
        outcome = parse_singular_blocker_output(
            chart,
            stdout_text,
            characteristic=probe.characteristic,
            algorithm=ALGORITHM,
        )
    except ValueError as error:
        return (
            {
                "status": "completed-marker-but-replay-failed",
                "return_code": return_code,
                "parse_marker_observed": True,
                "completion_marker_observed": True,
                "replay_error": str(error),
            },
            None,
        )
    if outcome.unit_ideal:
        status = "completed-unit-ideal"
    elif outcome.representation is not None:
        status = "completed-finite-quotient"
    else:
        status = "completed-positive-dimensional"
    return (
        {
            "status": status,
            "return_code": return_code,
            "parse_marker_observed": True,
            "completion_marker_observed": True,
            "timer_ticks": outcome.timer_ticks,
            "groebner_basis_size": outcome.groebner_basis_size,
            "unit_ideal": outcome.unit_ideal,
            "krull_dimension": outcome.krull_dimension,
            "backend_reported_quotient_dimension": (
                outcome.backend_quotient_dimension
            ),
        },
        outcome,
    )


def _sweep_verification_for_chart(
    probe: PrimeProbe,
    chart: TutteBarrierAffineChart,
    representation: QuotientRepresentation,
) -> dict[str, object]:
    """Wrap the frozen K5 wire verifier in a chart-generic sweep schema."""

    if (
        chart.n != probe.spec.blocker_type.n
        or chart.barrier_vertices
        != probe.spec.blocker_type.barrier_vertices
        or tuple(map(len, chart.components))
        != probe.spec.blocker_type.odd_component_sizes
        or chart.variable_count
        != probe.spec.blocker_type.qutrit_chart_variable_count
        or chart.equation_count
        != len(probe.spec.blocker_type.blocker_edges)
    ):
        raise KrennBlockerPrimeSweepError(
            "verification chart does not match the blocker profile"
        )
    legacy = verify_quotient_representation(chart, representation)
    expected_field = f"F_{probe.characteristic}"
    if (
        legacy["field"] != expected_field
        or legacy["chart_sha256"] != chart_sha256(chart)
        or legacy["representation_dimension"] != representation.dimension
        or legacy["commutator_checks"]
        != chart.variable_count * (chart.variable_count - 1) // 2
        or legacy["equation_matrix_checks"] != chart.equation_count
        or legacy["cyclic_basis_checks"] != representation.dimension
        or legacy["exact_replay"] is not True
    ):
        raise KrennBlockerPrimeSweepError(
            "legacy exact verifier returned inconsistent metadata"
        )
    return {
        "schema": "krenn-n8-square-blocker-prime-verification-v2",
        "chart_key": probe.spec.key,
        "chart_sha256": chart_sha256(chart),
        "blocker_type": {
            "barrier_size": probe.spec.blocker_type.barrier_size,
            "odd_component_sizes": list(
                probe.spec.blocker_type.odd_component_sizes
            ),
        },
        "field": expected_field,
        "variable_count": chart.variable_count,
        "equation_count": chart.equation_count,
        "representation_dimension": representation.dimension,
        "commutator_checks": legacy["commutator_checks"],
        "equation_matrix_checks": legacy["equation_matrix_checks"],
        "cyclic_basis_checks": legacy["cyclic_basis_checks"],
        "exact_replay": True,
        "legacy_wire_parser": {
            "transcript_schema": TRANSCRIPT_SCHEMA,
            "legacy_verification_schema": legacy["schema"],
            "reused_for_wire_parsing_and_exact_matrix_arithmetic_only": True,
        },
        "exact_conclusion": {
            "nonzero_unital_cyclic_representation": True,
            "fixed_specialized_affine_ideal_proper_over_field": True,
            "point_over_algebraic_closure_of_finite_field": True,
        },
        "claim_boundary": {
            key: False for key in VERIFICATION_FALSE_KEYS
        },
    }


def sweep_verification(
    probe: PrimeProbe,
    representation: QuotientRepresentation,
) -> dict[str, object]:
    """Exactly verify one retained deterministic sweep representation."""

    return _sweep_verification_for_chart(
        probe, chart_for_spec(probe.spec), representation
    )


def _run_probe_locked(
    probe: PrimeProbe,
    scratch_root: Path,
    *,
    resume: bool,
) -> dict[str, object]:
    input_record = prepare_probe_input(probe, scratch_root)
    output_directory = scratch_root / "runs"
    _real_directory(output_directory, create=True)
    receipt_path = output_directory / f"{probe.key}.receipt.json"
    if resume and receipt_path.is_file() and not receipt_path.is_symlink():
        return replay_probe(probe, scratch_root)
    if receipt_path.exists():
        raise KrennBlockerPrimeSweepError(
            f"receipt exists for {probe.key}; use --resume"
        )
    stdout_path = output_directory / f"{probe.key}.stdout.txt"
    stderr_path = output_directory / f"{probe.key}.stderr.txt"
    if stdout_path.exists() or stderr_path.exists():
        raise KrennBlockerPrimeSweepError(
            f"orphaned backend streams exist for {probe.key}"
        )
    inspect_engine()
    argv = docker_argv(probe, scratch_root)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with (
        stdout_path.open("xb") as stdout,
        stderr_path.open("xb") as stderr,
    ):
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=ENGINE_BUDGET_SECONDS + HOST_GRACE_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise KrennBlockerPrimeSweepError(
                "host timeout exceeded the engine limit plus grace"
            ) from error
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    elapsed = round(time.monotonic() - started, 6)
    stdout_text = _read_transcript(stdout_path)
    outcome_payload, outcome = _parse_completed_probe(
        probe, stdout_text, completed.returncode
    )
    representation_record = None
    verification = None
    if outcome is not None and outcome.representation is not None:
        representation_path = (
            output_directory / f"{probe.key}.representation.json"
        )
        _write_json_atomic(
            representation_path, outcome.representation.to_dict()
        )
        representation_record = _stream_record(representation_path)
        verification = sweep_verification(
            probe, outcome.representation
        )
    receipt = {
        "schema": RUN_SCHEMA,
        "probe": _probe_payload(probe),
        "input": input_record,
        "engine": _engine_payload(probe, scratch_root),
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed,
        "stdout": _stream_record(stdout_path),
        "stderr": _stream_record(stderr_path),
        "outcome": outcome_payload,
        "representation": representation_record,
        "verification": verification,
        "claim_boundary": claim_boundary(),
    }
    _write_json_atomic(receipt_path, receipt)
    return replay_probe(probe, scratch_root)


def run_probe(
    probe: PrimeProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Run or replay one predeclared probe under the sweep lock."""

    scratch_root = Path(scratch_root)
    with _exclusive_sweep_lock(scratch_root):
        return _run_probe_locked(
            probe, scratch_root, resume=resume
        )


def replay_probe(
    probe: PrimeProbe,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, object]:
    """Replay one retained receipt, transcript, and exact representation."""

    scratch_root = Path(scratch_root)
    output_directory = scratch_root / "runs"
    receipt_path = output_directory / f"{probe.key}.receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise KrennBlockerPrimeSweepError(
            f"missing regular receipt for {probe.key}"
        )
    receipt = _strict_json(receipt_path)
    if set(receipt) != {
        "schema",
        "probe",
        "input",
        "engine",
        "started_at_utc",
        "elapsed_seconds",
        "stdout",
        "stderr",
        "outcome",
        "representation",
        "verification",
        "claim_boundary",
    } or receipt["schema"] != RUN_SCHEMA:
        raise KrennBlockerPrimeSweepError("probe receipt schema changed")
    if receipt["probe"] != _probe_payload(probe):
        raise KrennBlockerPrimeSweepError("probe metadata changed")
    expected_input = prepare_probe_input(probe, scratch_root)
    if receipt["input"] != expected_input:
        raise KrennBlockerPrimeSweepError("probe input receipt changed")
    if receipt["engine"] != _engine_payload(probe, scratch_root):
        raise KrennBlockerPrimeSweepError("probe engine metadata changed")
    try:
        started_at = datetime.fromisoformat(receipt["started_at_utc"])
    except (TypeError, ValueError) as error:
        raise KrennBlockerPrimeSweepError(
            "probe start time is invalid"
        ) from error
    elapsed = receipt["elapsed_seconds"]
    if (
        started_at.tzinfo is None
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not 0 <= elapsed
        <= ENGINE_BUDGET_SECONDS + HOST_GRACE_SECONDS
    ):
        raise KrennBlockerPrimeSweepError("probe timing metadata changed")
    for stream in ("stdout", "stderr"):
        record = receipt[stream]
        if (
            type(record) is not dict
            or set(record) != {"path", "bytes", "sha256"}
            or not isinstance(record["path"], str)
        ):
            raise KrennBlockerPrimeSweepError(
                f"invalid {stream} record"
            )
        path = output_directory / record["path"]
        if (
            path.parent != output_directory
            or path.is_symlink()
            or not path.is_file()
            or record != _stream_record(path)
        ):
            raise KrennBlockerPrimeSweepError(
                f"{stream} hash replay failed"
            )
    stdout_path = output_directory / receipt["stdout"]["path"]
    stdout_text = _read_transcript(stdout_path)
    outcome_record = receipt["outcome"]
    if type(outcome_record) is not dict:
        raise KrennBlockerPrimeSweepError("invalid outcome record")
    return_code = outcome_record.get("return_code")
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        raise KrennBlockerPrimeSweepError("invalid retained return code")
    replayed_outcome, parsed = _parse_completed_probe(
        probe, stdout_text, return_code
    )
    if outcome_record != replayed_outcome:
        raise KrennBlockerPrimeSweepError(
            "backend outcome replay changed"
        )
    representation_record = receipt["representation"]
    if parsed is not None and parsed.representation is not None:
        if (
            type(representation_record) is not dict
            or set(representation_record)
            != {"path", "bytes", "sha256"}
            or not isinstance(representation_record["path"], str)
        ):
            raise KrennBlockerPrimeSweepError(
                "finite probe lacks a representation record"
            )
        representation_path = (
            output_directory / representation_record["path"]
        )
        if (
            representation_path.parent != output_directory
            or representation_path.is_symlink()
            or not representation_path.is_file()
            or representation_record != _stream_record(
                representation_path
            )
        ):
            raise KrennBlockerPrimeSweepError(
                "representation hash replay failed"
            )
        retained = _strict_json(representation_path)
        if retained != parsed.representation.to_dict():
            raise KrennBlockerPrimeSweepError(
                "representation semantic replay failed"
            )
        verification = sweep_verification(
            probe, parsed.representation
        )
        if receipt["verification"] != verification:
            raise KrennBlockerPrimeSweepError(
                "exact matrix verification replay changed"
            )
    elif (
        representation_record is not None
        or receipt["verification"] is not None
    ):
        raise KrennBlockerPrimeSweepError(
            "non-finite probe retained a representation claim"
        )
    expected_claims = claim_boundary()
    if (
        type(receipt["claim_boundary"]) is not dict
        or set(receipt["claim_boundary"]) != set(CLAIM_KEYS)
        or receipt["claim_boundary"] != expected_claims
        or any(value is not False for value in expected_claims.values())
    ):
        raise KrennBlockerPrimeSweepError(
            "probe receipt escalates a sweep claim"
        )
    return receipt


def _summary_run(probe: PrimeProbe, receipt: Mapping) -> dict[str, object]:
    expected_claims = claim_boundary()
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != RUN_SCHEMA
        or receipt.get("probe") != _probe_payload(probe)
        or receipt.get("claim_boundary") != expected_claims
    ):
        raise KrennBlockerPrimeSweepError(
            "summary receipt is not bound to its paired probe"
        )
    outcome = receipt["outcome"]
    verification = receipt["verification"]
    verified_dimension = (
        None
        if verification is None
        else verification["representation_dimension"]
    )
    input_path = Path(receipt["input"]["path"])
    scratch_root = input_path.parent.parent
    receipt_path = (
        scratch_root / "runs" / f"{probe.key}.receipt.json"
    )
    if (
        input_path.parent.name != "inputs"
        or receipt_path.parent.name != "runs"
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or _strict_json(receipt_path) != receipt
    ):
        raise KrennBlockerPrimeSweepError(
            "summary receipt provenance path is invalid"
        )
    return {
        "key": probe.key,
        "chart_key": probe.spec.key,
        "characteristic": probe.characteristic,
        "coefficient_audit": receipt["probe"]["coefficient_audit"],
        "status": outcome["status"],
        "started_at_utc": receipt["started_at_utc"],
        "engine_subprocess_wall_seconds": receipt["elapsed_seconds"],
        "timer_ticks": outcome.get("timer_ticks"),
        "groebner_basis_size": outcome.get("groebner_basis_size"),
        "krull_dimension": outcome.get("krull_dimension"),
        "backend_reported_quotient_dimension": outcome.get(
            "backend_reported_quotient_dimension"
        ),
        "verified_cyclic_module_dimension": verified_dimension,
        "input": {
            "filename": input_path.name,
            "bytes": receipt["input"]["bytes"],
            "sha256": receipt["input"]["sha256"],
        },
        "stdout": {
            "filename": receipt["stdout"]["path"],
            "bytes": receipt["stdout"]["bytes"],
            "sha256": receipt["stdout"]["sha256"],
        },
        "representation": (
            None
            if receipt["representation"] is None
            else {
                "filename": receipt["representation"]["path"],
                "bytes": receipt["representation"]["bytes"],
                "sha256": receipt["representation"]["sha256"],
            }
        ),
        "scratch_receipt": {
            "filename": receipt_path.name,
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256(receipt_path),
        },
        "exact_matrix_replay": (
            None
            if verification is None
            else {
                "commutator_checks": verification[
                    "commutator_checks"
                ],
                "equation_matrix_checks": verification[
                    "equation_matrix_checks"
                ],
                "cyclic_basis_checks": verification[
                    "cyclic_basis_checks"
                ],
                "passed": verification["exact_replay"],
            }
        ),
    }


def _baseline_provenance() -> dict[str, object]:
    files = {
        "manifest": BASELINE_BUNDLE / "manifest.json",
        "reconnaissance": BASELINE_BUNDLE / "reconnaissance.json",
        "p31_receipt": (
            BASELINE_BUNDLE
            / "seed_80320260725_p31_slimgb.receipt.json"
        ),
        "q_receipt": (
            BASELINE_BUNDLE
            / "seed_80320260725_q_slimgb.receipt.json"
        ),
    }
    records = {}
    for key, path in files.items():
        if path.is_symlink() or not path.is_file():
            raise KrennBlockerPrimeSweepError(
                "existing degree-24 baseline provenance is missing"
            )
        records[key] = {
            "repository_path": path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    reconnaissance = _strict_json(files["reconnaissance"])
    p31_receipt = _strict_json(files["p31_receipt"])
    q_receipt = _strict_json(files["q_receipt"])
    p31_dimension = p31_receipt.get("verification", {}).get(
        "representation_dimension"
    )
    q_dimension = q_receipt.get("verification", {}).get(
        "representation_dimension"
    )
    conclusions = reconnaissance.get("exact_conclusions", {})
    degree24_chart = chart_for_spec(DEGREE24_SPEC)
    degree24_digest = chart_sha256(degree24_chart)
    p31_distinct = len(
        {
            value.numerator % 31
            for value in _coefficient_vector(degree24_chart)
        }
    )
    if (
        p31_dimension != 22
        or q_dimension != 24
        or conclusions.get(
            "f31_nonzero_unital_cyclic_quotient_dimension"
        )
        != p31_dimension
        or conclusions.get(
            "q_nonzero_unital_cyclic_quotient_dimension"
        )
        != q_dimension
        or p31_receipt["verification"].get("chart_sha256")
        != degree24_digest
        or q_receipt["verification"].get("chart_sha256")
        != degree24_digest
        or p31_receipt["verification"].get("field") != "F_31"
        or q_receipt["verification"].get("field") != "Q"
        or p31_distinct != 30
    ):
        raise KrennBlockerPrimeSweepError(
            "existing degree-24 baseline semantics changed"
        )
    return {
        "degree24_chart_sha256": degree24_digest,
        "p31_control": {
            "coefficient_residue_distinct_count": p31_distinct,
            "verified_cyclic_module_dimension": p31_dimension,
            "coefficient_injective": False,
        },
        "q_fixed_specialization": {
            "verified_cyclic_module_dimension": q_dimension,
        },
        "files": records,
    }


def build_sweep_summary(
    receipts: Sequence[Mapping],
) -> dict[str, object]:
    """Build the small canonical summary from all forty exact receipts."""

    if len(receipts) != len(PROBES):
        raise KrennBlockerPrimeSweepError(
            "complete sweep summary requires every predeclared probe"
        )
    runs = [
        _summary_run(probe, receipt)
        for probe, receipt in zip(PROBES, receipts, strict=True)
    ]
    if any(
        row["status"] != "completed-finite-quotient"
        or row["verified_cyclic_module_dimension"] is None
        for row in runs
    ):
        raise KrennBlockerPrimeSweepError(
            "complete sweep summary requires forty finite exact replays"
        )
    chart_summaries = []
    for spec in SPECS:
        selected = [
            row for row in runs if row["chart_key"] == spec.key
        ]
        histogram: dict[str, int] = {}
        for row in selected:
            key = str(row["verified_cyclic_module_dimension"])
            histogram[key] = histogram.get(key, 0) + 1
        chart = chart_for_spec(spec)
        chart_summaries.append(
            {
                "chart_key": spec.key,
                "blocker_type": {
                    "barrier_size": spec.blocker_type.barrier_size,
                    "odd_component_sizes": list(
                        spec.blocker_type.odd_component_sizes
                    ),
                    "labeled_orbit_size": (
                        spec.blocker_type.labeled_orbit_size
                    ),
                },
                "projective_chow_degree": (
                    spec.projective_chow_degree
                ),
                "chart_sha256": chart_sha256(chart),
                "variable_count": chart.variable_count,
                "equation_count": chart.equation_count,
                "coefficient_slot_count": spec.coefficient_slot_count,
                "completed_prime_count": len(selected),
                "verified_dimension_histogram": dict(
                    sorted(histogram.items(), key=lambda item: int(item[0]))
                ),
            }
        )
    return {
        "schema": SWEEP_SCHEMA,
        "status": "complete-exact-modular-replay",
        "specialization": {
            "seed": DEFAULT_SPECIALIZATION_SEED,
            "coefficient_height": DEFAULT_COEFFICIENT_HEIGHT,
            "coefficient_slots_only": True,
            "within_each_chart_coefficients_pairwise_distinct_over_Z": (
                True
            ),
            "degree24_and_degree30_use_separate_fixed_specializations": (
                True
            ),
        },
        "baseline_provenance": _baseline_provenance(),
        "prime_policy": verify_prime_policy(),
        "engine": {
            "image": DOCKER_IMAGE,
            "image_id": DOCKER_IMAGE_ID,
            "singular_version": SINGULAR_VERSION,
            "dockerfile": {
                "repository_path": DOCKERFILE_PATH.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "bytes": DOCKERFILE_PATH.stat().st_size,
                "sha256": _sha256(DOCKERFILE_PATH),
            },
            "algorithm": ALGORITHM,
            "network": "none",
            "cpus": CPUS,
            "maximum_workers": MAXIMUM_WORKERS,
            "memory_gib": MEMORY_GIB,
            "per_probe_engine_budget_seconds": (
                ENGINE_BUDGET_SECONDS
            ),
        },
        "charts": chart_summaries,
        "runs": runs,
        "complete": True,
        "claim_boundary": claim_boundary(),
        "interpretation": (
            "Each row is an exact finite-field cyclic-module certificate "
            "for one fixed specialization. Histograms are measurements, "
            "not generic-fiber, flatness, lift, Q/C, EqSystem, or n=8 "
            "existence/nonexistence claims."
        ),
    }


def build_receipt_archive(
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, object]:
    """Collect the forty small receipt records; matrices remain in scratch."""

    scratch_root = Path(scratch_root)
    rows = []
    with _exclusive_sweep_lock(scratch_root):
        for probe in PROBES:
            payload = replay_probe(probe, scratch_root)
            path = (
                scratch_root
                / "runs"
                / f"{probe.key}.receipt.json"
            )
            if (
                payload["outcome"]["status"]
                != "completed-finite-quotient"
                or payload["verification"] is None
                or path.read_bytes() != _canonical_json_bytes(payload)
            ):
                raise KrennBlockerPrimeSweepError(
                    "receipt archive requires a finite exact replay for "
                    f"{probe.key}"
                )
            rows.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "receipt": payload,
                }
            )
    return {
        "schema": RECEIPT_ARCHIVE_SCHEMA,
        "ordered_probe_keys": [probe.key for probe in PROBES],
        "receipt_count": len(rows),
        "receipts": rows,
        "large_transcripts_or_matrices_in_repository": False,
        "scratch_root": str(scratch_root),
        "claim_boundary": claim_boundary(),
    }


def run_sweep(
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Run all forty probes serially and retain a canonical scratch summary."""

    scratch_root = Path(scratch_root)
    verify_prime_policy()
    with _exclusive_sweep_lock(scratch_root):
        receipts = []
        progress_path = scratch_root / "progress.json"
        for probe in PROBES:
            receipt = _run_probe_locked(
                probe, scratch_root, resume=resume
            )
            receipts.append(receipt)
            status = receipt["outcome"]["status"]
            _write_json_atomic(
                progress_path,
                {
                    "schema": PROGRESS_SCHEMA,
                    "completed_receipt_count": len(receipts),
                    "last_probe_key": probe.key,
                    "last_status": status,
                    "all_receipts_present": (
                        len(receipts) == len(PROBES)
                    ),
                    "complete": False,
                    "claim_boundary": claim_boundary(),
                },
            )
            if status != "completed-finite-quotient":
                raise KrennBlockerPrimeSweepError(
                    "sweep stopped after the first non-finite or failed "
                    f"probe: {probe.key} ({status})"
                )
        summary = build_sweep_summary(receipts)
        summary_path = scratch_root / "sweep.json"
        payload = _canonical_json_bytes(summary)
        if summary_path.exists():
            if (
                summary_path.is_symlink()
                or not summary_path.is_file()
                or summary_path.read_bytes() != payload
            ):
                raise KrennBlockerPrimeSweepError(
                    "retained sweep summary changed"
                )
        else:
            _write_bytes_atomic(summary_path, payload)
        _write_json_atomic(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "completed_receipt_count": len(receipts),
                "last_probe_key": PROBES[-1].key,
                "last_status": "completed-finite-quotient",
                "all_receipts_present": True,
                "complete": True,
                "claim_boundary": claim_boundary(),
            },
        )
        return summary


def replay_sweep(
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, object]:
    """Replay every retained matrix certificate and the scratch summary."""

    scratch_root = Path(scratch_root)
    with _exclusive_sweep_lock(scratch_root):
        receipts = [
            replay_probe(probe, scratch_root) for probe in PROBES
        ]
        summary = build_sweep_summary(receipts)
        path = scratch_root / "sweep.json"
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != _canonical_json_bytes(summary)
        ):
            raise KrennBlockerPrimeSweepError(
                "scratch sweep summary replay changed"
            )
        progress = _strict_json(scratch_root / "progress.json")
        if progress != {
            "schema": PROGRESS_SCHEMA,
            "completed_receipt_count": len(PROBES),
            "last_probe_key": PROBES[-1].key,
            "last_status": "completed-finite-quotient",
            "all_receipts_present": True,
            "complete": True,
            "claim_boundary": claim_boundary(),
        }:
            raise KrennBlockerPrimeSweepError(
                "scratch sweep progress replay changed"
            )
        return summary


def _probe_by_key(key: str) -> PrimeProbe:
    for probe in PROBES:
        if probe.key == key:
            return probe
    raise argparse.ArgumentTypeError("unknown prime-sweep probe key")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded n=8 square-blocker prime sweep."
    )
    parser.add_argument(
        "--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--probe", type=_probe_by_key)
    arguments = parser.parse_args(argv)
    if arguments.replay and arguments.probe is not None:
        parser.error("--replay and --probe are mutually exclusive")
    if arguments.probe is not None:
        with _exclusive_sweep_lock(arguments.scratch_root):
            payload = _run_probe_locked(
                arguments.probe,
                arguments.scratch_root,
                resume=arguments.resume,
            )
    elif arguments.replay:
        payload = replay_sweep(arguments.scratch_root)
    else:
        payload = run_sweep(
            arguments.scratch_root, resume=arguments.resume
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
