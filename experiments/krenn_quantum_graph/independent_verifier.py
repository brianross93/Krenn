"""Independent semantic verifier for Krenn systems and witnesses.

This module intentionally imports no other Krenn module.  In particular, its
perfect matchings are found by filtering edge subsets, not by the primary
least-unused-vertex recursion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from math import comb, gcd
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "krenn-quantum-graph-artifact-manifest-v1"
CERTIFICATE_SCHEMA = "krenn-quantum-graph-artifact-certificate-v1"
WITNESS_SCHEMA = "krenn-quantum-graph-sparse-witness-v1"
SYSTEM_ARCHIVE_KEYS = {
    "equation_offsets",
    "monomial_variable_indices",
    "rhs_values",
}
SUPPORTED_ARTIFACT_PARAMETERS = {
    (4, 3),
    (6, 2),
    (4, 4),
    (6, 4),
}

_BASE_INPUT_PATHS = {
    "experiments/krenn_quantum_graph/generate_results.py",
    "experiments/krenn_quantum_graph/independent_verifier.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "experiments/krenn_quantum_graph/witness.py",
}
_WITNESS_INPUT_PATH = (
    "experiments/krenn_quantum_graph/fixtures.py"
)
_DEFORMATION_INPUT_PATH = (
    "experiments/krenn_quantum_graph/deformation.py"
)

_BOUNDARY_STATEMENTS = {
    (4, 4): {
        "status": "known-negative-benchmark-not-reproved",
        "statement": (
            "The benchmark is recorded as having no solution, but this "
            "milestone emits no new independently checkable nonexistence "
            "certificate and makes no computational-proof claim."
        ),
    },
    (6, 4): {
        "status": "bounded-native-support-analysis-begun",
        "statement": (
            "The compact 4,096-equation production system is generated and "
            "structurally verified. Bounded native support analysis has "
            "begun and tests only necessary combinatorial conditions. No "
            "complex weight search was performed, and no witness or "
            "nonexistence certificate is emitted."
        ),
    },
}


class IndependentVerificationError(RuntimeError):
    """Independent reconstruction or artifact replay failed."""


def canonical_claim_boundary(n: int, d: int) -> Mapping:
    """Return the only accepted v1 search/nonexistence boundary."""

    n, d = int(n), int(d)
    try:
        statement = _BOUNDARY_STATEMENTS[(n, d)]
    except KeyError as error:
        raise IndependentVerificationError(
            "this parameter pair has no structural claim boundary"
        ) from error
    return {
        "schema": "krenn-quantum-graph-claim-boundary-v1",
        "status": statement["status"],
        "parameters": {"n": n, "d": d},
        "search_performed": False,
        "no_solution_certificate_emitted": False,
        "statement": statement["statement"],
    }


def expected_bundle_status(
    n: int,
    d: int,
    *,
    witness_present: bool,
    deformation_present: bool,
    boundary: Mapping | None,
) -> str:
    """Derive the headline status from verified bundle contents."""

    if boundary is not None:
        expected_boundary = canonical_claim_boundary(n, d)
        if dict(boundary) != expected_boundary:
            raise IndependentVerificationError(
                "claim boundary is not the canonical no-proof boundary"
            )
        if witness_present or deformation_present:
            raise IndependentVerificationError(
                "a structural boundary bundle cannot contain a witness"
            )
        return str(expected_boundary["status"])
    if deformation_present:
        if not witness_present or (n, d) != (4, 3):
            raise IndependentVerificationError(
                "only the n=4,d=3 witness has a deformation certificate"
            )
        return "exact-witness-and-deformation-certified"
    if witness_present:
        return "exact-witness-certified"
    return "structural-system-generated"


def required_input_paths(
    *,
    witness_present: bool,
    deformation_present: bool,
) -> set[str]:
    result = set(_BASE_INPUT_PATHS)
    if witness_present:
        result.add(_WITNESS_INPUT_PATH)
    if deformation_present:
        result.add(_DEFORMATION_INPUT_PATH)
    return result


def _validate_parameters(n: int, d: int) -> tuple[int, int]:
    n = int(n)
    d = int(d)
    if n < 2 or n % 2 or d < 1:
        raise IndependentVerificationError(
            "invalid Krenn parameters in independent verifier"
        )
    return n, d


@lru_cache(maxsize=None)
def independent_perfect_matchings(
    n: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Filter all ``n/2``-edge subsets for exact vertex coverage."""

    n, _ = _validate_parameters(n, 1)
    all_edges = tuple(combinations(range(n), 2))
    result = []
    for candidate in combinations(all_edges, n // 2):
        covered = tuple(vertex for edge in candidate for vertex in edge)
        if len(set(covered)) == n:
            result.append(candidate)
    return tuple(result)


def _variable_count(n: int, d: int) -> int:
    return comb(n, 2) * d * d


def _variable_index(
    n: int, d: int, i: int, j: int, a: int, b: int
) -> int:
    if not (0 <= i < j < n and 0 <= a < d and 0 <= b < d):
        raise IndependentVerificationError(
            "noncanonical variable in independent replay"
        )
    edge_index = i * (2 * n - i - 1) // 2 + (j - i - 1)
    return (edge_index * d + a) * d + b


def _variable_key(
    n: int, d: int, index: int
) -> tuple[int, int, int, int]:
    if not 0 <= index < _variable_count(n, d):
        raise IndependentVerificationError(
            "variable index is outside the independent system"
        )
    edge_index, color_index = divmod(index, d * d)
    a, b = divmod(color_index, d)
    edge = tuple(combinations(range(n), 2))[edge_index]
    return edge[0], edge[1], a, b


def _expected_matching_count(n: int) -> int:
    result = 1
    for odd in range(n - 1, 0, -2):
        result *= odd
    return result


@dataclass(frozen=True)
class IndependentSystemReport:
    n: int
    d: int
    variable_count: int
    equation_count: int
    degree: int
    matching_count: int
    monomial_count: int
    exact: bool = True


def verify_system_arrays(
    n: int,
    d: int,
    equation_offsets: Sequence[int] | np.ndarray,
    monomial_variable_indices: Sequence[Sequence[int]] | np.ndarray,
    rhs_values: Sequence[int] | np.ndarray,
) -> IndependentSystemReport:
    """Reconstruct and compare every sparse equation independently."""

    n, d = _validate_parameters(n, d)
    offsets = np.asarray(equation_offsets)
    monomials = np.asarray(monomial_variable_indices)
    rhs = np.asarray(rhs_values)
    for name, array in (
        ("equation_offsets", offsets),
        ("monomial_variable_indices", monomials),
        ("rhs_values", rhs),
    ):
        if not np.issubdtype(array.dtype, np.integer):
            raise IndependentVerificationError(
                f"{name} must use an integer dtype"
            )
    equation_total = d**n
    degree = n // 2
    matchings = independent_perfect_matchings(n)
    matching_total = _expected_matching_count(n)
    variable_total = _variable_count(n, d)
    monomial_total = equation_total * matching_total
    if len(matchings) != matching_total:
        raise IndependentVerificationError(
            "independent matching census failed"
        )
    if offsets.shape != (equation_total + 1,):
        raise IndependentVerificationError(
            "independent offset-shape replay failed"
        )
    if monomials.shape != (monomial_total, degree):
        raise IndependentVerificationError(
            "independent monomial-shape replay failed"
        )
    if rhs.shape != (equation_total,):
        raise IndependentVerificationError(
            "independent RHS-shape replay failed"
        )
    expected_offsets = np.arange(
        0,
        monomial_total + 1,
        matching_total,
        dtype=np.uint64,
    )
    if not np.array_equal(offsets.astype(np.uint64), expected_offsets):
        raise IndependentVerificationError(
            "equation offsets failed independent semantic replay"
        )
    if np.any(monomials < 0) or np.any(monomials >= variable_total):
        raise IndependentVerificationError(
            "independent replay found an out-of-range variable"
        )

    equation = 0
    row = 0
    for coloring in product(range(d), repeat=n):
        expected_rhs = int(
            all(color == coloring[0] for color in coloring[1:])
        )
        if int(rhs[equation]) != expected_rhs:
            raise IndependentVerificationError(
                f"RHS corruption at equation {equation}"
            )
        for matching in matchings:
            expected = tuple(
                _variable_index(
                    n, d, i, j, coloring[i], coloring[j]
                )
                for i, j in matching
            )
            if tuple(map(int, monomials[row])) != expected:
                raise IndependentVerificationError(
                    f"monomial corruption at flattened row {row}"
                )
            row += 1
        equation += 1
    return IndependentSystemReport(
        n=n,
        d=d,
        variable_count=variable_total,
        equation_count=equation_total,
        degree=degree,
        matching_count=matching_total,
        monomial_count=monomial_total,
    )


@dataclass(frozen=True)
class IndependentWitnessReport:
    n: int
    d: int
    support_size: int
    exact_over_q: bool
    exact_over_f31: bool

    @property
    def exact(self) -> bool:
        return self.exact_over_q and self.exact_over_f31


def _fraction_mod(value: Fraction, modulus: int) -> int:
    denominator = value.denominator % modulus
    if gcd(denominator, modulus) != 1:
        raise IndependentVerificationError(
            "witness denominator is not invertible modulo 31"
        )
    return value.numerator * pow(denominator, -1, modulus) % modulus


def verify_witness_entries(
    n: int,
    d: int,
    entries: Iterable[tuple[int, Fraction | int]],
) -> IndependentWitnessReport:
    """Evaluate the sparse vector directly, without a primary system."""

    n, d = _validate_parameters(n, d)
    normalized = tuple(
        (int(index), Fraction(value)) for index, value in entries
    )
    indices = tuple(index for index, _value in normalized)
    if (
        indices != tuple(sorted(indices))
        or len(indices) != len(set(indices))
        or any(value == 0 for _index, value in normalized)
        or any(
            index < 0 or index >= _variable_count(n, d)
            for index in indices
        )
    ):
        raise IndependentVerificationError(
            "sparse witness entries are not canonical"
        )
    exact_values = dict(normalized)
    mod_values = {
        index: _fraction_mod(value, 31)
        for index, value in normalized
    }
    matchings = independent_perfect_matchings(n)
    exact_over_q = True
    exact_over_f31 = True
    for coloring in product(range(d), repeat=n):
        target = int(
            all(color == coloring[0] for color in coloring[1:])
        )
        exact_total = Fraction(0)
        mod_total = 0
        for matching in matchings:
            exact_term = Fraction(1)
            mod_term = 1
            for i, j in matching:
                index = _variable_index(
                    n, d, i, j, coloring[i], coloring[j]
                )
                exact_term *= exact_values.get(index, Fraction(0))
                mod_term = mod_term * mod_values.get(index, 0) % 31
            exact_total += exact_term
            mod_total = (mod_total + mod_term) % 31
        exact_over_q &= exact_total == target
        exact_over_f31 &= mod_total == target
    report = IndependentWitnessReport(
        n=n,
        d=d,
        support_size=len(normalized),
        exact_over_q=exact_over_q,
        exact_over_f31=exact_over_f31,
    )
    if not report.exact:
        raise IndependentVerificationError(
            "witness failed independent exact evaluation"
        )
    return report


def _rank_over_q(matrix: Sequence[Sequence[Fraction | int]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    column_count = len(rows[0])
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = (
            rows[selected],
            rows[pivot_row],
        )
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [
            value / pivot for value in rows[pivot_row]
        ]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    rows[row], rows[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def _pivot_columns_over_q(
    matrix: Sequence[Sequence[Fraction | int]],
) -> tuple[int, ...]:
    """Independent deterministic pivot extraction."""

    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return ()
    pivot_row = 0
    pivots: list[int] = []
    for column in range(len(rows[0])):
        selected = next(
            (
                row
                for row in range(pivot_row, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = (
            rows[selected],
            rows[pivot_row],
        )
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [
            value / pivot for value in rows[pivot_row]
        ]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    rows[row], rows[pivot_row], strict=True
                )
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(pivots)


def _inverse_over_q_independent(
    matrix: Sequence[Sequence[Fraction | int]],
) -> list[list[Fraction]]:
    rows = [list(map(Fraction, row)) for row in matrix]
    size = len(rows)
    augmented = [
        row
        + [
            Fraction(int(row_index == column))
            for column in range(size)
        ]
        for row_index, row in enumerate(rows)
    ]
    for column in range(size):
        selected = next(
            row
            for row in range(column, size)
            if augmented[row][column]
        )
        augmented[column], augmented[selected] = (
            augmented[selected],
            augmented[column],
        )
        pivot = augmented[column][column]
        augmented[column] = [
            value / pivot for value in augmented[column]
        ]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [row[size:] for row in augmented]


def _multiply_over_q_independent(
    left: Sequence[Sequence[Fraction | int]],
    right: Sequence[Sequence[Fraction | int]],
) -> list[list[Fraction]]:
    left = [list(map(Fraction, row)) for row in left]
    right = [list(map(Fraction, row)) for row in right]
    return [
        [
            sum(
                left[row][middle] * right[middle][column]
                for middle in range(len(right))
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def _rank_mod31(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.int64).copy() % 31
    pivot_row = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            work[[pivot_row, selected]] = work[[selected, pivot_row]]
        work[pivot_row] = (
            work[pivot_row]
            * pow(int(work[pivot_row, column]), -1, 31)
        ) % 31
        factors = work[:, column].copy()
        factors[pivot_row] = 0
        work = (
            work - factors[:, np.newaxis] * work[pivot_row]
        ) % 31
        pivot_row += 1
        if pivot_row == work.shape[0]:
            break
    return pivot_row


def _independent_deformation_payload(
    equation_offsets: np.ndarray,
    monomial_variable_indices: np.ndarray,
    witness_entries: Sequence[tuple[int, Fraction]],
) -> Mapping:
    """Recompute the native exact-Q quotient without primary code."""

    values = dict(witness_entries)
    jacobian = [
        [Fraction(0) for _ in range(54)] for _ in range(81)
    ]
    for equation in range(81):
        for term in range(
            int(equation_offsets[equation]),
            int(equation_offsets[equation + 1]),
        ):
            monomial = tuple(
                map(int, monomial_variable_indices[term])
            )
            for position, variable in enumerate(monomial):
                derivative = Fraction(1)
                for factor_position, factor in enumerate(monomial):
                    if factor_position != position:
                        derivative *= values.get(
                            factor, Fraction(0)
                        )
                jacobian[equation][variable] += derivative

    gauge_edges = {
        (0, 1): (0, 1),
        (2, 3): (0, -1),
        (0, 2): (1, 1),
        (1, 3): (1, -1),
        (0, 3): (2, 1),
        (1, 2): (2, -1),
    }
    global_gauge_action = True
    for monomial in monomial_variable_indices:
        balances = [0, 0, 0]
        for index in map(int, monomial):
            i, j, _a, _b = _variable_key(4, 3, index)
            direction, sign = gauge_edges[(i, j)]
            balances[direction] += sign
        global_gauge_action &= balances == [0, 0, 0]

    gauge = [
        [Fraction(0), Fraction(0), Fraction(0)]
        for _ in range(54)
    ]
    for index, value in witness_entries:
        i, j, _a, _b = _variable_key(4, 3, index)
        direction, sign = gauge_edges[(i, j)]
        gauge[index][direction] = sign * value
    rank_q = _rank_over_q(jacobian)
    gauge_rank_q = _rank_over_q(gauge)
    q_complex = all(
        sum(
            jacobian[row][middle] * gauge[middle][column]
            for middle in range(54)
        )
        == 0
        for row in range(81)
        for column in range(3)
    )
    nullity_q = 54 - rank_q
    kernel_equals_gauge_q = (
        q_complex and gauge_rank_q == nullity_q == 3
    )

    transposed_gauge = [
        [gauge[row][column] for row in range(54)]
        for column in range(3)
    ]
    pivot_rows = _pivot_columns_over_q(transposed_gauge)
    pivot_set = set(pivot_rows)
    free_rows = tuple(row for row in range(54) if row not in pivot_set)
    pivot_block = [
        [gauge[row][column] for column in range(3)]
        for row in pivot_rows
    ]
    pivot_inverse = _inverse_over_q_independent(pivot_block)
    quotient_map = [
        [Fraction(0) for _ in range(54)] for _ in range(51)
    ]
    for quotient_row, ambient_row in enumerate(free_rows):
        quotient_map[quotient_row][ambient_row] = Fraction(1)
        for pivot_column, pivot_row in enumerate(pivot_rows):
            quotient_map[quotient_row][pivot_row] = -sum(
                gauge[ambient_row][gauge_column]
                * pivot_inverse[gauge_column][pivot_column]
                for gauge_column in range(3)
            )
    section = [
        [Fraction(0) for _ in range(51)] for _ in range(54)
    ]
    for quotient_column, ambient_row in enumerate(free_rows):
        section[ambient_row][quotient_column] = Fraction(1)
    quotient_kills_gauge = _multiply_over_q_independent(
        quotient_map, gauge
    )
    quotient_section = _multiply_over_q_independent(
        quotient_map, section
    )
    induced_jacobian = _multiply_over_q_independent(
        jacobian, section
    )
    factorization = _multiply_over_q_independent(
        induced_jacobian, quotient_map
    )
    quotient_rank_q = _rank_over_q(induced_jacobian)
    quotient_nullity_q = 51 - quotient_rank_q
    quotient_kills_gauge_exact = all(
        value == 0 for row in quotient_kills_gauge for value in row
    )
    quotient_section_exact = all(
        quotient_section[row][column] == int(row == column)
        for row in range(51)
        for column in range(51)
    )
    quotient_kernel_equals_gauge = (
        quotient_kills_gauge_exact
        and quotient_section_exact
        and gauge_rank_q == 3
        and len(free_rows) == 51
    )

    jacobian_f31 = np.asarray(
        [
            [_fraction_mod(value, 31) for value in row]
            for row in jacobian
        ],
        dtype=np.int64,
    )
    gauge_f31 = np.asarray(
        [
            [_fraction_mod(value, 31) for value in row]
            for row in gauge
        ],
        dtype=np.int64,
    )
    rank_f31 = _rank_mod31(jacobian_f31)
    gauge_rank_f31 = _rank_mod31(gauge_f31)
    f31_complex = not np.any(jacobian_f31 @ gauge_f31 % 31)
    nullity_f31 = 54 - rank_f31
    kernel_equals_gauge_f31 = (
        f31_complex and gauge_rank_f31 == nullity_f31 == 3
    )
    exact_checks = {
        "witness_exact_over_Q": True,
        "gauge_action_preserves_every_system_monomial": (
            global_gauge_action
        ),
        "jacobian_rank_51_over_Q": rank_q == 51,
        "jacobian_nullity_3_over_Q": nullity_q == 3,
        "gauge_rank_3_over_Q": gauge_rank_q == 3,
        "jacobian_kills_gauge_over_Q": q_complex,
        "raw_kernel_equals_gauge_over_Q": kernel_equals_gauge_q,
        "native_quotient_map_kills_gauge_over_Q": (
            quotient_kills_gauge_exact
        ),
        "native_quotient_section_identity_over_Q": (
            quotient_section_exact
        ),
        "native_quotient_kernel_equals_gauge_over_Q": (
            quotient_kernel_equals_gauge
        ),
        "native_quotient_jacobian_factorization_over_Q": (
            factorization == jacobian
        ),
        "native_quotient_jacobian_injective_over_Q": (
            quotient_rank_q == 51 and quotient_nullity_q == 0
        ),
    }
    f31_checks = {
        "witness_exact_over_F31": True,
        "jacobian_rank_51_over_F31": rank_f31 == 51,
        "jacobian_nullity_3_over_F31": nullity_f31 == 3,
        "gauge_rank_3_over_F31": gauge_rank_f31 == 3,
        "jacobian_kills_gauge_over_F31": f31_complex,
        "raw_kernel_equals_gauge_over_F31": (
            kernel_equals_gauge_f31
        ),
    }
    return {
        "schema": "krenn-quantum-graph-deformation-certificate-v2",
        "parameters": {"n": 4, "d": 3, "prime": 31},
        "jacobian": {
            "shape": [81, 54],
            "rank_over_Q": rank_q,
            "nullity_over_Q": nullity_q,
        },
        "reciprocal_edge_rescaling_gauge": {
            "shape": [54, 3],
            "rank_over_Q": gauge_rank_q,
            "directions": [
                {
                    "positive_edge": [0, 1],
                    "negative_edge": [2, 3],
                },
                {
                    "positive_edge": [0, 2],
                    "negative_edge": [1, 3],
                },
                {
                    "positive_edge": [0, 3],
                    "negative_edge": [1, 2],
                },
            ],
            "action": (
                "all color slots on the positive edge scale by lambda; "
                "all slots on its complementary negative edge scale by "
                "lambda^-1"
            ),
        },
        "native_gauge_quotient_over_Q": {
            "construction": "deterministic exact pivot-row complement",
            "ambient_dimension": 54,
            "gauge_image_dimension": gauge_rank_q,
            "quotient_dimension": len(free_rows),
            "pivot_rows": list(pivot_rows),
            "quotient_map_shape": [len(quotient_map), 54],
            "section_shape": [len(section), len(section[0])],
            "induced_jacobian_shape": [81, 51],
            "induced_jacobian_rank": quotient_rank_q,
            "induced_jacobian_nullity": quotient_nullity_q,
        },
        "optional_arithmetic_regression_F31": {
            "proof_role": (
                "supplemental arithmetic regression; not used in the "
                "characteristic-zero quotient proof"
            ),
            "jacobian_rank": rank_f31,
            "jacobian_nullity": nullity_f31,
            "gauge_rank": gauge_rank_f31,
            "checks": f31_checks,
            "passed": all(f31_checks.values()),
        },
        "exact_checks": exact_checks,
        "claim_boundary": {
            "characteristic_zero": (
                "The native exact-Q split quotient proves that the three "
                "gauge directions are the full Jacobian kernel and that the "
                "induced 51-dimensional Jacobian is injective; this persists "
                "after scalar extension to C."
            ),
            "finite_field": (
                "The standalone F_31 calculation is an optional arithmetic "
                "regression and is not used as a proof over C."
            ),
        },
    }


def _load_witness_payload(
    path: Path, n: int, d: int
) -> tuple[tuple[int, Fraction], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(
            "could not decode witness artifact"
        ) from error
    if (
        payload.get("schema") != WITNESS_SCHEMA
        or int(payload.get("n", -1)) != n
        or int(payload.get("d", -1)) != d
        or payload.get("omitted_coordinates") != "zero"
    ):
        raise IndependentVerificationError(
            "witness artifact identity changed"
        )
    entries = []
    previous = -1
    for row in payload.get("entries", ()):
        try:
            index = int(row["variable_index"])
            coordinate = tuple(map(int, row["coordinate"]))
            numerator = int(row["numerator"])
            denominator = int(row["denominator"])
        except (KeyError, TypeError, ValueError) as error:
            raise IndependentVerificationError(
                "malformed witness row"
            ) from error
        if (
            len(coordinate) != 4
            or not 0 <= coordinate[0] < coordinate[1] < n
            or not 0 <= coordinate[2] < d
            or not 0 <= coordinate[3] < d
            or _variable_index(n, d, *coordinate) != index
            or index <= previous
            or denominator <= 0
            or gcd(abs(numerator), denominator) != 1
            or numerator == 0
        ):
            raise IndependentVerificationError(
                "witness row failed canonical replay"
            )
        previous = index
        entries.append((index, Fraction(numerator, denominator)))
    if int(payload.get("support_size", -1)) != len(entries):
        raise IndependentVerificationError(
            "witness support census changed"
        )
    return tuple(entries)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_record(
    directory: Path, record: Mapping
) -> Path:
    if set(record) != {"path", "bytes", "sha256"}:
        raise IndependentVerificationError(
            "artifact manifest record schema changed"
        )
    label = str(record.get("path", ""))
    if (
        not label
        or Path(label).name != label
        or "/" in label
        or "\\" in label
    ):
        raise IndependentVerificationError(
            "artifact manifest path escaped its bundle"
        )
    path = directory / label
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or _sha256(path) != record.get("sha256")
    ):
        raise IndependentVerificationError(
            f"artifact manifest replay failed for {label}"
        )
    return path


def _canonical_source_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_source_record(path: Path | str) -> Mapping:
    """Hash UTF-8 Python source with normalized LF line endings."""

    path = Path(path).resolve()
    try:
        label = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError as error:
        raise IndependentVerificationError(
            "source manifest path is outside the repository"
        ) from error
    payload = _canonical_source_bytes(path)
    return {
        "path": label,
        "canonical_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hash_mode": "canonical-lf-text-v1",
    }


def _verify_root_record(record: Mapping) -> Path:
    if set(record) != {
        "path",
        "canonical_bytes",
        "sha256",
        "hash_mode",
    }:
        raise IndependentVerificationError(
            "producer/input source-record schema changed"
        )
    label = str(record.get("path", ""))
    pure = Path(label)
    if (
        not label
        or pure.is_absolute()
        or "\\" in label
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise IndependentVerificationError(
            "producer/input manifest path is unsafe"
        )
    candidate = ROOT / pure
    current = ROOT
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise IndependentVerificationError(
                "producer/input manifest path traverses a symlink"
            )
    path = candidate.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise IndependentVerificationError(
            "producer/input manifest path escaped the repository"
        ) from error
    if not path.is_file() or dict(record) != canonical_source_record(path):
        raise IndependentVerificationError(
            f"producer/input replay failed for {label}"
        )
    return path


@dataclass(frozen=True)
class IndependentArtifactReport:
    directory: Path
    system: IndependentSystemReport
    witness: IndependentWitnessReport | None
    status: str
    manifest_replayed: bool = True
    boundary_replayed: bool = True
    deformation_replayed: bool = True

    @property
    def exact(self) -> bool:
        return (
            self.system.exact
            and (self.witness is None or self.witness.exact)
            and self.manifest_replayed
            and self.boundary_replayed
            and self.deformation_replayed
        )


def verify_artifact_bundle(
    directory: Path | str,
) -> IndependentArtifactReport:
    """Independently replay hashes, structure, equations, and any witness."""

    requested_directory = Path(directory)
    if (
        requested_directory.is_symlink()
        or not requested_directory.is_dir()
    ):
        raise IndependentVerificationError(
            "artifact bundle directory is absent or linked"
        )
    directory = requested_directory.resolve()
    try:
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        certificate = json.loads(
            (directory / "certificate.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(
            "could not decode artifact manifest/certificate"
        ) from error
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise IndependentVerificationError("artifact manifest schema changed")
    if certificate.get("schema") != CERTIFICATE_SCHEMA:
        raise IndependentVerificationError(
            "artifact certificate schema changed"
        )

    artifacts = certificate.get("artifacts", {})
    if set(artifacts) != {
        "system",
        "witness",
        "deformation",
        "claim_boundary",
        "certificate",
        "manifest",
    } or (
        artifacts.get("system") != "system.npz"
        or artifacts.get("certificate") != "certificate.json"
        or artifacts.get("manifest") != "manifest.json"
    ):
        raise IndependentVerificationError(
            "certificate artifact inventory schema changed"
        )
    witness_present = artifacts.get("witness") is not None
    deformation_present = artifacts.get("deformation") is not None

    records = tuple(manifest.get("artifacts", ()))
    record_labels = tuple(
        str(record.get("path", "")) for record in records
    )
    if (
        record_labels != tuple(sorted(record_labels))
        or len(record_labels) != len(set(record_labels))
    ):
        raise IndependentVerificationError(
            "artifact manifest ledger is not unique and sorted"
        )
    replayed = {
        str(record.get("path")): _verify_local_record(directory, record)
        for record in records
    }
    expected_names = {
        str(name)
        for key, name in artifacts.items()
        if key != "manifest" and name is not None
    }
    if set(replayed) != expected_names:
        raise IndependentVerificationError(
            "certificate and manifest artifact inventories differ"
        )
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != expected_names | {"manifest.json"}:
        raise IndependentVerificationError(
            "artifact bundle contains an unexpected file inventory"
        )

    expected_producer = canonical_source_record(
        ROOT / "experiments/krenn_quantum_graph/artifacts.py"
    )
    if manifest.get("producer") != expected_producer:
        raise IndependentVerificationError(
            "artifact producer identity was substituted"
        )
    _verify_root_record(expected_producer)
    input_records = tuple(manifest.get("inputs", ()))
    input_labels = tuple(
        str(record.get("path", "")) for record in input_records
    )
    if (
        input_labels != tuple(sorted(input_labels))
        or len(input_labels) != len(set(input_labels))
    ):
        raise IndependentVerificationError(
            "source input ledger is not unique and sorted"
        )
    required_inputs = required_input_paths(
        witness_present=witness_present,
        deformation_present=deformation_present,
    )
    if not required_inputs.issubset(set(input_labels)):
        raise IndependentVerificationError(
            "source input ledger omitted a mandatory dependency"
        )
    for record in input_records:
        _verify_root_record(record)

    parameters = certificate.get("parameters", {})
    n = int(parameters.get("n", -1))
    d = int(parameters.get("d", -1))
    if (n, d) not in SUPPORTED_ARTIFACT_PARAMETERS:
        raise IndependentVerificationError(
            "artifact parameters exceed the bounded v1 milestone"
        )
    archive_path = replayed.get("system.npz")
    if archive_path is None:
        raise IndependentVerificationError("system archive is absent")
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != SYSTEM_ARCHIVE_KEYS:
                raise IndependentVerificationError(
                    "system archive key set changed"
                )
            offsets = np.asarray(archive["equation_offsets"])
            monomials = np.asarray(
                archive["monomial_variable_indices"]
            )
            rhs = np.asarray(archive["rhs_values"])
    except (OSError, ValueError) as error:
        raise IndependentVerificationError(
            "could not load system archive"
        ) from error
    if (
        offsets.dtype != np.dtype(np.uint64)
        or monomials.dtype != np.dtype(np.uint16)
        or rhs.dtype != np.dtype(np.uint8)
    ):
        raise IndependentVerificationError(
            "compact system archive dtype contract changed"
        )
    system_report = verify_system_arrays(
        n, d, offsets, monomials, rhs
    )
    counts = certificate.get("counts", {})
    expected_counts = {
        "variables": system_report.variable_count,
        "equations": system_report.equation_count,
        "degree": system_report.degree,
        "matchings_per_equation": system_report.matching_count,
        "monomials": system_report.monomial_count,
        "constant_rhs_equations": d,
        "witness_support": 0,
    }

    witness_report = None
    witness_entries: tuple[tuple[int, Fraction], ...] = ()
    witness_name = artifacts.get("witness")
    if witness_name is not None:
        witness_entries = _load_witness_payload(
            replayed[str(witness_name)], n, d
        )
        witness_report = verify_witness_entries(
            n, d, witness_entries
        )
        expected_counts["witness_support"] = (
            witness_report.support_size
        )
        if not certificate.get("exact_checks", {}).get(
            "witness_exact_over_Q", False
        ) or not certificate.get("exact_checks", {}).get(
            "witness_exact_over_F31", False
        ):
            raise IndependentVerificationError(
                "certificate omitted an exact witness check"
            )

    if any(
        int(counts.get(key, -1)) != value
        for key, value in expected_counts.items()
    ):
        raise IndependentVerificationError(
            "certificate structural counts failed independent replay"
        )

    boundary = None
    boundary_name = artifacts.get("claim_boundary")
    if boundary_name is not None:
        try:
            boundary = json.loads(
                replayed[str(boundary_name)].read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            raise IndependentVerificationError(
                "could not decode claim-boundary artifact"
            ) from error
        if boundary != canonical_claim_boundary(n, d):
            raise IndependentVerificationError(
                "claim-boundary semantics changed"
            )
    expected_search_boundary = (
        boundary
        if boundary is not None
        else {
            "search_performed": False,
            "no_solution_certificate_emitted": False,
        }
    )
    if (
        certificate.get("claim_boundary", {}).get("search")
        != expected_search_boundary
    ):
        raise IndependentVerificationError(
            "certificate and claim-boundary artifact disagree"
        )

    deformation_replayed = True
    deformation_name = artifacts.get("deformation")
    if deformation_name is not None:
        if (n, d) != (4, 3) or witness_report is None:
            raise IndependentVerificationError(
                "deformation artifact has the wrong parameter scope"
            )
        try:
            deformation = json.loads(
                replayed[str(deformation_name)].read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            raise IndependentVerificationError(
                "could not decode deformation artifact"
            ) from error
        expected_deformation = _independent_deformation_payload(
            offsets, monomials, witness_entries
        )
        deformation_replayed = deformation == expected_deformation
        if not deformation_replayed or not all(
            bool(value)
            for value in deformation.get("exact_checks", {}).values()
        ):
            raise IndependentVerificationError(
                "deformation claims failed independent semantic replay"
            )

    status = expected_bundle_status(
        n,
        d,
        witness_present=witness_present,
        deformation_present=deformation_present,
        boundary=boundary,
    )
    if certificate.get("status") != status:
        raise IndependentVerificationError(
            "headline certificate status is not content-derived"
        )
    checks = certificate.get("exact_checks", {})
    mandatory_checks = {
        "primary_sparse_system_valid",
        "independent_sparse_system_replay",
    }
    if witness_present:
        mandatory_checks.update(
            {
                "witness_exact_over_Q",
                "witness_exact_over_F31",
                "independent_witness_replay",
            }
        )
    if deformation_present:
        mandatory_checks.add("deformation_certificate_exact")
    if boundary is not None:
        mandatory_checks.add("claim_boundary_explicit")
    if not mandatory_checks.issubset(checks) or any(
        checks[key] is not True for key in mandatory_checks
    ):
        raise IndependentVerificationError(
            "certificate exact-check ledger is incomplete"
        )
    return IndependentArtifactReport(
        directory=directory,
        system=system_report,
        witness=witness_report,
        status=status,
        boundary_replayed=True,
        deformation_replayed=deformation_replayed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify one Krenn artifact bundle."
    )
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    report = verify_artifact_bundle(arguments.directory)
    print(
        json.dumps(
            {
                "exact": report.exact,
                "status": report.status,
                "n": report.system.n,
                "d": report.system.d,
                "monomials": report.system.monomial_count,
                "witness_present": report.witness is not None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
