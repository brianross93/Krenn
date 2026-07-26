"""Exact positive-circuit gate for one bounded ``n=8,d=3`` H5 layer.

This module enumerates the 20,736 minimum-cost depth-two repair branches of
one fixed H5 first-shell parent.  It then classifies the 860 support-24
branches with positive rational quotient dimension under the *parent*
stabilizer, a group of order two.  There are 444 resulting classes.

The exclusion is an exact theorem about this bounded family.  For every
class, a repository-local inventory supplies mixed support-singleton
equations and primitive positive integer coefficients.  If ``T`` is the
six-row repair-tie matrix and ``s_e`` is the restricted exponent row of the
unique support monomial in equation ``e``, exact replay proves

    sum_e alpha_e s_e  in rowspan_Q(T),       alpha_e > 0.

Every active monochromatic target monomial has its restricted exponent row
in the same tie-row span.  Hence all target terms have order zero on
``ker(T)``, while the positive relation forces at least one unique mixed
support term to have order at most zero.  Such a term is noncancellable on
the declared leading-support chart.  This argument does not depend on a
canonical quotient lift or on sampled order counts.

The result is deliberately narrow.  It is not a global ``n=8`` theorem and
does not classify another parent, a deeper closure, a different support, or
an initial form in which an outside-support term enters at leading order.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from math import gcd
import os
from pathlib import Path
from typing import Mapping, Sequence

import experiments.krenn_quantum_graph.n8_toric_first_shell as first_shell


N = 8
D = 3
SOURCE_VARIABLES = 252
GAUGE_DIMENSION = 21
COLORINGS = D**N

SUPPORT24_CIRCUIT_SCHEMA = "krenn-n8-d3-toric-support24-circuits-v1"
SUPPORT24_CIRCUIT_MANIFEST_SCHEMA = (
    "krenn-n8-d3-toric-support24-circuits-manifest-v1"
)
CIRCUIT_INVENTORY_SCHEMA = (
    "krenn-n8-d3-support24-positive-circuit-inventory-v1"
)
CIRCUIT_INVENTORY_PATH = Path(__file__).with_name(
    "n8_support24_positive_circuits.json"
)
DEFAULT_RESULTS_DIRECTORY = (
    Path("results")
    / "krenn_quantum_graph"
    / "n8_d3_toric_support24_circuits"
)
CERTIFICATE_FILE = "certificate.json"
README_FILE = "README.md"
MANIFEST_FILE = "manifest.json"

SOURCE_PATHS = (
    "experiments/krenn_quantum_graph/independent_verifier.py",
    "experiments/krenn_quantum_graph/n8_seed_orbits.py",
    "experiments/krenn_quantum_graph/n8_toric_first_shell.py",
    "experiments/krenn_quantum_graph/n8_toric_support24_circuits.py",
    "experiments/krenn_quantum_graph/n8_support24_positive_circuits.json",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "tests/test_krenn_n8_toric_first_shell.py",
    "tests/test_krenn_n8_toric_support24_circuits.py",
)

PARENT_EXPECTED = {
    "label": "H5",
    "orbit_index": 3,
    "repair_matching_indices": (0, 11),
    "support": (
        0,
        13,
        26,
        85,
        98,
        117,
        151,
        161,
        184,
        194,
        198,
        205,
        215,
        238,
        243,
        250,
    ),
    "spill_equations": (7, 63, 4_049, 5_791),
    "repair_options_per_spill": 12,
    "raw_branch_count": 20_736,
}

EXPECTED_BRANCH_HISTOGRAM = {
    (22, 0): 284,
    (22, 1): 4,
    (24, 0): 19_588,
    (24, 1): 852,
    (24, 2): 8,
}

EXPECTED_CLASS_DATA = {
    "raw_positive_support24": 860,
    "classes": 444,
    "rank_one_raw": 852,
    "rank_one_classes": 440,
    "rank_two_raw": 8,
    "rank_two_classes": 4,
    "rank_one_orbit_size_histogram": {1: 28, 2: 412},
    "rank_two_orbit_size_histogram": {2: 4},
    "rank_one_distinct_supports": 850,
    "rank_two_distinct_supports": 8,
    "parent_stabilizer_order": 2,
}

EXPECTED_CIRCUIT_LENGTH_HISTOGRAMS = {
    1: {2: 27, 3: 48, 4: 79, 5: 81, 6: 111, 7: 94},
    2: {4: 1, 6: 2, 7: 1},
}

EXPECTED_CIRCUIT_COEFFICIENT_HISTOGRAMS = {
    1: {1: 1_576, 2: 488, 3: 137, 4: 38, 5: 4},
    2: {1: 19, 2: 4},
}

EXPECTED_RANK_ONE_TARGET_PATTERN_HISTOGRAM = {
    (1, 1, 1): 244,
    (2, 1, 1): 72,
    (1, 1, 2): 48,
    (1, 2, 1): 32,
    (3, 1, 1): 26,
    (1, 2, 2): 10,
    (4, 1, 1): 8,
}

NONTRIVIAL_PARENT_SYMMETRY = {
    "vertices": (0, 1, 3, 2, 5, 4, 7, 6),
    "colors": (0, 2, 1),
}


class KrennN8ToricSupport24CircuitError(ValueError):
    """A bounded support-24 circuit certificate failed exact replay."""


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _json_sha256(payload: object) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(data).hexdigest()


def _strict_json(path: Path) -> object:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise KrennN8ToricSupport24CircuitError(
                    f"duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str):
        raise KrennN8ToricSupport24CircuitError(
            f"nonfinite JSON constant {value!r}"
        )

    try:
        return json.loads(
            path.read_bytes().decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennN8ToricSupport24CircuitError(
            f"could not decode {path.name}"
        ) from error


def _rref(
    matrix: Sequence[Sequence[int | Fraction]],
) -> tuple[list[list[Fraction]], tuple[int, ...]]:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return rows, ()
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennN8ToricSupport24CircuitError(
            "an exact matrix is ragged"
        )
    pivot_columns = []
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(pivot_row, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        divisor = rows[pivot_row][column]
        rows[pivot_row] = [value / divisor for value in rows[pivot_row]]
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
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return rows, tuple(pivot_columns)


def _row_span_coefficients(
    rows: Sequence[Sequence[int | Fraction]],
    target: Sequence[int | Fraction],
) -> tuple[Fraction, ...]:
    rows = tuple(tuple(map(Fraction, row)) for row in rows)
    target = tuple(map(Fraction, target))
    if not rows or any(len(row) != len(target) for row in rows):
        raise KrennN8ToricSupport24CircuitError(
            "a row-span system has incompatible dimensions"
        )
    system = [
        [rows[column][coordinate] for column in range(len(rows))]
        + [target[coordinate]]
        for coordinate in range(len(target))
    ]
    reduced, pivots = _rref(system)
    variable_count = len(rows)
    for row in reduced:
        if not any(row[:variable_count]) and row[variable_count]:
            raise KrennN8ToricSupport24CircuitError(
                "an exponent row is not in the repair-tie row span"
            )
    solution = [Fraction(0)] * variable_count
    for row, pivot in enumerate(pivots):
        if pivot < variable_count:
            solution[pivot] = reduced[row][variable_count]
    reconstructed = tuple(
        sum(
            solution[row] * rows[row][coordinate]
            for row in range(variable_count)
        )
        for coordinate in range(len(target))
    )
    if reconstructed != target:
        raise KrennN8ToricSupport24CircuitError(
            "a displayed tie-row combination does not reconstruct"
        )
    return tuple(solution)


def _fraction_records(values: Sequence[Fraction]) -> list[list[int]]:
    return [[value.numerator, value.denominator] for value in values]


def _incidence_row(
    coordinates: Sequence[int],
    monomial: Sequence[int],
) -> tuple[int, ...]:
    counts = Counter(map(int, monomial))
    return tuple(counts[int(index)] for index in coordinates)


def _restricted_difference_row(
    coordinates: Sequence[int],
    positive: Sequence[int],
    negative: Sequence[int],
) -> tuple[int, ...]:
    coefficients = Counter(map(int, positive))
    coefficients.subtract(map(int, negative))
    return tuple(coefficients[int(index)] for index in coordinates)


def _support_hash(support: Sequence[int]) -> str:
    return _json_sha256(list(map(int, support)))


def _signature_payload(
    signature: Mapping[
        int,
        Sequence[
            tuple[Sequence[tuple[int, int]], Sequence[int]]
        ],
    ],
) -> list[object]:
    return [
        [
            int(equation),
            [
                [
                    [list(map(int, edge)) for edge in matching],
                    list(map(int, monomial)),
                ]
                for matching, monomial in records
            ],
        ]
        for equation, records in sorted(signature.items())
    ]


def _validate_inventory(
    path: Path | None = None,
) -> dict[tuple[int, int, int, int], dict[str, object]]:
    path = CIRCUIT_INVENTORY_PATH if path is None else Path(path)
    payload = _strict_json(path)
    expected_keys = {
        "schema",
        "rank_one_class_count",
        "rank_one",
        "rank_two_class_count",
        "rank_two",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise KrennN8ToricSupport24CircuitError(
            "the circuit inventory top-level schema changed"
        )
    if payload["schema"] != CIRCUIT_INVENTORY_SCHEMA:
        raise KrennN8ToricSupport24CircuitError(
            "the circuit inventory schema label changed"
        )
    if (
        type(payload["rank_one_class_count"]) is not int
        or type(payload["rank_two_class_count"]) is not int
        or payload["rank_one_class_count"] != 440
        or payload["rank_two_class_count"] != 4
        or not isinstance(payload["rank_one"], list)
        or not isinstance(payload["rank_two"], list)
        or len(payload["rank_one"]) != 440
        or len(payload["rank_two"]) != 4
    ):
        raise KrennN8ToricSupport24CircuitError(
            "the circuit inventory class census changed"
        )

    result: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for rank, section in ((1, "rank_one"), (2, "rank_two")):
        representatives = []
        for raw_record in payload[section]:
            if (
                not isinstance(raw_record, Mapping)
                or set(raw_record)
                != {
                    "representative_secondary_matching_indices",
                    "positive_singleton_circuit",
                }
            ):
                raise KrennN8ToricSupport24CircuitError(
                    "a circuit inventory record changed shape"
                )
            raw_representative = raw_record[
                "representative_secondary_matching_indices"
            ]
            if (
                not isinstance(raw_representative, list)
                or len(raw_representative) != 4
                or any(
                    type(value) is not int
                    or value < 0
                    or value >= len(first_shell._primary_matchings())
                    for value in raw_representative
                )
            ):
                raise KrennN8ToricSupport24CircuitError(
                    "an inventory representative is invalid"
                )
            representative = tuple(raw_representative)
            raw_circuit = raw_record["positive_singleton_circuit"]
            if not isinstance(raw_circuit, list) or not raw_circuit:
                raise KrennN8ToricSupport24CircuitError(
                    "an inventory circuit is empty"
                )
            circuit = []
            for entry in raw_circuit:
                if (
                    not isinstance(entry, list)
                    or len(entry) != 2
                    or type(entry[0]) is not int
                    or not 0 <= entry[0] < COLORINGS
                    or type(entry[1]) is not int
                    or entry[1] <= 0
                ):
                    raise KrennN8ToricSupport24CircuitError(
                        "an inventory circuit entry is invalid"
                    )
                circuit.append((entry[0], entry[1]))
            equations = tuple(equation for equation, _weight in circuit)
            weights = tuple(weight for _equation, weight in circuit)
            if (
                tuple(sorted(equations)) != equations
                or len(set(equations)) != len(equations)
                or gcd(*weights) != 1
            ):
                raise KrennN8ToricSupport24CircuitError(
                    "an inventory circuit is not sorted primitive positive"
                )
            if representative in result:
                raise KrennN8ToricSupport24CircuitError(
                    "the circuit inventory repeats a representative"
                )
            representatives.append(representative)
            result[representative] = {
                "quotient_dimension": rank,
                "circuit": tuple(circuit),
            }
        if tuple(sorted(representatives)) != tuple(representatives):
            raise KrennN8ToricSupport24CircuitError(
                f"the {section} inventory is not canonically sorted"
            )
    return result


def _minimum_new_coordinate_repairs(
    coloring: Sequence[int],
    support: Sequence[int],
) -> tuple[dict[str, object], ...]:
    support_set = set(map(int, support))
    candidates = []
    positive_costs = []
    for matching_index, matching in enumerate(
        first_shell._primary_matchings()
    ):
        monomial = first_shell._monomial_for_coloring(
            coloring, matching
        )
        new_coordinates = tuple(
            sorted(set(monomial).difference(support_set))
        )
        if new_coordinates:
            positive_costs.append(len(new_coordinates))
        candidates.append(
            {
                "matching_index": matching_index,
                "monomial_variable_indices": tuple(monomial),
                "new_coordinate_indices": new_coordinates,
            }
        )
    if not positive_costs:
        raise KrennN8ToricSupport24CircuitError(
            "a parent spill has no outside repair"
        )
    minimum = min(positive_costs)
    return tuple(
        candidate
        for candidate in candidates
        if len(candidate["new_coordinate_indices"]) == minimum
    )


def _depth_two_branch_lattice(
    seed_support: Sequence[int],
    parent_support: Sequence[int],
    tie_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    support = tuple(
        sorted(
            set(map(int, parent_support)).union(
                *(
                    set(map(int, record["positive_source_monomial"]))
                    for record in tie_records
                )
            )
        )
    )
    seed_set = set(map(int, seed_support))
    nonseed = tuple(index for index in support if index not in seed_set)
    tie_matrix = tuple(
        _restricted_difference_row(
            nonseed,
            record["positive_source_monomial"],
            record["negative_source_monomial"],
        )
        for record in tie_records
    )
    tie_rank = first_shell._rank_over_q(tie_matrix)
    gauge = first_shell.target_preserving_gauge_matrix()
    seed_gauge_rank = first_shell._rank_over_q(
        tuple(gauge[index] for index in seed_support)
    )
    support_gauge_rank = first_shell._rank_over_q(
        tuple(gauge[index] for index in support)
    )
    residual_gauge_rank = support_gauge_rank - seed_gauge_rank
    quotient_dimension = (
        len(nonseed) - tie_rank - residual_gauge_rank
    )
    tie_characters = tuple(
        tuple(
            sum(
                gauge[index][column]
                for index in record["positive_source_monomial"]
            )
            - sum(
                gauge[index][column]
                for index in record["negative_source_monomial"]
            )
            for column in range(GAUGE_DIMENSION)
        )
        for record in tie_records
    )
    checks = {
        "seed_gauge_rank_is_nine": seed_gauge_rank == 9,
        "all_six_ties_are_full_gauge_invariant": (
            len(tie_records) == 6
            and all(not any(character) for character in tie_characters)
        ),
        "residual_gauge_fits_tie_kernel": (
            residual_gauge_rank <= len(nonseed) - tie_rank
        ),
        "quotient_dimension_is_nonnegative": quotient_dimension >= 0,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricSupport24CircuitError(
            f"a depth-two branch lattice failed: {failed}"
        )
    return {
        "support": support,
        "nonseed": nonseed,
        "tie_matrix": tie_matrix,
        "tie_rank": tie_rank,
        "support_gauge_rank": support_gauge_rank,
        "residual_gauge_rank": residual_gauge_rank,
        "quotient_dimension": quotient_dimension,
        "exact_checks": checks,
    }


def _base_decoration_key(
    decoration: Sequence[
        tuple[Sequence[int], Sequence[int]]
    ],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    return first_shell._decoration_key(
        decoration, tuple(range(N)), tuple(range(D))
    )


def _full_decoration(
    context: Mapping[str, object],
    secondary_options: Sequence[Mapping[str, object]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    decoration = []
    hard_case = context["hard_case"]
    parent = context["parent"]
    for victim, matching_index_value in zip(
        hard_case["original_singleton_victims"],
        parent["representative_repair_matching_indices"],
        strict=True,
    ):
        matching_index = int(matching_index_value)
        decoration.append(
            (
                tuple(map(int, victim["coloring"])),
                first_shell._monomial_for_coloring(
                    victim["coloring"],
                    first_shell._primary_matchings()[matching_index],
                ),
            )
        )
    for spill, option in zip(
        context["spills"], secondary_options, strict=True
    ):
        decoration.append(
            (
                tuple(map(int, spill["coloring"])),
                tuple(map(int, option["monomial_variable_indices"])),
            )
        )
    return tuple(decoration)


@lru_cache(maxsize=1)
def _parent_context() -> dict[str, object]:
    certificate = first_shell.build_n8_toric_first_shell_certificate()
    hard_case = certificate["flat_first_shell"]["hard_cases"]["H5"]
    parent = hard_case["orbit_records"][PARENT_EXPECTED["orbit_index"]]
    parent_support = tuple(
        map(
            int,
            parent["flat_valuation"]["order_zero_coordinate_indices"],
        )
    )
    spills = tuple(
        sorted(
            parent["initial_form_replay"][
                "zero_target_singleton_initial_forms"
            ],
            key=lambda record: int(record["equation"]),
        )
    )
    if (
        tuple(parent["representative_repair_matching_indices"])
        != PARENT_EXPECTED["repair_matching_indices"]
        or parent_support != PARENT_EXPECTED["support"]
        or tuple(int(record["equation"]) for record in spills)
        != PARENT_EXPECTED["spill_equations"]
    ):
        raise KrennN8ToricSupport24CircuitError(
            "the deterministic H5 depth-two parent changed"
        )
    secondary_tables = tuple(
        _minimum_new_coordinate_repairs(
            spill["coloring"], parent_support
        )
        for spill in spills
    )
    if any(
        len(table) != PARENT_EXPECTED["repair_options_per_spill"]
        or any(
            len(option["new_coordinate_indices"]) != 2
            for option in table
        )
        for table in secondary_tables
    ):
        raise KrennN8ToricSupport24CircuitError(
            "the four minimum-cost repair tables changed"
        )
    seed_normalization = certificate["gauge_quotient"][
        "seed_normalizations"
    ]["H5"]
    seed_support = tuple(map(int, seed_normalization["seed_support"]))
    original_ties = []
    parent_decoration = []
    for victim, matching_index_value in zip(
        hard_case["original_singleton_victims"],
        parent["representative_repair_matching_indices"],
        strict=True,
    ):
        matching_index = int(matching_index_value)
        positive = first_shell._monomial_for_coloring(
            victim["coloring"],
            first_shell._primary_matchings()[matching_index],
        )
        negative = tuple(
            map(int, victim["source_monomial_variable_indices"])
        )
        original_ties.append(
            {
                "kind": "original_victim_tie",
                "positive_source_monomial": tuple(positive),
                "negative_source_monomial": negative,
            }
        )
        parent_decoration.append(
            (
                tuple(map(int, victim["coloring"])),
                tuple(positive),
            )
        )
    parent_key = _base_decoration_key(parent_decoration)
    parent_stabilizer = tuple(
        (vertices, colors)
        for vertices, colors in first_shell._seed_stabilizer("H5")
        if first_shell._decoration_key(
            parent_decoration, vertices, colors
        )
        == parent_key
    )
    nonidentity = tuple(
        (vertices, colors)
        for vertices, colors in parent_stabilizer
        if vertices != tuple(range(N)) or colors != tuple(range(D))
    )
    expected_nonidentity = (
        NONTRIVIAL_PARENT_SYMMETRY["vertices"],
        NONTRIVIAL_PARENT_SYMMETRY["colors"],
    )
    if (
        len(parent_stabilizer)
        != EXPECTED_CLASS_DATA["parent_stabilizer_order"]
        or nonidentity != (expected_nonidentity,)
        or tuple(
            expected_nonidentity[0][expected_nonidentity[0][index]]
            for index in range(N)
        )
        != tuple(range(N))
        or tuple(
            expected_nonidentity[1][expected_nonidentity[1][index]]
            for index in range(D)
        )
        != tuple(range(D))
    ):
        raise KrennN8ToricSupport24CircuitError(
            "the fixed H5 parent stabilizer is not the expected C2"
        )
    return {
        "certificate": certificate,
        "hard_case": hard_case,
        "parent": parent,
        "parent_support": parent_support,
        "spills": spills,
        "secondary_tables": secondary_tables,
        "seed_support": seed_support,
        "original_ties": tuple(original_ties),
        "parent_decoration": tuple(parent_decoration),
        "parent_stabilizer": parent_stabilizer,
        "nonidentity_parent_symmetry": expected_nonidentity,
    }


@lru_cache(maxsize=1)
def _enumerated_support24_layer() -> dict[str, object]:
    context = _parent_context()
    seed_support = context["seed_support"]
    seed_set = set(seed_support)
    parent_support = context["parent_support"]
    branch_histogram: Counter[tuple[int, int]] = Counter()
    lattice_cache: dict[object, dict[str, object]] = {}
    positive_support24 = []
    raw_count = 0
    for secondary_options in product(*context["secondary_tables"]):
        raw_count += 1
        secondary_ties = tuple(
            {
                "kind": "spill_repair_tie",
                "equation": int(spill["equation"]),
                "positive_source_monomial": tuple(
                    map(int, option["monomial_variable_indices"])
                ),
                "negative_source_monomial": tuple(
                    map(int, spill["monomial_variable_indices"])
                ),
            }
            for spill, option in zip(
                context["spills"], secondary_options, strict=True
            )
        )
        tie_records = (*context["original_ties"], *secondary_ties)
        support_key = tuple(
            sorted(
                set(parent_support).union(
                    *(
                        set(record["positive_source_monomial"])
                        for record in secondary_ties
                    )
                )
            )
        )
        nonseed_key = tuple(
            index for index in support_key if index not in seed_set
        )
        matrix_key = tuple(
            _restricted_difference_row(
                nonseed_key,
                record["positive_source_monomial"],
                record["negative_source_monomial"],
            )
            for record in tie_records
        )
        cache_key = (support_key, matrix_key)
        lattice = lattice_cache.get(cache_key)
        if lattice is None:
            lattice = _depth_two_branch_lattice(
                seed_support, parent_support, tie_records
            )
            lattice_cache[cache_key] = lattice
        support_size = len(lattice["support"])
        quotient_dimension = int(lattice["quotient_dimension"])
        branch_histogram[(support_size, quotient_dimension)] += 1
        if support_size == 24 and quotient_dimension > 0:
            decoration = _full_decoration(context, secondary_options)
            positive_support24.append(
                {
                    "secondary_matching_indices": tuple(
                        int(option["matching_index"])
                        for option in secondary_options
                    ),
                    "tie_records": tie_records,
                    "lattice": lattice,
                    "decoration": decoration,
                    "decoration_key": _base_decoration_key(decoration),
                }
            )
    if (
        raw_count != PARENT_EXPECTED["raw_branch_count"]
        or dict(sorted(branch_histogram.items()))
        != EXPECTED_BRANCH_HISTOGRAM
        or len(positive_support24)
        != EXPECTED_CLASS_DATA["raw_positive_support24"]
        or not all(
            all(lattice["exact_checks"].values())
            for lattice in lattice_cache.values()
        )
    ):
        raise KrennN8ToricSupport24CircuitError(
            "the fixed-parent depth-two branch census changed"
        )

    stabilizer = context["parent_stabilizer"]
    canonical_groups: dict[object, list[dict[str, object]]] = {}
    for branch in positive_support24:
        orbit_images = tuple(
            first_shell._decoration_key(
                branch["decoration"], vertices, colors
            )
            for vertices, colors in stabilizer
        )
        canonical_groups.setdefault(min(orbit_images), []).append(branch)
    groups = []
    for members in canonical_groups.values():
        members = sorted(
            members,
            key=lambda record: record["secondary_matching_indices"],
        )
        representative = members[0]
        orbit_images = {
            first_shell._decoration_key(
                representative["decoration"], vertices, colors
            )
            for vertices, colors in stabilizer
        }
        if orbit_images != {
            member["decoration_key"] for member in members
        }:
            raise KrennN8ToricSupport24CircuitError(
                "a complete-decoration group is not a parent-C2 orbit"
            )
        dimensions = {
            int(member["lattice"]["quotient_dimension"])
            for member in members
        }
        if len(dimensions) != 1:
            raise KrennN8ToricSupport24CircuitError(
                "a C2 orbit mixes quotient dimensions"
            )
        groups.append(
            {
                "representative": representative,
                "members": tuple(members),
                "quotient_dimension": dimensions.pop(),
            }
        )
    groups.sort(
        key=lambda group: group["representative"][
            "secondary_matching_indices"
        ]
    )
    rank_one = tuple(
        group for group in groups if group["quotient_dimension"] == 1
    )
    rank_two = tuple(
        group for group in groups if group["quotient_dimension"] == 2
    )
    rank_one_raw = sum(len(group["members"]) for group in rank_one)
    rank_two_raw = sum(len(group["members"]) for group in rank_two)
    rank_one_orbits = Counter(len(group["members"]) for group in rank_one)
    rank_two_orbits = Counter(len(group["members"]) for group in rank_two)
    rank_one_supports = {
        tuple(branch["lattice"]["support"])
        for group in rank_one
        for branch in group["members"]
    }
    rank_two_supports = {
        tuple(branch["lattice"]["support"])
        for group in rank_two
        for branch in group["members"]
    }
    if (
        len(groups) != EXPECTED_CLASS_DATA["classes"]
        or len(rank_one) != EXPECTED_CLASS_DATA["rank_one_classes"]
        or len(rank_two) != EXPECTED_CLASS_DATA["rank_two_classes"]
        or rank_one_raw != EXPECTED_CLASS_DATA["rank_one_raw"]
        or rank_two_raw != EXPECTED_CLASS_DATA["rank_two_raw"]
        or dict(rank_one_orbits)
        != EXPECTED_CLASS_DATA["rank_one_orbit_size_histogram"]
        or dict(rank_two_orbits)
        != EXPECTED_CLASS_DATA["rank_two_orbit_size_histogram"]
        or len(rank_one_supports)
        != EXPECTED_CLASS_DATA["rank_one_distinct_supports"]
        or len(rank_two_supports)
        != EXPECTED_CLASS_DATA["rank_two_distinct_supports"]
    ):
        raise KrennN8ToricSupport24CircuitError(
            "the support-24 parent-C2 class census changed"
        )
    return {
        "context": context,
        "raw_count": raw_count,
        "branch_histogram": dict(sorted(branch_histogram.items())),
        "distinct_lattice_systems": len(lattice_cache),
        "positive_support24": tuple(positive_support24),
        "groups": tuple(groups),
        "rank_one_groups": rank_one,
        "rank_two_groups": rank_two,
        "rank_one_raw": rank_one_raw,
        "rank_two_raw": rank_two_raw,
        "rank_one_orbit_size_histogram": dict(sorted(rank_one_orbits.items())),
        "rank_two_orbit_size_histogram": dict(sorted(rank_two_orbits.items())),
        "rank_one_distinct_supports": len(rank_one_supports),
        "rank_two_distinct_supports": len(rank_two_supports),
    }


@lru_cache(maxsize=None)
def _active_signature(
    support: tuple[int, ...],
) -> tuple[
    dict[
        int,
        tuple[
            tuple[tuple[tuple[int, int], ...], tuple[int, ...]],
            ...,
        ],
    ],
    str,
]:
    primary = first_shell._active_term_signature(
        support, first_shell._primary_matchings()
    )
    independent = first_shell._active_term_signature(
        support, first_shell._independent_matchings()
    )
    if primary != independent:
        raise KrennN8ToricSupport24CircuitError(
            "the two perfect-matching enumerators disagree"
        )
    return primary, _json_sha256(_signature_payload(primary))


def _target_receipt(
    branch: Mapping[str, object],
) -> dict[str, object]:
    lattice = branch["lattice"]
    support = tuple(lattice["support"])
    nonseed = tuple(lattice["nonseed"])
    tie_matrix = tuple(lattice["tie_matrix"])
    signature, signature_hash = _active_signature(support)
    matching_index = {
        matching: index
        for index, matching in enumerate(first_shell._primary_matchings())
    }
    records = []
    term_counts = []
    for color in range(D):
        equation = first_shell.coloring_index(N, D, (color,) * N)
        terms = signature.get(equation, ())
        if not terms:
            raise KrennN8ToricSupport24CircuitError(
                "a support-24 branch lost a monochromatic target"
            )
        term_counts.append(len(terms))
        for matching, monomial in terms:
            incidence = _incidence_row(nonseed, monomial)
            coefficients = _row_span_coefficients(
                tie_matrix, incidence
            )
            records.append(
                {
                    "color": color,
                    "equation": equation,
                    "matching_index": matching_index[matching],
                    "monomial_variable_indices": list(map(int, monomial)),
                    "restricted_nonseed_incidence": list(incidence),
                    "tie_row_combination_coefficients": (
                        _fraction_records(coefficients)
                    ),
                }
            )
    return {
        "active_signature_sha256": signature_hash,
        "active_target_term_count_triple": term_counts,
        "active_target_terms": records,
        "all_active_target_incidence_rows_lie_in_tie_row_span": True,
        "common_target_order_on_tie_kernel": 0,
    }


def _circuit_receipt(
    branch: Mapping[str, object],
    circuit: Sequence[tuple[int, int]],
) -> dict[str, object]:
    lattice = branch["lattice"]
    support = tuple(lattice["support"])
    nonseed = tuple(lattice["nonseed"])
    tie_matrix = tuple(lattice["tie_matrix"])
    signature, signature_hash = _active_signature(support)
    targets = {
        first_shell.coloring_index(N, D, (color,) * N)
        for color in range(D)
    }
    matching_index = {
        matching: index
        for index, matching in enumerate(first_shell._primary_matchings())
    }
    terms = []
    singleton_rows = []
    weights = []
    for equation_value, weight_value in circuit:
        equation = int(equation_value)
        weight = int(weight_value)
        active = signature.get(equation, ())
        if equation in targets or len(active) != 1 or weight <= 0:
            raise KrennN8ToricSupport24CircuitError(
                "an inventoried equation is not a positive mixed singleton"
            )
        matching, monomial = active[0]
        incidence = _incidence_row(nonseed, monomial)
        singleton_rows.append(incidence)
        weights.append(weight)
        terms.append(
            {
                "equation": equation,
                "coloring": list(
                    first_shell.coloring_from_index(N, D, equation)
                ),
                "positive_coefficient": weight,
                "matching_index": matching_index[matching],
                "matching": [list(edge) for edge in matching],
                "monomial_variable_indices": list(map(int, monomial)),
                "restricted_nonseed_incidence": list(incidence),
            }
        )
    weighted = tuple(
        sum(
            weight * row[column]
            for weight, row in zip(weights, singleton_rows, strict=True)
        )
        for column in range(len(nonseed))
    )
    combination = _row_span_coefficients(tie_matrix, weighted)
    tie_rank = first_shell._rank_over_q(tie_matrix)
    augmented_rank = first_shell._rank_over_q(
        (*tie_matrix, *singleton_rows)
    )
    checks = {
        "primary_independent_active_signatures_agree": True,
        "all_listed_equations_are_mixed_support_singletons": (
            len(terms) == len(circuit)
        ),
        "all_coefficients_are_strictly_positive": all(
            weight > 0 for weight in weights
        ),
        "positive_coefficients_are_primitive": gcd(*weights) == 1,
        "tie_matrix_has_six_rows_and_rank_six": (
            len(tie_matrix) == tie_rank == 6
        ),
        "weighted_incidence_row_lies_in_tie_row_span": True,
        "circuit_is_minimal_modulo_tie_span": (
            augmented_rank == tie_rank + len(singleton_rows) - 1
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricSupport24CircuitError(
            f"a positive singleton circuit failed: {failed}"
        )
    return {
        "active_signature_sha256": signature_hash,
        "terms": terms,
        "weighted_nonseed_incidence": list(weighted),
        "tie_row_combination_coefficients": _fraction_records(combination),
        "tie_rank": tie_rank,
        "tie_plus_singleton_rows_rank": augmented_rank,
        "positive_relation_consequence": (
            "on every valuation in the six-tie kernel, at least one "
            "listed unique mixed support monomial has order at most zero"
        ),
        "exact_checks": checks,
    }


def _compact_target_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Keep replay commitments without duplicating reconstructed terms."""

    terms = receipt["active_target_terms"]
    return {
        "active_signature_sha256": receipt[
            "active_signature_sha256"
        ],
        "active_target_term_count_triple": list(
            receipt["active_target_term_count_triple"]
        ),
        "active_target_term_count": len(terms),
        "active_target_terms_sha256": _json_sha256(terms),
        "all_active_target_incidence_rows_lie_in_tie_row_span": (
            receipt[
                "all_active_target_incidence_rows_lie_in_tie_row_span"
            ]
        ),
        "common_target_order_on_tie_kernel": receipt[
            "common_target_order_on_tie_kernel"
        ],
    }


def _compact_circuit_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Commit to every reconstructed singleton while storing no duplicate."""

    terms = receipt["terms"]
    return {
        "active_signature_sha256": receipt[
            "active_signature_sha256"
        ],
        "circuit": [
            [
                int(record["equation"]),
                int(record["positive_coefficient"]),
            ]
            for record in terms
        ],
        "reconstructed_singleton_term_count": len(terms),
        "reconstructed_singleton_terms_sha256": _json_sha256(terms),
        "weighted_nonseed_incidence": list(
            receipt["weighted_nonseed_incidence"]
        ),
        "tie_row_combination_coefficients": deepcopy(
            receipt["tie_row_combination_coefficients"]
        ),
        "tie_rank": receipt["tie_rank"],
        "tie_plus_singleton_rows_rank": receipt[
            "tie_plus_singleton_rows_rank"
        ],
        "positive_relation_consequence": receipt[
            "positive_relation_consequence"
        ],
        "exact_checks": deepcopy(receipt["exact_checks"]),
    }


def _compact_partner_transport(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    return {
        "vertices": list(receipt["vertices"]),
        "colors": list(receipt["colors"]),
        "partner_secondary_matching_indices": list(
            receipt["partner_secondary_matching_indices"]
        ),
        "partner_support_sha256": receipt[
            "partner_support_sha256"
        ],
        "transported_circuit": _compact_circuit_receipt(
            receipt["transported_circuit"]
        ),
        "transported_equations_and_monomials_match_exactly": (
            receipt[
                "transported_equations_and_monomials_match_exactly"
            ]
        ),
    }


def _transported_partner_receipt(
    representative: Mapping[str, object],
    partner: Mapping[str, object],
    representative_receipt: Mapping[str, object],
    symmetry: tuple[Sequence[int], Sequence[int]],
) -> dict[str, object]:
    vertices, colors = symmetry
    transformed_support = tuple(
        sorted(
            first_shell._transport_variable(index, vertices, colors)
            for index in representative["lattice"]["support"]
        )
    )
    if transformed_support != tuple(partner["lattice"]["support"]):
        raise KrennN8ToricSupport24CircuitError(
            "the parent involution does not transport the support"
        )
    transformed_entries = []
    expected_terms = {}
    for record in representative_receipt["terms"]:
        coloring = first_shell.coloring_from_index(
            N, D, int(record["equation"])
        )
        transformed_coloring = first_shell._transport_coloring(
            coloring, vertices, colors
        )
        equation = first_shell.coloring_index(
            N, D, transformed_coloring
        )
        monomial = tuple(
            sorted(
                first_shell._transport_variable(
                    index, vertices, colors
                )
                for index in record["monomial_variable_indices"]
            )
        )
        matching = first_shell._transport_matching(
            tuple(tuple(edge) for edge in record["matching"]), vertices
        )
        transformed_entries.append(
            (equation, int(record["positive_coefficient"]))
        )
        expected_terms[equation] = (matching, monomial)
    transformed_entries.sort()
    partner_receipt = _circuit_receipt(partner, transformed_entries)
    actual_terms = {
        int(record["equation"]): (
            tuple(tuple(edge) for edge in record["matching"]),
            tuple(record["monomial_variable_indices"]),
        )
        for record in partner_receipt["terms"]
    }
    if actual_terms != expected_terms:
        raise KrennN8ToricSupport24CircuitError(
            "a circuit term failed exact C2 transport"
        )
    return {
        "vertices": list(map(int, vertices)),
        "colors": list(map(int, colors)),
        "partner_secondary_matching_indices": list(
            partner["secondary_matching_indices"]
        ),
        "partner_support_sha256": _support_hash(
            partner["lattice"]["support"]
        ),
        "transported_circuit": partner_receipt,
        "transported_equations_and_monomials_match_exactly": True,
    }


@lru_cache(maxsize=1)
def _build_certificate() -> dict[str, object]:
    layer = _enumerated_support24_layer()
    inventory = _validate_inventory()
    representative_tuples = {
        group["representative"]["secondary_matching_indices"]
        for group in layer["groups"]
    }
    if set(inventory) != representative_tuples:
        raise KrennN8ToricSupport24CircuitError(
            "the circuit inventory does not cover exactly the 444 C2 reps"
        )

    context = layer["context"]
    vertices, colors = context["nonidentity_parent_symmetry"]
    class_records = []
    length_histograms = {1: Counter(), 2: Counter()}
    coefficient_histograms = {1: Counter(), 2: Counter()}
    target_pattern_histograms = {1: Counter(), 2: Counter()}
    for group in layer["groups"]:
        representative = group["representative"]
        representative_tuple = representative[
            "secondary_matching_indices"
        ]
        quotient_dimension = int(group["quotient_dimension"])
        inventory_record = inventory[representative_tuple]
        if (
            int(inventory_record["quotient_dimension"])
            != quotient_dimension
        ):
            raise KrennN8ToricSupport24CircuitError(
                "an inventory section disagrees with exact quotient rank"
            )
        circuit = tuple(inventory_record["circuit"])
        representative_circuit = _circuit_receipt(
            representative, circuit
        )
        representative_targets = _target_receipt(representative)
        length_histograms[quotient_dimension][len(circuit)] += 1
        coefficient_histograms[quotient_dimension].update(
            weight for _equation, weight in circuit
        )
        target_pattern_histograms[quotient_dimension][
            tuple(
                representative_targets[
                    "active_target_term_count_triple"
                ]
            )
        ] += 1

        partner_key = first_shell._decoration_key(
            representative["decoration"], vertices, colors
        )
        partner = next(
            (
                member
                for member in group["members"]
                if member["decoration_key"] == partner_key
            ),
            None,
        )
        if partner is None:
            raise KrennN8ToricSupport24CircuitError(
                "the C2 partner left its computed orbit"
            )
        transport = _transported_partner_receipt(
            representative,
            partner,
            representative_circuit,
            (vertices, colors),
        )
        partner_targets = _target_receipt(partner)
        lattice = representative["lattice"]
        class_checks = {
            "support_size_is_24": len(lattice["support"]) == 24,
            "nonseed_coordinate_count_is_12": (
                len(lattice["nonseed"]) == 12
            ),
            "tie_rank_is_6": lattice["tie_rank"] == 6,
            "quotient_dimension_matches_inventory_section": (
                lattice["quotient_dimension"] == quotient_dimension
            ),
            "residual_gauge_rank_is_expected": (
                lattice["residual_gauge_rank"]
                == (5 if quotient_dimension == 1 else 4)
            ),
            "representative_is_lexicographically_least_in_C2_orbit": (
                representative_tuple
                == min(
                    member["secondary_matching_indices"]
                    for member in group["members"]
                )
            ),
            "representative_target_rows_are_forced_zero": (
                representative_targets[
                    "all_active_target_incidence_rows_lie_in_tie_row_span"
                ]
            ),
            "partner_target_rows_are_forced_zero": (
                partner_targets[
                    "all_active_target_incidence_rows_lie_in_tie_row_span"
                ]
            ),
            "representative_positive_circuit_passes": all(
                representative_circuit["exact_checks"].values()
            ),
            "transported_partner_positive_circuit_passes": all(
                transport["transported_circuit"][
                    "exact_checks"
                ].values()
            ),
            "partner_transport_matches_equations_and_monomials": (
                transport[
                    "transported_equations_and_monomials_match_exactly"
                ]
            ),
        }
        if not all(class_checks.values()):
            failed = sorted(
                key for key, value in class_checks.items() if not value
            )
            raise KrennN8ToricSupport24CircuitError(
                f"support-24 class {representative_tuple} failed: {failed}"
            )
        class_records.append(
            {
                "representative_secondary_matching_indices": list(
                    representative_tuple
                ),
                "parent_C2_orbit_size": len(group["members"]),
                "support_sha256": _support_hash(lattice["support"]),
                "nonseed_coordinates_sha256": _json_sha256(
                    list(lattice["nonseed"])
                ),
                "tie_matrix_sha256": _json_sha256(
                    [list(row) for row in lattice["tie_matrix"]]
                ),
                "tie_matrix_rank": lattice["tie_rank"],
                "residual_gauge_rank": lattice[
                    "residual_gauge_rank"
                ],
                "quotient_dimension": quotient_dimension,
                "representative_target_order_receipt": (
                    _compact_target_receipt(representative_targets)
                ),
                "representative_positive_singleton_circuit": (
                    _compact_circuit_receipt(representative_circuit)
                ),
                "parent_C2_partner_transport": (
                    _compact_partner_transport(transport)
                ),
                "partner_target_order_receipt": (
                    _compact_target_receipt(partner_targets)
                ),
                "exact_checks": class_checks,
            }
        )

    histogram_checks = {
        "rank_one_circuit_length_histogram_exact": (
            dict(sorted(length_histograms[1].items()))
            == EXPECTED_CIRCUIT_LENGTH_HISTOGRAMS[1]
        ),
        "rank_two_circuit_length_histogram_exact": (
            dict(sorted(length_histograms[2].items()))
            == EXPECTED_CIRCUIT_LENGTH_HISTOGRAMS[2]
        ),
        "rank_one_coefficient_histogram_exact": (
            dict(sorted(coefficient_histograms[1].items()))
            == EXPECTED_CIRCUIT_COEFFICIENT_HISTOGRAMS[1]
        ),
        "rank_two_coefficient_histogram_exact": (
            dict(sorted(coefficient_histograms[2].items()))
            == EXPECTED_CIRCUIT_COEFFICIENT_HISTOGRAMS[2]
        ),
        "rank_one_target_pattern_histogram_exact": (
            dict(sorted(target_pattern_histograms[1].items()))
            == EXPECTED_RANK_ONE_TARGET_PATTERN_HISTOGRAM
        ),
        "all_rank_two_target_patterns_are_1_1_1": (
            dict(target_pattern_histograms[2]) == {(1, 1, 1): 4}
        ),
    }
    checks = {
        "all_20736_fixed_parent_branches_enumerated": (
            layer["raw_count"] == 20_736
        ),
        "branch_support_and_quotient_histogram_exact": (
            layer["branch_histogram"] == EXPECTED_BRANCH_HISTOGRAM
        ),
        "exactly_860_positive_support24_raw_branches": (
            len(layer["positive_support24"]) == 860
        ),
        "exactly_444_parent_C2_classes": len(class_records) == 444,
        "all_inventory_representatives_used_once": (
            len({tuple(record[
                "representative_secondary_matching_indices"
            ]) for record in class_records}) == len(inventory) == 444
        ),
        "all_representative_and_partner_circuits_pass": all(
            all(record["exact_checks"].values())
            for record in class_records
        ),
        "all_active_target_terms_forced_to_order_zero": all(
            record["representative_target_order_receipt"][
                "all_active_target_incidence_rows_lie_in_tie_row_span"
            ]
            and record["partner_target_order_receipt"][
                "all_active_target_incidence_rows_lie_in_tie_row_span"
            ]
            for record in class_records
        ),
        **histogram_checks,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricSupport24CircuitError(
            f"the support-24 circuit certificate failed: {failed}"
        )

    branch_histogram_json = {
        f"{support},{dimension}": count
        for (support, dimension), count in sorted(
            layer["branch_histogram"].items()
        )
    }
    return {
        "schema": SUPPORT24_CIRCUIT_SCHEMA,
        "scope": {
            "parameters": {"n": N, "d": D},
            "seed_orbit": "H5",
            "fixed_first_shell_parent_orbit_index": 3,
            "fixed_parent_repair_matching_indices": [0, 11],
            "fixed_parent_spill_equations": [7, 63, 4_049, 5_791],
            "raw_branches_enumerated": 20_736,
            "positive_support24_raw_branches": 860,
            "parent_C2_classes": 444,
            "floating_point_used": False,
            "random_seeds": [],
            "cpu_workers": 1,
        },
        "declared_family": {
            "exact_source_support_equals_recorded_24_coordinates": True,
            "all_recorded_support_coordinates_have_nonzero_leading_coefficients": (
                True
            ),
            "outside_source_coordinates_are_zero_on_exact_support_torus": (
                True
            ),
            "degeneration_interpretation_requires_every_outside_monomial_strictly_above_relevant_target_and_singleton_orders": (
                True
            ),
            "outside_term_entry_cones_classified": False,
            "common_active_target_monomial_order_normalized_to_zero": (
                True
            ),
            "degree_four_base_change_allowed": True,
            "residual_target_preserving_gauge_covered_by_identity_on_full_tie_kernel": (
                True
            ),
            "interpretation": (
                "the exact-support torus sets every coordinate outside "
                "the recorded support to zero; a degeneration with "
                "additional nonzero coordinates is covered only while "
                "every outside monomial remains strictly above the "
                "recorded target and singleton initial orders"
            ),
        },
        "inventory": {
            "schema": CIRCUIT_INVENTORY_SCHEMA,
            "path": (
                "experiments/krenn_quantum_graph/"
                "n8_support24_positive_circuits.json"
            ),
            "bytes": CIRCUIT_INVENTORY_PATH.stat().st_size,
            "sha256": sha256(
                CIRCUIT_INVENTORY_PATH.read_bytes()
            ).hexdigest(),
            "rank_one_classes": 440,
            "rank_two_classes": 4,
        },
        "fixed_parent_C2": {
            "order": 2,
            "nonidentity_vertices": list(vertices),
            "nonidentity_colors": list(colors),
            "rank_one_orbit_size_histogram": {
                str(key): value
                for key, value in sorted(
                    layer["rank_one_orbit_size_histogram"].items()
                )
            },
            "rank_two_orbit_size_histogram": {
                str(key): value
                for key, value in sorted(
                    layer["rank_two_orbit_size_histogram"].items()
                )
            },
        },
        "branch_census": {
            "support_size_and_quotient_dimension_histogram": (
                branch_histogram_json
            ),
            "distinct_support_and_tie_lattice_systems": layer[
                "distinct_lattice_systems"
            ],
            "rank_one_raw_branches": layer["rank_one_raw"],
            "rank_two_raw_branches": layer["rank_two_raw"],
            "rank_one_distinct_supports": layer[
                "rank_one_distinct_supports"
            ],
            "rank_two_distinct_supports": layer[
                "rank_two_distinct_supports"
            ],
        },
        "positive_circuit_theorem": {
            "tie_rows_per_class": 6,
            "nonseed_coordinates_per_class": 12,
            "target_order_statement": (
                "every active monochromatic target monomial incidence "
                "row lies in the six-tie row span and therefore has "
                "order zero on the tie kernel"
            ),
            "mixed_order_statement": (
                "a primitive strictly-positive combination of mixed "
                "singleton incidence rows lies in the six-tie row span, "
                "so at least one unique mixed term has order at most zero"
            ),
            "target_leading_coefficient_cancellation_does_not_rescue": (
                "cancellation can only raise the target order above zero"
            ),
            "canonical_quotient_lift_used": False,
            "sampled_order_counts_used": False,
            "finite_field_used": False,
            "classes": class_records,
        },
        "rank_two_fan_diagnostic": {
            "included": False,
            "used_for_exclusion": False,
            "reason": (
                "the positive-circuit theorem is basis-free; no quotient "
                "fan convention is needed for this certificate"
            ),
        },
        "conclusion": {
            "classification": (
                "exact-negative-bounded-fixed-parent-support24-circuits"
            ),
            "bounded_classes_excluded": 444,
            "bounded_raw_branches_excluded": 860,
            "exclusion_basis": "primitive positive singleton circuits",
            "all_circuits_verified_by_both_matching_enumerators": True,
            "all_partner_transports_verified": True,
            "all_target_orders_verified_zero_on_tie_kernel": True,
        },
        "claim_boundary": {
            "bounded_fixed_H5_parent_support24_positive_quotient_layer_excluded": (
                True
            ),
            "all_20736_fixed_parent_branches_enumerated": True,
            "all_support24_positive_quotient_branches_in_that_enumeration_excluded": (
                True
            ),
            "exclusion_is_on_declared_exact_support_or_no_outside_entry_stratum": (
                True
            ),
            "support22_layer_reproved_here": False,
            "support24_zero_quotient_branches_excluded_here": False,
            "another_H5_parent_excluded": False,
            "all_H5_depth_two_parents_excluded": False,
            "deeper_repair_closures_excluded": False,
            "outside_support_terms_at_leading_order_classified": False,
            "all_31_n8_seed_orbits_classified": False,
            "n8_projective_border_membership_proved": False,
            "n8_strict_border_membership_proved": False,
            "n8_affine_membership_proved": False,
            "n8_nonexistence_proved": False,
            "finite_counterexample_found": False,
        },
        "exact_checks": checks,
    }


def build_n8_toric_support24_circuit_certificate() -> dict[str, object]:
    """Return the exact bounded support-24 positive-circuit certificate."""

    return deepcopy(_build_certificate())


def verify_n8_toric_support24_circuit_certificate(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Recompute all 20,736 branches and reject any payload change."""

    if not isinstance(payload, Mapping):
        raise KrennN8ToricSupport24CircuitError(
            "the support-24 circuit certificate must be a mapping"
        )
    expected = build_n8_toric_support24_circuit_certificate()
    if dict(payload) != expected:
        raise KrennN8ToricSupport24CircuitError(
            "the support-24 circuit certificate differs from exact replay"
        )
    return expected


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _readme_text(certificate: Mapping[str, object]) -> str:
    census = certificate["branch_census"]
    conclusion = certificate["conclusion"]
    return f"""# Exact bounded `n=8,d=3` support-24 circuit gate

This bundle covers one fixed H5 first-shell parent only.  It enumerates all
20,736 choices of one minimum-cost repair for each of the parent's four
mixed singleton spills.  The positive support-24 layer contains
{conclusion["bounded_raw_branches_excluded"]} raw branches in
{conclusion["bounded_classes_excluded"]} classes under the true order-two
parent stabilizer.

## Exact exclusion

Every class has six repair-tie rows on twelve nonseed coordinates.  Every
active monochromatic target monomial has incidence row in their rational
row span, so its order is zero on the tie kernel.

For every class, `certificate.json` also records a primitive positive
integer combination of mixed support-singleton incidence rows that lies in
the same tie-row span.  Therefore at least one unique mixed support term has
order at most zero and cannot cancel on the declared leading-support chart.
Both independent perfect-matching enumerators replay every active signature,
and the parent involution transports every representative circuit to its
partner exactly.

| quotient dimension | raw branches | C2 classes | distinct supports |
|---:|---:|---:|---:|
| 1 | {census["rank_one_raw_branches"]} | 440 | {census["rank_one_distinct_supports"]} |
| 2 | {census["rank_two_raw_branches"]} | 4 | {census["rank_two_distinct_supports"]} |

The fact that 852 rank-one branches use only 850 supports is why the
certificate keys branches by complete decorated tie system, not support
alone.

## Boundary

This is not a global `n=8` nonexistence proof.  It does not exclude another
first-shell parent, a deeper repair closure, a zero-quotient support-24
branch, a different support, or a chart where an outside-support term enters
at leading order.  It neither finds nor rules out a finite counterexample.

The rank-two fan is not used as evidence here.  The positive-circuit theorem
is exact and basis-free.

## Reproduce

```powershell
C:\\tmp\\Krenn-obstruction-venv\\Scripts\\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_support24_circuits --results-directory results\\krenn_quantum_graph\\n8_d3_toric_support24_circuits
```

```powershell
C:\\tmp\\Krenn-obstruction-venv\\Scripts\\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_support24_circuits --results-directory results\\krenn_quantum_graph\\n8_d3_toric_support24_circuits --verify-only
```
"""


def _expected_manifest(
    directory: Path,
    certificate: Mapping[str, object],
) -> dict[str, object]:
    root = _repo_root()
    return {
        "schema": SUPPORT24_CIRCUIT_MANIFEST_SCHEMA,
        "artifacts": [
            _file_record(directory / CERTIFICATE_FILE, relative_to=directory),
            _file_record(directory / README_FILE, relative_to=directory),
        ],
        "inputs": [
            _file_record(root / relative, relative_to=root)
            for relative in SOURCE_PATHS
        ],
        "claim_boundary": certificate["claim_boundary"],
        "scratch_data_required_for_verification": False,
    }


def write_n8_toric_support24_circuit_bundle(
    directory: Path | str = DEFAULT_RESULTS_DIRECTORY,
) -> dict[str, object]:
    """Write the exact certificate, README, and source-hash manifest."""

    directory = Path(directory)
    if not directory.is_absolute():
        directory = _repo_root() / directory
    directory.mkdir(parents=True, exist_ok=True)
    expected_names = {CERTIFICATE_FILE, README_FILE, MANIFEST_FILE}
    unexpected = {
        path.name
        for path in directory.iterdir()
        if path.name not in expected_names
    }
    if unexpected:
        raise KrennN8ToricSupport24CircuitError(
            f"unexpected support-24 artifact files: {sorted(unexpected)}"
        )
    certificate = build_n8_toric_support24_circuit_certificate()
    _write_bytes_atomic(
        directory / CERTIFICATE_FILE,
        _canonical_json_bytes(certificate),
    )
    _write_bytes_atomic(
        directory / README_FILE,
        _readme_text(certificate).encode("ascii"),
    )
    manifest = _expected_manifest(directory, certificate)
    _write_bytes_atomic(
        directory / MANIFEST_FILE,
        _canonical_json_bytes(manifest),
    )
    return certificate


def verify_n8_toric_support24_circuit_bundle(
    directory: Path | str = DEFAULT_RESULTS_DIRECTORY,
) -> dict[str, object]:
    """Replay inventory, hashes, symmetry, and all exact circuits."""

    directory = Path(directory)
    if not directory.is_absolute():
        directory = _repo_root() / directory
    if not directory.is_dir() or directory.is_symlink():
        raise KrennN8ToricSupport24CircuitError(
            "the support-24 bundle directory is absent or linked"
        )
    expected_names = {CERTIFICATE_FILE, README_FILE, MANIFEST_FILE}
    paths = tuple(directory.iterdir())
    if {path.name for path in paths} != expected_names or any(
        path.is_symlink() or not path.is_file() for path in paths
    ):
        raise KrennN8ToricSupport24CircuitError(
            "the support-24 bundle inventory changed"
        )
    certificate = _strict_json(directory / CERTIFICATE_FILE)
    if not isinstance(certificate, Mapping):
        raise KrennN8ToricSupport24CircuitError(
            "the support-24 certificate is not a mapping"
        )
    verified = verify_n8_toric_support24_circuit_certificate(certificate)
    if (directory / README_FILE).read_bytes() != _readme_text(
        verified
    ).encode("ascii"):
        raise KrennN8ToricSupport24CircuitError(
            "the support-24 README changed"
        )
    manifest = _strict_json(directory / MANIFEST_FILE)
    if manifest != _expected_manifest(directory, verified):
        raise KrennN8ToricSupport24CircuitError(
            "the support-24 manifest or source ledger changed"
        )
    return verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exact n=8,d=3 bounded H5 support-24 positive-circuit gate"
        )
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        certificate = verify_n8_toric_support24_circuit_bundle(
            args.results_directory
        )
    else:
        certificate = write_n8_toric_support24_circuit_bundle(
            args.results_directory
        )
        verify_n8_toric_support24_circuit_bundle(args.results_directory)
    print(
        json.dumps(
            {
                "classification": certificate["conclusion"][
                    "classification"
                ],
                "bounded_raw_branches_excluded": certificate[
                    "conclusion"
                ]["bounded_raw_branches_excluded"],
                "bounded_classes_excluded": certificate["conclusion"][
                    "bounded_classes_excluded"
                ],
                "n8_nonexistence_proved": certificate["claim_boundary"][
                    "n8_nonexistence_proved"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = (
    "CIRCUIT_INVENTORY_PATH",
    "DEFAULT_RESULTS_DIRECTORY",
    "KrennN8ToricSupport24CircuitError",
    "build_n8_toric_support24_circuit_certificate",
    "verify_n8_toric_support24_circuit_bundle",
    "verify_n8_toric_support24_circuit_certificate",
    "write_n8_toric_support24_circuit_bundle",
)


if __name__ == "__main__":
    raise SystemExit(main())
