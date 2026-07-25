"""Deterministic SAT gate for Gallagher's common ``n=8,d=3`` support rules.

The gate ports only the five common Boolean support families:

* matching-term support is exactly the AND of its edge supports;
* every monochromatic target has an active term;
* a zero-target coloring cannot have exactly one active term;
* star anchors;
* pair pencils;
* full-column anchors.

No orbit-specific exact-support, Laurent, equation-pattern, or closed-orbit
no-good is included.  Branches are the 31 exact ``S_8 x S_3`` seed-orbit
representatives.  The campaign processes them in canonical order and stops
at the first independently replayed SAT model.  Only 31 independently
checked UNSAT certificates could close the gate globally.

Large DIMACS, solver models, logs, and checkpoints remain under
``D:\\KrennScratch\\counterexample_search``.  A compact physical-edge model
is sufficient for the committed certificate because all term and gate
variables are definitionally determined by the 252 physical support bits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable, Iterator, Mapping, Sequence

from experiments.krenn_quantum_graph.independent_verifier import (
    independent_perfect_matchings,
)
from experiments.krenn_quantum_graph.n8_seed_orbits import (
    EXPECTED_REPRESENTATIVES_SIZES_AND_SINGLETONS,
)
from experiments.krenn_quantum_graph.system import (
    Coloring,
    Matching,
    coloring_from_index,
    coloring_index,
    equation_count,
    perfect_matchings,
    variable_count,
    variable_index,
)


N = 8
D = 3
EDGE_VARIABLE_COUNT = 252
MATCHING_COUNT = 105
COLORING_COUNT = 6_561
TERM_VARIABLE_COUNT = 688_905

# Zero-based offsets matching Gallagher's convention, converted to one-based
# DIMACS IDs only at the public variable constructors.
EDGE_VARIABLE_START = 0
TERM_VARIABLE_START = 252
STAR_VARIABLE_START = 689_157
PAIR_VARIABLE_START = 689_325
FULL_COLUMN_VARIABLE_START = 690_789
TOTAL_VARIABLE_COUNT = 691_461

STAR_VARIABLE_COUNT = 168
PAIR_VARIABLE_COUNT = 1_464
FULL_COLUMN_VARIABLE_COUNT = 672

COMMON_EVENT_COUNT = 1_379_874
COMMON_CLAUSE_COUNT = 4_141_782
COMMON_LITERAL_COUNT = 81_279_582
BRANCH_ANCHOR_COUNT = 12
BRANCH_CLAUSE_COUNT = 4_141_794
BRANCH_LITERAL_COUNT = 81_279_594

FAMILY_CLAUSE_COUNTS = {
    "term_definition": 3_444_525,
    "target_nonempty": 3,
    "non_target_not_singleton": 688_590,
    "star_definition": 672,
    "star_assertion": 24,
    "pair_definition": 5_232,
    "pair_assertion": 24,
    "full_column_definition": 2_688,
    "full_column_assertion": 24,
    "branch_anchor": 12,
}

FAMILY_LITERAL_COUNTS = {
    "term_definition": 8_955_765,
    "target_nonempty": 315,
    "non_target_not_singleton": 72_301_950,
    "star_definition": 1_680,
    "star_assertion": 168,
    "pair_definition": 12_768,
    "pair_assertion": 48,
    "full_column_definition": 6_720,
    "full_column_assertion": 168,
    "branch_anchor": 12,
}

RULE_FAMILIES = (
    "term_definition",
    "target_nonempty",
    "non_target_not_singleton",
    "star_anchor",
    "pair_pencil",
    "full_column_anchor",
)

EXCLUDED_CUT_FAMILIES = (
    "closed_target_orbit",
    "equation_pattern_no_good",
    "exact_support_no_good",
    "generated_branch_tail",
    "laurent_identity_no_good",
)

SCHEMA = "krenn-n8-base-support-cnf-gate-v1"
CNF_RECEIPT_SCHEMA = "krenn-n8-base-support-cnf-receipt-v1"
SOLVER_RECEIPT_SCHEMA = "krenn-n8-base-support-solver-receipt-v1"
CERTIFICATE_SCHEMA = "krenn-n8-base-support-sat-certificate-v1"
MANIFEST_SCHEMA = "krenn-n8-base-support-sat-manifest-v1"
PROGRESS_SCHEMA = "krenn-n8-base-support-cnf-progress-v1"

DEFAULT_SCRATCH = Path(
    r"D:\KrennScratch\counterexample_search\n8_base_support_cnf_v1"
)
DEFAULT_RESULTS = Path(
    "results/krenn_quantum_graph/n8_d3_base_support_cnf_gate"
)
DEFAULT_SOLVER = Path(
    r"D:\KrennScratch\counterexample_search\tools\elan"
    r"\toolchains\leanprover--lean4---v4.27.0\bin\cadical.exe"
)
SOLVER_VERSION = "2.1.2"
SOLVER_SHA256 = (
    "50c056adb8758823c0108a0fc667944948bebd57c0f3728f233ccaa4891591cf"
)
SOLVER_SEED = 0
SOLVER_SAFETY_TIMEOUT_SECONDS = 7_200

CLAIM_BOUNDARY = {
    "all_31_branches_unsat": False,
    "base_cnf_sat_model_is_complex_weight_witness": False,
    "common_support_rules_decide_complex_cancellation": False,
    "external_lrat_checked": False,
    "n8_affine_membership_proved": False,
    "n8_border_membership_proved": False,
    "n8_existence_proved": False,
    "n8_nonexistence_proved": False,
    "orbit_specific_algebraic_cuts_included": False,
    "sat_model_assigns_equal_weights_on_symmetric_coordinates": False,
    "star_pair_full_column_port_kernel_checked_in_lean": False,
    "timeout_or_solver_failure_is_proof": False,
}

SOURCE_FILES = (
    "experiments/krenn_quantum_graph/independent_verifier.py",
    "experiments/krenn_quantum_graph/n8_base_support_cnf.py",
    "experiments/krenn_quantum_graph/n8_seed_orbits.py",
    "experiments/krenn_quantum_graph/system.py",
    "tests/test_krenn_n8_base_support_cnf.py",
)

COLOR_PAIRS = ((0, 1), (0, 2), (1, 2))
TARGET_COLORING_INDICES = tuple(
    coloring_index(N, D, (color,) * N) for color in range(D)
)


class KrennN8BaseSupportCNFError(RuntimeError):
    """The exact CNF, model, solver receipt, or bundle failed replay."""


@dataclass(frozen=True)
class ClauseRecord:
    family: str
    literals: tuple[int, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exact_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KrennN8BaseSupportCNFError(
            f"{label} must be an exact integer"
        )
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _json_exactly_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's ``False == 0`` alias."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if set(left) != set(right):
            return False
        return all(
            type(key) is str
            and _json_exactly_equal(left[key], right[key])
            for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_exactly_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return type(left) in {str, int, bool, type(None)} and left == right


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(payload))
    temporary.replace(path)


def _strict_json(path: Path) -> object:
    def pairs(values: Iterable[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise KrennN8BaseSupportCNFError(
                    f"{path.name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_float(value: str) -> object:
        raise KrennN8BaseSupportCNFError(
            f"{path.name} contains non-integer number {value}"
        )

    try:
        return json.loads(
            path.read_text("utf-8"),
            object_pairs_hook=pairs,
            parse_float=reject_float,
            parse_constant=reject_float,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KrennN8BaseSupportCNFError(
            f"could not read strict JSON from {path}"
        ) from error


@lru_cache(maxsize=1)
def _primary_matchings() -> tuple[Matching, ...]:
    result = perfect_matchings(N)
    if len(result) != MATCHING_COUNT:
        raise KrennN8BaseSupportCNFError(
            "the primary K8 matching census changed"
        )
    return result


@lru_cache(maxsize=1)
def _independent_matchings() -> tuple[Matching, ...]:
    result = independent_perfect_matchings(N)
    if (
        len(result) != MATCHING_COUNT
        or set(result) != set(_primary_matchings())
    ):
        raise KrennN8BaseSupportCNFError(
            "the independent K8 matching census disagrees"
        )
    return result


def edge_variable(
    root: int,
    neighbor: int,
    root_color: int,
    neighbor_color: int,
) -> int:
    """Return a one-based physical edge-support variable."""

    result = (
        variable_index(
            N,
            D,
            root,
            neighbor,
            root_color,
            neighbor_color,
        )
        + 1
    )
    if not 1 <= result <= EDGE_VARIABLE_COUNT:
        raise KrennN8BaseSupportCNFError(
            "an edge variable escaped its DIMACS block"
        )
    return result


def term_variable(coloring_index_value: int, matching_index: int) -> int:
    coloring_index_value = _exact_integer(
        coloring_index_value, "coloring index"
    )
    matching_index = _exact_integer(
        matching_index, "matching index"
    )
    if (
        not 0 <= coloring_index_value < COLORING_COUNT
        or not 0 <= matching_index < MATCHING_COUNT
    ):
        raise KrennN8BaseSupportCNFError(
            "a term variable index is outside its block"
        )
    return (
        TERM_VARIABLE_START
        + coloring_index_value * MATCHING_COUNT
        + matching_index
        + 1
    )


def _neighbor_rank(root: int, neighbor: int) -> int:
    if not 0 <= root < N or not 0 <= neighbor < N or root == neighbor:
        raise KrennN8BaseSupportCNFError(
            "neighbor rank needs two distinct K8 vertices"
        )
    return neighbor if neighbor < root else neighbor - 1


def _other_neighbors(root: int) -> tuple[int, ...]:
    return tuple(vertex for vertex in range(N) if vertex != root)


def star_variable(root: int, color: int, neighbor: int) -> int:
    return (
        STAR_VARIABLE_START
        + (root * D + color) * (N - 1)
        + _neighbor_rank(root, neighbor)
        + 1
    )


def _pair_block_start(root: int, pair_index: int) -> int:
    block_size = 2 * (N - 1) * 4 + 5
    return PAIR_VARIABLE_START + (root * len(COLOR_PAIRS) + pair_index) * (
        block_size
    )


def pair_cell_start(
    root: int,
    pair_index: int,
    output_slot: int,
    neighbor: int,
) -> int:
    return (
        _pair_block_start(root, pair_index)
        + (output_slot * (N - 1) + _neighbor_rank(root, neighbor)) * 4
        + 1
    )


def pair_has_pure_variable(
    root: int, pair_index: int, output_slot: int
) -> int:
    return (
        _pair_block_start(root, pair_index)
        + 2 * (N - 1) * 4
        + output_slot
        + 1
    )


def pair_both_pure_variable(root: int, pair_index: int) -> int:
    return (
        _pair_block_start(root, pair_index)
        + 2 * (N - 1) * 4
        + 2
        + 1
    )


def pair_leaves_variable(root: int, pair_index: int) -> int:
    return (
        _pair_block_start(root, pair_index)
        + 2 * (N - 1) * 4
        + 3
        + 1
    )


def pair_preserved_variable(root: int, pair_index: int) -> int:
    return (
        _pair_block_start(root, pair_index)
        + 2 * (N - 1) * 4
        + 4
        + 1
    )


def full_column_cell_start(
    root: int, color: int, neighbor: int
) -> int:
    return (
        FULL_COLUMN_VARIABLE_START
        + ((root * D + color) * (N - 1) + _neighbor_rank(root, neighbor))
        * 4
        + 1
    )


def _and_definition(
    family: str, output: int, inputs: Sequence[int]
) -> Iterator[ClauseRecord]:
    checked = tuple(inputs)
    for literal in checked:
        yield ClauseRecord(family, (-output, literal))
    yield ClauseRecord(
        family,
        (output, *tuple(-literal for literal in checked)),
    )


def _or_definition(
    family: str, output: int, inputs: Sequence[int]
) -> Iterator[ClauseRecord]:
    checked = tuple(inputs)
    for literal in checked:
        yield ClauseRecord(family, (output, -literal))
    yield ClauseRecord(family, (-output, *checked))


def _not_definition(
    family: str, output: int, input_variable: int
) -> Iterator[ClauseRecord]:
    yield ClauseRecord(family, (-output, -input_variable))
    yield ClauseRecord(family, (output, input_variable))


def _term_edge_variables(
    coloring: Coloring, matching: Matching
) -> tuple[int, ...]:
    return tuple(
        edge_variable(
            first,
            second,
            coloring[first],
            coloring[second],
        )
        for first, second in matching
    )


def iter_common_clause_records() -> Iterator[ClauseRecord]:
    """Yield the exact common prefix in deterministic Gallagher order."""

    matchings = _primary_matchings()
    target_indices = set(TARGET_COLORING_INDICES)

    for coloring_index_value in range(COLORING_COUNT):
        coloring = coloring_from_index(N, D, coloring_index_value)
        for matching_index, matching in enumerate(matchings):
            yield from _and_definition(
                "term_definition",
                term_variable(coloring_index_value, matching_index),
                _term_edge_variables(coloring, matching),
            )

    for coloring_index_value in range(COLORING_COUNT):
        terms = tuple(
            term_variable(coloring_index_value, matching_index)
            for matching_index in range(MATCHING_COUNT)
        )
        if coloring_index_value in target_indices:
            yield ClauseRecord("target_nonempty", terms)
        else:
            for matching_index, selected in enumerate(terms):
                yield ClauseRecord(
                    "non_target_not_singleton",
                    (
                        -selected,
                        *terms[:matching_index],
                        *terms[matching_index + 1 :],
                    ),
                )

    for root in range(N):
        for color in range(D):
            for neighbor in _other_neighbors(root):
                inputs = (
                    edge_variable(root, neighbor, color, color),
                    *tuple(
                        -edge_variable(
                            root, neighbor, color, other
                        )
                        for other in range(D)
                        if other != color
                    ),
                )
                yield from _and_definition(
                    "star_definition",
                    star_variable(root, color, neighbor),
                    inputs,
                )
            yield ClauseRecord(
                "star_assertion",
                tuple(
                    star_variable(root, color, neighbor)
                    for neighbor in _other_neighbors(root)
                ),
            )

    for root in range(N):
        for pair_index, (color_a, color_b) in enumerate(COLOR_PAIRS):
            outputs = (color_a, color_b)
            for output_slot, output in enumerate(outputs):
                for neighbor in _other_neighbors(root):
                    cell = pair_cell_start(
                        root, pair_index, output_slot, neighbor
                    )
                    target_edges = tuple(
                        edge_variable(
                            root,
                            neighbor,
                            root_color,
                            output,
                        )
                        for root_color in (color_a, color_b)
                    )
                    contaminated_edges = tuple(
                        edge_variable(
                            root,
                            neighbor,
                            root_color,
                            other,
                        )
                        for root_color in (color_a, color_b)
                        for other in range(D)
                        if other != output
                    )
                    yield from _or_definition(
                        "pair_definition", cell, target_edges
                    )
                    yield from _or_definition(
                        "pair_definition",
                        cell + 1,
                        contaminated_edges,
                    )
                    yield from _not_definition(
                        "pair_definition", cell + 2, cell + 1
                    )
                    yield from _and_definition(
                        "pair_definition",
                        cell + 3,
                        (cell, cell + 2),
                    )
            for output_slot in range(2):
                yield from _or_definition(
                    "pair_definition",
                    pair_has_pure_variable(
                        root, pair_index, output_slot
                    ),
                    tuple(
                        pair_cell_start(
                            root,
                            pair_index,
                            output_slot,
                            neighbor,
                        )
                        + 3
                        for neighbor in _other_neighbors(root)
                    ),
                )
            yield from _and_definition(
                "pair_definition",
                pair_both_pure_variable(root, pair_index),
                (
                    pair_has_pure_variable(root, pair_index, 0),
                    pair_has_pure_variable(root, pair_index, 1),
                ),
            )
            outside_colors = tuple(
                color
                for color in range(D)
                if color not in (color_a, color_b)
            )
            yield from _or_definition(
                "pair_definition",
                pair_leaves_variable(root, pair_index),
                tuple(
                    edge_variable(
                        root,
                        neighbor,
                        root_color,
                        outside,
                    )
                    for neighbor in _other_neighbors(root)
                    for root_color in (color_a, color_b)
                    for outside in outside_colors
                ),
            )
            yield from _not_definition(
                "pair_definition",
                pair_preserved_variable(root, pair_index),
                pair_leaves_variable(root, pair_index),
            )
            yield ClauseRecord(
                "pair_assertion",
                (
                    pair_both_pure_variable(root, pair_index),
                    pair_preserved_variable(root, pair_index),
                ),
            )

    for root in range(N):
        for color in range(D):
            for neighbor in _other_neighbors(root):
                cell = full_column_cell_start(
                    root, color, neighbor
                )
                column_edges = tuple(
                    edge_variable(
                        root, neighbor, root_color, color
                    )
                    for root_color in range(D)
                )
                contaminated_edges = tuple(
                    edge_variable(
                        root, neighbor, root_color, other
                    )
                    for root_color in range(D)
                    for other in range(D)
                    if other != color
                )
                yield from _or_definition(
                    "full_column_definition", cell, column_edges
                )
                yield from _or_definition(
                    "full_column_definition",
                    cell + 1,
                    contaminated_edges,
                )
                yield from _not_definition(
                    "full_column_definition", cell + 2, cell + 1
                )
                yield from _and_definition(
                    "full_column_definition",
                    cell + 3,
                    (cell, cell + 2),
                )
            yield ClauseRecord(
                "full_column_assertion",
                tuple(
                    full_column_cell_start(
                        root, color, neighbor
                    )
                    + 3
                    for neighbor in _other_neighbors(root)
                ),
            )


@lru_cache(maxsize=1)
def branch_representatives() -> tuple[tuple[int, int, int], ...]:
    result = tuple(
        representative
        for representative, _orbit_size, _victims
        in EXPECTED_REPRESENTATIVES_SIZES_AND_SINGLETONS
    )
    if len(result) != 31 or len(set(result)) != 31:
        raise KrennN8BaseSupportCNFError(
            "the exact n8 branch representative census changed"
        )
    return result


def branch_anchor_variables(branch_index: int) -> tuple[int, ...]:
    branch_index = _exact_integer(branch_index, "branch index")
    if not 0 <= branch_index < len(branch_representatives()):
        raise KrennN8BaseSupportCNFError(
            "branch index is outside range(31)"
        )
    representative = branch_representatives()[branch_index]
    anchors = tuple(
        sorted(
            edge_variable(
                first, second, color, color
            )
            for color, matching_index in enumerate(representative)
            for first, second in _primary_matchings()[matching_index]
        )
    )
    if len(anchors) != BRANCH_ANCHOR_COUNT or len(set(anchors)) != len(
        anchors
    ):
        raise KrennN8BaseSupportCNFError(
            "a branch does not have twelve distinct colored anchors"
        )
    return anchors


def iter_branch_clause_records(
    branch_index: int,
) -> Iterator[ClauseRecord]:
    yield from iter_common_clause_records()
    for variable in branch_anchor_variables(branch_index):
        yield ClauseRecord("branch_anchor", (variable,))


def _matching_table_hash(matchings: Sequence[Matching]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            [
                [list(edge) for edge in matching]
                for matching in matchings
            ]
        )
    ).hexdigest()


def _branch_table_hash() -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            [
                {
                    "branch_index": branch_index,
                    "representative_matching_indices": list(
                        representative
                    ),
                    "anchor_variables": list(
                        branch_anchor_variables(branch_index)
                    ),
                }
                for branch_index, representative in enumerate(
                    branch_representatives()
                )
            ]
        )
    ).hexdigest()


def formula_specification() -> dict[str, object]:
    primary = _primary_matchings()
    independent = _independent_matchings()
    return {
        "schema": SCHEMA,
        "parameters": {"n": N, "d": D},
        "variable_blocks": {
            "edge": {
                "one_based_start": 1,
                "count": EDGE_VARIABLE_COUNT,
            },
            "term": {
                "one_based_start": TERM_VARIABLE_START + 1,
                "count": TERM_VARIABLE_COUNT,
            },
            "star": {
                "one_based_start": STAR_VARIABLE_START + 1,
                "count": STAR_VARIABLE_COUNT,
            },
            "pair": {
                "one_based_start": PAIR_VARIABLE_START + 1,
                "count": PAIR_VARIABLE_COUNT,
            },
            "full_column": {
                "one_based_start": FULL_COLUMN_VARIABLE_START + 1,
                "count": FULL_COLUMN_VARIABLE_COUNT,
            },
        },
        "counts": {
            "variables": TOTAL_VARIABLE_COUNT,
            "common_events": COMMON_EVENT_COUNT,
            "common_clauses": COMMON_CLAUSE_COUNT,
            "common_literals": COMMON_LITERAL_COUNT,
            "branch_anchor_units": BRANCH_ANCHOR_COUNT,
            "branch_clauses": BRANCH_CLAUSE_COUNT,
            "branch_literals": BRANCH_LITERAL_COUNT,
            "colorings": COLORING_COUNT,
            "perfect_matchings_per_coloring": MATCHING_COUNT,
            "matching_terms": TERM_VARIABLE_COUNT,
            "target_colorings": len(TARGET_COLORING_INDICES),
            "non_target_colorings": (
                COLORING_COUNT - len(TARGET_COLORING_INDICES)
            ),
            "branches": len(branch_representatives()),
        },
        "family_clause_counts": FAMILY_CLAUSE_COUNTS,
        "family_literal_counts": FAMILY_LITERAL_COUNTS,
        "rule_families": list(RULE_FAMILIES),
        "excluded_cut_families": list(EXCLUDED_CUT_FAMILIES),
        "target_coloring_indices": list(TARGET_COLORING_INDICES),
        "matching_tables": {
            "primary_sha256": _matching_table_hash(primary),
            "independent_sha256": _matching_table_hash(independent),
            "sets_equal": set(primary) == set(independent),
        },
        "branch_table_sha256": _branch_table_hash(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _validate_family_totals(
    family_clauses: Mapping[str, int],
    family_literals: Mapping[str, int],
) -> None:
    if (
        dict(family_clauses) != FAMILY_CLAUSE_COUNTS
        or dict(family_literals) != FAMILY_LITERAL_COUNTS
        or sum(family_clauses.values()) != BRANCH_CLAUSE_COUNT
        or sum(family_literals.values()) != BRANCH_LITERAL_COUNT
    ):
        raise KrennN8BaseSupportCNFError(
            "the exact clause-family totals changed"
        )


def write_dimacs(
    path: Path,
    *,
    branch_index: int,
    progress_path: Path | None = None,
) -> dict[str, object]:
    """Atomically stream one exact branch DIMACS file."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    hasher = hashlib.sha256()
    family_clauses: Counter[str] = Counter()
    family_literals: Counter[str] = Counter()
    started_at = _utc_now()
    header = (
        f"p cnf {TOTAL_VARIABLE_COUNT} {BRANCH_CLAUSE_COUNT}\n"
    ).encode("ascii")
    hasher.update(header)
    with temporary.open("wb", buffering=8 * 1024 * 1024) as handle:
        handle.write(header)
        for clause_index, record in enumerate(
            iter_branch_clause_records(branch_index), start=1
        ):
            if (
                not record.literals
                or any(
                    not literal
                    or abs(literal) > TOTAL_VARIABLE_COUNT
                    for literal in record.literals
                )
            ):
                raise KrennN8BaseSupportCNFError(
                    "a generated clause has an invalid literal"
                )
            line = (
                " ".join(str(literal) for literal in record.literals)
                + " 0\n"
            ).encode("ascii")
            handle.write(line)
            hasher.update(line)
            family_clauses[record.family] += 1
            family_literals[record.family] += len(record.literals)
            if progress_path is not None and clause_index % 250_000 == 0:
                _atomic_json(
                    progress_path,
                    {
                        "schema": PROGRESS_SCHEMA,
                        "stage": "generating_dimacs",
                        "branch_index": branch_index,
                        "clauses_written": clause_index,
                        "expected_clauses": BRANCH_CLAUSE_COUNT,
                        "started_at": started_at,
                        "updated_at": _utc_now(),
                        "complete": False,
                    },
                )
        handle.flush()
        os.fsync(handle.fileno())
    _validate_family_totals(family_clauses, family_literals)
    temporary.replace(path)
    receipt = {
        "schema": CNF_RECEIPT_SCHEMA,
        "branch_index": branch_index,
        "representative_matching_indices": list(
            branch_representatives()[branch_index]
        ),
        "anchor_variables": list(
            branch_anchor_variables(branch_index)
        ),
        "variables": TOTAL_VARIABLE_COUNT,
        "clauses": BRANCH_CLAUSE_COUNT,
        "literals": BRANCH_LITERAL_COUNT,
        "family_clause_counts": dict(family_clauses),
        "family_literal_counts": dict(family_literals),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "complete": True,
    }
    if progress_path is not None:
        _atomic_json(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "stage": "dimacs_complete",
                "branch_index": branch_index,
                "clauses_written": BRANCH_CLAUSE_COUNT,
                "expected_clauses": BRANCH_CLAUSE_COUNT,
                "started_at": started_at,
                "updated_at": _utc_now(),
                "complete": False,
            },
        )
    return receipt


def _parse_solver_model(
    path: Path,
) -> tuple[bool, ...]:
    status = None
    assignment: dict[int, bool] = {}
    for raw_line in path.read_text("ascii").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("s "):
            if status is not None:
                raise KrennN8BaseSupportCNFError(
                    "solver model contains multiple status lines"
                )
            status = line
            continue
        if not line.startswith("v "):
            raise KrennN8BaseSupportCNFError(
                "solver model contains an unknown record"
            )
        for token in line[2:].split():
            literal = int(token)
            if literal == 0:
                continue
            variable = abs(literal)
            if not 1 <= variable <= TOTAL_VARIABLE_COUNT:
                raise KrennN8BaseSupportCNFError(
                    "solver model variable is outside the CNF"
                )
            value = literal > 0
            if variable in assignment and assignment[variable] != value:
                raise KrennN8BaseSupportCNFError(
                    "solver model assigns a variable both ways"
                )
            assignment[variable] = value
    if status != "s SATISFIABLE":
        raise KrennN8BaseSupportCNFError(
            "solver model is not a SATISFIABLE competition witness"
        )
    if set(assignment) != set(range(1, TOTAL_VARIABLE_COUNT + 1)):
        missing = TOTAL_VARIABLE_COUNT - len(assignment)
        raise KrennN8BaseSupportCNFError(
            f"solver model is not complete ({missing} variables missing)"
        )
    return tuple(
        [False]
        + [
            assignment[variable]
            for variable in range(1, TOTAL_VARIABLE_COUNT + 1)
        ]
    )


def _set_output(
    assignment: list[bool], variable: int, value: bool
) -> None:
    if assignment[variable] is not False and assignment[variable] != value:
        raise KrennN8BaseSupportCNFError(
            "a derived gate variable was assigned inconsistently"
        )
    assignment[variable] = value


def derive_assignment_and_semantic_report(
    active_edge_variables: Sequence[int],
    *,
    branch_index: int,
) -> tuple[tuple[bool, ...], dict[str, object]]:
    """Derive every auxiliary and replay all semantic assertions."""

    checked_active = tuple(
        sorted(
            _exact_integer(variable, "active edge variable")
            for variable in active_edge_variables
        )
    )
    if (
        len(checked_active) != len(set(checked_active))
        or any(
            variable < 1 or variable > EDGE_VARIABLE_COUNT
            for variable in checked_active
        )
    ):
        raise KrennN8BaseSupportCNFError(
            "active edge variables are not a valid physical support"
        )
    assignment = [False] * (TOTAL_VARIABLE_COUNT + 1)
    for variable in checked_active:
        assignment[variable] = True

    primary_counts = []
    primary_active_total = 0
    for coloring_index_value in range(COLORING_COUNT):
        coloring = coloring_from_index(N, D, coloring_index_value)
        active_count = 0
        for matching_index, matching in enumerate(_primary_matchings()):
            value = all(
                assignment[edge]
                for edge in _term_edge_variables(coloring, matching)
            )
            assignment[
                term_variable(coloring_index_value, matching_index)
            ] = value
            active_count += int(value)
        primary_counts.append(active_count)
        primary_active_total += active_count

    independent_counts = []
    for coloring_index_value in range(COLORING_COUNT):
        coloring = coloring_from_index(N, D, coloring_index_value)
        independent_counts.append(
            sum(
                all(
                    assignment[edge]
                    for edge in _term_edge_variables(
                        coloring, matching
                    )
                )
                for matching in _independent_matchings()
            )
        )
    if tuple(primary_counts) != tuple(independent_counts):
        raise KrennN8BaseSupportCNFError(
            "primary and independent active-term counts disagree"
        )

    star_assertions = []
    for root in range(N):
        for color in range(D):
            values = []
            for neighbor in _other_neighbors(root):
                value = assignment[
                    edge_variable(root, neighbor, color, color)
                ] and all(
                    not assignment[
                        edge_variable(
                            root, neighbor, color, other
                        )
                    ]
                    for other in range(D)
                    if other != color
                )
                variable = star_variable(root, color, neighbor)
                assignment[variable] = value
                values.append(value)
            star_assertions.append(any(values))

    pair_assertions = []
    for root in range(N):
        for pair_index, (color_a, color_b) in enumerate(COLOR_PAIRS):
            outputs = (color_a, color_b)
            for output_slot, output in enumerate(outputs):
                for neighbor in _other_neighbors(root):
                    cell = pair_cell_start(
                        root, pair_index, output_slot, neighbor
                    )
                    target = any(
                        assignment[
                            edge_variable(
                                root,
                                neighbor,
                                root_color,
                                output,
                            )
                        ]
                        for root_color in (color_a, color_b)
                    )
                    contaminated = any(
                        assignment[
                            edge_variable(
                                root,
                                neighbor,
                                root_color,
                                other,
                            )
                        ]
                        for root_color in (color_a, color_b)
                        for other in range(D)
                        if other != output
                    )
                    assignment[cell] = target
                    assignment[cell + 1] = contaminated
                    assignment[cell + 2] = not contaminated
                    assignment[cell + 3] = target and not contaminated
            has_pure_values = []
            for output_slot in range(2):
                value = any(
                    assignment[
                        pair_cell_start(
                            root,
                            pair_index,
                            output_slot,
                            neighbor,
                        )
                        + 3
                    ]
                    for neighbor in _other_neighbors(root)
                )
                assignment[
                    pair_has_pure_variable(
                        root, pair_index, output_slot
                    )
                ] = value
                has_pure_values.append(value)
            both_pure = all(has_pure_values)
            assignment[
                pair_both_pure_variable(root, pair_index)
            ] = both_pure
            outside_colors = tuple(
                color
                for color in range(D)
                if color not in (color_a, color_b)
            )
            leaves = any(
                assignment[
                    edge_variable(
                        root,
                        neighbor,
                        root_color,
                        outside,
                    )
                ]
                for neighbor in _other_neighbors(root)
                for root_color in (color_a, color_b)
                for outside in outside_colors
            )
            assignment[pair_leaves_variable(root, pair_index)] = leaves
            assignment[
                pair_preserved_variable(root, pair_index)
            ] = not leaves
            pair_assertions.append(both_pure or not leaves)

    full_column_assertions = []
    for root in range(N):
        for color in range(D):
            values = []
            for neighbor in _other_neighbors(root):
                cell = full_column_cell_start(
                    root, color, neighbor
                )
                column = any(
                    assignment[
                        edge_variable(
                            root, neighbor, root_color, color
                        )
                    ]
                    for root_color in range(D)
                )
                contaminated = any(
                    assignment[
                        edge_variable(
                            root, neighbor, root_color, other
                        )
                    ]
                    for root_color in range(D)
                    for other in range(D)
                    if other != color
                )
                assignment[cell] = column
                assignment[cell + 1] = contaminated
                assignment[cell + 2] = not contaminated
                assignment[cell + 3] = column and not contaminated
                values.append(column and not contaminated)
            full_column_assertions.append(any(values))

    target_counts = {
        str(index): primary_counts[index]
        for index in TARGET_COLORING_INDICES
    }
    non_target_counts = [
        count
        for index, count in enumerate(primary_counts)
        if index not in set(TARGET_COLORING_INDICES)
    ]
    branch_satisfaction = tuple(
        all(
            assignment[variable]
            for variable in branch_anchor_variables(candidate)
        )
        for candidate in range(len(branch_representatives()))
    )
    semantic_checks = {
        "target_nonempty": all(count > 0 for count in target_counts.values()),
        "non_target_not_singleton": all(
            count != 1 for count in non_target_counts
        ),
        "star_assertions": all(star_assertions),
        "pair_assertions": all(pair_assertions),
        "full_column_assertions": all(full_column_assertions),
        "selected_branch_anchors": branch_satisfaction[branch_index],
        "primary_independent_matching_replay": True,
    }
    if not all(semantic_checks.values()):
        failed = tuple(
            key for key, value in semantic_checks.items() if not value
        )
        raise KrennN8BaseSupportCNFError(
            f"the physical support fails semantic checks {failed}"
        )
    report = {
        "active_edge_variable_count": len(checked_active),
        "active_term_variable_count": primary_active_total,
        "target_active_term_counts": target_counts,
        "non_target_active_term_count_histogram": {
            str(count): frequency
            for count, frequency in sorted(
                Counter(non_target_counts).items()
            )
        },
        "non_target_singleton_count": sum(
            count == 1 for count in non_target_counts
        ),
        "star_assertions_true": sum(star_assertions),
        "star_assertions_total": len(star_assertions),
        "pair_assertions_true": sum(pair_assertions),
        "pair_assertions_total": len(pair_assertions),
        "full_column_assertions_true": sum(
            full_column_assertions
        ),
        "full_column_assertions_total": len(
            full_column_assertions
        ),
        "branches_satisfied_by_same_model": [
            index
            for index, satisfied in enumerate(branch_satisfaction)
            if satisfied
        ],
        "semantic_checks": semantic_checks,
    }
    return tuple(assignment), report


def verify_dimacs_against_generator_and_model(
    path: Path,
    *,
    branch_index: int,
    assignment: Sequence[bool],
) -> dict[str, object]:
    """Independently parse every line, compare every clause, and evaluate."""

    if len(assignment) != TOTAL_VARIABLE_COUNT + 1:
        raise KrennN8BaseSupportCNFError(
            "model assignment has the wrong length"
        )
    hasher = hashlib.sha256()
    family_clauses: Counter[str] = Counter()
    family_literals: Counter[str] = Counter()
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        header = handle.readline()
        hasher.update(header)
        expected_header = (
            f"p cnf {TOTAL_VARIABLE_COUNT} {BRANCH_CLAUSE_COUNT}\n"
        ).encode("ascii")
        if header != expected_header:
            raise KrennN8BaseSupportCNFError(
                "DIMACS header does not match the exact formula"
            )
        for clause_index, expected in enumerate(
            iter_branch_clause_records(branch_index), start=1
        ):
            line = handle.readline()
            if not line:
                raise KrennN8BaseSupportCNFError(
                    "DIMACS ended before the generated formula"
                )
            hasher.update(line)
            tokens = line.split()
            if not tokens or tokens[-1] != b"0":
                raise KrennN8BaseSupportCNFError(
                    "DIMACS clause lacks a zero terminator"
                )
            observed = tuple(int(token) for token in tokens[:-1])
            if observed != expected.literals:
                raise KrennN8BaseSupportCNFError(
                    f"DIMACS clause {clause_index} differs from generator"
                )
            if not any(
                assignment[abs(literal)] == (literal > 0)
                for literal in observed
            ):
                raise KrennN8BaseSupportCNFError(
                    f"model falsifies DIMACS clause {clause_index}"
                )
            family_clauses[expected.family] += 1
            family_literals[expected.family] += len(observed)
        if handle.read(1):
            raise KrennN8BaseSupportCNFError(
                "DIMACS has trailing data after the exact formula"
            )
    _validate_family_totals(family_clauses, family_literals)
    return {
        "variables": TOTAL_VARIABLE_COUNT,
        "clauses": sum(family_clauses.values()),
        "literals": sum(family_literals.values()),
        "family_clause_counts": dict(family_clauses),
        "family_literal_counts": dict(family_literals),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
        "all_clauses_match_generator": True,
        "all_clauses_satisfied": True,
    }


def _validate_solver(solver: Path) -> dict[str, object]:
    solver = solver.resolve()
    if not solver.is_file():
        raise KrennN8BaseSupportCNFError(
            "the pinned CaDiCaL executable is missing"
        )
    record = _file_record(solver)
    if record["sha256"].lower() != SOLVER_SHA256:
        raise KrennN8BaseSupportCNFError(
            "the pinned CaDiCaL executable hash changed"
        )
    completed = subprocess.run(
        [str(solver), "--version"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )
    if completed.returncode != 0 or completed.stdout.strip() != SOLVER_VERSION:
        raise KrennN8BaseSupportCNFError(
            "the pinned CaDiCaL version changed"
        )
    return {
        "path": str(solver),
        "version": SOLVER_VERSION,
        **record,
    }


def run_solver(
    cnf_path: Path,
    model_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    *,
    solver: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    solver_record = _validate_solver(solver)
    command = [
        str(solver.resolve()),
        "-q",
        f"--seed={SOLVER_SEED}",
        "--check=true",
        "-w",
        str(model_path.resolve()),
        str(cnf_path.resolve()),
    ]
    started_at = _utc_now()
    environment = dict(os.environ)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open(
            "wb"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                env=environment,
                timeout=timeout_seconds,
            )
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        completed = None
    receipt = {
        "schema": SOLVER_RECEIPT_SCHEMA,
        "command": command,
        "solver": solver_record,
        "seed": SOLVER_SEED,
        "single_process": True,
        "timeout_seconds_is_safety_guard_not_search_claim": True,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "stdout": _file_record(stdout_path),
        "stderr": _file_record(stderr_path),
        "return_code": (
            None if completed is None else completed.returncode
        ),
        "status": (
            "timeout"
            if timed_out
            else "sat"
            if completed.returncode == 10
            else "unsat"
            if completed.returncode == 20
            else "failure"
        ),
    }
    if model_path.exists():
        receipt["model"] = _file_record(model_path)
    return receipt


def _false_claims(payload: object) -> bool:
    return (
        type(payload) is dict
        and payload == CLAIM_BOUNDARY
        and all(value is False for value in payload.values())
    )


def _valid_hash_record(payload: object, *, allow_zero_bytes: bool) -> bool:
    return (
        type(payload) is dict
        and set(payload) == {"bytes", "sha256"}
        and type(payload["bytes"]) is int
        and payload["bytes"] >= int(not allow_zero_bytes)
        and type(payload["sha256"]) is str
        and len(payload["sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in payload["sha256"]
        )
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_certificate(
    *,
    branch_index: int,
    active_edges: Sequence[int],
    semantic_report: Mapping[str, object],
    cnf_receipt: Mapping[str, object],
    solver_receipt: Mapping[str, object],
    dimacs_replay: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": CERTIFICATE_SCHEMA,
        "status": "sat-common-prefix-insufficient",
        "parameters": {"n": N, "d": D},
        "formula_specification": formula_specification(),
        "terminal_branch": {
            "branch_index": branch_index,
            "representative_matching_indices": list(
                branch_representatives()[branch_index]
            ),
            "anchor_variables": list(
                branch_anchor_variables(branch_index)
            ),
        },
        "compact_model": {
            "active_edge_variables": list(active_edges),
            "all_other_edge_variables_false": True,
            "all_auxiliaries_definitionally_reconstructed": True,
        },
        "semantic_replay": dict(semantic_report),
        "dimacs_replay": dict(dimacs_replay),
        "cnf_receipt": dict(cnf_receipt),
        "solver_receipt": dict(solver_receipt),
        "terminal_logic": {
            "stop_on_first_independently_replayed_sat_model": True,
            "branches_attempted": [branch_index],
            "remaining_branches_not_run": list(
                range(branch_index + 1, len(branch_representatives()))
            ),
            "conclusion": (
                "Gallagher's common support prefix alone cannot refute "
                "all 31 n=8 seed branches."
            ),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _build_manifest(results_directory: Path) -> dict[str, object]:
    root = _repo_root()
    return {
        "schema": MANIFEST_SCHEMA,
        "bundle_files": {
            filename: _file_record(results_directory / filename)
            for filename in ("README.md", "certificate.json")
        },
        "source_files": {
            filename: _file_record(root / filename)
            for filename in SOURCE_FILES
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def verify_compact_certificate(certificate: Mapping[str, object]) -> None:
    if (
        type(certificate) is not dict
        or set(certificate)
        != {
            "schema",
            "status",
            "parameters",
            "formula_specification",
            "terminal_branch",
            "compact_model",
            "semantic_replay",
            "dimacs_replay",
            "cnf_receipt",
            "solver_receipt",
            "terminal_logic",
            "claim_boundary",
        }
        or certificate.get("schema") != CERTIFICATE_SCHEMA
        or certificate.get("status")
        != "sat-common-prefix-insufficient"
        or not _json_exactly_equal(
            certificate.get("parameters"), {"n": N, "d": D}
        )
        or not _false_claims(certificate.get("claim_boundary"))
        or not _json_exactly_equal(
            certificate.get("formula_specification"),
            formula_specification(),
        )
    ):
        raise KrennN8BaseSupportCNFError(
            "compact certificate schema, status, or formula changed"
        )
    branch = certificate.get("terminal_branch")
    model = certificate.get("compact_model")
    if type(branch) is not dict or type(model) is not dict:
        raise KrennN8BaseSupportCNFError(
            "compact certificate lacks branch or model records"
        )
    branch_index = _exact_integer(
        branch.get("branch_index"), "certificate branch index"
    )
    if not 0 <= branch_index < len(branch_representatives()):
        raise KrennN8BaseSupportCNFError(
            "compact certificate branch index is outside range(31)"
        )
    if not _json_exactly_equal(
        branch,
        {
            "branch_index": branch_index,
            "representative_matching_indices": list(
                branch_representatives()[branch_index]
            ),
            "anchor_variables": list(
                branch_anchor_variables(branch_index)
            ),
        },
    ):
        raise KrennN8BaseSupportCNFError(
            "compact certificate branch record changed"
        )
    if (
        set(model)
        != {
            "active_edge_variables",
            "all_other_edge_variables_false",
            "all_auxiliaries_definitionally_reconstructed",
        }
        or model["all_other_edge_variables_false"] is not True
        or model["all_auxiliaries_definitionally_reconstructed"] is not True
    ):
        raise KrennN8BaseSupportCNFError(
            "compact model semantics changed"
        )
    _assignment, replay = derive_assignment_and_semantic_report(
        model["active_edge_variables"],
        branch_index=branch_index,
    )
    if not _json_exactly_equal(
        replay, certificate.get("semantic_replay")
    ):
        raise KrennN8BaseSupportCNFError(
            "compact semantic replay differs from certificate"
        )
    dimacs = certificate.get("dimacs_replay")
    cnf = certificate.get("cnf_receipt")
    solver = certificate.get("solver_receipt")
    terminal_logic = certificate.get("terminal_logic")
    if (
        type(dimacs) is not dict
        or not _json_exactly_equal(
            dimacs,
            {
                "variables": TOTAL_VARIABLE_COUNT,
                "clauses": BRANCH_CLAUSE_COUNT,
                "literals": BRANCH_LITERAL_COUNT,
                "family_clause_counts": FAMILY_CLAUSE_COUNTS,
                "family_literal_counts": FAMILY_LITERAL_COUNTS,
                "bytes": dimacs.get("bytes"),
                "sha256": dimacs.get("sha256"),
                "all_clauses_match_generator": True,
                "all_clauses_satisfied": True,
            },
        )
        or type(dimacs.get("bytes")) is not int
        or dimacs["bytes"] <= 0
        or type(dimacs.get("sha256")) is not str
        or len(dimacs["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in dimacs["sha256"]
        )
        or type(cnf) is not dict
        or set(cnf)
        != {
            "schema",
            "branch_index",
            "representative_matching_indices",
            "anchor_variables",
            "variables",
            "clauses",
            "literals",
            "family_clause_counts",
            "family_literal_counts",
            "bytes",
            "sha256",
            "started_at",
            "completed_at",
            "complete",
        }
        or cnf.get("schema") != CNF_RECEIPT_SCHEMA
        or not _json_exactly_equal(
            cnf.get("branch_index"), branch_index
        )
        or not _json_exactly_equal(
            cnf.get("representative_matching_indices"),
            branch["representative_matching_indices"],
        )
        or not _json_exactly_equal(
            cnf.get("anchor_variables"),
            branch["anchor_variables"],
        )
        or cnf.get("variables") != TOTAL_VARIABLE_COUNT
        or cnf.get("clauses") != BRANCH_CLAUSE_COUNT
        or cnf.get("literals") != BRANCH_LITERAL_COUNT
        or cnf.get("family_clause_counts") != FAMILY_CLAUSE_COUNTS
        or cnf.get("family_literal_counts") != FAMILY_LITERAL_COUNTS
        or cnf.get("complete") is not True
        or type(cnf.get("started_at")) is not str
        or type(cnf.get("completed_at")) is not str
        or dimacs.get("sha256") != cnf.get("sha256")
        or dimacs.get("bytes") != cnf.get("bytes")
        or type(solver) is not dict
        or set(solver)
        != {
            "schema",
            "command",
            "solver",
            "seed",
            "single_process",
            "timeout_seconds_is_safety_guard_not_search_claim",
            "timeout_seconds",
            "timed_out",
            "started_at",
            "completed_at",
            "stdout",
            "stderr",
            "return_code",
            "status",
            "model",
        }
        or solver.get("schema") != SOLVER_RECEIPT_SCHEMA
        or solver.get("status") != "sat"
        or solver.get("return_code") != 10
        or solver.get("timed_out") is not False
        or not _json_exactly_equal(solver.get("seed"), SOLVER_SEED)
        or solver.get("single_process") is not True
        or solver.get(
            "timeout_seconds_is_safety_guard_not_search_claim"
        )
        is not True
        or type(solver.get("timeout_seconds")) is not int
        or solver["timeout_seconds"] < 60
        or type(solver.get("started_at")) is not str
        or type(solver.get("completed_at")) is not str
        or not _valid_hash_record(
            solver.get("stdout"), allow_zero_bytes=True
        )
        or not _valid_hash_record(
            solver.get("stderr"), allow_zero_bytes=True
        )
        or not _valid_hash_record(
            solver.get("model"), allow_zero_bytes=False
        )
        or type(solver.get("solver")) is not dict
        or set(solver["solver"])
        != {"path", "version", "bytes", "sha256"}
        or solver["solver"].get("version") != SOLVER_VERSION
        or solver["solver"].get("sha256") != SOLVER_SHA256
        or type(solver["solver"].get("path")) is not str
        or type(solver["solver"].get("bytes")) is not int
        or solver["solver"]["bytes"] <= 0
        or type(solver.get("command")) is not list
        or len(solver["command"]) != 7
        or solver["command"][0] != solver["solver"]["path"]
        or solver["command"][1:5]
        != ["-q", "--seed=0", "--check=true", "-w"]
        or type(solver["command"][5]) is not str
        or not solver["command"][5].endswith("solver.model")
        or type(solver["command"][6]) is not str
        or not solver["command"][6].endswith("common_prefix.cnf")
        or not _json_exactly_equal(
            terminal_logic,
            {
                "stop_on_first_independently_replayed_sat_model": True,
                "branches_attempted": [branch_index],
                "remaining_branches_not_run": list(
                    range(
                        branch_index + 1,
                        len(branch_representatives()),
                    )
                ),
                "conclusion": (
                    "Gallagher's common support prefix alone cannot "
                    "refute all 31 n=8 seed branches."
                ),
            },
        )
    ):
        raise KrennN8BaseSupportCNFError(
            "DIMACS or solver SAT receipt failed closed replay"
        )


def verify_bundle(results_directory: Path) -> None:
    results_directory = results_directory.resolve()
    expected_inventory = {
        "README.md",
        "certificate.json",
        "manifest.json",
    }
    if (
        not results_directory.is_dir()
        or {path.name for path in results_directory.iterdir()}
        != expected_inventory
    ):
        raise KrennN8BaseSupportCNFError(
            "result bundle inventory changed"
        )
    manifest = _strict_json(results_directory / "manifest.json")
    if (
        type(manifest) is not dict
        or set(manifest)
        != {
            "schema",
            "bundle_files",
            "source_files",
            "claim_boundary",
        }
        or manifest["schema"] != MANIFEST_SCHEMA
        or not _false_claims(manifest["claim_boundary"])
        or type(manifest["bundle_files"]) is not dict
        or type(manifest["source_files"]) is not dict
        or set(manifest["bundle_files"])
        != {"README.md", "certificate.json"}
        or set(manifest["source_files"]) != set(SOURCE_FILES)
    ):
        raise KrennN8BaseSupportCNFError(
            "result manifest schema or inventory changed"
        )
    root = _repo_root()
    for filename, record in manifest["bundle_files"].items():
        if not _json_exactly_equal(
            record, _file_record(results_directory / filename)
        ):
            raise KrennN8BaseSupportCNFError(
                f"bundle hash mismatch for {filename}"
            )
    for filename, record in manifest["source_files"].items():
        if not _json_exactly_equal(
            record, _file_record(root / filename)
        ):
            raise KrennN8BaseSupportCNFError(
                f"source hash mismatch for {filename}"
            )
    certificate = _strict_json(
        results_directory / "certificate.json"
    )
    verify_compact_certificate(certificate)


def run_campaign(
    *,
    scratch_directory: Path,
    results_directory: Path,
    solver: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    """Run canonical branches until the first exact SAT model."""

    scratch_directory = scratch_directory.resolve()
    results_directory = results_directory.resolve()
    scratch_directory.mkdir(parents=True, exist_ok=True)
    lock_path = scratch_directory / "campaign.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise KrennN8BaseSupportCNFError(
            "another campaign lock already exists"
        ) from error
    os.close(descriptor)
    progress_path = scratch_directory / "progress.json"
    try:
        for branch_index in range(len(branch_representatives())):
            branch_directory = (
                scratch_directory / f"branch_{branch_index:02d}"
            )
            branch_directory.mkdir(parents=True, exist_ok=True)
            cnf_path = branch_directory / "common_prefix.cnf"
            cnf_receipt_path = branch_directory / "cnf_receipt.json"
            model_path = branch_directory / "solver.model"
            stdout_path = branch_directory / "solver.stdout.txt"
            stderr_path = branch_directory / "solver.stderr.txt"
            solver_receipt_path = (
                branch_directory / "solver_receipt.json"
            )
            if cnf_path.exists() and cnf_receipt_path.exists():
                cnf_receipt = _strict_json(cnf_receipt_path)
                if (
                    type(cnf_receipt) is not dict
                    or cnf_receipt.get("schema")
                    != CNF_RECEIPT_SCHEMA
                    or cnf_receipt.get("branch_index")
                    != branch_index
                    or cnf_receipt.get("complete") is not True
                    or cnf_receipt.get("bytes")
                    != cnf_path.stat().st_size
                    or cnf_receipt.get("sha256")
                    != _sha256_file(cnf_path)
                ):
                    raise KrennN8BaseSupportCNFError(
                        "existing CNF checkpoint failed hash replay"
                    )
            else:
                if cnf_path.exists() or cnf_receipt_path.exists():
                    raise KrennN8BaseSupportCNFError(
                        "partial CNF checkpoint fails closed"
                    )
                cnf_receipt = write_dimacs(
                    cnf_path,
                    branch_index=branch_index,
                    progress_path=progress_path,
                )
                _atomic_json(cnf_receipt_path, cnf_receipt)
            _atomic_json(
                progress_path,
                {
                    "schema": PROGRESS_SCHEMA,
                    "stage": "solving",
                    "branch_index": branch_index,
                    "updated_at": _utc_now(),
                    "complete": False,
                },
            )
            solver_receipt = run_solver(
                cnf_path,
                model_path,
                stdout_path,
                stderr_path,
                solver=solver,
                timeout_seconds=timeout_seconds,
            )
            _atomic_json(solver_receipt_path, solver_receipt)
            if solver_receipt["status"] == "sat":
                solver_assignment = _parse_solver_model(model_path)
                active_edges = tuple(
                    variable
                    for variable in range(1, EDGE_VARIABLE_COUNT + 1)
                    if solver_assignment[variable]
                )
                derived_assignment, semantic_report = (
                    derive_assignment_and_semantic_report(
                        active_edges,
                        branch_index=branch_index,
                    )
                )
                if derived_assignment != solver_assignment:
                    raise KrennN8BaseSupportCNFError(
                        "solver auxiliaries differ from exact definitions"
                    )
                dimacs_replay = (
                    verify_dimacs_against_generator_and_model(
                        cnf_path,
                        branch_index=branch_index,
                        assignment=derived_assignment,
                    )
                )
                if (
                    dimacs_replay["sha256"]
                    != cnf_receipt["sha256"]
                    or dimacs_replay["bytes"]
                    != cnf_receipt["bytes"]
                ):
                    raise KrennN8BaseSupportCNFError(
                        "DIMACS replay differs from generation receipt"
                    )
                certificate = _build_certificate(
                    branch_index=branch_index,
                    active_edges=active_edges,
                    semantic_report=semantic_report,
                    cnf_receipt=cnf_receipt,
                    solver_receipt=solver_receipt,
                    dimacs_replay=dimacs_replay,
                )
                verify_compact_certificate(certificate)
                results_directory.mkdir(parents=True, exist_ok=True)
                if not (results_directory / "README.md").is_file():
                    raise KrennN8BaseSupportCNFError(
                        "result README must exist before campaign write"
                    )
                _atomic_json(
                    results_directory / "certificate.json",
                    certificate,
                )
                _atomic_json(
                    results_directory / "manifest.json",
                    _build_manifest(results_directory),
                )
                verify_bundle(results_directory)
                _atomic_json(
                    progress_path,
                    {
                        "schema": PROGRESS_SCHEMA,
                        "stage": "complete_sat",
                        "branch_index": branch_index,
                        "updated_at": _utc_now(),
                        "complete": True,
                    },
                )
                return certificate
            if solver_receipt["status"] == "unsat":
                raise KrennN8BaseSupportCNFError(
                    "UNSAT requires the separate text-LRAT/Lean path; "
                    "the campaign refuses to promote CaDiCaL status alone"
                )
            raise KrennN8BaseSupportCNFError(
                "timeout or solver failure is not a claim"
            )
        raise KrennN8BaseSupportCNFError(
            "all branches ended without a checked terminal result"
        )
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-directory",
        type=Path,
        default=DEFAULT_SCRATCH,
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=DEFAULT_RESULTS,
    )
    parser.add_argument(
        "--solver",
        type=Path,
        default=DEFAULT_SOLVER,
    )
    parser.add_argument(
        "--solver-timeout-seconds",
        type=int,
        default=SOLVER_SAFETY_TIMEOUT_SECONDS,
    )
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.verify_only:
        verify_bundle(arguments.results_directory)
        return 0
    timeout_seconds = _exact_integer(
        arguments.solver_timeout_seconds, "solver timeout"
    )
    if timeout_seconds < 60:
        raise KrennN8BaseSupportCNFError(
            "solver safety timeout must be at least 60 seconds"
        )
    run_campaign(
        scratch_directory=arguments.scratch_directory,
        results_directory=arguments.results_directory,
        solver=arguments.solver,
        timeout_seconds=timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
