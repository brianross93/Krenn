r"""Exact replay of the bounded ``k=2`` one-shell reconnaissance.

The one-shell matrix contains every fine-graded column which has an output
in ``supp(D^2)``.  It does *not* contain columns having no such output.
The recorded large-matrix counts, fingerprints, and modular ranks therefore
remain reconnaissance.  Normal verification in this module reconstructs
only two row orbits and the three full-domain column orbits which touch
them.

The exact local statement is:

* on the one-shell columns, ``3 e_S - e_E`` is an integer separator with
  pairing ``1080`` against ``D^2``;
* two columns outside that shell each pair to ``-1`` with the same
  functional.

Consequently the separator proves nonmembership only in the controlled
one-shell span.  It does not decide ``D^2 in J_mix``,
``D in radical(J_mix)``, or exact affine GHZ membership.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import permutations, product
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.higher_power_source_ideal import (
    K6_EDGE_INDEX,
    squared_hafnian_support,
)
from experiments.krenn_quantum_graph.system import (
    coloring_index,
    generate_sparse_system,
    variable_index,
    variable_key,
)


ONE_SHELL_RECONNAISSANCE_SCHEMA = (
    "krenn.n6_d3.k2.one_shell_reconnaissance.v1"
)

SUPPORT_ROW = (
    0, 0, 13, 13, 26, 26, 53, 53, 67,
    67, 99, 99, 108, 108, 121, 121, 134, 134,
)
ESCAPE_ROW = (
    0, 0, 5, 13, 26, 26, 58, 58, 71,
    90, 90, 95, 107, 107, 117, 117, 130, 130,
)

SHELL_COLUMN = (
    (0, 0, 1, 1, 2, 2),
    (
        0, 17, 17, 31, 31, 62, 62, 76,
        76, 85, 90, 90, 117, 117, 134,
    ),
)
GLOBAL_ESCAPE_COLUMNS = (
    (
        (0, 0, 0, 0, 1, 2),
        (
            0, 5, 17, 17, 31, 49, 49, 80,
            81, 116, 116, 121, 121, 126, 126,
        ),
    ),
    (
        (0, 0, 1, 1, 2, 2),
        (
            0, 17, 17, 31, 31, 62, 62, 76,
            76, 90, 90, 95, 117, 117, 122,
        ),
    ),
)

EXPECTED_COLUMN_ORBIT_FINGERPRINT = (
    "f5ca49ab68e93c70640a16b857a2dccce7b995df357277f17a373e0e51a7692a"
)
EXPECTED_SPARSE_SIGNATURE_FINGERPRINT = (
    "627ad8463b617c962b44c56af9e470dc8c2b685aaf81b63cf2ef0f5ca19bb156"
)

SCOPE_COUNTS = {
    "D_squared_support_orbits": 663,
    "mixed_colorings": 726,
    "matching_terms_per_generator": 15,
    "divisibility_tests": 7_220_070,
    "raw_columns_hitting_support_representatives": 11_148,
    "S6_x_S3_column_orbits": 4_493,
    "complete_polynomial_output_signatures": 4_493,
    "output_raw_rows": 127_693,
    "output_row_orbits": 39_033,
    "matrix_nonzeros": 60_850,
}

MODULAR_RANK_RECORDS = (
    (31, 4_493, 4_494),
    (1_009, 4_493, 4_494),
    (1_000_003, 4_493, 4_494),
)

SCRATCH_ARTIFACT_HASHES = {
    "one_shell_result.json": (
        "4f4a7efc4fe34fe9efa65c681d7a86f408b46c05b7c8dfab666268a382ac3b34"
    ),
    "one_shell_private_analysis.json": (
        "ee589358ad0db8703fb4e09adc75e72cb638bad7ad1d38504419603d6c43f388"
    ),
    "one_shell_separator_expansion.json": (
        "f8e832bdecbfc18a450b8882ddd364bacce35e74a2fed08841a30ecacdf361ab"
    ),
    "one_shell_coverage_audit.json": (
        "b4ab8cfe80992698877070da60065f6fe2d9906950bc68974464927759b6e039"
    ),
    "one_shell_full_separator_beam_search.json": (
        "2e2f8dc7c0ac3f5e10614b9328943a83a41bc4b2280d57ebe767ceeb94bd545b"
    ),
}


class KrennOneShellReconnaissanceError(ValueError):
    """A one-shell receipt or its exact local replay is malformed."""


def _row_record(
    monomial: Sequence[int],
    *,
    role: str,
    weight: int,
    orbit_size: int,
    D_squared_coefficient: int,
) -> dict:
    return {
        "role": role,
        "variable_indices": list(monomial),
        "weight": weight,
        "orbit_size": orbit_size,
        "D_squared_coefficient": D_squared_coefficient,
    }


def _column_record(
    column: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    support_hits: int,
    escape_hits: int,
    separator_value: int,
    shell_membership: str,
) -> dict:
    coloring, multiplier = column
    return {
        "coloring": list(coloring),
        "multiplier_variable_indices": list(multiplier),
        "support_row_hits": support_hits,
        "escape_row_hits": escape_hits,
        "separator_value": separator_value,
        "shell_membership": shell_membership,
    }


def _receipt_body() -> dict:
    return {
        "schema": ONE_SHELL_RECONNAISSANCE_SCHEMA,
        "power": 2,
        "scope": {
            "exhaustive_within_declared_scope": True,
            "sampled": False,
            "full_k2_domain_enumerated": False,
            "description": (
                "every mixed-generator column orbit having at least one "
                "output in supp(D^2)"
            ),
            "recompute_command": (
                "python -m experiments.krenn_quantum_graph."
                "one_shell_full_recompute"
            ),
            "random_seed": None,
            "counts": dict(SCOPE_COUNTS),
            "coverage_argument": (
                "if m*F_c has output r in supp(D^2), its selected term "
                "divides r and m is uniquely r/term; transporting r to "
                "one of 663 support representatives transports the column "
                "to one of the exhaustively tested divisions"
            ),
        },
        "deterministic_sparse_layout": {
            "row_order": (
                "lexicographic order of the least sorted 18-byte "
                "variable-index monomial in each S6 x S3 row orbit"
            ),
            "column_order": (
                "lexicographic order of the least "
                "(6-byte coloring, 15-byte sorted multiplier) pair in "
                "each S6 x S3 column orbit"
            ),
            "entry_rule": (
                "number from 0 through 15 of the column's complete "
                "polynomial outputs lying in the row orbit"
            ),
            "column_orbit_fingerprint": (
                EXPECTED_COLUMN_ORBIT_FINGERPRINT
            ),
            "column_orbit_fingerprint_recipe": (
                "encode each canonical column as coloring bytes, ff, "
                "multiplier bytes, 0a; lexicographically sort encodings; "
                "SHA-256 their concatenation"
            ),
            "sparse_signature_fingerprint": (
                EXPECTED_SPARSE_SIGNATURE_FINGERPRINT
            ),
            "sparse_signature_fingerprint_recipe": (
                "for each column concatenate its sorted canonical "
                "row[18]+coefficient_uint16_big_endian entries and append "
                "0a; sort the column byte strings; SHA-256 their "
                "concatenation"
            ),
            "recomputed_in_normal_verification": False,
        },
        "modular_rank_diagnostics": {
            "role": "nonproof-reconnaissance",
            "records": [
                {
                    "role": "nonproof-reconnaissance",
                    "prime": prime,
                    "matrix_rank": rank,
                    "augmented_rank": augmented_rank,
                }
                for prime, rank, augmented_rank in MODULAR_RANK_RECORDS
            ],
            "used_as_characteristic_zero_certificate": False,
        },
        "exact_shell_separator": {
            "basis": (
                "S6 x S3 invariant row-orbit indicator functionals"
            ),
            "support_row": _row_record(
                SUPPORT_ROW,
                role="D_squared_support",
                weight=3,
                orbit_size=360,
                D_squared_coefficient=1,
            ),
            "escape_row": _row_record(
                ESCAPE_ROW,
                role="off_support_escape",
                weight=-1,
                orbit_size=1_080,
                D_squared_coefficient=0,
            ),
            "sole_included_touching_column": _column_record(
                SHELL_COLUMN,
                support_hits=1,
                escape_hits=3,
                separator_value=0,
                shell_membership="included_direct_support_hit",
            ),
            "included_touching_column_orbits": 1,
            "integer_rhs_pairing": 1_080,
            "exact_symbolic_replay_required": True,
        },
        "explicit_global_escape_columns": [
            _column_record(
                column,
                support_hits=0,
                escape_hits=1,
                separator_value=-1,
                shell_membership="omitted_no_direct_support_hit",
            )
            for column in GLOBAL_ESCAPE_COLUMNS
        ],
        "bounded_extension_search": {
            "role": "nonproof-reconnaissance",
            "maximum_functional_support": 12,
            "states_seen_from_best_start": 1_438,
            "best_terminal_violating_column_orbits": 12,
            "exact_extension_found": False,
            "claim": "search boundary only",
        },
        "scratch_artifact_hashes": {
            "role": "recorded-audit-trail-not-read-by-normal-tests",
            "directory": (
                "D:\\KrennScratch\\obstruction_certificate"
            ),
            "sha256": dict(SCRATCH_ARTIFACT_HASHES),
        },
        "computations_performed_in_normal_verification": {
            "full_39033_by_4493_matrix_reconstructed": False,
            "selected_row_orbits_reconstructed": 2,
            "selected_full_domain_column_orbits_reconstructed": 3,
        },
        "claims": {
            "controlled_one_shell_nonmembership_proved": True,
            "D_squared_global_nonmembership_proved": False,
            "D_squared_in_J_mix_decided": False,
            "D_in_radical_J_mix_decided": False,
            "global_GHZ_nonexistence_proved": False,
            "exact_affine_GHZ_membership_decided": False,
            "evidence_status": "global affine membership remains undecided",
        },
    }


def one_shell_receipt_fingerprint(payload: Mapping) -> str:
    """Hash a JSON-ready receipt while excluding its hash field."""

    if not isinstance(payload, Mapping):
        raise KrennOneShellReconnaissanceError(
            "the one-shell receipt must be a mapping"
        )
    body = {
        key: value
        for key, value in payload.items()
        if key != "receipt_fingerprint"
    }
    try:
        encoded = (
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise KrennOneShellReconnaissanceError(
            "the one-shell receipt is not JSON-ready"
        ) from error
    return sha256(encoded).hexdigest()


def one_shell_reconnaissance_receipt() -> dict:
    """Return the small, round-trippable bounded-search receipt."""

    payload = _receipt_body()
    payload["receipt_fingerprint"] = one_shell_receipt_fingerprint(
        payload
    )
    return payload


@lru_cache(maxsize=1)
def _system():
    return generate_sparse_system(6, 3)


def _transport_coloring(
    coloring: Sequence[int],
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, ...]:
    transported = [0] * 6
    for old_vertex, color in enumerate(coloring):
        transported[int(vertex_permutation[old_vertex])] = (
            int(color_permutation[int(color)])
        )
    return tuple(transported)


@lru_cache(maxsize=1)
def _group_actions() -> tuple[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...
]:
    actions = []
    for vertex_permutation in permutations(range(6)):
        for color_permutation in permutations(range(3)):
            variable_permutation = tuple(
                variable_index(
                    6,
                    3,
                    vertex_permutation[i],
                    vertex_permutation[j],
                    color_permutation[a],
                    color_permutation[b],
                )
                for i, j, a, b in (
                    variable_key(6, 3, variable)
                    for variable in range(135)
                )
            )
            actions.append((
                vertex_permutation,
                color_permutation,
                variable_permutation,
            ))
    if len(actions) != 4_320:
        raise KrennOneShellReconnaissanceError(
            "the S6 x S3 action census changed"
        )
    return tuple(actions)


def _transport_monomial(
    monomial: Sequence[int],
    variable_permutation: Sequence[int],
) -> tuple[int, ...]:
    return tuple(sorted(
        int(variable_permutation[int(variable)])
        for variable in monomial
    ))


@lru_cache(maxsize=None)
def _canonical_monomial(
    monomial: tuple[int, ...],
) -> tuple[int, ...]:
    return min(
        _transport_monomial(monomial, variable_permutation)
        for _vertex, _color, variable_permutation in _group_actions()
    )


@lru_cache(maxsize=None)
def _monomial_orbit(
    monomial: tuple[int, ...],
) -> frozenset[tuple[int, ...]]:
    return frozenset(
        _transport_monomial(monomial, variable_permutation)
        for _vertex, _color, variable_permutation in _group_actions()
    )


Column = tuple[tuple[int, ...], tuple[int, ...]]


@lru_cache(maxsize=None)
def _canonical_column(column: Column) -> Column:
    coloring, multiplier = column
    return min(
        (
            _transport_coloring(
                coloring,
                vertex_permutation,
                color_permutation,
            ),
            _transport_monomial(multiplier, variable_permutation),
        )
        for (
            vertex_permutation,
            color_permutation,
            variable_permutation,
        ) in _group_actions()
    )


def _validate_monomial(
    value: Sequence[int],
    *,
    length: int,
    label: str,
) -> tuple[int, ...]:
    try:
        monomial = tuple(value)
    except TypeError as error:
        raise KrennOneShellReconnaissanceError(
            f"{label} is not a variable-index sequence"
        ) from error
    if (
        len(monomial) != length
        or any(
            isinstance(variable, bool)
            or not isinstance(variable, int)
            or variable not in range(135)
            for variable in monomial
        )
        or tuple(sorted(monomial)) != monomial
    ):
        raise KrennOneShellReconnaissanceError(
            f"{label} is not a sorted degree-{length} monomial"
        )
    return monomial


def _validate_column_record(record: Mapping, label: str) -> Column:
    if not isinstance(record, Mapping):
        raise KrennOneShellReconnaissanceError(
            f"{label} is not a column record"
        )
    try:
        coloring = tuple(record["coloring"])
        multiplier_value = record["multiplier_variable_indices"]
    except (KeyError, TypeError) as error:
        raise KrennOneShellReconnaissanceError(
            f"{label} is missing its coloring or multiplier"
        ) from error
    if (
        len(coloring) != 6
        or any(
            isinstance(color, bool)
            or not isinstance(color, int)
            or color not in range(3)
            for color in coloring
        )
        or len(set(coloring)) == 1
    ):
        raise KrennOneShellReconnaissanceError(
            f"{label} does not use a mixed coloring"
        )
    multiplier = _validate_monomial(
        multiplier_value,
        length=15,
        label=f"{label} multiplier",
    )
    return coloring, multiplier


def _fine_degree(monomial: Sequence[int]) -> tuple[int, ...]:
    degree = [0] * 18
    for variable in monomial:
        i, j, a, b = variable_key(6, 3, int(variable))
        degree[3 * i + a] += 1
        degree[3 * j + b] += 1
    return tuple(degree)


def _column_outputs(column: Column) -> tuple[tuple[int, ...], ...]:
    coloring, multiplier = column
    equation = coloring_index(6, 3, coloring)
    return tuple(
        tuple(sorted((*multiplier, *term)))
        for term in _system().equation_monomials(equation)
    )


def _D_squared_coefficient(monomial: Sequence[int]) -> int:
    graphs = [[0] * 15 for _color in range(3)]
    for variable in monomial:
        i, j, a, b = variable_key(6, 3, int(variable))
        if a != b:
            return 0
        graphs[a][K6_EDGE_INDEX[(i, j)]] += 1
    support = set(squared_hafnian_support())
    graph_tuples = tuple(tuple(graph) for graph in graphs)
    if any(graph not in support for graph in graph_tuples):
        return 0
    coefficient = 1
    for graph in graph_tuples:
        coefficient *= (
            1 if sum(exponent == 2 for exponent in graph) == 3 else 2
        )
    return coefficient


def _factor_columns(monomial: tuple[int, ...]) -> frozenset[Column]:
    """Enumerate every full-domain column having this exact output."""

    available = Counter(monomial)
    columns = set()
    for coloring in product(range(3), repeat=6):
        if len(set(coloring)) == 1:
            continue
        terms = _system().equation_monomials(
            coloring_index(6, 3, coloring)
        )
        for term in terms:
            required = Counter(term)
            if any(
                required[variable] > available[variable]
                for variable in required
            ):
                continue
            residual = available.copy()
            residual.subtract(term)
            multiplier = tuple(sorted(residual.elements()))
            columns.add((tuple(coloring), multiplier))
    return frozenset(columns)


def _incident_column_orbits(
    monomial: tuple[int, ...],
) -> frozenset[Column]:
    return frozenset(
        _canonical_column(column)
        for column in _factor_columns(monomial)
    )


def _row_hits(
    column: Column,
    support_orbit: frozenset[tuple[int, ...]],
    escape_orbit: frozenset[tuple[int, ...]],
) -> tuple[int, int]:
    outputs = _column_outputs(column)
    return (
        sum(output in support_orbit for output in outputs),
        sum(output in escape_orbit for output in outputs),
    )


def replay_one_shell_separator(payload: Mapping) -> Mapping[str, object]:
    """Exactly replay the local separator and both global escapes."""

    try:
        separator = payload["exact_shell_separator"]
        support_record = separator["support_row"]
        escape_record = separator["escape_row"]
        shell_record = separator["sole_included_touching_column"]
        global_records = payload["explicit_global_escape_columns"]
    except (KeyError, TypeError) as error:
        raise KrennOneShellReconnaissanceError(
            "the exact separator records are missing"
        ) from error

    support_row = _validate_monomial(
        support_record["variable_indices"],
        length=18,
        label="support row",
    )
    escape_row = _validate_monomial(
        escape_record["variable_indices"],
        length=18,
        label="escape row",
    )
    if (
        _fine_degree(support_row) != (2,) * 18
        or _fine_degree(escape_row) != (2,) * 18
    ):
        raise KrennOneShellReconnaissanceError(
            "a separator row has the wrong fine degree"
        )
    if (
        _canonical_monomial(support_row) != support_row
        or _canonical_monomial(escape_row) != escape_row
    ):
        raise KrennOneShellReconnaissanceError(
            "a separator row is not the canonical orbit representative"
        )

    support_orbit = _monomial_orbit(support_row)
    escape_orbit = _monomial_orbit(escape_row)
    if support_orbit & escape_orbit:
        raise KrennOneShellReconnaissanceError(
            "the two separator rows share an orbit"
        )
    support_coefficient = _D_squared_coefficient(support_row)
    escape_coefficient = _D_squared_coefficient(escape_row)
    if (
        len(support_orbit) != support_record["orbit_size"]
        or len(escape_orbit) != escape_record["orbit_size"]
        or support_coefficient
        != support_record["D_squared_coefficient"]
        or escape_coefficient
        != escape_record["D_squared_coefficient"]
    ):
        raise KrennOneShellReconnaissanceError(
            "a separator row orbit or D^2 coefficient changed"
        )

    shell_column = _validate_column_record(
        shell_record, "shell column"
    )
    escape_columns = tuple(
        _validate_column_record(
            record, f"global escape column {index}"
        )
        for index, record in enumerate(global_records)
    )
    if (
        _canonical_column(shell_column) != shell_column
        or any(
            _canonical_column(column) != column
            for column in escape_columns
        )
    ):
        raise KrennOneShellReconnaissanceError(
            "a selected column is not its canonical orbit representative"
        )

    support_weight = support_record["weight"]
    escape_weight = escape_record["weight"]
    shell_hits = _row_hits(
        shell_column, support_orbit, escape_orbit
    )
    if (
        shell_hits
        != (
            shell_record["support_row_hits"],
            shell_record["escape_row_hits"],
        )
        or (
            support_weight * shell_hits[0]
            + escape_weight * shell_hits[1]
        )
        != shell_record["separator_value"]
        or shell_record["separator_value"] != 0
    ):
        raise KrennOneShellReconnaissanceError(
            "the exact shell column no longer annihilates the separator"
        )

    escape_hits = []
    for record, column in zip(
        global_records, escape_columns, strict=True
    ):
        hits = _row_hits(column, support_orbit, escape_orbit)
        value = (
            support_weight * hits[0]
            + escape_weight * hits[1]
        )
        if (
            hits
            != (
                record["support_row_hits"],
                record["escape_row_hits"],
            )
            or value != record["separator_value"]
            or value != -1
        ):
            raise KrennOneShellReconnaissanceError(
                "an explicit global escape column changed"
            )
        escape_hits.append(hits)

    incident = (
        _incident_column_orbits(support_row)
        | _incident_column_orbits(escape_row)
    )
    expected_incident = frozenset((
        shell_column,
        *escape_columns,
    ))
    if incident != expected_incident:
        raise KrennOneShellReconnaissanceError(
            "the complete local incident-column census changed"
        )
    directly_hits_support = {
        column
        for column in incident
        if any(
            _D_squared_coefficient(output)
            for output in _column_outputs(column)
        )
    }
    if directly_hits_support != {shell_column}:
        raise KrennOneShellReconnaissanceError(
            "the sole included touching column claim changed"
        )

    rhs_pairing = (
        support_weight
        * len(support_orbit)
        * support_coefficient
        + escape_weight
        * len(escape_orbit)
        * escape_coefficient
    )
    if rhs_pairing != separator["integer_rhs_pairing"]:
        raise KrennOneShellReconnaissanceError(
            "the exact integer RHS pairing changed"
        )
    return {
        "support_orbit_size": len(support_orbit),
        "escape_orbit_size": len(escape_orbit),
        "shell_column_hits": list(shell_hits),
        "global_escape_column_hits": [
            list(hits) for hits in escape_hits
        ],
        "local_incident_column_orbits": len(incident),
        "included_touching_column_orbits": len(
            directly_hits_support
        ),
        "integer_rhs_pairing": rhs_pairing,
        "exact_shell_separator_replayed": True,
        "explicit_global_escapes_replayed": True,
    }


def verify_one_shell_reconnaissance_receipt(
    payload: Mapping,
) -> Mapping[str, object]:
    """Fail closed on metadata or exact semantic corruption."""

    if not isinstance(payload, Mapping):
        raise KrennOneShellReconnaissanceError(
            "the one-shell receipt must be a mapping"
        )
    recorded_fingerprint = payload.get("receipt_fingerprint")
    if (
        not isinstance(recorded_fingerprint, str)
        or len(recorded_fingerprint) != 64
        or recorded_fingerprint
        != one_shell_receipt_fingerprint(payload)
    ):
        raise KrennOneShellReconnaissanceError(
            "the one-shell receipt fingerprint changed"
        )
    actual_body = {
        key: value
        for key, value in payload.items()
        if key != "receipt_fingerprint"
    }
    if actual_body != _receipt_body():
        raise KrennOneShellReconnaissanceError(
            "the recorded one-shell rows, columns, counts, or claims changed"
        )

    replay = dict(replay_one_shell_separator(payload))
    claims = payload["claims"]
    if (
        claims["D_squared_global_nonmembership_proved"]
        or claims["D_squared_in_J_mix_decided"]
        or claims["D_in_radical_J_mix_decided"]
        or claims["global_GHZ_nonexistence_proved"]
        or claims["exact_affine_GHZ_membership_decided"]
    ):
        raise KrennOneShellReconnaissanceError(
            "a bounded one-shell result was promoted to a global claim"
        )
    replay.update({
        "receipt_fingerprint_valid": True,
        "exhaustive_within_declared_scope": True,
        "full_matrix_reconstructed": False,
        "controlled_one_shell_nonmembership_proved": True,
        "D_squared_global_nonmembership_proved": False,
        "D_in_radical_J_mix_decided": False,
        "global_GHZ_nonexistence_proved": False,
        "exact_affine_GHZ_membership_decided": False,
    })
    return replay


def main() -> int:
    payload = one_shell_reconnaissance_receipt()
    verify_one_shell_reconnaissance_receipt(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
