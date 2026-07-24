r"""Opt-in deterministic recomputation of the complete ``k=2`` one shell.

This module consolidates the disposable one-shell construction used during
the ``D^2 in J_mix`` reconnaissance.  The one shell consists of every
fine-graded mixed-generator column having at least one output monomial in
``supp(D^2)``.  It is exhaustive for that declared scope, but it is not the
full ``k=2`` source-ideal matrix.

The expensive path is never run on import and is opt-in at the command line::

    python -m experiments.krenn_quantum_graph.one_shell_full_recompute \
        --run-full

With no mode flag, the command performs only the small committed semantic
validation.  Full-run receipts, a gzip-compressed deterministic sparse cache,
and logs default to ``D:\KrennScratch\obstruction_certificate``.

The deterministic conventions are:

* a row is named by the lexicographically least sorted 18-byte monomial in
  its ``S_6 x S_3`` orbit;
* a column is named by the lexicographically least pair consisting of its
  six-byte coloring and sorted fifteen-byte multiplier;
* rows and columns are ordered lexicographically by those names;
* an entry is the number of the column's fifteen complete polynomial
  outputs lying in the row orbit.

Modular ranks are explicitly reconnaissance.  The exact two-row shell
separator and its two global escape columns are verified by the small
receipt module, not inferred from the modular computations here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

from experiments.krenn_quantum_graph.higher_power_source_ideal import (
    K6_EDGES,
    d_squared_support_orbits,
    squared_hafnian_support,
)
from experiments.krenn_quantum_graph.one_shell_reconnaissance import (
    EXPECTED_COLUMN_ORBIT_FINGERPRINT,
    EXPECTED_SPARSE_SIGNATURE_FINGERPRINT,
    MODULAR_RANK_RECORDS,
    SCOPE_COUNTS,
    one_shell_reconnaissance_receipt,
    verify_one_shell_reconnaissance_receipt,
)
from experiments.krenn_quantum_graph.system import (
    coloring_index,
    generate_sparse_system,
    variable_index,
    variable_key,
)


DEFAULT_SCRATCH_DIRECTORY = Path(
    r"D:\KrennScratch\obstruction_certificate"
)
DEFAULT_RECEIPT_NAME = "one_shell_full_recompute_receipt.json"
DEFAULT_REPORT_NAME = "one_shell_full_recompute_report.json"
DEFAULT_CACHE_NAME = "one_shell_full_recompute_cache.json.gz"
DEFAULT_LOG_NAME = "one_shell_full_recompute.log"

FULL_RECOMPUTE_SCHEMA = (
    "krenn.n6_d3.k2.one_shell_full_recompute.v1"
)
FULL_CACHE_SCHEMA = (
    "krenn.n6_d3.k2.one_shell_deterministic_sparse_cache.v1"
)
PRIMES = tuple(record[0] for record in MODULAR_RANK_RECORDS)


class KrennOneShellFullRecomputeError(ValueError):
    """The deterministic one-shell reconstruction changed."""


@dataclass(frozen=True)
class GroupArrays:
    vertex_permutations: np.ndarray
    color_permutations: np.ndarray
    variable_permutations: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.vertex_permutations.shape != (4_320, 6)
            or self.color_permutations.shape != (4_320, 3)
            or self.variable_permutations.shape != (4_320, 135)
            or self.vertex_permutations.dtype != np.uint8
            or self.color_permutations.dtype != np.uint8
            or self.variable_permutations.dtype != np.uint8
        ):
            raise KrennOneShellFullRecomputeError(
                "the vectorized S6 x S3 action has the wrong shape"
            )


@dataclass(frozen=True)
class ModularRankRecord:
    prime: int
    matrix_rank: int
    augmented_rank: int
    rhs_residual_nonzeros: int
    seconds: float
    role: str = "nonproof-reconnaissance"

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "prime": self.prime,
            "matrix_rank": self.matrix_rank,
            "augmented_rank": self.augmented_rank,
            "rhs_residual_nonzeros": self.rhs_residual_nonzeros,
            "seconds": self.seconds,
        }


Column = tuple[tuple[int, ...], tuple[int, ...]]
RowKey = bytes
RowSignature = tuple[tuple[RowKey, int], ...]
IndexedSignature = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FullRecomputeResult:
    raw_column_count: int
    raw_row_count: int
    row_representatives: tuple[RowKey, ...]
    columns: tuple[Column, ...]
    signatures: tuple[IndexedSignature, ...]
    rhs_entries: tuple[tuple[int, int], ...]
    matrix_nonzeros: int
    column_orbit_fingerprint: str
    sparse_signature_fingerprint: str
    modular_ranks: tuple[ModularRankRecord, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if (
            self.raw_column_count
            != SCOPE_COUNTS[
                "raw_columns_hitting_support_representatives"
            ]
            or self.raw_row_count != SCOPE_COUNTS["output_raw_rows"]
            or len(self.row_representatives)
            != SCOPE_COUNTS["output_row_orbits"]
            or len(self.columns)
            != SCOPE_COUNTS["S6_x_S3_column_orbits"]
            or len(self.signatures) != len(self.columns)
            or self.matrix_nonzeros
            != SCOPE_COUNTS["matrix_nonzeros"]
            or len(self.rhs_entries)
            != SCOPE_COUNTS["D_squared_support_orbits"]
            or self.column_orbit_fingerprint
            != EXPECTED_COLUMN_ORBIT_FINGERPRINT
            or self.sparse_signature_fingerprint
            != EXPECTED_SPARSE_SIGNATURE_FINGERPRINT
        ):
            raise KrennOneShellFullRecomputeError(
                "the deterministic one-shell census or fingerprint changed"
            )
        observed_ranks = tuple(
            (
                record.prime,
                record.matrix_rank,
                record.augmented_rank,
            )
            for record in self.modular_ranks
        )
        if observed_ranks != MODULAR_RANK_RECORDS:
            raise KrennOneShellFullRecomputeError(
                "a one-shell modular rank pair changed"
            )

    def report(self) -> dict:
        return {
            "schema": FULL_RECOMPUTE_SCHEMA,
            "scope": {
                "exhaustive_within_declared_scope": True,
                "sampled": False,
                "full_k2_domain_enumerated": False,
                "support_representatives": (
                    SCOPE_COUNTS["D_squared_support_orbits"]
                ),
                "mixed_generator_terms_tested_per_representative": (
                    SCOPE_COUNTS["mixed_colorings"]
                    * SCOPE_COUNTS["matching_terms_per_generator"]
                ),
                "total_divisibility_tests": (
                    SCOPE_COUNTS["divisibility_tests"]
                ),
            },
            "matrix": {
                "raw_columns": self.raw_column_count,
                "raw_rows": self.raw_row_count,
                "orbit_rows": len(self.row_representatives),
                "orbit_columns": len(self.columns),
                "nonzeros": self.matrix_nonzeros,
                "rhs_orbits": len(self.rhs_entries),
            },
            "deterministic_layout": {
                "row_order": (
                    "lexicographic canonical 18-byte row representative"
                ),
                "column_order": (
                    "lexicographic canonical "
                    "(6-byte coloring,15-byte multiplier)"
                ),
                "column_orbit_fingerprint": (
                    self.column_orbit_fingerprint
                ),
                "sparse_signature_fingerprint": (
                    self.sparse_signature_fingerprint
                ),
            },
            "modular_rank_diagnostics": {
                "role": "nonproof-reconnaissance",
                "records": [
                    record.to_dict()
                    for record in self.modular_ranks
                ],
            },
            "elapsed_seconds": self.elapsed_seconds,
            "claims": {
                "controlled_one_shell_nonmembership_proved": True,
                "D_squared_global_nonmembership_proved": False,
                "D_squared_in_J_mix_decided": False,
                "D_in_radical_J_mix_decided": False,
                "global_GHZ_nonexistence_proved": False,
                "exact_affine_GHZ_membership_decided": False,
            },
        }


class RunLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        line = str(message)
        print(line, flush=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def _json_bytes(payload: Mapping) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )


def _write_deterministic_gzip_json(
    path: Path,
    payload: Mapping,
) -> tuple[str, str]:
    """Write canonical JSON in a gzip stream with timestamp zero."""

    encoded = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            compressed.write(encoded)
    return sha256(encoded).hexdigest(), sha256(
        path.read_bytes()
    ).hexdigest()


def _build_group_arrays() -> GroupArrays:
    vertex_rows = []
    color_rows = []
    variable_rows = []
    for vertex_permutation in permutations(range(6)):
        for color_permutation in permutations(range(3)):
            vertex_rows.append(vertex_permutation)
            color_rows.append(color_permutation)
            variable_rows.append(tuple(
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
            ))
    return GroupArrays(
        vertex_permutations=np.asarray(
            vertex_rows, dtype=np.uint8
        ),
        color_permutations=np.asarray(
            color_rows, dtype=np.uint8
        ),
        variable_permutations=np.asarray(
            variable_rows, dtype=np.uint8
        ),
    )


def _least_matrix_row(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or not len(matrix):
        raise KrennOneShellFullRecomputeError(
            "cannot canonicalize an empty matrix of orbit images"
        )
    # np.lexsort uses its last key as primary.  Reversing the columns
    # therefore produces ordinary left-to-right lexicographic order.
    order = np.lexsort(matrix[:, ::-1].T)
    return matrix[int(order[0])]


def _canonical_column(
    column: Column,
    group: GroupArrays,
) -> Column:
    coloring, multiplier = column
    coloring_array = np.asarray(coloring, dtype=np.uint8)
    row_indices = np.arange(4_320)
    colors_at_old_vertices = group.color_permutations[
        row_indices[:, None],
        coloring_array[None, :],
    ]
    transported_coloring = np.empty((4_320, 6), dtype=np.uint8)
    for old_vertex in range(6):
        transported_coloring[
            row_indices,
            group.vertex_permutations[:, old_vertex],
        ] = colors_at_old_vertices[:, old_vertex]

    multiplier_array = np.asarray(multiplier, dtype=np.uint8)
    transported_multiplier = np.sort(
        group.variable_permutations[:, multiplier_array],
        axis=1,
    )
    combined = np.concatenate(
        (transported_coloring, transported_multiplier),
        axis=1,
    )
    canonical = _least_matrix_row(combined)
    return (
        tuple(map(int, canonical[:6])),
        tuple(map(int, canonical[6:])),
    )


def _support_row(
    graph_indices: Sequence[int],
) -> RowKey:
    support = squared_hafnian_support()
    variables = []
    for color, graph_index in enumerate(graph_indices):
        graph = support[int(graph_index)]
        for edge, exponent in enumerate(graph):
            if not exponent:
                continue
            left, right = K6_EDGES[edge]
            variables.extend(
                [variable_index(
                    6,
                    3,
                    left,
                    right,
                    color,
                    color,
                )]
                * int(exponent)
            )
    if len(variables) != 18:
        raise KrennOneShellFullRecomputeError(
            "a D^2 support row does not have degree eighteen"
        )
    return bytes(sorted(variables))


def _mixed_generator_terms() -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]], ...
]:
    system = generate_sparse_system(6, 3)
    records = []
    for coloring in product(range(3), repeat=6):
        if len(set(coloring)) == 1:
            continue
        equation = coloring_index(6, 3, coloring)
        for term in system.equation_monomials(equation):
            monomial = tuple(sorted(map(int, term)))
            if len(monomial) != 3 or len(set(monomial)) != 3:
                raise KrennOneShellFullRecomputeError(
                    "a mixed generator term is not squarefree cubic"
                )
            records.append((tuple(coloring), monomial))
    expected = (
        SCOPE_COUNTS["mixed_colorings"]
        * SCOPE_COUNTS["matching_terms_per_generator"]
    )
    if len(records) != expected:
        raise KrennOneShellFullRecomputeError(
            "the mixed generator-term census changed"
        )
    return tuple(records)


def _factor_columns(
    row: RowKey,
    generator_terms: Sequence[
        tuple[tuple[int, ...], tuple[int, ...]]
    ],
) -> set[Column]:
    available = Counter(row)
    columns = set()
    for coloring, term in generator_terms:
        if any(available[variable] == 0 for variable in term):
            continue
        residual = available.copy()
        residual.subtract(term)
        multiplier = tuple(sorted(residual.elements()))
        if len(multiplier) != 15:
            raise KrennOneShellFullRecomputeError(
                "division by a generator term produced the wrong degree"
            )
        columns.add((coloring, multiplier))
    return columns


def _column_outputs(
    column: Column,
) -> tuple[RowKey, ...]:
    coloring, multiplier = column
    system = generate_sparse_system(6, 3)
    equation = coloring_index(6, 3, coloring)
    outputs = tuple(
        bytes(sorted((*multiplier, *term)))
        for term in system.equation_monomials(equation)
    )
    if len(outputs) != 15 or any(
        len(output) != 18 for output in outputs
    ):
        raise KrennOneShellFullRecomputeError(
            "a one-shell column does not have fifteen degree-18 outputs"
        )
    return outputs


def _classify_row_orbits(
    raw_rows: set[RowKey],
    group: GroupArrays,
    logger: RunLogger,
) -> tuple[
    dict[RowKey, RowKey],
    dict[RowKey, int],
]:
    pending = set(raw_rows)
    raw_to_canonical: dict[RowKey, RowKey] = {}
    orbit_sizes: dict[RowKey, int] = {}
    while pending:
        seed = min(pending)
        variables = np.frombuffer(seed, dtype=np.uint8)
        transported = np.sort(
            group.variable_permutations[:, variables],
            axis=1,
        )
        orbit = np.unique(transported, axis=0)
        orbit_keys = tuple(
            row.tobytes() for row in orbit
        )
        canonical = min(orbit_keys)
        hits = pending.intersection(orbit_keys)
        if not hits:
            raise KrennOneShellFullRecomputeError(
                "a row orbit failed to contain its seed"
            )
        for key in hits:
            raw_to_canonical[key] = canonical
        pending.difference_update(hits)
        previous = orbit_sizes.setdefault(canonical, len(orbit_keys))
        if previous != len(orbit_keys):
            raise KrennOneShellFullRecomputeError(
                "a canonical row received inconsistent orbit sizes"
            )
        orbit_count = len(orbit_sizes)
        if orbit_count % 2_000 == 0:
            logger.write(
                "row orbits "
                f"{orbit_count:,}; raw rows pending {len(pending):,}"
            )
    if len(raw_to_canonical) != len(raw_rows):
        raise KrennOneShellFullRecomputeError(
            "the row-orbit map did not cover every raw output"
        )
    return raw_to_canonical, orbit_sizes


def _column_orbit_fingerprint(
    columns: Iterable[Column],
) -> str:
    encoded = []
    for coloring, multiplier in columns:
        encoded.append(
            bytes((*coloring, 255, *multiplier)) + b"\n"
        )
    return sha256(b"".join(sorted(encoded))).hexdigest()


def _sparse_signature_fingerprint(
    signatures: Iterable[RowSignature],
) -> str:
    encoded = []
    for signature in signatures:
        column = b"".join(
            row + int(coefficient).to_bytes(2, "big")
            for row, coefficient in signature
        )
        encoded.append(column + b"\n")
    return sha256(b"".join(sorted(encoded))).hexdigest()


def _modular_rank(
    signatures: Sequence[IndexedSignature],
    rhs: Mapping[int, int],
    prime: int,
) -> tuple[int, int, int]:
    pivots: dict[int, dict[int, int]] = {}
    for signature in signatures:
        vector = {
            row: coefficient % prime
            for row, coefficient in signature
            if coefficient % prime
        }
        while vector:
            pivot = min(vector)
            existing = pivots.get(pivot)
            if existing is None:
                inverse = pow(vector[pivot], -1, prime)
                pivots[pivot] = {
                    row: value * inverse % prime
                    for row, value in vector.items()
                }
                break
            factor = vector[pivot]
            for row, value in existing.items():
                updated = (
                    vector.get(row, 0) - factor * value
                ) % prime
                if updated:
                    vector[row] = updated
                else:
                    vector.pop(row, None)

    residual = {
        row: value % prime
        for row, value in rhs.items()
        if value % prime
    }
    while residual:
        pivot = min(residual)
        existing = pivots.get(pivot)
        if existing is None:
            break
        factor = residual[pivot]
        for row, value in existing.items():
            updated = (
                residual.get(row, 0) - factor * value
            ) % prime
            if updated:
                residual[row] = updated
            else:
                residual.pop(row, None)
    rank = len(pivots)
    return rank, rank + int(bool(residual)), len(residual)


def recompute_full_one_shell(
    logger: RunLogger,
) -> FullRecomputeResult:
    """Construct and rank the deterministic exhaustive one-shell matrix."""

    started = time.perf_counter()
    logger.write("building vectorized S6 x S3 action")
    group = _build_group_arrays()

    support_records = d_squared_support_orbits()
    support_rows = tuple(
        _support_row(record.graph_indices)
        for record in support_records
    )
    if len(support_rows) != SCOPE_COUNTS["D_squared_support_orbits"]:
        raise KrennOneShellFullRecomputeError(
            "the D^2 support representative census changed"
        )

    generator_terms = _mixed_generator_terms()
    raw_columns: set[Column] = set()
    for index, row in enumerate(support_rows, start=1):
        raw_columns.update(_factor_columns(row, generator_terms))
        if index % 100 == 0:
            logger.write(
                "support representatives "
                f"{index:,}/{len(support_rows):,}; "
                f"raw columns {len(raw_columns):,}"
            )
    if len(raw_columns) != SCOPE_COUNTS[
        "raw_columns_hitting_support_representatives"
    ]:
        raise KrennOneShellFullRecomputeError(
            "the exhaustive support-division column census changed"
        )
    logger.write(
        f"exhaustive division produced {len(raw_columns):,} raw columns"
    )

    raw_column_outputs = {
        column: _column_outputs(column)
        for column in sorted(raw_columns)
    }
    raw_rows = {
        row
        for outputs in raw_column_outputs.values()
        for row in outputs
    }
    if len(raw_rows) != SCOPE_COUNTS["output_raw_rows"]:
        raise KrennOneShellFullRecomputeError(
            "the raw one-shell output-row census changed"
        )
    logger.write(
        f"complete polynomial outputs produced {len(raw_rows):,} raw rows"
    )

    raw_to_canonical, orbit_sizes = _classify_row_orbits(
        raw_rows, group, logger
    )
    row_representatives = tuple(sorted(orbit_sizes))
    if len(row_representatives) != SCOPE_COUNTS[
        "output_row_orbits"
    ]:
        raise KrennOneShellFullRecomputeError(
            "the deterministic output-row orbit census changed"
        )

    canonical_signatures: dict[Column, RowSignature] = {}
    for index, raw_column in enumerate(
        sorted(raw_column_outputs),
        start=1,
    ):
        canonical_column = _canonical_column(raw_column, group)
        signature = tuple(sorted(Counter(
            raw_to_canonical[row]
            for row in raw_column_outputs[raw_column]
        ).items()))
        previous = canonical_signatures.setdefault(
            canonical_column, signature
        )
        if previous != signature:
            raise KrennOneShellFullRecomputeError(
                "one column orbit produced two different row signatures"
            )
        if index % 1_000 == 0:
            logger.write(
                "canonicalized raw columns "
                f"{index:,}/{len(raw_column_outputs):,}; "
                f"column orbits {len(canonical_signatures):,}"
            )

    columns = tuple(sorted(canonical_signatures))
    row_signatures = tuple(
        canonical_signatures[column] for column in columns
    )
    if (
        len(columns) != SCOPE_COUNTS["S6_x_S3_column_orbits"]
        or len(set(row_signatures))
        != SCOPE_COUNTS["complete_polynomial_output_signatures"]
    ):
        raise KrennOneShellFullRecomputeError(
            "the column-orbit or complete-signature census changed"
        )

    column_fingerprint = _column_orbit_fingerprint(columns)
    signature_fingerprint = _sparse_signature_fingerprint(
        row_signatures
    )
    if (
        column_fingerprint != EXPECTED_COLUMN_ORBIT_FINGERPRINT
        or signature_fingerprint
        != EXPECTED_SPARSE_SIGNATURE_FINGERPRINT
    ):
        raise KrennOneShellFullRecomputeError(
            "a deterministic one-shell fingerprint changed"
        )
    logger.write(
        "column fingerprint " + column_fingerprint
    )
    logger.write(
        "sparse-signature fingerprint " + signature_fingerprint
    )

    row_index = {
        row: index
        for index, row in enumerate(row_representatives)
    }
    indexed_signatures = tuple(
        tuple(
            (row_index[row], coefficient)
            for row, coefficient in signature
        )
        for signature in row_signatures
    )
    matrix_nonzeros = sum(map(len, indexed_signatures))
    if matrix_nonzeros != SCOPE_COUNTS["matrix_nonzeros"]:
        raise KrennOneShellFullRecomputeError(
            "the deterministic sparse matrix nonzero count changed"
        )

    rhs_by_row: Counter[RowKey] = Counter()
    for record, row in zip(
        support_records, support_rows, strict=True
    ):
        canonical = raw_to_canonical.get(row)
        if canonical is None:
            raise KrennOneShellFullRecomputeError(
                "a D^2 support row is absent from one-shell outputs"
            )
        if orbit_sizes[canonical] != record.orbit_size:
            raise KrennOneShellFullRecomputeError(
                "a D^2 support row orbit size changed"
            )
        rhs_by_row[canonical] += (
            record.orbit_size * record.coefficient
        )
    if len(rhs_by_row) != SCOPE_COUNTS["D_squared_support_orbits"]:
        raise KrennOneShellFullRecomputeError(
            "the compressed D^2 RHS orbit census changed"
        )
    rhs = {
        row_index[row]: value
        for row, value in rhs_by_row.items()
    }

    rank_records = []
    for prime in PRIMES:
        rank_started = time.perf_counter()
        rank, augmented_rank, residual_nonzeros = _modular_rank(
            indexed_signatures, rhs, prime
        )
        record = ModularRankRecord(
            prime=prime,
            matrix_rank=rank,
            augmented_rank=augmented_rank,
            rhs_residual_nonzeros=residual_nonzeros,
            seconds=time.perf_counter() - rank_started,
        )
        rank_records.append(record)
        logger.write(
            f"prime {prime:,}: rank {rank:,}, "
            f"augmented rank {augmented_rank:,}, "
            f"RHS residual nnz {residual_nonzeros:,} "
            "(nonproof-reconnaissance)"
        )

    result = FullRecomputeResult(
        raw_column_count=len(raw_columns),
        raw_row_count=len(raw_rows),
        row_representatives=row_representatives,
        columns=columns,
        signatures=indexed_signatures,
        rhs_entries=tuple(sorted(rhs.items())),
        matrix_nonzeros=matrix_nonzeros,
        column_orbit_fingerprint=column_fingerprint,
        sparse_signature_fingerprint=signature_fingerprint,
        modular_ranks=tuple(rank_records),
        elapsed_seconds=time.perf_counter() - started,
    )
    logger.write(
        f"full deterministic one-shell recompute completed in "
        f"{result.elapsed_seconds:.3f} seconds"
    )
    return result


def _cache_payload(result: FullRecomputeResult) -> dict:
    return {
        "schema": FULL_CACHE_SCHEMA,
        "ordering": {
            "rows": "lexicographic canonical row bytes",
            "columns": (
                "lexicographic canonical coloring/multiplier pairs"
            ),
            "entries": "ascending row index within each column",
        },
        "shape": [
            len(result.row_representatives),
            len(result.columns),
        ],
        "nonzeros": result.matrix_nonzeros,
        "row_representatives_hex": [
            row.hex() for row in result.row_representatives
        ],
        "columns": [
            {
                "coloring": list(column[0]),
                "multiplier_variable_indices": list(column[1]),
                "entries": [
                    [row, coefficient]
                    for row, coefficient in signature
                ],
            }
            for column, signature in zip(
                result.columns,
                result.signatures,
                strict=True,
            )
        ],
        "rhs_entries": [
            [row, coefficient]
            for row, coefficient in result.rhs_entries
        ],
        "column_orbit_fingerprint": (
            result.column_orbit_fingerprint
        ),
        "sparse_signature_fingerprint": (
            result.sparse_signature_fingerprint
        ),
    }


def _validate_result_against_small_receipt(
    result: FullRecomputeResult,
) -> dict:
    receipt = one_shell_reconnaissance_receipt()
    verify_one_shell_reconnaissance_receipt(receipt)
    scope = receipt["scope"]["counts"]
    layout = receipt["deterministic_sparse_layout"]
    ranks = receipt["modular_rank_diagnostics"]["records"]
    if (
        scope["raw_columns_hitting_support_representatives"]
        != result.raw_column_count
        or scope["output_raw_rows"] != result.raw_row_count
        or scope["output_row_orbits"]
        != len(result.row_representatives)
        or scope["S6_x_S3_column_orbits"] != len(result.columns)
        or scope["matrix_nonzeros"] != result.matrix_nonzeros
        or layout["column_orbit_fingerprint"]
        != result.column_orbit_fingerprint
        or layout["sparse_signature_fingerprint"]
        != result.sparse_signature_fingerprint
        or tuple(
            (
                record["prime"],
                record["matrix_rank"],
                record["augmented_rank"],
            )
            for record in ranks
        )
        != tuple(
            (
                record.prime,
                record.matrix_rank,
                record.augmented_rank,
            )
            for record in result.modular_ranks
        )
    ):
        raise KrennOneShellFullRecomputeError(
            "the full recompute does not match the small receipt"
        )
    return receipt


def run_full_and_write(
    *,
    receipt_path: Path,
    report_path: Path,
    cache_path: Path | None,
    log_path: Path | None,
) -> FullRecomputeResult:
    """Run the expensive construction and write deterministic outputs."""

    with RunLogger(log_path) as logger:
        logger.write(
            "starting exhaustive deterministic one-shell recompute"
        )
        result = recompute_full_one_shell(logger)
        receipt = _validate_result_against_small_receipt(result)
        _write_json(receipt_path, receipt)
        logger.write(f"wrote small receipt: {receipt_path}")

        report = result.report()
        if cache_path is not None:
            cache_json_hash, cache_gzip_hash = (
                _write_deterministic_gzip_json(
                    cache_path, _cache_payload(result)
                )
            )
            report["deterministic_sparse_cache"] = {
                "path": str(cache_path),
                "uncompressed_canonical_json_sha256": (
                    cache_json_hash
                ),
                "gzip_sha256": cache_gzip_hash,
            }
            logger.write(f"wrote sparse cache: {cache_path}")
            logger.write(
                "cache canonical JSON fingerprint "
                + cache_json_hash
            )
        else:
            report["deterministic_sparse_cache"] = {
                "written": False,
            }
        _write_json(report_path, report)
        logger.write(f"wrote full recompute report: {report_path}")
    return result


def static_validation() -> Mapping[str, object]:
    """Replay only the small exact receipt; never build the full matrix."""

    receipt = one_shell_reconnaissance_receipt()
    checks = verify_one_shell_reconnaissance_receipt(receipt)
    return {
        "schema": FULL_RECOMPUTE_SCHEMA,
        "mode": "static-validation",
        "full_matrix_recomputed": False,
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "column_orbit_fingerprint": (
            receipt["deterministic_sparse_layout"][
                "column_orbit_fingerprint"
            ]
        ),
        "sparse_signature_fingerprint": (
            receipt["deterministic_sparse_layout"][
                "sparse_signature_fingerprint"
            ]
        ),
        "exact_shell_separator_replayed": checks[
            "exact_shell_separator_replayed"
        ],
        "explicit_global_escapes_replayed": checks[
            "explicit_global_escapes_replayed"
        ],
        "claims": receipt["claims"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically recompute the exhaustive k=2 one-shell "
            "matrix, or perform the default small static validation."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--run-full",
        action="store_true",
        help=(
            "opt in to the expensive 39,033 by 4,493 construction "
            "and three modular eliminations"
        ),
    )
    modes.add_argument(
        "--static-validate",
        action="store_true",
        help=(
            "replay only the small receipt (this is also the default)"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_SCRATCH_DIRECTORY,
        help=(
            "directory for receipt/cache/report/log outputs "
            f"(default: {DEFAULT_SCRATCH_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--receipt-name",
        default=DEFAULT_RECEIPT_NAME,
        help="small receipt filename within the output directory",
    )
    parser.add_argument(
        "--report-name",
        default=DEFAULT_REPORT_NAME,
        help="full recompute report filename within the output directory",
    )
    parser.add_argument(
        "--cache-name",
        default=DEFAULT_CACHE_NAME,
        help="gzip sparse-cache filename within the output directory",
    )
    parser.add_argument(
        "--log-name",
        default=DEFAULT_LOG_NAME,
        help="progress-log filename within the output directory",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not write the large deterministic sparse cache",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="do not write a progress log",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.run_full:
        print(json.dumps(
            static_validation(),
            indent=2,
            sort_keys=True,
        ))
        return 0

    output_directory = args.output_directory.resolve()
    receipt_path = output_directory / args.receipt_name
    report_path = output_directory / args.report_name
    cache_path = (
        None
        if args.no_cache
        else output_directory / args.cache_name
    )
    log_path = (
        None
        if args.no_log
        else output_directory / args.log_name
    )
    run_full_and_write(
        receipt_path=receipt_path,
        report_path=report_path,
        cache_path=cache_path,
        log_path=log_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
