r"""Small exact obstruction searches for ``D^2 in J_mix``.

The full ``k=2`` source-ideal matrix is far beyond the reviewed construction
bounds.  This module searches the smallest invariant row space that can see
the right-hand side: the 1,728,000 monomials in ``supp(D^2)``, compressed to
663 ``S_6 x S_3`` orbits.

The result is a rigorous negative search result, not radical membership:

* 632 support orbits contain a rainbow perfect matching.  A ``2+2+2`` mixed
  generator gives a column whose only output still in ``supp(D^2)`` is that
  row, so an annihilating functional supported on ``supp(D^2)`` must vanish
  there.
* The other 31 orbits are killed by an exact full-rank system of ``4+2``
  matching-switch constraints.  A deterministic 31-row integer minor is
  retained as the compact certificate.

Consequently there is no nonzero invariant dual functional supported only on
``supp(D^2)``.  Reynolds averaging then rules out every support-limited dual
that pairs nontrivially with ``D^2``.  A valid global dual may still use
codomain monomials outside ``supp(D^2)``, and an identity for ``D^2`` may use
columns outside the one-shell search.  Nothing here decides
``D^2 in J_mix`` or ``D in radical(J_mix)``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import permutations
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.higher_power_source_ideal import (
    COLOR_COUNT,
    EXPECTED_D_SQUARED_SUPPORT_ORBITS,
    GROUP_ORDER,
    K6_EDGES,
    K6_EDGE_INDEX,
    KrennHigherPowerError,
    _support_state_encode,
    d_squared_support_orbit_index,
    d_squared_support_orbits,
    k6_perfect_matchings,
    squared_hafnian_support,
)
from experiments.krenn_quantum_graph.system import (
    coloring_index,
    generate_sparse_system,
    variable_index,
    variable_key,
)


RADICAL_OBSTRUCTION_SCHEMA = (
    "krenn.n6_d3.radical_obstruction.support_rows.v1"
)

EXPECTED_RAINBOW_SUPPORT_ORBITS = 632
EXPECTED_NONRAINBOW_SUPPORT_ORBITS = 31
EXPECTED_NONRAINBOW_RAW_MONOMIALS = 34_560
EXPECTED_SWITCH_PREDECESSORS = 291
EXPECTED_UNIQUE_SWITCH_CONSTRAINTS = 56
EXPECTED_SWITCH_CONSTRAINT_RANK = 31
CONTRACTION_PRIMES = (31, 1_009, 1_000_003)
EXPECTED_CONTRACTION_RANKS = {1: 71, 2: 81, 3: 81}


class KrennRadicalObstructionError(ValueError):
    """A support-orbit, switch constraint, or exact receipt is malformed."""


def _perfect_matchings(
    vertices: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not vertices:
        return ((),)
    first = vertices[0]
    result = []
    for partner_index in range(1, len(vertices)):
        partner = vertices[partner_index]
        remaining = (
            vertices[1:partner_index]
            + vertices[partner_index + 1:]
        )
        edge = tuple(sorted((first, partner)))
        for tail in _perfect_matchings(remaining):
            result.append(tuple(sorted((edge, *tail))))
    return tuple(sorted(result))


def _rainbow_witness(
    graph_indices: Sequence[int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    graphs = tuple(
        squared_hafnian_support()[int(index)]
        for index in graph_indices
    )
    for matching in k6_perfect_matchings():
        for edge_positions in permutations(range(COLOR_COUNT)):
            if all(
                graphs[color][matching[edge_positions[color]]] > 0
                for color in range(COLOR_COUNT)
            ):
                return matching, tuple(edge_positions)
    return None


@dataclass(frozen=True)
class SupportOrbitPartition:
    rainbow_orbit_indices: tuple[int, ...]
    nonrainbow_orbit_indices: tuple[int, ...]
    rainbow_witnesses: tuple[
        tuple[tuple[int, int, int], tuple[int, int, int]], ...
    ]
    nonrainbow_raw_monomials: int

    def __post_init__(self) -> None:
        if (
            len(self.rainbow_orbit_indices)
            != EXPECTED_RAINBOW_SUPPORT_ORBITS
            or len(self.nonrainbow_orbit_indices)
            != EXPECTED_NONRAINBOW_SUPPORT_ORBITS
            or len(self.rainbow_witnesses)
            != len(self.rainbow_orbit_indices)
            or set(self.rainbow_orbit_indices).intersection(
                self.nonrainbow_orbit_indices
            )
            or tuple(sorted((
                *self.rainbow_orbit_indices,
                *self.nonrainbow_orbit_indices,
            )))
            != tuple(range(EXPECTED_D_SQUARED_SUPPORT_ORBITS))
            or self.nonrainbow_raw_monomials
            != EXPECTED_NONRAINBOW_RAW_MONOMIALS
        ):
            raise KrennRadicalObstructionError(
                "the D^2 support rainbow partition is malformed"
            )


@lru_cache(maxsize=1)
def support_orbit_partition() -> SupportOrbitPartition:
    rainbow = []
    nonrainbow = []
    witnesses = []
    nonrainbow_raw = 0
    for orbit_index, record in enumerate(d_squared_support_orbits()):
        witness = _rainbow_witness(record.graph_indices)
        if witness is None:
            nonrainbow.append(orbit_index)
            nonrainbow_raw += record.orbit_size
        else:
            rainbow.append(orbit_index)
            witnesses.append(witness)
    return SupportOrbitPartition(
        rainbow_orbit_indices=tuple(rainbow),
        nonrainbow_orbit_indices=tuple(nonrainbow),
        rainbow_witnesses=tuple(witnesses),
        nonrainbow_raw_monomials=nonrainbow_raw,
    )


def _rainbow_column(
    orbit_index: int,
    witness: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
    ],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    record = d_squared_support_orbits()[int(orbit_index)]
    graphs = [
        list(squared_hafnian_support()[index])
        for index in record.graph_indices
    ]
    matching, edge_positions = witness
    coloring = [None] * 6
    for color, position in enumerate(edge_positions):
        edge = matching[position]
        left, right = K6_EDGES[edge]
        coloring[left] = coloring[right] = color
        graphs[color][edge] -= 1
    if any(color is None for color in coloring):
        raise KrennRadicalObstructionError(
            "a rainbow witness did not color all six vertices"
        )
    return tuple(coloring), _graph_triple_to_monomial(graphs)


@lru_cache(maxsize=1)
def verify_rainbow_singleton_constraints() -> str:
    """Replay all 632 ``2+2+2`` columns against the full 15-term map."""

    partition = support_orbit_partition()
    system = generate_sparse_system(6, 3)
    fingerprint_rows = []
    for orbit_index, witness in zip(
        partition.rainbow_orbit_indices,
        partition.rainbow_witnesses,
        strict=True,
    ):
        coloring, multiplier = _rainbow_column(
            orbit_index, witness
        )
        if sorted(Counter(coloring).values()) != [2, 2, 2]:
            raise KrennRadicalObstructionError(
                "a rainbow singleton coloring is not 2+2+2"
            )
        equation = coloring_index(6, 3, coloring)
        support_hits = []
        for term in system.equation_monomials(equation):
            state = _monomial_support_state((*multiplier, *term))
            if state is not None:
                support_hits.append(
                    d_squared_support_orbit_index(state)
                )
        if support_hits != [orbit_index]:
            raise KrennRadicalObstructionError(
                "a rainbow column is not singleton on supp(D^2)"
            )
        matching, edge_positions = witness
        fingerprint_rows.append((
            orbit_index,
            matching,
            edge_positions,
            coloring,
            multiplier,
        ))
    encoded = json.dumps(
        fingerprint_rows, separators=(",", ":")
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _graph_triple_to_monomial(
    graphs: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    variables = []
    for color, graph in enumerate(graphs):
        for edge, exponent in enumerate(graph):
            left, right = K6_EDGES[edge]
            variables.extend(
                [variable_index(6, 3, left, right, color, color)]
                * int(exponent)
            )
    return tuple(sorted(variables))


def _monomial_support_state(
    monomial: Sequence[int],
) -> int | None:
    graph_exponents = [
        [0] * len(K6_EDGES) for _ in range(COLOR_COUNT)
    ]
    for variable in monomial:
        left, right, left_color, right_color = variable_key(
            6, 3, int(variable)
        )
        if left_color != right_color:
            return None
        edge = K6_EDGE_INDEX[tuple(sorted((left, right)))]
        graph_exponents[left_color][edge] += 1
    support_index = {
        graph: index
        for index, graph in enumerate(squared_hafnian_support())
    }
    graph_indices = []
    for graph in graph_exponents:
        index = support_index.get(tuple(graph))
        if index is None:
            return None
        graph_indices.append(index)
    return _support_state_encode(graph_indices)


@dataclass(frozen=True)
class SwitchConstraint:
    coloring: tuple[int, ...]
    multiplier: tuple[int, ...]
    entries: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        partition = support_orbit_partition()
        if (
            len(self.coloring) != 6
            or sorted(Counter(self.coloring).values()) != [2, 4]
            or len(self.multiplier) != 15
            or tuple(sorted(self.multiplier)) != self.multiplier
            or not self.entries
            or tuple(sorted(self.entries)) != self.entries
            or len({index for index, _value in self.entries})
            != len(self.entries)
            or any(
                index not in range(len(
                    partition.nonrainbow_orbit_indices
                ))
                or value <= 0
                for index, value in self.entries
            )
        ):
            raise KrennRadicalObstructionError(
                "a 4+2 switch constraint is malformed"
            )

    def key(self) -> tuple:
        return self.entries, self.coloring, self.multiplier

    def to_dict(self) -> dict:
        nonrainbow = (
            support_orbit_partition().nonrainbow_orbit_indices
        )
        return {
            "coloring": list(self.coloring),
            "multiplier_variable_indices": list(self.multiplier),
            "multiplier_variable_keys": [
                list(variable_key(6, 3, variable))
                for variable in self.multiplier
            ],
            "projected_entries": [
                [
                    d_squared_support_orbits()[
                        nonrainbow[index]
                    ].representative_state,
                    value,
                ]
                for index, value in self.entries
            ],
        }


def _four_vertex_matching_edge_indices(
    vertices: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(
            K6_EDGE_INDEX[tuple(sorted(edge))]
            for edge in matching
        )
        for matching in _perfect_matchings(tuple(sorted(vertices)))
    )


def _constraint_from_column(
    coloring: tuple[int, ...],
    multiplier: tuple[int, ...],
) -> SwitchConstraint:
    system = generate_sparse_system(6, 3)
    equation = coloring_index(6, 3, coloring)
    nonrainbow = support_orbit_partition().nonrainbow_orbit_indices
    local_index = {
        orbit_index: index
        for index, orbit_index in enumerate(nonrainbow)
    }
    entries: Counter[int] = Counter()
    for term in system.equation_monomials(equation):
        state = _monomial_support_state((*multiplier, *term))
        if state is None:
            continue
        orbit_index = d_squared_support_orbit_index(state)
        if orbit_index in local_index:
            entries[local_index[orbit_index]] += 1
    return SwitchConstraint(
        coloring=coloring,
        multiplier=multiplier,
        entries=tuple(sorted(entries.items())),
    )


@dataclass(frozen=True)
class SwitchConstraintSystem:
    constraints: tuple[SwitchConstraint, ...]
    predecessor_count: int

    def __post_init__(self) -> None:
        if (
            len(self.constraints) != EXPECTED_UNIQUE_SWITCH_CONSTRAINTS
            or self.predecessor_count != EXPECTED_SWITCH_PREDECESSORS
            or tuple(sorted(
                constraint.key() for constraint in self.constraints
            ))
            != tuple(
                constraint.key() for constraint in self.constraints
            )
        ):
            raise KrennRadicalObstructionError(
                "the nonrainbow switch system is malformed"
            )


@lru_cache(maxsize=1)
def switch_constraint_system() -> SwitchConstraintSystem:
    support = squared_hafnian_support()
    nonrainbow = support_orbit_partition().nonrainbow_orbit_indices
    by_entries: dict[
        tuple[tuple[int, int], ...],
        SwitchConstraint,
    ] = {}
    predecessor_count = 0
    for orbit_index in nonrainbow:
        record = d_squared_support_orbits()[orbit_index]
        graphs = [
            list(support[index]) for index in record.graph_indices
        ]
        for majority_color in range(COLOR_COUNT):
            for minority_color in range(COLOR_COUNT):
                if majority_color == minority_color:
                    continue
                for left in range(6):
                    for right in range(left + 1, 6):
                        minority_edge = K6_EDGE_INDEX[(left, right)]
                        if graphs[minority_color][minority_edge] <= 0:
                            continue
                        remaining = tuple(
                            vertex
                            for vertex in range(6)
                            if vertex not in (left, right)
                        )
                        matchings = _four_vertex_matching_edge_indices(
                            remaining
                        )
                        for matching in matchings:
                            if not all(
                                graphs[majority_color][edge] > 0
                                for edge in matching
                            ):
                                continue
                            predecessor_count += 1
                            coloring = [majority_color] * 6
                            coloring[left] = minority_color
                            coloring[right] = minority_color
                            multiplier_graphs = [
                                graph.copy() for graph in graphs
                            ]
                            multiplier_graphs[
                                minority_color
                            ][minority_edge] -= 1
                            for edge in matching:
                                multiplier_graphs[
                                    majority_color
                                ][edge] -= 1
                            multiplier = _graph_triple_to_monomial(
                                multiplier_graphs
                            )
                            constraint = _constraint_from_column(
                                tuple(coloring),
                                multiplier,
                            )
                            old = by_entries.get(constraint.entries)
                            if old is None or constraint.key() < old.key():
                                by_entries[constraint.entries] = constraint
    constraints = tuple(sorted(
        by_entries.values(), key=SwitchConstraint.key
    ))
    return SwitchConstraintSystem(
        constraints=constraints,
        predecessor_count=predecessor_count,
    )


def _rational_rank(rows: Sequence[Sequence[int]]) -> int:
    matrix = [list(map(Fraction, row)) for row in rows]
    if not matrix:
        return 0
    rank = 0
    for column in range(len(matrix[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(matrix))
                if matrix[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [
            value / pivot_value for value in matrix[rank]
        ]
        for row in range(len(matrix)):
            if row == rank or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                left - factor * right
                for left, right in zip(
                    matrix[row], matrix[rank], strict=True
                )
            ]
        rank += 1
    return rank


def _constraint_row(
    constraint: SwitchConstraint,
) -> tuple[int, ...]:
    row = [0] * EXPECTED_NONRAINBOW_SUPPORT_ORBITS
    for index, value in constraint.entries:
        row[index] = value
    return tuple(row)


def _bareiss_determinant(matrix: Sequence[Sequence[int]]) -> int:
    rows = [list(map(int, row)) for row in matrix]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise KrennRadicalObstructionError(
            "Bareiss determinant needs a square matrix"
        )
    if not size:
        return 1
    sign = 1
    denominator = 1
    for pivot_index in range(size - 1):
        pivot_row = next(
            (
                row
                for row in range(pivot_index, size)
                if rows[row][pivot_index]
            ),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            rows[pivot_index], rows[pivot_row] = (
                rows[pivot_row],
                rows[pivot_index],
            )
            sign *= -1
        pivot = rows[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = (
                    rows[row][column] * pivot
                    - rows[row][pivot_index]
                    * rows[pivot_index][column]
                )
                if numerator % denominator:
                    raise KrennRadicalObstructionError(
                        "Bareiss division was not exact"
                    )
                rows[row][column] = numerator // denominator
        denominator = pivot
    return sign * rows[-1][-1]


@dataclass(frozen=True)
class SupportFunctionalCertificate:
    basis_constraints: tuple[SwitchConstraint, ...]
    basis_determinant: int
    full_constraint_rank: int

    def __post_init__(self) -> None:
        if (
            len(self.basis_constraints)
            != EXPECTED_SWITCH_CONSTRAINT_RANK
            or not self.basis_determinant
            or self.full_constraint_rank
            != EXPECTED_SWITCH_CONSTRAINT_RANK
        ):
            raise KrennRadicalObstructionError(
                "the support-functional certificate is malformed"
            )

    @property
    def supported_invariant_annihilator_dimension(self) -> int:
        return 0

    @property
    def D_squared_membership_decided(self) -> bool:
        return False

    def to_dict(self) -> dict:
        partition = support_orbit_partition()
        return {
            "support_orbits": EXPECTED_D_SQUARED_SUPPORT_ORBITS,
            "rainbow_support_orbits": len(
                partition.rainbow_orbit_indices
            ),
            "rainbow_singleton_constraints": len(
                partition.rainbow_witnesses
            ),
            "nonrainbow_support_orbits": len(
                partition.nonrainbow_orbit_indices
            ),
            "nonrainbow_representative_states": [
                d_squared_support_orbits()[
                    orbit_index
                ].representative_state
                for orbit_index in partition.nonrainbow_orbit_indices
            ],
            "nonrainbow_raw_monomials": (
                partition.nonrainbow_raw_monomials
            ),
            "switch_predecessors": (
                switch_constraint_system().predecessor_count
            ),
            "unique_switch_constraints": len(
                switch_constraint_system().constraints
            ),
            "basis_constraints": [
                constraint.to_dict()
                for constraint in self.basis_constraints
            ],
            "matrix_entry_normalization": (
                "number of the 15 raw F_c matching terms landing in "
                "the indicated D^2 support orbit"
            ),
            "basis_determinant": self.basis_determinant,
            "full_constraint_rank_over_Q": self.full_constraint_rank,
            "supported_invariant_annihilator_dimension": (
                self.supported_invariant_annihilator_dimension
            ),
            "Reynolds_consequence": (
                "no dual supported only on supp(D^2) can pair "
                "nontrivially with D^2"
            ),
            "claim_boundary": (
                "This excludes only support-limited row separators. "
                "A global dual may use off-support codomain monomials, "
                "and D^2 membership remains undecided."
            ),
        }


@lru_cache(maxsize=1)
def exact_support_functional_certificate(
) -> SupportFunctionalCertificate:
    constraints = switch_constraint_system().constraints
    rows = [_constraint_row(constraint) for constraint in constraints]
    full_rank = _rational_rank(rows)
    if full_rank != EXPECTED_SWITCH_CONSTRAINT_RANK:
        raise KrennRadicalObstructionError(
            "the nonrainbow switch system lost full rank"
        )
    basis = []
    basis_rows = []
    old_rank = 0
    for constraint, row in zip(constraints, rows, strict=True):
        new_rank = _rational_rank((*basis_rows, row))
        if new_rank > old_rank:
            basis.append(constraint)
            basis_rows.append(row)
            old_rank = new_rank
        if old_rank == EXPECTED_SWITCH_CONSTRAINT_RANK:
            break
    determinant = _bareiss_determinant(basis_rows)
    return SupportFunctionalCertificate(
        basis_constraints=tuple(basis),
        basis_determinant=determinant,
        full_constraint_rank=full_rank,
    )


def verify_support_functional_certificate(
    certificate: SupportFunctionalCertificate | None = None,
) -> SupportFunctionalCertificate:
    """Replay every retained source column and the exact integer minor."""

    expected = exact_support_functional_certificate()
    candidate = expected if certificate is None else certificate
    verify_rainbow_singleton_constraints()
    if candidate != expected:
        raise KrennRadicalObstructionError(
            "the support-functional certificate changed"
        )
    replayed = tuple(
        _constraint_from_column(
            constraint.coloring,
            constraint.multiplier,
        )
        for constraint in candidate.basis_constraints
    )
    if replayed != candidate.basis_constraints:
        raise KrennRadicalObstructionError(
            "a retained switch column failed symbolic replay"
        )
    determinant = _bareiss_determinant(tuple(
        _constraint_row(constraint) for constraint in replayed
    ))
    if determinant != candidate.basis_determinant or not determinant:
        raise KrennRadicalObstructionError(
            "the exact switch minor failed determinant replay"
        )
    return candidate


def _determinant_mod(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> int:
    rows = [
        [int(value) % prime for value in row] for row in matrix
    ]
    determinant = 1
    for column in range(len(rows)):
        pivot = next(
            (
                row
                for row in range(column, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        pivot_value = rows[column][column]
        determinant = determinant * pivot_value % prime
        inverse = pow(pivot_value, -1, prime)
        for row in range(column + 1, len(rows)):
            factor = rows[row][column] * inverse % prime
            if not factor:
                continue
            for entry in range(column, len(rows)):
                rows[row][entry] = (
                    rows[row][entry]
                    - factor * rows[column][entry]
                ) % prime
    return determinant % prime


def _rank_minor_mod(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...], int]:
    rows = [
        [int(value) % prime for value in row] for row in matrix
    ]
    original = tuple(tuple(row) for row in rows)
    row_ids = list(range(len(rows)))
    rank = 0
    pivot_columns = []
    for column in range(len(rows[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        row_ids[rank], row_ids[pivot] = (
            row_ids[pivot],
            row_ids[rank],
        )
        inverse = pow(rows[rank][column], -1, prime)
        rows[rank] = [
            value * inverse % prime for value in rows[rank]
        ]
        for other in range(rank + 1, len(rows)):
            factor = rows[other][column]
            if factor:
                rows[other] = [
                    (left - factor * right) % prime
                    for left, right in zip(
                        rows[other], rows[rank], strict=True
                    )
                ]
        pivot_columns.append(column)
        rank += 1
        if rank == len(rows):
            break
    pivot_rows = tuple(row_ids[:rank])
    pivot_columns = tuple(pivot_columns)
    minor = tuple(
        tuple(
            original[row][column] for column in pivot_columns
        )
        for row in pivot_rows
    )
    determinant = _determinant_mod(minor, prime)
    if not determinant:
        raise KrennRadicalObstructionError(
            "a contraction pivot minor became singular"
        )
    return rank, pivot_rows, pivot_columns, determinant


def _coloring_from_index(
    index: int,
    length: int,
) -> tuple[int, ...]:
    digits = [0] * int(length)
    value = int(index)
    for position in range(length - 1, -1, -1):
        value, digits[position] = divmod(value, COLOR_COUNT)
    return tuple(digits)


def _contraction_jacobian_integer(
    contraction_rank: int,
) -> tuple[tuple[int, ...], ...]:
    contraction_rank = int(contraction_rank)
    if contraction_rank not in (1, 2, 3):
        raise KrennRadicalObstructionError(
            "contraction rank must be 1, 2, or 3"
        )
    system = generate_sparse_system(6, 3)
    values = tuple(
        (
            4
            + 16 * variable
            + 30 * variable * variable
            + 15 * variable * variable * variable
        )
        for variable in range(system.variable_count)
    )
    rows = []
    for output_index in range(COLOR_COUNT**4):
        output_coloring = _coloring_from_index(output_index, 4)
        row = [0] * system.variable_count
        for contracted_color in range(contraction_rank):
            coloring = (
                contracted_color,
                contracted_color,
                *output_coloring,
            )
            equation = coloring_index(6, 3, coloring)
            for monomial in system.equation_monomials(equation):
                for position, variable in enumerate(monomial):
                    term = 1
                    for other_position, other in enumerate(monomial):
                        if other_position != position:
                            term *= values[other]
                    row[variable] += term
        rows.append(tuple(row))
    return tuple(rows)


def _integer_matrix_fingerprint(
    matrix: Sequence[Sequence[int]],
) -> str:
    payload = {
        "shape": [len(matrix), len(matrix[0])],
        "entries": [
            [row_index, column_index, int(value)]
            for row_index, row in enumerate(matrix)
            for column_index, value in enumerate(row)
            if value
        ],
    }
    return sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()


def _phi4_dimension_and_equivariance_certificate() -> Mapping:
    matchings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    gauge_actions = [
        {
            "positive_edge": list(left),
            "negative_edge": list(right),
            "weights": (
                "+1 on all 9 variables of positive_edge, "
                "-1 on all 9 variables of negative_edge, 0 otherwise"
            ),
        }
        for left, right in matchings
    ]
    return {
        "Phi_4_3_source_parameter_count": 54,
        "Phi_4_3_perfect_matchings": [
            [list(edge) for edge in matching]
            for matching in matchings
        ],
        "independent_reciprocal_gauge_actions": gauge_actions,
        "every_Phi_4_3_monomial_has_gauge_weight_zero": True,
        "gauge_action_dimension_on_dense_source_torus": 3,
        "generic_fiber_dimension_lower_bound": 3,
        "affine_image_dimension_upper_bound": 51,
        "local_GL_equivariance": {
            "matching_incidence_law": (
                "every perfect matching has exactly one edge incident "
                "to each physical vertex"
            ),
            "source_lift": (
                "a local GL3 matrix on vertex i acts on the i-color "
                "slot of every edge variable incident to i"
            ),
            "contraction_identity": (
                "C_K((A at vertex 0)(B at vertex 1)T) "
                "= C_(A^T K B)(T)"
            ),
            "rank_normal_form_dependency": (
                "every rank-r 3x3 matrix over characteristic zero is "
                "A^T diag(I_r,0) B for invertible A,B"
            ),
            "rank_strata_representatives_are_sufficient": True,
        },
    }


@lru_cache(maxsize=1)
def two_vertex_contraction_audit() -> Mapping:
    """Exact 52-minor receipts for rank-stratified contractions.

    The larger ranks at three primes are retained only as reconnaissance.
    Each exact conclusion uses a separately replayed integer 52 by 52
    determinant, one row beyond the dimension bound for ``Phi_4,3``.
    """

    rank_records = []
    for contraction_rank in (1, 2, 3):
        integer_matrix = _contraction_jacobian_integer(
            contraction_rank
        )
        prime_records = []
        exact_pivot_rows = None
        exact_pivot_columns = None
        for prime in CONTRACTION_PRIMES:
            rank, pivot_rows, pivot_columns, determinant = (
                _rank_minor_mod(integer_matrix, prime)
            )
            if rank != EXPECTED_CONTRACTION_RANKS[contraction_rank]:
                raise KrennRadicalObstructionError(
                    "a contraction Jacobian rank changed"
                )
            prime_records.append({
                "prime": prime,
                "rank_mod_prime": rank,
                "full_rank_minor_determinant_mod_prime": determinant,
                "role": "nonproof-reconnaissance",
            })
            if prime == CONTRACTION_PRIMES[-1]:
                exact_pivot_rows = pivot_rows[:52]
                exact_pivot_columns = pivot_columns[:52]
        exact_minor = tuple(
            tuple(
                integer_matrix[row][column]
                for column in exact_pivot_columns
            )
            for row in exact_pivot_rows
        )
        exact_determinant = _bareiss_determinant(exact_minor)
        if not exact_determinant:
            raise KrennRadicalObstructionError(
                "the exact contraction minor became singular"
            )
        rank_records.append({
            "contraction_matrix_rank": contraction_rank,
            "representative": (
                "diag(" + ",".join(
                    "1" if index < contraction_rank else "0"
                    for index in range(COLOR_COUNT)
                ) + ")"
            ),
            "integer_jacobian_sha256": (
                _integer_matrix_fingerprint(integer_matrix)
            ),
            "exact_dimension_obstruction": {
                "minor_size": 52,
                "pivot_rows": list(exact_pivot_rows),
                "pivot_columns": list(exact_pivot_columns),
                "integer_determinant": exact_determinant,
                "rank_over_Q_at_least": 52,
            },
            "modular_rank_reconnaissance": prime_records,
        })
    return {
        "contracted_vertices": [0, 1],
        "integer_probe_formula": "4+16*i+30*i^2+15*i^3",
        "jacobian_shape": [81, 135],
        "rank_strata": rank_records,
        "dimension_and_equivariance_certificate": (
            _phi4_dimension_and_equivariance_certificate()
        ),
        "Phi_4_3_affine_image_dimension_upper_bound": 51,
        "all_nonzero_contraction_strata_have_exact_rank_lower_bound_52": (
            True
        ),
        "rank_strata_cover_all_nonzero_3x3_matrices": (
            "GL3 x GL3 equivalence"
        ),
        "universal_nonzero_bilinear_contraction_preserves_image": False,
        "contracted_GHZ_target": "sum_a K_aa * |a^4>",
        "contracted_GHZ_target_witness": {
            "matching_for_color_0": [[0, 1], [2, 3]],
            "matching_for_color_1": [[0, 2], [1, 3]],
            "matching_for_color_2": [[0, 3], [1, 2]],
            "construction": (
                "For color a, set one edge on its displayed matching "
                "to K_aa and the other to 1; set every undisplayed "
                "variable to 0."
            ),
            "symbolically_replayed": True,
        },
        "claim_boundary": (
            "Two-vertex contraction cannot propagate a lower-n "
            "nonimage obstruction; this does not decide GHZ_6,3."
        ),
    }


def _rainbow_witness_fingerprint() -> str:
    partition = support_orbit_partition()
    payload = [
        {
            "orbit_index": orbit_index,
            "matching": list(matching),
            "color_edge_positions": list(edge_positions),
        }
        for orbit_index, (matching, edge_positions) in zip(
            partition.rainbow_orbit_indices,
            partition.rainbow_witnesses,
            strict=True,
        )
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def exact_radical_obstruction_summary() -> Mapping:
    certificate = verify_support_functional_certificate()
    payload = {
        "schema": RADICAL_OBSTRUCTION_SCHEMA,
        "support_row_search": certificate.to_dict(),
        "rainbow_witness_fingerprint": _rainbow_witness_fingerprint(),
        "rainbow_singleton_column_fingerprint": (
            verify_rainbow_singleton_constraints()
        ),
        "propagation_audits": {
            "two_vertex_contraction": (
                two_vertex_contraction_audit()
            ),
        },
        "computations_performed": {
            "full_k2_matrix_constructed": False,
            "support_row_orbits_enumerated": True,
            "exact_integer_minor_replayed": True,
            "exact_contraction_minors_replayed": True,
            "modular_arithmetic_needed_for_exact_claim": False,
        },
        "dependencies": {
            "prior_k1_result": "D not in J_mix at k=1",
            "prior_border_result": (
                "GHZ_6,3 is in the Euclidean and Zariski border image"
            ),
            "prior_support_conditional_result": (
                "retaining all nine natural seed coordinates forces "
                "total support at least 22; the six minimal "
                "support-21 terminals are exactly excluded"
            ),
        },
        "claims": {
            "support_limited_D_squared_separator_exists": False,
            "D_squared_in_J_mix_decided": False,
            "D_in_radical_J_mix_decided": False,
            "new_support_conditional_GHZ_nonexistence_proved": False,
            "global_GHZ_nonexistence_proved": False,
            "exact_affine_GHZ_membership_decided": False,
            "evidence_status": "undecided",
        },
    }
    return payload


def main() -> int:
    print(json.dumps(
        exact_radical_obstruction_summary(),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
