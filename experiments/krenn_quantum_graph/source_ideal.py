r"""Exact fine-graded ``D in J_mix`` experiment for ``n=6,d=3``.

Write ``F_c`` for the homogeneous perfect-matching coefficient indexed by a
coloring ``c``.  This module studies the single, bounded ideal-membership
question

.. math::

    D = F_{000000}F_{111111}F_{222222}
      \stackrel{?}{\in}
    J_{\rm mix} = \langle F_c : c\text{ mixed}\rangle .

All generators have ordinary degree three and ``D`` has degree nine.  The
fine ``Z^(6*3)`` grading therefore reduces every possible degree-nine identity
to degree-six multipliers of one exact complementary multidegree.

The raw linear map has 4,385,040 columns and 11,608,920 rows.  Averaging over
``S_6 x S_3`` is lossless over ``Q`` and compresses it to 1,314 domain orbits
and 3,102 codomain orbits.  The primary result here is an exact two-row
integer dual obstruction.  Modular ranks are retained only as independently
repeatable diagnostics; they are not used as characteristic-zero evidence.

The expensive full compressed matrix and modular elimination are opt-in and
cacheable.  Routine semantic checks can replay the exact two-row obstruction
without constructing the whole matrix.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import permutations, product
import json
from math import comb
import os
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

from experiments.krenn_quantum_graph.system import (
    Monomial,
    coloring_index,
    generate_sparse_system,
    variable_index,
    variable_key,
)


N = 6
D = 3
VARIABLE_COUNT = 135
GROUP_ORDER = 4_320
MIXED_COLORING_COUNT = 726
PERFECT_MATCHING_COUNT = 15

EXPECTED_MULTIPLIERS_PER_COLORING = 6_040
EXPECTED_RAW_DOMAIN_COLUMNS = 4_385_040
EXPECTED_DOMAIN_ORBITS = 1_314
EXPECTED_RAW_CODOMAIN_MONOMIALS = 11_608_920
EXPECTED_CODOMAIN_ORBITS = 3_102
EXPECTED_D_MONOMIALS = 3_375
EXPECTED_D_ORBITS = 8
EXPECTED_COMPRESSED_NNZ = 16_343
EXPECTED_COMPRESSED_FINGERPRINT = (
    "296b0868df8e84f1ac38b4ea7aea5af4"
    "65d838ae81a774a5a7a962b2f9002a9e"
)

DEFAULT_MODULAR_PRIMES = (31, 1_009, 1_000_003)
EXPECTED_MODULAR_MATRIX_RANK = 1_193
EXPECTED_MODULAR_AUGMENTED_RANK = 1_194

SOURCE_IDEAL_SCHEMA = "krenn.n6_d3.source_ideal.k1.v1"
MODULAR_CACHE_SCHEMA = "krenn.n6_d3.source_ideal.modular_cache.v1"

ALL_EQUAL_COLORINGS = (
    (0, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2),
)

# Lexicographically least coloring in each S_6 x S_3 mixed-color orbit.
# The third entry is the full coloring-orbit size; the fourth is the expected
# number of stabilizer orbits on the 6,040 multiplier monomials.
OCCUPATION_REPRESENTATIVES = (
    ("5+1", (0, 0, 0, 0, 0, 1), 36, 70),
    ("4+2", (0, 0, 0, 0, 1, 1), 90, 172),
    ("4+1+1", (0, 0, 0, 0, 1, 2), 90, 188),
    ("3+3", (0, 0, 0, 1, 1, 1), 60, 126),
    ("3+2+1", (0, 0, 0, 1, 1, 2), 360, 556),
    ("2+2+2", (0, 0, 1, 1, 2, 2), 90, 202),
)

EXPECTED_OCCUPATION_COLORINGS = {
    "5+1": 36,
    "4+2": 90,
    "3+3": 60,
    "4+1+1": 90,
    "3+2+1": 360,
    "2+2+2": 90,
}

MULTIPLIER_GRAPH_TYPE_COUNTS = {
    "6-cycle": 3_840,
    "4-cycle+double-edge": 1_440,
    "two-triangles": 640,
    "three-double-edges": 120,
}

# These two canonical codomain monomials give the exact integer dual.
DUAL_ROW_KEYS = (
    (0, 4, 17, 62, 90, 94, 117, 121, 134),
    (0, 4, 17, 62, 90, 103, 112, 117, 134),
)
EXPECTED_DUAL_ROW_INDICES = (550, 568)
EXPECTED_DUAL_ROW_ORBIT_SIZES = (360, 1_080)
EXPECTED_INTEGER_DUAL_RHS = -720


class KrennSourceIdealError(ValueError):
    """A grading, orbit census, compressed map, or cache is malformed."""


@lru_cache(maxsize=1)
def _system():
    return generate_sparse_system(N, D)


def _validate_coloring(coloring: Sequence[int]) -> tuple[int, ...]:
    result = tuple(map(int, coloring))
    if len(result) != N or any(color not in range(D) for color in result):
        raise KrennSourceIdealError("expected a coloring in range(3)^6")
    return result


def _odd_double_factorial(value: int) -> int:
    if value == -1:
        return 1
    if value < -1 or value % 2 == 0:
        raise KrennSourceIdealError("odd double factorial input is invalid")
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return result


def variable_fine_degree(index: int) -> tuple[int, ...]:
    """Return ``e_(i,a) + e_(j,b)`` in the canonical 18-entry order."""

    i, j, a, b = variable_key(N, D, int(index))
    degree = [0] * (N * D)
    degree[i * D + a] += 1
    degree[j * D + b] += 1
    return tuple(degree)


def coloring_fine_degree(coloring: Sequence[int]) -> tuple[int, ...]:
    """Return the fine degree of the homogeneous coefficient ``F_c``."""

    coloring = _validate_coloring(coloring)
    degree = [0] * (N * D)
    for vertex, color in enumerate(coloring):
        degree[vertex * D + color] = 1
    return tuple(degree)


def target_fine_degree() -> tuple[int, ...]:
    """Return the fine degree of ``D``: one of every vertex-color token."""

    return (1,) * (N * D)


def multiplier_fine_degree(coloring: Sequence[int]) -> tuple[int, ...]:
    """Return ``deg(D)-deg(F_c)``, the only relevant multiplier degree."""

    generator = coloring_fine_degree(coloring)
    return tuple(1 - entry for entry in generator)


def monomial_fine_degree(monomial: Sequence[int]) -> tuple[int, ...]:
    degree = [0] * (N * D)
    for variable in monomial:
        for position, entry in enumerate(variable_fine_degree(variable)):
            degree[position] += entry
    return tuple(degree)


def monomial_mask(monomial: Sequence[int]) -> int:
    """Encode a squarefree monomial as a deterministic 135-bit integer."""

    mask = 0
    for variable in monomial:
        variable = int(variable)
        if not 0 <= variable < VARIABLE_COUNT:
            raise KrennSourceIdealError("monomial variable is out of range")
        bit = 1 << variable
        if mask & bit:
            raise KrennSourceIdealError(
                "fine-graded multiplier unexpectedly repeats a variable"
            )
        mask |= bit
    return mask


def _multiplier_tokens(
    coloring: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    coloring = _validate_coloring(coloring)
    return tuple(
        (vertex, color)
        for vertex in range(N)
        for color in range(D)
        if color != coloring[vertex]
    )


@lru_cache(maxsize=None)
def enumerate_multiplier_monomials(
    coloring: tuple[int, ...],
) -> tuple[Monomial, ...]:
    """Enumerate the 6,040 complementary degree-six monomials.

    The lexicographically first unused vertex-color token is paired with every
    later token at a distinct vertex.  Sorting the six canonical variable
    indices gives a unique commutative-monomial encoding.
    """

    coloring = _validate_coloring(coloring)
    tokens = _multiplier_tokens(coloring)
    result: list[Monomial] = []

    def recurse(
        remaining: tuple[tuple[int, int], ...],
        variables: tuple[int, ...],
    ) -> None:
        if not remaining:
            result.append(tuple(sorted(variables)))
            return
        i, a = remaining[0]
        for partner in range(1, len(remaining)):
            j, b = remaining[partner]
            if i == j:
                continue
            recurse(
                remaining[1:partner] + remaining[partner + 1 :],
                variables + (variable_index(N, D, i, j, a, b),),
            )

    recurse(tokens, ())
    result.sort()
    monomials = tuple(result)
    if (
        len(monomials) != EXPECTED_MULTIPLIERS_PER_COLORING
        or len(set(monomials)) != len(monomials)
        or any(len(monomial) != 6 for monomial in monomials)
        or any(monomial_fine_degree(monomial) != multiplier_fine_degree(coloring)
               for monomial in monomials)
    ):
        raise KrennSourceIdealError(
            "fine-graded multiplier enumeration failed"
        )
    return monomials


def multiplier_count_by_inclusion_exclusion() -> int:
    """Count perfect matchings of ``K_(2,2,2,2,2,2)`` exactly."""

    return sum(
        (-1) ** selected_vertices
        * comb(N, selected_vertices)
        * _odd_double_factorial(11 - 2 * selected_vertices)
        for selected_vertices in range(N + 1)
    )


def occupation_type(coloring: Sequence[int]) -> str:
    coloring = _validate_coloring(coloring)
    counts = sorted(Counter(coloring).values(), reverse=True)
    return "+".join(map(str, counts))


def mixed_occupation_census() -> Mapping[str, int]:
    census = Counter(
        occupation_type(coloring)
        for coloring in product(range(D), repeat=N)
        if len(set(coloring)) > 1
    )
    result = dict(sorted(census.items()))
    if result != EXPECTED_OCCUPATION_COLORINGS:
        raise KrennSourceIdealError("mixed occupation census changed")
    return result


def raw_domain_column_count() -> int:
    count = (
        sum(mixed_occupation_census().values())
        * multiplier_count_by_inclusion_exclusion()
    )
    if count != EXPECTED_RAW_DOMAIN_COLUMNS:
        raise KrennSourceIdealError("raw domain-column census changed")
    return count


def raw_codomain_monomial_count() -> int:
    """Count the ``delta``-graded degree-nine monomials."""

    count = sum(
        (-1) ** selected_vertices
        * comb(N, selected_vertices)
        * (3**selected_vertices)
        * _odd_double_factorial(17 - 2 * selected_vertices)
        for selected_vertices in range(N + 1)
    )
    if count != EXPECTED_RAW_CODOMAIN_MONOMIALS:
        raise KrennSourceIdealError("raw codomain census changed")
    return count


def compression_convention() -> Mapping[str, str]:
    """Describe the integer orbit-total coordinates used by ``B*x=b``."""

    return {
        "codomain_row": (
            "sum of all raw equations in one codomain monomial orbit"
        ),
        "domain_coordinate": (
            "one common coefficient on every raw (coloring, multiplier) "
            "basis element in one domain orbit"
        ),
        "matrix_entry": (
            "total raw incidences from a domain orbit into a codomain orbit"
        ),
        "rhs_entry": (
            "sum of D coefficients across the codomain orbit"
        ),
    }


def _transport_coloring(
    coloring: tuple[int, ...],
    vertex_permutation: tuple[int, ...],
    color_permutation: tuple[int, ...],
) -> tuple[int, ...]:
    transported = [0] * N
    for old_vertex, color in enumerate(coloring):
        transported[vertex_permutation[old_vertex]] = color_permutation[color]
    return tuple(transported)


def _transport_variable(
    index: int,
    vertex_permutation: tuple[int, ...],
    color_permutation: tuple[int, ...],
) -> int:
    i, j, a, b = variable_key(N, D, index)
    return variable_index(
        N,
        D,
        vertex_permutation[i],
        vertex_permutation[j],
        color_permutation[a],
        color_permutation[b],
    )


@lru_cache(maxsize=1)
def _group_actions() -> tuple[
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    ...,
]:
    actions = []
    for vertex_permutation in permutations(range(N)):
        for color_permutation in permutations(range(D)):
            variable_permutation = tuple(
                _transport_variable(
                    variable,
                    vertex_permutation,
                    color_permutation,
                )
                for variable in range(VARIABLE_COUNT)
            )
            actions.append(
                (
                    vertex_permutation,
                    color_permutation,
                    variable_permutation,
                )
            )
    if len(actions) != GROUP_ORDER:
        raise KrennSourceIdealError("S6 x S3 action census changed")
    return tuple(actions)


def _transport_monomial(
    monomial: Sequence[int],
    variable_permutation: Sequence[int],
) -> Monomial:
    return tuple(sorted(variable_permutation[index] for index in monomial))


@dataclass(frozen=True)
class DomainOrbit:
    occupation: str
    coloring: tuple[int, ...]
    multiplier: Monomial
    coloring_orbit_size: int
    fiber_orbit_size: int
    full_orbit_size: int

    def __post_init__(self) -> None:
        if (
            self.occupation != occupation_type(self.coloring)
            or len(self.multiplier) != 6
            or self.full_orbit_size
            != self.coloring_orbit_size * self.fiber_orbit_size
            or monomial_fine_degree(self.multiplier)
            != multiplier_fine_degree(self.coloring)
        ):
            raise KrennSourceIdealError("domain-orbit record is malformed")


@lru_cache(maxsize=1)
def domain_orbits() -> tuple[DomainOrbit, ...]:
    """Return all 1,314 exact domain-orbit representatives."""

    actions = _group_actions()
    records: list[DomainOrbit] = []
    raw_total = 0
    per_occupation = Counter()
    for occupation, coloring, expected_coloring_orbit, expected_fiber_orbits in (
        OCCUPATION_REPRESENTATIVES
    ):
        coloring_orbit = {
            _transport_coloring(coloring, vertex, color)
            for vertex, color, _variables in actions
        }
        if (
            min(coloring_orbit) != coloring
            or len(coloring_orbit) != expected_coloring_orbit
        ):
            raise KrennSourceIdealError(
                "coloring orbit representative or size changed"
            )
        stabilizer = tuple(
            variables
            for vertex, color, variables in actions
            if _transport_coloring(coloring, vertex, color) == coloring
        )
        if len(stabilizer) * len(coloring_orbit) != GROUP_ORDER:
            raise KrennSourceIdealError(
                "coloring orbit-stabilizer replay failed"
            )

        multipliers = enumerate_multiplier_monomials(coloring)
        multiplier_set = set(multipliers)
        pending = set(multiplier_set)
        local_records = []
        while pending:
            seed = min(pending)
            orbit = {
                _transport_monomial(seed, variables)
                for variables in stabilizer
            }
            if not orbit <= multiplier_set:
                raise KrennSourceIdealError(
                    "coloring stabilizer left the multiplier fiber"
                )
            representative = min(orbit)
            pending.difference_update(orbit)
            local_records.append(
                DomainOrbit(
                    occupation=occupation,
                    coloring=coloring,
                    multiplier=representative,
                    coloring_orbit_size=len(coloring_orbit),
                    fiber_orbit_size=len(orbit),
                    full_orbit_size=len(coloring_orbit) * len(orbit),
                )
            )
        local_records.sort(key=lambda record: record.multiplier)
        if len(local_records) != expected_fiber_orbits:
            raise KrennSourceIdealError(
                "multiplier stabilizer-orbit census changed"
            )
        if sum(record.fiber_orbit_size for record in local_records) != 6_040:
            raise KrennSourceIdealError(
                "multiplier fiber orbits do not cover all monomials"
            )
        records.extend(local_records)
        per_occupation[occupation] = len(local_records)
        raw_total += sum(record.full_orbit_size for record in local_records)

    records.sort(key=lambda record: (record.coloring, record.multiplier))
    if (
        len(records) != EXPECTED_DOMAIN_ORBITS
        or raw_total != EXPECTED_RAW_DOMAIN_COLUMNS
        or sum(per_occupation.values()) != EXPECTED_DOMAIN_ORBITS
    ):
        raise KrennSourceIdealError("compressed domain census changed")
    return tuple(records)


def domain_orbit_census() -> Mapping[str, int]:
    return dict(sorted(Counter(
        record.occupation for record in domain_orbits()
    ).items()))


@dataclass(frozen=True)
class ReynoldsReduction:
    """Exact reason invariant multipliers lose no identities over ``Q``."""

    group_order: int
    action_count: int
    mixed_generator_set_stable: bool
    D_factor_set_stable: bool
    averaging_denominator_nonzero_over_Q: bool

    def __post_init__(self) -> None:
        if (
            self.group_order != GROUP_ORDER
            or self.action_count != GROUP_ORDER
            or not self.mixed_generator_set_stable
            or not self.D_factor_set_stable
            or not self.averaging_denominator_nonzero_over_Q
        ):
            raise KrennSourceIdealError(
                "Reynolds-averaging reduction is incomplete"
            )

    @property
    def invariant_search_lossless_over_Q(self) -> bool:
        return (
            self.action_count == self.group_order
            and self.mixed_generator_set_stable
            and self.D_factor_set_stable
            and self.averaging_denominator_nonzero_over_Q
        )


@lru_cache(maxsize=1)
def certify_reynolds_reduction() -> ReynoldsReduction:
    """Replay the finite-group hypotheses behind orbit compression.

    Vertex/color permutations are bijections.  They preserve whether a
    coloring is all-equal, hence preserve the complementary mixed set.  On
    the three all-equal factors they act by a permutation, so their product
    ``D`` is fixed.  Division by 4,320 is valid in ``Q``.
    """

    actions = _group_actions()
    all_equal_set = set(ALL_EQUAL_COLORINGS)
    factor_set_stable = all(
        {
            _transport_coloring(coloring, vertex, color)
            for coloring in ALL_EQUAL_COLORINGS
        }
        == all_equal_set
        for vertex, color, _variables in actions
    )
    # A coloring is mixed precisely when it is outside the all-equal set;
    # every group action is a coloring bijection and the set is stable.
    mixed_stable = factor_set_stable and len(actions) == GROUP_ORDER
    return ReynoldsReduction(
        group_order=GROUP_ORDER,
        action_count=len(actions),
        mixed_generator_set_stable=mixed_stable,
        D_factor_set_stable=factor_set_stable,
        averaging_denominator_nonzero_over_Q=True,
    )


def _cycle_type(permutation: Sequence[int]) -> tuple[int, ...]:
    seen = set()
    cycles = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cursor = start
        length = 0
        while cursor not in seen:
            seen.add(cursor)
            cursor = permutation[cursor]
            length += 1
        cycles.append(length)
    return tuple(sorted(cycles, reverse=True))


def _fixed_codomain_matchings(
    vertex_permutation: tuple[int, ...],
    color_permutation: tuple[int, ...],
) -> int:
    """Count delta-monomials fixed by one group element.

    An invariant matching is a disjoint union of edge orbits.  Only edge
    orbits that are themselves matchings can occur; an exact-cover recursion
    then counts their unions.
    """

    token_count = N * D
    token_permutation = tuple(
        vertex_permutation[vertex] * D + color_permutation[color]
        for vertex in range(N)
        for color in range(D)
    )
    allowed_edges = {
        (left, right)
        for left in range(token_count)
        for right in range(left + 1, token_count)
        if left // D != right // D
    }
    pending = set(allowed_edges)
    matching_orbit_masks: list[int] = []
    while pending:
        seed = min(pending)
        orbit = set()
        edge = seed
        while edge not in orbit:
            orbit.add(edge)
            left, right = edge
            image = sorted(
                (token_permutation[left], token_permutation[right])
            )
            edge = (image[0], image[1])
        pending.difference_update(orbit)
        mask = 0
        valid = True
        for left, right in orbit:
            edge_mask = (1 << left) | (1 << right)
            if mask & edge_mask:
                valid = False
                break
            mask |= edge_mask
        if valid:
            matching_orbit_masks.append(mask)

    covering: list[list[int]] = [[] for _ in range(token_count)]
    for mask in matching_orbit_masks:
        for token in range(token_count):
            if mask >> token & 1:
                covering[token].append(mask)

    full = (1 << token_count) - 1

    @lru_cache(maxsize=None)
    def exact_covers(remaining: int) -> int:
        if not remaining:
            return 1
        low_bit = remaining & -remaining
        token = low_bit.bit_length() - 1
        return sum(
            exact_covers(remaining ^ mask)
            for mask in covering[token]
            if mask & remaining == mask
        )

    return exact_covers(full)


@lru_cache(maxsize=1)
def codomain_orbit_count() -> int:
    """Replay Burnside's lemma for the 3,102 codomain orbits."""

    representatives: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[int, ...], tuple[int, ...], int],
    ] = {}
    for vertex in permutations(range(N)):
        for color in permutations(range(D)):
            key = (_cycle_type(vertex), _cycle_type(color))
            if key in representatives:
                old_vertex, old_color, multiplicity = representatives[key]
                representatives[key] = (
                    old_vertex,
                    old_color,
                    multiplicity + 1,
                )
            else:
                representatives[key] = (vertex, color, 1)
    burnside_sum = 0
    for vertex, color, multiplicity in representatives.values():
        burnside_sum += (
            multiplicity * _fixed_codomain_matchings(vertex, color)
        )
    if burnside_sum % GROUP_ORDER:
        raise KrennSourceIdealError("codomain Burnside sum is not integral")
    count = burnside_sum // GROUP_ORDER
    if count != EXPECTED_CODOMAIN_ORBITS:
        raise KrennSourceIdealError("codomain orbit census changed")
    return count


@lru_cache(maxsize=1)
def d_monomials() -> tuple[Monomial, ...]:
    """Expand ``D`` into its 3,375 distinct degree-nine monomials."""

    system = _system()
    equations = tuple(
        coloring_index(N, D, coloring)
        for coloring in ALL_EQUAL_COLORINGS
    )
    term_sets = tuple(
        system.equation_monomials(equation) for equation in equations
    )
    monomials = tuple(sorted({
        tuple(sorted((*left, *middle, *right)))
        for left in term_sets[0]
        for middle in term_sets[1]
        for right in term_sets[2]
    }))
    if (
        len(monomials) != EXPECTED_D_MONOMIALS
        or any(len(monomial) != 9 for monomial in monomials)
        or any(monomial_fine_degree(monomial) != target_fine_degree()
               for monomial in monomials)
    ):
        raise KrennSourceIdealError("D expansion census changed")
    return monomials


def _full_monomial_orbit(seed: Monomial) -> set[Monomial]:
    return {
        _transport_monomial(seed, variables)
        for _vertex, _color, variables in _group_actions()
    }


def _classify_queried_codomain_orbits(
    queried: Iterable[Monomial],
) -> tuple[
    dict[Monomial, tuple[Monomial, int]],
    dict[Monomial, int],
]:
    """Classify only queried rows while still computing their full orbits."""

    pending = set(queried)
    classification: dict[Monomial, tuple[Monomial, int]] = {}
    orbit_sizes: dict[Monomial, int] = {}
    while pending:
        seed = min(pending)
        orbit = _full_monomial_orbit(seed)
        representative = min(orbit)
        size = len(orbit)
        hits = orbit & pending
        if not hits:
            raise KrennSourceIdealError("codomain orbit missed its seed")
        for monomial in hits:
            classification[monomial] = (representative, size)
        pending.difference_update(hits)
        previous = orbit_sizes.setdefault(representative, size)
        if previous != size:
            raise KrennSourceIdealError("codomain orbit size changed")
    return classification, orbit_sizes


@lru_cache(maxsize=1)
def d_orbit_records() -> tuple[tuple[Monomial, int], ...]:
    classification, sizes = _classify_queried_codomain_orbits(d_monomials())
    counts = Counter(
        classification[monomial][0] for monomial in d_monomials()
    )
    records = tuple(sorted(sizes.items()))
    if (
        len(records) != EXPECTED_D_ORBITS
        or sum(counts.values()) != EXPECTED_D_MONOMIALS
        or any(counts[key] != size for key, size in records)
    ):
        raise KrennSourceIdealError("D is not a union of eight full orbits")
    return records


@dataclass(frozen=True)
class SparseCompressedSystem:
    """The orbit-total integer system ``B*x=b``."""

    row_keys: tuple[Monomial, ...]
    row_orbit_sizes: tuple[int, ...]
    columns: tuple[DomainOrbit, ...]
    column_entries: tuple[tuple[tuple[int, int], ...], ...]
    rhs: tuple[int, ...]
    schema: str = SOURCE_IDEAL_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != SOURCE_IDEAL_SCHEMA
            or len(self.row_keys) != EXPECTED_CODOMAIN_ORBITS
            or len(self.row_orbit_sizes) != len(self.row_keys)
            or len(self.columns) != EXPECTED_DOMAIN_ORBITS
            or len(self.column_entries) != len(self.columns)
            or len(self.rhs) != len(self.row_keys)
            or tuple(sorted(self.row_keys)) != self.row_keys
            or sum(self.row_orbit_sizes)
            != EXPECTED_RAW_CODOMAIN_MONOMIALS
            or self.nnz != EXPECTED_COMPRESSED_NNZ
        ):
            raise KrennSourceIdealError(
                "compressed integer system census changed"
            )
        for record, entries in zip(
            self.columns,
            self.column_entries,
            strict=True,
        ):
            if (
                tuple(sorted(entries)) != entries
                or len({row for row, _value in entries}) != len(entries)
                or any(
                    row not in range(len(self.row_keys)) or value <= 0
                    for row, value in entries
                )
            ):
                raise KrennSourceIdealError(
                    "compressed column is malformed"
                )
            if sum(value for _row, value in entries) != (
                PERFECT_MATCHING_COUNT * record.full_orbit_size
            ):
                raise KrennSourceIdealError(
                    "compressed column violates orbit-total mass"
                )
        expected_D_rows = {key for key, _size in d_orbit_records()}
        if any(
            value
            != (
                self.row_orbit_sizes[row]
                if self.row_keys[row] in expected_D_rows
                else 0
            )
            for row, value in enumerate(self.rhs)
        ):
            raise KrennSourceIdealError("compressed D vector is malformed")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.row_keys), len(self.columns)

    @property
    def nnz(self) -> int:
        return sum(map(len, self.column_entries))

    def row_dictionaries(self) -> tuple[dict[int, int], ...]:
        rows = [dict() for _ in self.row_keys]
        for column, entries in enumerate(self.column_entries):
            for row, value in entries:
                rows[row][column] = value
        return tuple(rows)

    def fingerprint(self) -> str:
        digest = sha256()
        digest.update((self.schema + "\n").encode("ascii"))
        for key, orbit_size, rhs in zip(
            self.row_keys,
            self.row_orbit_sizes,
            self.rhs,
            strict=True,
        ):
            digest.update(bytes(key))
            digest.update(orbit_size.to_bytes(8, "little"))
            digest.update(rhs.to_bytes(8, "little", signed=True))
        for record, entries in zip(
            self.columns, self.column_entries, strict=True
        ):
            digest.update(bytes(record.coloring))
            digest.update(bytes(record.multiplier))
            digest.update(record.full_orbit_size.to_bytes(8, "little"))
            for row, value in entries:
                digest.update(row.to_bytes(4, "little"))
                digest.update(value.to_bytes(8, "little"))
            digest.update(b"\xff")
        return digest.hexdigest()


@lru_cache(maxsize=1)
def build_compressed_system() -> SparseCompressedSystem:
    """Construct the exact 3,102 by 1,314 orbit-total integer system."""

    system = _system()
    columns = domain_orbits()
    raw_column_rows: list[tuple[Monomial, ...]] = []
    queried: set[Monomial] = set(d_monomials())
    for record in columns:
        equation = coloring_index(N, D, record.coloring)
        rows = tuple(
            tuple(sorted((*record.multiplier, *term)))
            for term in system.equation_monomials(equation)
        )
        if len(set(rows)) != PERFECT_MATCHING_COUNT:
            raise KrennSourceIdealError(
                "a compressed column lost a matching term"
            )
        raw_column_rows.append(rows)
        queried.update(rows)

    classification, orbit_sizes = _classify_queried_codomain_orbits(queried)
    if (
        len(orbit_sizes) != codomain_orbit_count()
        or sum(orbit_sizes.values()) != raw_codomain_monomial_count()
    ):
        raise KrennSourceIdealError(
            "queried rows did not cover every codomain orbit"
        )
    row_keys = tuple(sorted(orbit_sizes))
    row_index = {key: index for index, key in enumerate(row_keys)}
    row_orbit_sizes = tuple(orbit_sizes[key] for key in row_keys)

    compressed_columns = []
    for record, raw_rows in zip(columns, raw_column_rows, strict=True):
        counts = Counter(
            classification[row][0] for row in raw_rows
        )
        entries = tuple(sorted(
            (
                row_index[key],
                record.full_orbit_size * multiplicity,
            )
            for key, multiplicity in counts.items()
        ))
        compressed_columns.append(entries)

    d_counts = Counter(
        classification[monomial][0] for monomial in d_monomials()
    )
    if (
        len(d_counts) != EXPECTED_D_ORBITS
        or any(
            d_counts[key] != orbit_sizes[key] for key in d_counts
        )
    ):
        raise KrennSourceIdealError("compressed D orbit totals changed")
    rhs = tuple(
        orbit_sizes[key] if key in d_counts else 0
        for key in row_keys
    )
    result = SparseCompressedSystem(
        row_keys=row_keys,
        row_orbit_sizes=row_orbit_sizes,
        columns=columns,
        column_entries=tuple(compressed_columns),
        rhs=rhs,
    )
    if result.fingerprint() != EXPECTED_COMPRESSED_FINGERPRINT:
        raise KrennSourceIdealError(
            "compressed integer system fingerprint changed"
        )
    verify_exact_dual(result)
    return result


@dataclass(frozen=True)
class ExactIntegerDual:
    """A two-row integer functional annihilating ``B`` but not ``b``."""

    row_indices: tuple[int, int]
    row_keys: tuple[Monomial, Monomial]
    row_orbit_sizes: tuple[int, int]
    shared_column_index: int
    shared_column_coefficient: int
    integer_lambda: tuple[int, int]
    lambda_transpose_b: int

    def __post_init__(self) -> None:
        if (
            self.row_indices != EXPECTED_DUAL_ROW_INDICES
            or self.row_keys != DUAL_ROW_KEYS
            or self.row_orbit_sizes != EXPECTED_DUAL_ROW_ORBIT_SIZES
            or self.integer_lambda != (1, -1)
            or self.lambda_transpose_b
            != (
                self.integer_lambda[0] * self.row_orbit_sizes[0]
                + self.integer_lambda[1] * self.row_orbit_sizes[1]
            )
            or self.lambda_transpose_b != EXPECTED_INTEGER_DUAL_RHS
            or self.shared_column_coefficient <= 0
            or any(
                monomial_fine_degree(key) != target_fine_degree()
                for key in self.row_keys
            )
        ):
            raise KrennSourceIdealError(
                "exact integer dual record is malformed"
            )
        record = domain_orbits()[self.shared_column_index]
        if (
            record.coloring != (0, 0, 0, 0, 1, 1)
            or record.multiplier != (4, 17, 71, 85, 125, 126)
            or record.full_orbit_size != 2_160
            or self.shared_column_coefficient != 2_160
        ):
            raise KrennSourceIdealError(
                "exact integer dual shared column changed"
            )

    @property
    def lambda_transpose_B_zero(self) -> bool:
        # Both rows have the single profile recorded by
        # (shared_column_index, shared_column_coefficient).
        return sum(self.integer_lambda) == 0

    @property
    def exact_Q_nonmembership_proved(self) -> bool:
        return (
            self.lambda_transpose_B_zero
            and self.lambda_transpose_b != 0
            and certify_reynolds_reduction().invariant_search_lossless_over_Q
        )

    @property
    def radical_nonmembership_proved(self) -> bool:
        """A k=1 dual says nothing decisive about a higher power ``D^k``."""

        return False

    @property
    def GHZ_nonexistence_proved(self) -> bool:
        """Nonmembership of this one source polynomial is not a GHZ no-go."""

        return False

    def to_dict(self) -> dict:
        return {
            "row_indices": list(self.row_indices),
            "row_keys": [list(key) for key in self.row_keys],
            "row_variable_keys": [
                [list(variable_key(N, D, index)) for index in key]
                for key in self.row_keys
            ],
            "row_orbit_sizes_and_rhs": list(self.row_orbit_sizes),
            "shared_column_index": self.shared_column_index,
            "shared_column_coefficient": self.shared_column_coefficient,
            "integer_lambda": list(self.integer_lambda),
            "lambda_transpose_B_zero": self.lambda_transpose_B_zero,
            "lambda_transpose_b": self.lambda_transpose_b,
            "normalized_Q_dual": {
                "rows": [
                    self.row_indices[1],
                    self.row_indices[0],
                ],
                "numerators": [1, -1],
                "denominator": -self.lambda_transpose_b,
                "transpose_b": 1,
            },
            "exact_Q_nonmembership_proved": (
                self.exact_Q_nonmembership_proved
            ),
            "compression_convention": compression_convention(),
            "claim_boundary": (
                "This proves D is not in J_mix at k=1. It does not prove "
                "D is outside radical(J_mix), because a higher power D^k "
                "may still lie in J_mix."
            ),
        }


def _row_total_against_orbit(
    record: DomainOrbit,
    target_orbit: set[Monomial],
) -> int:
    system = _system()
    equation = coloring_index(N, D, record.coloring)
    hits = sum(
        tuple(sorted((*record.multiplier, *term))) in target_orbit
        for term in system.equation_monomials(equation)
    )
    return record.full_orbit_size * hits


@lru_cache(maxsize=1)
def exact_two_row_dual() -> ExactIntegerDual:
    """Derive the exact obstruction without building all of ``B``."""

    target_orbits = tuple(
        _full_monomial_orbit(key) for key in DUAL_ROW_KEYS
    )
    orbit_sizes = tuple(map(len, target_orbits))
    if orbit_sizes != EXPECTED_DUAL_ROW_ORBIT_SIZES:
        raise KrennSourceIdealError("dual row orbit sizes changed")
    d_support = set(d_monomials())
    if any(not orbit <= d_support for orbit in target_orbits):
        raise KrennSourceIdealError("dual row orbit left D support")

    rows = []
    for target_orbit in target_orbits:
        row = {}
        for column, record in enumerate(domain_orbits()):
            value = _row_total_against_orbit(record, target_orbit)
            if value:
                row[column] = value
        rows.append(row)
    if rows[0] != rows[1] or len(rows[0]) != 1:
        raise KrennSourceIdealError(
            "the two exact compressed rows are no longer identical"
        )
    shared_column, coefficient = next(iter(rows[0].items()))
    lambda_b = orbit_sizes[0] - orbit_sizes[1]
    if lambda_b != EXPECTED_INTEGER_DUAL_RHS:
        raise KrennSourceIdealError("integer dual RHS changed")
    return ExactIntegerDual(
        row_indices=EXPECTED_DUAL_ROW_INDICES,
        row_keys=DUAL_ROW_KEYS,
        row_orbit_sizes=orbit_sizes,
        shared_column_index=shared_column,
        shared_column_coefficient=coefficient,
        integer_lambda=(1, -1),
        lambda_transpose_b=lambda_b,
    )


def verify_exact_dual(
    compressed: SparseCompressedSystem | None = None,
) -> ExactIntegerDual:
    """Replay the exact integer dual against the full matrix when supplied."""

    certificate = exact_two_row_dual()
    if compressed is None:
        return certificate
    indices = tuple(
        compressed.row_keys.index(key) for key in certificate.row_keys
    )
    if indices != certificate.row_indices:
        raise KrennSourceIdealError("dual row indices changed")
    rows = compressed.row_dictionaries()
    left, right = indices
    if (
        rows[left] != rows[right]
        or rows[left]
        != {
            certificate.shared_column_index:
            certificate.shared_column_coefficient
        }
        or compressed.rhs[left] - compressed.rhs[right]
        != certificate.lambda_transpose_b
    ):
        raise KrennSourceIdealError(
            "full compressed matrix failed exact dual replay"
        )
    return certificate


def _is_prime(value: int) -> bool:
    value = int(value)
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def sparse_rank_mod_prime(
    rows: Iterable[Mapping[int, int]],
    column_count: int,
    prime: int,
) -> int:
    """Sparse deterministic Gaussian elimination over one prime field."""

    prime = int(prime)
    if (
        not _is_prime(prime)
        or GROUP_ORDER % prime == 0
        or column_count < 0
    ):
        raise KrennSourceIdealError(
            "modular rank needs a prime not dividing 4320"
        )
    pivots: dict[int, dict[int, int]] = {}
    for source in rows:
        row = {
            int(column): int(value) % prime
            for column, value in source.items()
            if int(value) % prime
        }
        if any(column < 0 or column >= column_count for column in row):
            raise KrennSourceIdealError("modular row column is out of range")
        while row:
            pivot = min(row)
            existing = pivots.get(pivot)
            if existing is None:
                inverse = pow(row[pivot], -1, prime)
                pivots[pivot] = {
                    column: value * inverse % prime
                    for column, value in row.items()
                }
                break
            factor = row[pivot]
            for column, value in existing.items():
                updated = (row.get(column, 0) - factor * value) % prime
                if updated:
                    row[column] = updated
                else:
                    row.pop(column, None)
    return len(pivots)


@dataclass(frozen=True)
class ModularRankRecord:
    prime: int
    matrix_rank: int
    augmented_rank: int
    runtime_seconds: float

    def __post_init__(self) -> None:
        if (
            not _is_prime(self.prime)
            or GROUP_ORDER % self.prime == 0
            or self.matrix_rank < 0
            or self.augmented_rank < self.matrix_rank
            or self.augmented_rank > self.matrix_rank + 1
            or self.runtime_seconds < 0
        ):
            raise KrennSourceIdealError(
                "modular rank record is malformed"
            )

    def to_dict(self) -> dict:
        return {
            "prime": self.prime,
            "matrix_rank": self.matrix_rank,
            "augmented_rank": self.augmented_rank,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(frozen=True)
class ModularPreflight:
    matrix_fingerprint: str
    shape: tuple[int, int]
    nnz: int
    records: tuple[ModularRankRecord, ...]
    loaded_from_cache: bool
    schema: str = MODULAR_CACHE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != MODULAR_CACHE_SCHEMA
            or len(self.matrix_fingerprint) != 64
            or self.shape
            != (EXPECTED_CODOMAIN_ORBITS, EXPECTED_DOMAIN_ORBITS)
            or self.nnz != EXPECTED_COMPRESSED_NNZ
            or len(self.records) < 2
            or len({record.prime for record in self.records})
            != len(self.records)
        ):
            raise KrennSourceIdealError("modular preflight is malformed")

    @property
    def expected_rank_regression(self) -> bool:
        return all(
            record.matrix_rank == EXPECTED_MODULAR_MATRIX_RANK
            and record.augmented_rank
            == EXPECTED_MODULAR_AUGMENTED_RANK
            for record in self.records
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "matrix_fingerprint": self.matrix_fingerprint,
            "shape": list(self.shape),
            "nnz": self.nnz,
            "records": [record.to_dict() for record in self.records],
            "loaded_from_cache": self.loaded_from_cache,
            "diagnostic_only": True,
            "characteristic_zero_claim_comes_from": (
                "the separately replayed exact integer two-row dual"
            ),
        }


def _load_modular_cache(
    path: Path,
    fingerprint: str,
    shape: tuple[int, int],
    nnz: int,
    primes: tuple[int, ...],
) -> ModularPreflight | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != MODULAR_CACHE_SCHEMA
            or payload.get("matrix_fingerprint") != fingerprint
            or tuple(payload.get("shape", ())) != shape
            or payload.get("nnz") != nnz
        ):
            return None
        records = tuple(
            ModularRankRecord(
                prime=int(record["prime"]),
                matrix_rank=int(record["matrix_rank"]),
                augmented_rank=int(record["augmented_rank"]),
                runtime_seconds=float(record["runtime_seconds"]),
            )
            for record in payload["records"]
        )
        if tuple(record.prime for record in records) != primes:
            return None
        result = ModularPreflight(
            matrix_fingerprint=fingerprint,
            shape=shape,
            nnz=nnz,
            records=records,
            loaded_from_cache=True,
        )
        if not result.expected_rank_regression:
            return None
        return result
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def run_modular_preflight(
    primes: Sequence[int] = DEFAULT_MODULAR_PRIMES,
    *,
    cache_path: Path | str | None = None,
) -> ModularPreflight:
    """Compute or load diagnostic ranks over at least two good primes."""

    primes = tuple(map(int, primes))
    if (
        len(primes) < 2
        or len(set(primes)) != len(primes)
        or any(not _is_prime(prime) or GROUP_ORDER % prime == 0
               for prime in primes)
    ):
        raise KrennSourceIdealError(
            "preflight needs at least two distinct primes not dividing 4320"
        )
    compressed = build_compressed_system()
    fingerprint = compressed.fingerprint()
    path = Path(cache_path).resolve() if cache_path is not None else None
    if path is not None:
        cached = _load_modular_cache(
            path,
            fingerprint,
            compressed.shape,
            compressed.nnz,
            primes,
        )
        if cached is not None:
            return cached

    rows = compressed.row_dictionaries()
    records = []
    for prime in primes:
        started = time.perf_counter()
        matrix_rank = sparse_rank_mod_prime(
            rows,
            len(compressed.columns),
            prime,
        )
        augmented_rows = tuple(
            (
                row
                if not compressed.rhs[index]
                else {
                    **row,
                    len(compressed.columns): compressed.rhs[index],
                }
            )
            for index, row in enumerate(rows)
        )
        augmented_rank = sparse_rank_mod_prime(
            augmented_rows,
            len(compressed.columns) + 1,
            prime,
        )
        records.append(
            ModularRankRecord(
                prime=prime,
                matrix_rank=matrix_rank,
                augmented_rank=augmented_rank,
                runtime_seconds=time.perf_counter() - started,
            )
        )
    result = ModularPreflight(
        matrix_fingerprint=fingerprint,
        shape=compressed.shape,
        nnz=compressed.nnz,
        records=tuple(records),
        loaded_from_cache=False,
    )
    if not result.expected_rank_regression:
        raise KrennSourceIdealError("modular rank regression changed")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = result.to_dict()
        payload["loaded_from_cache"] = False
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    return result


def exact_summary() -> Mapping:
    dual = exact_two_row_dual()
    reynolds = certify_reynolds_reduction()
    exact_nonmembership = (
        dual.exact_Q_nonmembership_proved
        and reynolds.invariant_search_lossless_over_Q
    )
    return {
        "schema": SOURCE_IDEAL_SCHEMA,
        "fine_grading": {
            "lattice_rank": 18,
            "variable_degree": "e_(i,a)+e_(j,b)",
            "D_degree": list(target_fine_degree()),
            "complementary_multiplier_description_exact": True,
        },
        "raw_census": {
            "mixed_colorings": MIXED_COLORING_COUNT,
            "multipliers_per_coloring": (
                multiplier_count_by_inclusion_exclusion()
            ),
            "domain_columns": raw_domain_column_count(),
            "codomain_monomials": raw_codomain_monomial_count(),
            "D_monomials": len(d_monomials()),
        },
        "orbit_census": {
            "domain_orbits": len(domain_orbits()),
            "domain_orbits_by_occupation": domain_orbit_census(),
            "codomain_orbits": codomain_orbit_count(),
            "D_orbits": len(d_orbit_records()),
        },
        "compression_convention": compression_convention(),
        "Reynolds_reduction": {
            "group_order": reynolds.group_order,
            "mixed_generator_set_stable": (
                reynolds.mixed_generator_set_stable
            ),
            "D_factor_set_stable": reynolds.D_factor_set_stable,
            "averaging_denominator_nonzero_over_Q": (
                reynolds.averaging_denominator_nonzero_over_Q
            ),
            "invariant_search_lossless_over_Q": (
                reynolds.invariant_search_lossless_over_Q
            ),
        },
        "exact_integer_dual": dual.to_dict(),
        "claims": {
            "D_in_J_mix_at_k1": not exact_nonmembership,
            "exact_Q_nonmembership_proved": exact_nonmembership,
            "D_outside_radical_J_mix_proved": (
                dual.radical_nonmembership_proved
            ),
            "GHZ_nonexistence_proved": (
                dual.GHZ_nonexistence_proved
            ),
            "modular_ranks_needed_for_Q_claim": (
                not dual.lambda_transpose_B_zero
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the exact n=6,d=3 k=1 source-ideal experiment."
    )
    parser.add_argument("--modular-preflight", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--primes",
        type=int,
        nargs="+",
        default=DEFAULT_MODULAR_PRIMES,
    )
    arguments = parser.parse_args(argv)
    payload = dict(exact_summary())
    if arguments.modular_preflight:
        payload["modular_preflight"] = run_modular_preflight(
            arguments.primes,
            cache_path=arguments.cache,
        ).to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
