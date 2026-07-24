"""Deterministic sparse polynomial systems for Krenn's equations.

This module is the primary constructor.  The independent verifier deliberately
does not import it and enumerates perfect matchings by a different algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from math import comb
from typing import Iterator, Sequence

from experiments.krenn_quantum_graph.targets import (
    ColoringTarget,
    ExactRational,
    KrennTargetError,
    canonical_ghz_target,
)


Edge = tuple[int, int]
Matching = tuple[Edge, ...]
Coloring = tuple[int, ...]
VariableKey = tuple[int, int, int, int]
Monomial = tuple[int, ...]

SYSTEM_SCHEMA = "krenn-quantum-graph-sparse-system-v1"


class KrennSystemError(ValueError):
    """The Krenn parameters or sparse system encoding are invalid."""


def validate_parameters(n: int, d: int) -> tuple[int, int]:
    n = int(n)
    d = int(d)
    if n < 2 or n % 2:
        raise KrennSystemError("n must be an even integer at least 2")
    if d < 1:
        raise KrennSystemError("d must be a positive integer")
    return n, d


@lru_cache(maxsize=None)
def canonical_edges(n: int) -> tuple[Edge, ...]:
    n = int(n)
    if n < 2:
        raise KrennSystemError("n must be at least 2")
    return tuple(combinations(range(n), 2))


@lru_cache(maxsize=None)
def perfect_matchings(n: int) -> tuple[Matching, ...]:
    """Enumerate matchings by pairing the least unused vertex first.

    Partners and recursive tails are visited in increasing order, so the
    returned tuple is deterministic and lexicographically ordered.
    """

    n, _ = validate_parameters(n, 1)

    @lru_cache(maxsize=None)
    def recurse(vertices: tuple[int, ...]) -> tuple[Matching, ...]:
        if not vertices:
            return ((),)
        first = vertices[0]
        result: list[Matching] = []
        for partner_position in range(1, len(vertices)):
            partner = vertices[partner_position]
            remainder = (
                vertices[1:partner_position]
                + vertices[partner_position + 1 :]
            )
            for tail in recurse(remainder):
                result.append(((first, partner), *tail))
        return tuple(result)

    result = recurse(tuple(range(n)))
    if len(result) != perfect_matching_count(n):
        raise KrennSystemError("perfect-matching census failed")
    return result


def perfect_matching_count(n: int) -> int:
    n, _ = validate_parameters(n, 1)
    count = 1
    for odd in range(n - 1, 0, -2):
        count *= odd
    return count


def variable_count(n: int, d: int) -> int:
    n, d = validate_parameters(n, d)
    return comb(n, 2) * d * d


def equation_count(n: int, d: int) -> int:
    n, d = validate_parameters(n, d)
    return d**n


def canonical_variable_key(
    n: int,
    d: int,
    i: int,
    j: int,
    a: int,
    b: int,
) -> VariableKey:
    """Canonicalize endpoints, swapping their color slots when needed."""

    n, d = validate_parameters(n, d)
    i, j, a, b = map(int, (i, j, a, b))
    if not (0 <= i < n and 0 <= j < n) or i == j:
        raise KrennSystemError("a variable needs two distinct valid vertices")
    if not (0 <= a < d and 0 <= b < d):
        raise KrennSystemError("a variable color is outside range(d)")
    if i < j:
        return i, j, a, b
    return j, i, b, a


def variable_index(
    n: int,
    d: int,
    i: int,
    j: int,
    a: int,
    b: int,
) -> int:
    """Return the index of ``W[i,j,a,b]`` in canonical lexicographic order."""

    n, d = validate_parameters(n, d)
    i, j, a, b = canonical_variable_key(n, d, i, j, a, b)
    edge_index = i * (2 * n - i - 1) // 2 + (j - i - 1)
    return (edge_index * d + a) * d + b


def variable_key(n: int, d: int, index: int) -> VariableKey:
    n, d = validate_parameters(n, d)
    index = int(index)
    count = variable_count(n, d)
    if not 0 <= index < count:
        raise KrennSystemError("variable index is outside the system")
    edge_index, color_index = divmod(index, d * d)
    a, b = divmod(color_index, d)
    i, j = canonical_edges(n)[edge_index]
    return i, j, a, b


def enumerate_colorings(n: int, d: int) -> Iterator[Coloring]:
    n, d = validate_parameters(n, d)
    return product(range(d), repeat=n)


def coloring_index(n: int, d: int, coloring: Sequence[int]) -> int:
    n, d = validate_parameters(n, d)
    coloring = tuple(map(int, coloring))
    if len(coloring) != n or any(
        color < 0 or color >= d for color in coloring
    ):
        raise KrennSystemError("coloring is not an element of range(d)^n")
    result = 0
    for color in coloring:
        result = result * d + color
    return result


def coloring_from_index(n: int, d: int, index: int) -> Coloring:
    n, d = validate_parameters(n, d)
    index = int(index)
    if not 0 <= index < equation_count(n, d):
        raise KrennSystemError("equation index is outside the system")
    digits = [0] * n
    for position in range(n - 1, -1, -1):
        index, digits[position] = divmod(index, d)
    return tuple(digits)


def rhs_for_coloring(coloring: Sequence[int]) -> int:
    coloring = tuple(map(int, coloring))
    if not coloring:
        raise KrennSystemError("a coloring cannot be empty")
    return int(all(color == coloring[0] for color in coloring[1:]))


@dataclass(frozen=True)
class SparsePolynomialSystem:
    """Flattened sparse equations with offsets into matching monomials."""

    n: int
    d: int
    equation_offsets: tuple[int, ...]
    monomial_variable_indices: tuple[Monomial, ...]
    rhs_values: tuple[ExactRational, ...]
    schema: str = SYSTEM_SCHEMA

    def __post_init__(self) -> None:
        n, d = validate_parameters(self.n, self.d)
        offsets = tuple(map(int, self.equation_offsets))
        monomials = tuple(
            tuple(map(int, monomial))
            for monomial in self.monomial_variable_indices
        )
        try:
            target = ColoringTarget.from_dense(
                n, d, tuple(self.rhs_values)
            )
        except KrennTargetError as error:
            raise KrennSystemError(
                "RHS vector is not an exact coloring target"
            ) from error
        rhs = target.dense_coefficients()
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "equation_offsets", offsets)
        object.__setattr__(
            self, "monomial_variable_indices", monomials
        )
        object.__setattr__(self, "rhs_values", rhs)

        if self.schema != SYSTEM_SCHEMA:
            raise KrennSystemError("sparse system schema changed")
        expected_equations = equation_count(n, d)
        expected_variables = variable_count(n, d)
        expected_terms = perfect_matching_count(n)
        if len(offsets) != expected_equations + 1:
            raise KrennSystemError("equation-offset census is wrong")
        if not offsets or offsets[0] != 0:
            raise KrennSystemError("equation offsets must start at zero")
        if any(
            offsets[index + 1] - offsets[index] != expected_terms
            for index in range(expected_equations)
        ):
            raise KrennSystemError(
                "each equation must contain every perfect matching once"
            )
        if offsets[-1] != len(monomials):
            raise KrennSystemError(
                "terminal equation offset does not match monomial count"
            )
        if any(len(monomial) != n // 2 for monomial in monomials):
            raise KrennSystemError("a monomial has the wrong degree")
        if any(
            variable < 0 or variable >= expected_variables
            for monomial in monomials
            for variable in monomial
        ):
            raise KrennSystemError(
                "a monomial contains an out-of-range variable index"
            )
        matchings = perfect_matchings(n)
        for equation in range(expected_equations):
            coloring = coloring_from_index(n, d, equation)
            start = offsets[equation]
            stop = offsets[equation + 1]
            expected_monomials = tuple(
                tuple(
                    variable_index(
                        n,
                        d,
                        i,
                        j,
                        coloring[i],
                        coloring[j],
                    )
                    for i, j in matching
                )
                for matching in matchings
            )
            if monomials[start:stop] != expected_monomials:
                raise KrennSystemError(
                    "a sparse equation failed exact semantic replay"
                )

    @property
    def degree(self) -> int:
        return self.n // 2

    @property
    def variable_count(self) -> int:
        return variable_count(self.n, self.d)

    @property
    def equation_count(self) -> int:
        return equation_count(self.n, self.d)

    @property
    def matching_count(self) -> int:
        return perfect_matching_count(self.n)

    @property
    def monomial_count(self) -> int:
        return len(self.monomial_variable_indices)

    @property
    def target(self) -> ColoringTarget:
        """Reconstruct the exact computational-basis target."""

        return ColoringTarget.from_dense(
            self.n, self.d, self.rhs_values
        )

    @property
    def has_canonical_ghz_target(self) -> bool:
        """Whether this system retains the original Krenn right-hand side."""

        return self.target == canonical_ghz_target(self.n, self.d)

    def equation_monomials(self, equation: int) -> tuple[Monomial, ...]:
        equation = int(equation)
        if not 0 <= equation < self.equation_count:
            raise KrennSystemError("equation index is outside the system")
        start = self.equation_offsets[equation]
        stop = self.equation_offsets[equation + 1]
        return self.monomial_variable_indices[start:stop]


def generate_sparse_system(
    n: int,
    d: int,
    target: ColoringTarget | None = None,
) -> SparsePolynomialSystem:
    """Generate the exact system for a computational-basis target."""

    n, d = validate_parameters(n, d)
    target = canonical_ghz_target(n, d) if target is None else target
    if not isinstance(target, ColoringTarget):
        raise KrennSystemError(
            "target must be a canonical ColoringTarget"
        )
    if (target.n, target.d) != (n, d):
        raise KrennSystemError(
            "target dimensions do not match the system"
        )
    matchings = perfect_matchings(n)
    offsets = [0]
    monomials: list[Monomial] = []
    rhs = target.dense_coefficients()
    for coloring in enumerate_colorings(n, d):
        for matching in matchings:
            monomials.append(
                tuple(
                    variable_index(
                        n,
                        d,
                        i,
                        j,
                        coloring[i],
                        coloring[j],
                    )
                    for i, j in matching
                )
            )
        offsets.append(len(monomials))
    return SparsePolynomialSystem(
        n=n,
        d=d,
        equation_offsets=tuple(offsets),
        monomial_variable_indices=tuple(monomials),
        rhs_values=rhs,
    )
