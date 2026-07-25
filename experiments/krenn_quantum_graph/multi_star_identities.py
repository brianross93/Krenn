r"""Exact simultaneous star identities for the perfect-matching tensor map.

Every perfect matching covers a selected root exactly once.  Partitioning
the matching by that edge gives the star expansion

    H_V(c) = sum_{s != r} W_rs[c_r,c_s] H_{V \ {r,s}}(c).

Doing this at every root produces linear systems whose edge blocks must agree
up to transpose.  A second expansion at two roots partitions matchings into
those using the root edge and those pairing the roots with distinct residual
vertices.

These are exact combinatorial identities for every weight assignment.  They
are useful reformulations, but they do not by themselves prove membership or
nonmembership of any target.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb
from typing import Mapping, Sequence, TypeAlias

from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    enumerate_colorings,
    perfect_matchings,
    validate_parameters,
    variable_index,
)
from experiments.krenn_quantum_graph.targets import ColoringTarget
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.witness import SparseWitness


ExactScalar: TypeAlias = int | Fraction
ExactVector: TypeAlias = tuple[Fraction, ...]
ExactMatrix: TypeAlias = tuple[ExactVector, ...]
MULTI_STAR_CERTIFICATE_SCHEMA = "krenn-multi-star-identity-certificate-v1"


class KrennMultiStarIdentityError(ValueError):
    """A multi-star input or exact identity is malformed."""


def _exact_fraction(value: ExactScalar, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise KrennMultiStarIdentityError(
            f"{label} must be an exact integer or rational"
        )
    return Fraction(value)


def _exact_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KrennMultiStarIdentityError(
            f"{label} must be an exact integer"
        )
    return value


def _coloring(
    witness: SparseWitness,
    coloring: Sequence[int],
) -> tuple[int, ...]:
    result = tuple(
        _exact_integer(color, "coloring entry") for color in coloring
    )
    if len(result) != witness.n or any(
        color < 0 or color >= witness.d for color in result
    ):
        raise KrennMultiStarIdentityError(
            "coloring is outside range(d)^n"
        )
    return result


def _vertices(
    n: int,
    vertices: Sequence[int],
    *,
    require_even: bool = True,
) -> tuple[int, ...]:
    result = tuple(
        _exact_integer(vertex, "induced vertex")
        for vertex in vertices
    )
    if (
        len(result) != len(set(result))
        or any(vertex < 0 or vertex >= n for vertex in result)
    ):
        raise KrennMultiStarIdentityError(
            "induced vertices must be distinct elements of range(n)"
        )
    result = tuple(sorted(result))
    if require_even and len(result) % 2:
        raise KrennMultiStarIdentityError(
            "an induced matching sum needs an even vertex count"
        )
    return result


def _weight(
    values: Mapping[int, Fraction],
    n: int,
    d: int,
    i: int,
    j: int,
    a: int,
    b: int,
) -> Fraction:
    return values.get(variable_index(n, d, i, j, a, b), Fraction(0))


def _induced_matching_sum(
    values: Mapping[int, Fraction],
    n: int,
    d: int,
    coloring: tuple[int, ...],
    vertices: tuple[int, ...],
) -> Fraction:
    if not vertices:
        return Fraction(1)
    total = Fraction(0)
    for local_matching in perfect_matchings(len(vertices)):
        term = Fraction(1)
        for local_i, local_j in local_matching:
            i = vertices[local_i]
            j = vertices[local_j]
            term *= _weight(
                values,
                n,
                d,
                i,
                j,
                coloring[i],
                coloring[j],
            )
        total += term
    return total


def induced_matching_sum(
    witness: SparseWitness,
    coloring: Sequence[int],
    vertices: Sequence[int],
) -> Fraction:
    """Return the exact matching sum on an induced even vertex set.

    The empty induced set has matching sum one.
    """

    checked_coloring = _coloring(witness, coloring)
    checked_vertices = _vertices(witness.n, vertices)
    return _induced_matching_sum(
        witness.as_dict(),
        witness.n,
        witness.d,
        checked_coloring,
        checked_vertices,
    )


def matching_cofactor(
    witness: SparseWitness,
    coloring: Sequence[int],
    first: int,
    second: int,
) -> Fraction:
    """Return ``H_(V \\ {first,second})(coloring)`` exactly."""

    first = _exact_integer(first, "first cofactor vertex")
    second = _exact_integer(second, "second cofactor vertex")
    if (
        first == second
        or first < 0
        or second < 0
        or first >= witness.n
        or second >= witness.n
    ):
        raise KrennMultiStarIdentityError(
            "a matching cofactor needs two distinct valid vertices"
        )
    residual = tuple(
        vertex
        for vertex in range(witness.n)
        if vertex not in (first, second)
    )
    return induced_matching_sum(witness, coloring, residual)


def star_expansion(
    witness: SparseWitness,
    coloring: Sequence[int],
    root: int,
) -> Fraction:
    """Expand one tensor coefficient by the edge covering ``root``."""

    checked = _coloring(witness, coloring)
    root = _exact_integer(root, "star root")
    if root < 0 or root >= witness.n:
        raise KrennMultiStarIdentityError("star root is outside range(n)")
    values = witness.as_dict()
    total = Fraction(0)
    for neighbor in range(witness.n):
        if neighbor == root:
            continue
        total += _weight(
            values,
            witness.n,
            witness.d,
            root,
            neighbor,
            checked[root],
            checked[neighbor],
        ) * matching_cofactor(witness, checked, root, neighbor)
    return total


def two_star_expansion(
    witness: SparseWitness,
    coloring: Sequence[int],
    first: int,
    second: int,
) -> Fraction:
    """Expand a coefficient simultaneously at two distinct roots."""

    checked = _coloring(witness, coloring)
    first = _exact_integer(first, "first two-star root")
    second = _exact_integer(second, "second two-star root")
    if (
        first == second
        or first < 0
        or second < 0
        or first >= witness.n
        or second >= witness.n
    ):
        raise KrennMultiStarIdentityError(
            "two-star expansion needs two distinct valid roots"
        )
    values = witness.as_dict()
    residual = tuple(
        vertex
        for vertex in range(witness.n)
        if vertex not in (first, second)
    )
    total = _weight(
        values,
        witness.n,
        witness.d,
        first,
        second,
        checked[first],
        checked[second],
    ) * _induced_matching_sum(
        values,
        witness.n,
        witness.d,
        checked,
        residual,
    )
    for first_partner in residual:
        for second_partner in residual:
            if first_partner == second_partner:
                continue
            remaining = tuple(
                vertex
                for vertex in residual
                if vertex not in (first_partner, second_partner)
            )
            total += (
                _weight(
                    values,
                    witness.n,
                    witness.d,
                    first,
                    first_partner,
                    checked[first],
                    checked[first_partner],
                )
                * _weight(
                    values,
                    witness.n,
                    witness.d,
                    second,
                    second_partner,
                    checked[second],
                    checked[second_partner],
                )
                * _induced_matching_sum(
                    values,
                    witness.n,
                    witness.d,
                    checked,
                    remaining,
                )
            )
    return total


@dataclass(frozen=True)
class StarLinearization:
    """One exact root-linearization ``A_r X_r`` of the tensor map."""

    n: int
    d: int
    root: int
    other_vertices: tuple[int, ...]
    residual_colorings: tuple[tuple[int, ...], ...]
    column_keys: tuple[tuple[int, int], ...]
    coefficient_matrix: ExactMatrix
    star_weight_matrix: ExactMatrix

    def __post_init__(self) -> None:
        n, d = validate_parameters(
            _exact_integer(self.n, "linearization n"),
            _exact_integer(self.d, "linearization d"),
        )
        root = _exact_integer(self.root, "linearization root")
        if root < 0 or root >= n:
            raise KrennMultiStarIdentityError(
                "linearization root is outside range(n)"
            )
        expected_vertices = tuple(
            vertex for vertex in range(n) if vertex != root
        )
        expected_residuals = tuple(
            product(range(d), repeat=n - 1)
        )
        expected_columns = tuple(
            (neighbor, color)
            for neighbor in expected_vertices
            for color in range(d)
        )
        other_vertices = tuple(
            _exact_integer(vertex, "residual vertex")
            for vertex in self.other_vertices
        )
        residual_colorings = tuple(
            tuple(
                _exact_integer(color, "residual color")
                for color in coloring
            )
            for coloring in self.residual_colorings
        )
        column_keys = tuple(
            tuple(
                _exact_integer(value, "column key") for value in key
            )
            for key in self.column_keys
        )
        coefficient_matrix = tuple(
            tuple(
                _exact_fraction(value, "linearization coefficient")
                for value in row
            )
            for row in self.coefficient_matrix
        )
        star_weight_matrix = tuple(
            tuple(
                _exact_fraction(value, "star weight")
                for value in row
            )
            for row in self.star_weight_matrix
        )
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "root", root)
        object.__setattr__(
            self, "other_vertices", other_vertices
        )
        object.__setattr__(
            self, "residual_colorings", residual_colorings
        )
        object.__setattr__(self, "column_keys", column_keys)
        object.__setattr__(
            self, "coefficient_matrix", coefficient_matrix
        )
        object.__setattr__(
            self, "star_weight_matrix", star_weight_matrix
        )
        width = (n - 1) * d
        if other_vertices != expected_vertices:
            raise KrennMultiStarIdentityError(
                "linearization residual vertices changed"
            )
        if residual_colorings != expected_residuals:
            raise KrennMultiStarIdentityError(
                "linearization residual-coloring order changed"
            )
        if column_keys != expected_columns:
            raise KrennMultiStarIdentityError(
                "linearization column order changed"
            )
        if (
            len(coefficient_matrix) != d ** (n - 1)
            or any(len(row) != width for row in coefficient_matrix)
            or len(star_weight_matrix) != width
            or any(len(row) != d for row in star_weight_matrix)
        ):
            raise KrennMultiStarIdentityError(
                "linearization matrix dimensions are inconsistent"
            )

    def oriented_edge_matrix(self, neighbor: int) -> ExactMatrix:
        """Return ``W_(root->neighbor)`` as root-color by neighbor-color."""

        neighbor = _exact_integer(neighbor, "star neighbor")
        if neighbor not in self.other_vertices:
            raise KrennMultiStarIdentityError(
                "neighbor is not in this star"
            )
        row_by_key = {
            key: row
            for key, row in zip(
                self.column_keys,
                self.star_weight_matrix,
                strict=True,
            )
        }
        return tuple(
            tuple(
                row_by_key[(neighbor, neighbor_color)][root_color]
                for neighbor_color in range(self.d)
            )
            for root_color in range(self.d)
        )

    def reconstructed_target(self) -> ColoringTarget:
        """Multiply ``A_r X_r`` and return the full exact tensor."""

        residual_index = {
            coloring: index
            for index, coloring in enumerate(self.residual_colorings)
        }
        coefficients = []
        for coloring in enumerate_colorings(self.n, self.d):
            residual = tuple(
                coloring[vertex] for vertex in self.other_vertices
            )
            row = self.coefficient_matrix[residual_index[residual]]
            root_color = coloring[self.root]
            coefficients.append(
                sum(
                    (
                        coefficient
                        * self.star_weight_matrix[column][root_color]
                        for column, coefficient in enumerate(row)
                    ),
                    Fraction(0),
                )
            )
        return ColoringTarget.from_dense(self.n, self.d, coefficients)


def build_star_linearization(
    witness: SparseWitness,
    root: int,
) -> StarLinearization:
    """Build the exact shared-cofactor system at one root."""

    root = _exact_integer(root, "star root")
    if root < 0 or root >= witness.n:
        raise KrennMultiStarIdentityError("star root is outside range(n)")
    other_vertices = tuple(
        vertex for vertex in range(witness.n) if vertex != root
    )
    residual_colorings = tuple(
        product(range(witness.d), repeat=witness.n - 1)
    )
    column_keys = tuple(
        (neighbor, color)
        for neighbor in other_vertices
        for color in range(witness.d)
    )
    coefficient_rows = []
    values = witness.as_dict()
    cofactor_cache: dict[
        tuple[int, tuple[int, ...]], Fraction
    ] = {}
    for residual in residual_colorings:
        coloring = [0] * witness.n
        for vertex, color in zip(
            other_vertices, residual, strict=True
        ):
            coloring[vertex] = color
        checked_coloring = tuple(coloring)
        cofactors = {}
        for neighbor in other_vertices:
            remaining = tuple(
                vertex
                for vertex in other_vertices
                if vertex != neighbor
            )
            cache_key = (
                neighbor,
                tuple(
                    checked_coloring[vertex]
                    for vertex in remaining
                ),
            )
            if cache_key not in cofactor_cache:
                cofactor_cache[cache_key] = _induced_matching_sum(
                    values,
                    witness.n,
                    witness.d,
                    checked_coloring,
                    remaining,
                )
            cofactors[neighbor] = cofactor_cache[cache_key]
        residual_by_vertex = dict(
            zip(other_vertices, residual, strict=True)
        )
        coefficient_rows.append(
            tuple(
                (
                    cofactors[neighbor]
                    if residual_by_vertex[neighbor] == color
                    else Fraction(0)
                )
                for neighbor, color in column_keys
            )
        )
    star_weight_matrix = tuple(
        tuple(
            _weight(
                values,
                witness.n,
                witness.d,
                root,
                neighbor,
                root_color,
                neighbor_color,
            )
            for root_color in range(witness.d)
        )
        for neighbor, neighbor_color in column_keys
    )
    return StarLinearization(
        n=witness.n,
        d=witness.d,
        root=root,
        other_vertices=other_vertices,
        residual_colorings=residual_colorings,
        column_keys=column_keys,
        coefficient_matrix=tuple(coefficient_rows),
        star_weight_matrix=star_weight_matrix,
    )


def monochromatic_matching_marginals(
    witness: SparseWitness,
    color: int,
) -> ExactMatrix:
    """Return ``C_rs = W_rs[aa] H_(V\\{r,s})(a,...,a)``."""

    color = _exact_integer(color, "marginal color")
    if color < 0 or color >= witness.d:
        raise KrennMultiStarIdentityError(
            "marginal color is outside range(d)"
        )
    coloring = (color,) * witness.n
    values = witness.as_dict()
    result = [
        [Fraction(0) for _second in range(witness.n)]
        for _first in range(witness.n)
    ]
    for first, second in canonical_edges(witness.n):
        value = _weight(
            values,
            witness.n,
            witness.d,
            first,
            second,
            color,
            color,
        ) * matching_cofactor(
            witness, coloring, first, second
        )
        result[first][second] = value
        result[second][first] = value
    return tuple(tuple(row) for row in result)


@dataclass(frozen=True)
class MultiStarIdentityCertificate:
    """Census of exact all-root and two-root identity checks."""

    n: int
    d: int
    star_coefficient_checks: int
    cofactor_recursion_checks: int
    shared_edge_entry_checks: int
    two_star_coefficient_checks: int
    all_star_expansions_exact: bool
    all_cofactor_recursions_exact: bool
    all_shared_edge_blocks_exact: bool
    all_two_star_expansions_exact: bool
    schema: str = MULTI_STAR_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        n, d = validate_parameters(
            _exact_integer(self.n, "certificate n"),
            _exact_integer(self.d, "certificate d"),
        )
        counts = (
            self.star_coefficient_checks,
            self.cofactor_recursion_checks,
            self.shared_edge_entry_checks,
            self.two_star_coefficient_checks,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts
        ):
            raise KrennMultiStarIdentityError(
                "multi-star check counts must be nonnegative integers"
            )
        expected_counts = (
            n * d**n,
            (
                comb(n, 2) * (n - 2) * d ** (n - 2)
                if n >= 4
                else 0
            ),
            comb(n, 2) * d**2,
            comb(n, 2) * d**n,
        )
        if counts != expected_counts:
            raise KrennMultiStarIdentityError(
                "multi-star check census is inconsistent"
            )
        booleans = (
            self.all_star_expansions_exact,
            self.all_cofactor_recursions_exact,
            self.all_shared_edge_blocks_exact,
            self.all_two_star_expansions_exact,
        )
        if (
            self.schema != MULTI_STAR_CERTIFICATE_SCHEMA
            or any(type(value) is not bool for value in booleans)
        ):
            raise KrennMultiStarIdentityError(
                "multi-star certificate schema or flags changed"
            )
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)

    @property
    def exact(self) -> bool:
        return (
            self.all_star_expansions_exact
            and self.all_cofactor_recursions_exact
            and self.all_shared_edge_blocks_exact
            and self.all_two_star_expansions_exact
        )


def certify_multi_star_identities(
    witness: SparseWitness,
) -> MultiStarIdentityCertificate:
    """Replay all one-root, two-root, cofactor, and gluing identities."""

    output = MatchingTensorMap(
        witness.n, witness.d
    ).evaluate_exact(witness)
    output_dense = output.dense_coefficients()
    colorings = tuple(enumerate_colorings(witness.n, witness.d))
    linearizations = tuple(
        build_star_linearization(witness, root)
        for root in range(witness.n)
    )

    star_results = tuple(
        linearization.reconstructed_target().dense_coefficients()
        == output_dense
        for linearization in linearizations
    )
    star_exact = all(star_results)
    star_checks = witness.n * len(colorings)

    shared_exact = True
    for first, second in canonical_edges(witness.n):
        left = linearizations[first].oriented_edge_matrix(second)
        right = linearizations[second].oriented_edge_matrix(first)
        comparison = all(
            left[a][b] == right[b][a]
            for a in range(witness.d)
            for b in range(witness.d)
        )
        shared_exact = comparison and shared_exact
    shared_checks = len(canonical_edges(witness.n)) * witness.d**2

    tensor_map = MatchingTensorMap(witness.n, witness.d)
    two_star_exact = True
    for first, second in canonical_edges(witness.n):
        for coloring in colorings:
            comparison = (
                two_star_expansion(
                    witness, coloring, first, second
                )
                == tensor_map.coefficient_exact(witness, coloring)
            )
            two_star_exact = comparison and two_star_exact
    two_star_checks = len(canonical_edges(witness.n)) * len(colorings)

    values = witness.as_dict()
    cofactor_exact = True
    cofactor_checks = 0
    if witness.n >= 4:
        for removed in canonical_edges(witness.n):
            residual = tuple(
                vertex
                for vertex in range(witness.n)
                if vertex not in removed
            )
            for residual_colors in product(
                range(witness.d), repeat=len(residual)
            ):
                coloring = [0] * witness.n
                for vertex, color in zip(
                    residual, residual_colors, strict=True
                ):
                    coloring[vertex] = color
                checked_coloring = tuple(coloring)
                left = _induced_matching_sum(
                    values,
                    witness.n,
                    witness.d,
                    checked_coloring,
                    residual,
                )
                for root in residual:
                    right = sum(
                        (
                            _weight(
                                values,
                                witness.n,
                                witness.d,
                                root,
                                neighbor,
                                checked_coloring[root],
                                checked_coloring[neighbor],
                            )
                            * _induced_matching_sum(
                                values,
                                witness.n,
                                witness.d,
                                checked_coloring,
                                tuple(
                                    vertex
                                    for vertex in residual
                                    if vertex
                                    not in (root, neighbor)
                                ),
                            )
                            for neighbor in residual
                            if neighbor != root
                        ),
                        Fraction(0),
                    )
                    comparison = left == right
                    cofactor_exact = comparison and cofactor_exact
                    cofactor_checks += 1

    certificate = MultiStarIdentityCertificate(
        n=witness.n,
        d=witness.d,
        star_coefficient_checks=star_checks,
        cofactor_recursion_checks=cofactor_checks,
        shared_edge_entry_checks=shared_checks,
        two_star_coefficient_checks=two_star_checks,
        all_star_expansions_exact=star_exact,
        all_cofactor_recursions_exact=cofactor_exact,
        all_shared_edge_blocks_exact=shared_exact,
        all_two_star_expansions_exact=two_star_exact,
    )
    if not certificate.exact:
        raise KrennMultiStarIdentityError(
            "an exact multi-star identity failed"
        )
    return certificate


__all__ = [
    "KrennMultiStarIdentityError",
    "MultiStarIdentityCertificate",
    "StarLinearization",
    "build_star_linearization",
    "certify_multi_star_identities",
    "induced_matching_sum",
    "matching_cofactor",
    "monochromatic_matching_marginals",
    "star_expansion",
    "two_star_expansion",
]
