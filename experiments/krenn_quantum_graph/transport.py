"""Exact ``S_n x S_d`` transport for Krenn variables and witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from experiments.krenn_quantum_graph.system import (
    Coloring,
    KrennSystemError,
    Monomial,
    SparsePolynomialSystem,
    VariableKey,
    coloring_index,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


class KrennTransportError(ValueError):
    """A permutation or transport replay is invalid."""


def validate_permutation(
    permutation: Sequence[int], size: int
) -> tuple[int, ...]:
    permutation = tuple(map(int, permutation))
    size = int(size)
    if len(permutation) != size or set(permutation) != set(range(size)):
        raise KrennTransportError(
            f"expected a permutation of range({size})"
        )
    return permutation


def identity_permutation(size: int) -> tuple[int, ...]:
    size = int(size)
    if size < 0:
        raise KrennTransportError("permutation size cannot be negative")
    return tuple(range(size))


def inverse_permutation(
    permutation: Sequence[int],
) -> tuple[int, ...]:
    permutation = validate_permutation(permutation, len(permutation))
    inverse = [0] * len(permutation)
    for old, new in enumerate(permutation):
        inverse[new] = old
    return tuple(inverse)


def compose_permutations(
    after: Sequence[int], before: Sequence[int]
) -> tuple[int, ...]:
    """Return the old-to-new permutation ``after o before``."""

    before = validate_permutation(before, len(before))
    after = validate_permutation(after, len(before))
    return tuple(after[before[index]] for index in range(len(before)))


def transport_variable_key(
    n: int,
    d: int,
    key: VariableKey,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> VariableKey:
    """Push a canonical variable forward, swapping slots on edge reversal."""

    vertex_permutation = validate_permutation(vertex_permutation, n)
    color_permutation = validate_permutation(color_permutation, d)
    i, j, a, b = map(int, key)
    if not (0 <= i < j < n and 0 <= a < d and 0 <= b < d):
        raise KrennTransportError(
            "transport input must be a canonical variable key"
        )
    target_i = vertex_permutation[i]
    target_j = vertex_permutation[j]
    target_a = color_permutation[a]
    target_b = color_permutation[b]
    if target_i < target_j:
        return target_i, target_j, target_a, target_b
    return target_j, target_i, target_b, target_a


def transport_variable_index(
    n: int,
    d: int,
    index: int,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> int:
    return variable_index(
        n,
        d,
        *transport_variable_key(
            n,
            d,
            variable_key(n, d, index),
            vertex_permutation,
            color_permutation,
        ),
    )


def variable_permutation(
    n: int,
    d: int,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, ...]:
    count = len(range(n * (n - 1) // 2 * d * d))
    return tuple(
        transport_variable_index(
            n,
            d,
            index,
            vertex_permutation,
            color_permutation,
        )
        for index in range(count)
    )


def transport_coloring(
    coloring: Sequence[int],
    d: int,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> Coloring:
    """Push ``c`` forward via ``c'[sigma(i)] = tau(c[i])``."""

    coloring = tuple(map(int, coloring))
    n = len(coloring)
    vertex_permutation = validate_permutation(vertex_permutation, n)
    color_permutation = validate_permutation(color_permutation, d)
    if any(color < 0 or color >= d for color in coloring):
        raise KrennTransportError("coloring has an out-of-range color")
    result = [0] * n
    for old_vertex, color in enumerate(coloring):
        result[vertex_permutation[old_vertex]] = color_permutation[color]
    return tuple(result)


def equation_permutation(
    n: int,
    d: int,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        coloring_index(
            n,
            d,
            transport_coloring(
                _coloring_from_index_local(n, d, equation),
                d,
                vertex_permutation,
                color_permutation,
            ),
        )
        for equation in range(d**n)
    )


def _coloring_from_index_local(n: int, d: int, index: int) -> Coloring:
    digits = [0] * n
    for position in range(n - 1, -1, -1):
        index, digits[position] = divmod(index, d)
    return tuple(digits)


def transport_monomial(
    system: SparsePolynomialSystem,
    monomial: Monomial,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> Monomial:
    """Transport a commutative monomial into canonical variable order."""

    return tuple(
        sorted(
            transport_variable_index(
                system.n,
                system.d,
                variable,
                vertex_permutation,
                color_permutation,
            )
            for variable in monomial
        )
    )


def transport_witness(
    witness: SparseWitness,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> SparseWitness:
    vertex_permutation = validate_permutation(
        vertex_permutation, witness.n
    )
    color_permutation = validate_permutation(
        color_permutation, witness.d
    )
    transported = {
        transport_variable_index(
            witness.n,
            witness.d,
            index,
            vertex_permutation,
            color_permutation,
        ): value
        for index, value in witness.entries
    }
    if len(transported) != witness.support_size:
        raise KrennTransportError(
            "a permutation unexpectedly collided witness coordinates"
        )
    return SparseWitness.from_index_values(
        witness.n, witness.d, transported
    )


@dataclass(frozen=True)
class TransportCertificate:
    n: int
    d: int
    vertex_permutation: tuple[int, ...]
    color_permutation: tuple[int, ...]
    variable_bijection: bool
    equation_bijection: bool
    rhs_preserved: bool
    all_monomials_preserved: bool
    reversed_variable_count: int

    @property
    def exact(self) -> bool:
        return (
            self.variable_bijection
            and self.equation_bijection
            and self.rhs_preserved
            and self.all_monomials_preserved
        )


def certify_system_transport(
    system: SparsePolynomialSystem,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> TransportCertificate:
    """Replay the complete system automorphism without orbit identifications."""

    vertex_permutation = validate_permutation(
        vertex_permutation, system.n
    )
    color_permutation = validate_permutation(
        color_permutation, system.d
    )
    variables = variable_permutation(
        system.n,
        system.d,
        vertex_permutation,
        color_permutation,
    )
    equations = equation_permutation(
        system.n,
        system.d,
        vertex_permutation,
        color_permutation,
    )
    variable_bijection = set(variables) == set(
        range(system.variable_count)
    )
    equation_bijection = set(equations) == set(
        range(system.equation_count)
    )
    rhs_preserved = True
    monomials_preserved = True
    for source_equation, target_equation in enumerate(equations):
        rhs_preserved &= (
            system.rhs_values[source_equation]
            == system.rhs_values[target_equation]
        )
        transported = tuple(
            sorted(
                transport_monomial(
                    system,
                    monomial,
                    vertex_permutation,
                    color_permutation,
                )
                for monomial in system.equation_monomials(
                    source_equation
                )
            )
        )
        expected = tuple(
            sorted(
                tuple(sorted(monomial))
                for monomial in system.equation_monomials(
                    target_equation
                )
            )
        )
        monomials_preserved &= transported == expected
    reversed_count = sum(
        vertex_permutation[i] > vertex_permutation[j]
        for i in range(system.n)
        for j in range(i + 1, system.n)
        for _a in range(system.d)
        for _b in range(system.d)
    )
    certificate = TransportCertificate(
        n=system.n,
        d=system.d,
        vertex_permutation=vertex_permutation,
        color_permutation=color_permutation,
        variable_bijection=variable_bijection,
        equation_bijection=equation_bijection,
        rhs_preserved=rhs_preserved,
        all_monomials_preserved=monomials_preserved,
        reversed_variable_count=reversed_count,
    )
    if not certificate.exact:
        raise KrennSystemError("S_n x S_d system transport replay failed")
    return certificate

