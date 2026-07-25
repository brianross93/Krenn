"""Exact finite representation censuses for the Krenn tensor map.

The module records two structural decompositions.

The permutation module on perfect matchings of ``K_(2m)`` is the
multiplicity-free sum of the symmetric-group irreducibles indexed by doubled
partitions ``2*lambda`` for ``lambda`` a partition of ``m``.

For three colors, write the color permutation module as ``1 + U`` with
``U`` the two-dimensional standard representation of ``S_3``.  Splitting an
edge matrix into endpoint-symmetric and endpoint-antisymmetric parts gives
the ``S_n x S_3`` decomposition of the qutrit weight space used below.

These are representation and dimension censuses.  They do not imply that an
arbitrary solution has a nontrivial stabilizer or can be symmetrized.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial

from experiments.krenn_quantum_graph.system import (
    perfect_matching_count,
    perfect_matchings,
    variable_count,
)


class KrennSymmetryStructureError(ValueError):
    """A partition or representation census is malformed."""


def partitions(total: int) -> tuple[tuple[int, ...], ...]:
    """Return integer partitions in deterministic reverse lexicographic order."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise KrennSymmetryStructureError(
            "partition total must be a nonnegative integer"
        )

    def recurse(
        remainder: int,
        maximum: int,
    ) -> tuple[tuple[int, ...], ...]:
        if remainder == 0:
            return ((),)
        result = []
        for first in range(min(remainder, maximum), 0, -1):
            for tail in recurse(remainder - first, first):
                result.append((first, *tail))
        return tuple(result)

    return recurse(total, total)


def hook_length_dimension(partition: tuple[int, ...]) -> int:
    """Return the dimension of the corresponding symmetric-group irrep."""

    shape = tuple(partition)
    if (
        not shape
        or any(
            isinstance(part, bool) or not isinstance(part, int)
            for part in shape
        )
        or any(part < 1 for part in shape)
        or tuple(sorted(shape, reverse=True)) != shape
    ):
        raise KrennSymmetryStructureError(
            "Young diagram must be a nonempty integer partition"
        )
    hook_product = 1
    for row, row_length in enumerate(shape):
        for column in range(row_length):
            below = sum(
                other_length > column
                for other_length in shape[row + 1 :]
            )
            hook_product *= row_length - column + below
    return factorial(sum(shape)) // hook_product


@dataclass(frozen=True)
class SymmetricGroupIrrep:
    partition: tuple[int, ...]
    multiplicity: int = 1

    def __post_init__(self) -> None:
        partition = tuple(self.partition)
        if (
            isinstance(self.multiplicity, bool)
            or not isinstance(self.multiplicity, int)
            or self.multiplicity < 1
        ):
            raise KrennSymmetryStructureError(
                "irrep multiplicity must be a positive integer"
            )
        hook_length_dimension(partition)
        object.__setattr__(self, "partition", partition)

    @property
    def dimension(self) -> int:
        return hook_length_dimension(self.partition)

    @property
    def total_dimension(self) -> int:
        return self.multiplicity * self.dimension


def perfect_matching_module(
    n: int,
) -> tuple[SymmetricGroupIrrep, ...]:
    """Return ``C[PM(K_n)] = direct_sum_(lambda |- n/2) [2 lambda]``."""

    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n < 2
        or n % 2
    ):
        raise KrennSymmetryStructureError(
            "perfect-matching module needs positive even n"
        )
    result = tuple(
        SymmetricGroupIrrep(
            partition=tuple(2 * part for part in partition)
        )
        for partition in partitions(n // 2)
    )
    if sum(irrep.total_dimension for irrep in result) != (
        perfect_matching_count(n)
    ):
        raise KrennSymmetryStructureError(
            "perfect-matching module dimension census failed"
        )
    return result


_COLOR_DIMENSIONS = {
    "trivial": 1,
    "standard": 2,
    "sign": 1,
}


@dataclass(frozen=True)
class ProductIrrep:
    """One ``S_n x S_3`` constituent with an explicit multiplicity."""

    vertex_partition: tuple[int, ...]
    color_irrep: str
    multiplicity: int
    sector: str

    def __post_init__(self) -> None:
        partition = tuple(self.vertex_partition)
        if (
            self.color_irrep not in _COLOR_DIMENSIONS
            or isinstance(self.multiplicity, bool)
            or not isinstance(self.multiplicity, int)
            or self.multiplicity < 1
            or self.sector
            not in {"endpoint_symmetric", "endpoint_antisymmetric"}
        ):
            raise KrennSymmetryStructureError(
                "invalid product-irrep record"
            )
        hook_length_dimension(partition)
        object.__setattr__(self, "vertex_partition", partition)

    @property
    def vertex_dimension(self) -> int:
        return hook_length_dimension(self.vertex_partition)

    @property
    def color_dimension(self) -> int:
        return _COLOR_DIMENSIONS[self.color_irrep]

    @property
    def total_dimension(self) -> int:
        return (
            self.multiplicity
            * self.vertex_dimension
            * self.color_dimension
        )


def qutrit_weight_space_decomposition(
    n: int,
) -> tuple[ProductIrrep, ...]:
    """Return the exact ``S_n x S_3`` census of all qutrit edge weights."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 4:
        raise KrennSymmetryStructureError(
            "stable qutrit decomposition is recorded for n at least 4"
        )
    symmetric_vertex_parts = (
        (n,),
        (n - 1, 1),
        (n - 2, 2),
    )
    antisymmetric_vertex_parts = (
        (n - 1, 1),
        (n - 2, 1, 1),
    )
    result = []
    for partition in symmetric_vertex_parts:
        result.extend(
            (
                ProductIrrep(
                    partition,
                    "trivial",
                    2,
                    "endpoint_symmetric",
                ),
                ProductIrrep(
                    partition,
                    "standard",
                    2,
                    "endpoint_symmetric",
                ),
            )
        )
    for partition in antisymmetric_vertex_parts:
        result.extend(
            (
                ProductIrrep(
                    partition,
                    "standard",
                    1,
                    "endpoint_antisymmetric",
                ),
                ProductIrrep(
                    partition,
                    "sign",
                    1,
                    "endpoint_antisymmetric",
                ),
            )
        )
    decomposition = tuple(result)
    if sum(item.total_dimension for item in decomposition) != (
        variable_count(n, 3)
    ):
        raise KrennSymmetryStructureError(
            "qutrit weight-space dimension census failed"
        )
    return decomposition


def k4_matching_action_audit() -> dict[str, object]:
    """Compute the exceptional ``S4 -> S3`` matching action exactly."""

    matchings = perfect_matchings(4)
    matching_index = {
        frozenset(matching): index
        for index, matching in enumerate(matchings)
    }
    action_by_permutation = {}
    for permutation in permutations(range(4)):
        action = []
        for matching in matchings:
            image = frozenset(
                tuple(
                    sorted(
                        (
                            permutation[first],
                            permutation[second],
                        )
                    )
                )
                for first, second in matching
            )
            action.append(matching_index[image])
        action_by_permutation[permutation] = tuple(action)
    image = set(action_by_permutation.values())
    identity_action = tuple(range(len(matchings)))
    kernel = tuple(
        permutation
        for permutation, action in action_by_permutation.items()
        if action == identity_action
    )
    return {
        "vertex_group_order": len(action_by_permutation),
        "matching_count": len(matchings),
        "image_order": len(image),
        "kernel_order": len(kernel),
        "kernel": [list(permutation) for permutation in kernel],
        "surjective_to_S3": len(image) == factorial(3),
        "kernel_is_klein_four": set(kernel)
        == {
            (0, 1, 2, 3),
            (1, 0, 3, 2),
            (2, 3, 0, 1),
            (3, 2, 1, 0),
        },
    }


__all__ = [
    "KrennSymmetryStructureError",
    "ProductIrrep",
    "SymmetricGroupIrrep",
    "hook_length_dimension",
    "k4_matching_action_audit",
    "partitions",
    "perfect_matching_module",
    "qutrit_weight_space_decomposition",
]
