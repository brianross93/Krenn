"""Bounded sparse-support search for the native ``n=6,d=4`` system.

The search in this module is combinatorial.  A support records coordinates
that are intended to carry nonzero complex weights.  If a zero-RHS equation
has exactly one feasible matching monomial, that monomial cannot cancel and
the support is obstructed.  Resolving such an obstruction requires completing
at least one alternative matching in the same equation.

Absence of singleton obstructions is only a necessary condition.  This module
does not solve for weights and never certifies a solution over ``C``.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable, Mapping

from experiments.krenn_quantum_graph.system import (
    Coloring,
    Monomial,
    SparsePolynomialSystem,
    coloring_from_index,
    enumerate_colorings,
    validate_parameters,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


SUPPORT_SEARCH_SCHEMA = "krenn-quantum-graph-support-closure-v1"
MAX_NODE_CAP = 1_000_000

# Four edge-disjoint one-factors.  Color ``a`` uses the three diagonal
# coordinates W[i,j,a,a] on factor ``a``.
N6_D4_SEED_FACTORS = (
    ((0, 1), (2, 3), (4, 5)),
    ((0, 2), (1, 4), (3, 5)),
    ((0, 3), (1, 5), (2, 4)),
    ((0, 4), (1, 3), (2, 5)),
)

N6_D4_SEED_DEFECTS = (
    (153, (0, 0, 2, 1, 2, 1), ((0, 1), (2, 4), (3, 5))),
    (1904, (1, 3, 1, 3, 0, 0), ((0, 2), (1, 3), (4, 5))),
    (2535, (2, 1, 3, 2, 1, 3), ((0, 3), (1, 4), (2, 5))),
    (3598, (3, 2, 0, 0, 3, 2), ((0, 4), (1, 5), (2, 3))),
)


class KrennSupportSearchError(ValueError):
    """A support, bounded plan, or support replay is malformed."""


def n6_d4_seed_support() -> tuple[int, ...]:
    """Return the canonical 12-coordinate near-solution support."""

    return tuple(
        sorted(
            variable_index(6, 4, i, j, color, color)
            for color, factor in enumerate(N6_D4_SEED_FACTORS)
            for i, j in factor
        )
    )


def n6_d4_seed_witness() -> SparseWitness:
    """Return the exact unit-weight seed; it is not a solution."""

    return SparseWitness.from_index_values(
        6, 4, ((index, 1) for index in n6_d4_seed_support())
    )


def n6_d4_seed_coordinates() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        variable_key(6, 4, index) for index in n6_d4_seed_support()
    )


def coloring_partition(coloring: Iterable[int]) -> tuple[int, ...]:
    """Return color-class sizes in decreasing order."""

    counts: dict[int, int] = {}
    for raw_color in coloring:
        color = int(raw_color)
        counts[color] = counts.get(color, 0) + 1
    if not counts:
        raise KrennSupportSearchError("a coloring cannot be empty")
    return tuple(sorted(counts.values(), reverse=True))


def color_partition_census(
    n: int, d: int
) -> Mapping[tuple[int, ...], int]:
    """Count all colorings by their unordered color-class sizes."""

    census: dict[tuple[int, ...], int] = {}
    for coloring in enumerate_colorings(n, d):
        partition = coloring_partition(coloring)
        census[partition] = census.get(partition, 0) + 1
    return dict(sorted(census.items(), reverse=True))


def _canonical_support(
    system: SparsePolynomialSystem, support: Iterable[int]
) -> tuple[int, ...]:
    values = tuple(map(int, support))
    if len(values) != len(set(values)):
        raise KrennSupportSearchError(
            "support coordinates must be unique"
        )
    canonical = tuple(sorted(values))
    if any(
        index < 0 or index >= system.variable_count
        for index in canonical
    ):
        raise KrennSupportSearchError(
            "support coordinate is outside the ambient system"
        )
    return canonical


def _support_mask(support: Iterable[int]) -> int:
    mask = 0
    for index in support:
        mask |= 1 << int(index)
    return mask


@dataclass(frozen=True)
class SupportIndex:
    """Bit-mask encoding of every matching monomial."""

    n: int
    d: int
    variable_count: int
    equation_offsets: tuple[int, ...]
    monomial_masks: tuple[int, ...]


def build_support_index(
    system: SparsePolynomialSystem,
) -> SupportIndex:
    return SupportIndex(
        n=system.n,
        d=system.d,
        variable_count=system.variable_count,
        equation_offsets=system.equation_offsets,
        monomial_masks=tuple(
            _support_mask(monomial)
            for monomial in system.monomial_variable_indices
        ),
    )


def _check_index(
    system: SparsePolynomialSystem, index: SupportIndex
) -> None:
    if (
        (index.n, index.d) != (system.n, system.d)
        or index.variable_count != system.variable_count
        or index.equation_offsets != system.equation_offsets
        or len(index.monomial_masks) != system.monomial_count
    ):
        raise KrennSupportSearchError(
            "support index does not match the polynomial system"
        )


@dataclass(frozen=True)
class SupportProfile:
    """Complete feasible-matching census for one exact support."""

    n: int
    d: int
    support: tuple[int, ...]
    feasible_matching_counts: tuple[int, ...]
    missing_constant_equations: tuple[int, ...]
    singleton_nonconstant_equations: tuple[int, ...]
    unit_weight_defect_equations: tuple[int, ...]

    @property
    def support_size(self) -> int:
        return len(self.support)

    @property
    def active_equation_count(self) -> int:
        return sum(count > 0 for count in self.feasible_matching_counts)

    @property
    def singleton_free_necessary_condition(self) -> bool:
        return (
            not self.missing_constant_equations
            and not self.singleton_nonconstant_equations
        )

    @property
    def solution_certified(self) -> bool:
        return False

    @property
    def proof_over_C(self) -> bool:
        return False


def support_profile(
    system: SparsePolynomialSystem,
    support: Iterable[int],
    *,
    index: SupportIndex | None = None,
) -> SupportProfile:
    """Count feasible matching monomials in every equation."""

    canonical = _canonical_support(system, support)
    index = build_support_index(system) if index is None else index
    _check_index(system, index)
    support_mask = _support_mask(canonical)
    counts: list[int] = []
    missing_constant: list[int] = []
    singleton_nonconstant: list[int] = []
    unit_defects: list[int] = []
    for equation in range(system.equation_count):
        start = index.equation_offsets[equation]
        stop = index.equation_offsets[equation + 1]
        count = sum(
            (monomial_mask & support_mask) == monomial_mask
            for monomial_mask in index.monomial_masks[start:stop]
        )
        counts.append(count)
        rhs = system.rhs_values[equation]
        if rhs and count == 0:
            missing_constant.append(equation)
        if not rhs and count == 1:
            singleton_nonconstant.append(equation)
        # With every selected coordinate set to one, the equation value is
        # exactly the number of feasible matching monomials.
        if count != rhs:
            unit_defects.append(equation)
    return SupportProfile(
        n=system.n,
        d=system.d,
        support=canonical,
        feasible_matching_counts=tuple(counts),
        missing_constant_equations=tuple(missing_constant),
        singleton_nonconstant_equations=tuple(
            singleton_nonconstant
        ),
        unit_weight_defect_equations=tuple(unit_defects),
    )


@dataclass(frozen=True)
class SingletonObstruction:
    """One zero-RHS equation with one noncancellable support monomial."""

    equation: int
    coloring: Coloring
    matching_index: int
    monomial: Monomial


def singleton_obstructions(
    system: SparsePolynomialSystem,
    support: Iterable[int],
    *,
    index: SupportIndex | None = None,
    profile: SupportProfile | None = None,
) -> tuple[SingletonObstruction, ...]:
    canonical = _canonical_support(system, support)
    index = build_support_index(system) if index is None else index
    _check_index(system, index)
    profile = (
        support_profile(system, canonical, index=index)
        if profile is None
        else profile
    )
    if profile.support != canonical:
        raise KrennSupportSearchError(
            "support profile belongs to another support"
        )
    support_mask = _support_mask(canonical)
    result: list[SingletonObstruction] = []
    for equation in profile.singleton_nonconstant_equations:
        feasible = tuple(
            (matching_index, monomial)
            for matching_index, monomial in enumerate(
                system.equation_monomials(equation)
            )
            if (
                _support_mask(monomial) & support_mask
            ) == _support_mask(monomial)
        )
        if len(feasible) != 1:
            raise KrennSupportSearchError(
                "singleton obstruction failed semantic replay"
            )
        matching_index, monomial = feasible[0]
        result.append(
            SingletonObstruction(
                equation=equation,
                coloring=coloring_from_index(
                    system.n, system.d, equation
                ),
                matching_index=matching_index,
                monomial=monomial,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class SupportClosurePlan:
    n: int
    d: int
    initial_support: tuple[int, ...]
    node_cap: int
    support_cap: int
    schema: str = SUPPORT_SEARCH_SCHEMA

    def __post_init__(self) -> None:
        n, d = validate_parameters(self.n, self.d)
        initial = tuple(map(int, self.initial_support))
        node_cap = int(self.node_cap)
        support_cap = int(self.support_cap)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "initial_support", initial)
        object.__setattr__(self, "node_cap", node_cap)
        object.__setattr__(self, "support_cap", support_cap)
        if self.schema != SUPPORT_SEARCH_SCHEMA:
            raise KrennSupportSearchError(
                "support-search schema changed"
            )
        ambient = variable_count(n, d)
        if (
            initial != tuple(sorted(initial))
            or len(initial) != len(set(initial))
            or any(index < 0 or index >= ambient for index in initial)
        ):
            raise KrennSupportSearchError(
                "initial support is not canonical"
            )
        if not 1 <= node_cap <= MAX_NODE_CAP:
            raise KrennSupportSearchError(
                "node cap is outside the safety bound"
            )
        if not len(initial) <= support_cap <= ambient:
            raise KrennSupportSearchError(
                "support cap is outside the ambient range"
            )

    @property
    def symmetry_weight_equalities(self) -> int:
        return 0


@dataclass(frozen=True)
class SupportClosureResult:
    plan: SupportClosurePlan
    termination: str
    nodes_examined: int
    supports_discovered: int
    frontier_remaining: int
    singleton_free_supports: tuple[tuple[int, ...], ...]
    necessary_condition_only: bool = True
    solution_certified: bool = False
    proof_over_C: bool = False
    nonexistence_proved: bool = False
    symmetry_weight_equalities: int = 0

    def __post_init__(self) -> None:
        if (
            self.necessary_condition_only is not True
            or self.solution_certified is not False
            or self.proof_over_C is not False
            or self.nonexistence_proved is not False
            or self.symmetry_weight_equalities != 0
        ):
            raise KrennSupportSearchError(
                "support-search claim boundary changed"
            )
        if self.termination not in {
            "singleton-free-support-found",
            "node-cap-reached",
            "frontier-exhausted-under-support-cap",
        }:
            raise KrennSupportSearchError(
                "support-search termination changed"
            )
        if (
            self.termination == "singleton-free-support-found"
        ) != bool(self.singleton_free_supports):
            raise KrennSupportSearchError(
                "support-search status is not content-derived"
            )


def _branch_monomials(
    system: SparsePolynomialSystem,
    profile: SupportProfile,
    obstruction: SingletonObstruction | None,
) -> tuple[Monomial, ...]:
    if obstruction is not None:
        return tuple(
            monomial
            for matching_index, monomial in enumerate(
                system.equation_monomials(obstruction.equation)
            )
            if matching_index != obstruction.matching_index
        )
    if not profile.missing_constant_equations:
        return ()
    return system.equation_monomials(
        profile.missing_constant_equations[0]
    )


def bounded_support_closure(
    system: SparsePolynomialSystem,
    plan: SupportClosurePlan,
) -> SupportClosureResult:
    """Search support supersets until a cap or a necessary template.

    The first obstruction is chosen by equation order.  Children complete
    alternative matching monomials and are ordered by support size followed
    by lexicographic variable indices.
    """

    if (system.n, system.d) != (plan.n, plan.d):
        raise KrennSupportSearchError(
            "support-search plan does not match the system"
        )
    initial = _canonical_support(system, plan.initial_support)
    index = build_support_index(system)
    frontier: list[tuple[int, tuple[int, ...]]] = [
        (len(initial), initial)
    ]
    seen = {initial}
    nodes_examined = 0
    found: tuple[tuple[int, ...], ...] = ()

    while frontier and nodes_examined < plan.node_cap:
        _size, support = heapq.heappop(frontier)
        nodes_examined += 1
        profile = support_profile(system, support, index=index)
        if profile.singleton_free_necessary_condition:
            found = (support,)
            break
        obstructions = singleton_obstructions(
            system,
            support,
            index=index,
            profile=profile,
        )
        obstruction = obstructions[0] if obstructions else None
        for monomial in _branch_monomials(
            system, profile, obstruction
        ):
            child = tuple(sorted(set(support).union(monomial)))
            if len(child) > plan.support_cap or child in seen:
                continue
            seen.add(child)
            heapq.heappush(frontier, (len(child), child))

    if found:
        termination = "singleton-free-support-found"
    elif frontier:
        termination = "node-cap-reached"
    else:
        termination = "frontier-exhausted-under-support-cap"
    return SupportClosureResult(
        plan=plan,
        termination=termination,
        nodes_examined=nodes_examined,
        supports_discovered=len(seen),
        frontier_remaining=len(frontier),
        singleton_free_supports=found,
    )
