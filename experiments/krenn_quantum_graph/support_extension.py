"""Bounded support-extension audit for the natural ``n=6,d=3`` seed.

The natural nine-slot support has one mixed coefficient, coloring
``002121``, with one active perfect-matching monomial.  This module answers
three narrowly separated questions without imposing any weight equalities:

1. How many new coordinates are needed to give that coefficient a second
   monomial?
2. Can a finite, nonzero assignment on such a support avoid singleton mixed
   equations?
3. What is the first support size passing that combinatorial test, and do
   those supports pass the resulting binomial equations?

The exact answers are:

* two new coordinates are necessary and sufficient locally;
* each of the six two-coordinate repairs creates two singleton defects;
* exhaustive missing-set closure visits 1,632,189 raw supports through total
  support size 21 and finds exactly six singleton-free supports, all of size
  21; and
* every one of those six supports has an odd three-binomial exponent cycle,
  forcing ``1 = -1`` over characteristic different from two.

Consequently, an exact finite extension retaining all nine natural nonzero
slots needs total support at least 22.  This is not a no-go theorem at support
22 or in the unrestricted 135-variable system.

The cap-21 exhaustive replay is intentionally exposed as a separate bounded
operation because it is substantially more expensive than routine semantic
verification of the six stored receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
from itertools import combinations, product
from typing import Iterable, Sequence

from experiments.krenn_quantum_graph.system import (
    Monomial,
    SparsePolynomialSystem,
    coloring_from_index,
    generate_sparse_system,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    n6_d3_seed_values,
)


N = 6
D = 3
VICTIM_EQUATION = 70
VICTIM_COLORING = (0, 0, 2, 1, 2, 1)
AMBIENT_VARIABLES = 135
NATURAL_SUPPORT_SIZE = 9
MAX_REPLAY_SUPPORT = 21
DEFAULT_NODE_CAP = 2_000_000

EXPECTED_DISCOVERED_BY_ADDITION_SIZE = (
    1,
    0,
    6,
    8,
    36,
    110,
    395,
    1_310,
    5_002,
    18_571,
    71_309,
    294_869,
    1_240_572,
)
EXPECTED_CAP20_SUPPORTS = 391_617
EXPECTED_CAP21_SUPPORTS = 1_632_189


class KrennSupportExtensionError(ValueError):
    """A support, bounded replay, or binomial receipt is malformed."""


def natural_support() -> tuple[int, ...]:
    """Return the exact nine nonzero coordinates of the natural seed."""

    return tuple(
        index
        for index, value in enumerate(n6_d3_seed_values())
        if value
    )


def _mask(indices: Iterable[int]) -> int:
    result = 0
    for index in indices:
        index = int(index)
        if not 0 <= index < AMBIENT_VARIABLES:
            raise KrennSupportExtensionError(
                "support coordinate is outside the n=6,d=3 ambient space"
            )
        result |= 1 << index
    return result


def _indices(mask: int) -> tuple[int, ...]:
    if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0:
        raise KrennSupportExtensionError(
            "support mask must be a nonnegative integer"
        )
    result = []
    while mask:
        low_bit = mask & -mask
        result.append(low_bit.bit_length() - 1)
        mask -= low_bit
    return tuple(result)


@lru_cache(maxsize=1)
def _system() -> SparsePolynomialSystem:
    return generate_sparse_system(N, D)


@lru_cache(maxsize=1)
def _base_mask() -> int:
    return _mask(natural_support())


@lru_cache(maxsize=1)
def _mixed_missing_index() -> tuple[
    tuple[int, tuple[int, ...]], ...
]:
    """Store every mixed monomial only by coordinates missing from the seed."""

    system = _system()
    base = _base_mask()
    rows = []
    for equation, rhs in enumerate(system.rhs_values):
        if rhs:
            continue
        rows.append(
            (
                equation,
                tuple(
                    _mask(monomial) & ~base
                    for monomial in system.equation_monomials(equation)
                ),
            )
        )
    return tuple(rows)


def active_matching_indices(
    support: Iterable[int],
    equation: int,
) -> tuple[int, ...]:
    """Return all matching terms contained in one raw coordinate support."""

    system = _system()
    equation = int(equation)
    if not 0 <= equation < system.equation_count:
        raise KrennSupportExtensionError(
            "equation is outside the n=6,d=3 system"
        )
    support_mask = _mask(support)
    return tuple(
        matching_index
        for matching_index, monomial in enumerate(
            system.equation_monomials(equation)
        )
        if _mask(monomial) & ~support_mask == 0
    )


def victim_missing_coordinate_histogram() -> tuple[int, int, int, int]:
    """Count victim monomials requiring zero, one, two, or three additions."""

    base = set(natural_support())
    counts = [0, 0, 0, 0]
    for monomial in _system().equation_monomials(VICTIM_EQUATION):
        counts[len(set(monomial) - base)] += 1
    result = tuple(counts)
    if result != (1, 0, 6, 8):
        raise KrennSupportExtensionError(
            "victim missing-coordinate histogram changed"
        )
    return result


@dataclass(frozen=True)
class MinimalVictimRepair:
    """One minimum-cardinality second monomial in equation 70."""

    matching_index: int
    matching_monomial: Monomial
    shared_natural_coordinate: int
    added_coordinates: tuple[int, int]
    forced_singleton_equations: tuple[int, int]

    @property
    def added_variable_keys(
        self,
    ) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            variable_key(N, D, index)
            for index in self.added_coordinates
        )


def minimal_victim_repairs() -> tuple[MinimalVictimRepair, ...]:
    """Enumerate all six ways to repair the victim with two coordinates."""

    system = _system()
    base = set(natural_support())
    histogram = victim_missing_coordinate_histogram()
    if histogram[1] != 0:
        raise KrennSupportExtensionError(
            "the victim unexpectedly has a one-coordinate repair"
        )
    rows = []
    for matching_index, monomial in enumerate(
        system.equation_monomials(VICTIM_EQUATION)
    ):
        shared = tuple(variable for variable in monomial if variable in base)
        added = tuple(variable for variable in monomial if variable not in base)
        if len(added) != 2:
            continue
        support = tuple(sorted((*base, *added)))
        singletons = []
        for equation, rhs in enumerate(system.rhs_values):
            if rhs:
                continue
            active = active_matching_indices(support, equation)
            if len(active) == 1:
                singletons.append(equation)
        if len(shared) != 1 or len(singletons) != 2:
            raise KrennSupportExtensionError(
                "minimum victim repair failed singleton replay"
            )
        rows.append(
            MinimalVictimRepair(
                matching_index=matching_index,
                matching_monomial=monomial,
                shared_natural_coordinate=shared[0],
                added_coordinates=(added[0], added[1]),
                forced_singleton_equations=(
                    singletons[0],
                    singletons[1],
                ),
            )
        )
    if len(rows) != 6:
        raise KrennSupportExtensionError(
            "minimum victim-repair census changed"
        )
    return tuple(rows)


@dataclass(frozen=True)
class FiniteValuationAudit:
    """The nonnegative-valuation obstruction for a two-slot repair."""

    repair: MinimalVictimRepair
    natural_constant_product_orders: tuple[int, int, int]
    natural_coordinate_orders_forced_zero: bool
    added_order_sum: int
    added_coordinate_orders_forced_zero: bool
    singleton_orders: tuple[int, int]
    finite_regularization_possible: bool
    laurent_poles_excluded: bool = True


def finite_valuation_audits() -> tuple[FiniteValuationAudit, ...]:
    """Replay the leading-order argument with all coordinate orders finite.

    A finite limit means every coordinate order is nonnegative.  Each natural
    constant product has order zero, hence all nine natural coordinate orders
    are zero.  Matching the order-zero victim term forces the two added orders
    to sum to zero, hence both are zero.  Each forced singleton consequently
    also has order zero and cannot disappear.
    """

    system = _system()
    base = set(natural_support())
    constant_monomials = []
    for equation in (0, 364, 728):
        feasible = tuple(
            monomial
            for monomial in system.equation_monomials(equation)
            if set(monomial) <= base
        )
        if len(feasible) != 1:
            raise KrennSupportExtensionError(
                "natural constant term is not uniquely supported"
            )
        constant_monomials.append(feasible[0])
    covered = tuple(
        variable
        for monomial in constant_monomials
        for variable in monomial
    )
    if len(covered) != 9 or set(covered) != base:
        raise KrennSupportExtensionError(
            "constant products do not force all natural orders"
        )

    natural_orders = {variable: 0 for variable in base}
    natural_product_orders = tuple(
        sum(natural_orders[variable] for variable in monomial)
        for monomial in constant_monomials
    )
    result = []
    for repair in minimal_victim_repairs():
        alternative = system.equation_monomials(
            VICTIM_EQUATION
        )[repair.matching_index]
        if alternative != repair.matching_monomial:
            raise KrennSupportExtensionError(
                "victim repair matching index changed"
            )
        shared_order = natural_orders[
            repair.shared_natural_coordinate
        ]
        added_order_sum = -shared_order
        extended_orders = dict(natural_orders)
        extended_orders.update(
            (variable, 0) for variable in repair.added_coordinates
        )
        singleton_orders = []
        support = tuple(extended_orders)
        for equation in repair.forced_singleton_equations:
            active = active_matching_indices(support, equation)
            if len(active) != 1:
                raise KrennSupportExtensionError(
                    "forced singleton valuation receipt changed"
                )
            monomial = system.equation_monomials(equation)[active[0]]
            singleton_orders.append(
                sum(extended_orders[variable] for variable in monomial)
            )
        result.append(
            FiniteValuationAudit(
                repair=repair,
                natural_constant_product_orders=natural_product_orders,
                natural_coordinate_orders_forced_zero=True,
                added_order_sum=added_order_sum,
                added_coordinate_orders_forced_zero=(
                    added_order_sum == 0
                ),
                singleton_orders=tuple(singleton_orders),
                finite_regularization_possible=False,
            )
        )
    return tuple(result)


def natural_laurent_border_orders() -> tuple[int, ...]:
    """A no-addition Laurent degeneration with residual order one.

    The orders are ``+1`` on ``W[01,0,0]``, ``-1`` on ``W[23,0,0]``, and
    zero on the other seven natural coordinates.  All three constant products
    retain order zero, while the sole victim monomial has order one.
    """

    support = natural_support()
    orders = {
        variable: 0 for variable in support
    }
    orders[0] = 1
    orders[81] = -1
    result = tuple(orders[variable] for variable in support)
    system = _system()
    constant_orders = []
    for equation in (0, 364, 728):
        feasible = tuple(
            monomial
            for monomial in system.equation_monomials(equation)
            if all(variable in orders for variable in monomial)
        )
        if len(feasible) != 1:
            raise KrennSupportExtensionError(
                "natural constant matching failed support replay"
            )
        constant_orders.append(
            sum(orders[variable] for variable in feasible[0])
        )
    victim_order = sum(
        orders[variable]
        for variable in system.equation_monomials(VICTIM_EQUATION)[1]
    )
    if tuple(constant_orders) != (0, 0, 0) or victim_order != 1:
        raise KrennSupportExtensionError(
            "natural Laurent border orders failed replay"
        )
    return result


def _first_singleton_missing(
    added_mask: int,
) -> tuple[int, tuple[int, ...]] | None:
    for _equation, missing_sets in _mixed_missing_index():
        active = -1
        count = 0
        for matching_index, missing in enumerate(missing_sets):
            if missing & ~added_mask == 0:
                active = matching_index
                count += 1
                if count > 1:
                    break
        if count == 1:
            return active, missing_sets
    return None


@dataclass(frozen=True)
class SupportClosureReplay:
    """A deterministic bounded missing-set closure census."""

    max_total_support: int
    node_cap: int
    nodes_examined: int
    supports_discovered: int
    discovered_by_addition_size: tuple[int, ...]
    singleton_free_supports: tuple[tuple[int, ...], ...]
    termination: str
    raw_supports_without_symmetry_quotient: bool = True

    @property
    def exhaustive(self) -> bool:
        return self.termination == "frontier-exhausted"


def bounded_missing_set_closure(
    *,
    max_total_support: int,
    node_cap: int = DEFAULT_NODE_CAP,
) -> SupportClosureReplay:
    """Exhaust raw support extensions under an explicit size and node cap.

    Children complete an alternative matching in the first singleton mixed
    equation.  This is exhaustive: every singleton-free superset must contain
    at least one of those alternative monomials.  No symmetry quotient or
    weight equality is used.
    """

    max_total_support = int(max_total_support)
    node_cap = int(node_cap)
    if not NATURAL_SUPPORT_SIZE <= max_total_support <= MAX_REPLAY_SUPPORT:
        raise KrennSupportExtensionError(
            "bounded replay supports total sizes 9 through 21"
        )
    if not 1 <= node_cap <= DEFAULT_NODE_CAP:
        raise KrennSupportExtensionError(
            "bounded replay node cap is outside the certified range"
        )
    addition_cap = max_total_support - NATURAL_SUPPORT_SIZE
    frontier: list[tuple[int, int]] = [(0, 0)]
    seen = {0}
    found: list[int] = []
    nodes = 0

    while frontier and nodes < node_cap:
        _size, added = heapq.heappop(frontier)
        nodes += 1
        obstruction = _first_singleton_missing(added)
        if obstruction is None:
            found.append(added)
            continue
        active, missing_sets = obstruction
        for matching_index, missing in enumerate(missing_sets):
            if matching_index == active:
                continue
            child = added | missing
            child_size = child.bit_count()
            if child_size > addition_cap or child in seen:
                continue
            seen.add(child)
            heapq.heappush(frontier, (child_size, child))

    counts = [0] * (addition_cap + 1)
    for added in seen:
        counts[added.bit_count()] += 1
    base = set(natural_support())
    supports = tuple(
        tuple(sorted((*base, *_indices(added))))
        for added in sorted(found)
    )
    return SupportClosureReplay(
        max_total_support=max_total_support,
        node_cap=node_cap,
        nodes_examined=nodes,
        supports_discovered=len(seen),
        discovered_by_addition_size=tuple(counts),
        singleton_free_supports=supports,
        termination=(
            "frontier-exhausted" if not frontier else "node-cap-reached"
        ),
    )


# Exact output of the cap-21 raw replay.  Routine verification replays the
# content of these supports and their algebraic contradictions.  Calling
# bounded_missing_set_closure(max_total_support=21) independently regenerates
# the full 1,632,189-support census.
SIZE21_ADDITIONS = (
    (6, 18, 20, 24, 29, 35, 45, 47, 83, 87, 89, 92),
    (3, 9, 10, 12, 37, 40, 54, 55, 82, 84, 85, 118),
    (68, 70, 71, 76, 77, 79, 97, 106, 107, 112, 113, 122),
    (14, 16, 17, 22, 23, 25, 95, 103, 106, 113, 116, 124),
    (1, 36, 37, 55, 58, 63, 64, 66, 120, 127, 129, 130),
    (2, 27, 29, 47, 53, 72, 74, 78, 96, 128, 132, 134),
)


@dataclass(frozen=True)
class OddBinomialCycle:
    """Three two-term equations whose exponent ratios multiply to ``-1``."""

    equations: tuple[int, int, int]
    matching_pairs: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]
    signs: tuple[int, int, int]
    exponent_rows: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    exponent_relation_zero: bool
    rhs_product: int

    @property
    def signed_exponent_sum(self) -> tuple[int, ...]:
        return tuple(
            sum(
                self.signs[row] * self.exponent_rows[row][column]
                for row in range(3)
            )
            for column in range(len(self.exponent_rows[0]))
        )

    @property
    def contradiction_over_characteristic_not_two(self) -> bool:
        return (
            self.exponent_relation_zero
            and not any(self.signed_exponent_sum)
            and self.rhs_product == -1
        )


@dataclass(frozen=True)
class Size21SupportAudit:
    support: tuple[int, ...]
    added_coordinates: tuple[int, ...]
    mixed_equation_count: int
    active_terms_per_mixed_equation: tuple[int, ...]
    odd_cycle: OddBinomialCycle

    @property
    def singleton_free(self) -> bool:
        return all(
            count != 1 for count in self.active_terms_per_mixed_equation
        )

    @property
    def finite_nonzero_solution_possible(self) -> bool:
        return not self.odd_cycle.contradiction_over_characteristic_not_two


def _exponent_difference(
    support_positions: dict[int, int],
    left: Monomial,
    right: Monomial,
) -> tuple[int, ...]:
    row = [0] * len(support_positions)
    for variable in left:
        row[support_positions[variable]] += 1
    for variable in right:
        row[support_positions[variable]] -= 1
    return tuple(row)


def _find_odd_three_cycle(
    support: Sequence[int],
    binomials: Sequence[
        tuple[int, int, Monomial, int, Monomial]
    ],
) -> OddBinomialCycle:
    positions = {
        variable: position
        for position, variable in enumerate(support)
    }
    rows = tuple(
        _exponent_difference(positions, left, right)
        for _equation, _left_index, left, _right_index, right in binomials
    )
    for selected in combinations(range(len(rows)), 3):
        for signs in product((1, -1), repeat=3):
            if signs[0] != 1:
                continue
            if not all(
                sum(
                    signs[position] * rows[selected[position]][column]
                    for position in range(3)
                ) == 0
                for column in range(len(support))
            ):
                continue
            chosen = tuple(binomials[index] for index in selected)
            rhs_product = -1 if sum(signs) % 2 else 1
            return OddBinomialCycle(
                equations=tuple(row[0] for row in chosen),
                matching_pairs=tuple(
                    (row[1], row[3]) for row in chosen
                ),
                signs=(signs[0], signs[1], signs[2]),
                exponent_rows=tuple(rows[index] for index in selected),
                exponent_relation_zero=True,
                rhs_product=rhs_product,
            )
    raise KrennSupportExtensionError(
        "size-21 support has no odd three-binomial contradiction"
    )


def size21_support_audits() -> tuple[Size21SupportAudit, ...]:
    """Replay the six minimum singleton-free supports and their contradictions."""

    system = _system()
    base = set(natural_support())
    result = []
    for additions in SIZE21_ADDITIONS:
        support = tuple(sorted((*base, *additions)))
        if len(support) != 21 or len(set(support)) != 21:
            raise KrennSupportExtensionError(
                "stored size-21 support is malformed"
            )
        counts = []
        binomials = []
        for equation, rhs in enumerate(system.rhs_values):
            if rhs:
                continue
            active = tuple(
                (matching_index, monomial)
                for matching_index, monomial in enumerate(
                    system.equation_monomials(equation)
                )
                if all(variable in support for variable in monomial)
            )
            if not active:
                continue
            counts.append(len(active))
            if len(active) != 2:
                raise KrennSupportExtensionError(
                    "stored size-21 mixed equation is not binomial"
                )
            binomials.append(
                (
                    equation,
                    active[0][0],
                    active[0][1],
                    active[1][0],
                    active[1][1],
                )
            )
        cycle = _find_odd_three_cycle(support, binomials)
        result.append(
            Size21SupportAudit(
                support=support,
                added_coordinates=additions,
                mixed_equation_count=len(counts),
                active_terms_per_mixed_equation=tuple(counts),
                odd_cycle=cycle,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class SupportExtensionCertificate:
    """Fast semantic receipt plus a ledger for the separate exhaustive replay."""

    minimum_local_additions: int
    minimal_repairs: tuple[MinimalVictimRepair, ...]
    finite_valuation_audits: tuple[FiniteValuationAudit, ...]
    recorded_cap20_supports: int
    recorded_cap21_supports: int
    recorded_discovered_by_addition_size: tuple[int, ...]
    size21_audits: tuple[Size21SupportAudit, ...]
    finite_exact_total_support_lower_bound: int | None
    exhaustive_replay: SupportClosureReplay | None
    retains_all_natural_slots: bool = True
    symmetry_weight_equalities: int = 0
    unrestricted_nonexistence_proved: bool = False
    schema: str = "krenn.n6_d3.support_extension.v1"

    @property
    def exhaustive_census_recomputed(self) -> bool:
        """Whether this instance carries a completed cap-21 replay receipt."""

        return self.exhaustive_replay is not None

    def _validate(self) -> None:
        """Recompute compact semantics and reject altered receipts."""

        expected_repairs = minimal_victim_repairs()
        expected_valuations = finite_valuation_audits()
        expected_size21 = size21_support_audits()
        if (
            self.schema != "krenn.n6_d3.support_extension.v1"
            or self.minimum_local_additions != 2
            or self.minimal_repairs != expected_repairs
            or self.finite_valuation_audits != expected_valuations
            or self.recorded_cap20_supports != EXPECTED_CAP20_SUPPORTS
            or self.recorded_cap21_supports != EXPECTED_CAP21_SUPPORTS
            or self.recorded_discovered_by_addition_size
            != EXPECTED_DISCOVERED_BY_ADDITION_SIZE
            or self.size21_audits != expected_size21
            or self.finite_exact_total_support_lower_bound
            != (22 if self.exhaustive_replay is not None else None)
            or not self.retains_all_natural_slots
            or self.symmetry_weight_equalities != 0
            or self.unrestricted_nonexistence_proved
        ):
            raise KrennSupportExtensionError(
                "support-extension certificate was altered or is incomplete"
            )
        if self.exhaustive_replay is not None:
            _validate_cap21_replay(self.exhaustive_replay)

    def to_dict(self) -> dict:
        """Return a JSON-safe, fail-closed proof receipt."""

        self._validate()
        natural = natural_support()
        laurent_orders = natural_laurent_border_orders()
        repairs = []
        for repair, valuation in zip(
            self.minimal_repairs,
            self.finite_valuation_audits,
        ):
            repairs.append(
                {
                    "matching_index": repair.matching_index,
                    "matching_monomial": list(repair.matching_monomial),
                    "matching_variable_keys": [
                        list(variable_key(N, D, index))
                        for index in repair.matching_monomial
                    ],
                    "shared_natural_coordinate": (
                        repair.shared_natural_coordinate
                    ),
                    "added_coordinates": list(repair.added_coordinates),
                    "added_variable_keys": [
                        list(key) for key in repair.added_variable_keys
                    ],
                    "forced_singleton_equations": list(
                        repair.forced_singleton_equations
                    ),
                    "forced_singleton_colorings": [
                        list(coloring_from_index(N, D, equation))
                        for equation in repair.forced_singleton_equations
                    ],
                    "finite_valuation_receipt": {
                        "natural_constant_product_orders": list(
                            valuation.natural_constant_product_orders
                        ),
                        "natural_coordinate_orders_forced_zero": (
                            valuation
                            .natural_coordinate_orders_forced_zero
                        ),
                        "added_order_sum": valuation.added_order_sum,
                        "added_coordinate_orders_forced_zero": (
                            valuation
                            .added_coordinate_orders_forced_zero
                        ),
                        "singleton_orders": list(
                            valuation.singleton_orders
                        ),
                        "finite_regularization_possible": (
                            valuation.finite_regularization_possible
                        ),
                        "laurent_poles_excluded": (
                            valuation.laurent_poles_excluded
                        ),
                    },
                }
            )

        support_receipts = []
        for audit in self.size21_audits:
            cycle = audit.odd_cycle
            support_receipts.append(
                {
                    "support": list(audit.support),
                    "support_variable_keys": [
                        list(variable_key(N, D, index))
                        for index in audit.support
                    ],
                    "added_coordinates": list(audit.added_coordinates),
                    "mixed_equation_count": audit.mixed_equation_count,
                    "active_terms_per_mixed_equation": list(
                        audit.active_terms_per_mixed_equation
                    ),
                    "singleton_free": audit.singleton_free,
                    "finite_nonzero_solution_possible": (
                        audit.finite_nonzero_solution_possible
                    ),
                    "odd_three_binomial_cycle": {
                        "equations": list(cycle.equations),
                        "colorings": [
                            list(coloring_from_index(N, D, equation))
                            for equation in cycle.equations
                        ],
                        "matching_pairs": [
                            list(pair) for pair in cycle.matching_pairs
                        ],
                        "signs": list(cycle.signs),
                        "exponent_rows_in_support_order": [
                            list(row) for row in cycle.exponent_rows
                        ],
                        "signed_exponent_sum": list(
                            cycle.signed_exponent_sum
                        ),
                        "exponent_relation_zero": (
                            cycle.exponent_relation_zero
                        ),
                        "rhs_product": cycle.rhs_product,
                        "contradiction_over_characteristic_not_two": (
                            cycle
                            .contradiction_over_characteristic_not_two
                        ),
                    },
                }
            )

        replay = self.exhaustive_replay
        exhaustive = replay is not None
        statement = (
            "Among finite nonzero coordinate assignments over "
            "characteristic not two that retain all nine natural slots, "
            "the exhaustive raw support replay excludes total support at "
            "most 21; any such exact extension therefore has total support "
            "at least 22."
            if exhaustive
            else
            "The six local repairs and six stored size-21 algebraic "
            "receipts were rechecked, but this payload did not rerun the "
            "exhaustive raw support census and therefore does not by itself "
            "assert the support-22 lower bound."
        )
        return {
            "schema": self.schema,
            "parameters": {
                "n": N,
                "d": D,
                "ambient_variable_count": AMBIENT_VARIABLES,
                "victim_equation": VICTIM_EQUATION,
                "victim_coloring": list(VICTIM_COLORING),
                "natural_support": list(natural),
                "natural_support_variable_keys": [
                    list(variable_key(N, D, index))
                    for index in natural
                ],
            },
            "minimum_local_repair": {
                "minimum_added_coordinates": (
                    self.minimum_local_additions
                ),
                "victim_missing_coordinate_histogram_0_1_2_3": list(
                    victim_missing_coordinate_histogram()
                ),
                "repair_count": len(repairs),
                "repairs": repairs,
            },
            "laurent_border_warning": {
                "orders_in_natural_support_order": list(laurent_orders),
                "has_negative_coordinate_order": (
                    any(order < 0 for order in laurent_orders)
                ),
                "finite_regularization_receipt": False,
            },
            "bounded_missing_set_closure": {
                "raw_supports_without_symmetry_quotient": True,
                "retains_all_natural_slots": (
                    self.retains_all_natural_slots
                ),
                "maximum_total_support_replayed": (
                    replay.max_total_support if replay else None
                ),
                "node_cap": replay.node_cap if replay else None,
                "nodes_examined": replay.nodes_examined if replay else None,
                "supports_discovered": (
                    replay.supports_discovered if replay else None
                ),
                "discovered_layer_sum": (
                    sum(replay.discovered_by_addition_size)
                    if replay
                    else None
                ),
                "discovered_by_addition_size": (
                    list(replay.discovered_by_addition_size)
                    if replay
                    else None
                ),
                "termination": replay.termination if replay else None,
                "exhaustive_census_recomputed": exhaustive,
                "recorded_cap20_supports": (
                    self.recorded_cap20_supports
                ),
                "recorded_cap21_supports": (
                    self.recorded_cap21_supports
                ),
                "recorded_discovered_by_addition_size": list(
                    self.recorded_discovered_by_addition_size
                ),
                "singleton_free_size21_support_count": (
                    len(support_receipts)
                ),
                "size21_support_receipts": support_receipts,
            },
            "claim_boundary": {
                "statement": statement,
                "field_characteristic_not_two": True,
                "finite_nonzero_coordinates_on_declared_support": True,
                "retains_all_nine_natural_slots": True,
                "raw_supports_without_symmetry_quotient": True,
                "symmetry_weight_equalities_imposed": (
                    self.symmetry_weight_equalities
                ),
                "cap21_exhaustive_census_replayed": exhaustive,
                "finite_exact_total_support_lower_bound": (
                    self.finite_exact_total_support_lower_bound
                ),
                "finite_exact_total_support_at_least_22_proved": (
                    exhaustive
                ),
                "supports_of_total_size_22_or_more_excluded": False,
                "supports_dropping_a_natural_slot_excluded": False,
                "Laurent_or_border_paths_excluded": False,
                "unrestricted_nonexistence_proved": (
                    self.unrestricted_nonexistence_proved
                ),
            },
        }


def _validate_cap21_replay(replay: SupportClosureReplay) -> None:
    expected_supports = tuple(
        tuple(sorted((*natural_support(), *additions)))
        for additions in SIZE21_ADDITIONS
    )
    if (
        replay.max_total_support != MAX_REPLAY_SUPPORT
        or replay.node_cap != DEFAULT_NODE_CAP
        or not replay.raw_supports_without_symmetry_quotient
        or not replay.exhaustive
        or replay.nodes_examined != EXPECTED_CAP21_SUPPORTS
        or replay.supports_discovered != EXPECTED_CAP21_SUPPORTS
        or sum(replay.discovered_by_addition_size)
        != replay.supports_discovered
        or replay.discovered_by_addition_size
        != EXPECTED_DISCOVERED_BY_ADDITION_SIZE
        or set(replay.singleton_free_supports)
        != set(expected_supports)
    ):
        raise KrennSupportExtensionError(
            "cap-21 support census failed exact replay"
        )


@lru_cache(maxsize=1)
def cached_exhaustive_cap21_replay() -> SupportClosureReplay:
    """Run the exact cap-21 census once per process and cache its receipt."""

    replay = bounded_missing_set_closure(
        max_total_support=MAX_REPLAY_SUPPORT,
        node_cap=DEFAULT_NODE_CAP,
    )
    _validate_cap21_replay(replay)
    return replay


def certify_support_extension(
    *,
    recompute_exhaustive_census: bool = False,
) -> SupportExtensionCertificate:
    """Return the support-extension receipt.

    Set ``recompute_exhaustive_census=True`` for the separately bounded,
    expensive cap-21 traversal.  The default routine path still recomputes
    every local repair, every size-21 support profile, and every odd-cycle
    exponent identity.
    """

    repairs = minimal_victim_repairs()
    valuation_audits = finite_valuation_audits()
    size21 = size21_support_audits()
    replay = None
    if recompute_exhaustive_census:
        replay = cached_exhaustive_cap21_replay()
    if (
        len(repairs) != 6
        or any(audit.finite_regularization_possible for audit in valuation_audits)
        or len(size21) != 6
        or any(
            not audit.singleton_free
            or audit.finite_nonzero_solution_possible
            or audit.mixed_equation_count != 18
            for audit in size21
        )
    ):
        raise KrennSupportExtensionError(
            "support-extension semantic receipt failed"
        )
    certificate = SupportExtensionCertificate(
        minimum_local_additions=2,
        minimal_repairs=repairs,
        finite_valuation_audits=valuation_audits,
        recorded_cap20_supports=EXPECTED_CAP20_SUPPORTS,
        recorded_cap21_supports=EXPECTED_CAP21_SUPPORTS,
        recorded_discovered_by_addition_size=(
            EXPECTED_DISCOVERED_BY_ADDITION_SIZE
        ),
        size21_audits=size21,
        finite_exact_total_support_lower_bound=(22 if replay else None),
        exhaustive_replay=replay,
    )
    certificate._validate()
    return certificate
