"""Bounded fixed-target ``{-1,0,1}`` search for native ``n=6,d=3``.

This is a resumable deterministic tree interface, not an uncontrolled search.
Every report carries explicit node, wall-clock, support, and frontier caps.
Completed assignments are checked in the original 729 cubic equations over
the integers, including constant amplitudes exactly equal to one.  This is
distinct from a GHZ-orbit search, where constant amplitudes need only be
nonzero and normalization may leave the ternary weight domain.  Partial nodes
are pruned only by sound interval and support bounds.

No nonexistence claim is made: this milestone emits no independently
replayable complete-tree certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import time
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    coloring_from_index,
    rhs_for_coloring,
    variable_count,
    variable_index,
)
from experiments.krenn_quantum_graph.witness import (
    SparseWitness,
    evaluate_exact,
)


TERNARY_SEARCH_SCHEMA = "krenn-quantum-graph-ternary-search-v1"
TERNARY_VALUES = (-1, 0, 1)
FIXED_TARGET_MODE = "fixed-target"
MAX_NODE_CAP = 1_000_000
MAX_TIME_CAP_SECONDS = 3_600.0
MAX_FRONTIER_CAP = 1_000_000

N6_D3_SEED_FACTORS = (
    ((0, 1), (2, 3), (4, 5)),
    ((0, 2), (1, 4), (3, 5)),
    ((0, 3), (1, 5), (2, 4)),
)
N6_D3_SEED_DEFECT_EQUATION = 70


class KrennTernarySearchError(ValueError):
    """A ternary plan, checkpoint, node, or replay is invalid."""


def has_canonical_ghz_target(
    system: SparsePolynomialSystem,
) -> bool:
    """Whether the RHS is one exactly on constant colorings."""

    try:
        if len(system.rhs_values) != system.d**system.n:
            return False
        return all(
            int(system.rhs_values[equation])
            == rhs_for_coloring(
                coloring_from_index(system.n, system.d, equation)
            )
            for equation in range(system.d**system.n)
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _require_canonical_ghz_target(
    system: SparsePolynomialSystem,
) -> None:
    if not has_canonical_ghz_target(system):
        raise KrennTernarySearchError(
            "fixed-target search requires the canonical GHZ RHS"
        )


def n6_d3_seed_values() -> tuple[int, ...]:
    """The nine-weight unit seed, which misses exactly one equation."""

    values = [0] * variable_count(6, 3)
    for color, factor in enumerate(N6_D3_SEED_FACTORS):
        for i, j in factor:
            values[variable_index(6, 3, i, j, color, color)] = 1
    return tuple(values)


def n6_d3_seed_witness() -> SparseWitness:
    return SparseWitness.from_index_values(
        6,
        3,
        (
            (index, value)
            for index, value in enumerate(n6_d3_seed_values())
            if value
        ),
    )


@dataclass(frozen=True)
class TernaryCandidate:
    """One dense ternary assignment with exact integer residuals."""

    values: tuple[int, ...]
    residuals: tuple[int, ...]
    support_size: int
    nonzero_residual_count: int
    residual_l1: int

    def __post_init__(self) -> None:
        values = tuple(map(int, self.values))
        residuals = tuple(map(int, self.residuals))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "residuals", residuals)
        if len(values) != variable_count(6, 3) or any(
            value not in TERNARY_VALUES for value in values
        ):
            raise KrennTernarySearchError(
                "candidate is not a canonical n=6,d=3 ternary vector"
            )
        if len(residuals) != 3**6:
            raise KrennTernarySearchError(
                "candidate residual census is wrong"
            )
        if self.support_size != sum(value != 0 for value in values):
            raise KrennTernarySearchError(
                "candidate support count is not content-derived"
            )
        if self.nonzero_residual_count != sum(
            residual != 0 for residual in residuals
        ):
            raise KrennTernarySearchError(
                "candidate defect count is not content-derived"
            )
        if self.residual_l1 != sum(map(abs, residuals)):
            raise KrennTernarySearchError(
                "candidate residual norm is not content-derived"
            )

    @property
    def exact_solution(self) -> bool:
        return self.nonzero_residual_count == 0

    def to_dict(self) -> dict:
        return {
            "values": list(self.values),
            "residuals": list(self.residuals),
            "support_size": self.support_size,
            "nonzero_residual_count": self.nonzero_residual_count,
            "residual_l1": self.residual_l1,
            "exact_solution": self.exact_solution,
        }


def evaluate_ternary_candidate(
    system: SparsePolynomialSystem,
    values: Sequence[int],
) -> TernaryCandidate:
    """Check all original equations exactly over ``Z``."""

    if (system.n, system.d) != (6, 3):
        raise KrennTernarySearchError(
            "the ternary milestone is bounded to n=6,d=3"
        )
    _require_canonical_ghz_target(system)
    normalized = tuple(map(int, values))
    if len(normalized) != system.variable_count or any(
        value not in TERNARY_VALUES for value in normalized
    ):
        raise KrennTernarySearchError(
            "ternary assignment has the wrong domain or dimension"
        )
    witness = SparseWitness.from_index_values(
        6,
        3,
        (
            (index, value)
            for index, value in enumerate(normalized)
            if value
        ),
    )
    evaluation = evaluate_exact(system, witness)
    if any(
        not isinstance(residual, Fraction)
        or residual.denominator != 1
        for residual in evaluation.residuals
    ):
        raise KrennTernarySearchError(
            "integer candidate produced a nonintegral residual"
        )
    residuals = tuple(
        residual.numerator for residual in evaluation.residuals
    )
    return TernaryCandidate(
        values=normalized,
        residuals=residuals,
        support_size=sum(value != 0 for value in normalized),
        nonzero_residual_count=sum(
            residual != 0 for residual in residuals
        ),
        residual_l1=sum(map(abs, residuals)),
    )


def _candidate_key(
    candidate: TernaryCandidate,
) -> tuple[int, int, int, tuple[int, ...]]:
    return (
        candidate.nonzero_residual_count,
        candidate.residual_l1,
        candidate.support_size,
        candidate.values,
    )


def deterministic_variable_order(
    system: SparsePolynomialSystem,
    preferred_values: Sequence[int],
) -> tuple[int, ...]:
    """Lead with zero repair slots in the seed-defect equations."""

    preferred = tuple(map(int, preferred_values))
    seed = evaluate_ternary_candidate(system, preferred)
    defect_equations = tuple(
        equation
        for equation, residual in enumerate(seed.residuals)
        if residual
    )
    defect_variables = {
        variable
        for equation in defect_equations
        for monomial in system.equation_monomials(equation)
        for variable in monomial
    }
    return tuple(
        sorted(
            range(system.variable_count),
            key=lambda variable: (
                variable not in defect_variables,
                preferred[variable] != 0,
                variable,
            ),
        )
    )


@dataclass(frozen=True)
class TernarySearchPlan:
    node_cap: int
    time_cap_seconds: float
    support_cap: int
    frontier_cap: int
    n: int = 6
    d: int = 3
    mode: str = FIXED_TARGET_MODE
    schema: str = TERNARY_SEARCH_SCHEMA

    def __post_init__(self) -> None:
        node_cap = int(self.node_cap)
        time_cap = float(self.time_cap_seconds)
        support_cap = int(self.support_cap)
        frontier_cap = int(self.frontier_cap)
        object.__setattr__(self, "node_cap", node_cap)
        object.__setattr__(self, "time_cap_seconds", time_cap)
        object.__setattr__(self, "support_cap", support_cap)
        object.__setattr__(self, "frontier_cap", frontier_cap)
        if self.schema != TERNARY_SEARCH_SCHEMA:
            raise KrennTernarySearchError(
                "ternary-search schema changed"
            )
        if (int(self.n), int(self.d)) != (6, 3):
            raise KrennTernarySearchError(
                "the ternary-search plan is bounded to n=6,d=3"
            )
        if self.mode != FIXED_TARGET_MODE:
            raise KrennTernarySearchError(
                "this engine implements fixed-target mode only; "
                "GHZ-orbit normalization can leave {-1,0,1}"
            )
        if not 1 <= node_cap <= MAX_NODE_CAP:
            raise KrennTernarySearchError(
                "node cap is outside the safety bound"
            )
        if not 0 < time_cap <= MAX_TIME_CAP_SECONDS:
            raise KrennTernarySearchError(
                "time cap is outside the safety bound"
            )
        if not 9 <= support_cap <= variable_count(6, 3):
            raise KrennTernarySearchError(
                "support cap must contain the nine-weight seed"
            )
        if not 1 <= frontier_cap <= MAX_FRONTIER_CAP:
            raise KrennTernarySearchError(
                "frontier cap is outside the safety bound"
            )

    @property
    def deterministic_branching(self) -> bool:
        return True

    @property
    def symmetry_weight_equalities(self) -> int:
        return 0

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "n": self.n,
            "d": self.d,
            "mode": self.mode,
            "ambient_variables": 135,
            "equations": 729,
            "monomials": 10_935,
            "ambient_assignments": "3^135",
            "node_cap": self.node_cap,
            "time_cap_seconds": self.time_cap_seconds,
            "support_cap": self.support_cap,
            "frontier_cap": self.frontier_cap,
            "deterministic_branching": True,
            "symmetry_weight_equalities": 0,
        }


@dataclass(frozen=True)
class TernaryPartialAssignment:
    """Values for a prefix of the deterministic variable order."""

    prefix_values: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized = tuple(map(int, self.prefix_values))
        object.__setattr__(self, "prefix_values", normalized)
        if len(normalized) > variable_count(6, 3) or any(
            value not in TERNARY_VALUES for value in normalized
        ):
            raise KrennTernarySearchError(
                "partial ternary assignment is malformed"
            )

    @property
    def depth(self) -> int:
        return len(self.prefix_values)

    @property
    def assigned_support_size(self) -> int:
        return sum(value != 0 for value in self.prefix_values)

    def to_dict(self) -> dict:
        return {"prefix_values": list(self.prefix_values)}


@dataclass(frozen=True)
class TernaryCheckpoint:
    """Complete resumable frontier plus the best exact replay so far."""

    variable_order: tuple[int, ...]
    preferred_values: tuple[int, ...]
    frontier: tuple[TernaryPartialAssignment, ...]
    best_candidate: TernaryCandidate
    total_nodes_examined: int
    support_cap: int
    schema: str = TERNARY_SEARCH_SCHEMA

    def __post_init__(self) -> None:
        order = tuple(map(int, self.variable_order))
        preferred = tuple(map(int, self.preferred_values))
        frontier = tuple(self.frontier)
        object.__setattr__(self, "variable_order", order)
        object.__setattr__(self, "preferred_values", preferred)
        object.__setattr__(self, "frontier", frontier)
        if self.schema != TERNARY_SEARCH_SCHEMA:
            raise KrennTernarySearchError(
                "ternary-checkpoint schema changed"
            )
        ambient = variable_count(6, 3)
        if len(order) != ambient or set(order) != set(range(ambient)):
            raise KrennTernarySearchError(
                "checkpoint variable order is not a permutation"
            )
        if len(preferred) != ambient or any(
            value not in TERNARY_VALUES for value in preferred
        ):
            raise KrennTernarySearchError(
                "checkpoint preferred assignment is malformed"
            )
        if len(frontier) != len(set(frontier)):
            raise KrennTernarySearchError(
                "checkpoint frontier must be unique"
            )
        if any(
            node.assigned_support_size > self.support_cap
            for node in frontier
        ):
            raise KrennTernarySearchError(
                "checkpoint node exceeds the support cap"
            )
        if int(self.total_nodes_examined) < 0:
            raise KrennTernarySearchError(
                "checkpoint node census cannot be negative"
            )

    @property
    def tree_certificate_complete(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "variable_order": list(self.variable_order),
            "preferred_values": list(self.preferred_values),
            "frontier": [node.to_dict() for node in self.frontier],
            "best_candidate": self.best_candidate.to_dict(),
            "total_nodes_examined": self.total_nodes_examined,
            "support_cap": self.support_cap,
            "tree_certificate_complete": False,
        }


def checkpoint_from_dict(payload: Mapping) -> TernaryCheckpoint:
    candidate_payload = payload["best_candidate"]
    return TernaryCheckpoint(
        schema=str(payload["schema"]),
        variable_order=tuple(map(int, payload["variable_order"])),
        preferred_values=tuple(
            map(int, payload["preferred_values"])
        ),
        frontier=tuple(
            TernaryPartialAssignment(
                tuple(map(int, row["prefix_values"]))
            )
            for row in payload["frontier"]
        ),
        best_candidate=TernaryCandidate(
            values=tuple(map(int, candidate_payload["values"])),
            residuals=tuple(
                map(int, candidate_payload["residuals"])
            ),
            support_size=int(candidate_payload["support_size"]),
            nonzero_residual_count=int(
                candidate_payload["nonzero_residual_count"]
            ),
            residual_l1=int(candidate_payload["residual_l1"]),
        ),
        total_nodes_examined=int(payload["total_nodes_examined"]),
        support_cap=int(payload["support_cap"]),
    )


def _complete_values(
    node: TernaryPartialAssignment,
    order: Sequence[int],
    preferred: Sequence[int],
) -> tuple[int, ...]:
    values = list(preferred)
    for position, value in enumerate(node.prefix_values):
        values[order[position]] = value
    return tuple(values)


def _assigned_values(
    node: TernaryPartialAssignment, order: Sequence[int]
) -> dict[int, int]:
    return {
        order[position]: value
        for position, value in enumerate(node.prefix_values)
    }


def partial_interval_feasible(
    system: SparsePolynomialSystem,
    node: TernaryPartialAssignment,
    variable_order: Sequence[int],
    support_cap: int,
) -> bool:
    """Soundly reject a partial node whose target leaves every interval."""

    if node.assigned_support_size > support_cap:
        return False
    assigned = _assigned_values(node, variable_order)
    for equation in range(system.equation_count):
        fixed_sum = 0
        unresolved = 0
        for monomial in system.equation_monomials(equation):
            term = 1
            complete = True
            for variable in monomial:
                if variable not in assigned:
                    complete = False
                    continue
                term *= assigned[variable]
                if term == 0:
                    break
            if term == 0:
                continue
            if complete:
                fixed_sum += term
            else:
                # Any incomplete ternary monomial lies in {-1,0,1}.
                unresolved += 1
        target = system.rhs_values[equation]
        if not fixed_sum - unresolved <= target <= (
            fixed_sum + unresolved
        ):
            return False
    return True


def _branch_values(preferred: int) -> tuple[int, ...]:
    return (preferred, *(
        value for value in TERNARY_VALUES if value != preferred
    ))


@dataclass(frozen=True)
class TernarySearchResult:
    plan: TernarySearchPlan
    checkpoint: TernaryCheckpoint
    termination: str
    nodes_examined_this_run: int
    exact_original_equations_checked: bool = True
    tree_certificate_complete: bool = False
    exhaustive_nonexistence: bool = False
    nonexistence_proved: bool = False
    symmetry_weight_equalities: int = 0

    def __post_init__(self) -> None:
        if self.termination not in {
            "exact-solution-found",
            "node-cap-reached",
            "time-cap-reached",
            "frontier-cap-reached",
            "frontier-exhausted-without-tree-certificate",
        }:
            raise KrennTernarySearchError(
                "ternary-search termination changed"
            )
        if not 0 <= self.nodes_examined_this_run <= (
            self.plan.node_cap
        ):
            raise KrennTernarySearchError(
                "per-run node census exceeds its cap"
            )
        if (
            self.exact_original_equations_checked is not True
            or self.tree_certificate_complete is not False
            or self.exhaustive_nonexistence is not False
            or self.nonexistence_proved is not False
            or self.symmetry_weight_equalities != 0
        ):
            raise KrennTernarySearchError(
                "ternary-search claim boundary changed"
            )
        if (
            self.termination == "exact-solution-found"
        ) != self.checkpoint.best_candidate.exact_solution:
            raise KrennTernarySearchError(
                "solution status is not content-derived"
            )

    @property
    def solution_certified(self) -> bool:
        return self.checkpoint.best_candidate.exact_solution

    @property
    def search_mode(self) -> str:
        return self.plan.mode

    @property
    def proof_over_C(self) -> bool:
        # An exact integer solution embeds in C.  No numerical inference occurs.
        return self.solution_certified

    @property
    def resumable(self) -> bool:
        return (
            self.termination != "exact-solution-found"
            and bool(self.checkpoint.frontier)
        )

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "termination": self.termination,
            "search_mode": self.search_mode,
            "nodes_examined_this_run": self.nodes_examined_this_run,
            "exact_original_equations_checked": True,
            "solution_certified": self.solution_certified,
            "proof_over_C": self.proof_over_C,
            "tree_certificate_complete": False,
            "exhaustive_nonexistence": False,
            "nonexistence_proved": False,
            "symmetry_weight_equalities": 0,
            "resumable": self.resumable,
        }


def _new_checkpoint(
    system: SparsePolynomialSystem,
    plan: TernarySearchPlan,
) -> TernaryCheckpoint:
    _require_canonical_ghz_target(system)
    preferred = n6_d3_seed_values()
    return TernaryCheckpoint(
        variable_order=deterministic_variable_order(
            system, preferred
        ),
        preferred_values=preferred,
        frontier=(TernaryPartialAssignment(()),),
        best_candidate=evaluate_ternary_candidate(
            system, preferred
        ),
        total_nodes_examined=0,
        support_cap=plan.support_cap,
    )


def replay_checkpoint(
    system: SparsePolynomialSystem,
    plan: TernarySearchPlan,
    checkpoint: TernaryCheckpoint,
) -> TernaryCheckpoint:
    """Validate all resumable data and exactly replay the best candidate."""

    if (system.n, system.d) != (6, 3):
        raise KrennTernarySearchError(
            "checkpoint replay has the wrong system"
        )
    _require_canonical_ghz_target(system)
    expected_preferred = n6_d3_seed_values()
    expected_order = deterministic_variable_order(
        system, expected_preferred
    )
    if (
        checkpoint.schema != TERNARY_SEARCH_SCHEMA
        or checkpoint.support_cap != plan.support_cap
        or checkpoint.preferred_values != expected_preferred
        or checkpoint.variable_order != expected_order
        or len(checkpoint.frontier) > plan.frontier_cap
    ):
        raise KrennTernarySearchError(
            "checkpoint does not match the deterministic search problem"
        )
    replayed = evaluate_ternary_candidate(
        system, checkpoint.best_candidate.values
    )
    if replayed != checkpoint.best_candidate:
        raise KrennTernarySearchError(
            "checkpoint best candidate failed exact replay"
        )
    return checkpoint


def run_bounded_ternary_search(
    system: SparsePolynomialSystem,
    plan: TernarySearchPlan,
    *,
    checkpoint: TernaryCheckpoint | None = None,
) -> TernarySearchResult:
    """Run or resume one explicitly bounded deterministic tree segment."""

    if (system.n, system.d) != (6, 3):
        raise KrennTernarySearchError(
            "the ternary milestone requires the n=6,d=3 system"
        )
    _require_canonical_ghz_target(system)
    state = (
        _new_checkpoint(system, plan)
        if checkpoint is None
        else replay_checkpoint(system, plan, checkpoint)
    )
    frontier = list(state.frontier)
    best = state.best_candidate
    nodes_this_run = 0
    started = time.monotonic()
    termination = "node-cap-reached"

    while frontier and nodes_this_run < plan.node_cap:
        if time.monotonic() - started >= plan.time_cap_seconds:
            termination = "time-cap-reached"
            break
        node = frontier.pop()
        nodes_this_run += 1
        completed = _complete_values(
            node, state.variable_order, state.preferred_values
        )
        if sum(value != 0 for value in completed) <= plan.support_cap:
            candidate = evaluate_ternary_candidate(system, completed)
            if _candidate_key(candidate) < _candidate_key(best):
                best = candidate
            if candidate.exact_solution:
                termination = "exact-solution-found"
                break
        if not partial_interval_feasible(
            system,
            node,
            state.variable_order,
            plan.support_cap,
        ) or node.depth == system.variable_count:
            continue

        variable = state.variable_order[node.depth]
        children = tuple(
            TernaryPartialAssignment(
                (*node.prefix_values, value)
            )
            for value in _branch_values(
                state.preferred_values[variable]
            )
            if (
                node.assigned_support_size
                + int(value != 0)
                <= plan.support_cap
            )
        )
        if len(frontier) + len(children) > plan.frontier_cap:
            frontier.append(node)
            termination = "frontier-cap-reached"
            break
        # Stack order makes the preferred value the next node visited.
        frontier.extend(reversed(children))

    else:
        if not frontier:
            termination = (
                "frontier-exhausted-without-tree-certificate"
            )
        elif nodes_this_run >= plan.node_cap:
            termination = "node-cap-reached"

    updated = TernaryCheckpoint(
        variable_order=state.variable_order,
        preferred_values=state.preferred_values,
        frontier=tuple(frontier),
        best_candidate=best,
        total_nodes_examined=(
            state.total_nodes_examined + nodes_this_run
        ),
        support_cap=state.support_cap,
    )
    return TernarySearchResult(
        plan=plan,
        checkpoint=updated,
        termination=termination,
        nodes_examined_this_run=nodes_this_run,
    )
