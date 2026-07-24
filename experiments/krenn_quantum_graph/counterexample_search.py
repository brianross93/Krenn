"""Bounded support generation for the ``n=6,d=3`` counterexample campaign.

This module searches *support templates*, not weights.  It combines:

* exact perfect-matching incidence;
* singleton closure for zero-target equations;
* the eight diagonal three-matching seed orbits;
* roots omitting one or two coordinates of the natural seed; and
* exact ``S_6 x S_3`` orbit representatives for emitted supports.

Symmetry is used only to deduplicate support templates.  Coordinates in one
orbit are never assigned equal weights.  Passing singleton closure is only a
necessary condition for a finite solution, and every result records that
claim boundary explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from experiments.krenn_quantum_graph.support_extension import (
    natural_support,
    size21_support_audits,
)
from experiments.krenn_quantum_graph.support_search import (
    SupportIndex,
    build_support_index,
    support_profile,
)
from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    generate_sparse_system,
    perfect_matchings,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    EXPECTED_REPRESENTATIVES_AND_SIZES,
)
from experiments.krenn_quantum_graph.transport import (
    transport_variable_index,
)


N = 6
D = 3
AMBIENT_VARIABLES = 135
MIN_SEARCH_SUPPORT = 22
MAX_WORKERS = 16
MAX_DISCOVERED_SUPPORTS = 5_000_000
COUNTEREXAMPLE_SUPPORT_SCHEMA = (
    "krenn-n6-d3-counterexample-support-campaign-v1"
)


class KrennCounterexampleSearchError(ValueError):
    """A bounded support campaign or its replay is malformed."""


def _canonical_support(values: Iterable[int]) -> tuple[int, ...]:
    support = tuple(map(int, values))
    if len(support) != len(set(support)):
        raise KrennCounterexampleSearchError(
            "support coordinates must be unique"
        )
    support = tuple(sorted(support))
    if any(index < 0 or index >= AMBIENT_VARIABLES for index in support):
        raise KrennCounterexampleSearchError(
            "support coordinate is outside the n=6,d=3 system"
        )
    return support


def _mask(values: Iterable[int]) -> int:
    result = 0
    for index in values:
        result |= 1 << int(index)
    return result


def _indices(mask: int) -> tuple[int, ...]:
    result: list[int] = []
    while mask:
        bit = mask & -mask
        result.append(bit.bit_length() - 1)
        mask -= bit
    return tuple(result)


@lru_cache(maxsize=1)
def n6_d3_system() -> SparsePolynomialSystem:
    """Return the canonical fixed-target system used by the campaign."""

    return generate_sparse_system(N, D)


@lru_cache(maxsize=1)
def n6_d3_support_index() -> SupportIndex:
    """Return the exact 135-bit monomial incidence index."""

    return build_support_index(n6_d3_system())


@lru_cache(maxsize=1)
def symmetry_action_table() -> np.ndarray:
    """Return all 4,320 old-to-new variable permutations.

    The table is a read-only ``uint8`` array with shape ``(4320,135)``.
    Endpoint reversal is delegated to the canonical transport implementation,
    which also swaps the two color slots.
    """

    rows = np.empty((720 * 6, AMBIENT_VARIABLES), dtype=np.uint8)
    row = 0
    for vertex_permutation in permutations(range(N)):
        for color_permutation in permutations(range(D)):
            rows[row] = tuple(
                transport_variable_index(
                    N,
                    D,
                    index,
                    vertex_permutation,
                    color_permutation,
                )
                for index in range(AMBIENT_VARIABLES)
            )
            row += 1
    if row != 4_320 or any(
        len(set(map(int, candidate))) != AMBIENT_VARIABLES
        for candidate in rows
    ):
        raise KrennCounterexampleSearchError(
            "S6 x S3 variable action failed permutation replay"
        )
    rows.setflags(write=False)
    return rows


def transport_support(
    support: Iterable[int],
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> tuple[int, ...]:
    """Transport a support without transporting or identifying its weights."""

    canonical = _canonical_support(support)
    return tuple(
        sorted(
            transport_variable_index(
                N,
                D,
                index,
                vertex_permutation,
                color_permutation,
            )
            for index in canonical
        )
    )


def canonical_support_representative(
    support: Iterable[int],
) -> tuple[int, ...]:
    """Return the lexicographically least ``S_6 x S_3`` support image."""

    canonical = _canonical_support(support)
    if not canonical:
        return ()
    images = np.sort(
        symmetry_action_table()[:, np.asarray(canonical, dtype=np.intp)],
        axis=1,
    )
    # Python's tuple comparison is intentionally used here: it is simple,
    # deterministic across NumPy versions, and this is called only for
    # emitted candidates rather than every closure node.
    return min(tuple(map(int, row)) for row in images)


def support_orbit_size(support: Iterable[int]) -> int:
    """Return the exact number of distinct transported supports."""

    canonical = _canonical_support(support)
    if not canonical:
        return 1
    images = np.sort(
        symmetry_action_table()[:, np.asarray(canonical, dtype=np.intp)],
        axis=1,
    )
    return int(np.unique(images, axis=0).shape[0])


def diagonal_seed_support(
    matching_indices: Sequence[int],
) -> tuple[int, ...]:
    """Lift an ordered three-matching seed to nine diagonal coordinates."""

    matching_indices = tuple(map(int, matching_indices))
    matchings = perfect_matchings(N)
    if len(matching_indices) != D or any(
        index < 0 or index >= len(matchings)
        for index in matching_indices
    ):
        raise KrennCounterexampleSearchError(
            "a diagonal seed needs three valid matching indices"
        )
    return tuple(
        sorted(
            variable_index(N, D, i, j, color, color)
            for color, matching_index in enumerate(matching_indices)
            for i, j in matchings[matching_index]
        )
    )


@dataclass(frozen=True)
class SupportSeed:
    """One deterministic root stratum for singleton closure."""

    name: str
    support: tuple[int, ...]
    forbidden_coordinates: tuple[int, ...] = ()
    family: str = "unspecified"

    def __post_init__(self) -> None:
        support = _canonical_support(self.support)
        forbidden = _canonical_support(self.forbidden_coordinates)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "forbidden_coordinates", forbidden)
        if not self.name or not self.family:
            raise KrennCounterexampleSearchError(
                "support seeds need nonempty names and families"
            )
        if set(support).intersection(forbidden):
            raise KrennCounterexampleSearchError(
                "a seed contains a forbidden coordinate"
            )

    @property
    def omits_natural_coordinates(self) -> tuple[int, ...]:
        natural = set(natural_support())
        return tuple(sorted(natural.difference(self.support)))


def default_support_seeds(
    *,
    include_two_coordinate_omissions: bool = True,
) -> tuple[SupportSeed, ...]:
    """Build the bounded campaign's deterministic support roots."""

    natural = natural_support()
    rows: list[SupportSeed] = []

    for index, audit in enumerate(size21_support_audits()):
        rows.append(
            SupportSeed(
                name=f"natural-size21-{index}",
                family="retains-natural-size21",
                support=audit.support,
            )
        )

    for orbit_index, (representative, _size) in enumerate(
        EXPECTED_REPRESENTATIVES_AND_SIZES
    ):
        rows.append(
            SupportSeed(
                name=f"diagonal-seed-orbit-{orbit_index}",
                family="diagonal-seed-orbit",
                support=diagonal_seed_support(representative),
            )
        )

    for position, omitted in enumerate(natural):
        rows.append(
            SupportSeed(
                name=f"omit-natural-one-{position}",
                family="omits-one-natural",
                support=tuple(
                    index for index in natural if index != omitted
                ),
                forbidden_coordinates=(omitted,),
            )
        )

    if include_two_coordinate_omissions:
        for left, right in combinations(range(len(natural)), 2):
            omitted = (natural[left], natural[right])
            rows.append(
                SupportSeed(
                    name=f"omit-natural-two-{left}-{right}",
                    family="omits-two-natural",
                    support=tuple(
                        index
                        for index in natural
                        if index not in omitted
                    ),
                    forbidden_coordinates=omitted,
                )
            )
    return tuple(rows)


def greedy_pruned_omission_seed(
    forbidden_coordinates: Iterable[int],
    *,
    deterministic_seed: int,
    minimum_support: int = MIN_SEARCH_SUPPORT,
    name: str = "greedy-pruned-omission",
) -> SupportSeed:
    """Delete coordinates from the dense support while closure stays valid.

    This supplies bounded omission strata even when forward singleton closure
    from a nine-slot root has an enormous frontier.  It is a deterministic
    incidence heuristic, not a minimality or exhaustion certificate.
    """

    forbidden = _canonical_support(forbidden_coordinates)
    if not forbidden:
        raise KrennCounterexampleSearchError(
            "a pruned omission seed must forbid at least one coordinate"
        )
    minimum_support = int(minimum_support)
    if not MIN_SEARCH_SUPPORT <= minimum_support <= AMBIENT_VARIABLES:
        raise KrennCounterexampleSearchError(
            "pruned support floor is outside the searched range"
        )
    support = set(range(AMBIENT_VARIABLES)).difference(forbidden)
    if len(support) < minimum_support:
        raise KrennCounterexampleSearchError(
            "forbidden coordinates leave too small a support"
        )
    rng = np.random.Generator(np.random.PCG64(int(deterministic_seed)))
    order = tuple(
        map(int, rng.permutation(tuple(sorted(support))))
    )
    system = n6_d3_system()
    index = n6_d3_support_index()
    changed = True
    while changed and len(support) > minimum_support:
        changed = False
        for variable in order:
            if variable not in support or len(support) <= minimum_support:
                continue
            child = tuple(sorted(support.difference((variable,))))
            profile = support_profile(system, child, index=index)
            if profile.singleton_free_necessary_condition:
                support.remove(variable)
                changed = True
    result = SupportSeed(
        name=name,
        family="greedy-pruned-natural-omission",
        support=tuple(sorted(support)),
        forbidden_coordinates=forbidden,
    )
    profile = support_profile(system, result.support, index=index)
    if not profile.singleton_free_necessary_condition:
        raise KrennCounterexampleSearchError(
            "greedy pruned support failed final closure replay"
        )
    return result


@dataclass(frozen=True)
class SupportCampaignPlan:
    """Explicit safety and breadth bounds for support generation."""

    node_cap: int
    support_cap: int
    candidate_cap: int
    omission_candidate_quota: int
    activation_width: int
    worker_count: int
    deterministic_seed: int
    scratch_directory: str
    per_support_size_candidate_quota: int | None = None
    discovered_support_cap: int = 2_000_000
    include_two_coordinate_omissions: bool = True
    schema: str = COUNTEREXAMPLE_SUPPORT_SCHEMA

    def __post_init__(self) -> None:
        node_cap = int(self.node_cap)
        support_cap = int(self.support_cap)
        candidate_cap = int(self.candidate_cap)
        omission_candidate_quota = int(self.omission_candidate_quota)
        activation_width = int(self.activation_width)
        worker_count = int(self.worker_count)
        deterministic_seed = int(self.deterministic_seed)
        per_size_quota = (
            candidate_cap
            if self.per_support_size_candidate_quota is None
            else int(self.per_support_size_candidate_quota)
        )
        discovered_support_cap = int(self.discovered_support_cap)
        scratch = str(self.scratch_directory)
        object.__setattr__(self, "node_cap", node_cap)
        object.__setattr__(self, "support_cap", support_cap)
        object.__setattr__(self, "candidate_cap", candidate_cap)
        object.__setattr__(
            self,
            "omission_candidate_quota",
            omission_candidate_quota,
        )
        object.__setattr__(self, "activation_width", activation_width)
        object.__setattr__(self, "worker_count", worker_count)
        object.__setattr__(self, "deterministic_seed", deterministic_seed)
        object.__setattr__(
            self,
            "per_support_size_candidate_quota",
            per_size_quota,
        )
        object.__setattr__(
            self,
            "discovered_support_cap",
            discovered_support_cap,
        )
        object.__setattr__(self, "scratch_directory", scratch)
        if self.schema != COUNTEREXAMPLE_SUPPORT_SCHEMA:
            raise KrennCounterexampleSearchError(
                "support campaign schema changed"
            )
        if not 1 <= node_cap <= 1_000_000:
            raise KrennCounterexampleSearchError(
                "node cap is outside the deterministic safety bound"
            )
        if not MIN_SEARCH_SUPPORT <= support_cap <= AMBIENT_VARIABLES:
            raise KrennCounterexampleSearchError(
                "support cap must search size 22 or above"
            )
        if not 1 <= candidate_cap <= 10_000:
            raise KrennCounterexampleSearchError(
                "candidate cap is outside the safety bound"
            )
        if not 1 <= per_size_quota <= candidate_cap:
            raise KrennCounterexampleSearchError(
                "per-support-size candidate quota is outside the total cap"
            )
        if not 0 <= omission_candidate_quota <= candidate_cap:
            raise KrennCounterexampleSearchError(
                "natural-omission quota is outside the candidate cap"
            )
        if not 1 <= activation_width <= AMBIENT_VARIABLES:
            raise KrennCounterexampleSearchError(
                "activation width is outside the ambient range"
            )
        if not 1 <= worker_count <= MAX_WORKERS:
            raise KrennCounterexampleSearchError(
                "worker count must be between 1 and 16"
            )
        if not 0 <= deterministic_seed <= (1 << 63) - 1:
            raise KrennCounterexampleSearchError(
                "deterministic seed is outside the uint63 range"
            )
        if not 1 <= discovered_support_cap <= MAX_DISCOVERED_SUPPORTS:
            raise KrennCounterexampleSearchError(
                "discovered-support cap is outside the safety bound"
            )
        parts = {part.casefold() for part in Path(scratch).parts}
        if "onedrive" in parts or "onedrive" in scratch.casefold():
            raise KrennCounterexampleSearchError(
                "counterexample scratch data must not use OneDrive"
            )
        if not scratch:
            raise KrennCounterexampleSearchError(
                "a scratch directory is required for checkpoints"
            )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "node_cap": self.node_cap,
            "support_cap": self.support_cap,
            "minimum_searched_support": MIN_SEARCH_SUPPORT,
            "candidate_cap": self.candidate_cap,
            "per_support_size_candidate_quota": (
                self.per_support_size_candidate_quota
            ),
            "discovered_support_cap": self.discovered_support_cap,
            "omission_candidate_quota": self.omission_candidate_quota,
            "activation_width": self.activation_width,
            "worker_count": self.worker_count,
            "deterministic_seed": self.deterministic_seed,
            "deterministic_seed_role": (
                "cyclic-tie-break-for-activation-ranking"
            ),
            "scratch_directory": self.scratch_directory,
            "include_two_coordinate_omissions": (
                self.include_two_coordinate_omissions
            ),
            "symmetry_weight_equalities": 0,
        }


@dataclass(frozen=True)
class SupportCandidate:
    """A singleton-free support template and its exact incidence receipt."""

    seed_name: str
    seed_family: str
    support: tuple[int, ...]
    forbidden_coordinates: tuple[int, ...]
    omitted_natural_coordinates: tuple[int, ...]
    orbit_representative: tuple[int, ...]
    orbit_size: int
    active_equation_count: int
    feasible_matching_counts: tuple[int, ...]
    closure_depth: int
    symmetry_weight_equalities: int = 0

    def __post_init__(self) -> None:
        support = _canonical_support(self.support)
        forbidden = _canonical_support(self.forbidden_coordinates)
        representative = _canonical_support(self.orbit_representative)
        omitted = _canonical_support(self.omitted_natural_coordinates)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "forbidden_coordinates", forbidden)
        object.__setattr__(self, "orbit_representative", representative)
        object.__setattr__(self, "omitted_natural_coordinates", omitted)
        if len(support) < MIN_SEARCH_SUPPORT:
            raise KrennCounterexampleSearchError(
                "a numerical support candidate must have size at least 22"
            )
        if set(support).intersection(forbidden):
            raise KrennCounterexampleSearchError(
                "candidate reactivated a forbidden coordinate"
            )
        if representative != canonical_support_representative(support):
            raise KrennCounterexampleSearchError(
                "candidate orbit representative failed replay"
            )
        if int(self.orbit_size) != support_orbit_size(support):
            raise KrennCounterexampleSearchError(
                "candidate orbit size failed replay"
            )
        profile = support_profile(
            n6_d3_system(),
            support,
            index=n6_d3_support_index(),
        )
        if not profile.singleton_free_necessary_condition:
            raise KrennCounterexampleSearchError(
                "candidate failed singleton-closure replay"
            )
        if (
            tuple(map(int, self.feasible_matching_counts))
            != profile.feasible_matching_counts
            or int(self.active_equation_count)
            != profile.active_equation_count
        ):
            raise KrennCounterexampleSearchError(
                "candidate incidence counts failed exact replay"
            )
        expected_omitted = tuple(
            sorted(set(natural_support()).difference(support))
        )
        if omitted != expected_omitted:
            raise KrennCounterexampleSearchError(
                "candidate natural-omission receipt changed"
            )
        if self.symmetry_weight_equalities != 0:
            raise KrennCounterexampleSearchError(
                "support symmetry must never identify numerical weights"
            )

    @property
    def support_size(self) -> int:
        return len(self.support)

    def to_dict(self) -> dict:
        return {
            "seed_name": self.seed_name,
            "seed_family": self.seed_family,
            "support": list(self.support),
            "support_coordinates": [
                list(variable_key(N, D, index)) for index in self.support
            ],
            "support_size": self.support_size,
            "forbidden_coordinates": list(self.forbidden_coordinates),
            "omitted_natural_coordinates": list(
                self.omitted_natural_coordinates
            ),
            "orbit_representative": list(self.orbit_representative),
            "orbit_size": self.orbit_size,
            "active_equation_count": self.active_equation_count,
            "feasible_matching_counts": list(
                self.feasible_matching_counts
            ),
            "closure_depth": self.closure_depth,
            "singleton_free_necessary_condition": True,
            "symmetry_weight_equalities": 0,
            "solution_certified": False,
        }


@dataclass
class _RootState:
    seed: SupportSeed
    frontier: list[tuple[int, int, int]]
    seen: set[int]


def _equation_masks(
    index: SupportIndex,
    equation: int,
) -> tuple[int, ...]:
    start = index.equation_offsets[equation]
    stop = index.equation_offsets[equation + 1]
    return index.monomial_masks[start:stop]


def _first_closure_branches(
    support_mask: int,
    forbidden_mask: int,
    index: SupportIndex,
) -> tuple[int, tuple[int, ...]] | None:
    """Return the first failed equation and admissible matching completions."""

    system = n6_d3_system()
    for equation, rhs in enumerate(system.rhs_values):
        monomials = _equation_masks(index, equation)
        active = tuple(
            matching
            for matching, monomial in enumerate(monomials)
            if monomial & ~support_mask == 0
        )
        if rhs and not active:
            branches = tuple(
                monomial
                for monomial in monomials
                if monomial & forbidden_mask == 0
            )
            return equation, branches
        if not rhs and len(active) == 1:
            branches = tuple(
                monomial
                for matching, monomial in enumerate(monomials)
                if matching != active[0] and monomial & forbidden_mask == 0
            )
            return equation, branches
    return None


def _activation_coordinates(
    support_mask: int,
    forbidden_mask: int,
    index: SupportIndex,
    width: int,
    deterministic_seed: int,
) -> tuple[int, ...]:
    """Rank deliberate activations by near-complete matching incidence."""

    scores = [0] * AMBIENT_VARIABLES
    for monomial in index.monomial_masks:
        missing = monomial & ~support_mask
        missing_count = missing.bit_count()
        if not 1 <= missing_count <= 3:
            continue
        weight = 4 - missing_count
        for variable in _indices(missing):
            if not (forbidden_mask >> variable) & 1:
                scores[variable] += weight
    allowed = tuple(
        variable
        for variable in range(AMBIENT_VARIABLES)
        if not (support_mask >> variable) & 1
        and not (forbidden_mask >> variable) & 1
    )
    offset = int(deterministic_seed) % AMBIENT_VARIABLES
    return tuple(
        sorted(
            allowed,
            key=lambda variable: (
                -scores[variable],
                (variable - offset) % AMBIENT_VARIABLES,
                variable,
            ),
        )[:width]
    )


def _candidate_from_state(
    state: _RootState,
    support_mask: int,
    depth: int,
) -> SupportCandidate:
    support = _indices(support_mask)
    profile = support_profile(
        n6_d3_system(),
        support,
        index=n6_d3_support_index(),
    )
    return SupportCandidate(
        seed_name=state.seed.name,
        seed_family=state.seed.family,
        support=support,
        forbidden_coordinates=state.seed.forbidden_coordinates,
        omitted_natural_coordinates=tuple(
            sorted(set(natural_support()).difference(support))
        ),
        orbit_representative=canonical_support_representative(support),
        orbit_size=support_orbit_size(support),
        active_equation_count=profile.active_equation_count,
        feasible_matching_counts=profile.feasible_matching_counts,
        closure_depth=depth,
    )


@dataclass(frozen=True)
class SupportCampaignResult:
    """A deterministic bounded support search with fail-closed claims."""

    plan: SupportCampaignPlan
    nodes_examined: int
    supports_discovered: int
    frontier_remaining: int
    candidates: tuple[SupportCandidate, ...]
    seed_count: int
    termination: str
    schema: str = COUNTEREXAMPLE_SUPPORT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SupportCampaignPlan):
            raise KrennCounterexampleSearchError(
                "support result requires a canonical plan"
            )
        integer_fields = (
            "nodes_examined",
            "supports_discovered",
            "frontier_remaining",
            "seed_count",
        )
        for name in integer_fields:
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.schema != COUNTEREXAMPLE_SUPPORT_SCHEMA:
            raise KrennCounterexampleSearchError(
                "support result schema changed"
            )
        if self.termination not in {
            "candidate-cap-reached",
            "node-cap-reached",
            "discovered-support-cap-reached",
            "frontier-exhausted-under-support-cap",
        }:
            raise KrennCounterexampleSearchError(
                "support result termination changed"
            )
        representatives = tuple(
            candidate.orbit_representative for candidate in self.candidates
        )
        if len(representatives) != len(set(representatives)):
            raise KrennCounterexampleSearchError(
                "emitted supports are not orbit-deduplicated"
            )
        if len(self.candidates) > self.plan.candidate_cap:
            raise KrennCounterexampleSearchError(
                "support result exceeded its candidate cap"
            )
        if (
            self.seed_count < 1
            or self.nodes_examined < 0
            or self.nodes_examined > self.plan.node_cap
            or self.frontier_remaining < 0
            or self.supports_discovered < self.seed_count
            or self.supports_discovered
            > self.plan.discovered_support_cap
            or self.supports_discovered
            != self.nodes_examined + self.frontier_remaining
        ):
            raise KrennCounterexampleSearchError(
                "support result accounting is inconsistent"
            )
        if (
            self.termination == "candidate-cap-reached"
            and len(self.candidates) != self.plan.candidate_cap
        ):
            raise KrennCounterexampleSearchError(
                "candidate-cap termination has the wrong candidate count"
            )
        if (
            self.termination == "node-cap-reached"
            and (
                self.nodes_examined != self.plan.node_cap
                or not self.frontier_remaining
            )
        ):
            raise KrennCounterexampleSearchError(
                "node-cap termination has inconsistent accounting"
            )
        if (
            self.termination == "discovered-support-cap-reached"
            and (
                self.supports_discovered
                != self.plan.discovered_support_cap
                or not self.frontier_remaining
            )
        ):
            raise KrennCounterexampleSearchError(
                "discovered-support-cap termination is inconsistent"
            )
        if (
            self.termination == "frontier-exhausted-under-support-cap"
            and self.frontier_remaining != 0
        ):
            raise KrennCounterexampleSearchError(
                "frontier exhaustion was claimed with queued supports"
            )
        size_counts: dict[int, int] = {}
        for candidate in self.candidates:
            size_counts[candidate.support_size] = (
                size_counts.get(candidate.support_size, 0) + 1
            )
        if any(
            count > self.plan.per_support_size_candidate_quota
            for count in size_counts.values()
        ):
            raise KrennCounterexampleSearchError(
                "support result exceeded its per-size candidate quota"
            )

    @property
    def includes_natural_omission(self) -> bool:
        return any(
            candidate.omitted_natural_coordinates
            for candidate in self.candidates
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "parameters": {
                "n": N,
                "d": D,
                "ambient_variables": AMBIENT_VARIABLES,
                "equations": 729,
                "perfect_matchings_per_equation": 15,
            },
            "plan": self.plan.to_dict(),
            "seed_count": self.seed_count,
            "nodes_examined": self.nodes_examined,
            "supports_discovered": self.supports_discovered,
            "frontier_remaining": self.frontier_remaining,
            "termination": self.termination,
            "candidate_count": len(self.candidates),
            "includes_support_omitting_natural_coordinates": (
                self.includes_natural_omission
            ),
            "candidates": [
                candidate.to_dict() for candidate in self.candidates
            ],
            "claim_boundary": {
                "bounded_deterministic_search_only": True,
                "singleton_closure_is_necessary_only": True,
                "symmetry_used_only_for_support_orbit_representatives": True,
                "symmetry_weight_equalities": 0,
                "bounded_miss_proves_nonexistence": False,
                "numerical_solution_certified": False,
                "exact_counterexample_certified": False,
            },
        }


def run_support_campaign(
    plan: SupportCampaignPlan,
    *,
    seeds: Sequence[SupportSeed] | None = None,
) -> SupportCampaignResult:
    """Run round-robin singleton closure under explicit deterministic caps."""

    if not isinstance(plan, SupportCampaignPlan):
        raise KrennCounterexampleSearchError(
            "support campaign requires a canonical plan"
        )
    seed_rows = (
        default_support_seeds(
            include_two_coordinate_omissions=(
                plan.include_two_coordinate_omissions
            )
        )
        if seeds is None
        else tuple(seeds)
    )
    if not seed_rows or any(
        not isinstance(seed, SupportSeed) for seed in seed_rows
    ):
        raise KrennCounterexampleSearchError(
            "support campaign needs canonical support seeds"
        )

    states: list[_RootState] = []
    for seed in seed_rows:
        if len(seed.support) > plan.support_cap:
            raise KrennCounterexampleSearchError(
                f"seed {seed.name!r} exceeds the support cap"
            )
        mask = _mask(seed.support)
        states.append(
            _RootState(
                seed=seed,
                frontier=[(len(seed.support), 0, mask)],
                seen={mask},
            )
        )
    if len(states) > plan.discovered_support_cap:
        raise KrennCounterexampleSearchError(
            "support seeds exceed the discovered-support cap"
        )

    index = n6_d3_support_index()
    nodes = 0
    supports_discovered = len(states)
    candidate_by_orbit: dict[tuple[int, ...], SupportCandidate] = {}
    cursor = 0
    while (
        nodes < plan.node_cap
        and supports_discovered < plan.discovered_support_cap
        and len(candidate_by_orbit) < plan.candidate_cap
        and any(state.frontier for state in states)
    ):
        state = states[cursor % len(states)]
        cursor += 1
        if not state.frontier:
            continue
        _size, depth, support_mask = heapq.heappop(state.frontier)
        nodes += 1
        forbidden_mask = _mask(state.seed.forbidden_coordinates)
        obstruction = _first_closure_branches(
            support_mask, forbidden_mask, index
        )
        if obstruction is None:
            if support_mask.bit_count() >= MIN_SEARCH_SUPPORT:
                candidate = _candidate_from_state(
                    state, support_mask, depth
                )
                omission_count = sum(
                    bool(row.omitted_natural_coordinates)
                    for row in candidate_by_orbit.values()
                )
                retaining_count = (
                    len(candidate_by_orbit) - omission_count
                )
                retaining_cap = (
                    plan.candidate_cap
                    - plan.omission_candidate_quota
                )
                size_count = sum(
                    row.support_size == candidate.support_size
                    for row in candidate_by_orbit.values()
                )
                if (
                    (
                        candidate.omitted_natural_coordinates
                        or retaining_count < retaining_cap
                    )
                    and size_count
                    < plan.per_support_size_candidate_quota
                ):
                    candidate_by_orbit.setdefault(
                        candidate.orbit_representative, candidate
                    )
            if support_mask.bit_count() < plan.support_cap:
                for variable in _activation_coordinates(
                    support_mask,
                    forbidden_mask,
                    index,
                    plan.activation_width,
                    plan.deterministic_seed,
                ):
                    child = support_mask | (1 << variable)
                    if child in state.seen:
                        continue
                    state.seen.add(child)
                    supports_discovered += 1
                    heapq.heappush(
                        state.frontier,
                        (child.bit_count(), depth + 1, child),
                    )
                    if (
                        supports_discovered
                        >= plan.discovered_support_cap
                    ):
                        break
            continue

        _equation, branch_monomials = obstruction
        children: set[tuple[int, tuple[int, ...], int]] = set()
        for monomial_mask in branch_monomials:
            child = support_mask | monomial_mask
            if (
                child.bit_count() > plan.support_cap
                or child in state.seen
                or child & forbidden_mask
            ):
                continue
            added = _indices(child & ~support_mask)
            children.add((child.bit_count(), added, child))
        for child_size, _added, child in sorted(children):
            state.seen.add(child)
            supports_discovered += 1
            heapq.heappush(
                state.frontier,
                (child_size, depth + 1, child),
            )
            if supports_discovered >= plan.discovered_support_cap:
                break

    frontier_remaining = sum(len(state.frontier) for state in states)
    if len(candidate_by_orbit) >= plan.candidate_cap:
        termination = "candidate-cap-reached"
    elif supports_discovered >= plan.discovered_support_cap:
        termination = "discovered-support-cap-reached"
    elif nodes >= plan.node_cap and frontier_remaining:
        termination = "node-cap-reached"
    else:
        termination = "frontier-exhausted-under-support-cap"
    candidates = tuple(
        sorted(
            candidate_by_orbit.values(),
            key=lambda candidate: (
                candidate.support_size,
                candidate.seed_family,
                candidate.support,
            ),
        )
    )
    return SupportCampaignResult(
        plan=plan,
        nodes_examined=nodes,
        supports_discovered=supports_discovered,
        frontier_remaining=frontier_remaining,
        candidates=candidates,
        seed_count=len(seed_rows),
        termination=termination,
    )


def support_campaign_from_dict(
    payload: Mapping,
) -> SupportCampaignResult:
    """Strictly reconstruct and semantically replay a compact result."""

    try:
        plan_payload = payload["plan"]
        plan = SupportCampaignPlan(
            node_cap=plan_payload["node_cap"],
            support_cap=plan_payload["support_cap"],
            candidate_cap=plan_payload["candidate_cap"],
            per_support_size_candidate_quota=plan_payload[
                "per_support_size_candidate_quota"
            ],
            discovered_support_cap=plan_payload[
                "discovered_support_cap"
            ],
            omission_candidate_quota=plan_payload[
                "omission_candidate_quota"
            ],
            activation_width=plan_payload["activation_width"],
            worker_count=plan_payload["worker_count"],
            deterministic_seed=plan_payload["deterministic_seed"],
            scratch_directory=plan_payload["scratch_directory"],
            include_two_coordinate_omissions=plan_payload[
                "include_two_coordinate_omissions"
            ],
            schema=plan_payload["schema"],
        )
        candidates = tuple(
            SupportCandidate(
                seed_name=row["seed_name"],
                seed_family=row["seed_family"],
                support=tuple(row["support"]),
                forbidden_coordinates=tuple(
                    row["forbidden_coordinates"]
                ),
                omitted_natural_coordinates=tuple(
                    row["omitted_natural_coordinates"]
                ),
                orbit_representative=tuple(
                    row["orbit_representative"]
                ),
                orbit_size=row["orbit_size"],
                active_equation_count=row["active_equation_count"],
                feasible_matching_counts=tuple(
                    row["feasible_matching_counts"]
                ),
                closure_depth=row["closure_depth"],
                symmetry_weight_equalities=row[
                    "symmetry_weight_equalities"
                ],
            )
            for row in payload["candidates"]
        )
        result = SupportCampaignResult(
            plan=plan,
            nodes_examined=int(payload["nodes_examined"]),
            supports_discovered=int(payload["supports_discovered"]),
            frontier_remaining=int(payload["frontier_remaining"]),
            candidates=candidates,
            seed_count=int(payload["seed_count"]),
            termination=str(payload["termination"]),
            schema=str(payload["schema"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise KrennCounterexampleSearchError(
            "could not decode support campaign result"
        ) from error
    if result.to_dict() != dict(payload):
        raise KrennCounterexampleSearchError(
            "support campaign payload failed semantic replay"
        )
    return result


__all__ = (
    "AMBIENT_VARIABLES",
    "COUNTEREXAMPLE_SUPPORT_SCHEMA",
    "KrennCounterexampleSearchError",
    "MAX_DISCOVERED_SUPPORTS",
    "MAX_WORKERS",
    "MIN_SEARCH_SUPPORT",
    "SupportCampaignPlan",
    "SupportCampaignResult",
    "SupportCandidate",
    "SupportSeed",
    "canonical_support_representative",
    "default_support_seeds",
    "diagonal_seed_support",
    "greedy_pruned_omission_seed",
    "n6_d3_support_index",
    "n6_d3_system",
    "run_support_campaign",
    "support_campaign_from_dict",
    "support_orbit_size",
    "symmetry_action_table",
    "transport_support",
)
