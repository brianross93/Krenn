"""Bounded direct-GHZ searches from projective victim repairs.

The natural ``n=6,d=3`` seed has one victim coloring, ``002121``.  In the
native perfect-matching order its only active matching is ``M*=1``.  The six
minimum two-coordinate repairs split into two ``S6 x S3`` support orbits,
represented by matchings ``N=0`` and ``N=2``.

This campaign is deliberately narrower than the general counterexample
campaign:

* it starts directly on the victim hyperplane rather than following the
  known Laurent family;
* it uses the full direct-GHZ color-diagonal gauge;
* it searches one size-22 support per repair orbit when such a support is
  available;
* all numerical searches are deterministic, norm bounded, and checkpointed;
  the bounded support preflight is deterministic but not checkpointed; and
* a numerical zero, a support-search miss, or a completed bounded campaign
  is never promoted to an exact witness or a nonexistence proof.

The committed support census already supplies a literal size-22 support for
the ``N=2`` orbit.  It supplies no size-22 ``N=0`` support, so ``N=0`` first
passes through a bounded singleton-closure preflight.  Supports activating
additional victim matchings are rejected fail-closed: they require a
multinomial initializer and are not mislabeled as the two-matching branch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


WORKER_THREAD_LIMITS = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
for _thread_name, _thread_value in WORKER_THREAD_LIMITS.items():
    os.environ[_thread_name] = _thread_value

import numpy as np

from experiments.krenn_quantum_graph.counterexample_exact import (
    ExactVerificationReport,
    ReconstructionBounds,
    reconstruct_complex_candidate,
)
from experiments.krenn_quantum_graph.counterexample_search import (
    MAX_DISCOVERED_SUPPORTS,
    MAX_WORKERS,
    SupportCampaignPlan,
    SupportSeed,
    n6_d3_support_index,
    n6_d3_system,
    run_support_campaign,
    support_campaign_from_dict,
    symmetry_action_table,
)
from experiments.krenn_quantum_graph.numerical_continuation import (
    GHZ_COLOR_DIAGONAL_GAUGE,
    InvariantMetrics,
    LM_RESULT_SCHEMA,
    LMPlan,
    NUMERICAL_BLAS_THREAD_LIMIT,
    NumericalSolveResult,
    ResidualMetrics,
    WeightMetrics,
    complex_output,
    complex_residual,
    natural_gauge_chart,
    project_hard_bounds,
    projective_victim_repair_initializer,
    solve_direct_ghz,
    system_fingerprint,
)
from experiments.krenn_quantum_graph.support_extension import (
    natural_support,
)
from experiments.krenn_quantum_graph.support_search import support_profile
from experiments.krenn_quantum_graph.system import variable_key
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (ROOT / "results").resolve(strict=False)
SCRATCH_ROOT = Path(
    r"D:\KrennScratch\counterexample_search"
).resolve(strict=False)
DEFAULT_SCRATCH = SCRATCH_ROOT / "projective_victim_v1"
DEFAULT_RESULTS = (
    ROOT
    / "results/krenn_quantum_graph/n6_d3_projective_repair_search"
)
SCHEMA = "krenn-n6-d3-projective-repair-campaign-v1"
MANIFEST_SCHEMA = "krenn-n6-d3-projective-repair-manifest-v1"
MASTER_SEED = 60_320_260_725
NATURAL_VICTIM_MATCHING = 1
PROJECTIVE_MATCHING_REPRESENTATIVES = (0, 2)
SUPPORT_SIZE = 22
CAMPAIGN_FILE = "campaign.json"
README_FILE = "README.md"
MANIFEST_FILE = "manifest.json"
EXACT_WITNESS_FILE = "exact_witness.json"
EXACT_VERIFICATION_FILE = "exact_verification.json"
REQUIRED_SOURCE_PATHS = (
    "experiments/krenn_quantum_graph/projective_repair_campaign.py",
    "experiments/krenn_quantum_graph/numerical_continuation.py",
    "experiments/krenn_quantum_graph/counterexample_search.py",
    "experiments/krenn_quantum_graph/counterexample_exact.py",
    "experiments/krenn_quantum_graph/formal_lift.py",
    "experiments/krenn_quantum_graph/independent_verifier.py",
    "experiments/krenn_quantum_graph/n6_deformation.py",
    "experiments/krenn_quantum_graph/support_extension.py",
    "experiments/krenn_quantum_graph/support_search.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "experiments/krenn_quantum_graph/ternary_search.py",
    "experiments/krenn_quantum_graph/ternary_seed_orbits.py",
    "experiments/krenn_quantum_graph/transport.py",
    "experiments/krenn_quantum_graph/witness.py",
    "tests/test_krenn_counterexample_projective_repair.py",
)

EXISTING_SIZE22_SUPPORT_A = (
    0,
    6,
    13,
    18,
    20,
    24,
    26,
    29,
    35,
    45,
    47,
    67,
    72,
    80,
    81,
    83,
    87,
    89,
    92,
    98,
    121,
    126,
)
EXISTING_SIZE22_SUPPORT_B = (
    0,
    13,
    14,
    26,
    67,
    68,
    70,
    71,
    76,
    77,
    79,
    80,
    81,
    97,
    98,
    106,
    107,
    112,
    113,
    121,
    122,
    126,
)


class KrennProjectiveRepairCampaignError(ValueError):
    """A projective branch, budget, support, or result is malformed."""


def _canonical_support(support: Iterable[int]) -> tuple[int, ...]:
    values = tuple(map(int, support))
    if (
        values != tuple(sorted(values))
        or len(values) != len(set(values))
        or any(index < 0 or index >= 135 for index in values)
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective support must contain unique increasing indices"
        )
    return values


def _complex_pair(value: complex) -> list[float]:
    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise KrennProjectiveRepairCampaignError(
            "projective audit contains a nonfinite complex value"
        )
    return [float(value.real), float(value.imag)]


def _seed_words(sequence: np.random.SeedSequence) -> tuple[int, ...]:
    return tuple(
        map(int, sequence.generate_state(4, dtype=np.uint32))
    )


@dataclass(frozen=True)
class ProjectiveBranchSpec:
    """One exact two-matching projective victim branch."""

    matching_index: int
    branch_support: tuple[int, ...]
    numerator_indices: tuple[int, int]
    denominator_indices: tuple[int, int]

    def __post_init__(self) -> None:
        matching = int(self.matching_index)
        branch = _canonical_support(self.branch_support)
        numerator = tuple(sorted(map(int, self.numerator_indices)))
        denominator = tuple(sorted(map(int, self.denominator_indices)))
        object.__setattr__(self, "matching_index", matching)
        object.__setattr__(self, "branch_support", branch)
        object.__setattr__(self, "numerator_indices", numerator)
        object.__setattr__(self, "denominator_indices", denominator)
        if (
            matching not in PROJECTIVE_MATCHING_REPRESENTATIVES
            or len(branch) != 11
            or len(numerator) != 2
            or len(denominator) != 2
            or len(set((*numerator, *denominator))) != 4
            or not set(natural_support()).issubset(branch)
            or not set((*numerator, *denominator)).issubset(branch)
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective branch specification changed"
            )

    @property
    def expected_active_victim_matchings(self) -> tuple[int, int]:
        return tuple(
            sorted((NATURAL_VICTIM_MATCHING, self.matching_index))
        )

    def to_dict(self) -> dict:
        return {
            "matching_index": self.matching_index,
            "natural_victim_matching_index": (
                NATURAL_VICTIM_MATCHING
            ),
            "branch_support": list(self.branch_support),
            "support_size": len(self.branch_support),
            "ratio_numerator_indices": list(self.numerator_indices),
            "ratio_denominator_indices": list(
                self.denominator_indices
            ),
            "initial_relation": "r_N=-1",
            "cross_multiplied_relation": "m_N+m_Mstar=0",
        }


def projective_branch_specs() -> tuple[ProjectiveBranchSpec, ...]:
    """Derive and audit the two branch representatives from the system."""

    system = n6_d3_system()
    monomials = system.equation_monomials(
        N6_D3_SEED_DEFECT_EQUATION
    )
    natural = set(natural_support())
    reference = set(monomials[NATURAL_VICTIM_MATCHING])
    rows = []
    for matching in PROJECTIVE_MATCHING_REPRESENTATIVES:
        candidate = set(monomials[matching])
        rows.append(
            ProjectiveBranchSpec(
                matching_index=matching,
                branch_support=tuple(
                    sorted(natural.union(candidate))
                ),
                numerator_indices=tuple(
                    sorted(candidate.difference(reference))
                ),
                denominator_indices=tuple(
                    sorted(reference.difference(candidate))
                ),
            )
        )
    expected = (
        ProjectiveBranchSpec(
            matching_index=0,
            branch_support=(
                0,
                13,
                26,
                67,
                80,
                81,
                88,
                98,
                121,
                126,
                133,
            ),
            numerator_indices=(88, 133),
            denominator_indices=(98, 121),
        ),
        ProjectiveBranchSpec(
            matching_index=2,
            branch_support=(
                0,
                13,
                26,
                67,
                80,
                81,
                98,
                106,
                113,
                121,
                126,
            ),
            numerator_indices=(106, 113),
            denominator_indices=(98, 121),
        ),
    )
    result = tuple(rows)
    if result != expected:
        raise KrennProjectiveRepairCampaignError(
            "projective branch derivation changed"
        )
    return result


def branch_spec(matching_index: int) -> ProjectiveBranchSpec:
    matching_index = int(matching_index)
    try:
        return next(
            row
            for row in projective_branch_specs()
            if row.matching_index == matching_index
        )
    except StopIteration as error:
        raise KrennProjectiveRepairCampaignError(
            "unknown projective branch representative"
        ) from error


def _branch_orbit_images(
    spec: ProjectiveBranchSpec,
) -> frozenset[tuple[int, ...]]:
    table = symmetry_action_table()
    images = np.sort(
        table[:, np.asarray(spec.branch_support, dtype=np.intp)],
        axis=1,
    )
    result = frozenset(
        tuple(map(int, row)) for row in images.tolist()
    )
    if len(result) != 1_080:
        raise KrennProjectiveRepairCampaignError(
            "projective branch orbit size changed"
        )
    return result


def branch_orbit_hits(
    support: Iterable[int],
    spec: ProjectiveBranchSpec,
) -> tuple[tuple[int, ...], ...]:
    """Return every distinct branch-orbit image contained in ``support``."""

    canonical = _canonical_support(support)
    support_set = set(canonical)
    return tuple(
        sorted(
            image
            for image in _branch_orbit_images(spec)
            if set(image).issubset(support_set)
        )
    )


def active_victim_matchings(
    support: Iterable[int],
) -> tuple[int, ...]:
    canonical = _canonical_support(support)
    support_set = set(canonical)
    return tuple(
        matching
        for matching, monomial in enumerate(
            n6_d3_system().equation_monomials(
                N6_D3_SEED_DEFECT_EQUATION
            )
        )
        if set(monomial).issubset(support_set)
    )


def validate_projective_support(
    support: Iterable[int],
    spec: ProjectiveBranchSpec,
) -> dict:
    """Replay the exact support-level conditions used by numerical jobs."""

    canonical = _canonical_support(support)
    profile = support_profile(
        n6_d3_system(),
        canonical,
        index=n6_d3_support_index(),
    )
    active = active_victim_matchings(canonical)
    branch_contained = set(spec.branch_support).issubset(canonical)
    exact_two_matching_branch = (
        active == spec.expected_active_victim_matchings
    )
    eligible = bool(
        len(canonical) == SUPPORT_SIZE
        and branch_contained
        and profile.singleton_free_necessary_condition
        and exact_two_matching_branch
    )
    return {
        "support": list(canonical),
        "support_size": len(canonical),
        "matching_index": spec.matching_index,
        "branch_contained_literally": branch_contained,
        "singleton_free_necessary_condition": (
            profile.singleton_free_necessary_condition
        ),
        "active_victim_matching_indices": list(active),
        "expected_active_victim_matching_indices": list(
            spec.expected_active_victim_matchings
        ),
        "multinomial_victim_departure": (
            branch_contained and not exact_two_matching_branch
        ),
        "eligible_for_two_matching_initializer": eligible,
        "solution_certified": False,
        "nonexistence_proved": False,
    }


def existing_support_inventory() -> dict:
    """Audit the two committed size-22 supports against both branch orbits."""

    records = []
    for label, support in (
        ("committed-size22-A", EXISTING_SIZE22_SUPPORT_A),
        ("committed-size22-B", EXISTING_SIZE22_SUPPORT_B),
    ):
        hits = {
            str(spec.matching_index): branch_orbit_hits(
                support, spec
            )
            for spec in projective_branch_specs()
        }
        records.append(
            {
                "label": label,
                "support": list(support),
                "support_size": len(support),
                "branch_orbit_hit_counts": {
                    key: len(value) for key, value in hits.items()
                },
                "branch_orbit_images": {
                    key: [list(image) for image in value]
                    for key, value in hits.items()
                },
            }
        )
    if (
        tuple(
            record["branch_orbit_hit_counts"]
            for record in records
        )
        != ({"0": 0, "2": 2}, {"0": 0, "2": 2})
        or not validate_projective_support(
            EXISTING_SIZE22_SUPPORT_B, branch_spec(2)
        )["eligible_for_two_matching_initializer"]
    ):
        raise KrennProjectiveRepairCampaignError(
            "committed size-22 projective support inventory changed"
        )
    return {
        "supports": records,
        "n0_size22_support_present": False,
        "n2_literal_numerical_support": list(
            EXISTING_SIZE22_SUPPORT_B
        ),
    }


def run_n0_support_preflight(
    *,
    node_cap: int,
    discovered_support_cap: int,
    candidate_scan_cap: int,
    worker_count: int,
    scratch_directory: Path | str,
    master_seed: int = MASTER_SEED,
) -> tuple[dict, tuple[tuple[int, ...], ...]]:
    """Bounded singleton closure from the literal ``N=0`` branch."""

    spec = branch_spec(0)
    plan = SupportCampaignPlan(
        node_cap=int(node_cap),
        support_cap=SUPPORT_SIZE,
        candidate_cap=int(candidate_scan_cap),
        per_support_size_candidate_quota=int(candidate_scan_cap),
        discovered_support_cap=int(discovered_support_cap),
        omission_candidate_quota=0,
        # This is the exhaustive ambient activation list whenever closure
        # first reaches a singleton-free support below size 22.  It is not
        # the old width-8 heuristic.
        activation_width=135,
        worker_count=int(worker_count),
        deterministic_seed=int(master_seed),
        scratch_directory=str(scratch_directory),
        include_two_coordinate_omissions=False,
    )
    result = run_support_campaign(
        plan,
        seeds=(
            SupportSeed(
                name="projective-victim-N0",
                family="projective-victim-repair",
                support=spec.branch_support,
            ),
        ),
    )
    validations = tuple(
        validate_projective_support(candidate.support, spec)
        for candidate in result.candidates
    )
    eligible = tuple(
        tuple(row["support"])
        for row in validations
        if row["eligible_for_two_matching_initializer"]
    )
    return (
        {
            "support_campaign": result.to_dict(),
            "candidate_validations": list(validations),
            "eligible_support_count": len(eligible),
            "eligible_supports": [list(row) for row in eligible],
            "preflight_master_seed": int(master_seed),
            "activation_width": 135,
            "activation_policy": (
                "all-ambient-coordinates-ranked-deterministically"
            ),
            "bounded_miss_proves_n0_nonexistence": False,
            "multinomial_candidates_are_excluded_globally": False,
            "multinomial_initializer_implemented": False,
            "candidate_cap_applies_before_two_matching_filter": True,
            "support_orbit_deduplication": "full-S6xS3-support-only",
            "decorated_projective_branch_orbits_deduplicated": False,
            "bounded_preflight_is_exhaustive": False,
        },
        eligible,
    )


@dataclass(frozen=True)
class ProjectiveNumericalJob:
    """One deterministic direct-GHZ solve at one norm radius."""

    matching_index: int
    support: tuple[int, ...]
    start: int
    radius: float
    repair_scale: float
    master_seed: int = MASTER_SEED

    def __post_init__(self) -> None:
        spec = branch_spec(self.matching_index)
        support = _canonical_support(self.support)
        start = int(self.start)
        radius = float(self.radius)
        scale = float(self.repair_scale)
        seed = int(self.master_seed)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "repair_scale", scale)
        object.__setattr__(self, "master_seed", seed)
        validation = validate_projective_support(support, spec)
        if (
            not validation["eligible_for_two_matching_initializer"]
            or start < 0
            or not math.isfinite(radius)
            or radius < 1.0
            or not math.isfinite(scale)
            or scale <= 0.0
            or seed < 0
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective numerical job is malformed"
            )

    @property
    def seed_sequence(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(
            self.master_seed,
            spawn_key=(self.matching_index, self.start),
        )

    @property
    def seed_words(self) -> tuple[int, ...]:
        return _seed_words(self.seed_sequence)

    @property
    def run_id(self) -> str:
        radius = format(self.radius, ".17g").replace(".", "p")
        return (
            f"projective-N{self.matching_index}-"
            f"start-{self.start:04d}-radius-{radius}"
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "matching_index": self.matching_index,
            "support": list(self.support),
            "support_size": len(self.support),
            "start": self.start,
            "radius": self.radius,
            "repair_scale": self.repair_scale,
            "master_seed": self.master_seed,
            "seed_spawn_key": [self.matching_index, self.start],
            "seed_words_uint32": list(self.seed_words),
            "initializer_kind": "projective-victim-repair",
            "target": "direct-GHZ",
            "known_laurent_initializer_used": False,
            "continuation_used": False,
            "symmetry_weight_equalities": 0,
        }


def deterministic_jobs(
    supports_by_matching: Mapping[int, Sequence[int]],
    *,
    starts_per_support: int,
    radii: Sequence[float],
    repair_scales: Sequence[float],
    master_seed: int,
) -> tuple[ProjectiveNumericalJob, ...]:
    """Build paired-radius jobs with scheduling-independent seeds."""

    starts = int(starts_per_support)
    radius_values = tuple(map(float, radii))
    scales = tuple(map(float, repair_scales))
    if (
        not 1 <= starts <= 1_000
        or not radius_values
        or len(set(radius_values)) != len(radius_values)
        or any(
            not math.isfinite(radius) or radius < 1.0
            for radius in radius_values
        )
        or not scales
        or any(
            not math.isfinite(scale) or scale <= 0.0
            for scale in scales
        )
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective numerical budgets are invalid"
        )
    jobs = []
    for matching in sorted(map(int, supports_by_matching)):
        support = tuple(supports_by_matching[matching])
        for start in range(starts):
            for radius in radius_values:
                jobs.append(
                    ProjectiveNumericalJob(
                        matching_index=matching,
                        support=support,
                        start=start,
                        radius=radius,
                        repair_scale=scales[start % len(scales)],
                        master_seed=int(master_seed),
                    )
                )
    return tuple(jobs)


def victim_projective_audit(
    weights: Sequence[complex],
    spec: ProjectiveBranchSpec,
) -> dict:
    """Record all victim matching amplitudes and the distinguished ratio."""

    dense = np.asarray(weights, dtype=np.complex128)
    if dense.shape != (135,):
        raise KrennProjectiveRepairCampaignError(
            "projective audit needs 135 dense weights"
        )
    monomials = n6_d3_system().equation_monomials(
        N6_D3_SEED_DEFECT_EQUATION
    )
    amplitudes = tuple(
        complex(
            dense[monomial[0]]
            * dense[monomial[1]]
            * dense[monomial[2]]
        )
        for monomial in monomials
    )

    def product(indices: Sequence[int]) -> complex:
        result = 1.0 + 0.0j
        for index in indices:
            result *= complex(dense[index])
        return result

    numerator = product(spec.numerator_indices)
    denominator = product(spec.denominator_indices)
    ratio = numerator / denominator if denominator else None
    total = complex(math.fsum(value.real for value in amplitudes))
    total += 1.0j * math.fsum(
        value.imag for value in amplitudes
    )
    cross = numerator + denominator
    return {
        "matching_amplitudes": [
            _complex_pair(value) for value in amplitudes
        ],
        "nonzero_matching_indices_at_1e-14": [
            index
            for index, value in enumerate(amplitudes)
            if abs(value) > 1.0e-14
        ],
        "distinguished_matching_index": spec.matching_index,
        "ratio_numerator": _complex_pair(numerator),
        "ratio_denominator": _complex_pair(denominator),
        "r_N": _complex_pair(ratio) if ratio is not None else None,
        "cross_multiplied_relation_error": _complex_pair(cross),
        "full_victim_hyperplane_error": _complex_pair(total),
    }


def _run_job(
    task: tuple[
        ProjectiveNumericalJob,
        int,
        int,
        int,
        str,
        bool,
    ],
) -> tuple[ProjectiveNumericalJob, dict, NumericalSolveResult]:
    job, iterations, evaluations, checkpoint_interval, scratch, resume = (
        task
    )
    for name, value in WORKER_THREAD_LIMITS.items():
        os.environ[name] = value
    spec = branch_spec(job.matching_index)
    chart = natural_gauge_chart(
        job.support,
        action_group=GHZ_COLOR_DIAGONAL_GAUGE,
    )
    if not any(
        index in chart.free_indices
        for index in spec.numerator_indices
    ):
        raise KrennProjectiveRepairCampaignError(
            "distinguished projective numerator has no free gauge coordinate"
        )
    initializer = projective_victim_repair_initializer(
        job.support,
        job.seed_sequence,
        numerator_indices=spec.numerator_indices,
        denominator_indices=spec.denominator_indices,
        expected_active_victim_matching_indices=(
            spec.expected_active_victim_matchings
        ),
        repair_scale=job.repair_scale,
        chart=chart,
        l2_bound=job.radius,
        linf_bound=job.radius,
    )
    solver_initial, solver_projection = project_hard_bounds(
        initializer.dense_weights(),
        chart,
        l2_bound=job.radius,
        linf_bound=job.radius,
    )
    if (
        solver_projection.changed
        or not np.array_equal(
            solver_initial, initializer.dense_weights()
        )
    ):
        raise KrennProjectiveRepairCampaignError(
            "LM initial hard projection is not a bitwise fixed point"
        )
    initial_audit = victim_projective_audit(
        initializer.dense_weights(), spec
    )
    initial_audit["solver_initial_projection_bitwise_fixed"] = True
    initial_audit["solver_initial_projection_changed"] = False
    plan = LMPlan(
        max_iterations=int(iterations),
        max_evaluations=int(evaluations),
        l2_bound=job.radius,
        linf_bound=job.radius,
        checkpoint_interval=int(checkpoint_interval),
    )
    checkpoint_prefix = (
        Path(scratch) / job.run_id / "checkpoint"
    )
    resume_this_job = bool(
        resume and checkpoint_prefix.with_suffix(".json").is_file()
    )
    result = solve_direct_ghz(
        initializer,
        job.support,
        chart=chart,
        plan=plan,
        run_id=job.run_id,
        checkpoint_prefix=checkpoint_prefix,
        resume=resume_this_job,
    )
    return job, initial_audit, result


def _validate_paths(
    scratch_directory: Path | str,
    results_directory: Path | str,
) -> tuple[Path, Path]:
    scratch = Path(scratch_directory).resolve(strict=False)
    results = Path(results_directory).resolve(strict=False)
    try:
        scratch.relative_to(SCRATCH_ROOT)
    except ValueError as error:
        raise KrennProjectiveRepairCampaignError(
            "projective scratch must stay under "
            "D:\\KrennScratch\\counterexample_search"
        ) from error
    try:
        results.relative_to(RESULTS_ROOT)
    except ValueError as error:
        raise KrennProjectiveRepairCampaignError(
            "projective results must stay under the standalone Krenn "
            "repository results directory"
        ) from error
    if "onedrive" in str(scratch).casefold() or "onedrive" in str(
        results
    ).casefold():
        raise KrennProjectiveRepairCampaignError(
            "projective campaign paths cannot use OneDrive"
        )
    return scratch, results


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(
            (
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennProjectiveRepairCampaignError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str):
    raise KrennProjectiveRepairCampaignError(
        f"nonfinite JSON constant {value!r} is forbidden"
    )


def _load_json(path: Path) -> Mapping:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.parent.resolve() != path.resolve().parent
    ):
        raise KrennProjectiveRepairCampaignError(
            f"projective artifact is absent, linked, or unsafe: "
            f"{path.name}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise KrennProjectiveRepairCampaignError(
            f"could not decode projective artifact {path.name}"
        ) from error
    if not isinstance(payload, Mapping):
        raise KrennProjectiveRepairCampaignError(
            f"projective artifact {path.name} is not a JSON object"
        )
    return payload


def _file_record(path: Path, *, relative_to: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _readme_text(payload: Mapping) -> str:
    execution = payload["execution"]
    return f"""# Projective victim-repair numerical campaign

This bundle records a bounded, deterministic direct-GHZ search from the two
projective victim-repair orbit representatives.  It does not use the known
Laurent initializer or continuation.

Status: `{payload["status"]}`

- N=0 size-22 support found by this bounded preflight:
  `{str("0" in payload["selected_supports"]).lower()}`
- N=2 literal size-22 support used: `true`
- numerical jobs completed: `{execution["jobs_completed"]}`
- exact dual-enumerator candidate verified:
  `{str(payload["exact_candidate_dual_verified"]).lower()}`

The relation `r_N=-1` and the full victim hyperplane are satisfied only at
initialization (within the recorded floating tolerance).  They are not solver
constraints.  A bounded support miss or numerical miss proves nothing.

## Reproduction

```powershell
{execution["canonical_command"]}
```

Verify the compact bundle without scratch checkpoints:

```powershell
{execution["verify_command"]}
```
"""


def _write_bundle(
    directory: Path,
    payload: Mapping,
    exact_report: ExactVerificationReport | None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    expected = {CAMPAIGN_FILE, README_FILE, MANIFEST_FILE}
    if exact_report is not None:
        expected.update(
            {EXACT_WITNESS_FILE, EXACT_VERIFICATION_FILE}
        )
    unexpected = {
        path.name for path in directory.iterdir()
        if path.name not in expected
    }
    if unexpected:
        raise KrennProjectiveRepairCampaignError(
            f"refusing projective bundle with unexpected files: "
            f"{sorted(unexpected)}"
        )
    _write_json(directory / CAMPAIGN_FILE, payload)
    (directory / README_FILE).write_bytes(
        _readme_text(payload).encode("utf-8")
    )
    if exact_report is not None:
        _write_json(
            directory / EXACT_WITNESS_FILE,
            exact_report.witness.to_dict(),
        )
        _write_json(
            directory / EXACT_VERIFICATION_FILE,
            exact_report.to_dict(),
        )
    artifacts = tuple(
        sorted(expected.difference((MANIFEST_FILE,)))
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "producer": _file_record(
            Path(__file__), relative_to=ROOT
        ),
        "artifacts": [
            _file_record(directory / name, relative_to=directory)
            for name in artifacts
        ],
        "inputs": [
            _file_record(ROOT / name, relative_to=ROOT)
            for name in REQUIRED_SOURCE_PATHS
        ],
        "scratch_data_included": False,
        "scratch_data_required_for_verification": False,
    }
    _write_json(directory / MANIFEST_FILE, manifest)
    verify_bundle(directory)
    return directory / CAMPAIGN_FILE


def _reconstruction_record_from_weights(
    weights: Sequence[complex] | np.ndarray,
    support: Sequence[int],
    *,
    residual_linf: float | None,
    weights_finite: bool,
    trigger: float,
) -> tuple[dict, ExactVerificationReport | None]:
    if (
        residual_linf is None
        or residual_linf > trigger
        or not weights_finite
    ):
        return {"status": "not-attempted"}, None
    report = reconstruct_complex_candidate(
        weights,
        n=6,
        d=3,
        support=support,
        bounds=ReconstructionBounds(
            absolute_tolerance=max(trigger, 1.0e-10),
            relative_tolerance=max(trigger, 1.0e-10),
            zero_tolerance=1.0e-10,
            max_denominator=1_024,
            max_numerator=4_096,
            coefficient_grid_denominator=8,
            coefficient_grid_numerator=12,
            max_polynomial_checks=4_000,
            max_field_attempts=96,
        ),
        preserve_support=False,
    )
    exact = bool(
        report.exact
        and report.exact_attempts
        and report.exact_attempts[0].verification is not None
        and report.exact_attempts[0].verification.exact
    )
    exact_report = (
        report.exact_attempts[0].verification if exact else None
    )
    return {
        "status": (
            "exact-dual-verification-passed"
            if exact
            else "attempted-no-exact-reconstruction"
        ),
        "report": report.to_dict(),
        "dual_enumerator_witness_bundle_emitted": exact,
    }, exact_report


def _reconstruction_record(
    result: NumericalSolveResult,
    *,
    trigger: float,
) -> tuple[dict, ExactVerificationReport | None]:
    return _reconstruction_record_from_weights(
        result.dense_weights(),
        result.support,
        residual_linf=result.residual_metrics.linf,
        weights_finite=result.weight_metrics.finite,
        trigger=trigger,
    )


def run_campaign(
    args: argparse.Namespace,
    *,
    argv_receipt: Sequence[str] | None = None,
) -> Path:
    """Execute the bounded support preflight and direct numerical jobs."""

    scratch, results = _validate_paths(
        args.scratch_directory, args.results_directory
    )
    if (
        not 1 <= int(args.workers) <= MAX_WORKERS
        or int(args.supports_per_orbit) != 1
        or not 1 <= int(args.support_nodes) <= 1_000_000
        or not 1
        <= int(args.support_state_cap)
        <= MAX_DISCOVERED_SUPPORTS
        or not 1 <= int(args.support_candidate_scan) <= 10_000
        or not math.isfinite(float(args.reconstruction_trigger))
        or float(args.reconstruction_trigger) <= 0.0
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective campaign budgets are invalid"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    preflight, n0_supports = run_n0_support_preflight(
        node_cap=args.support_nodes,
        discovered_support_cap=args.support_state_cap,
        candidate_scan_cap=args.support_candidate_scan,
        worker_count=args.workers,
        scratch_directory=scratch,
        master_seed=args.master_seed,
    )
    supports_by_matching: dict[int, tuple[int, ...]] = {
        2: EXISTING_SIZE22_SUPPORT_B,
    }
    if n0_supports:
        supports_by_matching[0] = n0_supports[0]
    jobs = deterministic_jobs(
        supports_by_matching,
        starts_per_support=args.starts_per_support,
        radii=args.radii,
        repair_scales=args.repair_scales,
        master_seed=args.master_seed,
    )
    plan_arguments = (
        int(args.iterations),
        int(args.evaluations),
        int(args.checkpoint_interval),
        str(scratch),
        bool(args.resume),
    )
    tasks = tuple(
        (job, *plan_arguments) for job in jobs
    )
    with ProcessPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        completed = tuple(executor.map(_run_job, tasks))

    rows = []
    exact_report: ExactVerificationReport | None = None
    for job, initial_audit, result in completed:
        spec = branch_spec(job.matching_index)
        reconstruction, candidate_report = _reconstruction_record(
            result, trigger=float(args.reconstruction_trigger)
        )
        if candidate_report is not None and exact_report is None:
            exact_report = candidate_report
        rows.append(
            {
                "job": job.to_dict(),
                "initializer": {
                    "projective_audit": initial_audit,
                    "full_victim_hyperplane_satisfied_at_initialization_only": (
                        True
                    ),
                    "initialization_tolerance": (
                        256.0 * np.finfo(np.float64).eps
                    ),
                    "solver_constrains_projective_relation": False,
                    "relation_projection_or_clipping_used": False,
                },
                "result": result.to_dict(include_trace=False),
                "final_projective_audit": victim_projective_audit(
                    result.dense_weights(), spec
                ),
                "reconstruction": reconstruction,
            }
        )

    exact_candidate = exact_report is not None
    effective_argv = tuple(
        map(str, argv_receipt if argv_receipt is not None else ())
    )
    python = sys.executable
    canonical_command = subprocess.list2cmdline(
        (
            python,
            "-B",
            "-m",
            "experiments.krenn_quantum_graph.projective_repair_campaign",
            *effective_argv,
        )
    )
    verify_command = subprocess.list2cmdline(
        (
            python,
            "-B",
            "-m",
            "experiments.krenn_quantum_graph.projective_repair_campaign",
            "--verify-only",
            "--scratch-directory",
            str(scratch),
            "--results-directory",
            str(results),
        )
    )
    payload = {
        "schema": SCHEMA,
        "status": (
            "exact-candidate-dual-verified"
            if exact_candidate
            else "bounded-search-complete-no-exact-candidate"
        ),
        "system_fingerprint": system_fingerprint(),
        "branch_specs": [
            spec.to_dict() for spec in projective_branch_specs()
        ],
        "existing_support_inventory": existing_support_inventory(),
        "n0_support_preflight": preflight,
        "selected_supports": {
            str(key): list(value)
            for key, value in sorted(supports_by_matching.items())
        },
        "execution": {
            "workers": int(args.workers),
            "blas_threads_per_worker": (
                NUMERICAL_BLAS_THREAD_LIMIT
            ),
            "worker_thread_environment": dict(
                WORKER_THREAD_LIMITS
            ),
            "producer_python": sys.executable,
            "master_seed": int(args.master_seed),
            "support_preflight_master_seed": int(args.master_seed),
            "support_preflight_node_cap": int(args.support_nodes),
            "support_preflight_discovered_state_cap": int(
                args.support_state_cap
            ),
            "support_preflight_raw_candidate_cap": int(
                args.support_candidate_scan
            ),
            "support_preflight_activation_width": 135,
            "support_preflight_activation_policy": (
                "all-ambient-coordinates-ranked-deterministically"
            ),
            "supports_per_projective_orbit": int(
                args.supports_per_orbit
            ),
            "starts_per_support": int(args.starts_per_support),
            "radii": list(map(float, args.radii)),
            "repair_scales": list(
                map(float, args.repair_scales)
            ),
            "iterations_per_job": int(args.iterations),
            "evaluations_per_job": int(args.evaluations),
            "checkpoint_interval": int(
                args.checkpoint_interval
            ),
            "reconstruction_trigger": float(
                args.reconstruction_trigger
            ),
            "scratch_directory": str(scratch),
            "results_directory": str(results),
            "known_laurent_initializer_used": False,
            "continuation_used": False,
            "target": "direct-GHZ",
            "jobs_scheduled": len(jobs),
            "jobs_completed": len(completed),
            "argv_receipt": list(effective_argv),
            "canonical_command": canonical_command,
            "verify_command": verify_command,
        },
        "numerical_jobs": rows,
        "exact_candidate_dual_verified": exact_candidate,
        "claim_boundary": {
            "bounded_support_miss_proves_nonexistence": False,
            "bounded_numerical_miss_proves_nonexistence": False,
            "numerical_zero_is_exact_witness": False,
            "n0_support_22_nonexistence_proved": False,
            "multinomial_n0_branches_excluded": False,
            "known_laurent_continuation_used": False,
            "all_729_equations_evaluated_per_job": True,
            "exact_counterexample_certified": exact_candidate,
        },
    }
    return _write_bundle(results, payload, exact_report)


def verify_payload(payload: Mapping) -> dict:
    """Verify immutable structure and fail-closed claims of one result."""

    expected_payload_keys = {
        "schema",
        "status",
        "system_fingerprint",
        "branch_specs",
        "existing_support_inventory",
        "n0_support_preflight",
        "selected_supports",
        "execution",
        "numerical_jobs",
        "exact_candidate_dual_verified",
        "claim_boundary",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_payload_keys
        or payload.get("schema") != SCHEMA
        or payload.get("system_fingerprint") != system_fingerprint()
        or payload.get("branch_specs")
        != [spec.to_dict() for spec in projective_branch_specs()]
        or payload.get("existing_support_inventory")
        != existing_support_inventory()
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective campaign payload failed exact replay"
        )
    execution = payload.get("execution")
    expected_execution_keys = {
        "workers",
        "blas_threads_per_worker",
        "worker_thread_environment",
        "producer_python",
        "master_seed",
        "support_preflight_master_seed",
        "support_preflight_node_cap",
        "support_preflight_discovered_state_cap",
        "support_preflight_raw_candidate_cap",
        "support_preflight_activation_width",
        "support_preflight_activation_policy",
        "supports_per_projective_orbit",
        "starts_per_support",
        "radii",
        "repair_scales",
        "iterations_per_job",
        "evaluations_per_job",
        "checkpoint_interval",
        "reconstruction_trigger",
        "scratch_directory",
        "results_directory",
        "known_laurent_initializer_used",
        "continuation_used",
        "target",
        "jobs_scheduled",
        "jobs_completed",
        "argv_receipt",
        "canonical_command",
        "verify_command",
    }
    if (
        not isinstance(execution, Mapping)
        or set(execution) != expected_execution_keys
        or not 1 <= int(execution.get("workers", 0)) <= MAX_WORKERS
        or execution.get("blas_threads_per_worker")
        != NUMERICAL_BLAS_THREAD_LIMIT
        or execution.get("worker_thread_environment")
        != WORKER_THREAD_LIMITS
        or not isinstance(execution.get("producer_python"), str)
        or not execution.get("producer_python")
        or not Path(execution["producer_python"]).is_absolute()
        or execution.get(
            "known_laurent_initializer_used"
        )
        is not False
        or execution.get("continuation_used")
        is not False
        or execution.get("target") != "direct-GHZ"
        or not math.isfinite(
            float(execution.get("reconstruction_trigger", 0.0))
        )
        or float(execution.get("reconstruction_trigger", 0.0))
        <= 0.0
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective campaign execution ledger changed"
        )
    scratch, results = _validate_paths(
        execution["scratch_directory"],
        execution["results_directory"],
    )
    argv_receipt = execution.get("argv_receipt")
    if (
        not isinstance(argv_receipt, Sequence)
        or isinstance(argv_receipt, (str, bytes, bytearray))
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective command receipt changed"
        )
    expected_command = subprocess.list2cmdline(
        (
            execution["producer_python"],
            "-B",
            "-m",
            "experiments.krenn_quantum_graph.projective_repair_campaign",
            *tuple(map(str, argv_receipt)),
        )
    )
    expected_verify_command = subprocess.list2cmdline(
        (
            execution["producer_python"],
            "-B",
            "-m",
            "experiments.krenn_quantum_graph.projective_repair_campaign",
            "--verify-only",
            "--scratch-directory",
            str(scratch),
            "--results-directory",
            str(results),
        )
    )
    if (
        execution.get("canonical_command") != expected_command
        or execution.get("verify_command") != expected_verify_command
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective command receipt failed replay"
        )
    selected_payload = payload.get("selected_supports")
    if (
        not isinstance(selected_payload, Mapping)
        or set(selected_payload).difference({"0", "2"})
        or selected_payload.get("2")
        != list(EXISTING_SIZE22_SUPPORT_B)
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective selected-support ledger changed"
        )
    selected = {
        int(key): tuple(value)
        for key, value in selected_payload.items()
    }
    if any(
        not validate_projective_support(
            support, branch_spec(matching)
        )["eligible_for_two_matching_initializer"]
        for matching, support in selected.items()
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective selected support failed semantic replay"
        )
    preflight = payload.get("n0_support_preflight")
    expected_preflight_keys = {
        "support_campaign",
        "candidate_validations",
        "eligible_support_count",
        "eligible_supports",
        "preflight_master_seed",
        "activation_width",
        "activation_policy",
        "bounded_miss_proves_n0_nonexistence",
        "multinomial_candidates_are_excluded_globally",
        "multinomial_initializer_implemented",
        "candidate_cap_applies_before_two_matching_filter",
        "support_orbit_deduplication",
        "decorated_projective_branch_orbits_deduplicated",
        "bounded_preflight_is_exhaustive",
    }
    if (
        not isinstance(preflight, Mapping)
        or set(preflight) != expected_preflight_keys
        or preflight.get("preflight_master_seed")
        != execution.get("master_seed")
        or preflight.get("activation_width") != 135
        or preflight.get(
            "bounded_miss_proves_n0_nonexistence"
        )
        is not False
        or preflight.get(
            "multinomial_candidates_are_excluded_globally"
        )
        is not False
        or preflight.get("multinomial_initializer_implemented")
        is not False
        or preflight.get(
            "candidate_cap_applies_before_two_matching_filter"
        )
        is not True
        or preflight.get("support_orbit_deduplication")
        != "full-S6xS3-support-only"
        or preflight.get(
            "decorated_projective_branch_orbits_deduplicated"
        )
        is not False
        or preflight.get("bounded_preflight_is_exhaustive")
        is not False
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective N0 preflight receipt changed"
        )
    support_result = support_campaign_from_dict(
        preflight["support_campaign"]
    )
    if (
        support_result.to_dict() != preflight["support_campaign"]
        or support_result.plan.node_cap
        != execution.get("support_preflight_node_cap")
        or support_result.plan.discovered_support_cap
        != execution.get("support_preflight_discovered_state_cap")
        or support_result.plan.candidate_cap
        != execution.get("support_preflight_raw_candidate_cap")
        or support_result.plan.per_support_size_candidate_quota
        != execution.get("support_preflight_raw_candidate_cap")
        or support_result.plan.activation_width != 135
        or support_result.plan.deterministic_seed
        != execution.get("master_seed")
        or execution.get("supports_per_projective_orbit") != 1
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective N0 support plan or accounting changed"
        )
    replayed_validations = [
        validate_projective_support(candidate.support, branch_spec(0))
        for candidate in support_result.candidates
    ]
    replayed_eligible = [
        row["support"]
        for row in replayed_validations
        if row["eligible_for_two_matching_initializer"]
    ]
    if (
        preflight["candidate_validations"] != replayed_validations
        or preflight["eligible_supports"] != replayed_eligible
        or preflight["eligible_support_count"]
        != len(replayed_eligible)
        or (
            selected_payload.get("0")
            != (
                replayed_eligible[0]
                if replayed_eligible
                else None
            )
        )
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective N0 candidate filtering changed"
        )
    expected_jobs = deterministic_jobs(
        selected,
        starts_per_support=execution["starts_per_support"],
        radii=execution["radii"],
        repair_scales=execution["repair_scales"],
        master_seed=execution["master_seed"],
    )
    rows = payload.get("numerical_jobs")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or len(rows) != len(expected_jobs)
        or execution.get("jobs_scheduled") != len(expected_jobs)
        or execution.get("jobs_completed") != len(expected_jobs)
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective numerical job accounting changed"
        )
    exact_reports: list[ExactVerificationReport] = []
    expected_result_claim_boundary = {
        "bounded_deterministic_numerical_search_only": True,
        "all_729_residuals_evaluated": True,
        "fixed_gauge_chart": True,
        "hard_l2_and_linf_controls": True,
        "numerical_zero_is_exact_witness": False,
        "exact_verification_performed": False,
        "bounded_miss_proves_nonexistence": False,
        "finite_field_transfer_used": False,
        "nonexistence_proved": False,
    }
    expected_result_keys = {
        "schema",
        "run_id",
        "parameters",
        "support",
        "support_size",
        "gauge_chart",
        "plan",
        "target",
        "status",
        "iterations",
        "evaluations",
        "accepted_steps",
        "rejected_steps",
        "projection_count",
        "configured_blas_thread_limit_per_process",
        "weights",
        "residual_metrics",
        "weight_metrics",
        "gauge_invariant_metrics",
        "trace",
        "numerical_zero_under_declared_tolerance",
        "weights_remain_finite",
        "interior_bounded_candidate",
        "exact_solution_certified",
        "claim_boundary",
    }
    allowed_statuses = {
        "numerical-residual-tolerance",
        "iteration-cap-reached",
        "evaluation-cap-reached",
        "gradient-stalled",
        "step-stalled",
        "norm-bound-stalled",
        "nonfinite-evaluation",
        "linear-algebra-failure",
        "invalid-step",
        "checkpoint-paused",
    }
    for row, job in zip(rows, expected_jobs, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "job",
                "initializer",
                "result",
                "final_projective_audit",
                "reconstruction",
            }
            or row.get("job") != job.to_dict()
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective numerical seed receipt changed"
            )
        spec = branch_spec(job.matching_index)
        chart = natural_gauge_chart(
            job.support,
            action_group=GHZ_COLOR_DIAGONAL_GAUGE,
        )
        replayed_initializer = projective_victim_repair_initializer(
            job.support,
            job.seed_sequence,
            numerator_indices=spec.numerator_indices,
            denominator_indices=spec.denominator_indices,
            expected_active_victim_matching_indices=(
                spec.expected_active_victim_matchings
            ),
            repair_scale=job.repair_scale,
            chart=chart,
            l2_bound=job.radius,
            linf_bound=job.radius,
        )
        reprojected_initializer, initializer_projection = (
            project_hard_bounds(
                replayed_initializer.dense_weights(),
                chart,
                l2_bound=job.radius,
                linf_bound=job.radius,
            )
        )
        if (
            initializer_projection.changed
            or not np.array_equal(
                reprojected_initializer,
                replayed_initializer.dense_weights(),
            )
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective initializer stopped being a fixed point"
            )
        replayed_initializer_audit = victim_projective_audit(
            replayed_initializer.dense_weights(), spec
        )
        replayed_initializer_audit[
            "solver_initial_projection_bitwise_fixed"
        ] = True
        replayed_initializer_audit[
            "solver_initial_projection_changed"
        ] = False
        expected_initializer = {
            "projective_audit": replayed_initializer_audit,
            "full_victim_hyperplane_satisfied_at_initialization_only": (
                True
            ),
            "initialization_tolerance": (
                256.0 * np.finfo(np.float64).eps
            ),
            "solver_constrains_projective_relation": False,
            "relation_projection_or_clipping_used": False,
        }
        initializer = row.get("initializer")
        result = row.get("result")
        if (
            initializer != expected_initializer
            or not isinstance(result, Mapping)
            or set(result) != expected_result_keys
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective numerical result boundary changed"
            )
        plan = LMPlan(
            max_iterations=int(execution["iterations_per_job"]),
            max_evaluations=int(execution["evaluations_per_job"]),
            l2_bound=job.radius,
            linf_bound=job.radius,
            checkpoint_interval=int(execution["checkpoint_interval"]),
        )
        iterations = int(result.get("iterations", -1))
        evaluations = int(result.get("evaluations", -1))
        accepted = int(result.get("accepted_steps", -1))
        rejected = int(result.get("rejected_steps", -1))
        projection_count = int(result.get("projection_count", -1))
        if (
            result.get("schema") != LM_RESULT_SCHEMA
            or result.get("run_id") != job.run_id
            or result.get("parameters")
            != {
                "n": 6,
                "d": 3,
                "ambient_variables": 135,
                "equations": 729,
            }
            or result.get("support") != list(job.support)
            or result.get("support_size") != len(job.support)
            or result.get("gauge_chart") != chart.to_dict()
            or result.get("plan") != plan.to_dict()
            or result.get("target")
            != {
                "canonical_GHZ_plus_victim_amplitude": [0.0, 0.0],
                "victim_equation": N6_D3_SEED_DEFECT_EQUATION,
                "is_direct_GHZ_target": True,
            }
            or result.get("status") not in allowed_statuses
            or not 0 <= iterations <= plan.max_iterations
            or not 1 <= evaluations <= plan.max_evaluations
            or accepted < 0
            or rejected < 0
            or accepted + rejected != iterations
            or projection_count < 0
            or result.get("configured_blas_thread_limit_per_process")
            != NUMERICAL_BLAS_THREAD_LIMIT
            or result.get("trace") is not None
            or result.get("exact_solution_certified") is not False
            or result.get("claim_boundary")
            != expected_result_claim_boundary
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective numerical result metadata changed"
            )
        weight_rows = result.get("weights")
        if (
            not isinstance(weight_rows, Sequence)
            or isinstance(weight_rows, (str, bytes, bytearray))
            or tuple(row.get("index") for row in weight_rows)
            != job.support
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective numerical weight ledger changed"
            )
        dense = np.zeros(135, dtype=np.complex128)
        for weight_row in weight_rows:
            index = int(weight_row["index"])
            if (
                not isinstance(weight_row, Mapping)
                or set(weight_row) != {"index", "variable_key", "value"}
                or weight_row["variable_key"]
                != list(variable_key(6, 3, index))
            ):
                raise KrennProjectiveRepairCampaignError(
                    "projective numerical weight metadata changed"
                )
            value = complex(*weight_row["value"])
            if not math.isfinite(value.real) or not math.isfinite(
                value.imag
            ):
                raise KrennProjectiveRepairCampaignError(
                    "projective numerical weight became nonfinite"
                )
            dense[index] = value
        replayed_residual_metrics = ResidualMetrics.from_residual(
            complex_residual(dense)
        )
        replayed_weight_metrics = WeightMetrics.from_weights(
            dense,
            chart,
            l2_bound=job.radius,
            linf_bound=job.radius,
        )
        replayed_invariant_metrics = InvariantMetrics.from_weights(
            dense, job.support
        )
        bounded_dense, final_projection = project_hard_bounds(
            dense,
            chart,
            l2_bound=job.radius,
            linf_bound=job.radius,
        )
        final_projection_error = float(
            np.abs(bounded_dense - dense).max(initial=0.0)
        )
        projection_tolerance = (
            512.0
            * np.finfo(np.float64).eps
            * max(1.0, job.radius)
        )
        replayed_numerical_zero = bool(
            replayed_residual_metrics.finite
            and replayed_residual_metrics.linf is not None
            and replayed_residual_metrics.linf
            <= plan.residual_tolerance
        )
        replayed_interior = bool(
            replayed_weight_metrics.finite
            and not replayed_weight_metrics.on_hard_boundary
        )
        if (
            final_projection.linf_clipped_coordinates
            or final_projection.support_zeroed_coordinates
            or final_projection.anchor_resets
            or final_projection_error > projection_tolerance
            or replayed_residual_metrics.to_dict()
            != result.get("residual_metrics")
            or replayed_weight_metrics.to_dict()
            != result.get("weight_metrics")
            or replayed_invariant_metrics.to_dict()
            != result.get("gauge_invariant_metrics")
            or result.get("numerical_zero_under_declared_tolerance")
            is not replayed_numerical_zero
            or result.get("weights_remain_finite")
            is not replayed_weight_metrics.finite
            or result.get("interior_bounded_candidate")
            is not replayed_interior
            or victim_projective_audit(
                dense, spec
            )
            != row.get("final_projective_audit")
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective numerical result failed all-729 replay"
            )
        reconstruction, exact_report = (
            _reconstruction_record_from_weights(
                dense,
                job.support,
                residual_linf=replayed_residual_metrics.linf,
                weights_finite=replayed_weight_metrics.finite,
                trigger=float(execution["reconstruction_trigger"]),
            )
        )
        if row.get("reconstruction") != reconstruction:
            raise KrennProjectiveRepairCampaignError(
                "projective reconstruction ledger changed"
            )
        if exact_report is not None:
            exact_reports.append(exact_report)
    exact = bool(exact_reports)
    expected_boundary = {
        "bounded_support_miss_proves_nonexistence": False,
        "bounded_numerical_miss_proves_nonexistence": False,
        "numerical_zero_is_exact_witness": False,
        "n0_support_22_nonexistence_proved": False,
        "multinomial_n0_branches_excluded": False,
        "known_laurent_continuation_used": False,
        "all_729_equations_evaluated_per_job": True,
        "exact_counterexample_certified": exact,
    }
    expected_status = (
        "exact-candidate-dual-verified"
        if exact
        else "bounded-search-complete-no-exact-candidate"
    )
    if (
        payload.get("exact_candidate_dual_verified") is not exact
        or payload.get("claim_boundary") != expected_boundary
        or payload.get("status") != expected_status
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective claim boundary or exact-candidate status changed"
        )
    return dict(payload)


def _payload_exact_verifications(payload: Mapping) -> tuple[Mapping, ...]:
    result = []
    for row in payload["numerical_jobs"]:
        reconstruction = row["reconstruction"]
        if reconstruction.get("status") != "exact-dual-verification-passed":
            continue
        report = reconstruction.get("report")
        if not isinstance(report, Mapping):
            raise KrennProjectiveRepairCampaignError(
                "exact reconstruction omitted its report"
            )
        exact_attempts = [
            attempt
            for attempt in report.get("attempts", ())
            if isinstance(attempt, Mapping)
            and attempt.get("exact") is True
        ]
        if (
            not exact_attempts
            or not isinstance(
                exact_attempts[0].get("verification"), Mapping
            )
        ):
            raise KrennProjectiveRepairCampaignError(
                "exact reconstruction omitted its dual verification"
            )
        result.append(exact_attempts[0]["verification"])
    return tuple(result)


def verify_bundle(directory: Path | str) -> dict:
    """Verify exact inventory, hashes, sources, and campaign semantics."""

    root = Path(directory).resolve()
    manifest = _load_json(root / MANIFEST_FILE)
    if (
        not isinstance(manifest, Mapping)
        or set(manifest)
        != {
            "schema",
            "producer",
            "artifacts",
            "inputs",
            "scratch_data_included",
            "scratch_data_required_for_verification",
        }
        or manifest["schema"] != MANIFEST_SCHEMA
        or manifest["scratch_data_included"] is not False
        or manifest["scratch_data_required_for_verification"] is not False
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective manifest identity changed"
        )
    artifacts = manifest["artifacts"]
    if (
        not isinstance(artifacts, Sequence)
        or isinstance(artifacts, (str, bytes, bytearray))
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective artifact ledger changed"
        )
    artifact_names = tuple(record.get("path") for record in artifacts)
    bounded = tuple(sorted((CAMPAIGN_FILE, README_FILE)))
    exact = tuple(
        sorted(
            (
                CAMPAIGN_FILE,
                README_FILE,
                EXACT_WITNESS_FILE,
                EXACT_VERIFICATION_FILE,
            )
        )
    )
    if artifact_names not in {bounded, exact}:
        raise KrennProjectiveRepairCampaignError(
            "projective artifact inventory changed"
        )
    if {path.name for path in root.iterdir()} != set(
        (*artifact_names, MANIFEST_FILE)
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective bundle contains unexpected files"
        )
    if any(
        record != _file_record(root / record["path"], relative_to=root)
        for record in artifacts
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective artifact hash changed"
        )
    expected_inputs = [
        _file_record(ROOT / name, relative_to=ROOT)
        for name in REQUIRED_SOURCE_PATHS
    ]
    if (
        manifest["inputs"] != expected_inputs
        or manifest["producer"]
        != _file_record(Path(__file__), relative_to=ROOT)
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective source hash ledger changed"
        )
    payload = verify_payload(_load_json(root / CAMPAIGN_FILE))
    if (
        Path(payload["execution"]["results_directory"]).resolve()
        != root
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective bundle path differs from its execution receipt"
        )
    exact_payloads = _payload_exact_verifications(payload)
    has_exact = EXACT_VERIFICATION_FILE in artifact_names
    if has_exact:
        report = ExactVerificationReport.from_dict(
            _load_json(root / EXACT_VERIFICATION_FILE)
        )
        witness = _load_json(root / EXACT_WITNESS_FILE)
        if (
            not report.exact
            or report.witness.n != 6
            or report.witness.d != 3
            or witness != report.witness.to_dict()
            or payload["exact_candidate_dual_verified"] is not True
            or not exact_payloads
            or report.to_dict() != exact_payloads[0]
        ):
            raise KrennProjectiveRepairCampaignError(
                "projective exact witness bundle failed dual replay"
            )
    elif (
        payload["exact_candidate_dual_verified"] is not False
        or exact_payloads
    ):
        raise KrennProjectiveRepairCampaignError(
            "projective exact status lacks witness artifacts"
        )
    if (root / README_FILE).read_text(
        encoding="utf-8"
    ) != _readme_text(payload):
        raise KrennProjectiveRepairCampaignError(
            "projective README changed"
        )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded direct-GHZ n=6,d=3 projective repair "
            "campaign."
        )
    )
    parser.add_argument(
        "--scratch-directory", default=str(DEFAULT_SCRATCH)
    )
    parser.add_argument(
        "--results-directory", default=str(DEFAULT_RESULTS)
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--support-nodes", type=int, default=100_000)
    parser.add_argument(
        "--support-state-cap", type=int, default=2_000_000
    )
    parser.add_argument(
        "--support-candidate-scan", type=int, default=16
    )
    parser.add_argument(
        "--supports-per-orbit", type=int, default=1
    )
    parser.add_argument(
        "--starts-per-support", type=int, default=8
    )
    parser.add_argument(
        "--radii", type=float, nargs="+", default=(8.0, 32.0)
    )
    parser.add_argument(
        "--repair-scales",
        type=float,
        nargs="+",
        default=(0.03, 0.1, 0.3, 1.0),
    )
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--evaluations", type=int, default=2_250)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=20
    )
    parser.add_argument(
        "--master-seed", type=int, default=MASTER_SEED
    )
    parser.add_argument(
        "--reconstruction-trigger", type=float, default=1.0e-8
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    if args.verify_only:
        _scratch, results = _validate_paths(
            args.scratch_directory, args.results_directory
        )
        verified = verify_bundle(results)
        print(
            json.dumps(
                {
                    "status": verified["status"],
                    "exact_candidate_dual_verified": verified[
                        "exact_candidate_dual_verified"
                    ],
                    "directory": str(results),
                },
                sort_keys=True,
            )
        )
        return 0
    output = run_campaign(args, argv_receipt=raw_argv)
    payload = verify_bundle(output.parent)
    print(
        json.dumps(
            {
                "campaign": str(output),
                "status": payload["status"],
                "jobs_completed": payload["execution"][
                    "jobs_completed"
                ],
                "exact_candidate_dual_verified": payload[
                    "exact_candidate_dual_verified"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_RESULTS",
    "DEFAULT_SCRATCH",
    "EXISTING_SIZE22_SUPPORT_A",
    "EXISTING_SIZE22_SUPPORT_B",
    "KrennProjectiveRepairCampaignError",
    "MASTER_SEED",
    "ProjectiveBranchSpec",
    "ProjectiveNumericalJob",
    "active_victim_matchings",
    "branch_orbit_hits",
    "branch_spec",
    "deterministic_jobs",
    "existing_support_inventory",
    "projective_branch_specs",
    "run_campaign",
    "run_n0_support_preflight",
    "validate_projective_support",
    "verify_bundle",
    "verify_payload",
    "victim_projective_audit",
)
