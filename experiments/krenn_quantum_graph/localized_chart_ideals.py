r"""Exact localized chart ideals for the ``n=6,d=3`` Krenn problem.

This module replaces an unbounded search along one Laurent branch by a
finite algebraic cover of every possible affine GHZ witness.

Write ``F_c`` for the perfect-matching output with coloring ``c``.  A direct
GHZ witness exists exactly when all mixed ``F_c`` vanish and the three pure
outputs are nonzero: the pure values can subsequently be scaled to one by
the endpoint-color torus.

Every nonzero pure output contains a nonzero matching monomial.  Choosing
one such monomial for each color gives 15^3 ordered seed charts.  The
existing exact ``S_6 x S_3`` census reduces these to eight representatives,
without ever equating symmetry-related weights.  On a seed chart the nine
chosen diagonal weights are nonzero.  Because each chosen matching
partitions the six vertices, endpoint-color scaling sets all nine weights
to one.

For a normalized seed ``r``, let ``G_(r,c)`` be ``F_c`` after that
substitution and let ``A_(r,a)`` be the pure output for color ``a``.  The
ordinary affine chart ideal is

    <G_(r,c) : c mixed> + <u_a A_(r,a) - 1 : a=0,1,2>.

It has 129 variables, 729 generators, 10,938 sparse terms, and maximum
degree four.  A unit ideal excludes its chart.  A proper ideal proves that
the chart has a point over the algebraic closure of Q.  Modular Gröbner
results and bounded-degree certificate misses are deliberately not promoted
to either conclusion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    generate_sparse_system,
    perfect_matchings,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.localized_chart_independent import (
    independent_chart_fingerprint,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    EXPECTED_REPRESENTATIVES_AND_SIZES,
    ternary_seed_orbit_census,
)


N = 6
D = 3
AMBIENT_WEIGHT_COUNT = 135
PURE_EQUATIONS = (0, 364, 728)
MIXED_EQUATIONS = tuple(
    equation for equation in range(D**N)
    if equation not in PURE_EQUATIONS
)
CHART_WEIGHT_COUNT = 126
INVERSE_VARIABLES = (126, 127, 128)
CHART_VARIABLE_COUNT = 129
CHART_GENERATOR_COUNT = 729
CHART_TERM_COUNT = 10_938

LOCALIZED_CHART_SCHEMA = "krenn-n6-d3-localized-chart-ideal-v1"
CHART_COVER_SCHEMA = "krenn-n6-d3-localized-chart-cover-v1"
CHART_GERM_SCHEMA = "krenn-n6-d3-localized-chart-germ-separation-v1"
SINGULAR_EXPORT_SCHEMA = "krenn-n6-d3-localized-chart-singular-export-v1"

EXPECTED_SEED_SUPPORTS = (
    (0, 4, 8, 81, 85, 89, 126, 130, 134),
    (0, 4, 8, 81, 85, 98, 125, 126, 130),
    (0, 4, 17, 71, 81, 85, 125, 126, 130),
    (0, 4, 8, 81, 94, 107, 116, 121, 126),
    (0, 4, 17, 62, 81, 94, 121, 126, 134),
    (0, 4, 17, 80, 81, 94, 116, 121, 126),
    (0, 13, 26, 67, 80, 81, 98, 121, 126),
    (0, 13, 44, 62, 67, 81, 98, 121, 126),
)
EXPECTED_DEFECT_COUNTS = (24, 12, 6, 6, 5, 2, 1, 3)
EXPECTED_DISJOINT_CORE_COUNTS = (9, 5, 3, 3, 4, 2, 1, 3)
EXPECTED_CORE_UNION_SIZES = (108, 84, 48, 60, 52, 24, 12, 36)
EXPECTED_EQUATION_ORBIT_COUNTS = (14, 88, 76, 30, 198, 192, 74, 29)


class KrennLocalizedChartError(RuntimeError):
    """A chart construction, replay, export, or claim boundary failed."""


def strict_json_equal(left, right) -> bool:
    """Compare JSON values without Python's bool/int/float aliases."""

    try:
        options = {
            "allow_nan": False,
            "ensure_ascii": True,
            "separators": (",", ":"),
            "sort_keys": True,
        }
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError):
        return False


def validate_singular_characteristic(value: int) -> int:
    """Return zero or a prime characteristic; reject composite rings."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise KrennLocalizedChartError(
            "Singular characteristic must be zero or prime"
        )
    characteristic = int(value)
    if characteristic == 0:
        return 0
    if characteristic < 2:
        raise KrennLocalizedChartError(
            "Singular characteristic must be zero or prime"
        )
    divisor = 2
    while divisor * divisor <= characteristic:
        if characteristic % divisor == 0:
            raise KrennLocalizedChartError(
                "Singular characteristic must be zero or prime"
            )
        divisor += 1 if divisor == 2 else 2
    return characteristic


def _canonical_monomial(monomial: Sequence[int]) -> tuple[int, ...]:
    try:
        result = tuple(sorted(map(int, monomial)))
    except (TypeError, ValueError) as error:
        raise KrennLocalizedChartError(
            "a chart monomial must be an integer sequence"
        ) from error
    if any(index < 0 or index >= CHART_VARIABLE_COUNT for index in result):
        raise KrennLocalizedChartError(
            "a chart monomial contains an out-of-range variable"
        )
    return result


@dataclass(frozen=True)
class SparseChartPolynomial:
    """A deterministic integer sparse polynomial."""

    terms: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        normalized = tuple(
            (int(coefficient), _canonical_monomial(monomial))
            for coefficient, monomial in self.terms
        )
        expected = tuple(
            sorted(normalized, key=lambda term: (len(term[1]), term[1]))
        )
        if (
            normalized != expected
            or any(coefficient == 0 for coefficient, _ in normalized)
            or len({monomial for _, monomial in normalized})
            != len(normalized)
        ):
            raise KrennLocalizedChartError(
                "chart polynomial terms are not canonical and unique"
            )
        object.__setattr__(self, "terms", normalized)

    @classmethod
    def from_mapping(
        cls, coefficients: Mapping[Sequence[int], int]
    ) -> "SparseChartPolynomial":
        combined: dict[tuple[int, ...], int] = {}
        for raw_monomial, raw_coefficient in coefficients.items():
            monomial = _canonical_monomial(raw_monomial)
            coefficient = int(raw_coefficient)
            combined[monomial] = combined.get(monomial, 0) + coefficient
        return cls(tuple(
            (coefficient, monomial)
            for monomial, coefficient in sorted(
                combined.items(),
                key=lambda item: (len(item[0]), item[0]),
            )
            if coefficient
        ))

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def degree(self) -> int:
        return max((len(monomial) for _, monomial in self.terms), default=-1)

    def coefficient(self, monomial: Sequence[int]) -> int:
        target = _canonical_monomial(monomial)
        return next(
            (
                coefficient
                for coefficient, candidate in self.terms
                if candidate == target
            ),
            0,
        )

    def evaluate(self, values: Sequence) -> object:
        if len(values) != CHART_VARIABLE_COUNT:
            raise KrennLocalizedChartError(
                "chart evaluation needs exactly 129 values"
            )
        total = 0
        for coefficient, monomial in self.terms:
            term = coefficient
            for variable in monomial:
                term *= values[variable]
            total += term
        return total

    def to_dict(self) -> dict:
        return {
            "terms": [
                {
                    "coefficient": coefficient,
                    "variables": list(monomial),
                }
                for coefficient, monomial in self.terms
            ]
        }


@dataclass(frozen=True)
class NormalizedSeedChart:
    """One exact all-nine-normalized affine chart."""

    orbit_index: int
    seed: tuple[int, int, int]
    ordered_seed_orbit_size: int
    fixed_weight_indices: tuple[int, ...]
    remaining_weight_indices: tuple[int, ...]
    generator_labels: tuple[tuple[str, int], ...]
    generators: tuple[SparseChartPolynomial, ...]
    schema: str = LOCALIZED_CHART_SCHEMA

    def __post_init__(self) -> None:
        orbit_index = int(self.orbit_index)
        if not 0 <= orbit_index < len(EXPECTED_REPRESENTATIVES_AND_SIZES):
            raise KrennLocalizedChartError("chart orbit index is invalid")
        expected_seed, expected_size = EXPECTED_REPRESENTATIVES_AND_SIZES[
            orbit_index
        ]
        if (
            self.schema != LOCALIZED_CHART_SCHEMA
            or tuple(self.seed) != expected_seed
            or int(self.ordered_seed_orbit_size) != expected_size
            or tuple(self.fixed_weight_indices)
            != EXPECTED_SEED_SUPPORTS[orbit_index]
            or len(self.remaining_weight_indices) != CHART_WEIGHT_COUNT
            or set(self.fixed_weight_indices).intersection(
                self.remaining_weight_indices
            )
            or tuple(sorted(
                (*self.fixed_weight_indices, *self.remaining_weight_indices)
            )) != tuple(range(AMBIENT_WEIGHT_COUNT))
            or len(self.generator_labels) != CHART_GENERATOR_COUNT
            or len(self.generators) != CHART_GENERATOR_COUNT
        ):
            raise KrennLocalizedChartError(
                "normalized chart metadata failed exact replay"
            )
        expected_labels = (
            tuple(("mixed", equation) for equation in MIXED_EQUATIONS)
            + tuple(("pure-localizer", color) for color in range(D))
        )
        if tuple(self.generator_labels) != expected_labels:
            raise KrennLocalizedChartError(
                "normalized chart generator labels changed"
            )
        if (
            sum(polynomial.term_count for polynomial in self.generators)
            != CHART_TERM_COUNT
            or max(polynomial.degree for polynomial in self.generators) != 4
        ):
            raise KrennLocalizedChartError(
                "normalized chart sparse census changed"
            )

    @property
    def term_count(self) -> int:
        return sum(polynomial.term_count for polynomial in self.generators)

    @property
    def maximum_degree(self) -> int:
        return max(polynomial.degree for polynomial in self.generators)

    @property
    def mixed_generators(self) -> tuple[SparseChartPolynomial, ...]:
        return self.generators[:len(MIXED_EQUATIONS)]

    @property
    def pure_localizers(self) -> tuple[SparseChartPolynomial, ...]:
        return self.generators[len(MIXED_EQUATIONS):]

    def ambient_weight_index(self, chart_weight_index: int) -> int:
        chart_weight_index = int(chart_weight_index)
        if not 0 <= chart_weight_index < CHART_WEIGHT_COUNT:
            raise KrennLocalizedChartError(
                "local chart weight index is outside range(126)"
            )
        return self.remaining_weight_indices[chart_weight_index]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.orbit_index}|{self.seed}|"
                f"{self.ordered_seed_orbit_size}|"
                f"{self.fixed_weight_indices}|"
                f"{self.remaining_weight_indices}\n"
            ).encode("ascii")
        )
        for label, polynomial in zip(
            self.generator_labels, self.generators, strict=True
        ):
            digest.update(f"{label[0]}:{label[1]}|".encode("ascii"))
            for coefficient, monomial in polynomial.terms:
                digest.update(
                    f"{coefficient}:{','.join(map(str, monomial))};".encode(
                        "ascii"
                    )
                )
            digest.update(b"\n")
        return digest.hexdigest()

    def summary(self) -> dict:
        return {
            "schema": self.schema,
            "orbit_index": self.orbit_index,
            "representative_matching_indices": list(self.seed),
            "ordered_seed_orbit_size": self.ordered_seed_orbit_size,
            "fixed_ambient_weight_indices": list(
                self.fixed_weight_indices
            ),
            "remaining_weight_variables": len(
                self.remaining_weight_indices
            ),
            "inverse_amplitude_variables": D,
            "total_variables": CHART_VARIABLE_COUNT,
            "mixed_generators": len(MIXED_EQUATIONS),
            "pure_inverse_generators": D,
            "total_generators": len(self.generators),
            "sparse_terms": self.term_count,
            "maximum_degree": self.maximum_degree,
            "equation_orbits_under_ordered_seed_stabilizer":
                EXPECTED_EQUATION_ORBIT_COUNTS[self.orbit_index],
            "sha256": self.fingerprint(),
        }


def _add_coefficient(
    coefficients: dict[tuple[int, ...], int],
    monomial: Sequence[int],
    coefficient: int,
) -> None:
    canonical = _canonical_monomial(monomial)
    total = coefficients.get(canonical, 0) + int(coefficient)
    if total:
        coefficients[canonical] = total
    else:
        coefficients.pop(canonical, None)


@lru_cache(maxsize=8)
def normalized_seed_chart(orbit_index: int) -> NormalizedSeedChart:
    """Construct one representative chart from the canonical enumerator."""

    orbit_index = int(orbit_index)
    if not 0 <= orbit_index < len(EXPECTED_REPRESENTATIVES_AND_SIZES):
        raise KrennLocalizedChartError("chart orbit index is invalid")
    seed, orbit_size = EXPECTED_REPRESENTATIVES_AND_SIZES[orbit_index]
    fixed = tuple(sorted(
        variable_index(N, D, i, j, color, color)
        for color, matching_index in enumerate(seed)
        for i, j in perfect_matchings(N)[matching_index]
    ))
    if fixed != EXPECTED_SEED_SUPPORTS[orbit_index]:
        raise KrennLocalizedChartError(
            "the representative seed support changed"
        )
    fixed_set = set(fixed)
    remaining = tuple(
        index for index in range(AMBIENT_WEIGHT_COUNT)
        if index not in fixed_set
    )
    remap = {
        ambient: local for local, ambient in enumerate(remaining)
    }

    system = generate_sparse_system(N, D)
    reduced_outputs = []
    for equation in range(D**N):
        coefficients: dict[tuple[int, ...], int] = {}
        for monomial in system.equation_monomials(equation):
            _add_coefficient(
                coefficients,
                tuple(
                    remap[variable]
                    for variable in monomial
                    if variable not in fixed_set
                ),
                1,
            )
        polynomial = SparseChartPolynomial.from_mapping(coefficients)
        if (
            polynomial.term_count != 15
            or any(coefficient != 1 for coefficient, _ in polynomial.terms)
        ):
            raise KrennLocalizedChartError(
                "seed substitution caused an unexpected monomial collision"
            )
        reduced_outputs.append(polynomial)

    labels = list(("mixed", equation) for equation in MIXED_EQUATIONS)
    generators = [reduced_outputs[equation] for equation in MIXED_EQUATIONS]
    for color, equation in enumerate(PURE_EQUATIONS):
        coefficients = {}
        for coefficient, monomial in reduced_outputs[equation].terms:
            _add_coefficient(
                coefficients,
                (*monomial, INVERSE_VARIABLES[color]),
                coefficient,
            )
        _add_coefficient(coefficients, (), -1)
        localizer = SparseChartPolynomial.from_mapping(coefficients)
        if localizer.term_count != 16:
            raise KrennLocalizedChartError(
                "a pure inverse localizer changed sparse size"
            )
        labels.append(("pure-localizer", color))
        generators.append(localizer)

    return NormalizedSeedChart(
        orbit_index=orbit_index,
        seed=seed,
        ordered_seed_orbit_size=orbit_size,
        fixed_weight_indices=fixed,
        remaining_weight_indices=remaining,
        generator_labels=tuple(labels),
        generators=tuple(generators),
    )


def normalized_seed_charts() -> tuple[NormalizedSeedChart, ...]:
    """Return the complete eight-representative exact chart cover."""

    return tuple(
        normalized_seed_chart(index)
        for index in range(len(EXPECTED_REPRESENTATIVES_AND_SIZES))
    )


def seed_gauge_split_audit(orbit_index: int) -> dict:
    """Replay a split 9x9 identity minor in the full endpoint torus.

    Columns are the 18 endpoint-color factors ``(vertex,color)``.  For each
    selected matching edge choose its smaller endpoint as a pivot and set
    the other endpoint factor to one.  Matching edges partition the vertices
    separately in each color, so the selected minor is literally identity.
    """

    chart = normalized_seed_chart(int(orbit_index))
    rows = []
    pivots = []
    for ambient in chart.fixed_weight_indices:
        i, j, a, b = variable_key(N, D, ambient)
        if a != b:
            raise KrennLocalizedChartError(
                "a seed gauge row is not a diagonal weight"
            )
        row = [0] * (N * D)
        row[i * D + a] += 1
        row[j * D + b] += 1
        rows.append(tuple(row))
        pivots.append(i * D + a)
    minor = tuple(
        tuple(row[column] for column in pivots)
        for row in rows
    )
    identity = tuple(
        tuple(int(row == column) for column in range(9))
        for row in range(9)
    )
    if len(set(pivots)) != 9 or minor != identity:
        raise KrennLocalizedChartError(
            "the nine seed characters lost their split identity minor"
        )
    return {
        "orbit_index": chart.orbit_index,
        "seed_exponent_rows": [list(row) for row in rows],
        "pivot_endpoint_color_columns": pivots,
        "identity_minor": [list(row) for row in minor],
        "split_rank": 9,
        "smith_diagonal": [1] * 9,
    }


def _build_chart_cover_audit() -> dict:
    charts = normalized_seed_charts()
    primary_fingerprints = tuple(chart.fingerprint() for chart in charts)
    independent_fingerprints = tuple(
        independent_chart_fingerprint(index)
        for index in range(len(charts))
    )
    if independent_fingerprints != primary_fingerprints:
        raise KrennLocalizedChartError(
            "the independent chart reconstruction changed"
        )
    orbit_sum = sum(chart.ordered_seed_orbit_size for chart in charts)
    orbit_census = ternary_seed_orbit_census()
    census_rows = tuple(
        (
            tuple(row["representative_matching_indices"]),
            int(row["ordered_seed_orbit_size"]),
        )
        for row in orbit_census["orbits"]
    )
    if census_rows != EXPECTED_REPRESENTATIVES_AND_SIZES:
        raise KrennLocalizedChartError(
            "the independently enumerated seed-orbit cover changed"
        )
    return {
        "schema": CHART_COVER_SCHEMA,
        "parameters": {"n": N, "d": D},
        "ambient_equations": D**N,
        "ambient_weight_variables": variable_count(N, D),
        "open_locus": {
            "mixed_outputs_zero": len(MIXED_EQUATIONS),
            "pure_outputs_nonzero": D,
            "equivalent_to_direct_GHZ_membership": True,
            "normalization_back_to_direct_GHZ": (
                "scale vertex 0/color a by u_a"
            ),
        },
        "cover": {
            "perfect_matchings_per_pure_output": 15,
            "ordered_seed_charts_before_symmetry": 15**3,
            "symmetry_group": "S_6 x S_3",
            "representative_charts": len(charts),
            "sum_of_ordered_orbit_sizes": orbit_sum,
            "symmetry_related_weights_equated": False,
            "exact_seed_orbit_census": orbit_census,
            "independent_chart_reconstruction": {
                "replayed": True,
                "fingerprints": list(independent_fingerprints),
            },
        },
        "gauge_slice": {
            "group": "(G_m)^18 endpoint-color torus",
            "selected_seed_weights": 9,
            "all_selected_weights_fixed_to_one": True,
            "reason": (
                "each color matching partitions the six endpoints into "
                "three disjoint edges"
            ),
            "preserves_mixed_zero_locus": True,
            "preserves_literal_pure_coefficients_one": False,
            "preserves_pure_nonvanishing": True,
            "split_seed_character_audits": [
                seed_gauge_split_audit(index)
                for index in range(len(charts))
            ],
        },
        "normalized_ideal": {
            "remaining_weight_variables": CHART_WEIGHT_COUNT,
            "inverse_amplitude_variables": D,
            "total_variables": CHART_VARIABLE_COUNT,
            "mixed_generators": len(MIXED_EQUATIONS),
            "pure_inverse_generators": D,
            "total_generators": CHART_GENERATOR_COUNT,
            "sparse_terms_per_chart": CHART_TERM_COUNT,
            "maximum_degree": 4,
        },
        "charts": [chart.summary() for chart in charts],
        "exact_checks": {
            "three_nonzero_pure_outputs_supply_three_seed_monomials": True,
            "ordered_seed_census_3375": 15**3 == 3_375,
            "eight_orbits_cover_all_ordered_seeds": (
                len(charts) == 8 and orbit_sum == 3_375
            ),
            "nine_seed_weights_are_distinct_in_every_chart": all(
                len(set(chart.fixed_weight_indices)) == 9
                for chart in charts
            ),
            "all_seed_gauge_maps_have_split_rank_nine": all(
                seed_gauge_split_audit(index)["split_rank"] == 9
                for index in range(len(charts))
            ),
            "all_chart_sparse_censuses_replayed": all(
                chart.term_count == CHART_TERM_COUNT
                and chart.maximum_degree == 4
                for chart in charts
            ),
            "all_independent_chart_fingerprints_match": (
                independent_fingerprints == primary_fingerprints
            ),
        },
        "decision_rule": {
            "all_eight_verified_unit_ideals": (
                "proves no finite n=6,d=3 GHZ witness"
            ),
            "one_verified_proper_ideal": (
                "proves a point exists over the algebraic closure of Q"
            ),
            "one_exact_reconstructed_point": (
                "is a finite exact counterexample witness"
            ),
        },
        "claim_boundary": {
            "finite_affine_GHZ_membership_status": "undecided",
            "unit_ideal_certificates_emitted": 0,
            "exact_chart_points_emitted": 0,
            "modular_results_are_proofs": False,
            "bounded_degree_misses_are_proofs": False,
            "numerical_results_are_proofs": False,
        },
    }


def localized_chart_cover_audit() -> dict:
    """Return a fresh JSON-ready exact chart-cover audit."""

    return json.loads(json.dumps(_build_chart_cover_audit()))


def verify_localized_chart_cover_audit(payload: Mapping) -> dict:
    """Rebuild the chart cover and reject any changed claim or datum."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennLocalizedChartError(
            "localized chart audit is not strict JSON"
        ) from error
    expected = _build_chart_cover_audit()
    if not strict_json_equal(normalized, expected):
        raise KrennLocalizedChartError(
            "localized chart audit failed exact replay"
        )
    return normalized


def _maximum_disjoint_core_indices(
    cores: Sequence[frozenset[int]],
) -> tuple[int, ...]:
    """Return the lexicographically least maximum disjoint subfamily."""

    best: tuple[int, ...] = ()

    def recurse(
        start: int,
        chosen: tuple[int, ...],
        used: frozenset[int],
    ) -> None:
        nonlocal best
        if len(chosen) + len(cores) - start < len(best):
            return
        if (
            len(chosen) > len(best)
            or (len(chosen) == len(best) and chosen < best)
        ):
            best = chosen
        for index in range(start, len(cores)):
            if not used.intersection(cores[index]):
                recurse(
                    index + 1,
                    (*chosen, index),
                    used.union(cores[index]),
                )

    recurse(0, (), frozenset())
    return best


def _build_chart_germ_audit() -> dict:
    rows = []
    for chart in normalized_seed_charts():
        defects = []
        for equation, polynomial in zip(
            MIXED_EQUATIONS, chart.mixed_generators, strict=True
        ):
            if polynomial.coefficient(()) != 1:
                continue
            degrees = [
                len(monomial)
                for _coefficient, monomial in polynomial.terms
                if monomial
            ]
            quadratic = [
                monomial
                for coefficient, monomial in polynomial.terms
                if coefficient == 1 and len(monomial) == 2
            ]
            cubic = [
                monomial
                for coefficient, monomial in polynomial.terms
                if coefficient == 1 and len(monomial) == 3
            ]
            core = frozenset(
                variable for monomial in quadratic for variable in monomial
            )
            incidences = {
                variable: sum(variable in monomial for monomial in cubic)
                for variable in core
            }
            if (
                sorted(degrees) != [2] * 6 + [3] * 8
                or len(core) != 12
                or len(set(quadratic)) != 6
                or any(
                    sum(variable in monomial for monomial in quadratic) != 1
                    for variable in core
                )
                or set(incidences.values()) != {2}
                or any(set(monomial).difference(core) for monomial in cubic)
            ):
                raise KrennLocalizedChartError(
                    "a normalized seed defect lost its universal 6/8 core"
                )
            defects.append({
                "equation": equation,
                "coloring": list(coloring_from_index(N, D, equation)),
                "quadratic_repair_terms": 6,
                "cubic_repair_terms": 8,
                "repair_core_local_variables": sorted(core),
                "repair_core_ambient_weights": sorted(
                    chart.ambient_weight_index(variable)
                    for variable in core
                ),
            })

        cores = tuple(
            frozenset(defect["repair_core_local_variables"])
            for defect in defects
        )
        if any(
            len(left.intersection(right)) not in (0, 4)
            for left_index, left in enumerate(cores)
            for right in cores[left_index + 1:]
        ):
            raise KrennLocalizedChartError(
                "defect core intersections are no longer zero or four"
            )
        selected = _maximum_disjoint_core_indices(cores)
        union = frozenset().union(*cores) if cores else frozenset()
        if (
            len(defects) != EXPECTED_DEFECT_COUNTS[chart.orbit_index]
            or len(selected)
            != EXPECTED_DISJOINT_CORE_COUNTS[chart.orbit_index]
            or len(union) != EXPECTED_CORE_UNION_SIZES[chart.orbit_index]
        ):
            raise KrennLocalizedChartError(
                "the exact chart defect census changed"
            )
        rows.append({
            "orbit_index": chart.orbit_index,
            "seed": list(chart.seed),
            "defect_count": len(defects),
            "defects": defects,
            "repair_core_union_size": len(union),
            "maximum_pairwise_disjoint_defect_cores": len(selected),
            "exhibited_disjoint_defect_equations": [
                defects[index]["equation"] for index in selected
            ],
        })

    lower = Fraction(338_825, 1_000_000)
    upper = Fraction(338_826, 1_000_000)
    polynomial = lambda value: 8 * value**3 + 6 * value**2 - 1
    if not polynomial(lower) < 0 < polynomial(upper):
        raise KrennLocalizedChartError(
            "the rational isolating interval for rho changed"
        )
    return {
        "schema": CHART_GERM_SCHEMA,
        "universal_defect_equation": {
            "constant_terms": 1,
            "quadratic_repair_terms": 6,
            "cubic_repair_terms": 8,
            "coordinate_sup_norm_inequality": "1 <= 6 R^2 + 8 R^3",
            "positive_root_polynomial": "8*r^3 + 6*r^2 - 1",
            "positive_root_isolating_interval": {
                "lower": [lower.numerator, lower.denominator],
                "upper": [upper.numerator, upper.denominator],
            },
            "approximate_positive_root": "0.33882535",
            "slice_dependent_coordinate_statement": True,
            "gauge_invariant_statement": (
                "some alternative matching monomial has absolute value "
                "at least 1/14"
            ),
        },
        "charts": rows,
        "exact_checks": {
            "all_defects_have_six_quadratics_and_eight_cubics": True,
            "quadratic_cores_are_six_disjoint_pairs": True,
            "each_core_variable_occurs_in_two_cubics": True,
            "pairwise_core_intersections_are_zero_or_four": True,
            "rho_isolated_exactly": True,
        },
        "claim_boundary": {
            "excludes_only_a_small_neighborhood_of_each_seed_germ": True,
            "excludes_entire_chart": False,
            "proves_nonexistence": False,
        },
    }


def localized_chart_germ_audit() -> dict:
    """Return exact finite-separation structure around all eight seed germs."""

    return json.loads(json.dumps(_build_chart_germ_audit()))


def _term_text(
    coefficient: int,
    monomial: Sequence[int],
    characteristic: int,
) -> str:
    coefficient = int(coefficient)
    characteristic = int(characteristic)
    if characteristic:
        coefficient %= characteristic
    factors = [f"x{variable}" for variable in monomial]
    product_text = "*".join(factors)
    if not factors:
        return str(coefficient)
    if coefficient == 1:
        return product_text
    if coefficient == -1:
        return f"-{product_text}"
    return f"{coefficient}*{product_text}"


def _polynomial_text(
    polynomial: SparseChartPolynomial,
    characteristic: int,
) -> str:
    pieces = [
        _term_text(coefficient, monomial, characteristic)
        for coefficient, monomial in polynomial.terms
        if not characteristic or coefficient % characteristic
    ]
    if not pieces:
        return "0"
    if characteristic:
        return "+".join(pieces)
    result = pieces[0]
    for piece in pieces[1:]:
        result += piece if piece.startswith("-") else f"+{piece}"
    return result


def singular_chart_script(
    orbit_index: int,
    *,
    characteristic: int = 31,
    algorithm: str = "std",
) -> str:
    """Return a deterministic unshifted-inverse Singular script.

    The script reports a unit ideal only when reduction of 1 by the computed
    standard basis is exactly zero in the requested coefficient field.
    Characteristic-zero output is potentially conclusive; positive
    characteristic output is reconnaissance only.
    """

    characteristic = validate_singular_characteristic(characteristic)
    if algorithm not in ("std", "slimgb"):
        raise KrennLocalizedChartError(
            "Singular algorithm must be std or slimgb"
        )
    chart = normalized_seed_chart(orbit_index)
    variable_names = ",".join(
        f"x{index}" for index in range(CHART_VARIABLE_COUNT)
    )
    generators = ",\n".join(
        _polynomial_text(polynomial, characteristic)
        for polynomial in chart.generators
    )
    command = f"{algorithm}(I)"
    return "\n".join((
        "// Exact sparse Krenn n=6,d=3 localized chart.",
        f"// schema={SINGULAR_EXPORT_SCHEMA}",
        (
            f"// orbit_index={chart.orbit_index}; seed={chart.seed}; "
            f"ordered_orbit_size={chart.ordered_seed_orbit_size}"
        ),
        "// x0..x125 are remaining weights; x126..x128 are u0..u2.",
        "// The inverse variables are unshifted: u_a*A_a-1.",
        f"ring r={characteristic},({variable_names}),dp;",
        f"ideal I={generators};",
        'print("KRENN_CHART_PARSE_OK");',
        f'print("schema={SINGULAR_EXPORT_SCHEMA}");',
        f'print("orbit_index={chart.orbit_index}");',
        f'print("characteristic={characteristic}");',
        f'print("algorithm={algorithm}");',
        'print("variables="+string(nvars(basering)));',
        'print("generators="+string(size(I)));',
        "int started=timer;",
        f"ideal G={command};",
        "int elapsed=timer-started;",
        "poly unit_remainder=reduce(1,G);",
        "int is_unit=0;",
        "if (unit_remainder==0) { is_unit=1; }",
        'print("KRENN_CHART_GROEBNER_DONE");',
        'print("timer_ticks="+string(elapsed));',
        'print("basis_size="+string(size(G)));',
        'print("unit_ideal="+string(is_unit));',
        "exit;",
        "",
    ))


def write_singular_chart_export(
    output_directory: Path,
    *,
    characteristics: Sequence[int] = (31,),
    algorithm: str = "std",
) -> dict:
    """Write deterministic scripts and a scratch manifest."""

    normalized_characteristics = tuple(
        validate_singular_characteristic(value)
        for value in characteristics
    )
    if not normalized_characteristics:
        raise KrennLocalizedChartError(
            "Singular export needs at least one characteristic"
        )
    if algorithm not in ("std", "slimgb"):
        raise KrennLocalizedChartError(
            "Singular algorithm must be std or slimgb"
        )
    if len(set(normalized_characteristics)) != len(
        normalized_characteristics
    ):
        raise KrennLocalizedChartError(
            "Singular export characteristics must be unique"
        )
    output_directory = Path(output_directory)
    expected_inventory = {"export_manifest.json"}
    for orbit_index in range(
        len(EXPECTED_REPRESENTATIVES_AND_SIZES)
    ):
        for characteristic in normalized_characteristics:
            stem = (
                f"chart{orbit_index}_char{characteristic}_{algorithm}"
            )
            expected_inventory.update((
                f"{stem}.sing",
                f"{stem}.variables.tsv",
            ))
    if output_directory.exists():
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise KrennLocalizedChartError(
                "Singular export path must be a real directory"
            )
    else:
        output_directory.mkdir(parents=True)
    existing = tuple(output_directory.iterdir())
    if (
        any(path.name not in expected_inventory for path in existing)
        or any(not path.is_file() or path.is_symlink() for path in existing)
    ):
        raise KrennLocalizedChartError(
            "Singular export directory contains an undeclared entry"
        )
    files = []
    for orbit_index in range(len(EXPECTED_REPRESENTATIVES_AND_SIZES)):
        chart = normalized_seed_chart(orbit_index)
        for characteristic in normalized_characteristics:
            script = singular_chart_script(
                orbit_index,
                characteristic=characteristic,
                algorithm=algorithm,
            )
            stem = (
                f"chart{orbit_index}_char{characteristic}_{algorithm}"
            )
            script_path = output_directory / f"{stem}.sing"
            variable_path = output_directory / f"{stem}.variables.tsv"
            script_path.write_text(script, encoding="ascii", newline="\n")
            variable_path.write_text(
                "chart_weight_index\tambient_weight_index\n"
                + "".join(
                    f"{local}\t{ambient}\n"
                    for local, ambient in enumerate(
                        chart.remaining_weight_indices
                    )
                ),
                encoding="ascii",
                newline="\n",
            )
            files.extend((
                {
                    "path": script_path.name,
                    "sha256": hashlib.sha256(
                        script_path.read_bytes()
                    ).hexdigest(),
                },
                {
                    "path": variable_path.name,
                    "sha256": hashlib.sha256(
                        variable_path.read_bytes()
                    ).hexdigest(),
                },
            ))
    manifest = {
        "schema": SINGULAR_EXPORT_SCHEMA,
        "algorithm": algorithm,
        "characteristics": list(normalized_characteristics),
        "charts": [
            normalized_seed_chart(index).summary()
            for index in range(len(EXPECTED_REPRESENTATIVES_AND_SIZES))
        ],
        "files": sorted(files, key=lambda row: row["path"]),
        "claim_boundary": {
            "positive_characteristic_is_reconnaissance_only": True,
            "unit_ideal_claimed": False,
            "proper_ideal_claimed": False,
        },
    }
    manifest_path = output_directory / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _parse_characteristics(raw: str) -> tuple[int, ...]:
    try:
        result = tuple(
            validate_singular_characteristic(int(value))
            for value in raw.split(",")
        )
    except (ValueError, KrennLocalizedChartError) as error:
        raise argparse.ArgumentTypeError(
            "characteristics must be unique comma-separated primes or zero"
        ) from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "characteristics must be unique comma-separated primes or zero"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact localized n=6,d=3 Krenn chart data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument(
        "--kind", choices=("cover", "germ"), default="cover"
    )
    export_parser = subparsers.add_parser("export-singular")
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument(
        "--characteristics",
        type=_parse_characteristics,
        default=(31,),
    )
    export_parser.add_argument(
        "--algorithm", choices=("std", "slimgb"), default="std"
    )
    args = parser.parse_args()
    if args.command == "audit":
        payload = (
            localized_chart_cover_audit()
            if args.kind == "cover"
            else localized_chart_germ_audit()
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    manifest = write_singular_chart_export(
        args.output,
        characteristics=args.characteristics,
        algorithm=args.algorithm,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
