r"""Exact repair-monomial atlas for the natural localized chart.

On the normalized natural seed, equation 70 is

    1 + (six quadratic repair monomials)
      + (eight cubic repair monomials) = 0.

Thus at least one repair monomial is nonzero.  Its factors are all nonzero,
and the residual nine-dimensional seed-preserving torus can set independent
factor characters to one.  The strict seed stabilizer has four orbits on the
14 monomials, so four exact gauge-localized affine ideals cover the complete
natural chart.

This is a finite structural cover, not a support restriction: every other
weight remains independent and unrestricted.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    NATURAL_DEFECT_EQUATION,
    NATURAL_ORBIT_INDEX,
    natural_defect_polynomial,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    KrennLocalizedChartError,
    SparseChartPolynomial,
    normalized_seed_chart,
    validate_singular_characteristic,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    ordered_seed_stabilizer,
    residual_torus_characters,
)


MONOMIAL_ATLAS_SCHEMA = (
    "krenn-n6-d3-natural-repair-monomial-atlas-v1"
)
MONOMIAL_CHART_SCHEMA = (
    "krenn-n6-d3-natural-repair-monomial-chart-v1"
)
EXPECTED_REPAIR_MONOMIAL_ORBITS = (
    ((11, 65), (19, 73), (88, 133)),
    ((29, 47), (37, 55), (106, 113)),
    (
        (11, 55, 133),
        (11, 73, 113),
        (19, 47, 133),
        (19, 65, 106),
        (29, 73, 88),
        (37, 65, 88),
    ),
    ((29, 55, 106), (37, 47, 113)),
)
REPAIR_MONOMIAL_REPRESENTATIVES = (
    (11, 65),
    (29, 47),
    (11, 55, 133),
    (29, 55, 106),
)


class KrennMonomialAtlasError(RuntimeError):
    """A repair-monomial cover, gauge, or chart replay failed."""


def natural_repair_monomial_orbits() -> tuple[
    tuple[tuple[int, ...], ...], ...
]:
    """Return the four strict-stabilizer orbits on the 14 repair terms."""

    chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
    symmetry = ordered_seed_stabilizer(NATURAL_ORBIT_INDEX)
    local_by_ambient = {
        ambient: local
        for local, ambient in enumerate(chart.remaining_weight_indices)
    }
    terms = tuple(
        monomial
        for monomial in natural_defect_polynomial()
        if monomial
    )
    term_set = set(terms)
    unseen = set(terms)
    orbits = []
    while unseen:
        monomial = min(unseen)
        local_monomial = tuple(
            local_by_ambient[ambient] for ambient in monomial
        )
        orbit = {
            tuple(sorted(
                chart.ambient_weight_index(permutation[variable])
                for variable in local_monomial
            ))
            for permutation in symmetry.chart_variable_permutations
        }
        if not orbit.issubset(term_set):
            raise KrennMonomialAtlasError(
                "the strict stabilizer moved a repair term outside equation 70"
            )
        orbits.append(tuple(sorted(orbit)))
        unseen.difference_update(orbit)
    result = tuple(sorted(
        orbits, key=lambda orbit: (len(orbit[0]), orbit)
    ))
    if result != EXPECTED_REPAIR_MONOMIAL_ORBITS:
        raise KrennMonomialAtlasError(
            "repair-monomial stabilizer orbits changed"
        )
    return result


def _rank_over_q(matrix: Sequence[Sequence[int]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    rank = 0
    width = len(rows[0])
    for column in range(width):
        pivot = next(
            (
                row for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        value = rows[rank][column]
        rows[rank] = [entry / value for entry in rows[rank]]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(
                    rows[row], rows[rank], strict=True
                )
            ]
        rank += 1
    return rank


def _add(
    polynomial: dict[tuple[int, ...], int],
    monomial: Sequence[int],
    coefficient: int,
) -> None:
    monomial = tuple(sorted(map(int, monomial)))
    total = polynomial.get(monomial, 0) + int(coefficient)
    if total:
        polynomial[monomial] = total
    else:
        polynomial.pop(monomial, None)


@dataclass(frozen=True)
class RepairMonomialChart:
    """One residual-gauge-normalized repair-monomial chart."""

    representative_ambient_monomial: tuple[int, ...]
    fixed_original_chart_variables: tuple[int, ...]
    localized_original_chart_variable: int
    retained_original_chart_variables: tuple[int, ...]
    branch_inverse_variable: int
    generator_labels: tuple[tuple[str, int], ...]
    generators: tuple[SparseChartPolynomial, ...]
    schema: str = MONOMIAL_CHART_SCHEMA

    def __post_init__(self) -> None:
        expected_variables = (
            129 - len(self.fixed_original_chart_variables) + 1
        )
        if (
            self.schema != MONOMIAL_CHART_SCHEMA
            or self.representative_ambient_monomial
            not in REPAIR_MONOMIAL_REPRESENTATIVES
            or len(self.fixed_original_chart_variables)
            != len(self.representative_ambient_monomial) - 1
            or self.localized_original_chart_variable
            in self.fixed_original_chart_variables
            or set(self.fixed_original_chart_variables).intersection(
                self.retained_original_chart_variables
            )
            or tuple(sorted((
                *self.fixed_original_chart_variables,
                *self.retained_original_chart_variables,
            ))) != tuple(range(129))
            or self.branch_inverse_variable
            != len(self.retained_original_chart_variables)
            or expected_variables
            != len(self.retained_original_chart_variables) + 1
            or len(self.generator_labels) != 730
            or len(self.generators) != 730
        ):
            raise KrennMonomialAtlasError(
                "a repair-monomial chart failed metadata replay"
            )

    @property
    def variable_count(self) -> int:
        return len(self.retained_original_chart_variables) + 1

    @property
    def term_count(self) -> int:
        return sum(polynomial.term_count for polynomial in self.generators)

    @property
    def maximum_degree(self) -> int:
        return max(polynomial.degree for polynomial in self.generators)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.representative_ambient_monomial}|"
                f"{self.fixed_original_chart_variables}|"
                f"{self.localized_original_chart_variable}|"
                f"{self.retained_original_chart_variables}|"
                f"{self.branch_inverse_variable}\n"
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
        chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
        return {
            "schema": self.schema,
            "representative_ambient_monomial": list(
                self.representative_ambient_monomial
            ),
            "fixed_to_one_ambient_weights": [
                chart.ambient_weight_index(variable)
                for variable in self.fixed_original_chart_variables
            ],
            "localized_ambient_weight":
                chart.ambient_weight_index(
                    self.localized_original_chart_variable
                ),
            "retained_original_chart_variables": len(
                self.retained_original_chart_variables
            ),
            "branch_inverse_variable": self.branch_inverse_variable,
            "total_variables": self.variable_count,
            "total_generators": len(self.generators),
            "sparse_terms": self.term_count,
            "maximum_degree": self.maximum_degree,
            "sha256": self.fingerprint(),
        }


def repair_monomial_chart(
    representative_ambient_monomial: Sequence[int],
) -> RepairMonomialChart:
    """Construct one of the four exact natural repair charts."""

    representative = tuple(sorted(map(
        int, representative_ambient_monomial
    )))
    if representative not in REPAIR_MONOMIAL_REPRESENTATIVES:
        raise KrennMonomialAtlasError(
            "repair monomial is not a strict-orbit representative"
        )
    chart = normalized_seed_chart(NATURAL_ORBIT_INDEX)
    local_monomial = tuple(
        chart.remaining_weight_indices.index(ambient)
        for ambient in representative
    )
    fixed = local_monomial[:-1]
    localized = local_monomial[-1]
    characters = residual_torus_characters(NATURAL_ORBIT_INDEX)
    if _rank_over_q([characters[variable] for variable in fixed]) != len(
        fixed
    ):
        raise KrennMonomialAtlasError(
            "the chosen repair factors do not have independent characters"
        )
    retained = tuple(
        variable for variable in range(129)
        if variable not in set(fixed)
    )
    remap = {
        original: reduced for reduced, original in enumerate(retained)
    }
    labels = []
    generators = []
    for label, polynomial in zip(
        chart.generator_labels, chart.generators, strict=True
    ):
        reduced = {}
        for coefficient, monomial in polynomial.terms:
            _add(
                reduced,
                (
                    remap[variable]
                    for variable in monomial
                    if variable not in fixed
                ),
                coefficient,
            )
        labels.append(label)
        generators.append(SparseChartPolynomial.from_mapping(reduced))
    inverse = len(retained)
    localizer = {
        (): -1,
        tuple(sorted((remap[localized], inverse))): 1,
    }
    labels.append(("repair-localizer", representative[-1]))
    generators.append(SparseChartPolynomial.from_mapping(localizer))
    return RepairMonomialChart(
        representative_ambient_monomial=representative,
        fixed_original_chart_variables=fixed,
        localized_original_chart_variable=localized,
        retained_original_chart_variables=retained,
        branch_inverse_variable=inverse,
        generator_labels=tuple(labels),
        generators=tuple(generators),
    )


def natural_repair_monomial_atlas_audit() -> dict:
    """Return the exact four-chart repair-monomial cover."""

    orbits = natural_repair_monomial_orbits()
    charts = tuple(
        repair_monomial_chart(representative)
        for representative in REPAIR_MONOMIAL_REPRESENTATIVES
    )
    return json.loads(json.dumps({
        "schema": MONOMIAL_ATLAS_SCHEMA,
        "natural_seed": [0, 4, 8],
        "defect_equation": NATURAL_DEFECT_EQUATION,
        "repair_terms": {
            "quadratic": 6,
            "cubic": 8,
            "total": 14,
            "nonzero_cover_reason": (
                "their sum is -1 on the normalized defect hypersurface"
            ),
        },
        "strict_stabilizer": {
            "order": ordered_seed_stabilizer(
                NATURAL_ORBIT_INDEX
            ).order,
            "repair_monomial_orbits": [
                [list(monomial) for monomial in orbit]
                for orbit in orbits
            ],
            "representatives": [
                list(monomial)
                for monomial in REPAIR_MONOMIAL_REPRESENTATIVES
            ],
            "symmetry_related_weights_equated": False,
        },
        "residual_torus_gauge": {
            "rank": 9,
            "fixed_factor_characters_are_independent": True,
            "surjective_over_algebraic_closure": True,
            "all_unfixed_weights_remain_independent": True,
        },
        "charts": [chart.summary() for chart in charts],
        "decision_rule": {
            "all_four_verified_unit": "excludes the natural chart",
            "one_verified_proper": (
                "proves a finite point over algebraic closure of Q"
            ),
        },
        "claim_boundary": {
            "natural_chart_decided": False,
            "full_eight_chart_cover_decided": False,
            "unit_certificates_emitted": 0,
            "proper_ideal_certificates_emitted": 0,
        },
    }))


def _singular_polynomial_text(
    polynomial: SparseChartPolynomial,
    characteristic: int,
) -> str:
    pieces = []
    for raw_coefficient, monomial in polynomial.terms:
        coefficient = (
            raw_coefficient % characteristic
            if characteristic else raw_coefficient
        )
        factors = "*".join(f"x{variable}" for variable in monomial)
        if not factors:
            piece = str(coefficient)
        elif coefficient == 1:
            piece = factors
        elif coefficient == -1:
            piece = f"-{factors}"
        else:
            piece = f"{coefficient}*{factors}"
        pieces.append(piece)
    if characteristic:
        return "+".join(pieces)
    result = pieces[0]
    for piece in pieces[1:]:
        result += piece if piece.startswith("-") else f"+{piece}"
    return result


def singular_repair_monomial_script(
    representative_ambient_monomial: Sequence[int],
    *,
    characteristic: int = 31,
    algorithm: str = "std",
) -> str:
    """Export one exact repair-monomial chart to Singular."""

    try:
        characteristic = validate_singular_characteristic(characteristic)
    except KrennLocalizedChartError as error:
        raise KrennMonomialAtlasError(
            "Singular characteristic must be zero or prime"
        ) from error
    if algorithm not in ("std", "slimgb"):
        raise KrennMonomialAtlasError(
            "Singular algorithm must be std or slimgb"
        )
    chart = repair_monomial_chart(
        representative_ambient_monomial
    )
    names = ",".join(
        f"x{index}" for index in range(chart.variable_count)
    )
    generators = ",\n".join(
        _singular_polynomial_text(polynomial, characteristic)
        for polynomial in chart.generators
    )
    representative_text = "_".join(
        map(str, chart.representative_ambient_monomial)
    )
    return "\n".join((
        "// Exact natural repair-monomial Krenn chart.",
        f"// schema={MONOMIAL_CHART_SCHEMA}",
        f"// representative={representative_text}",
        f"ring r={characteristic},({names}),dp;",
        f"ideal I={generators};",
        'print("KRENN_REPAIR_CHART_PARSE_OK");',
        f'print("representative={representative_text}");',
        f'print("characteristic={characteristic}");',
        f'print("algorithm={algorithm}");',
        'print("variables="+string(nvars(basering)));',
        'print("generators="+string(size(I)));',
        "int started=timer;",
        f"ideal G={algorithm}(I);",
        "int elapsed=timer-started;",
        "poly unit_remainder=reduce(1,G);",
        "int is_unit=0;",
        "if (unit_remainder==0) { is_unit=1; }",
        'print("KRENN_REPAIR_CHART_GROEBNER_DONE");',
        'print("timer_ticks="+string(elapsed));',
        'print("basis_size="+string(size(G)));',
        'print("unit_ideal="+string(is_unit));',
        "exit;",
        "",
    ))
