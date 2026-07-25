r"""Exact projective preflight for the three ``n=6,d=3`` star slices.

The gauge-normalized star-pivot slices form an exhaustive affine cover of
the direct-GHZ fibre.  They do not require any individual weight to be
nonzero: only three quadratic matching sums are normalized to one.  Thus
homogenizing a pivot slice retains zero-coordinate and support strata that
would be lost by hard coordinate anchoring.

No Groebner basis or saturation is run here.  The module constructs and
replays the exact four-factor Cox-homogeneous presentation, audits the
smaller single-h alternative, and records a staged
affine-U/projective-star probe.  Generatorwise homogenization is not
asserted to be the scheme-theoretic projective closure before saturation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import combinations, permutations
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.star_pivot_affine_slices import (
    PIVOT_ORBITS,
    SparsePolynomial,
    star_pivot_affine_presentation,
)
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    variable_key,
)


N = 6
D = 3
APEX = 0
AMBIENT_WEIGHTS = 135
NONSTAR_WEIGHTS = 90
STAR_WEIGHTS_PER_COLOR = 15
OUTPUT_EQUATIONS = 729
PIVOT_EQUATIONS = 3

# Cox-coordinate order for the four-factor compactification:
# 0..134 are the original weights, followed by h_U,h_0,h_1,h_2.
H_U = 135
H_BY_COLOR = (136, 137, 138)
COX_VARIABLES = 139
BLOCK_COUNT = 4

STAR_PIVOT_COMPACTIFICATION_SCHEMA = (
    "krenn-n6-d3-star-pivot-compactification-v1"
)
STAR_PIVOT_BLOCK_PRESENTATION_SCHEMA = (
    "krenn-n6-d3-star-pivot-block-projective-presentation-v1"
)


class KrennStarPivotCompactificationError(RuntimeError):
    """A block degree, boundary stratum, or exact replay changed."""


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            set(left) == set(right)
            and all(
                _strict_json_equal(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, list):
        return (
            len(left) == len(right)
            and all(
                _strict_json_equal(a, b)
                for a, b in zip(left, right, strict=True)
            )
        )
    return left == right


def _weight_block(variable: int) -> int:
    """Return 0 for ``U`` and ``1+a`` for the apex-color block."""

    variable = int(variable)
    if not 0 <= variable < AMBIENT_WEIGHTS:
        raise KrennStarPivotCompactificationError(
            "an ambient weight index is outside 0..134"
        )
    i, j, a, b = variable_key(N, D, variable)
    if APEX not in (i, j):
        return 0
    apex_color = a if i == APEX else b
    return 1 + apex_color


def _weight_block_census() -> tuple[int, int, int, int]:
    counts = [0] * BLOCK_COUNT
    for variable in range(AMBIENT_WEIGHTS):
        counts[_weight_block(variable)] += 1
    result = tuple(counts)
    if result != (
        NONSTAR_WEIGHTS,
        STAR_WEIGHTS_PER_COLOR,
        STAR_WEIGHTS_PER_COLOR,
        STAR_WEIGHTS_PER_COLOR,
    ):
        raise KrennStarPivotCompactificationError(
            "the U/Y star-block census changed"
        )
    return result


def _block_degree(monomial: Sequence[int]) -> tuple[int, int, int, int]:
    degree = [0] * BLOCK_COUNT
    for variable in monomial:
        degree[_weight_block(variable)] += 1
    return tuple(degree)


def _target_multidegree(
    generator_index: int,
) -> tuple[int, int, int, int]:
    generator_index = int(generator_index)
    if 0 <= generator_index < OUTPUT_EQUATIONS:
        apex_color = coloring_from_index(
            N, D, generator_index
        )[APEX]
        degree = [2, 0, 0, 0]
        degree[1 + apex_color] = 1
        return tuple(degree)
    if OUTPUT_EQUATIONS <= generator_index < (
        OUTPUT_EQUATIONS + PIVOT_EQUATIONS
    ):
        return (2, 0, 0, 0)
    raise KrennStarPivotCompactificationError(
        "a compactified generator index is outside 0..731"
    )


def _homogenized_monomial(
    monomial: Sequence[int],
    target_degree: Sequence[int],
) -> tuple[int, ...]:
    source_degree = _block_degree(monomial)
    target_degree = tuple(map(int, target_degree))
    if (
        len(target_degree) != BLOCK_COUNT
        or any(
            source > target
            for source, target in zip(
                source_degree, target_degree, strict=True
            )
        )
    ):
        raise KrennStarPivotCompactificationError(
            "a term exceeds its declared star-block multidegree"
        )
    result = list(map(int, monomial))
    result.extend([H_U] * (target_degree[0] - source_degree[0]))
    for color in range(D):
        result.extend(
            [H_BY_COLOR[color]]
            * (
                target_degree[1 + color]
                - source_degree[1 + color]
            )
        )
    return tuple(sorted(result))


@dataclass(frozen=True)
class BlockHomogeneousPolynomial:
    """One collected polynomial in the 139-coordinate Cox ring."""

    terms: tuple[tuple[int, tuple[int, ...]], ...]
    multidegree: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        normalized = tuple(
            (int(coefficient), tuple(map(int, monomial)))
            for coefficient, monomial in self.terms
        )
        if (
            len(self.multidegree) != BLOCK_COUNT
            or any(value < 0 for value in self.multidegree)
            or normalized
            != tuple(sorted(
                normalized,
                key=lambda term: (len(term[1]), term[1]),
            ))
            or any(not coefficient for coefficient, _ in normalized)
            or len({monomial for _coefficient, monomial in normalized})
            != len(normalized)
            or any(
                monomial != tuple(sorted(monomial))
                or any(
                    variable < 0 or variable >= COX_VARIABLES
                    for variable in monomial
                )
                for _coefficient, monomial in normalized
            )
        ):
            raise KrennStarPivotCompactificationError(
                "a block-homogeneous polynomial is not canonical"
            )
        for _coefficient, monomial in normalized:
            degree = [0] * BLOCK_COUNT
            for variable in monomial:
                if variable < AMBIENT_WEIGHTS:
                    degree[_weight_block(variable)] += 1
                elif variable == H_U:
                    degree[0] += 1
                else:
                    degree[1 + H_BY_COLOR.index(variable)] += 1
            if tuple(degree) != self.multidegree:
                raise KrennStarPivotCompactificationError(
                    "a Cox monomial has the wrong star-block multidegree"
                )
        object.__setattr__(self, "terms", normalized)

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def ordinary_degree(self) -> int:
        return sum(self.multidegree)

    def dehomogenized(self) -> SparsePolynomial:
        """Set all four homogenizers to one."""

        coefficients: dict[tuple[int, ...], int] = {}
        for coefficient, monomial in self.terms:
            residual = tuple(
                variable
                for variable in monomial
                if variable < AMBIENT_WEIGHTS
            )
            coefficients[residual] = (
                coefficients.get(residual, 0) + coefficient
            )
        return SparsePolynomial(tuple(
            (coefficient, monomial)
            for monomial, coefficient in sorted(
                coefficients.items(),
                key=lambda item: (len(item[0]), item[0]),
            )
            if coefficient
        ))

    def term_count_after_zeroing(
        self, homogenizers: Sequence[int]
    ) -> int:
        zero = frozenset(map(int, homogenizers))
        if not zero.issubset({H_U, *H_BY_COLOR}):
            raise KrennStarPivotCompactificationError(
                "a boundary specialization names a non-homogenizer"
            )
        return sum(
            not any(variable in zero for variable in monomial)
            for _coefficient, monomial in self.terms
        )


@dataclass(frozen=True)
class StarPivotBlockPresentation:
    """One four-block Cox-homogeneous presentation of a retained slice."""

    orbit_index: int
    generators: tuple[BlockHomogeneousPolynomial, ...]
    schema: str = STAR_PIVOT_BLOCK_PRESENTATION_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != STAR_PIVOT_BLOCK_PRESENTATION_SCHEMA
            or self.orbit_index not in range(3)
            or len(self.generators)
            != OUTPUT_EQUATIONS + PIVOT_EQUATIONS
            or self.term_count != 10_950
            or self.maximum_ordinary_degree != 3
            or self.multidegree_histogram
            != {
                (2, 0, 0, 0): 3,
                (2, 1, 0, 0): 243,
                (2, 0, 1, 0): 243,
                (2, 0, 0, 1): 243,
            }
        ):
            raise KrennStarPivotCompactificationError(
                "a star-block projective census changed"
            )
        affine = star_pivot_affine_presentation(
            self.orbit_index, "retained"
        )
        if tuple(
            generator.dehomogenized()
            for generator in self.generators
        ) != affine.generators:
            raise KrennStarPivotCompactificationError(
                "dehomogenization did not recover the retained slice"
            )

    @property
    def term_count(self) -> int:
        return sum(generator.term_count for generator in self.generators)

    @property
    def maximum_ordinary_degree(self) -> int:
        return max(
            generator.ordinary_degree for generator in self.generators
        )

    @property
    def multidegree_histogram(
        self,
    ) -> dict[tuple[int, int, int, int], int]:
        result: dict[tuple[int, int, int, int], int] = {}
        for generator in self.generators:
            result[generator.multidegree] = (
                result.get(generator.multidegree, 0) + 1
            )
        return dict(sorted(result.items()))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.orbit_index}|"
                f"{_weight_block_census()}\n"
            ).encode("ascii")
        )
        for generator_index, generator in enumerate(self.generators):
            digest.update(
                (
                    f"{generator_index}|{generator.multidegree}|"
                ).encode("ascii")
            )
            for coefficient, monomial in generator.terms:
                digest.update(
                    (
                        f"{coefficient}:"
                        f"{','.join(map(str, monomial))};"
                    ).encode("ascii")
                )
            digest.update(b"\n")
        return digest.hexdigest()


@lru_cache(maxsize=3)
def star_pivot_block_presentation(
    orbit_index: int,
) -> StarPivotBlockPresentation:
    """Construct and replay one exact four-block homogenization."""

    if isinstance(orbit_index, bool):
        raise KrennStarPivotCompactificationError(
            "a pivot orbit index must be an integer"
        )
    orbit_index = int(orbit_index)
    if orbit_index not in range(3):
        raise KrennStarPivotCompactificationError(
            "a pivot orbit index is outside 0..2"
        )
    affine = star_pivot_affine_presentation(
        orbit_index, "retained"
    )
    generators = []
    for generator_index, polynomial in enumerate(affine.generators):
        degree = _target_multidegree(generator_index)
        terms = tuple(sorted(
            (
                (
                    coefficient,
                    _homogenized_monomial(monomial, degree),
                )
                for coefficient, monomial in polynomial.terms
            ),
            key=lambda term: (len(term[1]), term[1]),
        ))
        generators.append(BlockHomogeneousPolynomial(
            terms=terms,
            multidegree=degree,
        ))
    return StarPivotBlockPresentation(
        orbit_index=orbit_index,
        generators=tuple(generators),
    )


def _color_stabilizer(orbit_index: int) -> tuple[tuple[int, ...], ...]:
    partners = PIVOT_ORBITS[int(orbit_index)].partner_by_color
    result = []
    for permutation in permutations(range(D)):
        if all(
            (partners[left] == partners[right])
            == (
                partners[permutation[left]]
                == partners[permutation[right]]
            )
            for left in range(D)
            for right in range(D)
        ):
            result.append(tuple(permutation))
    expected = 2 if int(orbit_index) == 1 else 6
    if len(result) != expected:
        raise KrennStarPivotCompactificationError(
            "the pivot representative color stabilizer changed"
        )
    return tuple(result)


def _act_on_boundary(
    boundary: tuple[bool, tuple[int, ...]],
    permutation: Sequence[int],
) -> tuple[bool, tuple[int, ...]]:
    h_u_zero, zero_colors = boundary
    return (
        h_u_zero,
        tuple(sorted(permutation[color] for color in zero_colors)),
    )


def boundary_orbits(orbit_index: int) -> tuple[dict, ...]:
    """Return the 15 nonfinite four-factor strata modulo symmetry."""

    orbit_index = int(orbit_index)
    group = _color_stabilizer(orbit_index)
    boundaries = tuple(
        (h_u_zero, zero_colors)
        for h_u_zero in (False, True)
        for size in range(D + 1)
        for zero_colors in combinations(range(D), size)
        if h_u_zero or zero_colors
    )
    unseen = set(boundaries)
    rows = []
    while unseen:
        representative = min(unseen)
        orbit = {
            _act_on_boundary(representative, permutation)
            for permutation in group
        }
        unseen.difference_update(orbit)
        h_u_zero, zero_colors = representative
        zero_homogenizers = (
            ((H_U,) if h_u_zero else ())
            + tuple(H_BY_COLOR[color] for color in zero_colors)
        )
        presentation = star_pivot_block_presentation(orbit_index)
        rows.append({
            "h_U_zero": h_u_zero,
            "zero_star_homogenizer_colors": list(zero_colors),
            "orbit_size": len(orbit),
            "surviving_target_RHS_colors": (
                []
                if h_u_zero
                else [
                    color
                    for color in range(D)
                    if color not in zero_colors
                ]
            ),
            "pivot_RHS_survives": not h_u_zero,
            "specialized_nonzero_term_count": sum(
                generator.term_count_after_zeroing(
                    zero_homogenizers
                )
                for generator in presentation.generators
            ),
            "all_729_U2Y_terms_survive": True,
            "whole_weight_block_forced_nonzero_by_Cox_irrelevant_ideal":
                (
                    ["U"] if h_u_zero else []
                )
                + [
                    f"Y{color}" for color in zero_colors
                ],
        })
    result = tuple(rows)
    expected = 11 if orbit_index == 1 else 7
    if (
        len(result) != expected
        or sum(row["orbit_size"] for row in result) != 15
    ):
        raise KrennStarPivotCompactificationError(
            "the compactification boundary-orbit census changed"
        )
    return result


def star_only_boundary_orbits(orbit_index: int) -> tuple[dict, ...]:
    """Return the seven affine-U/projective-star boundary strata."""

    orbit_index = int(orbit_index)
    group = _color_stabilizer(orbit_index)
    boundaries = tuple(
        zero_colors
        for size in range(1, D + 1)
        for zero_colors in combinations(range(D), size)
    )
    unseen = set(boundaries)
    rows = []
    while unseen:
        representative = min(unseen)
        orbit = {
            tuple(sorted(permutation[color] for color in representative))
            for permutation in group
        }
        unseen.difference_update(orbit)
        rows.append({
            "zero_star_homogenizer_colors": list(representative),
            "orbit_size": len(orbit),
            "rank_condition": (
                "rank(A(U)) <= 14 because each zero h_a forces "
                "a nonzero Y^a in ker(A(U))"
            ),
            "pivot_rank_floor": 3,
            "individual_U_or_Y_coordinates_may_still_be_zero": True,
        })
    result = tuple(rows)
    expected = 5 if orbit_index == 1 else 3
    if (
        len(result) != expected
        or sum(row["orbit_size"] for row in result) != 7
    ):
        raise KrennStarPivotCompactificationError(
            "the star-only boundary-orbit census changed"
        )
    return result


def _degree_histogram(polynomials: Sequence[SparsePolynomial]) -> dict:
    result: dict[int, int] = {}
    for polynomial in polynomials:
        result[polynomial.degree] = result.get(polynomial.degree, 0) + 1
    return {
        str(degree): count
        for degree, count in sorted(result.items())
    }


def _leading_term_count(polynomials: Sequence[SparsePolynomial]) -> int:
    """Count terms surviving ordinary single-h specialization ``h=0``."""

    return sum(
        sum(
            len(monomial) == polynomial.degree
            for _coefficient, monomial in polynomial.terms
        )
        for polynomial in polynomials
    )


@lru_cache(maxsize=1)
def star_pivot_compactification_audit() -> dict:
    """Return exact counts, branch semantics, and a staged first probe."""

    orbit_rows = []
    for orbit_index, orbit in enumerate(PIVOT_ORBITS):
        retained = star_pivot_affine_presentation(
            orbit_index, "retained"
        )
        eliminated = star_pivot_affine_presentation(
            orbit_index, "eliminated"
        )
        block = star_pivot_block_presentation(orbit_index)
        orbit_rows.append({
            "orbit": orbit.to_dict(),
            "ordinary_single_h_retained": {
                "homogeneous_coordinates": 136,
                "projective_ambient": "P^135",
                "formula_output": (
                    "sum_matching W_i*W_j*W_k "
                    "- delta_target*h^3"
                ),
                "formula_pivot": "P_v(a^4)-h^2",
                "generators": len(retained.generators),
                "terms": retained.term_count,
                "degree_histogram": _degree_histogram(
                    retained.generators
                ),
                "maximum_degree": retained.maximum_degree,
                "finite_open": "h != 0",
                "boundary": "h = 0",
                "boundary_generators": (
                    "all 729 homogeneous output cubics and the three "
                    "selected homogeneous pivot quadrics"
                ),
                "boundary_terms": _leading_term_count(
                    retained.generators
                ),
                "whole_U_zero_boundary_piece_present": True,
            },
            "ordinary_single_h_eliminated": {
                "homogeneous_coordinates": 127,
                "projective_ambient": "P^126",
                "generators": len(eliminated.generators),
                "terms": eliminated.term_count,
                "degree_histogram": _degree_histogram(
                    eliminated.generators
                ),
                "maximum_degree": eliminated.maximum_degree,
                "finite_open": "h != 0",
                "boundary": "h = 0",
                "boundary_leading_terms": _leading_term_count(
                    eliminated.generators
                ),
                "polynomial_elimination_boundary_may_have_extraneous_components":
                    True,
            },
            "star_block_multi_projective": {
                "Cox_coordinates": COX_VARIABLES,
                "projective_factors": [
                    "P^90_[U:h_U]",
                    "P^15_[Y0:h_0]",
                    "P^15_[Y1:h_1]",
                    "P^15_[Y2:h_2]",
                ],
                "formula_output": (
                    "sum_(v,b) Y^a_(v,b)*P_(v,b)(U) "
                    "- delta_(r,a^5)*h_U^2*h_a"
                ),
                "formula_pivot": "P_(v_a,a)(U)-h_U^2",
                "multi_projective_dimension": 135,
                "Cox_scaling_rank": 4,
                "weight_block_sizes": list(_weight_block_census()),
                "generators": len(block.generators),
                "terms": block.term_count,
                "maximum_ordinary_degree":
                    block.maximum_ordinary_degree,
                "multidegree_histogram": {
                    ",".join(map(str, degree)): count
                    for degree, count
                    in block.multidegree_histogram.items()
                },
                "finite_open": "h_U*h_0*h_1*h_2 != 0",
                "boundary_divisor_subsets": 15,
                "boundary_orbits_under_representative_stabilizer":
                    list(boundary_orbits(orbit_index)),
                "color_stabilizer_order": len(
                    _color_stabilizer(orbit_index)
                ),
                "presentation_sha256": block.fingerprint(),
            },
            "affine_U_projective_star_probe": {
                "Cox_coordinates": 138,
                "ambient": "A^90_U x (P^15)^3",
                "geometric_dimension": 135,
                "formula_output": (
                    "sum_(v,b) Y^a_(v,b)*P_(v,b)(U) "
                    "- delta_(r,a^5)*h_a"
                ),
                "formula_pivot": "P_(v_a,a)(U)-1",
                "generators": 732,
                "terms": 10_950,
                "maximum_UY_ordinary_degree": 3,
                "finite_open": "h_0*h_1*h_2 != 0",
                "boundary_divisor_subsets": 7,
                "boundary_orbits_under_representative_stabilizer":
                    list(star_only_boundary_orbits(orbit_index)),
                "advantage": (
                    "P=1 remains valid at the boundary, so rank(A)>=3 "
                    "and the enormous h_U=0, P=0 base locus is absent"
                ),
                "limitation": (
                    "U remains affine; this is a decisive saturation "
                    "formulation, not a proper compactification of U"
                ),
            },
        })

    return {
        "schema": STAR_PIVOT_COMPACTIFICATION_SCHEMA,
        "parameters": {
            "n": N,
            "d": D,
            "apex": APEX,
            "pivot_orbit_representatives": 3,
            "affine_weight_variables": AMBIENT_WEIGHTS,
        },
        "exact_cover": {
            "nonzero_target_minor_choices": 125,
            "S5_x_S3_orbits": 3,
            "requires_individual_weight_nonzero": False,
            "includes_zero_coordinate_and_support_strata": True,
            "three_quadratic_pivot_sums_normalized_to_one": True,
        },
        "orbit_preflights": orbit_rows,
        "recommended_first_exact_probe": {
            "formulation": "affine-U/projective-star retained slice",
            "pivot_orbit": "all-distinct (1,2,3)",
            "why_this_orbit": (
                "it has the full S3 consistency check and is the pivot "
                "type containing the natural nine-slot configuration"
            ),
            "ring": "Q[U_0..U_89,Y^0,Y^1,Y^2,h_0,h_1,h_2]",
            "ideal": (
                "J_star=<A(U)Y^a-delta*h_a, "
                "P_(v_a,a)(U)-1>"
            ),
            "sequential_exact_test": [
                "K0=J_star",
                "K1=K0:h_0^infinity",
                "K2=K1:h_1^infinity",
                "K3=K2:h_2^infinity",
            ],
            "conclusion_if_K3_is_unit": (
                "the all-distinct pivot orbit contains no finite witness"
            ),
            "conclusion_if_K3_is_proper": (
                "a finite witness exists over the algebraic closure in "
                "the all-distinct pivot orbit"
            ),
            "boundary_preflight_before_full_colon": (
                "let B_Y0=<all 15 coordinates of Y0> and compute "
                "B0=Sat_B_Y0((J_star+(h_0)):"
                "(h_1*h_2)^infinity) "
                "(equivalently, cover the 15 charts Y0_j != 0); only "
                "after this Cox-irrelevant saturation is the elimination "
                "the exact incidence rank(A)<=14 with "
                "e_11111,e_22222 in im(A)"
            ),
            "boundary_Cox_irrelevant_block":
                "B_Y0=<all 15 coordinates of Y0>",
            "boundary_Cox_irrelevant_saturation_required": True,
            "raw_boundary_colon_without_irrelevant_saturation_is_decisive":
                False,
            "required_exactness": (
                "characteristic zero colon/primary computation, including "
                "the indicated Cox-irrelevant saturation on a boundary; "
                "modular runs are reconnaissance only"
            ),
            "grading": (
                "use the residual Z^12 character blocks together with "
                "the three Cox degrees; do not impose weight anchors"
            ),
            "stage_gate": (
                "do not launch all three global saturations unless the "
                "Cox-irrelevant-saturated one-divisor boundary/first-colon "
                "computation shows bounded block growth"
            ),
            "existing_timing_warning": (
                "the affine retained p31 slimgb probes all timed out "
                "after 600 seconds, so there is no basis for claiming "
                "this is one hour from a decision"
            ),
        },
        "decisive_exact_test": {
            "four_factor_Cox_ring": (
                "Q[U,Y0,Y1,Y2,h_U,h_0,h_1,h_2]"
            ),
            "homogenizer_product": "H=h_U*h_0*h_1*h_2",
            "finite_ideal": "J_fin = J : H^infinity",
            "unit_J_fin": "no finite point in this pivot orbit",
            "proper_J_fin": (
                "a finite direct-GHZ point exists over the algebraic "
                "closure in this pivot orbit"
            ),
            "global_nonexistence": (
                "J_fin is unit for all three pivot orbit representatives"
            ),
            "counterexample": (
                "J_fin is proper for at least one representative; "
                "extract and exactly verify a closed point"
            ),
            "boundary_point_alone_is_a_decision": False,
            "finite_open_H_nonzero_avoids_all_Cox_irrelevant_blocks": True,
            "colon_chain_or_primary_decomposition_must_be_exact": True,
            "modular_or_timeout_result_is_a_decision": False,
        },
        "grading_and_symmetry": {
            "direct_GHZ_color_gauge_character_rank": 15,
            "primitive_pivot_characters_quotiented": 3,
            "residual_character_rank": 12,
            "single_h_combined_grading_rank": 13,
            "four_factor_combined_grading_rank": 16,
            "affine_U_projective_star_combined_grading_rank": 15,
            "rank_9_seed_chart_grading_is_the_same_grading": False,
            "rank_9_seed_chart_route_requires_eight_seed_orbits": True,
            "rank_9_route_variables_generators_terms_max_degree": [
                130, 729, 10_938, 4
            ],
            "finite_color_symmetry": (
                "the three Y blocks are permuted; the all-same and "
                "all-distinct pivot types retain S3, while the "
                "exactly-two-same representative retains S2"
            ),
        },
        "claim_boundary": {
            "Cox_homogeneous_presentation_constructed_exactly": True,
            "scheme_theoretic_projective_closure_computed": False,
            "compactification_constructed_exactly": False,
            "Cox_irrelevant_saturation_computed": False,
            "large_CAS_launched": False,
            "saturation_computed": False,
            "finite_counterexample_found": False,
            "global_nonexistence_proved": False,
            "border_membership_changed": False,
        },
    }


def verify_star_pivot_compactification_audit(
    payload: Mapping,
) -> dict:
    """Regenerate the audit and reject altered counts or claims."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennStarPivotCompactificationError(
            "the compactification audit is not strict JSON"
        ) from error
    expected = star_pivot_compactification_audit()
    if not _strict_json_equal(normalized, expected):
        raise KrennStarPivotCompactificationError(
            "the compactification audit failed exact replay"
        )
    return normalized


__all__ = [
    "BlockHomogeneousPolynomial",
    "COX_VARIABLES",
    "H_BY_COLOR",
    "H_U",
    "KrennStarPivotCompactificationError",
    "StarPivotBlockPresentation",
    "boundary_orbits",
    "star_only_boundary_orbits",
    "star_pivot_block_presentation",
    "star_pivot_compactification_audit",
    "verify_star_pivot_compactification_audit",
]
