"""Forensic audit of the ``n=6,d=3`` nine-slot near-miss.

The natural seed selects one diagonal perfect matching for each color.  Its
sole residual is not a global invariant of the perfect-matching tensor map:
it is a unique extra matching admitted by that sparse support.  Exhausting the
eight ``S_6 x S_3`` support orbits nevertheless proves a clean restricted
no-go theorem:

    one nonzero diagonal perfect matching per color can never realize GHZ_3.

The report emitted here records the exact residual polynomial, all orbit
representative residuals, the weighted-hafnian shadow, a small fixed-support
Nullstellensatz identity, and negative audits for linear, sign,
representation, and ``n=4`` contraction explanations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import combinations, permutations
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from experiments.krenn_quantum_graph.border_image import (
    certify_n6_d3_laurent_border,
)
from experiments.krenn_quantum_graph.border_valuation import (
    certify_natural_border_valuation_obstruction,
)
from experiments.krenn_quantum_graph.cyclotomic_shadow import (
    certify_cyclotomic_shadow_witness,
)
from experiments.krenn_quantum_graph.fixtures import fixture_n4_d3
from experiments.krenn_quantum_graph.formal_lift import (
    certify_n6_d3_formal_lift,
)
from experiments.krenn_quantum_graph.hafnian_identities import (
    certify_equal_g_identity,
    certify_hafnian_contraction,
    generate_equal_g_shadow_system,
    occupations,
    shadow_structure_summary,
    singular_groebner_script,
)
from experiments.krenn_quantum_graph.n6_deformation import (
    certify_n6_natural_deformation,
)
from experiments.krenn_quantum_graph.system import (
    SparsePolynomialSystem,
    canonical_edges,
    coloring_from_index,
    coloring_index,
    generate_sparse_system,
    perfect_matchings,
    variable_count,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.targets import (
    ColoringTarget,
    canonical_ghz_target,
)
from experiments.krenn_quantum_graph.shadow_slices import (
    canonical_ghz_shadow_star_solution,
    generate_n6_d3_shadow_star_slice,
)
from experiments.krenn_quantum_graph.support_extension import (
    certify_support_extension,
)
from experiments.krenn_quantum_graph.tensor_invariants import (
    universal_image_dimension_bounds,
)
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    N6_D3_SEED_FACTORS,
    evaluate_ternary_candidate,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    canonical_seed_representative,
    ternary_seed_orbit_census,
    transport_matching,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "results/krenn_quantum_graph/n6_d3_seed_defect_mining"
)

DEFECT_REPORT_SCHEMA = "krenn-n6-d3-seed-defect-mining-report-v4"
DEFECT_CERTIFICATE_SCHEMA = (
    "krenn-n6-d3-seed-defect-mining-certificate-v4"
)
DEFECT_MANIFEST_SCHEMA = (
    "krenn-n6-d3-seed-defect-mining-manifest-v4"
)

REPORT_FILE = "defect_report.json"
CERTIFICATE_FILE = "certificate.json"
MANIFEST_FILE = "manifest.json"
GROEBNER_D3_FILE = "equal_g_n6_d3_F31.sing"
GROEBNER_D4_FILE = "equal_g_n6_d4_F31.sing"
ALL_FILES = (
    CERTIFICATE_FILE,
    REPORT_FILE,
    GROEBNER_D3_FILE,
    GROEBNER_D4_FILE,
    MANIFEST_FILE,
)

NATURAL_REPRESENTATIVE = (0, 4, 8)
SECOND_DISJOINT_REPRESENTATIVE = (0, 4, 13)
SHARED_EDGE_VICTIM_EQUATION = 40

SOURCE_INPUTS = (
    "experiments/krenn_quantum_graph/border_image.py",
    "experiments/krenn_quantum_graph/border_valuation.py",
    "experiments/krenn_quantum_graph/cyclotomic_shadow.py",
    "experiments/krenn_quantum_graph/defect_mining.py",
    "experiments/krenn_quantum_graph/fixtures.py",
    "experiments/krenn_quantum_graph/formal_lift.py",
    "experiments/krenn_quantum_graph/hafnian_identities.py",
    "experiments/krenn_quantum_graph/n6_deformation.py",
    "experiments/krenn_quantum_graph/shadow_slices.py",
    "experiments/krenn_quantum_graph/support_extension.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "experiments/krenn_quantum_graph/tensor_invariants.py",
    "experiments/krenn_quantum_graph/tensor_map.py",
    "experiments/krenn_quantum_graph/ternary_search.py",
    "experiments/krenn_quantum_graph/ternary_seed_orbits.py",
    "experiments/krenn_quantum_graph/transport.py",
    "experiments/krenn_quantum_graph/witness.py",
)


class KrennDefectMiningError(RuntimeError):
    """The exact seed-defect report or its artifact replay failed."""


def _fraction_payload(value: Fraction | int) -> Mapping:
    value = Fraction(value)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _qomega_payload(value) -> Mapping:
    return {
        "constant": _fraction_payload(value.constant),
        "omega": _fraction_payload(value.omega),
    }


def seed_witness(
    representative: Sequence[int],
    *,
    coordinate_values: Mapping[tuple[int, int, int, int], int] | None = None,
) -> SparseWitness:
    """Return the diagonal nine-slot witness for an ordered matching seed."""

    representative = tuple(map(int, representative))
    if len(representative) != 3 or any(
        index < 0 or index >= len(perfect_matchings(6))
        for index in representative
    ):
        raise KrennDefectMiningError(
            "seed representative needs three matching indices"
        )
    coordinates = {}
    for color, matching_index in enumerate(representative):
        for i, j in perfect_matchings(6)[matching_index]:
            key = (i, j, color, color)
            coordinates[key] = (
                int(coordinate_values[key])
                if coordinate_values is not None and key in coordinate_values
                else 1
            )
    if any(value == 0 for value in coordinates.values()):
        raise KrennDefectMiningError(
            "the fixed-support theorem requires all nine slots nonzero"
        )
    return SparseWitness.from_coordinates(6, 3, coordinates)


def _dense_values(witness: SparseWitness) -> tuple[int, ...]:
    values = [0] * variable_count(witness.n, witness.d)
    for index, value in witness.entries:
        if value.denominator != 1:
            raise KrennDefectMiningError(
                "defect census expects an integral seed witness"
            )
        values[index] = value.numerator
    return tuple(values)


def _monomial_value(
    monomial: Sequence[int],
    values: Mapping[int, Fraction],
) -> Fraction:
    result = Fraction(1)
    for variable in monomial:
        result *= values.get(variable, 0)
    return result


def _equation_polynomial_payload(
    system: SparsePolynomialSystem,
    equation: int,
    witness: SparseWitness,
) -> Mapping:
    coloring = coloring_from_index(system.n, system.d, equation)
    values = witness.as_dict()
    terms = []
    for matching_index, monomial in enumerate(
        system.equation_monomials(equation)
    ):
        value = _monomial_value(monomial, values)
        terms.append(
            {
                "matching_index": matching_index,
                "matching": [
                    list(edge)
                    for edge in perfect_matchings(system.n)[
                        matching_index
                    ]
                ],
                "variables": [
                    list(variable_key(system.n, system.d, variable))
                    for variable in monomial
                ],
                "seed_term_value": value.numerator,
            }
        )
    expression_terms = [
        "*".join(
            "W[" + ",".join(map(str, variable)) + "]"
            for variable in term["variables"]
        )
        for term in terms
    ]
    target_rhs = int(system.rhs_values[equation])
    expression = " + ".join(expression_terms)
    if target_rhs:
        expression += f" - {target_rhs}"
    return {
        "equation": equation,
        "coloring": list(coloring),
        "color_occupation": [
            coloring.count(color) for color in range(system.d)
        ],
        "target_rhs": target_rhs,
        "residual_polynomial": "sum(15 matching monomials) - target_rhs",
        "expanded_expression": expression,
        "terms": terms,
        "surviving_matching_indices": [
            term["matching_index"]
            for term in terms
            if term["seed_term_value"]
        ],
    }


def _defect_rows(
    system: SparsePolynomialSystem,
    witness: SparseWitness,
) -> tuple[Mapping, ...]:
    candidate = evaluate_ternary_candidate(
        system, _dense_values(witness)
    )
    values = witness.as_dict()
    rows = []
    for equation, residual in enumerate(candidate.residuals):
        if not residual:
            continue
        coloring = coloring_from_index(6, 3, equation)
        surviving = []
        for matching_index, monomial in enumerate(
            system.equation_monomials(equation)
        ):
            value = _monomial_value(monomial, values)
            if value:
                surviving.append(
                    {
                        "matching_index": matching_index,
                        "value": value.numerator,
                        "variables": [
                            list(variable_key(6, 3, variable))
                            for variable in monomial
                        ],
                    }
                )
        rows.append(
            {
                "equation": equation,
                "coloring": list(coloring),
                "occupation": [
                    coloring.count(color) for color in range(3)
                ],
                "occupation_partition": sorted(
                    (
                        coloring.count(color)
                        for color in range(3)
                        if coloring.count(color)
                    ),
                    reverse=True,
                ),
                "residual": residual,
                "surviving_terms": surviving,
            }
        )
    return tuple(rows)


def _pairwise_intersection_sum(
    representative: Sequence[int],
) -> int:
    matchings = perfect_matchings(6)
    return sum(
        len(
            set(matchings[representative[left]])
            & set(matchings[representative[right]])
        )
        for left, right in combinations(range(3), 2)
    )


def _occupation_coefficients(
    defects: Iterable[Mapping],
) -> tuple[Mapping, ...]:
    coefficients: Counter[tuple[int, ...]] = Counter()
    for defect in defects:
        coefficients[tuple(defect["occupation"])] += defect["residual"]
    return tuple(
        {
            "occupation": list(occupation),
            "coefficient": coefficient,
        }
        for occupation, coefficient in sorted(coefficients.items())
    )


def _orbit_row(
    system: SparsePolynomialSystem,
    representative: tuple[int, int, int],
    orbit_size: int,
) -> Mapping:
    witness = seed_witness(representative)
    defects = _defect_rows(system, witness)
    shared = _pairwise_intersection_sum(representative)
    rainbow = sum(
        defect["occupation_partition"] == [2, 2, 2]
        for defect in defects
    )
    four_plus_two = sum(
        defect["occupation_partition"] == [4, 2]
        for defect in defects
    )
    if len(defects) != 2 * shared + rainbow:
        raise KrennDefectMiningError(
            "support-defect formula failed exact replay"
        )
    if four_plus_two != 2 * shared:
        raise KrennDefectMiningError(
            "shared-edge defect census failed exact replay"
        )
    return {
        "representative_matching_indices": list(representative),
        "representative_matchings": [
            [list(edge) for edge in perfect_matchings(6)[index]]
            for index in representative
        ],
        "ordered_seed_orbit_size": orbit_size,
        "pairwise_shared_edge_sum": shared,
        "four_plus_two_defects": four_plus_two,
        "rainbow_two_plus_two_plus_two_defects": rainbow,
        "defect_formula_2s_plus_r": 2 * shared + rainbow,
        "nonzero_residual_count": len(defects),
        "residual_l1_for_unit_weights": sum(
            abs(defect["residual"]) for defect in defects
        ),
        "equal_g_residual_coefficients": list(
            _occupation_coefficients(defects)
        ),
        "defects": list(defects),
    }


def _matching_incidence_audit(
    system: SparsePolynomialSystem,
) -> Mapping:
    edges = canonical_edges(6)
    matrix = tuple(
        tuple(int(edge in matching) for edge in edges)
        for matching in perfect_matchings(6)
    )
    rank = _rational_rank(matrix)
    relation = [0] * 15
    for index, coefficient in (
        (1, -1),
        (2, 1),
        (4, 1),
        (5, -1),
        (7, -1),
        (8, 1),
    ):
        relation[index] = coefficient
    relation_zero = all(
        sum(
            relation[row] * matrix[row][column]
            for row in range(15)
        )
        == 0
        for column in range(15)
    )
    fixed_monomials = system.equation_monomials(
        N6_D3_SEED_DEFECT_EQUATION
    )
    all_monomials = tuple(
        monomial
        for equation in range(system.equation_count)
        for monomial in system.equation_monomials(equation)
    )
    fixed_distinct = (
        len(fixed_monomials) == len(set(fixed_monomials)) == 15
    )
    global_distinct = (
        len(all_monomials) == len(set(all_monomials)) == 10_935
    )
    return {
        "fixed_coloring_matching_monomials": 15,
        "fixed_coloring_monomials_pairwise_distinct": fixed_distinct,
        "global_n6_d3_monomials": 10_935,
        "global_monomials_pairwise_distinct": global_distinct,
        "matching_edge_incidence_rank": rank,
        "matching_edge_incidence_nullity": 15 - rank,
        "sample_incidence_relation": relation,
        "sample_relation_replays_edgewise": relation_zero,
        "induced_toric_binomial": "m2*m4*m8 - m1*m5*m7 = 0",
        "linear_matching_monomial_relation_found": not fixed_distinct,
        "conclusion": (
            "The incidence relation is multiplicative after monomial "
            "parameterization; it does not cancel the sole surviving term."
        ),
    }


def _rational_rank(matrix: Sequence[Sequence[int]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    rank = 0
    for column in range(len(rows[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                left - factor * right
                for left, right in zip(rows[row], rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _permutation_sign(permutation: Sequence[int]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _representation_audit(system: SparsePolynomialSystem) -> Mapping:
    matchings = perfect_matchings(6)
    signed_fixed_sum = 0
    for permutation in permutations(range(6)):
        fixed = sum(
            transport_matching(matching, permutation) == matching
            for matching in matchings
        )
        signed_fixed_sum += _permutation_sign(permutation) * fixed
    sign_multiplicity = signed_fixed_sum // 720
    all_monomials = tuple(
        monomial
        for equation in range(system.equation_count)
        for monomial in system.equation_monomials(equation)
    )
    invariant_occupation_partitions = {
        tuple(
            sorted(
                Counter(coloring_from_index(6, 3, equation)).values(),
                reverse=True,
            )
        )
        for equation in range(system.equation_count)
    }
    return {
        "S6_sign_multiplicity_in_matching_permutation_module": (
            sign_multiplicity
        ),
        "S6_sign_character_forces_defect": False,
        "S6xS3_invariant_output_directions": len(
            invariant_occupation_partitions
        ),
        "invariant_color_partitions": [
            list(partition)
            for partition in sorted(
                invariant_occupation_partitions, reverse=True
            )
        ],
        "GHZ_partition": [6],
        "natural_defect_partition": [2, 2, 2],
        "all_coordinate_monomials_distinct": (
            len(all_monomials) == len(set(all_monomials)) == 10_935
        ),
        "linear_span_of_tensor_map_output": 3**6,
        "linear_output_invariant_exists": False,
        "conclusion": (
            "Symmetry leaves the GHZ and balanced-defect directions "
            "independent. Any full-image obstruction must be nonlinear."
        ),
    }


def _signed_natural_seed_audit(
    system: SparsePolynomialSystem,
) -> Mapping:
    flipped = {
        (0, 1, 0, 0): -1,
        (2, 3, 0, 0): -1,
    }
    witness = seed_witness(
        NATURAL_REPRESENTATIVE,
        coordinate_values=flipped,
    )
    candidate = evaluate_ternary_candidate(
        system, _dense_values(witness)
    )
    constant_equations = (
        coloring_index(6, 3, (0,) * 6),
        coloring_index(6, 3, (1,) * 6),
        coloring_index(6, 3, (2,) * 6),
    )
    return {
        "two_flipped_coordinates": [
            [0, 1, 0, 0],
            [2, 3, 0, 0],
        ],
        "constant_residuals": [
            candidate.residuals[equation]
            for equation in constant_equations
        ],
        "victim_residual": candidate.residuals[
            N6_D3_SEED_DEFECT_EQUATION
        ],
        "plus_one_is_sign_forced": False,
        "support_nonvanishing_is_forced": True,
    }


def _nullstellensatz_payload(
    system: SparsePolynomialSystem,
) -> Mapping:
    witness = seed_witness(NATURAL_REPRESENTATIVE)
    values = witness.as_dict()
    defect_monomial = next(
        monomial
        for monomial in system.equation_monomials(
            N6_D3_SEED_DEFECT_EQUATION
        )
        if _monomial_value(monomial, values)
    )
    pure_products = []
    for color in range(3):
        equation = coloring_index(6, 3, (color,) * 6)
        pure_products.append(
            next(
                monomial
                for monomial in system.equation_monomials(equation)
                if _monomial_value(monomial, values)
            )
        )
    all_support = {
        variable for monomial in pure_products for variable in monomial
    }
    defect_set = set(defect_monomial)
    if len(all_support) != 9 or not defect_set <= all_support:
        raise KrennDefectMiningError(
            "fixed-support certificate variables changed"
        )
    quotient = tuple(sorted(all_support - defect_set))
    multiply = lambda *monomials: tuple(
        sorted(
            variable
            for monomial in monomials
            for variable in monomial
        )
    )
    product_012 = multiply(*pure_products)
    product_dq = multiply(defect_monomial, quotient)
    collected = Counter()
    for coefficient, monomial in (
        (1, product_dq),
        (-1, product_012),
        (1, multiply(pure_products[1], pure_products[2])),
        (-1, multiply(pure_products[1], pure_products[2])),
        (1, tuple(pure_products[2])),
        (-1, tuple(pure_products[2])),
        (1, ()),
    ):
        collected[monomial] += coefficient
    collected = Counter(
        {
            monomial: coefficient
            for monomial, coefficient in collected.items()
            if coefficient
        }
    )
    content_identity_exact = product_012 == product_dq
    symbolic_replay = (
        content_identity_exact
        and collected == Counter({(): 1})
    )
    return {
        "constant_products": [
            [list(variable_key(6, 3, variable)) for variable in monomial]
            for monomial in pure_products
        ],
        "defect_monomial_D": [
            list(variable_key(6, 3, variable))
            for variable in defect_monomial
        ],
        "complementary_monomial_Q": [
            list(variable_key(6, 3, variable)) for variable in quotient
        ],
        "content_identity": "P0*P1*P2 = D*Q",
        "content_identity_replays_symbolically": content_identity_exact,
        "nullstellensatz_identity": (
            "1 = D*Q - ((P0-1)*P1*P2 + (P1-1)*P2 + (P2-1))"
        ),
        "collected_identity_terms": [
            {
                "monomial_variable_indices": list(monomial),
                "coefficient": coefficient,
            }
            for monomial, coefficient in sorted(collected.items())
        ],
        "certificate_replays_symbolically": symbolic_replay,
        "scope": (
            "This is a unit-ideal certificate only after restricting to "
            "the nine selected diagonal support variables."
        ),
    }


def _add_coordinate(
    witness: SparseWitness,
    key: tuple[int, int, int, int],
) -> SparseWitness:
    coordinates = dict(witness.coordinate_entries())
    if key in coordinates:
        raise KrennDefectMiningError(
            "the proposed n=4 repair coordinate already exists"
        )
    coordinates[key] = Fraction(1)
    return SparseWitness.from_coordinates(4, 3, coordinates)


def _ghz_plus_coloring(coloring: Sequence[int]) -> ColoringTarget:
    entries = dict(canonical_ghz_target(4, 3).entries)
    coloring = tuple(map(int, coloring))
    entries[coloring] = entries.get(coloring, 0) + 1
    return ColoringTarget.from_sparse(4, 3, entries)


def _bell_contraction_audit() -> Mapping:
    victim = coloring_from_index(
        6, 3, N6_D3_SEED_DEFECT_EQUATION
    )
    retained = []
    killed = []
    repair_keys = {
        (2, 1, 2, 1): (0, 2, 2, 2),
        (0, 0, 1, 1): (2, 3, 1, 1),
        (0, 0, 2, 2): (2, 3, 2, 2),
    }
    for u, v in canonical_edges(6):
        remainder = tuple(
            vertex for vertex in range(6) if vertex not in {u, v}
        )
        contracted = tuple(victim[vertex] for vertex in remainder)
        if victim[u] == victim[v]:
            witness = _add_coordinate(
                fixture_n4_d3(), repair_keys[contracted]
            )
            exact_preimage = (
                MatchingTensorMap(4, 3).evaluate_exact(witness)
                == _ghz_plus_coloring(contracted)
            )
            retained.append(
                {
                    "contracted_pair": [u, v],
                    "remaining_coloring": list(contracted),
                    "explicit_n4_preimage_extra_coordinate": list(
                        repair_keys[contracted]
                    ),
                    "contracted_target_is_in_Phi_4_3_image": (
                        exact_preimage
                    ),
                }
            )
        else:
            killed.append([u, v])
    dominance = _bell_contraction_dominance_certificate()
    image_bound = universal_image_dimension_bounds(
        MatchingTensorMap(4, 3)
    )
    return {
        "Bell_contraction": (
            "C_uv(T)_x = sum_a T_(c_u=a,c_v=a,c_remaining=x)"
        ),
        "GHZ6_contracts_to_GHZ4": True,
        "pairs_retaining_the_seed_defect": retained,
        "pairs_killing_the_seed_defect": killed,
        "retained_count": len(retained),
        "killed_count": len(killed),
        "contracted_defects_are_n4_nonimage_obstructions": False,
        "dominance_audit": dominance,
        "Phi_4_3_image_dimension_upper_bound": (
            image_bound.affine_image_dimension_upper_bound
        ),
        "universal_Bell_contraction_preserves_Phi_image": False,
        "conclusion": (
            "Edge extraction exposes ordinary n=4 mixed coefficients, "
            "but neither deletion nor Bell contraction yields an n=4 "
            "nonimage theorem."
        ),
    }


def _bell_contraction_dominance_certificate() -> Mapping:
    system = generate_sparse_system(6, 3)
    modulus = 31
    values = tuple(
        (
            4
            + 16 * index
            + 30 * index * index
            + 15 * index * index * index
        )
        % modulus
        for index in range(system.variable_count)
    )
    rows = []
    for output_index in range(3**4):
        output_coloring = coloring_from_index(4, 3, output_index)
        row = [0] * system.variable_count
        for contracted_color in range(3):
            coloring = (
                contracted_color,
                contracted_color,
                *output_coloring,
            )
            equation = coloring_index(6, 3, coloring)
            for monomial in system.equation_monomials(equation):
                for position, variable in enumerate(monomial):
                    term = 1
                    for other_position, other in enumerate(monomial):
                        if other_position != position:
                            term = term * values[other] % modulus
                    row[variable] = (
                        row[variable] + term
                    ) % modulus
        rows.append(tuple(row))
    rank, pivots, determinant = _full_row_rank_minor(rows, modulus)
    return {
        "contracted_pair": [0, 1],
        "modulus": modulus,
        "integer_probe_formula": "4+16*i+30*i^2+15*i^3",
        "jacobian_shape": [81, 135],
        "rank_mod_31": rank,
        "pivot_minor_determinant_mod_31": determinant,
        "pivot_columns": list(pivots),
        "nonzero_integer_minor_proves_rank_81_over_Q": (
            rank == 81 and determinant != 0
        ),
        "proof_role": (
            "A nonzero reduction of a concrete integer minor proves that "
            "minor nonzero in characteristic zero; no F31 membership claim "
            "is made."
        ),
    }


def _full_row_rank_minor(
    matrix: Sequence[Sequence[int]],
    modulus: int,
) -> tuple[int, tuple[int, ...], int]:
    rows = [list(int(value) % modulus for value in row) for row in matrix]
    original = tuple(tuple(row) for row in rows)
    rank = 0
    pivots = []
    for column in range(len(rows[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], -1, modulus)
        rows[rank] = [
            value * inverse % modulus for value in rows[rank]
        ]
        for other in range(rank + 1, len(rows)):
            factor = rows[other][column]
            if factor:
                rows[other] = [
                    (left - factor * right) % modulus
                    for left, right in zip(rows[other], rows[rank])
                ]
        pivots.append(column)
        rank += 1
        if rank == len(rows):
            break
    determinant = 0
    if rank == len(rows):
        minor = [
            [original[row][column] for column in pivots]
            for row in range(len(rows))
        ]
        determinant = _determinant_mod(minor, modulus)
    return rank, tuple(pivots), determinant


def _determinant_mod(
    matrix: Sequence[Sequence[int]],
    modulus: int,
) -> int:
    rows = [list(row) for row in matrix]
    determinant = 1
    for column in range(len(rows)):
        pivot = next(
            (
                row
                for row in range(column, len(rows))
                if rows[row][column] % modulus
            ),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        pivot_value = rows[column][column] % modulus
        determinant = determinant * pivot_value % modulus
        inverse = pow(pivot_value, -1, modulus)
        for row in range(column + 1, len(rows)):
            factor = rows[row][column] * inverse % modulus
            for entry in range(column, len(rows)):
                rows[row][entry] = (
                    rows[row][entry]
                    - factor * rows[column][entry]
                ) % modulus
    return determinant % modulus


def build_defect_report() -> Mapping:
    """Recompute the exact support census and all mechanism audits."""

    system = generate_sparse_system(6, 3)
    natural_indices = tuple(
        perfect_matchings(6).index(tuple(factor))
        for factor in N6_D3_SEED_FACTORS
    )
    if (
        natural_indices != NATURAL_REPRESENTATIVE
        or canonical_seed_representative(natural_indices)
        != NATURAL_REPRESENTATIVE
    ):
        raise KrennDefectMiningError(
            "natural seed orbit convention changed"
        )
    natural_witness = seed_witness(NATURAL_REPRESENTATIVE)
    natural_defects = _defect_rows(system, natural_witness)
    if (
        len(natural_defects) != 1
        or natural_defects[0]["equation"]
        != N6_D3_SEED_DEFECT_EQUATION
    ):
        raise KrennDefectMiningError(
            "natural seed is no longer the exact 728/729 near-miss"
        )
    seed_census = ternary_seed_orbit_census()
    census_representatives_and_sizes = tuple(
        (
            tuple(row["representative_matching_indices"]),
            int(row["ordered_seed_orbit_size"]),
        )
        for row in seed_census["orbits"]
    )
    orbit_rows = tuple(
        _orbit_row(system, representative, orbit_size)
        for representative, orbit_size in census_representatives_and_sizes
    )
    equation_40_values = [
        int(
            any(
                defect["equation"] == SHARED_EDGE_VICTIM_EQUATION
                for defect in row["defects"]
            )
        )
        for row in orbit_rows
    ]
    equation_70_values = [
        int(
            any(
                defect["equation"] == N6_D3_SEED_DEFECT_EQUATION
                for defect in row["defects"]
            )
        )
        for row in orbit_rows
    ]

    ghz = canonical_ghz_target(6, 3)
    total_sum = certify_hafnian_contraction(
        natural_witness,
        ((1, 1, 1),) * 6,
        target=ghz,
    )
    equal_g = certify_equal_g_identity(
        natural_witness, target=ghz
    )
    shadow = generate_shadow_payload(equal_g, total_sum)
    border_certificate = certify_n6_d3_laurent_border()
    border_payload = border_certificate.to_dict()
    border_valuation = (
        certify_natural_border_valuation_obstruction()
    )
    border_valuation_payload = border_valuation.to_dict()
    formal_lift = certify_n6_d3_formal_lift()
    formal_lift_payload = formal_lift.to_dict()
    support_extension = certify_support_extension(
        recompute_exhaustive_census=True
    )
    support_extension_payload = support_extension.to_dict()
    n6_deformation = certify_n6_natural_deformation()
    n6_deformation_payload = n6_deformation.to_dict()
    matching_audit = _matching_incidence_audit(system)
    representation_audit = _representation_audit(system)
    signed_audit = _signed_natural_seed_audit(system)
    contraction_audit = _bell_contraction_audit()
    nullstellensatz = _nullstellensatz_payload(system)

    all_covered = (
        all(seed_census["exact_checks"].values())
        and seed_census["counts"]["ordered_three_matching_seeds"] == 3375
        and seed_census["counts"]["symmetry_orbits"] == 8
        and sum(
            row["ordered_seed_orbit_size"] for row in orbit_rows
        )
        == 3375
    )
    every_orbit_obstructed = all(
        row["nonzero_residual_count"] > 0 for row in orbit_rows
    )
    every_defect_unique = all(
        len(defect["surviving_terms"]) == 1
        for row in orbit_rows
        for defect in row["defects"]
    )
    common_defect_equations = sorted(
        set.intersection(
            *(
                {
                    defect["equation"]
                    for defect in row["defects"]
                }
                for row in orbit_rows
            )
        )
    )
    exact_checks = {
        "natural_seed_is_orbit_0_4_8": (
            natural_indices == NATURAL_REPRESENTATIVE
            and canonical_seed_representative(natural_indices)
            == NATURAL_REPRESENTATIVE
        ),
        "victim_is_equation_70_coloring_002121": (
            natural_defects[0]["coloring"] == [0, 0, 2, 1, 2, 1]
        ),
        "victim_is_balanced_2_2_2": (
            natural_defects[0]["occupation"] == [2, 2, 2]
        ),
        "natural_seed_has_one_residual_plus_one": (
            natural_defects[0]["residual"] == 1
        ),
        "eight_orbits_cover_all_3375_ordered_seeds": all_covered,
        "every_seed_orbit_has_a_mixed_defect": every_orbit_obstructed,
        "every_support_defect_has_one_surviving_monomial": (
            every_defect_unique
        ),
        "defect_count_formula_2s_plus_r_on_all_orbits": all(
            row["nonzero_residual_count"]
            == row["defect_formula_2s_plus_r"]
            for row in orbit_rows
        ),
        "weighted_hafnian_identity_exact": (
            total_sum.map_identity_exact
        ),
        "equal_g_coefficient_identity_exact": (
            equal_g.map_identity_exact
        ),
        "equal_g_d3_star_slice_is_target_generic_over_Q": (
            shadow["n6_d3_exact_star_linear_slice"][
                "target_generic_over_Q"
            ]
        ),
        "equal_g_d3_GHZ_fiber_has_exact_Q_point": (
            shadow["n6_d3_exact_star_linear_slice"][
                "canonical_GHZ_shadow_solution_exact"
            ]
        ),
        "equal_g_d3_sparse_cyclotomic_shadow_replays": (
            shadow["n6_d3_sparse_cyclotomic_shadow"][
                "all_28_shadow_equations_satisfied"
            ]
        ),
        "laurent_border_identity_replays_all_729_coefficients": (
            border_certificate.exact
        ),
        "GHZ_n6_d3_border_image_membership_certified": (
            border_certificate.border_image_membership_proved
        ),
        "natural_border_pole_survives_full_vertex_gauge": (
            border_valuation.this_family_vertex_gauge_regularization_impossible
        ),
        "natural_repair_has_all_order_formal_lift": (
            formal_lift_payload["claim_boundary"][
                "first_order_repair_lifts_to_every_finite_order"
            ]
        ),
        "natural_repair_formal_lift_is_unique_in_declared_linear_slice": (
            formal_lift_payload["claim_boundary"][
                "moving_target_formal_lift_unique_in_declared_linear_slice"
            ]
        ),
        "formal_lift_does_not_specialize_at_GHZ_endpoint": (
            not formal_lift_payload["claim_boundary"][
                "power_series_specialization_at_s_1_is_defined"
            ]
        ),
        "natural_support_extension_census_through_21_is_exhaustive": (
            support_extension.exhaustive_census_recomputed
            and support_extension.exhaustive_replay is not None
            and support_extension.exhaustive_replay.exhaustive
            and support_extension_payload["claim_boundary"][
                "cap21_exhaustive_census_replayed"
            ]
        ),
        "all_six_size21_terminal_supports_have_odd_cycle_contradictions": (
            len(support_extension.size21_audits) == 6
            and all(
                audit.odd_cycle
                .contradiction_over_characteristic_not_two
                for audit in support_extension.size21_audits
            )
        ),
        "natural_support_finite_exact_extension_needs_at_least_22_slots": (
            support_extension.finite_exact_total_support_lower_bound
            == 22
            and support_extension_payload["claim_boundary"][
                "finite_exact_total_support_at_least_22_proved"
            ]
        ),
        "n6_natural_jacobian_rank_and_gauge_certified": (
            n6_deformation.exact
        ),
        "fixed_support_nullstellensatz_identity_exact": (
            nullstellensatz["certificate_replays_symbolically"]
        ),
        "matching_monomials_have_no_linear_dependence": (
            matching_audit[
                "fixed_coloring_monomials_pairwise_distinct"
            ]
            and matching_audit["global_monomials_pairwise_distinct"]
            and not matching_audit[
                "linear_matching_monomial_relation_found"
            ]
        ),
        "S6_sign_multiplicity_zero": (
            representation_audit[
                "S6_sign_multiplicity_in_matching_permutation_module"
            ]
            == 0
        ),
        "plus_one_not_sign_forced": (
            signed_audit["victim_residual"] == -1
        ),
        "n4_contracted_defects_have_explicit_preimages": all(
            row["contracted_target_is_in_Phi_4_3_image"]
            for row in contraction_audit[
                "pairs_retaining_the_seed_defect"
            ]
        ),
        "Bell_contraction_map_is_dominant_over_Q": (
            contraction_audit["dominance_audit"][
                "nonzero_integer_minor_proves_rank_81_over_Q"
            ]
        ),
    }
    if not all(exact_checks.values()):
        raise KrennDefectMiningError(
            "one or more defect-mining exact checks failed"
        )
    return {
        "schema": DEFECT_REPORT_SCHEMA,
        "parameters": {
            "n": 6,
            "d": 3,
            "target": "canonical GHZ with constant coefficients one",
            "seed_family": (
                "one nonzero diagonal perfect matching per color"
            ),
        },
        "natural_near_miss": {
            "representative_matching_indices": list(
                NATURAL_REPRESENTATIVE
            ),
            "satisfied_equations": 728,
            "total_equations": 729,
            "victim": natural_defects[0],
            "exact_residual_polynomial": _equation_polynomial_payload(
                system,
                N6_D3_SEED_DEFECT_EQUATION,
                natural_witness,
            ),
        },
        "fixed_coordinate_evaluations_on_eight_representatives": {
            "R_40_coloring_001111": equation_40_values,
            "R_70_coloring_002121": equation_70_values,
            "common_defect_equations": common_defect_equations,
            "same_literal_defect_on_all_orbits": bool(
                common_defect_equations
            ),
        },
        "seed_orbit_census_replay": {
            "counts": seed_census["counts"],
            "exact_checks": seed_census["exact_checks"],
        },
        "orbit_representatives": list(orbit_rows),
        "hafnian_shadow": shadow,
        "full_tensor_laurent_border_certificate": border_payload,
        "border_regularization_analysis": {
            "vertex_gauge_valuation_obstruction": (
                border_valuation_payload
            ),
            "all_order_formal_lift": formal_lift_payload,
            "finite_support_extension": support_extension_payload,
        },
        "n6_natural_deformation_certificate": (
            n6_deformation_payload
        ),
        "fixed_support_nullstellensatz_certificate": nullstellensatz,
        "mechanism_audit": {
            "matching_linear_dependence": matching_audit,
            "sign": signed_audit,
            "representation": representation_audit,
            "n4_contraction": contraction_audit,
        },
        "restricted_family_theorem": {
            "statement": (
                "Over any integral domain, no assignment supported on "
                "exactly one nonzero diagonal perfect matching for each of "
                "three colors can map to GHZ_6,3."
            ),
            "proof_dichotomy": (
                "A shared edge forces a 4+2 mixed coefficient. If all "
                "three matchings are pairwise edge-disjoint, their cubic "
                "union has a rainbow perfect matching, forcing a 2+2+2 "
                "mixed coefficient."
            ),
            "signs_or_phases_can_cancel_the_forced_coefficient": False,
            "full_135_variable_nonimage_theorem": False,
            "global_polynomial_invariant_of_im_Phi": False,
        },
        "exact_checks": exact_checks,
        "claim_boundary": {
            "support_ansatz_no_go_certified": True,
            "all_ternary_weights_exhausted": False,
            "all_complex_weights_exhausted": False,
            "GHZ_nonimage_proved": False,
            "GHZ_border_image_membership_proved": True,
            "GHZ_exact_affine_image_membership_decided": False,
            "natural_border_pole_removable_by_vertex_gauge": False,
            "natural_repair_lifts_formally_to_all_orders": True,
            "natural_repair_formal_lift_unique_in_declared_linear_slice": (
                True
            ),
            "global_formal_branch_uniqueness_proved": False,
            "formal_lift_specializes_at_GHZ_endpoint": False,
            "natural_support_extensions_through_21_excluded": True,
            "natural_support_finite_exact_total_support_lower_bound": 22,
            "support_bound_retains_all_nine_natural_slots": True,
            "supports_dropping_a_natural_slot_excluded": False,
            "supports_of_size_22_or_more_excluded": False,
            "candidate_full_image_invariant_found": False,
            "equal_g_GHZ_exact_fiber_decided": True,
            "equal_g_GHZ_exact_fiber_nonempty_over_Q": True,
            "modular_Groebner_run_performed": False,
            "statement": (
                "The 728/729 defect is a certified sparse-support "
                "obstruction with an exact Laurent border degeneration. "
                "That branch lifts formally to all orders but has a "
                "gauge-invariant pole. An exhaustive raw support closure "
                "also proves that any finite exact extension retaining all "
                "nine natural slots needs at least 22 nonzero coordinates. "
                "Neither result is a finite GHZ witness or a full nonimage "
                "theorem."
            ),
        },
    }


def generate_shadow_payload(
    equal_g,
    total_sum,
) -> Mapping:
    system = generate_sparse_system(6, 3)
    shadow_system = generate_equal_g_shadow_system(6, 3)
    natural = seed_witness(NATURAL_REPRESENTATIVE)
    star_slice = generate_n6_d3_shadow_star_slice()
    star_solution = canonical_ghz_shadow_star_solution()
    cyclotomic = certify_cyclotomic_shadow_witness()
    edge_sums = []
    values = natural.as_dict()
    for i, j in canonical_edges(6):
        total = sum(
            values.get(variable_index(6, 3, i, j, a, b), 0)
            for a in range(3)
            for b in range(3)
        )
        edge_sums.append(
            {"edge": [i, j], "s_ij": int(total)}
        )
    residual_coefficients = []
    for occupation, residual in zip(
        occupations(6, 3),
        equal_g.residual_coefficients,
    ):
        if residual:
            residual_coefficients.append(
                {
                    "occupation": list(occupation),
                    "coefficient": residual.numerator,
                }
            )
    return {
        "general_identity": (
            "sum_c Phi(W)_c*prod_i g_i[c_i] = "
            "haf((g_i^T W_ij g_j)_(i,j))"
        ),
        "equal_g_identity": (
            "haf((g^T W_ij g)_(i,j)) = sum_a g_a^6 "
            "for a GHZ solution"
        ),
        "natural_edge_total_sums": edge_sums,
        "all_ones_contraction": {
            "hafnian_of_edge_sums": total_sum.hafnian_value.numerator,
            "sum_of_all_729_outputs": (
                total_sum.tensor_map_value.numerator
            ),
            "sum_of_GHZ_target": total_sum.target_value.numerator,
            "residual": total_sum.target_residual.numerator,
        },
        "natural_equal_g_residual_coefficients": residual_coefficients,
        "n6_d3_exact_star_linear_slice": {
            "schema": star_slice.schema,
            "hub_vertex": star_slice.hub_vertex,
            "star_variables": list(star_slice.star_variables),
            "nonstar_variables": list(star_slice.nonstar_variables),
            "fixed_nonstar_signs": [
                star_slice.fixed_shadow_values[variable]
                for variable in star_slice.nonstar_variables
            ],
            "equations": len(star_slice.coefficient_matrix),
            "star_variables_count": len(star_slice.star_variables),
            "free_star_variables": [
                star_slice.star_variables[column]
                for column in star_slice.free_columns
            ],
            "pivot_columns": list(star_slice.pivot_columns),
            "pivot_determinant_over_Z": (
                star_slice.pivot_determinant
            ),
            "pivot_determinant_factorization": "2^54",
            "pivot_determinant_mod_31": (
                star_slice.pivot_determinant_mod_31
            ),
            "every_matching_monomial_uses_one_star_variable": all(
                sum(
                    variable in set(star_slice.star_variables)
                    for variable in monomial
                )
                == 1
                for equation in shadow_system.equation_terms
                for monomial in equation
            ),
            "zero_star_constant_term": not any(
                shadow_system.evaluate(
                    star_slice.fixed_shadow_values
                )
            ),
            "target_generic_over_Q": star_slice.target_generic_over_q,
            "canonical_GHZ_shadow_solution": [
                _fraction_payload(value)
                for value in star_solution.shadow_values
            ],
            "canonical_GHZ_shadow_solution_exact": (
                star_solution.exact
            ),
            "claim_boundary": (
                "This solves every rational occupation-coefficient target "
                "in the equal-g shadow. It does not solve the 729 "
                "coloring-resolved tensor equations."
            ),
        },
        "n6_d3_sparse_cyclotomic_shadow": {
            "coefficient_field": "Q(omega), omega^2+omega+1=0",
            "nonzero_shadow_coordinates": [
                {
                    "coordinate": list(key),
                    "value": _qomega_payload(value),
                }
                for key, value in zip(
                    shadow_system.variable_keys,
                    cyclotomic.values,
                    strict=True,
                )
                if not value.is_zero
            ],
            "all_28_shadow_equations_satisfied": (
                cyclotomic.all_28_shadow_equations_satisfied
            ),
            "all_equal_full_tensor_amplitudes": [
                _qomega_payload(value)
                for value in cyclotomic.all_equal_amplitudes
            ],
            "full_tensor_victim": {
                "coloring": list(cyclotomic.victim_coloring),
                "amplitude": _qomega_payload(
                    cyclotomic.victim_amplitude
                ),
                "nonzero": cyclotomic.full_tensor_victim_nonzero,
            },
            "full_tensor_solution_claimed": (
                cyclotomic.full_tensor_solution_claimed
            ),
            "identity": (
                "(A^3+B^3+C^3+6*A*B*C)/9 = X^3+Y^3+Z^3"
            ),
        },
        "n6_d3_shadow": shadow_structure_summary(6, 3),
        "n6_d4_shadow": shadow_structure_summary(6, 4),
        "groebner_backend_run": False,
        "groebner_export_available": (
            "hafnian_identities.singular_groebner_script"
        ),
        "interpretation": (
            "The n=6,d=3 equal-g map has a fixed rational star-linear "
            "slice surjecting onto all 28 occupation coefficients, so its "
            "GHZ fiber is exactly nonempty and no d=3 shadow Groebner run "
            "is needed. The d=4 shadow is dominant in characteristic zero "
            "but its special GHZ fiber remains undecided. Neither shadow "
            "result is a solution of the coloring-resolved tensor system."
        ),
    }


def _canonical_json_bytes(payload: Mapping) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json_bytes(payload))
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(text.encode("utf-8"))
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> Mapping:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_record(relative_path: str) -> Mapping:
    path = ROOT / relative_path
    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return {
        "path": relative_path,
        "canonical_bytes": len(canonical),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "hash_mode": "canonical-lf-text-v1",
    }


def _build_certificate(report: Mapping) -> Mapping:
    orbit_rows = report["orbit_representatives"]
    return {
        "schema": DEFECT_CERTIFICATE_SCHEMA,
        "report_schema": DEFECT_REPORT_SCHEMA,
        "status": (
            "restricted-support-border-regularization-and-extension-certified"
        ),
        "parameters": {"n": 6, "d": 3},
        "counts": {
            "equations": 729,
            "natural_seed_nonzero_residuals": 1,
            "seed_orbits": len(orbit_rows),
            "ordered_seeds_covered": sum(
                row["ordered_seed_orbit_size"] for row in orbit_rows
            ),
            "orbit_defect_counts": [
                row["nonzero_residual_count"] for row in orbit_rows
            ],
            "natural_support_extension_states_through_size_21": (
                report["border_regularization_analysis"][
                    "finite_support_extension"
                ]["bounded_missing_set_closure"]["nodes_examined"]
            ),
            "singleton_free_size_21_terminal_supports": (
                report["border_regularization_analysis"][
                    "finite_support_extension"
                ]["bounded_missing_set_closure"][
                    "singleton_free_size21_support_count"
                ]
            ),
        },
        "victim": {
            "equation": report["natural_near_miss"]["victim"][
                "equation"
            ],
            "coloring": report["natural_near_miss"]["victim"][
                "coloring"
            ],
            "occupation": report["natural_near_miss"]["victim"][
                "occupation"
            ],
        },
        "exact_checks": report["exact_checks"],
        "claims": {
            "restricted_support_no_go": True,
            "full_tensor_GHZ_nonimage": False,
            "full_tensor_GHZ_border_image_membership": True,
            "full_tensor_GHZ_exact_image_membership_decided": False,
            "natural_border_pole_survives_vertex_gauge": True,
            "natural_repair_lifts_to_all_formal_orders": True,
            "natural_repair_formal_lift_unique_in_declared_linear_slice": (
                True
            ),
            "global_formal_branch_uniqueness_proved": False,
            "natural_formal_branch_specializes_at_GHZ": False,
            "natural_support_extensions_through_size_21_excluded": True,
            "natural_support_finite_exact_total_support_lower_bound": 22,
            "support_extension_bound_retains_all_natural_slots": True,
            "supports_dropping_a_natural_slot_excluded": False,
            "supports_of_size_22_or_more_excluded": False,
            "equal_g_d3_target_generic_over_Q": True,
            "equal_g_d3_GHZ_fiber_nonempty_over_Q": True,
            "natural_seed_full_jacobian_rank_over_Q": 130,
            "natural_seed_full_jacobian_nullity_over_Q": 5,
            "natural_seed_kernel_equals_vertex_gauge": True,
            "global_image_invariant_found": False,
            "n4_contraction_obstruction": False,
            "modular_Groebner_certificate": False,
            "statement": report["claim_boundary"]["statement"],
        },
        "artifacts": {
            "report": REPORT_FILE,
            "certificate": CERTIFICATE_FILE,
            "equal_g_n6_d3_F31_singular": GROEBNER_D3_FILE,
            "equal_g_n6_d4_F31_singular": GROEBNER_D4_FILE,
            "manifest": MANIFEST_FILE,
        },
    }


def write_defect_mining_bundle(
    output_directory: Path | str = DEFAULT_OUTPUT,
) -> Path:
    raw = Path(output_directory)
    if raw.is_symlink():
        raise KrennDefectMiningError(
            "defect output directory must not be a symbolic link"
        )
    output = raw.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir()}
    if not existing <= set(ALL_FILES):
        raise KrennDefectMiningError(
            "defect output directory has unexpected entries"
        )
    report = build_defect_report()
    certificate = _build_certificate(report)
    _write_json_atomic(output / REPORT_FILE, report)
    _write_json_atomic(output / CERTIFICATE_FILE, certificate)
    _write_text_atomic(
        output / GROEBNER_D3_FILE,
        singular_groebner_script(
            generate_equal_g_shadow_system(6, 3),
            canonical_ghz_target(6, 3),
        ),
    )
    _write_text_atomic(
        output / GROEBNER_D4_FILE,
        singular_groebner_script(
            generate_equal_g_shadow_system(6, 4),
            canonical_ghz_target(6, 4),
        ),
    )
    manifest = {
        "schema": DEFECT_MANIFEST_SCHEMA,
        "artifacts": [
            _artifact_record(output / filename)
            for filename in (
                CERTIFICATE_FILE,
                GROEBNER_D3_FILE,
                GROEBNER_D4_FILE,
                REPORT_FILE,
            )
        ],
        "inputs": [
            _source_record(relative_path)
            for relative_path in SOURCE_INPUTS
        ],
    }
    _write_json_atomic(output / MANIFEST_FILE, manifest)
    verify_defect_mining_bundle(output)
    return output / CERTIFICATE_FILE


def _load_json(path: Path) -> Mapping:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennDefectMiningError(
            f"could not load {path.name}"
        ) from error
    if not isinstance(payload, dict):
        raise KrennDefectMiningError(
            f"{path.name} must contain a JSON object"
        )
    return payload


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KrennDefectMiningError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


@dataclass(frozen=True)
class LoadedDefectMiningBundle:
    directory: Path
    report: Mapping
    certificate: Mapping
    manifest: Mapping

    @property
    def exact(self) -> bool:
        return all(self.report["exact_checks"].values())


def verify_defect_mining_bundle(
    output_directory: Path | str = DEFAULT_OUTPUT,
) -> LoadedDefectMiningBundle:
    raw = Path(output_directory)
    if raw.is_symlink():
        raise KrennDefectMiningError(
            "defect bundle directory must not be a symbolic link"
        )
    output = raw.resolve()
    if (
        not output.is_dir()
        or {path.name for path in output.iterdir()} != set(ALL_FILES)
        or any(path.is_symlink() or not path.is_file() for path in output.iterdir())
    ):
        raise KrennDefectMiningError(
            "defect bundle inventory changed"
        )
    report = _load_json(output / REPORT_FILE)
    certificate = _load_json(output / CERTIFICATE_FILE)
    manifest = _load_json(output / MANIFEST_FILE)
    if manifest.get("schema") != DEFECT_MANIFEST_SCHEMA:
        raise KrennDefectMiningError(
            "defect manifest schema changed"
        )
    expected_artifacts = [
        _artifact_record(output / filename)
        for filename in (
            CERTIFICATE_FILE,
            GROEBNER_D3_FILE,
            GROEBNER_D4_FILE,
            REPORT_FILE,
        )
    ]
    if manifest.get("artifacts") != expected_artifacts:
        raise KrennDefectMiningError(
            "defect artifact hash replay failed"
        )
    expected_inputs = [
        _source_record(relative_path) for relative_path in SOURCE_INPUTS
    ]
    if manifest.get("inputs") != expected_inputs:
        raise KrennDefectMiningError(
            "defect source ledger changed"
        )
    expected_report = build_defect_report()
    if report != expected_report:
        raise KrennDefectMiningError(
            "defect report failed deterministic semantic replay"
        )
    expected_certificate = _build_certificate(expected_report)
    if certificate != expected_certificate:
        raise KrennDefectMiningError(
            "defect certificate failed semantic replay"
        )
    expected_scripts = {
        GROEBNER_D3_FILE: singular_groebner_script(
            generate_equal_g_shadow_system(6, 3),
            canonical_ghz_target(6, 3),
        ),
        GROEBNER_D4_FILE: singular_groebner_script(
            generate_equal_g_shadow_system(6, 4),
            canonical_ghz_target(6, 4),
        ),
    }
    for filename, expected in expected_scripts.items():
        try:
            actual = (output / filename).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise KrennDefectMiningError(
                "could not replay a Singular export"
            ) from error
        if actual != expected:
            raise KrennDefectMiningError(
                "Singular export failed deterministic semantic replay"
            )
    bundle = LoadedDefectMiningBundle(
        directory=output,
        report=report,
        certificate=certificate,
        manifest=manifest,
    )
    if not bundle.exact:
        raise KrennDefectMiningError(
            "defect bundle exact checks failed"
        )
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the exact n=6,d=3 defect audit."
    )
    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify:
        write_defect_mining_bundle(arguments.output_directory)
    bundle = verify_defect_mining_bundle(arguments.output_directory)
    print(
        json.dumps(
            {
                "exact": bundle.exact,
                "status": bundle.certificate["status"],
                "victim": bundle.certificate["victim"],
                "orbit_defect_counts": bundle.certificate["counts"][
                    "orbit_defect_counts"
                ],
                "restricted_support_no_go": True,
                "full_tensor_GHZ_nonimage": False,
                "full_tensor_GHZ_border_image_membership": (
                    bundle.certificate["claims"][
                        "full_tensor_GHZ_border_image_membership"
                    ]
                ),
                "equal_g_d3_GHZ_fiber_nonempty_over_Q": (
                    bundle.certificate["claims"][
                        "equal_g_d3_GHZ_fiber_nonempty_over_Q"
                    ]
                ),
                "natural_border_pole_survives_vertex_gauge": (
                    bundle.certificate["claims"][
                        "natural_border_pole_survives_vertex_gauge"
                    ]
                ),
                "natural_repair_lifts_to_all_formal_orders": (
                    bundle.certificate["claims"][
                        "natural_repair_lifts_to_all_formal_orders"
                    ]
                ),
                "natural_repair_formal_lift_unique_in_declared_linear_slice": (
                    bundle.certificate["claims"][
                        "natural_repair_formal_lift_unique_in_declared_linear_slice"
                    ]
                ),
                "natural_support_finite_exact_total_support_lower_bound": (
                    bundle.certificate["claims"][
                        "natural_support_finite_exact_total_support_lower_bound"
                    ]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
