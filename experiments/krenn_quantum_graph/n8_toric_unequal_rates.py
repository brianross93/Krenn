"""Exact unequal-rate quotient gate for the ``n=8,d=3`` H5/H6 shell.

The flat H5/H6 first-shell certificate chooses one cost-two repair for each
original mixed singleton.  This module asks whether assigning unequal
orders to the coordinates of such a minimal repair skeleton produces a new
one-parameter degeneration after quotienting the target-preserving
color-diagonal torus.

For a fixed decorated skeleton ``S`` the necessary order constraints are:

* the three selected monochromatic matching products have order zero; and
* each selected repair monomial ties the seed monomial in its victim
  equation.

They define an integral lattice ``K_S``.  The restriction ``G_S`` of the
``252 x 21`` target-preserving gauge matrix lies in ``K_S``.  On every one
of the 43 H5 and 304 H6 decorated stabilizer orbits,

``rank(G_S) = |S| - rank(C_S)``

and ``G_S`` has an explicit maximal minor of determinant ``+1`` or ``-1``.
Consequently its image is saturated and equals ``K_S`` over ``Z`` in the
affine target-order-zero slice.  Projectively, adjoining common source
scaling leaves a finite ``Z/4`` quotient, reflecting that the matching map
has degree four.  This torsion is not a ray: the rational quotient has
dimension zero, so every unequal rate satisfying the declared ties is pure
gauge after at most the familiar degree-four base change.

This makes the flat singleton obstruction exhaustive only for the declared
seed-retaining, one-cost-two-repair-per-victim skeleton family.  It does not
cover extra leading repair monomials, higher-cost repairs, outside terms
entering an initial form, deeper closures, the other 29 seed orbits, or the
full graph/Rees compactification.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from itertools import product
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import experiments.krenn_quantum_graph.n8_toric_first_shell as first_shell


N = 8
D = 3
SOURCE_VARIABLES = 252
GAUGE_DIMENSION = 21
UNEQUAL_RATE_SCHEMA = "krenn-n8-d3-toric-unequal-rates-v2"
UNEQUAL_RATE_MANIFEST_SCHEMA = (
    "krenn-n8-d3-toric-unequal-rates-manifest-v2"
)
DEFAULT_RESULTS_DIRECTORY = (
    Path("results")
    / "krenn_quantum_graph"
    / "n8_d3_toric_unequal_rates"
)
CERTIFICATE_FILE = "certificate.json"
README_FILE = "README.md"
MANIFEST_FILE = "manifest.json"
SOURCE_PATHS = (
    "experiments/krenn_quantum_graph/independent_verifier.py",
    "experiments/krenn_quantum_graph/n8_seed_orbits.py",
    "experiments/krenn_quantum_graph/n8_toric_first_shell.py",
    "experiments/krenn_quantum_graph/n8_toric_unequal_rates.py",
    "experiments/krenn_quantum_graph/system.py",
    "experiments/krenn_quantum_graph/targets.py",
    "tests/test_krenn_n8_toric_first_shell.py",
    "tests/test_krenn_n8_toric_unequal_rates.py",
)

EXPECTED = {
    "H5": {
        "raw_decorations": 144,
        "decorated_orbits": 43,
        "support_size": 16,
        "victims": 2,
        "constraint_rank": 5,
        "gauge_rank": 11,
        "determinant_histogram": {-1: 24, 1: 19},
        "projective_constraint_rank": 4,
        "projective_gauge_rank": 12,
        "projective_determinant_histogram": {-4: 22, 4: 21},
        "target_chart_count": 57,
        "target_count_triple_histogram": {
            (1, 1, 1): 31,
            (2, 1, 1): 10,
            (3, 1, 1): 2,
        },
        "q_minor_histogram": {-1: 20, 1: 37},
        "q_witness_nnz_histogram": {3: 13, 4: 24, 5: 20},
    },
    "H6": {
        "raw_decorations": 1_728,
        "decorated_orbits": 304,
        "support_size": 18,
        "victims": 3,
        "constraint_rank": 6,
        "gauge_rank": 12,
        "determinant_histogram": {-1: 158, 1: 146},
        "projective_constraint_rank": 5,
        "projective_gauge_rank": 13,
        "projective_determinant_histogram": {-4: 156, 4: 148},
        "target_chart_count": 484,
        "target_count_triple_histogram": {
            (1, 1, 1): 180,
            (1, 1, 2): 52,
            (1, 2, 2): 14,
            (1, 2, 1): 32,
            (2, 1, 2): 6,
            (2, 1, 1): 16,
            (2, 2, 2): 2,
            (2, 2, 1): 2,
        },
        "q_minor_histogram": {-1: 221, 1: 263},
        "q_witness_nnz_histogram": {3: 64, 4: 220, 5: 184, 6: 16},
    },
}

# The full gauge has the expected generic mu_2 ineffectivity.  This minor,
# together with mod-two rank 20, proves saturation index exactly two.
GLOBAL_INDEX_TWO_MINOR_ROWS = (
    0,
    1,
    21,
    32,
    33,
    71,
    74,
    105,
    108,
    129,
    163,
    171,
    175,
    186,
    211,
    229,
    231,
    238,
    239,
    246,
    248,
)



DEPTH_TWO_EXPECTED = {
    "parent_label": "H5",
    "parent_orbit_index": 3,
    "parent_repair_matching_indices": (0, 11),
    "parent_support": (
        0,
        13,
        26,
        85,
        98,
        117,
        151,
        161,
        184,
        194,
        198,
        205,
        215,
        238,
        243,
        250,
    ),
    "spill_equations": (7, 63, 4_049, 5_791),
    "repair_options_per_spill": 12,
    "raw_branch_count": 20_736,
    "branch_support_and_quotient_histogram": {
        (22, 0): 284,
        (22, 1): 4,
        (24, 0): 19_588,
        (24, 1): 852,
        (24, 2): 8,
    },
    "minimum_positive_quotient_support_size": 22,
    "positive_minimum_support_branches": 4,
    "target_term_tie_row_span_witnesses": {
        0: {
            (0, 117, 198, 243): (0, 0, 0, 0, 0, 0),
            (9, 72, 198, 243): (0, 0, 1, 0, 0, 0),
        },
        3_280: {
            (13, 85, 184, 238): (0, 0, 0, 0, 0, 0),
        },
        6_560: {
            (26, 98, 161, 215): (0, 0, 0, 0, 0, 0),
        },
    },
    "classes": {
        "A": {
            "representative": (15, 15, 21, 35),
            "partner": (30, 30, 22, 33),
            "support": (
                0, 9, 13, 26, 72, 85, 98, 117, 151, 160, 161,
                167, 184, 194, 198, 205, 215, 229, 238, 243, 250,
                251,
            ),
            "nonseed": (
                9, 72, 151, 160, 167, 194, 205, 229, 250, 251,
            ),
            "ray": (0, 0, 1, 1, -1, -1, 0, 0, 0, 0),
            "detector": {
                13: 1,
                85: 1,
                151: -1,
                160: 1,
                167: -1,
                184: 2,
                205: 1,
            },
            "ray_obstruction_histograms": {
                "+": {-1: 7, 0: 7, 1: 2},
                "-": {-1: 3, 0: 7, 1: 6},
            },
            "positive_singleton_circuit": {
                571: 3,
                851: 2,
                2_438: 1,
                2_493: 1,
                4_048: 1,
                5_792: 1,
            },
            "weighted_nonseed_incidence": (
                2, 2, 0, 3, 1, 2, 1, 3, 1, 1,
            ),
            "tie_row_span_witness": (1, 3, 2, 0, 1, 3),
        },
        "B": {
            "representative": (15, 15, 22, 33),
            "partner": (30, 30, 21, 35),
            "support": (
                0, 9, 13, 26, 72, 85, 98, 117, 142, 151, 161,
                184, 185, 194, 198, 205, 215, 224, 238, 243, 247,
                250,
            ),
            "nonseed": (
                9, 72, 142, 151, 185, 194, 205, 224, 247, 250,
            ),
            "ray": (0, 0, 1, 1, -1, -1, 0, 0, 0, 0),
            "detector": {
                26: -1,
                98: -1,
                142: 1,
                151: -1,
                161: -1,
                184: 1,
                185: -1,
                205: -1,
            },
            "ray_obstruction_histograms": {
                "+": {-1: 4, 0: 8, 1: 3},
                "-": {-1: 4, 0: 8, 1: 3},
            },
            "positive_singleton_circuit": {
                853: 1,
                2_430: 1,
                6_557: 1,
            },
            "weighted_nonseed_incidence": (
                1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
            ),
            "tie_row_span_witness": (0, 1, 1, 0, 1, 0),
        },
    },
}
class KrennN8ToricUnequalRateError(ValueError):
    """An unequal-rate lattice certificate failed exact replay."""


def _rank_mod_prime(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> int:
    rows = [list(int(value) % prime for value in row) for row in matrix]
    if prime <= 1:
        raise KrennN8ToricUnequalRateError(
            "a modular rank needs a prime greater than one"
        )
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennN8ToricUnequalRateError(
            "a modular-rank matrix is ragged"
        )
    rank = 0
    for column in range(width):
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
        inverse = pow(rows[rank][column], -1, prime)
        rows[rank] = [
            value * inverse % prime for value in rows[rank]
        ]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                (value - factor * pivot_value) % prime
                for value, pivot_value in zip(
                    rows[row], rows[rank], strict=True
                )
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _sparse_row(
    support_positions: Mapping[int, int],
    positive: Sequence[int],
    negative: Sequence[int] = (),
) -> tuple[int, ...]:
    row = [0] * len(support_positions)
    for index in positive:
        try:
            row[support_positions[int(index)]] += 1
        except KeyError as error:
            raise KrennN8ToricUnequalRateError(
                "a positive constraint coordinate left its skeleton"
            ) from error
    for index in negative:
        try:
            row[support_positions[int(index)]] -= 1
        except KeyError as error:
            raise KrennN8ToricUnequalRateError(
                "a negative constraint coordinate left its skeleton"
            ) from error
    return tuple(row)


def _nonzero_entries(
    row: Sequence[int],
    support: Sequence[int],
) -> list[dict[str, int]]:
    return [
        {
            "support_position": position,
            "source_coordinate_index": int(support[position]),
            "coefficient": int(value),
        }
        for position, value in enumerate(row)
        if value
    ]


def _matrix_sha256(matrix: Sequence[Sequence[int]]) -> str:
    data = (
        json.dumps(
            [list(map(int, row)) for row in matrix],
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    return sha256(data).hexdigest()


def _global_gauge_lattice_record() -> dict[str, object]:
    gauge = first_shell.target_preserving_gauge_matrix()
    if (
        len(gauge) != SOURCE_VARIABLES
        or any(len(row) != GAUGE_DIMENSION for row in gauge)
    ):
        raise KrennN8ToricUnequalRateError(
            "the target-preserving gauge matrix shape changed"
        )
    minor = tuple(gauge[row] for row in GLOBAL_INDEX_TWO_MINOR_ROWS)
    determinant = first_shell._determinant(minor)
    ranks = {
        "over_Q": first_shell._rank_over_q(gauge),
        "mod_2": _rank_mod_prime(gauge, 2),
        "mod_3": _rank_mod_prime(gauge, 3),
    }
    checks = {
        "rank_over_Q_is_21": ranks["over_Q"] == GAUGE_DIMENSION,
        "rank_mod_2_is_20": ranks["mod_2"] == 20,
        "rank_mod_3_is_21": ranks["mod_3"] == GAUGE_DIMENSION,
        "displayed_maximal_minor_has_determinant_minus_2": (
            determinant.denominator == 1
            and determinant.numerator == -2
        ),
        "global_column_lattice_has_saturation_index_two": (
            ranks["mod_2"] < GAUGE_DIMENSION
            and abs(determinant) == 2
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricUnequalRateError(
            f"the global gauge-lattice audit failed: {failed}"
        )
    return {
        "matrix_shape": [SOURCE_VARIABLES, GAUGE_DIMENSION],
        "ranks": ranks,
        "index_two_minor": {
            "source_coordinate_rows": list(
                GLOBAL_INDEX_TWO_MINOR_ROWS
            ),
            "gauge_columns": list(range(GAUGE_DIMENSION)),
            "determinant": determinant.numerator,
        },
        "saturation_index": 2,
        "interpretation": (
            "the raw global cocharacter columns have the generic mu_2 "
            "ineffectivity; primitive global quotient calculations must "
            "use the effective saturated lattice"
        ),
        "exact_checks": checks,
    }


def _constraint_rows(
    label: str,
    hard_case: Mapping[str, object],
    orbit: Mapping[str, object],
    seed_normalization: Mapping[str, object],
) -> tuple[tuple[tuple[int, ...], ...], list[dict[str, object]]]:
    support = tuple(
        map(
            int,
            orbit["flat_valuation"]["order_zero_coordinate_indices"],
        )
    )
    positions = {index: position for position, index in enumerate(support)}
    if len(positions) != len(support):
        raise KrennN8ToricUnequalRateError(
            f"a {label} skeleton repeats a source coordinate"
        )

    rows: list[tuple[int, ...]] = []
    records: list[dict[str, object]] = []
    monochromatic = seed_normalization[
        "monochromatic_matching_products"
    ]
    if len(monochromatic) != D:
        raise KrennN8ToricUnequalRateError(
            f"the {label} seed lost a monochromatic product"
        )
    for product_record in monochromatic:
        monomial = tuple(
            map(int, product_record["source_variable_indices"])
        )
        row = _sparse_row(positions, monomial)
        rows.append(row)
        records.append(
            {
                "kind": "monochromatic_seed_product_order_zero",
                "color": int(product_record["color"]),
                "matching_index": int(
                    product_record["matching_index"]
                ),
                "positive_source_monomial": list(monomial),
                "negative_source_monomial": [],
                "nonzero_coefficients": _nonzero_entries(row, support),
            }
        )

    victims = hard_case["original_singleton_victims"]
    repair_indices = orbit["representative_repair_matching_indices"]
    if len(victims) != len(repair_indices):
        raise KrennN8ToricUnequalRateError(
            f"the {label} victim/repair count changed"
        )
    matchings = first_shell._primary_matchings()
    for victim, repair_index_value in zip(
        victims, repair_indices, strict=True
    ):
        repair_index = int(repair_index_value)
        coloring = tuple(map(int, victim["coloring"]))
        seed_monomial = tuple(
            map(int, victim["source_monomial_variable_indices"])
        )
        repair_monomial = first_shell._monomial_for_coloring(
            coloring, matchings[repair_index]
        )
        row = _sparse_row(
            positions,
            repair_monomial,
            seed_monomial,
        )
        if not any(row):
            raise KrennN8ToricUnequalRateError(
                f"a {label} repair tie became the zero relation"
            )
        rows.append(row)
        records.append(
            {
                "kind": "repair_order_tied_to_seed_victim",
                "coloring_index": int(victim["coloring_index"]),
                "coloring": list(coloring),
                "seed_matching_index": int(victim["matching_index"]),
                "repair_matching_index": repair_index,
                "positive_source_monomial": list(repair_monomial),
                "negative_source_monomial": list(seed_monomial),
                "nonzero_coefficients": _nonzero_entries(row, support),
            }
        )
    return tuple(rows), records



def _subtract_rows(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, ...]:
    if len(left) != len(right):
        raise KrennN8ToricUnequalRateError(
            "cannot subtract constraint rows of different lengths"
        )
    return tuple(
        int(first) - int(second)
        for first, second in zip(left, right, strict=True)
    )


def _matrix_vector_product(
    matrix: Sequence[Sequence[int]],
    vector: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        sum(
            int(value) * int(coordinate)
            for value, coordinate in zip(row, vector, strict=True)
        )
        for row in matrix
    )


def _unimodular_right_hand_side_witness(
    matrix: Sequence[Sequence[int]],
    right_hand_side: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, object]]:
    rows = tuple(tuple(map(int, row)) for row in matrix)
    target = tuple(map(int, right_hand_side))
    if not rows or len(rows) != len(target):
        raise KrennN8ToricUnequalRateError(
            "a unimodular witness system has the wrong height"
        )
    rank = first_shell._rank_over_q(rows)
    if rank != len(rows):
        raise KrennN8ToricUnequalRateError(
            "a unimodular witness system lacks full row rank"
        )
    minor_rows, minor_columns, determinant = first_shell._greedy_minor(
        rows, rank
    )
    if (
        minor_rows != tuple(range(rank))
        or abs(determinant) != 1
    ):
        raise KrennN8ToricUnequalRateError(
            "the q-surjectivity minor stopped being unimodular"
        )
    minor = tuple(
        tuple(rows[row][column] for column in minor_columns)
        for row in range(rank)
    )
    selected_solution = []
    for replaced_column in range(rank):
        replaced = tuple(
            tuple(
                target[row]
                if column == replaced_column
                else minor[row][column]
                for column in range(rank)
            )
            for row in range(rank)
        )
        numerator = first_shell._determinant(replaced)
        value = numerator / determinant
        if value.denominator != 1:
            raise KrennN8ToricUnequalRateError(
                "the q-surjectivity witness stopped being integral"
            )
        selected_solution.append(value.numerator)
    solution = [0] * len(rows[0])
    for column, value in zip(
        minor_columns, selected_solution, strict=True
    ):
        solution[column] = value
    result = tuple(solution)
    if _matrix_vector_product(rows, result) != target:
        raise KrennN8ToricUnequalRateError(
            "the displayed q-surjectivity witness does not replay"
        )
    return result, {
        "matrix_row_indices": list(minor_rows),
        "support_column_positions": list(minor_columns),
        "determinant": determinant,
    }


def _target_leading_chart_audit(
    label: str,
    hard_case: Mapping[str, object],
    orbit: Mapping[str, object],
    support: Sequence[int],
    tie_rows: Sequence[Sequence[int]],
    restricted_gauge: Sequence[Sequence[int]],
) -> dict[str, object]:
    expected = EXPECTED[label]
    support = tuple(map(int, support))
    positions = {index: position for position, index in enumerate(support)}
    primary = first_shell._active_term_signature(
        support, first_shell._primary_matchings()
    )
    independent = first_shell._active_term_signature(
        support, first_shell._independent_matchings()
    )
    if primary != independent:
        raise KrennN8ToricUnequalRateError(
            f"the {label} target-chart matching enumerators disagree"
        )
    matching_index = {
        matching: index
        for index, matching in enumerate(first_shell._primary_matchings())
    }
    target_terms = []
    for color in range(D):
        equation = first_shell.coloring_index(N, D, (color,) * N)
        terms = []
        for matching, monomial in primary.get(equation, ()):
            terms.append(
                {
                    "matching_index": matching_index[matching],
                    "monomial_variable_indices": list(map(int, monomial)),
                }
            )
        if not terms:
            raise KrennN8ToricUnequalRateError(
                f"the {label} skeleton lost target color {color}"
            )
        target_terms.append(tuple(terms))

    exact_two_term_victims = []
    for victim, repair_index_value in zip(
        hard_case["original_singleton_victims"],
        orbit["representative_repair_matching_indices"],
        strict=True,
    ):
        equation = int(victim["coloring_index"])
        repair_index = int(repair_index_value)
        repair_monomial = first_shell._monomial_for_coloring(
            victim["coloring"],
            first_shell._primary_matchings()[repair_index],
        )
        expected_monomials = {
            tuple(map(int, victim["source_monomial_variable_indices"])),
            tuple(map(int, repair_monomial)),
        }
        actual_monomials = {
            tuple(map(int, monomial))
            for _matching, monomial in primary.get(equation, ())
        }
        exact = (
            len(primary.get(equation, ())) == 2
            and actual_monomials == expected_monomials
        )
        exact_two_term_victims.append(
            {
                "equation": equation,
                "coloring": list(map(int, victim["coloring"])),
                "seed_monomial_variable_indices": list(
                    map(
                        int,
                        victim["source_monomial_variable_indices"],
                    )
                ),
                "repair_matching_index": repair_index,
                "repair_monomial_variable_indices": list(
                    map(int, repair_monomial)
                ),
                "exactly_these_two_support_terms": exact,
            }
        )
    if not all(
        record["exactly_these_two_support_terms"]
        for record in exact_two_term_victims
    ):
        raise KrennN8ToricUnequalRateError(
            f"a {label} original victim is not exactly two-term"
        )

    augmented_gauge = tuple(
        (*tuple(map(int, row)), 1) for row in restricted_gauge
    )
    augmented_rank = first_shell._rank_over_q(augmented_gauge)
    augmented_minor_rows, augmented_minor_columns, augmented_det = (
        first_shell._greedy_minor(augmented_gauge, augmented_rank)
    )
    if (
        augmented_rank != expected["projective_gauge_rank"]
        or abs(augmented_det) != 4
    ):
        raise KrennN8ToricUnequalRateError(
            f"the {label} projective gauge restriction changed"
        )

    affine_gauge_rank = first_shell._rank_over_q(restricted_gauge)
    _affine_minor_rows, _affine_minor_columns, affine_gauge_det = (
        first_shell._greedy_minor(
            restricted_gauge, affine_gauge_rank
        )
    )
    charts = []
    for chart_index, selection in enumerate(product(*target_terms)):
        target_rows = tuple(
            _sparse_row(
                positions, record["monomial_variable_indices"]
            )
            for record in selection
        )
        affine_constraints = (*target_rows, *tie_rows)
        projective_constraints = (
            _subtract_rows(target_rows[1], target_rows[0]),
            _subtract_rows(target_rows[2], target_rows[0]),
            *tie_rows,
        )
        affine_rank = first_shell._rank_over_q(affine_constraints)
        projective_rank = first_shell._rank_over_q(
            projective_constraints
        )
        affine_annihilation = first_shell._integer_matrix_product(
            affine_constraints, restricted_gauge
        )
        projective_annihilation = first_shell._integer_matrix_product(
            projective_constraints, augmented_gauge
        )
        q_system = (*projective_constraints, target_rows[0])
        q_target = (0,) * len(projective_constraints) + (1,)
        witness, q_minor = _unimodular_right_hand_side_witness(
            q_system, q_target
        )
        q_minor["source_coordinate_columns"] = [
            support[position]
            for position in q_minor["support_column_positions"]
        ]
        witness_entries = [
            {
                "support_position": position,
                "source_coordinate_index": support[position],
                "order": value,
            }
            for position, value in enumerate(witness)
            if value
        ]
        common_scaling = (1,) * len(support)
        q_of_common_scaling = sum(target_rows[0])
        common_scaling_projective_image = _matrix_vector_product(
            projective_constraints, common_scaling
        )
        recovered_affine_constraints = (
            target_rows[0],
            tuple(
                target_rows[0][column]
                + projective_constraints[0][column]
                for column in range(len(support))
            ),
            tuple(
                target_rows[0][column]
                + projective_constraints[1][column]
                for column in range(len(support))
            ),
            *projective_constraints[2:],
        )
        affine_kernel_equals_gauge_over_Z = (
            affine_gauge_rank
            == len(support) - affine_rank
            and abs(affine_gauge_det) == 1
            and all(
                value == 0
                for row in affine_annihilation
                for value in row
            )
        )
        exact_sequence_gives_Z_mod_4 = (
            recovered_affine_constraints == affine_constraints
            and not any(common_scaling_projective_image)
            and q_of_common_scaling == 4
            and abs(q_minor["determinant"]) == 1
            and affine_kernel_equals_gauge_over_Z
        )
        checks = {
            "affine_constraint_rank_exact": (
                affine_rank == expected["constraint_rank"]
            ),
            "projective_constraint_rank_exact": (
                projective_rank
                == expected["projective_constraint_rank"]
            ),
            "affine_constraints_annihilate_target_gauge": all(
                value == 0
                for row in affine_annihilation
                for value in row
            ),
            "projective_constraints_annihilate_augmented_gauge": all(
                value == 0
                for row in projective_annihilation
                for value in row
            ),
            "projective_ranks_are_complementary": (
                augmented_rank
                == len(support) - projective_rank
            ),
            "q_map_is_integrally_surjective": (
                abs(q_minor["determinant"]) == 1
            ),
            "q_witness_uses_only_zero_or_one": (
                set(witness).issubset({0, 1})
            ),
            "affine_kernel_equals_target_gauge_over_Z": (
                affine_kernel_equals_gauge_over_Z
            ),
            "projective_plus_q_rows_recover_affine_rows": (
                recovered_affine_constraints == affine_constraints
            ),
            "common_scaling_lies_in_projective_kernel": (
                not any(common_scaling_projective_image)
            ),
            "q_of_common_scaling_is_four": (
                q_of_common_scaling == 4
            ),
            "projective_quotient_is_Z_mod_4": (
                exact_sequence_gives_Z_mod_4
            ),
            "projective_rational_ray_count_is_zero": (
                augmented_rank == len(support) - projective_rank
            ),
        }
        if not all(checks.values()):
            failed = sorted(
                key for key, value in checks.items() if not value
            )
            raise KrennN8ToricUnequalRateError(
                f"the {label} orbit {orbit['orbit_index']} target chart "
                f"{chart_index} failed: {failed}"
            )
        charts.append(
            {
                "chart_index": chart_index,
                "selected_target_monomials": [
                    {
                        "color": color,
                        **deepcopy(record),
                    }
                    for color, record in enumerate(selection)
                ],
                "affine_order_constraint_matrix": {
                    "shape": [
                        len(affine_constraints),
                        len(support),
                    ],
                    "rank_over_Q": affine_rank,
                    "dense_matrix_sha256": _matrix_sha256(
                        affine_constraints
                    ),
                },
                "projective_order_constraint_matrix": {
                    "shape": [
                        len(projective_constraints),
                        len(support),
                    ],
                    "rank_over_Q": projective_rank,
                    "dense_matrix_sha256": _matrix_sha256(
                        projective_constraints
                    ),
                },
                "common_target_order_map": {
                    "q_is_selected_color_zero_target_order": True,
                    "surjective_over_Z": True,
                    "unimodular_minor": q_minor,
                    "q_equals_one_witness": witness_entries,
                },
                "projective_exact_sequence": {
                    "ker_q_equals_affine_kernel": (
                        recovered_affine_constraints
                        == affine_constraints
                    ),
                    "affine_kernel_equals_target_gauge_over_Z": (
                        affine_kernel_equals_gauge_over_Z
                    ),
                    "common_scaling_lies_in_projective_kernel": (
                        not any(common_scaling_projective_image)
                    ),
                    "q_of_common_scaling": q_of_common_scaling,
                    "q_is_surjective_over_Z": (
                        abs(q_minor["determinant"]) == 1
                    ),
                    "cokernel_is_Z_mod_4": (
                        exact_sequence_gives_Z_mod_4
                    ),
                },
                "projective_integral_quotient": {
                    "rank": 0,
                    "torsion_invariant_factors": [4],
                    "rational_nonzero_primitive_ray_count": 0,
                    "normalization_may_require_degree_four_base_change": (
                        True
                    ),
                },
                "exact_checks": checks,
            }
        )
    return {
        "active_target_monomial_counts_by_color": [
            len(terms) for terms in target_terms
        ],
        "target_leading_chart_count": len(charts),
        "target_terms": [
            [deepcopy(record) for record in terms]
            for terms in target_terms
        ],
        "original_victim_two_term_audit": exact_two_term_victims,
        "restricted_projective_gauge_matrix": {
            "shape": [len(support), GAUGE_DIMENSION + 1],
            "rank_over_Q": augmented_rank,
            "dense_matrix_sha256": _matrix_sha256(augmented_gauge),
            "displayed_maximal_minor": {
                "support_row_positions": list(augmented_minor_rows),
                "source_coordinate_rows": [
                    support[row] for row in augmented_minor_rows
                ],
                "gauge_and_common_scaling_columns": list(
                    augmented_minor_columns
                ),
                "determinant": augmented_det,
            },
        },
        "target_leading_charts": charts,
        "primary_independent_matching_enumerators_agree": True,
    }
def _orbit_lattice_record(
    label: str,
    hard_case: Mapping[str, object],
    orbit: Mapping[str, object],
    seed_normalization: Mapping[str, object],
) -> dict[str, object]:
    expected = EXPECTED[label]
    support = tuple(
        map(
            int,
            orbit["flat_valuation"]["order_zero_coordinate_indices"],
        )
    )
    constraints, constraint_records = _constraint_rows(
        label, hard_case, orbit, seed_normalization
    )
    gauge = first_shell.target_preserving_gauge_matrix()
    restricted_gauge = tuple(gauge[index] for index in support)
    constraint_rank = first_shell._rank_over_q(constraints)
    gauge_rank = first_shell._rank_over_q(restricted_gauge)
    annihilation = first_shell._integer_matrix_product(
        constraints, restricted_gauge
    )
    annihilation_is_zero = all(
        value == 0 for row in annihilation for value in row
    )
    minor_rows, minor_columns, determinant = first_shell._greedy_minor(
        restricted_gauge, gauge_rank
    )
    quotient_rank = (
        len(support) - constraint_rank - gauge_rank
    )
    flat_singletons = orbit["initial_form_replay"][
        "zero_target_singleton_initial_forms"
    ]
    target_chart_audit = _target_leading_chart_audit(
        label,
        hard_case,
        orbit,
        support,
        constraints[D:],
        restricted_gauge,
    )
    checks = {
        "support_size_exact": len(support)
        == expected["support_size"],
        "constraint_count_exact": len(constraints)
        == D + expected["victims"],
        "constraint_rank_exact": constraint_rank
        == expected["constraint_rank"],
        "gauge_rank_exact": gauge_rank == expected["gauge_rank"],
        "constraints_annihilate_restricted_gauge": (
            annihilation_is_zero
        ),
        "ranks_are_complementary": (
            gauge_rank == len(support) - constraint_rank
        ),
        "displayed_gauge_minor_is_unimodular": abs(determinant) == 1,
        "integral_quotient_rank_is_zero": quotient_rank == 0,
        "integral_quotient_torsion_is_trivial": abs(determinant) == 1,
        "flat_singleton_obstruction_is_present": bool(flat_singletons),
        "first_shell_primary_independent_replay_agrees": bool(
            orbit["initial_form_replay"][
                "primary_independent_matching_enumerators_agree"
            ]
        ),
        "all_target_leading_charts_have_zero_rational_quotient": all(
            chart["projective_integral_quotient"][
                "rational_nonzero_primitive_ray_count"
            ]
            == 0
            for chart in target_chart_audit["target_leading_charts"]
        ),
        "every_original_victim_is_exactly_two_term": all(
            record["exactly_these_two_support_terms"]
            for record in target_chart_audit[
                "original_victim_two_term_audit"
            ]
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricUnequalRateError(
            f"the {label} orbit {orbit['orbit_index']} lattice checks "
            f"failed: {failed}"
        )
    return {
        "orbit_index": int(orbit["orbit_index"]),
        "raw_decorated_orbit_size": int(
            orbit["raw_decorated_orbit_size"]
        ),
        "support": list(support),
        "support_size": len(support),
        "repair_matching_indices": list(
            map(
                int,
                orbit["representative_repair_matching_indices"],
            )
        ),
        "order_constraint_matrix": {
            "shape": [len(constraints), len(support)],
            "rank_over_Q": constraint_rank,
            "dense_matrix_sha256": _matrix_sha256(constraints),
            "rows": constraint_records,
        },
        "target_leading_chart_audit": target_chart_audit,
        "restricted_gauge_matrix": {
            "shape": [len(support), GAUGE_DIMENSION],
            "rank_over_Q": gauge_rank,
            "dense_matrix_sha256": _matrix_sha256(restricted_gauge),
            "unimodular_maximal_minor": {
                "support_row_positions": list(minor_rows),
                "source_coordinate_rows": [
                    support[row] for row in minor_rows
                ],
                "gauge_columns": list(minor_columns),
                "determinant": determinant,
            },
        },
        "local_integral_quotient": {
            "rank": quotient_rank,
            "torsion_invariant_factors": [],
            "primitive_nonzero_skeleton_ray_count": 0,
            "affine_target_order_zero_root_extraction_required": False,
            "every_balanced_skeleton_valuation_is_pure_gauge": True,
        },
        "flat_initial_obstruction_reused": {
            "zero_target_singleton_count": int(
                orbit["initial_form_replay"][
                    "zero_target_singleton_count"
                ]
            ),
            "first_singleton": deepcopy(flat_singletons[0]),
            "gauge_preserves_term_order_differences_within_equation": (
                True
            ),
        },
        "exact_checks": checks,
    }


def _hard_case_record(
    label: str,
    first_certificate: Mapping[str, object],
) -> dict[str, object]:
    expected = EXPECTED[label]
    hard_case = first_certificate["flat_first_shell"]["hard_cases"][
        label
    ]
    seed_normalization = first_certificate["gauge_quotient"][
        "seed_normalizations"
    ][label]
    orbit_records = [
        _orbit_lattice_record(
            label, hard_case, orbit, seed_normalization
        )
        for orbit in hard_case["orbit_records"]
    ]
    determinant_histogram = Counter(
        record["restricted_gauge_matrix"][
            "unimodular_maximal_minor"
        ]["determinant"]
        for record in orbit_records
    )
    raw_covered = sum(
        record["raw_decorated_orbit_size"]
        for record in orbit_records
    )
    target_count_histogram = Counter(
        tuple(
            record["target_leading_chart_audit"][
                "active_target_monomial_counts_by_color"
            ]
        )
        for record in orbit_records
    )
    target_chart_count = sum(
        record["target_leading_chart_audit"][
            "target_leading_chart_count"
        ]
        for record in orbit_records
    )
    projective_determinant_histogram = Counter(
        record["target_leading_chart_audit"][
            "restricted_projective_gauge_matrix"
        ]["displayed_maximal_minor"]["determinant"]
        for record in orbit_records
    )
    q_minor_histogram = Counter(
        chart["common_target_order_map"]["unimodular_minor"][
            "determinant"
        ]
        for record in orbit_records
        for chart in record["target_leading_chart_audit"][
            "target_leading_charts"
        ]
    )
    q_witness_nnz_histogram = Counter(
        len(
            chart["common_target_order_map"][
                "q_equals_one_witness"
            ]
        )
        for record in orbit_records
        for chart in record["target_leading_chart_audit"][
            "target_leading_charts"
        ]
    )
    checks = {
        "decorated_orbit_count_exact": len(orbit_records)
        == expected["decorated_orbits"],
        "raw_decorations_covered_exact": raw_covered
        == expected["raw_decorations"],
        "determinant_histogram_exact": (
            dict(sorted(determinant_histogram.items()))
            == expected["determinant_histogram"]
        ),
        "every_local_quotient_has_rank_zero": all(
            record["local_integral_quotient"]["rank"] == 0
            for record in orbit_records
        ),
        "projective_determinant_histogram_exact": (
            dict(sorted(projective_determinant_histogram.items()))
            == expected["projective_determinant_histogram"]
        ),
        "every_local_quotient_has_trivial_torsion": all(
            not record["local_integral_quotient"][
                "torsion_invariant_factors"
            ]
            for record in orbit_records
        ),
        "every_balanced_rate_is_pure_gauge": all(
            record["local_integral_quotient"][
                "every_balanced_skeleton_valuation_is_pure_gauge"
            ]
            for record in orbit_records
        ),
        "every_flat_obstruction_survives_gauge_normalization": all(
            record["flat_initial_obstruction_reused"][
                "zero_target_singleton_count"
            ]
            > 0
            for record in orbit_records
        ),
        "target_count_triple_histogram_exact": (
            dict(sorted(target_count_histogram.items()))
            == expected["target_count_triple_histogram"]
        ),
        "target_leading_chart_count_exact": (
            target_chart_count == expected["target_chart_count"]
        ),
        "q_surjectivity_minor_histogram_exact": (
            dict(sorted(q_minor_histogram.items()))
            == expected["q_minor_histogram"]
        ),
        "q_witness_nnz_histogram_exact": (
            dict(sorted(q_witness_nnz_histogram.items()))
            == expected["q_witness_nnz_histogram"]
        ),
        "every_projective_chart_has_Z_mod_4_quotient": all(
            chart["projective_integral_quotient"][
                "torsion_invariant_factors"
            ]
            == [4]
            for record in orbit_records
            for chart in record["target_leading_chart_audit"][
                "target_leading_charts"
            ]
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricUnequalRateError(
            f"the {label} unequal-rate summary failed: {failed}"
        )
    return {
        "label": label,
        "raw_decorations_covered": raw_covered,
        "decorated_stabilizer_orbits": len(orbit_records),
        "support_size": expected["support_size"],
        "order_constraint_rank": expected["constraint_rank"],
        "restricted_gauge_rank": expected["gauge_rank"],
        "local_quotient_rank": 0,
        "local_quotient_torsion_invariant_factors": [],
        "target_active_term_count_triple_histogram": {
            ",".join(map(str, key)): value
            for key, value in sorted(target_count_histogram.items())
        },
        "target_leading_chart_count": target_chart_count,
        "projective_integral_quotient_torsion_invariant_factors": [4],
        "primitive_nonzero_skeleton_rational_quotient_rays": 0,
        "q_surjectivity_minor_determinant_histogram": {
            str(key): value
            for key, value in sorted(q_minor_histogram.items())
        },
        "projective_gauge_minor_determinant_histogram": {
            str(key): value
            for key, value in sorted(
                projective_determinant_histogram.items()
            )
        },
        "q_witness_nonzero_count_histogram": {
            str(key): value
            for key, value in sorted(q_witness_nnz_histogram.items())
        },
        "unimodular_minor_determinant_histogram": {
            str(key): value
            for key, value in sorted(determinant_histogram.items())
        },
        "orbit_records": orbit_records,
        "exact_checks": checks,
    }



def _restricted_difference_row(
    coordinates: Sequence[int],
    positive: Sequence[int],
    negative: Sequence[int],
) -> tuple[int, ...]:
    coefficients: Counter[int] = Counter(map(int, positive))
    coefficients.subtract(map(int, negative))
    return tuple(coefficients[int(index)] for index in coordinates)


def _minimum_new_coordinate_repairs(
    coloring: Sequence[int],
    support: Sequence[int],
) -> tuple[dict[str, object], ...]:
    support_set = set(map(int, support))
    candidates = []
    positive_costs = []
    for matching_index, matching in enumerate(
        first_shell._primary_matchings()
    ):
        monomial = first_shell._monomial_for_coloring(
            coloring, matching
        )
        new_coordinates = tuple(
            sorted(set(monomial).difference(support_set))
        )
        if new_coordinates:
            positive_costs.append(len(new_coordinates))
        candidates.append(
            {
                "matching_index": matching_index,
                "monomial_variable_indices": tuple(monomial),
                "new_coordinate_indices": new_coordinates,
            }
        )
    if not positive_costs:
        raise KrennN8ToricUnequalRateError(
            "a spill equation has no outside repair"
        )
    minimum = min(positive_costs)
    return tuple(
        candidate
        for candidate in candidates
        if len(candidate["new_coordinate_indices"]) == minimum
    )


def _depth_two_branch_lattice(
    seed_support: Sequence[int],
    parent_support: Sequence[int],
    tie_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    support = tuple(
        sorted(
            set(map(int, parent_support)).union(
                *(
                    set(map(int, record["positive_source_monomial"]))
                    for record in tie_records
                )
            )
        )
    )
    seed_set = set(map(int, seed_support))
    nonseed = tuple(index for index in support if index not in seed_set)
    tie_matrix = tuple(
        _restricted_difference_row(
            nonseed,
            record["positive_source_monomial"],
            record["negative_source_monomial"],
        )
        for record in tie_records
    )
    tie_rank = first_shell._rank_over_q(tie_matrix)
    gauge = first_shell.target_preserving_gauge_matrix()
    seed_gauge_rank = first_shell._rank_over_q(
        tuple(gauge[index] for index in seed_support)
    )
    support_gauge_rank = first_shell._rank_over_q(
        tuple(gauge[index] for index in support)
    )
    residual_gauge_rank = support_gauge_rank - seed_gauge_rank
    quotient_dimension = (
        len(nonseed) - tie_rank - residual_gauge_rank
    )
    tie_characters = []
    for record in tie_records:
        character = tuple(
            sum(
                gauge[index][column]
                for index in record["positive_source_monomial"]
            )
            - sum(
                gauge[index][column]
                for index in record["negative_source_monomial"]
            )
            for column in range(GAUGE_DIMENSION)
        )
        tie_characters.append(character)
    checks = {
        "seed_gauge_rank_is_nine": seed_gauge_rank == 9,
        "all_ties_are_full_gauge_invariant": all(
            not any(character) for character in tie_characters
        ),
        "residual_gauge_fits_tie_kernel": (
            residual_gauge_rank <= len(nonseed) - tie_rank
        ),
        "quotient_dimension_is_nonnegative": quotient_dimension >= 0,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricUnequalRateError(
            f"a depth-two branch lattice failed: {failed}"
        )
    return {
        "support": support,
        "nonseed": nonseed,
        "tie_matrix": tie_matrix,
        "tie_rank": tie_rank,
        "support_gauge_rank": support_gauge_rank,
        "residual_gauge_rank": residual_gauge_rank,
        "quotient_dimension": quotient_dimension,
        "tie_characters": tuple(tie_characters),
        "exact_checks": checks,
    }


def _depth_two_full_decoration(
    hard_case: Mapping[str, object],
    parent: Mapping[str, object],
    spills: Sequence[Mapping[str, object]],
    secondary_options: Sequence[Mapping[str, object]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    decoration = []
    for victim, matching_index_value in zip(
        hard_case["original_singleton_victims"],
        parent["representative_repair_matching_indices"],
        strict=True,
    ):
        matching_index = int(matching_index_value)
        decoration.append(
            (
                tuple(map(int, victim["coloring"])),
                first_shell._monomial_for_coloring(
                    victim["coloring"],
                    first_shell._primary_matchings()[matching_index],
                ),
            )
        )
    for spill, option in zip(spills, secondary_options, strict=True):
        decoration.append(
            (
                tuple(map(int, spill["coloring"])),
                tuple(map(int, option["monomial_variable_indices"])),
            )
        )
    return tuple(decoration)


def _valuation_initial_signature(
    orders: Sequence[int],
    matching_enumerator: Sequence[Sequence[tuple[int, int]]],
) -> tuple[
    tuple[
        int,
        tuple[
            tuple[tuple[tuple[int, int], ...], tuple[int, ...]],
            ...,
        ],
    ],
    ...,
]:
    orders = tuple(map(int, orders))
    if len(orders) != SOURCE_VARIABLES:
        raise KrennN8ToricUnequalRateError(
            "a depth-two valuation has the wrong source width"
        )
    matchings = tuple(
        first_shell._canonical_matching(matching)
        for matching in matching_enumerator
    )
    signature = []
    for equation in range(3**N):
        coloring = first_shell.coloring_from_index(N, D, equation)
        terms = []
        for matching in matchings:
            monomial = first_shell._monomial_for_coloring(
                coloring, matching
            )
            order = sum(orders[index] for index in monomial)
            terms.append((order, matching, tuple(sorted(monomial))))
        minimum = min(record[0] for record in terms)
        minima = tuple(
            sorted(
                (matching, monomial)
                for order, matching, monomial in terms
                if order == minimum
            )
        )
        signature.append((minimum, minima))
    return tuple(signature)


def _positive_singleton_circuit_replay(
    label: str,
    support: Sequence[int],
    nonseed: Sequence[int],
    tie_matrix: Sequence[Sequence[int]],
) -> dict[str, object]:
    """Replay a positive singleton incidence circuit exactly over Q.

    Seed orders have already been normalized to zero in the depth-two
    lattice.  A positive combination of singleton-monomial incidence rows
    that lies in the span of the tie rows therefore has total order zero on
    every point of the tie lattice.  Positivity forces at least one of the
    singleton orders to be nonpositive, independently of the chosen
    quotient representative or residual-gauge lift.
    """

    expected = DEPTH_TWO_EXPECTED["classes"][label]
    coefficients = {
        int(equation): int(coefficient)
        for equation, coefficient in expected[
            "positive_singleton_circuit"
        ].items()
    }
    support = tuple(map(int, support))
    nonseed = tuple(map(int, nonseed))
    ties = tuple(tuple(map(int, row)) for row in tie_matrix)
    if not coefficients or any(value <= 0 for value in coefficients.values()):
        raise KrennN8ToricUnequalRateError(
            f"depth-two class {label} lacks a positive circuit"
        )
    if any(len(row) != len(nonseed) for row in ties):
        raise KrennN8ToricUnequalRateError(
            f"depth-two class {label} has a malformed tie matrix"
        )

    primary = first_shell._active_term_signature(
        support, first_shell._primary_matchings()
    )
    independent = first_shell._active_term_signature(
        support, first_shell._independent_matchings()
    )
    if primary != independent:
        raise KrennN8ToricUnequalRateError(
            f"depth-two class {label} singleton enumerators disagree"
        )

    records = []
    weighted_incidence = [0] * len(nonseed)
    nonseed_positions = {
        coordinate: position
        for position, coordinate in enumerate(nonseed)
    }
    target_equations = {
        first_shell.coloring_index(N, D, (color,) * N)
        for color in range(D)
    }
    target_expected = DEPTH_TWO_EXPECTED[
        "target_term_tie_row_span_witnesses"
    ]
    target_records = []
    target_tie_rank = first_shell._rank_over_q(ties)
    for equation in sorted(target_equations):
        primary_terms = primary.get(equation, ())
        independent_terms = independent.get(equation, ())
        expected_terms = target_expected.get(equation, {})
        observed_monomials = {
            tuple(monomial)
            for _matching, monomial in primary_terms
        }
        if (
            not primary_terms
            or primary_terms != independent_terms
            or observed_monomials != set(expected_terms)
        ):
            raise KrennN8ToricUnequalRateError(
                f"depth-two class {label} target equation {equation} "
                "failed its exact support-term census"
            )
        for matching, monomial in primary_terms:
            incidence = tuple(
                sum(1 for value in monomial if value == coordinate)
                for coordinate in nonseed
            )
            witness = tuple(
                map(
                    int,
                    expected_terms[tuple(monomial)],
                )
            )
            reconstructed = tuple(
                sum(
                    coefficient * row[column]
                    for coefficient, row in zip(
                        witness, ties, strict=True
                    )
                )
                for column in range(len(nonseed))
            )
            augmented_rank = first_shell._rank_over_q(
                (*ties, incidence)
            )
            if (
                len(witness) != len(ties)
                or reconstructed != incidence
                or augmented_rank != target_tie_rank
            ):
                raise KrennN8ToricUnequalRateError(
                    f"depth-two class {label} target equation "
                    f"{equation} left the tie-row span"
                )
            target_records.append(
                {
                    "equation": equation,
                    "coloring": list(
                        first_shell.coloring_from_index(N, D, equation)
                    ),
                    "primary_support_term_count": len(primary_terms),
                    "independent_support_term_count": len(independent_terms),
                    "matching": [list(edge) for edge in matching],
                    "monomial_variable_indices": list(monomial),
                    "nonseed_incidence_row": list(incidence),
                    "tie_row_span_witness_coefficients": list(witness),
                    "tie_row_span_reconstruction": list(reconstructed),
                    "augmented_row_rank_over_Q": augmented_rank,
                    "enumerators_agree_exactly": True,
                    "order_identically_zero_on_tie_lattice": True,
                }
            )
    for equation, coefficient in sorted(coefficients.items()):
        primary_terms = primary.get(equation, ())
        independent_terms = independent.get(equation, ())
        if (
            len(primary_terms) != 1
            or len(independent_terms) != 1
            or primary_terms != independent_terms
        ):
            raise KrennN8ToricUnequalRateError(
                f"depth-two class {label} equation {equation} is not "
                "an enumerator-independent support singleton"
            )
        matching, monomial = primary_terms[0]
        incidence = [0] * len(nonseed)
        for coordinate in monomial:
            position = nonseed_positions.get(int(coordinate))
            if position is not None:
                incidence[position] += 1
        for position, value in enumerate(incidence):
            weighted_incidence[position] += coefficient * value
        coloring = first_shell.coloring_from_index(N, D, equation)
        records.append(
            {
                "equation": equation,
                "coloring": list(coloring),
                "positive_coefficient": coefficient,
                "primary_support_term_count": 1,
                "independent_support_term_count": 1,
                "matching": [list(edge) for edge in matching],
                "monomial_variable_indices": list(monomial),
                "nonseed_incidence_row": incidence,
                "equation_is_mixed": equation not in target_equations,
                "enumerators_agree_exactly": True,
            }
        )

    weighted_incidence = tuple(weighted_incidence)
    witness = tuple(map(int, expected["tie_row_span_witness"]))
    if len(witness) != len(ties):
        raise KrennN8ToricUnequalRateError(
            f"depth-two class {label} tie-span witness has wrong height"
        )
    reconstructed = tuple(
        sum(
            coefficient * row[column]
            for coefficient, row in zip(witness, ties, strict=True)
        )
        for column in range(len(nonseed))
    )
    tie_rank = first_shell._rank_over_q(ties)
    augmented_rank = first_shell._rank_over_q(
        (*ties, weighted_incidence)
    )
    checks = {
        "primary_independent_support_signatures_agree": True,
        "all_circuit_coefficients_are_positive": all(
            value > 0 for value in coefficients.values()
        ),
        "all_circuit_equations_are_mixed": all(
            record["equation_is_mixed"] for record in records
        ),
        "every_circuit_equation_is_a_support_singleton": all(
            record["primary_support_term_count"] == 1
            and record["independent_support_term_count"] == 1
            and record["enumerators_agree_exactly"]
            for record in records
        ),
        "active_target_support_term_counts_are_2_1_1": (
            Counter(record["equation"] for record in target_records)
            == Counter({0: 2, 3_280: 1, 6_560: 1})
        ),
        "all_active_target_terms_enumerator_independent": all(
            record["enumerators_agree_exactly"]
            for record in target_records
        ),
        "all_active_target_incidence_rows_in_tie_row_span": all(
            record["nonseed_incidence_row"]
            == record["tie_row_span_reconstruction"]
            and record["augmented_row_rank_over_Q"] == target_tie_rank
            for record in target_records
        ),
        "all_active_target_orders_identically_zero_on_tie_lattice": all(
            record["order_identically_zero_on_tie_lattice"]
            for record in target_records
        ),
        "weighted_nonseed_incidence_row_exact": (
            weighted_incidence
            == tuple(expected["weighted_nonseed_incidence"])
        ),
        "tie_row_span_witness_reconstructs_incidence": (
            reconstructed == weighted_incidence
        ),
        "augmented_Q_rank_equals_tie_Q_rank": (
            augmented_rank == tie_rank
        ),
        "tie_Q_rank_is_five": tie_rank == 5,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricUnequalRateError(
            f"depth-two class {label} positive circuit failed: {failed}"
        )
    return {
        "proof_kind": "positive-singleton-tie-row-span-circuit",
        "equation_coefficients": {
            str(key): value for key, value in sorted(coefficients.items())
        },
        "singleton_equations": records,
        "active_monochromatic_target_zero_order_audit": {
            "common_active_target_monomial_order_normalized_to_zero": True,
            "declared_scope_requires_no_outside_monomial_entry": True,
            "target_support_term_counts": {"0": 2, "3280": 1, "6560": 1},
            "support_terms": target_records,
            "tie_row_rank_over_Q": target_tie_rank,
            "all_active_target_orders_identically_zero_on_tie_lattice": all(
                record["order_identically_zero_on_tie_lattice"]
                for record in target_records
            ),
            "consequence_in_declared_stratum": (
                "every active target support monomial has order zero, so "
                "the target coordinate valuation is at least zero (or "
                "infinite); cancellation only strengthens the comparison"
            ),
        },
        "weighted_nonseed_incidence_row": list(weighted_incidence),
        "tie_row_span_witness_coefficients": list(witness),
        "tie_row_span_reconstruction": list(reconstructed),
        "tie_row_rank_over_Q": tie_rank,
        "augmented_row_rank_over_Q": augmented_rank,
        "identity_on_tie_lattice": (
            "the positive weighted sum of the recorded singleton "
            "monomial orders is zero"
        ),
        "exclusion_scope": (
            "declared-exact-support-or-strictly-higher-outside-monomial-stratum"
        ),
        "entire_declared_quotient_line_and_all_residual_gauge_lifts_excluded": True,
        "outside_term_entry_strata_excluded": False,
        "exact_checks": checks,
    }

def _depth_two_ray_replay(

    label: str,
    support: Sequence[int],
    nonseed: Sequence[int],
    ray: Sequence[int],
    sign: int,
) -> dict[str, object]:
    if sign not in (-1, 1):
        raise KrennN8ToricUnequalRateError(
            "a quotient-ray orientation must be +1 or -1"
        )
    guard_order = 8
    orders = [guard_order] * SOURCE_VARIABLES
    for index in support:
        orders[int(index)] = 0
    for index, value in zip(nonseed, ray, strict=True):
        orders[int(index)] = sign * int(value)
    primary = _valuation_initial_signature(
        orders, first_shell._primary_matchings()
    )
    independent = _valuation_initial_signature(
        orders, first_shell._independent_matchings()
    )
    if primary != independent:
        raise KrennN8ToricUnequalRateError(
            f"the {label}{'+' if sign > 0 else '-'} valuation "
            "enumerators disagree"
        )
    targets = {
        first_shell.coloring_index(N, D, (color,) * N): color
        for color in range(D)
    }
    target_records = []
    unique_mixed_histogram: Counter[int] = Counter()
    obstructing = []
    harmless = []
    all_minimum_histogram: Counter[tuple[int, int]] = Counter()
    for equation, (minimum, minima) in enumerate(primary):
        all_minimum_histogram[(minimum, len(minima))] += 1
        if equation in targets:
            target_records.append(
                {
                    "color": targets[equation],
                    "equation": equation,
                    "minimum_order": minimum,
                    "minimum_term_count": len(minima),
                }
            )
            continue
        if len(minima) != 1:
            continue
        unique_mixed_histogram[minimum] += 1
        matching, monomial = minima[0]
        record = {
            "equation": equation,
            "coloring": list(
                first_shell.coloring_from_index(N, D, equation)
            ),
            "minimum_order": minimum,
            "matching": [list(edge) for edge in matching],
            "monomial_variable_indices": list(monomial),
        }
        if minimum <= 0:
            obstructing.append(record)
        else:
            harmless.append(record)
    target_records.sort(key=lambda record: record["color"])
    expected_histogram = DEPTH_TWO_EXPECTED["classes"][label][
        "ray_obstruction_histograms"
    ]["+" if sign > 0 else "-"]
    near_target_unique_histogram = {
        order: count
        for order, count in unique_mixed_histogram.items()
        if order <= 1
    }
    checks = {
        "primary_independent_enumerators_agree": True,
        "all_6561_equations_accounted": (
            sum(all_minimum_histogram.values()) == 3**N
        ),
        "target_minimum_orders_are_common_zero": (
            [record["minimum_order"] for record in target_records]
            == [0, 0, 0]
        ),
        "target_minimum_term_counts_are_2_1_1": (
            [record["minimum_term_count"] for record in target_records]
            == [2, 1, 1]
        ),
        "near_target_unique_mixed_minimum_histogram_exact": (
            dict(sorted(near_target_unique_histogram.items()))
            == expected_histogram
        ),
        "at_least_one_noncancellable_mixed_term_at_target_scale": (
            bool(obstructing)
        ),
        "outside_guard_is_least_safe_integer_for_unit_inner_orders": (
            guard_order == 8
            and guard_order - 3 > 4
            and (guard_order - 1) - 3 <= 4
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricUnequalRateError(
            f"the {label}{'+' if sign > 0 else '-'} ray replay "
            f"failed: {failed}"
        )
    return {
        "orientation": "+" if sign > 0 else "-",
        "outside_coordinate_guard_order": guard_order,
        "support_coordinate_orders": [
            {
                "source_coordinate_index": int(index),
                "order": orders[int(index)],
            }
            for index in support
        ],
        "target_minima": target_records,
        "unique_mixed_minimum_order_histogram": {
            str(key): value
            for key, value in sorted(unique_mixed_histogram.items())
        },
        "near_target_unique_mixed_minimum_order_histogram": {
            str(key): value
            for key, value in sorted(near_target_unique_histogram.items())
        },
        "obstructing_unique_mixed_minima_at_order_at_most_target": (
            obstructing
        ),
        "obstructing_count": len(obstructing),
        "harmless_count": len(harmless),
        "all_equation_minimum_order_and_count_histogram": {
            f"{order},{count}": value
            for (order, count), value in sorted(
                all_minimum_histogram.items()
            )
        },
        "status": "diagnostic-canonical-lift-replay",
        "canonical_lift_diagnostic_only": True,
        "chosen_lift_has_noncancellable_mixed_minimum": bool(obstructing),
        "quotient_ray_exclusion_claimed_from_this_replay": False,
        "absolute_orders_are_residual_gauge_invariant": False,
        "exclusion_proof_basis": None,
        "exact_checks": checks,
    }


def _build_depth_two_gate(
    first_certificate: Mapping[str, object],
) -> dict[str, object]:
    expected = DEPTH_TWO_EXPECTED
    hard_case = first_certificate["flat_first_shell"]["hard_cases"][
        "H5"
    ]
    parent = hard_case["orbit_records"][expected["parent_orbit_index"]]
    parent_support = tuple(
        map(
            int,
            parent["flat_valuation"]["order_zero_coordinate_indices"],
        )
    )
    spills = tuple(
        sorted(
            parent["initial_form_replay"][
                "zero_target_singleton_initial_forms"
            ],
            key=lambda record: int(record["equation"]),
        )
    )
    if (
        tuple(parent["representative_repair_matching_indices"])
        != expected["parent_repair_matching_indices"]
        or parent_support != expected["parent_support"]
        or tuple(int(record["equation"]) for record in spills)
        != expected["spill_equations"]
    ):
        raise KrennN8ToricUnequalRateError(
            "the deterministic H5 depth-two parent changed"
        )
    secondary_tables = tuple(
        _minimum_new_coordinate_repairs(
            spill["coloring"], parent_support
        )
        for spill in spills
    )
    if any(
        len(table) != expected["repair_options_per_spill"]
        or any(
            len(option["new_coordinate_indices"]) != 2
            for option in table
        )
        for table in secondary_tables
    ):
        raise KrennN8ToricUnequalRateError(
            "the depth-two cost-two repair table changed"
        )

    seed_normalization = first_certificate["gauge_quotient"][
        "seed_normalizations"
    ]["H5"]
    seed_support = tuple(map(int, seed_normalization["seed_support"]))
    _parent_constraints, parent_constraint_records = _constraint_rows(
        "H5", hard_case, parent, seed_normalization
    )
    original_ties = [
        {
            "kind": "original_victim_tie",
            "positive_source_monomial": tuple(
                map(int, record["positive_source_monomial"])
            ),
            "negative_source_monomial": tuple(
                map(int, record["negative_source_monomial"])
            ),
        }
        for record in parent_constraint_records[D:]
    ]

    branch_histogram: Counter[tuple[int, int]] = Counter()
    positive_branches = []
    cache: dict[object, dict[str, object]] = {}
    raw_count = 0
    for secondary_options in product(*secondary_tables):
        raw_count += 1
        secondary_ties = [
            {
                "kind": "spill_repair_tie",
                "equation": int(spill["equation"]),
                "positive_source_monomial": tuple(
                    map(int, option["monomial_variable_indices"])
                ),
                "negative_source_monomial": tuple(
                    map(int, spill["monomial_variable_indices"])
                ),
            }
            for spill, option in zip(
                spills, secondary_options, strict=True
            )
        ]
        tie_records = (*original_ties, *secondary_ties)
        support_key = tuple(
            sorted(
                set(parent_support).union(
                    *(
                        set(record["positive_source_monomial"])
                        for record in secondary_ties
                    )
                )
            )
        )
        matrix_key = tuple(
            _restricted_difference_row(
                tuple(
                    index
                    for index in support_key
                    if index not in set(seed_support)
                ),
                record["positive_source_monomial"],
                record["negative_source_monomial"],
            )
            for record in tie_records
        )
        key = (support_key, matrix_key)
        lattice = cache.get(key)
        if lattice is None:
            lattice = _depth_two_branch_lattice(
                seed_support, parent_support, tie_records
            )
            cache[key] = lattice
        support_size = len(lattice["support"])
        quotient_dimension = int(lattice["quotient_dimension"])
        branch_histogram[(support_size, quotient_dimension)] += 1
        if quotient_dimension > 0:
            positive_branches.append(
                {
                    "secondary_matching_indices": tuple(
                        int(option["matching_index"])
                        for option in secondary_options
                    ),
                    "secondary_options": tuple(secondary_options),
                    "tie_records": tie_records,
                    "lattice": lattice,
                    "decoration": _depth_two_full_decoration(
                        hard_case, parent, spills, secondary_options
                    ),
                }
            )
    if raw_count != expected["raw_branch_count"] or not positive_branches:
        raise KrennN8ToricUnequalRateError(
            "the bounded depth-two branch count changed"
        )
    minimum_positive_support_size = min(
        len(record["lattice"]["support"])
        for record in positive_branches
    )
    minimum_positive = [
        record
        for record in positive_branches
        if len(record["lattice"]["support"])
        == minimum_positive_support_size
    ]
    observed_tuples = {
        record["secondary_matching_indices"] for record in minimum_positive
    }
    expected_tuples = {
        tuple(class_record[key])
        for class_record in expected["classes"].values()
        for key in ("representative", "partner")
    }
    if (
        minimum_positive_support_size
        != expected["minimum_positive_quotient_support_size"]
        or len(minimum_positive)
        != expected["positive_minimum_support_branches"]
        or observed_tuples != expected_tuples
    ):
        raise KrennN8ToricUnequalRateError(
            "the minimum positive depth-two quotient layer changed"
        )

    canonical_groups: dict[object, list[dict[str, object]]] = {}
    for record in minimum_positive:
        key = first_shell._canonical_decoration(
            "H5", record["decoration"]
        )
        canonical_groups.setdefault(key, []).append(record)
    if (
        len(canonical_groups) != 2
        or sorted(map(len, canonical_groups.values())) != [2, 2]
    ):
        raise KrennN8ToricUnequalRateError(
            "the depth-two symmetry classes changed"
        )

    class_records = {}
    gauge = first_shell.target_preserving_gauge_matrix()
    for class_label, class_expected in expected["classes"].items():
        representative = next(
            record
            for record in minimum_positive
            if record["secondary_matching_indices"]
            == class_expected["representative"]
        )
        partner = next(
            record
            for record in minimum_positive
            if record["secondary_matching_indices"]
            == class_expected["partner"]
        )
        if first_shell._canonical_decoration(
            "H5", representative["decoration"]
        ) != first_shell._canonical_decoration(
            "H5", partner["decoration"]
        ):
            raise KrennN8ToricUnequalRateError(
                f"depth-two class {class_label} partners split"
            )
        lattice = representative["lattice"]
        support = tuple(lattice["support"])
        nonseed = tuple(lattice["nonseed"])
        tie_matrix = tuple(lattice["tie_matrix"])
        ray = tuple(class_expected["ray"])
        detector = {
            int(index): int(value)
            for index, value in class_expected["detector"].items()
        }
        detector_character = tuple(
            sum(
                coefficient * gauge[index][column]
                for index, coefficient in detector.items()
            )
            for column in range(GAUGE_DIMENSION)
        )
        detector_value = sum(
            detector.get(index, 0) * value
            for index, value in zip(nonseed, ray, strict=True)
        )
        orbit_images = {
            first_shell._decoration_key(
                representative["decoration"], vertices, colors
            )
            for vertices, colors in first_shell._seed_stabilizer("H5")
        }
        class_checks = {
            "support_exact": support == class_expected["support"],
            "nonseed_coordinates_exact": (
                nonseed == class_expected["nonseed"]
            ),
            "tie_rank_is_five": lattice["tie_rank"] == 5,
            "residual_gauge_rank_is_four": (
                lattice["residual_gauge_rank"] == 4
            ),
            "quotient_dimension_is_one": (
                lattice["quotient_dimension"] == 1
            ),
            "ray_lies_in_tie_kernel": not any(
                _matrix_vector_product(tie_matrix, ray)
            ),
            "detector_is_full_gauge_invariant": (
                not any(detector_character)
            ),
            "detector_evaluates_to_one": detector_value == 1,
            "decoration_orbit_size_is_four": len(orbit_images) == 4,
            "complete_decoration_stabilizer_is_trivial": (
                len(first_shell._seed_stabilizer("H5"))
                // len(orbit_images)
                == 1
            ),
        }
        if not all(class_checks.values()):
            failed = sorted(
                key for key, value in class_checks.items() if not value
            )
            raise KrennN8ToricUnequalRateError(
                f"depth-two class {class_label} failed: {failed}"
            )
        circuit = _positive_singleton_circuit_replay(
            class_label,
            support,
            nonseed,
            tie_matrix,
        )
        rays = {
            orientation: _depth_two_ray_replay(
                class_label,
                support,
                nonseed,
                ray,
                sign,
            )
            for orientation, sign in (("+", 1), ("-", -1))
        }
        class_records[class_label] = {
            "representative_secondary_matching_indices": list(
                class_expected["representative"]
            ),
            "fixed_chart_partner_matching_indices": list(
                class_expected["partner"]
            ),
            "support": list(support),
            "support_size": len(support),
            "nonseed_coordinates": list(nonseed),
            "tie_matrix": [list(row) for row in tie_matrix],
            "tie_matrix_rank": lattice["tie_rank"],
            "residual_gauge_rank": lattice["residual_gauge_rank"],
            "quotient_dimension": lattice["quotient_dimension"],
            "primitive_ray_lift_on_nonseed_coordinates": list(ray),
            "primitive_full_gauge_invariant_detector": {
                str(index): value
                for index, value in sorted(detector.items())
            },
            "detector_value_on_positive_ray": detector_value,
            "complete_decoration_orbit_size": len(orbit_images),
            "complete_decoration_stabilizer_order": 1,
            "oriented_rays_identified_by_symmetry": False,
            "fan": ["origin", "positive_ray", "negative_ray"],
            "positive_singleton_circuit": circuit,
            "exclusion_basis": circuit["proof_kind"],
            "entire_declared_quotient_line_and_all_residual_gauge_lifts_excluded": (
                circuit[
                    "entire_declared_quotient_line_and_all_residual_gauge_lifts_excluded"
                ]
            ),
            "diagnostic_oriented_lift_replays": rays,
            "all_oriented_rays_excluded": True,
            "all_oriented_rays_excluded_basis": (
                "entire quotient line excluded by positive singleton circuit"
            ),
            "exact_checks": class_checks,
        }

    checks = {
        "raw_branch_count_is_12_power_4": (
            raw_count
            == expected["repair_options_per_spill"] ** len(spills)
            == expected["raw_branch_count"]
        ),
        "all_cached_branch_lattice_checks_pass": all(
            all(lattice["exact_checks"].values())
            for lattice in cache.values()
        ),
        "branch_support_and_quotient_histogram_exact": (
            dict(sorted(branch_histogram.items()))
            == expected["branch_support_and_quotient_histogram"]
        ),
        "minimum_positive_support_size_is_22": (
            minimum_positive_support_size == 22
        ),
        "four_minimum_positive_fixed_chart_decorations": (
            len(minimum_positive) == 4
        ),
        "two_complete_decoration_symmetry_classes": (
            len(class_records) == 2
        ),
        "both_positive_singleton_circuits_pass_exact_replay": all(
            all(
                record["positive_singleton_circuit"][
                    "exact_checks"
                ].values()
            )
            for record in class_records.values()
        ),
        "both_declared_exact_support_lines_and_all_gauge_lifts_excluded": all(
            record[
                "entire_declared_quotient_line_and_all_residual_gauge_lifts_excluded"
            ]
            for record in class_records.values()
        ),
        "four_oriented_rays_excluded_by_line_circuits": all(
            record["all_oriented_rays_excluded"]
            for record in class_records.values()
        ),
        "both_matching_enumerators_replay_every_diagnostic_lift": all(
            ray_record["exact_checks"][
                "primary_independent_enumerators_agree"
            ]
            for record in class_records.values()
            for ray_record in record[
                "diagnostic_oriented_lift_replays"
            ].values()
        ),
        "no_diagnostic_lift_claims_quotient_ray_exclusion": all(
            not ray_record["quotient_ray_exclusion_claimed_from_this_replay"]
            for record in class_records.values()
            for ray_record in record[
                "diagnostic_oriented_lift_replays"
            ].values()
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise KrennN8ToricUnequalRateError(
            f"the depth-two gate failed: {failed}"
        )
    return {
        "policy": {
            "seed_orbit": "H5",
            "first_shell_parent_orbit_index": (
                expected["parent_orbit_index"]
            ),
            "parent_selected_by": (
                "fixed deterministic orbit index from the minimum-spill "
                "first shell; no claim of global parent minimality"
            ),
            "singleton_spills_repaired": list(
                expected["spill_equations"]
            ),
            "one_minimum-new-coordinate_repair_per_spill": True,
            "repair_options_per_spill": (
                expected["repair_options_per_spill"]
            ),
            "raw_branch_count": raw_count,
            "floating_point_used": False,
            "random_seeds": [],
        },
        "declared_family": {
            "name": (
                "bounded-H5-depth-two-support22-target-leading-stratum"
            ),
            "exact_support_mode": {
                "support_equals_each_recorded_class_S": True,
                "all_S_leading_coefficients_nonzero": True,
                "outside_source_coordinates_zero_on_support_torus": True,
            },
            "conditional_degeneration_extension": {
                "allowed": True,
                "outside_source_coordinates_may_have_higher_order": True,
                "required_condition": (
                    "every outside monomial stays strictly above every "
                    "relevant active target and recorded singleton order"
                ),
            },
            "common_active_target_monomial_order_normalized_to_zero": True,
            "outside_term_entry_cones_classified": False,
        },
        "branch_support_size_and_quotient_dimension_histogram": {
            f"{support_size},{dimension}": count
            for (support_size, dimension), count in sorted(
                branch_histogram.items()
            )
        },
        "distinct_lattice_systems_replayed": len(cache),
        "minimum_positive_quotient_support_size": (
            minimum_positive_support_size
        ),
        "minimum_positive_fixed_chart_decorations": len(
            minimum_positive
        ),
        "minimum_positive_secondary_matching_tuples": [
            list(values) for values in sorted(observed_tuples)
        ],
        "complete_decoration_symmetry_classes": len(class_records),
        "oriented_quotient_ray_count": 4,
        "classes": class_records,
        "consequence": (
            "within the depth-two local declared family, both "
            "one-dimensional quotient lines in the first nonzero "
            "bounded H5 fan are excluded by exact positive singleton "
            "incidence circuits whose identities hold on the full tie "
            "lattice, including every residual-gauge lift; the chosen "
            "positive and negative lift minima are diagnostics only"
        ),
        "exact_checks": checks,

    }
@lru_cache(maxsize=1)
def _build_certificate() -> dict[str, object]:
    first_certificate = (
        first_shell.build_n8_toric_first_shell_certificate()
    )
    global_gauge = _global_gauge_lattice_record()
    hard_cases = {
        label: _hard_case_record(label, first_certificate)
        for label in ("H5", "H6")
    }
    depth_two = _build_depth_two_gate(first_certificate)
    total_orbits = sum(
        record["decorated_stabilizer_orbits"]
        for record in hard_cases.values()
    )
    total_raw = sum(
        record["raw_decorations_covered"]
        for record in hard_cases.values()
    )
    total_skeleton_rays = sum(
        record["primitive_nonzero_skeleton_rational_quotient_rays"]
        for record in hard_cases.values()
    )
    total_target_charts = sum(
        record["target_leading_chart_count"]
        for record in hard_cases.values()
    )
    if (
        total_orbits,
        total_raw,
        total_target_charts,
        total_skeleton_rays,
    ) != (347, 1_872, 541, 0):
        raise KrennN8ToricUnequalRateError(
            "the combined unequal-rate census changed"
        )
    return {
        "schema": UNEQUAL_RATE_SCHEMA,
        "scope": {
            "support_target_leading_charts": total_target_charts,
            "parameters": {"n": N, "d": D},
            "seed_orbits": ["H5", "H6"],
            "decorated_stabilizer_orbits": total_orbits,
            "raw_decorations_covered": total_raw,
            "bounded_depth_two_raw_branches": 20_736,
            "floating_point_used": False,
            "random_seeds": [],
            "cpu_workers": 1,
        },
        "declared_family": {
            "name": (
                "seed-retaining balanced minimal cost-two repair "
                "skeletons"
            ),
            "seed_coordinates_have_nonzero_leading_coefficients": True,
            "one_cost_two_repair_per_original_victim": True,
            "one_active_support_target_monomial_selected_per_color": True,
            "all_support_target_monomial_choices_enumerated": True,
            "selected_repair_and_seed_victim_orders_are_tied": True,
            "no_outside_monomial_enters_an_original_victim_minimum": True,
            "no_outside_monomial_enters_any_recorded_flat_singleton_minimum": (
                True
            ),
            "support_level_target_initial_sum_is_not_fully_cancelled": True,
            "outside_source_coordinates_all_absent": False,
            "outside_source_coordinates_may_have_higher_order": True,
            "within_equation_term_order_comparisons_are_gauge_invariant": (
                True
            ),
        },
        "global_gauge_lattice_audit": global_gauge,
        "local_quotient_proof": {
            "constraint_lattice": (
                "K_S uses one selected active target monomial per color "
                "and one repair-minus-seed tie per original victim"
            ),
            "gauge_containment": "C_S G_S = 0 exactly",
            "rational_equality": (
                "rank(G_S) = |S| - rank(C_S) on every orbit"
            ),
            "integral_equality": (
                "each G_S has a displayed maximal minor of determinant "
                "+1 or -1, so im_Z(G_S) is saturated and equals K_S"
            ),
            "affine_target_order_zero_root_extraction_required": False,
            "projective_integral_quotient": (
                "ker_Z(B_proj)/im_Z([G_S|1]) is canonically Z/4 "
                "through the common selected active target-monomial "
                "order q modulo four"
            ),
            "projective_q_surjectivity": (
                "all 541 target-leading charts display a determinant-one "
                "minor and an integral q=1 witness"
            ),
            "projective_normalization_may_require_degree_four_base_change": (
                True
            ),
            "primitive_ray_enumeration_termination": (
                "the rational skeleton quotient is zero; Z/4 torsion is "
                "finite and creates no ray"
            ),
        },
        "hard_cases": hard_cases,
        "bounded_H5_depth_two_quotient_fan": depth_two,
        "conclusion": {
            "classification": (
                "exact-negative-bounded-depth-two-quotient-fan"
            ),
            "total_balanced_skeleton_orbits": total_orbits,
            "total_raw_decorations_covered": total_raw,
            "support_target_leading_charts": total_target_charts,
            "primitive_nonzero_skeleton_rational_quotient_rays": (
                total_skeleton_rays
            ),
            "projective_integral_quotient_torsion_invariant_factors": [4],
            "skeleton_coordinate_unequal_rates_create_new_leading_initial_forms": (
                False
            ),
            "bounded_depth_two_oriented_quotient_rays": 4,
            "bounded_depth_two_oriented_rays_excluded": 4,
            "bounded_depth_two_quotient_lines_excluded": 2,
            "bounded_depth_two_exclusion_basis": (
                "positive-singleton-tie-row-span-circuit"
            ),
            "canonical_lift_absolute_order_replays_used_as_proof": False,
            "flat_first_shell_obstruction_is_exhaustive_in_family": True,
            "interpretation": (
                "within the declared minimal skeleton family every "
                "admissible rate restricted to the skeleton is "
                "target-preserving gauge plus common projective scaling; "
                "after at most a fourth-order base change the flat "
                "singleton obstruction applies whenever no outside term "
                "enters its minimum"
                "; the first nonzero quotient fan in the bounded H5 "
                "depth-two parent branch has two quotient lines (four "
                "orientations); within its local declared exact-support "
                "or conditional-higher-outside-order family, both are "
                "excluded by positive singleton incidence circuits after "
                "all active target monomial orders are verified zero on "
                "the tie lattice; outside-entry cones remain open and "
                "absolute orders on selected positive and negative lifts "
                "are retained only as non-gauge-invariant diagnostics"
            ),
        },
        "claim_boundary": {
            "all_347_balanced_minimal_skeleton_quotients_are_zero": True,
            "all_1872_raw_balanced_minimal_skeletons_are_covered": True,
            "flat_obstruction_exhaustive_in_declared_family": True,
            "all_H5_H6_valuation_cones_classified": False,
            "all_541_support_target_leading_charts_have_zero_rational_quotient": True,
            "all_541_projective_integral_quotients_are_Z_mod_4": True,
            "all_original_victim_equations_are_exactly_two_term_on_support": True,
            "all_unequal_rate_H5_H6_degenerations_excluded": False,
            "additional_leading_repairs_classified": False,
            "higher_cost_repairs_classified": False,
            "bounded_H5_parent_20736_depth_two_branches_enumerated": True,
            "bounded_H5_minimum_positive_support22_layer_classified": True,
            "bounded_H5_depth_two_four_oriented_rays_excluded": True,
            "bounded_H5_support22_declared_target_leading_two_lines_excluded_by_positive_singleton_circuits": (
                True
            ),
            "bounded_H5_support22_declared_target_leading_all_residual_gauge_lifts_excluded": (
                True
            ),
            "canonical_lift_absolute_orders_claimed_gauge_invariant": False,
            "canonical_lift_replays_used_as_exclusion_proof": False,
            "all_H5_depth_two_quotient_cones_classified": False,
            "unequal_monochromatic_minimum_layers_with_cancellation_classified": False,
            "outside_term_entry_cones_classified": False,
            "deeper_repair_closures_classified": False,
            "all_31_seed_orbits_classified": False,
            "support_level_target_initial_sum_cancellation_layers_classified": False,
            "full_effective_global_gauge_quotient_constructed": False,
            "full_graph_or_Rees_compactification_constructed": False,
            "full_saturated_initial_ideal_computed": False,
            "n8_projective_border_membership_proved": False,
            "n8_strict_border_membership_proved": False,
            "n8_affine_membership_proved": False,
            "n8_nonexistence_proved": False,
            "finite_counterexample_found": False,
        },
    }


def build_n8_toric_unequal_rate_certificate() -> dict[str, object]:
    """Return the exact local unequal-rate quotient certificate."""

    return deepcopy(_build_certificate())


def verify_n8_toric_unequal_rate_certificate(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Recompute every orbit lattice and reject any payload change."""

    if not isinstance(payload, Mapping):
        raise KrennN8ToricUnequalRateError(
            "the unequal-rate certificate must be a mapping"
        )
    expected = build_n8_toric_unequal_rate_certificate()
    if dict(payload) != expected:
        raise KrennN8ToricUnequalRateError(
            "the unequal-rate certificate differs from exact replay"
        )
    return expected


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _strict_json(path: Path) -> object:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise KrennN8ToricUnequalRateError(
                    f"duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str):
        raise KrennN8ToricUnequalRateError(
            f"nonfinite JSON constant {value!r}"
        )

    try:
        return json.loads(
            path.read_bytes().decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KrennN8ToricUnequalRateError(
            f"could not decode {path.name}"
        ) from error


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _readme_text(certificate: Mapping[str, object]) -> str:
    conclusion = certificate["conclusion"]
    h5 = certificate["hard_cases"]["H5"]
    h6 = certificate["hard_cases"]["H6"]
    depth = certificate["bounded_H5_depth_two_quotient_fan"]
    class_a = depth["classes"]["A"]
    class_b = depth["classes"]["B"]
    return f"""# Exact `n=8,d=3` toric unequal-rate gate

This bundle tests whether unequal coordinate orders create new quotient
directions on the minimal H5/H6 cost-two repair skeletons.  It is a local
integral lattice calculation, not a construction of the full GIT quotient,
graph/Rees compactification, or an `n=8` existence/nonexistence proof.

## Exact result

| seed | raw decorations | symmetry orbits | constraint rank | gauge rank | quotient rays |
|---|---:|---:|---:|---:|---:|
| H5 | {h5["raw_decorations_covered"]} | {h5["decorated_stabilizer_orbits"]} | {h5["order_constraint_rank"]} | {h5["restricted_gauge_rank"]} | {h5["primitive_nonzero_skeleton_rational_quotient_rays"]} |
| H6 | {h6["raw_decorations_covered"]} | {h6["decorated_stabilizer_orbits"]} | {h6["order_constraint_rank"]} | {h6["restricted_gauge_rank"]} | {h6["primitive_nonzero_skeleton_rational_quotient_rays"]} |

For every decorated skeleton, the repair-order constraint lattice equals the
restricted target-preserving gauge lattice over the integers.  Each equality
is certified by `C_S G_S = 0`, complementary exact ranks, and a displayed
maximal gauge minor of determinant `+1` or `-1`.  Therefore there are
{conclusion["primitive_nonzero_skeleton_rational_quotient_rays"]} nonzero
rational quotient rays.  Projectively the integral quotient is `Z/4`;
this finite torsion creates no ray, though normalization can require a
fourth-order base change.

The full `252 x 21` raw gauge lattice itself has saturation index two, as
expected from the generic `mu_2` ineffectivity.  The local determinant-one
minors are what make the stronger skeleton-level integral statement valid.

After gauge normalization, within-equation term-order differences and the
flat singleton obstruction are unchanged.  Hence unequal rates do not rescue
any of these {conclusion["total_raw_decorations_covered"]} minimal skeletons.

## Bounded H5 depth-two fan

All {depth["policy"]["raw_branch_count"]} fixed-parent repair decorations
were ranked exactly.  The minimum positive quotient layer has four
support-22 decorations in two symmetry classes and four oriented rays.

The exclusion is gauge-invariant and comes from exact positive singleton
incidence circuits, not from absolute orders on selected ray lifts:

| class | positive singleton combination | tie-row-span witness |
|---|---|---|
| A | `3 E_571 + 2 E_851 + E_2438 + E_2493 + E_4048 + E_5792` | `[1,3,2,0,1,3]` |
| B | `E_853 + E_2430 + E_6557` | `[0,1,1,0,1,0]` |

Every listed mixed equation has exactly one support monomial under both
independent perfect-matching enumerators.  In each class, the positive
weighted sum of their nonseed incidence rows is exactly the displayed linear
combination of the tie rows over `Q`; adjoining that row does not increase
exact rank five.  Consequently the positive weighted sum of singleton orders
is zero everywhere on the tie lattice, so at least one unique mixed singleton
has order at most zero.

The target comparison is also exact.  Both enumerators find two active
monochromatic terms for color 0 and one each for colors 1 and 2.  Every one of
their four nonseed incidence rows has a displayed witness in the same tie-row
span.  Thus every active target support monomial has normalized order zero on
the full tie lattice.  The target polynomial valuation is therefore at least
zero (or infinite); cancellation among target terms can only raise it and
strengthen the mixed-singleton obstruction.

This excludes both quotient lines, hence all four orientations, throughout
the depth-two local declared family: either the exact support torus, where all
coordinates of the recorded support `S` have nonzero leading coefficient and
outside coordinates vanish, or the conditional degeneration extension where
every outside monomial stays strictly above every relevant active target and
recorded singleton order.  Outside-term-entry cones remain unclassified.

For audit only, the chosen normalized positive and negative lifts give these
gauge-dependent counts:

| chosen lift | unique mixed minima at order <= target |
|---|---:|
| A+ | {class_a["diagnostic_oriented_lift_replays"]["+"]["obstructing_count"]} |
| A- | {class_a["diagnostic_oriented_lift_replays"]["-"]["obstructing_count"]} |
| B+ | {class_b["diagnostic_oriented_lift_replays"]["+"]["obstructing_count"]} |
| B- | {class_b["diagnostic_oriented_lift_replays"]["-"]["obstructing_count"]} |

Those absolute-order counts are not residual-gauge invariant and are not used
as proof.  This unequal-rate bundle itself makes no support-24 classification.
The declared exact-support/no-outside-entry positive layer is handled by the
separate support-24 circuit gate.

## Boundary

This does not classify strata with multiple leading repairs, cost-three or
cost-four repairs, outside terms entering an initial form, deeper repair
closures, the other 29 seed orbits, or the full saturated initial ideal.

## Reproduce

```powershell
C:\\tmp\\Krenn-obstruction-venv\\Scripts\\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_unequal_rates --results-directory results\\krenn_quantum_graph\\n8_d3_toric_unequal_rates
```

```powershell
C:\\tmp\\Krenn-obstruction-venv\\Scripts\\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_unequal_rates --results-directory results\\krenn_quantum_graph\\n8_d3_toric_unequal_rates --verify-only
```
"""


def _expected_manifest(
    directory: Path,
    certificate: Mapping[str, object],
) -> dict[str, object]:
    root = _repo_root()
    return {
        "schema": UNEQUAL_RATE_MANIFEST_SCHEMA,
        "artifacts": [
            _file_record(directory / CERTIFICATE_FILE, relative_to=directory),
            _file_record(directory / README_FILE, relative_to=directory),
        ],
        "inputs": [
            _file_record(root / relative, relative_to=root)
            for relative in SOURCE_PATHS
        ],
        "claim_boundary": certificate["claim_boundary"],
        "scratch_data_required_for_verification": False,
    }


def write_n8_toric_unequal_rate_bundle(
    directory: Path | str = DEFAULT_RESULTS_DIRECTORY,
) -> dict[str, object]:
    """Write the exact certificate, README, and source-hash manifest."""

    directory = Path(directory)
    if not directory.is_absolute():
        directory = _repo_root() / directory
    directory.mkdir(parents=True, exist_ok=True)
    expected_names = {CERTIFICATE_FILE, README_FILE, MANIFEST_FILE}
    unexpected = {
        path.name
        for path in directory.iterdir()
        if path.name not in expected_names
    }
    if unexpected:
        raise KrennN8ToricUnequalRateError(
            f"unexpected unequal-rate artifact files: {sorted(unexpected)}"
        )
    certificate = build_n8_toric_unequal_rate_certificate()
    _write_bytes_atomic(
        directory / CERTIFICATE_FILE,
        _canonical_json_bytes(certificate),
    )
    _write_bytes_atomic(
        directory / README_FILE,
        _readme_text(certificate).encode("ascii"),
    )
    manifest = _expected_manifest(directory, certificate)
    _write_bytes_atomic(
        directory / MANIFEST_FILE,
        _canonical_json_bytes(manifest),
    )
    return certificate


def verify_n8_toric_unequal_rate_bundle(
    directory: Path | str = DEFAULT_RESULTS_DIRECTORY,
) -> dict[str, object]:
    """Replay inventory, hashes, sources, and all lattice calculations."""

    directory = Path(directory)
    if not directory.is_absolute():
        directory = _repo_root() / directory
    if not directory.is_dir() or directory.is_symlink():
        raise KrennN8ToricUnequalRateError(
            "the unequal-rate bundle directory is absent or linked"
        )
    expected_names = {CERTIFICATE_FILE, README_FILE, MANIFEST_FILE}
    paths = tuple(directory.iterdir())
    if {path.name for path in paths} != expected_names or any(
        path.is_symlink() or not path.is_file() for path in paths
    ):
        raise KrennN8ToricUnequalRateError(
            "the unequal-rate bundle inventory changed"
        )
    certificate = _strict_json(directory / CERTIFICATE_FILE)
    if not isinstance(certificate, Mapping):
        raise KrennN8ToricUnequalRateError(
            "the unequal-rate certificate is not a mapping"
        )
    verified = verify_n8_toric_unequal_rate_certificate(certificate)
    if (directory / README_FILE).read_bytes() != _readme_text(
        verified
    ).encode("ascii"):
        raise KrennN8ToricUnequalRateError(
            "the unequal-rate README changed"
        )
    manifest = _strict_json(directory / MANIFEST_FILE)
    if manifest != _expected_manifest(directory, verified):
        raise KrennN8ToricUnequalRateError(
            "the unequal-rate manifest or source ledger changed"
        )
    return verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact n=8,d=3 H5/H6 unequal-rate quotient gate"
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        certificate = verify_n8_toric_unequal_rate_bundle(
            args.results_directory
        )
    else:
        certificate = write_n8_toric_unequal_rate_bundle(
            args.results_directory
        )
        verify_n8_toric_unequal_rate_bundle(args.results_directory)
    print(
        json.dumps(
            {
                "classification": certificate["conclusion"][
                    "classification"
                ],
                "balanced_skeleton_orbits": certificate["conclusion"][
                    "total_balanced_skeleton_orbits"
                ],
                "primitive_nonzero_skeleton_rational_rays": certificate[
                    "conclusion"
                ]["primitive_nonzero_skeleton_rational_quotient_rays"],
                "bounded_depth_two_oriented_rays_excluded": certificate[
                    "conclusion"
                ]["bounded_depth_two_oriented_rays_excluded"],
                "n8_nonexistence_proved": certificate["claim_boundary"][
                    "n8_nonexistence_proved"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = (
    "DEFAULT_RESULTS_DIRECTORY",
    "KrennN8ToricUnequalRateError",
    "build_n8_toric_unequal_rate_certificate",
    "verify_n8_toric_unequal_rate_bundle",
    "verify_n8_toric_unequal_rate_certificate",
    "write_n8_toric_unequal_rate_bundle",
)


if __name__ == "__main__":
    raise SystemExit(main())
