r"""Valuation obstruction for regularizing the natural Laurent degeneration.

The full vertex-scalar fiber gauge acts by

.. math::

    W_{ij}^{ab}\longmapsto\lambda_i\lambda_j W_{ij}^{ab},
    \qquad \prod_{i=0}^5\lambda_i=1.

For the natural nine-slot border family, let ``x_i=val(lambda_i)``.  The
product-one condition gives ``sum_i x_i=0``, and every active edge valuation
changes by ``x_i+x_j``.

This module gives an exact integer Farkas certificate that no such gauge can
make all nine active coordinates regular at the degeneration point.  It also
constructs a complete four-dimensional basis of gauge-invariant valuation
forms and identifies the obstructing form with the valuation of the invariant
six-weight monomial

.. math::

    Q=W_{02}^{11}W_{23}^{00}W_{03}^{22}
      W_{14}^{11}W_{45}^{00}W_{15}^{22}.

Along the path, ``Q=t^-1``.  The result concerns this support, path, and gauge
group only.  It is not an affine nonimage theorem and does not exclude other
degenerations or a finite exact witness.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.border_image import (
    natural_laurent_weight_entries,
)
from experiments.krenn_quantum_graph.system import (
    VariableKey,
    variable_index,
)


BORDER_VALUATION_SCHEMA = "krenn-natural-border-valuation-obstruction-v1"

ACTIVE_COORDINATES: tuple[VariableKey, ...] = (
    (0, 1, 0, 0),
    (2, 3, 0, 0),
    (4, 5, 0, 0),
    (0, 2, 1, 1),
    (1, 4, 1, 1),
    (3, 5, 1, 1),
    (0, 3, 2, 2),
    (1, 5, 2, 2),
    (2, 4, 2, 2),
)

INVARIANT_NAMES = (
    "S0_color_0_matching",
    "S1_color_1_matching",
    "S2_color_2_matching",
    "F_two_disjoint_triangles",
)

INVARIANT_MONOMIAL_Q_COORDINATES: tuple[VariableKey, ...] = (
    (0, 2, 1, 1),
    (2, 3, 0, 0),
    (0, 3, 2, 2),
    (1, 4, 1, 1),
    (4, 5, 0, 0),
    (1, 5, 2, 2),
)


class KrennBorderValuationError(RuntimeError):
    """The valuation, gauge, invariant basis, or dual replay failed."""


def natural_active_valuations() -> tuple[int, ...]:
    """Read the nine Laurent exponents from the full border family."""

    entries = dict(natural_laurent_weight_entries())
    result = []
    for coordinate in ACTIVE_COORDINATES:
        value = entries.get(variable_index(6, 3, *coordinate))
        if (
            value is None
            or len(value.terms) != 1
            or value.terms[0][1] != 1
        ):
            raise KrennBorderValuationError(
                "an active border weight is not one Laurent monomial"
            )
        result.append(value.terms[0][0])
    return tuple(result)


def active_unsigned_incidence_matrix() -> tuple[tuple[int, ...], ...]:
    """Return the ``6 x 9`` vertex-edge incidence matrix."""

    return tuple(
        tuple(
            int(vertex == coordinate[0])
            + int(vertex == coordinate[1])
            for coordinate in ACTIVE_COORDINATES
        )
        for vertex in range(6)
    )


def vertex_gauge_valuation_matrix() -> tuple[tuple[int, ...], ...]:
    r"""Return the ``9 x 5`` gauge matrix in basis ``e_r-e_5``.

    A gauge valuation vector with sum zero is written uniquely as
    ``sum_(r=0)^4 z_r(e_r-e_5)``.  The returned matrix maps ``z`` to the
    resulting changes of the nine active edge valuations.
    """

    return tuple(
        tuple(
            int(i == direction)
            + int(j == direction)
            - int(i == 5)
            - int(j == 5)
            for direction in range(5)
        )
        for i, j, _a, _b in ACTIVE_COORDINATES
    )


def invariant_valuation_basis() -> tuple[tuple[int, ...], ...]:
    """Return three matching sums and the two-triangle Farkas form."""

    return (
        (1, 1, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 1, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 1, 1),
        (0, 1, 1, 1, 1, 0, 1, 1, 0),
    )


def integer_farkas_vector() -> tuple[int, ...]:
    """Return the nonnegative dual vector selecting the two triangles."""

    return invariant_valuation_basis()[3]


def _rank_over_q(matrix: Sequence[Sequence[int]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennBorderValuationError(
            "exact rank requires a rectangular matrix"
        )
    rank = 0
    for column in range(width):
        selected = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        rows[rank], rows[selected] = rows[selected], rows[rank]
        pivot = rows[rank][column]
        rows[rank] = [value / pivot for value in rows[rank]]
        for row in range(rank + 1, len(rows)):
            factor = rows[row][column]
            if factor:
                rows[row] = [
                    left - factor * right
                    for left, right in zip(rows[row], rows[rank])
                ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _matrix_product(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    left = tuple(tuple(map(int, row)) for row in left)
    right = tuple(tuple(map(int, row)) for row in right)
    if not left or not right or len(left[0]) != len(right):
        raise KrennBorderValuationError(
            "valuation matrix-product dimensions differ"
        )
    width = len(right[0])
    if any(len(row) != width for row in right):
        raise KrennBorderValuationError(
            "the right valuation matrix is ragged"
        )
    return tuple(
        tuple(
            sum(
                left[row][middle] * right[middle][column]
                for middle in range(len(right))
            )
            for column in range(width)
        )
        for row in range(len(left))
    )


def _matrix_vector_product(
    matrix: Sequence[Sequence[int]], vector: Sequence[int]
) -> tuple[int, ...]:
    vector = tuple(map(int, vector))
    if any(len(row) != len(vector) for row in matrix):
        raise KrennBorderValuationError(
            "valuation matrix-vector dimensions differ"
        )
    return tuple(
        sum(int(value) * vector[column] for column, value in enumerate(row))
        for row in matrix
    )


def _dot(left: Sequence[int], right: Sequence[int]) -> int:
    left = tuple(map(int, left))
    right = tuple(map(int, right))
    if len(left) != len(right):
        raise KrennBorderValuationError(
            "valuation dot-product dimensions differ"
        )
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class BorderValuationCertificate:
    """Exact primal, dual, invariant-monomial, and scope record."""

    active_coordinates: tuple[VariableKey, ...]
    initial_valuations: tuple[int, ...]
    incidence_matrix: tuple[tuple[int, ...], ...]
    gauge_matrix: tuple[tuple[int, ...], ...]
    invariant_basis: tuple[tuple[int, ...], ...]
    farkas_vector: tuple[int, ...]
    invariant_values: tuple[int, ...]
    exact_checks: Mapping[str, bool]
    schema: str = BORDER_VALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BORDER_VALUATION_SCHEMA:
            raise KrennBorderValuationError(
                "the border-valuation schema changed"
            )
        if (
            self.active_coordinates != ACTIVE_COORDINATES
            or len(self.initial_valuations) != 9
            or len(self.incidence_matrix) != 6
            or any(len(row) != 9 for row in self.incidence_matrix)
            or len(self.gauge_matrix) != 9
            or any(len(row) != 5 for row in self.gauge_matrix)
            or len(self.invariant_basis) != 4
            or any(len(row) != 9 for row in self.invariant_basis)
            or len(self.farkas_vector) != 9
            or len(self.invariant_values) != 4
        ):
            raise KrennBorderValuationError(
                "the border-valuation certificate dimensions changed"
            )
        if not self.exact:
            failed = [
                name for name, passed in self.exact_checks.items() if not passed
            ]
            raise KrennBorderValuationError(
                f"border-valuation certificate failed: {failed}"
            )

    @property
    def exact(self) -> bool:
        return bool(self.exact_checks) and all(self.exact_checks.values())

    @property
    def this_family_vertex_gauge_regularization_impossible(self) -> bool:
        return self.exact

    @property
    def affine_nonimage_proved(self) -> bool:
        return False

    def to_dict(self) -> dict:
        primal_constraints = [
            {
                "coordinate": list(coordinate),
                "constant_valuation": self.initial_valuations[index],
                "gauge_coefficients_in_e_r_minus_e_5_basis": list(
                    self.gauge_matrix[index]
                ),
                "relation": ">= 0",
            }
            for index, coordinate in enumerate(self.active_coordinates)
        ]
        return {
            "schema": self.schema,
            "parameters": {
                "n": 6,
                "d": 3,
                "active_coordinates": 9,
                "vertex_gauge_dimension": 5,
            },
            "primal_integer_linear_program": {
                "variables": ["z0", "z1", "z2", "z3", "z4"],
                "vertex_valuation_parameterization": (
                    "x=sum_(r=0)^4 z_r*(e_r-e_5), so sum_i x_i=0"
                ),
                "constraints": primal_constraints,
                "feasible_over_Z": False,
                "feasible_over_Q": False,
            },
            "integer_Farkas_certificate": {
                "dual_vector": list(self.farkas_vector),
                "selected_edges": [
                    list(self.active_coordinates[index])
                    for index, coefficient in enumerate(self.farkas_vector)
                    if coefficient
                ],
                "dual_vector_nonnegative": True,
                "dual_times_gauge_matrix": [0, 0, 0, 0, 0],
                "dual_times_initial_valuation": -1,
                "summed_inequality": "-1 >= 0",
                "contradiction": True,
            },
            "complete_invariant_valuation_basis": {
                "ambient_dimension": 9,
                "gauge_orbit_dimension": 5,
                "quotient_dimension": 4,
                "names": list(INVARIANT_NAMES),
                "matrix": [list(row) for row in self.invariant_basis],
                "rank_over_Q": 4,
                "values_on_path": list(self.invariant_values),
                "basis_annihilates_gauge_matrix": True,
            },
            "invariant_monomial_Q": {
                "expression": (
                    "W[0,2,1,1]*W[2,3,0,0]*W[0,3,2,2]*"
                    "W[1,4,1,1]*W[4,5,0,0]*W[1,5,2,2]"
                ),
                "coordinates": [
                    list(coordinate)
                    for coordinate in INVARIANT_MONOMIAL_Q_COORDINATES
                ],
                "vertex_degrees": [2, 2, 2, 2, 2, 2],
                "gauge_multiplier": "(product_i lambda_i)^2 = 1",
                "value_on_path": "t^-1",
                "valuation_on_path": -1,
                "regular_coordinates_would_force_nonnegative_valuation": True,
            },
            "valuation_scope": {
                "Laurent_monomial_gauges_Z_valued": True,
                "Puiseux_gauges_Q_valued": True,
                "proof_uses_only_additivity_and_order_of_valuation": True,
                "ramified_reparameterization": (
                    "under t=s^k with k>0, the obstructing invariant "
                    "valuation is -k"
                ),
                "positive_reparameterizations_remain_obstructed": True,
            },
            "exact_checks": dict(self.exact_checks),
            "claim_boundary": {
                "specific_nine_support_path_certified": True,
                "full_product_one_vertex_scalar_gauge_certified": True,
                "pole_removable_by_that_gauge": False,
                "other_supports_or_paths_excluded": False,
                "support_changing_degenerations_excluded": False,
                "other_gauge_groups_excluded": False,
                "finite_exact_GHZ_witness_excluded": False,
                "affine_nonimage_proved": False,
                "border_membership_proved_by_this_obstruction": False,
                "consistent_with_the_Laurent_border_certificate": True,
            },
        }


def certify_natural_border_valuation_obstruction(
) -> BorderValuationCertificate:
    """Prove that the natural border pole survives vertex-scalar gauge."""

    valuations = natural_active_valuations()
    incidence = active_unsigned_incidence_matrix()
    gauge = vertex_gauge_valuation_matrix()
    invariants = invariant_valuation_basis()
    farkas = integer_farkas_vector()

    incidence_rank = _rank_over_q(incidence)
    gauge_rank = _rank_over_q(gauge)
    invariant_rank = _rank_over_q(invariants)
    invariants_times_gauge = _matrix_product(invariants, gauge)
    invariant_values = _matrix_vector_product(invariants, valuations)
    farkas_times_gauge = tuple(
        sum(
            farkas[row] * gauge[row][column]
            for row in range(9)
        )
        for column in range(5)
    )
    farkas_vertex_degrees = _matrix_vector_product(
        incidence, farkas
    )
    q_indices = tuple(
        ACTIVE_COORDINATES.index(coordinate)
        for coordinate in INVARIANT_MONOMIAL_Q_COORDINATES
    )
    q_indicator = tuple(
        int(index in q_indices) for index in range(9)
    )

    checks = {
        "natural_path_valuation_vector_is_1_minus1_then_zeros": (
            valuations == (1, -1, 0, 0, 0, 0, 0, 0, 0)
        ),
        "active_incidence_shape_6_by_9": (
            len(incidence) == 6
            and all(len(row) == 9 for row in incidence)
        ),
        "active_unsigned_incidence_rank_6_over_Q": (
            incidence_rank == 6
        ),
        "sum_zero_vertex_gauge_shape_9_by_5": (
            len(gauge) == 9
            and all(len(row) == 5 for row in gauge)
        ),
        "sum_zero_vertex_gauge_rank_5_over_Q": gauge_rank == 5,
        "invariant_basis_shape_4_by_9": (
            len(invariants) == 4
            and all(len(row) == 9 for row in invariants)
        ),
        "invariant_basis_rank_4_over_Q": invariant_rank == 4,
        "invariant_basis_annihilates_gauge": all(
            value == 0
            for row in invariants_times_gauge
            for value in row
        ),
        "four_invariants_complete_the_9_minus_5_quotient": (
            invariant_rank == 9 - gauge_rank == 4
        ),
        "path_invariant_values_are_0_0_0_minus1": (
            invariant_values == (0, 0, 0, -1)
        ),
        "integer_Farkas_vector_is_nonnegative": all(
            value >= 0 for value in farkas
        ),
        "integer_Farkas_vector_is_fourth_invariant": (
            farkas == invariants[3]
        ),
        "integer_Farkas_vector_kills_gauge": (
            farkas_times_gauge == (0, 0, 0, 0, 0)
        ),
        "integer_Farkas_vertex_degrees_are_all_2": (
            farkas_vertex_degrees == (2, 2, 2, 2, 2, 2)
        ),
        "integer_Farkas_constant_is_minus1": (
            _dot(farkas, valuations) == -1
        ),
        "invariant_Q_support_equals_Farkas_support": (
            q_indicator == farkas
        ),
        "invariant_Q_valuation_is_minus1": (
            sum(valuations[index] for index in q_indices) == -1
        ),
        "dual_certificate_proves_ILP_infeasible_over_Z": (
            all(value >= 0 for value in farkas)
            and farkas_times_gauge == (0, 0, 0, 0, 0)
            and _dot(farkas, valuations) < 0
        ),
        "same_dual_certificate_proves_infeasible_over_Q": (
            all(value >= 0 for value in farkas)
            and farkas_times_gauge == (0, 0, 0, 0, 0)
            and _dot(farkas, valuations) == -1
        ),
        "positive_reparameterization_scales_obstruction_to_minus_k": (
            _dot(farkas, valuations) == -1
        ),
    }
    return BorderValuationCertificate(
        active_coordinates=ACTIVE_COORDINATES,
        initial_valuations=valuations,
        incidence_matrix=incidence,
        gauge_matrix=gauge,
        invariant_basis=invariants,
        farkas_vector=farkas,
        invariant_values=invariant_values,
        exact_checks=checks,
    )


__all__ = (
    "ACTIVE_COORDINATES",
    "BORDER_VALUATION_SCHEMA",
    "BorderValuationCertificate",
    "INVARIANT_MONOMIAL_Q_COORDINATES",
    "INVARIANT_NAMES",
    "KrennBorderValuationError",
    "active_unsigned_incidence_matrix",
    "certify_natural_border_valuation_obstruction",
    "integer_farkas_vector",
    "invariant_valuation_basis",
    "natural_active_valuations",
    "vertex_gauge_valuation_matrix",
)
