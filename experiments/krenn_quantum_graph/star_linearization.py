r"""Exact six-apex linearization of the ``n=6,d=3`` tensor map.

Every perfect matching pairs a chosen apex with exactly one other vertex.
For fixed apex color, the 729 cubic equations therefore become one
``243 x 15`` linear system in that star's weights.  Its entries are the
quadratic perfect-matching tensors on the four residual vertices, and the
same coefficient matrix serves all three apex colors.

This is an exact factorization, not a symmetry ansatz: all 135 weights
remain independent.  It does not by itself decide whether the three target
vectors lie in the matrix column space for some choice of the 90 nonstar
weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from itertools import product
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.star_linearization_independent import (
    independent_star_linearization_audit,
)
from experiments.krenn_quantum_graph.system import (
    coloring_index,
    generate_sparse_system,
    perfect_matchings,
    variable_index,
)


N = 6
D = 3
STAR_LINEARIZATION_SCHEMA = "krenn-n6-d3-star-linearization-v1"
STAR_LINEARIZATION_AUDIT_SCHEMA = (
    "krenn-n6-d3-six-apex-star-linearization-audit-v1"
)
QuadraticMonomial = tuple[int, int]
QuadraticPolynomial = tuple[QuadraticMonomial, ...]


class KrennStarLinearizationError(RuntimeError):
    """An exact apex-star factorization failed replay."""


def _validated_apex(apex: int) -> int:
    apex = int(apex)
    if not 0 <= apex < N:
        raise KrennStarLinearizationError(
            "the apex must be a K6 vertex"
        )
    return apex


@lru_cache(maxsize=None)
def _matching_rows(
    vertices: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not vertices:
        return ((),)
    first = vertices[0]
    result = []
    for position in range(1, len(vertices)):
        partner = vertices[position]
        residual = vertices[1:position] + vertices[position + 1:]
        for tail in _matching_rows(residual):
            result.append(((first, partner), *tail))
    return tuple(result)


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


@dataclass(frozen=True)
class StarLinearization:
    """One exact shared ``243 x 15`` quadratic coefficient matrix."""

    apex: int
    remaining_vertices: tuple[int, ...]
    row_colorings: tuple[tuple[int, ...], ...]
    columns: tuple[tuple[int, int], ...]
    star_variable_blocks: tuple[tuple[int, ...], ...]
    entries: tuple[tuple[QuadraticPolynomial, ...], ...]
    schema: str = STAR_LINEARIZATION_SCHEMA

    def __post_init__(self) -> None:
        apex = _validated_apex(self.apex)
        expected_remaining = tuple(
            vertex for vertex in range(N) if vertex != apex
        )
        expected_columns = tuple(
            (partner, partner_color)
            for partner in expected_remaining
            for partner_color in range(D)
        )
        if (
            self.schema != STAR_LINEARIZATION_SCHEMA
            or self.remaining_vertices != expected_remaining
            or self.row_colorings
            != tuple(product(range(D), repeat=5))
            or self.columns != expected_columns
            or len(self.star_variable_blocks) != D
            or any(len(block) != 15 for block in self.star_variable_blocks)
            or len(self.entries) != 243
            or any(len(row) != 15 for row in self.entries)
        ):
            raise KrennStarLinearizationError(
                "a star-linearization shape changed"
            )
        expected_star = tuple(
            tuple(
                variable_index(
                    N,
                    D,
                    apex,
                    partner,
                    apex_color,
                    partner_color,
                )
                for partner, partner_color in self.columns
            )
            for apex_color in range(D)
        )
        if self.star_variable_blocks != expected_star:
            raise KrennStarLinearizationError(
                "a star-variable block changed"
            )
        if (
            self.nonzero_entry_count != 1_215
            or self.coefficient_term_count != 3_645
            or any(
                len(polynomial) not in (0, 3)
                for row in self.entries
                for polynomial in row
            )
        ):
            raise KrennStarLinearizationError(
                "the quadratic star-matrix census changed"
            )

    @property
    def target_rows(self) -> tuple[int, int, int]:
        return (0, 121, 242)

    @property
    def nonzero_entry_count(self) -> int:
        return sum(
            bool(polynomial)
            for row in self.entries
            for polynomial in row
        )

    @property
    def coefficient_term_count(self) -> int:
        return sum(
            len(polynomial)
            for row in self.entries
            for polynomial in row
        )

    def coefficient(
        self, row: int, column: int
    ) -> QuadraticPolynomial:
        row = int(row)
        column = int(column)
        if not 0 <= row < 243 or not 0 <= column < 15:
            raise KrennStarLinearizationError(
                "a star-matrix coordinate is out of range"
            )
        return self.entries[row][column]

    def reconstructed_equation_monomials(
        self, apex_color: int, row: int
    ) -> tuple[tuple[int, int, int], ...]:
        apex_color = int(apex_color)
        row = int(row)
        if not 0 <= apex_color < D or not 0 <= row < 243:
            raise KrennStarLinearizationError(
                "a star equation coordinate is invalid"
            )
        monomials = []
        for column, polynomial in enumerate(self.entries[row]):
            star = self.star_variable_blocks[apex_color][column]
            monomials.extend(
                tuple(sorted((*quadratic, star)))
                for quadratic in polynomial
            )
        return tuple(sorted(monomials))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.apex}|{self.remaining_vertices}|"
                f"{self.columns}|{self.target_rows}\n"
            ).encode("ascii")
        )
        for color, block in enumerate(self.star_variable_blocks):
            digest.update(
                f"star:{color}:{','.join(map(str, block))}\n".encode(
                    "ascii"
                )
            )
        for row_index, row in enumerate(self.entries):
            for column_index, polynomial in enumerate(row):
                if not polynomial:
                    continue
                digest.update(
                    f"{row_index}:{column_index}|".encode("ascii")
                )
                for monomial in polynomial:
                    digest.update(
                        f"{monomial[0]},{monomial[1]};".encode(
                            "ascii"
                        )
                    )
                digest.update(b"\n")
        return digest.hexdigest()


@lru_cache(maxsize=6)
def star_linearization(apex: int) -> StarLinearization:
    """Construct and replay one apex factorization exactly."""

    apex = _validated_apex(apex)
    remaining = tuple(vertex for vertex in range(N) if vertex != apex)
    rows = tuple(product(range(D), repeat=5))
    columns = tuple(
        (partner, partner_color)
        for partner in remaining
        for partner_color in range(D)
    )
    star_blocks = tuple(
        tuple(
            variable_index(
                N,
                D,
                apex,
                partner,
                apex_color,
                partner_color,
            )
            for partner, partner_color in columns
        )
        for apex_color in range(D)
    )
    matrix = []
    for row_coloring in rows:
        colors = dict(zip(remaining, row_coloring, strict=True))
        matrix_row = []
        for partner, partner_color in columns:
            if colors[partner] != partner_color:
                matrix_row.append(())
                continue
            residual = tuple(
                vertex
                for vertex in range(N)
                if vertex not in (apex, partner)
            )
            polynomial = tuple(sorted(
                tuple(sorted(
                    variable_index(
                        N,
                        D,
                        left,
                        right,
                        colors[left],
                        colors[right],
                    )
                    for left, right in matching
                ))
                for matching in _matching_rows(residual)
            ))
            matrix_row.append(polynomial)
        matrix.append(tuple(matrix_row))
    result = StarLinearization(
        apex=apex,
        remaining_vertices=remaining,
        row_colorings=rows,
        columns=columns,
        star_variable_blocks=star_blocks,
        entries=tuple(matrix),
    )

    system = generate_sparse_system(N, D)
    for apex_color in range(D):
        for row, row_coloring in enumerate(rows):
            coloring = [0] * N
            coloring[apex] = apex_color
            for vertex, color in zip(
                remaining, row_coloring, strict=True
            ):
                coloring[vertex] = color
            expected = tuple(sorted(
                tuple(sorted(monomial))
                for monomial in system.equation_monomials(
                    coloring_index(N, D, coloring)
                )
            ))
            if result.reconstructed_equation_monomials(
                apex_color, row
            ) != expected:
                raise KrennStarLinearizationError(
                    "a primary equation failed exact star replay"
                )
    return result


def _differencing_audit() -> dict:
    """Audit the omitted cross terms in the proposed one-column lemma."""

    # If one non-apex vertex color changes, every perfect matching changes
    # one incident colored-edge variable.  Of the 30 signed monomials in
    # the difference, six come from the column paired to that vertex and
    # 24 come from the other four possible apex partners.
    for apex in range(N):
        factorization = star_linearization(apex)
        for varied_partner in factorization.remaining_vertices:
            other_partners = tuple(
                partner
                for partner in factorization.remaining_vertices
                if partner != varied_partner
            )
            if len(other_partners) != 4:
                raise KrennStarLinearizationError(
                    "the differencing partner census changed"
                )
            for other_partner in other_partners:
                residual = tuple(
                    vertex
                    for vertex in range(N)
                    if vertex not in (apex, other_partner)
                )
                if varied_partner not in residual:
                    raise KrennStarLinearizationError(
                        "a cross term unexpectedly lost the varied vertex"
                    )
                if any(
                    all(
                        varied_partner not in edge
                        for edge in matching
                    )
                    for matching in _matching_rows(residual)
                ):
                    raise KrennStarLinearizationError(
                        "a K4 matching failed to cover the varied vertex"
                    )
    return {
        "proposed_single_column_differencing_lemma_valid": False,
        "signed_monomials_in_full_row_difference": 30,
        "signed_monomials_from_varied_partner_column": 6,
        "signed_cross_monomials_from_other_four_columns": 24,
        "reason": (
            "changing c_v also changes P_u for every u != v because "
            "the residual K4 defining P_u contains v"
        ),
        "constant_color_case_tree_derived_from_the_lemma_is_void": True,
        "all_constant_branch_is_forced": False,
        "all_constant_branch_used_downstream": False,
    }


def star_linearization_audit() -> dict:
    """Return the exact six-apex factorization and claim boundaries."""

    rows = []
    for apex in range(N):
        primary = star_linearization(apex)
        independent = independent_star_linearization_audit(apex)
        if primary.fingerprint() != independent["fingerprint"]:
            raise KrennStarLinearizationError(
                "primary and independent star fingerprints differ"
            )
        rows.append({
            "apex": apex,
            "rows_per_apex_color": 243,
            "columns_per_apex_color": 15,
            "apex_color_blocks": 3,
            "star_variables": 45,
            "nonstar_variables": 90,
            "coefficient_degree": 2,
            "nonzero_matrix_entries": primary.nonzero_entry_count,
            "coefficient_terms": primary.coefficient_term_count,
            "shared_matrix_for_all_three_targets": True,
            "target_rows": list(primary.target_rows),
            "all_729_equations_reconstructed": True,
            "primary_sha256": primary.fingerprint(),
            "independent_sha256": independent["fingerprint"],
        })
    return json.loads(json.dumps({
        "schema": STAR_LINEARIZATION_AUDIT_SCHEMA,
        "parameters": {"n": N, "d": D},
        "apex_factorizations": rows,
        "exact_equivalence": {
            "formula": (
                "F(a,c)=sum_v y_(v,a,c_v) P_v(c_without_v)"
            ),
            "target_condition": (
                "one shared 243x15 matrix must contain e_00000, "
                "e_11111, and e_22222 in its column space"
            ),
            "all_weights_remain_independent": True,
            "one_apex_factorization_is_equivalent_to_all_729_equations":
                True,
            "six_apex_factorizations_are_simultaneous_cross_checks":
                True,
            "six_apex_factorizations_are_independent_constraints": False,
        },
        "differencing_review": _differencing_audit(),
        "recommended_next_use": {
            "degree_seven_macaulay_run": False,
            "raw_global_grobner_retry": False,
            "exact_star_rank_consequences": True,
            "bounded_alternating_star_solves": True,
        },
        "claim_boundary": {
            "finite_counterexample_found": False,
            "natural_chart_decided": False,
            "global_affine_membership_decided": False,
            "column_space_condition_solved": False,
        },
    }, allow_nan=False))


def verify_star_linearization_audit(payload: Mapping) -> dict:
    """Rebuild all six matrices and reject altered conclusions."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennStarLinearizationError(
            "the star audit is not strict JSON"
        ) from error
    expected = star_linearization_audit()
    if not _strict_json_equal(normalized, expected):
        raise KrennStarLinearizationError(
            "the star-linearization audit failed exact replay"
        )
    return normalized


__all__ = [
    "KrennStarLinearizationError",
    "STAR_LINEARIZATION_AUDIT_SCHEMA",
    "STAR_LINEARIZATION_SCHEMA",
    "StarLinearization",
    "star_linearization",
    "star_linearization_audit",
    "verify_star_linearization_audit",
]
