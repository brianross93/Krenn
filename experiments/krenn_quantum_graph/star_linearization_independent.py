"""Independent subset-filter replay of the six-apex star factorization.

This module deliberately does not import the primary tensor-map or star
constructors.  It enumerates perfect matchings by filtering edge subsets
and reproduces the canonical star-matrix fingerprint independently.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from itertools import combinations, product


N = 6
D = 3
STAR_LINEARIZATION_SCHEMA = "krenn-n6-d3-star-linearization-v1"


class IndependentStarLinearizationError(RuntimeError):
    """An independently reconstructed star factorization changed."""


def _validated_apex(apex: int) -> int:
    apex = int(apex)
    if not 0 <= apex < N:
        raise IndependentStarLinearizationError(
            "the apex must be a K6 vertex"
        )
    return apex


@lru_cache(maxsize=None)
def _matchings(
    vertices: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    edges = tuple(combinations(vertices, 2))
    rows = []
    for candidate in combinations(edges, len(vertices) // 2):
        covered = tuple(vertex for edge in candidate for vertex in edge)
        if len(set(covered)) == len(vertices):
            rows.append(candidate)
    return tuple(rows)


def _variable_index(
    left: int,
    right: int,
    left_color: int,
    right_color: int,
) -> int:
    if left > right:
        left, right = right, left
        left_color, right_color = right_color, left_color
    if (
        not 0 <= left < right < N
        or not 0 <= left_color < D
        or not 0 <= right_color < D
    ):
        raise IndependentStarLinearizationError(
            "an independent variable coordinate is invalid"
        )
    edge_index = left * (2 * N - left - 1) // 2 + (
        right - left - 1
    )
    return (edge_index * D + left_color) * D + right_color


def _entry(
    apex: int,
    remaining: tuple[int, ...],
    row_coloring: tuple[int, ...],
    partner: int,
    partner_color: int,
) -> tuple[tuple[int, int], ...]:
    colors = dict(zip(remaining, row_coloring, strict=True))
    if colors[partner] != partner_color:
        return ()
    residual = tuple(
        vertex
        for vertex in range(N)
        if vertex not in (apex, partner)
    )
    monomials = []
    for matching in _matchings(residual):
        monomials.append(tuple(sorted(
            _variable_index(
                left,
                right,
                colors[left],
                colors[right],
            )
            for left, right in matching
        )))
    return tuple(sorted(monomials))


def _fingerprint(
    apex: int,
    remaining: tuple[int, ...],
    columns: tuple[tuple[int, int], ...],
    star_blocks: tuple[tuple[int, ...], ...],
    entries: tuple[tuple[tuple[tuple[int, int], ...], ...], ...],
) -> str:
    targets = (0, 121, 242)
    digest = hashlib.sha256()
    digest.update(
        (
            f"{STAR_LINEARIZATION_SCHEMA}|{apex}|{remaining}|"
            f"{columns}|{targets}\n"
        ).encode("ascii")
    )
    for color, block in enumerate(star_blocks):
        digest.update(
            f"star:{color}:{','.join(map(str, block))}\n".encode(
                "ascii"
            )
        )
    for row_index, row in enumerate(entries):
        for column_index, polynomial in enumerate(row):
            if not polynomial:
                continue
            digest.update(f"{row_index}:{column_index}|".encode("ascii"))
            for monomial in polynomial:
                digest.update(
                    f"{monomial[0]},{monomial[1]};".encode("ascii")
                )
            digest.update(b"\n")
    return digest.hexdigest()


@lru_cache(maxsize=6)
def independent_star_linearization_audit(apex: int) -> dict:
    """Rebuild one star matrix and all 729 equations independently."""

    apex = _validated_apex(apex)
    remaining = tuple(vertex for vertex in range(N) if vertex != apex)
    columns = tuple(
        (partner, partner_color)
        for partner in remaining
        for partner_color in range(D)
    )
    star_blocks = tuple(
        tuple(
            _variable_index(
                apex, partner, apex_color, partner_color
            )
            for partner, partner_color in columns
        )
        for apex_color in range(D)
    )
    row_colorings = tuple(product(range(D), repeat=5))
    entries = tuple(
        tuple(
            _entry(
                apex,
                remaining,
                row_coloring,
                partner,
                partner_color,
            )
            for partner, partner_color in columns
        )
        for row_coloring in row_colorings
    )

    full_matchings = _matchings(tuple(range(N)))
    if len(full_matchings) != 15:
        raise IndependentStarLinearizationError(
            "the independent K6 matching census changed"
        )
    for apex_color in range(D):
        for row_index, row_coloring in enumerate(row_colorings):
            colors = dict(zip(remaining, row_coloring, strict=True))
            colors[apex] = apex_color
            reconstructed = []
            for column_index, polynomial in enumerate(entries[row_index]):
                star = star_blocks[apex_color][column_index]
                reconstructed.extend(
                    tuple(sorted((*monomial, star)))
                    for monomial in polynomial
                )
            expected = [
                tuple(sorted(
                    _variable_index(
                        left,
                        right,
                        colors[left],
                        colors[right],
                    )
                    for left, right in matching
                ))
                for matching in full_matchings
            ]
            if sorted(reconstructed) != sorted(expected):
                raise IndependentStarLinearizationError(
                    "an equation failed independent star replay"
                )

    nonzero_entries = sum(
        bool(polynomial)
        for row in entries
        for polynomial in row
    )
    coefficient_terms = sum(
        len(polynomial)
        for row in entries
        for polynomial in row
    )
    if (
        len(columns) != 15
        or len(row_colorings) != 243
        or nonzero_entries != 1_215
        or coefficient_terms != 3_645
    ):
        raise IndependentStarLinearizationError(
            "the independent star-matrix census changed"
        )
    return {
        "apex": apex,
        "rows": 243,
        "columns": 15,
        "star_variables": 45,
        "nonstar_variables": 90,
        "nonzero_entries": nonzero_entries,
        "coefficient_terms": coefficient_terms,
        "coefficient_degree": 2,
        "target_rows": [0, 121, 242],
        "fingerprint": _fingerprint(
            apex, remaining, columns, star_blocks, entries
        ),
        "all_729_equations_reconstructed": True,
    }


__all__ = [
    "IndependentStarLinearizationError",
    "STAR_LINEARIZATION_SCHEMA",
    "independent_star_linearization_audit",
]
