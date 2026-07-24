"""Exact necessary invariants for perfect-matching tensor targets.

The routines in this module are deliberately one-sided.  Dimension bounds
describe the closure of the whole image, flattenings and support characters
describe a target exactly, and support profiles can obstruct one declared
nonzero witness support.  Passing any of these checks does not certify a
witness or image membership, and no profile here proves that a target is
outside the image after all possible supports are considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb
from numbers import Integral
from typing import Iterable, Sequence

from experiments.krenn_quantum_graph.targets import ColoringTarget
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap


class KrennTensorInvariantError(ValueError):
    """An invariant request, vertex cut, or exact support is malformed."""


def _rational_rank(
    matrix: Sequence[Sequence[int | Fraction]],
) -> int:
    """Return deterministic Gaussian rank over ``Q``."""

    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise KrennTensorInvariantError(
            "exact rank requires a rectangular matrix"
        )
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        if selected != pivot_row:
            rows[pivot_row], rows[selected] = (
                rows[selected],
                rows[pivot_row],
            )
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot for value in rows[pivot_row]]
        for row in range(pivot_row + 1, len(rows)):
            if not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    rows[row], rows[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


@dataclass(frozen=True)
class ImageDimensionBounds:
    """Universal closure-dimension bounds from scalar vertex gauge."""

    n: int
    d: int
    parameter_dimension: int
    affine_ambient_dimension: int
    projective_ambient_dimension: int
    scalar_vertex_gauge_dimension: int
    affine_image_dimension_upper_bound: int
    projective_image_dimension_upper_bound: int

    @property
    def affine_codimension_lower_bound(self) -> int:
        return (
            self.affine_ambient_dimension
            - self.affine_image_dimension_upper_bound
        )

    @property
    def projective_codimension_lower_bound(self) -> int:
        return (
            self.projective_ambient_dimension
            - self.projective_image_dimension_upper_bound
        )

    @property
    def proper_affine_image_closure_certified(self) -> bool:
        return self.affine_codimension_lower_bound > 0

    @property
    def proper_projective_image_closure_certified(self) -> bool:
        return self.projective_codimension_lower_bound > 0

    @property
    def individual_target_nonimage_proved(self) -> bool:
        return False

    @property
    def necessary_invariant_only(self) -> bool:
        return True

    @property
    def claim_boundary(self) -> str:
        return (
            "The gauge bound may certify that the image closure is proper "
            "and hence excludes a generic target.  It does not prove that "
            "any named target lies outside the image."
        )


def universal_image_dimension_bounds(
    tensor_map: MatchingTensorMap,
) -> ImageDimensionBounds:
    """Bound affine and projective image dimensions for even ``n >= 4``.

    The color-independent action

    ``W[i,j,a,b] -> lambda_i lambda_j W[i,j,a,b]``

    fixes every coefficient when ``prod_i lambda_i = 1``.  For ``n >= 4``
    it gives an effective generic fiber of dimension ``n-1``.  At ``n=2``
    this action is trivial on the sole edge, so applying the same subtraction
    would be false; that case is rejected explicitly.
    """

    if not isinstance(tensor_map, MatchingTensorMap):
        raise KrennTensorInvariantError(
            "dimension bounds require a MatchingTensorMap"
        )
    if tensor_map.n < 4:
        raise KrennTensorInvariantError(
            "the effective (n-1)-gauge dimension bound requires n >= 4"
        )
    parameter_dimension = comb(tensor_map.n, 2) * tensor_map.d**2
    affine_ambient = tensor_map.d**tensor_map.n
    projective_ambient = affine_ambient - 1
    gauge_dimension = tensor_map.n - 1
    affine_upper = min(
        affine_ambient, parameter_dimension - gauge_dimension
    )
    projective_upper = min(
        projective_ambient,
        parameter_dimension - 1 - gauge_dimension,
    )
    return ImageDimensionBounds(
        n=tensor_map.n,
        d=tensor_map.d,
        parameter_dimension=parameter_dimension,
        affine_ambient_dimension=affine_ambient,
        projective_ambient_dimension=projective_ambient,
        scalar_vertex_gauge_dimension=gauge_dimension,
        affine_image_dimension_upper_bound=affine_upper,
        projective_image_dimension_upper_bound=projective_upper,
    )


@dataclass(frozen=True)
class SupportCharacterInvariant:
    """Exact support-character matrix for the local diagonal action."""

    n: int
    d: int
    support_size: int
    matrix: tuple[tuple[int, ...], ...]
    rank_over_q: int
    local_diagonal_stabilizer_dimension: int

    @property
    def column_count(self) -> int:
        return self.n * self.d

    @property
    def exact(self) -> bool:
        return (
            len(self.matrix) == self.support_size
            and all(len(row) == self.column_count for row in self.matrix)
            and self.local_diagonal_stabilizer_dimension
            == self.column_count - self.rank_over_q
        )

    @property
    def individual_target_nonimage_proved(self) -> bool:
        return False

    @property
    def claim_boundary(self) -> str:
        return (
            "This is the exact dimension of the target's affine "
            "local-diagonal stabilizer.  It is not an image-membership or "
            "nonimage certificate."
        )


def support_character_invariant(
    target: ColoringTarget,
) -> SupportCharacterInvariant:
    """Build rows ``sum_i e_(i,c_i)`` for nonzero target coefficients."""

    matrix = tuple(
        tuple(
            int(coloring[vertex] == color)
            for vertex in range(target.n)
            for color in range(target.d)
        )
        for coloring, _value in target.entries
    )
    rank = _rational_rank(matrix)
    result = SupportCharacterInvariant(
        n=target.n,
        d=target.d,
        support_size=target.support_size,
        matrix=matrix,
        rank_over_q=rank,
        local_diagonal_stabilizer_dimension=target.n * target.d - rank,
    )
    if not result.exact:
        raise KrennTensorInvariantError(
            "support-character invariant failed exact replay"
        )
    return result


def _canonical_vertex_cut(
    n: int, left_vertices: Iterable[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        raw = tuple(left_vertices)
    except TypeError as error:
        raise KrennTensorInvariantError(
            "a flattening cut must be an iterable of vertices"
        ) from error
    if any(not isinstance(vertex, Integral) for vertex in raw):
        raise KrennTensorInvariantError(
            "flattening vertices must be exact integers"
        )
    vertices = tuple(map(int, raw))
    if len(vertices) != len(set(vertices)):
        raise KrennTensorInvariantError(
            "flattening vertices must be unique"
        )
    if any(vertex < 0 or vertex >= n for vertex in vertices):
        raise KrennTensorInvariantError(
            "a flattening vertex is outside the tensor factors"
        )
    left = tuple(sorted(vertices))
    if not left or len(left) == n:
        raise KrennTensorInvariantError(
            "a flattening cut must have two nonempty sides"
        )
    left_set = set(left)
    right = tuple(
        vertex for vertex in range(n) if vertex not in left_set
    )
    return left, right


@dataclass(frozen=True)
class RationalFlattening:
    """One exact computational-basis flattening matrix over ``Q``."""

    n: int
    d: int
    left_vertices: tuple[int, ...]
    right_vertices: tuple[int, ...]
    matrix: tuple[tuple[Fraction, ...], ...]
    rank_over_q: int

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.matrix), len(self.matrix[0])

    @property
    def exact(self) -> bool:
        return self.rank_over_q == _rational_rank(self.matrix)

    @property
    def individual_target_nonimage_proved(self) -> bool:
        return False

    @property
    def claim_boundary(self) -> str:
        return (
            "The flattening rank is an exact target invariant.  Without a "
            "separate image or ansatz rank upper bound, it proves neither "
            "membership nor nonmembership."
        )


def rational_flattening(
    target: ColoringTarget,
    left_vertices: Iterable[int],
) -> RationalFlattening:
    """Return the exact flattening across ``left | complement(left)``."""

    left, right = _canonical_vertex_cut(
        target.n, left_vertices
    )
    coefficients = dict(target.entries)
    rows: list[tuple[Fraction, ...]] = []
    for left_coloring in product(
        range(target.d), repeat=len(left)
    ):
        row: list[Fraction] = []
        for right_coloring in product(
            range(target.d), repeat=len(right)
        ):
            coloring = [0] * target.n
            for vertex, color in zip(
                left, left_coloring, strict=True
            ):
                coloring[vertex] = color
            for vertex, color in zip(
                right, right_coloring, strict=True
            ):
                coloring[vertex] = color
            row.append(
                Fraction(coefficients.get(tuple(coloring), 0))
            )
        rows.append(tuple(row))
    matrix = tuple(rows)
    result = RationalFlattening(
        n=target.n,
        d=target.d,
        left_vertices=left,
        right_vertices=right,
        matrix=matrix,
        rank_over_q=_rational_rank(matrix),
    )
    if not result.exact:
        raise KrennTensorInvariantError(
            "rational flattening failed exact replay"
        )
    return result


def _canonical_exact_support(
    tensor_map: MatchingTensorMap,
    support: Iterable[int],
) -> tuple[int, ...]:
    try:
        raw = tuple(support)
    except TypeError as error:
        raise KrennTensorInvariantError(
            "a witness support must be an iterable"
        ) from error
    if any(not isinstance(index, Integral) for index in raw):
        raise KrennTensorInvariantError(
            "support variable indices must be exact integers"
        )
    values = tuple(map(int, raw))
    if len(values) != len(set(values)):
        raise KrennTensorInvariantError(
            "support variable indices must be unique"
        )
    canonical = tuple(sorted(values))
    if any(
        index < 0 or index >= tensor_map.variable_count
        for index in canonical
    ):
        raise KrennTensorInvariantError(
            "a support variable lies outside the tensor map"
        )
    return canonical


@dataclass(frozen=True)
class TargetSupportProfile:
    """Necessary monomial-count profile for one exact nonzero support."""

    n: int
    d: int
    support: tuple[int, ...]
    feasible_matching_counts: tuple[int, ...]
    missing_nonzero_target_coefficients: tuple[int, ...]
    singleton_zero_target_coefficients: tuple[int, ...]

    @property
    def necessary_condition_satisfied(self) -> bool:
        return not (
            self.missing_nonzero_target_coefficients
            or self.singleton_zero_target_coefficients
        )

    @property
    def exact_support_excluded(self) -> bool:
        return not self.necessary_condition_satisfied

    @property
    def necessary_condition_only(self) -> bool:
        return True

    @property
    def solution_certified(self) -> bool:
        return False

    @property
    def individual_target_nonimage_proved(self) -> bool:
        return False

    @property
    def claim_boundary(self) -> str:
        if self.exact_support_excluded:
            return (
                "The declared exact nonzero support cannot realize the "
                "target over Q or C.  Other supports remain unexamined, so "
                "target nonimage is not proved."
            )
        return (
            "The support passed a necessary monomial-count condition only; "
            "no weights, solution, image membership, or border membership "
            "are certified."
        )


def target_support_profile(
    tensor_map: MatchingTensorMap,
    target: ColoringTarget,
    support: Iterable[int],
) -> TargetSupportProfile:
    """Profile one exact nonzero support against an arbitrary target."""

    if (target.n, target.d) != (tensor_map.n, tensor_map.d):
        raise KrennTensorInvariantError(
            "target dimensions do not match the tensor map"
        )
    canonical = _canonical_exact_support(tensor_map, support)
    support_set = set(canonical)
    target_values = target.dense_coefficients()
    counts: list[int] = []
    missing_nonzero: list[int] = []
    singleton_zero: list[int] = []
    for equation, expected in enumerate(target_values):
        count = sum(
            all(variable in support_set for variable in monomial)
            for monomial in tensor_map.equation_monomials(equation)
        )
        counts.append(count)
        if expected != 0 and count == 0:
            missing_nonzero.append(equation)
        if expected == 0 and count == 1:
            singleton_zero.append(equation)
    return TargetSupportProfile(
        n=tensor_map.n,
        d=tensor_map.d,
        support=canonical,
        feasible_matching_counts=tuple(counts),
        missing_nonzero_target_coefficients=tuple(missing_nonzero),
        singleton_zero_target_coefficients=tuple(singleton_zero),
    )
