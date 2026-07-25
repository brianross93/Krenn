r"""Exact gauge-normalized affine slices of the ``n=6,d=3`` star system.

Fix apex vertex zero.  On the three residual constant-color rows, the
shared star matrix has the block form

``B[(b), (v,c)] = delta_(b,c) P_v(bbbb)``.

At a direct-GHZ solution ``B Y = I_3``.  Consequently, for every color
``b`` at least one of the five factors ``P_v(bbbb)`` is nonzero.  The
``5**3`` choices of one such partner per color form an exact open cover.
They have three orbits under the residual ``S5`` vertex action and the
``S3`` color action.

The direct-GHZ color-diagonal gauge can normalize the three selected
factors to one.  The resulting polynomial systems are honest affine
slices: they contain the equations ``P_v(bbbb)-1`` and require neither
Rabinowitsch variables nor ideal saturation.  This is an
existence-equivalent gauge slice, not an assertion that the original
principal open is itself isomorphic to the slice.

Two deterministic sparse presentations are retained:

* ``retained`` keeps all 135 weights and the original 729 cubic
  equations, then adds three quadratic normalizations;
* ``eliminated`` uses the nine now-monic restricted equations to solve
  nine star weights.  It has 126 variables and 723 generators, but
  substitution raises the maximum degree to five.

An independent subset-filter matching enumerator rebuilds both
presentations and all nine reconstruction formulas.  No symmetry-related
weights are identified.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
from math import isqrt
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.star_linearization import (
    star_linearization,
)
from experiments.krenn_quantum_graph.system import (
    generate_sparse_system,
    variable_index,
)


N = 6
D = 3
APEX = 0
AMBIENT_VARIABLES = 135
TARGET_ROWS = (0, 121, 242)
PRESENTATION_KINDS = ("retained", "eliminated")
STAR_PIVOT_AFFINE_PRESENTATION_SCHEMA = (
    "krenn-n6-d3-star-pivot-affine-presentation-v1"
)
STAR_PIVOT_AFFINE_AUDIT_SCHEMA = (
    "krenn-n6-d3-star-pivot-affine-audit-v1"
)
STAR_PIVOT_AFFINE_SINGULAR_SCHEMA = (
    "krenn-n6-d3-star-pivot-affine-singular-v1"
)
EXPECTED_PRESENTATION_SHA256 = {
    (0, "retained"):
        "41c7aebc6263ed0d29c036566e357cd9caa2c6e3ac79f29ba6b7617c93ff9f18",
    (0, "eliminated"):
        "67de119eaba3c9d6f73e9f52a13b38fde61febf1d47ca7f54a3a994abc54ad5a",
    (1, "retained"):
        "c50942b52b9fdac74aee50769545f51f3bf346599505f7791aa451c71cc9301f",
    (1, "eliminated"):
        "ade0f9391ec983bd33d3c4ab125e94e650a59d8644385ecc4b65dbec41db493f",
    (2, "retained"):
        "6b795fa5cdf087999d2fc6a8fc1fae12c5ab4a8dfe2a42126e046f1266c8eb3f",
    (2, "eliminated"):
        "255aef6e3adb51c24143b3094e688f79cc810d509bb4095841c3e3fc3ce936cf",
}
EXPECTED_AUDIT_SHA256 = (
    "9e21df0acf374e499e4732649621e67b363071afbbef7d65d8eb72437b112c8a"
)


class KrennStarPivotAffineSliceError(RuntimeError):
    """An affine slice, exact replay, or claim boundary changed."""


@dataclass(frozen=True)
class PivotOrbit:
    """One ``S5 x S3`` orbit of partner triples."""

    index: int
    equality_pattern: str
    partner_by_color: tuple[int, int, int]
    orbit_size: int
    stabilizer_order: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or self.index not in range(3)
            or len(self.partner_by_color) != D
            or any(
                partner not in range(1, N)
                for partner in self.partner_by_color
            )
            or self.orbit_size * self.stabilizer_order != 720
        ):
            raise KrennStarPivotAffineSliceError(
                "a pivot orbit specification is malformed"
            )

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "equality_pattern": self.equality_pattern,
            "partner_triple_by_color": list(self.partner_by_color),
            "orbit_size": self.orbit_size,
            "stabilizer_order": self.stabilizer_order,
        }


PIVOT_ORBITS = (
    PivotOrbit(0, "all-same", (1, 1, 1), 5, 144),
    PivotOrbit(1, "exactly-two-same", (1, 1, 2), 60, 12),
    PivotOrbit(2, "all-distinct", (1, 2, 3), 60, 12),
)


def _orbit(orbit_index: int) -> PivotOrbit:
    if isinstance(orbit_index, bool):
        raise KrennStarPivotAffineSliceError(
            "a pivot orbit index must be an integer"
        )
    try:
        orbit_index = int(orbit_index)
    except (TypeError, ValueError) as error:
        raise KrennStarPivotAffineSliceError(
            "a pivot orbit index must be an integer"
        ) from error
    if orbit_index not in range(len(PIVOT_ORBITS)):
        raise KrennStarPivotAffineSliceError(
            "a pivot orbit index is out of range"
        )
    return PIVOT_ORBITS[orbit_index]


def _kind(kind: str) -> str:
    if kind not in PRESENTATION_KINDS:
        raise KrennStarPivotAffineSliceError(
            "presentation kind must be retained or eliminated"
        )
    return kind


def _monomial_key(monomial: tuple[int, ...]) -> tuple:
    return len(monomial), monomial


@dataclass(frozen=True)
class SparsePolynomial:
    """A collected integer polynomial in ambient weight indices."""

    terms: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        normalized = tuple(
            (int(coefficient), tuple(map(int, monomial)))
            for coefficient, monomial in self.terms
        )
        if (
            not normalized
            or normalized
            != tuple(sorted(
                normalized,
                key=lambda term: _monomial_key(term[1]),
            ))
            or any(not coefficient for coefficient, _ in normalized)
            or any(
                monomial != tuple(sorted(monomial))
                or any(
                    variable not in range(AMBIENT_VARIABLES)
                    for variable in monomial
                )
                for _coefficient, monomial in normalized
            )
            or len({monomial for _coefficient, monomial in normalized})
            != len(normalized)
        ):
            raise KrennStarPivotAffineSliceError(
                "a sparse polynomial is not canonical"
            )
        object.__setattr__(self, "terms", normalized)

    @property
    def degree(self) -> int:
        return max(len(monomial) for _coefficient, monomial in self.terms)

    @property
    def degree_set(self) -> tuple[int, ...]:
        return tuple(sorted({
            len(monomial) for _coefficient, monomial in self.terms
        }))

    @property
    def term_count(self) -> int:
        return len(self.terms)

    def evaluate(
        self,
        values: Mapping[int, Fraction | int],
    ) -> Fraction:
        total = Fraction(0)
        for coefficient, monomial in self.terms:
            term = Fraction(coefficient)
            for variable in monomial:
                try:
                    term *= Fraction(values[variable])
                except KeyError as error:
                    raise KrennStarPivotAffineSliceError(
                        "a polynomial evaluation is missing a variable"
                    ) from error
            total += term
        return total


def _polynomial(counter: Mapping[tuple[int, ...], int]) -> SparsePolynomial:
    collected = {
        tuple(sorted(map(int, monomial))): int(coefficient)
        for monomial, coefficient in counter.items()
        if coefficient
    }
    return SparsePolynomial(tuple(
        (coefficient, monomial)
        for monomial, coefficient in sorted(
            collected.items(), key=lambda item: _monomial_key(item[0])
        )
    ))


def _counter(polynomial: SparsePolynomial) -> Counter:
    return Counter({
        monomial: coefficient
        for coefficient, monomial in polynomial.terms
    })


def _multiply(
    left: SparsePolynomial,
    right: SparsePolynomial,
) -> SparsePolynomial:
    result: Counter = Counter()
    for left_coefficient, left_monomial in left.terms:
        for right_coefficient, right_monomial in right.terms:
            result[tuple(sorted(
                (*left_monomial, *right_monomial)
            ))] += left_coefficient * right_coefficient
    return _polynomial(result)


def _substitute(
    polynomial: SparsePolynomial,
    replacements: Mapping[int, SparsePolynomial],
) -> SparsePolynomial:
    result: Counter = Counter()
    for coefficient, monomial in polynomial.terms:
        partial = SparsePolynomial(((coefficient, ()),))
        for variable in monomial:
            factor = replacements.get(
                variable,
                SparsePolynomial(((1, (variable,)),)),
            )
            partial = _multiply(partial, factor)
        result.update(_counter(partial))
    return _polynomial(result)


def _polynomial_fingerprint(
    polynomials: Sequence[SparsePolynomial],
) -> str:
    digest = hashlib.sha256()
    for polynomial_index, polynomial in enumerate(polynomials):
        digest.update(f"{polynomial_index}|".encode("ascii"))
        for coefficient, monomial in polynomial.terms:
            digest.update(
                (
                    f"{coefficient}:"
                    f"{','.join(map(str, monomial))};"
                ).encode("ascii")
            )
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class StarPivotAffinePresentation:
    """One retained or nine-variable-eliminated affine slice."""

    orbit: PivotOrbit
    kind: str
    variable_indices: tuple[int, ...]
    eliminated_variable_indices: tuple[int, ...]
    pivot_columns_by_color: tuple[int, int, int]
    generators: tuple[SparsePolynomial, ...]
    reconstruction_polynomials: tuple[
        tuple[int, SparsePolynomial], ...
    ]
    schema: str = STAR_PIVOT_AFFINE_PRESENTATION_SCHEMA

    def __post_init__(self) -> None:
        kind = _kind(self.kind)
        if (
            self.schema != STAR_PIVOT_AFFINE_PRESENTATION_SCHEMA
            or self.orbit != _orbit(self.orbit.index)
            or len(self.pivot_columns_by_color) != D
            or len(set(self.variable_indices))
            != len(self.variable_indices)
            or self.variable_indices
            != tuple(sorted(self.variable_indices))
            or self.eliminated_variable_indices
            != tuple(sorted(self.eliminated_variable_indices))
            or set(self.variable_indices).intersection(
                self.eliminated_variable_indices
            )
            or set(self.variable_indices).union(
                self.eliminated_variable_indices
            )
            != set(range(AMBIENT_VARIABLES))
        ):
            raise KrennStarPivotAffineSliceError(
                "an affine presentation has malformed coordinates"
            )
        expected = (
            (AMBIENT_VARIABLES, 732, 3, 10_950, 0)
            if kind == "retained"
            else (126, 723, 5, 35_292, 9)
        )
        actual = (
            len(self.variable_indices),
            len(self.generators),
            self.maximum_degree,
            self.term_count,
            len(self.eliminated_variable_indices),
        )
        if actual != expected:
            raise KrennStarPivotAffineSliceError(
                f"the {kind} affine census changed: {actual}"
            )
        reconstruction = dict(self.reconstruction_polynomials)
        if (
            tuple(sorted(reconstruction))
            != self.eliminated_variable_indices
            or (
                kind == "retained"
                and self.reconstruction_polynomials
            )
            or any(
                set(
                    variable
                    for _coefficient, monomial in polynomial.terms
                    for variable in monomial
                ).intersection(self.eliminated_variable_indices)
                for polynomial in reconstruction.values()
            )
        ):
            raise KrennStarPivotAffineSliceError(
                "the nine star reconstruction formulas changed"
            )

    @property
    def maximum_degree(self) -> int:
        return max(generator.degree for generator in self.generators)

    @property
    def term_count(self) -> int:
        return sum(generator.term_count for generator in self.generators)

    @property
    def degree_term_census(self) -> dict[int, int]:
        result: Counter = Counter()
        for generator in self.generators:
            for _coefficient, monomial in generator.terms:
                result[len(monomial)] += 1
        return dict(sorted(result.items()))

    @property
    def maximum_degree_histogram(self) -> dict[int, int]:
        return dict(sorted(Counter(
            generator.degree for generator in self.generators
        ).items()))

    @property
    def generator_term_count_histogram(self) -> dict[int, int]:
        return dict(sorted(Counter(
            generator.term_count for generator in self.generators
        ).items()))

    @property
    def degree_set_histogram(self) -> dict[str, int]:
        return dict(sorted(Counter(
            ",".join(map(str, generator.degree_set))
            for generator in self.generators
        ).items()))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.orbit.index}|{self.kind}|"
                f"{self.variable_indices}|"
                f"{self.eliminated_variable_indices}|"
                f"{self.pivot_columns_by_color}\n"
            ).encode("ascii")
        )
        digest.update(
            _polynomial_fingerprint(self.generators).encode("ascii")
        )
        digest.update(b"\nreconstruction\n")
        for variable, polynomial in self.reconstruction_polynomials:
            digest.update(f"{variable}|".encode("ascii"))
            digest.update(
                _polynomial_fingerprint((polynomial,)).encode("ascii")
            )
            digest.update(b"\n")
        return digest.hexdigest()

    def to_summary(self) -> dict:
        return {
            "schema": self.schema,
            "orbit_index": self.orbit.index,
            "equality_pattern": self.orbit.equality_pattern,
            "kind": self.kind,
            "ambient_weight_variables": AMBIENT_VARIABLES,
            "ring_variable_count": len(self.variable_indices),
            "generator_count": len(self.generators),
            "maximum_degree": self.maximum_degree,
            "collected_term_count": self.term_count,
            "degree_term_census": {
                str(degree): count
                for degree, count in self.degree_term_census.items()
            },
            "maximum_degree_histogram": {
                str(degree): count
                for degree, count
                in self.maximum_degree_histogram.items()
            },
            "generator_term_count_histogram": {
                str(count): generators
                for count, generators
                in self.generator_term_count_histogram.items()
            },
            "degree_set_histogram": self.degree_set_histogram,
            "eliminated_star_variable_count": len(
                self.eliminated_variable_indices
            ),
            "eliminated_star_variable_indices": list(
                self.eliminated_variable_indices
            ),
            "remaining_variables_are_independent_coordinates": True,
            "symmetry_related_weights_identified": False,
            "sha256": self.fingerprint(),
        }


@lru_cache(maxsize=1)
def _primary_original_polynomials() -> tuple[SparsePolynomial, ...]:
    system = generate_sparse_system(N, D)
    rows = []
    for equation in range(system.equation_count):
        counter: Counter = Counter()
        for monomial in system.equation_monomials(equation):
            counter[tuple(sorted(monomial))] += 1
        if system.rhs_values[equation]:
            counter[()] -= int(system.rhs_values[equation])
        rows.append(_polynomial(counter))
    result = tuple(rows)
    if (
        len(result) != 729
        or sum(row.term_count for row in result) != 10_938
    ):
        raise KrennStarPivotAffineSliceError(
            "the original sparse polynomial census changed"
        )
    return result


def _primary_chart_data(
    orbit_index: int,
) -> tuple[
    tuple[int, int, int],
    tuple[SparsePolynomial, SparsePolynomial, SparsePolynomial],
    tuple[tuple[int, SparsePolynomial], ...],
]:
    orbit = _orbit(orbit_index)
    factorization = star_linearization(APEX)
    pivot_columns = tuple(
        factorization.columns.index((partner, color))
        for color, partner in enumerate(orbit.partner_by_color)
    )
    normalizations = []
    for color, column in enumerate(pivot_columns):
        counter: Counter = Counter({(): -1})
        for monomial in factorization.entries[
            TARGET_ROWS[color]
        ][column]:
            counter[monomial] += 1
        normalizations.append(_polynomial(counter))

    reconstruction = []
    for apex_color in range(D):
        for residual_color, pivot_column in enumerate(pivot_columns):
            counter: Counter = Counter()
            if apex_color == residual_color:
                counter[()] = 1
            target_row = TARGET_ROWS[residual_color]
            for column, coefficient in enumerate(
                factorization.entries[target_row]
            ):
                if column == pivot_column or not coefficient:
                    continue
                star = factorization.star_variable_blocks[
                    apex_color
                ][column]
                for monomial in coefficient:
                    counter[tuple(sorted((*monomial, star)))] -= 1
            pivot_variable = factorization.star_variable_blocks[
                apex_color
            ][pivot_column]
            reconstruction.append((
                pivot_variable, _polynomial(counter)
            ))
    reconstruction.sort(key=lambda row: row[0])
    return (
        pivot_columns,
        tuple(normalizations),
        tuple(reconstruction),
    )


def _restricted_equations() -> frozenset[int]:
    return frozenset(
        apex_color * 243 + target_row
        for apex_color in range(D)
        for target_row in TARGET_ROWS
    )


@lru_cache(maxsize=6)
def _star_pivot_affine_presentation_cached(
    orbit_index: int,
    kind: str,
) -> StarPivotAffinePresentation:
    orbit = _orbit(orbit_index)
    pivot_columns, normalizations, reconstruction = (
        _primary_chart_data(orbit.index)
    )
    original = _primary_original_polynomials()
    eliminated = tuple(variable for variable, _ in reconstruction)
    if len(eliminated) != 9 or len(set(eliminated)) != 9:
        raise KrennStarPivotAffineSliceError(
            "the selected target minor did not yield nine star weights"
        )

    if kind == "retained":
        return StarPivotAffinePresentation(
            orbit=orbit,
            kind=kind,
            variable_indices=tuple(range(AMBIENT_VARIABLES)),
            eliminated_variable_indices=(),
            pivot_columns_by_color=pivot_columns,
            generators=(*original, *normalizations),
            reconstruction_polynomials=(),
        )

    replacements = dict(reconstruction)
    restricted = _restricted_equations()
    generators = tuple(
        _substitute(polynomial, replacements)
        for equation, polynomial in enumerate(original)
        if equation not in restricted
    ) + normalizations
    return StarPivotAffinePresentation(
        orbit=orbit,
        kind=kind,
        variable_indices=tuple(
            variable
            for variable in range(AMBIENT_VARIABLES)
            if variable not in set(eliminated)
        ),
        eliminated_variable_indices=eliminated,
        pivot_columns_by_color=pivot_columns,
        generators=generators,
        reconstruction_polynomials=reconstruction,
    )


def star_pivot_affine_presentation(
    orbit_index: int,
    kind: str,
) -> StarPivotAffinePresentation:
    """Build one exact retained or eliminated affine presentation."""

    orbit = _orbit(orbit_index)
    normalized_kind = _kind(kind)
    return _star_pivot_affine_presentation_cached(
        orbit.index, normalized_kind
    )


def reconstruct_nine_star_weights(
    orbit_index: int,
    free_values: (
        Mapping[int, Fraction | int]
        | Sequence[Fraction | int]
    ),
) -> tuple[Fraction, ...]:
    """Reconstruct the nine eliminated weights exactly.

    A sequence is interpreted in the presentation's sorted
    ``variable_indices`` order.  A mapping must contain exactly those
    indices, preventing an accidental second value for an eliminated
    coordinate.
    """

    presentation = star_pivot_affine_presentation(
        orbit_index, "eliminated"
    )
    if isinstance(free_values, Mapping):
        if set(free_values) != set(presentation.variable_indices):
            raise KrennStarPivotAffineSliceError(
                "free-value mapping does not match the 126 coordinates"
            )
        values = {
            int(variable): Fraction(value)
            for variable, value in free_values.items()
        }
    else:
        normalized = tuple(map(Fraction, free_values))
        if len(normalized) != len(presentation.variable_indices):
            raise KrennStarPivotAffineSliceError(
                "free-value sequence must contain 126 entries"
            )
        values = dict(zip(
            presentation.variable_indices,
            normalized,
            strict=True,
        ))
    for variable, polynomial in presentation.reconstruction_polynomials:
        values[variable] = polynomial.evaluate(values)
    if set(values) != set(range(AMBIENT_VARIABLES)):
        raise KrennStarPivotAffineSliceError(
            "nine-weight reconstruction did not fill the ambient vector"
        )
    return tuple(values[index] for index in range(AMBIENT_VARIABLES))


# The following constructors intentionally do not call the primary matching,
# system, or star-factorization APIs.


@lru_cache(maxsize=None)
def _independent_matchings(
    vertices: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    edges = tuple(combinations(vertices, 2))
    rows = []
    for candidate in combinations(edges, len(vertices) // 2):
        covered = tuple(vertex for edge in candidate for vertex in edge)
        if len(set(covered)) == len(vertices):
            rows.append(candidate)
    return tuple(rows)


def _independent_variable_index(
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
        raise KrennStarPivotAffineSliceError(
            "an independent variable coordinate is invalid"
        )
    edge_index = left * (2 * N - left - 1) // 2 + (
        right - left - 1
    )
    return (edge_index * D + left_color) * D + right_color


@lru_cache(maxsize=1)
def _independent_original_polynomials() -> tuple[
    SparsePolynomial, ...
]:
    matchings = _independent_matchings(tuple(range(N)))
    if len(matchings) != 15:
        raise KrennStarPivotAffineSliceError(
            "the independent K6 matching census changed"
        )
    rows = []
    for coloring in product(range(D), repeat=N):
        counter: Counter = Counter()
        for matching in matchings:
            monomial = tuple(sorted(
                _independent_variable_index(
                    left,
                    right,
                    coloring[left],
                    coloring[right],
                )
                for left, right in matching
            ))
            counter[monomial] += 1
        if all(color == coloring[0] for color in coloring[1:]):
            counter[()] -= 1
        rows.append(_polynomial(counter))
    return tuple(rows)


def _independent_normalizations(
    orbit_index: int,
) -> tuple[SparsePolynomial, SparsePolynomial, SparsePolynomial]:
    orbit = _orbit(orbit_index)
    rows = []
    for color, partner in enumerate(orbit.partner_by_color):
        residual = tuple(
            vertex
            for vertex in range(N)
            if vertex not in (APEX, partner)
        )
        matchings = _independent_matchings(residual)
        if len(matchings) != 3:
            raise KrennStarPivotAffineSliceError(
                "an independent residual K4 census changed"
            )
        counter: Counter = Counter({(): -1})
        for matching in matchings:
            counter[tuple(sorted(
                _independent_variable_index(
                    left, right, color, color
                )
                for left, right in matching
            ))] += 1
        rows.append(_polynomial(counter))
    return tuple(rows)


def _independent_reconstruction(
    orbit_index: int,
) -> tuple[tuple[int, SparsePolynomial], ...]:
    orbit = _orbit(orbit_index)
    original = _independent_original_polynomials()
    rows = []
    for apex_color in range(D):
        for residual_color, partner in enumerate(
            orbit.partner_by_color
        ):
            pivot = _independent_variable_index(
                APEX,
                partner,
                apex_color,
                residual_color,
            )
            equation = original[
                apex_color * 243 + TARGET_ROWS[residual_color]
            ]
            pivot_terms = 0
            counter: Counter = Counter()
            for coefficient, monomial in equation.terms:
                if pivot in monomial:
                    pivot_terms += 1
                else:
                    counter[monomial] -= coefficient
            if pivot_terms != 3:
                raise KrennStarPivotAffineSliceError(
                    "an independent target pivot lost a K4 term"
                )
            rows.append((pivot, _polynomial(counter)))
    rows.sort(key=lambda row: row[0])
    return tuple(rows)


def _independent_presentations(
    orbit_index: int,
) -> tuple[
    tuple[SparsePolynomial, ...],
    tuple[SparsePolynomial, ...],
    tuple[tuple[int, SparsePolynomial], ...],
]:
    original = _independent_original_polynomials()
    normalizations = _independent_normalizations(orbit_index)
    reconstruction = _independent_reconstruction(orbit_index)
    replacements = dict(reconstruction)
    restricted = _restricted_equations()
    retained = (*original, *normalizations)
    eliminated = tuple(
        _substitute(polynomial, replacements)
        for equation, polynomial in enumerate(original)
        if equation not in restricted
    ) + normalizations
    return retained, eliminated, reconstruction


def _target_identity_replay(
    orbit_index: int,
    normalizations: Sequence[SparsePolynomial],
    reconstruction: Sequence[tuple[int, SparsePolynomial]],
) -> bool:
    original = _independent_original_polynomials()
    replacements = dict(reconstruction)
    orbit = _orbit(orbit_index)
    for apex_color in range(D):
        for residual_color, partner in enumerate(
            orbit.partner_by_color
        ):
            pivot = _independent_variable_index(
                APEX,
                partner,
                apex_color,
                residual_color,
            )
            equation = original[
                apex_color * 243 + TARGET_ROWS[residual_color]
            ]
            substituted = _substitute(equation, replacements)
            expected = _multiply(
                replacements[pivot],
                normalizations[residual_color],
            )
            if substituted != expected:
                return False
    return True


def _independent_replay_row(orbit_index: int) -> dict:
    retained = star_pivot_affine_presentation(
        orbit_index, "retained"
    )
    eliminated = star_pivot_affine_presentation(
        orbit_index, "eliminated"
    )
    independent_retained, independent_eliminated, reconstruction = (
        _independent_presentations(orbit_index)
    )
    primary_reconstruction = eliminated.reconstruction_polynomials
    normalizations = independent_retained[-3:]
    checks = {
        "retained_generators_match_subset_filter_replay": (
            retained.generators == independent_retained
        ),
        "eliminated_generators_match_subset_filter_replay": (
            eliminated.generators == independent_eliminated
        ),
        "nine_reconstruction_formulas_match_independently": (
            primary_reconstruction == reconstruction
        ),
        "all_nine_target_equation_identities_replayed": (
            _target_identity_replay(
                orbit_index, normalizations, reconstruction
            )
        ),
        "remaining_126_variables_are_not_identified": (
            len(eliminated.variable_indices)
            == len(set(eliminated.variable_indices))
            == 126
        ),
    }
    if not all(checks.values()):
        raise KrennStarPivotAffineSliceError(
            "an independent affine-slice replay failed"
        )
    return {
        "orbit_index": orbit_index,
        "checks": checks,
        "independent_matching_method": (
            "filter edge subsets for exact vertex coverage"
        ),
    }


def _json_sha256(payload: Mapping) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def star_pivot_affine_slice_audit() -> dict:
    """Return the strict three-orbit affine-slice audit."""

    presentations = []
    independent = []
    for orbit in PIVOT_ORBITS:
        retained = star_pivot_affine_presentation(
            orbit.index, "retained"
        )
        eliminated = star_pivot_affine_presentation(
            orbit.index, "eliminated"
        )
        presentations.append({
            "orbit": orbit.to_dict(),
            "retained": retained.to_summary(),
            "eliminated": eliminated.to_summary(),
        })
        for presentation in (retained, eliminated):
            if (
                presentation.fingerprint()
                != EXPECTED_PRESENTATION_SHA256[
                    (orbit.index, presentation.kind)
                ]
            ):
                raise KrennStarPivotAffineSliceError(
                    "an affine presentation fingerprint changed"
                )
        independent.append(_independent_replay_row(orbit.index))

    payload = {
        "schema": STAR_PIVOT_AFFINE_AUDIT_SCHEMA,
        "parameters": {
            "n": N,
            "d": D,
            "fixed_apex": APEX,
            "ambient_weight_variables": AMBIENT_VARIABLES,
        },
        "cover": {
            "target_restriction": "B*Y=I_3",
            "open_triples_before_symmetry": 125,
            "S5_times_S3_orbit_count": 3,
            "orbit_sizes": [5, 60, 60],
            "every_solution_is_in_at_least_one_open": True,
            "reason": (
                "each of the three disjoint five-column target-row "
                "blocks must contain a nonzero P_v(bbbb)"
            ),
            "one_fixed_apex_already_covers_every_solution": True,
        },
        "gauge_normalization": {
            "gauge": "direct-GHZ color-diagonal torus",
            "independent_parameters_after_apex_elimination": 15,
            "normalizations_used": 3,
            "residual_gauge_dimension": 12,
            "factor_character": (
                "P_v(bbbb) -> "
                "(product_(u != apex,v) lambda_(u,b))*P_v(bbbb)"
            ),
            "residual_vertex_parameters_that_change_one_factor": 4,
            "apex_parameter_absorbs_product_constraint": True,
            "requires_root_choice": False,
            "requires_Rabinowitsch_variable": False,
            "requires_colon_or_saturation": False,
            "is_a_gauge_normalized_affine_slice": True,
            "claimed_isomorphic_to_the_original_principal_open": False,
        },
        "presentations": presentations,
        "independent_replay": independent,
        "exact_reconstruction": {
            "selected_target_minor_becomes_identity": True,
            "monic_restricted_equations": 9,
            "star_weights_reconstructed": 9,
            "target_identity": (
                "E_(a,b) after reconstruction equals "
                "y_reconstructed_(a,b)*(P_(v_b)(b^4)-1)"
            ),
            "every_solution_of_an_eliminated_slice_reconstructs_a_"
            "135_weight_solution": True,
        },
        "CAS_guidance": {
            "recommended_first_presentation": "retained",
            "reason": (
                "maximum degree three and 10950 terms, versus degree "
                "five and 35292 terms after eliminating nine weights"
            ),
            "Singular_exports_generated_by_this_audit": False,
            "positive_characteristic_result_is_proof": False,
            "timeout_or_bounded_miss_is_proof": False,
        },
        "claim_boundary": {
            "any_affine_slice_solved": False,
            "finite_counterexample_found": False,
            "nonexistence_certificate_found": False,
            "global_affine_membership_decided": False,
            "strict_border_membership_decided": False,
        },
    }
    payload["sha256"] = _json_sha256(payload)
    if payload["sha256"] != EXPECTED_AUDIT_SHA256:
        raise KrennStarPivotAffineSliceError(
            "the affine-slice audit fingerprint changed"
        )
    return json.loads(json.dumps(payload, allow_nan=False))


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


def verify_star_pivot_affine_slice_audit(
    payload: Mapping,
) -> dict:
    """Regenerate every polynomial and reject altered claims."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennStarPivotAffineSliceError(
            "the affine-slice audit is not strict JSON"
        ) from error
    expected = star_pivot_affine_slice_audit()
    if not _strict_json_equal(normalized, expected):
        raise KrennStarPivotAffineSliceError(
            "the affine-slice audit failed strict exact replay"
        )
    return normalized


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    return all(
        value % divisor
        for divisor in range(3, isqrt(value) + 1, 2)
    )


def _singular_term(
    coefficient: int,
    monomial: Sequence[int],
    characteristic: int,
) -> str:
    coefficient = int(coefficient)
    if characteristic:
        coefficient %= characteristic
    factors = "*".join(f"x{variable}" for variable in monomial)
    if not factors:
        return str(coefficient)
    if coefficient == 1:
        return factors
    if coefficient == -1:
        return f"-{factors}"
    return f"{coefficient}*{factors}"


def _singular_polynomial(
    polynomial: SparsePolynomial,
    characteristic: int,
) -> str:
    pieces = [
        _singular_term(coefficient, monomial, characteristic)
        for coefficient, monomial in polynomial.terms
        if not characteristic or coefficient % characteristic
    ]
    if characteristic:
        return "+".join(pieces)
    result = pieces[0]
    for piece in pieces[1:]:
        result += piece if piece.startswith("-") else f"+{piece}"
    return result


def singular_star_pivot_affine_slice_script(
    orbit_index: int,
    kind: str = "retained",
    *,
    characteristic: int = 0,
    algorithm: str = "slimgb",
) -> str:
    """Return, but do not run, one deterministic Singular input.

    Characteristic zero can be conclusive if the exact standard basis
    finishes.  Positive-characteristic runs remain reconnaissance until
    an exact characteristic-zero certificate is reconstructed.
    """

    orbit = _orbit(orbit_index)
    kind = _kind(kind)
    if isinstance(characteristic, bool):
        raise KrennStarPivotAffineSliceError(
            "Singular characteristic must be zero or prime"
        )
    characteristic = int(characteristic)
    if characteristic != 0 and not _is_prime(characteristic):
        raise KrennStarPivotAffineSliceError(
            "Singular characteristic must be zero or prime"
        )
    if algorithm not in ("std", "slimgb"):
        raise KrennStarPivotAffineSliceError(
            "Singular algorithm must be std or slimgb"
        )
    presentation = star_pivot_affine_presentation(
        orbit.index, kind
    )
    variable_names = ",".join(
        f"x{variable}" for variable in presentation.variable_indices
    )
    generators = ",\n".join(
        _singular_polynomial(generator, characteristic)
        for generator in presentation.generators
    )
    return "\n".join((
        "// Exact Krenn n=6,d=3 gauge-normalized star slice.",
        f"// schema={STAR_PIVOT_AFFINE_SINGULAR_SCHEMA}",
        (
            f"// orbit={orbit.index}; pattern={orbit.equality_pattern}; "
            f"kind={kind}"
        ),
        (
            f"// variables={len(presentation.variable_indices)}; "
            f"generators={len(presentation.generators)}; "
            f"terms={presentation.term_count}"
        ),
        "// No Rabinowitsch variable, localization, or saturation.",
        f"ring r={characteristic},({variable_names}),dp;",
        f"ideal I={generators};",
        'print("KRENN_STAR_AFFINE_PARSE_OK");',
        f'print("schema={STAR_PIVOT_AFFINE_SINGULAR_SCHEMA}");',
        f'print("orbit_index={orbit.index}");',
        f'print("presentation={kind}");',
        f'print("characteristic={characteristic}");',
        f'print("algorithm={algorithm}");',
        "int started=timer;",
        f"ideal G={algorithm}(I);",
        "int elapsed=timer-started;",
        "poly unit_remainder=reduce(1,G);",
        "int is_unit=0;",
        "if (unit_remainder==0) { is_unit=1; }",
        'print("KRENN_STAR_AFFINE_GROEBNER_DONE");',
        'print("timer_ticks="+string(elapsed));',
        'print("basis_size="+string(size(G)));',
        'print("unit_ideal="+string(is_unit));',
        "exit;",
        "",
    ))


__all__ = [
    "APEX",
    "D",
    "EXPECTED_AUDIT_SHA256",
    "EXPECTED_PRESENTATION_SHA256",
    "KrennStarPivotAffineSliceError",
    "N",
    "PIVOT_ORBITS",
    "PivotOrbit",
    "SparsePolynomial",
    "STAR_PIVOT_AFFINE_AUDIT_SCHEMA",
    "STAR_PIVOT_AFFINE_PRESENTATION_SCHEMA",
    "STAR_PIVOT_AFFINE_SINGULAR_SCHEMA",
    "StarPivotAffinePresentation",
    "reconstruct_nine_star_weights",
    "singular_star_pivot_affine_slice_script",
    "star_pivot_affine_presentation",
    "star_pivot_affine_slice_audit",
    "verify_star_pivot_affine_slice_audit",
]
