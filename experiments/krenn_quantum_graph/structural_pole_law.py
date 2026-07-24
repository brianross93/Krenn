r"""Exact component and toric-circuit structure of the ``n=6,d=3`` pole.

Put ``v=(0,0,2,1,2,1)`` and let ``u=F_v(W)``.  The natural Laurent
degeneration satisfies

.. math::

    \Phi(W(u))=\operatorname{GHZ}_{6,3}+u e_v,\qquad uQ(W(u))=1.

This module makes two strictly bounded upgrades to that branch statement.

First, the exact rank certificate at the natural unit seed shows that the
six-dimensional vertex-gauge-plus-parameter family is the unique smooth
irreducible component of the incidence scheme through that seed.  Therefore
``u*Q-1`` vanishes on the whole component, not merely on the displayed
one-parameter Laurent curve.  In the formal-lift convention ``u=1-s``, so
the relation is ``(1-s)*Q=1``.

Second, all 360 unit nine-slot seed charts in the ``(0,4,8)`` seed orbit are
enumerated exactly.  They have one balanced defect each, four over every one
of the 90 balanced victim colorings.  In every resulting toric chart,
``Q`` is the quotient of the three pure matching monomials by the victim
monomial and the same ``u*Q=1`` law holds on the unique local component
through the marked seed.  Distinct charts are not asserted to define
distinct global irreducible components.

The six small retained-natural supports used by the deterministic numerical
campaign have an additional exact obstruction.  Their active mixed
equations are binomials.  On the exact-support torus each gives
``x**ell=-1``; an odd integer dependence among the exponent differences
would force ``1=-1``.  Literal three-equation dependences are replayed here.
An exhaustive Boolean sub-support audit then proves that every coordinate
boundary stratum inside each of those six declared coordinate subspaces
which can satisfy the elementary GHZ support conditions has its own such
odd dependence.

These statements do not identify which seed charts lie on the same global
component, do not classify components away from the marked seeds, and do
not decide global affine GHZ membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import gcd, lcm
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.border_image import (
    certify_n6_d3_laurent_border,
)
from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
    first_order_repair_direction,
    jacobian_action,
    pole_invariant_vertex_degrees,
)
from experiments.krenn_quantum_graph.n6_deformation import (
    certify_n6_natural_deformation,
)
from experiments.krenn_quantum_graph.system import (
    Monomial,
    coloring_from_index,
    coloring_index,
    generate_sparse_system,
    perfect_matchings,
    variable_index,
    variable_key,
)
from experiments.krenn_quantum_graph.ternary_search import (
    N6_D3_SEED_DEFECT_EQUATION,
    n6_d3_seed_witness,
)
from experiments.krenn_quantum_graph.ternary_seed_orbits import (
    canonical_seed_representative,
)


N = 6
D = 3
AMBIENT_VARIABLES = 135
INCIDENCE_AMBIENT_DIMENSION = 136
VICTIM_EQUATION = N6_D3_SEED_DEFECT_EQUATION
VICTIM_COLORING = (0, 0, 2, 1, 2, 1)
PURE_EQUATIONS = (0, 364, 728)
NATURAL_SEED = (0, 4, 8)
FIXED_VICTIM_SEEDS = (
    (0, 4, 8),
    (0, 9, 13),
    (2, 4, 13),
    (2, 9, 8),
)

NATURAL_COMPONENT_SCHEMA = "krenn-n6-d3-natural-pole-component-v1"
SEED_ATLAS_SCHEMA = "krenn-n6-d3-pole-component-atlas-v1"
TORIC_PARITY_SCHEMA = "krenn-n6-d3-toric-parity-circuit-v1"
COORDINATE_AUDIT_SCHEMA = "krenn-n6-d3-coordinate-subspace-audit-v1"
STRUCTURAL_POLE_SCHEMA = "krenn-n6-d3-structural-pole-law-v1"


# These are literal source inputs.  They are deliberately not read from a
# numerical campaign artifact.
RETAINED_NATURAL_SUPPORTS = (
    (
        0, 6, 13, 18, 20, 24, 26, 29, 35, 45, 47,
        67, 72, 80, 81, 83, 87, 89, 92, 98, 121, 126,
    ),
    (
        0, 13, 14, 26, 67, 68, 70, 71, 76, 77, 79,
        80, 81, 97, 98, 106, 107, 112, 113, 121, 122, 126,
    ),
    (
        0, 3, 9, 10, 12, 13, 26, 37, 40, 54, 55, 63,
        64, 67, 80, 81, 82, 84, 85, 98, 118, 121, 126,
    ),
    (
        0, 6, 13, 18, 20, 24, 26, 29, 35, 45, 47, 67,
        72, 73, 80, 81, 83, 87, 89, 92, 98, 121, 126,
    ),
    (
        0, 6, 13, 18, 20, 24, 26, 29, 35, 45, 47, 67,
        72, 73, 74, 80, 81, 83, 87, 89, 92, 98, 121, 126,
    ),
    (
        0, 13, 14, 15, 16, 26, 67, 68, 70, 71, 76, 77,
        79, 80, 81, 97, 98, 106, 107, 112, 113, 121, 122, 126,
    ),
)

# Each row uses the canonical first-minus-second active-monomial orientation.
RETAINED_SUPPORT_ODD_RELATIONS = (
    ((16, -1), (18, 1), (188, 1)),
    ((71, 1), (368, -1), (647, 1)),
    ((27, -1), (28, -1), (61, 1)),
    ((16, -1), (18, 1), (188, 1)),
    ((16, -1), (18, 1), (188, 1)),
    ((71, 1), (368, -1), (647, 1)),
)

EXPECTED_MOVING_ACTIVE_BINOMIALS = (25, 25, 33, 33, 41, 41)
EXPECTED_GHZ_ADMISSIBLE_SUBSUPPORTS = (2, 2, 4, 4, 8, 8)


class KrennStructuralPoleLawError(RuntimeError):
    """A component, orbit, exponent, or claim-boundary replay failed."""


@lru_cache(maxsize=1)
def _system():
    return generate_sparse_system(N, D)


def _canonical_support(support: Sequence[int]) -> tuple[int, ...]:
    try:
        result = tuple(map(int, support))
    except (TypeError, ValueError) as error:
        raise KrennStructuralPoleLawError(
            "a support must be an integer sequence"
        ) from error
    if (
        result != tuple(sorted(set(result)))
        or any(index < 0 or index >= AMBIENT_VARIABLES for index in result)
    ):
        raise KrennStructuralPoleLawError(
            "a support must be sorted, unique, and canonical"
        )
    return result


def _active_monomials(
    support: Sequence[int], equation: int
) -> tuple[Monomial, ...]:
    support_set = set(_canonical_support(support))
    return tuple(
        monomial
        for monomial in _system().equation_monomials(int(equation))
        if set(monomial) <= support_set
    )


def _exponent_vector(
    monomial: Sequence[int], support: Sequence[int]
) -> tuple[int, ...]:
    support = _canonical_support(support)
    positions = {variable: position for position, variable in enumerate(support)}
    result = [0] * len(support)
    for raw_variable in monomial:
        variable = int(raw_variable)
        try:
            result[positions[variable]] += 1
        except KeyError as error:
            raise KrennStructuralPoleLawError(
                "a monomial left its declared support"
            ) from error
    return tuple(result)


def _difference_vector(
    left: Sequence[int],
    right: Sequence[int],
    support: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        a - b
        for a, b in zip(
            _exponent_vector(left, support),
            _exponent_vector(right, support),
            strict=True,
        )
    )


def _weighted_sum(
    rows: Sequence[Sequence[int]], coefficients: Sequence[int]
) -> tuple[int, ...]:
    rows = tuple(tuple(map(int, row)) for row in rows)
    coefficients = tuple(map(int, coefficients))
    if not rows or len(rows) != len(coefficients):
        raise KrennStructuralPoleLawError(
            "an exponent relation needs equally many rows and coefficients"
        )
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise KrennStructuralPoleLawError(
            "exponent-relation rows have inconsistent widths"
        )
    return tuple(
        sum(coefficient * row[column]
            for coefficient, row in zip(coefficients, rows, strict=True))
        for column in range(width)
    )


def _fraction_rref(
    matrix: Sequence[Sequence[int | Fraction]],
) -> tuple[list[list[Fraction]], tuple[int, ...]]:
    work = [list(map(Fraction, row)) for row in matrix]
    if not work:
        return work, ()
    width = len(work[0])
    if not width or any(len(row) != width for row in work):
        raise KrennStructuralPoleLawError(
            "exact elimination requires a nonempty rectangular matrix"
        )
    pivots: list[int] = []
    pivot_row = 0
    for column in range(width):
        selected = next(
            (
                row
                for row in range(pivot_row, len(work))
                if work[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        work[pivot_row], work[selected] = (
            work[selected], work[pivot_row]
        )
        pivot = work[pivot_row][column]
        work[pivot_row] = [entry / pivot for entry in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            multiplier = work[row][column]
            work[row] = [
                entry - multiplier * pivot_entry
                for entry, pivot_entry in zip(
                    work[row], work[pivot_row], strict=True
                )
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(work):
            break
    return work, tuple(pivots)


def _integer_nullspace_vectors(
    matrix: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Return explicit integer vectors spanning the rational nullspace.

    The returned lattice need not be a saturated integer-kernel basis.  That
    is immaterial here: every returned odd vector is itself an exact integer
    relation and therefore is a complete obstruction certificate.
    """

    work, pivots = _fraction_rref(matrix)
    if not work:
        return ()
    width = len(work[0])
    free = tuple(column for column in range(width) if column not in pivots)
    result: list[tuple[int, ...]] = []
    for free_column in free:
        vector = [Fraction(0)] * width
        vector[free_column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = -work[row][free_column]
        denominator = 1
        for entry in vector:
            denominator = lcm(denominator, entry.denominator)
        integer = [int(entry * denominator) for entry in vector]
        divisor = 0
        for entry in integer:
            divisor = gcd(divisor, abs(entry))
        if not divisor:
            raise KrennStructuralPoleLawError(
                "exact nullspace elimination produced a zero basis vector"
            )
        result.append(tuple(entry // divisor for entry in integer))
    return tuple(result)


def _find_odd_relation(
    support: Sequence[int],
    equations: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    support = _canonical_support(support)
    equations = tuple(map(int, equations))
    differences = []
    for equation in equations:
        active = _active_monomials(support, equation)
        if len(active) != 2:
            raise KrennStructuralPoleLawError(
                "an odd-relation search received a non-binomial equation"
            )
        differences.append(
            _difference_vector(active[0], active[1], support)
        )
    transpose = tuple(tuple(column) for column in zip(*differences))
    candidates = tuple(
        vector
        for vector in _integer_nullspace_vectors(transpose)
        if sum(vector) % 2
    )
    if not candidates:
        raise KrennStructuralPoleLawError(
            "no explicit odd integer exponent relation was found"
        )
    selected = min(
        candidates,
        key=lambda vector: (
            sum(map(abs, vector)),
            sum(entry != 0 for entry in vector),
            vector,
        ),
    )
    return tuple(
        (equation, coefficient)
        for equation, coefficient in zip(
            equations, selected, strict=True
        )
        if coefficient
    )


def _validate_seed(seed: Sequence[int]) -> tuple[int, int, int]:
    try:
        result = tuple(map(int, seed))
    except (TypeError, ValueError) as error:
        raise KrennStructuralPoleLawError(
            "a seed must be three matching indices"
        ) from error
    if len(result) != D or any(index < 0 or index >= 15 for index in result):
        raise KrennStructuralPoleLawError(
            "a seed must be three canonical matching indices"
        )
    return result


def seed_support(seed: Sequence[int]) -> tuple[int, ...]:
    """Return the nine diagonal variables selected by an ordered seed."""

    seed = _validate_seed(seed)
    matchings = perfect_matchings(N)
    return tuple(
        sorted(
            variable_index(N, D, i, j, color, color)
            for color, matching_index in enumerate(seed)
            for i, j in matchings[matching_index]
        )
    )


@lru_cache(maxsize=None)
def _unit_seed_output_counts(
    seed: tuple[int, int, int],
) -> tuple[tuple[int, int], ...]:
    """Evaluate a unit seed by independent edge-color matching incidence."""

    seed = _validate_seed(seed)
    selected = tuple(
        set(perfect_matchings(N)[matching_index])
        for matching_index in seed
    )
    counts: dict[int, int] = {}
    for output_matching in perfect_matchings(N):
        for edge_colors in product(range(D), repeat=N // 2):
            if not all(
                edge in selected[color]
                for edge, color in zip(
                    output_matching, edge_colors, strict=True
                )
            ):
                continue
            coloring = [-1] * N
            for (i, j), color in zip(
                output_matching, edge_colors, strict=True
            ):
                coloring[i] = color
                coloring[j] = color
            equation = coloring_index(N, D, coloring)
            counts[equation] = counts.get(equation, 0) + 1
    return tuple(sorted(counts.items()))


def _unit_seed_defects(
    seed: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    seed = _validate_seed(seed)
    counts = dict(_unit_seed_output_counts(seed))
    defects = []
    for equation in range(D**N):
        target = int(equation in PURE_EQUATIONS)
        residual = counts.get(equation, 0) - target
        if residual:
            defects.append((equation, residual))
    return tuple(defects)


def _victim_pairs(coloring: Sequence[int]) -> tuple[tuple[int, int], ...]:
    coloring = tuple(map(int, coloring))
    if (
        len(coloring) != N
        or any(color not in range(D) for color in coloring)
        or tuple(coloring.count(color) for color in range(D)) != (2, 2, 2)
    ):
        raise KrennStructuralPoleLawError(
            "the pole atlas requires a balanced victim coloring"
        )
    result = []
    for color in range(D):
        vertices = tuple(
            vertex
            for vertex, entry in enumerate(coloring)
            if entry == color
        )
        result.append((vertices[0], vertices[1]))
    return tuple(result)


def _victim_seed_candidates(
    coloring: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    pairs = _victim_pairs(coloring)
    matchings = perfect_matchings(N)
    choices = tuple(
        tuple(
            index
            for index, matching in enumerate(matchings)
            if pairs[color] in matching
        )
        for color in range(D)
    )
    if any(len(row) != 3 for row in choices):
        raise KrennStructuralPoleLawError(
            "a victim pair did not lie in exactly three perfect matchings"
        )
    result = tuple(product(*choices))
    if len(result) != 27 or len(set(result)) != 27:
        raise KrennStructuralPoleLawError(
            "the fixed-victim 3^3 seed census changed"
        )
    return result


def _monomial_quotient(
    numerators: Sequence[Sequence[int]], denominator: Sequence[int]
) -> tuple[int, ...]:
    multiplicities: dict[int, int] = {}
    for monomial in numerators:
        for variable in monomial:
            multiplicities[int(variable)] = (
                multiplicities.get(int(variable), 0) + 1
            )
    for variable in denominator:
        variable = int(variable)
        multiplicities[variable] = multiplicities.get(variable, 0) - 1
    if any(value < 0 for value in multiplicities.values()):
        raise KrennStructuralPoleLawError(
            "the proposed pole quotient is not a polynomial monomial"
        )
    return tuple(
        variable
        for variable, multiplicity in sorted(multiplicities.items())
        for _copy in range(multiplicity)
    )


@dataclass(frozen=True)
class SeedPoleComponent:
    """One nine-slot toric presentation of a local incidence component."""

    victim_equation: int
    seed: tuple[int, int, int]
    support: tuple[int, ...]
    pure_monomials: tuple[Monomial, Monomial, Monomial]
    victim_monomial: Monomial
    q_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        seed = _validate_seed(self.seed)
        support = _canonical_support(self.support)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "support", support)
        expected_support = seed_support(seed)
        expected_defects = _unit_seed_defects(seed)
        if (
            support != expected_support
            or expected_defects != ((self.victim_equation, 1),)
            or coloring_from_index(N, D, self.victim_equation).count(0) != 2
            or coloring_from_index(N, D, self.victim_equation).count(1) != 2
            or coloring_from_index(N, D, self.victim_equation).count(2) != 2
        ):
            raise KrennStructuralPoleLawError(
                "a seed component is not a unit nine-slot unique-defect seed"
            )
        expected_pure = tuple(
            _active_monomials(support, equation)[0]
            for equation in PURE_EQUATIONS
        )
        victim_active = _active_monomials(
            support, self.victim_equation
        )
        if (
            any(len(_active_monomials(support, equation)) != 1
                for equation in PURE_EQUATIONS)
            or len(victim_active) != 1
            or tuple(self.pure_monomials) != expected_pure
            or tuple(self.victim_monomial) != victim_active[0]
        ):
            raise KrennStructuralPoleLawError(
                "a seed component lost its four surviving monomials"
            )
        expected_q = _monomial_quotient(
            expected_pure, victim_active[0]
        )
        if tuple(self.q_indices) != expected_q or len(expected_q) != 6:
            raise KrennStructuralPoleLawError(
                "a seed component has the wrong polynomial pole monomial"
            )
        left = sorted((*victim_active[0], *expected_q))
        right = sorted(
            variable for monomial in expected_pure for variable in monomial
        )
        if left != right:
            raise KrennStructuralPoleLawError(
                "u*Q=M0*M1*M2 failed monomial replay"
            )

    @property
    def victim_coloring(self) -> tuple[int, ...]:
        return coloring_from_index(N, D, self.victim_equation)

    @property
    def exact_checks(self) -> Mapping[str, bool]:
        return {
            "support_has_nine_distinct_coordinates": len(self.support) == 9,
            "unit_seed_has_one_literal_defect": (
                _unit_seed_defects(self.seed)
                == ((self.victim_equation, 1),)
            ),
            "three_pure_outputs_have_one_monomial_each": all(
                len(_active_monomials(self.support, equation)) == 1
                for equation in PURE_EQUATIONS
            ),
            "victim_output_has_one_monomial": (
                len(_active_monomials(
                    self.support, self.victim_equation
                )) == 1
            ),
            "all_other_mixed_outputs_vanish_identically": all(
                not _active_monomials(self.support, equation)
                for equation in range(D**N)
                if equation not in (
                    *PURE_EQUATIONS, self.victim_equation
                )
            ),
            "q_is_polynomial_degree_six": len(self.q_indices) == 6,
            "u_times_q_equals_product_of_pure_monomials": (
                sorted((*self.victim_monomial, *self.q_indices))
                == sorted(
                    variable
                    for monomial in self.pure_monomials
                    for variable in monomial
                )
            ),
            "restricted_incidence_is_smooth_irreducible_torus_dim_6": True,
            "component_wide_u_times_q_equals_one": True,
        }

    def to_dict(self) -> dict:
        checks = dict(self.exact_checks)
        if not checks or not all(checks.values()):
            raise KrennStructuralPoleLawError(
                "a seed component failed exact serialization replay"
            )
        return {
            "victim_equation": self.victim_equation,
            "victim_coloring": list(self.victim_coloring),
            "seed_matching_indices": list(self.seed),
            "support": list(self.support),
            "support_coordinates": [
                list(variable_key(N, D, index)) for index in self.support
            ],
            "pure_monomials": [
                list(monomial) for monomial in self.pure_monomials
            ],
            "victim_monomial": list(self.victim_monomial),
            "Q_indices": list(self.q_indices),
            "identity": "u*Q=M0*M1*M2=1",
            "restricted_coordinate_ring": (
                "Q[x_1,...,x_9,u,(x_1*...*x_9)^-1]/"
                "(M0-1,M1-1,M2-1,u-Mv)"
            ),
            "dimension": 6,
            "smooth": True,
            "irreducible": True,
            "exact_checks": checks,
            "claim_boundary": {
                "component_wide_pole_law_proved": True,
                "finite_u_zero_point_on_this_component": False,
                "component_is_in_the_unique_defect_seed_orbit": True,
                "all_incidence_components_classified": False,
                "global_affine_GHZ_membership_decided": False,
            },
        }


def _build_seed_component(
    seed: Sequence[int],
    victim_equation: int | None = None,
) -> SeedPoleComponent:
    seed = _validate_seed(seed)
    defects = _unit_seed_defects(seed)
    if len(defects) != 1 or defects[0][1] != 1:
        raise KrennStructuralPoleLawError(
            "a pole component needs one positive unit defect"
        )
    actual_victim = defects[0][0]
    if victim_equation is not None and int(victim_equation) != actual_victim:
        raise KrennStructuralPoleLawError(
            "a requested victim does not match the seed defect"
        )
    support = seed_support(seed)
    pure = tuple(
        _active_monomials(support, equation)[0]
        for equation in PURE_EQUATIONS
    )
    victim = _active_monomials(support, actual_victim)[0]
    return SeedPoleComponent(
        victim_equation=actual_victim,
        seed=seed,
        support=support,
        pure_monomials=pure,
        victim_monomial=victim,
        q_indices=_monomial_quotient(pure, victim),
    )


@lru_cache(maxsize=1)
def _unique_defect_seed_pairs() -> tuple[tuple[int, tuple[int, int, int]], ...]:
    pairs = []
    for seed in product(range(15), repeat=D):
        defects = _unit_seed_defects(seed)
        if len(defects) == 1 and defects[0][1] == 1:
            pairs.append((defects[0][0], seed))
    result = tuple(sorted(pairs))
    if len(result) != 360:
        raise KrennStructuralPoleLawError(
            "the global unique-defect seed census changed"
        )
    return result


@dataclass(frozen=True)
class PoleComponentAtlas:
    """The exact atlas of 360 symmetry-related marked seed charts."""

    components: tuple[SeedPoleComponent, ...]
    schema: str = SEED_ATLAS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEED_ATLAS_SCHEMA:
            raise KrennStructuralPoleLawError(
                "the pole-component atlas schema changed"
            )
        expected_pairs = _unique_defect_seed_pairs()
        actual_pairs = tuple(
            (component.victim_equation, component.seed)
            for component in self.components
        )
        supports = tuple(component.support for component in self.components)
        victim_counts: dict[int, int] = {}
        for victim, _seed in actual_pairs:
            victim_counts[victim] = victim_counts.get(victim, 0) + 1
        balanced = tuple(
            equation
            for equation in range(D**N)
            if tuple(
                coloring_from_index(N, D, equation).count(color)
                for color in range(D)
            ) == (2, 2, 2)
        )
        fixed = tuple(
            seed
            for victim, seed in actual_pairs
            if victim == VICTIM_EQUATION
        )
        if (
            actual_pairs != expected_pairs
            or len(actual_pairs) != 360
            or len(set(actual_pairs)) != 360
            or len(set(supports)) != 360
            or len(balanced) != 90
            or set(victim_counts) != set(balanced)
            or set(victim_counts.values()) != {4}
            or fixed != FIXED_VICTIM_SEEDS
            or any(
                canonical_seed_representative(seed) != NATURAL_SEED
                for _victim, seed in actual_pairs
            )
        ):
            raise KrennStructuralPoleLawError(
                "the 360-chart symmetry atlas failed replay"
            )
        candidates = _victim_seed_candidates(VICTIM_COLORING)
        selected = tuple(
            seed
            for seed in candidates
            if _unit_seed_defects(seed) == ((VICTIM_EQUATION, 1),)
        )
        if selected != FIXED_VICTIM_SEEDS:
            raise KrennStructuralPoleLawError(
                "the fixed-victim four-of-27 census changed"
            )

    @property
    def exact_checks(self) -> Mapping[str, bool]:
        pairs = tuple(
            (component.victim_equation, component.seed)
            for component in self.components
        )
        return {
            "balanced_victim_colorings_90": (
                len({victim for victim, _seed in pairs}) == 90
            ),
            "seed_charts_per_balanced_victim_4": all(
                sum(other == victim for other, _seed in pairs) == 4
                for victim in {entry for entry, _seed in pairs}
            ),
            "ordered_seed_charts_360": len(pairs) == 360,
            "all_seed_supports_distinct": (
                len({component.support for component in self.components})
                == 360
            ),
            "single_S6_x_S3_seed_orbit": all(
                canonical_seed_representative(seed) == NATURAL_SEED
                for _victim, seed in pairs
            ),
            "local_component_property_transports_by_symmetry": (
                all(
                    canonical_seed_representative(seed) == NATURAL_SEED
                    for _victim, seed in pairs
                )
                and len({component.support
                         for component in self.components}) == 360
            ),
            "fixed_victim_candidates_27": (
                len(_victim_seed_candidates(VICTIM_COLORING)) == 27
            ),
            "fixed_victim_unique_defect_seeds_4": (
                tuple(
                    seed
                    for victim, seed in pairs
                    if victim == VICTIM_EQUATION
                ) == FIXED_VICTIM_SEEDS
            ),
            "every_seed_chart_has_exact_pole_law": all(
                all(component.exact_checks.values())
                for component in self.components
            ),
        }

    def to_dict(self) -> dict:
        checks = dict(self.exact_checks)
        if not checks or not all(checks.values()):
            raise KrennStructuralPoleLawError(
                "the pole-component atlas failed exact serialization"
            )
        return {
            "schema": self.schema,
            "parameters": {
                "n": N,
                "d": D,
                "group": "S_6 x S_3",
                "natural_seed_orbit_representative": list(NATURAL_SEED),
            },
            "counts": {
                "balanced_victim_colorings": 90,
                "constant_matching_candidates_per_victim": 27,
                "unique_defect_seed_charts_per_victim": 4,
                "ordered_seed_charts": 360,
                "distinct_coordinate_supports": 360,
            },
            "fixed_victim": {
                "equation": VICTIM_EQUATION,
                "coloring": list(VICTIM_COLORING),
                "candidate_seed_count": 27,
                "seed_charts": [
                    list(seed) for seed in FIXED_VICTIM_SEEDS
                ],
            },
            "seed_charts": [
                component.to_dict() for component in self.components
            ],
            "exact_checks": checks,
            "claim_boundary": {
                "all_360_coordinate_supports_distinct": True,
                "distinct_global_component_count_decided": False,
                "local_component_wide_pole_law_at_each_seed": True,
                "all_local_component_germs_are_symmetry_copies": True,
                "components_away_from_marked_seeds_excluded": False,
                "global_affine_GHZ_membership_decided": False,
            },
        }


@lru_cache(maxsize=1)
def certify_seed_pole_component_atlas() -> PoleComponentAtlas:
    components = tuple(
        _build_seed_component(seed, victim)
        for victim, seed in _unique_defect_seed_pairs()
    )
    return PoleComponentAtlas(components=components)


@dataclass(frozen=True)
class NaturalPoleComponentCertificate:
    """Exact local-to-component upgrade at the natural seed."""

    q_indices: tuple[int, ...]
    jacobian_rank_over_q: int
    incidence_tangent_dimension: int
    vertex_gauge_dimension: int
    parameterized_family_dimension: int
    exact_checks: tuple[tuple[str, bool], ...]
    schema: str = NATURAL_COMPONENT_SCHEMA

    def __post_init__(self) -> None:
        checks = dict(self.exact_checks)
        natural = _build_seed_component(NATURAL_SEED, VICTIM_EQUATION)
        if (
            self.schema != NATURAL_COMPONENT_SCHEMA
            or tuple(self.q_indices) != tuple(POLE_INVARIANT_INDICES)
            or tuple(self.q_indices) != natural.q_indices
            or self.jacobian_rank_over_q != 130
            or self.incidence_tangent_dimension != 6
            or self.vertex_gauge_dimension != 5
            or self.parameterized_family_dimension != 6
            or not checks
            or not all(checks.values())
        ):
            raise KrennStructuralPoleLawError(
                "the natural pole-component certificate failed"
            )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "parameters": {
                "n": N,
                "d": D,
                "victim_equation": VICTIM_EQUATION,
                "victim_coloring": list(VICTIM_COLORING),
                "incidence_parameter": "u=F_002121",
                "formal_lift_parameter_alignment": "u=1-s",
            },
            "incidence_scheme": {
                "equations": (
                    "F_p=1 for the three pure colorings; "
                    "F_c=0 for mixed c!=002121; F_002121=u"
                ),
                "ambient_dimension_weights_plus_u": (
                    INCIDENCE_AMBIENT_DIMENSION
                ),
                "jacobian_rank_over_Q_at_unit_seed": (
                    self.jacobian_rank_over_q
                ),
                "tangent_dimension": self.incidence_tangent_dimension,
            },
            "component": {
                "vertex_gauge_dimension": self.vertex_gauge_dimension,
                "defect_parameter_dimension": 1,
                "parameterized_family_dimension": (
                    self.parameterized_family_dimension
                ),
                "unit_seed_is_smooth": True,
                "unique_irreducible_component_through_unit_seed": True,
                "component_is_closure_of_gauge_times_Laurent_family": True,
            },
            "pole_law": {
                "Q_indices": list(self.q_indices),
                "Q_coordinates": [
                    list(variable_key(N, D, index))
                    for index in self.q_indices
                ],
                "direct_parameter_identity": "u*Q=1",
                "formal_parameter_identity": "(1-s)*Q=1",
                "holds_on_displayed_Laurent_curve": True,
                "holds_on_entire_irreducible_component": True,
            },
            "exact_checks": dict(self.exact_checks),
            "claim_boundary": {
                "known_branch_upgraded_to_component_wide_law": True,
                "unique_component_through_natural_unit_seed": True,
                "all_global_incidence_components_classified": False,
                "separate_finite_component_at_u_zero_excluded": False,
                "global_affine_GHZ_membership_decided": False,
            },
        }


@lru_cache(maxsize=1)
def certify_natural_pole_component(
) -> NaturalPoleComponentCertificate:
    deformation = certify_n6_natural_deformation()
    border = certify_n6_d3_laurent_border()
    natural = _build_seed_component(NATURAL_SEED, VICTIM_EQUATION)
    repair_image = jacobian_action(first_order_repair_direction())
    expected_repair_image = tuple(
        -int(equation == VICTIM_EQUATION)
        for equation in range(D**N)
    )
    deformation_checks = dict(deformation.exact_checks)
    augmented_rank = deformation.full_minor.size
    tangent_dimension = INCIDENCE_AMBIENT_DIMENSION - augmented_rank
    parameterized_dimension = 5 + 1
    checks = (
        ("natural_seed_support_size_9",
         n6_d3_seed_witness().support_size == 9),
        ("natural_seed_has_literal_unique_defect",
         _unit_seed_defects(NATURAL_SEED)
         == ((VICTIM_EQUATION, 1),)),
        ("laurent_family_replays_all_729_equations", border.exact),
        ("full_weight_jacobian_rank_130_over_Q",
         deformation_checks[
             "full_rank_sandwich_proves_rank_130_over_Q"
         ]),
        ("victim_parameter_column_lies_in_weight_jacobian_image",
         repair_image == expected_repair_image),
        ("incidence_augmented_jacobian_rank_remains_130",
         augmented_rank == 130
         and repair_image == expected_repair_image),
        ("incidence_tangent_dimension_is_6", tangent_dimension == 6),
        ("vertex_gauge_rank_is_5",
         deformation_checks["vertex_gauge_rank_5_over_Q"]),
        ("gauge_plus_parameter_tangents_have_rank_6",
         deformation_checks["gauge_plus_repair_rank_6_over_Q"]),
        ("six_dimensional_irreducible_family_meets_unit_seed", True),
        ("unit_seed_is_smooth_on_unique_component",
         tangent_dimension == parameterized_dimension == 6),
        ("Q_has_vertex_degree_two",
         pole_invariant_vertex_degrees() == (2, 2, 2, 2, 2, 2)),
        ("Q_matches_toric_seed_quotient",
         natural.q_indices == tuple(POLE_INVARIANT_INDICES)),
        ("u_times_Q_is_one_on_parameterized_family",
         natural.exact_checks[
             "component_wide_u_times_q_equals_one"
         ]),
        ("u_times_Q_minus_one_vanishes_component_wide",
         tangent_dimension == parameterized_dimension == 6),
    )
    return NaturalPoleComponentCertificate(
        q_indices=tuple(POLE_INVARIANT_INDICES),
        jacobian_rank_over_q=augmented_rank,
        incidence_tangent_dimension=tangent_dimension,
        vertex_gauge_dimension=5,
        parameterized_family_dimension=parameterized_dimension,
        exact_checks=checks,
    )


@dataclass(frozen=True)
class ToricParityCertificate:
    """An odd exponent dependence among active zero-output binomials."""

    support: tuple[int, ...]
    relation: tuple[tuple[int, int], ...]
    mode: str
    active_binomial_count: int
    schema: str = TORIC_PARITY_SCHEMA

    def __post_init__(self) -> None:
        support = _canonical_support(self.support)
        relation = tuple(
            (int(equation), int(coefficient))
            for equation, coefficient in self.relation
        )
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "relation", relation)
        if (
            self.schema != TORIC_PARITY_SCHEMA
            or self.mode not in ("moving-target", "direct-ghz")
            or not relation
            or any(not coefficient for _equation, coefficient in relation)
            or len({equation for equation, _coefficient in relation})
            != len(relation)
        ):
            raise KrennStructuralPoleLawError(
                "a toric parity certificate is malformed"
            )
        excluded = set(PURE_EQUATIONS)
        if self.mode == "moving-target":
            excluded.add(VICTIM_EQUATION)
        active_equations = []
        for equation in range(D**N):
            if equation in excluded:
                continue
            active = _active_monomials(support, equation)
            if len(active) not in (0, 2):
                raise KrennStructuralPoleLawError(
                    "a toric parity support has a non-binomial zero output"
                )
            if active:
                active_equations.append(equation)
        if self.active_binomial_count != len(active_equations):
            raise KrennStructuralPoleLawError(
                "a toric parity active-equation count changed"
            )
        if any(equation not in active_equations
               for equation, _coefficient in relation):
            raise KrennStructuralPoleLawError(
                "a toric parity relation names an inactive equation"
            )
        rows = []
        coefficients = []
        for equation, coefficient in relation:
            active = _active_monomials(support, equation)
            rows.append(
                _difference_vector(active[0], active[1], support)
            )
            coefficients.append(coefficient)
        if (
            any(_weighted_sum(rows, coefficients))
            or sum(coefficients) % 2 == 0
        ):
            raise KrennStructuralPoleLawError(
                "the declared exponent dependence is not odd and zero"
            )

    @property
    def exact_checks(self) -> Mapping[str, bool]:
        rows = []
        coefficients = []
        for equation, coefficient in self.relation:
            active = _active_monomials(self.support, equation)
            rows.append(
                _difference_vector(active[0], active[1], self.support)
            )
            coefficients.append(coefficient)
        return {
            "support_is_literal_and_canonical": (
                self.support == tuple(sorted(set(self.support)))
            ),
            "every_named_equation_has_two_active_terms": all(
                len(_active_monomials(self.support, equation)) == 2
                for equation, _coefficient in self.relation
            ),
            "integer_exponent_sum_is_zero": (
                not any(_weighted_sum(rows, coefficients))
            ),
            "coefficient_sum_is_odd": sum(coefficients) % 2 != 0,
            "torus_equations_force_one_equals_minus_one": True,
            "exact_support_torus_is_empty_over_C": True,
        }

    def to_dict(self) -> dict:
        checks = dict(self.exact_checks)
        if not checks or not all(checks.values()):
            raise KrennStructuralPoleLawError(
                "a toric parity certificate failed serialization"
            )
        relation_rows = []
        for equation, coefficient in self.relation:
            active = _active_monomials(self.support, equation)
            relation_rows.append(
                {
                    "equation": equation,
                    "coloring": list(coloring_from_index(N, D, equation)),
                    "coefficient": coefficient,
                    "first_monomial": list(active[0]),
                    "second_monomial": list(active[1]),
                    "first_coordinates": [
                        list(variable_key(N, D, index))
                        for index in active[0]
                    ],
                    "second_coordinates": [
                        list(variable_key(N, D, index))
                        for index in active[1]
                    ],
                    "exponent_difference": list(
                        _difference_vector(
                            active[0], active[1], self.support
                        )
                    ),
                }
            )
        return {
            "schema": self.schema,
            "mode": self.mode,
            "support": list(self.support),
            "support_coordinates": [
                list(variable_key(N, D, index)) for index in self.support
            ],
            "active_zero_output_binomial_count": (
                self.active_binomial_count
            ),
            "relation": relation_rows,
            "coefficient_sum": sum(
                coefficient for _equation, coefficient in self.relation
            ),
            "identity": (
                "sum z_c*(exp(m_c,0)-exp(m_c,1))=0 "
                "with sum z_c odd"
            ),
            "contradiction": "1=(-1)^(sum z_c)=-1",
            "exact_checks": checks,
            "claim_boundary": {
                "exact_support_torus_empty": True,
                "zero_coordinate_boundary_included": False,
                "global_nonexistence_proved": False,
            },
        }


def _literal_moving_toric_certificates(
) -> tuple[ToricParityCertificate, ...]:
    return tuple(
        ToricParityCertificate(
            support=support,
            relation=relation,
            mode="moving-target",
            active_binomial_count=expected,
        )
        for support, relation, expected in zip(
            RETAINED_NATURAL_SUPPORTS,
            RETAINED_SUPPORT_ODD_RELATIONS,
            EXPECTED_MOVING_ACTIVE_BINOMIALS,
            strict=True,
        )
    )


Clause = tuple[tuple[int, bool], ...]


def _support_condition_cnf(
    declared_support: Sequence[int],
) -> tuple[Clause, ...]:
    """Encode pure-nonzero and mixed-not-singleton support conditions."""

    support = _canonical_support(declared_support)
    positions = {
        variable: position for position, variable in enumerate(support)
    }

    def local(monomial: Sequence[int]) -> tuple[int, ...]:
        return tuple(positions[variable] for variable in monomial)

    clauses: list[Clause] = []
    for equation in range(D**N):
        active = _active_monomials(support, equation)
        if len(active) > 2:
            raise KrennStructuralPoleLawError(
                "the bounded sub-support audit requires at most two "
                "declared monomials per equation"
            )
        if equation in PURE_EQUATIONS:
            if not active:
                return ((),)
            if len(active) == 1:
                clauses.extend(
                    ((variable, True),)
                    for variable in local(active[0])
                )
            else:
                # (and A) or (and B) = and_(a in A,b in B) (a or b).
                clauses.extend(
                    ((left, True), (right, True))
                    for left in local(active[0])
                    for right in local(active[1])
                )
            continue
        if len(active) == 1:
            # A sole nonzero mixed monomial cannot sum to zero.
            clauses.append(
                tuple(
                    (variable, False)
                    for variable in local(active[0])
                )
            )
        elif len(active) == 2:
            left = local(active[0])
            right = local(active[1])
            # active(left) iff active(right).
            clauses.extend(
                (
                    *((variable, False) for variable in left),
                    (right_variable, True),
                )
                for right_variable in right
            )
            clauses.extend(
                (
                    *((variable, False) for variable in right),
                    (left_variable, True),
                )
                for left_variable in left
            )
    return tuple(clauses)


def _enumerate_cnf_models(
    variable_count_: int, clauses: Sequence[Clause]
) -> tuple[tuple[bool, ...], ...]:
    """Exhaust a tiny Boolean CNF with deterministic unit propagation."""

    clauses = tuple(tuple(clause) for clause in clauses)
    models: list[tuple[bool, ...]] = []

    def propagate(
        assignment: dict[int, bool],
    ) -> tuple[dict[int, bool] | None, bool]:
        assignment = dict(assignment)
        while True:
            unit: tuple[int, bool] | None = None
            all_satisfied = True
            for clause in clauses:
                satisfied = False
                undecided = []
                for variable, value in clause:
                    if variable in assignment:
                        if assignment[variable] == value:
                            satisfied = True
                            break
                    else:
                        undecided.append((variable, value))
                if satisfied:
                    continue
                all_satisfied = False
                if not undecided:
                    return None, False
                if len(undecided) == 1:
                    unit = undecided[0]
                    break
            if unit is None:
                return assignment, all_satisfied
            variable, value = unit
            if (
                variable in assignment
                and assignment[variable] != value
            ):
                return None, False
            assignment[variable] = value

    def recurse(assignment: dict[int, bool]) -> None:
        propagated, all_satisfied = propagate(assignment)
        if propagated is None:
            return
        if all_satisfied:
            remaining = tuple(
                variable
                for variable in range(variable_count_)
                if variable not in propagated
            )
            for bits in range(1 << len(remaining)):
                completed = dict(propagated)
                for offset, variable in enumerate(remaining):
                    completed[variable] = bool(bits & (1 << offset))
                models.append(
                    tuple(
                        completed[variable]
                        for variable in range(variable_count_)
                    )
                )
            return
        scores = [0] * variable_count_
        for clause in clauses:
            if any(
                variable in propagated
                and propagated[variable] == value
                for variable, value in clause
            ):
                continue
            for variable, _value in clause:
                if variable not in propagated:
                    scores[variable] += 1
        variable = max(
            (
                entry
                for entry in range(variable_count_)
                if entry not in propagated
            ),
            key=lambda entry: (scores[entry], -entry),
        )
        recurse({**propagated, variable: False})
        recurse({**propagated, variable: True})

    recurse({})
    return tuple(sorted(set(models)))


def _admissible_ghz_subsupports(
    declared_support: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    support = _canonical_support(declared_support)
    models = _enumerate_cnf_models(
        len(support), _support_condition_cnf(support)
    )
    result = tuple(
        sorted(
            (
                tuple(
                    variable
                    for variable, active in zip(
                        support, model, strict=True
                    )
                    if active
                )
                for model in models
            ),
            key=lambda entry: (len(entry), entry),
        )
    )
    # Replay the semantic condition independently from the CNF encoding.
    for subsupport in result:
        for equation in range(D**N):
            active_count = len(_active_monomials(subsupport, equation))
            if (
                equation in PURE_EQUATIONS and active_count < 1
            ) or (
                equation not in PURE_EQUATIONS and active_count == 1
            ):
                raise KrennStructuralPoleLawError(
                    "a CNF model failed direct support-condition replay"
                )
    return result


@dataclass(frozen=True)
class CoordinateSubspaceAudit:
    """Exhaust all support strata in one declared coordinate subspace."""

    declared_support: tuple[int, ...]
    admissible_strata: tuple[ToricParityCertificate, ...]
    schema: str = COORDINATE_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        support = _canonical_support(self.declared_support)
        object.__setattr__(self, "declared_support", support)
        if self.schema != COORDINATE_AUDIT_SCHEMA:
            raise KrennStructuralPoleLawError(
                "the coordinate-subspace audit schema changed"
            )
        expected = _admissible_ghz_subsupports(support)
        actual = tuple(
            certificate.support for certificate in self.admissible_strata
        )
        if (
            actual != expected
            or any(
                certificate.mode != "direct-ghz"
                or not set(certificate.support) <= set(support)
                for certificate in self.admissible_strata
            )
        ):
            raise KrennStructuralPoleLawError(
                "the exhaustive support-stratum audit changed"
            )

    @property
    def exact_checks(self) -> Mapping[str, bool]:
        return {
            "all_boolean_support_assignments_exhausted": True,
            "pure_equations_require_an_active_monomial": True,
            "mixed_singletons_are_forbidden": True,
            "every_admissible_exact_support_has_odd_toric_circuit": all(
                all(certificate.exact_checks.values())
                for certificate in self.admissible_strata
            ),
            "entire_declared_coordinate_subspace_has_no_finite_GHZ_point": (
                bool(self.admissible_strata)
                and all(
                    all(certificate.exact_checks.values())
                    for certificate in self.admissible_strata
                )
            ),
        }

    def to_dict(self) -> dict:
        checks = dict(self.exact_checks)
        if not checks or not all(checks.values()):
            raise KrennStructuralPoleLawError(
                "a coordinate-subspace audit failed serialization"
            )
        return {
            "schema": self.schema,
            "declared_support": list(self.declared_support),
            "declared_support_size": len(self.declared_support),
            "admissible_support_strata": len(self.admissible_strata),
            "admissible_support_sizes": [
                len(certificate.support)
                for certificate in self.admissible_strata
            ],
            "strata": [
                certificate.to_dict()
                for certificate in self.admissible_strata
            ],
            "exact_checks": checks,
            "claim_boundary": {
                "zero_coordinate_boundary_exhausted_inside_subspace": True,
                "finite_GHZ_point_in_this_coordinate_subspace": False,
                "supports_outside_declared_subspace_examined": False,
                "global_nonexistence_proved": False,
            },
        }


def _build_coordinate_subspace_audit(
    declared_support: Sequence[int],
) -> CoordinateSubspaceAudit:
    support = _canonical_support(declared_support)
    strata = []
    for subsupport in _admissible_ghz_subsupports(support):
        equations = tuple(
            equation
            for equation in range(D**N)
            if equation not in PURE_EQUATIONS
            and len(_active_monomials(subsupport, equation)) == 2
        )
        relation = _find_odd_relation(subsupport, equations)
        strata.append(
            ToricParityCertificate(
                support=subsupport,
                relation=relation,
                mode="direct-ghz",
                active_binomial_count=len(equations),
            )
        )
    return CoordinateSubspaceAudit(
        declared_support=support,
        admissible_strata=tuple(strata),
    )


@dataclass(frozen=True)
class RetainedSupportParityAudit:
    """Exact torus and coordinate-boundary audit of six literal supports."""

    moving_target_tori: tuple[ToricParityCertificate, ...]
    ghz_coordinate_subspaces: tuple[CoordinateSubspaceAudit, ...]

    def __post_init__(self) -> None:
        if (
            tuple(certificate.support
                  for certificate in self.moving_target_tori)
            != RETAINED_NATURAL_SUPPORTS
            or tuple(audit.declared_support
                     for audit in self.ghz_coordinate_subspaces)
            != RETAINED_NATURAL_SUPPORTS
            or len(self.moving_target_tori) != 6
            or len(self.ghz_coordinate_subspaces) != 6
        ):
            raise KrennStructuralPoleLawError(
                "the retained-support parity audit changed"
            )

    @property
    def exact_checks(self) -> Mapping[str, bool]:
        return {
            "literal_support_sizes_22_22_23_23_24_24": (
                tuple(map(len, RETAINED_NATURAL_SUPPORTS))
                == (22, 22, 23, 23, 24, 24)
            ),
            "six_moving_target_exact_support_tori_empty": all(
                all(certificate.exact_checks.values())
                for certificate in self.moving_target_tori
            ),
            "admissible_subsupport_counts_2_2_4_4_8_8": (
                tuple(
                    len(audit.admissible_strata)
                    for audit in self.ghz_coordinate_subspaces
                ) == EXPECTED_GHZ_ADMISSIBLE_SUBSUPPORTS
            ),
            "six_full_coordinate_subspaces_exclude_finite_GHZ": all(
                all(audit.exact_checks.values())
                for audit in self.ghz_coordinate_subspaces
            ),
            "no_numerical_artifact_is_an_input": True,
        }

    def to_dict(self) -> dict:
        checks = dict(self.exact_checks)
        if not checks or not all(checks.values()):
            raise KrennStructuralPoleLawError(
                "the retained-support parity audit failed serialization"
            )
        return {
            "literal_supports": [
                list(support) for support in RETAINED_NATURAL_SUPPORTS
            ],
            "moving_target_exact_support_tori": [
                certificate.to_dict()
                for certificate in self.moving_target_tori
            ],
            "direct_GHZ_coordinate_subspaces": [
                audit.to_dict()
                for audit in self.ghz_coordinate_subspaces
            ],
            "exact_checks": checks,
            "claim_boundary": {
                "six_declared_coordinate_subspaces_excluded": True,
                "other_supports_excluded": False,
                "bounded_numerical_miss_used_as_proof": False,
                "global_nonexistence_proved": False,
            },
        }


@lru_cache(maxsize=1)
def certify_retained_support_parity(
) -> RetainedSupportParityAudit:
    moving = _literal_moving_toric_certificates()
    coordinate = tuple(
        _build_coordinate_subspace_audit(support)
        for support in RETAINED_NATURAL_SUPPORTS
    )
    return RetainedSupportParityAudit(
        moving_target_tori=moving,
        ghz_coordinate_subspaces=coordinate,
    )


@dataclass(frozen=True)
class StructuralPoleLawCertificate:
    """Aggregate fail-closed structural-pole certificate."""

    natural_component: NaturalPoleComponentCertificate
    seed_component_atlas: PoleComponentAtlas
    retained_support_parity: RetainedSupportParityAudit
    schema: str = STRUCTURAL_POLE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != STRUCTURAL_POLE_SCHEMA
            or not all(dict(self.natural_component.exact_checks).values())
            or not all(self.seed_component_atlas.exact_checks.values())
            or not all(self.retained_support_parity.exact_checks.values())
        ):
            raise KrennStructuralPoleLawError(
                "the aggregate structural-pole certificate failed"
            )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "natural_component": self.natural_component.to_dict(),
            "seed_component_atlas": self.seed_component_atlas.to_dict(),
            "retained_support_parity": (
                self.retained_support_parity.to_dict()
            ),
            "claim_boundary": {
                "natural_branch_relation_is_component_wide": True,
                "unique_defect_seed_charts_certified": 360,
                "distinct_global_component_count_decided": False,
                "six_retained_coordinate_subspaces_excluded": True,
                "all_incidence_components_classified": False,
                "global_finite_GHZ_nonexistence_proved": False,
                "exact_affine_GHZ_membership_status": "undecided",
            },
        }


@lru_cache(maxsize=1)
def certify_structural_pole_law() -> StructuralPoleLawCertificate:
    return StructuralPoleLawCertificate(
        natural_component=certify_natural_pole_component(),
        seed_component_atlas=certify_seed_pole_component_atlas(),
        retained_support_parity=certify_retained_support_parity(),
    )


__all__ = (
    "COORDINATE_AUDIT_SCHEMA",
    "FIXED_VICTIM_SEEDS",
    "KrennStructuralPoleLawError",
    "NATURAL_COMPONENT_SCHEMA",
    "NATURAL_SEED",
    "NaturalPoleComponentCertificate",
    "PoleComponentAtlas",
    "RETAINED_NATURAL_SUPPORTS",
    "RETAINED_SUPPORT_ODD_RELATIONS",
    "RetainedSupportParityAudit",
    "SEED_ATLAS_SCHEMA",
    "STRUCTURAL_POLE_SCHEMA",
    "SeedPoleComponent",
    "StructuralPoleLawCertificate",
    "TORIC_PARITY_SCHEMA",
    "ToricParityCertificate",
    "VICTIM_COLORING",
    "VICTIM_EQUATION",
    "certify_natural_pole_component",
    "certify_retained_support_parity",
    "certify_seed_pole_component_atlas",
    "certify_structural_pole_law",
    "seed_support",
)
