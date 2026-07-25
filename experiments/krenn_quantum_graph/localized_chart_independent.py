"""Independent replay of the eight localized ``n=6,d=3`` chart ideals.

This module deliberately imports no other Krenn code.  It finds perfect
matchings by filtering edge subsets, reconstructs variable indices and
colorings locally, and reproduces the normalized chart polynomials.  The
primary constructor uses a different least-unused-vertex recursion.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from itertools import combinations, product
from typing import Sequence


N = 6
D = 3
AMBIENT_WEIGHT_COUNT = 135
CHART_WEIGHT_COUNT = 126
CHART_VARIABLE_COUNT = 129
PURE_EQUATIONS = (0, 364, 728)
MIXED_EQUATIONS = tuple(
    equation for equation in range(D**N)
    if equation not in PURE_EQUATIONS
)
LOCALIZED_CHART_SCHEMA = "krenn-n6-d3-localized-chart-ideal-v1"

REPRESENTATIVES_AND_SIZES = (
    ((0, 0, 0), 15),
    ((0, 0, 1), 270),
    ((0, 0, 4), 360),
    ((0, 1, 2), 90),
    ((0, 1, 3), 1080),
    ((0, 1, 5), 1080),
    ((0, 4, 8), 360),
    ((0, 4, 13), 120),
)


class IndependentLocalizedChartError(RuntimeError):
    """Independent chart reconstruction or evaluation failed."""


@lru_cache(maxsize=1)
def independent_perfect_matchings() -> tuple[
    tuple[tuple[int, int], ...], ...
]:
    """Find K6 perfect matchings by filtering all three-edge subsets."""

    edges = tuple(combinations(range(N), 2))
    result = tuple(
        candidate
        for candidate in combinations(edges, N // 2)
        if len({
            vertex for edge in candidate for vertex in edge
        }) == N
    )
    if len(result) != 15:
        raise IndependentLocalizedChartError(
            "independent K6 matching census changed"
        )
    return result


def _variable_index(i: int, j: int, a: int, b: int) -> int:
    i, j, a, b = map(int, (i, j, a, b))
    if not (0 <= i < j < N and 0 <= a < D and 0 <= b < D):
        raise IndependentLocalizedChartError(
            "independent variable coordinates are invalid"
        )
    edges = tuple(combinations(range(N), 2))
    return (edges.index((i, j)) * D + a) * D + b


def _coloring_from_index(index: int) -> tuple[int, ...]:
    index = int(index)
    if not 0 <= index < D**N:
        raise IndependentLocalizedChartError(
            "independent coloring index is invalid"
        )
    digits = [0] * N
    for position in range(N - 1, -1, -1):
        index, digits[position] = divmod(index, D)
    return tuple(digits)


def _seed_support(seed: Sequence[int]) -> tuple[int, ...]:
    try:
        seed = tuple(map(int, seed))
    except (TypeError, ValueError) as error:
        raise IndependentLocalizedChartError(
            "independent seed is not integral"
        ) from error
    if len(seed) != D or any(index < 0 or index >= 15 for index in seed):
        raise IndependentLocalizedChartError(
            "independent seed has invalid matching indices"
        )
    matchings = independent_perfect_matchings()
    support = tuple(sorted(
        _variable_index(i, j, color, color)
        for color, matching_index in enumerate(seed)
        for i, j in matchings[matching_index]
    ))
    if len(support) != 9 or len(set(support)) != 9:
        raise IndependentLocalizedChartError(
            "independent seed support is not nine distinct variables"
        )
    return support


def _add(
    polynomial: dict[tuple[int, ...], int],
    monomial: Sequence[int],
    coefficient: int,
) -> None:
    monomial = tuple(sorted(map(int, monomial)))
    if any(index < 0 or index >= CHART_VARIABLE_COUNT
           for index in monomial):
        raise IndependentLocalizedChartError(
            "independent reduced monomial has an invalid variable"
        )
    total = polynomial.get(monomial, 0) + int(coefficient)
    if total:
        polynomial[monomial] = total
    else:
        polynomial.pop(monomial, None)


def _terms(
    polynomial: dict[tuple[int, ...], int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (coefficient, monomial)
        for monomial, coefficient in sorted(
            polynomial.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
        if coefficient
    )


@lru_cache(maxsize=8)
def independent_normalized_chart(
    orbit_index: int,
) -> tuple[
    tuple[int, int, int],
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[str, int], ...],
    tuple[tuple[tuple[int, tuple[int, ...]], ...], ...],
]:
    """Reconstruct one chart without importing the primary enumerator."""

    orbit_index = int(orbit_index)
    if not 0 <= orbit_index < len(REPRESENTATIVES_AND_SIZES):
        raise IndependentLocalizedChartError(
            "independent chart orbit index is invalid"
        )
    seed, orbit_size = REPRESENTATIVES_AND_SIZES[orbit_index]
    fixed = _seed_support(seed)
    fixed_set = set(fixed)
    remaining = tuple(
        index for index in range(AMBIENT_WEIGHT_COUNT)
        if index not in fixed_set
    )
    remap = {
        ambient: local for local, ambient in enumerate(remaining)
    }
    matchings = independent_perfect_matchings()
    outputs = []
    for equation in range(D**N):
        coloring = _coloring_from_index(equation)
        polynomial: dict[tuple[int, ...], int] = {}
        for matching in matchings:
            ambient_monomial = tuple(
                _variable_index(
                    i, j, coloring[i], coloring[j]
                )
                for i, j in matching
            )
            _add(
                polynomial,
                (
                    remap[variable]
                    for variable in ambient_monomial
                    if variable not in fixed_set
                ),
                1,
            )
        output = _terms(polynomial)
        if (
            len(output) != 15
            or any(coefficient != 1 for coefficient, _ in output)
        ):
            raise IndependentLocalizedChartError(
                "independent seed substitution collided"
            )
        outputs.append(output)

    labels = list(("mixed", equation) for equation in MIXED_EQUATIONS)
    generators = [outputs[equation] for equation in MIXED_EQUATIONS]
    for color, equation in enumerate(PURE_EQUATIONS):
        polynomial = {}
        for coefficient, monomial in outputs[equation]:
            _add(polynomial, (*monomial, CHART_WEIGHT_COUNT + color),
                 coefficient)
        _add(polynomial, (), -1)
        localizer = _terms(polynomial)
        if len(localizer) != 16:
            raise IndependentLocalizedChartError(
                "independent localizer changed sparse size"
            )
        labels.append(("pure-localizer", color))
        generators.append(localizer)
    if (
        len(generators) != 729
        or sum(len(polynomial) for polynomial in generators) != 10_938
    ):
        raise IndependentLocalizedChartError(
            "independent chart sparse census changed"
        )
    return (
        seed,
        orbit_size,
        fixed,
        remaining,
        tuple(labels),
        tuple(generators),
    )


def independent_chart_fingerprint(orbit_index: int) -> str:
    """Hash the independent reconstruction in the primary wire format."""

    (
        seed,
        orbit_size,
        fixed,
        remaining,
        labels,
        generators,
    ) = independent_normalized_chart(orbit_index)
    digest = hashlib.sha256()
    digest.update(
        (
            f"{LOCALIZED_CHART_SCHEMA}|{int(orbit_index)}|{seed}|"
            f"{orbit_size}|{fixed}|{remaining}\n"
        ).encode("ascii")
    )
    for label, polynomial in zip(labels, generators, strict=True):
        digest.update(f"{label[0]}:{label[1]}|".encode("ascii"))
        for coefficient, monomial in polynomial:
            digest.update(
                f"{coefficient}:{','.join(map(str, monomial))};".encode(
                    "ascii"
                )
            )
        digest.update(b"\n")
    return digest.hexdigest()


def independent_evaluate_chart(
    orbit_index: int,
    values: Sequence,
) -> tuple:
    """Evaluate all 729 independent chart generators exactly."""

    if len(values) != CHART_VARIABLE_COUNT:
        raise IndependentLocalizedChartError(
            "independent chart evaluation needs 129 values"
        )
    generators = independent_normalized_chart(orbit_index)[-1]
    results = []
    for polynomial in generators:
        total = 0
        for coefficient, monomial in polynomial:
            term = coefficient
            for variable in monomial:
                term *= values[variable]
            total += term
        results.append(total)
    return tuple(results)
