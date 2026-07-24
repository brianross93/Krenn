"""Exact computational-basis targets for Krenn polynomial systems.

A coloring ``(c_0, ..., c_{n-1}) in range(d)^n`` labels the corresponding
computational-basis coefficient.  Colorings are ordered lexicographically,
with the last coordinate changing fastest.  Coefficients are exact rationals,
so every target has a canonical scalar extension to ``C``.

The named state constructors below are deliberately unnormalized: normalized
GHZ, Dicke, and graph states generally require irrational square roots, while
the polynomial equations use their exact rational coefficient tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb
from numbers import Integral
from typing import Iterable, Mapping, Sequence, TypeAlias


Coloring: TypeAlias = tuple[int, ...]
ExactRational: TypeAlias = int | Fraction
TARGET_SCHEMA = "krenn-computational-basis-target-v1"
BASIS_CONVENTION = (
    "colorings (c_0,...,c_{n-1}) in range(d)^n; lexicographic order "
    "with c_{n-1} changing fastest"
)


class KrennTargetError(ValueError):
    """An exact target coefficient or basis label is malformed."""


def _exact_integer(value, label: str) -> int:
    if not isinstance(value, Integral):
        raise KrennTargetError(f"{label} must be an exact integer")
    return int(value)


def _validate_dimensions(n: int, d: int) -> tuple[int, int]:
    n = _exact_integer(n, "target factor count")
    d = _exact_integer(d, "target local dimension")
    if n < 1:
        raise KrennTargetError("a target needs at least one tensor factor")
    if d < 1:
        raise KrennTargetError("a target needs positive local dimension")
    return n, d


def _canonical_coloring(
    n: int, d: int, coloring: Sequence[int]
) -> Coloring:
    try:
        result = tuple(
            _exact_integer(color, "basis color") for color in coloring
        )
    except (TypeError, ValueError) as error:
        raise KrennTargetError("a basis coloring is malformed") from error
    if len(result) != n or any(
        color < 0 or color >= d for color in result
    ):
        raise KrennTargetError(
            "a basis coloring is not in range(d)^n"
        )
    return result


def _canonical_coefficient(value: ExactRational) -> ExactRational:
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, Fraction):
        raise KrennTargetError(
            "target coefficients must be exact integers or Fractions"
        )
    return (
        value.numerator
        if value.denominator == 1
        else value
    )


def _coloring_from_index(n: int, d: int, index: int) -> Coloring:
    digits = [0] * n
    for position in range(n - 1, -1, -1):
        index, digits[position] = divmod(index, d)
    return tuple(digits)


@dataclass(frozen=True)
class ColoringTarget:
    """Canonical sparse exact target indexed by computational colorings."""

    n: int
    d: int
    entries: tuple[tuple[Coloring, ExactRational], ...]
    schema: str = TARGET_SCHEMA

    def __post_init__(self) -> None:
        n, d = _validate_dimensions(self.n, self.d)
        if self.schema != TARGET_SCHEMA:
            raise KrennTargetError("target schema changed")
        canonical: list[tuple[Coloring, ExactRational]] = []
        seen: set[Coloring] = set()
        for raw_coloring, raw_value in self.entries:
            coloring = _canonical_coloring(n, d, raw_coloring)
            if coloring in seen:
                raise KrennTargetError(
                    "a target coloring was listed more than once"
                )
            seen.add(coloring)
            value = _canonical_coefficient(raw_value)
            if value:
                canonical.append((coloring, value))
        canonical.sort(key=lambda row: row[0])
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "entries", tuple(canonical))

    @classmethod
    def from_sparse(
        cls,
        n: int,
        d: int,
        coefficients: (
            Mapping[Sequence[int], ExactRational]
            | Iterable[tuple[Sequence[int], ExactRational]]
        ),
    ) -> "ColoringTarget":
        rows = (
            coefficients.items()
            if isinstance(coefficients, Mapping)
            else coefficients
        )
        return cls(
            n=n,
            d=d,
            entries=tuple(
                (tuple(coloring), value) for coloring, value in rows
            ),
        )

    @classmethod
    def from_dense(
        cls,
        n: int,
        d: int,
        coefficients: Sequence[ExactRational],
    ) -> "ColoringTarget":
        n, d = _validate_dimensions(n, d)
        coefficients = tuple(coefficients)
        if len(coefficients) != d**n:
            raise KrennTargetError(
                "dense target length must equal d^n"
            )
        return cls(
            n=n,
            d=d,
            entries=tuple(
                (_coloring_from_index(n, d, index), value)
                for index, value in enumerate(coefficients)
                if _canonical_coefficient(value)
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping) -> "ColoringTarget":
        if set(payload) != {
            "schema",
            "basis_convention",
            "n",
            "d",
            "entries",
        }:
            raise KrennTargetError("target payload schema changed")
        if (
            payload.get("schema") != TARGET_SCHEMA
            or payload.get("basis_convention") != BASIS_CONVENTION
        ):
            raise KrennTargetError(
                "target payload identity or basis convention changed"
            )
        rows = []
        for entry in payload.get("entries", ()):
            if set(entry) != {
                "coloring",
                "numerator",
                "denominator",
            }:
                raise KrennTargetError(
                    "target payload entry schema changed"
                )
            denominator = _exact_integer(
                entry["denominator"], "target payload denominator"
            )
            if denominator <= 0:
                raise KrennTargetError(
                    "target payload denominator must be positive"
                )
            rows.append(
                (
                    tuple(entry["coloring"]),
                    Fraction(
                        _exact_integer(
                            entry["numerator"],
                            "target payload numerator",
                        ),
                        denominator,
                    ),
                )
            )
        return cls.from_sparse(
            _exact_integer(payload["n"], "target factor count"),
            _exact_integer(payload["d"], "target local dimension"),
            rows,
        )

    @property
    def support_size(self) -> int:
        return len(self.entries)

    def coefficient(
        self, coloring: Sequence[int]
    ) -> ExactRational:
        key = _canonical_coloring(self.n, self.d, coloring)
        return dict(self.entries).get(key, 0)

    def dense_coefficients(self) -> tuple[ExactRational, ...]:
        values = dict(self.entries)
        return tuple(
            values.get(coloring, 0)
            for coloring in product(range(self.d), repeat=self.n)
        )

    def to_payload(self) -> Mapping:
        return {
            "schema": TARGET_SCHEMA,
            "basis_convention": BASIS_CONVENTION,
            "n": self.n,
            "d": self.d,
            "entries": [
                {
                    "coloring": list(coloring),
                    "numerator": Fraction(value).numerator,
                    "denominator": Fraction(value).denominator,
                }
                for coloring, value in self.entries
            ],
        }


def canonical_ghz_target(n: int, d: int) -> ColoringTarget:
    """Return ``sum_a |a,...,a>`` in the documented basis convention."""

    n, d = _validate_dimensions(n, d)
    return ColoringTarget.from_sparse(
        n,
        d,
        {
            (color,) * n: 1
            for color in range(d)
        },
    )


def heralded_ghz_target(
    n: int,
    d: int,
    k: int,
    trigger_color: int = 0,
) -> ColoringTarget:
    """Return the Krenn--Gu--Soltész ``k``-monochromatic target.

    Explicitly, this is ``sum_a |a>^k |trigger>^(n-k)``: the first ``k``
    registers carry one varying common color and all remaining registers are
    fixed to ``trigger_color``.  The case ``k=n`` is the canonical GHZ target.
    """

    n, d = _validate_dimensions(n, d)
    k = _exact_integer(k, "monochromatic register count")
    trigger_color = _exact_integer(
        trigger_color, "herald trigger color"
    )
    if not 1 <= k <= n:
        raise KrennTargetError(
            "monochromatic register count must lie between 1 and n"
        )
    if not 0 <= trigger_color < d:
        raise KrennTargetError(
            "herald trigger color is outside range(d)"
        )
    return ColoringTarget.from_sparse(
        n,
        d,
        {
            (color,) * k + (trigger_color,) * (n - k): 1
            for color in range(d)
        },
    )


def unnormalized_qudit_dicke_target(
    occupations: Sequence[int],
) -> ColoringTarget:
    """Return the equal-weight tensor for an exact occupation tuple.

    ``occupations[a]`` is the number of registers carrying basis color ``a``.
    Every distinct computational-basis word with those occupations has
    coefficient one.
    """

    try:
        counts = tuple(
            _exact_integer(value, "Dicke occupation")
            for value in occupations
        )
    except TypeError as error:
        raise KrennTargetError(
            "Dicke occupations must be an integer sequence"
        ) from error
    if not counts or any(count < 0 for count in counts):
        raise KrennTargetError(
            "Dicke occupations must be a nonempty nonnegative tuple"
        )
    n = sum(counts)
    if n < 1:
        raise KrennTargetError(
            "Dicke occupations must contain at least one register"
        )
    d = len(counts)
    words: list[Coloring] = []
    remaining = list(counts)
    prefix: list[int] = []

    def recurse() -> None:
        if len(prefix) == n:
            words.append(tuple(prefix))
            return
        for color in range(d):
            if not remaining[color]:
                continue
            remaining[color] -= 1
            prefix.append(color)
            recurse()
            prefix.pop()
            remaining[color] += 1

    recurse()
    return ColoringTarget.from_sparse(
        n, d, ((word, 1) for word in words)
    )


def unnormalized_dicke_target(n: int, weight: int) -> ColoringTarget:
    """Return the unit-coefficient qubit tensor of Hamming weight ``k``."""

    n, _d = _validate_dimensions(n, 2)
    weight = _exact_integer(weight, "Dicke weight")
    if not 0 <= weight <= n:
        raise KrennTargetError("Dicke weight must lie between 0 and n")
    return unnormalized_qudit_dicke_target(
        (n - weight, weight)
    )


def unnormalized_w_target(n: int) -> ColoringTarget:
    """Return the qubit W tensor, i.e. the weight-one Dicke tensor."""

    return unnormalized_dicke_target(n, 1)


def unnormalized_graph_state_target(
    n: int, edges: Iterable[tuple[int, int]]
) -> ColoringTarget:
    """Return ``prod_{ij in E} CZ_ij |+>^n`` without ``2^(-n/2)``.

    Thus the coefficient of ``|x_0,...,x_{n-1}>`` is
    ``(-1)^(sum_{ij in E} x_i x_j)`` for a simple undirected graph.
    """

    n, _d = _validate_dimensions(n, 2)
    canonical_edges: set[tuple[int, int]] = set()
    for raw_i, raw_j in edges:
        i = _exact_integer(raw_i, "graph-state vertex")
        j = _exact_integer(raw_j, "graph-state vertex")
        if i == j or not (0 <= i < n and 0 <= j < n):
            raise KrennTargetError(
                "graph-state edges need distinct valid vertices"
            )
        edge = (i, j) if i < j else (j, i)
        if edge in canonical_edges:
            raise KrennTargetError(
                "graph-state edges must be unique"
            )
        canonical_edges.add(edge)
    return ColoringTarget.from_dense(
        n,
        2,
        tuple(
            (
                -1
                if sum(
                    coloring[i] * coloring[j]
                    for i, j in canonical_edges
                )
                % 2
                else 1
            )
            for coloring in product(range(2), repeat=n)
        ),
    )


def dicke_support_size(n: int, weight: int) -> int:
    """Return the support census of ``unnormalized_dicke_target``."""

    n, _d = _validate_dimensions(n, 2)
    weight = _exact_integer(weight, "Dicke weight")
    if not 0 <= weight <= n:
        raise KrennTargetError("Dicke weight must lie between 0 and n")
    return comb(n, weight)
