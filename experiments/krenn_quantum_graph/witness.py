"""Sparse exact witnesses and equation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Iterable, Mapping

from experiments.krenn_quantum_graph.system import (
    KrennSystemError,
    SparsePolynomialSystem,
    VariableKey,
    canonical_variable_key,
    validate_parameters,
    variable_count,
    variable_index,
    variable_key,
)


ExactScalar = int | Fraction

WITNESS_SCHEMA = "krenn-quantum-graph-sparse-witness-v1"


class KrennWitnessError(ValueError):
    """A sparse witness or its coefficient projection is malformed."""


@dataclass(frozen=True)
class SparseWitness:
    """A sorted sparse rational vector; every omitted coordinate is zero."""

    n: int
    d: int
    entries: tuple[tuple[int, Fraction], ...]
    schema: str = WITNESS_SCHEMA

    def __post_init__(self) -> None:
        n, d = validate_parameters(self.n, self.d)
        normalized = tuple(
            (int(index), Fraction(value))
            for index, value in self.entries
        )
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "entries", normalized)
        if self.schema != WITNESS_SCHEMA:
            raise KrennWitnessError("sparse witness schema changed")
        indices = tuple(index for index, _value in normalized)
        if indices != tuple(sorted(indices)) or len(indices) != len(
            set(indices)
        ):
            raise KrennWitnessError(
                "witness entries must have unique increasing indices"
            )
        count = variable_count(n, d)
        if any(index < 0 or index >= count for index in indices):
            raise KrennWitnessError(
                "a witness variable index is outside the system"
            )
        if any(value == 0 for _index, value in normalized):
            raise KrennWitnessError(
                "zero entries must be omitted from a sparse witness"
            )

    @classmethod
    def from_index_values(
        cls,
        n: int,
        d: int,
        values: Mapping[int, ExactScalar]
        | Iterable[tuple[int, ExactScalar]],
    ) -> SparseWitness:
        items = values.items() if isinstance(values, Mapping) else values
        combined: dict[int, Fraction] = {}
        for raw_index, raw_value in items:
            index = int(raw_index)
            if index in combined:
                raise KrennWitnessError(
                    f"duplicate witness variable index {index}"
                )
            value = Fraction(raw_value)
            if value:
                combined[index] = value
        return cls(n, d, tuple(sorted(combined.items())))

    @classmethod
    def from_coordinates(
        cls,
        n: int,
        d: int,
        values: Mapping[VariableKey, ExactScalar]
        | Iterable[tuple[VariableKey, ExactScalar]],
    ) -> SparseWitness:
        items = values.items() if isinstance(values, Mapping) else values
        indexed: list[tuple[int, ExactScalar]] = []
        seen: set[int] = set()
        for raw_key, value in items:
            if len(raw_key) != 4:
                raise KrennWitnessError(
                    "a witness coordinate must have four entries"
                )
            key = tuple(map(int, raw_key))
            canonical = canonical_variable_key(n, d, *key)
            if key != canonical:
                raise KrennWitnessError(
                    "witness coordinates must use canonical endpoints i<j"
                )
            index = variable_index(n, d, *key)
            if index in seen:
                raise KrennWitnessError(
                    f"duplicate witness coordinate {key}"
                )
            seen.add(index)
            indexed.append((index, value))
        return cls.from_index_values(n, d, indexed)

    @property
    def support_size(self) -> int:
        return len(self.entries)

    @property
    def integral(self) -> bool:
        return all(value.denominator == 1 for _index, value in self.entries)

    def value(self, index: int) -> Fraction:
        index = int(index)
        for candidate, value in self.entries:
            if candidate == index:
                return value
            if candidate > index:
                break
        return Fraction(0)

    def as_dict(self) -> dict[int, Fraction]:
        return dict(self.entries)

    def coordinate_entries(
        self,
    ) -> tuple[tuple[VariableKey, Fraction], ...]:
        return tuple(
            (variable_key(self.n, self.d, index), value)
            for index, value in self.entries
        )


@dataclass(frozen=True)
class WitnessEvaluation:
    """Complete exact residual replay in one coefficient domain."""

    coefficient_domain: str
    equation_values: tuple[Fraction | int, ...]
    residuals: tuple[Fraction | int, ...]

    @property
    def satisfied(self) -> bool:
        return all(residual == 0 for residual in self.residuals)

    @property
    def nonzero_residual_count(self) -> int:
        return sum(residual != 0 for residual in self.residuals)


def _check_compatible(
    system: SparsePolynomialSystem, witness: SparseWitness
) -> None:
    if (system.n, system.d) != (witness.n, witness.d):
        raise KrennWitnessError(
            "witness parameters do not match the polynomial system"
        )


def evaluate_exact(
    system: SparsePolynomialSystem,
    witness: SparseWitness,
) -> WitnessEvaluation:
    """Evaluate all equations over ``Q`` (integers embed exactly)."""

    _check_compatible(system, witness)
    values = witness.as_dict()
    equation_values: list[Fraction] = []
    residuals: list[Fraction] = []
    for equation in range(system.equation_count):
        total = Fraction(0)
        for monomial in system.equation_monomials(equation):
            term = Fraction(1)
            for variable in monomial:
                term *= values.get(variable, Fraction(0))
            total += term
        equation_values.append(total)
        residuals.append(total - system.rhs_values[equation])
    return WitnessEvaluation(
        coefficient_domain="Q",
        equation_values=tuple(equation_values),
        residuals=tuple(residuals),
    )


def fraction_mod(value: ExactScalar, modulus: int = 31) -> int:
    value = Fraction(value)
    modulus = int(modulus)
    if modulus < 2:
        raise KrennWitnessError("modulus must be at least 2")
    denominator = value.denominator % modulus
    if gcd(denominator, modulus) != 1:
        raise KrennWitnessError(
            "a rational denominator is not invertible modulo the modulus"
        )
    return (
        value.numerator
        * pow(denominator, -1, modulus)
    ) % modulus


def evaluate_mod(
    system: SparsePolynomialSystem,
    witness: SparseWitness,
    modulus: int = 31,
) -> WitnessEvaluation:
    """Evaluate all equations after exact rational reduction modulo ``p``."""

    _check_compatible(system, witness)
    modulus = int(modulus)
    values = {
        index: fraction_mod(value, modulus)
        for index, value in witness.entries
    }
    equation_values: list[int] = []
    residuals: list[int] = []
    for equation in range(system.equation_count):
        total = 0
        for monomial in system.equation_monomials(equation):
            term = 1
            for variable in monomial:
                term = term * values.get(variable, 0) % modulus
            total = (total + term) % modulus
        equation_values.append(total)
        target_value = fraction_mod(
            system.rhs_values[equation], modulus
        )
        residuals.append((total - target_value) % modulus)
    return WitnessEvaluation(
        coefficient_domain=f"F_{modulus}",
        equation_values=tuple(equation_values),
        residuals=tuple(residuals),
    )


def require_exact_witness(
    system: SparsePolynomialSystem,
    witness: SparseWitness,
) -> tuple[WitnessEvaluation, WitnessEvaluation]:
    """Return the exact-Q and F_31 replays, rejecting either failure."""

    rational = evaluate_exact(system, witness)
    finite = evaluate_mod(system, witness, 31)
    if not rational.satisfied or not finite.satisfied:
        raise KrennSystemError(
            "the proposed witness does not satisfy the Krenn system"
        )
    return rational, finite
