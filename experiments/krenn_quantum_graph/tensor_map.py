r"""Target-independent perfect-matching tensor map over the rationals.

For canonical endpoints ``i < j``, a sparse witness supplies the coefficients
``W[i,j,a,b]`` of a two-site tensor.  This module assembles the coefficient
tensor

.. math::

    \Phi(W)_c =
    \sum_{M \in \operatorname{PM}(K_n)}
    \prod_{\{i,j\}\in M} W[i,j,c_i,c_j]

without choosing a target.  Computational-basis colorings are lexicographic,
with the last vertex changing fastest, exactly as in ``system.py``.

Only affine equality and exact real-rational fidelity are evaluated here.
Projective, local-diagonal-orbit, and border-image questions have explicit
claim contracts below, but this module does not pretend to solve them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from typing import Sequence

from experiments.krenn_quantum_graph.system import (
    Coloring,
    Monomial,
    coloring_index,
    enumerate_colorings,
    equation_count,
    perfect_matchings,
    validate_parameters,
    variable_count,
    variable_index,
)
from experiments.krenn_quantum_graph.targets import ColoringTarget
from experiments.krenn_quantum_graph.transport import (
    equation_permutation,
    transport_coloring,
    transport_witness,
    validate_permutation,
    variable_permutation,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


TENSOR_MAP_SCHEMA = "krenn-perfect-matching-tensor-map-v1"


class KrennTensorMapError(ValueError):
    """A tensor map, target comparison, or fidelity request is malformed."""


class ClaimMode(str, Enum):
    """Mathematically distinct relations between ``Phi(W)`` and a target."""

    AFFINE = "affine"
    PROJECTIVE = "projective"
    LOCAL_DIAGONAL = "local-diagonal"
    BORDER = "border"
    FIDELITY = "fidelity"


@dataclass(frozen=True)
class ClaimModeRecord:
    """Fail-closed capability record for one target relation."""

    mode: ClaimMode
    relation: str
    finite_weight_witness: bool
    implemented_here: bool
    proves_affine_image_membership: bool
    proves_border_image_membership: bool
    metric_only: bool
    claim_boundary: str


CLAIM_MODE_RECORDS = {
    ClaimMode.AFFINE: ClaimModeRecord(
        mode=ClaimMode.AFFINE,
        relation="Phi(W) = T",
        finite_weight_witness=True,
        implemented_here=True,
        proves_affine_image_membership=True,
        proves_border_image_membership=False,
        metric_only=False,
        claim_boundary=(
            "Exact rational coefficient equality for one finite witness."
        ),
    ),
    ClaimMode.PROJECTIVE: ClaimModeRecord(
        mode=ClaimMode.PROJECTIVE,
        relation="Phi(W) = lambda*T for some nonzero lambda",
        finite_weight_witness=True,
        implemented_here=False,
        proves_affine_image_membership=False,
        proves_border_image_membership=False,
        metric_only=False,
        claim_boundary=(
            "Recorded as a distinct exact relation; no projective solver or "
            "normalization certificate is implemented here."
        ),
    ),
    ClaimMode.LOCAL_DIAGONAL: ClaimModeRecord(
        mode=ClaimMode.LOCAL_DIAGONAL,
        relation=(
            "Phi(W) lies in the invertible local-diagonal orbit of T"
        ),
        finite_weight_witness=True,
        implemented_here=False,
        proves_affine_image_membership=False,
        proves_border_image_membership=False,
        metric_only=False,
        claim_boundary=(
            "Local-diagonal equivalence is not conflated with a common "
            "projective scalar and requires its own exact certificate."
        ),
    ),
    ClaimMode.BORDER: ClaimModeRecord(
        mode=ClaimMode.BORDER,
        relation="T lies in the algebraic closure of the image of Phi",
        finite_weight_witness=False,
        implemented_here=False,
        proves_affine_image_membership=False,
        proves_border_image_membership=False,
        metric_only=False,
        claim_boundary=(
            "No degeneration or closure-membership solver is implemented; "
            "numerical convergence is not a border-image certificate."
        ),
    ),
    ClaimMode.FIDELITY: ClaimModeRecord(
        mode=ClaimMode.FIDELITY,
        relation=(
            "<T,Phi(W)>^2/(<T,T>*<Phi(W),Phi(W)>) over real Q"
        ),
        finite_weight_witness=True,
        implemented_here=True,
        proves_affine_image_membership=False,
        proves_border_image_membership=False,
        metric_only=True,
        claim_boundary=(
            "The returned rational is an exact metric value only and is "
            "not a border-image certificate or exact image membership."
        ),
    ),
}


def claim_mode_record(mode: ClaimMode | str) -> ClaimModeRecord:
    """Return the immutable capability record for ``mode``."""

    try:
        canonical = mode if isinstance(mode, ClaimMode) else ClaimMode(mode)
    except ValueError as error:
        raise KrennTensorMapError("unknown tensor claim mode") from error
    return CLAIM_MODE_RECORDS[canonical]


@lru_cache(maxsize=None)
def _canonical_structure(
    n: int, d: int
) -> tuple[tuple[int, ...], tuple[Monomial, ...]]:
    """Build the flattened monomial structure without constructing a target."""

    n, d = validate_parameters(n, d)
    matchings = perfect_matchings(n)
    offsets = [0]
    monomials: list[Monomial] = []
    for coloring in enumerate_colorings(n, d):
        for matching in matchings:
            monomials.append(
                tuple(
                    variable_index(
                        n,
                        d,
                        i,
                        j,
                        coloring[i],
                        coloring[j],
                    )
                    for i, j in matching
                )
            )
        offsets.append(len(monomials))
    return tuple(offsets), tuple(monomials)


@dataclass(frozen=True)
class ExactTensorComparison:
    """Exact affine comparison of a rational output with a rational target."""

    output: ColoringTarget
    target: ColoringTarget
    residual: ColoringTarget

    def __post_init__(self) -> None:
        dimensions = {
            (self.output.n, self.output.d),
            (self.target.n, self.target.d),
            (self.residual.n, self.residual.d),
        }
        if len(dimensions) != 1:
            raise KrennTensorMapError(
                "output, target, and residual dimensions differ"
            )
        expected = tuple(
            Fraction(output) - Fraction(target)
            for output, target in zip(
                self.output.dense_coefficients(),
                self.target.dense_coefficients(),
                strict=True,
            )
        )
        actual = tuple(
            map(Fraction, self.residual.dense_coefficients())
        )
        if actual != expected:
            raise KrennTensorMapError(
                "the stored tensor residual is not output minus target"
            )

    @property
    def satisfied(self) -> bool:
        return self.residual.support_size == 0

    @property
    def nonzero_residual_count(self) -> int:
        return self.residual.support_size

    @property
    def residual_norm_squared(self) -> Fraction:
        return sum(
            (
                Fraction(value) * Fraction(value)
                for value in self.residual.dense_coefficients()
            ),
            Fraction(0),
        )


@dataclass(frozen=True)
class ExactRationalFidelity:
    """Exact fidelity data for two nonzero real-rational tensors."""

    value: Fraction
    inner_product: Fraction
    output_norm_squared: Fraction
    target_norm_squared: Fraction

    def __post_init__(self) -> None:
        values = tuple(
            map(
                Fraction,
                (
                    self.value,
                    self.inner_product,
                    self.output_norm_squared,
                    self.target_norm_squared,
                ),
            )
        )
        object.__setattr__(self, "value", values[0])
        object.__setattr__(self, "inner_product", values[1])
        object.__setattr__(self, "output_norm_squared", values[2])
        object.__setattr__(self, "target_norm_squared", values[3])
        if self.output_norm_squared <= 0 or self.target_norm_squared <= 0:
            raise KrennTensorMapError(
                "fidelity requires two nonzero real-rational tensors"
            )
        expected = (
            self.inner_product * self.inner_product
            / (self.output_norm_squared * self.target_norm_squared)
        )
        if self.value != expected or not 0 <= self.value <= 1:
            raise KrennTensorMapError(
                "the exact rational fidelity record is inconsistent"
            )


@dataclass(frozen=True)
class MatchingTensorMap:
    """Canonical target-independent perfect-matching polynomial map."""

    n: int
    d: int
    schema: str = TENSOR_MAP_SCHEMA
    equation_offsets: tuple[int, ...] = field(init=False, repr=False)
    monomial_variable_indices: tuple[Monomial, ...] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        n, d = validate_parameters(self.n, self.d)
        if self.schema != TENSOR_MAP_SCHEMA:
            raise KrennTensorMapError("tensor-map schema changed")
        offsets, monomials = _canonical_structure(n, d)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "equation_offsets", offsets)
        object.__setattr__(
            self, "monomial_variable_indices", monomials
        )

    @property
    def variable_count(self) -> int:
        return variable_count(self.n, self.d)

    @property
    def coefficient_count(self) -> int:
        return equation_count(self.n, self.d)

    @property
    def matching_count(self) -> int:
        return len(perfect_matchings(self.n))

    @property
    def degree(self) -> int:
        return self.n // 2

    @property
    def monomial_count(self) -> int:
        return len(self.monomial_variable_indices)

    def equation_monomials(self, equation: int) -> tuple[Monomial, ...]:
        equation = int(equation)
        if not 0 <= equation < self.coefficient_count:
            raise KrennTensorMapError(
                "tensor coefficient index is outside the map"
            )
        start = self.equation_offsets[equation]
        stop = self.equation_offsets[equation + 1]
        return self.monomial_variable_indices[start:stop]

    def _check_witness(self, witness: SparseWitness) -> None:
        if (witness.n, witness.d) != (self.n, self.d):
            raise KrennTensorMapError(
                "witness dimensions do not match the tensor map"
            )

    def _check_target(self, target: ColoringTarget) -> None:
        if (target.n, target.d) != (self.n, self.d):
            raise KrennTensorMapError(
                "target dimensions do not match the tensor map"
            )

    def coefficient_exact(
        self,
        witness: SparseWitness,
        coloring: Sequence[int],
    ) -> Fraction:
        """Evaluate one computational-basis coefficient over ``Q``."""

        self._check_witness(witness)
        equation = coloring_index(
            self.n, self.d, tuple(map(int, coloring))
        )
        values = witness.as_dict()
        total = Fraction(0)
        for monomial in self.equation_monomials(equation):
            term = Fraction(1)
            for variable in monomial:
                term *= values.get(variable, Fraction(0))
            total += term
        return total

    def evaluate_exact(self, witness: SparseWitness) -> ColoringTarget:
        """Evaluate the complete output tensor over ``Q``."""

        self._check_witness(witness)
        values = witness.as_dict()
        coefficients: list[Fraction] = []
        for equation in range(self.coefficient_count):
            total = Fraction(0)
            for monomial in self.equation_monomials(equation):
                term = Fraction(1)
                for variable in monomial:
                    term *= values.get(variable, Fraction(0))
                total += term
            coefficients.append(total)
        return ColoringTarget.from_dense(
            self.n, self.d, coefficients
        )

    def exact_residual(
        self,
        witness: SparseWitness,
        target: ColoringTarget,
    ) -> ColoringTarget:
        """Return ``Phi(witness) - target`` coefficient by coefficient."""

        self._check_target(target)
        output = self.evaluate_exact(witness)
        return ColoringTarget.from_dense(
            self.n,
            self.d,
            tuple(
                Fraction(value) - Fraction(expected)
                for value, expected in zip(
                    output.dense_coefficients(),
                    target.dense_coefficients(),
                    strict=True,
                )
            ),
        )

    def compare_exact(
        self,
        witness: SparseWitness,
        target: ColoringTarget,
    ) -> ExactTensorComparison:
        """Return a fail-closed exact affine comparison record."""

        self._check_target(target)
        output = self.evaluate_exact(witness)
        residual = ColoringTarget.from_dense(
            self.n,
            self.d,
            tuple(
                Fraction(value) - Fraction(expected)
                for value, expected in zip(
                    output.dense_coefficients(),
                    target.dense_coefficients(),
                    strict=True,
                )
            ),
        )
        return ExactTensorComparison(
            output=output,
            target=target,
            residual=residual,
        )

    def exact_fidelity(
        self,
        witness: SparseWitness,
        target: ColoringTarget,
    ) -> ExactRationalFidelity:
        """Return exact real-rational fidelity, rejecting zero tensors."""

        self._check_target(target)
        output = self.evaluate_exact(witness)
        output_values = tuple(
            map(Fraction, output.dense_coefficients())
        )
        target_values = tuple(
            map(Fraction, target.dense_coefficients())
        )
        inner = sum(
            (
                value * expected
                for value, expected in zip(
                    output_values, target_values, strict=True
                )
            ),
            Fraction(0),
        )
        output_norm = sum(
            (value * value for value in output_values),
            Fraction(0),
        )
        target_norm = sum(
            (value * value for value in target_values),
            Fraction(0),
        )
        if output_norm == 0:
            raise KrennTensorMapError(
                "fidelity is undefined for zero tensor-map output"
            )
        if target_norm == 0:
            raise KrennTensorMapError(
                "fidelity is undefined for the zero target"
            )
        return ExactRationalFidelity(
            value=inner * inner / (output_norm * target_norm),
            inner_product=inner,
            output_norm_squared=output_norm,
            target_norm_squared=target_norm,
        )


def transport_target(
    target: ColoringTarget,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> ColoringTarget:
    """Push target coefficients forward by the existing transport law."""

    vertex_permutation = validate_permutation(
        vertex_permutation, target.n
    )
    color_permutation = validate_permutation(
        color_permutation, target.d
    )
    transported = tuple(
        (
            transport_coloring(
                coloring,
                target.d,
                vertex_permutation,
                color_permutation,
            ),
            value,
        )
        for coloring, value in target.entries
    )
    result = ColoringTarget.from_sparse(
        target.n, target.d, transported
    )
    if result.support_size != target.support_size:
        raise KrennTensorMapError(
            "target transport unexpectedly collided basis coefficients"
        )
    return result


@dataclass(frozen=True)
class TensorTransportCertificate:
    """Exact equivariance replay for output, target, and affine residual."""

    n: int
    d: int
    vertex_permutation: tuple[int, ...]
    color_permutation: tuple[int, ...]
    variable_bijection: bool
    coefficient_bijection: bool
    target_support_bijection: bool
    map_equivariant: bool
    residual_equivariant: bool
    source_satisfied: bool
    transported_satisfied: bool

    @property
    def exact(self) -> bool:
        return (
            self.variable_bijection
            and self.coefficient_bijection
            and self.target_support_bijection
            and self.map_equivariant
            and self.residual_equivariant
            and self.source_satisfied == self.transported_satisfied
        )


def certify_map_equivariance(
    tensor_map: MatchingTensorMap,
    witness: SparseWitness,
    target: ColoringTarget,
    vertex_permutation: Sequence[int],
    color_permutation: Sequence[int],
) -> TensorTransportCertificate:
    """Certify ``Phi(g.W)=g.Phi(W)`` against the transported target."""

    tensor_map._check_witness(witness)
    tensor_map._check_target(target)
    vertex_permutation = validate_permutation(
        vertex_permutation, tensor_map.n
    )
    color_permutation = validate_permutation(
        color_permutation, tensor_map.d
    )

    source = tensor_map.compare_exact(witness, target)
    transported_witness = transport_witness(
        witness, vertex_permutation, color_permutation
    )
    transported_target = transport_target(
        target, vertex_permutation, color_permutation
    )
    transported = tensor_map.compare_exact(
        transported_witness, transported_target
    )
    expected_output = transport_target(
        source.output, vertex_permutation, color_permutation
    )
    expected_residual = transport_target(
        source.residual, vertex_permutation, color_permutation
    )

    variables = variable_permutation(
        tensor_map.n,
        tensor_map.d,
        vertex_permutation,
        color_permutation,
    )
    coefficients = equation_permutation(
        tensor_map.n,
        tensor_map.d,
        vertex_permutation,
        color_permutation,
    )
    certificate = TensorTransportCertificate(
        n=tensor_map.n,
        d=tensor_map.d,
        vertex_permutation=vertex_permutation,
        color_permutation=color_permutation,
        variable_bijection=(
            set(variables) == set(range(tensor_map.variable_count))
        ),
        coefficient_bijection=(
            set(coefficients) == set(range(tensor_map.coefficient_count))
        ),
        target_support_bijection=(
            transported_target.support_size == target.support_size
        ),
        map_equivariant=transported.output == expected_output,
        residual_equivariant=(
            transported.residual == expected_residual
        ),
        source_satisfied=source.satisfied,
        transported_satisfied=transported.satisfied,
    )
    if not certificate.exact:
        raise KrennTensorMapError(
            "perfect-matching tensor transport replay failed"
        )
    return certificate
