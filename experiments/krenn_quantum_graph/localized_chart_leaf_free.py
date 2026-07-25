r"""Exact inverse-leaf removal and sequential-saturation chart systems.

For a polynomial factor ``A``, adjoining an inverse leaf ``u*A-1`` is the
same localization as saturating by ``A`` after eliminating ``u``.  Removing
the three pure-amplitude inverse leaves therefore replaces them by
sequential saturation with the three pure outputs.  Repair charts also
remove their branch inverse and first saturate by the localized weight.

This module constructs the smaller degree-three source ideals and their
individual saturation factors.  It does not claim that any saturation has
completed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    DERIVATIVE_REPRESENTATIVES,
    sparse_derivative_gauge_slice,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    KrennLocalizedChartError,
    SparseChartPolynomial,
    strict_json_equal,
    validate_singular_characteristic,
)
from experiments.krenn_quantum_graph.localized_chart_monomial_atlas import (
    REPAIR_MONOMIAL_REPRESENTATIVES,
    repair_monomial_chart,
)


LEAF_FREE_SYSTEM_SCHEMA = "krenn-n6-d3-leaf-free-chart-system-v1"
LEAF_FREE_AUDIT_SCHEMA = "krenn-n6-d3-leaf-free-saturation-atlas-v1"


class KrennLeafFreeError(RuntimeError):
    """An inverse-leaf elimination or saturation factor failed replay."""


def _remap_polynomial(
    polynomial: SparseChartPolynomial,
    remap: Mapping[int, int],
) -> SparseChartPolynomial:
    result = {}
    for coefficient, monomial in polynomial.terms:
        try:
            key = tuple(sorted(remap[variable] for variable in monomial))
        except KeyError as error:
            raise KrennLeafFreeError(
                "a supposedly leaf-free polynomial retained an inverse"
            ) from error
        total = result.get(key, 0) + coefficient
        if total:
            result[key] = total
        else:
            result.pop(key, None)
    return SparseChartPolynomial.from_mapping(result)


def _extract_factor(
    localizer: SparseChartPolynomial,
    inverse_variable: int,
    remap: Mapping[int, int],
) -> SparseChartPolynomial:
    factor = {}
    constant = 0
    for coefficient, monomial in localizer.terms:
        multiplicity = monomial.count(inverse_variable)
        if not multiplicity:
            if monomial:
                raise KrennLeafFreeError(
                    "a localizer has a nonconstant inverse-free term"
                )
            constant += coefficient
            continue
        if multiplicity != 1:
            raise KrennLeafFreeError(
                "an inverse leaf appears nonlinearly"
            )
        reduced = list(monomial)
        reduced.remove(inverse_variable)
        try:
            key = tuple(sorted(remap[variable] for variable in reduced))
        except KeyError as error:
            raise KrennLeafFreeError(
                "a pure factor retained another inverse leaf"
            ) from error
        factor[key] = factor.get(key, 0) + coefficient
    if constant != -1:
        raise KrennLeafFreeError(
            "a localizer constant changed from -1"
        )
    result = SparseChartPolynomial.from_mapping(factor)
    if result.term_count != 15 or result.degree != 3:
        raise KrennLeafFreeError(
            "a pure-output saturation factor changed"
        )
    return result


@dataclass(frozen=True)
class LeafFreeChartSystem:
    """A compact weight-only ideal and its sequential saturation factors."""

    key: tuple[int, ...]
    kind: str
    source_variable_indices: tuple[int, ...]
    generator_labels: tuple[tuple[str, int], ...]
    generators: tuple[SparseChartPolynomial, ...]
    saturation_factor_labels: tuple[str, ...]
    saturation_factors: tuple[SparseChartPolynomial, ...]
    schema: str = LEAF_FREE_SYSTEM_SCHEMA

    def __post_init__(self) -> None:
        if self.kind == "derivative":
            expected = (126, 727, 10_894, 3, 3)
        elif self.kind == "quadratic-repair":
            expected = (125, 726, 10_890, 3, 4)
        elif self.kind == "cubic-repair":
            expected = (124, 726, 10_890, 3, 4)
        else:
            raise KrennLeafFreeError("unknown leaf-free chart kind")
        observed = (
            self.variable_count,
            len(self.generators),
            self.term_count,
            self.maximum_degree,
            len(self.saturation_factors),
        )
        if (
            self.schema != LEAF_FREE_SYSTEM_SCHEMA
            or observed != expected
            or len(self.source_variable_indices) != self.variable_count
            or len(set(self.source_variable_indices))
            != self.variable_count
            or len(self.generator_labels) != len(self.generators)
            or len(self.saturation_factor_labels)
            != len(self.saturation_factors)
            or self.saturation_factor_labels[-3:]
            != ("A0", "A1", "A2")
            or any(
                factor.term_count != 15 or factor.degree != 3
                for factor in self.saturation_factors[-3:]
            )
        ):
            raise KrennLeafFreeError(
                "a leaf-free chart census changed"
            )
        if self.kind != "derivative":
            branch = self.saturation_factors[0]
            if branch.term_count != 1 or branch.degree != 1:
                raise KrennLeafFreeError(
                    "a repair branch factor is not one variable"
                )

    @property
    def variable_count(self) -> int:
        return len(self.source_variable_indices)

    @property
    def term_count(self) -> int:
        return sum(polynomial.term_count for polynomial in self.generators)

    @property
    def maximum_degree(self) -> int:
        return max(polynomial.degree for polynomial in self.generators)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.key}|{self.kind}|"
                f"{self.source_variable_indices}\n"
            ).encode("ascii")
        )
        for label, polynomial in zip(
            self.generator_labels, self.generators, strict=True
        ):
            digest.update(f"{label[0]}:{label[1]}|".encode("ascii"))
            for coefficient, monomial in polynomial.terms:
                digest.update(
                    f"{coefficient}:{','.join(map(str, monomial))};".encode(
                        "ascii"
                    )
                )
            digest.update(b"\n")
        for label, polynomial in zip(
            self.saturation_factor_labels,
            self.saturation_factors,
            strict=True,
        ):
            digest.update(f"factor:{label}|".encode("ascii"))
            for coefficient, monomial in polynomial.terms:
                digest.update(
                    f"{coefficient}:{','.join(map(str, monomial))};".encode(
                        "ascii"
                    )
                )
            digest.update(b"\n")
        return digest.hexdigest()

    def summary(self) -> dict:
        return {
            "schema": self.schema,
            "key": list(self.key),
            "kind": self.kind,
            "variables": self.variable_count,
            "generators": len(self.generators),
            "terms": self.term_count,
            "maximum_degree": self.maximum_degree,
            "saturation_order": list(self.saturation_factor_labels),
            "saturation_factors": [
                {
                    "label": label,
                    "terms": factor.term_count,
                    "degree": factor.degree,
                }
                for label, factor in zip(
                    self.saturation_factor_labels,
                    self.saturation_factors,
                    strict=True,
                )
            ],
            "sha256": self.fingerprint(),
        }


def derivative_leaf_free_system(
    derivative_ambient_weight: int,
) -> LeafFreeChartSystem:
    """Remove the three pure inverse leaves from one ``d=1`` slice."""

    derivative_ambient_weight = int(derivative_ambient_weight)
    if derivative_ambient_weight not in DERIVATIVE_REPRESENTATIVES:
        raise KrennLeafFreeError(
            "unknown derivative representative"
        )
    chart = sparse_derivative_gauge_slice(
        derivative_ambient_weight
    )
    source_variables = tuple(range(126))
    remap = {variable: variable for variable in source_variables}
    mixed_rows = tuple(
        (label, polynomial)
        for label, polynomial in zip(
            chart.generator_labels, chart.generators, strict=True
        )
        if label[0] == "mixed"
    )
    pure_rows = tuple(
        (label, polynomial)
        for label, polynomial in zip(
            chart.generator_labels, chart.generators, strict=True
        )
        if label[0] == "pure-localizer"
    )
    if len(mixed_rows) != 726 or len(pure_rows) != 3:
        raise KrennLeafFreeError(
            "the derivative generator partition changed"
        )
    factors = tuple(
        _extract_factor(polynomial, 126 + color, remap)
        for color, (_label, polynomial) in enumerate(pure_rows)
    )
    return LeafFreeChartSystem(
        key=(derivative_ambient_weight,),
        kind="derivative",
        source_variable_indices=source_variables,
        generator_labels=(
            *(label for label, _polynomial in mixed_rows),
            ("derivative-gauge", derivative_ambient_weight),
        ),
        generators=(
            *(
                _remap_polynomial(polynomial, remap)
                for _label, polynomial in mixed_rows
            ),
            _remap_polynomial(chart.generators[-1], remap),
        ),
        saturation_factor_labels=("A0", "A1", "A2"),
        saturation_factors=factors,
    )


def repair_leaf_free_system(
    representative_ambient_monomial: Sequence[int],
) -> LeafFreeChartSystem:
    """Remove all four inverse leaves from one repair chart."""

    representative = tuple(sorted(map(
        int, representative_ambient_monomial
    )))
    if representative not in REPAIR_MONOMIAL_REPRESENTATIVES:
        raise KrennLeafFreeError("unknown repair representative")
    chart = repair_monomial_chart(representative)
    weight_chart_variables = tuple(
        reduced
        for reduced, original in enumerate(
            chart.retained_original_chart_variables
        )
        if original < 126
    )
    source_variables = tuple(
        chart.retained_original_chart_variables[reduced]
        for reduced in weight_chart_variables
    )
    remap = {
        reduced: compact
        for compact, reduced in enumerate(weight_chart_variables)
    }
    mixed_rows = tuple(
        (label, polynomial)
        for label, polynomial in zip(
            chart.generator_labels, chart.generators, strict=True
        )
        if label[0] == "mixed"
    )
    pure_rows = tuple(
        (label, polynomial)
        for label, polynomial in zip(
            chart.generator_labels, chart.generators, strict=True
        )
        if label[0] == "pure-localizer"
    )
    repair_rows = tuple(
        (label, polynomial)
        for label, polynomial in zip(
            chart.generator_labels, chart.generators, strict=True
        )
        if label[0] == "repair-localizer"
    )
    if (
        len(mixed_rows) != 726
        or len(pure_rows) != 3
        or len(repair_rows) != 1
    ):
        raise KrennLeafFreeError(
            "the repair generator partition changed"
        )
    factors = []
    for color, (_label, polynomial) in enumerate(pure_rows):
        original_inverse = 126 + color
        inverse = chart.retained_original_chart_variables.index(
            original_inverse
        )
        factors.append(_extract_factor(polynomial, inverse, remap))
    localized_reduced = (
        chart.retained_original_chart_variables.index(
            chart.localized_original_chart_variable
        )
    )
    repair_inverse = len(chart.retained_original_chart_variables)
    expected_localizer_terms = (
        (-1, ()),
        (1, tuple(sorted((localized_reduced, repair_inverse)))),
    )
    if repair_rows[0][1].terms != expected_localizer_terms:
        raise KrennLeafFreeError(
            "the dropped repair localizer is not exactly u*y-1"
        )
    localized_compact = remap[localized_reduced]
    branch = SparseChartPolynomial.from_mapping({
        (localized_compact,): 1
    })
    return LeafFreeChartSystem(
        key=representative,
        kind=(
            "quadratic-repair"
            if len(representative) == 2 else "cubic-repair"
        ),
        source_variable_indices=source_variables,
        generator_labels=tuple(
            label for label, _polynomial in mixed_rows
        ),
        generators=tuple(
            _remap_polynomial(polynomial, remap)
            for _label, polynomial in mixed_rows
        ),
        saturation_factor_labels=("y", "A0", "A1", "A2"),
        saturation_factors=(branch, *factors),
    )


def leaf_free_saturation_audit() -> dict:
    """Return all six exact leaf-free systems and fail-closed claims."""

    derivative = tuple(
        derivative_leaf_free_system(ambient)
        for ambient in DERIVATIVE_REPRESENTATIVES
    )
    repair = tuple(
        repair_leaf_free_system(representative)
        for representative in REPAIR_MONOMIAL_REPRESENTATIVES
    )
    return json.loads(json.dumps({
        "schema": LEAF_FREE_AUDIT_SCHEMA,
        "derivative_systems": [
            system.summary() for system in derivative
        ],
        "repair_systems": [
            system.summary() for system in repair
        ],
        "exact_equivalence": {
            "single_leaf": (
                "k[w,u]/(I,u*A-1) is the localization "
                "(k[w]/I)_A"
            ),
            "all_dropped_localizers_replayed_exactly": True,
            "repair_localizers_are_u_times_y_minus_one": True,
            "sequential_saturation": (
                "((I:f^infinity):g^infinity) "
                "= I:(f*g)^infinity"
            ),
            "one_shot_product_formed": False,
        },
        "decision_rule": {
            "exact_Q_unit": "excludes that localized chart",
            "exact_Q_proper": (
                "proves an algebraic-complex finite point in that chart"
            ),
            "finite_field_or_timeout": "reconnaissance only",
        },
        "claim_boundary": {
            "saturation_stages_completed": 0,
            "natural_chart_decided": False,
            "full_chart_cover_decided": False,
        },
    }))


def verify_leaf_free_saturation_audit(payload: Mapping) -> dict:
    """Rebuild all six systems and reject any altered claim."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennLeafFreeError(
            "leaf-free audit is not strict JSON"
        ) from error
    expected = leaf_free_saturation_audit()
    if not strict_json_equal(normalized, expected):
        raise KrennLeafFreeError(
            "leaf-free saturation audit failed exact replay"
        )
    return normalized


def _singular_polynomial_text(
    polynomial: SparseChartPolynomial,
    characteristic: int,
) -> str:
    pieces = []
    for raw_coefficient, monomial in polynomial.terms:
        coefficient = (
            raw_coefficient % characteristic
            if characteristic else raw_coefficient
        )
        factors = "*".join(f"x{variable}" for variable in monomial)
        if not factors:
            piece = str(coefficient)
        elif coefficient == 1:
            piece = factors
        elif coefficient == -1:
            piece = f"-{factors}"
        else:
            piece = f"{coefficient}*{factors}"
        pieces.append(piece)
    if characteristic:
        return "+".join(pieces)
    result = pieces[0]
    for piece in pieces[1:]:
        result += piece if piece.startswith("-") else f"+{piece}"
    return result


def _system_from_key(key: Sequence[int]) -> LeafFreeChartSystem:
    key = tuple(map(int, key))
    if len(key) == 1:
        return derivative_leaf_free_system(key[0])
    return repair_leaf_free_system(key)


def singular_leaf_free_saturation_script(
    key: Sequence[int],
    *,
    characteristic: int = 31,
    initial_stage_only: bool = False,
) -> str:
    """Export deterministic sequential saturation to Singular.

    With ``initial_stage_only=True``, only the first factor is attempted.
    That form is useful as a bounded feasibility probe.  A completed later
    stage must start from a retained exact standard-basis checkpoint.
    """

    try:
        characteristic = validate_singular_characteristic(characteristic)
    except KrennLocalizedChartError as error:
        raise KrennLeafFreeError(
            "Singular characteristic must be zero or prime"
        ) from error
    system = _system_from_key(key)
    names = ",".join(
        f"x{variable}" for variable in range(system.variable_count)
    )
    generators = ",\n".join(
        _singular_polynomial_text(polynomial, characteristic)
        for polynomial in system.generators
    )
    stage_count = 1 if initial_stage_only else len(
        system.saturation_factors
    )
    key_text = "_".join(map(str, system.key))
    marker = f"KRENN_LEAF_FREE_{key_text}"
    lines = [
        "// Exact leaf-free sequential-saturation chart.",
        f"// schema={LEAF_FREE_SYSTEM_SCHEMA}",
        f"// key={system.key}",
        'LIB "elim.lib";',
        f"ring r={characteristic},({names}),dp;",
        f"ideal S0={generators};",
        f'print("{marker}_PARSE_OK");',
        f'print("characteristic={characteristic}");',
        f'print("variables={system.variable_count}");',
        f'print("generators={len(system.generators)}");',
        "int total_started=timer;",
    ]
    for index, (label, factor) in enumerate(zip(
        system.saturation_factor_labels,
        system.saturation_factors,
        strict=True,
    )):
        if index >= stage_count:
            break
        factor_text = _singular_polynomial_text(
            factor, characteristic
        )
        lines.extend((
            f"poly F{index}={factor_text};",
            f"ideal H{index}=F{index};",
            f"int started{index}=timer;",
            f"ideal S{index + 1}=sat(S{index},H{index});",
            f"int elapsed{index}=timer-started{index};",
            (
                f'print("KRENN_LEAF_FREE_STAGE_DONE '
                f'index={index} factor={label}");'
            ),
            (
                f'print("stage_{index}_timer_ticks="'
                f'+string(elapsed{index}));'
            ),
            (
                f'print("stage_{index}_basis_size="'
                f'+string(size(S{index + 1})));'
            ),
        ))
    final_index = stage_count
    lines.extend((
        f'print("stages_completed={stage_count}");',
        "int total_elapsed=timer-total_started;",
        f"poly unit_remainder=reduce(1,S{final_index});",
        "int is_unit=0;",
        "if (unit_remainder==0) { is_unit=1; }",
        f'print("{marker}_SATURATION_DONE");',
        f'print("basis_size="+string(size(S{final_index})));',
        'print("timer_ticks="+string(total_elapsed));',
        'print("unit_ideal="+string(is_unit));',
        "exit;",
        "",
    ))
    return "\n".join(lines)


__all__ = [
    "KrennLeafFreeError",
    "LEAF_FREE_AUDIT_SCHEMA",
    "LEAF_FREE_SYSTEM_SCHEMA",
    "LeafFreeChartSystem",
    "derivative_leaf_free_system",
    "leaf_free_saturation_audit",
    "repair_leaf_free_system",
    "singular_leaf_free_saturation_script",
    "verify_leaf_free_saturation_audit",
]
