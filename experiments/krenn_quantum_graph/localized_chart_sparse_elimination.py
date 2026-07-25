r"""Exact quotient-graded triangular forms of the sparse derivative slices.

For a selected derivative ``d`` of the natural defect,

    f = x*d + h,       d = 1,

the generator ``f`` may be replaced by ``x+h`` because

    x+h = f - x*(d-1).

This exposes a unit-coefficient pivot without performing the degree-six
substitution into every other equation.  The primitive character of ``d``
is killed by the gauge slice, leaving a split residual ``Z^8`` grading.
Every transformed generator is checked term by term in that quotient.

The grading and triangular rewrite are exact structure.  A bounded Gröbner
timeout or a finite-field result does not decide the characteristic-zero
ideal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.localized_chart_derivative import (
    DERIVATIVE_REPRESENTATIVES,
    KrennDerivativeChartError,
    NATURAL_DEFECT_EQUATION,
    NATURAL_ORBIT_INDEX,
    sparse_derivative_gauge_slice,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    KrennLocalizedChartError,
    SparseChartPolynomial,
    normalized_seed_chart,
    strict_json_equal,
    validate_singular_characteristic,
)
from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    generator_characters,
    ordered_seed_stabilizer,
    residual_torus_characters,
)


SPARSE_ELIMINATION_SCHEMA = (
    "krenn-n6-d3-sparse-derivative-triangular-system-v1"
)
SPARSE_ELIMINATION_AUDIT_SCHEMA = (
    "krenn-n6-d3-sparse-derivative-quotient-grading-v1"
)
EXPECTED_QUOTIENT_ROWS = {
    11: {
        "generator_blocks": 558,
        "generator_histogram": (
            (1, 415), (2, 126), (3, 8), (4, 8), (7, 1),
        ),
    },
    29: {
        "generator_blocks": 642,
        "generator_histogram": (
            (1, 567), (2, 66), (3, 8), (7, 1),
        ),
    },
}

Character = tuple[int, ...]


class KrennSparseEliminationError(RuntimeError):
    """A triangular rewrite, quotient grading, or export failed closed."""


def _add(
    target: dict[tuple[int, ...], int],
    monomial: Sequence[int],
    coefficient: int,
) -> None:
    key = tuple(sorted(map(int, monomial)))
    total = target.get(key, 0) + int(coefficient)
    if total:
        target[key] = total
    else:
        target.pop(key, None)


def _triangular_polynomial(
    derivative_ambient_weight: int,
) -> SparseChartPolynomial:
    chart = sparse_derivative_gauge_slice(derivative_ambient_weight)
    defect_position = chart.generator_labels.index(
        ("mixed", NATURAL_DEFECT_EQUATION)
    )
    selected = chart.derivative_original_local_variable
    result = {
        monomial: coefficient
        for coefficient, monomial in chart.generators[
            defect_position
        ].terms
    }
    for coefficient, monomial in chart.generators[-1].terms:
        _add(result, (*monomial, selected), -coefficient)
    return SparseChartPolynomial.from_mapping(result)


def _project_character(
    character: Sequence[int],
    derivative_character: Sequence[int],
) -> Character:
    """Project ``Z^9`` to a split quotient by the derivative character."""

    character = tuple(map(int, character))
    derivative_character = tuple(map(int, derivative_character))
    if (
        len(character) != 9
        or len(derivative_character) != 9
        or derivative_character[0] != -1
        or any(derivative_character[index] for index in range(1, 8))
        or abs(derivative_character[8]) != 1
    ):
        raise KrennSparseEliminationError(
            "the retained derivative quotient coordinates changed"
        )
    return (
        *character[1:8],
        character[8] + derivative_character[8] * character[0],
    )


def _monomial_character(
    monomial: Sequence[int],
    variable_characters: Sequence[Character],
) -> Character:
    return tuple(
        sum(
            variable_characters[variable][coordinate]
            for variable in monomial
        )
        for coordinate in range(8)
    )


def _character_digest(
    variable_characters: Sequence[Character],
    generator_characters_: Sequence[Character],
) -> str:
    digest = hashlib.sha256()
    for label, rows in (
        ("variables", variable_characters),
        ("generators", generator_characters_),
    ):
        digest.update(f"{label}\n".encode("ascii"))
        for row in rows:
            digest.update(",".join(map(str, row)).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class TriangularSparseDerivativeSystem:
    """One sparse ``d=1`` system with its defect pivot exposed."""

    derivative_ambient_weight: int
    selected_original_local_variable: int
    generator_labels: tuple[tuple[str, int], ...]
    generators: tuple[SparseChartPolynomial, ...]
    schema: str = SPARSE_ELIMINATION_SCHEMA

    def __post_init__(self) -> None:
        original = sparse_derivative_gauge_slice(
            self.derivative_ambient_weight
        )
        defect_position = original.generator_labels.index(
            ("mixed", NATURAL_DEFECT_EQUATION)
        )
        expected = list(original.generators)
        expected[defect_position] = _triangular_polynomial(
            self.derivative_ambient_weight
        )
        if (
            self.schema != SPARSE_ELIMINATION_SCHEMA
            or self.derivative_ambient_weight
            not in DERIVATIVE_REPRESENTATIVES
            or self.selected_original_local_variable
            != original.derivative_original_local_variable
            or self.generator_labels != original.generator_labels
            or self.generators != tuple(expected)
            or len(self.generators) != 730
            or self.variable_count != 129
            or self.term_count != 10_940
            or self.maximum_degree != 4
        ):
            raise KrennSparseEliminationError(
                "a triangular sparse derivative system failed replay"
            )
        selected = self.selected_original_local_variable
        triangular = self.generators[defect_position]
        selected_occurrences = tuple(
            (coefficient, monomial)
            for coefficient, monomial in triangular.terms
            if selected in monomial
        )
        if selected_occurrences != ((1, (selected,)),):
            raise KrennSparseEliminationError(
                "the transformed defect lacks its unique monic pivot"
            )

    @property
    def variable_count(self) -> int:
        return 129

    @property
    def term_count(self) -> int:
        return sum(polynomial.term_count for polynomial in self.generators)

    @property
    def maximum_degree(self) -> int:
        return max(polynomial.degree for polynomial in self.generators)

    @property
    def secondary_pivot_variable(self) -> int:
        gauge = self.generators[-1]
        unit_linear_variables = tuple(
            monomial[0]
            for coefficient, monomial in gauge.terms
            if len(monomial) == 1 and coefficient == 1
        )
        if len(unit_linear_variables) != 1:
            raise KrennSparseEliminationError(
                "the derivative gauge lacks its unique monic pivot"
            )
        pivot = unit_linear_variables[0]
        occurrences = tuple(
            (coefficient, monomial)
            for coefficient, monomial in gauge.terms
            if pivot in monomial
        )
        if occurrences != ((1, (pivot,)),):
            raise KrennSparseEliminationError(
                "the derivative gauge pivot occurs nonlinearly"
            )
        return pivot

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.schema}|{self.derivative_ambient_weight}|"
                f"{self.selected_original_local_variable}\n"
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
        return digest.hexdigest()


def triangular_sparse_derivative_system(
    derivative_ambient_weight: int,
) -> TriangularSparseDerivativeSystem:
    """Return the exact ideal-equivalent triangular generator system."""

    derivative_ambient_weight = int(derivative_ambient_weight)
    original = sparse_derivative_gauge_slice(
        derivative_ambient_weight
    )
    generators = list(original.generators)
    defect_position = original.generator_labels.index(
        ("mixed", NATURAL_DEFECT_EQUATION)
    )
    generators[defect_position] = _triangular_polynomial(
        derivative_ambient_weight
    )
    return TriangularSparseDerivativeSystem(
        derivative_ambient_weight=derivative_ambient_weight,
        selected_original_local_variable=(
            original.derivative_original_local_variable
        ),
        generator_labels=original.generator_labels,
        generators=tuple(generators),
    )


def sparse_derivative_quotient_grading(
    derivative_ambient_weight: int,
) -> dict:
    """Replay the split ``Z^8`` grading of one triangular slice."""

    system = triangular_sparse_derivative_system(
        derivative_ambient_weight
    )
    original = sparse_derivative_gauge_slice(
        derivative_ambient_weight
    )
    original_variable_characters = residual_torus_characters(
        NATURAL_ORBIT_INDEX
    )
    variable_characters = tuple(
        _project_character(
            character, original.derivative_character
        )
        for character in original_variable_characters
    )
    projected_generator_characters = []
    for polynomial in system.generators:
        term_characters = {
            _monomial_character(monomial, variable_characters)
            for _coefficient, monomial in polynomial.terms
        }
        if len(term_characters) != 1:
            raise KrennSparseEliminationError(
                "a triangular generator is not quotient homogeneous"
            )
        projected_generator_characters.append(
            next(iter(term_characters))
        )
    projected_generator_characters = tuple(
        projected_generator_characters
    )
    blocks = Counter(projected_generator_characters)
    histogram = tuple(sorted(Counter(blocks.values()).items()))
    variable_blocks = Counter(variable_characters)
    variable_histogram = tuple(
        sorted(Counter(variable_blocks.values()).items())
    )
    zero = (0,) * 8
    expected = EXPECTED_QUOTIENT_ROWS[
        int(derivative_ambient_weight)
    ]
    if (
        _project_character(
            original.derivative_character,
            original.derivative_character,
        ) != zero
        or len(variable_blocks) != 105
        or variable_histogram != ((1, 84), (2, 20), (5, 1))
        or variable_blocks[zero] != 5
        or len(blocks) != expected["generator_blocks"]
        or histogram != expected["generator_histogram"]
        or blocks[zero] != 7
    ):
        raise KrennSparseEliminationError(
            "the sparse derivative quotient census changed"
        )
    stabilizer = ordered_seed_stabilizer(NATURAL_ORBIT_INDEX)
    strict_fixers = tuple(
        index
        for index, permutation in enumerate(
            stabilizer.chart_variable_permutations
        )
        if permutation[system.selected_original_local_variable]
        == system.selected_original_local_variable
    )
    if len(strict_fixers) != 2:
        raise KrennSparseEliminationError(
            "the derivative representative stabilizer changed"
        )
    return {
        "derivative_ambient_weight": int(
            derivative_ambient_weight
        ),
        "derivative_character": list(
            original.derivative_character
        ),
        "quotient": {
            "ambient_rank": 9,
            "killed_rank": 1,
            "smith_diagonal": [1],
            "residual_rank": 8,
            "projection_formula": (
                "(a1,...,a7,a8+chi8*a0), where chi8 is the "
                "last derivative-character coordinate"
            ),
            "split_over_Z": True,
        },
        "variables": {
            "count": system.variable_count,
            "character_blocks": len(variable_blocks),
            "zero_character_variables": variable_blocks[zero],
            "block_multiplicity_histogram": [
                {"multiplicity": multiplicity, "blocks": count}
                for multiplicity, count in variable_histogram
            ],
        },
        "generators": {
            "count": len(system.generators),
            "terms": system.term_count,
            "maximum_degree": system.maximum_degree,
            "all_terms_quotient_homogeneous": True,
            "character_blocks": len(blocks),
            "zero_character_generators": blocks[zero],
            "block_multiplicity_histogram": [
                {"multiplicity": multiplicity, "blocks": count}
                for multiplicity, count in histogram
            ],
            "variable_and_generator_character_sha256":
                _character_digest(
                    variable_characters,
                    projected_generator_characters,
                ),
        },
        "triangular_rewrite": {
            "identity": "x+h = f-x*(d-1)",
            "selected_local_variable":
                system.selected_original_local_variable,
            "derivative_linear_pivot":
                system.secondary_pivot_variable,
            "unit_linear_pivot": True,
            "two_monic_pivots": True,
            "ideal_preserved_exactly": True,
            "system_sha256": system.fingerprint(),
        },
        "strict_derivative_stabilizer": {
            "order": len(strict_fixers),
            "symmetry_related_weights_equated": False,
        },
        "claim_boundary": {
            "unit_or_proper_status_decided": False,
            "finite_counterexample_found": False,
            "global_nonexistence_proved": False,
            "grading_alone_is_a_grobner_basis": False,
            "grading_alone_is_a_tropical_decomposition": False,
        },
    }


def sparse_elimination_audit() -> dict:
    """Return the exact quotient grading and triangular systems."""

    return json.loads(json.dumps({
        "schema": SPARSE_ELIMINATION_AUDIT_SCHEMA,
        "natural_seed": [0, 4, 8],
        "systems": [
            sparse_derivative_quotient_grading(ambient)
            for ambient in DERIVATIVE_REPRESENTATIVES
        ],
        "decision_rule": {
            "both_exact_Q_unit": "excludes the full natural chart",
            "either_exact_Q_proper": (
                "proves a finite point over the algebraic closure of Q"
            ),
            "finite_field_or_timeout": "reconnaissance only",
        },
        "claim_boundary": {
            "natural_chart_decided": False,
            "full_eight_chart_cover_decided": False,
        },
    }))


def verify_sparse_elimination_audit(payload: Mapping) -> dict:
    """Recompute every exact datum and reject changed claims."""

    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KrennSparseEliminationError(
            "sparse elimination audit is not strict JSON"
        ) from error
    expected = sparse_elimination_audit()
    if not strict_json_equal(normalized, expected):
        raise KrennSparseEliminationError(
            "sparse elimination audit failed exact replay"
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


def singular_triangular_derivative_script(
    derivative_ambient_weight: int,
    *,
    characteristic: int = 31,
    order: str = "dp",
    algorithm: str = "std",
) -> str:
    """Export a quotient-graded triangular probe to Singular."""

    try:
        characteristic = validate_singular_characteristic(characteristic)
    except KrennLocalizedChartError as error:
        raise KrennSparseEliminationError(
            "Singular characteristic must be zero or prime"
        ) from error
    if order not in (
        "dp",
        "selected-elimination",
        "two-pivot-elimination",
    ):
        raise KrennSparseEliminationError(
            "Singular order must be dp, selected-elimination, "
            "or two-pivot-elimination"
        )
    if algorithm not in ("std", "slimgb"):
        raise KrennSparseEliminationError(
            "Singular algorithm must be std or slimgb"
        )
    system = triangular_sparse_derivative_system(
        derivative_ambient_weight
    )
    selected = system.selected_original_local_variable
    if order == "dp":
        variable_order = tuple(range(system.variable_count))
        order_text = "dp"
    elif order == "selected-elimination":
        variable_order = (
            selected,
            *(
                variable
                for variable in range(system.variable_count)
                if variable != selected
            ),
        )
        order_text = "(lp(1),dp(128))"
    else:
        secondary = system.secondary_pivot_variable
        variable_order = (
            selected,
            secondary,
            *(
                variable
                for variable in range(system.variable_count)
                if variable not in (selected, secondary)
            ),
        )
        order_text = "(lp(2),dp(127))"
    names = ",".join(f"x{variable}" for variable in variable_order)
    generators = ",\n".join(
        _singular_polynomial_text(polynomial, characteristic)
        for polynomial in system.generators
    )
    marker = (
        "KRENN_SPARSE_TRIANGULAR_"
        f"{order.upper().replace('-', '_')}"
    )
    return "\n".join((
        "// Exact quotient-graded triangular derivative slice.",
        f"// schema={SPARSE_ELIMINATION_SCHEMA}",
        (
            "// derivative_ambient_weight="
            f"{system.derivative_ambient_weight}"
        ),
        f"// formulation={order}",
        f"ring r={characteristic},({names}),{order_text};",
        f"ideal I={generators};",
        f'print("{marker}_PARSE_OK");',
        (
            'print("derivative_ambient_weight='
            f'{system.derivative_ambient_weight}");'
        ),
        f'print("characteristic={characteristic}");',
        f'print("algorithm={algorithm}");',
        f'print("formulation={order}");',
        'print("variables="+string(nvars(basering)));',
        'print("generators="+string(size(I)));',
        "int started=timer;",
        f"ideal G={algorithm}(I);",
        "int elapsed=timer-started;",
        "poly unit_remainder=reduce(1,G);",
        "int is_unit=0;",
        "if (unit_remainder==0) { is_unit=1; }",
        f'print("{marker}_GROEBNER_DONE");',
        'print("timer_ticks="+string(elapsed));',
        'print("basis_size="+string(size(G)));',
        'print("unit_ideal="+string(is_unit));',
        "exit;",
        "",
    ))


__all__ = [
    "KrennSparseEliminationError",
    "SPARSE_ELIMINATION_AUDIT_SCHEMA",
    "SPARSE_ELIMINATION_SCHEMA",
    "TriangularSparseDerivativeSystem",
    "singular_triangular_derivative_script",
    "sparse_derivative_quotient_grading",
    "sparse_elimination_audit",
    "triangular_sparse_derivative_system",
    "verify_sparse_elimination_audit",
]
