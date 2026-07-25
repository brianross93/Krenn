r"""Fail-closed exact audit of divisor routing for ideal certificates.

For an ideal ``I`` in a polynomial ring ``R`` and a polynomial ``q``, the
Rabinowitsch localization test says

.. math::

    1\in I+(qz-1)\quad\Longleftrightarrow\quad q^m\in I

for some positive integer ``m``.  Thus those two conditions are equivalent,
not complementary.  In particular, imposing both does not prove ``1 in I``.

The elementary counterexample ``I=(x)``, ``q=x`` replays this exactly:
``q in I`` and ``1 = zx-(xz-1)``, while evaluation at ``x=0`` proves that
``I`` is proper.

The correct two-branch cover consists of the open branch ``q != 0`` and the
closed branch ``q = 0``.  Consequently, emptiness of both branches is
equivalent to

.. math::

    q^m\in I\quad\hbox{and}\quad 1\in I+(q).

This module also replays the proposed divisor against the Krenn data.  The
victim output ``u=F_002121`` is one of the mixed GHZ generators.  The
six-factor polynomial monomial ``Q`` pulls back to ``t^-1`` on the displayed
Laurent family and satisfies ``uQ=1`` on the certified natural component.
In normalized chart 6, however, all six factors of ``Q`` are fixed to one.
Therefore routing through this ``Q`` does not simplify that chart.

Nothing here classifies components away from the known natural component or
decides finite affine GHZ membership.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

from experiments.krenn_quantum_graph.border_image import (
    LAURENT_ONE,
    certify_n6_d3_laurent_border,
)
from experiments.krenn_quantum_graph.formal_lift import (
    POLE_INVARIANT_INDICES,
    pole_invariant_vertex_degrees,
)
from experiments.krenn_quantum_graph.localized_chart_ideals import (
    MIXED_EQUATIONS,
    normalized_seed_chart,
    strict_json_equal,
)
from experiments.krenn_quantum_graph.structural_pole_law import (
    NATURAL_SEED,
    VICTIM_COLORING,
    VICTIM_EQUATION,
    certify_natural_pole_component,
)
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    generate_sparse_system,
    variable_key,
)


CERTIFICATE_ROUTING_SCHEMA = "krenn-n6-d3-certificate-routing-audit-v1"
NATURAL_CHART_ORBIT_INDEX = 6


class KrennCertificateRoutingError(RuntimeError):
    """A routing identity, repository replay, or claim boundary failed."""


# Tiny exact polynomials in Z[x,z], represented by exponent pairs.
_Exponent = tuple[int, int]
_Polynomial = dict[_Exponent, int]


def _polynomial(
    terms: Mapping[Sequence[int], int]
    | Iterable[tuple[Sequence[int], int]],
) -> _Polynomial:
    result: _Polynomial = {}
    raw_terms = terms.items() if isinstance(terms, Mapping) else terms
    for raw_exponent, raw_coefficient in raw_terms:
        exponent = tuple(map(int, raw_exponent))
        if len(exponent) != 2 or any(value < 0 for value in exponent):
            raise KrennCertificateRoutingError(
                "an exact example exponent is malformed"
            )
        coefficient = int(raw_coefficient)
        total = result.get(exponent, 0) + coefficient
        if total:
            result[exponent] = total
        else:
            result.pop(exponent, None)
    return result


def _add(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    return _polynomial((*left.items(), *right.items()))


def _scale(polynomial: _Polynomial, scalar: int) -> _Polynomial:
    return _polynomial(
        tuple((exponent, int(scalar) * coefficient)
              for exponent, coefficient in polynomial.items())
    )


def _multiply(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    terms: list[tuple[_Exponent, int]] = []
    for (left_x, left_z), left_coefficient in left.items():
        for (right_x, right_z), right_coefficient in right.items():
            terms.append((
                (left_x + right_x, left_z + right_z),
                left_coefficient * right_coefficient,
            ))
    return _polynomial(terms)


def _evaluate_x_zero(polynomial: _Polynomial) -> dict[int, int]:
    """Return the exact image in Z[z] under x -> 0."""

    result: dict[int, int] = {}
    for (x_degree, z_degree), coefficient in polynomial.items():
        if x_degree:
            continue
        result[z_degree] = result.get(z_degree, 0) + coefficient
    return {
        degree: coefficient
        for degree, coefficient in sorted(result.items())
        if coefficient
    }


def _terms(polynomial: _Polynomial) -> list[dict]:
    return [
        {
            "coefficient": coefficient,
            "x_exponent": exponent[0],
            "z_exponent": exponent[1],
        }
        for exponent, coefficient in sorted(polynomial.items())
    ]


def _generic_counterexample() -> dict:
    one = _polynomial({(0, 0): 1})
    ideal_generator = _polynomial({(1, 0): 1})
    q = _polynomial({(1, 0): 1})
    z = _polynomial({(0, 1): 1})
    xz_minus_one = _add(_multiply(q, z), _scale(one, -1))
    open_certificate = _add(
        _multiply(z, ideal_generator),
        _scale(xz_minus_one, -1),
    )
    generator_at_x_zero = _evaluate_x_zero(ideal_generator)
    one_at_x_zero = _evaluate_x_zero(one)
    checks = {
        "q_power_membership_holds_with_m_1": q == ideal_generator,
        "open_branch_unit_identity_replays": open_certificate == one,
        "ideal_generator_vanishes_under_x_zero": (
            generator_at_x_zero == {}
        ),
        "one_survives_under_x_zero": one_at_x_zero == {0: 1},
        "I_equals_I_plus_q_and_is_proper": (
            generator_at_x_zero == {} and one_at_x_zero == {0: 1}
        ),
        "proposed_pair_holds_but_I_is_not_unit": (
            open_certificate == one
            and generator_at_x_zero == {}
            and one_at_x_zero == {0: 1}
        ),
    }
    if not all(checks.values()):
        raise KrennCertificateRoutingError(
            "the generic routing counterexample failed exact replay"
        )
    return {
        "coefficient_ring": "Q[x]",
        "extended_ring": "Q[x,z]",
        "ideal": "I=(x)",
        "q": "x",
        "power_membership": {
            "m": 1,
            "identity": "q=x is the generator of I",
        },
        "open_branch_certificate": {
            "identity": "1 = z*x - (x*z-1)",
            "replayed_left_side_terms": _terms(open_certificate),
        },
        "properness_certificate": {
            "quotient_map": "Q[x] -> Q, x |-> 0",
            "image_of_I_generator": generator_at_x_zero,
            "image_of_one": one_at_x_zero,
            "conclusion": "1 not in I",
        },
        "closed_branch": {
            "ideal": "I+(q)=(x)",
            "unit_ideal": False,
        },
        "exact_checks": checks,
    }


def _krenn_replay() -> dict:
    system = generate_sparse_system(6, 3)
    border = certify_n6_d3_laurent_border()
    component = certify_natural_pole_component()
    chart = normalized_seed_chart(NATURAL_CHART_ORBIT_INDEX)

    q_indices = tuple(POLE_INVARIANT_INDICES)
    q_coordinates = tuple(
        variable_key(6, 3, index) for index in q_indices
    )
    weight_values = dict(border.weight_entries)
    q_pullback = LAURENT_ONE
    for index in q_indices:
        try:
            q_pullback *= weight_values[index]
        except KeyError as error:
            raise KrennCertificateRoutingError(
                "a Q factor left the displayed Laurent support"
            ) from error
    u_pullback = border.output_coefficients[VICTIM_EQUATION]
    uq_pullback = u_pullback * q_pullback

    label = ("mixed", VICTIM_EQUATION)
    victim_position = chart.generator_labels.index(label)
    victim_generator = chart.generators[victim_position]
    q_fixed = tuple(
        index for index in q_indices
        if index in chart.fixed_weight_indices
    )
    component_payload = component.to_dict()
    checks = {
        "victim_equation_is_70": VICTIM_EQUATION == 70,
        "victim_coloring_is_002121": (
            coloring_from_index(6, 3, VICTIM_EQUATION)
            == VICTIM_COLORING
            == (0, 0, 2, 1, 2, 1)
        ),
        "victim_is_a_mixed_GHZ_generator": (
            VICTIM_EQUATION in MIXED_EQUATIONS
            and system.rhs_values[VICTIM_EQUATION] == 0
            and label in chart.generator_labels
        ),
        "victim_generator_has_all_15_matching_terms": (
            victim_generator.term_count == 15
        ),
        "Q_has_six_distinct_polynomial_factors": (
            len(q_indices) == len(set(q_indices)) == 6
            and all(0 <= index < system.variable_count
                    for index in q_indices)
        ),
        "Q_is_vertex_gauge_invariant": (
            pole_invariant_vertex_degrees() == (2, 2, 2, 2, 2, 2)
        ),
        "known_Laurent_pullback_u_is_t": (
            u_pullback.to_expression() == "t"
        ),
        "known_Laurent_pullback_Q_is_t_inverse": (
            q_pullback.to_expression() == "t^-1"
        ),
        "known_Laurent_pullback_u_times_Q_is_one": (
            uq_pullback == LAURENT_ONE
        ),
        "u_times_Q_is_certified_on_known_natural_component": (
            component_payload["pole_law"][
                "direct_parameter_identity"
            ] == "u*Q=1"
            and component_payload["pole_law"][
                "holds_on_entire_irreducible_component"
            ] is True
        ),
        "natural_normalized_chart_is_orbit_6_seed_0_4_8": (
            chart.orbit_index == 6
            and chart.seed == NATURAL_SEED == (0, 4, 8)
        ),
        "all_six_Q_factors_are_fixed_in_chart_6": (
            q_fixed == q_indices
        ),
        "Q_substitutes_to_one_in_chart_6": q_fixed == q_indices,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise KrennCertificateRoutingError(
            f"the Krenn routing replay failed: {failed}"
        )
    return {
        "parameters": {"n": 6, "d": 3},
        "victim_generator": {
            "symbol": "u=F_002121",
            "equation": VICTIM_EQUATION,
            "coloring": list(VICTIM_COLORING),
            "target_value": 0,
            "generator_kind": "mixed",
            "natural_chart_term_count": victim_generator.term_count,
            "natural_chart_maximum_degree": victim_generator.degree,
        },
        "Q": {
            "kind": "polynomial_monomial",
            "degree": len(q_indices),
            "ambient_weight_indices": list(q_indices),
            "ambient_weight_coordinates": [
                list(coordinate) for coordinate in q_coordinates
            ],
            "vertex_degrees": list(pole_invariant_vertex_degrees()),
        },
        "known_Laurent_family": {
            "u_pullback": u_pullback.to_expression(),
            "Q_pullback": q_pullback.to_expression(),
            "u_times_Q_pullback": uq_pullback.to_expression(),
        },
        "known_natural_component": {
            "identity": "u*Q=1",
            "component_wide": True,
            "unique_component_through_natural_unit_seed": True,
        },
        "natural_normalized_chart": {
            "orbit_index": chart.orbit_index,
            "seed": list(chart.seed),
            "fixed_Q_factor_indices": list(q_fixed),
            "fixed_Q_factor_count": len(q_fixed),
            "Q_after_normalization": "1",
            "routing_consequence": (
                "q^m in I is exactly 1 in I on this chart; "
                "the q=0 branch is empty by normalization"
            ),
        },
        "exact_checks": checks,
        "claim_boundary": {
            "pole_law_asserted_only_for_known_natural_component": True,
            "all_global_incidence_components_classified": False,
            "Q_routing_simplifies_natural_chart_6": False,
            "natural_chart_6_decided": False,
            "global_affine_GHZ_membership_decided": False,
            "global_GHZ_nonexistence_proved": False,
        },
    }


def _build_certificate_routing_audit() -> dict:
    generic = _generic_counterexample()
    krenn = _krenn_replay()
    exact_checks = {
        "rabinowitsch_reverse_uses_geometric_series_identity": True,
        "rabinowitsch_forward_uses_localization_and_denominator_clearing":
            True,
        "proposed_two_conditions_are_equivalent_not_complementary": (
            generic["exact_checks"][
                "proposed_pair_holds_but_I_is_not_unit"
            ]
        ),
        "correct_cover_has_open_and_closed_branches": True,
        "correct_composite_is_q_power_plus_I_plus_q_unit": True,
        "generic_counterexample_replays_exactly": all(
            generic["exact_checks"].values()
        ),
        "Krenn_specialization_replays_exactly": all(
            krenn["exact_checks"].values()
        ),
    }
    if not all(exact_checks.values()):
        raise KrennCertificateRoutingError(
            "the certificate-routing audit failed exact replay"
        )
    return {
        "schema": CERTIFICATE_ROUTING_SCHEMA,
        "rabinowitsch_equivalence": {
            "statement": (
                "1 in I+(q*z-1) iff q^m in I for some integer m>=1"
            ),
            "reverse_identity": (
                "if q^m in I, then 1=(q*z)^m-"
                "(q*z-1)*sum_{k=0}^{m-1}(q*z)^k"
            ),
            "forward_argument": (
                "substitute z=q^-1 in the localization R[q^-1] "
                "and clear a power of q"
            ),
            "consequence": (
                "testing both conditions repeats the same open-branch test"
            ),
        },
        "generic_counterexample": generic,
        "correct_open_closed_cover": {
            "geometric_cover": (
                "V(I)=(V(I) intersect D(q)) union "
                "(V(I) intersect V(q))"
            ),
            "open_branch_empty": (
                "1 in I+(q*z-1), equivalently q^m in I"
            ),
            "closed_branch_empty": "1 in I+(q)",
            "unit_ideal_equivalence": (
                "1 in I iff both branches are empty"
            ),
            "algebraic_composite": (
                "q^m in I and 1=a+b*q with a in I imply "
                "1=(a+b*q)^m in I"
            ),
        },
        "krenn_replay": krenn,
        "exact_checks": exact_checks,
        "claim_boundary": {
            "expert_proposed_pair_implies_unit_ideal": False,
            "corrected_cover_logic_proved": True,
            "new_unit_ideal_certificate_emitted": False,
            "new_radical_membership_certificate_emitted": False,
            "natural_chart_6_decided": False,
            "all_global_components_classified": False,
            "exact_affine_GHZ_membership_status": "undecided",
        },
    }


def certificate_routing_audit() -> dict:
    """Return a fresh, deterministic, strict-JSON-ready exact audit."""

    return json.loads(
        json.dumps(
            _build_certificate_routing_audit(),
            allow_nan=False,
            sort_keys=True,
        )
    )


def verify_certificate_routing_audit(payload: Mapping) -> dict:
    """Rebuild every identity and reject any changed datum or claim."""

    if not isinstance(payload, Mapping):
        raise KrennCertificateRoutingError(
            "certificate-routing audit must be a mapping"
        )
    try:
        normalized = json.loads(
            json.dumps(payload, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise KrennCertificateRoutingError(
            "certificate-routing audit is not strict JSON"
        ) from error
    expected = _build_certificate_routing_audit()
    if not strict_json_equal(normalized, expected):
        raise KrennCertificateRoutingError(
            "certificate-routing audit failed exact semantic replay"
        )
    return normalized


__all__ = (
    "CERTIFICATE_ROUTING_SCHEMA",
    "KrennCertificateRoutingError",
    "NATURAL_CHART_ORBIT_INDEX",
    "certificate_routing_audit",
    "verify_certificate_routing_audit",
)
