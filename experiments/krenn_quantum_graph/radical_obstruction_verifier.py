"""Independent stdlib-only verifier for the support-row obstruction.

This verifier intentionally imports none of the Krenn producer modules.  It
rebuilds the 15 perfect matchings by filtering edge triples, constructs the
120 one-color terms in ``supp(F^2)``, enumerates the 663
``S_6 x S_3`` orbits by direct group actions, replays every retained
15-term source column, and recomputes the 31 by 31 determinant with exact
fractions.
"""

from __future__ import annotations

from array import array
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from math import comb
from typing import Mapping, Sequence


N = 6
D = 3
GROUP_ORDER = 4_320
EDGE_COUNT = 15
SUPPORT_GRAPH_COUNT = 120
SUPPORT_STATE_COUNT = SUPPORT_GRAPH_COUNT**D
SENTINEL = 0xFFFFFFFF


class KrennRadicalIndependentVerificationError(ValueError):
    """The standalone support-row certificate failed exact replay."""


EDGES = tuple(combinations(range(N), 2))
EDGE_INDEX = {edge: index for index, edge in enumerate(EDGES)}


def _variable_index(i: int, j: int, a: int, b: int) -> int:
    i, j, a, b = map(int, (i, j, a, b))
    if i > j:
        i, j, a, b = j, i, b, a
    if not (
        0 <= i < j < N
        and 0 <= a < D
        and 0 <= b < D
    ):
        raise KrennRadicalIndependentVerificationError(
            "invalid variable key"
        )
    edge_index = i * (2 * N - i - 1) // 2 + (j - i - 1)
    return (edge_index * D + a) * D + b


def _variable_key(index: int) -> tuple[int, int, int, int]:
    index = int(index)
    if index not in range(comb(N, 2) * D * D):
        raise KrennRadicalIndependentVerificationError(
            "variable index is out of range"
        )
    edge_index, colors = divmod(index, D * D)
    a, b = divmod(colors, D)
    i, j = EDGES[edge_index]
    return i, j, a, b


def _filtered_perfect_matchings() -> tuple[tuple[int, int, int], ...]:
    result = []
    for edge_indices in combinations(range(len(EDGES)), N // 2):
        vertices = [
            vertex
            for edge_index in edge_indices
            for vertex in EDGES[edge_index]
        ]
        if len(set(vertices)) == N:
            result.append(edge_indices)
    matchings = tuple(result)
    if len(matchings) != 15:
        raise KrennRadicalIndependentVerificationError(
            "independent K6 matching census changed"
        )
    return matchings


MATCHINGS = _filtered_perfect_matchings()


def _squared_hafnian_support() -> tuple[tuple[int, ...], ...]:
    result = set()
    for left_index, left in enumerate(MATCHINGS):
        for right in MATCHINGS[left_index:]:
            exponents = [0] * EDGE_COUNT
            for edge in (*left, *right):
                exponents[edge] += 1
            result.add(tuple(exponents))
    support = tuple(sorted(result))
    if len(support) != SUPPORT_GRAPH_COUNT:
        raise KrennRadicalIndependentVerificationError(
            "independent squared-hafnian support changed"
        )
    return support


SUPPORT = _squared_hafnian_support()
SUPPORT_INDEX = {graph: index for index, graph in enumerate(SUPPORT)}


def _encode_state(graph_indices: Sequence[int]) -> int:
    left, middle, right = tuple(map(int, graph_indices))
    if any(
        index not in range(SUPPORT_GRAPH_COUNT)
        for index in (left, middle, right)
    ):
        raise KrennRadicalIndependentVerificationError(
            "support graph index is out of range"
        )
    return (
        left * SUPPORT_GRAPH_COUNT + middle
    ) * SUPPORT_GRAPH_COUNT + right


def _decode_state(state: int) -> tuple[int, int, int]:
    state = int(state)
    if state not in range(SUPPORT_STATE_COUNT):
        raise KrennRadicalIndependentVerificationError(
            "support state is out of range"
        )
    left, remainder = divmod(
        state, SUPPORT_GRAPH_COUNT * SUPPORT_GRAPH_COUNT
    )
    middle, right = divmod(remainder, SUPPORT_GRAPH_COUNT)
    return left, middle, right


def _transport_graph(
    graph: Sequence[int],
    vertex_permutation: Sequence[int],
) -> tuple[int, ...]:
    transported = [0] * EDGE_COUNT
    for edge_index, exponent in enumerate(graph):
        left, right = EDGES[edge_index]
        image = tuple(sorted((
            int(vertex_permutation[left]),
            int(vertex_permutation[right]),
        )))
        transported[EDGE_INDEX[image]] = int(exponent)
    return tuple(transported)


def _vertex_actions() -> tuple[tuple[int, ...], ...]:
    actions = []
    for vertex_permutation in permutations(range(N)):
        actions.append(tuple(
            SUPPORT_INDEX[
                _transport_graph(graph, vertex_permutation)
            ]
            for graph in SUPPORT
        ))
    return tuple(actions)


VERTEX_ACTIONS = _vertex_actions()
COLOR_ACTIONS = tuple(permutations(range(D)))


def _direct_support_orbits() -> tuple[
    tuple[tuple[int, int, int], ...],
    array,
]:
    state_to_representative = array(
        "I", [SENTINEL]
    ) * SUPPORT_STATE_COUNT
    records = []
    for seed in range(SUPPORT_STATE_COUNT):
        if state_to_representative[seed] != SENTINEL:
            continue
        graph_indices = _decode_state(seed)
        orbit = set()
        for action in VERTEX_ACTIONS:
            transported = tuple(
                action[index] for index in graph_indices
            )
            for color_action in COLOR_ACTIONS:
                orbit.add(_encode_state(tuple(
                    transported[color_action[color]]
                    for color in range(D)
                )))
        representative = min(orbit)
        if representative != seed or GROUP_ORDER % len(orbit):
            raise KrennRadicalIndependentVerificationError(
                "direct support orbit is malformed"
            )
        for state in orbit:
            state_to_representative[state] = representative
        coefficient = 1
        for graph_index in graph_indices:
            graph = SUPPORT[graph_index]
            coefficient *= (
                1 if sum(value == 2 for value in graph) == 3 else 2
            )
        records.append((representative, len(orbit), coefficient))
    result = tuple(records)
    if (
        len(result) != 663
        or sum(size for _state, size, _coefficient in result)
        != SUPPORT_STATE_COUNT
        or any(
            representative == SENTINEL
            for representative in state_to_representative
        )
    ):
        raise KrennRadicalIndependentVerificationError(
            "direct support-orbit partition changed"
        )
    return result, state_to_representative


def _rainbow_witness(
    graph_indices: Sequence[int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    graphs = tuple(SUPPORT[int(index)] for index in graph_indices)
    for matching in MATCHINGS:
        for edge_positions in permutations(range(D)):
            if all(
                graphs[color][matching[edge_positions[color]]] > 0
                for color in range(D)
            ):
                return matching, edge_positions
    return None


def _monomial_support_state(
    monomial: Sequence[int],
) -> int | None:
    graphs = [[0] * EDGE_COUNT for _ in range(D)]
    for variable in monomial:
        left, right, left_color, right_color = _variable_key(variable)
        if left_color != right_color:
            return None
        graphs[left_color][
            EDGE_INDEX[tuple(sorted((left, right)))]
        ] += 1
    graph_indices = []
    for graph in graphs:
        graph_index = SUPPORT_INDEX.get(tuple(graph))
        if graph_index is None:
            return None
        graph_indices.append(graph_index)
    return _encode_state(graph_indices)


def _equation_terms(
    coloring: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    coloring = tuple(map(int, coloring))
    if len(coloring) != N or any(
        color not in range(D) for color in coloring
    ):
        raise KrennRadicalIndependentVerificationError(
            "coloring is malformed"
        )
    return tuple(
        tuple(
            _variable_index(
                left,
                right,
                coloring[left],
                coloring[right],
            )
            for left, right in (
                EDGES[edge_index] for edge_index in matching
            )
        )
        for matching in MATCHINGS
    )


def _rainbow_column(
    representative_state: int,
    witness: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
    ],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    graphs = [
        list(SUPPORT[index])
        for index in _decode_state(representative_state)
    ]
    matching, edge_positions = witness
    coloring = [None] * N
    for color, position in enumerate(edge_positions):
        edge_index = matching[position]
        left, right = EDGES[edge_index]
        coloring[left] = coloring[right] = color
        graphs[color][edge_index] -= 1
    multiplier = []
    for color, graph in enumerate(graphs):
        for edge_index, exponent in enumerate(graph):
            left, right = EDGES[edge_index]
            multiplier.extend(
                [_variable_index(
                    left, right, color, color
                )] * exponent
            )
    return tuple(coloring), tuple(sorted(multiplier))


def _exact_determinant(matrix: Sequence[Sequence[int]]) -> Fraction:
    rows = [
        [Fraction(int(value)) for value in row]
        for row in matrix
    ]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise KrennRadicalIndependentVerificationError(
            "certificate matrix is not square"
        )
    determinant = Fraction(1)
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant *= -1
        pivot_value = rows[column][column]
        determinant *= pivot_value
        rows[column] = [
            value / pivot_value for value in rows[column]
        ]
        for row in range(column + 1, size):
            if not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                left - factor * right
                for left, right in zip(
                    rows[row], rows[column], strict=True
                )
            ]
    return determinant


def _contraction_integer_jacobian(
    contraction_rank: int,
) -> tuple[tuple[int, ...], ...]:
    if contraction_rank not in (1, 2, 3):
        raise KrennRadicalIndependentVerificationError(
            "invalid contraction rank"
        )
    values = tuple(
        4 + 16 * index + 30 * index**2 + 15 * index**3
        for index in range(comb(N, 2) * D * D)
    )
    rows = []
    for output_coloring in product(range(D), repeat=4):
        row = [0] * len(values)
        for contracted_color in range(contraction_rank):
            coloring = (
                contracted_color,
                contracted_color,
                *output_coloring,
            )
            for monomial in _equation_terms(coloring):
                for position, variable in enumerate(monomial):
                    term = 1
                    for other_position, other in enumerate(monomial):
                        if other_position != position:
                            term *= values[other]
                    row[variable] += term
        rows.append(tuple(row))
    return tuple(rows)


def _integer_matrix_fingerprint(
    matrix: Sequence[Sequence[int]],
) -> str:
    encoded = json.dumps(
        {
            "shape": [len(matrix), len(matrix[0])],
            "entries": [
                [row_index, column_index, int(value)]
                for row_index, row in enumerate(matrix)
                for column_index, value in enumerate(row)
                if value
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _rank_and_minor_determinant_mod(
    matrix: Sequence[Sequence[int]],
    prime: int,
) -> tuple[int, int]:
    rows = [
        [int(value) % prime for value in row] for row in matrix
    ]
    original = tuple(tuple(row) for row in rows)
    row_ids = list(range(len(rows)))
    pivot_columns = []
    rank = 0
    for column in range(len(rows[0])):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        row_ids[rank], row_ids[pivot] = (
            row_ids[pivot],
            row_ids[rank],
        )
        inverse = pow(rows[rank][column], -1, prime)
        rows[rank] = [
            value * inverse % prime for value in rows[rank]
        ]
        for other in range(rank + 1, len(rows)):
            factor = rows[other][column]
            if factor:
                rows[other] = [
                    (left - factor * right) % prime
                    for left, right in zip(
                        rows[other], rows[rank], strict=True
                    )
                ]
        pivot_columns.append(column)
        rank += 1
        if rank == len(rows):
            break
    minor = [
        [original[row][column] for column in pivot_columns]
        for row in row_ids[:rank]
    ]
    determinant = 1
    for column in range(rank):
        pivot = next(
            (
                row
                for row in range(column, rank)
                if minor[row][column]
            ),
            None,
        )
        if pivot is None:
            return rank, 0
        if pivot != column:
            minor[column], minor[pivot] = (
                minor[pivot],
                minor[column],
            )
            determinant = -determinant
        pivot_value = minor[column][column]
        determinant = determinant * pivot_value % prime
        inverse = pow(pivot_value, -1, prime)
        for row in range(column + 1, rank):
            factor = minor[row][column] * inverse % prime
            if factor:
                for entry in range(column, rank):
                    minor[row][entry] = (
                        minor[row][entry]
                        - factor * minor[column][entry]
                    ) % prime
    return rank, determinant % prime


def _verify_contracted_ghz_witness(record: Mapping) -> None:
    _strict_keys(
        record,
        {
            "matching_for_color_0",
            "matching_for_color_1",
            "matching_for_color_2",
            "construction",
            "symbolically_replayed",
        },
        "contracted GHZ witness",
    )
    expected_matchings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    if tuple(
        tuple(tuple(edge) for edge in record[
            f"matching_for_color_{color}"
        ])
        for color in range(D)
    ) != expected_matchings:
        raise KrennRadicalIndependentVerificationError(
            "the contracted GHZ witness changed"
        )
    if (
        record["construction"] != (
            "For color a, set one edge on its displayed matching "
            "to K_aa and the other to 1; set every undisplayed "
            "variable to 0."
        )
        or record["symbolically_replayed"] is not True
    ):
        raise KrennRadicalIndependentVerificationError(
            "the contracted GHZ witness metadata changed"
        )
    edges = tuple(combinations(range(4), 2))
    matchings = tuple(
        tuple(edges[index] for index in indices)
        for indices in combinations(range(len(edges)), 2)
        if len({
            vertex
            for index in indices
            for vertex in edges[index]
        }) == 4
    )
    for basis_color in range(D):
        q = tuple(
            int(color == basis_color) for color in range(D)
        )
        weights = {}
        for color, matching in enumerate(expected_matchings):
            weights[(*matching[0], color, color)] = q[color]
            weights[(*matching[1], color, color)] = 1
        for coloring in product(range(D), repeat=4):
            value = 0
            for matching in matchings:
                term = 1
                for left, right in matching:
                    term *= weights.get((
                        left,
                        right,
                        coloring[left],
                        coloring[right],
                    ), 0)
                value += term
            expected = (
                q[coloring[0]]
                if len(set(coloring)) == 1
                else 0
            )
            if value != expected:
                raise KrennRadicalIndependentVerificationError(
                    "the contracted GHZ witness failed symbolic replay"
                )


def _verify_phi4_dimension_and_equivariance(
    payload: Mapping,
) -> None:
    _strict_keys(
        payload,
        {
            "Phi_4_3_source_parameter_count",
            "Phi_4_3_perfect_matchings",
            "independent_reciprocal_gauge_actions",
            "every_Phi_4_3_monomial_has_gauge_weight_zero",
            "gauge_action_dimension_on_dense_source_torus",
            "generic_fiber_dimension_lower_bound",
            "affine_image_dimension_upper_bound",
            "local_GL_equivariance",
        },
        "Phi_4_3 dimension certificate",
    )
    edges4 = tuple(combinations(range(4), 2))
    matchings4 = tuple(
        tuple(edges4[index] for index in indices)
        for indices in combinations(range(len(edges4)), 2)
        if len({
            vertex
            for index in indices
            for vertex in edges4[index]
        }) == 4
    )
    if payload["Phi_4_3_perfect_matchings"] != [
        [list(edge) for edge in matching]
        for matching in matchings4
    ]:
        raise KrennRadicalIndependentVerificationError(
            "the independent Phi_4_3 matching census changed"
        )
    actions = payload["independent_reciprocal_gauge_actions"]
    if len(actions) != 3:
        raise KrennRadicalIndependentVerificationError(
            "the Phi_4_3 gauge-action census changed"
        )
    used_edges = set()
    for action, matching in zip(
        actions, matchings4, strict=True
    ):
        _strict_keys(
            action,
            {"positive_edge", "negative_edge", "weights"},
            "Phi_4_3 gauge action",
        )
        positive = tuple(action["positive_edge"])
        negative = tuple(action["negative_edge"])
        if (
            (positive, negative) != matching
            or action["weights"] != (
                "+1 on all 9 variables of positive_edge, "
                "-1 on all 9 variables of negative_edge, 0 otherwise"
            )
        ):
            raise KrennRadicalIndependentVerificationError(
                "a Phi_4_3 gauge action changed"
            )
        used_edges.update((positive, negative))
        for monomial_matching in matchings4:
            weight = (
                int(positive in monomial_matching)
                - int(negative in monomial_matching)
            )
            if weight:
                raise KrennRadicalIndependentVerificationError(
                    "a Phi_4_3 monomial has nonzero gauge weight"
                )
    source_parameters = len(edges4) * D * D
    if (
        len(used_edges) != len(edges4)
        or payload["Phi_4_3_source_parameter_count"]
        != source_parameters
        or payload[
            "every_Phi_4_3_monomial_has_gauge_weight_zero"
        ] is not True
        or payload[
            "gauge_action_dimension_on_dense_source_torus"
        ] != 3
        or payload["generic_fiber_dimension_lower_bound"] != 3
        or payload["affine_image_dimension_upper_bound"]
        != source_parameters - 3
    ):
        raise KrennRadicalIndependentVerificationError(
            "the Phi_4_3 dimension bound changed"
        )
    equivariance = payload["local_GL_equivariance"]
    _strict_keys(
        equivariance,
        {
            "matching_incidence_law",
            "source_lift",
            "contraction_identity",
            "rank_normal_form_dependency",
            "rank_strata_representatives_are_sufficient",
        },
        "local GL equivariance",
    )
    if any(
        sum(vertex in EDGES[edge] for edge in matching) != 1
        for matching in MATCHINGS
        for vertex in range(N)
    ):
        raise KrennRadicalIndependentVerificationError(
            "the perfect-matching incidence law changed"
        )
    contraction_lhs = Counter(
        (contracted_left, contracted_right, old_left, old_right)
        for contracted_left in range(D)
        for contracted_right in range(D)
        for old_left in range(D)
        for old_right in range(D)
    )
    contraction_rhs = Counter(
        (contracted_left, contracted_right, old_left, old_right)
        for old_left in range(D)
        for old_right in range(D)
        for contracted_left in range(D)
        for contracted_right in range(D)
    )
    if (
        contraction_lhs != contraction_rhs
        or len(contraction_lhs) != D**4
        or any(value != 1 for value in contraction_lhs.values())
    ):
        raise KrennRadicalIndependentVerificationError(
            "the bilinear contraction identity failed expansion"
        )
    expected_equivariance = {
        "matching_incidence_law": (
            "every perfect matching has exactly one edge incident "
            "to each physical vertex"
        ),
        "source_lift": (
            "a local GL3 matrix on vertex i acts on the i-color "
            "slot of every edge variable incident to i"
        ),
        "contraction_identity": (
            "C_K((A at vertex 0)(B at vertex 1)T) "
            "= C_(A^T K B)(T)"
        ),
        "rank_normal_form_dependency": (
            "every rank-r 3x3 matrix over characteristic zero is "
            "A^T diag(I_r,0) B for invertible A,B"
        ),
        "rank_strata_representatives_are_sufficient": True,
    }
    if equivariance != expected_equivariance:
        raise KrennRadicalIndependentVerificationError(
            "the local GL equivariance record changed"
        )


def _verify_two_vertex_contraction(payload: Mapping) -> None:
    _strict_keys(
        payload,
        {
            "contracted_vertices",
            "integer_probe_formula",
            "jacobian_shape",
            "rank_strata",
            "dimension_and_equivariance_certificate",
            "Phi_4_3_affine_image_dimension_upper_bound",
            "all_nonzero_contraction_strata_have_exact_rank_lower_bound_52",
            "rank_strata_cover_all_nonzero_3x3_matrices",
            "universal_nonzero_bilinear_contraction_preserves_image",
            "contracted_GHZ_target",
            "contracted_GHZ_target_witness",
            "claim_boundary",
        },
        "two-vertex contraction audit",
    )
    if (
        payload["contracted_vertices"] != [0, 1]
        or payload["integer_probe_formula"]
        != "4+16*i+30*i^2+15*i^3"
        or payload["jacobian_shape"] != [81, 135]
        or payload["Phi_4_3_affine_image_dimension_upper_bound"] != 51
        or payload[
            "all_nonzero_contraction_strata_have_exact_rank_lower_bound_52"
        ] is not True
        or payload["rank_strata_cover_all_nonzero_3x3_matrices"]
        != "GL3 x GL3 equivalence"
        or payload[
            "universal_nonzero_bilinear_contraction_preserves_image"
        ] is not False
        or payload["contracted_GHZ_target"]
        != "sum_a K_aa * |a^4>"
    ):
        raise KrennRadicalIndependentVerificationError(
            "the contraction audit metadata changed"
        )
    dimension_certificate = payload[
        "dimension_and_equivariance_certificate"
    ]
    _verify_phi4_dimension_and_equivariance(
        dimension_certificate
    )
    if (
        dimension_certificate[
            "affine_image_dimension_upper_bound"
        ]
        != payload["Phi_4_3_affine_image_dimension_upper_bound"]
    ):
        raise KrennRadicalIndependentVerificationError(
            "the contraction dimension bound is inconsistent"
        )
    strata = payload["rank_strata"]
    if len(strata) != 3:
        raise KrennRadicalIndependentVerificationError(
            "the contraction rank-strata census changed"
        )
    expected_modular_ranks = {1: 71, 2: 81, 3: 81}
    for contraction_rank, record in enumerate(strata, 1):
        _strict_keys(
            record,
            {
                "contraction_matrix_rank",
                "representative",
                "integer_jacobian_sha256",
                "exact_dimension_obstruction",
                "modular_rank_reconnaissance",
            },
            "contraction stratum",
        )
        matrix = _contraction_integer_jacobian(contraction_rank)
        if (
            record["contraction_matrix_rank"] != contraction_rank
            or record["representative"] != (
                "diag(" + ",".join(
                    "1" if index < contraction_rank else "0"
                    for index in range(D)
                ) + ")"
            )
            or record["integer_jacobian_sha256"]
            != _integer_matrix_fingerprint(matrix)
        ):
            raise KrennRadicalIndependentVerificationError(
                "a contraction integer matrix changed"
            )
        exact = record["exact_dimension_obstruction"]
        _strict_keys(
            exact,
            {
                "minor_size",
                "pivot_rows",
                "pivot_columns",
                "integer_determinant",
                "rank_over_Q_at_least",
            },
            "exact contraction minor",
        )
        rows = tuple(map(int, exact["pivot_rows"]))
        columns = tuple(map(int, exact["pivot_columns"]))
        if (
            exact["minor_size"] != 52
            or len(rows) != 52
            or len(set(rows)) != 52
            or any(row not in range(81) for row in rows)
            or len(columns) != 52
            or len(set(columns)) != 52
            or any(column not in range(135) for column in columns)
            or exact["rank_over_Q_at_least"] != 52
        ):
            raise KrennRadicalIndependentVerificationError(
                "an exact contraction minor index set changed"
            )
        minor = [
            [matrix[row][column] for column in columns]
            for row in rows
        ]
        determinant = _exact_determinant(minor)
        if (
            determinant.denominator != 1
            or determinant.numerator != exact["integer_determinant"]
            or not determinant
        ):
            raise KrennRadicalIndependentVerificationError(
                "an exact contraction minor failed replay"
            )
        modular = record["modular_rank_reconnaissance"]
        if len(modular) != 3:
            raise KrennRadicalIndependentVerificationError(
                "the contraction modular receipt changed"
            )
        for prime_record, prime in zip(
            modular, (31, 1_009, 1_000_003), strict=True
        ):
            rank, modular_determinant = (
                _rank_and_minor_determinant_mod(matrix, prime)
            )
            if prime_record != {
                "prime": prime,
                "rank_mod_prime": expected_modular_ranks[
                    contraction_rank
                ],
                "full_rank_minor_determinant_mod_prime": (
                    modular_determinant
                ),
                "role": "nonproof-reconnaissance",
            } or rank != expected_modular_ranks[contraction_rank]:
                raise KrennRadicalIndependentVerificationError(
                    "a modular contraction diagnostic changed"
                )
    _verify_contracted_ghz_witness(
        payload["contracted_GHZ_target_witness"]
    )


def _strict_keys(
    payload: Mapping,
    expected: set[str],
    label: str,
) -> None:
    if set(payload) != expected:
        raise KrennRadicalIndependentVerificationError(
            f"{label} schema changed"
        )


def verify_support_obstruction_payload(payload: Mapping) -> Mapping:
    """Independently replay a producer support-obstruction summary."""

    _strict_keys(
        payload,
        {
            "schema",
            "support_row_search",
            "rainbow_witness_fingerprint",
            "rainbow_singleton_column_fingerprint",
            "propagation_audits",
            "computations_performed",
            "dependencies",
            "claims",
        },
        "support obstruction",
    )
    if payload["schema"] != (
        "krenn.n6_d3.radical_obstruction.support_rows.v1"
    ):
        raise KrennRadicalIndependentVerificationError(
            "support obstruction schema changed"
        )
    propagation_audits = payload["propagation_audits"]
    _strict_keys(
        propagation_audits,
        {"two_vertex_contraction"},
        "propagation audits",
    )
    _verify_two_vertex_contraction(
        propagation_audits["two_vertex_contraction"]
    )
    search = payload["support_row_search"]
    _strict_keys(
        search,
        {
            "support_orbits",
            "rainbow_support_orbits",
            "rainbow_singleton_constraints",
            "nonrainbow_support_orbits",
            "nonrainbow_representative_states",
            "nonrainbow_raw_monomials",
            "switch_predecessors",
            "unique_switch_constraints",
            "basis_constraints",
            "matrix_entry_normalization",
            "basis_determinant",
            "full_constraint_rank_over_Q",
            "supported_invariant_annihilator_dimension",
            "Reynolds_consequence",
            "claim_boundary",
        },
        "support row search",
    )
    orbit_records, state_to_representative = _direct_support_orbits()
    rainbow = []
    nonrainbow = []
    witnesses = []
    nonrainbow_mass = 0
    orbit_size = {
        state: size for state, size, _coefficient in orbit_records
    }
    for representative_state, size, _coefficient in orbit_records:
        witness = _rainbow_witness(
            _decode_state(representative_state)
        )
        if witness is None:
            nonrainbow.append(representative_state)
            nonrainbow_mass += size
        else:
            rainbow.append(representative_state)
            witnesses.append(witness)
    if (
        len(rainbow) != 632
        or len(nonrainbow) != 31
        or nonrainbow_mass != 34_560
        or search["support_orbits"] != 663
        or search["rainbow_support_orbits"] != 632
        or search["rainbow_singleton_constraints"] != 632
        or search["nonrainbow_support_orbits"] != 31
        or search["nonrainbow_representative_states"] != nonrainbow
        or search["nonrainbow_raw_monomials"] != 34_560
    ):
        raise KrennRadicalIndependentVerificationError(
            "rainbow/nonrainbow support partition changed"
        )

    witness_payload = [
        {
            "orbit_index": orbit_index,
            "matching": list(matching),
            "color_edge_positions": list(edge_positions),
        }
        for orbit_index, (matching, edge_positions) in zip(
            (
                index
                for index, (state, _size, _coefficient)
                in enumerate(orbit_records)
                if state in set(rainbow)
            ),
            witnesses,
            strict=True,
        )
    ]
    witness_fingerprint = sha256(json.dumps(
        witness_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")).hexdigest()

    singleton_rows = []
    for representative_state, witness in zip(
        rainbow, witnesses, strict=True
    ):
        coloring, multiplier = _rainbow_column(
            representative_state, witness
        )
        support_hits = []
        for term in _equation_terms(coloring):
            state = _monomial_support_state((*multiplier, *term))
            if state is not None:
                support_hits.append(
                    int(state_to_representative[state])
                )
        if support_hits != [representative_state]:
            raise KrennRadicalIndependentVerificationError(
                "a rainbow singleton column failed independent replay"
            )
        orbit_index = next(
            index
            for index, (state, _size, _coefficient)
            in enumerate(orbit_records)
            if state == representative_state
        )
        matching, edge_positions = witness
        singleton_rows.append((
            orbit_index,
            matching,
            edge_positions,
            coloring,
            multiplier,
        ))
    singleton_fingerprint = sha256(json.dumps(
        singleton_rows, separators=(",", ":")
    ).encode("ascii")).hexdigest()
    if (
        payload["rainbow_witness_fingerprint"]
        != witness_fingerprint
        or payload["rainbow_singleton_column_fingerprint"]
        != singleton_fingerprint
    ):
        raise KrennRadicalIndependentVerificationError(
            "rainbow witness fingerprint changed"
        )

    nonrainbow_set = set(nonrainbow)
    nonrainbow_index = {
        state: index for index, state in enumerate(nonrainbow)
    }
    matrix = []
    constraints = search["basis_constraints"]
    if len(constraints) != 31:
        raise KrennRadicalIndependentVerificationError(
            "the retained basis does not have 31 rows"
        )
    for constraint in constraints:
        _strict_keys(
            constraint,
            {
                "coloring",
                "multiplier_variable_indices",
                "multiplier_variable_keys",
                "projected_entries",
            },
            "basis constraint",
        )
        coloring = tuple(map(int, constraint["coloring"]))
        multiplier = tuple(map(
            int, constraint["multiplier_variable_indices"]
        ))
        if (
            len(multiplier) != 15
            or tuple(sorted(multiplier)) != multiplier
            or [
                list(_variable_key(variable))
                for variable in multiplier
            ]
            != constraint["multiplier_variable_keys"]
        ):
            raise KrennRadicalIndependentVerificationError(
                "a multiplier record changed"
            )
        projected = Counter()
        for term in _equation_terms(coloring):
            state = _monomial_support_state((*multiplier, *term))
            if state is None:
                continue
            representative = int(state_to_representative[state])
            if representative in nonrainbow_set:
                projected[representative] += 1
        expected_entries = [
            [state, value]
            for state, value in sorted(projected.items())
        ]
        if constraint["projected_entries"] != expected_entries:
            raise KrennRadicalIndependentVerificationError(
                "a projected source column failed symbolic replay"
            )
        row = [0] * len(nonrainbow)
        for state, value in projected.items():
            row[nonrainbow_index[state]] = value
        matrix.append(row)
    determinant = _exact_determinant(matrix)
    if (
        determinant != -1
        or search["basis_determinant"] != -1
        or search["full_constraint_rank_over_Q"] != 31
        or search["supported_invariant_annihilator_dimension"] != 0
        or search["switch_predecessors"] != 291
        or search["unique_switch_constraints"] != 56
    ):
        raise KrennRadicalIndependentVerificationError(
            "the exact support constraint minor changed"
        )
    if search["matrix_entry_normalization"] != (
        "number of the 15 raw F_c matching terms landing in "
        "the indicated D^2 support orbit"
    ):
        raise KrennRadicalIndependentVerificationError(
            "matrix entry normalization changed"
        )

    if payload["dependencies"] != {
        "prior_k1_result": "D not in J_mix at k=1",
        "prior_border_result": (
            "GHZ_6,3 is in the Euclidean and Zariski border image"
        ),
        "prior_support_conditional_result": (
            "retaining all nine natural seed coordinates forces "
            "total support at least 22; the six minimal "
            "support-21 terminals are exactly excluded"
        ),
    }:
        raise KrennRadicalIndependentVerificationError(
            "the exact dependency boundary changed"
        )
    if payload["computations_performed"] != {
        "full_k2_matrix_constructed": False,
        "support_row_orbits_enumerated": True,
        "exact_integer_minor_replayed": True,
        "exact_contraction_minors_replayed": True,
        "modular_arithmetic_needed_for_exact_claim": False,
    }:
        raise KrennRadicalIndependentVerificationError(
            "the computation boundary changed"
        )
    claims = payload["claims"]
    expected_claims = {
        "support_limited_D_squared_separator_exists": False,
        "D_squared_in_J_mix_decided": False,
        "D_in_radical_J_mix_decided": False,
        "new_support_conditional_GHZ_nonexistence_proved": False,
        "global_GHZ_nonexistence_proved": False,
        "exact_affine_GHZ_membership_decided": False,
        "evidence_status": "undecided",
    }
    if claims != expected_claims:
        raise KrennRadicalIndependentVerificationError(
            "support-obstruction claim boundary changed"
        )
    return {
        "support_orbit_census_replayed": True,
        "rainbow_singletons_replayed": True,
        "switch_columns_replayed_symbolically": True,
        "unimodular_minor_replayed": determinant == -1,
        "exact_contraction_minors_replayed": True,
        "contracted_GHZ_witness_replayed": True,
        "support_limited_separator_excluded": True,
        "global_radical_membership_decided": False,
        "exact_affine_GHZ_membership_decided": False,
    }


__all__ = [
    "KrennRadicalIndependentVerificationError",
    "verify_support_obstruction_payload",
]
