from dataclasses import replace
from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.fixtures import (
    fixture_n4_d3,
    fixture_n6_d2,
)
from experiments.krenn_quantum_graph.hafnian_identities import (
    certify_hafnian_contraction,
)
from experiments.krenn_quantum_graph.independent_verifier import (
    independent_perfect_matchings,
)
from experiments.krenn_quantum_graph.multi_star_identities import (
    KrennMultiStarIdentityError,
    MultiStarIdentityCertificate,
    build_star_linearization,
    certify_multi_star_identities,
    induced_matching_sum,
    matching_cofactor,
    monochromatic_matching_marginals,
    star_expansion,
    two_star_expansion,
)
from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    variable_index,
)
from experiments.krenn_quantum_graph.targets import canonical_ghz_target
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.ternary_search import (
    n6_d3_seed_witness,
)
from experiments.krenn_quantum_graph.witness import SparseWitness


def _independent_induced_sum(witness, coloring, vertices):
    if not vertices:
        return Fraction(1)
    values = witness.as_dict()
    total = Fraction(0)
    for matching in independent_perfect_matchings(len(vertices)):
        term = Fraction(1)
        for local_i, local_j in matching:
            i = vertices[local_i]
            j = vertices[local_j]
            term *= values.get(
                variable_index(
                    witness.n,
                    witness.d,
                    i,
                    j,
                    coloring[i],
                    coloring[j],
                ),
                Fraction(0),
            )
        total += term
    return total


class KrennMultiStarIdentityTest(unittest.TestCase):
    def test_asymmetric_orientation_and_independent_cofactors(self):
        witness = SparseWitness.from_coordinates(
            4,
            3,
            {
                (0, 1, 0, 1): Fraction(2, 3),
                (0, 1, 1, 0): Fraction(-5, 7),
                (2, 3, 2, 1): Fraction(11, 5),
                (0, 2, 0, 2): 3,
                (1, 3, 1, 0): -2,
            },
        )
        certificate = certify_multi_star_identities(witness)
        self.assertTrue(certificate.exact)
        self.assertEqual(certificate.star_coefficient_checks, 4 * 3**4)
        self.assertEqual(
            certificate.shared_edge_entry_checks,
            len(canonical_edges(4)) * 3**2,
        )

        left = build_star_linearization(
            witness, 0
        ).oriented_edge_matrix(1)
        right = build_star_linearization(
            witness, 1
        ).oriented_edge_matrix(0)
        self.assertEqual(left[0][1], Fraction(2, 3))
        self.assertEqual(left[1][0], Fraction(-5, 7))
        self.assertEqual(
            left,
            tuple(tuple(row) for row in zip(*right, strict=True)),
        )

        coloring = (0, 1, 2, 1)
        for removed in canonical_edges(4):
            vertices = tuple(
                vertex for vertex in range(4) if vertex not in removed
            )
            self.assertEqual(
                induced_matching_sum(witness, coloring, vertices),
                _independent_induced_sum(
                    witness, coloring, vertices
                ),
            )
        changed_removed_colors = (2, 0, 2, 1)
        self.assertEqual(
            matching_cofactor(witness, coloring, 0, 1),
            matching_cofactor(
                witness, changed_removed_colors, 0, 1
            ),
        )

    def test_asymmetric_n6_public_expansions_and_k4_cofactor(self):
        coordinates = {}
        for edge_number, (first, second) in enumerate(
            canonical_edges(6)
        ):
            for first_color in range(3):
                for second_color in range(3):
                    numerator = (
                        7 * edge_number
                        + 5 * first_color
                        + 3 * second_color
                    ) % 17 - 8
                    if numerator:
                        coordinates[
                            (
                                first,
                                second,
                                first_color,
                                second_color,
                            )
                        ] = Fraction(
                            numerator,
                            1
                            + (
                                edge_number
                                + 2 * first_color
                                + second_color
                            )
                            % 5,
                        )
        witness = SparseWitness.from_coordinates(6, 3, coordinates)
        tensor_map = MatchingTensorMap(6, 3)
        for equation in range(tensor_map.coefficient_count):
            coloring = tuple(
                (equation // 3 ** (5 - vertex)) % 3
                for vertex in range(6)
            )
            expected = tensor_map.coefficient_exact(
                witness, coloring
            )
            for root in range(6):
                self.assertEqual(
                    star_expansion(witness, coloring, root),
                    expected,
                )
            self.assertEqual(
                two_star_expansion(witness, coloring, 0, 5),
                expected,
            )
            self.assertEqual(
                two_star_expansion(witness, coloring, 5, 0),
                expected,
            )

        for residual_colors in (
            (0, 1, 2, 0),
            (2, 2, 1, 0),
            (1, 0, 1, 2),
        ):
            coloring = (0, 0, *residual_colors)
            self.assertEqual(
                matching_cofactor(witness, coloring, 0, 1),
                _independent_induced_sum(
                    witness, coloring, (2, 3, 4, 5)
                ),
            )

    def test_k4_exception_reconstructs_ghz_at_every_root(self):
        witness = fixture_n4_d3()
        target = canonical_ghz_target(4, 3)
        certificate = certify_multi_star_identities(witness)
        self.assertTrue(certificate.exact)
        for root in range(4):
            self.assertEqual(
                build_star_linearization(
                    witness, root
                ).reconstructed_target(),
                target,
            )
        fixture_matchings = (
            {(0, 1), (2, 3)},
            {(0, 2), (1, 3)},
            {(0, 3), (1, 2)},
        )
        for color, expected_edges in enumerate(fixture_matchings):
            marginal = monochromatic_matching_marginals(
                witness, color
            )
            self.assertEqual(
                {
                    edge
                    for edge in canonical_edges(4)
                    if marginal[edge[0]][edge[1]]
                },
                expected_edges,
            )
            self.assertEqual(
                tuple(sum(row) for row in marginal),
                (Fraction(1),) * 4,
            )

        contraction = certify_hafnian_contraction(
            witness,
            (
                (1, 2, -1),
                (3, -2, 4),
                (Fraction(1, 2), 0, 5),
                (-1, Fraction(3, 5), 2),
            ),
            target=target,
        )
        self.assertTrue(contraction.target_contraction_satisfied)

    def test_n6_d3_seed_reconstructs_its_unique_victim(self):
        witness = n6_d3_seed_witness()
        tensor_map = MatchingTensorMap(6, 3)
        output = tensor_map.evaluate_exact(witness)
        target = canonical_ghz_target(6, 3)
        certificate = certify_multi_star_identities(witness)
        self.assertTrue(certificate.exact)
        self.assertEqual(certificate.star_coefficient_checks, 6 * 729)
        self.assertEqual(
            certificate.two_star_coefficient_checks, 15 * 729
        )
        for root in range(6):
            self.assertEqual(
                build_star_linearization(
                    witness, root
                ).reconstructed_target(),
                output,
            )

        residual = tensor_map.exact_residual(witness, target)
        self.assertEqual(
            residual.entries,
            (((0, 0, 2, 1, 2, 1), Fraction(1)),),
        )
        victim = (0, 0, 2, 1, 2, 1)
        for root in range(6):
            self.assertEqual(
                star_expansion(witness, victim, root),
                Fraction(1),
            )
        for first, second in canonical_edges(6):
            self.assertEqual(
                two_star_expansion(
                    witness, victim, first, second
                ),
                Fraction(1),
            )

        for color in range(3):
            marginal = monochromatic_matching_marginals(
                witness, color
            )
            self.assertEqual(
                tuple(sum(row) for row in marginal),
                (Fraction(1),) * 6,
            )

    def test_n6_d2_fixture_is_exact_ghz_under_all_roots(self):
        witness = fixture_n6_d2()
        target = canonical_ghz_target(6, 2)
        certificate = certify_multi_star_identities(witness)
        self.assertTrue(certificate.exact)
        for root in range(6):
            self.assertEqual(
                build_star_linearization(
                    witness, root
                ).reconstructed_target(),
                target,
            )
        for color in range(2):
            marginal = monochromatic_matching_marginals(
                witness, color
            )
            self.assertEqual(
                tuple(sum(row) for row in marginal),
                (Fraction(1),) * 6,
            )

    def test_validation_fails_closed(self):
        witness = fixture_n4_d3()
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "even vertex"
        ):
            induced_matching_sum(witness, (0, 0, 0, 0), (0, 1, 2))
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "distinct"
        ):
            induced_matching_sum(witness, (0, 0, 0, 0), (0, 0))
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "coloring"
        ):
            star_expansion(witness, (0, 0, 0), 0)
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "root"
        ):
            star_expansion(witness, (0, 0, 0, 0), 4)
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "distinct"
        ):
            two_star_expansion(witness, (0, 0, 0, 0), 1, 1)
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "exact integer"
        ):
            star_expansion(witness, (0, 0, 0, 0), 0.5)
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "exact integer"
        ):
            star_expansion(witness, (0, 0, 0, True), 0)
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "exact integer"
        ):
            induced_matching_sum(
                witness, (0, 0, 0, 0), (0, 1.5)
            )
        certificate = certify_multi_star_identities(witness)
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "census"
        ):
            replace(
                certificate,
                star_coefficient_checks=(
                    certificate.star_coefficient_checks - 1
                ),
            )
        with self.assertRaisesRegex(
            KrennMultiStarIdentityError, "flags"
        ):
            MultiStarIdentityCertificate(
                n=certificate.n,
                d=certificate.d,
                star_coefficient_checks=(
                    certificate.star_coefficient_checks
                ),
                cofactor_recursion_checks=(
                    certificate.cofactor_recursion_checks
                ),
                shared_edge_entry_checks=(
                    certificate.shared_edge_entry_checks
                ),
                two_star_coefficient_checks=(
                    certificate.two_star_coefficient_checks
                ),
                all_star_expansions_exact=1,
                all_cofactor_recursions_exact=True,
                all_shared_edge_blocks_exact=True,
                all_two_star_expansions_exact=True,
            )


if __name__ == "__main__":
    unittest.main()
