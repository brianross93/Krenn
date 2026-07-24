from fractions import Fraction
import unittest

from experiments.krenn_quantum_graph.deformation import (
    rescale_complementary_edges,
)
from experiments.krenn_quantum_graph.fixtures import fixture_n4_d3
from experiments.krenn_quantum_graph.system import (
    coloring_from_index,
    generate_sparse_system,
)
from experiments.krenn_quantum_graph.transport import (
    KrennTransportError,
    certify_system_transport,
    compose_permutations,
    equation_permutation,
    inverse_permutation,
    transport_variable_key,
    transport_witness,
)
from experiments.krenn_quantum_graph.witness import (
    evaluate_exact,
    evaluate_mod,
)


class KrennTransportTest(unittest.TestCase):
    def test_reversed_edge_swaps_endpoint_color_slots(self):
        vertex = (3, 1, 2, 0)
        color = (1, 2, 0)
        self.assertEqual(
            transport_variable_key(
                4, 3, (0, 3, 0, 2), vertex, color
            ),
            (0, 3, 0, 1),
        )

    def test_complete_system_transport_is_an_exact_bijection(self):
        system = generate_sparse_system(4, 3)
        certificate = certify_system_transport(
            system, (2, 0, 3, 1), (1, 2, 0)
        )
        self.assertTrue(certificate.exact)
        self.assertGreater(certificate.reversed_variable_count, 0)

    def test_witness_transport_is_equivariant_over_q_and_f31(self):
        system = generate_sparse_system(4, 3)
        witness = rescale_complementary_edges(
            fixture_n4_d3(),
            (Fraction(2), Fraction(3, 2), Fraction(5, 3)),
        )
        vertex = (2, 0, 3, 1)
        color = (1, 2, 0)
        transported = transport_witness(witness, vertex, color)
        equation_map = equation_permutation(4, 3, vertex, color)

        source_q = evaluate_exact(system, witness)
        target_q = evaluate_exact(system, transported)
        source_f = evaluate_mod(system, witness)
        target_f = evaluate_mod(system, transported)
        self.assertTrue(target_q.satisfied)
        self.assertTrue(target_f.satisfied)
        for source, target in enumerate(equation_map):
            self.assertEqual(
                source_q.equation_values[source],
                target_q.equation_values[target],
                msg=coloring_from_index(4, 3, source),
            )
            self.assertEqual(
                source_f.equation_values[source],
                target_f.equation_values[target],
            )

    def test_transport_composition_and_inverse_round_trip(self):
        witness = fixture_n4_d3()
        vertex_one = (2, 0, 3, 1)
        vertex_two = (1, 3, 0, 2)
        color_one = (1, 2, 0)
        color_two = (2, 0, 1)
        sequential = transport_witness(
            transport_witness(
                witness, vertex_one, color_one
            ),
            vertex_two,
            color_two,
        )
        combined = transport_witness(
            witness,
            compose_permutations(vertex_two, vertex_one),
            compose_permutations(color_two, color_one),
        )
        self.assertEqual(sequential, combined)
        self.assertEqual(
            transport_witness(
                transport_witness(
                    witness, vertex_one, color_one
                ),
                inverse_permutation(vertex_one),
                inverse_permutation(color_one),
            ),
            witness,
        )

    def test_invalid_permutations_fail_closed(self):
        with self.assertRaises(KrennTransportError):
            transport_witness(
                fixture_n4_d3(), (0, 0, 2, 3), (0, 1, 2)
            )


if __name__ == "__main__":
    unittest.main()

