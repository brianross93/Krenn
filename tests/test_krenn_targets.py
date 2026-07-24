from fractions import Fraction
import json
from pathlib import Path
import unittest

from experiments.krenn_quantum_graph import targets
from experiments.krenn_quantum_graph.system import (
    KrennSystemError,
    coloring_index,
    enumerate_colorings,
    generate_sparse_system,
    rhs_for_coloring,
)
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.witness import (
    KrennWitnessError,
    SparseWitness,
    evaluate_mod,
)


class KrennColoringTargetTest(unittest.TestCase):
    def test_canonical_ghz_preserves_the_existing_system_exactly(self):
        default = generate_sparse_system(4, 3)
        explicit = generate_sparse_system(
            4, 3, targets.canonical_ghz_target(4, 3)
        )
        self.assertEqual(default, explicit)
        self.assertEqual(
            default.rhs_values,
            tuple(
                rhs_for_coloring(coloring)
                for coloring in enumerate_colorings(4, 3)
            ),
        )
        self.assertTrue(
            all(type(value) is int for value in default.rhs_values)
        )
        self.assertEqual(
            bytes(default.rhs_values), bytes(explicit.rhs_values)
        )
        self.assertEqual(default.target, targets.canonical_ghz_target(4, 3))
        self.assertTrue(default.has_canonical_ghz_target)

    def test_sparse_dense_target_and_system_round_trip_exactly(self):
        sparse = targets.ColoringTarget.from_sparse(
            4,
            2,
            {
                (0, 0, 0, 1): Fraction(2, 3),
                (1, 0, 1, 0): -5,
                (1, 1, 1, 1): Fraction(7, 4),
            },
        )
        dense = [0] * 16
        for coloring, value in sparse.entries:
            dense[coloring_index(4, 2, coloring)] = value
        self.assertEqual(
            sparse,
            targets.ColoringTarget.from_dense(4, 2, dense),
        )
        system = generate_sparse_system(4, 2, sparse)
        self.assertEqual(system.rhs_values, tuple(dense))
        self.assertEqual(system.target, sparse)
        self.assertFalse(system.has_canonical_ghz_target)
        self.assertEqual(
            generate_sparse_system(4, 2, system.target), system
        )
        self.assertEqual(
            system.target.coefficient((0, 1, 0, 1)), 0
        )

    def test_target_payload_round_trip_is_lossless_and_canonical(self):
        target = targets.ColoringTarget.from_sparse(
            2,
            3,
            [
                ((2, 1), Fraction(-7, 5)),
                ((0, 2), 3),
                ((1, 1), 0),
            ],
        )
        payload = json.loads(json.dumps(target.to_payload()))
        self.assertEqual(
            targets.ColoringTarget.from_payload(payload), target
        )
        self.assertEqual(target.support_size, 2)
        self.assertEqual(
            tuple(coloring for coloring, _value in target.entries),
            ((0, 2), (2, 1)),
        )

    def test_exact_target_validation_fails_closed(self):
        with self.assertRaises(targets.KrennTargetError):
            targets.ColoringTarget.from_sparse(
                2,
                2,
                [((0, 0), 1), ((0, 0), 2)],
            )
        with self.assertRaises(targets.KrennTargetError):
            targets.ColoringTarget.from_sparse(
                2, 2, {((0, 0)): 0.5}
            )
        with self.assertRaises(targets.KrennTargetError):
            targets.ColoringTarget.from_dense(2, 2, [0, 1])
        with self.assertRaises(targets.KrennTargetError):
            targets.ColoringTarget.from_sparse(
                2, 2, {((0, 2)): 1}
            )
        with self.assertRaises(KrennSystemError):
            generate_sparse_system(
                4, 2, targets.canonical_ghz_target(2, 2)
            )

    def test_unnormalized_w_and_dicke_convention_is_exact(self):
        w = targets.unnormalized_w_target(4)
        self.assertEqual(w, targets.unnormalized_dicke_target(4, 1))
        self.assertEqual(w.support_size, 4)
        self.assertTrue(
            all(sum(coloring) == 1 for coloring, _value in w.entries)
        )
        dicke = targets.unnormalized_dicke_target(4, 2)
        self.assertEqual(dicke.support_size, 6)
        self.assertTrue(
            all(
                sum(coloring) == 2 and value == 1
                for coloring, value in dicke.entries
            )
        )
        self.assertEqual(
            dicke,
            targets.unnormalized_qudit_dicke_target((2, 2)),
        )

    def test_heralded_ghz_uses_k_monochromatic_registers(self):
        heralded = targets.heralded_ghz_target(
            6, 3, 2, trigger_color=2
        )
        self.assertEqual(
            heralded.entries,
            (
                ((0, 0, 2, 2, 2, 2), 1),
                ((1, 1, 2, 2, 2, 2), 1),
                ((2, 2, 2, 2, 2, 2), 1),
            ),
        )
        self.assertEqual(
            targets.heralded_ghz_target(4, 3, 4),
            targets.canonical_ghz_target(4, 3),
        )
        with self.assertRaises(targets.KrennTargetError):
            targets.heralded_ghz_target(4, 3, 0)
        with self.assertRaises(targets.KrennTargetError):
            targets.heralded_ghz_target(4, 3, 5)
        with self.assertRaises(targets.KrennTargetError):
            targets.heralded_ghz_target(
                4, 3, 1, trigger_color=3
            )

    def test_qudit_dicke_uses_an_exact_occupation_tuple(self):
        dicke = targets.unnormalized_qudit_dicke_target(
            (2, 1, 1)
        )
        self.assertEqual((dicke.n, dicke.d), (4, 3))
        self.assertEqual(dicke.support_size, 12)
        self.assertTrue(
            all(
                (
                    coloring.count(0),
                    coloring.count(1),
                    coloring.count(2),
                )
                == (2, 1, 1)
                and value == 1
                for coloring, value in dicke.entries
            )
        )
        with self.assertRaises(targets.KrennTargetError):
            targets.unnormalized_qudit_dicke_target((0, 0))
        with self.assertRaises(targets.KrennTargetError):
            targets.unnormalized_qudit_dicke_target((2, -1, 1))

    def test_graph_state_uses_the_documented_cz_phase_convention(self):
        graph = targets.unnormalized_graph_state_target(
            4, ((0, 1), (2, 1))
        )
        self.assertEqual(graph.support_size, 16)
        for coloring in enumerate_colorings(4, 2):
            expected = (
                -1
                if (
                    coloring[0] * coloring[1]
                    + coloring[1] * coloring[2]
                )
                % 2
                else 1
            )
            self.assertEqual(graph.coefficient(coloring), expected)
        with self.assertRaises(targets.KrennTargetError):
            targets.unnormalized_graph_state_target(
                4, ((0, 1), (1, 0))
            )

    def test_rational_target_reduces_exactly_modulo_31(self):
        tensor_map = MatchingTensorMap(2, 2)
        witness = SparseWitness.from_coordinates(
            2,
            2,
            {
                (0, 1, 0, 0): Fraction(1, 2),
                (0, 1, 1, 1): Fraction(-2, 3),
            },
        )
        target = tensor_map.evaluate_exact(witness)
        system = generate_sparse_system(2, 2, target)
        evaluation = evaluate_mod(system, witness, 31)

        self.assertTrue(evaluation.satisfied)
        self.assertEqual(
            evaluation.equation_values[
                coloring_index(2, 2, (0, 0))
            ],
            16,
        )
        self.assertTrue(
            all(type(residual) is int for residual in evaluation.residuals)
        )

    def test_noninvertible_target_denominator_fails_modular_replay(self):
        target = targets.ColoringTarget.from_sparse(
            2, 2, {(0, 0): Fraction(1, 31)}
        )
        system = generate_sparse_system(2, 2, target)
        zero = SparseWitness.from_index_values(2, 2, ())
        with self.assertRaisesRegex(
            KrennWitnessError, "denominator is not invertible"
        ):
            evaluate_mod(system, zero, 31)

    def test_target_core_has_no_section12_import(self):
        source = Path(targets.__file__).read_text(encoding="utf-8")
        self.assertNotIn("section12_instantiation", source)


if __name__ == "__main__":
    unittest.main()
