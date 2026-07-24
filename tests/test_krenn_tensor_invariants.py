from pathlib import Path
import unittest

from experiments.krenn_quantum_graph.system import variable_index
from experiments.krenn_quantum_graph.targets import (
    ColoringTarget,
    canonical_ghz_target,
    unnormalized_dicke_target,
    unnormalized_graph_state_target,
    unnormalized_w_target,
)
from experiments.krenn_quantum_graph.tensor_invariants import (
    KrennTensorInvariantError,
    rational_flattening,
    support_character_invariant,
    target_support_profile,
    universal_image_dimension_bounds,
)
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap


class KrennTensorInvariantTest(unittest.TestCase):
    def test_universal_dimension_bounds_use_only_effective_gauge(self):
        bounds = universal_image_dimension_bounds(
            MatchingTensorMap(6, 4)
        )
        self.assertEqual(bounds.parameter_dimension, 240)
        self.assertEqual(bounds.scalar_vertex_gauge_dimension, 5)
        self.assertEqual(bounds.affine_ambient_dimension, 4096)
        self.assertEqual(
            bounds.projective_ambient_dimension, 4095
        )
        self.assertEqual(
            bounds.affine_image_dimension_upper_bound, 235
        )
        self.assertEqual(
            bounds.projective_image_dimension_upper_bound, 234
        )
        self.assertEqual(bounds.affine_codimension_lower_bound, 3861)
        self.assertEqual(
            bounds.projective_codimension_lower_bound, 3861
        )
        self.assertTrue(
            bounds.proper_affine_image_closure_certified
        )
        self.assertFalse(bounds.individual_target_nonimage_proved)
        self.assertTrue(bounds.necessary_invariant_only)
        self.assertIn("generic target", bounds.claim_boundary)
        self.assertIn("does not prove", bounds.claim_boundary)

        # At n=2 the product-one vertex action is trivial on the sole edge;
        # subtracting n-1 would contradict the exact Phi(W)=W regression.
        with self.assertRaisesRegex(
            KrennTensorInvariantError, "requires n >= 4"
        ):
            universal_image_dimension_bounds(
                MatchingTensorMap(2, 3)
            )

    def test_named_target_support_character_dimensions(self):
        ghz = support_character_invariant(
            canonical_ghz_target(6, 4)
        )
        self.assertEqual(ghz.matrix[0][:8], (1, 0, 0, 0, 1, 0, 0, 0))
        self.assertEqual(ghz.support_size, 4)
        self.assertEqual(ghz.rank_over_q, 4)
        self.assertEqual(
            ghz.local_diagonal_stabilizer_dimension, 20
        )
        self.assertTrue(ghz.exact)

        dense_graph = support_character_invariant(
            unnormalized_graph_state_target(
                6, ((0, 1), (1, 2), (4, 5))
            )
        )
        self.assertEqual(dense_graph.support_size, 64)
        self.assertEqual(dense_graph.rank_over_q, 7)
        self.assertEqual(
            dense_graph.local_diagonal_stabilizer_dimension,
            6 - 1,
        )

        w_state = support_character_invariant(
            unnormalized_w_target(6)
        )
        dicke = support_character_invariant(
            unnormalized_dicke_target(6, 2)
        )
        self.assertEqual(w_state.rank_over_q, 6)
        self.assertEqual(dicke.rank_over_q, 6)
        self.assertEqual(
            w_state.local_diagonal_stabilizer_dimension, 6
        )
        self.assertEqual(
            dicke.local_diagonal_stabilizer_dimension, 6
        )
        self.assertFalse(dicke.individual_target_nonimage_proved)

    def test_ghz_w_and_dicke_flattening_ranks_are_exact(self):
        cut = (0, 1, 2)
        ghz = rational_flattening(
            canonical_ghz_target(6, 4), cut
        )
        self.assertEqual(ghz.left_vertices, cut)
        self.assertEqual(ghz.right_vertices, (3, 4, 5))
        self.assertEqual(ghz.shape, (64, 64))
        self.assertEqual(ghz.rank_over_q, 4)
        self.assertTrue(ghz.exact)

        w_state = rational_flattening(
            unnormalized_w_target(6), cut
        )
        dicke = rational_flattening(
            unnormalized_dicke_target(6, 2), cut
        )
        self.assertEqual(w_state.shape, (8, 8))
        self.assertEqual(w_state.rank_over_q, 2)
        self.assertEqual(dicke.rank_over_q, 3)
        self.assertFalse(
            dicke.individual_target_nonimage_proved
        )
        self.assertIn("neither", dicke.claim_boundary)

    def test_target_support_profile_obstructs_only_declared_support(self):
        tensor_map = MatchingTensorMap(4, 2)
        support = (
            variable_index(4, 2, 0, 1, 0, 0),
            variable_index(4, 2, 2, 3, 0, 0),
        )
        supported_target = ColoringTarget.from_sparse(
            4, 2, {(0, 0, 0, 0): 1}
        )
        passing = target_support_profile(
            tensor_map, supported_target, support
        )
        self.assertTrue(passing.necessary_condition_satisfied)
        self.assertFalse(passing.exact_support_excluded)
        self.assertFalse(passing.solution_certified)
        self.assertFalse(passing.individual_target_nonimage_proved)
        self.assertIn("necessary", passing.claim_boundary)
        self.assertIn("no weights", passing.claim_boundary)

        missing_target = ColoringTarget.from_sparse(
            4,
            2,
            {
                (0, 0, 0, 0): 1,
                (1, 1, 1, 1): 1,
            },
        )
        missing = target_support_profile(
            tensor_map, missing_target, support
        )
        self.assertFalse(missing.necessary_condition_satisfied)
        self.assertEqual(
            len(missing.missing_nonzero_target_coefficients), 1
        )
        self.assertTrue(missing.exact_support_excluded)
        self.assertFalse(missing.individual_target_nonimage_proved)

        zero_target = ColoringTarget.from_sparse(4, 2, ())
        singleton = target_support_profile(
            tensor_map, zero_target, support
        )
        self.assertEqual(
            len(singleton.singleton_zero_target_coefficients), 1
        )
        self.assertTrue(singleton.exact_support_excluded)
        self.assertIn("Other supports remain", singleton.claim_boundary)
        self.assertIn(
            "target nonimage is not proved",
            singleton.claim_boundary,
        )

    def test_invariant_validation_and_dependencies_fail_closed(self):
        target = canonical_ghz_target(4, 2)
        with self.assertRaises(KrennTensorInvariantError):
            rational_flattening(target, ())
        with self.assertRaises(KrennTensorInvariantError):
            rational_flattening(target, range(4))
        with self.assertRaises(KrennTensorInvariantError):
            rational_flattening(target, (0, 0))
        with self.assertRaises(KrennTensorInvariantError):
            target_support_profile(
                MatchingTensorMap(4, 2),
                target,
                (0, 0),
            )
        with self.assertRaises(KrennTensorInvariantError):
            target_support_profile(
                MatchingTensorMap(4, 2),
                canonical_ghz_target(4, 3),
                (),
            )

        import experiments.krenn_quantum_graph.tensor_invariants as subject

        source = Path(subject.__file__).read_text(encoding="utf-8")
        self.assertNotIn("support_search", source)
        self.assertNotIn("section12_instantiation", source)
        self.assertNotIn("artifacts", source)


if __name__ == "__main__":
    unittest.main()
