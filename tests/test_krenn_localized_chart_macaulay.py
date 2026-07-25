import unittest

from experiments.krenn_quantum_graph.localized_chart_macaulay import (
    EXPECTED_ORBIT_COLUMNS,
    EXPECTED_STABILIZER_SIZES,
    KrennMacaulayError,
    bounded_koszul_orbit_relations,
    bounded_orbit_columns,
    bounded_pair_orbits,
    generator_characters,
    modular_bounded_macaulay_preflight,
    ordered_seed_stabilizer,
    residual_torus_characters,
)


class KrennLocalizedChartMacaulayTest(unittest.TestCase):
    def test_residual_torus_and_seed_stabilizers_replay_exactly(self):
        self.assertEqual(
            tuple(
                ordered_seed_stabilizer(index).order
                for index in range(8)
            ),
            EXPECTED_STABILIZER_SIZES,
        )
        for index in range(8):
            characters = residual_torus_characters(index)
            self.assertEqual(len(characters), 129)
            self.assertEqual(len(set(characters[:126])), 126)
            self.assertTrue(all(any(row) for row in characters[:126]))
            self.assertTrue(
                all(not any(row) for row in characters[126:])
            )
            self.assertEqual(len(generator_characters(index)), 729)

    def test_degree_four_and_five_orbit_column_censuses(self):
        for degree in (4, 5):
            self.assertEqual(
                tuple(
                    len(bounded_pair_orbits(index, degree))
                    for index in range(8)
                ),
                EXPECTED_ORBIT_COLUMNS[degree],
            )

    def test_natural_degree_six_has_exact_rank_upper_and_lower_match(self):
        result = modular_bounded_macaulay_preflight(
            6, 6, prime=1_009
        )
        self.assertEqual(result.raw_torus_pairs, 61_154)
        self.assertEqual(result.invariant_columns, 5_318)
        self.assertEqual(result.invariant_rows, 42_683)
        self.assertEqual(result.nonzero_entries, 77_840)
        self.assertEqual(result.exact_koszul_relations, 8)
        self.assertEqual(
            result.independent_koszul_relation_rank_mod_p, 8
        )
        self.assertEqual(result.source_rank_upper_bound_over_q, 5_310)
        self.assertEqual(result.source_rank_mod_p, 5_310)
        self.assertEqual(result.augmented_rank_mod_p, 5_311)
        self.assertFalse(result.target_in_span_mod_p)
        self.assertTrue(
            result.bounded_nonmembership_over_q_certified
        )
        self.assertFalse(
            result.to_dict()["claim_boundary"][
                "full_chart_unit_ideal_decided"
            ]
        )

    def test_every_retained_koszul_relation_replays_exactly(self):
        columns = bounded_orbit_columns(6, 6)
        with self.assertRaises(TypeError):
            columns[0][()] = 99
        relations = bounded_koszul_orbit_relations(6, 6)
        self.assertEqual(len(relations), 8)
        for relation in relations:
            replay = {}
            for column, multiplier in relation:
                for monomial, coefficient in columns[column].items():
                    replay[monomial] = (
                        replay.get(monomial, 0)
                        + multiplier * coefficient
                    )
            self.assertTrue(all(value == 0 for value in replay.values()))

        tampered = list(relations[0])
        column, coefficient = tampered[0]
        tampered[0] = (column, coefficient + 1)
        replay = {}
        for column, multiplier in tampered:
            for monomial, value in columns[column].items():
                replay[monomial] = (
                    replay.get(monomial, 0) + multiplier * value
                )
        self.assertTrue(any(replay.values()))

    def test_modular_claims_reject_bad_characteristics(self):
        with self.assertRaises(KrennMacaulayError):
            modular_bounded_macaulay_preflight(0, 4, prime=2)
        with self.assertRaises(KrennMacaulayError):
            modular_bounded_macaulay_preflight(0, 4, prime=15)


if __name__ == "__main__":
    unittest.main()
