from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.independent_verifier import (
    independent_perfect_matchings,
)
from experiments.krenn_quantum_graph.n8_toric_first_shell import (
    KrennN8ToricFirstShellError,
    build_n8_toric_first_shell_certificate,
    target_preserving_gauge_matrix,
    verify_n8_toric_first_shell_bundle,
    verify_n8_toric_first_shell_certificate,
    write_n8_toric_first_shell_bundle,
)
from experiments.krenn_quantum_graph.system import (
    canonical_edges,
    coloring_from_index,
    coloring_index,
    perfect_matchings,
    variable_key,
)


RESULTS_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "krenn_quantum_graph"
    / "n8_d3_toric_first_shell"
)


EXPECTED_HARD_CASES = {
    "H5": {
        "seed": [0, 19, 38],
        "stabilizer": 4,
        "victims": 2,
        "individual_repairs": 24,
        "raw_assignments": 144,
        "orbits": 43,
        "support_histogram": {"16": 144},
        "orbit_size_histogram": {"1": 2, "2": 11, "4": 30},
        "singleton_histogram": {
            "4": 8,
            "5": 20,
            "6": 75,
            "7": 13,
            "8": 20,
            "9": 8,
        },
        "minimum_singletons": 4,
        "best_orbits": 3,
        "slice_determinant": -1,
    },
    "H6": {
        "seed": [0, 19, 43],
        "stabilizer": 6,
        "victims": 3,
        "individual_repairs": 36,
        "raw_assignments": 1_728,
        "orbits": 304,
        "support_histogram": {"18": 1_728},
        "orbit_size_histogram": {
            "1": 2,
            "2": 5,
            "3": 22,
            "6": 275,
        },
        "singleton_histogram": {
            "6": 17,
            "7": 78,
            "8": 162,
            "9": 212,
            "10": 399,
            "11": 234,
            "12": 158,
            "13": 180,
            "14": 111,
            "15": 96,
            "16": 45,
            "17": 27,
            "20": 9,
        },
        "minimum_singletons": 6,
        "best_orbits": 4,
        "slice_determinant": 1,
    },
}


def _canonical_matching(matching):
    return tuple(
        sorted(
            (min(int(first), int(second)), max(int(first), int(second)))
            for first, second in matching
        )
    )


def _rank_over_q(matrix):
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
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
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(len(rows)):
            if row == rank or not rows[row][column]:
                continue
            coefficient = rows[row][column]
            rows[row] = [
                value - coefficient * pivot_entry
                for value, pivot_entry in zip(
                    rows[row], rows[rank], strict=True
                )
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _determinant(matrix):
    rows = [list(map(Fraction, row)) for row in matrix]
    result = Fraction(1)
    for column in range(len(rows)):
        pivot = next(
            (
                row
                for row in range(column, len(rows))
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            result = -result
        pivot_value = rows[column][column]
        result *= pivot_value
        rows[column] = [
            value / pivot_value for value in rows[column]
        ]
        for row in range(column + 1, len(rows)):
            coefficient = rows[row][column]
            if coefficient:
                rows[row] = [
                    value - coefficient * pivot_entry
                    for value, pivot_entry in zip(
                        rows[row], rows[column], strict=True
                    )
                ]
    return result


def _integer_matrix_product(left, right):
    return tuple(
        tuple(
            sum(
                left[row][middle] * right[middle][column]
                for middle in range(len(right))
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _active_term_signature(support, matching_enumerator):
    edge_variables = defaultdict(list)
    for index in support:
        first, second, first_color, second_color = variable_key(
            8, 3, int(index)
        )
        edge_variables[(first, second)].append(
            (int(index), first_color, second_color)
        )
    terms = defaultdict(list)
    for raw_matching in matching_enumerator:
        matching = _canonical_matching(raw_matching)
        choices = tuple(edge_variables[edge] for edge in matching)
        if any(not choice for choice in choices):
            continue
        for selected in product(*choices):
            coloring = [-1] * 8
            monomial = []
            for (first, second), (
                index,
                first_color,
                second_color,
            ) in zip(matching, selected, strict=True):
                coloring[first] = first_color
                coloring[second] = second_color
                monomial.append(index)
            terms[coloring_index(8, 3, coloring)].append(
                (matching, tuple(sorted(monomial)))
            )
    return {
        equation: tuple(sorted(records))
        for equation, records in terms.items()
    }


class KrennN8ToricFirstShellTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.certificate = build_n8_toric_first_shell_certificate()

    def test_target_preserving_gauge_and_unimodular_seed_slices(self):
        gauge = target_preserving_gauge_matrix()
        self.assertEqual((len(gauge), len(gauge[0])), (252, 21))
        self.assertEqual(_rank_over_q(gauge), 21)

        records = self.certificate["gauge_quotient"][
            "seed_normalizations"
        ]
        for label, expected in EXPECTED_HARD_CASES.items():
            record = records[label]
            support = tuple(record["seed_support"])
            restricted = tuple(gauge[index] for index in support)
            self.assertEqual(record["restriction_matrix_shape"], [12, 21])
            self.assertEqual(_rank_over_q(restricted), 9)
            self.assertEqual(record["restriction_rank_over_Q"], 9)
            self.assertEqual(
                record["normalizable_seed_order_subspace_dimension"], 9
            )
            self.assertEqual(
                record["residual_gauge_dimension_after_seed_normalization"],
                12,
            )
            self.assertTrue(
                record["three_matching_product_orders_are_invariant"]
            )

            slice_record = record["saturated_unimodular_slice"]
            minor = tuple(
                tuple(
                    restricted[row][column]
                    for column in slice_record["column_indices"]
                )
                for row in slice_record["row_indices"]
            )
            determinant = _determinant(minor)
            self.assertEqual(
                determinant, Fraction(expected["slice_determinant"])
            )
            self.assertEqual(
                slice_record["determinant"],
                expected["slice_determinant"],
            )
            self.assertFalse(slice_record["root_extraction_required"])

            for product_record in record[
                "monochromatic_matching_products"
            ]:
                character = tuple(
                    sum(
                        gauge[index][column]
                        for index in product_record[
                            "source_variable_indices"
                        ]
                    )
                    for column in range(21)
                )
                self.assertEqual(character, (0,) * 21)
                self.assertEqual(
                    product_record["gauge_character"], [0] * 21
                )

    def test_k8_projector_formula_identities_and_equivariance(self):
        edges = tuple(canonical_edges(8))
        matchings = tuple(
            _canonical_matching(matching)
            for matching in perfect_matchings(8)
        )
        independent = tuple(
            _canonical_matching(matching)
            for matching in independent_perfect_matchings(8)
        )
        self.assertEqual(len(matchings), 105)
        self.assertEqual(set(matchings), set(independent))

        incidence = tuple(
            tuple(int(edge in matching) for matching in matchings)
            for edge in edges
        )
        self.assertEqual((len(incidence), len(incidence[0])), (28, 105))
        self.assertEqual(_rank_over_q(incidence), 21)
        transpose = tuple(zip(*incidence, strict=True))
        gram = _integer_matrix_product(transpose, incidence)
        gram_squared = _integer_matrix_product(gram, gram)
        scale = 1_080
        projector = tuple(
            tuple(
                (scale if row == column else 0)
                - 78 * gram[row][column]
                + gram_squared[row][column]
                for column in range(105)
            )
            for row in range(105)
        )
        self.assertTrue(
            all(
                projector[row][column] == projector[column][row]
                for row in range(105)
                for column in range(105)
            )
        )
        self.assertFalse(
            any(
                value
                for row in _integer_matrix_product(incidence, projector)
                for value in row
            )
        )
        projector_squared = _integer_matrix_product(projector, projector)
        self.assertTrue(
            all(
                projector_squared[row][column]
                == scale * projector[row][column]
                for row in range(105)
                for column in range(105)
            )
        )
        self.assertEqual(
            sum(projector[index][index] for index in range(105)),
            scale * 84,
        )
        projector_hash = sha256(
            json.dumps(
                projector,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()

        record = self.certificate["fixed_coloring_matching_rotor"]
        self.assertEqual(record["edge_matching_incidence_shape"], [28, 105])
        self.assertEqual(record["edge_matching_incidence_rank_over_Q"], 21)
        self.assertEqual(record["rotor_kernel_dimension"], 84)
        self.assertEqual(record["integer_scale"], scale)
        self.assertEqual(record["scaled_projector_sha256"], projector_hash)
        self.assertTrue(all(record["exact_checks"].values()))

        matching_index = {
            matching: index for index, matching in enumerate(matchings)
        }
        for adjacent in range(7):
            vertices = list(range(8))
            vertices[adjacent], vertices[adjacent + 1] = (
                vertices[adjacent + 1],
                vertices[adjacent],
            )
            action = tuple(
                matching_index[
                    _canonical_matching(
                        (vertices[first], vertices[second])
                        for first, second in matching
                    )
                ]
                for matching in matchings
            )
            self.assertTrue(
                all(
                    projector[action[row]][action[column]]
                    == projector[row][column]
                    for row in range(105)
                    for column in range(105)
                )
            )

        interpretation = record["interpretation"]
        self.assertEqual(interpretation["all_apex_edge_marginals"], "q=E*m")
        self.assertEqual(interpretation["rotor_coordinates"], "h=P*m")
        self.assertFalse(
            interpretation["raw_all_apex_regrouping_is_injective"]
        )
        self.assertTrue(
            interpretation[
                "rotor_retains_matching_interference_lost_by_marginals"
            ]
        )
        for key in (
            "rotor_used_to_exclude_first_shell",
            "rotor_is_a_Rees_exceptional_coordinate",
            "rotor_or_displayed_circuits_form_a_tropical_basis",
            "rotor_proves_H5_H6_or_global_obstruction",
        ):
            self.assertFalse(interpretation[key])

    def test_exact_h5_h6_raw_orbit_support_and_spill_censuses(self):
        flat = self.certificate["flat_first_shell"]
        self.assertEqual(flat["total_raw_decorated_assignments"], 1_872)
        self.assertEqual(flat["total_decorated_stabilizer_orbits"], 347)
        self.assertTrue(
            flat["all_enumerated_exact_nonzero_supports_excluded"]
        )
        self.assertTrue(flat["all_enumerated_strata_excluded"])

        for label, expected in EXPECTED_HARD_CASES.items():
            record = flat["hard_cases"][label]
            self.assertEqual(
                record["representative_matching_indices"], expected["seed"]
            )
            self.assertEqual(
                record["seed_stabilizer_order"], expected["stabilizer"]
            )
            self.assertEqual(
                record["original_singleton_victim_count"],
                expected["victims"],
            )
            self.assertEqual(
                record["repair_cost_histogram_per_original_victim"],
                {"0": 1, "2": 12, "3": 32, "4": 60},
            )
            self.assertEqual(
                record["individual_decorated_minimal_repairs"],
                expected["individual_repairs"],
            )
            self.assertEqual(
                record["individual_repair_stabilizer_orbits"], 7
            )
            self.assertEqual(
                record["simultaneous_raw_decorated_assignments"],
                expected["raw_assignments"],
            )
            self.assertEqual(
                record["simultaneous_distinct_exact_supports"],
                expected["raw_assignments"],
            )
            self.assertEqual(
                record["simultaneous_decorated_stabilizer_orbits"],
                expected["orbits"],
            )
            self.assertEqual(
                record["raw_support_size_histogram"],
                expected["support_histogram"],
            )
            self.assertEqual(
                record["decorated_orbit_size_histogram"],
                expected["orbit_size_histogram"],
            )
            self.assertEqual(
                record["raw_singleton_spill_histogram"],
                expected["singleton_histogram"],
            )
            self.assertEqual(
                record["minimum_singleton_spills"],
                expected["minimum_singletons"],
            )
            self.assertEqual(
                record["orbits_attaining_minimum_singleton_spills"],
                expected["best_orbits"],
            )
            self.assertEqual(len(record["orbit_records"]), expected["orbits"])
            self.assertTrue(all(record["exact_checks"].values()))
            self.assertEqual(
                sum(
                    orbit["raw_decorated_orbit_size"]
                    for orbit in record["orbit_records"]
                ),
                expected["raw_assignments"],
            )

            targets = {
                coloring_index(8, 3, (color,) * 8)
                for color in range(3)
            }
            for orbit in record["orbit_records"]:
                initial = orbit["initial_form_replay"]
                support = set(initial["support"])
                histogram = initial[
                    "active_term_count_histogram_all_6561_equations"
                ]
                self.assertEqual(sum(histogram.values()), 6_561)
                self.assertTrue(initial["all_6561_equations_accounted"])
                self.assertTrue(
                    initial[
                        "primary_independent_matching_enumerators_agree"
                    ]
                )
                self.assertFalse(
                    initial["initial_stratum_can_map_to_projective_GHZ"]
                )
                self.assertGreater(
                    initial["zero_target_singleton_count"], 0
                )
                self.assertEqual(
                    len(initial["zero_target_singleton_initial_forms"]),
                    initial["zero_target_singleton_count"],
                )
                self.assertTrue(
                    orbit[
                        "stratum_excluded_by_singleton_initial_monomial"
                    ]
                )
                self.assertTrue(
                    orbit["exact_nonzero_support_stratum_excluded"]
                )
                self.assertFalse(
                    orbit["flat_valuation"][
                        "common_weight_projectivization_only"
                    ]
                )
                for singleton in initial[
                    "zero_target_singleton_initial_forms"
                ]:
                    self.assertNotIn(singleton["equation"], targets)
                    self.assertEqual(
                        singleton["coloring"],
                        list(
                            coloring_from_index(
                                8, 3, singleton["equation"]
                            )
                        ),
                    )
                    self.assertLessEqual(
                        set(singleton["monomial_variable_indices"]),
                        support,
                    )

    def test_dual_enumerator_replay_on_h5_and_h6_representatives(self):
        primary = tuple(perfect_matchings(8))
        independent = tuple(independent_perfect_matchings(8))
        for label in ("H5", "H6"):
            orbit = self.certificate["flat_first_shell"]["hard_cases"][
                label
            ]["orbit_records"][0]
            support = orbit["initial_form_replay"]["support"]
            primary_signature = _active_term_signature(support, primary)
            independent_signature = _active_term_signature(
                support, independent
            )
            self.assertEqual(primary_signature, independent_signature)

    def test_compactification_and_claim_boundaries_fail_closed(self):
        protocol = self.certificate["compactification_protocol"]
        self.assertIn("closure of graph", protocol["correct_border_object"])
        self.assertIn("Rees/blow-up", protocol["correct_border_object"])
        self.assertFalse(
            protocol["naive_weight_projectivization_is_sufficient"]
        )
        self.assertFalse(
            protocol[
                "equation_T_of_w_equals_h4_GHZ_at_h0_is_sufficient"
            ]
        )
        self.assertFalse(protocol["full_Rees_algebra_constructed_here"])
        self.assertFalse(protocol["full_graph_or_Rees_fiber_computed"])
        self.assertFalse(protocol["gauge_quotient_constructed_here"])
        self.assertFalse(
            protocol["coarse_GIT_quotient_alone_decides_orbit_closure"]
        )
        self.assertTrue(
            protocol[
                "one_parameter_subgroup_orbit_incidence_must_be_retained"
            ]
        )

        quotient = self.certificate["gauge_quotient"]
        self.assertEqual(quotient["cocharacter_dimension"], 21)
        self.assertEqual(
            quotient["common_Cstar_projectivization_dimension"], 1
        )
        self.assertFalse(
            quotient["Hopf_or_common_projectivization_removes_full_gauge"]
        )

        conclusion = self.certificate["conclusion"]
        self.assertEqual(
            conclusion["classification"],
            "exact-negative-flat-first-shell",
        )
        self.assertFalse(
            conclusion[
                "projective_GHZ_first_shell_graph_fiber_candidate_hit"
            ]
        )
        self.assertIn(
            "saturated full initial ideal",
            conclusion["next_acceptance_test"],
        )
        self.assertIn(
            "prevariety", conclusion["next_acceptance_test"]
        )

        boundary = self.certificate["claim_boundary"]
        scoped_positive_claims = {
            "enumerated_flat_cost2_H5_H6_strata_excluded",
            "enumerated_cost2_H5_H6_exact_supports_excluded",
        }
        for key in scoped_positive_claims:
            self.assertTrue(boundary[key])
        for key, value in boundary.items():
            if key in scoped_positive_claims:
                continue
            self.assertIs(
                value,
                False,
                msg=f"claim boundary {key!r} was promoted",
            )

    def test_certificate_is_isolated_and_corruption_is_rejected(self):
        self.assertEqual(
            verify_n8_toric_first_shell_certificate(self.certificate),
            self.certificate,
        )
        mutable = build_n8_toric_first_shell_certificate()
        mutable["claim_boundary"]["n8_nonexistence_proved"] = True
        self.assertFalse(
            build_n8_toric_first_shell_certificate()["claim_boundary"][
                "n8_nonexistence_proved"
            ]
        )

        corruptions = []
        escalated = deepcopy(self.certificate)
        escalated["claim_boundary"]["n8_nonexistence_proved"] = True
        corruptions.append(escalated)

        coarse_quotient = deepcopy(self.certificate)
        coarse_quotient["compactification_protocol"][
            "naive_weight_projectivization_is_sufficient"
        ] = True
        corruptions.append(coarse_quotient)

        forged_projector = deepcopy(self.certificate)
        forged_projector["fixed_coloring_matching_rotor"][
            "rotor_kernel_dimension"
        ] = 83
        corruptions.append(forged_projector)

        forged_shell = deepcopy(self.certificate)
        forged_shell["flat_first_shell"]["hard_cases"]["H6"][
            "minimum_singleton_spills"
        ] = 0
        corruptions.append(forged_shell)

        for payload in corruptions:
            with self.assertRaisesRegex(
                KrennN8ToricFirstShellError, "differs from exact replay"
            ):
                verify_n8_toric_first_shell_certificate(payload)
        with self.assertRaisesRegex(
            KrennN8ToricFirstShellError, "must be a mapping"
        ):
            verify_n8_toric_first_shell_certificate([])

    def test_bundle_round_trip_readme_and_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            written = write_n8_toric_first_shell_bundle(directory)
            self.assertEqual(
                verify_n8_toric_first_shell_bundle(directory), written
            )
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {"certificate.json", "README.md", "manifest.json"},
            )
            readme_path = directory / "README.md"
            readme = readme_path.read_text(encoding="ascii")
            self.assertIn("graph compactification", readme)
            self.assertIn("Rees/blow-up data", readme)
            self.assertIn("not an `n=8` existence or", readme)
            self.assertIn("nonexistence proof", readme)
            self.assertIn(
                "Different minimal layers, negative quotient directions",
                readme,
            )

            readme_path.write_text(
                readme + "\ncoarse GIT quotient decides the border\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                KrennN8ToricFirstShellError, "README changed"
            ):
                verify_n8_toric_first_shell_bundle(directory)

            write_n8_toric_first_shell_bundle(directory)
            certificate_path = directory / "certificate.json"
            certificate_text = certificate_path.read_text(encoding="ascii")
            certificate_path.write_text(
                certificate_text.replace(
                    "{\n",
                    '{\n  "schema": "duplicate",\n',
                    1,
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                KrennN8ToricFirstShellError, "duplicate JSON key"
            ):
                verify_n8_toric_first_shell_bundle(directory)

            write_n8_toric_first_shell_bundle(directory)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            manifest["claim_boundary"]["n8_nonexistence_proved"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                KrennN8ToricFirstShellError,
                "manifest or source ledger changed",
            ):
                verify_n8_toric_first_shell_bundle(directory)

            write_n8_toric_first_shell_bundle(directory)
            (directory / "unexpected.cache").write_text(
                "not part of the exact bundle", encoding="ascii"
            )
            with self.assertRaisesRegex(
                KrennN8ToricFirstShellError,
                "unexpected first-shell artifact files",
            ):
                write_n8_toric_first_shell_bundle(directory)

    def test_committed_bundle_verifies(self):
        verify_n8_toric_first_shell_bundle(RESULTS_DIRECTORY)


if __name__ == "__main__":
    unittest.main()
