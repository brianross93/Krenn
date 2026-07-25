import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.krenn_quantum_graph.localized_chart_ideals import (
    CHART_GENERATOR_COUNT,
    CHART_TERM_COUNT,
    CHART_VARIABLE_COUNT,
    EXPECTED_CORE_UNION_SIZES,
    EXPECTED_DEFECT_COUNTS,
    EXPECTED_DISJOINT_CORE_COUNTS,
    EXPECTED_SEED_SUPPORTS,
    KrennLocalizedChartError,
    localized_chart_cover_audit,
    localized_chart_germ_audit,
    normalized_seed_chart,
    normalized_seed_charts,
    singular_chart_script,
    verify_localized_chart_cover_audit,
    write_singular_chart_export,
)
from experiments.krenn_quantum_graph.localized_chart_independent import (
    independent_chart_fingerprint,
    independent_evaluate_chart,
)


class KrennLocalizedChartIdealsTest(unittest.TestCase):
    def test_eight_charts_cover_all_ordered_pure_seeds(self):
        charts = normalized_seed_charts()
        self.assertEqual(len(charts), 8)
        self.assertEqual(
            sum(chart.ordered_seed_orbit_size for chart in charts),
            3_375,
        )
        self.assertEqual(
            tuple(chart.fixed_weight_indices for chart in charts),
            EXPECTED_SEED_SUPPORTS,
        )
        for chart in charts:
            self.assertEqual(len(chart.fixed_weight_indices), 9)
            self.assertEqual(len(chart.remaining_weight_indices), 126)
            self.assertEqual(len(chart.generators), CHART_GENERATOR_COUNT)
            self.assertEqual(chart.term_count, CHART_TERM_COUNT)
            self.assertEqual(chart.maximum_degree, 4)
            self.assertEqual(
                tuple(
                    polynomial.term_count
                    for polynomial in chart.pure_localizers
                ),
                (16, 16, 16),
            )

    def test_independent_enumerator_replays_every_chart_bit_for_bit(self):
        for orbit_index, chart in enumerate(normalized_seed_charts()):
            self.assertEqual(
                chart.fingerprint(),
                independent_chart_fingerprint(orbit_index),
            )

    def test_primary_and_independent_exact_evaluation_agree(self):
        values = tuple(
            Fraction((index % 11) - 5, (index % 7) + 1)
            for index in range(CHART_VARIABLE_COUNT)
        )
        for orbit_index in range(8):
            chart = normalized_seed_chart(orbit_index)
            primary = tuple(
                polynomial.evaluate(values)
                for polynomial in chart.generators
            )
            self.assertEqual(
                primary,
                independent_evaluate_chart(orbit_index, values),
            )

    def test_cover_audit_states_exact_equivalence_and_fails_closed(self):
        audit = localized_chart_cover_audit()
        self.assertTrue(
            audit["open_locus"][
                "equivalent_to_direct_GHZ_membership"
            ]
        )
        self.assertTrue(
            audit["gauge_slice"]["all_selected_weights_fixed_to_one"]
        )
        self.assertFalse(
            audit["gauge_slice"][
                "preserves_literal_pure_coefficients_one"
            ]
        )
        self.assertFalse(
            audit["cover"]["symmetry_related_weights_equated"]
        )
        self.assertTrue(
            audit["cover"]["independent_chart_reconstruction"][
                "replayed"
            ]
        )
        self.assertTrue(
            audit["exact_checks"][
                "all_independent_chart_fingerprints_match"
            ]
        )
        self.assertEqual(
            audit["claim_boundary"][
                "finite_affine_GHZ_membership_status"
            ],
            "undecided",
        )
        self.assertEqual(
            verify_localized_chart_cover_audit(audit), audit
        )
        corrupted = copy.deepcopy(audit)
        corrupted["claim_boundary"][
            "finite_affine_GHZ_membership_status"
        ] = "nonexistent"
        with self.assertRaises(KrennLocalizedChartError):
            verify_localized_chart_cover_audit(corrupted)
        type_corrupted = copy.deepcopy(audit)
        type_corrupted["open_locus"][
            "equivalent_to_direct_GHZ_membership"
        ] = 1
        with self.assertRaises(KrennLocalizedChartError):
            verify_localized_chart_cover_audit(type_corrupted)

    def test_every_seed_germ_has_exact_universal_repair_core(self):
        audit = localized_chart_germ_audit()
        rows = audit["charts"]
        self.assertEqual(
            tuple(row["defect_count"] for row in rows),
            EXPECTED_DEFECT_COUNTS,
        )
        self.assertEqual(
            tuple(
                row["maximum_pairwise_disjoint_defect_cores"]
                for row in rows
            ),
            EXPECTED_DISJOINT_CORE_COUNTS,
        )
        self.assertEqual(
            tuple(row["repair_core_union_size"] for row in rows),
            EXPECTED_CORE_UNION_SIZES,
        )
        for row in rows:
            for defect in row["defects"]:
                self.assertEqual(defect["quadratic_repair_terms"], 6)
                self.assertEqual(defect["cubic_repair_terms"], 8)
                self.assertEqual(
                    len(defect["repair_core_local_variables"]), 12
                )
        interval = audit["universal_defect_equation"][
            "positive_root_isolating_interval"
        ]
        lower = Fraction(*interval["lower"])
        upper = Fraction(*interval["upper"])
        polynomial = lambda value: 8 * value**3 + 6 * value**2 - 1
        self.assertLess(polynomial(lower), 0)
        self.assertGreater(polynomial(upper), 0)
        self.assertTrue(
            audit["claim_boundary"][
                "excludes_only_a_small_neighborhood_of_each_seed_germ"
            ]
        )
        self.assertFalse(
            audit["claim_boundary"]["excludes_entire_chart"]
        )

    def test_singular_export_uses_sparse_unshifted_inverse_form(self):
        script = singular_chart_script(
            6, characteristic=31, algorithm="std"
        )
        self.assertIn("ring r=31,", script)
        self.assertIn("ideal G=std(I);", script)
        self.assertIn("poly unit_remainder=reduce(1,G);", script)
        self.assertIn("The inverse variables are unshifted", script)
        self.assertNotIn("shifted pure inverses", script)
        self.assertNotIn("(1+x126)", script)
        with self.assertRaises(KrennLocalizedChartError):
            singular_chart_script(0, characteristic=1)
        with self.assertRaises(KrennLocalizedChartError):
            singular_chart_script(0, characteristic=4)
        with self.assertRaises(KrennLocalizedChartError):
            singular_chart_script(0, algorithm="mystery")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(KrennLocalizedChartError):
                write_singular_chart_export(
                    Path(temporary),
                    characteristics=(31, 31),
                )
            with self.assertRaises(KrennLocalizedChartError):
                write_singular_chart_export(
                    Path(temporary),
                    characteristics=(),
                )
            with self.assertRaises(KrennLocalizedChartError):
                write_singular_chart_export(
                    Path(temporary),
                    characteristics=(),
                    algorithm="mystery",
                )

    def test_scratch_export_manifest_hashes_every_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = write_singular_chart_export(
                output,
                characteristics=(31, 0),
                algorithm="slimgb",
            )
            self.assertEqual(manifest["characteristics"], [31, 0])
            self.assertEqual(len(manifest["files"]), 32)
            self.assertTrue(
                manifest["claim_boundary"][
                    "positive_characteristic_is_reconnaissance_only"
                ]
            )
            for row in manifest["files"]:
                path = output / row["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    row["sha256"],
                )
            stored = json.loads(
                (output / "export_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stored, manifest)
            with self.assertRaises(KrennLocalizedChartError):
                write_singular_chart_export(
                    output,
                    characteristics=(31,),
                    algorithm="slimgb",
                )


if __name__ == "__main__":
    unittest.main()
