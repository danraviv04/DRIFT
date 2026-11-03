import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "ks_meanfree" / "canonical_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("canonical_ks_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestCanonicalMeanFreeAnalysis(unittest.TestCase):
    def test_ks_is_tie_aware_and_symmetric(self):
        a = np.array([[0, 0, 1, 1], [2, 2, 2, 2]], dtype=float)
        b = np.array([[0, 1, 1, 1], [2, 2, 3, 3]], dtype=float)
        self.assertEqual(MODULE.ks_mean_equal(a, b), MODULE.ks_mean_equal(b, a))
        self.assertTrue(np.isclose(MODULE.ks_mean_equal(a, b), (0.25 + 0.5) / 2))

    def test_centred_ks_removes_pure_location_shift(self):
        wt = np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=float)
        mutant = wt + np.array([[10], [-3]], dtype=float)
        values = MODULE.descriptor_values(wt, mutant)
        self.assertGreater(values["ks_raw"], 0)
        self.assertTrue(np.isclose(values["ks_centred"], 0))
        self.assertGreater(values["d1"], 0)

    def test_centred_ks_retains_shape_change_with_equal_means(self):
        wt = np.array([[-1, -1, 1, 1], [-2, -1, 1, 2]], dtype=float) + 10
        mutant = np.array([[-2, 0, 0, 2], [-3, 0, 0, 3]], dtype=float) + 10
        self.assertTrue(np.allclose(wt.mean(axis=1), mutant.mean(axis=1)))
        values = MODULE.descriptor_values(wt, mutant)
        self.assertTrue(np.isclose(values["d1"], 0))
        self.assertGreater(values["ks_centred"], 0)
        self.assertGreater(values["sd_ratio"], 0)

    def test_variant_sampling_is_order_independent(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            wt = np.arange(60, dtype=float).reshape(6, 10)
            mutant = wt + np.linspace(0.1, 0.6, 6)[:, None]
            wt_path = tmp_path / "WT.csv"
            mutant_path = tmp_path / "A1V.csv"
            pd.DataFrame(wt).to_csv(wt_path)
            pd.DataFrame(mutant).to_csv(mutant_path)
            record = {
                "protein": "X", "mutation_code": "A1V", "mutation_type": "Driver",
                "label": 1, "label_provenance": "test", "source_family": "test",
                "source_file": str(mutant_path), "wt_source_file": str(wt_path),
            }
            first = MODULE.analyse_variant(record, sizes=(3, 7), seeds=(11, 12))
            second = MODULE.analyse_variant(record, sizes=(7, 3), seeds=(11, 12))
            key = lambda row: (row["seed"], row["requested_size"])
            self.assertEqual(
                {key(r): r["d1"] for r in first},
                {key(r): r["d1"] for r in second},
            )

    def test_uncertainty_tables_preserve_mean_seed_auc_endpoint(self):
        import pandas as pd
        from sklearn.metrics import roc_auc_score

        rows = []
        labels = [0, 0, 1, 1]
        scores = {
            0: [0.1, 0.4, 0.3, 0.9],
            1: [0.8, 0.2, 0.7, 0.6],
        }
        for seed_index, values in scores.items():
            for i, (label, value) in enumerate(zip(labels, values)):
                rows.append({
                    "protein": "POLE", "mutation_code": f"A{i + 1}V",
                    "label": label, "requested_size": 10,
                    "seed_index": seed_index, "seed": 100 + seed_index,
                    "d1": value, "ks_raw": value, "ks_centred": value,
                    "sd_ratio": value,
                })
        frame = pd.DataFrame(rows)
        expected = np.mean([
            roc_auc_score(labels, scores[0]), roc_auc_score(labels, scores[1])
        ])
        bootstrap = MODULE.bootstrap_summary(frame, reps=10, seed=7)
        permutation = MODULE.permutation_summary(frame, reps=20, seed=7)
        self.assertAlmostEqual(
            float(bootstrap.query("metric == 'ks_centred'")["observed_auc"].iloc[0]),
            expected,
        )
        self.assertAlmostEqual(
            float(permutation.query("metric == 'ks_centred'")["observed_auc"].iloc[0]),
            expected,
        )

    def test_hierarchical_bootstrap_preserves_class_sample_sizes(self):
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.arange(10, dtype=float).reshape(2, 5)

        def checked_auc(y, score):
            self.assertEqual(len(y), len(labels))
            self.assertEqual(int(np.sum(y == 0)), 3)
            self.assertEqual(int(np.sum(y == 1)), 2)
            self.assertEqual(len(score), len(labels))
            return 0.5

        with mock.patch.object(MODULE, "fast_auc", side_effect=checked_auc):
            draws = MODULE.hierarchical_bootstrap_auc(
                labels, scores, reps=4, rng=np.random.default_rng(17)
            )
        np.testing.assert_array_equal(draws, np.full(4, 0.5))

    def test_wt_null_comparison_reports_both_classes_separately(self):
        import pandas as pd

        rows = []
        for seed_index in (0, 1):
            for mutation_code, label, value in (("A1V", 1, 0.8), ("A2V", 0, 0.4)):
                rows.append({
                    "protein": "POLE", "mutation_code": mutation_code,
                    "label": label, "requested_size": 10,
                    "seed_index": seed_index, "d1": value, "ks_raw": value,
                    "ks_centred": value, "sd_ratio": value,
                })
        per_variant = pd.DataFrame(rows)
        wt_null = pd.DataFrame({
            "protein": ["POLE"] * 3, "requested_size": [10] * 3,
            "d1": [0.1, 0.2, 0.3], "ks_raw": [0.1, 0.2, 0.3],
            "ks_centred": [0.1, 0.2, 0.3], "sd_ratio": [0.1, 0.2, 0.3],
        })
        result = MODULE.compare_observed_to_wt_null(per_variant, wt_null)
        self.assertEqual(set(result["variant_group"]), {"all", "driver", "passenger"})
        self.assertEqual(len(result), 2 * 4 * 3)
        centred = result[result["metric"] == "ks_centred"]
        self.assertTrue(centred[centred["variant_group"] == "driver"]["n_variants"].eq(1).all())
        self.assertTrue(centred[centred["variant_group"] == "passenger"]["n_variants"].eq(1).all())


if __name__ == "__main__":
    unittest.main()
