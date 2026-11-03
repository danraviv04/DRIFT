"""Shared numerical definitions, caller contracts, and the N>=1 KS definition."""
import os
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import ks_2samp

from drift.metrics import REGISTRY, descriptor_values, residue_ks, residue_ks_statistics


class TestKSMatchesSciPy(unittest.TestCase):
    def test_randomized_backend_equivalence(self):
        rng = np.random.default_rng(712)
        with patch.dict(os.environ, {}, clear=True):
            for i in range(200):
                r, n, m = rng.integers(1, 15), rng.integers(2, 91), rng.integers(2, 91)
                a, b = rng.normal(70, 10, (r, n)), rng.normal(70, 10, (r, m))
                if i % 3:
                    a, b = np.round(a, i % 3 - 1), np.round(b, i % 3 - 1)
                with self.subTest(shape=(r, n, m), case=i):
                    expected = np.array([ks_2samp(x, y).statistic for x, y in zip(a, b)])
                    np.testing.assert_allclose(residue_ks_statistics(a, b, backend="fast"), expected, atol=1e-12, rtol=0)
                    for metric in REGISTRY.values():
                        if "scipy" in metric.backends:
                            self.assertAlmostEqual(metric(a, b, backend="fast"), metric(a, b, backend="scipy"), places=12)

    def test_minimum_samples_and_missing_rows(self):
        a = np.array([[1, 2, np.nan], [1, np.nan, np.inf], [2, 3, 4]])
        b = np.array([[1, 3, 4], [2, 3, 4], [2, 3, 4]])
        # Row 1 keeps one finite WT value; KS is defined from a single observation.
        expected = [ks_2samp([1, 2], b[0]).statistic, ks_2samp([1], b[1]).statistic, 0]
        for backend in ("auto", "fast", "scipy"):
            np.testing.assert_allclose(residue_ks_statistics(a, b, backend=backend), expected, equal_nan=True)
            # N=1: rows (1,1), (1,2), (2,2) -> discordance rate 1/3.
            self.assertAlmostEqual(residue_ks(a[:, :1], b[:, :1], backend=backend), 1 / 3)
            self.assertTrue(np.isnan(residue_ks(a[:, :1], b[:, :1], backend=backend, min_samples=2)))
            self.assertTrue(np.isnan(residue_ks(a[:, :0], b[:, :0], backend=backend)))
            self.assertTrue(np.isnan(residue_ks(a[:0], b[:0], backend=backend)))

    def test_all_callers_define_ks_at_n1_as_discordance_rate(self):
        """At N=1 every KS path returns the fraction of residues whose pLDDTs differ."""
        from metric_analysis.pipeline.library.ensemble_aware_pipeline import compute_plddt_ks_for_selected_columns
        from metric_analysis.pipeline.library.plddt_ks import compute_per_residue_plddt_ks
        from drift.ensemble_shift import _residue_ks_summaries
        from analyses.msh6_case_study.ks_meanfree.canonical_analysis import ks_mean_equal
        a = np.array([[70.], [80.], [90.], [60.]])
        b = np.array([[70.], [81.], [95.], [60.]])
        rate = 0.5
        values = descriptor_values(a, b)
        self.assertAlmostEqual(values["d1"], np.mean(np.abs(b[:, 0] / a[:, 0] - 1)))
        self.assertAlmostEqual(values["ks_raw"], rate)
        self.assertEqual(values["ks_centred"], 0.0)
        self.assertAlmostEqual(ks_mean_equal(a, b), rate)
        self.assertAlmostEqual(compute_plddt_ks_for_selected_columns(a, b, [0], [0])["plddt_ks_mean"], rate)
        residue = compute_per_residue_plddt_ks(a, b)
        np.testing.assert_array_equal(residue.ks_statistic, [0, 1, 1, 0])
        # Defined, but the test has no power at n = m = 1: every p-value is 1.
        np.testing.assert_array_equal(residue.ks_pvalue, 1.0)
        raw, centred = _residue_ks_summaries(a, b)
        self.assertAlmostEqual(raw, rate)
        self.assertEqual(centred, 0.0)

    def test_environment_forces_reference(self):
        from metric_analysis.pipeline.library import descriptor_metrics as m
        a = np.array([[1, 2], [2, 3]])
        with patch.dict(os.environ, {"DRIFT_KS_BACKEND": "scipy"}), patch.object(m.stats, "ks_2samp", wraps=ks_2samp) as reference:
            residue_ks(a, a, backend="fast")
            self.assertEqual(reference.call_count, 2)
        with self.assertRaises(ValueError):
            residue_ks(a, a, backend="typo")
        with self.assertRaises(ValueError):
            residue_ks(a[0], a[0])

    def test_inference_preserves_one_sided_pvalues_and_fdr(self):
        from metric_analysis.pipeline.library.plddt_ks import compute_per_residue_plddt_ks
        a, b = np.array([[1, 2, 3.]]), np.array([[3, 4, 5.]])
        expected = ks_2samp(a[0], b[0], alternative="greater", method="exact")
        got = compute_per_residue_plddt_ks(a, b, alternative="greater", method="exact").iloc[0]
        self.assertEqual(got.ks_statistic, expected.statistic)
        self.assertEqual(got.ks_pvalue, expected.pvalue)
        self.assertEqual(got.ks_qvalue, expected.pvalue)


class TestUnavailableMetricScoring(unittest.TestCase):
    def test_canonical_scoring_and_uncertainty_keep_unavailable_metrics_missing(self):
        import pandas as pd
        from analyses.msh6_case_study.ks_meanfree import canonical_analysis as analysis
        rows = []
        for seed in (0, 1):
            for i, label in enumerate((0, 0, 1, 1)):
                rows.append(dict(protein="POLE", mutation_code=f"M{i}", label=label,
                                 requested_size=1, seed_index=seed, seed=seed,
                                 d1=i / 10, ks_raw=np.nan, ks_centred=np.nan, sd_ratio=np.nan))
        frame = pd.DataFrame(rows)
        scored = analysis.score_gene_seeds(frame)
        self.assertTrue(scored.auc_ks_raw.isna().all())
        self.assertTrue(scored.auc_combo_d1_ks_raw.isna().all())
        self.assertTrue(scored.auc_d1.eq(1).all())
        bootstrap = analysis.bootstrap_summary(frame, reps=3, seed=2)
        self.assertTrue(bootstrap.loc[bootstrap.metric.eq("ks_raw"), "observed_auc"].isna().all())
        permutation = analysis.permutation_summary(frame, reps=3, seed=2)
        self.assertTrue(permutation.loc[permutation.metric.eq("ks_raw"), "p_one_sided"].isna().all())

    def test_nonlearned_sweep_keeps_only_available_arms(self):
        import pandas as pd
        from analyses.annotation.simple_fusion_benchmark import size_sweep
        frame = pd.DataFrame(dict(gene=["A"] * 4, label=[0, 0, 1, 1],
                                  afm_score=[.1, .2, .8, .9], d1_score=[.01, .02, .08, .09],
                                  plddt_ks_residue=np.nan, plddt_ks_centred=np.nan))
        folds = {"LOGO": [(np.array([0, 2]), np.array([1, 3])),
                           (np.array([1, 3]), np.array([0, 2]))]}
        records = size_sweep.cell_scores(frame, folds, ("LOGO",))
        self.assertEqual({r["arm"] for r in records}, {"AM", "AM+D1", "afm_score", "d1_score"})
        self.assertTrue(all(np.isfinite(r["score"]).all() for r in records))


if __name__ == "__main__":
    unittest.main()
