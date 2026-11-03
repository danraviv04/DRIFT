"""Paper KS panels must consume the residue estimator, never the profile field."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from drift import article_figures as article
from drift.ensemble_shift import _residue_ks_summaries


class TestArticleResidueKS(unittest.TestCase):
    def setUp(self):
        article.VALUES.clear()
        self.addCleanup(article.VALUES.clear)
        self.frame = pd.DataFrame({
            'gene': ['PTEN'] * 8, 'label': [0] * 4 + [1] * 4,
            'afm_score': np.linspace(.1, .9, 8),
            'd1_score': np.linspace(.01, .08, 8),
            'plddt_ks_residue': [.1, .2, .3, .6, .4, .7, .8, .9],
            # Opposite ordering exposes accidental use of the old estimator.
            'plddt_ks': np.linspace(.9, .1, 8),
        })

    def test_roc_and_violin_use_residue_scores(self):
        expected = roc_auc_score(self.frame.label, self.frame.plddt_ks_residue)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(article, 'save'):
            article.panel_roc(Path(tmp), self.frame, 'PTEN', 'roc', 'roc', scores=('ks',))
            article.panel_two_descriptor_violin(Path(tmp), self.frame, 'PTEN', 'violin', 'violin')
        values = {r['quantity']: r['value'] for r in article.VALUES}
        self.assertEqual(values['PTEN_KS_AUC'], expected)
        self.assertEqual(values['PTEN_Residue_KS_AUC'], expected)
        article.plt.close('all')

    def test_roc_no_longer_requires_profile_column(self):
        with mock.patch.object(article, 'save'):
            article.panel_roc(Path('.'), self.frame.drop(columns='plddt_ks'),
                              'PTEN', 'roc', 'roc', scores=('ks',))
        article.plt.close('all')

    def test_cached_estimator_matches_per_residue_scipy_definition(self):
        rng = np.random.default_rng(71)
        wt = pd.DataFrame(rng.integers(40, 100, size=(9, 12)))
        mutant = pd.DataFrame(rng.integers(40, 100, size=(9, 12)))
        raw, _ = _residue_ks_summaries(wt, mutant)
        expected = np.mean([ks_2samp(wt.iloc[r], mutant.iloc[r]).statistic for r in range(9)])
        self.assertAlmostEqual(raw, expected, places=14)

    def test_plot_loader_refuses_another_scientific_config(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(article, 'CACHE', Path(tmp)), \
                mock.patch.object(article, 'protein_config_fingerprint', return_value='corrected'):
            self.frame.to_csv(Path(tmp) / 'labeled_size90_seed42_ratio_of_means_proteinspooled_esfeat6.csv', index=False)
            with self.assertRaisesRegex(SystemExit, 'rebuild before plotting'):
                article.load_labeled(90, 42)
