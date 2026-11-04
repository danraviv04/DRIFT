"""The cached KS-score is the residue estimator: the mean per-residue two-sample KS."""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from drift.ensemble_shift import _residue_ks_summaries


class TestResidueKS(unittest.TestCase):
    def test_cached_estimator_matches_per_residue_scipy_definition(self):
        rng = np.random.default_rng(71)
        wt = pd.DataFrame(rng.integers(40, 100, size=(9, 12)))
        mutant = pd.DataFrame(rng.integers(40, 100, size=(9, 12)))
        raw, _ = _residue_ks_summaries(wt, mutant)
        expected = np.mean([ks_2samp(wt.iloc[r], mutant.iloc[r]).statistic for r in range(9)])
        self.assertAlmostEqual(raw, expected, places=14)


if __name__ == "__main__":
    unittest.main()
