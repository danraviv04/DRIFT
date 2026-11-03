import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from drift import ensemble_shift as es


class TestEnsembleSubunitScope(unittest.TestCase):
    def test_partner_changes_do_not_change_subunit_features(self):
        rng = np.random.default_rng(42)
        wt = rng.uniform(65, 90, (4, 8))
        mut = wt + rng.normal(size=(4, 8))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for gene, bounds in [('MSH2', (0, 2)), ('MSH6', (2, 4))]:
                def evaluate(a, b, row_range):
                    pd.DataFrame(a).to_csv(root / 'MutS_alpha_WT_plddt_values.csv')
                    pd.DataFrame(b).to_csv(root / f'MutS_alpha_{gene}_A1V_plddt_values.csv')
                    cfg = {gene: dict(mut='MutS_alpha', read_root=root,
                                     mut_csv_glob=f'*_{gene}_*plddt_values.csv',
                                     plddt_row_range=row_range)}
                    return es.build_ensemble_shift_features(
                        cfg, max_structures_per_variant=None, seed=42)[es.FEATURE_COLUMNS]
                expected = evaluate(wt[slice(*bounds)], mut[slice(*bounds)], None)
                actual = evaluate(wt, mut, bounds)
                poisoned = mut.copy()
                outside = np.ones(4, dtype=bool)
                outside[slice(*bounds)] = False
                poisoned[outside] = 1
                pd.testing.assert_frame_equal(expected, actual)
                pd.testing.assert_frame_equal(actual, evaluate(wt, poisoned, bounds))

    def test_numeric_labels_define_scope(self):
        frame = pd.DataFrame({'a': [1, 2, 3]}, index=['0', '2', '3'])
        self.assertEqual(es._apply_row_range(frame, (2, 4), 'test').index.tolist(), ['2', '3'])
        with self.assertRaises(ValueError):
            es._apply_row_range(frame, (4, 2), 'test')
        with self.assertRaises(ValueError):
            es._apply_row_range(frame.rename(index={'0': 'bad'}), (0, 4), 'test')
