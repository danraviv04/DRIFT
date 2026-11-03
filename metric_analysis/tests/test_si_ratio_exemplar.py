"""Regression checks for the shared manuscript exemplar sampler."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / 'experiments/si_figures/ratio_exemplar.py'
SPEC = importlib.util.spec_from_file_location('si_ratio_exemplar', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestRatioExemplarSampling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wt = np.arange(60, dtype=float).reshape(6, 10) + 20
        self.mut = self.wt + np.array([30, 30, 1, 2, 3, 4])[:, None]
        self.record = dict(protein='MSH6', mutation_code='A2V',
                           wt_source_file=str(Path(self.tmp.name) / 'wt.csv'),
                           source_file=str(Path(self.tmp.name) / 'mut.csv'))
        pd.DataFrame(self.wt).to_csv(self.record['wt_source_file'])
        pd.DataFrame(self.mut).to_csv(self.record['source_file'])

    def test_subunit_matches_canonical_and_numbers_locally(self):
        from canonical_analysis import analyse_variant
        rec = dict(self.record, plddt_row_range_start=2, plddt_row_range_stop=6,
                   label=1, mutation_type='Driver', label_provenance='test',
                   source_family='test')
        residues, w, m, n = MODULE.sampled_pair(rec, 7, 11)
        np.testing.assert_array_equal(residues, [1, 2, 3, 4])
        self.assertEqual((w.shape, m.shape, n), ((4, 7), (4, 7), 7))
        expected = analyse_variant(rec, sizes=(7,), seeds=(11,))[0]['d1']
        self.assertAlmostEqual(np.nanmean(MODULE.ratio_profile(w, m)[1]), expected, places=14)

    def test_unsplit_sampling_is_unchanged(self):
        rec = dict(self.record, protein='PTEN', plddt_row_range_start=np.nan,
                   plddt_row_range_stop=np.nan)
        residues, w, m, n = MODULE.sampled_pair(rec, 7, 11)
        wi = MODULE.stable_rng(11, 'PTEN', 'A2V', 'wt').permutation(10)[:7]
        mi = MODULE.stable_rng(11, 'PTEN', 'A2V', 'mutant').permutation(10)[:7]
        np.testing.assert_array_equal(w, self.wt[:, wi])
        np.testing.assert_array_equal(m, self.mut[:, mi])
        np.testing.assert_array_equal(residues, np.arange(1, 7))
        self.assertEqual(n, 7)

    def test_invalid_rows_do_not_shift_mutation_positions(self):
        self.mut[3, :] = np.nan
        pd.DataFrame(self.mut).to_csv(self.record['source_file'])
        rec = dict(self.record, plddt_row_range_start=2, plddt_row_range_stop=6)
        residues, _, _, _ = MODULE.sampled_pair(rec, 7, 11)
        np.testing.assert_array_equal(residues, [1, 3, 4])

    def test_out_of_bounds_block_is_rejected(self):
        rec = dict(self.record, plddt_row_range_start=2, plddt_row_range_stop=7)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            MODULE.sampled_pair(rec, 7, 11)
