"""Verify training boundaries, not just fold counts, across evaluation routes."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from drift import benchmark, run
from metric_analysis.pipeline.library.canonical_cohort import logo_splits, logo_fold_group


def frame(genes=('MSH2', 'MSH6', 'POLE', 'PTEN', 'VHL')):
    rng = np.random.default_rng(19)
    rows = []
    for gene in genes:
        for i in range(6):
            rows.append(dict(gene=gene, mutation=f'A{i+1}V', label=i % 2,
                             afm_score=rng.uniform(.1,.9), d1_score=rng.uniform(.01,.1)))
    return pd.DataFrame(rows).sort_values(['gene','mutation']).reset_index(drop=True)


class TestFoldPolicy(unittest.TestCase):
    def assert_boundary(self, genes, tr, va, mode='grouped'):
        train = {logo_fold_group(g, mode) for g in genes[tr]}
        validation = {logo_fold_group(g, mode) for g in genes[va]}
        self.assertTrue(train.isdisjoint(validation))
        self.assertEqual(len(set(genes[va])), 1)

    def test_grouped_excludes_partner_separate_retains_it(self):
        genes = frame().gene.to_numpy()
        for mode in ('grouped','separate'):
            for tr, va in logo_splits(genes, mode):
                self.assert_boundary(genes,tr,va,mode)
                if genes[va][0] == 'MSH2':
                    self.assertEqual('MSH6' in genes[tr], mode == 'separate')
        with self.assertRaises(ValueError): logo_splits(['MSH2','MSH6'])
        with self.assertRaises(ValueError): logo_splits(genes,'typo')

    def test_fixed_and_nested_fit_boundaries(self):
        df = frame(); seen=[]
        def predict(df, represented, y, genes, tr, va, spec, seed):
            self.assert_boundary(genes,tr,va)
            seen.append((set(genes[tr]),set(genes[va])))
            return df.afm_score.to_numpy()[va]
        spec=benchmark.BenchmarkSpec('raw2','variant','original',1.)
        with patch.object(benchmark,'_fit_predict',side_effect=predict):
            fixed=benchmark.evaluate_fixed_grid(df,[spec],90,42,42)
            nested, selections=benchmark.evaluate_nested_selection(df,[spec],90,42,42)
        self.assertGreater(len(seen),10)
        self.assertTrue({'MSH2','MSH6'}.issubset(set(fixed.scope)))
        pair=selections[selections.outer_gene.isin(['MSH2','MSH6'])]
        self.assertEqual(set(pair.outer_fold_group),{'MutS_alpha'})
        self.assertEqual(pair.candidate_id.nunique(),1)
        self.assertEqual(pair.inner_macro_auc.nunique(),1)

    def test_nested_requires_four_groups_not_four_dataset_names(self):
        df=frame(('MSH2','MSH6','POLE','PTEN'))
        spec=benchmark.BenchmarkSpec('raw2','variant','original',1.)
        with self.assertRaisesRegex(ValueError,'independent fold groups'):
            benchmark.evaluate_nested_selection(df,[spec],90,42,42)
        benchmark.evaluate_nested_selection(df,[spec],90,42,42,fold_grouping='separate')

    def test_generic_fusion_and_structure_only_use_grouped_training(self):
        df=frame(); genes=df.gene.to_numpy(); seen=[]
        original=run._impute_extras
        def impute(extras,tr,va):
            self.assert_boundary(genes,tr,va); seen.append(len(tr))
            return original(extras,tr,va)
        def train(x,y,xv,yv,**kw): return {'val_probs':np.full(len(yv),.5)}
        cfg=run._resolve_run_config({'epochs':1})
        with patch.object(run,'_impute_extras',side_effect=impute), \
             patch.object(run,'train_one_fold',side_effect=train):
            a=run.logo_cv(df,cfg)
            b=run.logo_cv_plain(df,cfg,['d1_score'])
        self.assertEqual(len(seen),10)
        self.assertEqual(set(a['per_gene'].gene),set(df.gene))
        self.assertEqual(set(b['per_gene'].gene),set(df.gene))

    def test_singleton_groups_preserve_predictions_and_selection_se(self):
        df=frame(('A','B','C','D')); spec=benchmark.BenchmarkSpec('raw2','variant','original',1.)
        a=benchmark.evaluate_fixed_grid(df,[spec],90,42,42)
        b=benchmark.evaluate_fixed_grid(df,[spec],90,42,42,fold_grouping='separate')
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        values=np.array([.1,.3,-.2,.4])
        self.assertAlmostEqual(benchmark._grouped_mean_se(values,['A','B','C','D']),
                               np.std(values,ddof=1)/2)
        self.assertAlmostEqual(benchmark._grouped_mean_se([1,1,-1,-1],['A','A','B','B']),1.)


if __name__=='__main__': unittest.main()
