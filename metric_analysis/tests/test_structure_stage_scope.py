"""Stage dispatch tests without raw structures, external drives or cache writes."""
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from metric_analysis.pipeline.orchestration import run_structure_metrics as runner


class TestStructureStageScope(unittest.TestCase):
    def cli(self, *args):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['run_structure_metrics', *args]), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            return runner.main()

    def test_subunit_allowlists_exclude_raw_extraction(self):
        # A subunit reads its parent complex's extracted pLDDT; raw extraction stays on the parent.
        for gene in ('MSH2', 'MSH6'):
            cfg = runner.get_protein_config(gene, runner.BASE_REQUIRED_KEYS)
            selected, unsupported = runner.partition_stages(cfg, runner.ALL_STAGES)
            self.assertEqual(unsupported, ['plddt'])
            self.assertEqual(selected, ['plddt_distribution'])

    def test_other_datasets_keep_all_stages(self):
        for gene in runner.supported_proteins(runner.BASE_REQUIRED_KEYS):
            if gene in ('MSH2', 'MSH6'):
                continue
            cfg = runner.get_protein_config(gene, runner.BASE_REQUIRED_KEYS)
            if gene in ('TP53', 'KRAS'):
                # No binary label since 2026-10-13: extraction only.
                self.assertEqual(runner.partition_stages(cfg, runner.ALL_STAGES),
                                 (['plddt'], runner.ALL_STAGES[1:]))
                continue
            self.assertEqual(runner.partition_stages(cfg, runner.ALL_STAGES),
                             (runner.ALL_STAGES, []))

    def test_explicit_unsupported_request_fails_before_any_protein_runs(self):
        with patch.object(runner, 'run_protein') as run:
            with self.assertRaises(SystemExit) as error:
                self.cli('--protein', 'POLE,MSH6', '--stage', 'plddt', '--force')
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_all_stages_selects_supported_subunit_stages(self):
        with patch.object(runner, 'run_protein', return_value={}) as run:
            self.assertEqual(self.cli('--protein', 'MSH2,MSH6', '--stage', 'all'), 0)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.args[1], ['plddt_distribution'])

    def test_all_proteins_extraction_request_skips_subunits(self):
        with patch.object(runner, 'run_protein', return_value={}) as run:
            self.assertEqual(self.cli('--protein', 'all', '--stage', 'plddt'), 0)
            genes = {call.args[0] for call in run.call_args_list}
            self.assertIn('MutS_alpha', genes)
            self.assertTrue(genes.isdisjoint({'MSH2', 'MSH6'}))

    def test_direct_call_cannot_bypass_boundary(self):
        with patch.object(runner, 'build_structure_metrics_config') as build:
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                runner.run_protein('MSH6', ['plddt'], force=True)
            build.assert_not_called()

    def test_supported_stage_runs_with_subunit_scope(self):
        def check_scope(cfg, config, state, force=False):
            self.assertEqual(runner.sm._PLDDT_ROW_BLOCK, (934, 2294))
            self.assertEqual(runner.sm._PLDDT_INCLUDE, 'MSH6')
        previous = (runner.sm._PLDDT_ROW_BLOCK, runner.sm._PLDDT_INCLUDE)
        with patch.dict(runner.STAGE_FUNCS, {'plddt_distribution': check_scope}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_protein('MSH6', ['plddt_distribution'])
        self.assertEqual(result['plddt_distribution'][0], 'DONE')
        self.assertEqual((runner.sm._PLDDT_ROW_BLOCK, runner.sm._PLDDT_INCLUDE), previous)


if __name__ == '__main__':
    unittest.main()
