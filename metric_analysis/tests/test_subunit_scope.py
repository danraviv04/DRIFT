#!/usr/bin/env python3
"""Subunit datasets in the ensemble-aware sweeps, and coordinate-chunk freshness (2026-10-15).

Regression tests for two bugs found while redoing the coordinate metrics in the core-fit
frame:

- run_d1_ensemble_sweep read the parent's whole pLDDT folder with all its rows. For MSH2
  (plddt_dir = PLDDTs/MutS_alpha) that meant both chains of every matrix plus the 42
  superseded MSH6 variants filed there; plddt_row_range and cohort_subunit_token were
  honoured by structure_metrics but not by the sweep's own loader.
- coord_chunks (copies of the coordinate cache) were reused even under --force, so a
  cache rebuilt in another frame would have been swept from the old chunks.

    PYTHONPATH="." python -m unittest metric_analysis.tests.test_subunit_scope -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metric_analysis.pipeline.library import ensemble_aware_pipeline as eap  # noqa: E402
from metric_analysis.pipeline.orchestration import run_ensemble_sweep_parallel as engine  # noqa: E402


def write_matrix(path, rows_a, rows_b, n_samples=4):
    """A two-subunit pLDDT table: 3 residues of subunit A, then 3 of subunit B."""
    values = np.r_[np.full(3, rows_a, float), np.full(3, rows_b, float)]
    frame = pd.DataFrame({f"seed-1_sample-{i}": values for i in range(n_samples)})
    frame.to_csv(path)


class TestSubunitD1Sweep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.plddt = root / "PLDDTs" / "PARENT"
        self.plddt.mkdir(parents=True)
        write_matrix(self.plddt / "PARENT_WT_plddt_values.csv", 80, 80)
        # MSH2 variants: A10G changes subunit A (the driver, hotspot 10-11), the passengers
        # A20G/A30G nothing.
        write_matrix(self.plddt / "PARENT_MSH2_A10G_plddt_values.csv", 90, 80)
        write_matrix(self.plddt / "PARENT_MSH2_A20G_plddt_values.csv", 80, 80)
        write_matrix(self.plddt / "PARENT_MSH2_A30G_plddt_values.csv", 80, 80)
        # The other subunit's file in the same folder, and a change only in subunit B's rows:
        # neither may reach MSH2's sweep.
        write_matrix(self.plddt / "PARENT_MSH6_A11G_plddt_values.csv", 80, 10)
        write_matrix(self.plddt / "PARENT_MSH2_A11G_plddt_values.csv", 80, 10)
        self.cfg = {
            "output_prefix": "MSH2", "sweep_dir": root / "MSH2" / "ensemble_sweeps",
            "plddt_dir": self.plddt, "plddt_row_range": [0, 3], "cohort_subunit_token": "MSH2",
            "ensemble_sizes": [1, 2], "ensemble_repeats_by_size": {1: 3, 2: 3},
            "label_mode": "hotspot_range", "hotspot_range": [10, 11],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_d1_sweep_reads_only_the_subunit_rows_and_files(self):
        raw = eap.run_d1_ensemble_sweep(self.cfg, force=True)["raw"]
        self.assertEqual(sorted(raw.artifact.unique()),
                         ["PARENT_MSH2_A10G", "PARENT_MSH2_A11G", "PARENT_MSH2_A20G", "PARENT_MSH2_A30G"])
        self.assertTrue((raw.n_residues_used == 3).all())
        d1 = raw.groupby("artifact").d1_mean.mean()
        self.assertAlmostEqual(d1["PARENT_MSH2_A10G"], 90 / 80 - 1)
        self.assertAlmostEqual(d1["PARENT_MSH2_A20G"], 0.0)
        # A11G only changed subunit B's rows, which are out of MSH2's scope
        self.assertAlmostEqual(d1["PARENT_MSH2_A11G"], 0.0)

    def test_whole_complex_without_scope_sees_everything(self):
        """The same call without the subunit keys reads both chains and every file (the old behaviour)."""
        cfg = {k: v for k, v in self.cfg.items() if k not in ("plddt_row_range", "cohort_subunit_token")}
        raw = eap.run_d1_ensemble_sweep(cfg, force=True)["raw"]
        self.assertIn("PARENT_MSH6_A11G", set(raw.artifact))
        self.assertTrue((raw.n_residues_used == 6).all())

    def test_ks_sweep_honours_the_scope(self):
        raw = eap.run_plddt_ks_ensemble_sweep(self.cfg, force=True)["raw"]
        self.assertNotIn("PARENT_MSH6_A11G", set(raw.artifact))
        self.assertTrue((raw.n_residues_used == 3).all())
        ks = raw.groupby("artifact").plddt_ks_mean.mean()
        self.assertAlmostEqual(ks["PARENT_MSH2_A10G"], 1.0)
        self.assertAlmostEqual(ks["PARENT_MSH2_A11G"], 0.0)

    def test_several_wt_csvs_need_wt_artifact(self):
        write_matrix(self.plddt / "PARENT_again_WT_plddt_values.csv", 70, 70)
        with self.assertRaises(ValueError):
            eap.find_plddt_csvs(self.plddt)
        wt, muts = eap.find_plddt_csvs(self.plddt, wt_artifact="PARENT_again_WT_data")
        self.assertEqual(wt.name, "PARENT_again_WT_plddt_values.csv")
        self.assertNotIn(self.plddt / "PARENT_WT_plddt_values.csv", muts)


class TestCoordChunkFreshness(unittest.TestCase):
    def test_chunks_from_another_cache_are_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir, new_dir, chunks = root / "cache", root / "cache_corefit", root / "coord_chunks"
            for d in (old_dir, new_dir, chunks):
                d.mkdir()
            frame = pd.DataFrame({"artifact": ["X_WT"], "structure_index": [1], "chain": ["A"], "resid": [1],
                                  "icode": [""], "x_aligned": [0.0], "y_aligned": [0.0], "z_aligned": [0.0]})
            frame.to_parquet(old_dir / "X_rmsf_aligned_ca_coordinate_cache.parquet")
            frame.assign(x_aligned=1.0).to_parquet(new_dir / "X_rmsf_aligned_ca_coordinate_cache.parquet")
            cfg = {"prefix": "X", "cache_dir": old_dir}
            self.assertIsNotNone(engine.stale_coord_chunks_reason(cfg, chunks))   # no stamp yet
            (chunks / "_source.json").write_text(engine.json.dumps(engine._chunk_source_stamp(cfg)))
            self.assertIsNone(engine.stale_coord_chunks_reason(cfg, chunks))
            cfg["coord_cache_dir"] = new_dir   # repointed at the core-fit cache
            self.assertIn("built from", engine.stale_coord_chunks_reason(cfg, chunks))
            self.assertFalse(engine.is_core_fit_cache(cfg))
            (new_dir / "X_core_fit_report.json").write_text("{}")
            self.assertTrue(engine.is_core_fit_cache(cfg))


if __name__ == "__main__":
    unittest.main()
