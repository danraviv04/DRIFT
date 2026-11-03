"""Pinned WT references must give D1 and ensemble descriptors the same N draws."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from drift import structure_draws as runtime
# Production imports core first; it reloads the runtime before ES binds aliases.
from drift import core, ensemble_shift as es
from drift.protein_config import load_protein_configs


class TestPinnedWT(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name = "POLE_268-471_WT_plddt_values.csv"
        self.cfg = dict(mut="POLE", read_root=self.root, chain_ids=["A"],
                        wt_csv_glob=self.name, wt_csv_expected_sources=1,
                        plddt_expected_residues=4)
        rng = np.random.default_rng(4)
        self.wt = pd.DataFrame(rng.uniform(70, 95, (4, 125)),
                               columns=[f"seed-{i // 5 + 1}_sample-{i % 5}" for i in range(125)])
        self.wt.to_csv(self.root / self.name)
        (self.wt - 20).to_csv(self.root / "POLE_WT_plddt_values.csv")
        (self.wt - 30).to_csv(self.root / "POLE_again_WT_plddt_values.csv")
        (self.wt - 1).to_csv(self.root / "POLE_A1V_plddt_values.csv")

    def reference(self, n, seed=42):
        return es._wt_reference(self.root, self.name, "POLE", n, seed,
                                expected_sources=1, expected_residues=4)

    def test_production_config_pins_construct(self):
        cfg = load_protein_configs()["POLE"]["d1"]
        self.assertEqual(cfg["wt_csv_glob"], self.name)
        self.assertEqual(cfg["wt_csv_expected_sources"], 1)
        self.assertEqual(cfg["plddt_expected_residues"], 204)

    def test_equal_n_nested_draws_and_d1_reference_agree(self):
        previous = []
        for n in (1, 2, 5, 10, 20, 40, 60, 90, 125):
            with self.subTest(n=n):
                matrix = self.reference(n)
                self.assertEqual(matrix.shape, (4, n))
                self.assertEqual(matrix.columns[:len(previous)].tolist(), previous)
                previous = matrix.columns.tolist()
                ref, count = runtime.compute_wt_ref_from_csv_sources(
                    [str(self.root / self.name)], n, seed=42, key_prefix="wt::POLE",
                    expected_residues=4)
                self.assertEqual(count, n)
                np.testing.assert_allclose(ref.loc[matrix.index], matrix.mean(axis=1), atol=1e-5)
                d1, audit = runtime.build_d1_table(
                    {"POLE": self.cfg}, max_structures_per_variant=n, random_seed=42)
                self.assertEqual(audit.wt_sources.item(), 1)
                self.assertEqual(audit.wt_structures.item(), n)
                self.assertEqual(d1.Num_Structures.item(), n)
        self.assertFalse(self.reference(90).equals(self.reference(90, seed=43)))

    def test_ambiguous_or_missing_reference_fails_before_n1_sampling(self):
        with self.assertRaisesRegex(ValueError, "expected 1 CSV"):
            es._wt_reference(self.root, "*WT*csv", "POLE", 1, 42, expected_sources=1)
        (self.root / self.name).unlink()
        for build in (lambda: self.reference(1), lambda: runtime.build_d1_table(
                {"POLE": self.cfg}, max_structures_per_variant=1, random_seed=42)):
            with self.assertRaisesRegex(ValueError, "found 0"):
                build()

    def test_bad_mutant_rows_fail_in_both_builders(self):
        for bad in (self.wt.iloc[:3], self.wt.set_axis([0, 1, 2, 4])):
            bad.to_csv(self.root / "POLE_A1V_plddt_values.csv")
            for build in (lambda: es.build_ensemble_shift_features(
                    {"POLE": self.cfg}, max_structures_per_variant=5, seed=42),
                    lambda: runtime.build_d1_table(
                        {"POLE": self.cfg}, max_structures_per_variant=5, random_seed=42)):
                with self.assertRaisesRegex(ValueError, "expected residue rows"):
                    build()

    def test_undersized_pool_fails_in_both_readers(self):
        self.wt.iloc[:, :3].to_csv(self.root / self.name)
        for build in (lambda: self.reference(5), lambda: runtime.build_d1_table(
                {"POLE": self.cfg}, max_structures_per_variant=5, random_seed=42)):
            with self.assertRaisesRegex(ValueError, "only 3 available"):
                build()

    def test_missing_pinned_root_cannot_silently_drop_gene(self):
        cfg = dict(self.cfg, read_root=self.root / "missing")
        for build in (lambda: es.build_ensemble_shift_features(
                {"POLE": cfg}, max_structures_per_variant=2, seed=42),
                lambda: runtime.build_d1_table(
                    {"POLE": cfg}, max_structures_per_variant=2, random_seed=42)):
            with self.assertRaisesRegex(ValueError, "pinned WT root does not exist"):
                build()

    def test_pin_cannot_be_bypassed_by_cif_or_tar_fallback(self):
        stray = self.root / "POLE_other_WT" / "model.cif"
        stray.parent.mkdir()
        stray.write_text("invalid CIF: the CSV pin must prevent this being read")
        for dirs_tars in (False, True):
            cfg = dict(self.cfg, dirs_tars=dirs_tars)
            _, audit = runtime.build_d1_table(
                {"POLE": cfg}, max_structures_per_variant=2, random_seed=42)
            self.assertEqual(audit.wt_structures.item(), 2)
        (self.root / self.name).unlink()
        with self.assertRaisesRegex(ValueError, "found 0"):
            runtime.build_d1_table(
                {"POLE": self.cfg}, max_structures_per_variant=2, random_seed=42)

    def test_wt_subset_is_shared_across_variants(self):
        (self.wt + 1).to_csv(self.root / "POLE_A2V_plddt_values.csv")
        with patch.object(es, "_residue_ks_summaries", wraps=es._residue_ks_summaries) as score:
            es.build_ensemble_shift_features(
                {"POLE": self.cfg}, max_structures_per_variant=5, seed=42)
        self.assertEqual(score.call_count, 2)
        pd.testing.assert_frame_equal(score.call_args_list[0].args[0],
                                      score.call_args_list[1].args[0])

    def test_duplicate_mutant_ensemble_features_still_average(self):
        duplicate = "POLE_268-471_A1V_plddt_values.csv"
        (self.wt - 3).to_csv(self.root / duplicate)
        parts = []
        for name in ("POLE_A1V_plddt_values.csv", duplicate):
            cfg = dict(self.cfg, mut_csv_glob=name)
            parts.append(es.build_ensemble_shift_features(
                {"POLE": cfg}, max_structures_per_variant=5, seed=42)[es.FEATURE_COLUMNS])
        combined = es.build_ensemble_shift_features(
            {"POLE": self.cfg}, max_structures_per_variant=5, seed=42)
        self.assertEqual(len(combined), 1)
        np.testing.assert_array_equal(combined[es.FEATURE_COLUMNS],
                                       (parts[0].to_numpy() + parts[1].to_numpy()) / 2)

class TestPublicationReferences(unittest.TestCase):
    def test_correlation_reference_tracks_scientific_config(self):
        from figures.si.lib import ks_vs_am as module
        rows = pd.DataFrame([dict(gene=g, subset="all", rho=r)
                             for g, r in module.V6_REFERENCE.items()])
        with tempfile.TemporaryDirectory() as directory, patch.object(
                module, "ORIGINAL", Path(directory)):
            with patch.object(module, "protein_config_fingerprint", return_value="d61d29328362"):
                module.check_against_original(rows, {})
            with patch.object(module, "protein_config_fingerprint", return_value="ebbbda31758d"):
                with self.assertRaisesRegex(AssertionError, "config reference"):
                    module.check_against_original(rows, {})
                rows.loc[rows.gene == "POLE", "rho"] = 0.6313611126606762
                module.check_against_original(rows, {})
            with patch.object(module, "protein_config_fingerprint", return_value="unverified"):
                with self.assertRaisesRegex(ValueError, "No verified correlation reference"):
                    module.check_against_original(rows, {})

