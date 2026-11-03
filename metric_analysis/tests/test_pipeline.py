#!/usr/bin/env python3
"""Fast tests for the metric_analysis pipeline. No external drive, no multi-GB caches.

    python3 -m unittest discover -s metric_analysis/tests -v
    python3 metric_analysis/tests/test_pipeline.py            # same thing

Every test in TestClinVarParsing, TestDegenerateFeatureGuard, TestLdaScoreExport and
TestChunkCountGuard is a regression test for a bug that actually shipped in this
repository. Each one silently produced plausible-looking wrong output rather than
failing, which is why they survived so long -- so each gets a test that would have
caught it.

Uses stdlib unittest deliberately: pytest is not installed in the thesis_core env and
these need to run without adding a dependency.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Put the repo root on the path so `metric_analysis.pipeline...` imports resolve even
# without the editable install (`pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metric_analysis.pipeline import labels  # noqa: E402
from metric_analysis.pipeline.library import structure_metrics as sm  # noqa: E402


def _write_clinvar(path, rows):
    """rows: list of (protein_change_cell, germline_classification)."""
    pd.DataFrame(rows, columns=["Protein change", "Germline classification"]).to_csv(path, index=False)


class TestClinVarParsing(unittest.TestCase):
    """Regression: 'Protein change' holds COMMA-SEPARATED alleles.

    Grouping on the raw cell meant 111 of MSH2's 120 rows could never match an allele.
    MSH2 produced ZERO drivers for 109 structural variants and MutS_alpha labelled
    15 of 151 -- while every cross-family agreement check passed, because all families
    agreed on the same wrong answer.
    """

    def test_comma_separated_cell_yields_every_allele(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "cv.csv"
            _write_clinvar(csv, [("M67L, M1L", "Pathogenic"), ("R106K, R40K", "Benign")])
            drv, psg = labels.load_clinvar_driver_passenger(csv)
        self.assertEqual(drv, {"M67L", "M1L"}, "both alleles in a comma cell must be extracted")
        self.assertEqual(psg, {"R106K", "R40K"})

    def test_single_allele_cell_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "cv.csv"
            _write_clinvar(csv, [("G12D", "Pathogenic"), ("V9N", "Benign")])
            drv, psg = labels.load_clinvar_driver_passenger(csv)
        self.assertEqual(drv, {"G12D"})
        self.assertEqual(psg, {"V9N"})

    def test_nonsense_and_frameshift_forms_are_skipped(self):
        """'G71*' cannot match an all-missense structural set; it must not become an allele."""
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "cv.csv"
            _write_clinvar(csv, [("G5*, D71Y, G71*", "Pathogenic")])
            drv, _ = labels.load_clinvar_driver_passenger(csv)
        self.assertEqual(drv, {"D71Y"}, "only simple missense codes are usable")

    def test_consensus_pathogenic_beats_benign_and_likely_counts(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "cv.csv"
            _write_clinvar(csv, [
                ("A1B", "Likely pathogenic"),          # 'Likely' must count
                ("C2D", "Benign/Likely benign"),
                ("E3F", "Uncertain significance"),     # neither -> unlabelled
                ("G4H", "Pathogenic"), ("G4H", "Benign"),  # two submissions disagree
                # ClinVar's OWN aggregate verdict, which is a different thing
                # from two disagreeing submissions. It contains "pathogenic"
                # inside "pathogenicity" and no "benign", so a bare substring
                # test reads it as a driver. It is uncertain.
                ("I5J", "Conflicting classifications of pathogenicity"),
                # ...and it must not silence a real submission on the same
                # allele: the conflicting record is dropped, not a veto.
                ("K6L", "Conflicting classifications of pathogenicity"),
                ("K6L", "Benign"),
            ])
            drv, psg = labels.load_clinvar_driver_passenger(csv)
        self.assertIn("A1B", drv)
        self.assertIn("C2D", psg)
        self.assertNotIn("E3F", drv | psg, "VUS must not be labelled either way")
        self.assertIn("G4H", drv, "any pathogenic submission outweighs benign ones")
        self.assertNotIn(
            "I5J", drv | psg,
            "ClinVar 'Conflicting classifications of pathogenicity' is uncertain, "
            "not pathogenic, and must be excluded like any other VUS")
        self.assertIn("K6L", psg,
                      "a conflicting record is dropped, so other submissions still decide")


class TestLabelModes(unittest.TestCase):
    """Each protein uses a different ground truth; the dispatcher must honour all five."""

    def test_hotspot_range(self):
        cfg = {"label_mode": "hotspot_range", "hotspot_range": (168, 269)}
        self.assertEqual(labels.classify_allele_or_artifact("FOXA1_R200H", cfg), "Driver")
        self.assertEqual(labels.classify_allele_or_artifact("FOXA1_R100H", cfg), "Passenger")
        self.assertEqual(labels.classify_allele_or_artifact("FOXA1_R168H", cfg), "Driver", "boundary is inclusive")
        self.assertEqual(labels.classify_allele_or_artifact("FOXA1_R269H", cfg), "Driver", "boundary is inclusive")

    def test_clinvar_by_gene_needs_the_full_name_not_a_bare_code(self):
        """Regression: MSH2/MSH6 number residues independently, so the gene must be
        resolved from the artifact name. Passing a bare code labelled everything UNK,
        which is how MutS_alpha ended up with 0 of 151 labelled."""
        with tempfile.TemporaryDirectory() as d:
            msh2, msh6 = Path(d) / "msh2.csv", Path(d) / "msh6.csv"
            _write_clinvar(msh2, [("A45V", "Benign")])
            _write_clinvar(msh6, [("A45V", "Pathogenic")])   # same code, opposite class
            cfg = {"label_mode": "clinvar_by_gene", "clinvar_csv_by_gene": {"MSH2": msh2, "MSH6": msh6}}
            self.assertEqual(labels.classify_allele_or_artifact("MutS_alpha_MSH2_A45V", cfg), "Passenger")
            self.assertEqual(labels.classify_allele_or_artifact("MutS_alpha_MSH6_A45V", cfg), "Driver")
            self.assertEqual(labels.classify_allele_or_artifact("A45V", cfg), "UNK",
                             "a bare code carries no gene, so it must not be guessed")

    def test_csv_lookup(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "labels.csv"
            pd.DataFrame({"Variant": ["A273E", "P286R"], "Label": [1, 0]}).to_csv(csv, index=False)
            cfg = {"label_mode": "csv_lookup", "labels_csv": csv}
            self.assertEqual(labels.classify_allele_or_artifact("POLE_A273E", cfg), "Driver")
            self.assertEqual(labels.classify_allele_or_artifact("POLE_P286R", cfg), "Passenger")
            self.assertEqual(labels.classify_allele_or_artifact("POLE_Q999Z", cfg), "UNK")

    def test_continuous_assay_never_assigns_a_class(self):
        """TP53/KRAS assays are continuous targets only (2026-10-13): no cutoff."""
        cfg = {"label_mode": "continuous_assay"}
        for name in ("KRAS_G12D", "P53_R175H", "KRAS_V9N"):
            self.assertEqual(labels.classify_allele_or_artifact(name, cfg), "UNK")
        self.assertEqual(labels.classify_allele_or_artifact("KRAS_WT", cfg), "WT")
        for retired in ("phenotypic_percentile", "dms_lfc"):
            with self.assertRaisesRegex(ValueError, "unknown label_mode"):
                labels.classify_allele_or_artifact("KRAS_G12D", {"label_mode": retired})

    def test_wt_is_always_wt_and_unknown_mode_raises(self):
        cfg = {"label_mode": "hotspot_range", "hotspot_range": (1, 10)}
        for name in ["POLE_WT", "MutS_alpha_MSH2_WT_data", "wt", "x/WT/y"]:
            self.assertEqual(labels.classify_allele_or_artifact(name, cfg), "WT", name)
        with self.assertRaises(ValueError):
            labels.classify_allele_or_artifact("A1B", {"label_mode": "nonsense_mode"})


class TestDegenerateFeatureGuard(unittest.TestCase):
    """Regression: Cohen's d divides by pooled SD, so a near-constant feature gets a huge
    |d| with no discriminative power. POLE's pct_very_low (3 distinct values across 1075
    mutations) ranked #1 at |d|=2.74 with ROC AUC 0.606, taking a slot in the 10-feature
    LDA and displacing a real feature."""

    def _frame(self, n=200):
        rng = np.random.default_rng(0)
        label = np.array([1] * (n // 4) + [0] * (n - n // 4))
        good = np.where(label == 1, rng.normal(6, 1, n), rng.normal(3, 1, n))

        # Reproduce pct_very_low faithfully: a handful of DISTINCT VALUES, almost all
        # identical. Adding smooth jitter here would defeat the test -- it gives the
        # column hundreds of distinct values and a healthy relative_std, at which point
        # the guard rightly declines to flag it. The real column had 3 distinct values
        # across 1075 rows (1066 of them identical), which is what makes |d| explode
        # while ROC AUC stays near chance.
        near_constant = np.where(label == 1, 0.19, 0.198413)
        near_constant[2] = 0.190001          # a couple of extra distinct values, so
        near_constant[-2] = 0.198414         # within-group SD is tiny but not exactly 0
        return pd.DataFrame({"label": label, "good_feature": good,
                             "near_constant": near_constant,
                             "truly_constant": np.ones(n)})

    def test_the_fixture_really_is_near_constant(self):
        """Guards the test itself: an earlier version added jitter and silently stopped
        exercising the condition it claimed to."""
        f = self._frame()
        self.assertLessEqual(f["near_constant"].nunique(), 4,
                             "fixture must have very few distinct values to be degenerate")

    def test_near_constant_is_flagged_and_demoted(self):
        disc = sm.run_feature_discrimination_analysis(self._frame())
        self.assertIn("degenerate", disc.columns)
        row = disc.set_index("feature").loc["near_constant"]
        self.assertTrue(bool(row["degenerate"]), "3-distinct-value feature must be flagged")
        self.assertGreater(abs(row["cohens_d"]), 1.0,
                           "its |d| is genuinely inflated -- that is the whole trap")
        self.assertEqual(disc.iloc[0]["feature"], "good_feature",
                         "the genuinely separating feature must rank first")
        self.assertFalse(bool(disc.iloc[0]["degenerate"]))

    def test_degenerate_rows_are_still_reported_not_dropped(self):
        disc = sm.run_feature_discrimination_analysis(self._frame())
        self.assertIn("near_constant", set(disc["feature"]),
                      "flagged features must stay visible for diagnosis")

    def test_degenerate_features_sort_last(self):
        disc = sm.run_feature_discrimination_analysis(self._frame())
        flags = disc["degenerate"].tolist()
        self.assertEqual(flags, sorted(flags), "all non-degenerate rows must precede degenerate ones")


class TestLdaScoreExport(unittest.TestCase):
    """Regression: stage_discrimination guarded on `"lda_scores" in lda_result`, but
    run_lda_analysis_cv returns the scores under "df_scores". The guard was always False,
    so *_mutations_with_scores.csv shipped with no scores and was byte-identical to
    *_mutation_features.csv for all 5 proteins."""

    def _result(self):
        rng = np.random.default_rng(1)
        n = 120
        label = np.array([1] * 30 + [0] * (n - 30))
        df = pd.DataFrame({
            "label": label,
            "f1": np.where(label == 1, rng.normal(5, 1, n), rng.normal(2, 1, n)),
            "f2": rng.normal(0, 1, n),
        })
        disc = sm.run_feature_discrimination_analysis(df)
        return df, sm.run_lda_analysis_cv(df, discrimination_df=disc, top_n=2,
                                          n_splits=3, n_repeats=1, random_state=0)

    def test_scores_live_under_df_scores_not_lda_scores(self):
        _, res = self._result()
        self.assertNotIn("lda_scores", res,
                         "if this key ever appears, revisit run_structure_metrics' guard")
        self.assertIn("df_scores", res)
        self.assertIn("lda_score", res["df_scores"].columns)

    def test_df_scores_index_aligns_so_a_join_is_safe(self):
        df, res = self._result()
        self.assertTrue(res["df_scores"].index.equals(df.index),
                        "the export joins on index, not position")

    def test_lda_score_separates_the_classes(self):
        df, res = self._result()
        s = res["df_scores"]["lda_score"]
        self.assertGreater(s[df.label == 1].mean(), s[df.label == 0].mean(),
                           "scores must be oriented so drivers score higher")


class TestChunkCountGuard(unittest.TestCase):
    """Regression: total_chunks was hardcoded to 1 against an on-disk 4-way split, so
    prepare() saw chunk_0 present, skipped, and the sweep covered a quarter of the
    artifacts while reporting success."""

    def test_mismatched_chunk_count_raises_rather_than_under_sweeping(self):
        from metric_analysis.pipeline.library import ensemble_aware_pipeline as eap
        with tempfile.TemporaryDirectory() as d:
            sweep = Path(d) / "ensemble_sweeps"
            (sweep / "coord_chunks").mkdir(parents=True)
            for i in range(4):
                (sweep / "coord_chunks" / f"TEST_coord_chunk_{i}.parquet").touch()
            cfg = {"output_prefix": "TEST", "sweep_dir": sweep, "cache_dir": Path(d) / "cache",
                   "protein_chains": ["A"], "ensemble_sizes": [1], "ensemble_repeats_by_size": {1: 1},
                   "label_mode": "hotspot_range", "hotspot_range": (1, 2)}
            # Chunks that record a different source cache are discarded and rebuilt (2026-10-15),
            # so stamp these as copies of the current cache: the count guard is about fresh chunks.
            from metric_analysis.pipeline.orchestration import run_ensemble_sweep_parallel as engine
            (Path(d) / "cache").mkdir()
            (Path(d) / "cache" / "TEST_rmsf_aligned_ca_coordinate_cache.parquet").touch()
            (sweep / "coord_chunks" / "_source.json").write_text(
                engine.json.dumps(engine._chunk_source_stamp(cfg)))
            orig = eap.get_ensemble_aware_config
            eap.get_ensemble_aware_config = lambda p: cfg
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    eap.run_rmsf_rmsd_ensemble_sweep("TEST", force=True, total_chunks=2)
                self.assertIn("coord chunks", str(ctx.exception))
            finally:
                eap.get_ensemble_aware_config = orig


class TestInterfacePartnerChains(unittest.TestCase):
    """Interface RMSF needs a second chain group; it used to be hardwired to DNA.

    `partner_chains` was added for VHL, whose AlphaFold3 CRL2 complex has PROTEIN
    partners (Elongin B/C, Cullin-2, RBX1, HIF-1a) rather than DNA. Reusing
    `dna_chains` for those would also switch on the protein-DNA distance stage and
    write `*_protein_dna_distance_*` files describing protein-protein contacts. These
    tests pin the resolution order so the four DNA-binding proteins keep working
    untouched and VHL does not silently lose its interface analysis.
    """

    def setUp(self):
        from metric_analysis.pipeline.config.protein_config import get_protein_config, supported_proteins
        self.get = get_protein_config
        self.proteins = supported_proteins(["output_prefix", "protein_chains"])

    @staticmethod
    def _resolve(cfg):
        """Mirror of stage_rmsf's selection: partner_chains wins, dna_chains is fallback."""
        return cfg.get("partner_chains") or cfg.get("dna_chains")

    def test_dna_binding_proteins_still_resolve_to_their_dna_chains(self):
        for p in ["FOXA1", "MutS_alpha", "POLE", "TP53"]:
            if p not in self.proteins:
                continue
            cfg = self.get(p, ["protein_chains"])
            self.assertIsNone(cfg.get("partner_chains"), f"{p} should not need partner_chains")
            self.assertEqual(self._resolve(cfg), cfg["dna_chains"], f"{p} lost its DNA partner group")

    def test_vhl_uses_protein_partners_and_no_dna_chains(self):
        if "VHL" not in self.proteins:
            self.skipTest("VHL not configured")
        cfg = self.get("VHL", ["protein_chains"])
        self.assertEqual(cfg["protein_chains"], ["A"], "VHL's mutated chain is A only")
        self.assertTrue(cfg.get("partner_chains"), "VHL needs partner_chains for interface RMSF")
        self.assertIsNone(cfg.get("dna_chains"),
                          "VHL has no DNA; setting dna_chains would wrongly enable the DNA stage")
        self.assertNotIn("A", cfg["partner_chains"], "the mutated chain must not be its own partner")

    def test_partner_group_never_overlaps_the_mutated_chains(self):
        for p in self.proteins:
            cfg = self.get(p, ["protein_chains"])
            partners = self._resolve(cfg) or []
            self.assertFalse(set(partners) & set(cfg["protein_chains"]),
                             f"{p}: partner group overlaps protein_chains")

    def test_proteins_without_a_partner_group_skip_interface_cleanly(self):
        for p in ["KRAS", "PTEN"]:
            if p not in self.proteins:
                continue
            self.assertIsNone(self._resolve(self.get(p, ["protein_chains"])),
                              f"{p} unexpectedly has a partner group")


class TestMutationCodeTokenBoundary(unittest.TestCase):
    def test_gene_name_containing_the_pattern_is_not_the_mutation(self):
        from metric_analysis.pipeline.labels import extract_mutation_code
        from metric_analysis.pipeline.library.structure_metrics import extract_mutation_code as sm_extract
        for fn in (extract_mutation_code, sm_extract):
            self.assertEqual(fn("PIK3CA_H1047R_plddt_values.csv"), "H1047R")
            self.assertEqual(fn("P53_C182S_with_ZN_and_DNA"), "C182S")
            self.assertEqual(fn("MSH2_R524P"), "R524P")


class TestClinVarSchemaTolerance(unittest.TestCase):
    """ClinVar's download UI ships two column layouts and this repo holds both.

    The whole point of the shared reader is that a refreshed export must not change any
    label. These tests pin the equivalence rather than the plumbing.
    """

    OLD = {"Name": ["NM_1(GENE):c.1A>G (p.Ala1Gly)"], "Gene(s)": ["GENE"],
           "Protein change": ["A1G"], "Germline classification": ["Pathogenic"],
           "Germline review status": ["reviewed by expert panel"]}
    NEW = {"Variation": ["NM_1(GENE):c.1A>G (p.Ala1Gly)"], "Genes": ["GENE"],
           "Protein change": ["A1G"], "Classification": ["G: Pathogenic"],
           "Review status": ["reviewed by expert panel"]}

    def _norm(self, d):
        from metric_analysis.pipeline.clinvar import normalize_clinvar_frame
        return normalize_clinvar_frame(pd.DataFrame(d))

    def test_tsv_trailing_empty_field_preserves_transcript_and_labels(self):
        import tempfile
        from metric_analysis.pipeline.clinvar import read_clinvar_csv, clinvar_schema
        from metric_analysis.pipeline.labels import load_clinvar_driver_passenger
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clinvar.tsv"
            path.write_text(
                "Name\tProtein change\tGermline classification\n"
                "NM_006218.4(PIK3CA):c.3140A>G (p.His1047Arg)\tH1047R\tPathogenic\t\n"
                "NM_006218.4(PIK3CA):c.1171A>G (p.Ile391Met)\tI391M\tBenign\t\n"
                "NM_000000.1(OTHER):c.3A>G (p.Ala2Gly)\tA2G\tPathogenic\t\n"
            )
            self.assertTrue(read_clinvar_csv(path).iloc[0]["Name"].startswith("NM_006218"))
            self.assertEqual(clinvar_schema(path), "old")
            self.assertEqual(load_clinvar_driver_passenger(path, target_gene="PIK3CA"),
                             ({"H1047R"}, {"I391M"}))

    def test_both_layouts_normalize_to_the_same_canonical_columns(self):
        old, new = self._norm(self.OLD), self._norm(self.NEW)
        for col in ["Name", "Gene(s)", "Protein change", "Germline classification"]:
            self.assertEqual(old[col].iloc[0], new[col].iloc[0], f"{col} differs")

    def test_germline_prefix_is_stripped(self):
        self.assertEqual(self._norm(self.NEW)["Germline classification"].iloc[0], "Pathogenic")

    def test_non_germline_prefix_is_blanked_not_stripped(self):
        """A somatic/oncogenicity call must NOT become a germline label.

        Downstream matching is substring-based ("pathogenic" in value.lower()), so
        leaving "S: Pathogenic" intact would silently label a somatic classification as
        a germline Driver.
        """
        d = dict(self.NEW, Classification=["S: Pathogenic"])
        self.assertTrue(pd.isna(self._norm(d)["Germline classification"].iloc[0]))

    def test_normalizing_an_old_frame_is_a_no_op_for_labels(self):
        before = pd.DataFrame(self.OLD)
        after = self._norm(self.OLD)
        for col in before.columns:
            self.assertEqual(list(before[col]), list(after[col]), f"{col} mutated")

    def test_stars_are_derived_when_the_export_lacks_them(self):
        self.assertEqual(self._norm(self.NEW)["stars"].iloc[0], 3)  # expert panel

    def test_real_exports_of_both_layouts_load_and_agree(self):
        """End-to-end on the actual files: no code may be classed both ways."""
        from metric_analysis.pipeline.clinvar import clinvar_schema
        from metric_analysis.pipeline.labels import load_clinvar_driver_passenger
        root = Path(__file__).resolve().parents[2] / "Ground_Truths"
        old_p, new_p = root / "clinvar_result_pten_old.csv", root / "ClinVar" / "clinvar_result_pten.csv"
        if not (old_p.exists() and new_p.exists()):
            self.skipTest("PTEN exports not present")
        self.assertEqual((clinvar_schema(old_p), clinvar_schema(new_p)), ("old", "new"))
        od, op = load_clinvar_driver_passenger(old_p)
        nd, np_ = load_clinvar_driver_passenger(new_p)
        self.assertTrue(od and nd, "both exports should yield drivers")
        self.assertFalse((od & np_) | (op & nd), "a code is Driver in one export and Passenger in the other")


class TestConfigIntegrity(unittest.TestCase):
    """The real protein_configs.json must stay internally consistent."""

    def setUp(self):
        from metric_analysis.pipeline.config.protein_config import get_protein_config, supported_proteins
        self.get = get_protein_config
        self.proteins = supported_proteins(["output_prefix", "label_mode"])

    CORE_PROTEINS = {"MutS_alpha", "POLE", "TP53", "KRAS"}

    def test_core_proteins_are_never_lost(self):
        """New proteins may be added; the original five must not silently disappear.

        Was an exact-equality check, which turned every legitimate addition into a test
        failure. The value here is catching a *removal* or a renamed key, so assert
        containment and let the config grow.
        """
        self.assertLessEqual(self.CORE_PROTEINS, set(self.proteins))

    def test_scaffolded_proteins_stay_out_of_default_all(self):
        """`enabled: false` entries must be addressable by name but excluded from 'all'.

        Guards the split between supported_proteins() (argparse choices, includes
        scaffolds so `--protein PTEN` still works) and enabled_proteins() (what a bare
        `--protein all` fans out to). Getting this backwards would either crash a full
        sweep on a protein with no data, or make a scaffolded protein unrunnable.
        """
        from metric_analysis.pipeline.config.protein_config import enabled_proteins
        keys = ["output_prefix", "label_mode"]
        enabled = set(enabled_proteins(keys))
        supported = set(self.proteins)
        self.assertLessEqual(enabled, supported)
        # TP53/KRAS are deliberately disabled for 'all' since 2026-10-13: no
        # binary label, extraction-only (they must still be addressable by name).
        assay_only = {"TP53", "KRAS"}
        self.assertLessEqual(self.CORE_PROTEINS - assay_only, enabled, "a core protein got disabled")
        self.assertFalse(assay_only & enabled)
        for scaffold in supported - enabled:
            self.assertFalse(
                self.get(scaffold, keys).get("enabled", True),
                f"{scaffold} excluded from 'all' but not marked enabled:false",
            )

    def test_each_protein_has_the_fields_its_label_mode_needs(self):
        needs = {"hotspot_range": ["hotspot_range"], "clinvar": ["clinvar_csv"],
                 "clinvar_by_gene": ["clinvar_csv_by_gene"], "csv_lookup": ["labels_csv"],
                 "continuous_assay": ["structure_metrics_stages"]}
        for p in self.proteins:
            cfg = self.get(p, ["label_mode"])
            for field in needs[cfg["label_mode"]]:
                self.assertIn(field, cfg, f"{p} ({cfg['label_mode']}) needs {field}")

    def test_label_source_files_exist_on_disk(self):
        for p in self.proteins:
            cfg = self.get(p, ["label_mode"])
            for key in ["clinvar_csv", "labels_csv"]:
                if key in cfg:
                    self.assertTrue(Path(cfg[key]).exists(), f"{p}: missing {key} -> {cfg[key]}")
            for path in cfg.get("clinvar_csv_by_gene", {}).values():
                self.assertTrue(Path(path).exists(), f"{p}: missing {path}")

    def test_ensemble_repeats_cover_every_ensemble_size(self):
        for p in self.proteins:
            cfg = self.get(p, ["ensemble_sizes", "ensemble_repeats_by_size"])
            for n in cfg["ensemble_sizes"]:
                self.assertIn(int(n), cfg["ensemble_repeats_by_size"],
                              f"{p}: ensemble size {n} has no repeat count")

    def test_outlier_threshold_is_explicit_and_consistent(self):
        for p in self.proteins:
            cfg = self.get(p, ["plddt_outlier_threshold"])
            self.assertAlmostEqual(
                float(cfg["plddt_outlier_threshold"]), 0.01,
                msg=f"{p}: supervisor-approved outlier threshold changed",
            )

    def test_every_protein_labels_both_classes_from_real_ground_truth(self):
        """Guards the class of bug where a parsing change silently empties a class.

        A subunit dataset (MSH2, MSH6) reads its parent's folder via plddt_dir and
        counts only the files carrying its own token. This used to skipTest inside the
        loop at the first protein without PLDDTs/<output_prefix> -- MSH2 -- so every
        protein after it was never checked. Now only a scaffold (enabled: false, pLDDT
        stage not run yet) may lack its folder; an enabled protein without one fails.
        """
        from pathlib import Path as P
        root = P(__file__).resolve().parents[2]
        for p in self.proteins:
            with self.subTest(protein=p):
                cfg = self.get(p, ["output_prefix", "label_mode"])
                if cfg["label_mode"] == "continuous_assay":
                    # No binary label by design; extraction-only (plddt stage).
                    self.assertEqual(cfg["structure_metrics_stages"], ["plddt"])
                    self.assertFalse(cfg.get("enabled", True))
                    continue
                plddt = P(cfg.get("plddt_dir") or root / "PLDDTs" / cfg["output_prefix"])
                if not plddt.is_dir():
                    self.assertFalse(cfg.get("enabled", True), f"{p}: enabled but no pLDDT dir at {plddt}")
                    continue
                with sm.plddt_scope(include=cfg.get("cohort_subunit_token")):
                    files = sm.list_plddt_csv_files(plddt)
                counts = {"Driver": 0, "Passenger": 0}
                for f in files:
                    cls = labels.classify_allele_or_artifact(f.stem, cfg)
                    if cls in counts:
                        counts[cls] += 1
                self.assertGreater(counts["Driver"], 0, f"{p}: zero Drivers -- suspect a parse failure")
                self.assertGreater(counts["Passenger"], 0, f"{p}: zero Passengers")


class TestClassDominance(unittest.TestCase):
    """Regression: the script refactor silently dropped this analysis.

    Driver-vs-Passenger Mann-Whitney on the RMSF/DNA metrics lived inline in the five
    notebooks. Replacing them with run_structure_metrics.py ported the compute but not
    this, so the pipeline shipped for weeks producing strictly less than the notebooks
    had -- and nothing failed, because a missing analysis leaves no error behind, only
    an absent file. These tests pin the statistics so the port is verifiable, not just
    present.
    """

    def _frame(self, driver_vals, passenger_vals):
        rows = [{"artifact": f"P_D{i}X_model", "status": "ok", "m": v}
                for i, v in enumerate(driver_vals, 1)]
        rows += [{"artifact": f"P_P{i}X_model", "status": "ok", "m": v}
                 for i, v in enumerate(passenger_vals, 1)]
        return pd.DataFrame(rows)

    @staticmethod
    def _classify(name):
        return "Driver" if "_D" in name else ("Passenger" if "_P" in name else "UNK")

    def test_perfect_separation_gives_probability_one(self):
        df = self._frame([10, 11, 12, 13], [1, 2, 3, 4])
        row = sm.class_dominance_stats(
            sm.attach_class_labels(df, ["m"], classify=self._classify), "m", "M")
        self.assertEqual(row["p_driver_gt_passenger"], 1.0)
        self.assertEqual(row["rank_biserial"], 1.0)
        self.assertEqual((row["n_driver"], row["n_passenger"]), (4, 4))

    def test_reversed_separation_gives_probability_zero(self):
        df = self._frame([1, 2, 3, 4], [10, 11, 12, 13])
        row = sm.class_dominance_stats(
            sm.attach_class_labels(df, ["m"], classify=self._classify), "m", "M")
        self.assertEqual(row["p_driver_gt_passenger"], 0.0)
        self.assertEqual(row["rank_biserial"], -1.0)

    def test_probability_equals_the_metrics_roc_auc(self):
        """P(driver > passenger) is the common-language effect size, which IS the AUC of
        the metric used alone. If these ever diverge, one of them is being computed wrong."""
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        df = self._frame(rng.normal(1.0, 1.0, 40), rng.normal(0.0, 1.0, 40))
        lab = sm.attach_class_labels(df, ["m"], classify=self._classify)
        row = sm.class_dominance_stats(lab, "m", "M")
        auc = roc_auc_score((lab["class"] == "Driver").astype(int), lab["m"])
        self.assertAlmostEqual(row["p_driver_gt_passenger"], auc, places=10)

    def test_too_few_members_returns_none_rather_than_a_bogus_test(self):
        df = self._frame([5.0], [1.0])
        self.assertIsNone(sm.class_dominance_stats(
            sm.attach_class_labels(df, ["m"], classify=self._classify), "m", "M"))

    def test_unlabelled_and_failed_artifacts_are_excluded(self):
        df = self._frame([10, 11, 12], [1, 2, 3])
        df.loc[len(df)] = {"artifact": "P_WT_model", "status": "ok", "m": 99.0}
        df.loc[len(df)] = {"artifact": "P_D9X_model", "status": "error", "m": 1e6}
        res = sm.run_class_dominance([(df, "m", "M")], classify=self._classify)
        row = res["dominance_df"].iloc[0]
        self.assertEqual((row["n_driver"], row["n_passenger"]), (3, 3),
                         "WT/UNK or status!=ok rows leaked into the test")

    def test_dominance_runs_over_multiple_source_frames(self):
        """Whole-protein metrics and delta-vs-WT metrics live in different frames but
        belong in one table -- that is how the notebooks presented them."""
        a = self._frame([10, 11, 12], [1, 2, 3]).rename(columns={"m": "rmsf_mean"})
        b = self._frame([1, 2, 3], [10, 11, 12]).rename(columns={"m": "delta_abs_mean"})
        res = sm.run_class_dominance(
            [(a, "rmsf_mean", "Mean RMSF"), (b, "delta_abs_mean", "Mean |delta|")],
            classify=self._classify)
        dom = res["dominance_df"].set_index("metric")
        self.assertEqual(len(dom), 2)
        self.assertEqual(dom.loc["rmsf_mean", "p_driver_gt_passenger"], 1.0)
        self.assertEqual(dom.loc["delta_abs_mean", "p_driver_gt_passenger"], 0.0)

    def test_figures_are_written_and_closed(self):
        """Closing matters: a --protein all run makes dozens of figures, and pyplot keeps
        every unclosed one alive."""
        import matplotlib.pyplot as plt
        df = self._frame([10, 11, 12, 13], [1, 2, 3, 4]).rename(columns={"m": "rmsf_mean"})
        res = sm.run_class_dominance([(df, "rmsf_mean", "Mean RMSF")], classify=self._classify)
        before = len(plt.get_fignums())
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dom.png"
            sm.plot_class_dominance_panels(
                res, violin_specs=[("rmsf_mean", "Mean RMSF", "A")], out_path=out)
            self.assertTrue(out.exists() and out.stat().st_size > 0)
        self.assertEqual(len(plt.get_fignums()), before, "figure left open")


class TestD1DistributionPlot(unittest.TestCase):
    def test_filters_unknown_and_nonfinite_values_and_writes_figure(self):
        import matplotlib.pyplot as plt
        df = pd.DataFrame({
            "mutation_type": ["Driver", "Driver", "Passenger", "Passenger", "UNK", "Driver"],
            "d1_mean": [0.4, 0.5, 0.1, 0.2, 999.0, np.nan],
        })
        before = len(plt.get_fignums())
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "d1.png"
            result = sm.plot_d1_distribution_by_class(df, dataset_name="TEST", out_path=out)
            self.assertTrue(out.exists() and out.stat().st_size > 0)
        self.assertEqual((result["n_driver"], result["n_passenger"]), (2, 2))
        self.assertFalse(result["log_scale"])
        self.assertEqual(len(plt.get_fignums()), before, "figure left open")

    def test_large_positive_dynamic_range_uses_log_axis(self):
        df = pd.DataFrame({
            "mutation_type": ["Driver", "Driver", "Passenger", "Passenger"],
            "d1_mean": [0.001, 1.0, 0.002, 0.5],
        })
        with tempfile.TemporaryDirectory() as d:
            result = sm.plot_d1_distribution_by_class(
                df, dataset_name="TEST", out_path=Path(d) / "d1.png")
        self.assertTrue(result["log_scale"])


class TestRmsfSharding(unittest.TestCase):
    """The rmsf stage was serialised (MutS_alpha: ~5 min x 152 artifacts = ~13 h) and got
    split across processes. Sharding is exactly the kind of change that can lose or double
    work while every output still looks well-formed, so these check the invariants rather
    than the RMSF numbers: shards partition the artifacts, WT appears once, a missing shard
    refuses to merge, and the merged deltas equal the single-process deltas.

    The per-artifact RMSF computation is faked out -- it needs USalign and the external
    drive, and what is under test is the slicing and merging, not the alignment.
    """

    N_ARTIFACTS = 11   # deliberately not a multiple of the shard count

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "rmsf").mkdir(parents=True)
        (root / "rmsf" / "global_wt_reference.pdb").write_text("FAKE REFERENCE\n")

        # WT first so it is picked as the reference; the rest are mutants.
        self.artifacts = [root / "TEST_WT_model.tar.gz"] + [
            root / f"TEST_A{i}G_model.tar.gz" for i in range(1, self.N_ARTIFACTS)
        ]
        for p in self.artifacts:
            p.touch()

        self.config = sm.StructureMetricsConfig(
            mut_name="TEST", read_root_path=str(root), chain_ids=("A",), source_type="tar",
            plddt_dir=root / "plddt", output_dir=root,
        )

        def fake_rmsf(artifact, **kwargs):
            # RMSF varying by artifact so a mix-up between artifacts is detectable.
            seed = abs(hash(Path(artifact).name)) % 997
            df = pd.DataFrame({
                "chain": ["A"] * 5,
                "resid": list(range(1, 6)),
                "resname": ["ALA"] * 5,
                "rmsf": [(seed + r) / 100.0 for r in range(5)],
            })
            return None, df, None

        self._orig_iter = sm.iter_artifacts
        self._orig_map = sm.rmsf_map_ca_from_tar
        sm.iter_artifacts = lambda cfg: list(self.artifacts)
        sm.rmsf_map_ca_from_tar = fake_rmsf

    def tearDown(self):
        sm.iter_artifacts = self._orig_iter
        sm.rmsf_map_ca_from_tar = self._orig_map
        self.tmp.cleanup()

    def _run_shard(self, i, n):
        return sm.run_rmsf_batch_vs_wt(
            self.config, show_progress=False, global_wt_batch_mode=True,
            reuse_existing=False, allow_recompute_if_missing=True,
            shard=(i, n), output_suffix=f"shard{i}of{n}",
            compute_deltas=False, reuse_global_reference=True,
        )

    def test_shards_partition_the_artifacts_with_wt_exactly_once(self):
        n = 4
        seen = []
        for i in range(n):
            seen.append(list(self._run_shard(i, n)["summary_df"]["artifact"]))
        flat = [a for s in seen for a in s]

        self.assertEqual(len(flat), len(set(flat)), "an artifact was processed by two shards")
        self.assertEqual(len(flat), self.N_ARTIFACTS, "shards did not cover every artifact")
        wt = [a for a in flat if "WT" in a.upper()]
        self.assertEqual(len(wt), 1, f"WT must be computed exactly once, got {len(wt)}")
        self.assertTrue(any("WT" in a.upper() for a in seen[0]), "WT belongs to shard 0")
        # Striding, not contiguous blocks: sizes differ by at most one.
        sizes = sorted(len(s) for s in seen)
        self.assertLessEqual(sizes[-1] - sizes[0], 2, f"shard sizes are lopsided: {sizes}")

    def test_more_shards_than_artifacts_still_covers_everything(self):
        n = self.N_ARTIFACTS + 3
        flat = [a for i in range(n) for a in self._run_shard(i, n)["summary_df"]["artifact"]]
        self.assertEqual(sorted(flat), sorted(p.stem.replace(".tar", "") for p in self.artifacts))

    def test_merge_refuses_when_a_shard_is_missing(self):
        n = 4
        for i in range(n - 1):          # deliberately skip the last shard
            self._run_shard(i, n)
        with self.assertRaises(FileNotFoundError) as ctx:
            sm.merge_rmsf_shards(self.config, n_shards=n)
        self.assertIn("missing RMSF shard", str(ctx.exception))

    def test_merge_refuses_when_artifact_count_falls_short(self):
        n = 4
        for i in range(n):
            self._run_shard(i, n)
        with self.assertRaises(ValueError) as ctx:
            sm.merge_rmsf_shards(self.config, n_shards=n, expected_artifacts=self.N_ARTIFACTS + 1)
        self.assertIn("were discovered on disk", str(ctx.exception))

    def test_sharded_then_merged_equals_single_process(self):
        """The point of the whole exercise: parallelising must not change the numbers."""
        single = sm.run_rmsf_batch_vs_wt(
            self.config, show_progress=False, global_wt_batch_mode=True,
            reuse_existing=False, allow_recompute_if_missing=True,
            output_suffix="single", reuse_global_reference=True,
        )
        for i in range(4):
            self._run_shard(i, 4)
        merged = sm.merge_rmsf_shards(self.config, n_shards=4,
                                      expected_artifacts=self.N_ARTIFACTS)

        key = ["artifact", "chain", "resid"]
        a = single["all_rmsf_df"].sort_values(key).reset_index(drop=True)
        b = merged["all_rmsf_df"].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[key + ["rmsf"]], b[key + ["rmsf"]])

        a = single["delta_summary_df"].sort_values("artifact").reset_index(drop=True)
        b = merged["delta_summary_df"].sort_values("artifact").reset_index(drop=True)
        self.assertFalse(a.empty, "the single-process run produced no deltas to compare")
        pd.testing.assert_frame_equal(a, b, check_like=True)

    def test_duplicated_wt_baseline_raises_instead_of_fanning_out(self):
        """If WT ever appeared in two shards, an inner merge would multiply every mutant
        row and delta_mean would quietly average over duplicates."""
        df = pd.DataFrame({
            "artifact": ["TEST_WT_model", "TEST_WT_model", "TEST_A1G_model"],
            "chain": ["A", "A", "A"], "resid": [1, 1, 1],
            "resname": ["ALA"] * 3, "rmsf": [1.0, 1.0, 2.0],
        })
        with self.assertRaises(ValueError) as ctx:
            sm._compute_rmsf_deltas(df, reference_hint="WT")
        self.assertIn("duplicated", str(ctx.exception))


class TestPlddtProfileLoader(unittest.TestCase):
    """load_plddt_profile AND load_plddt_matrix must drop the 0..N-1 residue-index column.

    That column is numeric and not all-NaN, so it used to survive into load_plddt_matrix
    and be averaged in as an extra ensemble member. Because mutant and WT files have
    different sample counts, the contamination was asymmetric and manufactured a spurious
    mut/WT ratio -- inflating/deflating d1_mean (and skewing mean_plddt) by amounts that
    scaled with residue count: negligible for POLE/KRAS (~200 res) but up to ~200 on a
    single MutS_alpha variant (2294 res). These tests pin the fix in both loaders.
    """

    def _write_csv(self, tmp, header0):
        # 3 residues x 2 ensemble members; index column present under `header0`.
        path = Path(tmp) / "PROT_A1G_plddt_values.csv"
        df = pd.DataFrame({header0: [0, 1, 2],
                           "seed-1_sample-0": [90.0, 80.0, 70.0],
                           "seed-1_sample-1": [92.0, 82.0, 72.0]})
        df.to_csv(path, index=False)
        return path

    def test_index_column_is_dropped_from_the_ensemble(self):
        for header0 in ("Unnamed: 0", "", "Residue"):
            with tempfile.TemporaryDirectory() as tmp:
                path = self._write_csv(tmp, header0 or "Unnamed: 0")
                idx, mean, std = sm.load_plddt_profile(path)
                np.testing.assert_array_equal(idx, [0, 1, 2])
                # Mean is over the two sample columns only, NOT the index column.
                np.testing.assert_allclose(mean, [91.0, 81.0, 71.0])
                np.testing.assert_allclose(std, [1.0, 1.0, 1.0])

    def test_index_kept_when_it_is_not_a_plain_range(self):
        """A leading column that is not exactly 0..N-1 is real data and stays."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PROT_A1G_plddt_values.csv"
            pd.DataFrame({"seed-1_sample-0": [90.0, 80.0],
                          "seed-1_sample-1": [92.0, 82.0]}).to_csv(path, index=False)
            _, mean, _ = sm.load_plddt_profile(path)
            np.testing.assert_allclose(mean, [91.0, 81.0])

    def test_load_plddt_matrix_drops_the_index_column(self):
        """load_plddt_matrix must return samples only, so row means are true pLDDT means."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_csv(tmp, "Unnamed: 0")
            matrix = sm.load_plddt_matrix(path)
            self.assertEqual(matrix.shape, (3, 2), "index column should have been dropped")
            np.testing.assert_allclose(matrix.mean(axis=1), [91.0, 81.0, 71.0])

    def test_exemplar_distribution_panels_are_paginated_and_share_raw_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name, offset, n_samples in (
                ("TEST_WT", 0.0, 4),
                ("TEST_A12G", -4.0, 3),
                ("TEST_V14L", -1.0, 2),
            ):
                frame = {"Residue": np.arange(5)}
                for sample in range(n_samples):
                    frame[f"sample-{sample}"] = (
                        np.linspace(80.0, 96.0, 5) + offset + sample * 0.4
                    )
                path = root / f"{name}_plddt_values.csv"
                pd.DataFrame(frame).to_csv(path, index=False)
                paths[name] = path

            result = sm.plot_plddt_exemplar_distributions_by_residue(
                paths["TEST_WT"], paths["TEST_A12G"], paths["TEST_V14L"],
                driver_label="Driver A12G", passenger_label="Passenger V14L",
                residue_start=10, panels_per_page=4, ncols=2,
                kde_bandwidth=0.7,
                out_base=root / "TEST_plddt_exemplar_distributions",
            )

            self.assertEqual(result["n_residues"], 5)
            self.assertEqual(result["n_pages"], 2)
            self.assertEqual(result["kde_bandwidth"], 0.7)
            self.assertEqual(
                result["sample_counts"],
                {"WT": 4, "Driver A12G": 3, "Passenger V14L": 2},
            )
            self.assertEqual(
                [(marker["site"], marker["row_index"], marker["residue_number"])
                 for marker in result["site_markers"]],
                [("D", 2, 12), ("P", 4, 14)],
            )
            self.assertTrue(Path(result["pdf_path"]).is_file())
            self.assertEqual(
                [Path(path).name for path in result["page_paths"]],
                [
                    "TEST_plddt_exemplar_distributions_residues_10-13.png",
                    "TEST_plddt_exemplar_distributions_residues_14-14.png",
                ],
            )
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in result["page_paths"]))

    def test_exemplar_distribution_marks_coincident_sites_on_every_selected_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in ("TEST_WT", "TEST_A2G", "TEST_A2V"):
                path = root / f"{name}_plddt_values.csv"
                pd.DataFrame({
                    "Residue": np.arange(4),
                    "sample-0": [80.0, 82.0, 84.0, 86.0],
                    "sample-1": [81.0, 83.0, 85.0, 87.0],
                }).to_csv(path, index=False)
                paths[name] = path

            result = sm.plot_plddt_exemplar_distributions_by_residue(
                paths["TEST_WT"], paths["TEST_A2G"], paths["TEST_A2V"],
                residue_segments=[
                    {"chain_id": "A", "length": 2},
                    {"chain_id": "B", "length": 2},
                ],
                driver_site_chains=["A", "B"],
                passenger_site_chains=["A", "B"],
                panels_per_page=4, ncols=2,
                out_base=root / "TEST_multichain_distributions",
            )

            self.assertEqual(
                [(marker["site"], marker["chain_id"], marker["row_index"])
                 for marker in result["site_markers"]],
                [("D", "A", 1), ("D", "B", 3),
                 ("P", "A", 1), ("P", "B", 3)],
            )

    def test_exemplar_distribution_rejects_incomplete_segment_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in ("TEST_WT", "TEST_A2G", "TEST_V3L"):
                path = root / f"{name}_plddt_values.csv"
                pd.DataFrame({
                    "Residue": np.arange(4),
                    "sample-0": [80.0, 82.0, 84.0, 86.0],
                }).to_csv(path, index=False)
                paths[name] = path

            with self.assertRaisesRegex(ValueError, "describe 3 residues"):
                sm.plot_plddt_exemplar_distributions_by_residue(
                    paths["TEST_WT"], paths["TEST_A2G"], paths["TEST_V3L"],
                    residue_segments=[{"chain_id": "A", "length": 3}],
                    out_base=root / "TEST_bad_mapping",
                )

    def test_d1_is_zero_when_mutant_equals_wt(self):
        """The bug's signature: identical mutant/WT pLDDT gave nonzero D1 because the
        index column was diluted differently across unequal sample counts. With the fix,
        D1 must be exactly 0 when the (non-index) values match, regardless of column count."""
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp) / "PROT_WT_plddt_values.csv"
            mut = Path(tmp) / "PROT_A1G_plddt_values.csv"
            # Same per-residue pLDDT, but WT has 2 samples and the mutant has 3.
            pd.DataFrame({"Unnamed: 0": [0, 1, 2],
                          "s0": [90.0, 80.0, 70.0], "s1": [90.0, 80.0, 70.0]}).to_csv(wt, index=False)
            pd.DataFrame({"Unnamed: 0": [0, 1, 2],
                          "s0": [90.0, 80.0, 70.0], "s1": [90.0, 80.0, 70.0],
                          "s2": [90.0, 80.0, 70.0]}).to_csv(mut, index=False)
            wt_summary = sm.summarize_wild_type(wt)
            row = sm.compute_feature_row(sm.load_plddt_matrix(mut),
                                         wt_matrix=wt_summary["wt_matrix"])
            self.assertAlmostEqual(row["d1_mean"], 0.0, places=9)


class TestPerResiduePlddtKs(unittest.TestCase):
    def test_detects_distribution_change_hidden_by_equal_means(self):
        # Both ensembles have mean pLDDT 5 at residue 0, so a mean-only metric
        # cannot see that one is bimodal and the other is concentrated at 5.
        wt = np.array([
            [0, 0, 0, 0, 10, 10, 10, 10],
            [80, 81, 82, 83, 84, 85, 86, 87],
        ], dtype=float)
        mutant = np.array([
            [5, 5, 5, 5, 5, 5, 5, 5],
            [80, 81, 82, 83, 84, 85, 86, 87],
        ], dtype=float)

        result = sm.compute_per_residue_plddt_ks(wt, mutant)

        self.assertEqual(list(result["residue_index"]), [0, 1])
        self.assertAlmostEqual(result.loc[0, "ks_statistic"], 0.5)
        self.assertAlmostEqual(result.loc[1, "ks_statistic"], 0.0)
        self.assertEqual(result.loc[1, "ks_pvalue"], 1.0)

    def test_fdr_is_applied_across_residues_and_nans_are_dropped_per_row(self):
        wt = np.array([
            [0.0] * 20,
            [1.0, np.nan] + [np.nan] * 18,
        ])
        mutant = np.array([
            [1.0] * 20,
            [1.0, 1.0] + [np.nan] * 18,
        ])

        result = sm.compute_per_residue_plddt_ks(wt, mutant)

        self.assertTrue(bool(result.loc[0, "ks_significant_fdr"]))
        # One finite WT observation still defines the KS statistic (here 0: 1 vs 1,1),
        # but the test cannot reject from it.
        self.assertEqual(result.loc[1, "n_wt"], 1)
        self.assertEqual(result.loc[1, "ks_statistic"], 0.0)
        self.assertFalse(bool(result.loc[1, "ks_significant_fdr"]))

    def test_residue_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "residue-count mismatch"):
            sm.compute_per_residue_plddt_ks(np.ones((2, 3)), np.ones((3, 3)))

    def test_residue_profile_has_classifier_ready_summaries(self):
        wt = np.array([[0.0] * 20, [5.0] * 20])
        mutant = np.array([[1.0] * 20, [5.0] * 20])

        profile = sm.compute_per_residue_plddt_ks(wt, mutant)
        row = sm.summarize_per_residue_plddt_ks(profile)

        self.assertAlmostEqual(row["plddt_ks_mean"], 0.5)
        self.assertAlmostEqual(row["plddt_ks_max"], 1.0)
        self.assertAlmostEqual(row["plddt_ks_fraction_fdr"], 0.5)
        self.assertEqual(row["plddt_ks_n_residues_tested"], 2)
        self.assertEqual(row["plddt_ks_min_wt_samples"], 20)
        self.assertEqual(row["plddt_ks_min_mutant_samples"], 20)

    def test_existing_feature_table_is_not_silently_changed(self):
        wt = np.array([[80.0, 81.0], [90.0, 91.0]])
        mutant = np.array([[79.0, 80.0], [90.0, 92.0]])

        existing_features = sm.compute_feature_row(mutant, wt_matrix=wt)

        self.assertFalse(
            any(name.startswith("plddt_ks_") for name in existing_features),
            "KS must be evaluated as a separate metric before entering the classifier",
        )

    def test_table_builder_preserves_the_per_residue_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            wt_path = Path(tmp) / "TEST_WT_plddt_values.csv"
            mutant_path = Path(tmp) / "TEST_A1G_plddt_values.csv"
            columns = {f"sample_{i}": [0.0, 5.0] for i in range(20)}
            pd.DataFrame({"Residue": [0, 1], **columns}).to_csv(wt_path, index=False)
            mutant_columns = {f"sample_{i}": [1.0, 5.0] for i in range(20)}
            pd.DataFrame({"Residue": [0, 1], **mutant_columns}).to_csv(
                mutant_path, index=False
            )

            summary, residues = sm.build_plddt_ks_tables([mutant_path, wt_path])

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "mutation_code"], "A1G")
        self.assertEqual(summary.loc[0, "status"], "ok")
        self.assertTrue(np.isfinite(summary.loc[0, "d1_mean"]))
        self.assertEqual(len(residues), 2)
        self.assertEqual(set(residues["mutation_code"]), {"A1G"})
        self.assertEqual(set(residues["residue_index"]), {0, 1})

    def test_table_builder_rejects_an_ambiguous_wt_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name in ("TEST_WT_plddt_values.csv", "TEST_again_WT_plddt_values.csv"):
                path = Path(tmp) / name
                pd.DataFrame({"Residue": [0], "sample_0": [90.0]}).to_csv(
                    path, index=False
                )
                paths.append(path)
            mutant = Path(tmp) / "TEST_A1G_plddt_values.csv"
            pd.DataFrame({"Residue": [0], "sample_0": [80.0]}).to_csv(
                mutant, index=False
            )

            with self.assertRaisesRegex(ValueError, "Multiple WT"):
                sm.build_plddt_ks_tables([*paths, mutant])

    def test_stage_maps_raw_tp53_wt_artifact_to_canonical_plddt_name(self):
        from metric_analysis.pipeline.orchestration import run_structure_metrics as rsm

        with tempfile.TemporaryDirectory() as tmp:
            tp53_wt = Path(tmp) / "P53_WT_plddt_values.csv"
            mutant = Path(tmp) / "P53_R175H_plddt_values.csv"
            tp53_wt.touch()
            mutant.touch()

            resolved = rsm._match_configured_wt_plddt_file(
                [mutant, tp53_wt],
                "P53_WT_with_ZN_and_DNA",
                prefix="TP53",
            )

        self.assertEqual(resolved, tp53_wt)

    def test_stage_keeps_pole_explicit_wt_selection(self):
        from metric_analysis.pipeline.orchestration import run_structure_metrics as rsm

        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / "POLE_268-471_WT_plddt_values.csv"
            alternate = Path(tmp) / "POLE_WT_plddt_values.csv"
            selected.touch()
            alternate.touch()

            resolved = rsm._match_configured_wt_plddt_file(
                [alternate, selected], "POLE_268-471_WT", prefix="POLE"
            )

        self.assertEqual(resolved, selected)

    def test_vectorized_sweep_ks_matches_canonical_residue_tests_with_ties(self):
        from metric_analysis.pipeline.library import ensemble_aware_pipeline as eap

        wt = np.array([
            [0, 0, 1, 2],
            [5, 5, 5, 5],
            [1, 2, 3, 4],
        ], dtype=float)
        mutant = np.array([
            [0, 1, 1, 3],
            [5, 6, 6, 6],
            [4, 3, 2, 1],
        ], dtype=float)
        canonical = sm.summarize_per_residue_plddt_ks(
            sm.compute_per_residue_plddt_ks(wt, mutant)
        )
        swept = eap.compute_plddt_ks_for_selected_columns(
            wt, mutant, np.arange(4), np.arange(4)
        )

        self.assertAlmostEqual(swept["plddt_ks_mean"], canonical["plddt_ks_mean"])
        self.assertAlmostEqual(swept["plddt_ks_median"], canonical["plddt_ks_median"])
        self.assertAlmostEqual(swept["plddt_ks_max"], canonical["plddt_ks_max"])

    def test_ks_sweep_defines_n_one_and_writes_auc_curve(self):
        from metric_analysis.pipeline.library import ensemble_aware_pipeline as eap

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plddt_dir = root / "plddt"
            plddt_dir.mkdir()
            wt_path = plddt_dir / "TEST_WT_plddt_values.csv"

            def write_matrix(path, value):
                frame = pd.DataFrame(
                    {f"sample_{i}": np.full(5, value, dtype=float) for i in range(4)}
                )
                frame.index.name = "Residue"
                frame.to_csv(path)

            write_matrix(wt_path, 80.0)
            mutant_paths = []
            for allele, value in (
                ("A10G", 90.0), ("A11G", 90.0),
                ("A20G", 80.0), ("A21G", 80.0),
            ):
                path = plddt_dir / f"TEST_{allele}_plddt_values.csv"
                write_matrix(path, value)
                mutant_paths.append(path)

            cfg = {
                "output_prefix": "TEST",
                "sweep_dir": root / "ensemble_sweeps",
                "ensemble_sizes": [1, 2],
                "ensemble_repeats_by_size": {1: 2, 2: 2},
                "label_mode": "hotspot_range",
                "hotspot_range": [10, 11],
            }
            result = eap.run_plddt_ks_ensemble_sweep(
                cfg,
                plddt_files=[wt_path, *mutant_paths],
                wt_path=wt_path,
                force=True,
            )

            n1 = result["raw"].loc[result["raw"]["ensemble_size"] == 1]
            n2 = result["summary"].loc[
                result["summary"]["ensemble_size"] == 2, "auc_mean"
            ].iloc[0]
            # N=1 is the discordance rate: 1 where 90 != 80, 0 where 80 == 80.
            by_allele = n1.groupby("Allele")["plddt_ks_mean"].mean()
            self.assertEqual(by_allele.to_dict(),
                             {"A10G": 1.0, "A11G": 1.0, "A20G": 0.0, "A21G": 0.0})
            n1_auc = result["summary"].loc[
                result["summary"]["ensemble_size"] == 1, "auc_mean"
            ].iloc[0]
            self.assertAlmostEqual(n1_auc, 1.0)
            self.assertAlmostEqual(n2, 1.0)
            self.assertTrue(
                (root / "plots" / "TEST_pLDDT_KS_N_sweep_AUC.png").exists()
            )

    def test_profile_and_d1_comparison_figures_are_written(self):
        summary = pd.DataFrame({
            "mutation_code": ["A1G", "A2G", "A3G", "A4G"],
            "mutation_type": ["Driver", "Driver", "Passenger", "Passenger"],
            "plddt_ks_mean": [0.70, 0.55, 0.20, 0.30],
            "d1_mean": [0.08, 0.05, 0.01, 0.02],
        })
        residue_rows = []
        for mutation, base in zip(summary["mutation_code"], [0.7, 0.5, 0.2, 0.3]):
            for residue in range(4):
                residue_rows.append({
                    "mutation_code": mutation,
                    "residue_index": residue,
                    "ks_statistic": base + residue * 0.01,
                    "ks_significant_fdr": base > 0.4,
                })
        residues = pd.DataFrame(residue_rows)

        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.png"
            comparison_path = Path(tmp) / "vs_d1.png"
            profile = sm.plot_plddt_ks_profile_by_class(
                residues, summary, out_path=profile_path,
                bootstrap_resamples=50, bootstrap_seed=7,
            )
            comparison = sm.plot_plddt_ks_vs_d1(
                summary, dataset_name="TEST", out_path=comparison_path
            )
            self.assertTrue(profile_path.exists())
            self.assertTrue(comparison_path.exists())
            self.assertGreater(profile_path.stat().st_size, 0)
            self.assertGreater(comparison_path.stat().st_size, 0)
            legend = [text.get_text() for text in profile["figure"].axes[0].get_legend().texts]
            self.assertTrue(all("bootstrap SD" in text for text in legend))
            self.assertEqual(profile["bootstrap_resamples"], 50)
            self.assertEqual(profile["bootstrap_seed"], 7)
        plt.close(profile["figure"])
        plt.close(comparison["figure"])


class TestPerResiduePlddtDistributionMetrics(unittest.TestCase):
    def test_bootstrap_mean_sd_is_reproducible_and_resamples_mutation_rows(self):
        from metric_analysis.pipeline.library.bootstrap_uncertainty import bootstrap_mean_sd

        values = np.array([
            [1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0],
        ])
        first = bootstrap_mean_sd(
            values, n_resamples=2_000, rng=np.random.default_rng(123),
        )
        second = bootstrap_mean_sd(
            values, n_resamples=2_000, rng=np.random.default_rng(123),
        )

        np.testing.assert_array_equal(first, second)
        self.assertGreater(first[0], 0.0)
        self.assertAlmostEqual(first[1] / first[0], 10.0, places=12)

    def test_equal_means_are_detected_as_shape_change(self):
        wt = np.array([[0, 0, 0, 0, 10, 10, 10, 10]], dtype=float)
        mutant = np.array([[5, 5, 5, 5, 5, 5, 5, 5]], dtype=float)

        row = sm.compute_per_residue_plddt_distribution_metrics(wt, mutant).iloc[0]

        self.assertAlmostEqual(row["wasserstein"], 5.0)
        self.assertAlmostEqual(row["location_shift"], 0.0)
        self.assertAlmostEqual(row["shape_excess"], 5.0)
        self.assertAlmostEqual(row["shape_fraction"], 1.0)
        self.assertGreater(row["energy_distance"], 0.0)
        self.assertGreater(row["cvm_effect"], 0.0)

    def test_pure_translation_has_no_shape_excess(self):
        wt = np.array([[1, 2, 4, 8]], dtype=float)
        mutant = wt + 2.0

        row = sm.compute_per_residue_plddt_distribution_metrics(wt, mutant).iloc[0]

        self.assertAlmostEqual(row["wasserstein"], 2.0)
        self.assertAlmostEqual(row["location_shift"], 2.0)
        self.assertAlmostEqual(row["shape_excess"], 0.0)
        self.assertAlmostEqual(row["shape_fraction"], 0.0)

    def test_wasserstein_and_energy_match_scipy_with_unequal_samples(self):
        wt = np.array([[0.0, 1.0, 1.0, np.nan, 5.0]])
        mutant = np.array([[0.0, 2.0, 3.0, 3.0, 8.0, 9.0]])

        row = sm.compute_per_residue_plddt_distribution_metrics(wt, mutant).iloc[0]
        finite_wt = wt[0, np.isfinite(wt[0])]
        finite_mutant = mutant[0, np.isfinite(mutant[0])]

        self.assertAlmostEqual(
            row["wasserstein"],
            stats.wasserstein_distance(finite_wt, finite_mutant),
        )
        self.assertAlmostEqual(
            row["energy_distance"],
            stats.energy_distance(finite_wt / 100.0, finite_mutant / 100.0),
        )
        self.assertEqual(row["n_wt"], 4)
        self.assertEqual(row["n_mutant"], 6)

    def test_thresholded_outlier_score_matches_supervisor_formula(self):
        wt = np.array([[80.0, 90.0, 100.0]])
        mutant = np.array([[90.0, 90.45, 91.8, 81.0]])

        row = sm.compute_per_residue_plddt_distribution_metrics(
            wt, mutant, outlier_threshold=0.01,
        ).iloc[0]

        # median(WT)=90. Deviations are 0, .005, .02, .10; only .02/.10
        # enter the pool, but all four mutant conformations remain in N.
        self.assertAlmostEqual(row["wt_median"], 90.0)
        self.assertAlmostEqual(row["outlier_score"], (0.02 + 0.10) / 4.0)
        self.assertAlmostEqual(row["outlier_fraction"], 2.0 / 4.0)
        self.assertAlmostEqual(row["outlier_mean_deviation"], 0.06)
        self.assertEqual(row["outlier_count"], 2)
        self.assertAlmostEqual(row["outlier_threshold"], 0.01)

    def test_outlier_score_uses_finite_mutant_count_and_strict_threshold(self):
        wt = np.array([[64.0, 64.0, 64.0]])
        mutant = np.array([[56.0, 72.0, 48.0, 80.0, np.nan]])

        row = sm.compute_per_residue_plddt_distribution_metrics(
            wt, mutant, outlier_threshold=0.125,
        ).iloc[0]

        # Deviations exactly at tau (.125) are excluded; .25 and .25 are
        # selected, and N is the four finite mutant conformations.
        self.assertAlmostEqual(row["outlier_score"], (0.25 + 0.25) / 4.0)
        self.assertAlmostEqual(row["outlier_fraction"], 0.5)
        self.assertEqual(row["n_mutant"], 4)

    def test_zero_wt_median_makes_only_ratio_metrics_undefined(self):
        wt = np.array([[0.0, 0.0, 0.0]])
        mutant = np.array([[1.0, 2.0, 3.0]])

        row = sm.compute_per_residue_plddt_distribution_metrics(wt, mutant).iloc[0]

        self.assertTrue(np.isnan(row["outlier_score"]))
        self.assertTrue(np.isnan(row["outlier_fraction"]))
        self.assertTrue(np.isfinite(row["wasserstein"]))

    def test_summary_reproduces_d1_and_uses_nonmax_reductions(self):
        wt = np.array([[10, 10], [20, 20]], dtype=float)
        mutant = np.array([[12, 12], [18, 18]], dtype=float)
        residue = sm.compute_per_residue_plddt_distribution_metrics(wt, mutant)

        summary = sm.summarize_per_residue_plddt_distribution_metrics(residue)

        self.assertAlmostEqual(summary["d1_from_distribution_mean"], 0.15)
        self.assertAlmostEqual(summary["plddt_w1_mean"], 2.0)
        self.assertAlmostEqual(summary["plddt_w1_shape_fraction"], 0.0)
        self.assertAlmostEqual(summary["plddt_outlier_threshold"], 0.01)
        self.assertGreater(summary["plddt_outlier_score_mean"], 0.0)
        self.assertNotIn("plddt_w1_max", summary)

    def test_distribution_profile_uses_bootstrap_sd_ribbons(self):
        summary = pd.DataFrame({
            "mutation_code": ["A1G", "A2G", "A3G", "A4G"],
            "mutation_type": ["Driver", "Driver", "Passenger", "Passenger"],
        })
        rows = []
        for mutation, offset in zip(summary["mutation_code"], [0.04, 0.03, 0.01, 0.02]):
            for residue in range(3):
                rows.append({
                    "mutation_code": mutation,
                    "residue_index": residue,
                    "wasserstein_unit": offset + residue * 0.001,
                    "outlier_score": offset / 2 + residue * 0.0002,
                    "outlier_threshold": 0.01,
                    "shape_excess_relative": offset / 10 + residue * 0.0001,
                })
        result = sm.plot_plddt_distribution_profile_by_class(
            pd.DataFrame(rows), summary,
            bootstrap_resamples=50, bootstrap_seed=9,
        )
        legend = [text.get_text() for text in result["figure"].axes[0].get_legend().texts]
        self.assertTrue(all("bootstrap SD" in text for text in legend))
        self.assertEqual(result["bootstrap_resamples"], 50)
        self.assertEqual(result["bootstrap_seed"], 9)
        plt.close(result["figure"])

    def test_d1_and_ratio_profiles_use_reproducible_bootstrap_sd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write_profile(name, values):
                path = root / name
                pd.DataFrame({"Residue": [0, 1], "s0": values, "s1": values}).to_csv(
                    path, index=False,
                )
                return path

            wt = write_profile("TEST_WT_plddt_values.csv", [80.0, 80.0])
            drivers = [
                write_profile("TEST_A1G_plddt_values.csv", [70.0, 75.0]),
                write_profile("TEST_A2G_plddt_values.csv", [72.0, 74.0]),
            ]
            passengers = [
                write_profile("TEST_A3G_plddt_values.csv", [79.0, 80.0]),
                write_profile("TEST_A4G_plddt_values.csv", [78.0, 79.0]),
            ]
            d1 = sm.plot_d1_profile_by_class(
                wt, drivers, passengers, bootstrap_resamples=50, bootstrap_seed=3,
            )
            ratio = sm.plot_plddt_ratio_by_class(
                wt, drivers, passengers, bootstrap_resamples=50, bootstrap_seed=3,
            )

        self.assertEqual(d1["bootstrap_resamples"], 50)
        self.assertEqual(ratio["bootstrap_resamples"], 50)
        plt.close("all")

    def test_table_builder_preserves_residue_effects_and_d1_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            wt_path = Path(tmp) / "TEST_WT_plddt_values.csv"
            mutant_path = Path(tmp) / "TEST_A1G_plddt_values.csv"
            pd.DataFrame({
                "Residue": [0, 1],
                "s0": [10.0, 20.0],
                "s1": [10.0, 20.0],
            }).to_csv(wt_path, index=False)
            pd.DataFrame({
                "Residue": [0, 1],
                "s0": [12.0, 18.0],
                "s1": [12.0, 18.0],
            }).to_csv(mutant_path, index=False)

            summary, residues = sm.build_plddt_distribution_tables(
                [mutant_path, wt_path]
            )

        self.assertEqual(len(summary), 1)
        self.assertEqual(len(residues), 2)
        self.assertEqual(summary.loc[0, "status"], "ok")
        self.assertLess(summary.loc[0, "d1_consistency_abs_error"], 1e-12)
        self.assertIn("plddt_outlier_score_mean", summary)
        self.assertIn("outlier_score", residues)
        self.assertTrue((residues["outlier_threshold"] == 0.01).all())
        self.assertEqual(set(residues["mutation_code"]), {"A1G"})

    def test_selected_column_summary_matches_canonical_summary(self):
        from metric_analysis.pipeline.library import ensemble_aware_pipeline as eap

        wt = np.array([[0, 1, 1, 2], [5, 5, 5, 5]], dtype=float)
        mutant = np.array([[0, 2, 3, 3], [5, 6, 6, 6]], dtype=float)
        canonical = sm.summarize_per_residue_plddt_distribution_metrics(
            sm.compute_per_residue_plddt_distribution_metrics(wt, mutant)
        )
        swept = eap.compute_plddt_distribution_for_selected_columns(
            wt, mutant, np.arange(4), np.arange(4)
        )

        for metric in (
            "plddt_w1_unit_mean", "plddt_w1_shape_fraction",
            "plddt_outlier_score_mean", "plddt_energy_mean", "plddt_cvm_mean",
        ):
            self.assertAlmostEqual(swept[metric], canonical[metric])

    def test_distribution_sweep_retains_n_one_and_writes_auc_curve(self):
        from metric_analysis.pipeline.library import ensemble_aware_pipeline as eap

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plddt_dir = root / "plddt"
            plddt_dir.mkdir()
            wt_path = plddt_dir / "TEST_WT_plddt_values.csv"

            def write_matrix(path, value):
                frame = pd.DataFrame(
                    {f"sample_{i}": np.full(5, value, dtype=float) for i in range(3)}
                )
                frame.index.name = "Residue"
                frame.to_csv(path)

            write_matrix(wt_path, 80.0)
            mutant_paths = []
            for allele, value in (
                ("A10G", 90.0), ("A11G", 90.0),
                ("A20G", 80.0), ("A21G", 80.0),
            ):
                path = plddt_dir / f"TEST_{allele}_plddt_values.csv"
                write_matrix(path, value)
                mutant_paths.append(path)
            cfg = {
                "output_prefix": "TEST",
                "sweep_dir": root / "ensemble_sweeps",
                "ensemble_sizes": [1, 2],
                "ensemble_repeats_by_size": {1: 1, 2: 1},
                "label_mode": "hotspot_range",
                "hotspot_range": [10, 11],
            }

            result = eap.run_plddt_distribution_ensemble_sweep(
                cfg,
                plddt_files=[wt_path, *mutant_paths],
                wt_path=wt_path,
                force=True,
            )

            n1 = result["raw"].loc[result["raw"]["ensemble_size"] == 1]
            w1_auc = result["summary"].loc[
                (result["summary"]["ensemble_size"] == 2)
                & (result["summary"]["metric"] == "plddt_w1_unit_mean"),
                "auc_mean",
            ].iloc[0]
            self.assertTrue(n1["plddt_w1_unit_mean"].notna().all())
            self.assertAlmostEqual(w1_auc, 1.0)
            outlier_auc = result["summary"].loc[
                (result["summary"]["ensemble_size"] == 2)
                & (result["summary"]["metric"] == "plddt_outlier_score_mean"),
                "auc_mean",
            ].iloc[0]
            self.assertAlmostEqual(outlier_auc, 1.0)
            self.assertTrue(
                (root / "plots" / "TEST_pLDDT_Distribution_N_sweep_AUC.png").exists()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
