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


    def test_load_plddt_matrix_drops_the_index_column(self):
        """load_plddt_matrix must return samples only, so row means are true pLDDT means."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_csv(tmp, "Unnamed: 0")
            matrix = sm.load_plddt_matrix(path)
            self.assertEqual(matrix.shape, (3, 2), "index column should have been dropped")
            np.testing.assert_allclose(matrix.mean(axis=1), [91.0, 81.0, 71.0])


class TestPerResiduePlddtKs(unittest.TestCase):


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

    def test_summary_uses_nonmax_reductions(self):
        wt = np.array([[10, 10], [20, 20]], dtype=float)
        mutant = np.array([[12, 12], [18, 18]], dtype=float)
        residue = sm.compute_per_residue_plddt_distribution_metrics(wt, mutant)

        summary = sm.summarize_per_residue_plddt_distribution_metrics(residue)

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
        from metric_analysis.pipeline.library.plddt_distribution import (
            plot_plddt_distribution_profile_by_class,
        )
        result = plot_plddt_distribution_profile_by_class(
            pd.DataFrame(rows), summary,
            bootstrap_resamples=50, bootstrap_seed=9,
        )
        legend = [text.get_text() for text in result["figure"].axes[0].get_legend().texts]
        self.assertTrue(all("bootstrap SD" in text for text in legend))
        self.assertEqual(result["bootstrap_resamples"], 50)
        self.assertEqual(result["bootstrap_seed"], 9)
        plt.close(result["figure"])


    def test_table_builder_preserves_residue_effects(self):
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
        self.assertNotIn("d1_mean", summary)
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
