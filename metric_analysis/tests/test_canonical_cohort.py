from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from metric_analysis.pipeline.library.canonical_cohort import (
    CANONICAL_PROTEINS,
    load_all_canonical_cohorts,
    load_canonical_cohort,
    source_family,
)
from metric_analysis.pipeline.library import canonical_cohort


EXPECTED_COUNTS = {
    "POLE": (1074, 38),
    "PTEN": (257, 248),
    "VHL": (124, 105),
    # MutS_alpha was split into its two subunits on 2026-09-07 (147 = 109 + 38).
    # On 2026-10-12 MSH6 moved to the G:T-mismatch rerun (38 -> 143) and PIK3CA
    # joined (91): total 3,643 -> 3,839. TP53 and KRAS left on 2026-10-13 (no
    # binary label; their assays are continuous targets): 2,620. POLE P356S (a
    # passenger with no coordinate structures) left on 2026-10-20: 2,619. FOXA1 (the
    # positional control) left the release on 2026-11-03: 1,798.
    "MSH2": (109, 71),
    "MSH6": (143, 66),
    "PIK3CA": (91, 84),
}


class TestCanonicalCohort(unittest.TestCase):
    def test_subunits_resolve_both_sources_in_current_parent_directory(self):
        # Each subunit reads its own parent complex's folder; MSH6's is the G:T rerun.
        for protein, parent_name in (("MSH2", "MutS_alpha"), ("MSH6", "MutS_alpha_MSH6_GT")):
            with self.subTest(protein=protein), tempfile.TemporaryDirectory() as tmp:
                original = load_canonical_cohort(protein)
                root = Path(tmp)
                parent = root / "PLDDTs" / parent_name
                parent.mkdir(parents=True)
                for column in ("source_file", "wt_source_file"):
                    for value in original[column]:
                        (parent / Path(value).name).touch()
                with mock.patch.object(canonical_cohort, "REPO_ROOT", root):
                    relocated = load_canonical_cohort(protein)
                for column in ("source_file", "wt_source_file"):
                    self.assertTrue(all(Path(p).parent == parent.resolve()
                                        for p in relocated[column]))
                metadata = ["protein", "mutation_code", "label", "source_family",
                            "plddt_row_range_start", "plddt_row_range_stop"]
                pd.testing.assert_frame_equal(original[metadata], relocated[metadata])

    def test_local_path_precedes_existing_external_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkout"
            local = root / "PLDDTs" / "PTEN" / "variant.csv"
            local.parent.mkdir(parents=True)
            local.touch()
            old = Path(tmp) / "variant.csv"
            old.touch()
            with mock.patch.object(canonical_cohort, "REPO_ROOT", root):
                self.assertEqual(canonical_cohort._existing_repo_path(old, protein="PTEN"),
                                 local.resolve())
                self.assertEqual(canonical_cohort._existing_repo_path(
                    "PLDDTs/PTEN/variant.csv", protein="PTEN"), local.resolve())

    def test_external_fallback_and_missing_path_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkout"
            external = Path(tmp) / "variant.csv"
            external.touch()
            with mock.patch.object(canonical_cohort, "REPO_ROOT", root):
                self.assertEqual(canonical_cohort._existing_repo_path(external, protein="PTEN"),
                                 external.resolve())
                self.assertEqual(canonical_cohort._existing_repo_path(
                    Path(tmp) / "missing.csv", protein="PTEN"),
                    (root / "PLDDTs/PTEN/missing.csv").resolve())

    def test_counts_and_unique_keys(self):
        cohort = load_all_canonical_cohorts()
        self.assertEqual(tuple(cohort["protein"].drop_duplicates()), CANONICAL_PROTEINS)
        self.assertEqual(len(cohort), 1798)
        self.assertEqual(sum(n for n, _ in EXPECTED_COUNTS.values()), 1798)
        for protein, (expected_n, expected_positive) in EXPECTED_COUNTS.items():
            block = cohort[cohort["protein"] == protein]
            self.assertEqual(len(block), expected_n)
            self.assertEqual(int(block["label"].sum()), expected_positive)
            self.assertFalse(block["mutation_code"].duplicated().any())
            self.assertTrue(block["source_exists"].all())
            self.assertTrue(block["wt_exists"].all())

    def test_pole_construct_variants_are_not_lost_to_filename_parsing(self):
        pole = load_canonical_cohort("POLE")
        construct = pole[pole["source_family"] == "POLE_268-471"]
        self.assertEqual(len(construct), 8)
        self.assertTrue(construct["label"].eq(1).all())
        self.assertLessEqual({"P286H", "P286R"}, set(construct["mutation_code"]))
        self.assertTrue(all("POLE_268-471_" in Path(p).name for p in construct["source_file"]))

    def test_source_family_preserves_muts_subunit_and_pole_construct(self):
        self.assertEqual(
            source_family("POLE", "POLE_268-471_P286H_plddt_values.csv"),
            "POLE_268-471",
        )
        self.assertEqual(source_family("POLE", "POLE_L424F_plddt_values.csv"), "POLE_other")
        self.assertEqual(
            source_family("MutS_alpha", "MutS_alpha_MSH2_A256G_plddt_values.csv"), "MSH2"
        )
        self.assertEqual(
            source_family("MutS_alpha", "MutS_alpha_MSH6_A25T_plddt_values.csv"), "MSH6"
        )

    def test_label_text_and_numeric_value_agree(self):
        cohort = load_all_canonical_cohorts()
        expected = cohort["mutation_type"].eq("Driver").astype(int)
        pd.testing.assert_series_equal(
            cohort["label"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_assay_genes_have_no_binary_cohort(self):
        # 2026-10-13: thresholding an assay into driver/passenger is gone.
        for protein in ("TP53", "KRAS"):
            self.assertNotIn(protein, CANONICAL_PROTEINS)
            with self.assertRaises(ValueError):
                load_canonical_cohort(protein)

    def test_transcript_invalid_and_start_loss_rows_are_excluded(self):
        excluded = {
            "PTEN": {"M1I", "M1L", "M1R", "M1T", "M1V"},
            "VHL": {"E186K", "E189D", "E189K", "L128V", "L135V", "N131D"},
            # these four initiator-codon variants are MSH6's
            "MSH6": {"M1I", "M1L", "M1T", "M1V"},
        }
        for protein, codes in excluded.items():
            self.assertTrue(codes.isdisjoint(set(load_canonical_cohort(protein)["mutation_code"])))

        audit = pd.read_csv(
            Path(__file__).resolve().parents[1] / "manifests" / "strict_canonical_exclusions.csv"
        )
        reasons = audit.groupby("exclusion_reason").size().to_dict()
        self.assertEqual(
            reasons,
            {
                "initiator_codon_variant_not_representable_as_full_length_missense": 9,
                "no_target_gene_nm_transcript_protein_substitution": 6,
                "user_decision_2026-10-20_no_coordinate_structures_one_cohort_for_all_descriptors": 1,
            },
        )
        provenance = {
            "source_record_name", "source_record_protein_change",
            "source_record_classification", "source_record_accession",
            "source_record_variation_id", "source_record_transcript",
        }
        self.assertLessEqual(provenance, set(audit.columns))
        # POLE P356S left by decision (no coordinate structures), not by a ClinVar record,
        # so it is the one row without ClinVar provenance.
        decided = audit["exclusion_reason"].str.startswith("user_decision_")
        self.assertEqual(list(zip(audit.loc[decided, "protein"], audit.loc[decided, "mutation_code"])),
                         [("POLE", "P356S")])
        self.assertFalse(audit.loc[~decided, list(provenance)].isna().any().any())


if __name__ == "__main__":
    unittest.main()
