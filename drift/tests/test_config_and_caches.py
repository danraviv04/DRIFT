"""Targeted regressions for configurable proteins and the score-cache builder."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from drift import run
from drift.core import apply_configured_label_policies, build_binary_label_table
from drift.protein_config import (
    SUPPORTED_LABEL_MODES,
    build_structure_configs,
    load_continuous_configs,
    load_protein_configs,
    protein_config_fingerprint,
)
from drift.structure_draws import SamplingPolicy, variant_keys


class TestProteinConfig(unittest.TestCase):
    def test_existing_proteins_and_filename_aliases_are_configured(self):
        configs = load_protein_configs()
        # MutSalpha is present in the JSON but disabled since the 2026-09-07
        # subunit split, so load_protein_configs() drops it: this config is
        # iterated wholesale, and leaving it active put the same 147 variants
        # into the variant table twice under two identities.
        # TP53 and KRAS are disabled for the binary pipeline since 2026-10-13
        # (no binary label); their structure sources are read by load_continuous_configs.
        self.assertEqual(set(configs), {
            "POLE", "MSH2", "MSH6", "VHL", "PTEN", "PIK3CA",
        })
        runtime = build_structure_configs(configs)
        continuous = build_structure_configs(load_continuous_configs(("TP53", "KRAS")))
        self.assertEqual(continuous["TP53"]["mut"], "P53")
        self.assertEqual(continuous["KRAS"]["read_root"].name, "KRAS")
        # Each subunit reads its modelled complex's files and slices its own rows.
        # MSH6's complex is the G:T-mismatch rerun since 2026-10-12.
        for subunit, parent, block in (("MSH2", "MutS_alpha", [0, 934]),
                                       ("MSH6", "MutS_alpha_MSH6_GT", [934, 2294])):
            self.assertEqual(runtime[subunit]["mut"], parent)
            self.assertEqual(runtime[subunit]["plddt_row_range"], block)
            self.assertEqual(runtime[subunit]["mut_csv_glob"],
                             f"*_{subunit}_*plddt_values.csv")

    def test_mutation_parser_ignores_codes_inside_gene_names(self):
        # Unbounded, "PIK3CA_H1047R" parsed as "K3C" -- every PIK3CA file, WT
        # included, collapsed onto one variant.
        from drift.structure_draws import normalize_mutation
        self.assertEqual(normalize_mutation("PIK3CA_H1047R_plddt_values.csv"), "H1047R")
        self.assertIsNone(normalize_mutation("PIK3CA_WT_plddt_values.csv"))
        self.assertEqual(normalize_mutation("P53_R175H_plddt_values.csv"), "R175H")
        self.assertEqual(normalize_mutation("POLE_268-471_P286R_plddt_values.csv"), "P286R")
        self.assertEqual(normalize_mutation("MutS_alpha_MSH6_GT_A1055P"), "A1055P")
        self.assertEqual(normalize_mutation("R175H"), "R175H")

    def test_pole_uses_explicit_binary_labels_and_keeps_unlisted_unknown(self):
        configs = load_protein_configs()
        label_cfg = configs["POLE"]["label"]
        self.assertEqual(label_cfg["mode"], "binary_csv")
        self.assertEqual(label_cfg["path"].name, "combined_pole_labels.csv")

        frame = pd.DataFrame({
            "gene": ["POLE", "POLE", "POLE"],
            "mutation": ["D275A", "A341D", "Q999Z"],
            "label": [1, 0, -1],
            "gt_source": ["explicit", "explicit", "none"],
        })
        result = apply_configured_label_policies(frame, {"POLE": configs["POLE"]})
        self.assertEqual(result["label"].tolist(), [1, 0, -1])

    def test_assay_genes_have_no_binary_label(self):
        # 2026-10-13: no assay is thresholded into driver/passenger. TP53/KRAS
        # carry the label-free continuous_assay mode and cannot be enabled.
        continuous = load_continuous_configs(("TP53", "KRAS"))
        for gene in ("TP53", "KRAS"):
            self.assertEqual(continuous[gene]["label"]["mode"], "continuous_assay")
        self.assertNotIn("score_threshold", SUPPORTED_LABEL_MODES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proteins.json"
            path.write_text(json.dumps({"KRAS": {
                "structures": {"mut": "KRAS", "read_root": "PLDDTs/KRAS", "chain_ids": ["A"]},
                "label": {"mode": "continuous_assay", "target": "DMS"},
            }}))
            with self.assertRaisesRegex(ValueError, "no binary label"):
                load_protein_configs(path)

    def test_missing_label_strategy_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proteins.json"
            path.write_text(json.dumps({
                "NEW": {
                    "structures": {"mut": "NEW", "read_root": "PLDDTs/NEW", "chain_ids": ["A"]}
                }
            }))
            with self.assertRaisesRegex(ValueError, "label"):
                load_protein_configs(path)

    def test_missing_structure_section_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proteins.json"
            path.write_text(json.dumps({
                "NEW": {"label": {"mode": "positive_list", "path": "labels.csv"}}
            }))
            with self.assertRaisesRegex(ValueError, "structures"):
                load_protein_configs(path)

    def test_variant_keys_do_not_drop_a_new_configured_gene(self):
        rows = pd.DataFrame({"gene": ["KRAS"], "mutation": ["G12D"]})
        self.assertEqual(variant_keys(rows).to_dict("records"), [
            {"gene": "KRAS", "mutation": "G12D"}
        ])

    def test_variant_keys_normalise_and_deduplicate(self):
        rows = pd.DataFrame({
            "gene": ["MSH2", "MSH2", "PTEN"],
            "mutation": ["MSH2_A256G", "A256G", "R130Q"],
            "Num_Structures": [90, 90, 90],
        })
        self.assertEqual(variant_keys(rows).to_dict("records"), [
            {"gene": "MSH2", "mutation": "A256G"}, {"gene": "PTEN", "mutation": "R130Q"},
        ])

    def test_strict_manifest_controls_membership_and_labels(self):
        frame = pd.DataFrame({
            "gene": ["TP53", "TP53", "MutSalpha", "VHL"],
            "mutation": ["R1A", "R2A", "M1A", "extra"],
            "label": [0, 0, 1, 1],
            "afm_score": [0.1, 0.2, 0.3, 0.4],
        })
        manifest = pd.DataFrame({
            "protein": ["TP53", "TP53", "MutS_alpha"],
            "mutation_code": ["R1A", "R2A", "M1A"],
            "label": [1, 0, 1],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict.csv"
            manifest.to_csv(path, index=False)
            result = run._apply_strict_canonical_cohort(frame, path)

        self.assertEqual(result[["gene", "mutation"]].values.tolist(), [
            ["TP53", "R1A"], ["TP53", "R2A"], ["MutSalpha", "M1A"],
        ])
        self.assertEqual(result["label"].tolist(), [1, 0, 1])

    def test_strict_manifest_refuses_a_table_missing_a_canonical_variant(self):
        frame = pd.DataFrame({"gene": ["TP53"], "mutation": ["R1A"], "label": [0]})
        manifest = pd.DataFrame({"protein": ["TP53", "TP53"], "mutation_code": ["R1A", "R2A"],
                                 "label": [0, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict.csv"
            manifest.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing 1 canonical"):
                run._apply_strict_canonical_cohort(frame, path)

    def test_generic_binary_label_table_supports_a_new_protein(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame({
                "variant": ["p.G12D", "V9N"], "is_driver": [1, 0]
            }).to_csv(path, index=False)
            out = build_binary_label_table(
                path, "KRAS", mutation_col="variant", label_col="is_driver"
            )
        self.assertEqual(out[["mutation", "label"]].to_dict("records"), [
            {"mutation": "G12D", "label": 1},
            {"mutation": "V9N", "label": 0},
        ])

    def test_cache_fingerprint_ignores_notes_but_not_scientific_fields(self):
        base = {
            "X": {
                "enabled": True,
                "_status": "draft",
                "structures": {"mut": "X", "read_root": "PLDDTs/X", "chain_ids": ["A"]},
                "label": {"mode": "positive_list", "path": "labels.csv"},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proteins.json"
            path.write_text(json.dumps(base))
            first = protein_config_fingerprint(path)
            base["X"]["_status"] = "ready"
            path.write_text(json.dumps(base))
            self.assertEqual(first, protein_config_fingerprint(path))
            base["X"]["structures"]["chain_ids"] = ["B"]
            path.write_text(json.dumps(base))
            self.assertNotEqual(first, protein_config_fingerprint(path))

    def test_ground_truth_groups_match_enabled_label_routes(self):
        # One binary label route since 2026-10-13: annotation. Every enabled
        # binary dataset belongs to it (the FOXA1 control left on 2026-11-03).
        self.assertEqual(set(run.GT_GROUPS), {"annotation"})
        self.assertEqual(run.GT_GROUPS["annotation"], {
            "POLE", "MSH2", "MSH6", "VHL", "PTEN", "PIK3CA",
        })
        self.assertEqual(run.GT_GROUPS["annotation"], set(load_protein_configs()))


class TestCacheBuilder(unittest.TestCase):
    def test_cache_name_carries_size_seed_fingerprint_and_policy(self):
        fp = protein_config_fingerprint()
        default = run.cache_path(90, 42)
        self.assertEqual(default.parent, run.CACHE_DIR)
        self.assertEqual(default.name,
                         f"labeled_size90_seed42_proteins{fp}_esfeat{run.ES_CACHE_VERSION}.csv")
        other = run.cache_path(5, 43, SamplingPolicy(mode="independent"))
        self.assertTrue(other.name.startswith(f"labeled_size5_seed43_proteins{fp}_"))
        self.assertNotEqual(other.name.removesuffix(".csv"), default.name.removesuffix(".csv"))

    def test_every_size_and_seed_builds_its_own_dataset(self):
        calls = []

        def fake_build(size, cfg, force=False):
            calls.append((size, cfg["seed"], cfg["max_structures_per_variant"],
                          cfg["sampling_mode"], cfg["selection"], force))

        with mock.patch.object(run, "build_or_load_dataset", side_effect=fake_build):
            run.main(["--sizes", "1,5", "--seeds", "42-43", "--force-recompute"])
        self.assertEqual(calls, [
            (1, 42, 1, "nested", "random", True), (5, 42, 5, "nested", "random", True),
            (1, 43, 1, "nested", "random", True), (5, 43, 5, "nested", "random", True),
        ])

    def test_cli_rejects_bad_sizes_seeds_and_policy(self):
        for argv in (["--sizes", "0"], ["--seeds", "51-42"],
                     ["--selection", "top_confidence", "--sampling-mode", "independent"]):
            with self.subTest(argv=argv), mock.patch.object(run, "build_or_load_dataset"), \
                    self.assertRaises(SystemExit):
                run.main(argv)

    def test_cached_table_is_reused_and_restricted_to_the_cohort(self):
        from drift.ensemble_shift import FEATURE_COLUMNS
        frame = pd.DataFrame({"gene": ["VHL"], "mutation": ["R167W"], "label": [1],
                              "afm_score": [0.9], **{c: [0.5] for c in FEATURE_COLUMNS}})
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(run, "CACHE_DIR", Path(directory)), \
                mock.patch.object(run, "_apply_strict_canonical_cohort", side_effect=lambda d: d), \
                mock.patch.object(run.core, "build_labeled_dataset_for_config") as rebuild:
            cfg = run._resolve_run_config({"seed": 42, "max_structures_per_variant": 90})
            frame.to_csv(run.cache_path(90, 42), index=False)
            out = run.build_or_load_dataset(90, cfg)
        rebuild.assert_not_called()
        self.assertEqual(out.to_dict("records"), frame.to_dict("records"))


if __name__ == "__main__":
    unittest.main()
