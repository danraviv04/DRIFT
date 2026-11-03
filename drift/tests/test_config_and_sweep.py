"""Targeted regressions for configurable proteins and seed-aware D1 sweeps."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from drift.structure_draws import aggregate_d1_rows_to_mutation_level
from drift import run
from drift.benchmark import (
    evaluate_fixed_grid,
    evaluate_nested_selection,
    make_specs,
    sample_weights,
)
from drift.core import (
    _normalize_artifact_scores,
    apply_configured_label_policies,
    build_binary_label_table,
    featurize_compact4_np,
    featurize_compact4_v2_np,
    featurize_fusion_representation_np,
    fit_fusion_feature_preprocessor_np,
    normalize_fusion_scores_np,
    transform_fusion_features_np,
)
from drift.protein_config import (
    SUPPORTED_LABEL_MODES,
    build_d1_runtime_configs,
    load_continuous_configs,
    load_protein_configs,
    protein_config_fingerprint,
)


class TestProteinConfig(unittest.TestCase):
    def test_existing_proteins_and_filename_aliases_are_configured(self):
        configs = load_protein_configs()
        # MutSalpha is present in the JSON but disabled since the 2026-09-07
        # subunit split, so load_protein_configs() drops it: this config is
        # iterated wholesale, and leaving it active put the same 147 variants
        # into the D1 table twice under two identities.
        # TP53 and KRAS are disabled for the binary pipeline since 2026-10-13
        # (no binary label); their D1 sources are read by load_continuous_configs.
        self.assertEqual(set(configs), {
            "POLE", "MSH2", "MSH6", "VHL", "PTEN", "PIK3CA",
        })
        runtime = build_d1_runtime_configs(configs)
        continuous = build_d1_runtime_configs(load_continuous_configs(("TP53", "KRAS")))
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
        result = apply_configured_label_policies(
            frame, {"POLE": configs["POLE"]}, {}
        )
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
                "d1": {"mut": "KRAS", "read_root": "PLDDTs/KRAS", "chain_ids": ["A"]},
                "label": {"mode": "continuous_assay", "target": "DMS"},
            }}))
            with self.assertRaisesRegex(ValueError, "no binary label"):
                load_protein_configs(path)

    def test_missing_label_strategy_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proteins.json"
            path.write_text(json.dumps({
                "NEW": {
                    "d1": {"mut": "NEW", "read_root": "PLDDTs/NEW", "chain_ids": ["A"]}
                }
            }))
            with self.assertRaisesRegex(ValueError, "label"):
                load_protein_configs(path)

    def test_runtime_aggregation_does_not_drop_a_new_configured_gene(self):
        rows = pd.DataFrame({
            "gene": ["KRAS"], "mutation": ["G12D"], "d1_score": [0.2]
        })
        out = aggregate_d1_rows_to_mutation_level(rows)
        self.assertEqual(out[["gene", "mutation"]].to_dict("records"), [
            {"gene": "KRAS", "mutation": "G12D"}
        ])

    def test_strict_manifest_controls_fusion_membership_and_labels(self):
        frame = pd.DataFrame({
            "gene": ["TP53", "TP53", "MutSalpha", "VHL"],
            "mutation": ["R1A", "R2A", "M1A", "extra"],
            "label": [0, 0, 1, 1],
            "d1_score": [0.1, 0.2, 0.3, 0.4],
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
                "d1": {"mut": "X", "read_root": "PLDDTs/X", "chain_ids": ["A"]},
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
            base["X"]["d1"]["chain_ids"] = ["B"]
            path.write_text(json.dumps(base))
            self.assertNotEqual(first, protein_config_fingerprint(path))


class TestSeedAwareSweep(unittest.TestCase):
    def test_every_size_and_seed_builds_its_own_d1_dataset(self):
        calls = []
        logo_calls = []
        frame = pd.DataFrame({
            "gene": ["A", "A", "B", "B"],
            "mutation": ["A1B", "A2B", "A3B", "A4B"],
            "label": [0, 1, 0, 1],
            "afm_score": [0.1, 0.9, 0.2, 0.8],
            "d1_score": [0.2, 0.8, 0.3, 0.7],
        })

        def fake_build(size, cfg, force=False):
            calls.append((size, cfg["seed"], force))
            return frame.copy()

        def fake_logo(
            df, cfg, feature_cols=None, fusion_representation="engineered6",
            fusion_normalization="original", models=None, progress_callback=None,
        ):
            logo_calls.append((fusion_representation, fusion_normalization))
            panel = [run.FUSION_MODEL_ROW_NAME[m]
                     for m in (models or run.DEFAULT_FUSION_MODELS)]
            y = df["label"].to_numpy()
            afm = df["afm_score"].to_numpy()
            d1 = df["d1_score"].to_numpy()
            per_gene = pd.DataFrame([{
                "gene": "A", "n": 2, "n_pos": 1,
                "fusion": 1.0, "logreg": 1.0, "afm": 1.0, "d1": 1.0,
            }])
            for name in panel:
                per_gene[name] = 1.0
            return {
                "y": y, "afm": afm, "d1": d1,
                "models": panel, "oof": {name: afm for name in panel},
                "oof_fusion": afm, "oof_logreg": afm,
                "per_gene": per_gene,
            }

        base_cfg = run._resolve_run_config({"seed": 42, "epochs": 1})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(run, "build_or_load_dataset", side_effect=fake_build), \
                 mock.patch.object(run, "logo_cv", side_effect=fake_logo):
                runs_df, summary = run.run_sweep(
                    base_cfg,
                    sizes=[1, 5],
                    seeds=[42, 43],
                    do_stratified=False,
                    out=Path(directory),
                    fusion_representations=["raw2", "compact4"],
                    fusion_models=["mlp", "logreg"],
                    fusion_normalizations=["original", "rank"],
                    force=True,
                )
            normalization_figure_exists = (
                Path(directory) / "sweep_normalization_ablation.png"
            ).is_file()

        self.assertEqual(calls, [
            (1, 42, True), (5, 42, True),
            (1, 43, True), (5, 43, True),
        ])
        for representation in ["raw2", "compact4"]:
            for normalization in ["original", "rank"]:
                self.assertEqual(logo_calls.count((representation, normalization)), 4)
        self.assertIn("representation", runs_df.columns)
        self.assertIn("normalization", runs_df.columns)
        # Probes representation x normalization coverage of the emitted rows;
        # the model class is incidental. Uses "logreg" rather than "fusion"
        # because the MLP is opt-in since the panel change and "fusion" rows
        # are absent under the default offset,logreg,gbm panel.
        self.assertEqual(set(runs_df.loc[runs_df.model == "logreg", "representation"]), {
            "raw2", "compact4",
        })
        self.assertEqual(set(runs_df.loc[runs_df.model == "logreg", "normalization"]), {
            "original", "rank",
        })
        self.assertTrue((runs_df["mlp_epochs"] == base_cfg["epochs"]).all())
        self.assertTrue((summary["mlp_hidden"] == base_cfg["hidden"]).all())
        self.assertTrue(normalization_figure_exists)

    def test_wandb_separates_training_runs_from_connected_size_curves(self):
        frame = pd.DataFrame({
            "gene": ["A", "A", "B", "B"],
            "mutation": ["A1B", "A2B", "A3B", "A4B"],
            "label": [0, 1, 0, 1],
            "afm_score": [0.1, 0.9, 0.2, 0.8],
            "d1_score": [0.2, 0.8, 0.3, 0.7],
        })
        captured = []
        fake_runs = []

        class FakeRun:
            def __init__(self, config):
                self.config = dict(config)
                self.summary = {}
                self.logged = []
                self.finished = False

            def log(self, values):
                self.logged.append(dict(values))

            def finish(self):
                self.finished = True

        def fake_start(project, entity, group, name, config, tags=None):
            captured.append(dict(config))
            fake_run = FakeRun(config)
            fake_run.group = group
            fake_run.name = name
            fake_run.tags = list(tags or [])
            fake_runs.append(fake_run)
            return fake_run

        def fake_logo(
            df, cfg, feature_cols=None, fusion_representation="engineered6",
            fusion_normalization="original", progress_callback=None, models=None,
        ):
            y = df["label"].to_numpy()
            afm = df["afm_score"].to_numpy()
            d1 = df["d1_score"].to_numpy()
            panel = [run.FUSION_MODEL_ROW_NAME[m]
                     for m in (models or run.DEFAULT_FUSION_MODELS)]
            result = {
                "y": y, "afm": afm, "d1": d1,
                "models": panel, "oof": {name: afm for name in panel},
                "oof_fusion": afm, "oof_logreg": afm,
                "per_gene": pd.DataFrame([{
                    **{name: 1.0 for name in panel},
                    "gene": "A", "n": 2, "n_pos": 1,
                    "fusion": 1.0, "logreg": 1.0, "afm": 1.0, "d1": 1.0,
                    "fusion_ap": 1.0, "logreg_ap": 1.0, "afm_ap": 1.0,
                    "fusion_f1_05": 1.0, "fusion_precision_05": 1.0,
                    "fusion_recall_05": 1.0, "fusion_mcc_05": 1.0,
                    "best_epoch": 1,
                }]),
                "history": pd.DataFrame([{
                    "epoch": 1, "phase": "select", "held_out_gene": "A",
                    "train_loss": 0.5, "val_loss": 0.6, "val_ap": 0.8,
                    "val_auc": 0.9, "val_f1_05": 0.7,
                    "val_precision_05": 0.75, "val_recall_05": 0.65,
                    "val_mcc_05": 0.6,
                }]),
            }
            if progress_callback is not None:
                progress_callback(1, 1, {"held_out_gene": "A"})
            return result

        base_cfg = run._resolve_run_config({"seed": 42, "epochs": 1})
        fake_wandb = types.SimpleNamespace()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(run, "build_or_load_dataset", return_value=frame), \
             mock.patch.object(run, "logo_cv", side_effect=fake_logo), \
             mock.patch.object(run, "_wandb_start", side_effect=fake_start), \
             mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            run.run_sweep(
                base_cfg, sizes=[1, 5], seeds=[42], do_stratified=False,
                out=Path(directory),
                fusion_representations=["raw2", "compact4_v2"],
                fusion_models=["mlp", "logreg"],
                fusion_normalizations=["original", "robust", "rank"],
                wandb_enabled=True,
            )

        training_runs = [
            fake_run for fake_run in fake_runs
            if fake_run.config["wandb_role"] != "ensemble-size evaluation curve"
        ]
        curve_runs = [
            fake_run for fake_run in fake_runs
            if fake_run.config["wandb_role"] == "ensemble-size evaluation curve"
        ]
        self.assertEqual(len(training_runs), 12)
        self.assertEqual(len(curve_runs), 6)
        self.assertEqual(len(captured), 18)
        # Run names carry the fusion partner: without it, partner=d1 and
        # partner=ks_residue produced identical names and collided in the W&B UI.
        self.assertEqual(
            {fake_run.name for fake_run in training_runs},
            {
                f"all-d1-{representation}-{normalization}-base-seed42-N{size}"
                for representation in ("raw2", "compact4_v2")
                for normalization in ("original", "robust", "rank")
                for size in (1, 5)
            },
        )
        # Tags are what the W&B UI can actually filter on, unlike names.
        for fake_run in training_runs:
            self.assertIn("cv:logo", fake_run.tags)
            self.assertIn("partner:d1", fake_run.tags)
            self.assertIn("gt:all", fake_run.tags)
        self.assertTrue(all(
            fake_run.group.startswith("fusion-all-")
            for fake_run in training_runs
        ))
        self.assertEqual(
            {fake_run.name for fake_run in curve_runs},
            {
                f"all-d1-{representation}-{normalization}-base-size-curve-seed42"
                for representation in ("raw2", "compact4_v2")
                for normalization in ("original", "robust", "rank")
            },
        )
        self.assertTrue(all(
            fake_run.group.startswith("fusion-size-curve-all-")
            for fake_run in curve_runs
        ))
        self.assertEqual(
            {config["normalization"] for config in captured},
            {"original", "robust", "rank"},
        )
        self.assertEqual(
            {
                config["normalization_scope"]
                for config in captured
                if config["normalization"] != "original"
            },
            {"within-gene complete unlabelled batch"},
        )
        for training_run in training_runs:
            logged_keys = set().union(*(
                values.keys() for values in training_run.logged
            ))
            self.assertIn("training/train_loss_mean", logged_keys)
            self.assertIn("training/inner_val_loss_mean", logged_keys)
            self.assertIn("training/inner_val_f1_at_0_5_mean", logged_keys)
            self.assertNotIn("evaluation/fusion_macro_auc", logged_keys)
            self.assertIn("evaluation/fusion_macro_auc", training_run.summary)
            self.assertIn(
                "evaluation/fusion_macro_precision_at_0_5",
                training_run.summary,
            )
            self.assertEqual(training_run.summary["progress/status"], "completed")
            self.assertTrue(training_run.finished)
        for curve_run in curve_runs:
            self.assertEqual(
                [row["d1_size"] for row in curve_run.logged],
                [1, 5],
            )
            for row in curve_run.logged:
                self.assertIn("evaluation/fusion_macro_auc", row)
                self.assertIn("evaluation/fusion_macro_f1_at_0_5", row)
                self.assertIn("evaluation/fusion_macro_precision_at_0_5", row)
            self.assertEqual(
                curve_run.summary["evaluation/curve_status"], "completed"
            )
            self.assertTrue(curve_run.finished)

    def test_wandb_start_defines_shared_training_and_evaluation_axes(self):
        initialized = []
        metric_defs = []

        class FakeRun:
            pass

        fake_wandb = types.SimpleNamespace(
            init=lambda **kwargs: initialized.append(kwargs) or FakeRun(),
            define_metric=lambda *args, **kwargs: metric_defs.append((args, kwargs)),
        )
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            run._wandb_start("project", "entity", "group", "name", {"seed": 42})

        self.assertEqual(
            initialized[0]["config"]["wandb_role"],
            "candidate training and evaluation monitor",
        )
        self.assertEqual(
            initialized[0]["config"]["scientific_results"],
            "local CSV tables and figures",
        )
        names = [args[0] for args, _ in metric_defs]
        self.assertIn("progress/fraction", names)
        self.assertIn("d1_size", names)
        self.assertIn("evaluation/fusion_macro_auc", names)
        self.assertIn("evaluation/fusion_macro_f1_at_0_5", names)
        self.assertIn("training/epoch", names)
        self.assertIn("training/train_loss_mean", names)
        self.assertIn("training/inner_val_precision_at_0_5_mean", names)
        self.assertNotIn("*", names)

    def test_wandb_progress_maps_fold_completion_to_global_fraction(self):
        class FakeRun:
            def __init__(self):
                self.summary = {}
                self.logged = []

            def log(self, values):
                self.logged.append(dict(values))

        fake_run = FakeRun()
        progress = run._WandbProgress(fake_run, total_evaluations=2)
        first = progress.callback("first candidate")
        first(1, 2, {"held_out_gene": "A"})
        first(2, 2, {"held_out_gene": "B"})
        progress.complete("first candidate")
        second = progress.callback("second candidate")
        second(1, 1, {"held_out_gene": "C"})
        progress.complete("second candidate")
        progress.finish()

        fractions = [row["progress/fraction"] for row in fake_run.logged]
        self.assertEqual(fractions, [0.0, 0.25, 0.5, 1.0])
        self.assertTrue(all(b >= a for a, b in zip(fractions, fractions[1:])))
        self.assertEqual(fake_run.summary["progress/completed_evaluations"], 2)
        self.assertEqual(fake_run.summary["progress/status"], "completed")


class TestFusionRepresentations(unittest.TestCase):
    @staticmethod
    def _frame():
        return pd.DataFrame({
            "gene": ["A", "A", "B", "B"],
            "mutation": ["A1B", "A2B", "A3B", "A4B"],
            "label": [0, 1, 0, 1],
            "afm_score": [0.1, 0.9, 0.2, 0.8],
            "d1_score": [2.0, 3.0, 4.0, 5.0],
        })

    def test_compact4_has_the_declared_algebra(self):
        raw = np.array([[0.2, 2.0], [0.5, 3.0]], dtype=np.float32)
        got = featurize_compact4_np(raw)
        expected = np.array([
            [0.2, 2.0, 4.0, 0.4],
            [0.5, 3.0, 9.0, 1.5],
        ], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_compact4_v2_has_the_declared_algebra_and_fold_safe_scaling(self):
        train = np.array([[0.1, 2.0], [0.3, 4.0], [0.5, 6.0]], dtype=np.float32)
        validation = np.array([[0.7, 8.0]], dtype=np.float32)
        expected = np.array([
            [0.1, 2.0, 0.2, 3.61],
            [0.3, 4.0, 1.2, 13.69],
            [0.5, 6.0, 3.0, 30.25],
        ], dtype=np.float32)
        np.testing.assert_allclose(featurize_compact4_v2_np(train), expected)

        xtr, xva, preprocessing = fit_fusion_feature_preprocessor_np(
            train, validation, representation="compact4_v2",
        )
        xtr_changed, _, changed = fit_fusion_feature_preprocessor_np(
            train, validation * 100, representation="compact4_v2",
        )
        np.testing.assert_allclose(xtr_changed, xtr, atol=1e-6)
        np.testing.assert_allclose(changed["scaler"].mean_, preprocessing["scaler"].mean_)
        np.testing.assert_allclose(
            transform_fusion_features_np(validation, "compact4_v2", preprocessing),
            xva, atol=1e-6,
        )

    def test_representation_dimensions_and_unknown_choice(self):
        raw = np.array([[0.2, 2.0], [0.5, 3.0]], dtype=np.float32)
        self.assertEqual(featurize_fusion_representation_np(raw, "raw2").shape, (2, 2))
        self.assertEqual(featurize_fusion_representation_np(raw, "compact4_v2").shape, (2, 4))
        self.assertEqual(featurize_fusion_representation_np(raw, "compact4").shape, (2, 4))
        self.assertEqual(featurize_fusion_representation_np(raw, "engineered6").shape, (2, 6))
        with self.assertRaisesRegex(ValueError, "unknown fusion representation"):
            featurize_fusion_representation_np(raw, "mystery")

    def test_shared_normalizations_are_per_gene_and_label_free(self):
        raw = np.array([
            [1.0, 10.0], [3.0, 30.0], [5.0, 20.0],
            [10.0, 2.0], [20.0, 4.0], [30.0, 6.0],
        ], dtype=np.float32)
        genes = np.array(["A", "A", "A", "B", "B", "B"])
        np.testing.assert_array_equal(
            normalize_fusion_scores_np(raw, genes, "original"), raw,
        )
        robust = normalize_fusion_scores_np(raw, genes, "robust")
        np.testing.assert_allclose(
            robust[:3],
            np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )
        ranks = normalize_fusion_scores_np(raw, genes, "rank")
        np.testing.assert_allclose(
            ranks[:3],
            np.array([[1 / 3, 1 / 3], [2 / 3, 1.0], [1.0, 2 / 3]], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "unknown normalization"):
            normalize_fusion_scores_np(raw, genes, "mystery")

    def test_transductive_artifact_prediction_requires_gene_batch(self):
        raw = np.array([[0.2, 2.0], [0.5, 3.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "complete prediction batch"):
            _normalize_artifact_scores(raw, {"normalization": "rank"}, genes=None)
        got = _normalize_artifact_scores(
            raw, {"normalization": "rank"}, genes=np.array(["A", "A"]),
        )
        np.testing.assert_allclose(
            got, np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32),
        )

    def test_logo_mlp_and_logreg_receive_identical_fold_matrices(self):
        mlp_train, mlp_val, lr_train, lr_val = [], [], [], []
        progress_events = []

        def fake_train(xtr, ytr, xva, yva, **kwargs):
            mlp_train.append(np.asarray(xtr).copy())
            mlp_val.append(np.asarray(xva).copy())
            return {
                "val_probs": np.linspace(0.2, 0.8, len(yva), dtype=np.float32),
                "best_epoch": 1,
                "used_fixed_epochs": False,
            }

        class FakeLogReg:
            def fit(self, x, y):
                lr_train.append(np.asarray(x).copy())
                return self

            def predict_proba(self, x):
                lr_val.append(np.asarray(x).copy())
                p = np.linspace(0.2, 0.8, len(x), dtype=np.float32)
                return np.column_stack([1.0 - p, p])

        cfg = run._resolve_run_config({"seed": 42, "epochs": 1})
        with mock.patch.object(run, "train_one_fold", side_effect=fake_train), \
             mock.patch.object(run, "LogisticRegression", return_value=FakeLogReg()):
            result = run.logo_cv(
                self._frame(), cfg, fusion_representation="compact4_v2",
                fusion_normalization="rank",
                progress_callback=lambda completed, total, detail: progress_events.append(
                    (completed, total, detail["held_out_gene"])
                ),
            )

        self.assertEqual(len(mlp_train), 2)
        for mt, mv, lt, lv in zip(mlp_train, mlp_val, lr_train, lr_val):
            self.assertEqual(mt.shape[1], 4)
            np.testing.assert_allclose(mt, lt)
            np.testing.assert_allclose(mv, lv)
        self.assertEqual(progress_events, [(1, 2, "A"), (2, 2, "B")])
        self.assertTrue({
            "fusion_ap", "logreg_ap", "fusion_f1_05",
            "fusion_precision_05", "fusion_recall_05", "fusion_mcc_05",
        }.issubset(result["per_gene"].columns))

    def test_training_history_contains_loss_and_inner_validation_diagnostics(self):
        run.seed_everything(42)
        x_train = np.array([
            [-1.0, -0.8], [-0.8, -1.0], [-0.5, -0.4], [-0.3, -0.6],
            [0.3, 0.5], [0.5, 0.3], [0.8, 1.0], [1.0, 0.8],
        ], dtype=np.float32)
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        out = run.train_one_fold(
            x_train, y_train, x_train[:4], np.array([0, 0, 1, 1]),
            epochs=2, batch_size=4, hidden=4, dropout=0.0,
            lr=1e-3, weight_decay=0.0, patience=2, model_kind="plain",
        )
        selected = out["history"][out["history"]["phase"] == "select"]
        self.assertFalse(selected.empty)
        expected = {
            "train_loss", "val_loss", "val_ap", "val_auc", "val_f1_05",
            "val_precision_05", "val_recall_05", "val_mcc_05",
        }
        self.assertTrue(expected.issubset(selected.columns))
        self.assertTrue(np.isfinite(selected[list(expected)].to_numpy()).all())


class TestCleanBenchmark(unittest.TestCase):
    @staticmethod
    def _frame():
        rows = []
        for gi, gene in enumerate(["A", "B", "C", "D"]):
            for i, label in enumerate([0, 0, 1, 1]):
                rows.append({
                    "gene": gene,
                    "mutation": f"X{gi * 10 + i}Y",
                    "label": label,
                    "afm_score": 0.1 + 0.25 * label + 0.01 * gi + 0.01 * i,
                    "d1_score": 0.2 + 0.20 * label + 0.02 * gi + 0.01 * i,
                })
        return pd.DataFrame(rows)

    def test_gene_balancing_retains_rows_and_equalizes_gene_class_cells(self):
        y = [0, 0, 1, 0, 1, 1]
        genes = ["A", "A", "A", "B", "B", "B"]
        weights = sample_weights(y, genes, "gene")
        self.assertEqual(len(weights), len(y))
        self.assertTrue((weights > 0).all())
        frame = pd.DataFrame({"y": y, "gene": genes, "weight": weights})
        cell_mass = frame.groupby(["gene", "y"])["weight"].sum()
        self.assertAlmostEqual(float(cell_mass.max()), float(cell_mass.min()))

    def test_fixed_and_nested_benchmarks_emit_macro_and_per_gene_rows(self):
        specs = make_specs(["raw2"], ["variant"], ["original"], [1.0])
        fixed = evaluate_fixed_grid(
            self._frame(), specs, size=5, d1_seed=42, model_seed=2026,
        )
        candidate = fixed[fixed["candidate_id"] == specs[0].candidate_id]
        self.assertEqual(set(candidate["scope_type"]), {
            "gene", "macro", "pooled_diagnostic",
        })
        self.assertEqual(int((candidate["scope_type"] == "gene").sum()), 4)

        nested, selections = evaluate_nested_selection(
            self._frame(), specs, size=5, d1_seed=42, model_seed=2026,
        )
        self.assertEqual(set(selections["outer_gene"]), {"A", "B", "C", "D"})
        self.assertEqual(int((nested["scope_type"] == "gene").sum()), 4)

    def test_ground_truth_groups_match_enabled_label_routes(self):
        # One binary label route since 2026-10-13: annotation. Every enabled
        # binary dataset belongs to it (the FOXA1 control left on 2026-11-03).
        self.assertEqual(set(run.GT_GROUPS), {"annotation"})
        self.assertEqual(run.GT_GROUPS["annotation"], {
            "POLE", "MSH2", "MSH6", "VHL", "PTEN", "PIK3CA",
        })
        enabled = set(load_protein_configs())
        self.assertEqual(run.GT_GROUPS["annotation"], enabled)
        for retired in ("functional", "evaluation"):
            with self.assertRaises(SystemExit):
                run._gt_group_list(retired)


if __name__ == "__main__":
    unittest.main()
