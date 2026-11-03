"""Thorough fast tests for the clean AFM+D1 benchmark.

The tests use small synthetic genes and never touch the real pLDDT cache, W&B,
or the network.  They pin the scientific definitions (macro vs pooled AUC,
weighting, label-free transformations, nested split boundaries) as well as the
output contract consumed by the results gallery.
"""

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from drift import benchmark, run
from drift.benchmark import (
    BenchmarkSpec,
    _contrast_text_color,
    _auc_rows,
    _fit_offset_predict,
    _safe_auc,
    _summarize,
    _validate_frame,
    _within_gene_scores,
    evaluate_fixed_grid,
    evaluate_nested_selection,
    make_specs,
    run_benchmark_sweep,
    sample_weights,
    write_gt_group_comparison,
)


def synthetic_frame(genes=("A", "B", "C", "D"), per_class=3):
    rows = []
    for gi, gene in enumerate(genes):
        for label in (0, 1):
            for repeat in range(per_class):
                rows.append({
                    "gene": gene,
                    "mutation": f"M{gi}_{label}_{repeat}",
                    "label": label,
                    "afm_score": 0.12 + 0.62 * label + 0.015 * gi + 0.01 * repeat,
                    "d1_score": 0.18 + 0.28 * label + 0.02 * gi + 0.012 * repeat,
                })
    return pd.DataFrame(rows)


class TestBenchmarkGrid(unittest.TestCase):
    def test_default_grid_is_full_unique_factorial(self):
        specs = make_specs()
        self.assertEqual(len(specs), 4 * 2 * 3 * 3)
        self.assertEqual(len({spec.candidate_id for spec in specs}), len(specs))

    def test_grid_order_and_candidate_ids_are_deterministic(self):
        specs = make_specs(["raw2"], ["variant"], ["original"], [0.1, 1, 10])
        self.assertEqual([spec.candidate_id for spec in specs], [
            "raw2__variant__original__C0p1",
            "raw2__variant__original__C1",
            "raw2__variant__original__C10",
        ])

    def test_repeated_cli_values_are_deduplicated(self):
        specs = make_specs(
            ["raw2", "raw2"], ["variant", "variant"],
            ["original", "original"], [1, 1.0],
        )
        self.assertEqual(len(specs), 1)

    def test_unknown_grid_choices_fail(self):
        for kwargs in [
            {"representations": ["mlp"]},
            {"weightings": ["drop_rows"]},
            {"normalizations": ["label_scaled"]},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "unknown"):
                make_specs(**kwargs)

    def test_invalid_regularization_fails(self):
        for value in [0, -1, np.nan, np.inf]:
            with self.subTest(C=value), self.assertRaisesRegex(ValueError, "finite and > 0"):
                make_specs(["raw2"], ["variant"], ["original"], [value])

    def test_empty_grid_axes_fail_instead_of_expanding_to_defaults(self):
        cases = [
            ([], ["variant"], ["original"], [1]),
            (["raw2"], [], ["original"], [1]),
            (["raw2"], ["variant"], [], [1]),
            (["raw2"], ["variant"], ["original"], []),
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "at least one"):
                make_specs(*values)


class TestBenchmarkInputValidation(unittest.TestCase):
    def test_valid_frame_passes(self):
        self.assertIsNone(_validate_frame(synthetic_frame()))

    def test_missing_required_column_fails(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            _validate_frame(synthetic_frame().drop(columns="d1_score"))

    def test_missing_and_infinite_values_fail(self):
        for value, message in [(np.nan, "missing"), (np.inf, "finite")]:
            frame = synthetic_frame()
            frame.loc[0, "d1_score"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                _validate_frame(frame)

    def test_non_binary_labels_fail_without_integer_coercion(self):
        frame = synthetic_frame()
        frame["label"] = frame["label"].astype(float)
        frame.loc[0, "label"] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            _validate_frame(frame)

    def test_duplicate_variant_keys_fail(self):
        frame = pd.concat([synthetic_frame(), synthetic_frame().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate gene/mutation"):
            _validate_frame(frame)

    def test_blank_identifiers_fail(self):
        for column in ["gene", "mutation"]:
            frame = synthetic_frame()
            frame.loc[0, column] = "  "
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "non-empty"):
                _validate_frame(frame)

    def test_afm_must_be_a_probability(self):
        for value in [-0.01, 1.01]:
            frame = synthetic_frame()
            frame.loc[0, "afm_score"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                _validate_frame(frame)

    def test_raw_d1_must_be_non_negative(self):
        frame = synthetic_frame()
        frame.loc[0, "d1_score"] = -0.01
        with self.assertRaisesRegex(ValueError, "D1 scores must be non-negative"):
            _validate_frame(frame)

    def test_ks_columns_must_lie_in_unit_interval(self):
        """KS is a max CDF gap, so [0, 1] is a property of the estimator itself."""
        for column in ("plddt_ks_residue", "plddt_ks_centred", "plddt_ks"):
            for bad in (-0.01, 1.01):
                with self.subTest(column=column, value=bad):
                    frame = synthetic_frame()
                    frame[column] = 0.5
                    frame.loc[0, column] = bad
                    with self.assertRaisesRegex(ValueError, "must lie in \\[0, 1\\]"):
                        _validate_frame(frame)

    def test_ks_columns_are_optional_but_checked_when_present(self):
        frame = synthetic_frame()
        _validate_frame(frame)                      # absent -> fine
        frame["plddt_ks_residue"] = 0.5
        _validate_frame(frame)                      # present and valid -> fine
        frame.loc[1, "plddt_ks_residue"] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _validate_frame(frame)

    def test_logo_requires_multiple_two_class_genes(self):
        with self.assertRaisesRegex(ValueError, "at least two genes"):
            _validate_frame(synthetic_frame(("A",)))
        frame = synthetic_frame()
        frame.loc[frame["gene"] == "A", "label"] = 1
        with self.assertRaisesRegex(ValueError, "one-class.*A"):
            _validate_frame(frame)


class TestLabelFreeNormalizations(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "gene": ["A"] * 4 + ["B"] * 4,
            "mutation": [f"M{i}" for i in range(8)],
            "label": [0, 1, 0, 1, 1, 0, 1, 0],
            "afm_score": [0.1, 0.2, 0.2, 0.9, 0.4, 0.5, 0.6, 0.7],
            "d1_score": [2.0, 4.0, 4.0, 8.0, 10.0, 20.0, 30.0, 40.0],
        })

    def test_original_is_an_exact_numeric_copy(self):
        got = _within_gene_scores(self.frame, "original")
        np.testing.assert_allclose(got, self.frame[["afm_score", "d1_score"]])
        got[0, 0] = 99
        self.assertNotEqual(self.frame.loc[0, "afm_score"], 99)

    def test_robust_has_zero_gene_medians_and_unit_iqrs(self):
        got = _within_gene_scores(self.frame, "robust")
        for gene in ["A", "B"]:
            block = got[self.frame["gene"].to_numpy() == gene]
            np.testing.assert_allclose(np.median(block, axis=0), [0, 0], atol=1e-12)
            np.testing.assert_allclose(
                np.percentile(block, 75, axis=0) - np.percentile(block, 25, axis=0),
                [1, 1], atol=1e-12,
            )

    def test_constant_robust_column_stays_finite_and_inert(self):
        frame = self.frame.copy()
        frame["d1_score"] = 5.0
        got = _within_gene_scores(frame, "robust")
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got[:, 1], 0)

    def test_rank_is_within_gene_and_averages_ties(self):
        got = _within_gene_scores(self.frame, "rank")
        # A's tied AFM=0.2 values occupy ranks 2 and 3 -> average percentile 2.5/4.
        np.testing.assert_allclose(got[[1, 2], 0], [0.625, 0.625])
        self.assertEqual(float(got[4, 0]), 0.25, "B must start its own rank scale")

    def test_normalization_does_not_use_labels(self):
        for method in ["robust", "rank"]:
            flipped = self.frame.copy()
            flipped["label"] = 1 - flipped["label"]
            np.testing.assert_allclose(
                _within_gene_scores(self.frame, method),
                _within_gene_scores(flipped, method),
            )

    def test_unknown_normalization_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown normalization"):
            _within_gene_scores(self.frame, "supervised")


class TestSampleWeights(unittest.TestCase):
    def test_variant_weights_equalize_global_class_mass_and_mean_one(self):
        y = np.array([0, 0, 0, 1])
        genes = np.array(["A", "A", "B", "B"])
        weights = sample_weights(y, genes, "variant")
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[y == 0].sum()), float(weights[y == 1].sum()))
        self.assertTrue((weights > 0).all())

    def test_gene_weights_equalize_every_gene_class_cell(self):
        y = np.array([0, 0, 1, 0, 1, 1])
        genes = np.array(["A", "A", "A", "B", "B", "B"])
        weights = sample_weights(y, genes, "gene")
        frame = pd.DataFrame({"gene": genes, "label": y, "weight": weights})
        masses = frame.groupby(["gene", "label"])["weight"].sum()
        self.assertAlmostEqual(float(masses.max()), float(masses.min()))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_weights_never_drop_rows(self):
        y = [0, 0, 1, 0, 1, 1]
        genes = ["A", "A", "A", "B", "B", "B"]
        for mode in ["variant", "gene"]:
            weights = sample_weights(y, genes, mode)
            self.assertEqual(len(weights), len(y))
            self.assertTrue((weights > 0).all())

    def test_invalid_weight_inputs_fail(self):
        cases = [
            ([], [], "variant", "non-empty"),
            ([0, 1], ["A"], "variant", "equally sized"),
            ([0, 0.5], ["A", "A"], "variant", "binary"),
            ([0, 1], ["A", "A"], "mystery", "unknown weighting"),
            ([0, 1, 1], ["A", "A", "B"], "gene", "both classes in 'B'"),
        ]
        for y, genes, mode, message in cases:
            with self.subTest(mode=mode, message=message), self.assertRaisesRegex(ValueError, message):
                sample_weights(y, genes, mode)


class TestMetricsAndModels(unittest.TestCase):
    def test_heatmap_text_uses_accessible_contrast_for_dark_and_light_cells(self):
        self.assertEqual(_contrast_text_color((0.02, 0.05, 0.15, 1.0)), "white")
        self.assertEqual(_contrast_text_color((0.45, 0.00, 0.08, 1.0)), "white")
        self.assertEqual(_contrast_text_color((0.98, 0.94, 0.90, 1.0)), "black")
        self.assertEqual(_contrast_text_color((1.00, 1.00, 1.00, 1.0)), "black")

    def test_safe_auc_returns_nan_for_one_class(self):
        self.assertTrue(np.isnan(_safe_auc([1, 1], [0.1, 0.9])))

    def test_macro_is_exact_mean_of_gene_aucs_not_pooled_auc(self):
        y = np.array([0, 1, 0, 1])
        genes = np.array(["A", "A", "B", "B"])
        afm = np.array([0.1, 0.9, 0.2, 0.8])
        # Perfect within each gene, deliberately miscalibrated between genes.
        pred = np.array([0.8, 0.9, 0.1, 0.2])
        rows = pd.DataFrame(_auc_rows(y, genes, afm, pred, {"model": "x"}))
        macro = rows.loc[rows.scope_type == "macro", "auc"].item()
        pooled = rows.loc[rows.scope_type == "pooled_diagnostic", "auc"].item()
        self.assertEqual(macro, 1.0)
        self.assertLess(pooled, macro)

    def test_offset_with_inverse_d1_does_not_reverse_perfect_afm(self):
        y = np.array([0, 0, 1, 1] * 3)
        genes = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
        afm = np.array([0.05, 0.15, 0.85, 0.95] * 3)
        d1 = 1 - afm
        tr, va = np.arange(8), np.arange(8, 12)
        spec = BenchmarkSpec("offset", "variant", "original", 1.0)
        pred = _fit_offset_predict(afm, d1, y, genes, tr, va, spec)
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())
        self.assertEqual(_safe_auc(y[va], pred), 1.0)

    def test_fixed_grid_row_contract_and_afm_delta(self):
        specs = make_specs(
            ["raw2", "engineered6", "offset"], ["variant"], ["original"], [1],
        )
        out = evaluate_fixed_grid(synthetic_frame(), specs, 5, 42, 2026)
        self.assertEqual(len(out), (len(specs) + 1) * (4 + 2))
        self.assertFalse(out[["auc", "afm_auc", "delta_vs_afm"]].isna().any().any())
        afm = out[out.candidate_id == "afm"]
        np.testing.assert_allclose(afm["delta_vs_afm"], 0)
        self.assertEqual(set(out.scope_type), {"gene", "macro", "pooled_diagnostic"})

    def test_summary_uses_population_sd_and_seed_percentiles(self):
        base = evaluate_fixed_grid(
            synthetic_frame(), make_specs(["raw2"], ["variant"], ["original"], [1]),
            5, 42, 2026,
        )
        other = base.copy()
        other["d1_seed"] = 43
        other["auc"] += 0.02
        other["delta_vs_afm"] += 0.02
        summary = _summarize(pd.concat([base, other], ignore_index=True))
        row = summary[
            (summary.candidate_id == "raw2__variant__original__C1")
            & (summary.scope_type == "macro")
        ].iloc[0]
        self.assertEqual(int(row.n_d1_seeds), 2)
        self.assertAlmostEqual(float(row.auc_std), 0.01)
        self.assertLess(row.auc_ci_low, row.auc_mean)
        self.assertGreater(row.auc_ci_high, row.auc_mean)


class TestNestedSelection(unittest.TestCase):
    def test_fewer_than_four_genes_is_rejected(self):
        spec = make_specs(["raw2"], ["variant"], ["original"], [1])
        with self.assertRaisesRegex(ValueError, "at least four genes"):
            evaluate_nested_selection(synthetic_frame(("A", "B", "C")), spec, 5, 42, 2026)

    def test_every_inner_fit_has_disjoint_train_validation_and_one_val_gene(self):
        frame = synthetic_frame()
        specs = make_specs(["raw2"], ["variant"], ["original"], [1])
        calls = []

        def fake_fit(df, represented, y, genes, tr, va, spec, model_seed):
            calls.append((tr.copy(), va.copy(), tuple(np.unique(genes[va]))))
            return df.iloc[va]["afm_score"].to_numpy()

        with mock.patch.object(benchmark, "_fit_predict", side_effect=fake_fit):
            nested, selections = evaluate_nested_selection(frame, specs, 5, 42, 2026)
        self.assertEqual(len(calls), 4 * 3)
        for tr, va, val_genes in calls:
            self.assertFalse(set(tr) & set(va))
            self.assertEqual(len(val_genes), 1)
        self.assertEqual(set(selections.outer_gene), {"A", "B", "C", "D"})
        self.assertEqual(set(selections.candidate_id), {"afm"})
        self.assertTrue(selections.afm_fallback_used.all())
        self.assertEqual(int((nested.scope_type == "gene").sum()), 4)

    def test_exact_inner_ties_fall_back_to_afm(self):
        frame = synthetic_frame()
        specs = make_specs(
            ["raw2", "engineered6"], ["variant"], ["original"], [1],
        )

        def same_predictions(df, represented, y, genes, tr, va, spec, model_seed):
            return df.iloc[va]["afm_score"].to_numpy()

        with mock.patch.object(benchmark, "_fit_predict", side_effect=same_predictions):
            nested, selections = evaluate_nested_selection(frame, specs, 5, 42, 2026)
        expected = min(spec.candidate_id for spec in specs)
        self.assertEqual(set(selections.candidate_id), {"afm"})
        self.assertEqual(set(selections.best_fusion_candidate_id), {expected})
        self.assertTrue(selections.afm_fallback_used.all())
        np.testing.assert_allclose(selections.selection_margin, 0)
        np.testing.assert_allclose(nested["delta_vs_afm"], 0)

    def test_clear_inner_gain_selects_and_refits_fusion(self):
        frame = synthetic_frame()
        frame["afm_score"] = 0.5
        specs = make_specs(["raw2"], ["variant"], ["original"], [1])
        calls = []

        def perfect_predictions(df, represented, y, genes, tr, va, spec, model_seed):
            calls.append((tr.copy(), va.copy()))
            return y[va].astype(float)

        with mock.patch.object(benchmark, "_fit_predict", side_effect=perfect_predictions):
            nested, selections = evaluate_nested_selection(frame, specs, 5, 42, 2026)
        self.assertEqual(len(calls), 4 * (3 + 1), "clear winners must be refit on each outer fold")
        self.assertEqual(set(selections.candidate_id), {specs[0].candidate_id})
        self.assertFalse(selections.afm_fallback_used.any())
        self.assertTrue((selections.best_fusion_delta_vs_afm > selections.best_fusion_delta_se).all())
        self.assertTrue((nested.loc[nested.scope_type == "gene", "delta_vs_afm"] > 0).all())

    def test_transductive_candidates_are_never_nested_selectable(self):
        frame = synthetic_frame()
        specs = make_specs(
            ["raw2"], ["variant"], ["original", "robust", "rank"], [1],
        )
        fitted_normalizations = []

        def record_normalization(df, represented, y, genes, tr, va, spec, model_seed):
            fitted_normalizations.append(spec.normalization)
            return df.iloc[va]["afm_score"].to_numpy()

        with mock.patch.object(benchmark, "_fit_predict", side_effect=record_normalization):
            _, selections = evaluate_nested_selection(frame, specs, 5, 42, 2026)
        self.assertEqual(set(fitted_normalizations), {"original"})
        self.assertTrue(selections.afm_fallback_used.all())

    def test_nested_rejects_a_transductive_only_grid(self):
        specs = make_specs(["raw2"], ["variant"], ["robust", "rank"], [1])
        with self.assertRaisesRegex(ValueError, "eligible original-score"):
            evaluate_nested_selection(synthetic_frame(), specs, 5, 42, 2026)

    def test_new_fixed_ablation_does_not_silently_expand_nested_grid(self):
        frame = synthetic_frame()
        specs = make_specs(
            ["raw2", "compact4"], ["variant"], ["original"], [1],
        )
        fitted_representations = []

        def record_representation(df, represented, y, genes, tr, va, spec, model_seed):
            fitted_representations.append(spec.representation)
            return df.iloc[va]["afm_score"].to_numpy()

        with mock.patch.object(benchmark, "_fit_predict", side_effect=record_representation):
            evaluate_nested_selection(frame, specs, 5, 42, 2026)
        self.assertEqual(set(fitted_representations), {"raw2"})

    def test_outer_gene_labels_cannot_change_its_selected_candidate(self):
        frame = synthetic_frame()
        flipped = frame.copy()
        idx = flipped["gene"] == "A"
        flipped.loc[idx, "label"] = 1 - flipped.loc[idx, "label"]
        specs = make_specs(["raw2", "engineered6"], ["variant"], ["original"], [1])
        _, first = evaluate_nested_selection(frame, specs, 5, 42, 2026)
        _, second = evaluate_nested_selection(flipped, specs, 5, 42, 2026)
        columns = [
            "candidate_id", "best_fusion_candidate_id", "inner_macro_auc",
            "best_fusion_inner_macro_auc", "afm_inner_macro_auc",
            "afm_fallback_used",
        ]
        pd.testing.assert_series_equal(
            first[first.outer_gene == "A"].iloc[0][columns],
            second[second.outer_gene == "A"].iloc[0][columns],
            check_names=False,
        )


class TestBenchmarkOrchestration(unittest.TestCase):
    def test_local_sweep_writes_contract_and_never_touches_wandb_when_off(self):
        frame = synthetic_frame()
        specs = make_specs(["raw2"], ["variant"], ["original"], [1])
        calls = []

        def build(size, cfg, force=False):
            calls.append((size, cfg["seed"], force))
            return frame.copy()

        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(benchmark, "_plot_macro_overview"), \
             mock.patch.object(benchmark, "_plot_normalization_stress"), \
             mock.patch.object(benchmark, "_plot_per_gene_heatmap"), \
             mock.patch.object(benchmark, "_plot_nested"), \
             mock.patch.object(benchmark, "start_wandb_progress") as wandb_start:
            runs, summary, selections = run_benchmark_sweep(
                {}, [1, 5], [42, 43], Path(directory), build, specs,
                model_seed=2026, nested=False, wandb_enabled=False,
            )
            self.assertTrue((Path(directory) / "benchmark_runs.csv").is_file())
            self.assertTrue((Path(directory) / "benchmark_summary.csv").is_file())
            manifest = json.loads((Path(directory) / "benchmark_manifest.json").read_text())
        self.assertEqual(calls, [(1, 42, False), (5, 42, False), (1, 43, False), (5, 43, False)])
        self.assertFalse(selections.shape[0])
        self.assertEqual(manifest["sizes"], [1, 5])
        self.assertEqual(manifest["d1_seeds"], [42, 43])
        self.assertEqual(manifest["model_seed"], 2026)
        self.assertEqual(manifest["primary_metric"], "mean of held-out-gene ROC-AUCs (macro-LOGO)")
        self.assertEqual(manifest["nested_selection_policy"]["eligible_normalizations"], ["original"])
        self.assertEqual(
            manifest["nested_selection_policy"]["eligible_representations"],
            ["raw2", "engineered6", "offset"],
        )
        self.assertTrue(manifest["nested_selection_policy"]["afm_is_candidate"])
        self.assertIn("one paired inner-gene SE", manifest["nested_selection_policy"]["rule"])
        self.assertFalse(manifest["wandb_enabled"])
        self.assertEqual(set(runs.evaluation), {"fixed"})
        self.assertEqual(int(summary.n_d1_seeds.max()), 2)
        wandb_start.assert_not_called()

    def test_small_group_skips_nested_selection_without_writing_selection_file(self):
        frame = synthetic_frame(("A", "B"))
        specs = make_specs(["raw2"], ["variant"], ["original"], [1])
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(benchmark, "_plot_macro_overview"), \
             mock.patch.object(benchmark, "_plot_normalization_stress"), \
             mock.patch.object(benchmark, "_plot_per_gene_heatmap"), \
             mock.patch.object(benchmark, "_plot_nested"):
            _, _, selections = run_benchmark_sweep(
                {}, [5], [42], Path(directory),
                lambda size, cfg, force=False: frame.copy(), specs, nested=True,
            )
            self.assertFalse((Path(directory) / "benchmark_nested_selections.csv").exists())
        self.assertTrue(selections.empty)

    def test_wandb_flag_streams_progress_without_logging_result_metrics(self):
        frame = synthetic_frame()
        specs = make_specs(["raw2"], ["variant"], ["original"], [1])

        class FakeRun:
            def __init__(self):
                self.summary = {}
                self.logged = []
                self.finished = False

            def log(self, values):
                self.logged.append(dict(values))

            def finish(self):
                self.finished = True

        fake_run = FakeRun()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(benchmark, "_plot_macro_overview"), \
             mock.patch.object(benchmark, "_plot_normalization_stress"), \
             mock.patch.object(benchmark, "_plot_per_gene_heatmap"), \
             mock.patch.object(benchmark, "_plot_nested"), \
             mock.patch.object(
                 benchmark, "start_wandb_progress", return_value=fake_run
             ) as wandb_start:
            runs, _, _ = run_benchmark_sweep(
                {}, [5], [42], Path(directory),
                lambda size, cfg, force=False: frame.copy(), specs,
                nested=False, wandb_enabled=True, wandb_project="proj", wandb_entity="entity",
            )
        wandb_start.assert_called_once()
        config = wandb_start.call_args.args[4]
        self.assertEqual(config["candidate_count"], 1)
        self.assertEqual(config["d1_sizes"], [5])
        self.assertEqual(config["d1_seeds"], [42])
        self.assertFalse(runs.empty)
        self.assertTrue(fake_run.logged)
        self.assertTrue(all(
            set(values).issubset({
                "progress/fraction", "progress/elapsed_minutes",
                "progress/eta_minutes",
            })
            for values in fake_run.logged
        ))
        self.assertEqual(fake_run.logged[-1]["progress/fraction"], 1.0)
        self.assertEqual(fake_run.summary["progress/status"], "completed")
        self.assertTrue(fake_run.finished)

    def test_fixed_and_nested_evaluators_report_compute_progress(self):
        frame = synthetic_frame()
        specs = make_specs(["raw2"], ["variant"], ["original"], [0.1, 1])
        fixed_events = []
        nested_events = []
        evaluate_fixed_grid(
            frame, specs, 5, 42, 2026,
            progress_callback=lambda completed, total, detail: fixed_events.append(
                (completed, total, detail["candidate_id"])
            ),
        )
        evaluate_nested_selection(
            frame, specs, 5, 42, 2026,
            progress_callback=lambda completed, total, detail: nested_events.append(
                (completed, total, detail["outer_gene"])
            ),
        )
        self.assertEqual([event[:2] for event in fixed_events], [(1, 2), (2, 2)])
        self.assertEqual([event[0] for event in nested_events], [1, 2, 3, 4])
        self.assertTrue(all(event[1] == 4 for event in nested_events))

    def test_group_comparison_counts_genes_and_writes_plot(self):
        candidate = "engineered6__variant__original__C1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group_dirs = {}
            for group, n_genes, delta in [("all", 4, 0.01), ("functional", 2, 0.02), ("annotation", 3, -0.03)]:
                group_dir = root / group
                group_dir.mkdir()
                rows = [{
                    "size": 90, "evaluation": "fixed", "candidate_id": candidate,
                    "scope_type": "macro", "scope": "macro", "auc_mean": 0.8 + delta,
                    "afm_auc_mean": 0.8, "delta_mean": delta,
                    "delta_ci_low": delta - 0.001, "delta_ci_high": delta + 0.001,
                }]
                rows += [{
                    **rows[0], "scope_type": "gene", "scope": f"G{i}",
                } for i in range(n_genes)]
                pd.DataFrame(rows).to_csv(group_dir / "benchmark_summary.csv", index=False)
                group_dirs[group] = group_dir
            out = root / "combined"
            comparison = write_gt_group_comparison(group_dirs, out, candidate)
            self.assertTrue((out / "benchmark_gt_group_comparison.csv").is_file())
            self.assertTrue((out / "benchmark_gt_group_delta.png").is_file())
        got = comparison.set_index("gt_group")
        self.assertEqual(int(got.loc["functional", "n_genes"]), 2)
        self.assertEqual(got.loc["all", "interpretation"], "primary")
        self.assertIn("descriptive", got.loc["annotation", "interpretation"])


class TestBenchmarkCli(unittest.TestCase):
    def assert_cli_error(self, args, pattern):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            run.main([*args, "--out", directory])
        self.assertEqual(caught.exception.code, 2)
        self.assertRegex(stderr.getvalue(), pattern)

    def test_non_fusion_tracks_require_sweep(self):
        for track in ["benchmark", "structure"]:
            with self.subTest(track=track):
                self.assert_cli_error(["--track", track], "requires --sweep")

    def test_wandb_is_not_silently_ignored_for_single_runs(self):
        self.assert_cli_error(["--wandb"], "available for sweep tracks only")

    def test_invalid_or_empty_seed_specs_fail_as_cli_errors(self):
        for seeds in ["", "51-42", "-1", "42,x"]:
            with self.subTest(seeds=seeds):
                self.assert_cli_error(["--seeds", seeds], "seeds must")

    def test_nonpositive_single_size_fails_as_cli_error(self):
        self.assert_cli_error(["--size", "0"], "--size must be a positive integer")

    def test_invalid_mlp_settings_fail_as_cli_errors(self):
        cases = [
            ("--mlp-epochs", "0", "must be positive"),
            ("--mlp-hidden", "0", "must be positive"),
            ("--mlp-batch-size", "0", "must be positive"),
            ("--mlp-dropout", "1", "dropout must be"),
            ("--mlp-lr", "0", "learning rate must be"),
            ("--mlp-weight-decay", "-1", "weight decay"),
            ("--mlp-patience", "-1", "patience"),
        ]
        for flag, value, pattern in cases:
            with self.subTest(flag=flag):
                self.assert_cli_error([flag, value], pattern)

    def test_invalid_sweep_sizes_fail_before_dataset_loading(self):
        for sizes in ["", "0", "5,-1", "five"]:
            with self.subTest(sizes=sizes):
                self.assert_cli_error([
                    "--sweep", "--track", "benchmark", "--sizes", sizes,
                ], "--sizes must be comma-separated positive integers")

    def test_invalid_or_empty_fusion_representations_fail_before_loading(self):
        for value in ["", "raw2,mystery"]:
            with self.subTest(value=value):
                self.assert_cli_error([
                    "--sweep", "--track", "fusion", "--sizes", "5",
                    "--fusion-representations", value,
                ], "fusion-representations must be")

    def test_invalid_or_empty_fusion_normalizations_fail_before_loading(self):
        for value in ["", "original,mystery"]:
            with self.subTest(value=value):
                self.assert_cli_error([
                    "--sweep", "--track", "fusion", "--sizes", "5",
                    "--fusion-normalizations", value,
                ], "fusion-normalizations must be")

    def test_invalid_or_empty_benchmark_axes_fail_as_cli_errors(self):
        cases = [
            ("--benchmark-representations", ""),
            ("--benchmark-weightings", "mystery"),
            ("--benchmark-normalizations", ""),
            ("--benchmark-c-values", "zero"),
        ]
        for flag, value in cases:
            with self.subTest(flag=flag):
                self.assert_cli_error([
                    "--sweep", "--track", "benchmark", "--sizes", "5",
                    flag, value,
                ], "benchmark grid axes|unknown benchmark|could not convert string to float")

    def test_nested_representations_must_be_explicit_fixed_grid_subset(self):
        for value in ["", "raw2,compact4"]:
            with self.subTest(value=value):
                self.assert_cli_error([
                    "--sweep", "--track", "benchmark", "--sizes", "5",
                    "--benchmark-representations", "raw2,engineered6,offset",
                    "--benchmark-nested-representations", value,
                ], "nested-representations must be a non-empty subset")

    def test_benchmark_cli_forwards_full_grid_and_wandb_settings(self):
        summary = pd.DataFrame([{
            "size": 5, "evaluation": "fixed",
            "candidate_id": "raw2__variant__original__C1",
            "scope_type": "macro", "auc_mean": 0.8, "afm_auc_mean": 0.79,
            "delta_mean": 0.01, "delta_ci_low": 0.009, "delta_ci_high": 0.011,
            "n_d1_seeds": 1,
        }])
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(
                 benchmark, "run_benchmark_sweep",
                 return_value=(pd.DataFrame(), summary, pd.DataFrame()),
             ) as sweep:
            run.main([
                "--sweep", "--track", "benchmark", "--sizes", "5,5", "--seeds", "42,42",
                "--benchmark-representations", "raw2,offset",
                "--benchmark-weightings", "variant,gene",
                "--benchmark-normalizations", "original,rank",
                "--benchmark-c-values", "0.1,1",
                "--benchmark-model-seed", "99", "--no-benchmark-nested",
                "--wandb", "--wandb-project", "test-project", "--wandb-entity", "team",
                "--out", directory,
            ])
        kwargs = sweep.call_args.kwargs
        self.assertEqual(kwargs["sizes"], [5])
        self.assertEqual(kwargs["d1_seeds"], [42])
        self.assertEqual(kwargs["model_seed"], 99)
        self.assertFalse(kwargs["nested"])
        self.assertTrue(kwargs["wandb_enabled"])
        self.assertEqual(kwargs["wandb_project"], "test-project")
        self.assertEqual(kwargs["wandb_entity"], "team")
        self.assertEqual(len(kwargs["specs"]), 2 * 2 * 2 * 2)

    def test_benchmark_cli_rejects_multiple_gt_groups_for_one_output(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assert_cli_error([
                "--sweep", "--track", "benchmark", "--sizes", "5", "--seeds", "42",
                "--gt-group", "each", "--out", directory,
            ], "accepts one --gt-group")


if __name__ == "__main__":
    unittest.main()
