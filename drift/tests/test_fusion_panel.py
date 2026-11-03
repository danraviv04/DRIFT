"""Regressions for partner identity, model eligibility and matched CV."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from drift import run
from drift.benchmark import BenchmarkSpec, _fit_offset_predict


class TestFusionPanel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(14)
        self.frame = pd.DataFrame({
            "gene": np.repeat(["A", "B", "C"], 20),
            "mutation": [str(i) for i in range(60)],
            "label": np.tile([0, 1], 30),
            "afm_score": rng.uniform(.1, .9, 60),
            "d1_score": rng.uniform(0, 1, 60),
            "plddt_ks_residue": rng.uniform(0, 1, 60),
        })
        self.cfg = run._resolve_run_config({"seed": 42, "epochs": 1})

    def test_panel_matches_balanced_mlp_and_scaled_spline_reference(self):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import SplineTransformer, StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.utils.class_weight import compute_sample_weight
        rng = np.random.default_rng(71)
        x = rng.normal(size=(160, 3)).astype(np.float32)
        y = np.array([0] * 120 + [1] * 40)
        v = rng.normal(size=(20, 3)).astype(np.float32)
        mlp = MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, solver="adam",
                            activation="relu", max_iter=300, early_stopping=True,
                            validation_fraction=.15, n_iter_no_change=20, random_state=42)
        mlp.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
        spline = make_pipeline(
            SplineTransformer(n_knots=5, degree=3, knots="quantile",
                              extrapolation="linear", include_bias=False),
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42,
                                                 class_weight="balanced"))
        spline.fit(x, y)
        for name, reference in [("mlp_sklearn", mlp), ("spline_logreg", spline)]:
            actual, details = run._fit_sklearn_panel_model(name, x, y, v, 42,
                                                          return_details=True)
            np.testing.assert_array_equal(actual, reference.predict_proba(v)[:, 1].astype(np.float32))
            self.assertEqual(details["balanced_sample_weight"], name == "mlp_sklearn")
            self.assertTrue(details["n_iter"])

    def test_svm_threshold_and_saved_fit_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(run, "build_or_load_dataset", return_value=self.frame), \
             mock.patch.object(run, "_plot_sweep_panel"):
            rows, _ = run.run_sweep(self.cfg, [1], [42], True, Path(tmp),
                                    fusion_models=["svm_linear", "spline_logreg"])
            saved = pd.read_csv(Path(tmp) / "sweep_runs.csv")
            for row in saved[(saved.scope == "pooled") &
                             saved.fit_diagnostics_json.notna()].itertuples():
                fits = json.loads(row.fit_diagnostics_json)
                if fits:
                    self.assertIn("warnings", fits[0])
                    self.assertIn("parameters", fits[0])
                    self.assertEqual(len(fits), 5 if row.cv == "stratified" else 3)
        result = run.logo_cv(self.frame, self.cfg, models=["svm_linear"])
        pg = result["per_gene"]
        self.assertIn("svm_linear_f1_decision_0", pg)
        self.assertNotIn("svm_linear_f1_05", pg)
        payload = run._logo_monitor_metrics(1, pg, result["models"])
        self.assertIn("evaluation/svm_linear_macro_f1_at_decision_0", payload)
        self.assertNotIn("evaluation/svm_linear_macro_f1_at_0_5", payload)

    def test_fit_warnings_are_captured(self):
        import warnings
        from sklearn.svm import LinearSVC
        original = LinearSVC.fit
        def fit(model, *args, **kwargs):
            warnings.warn("test convergence diagnostic", UserWarning)
            return original(model, *args, **kwargs)
        x = self.frame[["afm_score", "d1_score"]].to_numpy()
        with mock.patch.object(LinearSVC, "fit", fit):
            _, details = run._fit_sklearn_panel_model(
                "svm_linear", x[:40], self.frame.label.to_numpy()[:40], x[40:], 42,
                return_details=True)
        self.assertTrue(any(w["message"] == "test convergence diagnostic"
                            for w in details["warnings"]))

    def test_offset_details_preserve_predictions(self):
        df = self.frame
        args = (df.afm_score.to_numpy(), df.d1_score.to_numpy(), df.label.to_numpy(),
                df.gene.to_numpy(), np.arange(40), np.arange(40, 60),
                BenchmarkSpec("offset", "variant", "original", 1.0))
        old = _fit_offset_predict(*args)
        new, details = _fit_offset_predict(*args, return_details=True)
        np.testing.assert_array_equal(old, new)
        self.assertGreaterEqual(details["beta"], 0)
        self.assertGreater(details["descriptor_scale"], 0)

    def test_offset_fits_extras_and_gbm_rejects_engineered(self):
        # The offset used to refuse extras outright. It now fits one
        # non-negative slope per descriptor, so an extra column is fitted rather
        # than rejected -- and the fold's coefficient record has to grow with it,
        # which is what distinguishes a real k-df fit from a silently ignored
        # column.
        result = run.logo_cv(self.frame, self.cfg, models=["offset"],
                             feature_cols=["plddt_ks_residue"])
        fits = result["offset_coefficients"]
        self.assertTrue(fits)
        for fit in fits:
            self.assertEqual(fit["n_descriptors"], 2)
            self.assertEqual(len(fit["betas"]), 2)
            self.assertTrue(all(b >= 0 for b in fit["betas"]))
        with self.assertRaisesRegex(ValueError, "raw2"):
            run.logo_cv(self.frame, self.cfg, models=["gbm"],
                        fusion_representation="engineered6")

    def test_gbm_constraints_follow_descriptor_identity(self):
        with mock.patch.object(run, "_fit_monotone_gbm", side_effect=
                               lambda x, y, v, seed, constraints: np.full(len(v), .5)) as fit:
            run.logo_cv(self.frame, {**self.cfg, "fusion_partner": "ks_residue"},
                        models=["gbm"])
        self.assertTrue(all(call.args[-1] == [1, 0] for call in fit.call_args_list))

    def test_matched_sweep_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(run, "build_or_load_dataset", return_value=self.frame) as build, \
             mock.patch.object(run, "_plot_sweep_panel"):
            kwargs = dict(base_cfg=self.cfg, sizes=[1], seeds=[42], do_stratified=True,
                          out=Path(tmp), fusion_partners=["d1", "ks_residue"],
                          feature_variants=["base", "ks_residue"],
                          fusion_models=["offset", "logreg"])
            rows, summary = run.run_sweep(**kwargs)
            self.assertEqual(set(rows.cv), {"logo", "stratified"})
            keys = ["model", "fusion_partner", "variant", "scope"]
            self.assertEqual(set(map(tuple, rows[rows.cv == "logo"][keys].to_numpy())),
                             set(map(tuple, rows[rows.cv == "stratified"][keys].to_numpy())))
            # Offset now covers every arm, so the panel is rectangular: each
            # model reaches the same (partner, variant) combinations.
            arms = lambda m: set(map(tuple, rows.loc[rows.model == m,
                                                     ["fusion_partner", "variant"]].to_numpy()))
            self.assertEqual(arms("offset"), arms("logreg"))
            base = summary[(summary.model == "logreg") & (summary.variant == "base")]
            self.assertEqual(set(base.fusion_partner), {"d1", "ks_residue"})
            self.assertTrue((summary.n_seeds == 1).all())
            self.assertTrue(rows.loc[rows.scope == "coefficients", "offset_coefficients_json"].notna().all())
            resumed, _ = run.run_sweep(**kwargs, resume=True)
            self.assertEqual(len(resumed), len(rows))
            self.assertEqual(build.call_count, 1)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run.run_sweep(**{**kwargs, "fusion_models": ["logreg"]}, resume=True)
            with self.assertRaisesRegex(ValueError, "contains a checkpoint"):
                run.run_sweep(**kwargs)
            self.assertEqual(build.call_count, 1)

    def test_mirrored_engineered_arms_are_retained(self):
        frame = self.frame.assign(plddt_ks_centred=self.frame.plddt_ks_residue ** 2)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(run, "build_or_load_dataset", return_value=frame), \
             mock.patch.object(run, "_plot_sweep_panel"):
            rows, _ = run.run_sweep(self.cfg, [1], [42], False, Path(tmp),
                                   fusion_partners=["ks_residue", "ks_centred"],
                                   feature_variants=["ks_residue", "ks_centred"],
                                   fusion_representations=["engineered6"], fusion_models=["logreg"])
        fitted = rows[rows.model == "logreg"]
        self.assertEqual(set(zip(fitted.fusion_partner, fitted.variant)),
                         {("ks_residue", "ks_centred"), ("ks_centred", "ks_residue")})

    def test_stratification_refuses_silent_label_only_fallback(self):
        with self.assertRaisesRegex(ValueError, ">=5"):
            run.logo_cv(self.frame.iloc[:6], self.cfg, models=["logreg"], cv_mode="stratified")

    def test_d1_baseline_is_independent_of_partner(self):
        result = run.logo_cv(self.frame, {**self.cfg, "fusion_partner": "ks_residue"},
                             models=["logreg"])
        sorted_frame = self.frame.sort_values(["gene", "mutation"])
        np.testing.assert_array_equal(result["d1"], sorted_frame.d1_score.to_numpy())


if __name__ == "__main__":
    unittest.main()
