"""Regressions for structure-subsampling policy, seed allocation, and convergence.

The single most important test here is
`TestSamplingPolicy.test_default_policy_matches_pre_policy_sampler`: the default
nested/random policy must keep selecting exactly what the pre-policy code
selected, or every published D1 number silently changes.
"""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from drift import structure_draws as d1
from drift import convergence as cvg
from drift import sampling_plan as sp

_LIB = Path(__file__).resolve().parents[2] / "metric_analysis" / "pipeline" / "library"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
import structure_metrics as sm  # noqa: E402


def _legacy_sample(items, max_items, *, seed, key):
    """Verbatim copy of the sampler as it stood before SamplingPolicy existed."""
    items = sorted(str(x) for x in items)
    if len(items) <= 1:
        ranked = items
    else:
        payload = f"{seed}|{key}".encode("utf-8")
        sub = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
        rng = np.random.default_rng(sub)
        ranked = [items[i] for i in rng.permutation(len(items)).tolist()]
    if max_items is None or len(ranked) <= max_items:
        return ranked
    return ranked[:max_items]


ITEMS = [f"seed-{i}_sample-{j}" for i in range(8) for j in range(5)]
SIZES = [1, 5, 10, 20, 40]


class TestSamplingPolicy(unittest.TestCase):
    def test_default_policy_matches_pre_policy_sampler(self):
        for seed in (42, 51, 2026):
            for size in SIZES:
                key = f"mut::{seed}"
                self.assertEqual(
                    d1.select_structure_columns(ITEMS, size, seed=seed, key=key),
                    _legacy_sample(ITEMS, size, seed=seed, key=key),
                    msg=f"default policy diverged at seed={seed} size={size}",
                )

    def test_nested_sizes_are_prefixes(self):
        order = d1.rank_items_deterministically(ITEMS, seed=42, key="k")
        for size in SIZES:
            self.assertEqual(
                d1.sample_items_nested(ITEMS, size, seed=42, key="k"), order[:size]
            )

    def test_independent_sizes_are_not_prefixes(self):
        policy = d1.SamplingPolicy(mode="independent")
        draws = {
            size: d1.sample_items_nested(ITEMS, size, seed=42, key="k", policy=policy)
            for size in SIZES
        }
        for size, drawn in draws.items():
            self.assertEqual(len(drawn), size)
            self.assertEqual(len(set(drawn)), size, "draws must be without replacement")
        # At least one size must break the prefix relation, otherwise the mode
        # is silently behaving like nested.
        self.assertTrue(
            any(draws[b][: len(draws[a])] != draws[a]
                for a, b in zip(SIZES[:-1], SIZES[1:]))
        )

    def test_independent_mode_still_uniform_at_each_size(self):
        """Each size is a uniform subset; only the pairing across sizes changes."""
        policy = d1.SamplingPolicy(mode="independent")
        counts = {}
        for seed in range(200):
            picked = d1.sample_items_nested(ITEMS, 1, seed=seed, key="k", policy=policy)[0]
            counts[picked] = counts.get(picked, 0) + 1
        # With 40 items and 200 draws a uniform sampler touches many of them;
        # a degenerate sampler would concentrate on one or two.
        self.assertGreater(len(counts), 15)

    def test_top_confidence_ranks_by_score_then_name(self):
        scores = {c: float(i % 4) for i, c in enumerate(ITEMS)}
        policy = d1.SamplingPolicy(selection="top_confidence")
        picked = d1.select_structure_columns(
            ITEMS, 5, seed=42, key="k", policy=policy, scores=scores
        )
        self.assertTrue(all(scores[c] == 3.0 for c in picked))
        self.assertEqual(picked, sorted(picked), "ties must break deterministically")

    def test_top_confidence_is_pinned_to_wild_type_order(self):
        """The mutant must reuse WT's chosen structures, not re-rank its own."""
        scores = {c: float(i) for i, c in enumerate(ITEMS)}
        policy = d1.SamplingPolicy(selection="top_confidence")
        wt_order = d1.select_structure_columns(
            ITEMS, 3, seed=42, key="wt", policy=policy, scores=scores
        )
        mutant_scores = {c: -float(i) for i, c in enumerate(ITEMS)}  # opposite ranking
        picked = d1.select_structure_columns(
            ITEMS, 3, seed=42, key="mut", policy=policy,
            scores=mutant_scores, preferred_order=wt_order,
        )
        self.assertEqual(picked, wt_order)

    def test_wt_pinning_tops_up_when_mutant_lacks_columns(self):
        scores = {c: float(i) for i, c in enumerate(ITEMS)}
        policy = d1.SamplingPolicy(selection="top_confidence")
        available = ITEMS[5:]
        picked = d1.select_structure_columns(
            available, 4, seed=42, key="mut", policy=policy,
            scores={c: scores[c] for c in available},
            preferred_order=ITEMS[:3],  # none of these exist in the mutant
        )
        self.assertEqual(len(picked), 4)
        self.assertTrue(set(picked).issubset(available))

    def test_repeated_preferred_names_select_each_column_once(self):
        """A gene whose WT spans several CSVs repeats names in `preferred_order`.

        POLE has three WT files, two of them the same seed block, so the WT
        selection legitimately lists `seed-X_sample-Y` more than once. Selecting
        the column twice would double-weight it in the mutant mean and overstate
        the number of structures averaged.
        """
        policy = d1.SamplingPolicy(selection="random_pinned")
        repeated = [ITEMS[0], ITEMS[1], ITEMS[0], ITEMS[2], ITEMS[1]]
        picked = d1.select_structure_columns(
            ITEMS, 4, seed=42, key="mut", policy=policy, preferred_order=repeated
        )
        self.assertEqual(len(picked), len(set(picked)))
        self.assertEqual(picked[:3], [ITEMS[0], ITEMS[1], ITEMS[2]])
        self.assertEqual(len(picked), 4)

    def test_duplicate_preferred_names_do_not_inflate_structure_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v_plddt_values.csv"
            frame = pd.DataFrame({f"s{i}": np.full(4, 60.0 + i) for i in range(6)})
            frame.insert(0, "residue", range(4))
            frame.to_csv(path, index=False)

            chosen = []
            _, n = d1.extract_plddt_from_csv_mean(
                path, 3, seed=42, key="k",
                policy=d1.SamplingPolicy(selection="random_pinned"),
                preferred_order=["s0", "s1", "s0", "s1", "s2"],
                selection_out=chosen,
            )
            self.assertEqual(n, 3)
            self.assertEqual(chosen, ["s0", "s1", "s2"])

    def test_unscored_columns_sort_last_but_are_not_dropped(self):
        policy = d1.SamplingPolicy(selection="top_confidence")
        scores = {ITEMS[0]: 10.0, ITEMS[1]: float("nan")}
        picked = d1.select_structure_columns(
            ITEMS, len(ITEMS), seed=42, key="k", policy=policy, scores=scores
        )
        self.assertEqual(len(picked), len(ITEMS))
        self.assertEqual(picked[0], ITEMS[0])

    def test_random_pinned_keeps_the_random_order_but_couples_wt_and_mutant(self):
        """The control that separates confidence selection from WT/mutant pairing.

        `top_confidence` changes two things at once relative to the default: it
        ranks by confidence *and* it makes the mutant reuse WT's structures.
        `random_pinned` changes only the second, so the pair of arms attributes
        any gain to the right cause.
        """
        policy = d1.SamplingPolicy(selection="random_pinned")
        wt = d1.select_structure_columns(ITEMS, 5, seed=42, key="wt", policy=policy)
        # The WT side is drawn exactly as plain random does.
        self.assertEqual(wt, d1.select_structure_columns(ITEMS, 5, seed=42, key="wt"))
        # The mutant reuses WT identities despite a different key.
        pinned = d1.select_structure_columns(
            ITEMS, 5, seed=42, key="mut", policy=policy, preferred_order=wt
        )
        self.assertEqual(pinned, wt)
        # ...whereas the default draws the mutant independently.
        self.assertNotEqual(
            d1.select_structure_columns(ITEMS, 5, seed=42, key="mut"), wt
        )

    def test_random_pinned_does_not_consult_confidence(self):
        policy = d1.SamplingPolicy(selection="random_pinned")
        self.assertFalse(policy.uses_confidence)
        self.assertTrue(policy.pins_to_wild_type)
        scores = {c: float(i) for i, c in enumerate(ITEMS)}
        with_scores = d1.select_structure_columns(
            ITEMS, 4, seed=42, key="k", policy=policy, scores=scores
        )
        without = d1.select_structure_columns(ITEMS, 4, seed=42, key="k", policy=policy)
        self.assertEqual(with_scores, without)

    def test_invalid_policies_rejected(self):
        for kwargs in ({"mode": "ranked"}, {"selection": "best"}, {"mode": ""}):
            with self.assertRaises(ValueError):
                d1.SamplingPolicy(**kwargs)

    def test_tag_is_empty_for_default_so_caches_stay_valid(self):
        self.assertEqual(d1.SamplingPolicy().tag(), "")
        self.assertEqual(d1.SamplingPolicy(mode="independent").tag(), "independent-random")
        self.assertEqual(
            d1.SamplingPolicy.coerce("nested-top_confidence"),
            d1.SamplingPolicy(selection="top_confidence"),
        )
        self.assertEqual(d1.SamplingPolicy.coerce(None), d1.SamplingPolicy())

    def test_csv_reader_honours_policy_and_reports_selection(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v_plddt_values.csv"
            frame = pd.DataFrame(
                {f"s{i}": np.full(6, 50.0 + i) for i in range(10)}
            )
            frame.insert(0, "residue", range(6))
            frame.to_csv(path, index=False)

            chosen = []
            series, n = d1.extract_plddt_from_csv_mean(
                path, 3, seed=42, key="k",
                policy=d1.SamplingPolicy(selection="top_confidence"),
                selection_out=chosen,
            )
            self.assertEqual(n, 3)
            self.assertEqual(chosen, ["s9", "s8", "s7"])
            self.assertAlmostEqual(float(series.iloc[0]), 58.0)


class TestSamplingPlan(unittest.TestCase):
    def test_variance_factor_matches_finite_population_correction(self):
        self.assertAlmostEqual(sp.finite_population_variance_factor(1, 125), 1.0)
        self.assertAlmostEqual(
            sp.finite_population_variance_factor(90, 125), (1 / 90) * (35 / 124)
        )
        # Exhausting the pool leaves no sampling variance at all.
        self.assertEqual(sp.finite_population_variance_factor(100, 100), 0.0)
        self.assertEqual(sp.finite_population_variance_factor(120, 100), 0.0)

    def test_allocation_is_budget_neutral_and_monotone(self):
        sizes = [1, 5, 10, 20, 40, 60, 90]
        allocation = sp.allocate_seeds(sizes, 125, total_budget=70, floor=5)
        self.assertEqual(sum(allocation.values()), 70)
        counts = [allocation[s] for s in sizes]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertGreater(allocation[1], allocation[90])
        self.assertGreaterEqual(min(counts), 5)

    def test_floor_is_respected_and_infeasible_budget_rejected(self):
        with self.assertRaises(ValueError):
            sp.allocate_seeds([1, 5, 10], 125, total_budget=10, floor=5)

    def test_seed_plan_uses_prefixes_so_sizes_stay_paired(self):
        sizes = [1, 5, 10, 20, 40, 60, 90]
        seeds = list(range(42, 42 + 40))
        allocation = sp.allocate_seeds(sizes, 125, total_budget=70, floor=5)
        plan = sp.seed_plan(sizes, seeds, allocation)
        for size in sizes:
            self.assertEqual(plan[size], seeds[: allocation[size]])
        shared = sp.paired_seed_subset(plan)
        self.assertEqual(shared, seeds[: min(allocation.values())])
        for size in sizes:
            self.assertTrue(set(shared).issubset(plan[size]))

    def test_plan_rejects_more_seeds_than_supplied(self):
        allocation = sp.allocate_seeds([1, 5], 125, total_budget=40, floor=5)
        with self.assertRaises(ValueError):
            sp.seed_plan([1, 5], [42, 43], allocation)

    def test_allocation_table_reports_precision_gain_at_small_n(self):
        table = sp.allocation_table(
            [1, 90], 125, total_budget=40, floor=5
        ).set_index("size")
        self.assertGreater(table.loc[1, "seeds_allocated"], table.loc[90, "seeds_allocated"])
        # The point of the reallocation: N=1's mean gets more precise than the
        # uniform plan would make it.
        self.assertLess(
            table.loc[1, "rel_se_of_mean"],
            float(np.sqrt(1.0 / 10)),  # 10 uniform seeds at N=1
        )


def _synthetic_curve(means, sd=0.002, n_seeds=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for size, mean in means.items():
        # Correlated across sizes, as nested sampling makes them.
        for s in range(n_seeds):
            rows.append(
                {"size": size, "seed": 42 + s, "auc": mean + rng.normal(0, sd),
                 "model": "d1", "scope": "pooled_macro", "gt_group": "all", "cv": "logo"}
            )
    return pd.DataFrame(rows)


class TestConvergence(unittest.TestCase):
    def test_paired_deltas_are_computed_on_shared_seeds_only(self):
        curve = _synthetic_curve({1: 0.60, 5: 0.65, 10: 0.66})
        curve = curve[~((curve["size"] == 10) & (curve["seed"] > 46))]
        deltas = cvg.paired_deltas(curve[["size", "seed", "auc"]])
        by_step = deltas.set_index(["size_from", "size_to"])
        self.assertEqual(by_step.loc[(1, 5), "n_paired_seeds"], 10)
        self.assertEqual(by_step.loc[(5, 10), "n_paired_seeds"], 5)

    def test_flat_curve_converges_early(self):
        curve = _synthetic_curve({1: 0.60, 5: 0.70, 10: 0.7005, 20: 0.7007})
        result = cvg.analyze(curve, pool_size=125, epsilon=0.005)
        self.assertEqual(result["verdict"]["converged_size"], 5)

    def test_still_rising_curve_does_not_converge(self):
        curve = _synthetic_curve({1: 0.60, 5: 0.64, 10: 0.68, 20: 0.72})
        result = cvg.analyze(curve, pool_size=125, epsilon=0.005)
        self.assertIsNone(result["verdict"]["converged_size"])
        self.assertFalse(result["verdict"]["converged_within_tested_range"])

    def test_a_single_flat_step_does_not_count_as_convergence(self):
        # Flat from 5->10, then rising again: nothing has converged.
        curve = _synthetic_curve({1: 0.60, 5: 0.70, 10: 0.7005, 20: 0.75})
        result = cvg.analyze(curve, pool_size=125, epsilon=0.005)
        self.assertIsNone(result["verdict"]["converged_size"])

    def test_strict_criterion_is_harder_than_the_point_estimate(self):
        curve = _synthetic_curve({1: 0.60, 5: 0.70, 10: 0.704}, sd=0.004)
        strict = cvg.analyze(curve, epsilon=0.005, strict=True)["verdict"]
        loose = cvg.analyze(curve, epsilon=0.005, strict=False)["verdict"]
        self.assertEqual(strict["criterion"], "upper CI bound")
        self.assertEqual(loose["criterion"], "point estimate")
        if strict["converged_size"] is not None:
            self.assertIsNotNone(loose["converged_size"])

    def test_unstable_spread_blocks_convergence(self):
        curve = _synthetic_curve({1: 0.60, 5: 0.70, 10: 0.7005, 20: 0.7007}, sd=0.05)
        verdict = cvg.analyze(curve, epsilon=0.05, sd_threshold=0.001)["verdict"]
        self.assertIsNone(verdict["converged_size"])

    def test_no_descriptor_gets_an_unvalidated_finite_pool_auc_band(self):
        """A sample-mean correction is not a variance model for descriptor AUC."""
        curve = _synthetic_curve({1: 0.60, 90: 0.73})[["size", "seed", "auc"]]
        for mean_like in (True, False, None):
            summary = cvg.size_summary(curve, pool_size=125, mean_like=mean_like)
            self.assertNotIn("sd_predicted_by_pool", summary.columns)
            self.assertNotIn("fpc_variance_factor", summary.columns)
            self.assertIn("auc_sd", summary.columns)

    def test_size_summary_reports_observed_spread_only(self):
        curve = _synthetic_curve({1: 0.60, 90: 0.73})
        summary = cvg.size_summary(curve[["size", "seed", "auc"]], pool_size=125)
        self.assertEqual(set(summary.columns),
                         {"size", "auc_mean", "auc_sd", "n_seeds", "se_of_mean"})

    def test_strict_convergence_requires_an_interval_and_finite_spread(self):
        curve = _synthetic_curve({1: 0.70, 90: 0.701}, n_seeds=1)
        verdict = cvg.analyze(curve, strict=True)["verdict"]
        self.assertIsNone(verdict["converged_size"])
        self.assertFalse(verdict["converged_within_tested_range"])

    def test_ambiguous_selection_is_refused(self):
        curve = _synthetic_curve({1: 0.6, 5: 0.7})
        doubled = pd.concat([curve, curve], ignore_index=True)
        with self.assertRaises(ValueError):
            cvg.select_curve(doubled)
        with self.assertRaises(ValueError):
            cvg.select_curve(curve, model="does-not-exist")

    def test_missing_columns_reported_clearly(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runs.csv"
            pd.DataFrame({"size": [1], "auc": [0.5]}).to_csv(path, index=False)
            with self.assertRaises(ValueError) as ctx:
                cvg.load_sweep_runs(path)
            self.assertIn("seed", str(ctx.exception))


class TestSweepHonoursSeedPlan(unittest.TestCase):
    """The sweep must actually skip (size, seed) pairs the allocation drops."""

    def _run(self, seed_plan, sizes, seeds, policy_overrides=None):
        from unittest import mock

        from drift import run

        calls = []
        frame = pd.DataFrame({
            "gene": ["A", "A", "B", "B"],
            "mutation": ["M1", "M2", "M3", "M4"],
            "label": [0, 1, 0, 1],
            "afm_score": [0.1, 0.9, 0.2, 0.8],
            "d1_score": [0.2, 0.8, 0.3, 0.7],
        })

        def fake_build(size, cfg, force=False):
            calls.append((size, cfg["seed"]))
            return frame.copy()

        def fake_logo(df, cfg, feature_cols=None, fusion_representation="engineered6",
                      fusion_normalization="original", models=None,
                      progress_callback=None):
            y = df["label"].to_numpy()
            afm = df["afm_score"].to_numpy()
            per_gene = pd.DataFrame([{
                "gene": "A", "n": 2, "n_pos": 1,
                "fusion": 1.0, "logreg": 1.0, "afm": 1.0, "d1": 1.0,
            }])
            panel = [run.FUSION_MODEL_ROW_NAME[m]
                     for m in (models or run.DEFAULT_FUSION_MODELS)]
            for name in panel:
                per_gene[name] = 1.0
            return {"y": y, "afm": afm, "d1": df["d1_score"].to_numpy(),
                    "models": panel, "oof": {name: afm for name in panel},
                    "oof_fusion": afm, "oof_logreg": afm, "per_gene": per_gene}

        overrides = {"seed": 42, "epochs": 1}
        overrides.update(policy_overrides or {})
        base_cfg = run._resolve_run_config(overrides)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(run, "build_or_load_dataset", side_effect=fake_build), \
                 mock.patch.object(run, "logo_cv", side_effect=fake_logo):
                runs_df, _ = run.run_sweep(
                    base_cfg, sizes=sizes, seeds=seeds, do_stratified=False,
                    out=Path(directory), fusion_representations=["raw2"],
                    fusion_normalizations=["original"], force=True,
                    seed_plan=seed_plan,
                )
        return calls, runs_df

    def test_no_plan_runs_every_seed_at_every_size(self):
        calls, _ = self._run(None, [1, 5], [42, 43])
        self.assertEqual(sorted(calls), [(1, 42), (1, 43), (5, 42), (5, 43)])

    def test_plan_drops_large_n_seeds(self):
        plan = {1: [42, 43, 44], 5: [42]}
        calls, runs = self._run(plan, [1, 5], [42, 43, 44])
        self.assertEqual(sorted(calls), [(1, 42), (1, 43), (1, 44), (5, 42)])
        by_size = runs.groupby("size")["seed"].nunique().to_dict()
        self.assertEqual(by_size, {1: 3, 5: 1})

    def test_runs_table_records_the_sampling_policy(self):
        _, runs = self._run(
            None, [1], [42],
            policy_overrides={"d1_sampling_mode": "independent", "d1_selection": "random"},
        )
        self.assertEqual(set(runs["sampling_mode"]), {"independent"})
        self.assertEqual(set(runs["selection"]), {"random"})

    def test_default_config_stamps_the_primary_policy(self):
        _, runs = self._run(None, [1], [42])
        self.assertEqual(set(runs["sampling_mode"]), {"nested"})
        self.assertEqual(set(runs["selection"]), {"random"})

    def test_invalid_policy_in_config_fails_fast(self):
        from drift import run

        with self.assertRaises(ValueError):
            run._resolve_run_config({"d1_sampling_mode": "ranked"})


class TestReferenceStructureModes(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        base = np.array([[i * 3.8, 0.0, 0.0] for i in range(12)], dtype=float)
        self.keys = [("A", i + 1) for i in range(12)]
        self.resnames = {k: "ALA" for k in self.keys}
        self.frames = [base + rng.normal(0, 0.1, base.shape) for _ in range(4)]
        self.frames.append(base + 25.0)  # a clear outlier
        self.tensor = np.stack(self.frames)

    def test_medoid_ignores_the_outlier(self):
        self.assertNotEqual(sm._medoid_frame_index(self.tensor), 4)

    def test_medoid_of_identical_frames_is_well_defined(self):
        repeated = np.repeat(self.frames[0][None, :, :], 3, axis=0)
        self.assertEqual(sm._medoid_frame_index(repeated), 0)

    def test_mean_structure_round_trips_through_pdb(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mean.pdb"
            sm._write_mean_structure_pdb(self.tensor, self.keys, self.resnames, path)
            other = Path(td) / "one.pdb"
            sm._write_mean_structure_pdb(
                self.frames[0][None, :, :], self.keys, self.resnames, other
            )
            back, common, _, kept = sm._ca_tensor_from_aligned([path, other])
            self.assertEqual(kept, [0, 1])
            self.assertEqual(len(common), 12)
            np.testing.assert_allclose(back[0], self.tensor.mean(axis=0), atol=1e-3)

    def test_frames_without_ca_atoms_are_reported_as_skipped(self):
        """`kept` must let callers map tensor rows back to source files."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            empty = td / "empty.pdb"
            empty.write_text("END\n")
            good = []
            for i, frame in enumerate(self.frames[:3]):
                path = td / f"f{i}.pdb"
                sm._write_mean_structure_pdb(
                    frame[None, :, :], self.keys, self.resnames, path
                )
                good.append(path)
            tensor, _, _, kept = sm._ca_tensor_from_aligned([good[0], empty, good[1], good[2]])
            self.assertEqual(tensor.shape[0], 3)
            self.assertEqual(kept, [0, 2, 3])

    def test_reference_candidates_prefer_the_hint(self):
        files = [Path("X_WT_1.pdb"), Path("X_G12D.pdb"), Path("X_WT_2.pdb")]
        self.assertEqual(
            [p.name for p in sm._reference_candidates(files, "WT")],
            ["X_WT_1.pdb", "X_WT_2.pdb"],
        )
        # No match falls back to the whole ensemble rather than failing.
        self.assertEqual(len(sm._reference_candidates(files, "ZZZ")), 3)

    def test_invalid_reference_mode_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                sm._resolve_ensemble_reference(
                    [Path("a.pdb"), Path("b.pdb")],
                    reference_mode="ranked", reference_hint="WT", work_dir=Path(td),
                )

    def test_first_mode_needs_no_alignment_pass(self):
        files = [Path("X_G12D.pdb"), Path("X_WT_1.pdb")]
        with tempfile.TemporaryDirectory() as td:
            chosen = sm._resolve_ensemble_reference(
                files, reference_mode="first", reference_hint="WT", work_dir=Path(td)
            )
        self.assertEqual(chosen.name, "X_WT_1.pdb")


if __name__ == "__main__":
    unittest.main()
