#!/usr/bin/env python
"""PTEN worked example of the WT-vs-WT null: residue KS for WT-WT, WT-passenger, WT-driver.

One panel, three groups, residue KS on the y-axis, at a fixed ensemble size N
(default 50):

* WT vs WT       -- the WT pool split into two DISJOINT groups of N, repeated. One
                    dot per split. Splits share structures across repeats, so the
                    dots are not independent observations; with the 125-structure
                    canonical pool they overlap heavily, with the pooled 1,000
                    (seeds 1-25 canonical + 26-200 from PTEN_WT_seeds26-200) far less.
* WT vs passenger / WT vs driver -- for each variant, N of its structures against N
                    WT structures, averaged over repeated draws. One dot per VARIANT,
                    so more repeats sharpen each dot without adding dots.

Labels are the canonical PTEN cohort (ClinVar pathogenic = driver, benign =
passenger). The descriptor is the shared ``residue_ks`` (canonical ks_raw).

Several ``--wt-csv`` files are pooled column-wise. A column that is not a
``seed-X_sample-Y`` sample (the archive's top-level model) is dropped, after checking
that it duplicates a sample. When the pool spans several files, a batch check
compares KS between files against KS within the largest file at the same size.

Run from the repository root:
    PYTHONPATH="." python analyses/msh6_case_study/wt_backgrounds_200/pten_wt_null_panel.py      # pilot, 125 WT
    PYTHONPATH="." python analyses/msh6_case_study/wt_backgrounds_200/pten_wt_null_panel.py \
        --wt-csv PLDDTs/PTEN/PTEN_WT_plddt_values.csv \
                 analyses/msh6_case_study/wt_backgrounds_200/plddt/PTEN_WT_seeds26-200_plddt_values.csv \
        --n 100 --out analyses/msh6_case_study/wt_backgrounds_200/results_wt1000_n100
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from metric_analysis.pipeline.library.descriptor_metrics import residue_ks

MANIFEST = (REPO / "analyses" / "msh6_case_study" / "ks_meanfree" / "results" / "canonical_v3_subunit"
            / "canonical_cohort_manifest.csv")
BASE_SEED = 20260923
INK, GREY = "#12161d", "#6f7684"
GROUPS = [  # (key, axis label, colour)
    ("wt_wt", "WT vs WT", "#8d8d8d"),
    ("wt_passenger", "WT vs passenger", "#2a78d6"),
    ("wt_driver", "WT vs driver", "#d1495b"),
]


def load(path, rows=None):
    x = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
    return x if rows is None else x[rows]


def row_slice(cohort):
    """The gene's residue block in a complex's pLDDT table (MSH2/MSH6 in MutS_alpha)."""
    if "plddt_row_range_start" not in cohort or cohort.plddt_row_range_start.isna().all():
        return None
    start, stop = cohort.plddt_row_range_start.iloc[0], cohort.plddt_row_range_stop.iloc[0]
    if (cohort.plddt_row_range_start != start).any() or (cohort.plddt_row_range_stop != stop).any():
        raise SystemExit("cohort rows disagree on the residue block")
    return slice(int(start), int(stop))


SAMPLE = r"seed-\d+_sample-\d+"


def load_wt_pool(paths):
    """WT matrices pooled column-wise; non-sample columns must duplicate a sample."""
    frames = []
    for path in paths:
        f = pd.read_csv(path, index_col=0)
        extra = [c for c in f.columns if not pd.Series([c]).str.fullmatch(SAMPLE).iloc[0]]
        for c in extra:
            if not any(np.array_equal(f[c].values, f[s].values) for s in f.columns if s not in extra):
                raise SystemExit(f"{path}: column {c} is not a sample and duplicates none")
        frames.append(f.drop(columns=extra))
    for f in frames[1:]:
        if not f.index.equals(frames[0].index):
            raise SystemExit("WT files have different residue indices")
    names = [c for f in frames for c in f.columns]
    if len(names) != len(set(names)):
        raise SystemExit("WT files share seed/sample columns")
    return [f.to_numpy(dtype=float) for f in frames], names


def seed_of(names):
    return np.array([int(re.match(r"seed-(\d+)_", c).group(1)) for c in names])


SEEDS_PER_RUN = 25  # one AF3 run: 25 seeds x 5 diffusion samples = the 125-prediction pool


def batch_check(parts, n, reps, rng):
    """KS between the first file and the largest vs KS within the largest, at size n."""
    big = max(range(len(parts)), key=lambda i: parts[i].shape[1])
    other = next(i for i in range(len(parts)) if i != big)
    k = min(n, parts[other].shape[1], parts[big].shape[1] // 2)
    within, between = [], []
    for _ in range(reps):
        o = rng.permutation(parts[big].shape[1])
        within.append(residue_ks(parts[big][:, o[:k]], parts[big][:, o[k:2 * k]]))
        between.append(residue_ks(parts[other][:, rng.choice(parts[other].shape[1], k, replace=False)],
                                  parts[big][:, o[:k]]))
    return {"size": int(k), "within_median": float(np.median(within)),
            "between_median": float(np.median(between)),
            "between_minus_within": float(np.median(between) - np.median(within))}


def run_columns(seeds, rng, exclude=()):
    """Columns of SEEDS_PER_RUN randomly chosen seeds: a stand-in for one AF3 run."""
    pick = rng.choice(np.setdiff1d(np.unique(seeds), exclude), SEEDS_PER_RUN, replace=False)
    return np.flatnonzero(np.isin(seeds, pick)), pick


def wt_vs_wt(wt, n, reps, rng, seeds=None):
    """WT-vs-WT residue KS. With `seeds`, each side is its own set of 25 seeds (the
    seed-matched split); without, predictions are split at random. The diffusion
    samples of a seed are correlated, so the random split mixes seeds across sides
    and gives a background that is too narrow."""
    if 2 * n > wt.shape[1]:
        raise SystemExit(f"WT pool has {wt.shape[1]} structures; a disjoint {n} vs {n} split needs {2 * n}")
    out = []
    for _ in range(reps):
        if seeds is None:
            order = rng.permutation(wt.shape[1])
            a, b = order[:n], order[n:2 * n]
        else:
            ca, picked = run_columns(seeds, rng)
            cb, _ = run_columns(seeds, rng, exclude=picked)
            a, b = rng.choice(ca, n, replace=False), rng.choice(cb, n, replace=False)
        out.append(residue_ks(wt[:, a], wt[:, b]))
    return np.array(out)


def wt_vs_variant(wt, mutant, n, reps, rng, seeds=None):
    """Variant score: mean residue KS over `reps` draws. With `seeds`, the WT side of
    each draw comes from one 25-seed run, as the null's sides do."""
    k = min(n, mutant.shape[1])
    vals = []
    for _ in range(reps):
        cols = np.arange(wt.shape[1]) if seeds is None else run_columns(seeds, rng)[0]
        vals.append(residue_ks(wt[:, rng.choice(cols, n, replace=False)],
                               mutant[:, rng.choice(mutant.shape[1], k, replace=False)]))
    return float(np.mean(vals))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", default="PTEN",
                    help="PTEN reads the v3 manifest as before; any other gene the canonical "
                         "cohort of the active DRIFT_RUN_TAG, with its residue block applied")
    ap.add_argument("--wt-csv", type=Path, nargs="+", default=None,
                    help="WT pLDDT matrices to pool (default: the gene's canonical WT file)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--wt-reps", type=int, default=200)
    ap.add_argument("--variant-reps", type=int, default=20)
    ap.add_argument("--panel", action="store_true",
                    help="also write the Figure 1f version: paper size, no footnote, short title")
    ap.add_argument("--split", choices=("random", "seed"), default="random",
                    help="seed: each WT side is its own 25-seed run (needs the expanded pool)")
    ap.add_argument("--out", type=Path, default=HERE / "results_pilot_wt125")
    args = ap.parse_args(argv)

    if args.gene == "PTEN":
        cohort = pd.read_csv(MANIFEST).query("protein == 'PTEN'")
    else:
        from metric_analysis.pipeline.library.canonical_cohort import load_canonical_cohort
        cohort = load_canonical_cohort(args.gene)
    block = row_slice(cohort)
    wt_paths = args.wt_csv or [Path(cohort.wt_source_file.iloc[0])]
    parts, names = load_wt_pool(wt_paths)
    if block is not None:
        parts = [p[block] for p in parts]
    wt = np.concatenate(parts, axis=1)
    seeds = seed_of(names) if args.split == "seed" else None
    rng = np.random.default_rng(BASE_SEED)
    batch = batch_check(parts, args.n, args.wt_reps, rng) if len(parts) > 1 else None

    null = wt_vs_wt(wt, args.n, args.wt_reps, rng, seeds)
    rows = [dict(group="wt_wt", item=f"split_{i}", ks=v) for i, v in enumerate(null)]
    for rec in cohort.itertuples():
        mutant = load(rec.source_file, block)
        if mutant.shape[0] != wt.shape[0]:
            raise SystemExit(f"{rec.mutation_code}: residue mismatch {mutant.shape[0]} vs {wt.shape[0]}")
        group = "wt_driver" if rec.label == 1 else "wt_passenger"
        rows.append(dict(group=group, item=rec.mutation_code,
                         ks=wt_vs_variant(wt, mutant, args.n, args.variant_reps, rng, seeds)))
    values = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    values.to_csv(args.out / "values.csv", index=False)

    p95 = float(np.percentile(null, 95))
    variants = values[values.group != "wt_wt"]
    summary = {
        "gene": args.gene, "n_per_side": args.n, "wt_pool_structures": int(wt.shape[1]),
        "wt_csv": [str(p) for p in wt_paths], "wt_batch_check": batch,
        "wt_splits": args.wt_reps, "variant_draws": args.variant_reps, "seed": BASE_SEED,
        "split": args.split,
        "wt_wt_median": float(np.median(null)), "wt_wt_p95": p95,
        "groups": {g: {"n": int((values.group == g).sum()),
                       "median": float(values.loc[values.group == g, "ks"].median()),
                       "fraction_above_wt_wt_p95": float((values.loc[values.group == g, "ks"] > p95).mean())}
                   for g, _, _ in GROUPS},
        "driver_vs_passenger_auc": float(roc_auc_score(variants.group == "wt_driver", variants.ks)),
        "driver_vs_passenger_mwu_p": float(mannwhitneyu(
            variants.loc[variants.group == "wt_driver", "ks"],
            variants.loc[variants.group == "wt_passenger", "ks"]).pvalue),
        "note": "WT-WT dots are overlapping splits of one pool, not independent observations.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw(values, summary, args.out, args.gene)
    if args.panel:
        draw_panel(values, summary, args.out, args.gene)
    print(json.dumps(summary, indent=2))


def draw(values, summary, out, gene="PTEN"):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    rng = np.random.default_rng(BASE_SEED + 1)
    data = [values.loc[values.group == g, "ks"].to_numpy() for g, _, _ in GROUPS]
    parts = ax.violinplot(data, positions=range(3), widths=.8, showextrema=False)
    for body, (_, _, colour) in zip(parts["bodies"], GROUPS):
        body.set_facecolor(colour)
        body.set_edgecolor("none")
        body.set_alpha(.18)
    ax.boxplot(data, positions=range(3), widths=.22, showfliers=False, patch_artist=True,
               medianprops=dict(color=INK, lw=1.6),
               boxprops=dict(facecolor="white", edgecolor=INK, lw=.9),
               whiskerprops=dict(color=INK, lw=.9), capprops=dict(color=INK, lw=.9))
    for x, (vals, (_, _, colour)) in enumerate(zip(data, GROUPS)):
        jitter = rng.uniform(-.28, .28, len(vals))
        size = 22 if len(vals) < 30 else 9
        ax.scatter(x + jitter, vals, s=size, color=colour, alpha=.75 if len(vals) < 30 else .45,
                   edgecolor="white", linewidth=.4, zorder=3)
    ax.axhline(summary["wt_wt_p95"], ls=(0, (3, 2)), lw=1, color=GREY, zorder=1)
    ax.text(-.45, summary["wt_wt_p95"] + .008, "WT–WT 95th percentile", fontsize=7,
            color=GREY, va="bottom", ha="left")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{label}\n(n = {summary['groups'][g]['n']})" for g, label, _ in GROUPS],
                       fontsize=8.5)
    ax.set_ylabel(f"KS-score  (N = {summary['n_per_side']} per side)")
    ax.set_title(f"{gene}: mutation signal against the WT–WT null", fontsize=10, color=INK)
    ax.grid(axis="y", alpha=.25, lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.01, .005,
             f"WT pool: {summary['wt_pool_structures']} structures; WT–WT dots are {summary['wt_splits']} "
             f"overlapping disjoint splits ({'one 25-seed run per side' if summary['split'] == 'seed' else 'random'}).\nVariant dots: one per variant, mean of "
             f"{summary['variant_draws']} draws. Driver vs passenger AUC = {summary['driver_vs_passenger_auc']:.2f}.",
             fontsize=6.3, color=GREY, va="bottom")
    fig.tight_layout(rect=(0, .07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{gene}_wt_null_violin.{ext}", dpi=220)


def draw_panel(values, summary, out, gene="PTEN"):
    """Figure 1f: the same data at the size and style of the panel it replaces
    (figures/figure1/ks_combo Figure1f_violin_PTEN_resKS, ~3.7 x 3.3 in).
    Everything the footnote said moves to the legend."""
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(3.72, 3.32))
    rng = np.random.default_rng(BASE_SEED + 1)
    data = [values.loc[values.group == g, "ks"].to_numpy() for g, _, _ in GROUPS]
    parts = ax.violinplot(data, positions=range(3), widths=.8, showextrema=False)
    for body, (_, _, colour) in zip(parts["bodies"], GROUPS):
        body.set_facecolor(colour); body.set_edgecolor(colour); body.set_alpha(.2)
    for x, (vals, (_, _, colour)) in enumerate(zip(data, GROUPS)):
        ax.boxplot([vals], positions=[x], widths=.22, showfliers=False, patch_artist=True,
                   medianprops=dict(color="black", lw=1.4),
                   boxprops=dict(facecolor="white", edgecolor=colour, lw=1.2),
                   whiskerprops=dict(color="#4a4a4a", lw=1), capprops=dict(color="#4a4a4a", lw=1))
        ax.scatter(x + rng.uniform(-.2, .2, len(vals)), vals, s=14 if len(vals) < 30 else 7,
                   color=colour, alpha=.55, edgecolor="none", zorder=3)
    p95 = summary["wt_wt_p95"]
    ax.axhline(p95, ls=(0, (3, 2)), lw=.9, color=GREY, zorder=1)
    if gene == "PTEN":
        ax.text(2.45, p95 - .012, "WT–WT 95th pct.", fontsize=7, color=GREY, va="top", ha="right")
    else:   # drivers straddle the line, so the label sits in the gap before them
        ax.text(1.5, p95 + .003, "WT–WT\n95th pct.", fontsize=7, color=GREY, va="bottom",
                ha="center", linespacing=1.1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{lab}\n(n={summary['groups'][g]['n']})"
                        for (g, _, _), lab in zip(GROUPS, ("WT vs WT", "Passenger", "Driver"))])
    ax.set_ylabel(f"KS-score (N = {summary['n_per_side']})")
    ax.set_title(f"{gene}    driver vs passenger AUC {summary['driver_vs_passenger_auc']:.2f}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{gene}_wt_null_panel.{ext}", dpi=300)


if __name__ == "__main__":
    main()
