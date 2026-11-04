#!/usr/bin/env python
"""Figure 2, rebuilt on the annotation cohort after the MutS_alpha split.

V5 (2026-10-14): rebuilt on the v5_kras4b cohort. The annotation cohort is now
SIX datasets (PIK3CA joined; MSH6 is the G:T-mismatch rerun with its own WT), and
every input is read from the run-tagged v5 outputs: the current-fingerprint
fusion caches (a, b), the non-learned size sweep (e) and the per-gene tables
(f, g). The functional row (c, d) reads the continuous cohort
(analyses/continuous/continuous_functional): TP53 1,174 and the KRAS4B blind sample
(935, signed LFC). There are no binary functional labels any more, so the old
functional AUC audit (results_size_sweep_functional_v1) is gone.

SCOPE (frozen 2026-09-11, with the supervisor)
----------------------------------------------
Annotation cohort only: POLE, MSH2, MSH6, VHL, PTEN -- ClinVar plus POLE's
curated table. The functional datasets (TP53, KRAS) and the positional control
(FOXA1) are NOT here; they belong to a later section. This is
`drift.run.GT_GROUPS["annotation"]`, read from the module so the figure
cannot drift from the cohort the sweeps used.

WHICH KS
--------
residueKS (`plddt_ks_residue`): the mean over residues of the per-residue
two-sample KS between the WT and mutant ensembles at that residue -- paired
across residues, and sensitive to within-residue distribution.

This is what Methods DEFINES (01_main_manuscript.tex L664-669):

    K_r = sup_t |F_{v,r}(t) - F_{WT,r}(t)|,    KS(v) = R^-1 sum_r K_r

The published Figure 2 plotted profileKS (`plddt_ks`) instead -- each residue
collapsed to its mean first, then one KS across the residue axis. So this figure
is not a departure from the manuscript; it is the first figure that agrees with
the manuscript's own equation, and the published one was the anomaly.

The two are not interchangeable: they rank variants differently. Centred KS is
defined per residue too, so only resKS makes it "the mean-free counterpart of KS".

Figure 1's KS panels were moved to resKS on 2026-09-11 for the same reason.

PANELS (relettered 2026-09-15)
------------------------------
  1 DESCRIPTORS
  a  per-gene ROC-AUC against ensemble size, one sub-panel per descriptor.
  2 SUMMARY
  b  macro ROC-AUC against ensemble size over the five datasets.
  3 FUNCTIONAL
  c  per-variant score against the assay at N=90, TP53 and KRAS x resKS,
     with the least-squares line and its Spearman rho and R^2.
  4 FUSION   (non-learned, from 2026-09-15)
  e  fusion macro against ensemble size, against the flat AM reference.
  f  per-gene AUC against ensemble size on the functional cohort.
  g  added to AlphaMissense, per dataset, at N=90.

WHICH FUSION
------------
`gz_mean` x `AM+resKS`, from analyses/annotation/simple_fusion_benchmark's annotation
size sweep: the equal-weight mean of AlphaMissense and residue KS after one
global z-scaler fitted on each fold's TRAINING rows. Nothing is fitted to the
labels, so there is no model to overfit and no hyper-parameter that could have
been picked on the test folds.

It replaces the logistic arm the panels used to draw, and costs nothing to do
so: macro 0.9421 at N=90 against the logistic 0.942, over the same folds and
the same ten subsampling seeds.

Panel f moved with them, which CHANGED ITS METRIC. It was the continuous
ridge's Spearman rho against the assay; the non-learned sweep scores ROC-AUC,
average precision and precision@R, not rank correlation, so matching the
estimator meant taking ROC-AUC. The two agree on the finding either way: on
KRAS the fusion is worse than AlphaMissense alone at every N (0.595 against
0.683 at N=90; under the ridge, rho 0.13 against 0.31), and on TP53 it is flat
and level with it (0.877 against 0.874).

The scaler is fold-fitted, which makes gz the one non-learned variant that
depends on the protocol, so the benchmark runs it under LOGO only.

Release copy: the two audit tables that needed extra runs (the continuous ridge,
audit_ridge_continuous.csv, and the learned logistic arm's two-protocol
comparison, audit_logreg_protocols.csv) are no longer written; no panel used them.

DROPPED 2026-09-15
------------------
The old gain-from-N bars and N=90 descriptor comparison (c, d) restated panel
b's endpoints; the old R^2-against-N panel (f) is now read off panel c, where
it is a property of a line the reader can see, and its KRAS arm sat near -1.35
and flattened every other curve sharing the axis; the old both-protocols panel
(j) is still computed and written to panel_i_protocols.csv.

Usage:  python make_figure2.py [--gz DIR] [--tables DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from metric_analysis.pipeline.paths import tagged
from drift.protein_config import protein_config_fingerprint

CACHE = REPO / "drift" / "cache"
# The CURRENT protein-config fingerprint (v5: e1596fb1503b). Pinning it stops a
# cache of an earlier cohort being picked up silently.
FINGERPRINT = f"proteins{protein_config_fingerprint()}"

INK, GREY, AFM_C = "#12161d", "#6f7684", "#8d8d8d"
KS_C = "#6a4c93"                           # matches the published Figure 2
GENE_C = {"POLE": "#0f7b6c", "MSH2": "#2a78d6", "MSH6": "#c2255c",
          "VHL": "#b58900", "PTEN": "#6a4c93", "PIK3CA": "#e8590c"}
# The colour system, so a hue means one thing across the figure:
#   GENE_C   the five annotation datasets     -- panels a and g
#   FUNC_C   the four continuous datasets     -- panels c-f
#   KS_C     residue KS pooled over datasets  -- panel b
#   FUSION_C the fused arm, pooled            -- panel e
#   TREND_C  a fitted line, never data        -- the lines in c and d
# A dataset therefore carries the same colour wherever it appears, and the one
# mark in the figure that is a FIT rather than a measurement is the only thing
# drawn in TREND_C.
# PTEN and MSH2 keep their panel-a colours: same dataset, same hue, even though
# here they are scored against their assay rather than their ClinVar label.
FUNC_C = {"TP53": "#d1495b", "KRAS": "#e09f3e",
          "PTEN": GENE_C["PTEN"], "MSH2": GENE_C["MSH2"]}
# Panel titles: KRAS is the KRAS4B blind sample since v5.
FUNC_TITLE = {"TP53": "TP53", "KRAS": "KRAS4B", "PTEN": "PTEN", "MSH2": "MSH2"}
FUNC_YLABEL = {"TP53": "log$_{10}$ selection score",
               "KRAS": "HA1E log$_2$ fold change",
               "PTEN": "impairment ($-$fitness)",
               "MSH2": "LOF score"}
FUSION_C, TREND_C = "#1b9aaa", "#1d3557"
GZ_METHOD, GZ_ARM = "gz_mean", "AM+resKS"
# What the panels PRINT. Kept apart from GZ_ARM, which has to stay the literal
# value in the benchmark's `arm` column: a reader cannot tell a non-learned
# z-mean from a fitted logistic arm if both are labelled "AM+resKS".
GZ_LABEL = "AM + KS-score (equal weight)"
ARM_C = {GZ_LABEL: FUSION_C}
# The three frozen fusion arms, as (fusion_partner, variant) in the sweep table.
# Residue KS only.
DESCRIPTORS = [("KS-score", "plddt_ks_residue", KS_C)]
# Residue KS is defined for N>=1 (descriptor_metrics). At N=1 it is the fraction
# of residues whose WT and mutant pLDDTs differ. The N=1 caches carry it directly again;
# inputs/ks_n1* is the independent snapshot they are cross-checked against.
ANCHOR = ("KS at N=1", "plddt_ks_residue", KS_C)
DEFAULT_GZ = (REPO / "analyses" / "annotation" / "simple_fusion_benchmark"
              / tagged("results_size_sweep_annotation"))
# The fusion panels read the NON-LEARNED benchmark, not the logistic sweep, from
# 2026-09-15. gz_mean is the equal-weight mean of AlphaMissense and residue KS
# after a single global z-scaler fitted on each fold's TRAINING rows -- the same
# scaler the learned models use, and the only non-learned variant that is
# protocol-dependent, which is why the benchmark runs it under LOGO alone.
#
# Nothing is fitted to the labels, so there is no model to overfit and no
# hyper-parameter that could have been chosen on the test folds; the macro is
# 0.9421 at N=90 against the logistic arm's 0.942, so the figure gives up
# nothing by making the claim with the simpler estimator.
DEFAULT_TABLES = (REPO / "analyses" / "annotation" / "simple_fusion_benchmark"
                  / tagged("results_tables_annotation_N90"))
# The columns the tables print. Threshold-free ranking first, then one
# operating point, because ROC-AUC alone flatters POLE: 38 positives against
# 1036 negatives (1037 before P356S left the cohort, 2026-10-20), where AP is the metric that moves.
#
# pAUC(.1) and MCC were dropped on 2026-09-17. They are still computed by the
# benchmark and still in results_tables_annotation_N90, so nothing is lost --
# this is a display cut. pAUC(.1) is the same family as ROC-AUC read above 10%
# FPR, so it largely restated its neighbour; MCC is threshold-dependent on a
# cohort that is majority-driver in four of five datasets (PTEN 96.5%, VHL
# 84.7%, MSH6 68.4%, MSH2 65.1%), which makes it the least stable number in the
# row. Note prec@R remains SATURATED for that same reason: the top-R set holds
# nearly every positive whatever the ranking does, so it reports an identical
# value for AM and the fusion on four of five datasets. It is kept because that
# saturation is itself informative -- the gain lives in the imbalanced gene.
TABLE_COLS = ["n", "ROC-AUC", "AP", "prec@R", "F1", "Acc", "bal.acc"]
# Printed header -> source column. The abbreviated headers were what held the
# figure at 13.4 in when e shared their row; on a row of their own each table
# has half the page and can spell them out.
TABLE_SRC = {c: c for c in TABLE_COLS}
# Acc and bal.acc are carried TOGETHER on purpose. Plain accuracy is dominated
# by prevalence here -- the majority-class baseline runs from 0.651 (MSH2) to
# 0.965 (POLE, PTEN) -- so POLE's 0.461 sits far BELOW a trivial "all
# passenger" prediction and reads as a broken baseline when it is the
# threshold policy calling positives on a 3.5%-prevalence cohort. Balanced
# accuracy has null 0.5 at any prevalence, so printing the pair lets the reader
# see that effect in the row instead of taking it on trust from the caption.
TABLE_SRC["Acc"] = "accuracy"
TABLE_MODELS = [("AlphaMissense", "AlphaMissense", AFM_C),
                ("gz_mean", "AM + KS-score (equal weight)", FUSION_C)]
CONT_SRC = REPO / "analyses" / "continuous" / "continuous_functional"
# Score column -> label, for the assay scatter. AlphaMissense is deliberately
# NOT a column: section 3 asks what the two descriptors see on their own, and
# every comparison against the benchmark is made in section 4.
SCATTER_METRICS = [("plddt_ks_residue", "KS-score", KS_C)]
# All four continuous-assay datasets (analyses/continuous/continuous_functional).
FUNCTIONAL = ["TP53", "KRAS", "PTEN", "MSH2"]

plt.rcParams.update({
    "font.family": "sans-serif", "axes.edgecolor": "#c9ccd2",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GREY, "ytick.color": GREY, "figure.dpi": 120,
})


def annotation_genes():
    from drift.run import GT_GROUPS
    return sorted(GT_GROUPS["annotation"])


def singleton_scores(seed):
    """The N=1 cache. KS at N=1 is carried by the cache itself.

    The independent snapshot (inputs/ks_n1.csv.gz) was the guard while the N=1
    KS column could be blank; it records the pre-v5 cohort, so it is no longer
    compared row for row. The finiteness check below is the remaining guard.
    """
    current = pd.read_csv(CACHE / (
        f"labeled_size1_seed{seed}_{FINGERPRINT}_esfeat7.csv"
    )).sort_values(["gene", "mutation"]).reset_index(drop=True)
    if not np.isfinite(current.plddt_ks_residue).all():
        raise ValueError(f"N=1 cache for seed {seed} lacks KS; run restore_ks_n1.py")
    return current


def per_gene_curves(genes):
    """Per-gene ROC-AUC for each descriptor at every (N, seed) in the cache.

    Scores are driver-high: label 1 is Driver and the descriptor is used
    unsigned, which is what reproduces the published panel.
    """
    rows = []
    files = sorted(glob.glob(str(CACHE / f"labeled_size*_seed*_{FINGERPRINT}_esfeat7.csv")))
    if not files:
        raise SystemExit(f"no esfeat7 caches for {FINGERPRINT} under {CACHE}")
    cols = ["gene", "label"] + [c for _, c, _ in DESCRIPTORS + [ANCHOR]]
    for f in files:
        m = re.search(r"size(\d+)_seed(\d+)", f)
        n, seed = int(m.group(1)), int(m.group(2))
        d = singleton_scores(seed) if n == 1 else pd.read_csv(f, usecols=cols)
        for gene, block in d[d.gene.isin(genes)].groupby("gene"):
            if block.label.nunique() < 2:
                continue
            for label, col, _ in DESCRIPTORS + [ANCHOR]:
                if n == 1 and label != ANCHOR[0]:
                    continue
                # The anchor exists only at N=1; past that the KS curve is the
                # measurement and a second line would just crowd it.
                if label == ANCHOR[0] and n != 1:
                    continue
                rows.append({"N": n, "seed": seed, "gene": gene,
                             "descriptor": label,
                             "auc": roc_auc_score(block.label, block[col])})
    curves = pd.DataFrame(rows)
    if curves.empty:
        raise SystemExit("no annotation genes found in the caches")
    return curves


def functional_scatter():
    """Per-variant residue KS (N=90, mean over seeds 42--51) against the assay.

    Read from the continuous cohort's four-gene table, written by
    analyses/continuous/continuous_functional/plot_four_gene_ks.py: TP53 (log10
    selection score), the KRAS4B blind sample (signed HA1E LFC), PTEN
    (-Cum_score) and MSH2 (LOF score). No binary label is involved.
    """
    path = CONT_SRC / "four_gene_ks" / "variant_scores.csv"
    if not path.is_file():
        raise SystemExit(f"run analyses/continuous/continuous_functional/plot_four_gene_ks.py first ({path})")
    d = pd.read_csv(path)
    return d[d.gene.isin(FUNCTIONAL)].reset_index(drop=True)


def _gz(gz_dir, table, **eq):
    """One (method, arm) slice of the non-learned sweep, with its AM reference.

    Both tables carry the raw inputs alongside the fusions under kind="input",
    so AlphaMissense comes from the same file and the same folds as the arm it
    is compared against, rather than from a second sweep that would have to be
    trusted to have drawn its folds the same way.
    """
    d = pd.read_csv(gz_dir / table)
    for k, v in eq.items():
        d = d[d[k] == v]
    arm = d[(d.method == GZ_METHOD) & (d.arm == GZ_ARM)]
    afm = d[(d.method == "AlphaMissense (raw)") & (d.arm == "afm_score")]
    if arm.empty:
        raise SystemExit(f"{GZ_METHOD} x {GZ_ARM} missing from {gz_dir/table}")
    return arm, afm


def gz_macro_curve(gz_dir):
    """Macro against ensemble size for the non-learned arm, per seed.

    Same contract as the logistic fusion_macro_curve it replaces -- one entry
    keyed by arm name, indexed by size, carrying the AUC and its seed spread --
    so the panel does not have to know which estimator produced it.
    """
    arm, afm = _gz(gz_dir, "sweep_macro.csv", protocol="LOGO")
    ref = afm.groupby("size")["macro_roc_auc"].mean()
    per_seed = arm.pivot_table(index="size", columns="seed",
                               values="macro_roc_auc")
    # N=1 is recomputed below from the independently verified singleton input.
    per_seed = per_seed.drop(index=1, errors="ignore")
    delta = per_seed.sub(ref.reindex(per_seed.index), axis=0)
    out = {GZ_LABEL: pd.DataFrame({"auc": per_seed.mean(axis=1),
                                 "auc_sd": per_seed.std(axis=1, ddof=1),
                                 "delta": delta.mean(axis=1),
                                 "delta_sd": delta.std(axis=1, ddof=1)})}
    # Recompute the N=1 fusion with the same grouped LOGO scaler and raw KS.
    from analyses.annotation.simple_fusion_benchmark.fusion import normalise, combine
    from metric_analysis.pipeline.library.canonical_cohort import logo_splits
    singleton_rows = []
    for seed in sorted(arm.seed.unique()):
        frame = singleton_scores(seed)
        frame = frame[frame.gene.isin(annotation_genes())].reset_index(drop=True)
        genes = frame.gene.to_numpy()
        values = frame[["afm_score", "plddt_ks_residue"]].to_numpy()
        score = combine(normalise(values, genes, "gz", logo_splits(genes)), "mean")
        aucs = [roc_auc_score(frame.loc[genes == gene, "label"], score[genes == gene])
                for gene in sorted(set(genes))]
        singleton_rows.append((int(seed), float(np.mean(aucs))))
    singleton_auc = pd.Series(dict(singleton_rows))
    anchor = (float(singleton_auc.mean()), float(singleton_auc.std(ddof=1)))
    delta.loc[1] = singleton_auc.reindex(delta.columns) - float(ref.loc[1])
    delta = delta.sort_index()
    return out, {GZ_LABEL: delta}, ref, anchor


def gz_tables(tab_dir):
    """The per-gene metric battery at N=90, one frame per model.

    Read from results_tables_annotation_N90 rather than recomputed here. That
    experiment exists because the size sweep stores only three metrics per gene
    and the full battery in results_v1 was computed over SEVEN-gene folds --
    wrong for this cohort, since every threshold policy reads the training
    genes and the annotation cohort has five. Recomputing it in this script
    would reintroduce exactly that mistake.
    """
    d = pd.read_csv(tab_dir / "table1_raw.csv")
    out = {m: d[d.model == m].set_index("gene") for m, _, _ in TABLE_MODELS}
    return _am_anchored_thresholds(out) if AM_ANCHORED.is_file() else out


# 2026-10-17 (read-through C6, supervisor Q11): the thresholded columns (F1, Acc,
# bal.acc) are taken at AlphaMissense's published likely-pathogenic threshold, not
# at a cut-off VALUE transferred from the training genes. AM calls driver above
# 0.564; the combination calls the same number of variants per dataset, in order of
# its own score (analyses/annotation/am_operating_point/am_anchored.py). The transferred
# value landed in the wrong place on the combined score's gene-dependent scale
# (MSH6 too high, POLE too low) and made the combination look worse at an
# unchanged ranking. Ranking columns (ROC-AUC, AP, prec@R) are unchanged.
AM_ANCHORED = (REPO / "analyses" / "annotation" / "am_operating_point" / "results"
               / "am_anchored_operating_point.csv")
_ANCHORED_COLS = {"F1": "F1", "accuracy": "Acc", "bal.acc": "bal.acc"}
_ANCHORED_ARM = {"AlphaMissense": "AM", "gz_mean": "combo"}


def _am_anchored_thresholds(tables):
    a = pd.read_csv(AM_ANCHORED).set_index("gene")
    for model, frame in tables.items():
        arm = _ANCHORED_ARM[model]
        for col, src in _ANCHORED_COLS.items():
            values = a[f"{arm}_{src}"]
            missing = [g for g in frame.index if g not in values.index]
            if missing:
                raise SystemExit(f"{AM_ANCHORED.name}: no {src} for {missing}")
            frame[col] = values.reindex(frame.index).to_numpy(float)
    return tables


def gz_deltas(gz_dir, genes, size=90):
    """Per-gene change against AlphaMissense for the non-learned arm."""
    arm, afm = _gz(gz_dir, "sweep_per_gene.csv", protocol="LOGO")
    arm, afm = arm[arm["size"] == size], afm[afm["size"] == size]
    # AlphaMissense never sees a structure, so its per-gene AUC is identical
    # across the ten subsampling seeds; the mean is exact, not an estimate.
    ref = afm.groupby("gene")["roc_auc"].mean()
    per_seed = arm.pivot_table(index="seed", columns="gene", values="roc_auc")
    delta = per_seed.sub(ref.reindex(per_seed.columns), axis=1)
    missing = [g for g in genes if g not in delta.columns]
    if missing:
        raise SystemExit(f"{GZ_ARM}: no rows for {missing} in {gz_dir.name}")
    return ({GZ_LABEL: pd.DataFrame({"mean": delta.mean(),
                                     "sd": delta.std(ddof=1)})}, ref)


def band(fig, gs_row, text, sub):
    """A left-edge section label spanning one row of the grid.

    The figure has three arguments stacked on top of each other -- descriptor
    behaviour, its summary, and what it adds to AlphaMissense -- and without a
    band the reader has to infer the grouping from panel letters alone.
    """
    ax = fig.add_subplot(gs_row)
    ax.axis("off")
    # Two stacked rotated labels inside a narrow column: keep them well apart
    # or the subtitle rides on the heading and on the panel's own y-label.
    ax.text(.14, .60, text, rotation=90, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK,
            transform=ax.transAxes)
    # 9 pt, not 7.2: at 7.2 the subtitle printed smaller than every tick label
    # in the figure (2026-10-23, author's note).
    ax.text(.74, .60, sub, rotation=90, ha="center", va="center",
            fontsize=9, color="#555555", transform=ax.transAxes)


def trend(ax, x, y, colour):
    """Least-squares line through a scatter, plus the numbers it summarises.

    Spearman is the rank statistic the continuous work is scored on; R^2 is the
    variance the drawn line explains, so the annotation and the line are the
    same fit rather than two unrelated summaries of the same cloud.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color=colour, lw=1.7, zorder=4)
    return (float(spearmanr(x, y).statistic),
            float(np.corrcoef(x, y)[0, 1] ** 2), int(m.sum()))


def metric_table(ax, block, genes, title, accent, letter):
    """One per-gene metric table, drawn as an axes so it places like a panel."""
    ax.axis("off")
    rows = [g for g in genes if g in block.index]
    rows += [g for g in ("MACRO",) if g in block.index]
    body = []
    for g in rows:
        r = block.loc[g]
        # n is a count, not a measurement, and MACRO has none -- it is an
        # average over genes, not a sixth dataset with its own variants.
        cells = ["--" if pd.isna(r["n"]) else f"{int(r['n']):,}"]
        cells += [f"{float(r[TABLE_SRC[c]]):.3f}" for c in TABLE_COLS[1:]]
        body.append(cells)
    # Columns sized to their contents, not equally: an equal split gives "n"
    # the same width as "ROC-AUC", and with pAUC(.1) added the two long headers
    # became adjacent and ran into each other. Cells are 5 characters ("0.923"),
    # so the width each column needs is whichever of header and cell is longer.
    units = [max(len(c), 5) for c in TABLE_COLS]
    widths = [u / sum(units) for u in units]
    t = ax.table(cellText=body, rowLabels=rows, colLabels=TABLE_COLS,
                 colWidths=widths, cellLoc="right", rowLoc="left",
                 loc="center", bbox=[.135, .00, .85, .90])
    t.auto_set_font_size(False)
    # Keep all seven numeric columns legible at the final figure scale.
    t.set_fontsize(6.2)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#dfe1e5")
        c.set_linewidth(.5)
        if i == 0:                                   # header
            c.set_facecolor(accent)
            c.set_alpha(.16)
            c.get_text().set_fontweight("bold")
        elif rows[i - 1] == "MACRO":                 # the summary row
            c.set_facecolor("#f1f2f4")
            c.get_text().set_fontweight("bold")
        if j == -1:                                  # gene names
            c.get_text().set_fontweight("bold")
            c.get_text().set_color(GENE_C.get(rows[i - 1], INK)
                                   if i > 0 and rows[i - 1] != "MACRO" else INK)
    ax.set_title(title, fontsize=9, color=INK, pad=8)
    ax.text(-.02, 1.05, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK)


def draw(curves, genes, tables, scatter, macro_curve, afm_ref, out,
         fusion_anchor=None):
    # Keep the fusion curve and both tables together, reserving more width
    # for the seven-column tables than for the single-curve panel.
    fig = plt.figure(figsize=(10.8, 9.8))
    gs = fig.add_gridspec(3, 7, height_ratios=[1.0, 1.05, 1.0],
                          width_ratios=[.46] + [1] * 6,
                          left=.04, right=.985, bottom=.075, top=.945,
                          hspace=.48, wspace=.90)
    band(fig, gs[0, 0], "Driver vs passenger",
         "KS-score against ensemble size")
    # The row is the continuous functional endpoints of the text and caption,
    # set against row 1's categorical driver/passenger labels; "Functional
    # assays" named where the numbers come from, not what the row tests.
    band(fig, gs[1, 0], "Continuous endpoints",
         "KS-score against measured function")
    band(fig, gs[2, 0], "With AlphaMissense",
         "equal-weight average, nothing fitted")
    # The saved singleton rows use a historical label but measure the same KS.
    curves = curves.copy()
    curves.loc[curves.descriptor.eq(ANCHOR[0]), "descriptor"] = DESCRIPTORS[0][0]
    sizes = sorted(curves.N.unique())

    def logx(ax):
        ax.set_xscale("log")
        ax.set_xlim(min(sizes) / 1.35, max(sizes) * 1.08)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], fontsize=7.5)
        ax.grid(alpha=.25, lw=.6)

    label, _, colour = DESCRIPTORS[0]

    # a -- per dataset
    ax = fig.add_subplot(gs[0, 1:4])
    # End-of-curve labels live outside the axes, so give them room inside this
    # cell rather than letting them run into panel b's y axis.
    _p = ax.get_position()
    ax.set_position([_p.x0, _p.y0, _p.width * .88, _p.height])
    sub = curves[curves.descriptor == label]
    ends = []
    for gene in genes:
        g = (sub[sub.gene == gene].groupby("N")["auc"]
             .agg(["mean", "std"]).reindex(sizes))
        ax.errorbar(sizes, g["mean"], yerr=g["std"], marker="o", ms=3.6,
                    lw=1.5, capsize=2.4, color=GENE_C.get(gene), label=gene)
        ends.append([float(g["mean"].iloc[-1]), gene])
    # No AlphaMissense reference and no chance line here: section 1 shows how
    # the descriptor behaves on its own. The benchmark comparison is section 3.
    logx(ax)
    ax.set_ylim(.45, 1.0)
    # End-of-curve labels rather than a legend. A legend was used here because
    # the curves converge at large N (POLE on PTEN, VHL on POLE) and inline
    # text collided; nudging the labels apart fixes that without spending a
    # corner of the data area, and it removes the colour-to-name lookup the
    # reader otherwise has to do five times.
    ends.sort(key=lambda t: t[0])
    gap = .082 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, gene in ends:
        ax.annotate(gene, xy=(1.015, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=7, fontweight="bold",
                    color=GENE_C[gene], annotation_clip=False)
    ax.set_xlabel("Predictions per ensemble (N)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"{label}, per dataset", fontsize=9.5, color=INK)
    ax.text(-.14, 1.07, "a", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK)

    # b -- the same measurement pooled: cohort macro, sharing the row with a
    ax = fig.add_subplot(gs[0, 4:7])
    per_seed = (curves[curves.descriptor == label]
                .groupby(["N", "seed"])["auc"].mean()
                .groupby("N").agg(["mean", "std"]).reindex(sizes))
    ax.errorbar(sizes, per_seed["mean"], yerr=per_seed["std"], marker="o",
                ms=3.8, lw=1.6, capsize=2.6, color=colour)
    logx(ax)
    ax.set_xlabel("Predictions per ensemble (N)")
    ax.set_ylabel("Mean ROC-AUC")
    ax.set_title(f"Mean over the {len(genes)} annotated genes",
                 fontsize=9.5, color=INK)
    ax.margins(y=.14)
    ax.text(-.14, 1.07, "b", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK)

    # c -- the descriptor/assay relationship itself, at the largest N.
    # AlphaMissense is not a column here: this section is about what the two
    # descriptors see on their own, and the AM comparison lives in section 4.
    # Four panels in one row rather than a 2x2 block: gene-major, so the two
    # descriptors for a gene sit side by side on a shared assay axis, and the
    # row fills the width instead of leaving both margins empty.
    # c, d -- one scatter per gene, since resKS is now the only score column
    col, slabel, scolour = SCATTER_METRICS[0]
    func_row = gs[1, 1:7].subgridspec(1, len(FUNCTIONAL), wspace=.42)
    for k, gene in enumerate(FUNCTIONAL):
        ax = fig.add_subplot(func_row[k])
        block = scatter[scatter.gene == gene]
        # TP53 1174 and KRAS4B 935 points are dense; PTEN 253 and MSH2 106 are not.
        pts = int(np.isfinite(block[col]).sum())
        ax.scatter(block[col], block.y, s=9 if pts > 200 else 16,
                   color=FUNC_C[gene], alpha=.38 if pts > 200 else .8,
                   linewidths=0, zorder=2)
        rho, r2, n = trend(ax, block[col], block.y, TREND_C)
        ax.annotate(f"$\\rho$ = {rho:+.2f}\n$R^2$ = {r2:.2f}",
                    (.035, .96), xycoords="axes fraction", fontsize=7.5,
                    color=INK, va="top", linespacing=1.45)
        ax.grid(alpha=.22, lw=.6)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{FUNC_TITLE[gene]}  (n = {n:,})", fontsize=9.5, color=INK)
        ax.set_xlabel(f"Mean {slabel}, N=90", fontsize=8)
        ax.set_ylabel(FUNC_YLABEL[gene], fontsize=8)
        ax.text(-.30, 1.07, "cdef"[k], transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=INK)

    # e -- the same fusion from N=1 through N=90.
    fusion_row = gs[2, 1:7].subgridspec(1, 3,
        width_ratios=[.85, 1.28, 1.28], wspace=.26)
    ax = fig.add_subplot(fusion_row[0])
    for name, blk in macro_curve.items():
        blk = blk.copy()
        if fusion_anchor is not None and name == GZ_LABEL:
            blk.loc[1, ["auc", "auc_sd"]] = fusion_anchor
            blk = blk.sort_index()
        ax.errorbar(blk.index, blk["auc"], yerr=blk["auc_sd"], marker="o",
                    ms=3.6, lw=1.5, capsize=2.4, color=ARM_C[name], label=name)
    # Flat by construction: AlphaMissense never sees a structure, so its macro
    # cannot move with N. That is what makes it a reference rather than a curve.
    # axhline, not a line over the curve's own x: the reference has to run
    # across N=1 as well, to compare the same fusion across sampling depths.
    ax.axhline(float(afm_ref.iloc[0]), ls="--", lw=1.4, color=AFM_C,
               label="AlphaMissense", zorder=2)
    logx(ax)
    ax.set_xlabel("Predictions per ensemble (N)")
    # "driver-high" is already stated in panels a and b.
    ax.set_ylabel("Mean ROC-AUC")
    ax.set_title("Combination against ensemble size", fontsize=9,
                 color=INK)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.margins(y=.12)
    ax.text(-.05, 1.11, "g", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK)

    # f, g -- the per-gene battery at N=90, one table per model. These replace
    # the functional AUC curve and the per-dataset delta bars: the bars showed
    # one number per gene and the reader had to take the rest on trust, and
    # side-by-side raw tables let the comparison be made column by column.
    for k, (model, title, accent) in enumerate(TABLE_MODELS):
        ax = fig.add_subplot(fusion_row[1 + k])
        metric_table(ax, tables[model], genes, title, accent, "hi"[k])

    # No suptitle or footnote: the legend carries both (the N=1 meaning of KS is
    # its last sentence), and journals do not want titles inside the artwork.
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Figure2.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure2.png / .pdf -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gz", type=Path, default=DEFAULT_GZ,
                    help="non-learned fusion size sweep, for the fusion panels")
    ap.add_argument("--tables", type=Path, default=DEFAULT_TABLES,
                    help="per-gene metric battery at N=90, for panels f and g")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    genes = annotation_genes()
    print(f"[figure2] annotation cohort: {', '.join(genes)}")
    curves = per_gene_curves(genes)
    # Panels e and g come from the NON-LEARNED benchmark.
    deltas, afm = gz_deltas(args.gz, genes)
    scatter = functional_scatter()
    macro_curve, macro_seeds, afm_ref, fusion_anchor = gz_macro_curve(args.gz)
    tables = gz_tables(args.tables)
    print(f"[figure2] fusion panels: {GZ_METHOD} x {GZ_ARM} from {args.gz.name}")
    args.out.mkdir(parents=True, exist_ok=True)
    scatter.to_csv(args.out / "panel_cdef_mean_scores.csv", index=False)
    associations = []
    for gene, block in scatter.groupby("gene"):
        x, y = block.plddt_ks_residue, block.y
        associations.append({"gene": gene, "n": len(block),
                             "spearman_rho": spearmanr(x, y).statistic,
                             "r_squared": np.corrcoef(x, y)[0, 1] ** 2})
    pd.DataFrame(associations).to_csv(args.out / "panel_cdef_associations.csv", index=False)
    import json
    (args.out / "panel_cdef_provenance.json").write_text(json.dumps({
        "source": "analyses/continuous/continuous_functional/four_gene_ks/variant_scores.csv",
        "provenance": "analyses/continuous/continuous_functional/four_gene_ks/provenance.json",
        "cohort": "continuous: TP53 all modelled and scored; KRAS4B blind stratum; "
                  "PTEN and MSH2 assay-scored variants in the strict manifest",
        "fusion_caches": FINGERPRINT}, indent=2) + "\n")
    curves.to_csv(args.out / "panel_abcd_per_gene_auc.csv", index=False)
    pd.DataFrame([{"N": 1, "arm": GZ_ARM, "method": GZ_METHOD,
                   "auc": fusion_anchor[0], "auc_sd": fusion_anchor[1]}]).to_csv(
        args.out / "panel_g_singleton_fusion.csv", index=False)
    pd.concat({k: v for k, v in deltas.items()}, names=["arm"]).to_csv(
        args.out / "audit_gz_fusion_delta.csv")
    pd.concat({k: v for k, v in macro_curve.items()},
              names=["arm"]).to_csv(args.out / "panel_g_fusion_vs_size.csv")
    pd.concat({k: v for k, v in macro_seeds.items()},
              names=["arm"]).to_csv(args.out / "fusion_macro_per_seed.csv")
    draw(curves, genes, tables, scatter, macro_curve, afm_ref, args.out,
         fusion_anchor)
    print(f"[figure2] tables + figure written to {args.out}")


if __name__ == "__main__":
    main()
