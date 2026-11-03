#!/usr/bin/env python
"""Draft article panels, each written as an INDEPENDENT figure file.

Why independent files rather than one composite: the Brief Communication target
allows two or three display items, so the panel set has to be re-cut repeatedly.
Emitting one file per panel means the layout decision is made in the manuscript,
not here, and any single panel can be dropped into a slide or a supplement
without re-rendering its neighbours.

Every panel is computed from the live repository artifacts -- the D1 caches, the
per-gene ensemble sweeps, the benchmark summaries -- so re-running this after a
pipeline change updates the draft figures rather than leaving a stale PNG behind.
Numbers are also written to article_panel_values.csv so a caption can quote them
without re-deriving anything by hand.

Usage
-----
    python -m drift.article_figures
    python -m drift.article_figures --size 90 --seed 42 --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Type 42 (TrueType) rather than the Type 3 default: several journals reject
# Type 3 outright, and it is the single most common figure-submission bounce.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from .run import GT_GROUPS, _apply_strict_canonical_cohort
from .protein_config import protein_config_fingerprint

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# metric_analysis may be installed (editable) from another checkout; put this
# repository first so it imports its own tree.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metric_analysis.pipeline.paths import manuscript_root, tagged  # noqa: E402

RESULTS = HERE / "results"
CACHE = HERE / "cache"
CANONICAL_SIZE_SCORES = (
    RESULTS / "afm_fusion" / tagged("cohort_size_sweep")
    / "per_variant_seed_requested_sizes.csv.gz"
)
SAMPLING_POLICY_SCORES = RESULTS / tagged("sampling_policy_d1") / "summary.csv"
# The canonical AM fusion run whose models trained on the annotation datasets
# only (canonical_afm --cohort annotation). The mixed run in canonical_<tag>
# trained on the functional-assay genes too and is not read here (2026-10-12).
CANONICAL_RUN = RESULTS / "afm_fusion" / tagged("canonical_annotation")
CANONICAL_LRT = CANONICAL_RUN / "incremental_lrt_per_gene.csv"
# The 72-candidate fusion benchmark, trained and evaluated inside the annotation
# group (drift.run --track benchmark --gt-group annotation).
BENCHMARK_SUMMARY = (
    RESULTS / tagged("benchmark_gt_group") / "annotation" / "benchmark_summary.csv"
)

DRV, PAS, AMC, ACC, GRY = "#d1495b", "#2e86ab", "#2e86ab", "#6a4c93", "#8d8d8d"
# Figure1_resKS row 4 type scale. Panels f and g are drawn by
# experiments/figure1_ks_combo/make_panels_gh.py under these rcParams and panel
# h is drawn here, so h has to carry the same ones or the assembled row mixes
# two type scales. Copied from that script, not imported: importing it would
# apply its rcParams to every panel in this module. Keep the two in step.
FIG1_ROW4_RC = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#333333", "axes.linewidth": .8,
    "axes.labelcolor": "#000000", "axes.labelsize": 10,
    "axes.titlesize": 10.5, "axes.titleweight": "normal", "axes.titlecolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.fontsize": 8.5,
    "legend.edgecolor": "#bfbfbf", "legend.framealpha": 1.0,
    "legend.borderpad": .4, "legend.handlelength": 1.8,
}
# cache gene keys vs metric_analysis directory names
# MutSalpha is carried as its two subunits since the 2026-09-07 split. The
# fusion/metric_analysis name divergence ("MutSalpha" vs "MutS_alpha") does not
# apply to MSH2/MSH6 -- they spell the same in both.
# TP53/KRAS have no binary label since 2026-10-13 (continuous targets only).
GENES = ["POLE", "MSH2", "MSH6", "PTEN", "VHL", "PIK3CA"]
# The retired mixed set: functional-assay and annotation datasets together.
# No panel reads it since 2026-10-12; it goes when the mixed outputs do.
EVALUATION_GENES = ["POLE", "MSH2", "MSH6", "PTEN", "VHL", "PIK3CA"]
# The annotation-label datasets: the only ones the macro panels average over,
# and the only ones their fitted models trained on (2026-10-12). Read from
# run.GT_GROUPS so the panels cannot drift from the benchmark run they read.
# MSH6 is IN the macro by decision (2026-09-09): it is a real dataset and
# dropping it for having a wide interval would be selecting on the result.
MACRO_GENES = sorted(GT_GROUPS["annotation"])
# Must match canonical_afm.HEADLINE_MACRO -- the scope name these figures read
# out of the canonical AFM summary table.
HEADLINE_MACRO = "annotation"
SWEEP_DIR = {g: ("MutS_alpha" if g == "MutSalpha" else g) for g in GENES}

VALUES: list[dict] = []


def record(panel, key, value, note=""):
    VALUES.append({"panel": panel, "quantity": key, "value": value, "note": note})
    return value


def save(fig, out_dir, name):
    """Write PDF and PNG for the same panel.

    PDF is what goes into the manuscript -- vector, so text stays sharp at any
    magnification and a reviewer cannot object to resolution. PNG exists so the
    HTML gallery can render the panel inline; nothing should cite it.
    """
    out_dir = Path(out_dir)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}  (pdf + png)")
    return pdf


# ── inputs ────────────────────────────────────────────────────────────────────

def load_labeled(size, seed):
    fingerprint = protein_config_fingerprint()
    hits = [p for p in CACHE.glob(f"labeled_size{size}_seed{seed}_*proteins{fingerprint}_esfeat*.csv")
            if "nested-" not in p.name and "top_confidence" not in p.name]
    if not hits:
        raise SystemExit(f"no default-policy cache for size={size} seed={seed} "
                         f"proteins{fingerprint} under {CACHE}; rebuild before plotting")
    frame = pd.read_csv(max(hits, key=lambda p: p.stat().st_mtime))
    return _apply_strict_canonical_cohort(frame)


def annotation_macro_series():
    """D1 macro AUC over MACRO_GENES from the canonical seven-size descriptor table."""
    if not CANONICAL_SIZE_SCORES.is_file():
        raise FileNotFoundError(
            f"missing {CANONICAL_SIZE_SCORES}; run drift.cohort_size_sweep"
        )
    scores = pd.read_csv(
        CANONICAL_SIZE_SCORES,
        usecols=["protein", "label", "seed_index", "requested_size", "d1"],
    )
    rows = []
    for (size, seed_index), block in scores.groupby(
        ["requested_size", "seed_index"], sort=True
    ):
        aucs = []
        for protein in [g.replace("MutSalpha", "MutS_alpha") for g in MACRO_GENES]:
            gene = block[block.protein.eq(protein)]
            aucs.append(roc_auc_score(gene.label, gene.d1))
        rows.append({"size": int(size), "seed": int(seed_index), "auc": np.mean(aucs)})
    return pd.DataFrame(rows).groupby("size").auc.agg(["mean", "std"])


def macro_series(run_dir):
    """macro D1 AUC by ensemble size from one sweep directory."""
    p = RESULTS / run_dir / "sweep_runs.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    m = d[(d.model == "d1") & (d.scope == "pooled_macro")]
    if "gt_group" in m.columns:
        m = m[m.gt_group == "all"]
    return m.groupby("size").auc.agg(["mean", "std"]) if len(m) else None


def canonical_gene_size_series():
    """Within-gene D1 AUCs from the same audited table as the macro curve."""
    if not CANONICAL_SIZE_SCORES.is_file():
        raise FileNotFoundError(
            f"missing {CANONICAL_SIZE_SCORES}; run drift.cohort_size_sweep"
        )
    scores = pd.read_csv(
        CANONICAL_SIZE_SCORES,
        usecols=["protein", "label", "seed_index", "requested_size", "d1"],
    )
    rows = []
    for (protein, size, seed_index), block in scores.groupby(
        ["protein", "requested_size", "seed_index"], sort=True
    ):
        rows.append({
            "protein": protein.replace("MutS_alpha", "MutSalpha"),
            "size": int(size),
            "seed": int(seed_index),
            "auc": roc_auc_score(block.label, block.d1),
        })
    return pd.DataFrame(rows)


def benchmark_macro():
    if not BENCHMARK_SUMMARY.exists():
        return None
    d = pd.read_csv(BENCHMARK_SUMMARY)
    return d[d.scope == "macro"]


def canonical_am_d1_result():
    d = pd.read_csv(CANONICAL_RUN / "n90_bootstrap_summary.csv")
    row = d[
        (d.evaluation == "logo_logistic_original") &
        (d.scope == HEADLINE_MACRO) &
        (d.model == "afm_d1")
    ]
    if len(row) != 1:
        raise ValueError("canonical strict AM+D1 result is missing or duplicated")
    return row.iloc[0]


# ── the direct test: does D1 add information beyond AlphaMissense? ─────────────

def lr_test(d):
    """Likelihood-ratio test, 1 df: label ~ AM  vs  label ~ AM + log10(D1).

    Within one gene, so no between-gene scale confound. This asks whether D1
    carries information about the label that AlphaMissense does not -- a
    different and easier question than whether a cross-gene model can exploit
    it, which the nested benchmark answers separately (and negatively).
    Deliberately in-sample: it is an association test, not a performance claim.
    """
    y = d.label.values.astype(int)
    am = d.afm_score.values.reshape(-1, 1)
    ld = np.log10(d.d1_score.values).reshape(-1, 1)

    def ll(X):
        m = LogisticRegression(max_iter=5000, C=1e6).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    chi2 = max(2 * (ll(StandardScaler().fit_transform(np.hstack([am, ld])))
                    - ll(StandardScaler().fit_transform(am))), 0.0)
    return chi2, stats.chi2.sf(chi2, 1)


# ── panels ────────────────────────────────────────────────────────────────────

def panel_ensemble_size(out):
    s = annotation_macro_series()
    if s is None: return
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", ms=5, capsize=3, color=DRV, lw=1.9)
    cohort = f"mean over {len(MACRO_GENES)} annotation datasets"
    gain = record("F2a", "macro_gain_N1_to_N90",
                  float(s.loc[90, "mean"] - s.loc[1, "mean"]), cohort)
    ax.annotate("", xy=(90, s.loc[90, "mean"]), xytext=(1, s.loc[1, "mean"]),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.1, ls=":"))
    ax.text(16, (s.loc[1, "mean"] + s.loc[90, "mean"]) / 2 + .011, f"+{gain:.3f}",
            fontsize=13, weight="bold")
    record("F2a", "macro_auc_N1", float(s.loc[1, "mean"]), cohort)
    record("F2a", "macro_auc_N90", float(s.loc[90, "mean"]), cohort)
    ax.set_xlabel("Structures per variant (N)"); ax.set_ylabel("macro ROC-AUC")
    ax.set_title("One predicted structure is not enough", fontsize=10)
    ax.grid(alpha=.25)
    save(fig, out, "F2a_ensemble_size_macro")


def panel_per_gene_size(out):
    scores = canonical_gene_size_series()
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for g, c in zip(GENES, plt.cm.tab10.colors):
        s = scores[scores.protein.eq(g)].groupby("size").auc.agg(["mean", "std"])
        if s.empty:
            continue
        ax.plot(s.index, s["mean"], marker="o", ms=3, lw=1.3, color=c,
                label=g.replace("MutSalpha", "MutSa"))
        lo, hi = float(s["mean"].iloc[0]), float(s["mean"].iloc[-1])
        record("F2b", f"{g}_gain_N1_to_N90", hi - lo)
    ax.set_xlabel("N"); ax.set_ylabel("within-gene ROC-AUC")
    ax.set_title("Every dataset gains with ensemble size", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower right"); ax.grid(alpha=.25)
    save(fig, out, "F2b_ensemble_size_per_gene")


def panel_sampling_policies(out):
    if not SAMPLING_POLICY_SCORES.is_file():
        raise FileNotFoundError(
            f"missing {SAMPLING_POLICY_SCORES}; run drift.sampling_policy_d1"
        )
    scores = pd.read_csv(SAMPLING_POLICY_SCORES)
    grouped = {
        name: block.groupby("size").macro_auc.agg(["mean", "std"])
        for name, block in scores.groupby("policy")
    }
    ind, pin, con = (grouped["uniform_random"], grouped["wt_pinned"],
                     grouped["highest_plddt"])
    Ns = [N for N in (1, 20, 90) if all(N in x.index for x in (ind, pin, con))]
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    x = np.arange(len(Ns)); w = .26
    for k, (d, c, n) in enumerate(((ind, DRV, "uniform random"), (pin, GRY, "WT-pinned"),
                                   (con, ACC, "highest-pLDDT"))):
        ax.bar(x + (k - 1) * w, [d.loc[N, "mean"] for N in Ns], w, color=c, alpha=.88, label=n)
    if 90 in Ns:
        delta = record("F2c", "confidence_minus_pinned_N90",
                       float(con.loc[90, "mean"] - pin.loc[90, "mean"]),
                       "negative => selecting confident structures hurts")
        ax.annotate(f"{delta:+.4f}", xy=(x[-1] + w, con.loc[90, "mean"]),
                    xytext=(x[-1] + w, con.loc[90, "mean"] + .045), ha="center",
                    fontsize=10, weight="bold", color="#b00020",
                    arrowprops=dict(arrowstyle="->", color="#b00020", lw=1))
    if 1 in Ns:
        record("F2c", "pairing_gain_N1", float(pin.loc[1, "mean"] - ind.loc[1, "mean"]))
    ax.set_xticks(x); ax.set_xticklabels([f"N={N}" for N in Ns])
    ax.set_ylim(.55, .80); ax.set_ylabel("macro ROC-AUC")
    ax.set_title("Keeping only confident structures hurts at $N\\geq$20:\nlow-confidence members carry signal", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(axis="y", alpha=.25)
    save(fig, out, "F2c_sampling_policies")


def panel_continuum(out, labeled):
    gt_path = REPO / "Ground_Truths" / "Supplementary table 3.xlsx"
    if not gt_path.exists(): return
    tp = labeled.query("gene=='TP53'")
    gt = pd.read_excel(gt_path)
    if "Allele" not in gt.columns: return
    m = tp.merge(gt, left_on="mutation", right_on="Allele")
    m["y"] = pd.to_numeric(m.get("A549_p53WT_Nutlin-3_Z-score"), errors="coerce")
    m = m[m.y.notna()].copy()
    if len(m) < 50: return
    m["q"] = pd.qcut(m.d1_score, 5, labels=False)
    groups = [m[m.q == k].y.values for k in range(5)]
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    vp = ax.violinplot(groups, positions=range(1, 6), widths=.82, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(DRV); b.set_alpha(.30); b.set_edgecolor(DRV)
    ax.boxplot(groups, positions=range(1, 6), widths=.15, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white"), medianprops=dict(color="black", lw=1.4))
    r, p = stats.spearmanr(m.d1_score, m.y)
    record("F2d", "TP53_spearman_D1_vs_assay", float(r))
    record("F2d", "TP53_spearman_p", float(p))
    record("F2d", "TP53_n", int(len(m)))
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(["1\nlow", "2", "3", "4", "5\nhigh"])
    ax.set_xlabel("D1 quintile"); ax.set_ylabel("TP53 functional score (A549 Nutlin-3 Z)")
    ax.set_title(rf"D1 is graded, not just binary ($\rho$={r:+.2f}, n={len(m)})", fontsize=10)
    ax.grid(axis="y", alpha=.2)
    save(fig, out, "F2d_continuum_TP53")


def descriptor_families():
    """Best-direction AUC per descriptor family per protein, standalone sets.

    Reads the unified cross-metric comparison the per-protein pipeline writes.
    The families are NOT matched in size: D1 contributes a single pre-specified
    metric while RMSD contributes five and RMSF and DNA-distance eight each, so
    taking the best of each family is optimistic for every family EXCEPT D1.
    That asymmetry is the point of the panel and belongs in its caption.
    """
    rows = []
    for f in sorted(glob.glob(str(REPO / "metric_analysis" / "results" / "*" / "unified"
                                 / "*_all_metrics_unified_comparison.csv"))):
        prot = Path(f).parents[1].name
        d = pd.read_csv(f)
        if "roc_auc" not in d.columns:
            continue
        # A descriptor pointing the other way is still a descriptor; the sweep
        # reports raw AUC, so fold values below 0.5 up before comparing.
        d["best_dir"] = d.roc_auc.where(d.roc_auc >= .5, 1 - d.roc_auc)
        for fam, g in d.groupby("metric_family"):
            i = g.best_dir.idxmax()
            rows.append({"protein": prot.replace("MutS_alpha", "MutSa"),
                         "family": fam, "auc": float(g.loc[i, "best_dir"]),
                         "metric": g.loc[i, "metric_col"], "n_family": len(g)})
    return pd.DataFrame(rows)


def panel_descriptor_families(out):
    t = descriptor_families()
    if t.empty:
        print("    SKIP descriptor families: no unified comparison files"); return
    fams = ["D1", "RMSD_all_instances", "RMSF", "DNA_distance"]
    label = {"D1": "D1 (pLDDT ratio)", "RMSD_all_instances": "RMSD",
             "RMSF": "RMSF", "DNA_distance": "protein-DNA distance"}
    cols = {"D1": DRV, "RMSD_all_instances": PAS, "RMSF": ACC, "DNA_distance": GRY}
    order = (t[t.family == "D1"].sort_values("auc", ascending=False).protein.tolist())
    order += [p for p in t.protein.unique() if p not in order]

    fig, ax = plt.subplots(figsize=(8.0, 3.5), constrained_layout=True)
    w = 0.2
    x = np.arange(len(order))
    for k, fam in enumerate(fams):
        sub = t[t.family == fam].set_index("protein").reindex(order)
        ax.bar(x + (k - 1.5) * w, sub.auc.values, w, label=label[fam],
               color=cols[fam], alpha=.9 if fam == "D1" else .75)
    ax.axhline(.5, color="#666", lw=.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8)
    ax.set_ylim(.5, max(.9, float(t.auc.max()) + .04))
    ax.set_ylabel("best within-protein ROC-AUC")
    wins = int((t.pivot_table(index="protein", columns="family", values="auc")
                .idxmax(axis=1) == "D1").sum())
    record("F2a", "D1_best_family_count", wins, f"of {t.protein.nunique()} proteins")
    for fam in fams:
        sub = t[t.family == fam]
        if len(sub):
            record("F2a", f"{fam}_median_best_auc", float(sub.auc.median()),
                   f"best of {int(sub.n_family.median())} metrics per protein")
    ax.set_title(f"D1 is the strongest descriptor family in {wins} of {t.protein.nunique()} "
                 "proteins,\nand it is the only family contributing a single metric", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right"); ax.grid(axis="y", alpha=.25)
    save(fig, out, "F2a_descriptor_families")


def panel_am_d1_scatter(out, labeled):
    """AlphaMissense vs D1 as within-gene percentile RANKS.

    Ranks, not z-scores. Two reasons. AlphaMissense saturates near 1.0, so a
    standardized axis piles thousands of variants onto a hard right edge and
    draws vertical stripes that are an artifact of the score's ceiling rather
    than of the data. And log10 D1 has long tails -- a handful of variants sit
    eight standard deviations out -- which compresses everything else into a
    band. Ranking inside the gene fixes both, and it is what the reported
    Spearman coefficient measures anyway, so the picture and the statistic
    finally describe the same thing.

    Ranking must be WITHIN gene: D1's magnitude tracks each gene's wild-type
    confidence (POLE ~0.001, FOXA1 ~0.067), so pooling raw values across genes
    manufactures correlation out of the between-gene offset.
    """
    d = labeled.dropna(subset=["afm_score", "d1_score", "label"]).copy()
    d = d[d.d1_score > 0]
    rhos = {}
    for g, sub in d.groupby("gene"):
        if len(sub) < 10:
            continue
        rhos[g] = stats.spearmanr(sub.afm_score, sub.d1_score).statistic
        d.loc[sub.index, "r_am"] = sub.afm_score.rank(pct=True)
        d.loc[sub.index, "r_d1"] = sub.d1_score.rank(pct=True)
    d = d.dropna(subset=["r_am", "r_d1"])

    # Fisher-z average of the per-gene rho, weighted by n-3, then back-transform.
    n = d.groupby("gene").size()
    ks = [g for g in rhos if g in n.index]
    wz = np.average([np.arctanh(np.clip(rhos[g], -.999, .999)) for g in ks],
                    weights=[max(n[g] - 3, 1) for g in ks])
    agg = float(np.tanh(wz))
    lo, hi = float(min(rhos.values())), float(max(rhos.values()))
    record("F2c", "within_gene_rho_fisher_z", agg, f"{len(ks)} genes")
    record("F2c", "within_gene_shared_variance_pct", 100 * agg ** 2)
    record("F2c", "rho_min_gene", lo, min(rhos, key=rhos.get))
    record("F2c", "rho_max_gene", hi, max(rhos, key=rhos.get))

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for lab, c, nm in ((0, PAS, "Passenger"), (1, DRV, "Driver")):
        sub = d[d.label == lab]
        ax.scatter(sub.r_am, sub.r_d1, s=4.5, alpha=.35, color=c, lw=0, label=nm)
    # Perfect-agreement diagonal: how far the cloud sits off it IS the story.
    ax.plot([0, 1], [0, 1], ls=":", lw=1.0, color="#444")
    # Binned median of D1 rank against AM rank -- the trend without a model.
    b = np.linspace(0, 1, 11)
    mid = (b[:-1] + b[1:]) / 2
    med = [d.r_d1[(d.r_am >= b[i]) & (d.r_am < b[i + 1])].median() for i in range(10)]
    ax.plot(mid, med, marker="o", ms=4, lw=1.6, color="#222", label="binned median")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("AlphaMissense, within-gene percentile")
    ax.set_ylabel("D1, within-gene percentile")
    ax.set_title(rf"Only partly redundant: within-gene $\rho$={agg:.2f}" + "\n"
                 f"({100*agg**2:.0f}% shared variance; per gene {lo:.2f}-{hi:.2f})",
                 fontsize=10)
    ax.legend(fontsize=7.5, markerscale=2.4, loc="lower right"); ax.grid(alpha=.25)
    save(fig, out, "F2c_am_d1_scatter")


def panel_added_information(out, _labeled):
    if not CANONICAL_LRT.is_file():
        raise FileNotFoundError(
            f"missing {CANONICAL_LRT}; run drift.canonical_afm first"
        )
    t = pd.read_csv(CANONICAL_LRT)
    t = t[t.test.eq("d1_beyond_afm")].rename(
        columns={"protein": "gene", "p_value": "p"}
    )[["gene", "chi2", "p", "n"]]
    t["gene"] = t.gene.replace({"MutS_alpha": "MutSalpha"})
    t = t.sort_values("chi2")
    for row in t.itertuples(index=False):
        record("F2e", f"{row.gene}_LR_chi2", float(row.chi2))
        record("F2e", f"{row.gene}_LR_p", float(row.p))
    primary = t[t.gene.isin(MACRO_GENES)]
    st, pc = stats.combine_pvalues(primary.p.values, method="fisher")
    record(
        "F2e", "fisher_combined_p", float(pc),
        f"{len(primary)} annotation genes; FOXA1 displayed but excluded as a positional control",
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    display_gene = t.gene.where(~t.gene.eq("FOXA1"), "FOXA1 (control)")
    colors = [GRY if g == "FOXA1" else "#1a7f37" if p < .05 else GRY
              for g, p in zip(t.gene, t.p)]
    ax.barh(display_gene, t.chi2, color=colors, alpha=.85)
    crit = stats.chi2.ppf(.95, 1)
    ax.axvline(crit, ls="--", color="#b00020", lw=1.2)
    ax.text(crit + .3, -.4, "p=0.05", fontsize=7, color="#b00020")
    for i, row in enumerate(t.itertuples(index=False)):
        ax.text(row.chi2 + .45, i, f"p={row.p:.1e}", fontsize=7, va="center", color="#333")
    ax.set_xlim(0, max(t.chi2.max() * 1.35, 8))
    ax.set_xlabel(r"likelihood-ratio $\chi^2$ (1 df): label ~ AM  vs  label ~ AM + D1")
    # Count derived, never spelled out: MutS_alpha became MSH2 + MSH6 on
    # 2026-09-07 and the hardcoded "six" went stale silently. Same rule as
    # canonical_afm.DESCRIPTIVE_LABEL_GROUPS.
    ax.set_title(f"D1 adds information beyond AlphaMissense\n"
                 f"{len(primary)} annotation datasets "
                 f"(Fisher combined p={pc:.1e})",
                 fontsize=10)
    ax.grid(axis="x", alpha=.25)
    save(fig, out, "F2e_added_information_LR")


def panel_am_reference(out):
    s = annotation_macro_series(); bm = benchmark_macro()
    if s is None or bm is None: return
    afm = float(bm[bm.model == "afm"].auc_mean.iloc[0])
    record("F2f", "alphamissense_macro_auc", afm)
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    ax.plot(s.index, s["mean"], marker="o", ms=5, color=DRV, lw=2, label="DRIFT D1 (structure only)")
    ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=DRV, alpha=.15)
    ax.axhline(afm, ls="--", lw=1.7, color=AMC, label=f"AlphaMissense  {afm:.3f}")
    ax.axhline(.5, ls=":", lw=1, color="#bbb")
    ax.text(88, .525, "AlphaMissense uses conservation,\npopulation frequency and training\non clinical labels. D1 uses none.",
            fontsize=7.5, color="#555", ha="right")
    ax.set_xlabel("Structures per variant (N)"); ax.set_ylabel("macro ROC-AUC")
    ax.set_ylim(.47, .95); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.25)
    ax.set_title("Structure alone recovers much of the signal", fontsize=10)
    save(fig, out, "F2f_alphamissense_reference")


def panel_pole_violin(out, labeled):
    for gene, name in (("POLE", "F1e_D1_violin_POLE"), ("VHL", "F1f_D1_violin_VHL"),
                       ("PTEN", "F1g_D1_violin_PTEN")):
        panel_class_violin(out, labeled, gene, name)


def _violin_pair(ax, data, pos, log, cols=(DRV, PAS)):
    """One driver/passenger violin pair with box and jittered points.

    Extracted so a two-descriptor panel draws each half with the SAME code as
    the single-descriptor panels: side-by-side violins are only a comparison of
    descriptors if the drawing is identical and the descriptor is the one thing
    that differs.

    `log` fits the KDE in log space and un-logs the body afterwards, because a
    violin fitted on raw values and then shown on a log axis is a different
    (wrong) density. The caller still owns set_yscale.

    The RNG is seeded here, per pair, so the jitter is reproducible and a
    re-run does not silently reshuffle the point cloud.
    """
    cols = list(cols)
    vp = ax.violinplot([np.log10(v) for v in data] if log else data,
                       positions=pos, widths=.85, showextrema=False)
    for b, c in zip(vp["bodies"], cols):
        if log:
            b.get_paths()[0].vertices[:, 1] = 10 ** b.get_paths()[0].vertices[:, 1]
        b.set_facecolor(c); b.set_alpha(.22); b.set_edgecolor(c)
    bp = ax.boxplot(data, positions=pos, widths=.15, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.3),
                    whiskerprops=dict(color="#555"), capprops=dict(color="#555"))
    for p_, c in zip(bp["boxes"], cols):
        p_.set_facecolor("white"); p_.set_edgecolor(c)
    rng = np.random.default_rng(0)
    for p_, v, c in zip(pos, data, cols):
        ax.scatter(p_ + (rng.random(len(v)) - .5) * .24, v, s=7, color=c, alpha=.4,
                   edgecolor="none", zorder=3)


def panel_class_violin(out, labeled, gene, name, column="d1_score", log=True,
                       ylabel="D1 (log scale)", key=None, tag="D1"):
    """Driver against passenger for one scored column, one gene.

    Parameterised over the column so the KS twin of a D1 panel is the same
    plot and not a lookalike -- reading them side by side is only a comparison
    of the two descriptors if everything except the descriptor is identical.

    `log` is not cosmetic. D1 is a ratio spanning a decade and is plotted on a
    log axis; the KS statistic is bounded on 0-1 and a log axis on it would
    stretch the low end for nothing. The violin is fitted in whichever space
    the axis uses, so this flag decides the KDE too, not just the ticks.
    """
    d = labeled.query("gene==@gene")
    drv = d[d.label == 1][column].values; pas = d[d.label == 0][column].values
    if len(drv) < 3 or len(pas) < 3: return
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    pos = [1, 2]
    _violin_pair(ax, [drv, pas], pos, log)
    if log:
        ax.set_yscale("log")
    y = np.r_[np.ones(len(drv)), np.zeros(len(pas))]
    auc = roc_auc_score(y, np.r_[drv, pas])
    # `tag`, not `column`: these keys are cited in the manuscript and in
    # article_panel_values.csv, so POLE_D1_AUC has to stay POLE_D1_AUC.
    record(key or name.split("_")[0], f"{gene}_{tag}_AUC", float(auc))
    ax.set_xticks(pos)
    ax.set_xticklabels([f"Driver\n(n={len(drv)})", f"Passenger\n(n={len(pas)})"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{gene}   AUC {auc:.3f}", fontsize=10); ax.grid(axis="y", alpha=.2)
    save(fig, out, name)


def panel_two_descriptor_violin(out, labeled, gene, stem, key,
                                left=("d1_score", "D1 (log scale)", "D1", True),
                                right=("plddt_ks_residue", "Mean per-residue KS", "Residue KS", False)):
    """Driver against passenger for TWO descriptors in one panel, twin y axes.

    WHY TWIN AXES AND NOT ONE. The two descriptors have no common unit: D1 is a
    ratio spanning a decade and reads on a log axis, the residue KS statistic is
    bounded on 0-1 and reads linearly. Forcing them onto one axis needs z-scores
    or ranks, which buys comparability by throwing away the values a reader
    wants ("D1 about 0.01", "KS about 0.07").

    Twin axes are a well-earned way to mislead, so the two things that make them
    safe here are deliberate:

      * THE HALVES DO NOT OVERLAP IN X. Each descriptor has its own pair of
        positions and its own axis, with a divider between them. Nothing invites
        the reader to compare a height on the left with a height on the right;
        the comparison the panel is for -- driver against passenger -- happens
        inside one half, against one axis.
      * NO HORIZONTAL GRID. A gridline would read as one value across both
        halves and it means two different numbers. The single-descriptor panels
        keep their y grid; this one cannot.

    Class colour still means class (driver red, passenger blue) on both sides,
    so the only thing position encodes is which descriptor.
    """
    d = labeled.query("gene==@gene")
    if d.label.nunique() < 2:
        print(f"  {stem}  SKIPPED: {gene} has one class only")
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax2 = ax.twinx()
    aucs, npts = {}, {}
    for (col, ylab, short, log), target, pos in ((left, ax, [1, 2]),
                                                 (right, ax2, [3.5, 4.5])):
        drv = d[d.label == 1][col].values; pas = d[d.label == 0][col].values
        _violin_pair(target, [drv, pas], pos, log)
        if log:
            target.set_yscale("log")
        else:
            target.set_ylim(0, max(drv.max(), pas.max()) * 1.10)
        target.set_ylabel(ylab)
        y = np.r_[np.ones(len(drv)), np.zeros(len(pas))]
        aucs[short] = roc_auc_score(y, np.r_[drv, pas])
        npts[short] = (len(drv), len(pas))
        record(key, f"{gene}_{short.replace(' ', '_')}_AUC", float(aucs[short]))
        target.text(np.mean(pos), .985, short, transform=target.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, fontweight="bold", color="0.35")
    ax.axvline(2.75, color="0.75", lw=.9, ls=(0, (4, 3)))
    ax.set_xlim(.45, 5.05)
    ax.set_xticks([1, 2, 3.5, 4.5])
    nd, np_ = npts[left[2]]
    ax.set_xticklabels([f"Driver\n(n={nd})", f"Passenger\n(n={np_})"] * 2)
    ax.set_title(f"{gene}   "
                 + "   ".join(f"{k} AUC {v:.3f}" for k, v in aucs.items()),
                 fontsize=10)
    ax.grid(False); ax2.grid(False)
    save(fig, out, stem)


def panel_roc(out, labeled, gene, stem, key, scores=("afm", "d1"), title=None):
    """ROC for one gene, at the ensemble size the cache was loaded at.

    `scores` selects which curves are drawn, and that is a per-gene editorial
    decision rather than a stylistic one. POLE carries both, because the
    comparison against AlphaMissense IS the point of that panel. PTEN carries
    D1 alone on the manuscript's request: on that cohort AlphaMissense reaches
    0.993 against D1's 0.848, so drawing both would turn the panel into a
    statement about AlphaMissense. The AUCs are recorded either way, so the
    number is in article_panel_values.csv whichever curves are plotted.

    CLASS BALANCE IS NOT UNIFORM ACROSS GENES and it shows in the curve: PTEN
    has 9 passengers against 248 drivers, so its FPR axis advances in steps of
    0.11 and both AUCs carry a wide interval. Recorded as *_n_driver and
    *_n_passenger next to the AUCs.
    """
    d = labeled.query("gene == @gene")
    if d.label.nunique() < 2:
        print(f"  {stem}  SKIPPED: {gene} has one class only")
        return
    series = {"afm": (d.afm_score, AMC, "AlphaMissense"),
              "d1": (d.d1_score, DRV, "DRIFT D1"),
              "ks": (d.plddt_ks_residue, ACC, "DRIFT residue KS")}
    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    for which in scores:
        sc, c, n = series[which]
        fpr, tpr, _ = roc_curve(d.label, sc)
        a = roc_auc_score(d.label, sc)
        ax.plot(fpr, tpr, color=c, lw=1.8, label=f"{n}  {a:.3f}")
        record(key, f"{gene}_{ {'afm': 'AM', 'd1': 'D1', 'ks': 'KS'}[which] }_AUC",
               float(a))
    record(key, f"{gene}_n_driver", int(d.label.sum()))
    record(key, f"{gene}_n_passenger", int((1 - d.label).sum()))
    ax.plot([0, 1], [0, 1], ls="--", lw=.8, color="#aaa")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title or f"ROC, {gene} (case study)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.2)
    save(fig, out, stem)


def panel_per_gene_models(out, labeled):
    """Per gene: D1 alone, AlphaMissense alone, best fixed fusion, nested fusion.

    The panel reports the current strict-cohort fixed and nested fusion outputs
    directly, alongside the two single-score baselines.
    """
    if not BENCHMARK_SUMMARY.exists():
        return
    b = pd.read_csv(BENCHMARK_SUMMARY)
    g = b[b.get("scope_type") == "gene"]
    if g.empty:
        return
    afm = g[g.model == "afm"].set_index("scope").auc_mean
    nest = g[g.model == "nested_selection"].set_index("scope").auc_mean
    best = (g[g.model == "logreg"].sort_values("delta_mean", ascending=False)
            .groupby("scope").head(1).set_index("scope").auc_mean)
    genes = [x for x in GENES if x in afm.index]
    d1 = {}
    for k in genes:
        d = labeled.query("gene==@k")
        d1[k] = roc_auc_score(d.label, d.d1_score) if d.label.nunique() > 1 else np.nan
    order = sorted(genes, key=lambda k: d1[k] - float(afm[k]), reverse=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    x = np.arange(len(order)); w = .2
    series = [("DRIFT D1 alone", [d1[k] for k in order], DRV),
              ("AlphaMissense alone", [float(afm[k]) for k in order], AMC),
              ("AM + D1, best fixed\n(selected on held-out genes)",
               [float(best.get(k, np.nan)) for k in order], "#cccccc"),
              ("AM + D1, nested\n(honest)", [float(nest.get(k, np.nan)) for k in order], ACC)]
    for i, (name, vals, c) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, vals, w, color=c, alpha=.9, label=name)
    for k, v in zip(order, [d1[k] for k in order]):
        record("F2g", f"{k}_D1_minus_AM", float(v - float(afm[k])))
        record("F2g", f"{k}_nested_minus_AM", float(float(nest.get(k, np.nan)) - float(afm[k])))
    win = [k for k in order if d1[k] > float(afm[k])]
    record("F2g", "genes_where_D1_beats_AM", ", ".join(win) or "none")
    ax.set_xticks(x); ax.set_xticklabels([k.replace("MutSalpha", "MutSa") for k in order])
    ax.set_ylim(.5, 1.03); ax.set_ylabel("within-gene ROC-AUC")
    ax.legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=.95)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("Fixed and nested fusion performance by held-out dataset", fontsize=10)
    save(fig, out, "F2g_per_gene_models")


def panel_fusion_supp(out):
    bm = benchmark_macro()
    s = annotation_macro_series()
    if bm is None or s is None: return
    afm = float(bm[bm.model == "afm"].auc_mean.iloc[0])
    ns = bm[bm.model == "nested_selection"]
    if ns.empty: return
    fixed, nest = canonical_am_d1_result(), ns.iloc[0]
    record("S1a", "fixed_AM_D1_delta_vs_AM", float(fixed.observed_delta),
           "prespecified C=1 logistic comparator")
    record("S1a", "nested_delta_vs_AM", float(nest.delta_mean))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    names = ["D1\nalone", "AlphaMissense\nalone",
             "AM + D1\nfixed C=1\n(prespecified\ncomparator)",
             "AM + D1\n72-candidate grid\n(nested selection)"]
    vals = [float(s.loc[90, "mean"]), afm, float(fixed.observed_auc), float(nest.auc_mean)]
    for i, (v, c) in enumerate(zip(vals, [DRV, AMC, "#cccccc", ACC])):
        ax.bar(i, v, color=c, alpha=.88, width=.55)
        ax.text(i, v + .003, f"{v:.4f}", ha="center", fontsize=9, weight="bold")
    intervals = [
        (2, float(fixed.observed_auc), float(fixed.auc_ci_low),
         float(fixed.auc_ci_high), float(fixed.observed_delta)),
        (3, float(nest.auc_mean), float(nest.auc_ci_low),
         float(nest.auc_ci_high), float(nest.delta_mean)),
    ]
    # Limits follow the values: the fixed .72-.92 window was cut for the mixed
    # macro and clipped the annotation cohort, whose AUCs sit above .92.
    bottom = np.floor((min(vals + [lo for *_, lo, _, _ in intervals]) - .03) * 100) / 100
    top = max(vals + [hi for *_, hi, _ in intervals]) + .02
    for i, value, low, high, delta in intervals:
        ax.errorbar(i, value, yerr=[[value - low], [high - value]],
                    color="black", capsize=4, lw=1.1)
        ax.text(i, bottom + .007, f"{delta:+.4f}", ha="center", fontsize=9.5, weight="bold",
                color="#1a7f37" if delta > 0 else "#b00020")
    ax.axhline(afm, ls="--", lw=1.2, color=AMC, alpha=.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(bottom, top); ax.set_ylabel("macro-LOGO ROC-AUC")
    ax.set_title(f"Combining DRIFT with AlphaMissense across "
                 f"{len(MACRO_GENES)} annotation datasets", fontsize=10)
    ax.grid(axis="y", alpha=.25)
    save(fig, out, "S1a_fusion_fixed_vs_nested")


def panel_fusion_per_gene_supp(out):
    if not BENCHMARK_SUMMARY.exists(): return
    d = pd.read_csv(BENCHMARK_SUMMARY)
    g = d[(d.get("scope_type") == "gene") & (d.model == "nested_selection")]
    if g.empty: return
    g = g[["scope", "delta_mean"]].sort_values("delta_mean")
    with_all = float(g.delta_mean.mean())
    record("S1b", "macro_delta_all_genes", with_all)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    cols = ["#b00020" if v < -.001 else ("#1a7f37" if v > .001 else GRY) for v in g.delta_mean]
    ax.barh(g.scope, g.delta_mean, color=cols, alpha=.85)
    ax.axvline(0, color="#333", lw=1)
    worst = g.iloc[0].scope
    excl = float(g[g.scope != worst].delta_mean.mean())
    record("S1b", f"macro_delta_excluding_{worst}", excl)
    ax.set_xlabel("nested selection minus AlphaMissense, per held-out gene")
    ax.set_title(f"{worst} accounts for the whole deficit\n"
                 f"macro {with_all:+.4f} with it, {excl:+.4f} without", fontsize=10)
    ax.grid(axis="x", alpha=.25)
    save(fig, out, "S1b_fusion_per_gene")


def panel_nsweep_metrics(out, gene, stem, key,
                         columns=(("d1", "DRIFT D1", DRV),
                                  ("ks_raw", "DRIFT residue KS", ACC)),
                         anchor=None, row4_style=False, auc_table=None):
    """ROC-AUC against ensemble size for one gene, several descriptors, one axes.

    ``auc_table`` (2026-10-17): plot precomputed per-seed AUCs instead of computing
    them from CANONICAL_SIZE_SCORES. It takes Figure 2's
    experiments/figure2/figures/panel_abcd_per_gene_auc.csv (columns N, seed, gene,
    descriptor, auc), so Figure 1h and the same gene's line in Figure 2a are the
    same numbers. Only the first entry of ``columns`` is drawn; its N=1 point is the
    table's "KS at N=1" row.

    The canonical table's ks_raw is the mean per-residue two-sample KS,
    the same estimator as plddt_ks_residue in the ROC and violin caches.
    Sampling differs explicitly: this curve summarizes ten canonical seeds;
    the ROC and violin panels use the single cache seed selected by the CLI.
    They need not have identical N=90 AUCs.

    Two descriptors on ONE axes because both are ROC-AUC against N and so are
    directly comparable, where their raw values are not: D1 is a ratio spanning
    a decade and the KS statistic is bounded on 0-1, which is why they still
    need separate violin panels.

    Error bars are the SD across the table's 10 resampling seeds, matching
    F2a/F2b. ks_centred is deliberately not offered here: it is 0.500 at N=1 by
    construction, so a panel about ensemble size would be reporting an artifact
    of the centring at its most important point.

    ``anchor`` optionally adds a separate reference descriptor at N=1.
    KS itself is defined at N=1 (the residue mismatch fraction) and is drawn
    as part of its continuous series, without a separate reference marker.

    ``row4_style`` dresses the panel like its neighbour g in Figure1_resKS
    (make_panels_gh.panel_g): FIG1_ROW4_RC, the same 3.2 in height, open
    top/right spines, no grid, and no chance line. The height matters more
    than it looks: the assembler scales a row to ONE height, so a panel
    authored at 4.0 in printed its type at 80% of g's. The chance line goes
    because the curve never comes near 0.5; it only pulled the axis down.
    """
    if auc_table is not None:
        t = pd.read_csv(auc_table)
        t = t[t.gene.eq(gene) & t.descriptor.isin(["KS-score", "KS at N=1"])]
        if t.empty:
            print(f"  {stem}  SKIPPED: {gene} absent from {Path(auc_table).name}")
            return
        col, label, colour = columns[0]
        with plt.rc_context(FIG1_ROW4_RC if row4_style else None):
            fig, ax = plt.subplots(figsize=(3.7, 3.2) if row4_style else (4.8, 4.0),
                                   constrained_layout=True)
            bars = (dict(ms=3.4, lw=1.4, capsize=1.8, capthick=.7, elinewidth=.7)
                    if row4_style else dict(ms=4, lw=1.5, capsize=2.5))
            g = t.groupby("N").auc.agg(["mean", "std"]).sort_index()
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o",
                        color=colour, label=label, **bars)
            for n, a in g["mean"].items():
                record(key, f"{gene}_{col}_AUC_N{int(n)}", float(a),
                       f"from {Path(auc_table).name} (Figure 2a source)")
            ax.set_xlabel("Predictions per ensemble (N)")
            ax.set_ylabel("ROC-AUC (driver-high)")
            ax.set_title(f"{gene} ensemble-size sensitivity")
            ax.legend(loc="lower right")
            if row4_style:
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
            else:
                ax.axhline(.5, ls="--", lw=.8, color="#aaa"); ax.grid(alpha=.2)
            save(fig, out, stem)
        return
    if not CANONICAL_SIZE_SCORES.is_file():
        print(f"  {stem}  SKIPPED: missing {CANONICAL_SIZE_SCORES.name}")
        return
    want = ["protein", "label", "seed_index", "requested_size"] + [
        c for c, _, _ in (tuple(columns) + ((anchor,) if anchor else ()))]
    want = list(dict.fromkeys(want))
    d = pd.read_csv(CANONICAL_SIZE_SCORES, usecols=want)
    d = d[d.protein.eq(gene.replace("MutSalpha", "MutS_alpha"))]
    if d.empty:
        print(f"  {stem}  SKIPPED: {gene} absent from the canonical table")
        return
    # rc_context(None) is a no-op, so the default panels are drawn as before.
    with plt.rc_context(FIG1_ROW4_RC if row4_style else None):
        fig, ax = plt.subplots(figsize=(3.7, 3.2) if row4_style else (4.8, 4.0),
                               constrained_layout=True)
        bars = (dict(ms=3.4, lw=1.4, capsize=1.8, capthick=.7, elinewidth=.7)
                if row4_style else dict(ms=4, lw=1.5, capsize=2.5))
        first = {}
        for col, label, colour in columns:
            rows = []
            for (size, seed), blk in d.groupby(["requested_size", "seed_index"]):
                if blk.label.nunique() < 2 or blk[col].isna().any():
                    continue
                rows.append((int(size), roc_auc_score(blk.label, blk[col])))
            if not rows:
                continue
            g = pd.DataFrame(rows, columns=["size", "auc"]).groupby("size").auc.agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o",
                        color=colour, label=label, **bars)
            for n, a in g["mean"].items():
                record(key, f"{gene}_{col}_AUC_N{int(n)}", float(a))
            first[col] = (int(g.index.min()), float(g["mean"].iloc[0]))

        if anchor and first:
            col, label, colour = anchor
            vals = [roc_auc_score(blk.label, blk[col])
                    for _, blk in d[d.requested_size.eq(1)].groupby("seed_index")
                    if blk.label.nunique() > 1 and not blk[col].isna().any()]
            if vals:
                lo_n, lo_auc = min(first.values(), key=lambda t: t[0])
                mean = float(np.mean(vals))
                ax.plot([1, lo_n], [mean, lo_auc], ls=(0, (2.6, 1.9)), lw=1.1,
                        color=colour, alpha=.75, zorder=2)
                ax.errorbar([1], [mean], yerr=[float(np.std(vals, ddof=1))],
                            marker="s", ms=5, mfc="white", mew=1.5, color=colour,
                            capsize=2.5, elinewidth=1.0, ls="none", zorder=4,
                            label=label)
                record(key, f"{gene}_{col}_AUC_N1_anchor", mean,
                       "single-structure reference; KS is undefined at N=1")
        ax.set_xlabel("Predictions per ensemble (N)")
        ax.set_ylabel("ROC-AUC (driver-high)")
        if row4_style:
            ax.set_title(f"{gene} ensemble-size sensitivity")
            ax.legend(loc="lower right")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        else:
            ax.axhline(.5, ls="--", lw=.8, color="#aaa")
            ax.set_title(f"{gene} ensemble-size sensitivity", fontsize=10)
            ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.2)
        save(fig, out, stem)


# ── driver ────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=RESULTS / tagged("article_figures"))
    ap.add_argument(
        "--copy-to", type=Path, default=None,
        help=("manuscript repo root; the PDF panels are copied to "
              "<root>/figures/panels/ so \\includegraphics can reach them. "
              "Defaults to the sibling SEPA-Structural-Ensemble-Perturbation-Analysis "
              "checkout when it exists."))
    ap.add_argument("--no-copy", action="store_true",
                    help="write local panels only; do not copy into the manuscript repository")
    ap.add_argument("--fig1-h-auc-table", type=Path, default=None,
                    help=("with --fig1-h: plot per-seed AUCs from this table (Figure 2's "
                          "panel_abcd_per_gene_auc.csv) instead of the canonical size table"))
    ap.add_argument("--fig1-h", metavar="GENE", default=None,
                    help=("draw only Figure 1h (the KS-score N-sweep) for GENE into --out "
                          "and stop; nothing is copied"))
    args = ap.parse_args(argv)
    if args.fig1_h:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        VALUES.clear()
        panel_nsweep_metrics(out, args.fig1_h, f"{args.fig1_h}_Nsweep_resKS", "F1h_resks",
                             columns=(("ks_raw", "KS-score", ACC),), row4_style=True,
                             auc_table=args.fig1_h_auc_table)
        pd.DataFrame(VALUES).to_csv(out / f"{args.fig1_h}_Nsweep_resKS_values.csv", index=False)
        return 0
    if args.no_copy and args.copy_to is not None:
        ap.error("--no-copy and --copy-to are mutually exclusive")

    if args.copy_to is None and not args.no_copy:
        # Worktree-safe; see paths.manuscript_root. The old REPO.parent sibling
        # lookup silently resolved to None from a worktree and skipped the copy.
        found = manuscript_root(must_exist=True)
        args.copy_to = found if (found and (found / "figures").is_dir()) else None
        if args.copy_to is None:
            print("[article] manuscript repo not found -> panels not copied",
                  file=sys.stderr)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    VALUES.clear()
    labeled = load_labeled(args.size, args.seed)
    if "plddt_ks_residue" not in labeled or not np.isfinite(labeled.plddt_ks_residue).all():
        raise ValueError("article panels require finite plddt_ks_residue scores; rebuild the cache")
    print(f"[article] {len(labeled):,} variants, {labeled.gene.nunique()} genes "
          f"(N={args.size}, seed={args.seed})\n[article] writing panels to {out}")

    panel_pole_violin(out, labeled)
    panel_roc(out, labeled, "POLE", "F1h_ROC_POLE", "F1h")
    # Letterless stem: this one is Figure 1g today and the letters in this
    # figure have moved three times in two days.
    # PTEN's KS panels for Figure1_KS: the violin twin of F1g, the ROC with
    # both descriptors, and the sweep with both. See panel_nsweep_metrics for
    # why the two descriptors share the ROC and sweep axes but not the violin.
    panel_class_violin(out, labeled, "PTEN", "PTEN_KS_violin", column="plddt_ks_residue",
                       log=False,
                       ylabel="Mean per-residue KS (WT vs mutant)",
                       key="F1g_ks", tag="KS")
    panel_two_descriptor_violin(out, labeled, "PTEN", "PTEN_violin_D1_KS", "F1f_ks")
    panel_roc(out, labeled, "PTEN", "PTEN_ROC_D1_KS", "F1g_roc_ks",
              scores=("d1", "ks"), title="ROC, PTEN")
    panel_nsweep_metrics(out, "PTEN", "PTEN_Nsweep_D1_KS", "F1h_ks")
    panel_nsweep_metrics(out, "PTEN", "PTEN_Nsweep_D1", "F1h_d1",
                         columns=(("d1", "DRIFT D1", DRV),))
    # Figure1_resKS panel h: the same residue KS at every depth, including N=1.
    panel_nsweep_metrics(out, "PTEN", "PTEN_Nsweep_resKS", "F1h_resks",
                         columns=(("ks_raw", "KS-score", ACC),), row4_style=True)
    panel_roc(out, labeled, "PTEN", "PTEN_ROC_D1", "F1g_roc",
              scores=("d1",), title="ROC, PTEN")
    panel_ensemble_size(out)
    panel_per_gene_size(out)
    panel_sampling_policies(out)
    panel_continuum(out, labeled)
    panel_descriptor_families(out)
    panel_am_d1_scatter(out, labeled)
    panel_added_information(out, labeled)
    panel_am_reference(out)
    panel_per_gene_models(out, labeled)
    panel_fusion_supp(out)
    panel_fusion_per_gene_supp(out)

    vals = pd.DataFrame(VALUES)
    vals.to_csv(out / "article_panel_values.csv", index=False)
    (out / "article_panel_provenance.json").write_text(json.dumps({
        "cache_size": args.size, "cache_seed": args.seed,
        "ks_estimator": "mean per-residue two-sample KS",
        "ks_cache_column": "plddt_ks_residue",
        "ks_sweep_column": "ks_raw",
        "ks_min_samples": 1,
        "n1_policy": "KS defined for N>=1; at N=1 it is the residue discordance rate",
        "ks_sweep_source": str(CANONICAL_SIZE_SCORES.relative_to(REPO)),
        "sampling_note": "ROC/violin: one cache seed; depth sweep: mean and SD over canonical seeds",

        "n_variants": int(len(labeled)), "n_positive": int(labeled.label.sum()),
        "genes": sorted(labeled.gene.unique().tolist()),
    }, indent=2))
    print(f"  article_panel_values.csv  ({len(vals)} quantities)")

    if args.copy_to:
        import shutil
        dest = Path(args.copy_to) / "figures" / "panels"
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for pdf in sorted(out.glob("*.pdf")):
            shutil.copy2(pdf, dest / pdf.name)
            n += 1
        shutil.copy2(out / "article_panel_values.csv", dest / "article_panel_values.csv")
        print(f"[article] copied {n} vector panels -> {dest}")
        print("[article] LaTeX:  \\includegraphics[width=.32\\linewidth]{panels/F2a_ensemble_size_macro}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
