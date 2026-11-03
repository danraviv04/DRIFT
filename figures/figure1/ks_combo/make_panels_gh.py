#!/usr/bin/env python
"""PTEN panels: mean per-seed ROC, mean-score violin, historical depth panel.

Use --summary-only to refresh the ROC and violin from the current ten-seed
esfeat6 caches without rebuilding the separately maintained depth panel.
"""

from __future__ import annotations

import argparse
import sys
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from figures.figure1.seed_panel_scores import load_scores, mean_scores, save_provenance

CACHE = REPO / "drift" / "cache"
OUT = HERE / "results"
GENE = "PTEN"
SIZES = (1, 5, 10, 20, 40, 60, 90)

# Figure 1's own palette. The combination takes a third hue plus a dashed line
# and square markers, so identity never rests on colour alone.
D1, KS = "#d1495b", "#6a4c93"
COMBO = "#2e86ab"
GRID, RULE, INK, MUTED = "#d9d9d9", "#333333", "#000000", "#555555"

plt.rcParams.update({
    # Sized for the PAGE, not the file: make_figures places this row at about
    # 0.97, so 10 pt here lands at 9.7 pt, level with panels c and d. At the
    # old 8/7 the row printed 6.8-7.7 pt and was the smallest type in Figure 1.
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": RULE, "axes.linewidth": .8,
    "axes.labelcolor": INK, "axes.labelsize": 10,
    "axes.titlesize": 10.5, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "grid.color": GRID, "grid.linewidth": .5, "grid.linestyle": "-",
    "legend.frameon": True, "legend.fontsize": 8.5,
    "legend.edgecolor": "#bfbfbf", "legend.framealpha": 1.0,
    "legend.borderpad": .4, "legend.handlelength": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# residue KS only, from 2026-09-15. D1 and the D1+KS rank combination were
# dropped from both article figures. The combination never earned its place --
# below N~20 it was clearly WORSE than KS alone (0.761 vs 0.805 at N=1), and it
# ended 0.884 against 0.882, inside the seed spread -- and D1 alone loses to
# resKS on the PTEN panels and on the Figure 2 macro. Keeping them cost a
# descriptor and a series without buying a result.
#
# "pLDDT KS" is the name that let profileKS and residueKS be confused for each
# other in the first place. The series is residueKS, so the label says so.
SERIES = (("KS", "DRIFT KS", KS, "o", "-"),)


def load() -> pd.DataFrame:
    """Per-(size, seed) percentile ranks and labels for one gene."""
    rows = []
    # esfeat5, not esfeat3: the residue-KS column only exists in the five-feature
    # caches. These are also the split-cohort fingerprint, which does not affect
    # PTEN -- the MutS_alpha split leaves every other dataset untouched.
    for f in glob.glob(str(CACHE / "labeled_size*_seed*_*esfeat5.csv")):
        name = os.path.basename(f)
        if "_nested-" in name:            # policy-tagged caches are a different draw
            continue
        m = re.search(r"size(\d+)_seed(\d+)", name)
        if not m:
            continue
        size, seed = int(m.group(1)), int(m.group(2))
        if size not in SIZES:
            continue
        d = pd.read_csv(f, usecols=["gene", "mutation", "label", "d1_score",
                                    "plddt_ks_residue"])
        g = d[d.gene.eq(GENE)]
        if g.empty:
            continue
        n = len(g)
        rows.append(pd.DataFrame({
            "size": size, "seed": seed, "mutation": g.mutation.to_numpy(),
            "label": g.label.to_numpy(int),
            "D1": rankdata(g.d1_score.to_numpy(float)) / (n + 1.0),
            "KS": rankdata(g.plddt_ks_residue.to_numpy(float)) / (n + 1.0),
        }))
    if not rows:
        raise SystemExit(f"no cache files with {GENE} found under {CACHE}")
    out = pd.concat(rows, ignore_index=True)
    out["D1+KS"] = (out["D1"] + out["KS"]) / 2.0
    return out


def load_raw(size=90, seed=42):
    """Raw descriptor values for one (size, seed) -- what the violin shows.

    load() returns percentile RANKS, which is what the rank combination and the
    two AUC panels need. A violin has to show the descriptors on their own
    scales, so it reads the cache directly.
    """
    pat = str(CACHE / f"labeled_size{size}_seed{seed}_*esfeat5.csv")
    files = [f for f in sorted(glob.glob(pat)) if "_nested-" not in os.path.basename(f)]
    if not files:
        raise SystemExit(f"no cache matching {pat}")
    d = pd.read_csv(files[0], usecols=["gene", "label", "d1_score",
                                       "plddt_ks_residue"])
    g = d[d.gene.eq(GENE)].copy()
    if g.empty:
        raise SystemExit(f"{GENE} absent from {files[0]}")
    return g


def set_gene(gene, out):
    """Retarget the module at another case-study gene and output folder."""
    global GENE, OUT
    GENE, OUT = gene, Path(out)
    OUT.mkdir(parents=True, exist_ok=True)


def letter(fig, ax, ch, dx=.075, dy=.045):
    """Bold lowercase panel letter outside the axes, as in Figure 1."""
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, ch, fontsize=13, fontweight="bold",
             ha="left", va="bottom", color=INK)


def roc(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = np.asarray(y)[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]


def panel_g(df, size):
    """Mean ROC and pointwise sample SD over the ten subsampling seeds.

    ``size`` is the ensemble size the scores were drawn at (load_scores'
    provenance), printed beside the AUC so the panel says which N it is.
    """
    grid = np.linspace(0, 1, 2001)
    curves, rows = [], []
    for seed, block in df.groupby("seed", sort=True):
        fpr, tpr, _ = roc_curve(block.label, block.plddt_ks_residue)
        curves.append(np.interp(grid, fpr, tpr))
        rows.append({"seed": seed, "auc": roc_auc_score(block.label, block.plddt_ks_residue)})
    curves = np.asarray(curves)
    mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
    # Interpolation retains the upper endpoint at FPR=0. Prepend the origin
    # at the same FPR to preserve the vertical segment without changing area.
    grid = np.r_[0.0, grid]
    mean = np.r_[0.0, mean]
    sd = np.r_[0.0, sd]
    aucs = pd.DataFrame(rows)
    summary = pd.DataFrame({"fpr": grid, "mean_tpr": mean, "sd_tpr": sd})
    summary.to_csv(OUT / f"Figure1g_{GENE}_mean_roc.csv", index=False)
    aucs.to_csv(OUT / f"Figure1g_{GENE}_seed_auc.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), lw=.7, color="#9a9a9a")
    ax.fill_between(grid, np.maximum(0, mean-sd), np.minimum(1, mean+sd),
                    color=KS, alpha=.2, linewidth=0)
    ax.plot(grid, mean, color=KS, lw=1.5,
            label=f"AUC {aucs.auc.mean():.3f} ± {aucs.auc.std():.3f}, N = {size}")
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02),
           xlabel="False positive rate", ylabel="True positive rate",
           title=f"Mean ROC, {GENE}")
    ax.legend(loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    print(f"ROC AUC: {aucs.auc.mean():.6f} ± {aucs.auc.std():.6f}")
    return fig, mean_scores(df)


def panel_h(df):
    """AUC against ensemble size, mean +/- sd over the ten subsampling seeds."""
    rows = []
    for (size, seed), block in df.groupby(["size", "seed"], sort=True):
        for key, *_ in SERIES:
            rows.append({"size": size, "seed": seed, "series": key,
                         "auc": roc_auc_score(block.label, block[key])})
        # keep only the three series
    per = pd.DataFrame(rows)
    agg = per.groupby(["series", "size"]).auc.agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    # No chance line: the curve never approaches 0.5 (it starts at 0.731 and is
    # flat from N=10), so the rule only pulled the y-axis down and added an
    # element the reader has to identify before dismissing.
    for key, label, colour, marker, ls in SERIES:
        r = agg[agg.series.eq(key)].sort_values("size")
        ax.errorbar(r["size"], r["mean"], yerr=r["std"].fillna(0),
                    marker=marker, ms=3.4, lw=1.4, ls=ls, color=colour,
                    capsize=1.8, capthick=.7, elinewidth=.7, zorder=3,
                    label=label)
    ax.set_xlabel("Structures per variant (N)")
    ax.set_ylabel("ROC-AUC (driver-high)")
    ax.set_title(f"{GENE} ensemble-size sensitivity")
    # Floor at .63, not .70: N=1 is 0.731 +/- 0.089, so a .70 floor cuts the
    # bottom off the one error bar in the panel that actually has width.
    ax.set_ylim(.63, .89)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    # No baked panel letter: make_figures.py draws the overlay letter for every
    # placed panel and only suppresses its own for "_combo" stems, so a letter
    # baked in here would print on top of the overlay one.
    return fig, agg


def panel_resks_violin(frame, out):
    """Residue KS for one gene, drivers against passengers.

    Was panel_two_descriptor_violin: D1 on a log left axis, residue KS on a
    linear right axis, with a divider between them, because the two share no
    unit. With D1 gone that whole apparatus goes -- one bounded descriptor on
    [0, 1] needs one axis, and the panel gets its horizontal grid back, which
    the twin-axis version had to drop (a gridline belonged to only one half).
    """
    DRV, PAS = "#d1495b", "#2e86ab"
    col = "plddt_ks_residue"
    # 3.2 tall, matching panels g and h: matched_row scales the row to ONE
    # height, so a panel authored taller is scaled down further than its
    # neighbours and prints smaller type for no reason anyone can see.
    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    rng = np.random.default_rng(0)
    drv = frame.loc[frame.label == 1, col].to_numpy(float)
    pas = frame.loc[frame.label == 0, col].to_numpy(float)
    data, pos = [drv, pas], [1, 2]
    vp = ax.violinplot(data, positions=pos, widths=.8, showextrema=False)
    for b, c in zip(vp["bodies"], (DRV, PAS)):
        b.set_facecolor(c); b.set_alpha(.22); b.set_edgecolor(c)
    bp = ax.boxplot(data, positions=pos, widths=.14, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.3),
                    whiskerprops=dict(color="#555"), capprops=dict(color="#555"))
    for box, c in zip(bp["boxes"], (DRV, PAS)):
        box.set_facecolor("white"); box.set_edgecolor(c)
    for p_, v, c in zip(pos, data, (DRV, PAS)):
        ax.scatter(p_ + (rng.random(len(v)) - .5) * .22, v, s=15, color=c,
                   alpha=.42, edgecolor="none", zorder=3)
    y = np.r_[np.ones(len(drv)), np.zeros(len(pas))]
    auc = roc_auc_score(y, np.r_[drv, pas])
    ax.set_ylabel("Mean DRIFT KS")
    ax.set_xlim(.4, 2.6)
    ax.set_xticks(pos)
    # Set explicitly, so it does not stay at the old 8 while the rcParams tick
    # size moved: these two labels carry the class sizes and are read as often
    # as the axis is.
    ax.set_xticklabels([f"Driver\n(n={len(drv)})",
                        f"Passenger\n(n={len(pas)})"], fontsize=9.5)
    ax.set_title(f"{GENE}    mean-score AUC {auc:.3f}", fontsize=11, pad=8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"Figure1f_violin_{GENE}_resKS.{ext}", dpi=400,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"panel f: DRIFT KS AUC {auc:.3f}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--gene", default=GENE)
    parser.add_argument("--fingerprint", default=None,
                        help="cache fingerprint for load_scores (d61d29328362 = v6_ca)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    set_gene(args.gene, args.out)
    # Only GENE is drawn, so only GENE is checked against the manifest.
    scores, provenance = (load_scores(fingerprint=args.fingerprint, genes=[GENE])
                          if args.fingerprint else load_scores(genes=[GENE]))
    df = scores[scores.gene.eq(GENE)].copy()
    save_provenance(OUT / "summary_panel_provenance.json", provenance)
    fig, mean = panel_g(df, provenance["size"])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure1g_ROC_{GENE}_resKS.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    mean.to_csv(OUT / f"Figure1g_{GENE}_values.csv", index=False)

    panel_resks_violin(mean, OUT)
    if args.summary_only:
        return 0

    fig, agg = panel_h(load())
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure1h_Nsweep_{GENE}_resKS.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    agg.to_csv(OUT / f"Figure1h_{GENE}_values.csv", index=False)


    print("\npanel h, mean +/- sd over seeds:")
    print(agg.pivot(index="size", columns="series", values="mean").round(3).to_string())
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
