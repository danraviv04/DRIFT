#!/usr/bin/env python
"""Fig. 1g: the mean per-seed KS-score ROC for one gene (panel_g).

Called by figures/figure1/layout/rerender_efgh.py, which runs it in its own
process because the rcParams below are global.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from figures.figure1.seed_panel_scores import mean_scores

OUT = HERE / "results"
GENE = "PTEN"

# Figure 1's own palette.
KS = "#6a4c93"
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


def set_gene(gene, out):
    """Retarget the module at another case-study gene and output folder."""
    global GENE, OUT
    GENE, OUT = gene, Path(out)
    OUT.mkdir(parents=True, exist_ok=True)


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
