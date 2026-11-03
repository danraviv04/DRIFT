#!/usr/bin/env python
"""Supplementary Figs. S1 and S2, drawn whole instead of stacked from panel PDFs.

The old Fig. S1 was four panel PDFs made by three scripts at their own sizes and
stacked by tools/make_figures.py, so its fonts, widths and spacing did not match
and its caption ran off the page. It is split by question and redrawn at the
SI text width (168 mm), with figlib's style:

  FigureS_coordinate_descriptors.pdf  (S1)  a shift vs dRMSF in POLE P286R,
                                            b driver/passenger AUC per descriptor,
                                            c RMSD and dRMSF by class per gene
  FigureS_rigid_core.pdf              (S2)  WT RMSF against WT pLDDT, core coloured
                                            (the old panel d)

No number is recomputed from coordinates: every plotted value is read from the
outputs that made the old panels (inputs/, copied from the POLE 25-seed side
outputs and si_figures/results), and the script asserts that they reproduce the
numbers printed in the SI (Table S2, the old panel annotations).

    python figures/si/S1_S2/rebuild.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "figures/si/lib"))
from figlib import (ACC, DRV, GENE_COLOR, GRY, MM, PAS, class_violin,  # noqa: E402
                    pretty, save)

IN = HERE / "inputs"
OUT = HERE / "results"
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
W = 168 * MM                      # SI text width: letter paper, 24 mm margins
KS, POS, RMSD, DRMSF = "pLDDT KS-score", "Positional KS", "RMSD", "ΔRMSF"
SITE = 286
# as tools/make_figures.py: matplotlib has no bold Helvetica face here, so bold is DejaVu
LETTER = dict(fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")
PLDDT_MIN = 70


def letter(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, **LETTER, va="bottom", ha="left")


# ── inputs, checked against what the SI prints ───────────────────────────────

def load():
    auc = pd.read_csv(IN / "auc_matched_draws_25seed.csv")
    table_s2 = {  # Supplementary Table S2, as printed
        "POLE": (0.858, 0.762, 0.735, 0.718), "PTEN": (0.872, 0.897, 0.595, 0.471),
        "VHL": (0.916, 0.815, 0.585, 0.498), "MSH2": (0.847, 0.683, 0.658, 0.469),
        "MSH6": (0.903, 0.762, 0.552, 0.468), "PIK3CA": (0.707, 0.749, 0.766, 0.543),
        "Macro": (0.851, 0.778, 0.649, 0.528)}
    for g, vals in table_s2.items():
        for d, v in zip((KS, POS, RMSD, DRMSF), vals):
            got = auc[(auc.gene == g) & (auc.descriptor == d)].auc.item()
            assert round(got, 3) == v, (g, d, got, v)

    scores = pd.read_csv(IN / "variant_seed_scores_25seed.csv.gz")
    assert len(scores) == 17_970 and scores.seed.nunique() == 10
    # the AUC table is the mean of per-seed AUCs; the per-variant scores must give it back
    for g in GENES:
        for d, col in ((RMSD, "rmsd"), (DRMSF, "delta_rmsf")):
            b = scores[scores.gene == g]
            per_seed = [roc_auc_score(s.label, s[col]) for _, s in b.groupby("seed")]
            want = auc[(auc.gene == g) & (auc.descriptor == d)].auc.item()
            assert abs(np.mean(per_seed) - want) < 1e-9, (g, d, np.mean(per_seed), want)
    per_variant = scores.groupby(["gene", "mutation", "label"], as_index=False)[
        ["rmsd", "delta_rmsf"]].mean()

    shift = pd.read_csv(IN / "shift_vs_rmsf_P286R_25seed.csv")
    vals = pd.read_csv(IN / "shift_vs_rmsf_P286R_values_25seed.csv").set_index("quantity").value
    assert len(shift) == 204
    assert round(pearsonr(shift.centre_shift_A, shift.delta_rmsf_A).statistic, 3) == float(vals["pearson_r"])
    assert round(spearmanr(shift.centre_shift_A, shift.delta_rmsf_A).statistic, 3) == float(vals["spearman_rho"])

    resid = pd.read_csv(IN / "wt_rmsf_vs_plddt_residues.csv")
    printed = {"POLE": (0.08, -0.72), "PTEN": (0.11, -0.89), "VHL": (0.11, -0.88),
               "MSH2": (0.41, -0.68), "MSH6": (0.31, -0.88), "PIK3CA": (0.15, -0.79)}
    cores = {"POLE": 204, "PTEN": 322, "VHL": 146, "MSH2": 832, "MSH6": 896, "PIK3CA": 997}
    for g, (med, rho) in printed.items():
        t = resid[resid.gene == g]
        assert round(t[t.plddt >= PLDDT_MIN].rmsf.median(), 2) == med, g
        assert round(spearmanr(t.plddt, t.rmsf).statistic, 2) == rho, g
        assert int(t.core.sum()) == cores[g], g   # Supplementary Table S1
    return auc, per_variant, shift, resid


# ── S1: descriptors ──────────────────────────────────────────────────────────

def panel_shift(ax, shift):
    x, y = shift.centre_shift_A.to_numpy(), shift.delta_rmsf_A.to_numpy()
    ax.axhline(0, ls=(0, (3, 2)), lw=.6, color=GRY, zorder=1)
    ax.scatter(x, y, s=8, color=DRV, alpha=.55, edgecolor="none", zorder=2)
    fit = linregress(x, y)
    xs = np.array([0, x.max() * 1.03])
    ax.plot(xs, fit.intercept + fit.slope * xs, color="#2c3e50", lw=.9, zorder=3)
    j = int(np.flatnonzero(shift.pole_resid.to_numpy() == SITE)[0])
    ax.scatter(x[j], y[j], s=38, facecolor="none", edgecolor="black", lw=.9, zorder=4)
    ax.annotate(f"residue {SITE}", (x[j], y[j]), xytext=(-5, 6), textcoords="offset points",
                ha="right", fontsize=6)
    r, rho = pearsonr(x, y).statistic, spearmanr(x, y).statistic
    ax.text(.98, .04, f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6)
    ax.set_xlim(0, None)
    ax.set_xlabel("Shift of ensemble-mean Cα position (Å)")
    ax.set_ylabel("ΔRMSF, P286R − WT (Å)")
    ax.set_title("POLE P286R, 125 predictions per ensemble", loc="left", fontsize=6.5)


def panel_auc(ax, auc):
    rows = GENES + ["Macro"]
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[-1] -= 0.4                                   # set the macro row apart
    series = [(KS, ACC, "o", True), (POS, "#e07a1f", "s", True),
              (RMSD, "#6f6f6f", "^", False), (DRMSF, "#6f6f6f", "v", False)]
    offs = np.linspace(0.27, -0.27, len(series))
    for (lab, colour, mk, filled), off in zip(series, offs):
        s = auc[auc.descriptor == lab].set_index("gene").loc[rows]
        for y, (g, r) in zip(ys, s.iterrows()):
            ax.plot([r.lo, r.hi], [y + off] * 2, color=colour, lw=.7, solid_capstyle="butt")
            ax.plot(r.auc, y + off, mk, color=colour, ms=3.6 if g != "Macro" else 4.4,
                    mfc=colour if filled else "white", mew=.8,
                    label=lab if g == rows[0] else None)
    ax.axvline(.5, color=GRY, lw=.5, ls=":")
    ax.axhline((ys[-2] + ys[-1]) / 2, color=GRY, lw=.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty(g) for g in rows])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(.3, 1.0)
    ax.set_ylim(ys[-1] - .6, ys[0] + .6)
    ax.set_xlabel("Driver vs passenger ROC-AUC (N = 90)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handletextpad=.2,
              columnspacing=.9, borderaxespad=.2)


def panel_violins(fig, gs_row, per_variant, auc, col, descriptor, ylabel, first_row):
    sub = gs_row.subgridspec(1, len(GENES), wspace=.62)
    axes = []
    for c, g in enumerate(GENES):
        ax = fig.add_subplot(sub[0, c])
        b = per_variant[per_variant.gene == g]
        class_violin(ax, [(b[b.label == 1][col], "Driver", DRV),
                          (b[b.label == 0][col], "Passenger", PAS)], seed=c)
        if col == "delta_rmsf":
            ax.axhline(0, ls=(0, (3, 2)), lw=.6, color=GRY, zorder=1)
        a = auc[(auc.gene == g) & (auc.descriptor == descriptor)].iloc[0]
        head = f"{pretty(g)}\n" if first_row else ""
        ax.set_title(f"{head}AUC {a.auc:.2f} ({a.lo:.2f}–{a.hi:.2f})", fontsize=6, pad=3)
        ax.set_xticklabels([f"n={(b.label == k).sum():,}" for k in (1, 0)], fontsize=5.5)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=5.5)
        if c == 0:
            ax.set_ylabel(ylabel)
        axes.append(ax)
    return axes


def figure_descriptors(auc, per_variant, shift):
    fig = plt.figure(figsize=(W, 150 * MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.25, 2.15], hspace=.36,
                  left=.075, right=.99, top=.95, bottom=.05)
    top = gs[0].subgridspec(1, 2, width_ratios=[1, 1.25], wspace=.32)
    ax_a, ax_b = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    panel_shift(ax_a, shift)
    panel_auc(ax_b, auc)
    rows = gs[1].subgridspec(2, 1, hspace=.36)
    c1 = panel_violins(fig, rows[0], per_variant, auc, "rmsd", RMSD,
                       "Ensemble-mean RMSD (Å)", True)
    panel_violins(fig, rows[1], per_variant, auc, "delta_rmsf", DRMSF,
                  "ΔRMSF, mutant − WT (Å)", False)
    letter(ax_a, "a", dx=-.2, dy=1.06)
    letter(ax_b, "b", dx=-.16, dy=1.06)
    # c's letter and its class key share one line above the gene titles
    y_c = c1[0].get_position().y1 + .052
    fig.text(.012, y_c, "c", **LETTER, va="center", ha="left")
    fig.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4, color=DRV, label="Driver"),
                        plt.Line2D([], [], ls="", marker="o", ms=4, color=PAS, label="Passenger")],
               loc="center left", ncol=2, bbox_to_anchor=(.035, y_c), handletextpad=.2,
               borderaxespad=0)
    save(fig, "FigureS_coordinate_descriptors", out_dir=OUT)


# ── S2: rigid cores ──────────────────────────────────────────────────────────

def figure_rigid_core(resid):
    fig, axes = plt.subplots(2, 3, figsize=(W, 92 * MM), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=.32, wspace=.1, left=.075, right=.99,
                                              top=.95, bottom=.11))
    for ax, g in zip(axes.flat, GENES):
        t = resid[resid.gene == g]
        for core, colour, z in ((False, GRY, 1), (True, GENE_COLOR[g], 2)):
            s = t[t.core == core]
            ax.scatter(s.plddt, s.rmsf, s=2.6, lw=0, color=colour, alpha=.6, zorder=z,
                       rasterized=True)
        ax.axvline(PLDDT_MIN, color="#4a4a4a", lw=.5, ls=":")
        ax.set_yscale("log")
        n_core, n = int(t.core.sum()), len(t)
        ax.set_title(f"{pretty(g)}  ({n_core:,} of {n:,} residues in the core)", loc="left",
                     fontsize=6.5, color=GENE_COLOR[g], fontweight="bold")
        conf = t[t.plddt >= PLDDT_MIN]
        rho = spearmanr(t.plddt, t.rmsf).statistic
        ax.text(.03, .05, f"median RMSF, pLDDT ≥ 70: {conf.rmsf.median():.2f} Å\nρ = {rho:.2f}",
                transform=ax.transAxes, fontsize=6, va="bottom", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=.9))
    for ax in axes[1]:
        ax.set_xlabel("WT mean pLDDT")
    for ax in axes[:, 0]:
        ax.set_ylabel("WT RMSF (Å)")
    save(fig, "FigureS_rigid_core", out_dir=OUT)


def main():
    auc, per_variant, shift, resid = load()
    print("  inputs reproduce Table S2, Table S1 cores and the printed panel statistics")
    figure_descriptors(auc, per_variant, shift)
    figure_rigid_core(resid)


if __name__ == "__main__":
    main()
