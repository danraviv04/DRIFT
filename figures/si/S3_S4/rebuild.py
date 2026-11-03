#!/usr/bin/env python
"""Supplementary Figs. S3 and S4 (Note S2), regrouped by argument and drawn whole.

Before 2026-10-29 the WT-background figure carried two arguments: a, how high the
KS-score sits between two WT halves, and b, how often a raw per-residue
Mann-Whitney test fires between them. Panel b belongs with the per-residue test,
so the two figures are now:

  FigureS_per_residue_test.pdf  (S3)  a, the raw test between WT halves, six genes
                                      b-e, the MSH6 pair of Fig. 1e at every residue
  FigureS_wt_backgrounds.pdf    (S4)  KS-score of passengers and drivers against the
                                      WT-WT background, six genes

Nothing is recomputed: the values come from the POLE 25-seed outputs of
wt_split_12v12 and from fig1e_near_pair/gene_wide.py (inputs/), and the script
asserts they give the numbers printed in Note S2 before drawing.

    python figures/si/S3_S4/rebuild.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "figures/si/lib"))
from figlib import DRV, GRY, MM, PAS, WTC, class_violin, pretty, save  # noqa: E402

IN = HERE / "inputs"
OUT = HERE / "results"
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
W = 168 * MM                      # SI text width
WINDOW = (1047, 1062)             # the Fig. 1e window
DRIVER, PASSENGER = "A1055P", "I1054F"
BAND = "#f0c808"


def letter(fig, ax, s, x=.004):
    # just above the axes' top-left, clear of the y label
    fig.text(x, ax.get_position().y1 + .012, s, ha="left", va="bottom", fontsize=10,
             fontweight="bold", fontfamily="DejaVu Sans")  # as tools/make_figures.py


# ── inputs, checked against Note S2 ──────────────────────────────────────────

def load():
    splits = pd.read_csv(IN / "wt_background_splits.csv")
    variants = pd.read_csv(IN / "wt_background_variant_scores.csv")
    assert len(variants) == 1_798 and (splits.groupby(["gene", "kind"]).size() == 2_000).all()

    # Note S2 table: WT-WT 95th percentile, drivers above, passengers above
    table = {"POLE": (0.274, 37, 38, 589, 1036), "PTEN": (0.217, 243, 248, 7, 9),
             "VHL": (0.247, 101, 105, 9, 19), "MSH2": (0.245, 42, 71, 3, 38),
             "MSH6": (0.231, 26, 66, 3, 77), "PIK3CA": (0.280, 35, 84, 1, 7)}
    p95 = {}
    for g, (q, d_up, d_n, p_up, p_n) in table.items():
        p95[g] = np.quantile(splits[(splits.gene == g) & (splits.kind == "seed")].ks, .95)
        v = variants[variants.gene == g]
        drv, pas = v[v.label == 1].ks, v[v.label == 0].ks
        assert round(p95[g], 3) == q, (g, p95[g])
        assert ((drv > p95[g]).sum(), len(drv), (pas > p95[g]).sum(), len(pas)) == (d_up, d_n, p_up, p_n), g

    # Note S2: median 12-24% (seed-disjoint) against 0.5-3.7% (random)
    rate = (splits.assign(pct=100 * splits.frac_p05)
            .groupby(["gene", "kind"]).pct.quantile([.25, .5, .75]).unstack())
    med = rate[.5].unstack()
    assert (round(med.seed.min()), round(med.seed.max())) == (12, 24)
    assert (round(med.random.min(), 1), round(med.random.max(), 1)) == (0.5, 3.7)

    gw = pd.read_csv(IN / "MSH6_gene_wide_per_residue.csv")
    assert len(gw) == 1_360
    for code, ks, called, outside in ((DRIVER, .227, 276, 260), (PASSENGER, .147, 16, 13)):
        hit = gw[f"called_share_{code}"] >= .5
        out = ~gw.residue.between(*WINDOW)
        assert round(gw[f"ks_{code}"].mean(), 3) == ks, code
        assert (hit.sum(), (hit & out).sum()) == (called, outside), code
    return splits, variants, p95, rate, gw


# ── S3: the per-residue test ─────────────────────────────────────────────────

def panel_calibration(ax, rate):
    x = np.arange(len(GENES))
    for off, kind, colour, lab in ((-.19, "seed", "#3a3a3a", "Seed-disjoint halves"),
                                   (.19, "random", "#b5b5b5", "Random halves")):
        r = rate.xs(kind, level="kind").loc[GENES]
        ax.bar(x + off, r[.5], width=.36, color=colour, label=lab, zorder=2)
        ax.errorbar(x + off, r[.5], yerr=[r[.5] - r[.25], r[.75] - r[.5]], fmt="none",
                    ecolor="#1a1a1a", elinewidth=.6, capsize=1.6, capthick=.6, zorder=3)
    ax.axhline(5, color=DRV, ls=(0, (3, 2)), lw=.8, zorder=1, label="Nominal 5%")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(g) for g in GENES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Residues with\nP < 0.05 (%)")
    ax.set_title("Raw Mann–Whitney test between two WT halves, 60 predictions per side",
                 loc="left", fontsize=6.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.4, borderaxespad=0)


def gene_wide(fig, gs, gw):
    resid = gw.residue.to_numpy()
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    ax_wt, ax_ks, ax_c, ax_sh = axes
    for a in axes[:3]:
        a.sharex(ax_sh)
        a.tick_params(axis="x", labelbottom=False)
    for a in axes:
        a.axvspan(WINDOW[0] - .5, WINDOW[1] + .5, color=BAND, alpha=.30, lw=0, zorder=0)

    wt = gw.wt_mean_plddt_pool.to_numpy()
    ax_wt.fill_between(resid, 0, wt, color=WTC, alpha=.35, lw=0)
    ax_wt.plot(resid, wt, color=WTC, lw=.5)
    ax_wt.axhline(70, color="#666", ls=":", lw=.6)
    ax_wt.set(ylim=(0, 100), yticks=[0, 50, 100])
    ax_wt.set_ylabel("WT pLDDT")
    ax_wt.annotate("Fig. 1e", xy=(np.mean(WINDOW), 1.0), xycoords=("data", "axes fraction"),
                   xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                   fontsize=6, color="#8a7500")

    for code, c, role in ((DRIVER, DRV, "Driver"), (PASSENGER, PAS, "Passenger")):
        k = gw[f"ks_{code}"].to_numpy()
        ax_ks.plot(resid, k, color=c, lw=.45, alpha=.9,
                   label=f"{role} {code} (KS-score {k.mean():.3f}, dashed)")
        ax_ks.axhline(k.mean(), color=c, ls="--", lw=.8)
    ax_ks.set(ylim=(0, 1.04), yticks=[0, .5, 1])
    ax_ks.set_ylabel("Per-residue KS")
    ax_ks.legend(loc="upper left", handlelength=1.4, borderaxespad=.2)

    for y, code, c in ((1, DRIVER, DRV), (0, PASSENGER, PAS)):
        share = gw[f"called_share_{code}"].to_numpy()
        hit = share > 0
        rgba = np.array([matplotlib.colors.to_rgba(c)] * hit.sum())
        rgba[:, 3] = share[hit]
        ax_c.vlines(resid[hit], y - .38, y + .38, colors=rgba, lw=.6)
    ax_c.set(ylim=(-.6, 1.6), yticks=[0, 1])
    ax_c.set_yticklabels([PASSENGER, DRIVER], fontsize=6)
    for lab, c in zip(ax_c.get_yticklabels(), (PAS, DRV)):
        lab.set_color(c)
    ax_c.tick_params(axis="both", length=0)
    ax_c.set_ylabel("Called")
    for s in ("left", "bottom"):
        ax_c.spines[s].set_visible(False)

    sd = gw.wt_sd_plddt_pool.to_numpy()
    ax_sh.fill_between(resid, -sd, sd, color=WTC, alpha=.20, lw=0, label="WT ± 1 s.d.")
    ax_sh.axhline(0, color="#444", ls="--", lw=.5)
    shifts = [gw[f"shift_{c}"].to_numpy() for c in (PASSENGER, DRIVER)]
    for s, c in zip(shifts, (PAS, DRV)):
        ax_sh.plot(resid, s, color=c, lw=.45, alpha=.9)
    lo, hi = min(s.min() for s in shifts), max(s.max() for s in shifts)
    ax_sh.set_ylim(lo - .1 * (hi - lo), hi + .1 * (hi - lo))
    ax_sh.set_ylabel("pLDDT shift,\nvariant − WT")
    ax_sh.legend(loc="lower left", handlelength=1.2, borderaxespad=.2)
    ax_sh.set(xlim=(resid.min() - .5, resid.max() + .5), xlabel="MSH6 residue")
    ax_sh.set_xticks(np.arange(100, resid.max() + 1, 100))
    return axes


def figure_per_residue(rate, gw):
    fig = plt.figure(figsize=(W, 132 * MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 3.05], hspace=.2,
                  left=.105, right=.985, top=.93, bottom=.075)
    top = gs[0].subgridspec(1, 2, width_ratios=[2.1, 1], wspace=0)
    ax_a = fig.add_subplot(top[0, 0])
    panel_calibration(ax_a, rate)
    tracks = gs[1].subgridspec(4, 1, height_ratios=[.75, 1.6, .42, 1.45], hspace=.24)
    axes = gene_wide(fig, tracks, gw)
    fig.align_ylabels([ax_a] + axes)
    for ax, s in zip([ax_a] + axes, "abcde"):
        letter(fig, ax, s)
    save(fig, "FigureS_per_residue_test", out_dir=OUT)


# ── S4: KS-score backgrounds ─────────────────────────────────────────────────

def figure_backgrounds(splits, variants, p95):
    fig, axes = plt.subplots(2, 3, figsize=(W, 100 * MM),
                             gridspec_kw=dict(hspace=.5, wspace=.3, left=.075, right=.99,
                                              top=.9, bottom=.1))
    for i, (ax, g) in enumerate(zip(axes.flat, GENES)):
        bg = splits[(splits.gene == g) & (splits.kind == "seed")].ks
        v = variants[variants.gene == g]
        class_violin(ax, [(bg, "WT vs WT", WTC), (v[v.label == 0].ks, "Passengers", PAS),
                          (v[v.label == 1].ks, "Drivers", DRV)],
                     p95=p95[g], p95_label=None, seed=i)
        ax.set_title(f"{pretty(g)}   driver vs passenger AUC {roc_auc_score(v.label, v.ks):.2f}",
                     loc="left", fontsize=6.5)
        ax.tick_params(axis="x", labelsize=5.5, length=0)
        if i % 3 == 0:
            ax.set_ylabel("KS-score (60 per side)")
    fig.legend(handles=[plt.Line2D([], [], color=GRY, ls=(0, (3, 2)), lw=.8,
                                   label="WT–WT 95th percentile (seed-disjoint halves)")],
               loc="upper left", bbox_to_anchor=(.07, 1.0), borderaxespad=.2)
    save(fig, "FigureS_wt_backgrounds", out_dir=OUT)


def main():
    splits, variants, p95, rate, gw = load()
    print("  inputs reproduce the Note S2 table, the 12-24% / 0.5-3.7% rates and the MSH6 calls")
    figure_per_residue(rate, gw)
    figure_backgrounds(splits, variants, p95)


if __name__ == "__main__":
    main()
