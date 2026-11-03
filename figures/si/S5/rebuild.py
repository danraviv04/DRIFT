#!/usr/bin/env python
"""Supplementary Fig. S5: the KS-score against AlphaMissense, drawn whole.

The old figure stacked two panel PDFs (si_figures/ks_vs_am.py) at their own sizes.
This draws it at the SI text width (168 mm) with figlib's style:

  a  KS-score at N = 90 against AlphaMissense, per annotation gene
  b  Spearman rho per gene for all variants, drivers and passengers, and the macro

Inputs (inputs/) are the POLE 25-seed outputs: the per-variant table was averaged
from the ten N = 90 score caches of the 25-seed config (fingerprint 49028c16cbb3)
exactly as ks_vs_am.py does, and the correlations and macro are that run's
published values. The script asserts that the table gives back every published
rho before drawing.

    python figures/si/S5/rebuild.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "figures/si/lib"))
from figlib import DRV, GENE_COLOR, GRY, MM, PAS, pretty, save  # noqa: E402

IN = HERE / "inputs"
OUT = HERE / "results"
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
W = 168 * MM
INK = "#2c3e50"
MIN_CLASS_N = 10                  # classes smaller than this get no rho (PTEN, PIK3CA passengers)
# as tools/make_figures.py: matplotlib has no bold Helvetica face here, so bold is DejaVu
LETTER = dict(fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")


def load():
    v = pd.read_csv(IN / "ks_vs_am_variants_25seed.csv")
    st = pd.read_csv(IN / "ks_vs_am_correlations.csv")
    macro = pd.read_csv(IN / "ks_vs_am_macro.csv").set_index("unit").loc["position"]
    assert len(v) == 1_798
    for g in GENES:
        d = v[v.gene == g]
        for sub, x in (("all", d), ("drivers", d[d.label == 1]), ("passengers", d[d.label == 0])):
            want = st[(st.gene == g) & (st.subset == sub)].rho.item()
            assert abs(spearmanr(x.am, x.ks).statistic - want) < 1e-9, (g, sub)
    # what the SI prints: per-gene 0.47-0.76 (Main), macro 0.64 (0.59-0.68), POLE passengers 0.61
    allr = st[st.subset == "all"].rho
    assert (round(allr.min(), 2), round(allr.max(), 2)) == (0.47, 0.76)
    assert (round(macro.rho, 2), round(macro.ci_lo, 2), round(macro.ci_hi, 2)) == (0.64, 0.59, 0.68)
    assert round(st[(st.gene == "POLE") & (st.subset == "passengers")].rho.item(), 2) == 0.61
    return v, st, macro


def alpha(n):
    return float(np.clip(.9 * np.sqrt(40 / max(n, 1)), .3, .9))


def panel_scatter(fig, gs, v, st):
    sub = gs.subgridspec(2, 3, hspace=.38, wspace=.32)
    axes = []
    for i, g in enumerate(GENES):
        ax = fig.add_subplot(sub[i // 3, i % 3])
        d = v[v.gene == g]
        p, q = d[d.label == 0], d[d.label == 1]
        ax.scatter(p.am, p.ks, s=6, facecolors="none", edgecolors=PAS, lw=.5, alpha=alpha(len(p)))
        ax.scatter(q.am, q.ks, s=6, color=DRV, lw=0, alpha=alpha(len(q)))
        slope, icpt = np.polyfit(d.am, d.ks, 1)
        xs = np.array([d.am.min(), d.am.max()])
        ax.plot(xs, icpt + slope * xs, color=INK, lw=.9)
        r = st[(st.gene == g) & (st.subset == "all")].rho.item()
        ax.text(.03, .97, f"ρ = {r:.2f}\n{len(q)} D, {len(p):,} P", transform=ax.transAxes,
                va="top", fontsize=6, linespacing=1.15)
        ax.set_title(pretty(g), loc="left", fontsize=7, color=GENE_COLOR[g], fontweight="bold")
        ax.set_xlim(-.02, 1.02)
        ax.set_xticks([0, .5, 1])
        if i // 3 == 1:
            ax.set_xlabel("AlphaMissense score")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        if i % 3 == 0:
            ax.set_ylabel("KS-score (N = 90)")
        axes.append(ax)
    return axes


def panel_forest(ax, st, macro):
    rows = GENES + ["Macro"]
    ys = {g: float(len(rows) - 1 - i) for i, g in enumerate(rows)}
    ys["Macro"] -= .4
    parts = [("all", INK, "o", "white", "All variants"), ("drivers", DRV, "s", DRV, "Drivers"),
             ("passengers", PAS, "^", "white", "Passengers")]
    offs = {"all": .25, "drivers": 0., "passengers": -.25}
    for part, colour, mk, face, lab in parts:
        first = True
        for g in GENES:
            r = st[(st.gene == g) & (st.subset == part)].iloc[0]
            if r.n < MIN_CLASS_N:
                continue
            y = ys[g] + offs[part]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=colour, lw=.8, solid_capstyle="butt")
            ax.plot(r.rho, y, mk, color=colour, ms=3.6, mfc=colour if part != "passengers" else "white",
                    mew=.8, label=lab if first else None)
            first = False
    ax.plot([macro.ci_lo, macro.ci_hi], [ys["Macro"]] * 2, color=INK, lw=.8)
    ax.plot(macro.rho, ys["Macro"], "D", color=INK, ms=4.2)
    ax.axvline(0, color=GRY, lw=.5, ls=":")
    ax.axhline((ys["PIK3CA"] + ys["Macro"]) / 2, color=GRY, lw=.4)
    ax.set_yticks([ys[g] for g in rows])
    ax.set_yticklabels([pretty(g) for g in rows])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(ys["Macro"] - .6, ys["POLE"] + .6)
    ax.set_xlim(-.65, 1)
    ax.set_xticks([-.5, 0, .5, 1])
    ax.set_xlabel("Spearman ρ with AlphaMissense")
    ax.legend(loc="lower left", handletextpad=.3, borderaxespad=.3, labelspacing=.35)


def main():
    v, st, macro = load()
    print("  inputs reproduce every published rho and the numbers the SI prints")
    fig = plt.figure(figsize=(W, 92 * MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.35, 1], wspace=.26,
                  left=.07, right=.985, top=.9, bottom=.12)
    axes = panel_scatter(fig, gs[0], v, st)
    ax_b = fig.add_subplot(gs[1])
    panel_forest(ax_b, st, macro)
    for ax, s, x in ((axes[0], "a", .004), (ax_b, "b", ax_b.get_position().x0 - .085)):
        fig.text(x, ax.get_position().y1 + .04, s, ha="left", va="bottom", **LETTER)
    fig.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4, color=DRV, label="Driver"),
                        plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=PAS, mew=.8,
                                   label="Passenger")],
               loc="lower right", ncol=2, bbox_to_anchor=(axes[2].get_position().x1, axes[0].get_position().y1 + .045),
               handletextpad=.2, borderaxespad=0)
    save(fig, "FigureS_ks_vs_alphamissense", out_dir=OUT)


if __name__ == "__main__":
    main()
