"""Qualitative POLE driver-vs-passenger KS box plot for the graphical abstract.

Scores: residueKS (plddt_ks_residue) at N=90, averaged over the ten canonical
draws (seeds 42-51) of the POLE 25-seed caches in drift/cache (fingerprint
97f1cdbfda8e, P356S dropped; 38 drivers / 1,036 passengers).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[2] / "drift/cache"
FILES = [CACHE / f"labeled_size90_seed{s}_ratio_of_means_proteins97f1cdbfda8e_esfeat6.csv"
         for s in range(42, 52)]
DRIVER, PASSENGER = "#D1495B", "#2E86AB"
rng = np.random.default_rng(0)

frames = [pd.read_csv(f).query("gene == 'POLE'") for f in FILES]
df = (pd.concat(frames).groupby(["mutation", "label"], as_index=False)
      ["plddt_ks_residue"].mean())
df.to_csv(HERE / "pole_ks_n90_seedmean.csv", index=False)
d = df.loc[df.label == 1, "plddt_ks_residue"].to_numpy()
p = df.loc[df.label == 0, "plddt_ks_residue"].to_numpy()
u, pval = mannwhitneyu(d, p, alternative="greater")
auc = u / (len(d) * len(p))
print(f"drivers {len(d)}, passengers {len(p)}, AUC {auc:.3f}, p {pval:.2e}")


def draw(annotated):
    fig, ax = plt.subplots(figsize=(3.2, 3.6) if not annotated else (5, 5))
    for x, vals, c in [(0, d, DRIVER), (1, p, PASSENGER)]:
        v = ax.violinplot(vals, [x], widths=0.8, showextrema=False)
        for b in v["bodies"]:
            b.set_facecolor(c); b.set_edgecolor("none"); b.set_alpha(0.22)
        jit = x + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(jit, vals, s=9 if x == 0 else 4, color=c,
                   alpha=0.75 if x == 0 else 0.35, lw=0, zorder=2)
        ax.boxplot(vals, positions=[x], widths=0.32, showfliers=False,
                   patch_artist=True, zorder=3,
                   boxprops=dict(facecolor="white", edgecolor=c, lw=2),
                   medianprops=dict(color=c, lw=2.5),
                   whiskerprops=dict(color=c, lw=1.6),
                   capprops=dict(color=c, lw=1.6))
    ax.set_xticks([0, 1], ["Drivers", "Passengers"], fontsize=13)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("KS score", fontsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if annotated:
        ax.set_title(f"POLE (N=90)  AUC {auc:.2f}", fontsize=12)
        ax.set_xticks([0, 1], [f"Drivers\n(n={len(d)})",
                               f"Passengers\n(n={len(p)})"], fontsize=12)
    else:
        ax.set_yticks([])
        ax.annotate("", xy=(-0.12, 1), xytext=(-0.12, 0),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("")
        ax.text(-0.17, 0.5, "KS score", transform=ax.transAxes, rotation=90,
                va="center", ha="center", fontsize=13)
    fig.tight_layout()
    stem = "ga_ks_boxplot" + ("_annotated" if annotated else "")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=400, transparent=True)
    plt.close(fig)


draw(False)
draw(True)


def draw_ga_hires(with_dots):
    """Schematic GA panel: real POLE shapes, no numbers, thick strokes, 1200 dpi."""
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    for x, vals, c in [(0, d, DRIVER), (1, p, PASSENGER)]:
        v = ax.violinplot(vals, [x], widths=0.85, showextrema=False)
        for b in v["bodies"]:
            b.set_facecolor(c); b.set_edgecolor(c); b.set_linewidth(1.2)
            b.set_alpha(0.25)
        if with_dots:
            shown = vals if len(vals) < 100 else rng.choice(vals, 120, replace=False)
            ax.scatter(x + rng.uniform(-0.16, 0.16, len(shown)), shown, s=14,
                       color=c, alpha=0.8, lw=0, zorder=4)
        ax.boxplot(vals, positions=[x], widths=0.3, showfliers=False,
                   patch_artist=True, zorder=3,
                   boxprops=dict(facecolor="white", edgecolor=c, lw=3),
                   medianprops=dict(color=c, lw=4, solid_capstyle="butt"),
                   whiskerprops=dict(color=c, lw=2.5),
                   capprops=dict(color=c, lw=2.5))
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["Drivers", "Passengers"], fontsize=15,
                  fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.annotate("", xy=(-0.08, 1.0), xytext=(-0.08, 0.0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
                                color="black", lw=2))
    ax.text(-0.15, 0.5, "KS score", transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontsize=15, fontweight="bold")
    fig.tight_layout()
    stem = "ga_ks_hires" + ("" if with_dots else "_nodots")
    fig.savefig(HERE / f"{stem}.png", dpi=1200, transparent=True)
    fig.savefig(HERE / f"{stem}.svg", transparent=True)
    plt.close(fig)


draw_ga_hires(True)
draw_ga_hires(False)
