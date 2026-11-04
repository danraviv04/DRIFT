"""Fig. 1h: KS-score ROC-AUC against ensemble size for one gene.

Drawn from Figure 2's per-gene, per-seed AUC table
(analyses/annotation/stages/publication/figure2/panel_abcd_per_gene_auc.csv), so
Fig. 1h and the same gene's line in Fig. 2a are the same numbers. Used by
rerender_efgh.py (panel h).
"""
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Type 42 (TrueType) rather than the Type 3 default: several journals reject
# Type 3 outright, and it is the single most common figure-submission bounce.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ACC = "#6a4c93"
# Figure 1 row 4 type scale. Panels f and g are drawn by
# figures/figure1/ks_combo/make_panels_gh.py under these rcParams and panel
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

VALUES: list[dict] = []


def record(panel, key, value, note=""):
    VALUES.append({"panel": panel, "quantity": key, "value": value, "note": note})
    return value


def save(fig, out_dir, name):
    """Write PDF and PNG for the same panel (the PDF goes into the figure)."""
    out_dir = Path(out_dir)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}  (pdf + png)")
    return pdf


def panel_nsweep_metrics(out, gene, stem, key, columns, auc_table, row4_style=False):
    """ROC-AUC against ensemble size for one gene, from Figure 2's AUC table.

    ``auc_table`` has columns N, seed, gene, descriptor, auc. Only the first entry
    of ``columns`` (column, label, colour) is drawn; its N=1 point is the table's
    "KS at N=1" row. Error bars are the SD across the table's 10 resampling seeds,
    matching Fig. 2a,b.

    ``row4_style`` dresses the panel like its neighbour g in Figure 1
    (make_panels_gh.panel_g): FIG1_ROW4_RC, the same 3.2 in height, open
    top/right spines, no grid, and no chance line. The height matters more
    than it looks: the assembler scales a row to ONE height, so a panel
    authored at 4.0 in printed its type at 80% of g's. The chance line goes
    because the curve never comes near 0.5; it only pulled the axis down.
    """
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
