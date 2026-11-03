#!/usr/bin/env python
"""Supplementary Fig. S2 (WT backgrounds) rebuilt with a 12-vs-12 seed split and 2,000 splits.

Everything else is si_figures/wt_backgrounds.py unchanged (WT pools, N = 60, variant scores,
random-split arm, panels). Two changes:
- seed split: 25 seeds shuffled, 12 vs 12, the 25th left out of that split; each side is
  12 complete seeds = all 60 predictions, so no within-seed subsampling.
- 2,000 splits per gene instead of 200, so the 95th percentile is stable (rng_noise.py showed
  the 200-split p95 swings VHL passengers 7-17/19 and POLE 486-555/1,037).
A stability check re-estimates the p95 under 3 further independent streams of 2,000 splits.

Variant scores come from the POLE-repair caches, as in the published S2.
Outputs go to results_12v12_2000/ here; si_figures/results/ (the published S2) is not touched.

    PYTHONPATH="." python experiments/wt_split_12v12/run_12v12_2000.py
"""
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SI = HERE.parent / "si_figures"
sys.path.insert(0, str(SI))
import figlib  # noqa: E402
import wt_backgrounds as wb  # noqa: E402

# WT12_OUT / WT12_CACHE redirect a rerun (experiments/annotation_stages); unset = as published
OUT = Path(os.environ.get("WT12_OUT", HERE / "results_12v12_2000"))
OUT.mkdir(parents=True, exist_ok=True)
figlib.RESULTS = OUT
# the published S2 (pole_wt_repair/run_publication_stage.py, stage si_wt_backgrounds) read
# variant scores from the POLE single-WT caches, fingerprint ebbbda31758d
figlib.FUSION_CACHE = Path(os.environ.get("WT12_CACHE", HERE.parent / "pole_wt_repair" / "cache"))
wb.RESULTS = OUT
wb.SPLITS = 2000
CHECK_STREAMS = 3


def seed_split_12v12(seeds, rng):
    """Two disjoint sets of 12 complete seeds (60 predictions each); the 25th seed sits out."""
    s = rng.permutation(np.unique(seeds))
    a = np.flatnonzero(np.isin(seeds, s[:12]))
    b = np.flatnonzero(np.isin(seeds, s[12:24]))
    if not len(a) == len(b) == wb.N:
        raise ValueError(f"12-seed sides hold {len(a)} and {len(b)} predictions, need {wb.N} each")
    return a, b


wb.seed_split = seed_split_12v12


def panel_backgrounds(bg, var, summ):
    """wt_backgrounds.panel_backgrounds with the legend naming the 12-vs-12 split."""
    fig, axes = wb.grid(2, 3, w=wb.W2, h=wb.W2 * 0.66)
    fig.subplots_adjust(hspace=0.6, wspace=0.32)
    for ax, g in zip(axes.flat, wb.ANNOTATION_GENES):
        null = bg[(bg.gene == g) & (bg.kind == "seed")].ks.to_numpy()
        v = var[var.gene == g]
        wb.class_violin(ax, [(null, "WT vs WT", wb.WTC), (v[v.label == 0].ks, "Passenger", wb.PAS),
                             (v[v.label == 1].ks, "Driver", wb.DRV)],
                        p95=summ[g]["p95"], p95_label=None, seed=wb.BASE_SEED + zlib.crc32(g.encode()),
                        title=f"{wb.pretty(g)}    driver vs passenger AUC {summ[g]['auc']:.2f}")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"KS-score (N = {wb.N})")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls=(0, (3, 2)), lw=.7, color=wb.GRY,
                               label="WT–WT 95th percentile (seed-matched, 12 vs 12 seeds)")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    wb.save(fig, "wt_background_per_gene")


wb.panel_backgrounds = panel_backgrounds


def stability(var):
    rows = []
    for g in wb.ANNOTATION_GENES:
        x, seeds, _ = wb.wt_pool(g)
        v = var[var.gene == g]
        d, p = v[v.label == 1].ks.to_numpy(), v[v.label == 0].ks.to_numpy()
        for k in range(CHECK_STREAMS):
            rng = np.random.default_rng(zlib.crc32(f"12v12_2000_check/{g}/{k}".encode()))
            ks = []
            for _ in range(wb.SPLITS):
                a, b = seed_split_12v12(seeds, rng)
                ks.append(wb.residue_ks(x[:, a], x[:, b]))
            p95 = float(np.percentile(ks, 95))
            rows.append({"gene": g, "stream": k, "p95": p95,
                         "drivers_above": int((d > p95).sum()), "n_drivers": len(d),
                         "passengers_above": int((p > p95).sum()), "n_passengers": len(p)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    wb.main()
    # wt_backgrounds.main() hard-codes "12 vs 13 seeds" in the value notes
    vals = OUT / "wt_backgrounds_values.csv"
    vals.write_text(vals.read_text().replace("12 vs 13 seeds", "12 vs 12 seeds"))
    var = pd.read_csv(OUT / "wt_background_variant_scores.csv")
    st = stability(var)
    st.to_csv(OUT / "stability_streams.csv", index=False)
    print("\nStability, 3 further independent streams of 2,000 splits:")
    print(st.to_string(index=False, float_format=lambda f: f"{f:.4f}"))
