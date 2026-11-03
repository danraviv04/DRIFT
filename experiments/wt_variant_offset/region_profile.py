#!/usr/bin/env python
"""Per residue: where does the WT mean pLDDT fall among the variants' means?

A WT built like the variants sits inside the variant cloud (percentile spread over
0-100). An offset WT sits at the edge (percentile ~0 or ~100) at many residues.
Summaries by 50-residue window and by WT pLDDT band; figure of the profile.

    PYTHONPATH="." python experiments/wt_variant_offset/region_profile.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "si_figures"))
from figlib import cohort  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SAMPLE = re.compile(r"seed-(\d+)_sample-(\d+)")
GENES = ["MSH2", "PIK3CA", "MSH6", "PTEN", "VHL", "POLE"]


def load(path, rows):
    f = pd.read_csv(path, index_col=0)
    return f[[c for c in f.columns if SAMPLE.fullmatch(c)]].to_numpy(float)[rows]


def main():
    out, summ = [], []
    fig, axes = plt.subplots(len(GENES), 1, figsize=(11, 2.1 * len(GENES)))
    for ax, g in zip(axes, GENES):
        c = cohort(g)
        r0 = c.iloc[0]
        rows = (slice(int(r0.plddt_row_range_start), int(r0.plddt_row_range_stop))
                if pd.notna(r0.plddt_row_range_start) else slice(None))
        wt = load(r0.wt_source_file, rows)
        vm = np.stack([x.mean(1) for f in c.source_file if (x := load(f, rows)).shape == wt.shape])
        if len(vm) > 150:
            vm = vm[np.random.default_rng(0).choice(len(vm), 150, replace=False)]
        wm = wt.mean(1)
        pct = 100 * (vm < wm).mean(0)                     # % of variants below WT, per residue
        med = np.median(vm, 0)
        z = (wm - med) / (vm.std(0) + 1e-9)
        res = np.arange(len(wm)) + 1
        d = pd.DataFrame({"gene": g, "residue": res, "wt_mean": wm, "variant_median": med,
                          "wt_minus_variant_median": wm - med, "wt_pct_among_variants": pct, "z": z})
        out.append(d)
        edge = (pct <= 2.5) | (pct >= 97.5)
        band = pd.cut(wm, [0, 50, 70, 90, 101], labels=["<50", "50-70", "70-90", ">=90"])
        summ.append({"gene": g, "residues": len(wm), "n_variants": len(vm),
                     "frac_wt_outside_variant_95": round(float(edge.mean()), 3),
                     "frac_wt_above_all_variants": round(float((pct == 100).mean()), 3),
                     "frac_wt_below_all_variants": round(float((pct == 0).mean()), 3),
                     "median_abs_z": round(float(np.median(np.abs(z))), 2),
                     **{f"edge_frac_wtplddt_{b}": round(float(edge[band == b].mean()), 3) if (band == b).any() else None
                        for b in band.categories}})
        ax.fill_between(res, np.percentile(vm, 2.5, 0) - med, np.percentile(vm, 97.5, 0) - med,
                        color="0.8", lw=0, label="variants, central 95%")
        ax.plot(res, wm - med, color="crimson", lw=0.6, label="WT")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel(f"{g}\nΔ pLDDT vs\nvariant median", fontsize=8)
        ax.text(0.01, 0.92, f"WT outside variant 95% at {100 * edge.mean():.0f}% of residues",
                transform=ax.transAxes, fontsize=8, va="top")
        lim = np.percentile(np.abs(wm - med), 99) * 1.3 + 0.2
        ax.set_ylim(-lim, lim)
        ax.set_xlim(1, len(wm))
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    axes[-1].set_xlabel("residue (scored chain)")
    fig.tight_layout()
    fig.savefig(OUT / "wt_vs_variant_profile.png", dpi=150)
    prof = pd.concat(out)
    prof.to_csv(OUT / "wt_vs_variant_profile.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "wt_vs_variant_profile_summary.csv", index=False)
    print(s.to_string())
    for g in ["MSH2", "PIK3CA"]:
        p = prof[prof.gene == g].copy()
        p["win"] = (p.residue - 1) // 50 * 50 + 1
        w = p.groupby("win").agg(edge=("wt_pct_among_variants", lambda x: ((x <= 2.5) | (x >= 97.5)).mean()),
                                 mean_delta=("wt_minus_variant_median", "mean"), wt_plddt=("wt_mean", "mean"))
        print(f"\n{g} by 50-residue window (edge = frac residues where WT outside variant 95%):")
        print(w.round(2).T.to_string())


if __name__ == "__main__":
    main()
