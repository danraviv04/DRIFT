"""Between-seed share of WT pLDDT variance (2026-10-17, for SI Note S5).

Per residue of each annotation gene's 125-prediction WT pool (25 seeds x 5 diffusion
samples), the one-way ANOVA intraclass correlation with seed as the group:
ICC = (MSB - MSW) / (MSB + (n - 1) MSW), n = 5. It is the share of per-residue pLDDT
variance that lies between seeds; the rest is between samples of one seed.
Cross-sequence seed coupling (same seed-sample column in WT and variant jobs) is in
analyses/provenance/wt_variant_offset/results/seed_pairing.json.

    PYTHONPATH="." python analyses/seeds/seed_structure/seed_icc.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from metric_analysis.pipeline.library.canonical_cohort import load_all_canonical_cohorts
from metric_analysis.pipeline.library.descriptor_sampling import apply_row_block, row_block

OUT = Path(__file__).resolve().parent / "results"
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]


def icc(X, seeds):
    us = np.unique(seeds)
    k, n = len(us), X.shape[1] / len(us)
    means = np.stack([X[:, seeds == u].mean(1) for u in us], 1)
    msb = n * ((means - X.mean(1, keepdims=True)) ** 2).sum(1) / (k - 1)
    msw = sum(((X[:, seeds == u] - means[:, [i]]) ** 2).sum(1) for i, u in enumerate(us)) / (X.shape[1] - k)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (msb - msw) / (msb + (n - 1) * msw)
    return v[np.isfinite(v) & (msb + msw > 0)]


def main():
    c = load_all_canonical_cohorts()
    df = c if isinstance(c, pd.DataFrame) else pd.concat(c.values() if isinstance(c, dict) else c)
    rows = []
    for g in GENES:
        r = df[df.protein == g].iloc[0].to_dict()
        w = pd.read_csv(r["wt_source_file"], index_col=0)
        seeds = np.array([int(x.split("_")[0].split("-")[1]) for x in w.columns])
        v = icc(apply_row_block(w.to_numpy(float), row_block(r)), seeds)
        rows.append({"gene": g, "residues": len(v), "seeds": len(np.unique(seeds)), "predictions": w.shape[1],
                     "icc_median": round(float(np.median(v)), 3), "icc_q25": round(float(np.percentile(v, 25)), 3),
                     "icc_q75": round(float(np.percentile(v, 75)), 3), "wt_file": r["wt_source_file"]})
    OUT.mkdir(exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "seed_icc.csv", index=False)
    print(t.drop(columns="wt_file").to_string(index=False))


if __name__ == "__main__":
    main()
