"""WT-vs-WT KS-score: whole-seed split vs random split (2026-10-17, Dan's recollection check).

For each annotation gene's 125-prediction WT pool (25 seeds x 5 samples): 
- seed split: two disjoint sets of 5 seeds, all 25 predictions of each (N = 25 per side);
- random split: two disjoint random sets of 25 predictions, ignoring seed.
200 repetitions each. Reports median, 95th percentile and s.d. of the WT-WT KS-score.

    PYTHONPATH="." python analyses/seeds/seed_structure/seed_vs_random_split.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from metric_analysis.pipeline.library.canonical_cohort import load_all_canonical_cohorts
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks
from metric_analysis.pipeline.library.descriptor_sampling import apply_row_block, row_block

OUT = Path(__file__).resolve().parent / "results"
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
REPS = 200


def main():
    c = load_all_canonical_cohorts()
    df = c if isinstance(c, pd.DataFrame) else pd.concat(c.values() if isinstance(c, dict) else c)
    rng = np.random.default_rng(20261017)
    rows = []
    for g in GENES:
        r = df[df.protein == g].iloc[0].to_dict()
        w = pd.read_csv(r["wt_source_file"], index_col=0)
        X = apply_row_block(w.to_numpy(float), row_block(r))
        seeds = np.array([int(x.split("_")[0].split("-")[1]) for x in w.columns])
        us = np.unique(seeds)
        out = {"seed_split": [], "random_split": []}
        for _ in range(REPS):
            s = rng.permutation(us)
            a, b = np.isin(seeds, s[:5]), np.isin(seeds, s[5:10])
            out["seed_split"].append(residue_ks(X[:, a], X[:, b]))
            p = rng.permutation(X.shape[1])
            out["random_split"].append(residue_ks(X[:, p[:25]], X[:, p[25:50]]))
        for k, v in out.items():
            v = np.asarray(v)
            rows.append({"gene": g, "split": k, "n_per_side": 25, "median": round(float(np.median(v)), 4),
                         "p95": round(float(np.percentile(v, 95)), 4), "sd": round(float(v.std(ddof=1)), 4)})
    t = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / "seed_vs_random_split.csv", index=False)
    print(t.pivot(index="gene", columns="split", values=["median", "p95", "sd"]).round(3).to_string())


if __name__ == "__main__":
    main()
