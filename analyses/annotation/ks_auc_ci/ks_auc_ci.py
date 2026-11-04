#!/usr/bin/env python
"""Per-gene KS-score ROC-AUC at N=90 with 95% CIs, v6_ca (2026-10-15).

Hierarchical bootstrap as described in Methods ("Statistical analysis"): 2,000 reps;
positives and negatives resampled separately within each gene; the same variant draw is
applied to every descriptor seed; the ten seed indices are resampled with replacement and
the per-seed AUCs averaged. Reads the DRIFT feature caches of fingerprint d61d29328362.

    python -I analyses/annotation/ks_auc_ci/ks_auc_ci.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "drift/cache"
FP, SEEDS, REPS = "d61d29328362", range(42, 52), 2000
GENES = ("POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA")
OUT = Path(__file__).resolve().parent / "results" / "ks_auc_ci_v6.csv"


def main():
    frames = [pd.read_csv(CACHE / f"labeled_size90_seed{s}_proteins{FP}_esfeat7.csv")
              .assign(seed=s) for s in SEEDS]
    d = pd.concat(frames)
    d = d[d.gene.isin(GENES)]
    rng = np.random.default_rng(20261015)
    rows = []
    for g in GENES:
        w = d[d.gene == g].pivot_table(index=["mutation", "label"], columns="seed",
                                       values="plddt_ks_residue").dropna()
        y = w.index.get_level_values("label").to_numpy().astype(int)
        X = w.to_numpy()                      # variants x seeds
        obs = np.mean([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        boot = np.empty(REPS)
        for b in range(REPS):
            idx = np.r_[rng.choice(pos, pos.size), rng.choice(neg, neg.size)]
            cols = rng.integers(0, X.shape[1], X.shape[1])
            yy = y[idx]
            boot[b] = np.mean([roc_auc_score(yy, X[idx, j]) for j in cols])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(gene=g, n=len(y), pos=pos.size, neg=neg.size, auc=obs, lo=lo, hi=hi))
    r = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT, index=False)
    print(r.round(3).to_string(index=False))
    print("macro", round(r.auc.mean(), 4))


if __name__ == "__main__":
    main()
