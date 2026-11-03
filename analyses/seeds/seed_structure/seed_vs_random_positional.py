"""Positional version of seed_vs_random_split.py (2026-10-17, Dan: "positional variance?").

WT pools in the core-fit frame (cache_corefit), 125 structures = 25 seeds x 5 samples.
Ensembles of 25 structures, built two ways, 200 repetitions each:
- whole seeds: all 5 samples of 5 seeds;
- random: 25 structures drawn without regard to seed.
Reports, over CORE residues:
- within-ensemble positional spread: mean per-residue RMSF of one ensemble (and, for
  comparison, the within-ensemble pLDDT s.d.);
- between-ensemble difference: positional KS between two disjoint ensembles of the same
  kind (distance of each C-alpha from the full-pool WT centre, KS per residue, mean).

    PYTHONPATH="." python analyses/seeds/seed_structure/seed_vs_random_positional.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_analysis.pipeline.library.descriptor_metrics import residue_ks

REPO = Path(__file__).resolve().parents[3]
SM = REPO / "metric_analysis" / "structure_metrics"
OUT = Path(__file__).resolve().parent / "results"
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
REPS = 200


def load(g):
    d = SM / g / "ensemble_aware" / "cache_corefit"
    wt = json.loads((d / f"{g}_core_fit_report.json").read_text())["wt_artifact"]
    x = pd.read_parquet(d / f"{g}_rmsf_aligned_ca_coordinate_cache.parquet",
                        filters=[("artifact", "==", wt)])
    x = x[x.core]
    x["sample"] = x["sample"].astype(str)
    if not x["sample"].str.startswith("seed-").all():
        return None, None   # MSH2/PIK3CA regenerated WTs: added by stream_align, no seed-sample labels
    piv = {a: x.pivot_table(index="resid", columns="sample", values=a, observed=True)
           for a in ("x_aligned", "y_aligned", "z_aligned")}
    cols = piv["x_aligned"].columns
    C = np.stack([piv[a][cols].to_numpy() for a in ("x_aligned", "y_aligned", "z_aligned")], -1)  # R x S x 3
    seeds = np.array([int(c.split("_")[0].split("-")[1]) for c in cols])
    return C, seeds


def rmsf(C):
    return float(np.sqrt(((C - C.mean(1, keepdims=True)) ** 2).sum(-1).mean(1)).mean())


def main():
    rng = np.random.default_rng(20261017)
    rows = []
    for g in GENES:
        C, seeds = load(g)
        if C is None:
            print(f"  {g}: WT coordinate cache has no seed-sample labels, skipped")
            continue
        ok = np.isfinite(C).all((1, 2))
        C = C[ok]
        D = np.linalg.norm(C - C.mean(1, keepdims=True), axis=-1)   # distance from the full-pool WT centre
        us = np.unique(seeds)
        res = {k: {"rmsf": [], "posks": []} for k in ("whole_seeds", "random")}
        for _ in range(REPS):
            s = rng.permutation(us)
            a, b = np.isin(seeds, s[:5]), np.isin(seeds, s[5:10])
            res["whole_seeds"]["rmsf"].append(rmsf(C[:, a]))
            res["whole_seeds"]["posks"].append(residue_ks(D[:, a], D[:, b]))
            p = rng.permutation(len(seeds))
            res["random"]["rmsf"].append(rmsf(C[:, p[:25]]))
            res["random"]["posks"].append(residue_ks(D[:, p[:25]], D[:, p[25:50]]))
        for k, v in res.items():
            rows.append({"gene": g, "split": k, "core_residues": int(ok.sum()),
                         "rmsf_mean_A": round(float(np.mean(v["rmsf"])), 4),
                         "rmsf_sd_A": round(float(np.std(v["rmsf"], ddof=1)), 4),
                         "posks_median": round(float(np.median(v["posks"])), 4),
                         "posks_p95": round(float(np.percentile(v["posks"], 95)), 4),
                         "posks_sd": round(float(np.std(v["posks"], ddof=1)), 4)})
    t = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / "seed_vs_random_positional.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
