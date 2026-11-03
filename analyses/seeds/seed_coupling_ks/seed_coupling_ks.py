"""Does sharing seed numbers lower the WT-mutant KS-score? (Note S6 claim (4), and
Note S2's "this matches the structure of a wild-type--mutant comparison".)

Per gene, 30 random variants (125-prediction pools), 20 repetitions, whole seeds,
N = 60 per side (12 seeds x 5 samples), the 12-vs-12 design of Note S2:
  coupled   WT seeds S1  vs mutant seeds S1 (same seed numbers, as in production)
  disjoint  WT seeds S1  vs mutant seeds S2 (S1, S2 disjoint 12-seed sets)
  wt_wt     WT seeds S1  vs WT seeds S2 (the Note S2 background)
Reports medians and the paired coupled - disjoint difference.

    cd <repo>; PYTHONPATH="." python \
      analyses/seeds/seed_coupling_ks/seed_coupling_ks.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from metric_analysis.pipeline.library.canonical_cohort import load_all_canonical_cohorts
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks
from metric_analysis.pipeline.library.descriptor_sampling import apply_row_block, row_block

OUT = Path(__file__).resolve().parent
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
NV, REPS = 30, 20
rng = np.random.default_rng(20261020)
c = load_all_canonical_cohorts()
rows = []


def seed_of(cols):
    return np.array([int(x.split("_")[0].split("-")[1]) for x in cols])


for g in GENES:
    recs = c[(c.protein == g) & (c.n_mutant_structures == 125)]
    r0 = recs.iloc[0].to_dict()
    wf = pd.read_csv(r0["wt_source_file"], index_col=0)
    W = apply_row_block(wf.to_numpy(float), row_block(r0))
    ws = seed_of(wf.columns)
    us = np.unique(ws)
    pick = recs.sample(min(NV, len(recs)), random_state=5)
    mats = []
    for r in pick.to_dict("records"):
        mf = pd.read_csv(r["source_file"], index_col=0)
        assert sorted(mf.columns) == sorted(wf.columns)
        mats.append((r["mutation_code"], r["label"], apply_row_block(mf.to_numpy(float), row_block(r)), seed_of(mf.columns)))
    for rep in range(REPS):
        p = rng.permutation(us)
        a, b = np.isin(ws, p[:12]), np.isin(ws, p[12:24])
        wtwt = residue_ks(W[:, a], W[:, b])
        for m, lab, M, ms in mats:
            ma, mb = np.isin(ms, p[:12]), np.isin(ms, p[12:24])
            rows.append({"gene": g, "mutation": m, "label": lab, "rep": rep,
                         "coupled": residue_ks(W[:, a], M[:, ma]),
                         "disjoint": residue_ks(W[:, a], M[:, mb]), "wt_wt": wtwt})
t = pd.DataFrame(rows)
t.to_csv(OUT / "seed_coupling_ks.csv.gz", index=False, compression="gzip")
t["diff"] = t.coupled - t.disjoint
s = t.groupby("gene").agg(coupled_median=("coupled", "median"), disjoint_median=("disjoint", "median"),
                          diff_median=("diff", "median"), frac_coupled_lower=("diff", lambda x: float((x < 0).mean())),
                          wtwt_median=("wt_wt", "median"), wtwt_p95=("wt_wt", lambda x: float(np.percentile(x.unique(), 95))))
s = s.round(4)
s.to_csv(OUT / "seed_coupling_ks_summary.csv")
print(s.to_string())
