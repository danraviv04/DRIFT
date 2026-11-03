"""KRAS4B: KS-score and AlphaMissense across HA1E assay bins (blind 935 variants).

Question (SI Note S3 draft, 2026-10-20): is the weak KS-assay correlation the same
U-shape that makes AM's signed correlation weak (AM high at BOTH tails)?
Inputs: experiments/kras4b_coverage/results/{batch_strata.csv, ks/mean_scores.csv}
(hotspot_only excluded, as in every KRAS statistic). Bins as am_by_assay_bin.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "kras4b_coverage" / "results"
BINS = [-3, -0.5, -0.25, 0, 0.1, 0.68, 3]

b = pd.read_csv(SRC / "batch_strata.csv")
b = b[b.stratum != "hotspot_only"]
k = pd.read_csv(SRC / "ks" / "mean_scores.csv")[["mutation", "plddt_ks_residue"]]
m = b.merge(k, left_on="VAR", right_on="mutation", validate="one_to_one")
assert len(m) == 935, len(m)
m["bin"] = pd.cut(m.lfc, BINS)

by_bin = m.groupby("bin", observed=True).agg(
    n=("lfc", "size"), ks_median=("plddt_ks_residue", "median"), ks_mean=("plddt_ks_residue", "mean"),
    am_median=("am", "median"), am_mean=("am", "mean")).reset_index()
by_bin["bin"] = by_bin["bin"].astype(str)
by_bin.to_csv(HERE / "by_bin.csv", index=False)

neutral = m[(m.lfc > -0.25) & (m.lfc <= 0.1)]
depleted = m[m.lfc <= -0.5]
gof = m[m.lfc > 0.68]
rows = []
for name, grp in [("depleted_vs_neutral", depleted), ("gof_vs_neutral", gof)]:
    for col in ["plddt_ks_residue", "am"]:
        u, p = mannwhitneyu(grp[col], neutral[col], alternative="two-sided")
        rows.append(dict(contrast=name, score=col, n_group=len(grp), n_neutral=len(neutral),
                         median_group=grp[col].median(), median_neutral=neutral[col].median(),
                         auc=u / (len(grp) * len(neutral)), mwu_p=p))
for col in ["plddt_ks_residue", "am"]:
    rows.append(dict(contrast="spearman_signed_lfc", score=col, rho=spearmanr(m[col], m.lfc)[0]))
    rows.append(dict(contrast="spearman_abs_lfc", score=col, rho=spearmanr(m[col], m.lfc.abs())[0]))
rows.append(dict(contrast="spearman_ks_am", score="both", rho=spearmanr(m.plddt_ks_residue, m.am)[0]))
pd.DataFrame(rows).to_csv(HERE / "contrasts.csv", index=False)
print(by_bin.round(3).to_string(index=False))
print(pd.DataFrame(rows).round(4).to_string(index=False))
