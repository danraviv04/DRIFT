"""Figure-2-style descriptive KS/assay associations for the four functional genes.

Reads the label-free continuous cohort (dataset.py) at N=90, seeds 42-51:
TP53, KRAS (KRAS4B blind sample), PTEN, MSH2. Every target is oriented so that
higher = stronger cancer-relevant effect (PTEN is -fitness); see targets.py.
No binary label enters anything here.
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import dataset as D  # noqa: E402
import targets as T  # noqa: E402

SCORE = "plddt_ks_residue"
SIZE = 90
SEEDS = tuple(range(42, 52))
OUT = HERE / "four_gene_ks"
GENES = T.CONTINUOUS_GENES
LABELS = {"TP53": "log₁₀ phenotypic-selection score", "KRAS": "HA1E WT log₂ fold change",
          "PTEN": "PTEN impairment score (−fitness)", "MSH2": "Measured LOF score"}
COLORS = {"TP53": "#b44f68", "KRAS": "#168b82", "PTEN": "#6a4c93", "MSH2": "#2a78d6"}
TITLES = {"KRAS": "KRAS (KRAS4B blind sample)"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def associations(d, subset, xcol=SCORE):
    x, y = d[xcol].to_numpy(), d.y.to_numpy()
    fit = stats.linregress(x, y)
    pearson, spearman = stats.pearsonr(x, y), stats.spearmanr(x, y)
    # Bootstrap complete residue groups, preserving substitutions at a site.
    rng = np.random.default_rng(42)
    groups = [b.index.to_numpy() for _, b in d.groupby("position")]
    boot = []
    for _ in range(2000):
        b = d.loc[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])]
        if b[xcol].nunique() > 1 and b.y.nunique() > 1:
            r = np.corrcoef(b[xcol], b.y)[0, 1]
            boot.append((r, stats.spearmanr(b[xcol], b.y).statistic, r*r))
    ci = np.quantile(boot, [.025, .975], axis=0)
    result = dict(gene=d.gene.iloc[0], subset=subset, n=len(d), n_positions=len(groups),
                  pearson_r=pearson.statistic, pearson_p_iid=pearson.pvalue,
                  spearman_rho=spearman.statistic, spearman_p_iid=spearman.pvalue,
                  ols_r_squared=fit.rvalue**2, slope=fit.slope, intercept=fit.intercept)
    for j, name in enumerate(("pearson_r", "spearman_rho", "ols_r_squared")):
        result[name+"_ci_low"], result[name+"_ci_high"] = ci[:, j]
    residual_r2 = 1 - np.sum((y - (fit.intercept + fit.slope*x))**2)/np.sum((y-y.mean())**2)
    assert np.isclose(residual_r2, result["ols_r_squared"])
    return result


def seed_mean_scores():
    """One row per variant: descriptors and target, KS averaged over the seeds."""
    frames, sources = [], []
    for seed in SEEDS:
        frames.append(D.load(SIZE, seed).assign(seed=seed))
        path = D.cache_path(SIZE, seed)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
    runs = pd.concat(frames, ignore_index=True)
    keys = ["gene", "mutation"]
    first = runs.drop_duplicates(keys).set_index(keys)
    scores = runs.groupby(keys).agg(**{SCORE: (SCORE, "mean"), "n_seeds": ("seed", "nunique")})
    assert scores.n_seeds.eq(len(SEEDS)).all()
    keep = ["afm_score", "y", "y_raw", "high_conf", "known_splice_disruptive"]
    out = scores.join(first[[c for c in keep if c in first.columns]]).reset_index()
    out["position"] = out.mutation.str.extract(r"(\d+)")[0].astype(int)
    return out, sources


def main():
    d, sources = seed_mean_scores()
    records = [associations(d[d.gene.eq(g)].copy(), "all_measured") for g in GENES]
    records.append(associations(d[d.gene.eq("PTEN") & d.high_conf.eq(True)].copy(), "high_confidence_only"))
    records.append(associations(d[d.gene.eq("MSH2") & d.known_splice_disruptive.eq(False)].copy(),
                                "excluding_known_splice_disruptive"))
    raw_tp53 = d[d.gene.eq("TP53")].copy()
    raw_tp53["y"] = raw_tp53.y_raw
    records.append(associations(raw_tp53, "raw_score_sensitivity"))
    summary = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "variant_scores.csv", index=False)
    summary.to_csv(OUT / "associations.csv", index=False)
    for name in ("Ground_Truths/Supplementary table 3.xlsx", "Ground_Truths/KRAS/Extended Data Table 1.xlsx",
                 "Ground_Truths/PTEN/functional_scores.csv", "Ground_Truths/MSH2/functional_scores.csv"):
        sources.append({"path": name, "sha256": sha(ROOT / name)})
    (OUT / "provenance.json").write_text(json.dumps({
        "sources": sources, "N": SIZE, "seeds": list(SEEDS),
        "cohort": "analyses/continuous/continuous_functional/dataset.py cohort(): TP53 all modelled "
                  "and scored; KRAS4B blind stratum; PTEN/MSH2 canonical cohort with a measured "
                  "score; position-1 excluded. No binary labels.",
        "orientation": "TP53 log10 selection, KRAS signed LFC, PTEN -Cum_score, MSH2 LOF score",
        "CI": "2000 residue-cluster bootstrap draws, seed 42; percentile 95% intervals",
        "p_values": "Unadjusted conventional IID two-sided tests; substitutions at the same residue may be dependent",
        "R2": "Descriptive in-sample OLS with intercept, not cross-validated prediction"}, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))
    for ax, gene, rec in zip(axes.flat, GENES, records):
        b = d[d.gene.eq(gene)]
        ax.scatter(b[SCORE], b.y, s=15, alpha=.42 if len(b) > 200 else .75,
                   color=COLORS[gene], linewidths=0)
        if gene == "PTEN":
            flagged = b[b.high_conf.eq(False)]
            ax.scatter(flagged[SCORE], flagged.y, marker="x", color="#30343b", s=28, label="Low confidence")
        if gene == "MSH2":
            flagged = b[b.known_splice_disruptive.eq(True)]
            ax.scatter(flagged[SCORE], flagged.y, marker="x", color="#30343b", s=28, label="Splice-disruptive flag")
        xs = np.array([b[SCORE].min(), b[SCORE].max()])
        ax.plot(xs, rec["intercept"]+rec["slope"]*xs, color="#30343b", lw=1.4)
        ax.set_title(f"{TITLES.get(gene, gene)}  ·  n = {len(b):,}", loc="left", fontweight="bold")
        ax.text(.97, .97, f"Pearson r = {rec['pearson_r']:+.3f}\nOLS R² = {rec['ols_r_squared']:.3f}"
                f"\nSpearman ρ = {rec['spearman_rho']:+.3f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(facecolor="white", alpha=.88, edgecolor="none"))
        ax.set_xlabel("Mean residue-wise KS")
        ax.set_ylabel(LABELS[gene])
        ax.grid(alpha=.16)
        ax.margins(y=.20)
        if gene in ("PTEN", "MSH2"):
            ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("KS versus continuous functional scores", fontsize=16, y=.985)
    fig.text(.5, .944, f"N = {SIZE} structures per ensemble · mean over {len(SEEDS)} subsampling draws · "
             "one point per variant", ha="center", fontsize=10)
    fig.text(.5, .018, "Descriptive linear fits; R² is not held-out performance. "
             "PTEN sign reversed; assay units remain gene-specific.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .925], h_pad=2, w_pad=2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"KS_vs_functional_scores.{ext}", dpi=220)
    plt.close(fig)
    print(summary[["gene", "subset", "n", "pearson_r", "ols_r_squared", "spearman_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
