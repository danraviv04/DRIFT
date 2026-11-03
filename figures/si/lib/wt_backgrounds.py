#!/usr/bin/env python
"""SI figure: wild-type backgrounds per gene, and the per-residue test calibration.

WHAT IT ESTABLISHES
-------------------
Backs Note S4, which main Fig. 1e,f shows for PTEN only.
- Panel a: for each annotation gene, where the variants' KS-scores sit against
  the spread of KS between two halves of the same wild-type pool. That spread is
  the score that prediction noise alone produces. Answers whether a variant is
  detectable above that noise in every gene, or only in PTEN.
- Panel b: in every gene, a raw per-residue Mann-Whitney test of WT against WT
  rejects more often than its nominal 5% once the halves are split by seed. This
  is why Fig. 1e compares each residue's P with a WT-WT empirical distribution
  rather than using raw P values.

DESIGN
------
Every gene uses its canonical 125-prediction WT pool (25 AF3 seeds x 5 samples);
PTEN too (user, 2026-10-15), although Fig. 1f uses its 1,000-prediction pool.
- WT-WT, seed-matched: the 25 seeds split 12 vs 13, and 60 predictions per side.
  The diffusion samples of a seed are correlated, so this keeps the seed
  structure of a real comparison. Each side has fewer seeds than a real 25-seed
  run, so the background is WIDER than the true one. A variant above it would be
  above the true one; one below it is inconclusive.
- WT-WT, random: 60 vs 60 predictions split at random, with seeds mixed across sides
  (panel b only; it shows the too-narrow background the seed split avoids).
- 200 splits per gene. MSH2 and MSH6 use their own residue blocks (manifest row
  ranges): MSH2 the MutS-alpha WT, MSH6 the G:T-mismatch WT.
- Variant scores: the production residue KS at N = 60, from the fusion caches of
  the current protein config (v6_ca, d61d29328362), mean over seeds 42-51. These
  are the same scores as every other N = 60 number.

v6_ca (2026-10-16): the MSH2 and PIK3CA wild types were regenerated with their
variants' templates. Under the old WTs every MSH2 and PIK3CA variant sat above
the WT-WT 95th percentile, an artefact of the template mismatch
(analyses/provenance/wt_variant_offset); with the v6 WTs those genes look like the others
(MSH2 42/71 drivers and 3/38 passengers above it; PIK3CA 38/84 and 1/7).

SOURCE
------
WT pLDDT CSVs from the strict manifest (figlib.cohort; the v6 WTs for MSH2 and
PIK3CA); variant scores from the current caches (figlib.labeled). POST the strict cutover. The MSH6 numbers can be compared
with analyses/msh6_case_study/wt_pool/wt_null_12v13.json (same design, other RNG).

    python figures/si/lib/wt_backgrounds.py
"""
from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: F403,E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from metric_analysis.pipeline.library.descriptor_metrics import residue_ks  # noqa: E402

N = 60
SPLITS = 200
CACHE_SEEDS = tuple(range(42, 52))
BASE_SEED = 20261015
SAMPLE = re.compile(r"seed-(\d+)_sample-\d+")
INK = "#222222"


def wt_pool(gene):
    """(residues x predictions) WT pLDDT for the gene's scored residues, and each column's seed."""
    rec = cohort(gene).iloc[0]
    f = pd.read_csv(rec.wt_source_file, index_col=0)
    cols = [c for c in f.columns if SAMPLE.fullmatch(c)]
    x = f[cols].to_numpy(float)
    if pd.notna(rec.plddt_row_range_start):
        x = x[int(rec.plddt_row_range_start):int(rec.plddt_row_range_stop)]
    if x.shape[0] != int(rec.n_wt_residues):
        raise ValueError(f"{gene}: {x.shape[0]} WT residues, manifest says {rec.n_wt_residues}")
    seeds = np.array([int(SAMPLE.fullmatch(c).group(1)) for c in cols])
    return x, seeds, Path(rec.wt_source_file).name


def seed_split(seeds, rng):
    """Columns of two disjoint seed sets, 12 and 13 seeds, N predictions each."""
    s = rng.permutation(np.unique(seeds))
    half = len(s) // 2
    a = np.flatnonzero(np.isin(seeds, s[:half]))
    b = np.flatnonzero(np.isin(seeds, s[half:]))
    if min(len(a), len(b)) < N:
        raise ValueError(f"a seed side has {min(len(a), len(b))} predictions, need {N}")
    return rng.choice(a, N, replace=False), rng.choice(b, N, replace=False)


def random_split(n_cols, rng):
    o = rng.permutation(n_cols)
    return o[:N], o[N:2 * N]


def frac_p05(x, a, b):
    """Fraction of residues whose raw Mann-Whitney P (WT vs WT) is below 0.05.
    Residues where every value ties give no test and count as not rejected."""
    with np.errstate(all="ignore"):
        p = mannwhitneyu(x[:, a], x[:, b], axis=1).pvalue
    return float(np.mean(np.nan_to_num(p, nan=1.0) < 0.05))


def backgrounds(gene):
    x, seeds, name = wt_pool(gene)
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(gene.encode()))
    rows = []
    for i in range(SPLITS):
        a, b = seed_split(seeds, rng)
        rows.append({"gene": gene, "split": i, "kind": "seed", "ks": residue_ks(x[:, a], x[:, b]),
                     "frac_p05": frac_p05(x, a, b)})
        a, b = random_split(x.shape[1], rng)
        rows.append({"gene": gene, "split": i, "kind": "random", "ks": residue_ks(x[:, a], x[:, b]),
                     "frac_p05": frac_p05(x, a, b)})
    return pd.DataFrame(rows), name, x.shape


def variant_scores():
    parts = []
    for seed in CACHE_SEEDS:
        d = labeled(size=N, seed=seed, genes=ANNOTATION_GENES)
        check_cohort(d, verbose=False)
        parts.append(d[["gene", "mutation", "label", "plddt_ks_residue"]])
    v = (pd.concat(parts).groupby(["gene", "mutation", "label"], as_index=False)
         .plddt_ks_residue.mean().rename(columns={"plddt_ks_residue": "ks"}))
    check_cohort(v, verbose=False)
    return v


def panel_backgrounds(bg, var, summ):
    """Main Fig. 1f's layout for every gene: WT vs WT, passengers, drivers."""
    fig, axes = grid(2, 3, w=W2, h=W2 * 0.66)
    fig.subplots_adjust(hspace=0.6, wspace=0.32)
    for ax, g in zip(axes.flat, ANNOTATION_GENES):
        null = bg[(bg.gene == g) & (bg.kind == "seed")].ks.to_numpy()
        v = var[var.gene == g]
        class_violin(ax, [(null, "WT vs WT", WTC), (v[v.label == 0].ks, "Passenger", PAS),
                          (v[v.label == 1].ks, "Driver", DRV)],
                     p95=summ[g]["p95"], p95_label=None, seed=BASE_SEED + zlib.crc32(g.encode()),
                     title=f"{pretty(g)}    driver vs passenger AUC {summ[g]['auc']:.2f}")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"KS-score (N = {N})")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls=(0, (3, 2)), lw=.7, color=GRY,
                               label="WT–WT 95th percentile (seed-matched, 12 vs 13 seeds)")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "wt_background_per_gene")


def panel_calibration(bg):
    t = (bg.groupby(["gene", "kind"]).frac_p05
         .agg(median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
         .reset_index())
    fig, ax = panel(w=W1, h=W1 * 0.62)
    x = np.arange(len(ANNOTATION_GENES))
    w = 0.36
    for off, kind, colour, label in [(-w / 2, "seed", INK, "Seed-matched split"),
                                     (w / 2, "random", GRY, "Random split")]:
        s = t[t.kind == kind].set_index("gene").loc[ANNOTATION_GENES]
        err = 100 * np.vstack([s["median"] - s["q1"], s["q3"] - s["median"]])
        ax.bar(x + off, 100 * s["median"], w, yerr=err, color=colour, label=label,
               error_kw={"lw": 0.6, "capsize": 1.5})
        for g, r in s.iterrows():
            record("per_residue_calibration", f"{g}_{kind}_pct_p05", round(100 * r["median"], 2),
                   f"median over {SPLITS} WT-WT splits, IQR {100 * r['q1']:.1f}-{100 * r['q3']:.1f}, N={N} per side")
    ax.axhline(5, color=DRV, lw=0.7, ls="--", label="Nominal 5%")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(g) for g in ANNOTATION_GENES])
    ax.set_ylabel("WT-WT residues with raw P < 0.05 (%)\nmedian and IQR over splits")
    # above the axes: the PIK3CA and POLE seed-split IQRs reach ~40% and ran into it
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, "per_residue_calibration")


def main():
    parts, pools = [], {}
    for g in ANNOTATION_GENES:
        b, name, shape = backgrounds(g)
        parts.append(b)
        pools[g] = {"wt_file": name, "residues": shape[0], "predictions": shape[1]}
        print(f"  {g}: WT {name} {shape}")
    bg = pd.concat(parts, ignore_index=True)
    bg.to_csv(RESULTS / "wt_background_splits.csv", index=False)
    var = variant_scores()
    var.to_csv(RESULTS / "wt_background_variant_scores.csv", index=False)

    summ = {}
    for g in ANNOTATION_GENES:
        null = bg[(bg.gene == g) & (bg.kind == "seed")].ks
        v = var[var.gene == g]
        p95 = float(np.percentile(null, 95))
        d, p = v[v.label == 1].ks, v[v.label == 0].ks
        summ[g] = {"p95": p95, "median": float(null.median()),
                   "auc": float(roc_auc_score(v.label, v.ks)),
                   "n_drv": len(d), "drv_above": int((d > p95).sum()),
                   "n_pass": len(p), "pass_above": int((p > p95).sum())}
        record("wt_background_per_gene", f"{g}_wtwt_p95", round(p95, 4),
               f"seed-matched 12 vs 13 seeds, N={N}, {SPLITS} splits; median {null.median():.4f}")
        record("wt_background_per_gene", f"{g}_drivers_above_p95", f"{summ[g]['drv_above']}/{len(d)}", "")
        record("wt_background_per_gene", f"{g}_passengers_above_p95", f"{summ[g]['pass_above']}/{len(p)}", "")
    panel_backgrounds(bg, var, summ)
    panel_calibration(bg)
    write_values("wt_backgrounds_values.csv")
    provenance(figure="wt_backgrounds", n_per_side=N, splits=SPLITS, base_seed=BASE_SEED,
               cache_seeds=list(CACHE_SEEDS), wt_pools=pools, cohort="annotation (1,798)")
    print(pd.DataFrame(summ).T.to_string())


if __name__ == "__main__":
    main()
