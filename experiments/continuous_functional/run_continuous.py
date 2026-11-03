"""Continuous (regression) head on the four functional assays, across ensemble size.

Functional analyses are continuous only (2026-10-13): no assay is thresholded
into a driver/passenger label, so there is no binary head and no AUC here. The
head regresses each gene's assay score on AlphaMissense plus one structural
descriptor, built from the label-free cohort in dataset.py:

  TP53  log10 phenotypic-selection score (Giacomelli 2018)
  KRAS  HA1E WT LFC (Kwon 2024), KRAS4B blind sample
  PTEN  -fitness (Mighell 2018)
  MSH2  LOF score (Jia 2021)

Evaluation
----------
The four targets live on different scales, so pooling them is meaningless.
Three schemes are reported:

  within_gene : KFold(5) inside each gene, out-of-fold predictions, one metric
                per gene.
  within_gene_pos : the same, but folds are grouped by residue position, so no
                two substitutions at one residue are split across the train/test
                boundary. Stricter, and the one to quote.
  logo        : leave-one-gene-out (4 folds) against a WITHIN-GENE percentile-
                ranked target, which is the only way to make the scales
                commensurable. Spearman is invariant to that transform, so the
                reported rho is still against the raw assay. Four folds only --
                a directional check, not an estimate.

Models, all fit on training rows only:

  ridge         RidgeCV on the standardized engineered6 AFM+partner basis
  ridge_raw2    RidgeCV on AFM + partner only
  ridge_afm     AFM alone, rescaled onto the assay
  ridge_partner the structural descriptor alone, same treatment
  afm/partner   raw scores, unfitted reference (R^2 undefined)
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from drift.core import (  # noqa: E402
    fit_fusion_feature_preprocessor_np,
    normalize_fusion_scores_np,
)
import dataset as D  # noqa: E402
import targets as T  # noqa: E402

ALPHAS = np.logspace(-3, 3, 25)
# Results live under the run tag, so the pre-2026-10-13 two-gene / binary-head
# runs in results/ and results_resks/ are never overwritten.
from metric_analysis.pipeline.paths import tagged  # noqa: E402
DEFAULT_OUT = HERE / tagged("results")

# Which two scores the head sees. AFM is always the first column; the second is
# the structural descriptor under test. Both are built from the same cached
# frame, so a pairing costs no extra D1 work.
PAIRINGS = {
    "afm_d1": ("afm_score", "d1_score"),
    # residueKS: mean over residues of the per-residue two-sample KS. This is
    # what the annotation panels and every fusion result use, so it is the one
    # the rebuilt Figure 2 plots. Adding it is nearly free -- the pairings share
    # one cached frame, and only the ridge fits repeat.
    "afm_resks": ("afm_score", "plddt_ks_residue"),
}
PARTNER_LABEL = {"afm_d1": "d1", "afm_resks": "resks"}
# Pooled-OOF metrics and their calibration-free per-fold counterparts.
SUMMARY_METRICS = [
    "spearman", "pearson", "r2",
    "spearman_fold", "pearson_fold", "r2_fold",
]
REPRESENTATION = "engineered6"


def _prep(train_raw, val_raw, groups_tr, groups_va, representation=REPRESENTATION):
    """Exactly logo_cv's preprocessing: normalize scores, build the basis,
    standardize on training rows only."""
    rep_tr = normalize_fusion_scores_np(train_raw, groups_tr, normalization="original")
    rep_va = normalize_fusion_scores_np(val_raw, groups_va, normalization="original")
    Xtr, Xva, _ = fit_fusion_feature_preprocessor_np(
        rep_tr, rep_va, representation=representation,
    )
    return Xtr, Xva


def _standardize(col_tr, col_va):
    """One raw score, centred and scaled on training rows only."""
    mu, sd = float(np.mean(col_tr)), float(np.std(col_tr))
    sd = sd if sd > 1e-12 else 1.0
    return ((col_tr - mu) / sd)[:, None], ((col_va - mu) / sd)[:, None]


def _fit_predict(train_raw, val_raw, groups_tr, groups_va, y_tr, seed):
    """Every head on the same training rows.

    Continuous heads (R^2 meaningful, they predict on the target's own scale):
      ridge       engineered6 AFM+partner basis
      ridge_raw2  AFM+D1 only, no interaction/curvature terms -- shows whether
                  the engineered basis earns its extra parameters on a
                  regression target
      ridge_afm   AFM alone, rescaled onto the assay -- the scale-calibrated
                  version of the raw `afm` baseline, so its R^2 is comparable
      ridge_partner  the structural descriptor alone, same treatment
    """
    Xtr6, Xva6 = _prep(train_raw, val_raw, groups_tr, groups_va, "engineered6")
    Xtr2, Xva2 = _prep(train_raw, val_raw, groups_tr, groups_va, "raw2")
    Xtra, Xvaa = _standardize(train_raw[:, 0], val_raw[:, 0])
    Xtrd, Xvad = _standardize(train_raw[:, 1], val_raw[:, 1])

    preds = {
        "ridge": RidgeCV(alphas=ALPHAS).fit(Xtr6, y_tr).predict(Xva6),
        "ridge_raw2": RidgeCV(alphas=ALPHAS).fit(Xtr2, y_tr).predict(Xva2),
        "ridge_afm": RidgeCV(alphas=ALPHAS).fit(Xtra, y_tr).predict(Xvaa),
        "ridge_partner": RidgeCV(alphas=ALPHAS).fit(Xtrd, y_tr).predict(Xvad),
    }
    return preds


CONTINUOUS_MODELS = ("ridge", "ridge_raw2", "ridge_afm", "ridge_partner")
FITTED_MODELS = CONTINUOUS_MODELS
RAW_MODELS = ("afm", "partner")


def _score(y, pred, r2_valid=False):
    """Both axes for one (gene, model): continuous fit and binary separation."""
    out = {}
    ok = np.isfinite(pred) & np.isfinite(y)
    if ok.sum() < 5 or np.std(pred[ok]) < 1e-12:
        return {"spearman": np.nan, "pearson": np.nan, "r2": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["spearman"] = float(stats.spearmanr(pred[ok], y[ok]).statistic)
        out["pearson"] = float(stats.pearsonr(pred[ok], y[ok])[0])
    # R^2 only means anything when the prediction is on the target's own scale.
    if r2_valid:
        ss_res = float(np.sum((y[ok] - pred[ok]) ** 2))
        ss_tot = float(np.sum((y[ok] - np.mean(y[ok])) ** 2))
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        out["r2"] = np.nan
    return out


def _rows(scheme, size, seed, gene, n, model, scores, pairing="afm_d1"):
    return {
        "scheme": scheme, "pairing": pairing, "size": int(size),
        "seed": int(seed), "gene": gene, "n": int(n), "model": model, **scores,
    }


def _residue_position(mutation):
    """Residue number out of a substitution code, e.g. R175H -> 175."""
    m = re.search(r"(\d+)", str(mutation))
    return int(m.group(1)) if m else -1


def within_gene(df, size, seed, folds=5, by_position=False,
                scheme="within_gene", pairing="afm_d1"):
    """KFold inside each gene, scored two ways.

    `by_position` groups every substitution at one residue into the same fold.
    Plain KFold lets R175H train while R175C is held out; AFM and the structural
    descriptors are both strongly position-driven, so that inflates the apparent
    fit. The grouped variant is the stricter read.

    `spearman` / `r2` are computed on the POOLED out-of-fold vector. Each fold is
    a separately fitted model, so pooling mixes their intercepts and slopes; that
    is the honest cost a globally calibrated predictor must pay, and it is what
    R^2 should be judged on.

    `spearman_fold` etc. are the mean of the per-fold metrics. Within one fold
    there is exactly one model, so these are calibration-free -- the direct
    analogue of the pipeline's `scope='pooled_macro'` AUC, and the fair way to
    compare a fitted head against an unfitted raw score. Without it a ridge on
    AFM alone scores BELOW raw AFM purely from fold-to-fold rescaling.
    """
    rows = []
    for gene, sub in df.groupby("gene", sort=True):
        sub = sub.sort_values("mutation").reset_index(drop=True)
        raw = sub[list(PAIRINGS[pairing])].to_numpy(np.float32)
        y = sub["y"].to_numpy(float)
        g = sub["gene"].to_numpy(str)
        oof = {m: np.full(len(sub), np.nan) for m in FITTED_MODELS}
        per_fold = {m: [] for m in FITTED_MODELS + RAW_MODELS}
        if by_position:
            pos = sub["mutation"].map(_residue_position).to_numpy()
            n_splits = min(folds, len(np.unique(pos)))
            if n_splits < 2:
                continue
            # GroupKFold is deterministic; the seed still varies the D1 ensemble.
            splits = GroupKFold(n_splits=n_splits).split(raw, y, groups=pos)
        else:
            splits = KFold(
                n_splits=min(folds, len(sub)), shuffle=True, random_state=int(seed),
            ).split(raw)
        for tr, va in splits:
            preds = _fit_predict(raw[tr], raw[va], g[tr], g[va], y[tr], seed)
            for m, p in preds.items():
                oof[m][va] = p
                per_fold[m].append(
                    _score(y[va], p, r2_valid=m in CONTINUOUS_MODELS)
                )
            for m, col in (("afm", 0), ("partner", 1)):
                per_fold[m].append(
                    _score(y[va], raw[va, col].astype(float))
                )

        def _mean_fold(model):
            block = pd.DataFrame(per_fold[model])
            return {f"{k}_fold": float(v) for k, v in block.mean().items()}

        for m in FITTED_MODELS:
            scores = _score(y, oof[m], r2_valid=m in CONTINUOUS_MODELS)
            scores.update(_mean_fold(m))
            rows.append(_rows(scheme, size, seed, gene, len(sub), m, scores,
                              pairing))
        for m, col in (("afm", 0), ("partner", 1)):
            scores = _score(y, raw[:, col].astype(float))
            scores.update(_mean_fold(m))
            rows.append(_rows(scheme, size, seed, gene, len(sub), m, scores,
                              pairing))
    return rows


def logo(df, size, seed, pairing="afm_d1"):
    """Leave-one-gene-out against a within-gene percentile-ranked target.

    Ranking within gene is what makes TP53's log-selection score and KRAS's LFC
    commensurable for a single shared regression head. Spearman is invariant to
    it, so the reported rho on the held-out gene is still rho against that gene's
    raw assay value.
    """
    d = df.sort_values(["gene", "mutation"]).reset_index(drop=True).copy()
    d["y_rank"] = d.groupby("gene")["y"].rank(pct=True)
    raw = d[list(PAIRINGS[pairing])].to_numpy(np.float32)
    yr = d["y_rank"].to_numpy(float)
    y = d["y"].to_numpy(float)
    g = d["gene"].to_numpy(str)
    rows = []
    for gene in sorted(d["gene"].unique()):
        va = np.flatnonzero(g == gene)
        tr = np.flatnonzero(g != gene)
        if len(np.unique(g[tr])) < 1 or len(tr) < 10:
            continue
        preds = _fit_predict(raw[tr], raw[va], g[tr], g[va], yr[tr], seed)
        for m, p in preds.items():
            # A rank target is unitless: R^2 against the raw assay is not defined.
            s = _score(y[va], p)
            s["r2"] = np.nan
            # One fold IS one gene here, so the pooled and per-fold reads coincide.
            s.update({f"{k}_fold": v for k, v in s.items()})
            rows.append(_rows("logo", size, seed, gene, len(va), m, s, pairing))
        for m, col in (("afm", 0), ("partner", 1)):
            s = _score(y[va], raw[va, col].astype(float))
            s["r2"] = np.nan
            s.update({f"{k}_fold": v for k, v in s.items()})
            rows.append(_rows("logo", size, seed, gene, len(va), m, s, pairing))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="1,5,10,20,40,60,90")
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pairings", default=",".join(PAIRINGS),
                    help=f"comma list from {list(PAIRINGS)}")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true",
                    help="rebuild the continuous descriptor cache (dataset.py)")
    args = ap.parse_args()

    pairings = [x.strip() for x in args.pairings.split(",") if x.strip()]
    unknown = sorted(set(pairings) - set(PAIRINGS))
    if unknown:
        ap.error(f"unknown pairing(s) {unknown}; choose from {list(PAIRINGS)}")
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[continuous] sizes={sizes} seeds={seeds} folds={args.folds}")
    print("[continuous] pairings: "
          + "  ".join(f"{p}={PAIRINGS[p]}" for p in pairings))
    print(f"[continuous] cohort = targets.CONTINUOUS_GENES = "
          f"{list(T.CONTINUOUS_GENES)}\n")

    rows = []
    for size in sizes:
        for seed in seeds:
            fun = D.load(size, seed, force=args.force)
            for pairing in pairings:
                rows += within_gene(fun, size, seed, folds=args.folds,
                                    pairing=pairing)
                rows += within_gene(fun, size, seed, folds=args.folds,
                                    by_position=True,
                                    scheme="within_gene_pos", pairing=pairing)
                rows += logo(fun, size, seed, pairing=pairing)
            frame = pd.DataFrame(rows).query(
                "scheme=='within_gene_pos' and model=='ridge_raw2' "
                f"and size=={size} and seed=={seed}"
            )
            got = {
                f"{p}/{g}": f"{v:.3f}"
                for p, g, v in zip(frame["pairing"], frame["gene"],
                                   frame["spearman_fold"])
            }
            print(f"  [continuous] size={size:>2} seed={seed} "
                  f"ridge rho(per-fold): {got}")

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "continuous_runs.csv", index=False)

    summ = (
        runs.groupby(["scheme", "pairing", "model", "gene", "size"])
        [SUMMARY_METRICS + ["n"]]
        .agg(["mean", "std", "count"])
    )
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "continuous_summary.csv", index=False)

    macro = (
        runs.groupby(["scheme", "pairing", "model", "size", "seed"])
        [["spearman", "spearman_fold", "r2_fold"]].mean().reset_index()
        .groupby(["scheme", "pairing", "model", "size"])[["spearman", "spearman_fold", "r2_fold"]]
        .agg(["mean", "std"])
    )
    macro.columns = [f"{a}_{b}" for a, b in macro.columns]
    macro = macro.reset_index()
    macro.to_csv(out / "continuous_macro.csv", index=False)

    print(f"\n[continuous] wrote {out/'continuous_runs.csv'} ({len(runs)} rows)")
    print(f"[continuous] wrote {out/'continuous_summary.csv'}")
    print(f"[continuous] wrote {out/'continuous_macro.csv'}")
    print("\n[continuous] macro over the four continuous genes "
          "(mean over seeds):\n")
    for scheme in ("within_gene", "within_gene_pos", "logo"):
        block = macro[macro.scheme == scheme]
        if block.empty:
            continue
        print(f"--- {scheme} ---")
        for pairing in pairings:
            sub = block[block.pairing == pairing]
            if sub.empty:
                continue
            print(f"  [{pairing}]")
            print(sub.pivot(index="size", columns="model",
                            values="spearman_fold_mean").round(3).to_string())
        continue
        for metric in ("spearman_fold_mean", "spearman_mean"):
            print(f"  Spearman rho vs continuous assay [{metric}]:")
            print(block.pivot(index="size", columns="model", values=metric)
                  .round(3).to_string())
        print()


if __name__ == "__main__":
    main()
