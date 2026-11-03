"""Clean, auditable AFM+D1 benchmark used by :mod:`drift.run`.

This module intentionally contains only linear models.  It compares three
fusion definitions under the same outer leave-one-gene-out (LOGO) splits:

``raw2``
    Logistic regression over the two raw inputs, AFM and D1.
``compact4``
    AFM, D1, D1 squared, and AFM-by-D1; a compact nonlinear basis with no
    exactly redundant signed-difference feature. Retained as the legacy basis.
``compact4_v2``
    AFM, D1, AFM-by-D1, and squared AFM/D1 disagreement; a stable four-term
    subset of the successful historical engineered basis.
``engineered6``
    The project's existing six-feature logistic regression.
``offset``
    AFM log-odds plus a monotonic D1 term.  The AFM coefficient is fixed at one
    and D1's coefficient is constrained non-negative, so adding D1 cannot learn
    to reverse AlphaMissense wholesale.

Every family can be evaluated with global class balancing (the internal
``variant`` option) or with equal-gene, within-gene class balancing (the
internal ``gene`` option).  Both change sample weights only; neither discards
observations.  Label-free within-gene robust scaling and
percentile ranks are optional representations.  Because those transformations
use the full distribution of scores in the held-out gene (but never its labels),
they are transductive batch normalisations and are named as such in the output.

The fixed grid retains those transductive representations as explicit stress
tests.  Nested model selection is intentionally narrower: only original-score
candidates are eligible, raw AFM is a fallback, and fusion replaces AFM only
when its paired inner-gene gain exceeds one standard error.

The primary metric is macro-LOGO AUC: mean AUC over held-out genes.  Pooled OOF
AUC is retained as a calibration diagnostic, not as the AFM-vs-fusion headline.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from .core import (
    LogisticRegression,
    StandardScaler,
    fit_fusion_feature_preprocessor_np,
    normalize_fusion_scores_np,
    roc_auc_score,
    PROJECT_ROOT,
)
from .protein_config import protein_config_fingerprint
from .wandb_progress import start_wandb_progress, WandbProgress


from metric_analysis.pipeline.library.canonical_cohort import logo_fold_group, logo_splits


REPRESENTATIONS = ("raw2", "compact4_v2", "engineered6", "offset")
LEGACY_REPRESENTATIONS = ("compact4",)
WEIGHTINGS = ("variant", "gene")
WEIGHTING_DISPLAY_NAMES = {
    "variant": "global class-balanced",
    "gene": "equal-gene class-balanced",
}
NORMALIZATIONS = ("original", "robust", "rank")
NESTED_NORMALIZATIONS = ("original",)
NESTED_REPRESENTATIONS = ("raw2", "engineered6", "offset")
NESTED_SELECTION_RULE = "AFM fallback when best fusion is within one paired inner-gene SE"


@dataclass(frozen=True, order=True)
class BenchmarkSpec:
    representation: str
    weighting: str
    normalization: str
    c: float

    @property
    def candidate_id(self) -> str:
        c_text = f"{self.c:g}".replace(".", "p")
        return (
            f"{self.representation}__{self.weighting}__"
            f"{self.normalization}__C{c_text}"
        )


def make_specs(representations=None, weightings=None, normalizations=None, c_values=None):
    """Return a validated, deterministic benchmark grid."""
    def unique(values):
        return tuple(dict.fromkeys(values))

    representations = unique(REPRESENTATIONS if representations is None else representations)
    weightings = unique(WEIGHTINGS if weightings is None else weightings)
    normalizations = unique(NORMALIZATIONS if normalizations is None else normalizations)
    c_values = unique(float(c) for c in ((0.1, 1.0, 10.0) if c_values is None else c_values))
    if not representations or not weightings or not normalizations or not c_values:
        raise ValueError("benchmark grid axes must each contain at least one value")
    unknown = {
        "representation": sorted(
            set(representations).difference(REPRESENTATIONS + LEGACY_REPRESENTATIONS)
        ),
        "weighting": sorted(set(weightings).difference(WEIGHTINGS)),
        "normalization": sorted(set(normalizations).difference(NORMALIZATIONS)),
    }
    bad = {name: values for name, values in unknown.items() if values}
    if bad:
        raise ValueError(f"unknown benchmark choices: {bad}")
    if not c_values or any(not np.isfinite(c) or c <= 0 for c in c_values):
        raise ValueError("all benchmark C values must be finite and > 0")
    return [
        BenchmarkSpec(rep, weighting, norm, c)
        for rep in representations
        for weighting in weightings
        for norm in normalizations
        for c in c_values
    ]


def _safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def _validate_frame(df):
    required = {"gene", "mutation", "label", "afm_score", "d1_score"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"benchmark frame is missing columns: {missing}")
    if df[sorted(required)].isna().any().any():
        raise ValueError("benchmark inputs contain missing gene/mutation/label/AFM/D1 values")
    if (df["gene"].astype(str).str.strip() == "").any() or (
        df["mutation"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("benchmark gene and mutation identifiers must be non-empty")
    if df.duplicated(["gene", "mutation"]).any():
        duplicated = (
            df.loc[df.duplicated(["gene", "mutation"], keep=False), ["gene", "mutation"]]
            .drop_duplicates().head(5).to_dict("records")
        )
        raise ValueError(f"benchmark contains duplicate gene/mutation keys: {duplicated}")
    labels = df["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("benchmark labels must be binary values 0/1")
    afm = df["afm_score"].to_numpy(dtype=float)
    d1 = df["d1_score"].to_numpy(dtype=float)
    if not np.isfinite(afm).all() or not np.isfinite(d1).all():
        raise ValueError("benchmark AFM/D1 scores must be finite")
    if ((afm < 0) | (afm > 1)).any():
        raise ValueError("AlphaMissense scores must lie in [0, 1]")
    if (d1 < 0).any():
        raise ValueError("raw D1 scores must be non-negative")
    # KS descriptors are optional in the frame, but when present they carry a
    # bound D1 does not: a Kolmogorov-Smirnov statistic is a maximum CDF gap and
    # therefore lies in [0, 1] by construction. A value outside it is a broken
    # estimator rather than unusual data, so it is worth failing on -- nothing
    # else in the pipeline checks these columns at all.
    for column in ("plddt_ks_residue", "plddt_ks_centred", "plddt_ks"):
        if column not in df.columns:
            continue
        ks = df[column].to_numpy(dtype=float)
        if not np.isfinite(ks).all():
            raise ValueError(f"{column} scores must be finite")
        if ((ks < 0) | (ks > 1)).any():
            raise ValueError(f"{column} is a KS statistic and must lie in [0, 1]")
    genes = sorted(df["gene"].astype(str).unique())
    if len(genes) < 2:
        raise ValueError("LOGO benchmark requires at least two genes")
    one_class = [
        gene for gene, g in df.groupby("gene", sort=True)
        if g["label"].astype(int).nunique() < 2
    ]
    if one_class:
        raise ValueError(
            "macro-LOGO AUC is undefined for one-class held-out genes: "
            f"{one_class}"
        )


def _within_gene_scores(df, normalization):
    """Return two label-free score columns under the requested representation.

    ``robust`` and ``rank`` are computed independently inside every gene,
    including a held-out gene.  This is intentionally transductive: it assumes a
    batch of unlabelled variants from the new gene is available at prediction
    time.  No labels enter the transformation.
    """
    raw = df[["afm_score", "d1_score"]].to_numpy(dtype=np.float32)
    return normalize_fusion_scores_np(
        raw, df["gene"].astype(str).to_numpy(), normalization,
    )


def sample_weights(y, genes, weighting):
    """Weights with mean one; no rows are removed.

    ``variant`` reproduces ``class_weight='balanced'`` over the training rows.
    ``gene`` gives every training gene equal total mass and, within each gene,
    gives its two classes equal mass.
    """
    raw_y = np.asarray(y)
    genes = np.asarray(genes, dtype=str)
    if len(raw_y) == 0 or len(raw_y) != len(genes):
        raise ValueError("sample_weights requires equally sized, non-empty y and genes")
    if not np.isin(raw_y, [0, 1]).all():
        raise ValueError("sample_weights requires binary labels 0/1")
    y = raw_y.astype(int)
    weights = np.zeros(len(y), dtype=float)
    if weighting == "variant":
        for cls in np.unique(y):
            idx = y == cls
            weights[idx] = len(y) / (np.unique(y).size * idx.sum())
    elif weighting == "gene":
        unique_genes = np.unique(genes)
        for gene in unique_genes:
            gene_idx = genes == gene
            classes = np.unique(y[gene_idx])
            if classes.size != 2:
                raise ValueError(
                    f"gene-balanced training requires both classes in {str(gene)!r}"
                )
            for cls in classes:
                idx = gene_idx & (y == cls)
                weights[idx] = 1.0 / (len(unique_genes) * len(classes) * idx.sum())
        weights *= len(weights) / weights.sum()
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    return weights


def _fit_offset_predict(afm, d1_rep, y, genes, tr, va, spec, *,
                        return_details=False, extra_tr=None, extra_va=None):
    """AFM log-odds plus descriptor terms whose learned coefficients are non-negative.

    AFM's own coefficient stays pinned to 1 -- it enters as an offset, never a
    fitted term -- so the free parameters are an intercept plus one slope per
    descriptor. This class-weighted, penalized, non-negative predictive fit is
    not the manuscript's likelihood-ratio test, even with one descriptor.
    `extra_tr`/`extra_va` add already-split, already-imputed descriptor columns.
    """
    desc_tr = d1_rep[tr, None]
    desc_va = d1_rep[va, None]
    if extra_tr is not None and np.size(extra_tr):
        # Extras arrive pre-split because their median imputation is fitted on
        # the training rows of THIS fold; re-deriving them here would leak.
        desc_tr = np.column_stack([desc_tr, np.asarray(extra_tr, dtype=float)])
        desc_va = np.column_stack([desc_va, np.asarray(extra_va, dtype=float)])
    d1_scaler = StandardScaler()
    d1_tr = d1_scaler.fit_transform(desc_tr)
    d1_va = d1_scaler.transform(desc_va)
    n_desc = d1_tr.shape[1]
    afm_logodds_tr = logit(np.clip(afm[tr], 1e-6, 1 - 1e-6))
    afm_logodds_va = logit(np.clip(afm[va], 1e-6, 1 - 1e-6))
    ytr = y[tr]
    weights = sample_weights(ytr, genes[tr], spec.weighting)
    weighted_mean = float(np.average(ytr, weights=weights))
    intercept0 = float(logit(np.clip(weighted_mean, 1e-6, 1 - 1e-6)))

    def objective(theta):
        intercept, betas = theta[0], theta[1:]
        eta = afm_logodds_tr + intercept + d1_tr @ betas
        data_loss = np.average(np.logaddexp(0.0, eta) - ytr * eta, weights=weights)
        return float(data_loss + 0.5 * float(betas @ betas) / spec.c)

    fitted = minimize(
        objective,
        x0=np.concatenate([[intercept0], np.zeros(n_desc)]),
        method="L-BFGS-B",
        bounds=[(None, None)] + [(0.0, None)] * n_desc,
    )
    if not fitted.success:
        raise RuntimeError(
            f"offset fit failed for {spec.candidate_id}: {fitted.message}"
        )
    intercept, betas = float(fitted.x[0]), fitted.x[1:]
    prediction = expit(afm_logodds_va + intercept + d1_va @ betas)
    if return_details:
        return prediction, {
            "intercept": intercept,
            # `beta`/`descriptor_scale` stay scalars naming the PARTNER slope so
            # existing single-descriptor readers keep working unchanged; the
            # full vectors sit alongside them.
            "beta": float(betas[0]),
            "descriptor_scale": float(d1_scaler.scale_[0]),
            "betas": [float(b) for b in betas],
            "descriptor_scales": [float(s) for s in d1_scaler.scale_],
            "n_descriptors": int(n_desc),
        }
    return prediction


def _fit_predict(df, represented, y, genes, tr, va, spec, model_seed):
    raw = df[["afm_score", "d1_score"]].to_numpy(dtype=float)
    if spec.representation == "offset":
        # AFM remains the original globally-defined score.  The selected
        # normalisation applies only to the D1 correction.
        return _fit_offset_predict(raw[:, 0], represented[:, 1], y, genes, tr, va, spec)

    features = represented
    if spec.representation in {"raw2", "compact4_v2", "compact4", "engineered6"}:
        xtr, xva, _ = fit_fusion_feature_preprocessor_np(
            features[tr], features[va], representation=spec.representation,
        )
    else:
        raise ValueError(f"unknown representation {spec.representation!r}")
    weights = sample_weights(y[tr], genes[tr], spec.weighting)
    model = LogisticRegression(
        C=spec.c,
        max_iter=4000,
        class_weight=None,
        random_state=int(model_seed),
    )
    model.fit(xtr, y[tr], sample_weight=weights)
    return model.predict_proba(xva)[:, 1]


def _row_meta(size, d1_seed, model_seed, spec, evaluation):
    return {
        "size": int(size),
        "d1_seed": int(d1_seed),
        "model_seed": int(model_seed),
        "evaluation": evaluation,
        "candidate_id": spec.candidate_id if spec is not None else "afm",
        "model": (
            "afm" if spec is None else
            "offset" if spec.representation == "offset" else "logreg"
        ),
        "representation": "afm" if spec is None else spec.representation,
        "weighting": "none" if spec is None else spec.weighting,
        "normalization": "original" if spec is None else spec.normalization,
        "C": np.nan if spec is None else float(spec.c),
    }


def _auc_rows(y, genes, afm, predictions, meta):
    rows = []
    per_gene_model, per_gene_afm = [], []
    for gene in sorted(np.unique(genes)):
        idx = genes == gene
        model_auc = _safe_auc(y[idx], predictions[idx])
        afm_auc = _safe_auc(y[idx], afm[idx])
        per_gene_model.append(model_auc)
        per_gene_afm.append(afm_auc)
        rows.append({
            **meta, "scope": gene, "scope_type": "gene",
            "auc": model_auc, "afm_auc": afm_auc,
            "delta_vs_afm": model_auc - afm_auc,
            "n": int(idx.sum()), "n_pos": int(y[idx].sum()),
        })
    macro, afm_macro = float(np.mean(per_gene_model)), float(np.mean(per_gene_afm))
    rows.append({
        **meta, "scope": "macro", "scope_type": "macro",
        "auc": macro, "afm_auc": afm_macro, "delta_vs_afm": macro - afm_macro,
        "n": int(len(y)), "n_pos": int(y.sum()),
    })
    pooled, afm_pooled = _safe_auc(y, predictions), _safe_auc(y, afm)
    rows.append({
        **meta, "scope": "pooled", "scope_type": "pooled_diagnostic",
        "auc": pooled, "afm_auc": afm_pooled, "delta_vs_afm": pooled - afm_pooled,
        "n": int(len(y)), "n_pos": int(y.sum()),
    })
    return rows


def evaluate_fixed_grid(
    df, specs, size, d1_seed, model_seed, progress_callback=None, fold_grouping="grouped",
):
    """Evaluate every fixed candidate with one outer LOGO prediction per row."""
    df = df.sort_values(["gene", "mutation"]).reset_index(drop=True).copy()
    _validate_frame(df)
    y = df["label"].astype(int).to_numpy()
    genes = df["gene"].astype(str).to_numpy()
    afm = df["afm_score"].to_numpy(dtype=float)
    represented = {
        norm: _within_gene_scores(df, norm)
        for norm in sorted({spec.normalization for spec in specs})
    }
    folds = logo_splits(genes, fold_grouping)
    rows = []

    afm_meta = _row_meta(size, d1_seed, model_seed, None, "fixed")
    rows.extend(_auc_rows(y, genes, afm, afm, afm_meta))
    for spec_index, spec in enumerate(specs, start=1):
        oof = np.full(len(df), np.nan, dtype=float)
        for tr, va in folds:
            oof[va] = _fit_predict(
                df, represented[spec.normalization], y, genes,
                tr, va, spec, model_seed,
            )
        rows.extend(_auc_rows(
            y, genes, afm, oof,
            _row_meta(size, d1_seed, model_seed, spec, "fixed"),
        ))
        if progress_callback is not None:
            progress_callback(
                spec_index, len(specs),
                {"candidate_id": spec.candidate_id, "phase": "fixed"},
            )
    return pd.DataFrame(rows)


def _grouped_mean_se(values, groups):
    """Cluster SE of the equal-dataset mean; singleton groups recover ordinary SEM."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Selection SE requires at least two inner fold groups")
    residual = values - values.mean()
    sums = np.array([residual[groups == group].sum() for group in unique])
    return float(np.sqrt(len(unique) / (len(unique) - 1) * np.sum(sums ** 2)) / len(values))


def evaluate_nested_selection(
    df, specs, size, d1_seed, model_seed,
    nested_representations=NESTED_REPRESENTATIONS,
    progress_callback=None,
    fold_grouping="grouped",
):
    """Conservative nested gene-level selection with no outer-label reuse.

    For each held-out outer gene, candidate selection is the mean AUC from an
    inner LOGO over the remaining genes.  Only original-score candidates are
    eligible, and the eligible representation set is explicit so adding a new
    fixed-grid ablation cannot silently enlarge nested model selection.
    Robust/rank variants remain fixed-grid transductive stress tests.
    Raw AFM is an explicit fallback and is selected whenever the best fusion's
    paired inner-gene gain is no larger than one standard error.  A selected
    fusion is then refit on every outer-training row and evaluated once on the
    untouched outer gene.
    """
    df = df.sort_values(["gene", "mutation"]).reset_index(drop=True).copy()
    _validate_frame(df)
    y = df["label"].astype(int).to_numpy()
    genes = df["gene"].astype(str).to_numpy()
    afm = df["afm_score"].to_numpy(dtype=float)
    unique_genes = sorted(np.unique(genes))
    group_keys = np.array([logo_fold_group(g, fold_grouping) for g in genes])
    if len(np.unique(group_keys)) < 4:
        raise ValueError("nested gene-level selection requires at least four genes "
                         "in independent fold groups")
    nested_representations = tuple(dict.fromkeys(nested_representations))
    eligible_specs = [
        spec for spec in specs
        if spec.normalization in NESTED_NORMALIZATIONS
        and spec.representation in nested_representations
    ]
    if not eligible_specs:
        raise ValueError(
            "nested selection requires at least one candidate matching its "
            "eligible original-score representation set"
        )
    represented = {
        norm: _within_gene_scores(df, norm)
        for norm in sorted({spec.normalization for spec in eligible_specs})
    }
    outer_oof = np.full(len(df), np.nan, dtype=float)
    selection_rows = []

    for outer_index, outer_gene in enumerate(unique_genes, start=1):
        outer_train = group_keys != logo_fold_group(outer_gene, fold_grouping)
        scores = []
        inner_genes = sorted(np.unique(genes[outer_train]))
        afm_inner_aucs = []
        for inner_gene in inner_genes:
            inner_val = outer_train & (genes == inner_gene)
            afm_inner_aucs.append(_safe_auc(y[inner_val], afm[inner_val]))
        afm_inner_auc = float(np.mean(afm_inner_aucs))

        for spec in eligible_specs:
            inner_aucs = []
            for inner_gene in inner_genes:
                inner_val = outer_train & (genes == inner_gene)
                inner_train = outer_train & (group_keys != logo_fold_group(inner_gene, fold_grouping))
                pred = _fit_predict(
                    df, represented[spec.normalization], y, genes,
                    np.flatnonzero(inner_train), np.flatnonzero(inner_val),
                    spec, model_seed,
                )
                inner_aucs.append(_safe_auc(y[inner_val], pred))
            paired_delta = np.asarray(inner_aucs) - np.asarray(afm_inner_aucs)
            delta_se = _grouped_mean_se(paired_delta, [logo_fold_group(g, fold_grouping) for g in inner_genes])
            scores.append({
                "inner_macro_auc": float(np.mean(inner_aucs)),
                "inner_aucs": inner_aucs,
                "delta_vs_afm": float(np.mean(paired_delta)),
                "delta_se": delta_se,
                "spec": spec,
            })
        scores.sort(key=lambda item: (-item["inner_macro_auc"], item["spec"].candidate_id))
        best_fusion = scores[0]
        best = best_fusion["spec"]
        use_afm = best_fusion["delta_vs_afm"] <= best_fusion["delta_se"]
        selected = None if use_afm else best
        selected_auc = afm_inner_auc if use_afm else best_fusion["inner_macro_auc"]
        alternatives = [
            afm_inner_auc,
            *[
                score["inner_macro_auc"]
                for score in scores
                if selected is None or score["spec"] != selected
            ],
        ]
        if use_afm:
            alternatives = [score["inner_macro_auc"] for score in scores]
        runner_up_auc = max(alternatives) if alternatives else np.nan
        outer_val = genes == outer_gene
        if selected is None:
            outer_oof[outer_val] = afm[outer_val]
            selected_fields = {
                "representation": "afm", "weighting": "none",
                "normalization": "original", "c": np.nan,
                "candidate_id": "afm",
            }
        else:
            outer_oof[outer_val] = _fit_predict(
                df, represented[selected.normalization], y, genes,
                np.flatnonzero(outer_train), np.flatnonzero(outer_val),
                selected, model_seed,
            )
            selected_fields = {
                **asdict(selected), "candidate_id": selected.candidate_id,
            }
        selection_rows.append({
            "size": int(size), "d1_seed": int(d1_seed),
            "model_seed": int(model_seed), "outer_gene": outer_gene,
            "fold_grouping": fold_grouping,
            "outer_fold_group": logo_fold_group(outer_gene, fold_grouping),
            "inner_fold_groups": len(np.unique(group_keys[outer_train])),
            **selected_fields,
            "inner_macro_auc": selected_auc,
            "runner_up_inner_macro_auc": runner_up_auc,
            "selection_margin": selected_auc - runner_up_auc,
            "best_fusion_candidate_id": best.candidate_id,
            "best_fusion_inner_macro_auc": best_fusion["inner_macro_auc"],
            "afm_inner_macro_auc": afm_inner_auc,
            "best_fusion_delta_vs_afm": best_fusion["delta_vs_afm"],
            "best_fusion_delta_se": best_fusion["delta_se"],
            "afm_fallback_used": bool(use_afm),
            "selection_rule": NESTED_SELECTION_RULE,
        })
        if progress_callback is not None:
            progress_callback(
                outer_index, len(unique_genes),
                {"outer_gene": outer_gene, "phase": "nested"},
            )

    nested_meta = {
        **_row_meta(size, d1_seed, model_seed, None, "nested"),
        "candidate_id": "nested_selection",
        "model": "nested_selection",
        "representation": "selected_inner_logo",
        "weighting": "selected_inner_logo",
        "normalization": "selected_inner_logo",
    }
    rows = _auc_rows(y, genes, afm, outer_oof, nested_meta)
    return pd.DataFrame(rows), pd.DataFrame(selection_rows)


def _summarize(runs):
    group_cols = [
        "size", "evaluation", "candidate_id", "model", "representation",
        "weighting", "normalization", "C", "scope", "scope_type",
    ]
    # Keep NaN C (AFM/nested) as a real group.
    grouped = runs.groupby(group_cols, dropna=False, sort=True)
    summary = grouped.agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", lambda x: float(np.std(x, ddof=0))),
        auc_ci_low=("auc", lambda x: float(np.quantile(x, 0.025))),
        auc_ci_high=("auc", lambda x: float(np.quantile(x, 0.975))),
        afm_auc_mean=("afm_auc", "mean"),
        delta_mean=("delta_vs_afm", "mean"),
        delta_std=("delta_vs_afm", lambda x: float(np.std(x, ddof=0))),
        delta_ci_low=("delta_vs_afm", lambda x: float(np.quantile(x, 0.025))),
        delta_ci_high=("delta_vs_afm", lambda x: float(np.quantile(x, 0.975))),
        n_d1_seeds=("d1_seed", "nunique"),
        n=("n", "first"),
        n_pos=("n_pos", "first"),
    ).reset_index()
    summary["uncertainty"] = "percentile interval across D1 subsampling seeds"
    return summary


def _plot_macro_overview(summary, path):
    import matplotlib.pyplot as plt

    fixed = summary[
        (summary["evaluation"] == "fixed")
        & (summary["scope_type"] == "macro")
        & (summary["model"] != "afm")
    ].copy()
    if fixed.empty:
        return
    fixed = fixed[
        (fixed["size"] == fixed["size"].max())
        & (fixed["normalization"] == "original")
    ]
    c0 = min(fixed["C"].dropna().unique(), key=lambda c: abs(np.log10(c)))
    fixed = fixed[np.isclose(fixed["C"], c0)]
    reps = [r for r in REPRESENTATIONS if r in set(fixed["representation"])]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {"variant": "#4c78a8", "gene": "#f58518"}
    x = np.arange(len(reps))
    width = 0.36
    for wi, weighting in enumerate(WEIGHTINGS):
        vals, lows, highs = [], [], []
        for rep in reps:
            row = fixed[
                (fixed["representation"] == rep)
                & (fixed["weighting"] == weighting)
            ]
            vals.append(float(row["delta_mean"].iloc[0]) if len(row) else np.nan)
            lows.append(float(row["delta_ci_low"].iloc[0]) if len(row) else np.nan)
            highs.append(float(row["delta_ci_high"].iloc[0]) if len(row) else np.nan)
        vals = np.asarray(vals)
        yerr = np.vstack([vals - np.asarray(lows), np.asarray(highs) - vals])
        pos = x + (wi - 0.5) * width
        bars = ax.bar(
            pos, vals, width, color=colors[weighting],
            label=WEIGHTING_DISPLAY_NAMES[weighting],
        )
        ax.errorbar(pos, vals, yerr=yerr, fmt="none", color="0.2", capsize=3, lw=1)
        ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=8)
    ax.axhline(0, color="0.25", lw=1)
    ax.set_xticks(x, reps)
    ax.set_ylabel("Macro-LOGO AUC change vs AFM")
    ax.set_title(f"Pre-specified original-score AFM+D1 benchmark (C={c0:g})")
    ax.legend(title="training weighting (same LOGO splits)"); ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01, 0.01, "error bars: 2.5–97.5 percentiles across paired D1 seeds",
        transform=ax.transAxes, fontsize=8, color="0.35",
    )
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _contrast_text_color(rgba):
    """Choose black or white using WCAG contrast against an RGBA background."""
    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = float(np.dot(linear, [0.2126, 0.7152, 0.0722]))
    white_contrast = 1.05 / (luminance + 0.05)
    black_contrast = (luminance + 0.05) / 0.05
    return "white" if white_contrast >= black_contrast else "black"


def _plot_normalization_stress(summary, path):
    """Compact stress-test view; unlike faceted bars, every cell shares one scale."""
    import matplotlib.pyplot as plt

    d = summary[
        (summary["evaluation"] == "fixed")
        & (summary["scope_type"] == "macro")
        & (summary["model"] != "afm")
    ].copy()
    if d.empty:
        return
    d = d[d["size"] == d["size"].max()]
    c0 = min(d["C"].dropna().unique(), key=lambda c: abs(np.log10(c)))
    d = d[np.isclose(d["C"], c0)]
    d["method"] = d["representation"] + " / " + d["weighting"].map(
        WEIGHTING_DISPLAY_NAMES
    )
    table = d.pivot(index="method", columns="normalization", values="delta_mean")
    order = [
        f"{rep} / {WEIGHTING_DISPLAY_NAMES[weight]}"
        for rep in REPRESENTATIONS for weight in WEIGHTINGS
    ]
    table = table.reindex([name for name in order if name in table.index])
    table = table.reindex(columns=[n for n in NORMALIZATIONS if n in table.columns])
    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    vmax = max(0.01, float(np.nanmax(np.abs(table.to_numpy()))))
    image = ax.imshow(table.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.iloc[i, j]
            color = _contrast_text_color(image.cmap(image.norm(value)))
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8, color=color)
    labels = [n + ("*" if n != "original" else "") for n in table.columns]
    ax.set_xticks(range(len(table.columns)), labels)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title(f"Normalization stress test: macro-LOGO change vs AFM (C={c0:g})")
    ax.set_xlabel("* label-free but transductive held-out-gene batch normalization")
    fig.colorbar(image, ax=ax, label="AUC change")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_per_gene_heatmap(runs, path):
    import matplotlib.pyplot as plt

    d = runs[
        (runs["evaluation"] == "fixed")
        & (runs["scope_type"] == "gene")
        & (runs["model"] != "afm")
        & (runs["normalization"] == "original")
    ].copy()
    if d.empty:
        return
    d = d[d["size"] == d["size"].max()]
    c0 = min(d["C"].dropna().unique(), key=lambda c: abs(np.log10(c)))
    d = d[np.isclose(d["C"], c0)]
    d["method"] = d["representation"] + " / " + d["weighting"].map(
        WEIGHTING_DISPLAY_NAMES
    )
    table = d.groupby(["method", "scope"])["delta_vs_afm"].mean().unstack()
    order = [
        f"{rep} / {WEIGHTING_DISPLAY_NAMES[weight]}"
        for rep in REPRESENTATIONS for weight in WEIGHTINGS
    ]
    table = table.reindex([name for name in order if name in table.index])
    fig, ax = plt.subplots(figsize=(1.25 * len(table.columns) + 3.7, 0.62 * len(table) + 2.5))
    vmax = max(0.01, float(np.nanmax(np.abs(table.to_numpy()))))
    image = ax.imshow(table.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.iloc[i, j]
            color = _contrast_text_color(image.cmap(image.norm(value)))
            ax.text(
                j, i, f"{value:+.3f}", ha="center", va="center",
                fontsize=8, color=color,
            )
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title(f"Per-gene LOGO AUC change vs AFM (original scores, C={c0:g})")
    fig.colorbar(image, ax=ax, label="AUC change")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_nested(nested_runs, selections, path):
    import matplotlib.pyplot as plt

    d = nested_runs[nested_runs["scope_type"] == "gene"].copy()
    if d.empty:
        return
    d = d[d["size"] == d["size"].max()]
    means = d.groupby("scope")["delta_vs_afm"].agg(["mean", "std"]).sort_values("mean")
    sel = selections[selections["size"] == selections["size"].max()].copy()
    common = sel.groupby("outer_gene")["candidate_id"].agg(
        lambda x: x.value_counts().index[0]
    )
    def short_candidate(candidate):
        return (candidate.replace("engineered6", "eng6")
                         .replace("variant", "global-class")
                         .replace("gene", "equal-gene")
                         .replace("original", "orig")
                         .replace("__", " / ")
                         .replace("C0p1", "C=.1"))

    labels = [f"{gene}\n{short_candidate(common.get(gene, ''))}" for gene in means.index]
    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.65 * len(means))))
    colors = np.where(means["mean"].to_numpy() >= 0, "#54a24b", "#e45756")
    ax.barh(range(len(means)), means["mean"], xerr=means["std"].fillna(0),
            color=colors, alpha=0.9, capsize=3)
    ax.axvline(0, color="0.25", lw=1)
    ax.set_yticks(range(len(means)), labels, fontsize=8)
    ax.set_xlabel("Nested outer-LOGO AUC change vs AFM")
    ax.set_title("Nested original-score selection with conservative AFM fallback")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def write_gt_group_comparison(group_result_dirs, out, candidate_id="engineered6__variant__original__C1"):
    """Combine already-computed group summaries for the pre-specified headline.

    This deliberately does not relabel legacy group sweeps: each input directory
    must have been evaluated after subsetting with the current ``GT_GROUPS``.
    """
    rows = []
    for group, directory in group_result_dirs.items():
        summary = pd.read_csv(Path(directory) / "benchmark_summary.csv")
        selected = summary[
            (summary["evaluation"] == "fixed")
            & (summary["scope_type"] == "macro")
            & (summary["candidate_id"] == candidate_id)
        ]
        if selected.empty:
            raise ValueError(f"{candidate_id!r} not found for gt_group={group!r}")
        row = selected.loc[selected["size"].idxmax()].to_dict()
        row["gt_group"] = group
        row["n_genes"] = int(summary[
            (summary["evaluation"] == "fixed")
            & (summary["scope_type"] == "gene")
            & (summary["candidate_id"] == candidate_id)
            & (summary["size"] == row["size"])
        ]["scope"].nunique())
        row["interpretation"] = (
            "primary" if group == "all" else
            "descriptive label-route subgroup"
        )
        rows.append(row)
    comparison = pd.DataFrame(rows)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "benchmark_gt_group_comparison.csv", index=False)

    import matplotlib.pyplot as plt
    order = [g for g in ("all", "annotation") if g in set(comparison["gt_group"])]
    plot = comparison.set_index("gt_group").loc[order]
    x = np.arange(len(plot))
    vals = plot["delta_mean"].to_numpy()
    yerr = np.vstack([
        vals - plot["delta_ci_low"].to_numpy(),
        plot["delta_ci_high"].to_numpy() - vals,
    ])
    colors = ["#4c78a8", "#54a24b", "#e45756"][:len(plot)]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(x, vals, color=colors, yerr=yerr, capsize=4, width=0.62)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in vals], padding=3)
    ax.axhline(0, color="0.25", lw=1)
    ax.set_xticks(x, [f"{g}\n(n={int(plot.loc[g, 'n_genes'])} genes)" for g in order])
    ax.set_ylabel("Macro-LOGO AUC change vs AFM")
    ax.set_title("Pre-specified linear AFM+D1 effect by label-source group")
    ax.text(0.01, 0.01, "functional/annotation panels are descriptive; all-genes is primary",
            transform=ax.transAxes, fontsize=8, color="0.35")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "benchmark_gt_group_delta.png", dpi=180); plt.close(fig)
    return comparison


def run_benchmark_sweep(
    base_cfg,
    sizes,
    d1_seeds,
    out,
    build_dataset,
    specs,
    gt_subset=None,
    gt_group="all",
    model_seed=2026,
    seed_plan=None,
    nested=True,
    nested_representations=NESTED_REPRESENTATIONS,
    wandb_enabled=False,
    wandb_project="afm_d1_clean_benchmark_progress",
    wandb_entity=None,
    force=False,
):
    """Run the local benchmark grid, plots, and optional live W&B progress."""
    fold_grouping = base_cfg.get("logo_folds", "grouped")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wb = (
        start_wandb_progress(
            wandb_project,
            wandb_entity,
            "clean_benchmark_progress",
            f"{gt_group}-grid",
            {
                "gt_group": gt_group,
                "d1_sizes": [int(x) for x in sizes],
                "d1_seeds": [int(x) for x in d1_seeds],
                "model_seed": int(model_seed),
                "candidate_count": int(len(specs)),
                "representations": sorted({spec.representation for spec in specs}),
                "weightings": sorted({spec.weighting for spec in specs}),
                "normalizations": sorted({spec.normalization for spec in specs}),
                "C_values": sorted({float(spec.c) for spec in specs}),
                "nested_selection": bool(nested),
                "wandb_role": "linear-grid compute progress monitor",
                "metric_primary": "macro-LOGO ROC-AUC",
                "pooled_oof_role": "calibration diagnostic",
            },
        )
        if wandb_enabled else None
    )
    progress = (
        WandbProgress(wb, len(d1_seeds) * len(sizes))
        if wb is not None else None
    )
    fixed_parts, nested_parts, selection_parts = [], [], []
    dataset_counts = None
    for d1_seed in d1_seeds:
        for size in sizes:
            if seed_plan and int(d1_seed) not in seed_plan.get(int(size), []):
                continue
            cfg = dict(base_cfg)
            cfg["seed"] = int(d1_seed)
            cfg["d1_max_structures_per_variant"] = int(size)
            block_label = f"benchmark N={size} d1_seed={d1_seed}"
            if progress is not None:
                progress.set_status(f"{block_label}; building dataset")
            df = build_dataset(size, cfg, force=force)
            if gt_subset is not None:
                df = gt_subset(df, gt_group)
            dataset_counts = (
                df.groupby("gene")["label"].agg(n="size", n_pos="sum")
                .reset_index().to_dict("records")
            )
            print(
                f"  [benchmark] size={size:>2} d1_seed={d1_seed} "
                f"candidates={len(specs)} genes={df['gene'].nunique()}"
            )
            n_fold_groups = len({logo_fold_group(g, fold_grouping) for g in df["gene"]})
            nested_units = (
                int(df["gene"].nunique())
                if nested and n_fold_groups >= 4 else 0
            )
            block_units = len(specs) + nested_units
            block_callback = (
                progress.callback(block_label) if progress is not None else None
            )

            def fixed_progress(completed, total, detail):
                if block_callback is not None:
                    block_callback(completed, block_units, detail)

            fixed_parts.append(evaluate_fixed_grid(
                df, specs, size=size, d1_seed=d1_seed, model_seed=model_seed,
                progress_callback=fixed_progress if progress is not None else None,
                fold_grouping=fold_grouping,
            ))
            if nested:
                if n_fold_groups < 4:
                    print(
                        "  [benchmark:nested] skipped: fewer than four independent fold groups "
                        f"in gt_group={gt_group!r}"
                    )
                else:
                    def nested_progress(completed, total, detail):
                        if block_callback is not None:
                            block_callback(len(specs) + completed, block_units, detail)

                    nested_run, selections = evaluate_nested_selection(
                        df, specs, size=size, d1_seed=d1_seed,
                        model_seed=model_seed,
                        nested_representations=nested_representations,
                        fold_grouping=fold_grouping,
                        progress_callback=(
                            nested_progress if progress is not None else None
                        ),
                    )
                    nested_parts.append(nested_run)
                    selection_parts.append(selections)
            if progress is not None:
                progress.complete(block_label)

    fixed = pd.concat(fixed_parts, ignore_index=True)
    nested_runs = (
        pd.concat(nested_parts, ignore_index=True)
        if nested_parts else pd.DataFrame(columns=fixed.columns)
    )
    runs = fixed if nested_runs.empty else pd.concat([fixed, nested_runs], ignore_index=True)
    selections = (
        pd.concat(selection_parts, ignore_index=True)
        if selection_parts else pd.DataFrame()
    )
    summary = _summarize(runs)
    runs.to_csv(out / "benchmark_runs.csv", index=False)
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    if not selections.empty:
        selections.to_csv(out / "benchmark_nested_selections.csv", index=False)

    manifest = {
        "logo_folds": fold_grouping,
        "inner_selection_se": "cluster SE by fold group; equal-dataset mean retained",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protein_config_fingerprint": protein_config_fingerprint(),
        "cohort_manifest": str(
            Path(PROJECT_ROOT) / "metric_analysis" / "manifests" /
            "strict_canonical_manifest.csv"
        ),
        "cohort_manifest_sha256": hashlib.sha256(
            (Path(PROJECT_ROOT) / "metric_analysis" / "manifests" /
             "strict_canonical_manifest.csv").read_bytes()
        ).hexdigest(),
        "gt_group": gt_group,
        "dataset_counts_last_seed_size": dataset_counts,
        "sizes": [int(x) for x in sizes],
        "d1_seeds": [int(x) for x in d1_seeds],
        "model_seed": int(model_seed),
        "nested_selection": bool(nested),
        "nested_selection_policy": {
            "eligible_normalizations": list(NESTED_NORMALIZATIONS),
            "eligible_representations": list(nested_representations),
            "afm_is_candidate": True,
            "rule": NESTED_SELECTION_RULE,
            "transductive_candidates": "fixed-grid diagnostics only; never nested-selectable",
        },
        "candidates": [asdict(spec) | {"candidate_id": spec.candidate_id} for spec in specs],
        "primary_metric": "mean of held-out-gene ROC-AUCs (macro-LOGO)",
        "pooled_oof_metric": "diagnostic only; includes cross-gene calibration differences",
        "weighting": {
            "variant": "all rows retained; global inverse class-frequency weights",
            "gene": "all rows retained; equal gene mass and equal class mass within gene",
        },
        "normalization": {
            "original": "original AFM and D1 scores",
            "robust": "label-free within-gene median/IQR; transductive held-out-gene batch",
            "rank": "label-free within-gene percentile ranks; transductive held-out-gene batch",
            "offset_note": "AFM remains original log-odds; normalization applies to D1 correction only",
        },
        "uncertainty": "percentile interval across paired D1 structure-subsampling seeds",
        "wandb_enabled": bool(wandb_enabled),
    }
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    _plot_macro_overview(summary, out / "benchmark_macro_delta.png")
    _plot_normalization_stress(summary, out / "benchmark_normalization_stress.png")
    _plot_per_gene_heatmap(runs, out / "benchmark_per_gene_delta.png")
    if not nested_runs.empty:
        _plot_nested(nested_runs, selections, out / "benchmark_nested_delta.png")
    if wb is not None:
        progress.finish()
        wb.finish()
    return runs, summary, selections
