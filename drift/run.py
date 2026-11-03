"""Clean, reproducible DRIFT AFM+D1 fusion run (Weights & Biases optional).

Produces, per gene AND pooled, for four models
    - AFM-only     (raw AlphaMissense score; no training)
    - D1-only      (raw D1 score; no training)
    - LogReg       (logistic regression over an explicit AFM+D1 representation)
    - Fusion (MLP) (plain MLP over the exact same representation)
under two cross-validation regimes
    - LOGO         (leave-one-gene-out; the honest cross-gene generalization test)
    - stratified   (stratified k-fold; mixes genes across folds -- optimistic)
with ROC-AUC, average precision (AP), and bootstrap 95% CIs, averaged over seeds.

The labeled dataset (D1 recomputed from the pLDDT CSVs via the notebook's data
builders) is cached by protein configuration, ensemble size, and D1 sampling
seed. Use --force-recompute to ignore a matching cache.

Usage:
    python -m drift.run --mode both --size 90 --seeds 42-51 --out results/fusion_clean
    python -m drift.run --mode logo --size 90 --seeds 42-44        # quick
"""
import argparse
import json
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import core
from . import ensemble_shift
from . import convergence as convergence_mod
from . import sampling_plan
from .convergence import DEFAULT_EPSILON, DEFAULT_SD_THRESHOLD
from .sampling_plan import DEFAULT_FLOOR as DEFAULT_SEED_FLOOR
from .core import SAMPLING_MODES, SELECTION_RULES
from .ensemble_shift import build_ensemble_shift_features, FEATURE_COLUMNS as ES_FEATURE_COLUMNS
from .benchmark import BenchmarkSpec, _fit_offset_predict
from .protein_config import protein_config_fingerprint
from .wandb_progress import (
    start_wandb_progress as _wandb_start,
    WandbProgress as _WandbProgress,
)
from .core import (
    build_labeled_dataset_for_config,
    FUSION_NORMALIZATIONS,
    FUSION_REPRESENTATIONS,
    featurize_fusion_representation_np,
    fit_fusion_feature_preprocessor_np,
    normalize_fusion_scores_np,
    compute_binary_metrics_at_threshold,
    train_one_fold,
    run_cross_validated_training,
    _resolve_run_config,
    fusion_basis_columns,
    fusion_partner_name,
    FUSION_PARTNERS,
    seed_everything,
    SWEEP_BASE_CONFIG,
    StandardScaler,
    LogisticRegression,
    roc_auc_score,
    average_precision_score,
    PROJECT_ROOT,
)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
ES_CACHE_VERSION = 6  # Version 5 predates enforced subunit residue scope.
STRICT_COHORT_MANIFEST = (
    Path(PROJECT_ROOT) / "metric_analysis" / "manifests" /
    "strict_canonical_manifest.csv"
)

# --------------------------------------------------------------------------- #
# feature variants: which ensemble-shift descriptors get appended after the
# selected AFM+D1 representation.
# The sweep crosses ensemble-size x feature-variant so each descriptor's honest
# (LOGO) contribution is isolated rather than blurred into one "with-everything"
# number. "base" == the original afm+d1 fusion and reproduces prior results.
# --------------------------------------------------------------------------- #
FEATURE_VARIANTS = {
    "base": [],
    "spread": ["plddt_spread_ratio"],
    "d1std": ["plddt_d1_std"],
    "wass": ["plddt_wasserstein"],
    # profileKS variants ("ks", "dist", "all") removed 2026-10-13.
    "ks_residue": ["plddt_ks_residue"],
    "ks_centred": ["plddt_ks_centred"],
    # The three orthogonality axes that survived on the metric side. Unlike the
    # entries above -- which are all further pLDDT distribution summaries, and so
    # near-collinear with d1_score -- conservation is not derived from pLDDT at all
    # and the distal axis changes which residues are aggregated. Columns are added
    # to the labeled_df caches by augment_cache_with_axes.py.
    # Three axes total: d1_score already sits in the AFM+D1 basis, so only the two
    # non-pLDDT/re-aggregated ones are appended. axis_energy was dropped -- energy
    # distance correlates with D1 at rho=0.96 and recomputing it at all 70
    # (size, seed) points cost hours for no new information.
    "axes3": ["axis_conservation", "axis_distal"],
    "cons": ["axis_conservation"],
    "distal": ["axis_distal"],
}

# --------------------------------------------------------------------------- #
# Ground-truth groups. Since 2026-10-13 there is ONE binary label route:
#   annotation -- driver = explicit variant annotation / ClinVar class (POLE
#                 binary label table, MSH2/MSH6/PTEN/VHL/PIK3CA ClinVar).
# The former "functional" binary group (TP53 and KRAS labelled by thresholding
# their assays) is gone: thresholding a continuous assay into driver/passenger
# was abandoned, so TP53 and KRAS have no binary label and are disabled in
# protein_configs.json. Functional analyses are continuous only, in
# experiments/continuous_functional (TP53, KRAS, PTEN, MSH2 assay scores).
# The mixed "evaluation" group went with it.
#
# FOXA1, the positional control (label mode hotspot_range: a residue interval,
# not a clinical annotation), was in no GT group and left the release on 2026-11-03.
# --gt-group restricts training+CV to one group; 'all' is every enabled binary
# dataset (unchanged behavior/output paths); 'each' runs every group.
# --------------------------------------------------------------------------- #
GT_GROUPS = {
    "annotation": {"POLE", "MSH2", "MSH6", "VHL", "PTEN", "PIK3CA"},
}
RETIRED_GT_GROUPS = {
    "functional": "binary functional labels were removed on 2026-10-13; use "
                  "experiments/continuous_functional for the assay targets",
    "evaluation": "the mixed annotation+functional group was removed on 2026-10-13",
}


def _gt_subset(df, name):
    """Rows for a GT group; 'all' is the pooled baseline (every gene)."""
    if name == "all":
        return df
    if name in RETIRED_GT_GROUPS:
        raise SystemExit(f"[gt-group] {name!r} is retired: {RETIRED_GT_GROUPS[name]}")
    sub = df[df["gene"].isin(GT_GROUPS[name])].copy()
    if sub.empty:
        raise SystemExit(f"[gt-group] no variants for group {name!r} ({sorted(GT_GROUPS[name])})")
    return sub


def _gt_group_list(spec):
    """Resolve --gt-group into an ordered, de-duplicated list of groups to run.

    Accepts a comma-separated combination of 'all' and 'annotation', or the alias
    'each' == both. The retired 'functional'/'evaluation' groups fail loudly.
    """
    valid = ["all"] + list(GT_GROUPS)  # all, annotation
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "each":
            out.extend(valid)
        elif tok in valid:
            out.append(tok)
        elif tok in RETIRED_GT_GROUPS:
            raise SystemExit(f"[gt-group] {tok!r} is retired: {RETIRED_GT_GROUPS[tok]}")
        else:
            raise SystemExit(
                f"[gt-group] unknown group {tok!r}; choose from {valid} "
                f"(comma-separated) or 'each'")
    seen, ordered = set(), []
    for g in out:
        if g not in seen:
            seen.add(g); ordered.append(g)
    return ordered or ["all"]


def _add_derived_features(df):
    """Columns derived from the cached ensemble-shift features (no rebuild needed).

    plddt_d1_std = |spread_ratio - 1|: the D1 transform (|ratio-1|) applied to the
    ensemble SPREAD instead of the mean, so 'no change vs WT' sits at 0 and it is
    on the same footing as D1 itself.
    """
    if "plddt_spread_ratio" in df.columns:
        df["plddt_d1_std"] = (df["plddt_spread_ratio"] - 1.0).abs()
    return df


def _partner_list(args):
    """--fusion-partners if given, else the singular --fusion-partner."""
    spec = getattr(args, "fusion_partners", None)
    if not spec:
        return [args.fusion_partner]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in FUSION_PARTNERS:
            raise ValueError(
                f"unknown fusion partner {tok!r}; choose from {list(FUSION_PARTNERS)}"
            )
        if tok not in out:
            out.append(tok)
    if not out:
        raise ValueError("--fusion-partners resolved to nothing")
    return out


def _feature_cols(variant):
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"unknown feature variant {variant!r}; choose from {list(FEATURE_VARIANTS)}")
    return list(FEATURE_VARIANTS[variant])


def _apply_strict_canonical_cohort(df, manifest_path=STRICT_COHORT_MANIFEST):
    """Restrict a full fusion frame to the canonical cohort and its labels.

    Fusion caches predate the transcript-matched cohort audit. Their structural
    descriptors remain reusable, but their membership and labels are not
    authoritative. This merge makes the versioned strict manifest the single
    source of truth on every cache and recomputation path.
    """
    required = {"gene", "mutation", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"fusion dataset missing required columns: {sorted(missing)}")

    manifest = pd.read_csv(manifest_path)
    manifest_required = {"protein", "mutation_code", "label"}
    missing = manifest_required.difference(manifest.columns)
    if missing:
        raise ValueError(f"strict cohort manifest missing columns: {sorted(missing)}")

    canonical = manifest[["protein", "mutation_code", "label"]].rename(columns={
        "protein": "gene",
        "mutation_code": "mutation",
        "label": "strict_label",
    })
    # Legacy alias: MSH2/MSH6 spell the same in both conventions, so this is a
    # no-op for them and only still matters for a pre-split results file.
    canonical["gene"] = canonical["gene"].replace({"MutS_alpha": "MutSalpha"})
    keys = ["gene", "mutation"]
    if canonical.duplicated(keys).any():
        raise ValueError("strict cohort manifest contains duplicate gene/mutation keys")
    if df.duplicated(keys).any():
        raise ValueError("fusion dataset contains duplicate gene/mutation keys")

    source_keys = df[keys].drop_duplicates()
    missing_from_dataset = canonical.merge(source_keys, on=keys, how="left", indicator=True)
    missing_from_dataset = missing_from_dataset[missing_from_dataset["_merge"] == "left_only"]
    if not missing_from_dataset.empty:
        examples = missing_from_dataset[keys].head(5).to_dict("records")
        raise ValueError(
            f"fusion dataset is missing {len(missing_from_dataset)} canonical variants; "
            f"examples={examples}"
        )

    merged = df.merge(canonical, on=keys, how="inner", validate="one_to_one", sort=False)
    n_excluded = len(df) - len(merged)
    n_relabelled = int((merged["label"].astype(int) != merged["strict_label"].astype(int)).sum())
    merged["label"] = merged.pop("strict_label").astype(int)
    if n_excluded or n_relabelled:
        print(
            f"[cohort] strict canonical manifest: excluded={n_excluded}, "
            f"labels_overridden={n_relabelled}, retained={len(merged)}"
        )
    return merged


# --------------------------------------------------------------------------- #
# structure-only track: models built from D1 and its pLDDT-derived metrics,
# with NO AlphaMissense. Isolates how far the purely-structural signal carries
# and whether ensemble-shift features help when AFM isn't there to dominate.
# --------------------------------------------------------------------------- #
STRUCTURE_VARIANTS = {
    "d1": ["d1_score"],
    "d1+spread": ["d1_score", "plddt_spread_ratio"],
    "d1+wass": ["d1_score", "plddt_wasserstein"],
    # profileKS variants ("d1+dist", "d1+all") removed 2026-10-13.
}


def _structure_cols(variant):
    if variant not in STRUCTURE_VARIANTS:
        raise ValueError(f"unknown structure variant {variant!r}; choose from {list(STRUCTURE_VARIANTS)}")
    return list(STRUCTURE_VARIANTS[variant])

# --------------------------------------------------------------------------- #
# metrics helpers
# --------------------------------------------------------------------------- #
def _auc(y, s):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return np.nan


def _ap(y, s):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return np.nan


def _bootstrap_ci(y, s, n_boot=2000, seed=0, fn=_auc):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# dataset (cache-first, recompute from pLDDT as fallback)
# --------------------------------------------------------------------------- #
def build_or_load_dataset(size, cfg, force=False):
    """Labeled (gene, mutation, label, afm_score, d1_score) + ensemble-shift
    features, cached per (size, seed, ratio-mode, protein-config, sampling
    policy). The cache is tagged `esfeatN` so it is distinct from
    pre-ensemble-shift caches.

    The sampling-policy tag is empty for the default nested/random policy, so
    caches written before the policy existed stay valid; a non-default policy
    gets its own cache rather than colliding with the primary run.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config_hash = protein_config_fingerprint()
    policy = core.sampling_policy_from_config(cfg)
    policy_tag = f"_{policy.tag()}" if policy.tag() else ""
    cache = CACHE_DIR / (
        f"labeled_size{size}_seed{cfg['seed']}_{cfg['d1_ratio_mode']}_"
        f"proteins{config_hash}_esfeat{ES_CACHE_VERSION}{policy_tag}.csv"
    )
    if cache.exists() and not force:
        df = pd.read_csv(cache)
        if all(c in df.columns for c in ES_FEATURE_COLUMNS):
            print(f"[data] loading cached labeled_df: {cache.name}")
            return _add_derived_features(_apply_strict_canonical_cohort(df))
        print(f"[data] cache missing ensemble-shift columns -> recomputing: {cache.name}")
    print(f"[data] no cache -> recomputing D1 + ensemble-shift from pLDDT CSVs (size={size}) ...")
    pack = build_labeled_dataset_for_config(
        foxa1_hotspot_range=(cfg["foxa1_hotspot_start"], cfg["foxa1_hotspot_end"]),
        d1_max_structures_per_variant=size,
        d1_ratio_mode=cfg["d1_ratio_mode"],
        d1_seed=cfg["seed"],
        run_config=cfg,
        d1_sampling_policy=policy,
    )
    df = pack["labeled_df"][["gene", "mutation", "label", "afm_score", "d1_score"]].copy()

    # Ensemble-shift descriptors from the SAME pLDDT CSVs and structure sampling.
    feat = build_ensemble_shift_features(
        core.D1_RUNTIME_CONFIGS, max_structures_per_variant=size, seed=cfg["seed"],
        policy=policy,
    )
    df = df.merge(feat, on=["gene", "mutation"], how="left")
    na = df[ES_FEATURE_COLUMNS].isna().sum()
    undefined = [c for c in ES_FEATURE_COLUMNS if na[c] == len(df)]
    partial = {c: int(na[c]) for c in ES_FEATURE_COLUMNS if 0 < na[c] < len(df)}
    if undefined:
        # e.g. plddt_spread_ratio is undefined at size=1 (no spread from one
        # structure); it will impute to a constant, i.e. contribute nothing.
        print(f"[data] undefined at size={size} (expected): {undefined} "
              f"-> imputed to a constant (feature inert at this ensemble size)")
    if partial:
        print(f"[data] partially missing (median-imputed within CV folds): {partial}")

    df = _apply_strict_canonical_cohort(df)
    df.to_csv(cache, index=False)
    print(f"[data] cached -> {cache.name}  ({len(df)} labeled variants, "
          f"+{len(ES_FEATURE_COLUMNS)} ensemble-shift features)")
    return _add_derived_features(df)


# --------------------------------------------------------------------------- #
# LOGO cross-validation (implemented cleanly; the notebook's built-in CV is
# stratified only). Each held-out fold is exactly one gene, so per-fold == per-gene.
# --------------------------------------------------------------------------- #
def _impute_extras(extras, tr, va):
    """Median-impute extra feature columns from TRAIN rows only (no leakage).
    Returns (train_slice, val_slice) or (None, None) when there are no extras."""
    if extras is None or extras.shape[1] == 0:
        return None, None
    e = extras.copy()
    with warnings.catch_warnings():
        # A feature undefined at this ensemble size (e.g. spread at size=1) is an
        # all-NaN column; nanmedian warns then we fall back to a constant below.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(e[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = np.where(~np.isfinite(e))
    if bad[0].size:
        e[bad] = np.take(med, bad[1])
    return e[tr].astype(np.float32), e[va].astype(np.float32)


from metric_analysis.pipeline.library.canonical_cohort import logo_splits



# --------------------------------------------------------------------------- #
# Model panel for the LOGO fusion arms.
#
# The MLP was the default and is now opt-in. It lost to plain logistic on
# identical folds (macro 0.8858 vs 0.8934 under stratified CV, 2026-09-10) with
# ~3x the seed-to-seed SD, while costing ~11 s/fold against logistic's 0.004 --
# roughly 99.96% of a sweep's compute for the worst arm. With 2-3 features, 7
# LOGO domains and a target effect of ~0.006 AUC, the binding constraint is
# variance under domain shift, not capacity.
#
# The panel is REPORTED, never selected. Picking the best-scoring class after
# seeing held-out results is the same error as the draft's post-hoc +0.0035.
#
#   offset  AFM log-odds as a FIXED offset (coefficient pinned to 1) plus an
#           intercept and one non-negative descriptor slope. Two free
#           parameters with a regularized, constrained predictive objective;
#           this is not the likelihood-ratio test in Figure 2e. Reused from
#           benchmark.py rather than reimplemented -- one scientific model,
#           one implementation.
#   logreg  fixed-C logistic on the represented basis. Linear plus whatever
#           interactions the representation builds.
#   gbm     Raw inputs only. AM and D1 are constrained increasing; no direction
#           is imposed on KS, spread, or other added descriptors. Trees cannot
#           extrapolate beyond training feature ranges.
#   mlp     the historical arm, emitted as model="fusion" for continuity.
SKLEARN_PANEL_MODELS = ("extratrees", "rf", "svm_rbf", "svm_linear", "lda",
                        "qda", "spline_logreg", "mlp_sklearn")
FUSION_MODELS = ("offset", "logreg", "gbm", "mlp") + SKLEARN_PANEL_MODELS
# Row label per model. "fusion" is kept for the MLP so existing figure code
# that filters on it keeps meaning the same thing.
FUSION_MODEL_ROW_NAME = {
    "offset": "offset", "logreg": "logreg", "gbm": "gbm", "mlp": "fusion",
    **{m: m for m in SKLEARN_PANEL_MODELS},
}
DEFAULT_FUSION_MODELS = ("offset", "logreg", "gbm")


def _model_list(spec):
    """Resolve a comma-separated --fusion-models value, order preserved."""
    if not spec:
        return list(DEFAULT_FUSION_MODELS)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in FUSION_MODELS:
            raise ValueError(
                f"unknown fusion model {tok!r}; choose from {list(FUSION_MODELS)}"
            )
        if tok not in out:
            out.append(tok)
    if not out:
        raise ValueError("--fusion-models resolved to nothing")
    return out


# Panel members that are a plain fit on the shared standardized matrix. Each
# takes (Xtr, ytr, Xva, seed) and returns a score for Xva; ranking is all that
# ROC-AUC needs, so the SVMs return decision values rather than probabilities.
# Hyperparameters are fixed. Selecting them would require an inner training-only
# search, with the outer folds reserved for evaluating that selection procedure.
def _fit_sklearn_panel_model(name, Xtr, ytr, Xva, seed, *, return_details=False):
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.svm import SVC, LinearSVC

    seed = int(seed)
    balanced = {"class_weight": "balanced"}
    if name == "extratrees":
        model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5,
                                     n_jobs=1, random_state=seed, **balanced)
    elif name == "rf":
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                       n_jobs=1, random_state=seed, **balanced)
    elif name == "svm_rbf":
        model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed, **balanced)
    elif name == "svm_linear":
        model = LinearSVC(C=1.0, random_state=seed, **balanced)
    elif name == "lda":
        # Equal priors: the panel's other members are reweighted to balanced, and
        # LDA's empirical priors would otherwise encode POLE's 3.5% positive rate.
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                           priors=[0.5, 0.5])
    elif name == "qda":
        model = QuadraticDiscriminantAnalysis(reg_param=0.5, priors=[0.5, 0.5])
    elif name == "spline_logreg":
        model = make_pipeline(
            SplineTransformer(n_knots=5, degree=3, knots="quantile",
                              extrapolation="linear", include_bias=False),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed, **balanced))
    elif name == "mlp_sklearn":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, solver="adam",
                              activation="relu", max_iter=300, early_stopping=True,
                              validation_fraction=.15, n_iter_no_change=20,
                              random_state=seed)
    else:
        raise ValueError(f"unknown panel model {name!r}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_kwargs = ({"sample_weight": compute_sample_weight("balanced", ytr)}
                      if name == "mlp_sklearn" else {})
        model.fit(Xtr, ytr, **fit_kwargs)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(Xva)[:, 1].astype(np.float32)
        else:
            scores = model.decision_function(Xva).astype(np.float32)
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    details = {
        "model": name,
        "warnings": [{"category": w.category.__name__, "message": str(w.message)}
                     for w in caught],
        "n_iter": np.asarray(getattr(estimator, "n_iter_", [])).tolist(),
        "balanced_sample_weight": name == "mlp_sklearn",
        "score_kind": "probability" if hasattr(model, "predict_proba") else "decision",
        "parameters": {k: v for k, v in model.get_params(deep=True).items()
                       if v is None or isinstance(v, (str, int, float, bool, tuple))},
    }
    if return_details:
        return scores, details
    for warning in caught:
        warnings.warn(warning.message, warning.category, stacklevel=2)
    return scores


def _fit_monotone_gbm(Xtr, ytr, Xva, seed, constraints):
    """Raw-feature arm with explicitly specified directional constraints.

    Standardisation is affine and positive, so a +1 constraint on the scaled
    column is the same constraint on the raw descriptor.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        monotonic_cst=constraints,
        max_iter=200,
        learning_rate=0.05,
        early_stopping=False,
        random_state=int(seed),
        class_weight="balanced",
    )
    model.fit(Xtr, ytr)
    return model.predict_proba(Xva)[:, 1].astype(np.float32)


def logo_cv(
    labeled_df, cfg, feature_cols=None, fusion_representation="raw2",
    fusion_normalization="original", progress_callback=None, models=None,
    cv_mode="logo",
):
    """Like-for-like LOGO comparison on one explicit AFM+D1 representation.

    The requested original/robust/rank score normalization is applied first,
    followed by the AFM+D1 basis and outer-training-row standardization. The MLP
    and logistic comparator consume the exact same matrix. Robust/rank use the
    unlabelled held-out-gene batch and are transductive diagnostics.
    """
    feature_cols = list(feature_cols or [])
    n_extra = len(feature_cols)
    seed_everything(cfg["seed"])
    df = labeled_df.sort_values(["gene", "mutation"]).reset_index(drop=True).copy()
    basis_columns = fusion_basis_columns(cfg)
    partner = fusion_partner_name(cfg)
    raw = df[basis_columns].to_numpy(dtype=np.float32)
    extras = df[feature_cols].to_numpy(dtype=np.float32) if n_extra else None
    y = df["label"].astype(int).to_numpy()
    groups = df["gene"].astype(str).to_numpy()
    represented_raw = normalize_fusion_scores_np(
        raw, groups, normalization=fusion_normalization,
    )

    models = list(models or ("mlp", "logreg"))
    unknown = [m for m in models if m not in FUSION_MODELS]
    if unknown:
        raise ValueError(f"unknown fusion model(s) {unknown}; choose from {list(FUSION_MODELS)}")
    if "gbm" in models and fusion_representation != "raw2":
        raise ValueError("monotone GBM requires raw2 inputs")
    oof = {m: np.full(len(df), np.nan, dtype=np.float32) for m in models}
    # Aliases so the rest of the function reads as before when the historical
    # pair is in play; both point at the same arrays.
    oof_fusion = oof.get("mlp")
    oof_logreg = oof.get("logreg")

    per_gene = []
    histories = []
    coefficients = []
    fit_diagnostics = []
    if cv_mode == "logo":
        splits = logo_splits(groups, cfg.get("logo_folds", "grouped"))
    elif cv_mode == "stratified":
        from sklearn.model_selection import StratifiedKFold
        strata = pd.Series(groups) + "__" + pd.Series(y).astype(str)
        if strata.value_counts().min() < 5:
            raise ValueError("matched stratified CV requires >=5 variants per gene/class")
        splits = list(StratifiedKFold(5, shuffle=True, random_state=int(cfg["seed"]))
                      .split(raw, strata))
    else:
        raise ValueError(f"unknown cv_mode {cv_mode!r}")
    for fold_index, (tr, va) in enumerate(splits, start=1):
        gene = groups[va][0]
        if len(np.unique(y[tr])) < 2:
            if progress_callback is not None:
                progress_callback(
                    fold_index, len(splits),
                    {"held_out_gene": gene, "status": "skipped"},
                )
            continue
        etr, eva = _impute_extras(extras, tr, va)
        Xtr, Xva, _ = fit_fusion_feature_preprocessor_np(
            represented_raw[tr], represented_raw[va],
            representation=fusion_representation,
            train_extras=etr, validation_extras=eva,
        )
        out = None
        if "mlp" in oof:
            out = train_one_fold(
                Xtr, y[tr], Xva, y[va],
                epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                hidden=cfg["hidden"], dropout=cfg["dropout"], lr=cfg["lr"],
                weight_decay=cfg["weight_decay"], patience=cfg["patience"],
                model_kind="plain",
            )
            oof["mlp"][va] = out["val_probs"]
            history = out.get("history")
            if isinstance(history, pd.DataFrame) and not history.empty:
                histories.append(history.assign(held_out_gene=gene))
        if "logreg" in oof:
            # Identical standardized matrix to every other arm: the model class
            # is the only intended difference across the panel.
            lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                    random_state=int(cfg["seed"])).fit(Xtr, y[tr])
            oof["logreg"][va] = lr.predict_proba(Xva)[:, 1].astype(np.float32)
        if "gbm" in oof:
            constraints = [1 if c in ("afm_score", "d1_score") else 0
                           for c in [*basis_columns, *feature_cols]]
            oof["gbm"][va] = _fit_monotone_gbm(Xtr, y[tr], Xva, cfg["seed"], constraints)
        for panel_model in SKLEARN_PANEL_MODELS:
            if panel_model in oof:
                prediction, details = _fit_sklearn_panel_model(
                    panel_model, Xtr, y[tr], Xva, cfg["seed"], return_details=True)
                oof[panel_model][va] = prediction
                fit_diagnostics.append({"fold": fold_index,
                                        "held_out_genes": sorted(set(groups[va])),
                                        **details})
        if "offset" in oof:
            # Deliberately NOT the standardized basis: AFM enters as its own
            # log-odds with the coefficient pinned to 1, and only the partner
            # descriptor is scaled and given a slope. Reuses benchmark.py's
            # fitter so there is one implementation of this model.
            spec = BenchmarkSpec(
                representation="offset", weighting="variant",
                normalization=fusion_normalization, c=1.0,
            )
            prediction, details = _fit_offset_predict(
                raw[:, 0].astype(float), represented_raw[:, 1].astype(float),
                y, groups, tr, va, spec, return_details=True,
                extra_tr=etr, extra_va=eva,
            )
            oof["offset"][va] = prediction.astype(np.float32)
            coefficients.append({"fold": fold_index,
                                 "held_out_gene": gene if cv_mode == "logo" else None,
                                 "descriptor_names": [basis_columns[1], *feature_cols],
                                 **details})

        yv = y[va]
        afm_at_05 = compute_binary_metrics_at_threshold(yv, raw[va, 0], 0.5)
        # One block per fitted panel member. Row names come from
        # FUSION_MODEL_ROW_NAME so the MLP keeps emitting "fusion".
        model_metrics = {}
        for name in models:
            row_name = FUSION_MODEL_ROW_NAME[name]
            decision = name in ("svm_rbf", "svm_linear")
            at_threshold = compute_binary_metrics_at_threshold(
                yv, oof[name][va], 0.0 if decision else 0.5)
            suffix = "decision_0" if decision else "05"
            model_metrics[row_name] = _auc(yv, oof[name][va])
            model_metrics[f"{row_name}_ap"] = _ap(yv, oof[name][va])
            for metric in ("f1", "precision", "recall", "mcc"):
                model_metrics[f"{row_name}_{metric}_{suffix}"] = at_threshold[metric]
        per_gene.append({
            "gene": gene, "n": int(len(va)), "n_pos": int(yv.sum()),
            **model_metrics,
            "afm": _auc(yv, raw[va, 0]), "d1": _auc(yv, df["d1_score"].to_numpy()[va]),
            "fusion_partner": partner,
            "afm_ap": _ap(yv, raw[va, 0]),
            "afm_f1_05": afm_at_05["f1"],
            "afm_precision_05": afm_at_05["precision"],
            "afm_recall_05": afm_at_05["recall"],
            "afm_mcc_05": afm_at_05["mcc"],
            # Flags folds whose training half was too small for an honest inner split,
            # so the epoch count fell back to the fixed budget (see train_one_fold).
            "used_fixed_epochs": bool(out.get("used_fixed_epochs", False)) if out else False,
            "best_epoch": int(out.get("best_epoch", 0)) if out else 0,
        })
        if progress_callback is not None:
            progress_callback(
                fold_index, len(splits),
                {
                    "held_out_gene": gene,
                    "status": "completed",
                    "best_epoch": int(out.get("best_epoch", 0)) if out else 0,
                },
            )
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError("CV did not produce finite OOF predictions for every variant")
    if cv_mode == "stratified":
        per_gene = []
        for gene in sorted(set(groups)):
            idx = groups == gene
            per_gene.append({
                "gene": gene, "n": int(idx.sum()), "n_pos": int(y[idx].sum()),
                "afm": _auc(y[idx], raw[idx, 0]),
                "d1": _auc(y[idx], df.loc[idx, "d1_score"].to_numpy()),
                **{FUSION_MODEL_ROW_NAME[m]: _auc(y[idx], values[idx])
                   for m, values in oof.items()},
            })
    return {
        "y": y, "afm": raw[:, 0], "d1": df["d1_score"].to_numpy(),
        "offset_coefficients": coefficients,
        "fit_diagnostics": fit_diagnostics,
        "models": [FUSION_MODEL_ROW_NAME[m] for m in models],
        "oof": {FUSION_MODEL_ROW_NAME[m]: oof[m] for m in models},
        "oof_fusion": oof.get("mlp"), "oof_logreg": oof.get("logreg"),
        "per_gene": pd.DataFrame(per_gene),
        "history": (
            pd.concat(histories, ignore_index=True)
            if histories else pd.DataFrame()
        ),
        "representation": fusion_representation,
        "normalization": fusion_normalization,
        "fusion_partner": partner,
    }


def logo_cv_plain(labeled_df, cfg, cols, progress_callback=None):
    """Structure-only LOGO CV over an arbitrary column set (no afm/d1 interaction
    featurization): logreg (stable) + a plain MLP, per-fold median-imputed and
    standardized. `cols` is e.g. ['d1_score', 'plddt_spread_ratio']."""
    seed_everything(cfg["seed"])
    df = labeled_df.sort_values(["gene", "mutation"]).reset_index(drop=True).copy()
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(int).to_numpy()
    groups = df["gene"].astype(str).to_numpy()

    oof_mlp = np.full(len(df), np.nan, dtype=np.float32)
    oof_logreg = np.full(len(df), np.nan, dtype=np.float32)
    per_gene = []
    splits = logo_splits(groups, cfg.get("logo_folds", "grouped"))
    for fold_index, (tr, va) in enumerate(splits, start=1):
        gene = groups[va][0]
        if len(np.unique(y[tr])) < 2:
            if progress_callback is not None:
                progress_callback(
                    fold_index, len(splits),
                    {"held_out_gene": gene, "status": "skipped"},
                )
            continue
        Xtr_i, Xva_i = _impute_extras(X, tr, va)  # imputes any NaN col (e.g. spread at size 1)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr_i).astype(np.float32)
        Xva = sc.transform(Xva_i).astype(np.float32)
        out = train_one_fold(
            Xtr, y[tr], Xva, y[va],
            epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            hidden=cfg["hidden"], dropout=cfg["dropout"], lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], patience=cfg["patience"],
            model_kind="plain",
        )
        oof_mlp[va] = out["val_probs"]
        lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                random_state=int(cfg["seed"])).fit(Xtr, y[tr])
        oof_logreg[va] = lr.predict_proba(Xva)[:, 1].astype(np.float32)
        yv = y[va]
        per_gene.append({"gene": gene, "n": int(len(va)), "n_pos": int(yv.sum()),
                         "mlp": _auc(yv, oof_mlp[va]), "logreg": _auc(yv, oof_logreg[va]),
                         "used_fixed_epochs": bool(out.get("used_fixed_epochs", False)),
                         "best_epoch": int(out.get("best_epoch", 0))})
        if progress_callback is not None:
            progress_callback(
                fold_index, len(splits),
                {
                    "held_out_gene": gene,
                    "status": "completed",
                    "best_epoch": int(out.get("best_epoch", 0)),
                },
            )
    return {"y": y, "oof_mlp": oof_mlp, "oof_logreg": oof_logreg,
            "per_gene": pd.DataFrame(per_gene)}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _seed_list(spec):
    """Parse non-negative seeds, accepting either ``42-51`` or ``42,43``."""
    text = str(spec).strip()
    try:
        if "-" in text and "," not in text:
            bounds = [part.strip() for part in text.split("-")]
            if len(bounds) != 2 or not all(bounds):
                raise ValueError
            start, stop = (int(value) for value in bounds)
            if stop < start:
                raise ValueError
            values = list(range(start, stop + 1))
        else:
            values = [int(value) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("seeds must be a comma list or ascending range, e.g. 42-51") from exc
    values = list(dict.fromkeys(values))
    if not values or any(value < 0 for value in values):
        raise ValueError("seeds must be non-negative integers")
    return values


def _positive_int_list(spec, name):
    """Parse and de-duplicate a required comma-separated positive integer list."""
    try:
        values = [int(value) for value in str(spec).split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated positive integers") from exc
    values = list(dict.fromkeys(values))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must be comma-separated positive integers")
    return values


def run_logo(labeled_df, base_cfg, seeds, feature_cols=None,
             fusion_representation="raw2", fusion_normalization="original"):
    per_gene_by_seed = []
    pooled_rows = []
    afm = labeled_df.sort_values(["gene", "mutation"]).reset_index(drop=True)["afm_score"].to_numpy()
    d1 = labeled_df.sort_values(["gene", "mutation"]).reset_index(drop=True)["d1_score"].to_numpy()
    y = labeled_df.sort_values(["gene", "mutation"]).reset_index(drop=True)["label"].astype(int).to_numpy()
    for sd in seeds:
        cfg = dict(base_cfg); cfg["seed"] = sd
        r = logo_cv(
            labeled_df, cfg, feature_cols=feature_cols,
            fusion_representation=fusion_representation,
            fusion_normalization=fusion_normalization,
        )
        pg = r["per_gene"]
        per_gene_by_seed.append(
            pg.assign(
                seed=sd, representation=fusion_representation,
                normalization=fusion_normalization,
            )
        )
        pooled_rows.append({
            "seed": sd,
            "representation": fusion_representation,
            "normalization": fusion_normalization,
            # pooled-OOF AUC (trained models penalised by cross-gene calibration)
            "fusion": _auc(r["y"], r["oof_fusion"]),
            "logreg": _auc(r["y"], r["oof_logreg"]),
            "afm": _auc(r["y"], r["afm"]),
            "d1": _auc(r["y"], r["d1"]),
            # macro = mean of per-gene AUCs: calibration-free, the FAIR afm-vs-fusion metric
            "fusion_macro": float(pg["fusion"].mean()),
            "logreg_macro": float(pg["logreg"].mean()),
            "afm_macro": float(pg["afm"].mean()),
            "d1_macro": float(pg["d1"].mean()),
        })
    per_gene = pd.concat(per_gene_by_seed, ignore_index=True)
    # AFM-only / D1-only are seed-independent (no training); average is exact.
    pooled = pd.DataFrame(pooled_rows)
    # bootstrap CI for pooled AFM (fixed) and pooled fusion (first seed's OOF as representative)
    ci_afm = _bootstrap_ci(y, afm)
    return {"per_gene": per_gene, "pooled": pooled, "ci_afm": ci_afm,
            "y": y, "afm": afm, "d1": d1}


def summarize_logo(res):
    pooled = res["pooled"]
    # pooled summary across seeds
    pooled_summary = pd.DataFrame([{
        "model": m,
        "auc_mean": pooled[m].mean(),
        "auc_std": pooled[m].std(ddof=0),
    } for m in ["afm", "d1", "logreg", "fusion",
                "afm_macro", "d1_macro", "logreg_macro", "fusion_macro"]])
    pooled_summary["representation"] = pooled["representation"].iloc[0]
    pooled_summary["normalization"] = pooled["normalization"].iloc[0]
    pooled_summary["n"] = len(res["y"])
    pooled_summary["n_pos"] = int(res["y"].sum())
    # per-gene summary across seeds (fusion trained; afm/d1 seed-independent)
    pg = res["per_gene"]
    agg_spec = dict(n=("n", "first"), n_pos=("n_pos", "first"),
                    afm=("afm", "mean"),
                    fusion_mean=("fusion", "mean"), fusion_std=("fusion", "std"),
                    logreg_mean=("logreg", "mean"))
    # Surface the fixed-epoch fallback so the honest per-gene table shows which genes
    # were too small for validation-based epoch selection.
    if "used_fixed_epochs" in pg.columns:
        agg_spec["n_folds_fixed_epochs"] = ("used_fixed_epochs", "sum")
    per_gene_summary = pg.groupby("gene").agg(**agg_spec).reset_index()
    per_gene_summary["representation"] = pg["representation"].iloc[0]
    per_gene_summary["normalization"] = pg["normalization"].iloc[0]
    per_gene_summary["fusion_minus_afm"] = per_gene_summary["fusion_mean"] - per_gene_summary["afm"]
    return pooled_summary, per_gene_summary


def run_stratified(labeled_df, base_cfg, seeds,
                   fusion_representation="raw2",
                   fusion_normalization="original"):
    rows = []
    df = labeled_df.sort_values(["gene", "mutation"]).reset_index(drop=True)
    y = df["label"].astype(int).to_numpy()
    afm = df["afm_score"].to_numpy()
    d1 = df["d1_score"].to_numpy()
    for sd in seeds:
        cfg = dict(base_cfg); cfg["seed"] = sd
        cv = run_cross_validated_training(
            df, cfg, log_to_wandb=False,
            fusion_representation=fusion_representation,
            fusion_normalization=fusion_normalization,
        )
        rows.append({
            "seed": sd,
            "representation": fusion_representation,
            "normalization": fusion_normalization,
            "fusion": cv["metrics"]["oof_auc"],
            "logreg": cv["baseline"]["metrics"]["oof_auc"],
            "afm": _auc(y, afm),
            "d1": _auc(y, d1),
        })
    strat = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "model": m, "auc_mean": strat[m].mean(), "auc_std": strat[m].std(ddof=0),
    } for m in ["afm", "d1", "logreg", "fusion"]])
    summary["representation"] = fusion_representation
    summary["normalization"] = fusion_normalization
    return summary, strat


# --------------------------------------------------------------------------- #
# ensemble-size sweep (deterministic local loop; optional live W&B progress)
# --------------------------------------------------------------------------- #
def _logo_monitor_metrics(size, per_gene, models=None):
    """Build the W&B evaluation schema from per-gene LOGO metrics.

    Every column is emitted only if present. The panel change made this
    necessary: "fusion" exists only when the MLP arm is fitted, and hardcoding
    it crashed the first cell of a --wandb sweep while every wandb-off smoke
    test and all unit tests passed straight over it.
    """
    if per_gene.empty:
        return {}

    def mean(column):
        if column not in per_gene.columns:
            return None
        value = float(per_gene[column].mean())
        return value if np.isfinite(value) else None

    payload = {"d1_size": int(size)}
    # Trained arms: whichever panel members produced columns.
    candidates = list(models) if models else [
        c for c in ("offset", "logreg", "gbm", "fusion") if c in per_gene.columns
    ]
    for name in candidates:
        for suffix, column in (
            ("auc", name), ("ap", f"{name}_ap"),
            ("f1_at_0_5", f"{name}_f1_05"),
            ("precision_at_0_5", f"{name}_precision_05"),
            ("recall_at_0_5", f"{name}_recall_05"),
            ("mcc_at_0_5", f"{name}_mcc_05"),
            *((f"{metric}_at_decision_0", f"{name}_{metric}_decision_0")
              for metric in ("f1", "precision", "recall", "mcc")),
        ):
            value = mean(column)
            if value is not None:
                payload[f"evaluation/{name}_macro_{suffix}"] = value
    # Untrained baselines and the MLP-only epoch diagnostic.
    for key, column in (
        ("evaluation/afm_macro_auc", "afm"), ("evaluation/afm_macro_ap", "afm_ap"),
        ("evaluation/d1_macro_auc", "d1"),
        ("evaluation/best_epoch_mean", "best_epoch"),
    ):
        value = mean(column)
        if value is not None:
            payload[key] = value
    return payload


def _wandb_log_training_curve(wb, history):
    """Average honest inner-selection histories across outer LOGO folds."""
    if wb is None or history is None or history.empty:
        return
    selected = history[history["phase"] == "select"].copy()
    if selected.empty:
        wb.summary["training/curve_status"] = "unavailable: fixed-epoch fallback"
        return
    columns = [
        "train_loss", "val_loss", "val_ap", "val_auc", "val_f1_05",
        "val_precision_05", "val_recall_05", "val_mcc_05",
    ]
    curve = selected.groupby("epoch").agg(
        **{f"{column}_mean": (column, "mean") for column in columns},
        active_outer_folds=("held_out_gene", "nunique"),
    ).reset_index()
    for _, row in curve.iterrows():
        wb.log({
            "training/epoch": int(row["epoch"]),
            "training/train_loss_mean": float(row["train_loss_mean"]),
            "training/inner_val_loss_mean": float(row["val_loss_mean"]),
            "training/inner_val_ap_mean": float(row["val_ap_mean"]),
            "training/inner_val_auc_mean": float(row["val_auc_mean"]),
            "training/inner_val_f1_at_0_5_mean": float(row["val_f1_05_mean"]),
            "training/inner_val_precision_at_0_5_mean": float(
                row["val_precision_05_mean"]
            ),
            "training/inner_val_recall_at_0_5_mean": float(
                row["val_recall_05_mean"]
            ),
            "training/inner_val_mcc_at_0_5_mean": float(row["val_mcc_05_mean"]),
            "training/active_outer_folds": int(row["active_outer_folds"]),
        })
    wb.summary["training/curve_status"] = "inner-split mean across outer LOGO folds"


def _wandb_store_evaluation_summary(wb, payload):
    """Keep one candidate's final LOGO diagnostics out of its epoch history."""
    if wb is None:
        return
    for metric, value in payload.items():
        wb.summary[metric] = value


def _seeds_for_size(seed_plan, size, seeds):
    """Seeds this ensemble size runs.

    `seed_plan` (size -> seed list) comes from a variance-proportional
    allocation, where small N gets more repeats than large N because sampling
    variance carries the finite-population factor (M-N)/(M-1). None means the
    legacy uniform allocation: every size runs every seed.
    """
    if not seed_plan:
        return list(seeds)
    return list(seed_plan.get(int(size), []))


def _coefficient_rows(result, context):
    if not result.get("offset_coefficients"):
        return []
    return [{**context, "model": "offset", "scope": "coefficients", "auc": np.nan,
             "offset_coefficients_json": json.dumps(result["offset_coefficients"])}]


def _panel_rows(result, context, *, baselines=False):
    rows = []
    pg = result["per_gene"]
    for model in result["models"] + (["afm", "d1"] if baselines else []):
        baseline = model in ("afm", "d1")
        common = {**context, "model": model}
        if baseline:
            common.update(representation="baseline", normalization="baseline",
                          variant="base", fusion_partner="baseline")
        prediction = result[model] if baseline else result["oof"][model]
        rows.append({**common, "scope": "pooled", "auc": _auc(result["y"], prediction),
                     "fit_diagnostics_json": json.dumps([
                         fit for fit in result.get("fit_diagnostics", [])
                         if fit["model"] == model])})
        rows.append({**common, "scope": "pooled_macro", "auc": float(pg[model].mean())})
        rows.extend({**common, "scope": g["gene"], "auc": g[model]}
                    for _, g in pg.iterrows())
    return rows + _coefficient_rows(result, context)


def _check_sweep_configuration(out, configuration, resume):
    """Refuse to mix rows from different experiments or source revisions."""
    from metric_analysis.pipeline.library import canonical_cohort, descriptor_metrics
    sources = [Path(__file__), Path(core.__file__), Path(ensemble_shift.__file__),
               Path(canonical_cohort.__file__), Path(descriptor_metrics.__file__),
               Path(core.d1_runtime.__file__),
               Path(__file__).with_name("benchmark.py"), Path(STRICT_COHORT_MANIFEST)]
    configuration = json.loads(json.dumps({
        **configuration, "schema": 2,
        "protein_config": protein_config_fingerprint(),
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }, default=str))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "sweep_configuration.json"
    if not resume and (out / "sweep_runs_partial.csv").exists():
        raise ValueError("output contains a checkpoint; resume it or use a new output directory")
    if resume and (out / "sweep_runs_partial.csv").exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != configuration:
            raise ValueError("checkpoint configuration/source mismatch; use a new output directory")
    else:
        manifest.write_text(json.dumps(configuration, indent=2) + "\n")


def run_sweep(base_cfg, sizes, seeds, do_stratified, out, feature_variants=None,
              fusion_representations=None, fusion_normalizations=None,
              gt_groups=None, fusion_partners=None, resume=False, fusion_models=None,
              wandb_enabled=False, wandb_project="afm_d1_fusion_training",
              wandb_entity=None, force=False, seed_plan=None, save_oof=False,
              exclude_arms=()):
    """Cross size x representation x extra-feature variant x GT group.

    For each (seed, size) the dataset is built once (over every gene), then each
    GT group is a row-subset of that frame -- so grouping adds no extra D1 build.
    Within a group every AFM+D1 representation, normalization, and extra-feature
    variant is evaluated under LOGO. The MLP and logistic comparator consume the
    identical fold-standardized matrix. Robust/rank use each gene's complete
    unlabelled batch and are transductive diagnostics. AFM/D1 baselines are
    recorded once per group. Stratified evaluation uses the same arms and model
    panel with five gene-by-label folds. W&B, if enabled, uses one run per candidate
    fit for epoch/progress monitoring and one run per seed/configuration for the
    connected ensemble-size evaluation curve; the local CSVs and figures remain
    authoritative.
    """
    feature_variants = list(feature_variants or ["base"])
    fusion_representations = list(fusion_representations or ["raw2"])
    fusion_normalizations = list(fusion_normalizations or ["original"])
    gt_groups = list(gt_groups or ["all"])
    fusion_partners = list(fusion_partners or ["d1"])
    fusion_models = list(fusion_models or DEFAULT_FUSION_MODELS)
    if "gbm" in fusion_models and any(r != "raw2" for r in fusion_representations):
        raise ValueError("monotone GBM requires raw2; remove gbm for engineered sensitivity runs")
    # The partner sits INSIDE the basis (afm, partner); variants APPEND to it.
    # So (partner=ks, variant=ks) would put plddt_ks in twice -- a degenerate
    # duplicate column, not a model anyone wants. Skipped rather than fitted.
    combinations = []
    _seen_bases = set()
    for partner in fusion_partners:
        for representation in fusion_representations:
            for normalization in fusion_normalizations:
                for variant in feature_variants:
                    if (partner, variant) in exclude_arms:
                        continue
                    cols = [FUSION_PARTNERS[partner], *_feature_cols(variant)]
                    if len(set(cols)) != len(cols):
                        continue  # partner duplicated by the variant
                    # Preserve the partner role: it affects normalization,
                    # engineered interactions, and the single-slope offset.
                    key = (representation, normalization, partner, tuple(_feature_cols(variant)))
                    if key in _seen_bases:
                        continue
                    _seen_bases.add(key)
                    combinations.append(
                        (representation, normalization, variant, partner)
                    )
    oof_rows = []
    _check_sweep_configuration(out, {
        "cfg": base_cfg, "sizes": list(sizes), "seeds": list(seeds),
        "seed_plan": seed_plan, "stratified": do_stratified,
        "combinations": combinations, "models": fusion_models, "gt_groups": gt_groups,
    }, resume)
    rows = []
    # Resume support. The checkpoint below is written only after every GT group
    # of a (seed, size) cell has been appended, and via an atomic rename, so any
    # cell present in it is COMPLETE -- which makes (seed, size) a safe skip key.
    # Rows are carried forward so the final table is indistinguishable from an
    # uninterrupted run. Opt-in: without --resume the behaviour is unchanged.
    done_cells = set()
    checkpoint_path = Path(out) / "sweep_runs_partial.csv"
    if resume and checkpoint_path.is_file():
        previous = pd.read_csv(checkpoint_path)
        if not previous.empty and {"seed", "size"} <= set(previous.columns):
            rows.extend(previous.to_dict("records"))
            done_cells = {
                (int(a), int(b)) for a, b in zip(previous["seed"], previous["size"])
            }
            print(f"[sweep] resume: {len(rows)} rows from {len(done_cells)} "
                  f"completed (seed, size) cells in {checkpoint_path.name}")
        else:
            print(f"[sweep] resume: {checkpoint_path.name} unusable, starting fresh")
    elif resume:
        print("[sweep] resume: no checkpoint found, starting fresh")

    for seed in seeds:
        size_curve_payloads = {}
        for size in sizes:
            if seed not in _seeds_for_size(seed_plan, size, seeds):
                continue
            if (int(seed), int(size)) in done_cells:
                print(f"  [sweep] seed={seed} size={size:>2} -- skipped, already in checkpoint")
                continue
            cfg = dict(base_cfg); cfg["seed"] = seed; cfg["d1_max_structures_per_variant"] = size
            df = build_or_load_dataset(size, cfg, force=force)

            summary_by_group = {}
            for gname in gt_groups:
                gdf = _gt_subset(df, gname)
                dfs = gdf.sort_values(["gene", "mutation"]).reset_index(drop=True)
                y = dfs["label"].astype(int).to_numpy()
                afm = dfs["afm_score"].to_numpy()
                d1v = dfs["d1_score"].to_numpy()

                # Raw baselines do not depend on representation or extras.
                afm_auc, d1_auc = _auc(y, afm), _auc(y, d1v)
                rows.append({"size": size, "seed": seed, "cv": "logo", "model": "afm", "representation": "baseline", "normalization": "baseline", "variant": "base", "scope": "pooled", "gt_group": gname, "auc": afm_auc})
                rows.append({"size": size, "seed": seed, "cv": "logo", "model": "d1", "representation": "baseline", "normalization": "baseline", "variant": "base", "scope": "pooled", "gt_group": gname, "auc": d1_auc})

                fusion_by_variant = {}
                available_combinations = [
                    (representation, normalization, variant, partner)
                    for representation, normalization, variant, partner in combinations
                    if size >= 2 or not any(
                        c in ("plddt_ks_residue", "plddt_ks_centred") and gdf[c].isna().all()
                        for c in [FUSION_PARTNERS[partner], *_feature_cols(variant)]
                    )
                ]
                for ci, (representation, normalization, variant, partner) in enumerate(available_combinations):
                    cfg_c = dict(cfg); cfg_c["fusion_partner"] = partner
                    wb = _wandb_start(
                        wandb_project,
                        wandb_entity,
                        f"fusion-{gname}-{partner}-{representation}-{normalization}-{variant}",
                        (
                            f"{gname}-{partner}-{representation}-{normalization}-{variant}"
                            f"-seed{seed}-N{size}"
                        ),
                        {
                            "seed": int(seed),
                            "d1_size": int(size),
                            "gt_group": gname,
                            "representation": representation,
                            "normalization": normalization,
                            "feature_variant": variant,
                            "fusion_partner": partner,
                            "normalization_scope": (
                                "within-gene complete unlabelled batch"
                                if normalization in ("robust", "rank")
                                else "inductive original scores"
                            ),
                            "epochs": base_cfg["epochs"],
                            "hidden": base_cfg["hidden"],
                            "dropout": base_cfg["dropout"],
                            "lr": base_cfg["lr"],
                            "weight_decay": base_cfg["weight_decay"],
                            "patience": base_cfg["patience"],
                            "batch_size": base_cfg["batch_size"],
                            "wandb_role": (
                                "candidate training curve with evaluation summary"
                            ),
                            "training_curve_semantics": (
                                "mean train-only inner-selection history across outer LOGO folds"
                            ),
                            "threshold_metrics": (
                                "macro mean of per-gene metrics at probability 0.5 "
                                "or SVM decision score 0; calibration-sensitive diagnostic"
                            ),
                        },
                        tags=[
                            f"gt:{gname}",
                            f"partner:{partner}",
                            f"variant:{variant}",
                            f"repr:{representation}",
                            f"norm:{normalization}",
                            f"N:{size}",
                            f"seed:{seed}",
                            "cv:logo",
                            f"sweep:{Path(out).name}",
                        ],
                    ) if wandb_enabled else None
                    progress = _WandbProgress(wb, 1) if wb is not None else None
                    label = (
                        f"LOGO N={size} gt={gname} "
                        f"{representation}/{normalization}/{variant}"
                    )
                    logo_kwargs = {
                        "models": list(fusion_models),
                        "feature_cols": _feature_cols(variant),
                        "fusion_representation": representation,
                        "fusion_normalization": normalization,
                    }
                    if progress is not None:
                        logo_kwargs["progress_callback"] = progress.callback(label)
                    r = logo_cv(gdf, cfg_c, **logo_kwargs)
                    if progress is not None:
                        progress.complete(label)
                    if save_oof:
                        # Per-variant out-of-fold scores. Without them the only
                        # uncertainty available on a result is the across-seed
                        # SD, which measures WHICH STRUCTURES were drawn and not
                        # which variants are in the cohort -- on this cohort the
                        # latter is 5-40x larger. A paired bootstrap needs the
                        # per-variant scores, so they have to survive the run.
                        keyed = gdf.sort_values(["gene", "mutation"])
                        base = {
                            "size": size, "seed": seed, "cv": "logo",
                            "gt_group": gname, "representation": representation,
                            "normalization": normalization, "variant": variant,
                            "fusion_partner": partner,
                        }
                        for mname, scores in r["oof"].items():
                            oof_rows.append(pd.DataFrame({
                                **base, "model": mname,
                                "gene": keyed["gene"].to_numpy(),
                                "mutation": keyed["mutation"].to_numpy(),
                                "y": r["y"], "score": np.asarray(scores, dtype=float),
                            }))
                        oof_rows.append(pd.DataFrame({
                            **base, "model": "afm",
                            "gene": keyed["gene"].to_numpy(),
                            "mutation": keyed["mutation"].to_numpy(),
                            "y": r["y"], "score": np.asarray(r["afm"], dtype=float),
                        }))
                    pg = r["per_gene"]
                    if wandb_enabled:
                        evaluation_payload = _logo_monitor_metrics(size, pg, r["models"])
                        curve_key = (gname, representation, normalization, variant, partner)
                        size_curve_payloads.setdefault(curve_key, []).append(
                            evaluation_payload
                        )
                        _wandb_store_evaluation_summary(wb, evaluation_payload)
                    _wandb_log_training_curve(wb, r.get("history"))
                    panel = r["models"]
                    # scope='pooled'       : AUC over pooled out-of-fold predictions. For TRAINED
                    #   models this conflates cross-gene calibration (each LOGO fold is a different
                    #   model), penalising them vs raw AFM -> NOT a fair afm-vs-fusion comparison.
                    # scope='pooled_macro' : mean of per-gene AUCs. Calibration-free and directly
                    #   comparable to AFM (globally calibrated by construction). Use this as the
                    #   headline cross-gene LOGO number.
                    macro_by_model = {m: float(pg[m].mean()) for m in panel}
                    pooled_by_model = {m: _auc(r["y"], r["oof"][m]) for m in panel}
                    # Console summary shows the first eligible panel member.
                    # This is a progress display, not selection of a headline.
                    fusion_by_variant[(representation, normalization, variant, partner)] = \
                        macro_by_model[panel[0]]
                    for m, v, vm in [(m, pooled_by_model[m], macro_by_model[m]) for m in panel]:
                        rows.append({"size": size, "seed": seed, "cv": "logo", "model": m, "representation": representation, "normalization": normalization, "variant": variant, "fusion_partner": partner, "scope": "pooled", "gt_group": gname, "auc": v})
                        rows[-1]["fit_diagnostics_json"] = json.dumps([
                            fit for fit in r.get("fit_diagnostics", []) if fit["model"] == m])
                        rows.append({"size": size, "seed": seed, "cv": "logo", "model": m, "representation": representation, "normalization": normalization, "variant": variant, "fusion_partner": partner, "scope": "pooled_macro", "gt_group": gname, "auc": vm})
                    for _, g in pg.iterrows():
                        for m in panel:
                            rows.append({"size": size, "seed": seed, "cv": "logo", "model": m, "representation": representation, "normalization": normalization, "variant": variant, "fusion_partner": partner, "scope": g["gene"], "gt_group": gname, "auc": g[m]})
                        if ci == 0:  # raw baselines per gene once
                            rows.append({"size": size, "seed": seed, "cv": "logo", "model": "afm", "representation": "baseline", "normalization": "baseline", "variant": "base", "scope": g["gene"], "gt_group": gname, "auc": g["afm"]})
                            rows.append({"size": size, "seed": seed, "cv": "logo", "model": "d1", "representation": "baseline", "normalization": "baseline", "variant": "base", "scope": g["gene"], "gt_group": gname, "auc": g["d1"]})
                    if ci == 0:  # fair raw-baseline macro once
                        rows.append({"size": size, "seed": seed, "cv": "logo", "model": "afm", "representation": "baseline", "normalization": "baseline", "variant": "base", "scope": "pooled_macro", "gt_group": gname, "auc": float(pg["afm"].mean())})
                        rows.append({"size": size, "seed": seed, "cv": "logo", "model": "d1", "representation": "baseline", "normalization": "baseline", "variant": "base", "scope": "pooled_macro", "gt_group": gname, "auc": float(pg["d1"].mean())})
                    context = {"size": size, "seed": seed, "gt_group": gname,
                               "representation": representation, "normalization": normalization,
                               "variant": variant, "fusion_partner": partner}
                    rows.extend(_coefficient_rows(r, {**context, "cv": "logo"}))
                    if do_stratified:
                        sc = logo_cv(gdf, cfg_c, cv_mode="stratified",
                                     **{k: v for k, v in logo_kwargs.items()
                                        if k != "progress_callback"})
                        rows.extend(_panel_rows(sc, {**context, "cv": "stratified"},
                                                baselines=(ci == 0)))
                    if wb is not None:
                        progress.finish()
                        wb.finish()

                summary_by_group[gname] = (afm_auc, fusion_by_variant)

            for gname in gt_groups:
                afm_auc, fusion_by_variant = summary_by_group[gname]
                fus_str = " ".join(
                    f"{part}+{rep}/{norm}/{variant}={fusion_by_variant[(rep, norm, variant, part)]:.3f}"
                    for rep, norm, variant, part in fusion_by_variant
                )
                print(f"  [sweep] seed={seed} size={size:>2} gt={gname:<10} afm={afm_auc:.3f}  LOGO fusion[{fus_str}]")

            # Checkpoint after every (seed, size). sweep_runs.csv is only written
            # once every loop finishes, so a 7-size x 10-seed x 6-arm run that
            # dies late used to lose everything. This is a crash-recovery
            # artifact, not a result: the authoritative table is still
            # sweep_runs.csv written at the end.
            try:
                partial = pd.DataFrame(rows)
                if not partial.empty:
                    tmp = checkpoint_path.with_suffix(".csv.tmp")
                    Path(out).mkdir(parents=True, exist_ok=True)
                    partial.to_csv(tmp, index=False)
                    tmp.replace(checkpoint_path)
            except Exception as exc:  # never let checkpointing kill the sweep
                print(f"  [sweep] checkpoint failed (continuing): {exc}")

        # A training run has epoch as its only meaningful history axis. Create
        # separate compact runs after all sizes for this seed so W&B can draw
        # genuine connected size curves instead of one-point pseudo-curves.
        if wandb_enabled:
            for (gname, representation, normalization, variant, partner), payloads in sorted(
                size_curve_payloads.items()
            ):
                curve_wb = _wandb_start(
                    wandb_project,
                    wandb_entity,
                    f"fusion-size-curve-{gname}-{partner}-{representation}-{normalization}-{variant}",
                    (
                        f"{gname}-{partner}-{representation}-{normalization}-{variant}"
                        f"-size-curve-seed{seed}"
                    ),
                    {
                        "seed": int(seed),
                        "d1_sizes": sorted(int(payload["d1_size"]) for payload in payloads),
                        "gt_group": gname,
                        "representation": representation,
                        "normalization": normalization,
                        "feature_variant": variant,
                        "fusion_partner": partner,
                        "normalization_scope": (
                            "within-gene complete unlabelled batch"
                            if normalization in ("robust", "rank")
                            else "inductive original scores"
                        ),
                        "epochs": base_cfg["epochs"],
                        "hidden": base_cfg["hidden"],
                        "dropout": base_cfg["dropout"],
                        "lr": base_cfg["lr"],
                        "weight_decay": base_cfg["weight_decay"],
                        "patience": base_cfg["patience"],
                        "batch_size": base_cfg["batch_size"],
                        "wandb_role": "ensemble-size evaluation curve",
                        "evaluation_curve_semantics": (
                            "macro mean of metrics computed separately within each "
                            "outer held-out gene"
                        ),
                        "threshold_metrics": (
                            "macro mean of per-gene metrics at probability 0.5 "
                            "or SVM decision score 0; calibration-sensitive diagnostic"
                        ),
                    },
                )
                for payload in sorted(payloads, key=lambda item: item["d1_size"]):
                    curve_wb.log(payload)
                curve_wb.summary["evaluation/curve_status"] = "completed"
                curve_wb.finish()
    runs_df = pd.DataFrame(rows)
    _policy = core.sampling_policy_from_config(base_cfg)
    runs_df["sampling_mode"] = _policy.mode
    runs_df["selection"] = _policy.selection
    mlp_settings = {
        "logo_folds": base_cfg.get("logo_folds", "grouped"),
        "mlp_epochs": int(base_cfg["epochs"]),
        "mlp_hidden": int(base_cfg["hidden"]),
        "mlp_dropout": float(base_cfg["dropout"]),
        "mlp_lr": float(base_cfg["lr"]),
        "mlp_weight_decay": float(base_cfg["weight_decay"]),
        "mlp_patience": int(base_cfg["patience"]),
        "mlp_batch_size": int(base_cfg["batch_size"]),
    }
    for column, value in mlp_settings.items():
        runs_df[column] = value
    runs_df["fusion_partner"] = runs_df["fusion_partner"].fillna("baseline")
    runs_df.to_csv(out / "sweep_runs.csv", index=False)
    if save_oof and oof_rows:
        oof = pd.concat(oof_rows, ignore_index=True)
        oof.to_csv(out / "oof_predictions.csv.gz", index=False, compression="gzip")
        print(f"[sweep] wrote {len(oof):,} out-of-fold rows -> oof_predictions.csv.gz")
    summ = (runs_df[runs_df["scope"] != "coefficients"].groupby(["cv", "model", "representation", "normalization", "variant", "fusion_partner", "scope", "gt_group", "size", "sampling_mode", "selection"])["auc"]
                   .agg(auc_mean="mean", auc_std="std", n_seeds="count").reset_index())
    for column, value in mlp_settings.items():
        summ[column] = value
    summ.to_csv(out / "sweep_summary.csv", index=False)

    # One base figure per GT group; single-group 'all' keeps the historical name.
    for gname in gt_groups:
        gsumm = summ[summ["gt_group"] == gname]
        suffix = "" if (gname == "all" and gt_groups == ["all"]) else f"_{gname}"
        primary_representation = (
            "raw2" if "raw2" in fusion_representations
            else fusion_representations[0]
        )
        primary_normalization = (
            "original" if "original" in fusion_normalizations
            else fusion_normalizations[0]
        )
        _plot_sweep_panel(
            gsumm, out / f"sweep_auc_vs_size{suffix}.png",
            primary_representation, primary_normalization,
        )
        _plot_sweep_panel(
            gsumm, out / f"sweep_pooled_diagnostic{suffix}.png",
            primary_representation, primary_normalization, scope="pooled",
        )
        _plot_sweep_panel(
            gsumm, out / f"sweep_macro_delta_vs_afm{suffix}.png",
            primary_representation, primary_normalization, delta=True,
        )
        if len(fusion_representations) > 1:
            _plot_sweep_panel(gsumm, out / f"sweep_representation_ablation{suffix}.png")
        if len(fusion_normalizations) > 1:
            _plot_sweep_panel(gsumm, out / f"sweep_normalization_ablation{suffix}.png")
        if len(feature_variants) > 1 or feature_variants != ["base"]:
            _plot_sweep_panel(gsumm, out / f"sweep_fusion_ablation{suffix}.png",
                              primary_representation, primary_normalization)
    return runs_df, summ


def _plot_sweep_panel(summary, path, representation=None, normalization=None,
                      *, delta=False, scope="pooled_macro"):
    """Keep every model and descriptor arm separate, including new panel models."""
    import matplotlib.pyplot as plt
    sub = summary[summary.scope == scope]
    if representation is not None:
        sub = _primary_fusion_rows(sub, representation, normalization)
    fig, ax = plt.subplots(figsize=(12, 6))
    refs = sub[sub.model == "afm"].set_index(["cv", "size"])["auc_mean"]
    keys = ["cv", "model", "fusion_partner", "variant", "representation", "normalization"]
    for key, block in sub.groupby(keys):
        cv, model, partner, variant, rep, norm = key
        if delta and model in ("afm", "d1"):
            continue
        block = block.sort_values("size")
        values = block.auc_mean.to_numpy().copy()
        if delta:
            values -= np.array([refs.loc[(cv, size)] for size in block["size"]])
        label = f"{cv}: {model}"
        if model not in ("afm", "d1"):
            label += f" AM+{partner}/{variant} ({rep}, {norm})"
        ax.errorbar(block["size"], values, yerr=block.auc_std.fillna(0),
                    marker="o", capsize=2, label=label)
    if delta:
        ax.axhline(0, color="0.4", ls="--")
    ax.set_xlabel("Structures per variant (N)")
    ax.set_ylabel(("AUC change vs AM" if delta else "ROC-AUC") +
                  (" — per-gene macro" if scope == "pooled_macro" else " — pooled OOF diagnostic"))
    ax.set_title("Fusion model panel; error bars show across-seed SD")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _primary_fusion_rows(summ, representation, normalization):
    """Keep one fusion specification plus representation-independent baselines."""
    return summ[
        (
            (summ["representation"] == representation)
            & (summ["normalization"] == normalization)
        )
        | (summ["representation"] == "baseline")
    ]


def _plot_base_figure(summ, path, representation="raw2",
                      normalization="original"):
    """Primary size figure for one explicit AFM+D1 representation."""
    import matplotlib.pyplot as plt
    colors = {"afm": "C0", "d1": "C1", "logreg": "C2", "fusion": "C3"}
    base = _primary_fusion_rows(
        summ[summ["variant"] == "base"], representation, normalization,
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sub = base[(base["cv"] == "logo") & (base["scope"] == "pooled_macro")]
    for m in ["afm", "d1", "logreg", "fusion"]:
        s = sub[sub["model"] == m].sort_values("size")
        if s.empty:
            continue
        ax.plot(s["size"], s["auc_mean"], marker="o", lw=2.0, color=colors[m], label=m)
        ax.fill_between(s["size"], s["auc_mean"] - s["auc_std"],
                        s["auc_mean"] + s["auc_std"], color=colors[m], alpha=0.12)
    ax.set_xlabel("D1 ensemble size (structures per variant)")
    ax.set_ylabel("Macro-LOGO ROC-AUC (mean held-out-gene AUC)")
    ax.set_title(
        f"AFM+D1 fusion ({representation}, {normalization}) -- cross-gene discrimination"
    )
    ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_pooled_diagnostic(summ, path, representation="raw2",
                            normalization="original"):
    """Diagnostic pooled-OOF/stratified figure; never used as the headline."""
    import matplotlib.pyplot as plt
    colors = {"afm": "C0", "d1": "C1", "logreg": "C2", "fusion": "C3"}
    base = _primary_fusion_rows(
        summ[summ["variant"] == "base"], representation, normalization,
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cv_mode, ls, label in [
        ("logo", "-", "pooled OOF LOGO"),
        ("stratified", "--", "stratified k-fold"),
    ]:
        sub = base[(base["cv"] == cv_mode) & (base["scope"] == "pooled")]
        for model in ["afm", "d1", "logreg", "fusion"]:
            s = sub[sub["model"] == model].sort_values("size")
            if s.empty:
                continue
            ax.plot(s["size"], s["auc_mean"], marker="o", ls=ls, lw=1.7,
                    color=colors[model], label=model if cv_mode == "logo" else None)
    ax.plot([], [], color="0.35", ls="-", label="pooled OOF LOGO")
    if (base["cv"] == "stratified").any():
        ax.plot([], [], color="0.35", ls="--", label="stratified k-fold")
    ax.set_xlabel("D1 ensemble size (structures per variant)")
    ax.set_ylabel("Pooled ROC-AUC")
    ax.set_title("Calibration-sensitive diagnostic (not the AFM-vs-fusion headline)")
    ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_macro_delta(summ, path, representation="raw2",
                      normalization="original"):
    """Paired macro-LOGO change from AFM, which makes small gains legible."""
    import matplotlib.pyplot as plt
    logo = _primary_fusion_rows(
        summ[(summ["cv"] == "logo") & (summ["scope"] == "pooled_macro")],
        representation, normalization,
    )
    base_afm = logo[(logo["model"] == "afm") & (logo["variant"] == "base")]
    if base_afm.empty:
        return
    reference = base_afm.set_index("size")["auc_mean"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"logreg": "C2", "fusion": "C3"}
    for model in ["logreg", "fusion"]:
        s = logo[(logo["model"] == model) & (logo["variant"] == "base")].sort_values("size").copy()
        if s.empty:
            continue
        s["delta"] = s["auc_mean"] - s["size"].map(reference)
        ax.plot(s["size"], s["delta"], marker="o", lw=2, color=colors[model], label=model)
        ax.fill_between(s["size"], s["delta"] - s["auc_std"],
                        s["delta"] + s["auc_std"], color=colors[model], alpha=0.12)
    ax.axhline(0, color="0.25", lw=1)
    ax.set_xlabel("D1 ensemble size (structures per variant)")
    ax.set_ylabel("Macro-LOGO AUC change vs AlphaMissense")
    ax.set_title(
        f"Paired cross-gene effect ({representation}, {normalization})"
    )
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_fusion_ablation(summ, feature_variants, path,
                          representation="raw2", normalization="original"):
    """Ensemble-shift ablation: macro-LOGO fusion AUC vs size, one line
    per feature variant, against the flat AFM reference. Answers 'does adding a
    spread / distance feature lift cross-gene generalization?' at a glance."""
    import matplotlib.pyplot as plt
    logo = summ[
        (summ["cv"] == "logo")
        & (summ["scope"] == "pooled_macro")
        & (
            (
                (summ["representation"] == representation)
                & (summ["normalization"] == normalization)
            )
            | (summ["representation"] == "baseline")
        )
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # AFM reference (variant-independent, flat).
    a = logo[logo["model"] == "afm"].sort_values("size")
    if not a.empty:
        ax.plot(a["size"], a["auc_mean"], marker="s", lw=1.6, ls=":", color="0.35",
                label="AlphaMissense (ref)")
    cmap = plt.get_cmap("viridis")
    for i, variant in enumerate(feature_variants):
        s = logo[(logo["model"] == "fusion") & (logo["variant"] == variant)].sort_values("size")
        if s.empty:
            continue
        color = cmap(i / max(len(feature_variants) - 1, 1))
        ax.plot(s["size"], s["auc_mean"], marker="o", lw=2.0, color=color, label=f"fusion:{variant}")
        ax.fill_between(s["size"], s["auc_mean"] - s["auc_std"],
                        s["auc_mean"] + s["auc_std"], color=color, alpha=0.10)
    ax.set_xlabel("D1 ensemble size (structures per variant)")
    ax.set_ylabel("Macro-LOGO ROC-AUC")
    ax.set_title(
        f"Ensemble-shift ablation ({representation}, {normalization}) -- held-out-gene discrimination"
    )
    ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_representation_ablation(
    summ, representations, path, normalization="original",
):
    """Base-extras comparison of AFM+D1 bases under one normalization."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    logo = summ[
        (summ["cv"] == "logo")
        & (summ["scope"] == "pooled_macro")
        & (summ["variant"] == "base")
        & (
            (summ["normalization"] == normalization)
            | (summ["representation"] == "baseline")
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    afm = logo[logo["model"] == "afm"].sort_values("size")
    if not afm.empty:
        ax.plot(
            afm["size"], afm["auc_mean"], marker="s", ls=":", lw=1.7,
            color="0.3", label="AlphaMissense",
        )
    representation_colors = {
        "raw2": "#6A3D9A",          # purple
        "compact4_v2": "#0072B2",   # blue
        "compact4": "#009E73",      # legacy teal
        "engineered6": "#D55E00",   # high-contrast vermillion
    }
    fallback_colors = plt.get_cmap("tab10")
    for i, representation in enumerate(representations):
        color = representation_colors.get(representation, fallback_colors(i % 10))
        for model, linestyle, marker in [
            ("fusion", "-", "o"), ("logreg", "--", "D"),
        ]:
            s = logo[
                (logo["model"] == model)
                & (logo["representation"] == representation)
            ].sort_values("size")
            if s.empty:
                continue
            ax.plot(
                s["size"], s["auc_mean"], marker=marker, lw=1.9,
                ls=linestyle, color=color,
            )
            ax.fill_between(
                s["size"], s["auc_mean"] - s["auc_std"],
                s["auc_mean"] + s["auc_std"], color=color, alpha=0.08,
            )
    ax.set_xlabel("D1 ensemble size (structures per variant)")
    ax.set_ylabel("Macro-LOGO ROC-AUC")
    ax.set_title(
        f"AFM+D1 representation ablation ({normalization}) — matched inputs per model pair"
    )
    model_handles = [
        Line2D([0], [0], color="0.3", marker="s", ls=":", lw=1.7,
               label="AlphaMissense reference"),
        Line2D([0], [0], color="black", marker="o", ls="-", lw=1.9,
               label="MLP fusion"),
        Line2D([0], [0], color="black", marker="D", ls="--", lw=1.9,
               label="Logistic regression"),
    ]
    representation_display = {
        "raw2": "raw2",
        "compact4_v2": "compact4-v2",
        "compact4": "compact4 (legacy)",
        "engineered6": "engineered6",
    }
    representation_handles = [
        Line2D([0], [0], color=representation_colors.get(
                   representation, fallback_colors(i % 10)),
               marker="o", ls="none", markersize=7,
               label=representation_display.get(representation, representation))
        for i, representation in enumerate(representations)
    ]
    model_legend = ax.legend(
        handles=model_handles, title="Model", loc="upper center",
        bbox_to_anchor=(0.28, -0.13), ncol=3, fontsize=8, title_fontsize=8,
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=representation_handles, title="Input basis", loc="upper center",
        bbox_to_anchor=(0.77, -0.13), ncol=max(1, len(representation_handles)),
        fontsize=8, title_fontsize=8,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_normalization_ablation(summ, representations, normalizations, path):
    """Readable macro-LOGO normalization comparison faceted by model and basis.

    Keeping representation in columns and model in rows prevents a full factorial
    sweep from becoming one unreadable spaghetti plot. Robust/rank are explicitly
    labelled as transductive because the held-out gene's unlabelled score batch is
    needed to define them.
    """
    import matplotlib.pyplot as plt

    logo = summ[
        (summ["cv"] == "logo")
        & (summ["scope"] == "pooled_macro")
        & (summ["variant"] == "base")
    ]
    if logo.empty:
        return

    models = [model for model in ["logreg", "fusion"]
              if (logo["model"] == model).any()]
    representations = [representation for representation in representations
                       if (logo["representation"] == representation).any()]
    if not models or not representations:
        return

    fig, axes = plt.subplots(
        len(models), len(representations),
        figsize=(4.6 * len(representations), 3.5 * len(models) + 1.1),
        sharex=True, sharey=True, squeeze=False,
    )
    colors = {
        "original": "#2563eb",
        "robust": "#d97706",
        "rank": "#7c3aed",
    }
    display = {
        "original": "original (inductive)",
        "robust": "robust (transductive)",
        "rank": "rank (transductive)",
    }
    afm = logo[logo["model"] == "afm"].sort_values("size")
    sizes = sorted(logo["size"].dropna().unique())
    fixed_size = len(sizes) == 1
    handles, labels = [], []
    for row, model in enumerate(models):
        for col, representation in enumerate(representations):
            ax = axes[row, col]
            if fixed_size and not afm.empty:
                ref = ax.axhline(
                    float(afm["auc_mean"].iloc[0]), ls=":", lw=1.6,
                    color="0.25", label="AlphaMissense reference",
                )
                if "AlphaMissense reference" not in labels:
                    handles.append(ref); labels.append("AlphaMissense reference")
            elif not afm.empty:
                ref, = ax.plot(
                    afm["size"], afm["auc_mean"], marker="s", ls=":",
                    lw=1.6, color="0.25", label="AlphaMissense reference",
                )
                if "AlphaMissense reference" not in labels:
                    handles.append(ref); labels.append("AlphaMissense reference")
            for normalization in normalizations:
                values = logo[
                    (logo["model"] == model)
                    & (logo["representation"] == representation)
                    & (logo["normalization"] == normalization)
                ].sort_values("size")
                if values.empty:
                    continue
                if fixed_size:
                    x = normalizations.index(normalization)
                    mean = float(values["auc_mean"].iloc[0])
                    std = float(values["auc_std"].iloc[0])
                    line = ax.errorbar(
                        x, mean, yerr=std, marker="o", markersize=6,
                        capsize=3, color=colors.get(normalization),
                        label=display.get(normalization, normalization),
                    )
                else:
                    line, = ax.plot(
                        values["size"], values["auc_mean"], marker="o", lw=2.0,
                        color=colors.get(normalization),
                        label=display.get(normalization, normalization),
                    )
                    ax.fill_between(
                        values["size"], values["auc_mean"] - values["auc_std"],
                        values["auc_mean"] + values["auc_std"],
                        color=colors.get(normalization), alpha=0.10,
                    )
                label = display.get(normalization, normalization)
                if label not in labels:
                    handles.append(line); labels.append(label)
            ax.set_title(representation)
            if col == 0:
                ax.set_ylabel(f"{model}\nMacro-LOGO ROC-AUC")
            if row == len(models) - 1:
                if fixed_size:
                    ax.set_xlabel("Score normalization")
                    ax.set_xticks(
                        range(len(normalizations)),
                        [display.get(value, value).split(" ")[0]
                         for value in normalizations],
                    )
                else:
                    ax.set_xlabel("D1 ensemble size")
            ax.grid(alpha=0.25)

    size_note = f" (fixed N={int(sizes[0])})" if fixed_size else ""
    fig.suptitle(
        f"AFM+D1 normalization ablation{size_note}\n"
        "Robust/rank use each held-out gene's complete unlabelled score batch "
        "(transductive diagnostics)",
        y=0.995, fontsize=13,
    )
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_structure_sweep(base_cfg, sizes, seeds, do_stratified, out, structure_variants=None,
                        wandb_enabled=False, wandb_project="afm_d1_structure_progress",
                        wandb_entity=None, force=False, seed_plan=None):
    """Structure-only sweep: models built from D1 + its pLDDT-derived metrics,
    NO AlphaMissense. Crosses ensemble-size x structure-variant over seeds, under
    LOGO. AFM is recorded only as a flat local reference (not an input)."""
    structure_variants = list(structure_variants or list(STRUCTURE_VARIANTS))
    rows = []
    for seed in seeds:
        wb = _wandb_start(wandb_project, wandb_entity, "structure_only_sweep",
                          f"seed{seed}", {"seed": seed, "sizes": sizes,
                                          "structure_variants": structure_variants}) if wandb_enabled else None
        progress = (
            _WandbProgress(wb, len(sizes) * len(structure_variants))
            if wb is not None else None
        )
        for size in sizes:
            if seed not in _seeds_for_size(seed_plan, size, seeds):
                continue
            cfg = dict(base_cfg); cfg["seed"] = seed; cfg["d1_max_structures_per_variant"] = size
            if progress is not None:
                progress.set_status(f"building dataset for N={size}")
            df = build_or_load_dataset(size, cfg, force=force)
            dfs = df.sort_values(["gene", "mutation"]).reset_index(drop=True)
            y = dfs["label"].astype(int).to_numpy()
            afm_auc = _auc(y, dfs["afm_score"].to_numpy())  # reference only
            rows.append({"size": size, "seed": seed, "cv": "logo", "model": "afm", "variant": "ref", "scope": "pooled", "auc": afm_auc})

            best = {}
            for variant in structure_variants:
                label = f"structure LOGO N={size} variant={variant}"
                plain_kwargs = {}
                if progress is not None:
                    plain_kwargs["progress_callback"] = progress.callback(label)
                r = logo_cv_plain(
                    df, cfg, _structure_cols(variant), **plain_kwargs,
                )
                if progress is not None:
                    progress.complete(label)
                mlp, logg = _auc(r["y"], r["oof_mlp"]), _auc(r["y"], r["oof_logreg"])
                best[variant] = logg
                for m, v in [("mlp", mlp), ("logreg", logg)]:
                    rows.append({"size": size, "seed": seed, "cv": "logo", "model": m, "variant": variant, "scope": "pooled", "auc": v})
                for _, g in r["per_gene"].iterrows():
                    for m in ["mlp", "logreg"]:
                        rows.append({"size": size, "seed": seed, "cv": "logo", "model": m, "variant": variant, "scope": g["gene"], "auc": g[m]})
            lr_str = " ".join(f"{v}={best[v]:.3f}" for v in structure_variants)
            print(f"  [struct] seed={seed} size={size:>2} afm_ref={afm_auc:.3f}  LOGO logreg[{lr_str}]")
        if wb is not None:
            progress.finish()
            wb.finish()

    runs_df = pd.DataFrame(rows)
    _policy = core.sampling_policy_from_config(base_cfg)
    runs_df["sampling_mode"] = _policy.mode
    runs_df["selection"] = _policy.selection
    runs_df.to_csv(out / "structure_sweep_runs.csv", index=False)
    summ = (runs_df.groupby(["cv", "model", "variant", "scope", "size"])["auc"]
                   .agg(auc_mean="mean", auc_std="std", n_seeds="count").reset_index())
    summ.to_csv(out / "structure_sweep_summary.csv", index=False)
    _plot_structure_ablation(summ, structure_variants, out / "structure_ablation.png")
    return runs_df, summ


def _plot_structure_ablation(summ, structure_variants, path):
    """Structure-only ablation: pooled LOGO AUC vs size, logreg (solid) and plain
    MLP (dashed) per structure variant, against the flat AFM reference line."""
    import matplotlib.pyplot as plt
    logo = summ[(summ["cv"] == "logo") & (summ["scope"] == "pooled")]
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    a = logo[logo["model"] == "afm"].sort_values("size")
    if not a.empty:
        ax.plot(a["size"], a["auc_mean"], ls=":", lw=1.8, color="0.3", marker="s",
                label="AlphaMissense (ref, not an input)")
    for i, variant in enumerate(structure_variants):
        color = cmap(i / max(len(structure_variants) - 1, 1))
        for model, ls, lw in [("logreg", "-", 2.0), ("mlp", "--", 1.3)]:
            s = logo[(logo["model"] == model) & (logo["variant"] == variant)].sort_values("size")
            if s.empty:
                continue
            ax.plot(s["size"], s["auc_mean"], marker="o", ls=ls, lw=lw, color=color,
                    label=f"{variant}:{model}")
            if model == "logreg":
                ax.fill_between(s["size"], s["auc_mean"] - s["auc_std"],
                                s["auc_mean"] + s["auc_std"], color=color, alpha=0.10)
    ax.plot([], [], color="gray", ls="-", lw=2.0, label="logreg (stable)")
    ax.plot([], [], color="gray", ls="--", lw=1.3, label="plain MLP")
    ax.set_xlabel("D1 ensemble size (structures per variant)")
    ax.set_ylabel("Pooled ROC-AUC (LOGO, mean±sd over seeds)")
    ax.set_title("Structure-only track -- D1 + pLDDT metrics (no AlphaMissense) vs ensemble size")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["logo", "stratified", "both"], default="both")
    ap.add_argument("--size", type=int, default=90, help="D1 ensemble size (structures/variant)")
    ap.add_argument(
        "--seeds", default="42-51",
        help=("legacy fusion model/CV + D1-subsampling seeds; for --track benchmark "
              "these seed D1 only and --benchmark-model-seed is separate, e.g. "
              "'42-51' or '42,43,44'"),
    )
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--fusion-models", default=None,
                    help="comma-separated LOGO model panel from "
                         "offset,logreg,gbm,mlp. Default offset,logreg,gbm: the "
                         "MLP is opt-in because it lost to plain logistic on "
                         "identical folds while costing ~99.96%% of the compute. "
                         "The panel is reported, never selected.")
    ap.add_argument("--save-oof", action="store_true",
                    help="persist per-variant out-of-fold scores as "
                         "oof_predictions.csv.gz. Needed for any paired "
                         "bootstrap: across-seed SD measures which structures "
                         "were drawn, not which variants are in the cohort.")
    ap.add_argument("--resume", action="store_true",
                    help="skip (seed, size) cells already present in "
                         "<out>/sweep_runs_partial.csv and carry their rows "
                         "forward. Cells in the checkpoint are complete by "
                         "construction, so this is safe after an interrupt.")
    ap.add_argument("--fusion-partners", default=None,
                    help="comma-separated partners to sweep, e.g. 'd1,ks'. Each becomes "
                         "a separate arm, so d1,ks x base,ks yields AM+D1, AM+D1+KS and "
                         "AM+KS (the ks,ks overlap is skipped as a duplicate column). "
                         "Defaults to --fusion-partner. Sweep mode only.")
    ap.add_argument("--fusion-partner", choices=list(FUSION_PARTNERS), default="d1",
                    help="descriptor paired with AlphaMissense in the fusion basis. "
                         "'d1' (default) reproduces prior results; 'ks' gives the "
                         "AM+KS arm that feature variants cannot express, since they "
                         "only append to the basis and cannot replace D1 inside it.")
    ap.add_argument("--logo-folds", choices=("grouped", "separate"), default="grouped",
                    help="grouped excludes both shared-WT subunits from training; separate is a sensitivity")
    ap.add_argument("--force-recompute", action="store_true", help="ignore cached labeled_df")
    ap.add_argument("--gt-group", default="all",
                    help="GT semantic group(s) to run: comma-separated combination of "
                         "'all' and 'annotation' (or 'each' == both). The binary "
                         "'functional' group was removed on 2026-10-13. A lone 'all' keeps the historical "
                         "output paths; any other selection writes each group to its own "
                         "subdir. Multiple groups also emit a combined comparison "
                         "(single-run mode).")
    # ensemble-size sweep
    ap.add_argument("--sweep", action="store_true",
                    help="run the ensemble-size sweep over --sizes instead of a single --size")
    ap.add_argument("--sizes", default="1,5,10,20,40,60,90",
                    help="comma-separated D1 ensemble sizes for the sweep")
    ap.add_argument("--sweep-stratified", action="store_true",
                    help="also compute stratified AUCs in the sweep (default: LOGO only)")
    # ensemble-shift feature variants (appended after the AFM+D1 basis)
    ap.add_argument("--feature-variant", default="base", choices=list(FEATURE_VARIANTS),
                    help="feature set for a single (non-sweep) LOGO/stratified run")
    # "all" contained profileKS and was removed on 2026-10-13.
    ap.add_argument("--feature-variants", default="base,spread,wass",
                    help="comma-separated feature variants to cross with size in --sweep")
    ap.add_argument(
        "--fusion-representation", default="raw2",
        choices=list(FUSION_REPRESENTATIONS),
        help=("Fusion basis for a single run (default raw2: AM and partner, no engineered interactions); MLP and logreg receive the same "
              "train-fold-standardized matrix"),
    )
    ap.add_argument(
        "--fusion-representations", default="raw2",
        help=("comma-separated bases to cross in --sweep --track fusion: "
              "raw2,compact4_v2,engineered6; compact4 retains the legacy "
              "uncentred polynomial basis"),
    )
    ap.add_argument(
        "--fusion-normalization", default="original",
        choices=list(FUSION_NORMALIZATIONS),
        help=("AFM/D1 score normalization for a single run. original is inductive; "
              "robust/rank are label-free within-gene transductive diagnostics"),
    )
    ap.add_argument(
        "--fusion-normalizations", default="original",
        help=("comma-separated score normalizations to cross in --sweep --track "
              "fusion: original,robust,rank"),
    )
    ap.add_argument("--mlp-epochs", type=int, default=SWEEP_BASE_CONFIG["epochs"])
    ap.add_argument("--mlp-hidden", type=int, default=SWEEP_BASE_CONFIG["hidden"])
    ap.add_argument("--mlp-dropout", type=float, default=SWEEP_BASE_CONFIG["dropout"])
    ap.add_argument("--mlp-lr", type=float, default=SWEEP_BASE_CONFIG["lr"])
    ap.add_argument(
        "--mlp-weight-decay", type=float,
        default=SWEEP_BASE_CONFIG["weight_decay"],
    )
    ap.add_argument("--mlp-patience", type=int, default=SWEEP_BASE_CONFIG["patience"])
    ap.add_argument("--mlp-batch-size", type=int, default=SWEEP_BASE_CONFIG["batch_size"])
    ap.add_argument(
        "--sampling-mode", choices=list(SAMPLING_MODES), default="nested",
        help="nested (default): the size-N sample is a prefix of larger samples, "
             "so sizes are paired by common random numbers. independent: a fresh "
             "draw per size -- a supplementary control, not a replacement.",
    )
    ap.add_argument(
        "--selection", choices=list(SELECTION_RULES), default="random",
        help="random (default): uniform draw, so N=1 is a random-single-structure "
             "baseline. top_confidence: keep the highest mean-pLDDT structures, "
             "ranked on wild type and applied to the mutant.",
    )
    ap.add_argument(
        "--seed-allocation", choices=["uniform", "proportional"], default="uniform",
        help="uniform (default): every size runs every seed. proportional: seeds "
             "per size scale with sampling variance (1/N)(M-N)/(M-1), so N=1 gets "
             "many repeats and near-exhaustive sizes get few, at the same total cost.",
    )
    ap.add_argument("--seed-budget", type=int, default=None,
                    help="total (size, seed) D1 builds for --seed-allocation proportional "
                         "(default: len(seeds) x len(sizes), i.e. budget-neutral)")
    ap.add_argument("--seed-floor", type=int, default=DEFAULT_SEED_FLOOR,
                    help="minimum seeds per size; below ~5 the reported SD is itself "
                         "too noisy to quote")
    ap.add_argument("--pool-size", type=int, default=125,
                    help="structures generated per variant, used for the "
                         "finite-population correction (125 for most genes; "
                         "TP53 is 100)")
    ap.add_argument("--convergence-epsilon", type=float, default=DEFAULT_EPSILON,
                    help="AUC increment below which added structures stop counting")
    ap.add_argument("--convergence-sd-threshold", type=float, default=DEFAULT_SD_THRESHOLD,
                    help="max seed-to-seed SD still considered stable (blank to skip)")
    ap.add_argument("--convergence-point-estimate", action="store_true",
                    help="judge convergence on the mean increment instead of its "
                         "upper confidence bound (less strict)")
    ap.add_argument("--convergence-runs", default=None,
                    help="sweep_runs.csv to analyse for --track convergence "
                         "(default: <out>/sweep_runs.csv)")
    ap.add_argument("--track", choices=["fusion", "structure", "benchmark", "convergence"], default="fusion",
                    help="'fusion' = afm+d1 (+ensemble-shift); 'structure' = D1 + its "
                         "pLDDT-derived metrics only; 'benchmark' = clean linear AFM+D1 "
                         "grid with optional nested gene-level selection (--sweep only)")
    ap.add_argument("--structure-variants", default="d1,d1+spread,d1+wass",
                    help="comma-separated structure-only variants to cross with size in --sweep --track structure")
    # Clean linear benchmark.  The parameter grid is always run locally and written
    # to CSV first; --wandb only mirrors it after successful local computation.
    ap.add_argument("--benchmark-representations", default="raw2,compact4_v2,engineered6,offset",
                    help=("comma-separated benchmark families: raw2, compact4_v2, "
                          "engineered6, offset; compact4 is the legacy uncentred basis"))
    ap.add_argument("--benchmark-weightings", default="variant,gene",
                    help="variant=current global class balance; gene=equal gene and within-gene class mass")
    ap.add_argument("--benchmark-normalizations", default="original,robust,rank",
                    help="original or label-free transductive within-gene robust/rank")
    ap.add_argument("--benchmark-c-values", default="0.1,1,10",
                    help="comma-separated inverse L2 strengths for all benchmark families")
    ap.add_argument("--benchmark-model-seed", type=int, default=2026,
                    help="model seed, deliberately separate from D1 subsampling --seeds")
    ap.add_argument(
        "--benchmark-nested", action=argparse.BooleanOptionalAction, default=True,
        help=("conservative inner-LOGO selection inside every outer fold; only "
              "original-score candidates are eligible and AFM is the one-SE fallback"),
    )
    ap.add_argument(
        "--benchmark-nested-representations", default="raw2,engineered6,offset",
        help=("original-score representations eligible for nested selection; "
              "separate from the fixed grid so new ablations do not silently "
              "enlarge the selector"),
    )
    # W&B monitoring is optional and off by default; nothing imports or contacts
    # W&B without --wandb.
    ap.add_argument("--wandb", action="store_true",
                    help="log candidate training curves, LOGO metrics, and live progress to W&B (off by default)")
    ap.add_argument("--wandb-project", default="afm_d1_fusion_training")
    ap.add_argument("--wandb-entity", default=None)
    args = ap.parse_args(argv)

    # convergence is a post-hoc read of an existing sweep table, so it neither
    # needs nor accepts --sweep.
    if args.track not in ("fusion", "convergence") and not args.sweep:
        ap.error(f"--track {args.track} requires --sweep")
    if args.track == "convergence" and args.sweep:
        ap.error("--track convergence analyses an existing sweep; drop --sweep")
    if args.selection == "top_confidence" and args.sampling_mode == "independent":  # noqa: E501
        ap.error(
            "--selection top_confidence is deterministic given the pool, so "
            "--sampling-mode independent would change nothing; use nested"
        )
    if args.wandb and not args.sweep:
        ap.error("--wandb is available for sweep tracks only; add --sweep")

    try:
        seeds = _seed_list(args.seeds)
    except ValueError as exc:
        ap.error(str(exc))
    if args.size <= 0:
        ap.error("--size must be a positive integer")
    if args.mlp_epochs <= 0 or args.mlp_hidden <= 0 or args.mlp_batch_size <= 0:
        ap.error("MLP epochs, hidden width, and batch size must be positive")
    if not 0 <= args.mlp_dropout < 1:
        ap.error("--mlp-dropout must be in [0, 1)")
    if args.mlp_lr <= 0 or args.mlp_weight_decay < 0 or args.mlp_patience < 0:
        ap.error("MLP learning rate must be positive; weight decay and patience must be non-negative")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not args.resume and not (out / "sweep_runs_partial.csv").exists():
        (out / "fold_policy.json").write_text(json.dumps({
            "logo_folds": args.logo_folds,
            "grouped_datasets": ["MSH2", "MSH6"],
            "reporting": "separate per-dataset AUCs; unchanged macro weights",
        }, indent=2) + "\n")
    base_cfg = _resolve_run_config({
        "fusion_partner": args.fusion_partner,
        "logo_folds": args.logo_folds,
        "d1_sampling_mode": args.sampling_mode,
        "d1_selection": args.selection,
        "d1_max_structures_per_variant": args.size,
        "epochs": args.mlp_epochs,
        "hidden": args.mlp_hidden,
        "dropout": args.mlp_dropout,
        "lr": args.mlp_lr,
        "weight_decay": args.mlp_weight_decay,
        "patience": args.mlp_patience,
        "batch_size": args.mlp_batch_size,
    })

    def _build_seed_plan(sizes):
        """Seeds per size, plus a printed rationale. None means uniform."""
        if args.seed_allocation == "uniform":
            return None
        budget = args.seed_budget or (len(seeds) * len(set(sizes)))
        table, plan, shared = sampling_plan.describe_plan(
            sizes, seeds, args.pool_size,
            total_budget=budget, floor=args.seed_floor,
        )
        table.to_csv(out / "seed_allocation.csv", index=False)
        print(
            f"[seeds] variance-proportional allocation over pool M={args.pool_size}, "
            f"budget={budget} builds, floor={args.seed_floor}"
        )
        print(table[[
            "size", "frac_of_pool", "variance_factor_vs_n1",
            "seeds_allocated", "rel_se_of_mean",
        ]].round(4).to_string(index=False))
        print(f"[seeds] sizes are paired on the shared prefix: {shared}")
        print(f"[seeds] allocation table -> {out / 'seed_allocation.csv'}\n")
        return plan

    if args.track == "convergence":
        runs_path = Path(args.convergence_runs or (out / "sweep_runs.csv"))
        if not runs_path.exists():
            ap.error(
                f"{runs_path} not found; run a sweep first or pass --convergence-runs"
            )
        groups = _gt_group_list(args.gt_group)
        sd_threshold = args.convergence_sd_threshold
        result = convergence_mod.run_convergence_analysis(
            runs_path, out,
            gt_groups=groups,
            pool_size=args.pool_size,
            epsilon=args.convergence_epsilon,
            sd_threshold=None if sd_threshold is not None and sd_threshold <= 0 else sd_threshold,
            strict=not args.convergence_point_estimate,
        )
        print(f"[convergence] source: {runs_path}")
        print(f"[convergence] pool M={args.pool_size}, epsilon={args.convergence_epsilon}, "
              f"criterion={'upper CI bound' if not args.convergence_point_estimate else 'point estimate'}")
        print("\n[convergence] paired increments (seed-paired, 95% t-CI):")
        print(result["paired_deltas"].round(5).to_string(index=False))
        print("\n[convergence] per-size spread vs the spread the finite pool alone explains:")
        print(result["size_summary"].round(5).to_string(index=False))
        print("\n[convergence] verdict:")
        print(result["verdict"].to_string(index=False))
        print(f"\n[done] convergence tables and figures written to {out}")
        return

    if args.sweep and args.track == "benchmark":
        from .benchmark import make_specs, run_benchmark_sweep

        try:
            sizes = _positive_int_list(args.sizes, "--sizes")
        except ValueError as exc:
            ap.error(str(exc))
        representations = [x.strip() for x in args.benchmark_representations.split(",") if x.strip()]
        weightings = [x.strip() for x in args.benchmark_weightings.split(",") if x.strip()]
        normalizations = [x.strip() for x in args.benchmark_normalizations.split(",") if x.strip()]
        try:
            c_values = [float(x) for x in args.benchmark_c_values.split(",") if x.strip()]
            specs = make_specs(representations, weightings, normalizations, c_values)
        except ValueError as exc:
            ap.error(str(exc))
        nested_representations = list(dict.fromkeys(
            value.strip()
            for value in args.benchmark_nested_representations.split(",")
            if value.strip()
        ))
        if args.benchmark_nested and (
            not nested_representations
            or not set(nested_representations).issubset(representations)
        ):
            ap.error(
                "--benchmark-nested-representations must be a non-empty subset "
                "of --benchmark-representations when nested selection is enabled"
            )
        groups = _gt_group_list(args.gt_group)
        if len(groups) != 1:
            ap.error("--track benchmark accepts one --gt-group per output directory")
        gt_group = groups[0]
        print(
            f"[sweep:benchmark] sizes={sizes} d1_seeds={seeds} model_seed={args.benchmark_model_seed}\n"
            f"  representations={representations}\n"
            f"  weightings={weightings} (both retain every variant)\n"
            f"  normalizations={normalizations}\n"
            f"  C={c_values} candidates={len(specs)} nested={args.benchmark_nested}\n"
            f"  nested_policy=original {nested_representations}; AFM one-SE fallback\n"
            f"  gt_group={gt_group} wandb={'ON -> ' + args.wandb_project if args.wandb else 'off'}\n"
        )
        wandb_project = (
            args.wandb_project
            if args.wandb_project != ap.get_default("wandb_project")
            else "afm_d1_clean_benchmark_progress"
        )
        runs, summary, selections = run_benchmark_sweep(
            base_cfg=base_cfg,
            sizes=sizes,
            d1_seeds=seeds,
            seed_plan=_build_seed_plan(sizes),
            out=out,
            build_dataset=build_or_load_dataset,
            specs=specs,
            gt_subset=_gt_subset,
            gt_group=gt_group,
            model_seed=args.benchmark_model_seed,
            nested=args.benchmark_nested,
            nested_representations=nested_representations,
            wandb_enabled=args.wandb,
            wandb_project=wandb_project,
            wandb_entity=args.wandb_entity,
            force=args.force_recompute,
        )
        largest = max(sizes)
        headline = summary[
            (summary["size"] == largest)
            & (summary["scope_type"] == "macro")
            & (summary["evaluation"].isin(["fixed", "nested"]))
        ].sort_values(["evaluation", "delta_mean"], ascending=[True, False])
        print(f"\n[sweep:benchmark] N={largest} macro-LOGO headline (top rows):")
        print(headline[[
            "evaluation", "candidate_id", "auc_mean", "afm_auc_mean",
            "delta_mean", "delta_ci_low", "delta_ci_high", "n_d1_seeds",
        ]].head(12).round(4).to_string(index=False))
        print(f"\n[done] benchmark tables, manifest, and figures written to {out}")
        return

    if args.sweep and args.track == "structure":
        try:
            sizes = _positive_int_list(args.sizes, "--sizes")
        except ValueError as exc:
            ap.error(str(exc))
        svariants = [v.strip() for v in args.structure_variants.split(",") if v.strip()]
        for v in svariants:
            _structure_cols(v)  # validate early
        print(f"[sweep:structure] sizes={sizes}  seeds={seeds}  variants={svariants}  "
              f"wandb={'ON -> ' + args.wandb_project if args.wandb else 'off'}  device={core.DEVICE}\n"
              f"(structure-only: D1 + pLDDT metrics, NO AlphaMissense)\n")
        wandb_project = args.wandb_project if args.wandb_project != ap.get_default("wandb_project") else "afm_d1_structure_progress"
        runs_df, summ = run_structure_sweep(
            base_cfg, sizes, seeds, do_stratified=False, out=out,
            structure_variants=svariants, seed_plan=_build_seed_plan(sizes),
            wandb_enabled=args.wandb, wandb_project=wandb_project,
            wandb_entity=args.wandb_entity, force=args.force_recompute,
        )
        print("\n[sweep:structure] pooled LOGO logreg AUC vs size x variant (mean over seeds):")
        lr = summ[(summ.cv == "logo") & (summ.scope == "pooled") & (summ.model == "logreg")]
        print(lr.pivot(index="size", columns="variant", values="auc_mean")
                .reindex(columns=svariants).round(3).to_string())
        ref = summ[(summ.cv == "logo") & (summ.scope == "pooled") & (summ.model == "afm")]
        print(f"\n[sweep:structure] AFM reference (flat): {ref['auc_mean'].mean():.3f}")
        print(f"\n[done] structure sweep tables + figure written to {out}")
        return

    if args.sweep:
        try:
            sizes = _positive_int_list(args.sizes, "--sizes")
        except ValueError as exc:
            ap.error(str(exc))
        variants = [v.strip() for v in args.feature_variants.split(",") if v.strip()]
        for v in variants:
            _feature_cols(v)  # validate early
        representations = list(dict.fromkeys(
            value.strip() for value in args.fusion_representations.split(",")
            if value.strip()
        ))
        unknown_representations = sorted(
            set(representations).difference(FUSION_REPRESENTATIONS)
        )
        if not representations or unknown_representations:
            ap.error(
                "--fusion-representations must be a non-empty comma list from "
                f"{list(FUSION_REPRESENTATIONS)}; unknown={unknown_representations}"
            )
        normalizations = list(dict.fromkeys(
            value.strip() for value in args.fusion_normalizations.split(",")
            if value.strip()
        ))
        unknown_normalizations = sorted(
            set(normalizations).difference(FUSION_NORMALIZATIONS)
        )
        if not normalizations or unknown_normalizations:
            ap.error(
                "--fusion-normalizations must be a non-empty comma list from "
                f"{list(FUSION_NORMALIZATIONS)}; unknown={unknown_normalizations}"
            )
        gt_groups = _gt_group_list(args.gt_group)
        print(f"[sweep] sizes={sizes}  seeds={seeds}  "
              f"representations={representations}  normalizations={normalizations}  "
              f"extras={variants}  "
              f"gt_groups={gt_groups}  models={_model_list(args.fusion_models)}  "
              f"stratified={args.sweep_stratified}  "
              f"wandb={'ON -> ' + args.wandb_project if args.wandb else 'off'}  device={core.DEVICE}\n")
        if any(value != "original" for value in normalizations):
            print(
                "[sweep] NOTE: robust/rank use each evaluated gene's complete "
                "unlabelled AFM/D1 batch; treat them as transductive diagnostics.\n"
            )
        runs_df, summ = run_sweep(
            base_cfg, sizes, seeds, do_stratified=args.sweep_stratified, out=out,
            fusion_partners=_partner_list(args),
            fusion_models=_model_list(args.fusion_models),
            resume=args.resume,
            feature_variants=variants, fusion_representations=representations,
            fusion_normalizations=normalizations,
            gt_groups=gt_groups,
            wandb_enabled=args.wandb, wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity, force=args.force_recompute,
            seed_plan=_build_seed_plan(sizes),
            save_oof=args.save_oof,
        )
        for gname in gt_groups:
            gsumm = summ[summ["gt_group"] == gname]
            print(f"\n[sweep] gt_group={gname} -- macro-LOGO fusion AUC "
                  "vs size x representation x normalization x extras (mean over seeds):")
            fus = gsumm[(gsumm.cv == "logo") & (gsumm.scope == "pooled_macro") & (gsumm.model == "fusion")]
            columns = pd.MultiIndex.from_product(
                [representations, normalizations, variants]
            )
            print(fus.pivot(
                index="size",
                columns=["representation", "normalization", "variant"],
                values="auc_mean",
            )
                     .reindex(columns=columns).round(3).to_string())
            ref = gsumm[(gsumm.cv == "logo") & (gsumm.scope == "pooled_macro") & (gsumm.model == "afm")].sort_values("size")
            print(f"  AFM reference (flat) macro-LOGO AUC vs size: "
                  f"{ref[['size', 'auc_mean']].round(3).to_dict('records')}")
        print(f"\n[done] sweep tables + figures written to {out}")
        return

    labeled = build_or_load_dataset(args.size, base_cfg, force=args.force_recompute)
    counts = labeled.groupby("gene")["label"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "n_pos"})
    print("\n[dataset] per-gene label counts:\n", counts.to_string())
    print(f"[dataset] pooled: {len(labeled)} variants, {int(labeled['label'].sum())} drivers\n")
    print(f"[run] seeds={seeds}  size={args.size}  device={core.DEVICE}\n")

    feature_cols = _feature_cols(args.feature_variant)
    fusion_representation = args.fusion_representation
    fusion_normalization = args.fusion_normalization
    if feature_cols:
        print(f"[run] feature variant: {args.feature_variant} -> extra features {feature_cols}\n")
    print(
        f"[run] fusion representation: {fusion_representation}; "
        f"normalization: {fusion_normalization}\n"
    )
    if fusion_normalization != "original":
        print(
            "[run] NOTE: robust/rank use each evaluated gene's complete unlabelled "
            "AFM/D1 batch; this is a transductive diagnostic, not ordinary "
            "single-variant deployment.\n"
        )

    def _run_one_group(name, frame, gout):
        """Run the requested CV regimes on one frame; returns LOGO pooled summary
        (or None) for the cross-group comparison."""
        gout.mkdir(parents=True, exist_ok=True)
        gc = frame.groupby("gene")["label"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "n_pos"})
        print("\n" + "=" * 70)
        print(f"[gt-group={name}] genes={sorted(frame['gene'].unique())}  "
              f"{len(frame)} variants, {int(frame['label'].sum())} drivers")
        print(gc.to_string())
        logo_pooled = None
        if args.mode in ("logo", "both"):
            res = run_logo(
                frame, base_cfg, seeds, feature_cols=feature_cols,
                fusion_representation=fusion_representation,
                fusion_normalization=fusion_normalization,
            )
            logo_pooled, per_gene_summary = summarize_logo(res)
            logo_pooled.to_csv(gout / "logo_pooled.csv", index=False)
            per_gene_summary.to_csv(gout / "logo_per_gene.csv", index=False)
            res["per_gene"].to_csv(gout / "logo_per_gene_by_seed.csv", index=False)
            print("-" * 70)
            print("LOGO (leave-one-gene-out) -- MACRO AUC is the primary AFM comparison")
            macro_first = pd.concat([
                logo_pooled[logo_pooled["model"].str.endswith("_macro")],
                logo_pooled[~logo_pooled["model"].str.endswith("_macro")],
            ], ignore_index=True)
            print(macro_first.to_string(index=False))
            print(f"  AlphaMissense-only pooled AUC 95% CI: "
                  f"[{res['ci_afm'][0]:.3f}, {res['ci_afm'][1]:.3f}]")
            print("\nLOGO -- PER-GENE (fusion trained; AFM seed-independent)")
            print(per_gene_summary.to_string(index=False))
        if args.mode in ("stratified", "both"):
            strat_summary, strat_raw = run_stratified(
                frame, base_cfg, seeds,
                fusion_representation=fusion_representation,
                fusion_normalization=fusion_normalization,
            )
            strat_summary.to_csv(gout / "stratified_pooled.csv", index=False)
            strat_raw.to_csv(gout / "stratified_by_seed.csv", index=False)
            print("\nSTRATIFIED k-fold (optimistic; mixes genes) -- POOLED AUC across seeds")
            print(strat_summary.to_string(index=False))
        return logo_pooled

    group_names = _gt_group_list(args.gt_group)
    lone_all = group_names == ["all"]

    logo_by_group = {}
    for gname in group_names:
        # A lone 'all' keeps the historical output paths; otherwise each group
        # (including 'all') gets its own subdir.
        gout = out if (gname == "all" and lone_all) else out / gname
        logo_by_group[gname] = _run_one_group(gname, _gt_subset(labeled, gname), gout)

    if len(group_names) > 1 and any(v is not None for v in logo_by_group.values()):
        # Cross-group comparison uses mean held-out-gene AUC, not pooled OOF scores.
        comp = (pd.concat(
                    [v.assign(gt_group=g) for g, v in logo_by_group.items() if v is not None],
                    ignore_index=True)
                  .pivot(index="model", columns="gt_group", values="auc_mean")
                  .reindex(index=["afm_macro", "d1_macro", "logreg_macro", "fusion_macro"])
                  .reindex(columns=[g for g in group_names if logo_by_group[g] is not None]))
        comp.to_csv(out / "gt_group_comparison_logo_macro.csv")
        print("\n" + "=" * 70)
        print("GT-GROUP COMPARISON -- macro-LOGO AUC (mean held-out-gene AUC)")
        print(comp.round(3).to_string())
        print("=" * 70)

    print(f"\n[done] tables written to {out}")


if __name__ == "__main__":
    main()
