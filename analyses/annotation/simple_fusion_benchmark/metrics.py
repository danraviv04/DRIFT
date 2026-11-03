"""Threshold-free and threshold-based metrics for ranked variant scores.

Four of the five threshold policies are *rate* policies: they decide what
fraction of a gene's variants to call driver, then take that fraction from the
top of that gene's own scores. A rate policy is invariant to each model's output
scale, so a random forest probability, an SVM decision value and a mean rank are
put on the same operating point with no cross-model calibration step, and a
shift in one gene's score distribution cannot hurt it.

`train_youden_value` is the deliberate opposite: it transfers a score *value*
from the training genes, keeping exactly the absolute-scale information the rate
policies discard. AlphaMissense's own published cutoff works that way, and on
these genes it beats the rate policies, so the pair is reported side by side
rather than one of them being chosen.

None of the policies looks at the held-out gene's labels, except `oracle_rate`:
  train_prevalence    the training driver rate (the same gene's under stratified CV,
                      the mean over the training genes under leave-one-gene-out)
  train_youden_rate   the call rate maximising macro-mean Youden J on the training genes
  train_youden_value  the score value maximising it
  native_half         score >= 0.5 (probability outputs only)
  oracle_rate         the held-out gene's true driver count: an upper bound, not an
                      operating point
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Threshold policies, in report order. The last one is an upper bound, not an
# achievable operating point: it is told how many drivers the held-out gene has.
POLICIES = ("train_prevalence", "train_youden_rate", "train_youden_value",
            "native_half", "oracle_rate")
YOUDEN_GRID = np.linspace(0.0, 1.0, 201)
# A fixed follow-up budget for precision@k, in variants. Chosen once, not tuned.
FIXED_BUDGET = 20
VALUE_GRID = np.linspace(0.0, 1.0, 401)


def within_gene_rank(score, genes):
    """Percentile rank in (0, 1) within each gene; average ranks for ties."""
    from scipy.stats import rankdata
    out = np.empty(len(score), dtype=float)
    for gene in np.unique(genes):
        idx = genes == gene
        out[idx] = rankdata(score[idx]) / (idx.sum() + 1.0)
    return out


def top_rate_mask(score, rate):
    """Call the top `rate` fraction positive, by threshold value (ties included).

    Taking the k-th largest score as a threshold and predicting `score >= t`
    means tied scores are never split arbitrarily; the realised positive rate
    can therefore exceed `rate`, which every table reports as `ppr`.
    """
    n = len(score)
    k = int(np.clip(round(rate * n), 0, n))
    if k == 0:
        return np.zeros(n, dtype=bool), np.inf
    threshold = float(np.partition(score, n - k)[n - k])
    return score >= threshold, threshold


def youden_rate(score, y, genes):
    """Rate maximising the macro-mean Youden J over the training genes.

    The rate, not a score value, is what transfers: a held-out gene's scores may
    sit on a different scale, but "call the top 12%" always means the same thing.
    Genes are weighted equally, so POLE's 1,075 rows do not set the operating
    point for KRAS's 45.

    J is evaluated on the sorted score order, which splits tied scores; the rate
    is then applied by `top_rate_mask`, which does not. The two agree except on
    ties exactly at the chosen cut.
    """
    present = [g for g in np.unique(genes) if 0 < y[genes == g].sum() < (genes == g).sum()]
    if not present:
        return float("nan")
    curves = []
    for gene in present:
        idx = genes == gene
        target = y[idx].astype(bool)
        ordered = target[np.argsort(-score[idx], kind="stable")]
        hits = np.concatenate([[0], np.cumsum(ordered)])
        k = np.clip(np.round(YOUDEN_GRID * idx.sum()).astype(int), 0, idx.sum())
        tpr = hits[k] / target.sum()
        fpr = (k - hits[k]) / (~target).sum()
        curves.append(tpr - fpr)
    return float(YOUDEN_GRID[int(np.argmax(np.mean(curves, axis=0)))])


def youden_value(score, y, genes):
    """Score *value* maximising the macro-mean Youden J over the training genes.

    The complement of `youden_rate`: this one keeps the score's absolute scale
    and throws away nothing, which is how AlphaMissense's own published cutoff
    works. It is the better policy exactly when a score means the same thing in
    every gene, and the worse one when it does not -- a rate policy cannot be
    beaten by a shift in a gene's score distribution, and this one can.

    Candidates are quantiles of the pooled training scores, so the grid adapts
    to whatever range the score happens to occupy.
    """
    present = [g for g in np.unique(genes) if 0 < y[genes == g].sum() < (genes == g).sum()]
    if not present:
        return float("nan")
    candidates = np.unique(np.quantile(score, VALUE_GRID))
    js = np.zeros(len(candidates))
    for gene in present:
        idx = genes == gene
        target = y[idx].astype(bool)
        values = score[idx]
        # Column j of `called` would be values >= candidates[j]; counted by
        # searchsorted instead so the full matrix is never built.
        ordered = np.sort(values)
        above = len(values) - np.searchsorted(ordered, candidates, side="left")
        hits_ordered = np.sort(values[target])
        hits = target.sum() - np.searchsorted(hits_ordered, candidates, side="left")
        js += hits / target.sum() - (above - hits) / (~target).sum()
    return float(candidates[int(np.argmax(js))])


def label_cells(y, genes, folds, *, protocol):
    """Per (fold, gene) rates that depend on labels and folds only, not on scores.

    Computed once per protocol and reused by every evaluation.

    `train_prevalence` prefers the same gene's training rows when the protocol
    provides them (stratified CV) and falls back to the equally weighted mean
    prevalence of the training genes when it does not (LOGO, where the held-out
    gene contributes no training rows at all). Which of the two was used is
    recorded per cell rather than inferred from the protocol name.
    """
    cells = []
    for fold, (tr, va) in enumerate(folds):
        train_genes = genes[tr]
        macro_prevalence = float(np.mean([y[tr][train_genes == g].mean()
                                          for g in np.unique(train_genes)]))
        for gene in np.unique(genes[va]):
            rows = va[genes[va] == gene]
            same_gene = tr[train_genes == gene]
            if len(same_gene) and 0 < y[same_gene].sum() < len(same_gene):
                prevalence, source = float(y[same_gene].mean()), "same_gene_train"
            else:
                prevalence, source = macro_prevalence, "macro_train_genes"
            cells.append({"fold": fold, "gene": str(gene), "rows": rows,
                          "train_prevalence": prevalence,
                          "oracle_rate": float(y[rows].mean()),
                          "prevalence_source": source, "protocol": protocol})
    return cells


def apply_policies(score, y, genes, cells, folds, *, is_probability):
    """Boolean prediction vectors over all rows, one per policy.

    Returns the per-cell rates actually used, so the report can show the
    operating point alongside the metric it produced.
    """
    per_fold = {fold: (youden_rate(score[tr], y[tr], genes[tr]),
                       youden_value(score[tr], y[tr], genes[tr]))
                for fold, (tr, _) in enumerate(folds)}
    predictions = {p: np.zeros(len(score), dtype=bool) for p in POLICIES}
    valid = {p: np.zeros(len(score), dtype=bool) for p in POLICIES}
    used = []
    for cell in cells:
        rows = cell["rows"]
        rate, value = per_fold[cell["fold"]]
        rates = {"train_prevalence": cell["train_prevalence"],
                 "oracle_rate": cell["oracle_rate"], "train_youden_rate": rate}
        for policy, r in rates.items():
            if not np.isfinite(r):
                continue
            predictions[policy][rows], _ = top_rate_mask(score[rows], r)
            valid[policy][rows] = True
        if np.isfinite(value):
            # The one policy that reads the score's absolute scale rather than
            # only its within-gene order.
            predictions["train_youden_value"][rows] = score[rows] >= value
            valid["train_youden_value"][rows] = True
        used.append({**{k: v for k, v in cell.items() if k != "rows"}, **rates,
                     "train_youden_value": value})
    if is_probability:
        predictions["native_half"] = score >= 0.5
        valid["native_half"][:] = True
    return predictions, valid, used


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = index == b
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def threshold_free(y, score, *, is_probability):
    """Metrics that read the ranking only (plus calibration where defined)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    n, positives = len(y), int(y.sum())
    prevalence = positives / n
    order = np.argsort(-score, kind="stable")
    top_r = y[order[:positives]].mean() if positives else float("nan")
    decile = max(1, int(round(0.1 * n)))
    # A fixed follow-up budget, which unlike a decile does not grow with the
    # gene and so means the same thing on a 38-variant gene and a 1,075-variant
    # one.
    budget = min(FIXED_BUDGET, n)
    at_budget = y[order[:budget]].mean()
    # Precision on whichever class is RARE in this gene. Top-k metrics on the
    # positive class are structurally pinned near their ceiling wherever
    # positives are the majority (PTEN is 96.5% driver), so they cannot move
    # whatever the score does. Ranking towards the minority class instead keeps
    # the task hard in every gene and makes the column comparable across them.
    minority_positive = positives <= n - positives
    rare = positives if minority_positive else n - positives
    if rare:
        ordered = order if minority_positive else np.argsort(score, kind="stable")
        target = y if minority_positive else 1 - y
        rare_precision = float(target[ordered[:rare]].mean())
    else:
        rare_precision = float("nan")
    out = {
        "n": n, "positives": positives, "prevalence": prevalence,
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "ap_lift": average_precision_score(y, score) / prevalence,
        "precision_at_R": float(top_r),
        f"precision_at_{FIXED_BUDGET}": float(at_budget),
        "rare_class_precision_at_R": rare_precision,
        "rare_class": "positive" if minority_positive else "negative",
        "top_decile_lift": float(y[order[:decile]].mean() / prevalence),
        "partial_auc_fpr10": roc_auc_score(y, score, max_fpr=0.1),
    }
    # Calibration is only meaningful for a score that claims to be a
    # probability AND actually lies in [0, 1]; a decision value does neither.
    calibrated = is_probability and score.min() >= -1e-9 and score.max() <= 1 + 1e-9
    clipped = np.clip(score, 0.0, 1.0)
    out["brier"] = float(np.mean((clipped - y) ** 2)) if calibrated else np.nan
    out["ece"] = expected_calibration_error(y, clipped) if calibrated else np.nan
    return out


def thresholded(y, pred):
    """Confusion-matrix metrics for one boolean prediction vector.

    Computed in closed form from the four counts rather than through sklearn:
    this runs once per gene per policy per evaluation, and the degenerate cases
    (a constant prediction, a single-class gene) need defined values instead of
    warnings. MCC and kappa are 0 for a constant prediction, which is the limit
    of their definitions, and recall/specificity are undefined when the
    corresponding class is absent.
    """
    y, pred = y.astype(bool), pred.astype(bool)
    tp = int((y & pred).sum()); fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum()); tn = int((~y & ~pred).sum())
    n = len(y)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    agreement = (tp + tn) / n
    chance = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return {
        "accuracy": agreement,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "precision": precision, "recall": recall, "specificity": specificity,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "kappa": (agreement - chance) / (1 - chance) if chance < 1 else 0.0,
        "ppr": pred.mean(), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def gene_metrics(y, score, pred_by_policy, valid_by_policy, rates_for_gene, *, is_probability):
    """One flat record: threshold-free metrics plus every policy's metrics.

    Every array argument is already restricted to one gene's rows, so a policy
    counts as available only when it produced a prediction for that gene.
    """
    row = threshold_free(y, score, is_probability=is_probability)
    for policy in POLICIES:
        if not valid_by_policy[policy].all():
            row.update({f"{k}__{policy}": np.nan for k in
                        ("accuracy", "balanced_accuracy", "f1", "precision", "recall",
                         "specificity", "mcc", "kappa", "ppr", "operating_point")})
            continue
        row.update({f"{k}__{policy}": v for k, v in thresholded(y, pred_by_policy[policy]).items()
                    if k not in ("tp", "fp", "fn", "tn")})
        # A rate for the rate policies, a score threshold for train_youden_value;
        # one column either way, so the report can always show the cut that was used.
        row[f"operating_point__{policy}"] = (
            0.5 if policy == "native_half"
            else float(np.mean([c[policy] for c in rates_for_gene])))
    return row


METRIC_COLUMNS = (["roc_auc", "average_precision", "ap_lift", "precision_at_R",
                   f"precision_at_{FIXED_BUDGET}", "rare_class_precision_at_R",
                   "top_decile_lift", "partial_auc_fpr10", "brier", "ece"]
                  + [f"{k}__{p}" for p in POLICIES for k in
                     ("accuracy", "balanced_accuracy", "f1", "precision", "recall",
                      "specificity", "mcc", "kappa", "ppr", "operating_point")])

HIGHER_IS_WORSE = {"brier", "ece"}
