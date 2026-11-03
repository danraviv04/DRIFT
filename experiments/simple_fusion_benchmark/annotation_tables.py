"""Per-gene metric tables for AM+resKS on the annotation cohort.

The size sweep stores only three threshold-free metrics per gene, and the full
battery in `results_v1/` was computed with seven-gene folds -- wrong here,
because every threshold policy reads the training genes, and the annotation
cohort has five. So the battery is recomputed on the annotation cohort with its
own LOGO folds rather than sliced out of an existing table.

Two tables, as asked: raw values, and the change from raw AlphaMissense on the
same gene. Values are the mean over the ten descriptor seeds at a fixed ensemble
size; the seed SD is written alongside in the CSV.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from experiments.simple_fusion_benchmark import fusion, metrics, size_sweep

ARM = "AM+resKS"
# The two normalisations asked for. `gz` exists only as an equal-weight mean, so
# `z_mean` is its like-for-like partner; `z_wmean0.9` is carried too because it
# is the pre-specified headline method and omitting it would hide the weight.
METHODS = ["z_mean", "z_wmean0.9", "gz_mean"]
REFERENCE = "AlphaMissense"

# Simple first, then the ones that need more explaining. Thresholded metrics are
# quoted under one policy; every other policy is in the CSV.
POLICY = "train_youden_value"
SIMPLE = [("roc_auc", "ROC-AUC"),
          ("average_precision", "AP"),
          ("precision_at_R", "prec@R"),
          (f"accuracy__{POLICY}", "accuracy"),
          (f"f1__{POLICY}", "F1"),
          (f"precision__{POLICY}", "precision"),
          (f"recall__{POLICY}", "recall")]
COMPLEX = [(f"balanced_accuracy__{POLICY}", "bal.acc"),
           (f"mcc__{POLICY}", "MCC"),
           (f"kappa__{POLICY}", "kappa"),
           (f"specificity__{POLICY}", "specificity"),
           ("ap_lift", "AP lift"),
           # top-10% lift is deliberately absent: its ceiling is
           # min(1, R/decile)/prevalence, which is 1.04 on a 96.5%-positive gene,
           # and every score here attains it. The observed change was exactly
           # 0.0000 on four of the five genes.
           (f"precision_at_{metrics.FIXED_BUDGET}", f"prec@{metrics.FIXED_BUDGET}"),
           ("rare_class_precision_at_R", "rare-class prec@R"),
           ("partial_auc_fpr10", "pAUC(.1)"),
           ("brier", "Brier"),
           ("ece", "ECE")]
COLUMNS = SIMPLE + COMPLEX
# Lower is better for these two, so a negative change is an improvement.
LOWER_IS_BETTER = {"Brier", "ECE"}


# (metric label, one-line definition, why it is here or what to watch for).
DEFINITIONS = [
    ("prevalence", "positives / n \u2014 the fraction of this gene's variants that "
     "really are drivers.",
     "The base rate you would get by guessing. It runs from 3.5% (POLE) to 96.5% "
     "(PTEN) here, and most of the surprises in these tables trace back to it."),
    ("ROC-AUC", "probability a random driver outranks a random passenger.",
     "Chance is 0.5 and it ignores prevalence, which is both its strength and why "
     "it barely moves on genes that are almost entirely one class."),
    ("AP", "area under the precision-recall curve (average precision).",
     "Chance equals the prevalence, not 0.5. Sensitive to the rare class, which is "
     "why it registers changes ROC-AUC averages away."),
    ("prec@R", "precision among the top R scores, where R is the true number of "
     "drivers in that gene.",
     "At that cut precision = recall = F1. Parameter-free, but it needs the answer, "
     "so treat it as an upper bound."),
    ("accuracy", "fraction of all calls that are correct.",
     "Read it against the constant predictors: on PTEN, calling everything a driver "
     "is already 96.5% accurate."),
    ("F1", "harmonic mean of precision and recall.",
     "Ignores true negatives entirely, so it is generous on a high-prevalence gene."),
    ("precision", "of the variants called driver, the fraction that are.", ""),
    ("recall", "of the true drivers, the fraction that were called.", ""),
    ("bal.acc", "(recall + specificity) / 2.",
     "Plain accuracy with each class weighted equally; 0.5 for a constant predictor."),
    ("MCC", "correlation between the predicted and true labels, "
     "(tp\u00b7tn \u2212 fp\u00b7fn) / \u221a((tp+fp)(tp+fn)(tn+fp)(tn+fn)).",
     "Chance-corrected and symmetric in the classes; 0 for a constant predictor. "
     "The safest single threshold metric under imbalance."),
    ("kappa", "(observed agreement \u2212 expected) / (1 \u2212 expected).",
     "Same idea as MCC but more sensitive to a lopsided predicted-positive rate; "
     "where the two diverge, the operating point is misbehaving."),
    ("specificity", "of the true passengers, the fraction correctly not called.",
     "On PTEN this is estimated from 9 variants."),
    ("AP lift", "AP / prevalence \u2014 how many times better than chance.",
     "Bounded above by 1/prevalence, so it is NOT comparable across genes: PTEN's "
     "1.036 is 99.97% of the most it could possibly score, POLE's ceiling is 28.3."),
    (f"prec@{metrics.FIXED_BUDGET}", f"precision among the top "
     f"{metrics.FIXED_BUDGET} scores.",
     "A fixed follow-up budget, so it means the same thing on a 38-variant gene and "
     "a 1,074-variant one, unlike a percentage-based cut."),
    ("rare-class prec@R", "precision for whichever class is the MINORITY in that "
     "gene, ranking towards it.",
     "Positives on POLE (38 of 1,074), negatives on PTEN (9 of 257). Top-k metrics "
     "on the positive class are pinned near their ceiling wherever positives are "
     "the majority; this keeps the task hard in every gene."),
    ("pAUC(.1)", "AUC restricted to false-positive rate \u2264 0.1, "
     "McClish-standardised so 0.5 is still chance.",
     "The strict end of the ranking \u2014 the part that matters when most of what "
     "you follow up will be a passenger. POLE's full AUC is 0.82 but its pAUC is "
     "0.57, so its top-of-list is far weaker than its overall ranking."),
    ("Brier", "mean squared error of the predicted probability.",
     "Calibration and discrimination together. Blank for the fusions: a mean of "
     "z-scores is not a probability."),
    ("ECE", "expected calibration error over 10 bins.",
     "Pure calibration. Blank for the fusions, for the same reason."),
]


def per_seed_metrics(size, seed, gt_group):
    """Full battery per gene for AM and the three fusions, one cache cell."""
    frame, _ = size_sweep.load_cell(size, seed, gt_group)
    folds = size_sweep.make_folds(frame)
    genes = frame.gene.to_numpy()
    y = frame.label.to_numpy(dtype=int)
    cells = metrics.label_cells(y, genes, folds["LOGO"], protocol="LOGO")
    columns = fusion.ARMS[ARM]
    values = frame[columns].to_numpy(dtype=float)

    scores = {REFERENCE: (frame.afm_score.to_numpy(dtype=float), True)}
    for name, norm_kind, combiner, am_weight in fusion.specifications():
        if name not in METHODS:
            continue
        weights = None
        if am_weight is not None:
            rest = (1.0 - am_weight) / (len(columns) - 1)
            weights = np.array([am_weight] + [rest] * (len(columns) - 1))
        block = fusion.normalise(values, genes, norm_kind, folds["LOGO"])
        scores[name] = (fusion.combine(block, combiner, weights), False)

    rows, replay = [], {}
    for model, (score, is_probability) in scores.items():
        predictions, valid, used = metrics.apply_policies(
            score, y, genes, cells, folds["LOGO"], is_probability=is_probability)
        for gene in sorted(set(genes)):
            idx = genes == gene
            record = metrics.gene_metrics(
                y[idx], score[idx], {p: v[idx] for p, v in predictions.items()},
                {p: v[idx] for p, v in valid.items()},
                [c for c in used if c["gene"] == gene],
                is_probability=is_probability)
            rows.append({"seed": seed, "model": model, "gene": gene, **record})
            # The operating point is fitted on the OTHER genes, so resampling
            # this gene's variants must not move it. Carry the cut through.
            cut = [c[POLICY] for c in used if c["gene"] == gene]
            replay[(model, gene)] = {
                "score": score[idx], "y": y[idx],
                "cut": float(np.mean(cut)) if cut else float("nan"),
                "is_probability": is_probability}
    return pd.DataFrame(rows), replay


def _metrics_at(y, score, cut, is_probability):
    """Every table column for one gene's rows at a fixed operating point."""
    out = metrics.threshold_free(y, score, is_probability=is_probability)
    if np.isfinite(cut):
        out.update({f"{k}__{POLICY}": v
                    for k, v in metrics.thresholded(y, score >= cut).items()})
    return out


def bootstrap(replays, seeds, columns, draws, rng):
    """Paired variant bootstrap of the change from AlphaMissense.

    Each draw resamples a gene's variants with replacement AND picks one of the
    descriptor seeds, so the interval carries both the variant-sampling noise
    that decides whether a result transfers to new variants, and the
    ensemble-resampling noise. AlphaMissense is scored on the same resampled
    variants, which is what makes it paired: the comparison never gets credit
    for an easy draw.

    The operating point is held at the value fitted on the training genes.
    Resampling the held-out gene must not be allowed to move its own threshold.
    """
    genes = sorted({gene for (_, gene) in replays[seeds[0]]})
    out = {}
    macro = {model: {c: np.zeros(draws) for c in columns} for model in METHODS}
    for gene in genes:
        per_model = {model: {c: np.zeros(draws) for c in columns} for model in METHODS}
        n = len(replays[seeds[0]][(REFERENCE, gene)]["y"])
        for draw in range(draws):
            seed = seeds[rng.integers(len(seeds))]
            index = rng.integers(0, n, n)
            reference = replays[seed][(REFERENCE, gene)]
            if len(set(reference["y"][index])) < 2:
                for model in METHODS:
                    for c in columns:
                        per_model[model][c][draw] = np.nan
                continue
            base = _metrics_at(reference["y"][index], reference["score"][index],
                               reference["cut"], reference["is_probability"])
            for model in METHODS:
                cell = replays[seed][(model, gene)]
                got = _metrics_at(cell["y"][index], cell["score"][index],
                                  cell["cut"], cell["is_probability"])
                for c in columns:
                    per_model[model][c][draw] = got.get(c, np.nan) - base.get(c, np.nan)
        for model in METHODS:
            for c in columns:
                out[(model, gene, c)] = per_model[model][c]
                macro[model][c] += per_model[model][c] / len(genes)
    for model in METHODS:
        for c in columns:
            out[(model, "MACRO", c)] = macro[model][c]
    return out


def summarise_bootstrap(draws_by_key):
    rows = []
    for (model, gene, column), values in draws_by_key.items():
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append({"model": model, "gene": gene, "metric": column,
                         "lo": np.nan, "hi": np.nan, "p_gt0": np.nan,
                         "conclusive": False})
            continue
        lo, hi = np.percentile(values, [2.5, 97.5])
        rows.append({"model": model, "gene": gene, "metric": column,
                     "lo": lo, "hi": hi,
                     "p_gt0": float((values > 0).mean()),
                     # "Conclusive" here means only that the 95% interval
                     # excludes zero. It is not a multiplicity-corrected test,
                     # and there are 16 metrics x 6 rows per model.
                     "conclusive": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def build(size, seeds, gt_group, draws=2000, rng=None):
    frames, replays = [], {}
    for seed in seeds:
        frame, replay = per_seed_metrics(size, seed, gt_group)
        frames.append(frame)
        replays[seed] = replay
    print(f"  computed {len(frames)} seeds", flush=True)
    every = pd.concat(frames, ignore_index=True)
    keys = ["model", "gene"]
    # `rare_class` is a label ("positive"/"negative"), not a measurement, so it
    # is excluded from every aggregation rather than crashing the mean.
    numeric = [c for c in every.columns
               if c not in keys + ["seed"]
               and pd.api.types.is_numeric_dtype(every[c])]
    mean = every.groupby(keys, sort=False)[numeric].mean()
    sd = every.groupby(keys, sort=False)[numeric].std()
    # A MACRO row per model: the equally weighted mean over genes, computed per
    # seed and then averaged, so it matches the sweep's macro exactly.
    macro = (every.groupby(["model", "seed"], sort=False)[numeric].mean()
             .groupby("model", sort=False).mean())
    macro.index = pd.MultiIndex.from_product([macro.index, ["MACRO"]], names=keys)
    mean = pd.concat([mean, macro]).sort_index()
    columns = [c for c, _ in COLUMNS]
    print(f"  bootstrapping {draws} draws", flush=True)
    intervals = summarise_bootstrap(
        bootstrap(replays, list(seeds), columns, draws,
                  rng or np.random.default_rng(0)))
    counts = (every[every.model == REFERENCE]
              .groupby("gene")[["n", "positives"]].first().astype(int))
    counts["negatives"] = counts["n"] - counts["positives"]
    return every, mean, sd, intervals, counts


def _order(table):
    models = [REFERENCE] + METHODS
    genes = [g for g in table.index.get_level_values("gene").unique() if g != "MACRO"]
    return table.reindex(pd.MultiIndex.from_product(
        [models, sorted(genes) + ["MACRO"]], names=["model", "gene"])).dropna(how="all")


def tables(mean):
    raw = _order(mean)[[c for c, _ in COLUMNS]].copy()
    raw.columns = [label for _, label in COLUMNS]
    reference = raw.loc[REFERENCE]
    delta = raw.drop(index=REFERENCE).copy()
    for model in delta.index.get_level_values("model").unique():
        block = delta.loc[model]
        delta.loc[model] = (block - reference.loc[block.index]).to_numpy()
    return raw, delta


def _cell(value, lo, hi, conclusive, *, lower_is_better, signed):
    """One cell as (text, css class). Emphasis encodes reliability, not size.

    A colour scale over magnitude would make the largest cells the most salient,
    and on this cohort the largest cells are threshold-transfer artifacts rather
    than ranking changes. So the visual channel is spent on whether the interval
    excludes zero instead.
    """
    if not np.isfinite(value):
        return "\u2014", "na"
    text = f"{value:+.4f}" if signed else f"{value:.4f}"
    if not signed or not np.isfinite(lo):
        return text, "plain"
    improved = (value < 0) if lower_is_better else (value > 0)
    if not conclusive:
        return text, "unsure"
    return text, ("up" if improved else "down")


def _interval(lo, hi):
    return "\u2014" if not np.isfinite(lo) else f"[{lo:+.4f}, {hi:+.4f}]"


def with_counts(table, counts):
    """Prepend the gene's variant counts, as the rendered tables show them.

    Nullable Int64 rather than plain int: the MACRO row has no counts of its own,
    and a NaN would otherwise turn the whole column into floats ("109.0").
    """
    out = table.copy()
    for column in ("negatives", "positives", "n"):
        values = [counts.loc[gene, column] if gene in counts.index else pd.NA
                  for _, gene in out.index]
        out.insert(0, column, pd.array(values, dtype="Int64"))
    return out


def export_csvs(raw, delta, intervals, counts, out):
    """CSV twins of exactly what the HTML and markdown tables render.

    Three shapes, because they get used differently: Table 1 and Table 2 wide
    (one row per model/gene, as displayed), a tidy long form (one row per
    model/gene/metric, which is what downstream analysis actually wants), and the
    column definitions.
    """
    labels = dict(COLUMNS)
    lookup = intervals.set_index(["model", "gene", "metric"])

    table1 = with_counts(raw, counts)
    table1.to_csv(out / "table1_raw.csv")

    # Wide Table 2: every displayed cell is value + interval + the emphasis flag.
    wide = with_counts(delta, counts)
    for column, label in COLUMNS:
        lo, hi, ok, p = [], [], [], []
        for model, gene in delta.index:
            key = (model, gene, column)
            if key in lookup.index:
                row = lookup.loc[key]
                lo.append(row["lo"]); hi.append(row["hi"])
                ok.append(bool(row["conclusive"])); p.append(row["p_gt0"])
            else:
                lo.append(np.nan); hi.append(np.nan); ok.append(False); p.append(np.nan)
        wide[f"{label} ci_lo"] = lo
        wide[f"{label} ci_hi"] = hi
        wide[f"{label} p_gt0"] = p
        wide[f"{label} conclusive"] = ok
    wide.to_csv(out / "table2_change_vs_am.csv")

    # Tidy: one row per (model, gene, metric), raw and delta side by side.
    rows = []
    for model, gene in delta.index:
        counted = counts.loc[gene] if gene in counts.index else None
        for column, label in COLUMNS:
            key = (model, gene, column)
            entry = lookup.loc[key] if key in lookup.index else None
            rows.append({
                "model": model, "gene": gene,
                "n": int(counted.n) if counted is not None else pd.NA,
                "positives": int(counted.positives) if counted is not None else pd.NA,
                "negatives": int(counted.negatives) if counted is not None else pd.NA,
                "metric": column, "label": label,
                "lower_is_better": label in LOWER_IS_BETTER,
                "alphamissense": raw.loc[(REFERENCE, gene), label],
                "value": raw.loc[(model, gene), label],
                "delta_vs_am": delta.loc[(model, gene), label],
                "ci_lo": entry["lo"] if entry is not None else np.nan,
                "ci_hi": entry["hi"] if entry is not None else np.nan,
                "p_gt0": entry["p_gt0"] if entry is not None else np.nan,
                "conclusive": bool(entry["conclusive"]) if entry is not None else False,
            })
    tidy = pd.DataFrame(rows)
    tidy.to_csv(out / "tables_long.csv", index=False)

    pd.DataFrame(DEFINITIONS, columns=["column", "definition", "watch_for"]).to_csv(
        out / "column_definitions.csv", index=False)
    return table1, wide, tidy


def markdown(raw, delta, intervals, counts, *, header):
    """Markdown keeps every number; emphasis marks the conclusive rows only."""
    lines = [header, "### Table 1 \u2014 raw values", "",
             "Simple metrics first, then the ones that need more explaining. "
             "`n` / `pos` / `neg` are the gene's variant counts; note how few "
             "negatives some genes have, which is what the intervals in Table 2 "
             "are reacting to.", ""]
    head = "| model | gene | n | pos | neg | " + " | ".join(raw.columns) + " |"
    lines += [head, "|" + "---|" * (len(raw.columns) + 5)]
    for (model, gene), row in raw.iterrows():
        counted = counts.loc[gene] if gene in counts.index else None
        c = (f"{int(counted.n)} | {int(counted.positives)} | {int(counted.negatives)}"
             if counted is not None else "— | — | —")
        cells = [_cell(v, np.nan, np.nan, False, lower_is_better=False, signed=False)[0]
                 for v in row]
        label = f"**{gene}**" if gene == "MACRO" else gene
        lines.append(f"| {model} | {label} | {c} | " + " | ".join(cells) + " |")

    lines += ["", "### Table 2 \u2014 change from raw AlphaMissense", "",
              "Same gene, same folds, same seeds. Each cell carries a 95% "
              "confidence interval from a **paired bootstrap over variants** "
              "(2,000 draws; each draw resamples the gene's variants and picks one "
              "descriptor seed, scoring AlphaMissense on the same resample). "
              "**Bold** marks an interval that excludes zero; everything else is "
              "*not distinguishable from no change*, however large it looks. "
              "For Brier and ECE lower is better. The operating point is held at "
              "the value fitted on the training genes, so resampling a gene cannot "
              "move its own threshold.", ""]
    lines += [head, "|" + "---|" * (len(delta.columns) + 5)]
    lookup = intervals.set_index(["model", "gene", "metric"])
    for (model, gene), row in delta.iterrows():
        counted = counts.loc[gene] if gene in counts.index else None
        c = (f"{int(counted.n)} | {int(counted.positives)} | {int(counted.negatives)}"
             if counted is not None else "— | — | —")
        cells = []
        for (column, label), value in zip(COLUMNS, row):
            key = (model, gene, column)
            lo, hi, ok = (lookup.loc[key, ["lo", "hi", "conclusive"]]
                          if key in lookup.index else (np.nan, np.nan, False))
            text, css = _cell(value, lo, hi, bool(ok),
                              lower_is_better=label in LOWER_IS_BETTER, signed=True)
            cells.append(f"**{text}**" if css in ("up", "down") else text)
        gene_label = f"**{gene}**" if gene == "MACRO" else gene
        lines.append(f"| {model} | {gene_label} | {c} | " + " | ".join(cells) + " |")
    lines += ["", "### What each column means", "",
              "| column | definition | what to watch for |", "|---|---|---|"]
    for label, definition, caution in DEFINITIONS:
        lines.append(f"| **{label}** | {definition} | {caution} |")
    lines += ["",
              "Thresholded columns (accuracy, F1, precision, recall, bal.acc, MCC, "
              "kappa, specificity) all depend on the operating point, which here is "
              f"`{POLICY}`. The same score can move accuracy by 0.5 across policies, "
              "so the policy is part of the number.",
              "",
              "`top-10% lift` used to sit in these tables and has been removed. Its "
              "ceiling is min(1, R/decile)/prevalence \u2014 only 1.04 on a "
              "96.5%-positive gene \u2014 and every score here attains it, so the "
              "observed change was exactly 0.0000 on four of the five genes. It is "
              "not mathematically constant, but it cannot separate scores that are "
              f"all reasonable. `prec@{metrics.FIXED_BUDGET}` and `rare-class "
              "prec@R` replace it.",
              "", "### Table 2, with intervals", "",
              "The same changes with their bootstrap intervals written out, for "
              "the four headline metrics.", ""]
    headline = ["roc_auc", "average_precision", f"mcc__{POLICY}", f"f1__{POLICY}"]
    labels = dict(COLUMNS)
    lines += ["| model | gene | " + " | ".join(
        f"{labels[c]} (95% CI)" for c in headline) + " |",
        "|" + "---|" * (len(headline) + 2)]
    for (model, gene), row in delta.iterrows():
        cells = []
        for column in headline:
            key = (model, gene, column)
            value = row[labels[column]]
            if key in lookup.index:
                lo, hi, ok = lookup.loc[key, ["lo", "hi", "conclusive"]]
            else:
                lo = hi = np.nan; ok = False
            mark = "**" if ok else ""
            cells.append(f"{mark}{value:+.4f}{mark} {_interval(lo, hi)}")
        gene_label = f"**{gene}**" if gene == "MACRO" else gene
        lines.append(f"| {model} | {gene_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


HTML_STYLE = """<style>
:root{color-scheme:light}
body{font:14px/1.5 system-ui;margin:28px;color:#18222e;background:#fff;max-width:1600px}
h1{font-size:1.5rem;margin:0 0 .3rem}
h2{font-size:1.1rem;margin:2rem 0 .4rem;border-bottom:1px solid #e3e3e3;padding-bottom:.3rem}
p.sub{color:#5b6472;margin:.2rem 0 1rem;max-width:90ch}
.note{background:#fff5d7;padding:12px 16px;border-radius:5px;max-width:110ch}
.key{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 16px;font-size:12.5px;align-items:center}
.key span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:26px;height:14px;border-radius:3px;display:inline-block}
table{border-collapse:collapse;font-size:12px;margin-bottom:8px}
th,td{padding:5px 8px;border:1px solid #e0e4e9;text-align:right;font-variant-numeric:tabular-nums}
th.t,td.t{text-align:left}
th{background:#eef3f8;position:sticky;top:0;z-index:2}
tr.macro td,tr.macro th{border-top:2px solid #9aa6b5;font-weight:700;background:#f7f9fc}
td.up{color:#0f6c33;font-weight:700}
td.down{color:#a8200d;font-weight:700}
td.unsure{color:#98a1ad}
td.na{color:#c3c9d1}
td.plain{color:#18222e}
.ci{display:block;font-size:10px;color:#98a1ad;font-weight:400;font-variant-numeric:tabular-nums}
td.up .ci,td.down .ci{color:#6b7684}
.count{background:#f7f9fc;color:#5b6472}
table.defs{margin-top:8px;max-width:1150px}
table.defs td,table.defs th{white-space:normal;vertical-align:top;font-size:12.5px}
table.defs td.t:first-child{white-space:nowrap}
td.muted{color:#5b6472}
.wrap{overflow:auto;max-height:80vh;border:1px solid #e0e4e9;border-radius:6px}
</style>"""


def html_tables(raw, delta, intervals, counts, *, title, subtitle, policy_note):
    lookup = intervals.set_index(["model", "gene", "metric"])
    labels = [label for _, label in COLUMNS]

    def header_row():
        return ("<tr><th class='t'>model</th><th class='t'>gene</th>"
                "<th>n</th><th>pos</th><th>neg</th>"
                + "".join(f"<th>{html.escape(l)}</th>" for l in labels) + "</tr>")

    def counts_cells(gene):
        if gene not in counts.index:
            return "<td class='count'>&mdash;</td>" * 3
        row = counts.loc[gene]
        return "".join(f"<td class='count'>{int(row[k])}</td>"
                       for k in ("n", "positives", "negatives"))

    def body(table, *, signed):
        out = []
        for (model, gene), row in table.iterrows():
            cls = " class='macro'" if gene == "MACRO" else ""
            cells = []
            for (column, label), value in zip(COLUMNS, row):
                key = (model, gene, column)
                if signed and key in lookup.index:
                    lo, hi, ok = lookup.loc[key, ["lo", "hi", "conclusive"]]
                else:
                    lo = hi = np.nan; ok = False
                text, css = _cell(value, lo, hi, bool(ok),
                                  lower_is_better=label in LOWER_IS_BETTER, signed=signed)
                ci = (f"<span class='ci'>{html.escape(_interval(lo, hi))}</span>"
                      if signed and np.isfinite(lo) else "")
                cells.append(f"<td class='{css}'>{text}{ci}</td>")
            out.append(f"<tr{cls}><td class='t'>{html.escape(model)}</td>"
                       f"<td class='t'>{html.escape(gene)}</td>"
                       + counts_cells(gene) + "".join(cells) + "</tr>")
        return "".join(out)

    key = ("<div class='key'>"
           "<span><span class='swatch' style='background:#0f6c33'></span>"
           "better, 95% interval excludes zero</span>"
           "<span><span class='swatch' style='background:#a8200d'></span>"
           "worse, interval excludes zero</span>"
           "<span><span class='swatch' style='background:#c9ced6'></span>"
           "interval crosses zero &mdash; not distinguishable from no change</span>"
           "</div>")
    return (f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>"
            + HTML_STYLE
            + f"<h1>{html.escape(title)}</h1>"
            + f"<p class='sub'>{subtitle}</p>"
            + f"<p class='note'>{policy_note}</p>"
            + "<h2>Table 1 &mdash; raw values</h2>"
            "<p class='sub'>Simple metrics first, then the ones that need more "
            "explaining. No colour here on purpose: the columns span incomparable "
            "ranges (AUC 0.5&ndash;1, AP lift 1&ndash;8), so one scale would be "
            "meaningless and a per-column scale would imply rows are comparable when "
            "they differ 28&times; in variant count and 27&times; in prevalence.</p>"
            f"<div class='wrap'><table>{header_row()}{body(raw, signed=False)}"
            "</table></div>"
            "<h2>Table 2 &mdash; change from raw AlphaMissense</h2>"
            "<p class='sub'>95% confidence intervals from a paired bootstrap over "
            "variants: 2,000 draws, each resampling the gene's variants and picking "
            "one descriptor seed, with AlphaMissense scored on the same resample. "
            "Colour encodes <strong>reliability and direction, never magnitude</strong> "
            "&mdash; the largest cells on this cohort are threshold-transfer artifacts, "
            "and a magnitude scale would make exactly those the most salient.</p>"
            + key
            + f"<div class='wrap'><table>{header_row()}{body(delta, signed=True)}"
            "</table></div>"
            + "<h2>What each column means</h2>"
            "<p class='sub'>Thresholded columns all depend on the operating point, "
            f"which here is <code>{POLICY}</code>. The same score can move accuracy "
            "by 0.5 across policies, so the policy is part of the number.</p>"
            "<table class='defs'><tr><th class='t'>column</th>"
            "<th class='t'>definition</th><th class='t'>what to watch for</th></tr>"
            + "".join(
                f"<tr><td class='t'><strong>{html.escape(label)}</strong></td>"
                f"<td class='t'>{html.escape(definition)}</td>"
                f"<td class='t muted'>{html.escape(caution)}</td></tr>"
                for label, definition, caution in DEFINITIONS)
            + "</table>"
            "<p class='sub'><code>top-10% lift</code> used to sit in these tables "
            "and has been removed. Its ceiling is min(1,&nbsp;R/decile)/prevalence "
            "&mdash; only 1.04 on a 96.5%-positive gene &mdash; and every score here "
            "attains it, so the observed change was exactly 0.0000 on four of the "
            "five genes. It is not mathematically constant, but it cannot separate "
            f"scores that are all reasonable. <code>prec@{metrics.FIXED_BUDGET}</code> "
            "and <code>rare-class prec@R</code> replace it.</p>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-group", default="annotation", choices=["annotation"])
    parser.add_argument("--size", type=int, default=90)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(size_sweep.SEEDS))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    # Run-tagged since 2026-10-14; results_tables_annotation_N90 is the pre-v5 run.
    from metric_analysis.pipeline.paths import tagged
    out = args.out or HERE / tagged(f"results_tables_{args.gt_group}_N{args.size}")
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}; choose a fresh --out")
    out.mkdir(parents=True)

    print(f"{args.gt_group}, {ARM}, N={args.size}, {len(args.seeds)} seeds", flush=True)
    every, mean, sd, intervals, counts = build(
        args.size, args.seeds, args.gt_group, draws=args.draws)
    raw, delta = tables(mean)

    every.to_csv(out / "per_seed_per_gene.csv", index=False)
    mean.to_csv(out / "mean_over_seeds_all_metrics.csv")
    sd.to_csv(out / "sd_over_seeds_all_metrics.csv")
    intervals.to_csv(out / "bootstrap_intervals.csv", index=False)
    counts.to_csv(out / "gene_counts.csv")
    table1, wide, tidy = export_csvs(raw, delta, intervals, counts, out)
    print(f"  table1_raw.csv {table1.shape}  table2_change_vs_am.csv {wide.shape}  "
          f"tables_long.csv {tidy.shape}", flush=True)

    genes = ", ".join(g for g in raw.index.get_level_values("gene").unique()
                      if g != "MACRO")
    conclusive = intervals[intervals.conclusive]
    header = (f"# {ARM} on the {args.gt_group} cohort, N={args.size}\n\n"
              f"Genes: {genes}. LOGO cross-validation. Values are the mean over "
              f"{len(args.seeds)} descriptor seeds.\n\n"
              f"Thresholded metrics (accuracy, F1, precision, recall, bal.acc, MCC, "
              f"kappa, specificity) are quoted at the `{POLICY}` operating point: a "
              f"score threshold chosen to maximise macro-mean Youden J on the training "
              f"genes, never on the held-out one. Every other policy is in "
              f"`mean_over_seeds_all_metrics.csv`.\n\n"
              f"`MACRO` is the equally weighted mean over genes, computed per seed then "
              f"averaged. AP on the MACRO row is what is usually called MAP.\n\n"
              f"Of {len(intervals)} changes measured, **{len(conclusive)} have a 95% "
              f"bootstrap interval that excludes zero**. Read the rest as "
              f"inconclusive, not as small effects.\n\n")
    (out / "TABLES.md").write_text(markdown(raw, delta, intervals, counts, header=header))
    subtitle = (f"{html.escape(ARM)} on the {html.escape(args.gt_group)} cohort "
                f"&mdash; {html.escape(genes)}. N={args.size}, "
                f"{len(args.seeds)} descriptor seeds, LOGO cross-validation.")
    policy_note = (
        f"Thresholded metrics are quoted at the <code>{POLICY}</code> operating "
        "point: a score threshold maximising macro-mean Youden J on the <em>training</em> "
        "genes. Of " + str(len(intervals)) + " changes measured, <strong>"
        + str(len(conclusive)) + "</strong> have an interval excluding zero.")
    (out / "tables.html").write_text(html_tables(
        raw, delta, intervals, counts,
        title=f"{ARM} — {args.gt_group} cohort, N={args.size}",
        subtitle=subtitle, policy_note=policy_note))
    (out / "metadata.json").write_text(json.dumps({
        "gt_group": args.gt_group, "arm": ARM, "methods": METHODS,
        "size": args.size, "seeds": args.seeds, "protocol": "LOGO",
        "threshold_policy": POLICY, "reference": REFERENCE,
        "bootstrap_draws": args.draws,
        "bootstrap": "paired over variants; each draw also picks one descriptor seed",
        "conclusive_changes": int(len(conclusive)), "measured_changes": int(len(intervals)),
    }, indent=2))
    print(f"{len(conclusive)}/{len(intervals)} changes have an interval excluding zero")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
