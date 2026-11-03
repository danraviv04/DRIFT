"""When does adding structures stop paying? A paired convergence criterion.

The ensemble-size sweep is usually summarised by eye ("gains diminished after
roughly 40 structures"). This module replaces that with a stated rule.

Two things make the analysis specific to this design:

1. Sizes are *paired* by subsampling seed. Under nested sampling the size-N
   sample is a prefix of the size-N' sample within a seed, so the per-seed
   difference AUC(N') - AUC(N) is a within-variant contrast with much of the
   sampling noise cancelled. The right summary is therefore the mean paired
   difference and its interval, not the difference of two independent means.
   Where the seed sets differ between sizes -- as they do under a
   variance-proportional allocation -- only the shared seeds are paired.

2. The seed-to-seed spread shrinks with N partly because draws from a finite pool
   overlap increasingly. That fact does not supply a closed-form variance for
   ROC-AUC of a nonlinear descriptor, so this module reports the observed spread
   without presenting a sample-mean finite-population formula as its prediction.

Convergence is declared on two conditions, following the usual asymptotic
argument: the increment must be small, and the estimate must be stable. The
strict form of the first condition uses the upper confidence bound on the
increment rather than its point estimate, so "converged" means "a gain larger
than epsilon has been ruled out", not "the observed gain happened to be small".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # scipy ships with sklearn; fall back to the normal quantile without it.
    from scipy import stats as _stats

    def _t_quantile(confidence: float, df: int) -> float:
        return float(_stats.t.ppf(0.5 + confidence / 2.0, df))
except Exception:  # pragma: no cover - only without scipy
    from statistics import NormalDist

    def _t_quantile(confidence: float, df: int) -> float:
        # Anti-conservative for the small df this analysis runs at; scipy is the
        # supported path and this only keeps the module importable without it.
        return float(NormalDist().inv_cdf(0.5 + confidence / 2.0))


DEFAULT_EPSILON = 0.005
DEFAULT_SD_THRESHOLD = 0.01


def load_sweep_runs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"size", "seed", "auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: sweep runs table is missing columns {sorted(missing)}")
    return df


def select_curve(
    runs: pd.DataFrame,
    *,
    model: str = "d1",
    scope: str = "pooled_macro",
    gt_group: str | None = "all",
    cv: str | None = "logo",
    extra_filters: dict | None = None,
) -> pd.DataFrame:
    """Reduce a sweep-runs table to one (size, seed) -> auc curve."""
    out = runs.copy()
    wanted = {"model": model, "scope": scope, "gt_group": gt_group, "cv": cv}
    wanted.update(extra_filters or {})
    for column, value in wanted.items():
        if value is None or column not in out.columns:
            continue
        out = out[out[column] == value]
    if out.empty:
        raise ValueError(
            f"no rows match model={model!r} scope={scope!r} gt_group={gt_group!r} cv={cv!r}"
        )
    dupes = out.duplicated(["size", "seed"]).sum()
    if dupes:
        raise ValueError(
            f"{dupes} duplicate (size, seed) rows after filtering; tighten the filters "
            "so one curve is selected"
        )
    return out[["size", "seed", "auc"]].sort_values(["size", "seed"]).reset_index(drop=True)


def size_summary(curve: pd.DataFrame, pool_size: int | None = None,
                 *, mean_like: bool | None = None) -> pd.DataFrame:
    """Per-size mean and observed seed spread.

    ``pool_size`` and ``mean_like`` remain accepted for callers of the previous
    API, but no theoretical spread is emitted: the endpoint here is ROC-AUC of
    a nonlinear descriptor, not the sample mean for which the old correction
    was derived.
    """
    grouped = curve.groupby("size")["auc"]
    out = grouped.agg(
        auc_mean="mean", auc_sd=lambda s: float(s.std(ddof=1)), n_seeds="count"
    ).reset_index()
    out["se_of_mean"] = out["auc_sd"] / np.sqrt(out["n_seeds"].clip(lower=1))
    return out


def paired_deltas(curve: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Mean paired increment between each pair of consecutive sizes.

    Pairing is by seed, restricted to the seeds both sizes ran. Sizes that share
    fewer than two seeds yield a delta with no interval rather than being
    dropped, so a thin allocation stays visible instead of silently vanishing.
    """
    wide = curve.pivot(index="seed", columns="size", values="auc")
    sizes = sorted(int(s) for s in wide.columns)
    rows = []
    for lo, hi in zip(sizes[:-1], sizes[1:]):
        pair = wide[[lo, hi]].dropna()
        diffs = (pair[hi] - pair[lo]).to_numpy(dtype=float)
        n = int(diffs.size)
        mean = float(diffs.mean()) if n else np.nan
        sd = float(diffs.std(ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        half = _t_quantile(confidence, n - 1) * se if n > 1 else np.nan
        rows.append(
            {
                "size_from": lo,
                "size_to": hi,
                "n_paired_seeds": n,
                "delta_mean": mean,
                "delta_sd": sd,
                "delta_se": se,
                "delta_ci_low": mean - half if n > 1 else np.nan,
                "delta_ci_high": mean + half if n > 1 else np.nan,
                "structures_added": hi - lo,
                "delta_per_structure": mean / (hi - lo) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def declare_convergence(
    deltas: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    epsilon: float = DEFAULT_EPSILON,
    sd_threshold: float | None = DEFAULT_SD_THRESHOLD,
    strict: bool = True,
) -> dict:
    """Smallest size past which no further increment matters.

    `strict` compares the upper confidence bound of each increment against
    epsilon (a gain above epsilon has been excluded); otherwise the point
    estimate is used. A size qualifies only if *every* later increment also
    qualifies, so a single flat step in an otherwise rising curve cannot
    masquerade as an asymptote.
    """
    if deltas.empty:
        return {"converged_size": None, "reason": "no size pairs available"}

    column = "delta_ci_high" if strict else "delta_mean"
    values = deltas[column].to_numpy(dtype=float)

    small = np.isfinite(values) & (values < float(epsilon))
    # Walk backwards: a step qualifies only if it and everything after it do.
    qualifies_onward = np.zeros(len(small), dtype=bool)
    running = True
    for i in range(len(small) - 1, -1, -1):
        running = running and bool(small[i])
        qualifies_onward[i] = running

    sd_by_size = dict(zip(summary["size"].astype(int), summary["auc_sd"].astype(float)))
    converged_size = None
    for idx, qualifies in enumerate(qualifies_onward):
        if not qualifies:
            continue
        size_from = int(deltas.iloc[idx]["size_from"])
        if sd_threshold is not None:
            sd = sd_by_size.get(size_from, np.nan)
            if not np.isfinite(sd) or sd > float(sd_threshold):
                continue  # mean has settled but a single draw is still unstable
        converged_size = size_from
        break

    largest = int(summary["size"].max())
    return {
        "converged_size": converged_size,
        "criterion": "upper CI bound" if strict else "point estimate",
        "epsilon": float(epsilon),
        "sd_threshold": None if sd_threshold is None else float(sd_threshold),
        "largest_size_tested": largest,
        "converged_within_tested_range": converged_size is not None,
        "auc_at_converged": (
            float(summary.loc[summary["size"] == converged_size, "auc_mean"].iloc[0])
            if converged_size is not None
            else np.nan
        ),
        "auc_at_largest": float(
            summary.loc[summary["size"] == largest, "auc_mean"].iloc[0]
        ),
        "reason": (
            "increment below epsilon from this size onward"
            if converged_size is not None
            else "no size satisfied both conditions within the tested range"
        ),
    }


def analyze(
    runs: pd.DataFrame,
    *,
    model: str = "d1",
    scope: str = "pooled_macro",
    gt_group: str | None = "all",
    cv: str | None = "logo",
    pool_size: int | None = None,
    epsilon: float = DEFAULT_EPSILON,
    sd_threshold: float | None = DEFAULT_SD_THRESHOLD,
    strict: bool = True,
    confidence: float = 0.95,
) -> dict:
    curve = select_curve(runs, model=model, scope=scope, gt_group=gt_group, cv=cv)
    summary = size_summary(curve, pool_size=pool_size)
    deltas = paired_deltas(curve, confidence=confidence)
    verdict = declare_convergence(
        deltas, summary, epsilon=epsilon, sd_threshold=sd_threshold, strict=strict
    )
    verdict.update({"model": model, "scope": scope, "gt_group": gt_group, "cv": cv,
                    "finite_pool_spread_model": "not estimated"})
    if pool_size:
        verdict["pool_size"] = int(pool_size)
    return {"curve": curve, "summary": summary, "deltas": deltas, "verdict": verdict}


def plot_convergence(result: dict, path: str | Path, *, title: str | None = None):
    """Two panels: the curve with its spread, and the paired increments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary, deltas = result["summary"], result["deltas"]
    verdict = result["verdict"]
    fig, (ax_curve, ax_delta) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)

    # Both panels share one axis, so both must speak the same units. The sizes
    # are plotted on an even grid rather than a linear N scale: the grid is
    # 1..90 but the interesting part is 1-20, which a linear axis crushes into
    # a fifth of the width. Each increment is then a bar centred BETWEEN the
    # two sizes it connects, so it sits under the segment of the curve it
    # describes and needs no label of its own.
    positions = np.arange(len(summary))
    at = dict(zip(summary["size"].astype(int), positions))

    ax_curve.errorbar(
        positions, summary["auc_mean"], yerr=summary["auc_sd"],
        marker="o", capsize=3, color="#1f4e79", label="mean +/- SD across seeds",
    )
    if verdict.get("converged_size") in at:
        ax_curve.axvline(
            at[verdict["converged_size"]], color="#b03060", ls="--",
            label=f"converged at N={verdict['converged_size']}",
        )
    ax_curve.set_ylabel("ROC-AUC")
    ax_curve.legend(fontsize=8, loc="lower right")
    ax_curve.set_title(title or "D1 ensemble-size convergence")

    x = np.array([
        0.5 * (at[int(a)] + at[int(b)])
        for a, b in zip(deltas["size_from"], deltas["size_to"])
    ])
    ax_delta.bar(x, deltas["delta_mean"], color="#4c78a8", width=0.5)
    finite = deltas["delta_ci_low"].notna()
    if finite.any():
        ax_delta.errorbar(
            x[finite.to_numpy()],
            deltas.loc[finite, "delta_mean"],
            yerr=[
                (deltas.loc[finite, "delta_mean"] - deltas.loc[finite, "delta_ci_low"]),
                (deltas.loc[finite, "delta_ci_high"] - deltas.loc[finite, "delta_mean"]),
            ],
            fmt="none", ecolor="black", capsize=3, lw=1,
        )
    ax_delta.axhline(verdict["epsilon"], color="#b03060", ls="--",
                     label=f"epsilon = {verdict['epsilon']}")
    ax_delta.axhline(0, color="black", lw=0.8)
    ax_delta.set_xticks(positions)
    ax_delta.set_xticklabels([str(int(s)) for s in summary["size"]])
    ax_delta.set_xlabel(
        "ensemble size (structures per variant); bars are paired increments"
    )
    ax_delta.set_ylabel("paired AUC increment")
    ax_delta.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def run_convergence_analysis(
    runs_path: str | Path,
    out_dir: str | Path,
    *,
    model: str = "d1",
    scope: str = "pooled_macro",
    gt_groups=("all",),
    cv: str = "logo",
    pool_size: int | None = None,
    epsilon: float = DEFAULT_EPSILON,
    sd_threshold: float | None = DEFAULT_SD_THRESHOLD,
    strict: bool = True,
    make_figures: bool = True,
) -> dict:
    """Analyse every requested GT group and write the tables and figures."""
    runs = load_sweep_runs(runs_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries, deltas, verdicts, figures = [], [], [], []
    for group in gt_groups:
        result = analyze(
            runs, model=model, scope=scope, gt_group=group, cv=cv,
            pool_size=pool_size, epsilon=epsilon, sd_threshold=sd_threshold,
            strict=strict,
        )
        for frame, sink in ((result["summary"], summaries), (result["deltas"], deltas)):
            frame = frame.copy()
            frame.insert(0, "gt_group", group)
            frame.insert(0, "model", model)
            frame.insert(0, "scope", scope)
            sink.append(frame)
        verdicts.append(result["verdict"])
        if make_figures:
            figures.append(
                plot_convergence(
                    result,
                    out / f"convergence_{model}_{scope}_{group}.png",
                    title=f"{model} / {scope} / {group}",
                )
            )

    summary_df = pd.concat(summaries, ignore_index=True)
    delta_df = pd.concat(deltas, ignore_index=True)
    verdict_df = pd.DataFrame(verdicts)

    summary_df.to_csv(out / "convergence_size_summary.csv", index=False)
    delta_df.to_csv(out / "convergence_paired_deltas.csv", index=False)
    verdict_df.to_csv(out / "convergence_verdict.csv", index=False)

    return {
        "size_summary": summary_df,
        "paired_deltas": delta_df,
        "verdict": verdict_df,
        "figures": figures,
        "out_dir": out,
    }
