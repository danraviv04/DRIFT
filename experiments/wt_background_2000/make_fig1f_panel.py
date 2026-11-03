#!/usr/bin/env python
"""Fig. 1f (MSH6, N=100, seed-disjoint) redrawn with the 2,000-split WT-WT background.

Plot code: pten_wt_null_panel.draw_panel, imported unchanged (same size, colours, label
placement). Data: results/MSH6_n100_seed/ from run_wt_background_2000.py (variant scores
identical to the published panel; WT-WT = published 200 splits + 1,800 more).

Style check: the same function applied to the PUBLISHED values must reproduce the published
PNG pixel for pixel (asserted; rendered into results/fig1f_panel/style_check/).

    cd <repo>; PYTHONPATH="." python \
        experiments/wt_background_2000/make_fig1f_panel.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PUB = REPO / "experiments" / "pten_wt_null_panel"
sys.path.insert(0, str(PUB))
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pten_wt_null_panel as P  # noqa: E402

SRC = HERE / "results" / "MSH6_n100_seed"
OUT = HERE / "results" / "fig1f_panel"
PUB_DIR = PUB / "results_MSH6_v6ca_wt375_n100_seedsplit"


def main():
    chk = OUT / "style_check"
    chk.mkdir(parents=True, exist_ok=True)
    P.draw_panel(pd.read_csv(PUB_DIR / "values.csv"), json.loads((PUB_DIR / "summary.json").read_text()),
                 chk, "MSH6")
    a, b = mpimg.imread(PUB_DIR / "MSH6_wt_null_panel.png"), mpimg.imread(chk / "MSH6_wt_null_panel.png")
    assert a.shape == b.shape and np.array_equal(a, b), "draw_panel no longer reproduces the published panel"
    print("style check: published panel reproduced pixel for pixel")

    values = pd.read_csv(SRC / "values.csv")
    summary = json.loads((SRC / "summary.json").read_text())
    assert (values.group == "wt_wt").sum() == 2000
    P.draw_panel(values, summary, OUT, "MSH6")
    print(f"wrote {OUT / 'MSH6_wt_null_panel.pdf'} (+ png); p95 {summary['wt_wt_p95']:.4f}")
    draw_panel_lightdots(values, summary, OUT)
    print(f"wrote {OUT / 'MSH6_wt_null_panel_lightdots.pdf'} (+ png)")


def draw_panel_lightdots(values, summary, out, gene="MSH6"):
    """ALTERNATIVE rendering, not the published style: P.draw_panel with one change. With
    2,000 WT-WT dots at the published dot size (s=7, alpha .55, drawn above the box) the
    dot cloud hides the WT-WT box and median; here the WT-WT dots are s=1.5, alpha .2 and
    the WT-WT box is drawn above its dots (dark edge, no fill). Everything else is copied from P.draw_panel.

    2026-11-02: plus a bracket over passenger vs driver with the two-sided Mann-Whitney P
    the main text cites (Methods, "Hypothesis tests"). No bracket to WT-WT: its 2,000 dots
    are overlapping splits of one 375-structure pool, not independent observations, so a
    rank test against them has no valid n; the dashed 95th percentile is that comparison."""
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(3.72, 3.32))
    rng = np.random.default_rng(P.BASE_SEED + 1)
    data = [values.loc[values.group == g, "ks"].to_numpy() for g, _, _ in P.GROUPS]
    parts = ax.violinplot(data, positions=range(3), widths=.8, showextrema=False)
    for body, (_, _, colour) in zip(parts["bodies"], P.GROUPS):
        body.set_facecolor(colour); body.set_edgecolor(colour); body.set_alpha(.2)
    for x, (vals, (_, _, colour)) in enumerate(zip(data, P.GROUPS)):
        many = len(vals) > 500
        z = dict(zorder=4) if many else {}   # WT-WT box above its dots; variant boxes as published
        ax.boxplot([vals], positions=[x], widths=.22, showfliers=False, patch_artist=True,
                   medianprops=dict(color="black", lw=1.4, **({"zorder": 5} if many else {})),
                   boxprops=dict(facecolor="none" if many else "white",
                                 edgecolor="#4a4a4a" if many else colour, lw=1.2, **z),
                   whiskerprops=dict(color="#4a4a4a", lw=1, **z), capprops=dict(color="#4a4a4a", lw=1, **z))
        ax.scatter(x + rng.uniform(-.2, .2, len(vals)), vals,
                   s=1.5 if many else (14 if len(vals) < 30 else 7),
                   color=colour, alpha=.2 if many else .55, edgecolor="none", zorder=3)
    p95 = summary["wt_wt_p95"]
    ax.axhline(p95, ls=(0, (3, 2)), lw=.9, color=P.GREY, zorder=1)
    ax.text(1.5, p95 + .003, "WT–WT\n95th pct.", fontsize=7, color=P.GREY, va="bottom",
            ha="center", linespacing=1.1)
    p = mannwhitneyu(data[2], data[1], alternative="two-sided").pvalue
    assert np.isclose(p, summary["driver_vs_passenger_mwu_p"], rtol=1e-9, atol=0), p
    mant, exp = f"{p:.1e}".split("e")
    y = max(data[1].max(), data[2].max()) + .022
    ax.plot([1, 1, 2, 2], [y - .008, y, y, y - .008], color=P.INK, lw=.9, clip_on=False)
    ax.text(1.5, y + .004, rf"$p = {mant}\times10^{{{int(exp)}}}$", fontsize=8, color=P.INK,
            ha="center", va="bottom")
    ax.set_ylim(top=y + .045)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{lab}\n(n={summary['groups'][g]['n']})"
                        for (g, _, _), lab in zip(P.GROUPS, ("WT vs WT", "Passenger", "Driver"))])
    ax.set_ylabel(f"KS-score (N = {summary['n_per_side']})")
    ax.set_title(f"{gene}    driver vs passenger AUC {summary['driver_vs_passenger_auc']:.2f}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{gene}_wt_null_panel_lightdots.{ext}", dpi=300)


if __name__ == "__main__":
    main()
