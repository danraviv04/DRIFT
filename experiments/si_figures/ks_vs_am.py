#!/usr/bin/env python
"""SI figure: the KS-score against AlphaMissense, per gene.

WHAT IT ESTABLISHES
-------------------
How much the KS-score and AlphaMissense (AM) agree. The main text cites this
where it asks whether the ensemble signal is already encoded in the conservation
and other features AM uses ("Extended Fig. **" placeholder), before the fusion
result in Fig. 2g. The per-gene correlation is moderate (v6: rho 0.47-0.76,
equal-gene macro 0.64), and part of it is only both scores separating the
classes: within a class it falls for most genes (MSH6 0.73 -> 0.33/0.35, MSH2
0.76 -> 0.54/0.57). POLE is the exception, where KS and AM also agree within
the passengers.

Panel b is drawn in two versions, to keep the choice open (user, 2026-10-15):
  ks_vs_am_rho_forest  rho per gene for all variants, drivers and passengers (in the registry)
  ks_vs_am_pooled      the six genes pooled as within-gene percentiles (the 10-14 panel g)
Swap the stem in drift/si_panels.SI_FIGURE_LIST to use the other.

SOURCE
------
The fusion caches of the current protein config (fingerprint-pinned through
figlib.labeled), N = 90, KS averaged over
seeds 42-51; AM = afm_score in the same rows. Annotation cohort, 1,799 variants.
POST the strict cutover.

Cross-checks. experiments/ks_vs_am (2026-10-14) is the v5 build (MSH2 and PIK3CA
on their old wild types), so it is no longer reproduced exactly: the script
prints the per-gene change against it and checks the genes allowed to move for
the current config. Verified references cover v6_ca and the single-WT POLE
repair; unknown configs require an independently checked reference.

    python experiments/si_figures/ks_vs_am.py
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: F403,E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from drift.protein_config import protein_config_fingerprint  # noqa: E402
from metric_analysis.pipeline.library.position_bootstrap import clusters_of, unstratified_draw  # noqa: E402

N = 90
SEEDS = tuple(range(42, 52))
BOOT = 2000
BOOT_SEED = 20261014          # = experiments/ks_vs_am, so the intervals reproduce
MIN_CLASS_N = 10              # a within-class rho on fewer variants is not shown
ORIGINAL = REPO / "experiments" / "ks_vs_am"
INK = "#222222"


def load():
    parts = []
    for seed in SEEDS:
        d = labeled(size=N, seed=seed, genes=ANNOTATION_GENES)
        check_cohort(d, verbose=False)
        parts.append(d[["gene", "mutation", "label", "afm_score", "plddt_ks_residue"]])
    v = pd.concat(parts)
    out = v.groupby(["gene", "mutation", "label"], as_index=False).agg(
        ks=("plddt_ks_residue", "mean"), am=("afm_score", "first"))
    if out.am.isna().any() or out.ks.isna().any():
        raise ValueError("missing AM or KS scores")
    check_cohort(out, verbose=False)
    return out



def _rho(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _boot(x, y, rng, clusters=None):
    """BOOT resampled rho; variants, or whole residue positions when ``clusters`` is given."""
    n = len(x)
    out = np.empty(BOOT)
    for b in range(BOOT):
        i = rng.integers(0, n, n) if clusters is None else unstratified_draw(rng, clusters)
        out[b] = _rho(x[i], y[i])
    return out


def _fisher(r):
    return np.tanh(np.mean(np.arctanh(np.clip(r, -.9999, .9999)), axis=0))


def correlations(df, unit="variant"):
    """rho with 95% bootstrap CI per gene x subset, and the equal-gene macro (Fisher z).

    unit="variant" resamples variants (the published procedure until 2026-10-20);
    unit="position" resamples whole residue positions (the published unit since
    2026-10-20, Dan); unit="singleton" is that code with one variant per cluster and
    must reproduce "variant" exactly (checked in main).
    """
    rows, boots = [], {}
    for g in ANNOTATION_GENES:
        d = df[df.gene == g]
        rng = np.random.default_rng(BOOT_SEED + zlib.crc32(g.encode()))
        for part, s in [("all", d), ("drivers", d[d.label == 1]), ("passengers", d[d.label == 0])]:
            x, y = s.am.to_numpy(), s.ks.to_numpy()
            clusters = None if unit == "variant" else clusters_of(s.mutation, singleton=unit == "singleton")
            bs = _boot(x, y, rng, clusters) if len(s) >= 5 else np.full(BOOT, np.nan)
            if part == "all":
                boots[g] = bs
            lo, hi = np.nanquantile(bs, [.025, .975])
            rows.append({"gene": g, "subset": part, "n": len(s),
                         "rho": float(spearmanr(x, y).statistic), "ci_lo": float(lo), "ci_hi": float(hi)})
    st = pd.DataFrame(rows)
    per = st[st.subset == "all"].set_index("gene").loc[ANNOTATION_GENES, "rho"].to_numpy()
    mb = _fisher(np.vstack([boots[g] for g in ANNOTATION_GENES]))
    macro = {"rho": float(_fisher(per)), "ci_lo": float(np.quantile(mb, .025)),
             "ci_hi": float(np.quantile(mb, .975))}
    return st, macro


#: v6_ca per-gene rho (all variants), as reported by the v6_ca run (2026-10-16)
V6_REFERENCE = {"POLE": 0.64, "PTEN": 0.53, "VHL": 0.62, "MSH2": 0.76, "MSH6": 0.73, "PIK3CA": 0.47}
#: genes whose inputs v6 changed (wild types regenerated with the variants' templates)
V6_CHANGED = {"MSH2", "PIK3CA"}
REFERENCE_BY_CONFIG = {
    "d61d29328362": (V6_REFERENCE, V6_CHANGED),
    # Single-WT POLE repair; rho=0.6313611126606762 independently checked
    # with scipy and pandas on the ten-draw mean scores.
    "ebbbda31758d": (V6_REFERENCE | {"POLE": 0.63}, V6_CHANGED | {"POLE"}),
    # 2026-10-20: POLE on AF3 seeds 1-25 for all eight 32-seed drivers, P356S out of
    # the cohort (experiments/annotation_stages). Only POLE may move; its new rho is not yet
    # verified, so it is left out of the reference until the rerun reports it.
    "49028c16cbb3": ({g: r for g, r in V6_REFERENCE.items() if g != "POLE"}, V6_CHANGED | {"POLE"}),
    # 2026-11-03: the same six genes, FOXA1 removed from the config (no annotation gene changes).
    "97f1cdbfda8e": ({g: r for g, r in V6_REFERENCE.items() if g != "POLE"}, V6_CHANGED | {"POLE"}),
}


def check_against_original(st, macro):
    """Check the reference and allowed changed genes for this scientific config."""
    fingerprint = protein_config_fingerprint()
    if fingerprint not in REFERENCE_BY_CONFIG:
        raise ValueError(f"No verified correlation reference for config {fingerprint}")
    reference, changed_genes = REFERENCE_BY_CONFIG[fingerprint]
    allv = st[st.subset == "all"].set_index("gene").rho
    bad = {g: round(allv[g], 3) for g, r in reference.items() if round(allv[g], 2) != r}
    print(f"  config {fingerprint} reference check: {'OK' if not bad else bad}")
    if bad:
        raise AssertionError(f"per-gene rho differ from the config reference: {bad}")
    f = ORIGINAL / "ks_am_spearman.csv"
    if not f.is_file():
        print("  (experiments/ks_vs_am not found; v5 comparison skipped)")
        return
    o = pd.read_csv(f)
    m = st.merge(o, on=["gene", "subset"], suffixes=("", "_v5"))
    m["d"] = (m.rho - m.rho_v5).abs()
    moved = sorted(set(m.loc[m.d > 1e-9, "gene"]))
    print("  change against experiments/ks_vs_am (v5): "
          + (", ".join(f"{r.gene}/{r.subset} {r.rho_v5:.3f}->{r.rho:.3f}"
                       for r in m[m.d > 1e-9].itertuples()) or "none"))
    om = json.loads((ORIGINAL / "ks_vs_am.json").read_text())["macro_fisher_z_equal_weight"]
    print(f"  macro {om['rho']:.3f} (v5) -> {macro['rho']:.3f} (v6)")
    if not set(moved) <= changed_genes:
        raise AssertionError(f"genes this config did not touch moved: {sorted(set(moved) - changed_genes)}")


def _alpha(n):
    return float(np.clip(.9 * np.sqrt(40 / max(n, 1)), .3, .9))


def _class_handles():
    """Full-opacity legend markers (the plotted points are faded by count)."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="", marker="o", ms=4, color=DRV, label="Driver"),
            Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=PAS, mew=0.8,
                   label="Passenger")]


def panel_per_gene(df, st):
    fig, axes = grid(2, 3, w=W2, h=W2 * 0.66)
    fig.subplots_adjust(hspace=0.5, wspace=0.28)
    for ax, g in zip(axes.flat, ANNOTATION_GENES):
        d = df[df.gene == g]
        p, q = d[d.label == 0], d[d.label == 1]
        ax.scatter(p.am, p.ks, s=7, facecolors="none", edgecolors=PAS, lw=0.6,
                   alpha=_alpha(len(p)), label="Passenger")
        ax.scatter(q.am, q.ks, s=7, color=DRV, lw=0, alpha=_alpha(len(q)), label="Driver")
        slope, icpt = np.polyfit(d.am, d.ks, 1)
        xs = np.array([d.am.min(), d.am.max()])
        ax.plot(xs, icpt + slope * xs, color=INK, lw=0.9)
        r = st[(st.gene == g) & (st.subset == "all")].iloc[0]
        ax.text(0.03, 0.97, f"ρ = {r.rho:.2f}\nn = {len(d):,} ({len(q)} drivers)",
                transform=ax.transAxes, va="top", fontsize=6)
        ax.set_title(pretty(g), color=GENE_COLOR[g], loc="left", fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("AlphaMissense score")
    for ax in axes[:, 0]:
        ax.set_ylabel("KS-score (N = 90)")
    fig.legend(handles=_class_handles(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.2)
    save(fig, "ks_vs_am_per_gene")


def panel_rho_forest(st, macro):
    """Option A for panel b: rho per gene for all variants and within each class."""
    subsets = [("all", INK, "o", "All variants"), ("drivers", DRV, "s", "Drivers"),
               ("passengers", PAS, "^", "Passengers")]
    offs = {"all": 0.22, "drivers": 0.0, "passengers": -0.22}
    rows = ANNOTATION_GENES + ["Macro"]
    fig, ax = panel(w=W1, h=W1 * 0.8)
    ys = {g: len(rows) - 1 - i for i, g in enumerate(rows)}
    for part, colour, marker, label in subsets:
        first = True
        for g in ANNOTATION_GENES:
            r = st[(st.gene == g) & (st.subset == part)].iloc[0]
            if r.n < MIN_CLASS_N:
                continue
            y = ys[g] + offs[part]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=colour, lw=0.9)
            ax.plot(r.rho, y, marker, color=colour, ms=3.2, mfc=colour if part != "passengers" else "white",
                    label=label if first else None)
            first = False
    ax.plot([macro["ci_lo"], macro["ci_hi"]], [ys["Macro"]] * 2, color=INK, lw=0.9)
    ax.plot(macro["rho"], ys["Macro"], "D", color=INK, ms=3.6)
    ax.axvline(0, color=GRY, lw=0.6, ls=":")
    ax.axhline(0.5, color=GRY, lw=0.4)
    ax.set_yticks([ys[g] for g in rows])
    ax.set_yticklabels([pretty(g) for g in rows])
    ax.set_xlabel("Spearman ρ, KS-score vs AlphaMissense (N = 90)")
    ax.set_xlim(-1, 1)
    ax.legend(loc="upper left", fontsize=5.5)
    save(fig, "ks_vs_am_rho_forest")


def panel_pooled(df, macro):
    """Option B for panel b: the six genes pooled as within-gene percentiles."""
    d = df.copy()
    for col in ("am", "ks"):
        d[f"{col}_pct"] = d.groupby("gene")[col].rank(pct=True)
    fig, ax = panel(w=W1, h=W1 * 0.9)
    p, q = d[d.label == 0], d[d.label == 1]
    ax.scatter(p.am_pct, p.ks_pct, s=5, facecolors="none", edgecolors=PAS, lw=0.5, alpha=0.35,
               label="Passenger")
    ax.scatter(q.am_pct, q.ks_pct, s=5, color=DRV, lw=0, alpha=0.45, label="Driver")
    slope, icpt = np.polyfit(d.am_pct, d.ks_pct, 1)
    ax.plot([0, 1], [icpt, icpt + slope], color=INK, lw=0.9)
    ax.text(0.03, 0.97, f"macro ρ = {macro['rho']:.2f} (95% CI {macro['ci_lo']:.2f}–{macro['ci_hi']:.2f})"
            f"\n{len(d):,} variants, six genes", transform=ax.transAxes, va="top", fontsize=6)
    ax.set_xlabel("AlphaMissense, within-gene percentile")
    ax.set_ylabel("KS-score, within-gene percentile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(handles=_class_handles(), loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.1), handletextpad=0.2)
    save(fig, "ks_vs_am_pooled")


def main():
    df = load()
    df.to_csv(RESULTS / "ks_vs_am_variants.csv", index=False)   # Fig. S5 panel a (si_ks_vs_am_figure)
    # Residue positions are the published resampling unit since 2026-10-20 (Dan); the
    # variant-level intervals are kept for the record.
    st_variant, macro_variant = correlations(df, "variant")
    st_single, macro_single = correlations(df, "singleton")
    pd.testing.assert_frame_equal(st_variant, st_single, check_exact=True)
    assert macro_variant == macro_single, "singleton position clusters must reproduce the variant intervals"
    st, macro = correlations(df, "position")
    check_against_original(st, macro)
    st.assign(unit="position").to_csv(RESULTS / "ks_vs_am_correlations.csv", index=False)
    st_variant.assign(unit="variant").to_csv(RESULTS / "ks_vs_am_correlations_variant.csv", index=False)
    pd.DataFrame([dict(unit="position", **macro), dict(unit="variant", **macro_variant)]).to_csv(
        RESULTS / "ks_vs_am_macro.csv", index=False)
    panel_per_gene(df, st)
    panel_rho_forest(st, macro)
    panel_pooled(df, macro)
    for r in st.itertuples():
        note = f"95% CI [{r.ci_lo:.3f}, {r.ci_hi:.3f}] (residue positions), n={r.n}"
        if r.n < MIN_CLASS_N:
            note += "; not drawn (n < 10)"
        record("ks_vs_am", f"{r.gene}_{r.subset}_rho", round(r.rho, 4), note)
    record("ks_vs_am", "macro_rho_fisher_z", round(macro["rho"], 4),
           f"95% CI [{macro['ci_lo']:.3f}, {macro['ci_hi']:.3f}] (residue positions), equal-gene, Fisher z; "
           f"variant-level CI [{macro_variant['ci_lo']:.3f}, {macro_variant['ci_hi']:.3f}]")
    write_values("ks_vs_am_values.csv")
    provenance(figure="ks_vs_am", size=N, seeds=list(SEEDS), n_boot=BOOT, boot_seed=BOOT_SEED,
               cohort="annotation (1,798)")


if __name__ == "__main__":
    main()
