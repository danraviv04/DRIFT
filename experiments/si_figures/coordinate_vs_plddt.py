#!/usr/bin/env python
"""SI figure: wild-type positional spread, and position-based descriptors against the KS-score.

The combined coordinate figure (user, 2026-10-16). Its number comes from its position in
drift/si_panels.SI_FIGURE_LIST (S2).

WHAT IT ESTABLISHES
-------------------
- Panel a backs the main-text sentence that a confidently predicted residue barely moves
  across a WT ensemble (the "0.21 A" sentence, 01_main L154). Per gene, each residue's
  WT positional SD (RMSF) is plotted against its WT mean pLDDT. Spread is small wherever
  pLDDT is high, and it rises as pLDDT falls.
- Panel b backs Note S2 and the main-text comparison of pLDDT with positions. On the same
  canonical variants and labels at N = 90 it shows driver-vs-passenger ROC-AUC for the
  pLDDT KS-score, positional KS, ensemble-mean RMSD and RMSF. Positional KS is the best
  coordinate descriptor. It is above the pLDDT KS-score in PTEN and PIK3CA, and below it
  in the other four genes and on the macro.

SOURCE
------
- Coordinates: the core-fit caches (figlib.coords, frame="core_fit"). Every structure is
  re-fitted onto one WT rigid core per gene (experiments/coord_core_fit/REPORT.md,
  run_all.sh K_coordinates, 2026-10-16). Panel a's WT RMSF is computed here from the WT
  artifact named in each gene's core-fit report. Its WT pLDDT is the per-residue mean
  of the same WT's pLDDT CSV (the report's rows).
- Coordinate descriptors at N = 90: the core-fit ensemble sweeps,
  metric_analysis/structure_metrics/<gene>/ensemble_aware/ensemble_sweeps/
  <gene>_{RMSF,RMSD,POSKS}_aligned_coord_ensemble_sweep_raw.csv, averaged over repeats.
  Columns follow SI Note S2:
  - RMSD = rmsd_ensemble_mean_to_wt_mean (the two mean structures);
  - RMSF = rmsf_mean (the mutant's own mean RMSF);
  - positional KS = posks_distance (posks_axis is dropped).
- pLDDT KS-score: the fusion caches (figlib.labeled, v6_ca), N = 90, mean over the ten
  subsampling seeds. These are the same scores as S1 and S5.
- Variants are matched on the artifact each canonical variant was scored from, as in
  experiments/coord_core_fit/posks_vs_plddt_ks.py.
- AUC: per subsample (seed or sweep repeat), then averaged, as in Methods, Fig. 2a and
  S1. The pLDDT KS AUCs therefore equal S1's, and the coordinate AUCs equal the core-fit
  sweep summaries (coord_core_fit/results/compare_n90.csv); both are checked.
  coord_core_fit/results/posks_vs_plddt_ks.csv differs slightly, by design. It takes
  the AUC of draw-averaged scores, with pLDDT KS from the size-sweep's own draws.
  Annotation cohort, 1,799 variants, less POLE P356S, which has no coordinates.
- KRAS4B and TP53 have no coordinate descriptors. FOXA1 is not an annotation gene and
  its cache is damaged.

    DRIFT_RUN_TAG=v6_ca PYTHONPATH="." python \\
        experiments/si_figures/coordinate_vs_plddt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: F403,E402
from scipy.stats import spearmanr  # noqa: E402

from metric_analysis.pipeline.labels import artifact_key  # noqa: E402

N = 90
SEEDS = tuple(range(42, 52))
N_BOOT = 2000
BOOT_SEED = 20261016
PLDDT_MIN = 70.0
#: (label, sweep family, column, colour, marker); the pLDDT KS-score comes from the caches
COORD = [("Positional KS", "POSKS", "posks_distance", "#e07a1f", "s"),
         ("RMSD", "RMSD", "rmsd_ensemble_mean_to_wt_mean", GRY, "^"),
         ("RMSF", "RMSF", "rmsf_mean", "#9a9a9a", "v")]
KS_LABEL = "pLDDT KS-score"
#: the only canonical variant without coordinates (its archive could not be read)
NO_COORDS = {("POLE", "P356S")}
INK = "#222222"


# ── panel a: WT RMSF against WT pLDDT ─────────────────────────────────────────

def wt_rmsf_plddt(gene):
    """Per residue of the scored chain: WT mean pLDDT, WT RMSF (core-fit frame), core flag."""
    rep = coord_fit_report(gene)
    c = coords(gene, artifact=rep["wt_artifact"])
    c = c[c.chain.astype(str) == rep["chain"]]
    xyz = c[["x_aligned", "y_aligned", "z_aligned"]].to_numpy()
    c = c.assign(**{k: xyz[:, i] for i, k in enumerate("xyz")})
    mean = c.groupby("resid")[["x", "y", "z"]].transform("mean").to_numpy()
    c["sq"] = ((xyz - mean) ** 2).sum(axis=1)
    per = c.groupby("resid").agg(rmsf=("sq", lambda s: float(np.sqrt(s.mean()))),
                                 n=("sq", "size"), core=("core", "first")).reset_index()
    lo, hi = rep["plddt_rows"]
    pl = plddt_matrix(rep["wt_plddt_csv"])[lo:hi].mean(axis=1)
    if len(pl) != rep["residues"] or len(per) != rep["residues"]:
        raise ValueError(f"{gene}: {len(per)} coordinate residues, {len(pl)} pLDDT rows, "
                         f"report says {rep['residues']}")
    # residues map by order: the chain's residues are the pLDDT block's rows, in sequence
    per = per.sort_values("resid").reset_index(drop=True)
    per["plddt"] = pl
    per["gene"] = gene
    return per, rep


def panel_a(tables):
    fig, axes = grid(2, 3, w=W2, h=W2 * 0.56, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.42, wspace=0.12)
    for ax, g in zip(axes.flat, ANNOTATION_GENES):
        t = tables[g]
        for core, colour, lab, z in [(False, GRY, "outside the rigid core", 1),
                                     (True, GENE_COLOR[g], "rigid core", 2)]:
            s = t[t.core == core]
            ax.scatter(s.plddt, s.rmsf, s=2.2, lw=0, color=colour, alpha=0.55, zorder=z,
                       label=lab, rasterized=True)
        ax.axvline(PLDDT_MIN, color=INK, lw=0.5, ls=":")
        ax.set_yscale("log")
        ax.set_title(pretty(g), color=GENE_COLOR[g], loc="left", fontweight="bold")
        conf = t[t.plddt >= PLDDT_MIN]
        rho = spearmanr(t.plddt, t.rmsf).statistic
        ax.text(0.03, 0.04, f"median RMSF, pLDDT ≥ 70: {conf.rmsf.median():.2f} Å\nρ = {rho:.2f}",
                transform=ax.transAxes, fontsize=5.5, va="bottom", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
    for ax in axes[1]:
        ax.set_xlabel("WT mean pLDDT")
    for ax in axes[:, 0]:
        ax.set_ylabel("WT RMSF (Å)")
    fig.text(0.5, 1.0, "Each point is a residue of the scored chain. Coloured: in the gene's WT "
             "rigid core (the fitting frame); grey: outside it. Dotted line: pLDDT 70.",
             ha="center", va="bottom", fontsize=6)
    save(fig, "wt_rmsf_vs_plddt")


def record_a(tables):
    meds, rhos = [], []
    for g in ANNOTATION_GENES:
        t = tables[g]
        conf = t[t.plddt >= PLDDT_MIN]
        med = float(conf.rmsf.median())
        rho = float(spearmanr(t.plddt, t.rmsf).statistic)
        meds.append(med)
        rhos.append(rho)
        record("wt_rmsf_vs_plddt", f"{g}_median_rmsf_plddt70_A", round(med, 3),
               f"{len(conf)} of {len(t)} residues with WT mean pLDDT >= 70")
        record("wt_rmsf_vs_plddt", f"{g}_spearman_rmsf_vs_plddt", round(rho, 3),
               f"all {len(t)} residues of the scored chain")
        record("wt_rmsf_vs_plddt", f"{g}_median_rmsf_core_A",
               round(float(t[t.core].rmsf.median()), 3), f"{int(t.core.sum())} core residues")
    record("wt_rmsf_vs_plddt", "macro_median_rmsf_plddt70_A", round(float(np.mean(meds)), 3),
           f"equal-gene mean; range {min(meds):.2f}-{max(meds):.2f}")
    record("wt_rmsf_vs_plddt", "macro_spearman_rmsf_vs_plddt", round(float(np.mean(rhos)), 3),
           f"equal-gene mean; range {min(rhos):.2f} to {max(rhos):.2f}")


# ── panel b: AUC of the four descriptors on the same variants ────────────────

def auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def variant_table():
    """Long table: gene, mutation, label, key, descriptor, draw (seed or repeat), value.

    pLDDT KS: one value per subsampling seed (fusion caches). Coordinate descriptors:
    one value per sweep repeat at N = 90. AUCs are taken per draw and then averaged,
    as in Methods ("AUCs were calculated for each subsample and averaged"), Fig. 2a and S1.
    """
    parts = []
    for seed in SEEDS:
        d = labeled(size=N, seed=seed, genes=ANNOTATION_GENES)
        check_cohort(d, verbose=False)
        parts.append(d[["gene", "mutation", "label", "plddt_ks_residue"]]
                     .rename(columns={"plddt_ks_residue": "value"})
                     .assign(descriptor=KS_LABEL, draw=seed))
    ks = pd.concat(parts, ignore_index=True)
    man = pd.read_csv(STRICT_MANIFEST)
    man = man[man.cohort_status == "included"]
    man["key"] = man.source_file.map(lambda f: artifact_key(Path(f).name.replace("_plddt_values.csv", "")))
    keys = (ks[["gene", "mutation", "label"]].drop_duplicates()
            .merge(man[["protein", "mutation_code", "key"]],
                   left_on=["gene", "mutation"], right_on=["protein", "mutation_code"], how="left")
            [["gene", "mutation", "label", "key"]])
    if keys.key.isna().any() or len(keys) != sum(n for n, _ in EXPECTED_COHORT.values()):
        raise ValueError("canonical variants without a manifest artifact")
    out = [ks.merge(keys, on=["gene", "mutation", "label"])]
    for g in ANNOTATION_GENES:
        kg = keys[keys.gene == g]
        for lab, fam, col, _, _ in COORD:
            raw = pd.read_csv(STRUCTURE_METRICS / g / "ensemble_aware" / "ensemble_sweeps"
                              / f"{g}_{fam}_aligned_coord_ensemble_sweep_raw.csv",
                              usecols=["artifact", "ensemble_size", "repeat", col])
            raw = raw[raw.ensemble_size == N].assign(key=lambda r: r.artifact.map(artifact_key))
            m = kg.merge(raw[["key", "repeat", col]], on="key", how="inner")
            got = set(zip(m.gene, m.mutation))
            missing = set(zip(kg.gene, kg.mutation)) - got
            if missing - NO_COORDS:
                raise ValueError(f"{g} {lab}: no coordinates for {sorted(missing - NO_COORDS)}")
            out.append(m.rename(columns={col: "value", "repeat": "draw"})
                       [["gene", "mutation", "label", "key", "value", "draw"]].assign(descriptor=lab))
    v = pd.concat(out, ignore_index=True)
    # every descriptor on the same variants: drop the canonical variants without coordinates
    v = v[~v.set_index(["gene", "mutation"]).index.isin(list(NO_COORDS))]
    per = v.groupby(["gene", "descriptor", "mutation"]).draw.nunique()
    if per.nunique() != 1:
        raise ValueError("unequal numbers of draws per variant")
    return v


def auc_table(v):
    """Mean over draws of the per-draw AUC, per gene, with a class-stratified variant
    bootstrap (the same resampled variants for every draw and descriptor); the macro is
    the equal-gene mean within a replicate."""
    rng = np.random.default_rng(BOOT_SEED)
    labels = [KS_LABEL] + [lab for lab, _, _, _, _ in COORD]
    est, boots = {}, {}
    for g in ANNOTATION_GENES:
        b = v[v.gene == g]
        y = b.drop_duplicates("mutation").set_index("mutation").label
        mats = {lab: b[b.descriptor == lab].pivot(index="mutation", columns="draw", values="value")
                .loc[y.index].to_numpy() for lab in labels}
        y = y.to_numpy()
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        idx = [np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
               for _ in range(N_BOOT)]

        def mean_auc(M, i):
            return np.mean([auc(y[i], M[i, j]) for j in range(M.shape[1])])

        for lab in labels:
            M = mats[lab]
            est[(g, lab)] = mean_auc(M, np.arange(len(y)))
            boots[(g, lab)] = np.array([mean_auc(M, i) for i in idx])
    rows = []
    for lab in labels:
        for g in ANNOTATION_GENES:
            bs = boots[(g, lab)]
            rows.append({"gene": g, "descriptor": lab, "auc": est[(g, lab)],
                         "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
        mb = np.mean([boots[(g, lab)] for g in ANNOTATION_GENES], axis=0)
        rows.append({"gene": "Macro", "descriptor": lab,
                     "auc": np.mean([est[(g, lab)] for g in ANNOTATION_GENES]),
                     "lo": np.percentile(mb, 2.5), "hi": np.percentile(mb, 97.5)})
    return pd.DataFrame(rows)


def check_reference(t, v):
    """The pLDDT KS-score must reproduce S1 (same caches, same per-seed AUCs) where no
    variant was dropped; the coordinate AUCs must reproduce the core-fit sweep summaries
    (experiments/coord_core_fit/results/compare_n90.csv) to 0.002."""
    got = t.set_index(["gene", "descriptor"]).auc
    bad = {}
    s1 = pd.read_csv(RESULTS / "descriptors_auc_per_gene_size_seed.csv")
    s1 = s1[(s1["size"] == N) & (s1.descriptor == "KS-score")].groupby("gene").auc.mean()
    for g in ANNOTATION_GENES:
        if g in {gg for gg, _ in NO_COORDS}:
            continue
        if abs(got[(g, KS_LABEL)] - s1[g]) > 1e-12:
            bad[(g, KS_LABEL)] = (round(got[(g, KS_LABEL)], 4), round(s1[g], 4))
    cmp_ = REPO / "experiments" / "coord_core_fit" / "results" / "compare_n90.csv"
    if cmp_.is_file():
        c = pd.read_csv(cmp_).set_index(["dataset", "metric_col"]).auc_mean_core_fit
        for lab, _, col, _, _ in COORD:
            for g in ANNOTATION_GENES:
                if abs(got[(g, lab)] - c[(g, col)]) > 0.002:
                    bad[(g, lab)] = (round(got[(g, lab)], 4), c[(g, col)])
    print(f"  cross-check (S1 KS; coord_core_fit sweep AUCs): {'OK' if not bad else bad}")
    if bad:
        raise AssertionError(f"does not reproduce the reference AUCs: {bad}")


def panel_b(t):
    fig, ax = panel(w=W15, h=W1 * 0.85)
    genes = ANNOTATION_GENES + ["Macro"]
    ys = np.arange(len(genes))[::-1]
    series = [(KS_LABEL, ACC, "o")] + [(lab, colour, mk) for lab, _, _, colour, mk in COORD]
    offs = np.linspace(0.27, -0.27, len(series))
    for (lab, colour, mk), off in zip(series, offs):
        s = t[t.descriptor == lab].set_index("gene").loc[genes]
        for y, (g, r) in zip(ys, s.iterrows()):
            ax.plot([r.lo, r.hi], [y + off] * 2, color=colour, lw=0.7)
            ax.plot(r.auc, y + off, mk, color=colour, ms=3.4 if g != "Macro" else 4.2,
                    mfc=colour if lab in (KS_LABEL, "Positional KS") else "white", mew=0.8,
                    label=lab if g == genes[0] else None)
    ax.axvline(0.5, color=GRY, lw=0.5, ls=":")
    ax.axhline(0.5, color=GRY, lw=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty(g) for g in genes])
    ax.set_xlabel(f"Driver vs passenger ROC-AUC (N = {N})")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.2,
              columnspacing=0.9)
    save(fig, "positional_ks_vs_plddt_ks")


def record_b(t, v):
    for r in t.itertuples():
        record("positional_ks_vs_plddt_ks", f"{r.gene}_{r.descriptor.replace(' ', '_')}_AUC_N90",
               round(r.auc, 4), f"mean of per-draw AUCs; 95% CI [{r.lo:.3f}, {r.hi:.3f}], "
               f"{N_BOOT} class-stratified bootstraps")
    m = v.groupby(["gene", "mutation", "descriptor"]).value.mean().unstack()
    for g in ANNOTATION_GENES:
        b = m.loc[g]
        record("positional_ks_vs_plddt_ks", f"{g}_spearman_posks_vs_plddt_ks",
               round(float(spearmanr(b["Positional KS"], b[KS_LABEL]).statistic), 3),
               f"draw-mean scores, n={len(b)}")


def main():
    tables, reports = {}, {}
    for g in ANNOTATION_GENES:
        tables[g], reports[g] = wt_rmsf_plddt(g)
        print(f"  {g}: WT {reports[g]['wt_artifact']} chain {reports[g]['chain']}, "
              f"{len(tables[g])} residues, {int(tables[g].core.sum())} core")
    pd.concat(tables.values()).to_csv(RESULTS / "wt_rmsf_vs_plddt_residues.csv", index=False)
    panel_a(tables)
    record_a(tables)

    v = variant_table()
    v.to_csv(RESULTS / "positional_ks_vs_plddt_ks_variants.csv.gz", index=False)
    t = auc_table(v)
    check_reference(t, v)
    t.to_csv(RESULTS / "positional_ks_vs_plddt_ks_auc.csv", index=False)
    panel_b(t)
    record_b(t, v)
    write_values("coordinate_vs_plddt_values.csv")
    provenance(figure="coordinate_vs_plddt", size=N, seeds=list(SEEDS), n_boot=N_BOOT,
               boot_seed=BOOT_SEED, frame="core_fit (WT rigid core, experiments/coord_core_fit)",
               wt_artifacts={g: r["wt_artifact"] for g, r in reports.items()},
               cohort="annotation (1,798); POLE P356S (no coordinates) left the cohort 2026-10-20")
    print(t.pivot(index="gene", columns="descriptor", values="auc").round(3).loc[ANNOTATION_GENES + ["Macro"]].to_string())


if __name__ == "__main__":
    main()
