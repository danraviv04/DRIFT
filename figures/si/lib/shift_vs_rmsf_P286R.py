#!/usr/bin/env python
"""Per-residue centre shift against ΔRMSF for POLE P286R (proposed Supplementary Fig. S1a).

WHAT IT ESTABLISHES
-------------------
The residue-level view behind the main-text sentence on Fig. 1c,d. For every one of the
204 POLE residues, x is the distance between the WT and P286R ensemble-mean Cα
positions (Fig. 1c), and y is the change in Cα RMSF, P286R minus WT (Fig. 1d).
Variability rose at nearly every residue, and the size of a residue's shift only partly
predicted the change in its variability. Residues are spatially dependent, so the
correlations are descriptive and no significance test is reported.

SOURCE
------
- Cα coordinates from the WT rigid-core frame cache (figlib.coords, frame="core_fit"),
  the frame of Supplementary Note S1. The 125 predictions of AF3 seeds 1-25 on each
  side (2026-10-20, analyses/annotation/stages: P286R was predicted with seeds 1-32, the
  WT with 1-25; until then this took the first 125 by structure_index, which for
  P286R mixed in seeds 26-32).
- Cross-check: analyses/coordinates/rmsf_histogram/exemplar_P286R_rmsf.csv, the table behind
  Fig. 1c,d (US-align frame). For POLE the rigid core is all 204 residues, so the two
  frames should agree closely; the largest per-residue differences are recorded.

Replaces analyses/coordinates/rmsf_histogram/mean_shift_vs_spread.py, which plotted the mean
Cα distance from the ensemble centre rather than RMSF.

    PYTHONPATH="." python \\
        figures/si/lib/shift_vs_rmsf_P286R.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: F403,E402

from scipy.stats import linregress, pearsonr, spearmanr  # noqa: E402

WT, MUT = "POLE_268-471_WT", "POLE_268-471_P286R"
N = 125
FIRST_RESID = 268          # POLE construct numbering: resid 1 = residue 268
SITE = 286
XYZ = ["x_aligned", "y_aligned", "z_aligned"]
FIG1_TABLE = REPO / "analyses" / "coordinates" / "rmsf_histogram" / "exemplar_P286R_rmsf.csv"
OUT = RESULTS / "shift_vs_rmsf"
NAME = "shift_vs_rmsf_P286R"


def ensemble(artifact):
    """The predictions of AF3 seeds 1-25 (N = 125); per-residue centre and RMSF."""
    from metric_analysis.pipeline.library.af3_seeds import within_seed_cap
    d = coords("POLE", artifact, columns=["structure_index", "sample", "resid", "resname", *XYZ])
    d = d[d["sample"].notna() & within_seed_cap(d["sample"], "POLE")]
    if d.structure_index.nunique() != N:
        raise ValueError(f"{artifact}: {d.structure_index.nunique()} seed 1-25 predictions, need {N}")
    if d.groupby("resid").size().nunique() != 1:
        raise ValueError(f"{artifact}: residues missing from some predictions")
    centre = d.groupby("resid")[XYZ].mean()
    dev = d[XYZ].to_numpy() - centre.loc[d.resid].to_numpy()
    rmsf = (pd.Series((dev ** 2).sum(axis=1), index=d.resid.to_numpy())
            .groupby(level=0).mean() ** 0.5)
    resname = d.drop_duplicates("resid").set_index("resid").resname
    return centre, rmsf, resname


def table():
    cw, rw, nw = ensemble(WT)
    cm, rm, nm = ensemble(MUT)
    if not cw.index.equals(cm.index):
        raise ValueError("WT and mutant residues differ")
    t = pd.DataFrame({
        "pole_resid": cw.index + FIRST_RESID - 1,
        "centre_shift_A": np.linalg.norm(cm.to_numpy() - cw.to_numpy(), axis=1),
        "wt_rmsf_A": rw.loc[cw.index].to_numpy(),
        "mut_rmsf_A": rm.loc[cw.index].to_numpy(),
        "wt_resname": nw.loc[cw.index].to_numpy(),
        "mut_resname": nm.loc[cw.index].to_numpy(),
    })
    t["delta_rmsf_A"] = t.mut_rmsf_A - t.wt_rmsf_A
    if len(t) != 204 or t.pole_resid.tolist() != list(range(268, 472)):
        raise ValueError("unexpected residue range")
    changed = t.loc[t.wt_resname != t.mut_resname, "pole_resid"].tolist()
    if changed != [SITE]:
        raise ValueError(f"unexpected substitutions: {changed}")
    return t


def cross_check(t):
    """Against the Fig. 1c,d table (US-align frame)."""
    ref = pd.read_csv(FIG1_TABLE).set_index("pole_resid").loc[t.pole_resid]
    d_shift = np.abs(t.centre_shift_A.to_numpy() - ref.centre_shift.to_numpy()).max()
    d_rmsf = np.abs(t.delta_rmsf_A.to_numpy() - (ref.mut - ref.wt).to_numpy()).max()
    rho = spearmanr(t.delta_rmsf_A, ref.mut - ref.wt).statistic
    print(f"  vs Fig. 1c,d table: max |Δshift| {d_shift:.4f} Å, max |ΔΔRMSF| {d_rmsf:.4f} Å, "
          f"Spearman {rho:.3f}")
    return d_shift, d_rmsf, rho


def figure(t, fit, r, rho):
    x, y = t.centre_shift_A.to_numpy(), t.delta_rmsf_A.to_numpy()
    fig, ax = panel()
    ax.axhline(0, ls=(0, (3, 2)), lw=.6, color=GRY, zorder=1)
    ax.scatter(x, y, s=9, color=DRV, alpha=.55, edgecolor="none", zorder=2)
    xs = np.array([0, x.max()])
    ax.plot(xs, fit.intercept + fit.slope * xs, color="#2c3e50", lw=.9, zorder=3)
    j = int(np.flatnonzero(t.pole_resid.to_numpy() == SITE)[0])
    ax.scatter(x[j], y[j], s=40, facecolor="none", edgecolor="black", lw=.9, zorder=4)
    ax.annotate(f"P{SITE}R site", (x[j], y[j]), xytext=(-6, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=6, color="#4a4a4a")
    ax.text(.98, .04, f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="#4a4a4a")
    ax.set_xlim(left=0)
    ax.set_xlabel("Shift of ensemble-mean Cα position (Å)")
    ax.set_ylabel("ΔRMSF, P286R − WT (Å)")
    return fig


def main():
    t = table()
    d_shift, d_rmsf, rho_fig1 = cross_check(t)
    x, y = t.centre_shift_A.to_numpy(), t.delta_rmsf_A.to_numpy()
    r, rho, fit = pearsonr(x, y).statistic, spearmanr(x, y).statistic, linregress(x, y)
    rmsd = float(np.sqrt((x ** 2).mean()))

    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"{NAME}.csv", index=False)
    save(figure(t, fit, r, rho), NAME, OUT)

    p = "S1a"
    record(p, "n_residues", len(t))
    record(p, "n_per_ensemble", N)
    record(p, "mean_structure_rmsd_A", round(rmsd, 3), "RMS of the per-residue centre shift (Fig. 1c)")
    record(p, "residues_shift_gt_0.5A", int((x > .5).sum()))
    record(p, "residues_rmsf_higher_in_mutant", f"{int((y > 0).sum())}/{len(t)}")
    record(p, "pearson_r", round(float(r), 3), "descriptive; residues are spatially dependent")
    record(p, "spearman_rho", round(float(rho), 3), "descriptive; residues are spatially dependent")
    record(p, "ols_slope", round(float(fit.slope), 4), "ΔRMSF per Å of shift")
    record(p, "site_286_shift_A", round(float(x[t.pole_resid == SITE][0]), 3))
    record(p, "site_286_delta_rmsf_A", round(float(y[t.pole_resid == SITE][0]), 3))
    record(p, "crosscheck_max_abs_shift_diff_A", round(float(d_shift), 4), "vs exemplar_P286R_rmsf.csv (US-align)")
    record(p, "crosscheck_max_abs_delta_rmsf_diff_A", round(float(d_rmsf), 4), "vs exemplar_P286R_rmsf.csv (US-align)")
    record(p, "crosscheck_spearman_delta_rmsf", round(float(rho_fig1), 4), "vs exemplar_P286R_rmsf.csv (US-align)")
    write_values(f"{NAME}_values.csv", OUT)
    provenance(OUT, frame="core_fit (WT rigid core)", n_ensemble=N,
               wt_artifact=WT, mutant_artifact=MUT, crosscheck_table=str(FIG1_TABLE.relative_to(REPO)))


if __name__ == "__main__":
    main()
