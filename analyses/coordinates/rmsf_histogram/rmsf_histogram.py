#!/usr/bin/env python
"""Per-residue coordinate dispersion, POLE WT against a single mutant.

    conda activate thesis_core
    python analyses/coordinates/rmsf_histogram/rmsf_histogram.py --mutant P286R
    python analyses/coordinates/rmsf_histogram/rmsf_histogram.py --metric mad_median
    python analyses/coordinates/rmsf_histogram/rmsf_histogram.py --write-pdb

WHAT IS PLOTTED
Three per-residue quantities, all in Angstrom, all histogrammed over RESIDUES:
the x axis is a bin and the y axis is the fraction of the protein's residues in
that bin, so every curve integrates to 1 and pools of different depth stay
comparable.

  WT spread      dispersion of residue r across the WT structures, about the
                 WT mean position of r
  mutant spread  the same within the mutant ensemble
  mean shift     || mean_WT(r) - mean_mut(r) ||, the distance between the two
                 ensemble-mean positions of residue r

The first two are WITHIN-ensemble; they say how much a residue wanders. The
third is BETWEEN-ensemble; it says how far the average structure moved. They are
different quantities that happen to share units, which is the reason to plot
them together: a mutation can widen the ensemble without moving its centre, or
move the centre without widening it, and only the pair distinguishes those.
The mean-shift curve is the per-residue decomposition of the RMSD descriptor
(Methods eq. 3), whose value is the root-mean-square of exactly these numbers.

--metric selects the dispersion estimator for the first two curves:
  rmsf        sqrt( mean_i || p_i,r - mean_i(p_i,r) ||^2 )     (Methods eq. 4)
  mad_mean    mean_i || p_i,r - mean_i(p_i,r) ||               (default; the L1
              analogue of RMSF, same units and reading, less outlier-sensitive)
  mad_median  median_i || p_i,r - median_i(p_i,r) ||           (fully robust)
The mean-shift curve is unaffected by --metric; it is a distance between two
points, not a deviation statistic.

SOURCE
metric_analysis/structure_metrics/POLE/ensemble_aware/cache/
POLE_rmsf_aligned_ca_coordinate_cache.parquet -- CA coordinates already
superposed with US-align onto the shared reference frame. CA ONLY: there are no
side-chain or backbone atoms in this cache, which is what makes --write-pdb a
CA trace rather than a full model.

THREE CHOICES THAT MATTER, all made explicit rather than defaulted silently:

1. WHICH WT. The cache holds three POLE WT artifacts. Two of them,
   POLE_268-471_WT and POLE_WT, are the SAME ensemble stored in different
   superposition frames -- their coordinates differ by up to 0.54 A but their
   per-residue RMSF agrees to 0.006 A, which is what a rigid-body transform
   does. Only POLE_again_WT is an independent generation, and it differs from
   the primary by 0.017 A in mean RMSF. The pipeline takes the first in cache
   row order, POLE_268-471_WT, and that is what is used here.

   An earlier note here said POLE_268-471_WT's per-seed filenames had been
   collapsed, because the artifact has only 2 distinct values in the index's
   `structure_file` column. That column is the basename, and this construct
   names every member `model.cif` inside its `seed-N_sample-M/` directory --
   the `structure_path` column holds 126 distinct paths. Provenance is intact.

2. AF3's DUPLICATED TOP-RANKED MODEL, dropped by default; see `read_cache`.
   Each archive stores its best prediction twice, once as the seed-sample and
   once copied to the top as `{name}_model.cif`, which is what structure_index 0
   is. Coordinates identical, header different, so checksums do not catch it.
   Dropping it takes the pools from 126/161 to 125/160 and moves the P286R
   numbers a little: median spread ratio 1.556x -> 1.523x, median |MAD_wt -
   MAD_mut| 0.0370 -> 0.0344 A. "Broader at 201 of 204 residues" and the 0.21 A
   mean-structure RMSD are unchanged. --keep-duplicate restores the old
   behaviour for comparison.

3. ENSEMBLE DEPTH. Every one of these estimators grows with N, so comparing a
   125-structure pool against a 160-structure pool would confound mutation with
   depth. Both sides are truncated to the same N (the minimum over the two
   pools), taking the lowest structure_index values, which is deterministic.
   --no-match disables this. The mean-shift curve is affected too: a mean
   estimated from fewer structures is noisier, so matching matters there as well.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CACHE = (REPO / "metric_analysis" / "structure_metrics" / "POLE" / "ensemble_aware"
         / "cache" / "POLE_rmsf_aligned_ca_coordinate_cache.parquet")
INDEX = CACHE.parent / "POLE_structure_file_index.csv"

WT_PRIMARY = "POLE_268-471_WT"
XYZ = ["x_aligned", "y_aligned", "z_aligned"]

METRIC_LABEL = {
    "rmsf": "RMSF",
    "mad_mean": "MAD (mean absolute deviation)",
    "mad_median": "MAD (median absolute deviation)",
}


def cap_seeds(d, max_seed, quiet=False):
    """Drop structures whose AF3 seed is above ``max_seed`` (2026-10-20).

    Eight POLE drivers, P286R among them, were predicted with seeds 1-32; the WT and
    every other POLE ensemble have seeds 1-25, and the pLDDT analyses now use seeds
    1-25 for all of them (analyses/annotation/stages). The cache keeps all 160, so the
    seed is read from the structure file index and the extra 35 are dropped here.
    ``max_seed=None`` keeps every structure (the pre-2026-10-20 behaviour).
    """
    if max_seed is None:
        return d
    sys.path.insert(0, str(REPO))
    from metric_analysis.pipeline.library.af3_seeds import seed_numbers
    if not INDEX.exists():
        raise SystemExit(f"no structure file index at {INDEX}; cannot read AF3 seeds")
    idx = pd.read_csv(INDEX, usecols=["artifact", "structure_index", "structure_path", "structure_file"])
    idx = idx[idx.artifact.isin(set(d.artifact))]
    seeds = seed_numbers(idx.structure_path.astype(str) + " " + idx.structure_file.astype(str))
    over = set(zip(idx.artifact.astype(str)[seeds > max_seed],
                   idx.structure_index.astype("int64")[seeds > max_seed]))
    if not over:
        return d
    keep = [(art, i) not in over for art, i in
            zip(d.artifact.astype(str), d.structure_index.astype("int64"))]
    if not quiet:
        lost = pd.Series([a for a, _ in over]).value_counts()
        print(f"dropped AF3 seeds > {max_seed}: " + ", ".join(f"{a} -{int(k)}" for a, k in lost.items()))
    return d.loc[keep]


def read_cache(artifacts, keep_duplicate=False, quiet=False, max_seed=25):
    """Cache rows for the given artifacts, with AF3's duplicated top model removed.

    Structures with an AF3 seed above ``max_seed`` (default 25, the WT's seeds) are
    dropped too; see cap_seeds.

    Every archive holds its best-ranked prediction twice: once as the seed-sample it
    came from, and once copied to the top of the archive as `{name}_model.cif`. The
    coordinates are identical; only the file header differs, so a checksum does not
    catch it. Discovery globbed `**/*.cif`, so the cache carries both -- structure_index
    0 is always the top-level copy. Verified here against the cache itself: for
    POLE_268-471_WT index 0 equals index 57, for POLE_268-471_P286R it equals index 6.

    Left in, that one conformation gets weight 2/N in every mean, MAD and RMSF, and the
    subsampling sweeps can draw it twice -- which is exactly the ensemble diversity
    these statistics exist to measure. The duplicated member is a different one for each
    variant, whichever AF3 ranked best, so it is not even a consistent bias.

    The cache stores only (artifact, structure_index), so the duplicate is identified
    from the structure file index written beside it: an artifact's members that carry
    no `seed-N_sample-M` anywhere in their path. Matching the whole path matters here --
    POLE's truncated 268-471 construct names every member `model.cif` and carries the
    seed only in the directory name. Artifacts with no seed-named members at all are
    kept whole rather than emptied.

    This mirrors `_drop_top_ranked_duplicate_rows` in the main pipeline, which applies
    the same filter on cache load; the regex is imported from the pipeline so there is
    one definition of what a seed-sample looks like. These experiment scripts read the
    parquet directly and so bypassed it until now.
    """
    sys.path.insert(0, str(REPO))
    from metric_analysis.pipeline.library.structure_metrics import _SEED_SAMPLE_RE

    d = pd.read_parquet(CACHE, columns=["artifact", "structure_index", "resid",
                                        "resname", "chain"] + XYZ)
    d = d[d.artifact.isin(list(artifacts))]
    d = cap_seeds(d, max_seed, quiet)
    if keep_duplicate:
        if not quiet:
            print("KEEPING AF3's duplicated top-ranked model (--keep-duplicate): one "
                  "conformation counts twice in every statistic below.")
        return d
    if not INDEX.exists():
        raise SystemExit(
            f"no structure file index at {INDEX}; cannot identify AF3's duplicated "
            f"top-ranked model. Pass --keep-duplicate to proceed with the double-count.")

    idx = pd.read_csv(INDEX, usecols=["artifact", "structure_index",
                                      "structure_path", "structure_file"])
    idx = idx[idx.artifact.isin(list(artifacts))]
    is_seed = (idx.structure_path.astype(str) + " " + idx.structure_file.astype(str)
               ).str.contains(_SEED_SAMPLE_RE)
    has_seeded = idx.assign(_s=is_seed).groupby("artifact")["_s"].transform("any")
    dup = set(zip(idx.loc[has_seeded & ~is_seed, "artifact"].astype(str),
                  idx.loc[has_seeded & ~is_seed, "structure_index"].astype("int64")))
    if not dup:
        return d
    keep = [(art, i) not in dup for art, i in
            zip(d.artifact.astype(str), d.structure_index.astype("int64"))]
    out = d.loc[keep]
    if not quiet:
        lost = d.groupby("artifact").structure_index.nunique() - \
            out.groupby("artifact").structure_index.nunique()
        print("dropped AF3's duplicated top-ranked model: "
              + ", ".join(f"{a} -{int(k)}" for a, k in lost.items()))
    return out


def load(artifact, d, n_max=None):
    """Rows for one artifact, optionally truncated to the first n_max structures."""
    s = d[d.artifact == artifact]
    if s.empty:
        raise SystemExit(f"artifact not in cache: {artifact}")
    if n_max is not None:
        keep = sorted(s.structure_index.unique())[:n_max]
        s = s[s.structure_index.isin(keep)]
    return s.sort_values(["resid", "structure_index"])


def residue_table(s, metric):
    """Per-residue centre and dispersion for one ensemble."""
    g = s.groupby("resid")
    centre = g[XYZ].mean() if metric != "mad_median" else g[XYZ].median()
    ref = s[["resid"]].join(centre, on="resid")[XYZ].to_numpy()
    dist = np.linalg.norm(s[XYZ].to_numpy() - ref, axis=1)
    dd = pd.DataFrame({"resid": s.resid.to_numpy(), "dist": dist})
    if metric == "rmsf":
        disp = dd.assign(d2=dd.dist ** 2).groupby("resid").d2.mean().pow(0.5)
    elif metric == "mad_mean":
        disp = dd.groupby("resid").dist.mean()
    elif metric == "mad_median":
        disp = dd.groupby("resid").dist.median()
    else:
        raise SystemExit(f"unknown metric {metric}")
    out = centre.copy()
    out["disp"] = disp
    # resname is constant within a residue for a given artifact
    out["resname"] = s.groupby("resid").resname.first()
    out["chain"] = s.groupby("resid").chain.first()
    return out.reset_index()


def medoid(s, tab):
    """Index of the real structure whose CAs are closest to the ensemble mean.

    An averaged structure is not a structure: side chains occupy discrete
    rotamers, so a mean side-chain atom can land in physically impossible
    positions, and even averaged backbone has shortened bonds. When a full-atom
    model is wanted, the honest object is the medoid -- an actual prediction from
    the pool, chosen as the one minimising CA RMSD to the ensemble mean.
    """
    ref = tab.set_index("resid")[XYZ]
    best, best_rmsd = None, np.inf
    for idx, g in s.groupby("structure_index"):
        g = g.sort_values("resid")
        r = float(np.sqrt((((g[XYZ].to_numpy()
                             - ref.loc[g.resid].to_numpy()) ** 2).sum(1)).mean()))
        if r < best_rmsd:
            best, best_rmsd = idx, r
    return int(best), best_rmsd


def write_pdb(tab, path, title, offset=267):
    """CA-only PDB of the ensemble-mean coordinates, dispersion in the B-factor.

    The cache numbers the construct 1..204. offset=267 renumbers to the POLE
    numbering used throughout the manuscript (268..471), so P286R sits at
    residue 286 when the file is opened. offset=0 keeps construct numbering.
    """
    lines = [f"REMARK   1 {title}",
             f"REMARK   1 Residues numbered {1+offset}-{len(tab)+offset}"
             + (" (POLE numbering)." if offset else " (construct numbering)."),
             "REMARK   1 CA-ONLY ensemble-mean coordinates. This is a mean of",
             "REMARK   1 superposed CA positions, not a predicted structure: CA-CA",
             "REMARK   1 distances are not restrained and averaging shortens them.",
             "REMARK   1 Use for visualisation and per-residue comparison only.",
             "REMARK   1 B-factor column holds the per-residue dispersion in Angstrom."]
    for i, r in enumerate(tab.itertuples(), start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  {r.resname:>3} {r.chain}{int(r.resid)+offset:>4}    "
            f"{r.x_aligned:>8.3f}{r.y_aligned:>8.3f}{r.z_aligned:>8.3f}"
            f"{1.00:>6.2f}{min(r.disp, 999.99):>6.2f}           C")
    lines += ["TER", "END", ""]
    path.write_text("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", default="P286R")
    ap.add_argument("--metric", default="mad_mean",
                    choices=["rmsf", "mad_mean", "mad_median"])
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--no-match", action="store_true")
    ap.add_argument("--keep-duplicate", action="store_true",
                    help="keep AF3's duplicated top-ranked model (structure_index 0); "
                         "off by default, see read_cache")
    ap.add_argument("--write-pdb", action="store_true")
    ap.add_argument("--write-medoid", action="store_true",
                    help="copy out the full-atom source CIF of each ensemble's medoid")
    ap.add_argument("--pdb-numbering", default="pole", choices=["pole", "construct"],
                    help="pole = 268-471 as used in the manuscript; construct = 1-204")
    a = ap.parse_args()

    mut_artifact = f"POLE_268-471_{a.mutant}"
    d = read_cache([WT_PRIMARY, mut_artifact], a.keep_duplicate)

    depths = d.groupby("artifact").structure_index.nunique()
    n = None if a.no_match else int(depths.min())
    print("pool depths:"); print(depths.to_string())
    print(f"matched depth: {n if n else 'OFF (full pools)'}")
    print(f"metric: {METRIC_LABEL[a.metric]}\n")

    wt = residue_table(load(WT_PRIMARY, d, n), a.metric)
    mu = residue_table(load(mut_artifact, d, n), a.metric)
    assert (wt.resid.to_numpy() == mu.resid.to_numpy()).all(), "residue mismatch"

    shift = np.linalg.norm(wt[XYZ].to_numpy() - mu[XYZ].to_numpy(), axis=1)

    # Sanity check: the substituted residue should differ in resname.
    pos = int("".join(ch for ch in a.mutant[1:-1] if ch.isdigit()))
    idx = wt.index[wt.resid == pos - 267]
    if len(idx):
        i = idx[0]
        print(f"substitution check: construct resid {pos-267} (= POLE {pos})  "
              f"WT={wt.resname[i]}  mutant={mu.resname[i]}\n")

    tab = pd.DataFrame({
        "resid": wt.resid, "pole_resid": wt.resid + 267,
        "wt_resname": wt.resname, "mut_resname": mu.resname,
        f"wt_{a.metric}": wt.disp, f"mut_{a.metric}": mu.disp,
        "mean_shift": shift})
    tab.to_csv(HERE / f"per_residue_{a.metric}_{a.mutant}.csv", index=False)

    def describe(name, v):
        return dict(series=name, mean=v.mean(), median=np.median(v),
                    p90=np.quantile(v, .9), max=v.max())
    summ = pd.DataFrame([describe(f"WT {a.metric}", wt.disp.to_numpy()),
                         describe(f"{a.mutant} {a.metric}", mu.disp.to_numpy()),
                         describe("mean shift WT->mut", shift)])
    summ.to_csv(HERE / f"summary_{a.metric}_{a.mutant}.csv", index=False)
    print(summ.round(4).to_string(index=False))
    print(f"\ndelta of means (mutant - WT): {mu.disp.mean() - wt.disp.mean():+.4f} A")
    print(f"RMSD from the mean-shift curve: {np.sqrt((shift**2).mean()):.4f} A")

    # --- histogram ---------------------------------------------------------
    allv = np.concatenate([wt.disp.to_numpy(), mu.disp.to_numpy(), shift])
    lo, hi = 0.0, float(np.percentile(allv, 99.5))
    edges = np.linspace(lo, hi, a.bins + 1)
    ctr = (edges[:-1] + edges[1:]) / 2
    w = (edges[1] - edges[0]) * .92

    def frac(v):
        h, _ = np.histogram(np.clip(v, lo, hi), bins=edges)
        return h / h.sum()

    lbl = METRIC_LABEL[a.metric].split(" (")[0]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(ctr, frac(wt.disp.to_numpy()), width=w, color="#2980b9", alpha=.55,
           label=f"WT {lbl}  (n={n or depths[WT_PRIMARY]} structures)")
    ax.bar(ctr, frac(mu.disp.to_numpy()), width=w, color="#c0392b", alpha=.55,
           label=f"{a.mutant} {lbl}  (n={n or depths[mut_artifact]})")
    ax.step(ctr, frac(shift), where="mid", color="#2c3e50", lw=1.6,
            label=r"$\|$mean WT $-$ mean " + a.mutant + r"$\|$")
    rmsd = float(np.sqrt((shift ** 2).mean()))
    pct = 100 * (mu.disp.mean() - wt.disp.mean()) / wt.disp.mean()
    ax.text(0.975, 0.60,
            "mean structures superimpose\n"
            rf"to {rmsd:.2f} $\mathrm{{\AA}}$ RMSD" "\n\n"
            f"ensemble width {pct:+.0f}%\n"
            rf"({wt.disp.mean():.3f} $\to$ {mu.disp.mean():.3f} $\mathrm{{\AA}}$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="0.75", lw=.8))
    ax.set_xlabel(rf"per-residue {lbl} / displacement ($\mathrm{{\AA}}$)")
    ax.set_ylabel("fraction of residues")
    ax.set_title(f"{a.mutant} barely moves the predicted structure "
                 f"but broadens the ensemble\n"
                 f"POLE, {len(wt)} residues, matched ensemble depth", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    stem = f"POLE_{a.metric}_hist_WT_vs_{a.mutant}"
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=200)
    print(f"\nwrote {HERE}/{stem}.png")

    if a.write_pdb:
        off = 267 if a.pdb_numbering == "pole" else 0
        p1 = write_pdb(wt, HERE / "POLE_WT_mean.pdb",
                       f"POLE WT ensemble mean, n={n or depths[WT_PRIMARY]} structures", off)
        p2 = write_pdb(mu, HERE / f"POLE_{a.mutant}_mean.pdb",
                       f"POLE {a.mutant} ensemble mean, n={n or depths[mut_artifact]} structures", off)
        for p in (p1, p2):
            print(f"wrote {p}")

    if a.write_medoid:
        import shutil
        index = pd.read_csv(CACHE.parent / "POLE_structure_file_index.csv")
        for art, tab, tag in ((WT_PRIMARY, wt, "WT"), (mut_artifact, mu, a.mutant)):
            s_art = load(art, d, n)
            mi, r = medoid(s_art, tab)
            row = index[(index.artifact == art) & (index.structure_index == mi)]
            if row.empty:
                print(f"  {tag}: medoid index {mi} not in structure file index")
                continue
            src = Path(row.structure_path.iloc[0])
            dst = HERE / f"POLE_{tag}_medoid{src.suffix}"
            if not src.exists():
                print(f"  {tag}: medoid source not reachable: {src}")
                continue
            shutil.copy2(src, dst)
            print(f"wrote {dst}\n      medoid = structure_index {mi}, "
                  f"CA RMSD to ensemble mean {r:.3f} A\n      source {src}")


if __name__ == "__main__":
    main()
