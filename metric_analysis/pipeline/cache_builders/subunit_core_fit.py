"""Coordinate cache superposed on a WT-defined rigid core (2026-10-15).

Why: the cache builders call USalign with its defaults (monomer mode, ``-ter 2``), which
superposes on the FIRST chain only. For MutS-alpha that is MSH2 (chain A), so MSH6
(chain B) sits in MSH2's frame. For the single-chain caches the fit is on the mutated
chain, but on whatever USalign's TM-score weighting picks per structure, so the frame
is not the same set of residues across variants. This derives a cache from an existing
one, with no drive and no re-alignment, because a superposition is a rigid motion:
every cached structure is re-fitted (Kabsch, least squares on C-alpha) onto one fixed
target, using only the dataset's CORE residues -- the same residues for WT and every
variant, so a mutation cannot move the frame (standard MD practice).

Core = the largest RIGID group of confident residues, defined once from the WT ensemble:

1. candidates: WT mean pLDDT >= ``plddt_min`` (70) and present in >= 90% of the WT
   structures (FOXA1's cache has residue gaps; elsewhere every residue is present);
2. rigidity: for every candidate pair, the SD across the WT ensemble of their C-alpha
   distance. A rigid body keeps every internal distance, so its pairs have SD ~0;
3. average-linkage clustering on that SD, cut at ``rigid_sd`` (A); the largest cluster
   is the core.

pLDDT alone is not enough for multi-domain proteins: in MSH6 every ~200-residue block is
rigid (0.2-0.8 A) but the blocks move 15-20 A relative to each other, and a fit on all
confident residues averages over that motion (median fit RMSD 11 A, rejected). The
core's fit RMSD is reported, and a core whose median fit RMSD over all kept structures
is above ``max_fit_rmsd`` (2 A) is refused: the report is written with a ``_REJECTED``
suffix and no cache.

A structure holding under half of the core (only FOXA1's damaged cache has any) cannot be
placed in the frame and is dropped, with a count in the report.

Target: the mean core of the WT ensemble after three rounds of fitting the WT
structures onto it (generalised Procrustes, starting from the first WT structure).
RMSF/RMSD are then reported for EVERY residue in this frame, core or not.

Ensemble = exactly the pLDDT ensemble. Each cached structure is mapped to its AF3
``seed-N_sample-M`` through the structure file index, and only the samples that are
columns of the artifact's pLDDT CSV are kept (POLE's 268-471 construct has 160 samples
cached but 125 in its pLDDT CSV). Structures the index does not cover (artifacts added
later by stream_align) are kept unless they are exact coordinate copies of a kept one:
AF3's job-level top model is a copy of one seed-sample, and MutS-alpha G162R carries 37
copied rows. Only the configured WT and the artifacts with a pLDDT CSV in the dataset's
scope are kept (``token`` restricts a subunit to its own files).

Output keeps the pipeline schema (artifact, structure_index, chain, resid, icode,
resname, x/y/z_aligned) plus ``sample`` and ``core`` (bool), as
``<cache_dir>/<prefix>_rmsf_aligned_ca_coordinate_cache.parquet`` with
``<prefix>_core_residues.csv`` and ``<prefix>_core_fit_report.json`` next to it.
``<prefix>_core_fit_per_structure.csv`` keeps every structure's fit RMSD.

``--frame usalign`` writes the SAME structures (same pLDDT-sample selection, same copies
and thin structures dropped) with their original USalign coordinates, to
``ensemble_aware/cache_usalign`` with ``<prefix>_usalign_frame_report.json``: the control
that isolates the superposition when old and new sweeps are compared.

    python -m metric_analysis.pipeline.cache_builders.subunit_core_fit --dataset MSH6
    python -m metric_analysis.pipeline.cache_builders.subunit_core_fit --dataset all
    python -m metric_analysis.pipeline.cache_builders.subunit_core_fit --dataset MSH6 --probe
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from metric_analysis.pipeline.labels import artifact_key, is_wt_artifact
from metric_analysis.pipeline.paths import REPO_ROOT

SAMPLE = re.compile(r"seed-\d+_sample-\d+")
SM = REPO_ROOT / "metric_analysis/structure_metrics"
PL = REPO_ROOT / "PLDDTs"
OUT_SUBDIR = "ensemble_aware/cache_corefit"
FRAME_SUBDIR = {"core_fit": OUT_SUBDIR, "usalign": "ensemble_aware/cache_usalign"}
REPORT_NAME = {"core_fit": "core_fit_report", "usalign": "usalign_frame_report"}

# source: the parent cache's prefix (its cache is SM/<source>/ensemble_aware/cache);
# chain: the chain kept and fitted; rows: that chain's rows in the parent's pLDDT CSVs;
# plddt: the pLDDT folder; token: only CSVs with _<token>_ in the name (subunits);
# wt: the WT artifact in the source cache; wt_csv: its pLDDT CSV (by name, because the
# MSH2 WT CSV kept the parent's name).
DATASETS = {
    "MSH6": dict(source="MutS_alpha_MSH6_GT", chain="B", rows=(934, 2294), plddt="MutS_alpha_MSH6_GT", token="MSH6",
                 wt="MutS_alpha_MSH6_GT_WT_data", wt_csv="MutS_alpha_MSH6_GT_WT_plddt_values.csv"),
    "MSH2": dict(source="MutS_alpha", chain="A", rows=(0, 934), plddt="MutS_alpha", token="MSH2",
                 wt="MutS_alpha_MSH2_WT_vartmpl_data", wt_csv="MutS_alpha_WT_plddt_values.csv"),
    "POLE": dict(source="POLE", chain="A", plddt="POLE", wt="POLE_268-471_WT", wt_csv="POLE_268-471_WT_plddt_values.csv"),
    "PTEN": dict(source="PTEN", chain="A", plddt="PTEN", wt="PTEN_WT_data", wt_csv="PTEN_WT_plddt_values.csv"),
    "VHL": dict(source="VHL", chain="A", plddt="VHL", wt="VHL_WT_data", wt_csv="VHL_WT_plddt_values.csv"),
    "PIK3CA": dict(source="PIK3CA", chain="A", plddt="PIK3CA", wt="PIK3CA_WT_vartmpl_data",
                   wt_csv="PIK3CA_WT_vartmpl_plddt_values.csv"),
}


def kabsch(P, Q):
    """Rotation R and translation t minimising |(P @ R + t) - Q| (rows are points)."""
    pc_, qc = P.mean(0), Q.mean(0)
    H = (P - pc_).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return R, qc - pc_ @ R


def _csv_samples(path):
    cols = pd.read_csv(path, nrows=0).columns
    return [c for c in cols if SAMPLE.fullmatch(c)]


def plddt_scope(spec):
    """artifact_key -> pLDDT CSV for the dataset's variants (WT handled separately)."""
    out = {}
    for p in sorted((PL / spec["plddt"]).glob("*_plddt_values.csv")):
        stem = p.name[: -len("_plddt_values.csv")]
        if spec.get("token") and f"_{spec['token']}_" not in stem:
            continue
        if is_wt_artifact(stem):
            continue
        out[artifact_key(stem)] = p
    return out


def wt_core_candidates(spec, L):
    f = pd.read_csv(PL / spec["plddt"] / spec["wt_csv"], index_col=0)
    m = f[[c for c in f.columns if SAMPLE.fullmatch(c)]].to_numpy(float)
    r0, r1 = spec.get("rows") or (0, len(m))
    m = m[r0:r1].mean(1)
    if len(m) != L:
        raise SystemExit(f"WT pLDDT rows {r0}:{r1} give {len(m)} residues, the cache chain has {L}")
    return m


def load(spec):
    """Coordinates of the kept artifacts as X (structures, L, 3) with NaN for missing residues."""
    src = SM / spec["source"] / "ensemble_aware/cache"
    cache = src / f"{spec['source']}_rmsf_aligned_ca_coordinate_cache.parquet"
    dset = ds.dataset(cache)
    names = pc.unique(dset.to_table(columns=["artifact"])["artifact"]).to_pylist()
    scope = plddt_scope(spec)
    by_key = {}
    for a in names:
        by_key.setdefault(artifact_key(a), []).append(a)
    if spec["wt"] not in names:
        raise SystemExit(f"WT artifact {spec['wt']} is not in {cache}")
    keep = {spec["wt"]: PL / spec["plddt"] / spec["wt_csv"]}
    for k, csv in scope.items():
        hits = by_key.get(k, [])
        if len(hits) > 1:
            raise SystemExit(f"{k}: several cache artifacts {hits}")
        if hits:
            keep[hits[0]] = csv
    no_coords = sorted(csv.name for k, csv in scope.items() if k not in by_key)
    not_in_scope = sorted(a for a in names if a not in keep)

    cols = ["artifact", "structure_index", "chain", "resid", "icode", "resname", "x_aligned", "y_aligned", "z_aligned"]
    t = dset.to_table(columns=cols, filter=(pc.field("chain") == spec["chain"]) & pc.field("artifact").isin(list(keep)))
    t = t.to_pandas(strings_to_categorical=True)
    t = t.sort_values(["artifact", "structure_index", "resid"], kind="stable").reset_index(drop=True)
    L = int(t.resid.max())
    if int(t.resid.min()) < 1:
        raise SystemExit("residue numbers below 1")
    # t is sorted, so first-appearance group order is the order of `keys`
    pos = t.groupby(["artifact", "structure_index"], sort=False, observed=True).ngroup().to_numpy()
    first = np.r_[True, pos[1:] != pos[:-1]]
    art_names = np.asarray(t.artifact.cat.categories.astype(str))
    art_codes = t.artifact.cat.codes.to_numpy()
    keys = pd.DataFrame({"artifact": art_names[art_codes[first]],
                         "structure_index": t.structure_index.to_numpy()[first]})
    X = np.full((len(keys), L, 3), np.nan)
    X[pos, t.resid.to_numpy() - 1] = t[["x_aligned", "y_aligned", "z_aligned"]].to_numpy(float)
    # residue identity per (artifact, resid): a variant's resname differs at one site
    arts = sorted(keys.artifact.unique())
    a_of = {a: i for i, a in enumerate(arts)}
    ident = {}
    for col in ("icode", "resname"):
        cat = t[col].astype("category")
        codes = np.full((len(arts), L), -1, dtype=np.int32)
        codes[np.array([a_of.get(a, -1) for a in art_names])[art_codes], t.resid.to_numpy() - 1] = cat.cat.codes.to_numpy()
        ident[col] = (np.asarray(cat.cat.categories.astype(str)), codes)
    ident["artifact_index"] = a_of
    del t

    # seed-sample tag per cached structure, from the structure file index
    idx_path = src / f"{spec['source']}_structure_file_index.csv"
    keys["sample"] = None
    if idx_path.exists():
        idx = pd.read_csv(idx_path)
        pcol = [c for c in ("structure_member", "structure_path", "structure_file") if c in idx.columns]
        idx["sample"] = idx[pcol].astype(str).agg(" ".join, axis=1).str.extract(r"(seed-\d+_sample-\d+)")[0]
        idx["tag"] = np.where(idx["sample"].isna(), "top", idx["sample"])
        tag = dict(zip(zip(idx.artifact.astype(str), idx.structure_index.astype(int)), idx.tag))
        keys["sample"] = [tag.get((a, int(i))) for a, i in zip(keys.artifact, keys.structure_index)]
    return dict(X=X, keys=keys, L=L, ident=ident, keep=keep, cache=cache, index=idx_path,
                no_coords=no_coords, not_in_scope=not_in_scope)


def select_structures(d):
    """Keep exactly each artifact's pLDDT samples; drop index-less exact copies."""
    X, keys = d["X"], d["keys"]
    keep = np.zeros(len(keys), bool)
    reason = np.array([""] * len(keys), dtype=object)
    per = []
    for a, idx in keys.groupby("artifact").groups.items():
        idx = np.asarray(sorted(idx, key=lambda i: keys.structure_index[i]))
        want = set(_csv_samples(d["keep"][a]))
        tags = keys["sample"].to_numpy()[idx]
        seen = {}
        for i, tg in zip(idx, tags):
            if tg == "top":
                reason[i] = "top-level model"
                continue
            if tg is not None and tg not in want:
                reason[i] = "sample not in pLDDT CSV"
                continue
            h = np.nan_to_num(X[i], nan=1e9).tobytes()
            if h in seen:
                reason[i] = "exact copy"
                continue
            seen[h] = i
            keep[i] = True
        kept_tags = [t for t in tags[keep[idx]] if t is not None]
        per.append({"artifact": a, "cached": len(idx), "kept": int(keep[idx].sum()), "plddt_samples": len(want),
                    "tagged_kept": len(kept_tags), "missing_samples": len(want - set(kept_tags)) if kept_tags else None})
    return keep, reason, pd.DataFrame(per)


def rigid_core(W, cand, rigid_sd, method="average"):
    """Largest group of candidate residues whose pairwise C-alpha distances are stable in W."""
    ci = np.flatnonzero(cand)
    s1 = np.zeros((len(ci), len(ci)))
    s2 = np.zeros_like(s1)
    n = np.zeros_like(s1)
    for w in W[:, ci]:
        D = np.sqrt(((w[:, None, :] - w[None, :, :]) ** 2).sum(-1))
        ok = np.isfinite(D)
        D = np.where(ok, D, 0.0)
        s1 += D
        s2 += D * D
        n += ok
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    np.fill_diagonal(sd, 0.0)
    Z = linkage(squareform(sd, checks=False), method=method)
    lab = fcluster(Z, t=rigid_sd, criterion="distance")
    sizes = np.bincount(lab)
    core = np.zeros(len(cand), bool)
    core[ci[lab == sizes.argmax()]] = True
    clusters = sorted(sizes[sizes > 0].tolist(), reverse=True)
    return core, clusters, float(np.median(sd[np.ix_(lab == sizes.argmax(), lab == sizes.argmax())]))


def fit(X, core, wt_idx, rounds=3):
    ci = np.flatnonzero(core)

    def onto(P, Q):   # Kabsch on the residues present in both
        ok = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
        return kabsch(P[ok], Q[ok])

    # generalised Procrustes on the WT ensemble, from its most complete structure
    target = X[wt_idx[np.isfinite(X[wt_idx][:, ci]).all(2).sum(1).argmax()], ci]
    for _ in range(rounds):
        target = np.nanmean([X[i, ci] @ R + tt for i in wt_idx for R, tt in [onto(X[i, ci], target)]], 0)
    Y = np.full_like(X, np.nan)
    rmsd = np.full(len(X), np.nan)
    n_core = np.zeros(len(X), int)
    for i in range(len(X)):
        ok = np.isfinite(X[i, ci]).all(1)
        n_core[i] = ok.sum()
        if n_core[i] < 3:
            continue
        R, tt = kabsch(X[i, ci][ok], target[ok])
        Y[i] = X[i] @ R + tt
        rmsd[i] = np.sqrt(((Y[i, ci][ok] - target[ok]) ** 2).sum(1).mean())
    return Y, rmsd, n_core, target


def build(dataset, plddt_min=70.0, rigid_sd=1.0, max_fit_rmsd=2.0, out_dir=None, probe=None, min_presence=0.9,
          min_core_presence=0.5, frame="core_fit"):
    spec = DATASETS[dataset]
    d = load(spec)
    X, keys, L = d["X"], d["keys"], d["L"]
    kept, reason, per_artifact = select_structures(d)
    wt_idx = np.flatnonzero((keys.artifact == spec["wt"]).to_numpy() & kept)
    plddt = wt_core_candidates(spec, L)
    # present in (nearly) every WT structure: FOXA1's cache has residue gaps
    present = np.isfinite(X[wt_idx]).all(axis=2).mean(0) >= min_presence
    cand = (plddt >= plddt_min) & present

    if probe:
        rows = []
        for cut in probe:
            core, clusters, med_sd = rigid_core(X[wt_idx], cand, cut)
            ok = np.isfinite(X[:, core]).all(2).sum(1) >= min_core_presence * core.sum()
            _, r, _, _ = fit(X, core, wt_idx[ok[wt_idx]])
            r = np.where(ok, r, np.nan)
            rows.append({"rigid_sd": cut, "core": int(core.sum()), "clusters_top5": clusters[:5],
                         "core_pair_sd_median": round(med_sd, 3),
                         "fit_rmsd_wt_median": round(float(np.nanmedian(r[wt_idx])), 3),
                         "fit_rmsd_all_median": round(float(np.nanmedian(r[kept])), 3),
                         "fit_rmsd_all_p95": round(float(np.nanpercentile(r[kept], 95)), 3),
                         "structures_core_below_min": int((kept & np.isnan(r)).sum())})
            print(dataset, rows[-1], flush=True)
        return rows

    core, clusters, med_sd = rigid_core(X[wt_idx], cand, rigid_sd)
    # a structure holding under half the core cannot be placed in the frame (FOXA1: the
    # USalign builder kept only the residues it aligned, down to 122 of 472)
    thin = kept & (np.isfinite(X[:, core]).all(2).sum(1) < min_core_presence * core.sum())
    kept &= ~thin
    reason[thin] = f"core < {min_core_presence:.0%} present"
    wt_idx = np.flatnonzero((keys.artifact == spec["wt"]).to_numpy() & kept)
    Y, fit_rmsd, n_core, _ = fit(X, core, wt_idx)
    if frame == "usalign":
        Y = X   # the control: same structures, the source cache's own superposition

    out_dir = Path(out_dir or SM / dataset / FRAME_SUBDIR[frame])
    out_dir.mkdir(parents=True, exist_ok=True)
    k = keys[kept].assign(fit_rmsd=fit_rmsd[kept], n_core_used=n_core[kept])
    per_artifact_fit = k.groupby("artifact").fit_rmsd.median()
    variant_med = per_artifact_fit.drop(spec["wt"])

    # rigidity: every intra-structure distance is preserved by the transform
    rng = np.random.default_rng(0)
    pick = rng.choice(np.flatnonzero(kept), min(50, int(kept.sum())), replace=False)
    pairs = rng.choice(L, size=(2000, 2))
    dist = lambda A: np.linalg.norm(A[:, pairs[:, 0]] - A[:, pairs[:, 1]], axis=-1)
    rigid_err = float(np.nanmax(np.abs(dist(X[pick]) - dist(Y[pick]))))

    counts = k.groupby("artifact").size()
    per_artifact["kept"] = per_artifact.artifact.map(counts).fillna(0).astype(int)
    rep = {
        "dataset": dataset, "frame": frame,
        "coordinates": ("re-fitted on the core" if frame == "core_fit" else
                        "the source cache's own USalign superposition; the core and fit RMSD below are "
                        "the core-fit's, reported only to show which structures were kept"),
        "source_cache": str(d["cache"]), "structure_index_file": str(d["index"]),
        "chain": spec["chain"], "wt_artifact": spec["wt"], "wt_plddt_csv": str(PL / spec["plddt"] / spec["wt_csv"]),
        "plddt_rows": list(spec.get("rows") or (0, L)), "residues": L,
        "core_rule": {"plddt_min": plddt_min, "min_wt_presence": min_presence, "min_core_presence": min_core_presence, "rigid_sd_A": rigid_sd, "linkage": "average",
                      "max_fit_rmsd_A": max_fit_rmsd},
        "candidates_plddt": int(cand.sum()), "core_residues": int(core.sum()),
        "core_fraction_of_chain": round(float(core.mean()), 3),
        "candidate_clusters_top5": clusters[:5], "core_pair_distance_sd_median_A": round(med_sd, 3),
        "artifacts": int(counts.size), "variants": int(counts.size - 1), "structures_kept": int(kept.sum()),
        "dropped": pd.Series(reason[~kept]).value_counts().to_dict(),
        "structures_per_artifact": {str(a): int(b) for a, b in counts.value_counts().items()},
        "artifacts_not_matching_plddt_count": per_artifact.loc[per_artifact.kept != per_artifact.plddt_samples,
                                                                ["artifact", "kept", "plddt_samples"]].to_dict("records"),
        "artifacts_missing_a_plddt_sample": per_artifact.loc[per_artifact.missing_samples.fillna(0) > 0,
                                                             "artifact"].tolist(),
        "plddt_csvs_without_coordinates": d["no_coords"], "cache_artifacts_not_in_scope": d["not_in_scope"],
        "fit_rmsd_A": {
            "all": {q: round(float(v), 3) for q, v in zip(["median", "p95", "max"],
                                                          np.percentile(fit_rmsd[kept], [50, 95, 100]))},
            "wt": {q: round(float(v), 3) for q, v in zip(["median", "p95", "max"],
                                                         np.percentile(fit_rmsd[wt_idx], [50, 95, 100]))},
            "variant_medians": {q: round(float(v), 3) for q, v in zip(["min", "median", "max"],
                                                                       np.percentile(variant_med, [0, 50, 100]))},
        },
        "worst_fits": k.nlargest(5, "fit_rmsd")[["artifact", "structure_index", "sample", "fit_rmsd"]]
                       .round(3).to_dict("records"),
        "structures_with_core_gaps": int((k.n_core_used < core.sum()).sum()),
        "rigidity_max_distance_change_A": rigid_err,
    }
    core_table = pd.DataFrame({"resid": np.arange(1, L + 1), "wt_mean_plddt": plddt.round(2),
                               "candidate": cand, "core": core})
    accepted = rep["fit_rmsd_A"]["all"]["median"] <= max_fit_rmsd
    rep["accepted"] = bool(accepted)
    stem = f"{dataset}_{REPORT_NAME[frame]}" + ("" if accepted else "_REJECTED")
    core_table.to_csv(out_dir / f"{dataset}_core_residues.csv", index=False)
    if not accepted:
        (out_dir / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=str))
        raise SystemExit(f"{dataset}: median core fit RMSD {rep['fit_rmsd_A']['all']['median']} A > {max_fit_rmsd} A; "
                         f"core refused, no cache written ({out_dir / stem}.json)")

    # long table, in the pipeline schema, one row per present residue of a kept structure
    sel = np.flatnonzero(kept)
    Yk = Y[sel]
    s_idx, r_idx = np.nonzero(np.isfinite(Yk).all(-1))
    ka = keys.iloc[sel].reset_index(drop=True)
    ident = d["ident"]
    arts = sorted(ident["artifact_index"], key=ident["artifact_index"].get)
    ai = ka.artifact.map(ident["artifact_index"]).to_numpy()[s_idx]
    sample = pd.Categorical(ka["sample"].fillna("").to_numpy())
    cat = lambda codes, names: pd.Categorical.from_codes(codes, categories=list(names))
    out = pd.DataFrame({
        "artifact": cat(ai, arts), "structure_index": ka.structure_index.to_numpy()[s_idx].astype("int64"),
        "sample": cat(sample.codes[s_idx], sample.categories), "chain": cat(np.zeros(len(s_idx), int), [spec["chain"]]),
        "resid": (r_idx + 1).astype("int64"),
        **{col: cat(ident[col][1][ai, r_idx], ident[col][0]) for col in ("icode", "resname")},
        "x_aligned": Yk[s_idx, r_idx, 0], "y_aligned": Yk[s_idx, r_idx, 1], "z_aligned": Yk[s_idx, r_idx, 2],
        "core": core[r_idx],
    })
    del Yk
    path = out_dir / f"{dataset}_rmsf_aligned_ca_coordinate_cache.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), tmp, row_group_size=2_000_000)
    tmp.replace(path)
    k.to_csv(out_dir / f"{dataset}_core_fit_per_structure.csv", index=False)
    rep.update(output=str(path), rows=int(len(out)))
    (out_dir / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=str))
    for old in out_dir.glob(f"{dataset}_{REPORT_NAME[frame]}_REJECTED*.json"):
        old.unlink()
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help=f"one of {sorted(DATASETS)}, comma-separated, or all")
    ap.add_argument("--plddt-min", type=float, default=70.0)
    ap.add_argument("--rigid-sd", type=float, default=1.0, help="average-linkage cut on pair-distance SD (A)")
    ap.add_argument("--max-fit-rmsd", type=float, default=2.0)
    ap.add_argument("--out-dir")
    ap.add_argument("--frame", choices=sorted(FRAME_SUBDIR), default="core_fit",
                    help="usalign: same structures, original coordinates (comparison control)")
    ap.add_argument("--probe", help="comma-separated rigid_sd values: report core size and fit RMSD, write nothing")
    a = ap.parse_args(argv)
    names = sorted(DATASETS) if a.dataset == "all" else a.dataset.split(",")
    for name in names:
        if a.probe:
            build(name, a.plddt_min, probe=[float(x) for x in a.probe.split(",")])
        else:
            rep = build(name, a.plddt_min, a.rigid_sd, a.max_fit_rmsd, a.out_dir, frame=a.frame)
            print(json.dumps({k: rep[k] for k in ("dataset", "core_residues", "candidates_plddt", "structures_kept",
                                                  "structures_per_artifact", "fit_rmsd_A", "accepted")}, default=str),
                  flush=True)


if __name__ == "__main__":
    main()
