"""Validate the core-fit coordinate caches against their sources and the pLDDT cohort (2026-10-15).

For each dataset built by metric_analysis.pipeline.cache_builders.subunit_core_fit:

- rigidity: 40 random structures, every intra-structure C-alpha distance in the core-fit
  cache vs the USalign source cache (a superposition must not change any of them);
- ensembles: structures per artifact vs the columns of its pLDDT CSV, and the kept
  ``sample`` tags vs those columns; no exact coordinate copies left;
- cohort: WT + variants vs the pLDDT CSVs in scope and vs the strict canonical manifest
  (included rows), including the manifest's WT for each variant;
- fit RMSD distribution, from <dataset>_core_fit_per_structure.csv.

Writes results/validation.json and results/validation_summary.csv.

    PYTHONPATH="." python experiments/coord_core_fit/validate.py [--dataset POLE,PTEN,...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from metric_analysis.pipeline.cache_builders.subunit_core_fit import DATASETS, OUT_SUBDIR, PL, SM, SAMPLE, plddt_scope
from metric_analysis.pipeline.labels import artifact_key

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MANIFEST = SM.parent / "manifests/strict_canonical_manifest.csv"


def coords(path, chain, pairs):
    f = None
    for a, i in pairs:
        e = (pc.field("artifact") == a) & (pc.field("structure_index") == int(i))
        f = e if f is None else (f | e)
    t = ds.dataset(path).to_table(columns=["artifact", "structure_index", "chain", "resid", "x_aligned", "y_aligned",
                                           "z_aligned"], filter=f & (pc.field("chain") == chain)).to_pandas()
    t["artifact"] = t.artifact.astype(str)
    return {k: g.set_index("resid")[["x_aligned", "y_aligned", "z_aligned"]] for k, g in t.groupby(["artifact", "structure_index"])}


def check(name):
    spec = DATASETS[name]
    cdir = SM / name / OUT_SUBDIR
    new_path = cdir / f"{name}_rmsf_aligned_ca_coordinate_cache.parquet"
    src_path = SM / spec["source"] / "ensemble_aware/cache" / f"{spec['source']}_rmsf_aligned_ca_coordinate_cache.parquet"
    rep = json.loads((cdir / f"{name}_core_fit_report.json").read_text())
    t = ds.dataset(new_path).to_table(columns=["artifact", "structure_index", "sample", "chain", "resid"]).to_pandas()
    t["artifact"] = t.artifact.astype(str)
    t["sample"] = t["sample"].astype(str)
    st = t.drop_duplicates(["artifact", "structure_index"])
    out = {"dataset": name, "rows": len(t), "chains": sorted(t.chain.astype(str).unique())}

    # rigidity vs the source
    rng = np.random.default_rng(1)
    pick = st.sample(40, random_state=1)[["artifact", "structure_index"]].itertuples(index=False)
    pick = list(pick)
    a, b = coords(new_path, spec["chain"], pick), coords(src_path, spec["chain"], pick)
    worst = 0.0
    for k in a:
        A, B = a[k], b[k].loc[a[k].index]
        if len(A) != len(b[k]) and not np.isnan(b[k].to_numpy()).any():
            pass
        P, Q = A.to_numpy(), B.to_numpy()
        i, j = rng.integers(0, len(P), (2, 3000))
        worst = max(worst, float(np.abs(np.linalg.norm(P[i] - P[j], axis=1) - np.linalg.norm(Q[i] - Q[j], axis=1)).max()))
    out["rigidity_max_distance_change_A"] = worst
    out["rigidity_structures_checked"] = len(a)

    # ensembles vs pLDDT CSVs
    scope = plddt_scope(spec)
    csv_of = {artifact_key(k): v for k, v in scope.items()}
    csv_of[artifact_key(spec["wt"])] = PL / spec["plddt"] / spec["wt_csv"]
    rows = []
    for art, g in st.groupby("artifact"):
        csv = csv_of.get(artifact_key(art))
        cols = [c for c in pd.read_csv(csv, nrows=0).columns if SAMPLE.fullmatch(c)] if csv else []
        tags = set(g["sample"]) - {""}
        rows.append({"artifact": art, "structures": len(g), "plddt_samples": len(cols),
                     "tags_not_in_csv": len(tags - set(cols)), "untagged": int((g["sample"] == "").sum()),
                     "duplicate_tags": int(g["sample"][g["sample"] != ""].duplicated().sum())})
    per = pd.DataFrame(rows)
    out["artifacts"] = len(per)
    out["structures_per_artifact"] = {str(k): int(v) for k, v in per.structures.value_counts().items()}
    out["artifacts_equal_to_plddt"] = int((per.structures == per.plddt_samples).sum())
    out["artifacts_short_of_plddt"] = per.loc[per.structures != per.plddt_samples, ["artifact", "structures", "plddt_samples"]] \
        .sort_values("structures").head(10).to_dict("records")
    out["n_short_of_plddt"] = int((per.structures != per.plddt_samples).sum())
    out["tags_not_in_csv"] = int(per.tags_not_in_csv.sum())
    out["duplicate_tags"] = int(per.duplicate_tags.sum())
    out["untagged_structures"] = int(per.untagged.sum())
    out["exact_copies_left"] = None   # filled below from coordinates of a sample of artifacts

    # cohort: pLDDT CSVs in scope and the strict manifest
    cache_keys = {artifact_key(x) for x in per.artifact}
    out["wt_in_cache"] = artifact_key(spec["wt"]) in cache_keys
    out["plddt_variant_csvs"] = len(scope)
    out["plddt_variants_without_coordinates"] = sorted(set(scope) - cache_keys)
    m = pd.read_csv(MANIFEST)
    m = m[(m.protein == name) & (m.cohort_status == "included")]
    mk = m.source_file.map(lambda p: artifact_key(Path(p).name.replace("_plddt_values.csv", "")))
    wk = m.wt_source_file.map(lambda p: artifact_key(Path(p).name.replace("_plddt_values.csv", "")))
    out["manifest_included"] = int(len(m))
    out["manifest_labels"] = m.mutation_type.value_counts().to_dict()
    out["manifest_variants_without_coordinates"] = sorted(m.loc[~mk.isin(cache_keys), "mutation_code"])
    out["manifest_wt_csvs"] = sorted({Path(p).name for p in m.wt_source_file})
    out["manifest_wt_matches_cache_wt"] = bool((wk == artifact_key(Path(spec["wt_csv"]).name.replace("_plddt_values.csv", ""))).all())
    ms = per.set_index(per.artifact.map(artifact_key)).structures
    nm = m.n_mutant_structures.to_numpy()
    have = mk.isin(cache_keys).to_numpy()
    out["manifest_n_mutant_structures_equal"] = int((ms.reindex(mk[have]).to_numpy() == nm[have]).sum())
    out["manifest_n_with_coordinates"] = int(have.sum())

    # exact copies: hash every structure of 30 artifacts
    arts = list(per.artifact.sample(min(30, len(per)), random_state=2))
    tt = ds.dataset(new_path).to_table(columns=["artifact", "structure_index", "resid", "x_aligned", "y_aligned", "z_aligned"],
                                       filter=pc.field("artifact").isin(arts)).to_pandas()
    dup = 0
    for _, g in tt.groupby("artifact", observed=True):
        h = g.sort_values(["structure_index", "resid"]).groupby("structure_index")[["x_aligned", "y_aligned", "z_aligned"]] \
             .apply(lambda x: hash(x.to_numpy().round(6).tobytes()))
        dup += int(h.duplicated().sum())
    out["exact_copies_left"] = f"{dup} in {len(arts)} artifacts checked"

    # fit RMSD
    f = pd.read_csv(cdir / f"{name}_core_fit_per_structure.csv")
    q = lambda x: {p: round(float(np.percentile(x, p)), 3) for p in (50, 95, 99, 100)}
    out["fit_rmsd_A_all"] = q(f.fit_rmsd)
    out["fit_rmsd_A_wt"] = q(f.fit_rmsd[f.artifact == spec["wt"]])
    med = f.groupby("artifact").fit_rmsd.median()
    out["variants_median_fit_above_1A"] = med[med > 1].round(2).sort_values().to_dict()
    out["core"] = {k: rep[k] for k in ("candidates_plddt", "core_residues", "residues", "candidate_clusters_top5",
                                       "core_pair_distance_sd_median_A")}
    out["builder_dropped"] = rep["dropped"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=",".join(DATASETS),
                    help="comma list, as given to subunit_core_fit (default: every dataset it defines)")
    names = ap.parse_args().dataset.split(",")
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}")
    OUT.mkdir(exist_ok=True)
    res = []
    for name in names:
        r = check(name)
        res.append(r)
        print(json.dumps({k: r[k] for k in ("dataset", "rigidity_max_distance_change_A", "structures_per_artifact",
                                            "n_short_of_plddt", "tags_not_in_csv", "duplicate_tags", "exact_copies_left",
                                            "manifest_included", "manifest_variants_without_coordinates",
                                            "manifest_wt_csvs", "manifest_wt_matches_cache_wt",
                                            "manifest_n_mutant_structures_equal", "manifest_n_with_coordinates",
                                            "fit_rmsd_A_all")}, default=str), flush=True)
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    flat = pd.DataFrame([{
        "dataset": r["dataset"], "chain": ",".join(r["chains"]), "core": r["core"]["core_residues"],
        "candidates": r["core"]["candidates_plddt"], "residues": r["core"]["residues"],
        "artifacts": r["artifacts"], "manifest_included": r["manifest_included"],
        "manifest_without_coords": len(r["manifest_variants_without_coordinates"]),
        "n_short_of_plddt": r["n_short_of_plddt"], "rigidity_max_dA": r["rigidity_max_distance_change_A"],
        "fit_median": r["fit_rmsd_A_all"][50], "fit_p95": r["fit_rmsd_A_all"][95], "fit_wt_median": r["fit_rmsd_A_wt"][50],
        "variants_fit_above_1A": len(r["variants_median_fit_above_1A"]),
    } for r in res])
    flat.to_csv(OUT / "validation_summary.csv", index=False)
    print(flat.to_string(index=False))


if __name__ == "__main__":
    main()
