#!/usr/bin/env python
"""Supplementary Note S1 coordinate descriptors on the exact pLDDT KS-score draws.

Audit item 6 (experiments/manuscript_review/methods_si_audit_2026-10-19.md). SI l.290-292
says the coordinate descriptors used "the same subsampling sizes and seeds as the pLDDT
analyses". They did not: the core-fit coordinate sweep
(run_ensemble_sweep_parallel.py:503) draws from its own RNG stream. Here every
descriptor is recomputed on the SAME 90 WT + 90 mutant predictions the pLDDT KS-score
uses at N = 90 (DRIFT production draws, seeds 42-51, select_structure_columns +
source_draw_key), reconstructed with analyses/coordinates/s8_representatives
/n90_comparison.py (orders, load_matrix; imported, not modified).

Descriptors, WT rigid-core frame (core_fit caches), scored chain only:
  RMSD          RMSD between the mutant and WT ensemble-mean Ca structures
                (= run_ensemble_sweep_parallel.compute_rmsd_summary_vs_wt
                 rmsd_ensemble_mean_to_wt_mean)
  RMSF          mutant mean RMSF over residues, about its own mean (rmsf_mean)
  dRMSF         RMSF(mutant 90) - RMSF(WT 90) (delta_rmsf_mean)
  posKS         SI eq. si-positional-ks: distance of each Ca from the WT centre mu_r
                (mean over the FULL WT pool), per-residue KS mutant vs WT, mean over
                residues (= library/positional_ks.positional_ks, mode "distance")
  KS            pLDDT KS-score, read from the corrected caches
                (experiments/pole_wt_repair/cache, ebbbda31758d) and recomputed
                directly for every variant and seed.

Cohort 1,797: annotation cohort less POLE P356S (no coordinates) and MSH2 G162R
(eight structures without sample identities), as Note S9.

AUC = mean of per-seed AUCs, drivers high, never flipped; macro = equal-gene mean.
Per-gene CIs: S1's class-stratified bootstrap (2,000; same RNG seed and procedure as
figures/si/lib/coordinate_vs_plddt.auc_table). The KS + posKS combination
contrast uses the Methods paired bootstrap (simple_fusion_benchmark/annotation_tables
.bootstrap).

    PYTHONPATH="." python \\
        analyses/coordinates/s1_matched_draws/s1_matched_draws.py [--recompute]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent  # not HERE: figlib exports HERE
ROOT = THIS_DIR.parents[2]
REPR = ROOT / "analyses" / "coordinates" / "s8_representatives"
sys.path.insert(0, str(REPR))
sys.path.insert(0, str(ROOT / "figures" / "si" / "lib"))

from figlib import *  # noqa: F403,E402  (style, palette, save, record, provenance)
from scipy.stats import rankdata, spearmanr  # noqa: E402

import n90_comparison as n90  # noqa: E402  orders(), load_matrix(): production draws
from check_representatives import artifact_frames  # noqa: E402
from metric_analysis.pipeline.labels import artifact_key  # noqa: E402
from metric_analysis.pipeline.library.af3_seeds import within_seed_cap  # noqa: E402
from metric_analysis.pipeline.library.position_bootstrap import (  # noqa: E402
    clusters_of, describe, strata_of, stratified_draw, unstratified_draw)
from metric_analysis.pipeline.library.canonical_cohort import _existing_repo_path  # noqa: E402
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks  # noqa: E402
from metric_analysis.pipeline.library.positional_ks import positional_ks, reference_centre  # noqa: E402
from metric_analysis.pipeline.orchestration.run_ensemble_sweep_parallel import (  # noqa: E402
    _mean_coords_by_residue, compute_rmsd_summary_vs_wt, compute_rmsf_summary_from_coords)
from drift.protein_config import load_protein_configs  # noqa: E402

OUT = THIS_DIR / "results"
assert OUT.parent == Path(__file__).resolve().parent, OUT  # never write outside this folder
SEEDS = list(range(42, 52))
GENES = ANNOTATION_GENES  # POLE, PTEN, VHL, MSH2, MSH6, PIK3CA (S1 order)
N = 90
N_BOOT = 2000
BOOT_SEED = 20261016  # as in coordinate_vs_plddt.py
XYZ = ["x_aligned", "y_aligned", "z_aligned"]
EXPECTED = {"POLE": (1074, 38), "PTEN": (257, 248), "VHL": (124, 105),
            "MSH2": (108, 70), "MSH6": (143, 66), "PIK3CA": (91, 84)}
KS, POS, RMSD, DRMSF, RMSF = "pLDDT KS-score", "Positional KS", "RMSD", "ΔRMSF", "RMSF"
DESCRIPTORS = {KS: "ks", POS: "posks", RMSD: "rmsd", DRMSF: "delta_rmsf", RMSF: "rmsf_mut"}
CACHE = ROOT / "experiments/pole_wt_repair/cache"
FP = "ebbbda31758d"
COHORT = REPR / "annotation_cohort.csv"   # overridable (analyses/annotation/stages)
#: coordinate cache folder per gene (under structure_metrics/<gene>/ensemble_aware); the
#: published S1 frame is the WT rigid-core fit. analyses/coordinates/s1_frame_sensitivity sets
#: the US-align caches here to repeat S1 in that frame on the same draws.
COORD_FRAME: dict[str, str] = {}
EXCLUSIONS = REPR / "exclusions.csv"
OLD_S2B = ROOT / "experiments/pole_wt_repair/publication/si_figures/positional_ks_vs_plddt_ks_variants.csv.gz"
OLD_S2B_AUC = ROOT / "experiments/pole_wt_repair/publication/si_figures/positional_ks_vs_plddt_ks_auc.csv"
OLD_VIOLIN = ROOT / "figures/si/lib/results/coordinate_violins/coordinate_violins_variants.csv.gz"
OLD_VIOLIN_AUC = ROOT / "figures/si/lib/results/coordinate_violins/coordinate_violins_auc.csv"
# The old tables above are the 10-18 run's outputs, compared against when present (an
# analysis checkout); the release does not rebuild them. The bootstrap's variant order,
# which the published intervals depend on, is frozen in VARIANT_ORDER (taken from OLD_S2B).
VARIANT_ORDER = THIS_DIR / "variant_order.csv"
S9_AUC = REPR / "n90_results/auc_summary.csv"
SCORES = OUT / "variant_seed_scores.csv.gz"
HASHES: dict[str, str] = {}


def sha(path):
    path = Path(path)
    if str(path) not in HASHES:
        HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


# ── stage 1: descriptors on the production draws ─────────────────────────────

def cohort():
    c = pd.read_csv(COHORT)
    ex = pd.read_csv(EXCLUSIONS)
    sha(COHORT), sha(EXCLUSIONS)
    # Recorded paths name the main checkout; resolve them in this checkout, as the
    # canonical cohort does, so the source draw keys match the caches built here.
    for column in ("source_file", "wt_source_file"):
        c[column] = [str(_existing_repo_path(v, protein=p))
                     for v, p in zip(c[column], c.parent_dataset)]
    assert set(zip(ex.gene, ex.mutation)) == {("POLE", "P356S"), ("MSH2", "G162R")}
    drop = pd.MultiIndex.from_frame(c[["protein", "mutation_code"]]).isin(
        pd.MultiIndex.from_frame(ex[["gene", "mutation"]]))
    c = c[~drop].copy()
    got = {g: (int((c.protein == g).sum()), int(c[c.protein == g].label.sum())) for g in GENES}
    assert got == EXPECTED, got
    assert len(c) == 1797 and int(c.label.sum()) == 611
    return c


def tensor(frame, recovered=None):
    """(structures, residues, 3) float64 in structure_index order, plus sample tags."""
    frame = frame.copy()
    if recovered is not None:
        frame["sample"] = frame.structure_index.map(recovered)
    assert frame.chain.nunique() == 1
    assert not frame.duplicated(["structure_index", "resid"]).any()
    length = frame.resid.nunique()
    assert set(frame.resid) == set(range(1, length + 1))
    frame = frame.sort_values(["structure_index", "resid"])
    n = frame.structure_index.nunique()
    assert len(frame) == n * length, "every structure must carry every residue"
    xyz = frame[XYZ].to_numpy(np.float64).reshape(n, length, 3)
    assert np.isfinite(xyz).all()
    samples = frame.groupby("structure_index", sort=True)["sample"].first()
    return xyz, samples.tolist(), frame


def rows_of(samples, tags):
    pos = {s: i for i, s in enumerate(samples) if isinstance(s, str)}
    assert len(pos) == sum(isinstance(s, str) for s in samples), "duplicate sample tags"
    return np.array([pos[t] for t in tags])


def rmsf_mean(x):
    """Mean over residues of per-residue RMSF about the ensemble's own mean."""
    return float(np.sqrt(np.square(x - x.mean(0)).sum(2).mean(0)).mean())


def library_check(wt_frame, mut_frame, wt_rows_idx, mut_rows_idx, wt_si, mut_si, centre_df, got):
    """Recompute one (variant, seed) with the pipeline's own functions; return max |diff|."""
    w = wt_frame[wt_frame.structure_index.isin(wt_si[wt_rows_idx])].assign(icode="")
    m = mut_frame[mut_frame.structure_index.isin(mut_si[mut_rows_idx])].assign(icode="")
    assert w.structure_index.nunique() == N and m.structure_index.nunique() == N
    rmsd = compute_rmsd_summary_vs_wt(m, _mean_coords_by_residue(w))["rmsd_ensemble_mean_to_wt_mean"]
    rm, rw = (compute_rmsf_summary_from_coords(f)["rmsf_mean"] for f in (m, w))
    pk = positional_ks(w, m, centre_df, None, modes=("distance",))["distance"]
    return {"rmsd": abs(rmsd - got["rmsd"]), "rmsf_mut": abs(rm - got["rmsf_mut"]),
            "delta_rmsf": abs(rm - rw - got["delta_rmsf"]), "posks": abs(pk - got["posks"])}


def calculate():
    c = cohort()
    configs = load_protein_configs()
    reference = []
    for seed in SEEDS:
        path = CACHE / f"labeled_size90_seed{seed}_ratio_of_means_proteins{FP}_esfeat6.csv"
        sha(path)
        reference.append(pd.read_csv(path).assign(seed=seed))
    reference = pd.concat(reference).set_index(["gene", "mutation", "seed"])
    rows, audits, checks = [], [], []
    for gene in GENES:
        t0 = time.monotonic()
        records = c[c.protein.eq(gene)].to_dict("records")
        cfg = configs[gene]["d1"]
        source = records[0]["wt_source_file"]
        assert {r["wt_source_file"] for r in records} == {source}
        folder = ROOT / "metric_analysis/structure_metrics" / gene / "ensemble_aware/cache_corefit"
        report = json.loads((folder / f"{gene}_core_fit_report.json").read_text())
        sha(folder / f"{gene}_core_fit_report.json")
        if gene == "POLE":
            assert report["wt_artifact"] == "POLE_268-471_WT"
        index = pd.read_csv(folder / f"{gene}_core_fit_per_structure.csv")
        sha(folder / f"{gene}_core_fit_per_structure.csv")
        by_key = {artifact_key(a): a for a in index.artifact.unique()}
        wt_pl, wt_cols = n90.load_matrix(records[0], wt=True)
        wo, wt_tags = n90.orders(wt_cols, cfg, source, wt=True)
        mutants = {}
        for r in records:
            key = artifact_key(Path(r["source_file"]).name.removesuffix("_plddt_values.csv"))
            m, cols = n90.load_matrix(r)
            assert len(m) == len(wt_pl) == report["residues"]
            mo, mtags = n90.orders(cols, cfg, r["source_file"])
            mutants[by_key[key]] = (r, m, mo, mtags)
        assert len(mutants) == len(records)
        recovered = None
        if gene in ["MSH2", "PIK3CA"]:
            mp = ROOT / "analyses/seeds/seed_structure/results_rerun_20261018" / f"{gene}_recovered_seed_mapping.csv"
            sha(mp)
            mp = pd.read_csv(mp)
            recovered = dict(zip(mp.structure_index, mp["sample"]))
        coord_path = folder.parent / COORD_FRAME.get(gene, "cache_corefit") / f"{gene}_rmsf_aligned_ca_coordinate_cache.parquet"

        # WT first: full pool (centre) and the ten production WT draws
        (wt_name, wt_frame), = list(artifact_frames(coord_path, [report["wt_artifact"]]))
        wxyz, wsamples, wt_frame = tensor(wt_frame, recovered)
        assert wxyz.shape[1] == len(wt_pl)
        wt_si = np.array(sorted(wt_frame.structure_index.unique()))
        mu = wxyz.mean(0)  # SI eq. si-wt-centre: the whole WT pool, as the sweep's reference_centre
        centre_df = reference_centre(wt_frame.assign(icode=""))
        wt_draws = []
        for j in range(len(SEEDS)):
            idx = rows_of(wsamples, wt_tags[j])
            W = wxyz[idx]
            wt_draws.append({"idx": idx, "mean": W.mean(0), "rmsf": rmsf_mean(W),
                             "dist": np.linalg.norm(W - mu, axis=2).T})
        n_wt_mapped = sum(isinstance(s, str) for s in wsamples)

        seen, max_ks_err = set(), 0.0
        for artifact, frame in artifact_frames(coord_path, list(mutants)):
            r, m, mo, mtags = mutants[artifact]
            # The coordinate cache keeps all 160 structures of the eight 32-seed POLE
            # drivers; keep the AF3 seeds the pLDDT matrices use (1-25; 2026-10-20).
            frame = frame[within_seed_cap(frame["sample"], gene)]
            assert within_seed_cap([t for d in mtags for t in d], gene).all(), artifact
            mxyz, msamples, mframe = tensor(frame)
            assert mxyz.shape[1] == len(wt_pl)
            m_si = np.array(sorted(mframe.structure_index.unique()))
            seen.add(artifact)
            for j, seed in enumerate(SEEDS):
                wd = wt_draws[j]
                idx = rows_of(msamples, mtags[j])
                X = mxyz[idx]
                xm = X.mean(0)
                ref = reference.loc[(gene, r["mutation_code"], seed)]
                assert int(ref.label) == int(r["label"])
                ks_direct = residue_ks(wt_pl[:, wo[j]], m[:, mo[j]])
                err = abs(ks_direct - float(ref.plddt_ks_residue))
                max_ks_err = max(max_ks_err, err)
                assert err < 1e-12, (gene, r["mutation_code"], seed, err)
                rm = rmsf_mean(X)
                vals = {"rmsd": float(np.sqrt(np.square(xm - wd["mean"]).sum(1).mean())),
                        "rmsf_mut": rm, "rmsf_wt": wd["rmsf"], "delta_rmsf": rm - wd["rmsf"],
                        "posks": residue_ks(wd["dist"], np.linalg.norm(X - mu, axis=2).T)}
                rows.append({"gene": gene, "mutation": r["mutation_code"], "label": int(r["label"]),
                             "artifact": artifact, "seed": seed, "ks": float(ref.plddt_ks_residue),
                             "ks_direct": ks_direct, **vals,
                             "afm_score": float(ref.afm_score), "d1_score": float(ref.d1_score)})
                # pipeline functions on the first variant of each gene, every seed
                if len(seen) == 1:
                    diff = library_check(wt_frame, mframe, wd["idx"], idx, wt_si, m_si, centre_df, vals)
                    checks.append({"gene": gene, "artifact": artifact, "seed": seed, **diff})
        assert seen == set(mutants), f"{gene}: no coordinates for {set(mutants) - seen}"
        audits.append({"gene": gene, "variants": len(records), "residues": len(wt_pl),
                       "wt_artifact": report["wt_artifact"], "wt_pool_structures": len(wsamples),
                       "wt_pool_with_sample_tag": n_wt_mapped, "wt_plddt_columns": len(wt_cols),
                       "max_ks_cache_error": max_ks_err, "coordinate_cache": str(coord_path.relative_to(ROOT)),
                       "coordinate_bytes": coord_path.stat().st_size,
                       "coordinate_mtime_ns": coord_path.stat().st_mtime_ns})
        print(f"{gene}: {len(records)} variants x 10 draws, WT pool {len(wsamples)} "
              f"({n_wt_mapped} tagged); {time.monotonic() - t0:.1f}s", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(SCORES, index=False)
    pd.DataFrame(audits).to_csv(OUT / "reproduction_audit.csv", index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "library_checks.csv", index=False)
    print("  library checks, max |diff|:", chk[["rmsd", "rmsf_mut", "delta_rmsf", "posks"]].max().to_dict())
    assert (chk[["rmsd", "rmsf_mut", "delta_rmsf"]].to_numpy() < 1e-9).all()
    assert (chk.posks.to_numpy() < 1e-9).all()
    return scores


# ── stage 2: AUCs ─────────────────────────────────────────────────────────────

def auc(y, s):
    r = rankdata(s)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def have_old_tables():
    return all(p.exists() for p in (OLD_S2B, OLD_S2B_AUC, OLD_VIOLIN, OLD_VIOLIN_AUC))


def variant_order(scores):
    """Variants per gene in the old S2b table's order (so the S1 bootstrap reproduces
    the old KS intervals exactly where the cohort is unchanged), then any others.
    The order is frozen in VARIANT_ORDER (the KS rows of OLD_S2B, draw 42)."""
    old = pd.read_csv(VARIANT_ORDER)
    order = {}
    for g in GENES:
        have = set(scores[scores.gene == g].mutation)
        o = [m for m in old[old.gene == g].mutation if m in have]
        assert set(o) == have
        order[g] = o
    return order


def matrices(scores, order):
    """{gene: (y, {descriptor: (variants, seeds)})}."""
    out = {}
    for g in GENES:
        b = scores[scores.gene == g]
        y = b.groupby("mutation").label.first().loc[order[g]].to_numpy()
        out[g] = (y, {d: b.pivot(index="mutation", columns="seed", values=col).loc[order[g], SEEDS].to_numpy()
                      for d, col in DESCRIPTORS.items()})
    return out


def resamples(mats, order, unit="variant"):
    """{gene: N_BOOT class-stratified resamples}, one rng(BOOT_SEED) stream in GENES order.

    unit="variant": the published S1 resample (coordinate_vs_plddt.auc_table).
    unit="position": whole residue positions, stratified driver-only / passenger-only /
    mixed, the main-text estimator (published unit since 2026-10-20).
    unit="singleton": the position code with one variant per cluster; it must equal
    "variant" index for index (checked in main).
    """
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for g in GENES:
        y = mats[g][0]
        if unit == "variant":
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            out[g] = [np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)]) for _ in range(N_BOOT)]
        else:
            clusters = clusters_of(order[g], singleton=unit == "singleton")
            strata = strata_of(y, clusters)
            out[g] = [stratified_draw(rng, clusters, strata) for _ in range(N_BOOT)]
    return out


def auc_table(mats, idx_by_gene=None):
    """coordinate_vs_plddt.auc_table on these descriptors: mean of per-seed AUCs;
    class-stratified bootstrap (variants by default; see resamples), same resample for
    every seed and descriptor; macro = equal-gene mean within a replicate."""
    if idx_by_gene is None:
        idx_by_gene = resamples(mats, None, "variant")
    est, boots, per_seed = {}, {}, []
    for g in GENES:
        y, M = mats[g]
        idx = idx_by_gene[g]
        for d in DESCRIPTORS:
            A = M[d]
            seeds_auc = [auc(y, A[:, j]) for j in range(A.shape[1])]
            per_seed += [{"gene": g, "descriptor": d, "seed": s, "auc": a} for s, a in zip(SEEDS, seeds_auc)]
            est[(g, d)] = float(np.mean(seeds_auc))
            # a cluster resample can in principle miss a class; such a draw is dropped
            boots[(g, d)] = np.array([np.mean([auc(y[i], A[i, j]) for j in range(A.shape[1])])
                                      if 0 < y[i].sum() < len(i) else np.nan for i in idx])
    rows = []
    for d in DESCRIPTORS:
        for g in GENES:
            bs = boots[(g, d)]
            y = mats[g][0]
            rows.append({"gene": g, "descriptor": d, "auc": est[(g, d)], "lo": np.nanpercentile(bs, 2.5),
                         "hi": np.nanpercentile(bs, 97.5), "n_driver": int(y.sum()), "n_passenger": int((y == 0).sum()),
                         "valid_reps": int(np.isfinite(bs).sum())})
        mb = np.mean([boots[(g, d)] for g in GENES], axis=0)
        rows.append({"gene": "Macro", "descriptor": d, "auc": np.mean([est[(g, d)] for g in GENES]),
                     "lo": np.nanpercentile(mb, 2.5), "hi": np.nanpercentile(mb, 97.5),
                     "n_driver": 611, "n_passenger": 1186, "valid_reps": int(np.isfinite(mb).sum())})
    ps = pd.DataFrame(per_seed)
    # with one shared WT draw per gene and seed, dRMSF = RMSF - constant within a seed
    a, b = (ps[ps.descriptor == d].auc.to_numpy() for d in (DRMSF, RMSF))
    assert np.abs(a - b).max() < 1e-12, "dRMSF and RMSF per-seed AUCs must coincide"
    return pd.DataFrame(rows), ps


# ── stage 3: validation against the sweep and S9 ─────────────────────────────

def old_tables():
    """Old per-variant x per-repeat values (sweep draws), 1,798 variants."""
    s2b = pd.read_csv(OLD_S2B)
    vio = pd.read_csv(OLD_VIOLIN)
    sha(OLD_S2B), sha(OLD_VIOLIN)
    old = pd.concat([s2b[s2b.descriptor.isin([KS, POS, RMSD, RMSF])][["gene", "mutation", "label", "descriptor", "draw", "value"]],
                     vio[vio.descriptor == DRMSF][["gene", "mutation", "label", "descriptor", "draw", "value"]]])
    return old


def check_ks_against_s9(table):
    """KS must equal S9's matched values exactly."""
    s9 = pd.read_csv(S9_AUC)
    sha(S9_AUC)
    s9 = s9[s9.method == "KS"].set_index("gene").auc
    ks = table.set_index(["gene", "descriptor"]).xs(KS, level="descriptor").auc
    ks_err = {g: abs(ks[g] - s9[g if g != "Macro" else "MACRO"]) for g in GENES + ["Macro"]}
    assert max(ks_err.values()) < 1e-12, ks_err
    print(f"  KS reproduces S9 matched AUCs (max |diff| {max(ks_err.values()):.1e}); "
          f"macro {ks['Macro']:.6f}")


def validate(scores, table, per_seed):
    check_ks_against_s9(table)
    if not have_old_tables():
        print("  skipped: old-vs-new tables (the 10-18 S2b and violin outputs are absent)")
        return None, None
    old = old_tables()
    excluded = {("MSH2", "G162R")}
    old_s2b_auc = pd.read_csv(OLD_S2B_AUC).set_index(["gene", "descriptor"])
    old_vio_auc = pd.read_csv(OLD_VIOLIN_AUC).set_index(["gene", "descriptor"])
    sha(OLD_S2B_AUC), sha(OLD_VIOLIN_AUC)
    new = table.set_index(["gene", "descriptor"])
    rows = []
    for d in DESCRIPTORS:
        for g in GENES + ["Macro"]:
            o = old_vio_auc if d == DRMSF else old_s2b_auc
            old_quoted = float(o.loc[(g, d), "auc"]) if (g, d) in o.index else np.nan
            row = {"gene": g, "descriptor": d, "old_quoted_1798": old_quoted,
                   "new_matched_1797": float(new.loc[(g, d), "auc"]),
                   "new_lo": float(new.loc[(g, d), "lo"]), "new_hi": float(new.loc[(g, d), "hi"])}
            if g != "Macro":
                # old draws, restricted to the 1,797, so only the draws differ
                b = old[(old.gene == g) & (old.descriptor == d)]
                b = b[~b.set_index(["gene", "mutation"]).index.isin(list(excluded))]
                aucs = [auc(x.label.to_numpy(), x.value.to_numpy()) for _, x in b.groupby("draw")]
                ns = per_seed[(per_seed.gene == g) & (per_seed.descriptor == d)].auc
                row.update({"old_draws_1797": float(np.mean(aucs)), "old_seed_sd": float(np.std(aucs, ddof=1)),
                            "new_seed_sd": float(ns.std(ddof=1))})
            rows.append(row)
    v = pd.DataFrame(rows)
    for d in DESCRIPTORS:
        m = (v.descriptor == d) & (v.gene == "Macro")
        v.loc[m, "old_draws_1797"] = v[(v.descriptor == d) & (v.gene != "Macro")].old_draws_1797.mean()
    # coordinate_violins_auc.csv has no macro row: the quoted macro is the equal-gene mean
    for d in DESCRIPTORS:
        m = (v.descriptor == d) & (v.gene == "Macro")
        if v.loc[m, "old_quoted_1798"].isna().all():
            v.loc[m, "old_quoted_1798"] = v[(v.descriptor == d) & (v.gene != "Macro")].old_quoted_1798.mean()
    v["change_vs_quoted"] = v.new_matched_1797 - v.old_quoted_1798
    v["change_same_cohort"] = v.new_matched_1797 - v.old_draws_1797
    # subsample noise of a difference of two 10-draw means (independent draws)
    v["noise_se"] = np.sqrt((v.old_seed_sd ** 2 + v.new_seed_sd ** 2) / len(SEEDS))
    v["z"] = v.change_same_cohort / v.noise_se
    v["moves_gt_0.02"] = v.change_vs_quoted.abs() > 0.02
    v.to_csv(OUT / "auc_old_vs_new.csv", index=False)

    # per-variant agreement of 10-draw means with the sweep's 10-repeat means
    nm = scores.groupby(["gene", "mutation"])[list(DESCRIPTORS.values())].mean()
    om = old.groupby(["gene", "mutation", "descriptor"]).value.mean().unstack()
    rho = []
    for g in GENES:
        b = scores[scores.gene == g]
        for d, col in DESCRIPTORS.items():
            x = nm.loc[g][col]
            y = om.loc[g][d].reindex(x.index)
            assert y.notna().all()
            # split-half reliability of the new values: seeds 42-46 vs 47-51, Spearman-Brown to 10
            h = b.pivot(index="mutation", columns="seed", values=col)
            rh = spearmanr(h[SEEDS[:5]].mean(1), h[SEEDS[5:]].mean(1)).statistic
            rho.append({"gene": g, "descriptor": d, "n": len(x), "spearman_new_vs_sweep": spearmanr(x, y).statistic,
                        "pearson_new_vs_sweep": float(np.corrcoef(x, y)[0, 1]),
                        "mean_new": float(x.mean()), "mean_sweep": float(y.mean()),
                        "split_half_spearman_new": rh, "expected_10draw_reliability": 2 * rh / (1 + rh)})
    rho = pd.DataFrame(rho)
    rho.to_csv(OUT / "per_variant_agreement_with_sweep.csv", index=False)
    return v, rho


def ks_correlations(scores):
    nm = scores.groupby(["gene", "mutation"])[["ks", "posks"]].mean()
    om = None
    if have_old_tables():
        old = old_tables()
        om = old[old.descriptor.isin([KS, POS])].groupby(["gene", "mutation", "descriptor"]).value.mean().unstack()
    rows = []
    for g in GENES:
        rows.append({"gene": g, "n": len(nm.loc[g]),
                     "rho_new_matched": spearmanr(nm.loc[g].posks, nm.loc[g].ks).statistic,
                     **({"rho_old_quoted_1798": spearmanr(om.loc[g][POS], om.loc[g][KS]).statistic}
                        if om is not None else {})})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ks_vs_posks_spearman.csv", index=False)
    return r


# ── stage 4: KS + positional KS combination contrast ─────────────────────────

def seed_frames(scores):
    """One frame per seed, sorted (gene, mutation) like size_sweep.load_cell."""
    cols = ["gene", "mutation", "label", "afm_score", "d1_score", "ks", "posks"]
    out = []
    for s in SEEDS:
        f = (scores[scores.seed == s][cols].rename(columns={"ks": "plddt_ks_residue"})
             .sort_values(["gene", "mutation"]).reset_index(drop=True))
        out.append(f)
    for f in out[1:]:
        assert f[["gene", "mutation"]].equals(out[0][["gene", "mutation"]])
    return out


def combination_scores(frames):
    """{gene: (y, {"KS": (seeds, n), "posKS": ..., "KS+posKS": ...})}, gz equal-weight mean."""
    from analyses.annotation.simple_fusion_benchmark import fusion, size_sweep
    folds = size_sweep.make_folds(frames[0])["LOGO"]
    genes = frames[0].gene.to_numpy()
    per = []
    for f in frames:
        v = f[["plddt_ks_residue", "posks"]].to_numpy(float)
        per.append({"KS": v[:, 0], "posKS": v[:, 1],
                    "KS+posKS": fusion.combine(fusion.normalise(v, genes, "gz", folds), "mean")})
    out = {}
    for g in sorted(set(genes)):
        m = genes == g
        out[g] = (frames[0].label.to_numpy(int)[m], {k: np.stack([p[k][m] for p in per]) for k in per[0]})
    return out


def methods_bootstrap(comb, draws=N_BOOT, unit="variant", mutations=None):
    """simple_fusion_benchmark/annotation_tables.bootstrap (Methods): per gene and draw,
    one of the ten seeds, variants resampled with replacement (no stratification), the
    same draw for both scores; macro = equal-gene mean; percentile CI. rng = default_rng(0).

    unit="position" resamples whole residue positions instead (published unit since
    2026-10-20); unit="singleton" is that code with one variant per cluster and must
    reproduce unit="variant" exactly. ``mutations``: {gene: codes in comb order}.
    """
    rng = np.random.default_rng(0)
    label = {"variant": "Methods paired bootstrap (annotation_tables)",
             "position": "Methods paired bootstrap (annotation_tables), residue positions",
             "singleton": "singleton-cluster check"}[unit]
    genes = sorted(comb)
    pairs = [("KS+posKS", "KS"), ("KS+posKS", "posKS")]
    out = {p: {} for p in pairs}
    for g in genes:
        y, t = comb[g]
        n = len(y)
        clusters = None if unit == "variant" else clusters_of(mutations[g], singleton=unit == "singleton")
        d = {p: np.zeros(draws) for p in pairs}
        for k in range(draws):
            j = rng.integers(len(SEEDS))
            i = rng.integers(0, n, n) if clusters is None else unstratified_draw(rng, clusters)
            if len(set(y[i])) < 2:
                for p in pairs:
                    d[p][k] = np.nan
                continue
            for p in pairs:
                d[p][k] = auc(y[i], t[p[0]][j, i]) - auc(y[i], t[p[1]][j, i])
        for p in pairs:
            out[p][g] = d[p]
    rows = []
    for p in pairs:
        point = {g: np.mean([auc(comb[g][0], comb[g][1][p[0]][j]) - auc(comb[g][0], comb[g][1][p[1]][j])
                             for j in range(len(SEEDS))]) for g in genes}
        out[p]["MACRO"] = np.mean([out[p][g] for g in genes], axis=0)
        point["MACRO"] = np.mean([point[g] for g in genes])
        for g in genes + ["MACRO"]:
            x = out[p][g][np.isfinite(out[p][g])]
            lo, hi = np.percentile(x, [2.5, 97.5])
            rows.append({"procedure": label, "contrast": f"{p[0]} vs {p[1]}",
                         "scope": g, "delta_auc": point[g], "ci_lo": lo, "ci_hi": hi,
                         "p_gt0": float((x > 0).mean()), "draws": int(x.size)})
    return pd.DataFrame(rows)


def contrasts(scores):
    frames = seed_frames(scores)
    comb = combination_scores(frames)
    new = methods_bootstrap(comb)
    f0 = frames[0]
    mutations = {g: f0.mutation[f0.gene == g].to_numpy() for g in comb}
    check = methods_bootstrap(comb, unit="singleton", mutations=mutations)
    pd.testing.assert_frame_equal(check.drop(columns="procedure"), new.drop(columns="procedure"), check_exact=True)
    new = pd.concat([methods_bootstrap(comb, unit="position", mutations=mutations), new], ignore_index=True)
    # The published table also held the old ks_posks_am bootstrap and its reproduction on the
    # v6_ca caches, for comparison only; the SI value is the first row here.
    new.to_csv(OUT / "combination_contrast.csv", index=False)
    return new


# ── stage 5: panels ───────────────────────────────────────────────────────────

def violins(scores, table):
    per = scores.groupby(["gene", "mutation", "label"], as_index=False)[["rmsd", "delta_rmsf"]].mean()
    spec = [(RMSD, "rmsd", "RMSD to WT mean structure (Å)"), (DRMSF, "delta_rmsf", "ΔRMSF, mutant − WT (Å)")]
    fig, axes = plt.subplots(2, len(GENES), figsize=(W2, 0.6 * W2), gridspec_kw=dict(hspace=0.55, wspace=0.6))
    for r, (lab, col, ylabel) in enumerate(spec):
        for c, g in enumerate(GENES):
            ax = axes[r, c]
            b = per[per.gene == g]
            class_violin(ax, [(b[b.label == 1][col], "Driver", DRV), (b[b.label == 0][col], "Passenger", PAS)], seed=c)
            if lab == DRMSF:
                ax.axhline(0, ls=(0, (3, 2)), lw=.6, color=GRY, zorder=1)
            a = table[(table.gene == g) & (table.descriptor == lab)].iloc[0]
            ax.set_title(f"{pretty(g)}\nAUC {a.auc:.2f} [{a.lo:.2f}–{a.hi:.2f}]", fontsize=6)
            ax.set_xticklabels([f"n={(b.label == k).sum():,}" for k in (1, 0)])
            ax.tick_params(axis="x", labelsize=5.5)
            if c == 0:
                ax.set_ylabel(ylabel)
    fig.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4, color=DRV, label="Driver"),
                        plt.Line2D([], [], ls="", marker="o", ms=4, color=PAS, label="Passenger")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    save(fig, "S1b_coordinate_violins_matched_draws", out_dir=OUT)


def auc_panel(table):
    fig, ax = panel(w=W15, h=W1 * 0.85)
    genes = GENES + ["Macro"]
    ys = np.arange(len(genes))[::-1]
    series = [(KS, ACC, "o", True), (POS, "#e07a1f", "s", True), (RMSD, GRY, "^", False),
              (DRMSF, "#9a9a9a", "v", False)]
    offs = np.linspace(0.27, -0.27, len(series))
    for (lab, colour, mk, filled), off in zip(series, offs):
        s = table[table.descriptor == lab].set_index("gene").loc[genes]
        for y, (g, r) in zip(ys, s.iterrows()):
            ax.plot([r.lo, r.hi], [y + off] * 2, color=colour, lw=0.7)
            ax.plot(r.auc, y + off, mk, color=colour, ms=3.4 if g != "Macro" else 4.2,
                    mfc=colour if filled else "white", mew=0.8, label=lab if g == genes[0] else None)
    ax.axvline(0.5, color=GRY, lw=0.5, ls=":")
    ax.axhline(0.5, color=GRY, lw=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty(g) for g in genes])
    ax.set_xlabel(f"Driver vs passenger ROC-AUC (N = {N}, same predictions for every descriptor)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.2, columnspacing=0.9)
    save(fig, "S1c_auc_matched_draws", out_dir=OUT)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="recompute descriptors even if cached")
    args = ap.parse_args()
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    for p in [Path(__file__), REPR / "n90_comparison.py", REPR / "check_representatives.py",
              ROOT / "drift/structure_draws.py", ROOT / "drift/protein_configs.json",
              ROOT / "metric_analysis/pipeline/library/positional_ks.py",
              ROOT / "metric_analysis/pipeline/orchestration/run_ensemble_sweep_parallel.py"]:
        sha(p)
    if SCORES.exists() and not args.recompute:
        scores = pd.read_csv(SCORES, float_precision="round_trip")  # exact: 1-ulp parse errors break AUC ties
        print(f"  reusing {SCORES.relative_to(ROOT)} (pass --recompute to rebuild)")
    else:
        scores = calculate()
    sha(SCORES)
    assert len(scores) == 1797 * len(SEEDS)
    assert scores.groupby("gene").mutation.nunique().to_dict() == {g: n for g, (n, _) in EXPECTED.items()}
    assert (scores.ks - scores.ks_direct).abs().max() < 1e-12

    order = variant_order(scores)
    mats = matrices(scores, order)
    # Intervals in both resampling units. Residue positions are the published unit since
    # 2026-10-20 (Dan): the panels and recorded values use them; the variant-level table
    # is kept for the record (auc_matched_draws_variant.csv).
    variant_idx = resamples(mats, order, "variant")
    singleton_idx = resamples(mats, order, "singleton")
    assert all(len(variant_idx[g]) == len(singleton_idx[g]) and
               all(np.array_equal(a, b) for a, b in zip(variant_idx[g], singleton_idx[g])) for g in GENES), \
        "singleton position clusters must reproduce the variant resamples exactly"
    variant_table, per_seed = auc_table(mats, variant_idx)
    table, _ = auc_table(mats, resamples(mats, order, "position"))
    variant_table.assign(unit="variant").to_csv(OUT / "auc_matched_draws_variant.csv", index=False)
    table = table.assign(unit="position")
    table.to_csv(OUT / "auc_matched_draws.csv", index=False)
    pd.DataFrame([{"gene": g, **describe(mats[g][0], clusters_of(order[g]))} for g in GENES]).to_csv(
        OUT / "position_structure.csv", index=False)
    per_seed.to_csv(OUT / "auc_per_seed.csv", index=False)
    print(table.pivot(index="gene", columns="descriptor", values="auc").loc[GENES + ["Macro"]].round(4).to_string())

    v, rho = validate(scores, table, per_seed)
    if v is not None:
        print(v[["gene", "descriptor", "old_quoted_1798", "old_draws_1797", "new_matched_1797", "change_vs_quoted", "z"]]
              .round(4).to_string(index=False))
        print(rho.round(3).to_string(index=False))
    corr = ks_correlations(scores)
    print(corr.round(3).to_string(index=False))
    con = contrasts(scores)
    print(con[con.scope == "MACRO"].round(4).to_string(index=False))

    violins(scores, table)
    auc_panel(table)
    for r in table.itertuples():
        record("S1_matched", f"{r.gene}_{r.descriptor}_AUC_N90", round(r.auc, 4),
               f"mean of 10 per-seed AUCs; 95% CI [{r.lo:.3f}, {r.hi:.3f}], {N_BOOT} bootstraps of residue "
               f"positions, stratified driver-only/passenger-only/mixed")
    for r in variant_table.itertuples():
        record("S1_matched", f"{r.gene}_{r.descriptor}_AUC_N90_variant_CI", f"[{r.lo:.3f}, {r.hi:.3f}]",
               f"for the record: {N_BOOT} class-stratified variant bootstraps")
    for r in corr.itertuples():
        record("S1_matched", f"{r.gene}_spearman_posks_vs_plddt_ks", round(r.rho_new_matched, 3), f"10-seed means, n={r.n}")
    write_values("s1_matched_values.csv", out_dir=OUT)
    provenance(out_dir=OUT, N=N, seeds=SEEDS, n_boot=N_BOOT, boot_seed=BOOT_SEED,
               draws="DRIFT production N=90 draws (select_structure_columns + source_draw_key), reconstructed by "
                     "s8_representatives/n90_comparison.orders",
               frame="core_fit (WT rigid core); scored chain only; POLE WT POLE_268-471_WT",
               posks_centre="mean over the full WT pool in the core-fit cache (as the sweep's reference_centre)",
               cohort="1,797 = annotation 1,798 less MSH2 G162R (POLE P356S, no coordinates, left the cohort 2026-10-20)",
               expected_counts=EXPECTED, combination_bootstrap="Methods paired bootstrap (annotation_tables.bootstrap), "
               "2,000 draws, default_rng(0)",
               source_sha256=HASHES, runtime_seconds=round(time.monotonic() - t0, 1))


if __name__ == "__main__":
    main()
