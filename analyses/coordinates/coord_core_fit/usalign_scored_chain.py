"""US-align frame control on the SCORED chain, for MSH6 (2026-10-17, Dan's request).

The 10-15/16 control (`run_frame_control.py`, cache_usalign) reused the source cache's own
US-align coordinates. US-align ran with its defaults (monomer mode, -ter 2), which superpose
on the first chain of each model. For POLE, PTEN, VHL, PIK3CA and MSH2 that is the scored
chain, so the control there is "standard US-align of the scored chain vs the WT-core fit".
For MSH6 the first chain is MSH2, so MSH6 sat in MSH2's frame. This puts MSH6 on the same
footing as the other five genes: every structure's MSH6 chain (C-alpha only, from the same
control cache, so exactly the same structures) is re-superposed with US-align onto one WT
MSH6 chain, and the sweeps are rerun in that frame.

US-align runs with -TMscore 1 (TM-score superposition, residues paired by index). The
sequences are identical, so the structure-based alignment of the default mode only
re-finds that pairing: on three test structures the two superpositions differed by
0.04-0.06 A per residue, and -TMscore 1 is ~18x faster (0.45 s vs 8 s per structure).

Steps (each skips if its output exists, unless --force):
  1. align: writes metric_analysis/structure_metrics/MSH6/ensemble_aware/cache_usalign_scored_chain/
     (same schema and file name as the other caches) + MSH6_usalign_scored_chain_report.json.
  2. sweep: COORD_FRAME=usalign_scored_chain (protein_configs.json MSH6.coord_frames), the
     RMSF/RMSD/POSKS N-sweeps and RMSD over all structures, as run_frame_control.py does.
  3. compare: results/compare_msh6_scored_chain.csv / .md, core-fit vs both US-align frames.

Note S1's frame comparison (s1_frame_sensitivity/frame_sensitivity.py) reads the step-1 cache;
run_all.sh runs only that step. The sweep and compare steps also need the US-align-frame
sweeps of an earlier control (run_frame_control.py), which is not in this release.

    PYTHONPATH="." python \
        analyses/coordinates/coord_core_fit/usalign_scored_chain.py [--steps align,sweep,compare] [--workers 12]
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FRAME = "usalign_scored_chain"
os.environ.setdefault("COORD_FRAME", FRAME)          # before any config is loaded (sweep step)

REPO = Path(__file__).resolve().parents[3]
SM = REPO / "metric_analysis" / "structure_metrics" / "MSH6"
SRC = SM / "ensemble_aware" / "cache_usalign"
DST = SM / "ensemble_aware" / "cache_usalign_scored_chain"
CACHE = "MSH6_rmsf_aligned_ca_coordinate_cache.parquet"
WT = "MutS_alpha_MSH6_GT_WT_data"
USALIGN = shutil.which("USalign") or "USalign"
OUT = Path(__file__).resolve().parent / "results"


def write_ca_pdb(path, g):
    with open(path, "w") as f:
        for i, r in enumerate(g.itertuples(index=False), 1):
            ic = "" if pd.isna(r.icode) or str(r.icode) in ("", "nan", "?") else str(r.icode)[:1]
            f.write(f"ATOM  {i:5d}  CA  {str(r.resname)[:3]:>3s} B{int(r.resid):4d}{ic:1s}   "
                    f"{r.x_aligned:8.3f}{r.y_aligned:8.3f}{r.z_aligned:8.3f}  1.00  0.00           C\n")
        f.write("END\n")


def align_one(args):
    key, block, ref_pdb, tmp = args
    d = Path(tempfile.mkdtemp(dir=tmp))
    try:
        q, m = d / "q.pdb", d / "m.txt"
        write_ca_pdb(q, block)
        p = subprocess.run([USALIGN, str(q), str(ref_pdb), "-mol", "prot", "-TMscore", "1", "-m", str(m)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{key}: USalign failed: {p.stderr[:300]}")
        rows = [l.split() for l in m.read_text().splitlines() if l.strip()[:1] in "012" and len(l.split()) == 5]
        T = np.array([[float(x) for x in r[1:]] for r in rows])
        t, U = T[:, 0], T[:, 1:]
        tm = [float(l.split("=")[1].split()[0]) for l in p.stdout.splitlines() if l.startswith("TM-score=")]
        X = block[["x_aligned", "y_aligned", "z_aligned"]].to_numpy()
        return key, X @ U.T + t, tm[1] if len(tm) > 1 else np.nan   # TM normalised by the reference
    finally:
        shutil.rmtree(d, ignore_errors=True)


def step_align(workers, force):
    out = DST / CACHE
    if out.exists() and not force:
        print(f"[align] exists: {out}")
        return
    d = pd.read_parquet(SRC / CACHE)
    assert set(d.chain.astype(str)) == {"B"}, "the MSH6 control cache should hold chain B only"
    d = d.sort_values(["artifact", "structure_index", "resid"], kind="stable").reset_index(drop=True)
    groups = d.groupby(["artifact", "structure_index"], sort=False, observed=True).indices
    wt_keys = sorted(k for k in groups if k[0] == WT)
    ref_key = wt_keys[0]
    DST.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="usalign_msh6_"))
    ref_pdb = tmp / "ref.pdb"
    write_ca_pdb(ref_pdb, d.iloc[groups[ref_key]])
    cols = ["resid", "icode", "resname", "x_aligned", "y_aligned", "z_aligned"]
    jobs = ((k, d.iloc[ix][cols], ref_pdb, tmp) for k, ix in groups.items())
    xyz = d[["x_aligned", "y_aligned", "z_aligned"]].to_numpy().copy()
    tms = {}
    with ProcessPoolExecutor(workers) as ex:
        for n, (k, Y, tm) in enumerate(ex.map(align_one, jobs, chunksize=8), 1):
            xyz[groups[k]] = Y
            tms[k] = tm
            if n % 1000 == 0:
                print(f"[align] {n}/{len(groups)}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    d[["x_aligned", "y_aligned", "z_aligned"]] = xyz
    d.to_parquet(out, index=False)
    for f in SRC.glob("MSH6_core_residues.csv"):
        shutil.copy(f, DST / f.name)
    tm = pd.Series(tms)
    rep = {"dataset": "MSH6", "frame": FRAME,
           "coordinates": "each structure's MSH6 chain (C-alpha) superposed with US-align (-mol prot -TMscore 1) "
                          "onto one WT MSH6 chain; same structures as cache_usalign / cache_corefit",
           "source_cache": str(SRC / CACHE), "reference": {"artifact": ref_key[0], "structure_index": int(ref_key[1])},
           "usalign": USALIGN, "structures": len(groups), "artifacts": int(d.artifact.nunique()),
           "tm_score_vs_reference": {"median": round(float(tm.median()), 4), "min": round(float(tm.min()), 4),
                                     "p05": round(float(tm.quantile(0.05)), 4)}}
    (DST / f"MSH6_{FRAME}_report.json").write_text(json.dumps(rep, indent=1))
    print(f"[align] wrote {out} ({len(groups)} structures); TM median {rep['tm_score_vs_reference']['median']}")


def step_sweep():
    from metric_analysis.pipeline.library import ensemble_aware_pipeline as eap
    cfg = eap.get_ensemble_aware_config("MSH6")
    if FRAME not in str(cfg.get("coord_cache_dir")):
        raise SystemExit(f"COORD_FRAME override not applied: {cfg.get('coord_cache_dir')}")
    print(f"=== [MSH6] {FRAME}: {cfg['coord_cache_dir']} -> {cfg['sweep_dir']}", flush=True)
    eap.run_rmsf_rmsd_ensemble_sweep("MSH6", force=True)
    eap.run_rmsd_all_instances(cfg, force=True)


def step_compare():
    fams = {"RMSF": "RMSF_aligned_coord", "RMSD": "RMSD_aligned_coord", "POSKS": "POSKS_aligned_coord"}
    head = ["rmsf_mean", "rmsd_ensemble_mean_to_wt_mean", "posks_distance"]
    vers = {"core_fit": SM / "ensemble_aware" / "ensemble_sweeps",
            "usalign_first_chain": SM / "ensemble_aware_usalign" / "ensemble_sweeps",
            FRAME: SM / f"ensemble_aware_{FRAME}" / "ensemble_sweeps"}
    rows, raws = [], {}
    for fam, stem in fams.items():
        for v, p in vers.items():
            s = pd.read_csv(p / f"MSH6_{stem}_ensemble_sweep_summary.csv")
            s = s[(s.ensemble_size == s.ensemble_size.max()) & s.metric_col.isin(head)]
            rows += [{"metric_col": r.metric_col, "version": v, "auc_mean": r.auc_mean, "auc_std": r.auc_std}
                     for r in s.itertuples()]
            r = pd.read_csv(p / f"MSH6_{stem}_ensemble_sweep_raw.csv")
            r = r[r.ensemble_size == r.ensemble_size.max()]
            raws.setdefault(v, []).append(r.groupby("artifact")[[c for c in head if c in r]].mean())
    t = pd.DataFrame(rows).pivot(index="metric_col", columns="version", values="auc_mean")
    m = {v: pd.concat(x, axis=1) for v, x in raws.items()}
    from scipy.stats import spearmanr
    for v in ("usalign_first_chain", FRAME):
        j = m["core_fit"].join(m[v], rsuffix="_o", how="inner")
        t[f"rho_core_vs_{v}"] = [spearmanr(j[c], j[c + "_o"])[0] for c in t.index]
    OUT.mkdir(exist_ok=True)
    t.round(4).to_csv(OUT / "compare_msh6_scored_chain.csv")
    (OUT / "compare_msh6_scored_chain.md").write_text(
        "# MSH6: WT-core fit vs US-align on MSH2 (old control) vs US-align on MSH6 (scored chain)\n\n"
        "AUC = signed driver-high ROC-AUC, mean over repeats at N = 90; rho = per-variant Spearman "
        "against the core fit (variant means over repeats).\n\n" + t.round(3).to_markdown() + "\n")
    print(t.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="align,sweep,compare")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    steps = a.steps.split(",")
    if "align" in steps:
        step_align(a.workers, a.force)
    if "sweep" in steps:
        step_sweep()
    if "compare" in steps:
        step_compare()


if __name__ == "__main__":
    main()
