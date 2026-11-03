"""Checkpointed local precision/ranking experiment on fixed 25x5 AF3 pools.

PYTHONPATH=. python experiments/seed_structure/plateau_run.py
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from drift import structure_draws as runtime
from metric_analysis.pipeline.labels import artifact_key
from metric_analysis.pipeline.library.canonical_cohort import load_all_canonical_cohorts
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks
from metric_analysis.pipeline.library.descriptor_sampling import stable_rng
from ks_resamples import ResampledKS
from sampling_auc import GENES, TAGS, ROOT, HERE, coordinates, plddt

OUT = HERE / "plateau_20261018"
SIZES = [1, 5, 10, 20, 25, 40, 60, 90, 125]
REPS = 200
POLICIES = ["uniform_independent", "uniform_matched", "spread_independent",
            "spread_matched", "concentrated_independent", "concentrated_matched"]
METRICS = ["plddt_ks", "positional_ks"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def orders(gene, source, side):
    """Nested full orders. WT key is shared across variants within gene/draw."""
    output = np.empty((3, REPS, 125), dtype=np.int16)
    lookup = {tag: i for i, tag in enumerate(TAGS)}
    key = f"wt::{gene}::{Path(source).name}" if side == "wt" else runtime.source_draw_key("mut", source)
    for repeat in range(REPS):
        output[0, repeat] = [lookup[t] for t in runtime.rank_items_deterministically(TAGS, seed=42+repeat, key=key)]
        rng = stable_rng(20261018 + repeat, gene, key, "seed_allocation")
        seeds = rng.permutation(np.arange(1, 26))
        samples = np.stack([rng.permutation(5) for _ in seeds])
        tags = np.array([[lookup[f"seed-{s}_sample-{k}"] for k in samples[i]]
                         for i, s in enumerate(seeds)])
        output[1, repeat] = tags.T.reshape(-1)
        output[2, repeat] = tags.reshape(-1)
    return output


def evaluate(w, m, wo, mo):
    kernel = ResampledKS(w, m)
    full = residue_ks(w, m)
    result = np.empty((len(POLICIES), len(SIZES), REPS))
    for policy in range(len(POLICIES)):
        left = wo[policy // 2]
        right = left if policy % 2 else mo[policy // 2]
        for j, size in enumerate(SIZES):
            if size == 125:
                result[policy, j] = full
            elif size == 1:
                result[policy, j] = (w[:, left[:, 0]] != m[:, right[:, 0]]).mean(0)
            else:
                result[policy, j] = kernel(left[:, :size], right[:, :size])
    return result


def run_gene(gene):
    cohort = load_all_canonical_cohorts()
    records = cohort[cohort.protein == gene].to_dict("records")
    folder = OUT / gene
    folder.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "metric_analysis/structure_metrics" / gene / "ensemble_aware/cache_corefit"
    report = json.loads((cache / f"{gene}_core_fit_report.json").read_text())
    index = pd.read_csv(cache / f"{gene}_core_fit_per_structure.csv")
    by_key = {artifact_key(a): a for a in index.artifact.unique()}
    dataset = ds.dataset(cache / f"{gene}_rmsf_aligned_ca_coordinate_cache.parquet")
    recovered = None
    if gene in ["MSH2", "PIK3CA"]:
        mapping = pd.read_csv(HERE / "results_rerun_20261018" / f"{gene}_recovered_seed_mapping.csv")
        recovered = dict(zip(mapping.structure_index, mapping["sample"]))
    wc, resid = coordinates(dataset, report["wt_artifact"], recovered)
    centre = wc.mean(1, keepdims=True)
    wd = np.linalg.norm(wc - centre, axis=-1)
    wt_path = records[0]["wt_source_file"]
    assert {r["wt_source_file"] for r in records} == {wt_path}
    w = plddt(wt_path, records[0])
    wo = orders(gene, wt_path, "wt")
    np.savez_compressed(folder / "wt_inputs.npz", orders=wo, coordinates=wc, residues=resid)
    for number, record in enumerate(records, 1):
        target = folder / f"{record['mutation_code']}.npz"
        if target.exists():
            continue
        m = plddt(record["source_file"], record)
        mo = orders(gene, record["source_file"], "mutant")
        result = np.full((2, len(POLICIES), len(SIZES), REPS), np.nan)
        result[0] = evaluate(w, m, wo, mo)
        key = artifact_key(Path(record["source_file"]).name.replace("_plddt_values.csv", ""))
        exclusion = ""
        try:
            if key not in by_key:
                raise ValueError("coordinate artifact missing")
            mc, mr = coordinates(dataset, by_key[key])
            if not np.array_equal(mr, resid):
                raise ValueError("coordinate residue mismatch")
            md = np.linalg.norm(mc - centre, axis=-1)
            result[1] = evaluate(wd, md, wo, mo)
        except ValueError as error:
            exclusion = str(error)
        metadata = {"gene": gene, "mutation": record["mutation_code"], "label": record["label"],
                    "source": record["source_file"], "wt_source": wt_path,
                    "source_sha256": sha(record["source_file"]), "wt_sha256": sha(wt_path),
                    "coordinate_exclusion": exclusion,
                    "source_columns": len(pd.read_csv(record["source_file"], nrows=0, index_col=0).columns)}
        temporary = target.with_suffix(".partial.npz")
        np.savez_compressed(temporary, scores=result, mutant_orders=mo, metadata=json.dumps(metadata))
        temporary.replace(target)
        if number % 25 == 0:
            print(f"{gene}: {number}/{len(records)} checkpointed", flush=True)
    print(f"Completed {gene}: {len(records)} variants", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest = {"genes": GENES, "sizes": SIZES, "repetitions": REPS,
                "policies": POLICIES, "metrics": METRICS,
                "candidate_pool": "seeds 1-25, samples 0-4 on both sides",
                "WT_draw": "one draw shared across variants within gene/repeat",
                "mutation_draw": "source-specific, independent or WT-identity-matched",
                "score_direction": "driver-high", "coordinate_scope": "core_fit core residues",
                "plan_sha256": sha(HERE / "PLATEAU_ANALYSIS_PLAN.md"),
                "script_sha256": sha(__file__), "kernel_sha256": sha(HERE / "ks_resamples.py"),
                "status": "running"}
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ["sizes", "repetitions", "policies", "metrics", "candidate_pool", "script_sha256", "kernel_sha256"]:
            assert old[key] == manifest[key], f"stale checkpoint contract: {key}"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cohort = load_all_canonical_cohorts()
    cohort[cohort.protein.isin(GENES)].to_csv(OUT / "cohort.csv", index=False)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(run_gene, GENES))
    manifest["status"] = "scores_complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
