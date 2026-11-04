"""Non-learned fusion across ensemble sizes.

Swept over the seven ensemble sizes and ten descriptor seeds the score caches
provide. No model is fitted anywhere in it, so the whole 70-cell grid costs one
pass over the caches.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analyses.annotation.simple_fusion_benchmark import fusion, metrics
from metric_analysis.pipeline.library.canonical_cohort import logo_fold_group, logo_splits
from sklearn.model_selection import StratifiedKFold

CACHE = ROOT / "drift/cache"
# Pinned to the current protein-config fingerprint: caches of earlier cohorts
# share the directory, and an unpinned glob matched two of them once v5 existed.
CACHE_GLOB = "labeled_size{size}_seed{seed}_proteins{fingerprint}_esfeat7.csv"
SIZES = (1, 5, 10, 20, 40, 60, 90)
SEEDS = tuple(range(42, 52))
COLUMNS = ["gene", "mutation", "label", fusion.AFM, *fusion.DESCRIPTORS]
# AlphaMissense is a sequence model: it does not see the ensemble, so its column
# is identical in every cell. The run asserts this rather than assuming it.
SIZE_INVARIANT = ["afm_score"]
REFERENCES = {"AlphaMissense (raw)": "afm_score",
              "residue KS alone": "plddt_ks_residue", "centred KS alone": "plddt_ks_centred"}


def load_cell(size, seed, gt_group="evaluation"):
    """One (size, seed) cache, restricted to the strict canonical cohort.

    `gt_group` picks the label source; only `annotation` exists since the binary
    functional labels (thresholded TP53/KRAS assays) were removed on 2026-10-13.
    """
    from drift import run
    from drift.protein_config import protein_config_fingerprint
    matches = sorted(glob.glob(str(CACHE / CACHE_GLOB.format(
        size=size, seed=seed, fingerprint=protein_config_fingerprint()))))
    if len(matches) != 1:
        raise SystemExit(f"expected one cache for size={size} seed={seed}, found {matches}")
    frame = run._apply_strict_canonical_cohort(pd.read_csv(matches[0]))
    frame = run._gt_subset(frame, gt_group)
    frame = frame.sort_values(["gene", "mutation"]).reset_index(drop=True)[COLUMNS]
    if size < 2:
        frame[["plddt_ks_residue", "plddt_ks_centred"]] = np.nan
    return frame, Path(matches[0])


def make_folds(frame):
    genes = frame.gene.to_numpy()
    strata = frame.gene + "__" + frame.label.astype(str)
    folds = {"LOGO": logo_splits(genes, "grouped"),
             "stratified": list(StratifiedKFold(5, shuffle=True, random_state=42)
                                .split(frame, strata))}
    for protocol, pairs in folds.items():
        covered = np.zeros(len(frame), dtype=int)
        for tr, va in pairs:
            assert not set(tr).intersection(va)
            covered[va] += 1
            if protocol == "LOGO":
                assert {logo_fold_group(g) for g in genes[tr]}.isdisjoint(
                    {logo_fold_group(g) for g in genes[va]})
        assert (covered == 1).all()
    return folds


def cell_scores(frame, folds, protocols):
    """Every non-learned score plus the raw references, for one cell."""
    records = []
    available = [c for c in (fusion.AFM, *fusion.DESCRIPTORS) if not frame[c].isna().all()]
    arms = {name: columns for name, columns in fusion.ARMS.items()
            if set(columns).issubset(available)}
    for label, column in REFERENCES.items():
        if column not in available:
            continue
        for protocol in protocols:
            records.append({"kind": "input", "method": label, "family": "raw input",
                            "arm": column, "normalisation": "none", "combiner": "none",
                            "weights": "", "protocol": protocol,
                            "score": frame[column].to_numpy(dtype=float)})
    for record in fusion.build(frame, {p: folds[p] for p in protocols}, arms=arms):
        targets = protocols if record["protocol"] == "shared" else (record["protocol"],)
        for protocol in targets:
            records.append({**record, "kind": "non-learned",
                            "family": f"{record['normalisation']}-{record['combiner']}",
                            "protocol": protocol})
    return records


def sweep(sizes, seeds, protocols, out, gt_group="evaluation"):
    reference_frame = reference_keys = folds = cells = None
    genes = y = None
    macro_rows, gene_rows, cache_manifest = [], [], []
    started = time.perf_counter()
    for size in sizes:
        for seed in seeds:
            frame, path = load_cell(size, seed, gt_group)
            keys = frame[["gene", "mutation"]].to_numpy()
            if reference_frame is None:
                reference_frame, reference_keys = frame, keys
                folds = make_folds(frame)
                genes = frame.gene.to_numpy()
                y = frame.label.to_numpy(dtype=int)
                cells = {p: metrics.label_cells(y, genes, folds[p], protocol=p)
                         for p in protocols}
                frame.to_csv(out / "cohort.csv", index=False)
            else:
                # Same variants in the same order in every cell, so a curve over
                # sizes is a curve over ensemble depth and nothing else.
                assert (keys == reference_keys).all(), (size, seed)
                assert (frame.label.to_numpy() == reference_frame.label.to_numpy()).all()
                for column in SIZE_INVARIANT:
                    np.testing.assert_allclose(frame[column], reference_frame[column])
            # load_cell blanks both KS columns at N=1, so only AlphaMissense is required there.
            required = ["afm_score"] if size < 2 else COLUMNS[3:]
            assert np.isfinite(frame[required].to_numpy()).all(), (size, seed)
            cache_manifest.append({"size": size, "seed": seed, "gt_group": gt_group,
                                   "cache": path.name,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for record in cell_scores(frame, folds, protocols):
                score = np.asarray(record["score"], dtype=float)
                protocol = record["protocol"]
                predictions, valid, used = metrics.apply_policies(
                    score, y, genes, cells[protocol], folds[protocol],
                    is_probability=record["arm"] == "afm_score")
                context = {"size": size, "seed": seed, "gt_group": gt_group,
                           "protocol": protocol,
                           "kind": record["kind"], "method": record["method"],
                           "family": record["family"], "arm": record["arm"],
                           "normalisation": record["normalisation"],
                           "combiner": record["combiner"], "weights": record["weights"]}
                per_gene = []
                for gene in sorted(set(genes)):
                    idx = genes == gene
                    row = metrics.gene_metrics(
                        y[idx], score[idx], {p: v[idx] for p, v in predictions.items()},
                        {p: v[idx] for p, v in valid.items()},
                        [c for c in used if c["gene"] == gene],
                        is_probability=record["arm"] == "afm_score")
                    per_gene.append(row)
                    # Per-gene output is deliberately narrow: the full battery
                    # per gene per cell is a 50 MB table nobody reads.
                    gene_rows.append({**context, "gene": gene,
                                      **{k: row[k] for k in ("roc_auc", "average_precision",
                                                             "precision_at_R")}})
                table = pd.DataFrame(per_gene)
                macro_rows.append({**context,
                                   **{f"macro_{c}": table[c].mean()
                                      for c in metrics.METRIC_COLUMNS}})
            print(f"  size={size:<3} seed={seed}  "
                  f"{len(macro_rows)} evaluations  {time.perf_counter() - started:.0f}s",
                  flush=True)
    macro = pd.DataFrame(macro_rows)
    pd.DataFrame(gene_rows).to_csv(out / "sweep_per_gene.csv", index=False)
    macro.to_csv(out / "sweep_macro.csv", index=False)
    pd.DataFrame(cache_manifest).to_csv(out / "cache_manifest.csv", index=False)
    return macro


def summarise(macro, out):
    """Mean and spread over descriptor seeds, per (size, protocol, method, arm)."""
    keys = ["size", "gt_group", "protocol", "kind", "method", "family", "arm",
            "normalisation", "combiner", "weights"]
    values = [c for c in macro.columns if c.startswith("macro_")]
    grouped = macro.groupby(keys, dropna=False, sort=False)
    summary = grouped[values].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}__{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index()
    assert (summary["macro_roc_auc__count"] == macro.seed.nunique()).all()
    # Every curve is read against raw AlphaMissense at the same size, which does
    # not move with the ensemble; carrying it per row keeps deltas honest.
    baseline = summary[summary.method == "AlphaMissense (raw)"][
        ["size", "protocol", "macro_roc_auc__mean"]].rename(
        columns={"macro_roc_auc__mean": "afm_roc_auc"})
    summary = summary.merge(baseline, on=["size", "protocol"], validate="many_to_one")
    summary["delta_roc_auc_vs_afm"] = summary["macro_roc_auc__mean"] - summary["afm_roc_auc"]
    summary.to_csv(out / "sweep_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: results_size_sweep_<gt-group>_<RUN_TAG>")
    # Binary labels exist only for the annotation route since 2026-10-13 (no
    # assay is thresholded any more); 'functional'/'evaluation' are retired.
    parser.add_argument("--gt-group", default="annotation", choices=["annotation"],
                        help="label source: annotation (POLE/MSH2/MSH6/VHL/PTEN/PIK3CA)")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--protocols", nargs="+", default=["LOGO"],
                        choices=["LOGO", "stratified"],
                        help="LOGO is the honest protocol and the default; the "
                             "fixed-size benchmark already reports both")
    args = parser.parse_args()
    if args.out is None:
        from metric_analysis.pipeline.paths import tagged
        args.out = HERE / tagged(f"results_size_sweep_{args.gt_group}")
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}; choose a fresh --out")
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    macro = sweep(args.sizes, args.seeds, tuple(args.protocols), args.out,
                  args.gt_group)
    summary = summarise(macro, args.out)
    (args.out / "metadata.json").write_text(json.dumps({
        "sizes": args.sizes, "seeds": args.seeds, "protocols": args.protocols,
        "gt_group": args.gt_group,
        "genes": sorted(pd.read_csv(args.out / "cohort.csv").gene.unique().tolist()),
        "evaluations": len(macro), "summary_rows": len(summary),
        "cache_tag": "esfeat7", "cohort": "strict canonical",
        "arms": {k: v for k, v in fusion.ARMS.items()},
        "am_weights": list(fusion.AM_WEIGHTS),
        "methods": [s[0] for s in fusion.specifications()],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(HERE.glob("*.py"))},
        "selection": "no model is fitted; no configuration is selected",
        "elapsed_seconds": time.perf_counter() - start,
    }, indent=2))
    print(f"DONE: {len(macro)} evaluations -> {args.out} "
          f"in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
