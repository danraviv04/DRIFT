"""Rerun original WT seed controls, recovering unlabeled WT samples from archives.

Recovery follows the sorted, seed-sample-only enumeration documented in
analyses/provenance/v6_ca_run/ingest_new_wts.sh. Existing coordinate caches are read only.
Run with PYTHONPATH=. in the drift environment; reads DRIFT_ARCHIVE_ROOT.
"""
import json
import os
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd

import seed_icc
import seed_vs_random_split
import seed_vs_random_positional as positional


OUT = Path(__file__).resolve().parent / "results_rerun_20261018"
DRIVE = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT"))
ARCHIVES = {
    "MSH2": DRIVE / "EXTRA_WT/MutS_alpha_MSH2_WT_vartmpl_data.tar.gz",
    "PIK3CA": DRIVE / "EXTRA_WT/PIK3CA_WT_vartmpl_data.tar.gz",
}


def recover_load(gene):
    if gene not in ARCHIVES:
        return ORIGINAL_LOAD(gene)
    directory = positional.SM / gene / "ensemble_aware" / "cache_corefit"
    report = json.loads((directory / f"{gene}_core_fit_report.json").read_text())
    archive = ARCHIVES[gene]
    print(f"Recovering {gene} seed labels from {archive}", flush=True)
    with tarfile.open(archive, "r:*") as handle:
        names = sorted(
            member.name for member in handle
            if member.isfile()
            and member.name.lower().endswith((".cif", ".mmcif", ".pdb", ".ent"))
            and not (member.name.startswith("._") or "/._" in member.name)
            and re.search(r"seed-\d+_sample-\d+", member.name)
        )
    tags = [re.search(r"seed-\d+_sample-\d+", name)[0] for name in names]
    expected = {f"seed-{seed}_sample-{sample}" for seed in range(1, 26) for sample in range(5)}
    assert len(tags) == 125 and set(tags) == expected
    frame = pd.read_parquet(
        directory / f"{gene}_rmsf_aligned_ca_coordinate_cache.parquet",
        filters=[("artifact", "==", report["wt_artifact"])],
    )
    assert set(frame.structure_index.unique()) == set(range(125))
    assert frame["sample"].isna().all() or frame["sample"].astype(str).eq("").all()
    mapping = dict(enumerate(tags))
    frame["sample"] = frame.structure_index.map(mapping)
    frame = frame[frame.core]
    assert not frame.duplicated(["resid", "sample"]).any()
    axes = ["x_aligned", "y_aligned", "z_aligned"]
    pivots = {axis: frame.pivot(index="resid", columns="sample", values=axis) for axis in axes}
    columns = pivots[axes[0]].columns
    coordinates = np.stack([pivots[axis][columns].to_numpy() for axis in axes], -1)
    assert np.isfinite(coordinates).all()
    seeds = np.array([int(tag.split("_")[0].split("-")[1]) for tag in columns])
    pd.DataFrame({"structure_index": range(125), "sample": tags, "archive_member": names}).to_csv(
        OUT / f"{gene}_recovered_seed_mapping.csv", index=False
    )
    return coordinates, seeds


ORIGINAL_LOAD = positional.load


def main():
    OUT.mkdir(exist_ok=True)
    for module in (seed_icc, seed_vs_random_split, positional):
        module.OUT = OUT
    positional.load = recover_load
    seed_icc.main()
    seed_vs_random_split.main()
    positional.main()
    for name in ("seed_icc", "seed_vs_random_split"):
        pd.testing.assert_frame_equal(
            pd.read_csv(OUT / f"{name}.csv"),
            pd.read_csv(OUT.parent / "results" / f"{name}.csv"),
        )
    result = pd.read_csv(OUT / "seed_vs_random_positional.csv")
    assert set(result.gene) == set(positional.GENES) and len(result) == 12
    (OUT / "provenance.json").write_text(json.dumps({
        "date": "2026-10-18", "genes": positional.GENES,
        "repetitions": positional.REPS, "random_seed": 20261017,
        "conformations_per_side": 25,
        "comparison": "5 disjoint whole seeds per side versus disjoint random conformations",
        "coordinate_frame": "core_fit", "residue_scope": "core residues",
        "recovered_archives": {gene: str(path) for gene, path in ARCHIVES.items()},
        "recovery": "sorted seed-sample structure members, zero-based index, as original ingestion",
        "plddt_and_icc_saved_results_reproduced_exactly": True,
    }, indent=2) + "\n")
    print(f"All six genes completed. Results: {OUT}", flush=True)


if __name__ == "__main__":
    main()
