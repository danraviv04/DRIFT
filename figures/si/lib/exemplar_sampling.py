"""The canonical N-structure subsample of one variant and its WT, residues aligned.

Shared by the Fig. 1e panel (ks_sequence_panel.py) and the gene-wide residue null
(analyses/msh6_case_study/fig1e_near_pair/gene_wide.py), so both draw exactly the
structures the canonical KS-route table (ks_meanfree per_variant_seed.csv.gz) used:
the N=90 subsample at seed_index 0, from canonical_analysis's own `stable_rng`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "analyses" / "msh6_case_study" / "ks_meanfree"))
from canonical_analysis import apply_row_block, row_block, stable_rng  # noqa: E402
from metric_analysis.pipeline.paths import tagged  # noqa: E402
from drift import ensemble_shift  # noqa: E402
from drift.protein_config import load_continuous_configs  # noqa: E402
from drift.structure_draws import discover_wt_csv_sources  # noqa: E402

PER_VARIANT = (
    REPO / "analyses" / "msh6_case_study" / "ks_meanfree" / "results" / tagged("canonical")
    / "per_variant_seed.csv.gz"
)
#: seed_index 0 of the canonical ten
CANONICAL_SEED = 20260829


def sampled_pair(record, size, seed):
    """WT and mutant pLDDT matrices under the canonical subsample, residues aligned."""
    gene = record["protein"]
    cfg = load_continuous_configs((gene,))[gene]["structures"]
    if cfg.get("wt_csv_expected_sources") is not None:
        sources = discover_wt_csv_sources(
            cfg["read_root"], cfg["wt_csv_glob"],
            expected_sources=cfg["wt_csv_expected_sources"])
        if Path(record["wt_source_file"]).resolve() != Path(sources[0]).resolve():
            raise ValueError(f"{gene}: manifest WT differs from the configured reference")
    wt, mut = [ensemble_shift._read_plddt_matrix(
        record[key], None, seed=seed, key="ratio_exemplar",
        expected_residues=cfg.get("plddt_expected_residues"))
        for key in ("wt_source_file", "source_file")]
    if not wt.index.equals(mut.index):
        raise ValueError(f"{gene}: WT and mutant residue row labels differ")
    block = row_block(record)
    context = f"{record['protein']} {record['mutation_code']}"
    residues = (np.arange(block[1] - block[0], dtype=float) if block is not None
                else np.asarray(wt.index, dtype=float)) + 1.0
    wt = apply_row_block(wt.to_numpy(float), block, context=context)
    mut = apply_row_block(mut.to_numpy(float), block, context=context)
    if wt.shape[0] != mut.shape[0]:
        raise SystemExit(f"residue mismatch: {wt.shape[0]} != {mut.shape[0]}")

    code = record["mutation_code"]
    wt_order = stable_rng(seed, record["protein"], code, "wt").permutation(wt.shape[1])
    mut_order = stable_rng(seed, record["protein"], code, "mutant").permutation(mut.shape[1])
    n = min(size, wt.shape[1], mut.shape[1])

    w = wt[:, wt_order[:n]]
    m = mut[:, mut_order[:n]]
    valid = np.isfinite(w).all(axis=1) & np.isfinite(m).all(axis=1)
    resid = residues[valid]   # 1-based, subunit-local when a row block is present
    return resid, w[valid], m[valid], int(n)
