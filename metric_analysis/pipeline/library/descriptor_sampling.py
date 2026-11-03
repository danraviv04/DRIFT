"""Canonical descriptor sampling, shared by experiments and fusion sweeps."""
from functools import lru_cache
import zlib
import numpy as np
import pandas as pd
from .descriptor_metrics import descriptor_values


@lru_cache(maxsize=8)
def load_matrix(path: str) -> np.ndarray:
    return pd.read_csv(path, index_col=0).to_numpy(dtype=float)


def row_block(record: dict) -> tuple[int, int] | None:
    """The half-open pLDDT row block this dataset's descriptor is restricted to.

    A subunit dataset (MSH2, MSH6) is one contiguous block of the modelled
    complex's matrix; every other dataset uses its whole matrix and gets None.
    The bounds travel with the cohort row, so no caller has to know which
    datasets are subunits.
    """
    start = record.get("plddt_row_range_start")
    stop = record.get("plddt_row_range_stop")
    if start is None or stop is None or pd.isna(start) or pd.isna(stop):
        return None
    return int(start), int(stop)


def apply_row_block(matrix: np.ndarray, block: tuple[int, int] | None,
                    *, context: str = "") -> np.ndarray:
    if block is None:
        return matrix
    start, stop = block
    if stop > matrix.shape[0]:
        raise ValueError(
            f"{context}: row block {block} exceeds the {matrix.shape[0]}-row matrix"
        )
    return matrix[start:stop]


def stable_rng(base_seed: int, *parts: object) -> np.random.Generator:
    """Order-independent RNG for one variant/seed/side."""
    token = "|".join(map(str, parts)).encode()
    return np.random.default_rng(
        np.random.SeedSequence([int(base_seed), zlib.crc32(token)])
    )


def analyse_variant(
    record: dict[str, object],
    *,
    sizes: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    block = row_block(record)
    context = f"{record['protein']} {record['mutation_code']}"
    wt = apply_row_block(load_matrix(str(record["wt_source_file"])), block, context=context)
    mutant = apply_row_block(load_matrix(str(record["source_file"])), block, context=context)
    if wt.shape[0] != mutant.shape[0]:
        raise ValueError(
            f"{record['protein']} {record['mutation_code']}: residue mismatch "
            f"{wt.shape[0]} != {mutant.shape[0]}"
        )

    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds):
        wt_order = stable_rng(seed, record["protein"], record["mutation_code"], "wt").permutation(wt.shape[1])
        mutant_order = stable_rng(seed, record["protein"], record["mutation_code"], "mutant").permutation(mutant.shape[1])
        for requested_size in sizes:
            size = min(requested_size, wt.shape[1], mutant.shape[1])
            values = descriptor_values(
                wt[:, wt_order[:size]], mutant[:, mutant_order[:size]]
            )
            rows.append({
                "protein": record["protein"],
                "mutation_code": record["mutation_code"],
                "mutation_type": record["mutation_type"],
                "label": int(record["label"]),
                "label_provenance": record["label_provenance"],
                "source_family": record["source_family"],
                "seed_index": seed_index,
                "seed": int(seed),
                "requested_size": int(requested_size),
                "ensemble_size": int(size),
                **values,
            })
    return rows


