"""How many subsampling seeds each ensemble size deserves.

The D1 ensemble-size sweep draws N structures without replacement from a finite
pool of M structures per variant. The sampling variance of a variant's D1 value
therefore carries a finite population correction:

    Var(N)  proportional to  (1/N) * (M - N) / (M - 1)

which is 1 at N=1 and exactly 0 at N=M. Running the same number of subsampling
seeds at every N consequently over-measures the large-N end -- where every draw
overlaps every other draw by construction -- and under-measures N=1, which is
the random-single-structure baseline the ensemble argument rests on.

This module turns that observation into a concrete seed budget. Repeats are
allocated proportional to Var(N) subject to a floor, so the Monte-Carlo standard
error of each reported mean is roughly equal across sizes, at the same total
cost as a uniform allocation.

Nesting is preserved: the seed list for a large N is a *prefix* of the seed list
for a small N. Sizes therefore remain paired on the seeds they share, and the
extra small-N seeds only sharpen small-N point estimates. Cross-size paired
comparisons must be restricted to the shared prefix -- `paired_seed_subset`
returns it.

Caveat worth keeping in view: the seed-to-seed spread is subsampling noise only.
It is already the smallest uncertainty in the analysis and is not a substitute
for variant- or gene-level resampling. Equalising it is worth doing because it
is free, not because it is the uncertainty that matters.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FLOOR = 5


def finite_population_variance_factor(size: int, pool_size: int) -> float:
    """Relative sampling variance of a mean over `size` draws from `pool_size`.

    Normalised so that size=1 gives 1.0. Returns 0.0 once the draw exhausts the
    pool: every seed then sees the same structures and repeats measure nothing.
    """
    n = int(size)
    m = int(pool_size)
    if n <= 0:
        raise ValueError("size must be a positive integer")
    if m <= 1:
        return 0.0
    if n >= m:
        return 0.0
    return (1.0 / n) * (m - n) / (m - 1.0)


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Apportion `total` integer units across `weights` without drift."""
    if total <= 0 or weights.sum() <= 0:
        return np.zeros(len(weights), dtype=int)
    exact = weights / weights.sum() * total
    base = np.floor(exact).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(exact - base))
        base[order[:remainder]] += 1
    return base


def allocate_seeds(
    sizes,
    pool_size,
    *,
    total_budget: int | None = None,
    floor: int = DEFAULT_FLOOR,
) -> dict[int, int]:
    """Seeds per ensemble size, proportional to sampling variance.

    `total_budget` is the total number of (size, seed) D1 builds; it defaults to
    the uniform-allocation cost `floor * len(sizes)` rounded up to whatever the
    caller already spends, so the reallocation is budget-neutral by default.

    The floor exists because a standard deviation estimated from R seeds is
    itself uncertain by roughly 1/sqrt(2(R-1)) -- about +/-50% at R=3 and +/-24%
    at R=10. Below ~5 the reported spread stops meaning much, even where the
    true sampling variance is negligible.
    """
    sizes = [int(s) for s in sizes]
    if not sizes:
        raise ValueError("sizes must be non-empty")
    if any(s <= 0 for s in sizes):
        raise ValueError("sizes must be positive integers")
    if floor < 1:
        raise ValueError("floor must be at least 1")

    ordered = sorted(set(sizes))
    if total_budget is None:
        total_budget = floor * len(ordered)
    total_budget = int(total_budget)
    minimum = floor * len(ordered)
    if total_budget < minimum:
        raise ValueError(
            f"total_budget={total_budget} cannot honour floor={floor} across "
            f"{len(ordered)} sizes (needs at least {minimum})"
        )

    weights = np.array(
        [finite_population_variance_factor(s, pool_size) for s in ordered],
        dtype=float,
    )
    extra = _largest_remainder(weights, total_budget - minimum)
    counts = {s: int(floor + e) for s, e in zip(ordered, extra)}

    # Variance is non-increasing in N, so the allocation is too. Enforce it
    # explicitly: largest-remainder rounding can otherwise transpose neighbours
    # whose weights are nearly equal, which would break the prefix property the
    # seed plan relies on.
    running = max(counts.values())
    for s in ordered:
        running = min(running, counts[s])
        counts[s] = running
    return {s: counts[s] for s in sizes}


def seed_plan(sizes, seeds, allocation: dict[int, int]) -> dict[int, list[int]]:
    """Map each size to the seeds it runs -- always a prefix of `seeds`.

    Taking prefixes is what keeps sizes paired: any two sizes share the seeds up
    to the smaller of their two counts, and on those seeds the nested-sampling
    guarantee still holds.
    """
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    plan = {}
    for size in sizes:
        want = int(allocation[int(size)])
        if want > len(seeds):
            raise ValueError(
                f"size {size} needs {want} seeds but only {len(seeds)} were supplied"
            )
        plan[int(size)] = seeds[:want]
    return plan


def paired_seed_subset(plan: dict[int, list[int]]) -> list[int]:
    """Seeds common to every size -- the only ones valid for paired cross-N tests."""
    if not plan:
        return []
    shared = set(plan[next(iter(plan))])
    for seeds in plan.values():
        shared &= set(seeds)
    order = plan[max(plan, key=lambda k: len(plan[k]))]
    return [s for s in order if s in shared]


def allocation_table(
    sizes, pool_size, *, total_budget: int | None = None, floor: int = DEFAULT_FLOOR
) -> pd.DataFrame:
    """Reportable view of the allocation, with the precision it buys."""
    allocation = allocate_seeds(
        sizes, pool_size, total_budget=total_budget, floor=floor
    )
    uniform = (total_budget or floor * len(set(sizes))) / len(set(sizes))
    rows = []
    for size in sorted(set(int(s) for s in sizes)):
        var_factor = finite_population_variance_factor(size, pool_size)
        repeats = allocation[size]
        rows.append(
            {
                "size": size,
                "pool_size": int(pool_size),
                "frac_of_pool": size / float(pool_size),
                "variance_factor_vs_n1": var_factor,
                "relative_sd_vs_n1": float(np.sqrt(var_factor)),
                "seeds_uniform": uniform,
                "seeds_allocated": repeats,
                # Monte-Carlo SE of the reported mean, in units of the N=1
                # per-draw SD: equalising this column is the point of the plan.
                "rel_se_of_mean": float(np.sqrt(var_factor / repeats)),
                "sd_estimate_rel_error": float(1.0 / np.sqrt(2.0 * max(repeats - 1, 1))),
            }
        )
    return pd.DataFrame(rows)


def observed_pool_sizes(plddt_root: str | Path = "PLDDTs") -> pd.DataFrame:
    """Structures available per variant, per gene, read from the pLDDT CSVs.

    The pool size drives the whole allocation and differs by gene (N=90 is 90%
    of the TP53 pool but 56% of the FOXA1 pool), so it belongs in the methods
    table rather than being assumed constant.
    """
    root = Path(plddt_root)
    rows = []
    for gene_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        counts = []
        for csv_path in sorted(gene_dir.glob("*.csv")):
            with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
                header = handle.readline()
            if not header.strip():
                continue
            counts.append(header.count(",") or 0)
        if not counts:
            continue
        counts = np.array(counts, dtype=int)
        rows.append(
            {
                "gene": gene_dir.name,
                "n_variant_files": int(counts.size),
                "pool_min": int(counts.min()),
                "pool_median": int(np.median(counts)),
                "pool_max": int(counts.max()),
            }
        )
    return pd.DataFrame(rows)


def describe_plan(sizes, seeds, pool_size, *, total_budget=None, floor=DEFAULT_FLOOR):
    """Allocation table plus the concrete per-size seed lists."""
    allocation = allocate_seeds(
        sizes, pool_size, total_budget=total_budget, floor=floor
    )
    plan = seed_plan(sizes, seeds, allocation)
    table = allocation_table(
        sizes, pool_size, total_budget=total_budget, floor=floor
    )
    table["seeds"] = table["size"].map(
        lambda s: ",".join(str(x) for x in plan[int(s)])
    )
    return table, plan, paired_seed_subset(plan)
