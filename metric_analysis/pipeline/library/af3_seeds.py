"""AF3 seed numbers of predictions, and the per-gene seed cap (2026-10-20).

Every ensemble in the paper is 25 AF3 seeds x 5 samples = 125 predictions. Eight
POLE drivers (D275A, D275G, E277G, P286H, P286L, P286R, P286S, T278K) were predicted
with seeds 1-32; their pLDDT inputs are 25-seed copies (PLDDTs/POLE_25seed/, see
experiments/annotation_stages), but the coordinate caches hold all 160 structures. A
coordinate reader that takes POLE structures must drop seeds above the cap, so
the coordinates describe the same 125 predictions as the pLDDT matrices.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SEED_SAMPLE = re.compile(r"seed-(\d+)_sample-(\d+)")

#: highest AF3 seed number an ensemble may use, per gene; genes not listed are uncapped
MAX_AF3_SEED = {"POLE": 25}


def seed_numbers(texts) -> np.ndarray:
    """AF3 seed number in each text (a sample tag or a structure path), NaN if none."""
    found = pd.Series(list(texts), dtype="object").astype(str).str.extract(SEED_SAMPLE)[0]
    return pd.to_numeric(found, errors="coerce").to_numpy(float)


def within_seed_cap(texts, gene: str) -> np.ndarray:
    """Boolean mask: True where the seed is within the gene's cap.

    A text with no seed tag is kept: it is either AF3's duplicated top model, which
    the readers drop by their own rule, or an artifact without seed naming.
    """
    seeds = seed_numbers(texts)
    cap = MAX_AF3_SEED.get(gene)
    if cap is None:
        return np.ones(len(seeds), dtype=bool)
    return ~(seeds > cap)
