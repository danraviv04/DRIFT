"""Shared reader for the representative-structure comparisons.

artifact_frames streams one coordinate cache artifact by artifact; n90_comparison.py
and analyses/coordinates/s1_matched_draws use it. ROOT, OUT and GENES are shared
constants.
"""
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GENES = ['VHL', 'POLE', 'PTEN', 'MSH2', 'MSH6', 'PIK3CA']
XYZ = ['x_aligned', 'y_aligned', 'z_aligned']


def artifact_frames(path, artifacts):
    """Stream an artifact-sorted parquet once, keeping at most one carry group."""
    dataset = ds.dataset(path)
    columns = ['artifact', 'structure_index', 'sample', 'chain', 'resid'] + XYZ
    scanner = dataset.scanner(columns=columns,
                              filter=ds.field('artifact').isin(artifacts), batch_size=250000)
    carry = None
    seen = set()
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        names = frame.artifact.astype(str)
        last = names.iloc[-1]
        carry = frame[names.eq(last)]
        for artifact, group in frame[~names.eq(last)].groupby('artifact', observed=True, sort=False):
            assert artifact not in seen, 'cache must be artifact-contiguous'
            seen.add(artifact)
            yield str(artifact), group
    if carry is not None:
        artifact = str(carry.artifact.iloc[0])
        assert artifact not in seen
        yield artifact, carry

