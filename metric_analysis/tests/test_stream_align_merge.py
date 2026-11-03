"""stream_align --merge must append to the CURRENT cache, also on a second merge.

Regression (2026-10-15): with a backup left by an earlier merge, the merge rebuilt the
cache from that backup, silently dropping everything merged since (it would have erased
TP53's seeds 21-25 extension when the regenerated WT was added).

    PYTHONPATH=. python metric_analysis/tests/test_stream_align_merge.py
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq

from metric_analysis.pipeline.cache_builders.stream_align import merge


def _rows(artifact, n):
    return pd.DataFrame({"artifact": artifact, "structure_index": range(n), "chain": "A", "resid": 1,
                         "icode": "", "resname": "ALA", "x_aligned": 0.0, "y_aligned": 0.0, "z_aligned": 0.0})


def _merge_part(cache_dir, main, artifact, n):
    parts = cache_dir / "stream_parts"
    parts.mkdir(parents=True, exist_ok=True)
    for p in parts.glob("*.parquet"):
        p.unlink()
    _rows(artifact, n).to_parquet(parts / f"{artifact}.parquet", index=False)
    merge(SimpleNamespace(cache_dir=str(cache_dir), main=str(main)))


def test_second_merge_keeps_the_first():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        main = tmp / "X_rmsf_aligned_ca_coordinate_cache.parquet"
        _rows("OLD", 4).to_parquet(main, index=False)
        _merge_part(tmp / "ext1", main, "EXT", 3)
        _merge_part(tmp / "ext2", main, "NEWWT", 2)
        got = pd.read_parquet(main).groupby("artifact").size().to_dict()
        assert got == {"OLD": 4, "EXT": 3, "NEWWT": 2}, got
        backups = sorted(p.name for p in tmp.glob("*_pre_stream_backup*.parquet"))
        assert len(backups) == 2, backups
        # the second backup holds the state just before the second merge
        assert pq.ParquetFile(tmp / backups[-1]).metadata.num_rows in (7,), backups


def test_clash_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        main = tmp / "X_rmsf_aligned_ca_coordinate_cache.parquet"
        _rows("OLD", 4).to_parquet(main, index=False)
        try:
            _merge_part(tmp / "ext", main, "OLD", 2)
        except SystemExit as e:
            assert "already in" in str(e)
            return
        raise AssertionError("a clashing merge was accepted")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
