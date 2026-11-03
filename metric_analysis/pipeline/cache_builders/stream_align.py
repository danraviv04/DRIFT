"""Aligned C-alpha coordinate cache in ONE sequential pass over the archives.

``repair_rmsf_cache_chunk`` reads the drive twice (an index pass that unpacks
every archive just to list its members, then the alignment pass), and its
workers either read the drive concurrently (on the USB drive two readers drop to
a few MB/s each) or leave it idle while they align. Here:

- one prefetch thread copies the NEXT archive to local disk while the current
  one is aligned, so the drive always has exactly one sequential reader;
- members are listed from the local copy, so there is no index pass;
- USalign runs on a thread pool (each call is its own process), reusing
  ``repair_rmsf_cache_chunk._worker`` so every structure is aligned and parsed
  exactly as the existing cache builder does it;
- each archive's rows are saved as ``<cache_dir>/stream_parts/<artifact>.parquet``
  the moment it finishes, and a rerun skips archives that already have a part.

Two uses:

* a fresh cache (MSH6 G:T): structure_index enumerates the archive's sorted
  structure members, as ``build_structure_index`` does, top-level model included;
* an extension (TP53 seeds 21-25): ``--rename`` maps the extension archive's name
  onto the cached artifact, ``--member-regex`` keeps only the seed samples, and
  ``--offset-from`` numbers the new structures after the ones already cached.

``--exclude-cached-from <index.csv>`` additionally skips every seed-sample the
old structure index already lists for that artifact, so an extension can never
re-add a structure whatever the seed numbering (the TP53 WT's new seeds are
41-45, the variants' 21-25).

``--plddt-out DIR`` also writes the pipeline's per-archive pLDDT table
(``fast_plddt``, all predictions, identical values) from the same local copy, so
a pLDDT re-extraction and a coordinate extension share one read of the drive.

``--merge`` then writes the parts into the main cache parquet (streamed row group
by row group, so the 187M-row TP53 cache is never loaded whole), after backing
the old one up and refusing any (artifact, structure_index) already present.

    python -m metric_analysis.pipeline.cache_builders.stream_align \\
        --archive-root $DRIFT_ARCHIVE_ROOT/TP53_extensions \\
        --cache-dir metric_analysis/structure_metrics/TP53/ensemble_aware/cache_ext_seeds21-25 \\
        --ref metric_analysis/structure_metrics/TP53/ensemble_aware/cache/_rmsf_wt_reference.cif \\
        --chains A,B,C,D --rename '_seeds_\\d+-\\d+_data$' --member-regex 'seed-\\d+_sample-\\d+' \\
        --offset-from metric_analysis/structure_metrics/TP53/ensemble_aware/cache/TP53_rmsf_aligned_ca_coordinate_cache.parquet
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import tarfile
import tempfile
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from metric_analysis.pipeline.cache_builders import repair_rmsf_cache_chunk as R
from metric_analysis.pipeline.library.fast_plddt import archive_plddt_table
from metric_analysis.pipeline.library.structure_metrics import canonical_artifact_stem

Row = namedtuple("Row", "artifact structure_index")
COPY_BUF = 16 << 20


def artifact_name(archive: Path, rename: str | None) -> str:
    name = R._strip_archive_suffix(archive)
    return re.sub(rename, "", name) if rename else name


def _prefetch(archives, local_dir: Path, q: queue.Queue, stop: threading.Event):
    """Copy archives to local disk one at a time, in order. Single drive reader."""
    for a in archives:
        if stop.is_set():
            break
        dest = local_dir / a.name
        t0 = time.time()
        for attempt in range(4):
            try:
                with open(a, "rb") as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, COPY_BUF)
                q.put((a, dest, time.time() - t0, None))
                break
            except Exception as e:                   # noqa: BLE001 -- reported per archive
                dest.unlink(missing_ok=True)
                # A drive that was unplugged and replugged (or woke from sleep) leaves a
                # stale handle: wait and reopen. Give up after ~1.5 min.
                if attempt == 3 or stop.is_set():
                    q.put((a, None, time.time() - t0, e))
                    break
                time.sleep(30)
    q.put(None)


SAMPLE_RE = re.compile(r"seed-\d+_sample-\d+")


def _members(local_archive: Path, member_regex: str | None, exclude_tags=frozenset()):
    with tarfile.open(local_archive, "r:*") as tf:
        names = sorted(m.name for m in tf.getmembers()
                       if m.isfile() and m.name.lower().endswith(R.MEMBER_SUFFIXES)
                       and not (m.name.startswith("._") or "/._" in m.name))
    if member_regex:
        names = [n for n in names if re.search(member_regex, n)]
    if exclude_tags:
        names = [n for n in names if (t := SAMPLE_RE.search(n)) is None or t[0] not in exclude_tags]
    return names


def _cached_tags(index_csv: Path | None) -> dict:
    """artifact -> set of seed-sample tags an old structure index already lists."""
    if not index_csv:
        return {}
    idx = pd.read_csv(index_csv, usecols=["artifact", "structure_member"])
    idx["tag"] = idx.structure_member.str.extract(r"(seed-\d+_sample-\d+)")[0]
    return {a: frozenset(g.tag.dropna()) for a, g in idx.groupby("artifact")}


def _offsets(parquet: Path | None) -> dict:
    """artifact -> next free structure_index in an existing cache (streamed)."""
    if not parquet:
        return {}
    pf = pq.ParquetFile(parquet)
    nxt: dict = {}
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=["artifact", "structure_index"]).to_pandas()
        for a, m in t.groupby("artifact").structure_index.max().items():
            nxt[a] = max(nxt.get(a, 0), int(m) + 1)
    return nxt


def run(args):
    cache_dir = Path(args.cache_dir)
    parts = cache_dir / "stream_parts"
    parts.mkdir(parents=True, exist_ok=True)
    ref = Path(args.ref)
    if not ref.is_file():
        raise SystemExit(f"reference not found: {ref}")
    chains = args.chains.split(",")
    archives = sorted(p for p in Path(args.archive_root).glob("*")
                      if p.name.endswith(R.ARCHIVE_SUFFIXES) and not p.name.startswith("._"))
    if args.limit:
        archives = archives[: args.limit]
    plddt_dir = Path(args.plddt_out) if args.plddt_out else None
    if plddt_dir:
        plddt_dir.mkdir(parents=True, exist_ok=True)
    plddt_chains = (args.plddt_chains or args.chains).split(",")
    plddt_csv = lambda a: plddt_dir / f"{canonical_artifact_stem(a)}_plddt_values.csv"  # noqa: E731
    coord_done = lambda a: (parts / f"{artifact_name(a, args.rename)}.parquet").exists()  # noqa: E731
    todo = [a for a in archives if not coord_done(a) or (plddt_dir and not plddt_csv(a).exists())]
    cached = _cached_tags(Path(args.exclude_cached_from)) if args.exclude_cached_from else {}
    offsets = _offsets(Path(args.offset_from)) if args.offset_from else {}
    if args.offset_from:
        unknown = sorted({artifact_name(a, args.rename) for a in todo} - set(offsets))
        if unknown:     # aligned in full (see "fresh" below), not as an extension
            print(f"[stream] {len(unknown)} archives are not in the cache and get every "
                  f"structure aligned: {unknown}", flush=True)
    print(f"[stream] {len(archives)} archives, {len(archives) - len(todo)} already done, "
          f"{len(todo)} to do; workers={args.workers}", flush=True)

    errors_csv = cache_dir / "stream_errors.csv"
    local_dir = Path(tempfile.mkdtemp(prefix="stream_align_", dir=args.tmp))
    q: queue.Queue = queue.Queue(maxsize=1)          # at most one archive waiting locally
    stop = threading.Event()
    th = threading.Thread(target=_prefetch, args=(todo, local_dir, q, stop), daemon=True)
    th.start()
    t_start, n_done, n_struct = time.time(), 0, 0
    # Kept in cache_dir, NOT in the pLDDT folder: every .csv there is read as a variant.
    shapes_csv = cache_dir / "stream_plddt_shapes.csv"
    shapes = pd.read_csv(shapes_csv).to_dict("records") if shapes_csv.exists() else []
    try:
        while (item := q.get()) is not None:
            src, local, t_copy, err = item
            art = artifact_name(src, args.rename)
            t0 = time.time()
            errs = []
            if err is not None and not Path(args.archive_root).is_dir():
                # Drive unplugged / unmounted (sleep, cable). Stop rather than log every
                # remaining archive as failed; a rerun resumes from this archive.
                raise SystemExit(f"[stream] {args.archive_root} is gone; stopped at {art}. "
                                 "Reconnect the drive and rerun: finished archives are kept.")
            if err is not None:
                errs.append({"artifact": art, "structure_index": -1, "error": f"copy: {err!r}"[:400]})
                rows = []
            else:
                ext = Path(tempfile.mkdtemp(dir=local_dir))
                aln = Path(tempfile.mkdtemp(dir=local_dir))
                try:
                    if plddt_dir and not plddt_csv(src).exists():
                        tab = archive_plddt_table(local, plddt_chains)
                        tmp_csv = plddt_csv(src).with_suffix(f".{os.getpid()}.tmp")
                        tab.to_csv(tmp_csv, index=True)
                        os.replace(tmp_csv, plddt_csv(src))
                        shapes.append({"artifact": canonical_artifact_stem(src), "rows": tab.shape[0],
                                       "predictions": tab.shape[1]})
                        R._atomic_save_csv(pd.DataFrame(shapes), shapes_csv)
                    # An ensemble the old cache never had (TP53: the 12 variants of the
                    # output_gen_only/Folders batch) gets every structure aligned, numbered
                    # from 0 with the top model included, exactly as a fresh cache build.
                    fresh = bool(args.offset_from) and art not in offsets
                    if coord_done(src):
                        names = []
                    elif fresh:
                        names = _members(local, None)
                    else:
                        names = _members(local, args.member_regex, cached.get(art, frozenset()))
                    got = R.batch_extract_archive(local, names, ext) if names else {}
                    base = offsets.get(art, 0)
                    rows = []
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        futs = {pool.submit(R._worker, got[n], Row(art, base + j), ref, aln, chains): (j, n)
                                for j, n in enumerate(names) if n in got}
                        for j, n in enumerate(names):
                            if n not in got:
                                errs.append({"artifact": art, "structure_index": base + j,
                                             "error": f"member not extracted: {n}"})
                        for f in as_completed(futs):
                            j, n = futs[f]
                            try:
                                rows.extend(f.result())
                            except Exception as e:           # noqa: BLE001
                                errs.append({"artifact": art, "structure_index": base + j,
                                             "error": repr(e)[:400]})
                except Exception as e:                   # noqa: BLE001 -- one archive, keep going
                    errs.append({"artifact": art, "structure_index": -1, "error": f"archive: {e!r}"[:400]})
                    rows = []
                finally:
                    shutil.rmtree(ext, ignore_errors=True)
                    shutil.rmtree(aln, ignore_errors=True)
                    local.unlink(missing_ok=True)
            if rows:
                df = pd.DataFrame(rows).sort_values(["structure_index", "chain", "resid"], kind="stable")
                R._atomic_save_parquet(df, parts / f"{art}.parquet")
                n_struct += df.structure_index.nunique()
            if errs:
                prev = pd.read_csv(errors_csv) if errors_csv.exists() else pd.DataFrame()
                R._atomic_save_csv(pd.concat([prev, pd.DataFrame(errs)], ignore_index=True), errors_csv)
            n_done += 1
            el = time.time() - t_start
            print(f"[stream] {n_done}/{len(todo)} {art}: copy {t_copy:.0f}s, align {time.time() - t0:.0f}s, "
                  f"{len(rows) and df.structure_index.nunique()} structures, {len(errs)} errors; "
                  f"elapsed {el / 60:.1f} min, ETA {el / n_done * (len(todo) - n_done) / 60:.0f} min",
                  flush=True)
    finally:
        stop.set()
        shutil.rmtree(local_dir, ignore_errors=True)
    print(f"[stream] done: {n_done} archives, {n_struct} structures in {(time.time() - t_start) / 60:.1f} min",
          flush=True)


def merge(args):
    """Append stream_parts into the main cache parquet without loading it whole."""
    cache_dir = Path(args.cache_dir)
    parts = sorted((cache_dir / "stream_parts").glob("*.parquet"))
    if not parts:
        raise SystemExit("no parts to merge")
    main = Path(args.main)
    new = pa.concat_tables([pq.read_table(p) for p in parts])
    new_keys = set(zip(new.column("artifact").to_pylist(), new.column("structure_index").to_pylist()))
    print(f"[merge] {len(parts)} parts, {new.num_rows:,} rows, {len(new_keys):,} structures", flush=True)
    if main.exists():
        pf = pq.ParquetFile(main)
        schema = pf.schema_arrow
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i, columns=["artifact", "structure_index"])
            clash = new_keys & set(zip(t.column("artifact").to_pylist(), t.column("structure_index").to_pylist()))
            if clash:
                raise SystemExit(f"{len(clash)} structures already in {main.name}, e.g. {sorted(clash)[:3]}")
        new = new.select(schema.names).cast(schema)
        # Always back up the CURRENT main, under a fresh name when an older backup exists,
        # and rebuild from main itself. Rebuilding from a pre-existing backup (the old code)
        # silently dropped everything merged since that backup was taken: a second merge
        # into TP53 would have erased the seeds 21-25 extension of 10-15.
        backup = main.with_name(main.stem + "_pre_stream_backup.parquet")
        if backup.exists():
            backup = main.with_name(main.stem + f"_pre_stream_backup_{time.strftime('%Y%m%d-%H%M%S')}.parquet")
        shutil.copy2(main, backup)
        print(f"[merge] backup: {backup.name}", flush=True)
        tmp = main.with_name(main.name + f".{os.getpid()}.tmp")
        with pq.ParquetWriter(tmp, schema) as w:
            src = pq.ParquetFile(main)
            for i in range(src.metadata.num_row_groups):
                w.write_table(src.read_row_group(i))
            w.write_table(new, row_group_size=1_000_000)
        os.replace(tmp, main)
    else:
        pq.write_table(new, main, row_group_size=1_000_000)
    print(f"[merge] wrote {main} ({pq.ParquetFile(main).metadata.num_rows:,} rows)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-root", required=False)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ref", help="WT reference CIF (the existing cache's, for an extension)")
    ap.add_argument("--chains", default="A")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--rename", help="regex removed from the archive name to give the artifact")
    ap.add_argument("--member-regex", help="keep only structure members matching this")
    ap.add_argument("--offset-from", help="existing cache parquet: number new structures after its own")
    ap.add_argument("--exclude-cached-from", help="old structure index CSV: skip seed-samples it lists")
    ap.add_argument("--plddt-out", help="also write per-archive pLDDT CSVs (fast_plddt) here")
    ap.add_argument("--plddt-chains", help="chains for the pLDDT table (default: --chains)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tmp", help="local scratch for archive copies (default: system temp)")
    ap.add_argument("--merge", action="store_true", help="append stream_parts into --main")
    ap.add_argument("--main", help="main cache parquet to write (with --merge)")
    args = ap.parse_args(argv)
    if args.merge:
        if not args.main:
            ap.error("--merge needs --main")
        merge(args)
    else:
        if not (args.archive_root and args.ref):
            ap.error("--archive-root and --ref are required")
        run(args)


if __name__ == "__main__":
    main()
