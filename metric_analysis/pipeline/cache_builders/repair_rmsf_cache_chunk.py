#!/usr/bin/env python3
"""
Aligned C-alpha coordinate cache (US-align onto the WT), built in parallel chunks.

Splits the work across N independent processes, each handling a disjoint subset of
archives. Safe to run concurrently: each process writes its own chunk parquet, and
--merge combines them into the dataset's cache. run_all.sh (extract, X2) runs it for
every coordinate dataset; it needs the raw archives (DRIFT_ARCHIVE_ROOT).

Usage
-----
# Build POLE's cache with 4 parallel chunks:
    python -m metric_analysis.pipeline.cache_builders.repair_rmsf_cache_chunk --protein POLE --chunk 0 --total-chunks 4
    python -m metric_analysis.pipeline.cache_builders.repair_rmsf_cache_chunk --protein POLE --chunk 1 --total-chunks 4
    python -m metric_analysis.pipeline.cache_builders.repair_rmsf_cache_chunk --protein POLE --chunk 2 --total-chunks 4
    python -m metric_analysis.pipeline.cache_builders.repair_rmsf_cache_chunk --protein POLE --chunk 3 --total-chunks 4

# Merge after all chunks finish:
    python -m metric_analysis.pipeline.cache_builders.repair_rmsf_cache_chunk --protein POLE --merge --total-chunks 4

# Dry run (2 archives only):
    python -m metric_analysis.pipeline.cache_builders.repair_rmsf_cache_chunk --protein POLE --chunk 0 --total-chunks 4 --max-archives 2

# Datasets: any entry of metric_analysis/protein_configs.json with a coordinate cache
# (POLE, PTEN, VHL, PIK3CA, MutS_alpha, MutS_alpha_MSH6_GT in the paper)
"""

import argparse
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
except ImportError:
    raise RuntimeError("BioPython is required: pip install biopython")

from metric_analysis.pipeline.config.protein_config import get_protein_config, supported_proteins
from metric_analysis.pipeline.labels import is_wt_artifact as _is_wt_artifact

# ── Per-protein configuration (metric_analysis/protein_configs.json) ───────────

REQUIRED_CONFIG_KEYS = ["archive_root", "cache_dir", "protein_chains", "wt_hint"]

USALIGN_CMD     = "USalign"
MAX_WORKERS     = 10
MAX_ATTEMPTS    = 3
RETRY_SLEEP_SEC = 2.0
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")
MEMBER_SUFFIXES  = (".cif", ".mmcif", ".pdb", ".ent")

# ── Parsers ────────────────────────────────────────────────────────────────────

_CIF_PARSER = MMCIFParser(QUIET=True)
_PDB_PARSER = PDBParser(QUIET=True, PERMISSIVE=True)

# ── Shared helpers (ported from notebook) ─────────────────────────────────────
# _is_wt_artifact now imported from labels.py (see top of file).

def _strip_archive_suffix(path_or_name):
    name = Path(path_or_name).name
    for suffix in ARCHIVE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def split_tar_structure_ref(ref):
    ref = str(ref)
    if not ref.startswith("tar://") or "::" not in ref:
        return None, None
    archive, member = ref[len("tar://"):].split("::", 1)
    return Path(archive), member


def make_tar_structure_ref(archive_path, member_name):
    return f"tar://{Path(archive_path).resolve()}::{member_name}"


def structure_key_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["artifact", "structure_index"])
    if "artifact" not in df.columns or "structure_index" not in df.columns:
        return pd.DataFrame(columns=["artifact", "structure_index"])
    out = df[["artifact", "structure_index"]].dropna().copy()
    out["artifact"]        = out["artifact"].astype(str)
    out["structure_index"] = out["structure_index"].astype(int)
    return out.drop_duplicates().reset_index(drop=True)


def anti_join_missing(expected_df, cached_df):
    if expected_df is None or expected_df.empty:
        return (expected_df.iloc[0:0].copy() if expected_df is not None else pd.DataFrame())
    expected_keys = structure_key_df(expected_df)
    cached_keys   = structure_key_df(cached_df)
    if cached_keys.empty:
        return expected_df.copy()
    tmp = expected_keys.merge(
        cached_keys.assign(_cached=True), on=["artifact", "structure_index"], how="left"
    )
    missing_keys = tmp.loc[tmp["_cached"].isna(), ["artifact", "structure_index"]]
    return expected_df.merge(missing_keys, on=["artifact", "structure_index"], how="inner")


def batch_extract_archive(archive_path, member_names, dest_dir):
    dest_dir  = Path(dest_dir)
    needed    = set(member_names)
    extracted = {}
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile() or member.name not in needed:
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            safe_name = member.name.replace("/", "_")
            dest_path = dest_dir / safe_name
            dest_path.write_bytes(fobj.read())
            extracted[member.name] = dest_path
    return extracted


def _reextract_single_member(archive_path, member_name, dest_path):
    """Re-extract one member fresh, overwriting dest_path in place. Used on parse
    retry — a corrupted/truncated local copy from a one-off concurrent-I/O hiccup
    during the original batch extraction would otherwise fail identically on every
    retry, since retrying only re-ran USalign/parsing on the same bad local file.
    """
    with tarfile.open(archive_path, "r:*") as tf:
        fobj = tf.extractfile(member_name)
        if fobj is None:
            raise FileNotFoundError(f"member not found on re-extract: {member_name}")
        Path(dest_path).write_bytes(fobj.read())


def _run_usalign(query_path, ref_path, out_prefix):
    cmd  = [USALIGN_CMD, str(query_path), str(ref_path), "-mol", "prot", "-o", str(out_prefix)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"USalign failed: {(proc.stderr or '').strip()[:300]}")
    for candidate in [
        Path(str(out_prefix) + ".pdb"),
        Path(str(out_prefix)).with_suffix(".pdb"),
        Path(str(out_prefix) + ".cif"),
        Path(str(out_prefix)).with_suffix(".cif"),
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"USalign produced no output at {out_prefix}*")


def _parse_pdb_ca_coords(pdb_path, chain_ids=None):
    pdb_path  = Path(pdb_path)
    chain_set = set(chain_ids) if chain_ids else None
    if pdb_path.suffix.lower() in (".cif", ".mmcif"):
        # Tolerant mmCIF CA extraction via MMCIF2Dict instead of MMCIFParser.
        # MMCIFParser.get_structure() builds every chain up front and raises
        # PDBConstructionException on duplicate atoms/residues. USalign's aligned
        # output collapses the dsDNA strands into the protein chain, so their
        # shared residue numbering (both strands start at 1) collides and kills
        # the parse before the CA filter runs. MMCIF2Dict just reads the
        # _atom_site columns into lists (no uniqueness enforcement); filtering to
        # CA atoms keeps protein alpha carbons only (nucleotides have no CA).
        d = MMCIF2Dict(str(pdb_path))
        n         = len(d["_atom_site.label_atom_id"])
        atom_id   = d["_atom_site.label_atom_id"]
        group     = d.get("_atom_site.group_PDB", ["ATOM"] * n)
        comp      = d["_atom_site.label_comp_id"]
        asym      = d.get("_atom_site.auth_asym_id", d["_atom_site.label_asym_id"])
        seq       = d.get("_atom_site.auth_seq_id",  d["_atom_site.label_seq_id"])
        icodes    = d.get("_atom_site.pdbx_PDB_ins_code", ["?"] * n)
        xs, ys, zs = d["_atom_site.Cartn_x"], d["_atom_site.Cartn_y"], d["_atom_site.Cartn_z"]
        models    = d.get("_atom_site.pdbx_PDB_model_num", ["1"] * n)
        first_model = models[0]

        rows = []
        for i in range(n):
            if models[i] != first_model:        # first model only (matches prior behaviour)
                continue
            if atom_id[i] != "CA" or group[i] != "ATOM":
                continue
            if chain_set and str(asym[i]) not in chain_set:
                continue
            ic = icodes[i]
            ic = "" if ic in ("?", ".") else ic.strip()
            try:
                rows.append({
                    "chain": str(asym[i]), "resid": int(seq[i]),
                    "icode": ic, "resname": comp[i],
                    "x": float(xs[i]), "y": float(ys[i]), "z": float(zs[i]),
                })
            except (ValueError, IndexError):
                continue
        return pd.DataFrame(rows)
    rows = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain = line[21].strip()
            if chain_set and chain not in chain_set:
                continue
            try:
                rows.append({
                    "chain": chain, "resid": int(line[22:26].strip()),
                    "icode": line[26].strip(), "resname": line[17:20].strip(),
                    "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def _compute_cache_rows(local_path, row, ref_path, align_tmp_dir, protein_chains):
    my_tmp = Path(tempfile.mkdtemp(dir=align_tmp_dir))
    try:
        aligned = _run_usalign(Path(local_path), ref_path, my_tmp / "aligned")
        ca_df   = _parse_pdb_ca_coords(aligned, protein_chains)
        if ca_df.empty:
            raise ValueError("no CA coordinates in USalign output")
        # Allele/label_class/structure_file are per-artifact/per-structure attributes,
        # not per-residue ones — the RMSF/RMSD sweep re-derives them from `artifact`
        # rather than reading them off this cache, so they're dropped here rather than
        # duplicated onto every one of a structure's ~100s-1000s of residue rows.
        return [
            {
                "artifact":        row.artifact,
                "structure_index": int(row.structure_index),
                "chain":           mrow["chain"],
                "resid":           int(mrow["resid"]),
                "icode":           mrow["icode"],
                "resname":         mrow["resname"],
                "x_aligned":       float(mrow["x"]),
                "y_aligned":       float(mrow["y"]),
                "z_aligned":       float(mrow["z"]),
            }
            for _, mrow in ca_df.iterrows()
        ]
    finally:
        shutil.rmtree(my_tmp, ignore_errors=True)


def _worker(local_path, row, ref_path, align_tmp_dir, protein_chains, archive_path=None, member_name=None):
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _compute_cache_rows(local_path, row, ref_path, align_tmp_dir, protein_chains)
        except Exception as e:
            last_err = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SEC)
                if archive_path is not None and member_name is not None:
                    try:
                        _reextract_single_member(archive_path, member_name, local_path)
                    except Exception:
                        pass  # if re-extraction itself fails, the next attempt just fails again with a clear error
    raise last_err


def _atomic_save_parquet(df, path):
    tmp = Path(str(path) + f".{os.getpid()}.tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _atomic_save_csv(df, path):
    tmp = Path(str(path) + f".{os.getpid()}.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ── Auto-build helpers ─────────────────────────────────────────────────────────

def build_structure_index(cfg, tag=""):
    """Scan all archives and produce a structure_file_index.csv. Saves to cache_dir."""
    archive_root = cfg["archive_root"]
    cache_dir    = cfg["cache_dir"]
    prefix       = cfg["prefix"]

    archives = sorted({
        p.resolve()
        for suffix in ARCHIVE_SUFFIXES
        for p in archive_root.glob(f"**/*{suffix}")
        if p.is_file()
    })
    if not archives:
        raise FileNotFoundError(f"No archives found under {archive_root}")

    # Checkpoint after every archive: indexing a large batch reads every archive
    # end to end (hours on the external drive), and a killed run used to lose it
    # all. The partial file has its own name, so nothing mistakes it for the index.
    partial_csv = cache_dir / f"{prefix}_structure_file_index.partial.csv"
    rows, done = [], set()
    if partial_csv.exists():
        prev = pd.read_csv(partial_csv)
        rows = prev.to_dict("records")
        done = set(prev["artifact"].astype(str))
        print(f"{tag} Resuming index: {len(done)} archives already indexed in {partial_csv.name}", flush=True)

    print(f"{tag} Building structure index from {len(archives)} archives in {archive_root}", flush=True)
    t0   = time.time()
    for i, archive_path in enumerate(archives, 1):
        artifact = _strip_archive_suffix(archive_path)
        if artifact in done:
            continue
        members  = []
        try:
            with tarfile.open(archive_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name
                    if name.startswith("._") or "/._" in name:
                        continue
                    if name.lower().endswith(MEMBER_SUFFIXES):
                        members.append(name)
        except Exception as e:
            print(f"{tag} WARNING: could not read {archive_path.name}: {e}", flush=True)
            continue
        for j, member_name in enumerate(sorted(members)):
            rows.append({
                "artifact":         artifact,
                "Allele":           artifact,
                "label_class":      "UNK",
                "structure_index":  int(j),
                "structure_path":   make_tar_structure_ref(archive_path, member_name),
                "structure_archive": str(archive_path),
                "structure_member":  member_name,
                "structure_file":    Path(member_name).name,
            })
        _atomic_save_csv(pd.DataFrame(rows), partial_csv)
        if i == 1 or i % 10 == 0 or i == len(archives):
            print(
                f"{tag} [INDEX] {i}/{len(archives)} {artifact} members={len(members)} "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    df      = pd.DataFrame(rows)
    out_csv = cache_dir / f"{prefix}_structure_file_index.csv"
    _atomic_save_csv(df, out_csv)
    partial_csv.unlink(missing_ok=True)
    print(f"{tag} Structure index saved: {out_csv} shape={df.shape}", flush=True)
    return df


def extract_wt_reference(cfg, structure_index_df, tag=""):
    """Extract one WT structure CIF to use as USalign reference. Saves to cache_dir."""
    cache_dir = cfg["cache_dir"]
    wt_hint   = cfg["wt_hint"]
    dest      = cache_dir / "_rmsf_wt_reference.cif"

    # Find WT rows — try exact hint match first, then regex fallback
    wt_rows = structure_index_df[
        structure_index_df["artifact"].astype(str) == wt_hint
    ].copy()
    if wt_rows.empty:
        wt_rows = structure_index_df[
            structure_index_df["artifact"].apply(_is_wt_artifact)
        ].copy()
    if wt_rows.empty:
        raise ValueError(
            f"No WT artifact found in structure index for hint={wt_hint!r}. "
            "Check wt_hint in protein_configs.json."
        )

    best_artifact = wt_rows.groupby("artifact").size().sort_values(ascending=False).index[0]
    ref_row       = wt_rows[wt_rows["artifact"] == best_artifact].iloc[0]
    archive_path, member_name = split_tar_structure_ref(str(ref_row["structure_path"]))

    print(f"{tag} Extracting WT reference: {best_artifact} / {member_name}", flush=True)
    tmp = Path(tempfile.mkdtemp())
    try:
        extracted = batch_extract_archive(archive_path, [member_name], tmp)
        if not extracted:
            raise FileNotFoundError(f"member not found: {member_name}")
        src = next(iter(extracted.values()))
        shutil.copy2(src, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"{tag} WT reference saved: {dest}", flush=True)
    return dest


# ── Repair ─────────────────────────────────────────────────────────────────────

def repair_chunk(protein: str, chunk: int, total_chunks: int, max_archives: int | None = None):
    cfg    = get_protein_config(protein, REQUIRED_CONFIG_KEYS)
    tag    = f"[{protein} chunk {chunk}/{total_chunks-1}]"
    prefix = cfg["prefix"]

    cache_dir      = cfg["cache_dir"]
    protein_chains = cfg["protein_chains"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_csv      = cache_dir / f"{prefix}_structure_file_index.csv"
    main_parquet   = cache_dir / f"{prefix}_rmsf_aligned_ca_coordinate_cache.parquet"
    main_csv_gz    = cache_dir / f"{prefix}_rmsf_aligned_ca_coordinate_cache.csv.gz"
    main_errors    = cache_dir / f"{prefix}_rmsf_aligned_ca_coordinate_cache_errors.csv"
    wt_ref         = cache_dir / "_rmsf_wt_reference.cif"
    chunk_parquet  = cache_dir / f"{prefix}_rmsf_cache_chunk_{chunk}.parquet"
    chunk_errors   = cache_dir / f"{prefix}_rmsf_cache_chunk_{chunk}_errors.csv"

    print(f"{tag} Starting RMSF cache repair for {protein}", flush=True)

    # Load or build structure index
    if index_csv.exists():
        structure_index_df = pd.read_csv(index_csv)
        print(f"{tag} Structure index loaded: {structure_index_df.shape}", flush=True)
    else:
        print(f"{tag} Structure index not found — building from archives...", flush=True)
        structure_index_df = build_structure_index(cfg, tag=tag)

    # Load existing cache — only the two key columns are ever needed here (the
    # missing-structures check below reduces both sides to artifact+structure_index
    # anyway), so this avoids materializing the full coordinate cache (unpruned,
    # e.g. ~16.8GB in memory for MutS_alpha's 809MB-on-disk parquet) just to answer
    # "what's missing" — real risk when N parallel chunks each do this at once.
    if main_parquet.exists():
        # This is a per-residue cache (many rows per structure) — drop_duplicates
        # immediately collapses it to one row per (artifact, structure_index),
        # since that's the only granularity anti_join_missing ever needs.
        main_cache_df = pd.read_parquet(main_parquet, columns=["artifact", "structure_index"]).drop_duplicates()
        print(f"{tag} Main cache loaded (key columns only, deduped): {main_cache_df.shape}", flush=True)
    else:
        main_cache_df = pd.DataFrame()
        print(f"{tag} No main cache found — treating all structures as missing", flush=True)

    # Load previous chunk output (enables crash-resume)
    if chunk_parquet.exists():
        prev_chunk_df = pd.read_parquet(chunk_parquet)
        print(f"{tag} Resuming from previous chunk output: {prev_chunk_df.shape}", flush=True)
    else:
        prev_chunk_df = pd.DataFrame()

    # Identify missing structures
    already_cached = (
        pd.concat([main_cache_df, prev_chunk_df], ignore_index=True)
        if not main_cache_df.empty and not prev_chunk_df.empty
        else (main_cache_df if not main_cache_df.empty else prev_chunk_df)
    )
    missing_df = anti_join_missing(structure_index_df, already_cached)
    print(f"{tag} Missing structures (across all chunks): {len(missing_df):,}", flush=True)

    if missing_df.empty:
        print(f"{tag} Nothing to do — all structures already cached.", flush=True)
        return

    # Extract WT reference if needed
    if not wt_ref.exists():
        print(f"{tag} WT reference not found — extracting...", flush=True)
        extract_wt_reference(cfg, structure_index_df, tag=tag)
    ref_path = wt_ref

    # Assign archives to this chunk via interleaved indexing
    missing_df = missing_df.copy()
    missing_df["_archive_key"] = missing_df["structure_path"].apply(
        lambda p: str(split_tar_structure_ref(str(p))[0] or p)
    )
    all_archive_keys = sorted(missing_df["_archive_key"].unique())
    my_archive_keys  = set(all_archive_keys[chunk::total_chunks])
    work_df          = missing_df[missing_df["_archive_key"].isin(my_archive_keys)].copy()
    archive_groups   = list(work_df.groupby("_archive_key", sort=False))

    if max_archives is not None:
        archive_groups = archive_groups[:max_archives]
        print(f"{tag} Dry-run: capped at {max_archives} archives", flush=True)

    print(
        f"{tag} My archives: {len(my_archive_keys)}/{len(all_archive_keys)}  "
        f"structures: {len(work_df):,}  workers/archive: {MAX_WORKERS}",
        flush=True,
    )

    out_rows   = []
    error_rows = []
    n_archives = len(archive_groups)
    pbar       = tqdm(total=sum(len(g) for _, g in archive_groups), desc=f"{protein} chunk {chunk}", unit="struct")

    for archive_idx, (archive_key, archive_rows) in enumerate(archive_groups, 1):
        archive_path = Path(archive_key)
        member_names = []
        for row in archive_rows.itertuples(index=False):
            _, member = split_tar_structure_ref(str(row.structure_path))
            member_names.append(member)

        tmp_extract = Path(tempfile.mkdtemp())
        tmp_align   = Path(tempfile.mkdtemp())
        try:
            if all(m is not None for m in member_names):
                extracted = batch_extract_archive(archive_path, member_names, tmp_extract)
            else:
                extracted = {
                    str(r.structure_path): Path(str(r.structure_path))
                    for r in archive_rows.itertuples(index=False)
                }

            work_items = []
            for row in archive_rows.itertuples(index=False):
                _, member = split_tar_structure_ref(str(row.structure_path))
                key        = member if member is not None else str(row.structure_path)
                local_path = extracted.get(key)
                if local_path is None:
                    error_rows.append({
                        "artifact":        row.artifact,
                        "structure_index": int(row.structure_index),
                        "structure_path":  row.structure_path,
                        "error":           f"member not found: {key}",
                        "n_attempts":      1,
                    })
                    pbar.update(1)
                else:
                    work_items.append((row, local_path, key))

            n_workers = min(MAX_WORKERS, len(work_items)) if work_items else 1
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                # Only pass archive_path/key for tar-backed members (re-extract-on-retry
                # doesn't apply to the non-tar fallback branch above, where local_path is
                # already the original file, not a temp copy worth re-fetching).
                futures = {
                    pool.submit(
                        _worker, lp, row, ref_path, tmp_align, protein_chains,
                        archive_path if all(m is not None for m in member_names) else None,
                        key,
                    ): row
                    for row, lp, key in work_items
                }
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        out_rows.extend(future.result())
                    except Exception as e:
                        print(f"{tag} ERROR {row.artifact}[{row.structure_index}]: {repr(e)[:200]}", flush=True)
                        error_rows.append({
                            "artifact":        row.artifact,
                            "structure_index": int(row.structure_index),
                            "structure_path":  row.structure_path,
                            "error":           repr(e)[:400],
                            "n_attempts":      MAX_ATTEMPTS,
                        })
                    pbar.update(1)

        finally:
            shutil.rmtree(tmp_extract, ignore_errors=True)
            shutil.rmtree(tmp_align,   ignore_errors=True)

        # Checkpoint after every archive
        if out_rows:
            new_df        = pd.DataFrame(out_rows)
            checkpoint_df = (
                pd.concat([prev_chunk_df, new_df], ignore_index=True)
                if not prev_chunk_df.empty else new_df
            )
            checkpoint_df = checkpoint_df.drop_duplicates(
                subset=["artifact", "structure_index", "chain", "resid", "icode"], keep="last"
            ).reset_index(drop=True)
            _atomic_save_parquet(checkpoint_df, chunk_parquet)
            out_rows = []
            prev_chunk_df = checkpoint_df
            print(
                f"{tag} [CHECKPOINT] archive {archive_idx}/{n_archives}: {archive_path.name}  "
                f"chunk_rows={len(checkpoint_df):,}  errors={len(error_rows):,}",
                flush=True,
            )

        if error_rows:
            _atomic_save_csv(pd.DataFrame(error_rows), chunk_errors)

    pbar.close()
    print(f"{tag} Done. New rows: {len(out_rows):,}  errors: {len(error_rows):,}", flush=True)


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_chunks(protein: str, total_chunks: int):
    cfg    = get_protein_config(protein, REQUIRED_CONFIG_KEYS)
    prefix = cfg["prefix"]

    cache_dir    = cfg["cache_dir"]
    main_parquet = cache_dir / f"{prefix}_rmsf_aligned_ca_coordinate_cache.parquet"
    main_csv_gz  = cache_dir / f"{prefix}_rmsf_aligned_ca_coordinate_cache.csv.gz"
    main_errors  = cache_dir / f"{prefix}_rmsf_aligned_ca_coordinate_cache_errors.csv"

    print(f"[{protein}] Merging {total_chunks} chunks into main cache...", flush=True)

    existing_df = pd.DataFrame()
    if main_parquet.exists():
        existing_df = pd.read_parquet(main_parquet)
        print(f"  Existing main cache: {existing_df.shape}", flush=True)

    chunk_dfs = []
    for c in range(total_chunks):
        p = cache_dir / f"{prefix}_rmsf_cache_chunk_{c}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            print(f"  Chunk {c}: {df.shape}", flush=True)
            chunk_dfs.append(df)
        else:
            print(f"  Chunk {c}: NOT FOUND (skipping)", flush=True)

    if not chunk_dfs:
        print("No chunk outputs found. Nothing to merge.", flush=True)
        return

    all_new  = pd.concat(chunk_dfs, ignore_index=True)
    combined = (
        pd.concat([existing_df, all_new], ignore_index=True) if not existing_df.empty else all_new
    )
    combined = combined.drop_duplicates(
        subset=["artifact", "structure_index", "chain", "resid", "icode"], keep="last"
    ).reset_index(drop=True)
    print(f"  Combined after dedup: {combined.shape}", flush=True)

    if main_parquet.exists():
        backup = main_parquet.with_name(main_parquet.stem + "_pre_merge_backup.parquet")
        shutil.copy2(main_parquet, backup)
        print(f"  Backup saved: {backup}", flush=True)

    _atomic_save_parquet(combined, main_parquet)
    combined.to_csv(main_csv_gz, index=False)
    print(f"  Main cache updated: {main_parquet} shape={combined.shape}", flush=True)

    # Consolidate error CSVs — remove entries that are now cached
    error_parts = []
    for c in range(total_chunks):
        p = cache_dir / f"{prefix}_rmsf_cache_chunk_{c}_errors.csv"
        if p.exists():
            error_parts.append(pd.read_csv(p))
    if main_errors.exists():
        error_parts.append(pd.read_csv(main_errors))

    if error_parts:
        all_errors  = pd.concat(error_parts, ignore_index=True)
        cached_keys = structure_key_df(combined)
        still_err   = structure_key_df(all_errors).merge(
            cached_keys.assign(_c=True), on=["artifact", "structure_index"], how="left"
        )
        still_err   = still_err.loc[still_err["_c"].isna(), ["artifact", "structure_index"]]
        unresolved  = all_errors.merge(still_err, on=["artifact", "structure_index"], how="inner")
        unresolved  = unresolved.drop_duplicates(subset=["artifact", "structure_index"], keep="last")
        if not unresolved.empty:
            _atomic_save_csv(unresolved, main_errors)
            print(f"  Unresolved errors: {unresolved.shape}", flush=True)
        else:
            if main_errors.exists():
                main_errors.unlink()
            print("  No unresolved errors — error CSV removed.", flush=True)

    print(f"\n[{protein}] Merge complete! Re-run the notebook to use the updated cache.", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel RMSF coordinate cache repair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--protein",      required=True, choices=supported_proteins(REQUIRED_CONFIG_KEYS), help="Protein to repair")
    parser.add_argument("--chunk",        type=int,  help="0-indexed chunk number to process")
    parser.add_argument("--total-chunks", type=int,  default=4, help="Total parallel chunks (default: 4)")
    parser.add_argument("--merge",        action="store_true", help="Merge completed chunks into main cache")
    parser.add_argument("--workers",      type=int,  default=MAX_WORKERS, help=f"USalign threads per archive (default: {MAX_WORKERS})")
    parser.add_argument("--max-archives", type=int,  default=None, help="Cap archives per chunk (use 2 for dry-run)")
    args = parser.parse_args()

    MAX_WORKERS = args.workers

    if args.merge:
        merge_chunks(args.protein, args.total_chunks)
    elif args.chunk is not None:
        if not (0 <= args.chunk < args.total_chunks):
            parser.error(f"--chunk must be 0..{args.total_chunks - 1}")
        repair_chunk(args.protein, args.chunk, args.total_chunks, max_archives=args.max_archives)
    else:
        parser.print_help()
