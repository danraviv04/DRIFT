#!/usr/bin/env python
"""Validate one af3_jobs batch on $DRIFT_ARCHIVE_ROOT against the inputs that generated it.

    python \
        analyses/provenance/elements_drive_audit/validate_batch.py \
        --zip muts_jsons/MutS_alpha_MSH6_GT_Mutations.zip --drive-dir MSH6_GT

The expected set is every input JSON in the batch zip, so the af3_jobs manifest
is the reference, not the canonical cohort (a new batch is not in it yet). Each
archive is read ONCE, start to end, and these are checked on the way:

  readable       stat + open succeed. On this NTFS drive a name can be listed
                 and still fail to open (the 10-06 PTEN damage).
  gzip_ok        the whole stream decompresses and the gzip CRC matches, i.e. the
                 archive is not truncated or corrupt.
  name_ok        the AF3 job JSON inside names the job the input asked for.
  entities_ok    every chain in the job JSON (protein, DNA, ligand) equals the
                 input: same ids, sequences, CCD codes.
  seeds_ok       modelSeeds in the job JSON equal the input's.
  ensemble_ok    one seed-<s>_sample-<k> model per (input seed x sample), and the
                 seed set equals the input's.
  models_ok      EVERY model CIF (top-level + all seed samples) folds exactly the
                 input chains: residue sequence per chain from the CA / C1' atoms,
                 plus the ligand per chain. This is what was folded, not requested.
  success        _SUCCESS present, no _FAILED marker.

After the scan, variants are compared to the batch's WT as folded: the mutated
chain must differ from the WT chain at exactly the coded position, with the coded
WT and mutant residues, and every other chain must be identical. Byte-identical
archives (md5) and identical top models are reported as duplicates.

The `_data.json` decoys of the two-stage cluster run (0 bytes, or full size and
null-filled; see experiments/cohort_drive_validation) are skipped and counted.
Results append to results/<drive-dir>_archives.jsonl as they finish, so a rerun
skips what is already done (--fresh to start over).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

DRIVE = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT"))
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "results"

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}
NUC = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "A": "A", "C": "C", "G": "G", "U": "U"}
SEED_SAMPLE = re.compile(r"seed-(\d+)_sample-(\d+)")
CODE = re.compile(r"([A-Z])(\d+)([A-Z])$")


# --------------------------------------------------------------------------- inputs
def expected_entities(sequences: list) -> dict[str, tuple]:
    """chain id -> (kind, sequence or ccd codes), from an AF3 `sequences` list."""
    out = {}
    for entry in sequences:
        for kind, body in entry.items():
            ids = body.get("id")
            if kind == "ligand":
                value = tuple(body.get("ccdCodes") or [body.get("smiles")])
            else:
                value = body.get("sequence", "")
            for cid in (ids if isinstance(ids, list) else [ids]):
                out[str(cid)] = (kind, value)
    return out


def load_inputs(zip_path: Path) -> dict[str, dict]:
    """job name -> input JSON, for every job in the batch zip."""
    with zipfile.ZipFile(zip_path) as z:
        jobs = {}
        for member in z.namelist():
            if member.endswith(".json"):
                payload = json.loads(z.read(member))
                jobs[payload["name"]] = payload
    return jobs


# --------------------------------------------------------------------------- reading
class _Hashing(io.RawIOBase):
    """Raw reader that md5s the compressed bytes as they stream past."""

    def __init__(self, handle):
        self.handle, self.md5, self.nbytes = handle, hashlib.md5(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        n = self.handle.readinto(buffer)
        if n:
            self.md5.update(memoryview(buffer)[:n])
            self.nbytes += n
        return n


def cif_signature(text: str) -> dict[str, tuple]:
    """chain id -> (kind, sequence or ligand codes) as folded, from one model CIF."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("_atom_site."))
    cols = []
    j = start
    while lines[j].startswith("_atom_site."):
        cols.append(lines[j].split(".", 1)[1].strip())
        j += 1
    ix = {c: k for k, c in enumerate(cols)}
    g, atom, comp, asym, seq = (ix["group_PDB"], ix["label_atom_id"], ix["label_comp_id"],
                                ix["label_asym_id"], ix["label_seq_id"])
    protein: dict[str, dict[int, str]] = {}
    nucleic: dict[str, dict[int, str]] = {}
    ligand: dict[str, list[str]] = {}
    for line in lines[j:]:
        if not line or line[0] == "#" or line[0] == "_" or line.startswith("loop_"):
            break
        f = line.split()
        name = f[atom].strip('"')
        if f[g] == "HETATM":
            codes = ligand.setdefault(f[asym], [])
            if f[comp] not in codes:
                codes.append(f[comp])
        elif name == "CA" and f[comp] in AA3:
            protein.setdefault(f[asym], {})[int(f[seq])] = AA3[f[comp]]
        elif name == "C1'" and f[comp] in NUC:
            nucleic.setdefault(f[asym], {})[int(f[seq])] = NUC[f[comp]]
    out = {}
    for chain, residues in protein.items():
        out[chain] = ("protein", "".join(residues[k] for k in sorted(residues)))
    for chain, residues in nucleic.items():
        out[chain] = ("nucleic", "".join(residues[k] for k in sorted(residues)))
    for chain, codes in ligand.items():
        out[chain] = ("ligand", tuple(codes))
    return out


def _comparable(entities: dict[str, tuple]) -> dict[str, tuple]:
    """Fold dna/rna into one 'nucleic' kind -- a CIF cannot tell them apart by atom."""
    return {c: (("nucleic" if k in ("dna", "rna") else k), v) for c, (k, v) in entities.items()}


def validate_archive(path_str: str, job: dict) -> dict:
    path = Path(path_str)
    rec = {"archive": path.name, "job": job["name"]}
    try:
        rec["bytes"] = os.stat(path).st_size
        raw = open(path, "rb", buffering=0)
    except OSError as exc:
        rec.update(readable=False, error=f"{type(exc).__name__}: {exc}")
        return rec
    rec["readable"] = True

    expected = _comparable(expected_entities(job["sequences"]))
    seeds_in = sorted(int(s) for s in job["modelSeeds"])
    hashing = _Hashing(raw)
    signatures: dict[tuple, int] = {}
    seed_samples: dict[str, set] = {}
    conf = {"ptm": [], "iptm": [], "ranking_score": [], "fraction_disordered": []}
    rec.update(json_found=0, json_decoys=0, n_seed_cifs=0, n_top_cifs=0, n_clash=0,
               success=False, failed_marker=False, ranking_csv=False)
    t0 = time.time()
    try:
        gz = gzip.GzipFile(fileobj=io.BufferedReader(hashing, 1 << 20))
        with tarfile.open(fileobj=gz, mode="r|") as tar:
            for member in tar:
                name = member.name
                base = name.rsplit("/", 1)[-1]
                if base == "_SUCCESS":
                    rec["success"] = True
                if base == "_FAILED":
                    rec["failed_marker"] = True
                if not member.isfile():
                    continue
                if base.endswith("_ranking_scores.csv"):
                    rec["ranking_csv"] = True
                if base.endswith("_data.json"):
                    handle = tar.extractfile(member)
                    head = handle.read(512) if member.size else b""
                    if not head.lstrip(b"\x00\r\n\t ")[:1] == b"{":
                        rec["json_decoys"] += 1
                        continue
                    payload = json.loads(head + handle.read())
                    rec["json_found"] += 1
                    rec["json_name"] = payload.get("name")
                    rec["name_ok"] = payload.get("name") == job["name"]
                    rec["seeds_ok"] = sorted(int(s) for s in payload.get("modelSeeds", [])) == seeds_in
                    rec["entities_ok"] = _comparable(expected_entities(payload.get("sequences", []))) == expected
                    del payload
                elif base.endswith("_model.cif"):
                    text = tar.extractfile(member).read().decode("utf-8", "replace")
                    sig = cif_signature(text)
                    key = tuple(sorted(sig.items()))
                    signatures[key] = signatures.get(key, 0) + 1
                    m = SEED_SAMPLE.search(base)
                    if m:
                        rec["n_seed_cifs"] += 1
                        run_dir = name.split("/seed-")[0]
                        seed_samples.setdefault(run_dir, set()).add((int(m.group(1)), int(m.group(2))))
                    else:
                        rec["n_top_cifs"] += 1
                        rec["top_cif_sha"] = hashlib.sha1(text.encode()).hexdigest()
                        rec["_top_chains"] = {c: v for c, (k, v) in sig.items() if k == "protein"}
                elif base.endswith("_summary_confidences.json") and SEED_SAMPLE.search(base):
                    s = json.loads(tar.extractfile(member).read())
                    for k in conf:
                        if isinstance(s.get(k), (int, float)):
                            conf[k].append(float(s[k]))
                    rec["n_clash"] += int(bool(s.get("has_clash")))
        while gz.read(1 << 22):          # drain to EOF: gzip verifies its CRC here
            pass
        rec["gzip_ok"] = True
    except Exception as exc:              # truncated, corrupt, unreadable mid-stream
        rec["gzip_ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        raw.close()
    rec["md5"] = hashing.md5.hexdigest()
    rec["bytes_read"] = hashing.nbytes
    rec["seconds"] = round(time.time() - t0, 1)

    # ensemble completeness
    rec["n_run_dirs"] = len(seed_samples)
    pairs = set().union(*seed_samples.values()) if seed_samples else set()
    seeds_found = sorted({s for s, _ in pairs})
    per_seed = [sum(1 for s2, _ in pairs if s2 == s) for s in seeds_found]
    rec["n_seeds"] = len(seeds_found)
    rec["samples_per_seed"] = f"{min(per_seed)}-{max(per_seed)}" if per_seed else ""
    n_samples = max(per_seed) if per_seed else 0
    rec["ensemble_ok"] = (seeds_found == seeds_in and len(pairs) == rec["n_seed_cifs"]
                          and min(per_seed or [0]) == n_samples and n_samples > 0)
    # every folded model must equal the input
    expected_key = tuple(sorted(expected.items()))
    rec["n_model_signatures"] = len(signatures)
    rec["models_ok"] = list(signatures) == [expected_key]
    rec["n_models_mismatched"] = sum(n for k, n in signatures.items() if k != expected_key)
    for k, values in conf.items():
        rec[f"{k}_median"] = round(median(values), 4) if values else None
    rec["ptm_min"] = round(min(conf["ptm"]), 4) if conf["ptm"] else None
    return rec


# --------------------------------------------------------------------------- report
def compare_to_wt(rows: list[dict], jobs: dict[str, dict]) -> None:
    """Mutated chain differs from the folded WT at exactly the coded position."""
    wt = next((r for r in rows if r["job"].endswith("_WT") and r.get("_top_chains")), None)
    if wt is None:
        print("  !! no readable WT archive -- variant-vs-WT check skipped")
        return
    for r in rows:
        if r is wt or not r.get("_top_chains"):
            continue
        m = CODE.search(r["job"])
        diffs = []
        for chain, seq in r["_top_chains"].items():
            ref = wt["_top_chains"].get(chain, "")
            if len(seq) != len(ref):
                diffs.append(f"{chain}:len{len(seq)}!={len(ref)}")
                continue
            diffs += [f"{chain}:{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(ref, seq)) if a != b]
        r["diff_vs_wt"] = ";".join(diffs)
        r["vs_wt_ok"] = bool(m) and len(diffs) == 1 and diffs[0].split(":")[1] == m.group(0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="the batch's input zip under muts_jsons/")
    ap.add_argument("--drive-dir", required=True, help="folder under $DRIFT_ARCHIVE_ROOT")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results")
    ap.add_argument("--limit", type=int, default=0, help="read at most N new archives (smoke test)")
    args = ap.parse_args(argv)

    folder = DRIVE / args.drive_dir
    if not folder.is_dir():
        raise SystemExit(f"not found: {folder} -- is $DRIFT_ARCHIVE_ROOT mounted?")
    jobs = load_inputs(REPO / args.zip)
    listed = {n for n in os.listdir(folder) if n.endswith(".tar.gz")}
    want = {f"{name}_data.tar.gz": name for name in jobs}
    missing = sorted(set(want) - listed)
    extra = sorted(listed - set(want))
    print(f"[batch] {args.zip}: {len(jobs)} jobs | {folder}: {len(listed)} archives listed")
    print(f"[batch] missing {len(missing)} | extra {len(extra)}")
    for n in missing:
        print(f"   missing: {n}")
    for n in extra:
        print(f"   extra:   {n}")

    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{args.drive_dir}_archives.jsonl"
    done = {}
    if log.exists() and not args.fresh:
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done[r["archive"]] = r
    elif log.exists():
        log.unlink()
    todo = sorted(a for a in want if a in listed and a not in done)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[batch] {len(done)} already validated, {len(todo)} to read, {args.workers} workers\n")

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool, open(log, "a") as sink:
        futures = {pool.submit(validate_archive, str(folder / a), jobs[want[a]]): a for a in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            done[r["archive"]] = r
            sink.write(json.dumps(r) + "\n")
            sink.flush()
            flags = "ok" if all(r.get(k) for k in ("gzip_ok", "name_ok", "entities_ok", "seeds_ok",
                                                    "ensemble_ok", "models_ok", "success")) else "FAIL"
            print(f"  [{i}/{len(todo)}] {time.time() - t0:5.0f}s {flags:4s} {r['archive']} "
                  f"({r.get('seconds', '?')}s, {r.get('n_seed_cifs')} models)", flush=True)

    import pandas as pd
    rows = [done[a] for a in sorted(done) if a in want]
    compare_to_wt(rows, jobs)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_csv(OUT / f"{args.drive_dir}_validation.csv", index=False)

    checks = ["readable", "gzip_ok", "success", "name_ok", "entities_ok", "seeds_ok",
              "ensemble_ok", "models_ok", "vs_wt_ok"]
    print(f"\n[batch] {args.drive_dir}: {len(want)} expected, {len(missing)} missing, "
          f"{len(extra)} extra, {len(df)} read")
    for c in checks:
        if c in df:
            v = df[c].dropna()
            print(f"  {c:12s} {int(v.astype(bool).sum())}/{len(v)}")
    print(f"  failed_marker {int(df.get('failed_marker', pd.Series(dtype=bool)).fillna(False).sum())}"
          f" | json decoys skipped {int(df.get('json_decoys', pd.Series(dtype=float)).fillna(0).sum())}"
          f" | run dirs per archive {sorted(df['n_run_dirs'].dropna().unique().tolist())}")
    print(f"  models per archive: seed {sorted(df['n_seed_cifs'].dropna().unique().tolist())}"
          f", top {sorted(df['n_top_cifs'].dropna().unique().tolist())}")
    dup_md5 = df[df.md5.duplicated(keep=False)] if "md5" in df else df.iloc[0:0]
    dup_top = df[df.top_cif_sha.notna() & df.top_cif_sha.duplicated(keep=False)] if "top_cif_sha" in df else df.iloc[0:0]
    print(f"  duplicate archives (md5) {len(dup_md5)} | duplicate top models {len(dup_top)}")
    for c in ("ptm_median", "iptm_median", "ranking_score_median"):
        if c in df and df[c].notna().any():
            print(f"  {c:22s} min {df[c].min():.3f}  median {df[c].median():.3f}  max {df[c].max():.3f}")
    if "n_clash" in df:
        print(f"  models with has_clash  {int(df.n_clash.sum())} in {int((df.n_clash > 0).sum())} archives")
    ok = df[[c for c in checks[:-1] if c in df]].fillna(False).astype(bool).all(axis=1)
    ok &= df.json_found.eq(1) & ~df.failed_marker.fillna(False).astype(bool)
    if "vs_wt_ok" in df:            # the WT itself has no variant-vs-WT row
        ok &= df.vs_wt_ok.fillna(df.job.str.endswith("_WT")).astype(bool)
    bad = df[~ok]
    if len(bad):
        cols = [c for c in ["archive", "error"] + checks + ["diff_vs_wt", "n_models_mismatched",
                                                            "samples_per_seed"] if c in bad]
        print(f"\n  *** {len(bad)} archive(s) failing a check ***")
        print(bad[cols].to_string(index=False))
    print(f"\n[batch] wrote {(OUT / f'{args.drive_dir}_validation.csv').relative_to(REPO)}")
    return 1 if (missing or len(bad)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
