"""Seed-extension jobs from a finished AF3 job's ``*_data.json`` (MSA-ready).

A job that already ran keeps its alignments and templates in
``<job>/<job>_data.json``. Extending its ensemble with more seeds needs no new
data pipeline: the same file with a different ``modelSeeds`` list runs inference
only (``--run_data_pipeline=false``, which is what gen_local.sh and slurm/'s gen
worker do for every ``*_data.json``).

Each chunk is written as ``<job>_seeds_<a>-<b>_data.json`` with the job's own
``name`` unchanged. The runners name their output after the INPUT file, so the
archive comes back as ``<job>_seeds_<a>-<b>_data.tar.gz`` holding
``<job>_seeds_<a>-<b>_data/<job>/seed-*_sample-*`` -- the layout of the TP53
extensions that local_runner/merge_seed_extensions.py already pairs and merges.
Every other key of the source is kept byte-for-byte, so the merge's "data.json
must agree except modelSeeds" check passes.

    python -m af3_jobs.seed_extension SRC_data.json --seeds 26-75 --chunk 5 \\
        --out muts_jsons/MutS_alpha_MSH6_GT_WT_seed_ext
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path


def parse_seeds(text: str) -> list[int]:
    a, _, b = text.partition("-")
    lo, hi = int(a), int(b or a)
    if lo > hi:
        raise ValueError(f"bad seed range {text}")
    return list(range(lo, hi + 1))


def make_chunks(seeds: list[int], chunk: int) -> list[list[int]]:
    return [seeds[i:i + chunk] for i in range(0, len(seeds), chunk)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_json", type=Path, help="a finished job's <job>_data.json (with MSAs)")
    ap.add_argument("--seeds", required=True, help="new seed range, e.g. 26-75")
    ap.add_argument("--chunk", type=int, default=5, help="seeds per job (default 5)")
    ap.add_argument("--out", type=Path, required=True, help="output directory (a .zip is written next to it)")
    args = ap.parse_args(argv)

    raw = args.data_json.read_bytes()
    src = json.loads(raw)
    name = src["name"]
    old = set(src.get("modelSeeds") or [])
    seeds = parse_seeds(args.seeds)
    clash = sorted(old & set(seeds))
    if clash:
        raise SystemExit(f"{name}: seeds {clash} already in the source run {sorted(old)}")
    for seq in src["sequences"]:
        e = next(iter(seq.values()))
        if seq.get("protein") is not None and not e.get("unpairedMsa"):
            raise SystemExit(f"{name}: protein chain {e.get('id')} has no MSA -- not an MSA-ready data.json")

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for ch in make_chunks(seeds, args.chunk):
        job = dict(src, modelSeeds=ch)
        stem = f"{name}_seeds_{ch[0]}-{ch[-1]}_data"
        path = args.out / f"{stem}.json"
        path.write_text(json.dumps(job))
        written.append({"file": path.name, "name": name, "modelSeeds": ch})
    zpath = args.out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for w in written:
            zf.write(args.out / w["file"], w["file"])
    manifest = {
        "generated": date.today().isoformat(),
        "source_data_json": str(args.data_json),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_modelSeeds": sorted(old),
        "job_name": name,
        "new_seeds": seeds,
        "chunk": args.chunk,
        "jobs": written,
        "run": "inference only (--run_data_pipeline=false); merge with local_runner/merge_seed_extensions.py",
    }
    (args.out.parent / f"{args.out.name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(written)} jobs for {name}, seeds {seeds[0]}-{seeds[-1]} -> {args.out} and {zpath}")


if __name__ == "__main__":
    main()
