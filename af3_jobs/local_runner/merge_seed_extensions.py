#!/usr/bin/env python3
"""Merge AF3 seed-extension archives into their original runs.

Pairs, by job stem, an original generation archive

    <orig-dir>/<stem>.tar.gz                    root: <stem>/seed-*_sample-*/...
    (e.g. P53_A74D_with_ZN_and_DNA.tar.gz, seeds 1-20)

with its extension

    <ext-dir>/<stem>_seeds_<a>-<b>_data.tar.gz  root: <stem>_seeds_<a>-<b>_data/<stem>/...
    (e.g. P53_A74D_with_ZN_and_DNA_seeds_21-25_data.tar.gz, seeds 21-25)

and writes one archive in the local-runner layout (gen_local.sh):

    <out-dir>/<stem>_data.tar.gz                root: <stem>_data/<stem>/seed-*_sample-*/...
                                                      <stem>_data/_SUCCESS

Archives are streamed member by member; nothing is extracted to disk.

Several attempts in one archive (<stem>/ next to timestamped
<stem>_YYYYMMDD_HHMMSS/ dirs, from an MSA-stage leftover or a retry) are
resolved per archive: the single attempt holding structures is used and the
structure-less ones are dropped and logged. Two attempts with structures abort
the pair.

What the merged job directory holds:
  * every seed-*_sample-* directory from both runs (a seed/sample present in
    both aborts the pair);
  * <stem>_ranking_scores.csv with the rows of both runs, sorted by seed, sample;
  * <stem>_data.json with modelSeeds set to the union of both runs. The two
    data.json files must agree on every other key (sequences, ligands, ...),
    otherwise the pair is refused: that is how a wrong-variant extension shows;
  * the top-level <stem>_model.cif / _confidences.json /
    _summary_confidences.json of whichever run holds the best ranking score,
    so they again describe the best sample of the combined ensemble;
  * TERMS_OF_USE.md from the original run, and _MERGE_PROVENANCE.txt.

A run whose AF3 job wrote every sample but stopped before its job-level files
(no _ranking_scores.csv, no top-level best sample; seen once, TP53 D57G) is
still merged: its ranking rows are rebuilt from the per-sample
summary_confidences.json and, if it holds the best sample, the top-level files
are copied from that sample's directory. Those summaries round ranking_score to
2 decimals, so the rebuilt rows are coarser than AF3's and ties go to the lowest
(seed, sample); _MERGE_PROVENANCE.txt says which run was rebuilt. The fallback
first requires the samples to cover every seed of the run's data.json with the
same number of samples each, and every sample's JSONs to parse and its CIF to
end in its closing '#', so a job cut off part way through writing is refused.

Each output is written to <name>.partial, re-read and checked, then renamed,
so an interrupted run never leaves a truncated archive under the final name.
Existing outputs are skipped unless --overwrite. Each pair first checks that
the output disk has room for it. With --delete-sources the pair's two source
archives are deleted after its output is verified; a skipped or failed pair
keeps its sources.

Usage:
    python af3_jobs/local_runner/merge_seed_extensions.py \\
        --orig-dir $DRIFT_ARCHIVE_ROOT/output_gen_only \\
        --ext-dir  $DRIFT_ARCHIVE_ROOT/TP53_extensions \\
        --out-dir  /path/to/merged \\
        [--only A74D R175H ...] [--expect-structures 125] [--jobs 2]
        [--delete-sources] [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

EXT_RE = re.compile(r"^(?P<stem>.+)_seeds_(?P<a>\d+)-(?P<b>\d+)_data\.tar\.gz$")
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")
SAMPLE_RE = re.compile(r"^seed-(?P<seed>\d+)_sample-(?P<sample>\d+)$")
SAMPLE_FILES = ("model.cif", "confidences.json", "summary_confidences.json")
# Top-level files AF3 writes for the best-ranked sample of a run.
BEST_FILES = ("model.cif", "confidences.json", "summary_confidences.json")
# Files at the root of an archive (outside the job directory) that are dropped.
ROOT_MARKERS = {"_SUCCESS", "_PROVENANCE.txt", "_FAILED"}
FREE_SPACE_MARGIN = 2 * 10**9


class MergeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Gzip:
    """How archives are (de)compressed. With pigz, compression is multi-threaded
    and (de)compression runs in a child process alongside the Python work;
    without it, Python's single-threaded zlib does everything (~5x slower)."""
    level: int = 6
    pigz: str | None = None   # path to pigz, or None for Python's gzip
    threads: int = 1          # pigz compression threads


@contextmanager
def _tar_reader(path: Path, gz: Gzip):
    if gz.pigz is None:
        with tarfile.open(path, mode="r|gz") as t:
            yield t
        return
    proc = subprocess.Popen([gz.pigz, "-dc", str(path)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    ok = False
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as t:
            yield t
        while proc.stdout.read(1 << 20):  # drain the end-of-archive padding
            pass
        ok = True
    finally:
        if not ok:
            proc.kill()
        err = proc.stderr.read()
        rc = proc.wait()
    if rc:
        raise MergeError(f"pigz -dc {path.name} failed ({rc}): {err.decode(errors='replace').strip()}")


@contextmanager
def _tar_writer(path: Path, gz: Gzip):
    if gz.pigz is None:
        with tarfile.open(path, mode="w:gz", compresslevel=gz.level,
                          format=tarfile.PAX_FORMAT) as t:
            yield t
        return
    with open(path, "wb") as fh:
        proc = subprocess.Popen([gz.pigz, f"-{gz.level}", "-p", str(gz.threads)],
                                stdin=subprocess.PIPE, stdout=fh, stderr=subprocess.PIPE)
        ok = False
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT) as t:
                yield t
            ok = True
        finally:
            proc.stdin.close()
            if not ok:
                proc.kill()
            err = proc.stderr.read()
            rc = proc.wait()
    if rc:
        raise MergeError(f"pigz failed writing {path.name} ({rc}): {err.decode(errors='replace').strip()}")


@dataclass
class RunContents:
    """What one source archive contributed, after streaming it."""
    path: Path
    job_dir: str | None = None                       # the attempt that holds the structures
    samples: dict = field(default_factory=dict)      # (seed, sample) -> {suffix: sha1}
    top: dict = field(default_factory=dict)          # top-level filename -> bytes
    top_info: dict = field(default_factory=dict)     # top-level filename -> TarInfo
    dropped: list = field(default_factory=list)      # structure-less attempts left out
    extras: list = field(default_factory=list)       # unexpected names, reported
    summaries: dict = field(default_factory=dict)    # (seed, sample) -> summary_confidences.json bytes
    ranking_rebuilt: bool = False                    # no _ranking_scores.csv; rows from the summaries


# ---------------------------------------------------------------- pairing --

def find_pairs(orig_dir: Path, ext_dir: Path):
    def visible(p: Path) -> bool:  # skip macOS AppleDouble files on exFAT/NTFS
        return p.is_file() and not p.name.startswith("._")

    exts = {}
    for p in sorted(ext_dir.glob("*.tar.gz")):
        m = EXT_RE.match(p.name)
        if visible(p) and m:
            if m["stem"] in exts:
                raise MergeError(f"two extensions for {m['stem']}: {exts[m['stem']].name}, {p.name}")
            exts[m["stem"]] = p

    origs = {}
    for p in sorted(orig_dir.glob("*.tar.gz")):
        if not visible(p) or EXT_RE.match(p.name):
            continue
        stem = p.name[: -len(".tar.gz")]
        if stem.endswith("_data"):  # also accept local-runner-style originals
            stem = stem[: -len("_data")]
        stem = TIMESTAMP_RE.sub("", stem)  # AF3's timestamped run dir, <stem>_YYYYMMDD_HHMMSS
        if stem in origs:
            raise MergeError(f"two originals for {stem}: {origs[stem].name}, {p.name}")
        origs[stem] = p

    # Unpacked extension folders (e.g. an MSA-only run with no structures yet)
    # are not merged, only reported.
    ext_folders = sorted(p.name for p in ext_dir.iterdir()
                         if p.is_dir() and re.match(r"^.+_seeds_\d+-\d+_data$", p.name))

    pairs = [(s, origs[s], exts[s]) for s in sorted(exts) if s in origs]
    ext_only = sorted(s for s in exts if s not in origs)
    orig_only = sorted(s for s in origs if s not in exts)
    return pairs, ext_only, orig_only, ext_folders


# ---------------------------------------------------------------- merging --

def _job_relpath(name: str, stem: str):
    """(job_dir, path relative to it) for a member inside the job directory, or
    None for a member at the archive root (wrapper dir, _SUCCESS, ...). The job
    directory is <stem>/ or AF3's timestamped <stem>_YYYYMMDD_HHMMSS/, either at
    the top or under one wrapper dir (<stem>_seeds_a-b_data/, <stem>_data/)."""
    parts = [p for p in name.split("/") if p not in ("", ".")]
    for i, p in enumerate(parts[:2]):
        if _job_dir_re(stem).match(p) and (i == 0 or parts[0].startswith(stem)):
            return p, "/".join(parts[i + 1:])
    if len(parts) == 1 and parts[0] in ROOT_MARKERS:
        return None
    if len(parts) <= 2 and parts[0].startswith(stem):
        return None
    raise MergeError(f"unexpected member layout: {name}")


def _job_dir_re(stem: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(stem)}(_\d{{8}}_\d{{6}})?$")


def _stream_run(src: Path, stem: str, out_tar: tarfile.TarFile, job_prefix: str,
                written: dict, gz: Gzip = Gzip()) -> RunContents:
    """Stream one source archive into out_tar.

    An archive can hold several attempts at the same job: <stem>/ and/or
    timestamped <stem>_YYYYMMDD_HHMMSS/ siblings, which AF3 creates whenever the
    output dir already exists (an MSA-stage leftover, or a retry after a crash).
    Exactly one attempt may contain seed-*_sample-* structures; that one is used
    and the structure-less ones (typically just a _data.json, possibly the
    null-filled decoy) are dropped. Structures in two attempts abort the pair:
    which one is right is not decidable here. AF3 writes no sample until
    inference for all seeds is done, so a crashed attempt has none.
    """
    run = RunContents(src)
    attempt_top: dict = {}  # job_dir -> {rel: (bytes, TarInfo)}
    with _tar_reader(src, gz) as tin:
        for m in tin:
            loc = _job_relpath(m.name, stem)
            if loc is None:  # archive root: wrapper dir, _SUCCESS, _PROVENANCE.txt
                base = m.name.rstrip("/").split("/")[-1]
                if not m.isdir() and base not in ROOT_MARKERS:
                    run.extras.append(m.name)
                continue
            job_dir, rel = loc
            top = attempt_top.setdefault(job_dir, {})
            if rel == "":  # the job directory itself
                continue
            first, _, rest = rel.partition("/")
            sm = SAMPLE_RE.match(first)

            if sm:
                if run.job_dir is None:
                    run.job_dir = job_dir
                elif job_dir != run.job_dir:
                    raise MergeError(f"{src.name}: two attempts contain structures, "
                                     f"{run.job_dir}/ and {job_dir}/; pick one by hand")
                key = (int(sm["seed"]), int(sm["sample"]))
                owner = written.get(key)
                if owner is not None and owner != src:
                    raise MergeError(f"seed {key[0]} sample {key[1]} is in both "
                                     f"{owner.name} and {src.name}")
                written[key] = src
                files = run.samples.setdefault(key, {})
                if m.isdir():
                    if not rest:
                        out_tar.addfile(_retarget(m, f"{job_prefix}/{first}"))
                    continue
                if not m.isfile():
                    raise MergeError(f"non-regular member {m.name}")
                expected_prefix = f"{stem}_{first}_"
                suffix = rest[len(expected_prefix):] if rest.startswith(expected_prefix) else rest
                fobj = tin.extractfile(m)
                h = hashlib.sha1()
                if suffix == "summary_confidences.json":  # ~2 kB; kept in case the CSV is missing
                    data = fobj.read()
                    h.update(data)
                    run.summaries[key] = data
                    out_tar.addfile(_retarget(m, f"{job_prefix}/{rel}"), io.BytesIO(data))
                else:
                    out_tar.addfile(_retarget(m, f"{job_prefix}/{rel}"), _HashingReader(fobj, h))
                files[suffix] = h.hexdigest()
            else:
                if m.isdir():
                    raise MergeError(f"unexpected directory in job dir: {m.name}")
                if not m.isfile():
                    raise MergeError(f"non-regular member {m.name}")
                if rel in top:
                    raise MergeError(f"duplicate member {m.name}")
                top[rel] = (tin.extractfile(m).read(), m)

    if run.job_dir is None:
        raise MergeError(f"{src.name}: no seed-*_sample-* structures in any of "
                         f"{sorted(attempt_top) or 'its directories'}")
    for rel, (data, info) in attempt_top[run.job_dir].items():
        run.top[rel], run.top_info[rel] = data, info
    run.dropped = sorted(d for d in attempt_top if d != run.job_dir)
    return run


class _HashingReader(io.RawIOBase):
    def __init__(self, fobj, h):
        self._f, self._h = fobj, h

    def readable(self):
        return True

    def read(self, n=-1):
        b = self._f.read(n)
        self._h.update(b)
        return b


def _retarget(m: tarfile.TarInfo, name: str) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size = m.size if m.isfile() else 0
    ti.mtime, ti.mode, ti.type = m.mtime, m.mode, m.type
    ti.uid, ti.gid, ti.uname, ti.gname = m.uid, m.gid, m.uname, m.gname
    return ti


def _new_info(name: str, size: int, like: tarfile.TarInfo | None = None) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size = size
    ti.mtime = int(time.time())
    ti.mode = like.mode if like is not None else 0o644
    if like is not None:
        ti.uid, ti.gid, ti.uname, ti.gname = like.uid, like.gid, like.uname, like.gname
    return ti


def _read_ranking(run: RunContents, stem: str) -> list[tuple[int, int, float]]:
    name = f"{stem}_ranking_scores.csv"
    if name not in run.top:
        return _rebuild_ranking(run, stem)
    rows = list(csv.DictReader(io.StringIO(run.top[name].decode())))
    out = [(int(r["seed"]), int(r["sample"]), float(r["ranking_score"])) for r in rows]
    have = {(s, k) for s, k, _ in out}
    if have != set(run.samples):
        raise MergeError(f"{run.path.name}: ranking_scores.csv rows {len(have)} do not match "
                         f"the {len(run.samples)} sample directories")
    return out


def _rebuild_ranking(run: RunContents, stem: str) -> list[tuple[int, int, float]]:
    """Ranking rows from the per-sample summary_confidences.json, for a run
    without _ranking_scores.csv. Refused unless the samples cover every seed of
    the run's data.json, the same number per seed (a job cut off between seeds
    would otherwise pass as a smaller ensemble). The per-file checks are in
    _check_rebuilt_samples."""
    csv_name, data_name = f"{stem}_ranking_scores.csv", f"{stem}_data.json"
    if data_name not in run.top:
        raise MergeError(f"{run.path.name}: no {csv_name} and no {data_name} to check the samples against")
    want = sorted(json.loads(run.top[data_name])["modelSeeds"])
    have = sorted({s for s, _ in run.samples})
    if have != want:
        raise MergeError(f"{run.path.name}: no {csv_name}, and the samples cover seeds {_span(have)}, "
                         f"not data.json's {_span(want)}")
    per_seed = {s: sum(1 for t, _ in run.samples if t == s) for s in have}
    if len(set(per_seed.values())) != 1:
        raise MergeError(f"{run.path.name}: no {csv_name}, and seeds hold unequal sample counts {per_seed}")
    out = []
    for key in sorted(run.samples):
        try:
            score = float(json.loads(run.summaries[key])["ranking_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MergeError(f"{run.path.name}: no {csv_name}, and seed {key[0]} sample {key[1]} "
                             f"has no readable ranking_score ({type(exc).__name__}: {exc})")
        out.append((*key, score))
    run.ranking_rebuilt = True
    print(f"  NOTE {stem}: {run.path.name} has no {csv_name}; rebuilt {len(out)} rows from the "
          f"per-sample summary_confidences.json (2-decimal scores)", file=sys.stderr)
    return out


def _check_rebuilt_samples(run: RunContents, stem: str, gz: Gzip,
                           best: tuple[int, int] | None) -> dict[str, tuple[bytes, tarfile.TarInfo]]:
    """Second pass over a run with a rebuilt ranking: every sample's JSONs must
    parse and its CIF must end in the closing '#' (a job that died while writing
    leaves a short file, which tar stores as is). Returns the files of sample
    `best`, keyed by suffix, when the top level has to be taken from it."""
    found: dict = {}
    with _tar_reader(run.path, gz) as tin:
        for m in tin:
            loc = _job_relpath(m.name, stem)
            if loc is None or loc[0] != run.job_dir or not m.isfile():
                continue
            first, _, rest = loc[1].partition("/")
            sm = SAMPLE_RE.match(first)
            if not sm:
                continue
            key = (int(sm["seed"]), int(sm["sample"]))
            prefix = f"{stem}_{first}_"
            suffix = rest[len(prefix):] if rest.startswith(prefix) else rest
            data = tin.extractfile(m).read()
            try:
                if suffix.endswith(".json"):
                    json.loads(data)
                elif suffix == "model.cif" and not data.endswith(b"\n#\n"):
                    raise ValueError("does not end in the closing '#'")
            except ValueError as exc:
                raise MergeError(f"{run.path.name}: no {stem}_ranking_scores.csv, and {m.name} "
                                 f"is incomplete ({exc}); the AF3 job was cut off while writing")
            if key == best and suffix in BEST_FILES:
                found[suffix] = (data, m)
    if best is not None and set(found) != set(BEST_FILES):
        raise MergeError(f"{run.path.name}: seed {best[0]} sample {best[1]} lacks "
                         f"{sorted(set(BEST_FILES) - set(found))} on the second read")
    return found


def _merge_data_json(a: RunContents, b: RunContents, stem: str) -> bytes:
    name = f"{stem}_data.json"
    for r in (a, b):
        if name not in r.top:
            raise MergeError(f"{r.path.name}: no {name}")
    ja, jb = json.loads(a.top[name]), json.loads(b.top[name])
    for j, r in ((ja, a), (jb, b)):
        if j.get("name") != stem:
            raise MergeError(f"{r.path.name}: data.json name is {j.get('name')!r}, expected {stem!r}")
    diff = sorted(k for k in set(ja) | set(jb) if k != "modelSeeds" and ja.get(k) != jb.get(k))
    if diff:
        raise MergeError(f"data.json of the two runs differ in {diff} -- not the same job")
    seeds = sorted(set(ja["modelSeeds"]) | set(jb["modelSeeds"]))
    if set(ja["modelSeeds"]) & set(jb["modelSeeds"]):
        raise MergeError(f"modelSeeds overlap: {sorted(set(ja['modelSeeds']) & set(jb['modelSeeds']))}")
    merged = dict(ja)
    merged["modelSeeds"] = seeds
    return (json.dumps(merged, indent=2) + "\n").encode()


def merge_pair(stem: str, orig: Path, ext: Path, out_dir: Path,
               expect_structures: int | None, overwrite: bool, verify: bool,
               gz: Gzip, delete_sources: bool, out_name: str | None = None) -> str:
    # out_name: archive and wrapper-dir name. The default <stem>_data is the merged
    # full run. Chaining extension chunks into one extension archive passes
    # <stem>_seeds_<a>-<b>_data instead, so it can never be taken for (or replace)
    # the original <stem>_data archive.
    out_name = out_name or f"{stem}_data"
    final = out_dir / f"{out_name}.tar.gz"
    if final.exists() and not overwrite:
        return f"SKIP   {stem}: {final.name} exists"
    part = final.with_name(final.name + ".partial")
    part.unlink(missing_ok=True)
    job_prefix = f"{out_name}/{stem}"
    t0 = time.time()

    # The merged archive is about the size of its two sources; refuse rather
    # than fill the disk half way through.
    need = orig.stat().st_size + ext.stat().st_size
    free = shutil.disk_usage(out_dir).free
    if free < need + FREE_SPACE_MARGIN:
        raise MergeError(f"only {free / 1e9:.1f} GB free in {out_dir}, need ~{need / 1e9:.1f} GB "
                         f"+ {FREE_SPACE_MARGIN / 1e9:.0f} GB margin")

    try:
        written: dict = {}
        with _tar_writer(part, gz) as tout:
            tout.addfile(_dir_info(out_name))
            tout.addfile(_dir_info(job_prefix))
            a = _stream_run(orig, stem, tout, job_prefix, written, gz)
            b = _stream_run(ext, stem, tout, job_prefix, written, gz)

            # Every sample directory must be complete.
            for run in (a, b):
                for (seed, sample), files in run.samples.items():
                    missing = [f for f in SAMPLE_FILES if f not in files]
                    if missing:
                        raise MergeError(f"{run.path.name}: seed {seed} sample {sample} lacks {missing}")

            ra, rb = _read_ranking(a, stem), _read_ranking(b, stem)
            best_a, best_b = max(ra, key=lambda r: r[2]), max(rb, key=lambda r: r[2])
            best_run, best = (a, best_a) if best_a[2] >= best_b[2] else (b, best_b)

            # A run without AF3's CSV has no top-level best files either: check its
            # samples, and take the top level from its best sample if it won.
            from_sample = {}
            for run in (a, b):
                if run.ranking_rebuilt:
                    got = _check_rebuilt_samples(run, stem, gz, best[:2] if run is best_run else None)
                    if run is best_run:
                        from_sample = got

            def put(name: str, data: bytes, like: tarfile.TarInfo | None):
                tout.addfile(_new_info(f"{job_prefix}/{name}", len(data), like), io.BytesIO(data))

            if from_sample:
                for suffix in BEST_FILES:
                    put(f"{stem}_{suffix}", *from_sample[suffix])
            else:
                # The top-level CIF of the winning run should be its best sample's.
                top_cif = f"{stem}_model.cif"
                if top_cif not in best_run.top:
                    raise MergeError(f"{best_run.path.name}: no top-level {top_cif}")
                if hashlib.sha1(best_run.top[top_cif]).hexdigest() != best_run.samples[best[:2]]["model.cif"]:
                    print(f"  WARN {stem}: top-level model.cif of {best_run.path.name} is not "
                          f"byte-identical to seed {best[0]} sample {best[1]}", file=sys.stderr)
                for suffix in BEST_FILES:
                    name = f"{stem}_{suffix}"
                    if name not in best_run.top:
                        raise MergeError(f"{best_run.path.name}: no top-level {name}")
                    put(name, best_run.top[name], best_run.top_info[name])

            rows = sorted(ra + rb)
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(["seed", "sample", "ranking_score"])
            for seed, sample, score in rows:
                w.writerow([seed, sample, repr(score)])
            csv_name = f"{stem}_ranking_scores.csv"
            put(csv_name, buf.getvalue().encode(), a.top_info.get(csv_name) or b.top_info.get(csv_name))

            put(f"{stem}_data.json", _merge_data_json(a, b, stem), a.top_info[f"{stem}_data.json"])

            tou = "TERMS_OF_USE.md"
            if tou in a.top or tou in b.top:
                src = a if tou in a.top else b
                put(tou, src.top[tou], src.top_info[tou])

            known = {f"{stem}_{s}" for s in BEST_FILES} | {
                f"{stem}_ranking_scores.csv", f"{stem}_data.json", tou}
            for run in (a, b):
                for name in sorted(set(run.top) - known):
                    run.extras.append(f"{stem}/{name}")
                if run.extras:
                    print(f"  WARN {stem}: dropped unexpected members of {run.path.name}: "
                          f"{run.extras}", file=sys.stderr)

            for run in (a, b):
                if run.dropped:
                    print(f"  NOTE {stem}: {run.path.name}: using attempt {run.job_dir}/, "
                          f"dropped structure-less {[d + '/' for d in run.dropped]}", file=sys.stderr)

            n = len(written)
            if expect_structures is not None and n != expect_structures:
                raise MergeError(f"{n} structures, expected {expect_structures}")
            seeds_a = sorted({s for s, _ in a.samples})
            seeds_b = sorted({s for s, _ in b.samples})
            prov = (
                f"merged_by=af3_jobs/local_runner/merge_seed_extensions.py\n"
                f"merged_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
                f"original={orig} ({orig.stat().st_size} bytes, seeds {_span(seeds_a)}, "
                f"{len(a.samples)} structures, attempt {a.job_dir}/{_dropped_note(a)})\n"
                f"extension={ext} ({ext.stat().st_size} bytes, seeds {_span(seeds_b)}, "
                f"{len(b.samples)} structures, attempt {b.job_dir}/{_dropped_note(b)})\n"
                f"structures={n}\n"
                f"top_level_best=seed {best[0]} sample {best[1]} ranking_score {best[2]!r} "
                f"(from {best_run.path.name}"
                f"{', copied from its sample directory' if from_sample else ''})\n"
            )
            for run in (a, b):
                if run.ranking_rebuilt:
                    prov += (f"ranking_rebuilt={run.path.name}: AF3 wrote no ranking_scores.csv or "
                             f"top-level files; its {len(run.samples)} rows come from the per-sample "
                             f"summary_confidences.json, rounded to 2 decimals\n")
            prov = prov.encode()
            tout.addfile(_new_info(f"{out_name}/_MERGE_PROVENANCE.txt", len(prov)), io.BytesIO(prov))
            tout.addfile(_new_info(f"{out_name}/_SUCCESS", 0), io.BytesIO(b""))

        if verify:
            _verify(part, job_prefix, n, gz)
        os.replace(part, final)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    deleted = ""
    if delete_sources:  # only reached once the output was re-read and verified
        orig.unlink()
        ext.unlink()
        deleted = ", sources deleted"

    mb = final.stat().st_size / 1e6
    return (f"OK     {stem}: {n} structures (seeds {_span(seeds_a)} + {_span(seeds_b)}), "
            f"best seed {best[0]} sample {best[1]} -> {final.name} "
            f"[{mb:.0f} MB, {time.time() - t0:.0f}s{deleted}]")


def _dropped_note(run: RunContents) -> str:
    return f"; dropped structure-less {', '.join(d + '/' for d in run.dropped)}" if run.dropped else ""


def _dir_info(name: str) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.type, ti.mode, ti.mtime = tarfile.DIRTYPE, 0o755, int(time.time())
    return ti


def _span(seeds: list[int]) -> str:
    if seeds and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ",".join(map(str, seeds))


def _verify(path: Path, job_prefix: str, n_expected: int, gz: Gzip = Gzip()) -> None:
    """Re-read the whole archive (catches a bad gzip stream) and recount CIFs."""
    n = 0
    with _tar_reader(path, gz) as t:
        for m in t:
            if m.isfile():
                f = t.extractfile(m)
                while f.read(1 << 20):
                    pass
                if (m.name.startswith(job_prefix + "/seed-") and m.name.endswith("_model.cif")):
                    n += 1
    if n != n_expected:
        raise MergeError(f"verification: {n} sample CIFs in {path.name}, expected {n_expected}")


# -------------------------------------------------------------------- CLI --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-dir", type=Path, required=True, help="folder of <stem>.tar.gz originals")
    ap.add_argument("--ext-dir", type=Path, required=True, help="folder of <stem>_seeds_a-b_data.tar.gz extensions")
    ap.add_argument("--out-dir", type=Path, required=True, help="where <stem>_data.tar.gz is written")
    ap.add_argument("--only", nargs="+", metavar="VARIANT",
                    help="restrict to these variants (e.g. A74D) or full stems")
    ap.add_argument("--expect-structures", type=int, default=None,
                    help="fail a pair unless the merge holds exactly this many structures (e.g. 125)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="pairs merged in parallel (default 1); raise until the drive saturates")
    ap.add_argument("--compresslevel", type=int, default=6, help="gzip level 1-9 (default 6)")
    ap.add_argument("--threads", type=int, default=None,
                    help="pigz compression threads per job (default: CPUs / --jobs)")
    ap.add_argument("--no-pigz", action="store_true",
                    help="use Python's single-threaded gzip even if pigz is installed")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    ap.add_argument("--no-verify", action="store_true", help="skip re-reading each output")
    ap.add_argument("--delete-sources", action="store_true",
                    help="delete a pair's original and extension once its merged output is written "
                         "and verified (frees space; a skipped or failed pair is never deleted)")
    ap.add_argument("--dry-run", action="store_true", help="list the pairing and exit")
    args = ap.parse_args(argv)
    if args.delete_sources and args.no_verify:
        ap.error("--delete-sources requires verification; drop --no-verify")

    for d in (args.orig_dir, args.ext_dir):
        if not d.is_dir():
            ap.error(f"not a directory: {d}")

    pairs, ext_only, orig_only, ext_folders = find_pairs(args.orig_dir, args.ext_dir)
    if args.only:
        wanted = set(args.only)
        pairs = [p for p in pairs if p[0] in wanted or any(f"_{v}_" in f"_{p[0]}_" for v in wanted)]

    print(f"{len(pairs)} pairs to merge; {len(ext_only)} extensions without an original; "
          f"{len(orig_only)} originals without an extension")
    for s in ext_only:
        print(f"  NO ORIGINAL  {s}")
    for s in orig_only:
        print(f"  NO EXTENSION  {s}")
    for s in ext_folders:
        print(f"  NOT AN ARCHIVE, ignored  {args.ext_dir / s}/")
    if args.dry_run:
        for stem, o, e in pairs:
            print(f"  {o.name}  +  {e.name}  ->  {stem}_data.tar.gz")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(args.out_dir, os.W_OK):
        ap.error(f"not writable: {args.out_dir}")

    pigz = None if args.no_pigz else shutil.which("pigz")
    jobs_n = max(1, args.jobs)
    threads = args.threads or max(1, (os.cpu_count() or 1) // jobs_n)
    gz = Gzip(level=args.compresslevel, pigz=pigz, threads=threads)
    print(f"compression: {'pigz -p ' + str(threads) if pigz else 'Python gzip (single-threaded; install pigz for ~5x)'}"
          f", level {gz.level}, {jobs_n} pair(s) at a time")

    jobs = [(stem, o, e, args.out_dir, args.expect_structures, args.overwrite,
             not args.no_verify, gz, args.delete_sources) for stem, o, e in pairs]
    failed = []
    with cf.ProcessPoolExecutor(max_workers=jobs_n) as pool:
        futs = {pool.submit(merge_pair, *j): j[0] for j in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            stem = futs[fut]
            try:
                print(f"[{i}/{len(jobs)}] {fut.result()}", flush=True)
            except Exception as exc:  # report and carry on with the other pairs
                failed.append(stem)
                print(f"[{i}/{len(jobs)}] FAIL   {stem}: {type(exc).__name__}: {exc}", flush=True)

    print(f"done: {len(jobs) - len(failed)} ok/skipped, {len(failed)} failed")
    for s in failed:
        print(f"  FAILED {s}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
