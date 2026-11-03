"""Turn a JobSpec into AF3 job JSONs, a zip archive, and a provenance manifest."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from af3_jobs.mutations import collect_mutations
from af3_jobs.sequences import resolve_all
from af3_jobs.spec import REPO_ROOT, JobSpec


@dataclass
class Job:
    """One AF3 input file: its job name and the payload to serialise."""

    name: str
    payload: dict[str, Any]
    file_stem: str | None = None
    variant: str | None = None
    wt_aa: str | None = None
    position: int | None = None
    mut_aa: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.file_stem or self.name}.json"

    def text(self) -> str:
        return json.dumps(self.payload, indent=2)


@dataclass
class BuildResult:
    spec: JobSpec
    jobs: list[Job]
    skipped: list[str]
    log: list[str]
    sequences: dict[str, str | None]


def _base_payload(spec: JobSpec, sequences: dict[str, str | None]) -> dict[str, Any]:
    return {
        "name": f"{spec.prefix}_WT{spec.job_suffix}",
        "modelSeeds": list(spec.model_seeds),
        "sequences": [c.to_af3(sequences[c.id]) for c in spec.chains],
        "dialect": spec.dialect,
        "version": spec.version,
    }


def build_jobs(spec: JobSpec, *, refresh_sequences: bool = False) -> BuildResult:
    """Resolve sequences, collect variants, and build one Job per variant.

    A variant whose stated WT residue disagrees with the chain sequence is
    skipped, never silently applied -- that check is what caught the KRAS 4A/4B
    isoform mix-up, so it stays fatal-per-variant rather than a warning.
    """
    sequences = resolve_all(spec.chains, refresh=refresh_sequences)
    target = spec.target_chain
    target_seq = sequences[target.id]

    mutations, log = collect_mutations(
        spec.mutations, target_seq, first_residue=target.first_residue
    )

    base = _base_payload(spec, sequences)
    jobs: list[Job] = []
    skipped: list[str] = []

    if spec.include_wt:
        jobs.append(
            Job(
                name=f"{spec.prefix}_WT{spec.job_suffix}",
                file_stem=f"{spec.filename_prefix}_WT{spec.job_suffix}",
                payload=copy.deepcopy(base),
            )
        )

    last_residue = target.first_residue + len(target_seq) - 1
    target_index = next(i for i, c in enumerate(spec.chains) if c.mutate)

    for variant, (wt_aa, position, mut_aa) in sorted(
        mutations.items(), key=lambda kv: (kv[1][1], kv[1][2])
    ):
        if not target.first_residue <= position <= last_residue:
            skipped.append(
                f"{variant}: position {position} outside chain {target.id} "
                f"({target.first_residue}-{last_residue})"
            )
            continue
        idx = position - target.first_residue
        actual_aa = target_seq[idx]
        if actual_aa != wt_aa:
            skipped.append(
                f"{variant}: expected {wt_aa} at position {position}, "
                f"found {actual_aa} in chain {target.id}"
            )
            continue

        payload = copy.deepcopy(base)
        payload["name"] = f"{spec.prefix}_{variant}{spec.job_suffix}"
        mutated = target_seq[:idx] + mut_aa + target_seq[idx + 1 :]
        payload["sequences"][target_index][target.kind]["sequence"] = mutated
        jobs.append(
            Job(
                name=f"{spec.prefix}_{variant}{spec.job_suffix}",
                file_stem=f"{spec.filename_prefix}_{variant}{spec.job_suffix}",
                payload=payload,
                variant=variant,
                wt_aa=wt_aa,
                position=position,
                mut_aa=mut_aa,
            )
        )

    return BuildResult(
        spec=spec, jobs=jobs, skipped=skipped, log=log, sequences=sequences
    )


def build_manifest(result: BuildResult) -> dict[str, Any]:
    """Provenance record written next to the archive.

    Records the exact sequence hash per chain so a later question of the form
    'which construct produced these ensembles?' is answerable without the zip.
    """
    spec = result.spec
    chains = []
    for chain in spec.chains:
        seq = result.sequences[chain.id]
        chains.append(
            {
                "id": chain.id,
                "kind": chain.kind,
                "gene": chain.gene,
                "uniprot": chain.uniprot,
                "residue_range": list(chain.residue_range) if chain.residue_range else None,
                "first_residue": chain.first_residue,
                "length": len(seq) if seq else None,
                "ccd_codes": list(chain.ccd_codes) or None,
                "sha256": hashlib.sha256(seq.encode()).hexdigest() if seq else None,
                "mutate": chain.mutate,
                "note": chain.note,
            }
        )
    return {
        "name": spec.name,
        "job_prefix": spec.prefix,
        "file_prefix": spec.filename_prefix,
        "description": spec.description,
        "generated": date.today().isoformat(),
        "config": str(spec.source_path.relative_to(REPO_ROOT)) if spec.source_path else None,
        "dialect": spec.dialect,
        "version": spec.version,
        "model_seeds": list(spec.model_seeds),
        "structures_per_variant": len(spec.model_seeds) * 5,
        "chains": chains,
        "mutation_source": spec.mutations.source,
        "n_jobs": len(result.jobs),
        "n_variants": sum(1 for j in result.jobs if j.variant),
        "includes_wt": spec.include_wt,
        "variants": [j.variant for j in result.jobs if j.variant],
        "skipped": result.skipped,
        "log": result.log,
    }


def write_jobs(
    result: BuildResult,
    *,
    out_dir: Path | None = None,
    write_zip: bool = True,
    write_dir: bool = False,
) -> dict[str, Path]:
    """Write the batch. Returns the paths that were created."""
    spec = result.spec
    base_dir = out_dir or (REPO_ROOT / spec.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.name}_Mutations{spec.tag}"
    written: dict[str, Path] = {}

    if write_zip:
        zip_path = base_dir / f"{stem}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for job in result.jobs:
                zf.writestr(job.filename, job.text())
        written["zip"] = zip_path

    if write_dir:
        job_dir = base_dir / stem
        job_dir.mkdir(parents=True, exist_ok=True)
        for job in result.jobs:
            (job_dir / job.filename).write_text(job.text())
        written["dir"] = job_dir

    manifest_path = base_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(build_manifest(result), indent=2) + "\n")
    written["manifest"] = manifest_path
    return written
