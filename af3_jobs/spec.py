"""Declarative description of an AF3 job batch: chains, ligands, seeds, mutations.

A config is plain JSON (no PyYAML in ``thesis_core``) and lives in
``af3_jobs/configs/``. Every field that changes the physical system being
predicted is explicit here, so two batches can be compared by diffing configs
rather than by reading a script's edit history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

ChainKind = Literal["protein", "dna", "rna", "ligand"]


@dataclass(frozen=True)
class ChainSpec:
    """One entry of the AF3 ``sequences`` list.

    ``first_residue`` makes mutation numbering independent of the construct: a
    fragment chain such as the p85alpha niSH2 (UniProt 322-600) carries
    ``first_residue=322``, so a variant is still named in UniProt coordinates
    even though the stored sequence starts at index 0.
    """

    id: str | tuple[str, ...]
    kind: ChainKind = "protein"
    sequence: str | None = None
    uniprot: str | None = None
    residue_range: tuple[int, int] | None = None
    ccd_codes: tuple[str, ...] = ()
    gene: str | None = None
    mutate: bool = False
    first_residue: int = 1
    note: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "ligand":
            if not self.ccd_codes:
                raise ValueError(f"chain {self.id}: ligand needs ccd_codes")
        else:
            if self.sequence is None and self.uniprot is None:
                raise ValueError(
                    f"chain {self.id}: needs either 'sequence' or 'uniprot'"
                )
        if self.mutate and self.kind != "protein":
            raise ValueError(f"chain {self.id}: only protein chains can be mutated")

    @property
    def ids(self) -> tuple[str, ...]:
        """Every chain id of this entity: one, or several for a homo-oligomer."""
        return self.id if isinstance(self.id, tuple) else (self.id,)

    def to_af3(self, sequence: str | None = None) -> dict[str, Any]:
        """Render as one element of the AF3 ``sequences`` array.

        A tuple ``id`` is a homo-oligomer: AF3 takes ``"id": ["A", "B", ...]`` for
        identical copies, and a mutation then applies to every copy (the TP53
        tetramer batches were made this way)."""
        af3_id = list(self.id) if isinstance(self.id, tuple) else self.id
        if self.kind == "ligand":
            return {"ligand": {"id": af3_id, "ccdCodes": list(self.ccd_codes)}}
        seq = sequence if sequence is not None else self.sequence
        if seq is None:
            raise ValueError(f"chain {self.id}: sequence not resolved")
        return {self.kind: {"id": af3_id, "sequence": seq}}


@dataclass(frozen=True)
class MutationSourceSpec:
    """Where the variant list comes from, and what is filtered out of it.

    ``source`` is one of:
      ``clinvar``   -- rows of a ClinVar CSV export, gene-filtered
      ``list``      -- the names in ``include`` and nothing else
      ``file``      -- one variant name per line
      ``enumerate`` -- every substitution within ``max_nt_subs`` nt changes
      ``none``      -- WT only
    ``include`` is always added on top of the chosen source; ``exclude`` and
    ``done`` are always removed from it.
    """

    source: Literal["clinvar", "list", "file", "enumerate", "none"] = "none"
    csv: str | None = None
    path: str | None = None
    gene: str | None = None
    max_nt_subs: int | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    done: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source == "clinvar" and not self.csv:
            raise ValueError("mutations.source='clinvar' requires 'csv'")
        if self.source == "file" and not self.path:
            raise ValueError("mutations.source='file' requires 'path'")
        if self.source == "enumerate" and self.max_nt_subs is None:
            raise ValueError("mutations.source='enumerate' requires 'max_nt_subs'")


@dataclass(frozen=True)
class JobSpec:
    """A whole batch: the complex, the seeds, and the variants to apply to it."""

    name: str
    chains: tuple[ChainSpec, ...]
    job_prefix: str | None = None
    file_prefix: str | None = None
    job_suffix: str = ""
    model_seeds: tuple[int, ...] = tuple(range(1, 26))
    dialect: str = "alphafold3"
    version: int = 4
    mutations: MutationSourceSpec = field(default_factory=MutationSourceSpec)
    include_wt: bool = True
    output_dir: str = "muts_jsons"
    tag: str = ""
    description: str | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        ids = [i for c in self.chains for i in c.ids]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: duplicate chain ids {ids}")
        if not any(c.mutate for c in self.chains):
            raise ValueError(
                f"{self.name}: no chain has \"mutate\": true -- "
                "mark the subunit that carries the variants"
            )
        if sum(c.mutate for c in self.chains) > 1:
            raise ValueError(
                f"{self.name}: more than one chain marked 'mutate'. Generate one "
                "batch per mutated subunit so the variant names stay unambiguous."
            )

    @property
    def prefix(self) -> str:
        """Prefix of every job name, e.g. ``PIK3CA`` in ``PIK3CA_H1047R``."""
        return self.job_prefix or self.name

    @property
    def filename_prefix(self) -> str:
        """Prefix of the written files, which need not equal the job name.

        The MSH2/MSH6 batches were written as ``AF3_MSH2_<variant>.json`` while
        the job inside is named ``MutS_alpha_MSH2_<variant>``; downstream code
        keys off the archive filename, so the two are kept separable.
        """
        return self.file_prefix or self.prefix

    @property
    def target_chain(self) -> ChainSpec:
        return next(c for c in self.chains if c.mutate)


def _parse_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        start, _, end = value.partition("-")
        value = [int(start), int(end)]
    start, end = int(value[0]), int(value[1])
    if start > end:
        raise ValueError(f"invalid residue range {start}-{end}")
    return start, end


def _parse_seeds(value: Any) -> tuple[int, ...]:
    """An int ``n`` means seeds 1..n; a list is taken verbatim."""
    if value is None:
        return tuple(range(1, 26))
    if isinstance(value, int):
        if value < 1:
            raise ValueError("model_seeds count must be >= 1")
        return tuple(range(1, value + 1))
    return tuple(int(s) for s in value)


def _chain_from_dict(d: dict[str, Any]) -> ChainSpec:
    rng = _parse_range(d.get("residue_range") or d.get("range"))
    first = d.get("first_residue")
    if first is None:
        first = rng[0] if rng else 1
    return ChainSpec(
        id=tuple(d["id"]) if isinstance(d["id"], list) else d["id"],
        kind=d.get("kind") or d.get("type") or "protein",
        sequence=d.get("sequence"),
        uniprot=d.get("uniprot"),
        residue_range=rng,
        ccd_codes=tuple(d.get("ccd_codes") or d.get("ccdCodes") or ()),
        gene=d.get("gene"),
        mutate=bool(d.get("mutate", False)),
        first_residue=int(first),
        note=d.get("note"),
    )


def spec_from_dict(d: dict[str, Any], source_path: Path | None = None) -> JobSpec:
    muts = d.get("mutations") or {}
    output = d.get("output") or {}
    return JobSpec(
        name=d["name"],
        job_prefix=d.get("job_prefix"),
        file_prefix=d.get("file_prefix"),
        job_suffix=d.get("job_suffix") or "",
        chains=tuple(_chain_from_dict(c) for c in d["chains"]),
        model_seeds=_parse_seeds(d.get("model_seeds")),
        dialect=d.get("dialect", "alphafold3"),
        version=int(d.get("version", 4)),
        mutations=MutationSourceSpec(
            source=muts.get("source", "none"),
            csv=muts.get("csv"),
            path=muts.get("path"),
            gene=muts.get("gene"),
            max_nt_subs=muts.get("max_nt_subs"),
            include=tuple(muts.get("include") or ()),
            exclude=tuple(muts.get("exclude") or ()),
            done=tuple(muts.get("done") or ()),
        ),
        include_wt=bool(d.get("include_wt", True)),
        output_dir=output.get("dir", "muts_jsons"),
        tag=output.get("tag", ""),
        description=d.get("description"),
        source_path=source_path,
    )


def resolve_config(name_or_path: str) -> Path:
    """Accept a config name (``PIK3CA``) or an explicit path to a JSON file."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".json" and candidate.exists():
        return candidate
    for stem in (name_or_path, f"{name_or_path}.json"):
        path = CONFIG_DIR / stem
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        if path.exists():
            return path
    available = ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.json")))
    raise FileNotFoundError(
        f"no config '{name_or_path}' in {CONFIG_DIR} (have: {available or 'none'})"
    )


def load_spec(name_or_path: str) -> JobSpec:
    path = resolve_config(name_or_path)
    with path.open() as fh:
        return spec_from_dict(json.load(fh), source_path=path)


def available_configs() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.json"))
