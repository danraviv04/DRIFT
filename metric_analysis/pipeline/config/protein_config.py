"""Shared per-protein configuration for the chunked cache-repair/sweep scripts.

All protein settings live in protein_configs.json, one entry per protein holding
the union of fields any script needs. Each script only requires a subset (e.g.
repair_dna_cache_chunk.py needs dna_chains, repair_rmsf_cache_chunk.py needs
archive_root/wt_hint) and not every protein has every field — KRAS has no DNA
to repair, POLE currently has no RMSF-repair archive_root/wt_hint. Scripts
declare the fields they need via `required_keys` and get a clear error instead
of a bare KeyError if a protein doesn't support that task.

Usage in a script:
    from metric_analysis.pipeline.config.protein_config import get_protein_config, supported_proteins
    REQUIRED = ["cache_dir", "protein_chains", "dna_chains"]
    parser.add_argument("--protein", required=True, choices=supported_proteins(REQUIRED))
    ...
    cfg = get_protein_config(args.protein, REQUIRED)
"""

import json
import os
from pathlib import Path

from metric_analysis.pipeline.paths import ANALYSIS_DIR, REPO_ROOT

# ROOT resolves protein_configs.json's relative paths (Ground_Truths/, PLDDTs/, ...)
# against the repo root, exactly as before the package refactor.
ROOT = REPO_ROOT
CONFIG_PATH = ANALYSIS_DIR / "protein_configs.json"

_PATH_KEYS = {"cache_dir", "coord_cache_dir", "sweep_dir", "archive_root", "plddt_dir", "clinvar_csv", "labels_csv", "phenotypic_xlsx",
              "dms_xlsx", "read_root_path"}
_NESTED_PATH_KEYS = {"clinvar_csv_by_gene"}  # dict[str, str-path]

# Coordinate frame switch (2026-10-15). A protein's coordinate metrics read the cache in
# `coord_cache_dir` (the WT rigid-core fit). COORD_FRAME=<name> applies the overrides in that
# protein's `coord_frames[<name>]` instead -- e.g. usalign_first_chain, the control cache with
# the old superposition and its own sweep_dir -- to every protein that defines it. An env var
# rather than an argument so the chunk subprocesses of the sweep inherit it.
COORD_FRAME_ENV = "COORD_FRAME"

_configs = None


def _resolve_path(value):
    # Raw AlphaFold 3 archives live outside the repo: "$DRIFT_ARCHIVE_ROOT/<batch>". Left
    # unexpanded when the variable is unset, so a stage that needs them fails naming it.
    value = os.path.expandvars(str(value))
    p = Path(value)
    return p if p.is_absolute() or value.startswith("$") else ROOT / p


def _load():
    global _configs
    if _configs is not None:
        return _configs

    with open(CONFIG_PATH) as f:
        raw = json.load(f)

    configs = {}
    for protein, entry in raw.items():
        cfg = dict(entry)
        cfg["prefix"] = protein
        for key in _PATH_KEYS:
            if key in cfg:
                cfg[key] = _resolve_path(cfg[key])
        for key in _NESTED_PATH_KEYS:
            if key in cfg:
                cfg[key] = {gene: _resolve_path(v) for gene, v in cfg[key].items()}
        frame = os.environ.get(COORD_FRAME_ENV)
        override = (cfg.get("coord_frames") or {}).get(frame) if frame else None
        if override:
            for key, value in override.items():
                cfg[key] = _resolve_path(value) if key in _PATH_KEYS else value
            cfg["coord_frame"] = frame
        if "hotspot_range" in cfg:
            cfg["hotspot_range"] = tuple(cfg["hotspot_range"])
        if "ensemble_repeats_by_size" in cfg:
            cfg["ensemble_repeats_by_size"] = {int(k): v for k, v in cfg["ensemble_repeats_by_size"].items()}
        configs[protein] = cfg

    _configs = configs
    return _configs


def supported_proteins(required_keys):
    """Proteins whose config has every field in required_keys — use as argparse choices.

    Includes entries marked ``"enabled": false`` so an explicit ``--protein PTEN`` still
    works (a disabled protein is usually one whose *labels* are unusable while its
    structural stages are fine). Use enabled_proteins() for the default 'all' fan-out.
    """
    return [protein for protein, cfg in _load().items() if all(k in cfg for k in required_keys)]


def enabled_proteins(required_keys):
    """supported_proteins() minus entries marked ``"enabled": false``.

    Scaffolded proteins (config written, data/labels not ready) stay in the JSON so their
    intended setup is version-controlled and reviewable, but must not join a ``--protein
    all`` sweep and fail mid-run. Flip the flag when the blockers in their ``_status``
    field are cleared.
    """
    return [p for p in supported_proteins(required_keys) if _load()[p].get("enabled", True)]


def get_protein_config(protein, required_keys):
    configs = _load()
    if protein not in configs:
        raise KeyError(f"Unknown protein {protein!r}. Known proteins: {sorted(configs)}")
    cfg = configs[protein]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(
            f"Protein {protein!r} is missing config field(s) {missing} needed for this script "
            f"— is {protein} supported for this task? (see protein_configs.json)"
        )
    return cfg
