"""Load and validate the protein-specific inputs used by DRIFT fusion.

Adding a protein should start in ``protein_configs.json``.  Each enabled entry
must define its D1 sources and one explicit label mode; the loader rejects
incomplete entries instead of silently assigning scientifically invalid labels.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().with_name("protein_configs.json")

SUPPORTED_LABEL_MODES = {
    # No binary label at all: the protein's assay is a continuous target only
    # (TP53, KRAS since 2026-10-13). Such entries are disabled for the binary
    # pipeline and read through load_continuous_configs().
    "continuous_assay",
    "positive_list",
    "clinvar_by_subunit",
    "binary_csv",
}

_LABEL_REQUIRED_KEYS = {
    "continuous_assay": {"target"},
    "positive_list": {"path"},
    "clinvar_by_subunit": {"subunit_files"},
    "binary_csv": {"path", "mutation_col", "label_col"},
}

_PATH_KEYS = {"read_root", "path"}
_NESTED_PATH_KEYS = {"member_files", "subunit_files", "mut_csv_replacements"}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate(protein: str, cfg: dict) -> None:
    missing_sections = [name for name in ("d1", "label") if name not in cfg]
    if missing_sections:
        raise ValueError(
            f"Protein {protein!r} is missing required section(s): {missing_sections}"
        )

    d1 = cfg["d1"]
    missing_d1 = [name for name in ("mut", "read_root", "chain_ids") if name not in d1]
    if missing_d1:
        raise ValueError(f"Protein {protein!r} D1 config is missing: {missing_d1}")

    mode = cfg["label"].get("mode")
    if mode not in SUPPORTED_LABEL_MODES:
        raise ValueError(
            f"Protein {protein!r} has unsupported label mode {mode!r}; "
            f"choose one of {sorted(SUPPORTED_LABEL_MODES)}"
        )
    missing_label = sorted(_LABEL_REQUIRED_KEYS[mode].difference(cfg["label"]))
    if missing_label:
        raise ValueError(
            f"Protein {protein!r} label config for mode {mode!r} is missing: "
            f"{missing_label}"
        )


def _prepare(protein: str, raw_cfg: dict) -> dict:
    """Validate one entry and resolve its paths against the repository."""
    cfg = deepcopy(raw_cfg)
    _validate(protein, cfg)
    for section_name in ("d1", "label", "afm"):
        section = cfg.get(section_name, {})
        for key in _PATH_KEYS:
            if key in section:
                section[key] = _resolve_path(section[key])
        for key in _NESTED_PATH_KEYS:
            if key in section:
                section[key] = {
                    name: _resolve_path(value)
                    for name, value in section[key].items()
                }
    return cfg


def load_protein_configs(path: str | Path = CONFIG_PATH) -> dict[str, dict]:
    """Return enabled protein configs with paths resolved against the repository.

    These are the BINARY pipeline's proteins. A continuous-assay entry has no
    label and must not be enabled here.
    """
    config_path = Path(path)
    with config_path.open() as handle:
        raw = json.load(handle)

    configs: dict[str, dict] = {}
    for protein, raw_cfg in raw.items():
        if not raw_cfg.get("enabled", True):
            continue
        cfg = _prepare(protein, raw_cfg)
        if cfg["label"]["mode"] == "continuous_assay":
            raise ValueError(
                f"Protein {protein!r} has no binary label (continuous_assay) and "
                "must be disabled for the binary pipeline"
            )
        configs[protein] = cfg
    if not configs:
        raise ValueError(f"No enabled proteins found in {config_path}")
    return configs


def load_continuous_configs(
    proteins, path: str | Path = CONFIG_PATH,
) -> dict[str, dict]:
    """Configs for the continuous functional analyses, whatever their binary status.

    TP53 and KRAS are disabled for the binary pipeline (no label) but their D1
    sources still define how their descriptors are computed; PTEN and MSH2 are
    enabled binary datasets that ALSO have an assay. Same validation and path
    resolution as load_protein_configs, so descriptors match it exactly.
    """
    with Path(path).open() as handle:
        raw = json.load(handle)
    missing = [p for p in proteins if p not in raw]
    if missing:
        raise ValueError(f"No config for continuous protein(s): {missing}")
    return {protein: _prepare(protein, raw[protein]) for protein in proteins}


def build_d1_runtime_configs(protein_configs: dict[str, dict]) -> dict[str, dict]:
    """Extract the runtime-facing D1 dictionary without label/AFM metadata."""
    return {protein: deepcopy(cfg["d1"]) for protein, cfg in protein_configs.items()}


def protein_config_fingerprint(path: str | Path = CONFIG_PATH) -> str:
    """Hash scientific config fields while ignoring underscore-prefixed notes.

    ``_status``/``_todo`` fields are human documentation.  Including them in a
    data-cache key made prose-only edits trigger expensive D1 recomputation.
    Active fields such as paths, label modes, chain IDs, and ``enabled`` remain
    covered by the fingerprint.
    """
    with Path(path).open() as handle:
        raw = json.load(handle)

    def scientific_payload(value):
        if isinstance(value, dict):
            return {
                key: scientific_payload(item)
                for key, item in value.items()
                if not key.startswith("_")
            }
        if isinstance(value, list):
            return [scientific_payload(item) for item in value]
        return value

    payload = json.dumps(
        scientific_payload(raw), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
