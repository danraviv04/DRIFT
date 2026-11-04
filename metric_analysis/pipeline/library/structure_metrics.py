from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import glob
import os
import re
import tarfile
import tempfile
import warnings

import matplotlib.pyplot as plt
import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import MMCIFParser, PDBParser
from tqdm.auto import tqdm


from metric_analysis.pipeline.paths import REPO_ROOT as PROJECT_ROOT
from metric_analysis.pipeline.labels import artifact_key
from metric_analysis.pipeline.library.plddt_distribution import (
    DEFAULT_OUTLIER_THRESHOLD as DEFAULT_PLDDT_OUTLIER_THRESHOLD,
    compute_per_residue_plddt_distribution_metrics,
    summarize_per_residue_plddt_distribution_metrics,
)

DEFAULT_PLDDT_ROOT = PROJECT_ROOT / "PLDDTs"
DEFAULT_METRICS_ROOT = PROJECT_ROOT / "Embedding_Analysis" / "structure_metrics"


@dataclass(frozen=True)
class StructureMetricsConfig:
    """Configuration for extracting pLDDT matrices from generated structures."""

    mut_name: str
    read_root_path: Path | str
    chain_ids: Sequence[str] = ("A",)
    source_type: str = "tar"  # supported: tar, folder
    artifact_glob: str = "*.tar.gz"
    plddt_dir: Path | str | None = None
    output_dir: Path | str | None = None
    drop_matching_sample: bool = True
    skip_existing_csv: bool = True
    # Exact name of the WT artifact every mutant is measured against. Optional: with a
    # single WT artifact it is unambiguous and can stay None. Required once a protein has
    # more than one (POLE has three, across two different constructs) -- see
    # `_resolve_wt_artifact_path`, which raises rather than taking whichever sorted first.
    wt_artifact: str | None = None

    def resolved_plddt_dir(self) -> Path:
        return Path(self.plddt_dir) if self.plddt_dir else DEFAULT_PLDDT_ROOT / self.mut_name

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir) if self.output_dir else DEFAULT_METRICS_ROOT / self.mut_name


_PARSER = MMCIFParser(QUIET=True)


# AlphaFold3 writes the best-ranked prediction twice: once at the top of the archive
# as `{name}_model.cif`, and once as the seed-sample it actually came from. Verified on
# MutS_alpha A272V straight from the tar: the 125 seed-samples have 125 distinct CA
# coordinate sets, and the top-level file is an exact coordinate copy of
# seed-1_sample-1. Discovery globs `**/*.cif`, so every ensemble built from an archive
# silently carried 126 members with one counted twice -- a different member per variant,
# whichever AF3 ranked best.
_SEED_SAMPLE_RE = re.compile(r"seed-\d+_sample-\d+")


def _load_structure_any_format(structure_path: Path | str):
    path = Path(structure_path)
    suffix = path.suffix.lower()

    if suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    elif suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure file type: {path}")

    return parser.get_structure(path.stem, str(path))


def _resolve_wt_artifact_path(
    config: StructureMetricsConfig,
    all_artifacts: Sequence[Path],
    reference_hint: str,
) -> List[Path]:
    """Return the WT candidates with the chosen reference first, or raise if ambiguous.

    Callers take `[0]` as the reference, so the point of this function is that `[0]` is a
    decision rather than an accident of sort order. Previously this was a plain
    name-contains filter over a sorted artifact list, and the reference was whichever WT
    sorted first. That is fine for the six proteins with one WT and wrong for POLE, which
    has three across two constructs -- a truncated DNA-bound 268-471 model and two
    full-length ones -- where the pick silently decided what every mutant is compared to.

    `config.wt_artifact` names the reference exactly, matched on `artifact_key`, so it
    does not depend on `.tar.gz` or a trailing `_data` (configs, caches and archives
    disagree on both). The short canonical stem is still accepted for older configs.
    With no config and more than one candidate this raises instead of guessing.
    """
    candidates = [p for p in all_artifacts if reference_hint.upper() in p.name.upper()]
    configured = getattr(config, "wt_artifact", None)

    if configured:
        wanted = str(configured)
        chosen = [p for p in candidates
                  if artifact_key(p) == artifact_key(wanted) or canonical_artifact_stem(p) == wanted]
        if not chosen:
            raise ValueError(
                f"[{config.mut_name}] configured wt_artifact {wanted!r} matched none of the "
                f"WT artifacts found: {sorted(canonical_artifact_stem(p) for p in candidates)}"
            )
        rest = [p for p in candidates if p not in chosen]
        return [*chosen, *rest]

    if len(candidates) > 1:
        raise ValueError(
            f"[{config.mut_name}] {len(candidates)} artifacts match reference_hint "
            f"{reference_hint!r} and none is chosen: "
            f"{sorted(canonical_artifact_stem(p) for p in candidates)}. Set \"wt_artifact\" "
            f"in the protein config -- the alignment reference is a modelling decision and "
            f"must not fall out of sort order."
        )
    return candidates


def extract_cif_paths(cif_dir: Path | str) -> List[str]:
    return glob.glob(os.path.join(str(cif_dir), "**", "*.cif"), recursive=True)


def extract_plddt_from_cif(
    cif_path: Path | str,
    *,
    chain_ids: Sequence[str] = ("A",),
    prefer_atom: str = "CA",
    parser: MMCIFParser | None = None,
) -> Dict[int, float]:
    """Return residue-indexed pLDDT values for selected chains."""
    parser = parser or _PARSER
    structure = parser.get_structure("x", str(cif_path))
    model = next(structure.get_models())

    out: Dict[int, float] = {}
    index = 0

    for chain_id in chain_ids:
        if chain_id not in model:
            chains = [c.id for c in model.get_chains()]
            raise KeyError(f"Chain '{chain_id}' not found in {cif_path}. Chains: {chains}")

        chain = model[chain_id]
        for residue in chain:
            key = index
            index += 1

            if prefer_atom and (prefer_atom in residue):
                out[key] = float(residue[prefer_atom].get_bfactor())
            else:
                atoms = list(residue.get_atoms())
                if atoms:
                    out[key] = float(np.mean([a.get_bfactor() for a in atoms]))

    return out


def analyze_plddt_distribution(
    cif_dir: Path | str,
    *,
    chain_ids: Sequence[str] = ("A",),
    parser: MMCIFParser | None = None,
) -> Dict[str, Dict[int, float]]:
    """Collect sample_name -> residue pLDDT mapping from all CIF files in a directory."""
    parser = parser or _PARSER
    maps: Dict[str, Dict[int, float]] = {}

    for cif_path in extract_cif_paths(cif_dir):
        plddt_map = extract_plddt_from_cif(
            cif_path,
            chain_ids=chain_ids,
            parser=parser,
        )
        sample_name = Path(cif_path).parent.name
        maps[sample_name] = plddt_map

    return maps


def canonical_artifact_stem(path_or_name: Path | str) -> str:
    name = Path(path_or_name).name
    if name.endswith(".tar.gz"):
        name = name[: -len(".tar.gz")]
    else:
        name = Path(name).stem

    if "_with" in name:
        name = name.split("_with")[0]

    if name.lower().endswith("_data"):
        name = name[: -len("_data")]

    return name


def drop_matching_sample(plddt_maps: Dict[str, Dict[int, float]], artifact_stem: str) -> None:
    """Remove an ensemble sample key that echoes the mutation identifier itself."""
    search_name = canonical_artifact_stem(artifact_stem).lower()
    for key in list(plddt_maps.keys()):
        if search_name in key.lower():
            plddt_maps.pop(key)
            break


def iter_artifacts(config: StructureMetricsConfig) -> List[Path]:
    root = Path(config.read_root_path)

    if config.source_type == "tar":
        return sorted(root.glob(f"**/{config.artifact_glob}"))

    if config.source_type == "folder":
        return sorted([p for p in root.glob("*") if p.is_dir()])

    raise ValueError("source_type must be either 'tar' or 'folder'")


def _plddt_values_from_maps(plddt_maps: Mapping[str, Mapping[int, float]]) -> np.ndarray:
    values: List[float] = []
    for plddt_map in plddt_maps.values():
        values.extend(plddt_map.values())
    if not values:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _plot_histogram(values: np.ndarray, title: str, out_path: Path, bins: int = 20) -> None:
    if values.size == 0:
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel("pLDDT Value")
    plt.ylabel("Frequency")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def process_artifacts_to_plddt(config: StructureMetricsConfig, show_progress: bool = True) -> pd.DataFrame:
    """Create per-artifact pLDDT CSV and histogram files.

    Returns a summary table with one row per artifact.
    """
    plddt_dir = config.resolved_plddt_dir()
    plddt_dir.mkdir(parents=True, exist_ok=True)
    (plddt_dir / "hists").mkdir(parents=True, exist_ok=True)

    artifacts = iter_artifacts(config)
    iterator = tqdm(artifacts, desc=f"Processing {config.mut_name}", unit="artifact") if show_progress else artifacts

    rows: List[dict] = []

    for artifact in iterator:
        artifact_stem = canonical_artifact_stem(artifact)
        csv_path = plddt_dir / f"{artifact_stem}_plddt_values.csv"
        csv_alt = plddt_dir / f"{artifact_stem}_plddt_matrix.csv"
        hist_path = plddt_dir / "hists" / f"{artifact_stem}_plddt_histogram.png"

        status = "generated"
        num_samples = 0
        num_values = 0

        if config.skip_existing_csv and csv_path.exists():
            status = "existing_csv"
            matrix = load_plddt_matrix(csv_path)
            num_values = int(np.sum(np.isfinite(matrix)))
        elif config.skip_existing_csv and csv_alt.exists():
            status = "existing_csv_legacy_name"
            matrix = load_plddt_matrix(csv_alt)
            num_values = int(np.sum(np.isfinite(matrix)))
            csv_alt.rename(csv_path)
        else:
            if config.source_type == "tar":
                with tempfile.TemporaryDirectory() as tmpdir:
                    with tarfile.open(artifact, "r:*") as tar_ref:
                        tar_ref.extractall(tmpdir)
                    plddt_maps = analyze_plddt_distribution(tmpdir, chain_ids=config.chain_ids)
            else:
                plddt_maps = analyze_plddt_distribution(artifact, chain_ids=config.chain_ids)

            if config.drop_matching_sample:
                drop_matching_sample(plddt_maps, artifact_stem)

            num_samples = len(plddt_maps)
            values = _plddt_values_from_maps(plddt_maps)
            num_values = int(values.size)

            if plddt_maps:
                pd.DataFrame(plddt_maps).to_csv(csv_path, index=True)
                _plot_histogram(values, f"pLDDT Distribution for {artifact_stem}", hist_path)
            else:
                status = "empty"

        rows.append(
            {
                "artifact": artifact_stem,
                "source_path": str(artifact),
                "csv_path": str(csv_path),
                "hist_path": str(hist_path),
                "status": status,
                "num_samples": num_samples,
                "num_values": num_values,
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# pLDDT scope: run a subunit of a modelled complex as its own dataset.
#
# MSH2 and MSH6 have no PLDDTs folder of their own -- they read the MutS_alpha
# complex's CSVs and take only their own residue rows and their own files. Every
# pLDDT read in this module funnels through list_plddt_csv_files,
# load_plddt_matrix and load_plddt_profile (22 call sites across ~15 plotting
# and table functions, none of which receives the config), so the restriction is
# applied HERE rather than threaded through all of them.
#
# The runner enters this scope once per protein and the pipeline processes one
# protein per process, so there is no interleaving. It is a context manager
# rather than a bare global so the restriction cannot leak past the protein it
# was set for, and so a nested run restores the outer value.
# --------------------------------------------------------------------------- #
_PLDDT_ROW_BLOCK = None      # (start, stop) half-open, or None for the whole matrix
_PLDDT_INCLUDE = None        # substring every artifact filename must carry, or None


@contextlib.contextmanager
def plddt_scope(row_block=None, include=None):
    """Restrict pLDDT reads to one subunit's rows and files for the enclosed block."""
    global _PLDDT_ROW_BLOCK, _PLDDT_INCLUDE
    prev = (_PLDDT_ROW_BLOCK, _PLDDT_INCLUDE)
    _PLDDT_ROW_BLOCK = None if row_block is None else (int(row_block[0]), int(row_block[1]))
    _PLDDT_INCLUDE = None if include is None else str(include).upper()
    try:
        yield
    finally:
        _PLDDT_ROW_BLOCK, _PLDDT_INCLUDE = prev


def _apply_row_block(arr, filepath=""):
    """Slice a residue-indexed array/matrix to the active row block."""
    if _PLDDT_ROW_BLOCK is None:
        return arr
    start, stop = _PLDDT_ROW_BLOCK
    if len(arr) < stop:
        raise ValueError(
            f"pLDDT row block {(start, stop)} exceeds the {len(arr)}-row matrix "
            f"in {filepath}"
        )
    return arr[start:stop]

def list_plddt_csv_files(plddt_dir: Path | str) -> List[Path]:
    """Artifact CSVs under a pLDDT directory, honouring the active pLDDT scope.

    A subunit dataset shares its parent complex's directory, so MSH2 must not
    pick up MSH6's files. The wild type is always kept: one prediction of the
    complex is the reference for both subunits and its filename carries no
    subunit token.
    """
    files = sorted(Path(plddt_dir).glob("*.csv"))
    if _PLDDT_INCLUDE is None:
        return files
    token = f"_{_PLDDT_INCLUDE}_"
    return [f for f in files
            if token in f.name.upper() or "WT" in f.name.upper()]


def load_plddt_matrix(filepath: Path | str) -> np.ndarray:
    """Load pLDDT CSV as a residue x ensemble matrix (sliced to the active scope).

    Supports both numeric-only CSVs and CSVs with non-numeric metadata columns.

    The CSVs carry the residue index as their first column (header ``Unnamed: 0`` or
    ``Residue``, values ``0..N-1``). That column is numeric, so it survives the coerce
    below and would otherwise be averaged in as if it were an ensemble member --
    contaminating every per-residue mean, and asymmetrically so between mutant and WT
    (they have different sample counts), which manufactures a spurious mut/WT difference.
    The distortion scales with residue count, so it is negligible for ~200-residue
    proteins but severe for MutS_alpha (2294) / TP53 (1572). We drop the leading column
    whenever it is exactly ``0..N-1`` -- matching the index_col=0 read of the
    ensemble-aware sweeps.
    """
    raw_df = pd.read_csv(filepath)

    numeric_df = raw_df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if numeric_df.empty:
        raise ValueError(f"No numeric pLDDT values found in {filepath}")

    matrix = numeric_df.to_numpy(dtype=float)
    if matrix.shape[1] > 1 and np.array_equal(matrix[:, 0], np.arange(matrix.shape[0], dtype=float)):
        matrix = matrix[:, 1:]
    return _apply_row_block(matrix, filepath)


def extract_mutation_code(
    filename: str,
    known_mutations: Optional[Iterable[str]] = None,
    pattern: str = r"(?<![A-Z0-9])([A-Z]\d+[A-Z])(?![A-Z0-9])",  # token-bounded: PIK3CA is not K3C
) -> Optional[str]:
    if known_mutations:
        for mut in known_mutations:
            if mut in filename:
                return mut

    match = re.search(pattern, filename.upper())
    if match:
        return match.group(1)

    return None


def build_plddt_distribution_tables(
    plddt_files: Sequence[Path | str],
    *,
    known_mutations: Optional[Sequence[str]] = None,
    wt_tokens: Sequence[str] = ("WT",),
    wt_file: Path | str | None = None,
    min_samples: int = 2,
    outlier_threshold: float = DEFAULT_PLDDT_OUTLIER_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build variant and residue tables for full-distribution pLDDT effects.

    Labels are attached later by the orchestration layer and therefore cannot leak
    into metric construction.
    """
    files = [Path(path) for path in plddt_files]
    wt_candidates = [
        path
        for path in files
        if any(token.upper() in path.name.upper() for token in wt_tokens)
    ]
    if wt_file is not None:
        chosen_wt = Path(wt_file)
        if not chosen_wt.exists():
            raise FileNotFoundError(f"Configured WT pLDDT CSV not found: {chosen_wt}")
    elif len(wt_candidates) == 1:
        chosen_wt = wt_candidates[0]
    elif len(wt_candidates) > 1:
        raise ValueError(
            "Multiple WT pLDDT CSVs found; pass wt_file explicitly: "
            f"{[path.name for path in wt_candidates]}"
        )
    else:
        raise ValueError("No WT pLDDT CSV found for distribution comparison")

    wt_matrix = load_plddt_matrix(chosen_wt)
    summary_rows: list[dict[str, Any]] = []
    residue_tables: list[pd.DataFrame] = []
    empty_summary = summarize_per_residue_plddt_distribution_metrics(
        pd.DataFrame(
            columns=[
                "wasserstein", "wasserstein_unit", "wasserstein_relative",
                "shape_excess", "shape_excess_relative", "shape_fraction",
                "energy_distance", "cvm_effect", "location_shift_relative",
                "wt_median", "outlier_score", "outlier_fraction",
                "outlier_mean_deviation", "outlier_count",
                "outlier_threshold",
                "n_wt", "n_mutant",
            ]
        ),
        outlier_threshold=outlier_threshold,
    )

    for plddt_file in files:
        if plddt_file.resolve() == chosen_wt.resolve():
            continue
        if plddt_file in wt_candidates:
            # Alternate WT constructs are references, not mutant variants.
            continue
        mutation_code = extract_mutation_code(
            plddt_file.name, known_mutations=known_mutations
        )
        if not mutation_code:
            continue

        mutant_matrix = load_plddt_matrix(plddt_file)
        base = {
            "mutation_code": mutation_code,
            "source_file": str(plddt_file),
            "wt_source_file": str(chosen_wt),
            "n_wt_structures": int(wt_matrix.shape[1]),
            "n_mutant_structures": int(mutant_matrix.shape[1]),
            "n_wt_residues": int(wt_matrix.shape[0]),
            "n_mutant_residues": int(mutant_matrix.shape[0]),
        }
        if wt_matrix.shape[0] != mutant_matrix.shape[0]:
            warnings.warn(
                f"{mutation_code}: pLDDT residue-count mismatch "
                f"(WT={wt_matrix.shape[0]}, mutant={mutant_matrix.shape[0]}); "
                "distribution metrics set to NaN.",
                RuntimeWarning,
            )
            summary_rows.append(
                {
                    **base,
                    "status": "residue_count_mismatch",
                    **empty_summary,
                }
            )
            continue

        residue_result = compute_per_residue_plddt_distribution_metrics(
            wt_matrix, mutant_matrix, min_samples=min_samples,
            outlier_threshold=outlier_threshold,
        )
        residue_result.insert(0, "mutation_code", mutation_code)
        residue_result["source_file"] = str(plddt_file)
        residue_tables.append(residue_result)
        summary = summarize_per_residue_plddt_distribution_metrics(residue_result)
        summary_rows.append({**base, "status": "ok", **summary})

    summary_df = pd.DataFrame(summary_rows)
    residue_df = (
        pd.concat(residue_tables, ignore_index=True)
        if residue_tables
        else pd.DataFrame(
            columns=[
                "mutation_code", "residue_index", "wasserstein",
                "wasserstein_unit", "wasserstein_relative", "location_shift",
                "location_shift_relative", "shape_excess",
                "shape_excess_relative", "shape_fraction", "energy_distance",
                "cvm_effect", "wt_median", "outlier_score",
                "outlier_fraction", "outlier_mean_deviation", "outlier_count",
                "outlier_threshold", "n_wt", "n_mutant", "source_file",
            ]
        )
    )
    return summary_df, residue_df
