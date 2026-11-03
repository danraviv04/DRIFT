from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import glob
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import MMCIFParser, NeighborSearch, PDBIO, PDBParser
from scipy import stats
from scipy.spatial import cKDTree
from scipy.stats import levene, shapiro
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


from metric_analysis.pipeline.paths import REPO_ROOT as PROJECT_ROOT
from metric_analysis.pipeline.labels import artifact_key
from metric_analysis.pipeline.library.bootstrap_uncertainty import bootstrap_mean_sd
from metric_analysis.pipeline.library.plddt_ks import (
    compute_per_residue_plddt_ks,
    plot_plddt_ks_profile_by_class,
    plot_plddt_ks_vs_d1,
    summarize_per_residue_plddt_ks,
)
from metric_analysis.pipeline.library.plddt_distribution import (
    DEFAULT_OUTLIER_THRESHOLD as DEFAULT_PLDDT_OUTLIER_THRESHOLD,
    compute_per_residue_plddt_distribution_metrics,
    plot_plddt_distribution_profile_by_class,
    plot_plddt_distribution_summary,
    plot_plddt_distribution_vs_d1,
    summarize_per_residue_plddt_distribution_metrics,
)

# D1 residue alignment policy. WT and mutant pLDDT matrices are compared row-by-row
# (each row is one residue, positional index 0..N-1; the CSVs carry no residue label
# to re-index on). For a missense variant WT and mutant have identical residue counts,
# so a row-count mismatch signals a malformed/inconsistent artifact. Default: reject
# the comparison (D1/delta -> NaN, flagged via d1_length_mismatch) instead of silently
# truncating to the shorter length and reporting an invalid ratio. Set this True to opt
# back into the legacy min-length truncation.
D1_ALLOW_LENGTH_TRUNCATION = False
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


def _safe_extract_tar(tar_obj: tarfile.TarFile, target_dir: Path | str) -> None:
    root = Path(target_dir).resolve()
    for member in tar_obj.getmembers():
        member_path = (root / member.name).resolve()
        if not str(member_path).startswith(str(root) + os.sep):
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar_obj.extractall(root)


def _convert_cif_to_pdb(cif_path: Path | str, pdb_path: Path | str) -> None:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(Path(cif_path).stem, str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def _prepare_structure_files_for_rmsf(
    structure_files: Sequence[Path | str],
    *,
    conversion_dir: Path,
) -> List[Path]:
    prepared: List[Path] = []

    for idx, src in enumerate(structure_files):
        p = Path(src)
        suffix = p.suffix.lower()

        if suffix == ".pdb":
            prepared.append(p)
            continue

        if suffix == ".cif":
            out_pdb = conversion_dir / f"converted_{idx:04d}_{p.stem}.pdb"
            _convert_cif_to_pdb(p, out_pdb)
            prepared.append(out_pdb)
            continue

        raise ValueError(f"Unsupported structure file type for RMSF: {p}")

    return sorted(prepared)


REFERENCE_MODES = ("first", "medoid", "mean")


def _choose_reference_structure(
    structure_files: Sequence[Path],
    *,
    reference_hint: str = "WT",
    reference_path: Path | str | None = None,
) -> Path:
    if not structure_files:
        raise ValueError("No structure files provided")

    if reference_path is not None:
        ref = Path(reference_path)
        if not ref.exists():
            raise FileNotFoundError(f"Reference structure not found: {ref}")
        return ref

    for path in structure_files:
        if reference_hint and reference_hint.lower() in path.name.lower():
            return path

    return structure_files[0]


def _reference_candidates(
    structure_files: Sequence[Path], reference_hint: str
) -> List[Path]:
    """Structures eligible to serve as the superposition frame.

    When the hint matches anything (normally the wild-type structures), the
    reference is drawn from those only, so a mutant frame cannot become the
    reference the mutant is then measured against.
    """
    if reference_hint:
        matches = [p for p in structure_files if reference_hint.lower() in p.name.lower()]
        if matches:
            return matches
    return list(structure_files)


def _align_structures_to(
    structure_files: Sequence[Path],
    reference: Path,
    *,
    aligned_dir: Path,
    aligner: str = "USalign",
    verbose: bool = False,
) -> List[Path]:
    """Superpose every structure onto `reference`; reference copied through first."""
    aligned_dir.mkdir(parents=True, exist_ok=True)
    reference_out = aligned_dir / f"aligned_{reference.stem}.pdb"
    shutil.copy2(reference, reference_out)
    aligned_paths: List[Path] = [reference_out]

    for src in structure_files:
        if src.resolve() == reference.resolve():
            continue
        out_prefix = aligned_dir / f"aligned_{src.stem}_on_{reference.stem}"
        if verbose:
            print(f"[rmsf] Aligning {src.name} -> {reference.name}")
        proc = subprocess.run(
            [aligner, str(src), str(reference), "-o", str(out_prefix)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr_msg = (proc.stderr or "").strip()[:400]
            raise RuntimeError(
                f"Alignment failed for {src.name} using {aligner}. {stderr_msg}"
            )
        candidates = [out_prefix, out_prefix.with_suffix(".pdb"), Path(str(out_prefix) + ".pdb")]
        saved = next((p for p in candidates if p.exists()), None)
        if saved is None:
            raise FileNotFoundError(f"{aligner} did not produce aligned PDB for {src.name}")
        aligned_paths.append(saved)

    return aligned_paths


def _ca_tensor_from_aligned(
    aligned_paths: Sequence[Path],
) -> Tuple[np.ndarray, List[Tuple[str, int]], Dict[Tuple[str, int], str], List[int]]:
    """(frames x residues x 3) over CA positions shared by every aligned frame.

    Also returns the positions in `aligned_paths` that contributed a frame.
    Structures yielding no CA atoms are skipped, so callers that map a frame
    index back to a source file must go through this list rather than assuming
    the tensor and the input sequence line up.
    """
    maps, resnames, kept = [], {}, []
    for position, path in enumerate(aligned_paths):
        frame, frame_resnames = _frame_ca_map_from_pdb(path)
        if not frame:
            continue
        maps.append(frame)
        kept.append(position)
        for key, value in frame_resnames.items():
            resnames.setdefault(key, value)

    if len(maps) < 2:
        raise ValueError(f"Need at least 2 aligned frames, found {len(maps)}")

    common = sorted(set.intersection(*[set(m.keys()) for m in maps]))
    if not common:
        raise ValueError("No common CA positions across aligned frames")

    tensor = np.empty((len(maps), len(common), 3), dtype=np.float64)
    for t_idx, frame in enumerate(maps):
        for j_idx, key in enumerate(common):
            tensor[t_idx, j_idx] = frame[key]
    return tensor, common, resnames, kept


def _medoid_frame_index(tensor: np.ndarray) -> int:
    """Frame with the smallest mean RMSD to all others -- the ensemble's centre.

    This is the structural null the ensemble has to beat: if one central
    structure carries the same signal as the whole cloud, the cloud is
    redundant. Computed on the already-superposed coordinates, so it is a
    pairwise-RMSD medoid without a second round of alignments.
    """
    n_frames = tensor.shape[0]
    # rmsd[i, j] over residues, vectorised across j for each i.
    total = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        diffs = tensor - tensor[i][None, :, :]
        rmsd = np.sqrt(np.mean(np.sum(diffs * diffs, axis=2), axis=1))
        total[i] = rmsd.sum()
    return int(np.argmin(total))


def _write_mean_structure_pdb(
    tensor: np.ndarray,
    keys: Sequence[Tuple[str, int]],
    resnames: Dict[Tuple[str, int], str],
    path: Path,
) -> Path:
    """Write the ensemble-average CA coordinates as a CA-only PDB.

    The consensus/mean structure is not a physical conformation -- averaging
    coordinates shrinks bond geometry wherever the ensemble disagrees -- so it
    is only meaningful as a superposition frame and as a baseline to compare the
    full ensemble against, never as a predicted structure.
    """
    mean_coords = tensor.mean(axis=0)
    lines = []
    for serial, (key, coord) in enumerate(zip(keys, mean_coords), start=1):
        chain_id, resid = key
        resname = (resnames.get(key) or "GLY")[:3].rjust(3)
        lines.append(
            f"ATOM  {serial:5d}  CA  {resname} {str(chain_id)[:1]}{int(resid):4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _resolve_ensemble_reference(
    structure_files: Sequence[Path],
    *,
    reference_mode: str,
    reference_hint: str,
    work_dir: Path,
    aligner: str = "USalign",
    verbose: bool = False,
) -> Path:
    """Reference frame for `reference_mode`, via one provisional alignment pass.

    ``first`` keeps the historical behaviour: the first file matching the hint,
    which is an arbitrary member of the ensemble. ``medoid`` and ``mean`` cost a
    second round of superpositions but remove that arbitrariness, and give the
    central-structure baselines the ensemble claim needs to be tested against.
    """
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(
            f"reference_mode must be one of {REFERENCE_MODES}, got {reference_mode!r}"
        )
    candidates = _reference_candidates(structure_files, reference_hint)
    provisional = _choose_reference_structure(
        structure_files, reference_hint=reference_hint
    )
    if reference_mode == "first" or len(candidates) < 2:
        return provisional

    probe_dir = work_dir / "reference_probe"
    aligned = _align_structures_to(
        candidates, provisional, aligned_dir=probe_dir, aligner=aligner, verbose=False
    )
    tensor, keys, resnames, kept = _ca_tensor_from_aligned(aligned)

    if reference_mode == "medoid":
        # _align_structures_to emits the reference copy first, then `candidates`
        # with the reference skipped; `kept` maps tensor rows back onto that
        # order after any CA-less frames were dropped.
        ordered = [provisional] + [
            p for p in candidates if p.resolve() != provisional.resolve()
        ]
        chosen = ordered[kept[_medoid_frame_index(tensor)]]
        if verbose:
            print(f"[rmsf] medoid reference: {chosen.name} (of {len(ordered)} candidates)")
        return chosen

    mean_path = work_dir / "reference_mean_structure.pdb"
    _write_mean_structure_pdb(tensor, keys, resnames, mean_path)
    if verbose:
        print(f"[rmsf] consensus reference: mean of {tensor.shape[0]} structures "
              f"over {len(keys)} CA positions")
    return mean_path


def _discover_structure_files(root_dir: Path | str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Structure root directory not found: {root}")

    pdb_files = sorted(root.glob("**/*.pdb"))
    cif_files = sorted(root.glob("**/*.cif"))
    return [*pdb_files, *cif_files]


# AlphaFold3 writes the best-ranked prediction twice: once at the top of the archive
# as `{name}_model.cif`, and once as the seed-sample it actually came from. Verified on
# MutS_alpha A272V straight from the tar: the 125 seed-samples have 125 distinct CA
# coordinate sets, and the top-level file is an exact coordinate copy of
# seed-1_sample-1. Discovery globs `**/*.cif`, so every ensemble built from an archive
# silently carried 126 members with one counted twice -- a different member per variant,
# whichever AF3 ranked best.
_SEED_SAMPLE_RE = re.compile(r"seed-\d+_sample-\d+")


def drop_top_ranked_duplicate(structure_files: Sequence[Path | str]) -> List[Path]:
    """Drop AF3's top-level `{name}_model.cif` from an ensemble file list.

    Keeps only per-seed samples when any are present, which is the whole ensemble minus
    the duplicated best-ranked copy. Archives with no seed-sample naming at all are
    returned untouched rather than emptied -- one TP53 artifact
    (P53_D57G_with_ZN_and_DNA_20251204_115232) has no top-level model and is all
    seed-samples, and POLE's truncated 268-471 construct names its files
    `model.cif` inside `seed-N_sample-M/` directories, so the match has to be against
    the whole path rather than the basename.

    This is for the ENSEMBLE only. The alignment reference is chosen separately by
    `_choose_reference_structure`, where the top-ranked model is a defensible target
    and is deliberately left available.
    """
    paths = [Path(p) for p in structure_files]
    seeded = [p for p in paths if _SEED_SAMPLE_RE.search(str(p))]
    return seeded if seeded else paths


def _frame_ca_map_from_pdb(
    pdb_path: Path | str,
) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], str]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(pdb_path).stem, str(pdb_path))

    out: Dict[Tuple[str, int], np.ndarray] = {}
    resname_map: Dict[Tuple[str, int], str] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" not in residue:
                    continue

                resid = int(residue.id[1])
                resname = str(residue.resname).strip()
                key = (str(chain.id), resid)

                if key not in out:
                    out[key] = residue["CA"].coord.astype(np.float64, copy=True)
                    resname_map[key] = resname
        break

    return out, resname_map


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


def extract_chain_interface_residues(
    structure_path: Path | str,
    *,
    chain_group_a: Sequence[str],
    chain_group_b: Sequence[str],
    threshold: float = 5.0,
    include_resname: bool = True,
    first_model_only: bool = True,
) -> pd.DataFrame:
    """Extract protein-protein interface residues between two chain groups.

    A residue is considered interfacial if any of its atoms is within ``threshold``
    Angstrom of any atom from the opposite chain group. The output includes both
    directions (A->B and B->A), then deduplicates to one closest contact per
    query residue.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    group_a = [str(c) for c in chain_group_a]
    group_b = [str(c) for c in chain_group_b]
    if not group_a or not group_b:
        raise ValueError("chain_group_a and chain_group_b must be non-empty")
    if set(group_a).intersection(group_b):
        raise ValueError("chain_group_a and chain_group_b must be disjoint")

    structure = _load_structure_any_format(structure_path)

    rows: List[Dict[str, Any]] = []
    for model in structure:
        chain_map = {str(chain.id): chain for chain in model}
        chains_a = [chain_map[c] for c in group_a if c in chain_map]
        chains_b = [chain_map[c] for c in group_b if c in chain_map]

        if not chains_a or not chains_b:
            if first_model_only:
                break
            continue

        def _collect_direction(
            query_chains: Sequence[Any],
            partner_chains: Sequence[Any],
            query_label: str,
            partner_label: str,
        ) -> None:
            partner_atoms: List[Any] = []
            partner_meta: Dict[int, Tuple[str, int, str]] = {}

            for partner_chain in partner_chains:
                for partner_residue in partner_chain:
                    # Keep standard residues only.
                    if partner_residue.id[0] != " ":
                        continue
                    for partner_atom in partner_residue:
                        partner_atoms.append(partner_atom)
                        partner_meta[id(partner_atom)] = (
                            str(partner_chain.id),
                            int(partner_residue.id[1]),
                            str(partner_residue.resname).strip(),
                        )

            if not partner_atoms:
                return

            ns = NeighborSearch(partner_atoms)

            for query_chain in query_chains:
                for query_residue in query_chain:
                    if query_residue.id[0] != " ":
                        continue

                    best_partner = None
                    best_dist = float("inf")

                    for query_atom in query_residue:
                        hits = ns.search(query_atom.coord, float(threshold), level="A")
                        for partner_atom in hits:
                            dist = float(query_atom - partner_atom)
                            if dist < best_dist:
                                best_dist = dist
                                best_partner = partner_atom

                    if best_partner is None:
                        continue

                    partner_chain_id, partner_resid, partner_resname = partner_meta[id(best_partner)]
                    rows.append(
                        {
                            "model_id": str(model.id),
                            "query_group": query_label,
                            "partner_group": partner_label,
                            "query_chain": str(query_chain.id),
                            "query_resid": int(query_residue.id[1]),
                            "query_resname": str(query_residue.resname).strip() if include_resname else None,
                            "partner_chain": partner_chain_id,
                            "partner_resid": int(partner_resid),
                            "partner_resname": partner_resname if include_resname else None,
                            "min_distance": float(best_dist),
                        }
                    )

        _collect_direction(chains_a, chains_b, "A", "B")
        _collect_direction(chains_b, chains_a, "B", "A")

        if first_model_only:
            break

    out_cols = [
        "model_id",
        "query_group",
        "partner_group",
        "query_chain",
        "query_resid",
        "query_resname",
        "partner_chain",
        "partner_resid",
        "partner_resname",
        "min_distance",
    ]
    if not rows:
        return pd.DataFrame(columns=out_cols)

    out_df = pd.DataFrame(rows)
    out_df = (
        out_df.sort_values(["query_chain", "query_resid", "min_distance"], ascending=[True, True, True])
        .drop_duplicates(subset=["query_chain", "query_resid"], keep="first")
        .reset_index(drop=True)
    )
    return out_df[out_cols]


def interface_residue_index(
    interface_df: pd.DataFrame,
    *,
    chain_col: str = "query_chain",
    resid_col: str = "query_resid",
) -> set[Tuple[str, int]]:
    """Build a unique residue index set from an interface table."""
    if interface_df.empty:
        return set()

    missing = {chain_col, resid_col}.difference(interface_df.columns)
    if missing:
        raise KeyError(f"interface_df is missing required columns: {sorted(missing)}")

    tmp = interface_df[[chain_col, resid_col]].copy()
    tmp[chain_col] = tmp[chain_col].astype(str)
    tmp[resid_col] = pd.to_numeric(tmp[resid_col], errors="coerce")
    tmp = tmp[tmp[resid_col].notna()].copy()

    return {(str(chain), int(resid)) for chain, resid in zip(tmp[chain_col], tmp[resid_col])}


def _filter_df_to_residue_set(
    df: pd.DataFrame,
    *,
    residue_set: set[Tuple[str, int]],
    chain_col: str,
    resid_col: str,
) -> pd.DataFrame:
    if df.empty or not residue_set:
        return df.iloc[0:0].copy()

    missing = {chain_col, resid_col}.difference(df.columns)
    if missing:
        raise KeyError(f"Dataframe is missing required columns: {sorted(missing)}")

    out = df.copy()
    out[chain_col] = out[chain_col].astype(str)
    resid_numeric = pd.to_numeric(out[resid_col], errors="coerce")

    mask = [
        (chain_id, int(resid)) in residue_set if pd.notna(resid) else False
        for chain_id, resid in zip(out[chain_col], resid_numeric)
    ]
    return out.loc[mask].copy()


def filter_rmsf_outputs_to_residue_set(
    all_rmsf_df: pd.DataFrame,
    *,
    residue_set: Iterable[Tuple[str, int]],
    delta_rows_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Filter RMSF outputs to a residue subset and recompute summaries."""
    required_all = {"artifact", "chain", "resid", "rmsf"}
    missing_all = required_all.difference(all_rmsf_df.columns)
    if missing_all:
        raise KeyError(f"all_rmsf_df is missing required columns: {sorted(missing_all)}")

    normalized_residue_set = {(str(chain), int(resid)) for chain, resid in residue_set}

    filtered_all_df = _filter_df_to_residue_set(
        all_rmsf_df,
        residue_set=normalized_residue_set,
        chain_col="chain",
        resid_col="resid",
    )

    artifact_names = sorted(all_rmsf_df["artifact"].dropna().astype(str).unique().tolist())
    summary_df = pd.DataFrame({"artifact": artifact_names})

    if not filtered_all_df.empty:
        grouped_summary = (
            filtered_all_df.groupby("artifact", as_index=False)
            .agg(
                n_residues=("rmsf", "size"),
                rmsf_mean=("rmsf", "mean"),
                rmsf_median=("rmsf", "median"),
                rmsf_max=("rmsf", "max"),
                rmsf_std=("rmsf", lambda x: float(np.std(x, ddof=0))),
            )
            .copy()
        )
        summary_df = summary_df.merge(grouped_summary, on="artifact", how="left")
    else:
        summary_df["n_residues"] = 0
        summary_df["rmsf_mean"] = np.nan
        summary_df["rmsf_median"] = np.nan
        summary_df["rmsf_max"] = np.nan
        summary_df["rmsf_std"] = np.nan

    if "n_residues" not in summary_df.columns:
        summary_df["n_residues"] = 0

    summary_df["n_residues"] = summary_df["n_residues"].fillna(0).astype(int)
    summary_df["status"] = np.where(summary_df["n_residues"] > 0, "ok", "empty")
    summary_df["error"] = ""

    filtered_delta_rows_df = pd.DataFrame()
    filtered_delta_summary_df = pd.DataFrame()

    if delta_rows_df is not None and not delta_rows_df.empty:
        required_delta = {"artifact", "chain", "resid", "delta_rmsf_vs_wt"}
        missing_delta = required_delta.difference(delta_rows_df.columns)
        if missing_delta:
            raise KeyError(f"delta_rows_df is missing required columns: {sorted(missing_delta)}")

        filtered_delta_rows_df = _filter_df_to_residue_set(
            delta_rows_df,
            residue_set=normalized_residue_set,
            chain_col="chain",
            resid_col="resid",
        )

        if not filtered_delta_rows_df.empty:
            if "abs_delta_rmsf_vs_wt" not in filtered_delta_rows_df.columns:
                filtered_delta_rows_df["abs_delta_rmsf_vs_wt"] = filtered_delta_rows_df["delta_rmsf_vs_wt"].abs()

            filtered_delta_summary_df = (
                filtered_delta_rows_df.groupby("artifact", as_index=False)
                .agg(
                    n_residues=("delta_rmsf_vs_wt", "size"),
                    delta_mean=("delta_rmsf_vs_wt", "mean"),
                    delta_median=("delta_rmsf_vs_wt", "median"),
                    delta_abs_mean=("abs_delta_rmsf_vs_wt", "mean"),
                    delta_abs_max=("abs_delta_rmsf_vs_wt", "max"),
                )
                .sort_values("delta_abs_mean", ascending=False)
                .reset_index(drop=True)
            )

    return {
        "all_rmsf_df": filtered_all_df,
        "summary_df": summary_df,
        "delta_rows_df": filtered_delta_rows_df,
        "delta_summary_df": filtered_delta_summary_df,
    }


def filter_rmsf_outputs_to_interface(
    all_rmsf_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    *,
    delta_rows_df: Optional[pd.DataFrame] = None,
    chain_col: str = "query_chain",
    resid_col: str = "query_resid",
) -> Dict[str, pd.DataFrame]:
    """Filter RMSF outputs to residues listed in an interface dataframe."""
    residue_set = interface_residue_index(
        interface_df,
        chain_col=chain_col,
        resid_col=resid_col,
    )

    out = filter_rmsf_outputs_to_residue_set(
        all_rmsf_df,
        residue_set=residue_set,
        delta_rows_df=delta_rows_df,
    )
    out["interface_residue_index"] = pd.DataFrame(
        [{"chain": chain_id, "resid": resid} for chain_id, resid in sorted(residue_set)]
    )
    return out


def compute_protein_dna_min_distances(
    structure_path: Path | str,
    *,
    protein_chains: Sequence[str],
    dna_chains: Sequence[str],
    include_resname: bool = True,
    first_model_only: bool = True,
) -> pd.DataFrame:
    """Compute per-residue protein->DNA distance summaries.

    The function scans residues in ``protein_chains`` and finds each residue's
    nearest atom in ``dna_chains``. In addition to the minimum atom distance,
    it also reports atom-level mean and median nearest-neighbor distances for
    each protein residue. Distances are reported in Angstrom. A
    KD-tree is used for the nearest-neighbor lookup to avoid repeated
    protein-atom x DNA-atom full scans.
    """
    protein_group = [str(c).strip() for c in protein_chains if str(c).strip()]
    dna_group = [str(c).strip() for c in dna_chains if str(c).strip()]

    if not protein_group or not dna_group:
        raise ValueError("protein_chains and dna_chains must be non-empty")
    if set(protein_group).intersection(dna_group):
        raise ValueError("protein_chains and dna_chains must be disjoint")

    structure = _load_structure_any_format(structure_path)

    rows: List[Dict[str, Any]] = []

    for model in structure:
        chain_map = {str(chain.id): chain for chain in model}
        query_chains = [chain_map[c] for c in protein_group if c in chain_map]
        partner_chains = [chain_map[c] for c in dna_group if c in chain_map]

        if not query_chains or not partner_chains:
            if first_model_only:
                break
            continue

        partner_coords: List[np.ndarray] = []
        partner_meta: List[Tuple[str, int, str]] = []

        for dna_chain in partner_chains:
            for dna_residue in dna_chain:
                dna_resname = str(dna_residue.resname).strip().upper()
                if dna_resname in {"HOH", "WAT"}:
                    continue

                for dna_atom in dna_residue.get_atoms():
                    partner_coords.append(np.asarray(dna_atom.coord, dtype=np.float64))
                    partner_meta.append(
                        (
                            str(dna_chain.id),
                            int(dna_residue.id[1]),
                            str(dna_residue.resname).strip(),
                        )
                    )

        if not partner_coords:
            if first_model_only:
                break
            continue

        partner_arr = np.vstack(partner_coords)
        partner_tree = cKDTree(partner_arr)

        for protein_chain in query_chains:
            for protein_residue in protein_chain:
                # Keep standard protein residues only.
                if protein_residue.id[0] != " ":
                    continue

                residue_atoms = list(protein_residue.get_atoms())
                if not residue_atoms:
                    continue

                residue_coords = np.vstack(
                    [np.asarray(atom.coord, dtype=np.float64) for atom in residue_atoms]
                )
                nearest_distances, nearest_indices = partner_tree.query(residue_coords, k=1)

                if np.isscalar(nearest_distances):
                    best_dist = float(nearest_distances)
                    mean_dist = float(nearest_distances)
                    median_dist = float(nearest_distances)
                    best_meta = partner_meta[int(nearest_indices)]
                else:
                    nearest_distances = np.asarray(nearest_distances, dtype=np.float64)
                    nearest_indices = np.asarray(nearest_indices)
                    if nearest_distances.size == 0:
                        continue

                    mean_dist = float(np.mean(nearest_distances))
                    median_dist = float(np.median(nearest_distances))
                    best_atom_idx = int(np.argmin(nearest_distances))
                    best_dist = float(nearest_distances[best_atom_idx])
                    best_meta = partner_meta[int(nearest_indices[best_atom_idx])]

                dna_chain_id, dna_resid, dna_resname = best_meta
                rows.append(
                    {
                        "model_id": str(model.id),
                        "protein_chain": str(protein_chain.id),
                        "protein_resid": int(protein_residue.id[1]),
                        "protein_resname": str(protein_residue.resname).strip() if include_resname else None,
                        "dna_chain": dna_chain_id,
                        "dna_resid": int(dna_resid),
                        "dna_resname": dna_resname if include_resname else None,
                        "mean_distance": float(mean_dist),
                        "median_distance": float(median_dist),
                        "min_distance": float(best_dist),
                    }
                )

        if first_model_only:
            break

    out_cols = [
        "model_id",
        "protein_chain",
        "protein_resid",
        "protein_resname",
        "dna_chain",
        "dna_resid",
        "dna_resname",
        "mean_distance",
        "median_distance",
        "min_distance",
    ]

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out_df = pd.DataFrame(rows)
    out_df = (
        out_df.sort_values(["model_id", "protein_chain", "protein_resid", "min_distance"])
        .drop_duplicates(subset=["model_id", "protein_chain", "protein_resid"], keep="first")
        .reset_index(drop=True)
    )
    return out_df[out_cols]


def _aggregate_protein_dna_distances_from_structure_files(
    structure_files: Sequence[Path | str],
    *,
    protein_chains: Sequence[str],
    dna_chains: Sequence[str],
    include_resname: bool = True,
    first_model_only: bool = True,
    require_common: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_structure_tables: List[pd.DataFrame] = []

    for structure_idx, structure_file in enumerate(structure_files):
        dist_df = compute_protein_dna_min_distances(
            structure_file,
            protein_chains=protein_chains,
            dna_chains=dna_chains,
            include_resname=include_resname,
            first_model_only=first_model_only,
        )
        if dist_df.empty:
            continue

        dist_df = dist_df.copy()
        dist_df.insert(0, "structure_id", int(structure_idx))
        dist_df.insert(1, "structure_path", str(structure_file))
        per_structure_tables.append(dist_df)

    if not per_structure_tables:
        return pd.DataFrame(), pd.DataFrame()

    all_distances_df = pd.concat(per_structure_tables, ignore_index=True)

    key_cols = ["protein_chain", "protein_resid"]
    if include_resname and "protein_resname" in all_distances_df.columns:
        key_cols.append("protein_resname")

    total_structures = int(all_distances_df["structure_id"].nunique())

    residue_summary_df = (
        all_distances_df.groupby(key_cols, as_index=False)
        .agg(
            n_structures=("structure_id", "nunique"),
            mean_distance=("mean_distance", "mean"),
            median_distance=("median_distance", "mean"),
            mean_min_distance=("min_distance", "mean"),
            median_min_distance=("min_distance", "median"),
            min_distance=("min_distance", "min"),
            std_distance=("min_distance", lambda x: float(np.std(x, ddof=0))),
        )
        .copy()
    )

    if require_common and total_structures > 0:
        residue_summary_df = residue_summary_df[residue_summary_df["n_structures"] == total_structures].copy()

    residue_summary_df = residue_summary_df.sort_values("mean_min_distance", ascending=True).reset_index(drop=True)
    return residue_summary_df, all_distances_df


def summarize_protein_dna_distances(
    residue_summary_df: pd.DataFrame,
    *,
    interface_residue_set: Optional[Iterable[Tuple[str, int]]] = None,
    distance_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Summarize whole-protein and interface-only protein->DNA distance metrics."""
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive")

    out: Dict[str, Any] = {
        "n_residues": 0,
        "distance_threshold": float(distance_threshold),

        # Whole-protein statistics.
        "whole_protein_mean_of_mean_distance": np.nan,
        "whole_protein_mean_of_median_distance": np.nan,
        "whole_protein_median_of_mean_distance": np.nan,
        "whole_protein_median_of_median_distance": np.nan,
        "whole_protein_mean_min_distance": np.nan,
        "whole_protein_mean_of_mean_min_distance": np.nan,
        "whole_protein_mean_of_median_min_distance": np.nan,
        "whole_protein_median_min_distance": np.nan,
        "whole_protein_min_distance": np.nan,
        "whole_protein_max_min_distance": np.nan,
        "whole_protein_iqr_min_distance": np.nan,
        "whole_protein_p90_min_distance": np.nan,
        "whole_protein_std_min_distance": np.nan,
        "whole_protein_range_min_distance": np.nan,

        # Keep legacy naming, plus explicit fraction form.
        "pct_residues_within_threshold": np.nan,
        "whole_protein_fraction_within_threshold": np.nan,

        # Interface-only statistics.
        "interface_residue_count": 0,
        "interface_mean_of_mean_distance": np.nan,
        "interface_mean_of_median_distance": np.nan,
        "interface_median_of_mean_distance": np.nan,
        "interface_median_of_median_distance": np.nan,
        "interface_mean_min_distance": np.nan,
        "interface_mean_of_mean_min_distance": np.nan,
        "interface_mean_of_median_min_distance": np.nan,
        "interface_median_min_distance": np.nan,
        "interface_min_distance": np.nan,
        "interface_max_min_distance": np.nan,
        "interface_iqr_min_distance": np.nan,
        "interface_p90_min_distance": np.nan,
        "interface_std_min_distance": np.nan,
        "interface_range_min_distance": np.nan,
        "interface_fraction_within_threshold": np.nan,
        "interface_pct_within_threshold": np.nan,

        # Relationship metrics.
        "interface_over_whole_mean_ratio": np.nan,
        "interface_minus_whole_mean_distance": np.nan,
        "interface_minus_whole_iqr_distance": np.nan,
    }

    if residue_summary_df.empty:
        return out

    if "mean_min_distance" not in residue_summary_df.columns:
        raise KeyError("residue_summary_df is missing required column: mean_min_distance")

    whole = residue_summary_df.copy()
    whole_vals = pd.to_numeric(whole["mean_min_distance"], errors="coerce")
    whole_vals = whole_vals[whole_vals.notna()].astype(float).to_numpy()

    if whole_vals.size == 0:
        return out

    out["n_residues"] = int(len(whole_vals))
    whole_mean = float(np.mean(whole_vals))

    if "mean_distance" in whole.columns:
        whole_mean_distance_vals = pd.to_numeric(whole["mean_distance"], errors="coerce")
        whole_mean_distance_vals = whole_mean_distance_vals[whole_mean_distance_vals.notna()].astype(float).to_numpy()
    else:
        whole_mean_distance_vals = np.array([], dtype=float)

    if whole_mean_distance_vals.size == 0:
        whole_mean_distance_vals = whole_vals

    if "median_distance" in whole.columns:
        whole_median_distance_vals = pd.to_numeric(whole["median_distance"], errors="coerce")
        whole_median_distance_vals = (
            whole_median_distance_vals[whole_median_distance_vals.notna()].astype(float).to_numpy()
        )
    else:
        whole_median_distance_vals = np.array([], dtype=float)

    if whole_median_distance_vals.size == 0:
        whole_median_distance_vals = whole_mean_distance_vals

    out["whole_protein_mean_of_mean_distance"] = float(np.mean(whole_mean_distance_vals))
    out["whole_protein_mean_of_median_distance"] = float(np.mean(whole_median_distance_vals))
    out["whole_protein_median_of_mean_distance"] = float(np.median(whole_mean_distance_vals))
    out["whole_protein_median_of_median_distance"] = float(np.median(whole_median_distance_vals))

    out["whole_protein_mean_min_distance"] = whole_mean
    out["whole_protein_mean_of_mean_min_distance"] = whole_mean

    if "median_min_distance" in whole.columns:
        whole_median_vals = pd.to_numeric(whole["median_min_distance"], errors="coerce")
        whole_median_vals = whole_median_vals[whole_median_vals.notna()].astype(float).to_numpy()
    else:
        whole_median_vals = np.array([], dtype=float)

    if whole_median_vals.size == 0:
        whole_median_vals = whole_vals

    out["whole_protein_mean_of_median_min_distance"] = float(np.mean(whole_median_vals))
    out["whole_protein_median_min_distance"] = float(np.median(whole_vals))
    out["whole_protein_max_min_distance"] = float(np.max(whole_vals))
    out["whole_protein_iqr_min_distance"] = float(np.percentile(whole_vals, 75) - np.percentile(whole_vals, 25))
    out["whole_protein_p90_min_distance"] = float(np.percentile(whole_vals, 90))
    out["whole_protein_std_min_distance"] = float(np.std(whole_vals, ddof=0))
    out["whole_protein_range_min_distance"] = float(np.max(whole_vals) - np.min(whole_vals))

    if "min_distance" in whole.columns:
        min_vals = pd.to_numeric(whole["min_distance"], errors="coerce")
        min_vals = min_vals[min_vals.notna()].astype(float).to_numpy()
        out["whole_protein_min_distance"] = float(np.min(min_vals)) if min_vals.size else np.nan
    else:
        out["whole_protein_min_distance"] = float(np.min(whole_vals))

    whole_fraction_within_threshold = float(np.mean(whole_vals <= float(distance_threshold)))
    out["whole_protein_fraction_within_threshold"] = whole_fraction_within_threshold
    out["pct_residues_within_threshold"] = float(whole_fraction_within_threshold * 100.0)

    normalized_interface_set = {
        (str(chain_id), int(resid))
        for chain_id, resid in (interface_residue_set or [])
    }

    if not normalized_interface_set:
        return out

    if "protein_chain" not in whole.columns or "protein_resid" not in whole.columns:
        raise KeyError("residue_summary_df must include protein_chain and protein_resid for interface summary")

    resid_num = pd.to_numeric(whole["protein_resid"], errors="coerce")
    iface_mask = [
        (str(chain), int(resid)) in normalized_interface_set if pd.notna(resid) else False
        for chain, resid in zip(whole["protein_chain"], resid_num)
    ]

    interface_df = whole.loc[iface_mask].copy()
    if interface_df.empty:
        return out

    iface_vals = pd.to_numeric(interface_df["mean_min_distance"], errors="coerce")
    iface_vals = iface_vals[iface_vals.notna()].astype(float).to_numpy()
    if iface_vals.size == 0:
        return out

    out["interface_residue_count"] = int(len(iface_vals))
    iface_mean = float(np.mean(iface_vals))

    if "mean_distance" in interface_df.columns:
        iface_mean_distance_vals = pd.to_numeric(interface_df["mean_distance"], errors="coerce")
        iface_mean_distance_vals = iface_mean_distance_vals[iface_mean_distance_vals.notna()].astype(float).to_numpy()
    else:
        iface_mean_distance_vals = np.array([], dtype=float)

    if iface_mean_distance_vals.size == 0:
        iface_mean_distance_vals = iface_vals

    if "median_distance" in interface_df.columns:
        iface_median_distance_vals = pd.to_numeric(interface_df["median_distance"], errors="coerce")
        iface_median_distance_vals = (
            iface_median_distance_vals[iface_median_distance_vals.notna()].astype(float).to_numpy()
        )
    else:
        iface_median_distance_vals = np.array([], dtype=float)

    if iface_median_distance_vals.size == 0:
        iface_median_distance_vals = iface_mean_distance_vals

    out["interface_mean_of_mean_distance"] = float(np.mean(iface_mean_distance_vals))
    out["interface_mean_of_median_distance"] = float(np.mean(iface_median_distance_vals))
    out["interface_median_of_mean_distance"] = float(np.median(iface_mean_distance_vals))
    out["interface_median_of_median_distance"] = float(np.median(iface_median_distance_vals))

    out["interface_mean_min_distance"] = iface_mean
    out["interface_mean_of_mean_min_distance"] = iface_mean

    if "median_min_distance" in interface_df.columns:
        iface_median_vals = pd.to_numeric(interface_df["median_min_distance"], errors="coerce")
        iface_median_vals = iface_median_vals[iface_median_vals.notna()].astype(float).to_numpy()
    else:
        iface_median_vals = np.array([], dtype=float)

    if iface_median_vals.size == 0:
        iface_median_vals = iface_vals

    out["interface_mean_of_median_min_distance"] = float(np.mean(iface_median_vals))
    out["interface_median_min_distance"] = float(np.median(iface_vals))
    out["interface_max_min_distance"] = float(np.max(iface_vals))
    out["interface_iqr_min_distance"] = float(np.percentile(iface_vals, 75) - np.percentile(iface_vals, 25))
    out["interface_p90_min_distance"] = float(np.percentile(iface_vals, 90))
    out["interface_std_min_distance"] = float(np.std(iface_vals, ddof=0))
    out["interface_range_min_distance"] = float(np.max(iface_vals) - np.min(iface_vals))

    if "min_distance" in interface_df.columns:
        iface_min_vals = pd.to_numeric(interface_df["min_distance"], errors="coerce")
        iface_min_vals = iface_min_vals[iface_min_vals.notna()].astype(float).to_numpy()
        out["interface_min_distance"] = float(np.min(iface_min_vals)) if iface_min_vals.size else np.nan
    else:
        out["interface_min_distance"] = float(np.min(iface_vals))

    iface_fraction_within_threshold = float(np.mean(iface_vals <= float(distance_threshold)))
    out["interface_fraction_within_threshold"] = iface_fraction_within_threshold
    out["interface_pct_within_threshold"] = float(iface_fraction_within_threshold * 100.0)

    if np.isfinite(whole_mean) and np.abs(float(whole_mean)) > 1e-12 and np.isfinite(iface_mean):
        out["interface_over_whole_mean_ratio"] = float(float(iface_mean) / float(whole_mean))
        out["interface_minus_whole_mean_distance"] = float(float(iface_mean) - float(whole_mean))

    whole_iqr = out["whole_protein_iqr_min_distance"]
    iface_iqr = out["interface_iqr_min_distance"]
    if np.isfinite(whole_iqr) and np.isfinite(iface_iqr):
        out["interface_minus_whole_iqr_distance"] = float(float(iface_iqr) - float(whole_iqr))

    return out


def _backfill_protein_dna_summary_columns(
    summary_df: pd.DataFrame,
    residue_df: pd.DataFrame,
) -> pd.DataFrame:
    """Backfill newly added summary columns from residue-level aggregates.

    This keeps older cached summary CSVs usable when new aggregate columns are added.
    """
    out_df = summary_df.copy()

    target_cols = [
        "whole_protein_mean_of_mean_distance",
        "whole_protein_mean_of_median_distance",
        "whole_protein_median_of_mean_distance",
        "whole_protein_median_of_median_distance",
        "interface_mean_of_mean_distance",
        "interface_mean_of_median_distance",
        "interface_median_of_mean_distance",
        "interface_median_of_median_distance",
        "whole_protein_mean_of_median_min_distance",
        "interface_mean_of_median_min_distance",
    ]
    for col in target_cols:
        if col not in out_df.columns:
            out_df[col] = np.nan

    if out_df.empty or residue_df.empty:
        return out_df

    if "artifact" not in out_df.columns or "artifact" not in residue_df.columns:
        return out_df

    tmp = residue_df.copy()
    tmp["_artifact_key"] = tmp["artifact"].astype(str)
    artifact_key = out_df["artifact"].astype(str)

    def _fill_metric(col_name: str, series: pd.Series, reduce_fn: str, interface_only: bool = False) -> None:
        vals = pd.to_numeric(series, errors="coerce")
        work = tmp.copy()
        work["_metric"] = vals

        if interface_only:
            if "is_interface_residue" not in work.columns:
                return

            interface_raw = work["is_interface_residue"]
            if pd.api.types.is_bool_dtype(interface_raw):
                interface_mask = interface_raw.fillna(False)
            else:
                interface_mask = interface_raw.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
            work = work.loc[interface_mask].copy()
            if work.empty:
                return

        grouped = work.groupby("_artifact_key")["_metric"]
        if reduce_fn == "mean":
            mapping = grouped.mean()
        elif reduce_fn == "median":
            mapping = grouped.median()
        else:
            raise ValueError(f"Unsupported reduce_fn: {reduce_fn}")

        missing_mask = out_df[col_name].isna()
        out_df.loc[missing_mask, col_name] = artifact_key.loc[missing_mask].map(mapping)

    if "mean_distance" in tmp.columns:
        _fill_metric("whole_protein_mean_of_mean_distance", tmp["mean_distance"], "mean", interface_only=False)
        _fill_metric("whole_protein_median_of_mean_distance", tmp["mean_distance"], "median", interface_only=False)
        _fill_metric("interface_mean_of_mean_distance", tmp["mean_distance"], "mean", interface_only=True)
        _fill_metric("interface_median_of_mean_distance", tmp["mean_distance"], "median", interface_only=True)

    if "median_distance" in tmp.columns:
        _fill_metric("whole_protein_mean_of_median_distance", tmp["median_distance"], "mean", interface_only=False)
        _fill_metric("whole_protein_median_of_median_distance", tmp["median_distance"], "median", interface_only=False)
        _fill_metric("interface_mean_of_median_distance", tmp["median_distance"], "mean", interface_only=True)
        _fill_metric("interface_median_of_median_distance", tmp["median_distance"], "median", interface_only=True)

    if "median_min_distance" in tmp.columns:
        _fill_metric("whole_protein_mean_of_median_min_distance", tmp["median_min_distance"], "mean", interface_only=False)
        _fill_metric("interface_mean_of_median_min_distance", tmp["median_min_distance"], "mean", interface_only=True)

    return out_df


def load_protein_dna_distance_batch_outputs(
    config: StructureMetricsConfig,
    *,
    output_subdir: str = "dna_distance",
) -> Dict[str, Any]:
    """Load previously written protein->DNA distance batch outputs from CSV files."""
    out_dir = config.resolved_output_dir() / output_subdir
    summary_path = out_dir / f"{config.mut_name}_protein_dna_distance_summary.csv"
    residue_path = out_dir / f"{config.mut_name}_protein_dna_distance_residue_table.csv"

    required_paths = [summary_path, residue_path]
    missing_required = [str(p) for p in required_paths if not p.exists()]
    if missing_required:
        raise FileNotFoundError(
            "Missing required protein-DNA distance output files: " + ", ".join(missing_required)
        )

    summary_df = pd.read_csv(summary_path)
    residue_df = pd.read_csv(residue_path)
    summary_df = _backfill_protein_dna_summary_columns(summary_df, residue_df)
    return {
        "summary_df": summary_df,
        "residue_df": residue_df,
        "summary_path": str(summary_path),
        "residue_path": str(residue_path),
    }


def run_protein_dna_distance_batch(
    config: StructureMetricsConfig,
    *,
    protein_chains: Sequence[str],
    dna_chains: Sequence[str],
    interface_df: Optional[pd.DataFrame] = None,
    interface_chain_col: str = "query_chain",
    interface_resid_col: str = "query_resid",
    include_resname: bool = True,
    first_model_only: bool = True,
    require_common: bool = True,
    distance_threshold: float = 5.0,
    show_progress: bool = True,
    max_artifacts: int | None = None,
    output_subdir: str = "dna_distance",
    reuse_existing: bool = True,
    allow_recompute_if_missing: bool = False,
    checkpoint_every: int | None = 1,
    retry_errors: bool = False,
    print_cached_errors: bool = True,
    exclude_artifacts: Optional[Iterable[Path | str]] = None,
) -> Dict[str, Any]:
    """Run batch protein->DNA distance analysis for all artifacts in config.

    Existing summary/residue CSVs are treated as checkpoints when
    ``reuse_existing=True``. Complete cached outputs are returned immediately,
    incomplete outputs are resumed, and cached rows with ``status == "error"``
    are retried only when ``retry_errors=True``.
    """
    protein_group = [str(c).strip() for c in protein_chains if str(c).strip()]
    dna_group = [str(c).strip() for c in dna_chains if str(c).strip()]

    if not protein_group or not dna_group:
        raise ValueError("protein_chains and dna_chains must be non-empty")
    if set(protein_group).intersection(dna_group):
        raise ValueError("protein_chains and dna_chains must be disjoint")

    excluded_artifact_set = {
        canonical_artifact_stem(artifact)
        for artifact in (exclude_artifacts or [])
        if str(artifact).strip()
    }

    out_dir = config.resolved_output_dir() / output_subdir
    summary_path = out_dir / f"{config.mut_name}_protein_dna_distance_summary.csv"
    residue_path = out_dir / f"{config.mut_name}_protein_dna_distance_residue_table.csv"
    required_residue_cols = {"mean_distance", "median_distance"}

    def _empty_residue_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "artifact",
                "protein_chain",
                "protein_resid",
                "protein_resname",
                "n_structures",
                "mean_distance",
                "median_distance",
                "mean_min_distance",
                "median_min_distance",
                "min_distance",
                "std_distance",
                "is_interface_residue",
            ]
        )

    def _emit_progress_message(level: str, message: str) -> None:
        text = f"[{level}] {message}"
        if show_progress:
            tqdm.write(text)
        else:
            print(text)

    def _format_error_context(artifact_name: str, artifact_path: Any, exc: Exception) -> str:
        return f"{artifact_name} ({artifact_path}): {type(exc).__name__}: {str(exc)[:500]}"

    def _write_outputs(
        summary_rows_to_write: Sequence[Dict[str, Any]],
        residue_tables_to_write: Sequence[pd.DataFrame],
        *,
        is_checkpoint: bool,
    ) -> Dict[str, Any]:
        summary_out = pd.DataFrame(summary_rows_to_write)
        sort_cols = {"status", "whole_protein_mean_min_distance"}
        if not summary_out.empty and sort_cols.issubset(summary_out.columns):
            summary_out = summary_out.sort_values(
                ["status", "whole_protein_mean_min_distance"],
                ascending=[True, True],
            ).reset_index(drop=True)

        if residue_tables_to_write:
            residue_out = pd.concat(residue_tables_to_write, ignore_index=True)
        else:
            residue_out = _empty_residue_df()

        out_dir.mkdir(parents=True, exist_ok=True)
        summary_out.to_csv(summary_path, index=False)
        residue_out.to_csv(residue_path, index=False)

        if is_checkpoint:
            _emit_progress_message(
                "INFO",
                (
                    f"Saved DNA-distance checkpoint for {config.mut_name}: "
                    f"{len(summary_out)} summary rows -> {summary_path}"
                ),
            )

        return {
            "summary_df": summary_out,
            "residue_df": residue_out,
            "summary_path": str(summary_path),
            "residue_path": str(residue_path),
        }

    artifacts = iter_artifacts(config)
    if excluded_artifact_set:
        before_exclude = len(artifacts)
        artifacts = [
            artifact
            for artifact in artifacts
            if canonical_artifact_stem(artifact) not in excluded_artifact_set
        ]
        excluded_count = before_exclude - len(artifacts)
        if excluded_count:
            _emit_progress_message(
                "INFO",
                (
                    f"Skipping {excluded_count} explicitly excluded DNA-distance artifact(s): "
                    f"{', '.join(sorted(excluded_artifact_set))}"
                ),
            )

    if max_artifacts is not None:
        artifacts = artifacts[: int(max_artifacts)]

    if not artifacts:
        if reuse_existing:
            try:
                existing = load_protein_dna_distance_batch_outputs(
                    config,
                    output_subdir=output_subdir,
                )
                _emit_progress_message(
                    "WARN",
                    (
                        f"No artifacts found under {config.read_root_path}; "
                        f"using cached DNA-distance outputs for {config.mut_name}."
                    ),
                )
                return existing
            except FileNotFoundError:
                pass
        raise FileNotFoundError(
            f"No artifacts found for source_type={config.source_type!r} under {config.read_root_path}"
        )

    expected_artifact_names = [canonical_artifact_stem(artifact) for artifact in artifacts]
    expected_artifact_set = set(expected_artifact_names)

    existing_summary_df = pd.DataFrame()
    existing_residue_df = _empty_residue_df()
    has_new_distance_cols = False

    if reuse_existing:
        try:
            existing = load_protein_dna_distance_batch_outputs(
                config,
                output_subdir=output_subdir,
            )

            # If cached residue-level columns required for avg/median-distance
            # summaries are missing, allow optional recompute from source data.
            residue_cols = set(existing["residue_df"].columns)
            has_new_distance_cols = required_residue_cols.issubset(residue_cols)
            existing_summary_df = existing["summary_df"].copy()
            existing_residue_df = existing["residue_df"].copy()
            if excluded_artifact_set:
                if "artifact" in existing_summary_df.columns:
                    keep_summary = ~existing_summary_df["artifact"].astype(str).isin(excluded_artifact_set)
                    existing_summary_df = existing_summary_df.loc[keep_summary].copy()
                if "artifact" in existing_residue_df.columns:
                    keep_residue = ~existing_residue_df["artifact"].astype(str).isin(excluded_artifact_set)
                    existing_residue_df = existing_residue_df.loc[keep_residue].copy()
                existing = {
                    **existing,
                    "summary_df": existing_summary_df,
                    "residue_df": existing_residue_df,
                }

            existing_artifact_set = (
                set(existing_summary_df["artifact"].astype(str))
                if "artifact" in existing_summary_df.columns
                else set()
            )
            missing_artifacts = expected_artifact_set.difference(existing_artifact_set)

            error_artifact_set: set[str] = set()
            if {"artifact", "status"}.issubset(existing_summary_df.columns):
                error_mask = existing_summary_df["status"].astype(str).eq("error")
                error_artifact_set = set(existing_summary_df.loc[error_mask, "artifact"].astype(str))
            retry_error_artifact_set = expected_artifact_set.intersection(error_artifact_set)

            if print_cached_errors and error_artifact_set:
                cached_error_df = existing_summary_df[
                    existing_summary_df["artifact"].astype(str).isin(sorted(error_artifact_set))
                ]
                for row in cached_error_df.itertuples(index=False):
                    artifact_name = str(getattr(row, "artifact", "unknown"))
                    source_path = str(getattr(row, "source_path", "unknown"))
                    error_text = str(getattr(row, "error", ""))
                    _emit_progress_message(
                        "ERROR",
                        f"Cached DNA-distance error for {artifact_name} ({source_path}): {error_text[:500]}",
                    )

            if not has_new_distance_cols and allow_recompute_if_missing:
                _emit_progress_message(
                    "INFO",
                    (
                        f"Cached DNA-distance residue table for {config.mut_name} lacks "
                        f"{sorted(required_residue_cols)}; recomputing from source artifacts."
                    ),
                )
                existing_summary_df = pd.DataFrame()
                existing_residue_df = _empty_residue_df()
            elif not missing_artifacts and not (retry_errors and retry_error_artifact_set):
                if show_progress:
                    tqdm.write(
                        f"[INFO] Using cached DNA-distance outputs for {config.mut_name}; "
                        "set reuse_existing=False for a full live rerun or retry_errors=True "
                        "to reprocess cached error rows."
                    )
                return existing
            elif not has_new_distance_cols and not allow_recompute_if_missing:
                return existing
            else:
                if missing_artifacts:
                    _emit_progress_message(
                        "INFO",
                        (
                            f"Resuming DNA-distance batch for {config.mut_name}: "
                            f"{len(missing_artifacts)} missing artifact rows will be processed."
                        ),
                    )
                if retry_errors and retry_error_artifact_set:
                    retry_count = len(retry_error_artifact_set)
                    _emit_progress_message(
                        "INFO",
                        f"Retrying {retry_count} cached DNA-distance error rows for {config.mut_name}.",
                    )
        except FileNotFoundError:
            if not allow_recompute_if_missing:
                raise

    interface_residue_set: set[Tuple[str, int]] = set()
    if interface_df is not None and not interface_df.empty:
        interface_residue_set = interface_residue_index(
            interface_df,
            chain_col=interface_chain_col,
            resid_col=interface_resid_col,
        )

    artifacts_to_retry: set[str] = set()
    if retry_errors and {"artifact", "status"}.issubset(existing_summary_df.columns):
        error_mask = existing_summary_df["status"].astype(str).eq("error")
        artifacts_to_retry = set(existing_summary_df.loc[error_mask, "artifact"].astype(str))
        artifacts_to_retry = artifacts_to_retry.intersection(expected_artifact_set)

    existing_done: set[str] = set()
    if "artifact" in existing_summary_df.columns:
        existing_done = set(existing_summary_df["artifact"].astype(str)).difference(artifacts_to_retry)

    artifacts_to_process = [
        artifact
        for artifact, artifact_name in zip(artifacts, expected_artifact_names)
        if artifact_name not in existing_done
    ]

    if existing_summary_df.empty:
        summary_rows: List[Dict[str, Any]] = []
    else:
        keep_existing_mask = ~existing_summary_df["artifact"].astype(str).isin(artifacts_to_retry)
        summary_rows = existing_summary_df.loc[keep_existing_mask].to_dict("records")

    if existing_residue_df.empty or "artifact" not in existing_residue_df.columns:
        residue_tables: List[pd.DataFrame] = []
    else:
        keep_residue_mask = ~existing_residue_df["artifact"].astype(str).isin(artifacts_to_retry)
        kept_residue_df = existing_residue_df.loc[keep_residue_mask].copy()
        residue_tables = [kept_residue_df] if not kept_residue_df.empty else []

    if not artifacts_to_process:
        return _write_outputs(summary_rows, residue_tables, is_checkpoint=False)

    iterator = (
        tqdm(artifacts_to_process, desc=f"DNA distance {config.mut_name}", unit="artifact")
        if show_progress
        else artifacts_to_process
    )

    def _short_artifact_label(path_like: Any, max_len: int = 72) -> str:
        name = Path(str(path_like)).name
        if len(name) <= max_len:
            return name
        keep = max_len - 3
        return f"...{name[-keep:]}"

    checkpoint_every_n = int(checkpoint_every) if checkpoint_every else 0
    processed_since_checkpoint = 0

    for artifact in iterator:
        artifact_name = canonical_artifact_stem(artifact)
        if show_progress and hasattr(iterator, "set_postfix_str"):
            iterator.set_postfix_str(f"current={_short_artifact_label(artifact)}", refresh=True)

        try:
            if config.source_type == "folder":
                structure_files = drop_top_ranked_duplicate(_discover_structure_files(artifact))
            elif config.source_type == "tar":
                with tempfile.TemporaryDirectory() as tmpdir:
                    with tarfile.open(artifact, "r:*") as tar_obj:
                        _safe_extract_tar(tar_obj, tmpdir)
                    structure_files = drop_top_ranked_duplicate(_discover_structure_files(tmpdir))

                    residue_summary_df, _ = _aggregate_protein_dna_distances_from_structure_files(
                        structure_files,
                        protein_chains=protein_group,
                        dna_chains=dna_group,
                        include_resname=include_resname,
                        first_model_only=first_model_only,
                        require_common=require_common,
                    )
            else:
                raise ValueError(f"Unsupported config.source_type: {config.source_type}")

            if config.source_type == "folder":
                residue_summary_df, _ = _aggregate_protein_dna_distances_from_structure_files(
                    structure_files,
                    protein_chains=protein_group,
                    dna_chains=dna_group,
                    include_resname=include_resname,
                    first_model_only=first_model_only,
                    require_common=require_common,
                )

            n_structures_used = int(len(structure_files))

            if residue_summary_df.empty:
                summary_row: Dict[str, Any] = {
                    "artifact": artifact_name,
                    "source_path": str(artifact),
                    "status": "empty",
                    "n_structures_used": n_structures_used,
                    "error": "",
                }
                summary_row.update(
                    summarize_protein_dna_distances(
                        residue_summary_df,
                        interface_residue_set=interface_residue_set,
                        distance_threshold=distance_threshold,
                    )
                )
                summary_rows.append(summary_row)
                _emit_progress_message(
                    "WARN",
                    (
                        f"{artifact_name} ({artifact}): no comparable protein->DNA residues after "
                        f"aggregation (structures_used={n_structures_used}, require_common={require_common})"
                    ),
                )
            else:
                resid_numeric = pd.to_numeric(residue_summary_df["protein_resid"], errors="coerce")
                interface_mask = [
                    (str(chain), int(resid)) in interface_residue_set if (interface_residue_set and pd.notna(resid)) else False
                    for chain, resid in zip(residue_summary_df["protein_chain"], resid_numeric)
                ]

                residue_summary_df = residue_summary_df.copy()
                residue_summary_df.insert(0, "artifact", artifact_name)
                residue_summary_df["is_interface_residue"] = interface_mask
                residue_tables.append(residue_summary_df)

                summary_row = {
                    "artifact": artifact_name,
                    "source_path": str(artifact),
                    "status": "ok",
                    "n_structures_used": n_structures_used,
                    "error": "",
                }
                summary_row.update(
                    summarize_protein_dna_distances(
                        residue_summary_df,
                        interface_residue_set=interface_residue_set,
                        distance_threshold=distance_threshold,
                    )
                )
                summary_rows.append(summary_row)

        except Exception as exc:
            error_row = {
                "artifact": artifact_name,
                "source_path": str(artifact),
                "status": "error",
                "n_structures_used": 0,
                "error": str(exc)[:500],
            }
            error_row.update(
                summarize_protein_dna_distances(
                    pd.DataFrame(),
                    interface_residue_set=interface_residue_set,
                    distance_threshold=distance_threshold,
                )
            )
            summary_rows.append(error_row)
            _emit_progress_message("ERROR", _format_error_context(artifact_name, artifact, exc))

        processed_since_checkpoint += 1
        if checkpoint_every_n > 0 and processed_since_checkpoint >= checkpoint_every_n:
            _write_outputs(summary_rows, residue_tables, is_checkpoint=True)
            processed_since_checkpoint = 0

    return _write_outputs(summary_rows, residue_tables, is_checkpoint=False)


def rmsf_map_ca_from_structure_files(
    structure_files: Sequence[Path | str],
    *,
    include_resname: bool = False,
    require_common: bool = True,
    verbose: bool = True,
    reference_hint: str = "WT",
    aligner: str = "USalign",
    reference_path: Path | str | None = None,
    reference_mode: str = "first",
    save_aligned: bool = False,
    aligned_output_dir: Path | str | None = None,
) -> Tuple[Dict[Tuple[Any, ...], float], pd.DataFrame, List[Path]]:
    """
    Compute per-residue CA RMSF after explicit structural alignment.

    Parameters
    - structure_files: sequence of input structures (.pdb or .cif)
    - reference_path: optional explicit reference structure
    - reference_hint: used when reference_path is not provided
    - reference_mode: how the frame is picked when reference_path is absent.
      "first" (default) takes an arbitrary hint-matching structure and is what
      every prior result used. "medoid" takes the structure closest to all
      others; "mean" builds the consensus CA structure. Both cost one extra
      alignment pass and remove the arbitrariness of the default.
    - aligner: external aligner command (default: USalign)

    Returns
        - rmsf_dict: {(chain, resid) or (chain, resid, resname): rmsf_in_angstrom}
    - rmsf_df: dataframe sorted by RMSF descending
    - aligned_files: aligned PDB paths if persisted, else empty list

        Notes
        - Residue matching/alignment is always placement-based: (chain, resid).
            This preserves substituted positions when comparing mutants to WT.
        - include_resname controls output labeling only (metadata), not matching.
    """
    if len(structure_files) < 2:
        raise ValueError(f"Need at least 2 structures, found {len(structure_files)}")

    persisted_aligned = aligned_output_dir is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        conversion_dir = tmp_path / "converted"
        conversion_dir.mkdir(parents=True, exist_ok=True)

        prepared_files = _prepare_structure_files_for_rmsf(
            structure_files,
            conversion_dir=conversion_dir,
        )
        if len(prepared_files) < 2:
            raise ValueError(f"Need at least 2 valid structures, found {len(prepared_files)}")

        if reference_path is None and reference_mode != "first":
            reference_path = _resolve_ensemble_reference(
                prepared_files,
                reference_mode=reference_mode,
                reference_hint=reference_hint,
                work_dir=tmp_path,
                aligner=aligner,
                verbose=verbose,
            )

        reference = _choose_reference_structure(
            prepared_files,
            reference_hint=reference_hint,
            reference_path=reference_path,
        )

        if aligned_output_dir is not None:
            aligned_dir = Path(aligned_output_dir)
        else:
            aligned_dir = tmp_path / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        if save_aligned and aligned_output_dir is None and verbose:
            print("[rmsf] save_aligned=True but aligned_output_dir is None; aligned files will not be persisted")

        aligned_paths: List[Path] = []
        reference_out = aligned_dir / f"aligned_{reference.stem}.pdb"
        shutil.copy2(reference, reference_out)
        aligned_paths.append(reference_out)

        if verbose:
            print(f"[rmsf] Reference: {reference.name}")
            print(f"[rmsf] Aligning {len(prepared_files) - 1} structures")

        for src in prepared_files:
            if src.resolve() == reference.resolve():
                continue

            out_prefix = aligned_dir / f"aligned_{src.stem}_on_{reference.stem}"
            cmd = [aligner, str(src), str(reference), "-o", str(out_prefix)]
            if verbose:
                print(f"[rmsf] Aligning {src.name} -> {reference.name}")

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                stderr_msg = (proc.stderr or "").strip()[:400]
                raise RuntimeError(
                    f"Alignment failed for {src.name} using {aligner}. {stderr_msg}"
                )

            candidates = [
                out_prefix,
                out_prefix.with_suffix(".pdb"),
                Path(str(out_prefix) + ".pdb"),
            ]
            saved = next((p for p in candidates if p.exists()), None)
            if saved is None:
                raise FileNotFoundError(f"{aligner} did not produce aligned PDB for {src.name}")
            aligned_paths.append(saved)

        maps: List[Dict[Tuple[str, int], np.ndarray]] = []
        resname_maps: List[Dict[Tuple[str, int], str]] = []
        frame_is_reference: List[bool] = []
        for idx, aligned in enumerate(aligned_paths):
            frame, frame_resnames = _frame_ca_map_from_pdb(aligned)
            if frame:
                maps.append(frame)
                resname_maps.append(frame_resnames)
                frame_is_reference.append(idx == 0)

        if len(maps) < 2:
            raise ValueError(f"After filtering, need at least 2 aligned frames, found {len(maps)}")

        key_sets = [set(m.keys()) for m in maps]
        common_keys = set.intersection(*key_sets) if require_common else set.union(*key_sets)
        if len(common_keys) == 0:
            raise ValueError("No common CA positions across aligned frames")
        common_keys = sorted(common_keys)

        t_count = len(maps)
        n_keys = len(common_keys)
        x = np.full((t_count, n_keys, 3), np.nan, dtype=np.float64)
        for t_idx, frame in enumerate(maps):
            for j_idx, key in enumerate(common_keys):
                if key in frame:
                    x[t_idx, j_idx] = frame[key]

        if not require_common:
            keep = np.all(np.isfinite(x).all(axis=2), axis=0)
            x = x[:, keep, :]
            common_keys = [k for k, keep_flag in zip(common_keys, keep) if keep_flag]
            if x.shape[1] == 0:
                raise ValueError("No residues present in all aligned frames after filtering")

        mu = x.mean(axis=0)
        diffs = x - mu[None, :, :]
        rmsf = np.sqrt(np.mean(np.sum(diffs * diffs, axis=2), axis=0))

        if include_resname:
            labeled_keys: List[Tuple[str, int, str | None]] = []
            for chain_id, resid in common_keys:
                chosen_resname: str | None = None

                # Prefer a non-reference frame label so mutant residues are not
                # overwritten by a WT global reference label.
                for res_map, is_ref in zip(resname_maps, frame_is_reference):
                    if not is_ref and (chain_id, resid) in res_map:
                        chosen_resname = res_map[(chain_id, resid)]
                        break

                if chosen_resname is None:
                    for res_map in resname_maps:
                        if (chain_id, resid) in res_map:
                            chosen_resname = res_map[(chain_id, resid)]
                            break

                labeled_keys.append((chain_id, resid, chosen_resname))

            rmsf_dict: Dict[Tuple[Any, ...], float] = {
                key: float(val) for key, val in zip(labeled_keys, rmsf)
            }
        else:
            rmsf_dict = {key: float(val) for key, val in zip(common_keys, rmsf)}

        rmsf_df = (
            pd.DataFrame(
                [
                    {
                        "chain": key[0],
                        "resid": key[1],
                        "resname": key[2] if include_resname else None,
                        "rmsf": float(val),
                    }
                    for key, val in rmsf_dict.items()
                ]
            )
            .sort_values("rmsf", ascending=False)
            .reset_index(drop=True)
        )

        if verbose:
            print(f"[rmsf] Common CA positions used: {len(common_keys)}")
            print(
                f"[rmsf] RMSF mean={rmsf.mean():.2f} A, median={np.median(rmsf):.2f} A, max={rmsf.max():.2f} A"
            )

        if persisted_aligned and save_aligned and verbose:
            print(f"[rmsf] Aligned structures saved in: {aligned_dir}")

        returned_aligned = [Path(p) for p in aligned_paths] if persisted_aligned else []
        return rmsf_dict, rmsf_df, returned_aligned


def rmsf_map_ca_from_tar(
    tar_gz_path: Path | str,
    *,
    include_resname: bool = False,
    require_common: bool = True,
    verbose: bool = True,
    reference_hint: str = "WT",
    aligner: str = "USalign",
    reference_path: Path | str | None = None,
    reference_mode: str = "first",
    save_aligned: bool = False,
    aligned_output_dir: Path | str | None = None,
) -> Tuple[Dict[Tuple[Any, ...], float], pd.DataFrame, List[Path]]:
    """Extract a tar archive and compute CA RMSF after alignment."""
    tar_path = Path(tar_gz_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        if verbose:
            print(f"[rmsf] Extracting: {tar_path}")

        with tarfile.open(tar_path, "r:*") as tar_obj:
            _safe_extract_tar(tar_obj, tmpdir)

        structure_files = drop_top_ranked_duplicate(_discover_structure_files(tmpdir))

        if len(structure_files) < 2:
            raise ValueError(f"Need at least 2 structures in archive, found {len(structure_files)}")

        return rmsf_map_ca_from_structure_files(
            structure_files,
            include_resname=include_resname,
            require_common=require_common,
            verbose=verbose,
            reference_hint=reference_hint,
            aligner=aligner,
            reference_path=reference_path,
            reference_mode=reference_mode,
            save_aligned=save_aligned,
            aligned_output_dir=aligned_output_dir,
        )


def rmsf_map_ca_from_folder(
    folder_path: Path | str,
    *,
    include_resname: bool = False,
    require_common: bool = True,
    verbose: bool = True,
    reference_hint: str = "WT",
    aligner: str = "USalign",
    reference_path: Path | str | None = None,
    reference_mode: str = "first",
    save_aligned: bool = False,
    aligned_output_dir: Path | str | None = None,
) -> Tuple[Dict[Tuple[Any, ...], float], pd.DataFrame, List[Path]]:
    """Compute CA RMSF from a folder tree containing .pdb/.cif files."""
    structure_files = drop_top_ranked_duplicate(_discover_structure_files(folder_path))
    if len(structure_files) < 2:
        raise ValueError(f"Need at least 2 structures in folder, found {len(structure_files)}")

    return rmsf_map_ca_from_structure_files(
        structure_files,
        include_resname=include_resname,
        require_common=require_common,
        verbose=verbose,
        reference_hint=reference_hint,
        aligner=aligner,
        reference_path=reference_path,
        reference_mode=reference_mode,
        save_aligned=save_aligned,
        aligned_output_dir=aligned_output_dir,
    )


def _copy_or_convert_structure_to_pdb(source_path: Path | str, dest_pdb_path: Path | str) -> Path:
    src = Path(source_path)
    dst = Path(dest_pdb_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".pdb":
        shutil.copy2(src, dst)
        return dst

    if src.suffix.lower() == ".cif":
        _convert_cif_to_pdb(src, dst)
        return dst

    raise ValueError(f"Unsupported structure file type for reference: {src}")


def _materialize_global_reference_from_artifact(
    artifact_path: Path | str,
    *,
    source_type: str,
    reference_hint: str,
    output_dir: Path,
) -> Path:
    artifact = Path(artifact_path)
    reference_name = (reference_hint or "reference").strip().lower()
    global_reference_path = output_dir / f"global_{reference_name}_reference.pdb"

    if source_type == "folder":
        structure_files = _discover_structure_files(artifact)
        if not structure_files:
            raise FileNotFoundError(f"No .pdb/.cif files found in folder artifact: {artifact}")

        reference_src = _choose_reference_structure(
            structure_files,
            reference_hint=reference_hint,
        )
        return _copy_or_convert_structure_to_pdb(reference_src, global_reference_path)

    if source_type == "tar":
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(artifact, "r:*") as tar_obj:
                _safe_extract_tar(tar_obj, tmpdir)

            structure_files = _discover_structure_files(tmpdir)
            if not structure_files:
                raise FileNotFoundError(f"No .pdb/.cif files found in tar artifact: {artifact}")

            reference_src = _choose_reference_structure(
                structure_files,
                reference_hint=reference_hint,
            )
            return _copy_or_convert_structure_to_pdb(reference_src, global_reference_path)

    raise ValueError(f"Unsupported source_type for global reference: {source_type}")


RMSF_SUMMARY_COLUMNS = [
    "artifact", "source_path", "status", "n_residues",
    "rmsf_mean", "rmsf_median", "rmsf_max", "rmsf_std", "error",
]

RMSF_RESIDUE_COLUMNS = ["artifact", "chain", "resid", "resname", "rmsf"]


def _compute_rmsf_deltas(
    all_rmsf_df: pd.DataFrame,
    *,
    global_wt_artifact_name: str | None = None,
    reference_hint: str = "WT",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-residue RMSF deltas of every mutant against the WT baseline.

    Extracted from run_rmsf_batch_vs_wt so the sharded path (merge_rmsf_shards) reaches the
    delta numbers through exactly this code. A second copy of the merge/groupby would be a
    silent-divergence risk: it produces plausible numbers whichever way it drifts, and these
    feed the thesis figures.

    Returns (delta_summary_df, delta_rows_df); both empty if no WT baseline is present.
    """
    delta_summary_df = pd.DataFrame()
    delta_rows_df = pd.DataFrame()
    if all_rmsf_df.empty:
        return delta_summary_df, delta_rows_df

    present = set(all_rmsf_df["artifact"].dropna())
    if global_wt_artifact_name is not None and global_wt_artifact_name in present:
        wt_artifact_name = global_wt_artifact_name
    else:
        wt_artifact_name = next(
            (name for name in all_rmsf_df["artifact"].dropna().unique() if reference_hint.upper() in str(name).upper()),
            None,
        )
    if wt_artifact_name is None:
        return delta_summary_df, delta_rows_df

    merge_keys = ["chain", "resid"]
    wt_df = (
        all_rmsf_df[all_rmsf_df["artifact"] == wt_artifact_name][merge_keys + ["resname", "rmsf"]]
        .rename(columns={"resname": "wt_resname", "rmsf": "wt_rmsf"})
        .copy()
    )

    # One row per (chain, resid) is what makes the merge below a lookup rather than a cross
    # product. If WT were ever duplicated -- e.g. two shards both computing it -- every mutant
    # row would fan out and delta_mean would silently average over repeats.
    wt_dupes = int(wt_df.duplicated(merge_keys).sum())
    if wt_dupes:
        raise ValueError(
            f"WT baseline {wt_artifact_name!r} has {wt_dupes} duplicated (chain, resid) rows. "
            "Deltas would fan out across the duplicates. Deduplicate the residue table first."
        )

    mutant_df = all_rmsf_df[all_rmsf_df["artifact"] != wt_artifact_name].copy()
    if "resname" in mutant_df.columns:
        mutant_df = mutant_df.rename(columns={"resname": "artifact_resname"})

    delta_rows_df = mutant_df.merge(wt_df, on=merge_keys, how="inner")

    if not delta_rows_df.empty:
        delta_rows_df["delta_rmsf_vs_wt"] = delta_rows_df["rmsf"] - delta_rows_df["wt_rmsf"]
        delta_rows_df["abs_delta_rmsf_vs_wt"] = delta_rows_df["delta_rmsf_vs_wt"].abs()

        delta_summary_df = (
            delta_rows_df.groupby("artifact", as_index=False)
            .agg(
                n_residues=("delta_rmsf_vs_wt", "size"),
                delta_mean=("delta_rmsf_vs_wt", "mean"),
                delta_median=("delta_rmsf_vs_wt", "median"),
                delta_abs_mean=("abs_delta_rmsf_vs_wt", "mean"),
                delta_abs_max=("abs_delta_rmsf_vs_wt", "max"),
            )
            .sort_values("delta_abs_mean", ascending=False)
            .reset_index(drop=True)
        )

    return delta_summary_df, delta_rows_df


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


def run_rmsf_batch_vs_wt(
    config: StructureMetricsConfig,
    *,
    include_resname: bool = True,
    require_common: bool = True,
    show_progress: bool = True,
    verbose: bool = False,
    reference_hint: str = "WT",
    aligner: str = "USalign",
    global_wt_batch_mode: bool = False,
    max_artifacts: int | None = None,
    output_subdir: str = "rmsf",
    reuse_existing: bool = True,
    allow_recompute_if_missing: bool = False,
    shard: Tuple[int, int] | None = None,
    output_suffix: str = "",
    compute_deltas: bool = True,
    reuse_global_reference: bool = False,
) -> Dict[str, Any]:
    """Run RMSF for all artifacts and optionally compute deltas against WT.

    Returns a dictionary with summary/detail dataframes and written CSV paths.

    The stage is one USalign subprocess per structure and runs sequentially, so a large
    protein costs hours (MutS_alpha: ~5 min per artifact x 152). `shard` splits that across
    processes:

        shard=(i, n)             process only artifacts[i::n]
        output_suffix="shard0of4"  write to *_rmsf_summary.shard0of4.csv etc.
        compute_deltas=False     leave deltas to merge_rmsf_shards, which sees all shards
        reuse_global_reference=True  do not re-extract global_wt_reference.pdb

    That last one matters: the reference is materialised to one fixed path, so concurrent
    shards would race writing it while each other's USalign reads it. Shard 0 additionally
    processes the WT artifact itself, which is the delta baseline -- the other shards skip it
    so it is not computed (or duplicated in the merged table) four times.
    """
    if reuse_existing:
        try:
            loaded = load_rmsf_batch_outputs(
                config,
                output_subdir=output_subdir,
                reference_hint=reference_hint,
            )
            if verbose:
                print("[batch-rmsf] Reusing existing RMSF outputs from CSV files")
                print(f"[batch-rmsf] Summary: {loaded['summary_path']}")
                print(f"[batch-rmsf] Residues: {loaded['all_rows_path']}")
            return loaded
        except FileNotFoundError:
            if not allow_recompute_if_missing:
                raise
            if verbose:
                print("[batch-rmsf] Existing RMSF CSVs not found; recomputing")

    all_artifacts = iter_artifacts(config)
    if not all_artifacts:
        raise FileNotFoundError(
            f"No artifacts found for source_type={config.source_type!r} under {config.read_root_path}"
        )

    wt_candidates_all = _resolve_wt_artifact_path(config, all_artifacts, reference_hint)

    artifacts = list(all_artifacts)
    if max_artifacts is not None:
        artifacts = artifacts[: int(max_artifacts)]

    if shard is not None:
        shard_index, shard_count = int(shard[0]), int(shard[1])
        if shard_count < 1 or not (0 <= shard_index < shard_count):
            raise ValueError(f"shard must be (index, count) with 0 <= index < count; got {shard!r}")
        # Slice the mutants only, then give WT to shard 0. Striding (i::n) rather than
        # contiguous blocks keeps the per-shard cost even: artifacts are name-sorted, so
        # contiguous blocks would put all of one gene's (differently sized) structures in
        # one shard and that shard would finish hours after the others.
        wt_set = set(wt_candidates_all)
        mutants = [p for p in artifacts if p not in wt_set]
        artifacts = mutants[shard_index::shard_count]
        if shard_index == 0 and wt_candidates_all:
            artifacts = [wt_candidates_all[0]] + artifacts
    else:
        # In global-WT mode, keep WT in the selected subset even when max_artifacts is small.
        if global_wt_batch_mode and wt_candidates_all:
            wt_first = wt_candidates_all[0]
            if wt_first not in artifacts:
                artifacts = [wt_first] + artifacts
                if max_artifacts is not None:
                    artifacts = artifacts[: int(max_artifacts)]

        wt_candidates = [p for p in artifacts if reference_hint.upper() in p.name.upper()]
        if wt_candidates:
            wt_first = wt_candidates[0]
            artifacts = [wt_first] + [p for p in artifacts if p != wt_first]

    rmsf_out_dir = config.resolved_output_dir() / output_subdir
    rmsf_out_dir.mkdir(parents=True, exist_ok=True)

    global_reference_path: Path | None = None
    global_wt_artifact_name: str | None = None

    if global_wt_batch_mode:
        if not wt_candidates_all:
            raise ValueError(
                f"global_wt_batch_mode=True but no artifact name contains token {reference_hint!r}"
            )

        wt_artifact = wt_candidates_all[0]
        global_wt_artifact_name = canonical_artifact_stem(wt_artifact)
        existing_reference = rmsf_out_dir / f"global_{(reference_hint or 'reference').strip().lower()}_reference.pdb"
        if reuse_global_reference and existing_reference.exists():
            # Concurrent shards must not each rewrite this file: it is the alignment target
            # every USalign call reads, so a partial rewrite would corrupt another shard's
            # alignments rather than fail loudly.
            global_reference_path = existing_reference
            if verbose:
                print(f"[batch-rmsf] Reusing existing global reference: {global_reference_path}")
        else:
            global_reference_path = _materialize_global_reference_from_artifact(
                wt_artifact,
                source_type=config.source_type,
                reference_hint=reference_hint,
                output_dir=rmsf_out_dir,
            )

        if verbose:
            print(f"[batch-rmsf] Global WT reference mode: ON")
            print(f"[batch-rmsf] Global reference path: {global_reference_path}")
            print(f"[batch-rmsf] WT artifact used: {global_wt_artifact_name}")
    elif verbose:
        print("[batch-rmsf] Global WT reference mode: OFF")

    summary_rows: List[Dict[str, Any]] = []
    rmsf_tables: List[pd.DataFrame] = []

    desc = f"RMSF {config.mut_name}" + (f" shard {shard[0]}/{shard[1]}" if shard is not None else "")
    iterator = tqdm(artifacts, desc=desc, unit="artifact") if show_progress else artifacts

    for artifact in iterator:
        artifact_name = canonical_artifact_stem(artifact)

        try:
            if config.source_type == "tar":
                _, rmsf_df, _ = rmsf_map_ca_from_tar(
                    artifact,
                    include_resname=include_resname,
                    require_common=require_common,
                    verbose=verbose,
                    reference_hint=reference_hint,
                    aligner=aligner,
                    reference_path=global_reference_path,
                )
            elif config.source_type == "folder":
                _, rmsf_df, _ = rmsf_map_ca_from_folder(
                    artifact,
                    include_resname=include_resname,
                    require_common=require_common,
                    verbose=verbose,
                    reference_hint=reference_hint,
                    aligner=aligner,
                    reference_path=global_reference_path,
                )
            else:
                raise ValueError(f"Unsupported config.source_type: {config.source_type}")

            if rmsf_df.empty:
                summary_rows.append(
                    {
                        "artifact": artifact_name,
                        "source_path": str(artifact),
                        "status": "empty",
                        "n_residues": 0,
                        "rmsf_mean": np.nan,
                        "rmsf_median": np.nan,
                        "rmsf_max": np.nan,
                        "rmsf_std": np.nan,
                        "error": "",
                    }
                )
                continue

            rmsf_df = rmsf_df.copy()
            rmsf_df.insert(0, "artifact", artifact_name)
            rmsf_tables.append(rmsf_df)

            summary_rows.append(
                {
                    "artifact": artifact_name,
                    "source_path": str(artifact),
                    "status": "ok",
                    "n_residues": int(len(rmsf_df)),
                    "rmsf_mean": float(rmsf_df["rmsf"].mean()),
                    "rmsf_median": float(rmsf_df["rmsf"].median()),
                    "rmsf_max": float(rmsf_df["rmsf"].max()),
                    "rmsf_std": float(rmsf_df["rmsf"].std(ddof=0)),
                    "error": "",
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "artifact": artifact_name,
                    "source_path": str(artifact),
                    "status": "error",
                    "n_residues": 0,
                    "rmsf_mean": np.nan,
                    "rmsf_median": np.nan,
                    "rmsf_max": np.nan,
                    "rmsf_std": np.nan,
                    "error": str(exc)[:500],
                }
            )

    # Fixed schema even when empty: a shard can legitimately get no artifacts (more shards
    # than artifacts), and a column-less CSV would break the merge's checks downstream with a
    # KeyError rather than being treated as the empty slice it is.
    summary_df = pd.DataFrame(summary_rows, columns=RMSF_SUMMARY_COLUMNS)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["status", "rmsf_mean"], ascending=[True, False]).reset_index(drop=True)

    if rmsf_tables:
        all_rmsf_df = pd.concat(rmsf_tables, ignore_index=True)
    else:
        all_rmsf_df = pd.DataFrame(columns=RMSF_RESIDUE_COLUMNS)

    suffix = f".{output_suffix}" if output_suffix else ""
    summary_path = rmsf_out_dir / f"{config.mut_name}_rmsf_summary{suffix}.csv"
    all_rows_path = rmsf_out_dir / f"{config.mut_name}_rmsf_all_residues{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    all_rmsf_df.to_csv(all_rows_path, index=False)

    if compute_deltas:
        delta_summary_df, delta_rows_df = _compute_rmsf_deltas(
            all_rmsf_df,
            global_wt_artifact_name=global_wt_artifact_name if global_wt_batch_mode else None,
            reference_hint=reference_hint,
        )
    else:
        # A single shard holds only its slice of artifacts, so a delta table computed here
        # would be against whichever WT rows that slice happens to contain. merge_rmsf_shards
        # computes deltas once, over the full concatenated table.
        delta_summary_df = pd.DataFrame()
        delta_rows_df = pd.DataFrame()

    delta_summary_path = rmsf_out_dir / f"{config.mut_name}_rmsf_delta_vs_wt_summary{suffix}.csv"
    delta_rows_path = rmsf_out_dir / f"{config.mut_name}_rmsf_delta_vs_wt_residues{suffix}.csv"
    if compute_deltas:
        delta_summary_df.to_csv(delta_summary_path, index=False)
        delta_rows_df.to_csv(delta_rows_path, index=False)

    return {
        "summary_df": summary_df,
        "all_rmsf_df": all_rmsf_df,
        "delta_summary_df": delta_summary_df,
        "delta_rows_df": delta_rows_df,
        "summary_path": str(summary_path),
        "all_rows_path": str(all_rows_path),
        "delta_summary_path": str(delta_summary_path),
        "delta_rows_path": str(delta_rows_path),
        "global_reference_path": str(global_reference_path) if global_reference_path else None,
        "global_wt_artifact_name": global_wt_artifact_name,
    }


def merge_rmsf_shards(
    config: StructureMetricsConfig,
    *,
    n_shards: int,
    output_subdir: str = "rmsf",
    reference_hint: str = "WT",
    expected_artifacts: int | None = None,
    keep_shard_files: bool = True,
) -> Dict[str, Any]:
    """Concatenate sharded RMSF outputs into the canonical four CSVs.

    Writes exactly the files load_rmsf_batch_outputs() reads, so after this the ordinary
    `--stage rmsf` run (without --force) picks the results up through its reuse path and
    does the plots and interface RMSF with no sharding-specific code.

    Every shard must be present. A missing shard would silently produce a complete-looking
    table covering three quarters of the variants -- the same failure mode as the mismatched
    coord-chunk split guarded in ensemble_aware_pipeline, and the reason `rmsf` was stale for
    MutS_alpha's MSH6 half for three months without anything reporting a problem.
    """
    rmsf_out_dir = config.resolved_output_dir() / output_subdir
    summaries, residues = [], []
    missing = []
    for i in range(int(n_shards)):
        suffix = f"shard{i}of{n_shards}"
        s_path = rmsf_out_dir / f"{config.mut_name}_rmsf_summary.{suffix}.csv"
        r_path = rmsf_out_dir / f"{config.mut_name}_rmsf_all_residues.{suffix}.csv"
        if not s_path.exists() or not r_path.exists():
            missing.append(suffix)
            continue
        summaries.append(pd.read_csv(s_path))
        residues.append(pd.read_csv(r_path))
    if missing:
        raise FileNotFoundError(
            f"[{config.mut_name}] missing RMSF shard output(s): {', '.join(missing)}. "
            f"Merging {len(summaries)} of {n_shards} shards would write a table that looks "
            "complete but covers only part of the variants. Rerun the missing shard(s)."
        )

    # A shard can legitimately be empty (more shards than artifacts). Its header-only CSV
    # still proves it ran -- the presence check above is what guards against a lost shard --
    # but it is dropped here so it contributes no all-NA column to the concat.
    summary_df = pd.concat([d for d in summaries if not d.empty], ignore_index=True) \
        if any(not d.empty for d in summaries) else pd.DataFrame(columns=RMSF_SUMMARY_COLUMNS)
    all_rmsf_df = pd.concat([d for d in residues if not d.empty], ignore_index=True) \
        if any(not d.empty for d in residues) else pd.DataFrame(columns=RMSF_RESIDUE_COLUMNS)

    dupes = int(summary_df.duplicated("artifact").sum())
    if dupes:
        # Shards stride disjointly and only shard 0 takes WT, so this should be impossible;
        # if it happens the shard slicing is wrong and the deltas would fan out.
        raise ValueError(
            f"[{config.mut_name}] {dupes} artifact(s) appear in more than one shard. "
            "Shard slices must be disjoint; check the shard/n_shards arguments used."
        )

    summary_df = summary_df.sort_values(["status", "rmsf_mean"], ascending=[True, False]).reset_index(drop=True)

    if expected_artifacts is not None and len(summary_df) != int(expected_artifacts):
        raise ValueError(
            f"[{config.mut_name}] merged {len(summary_df)} artifacts but {expected_artifacts} "
            "were discovered on disk. Some shard processed fewer artifacts than it was given."
        )

    global_wt_artifact_name = next(
        (str(a) for a in summary_df["artifact"].dropna() if reference_hint.upper() in str(a).upper()), None
    )
    delta_summary_df, delta_rows_df = _compute_rmsf_deltas(
        all_rmsf_df,
        global_wt_artifact_name=global_wt_artifact_name,
        reference_hint=reference_hint,
    )

    paths = {
        "summary_path": rmsf_out_dir / f"{config.mut_name}_rmsf_summary.csv",
        "all_rows_path": rmsf_out_dir / f"{config.mut_name}_rmsf_all_residues.csv",
        "delta_summary_path": rmsf_out_dir / f"{config.mut_name}_rmsf_delta_vs_wt_summary.csv",
        "delta_rows_path": rmsf_out_dir / f"{config.mut_name}_rmsf_delta_vs_wt_residues.csv",
    }
    summary_df.to_csv(paths["summary_path"], index=False)
    all_rmsf_df.to_csv(paths["all_rows_path"], index=False)
    delta_summary_df.to_csv(paths["delta_summary_path"], index=False)
    delta_rows_df.to_csv(paths["delta_rows_path"], index=False)

    if not keep_shard_files:
        for i in range(int(n_shards)):
            for pattern in ("summary", "all_residues"):
                (rmsf_out_dir / f"{config.mut_name}_rmsf_{pattern}.shard{i}of{n_shards}.csv").unlink(missing_ok=True)

    global_reference_path = rmsf_out_dir / f"global_{reference_hint.strip().lower()}_reference.pdb"
    return {
        "summary_df": summary_df,
        "all_rmsf_df": all_rmsf_df,
        "delta_summary_df": delta_summary_df,
        "delta_rows_df": delta_rows_df,
        **{k: str(v) for k, v in paths.items()},
        "global_reference_path": str(global_reference_path) if global_reference_path.exists() else None,
        "global_wt_artifact_name": global_wt_artifact_name,
    }


def load_rmsf_batch_outputs(
    config: StructureMetricsConfig,
    *,
    output_subdir: str = "rmsf",
    reference_hint: str = "WT",
) -> Dict[str, Any]:
    """Load previously written RMSF batch outputs from CSV files.

    This is intended for notebook workflows where RMSF extraction is expensive
    and users want to reuse existing results.
    """
    rmsf_out_dir = config.resolved_output_dir() / output_subdir
    summary_path = rmsf_out_dir / f"{config.mut_name}_rmsf_summary.csv"
    all_rows_path = rmsf_out_dir / f"{config.mut_name}_rmsf_all_residues.csv"
    delta_summary_path = rmsf_out_dir / f"{config.mut_name}_rmsf_delta_vs_wt_summary.csv"
    delta_rows_path = rmsf_out_dir / f"{config.mut_name}_rmsf_delta_vs_wt_residues.csv"
    global_reference_path = rmsf_out_dir / f"global_{reference_hint.strip().lower()}_reference.pdb"

    required_paths = [summary_path, all_rows_path]
    missing_required = [str(p) for p in required_paths if not p.exists()]
    if missing_required:
        raise FileNotFoundError(
            "Missing required RMSF output files: " + ", ".join(missing_required)
        )

    summary_df = pd.read_csv(summary_path)
    all_rmsf_df = pd.read_csv(all_rows_path)

    if delta_summary_path.exists():
        delta_summary_df = pd.read_csv(delta_summary_path)
    else:
        delta_summary_df = pd.DataFrame()

    if delta_rows_path.exists():
        delta_rows_df = pd.read_csv(delta_rows_path)
    else:
        delta_rows_df = pd.DataFrame()

    global_ref_out = str(global_reference_path) if global_reference_path.exists() else None

    global_wt_artifact_name: str | None = None
    if "artifact" in summary_df.columns:
        wt_rows = summary_df[
            summary_df["artifact"].astype(str).str.upper().str.contains(reference_hint.upper(), na=False)
        ]
        if not wt_rows.empty:
            global_wt_artifact_name = str(wt_rows.iloc[0]["artifact"])

    return {
        "summary_df": summary_df,
        "all_rmsf_df": all_rmsf_df,
        "delta_summary_df": delta_summary_df,
        "delta_rows_df": delta_rows_df,
        "summary_path": str(summary_path),
        "all_rows_path": str(all_rows_path),
        "delta_summary_path": str(delta_summary_path),
        "delta_rows_path": str(delta_rows_path),
        "global_reference_path": global_ref_out,
        "global_wt_artifact_name": global_wt_artifact_name,
    }


def run_or_load_rmsf_batch_vs_wt(
    config: StructureMetricsConfig,
    *,
    include_resname: bool = True,
    require_common: bool = True,
    show_progress: bool = True,
    verbose: bool = False,
    reference_hint: str = "WT",
    aligner: str = "USalign",
    global_wt_batch_mode: bool = False,
    max_artifacts: int | None = None,
    output_subdir: str = "rmsf",
    reuse_existing: bool = True,
    allow_recompute_if_missing: bool = False,
) -> Dict[str, Any]:
    """Load existing RMSF outputs when available, otherwise optionally compute.

    Set allow_recompute_if_missing=False to hard-fail instead of triggering a
    costly recomputation.
    """
    if reuse_existing:
        try:
            loaded = load_rmsf_batch_outputs(
                config,
                output_subdir=output_subdir,
                reference_hint=reference_hint,
            )
            if verbose:
                print("[batch-rmsf] Reusing existing RMSF outputs from CSV files")
                print(f"[batch-rmsf] Summary: {loaded['summary_path']}")
                print(f"[batch-rmsf] Residues: {loaded['all_rows_path']}")
            return loaded
        except FileNotFoundError:
            if not allow_recompute_if_missing:
                raise
            if verbose:
                print("[batch-rmsf] Existing RMSF CSVs not found; recomputing")

    return run_rmsf_batch_vs_wt(
        config,
        include_resname=include_resname,
        require_common=require_common,
        show_progress=show_progress,
        verbose=verbose,
        reference_hint=reference_hint,
        aligner=aligner,
        global_wt_batch_mode=global_wt_batch_mode,
        max_artifacts=max_artifacts,
        output_subdir=output_subdir,
        reuse_existing=False,
        allow_recompute_if_missing=allow_recompute_if_missing,
    )


def plot_rmsf_summary_panels(
    summary_df: pd.DataFrame,
    *,
    delta_summary_df: Optional[pd.DataFrame] = None,
    top_n: int = 20,
    title: str = "RMSF Batch Summary",
    out_path: Path | str | None = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot a compact RMSF overview for artifact-level QC and ranking."""
    if summary_df.empty:
        raise ValueError("summary_df is empty")

    required_cols = {"artifact", "status", "n_residues", "rmsf_mean", "rmsf_std"}
    missing_cols = required_cols.difference(summary_df.columns)
    if missing_cols:
        raise KeyError(f"summary_df is missing required columns: {sorted(missing_cols)}")

    base = summary_df.copy()
    ok_df = base[base["status"] == "ok"].copy()
    if ok_df.empty:
        raise ValueError("No successful RMSF artifacts found (status == 'ok')")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Distribution of mean RMSF across successful artifacts.
    sns.histplot(ok_df["rmsf_mean"], bins=25, kde=True, color="#2C7FB8", ax=ax1)
    ax1.set_title("Distribution of mean RMSF")
    ax1.set_xlabel("Mean RMSF (A)")
    ax1.set_ylabel("Artifact count")
    ax1.grid(alpha=0.25)

    # Relationship between ensemble overlap (n residues) and mean RMSF.
    status_palette = {"ok": "#2CA25F", "empty": "#E67E22", "error": "#D7301F"}
    sns.scatterplot(
        data=base,
        x="n_residues",
        y="rmsf_mean",
        hue="status",
        palette=status_palette,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
        ax=ax2,
    )
    ax2.set_title("Mean RMSF vs residues used")
    ax2.set_xlabel("Common residues")
    ax2.set_ylabel("Mean RMSF (A)")
    ax2.grid(alpha=0.25)
    ax2.legend(title="status", loc="best")

    # Top artifacts by delta-vs-WT if available, otherwise top by mean RMSF.
    if delta_summary_df is not None and not delta_summary_df.empty and {
        "artifact",
        "delta_abs_mean",
    }.issubset(delta_summary_df.columns):
        top_df = (
            delta_summary_df.sort_values("delta_abs_mean", ascending=False)
            .head(max(int(top_n), 1))
            .sort_values("delta_abs_mean", ascending=True)
        )
        ax3.barh(top_df["artifact"], top_df["delta_abs_mean"], color="#F18F01", edgecolor="black", linewidth=0.5)
        ax3.set_title("Top artifacts by |delta RMSF| vs WT")
        ax3.set_xlabel("Mean |delta RMSF| (A)")
    else:
        top_df = (
            ok_df.sort_values("rmsf_mean", ascending=False)
            .head(max(int(top_n), 1))
            .sort_values("rmsf_mean", ascending=True)
        )
        ax3.barh(top_df["artifact"], top_df["rmsf_mean"], color="#F18F01", edgecolor="black", linewidth=0.5)
        ax3.set_title("Top artifacts by mean RMSF")
        ax3.set_xlabel("Mean RMSF (A)")
    ax3.set_ylabel("artifact")
    ax3.grid(axis="x", alpha=0.25)

    # Compact view of RMSF spread per artifact status.
    box_df = base[base["rmsf_mean"].notna()].copy()
    sns.boxplot(data=box_df, x="status", y="rmsf_mean", ax=ax4, color="#C7E9B4")
    sns.stripplot(data=box_df, x="status", y="rmsf_mean", ax=ax4, color="black", alpha=0.45, size=2.5)
    ax4.set_title("Mean RMSF by status")
    ax4.set_xlabel("status")
    ax4.set_ylabel("Mean RMSF (A)")
    ax4.grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, tuple(axes.ravel())


def plot_rmsf_top_delta_residues(
    delta_rows_df: pd.DataFrame,
    *,
    top_n: int = 25,
    metric_col: str = "abs_delta_rmsf_vs_wt",
    title: str = "Top residue-level RMSF shifts vs WT",
    out_path: Path | str | None = None,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Plot top residue-level RMSF shifts and return the ranked table used."""
    if delta_rows_df.empty:
        raise ValueError("delta_rows_df is empty")

    required_cols = {"artifact", "chain", "resid", "delta_rmsf_vs_wt"}
    missing_cols = required_cols.difference(delta_rows_df.columns)
    if missing_cols:
        raise KeyError(f"delta_rows_df is missing required columns: {sorted(missing_cols)}")

    rank_df = delta_rows_df.copy()
    if metric_col not in rank_df.columns:
        if metric_col == "abs_delta_rmsf_vs_wt":
            rank_df[metric_col] = rank_df["delta_rmsf_vs_wt"].abs()
        else:
            raise KeyError(f"metric column '{metric_col}' not found in delta_rows_df")

    rank_df = rank_df.sort_values(metric_col, ascending=False).head(max(int(top_n), 1)).copy()

    def _format_residue_label(row: pd.Series) -> str:
        resid = int(row["resid"]) if pd.notna(row["resid"]) else row["resid"]
        resname = row.get("resname")
        if pd.notna(resname):
            return f"{row['artifact']} | {row['chain']}:{resid} {resname}"
        return f"{row['artifact']} | {row['chain']}:{resid}"

    rank_df["residue_label"] = rank_df.apply(_format_residue_label, axis=1)

    plot_df = rank_df.iloc[::-1].copy()
    colors = np.where(plot_df["delta_rmsf_vs_wt"] >= 0, "#D95F0E", "#3182BD")

    fig, ax = plt.subplots(figsize=(14, max(6, 0.32 * len(plot_df))))
    ax.barh(
        plot_df["residue_label"],
        plot_df[metric_col],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title(title)
    ax.set_xlabel(f"{metric_col} (A)")
    ax.set_ylabel("artifact | residue")
    ax.grid(axis="x", alpha=0.25)

    # Add a small legend for delta direction.
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#D95F0E", markersize=8, label="Increase vs WT"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#3182BD", markersize=8, label="Decrease vs WT"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return rank_df, fig, ax


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


def resolve_plddt_dir(*candidates: Path | str) -> Path:
    for candidate in candidates:
        p = Path(candidate)
        if p.is_dir():
            return p
    raise FileNotFoundError("None of the candidate pLDDT directories exist")


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
    (they have different sample counts), which manufactures a spurious mut/WT ratio and
    inflates/deflates D1. The distortion scales with residue count, so it is negligible
    for ~200-residue proteins but severe for MutS_alpha (2294) / TP53 (1572). We drop the
    leading column whenever it is exactly ``0..N-1`` -- matching load_plddt_profile and
    the index_col=0 read the ensemble-aware D1 sweep already uses.
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


def flatten_plddt(plddt_matrix: np.ndarray) -> np.ndarray:
    return plddt_matrix[np.isfinite(plddt_matrix)]


def find_wild_type_file(plddt_files: Sequence[Path | str], wt_tokens: Sequence[str] = ("WT",)) -> Optional[Path]:
    for file_path in plddt_files:
        name_upper = Path(file_path).name.upper()
        if any(token.upper() in name_upper for token in wt_tokens):
            return Path(file_path)
    return None


def summarize_wild_type(wt_file: Path | str | None) -> dict:
    if wt_file is None:
        return {
            "wt_file": None,
            "wt_matrix": None,
            "wt_values": None,
            "wt_mean": None,
            "wt_std": None,
        }

    wt_matrix = load_plddt_matrix(wt_file)
    wt_values = flatten_plddt(wt_matrix)

    return {
        "wt_file": str(wt_file),
        "wt_matrix": wt_matrix,
        "wt_values": wt_values,
        "wt_mean": float(np.nanmean(wt_values)) if wt_values.size else None,
        "wt_std": float(np.nanstd(wt_values)) if wt_values.size else None,
    }


def load_plddt_profile(filepath: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-residue mean/std pLDDT across the ensemble for one ``*_plddt_values.csv``.

    Column 0 is a 0-based residue array index (its header is ``Unnamed: 0`` or
    ``Residue`` depending on the protein); it is dropped when it is exactly
    ``0..N-1`` so it is never mistaken for an ensemble member. Every remaining
    column is one AlphaFold sample. Returns ``(residue_index, mean, std)`` where
    ``residue_index`` is the 0-based array index and ``mean``/``std`` are taken
    across the ensemble (band = +/-1 SD across samples).
    """
    df = pd.read_csv(filepath)
    idx = np.arange(len(df))
    first = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if first.notna().all() and np.array_equal(first.to_numpy(), idx.astype(float)):
        df = df.iloc[:, 1:]
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values = _apply_row_block(values, filepath)
    # idx is a 0-based array index, so after slicing it must describe the
    # BLOCK's own rows -- not the parent complex's -- or every per-residue
    # plot would be offset by the block start.
    idx = np.arange(len(values))
    return idx, np.nanmean(values, axis=1), np.nanstd(values, axis=1)


def plot_plddt_exemplar_profiles(
    wt_file: Path | str,
    driver_file: Path | str | None,
    passenger_file: Path | str | None,
    *,
    driver_label: str = "Driver",
    passenger_label: str = "Passenger",
    residue_start: Optional[int] = None,
    title: Optional[str] = None,
    out_path: Path | str | None = None,
) -> dict:
    """Overlay mean +/-1 SD per-residue pLDDT for WT vs one driver vs one passenger.

    A quick, legible per-residue confidence view: the WT baseline against a single
    representative driver and a single representative passenger, each averaged over
    its AlphaFold ensemble with a +/-1 SD band. The x-axis is the 0-based residue
    array index; when ``residue_start`` is given (single-chain proteins) it becomes
    the real residue number and each variant's mutation site is marked with a dotted
    line. Returns a small summary dict (per-series file, residue count, mean pLDDT).
    """
    specs = [
        ("WT", wt_file, "#444444"),
        (driver_label, driver_file, "#d1495b"),
        (passenger_label, passenger_file, "#2e86ab"),
    ]
    fig, ax = plt.subplots(figsize=(13, 5))
    x_offset = int(residue_start) if residue_start is not None else 0
    result: dict = {}
    for name, path, color in specs:
        if path is None:
            continue
        idx, mean, std = load_plddt_profile(path)
        x = idx + x_offset
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, lw=1.7, label=f"{name} (mean +/-1 SD)")
        result[name] = {
            "file": str(path),
            "n_residues": int(len(idx)),
            "plddt_mean": float(np.nanmean(mean)),
        }
        # Mark the mutation site only when we can place it on a real residue axis.
        if residue_start is not None and name != "WT":
            code = extract_mutation_code(Path(path).name)
            digits = re.search(r"\d+", code) if code else None
            if digits:
                ax.axvline(int(digits.group()), color=color, ls=":", lw=1)
    ax.set_xlabel("residue number" if residue_start is not None else "residue array index")
    ax.set_ylabel("mean pLDDT")
    ax.set_title(title or "Per-residue pLDDT: WT vs driver vs passenger")
    ax.legend(loc="lower right", fontsize=9)
    # Fit the y-axis to the data -- pLDDT sits high and varies little, so a fixed
    # 40-100 window wastes most of the panel and flattens the driver dips. Keep the
    # autoscaled (data-fit) bottom and only cap the top at the pLDDT ceiling of 100.
    ax.set_ylim(top=min(ax.get_ylim()[1], 100))
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    result["out_path"] = str(out_path) if out_path else None
    return result


def plot_plddt_exemplar_distributions_by_residue(
    wt_file: Path | str,
    driver_file: Path | str,
    passenger_file: Path | str,
    *,
    driver_label: str = "Driver",
    passenger_label: str = "Passenger",
    residue_start: Optional[int] = None,
    residue_segments: Optional[Sequence[Mapping[str, Any]]] = None,
    driver_site_chains: Optional[Sequence[str]] = None,
    passenger_site_chains: Optional[Sequence[str]] = None,
    panels_per_page: int = 10,
    ncols: int = 2,
    kde_bandwidth: float = 0.35,
    title: Optional[str] = None,
    out_base: Path | str,
) -> dict:
    """Facet smooth WT, exemplar-driver, and exemplar-passenger pLDDT densities.

    Every residue gets one panel containing three ensemble distributions. Complete
    residue rows are retained, so each Gaussian kernel-density estimate represents
    AlphaFold samples rather than mutation-level means. A fixed bandwidth is shared
    across all residues and series. Both axes scale independently by residue (while
    remaining shared by WT/driver/passenger inside that panel) so narrow and broad
    distributions remain visible. ``residue_segments`` maps concatenated multi-chain
    matrices back to biological chain/residue identifiers; site-chain selectors then
    mark every physical copy of the exemplar mutation. Pages are written as
    gallery-friendly PNGs and as one multi-page PDF.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    panels_per_page = int(panels_per_page)
    ncols = int(ncols)
    if panels_per_page < 1:
        raise ValueError("panels_per_page must be at least 1")
    if ncols < 1 or ncols > panels_per_page:
        raise ValueError("ncols must be between 1 and panels_per_page")
    kde_bandwidth = float(kde_bandwidth)
    if not np.isfinite(kde_bandwidth) or kde_bandwidth <= 0:
        raise ValueError("kde_bandwidth must be a finite positive number")

    series = [
        ("WT", load_plddt_matrix(wt_file), "#555555"),
        (driver_label, load_plddt_matrix(driver_file), "#d1495b"),
        (passenger_label, load_plddt_matrix(passenger_file), "#2e86ab"),
    ]
    n_residues = min(matrix.shape[0] for _, matrix, _ in series)
    if n_residues < 1:
        raise ValueError("exemplar distribution panels need at least one residue")

    finite_values = [
        matrix[:n_residues][np.isfinite(matrix[:n_residues])]
        for _, matrix, _ in series
    ]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        raise ValueError("exemplar distribution panels contain no finite pLDDT values")
    pooled = np.concatenate(finite_values)
    data_min, data_max = float(np.min(pooled)), float(np.max(pooled))
    kde_normalizer = kde_bandwidth * np.sqrt(2.0 * np.pi)

    segment_specs = list(residue_segments or [])
    residue_metadata: list[dict[str, Any]] = []
    if segment_specs:
        for segment in segment_specs:
            chain_id = str(segment["chain_id"])
            display_name = str(segment.get("display_name", f"Chain {chain_id}"))
            segment_start = int(segment.get("residue_start", 1))
            segment_length = int(segment["length"])
            if segment_length < 1:
                raise ValueError(f"residue segment {chain_id!r} must have positive length")
            for local_index in range(segment_length):
                residue_metadata.append({
                    "chain_id": chain_id,
                    "display_name": display_name,
                    "residue_number": segment_start + local_index,
                })
        if len(residue_metadata) != n_residues:
            raise ValueError(
                "residue_segments describe "
                f"{len(residue_metadata)} residues, but the exemplar matrices have "
                f"{n_residues}"
            )
    else:
        x_offset = int(residue_start) if residue_start is not None else 0
        residue_metadata = [
            {
                "chain_id": None,
                "display_name": None,
                "residue_number": residue_index + x_offset,
            }
            for residue_index in range(n_residues)
        ]

    def estimate_density(values, grid):
        scaled = (grid[:, None] - values[None, :]) / kde_bandwidth
        return np.exp(-0.5 * scaled * scaled).mean(axis=1) / kde_normalizer

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    # Page boundaries are encoded in filenames. Remove pages from an earlier layout
    # before writing the new set so the gallery cannot mix stale and current ranges.
    for stale_page in out_base.parent.glob(f"{out_base.name}_residues_*.png"):
        stale_page.unlink()
    pdf_path = out_base.with_suffix(".pdf")
    page_paths: list[str] = []
    n_pages = int(np.ceil(n_residues / panels_per_page))
    mutation_sites: dict[int, list[str]] = {}
    site_markers: list[dict[str, Any]] = []
    has_biological_mapping = residue_start is not None or bool(segment_specs)
    if has_biological_mapping:
        site_specs = (
            ("D", driver_file, driver_site_chains),
            ("P", passenger_file, passenger_site_chains),
        )
        configured_chains = {metadata["chain_id"] for metadata in residue_metadata}
        for site_label, path, selected_chains in site_specs:
            code = extract_mutation_code(Path(path).name)
            match = re.search(r"\d+", code) if code else None
            if not match:
                raise ValueError(f"could not extract the {site_label}-site from {path}")
            mutation_residue = int(match.group())
            wanted_chains = (
                {str(chain) for chain in selected_chains}
                if selected_chains is not None else configured_chains
            )
            unknown_chains = wanted_chains.difference(configured_chains)
            if segment_specs and unknown_chains:
                raise ValueError(
                    f"{site_label}-site chains {sorted(unknown_chains)} are absent from "
                    "residue_segments"
                )
            matched_rows = []
            for row_index, metadata in enumerate(residue_metadata):
                chain_matches = (
                    not segment_specs or metadata["chain_id"] in wanted_chains
                )
                if chain_matches and metadata["residue_number"] == mutation_residue:
                    mutation_sites.setdefault(row_index, []).append(site_label)
                    matched_rows.append(row_index)
                    site_markers.append({
                        "site": site_label,
                        "row_index": row_index,
                        "chain_id": metadata["chain_id"],
                        "display_name": metadata["display_name"],
                        "residue_number": mutation_residue,
                    })
            if not matched_rows:
                raise ValueError(
                    f"{site_label}-site residue {mutation_residue} does not occur in "
                    "the configured pLDDT residue mapping"
                )

    with PdfPages(pdf_path) as pdf:
        for page_index, first in enumerate(range(0, n_residues, panels_per_page), start=1):
            last = min(first + panels_per_page, n_residues)
            page_count = last - first
            nrows = int(np.ceil(page_count / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5.6 * ncols, 2.8 * nrows + 0.9),
                sharex=False, sharey=False, squeeze=False,
            )
            for local_index, residue_index in enumerate(range(first, last)):
                row, col = divmod(local_index, ncols)
                ax = axes[row, col]
                residue_metadata_row = residue_metadata[residue_index]
                residue_number = int(residue_metadata_row["residue_number"])
                panel_series = []
                for _, matrix, color in series:
                    values = matrix[residue_index]
                    values = values[np.isfinite(values)]
                    if values.size:
                        panel_series.append((values, color))
                panel_values = np.concatenate(
                    [values for values, _ in panel_series]
                )
                panel_data_min = float(np.min(panel_values))
                panel_data_max = float(np.max(panel_values))
                panel_span = max(panel_data_max - panel_data_min, 1.0)
                panel_padding = max(3.0 * kde_bandwidth, 0.08 * panel_span)
                panel_x_min = max(0.0, panel_data_min - panel_padding)
                panel_x_max = min(100.0, panel_data_max + panel_padding)
                if panel_x_max - panel_x_min < 6.0:
                    midpoint = (panel_x_min + panel_x_max) / 2.0
                    panel_x_min = max(0.0, midpoint - 3.0)
                    panel_x_max = min(100.0, midpoint + 3.0)
                density_grid = np.linspace(panel_x_min, panel_x_max, 300)
                curves = [
                    (estimate_density(values, density_grid), color)
                    for values, color in panel_series
                ]
                panel_maximum = max(
                    (float(np.max(density)) for density, _ in curves), default=1.0,
                )
                for density, color in curves:
                    ax.fill_between(
                        density_grid, 0.0, density, color=color, alpha=0.13,
                        linewidth=0,
                    )
                    ax.plot(density_grid, density, color=color, linewidth=1.45)
                mutation_markers = mutation_sites.get(residue_index, [])
                display_name = residue_metadata_row["display_name"]
                panel_title = (
                    f"{display_name} residue {residue_number}"
                    if display_name else f"Residue {residue_number}"
                )
                title_kwargs: dict[str, Any] = {}
                if mutation_markers:
                    unique_markers = list(dict.fromkeys(mutation_markers))
                    site_text = (
                        f"Site {unique_markers[0]}"
                        if len(unique_markers) == 1
                        else f"Sites {' + '.join(unique_markers)}"
                    )
                    panel_title += f"  —  {site_text}"
                    title_kwargs = {
                        "fontweight": "bold",
                        "color": (
                            "#d1495b" if unique_markers == ["D"]
                            else "#2e86ab" if unique_markers == ["P"]
                            else "#7b2cbf"
                        ),
                    }
                ax.set_title(panel_title, fontsize=9.5, **title_kwargs)
                ax.set_xlim(panel_x_min, panel_x_max)
                ax.set_ylim(0.0, panel_maximum * 1.08)
                ax.tick_params(axis="both", labelsize=8, labelleft=(col == 0))
                ax.grid(axis="x", alpha=0.18, linewidth=0.6)
                if col == 0:
                    ax.set_ylabel("KDE density", fontsize=8.5)
                if row == nrows - 1:
                    ax.set_xlabel("pLDDT", fontsize=8.5)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)

            for unused in range(page_count, nrows * ncols):
                row, col = divmod(unused, ncols)
                axes[row, col].set_visible(False)

            legend = [
                Patch(
                    facecolor=color, edgecolor=color, alpha=0.15,
                    label=f"{name} (n={matrix.shape[1]})",
                )
                for name, matrix, color in series
            ]
            fig.legend(
                handles=legend, loc="upper center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, 0.965), fontsize=9,
            )
            fig.suptitle(
                f"{title or 'Per-residue pLDDT ensemble distributions'} "
                f"— page {page_index}/{n_pages}\n"
                "Axes scale within each residue; compare WT/driver/passenger inside a panel",
                y=0.998, fontsize=13,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
            pdf.savefig(fig, bbox_inches="tight")
            page_first = first + (int(residue_start) if residue_start is not None else 0)
            page_last = last - 1 + (int(residue_start) if residue_start is not None else 0)
            page_path = out_base.parent / (
                f"{out_base.name}_residues_{page_first}-{page_last}.png"
            )
            fig.savefig(page_path, dpi=240, bbox_inches="tight")
            page_paths.append(str(page_path))
            plt.close(fig)

    return {
        "wt_file": str(wt_file),
        "driver_file": str(driver_file),
        "passenger_file": str(passenger_file),
        "n_residues": int(n_residues),
        "sample_counts": {
            name: int(matrix.shape[1]) for name, matrix, _ in series
        },
        "panels_per_page": panels_per_page,
        "kde_bandwidth": kde_bandwidth,
        "n_pages": n_pages,
        "global_plddt_range": [data_min, data_max],
        "panel_specific_axes": True,
        "site_markers": site_markers,
        "pdf_path": str(pdf_path),
        "page_paths": page_paths,
    }


def _class_ratio_profiles(files, wt_mean):
    """Stack per-variant per-residue pLDDT ratios (wt_mean / mut_mean) into one array.

    Each file contributes one row: the shared WT per-residue mean divided residue-by-
    residue by that variant's per-residue mean pLDDT over the ensemble. Rows are
    truncated to the common residue length. Returns an (n_variants x n_residues) array.
    A destabilising variant has mut < WT, so its ratio rises above 1.
    """
    rows = []
    for f in files:
        _, mut_mean, _ = load_plddt_profile(f)
        n = min(len(mut_mean), len(wt_mean))
        denom = np.where(np.abs(mut_mean[:n]) > 1e-10, mut_mean[:n], 1e-10)
        rows.append(wt_mean[:n] / denom)
    if not rows:
        return np.empty((0, 0))
    length = min(len(r) for r in rows)
    return np.vstack([r[:length] for r in rows])


def _class_d1_profiles(files, wt_mean):
    """Stack per-variant per-residue D1 terms |mut_mean/wt_mean - 1| into one array.

    Same construction as _class_ratio_profiles but keeps the D1 residue term (the
    sign-agnostic per-residue magnitude), so averaging a class-mean row over residues
    reproduces that class's mean feature-table d1_mean (now a mean, not a sum).
    """
    rows = []
    for f in files:
        _, mut_mean, _ = load_plddt_profile(f)
        n = min(len(mut_mean), len(wt_mean))
        denom = np.where(np.abs(wt_mean[:n]) > 1e-10, wt_mean[:n], 1e-10)
        rows.append(np.abs(mut_mean[:n] / denom - 1.0))
    if not rows:
        return np.empty((0, 0))
    length = min(len(r) for r in rows)
    return np.vstack([r[:length] for r in rows])


def _plot_class_profile_panel(x, drv_mean, drv_band, pas_mean, pas_band, *, n_drv, n_pas,
                              ylabel, ref_y, ref_ls, xlabel, hotspot_range, title, out_path,
                              drv_label=None, pas_label=None, legend_loc="upper right"):
    """Shared single-panel renderer for the driver/passenger per-residue class-mean views."""
    C_DRV, C_PAS = "#d1495b", "#2e86ab"
    fig, ax = plt.subplots(figsize=(13, 5))
    if hotspot_range and len(hotspot_range) == 2:
        ax.axvspan(float(hotspot_range[0]), float(hotspot_range[1]),
                   color="#f0c419", alpha=0.12, label="hotspot range", zorder=0)
    if ref_y is not None:
        ax.axhline(ref_y, color="#888888", lw=0.9, ls=ref_ls, zorder=1)
    ax.fill_between(x, drv_mean - drv_band, drv_mean + drv_band, color=C_DRV, alpha=0.20, linewidth=0)
    ax.fill_between(x, pas_mean - pas_band, pas_mean + pas_band, color=C_PAS, alpha=0.20, linewidth=0)
    ax.plot(x, drv_mean, color=C_DRV, lw=1.7, label=drv_label or f"Driver mean ±bootstrap SD (n={n_drv})")
    ax.plot(x, pas_mean, color=C_PAS, lw=1.7, label=pas_label or f"Passenger mean ±bootstrap SD (n={n_pas})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=9)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")


def _mean_bootstrap_sd(arr, *, n_resamples, rng):
    mean = np.nanmean(arr, axis=0)
    return mean, bootstrap_mean_sd(arr, n_resamples=n_resamples, rng=rng)


def plot_d1_profile_by_class(
    wt_file: Path | str,
    driver_files: Sequence[Path | str],
    passenger_files: Sequence[Path | str],
    *,
    residue_start: Optional[int] = None,
    hotspot_range: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    out_path: Path | str | None = None,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 42,
) -> dict:
    """Single-panel per-residue D1 term |mut/WT - 1|, class-mean for drivers vs passengers.

    Per residue, per variant: d1(r) = |mean_pLDDT_mut(r)/mean_pLDDT_wt(r) - 1| (index
    column dropped). The two lines are class means with mutation-bootstrap SD bands;
    a residue near 0 is unchanged from WT, higher means more perturbed. This
    is the per-residue decomposition of the D1 score -- summing a class-mean line over
    residues gives that class's mean feature-table d1_mean -- so it shows WHERE along the
    sequence the driver/passenger D1 separation originates. x-axis is the residue array
    index unless residue_start is given.
    """
    _, wt_mean, _ = load_plddt_profile(wt_file)
    drv = _class_d1_profiles(list(driver_files), wt_mean)
    pas = _class_d1_profiles(list(passenger_files), wt_mean)
    if drv.size == 0 or pas.size == 0:
        raise ValueError("need at least one driver and one passenger profile to plot D1")
    length = min(drv.shape[1], pas.shape[1])
    drv, pas = drv[:, :length], pas[:, :length]
    x = np.arange(length) + (int(residue_start) if residue_start is not None else 0)
    rng = np.random.default_rng(int(bootstrap_seed))
    drv_mean, drv_bootstrap_sd = _mean_bootstrap_sd(
        drv, n_resamples=bootstrap_resamples, rng=rng,
    )
    pas_mean, pas_bootstrap_sd = _mean_bootstrap_sd(
        pas, n_resamples=bootstrap_resamples, rng=rng,
    )
    _plot_class_profile_panel(
        x, drv_mean, drv_bootstrap_sd, pas_mean, pas_bootstrap_sd,
        n_drv=drv.shape[0], n_pas=pas.shape[0],
        drv_label=(f"Driver mean ±bootstrap SD "
                   f"(n={drv.shape[0]}, B={bootstrap_resamples})"),
        pas_label=(f"Passenger mean ±bootstrap SD "
                   f"(n={pas.shape[0]}, B={bootstrap_resamples})"),
        ylabel="per-residue D1  |mutant/WT − 1|", ref_y=0.0, ref_ls="-",
        xlabel="residue number" if residue_start is not None else "residue array index",
        hotspot_range=hotspot_range,
        title=title or "Per-residue D1 term: driver vs passenger class means",
        out_path=out_path,
    )
    return {
        "n_drivers": int(drv.shape[0]),
        "n_passengers": int(pas.shape[0]),
        "n_residues": int(length),
        # Mean over residues of the class-mean per-residue D1 line -> reconciles with
        # the (now mean-based) feature-table d1_mean. Renamed from the old *_d1_sum so
        # legacy sum-based artifacts can't be mistaken for these.
        "driver_d1_mean": float(np.nanmean(drv_mean)),
        "passenger_d1_mean": float(np.nanmean(pas_mean)),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "out_path": str(out_path) if out_path else None,
    }


def plot_plddt_ratio_exemplars(
    wt_file: Path | str,
    driver_file: Path | str | None,
    passenger_file: Path | str | None,
    *,
    driver_label: str = "Driver",
    passenger_label: str = "Passenger",
    residue_start: Optional[int] = None,
    hotspot_range: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    out_path: Path | str | None = None,
) -> dict:
    """Single-panel per-residue pLDDT ratio (WT/mutant) for ONE driver and ONE passenger.

    The exemplar counterpart of plot_plddt_ratio_by_class: instead of class means, it
    shows two individual variants' per-residue ratio = mean_pLDDT_wt(r)/mean_pLDDT_mut(r)
    (index column dropped), each with a light band from the mut ensemble SD. WT is not
    drawn -- its ratio is identically 1, marked by the dashed reference line; a
    destabilising variant rises above 1. Each variant's mutation site is marked when
    residue_start places it on a real residue axis.
    """
    _, wt_mean, _ = load_plddt_profile(wt_file)
    specs = [(driver_label, driver_file, "#d1495b"), (passenger_label, passenger_file, "#2e86ab")]

    fig, ax = plt.subplots(figsize=(13, 5))
    x_offset = int(residue_start) if residue_start is not None else 0
    if hotspot_range and len(hotspot_range) == 2:
        ax.axvspan(float(hotspot_range[0]), float(hotspot_range[1]),
                   color="#f0c419", alpha=0.12, label="hotspot range", zorder=0)
    ax.axhline(1.0, color="#888888", lw=0.9, ls="--", zorder=1, label="WT (ratio ≡ 1)")

    result: dict = {}
    for name, path, color in specs:
        if path is None:
            continue
        _, mut_mean, mut_std = load_plddt_profile(path)
        n = min(len(mut_mean), len(wt_mean))
        denom = np.where(np.abs(mut_mean[:n]) > 1e-10, mut_mean[:n], 1e-10)
        ratio = wt_mean[:n] / denom
        band = ratio * (mut_std[:n] / denom)  # relative spread from the mut ensemble SD
        x = np.arange(n) + x_offset
        ax.fill_between(x, ratio - band, ratio + band, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, ratio, color=color, lw=1.7, label=f"{name} ratio")
        result[name] = {"file": str(path), "n_residues": int(n),
                        "max_ratio": float(np.nanmax(ratio))}
        if residue_start is not None:
            code = extract_mutation_code(Path(path).name)
            digits = re.search(r"\d+", code) if code else None
            if digits:
                ax.axvline(int(digits.group()), color=color, ls=":", lw=1)

    ax.set_xlabel("residue number" if residue_start is not None else "residue array index")
    ax.set_ylabel("pLDDT ratio (WT / mutant)")
    ax.set_title(title or "Per-residue pLDDT ratio: exemplar driver vs passenger")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    result["out_path"] = str(out_path) if out_path else None
    return result


def plot_d1_exemplars(
    wt_file: Path | str,
    driver_file: Path | str | None,
    passenger_file: Path | str | None,
    *,
    driver_label: str = "Driver",
    passenger_label: str = "Passenger",
    residue_start: Optional[int] = None,
    hotspot_range: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    out_path: Path | str | None = None,
) -> dict:
    """Single-panel per-residue D1 term |mut/WT - 1| for ONE driver and ONE passenger.

    The exemplar counterpart of plot_d1_profile_by_class: two individual variants'
    per-residue D1 term d1(r) = |mean_pLDDT_mut(r)/mean_pLDDT_wt(r) - 1| (index column
    dropped), each with a light band of mut ensemble SD / wt mean. WT is not drawn -- its
    D1 is identically 0, marked by the reference line. Summing a variant's line over
    residues gives that variant's feature-table d1_mean. Mutation sites are marked when
    residue_start places them on a real residue axis.
    """
    _, wt_mean, _ = load_plddt_profile(wt_file)
    specs = [(driver_label, driver_file, "#d1495b"), (passenger_label, passenger_file, "#2e86ab")]

    fig, ax = plt.subplots(figsize=(13, 5))
    x_offset = int(residue_start) if residue_start is not None else 0
    if hotspot_range and len(hotspot_range) == 2:
        ax.axvspan(float(hotspot_range[0]), float(hotspot_range[1]),
                   color="#f0c419", alpha=0.12, label="hotspot range", zorder=0)
    ax.axhline(0.0, color="#888888", lw=0.9, zorder=1, label="WT (D1 ≡ 0)")

    result: dict = {}
    for name, path, color in specs:
        if path is None:
            continue
        _, mut_mean, mut_std = load_plddt_profile(path)
        n = min(len(mut_mean), len(wt_mean))
        denom = np.where(np.abs(wt_mean[:n]) > 1e-10, wt_mean[:n], 1e-10)
        d1 = np.abs(mut_mean[:n] / denom - 1.0)
        band = mut_std[:n] / denom
        x = np.arange(n) + x_offset
        ax.fill_between(x, np.maximum(d1 - band, 0.0), d1 + band, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, d1, color=color, lw=1.7, label=f"{name} D1")
        # d1_mean (mean over residues) reconciles with the mean-based feature-table
        # d1_mean; renamed from the old d1_sum so the two conventions can't be confused.
        result[name] = {"file": str(path), "n_residues": int(n), "d1_mean": float(np.nanmean(d1))}
        if residue_start is not None:
            code = extract_mutation_code(Path(path).name)
            digits = re.search(r"\d+", code) if code else None
            if digits:
                ax.axvline(int(digits.group()), color=color, ls=":", lw=1)

    ax.set_xlabel("residue number" if residue_start is not None else "residue array index")
    ax.set_ylabel("per-residue D1  |mutant/WT − 1|")
    ax.set_title(title or "Per-residue D1 term: exemplar driver vs passenger")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    result["out_path"] = str(out_path) if out_path else None
    return result


def plot_plddt_ratio_by_class(
    wt_file: Path | str,
    driver_files: Sequence[Path | str],
    passenger_files: Sequence[Path | str],
    *,
    residue_start: Optional[int] = None,
    hotspot_range: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    out_path: Path | str | None = None,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 42,
) -> dict:
    """Single-panel per-residue pLDDT ratio (WT/mutant), class-mean for drivers vs passengers.

    For every variant, ratio(r) = the shared WT per-residue mean divided by its own
    per-residue mean pLDDT (each mean over that variant's ensemble, residue-index column
    dropped). The two lines are the class means of that ratio -- one over all drivers,
    one over all passengers -- with mutation-bootstrap SD bands for each class mean.
    A reference line at
    ratio = 1 marks "no change from WT": where a class sits above 1 it has lost
    confidence relative to WT, and the gap between the two lines is the driver-vs-
    passenger difference. x-axis is the residue array index unless residue_start is given.
    """
    _, wt_mean, _ = load_plddt_profile(wt_file)
    drv = _class_ratio_profiles(list(driver_files), wt_mean)
    pas = _class_ratio_profiles(list(passenger_files), wt_mean)
    if drv.size == 0 or pas.size == 0:
        raise ValueError("need at least one driver and one passenger profile to plot the ratio")

    length = min(drv.shape[1], pas.shape[1])
    drv, pas = drv[:, :length], pas[:, :length]
    x = np.arange(length) + (int(residue_start) if residue_start is not None else 0)
    rng = np.random.default_rng(int(bootstrap_seed))
    drv_mean, drv_bootstrap_sd = _mean_bootstrap_sd(
        drv, n_resamples=bootstrap_resamples, rng=rng,
    )
    pas_mean, pas_bootstrap_sd = _mean_bootstrap_sd(
        pas, n_resamples=bootstrap_resamples, rng=rng,
    )
    _plot_class_profile_panel(
        x, drv_mean, drv_bootstrap_sd, pas_mean, pas_bootstrap_sd,
        n_drv=drv.shape[0], n_pas=pas.shape[0],
        drv_label=(f"Driver mean ratio ±bootstrap SD "
                   f"(n={drv.shape[0]}, B={bootstrap_resamples})"),
        pas_label=(f"Passenger mean ratio ±bootstrap SD "
                   f"(n={pas.shape[0]}, B={bootstrap_resamples})"),
        ylabel="pLDDT ratio (WT / mutant)", ref_y=1.0, ref_ls="--",
        xlabel="residue number" if residue_start is not None else "residue array index",
        hotspot_range=hotspot_range, legend_loc="lower right",
        title=title or "Per-residue pLDDT ratio: driver vs passenger class means",
        out_path=out_path,
    )
    return {
        "n_drivers": int(drv.shape[0]),
        "n_passengers": int(pas.shape[0]),
        "n_residues": int(length),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "out_path": str(out_path) if out_path else None,
    }


def plot_ratio_of_means_distribution(
    driver_vals: Sequence[float],
    passenger_vals: Sequence[float],
    *,
    title: Optional[str] = None,
    ylabel: str = "pLDDT ratio of means (WT / mutant)",
    out_path: Path | str | None = None,
    ref_line: Optional[float] = 1.0,
    log_scale: bool = False,
    stat_label: str = "AUC (driver = higher ratio)",
) -> dict:
    """Scatter + violin + box of a per-variant scalar, Driver vs Passenger.

    Each point is one variant's whole-protein scalar -- for the pLDDT ratio that is
    mean(WT pLDDT) / mean(mutant pLDDT), one number per variant. The violin shows the
    class distribution, the box its quartiles/median, and the jittered points every
    variant. ``ref_line`` draws a dashed "no change from WT" reference (1.0 for the
    ratio, 0.0 for D1); pass None to omit it, which is what a log axis needs since D1
    spans orders of magnitude and never reaches its own reference. Annotated with the
    two-sided Mann-Whitney p and a separation AUC (drivers treated as the higher-valued
    class), so the panel states how well this scalar separates classes.
    """
    drv = np.asarray([v for v in driver_vals if np.isfinite(v)], dtype=float)
    pas = np.asarray([v for v in passenger_vals if np.isfinite(v)], dtype=float)
    if drv.size == 0 or pas.size == 0:
        raise ValueError("need at least one driver and one passenger value")

    data = [drv, pas]
    positions = [1, 2]
    colors = ["#d1495b", "#2e86ab"]
    labels = ["Driver", "Passenger"]

    fig, ax = plt.subplots(figsize=(7, 6))
    if ref_line is not None:
        ax.axhline(ref_line, color="#888888", ls="--", lw=0.9, zorder=0)

    # A log axis needs the violin KDE computed in log space, otherwise the density
    # is estimated on a linear grid and rendered on a log one -- the shape shown
    # would not be the shape of the data.
    if log_scale:
        if min(float(v.min()) for v in data) <= 0:
            raise ValueError("log_scale requires strictly positive values")
        ax.set_yscale("log")
        vp_data = [np.log10(v) for v in data]
    else:
        vp_data = data

    vp = ax.violinplot(vp_data, positions=positions, widths=0.85, showextrema=False)
    if log_scale:
        for body in vp["bodies"]:
            verts = body.get_paths()[0].vertices
            verts[:, 1] = 10.0 ** verts[:, 1]
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.22)
        body.set_edgecolor(c)
    bp = ax.boxplot(data, positions=positions, widths=0.16, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.4),
                    whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(c)

    rng = np.random.default_rng(0)
    for pos, vals, c in zip(positions, data, colors):
        jitter = (rng.random(len(vals)) - 0.5) * 0.28
        ax.scatter(pos + jitter, vals, s=11, color=c, alpha=0.40, edgecolor="none", zorder=3)

    try:
        _, mw_p = stats.mannwhitneyu(drv, pas, alternative="two-sided")
    except ValueError:
        mw_p = float("nan")
    # AUC with drivers as the higher-ratio class (WT/mut rises when mut destabilises):
    # score higher = more driver-like.
    y = np.r_[np.ones(len(drv)), np.zeros(len(pas))]
    score = np.r_[drv, pas]
    try:
        auc = float(roc_auc_score(y, score))
    except Exception:
        auc = float("nan")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{lab}\n(n={len(v)})" for lab, v in zip(labels, data)])
    ax.set_ylabel(ylabel)
    stat_line = f"Mann-Whitney p = {mw_p:.2e}    {stat_label} = {auc:.3f}"
    ax.set_title(f"{title or 'pLDDT ratio of means: driver vs passenger'}\n{stat_line}", fontsize=11)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return {
        "n_drivers": int(drv.size),
        "n_passengers": int(pas.size),
        "driver_median": float(np.median(drv)),
        "passenger_median": float(np.median(pas)),
        "mannwhitney_p": float(mw_p),
        "auc_driver_higher": auc,
        "out_path": str(out_path) if out_path else None,
    }


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


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    denom = np.where(np.abs(denominator) > eps, denominator, eps)
    return numerator / denom


def compute_feature_row(
    plddt_matrix: np.ndarray,
    *,
    mutation_code: Optional[str] = None,
    mutation_type: Optional[str] = None,
    label: Optional[int] = None,
    wt_matrix: Optional[np.ndarray] = None,
    wt_mean: Optional[float] = None,
    chain_slices: Optional[Mapping[str, slice]] = None,
) -> dict:
    values = flatten_plddt(plddt_matrix)
    if values.size == 0:
        raise ValueError("Mutation matrix has no numeric values")

    features = {
        "mutation_code": mutation_code,
        "mutation_type": mutation_type,
        "label": label,
        "mean_plddt": float(np.nanmean(values)),
        "median_plddt": float(np.nanmedian(values)),
        "std_plddt": float(np.nanstd(values)),
        "min_plddt": float(np.nanmin(values)),
        "max_plddt": float(np.nanmax(values)),
        "range_plddt": float(np.nanmax(values) - np.nanmin(values)),
        "q25_plddt": float(np.nanpercentile(values, 25)),
        "q75_plddt": float(np.nanpercentile(values, 75)),
        "iqr_plddt": float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)),
        "cv_plddt": float(np.nanstd(values) / (np.nanmean(values) + 1e-10)),
        "pct_very_high": float(np.sum(values > 90) / len(values) * 100),
        "pct_confident": float(np.sum((values > 70) & (values <= 90)) / len(values) * 100),
        "pct_low": float(np.sum((values > 50) & (values <= 70)) / len(values) * 100),
        "pct_very_low": float(np.sum(values <= 50) / len(values) * 100),
    }

    if chain_slices:
        for chain_name, chain_slice in chain_slices.items():
            chain_values = flatten_plddt(plddt_matrix[chain_slice, :])
            key = chain_name.lower()
            features[f"mean_plddt_{key}"] = float(np.nanmean(chain_values)) if chain_values.size else np.nan
            features[f"iqr_plddt_{key}"] = (
                float(np.nanpercentile(chain_values, 75) - np.nanpercentile(chain_values, 25))
                if chain_values.size
                else np.nan
            )
            features[f"cv_plddt_{key}"] = (
                float(np.nanstd(chain_values) / (np.nanmean(chain_values) + 1e-10))
                if chain_values.size
                else np.nan
            )

    if wt_matrix is not None:
        wt_n = int(wt_matrix.shape[0])
        mut_n = int(plddt_matrix.shape[0])
        length_mismatch = wt_n != mut_n
        features["d1_wt_n_residues"] = wt_n
        features["d1_mut_n_residues"] = mut_n
        features["d1_length_mismatch"] = bool(length_mismatch)

        # Names of the alignment-dependent features, so the reject path can NaN them out.
        _align_nan_keys = (
            "delta_mean", "delta_median", "delta_std", "delta_abs_mean",
            "max_local_destab", "max_local_stab",
            "d1_mean", "d1_median", "d1_std", "d1_cv",
        )

        if length_mismatch and not D1_ALLOW_LENGTH_TRUNCATION:
            # Fail visibly: a WT/mutant residue-count mismatch means the row-by-row
            # ratio is not well-defined, so emit NaN rather than a silently-truncated
            # (invalid) D1. Flip D1_ALLOW_LENGTH_TRUNCATION to restore legacy behavior.
            warnings.warn(
                f"pLDDT residue-count mismatch (WT={wt_n}, mutant={mut_n}); D1/delta "
                f"metrics set to NaN. Set structure_metrics.D1_ALLOW_LENGTH_TRUNCATION "
                f"= True to restore legacy min-length truncation.",
                RuntimeWarning,
            )
            for key in _align_nan_keys:
                features[key] = np.nan
            features["n_destabilized"] = 0
            features["n_stabilized"] = 0
            if chain_slices:
                for chain_name in chain_slices:
                    chain_key = chain_name.lower()
                    features[f"d1_mean_{chain_key}"] = np.nan
                    features[f"d1_median_{chain_key}"] = np.nan
                    features[f"d1_std_{chain_key}"] = np.nan
                    features[f"d1_cv_{chain_key}"] = np.nan
            return features

        n_rows = min(wt_n, mut_n)  # equal to both unless legacy truncation is enabled
        wt_aligned = wt_matrix[:n_rows, :]
        mut_aligned = plddt_matrix[:n_rows, :]

        wt_row_mean = np.nanmean(wt_aligned, axis=1)
        mut_row_mean = np.nanmean(mut_aligned, axis=1)

        delta_row = mut_row_mean - wt_row_mean
        # One per-residue D1 series from the ensemble-MEAN pLDDT (identical definition
        # to the ensemble-aware D1 path). d1_mean/median/std are ACROSS-RESIDUE reductions
        # of THIS series -- the suffix names the residue-axis reduction, not a pLDDT
        # reduction -- so they are length-independent and comparable across proteins.
        d1_per_residue = np.abs(_safe_ratio(mut_row_mean, wt_row_mean) - 1.0)
        d1_mean = float(np.nanmean(d1_per_residue))
        d1_std = float(np.nanstd(d1_per_residue))

        features.update(
            {
                "delta_mean": float(np.nanmean(delta_row)),
                "delta_median": float(np.nanmedian(delta_row)),
                "delta_std": float(np.nanstd(delta_row)),
                "delta_abs_mean": float(np.nanmean(np.abs(delta_row))),
                "n_destabilized": int(np.sum(delta_row < -5)),
                "n_stabilized": int(np.sum(delta_row > 5)),
                "max_local_destab": float(np.nanmin(delta_row)),
                "max_local_stab": float(np.nanmax(delta_row)),
                "d1_mean": d1_mean,
                "d1_median": float(np.nanmedian(d1_per_residue)),
                "d1_std": d1_std,
                "d1_cv": float(d1_std / (abs(d1_mean) + 1e-10)),
            }
        )

        if wt_mean is not None:
            features["pct_change"] = float((features["mean_plddt"] - wt_mean) / (wt_mean + 1e-10) * 100)

        if chain_slices:
            for chain_name, chain_slice in chain_slices.items():
                chain_key = chain_name.lower()
                wt_chain = wt_aligned[chain_slice, :]
                mut_chain = mut_aligned[chain_slice, :]
                if wt_chain.size == 0 or mut_chain.size == 0:
                    features[f"d1_mean_{chain_key}"] = np.nan
                    features[f"d1_median_{chain_key}"] = np.nan
                    features[f"d1_std_{chain_key}"] = np.nan
                    features[f"d1_cv_{chain_key}"] = np.nan
                    continue

                wt_chain_mean = np.nanmean(wt_chain, axis=1)
                mut_chain_mean = np.nanmean(mut_chain, axis=1)

                d1_chain = np.abs(_safe_ratio(mut_chain_mean, wt_chain_mean) - 1.0)
                d1_chain_mean_v = float(np.nanmean(d1_chain))
                d1_chain_std = float(np.nanstd(d1_chain))

                features[f"d1_mean_{chain_key}"] = d1_chain_mean_v
                features[f"d1_median_{chain_key}"] = float(np.nanmedian(d1_chain))
                features[f"d1_std_{chain_key}"] = d1_chain_std
                features[f"d1_cv_{chain_key}"] = d1_chain_std / (abs(d1_chain_mean_v) + 1e-10)

    return features


def build_mutation_feature_table(
    plddt_files: Sequence[Path | str],
    *,
    known_mutations: Optional[Sequence[str]] = None,
    label_map: Optional[Mapping[str, tuple[str, int]]] = None,
    wt_tokens: Sequence[str] = ("WT",),
    chain_slices: Optional[Mapping[str, slice]] = None,
) -> pd.DataFrame:
    files = [Path(p) for p in plddt_files]
    wt_file = find_wild_type_file(files, wt_tokens=wt_tokens)
    wt_summary = summarize_wild_type(wt_file)

    rows: List[dict] = []

    for plddt_file in files:
        if wt_file is not None and plddt_file.resolve() == wt_file.resolve():
            continue

        mutation_code = extract_mutation_code(plddt_file.name, known_mutations=known_mutations)
        if not mutation_code:
            continue

        mutation_type = None
        label = None
        if label_map and mutation_code in label_map:
            mutation_type, label = label_map[mutation_code]

        matrix = load_plddt_matrix(plddt_file)
        row = compute_feature_row(
            matrix,
            mutation_code=mutation_code,
            mutation_type=mutation_type,
            label=label,
            wt_matrix=wt_summary["wt_matrix"],
            wt_mean=wt_summary["wt_mean"],
            chain_slices=chain_slices,
        )
        row["source_file"] = str(plddt_file)
        rows.append(row)

    return pd.DataFrame(rows)


def build_plddt_ks_tables(
    plddt_files: Sequence[Path | str],
    *,
    known_mutations: Optional[Sequence[str]] = None,
    wt_tokens: Sequence[str] = ("WT",),
    wt_file: Path | str | None = None,
    fdr_alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build mutation-level summaries and residue-level WT-vs-mutant KS results.

    The summary table contains classifier-ready reductions of the KS-statistic
    profile. The residue table preserves every KS statistic, raw p-value, BH
    q-value, and sample count so the spatial signal remains inspectable.
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
        raise ValueError("No WT pLDDT CSV found for per-residue KS comparison")

    wt_matrix = load_plddt_matrix(chosen_wt)
    summary_rows: list[dict[str, Any]] = []
    residue_tables: list[pd.DataFrame] = []

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
                "KS metrics set to NaN.",
                RuntimeWarning,
            )
            summary_rows.append(
                {
                    **base,
                    "status": "residue_count_mismatch",
                    "plddt_ks_mean": np.nan,
                    "plddt_ks_median": np.nan,
                    "plddt_ks_max": np.nan,
                    "plddt_ks_fraction_fdr": np.nan,
                    "plddt_ks_n_residues_tested": 0,
                    "plddt_ks_min_wt_samples": 0,
                    "plddt_ks_min_mutant_samples": 0,
                    "d1_mean": np.nan,
                }
            )
            continue

        residue_result = compute_per_residue_plddt_ks(
            wt_matrix,
            mutant_matrix,
            fdr_alpha=fdr_alpha,
        )
        residue_result.insert(0, "mutation_code", mutation_code)
        residue_result["source_file"] = str(plddt_file)
        residue_tables.append(residue_result)
        summary_rows.append(
            {
                **base,
                "status": "ok",
                **summarize_per_residue_plddt_ks(
                    residue_result, fdr_alpha=fdr_alpha
                ),
                # Reuse the established D1 implementation so the comparison plot
                # cannot drift to a second ratio convention.
                "d1_mean": compute_feature_row(
                    mutant_matrix, wt_matrix=wt_matrix
                )["d1_mean"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    residue_df = (
        pd.concat(residue_tables, ignore_index=True)
        if residue_tables
        else pd.DataFrame(
            columns=[
                "mutation_code", "residue_index", "ks_statistic", "ks_pvalue",
                "n_wt", "n_mutant", "ks_qvalue", "ks_significant_fdr",
                "source_file",
            ]
        )
    )
    return summary_df, residue_df


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

    This is intentionally a separate output family from both the established D1
    feature table and the KS test table.  Labels are attached later by the
    orchestration layer and therefore cannot leak into metric construction.
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
                    "d1_mean": np.nan,
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
        d1_mean = compute_feature_row(
            mutant_matrix, wt_matrix=wt_matrix
        )["d1_mean"]
        summary_rows.append(
            {
                **base,
                "status": "ok",
                **summary,
                "d1_mean": d1_mean,
                "d1_consistency_abs_error": abs(
                    float(summary["d1_from_distribution_mean"]) - float(d1_mean)
                ),
            }
        )

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


def _normalize_plot_class_label(value: Any) -> str:
    text = str(value).strip()
    lower = text.lower()

    if lower == "driver":
        return "Driver"
    if lower == "passenger":
        return "Passenger"
    if lower in {"hotspot", "hot-spot"}:
        return "Hotspot"
    if lower in {"non-hotspot", "non hotspot", "non_hotspot", "nonhotspot"}:
        return "Non-Hotspot"

    return text


def plot_rmsf_vs_d1_scatter(
    mutation_df: pd.DataFrame,
    rmsf_summary_df: pd.DataFrame,
    *,
    mutation_code_col: str | None = None,
    d1_col: str | None = None,
    class_col: str | None = None,
    artifact_col: str = "artifact",
    status_col: str = "status",
    status_ok: str = "ok",
    rmsf_col: str | None = None,
    dataset_name: str = "Dataset",
    plot_title: str | None = None,
    class_order: Optional[Sequence[str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    show_regression: bool = True,
    marker_size: float = 80.0,
    alpha: float = 0.9,
    show_metrics_on_plot: bool = True,
    metrics_box_loc: str = "upper left",
    out_path: Path | str | None = None,
    ax: Optional[plt.Axes] = None,
) -> Dict[str, Any]:
    """Plot mutation-level RMSF vs D1 scatter with class coloring.

    The function aligns mutation-level D1 values from ``mutation_df`` with
    mutation-level RMSF values from ``rmsf_summary_df`` by inferred mutation code
    (e.g., ``A273E``), then creates a scatter plot.

    Returns a dictionary containing the merged table and plot handles.
    """

    if mutation_df is None or mutation_df.empty:
        raise ValueError("mutation_df is empty")
    if rmsf_summary_df is None or rmsf_summary_df.empty:
        raise ValueError("rmsf_summary_df is empty")

    def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    mut_code_col = mutation_code_col or _pick_column(
        mutation_df,
        ["mutation_code", "Allele", "allele", "variant", "Variant", "artifact"],
    )
    if mut_code_col is None:
        raise KeyError(
            "Could not infer mutation-code column in mutation_df. "
            "Provide mutation_code_col explicitly."
        )

    selected_d1_col = d1_col or _pick_column(
        mutation_df,
        ["d1_mean", "d1_median", "d1_avg", "d1", "D1_mean", "D1"],
    )
    if selected_d1_col is None:
        raise KeyError(
            "No D1 column found in mutation_df (expected one of: "
            "d1_mean, d1_median, d1_avg, d1, D1_mean, D1)."
        )

    selected_rmsf_col = rmsf_col or _pick_column(
        rmsf_summary_df,
        ["rmsf_mean", "rmsf_median", "rmsf_std"],
    )
    if selected_rmsf_col is None:
        raise KeyError(
            "No RMSF summary column found (expected one of: "
            "rmsf_mean, rmsf_median, rmsf_std)."
        )

    mut_df = mutation_df.copy()
    raw_mut_code = mut_df[mut_code_col].astype(str)
    mut_df["mutation_code"] = raw_mut_code.map(lambda x: extract_mutation_code(x) or x.strip().upper())

    if class_col and class_col in mut_df.columns:
        class_series = mut_df[class_col].astype(str)
    elif "mutation_type" in mut_df.columns:
        class_series = mut_df["mutation_type"].astype(str)
    elif "class" in mut_df.columns:
        class_series = mut_df["class"].astype(str)
    elif "is_driver" in mut_df.columns:
        class_series = np.where(mut_df["is_driver"].astype(bool), "Driver", "Passenger")
    elif "is_hotspot" in mut_df.columns:
        class_series = np.where(mut_df["is_hotspot"].astype(bool), "Hotspot", "Non-Hotspot")
    else:
        class_series = pd.Series(["Unknown"] * len(mut_df), index=mut_df.index)

    mut_df["plot_class"] = pd.Series(class_series, index=mut_df.index).map(_normalize_plot_class_label)
    mut_df[selected_d1_col] = pd.to_numeric(mut_df[selected_d1_col], errors="coerce")

    mut_by_code = (
        mut_df[["mutation_code", "plot_class", selected_d1_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["mutation_code", selected_d1_col])
        .groupby("mutation_code", as_index=False)
        .agg({"plot_class": "first", selected_d1_col: "mean"})
    )

    rmsf_df = rmsf_summary_df.copy()
    if status_col in rmsf_df.columns and status_ok is not None:
        rmsf_df = rmsf_df[rmsf_df[status_col].astype(str).str.lower() == str(status_ok).lower()].copy()

    if artifact_col not in rmsf_df.columns:
        raise KeyError(f"rmsf_summary_df is missing artifact column '{artifact_col}'")

    rmsf_df["mutation_code"] = rmsf_df[artifact_col].astype(str).map(lambda x: extract_mutation_code(x) or "")
    rmsf_df[selected_rmsf_col] = pd.to_numeric(rmsf_df[selected_rmsf_col], errors="coerce")

    rmsf_by_code = (
        rmsf_df[["mutation_code", selected_rmsf_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["mutation_code", selected_rmsf_col])
        .query("mutation_code != ''")
        .groupby("mutation_code", as_index=False)[selected_rmsf_col]
        .mean()
    )

    plot_df = pd.merge(mut_by_code, rmsf_by_code, on="mutation_code", how="inner")
    plot_df = plot_df.dropna(subset=[selected_d1_col, selected_rmsf_col]).copy()

    if plot_df.empty:
        raise ValueError(
            "No overlapping mutations between mutation_df and rmsf_summary_df "
            "after filtering and merge."
        )

    x_vals = plot_df[selected_rmsf_col].to_numpy(dtype=float)
    y_vals = plot_df[selected_d1_col].to_numpy(dtype=float)
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[finite_mask]
    y_vals = y_vals[finite_mask]

    correlation_metrics: Dict[str, float | int] = {
        "n_points": int(len(x_vals)),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "linear_slope": np.nan,
        "linear_intercept": np.nan,
        "linear_r": np.nan,
        "linear_p": np.nan,
        "linear_stderr": np.nan,
        "linear_intercept_stderr": np.nan,
        "linear_r2": np.nan,
    }

    if len(x_vals) >= 2:
        try:
            pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)
            correlation_metrics["pearson_r"] = float(pearson_r)
            correlation_metrics["pearson_p"] = float(pearson_p)
        except ValueError:
            pass

        try:
            spearman_rho, spearman_p = stats.spearmanr(x_vals, y_vals)
            correlation_metrics["spearman_rho"] = float(spearman_rho)
            correlation_metrics["spearman_p"] = float(spearman_p)
        except ValueError:
            pass

        try:
            linreg = stats.linregress(x_vals, y_vals)
            correlation_metrics["linear_slope"] = float(linreg.slope)
            correlation_metrics["linear_intercept"] = float(linreg.intercept)
            correlation_metrics["linear_r"] = float(linreg.rvalue)
            correlation_metrics["linear_p"] = float(linreg.pvalue)
            correlation_metrics["linear_stderr"] = float(linreg.stderr)
            correlation_metrics["linear_intercept_stderr"] = float(linreg.intercept_stderr)
            correlation_metrics["linear_r2"] = float(linreg.rvalue ** 2)
        except ValueError:
            pass

    classes_present = list(pd.unique(plot_df["plot_class"]))
    if class_order:
        hue_order = [c for c in class_order if c in classes_present]
        if not hue_order:
            hue_order = sorted(classes_present)
    elif {"Passenger", "Driver"}.issubset(set(classes_present)):
        hue_order = ["Passenger", "Driver"]
    elif {"Non-Hotspot", "Hotspot"}.issubset(set(classes_present)):
        hue_order = ["Non-Hotspot", "Hotspot"]
    else:
        hue_order = sorted(classes_present)

    base_colors = {
        "Driver": "#D1495B",
        "Passenger": "#2E86AB",
        "Hotspot": "#D1495B",
        "Non-Hotspot": "#2E86AB",
        "Unknown": "#6C757D",
    }
    if colors:
        base_colors.update(dict(colors))
    palette = {c: base_colors.get(c, "#6C757D") for c in hue_order}

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
        created_figure = True
    else:
        fig = ax.figure

    sns.scatterplot(
        data=plot_df,
        x=selected_rmsf_col,
        y=selected_d1_col,
        hue="plot_class",
        hue_order=hue_order,
        palette=palette,
        s=marker_size,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    if show_regression and len(plot_df) >= 3:
        sns.regplot(
            data=plot_df,
            x=selected_rmsf_col,
            y=selected_d1_col,
            scatter=False,
            color="#4c4c4c",
            line_kws={"linewidth": 1.5, "alpha": 0.8},
            ax=ax,
        )

    if show_metrics_on_plot:
        metrics_lines: List[str] = [f"n = {int(correlation_metrics['n_points'])}"]

        pearson_r = float(correlation_metrics["pearson_r"])
        pearson_p = float(correlation_metrics["pearson_p"])
        if np.isfinite(pearson_r):
            if np.isfinite(pearson_p):
                metrics_lines.append(f"Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})")
            else:
                metrics_lines.append(f"Pearson r = {pearson_r:.3f}")

        spearman_rho = float(correlation_metrics["spearman_rho"])
        spearman_p = float(correlation_metrics["spearman_p"])
        if np.isfinite(spearman_rho):
            if np.isfinite(spearman_p):
                metrics_lines.append(f"Spearman rho = {spearman_rho:.3f} (p={spearman_p:.2e})")
            else:
                metrics_lines.append(f"Spearman rho = {spearman_rho:.3f}")

        linear_r2 = float(correlation_metrics["linear_r2"])
        if np.isfinite(linear_r2):
            metrics_lines.append(f"Linear R^2 = {linear_r2:.3f}")

        loc_map = {
            "upper left": (0.02, 0.98, "left", "top"),
            "upper right": (0.98, 0.98, "right", "top"),
            "lower left": (0.02, 0.02, "left", "bottom"),
            "lower right": (0.98, 0.02, "right", "bottom"),
        }
        text_x, text_y, text_ha, text_va = loc_map.get(
            str(metrics_box_loc).strip().lower(),
            loc_map["upper left"],
        )

        ax.text(
            text_x,
            text_y,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            ha=text_ha,
            va=text_va,
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#4c4c4c",
                "alpha": 0.85,
            },
        )

    legend_title = "Mutation Type" if {"Passenger", "Driver"}.issubset(set(hue_order)) else "Mutation Class"

    ax.set_title(plot_title or f"{dataset_name} RMSF vs D1")
    ax.set_xlabel(f"{selected_rmsf_col.replace('_', ' ').title()} (A)")
    ax.set_ylabel(selected_d1_col.replace("_", " ").upper())
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(title=legend_title, frameon=True)
    fig.tight_layout()

    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")

    return {
        "plot_df": plot_df,
        "d1_col": selected_d1_col,
        "rmsf_col": selected_rmsf_col,
        "mutation_code_col": mut_code_col,
        "correlation_metrics": correlation_metrics,
        "figure": fig,
        "ax": ax,
        "created_figure": created_figure,
    }


def plot_rmsf_metric_scatter_grid(
    mutation_df: pd.DataFrame,
    *,
    whole_summary_df: Optional[pd.DataFrame],
    whole_delta_summary_df: Optional[pd.DataFrame],
    interface_summary_df: Optional[pd.DataFrame],
    interface_delta_summary_df: Optional[pd.DataFrame],
    mutation_code_col: str | None = None,
    d1_col: str | None = None,
    class_col: str | None = None,
    artifact_col: str = "artifact",
    status_col: str = "status",
    status_ok: str = "ok",
    dataset_name: str = "Dataset",
    class_order: Optional[Sequence[str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    show_regression: bool = True,
    marker_size: float = 80.0,
    alpha: float = 0.9,
    show_metrics_on_plot: bool = True,
    metrics_box_loc: str = "upper left",
    figsize: Tuple[float, float] = (24.0, 14.0),
    out_path: Path | str | None = None,
) -> Dict[str, Any]:
    """Plot a 2x3 grid of RMSF-vs-D1 scatter panels.

    Panel layout:
    - Row 1 (whole structure): mean RMSF, mean |delta RMSF| vs WT, signed delta RMSF vs WT
    - Row 2 (interface only): mean RMSF, mean |delta RMSF| vs WT, signed delta RMSF vs WT
    """

    if mutation_df is None or mutation_df.empty:
        raise ValueError("mutation_df is empty")

    panel_specs = [
        ("Whole mean RMSF", whole_summary_df, "rmsf_mean", "whole_mean_rmsf"),
        (
            "Whole mean |delta RMSF| vs WT",
            whole_delta_summary_df,
            "delta_abs_mean",
            "whole_delta_abs_mean",
        ),
        (
            "Whole signed delta RMSF vs WT",
            whole_delta_summary_df,
            "delta_mean",
            "whole_delta_mean",
        ),
        ("Interface mean RMSF", interface_summary_df, "rmsf_mean", "interface_mean_rmsf"),
        (
            "Interface mean |delta RMSF| vs WT",
            interface_delta_summary_df,
            "delta_abs_mean",
            "interface_delta_abs_mean",
        ),
        (
            "Interface signed delta RMSF vs WT",
            interface_delta_summary_df,
            "delta_mean",
            "interface_delta_mean",
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    flat_axes = np.asarray(axes).reshape(-1)

    panel_results: Dict[str, Dict[str, Any]] = {}
    for ax, (panel_title, panel_df, panel_col, panel_key) in zip(flat_axes, panel_specs):
        if panel_df is None or panel_df.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{panel_title}\nNo data available", ha="center", va="center")
            panel_results[panel_key] = {
                "panel_title": panel_title,
                "rmsf_col": panel_col,
                "error": "empty dataframe",
            }
            continue

        if panel_col not in panel_df.columns:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{panel_title}\nMissing column: {panel_col}",
                ha="center",
                va="center",
            )
            panel_results[panel_key] = {
                "panel_title": panel_title,
                "rmsf_col": panel_col,
                "error": f"missing column '{panel_col}'",
            }
            continue

        try:
            panel_result = plot_rmsf_vs_d1_scatter(
                mutation_df,
                panel_df,
                mutation_code_col=mutation_code_col,
                d1_col=d1_col,
                class_col=class_col,
                artifact_col=artifact_col,
                status_col=status_col,
                status_ok=status_ok,
                rmsf_col=panel_col,
                dataset_name=dataset_name,
                plot_title=f"{dataset_name} | {panel_title} vs D1",
                class_order=class_order,
                colors=colors,
                show_regression=show_regression,
                marker_size=marker_size,
                alpha=alpha,
                show_metrics_on_plot=show_metrics_on_plot,
                metrics_box_loc=metrics_box_loc,
                ax=ax,
            )
            panel_result["panel_title"] = panel_title
            panel_results[panel_key] = panel_result
        except Exception as exc:  # pragma: no cover - defensive plotting fallback
            ax.axis("off")
            ax.text(0.5, 0.5, f"{panel_title}\n{exc}", ha="center", va="center")
            panel_results[panel_key] = {
                "panel_title": panel_title,
                "rmsf_col": panel_col,
                "error": str(exc),
            }

    fig.tight_layout()

    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")

    return {
        "figure": fig,
        "axes": axes,
        "panel_results": panel_results,
        "panel_specs": panel_specs,
        "created_figure": True,
    }


def plot_metric_rank_with_highlights(
    summary_df: pd.DataFrame,
    *,
    metric_cols: Optional[Sequence[str]] = None,
    artifact_col: str = "artifact",
    class_col: str | None = None,
    class_lookup: Optional[Mapping[str, Any]] = None,
    hotspot_range: Optional[Tuple[int, int]] = None,
    highlight_items: Optional[Sequence[str]] = None,
    highlight_ranks: Optional[Sequence[int]] = None,
    index_color_groups: Optional[Mapping[str, Optional[Sequence[int]]]] = None,
    index_color_classes: Optional[Sequence[str]] = None,
    highlight_clusters: bool | Sequence[str] = False,
    cluster_highlight_label: bool = True,
    cluster_highlight_label_size: float = 8.5,
    cluster_highlight_label_offsets: Optional[Sequence[Tuple[float, float]]] = None,
    highlight_items_by_color: Optional[Mapping[str, Sequence[str]]] = None,
    highlight_ranks_by_color: Optional[Mapping[str, Sequence[int]]] = None,
    exclude_numeric_cols: Sequence[str] = ("n_residues", "n_interface_residues"),
    rank_ascending: bool = False,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (6.2, 4.4),
    colors: Optional[Mapping[str, str]] = None,
    point_size: float = 24.0,
    point_alpha: float = 0.35,
    highlight_size: float = 95.0,
    highlight_arrow_color: Optional[str] = None,
    highlight_arrow_darken_factor: float = 0.7,
    show_line: bool = True,
    title: str | None = None,
    out_path: Path | str | None = None,
    # highlight_wt: bool = False,
) -> Dict[str, Any]:
    """Plot per-metric rank curves and highlight selected mutations/artifacts.

    The function ranks each metric across all artifacts in ``summary_df`` and
    renders one panel per metric. Items listed in ``highlight_items`` and rows
    at ranks listed in ``highlight_ranks`` are emphasized with larger outlined
    markers and bold labels so it is easy to inspect their standing for each
    metric.

    Parameters
    ----------
    summary_df:
        DataFrame containing artifact-level metrics.
    metric_cols:
        Metrics to rank. If ``None``, all numeric columns are used except
        ``exclude_numeric_cols``.
    artifact_col:
        Column containing artifact names.
    class_col:
        Optional class column in ``summary_df`` (for color coding).
    class_lookup:
        Optional mapping from mutation code to class.
        Booleans are interpreted as Driver/Passenger.
    hotspot_range:
        Optional inclusive residue range (start, end) used to backfill class
        labels as Hotspot/Non-Hotspot when class is missing.
    highlight_items:
        Optional list like ["A246P", "C333S", ...]. Matches either full
        artifact strings or extracted mutation codes.
    highlight_ranks:
        Optional one-based ranks to highlight after each metric is sorted.
        For example, ``highlight_ranks=[1]`` highlights the top artifact in
        every metric panel. Rank 1 is highest unless ``rank_ascending=True``.
    index_color_groups:
        Optional mapping from color to residue-index groups. Rows whose
        mutation position is in a group's index list are drawn with that color.
        A single ``None`` group may be provided to color all rows not captured
        by any explicit index group.
    index_color_classes:
        Optional class labels eligible for ``index_color_groups``. For example,
        ``index_color_classes=["Driver"]`` colors only driver rows by residue
        index and leaves passengers/unknowns on their class colors.
    highlight_clusters:
        If ``True``, draw an extra colored outline around all explicit
        ``index_color_groups``. If a sequence of color strings is provided, only
        those explicit cluster colors are outlined. The ``None``/Other group is
        never treated as an explicit cluster.
    cluster_highlight_label:
        If ``True`` (default), add a small mutation-code and rank label to rows
        outlined by ``highlight_clusters``.
    cluster_highlight_label_size:
        Font size for ``highlight_clusters`` labels.
    cluster_highlight_label_offsets:
        Optional sequence of ``(x, y)`` text offsets in points. The offsets are
        cycled across cluster-highlighted rows within each metric panel to
        reduce label overlap.
    highlight_items_by_color:
        Optional mapping from color to artifact/mutation strings. These items
        are highlighted with a colored outline.
    highlight_ranks_by_color:
        Optional mapping from color to one-based ranks. These ranks are
        highlighted with a colored outline after each metric is sorted.
    highlight_arrow_color:
        Optional arrow color for highlight labels. ``None`` keeps arrows on
        the same color as the label/frame. Use ``"darker"`` to draw arrows as
        a darker same-hue version of each resolved highlight color, or pass any
        Matplotlib color string to use a fixed arrow color.
    highlight_arrow_darken_factor:
        Multiplier applied to RGB channels when
        ``highlight_arrow_color="darker"``. Lower values are darker.
    rank_ascending:
        If ``False`` (default), rank 1 is the highest metric value.
    # highlight_wt:
    #     If ``True``, also highlight the wild-type artifact if present (case-insensitive matching of "WT" in the artifact column).
    Returns
    -------
    Dict[str, Any]
        ``figure``, ``axes``, ``rank_tables`` (per-metric ranked tables), and
        ``highlights_df`` (stacked highlighted rows with ranks).
    """

    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty")
    if artifact_col not in summary_df.columns:
        raise KeyError(f"summary_df must include artifact column '{artifact_col}'")

    work_df = summary_df.copy()
    work_df[artifact_col] = work_df[artifact_col].astype(str)
    work_df["mutation_code"] = work_df[artifact_col].map(
        lambda x: extract_mutation_code(x) or str(x).strip().upper()
    )
    work_df["mutation_position"] = pd.to_numeric(
        work_df["mutation_code"].astype(str).str.extract(r"[A-Z](\d+)[A-Z]")[0],
        errors="coerce",
    )

    if metric_cols is None:
        numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_metrics = [c for c in numeric_cols if c not in set(exclude_numeric_cols)]
    else:
        selected_metrics = [c for c in metric_cols if c in work_df.columns]

    if not selected_metrics:
        raise ValueError("No plottable metric columns found")

    def _coerce_class(value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return "Driver" if bool(value) else "Passenger"
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            if float(value) in {0.0, 1.0}:
                return "Driver" if float(value) > 0 else "Passenger"
        text = str(value).strip()
        if not text:
            return None
        return _normalize_plot_class_label(text)

    class_series: pd.Series
    if class_col and class_col in work_df.columns:
        class_series = work_df[class_col].map(_coerce_class)
    elif class_lookup is not None:
        lookup = {str(k).strip().upper(): _coerce_class(v) for k, v in class_lookup.items()}
        class_series = work_df["mutation_code"].astype(str).str.upper().map(lookup)
    else:
        class_series = pd.Series([None] * len(work_df), index=work_df.index, dtype="object")

    if hotspot_range is not None:
        try:
            hs_start = int(hotspot_range[0])
            hs_end = int(hotspot_range[1])
            positions = pd.to_numeric(
                work_df["mutation_code"].astype(str).str.extract(r"[A-Z](\d+)[A-Z]")[0],
                errors="coerce",
            )
            valid_pos = positions.notna()
            inferred = pd.Series([None] * len(work_df), index=work_df.index, dtype="object")
            inferred.loc[valid_pos] = np.where(
                positions.loc[valid_pos].between(hs_start, hs_end, inclusive="both"),
                "Hotspot",
                "Non-Hotspot",
            )
            missing_mask = class_series.isna()
            class_series.loc[missing_mask] = inferred.loc[missing_mask]
        except (TypeError, ValueError):
            pass

    work_df["plot_class"] = class_series.fillna("Unknown").astype(str).map(_normalize_plot_class_label)

    def _list_like(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            return [value]
        try:
            return list(value)
        except TypeError:
            return [value]

    def _color_value_groups(
        value_by_color: Optional[Mapping[str, Sequence[Any]]],
        *,
        arg_name: str,
    ) -> List[Tuple[str, List[Any]]]:
        groups: List[Tuple[str, List[Any]]] = []
        for color, raw_values in (value_by_color or {}).items():
            color_text = str(color).strip()
            if not color_text:
                raise ValueError(f"{arg_name} contains an empty color key")
            values = _list_like(raw_values)
            if not values:
                continue
            groups.append((color_text, values))
        return groups

    def _append_highlight_source(df: pd.DataFrame, mask: pd.Series, source: str) -> None:
        if not bool(mask.any()):
            return
        current = df.loc[mask, "highlight_source"].fillna("").astype(str)
        updated: List[str] = []
        for text in current:
            pieces = [piece for piece in str(text).split("+") if piece]
            if source not in pieces:
                pieces.append(source)
            updated.append("+".join(pieces))
        df.loc[mask, "highlight_source"] = updated

    index_color_class_set: Optional[set[str]] = None
    if index_color_classes is not None:
        index_color_class_set = {
            _normalize_plot_class_label(str(value).strip())
            for value in _list_like(index_color_classes)
            if str(value).strip()
        }
        if not index_color_class_set:
            raise ValueError("index_color_classes was provided but did not contain any class labels")

    highlight_items = list(highlight_items or [])
    rank_highlights: set[int] = set()
    for rank_value in highlight_ranks or []:
        try:
            rank_int = int(rank_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"highlight_ranks must contain positive integers, got {rank_value!r}") from exc
        if rank_int < 1:
            raise ValueError(f"highlight_ranks must contain positive integers, got {rank_value!r}")
        rank_highlights.add(rank_int)

    rank_highlight_colors: Dict[int, str] = {}
    for color_text, rank_values in _color_value_groups(
        highlight_ranks_by_color,
        arg_name="highlight_ranks_by_color",
    ):
        for rank_value in rank_values:
            try:
                rank_int = int(rank_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"highlight_ranks_by_color must contain positive integers, got {rank_value!r}"
                ) from exc
            if rank_int < 1:
                raise ValueError(
                    f"highlight_ranks_by_color must contain positive integers, got {rank_value!r}"
                )
            existing_color = rank_highlight_colors.get(rank_int)
            if existing_color is not None and existing_color != color_text:
                raise ValueError(
                    f"Rank {rank_int} is assigned to multiple highlight colors: "
                    f"{existing_color!r} and {color_text!r}"
                )
            rank_highlight_colors[rank_int] = color_text

    highlight_raw: set[str] = set()
    highlight_mut: set[str] = set()
    for item in highlight_items:
        token = str(item).strip().upper()
        if not token:
            continue
        highlight_raw.add(token)
        extracted = extract_mutation_code(token)
        if extracted:
            highlight_mut.add(extracted.upper())

    artifact_upper = work_df[artifact_col].astype(str).str.upper()
    mut_upper = work_df["mutation_code"].astype(str).str.upper()
    work_df["is_highlight"] = artifact_upper.isin(highlight_raw) | mut_upper.isin(highlight_mut)
    work_df["highlight_color"] = pd.Series([None] * len(work_df), index=work_df.index, dtype="object")
    work_df["highlight_source"] = ""
    work_df.loc[work_df["is_highlight"], "highlight_color"] = "black"
    _append_highlight_source(work_df, work_df["is_highlight"], "item")

    colored_item_mask_seen = pd.Series(False, index=work_df.index)
    for color_text, items in _color_value_groups(
        highlight_items_by_color,
        arg_name="highlight_items_by_color",
    ):
        color_raw: set[str] = set()
        color_mut: set[str] = set()
        for item in items:
            token = str(item).strip().upper()
            if not token:
                continue
            color_raw.add(token)
            extracted = extract_mutation_code(token)
            if extracted:
                color_mut.add(extracted.upper())
        if not color_raw and not color_mut:
            continue

        color_mask = artifact_upper.isin(color_raw) | mut_upper.isin(color_mut)
        conflict_mask = colored_item_mask_seen & color_mask
        if conflict_mask.any():
            conflict_examples = work_df.loc[conflict_mask, artifact_col].head(5).tolist()
            raise ValueError(
                "The same artifact/mutation is assigned to multiple highlight item colors: "
                f"{conflict_examples}"
            )
        colored_item_mask_seen = colored_item_mask_seen | color_mask
        work_df.loc[color_mask, "is_highlight"] = True
        work_df.loc[color_mask, "highlight_color"] = color_text
        _append_highlight_source(work_df, color_mask, "item")

    index_color_map: Dict[int, str] = {}
    index_color_to_label: Dict[str, str] = {}
    index_color_label_order: List[str] = []
    other_index_color: Optional[str] = None

    def _format_index_group_label(positions: Sequence[int]) -> str:
        unique_positions = sorted(set(int(pos) for pos in positions))
        shown = ", ".join(str(pos) for pos in unique_positions[:5])
        suffix = ", ..." if len(unique_positions) > 5 else ""
        noun = "Index" if len(unique_positions) == 1 else "Indices"
        return f"{noun} {shown}{suffix}"

    for color_text, raw_positions in (index_color_groups or {}).items():
        color_text = str(color_text).strip()
        if not color_text:
            raise ValueError("index_color_groups contains an empty color key")

        if raw_positions is None:
            if other_index_color is not None and other_index_color != color_text:
                raise ValueError(
                    "index_color_groups may contain only one None group for unassigned indexes"
                )
            other_index_color = color_text
            index_color_to_label[color_text] = "Other index"
            if "Other index" not in index_color_label_order:
                index_color_label_order.append("Other index")
            continue

        positions: List[int] = []
        for raw_pos in _list_like(raw_positions):
            try:
                pos_int = int(raw_pos)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"index_color_groups contains a non-integer residue index: {raw_pos!r}"
                ) from exc
            if pos_int < 1:
                raise ValueError(
                    f"index_color_groups residue indexes must be positive integers, got {raw_pos!r}"
                )
            existing_color = index_color_map.get(pos_int)
            if existing_color is not None and existing_color != color_text:
                raise ValueError(
                    f"Residue index {pos_int} is assigned to multiple colors: "
                    f"{existing_color!r} and {color_text!r}"
                )
            index_color_map[pos_int] = color_text
            positions.append(pos_int)

        if positions:
            label = _format_index_group_label(positions)
            index_color_to_label[color_text] = label
            if label not in index_color_label_order:
                index_color_label_order.append(label)

    work_df["index_color"] = pd.Series([None] * len(work_df), index=work_df.index, dtype="object")
    work_df["index_color_label"] = pd.Series([None] * len(work_df), index=work_df.index, dtype="object")
    if index_color_groups is not None:
        def _position_color(value: Any) -> Optional[str]:
            if pd.isna(value):
                return other_index_color
            return index_color_map.get(int(value), other_index_color)

        eligible_index_color_mask = pd.Series(True, index=work_df.index)
        if index_color_class_set is not None:
            eligible_index_color_mask = work_df["plot_class"].isin(index_color_class_set)

        work_df.loc[eligible_index_color_mask, "index_color"] = work_df.loc[
            eligible_index_color_mask, "mutation_position"
        ].map(_position_color)
        work_df["index_color_label"] = work_df["index_color"].map(index_color_to_label)

    explicit_index_colors = set(index_color_map.values())
    cluster_highlight_colors: set[str] = set()
    if highlight_clusters is True:
        cluster_highlight_colors = set(explicit_index_colors)
    elif highlight_clusters is not False and highlight_clusters is not None:
        cluster_highlight_colors = {
            str(color).strip()
            for color in _list_like(highlight_clusters)
            if str(color).strip()
        }
        unknown_cluster_colors = cluster_highlight_colors.difference(explicit_index_colors)
        if unknown_cluster_colors:
            raise ValueError(
                "highlight_clusters contains colors that are not explicit cluster colors: "
                f"{sorted(unknown_cluster_colors)}. Available explicit cluster colors: "
                f"{sorted(explicit_index_colors)}"
            )

    work_df["is_cluster_highlight"] = False
    work_df["cluster_highlight_color"] = pd.Series([None] * len(work_df), index=work_df.index, dtype="object")
    if cluster_highlight_colors:
        cluster_highlight_mask = work_df["index_color"].isin(cluster_highlight_colors)
        work_df.loc[cluster_highlight_mask, "is_cluster_highlight"] = True
        work_df.loc[cluster_highlight_mask, "cluster_highlight_color"] = work_df.loc[
            cluster_highlight_mask, "index_color"
        ]

    def _short_class_label(value: Any) -> str:
        text = str(value).strip()
        if text == "Driver":
            return "D"
        if text == "Passenger":
            return "P"
        if text == "Hotspot":
            return "H"
        if text == "Non-Hotspot":
            return "NH"
        if text == "Unknown" or not text:
            return "WT/UNK"
        return text

    # class_palette = {
    #     "Driver": "#2E86AB",
    #     "Passenger": "#D1495B",
    #     "Hotspot": "#2E86AB",
    #     "Non-Hotspot": "#D1495B",
    #     "Unknown": "#6C757D",
    # }

    class_palette = {
        "Driver": "#D1495B",
        "Passenger": "#2E86AB",
        "Hotspot": "#D1495B",
        "Non-Hotspot": "#2E86AB",
        "Unknown": "#6C757D",
    }

    if colors:
        class_palette.update(dict(colors))

    cluster_label_offsets = list(
        cluster_highlight_label_offsets
        or [
            (6, 8),
            (8, 22),
            (12, -12),
            (18, 34),
            (20, -26),
            (30, 12),
            (34, 28),
            (38, -18),
        ]
    )
    if not cluster_label_offsets:
        cluster_label_offsets = [(6, 8)]

    def _highlight_arrow_color(label_color: Any) -> str:
        label_color_text = str(label_color)
        if highlight_arrow_color is None:
            return label_color_text

        arrow_color_text = str(highlight_arrow_color).strip()
        if not arrow_color_text:
            return label_color_text

        if arrow_color_text.lower() != "darker":
            try:
                return mcolors.to_hex(mcolors.to_rgb(arrow_color_text))
            except ValueError:
                return label_color_text

        try:
            darken_factor = float(highlight_arrow_darken_factor)
        except (TypeError, ValueError):
            darken_factor = 0.55
        darken_factor = min(max(darken_factor, 0.0), 1.0)

        try:
            rgb = mcolors.to_rgb(label_color_text)
        except ValueError:
            return label_color_text
        darker_rgb = tuple(channel * darken_factor for channel in rgb)
        return mcolors.to_hex(darker_rgb)

    def _dedupe_label_offsets(offsets: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        deduped: List[Tuple[float, float]] = []
        seen: set[Tuple[float, float]] = set()
        for x_offset, y_offset in offsets:
            key = (round(float(x_offset), 3), round(float(y_offset), 3))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((float(x_offset), float(y_offset)))
        return deduped

    def _expanded_label_offsets(seed_offsets: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        offsets = _dedupe_label_offsets(seed_offsets)
        angles = [45, 75, 20, 115, -35, -75, 155, -155, 0, 90, 180, -90]
        for radius in [12, 18, 26, 36, 50, 68, 90, 118, 152, 194, 244, 304]:
            for angle in angles:
                angle_rad = math.radians(angle)
                offsets.append((radius * math.cos(angle_rad), radius * math.sin(angle_rad)))
        return _dedupe_label_offsets(offsets)

    label_offset_candidates = _expanded_label_offsets(cluster_label_offsets)
    highlight_label_annotations_by_ax: Dict[Any, List[Any]] = {}

    def _bbox_overlap_area(bbox_a: Any, bbox_b: Any) -> float:
        x_overlap = max(0.0, min(bbox_a.x1, bbox_b.x1) - max(bbox_a.x0, bbox_b.x0))
        y_overlap = max(0.0, min(bbox_a.y1, bbox_b.y1) - max(bbox_a.y0, bbox_b.y0))
        return float(x_overlap * y_overlap)

    def _annotation_label_bbox(annotation: Any, renderer: Any) -> Any:
        bbox_patch = annotation.get_bbox_patch()
        if bbox_patch is not None:
            return bbox_patch.get_window_extent(renderer=renderer).padded(1.5)
        return annotation.get_window_extent(renderer=renderer).padded(1.5)

    def _resolve_label_overlaps(
        fig: Any,
        annotations_by_ax: Mapping[Any, Sequence[Any]],
        offset_candidates: Sequence[Tuple[float, float]],
    ) -> None:
        if not annotations_by_ax:
            return

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        px_per_pt = fig.dpi / 72.0

        annotations: List[Any] = []
        for ax_annotations in annotations_by_ax.values():
            annotations.extend(ax_annotations)

        placed_bboxes: List[Any] = []
        for annotation in annotations:
            base_offset = annotation.get_position()
            base_bbox = _annotation_label_bbox(annotation, renderer)
            candidates = _dedupe_label_offsets(
                [(float(base_offset[0]), float(base_offset[1])), *offset_candidates]
            )

            best_offset = candidates[0]
            best_bbox = base_bbox
            best_overlap = float("inf")

            for candidate_offset in candidates:
                dx_px = (candidate_offset[0] - float(base_offset[0])) * px_per_pt
                dy_px = (candidate_offset[1] - float(base_offset[1])) * px_per_pt
                candidate_bbox = base_bbox.translated(dx_px, dy_px)
                overlap_area = sum(
                    _bbox_overlap_area(candidate_bbox, placed_bbox)
                    for placed_bbox in placed_bboxes
                )
                if overlap_area <= 0:
                    best_offset = candidate_offset
                    best_bbox = candidate_bbox
                    best_overlap = 0.0
                    break
                if overlap_area < best_overlap:
                    best_offset = candidate_offset
                    best_bbox = candidate_bbox
                    best_overlap = overlap_area

            annotation.set_position((float(best_offset[0]), float(best_offset[1])))
            placed_bboxes.append(best_bbox)

    n_panels = len(selected_metrics)
    ncols = max(1, min(int(ncols), n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        constrained_layout=True,
    )
    flat_axes = np.asarray(axes).reshape(-1)

    rank_tables: Dict[str, pd.DataFrame] = {}
    highlight_tables: List[pd.DataFrame] = []
    cluster_highlight_tables: List[pd.DataFrame] = []

    for ax, metric in zip(flat_axes, selected_metrics):
        metric_df = work_df[
            [
                artifact_col,
                "mutation_code",
                "mutation_position",
                "plot_class",
                "is_highlight",
                "highlight_color",
                "highlight_source",
                "index_color",
                "index_color_label",
                "is_cluster_highlight",
                "cluster_highlight_color",
                metric,
            ]
        ].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
        metric_df = metric_df.dropna(subset=[metric]).copy()

        if metric_df.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{metric}\nNo finite values", ha="center", va="center")
            rank_tables[metric] = metric_df
            continue

        metric_df = metric_df.sort_values(metric, ascending=rank_ascending).reset_index(drop=True)
        metric_df["rank"] = np.arange(1, len(metric_df) + 1)
        metric_df["rank_pct"] = metric_df["rank"] / float(len(metric_df)) * 100.0
        if rank_highlight_colors:
            rank_color_series = metric_df["rank"].map(rank_highlight_colors)
            rank_color_mask = rank_color_series.notna()
            metric_df.loc[rank_color_mask, "is_highlight"] = True
            _append_highlight_source(metric_df, rank_color_mask, "rank")
            missing_highlight_color = rank_color_mask & metric_df["highlight_color"].isna()
            metric_df.loc[missing_highlight_color, "highlight_color"] = rank_color_series.loc[
                missing_highlight_color
            ]
        if rank_highlights:
            rank_mask = metric_df["rank"].isin(rank_highlights)
            metric_df.loc[rank_mask, "is_highlight"] = True
            _append_highlight_source(metric_df, rank_mask, "rank")
            metric_df.loc[rank_mask & metric_df["highlight_color"].isna(), "highlight_color"] = "black"

        if show_line and len(metric_df) >= 2:
            ax.plot(
                metric_df["rank"],
                metric_df[metric],
                color="#B0B0B0",
                linewidth=1.0,
                alpha=0.85,
                zorder=1,
            )

        class_draw_order = ["Passenger", "Non-Hotspot", "Unknown", "Hotspot", "Driver"]
        metric_df["_point_color"] = metric_df["plot_class"].map(class_palette).fillna("#6C757D")
        metric_df["_point_label"] = metric_df["plot_class"].astype(str)
        index_color_mask = metric_df["index_color"].notna()
        metric_df.loc[index_color_mask, "_point_color"] = metric_df.loc[index_color_mask, "index_color"]
        metric_df.loc[index_color_mask, "_point_label"] = (
            metric_df.loc[index_color_mask, "index_color_label"].fillna("Index group")
        )

        point_labels_present = list(pd.unique(metric_df["_point_label"]))
        ordered_point_labels: List[str] = []
        ordered_point_labels.extend(
            [label for label in class_draw_order if label in point_labels_present and label not in ordered_point_labels]
        )
        if index_color_groups is not None:
            ordered_point_labels.extend(
                [
                    label
                    for label in index_color_label_order
                    if label == "Other index" and label in point_labels_present
                ]
            )
            ordered_point_labels.extend(
                [
                    label
                    for label in index_color_label_order
                    if label != "Other index" and label in point_labels_present
                ]
            )
        ordered_point_labels.extend(
            [label for label in point_labels_present if label not in ordered_point_labels]
        )

        for point_label in ordered_point_labels:
            point_rows = metric_df[metric_df["_point_label"] == point_label]
            if point_rows.empty:
                continue
            ax.scatter(
                point_rows["rank"],
                point_rows[metric],
                s=point_size,
                alpha=point_alpha,
                color=str(point_rows["_point_color"].iloc[0]),
                edgecolors="none",
                zorder=4 if point_label in set(index_color_label_order).difference({"Other index"}) else 2,
            )

        cluster_highlight_rows = metric_df[metric_df["is_cluster_highlight"]].copy()
        if not cluster_highlight_rows.empty:
            cluster_highlight_rows["cluster_highlight_color"] = cluster_highlight_rows[
                "cluster_highlight_color"
            ].fillna(cluster_highlight_rows["index_color"])
            out_cluster_highlight = cluster_highlight_rows[
                [
                    artifact_col,
                    "mutation_code",
                    "mutation_position",
                    "plot_class",
                    metric,
                    "rank",
                    "rank_pct",
                    "cluster_highlight_color",
                    "index_color",
                    "index_color_label",
                ]
            ].copy()
            out_cluster_highlight = out_cluster_highlight.rename(
                columns={metric: "metric_value", "plot_class": "class"}
            )
            out_cluster_highlight.insert(0, "metric", metric)
            cluster_highlight_tables.append(out_cluster_highlight)

        highlight_rows = metric_df[metric_df["is_highlight"]].copy()
        if not highlight_rows.empty:
            highlight_rows["highlight_color"] = highlight_rows["highlight_color"].fillna("black")
            out_highlight = highlight_rows[
                [
                    artifact_col,
                    "mutation_code",
                    "mutation_position",
                    "plot_class",
                    metric,
                    "rank",
                    "rank_pct",
                    "highlight_source",
                    "highlight_color",
                    "index_color",
                    "index_color_label",
                ]
            ].copy()
            out_highlight = out_highlight.rename(columns={metric: "metric_value", "plot_class": "class"})
            out_highlight.insert(0, "metric", metric)
            highlight_tables.append(out_highlight)

        combined_highlight_rows = metric_df[
            metric_df["is_highlight"] | metric_df["is_cluster_highlight"]
        ].copy()
        if not combined_highlight_rows.empty:
            combined_highlight_rows["combined_highlight_color"] = combined_highlight_rows["highlight_color"]
            missing_color = (
                combined_highlight_rows["combined_highlight_color"].isna()
                | combined_highlight_rows["combined_highlight_color"].astype(str).str.strip().eq("")
            )
            combined_highlight_rows.loc[missing_color, "combined_highlight_color"] = combined_highlight_rows.loc[
                missing_color, "cluster_highlight_color"
            ]
            missing_color = (
                combined_highlight_rows["combined_highlight_color"].isna()
                | combined_highlight_rows["combined_highlight_color"].astype(str).str.strip().eq("")
            )
            combined_highlight_rows.loc[missing_color, "combined_highlight_color"] = "black"
            combined_highlight_rows["combined_highlight_linewidth"] = np.where(
                combined_highlight_rows["is_cluster_highlight"],
                2.0,
                1.6,
            )
            combined_highlight_rows["combined_highlight_size"] = np.where(
                combined_highlight_rows["is_cluster_highlight"],
                max(highlight_size * 0.72, point_size * 1.8),
                highlight_size,
            )

            ax.scatter(
                combined_highlight_rows["rank"],
                combined_highlight_rows[metric],
                s=combined_highlight_rows["combined_highlight_size"].tolist(),
                facecolors="none",
                edgecolors=combined_highlight_rows["combined_highlight_color"].tolist(),
                linewidths=combined_highlight_rows["combined_highlight_linewidth"].tolist(),
                zorder=4.7,
            )

            label_mask = combined_highlight_rows["is_highlight"] | (
                combined_highlight_rows["is_cluster_highlight"] & bool(cluster_highlight_label)
            )
            label_rows = combined_highlight_rows[label_mask].copy()
            for label_idx, (_, row) in enumerate(label_rows.iterrows()):
                label = row["mutation_code"] if row["mutation_code"] else str(row[artifact_col])
                label_color = row.get("combined_highlight_color")
                if pd.isna(label_color) or not str(label_color).strip():
                    label_color = "#333333"
                arrow_color = _highlight_arrow_color(label_color)
                x_offset, y_offset = cluster_label_offsets[label_idx % len(cluster_label_offsets)]
                if bool(row.get("is_highlight", False)):
                    class_label = _short_class_label(row.get("plot_class", "Unknown"))
                    label_text = f"{label} {class_label} (#{int(row['rank'])})"
                    label_size = 9
                    label_weight = "bold"
                else:
                    label_text = f"{label} #{int(row['rank'])}"
                    label_size = cluster_highlight_label_size
                    label_weight = "semibold"

                annotation = ax.annotate(
                    label_text,
                    (float(row["rank"]), float(row[metric])),
                    xytext=(float(x_offset), float(y_offset)),
                    textcoords="offset points",
                    fontsize=label_size,
                    color=str(label_color),
                    fontweight=label_weight,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": str(label_color),
                        "linewidth": 0.8,
                        "alpha": 0.88,
                    },
                    arrowprops={
                        "arrowstyle": "->",
                        "color": arrow_color,
                        "linewidth": 0.8,
                        "alpha": 0.85,
                        "shrinkA": 2,
                        "shrinkB": 3,
                    },
                    zorder=5,
                )
                highlight_label_annotations_by_ax.setdefault(ax, []).append(annotation)

        rank_out = metric_df[
            [
                artifact_col,
                "mutation_code",
                "mutation_position",
                "plot_class",
                metric,
                "rank",
                "rank_pct",
                "is_highlight",
                "highlight_source",
                "highlight_color",
                "index_color",
                "index_color_label",
                "is_cluster_highlight",
                "cluster_highlight_color",
            ]
        ].copy()
        rank_out = rank_out.rename(columns={metric: "metric_value", "plot_class": "class"})
        rank_tables[metric] = rank_out

        rank_direction = "lowest" if rank_ascending else "highest"
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel(f"Rank (1 = {rank_direction})")
        ax.set_ylabel("Metric value")
        if len(metric_df) > 1:
            ax.set_xlim(1, len(metric_df))
        ax.grid(axis="y", alpha=0.25)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    from matplotlib.lines import Line2D

    preferred_order = ["Passenger", "Driver", "Non-Hotspot", "Hotspot", "Unknown"]
    point_color_series = work_df["plot_class"].map(class_palette).fillna("#6C757D").astype("object")
    point_label_series = work_df["plot_class"].astype(str).astype("object")
    index_color_mask = work_df["index_color"].notna()
    point_color_series.loc[index_color_mask] = work_df.loc[index_color_mask, "index_color"]
    point_label_series.loc[index_color_mask] = work_df.loc[index_color_mask, "index_color_label"].fillna(
        "Index group"
    )

    present_point_labels = list(pd.unique(point_label_series))
    ordered_point_labels = []
    ordered_point_labels.extend(
        [label for label in preferred_order if label in present_point_labels and label not in ordered_point_labels]
    )
    if index_color_groups is not None:
        ordered_point_labels.extend(
            [
                label
                for label in index_color_label_order
                if label == "Other index" and label in present_point_labels
            ]
        )
        ordered_point_labels.extend(
            [
                label
                for label in index_color_label_order
                if label != "Other index" and label in present_point_labels
            ]
        )
    ordered_point_labels.extend(
        [label for label in sorted(present_point_labels) if label not in ordered_point_labels]
    )

    label_to_color: Dict[str, str] = {}
    for label, color in zip(point_label_series, point_color_series):
        label_to_color.setdefault(str(label), str(color))

    index_label_set = set(index_color_label_order)
    legend_handles = []
    for label in ordered_point_labels:
        display_label = str(label)
        if display_label not in index_label_set:
            display_label = f"{display_label} ({_short_class_label(display_label)})"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=label_to_color.get(str(label), "#6C757D"),
                markeredgecolor="none",
                alpha=0.8,
                label=display_label,
            )
        )

    if highlight_items or rank_highlights:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=1.4,
                label="Highlight",
            )
        )

    item_highlight_colors = [
        str(color)
        for color in pd.unique(work_df.loc[colored_item_mask_seen, "highlight_color"])
        if pd.notna(color) and str(color) != "black"
    ]
    for color_text in item_highlight_colors:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor=color_text,
                markeredgewidth=1.4,
                label="Artifact/mutation highlight",
            )
        )

    for color_text in dict.fromkeys(rank_highlight_colors.values()):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor=color_text,
                markeredgewidth=1.4,
                label="Rank highlight",
            )
        )

    cluster_highlight_label_by_color = {
        color: index_color_to_label.get(color, "Cluster")
        for color in cluster_highlight_colors
    }
    for color_text, label in sorted(cluster_highlight_label_by_color.items(), key=lambda item: item[1]):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor=color_text,
                markeredgewidth=1.8,
                label=f"Cluster highlight: {label}",
            )
        )

    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.suptitle(title or "Rank plot by metric")
    fig.tight_layout(rect=(0.0, 0.0, 0.96, 0.97))
    _resolve_label_overlaps(fig, highlight_label_annotations_by_ax, label_offset_candidates)

    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")

    highlights_df = (
        pd.concat(highlight_tables, ignore_index=True)
        if highlight_tables
        else pd.DataFrame(
            columns=[
                "metric",
                artifact_col,
                "mutation_code",
                "mutation_position",
                "class",
                "metric_value",
                "rank",
                "rank_pct",
                "highlight_source",
                "highlight_color",
                "index_color",
                "index_color_label",
            ]
        )
    )
    cluster_highlights_df = (
        pd.concat(cluster_highlight_tables, ignore_index=True)
        if cluster_highlight_tables
        else pd.DataFrame(
            columns=[
                "metric",
                artifact_col,
                "mutation_code",
                "mutation_position",
                "class",
                "metric_value",
                "rank",
                "rank_pct",
                "cluster_highlight_color",
                "index_color",
                "index_color_label",
            ]
        )
    )

    return {
        "figure": fig,
        "axes": axes,
        "metric_cols": selected_metrics,
        "plot_df": work_df,
        "rank_tables": rank_tables,
        "highlights_df": highlights_df,
        "cluster_highlights_df": cluster_highlights_df,
    }


def with_binary_label(
    df: pd.DataFrame,
    *,
    mutation_type_col: str = "mutation_type",
    positive_class: str = "Driver",
    label_col: str = "label",
) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = (out[mutation_type_col].astype(str) == positive_class).astype(int)
    return out


def plot_d1_distribution_by_class(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    out_path: Path | str,
    value_col: str = "d1_mean",
    class_col: str = "mutation_type",
    bins: int = 24,
    log_dynamic_range: float = 100.0,
) -> Dict[str, Any]:
    """Plot the per-variant mean D1 distribution for Driver and Passenger classes.

    Both classes use the same bin edges and density normalization, making the
    overlaid shapes comparable despite class imbalance. A logarithmic x-axis is
    selected only for a strictly positive distribution spanning at least
    ``log_dynamic_range``-fold (POLE is the current case); values are never
    transformed in the exported feature table.
    """
    missing = [col for col in (value_col, class_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing D1 distribution column(s): {missing}")

    plot_df = df[[class_col, value_col]].copy()
    plot_df[class_col] = plot_df[class_col].astype(str)
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df[
        plot_df[class_col].isin(["Driver", "Passenger"])
        & np.isfinite(plot_df[value_col])
    ].copy()
    counts = plot_df[class_col].value_counts()
    if not {"Driver", "Passenger"}.issubset(counts.index):
        raise ValueError(f"[{dataset_name}] D1 distribution requires both Driver and Passenger values")

    values = plot_df[value_col].to_numpy(dtype=float)
    positive = values[values > 0]
    use_log = bool(
        len(positive) == len(values)
        and positive.max() / positive.min() >= log_dynamic_range
    )
    bin_edges = np.geomspace(positive.min(), positive.max(), bins + 1) if use_log else bins

    palette = {"Passenger": "#2E86AB", "Driver": "#D1495B"}
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for class_name in ("Passenger", "Driver"):
        class_values = plot_df.loc[plot_df[class_col] == class_name, value_col]
        sns.histplot(
            class_values,
            bins=bin_edges,
            stat="density",
            common_norm=False,
            alpha=0.48,
            color=palette[class_name],
            edgecolor="black",
            linewidth=0.5,
            label=f"{class_name} (n={len(class_values)})",
            ax=ax,
        )
        if class_values.nunique() > 1:
            sns.kdeplot(class_values, color=palette[class_name], linewidth=1.8, ax=ax)

    if use_log:
        ax.set_xscale("log")
    ax.set_title(f"{dataset_name} D1 distribution - Driver vs Passenger")
    ax.set_xlabel("Mean D1 score per variant" + (" (log scale)" if use_log else ""))
    ax.set_ylabel("Density")
    ax.legend(title="Label class")
    ax.grid(axis="y", alpha=0.2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": out_path,
        "n_driver": int(counts["Driver"]),
        "n_passenger": int(counts["Passenger"]),
        "log_scale": use_log,
    }


def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std <= 0 or not np.isfinite(pooled_std):
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def separation_score(group1: Sequence[float], group2: Sequence[float]) -> float:
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    return float(abs(np.mean(g1) - np.mean(g2)) / (np.std(g1) + np.std(g2) + 1e-10))


# Numeric columns of the feature table that are bookkeeping, not structural features.
# The D1 residue-count fields exist to make a WT/mutant length mismatch visible (they
# are what `d1_length_mismatch` is derived from); ranking them as candidate features is
# meaningless at best. For KRAS it is actively misleading: `d1_mut_n_residues` varies
# only because 17 variants came from a 188-residue generation batch, so any apparent
# discrimination would be tracking the batch, not biology. They were being caught by the
# degenerate-feature guard and sorted last, but excluding them is the correct fix.
METADATA_FEATURE_COLS: tuple[str, ...] = (
    "position",
    "d1_wt_n_residues",
    "d1_mut_n_residues",
    "d1_length_mismatch",
)


def run_feature_discrimination_analysis(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    exclude_cols: Sequence[str] = METADATA_FEATURE_COLS,
    min_group_n: int = 2,
    min_distinct_values: int = 5,
    min_relative_std: float = 1e-4,
) -> pd.DataFrame:
    """Rank features by how well each separates the two label classes.

    Cohen's d divides by the pooled standard deviation, so a feature that is
    almost constant gets an enormous |d| from a collapsing denominator while
    having no real discriminative power. POLE's ``pct_very_low`` is the worst
    case: 3 distinct values across 1075 mutations (1066 of them identical),
    passenger std 1.2e-4, |d| = 2.74 -- top-ranked -- yet ROC AUC only 0.606,
    and a Mann-Whitney p of 1e-44 that comes from ties rather than separation.
    Ranked first, it took a slot in the 10-feature LDA and displaced a genuinely
    informative feature.

    Such features are therefore flagged ``degenerate`` and sorted to the bottom,
    so top-N selection skips them. They are still returned (never silently
    dropped) so the diagnosis stays visible. ``min_distinct_values`` and
    ``min_relative_std`` control the test; set both to 0 for the old behaviour.
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing label column '{label_col}'")

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [f for f in numeric_features if f != label_col and f not in exclude_cols]

    rows: List[dict] = []
    for feature in features:
        sub = df[[feature, label_col]].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue

        y = sub[label_col].astype(int).to_numpy()
        if np.unique(y).size < 2:
            continue

        vals = sub[feature].to_numpy(dtype=float)
        pos_vals = vals[y == 1]
        neg_vals = vals[y == 0]

        if len(pos_vals) < min_group_n or len(neg_vals) < min_group_n:
            continue

        t_stat, p_val_t = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)
        _, p_val_mw = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")

        mean_pos = float(np.mean(pos_vals))
        mean_neg = float(np.mean(neg_vals))

        score = vals if mean_pos >= mean_neg else -vals
        try:
            roc_auc = float(roc_auc_score(y, score))
        except ValueError:
            roc_auc = 0.5

        # Near-constant features produce a meaningless |Cohen's d| (see docstring).
        n_distinct = int(np.unique(vals).size)
        scale = float(np.mean(np.abs(vals)))
        relative_std = float(np.std(vals)) / scale if scale > 0 else 0.0
        degenerate = bool(n_distinct < min_distinct_values or relative_std < min_relative_std)

        rows.append(
            {
                "feature": feature,
                "driver_mean": mean_pos,
                "driver_std": float(np.std(pos_vals)),
                "passenger_mean": mean_neg,
                "passenger_std": float(np.std(neg_vals)),
                "mean_diff": mean_pos - mean_neg,
                "cohens_d": cohens_d(pos_vals, neg_vals),
                "p_value_ttest": float(p_val_t),
                "p_value_mann_whitney": float(p_val_mw),
                "separation_score": separation_score(pos_vals, neg_vals),
                "roc_auc": roc_auc,
                "n_distinct_values": n_distinct,
                "relative_std": relative_std,
                "degenerate": degenerate,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["abs_cohens_d"] = np.abs(out["cohens_d"])
    out["significant"] = out["p_value_ttest"] < 0.05
    # degenerate features sort last, so top-N selection never picks them up
    return (
        out.sort_values(["degenerate", "abs_cohens_d"], ascending=[True, False])
        .reset_index(drop=True)
    )


def plot_feature_distributions(
    df: pd.DataFrame,
    discrimination_df: pd.DataFrame,
    *,
    label_col: str = "label",
    top_n: int = 12,
    include_features: Optional[Sequence[str]] = None,
    class_names: Tuple[str, str] = ("Passenger", "Driver"),
    colors: Optional[Mapping[str, str]] = None,
    out_path: Path | str | None = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    if discrimination_df.empty:
        raise ValueError("discrimination_df is empty")

    colors = dict(colors or {"Driver": "#E74C3C", "Passenger": "#3498DB"})
    base_features = discrimination_df["feature"].head(top_n).tolist()
    if include_features:
        for feat in include_features:
            if feat not in base_features:
                base_features.append(feat)

    n_plots = len(base_features)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(5, n_rows * 4.2)))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for idx, feature in enumerate(base_features):
        ax = axes_arr[idx]
        sub = df[[feature, label_col]].copy().replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            ax.set_visible(False)
            continue

        driver_vals = sub[sub[label_col] == 1][feature].to_numpy(dtype=float)
        passenger_vals = sub[sub[label_col] == 0][feature].to_numpy(dtype=float)

        if len(driver_vals) == 0 or len(passenger_vals) == 0:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(
            [driver_vals, passenger_vals],
            positions=[0, 1],
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for i, pc in enumerate(parts["bodies"]):
            color = colors["Driver"] if i == 0 else colors["Passenger"]
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor("black")
            pc.set_linewidth(1.2)

        ax.boxplot(
            [driver_vals, passenger_vals],
            positions=[0, 1],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.2, alpha=0.8),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
        )

        for i, vals in enumerate([driver_vals, passenger_vals]):
            jitter = i + np.random.normal(0, 0.04, size=len(vals))
            color = colors["Driver"] if i == 0 else colors["Passenger"]
            ax.scatter(jitter, vals, s=35, alpha=0.65, color=color, edgecolors="black", linewidth=0.4, zorder=3)

        stat_row = discrimination_df[discrimination_df["feature"] == feature]
        if not stat_row.empty:
            row = stat_row.iloc[0]
            stat_text = f"d={row['cohens_d']:.2f} | AUC={row['roc_auc']:.3f}"
            ax.text(
                0.98,
                0.96,
                stat_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels([class_names[1], class_names[0]])
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.25)

    for idx in range(n_plots, len(axes_arr)):
        axes_arr[idx].set_visible(False)

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, list(axes_arr[:n_plots])


def plot_feature_roc_curves(
    df: pd.DataFrame,
    discrimination_df: pd.DataFrame,
    *,
    label_col: str = "label",
    top_n: int = 8,
    out_path: Path | str | None = None,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    if discrimination_df.empty:
        raise ValueError("discrimination_df is empty")

    top = discrimination_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    rows: List[dict] = []

    for _, row in top.iterrows():
        feature = row["feature"]
        sub = df[[feature, label_col]].copy().replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue

        y = sub[label_col].astype(int).to_numpy()
        if np.unique(y).size < 2:
            continue

        scores = sub[feature].to_numpy(dtype=float)
        if scores[y == 1].mean() < scores[y == 0].mean():
            scores = -scores

        fpr, tpr, _ = roc_curve(y, scores)
        auc_val = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, lw=2, label=f"{feature} (AUC={auc_val:.3f})")
        rows.append({"feature": feature, "roc_auc": auc_val})

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for Top Features")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    auc_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return auc_df, fig, ax


def run_lda_analysis(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    features: Optional[Sequence[str]] = None,
    discrimination_df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column '{label_col}'")

    if features is None:
        if discrimination_df is None or discrimination_df.empty:
            raise ValueError("Provide either explicit features or a non-empty discrimination_df")
        features = discrimination_df.head(top_n)["feature"].tolist()

    model_df = df[list(features) + [label_col]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    if model_df.empty:
        raise ValueError("No rows left after dropping NaN/inf values for LDA")

    y = model_df[label_col].astype(int).to_numpy()
    if np.unique(y).size < 2:
        raise ValueError("LDA requires two classes in label column")

    X = model_df[list(features)].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis()
    lda_scores = lda.fit_transform(X_scaled, y).ravel()

    if roc_auc_score(y, lda_scores) < 0.5:
        lda_scores = -lda_scores
        coef_sign = -1.0
    else:
        coef_sign = 1.0

    fpr, tpr, _ = roc_curve(y, lda_scores)
    roc_auc_val = float(auc(fpr, tpr))

    driver_scores = lda_scores[y == 1]
    passenger_scores = lda_scores[y == 0]
    t_stat, p_val = stats.ttest_ind(driver_scores, passenger_scores, equal_var=False)

    coef = lda.coef_[0] * coef_sign
    coef_df = pd.DataFrame(
        {
            "feature": list(features),
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False)

    out_df = model_df.copy()
    out_df["lda_score"] = lda_scores

    return {
        "df_scores": out_df,
        "features": list(features),
        "scaler": scaler,
        "lda": lda,
        "coef_df": coef_df,
        "roc_auc": roc_auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "driver_scores": driver_scores,
        "passenger_scores": passenger_scores,
        "cohens_d": cohens_d(driver_scores, passenger_scores),
        "p_value_ttest": float(p_val),
        "t_stat": float(t_stat),
    }


def run_lda_analysis_cv(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    features: Optional[Sequence[str]] = None,
    discrimination_df: Optional[pd.DataFrame] = None,
    exclude_cols: Sequence[str] = METADATA_FEATURE_COLS,
    top_n: int = 10,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Cross-validated counterpart to run_lda_analysis.

    Both the feature ranking (Cohen's d) and the LDA fit are redone inside
    each training fold, so roc_auc_cv / cohens_d here are out-of-fold
    estimates rather than the in-sample numbers run_lda_analysis reports.
    A final LDA is also fit on the full dataset (same as run_lda_analysis)
    solely to report interpretable coefficients -- that fit contributes no
    performance number in the returned dict.

    The candidate feature pool (re-ranked independently inside every fold)
    defaults to `features` if given, else every feature already present in
    `discrimination_df`, else every numeric column of `df`. Preferring
    discrimination_df keeps the pool restricted to columns the notebook
    already treats as modeling features -- scanning all numeric columns of
    df directly can pull in sparsely-populated metrics (e.g. interface- or
    structure-only columns) whose joint dropna wipes out every row.
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing label column '{label_col}'")

    if features is not None:
        candidate_features = list(features)
    elif discrimination_df is not None and not discrimination_df.empty:
        candidate_features = discrimination_df["feature"].tolist()
    else:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [f for f in numeric_features if f != label_col and f not in exclude_cols]
    if not candidate_features:
        raise ValueError("No candidate features found for cross-validated LDA")

    model_df = df[candidate_features + [label_col]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    if model_df.empty:
        raise ValueError("No rows left after dropping NaN/inf values for LDA")

    y = model_df[label_col].astype(int).to_numpy()
    class_counts = np.bincount(y)
    if class_counts.size < 2 or class_counts.min() < 2:
        raise ValueError("Cross-validated LDA requires at least two samples per class")

    effective_n_splits = max(2, int(min(n_splits, class_counts.min())))
    X_full = model_df[candidate_features]
    n_rows = len(model_df)

    cv = RepeatedStratifiedKFold(n_splits=effective_n_splits, n_repeats=n_repeats, random_state=random_state)

    repeat_scores = np.full((n_repeats, n_rows), np.nan)
    fold_feature_counter: Counter = Counter()
    n_folds_run = 0

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X_full, y)):
        repeat_idx = split_idx // effective_n_splits

        train_fold_df = X_full.iloc[train_idx].copy()
        train_fold_df[label_col] = y[train_idx]
        fold_discrimination = run_feature_discrimination_analysis(
            train_fold_df, label_col=label_col, exclude_cols=exclude_cols
        )
        if fold_discrimination.empty:
            continue

        fold_features = fold_discrimination.head(top_n)["feature"].tolist()
        fold_feature_counter.update(fold_features)

        X_train = X_full.iloc[train_idx][fold_features].to_numpy(dtype=float)
        X_test = X_full.iloc[test_idx][fold_features].to_numpy(dtype=float)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lda_fold = LinearDiscriminantAnalysis()
        lda_fold.fit(X_train_scaled, y[train_idx])
        repeat_scores[repeat_idx, test_idx] = lda_fold.decision_function(X_test_scaled)
        n_folds_run += 1

    if n_folds_run == 0:
        raise ValueError("No CV fold produced usable features; check the candidate feature set")

    repeat_aucs: List[float] = []
    for r in range(n_repeats):
        scores_r = repeat_scores[r]
        if np.any(np.isnan(scores_r)):
            continue
        repeat_aucs.append(float(roc_auc_score(y, scores_r)))

    if not repeat_aucs:
        raise ValueError("Every CV repeat left at least one row unscored out-of-fold")

    rep0_scores = repeat_scores[0]
    fpr, tpr, _ = roc_curve(y, rep0_scores)
    roc_auc_rep0 = float(auc(fpr, tpr))

    df_scores = model_df.copy()
    df_scores["lda_score"] = rep0_scores

    driver_scores_cv = rep0_scores[y == 1]
    passenger_scores_cv = rep0_scores[y == 0]
    t_stat_cv, p_val_cv = stats.ttest_ind(driver_scores_cv, passenger_scores_cv, equal_var=False)

    feature_stability = (
        pd.DataFrame(
            {
                "feature": list(fold_feature_counter.keys()),
                "n_folds_selected": list(fold_feature_counter.values()),
            }
        )
        .assign(selection_rate=lambda d: d["n_folds_selected"] / n_folds_run)
        .sort_values("selection_rate", ascending=False)
        .reset_index(drop=True)
    )

    # Full-data fit, kept ONLY for interpretable coefficients -- not a performance estimate.
    final_fit = run_lda_analysis(df, label_col=label_col, discrimination_df=discrimination_df, top_n=top_n)

    return {
        "roc_auc": roc_auc_rep0,
        "roc_auc_cv": float(np.mean(repeat_aucs)),
        "roc_auc_cv_std": float(np.std(repeat_aucs)) if len(repeat_aucs) > 1 else 0.0,
        "roc_auc_cv_per_repeat": repeat_aucs,
        "roc_auc_in_sample": final_fit["roc_auc"],
        "n_splits_requested": int(n_splits),
        "n_splits_used": int(effective_n_splits),
        "n_repeats": int(n_repeats),
        "n_folds_run": int(n_folds_run),
        "feature_stability": feature_stability,
        "cv_scores": rep0_scores,
        "df_scores": df_scores,
        "fpr": fpr,
        "tpr": tpr,
        "driver_scores": driver_scores_cv,
        "passenger_scores": passenger_scores_cv,
        "cohens_d": cohens_d(driver_scores_cv, passenger_scores_cv),
        "cohens_d_in_sample": final_fit["cohens_d"],
        "p_value_ttest": float(p_val_cv),
        "t_stat": float(t_stat_cv),
        "coef_df": final_fit["coef_df"],
        "features": final_fit["features"],
        "scaler": final_fit["scaler"],
        "lda": final_fit["lda"],
    }


def plot_lda_panels(
    lda_result: Mapping[str, Any],
    *,
    title: str = "Linear Discriminant Analysis",
    class_names: Tuple[str, str] = ("Passenger", "Driver"),
    out_path: Path | str | None = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    coef_df = pd.DataFrame(lda_result["coef_df"])
    driver_scores = np.asarray(lda_result["driver_scores"], dtype=float)
    passenger_scores = np.asarray(lda_result["passenger_scores"], dtype=float)
    fpr = np.asarray(lda_result["fpr"], dtype=float)
    tpr = np.asarray(lda_result["tpr"], dtype=float)
    roc_auc_val = float(lda_result["roc_auc"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    parts = ax1.violinplot(
        [driver_scores, passenger_scores],
        positions=[0, 1],
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#E74C3C" if i == 0 else "#3498DB")
        pc.set_alpha(0.3)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.2)

    ax1.boxplot(
        [driver_scores, passenger_scores],
        positions=[0, 1],
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.2, alpha=0.8),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
    )

    for i, vals in enumerate([driver_scores, passenger_scores]):
        jitter = i + np.random.normal(0, 0.04, size=len(vals))
        color = "#E74C3C" if i == 0 else "#3498DB"
        ax1.scatter(jitter, vals, s=35, alpha=0.65, color=color, edgecolors="black", linewidth=0.4, zorder=3)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([class_names[1], class_names[0]])
    ax1.set_ylabel("LDA score")
    ax1.set_title(f"LDA Distribution\nd={lda_result['cohens_d']:.3f}, p={lda_result['p_value_ttest']:.2e}")
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(fpr, tpr, color="#E74C3C", lw=2.5, label=f"LDA (AUC={roc_auc_val:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.25)

    coef_sorted = coef_df.sort_values("coefficient", ascending=True)
    colors = ["#3498DB" if v < 0 else "#E74C3C" for v in coef_sorted["coefficient"]]
    ax3.barh(range(len(coef_sorted)), coef_sorted["coefficient"], color=colors, edgecolor="black", linewidth=0.8)
    ax3.set_yticks(range(len(coef_sorted)))
    ax3.set_yticklabels(coef_sorted["feature"])
    ax3.axvline(0, color="black", lw=1.2)
    ax3.set_xlabel("LDA coefficient")
    ax3.set_title("Feature Importance")
    ax3.grid(axis="x", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, axes


def lda_assumption_tests(driver_scores: Sequence[float], passenger_scores: Sequence[float]) -> Dict[str, Any]:
    driver = np.asarray(driver_scores, dtype=float)
    passenger = np.asarray(passenger_scores, dtype=float)
    driver = driver[np.isfinite(driver)]
    passenger = passenger[np.isfinite(passenger)]

    out: Dict[str, Any] = {
        "n_driver": int(len(driver)),
        "n_passenger": int(len(passenger)),
    }

    # scipy emits "Precision loss ... catastrophic cancellation ... data are nearly
    # identical" from shapiro/levene whenever a score distribution is near-constant.
    # That is EXPECTED here: D1 deliberately magnifies small structural differences, so
    # many features sit in a very narrow band and the moment calculations lose
    # precision. It is a property of the input, not a fault to fix, and at hundreds of
    # repeats per run it drowned out warnings that do matter (e.g. the pLDDT
    # residue-count mismatches). Suppressed locally -- only around these two tests, so
    # any other RuntimeWarning still surfaces. The affected p-values are recorded as
    # usual but should be read as unreliable for near-constant features; see the
    # degenerate-feature flagging in run_feature_discrimination_analysis.
    _near_constant_msg = "Precision loss occurred in moment calculation"
    _precision_lost = False

    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always", RuntimeWarning)

        if len(driver) >= 3:
            w_d, p_d = shapiro(driver)
            out["shapiro_driver_W"] = float(w_d)
            out["shapiro_driver_p"] = float(p_d)
            out["driver_normal"] = bool(p_d > 0.05)
        else:
            out["shapiro_driver_W"] = np.nan
            out["shapiro_driver_p"] = np.nan
            out["driver_normal"] = None

        if len(passenger) >= 3:
            w_p, p_p = shapiro(passenger)
            out["shapiro_passenger_W"] = float(w_p)
            out["shapiro_passenger_p"] = float(p_p)
            out["passenger_normal"] = bool(p_p > 0.05)
        else:
            out["shapiro_passenger_W"] = np.nan
            out["shapiro_passenger_p"] = np.nan
            out["passenger_normal"] = None

        if len(driver) >= 2 and len(passenger) >= 2:
            stat_l, p_l = levene(driver, passenger)
            out["levene_F"] = float(stat_l)
            out["levene_p"] = float(p_l)
            out["equal_variance"] = bool(p_l > 0.05)
        else:
            out["levene_F"] = np.nan
            out["levene_p"] = np.nan
            out["equal_variance"] = None

    # Partition OUTSIDE the recorder. Re-raising inside it would be re-captured and
    # appended to the list being iterated -- an infinite loop.
    _passthrough = []
    for _w in _caught:
        if _near_constant_msg in str(_w.message):
            _precision_lost = True
        else:
            _passthrough.append(_w)

    # Anything that is NOT the expected near-constant precision loss still surfaces, so
    # this suppression can never hide a genuinely new warning.
    for _w in _passthrough:
        warnings.warn_explicit(_w.message, _w.category, _w.filename, _w.lineno)

    # Recorded, not printed: lets a reader see which rows had unreliable normality /
    # equal-variance p-values without scrolling past hundreds of identical warnings.
    out["precision_loss_near_constant"] = bool(_precision_lost)

    return out


def export_discrimination_bundle(
    output_dir: Path | str,
    *,
    dataset_prefix: str,
    discrimination_df: pd.DataFrame,
    mutation_df: Optional[pd.DataFrame] = None,
    lda_result: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}
    disc_path = out_dir / f"{dataset_prefix}_discrimination_analysis.csv"
    discrimination_df.to_csv(disc_path, index=False)
    written["discrimination_csv"] = str(disc_path)

    if mutation_df is not None:
        mut_path = out_dir / f"{dataset_prefix}_mutations_with_scores.csv"
        mutation_df.to_csv(mut_path, index=False)
        written["mutations_csv"] = str(mut_path)

    if lda_result is not None:
        coef_path = out_dir / f"{dataset_prefix}_lda_coefficients.csv"
        pd.DataFrame(lda_result["coef_df"]).to_csv(coef_path, index=False)
        written["lda_coefficients_csv"] = str(coef_path)

        summary_path = out_dir / f"{dataset_prefix}_discrimination_report.txt"
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("FEATURE DISCRIMINATION SUMMARY\n")
            handle.write("=" * 80 + "\n\n")
            if not discrimination_df.empty:
                handle.write("Top features by |Cohen's d|:\n")
                for _, row in discrimination_df.head(10).iterrows():
                    handle.write(
                        f"- {row['feature']}: d={row['cohens_d']:+.3f}, AUC={row['roc_auc']:.3f}, "
                        f"p={row['p_value_mann_whitney']:.3e}\n"
                    )
                handle.write("\n")

            handle.write("LDA Summary:\n")
            handle.write(f"- ROC AUC: {lda_result['roc_auc']:.3f}\n")
            if "roc_auc_cv" in lda_result:
                handle.write(
                    f"- ROC AUC (cross-validated, {lda_result.get('n_repeats', 1)}x"
                    f"{lda_result.get('n_splits_used', '?')}-fold): "
                    f"{lda_result['roc_auc_cv']:.3f} +/- {lda_result.get('roc_auc_cv_std', 0.0):.3f}\n"
                )
            if "roc_auc_in_sample" in lda_result:
                handle.write(
                    f"- ROC AUC (in-sample, full-data fit -- optimistic, for reference only): "
                    f"{lda_result['roc_auc_in_sample']:.3f}\n"
                )
            handle.write(f"- Cohen's d: {lda_result['cohens_d']:.3f}\n")
            handle.write(f"- t-test p-value: {lda_result['p_value_ttest']:.3e}\n")
            handle.write(f"- Features: {', '.join(lda_result['features'])}\n")

        written["report_txt"] = str(summary_path)

    return written


# ── Class dominance ───────────────────────────────────────────────────────────
# Non-parametric Driver-vs-Passenger comparison of structural metrics: Mann-Whitney U
# with the common-language effect size (P(driver > passenger)) and rank-biserial
# correlation.
#
# This existed only as inline code inside each of the five per-protein notebooks and was
# lost when they were replaced by run_structure_metrics.py -- a genuine regression, since
# it is evidence the discrimination stage does not produce: that stage ranks the *pLDDT
# feature table* by Cohen's d and fits an LDA, both parametric and both on a different
# input. This tests the RMSF and DNA-distance metrics directly, distribution-free.
#
# Labels arrive as an injected callable rather than being derived here, so the caller
# passes labels.classify_allele_or_artifact and there is no second labelling path to
# drift from the dispatcher (the notebooks each had their own hardcoded DRIVER_MUTATIONS
# / PASSENGER_MUTATIONS lists, which is precisely how they drifted apart).

DOMINANCE_CLASS_ORDER = ["Passenger", "Driver"]
DOMINANCE_PALETTE = {"Passenger": "#2E86AB", "Driver": "#D1495B"}


def attach_class_labels(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    classify,
    artifact_col: str = "artifact",
) -> pd.DataFrame:
    """Keep only Driver/Passenger rows, with a `class` column, for the given metrics.

    `classify` is called with the full artifact name, not a bare mutation code:
    clinvar_by_gene (MutS_alpha) resolves MSH2 vs MSH6 from the name, and those two
    genes number their residues independently.
    """
    keep = ["artifact_label", "class", *metric_cols]
    if df is None or df.empty or artifact_col not in df.columns:
        return pd.DataFrame(columns=[artifact_col, *keep])

    out = df.copy()
    out["class"] = out[artifact_col].astype(str).map(classify)
    out = out[out["class"].isin(DOMINANCE_CLASS_ORDER)].copy()
    cols = [artifact_col, "class", *[c for c in metric_cols if c in out.columns]]
    return out[cols]


def class_dominance_stats(
    df: pd.DataFrame,
    metric_col: str,
    panel_name: str,
    *,
    min_per_class: int = 2,
) -> Optional[Dict[str, Any]]:
    """Mann-Whitney U of Driver vs Passenger on one metric, with effect sizes.

    `p_driver_gt_passenger` is U / (n_d * n_p) -- the probability a randomly chosen
    driver exceeds a randomly chosen passenger, which is also the ROC-AUC of that metric
    used alone. `rank_biserial` is 2p - 1, so 0 means no separation and +/-1 is perfect.
    Returns None when either class is too small to test.
    """
    if df is None or df.empty or metric_col not in df.columns:
        return None

    d_vals = df.loc[df["class"] == "Driver", metric_col].dropna().astype(float).to_numpy()
    p_vals = df.loc[df["class"] == "Passenger", metric_col].dropna().astype(float).to_numpy()
    if len(d_vals) < min_per_class or len(p_vals) < min_per_class:
        return None

    u_stat, p_value = stats.mannwhitneyu(d_vals, p_vals, alternative="two-sided")
    p_driver_gt = float(u_stat / (len(d_vals) * len(p_vals)))
    return {
        "panel": panel_name,
        "metric": metric_col,
        "n_driver": int(len(d_vals)),
        "n_passenger": int(len(p_vals)),
        "driver_mean": float(np.mean(d_vals)),
        "passenger_mean": float(np.mean(p_vals)),
        "p_driver_gt_passenger": p_driver_gt,
        "p_passenger_gt_driver": float(1.0 - p_driver_gt),
        "rank_biserial": float(2.0 * p_driver_gt - 1.0),
        "p_value_mw": float(p_value),
    }


def run_class_dominance(
    metric_specs: Sequence[Tuple[pd.DataFrame, str, str]],
    *,
    classify,
    artifact_col: str = "artifact",
    status_col: str = "status",
    status_ok: str = "ok",
    out_csv: Path | str | None = None,
) -> Dict[str, Any]:
    """Dominance table over several (dataframe, metric column, panel label) triples.

    Taking the source frame per metric lets whole-protein metrics (from the RMSF summary)
    and delta-vs-WT metrics (from the delta summary) land in one table, which is how the
    notebooks presented them.

    Returns {"dominance_df", "labelled": {metric: labelled frame}}.
    """
    rows, labelled = [], {}
    for source_df, metric_col, panel_name in metric_specs:
        if source_df is None or source_df.empty:
            continue
        df = source_df
        if status_col in df.columns:
            df = df[df[status_col] == status_ok]
        lab = attach_class_labels(df, [metric_col], classify=classify, artifact_col=artifact_col)
        labelled[metric_col] = lab
        row = class_dominance_stats(lab, metric_col, panel_name)
        if row is not None:
            rows.append(row)

    dominance_df = pd.DataFrame(rows)
    if out_csv is not None and not dominance_df.empty:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        dominance_df.to_csv(out_csv, index=False)
    return {"dominance_df": dominance_df, "labelled": labelled}


def _violin_panel(ax, df, metric_col, title, ylabel):
    if df is None or df.empty or metric_col not in df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No class-labelled {metric_col}", ha="center", va="center")
        return
    order = [c for c in DOMINANCE_CLASS_ORDER if c in set(df["class"])]
    sns.violinplot(data=df, x="class", y=metric_col, order=order, hue="class",
                   palette=DOMINANCE_PALETTE, inner=None, legend=False, ax=ax)
    sns.stripplot(data=df, x="class", y=metric_col, order=order, color="black",
                  size=2.5, alpha=0.35, ax=ax)
    sns.pointplot(data=df, x="class", y=metric_col, order=order, estimator=np.mean,
                  errorbar=("ci", 95), color="black", markers="D", linestyles="", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def plot_class_dominance_panels(
    dominance_result: Mapping[str, Any],
    *,
    violin_specs: Sequence[Tuple[str, str, str]],
    dataset_name: str = "Dataset",
    title: str | None = None,
    out_path: Path | str | None = None,
):
    """2x2 figure: two metric violins, dominance probability, rank-biserial effect size.

    This is the layout the RMSF notebooks used. `violin_specs` is
    [(metric_col, panel_title, y_label), ...]; up to two are drawn.
    """
    dominance_df = dominance_result["dominance_df"]
    labelled = dominance_result["labelled"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax1, ax2, ax3, ax4 = axes.ravel()

    for ax, spec in zip((ax1, ax2), list(violin_specs)[:2]):
        metric_col, panel_title, ylabel = spec
        _violin_panel(ax, labelled.get(metric_col), metric_col, panel_title, ylabel)

    if not dominance_df.empty:
        prob = (dominance_df[["panel", "p_driver_gt_passenger", "p_passenger_gt_driver"]]
                .rename(columns={"p_driver_gt_passenger": "Driver > Passenger",
                                 "p_passenger_gt_driver": "Passenger > Driver"})
                .melt(id_vars="panel", var_name="direction", value_name="probability"))
        sns.barplot(data=prob, x="panel", y="probability", hue="direction", ax=ax3)
        ax3.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax3.set_ylim(0, 1)
        ax3.set_title("Dominance probability")
        ax3.set_xlabel("Metric"); ax3.set_ylabel("Probability")
        ax3.tick_params(axis="x", rotation=15)
        ax3.grid(axis="y", alpha=0.25)

        sns.barplot(data=dominance_df, x="panel", y="rank_biserial", color="#8E7DBE", ax=ax4)
        ax4.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax4.set_ylim(-1, 1)
        ax4.set_title("Rank-biserial effect size")
        ax4.set_xlabel("Metric"); ax4.set_ylabel("Effect size (Driver > Passenger)")
        ax4.tick_params(axis="x", rotation=15)
        ax4.grid(axis="y", alpha=0.25)
    else:
        for ax in (ax3, ax4):
            ax.axis("off")
            ax.text(0.5, 0.5, "Not enough class-labelled data", ha="center", va="center")

    fig.suptitle(title or f"{dataset_name} dominance: Driver vs Passenger")
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_class_dominance_ranked(
    dominance_result: Mapping[str, Any],
    *,
    dataset_name: str = "Dataset",
    title: str | None = None,
    top_n: int = 10,
    probability_path: Path | str | None = None,
    effect_size_path: Path | str | None = None,
    violin_grid_path: Path | str | None = None,
    ncols: int = 3,
) -> pd.DataFrame:
    """Three separate figures for a many-metric table (the DNA-distance layout).

    Ranks metrics by P(driver > passenger), keeps the top `top_n`, and writes a dominance
    probability bar chart, a rank-biserial bar chart, and a violin grid. Returns the
    ranked frame that was plotted, so the caller can see what the cut kept.
    """
    dominance_df = dominance_result["dominance_df"]
    labelled = dominance_result["labelled"]
    base_title = title or f"{dataset_name} class dominance: Driver vs Passenger"
    if dominance_df.empty:
        return dominance_df

    ranked = dominance_df.sort_values("p_driver_gt_passenger", ascending=False).head(top_n).copy()

    if probability_path is not None:
        prob = (ranked[["metric", "p_driver_gt_passenger", "p_passenger_gt_driver"]]
                .rename(columns={"p_driver_gt_passenger": "Driver > Passenger",
                                 "p_passenger_gt_driver": "Passenger > Driver"})
                .melt(id_vars="metric", var_name="direction", value_name="probability"))
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        sns.barplot(data=prob, x="metric", y="probability", hue="direction", ax=ax)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{base_title} - Dominance probability (top {len(ranked)})", fontsize=15)
        ax.set_xlabel("Metric"); ax.set_ylabel("Probability")
        ax.tick_params(axis="x", rotation=65)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Direction", frameon=True)
        Path(probability_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(probability_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if effect_size_path is not None:
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        sns.barplot(data=ranked, x="metric", y="rank_biserial", color="#8E7DBE", ax=ax)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{base_title} - Rank-biserial effect size (top {len(ranked)})", fontsize=15)
        ax.set_xlabel("Metric"); ax.set_ylabel("Effect size (Driver > Passenger)")
        ax.tick_params(axis="x", rotation=65)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=0.25)
        Path(effect_size_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(effect_size_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if violin_grid_path is not None:
        metrics = list(ranked["metric"])
        nrows = max(1, math.ceil(len(metrics) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.5 * nrows),
                                 constrained_layout=True, squeeze=False)
        flat = axes.ravel()
        for ax, metric_col in zip(flat, metrics):
            _violin_panel(ax, labelled.get(metric_col), metric_col,
                          metric_col.replace("_", " "), "Value")
        for ax in flat[len(metrics):]:
            ax.axis("off")
        fig.suptitle(f"{base_title} - top {len(metrics)} metrics", fontsize=14)
        Path(violin_grid_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(violin_grid_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return ranked


__all__ = [
    "StructureMetricsConfig",
    "PROJECT_ROOT",
    "DEFAULT_PLDDT_ROOT",
    "DEFAULT_METRICS_ROOT",
    "extract_cif_paths",
    "extract_plddt_from_cif",
    "analyze_plddt_distribution",
    "canonical_artifact_stem",
    "drop_matching_sample",
    "iter_artifacts",
    "rmsf_map_ca_from_structure_files",
    "rmsf_map_ca_from_tar",
    "rmsf_map_ca_from_folder",
    "extract_chain_interface_residues",
    "compute_protein_dna_min_distances",
    "summarize_protein_dna_distances",
    "interface_residue_index",
    "filter_rmsf_outputs_to_residue_set",
    "filter_rmsf_outputs_to_interface",
    "run_rmsf_batch_vs_wt",
    "load_rmsf_batch_outputs",
    "run_or_load_rmsf_batch_vs_wt",
    "run_protein_dna_distance_batch",
    "load_protein_dna_distance_batch_outputs",
    "plot_rmsf_summary_panels",
    "plot_rmsf_top_delta_residues",
    "process_artifacts_to_plddt",
    "resolve_plddt_dir",
    "list_plddt_csv_files",
    "load_plddt_matrix",
    "flatten_plddt",
    "compute_per_residue_plddt_ks",
    "summarize_per_residue_plddt_ks",
    "plot_plddt_ks_profile_by_class",
    "plot_plddt_ks_vs_d1",
    "compute_per_residue_plddt_distribution_metrics",
    "DEFAULT_PLDDT_OUTLIER_THRESHOLD",
    "summarize_per_residue_plddt_distribution_metrics",
    "plot_plddt_distribution_profile_by_class",
    "plot_plddt_distribution_summary",
    "plot_plddt_distribution_vs_d1",
    "find_wild_type_file",
    "summarize_wild_type",
    "extract_mutation_code",
    "compute_feature_row",
    "build_mutation_feature_table",
    "build_plddt_ks_tables",
    "build_plddt_distribution_tables",
    "plot_rmsf_vs_d1_scatter",
    "plot_metric_rank_with_highlights",
    "with_binary_label",
    "plot_d1_distribution_by_class",
    "cohens_d",
    "separation_score",
    "run_feature_discrimination_analysis",
    "plot_feature_distributions",
    "plot_feature_roc_curves",
    "run_lda_analysis",
    "run_lda_analysis_cv",
    "plot_lda_panels",
    "lda_assumption_tests",
    "export_discrimination_bundle",
    "merge_rmsf_shards",
    "attach_class_labels",
    "class_dominance_stats",
    "run_class_dominance",
    "plot_class_dominance_panels",
    "plot_class_dominance_ranked",
]
