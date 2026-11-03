"""Canonical labelled structural cohort and its recorded file provenance.

The structure-metrics pipeline writes one distribution-summary row per source
artifact, including the exact mutant pLDDT file, exact WT pLDDT reference, and
the label assigned by the protein-specific ground-truth route.  Downstream
experiments should consume that table directly.  Reconstructing mutation keys
from filenames loses construct and subunit information (most visibly for
``POLE_268-471_*``) and can silently change the evaluated cohort.

This module performs no metric calculation.  It only validates and exposes the
cohort already recorded by the primary pipeline.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from metric_analysis.pipeline.config.protein_config import get_protein_config
from metric_analysis.pipeline.paths import REPO_ROOT


# Datasets, not proteins: MutS_alpha is carried as its two subunits since the
# 2026-09-07 split. MSH2 and MSH6 have no distribution summary of their own --
# they read MutS_alpha's and keep their own rows (see load_canonical_cohort).
# TP53 and KRAS left on 2026-10-13: they have no binary label any more (their
# assays are continuous targets, analyses/continuous/continuous_functional). FOXA1, a
# positional control no paper result uses, left the release on 2026-11-03.
CANONICAL_PROTEINS = (
    "POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA",
)
# The two MutSalpha subunits are held out together by default. They shared one
# wild-type ensemble until MSH6 moved to the G:T-mismatch rerun (2026-10-12);
# they still come from models of the same complex, with MSH2's chain present in
# both. This is a conservative evaluation policy, not a claim of identical
# batches. Anything absent is its own group.
LOGO_FOLD_GROUP = {"MSH2": "MutS_alpha", "MSH6": "MutS_alpha"}


def logo_fold_group(dataset: str, fold_grouping: str = "grouped") -> str:
    """Hold out shared-WT subunits together, or explicitly select separate folds."""
    if fold_grouping not in {"grouped", "separate"}:
        raise ValueError(f"Unknown LOGO fold policy: {fold_grouping!r}")
    return LOGO_FOLD_GROUP.get(dataset, dataset) if fold_grouping == "grouped" else dataset


def logo_splits(datasets, fold_grouping: str = "grouped"):
    """Training/validation indices per reported dataset, excluding its fold group.

    Normalization, weighting and AUC aggregation still use dataset identities.
    MSH2/MSH6 share the training exclusion but retain separate validation rows.
    """
    datasets = np.asarray(datasets, dtype=str)
    groups = np.array([logo_fold_group(p, fold_grouping) for p in datasets])
    if len(np.unique(groups)) < 2:
        raise ValueError("LOGO requires at least two independent fold groups")
    return [(np.flatnonzero(groups != logo_fold_group(p, fold_grouping)),
             np.flatnonzero(datasets == p)) for p in sorted(np.unique(datasets))]


STRUCTURE_METRICS_ROOT = REPO_ROOT / "metric_analysis" / "structure_metrics"
STRICT_MANIFEST = REPO_ROOT / "metric_analysis" / "manifests" / "strict_canonical_manifest.csv"
REQUIRED_COLUMNS = {
    "mutation_code", "mutation_type", "label", "source_file", "wt_source_file",
    "n_wt_structures", "n_mutant_structures", "n_wt_residues",
    "n_mutant_residues", "status",
}


def _existing_repo_path(value: object, *, protein: str) -> Path:
    """Resolve a recorded path without depending on the checkout's old prefix."""
    path = Path(str(value))
    candidates = []
    if path.is_absolute() and path.is_relative_to(REPO_ROOT):
        candidates.append(path)
    elif not path.is_absolute():
        candidates.append(REPO_ROOT / path)
    local = REPO_ROOT / "PLDDTs" / protein / path.name
    if "PLDDTs" in path.parts[:-1]:
        # Another checkout's PLDDTs/<folder>/<file>: re-root it here, KEEPING the
        # folder. Only a file recorded in the protein's own folder may fall back to
        # PLDDTs/<protein>/<name>: a POLE_25seed/ copy (2026-10-20) has the same
        # basename as its 160-prediction original in POLE/, and falling back would
        # silently swap one for the other.
        cut = len(path.parts) - 1 - path.parts[::-1].index("PLDDTs")
        rerooted = REPO_ROOT.joinpath(*path.parts[cut:])
        own_folder = path.parent.name == protein
        candidates.extend([rerooted, local, path] if own_folder else [rerooted, path])
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return (local if own_folder else rerooted).resolve()
    candidates.extend([local, path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return local.resolve()


def source_family(protein: str, source_file: Path | str) -> str:
    """Human-readable construct/subunit family retained for stratified checks."""
    name = Path(source_file).name.upper()
    if protein == "POLE":
        return "POLE_268-471" if name.startswith("POLE_268-471_") else "POLE_other"
    if protein in ("MutS_alpha", "MSH2", "MSH6"):
        if "_MSH2_" in name:
            return "MSH2"
        if "_MSH6_" in name:
            return "MSH6"
        return "MutS_unspecified"
    if protein == "TP53":
        return "P53_tetramer"
    return protein


def _label_source(cfg: dict) -> str:
    mode = cfg["label_mode"]
    if mode == "clinvar":
        return str(cfg["clinvar_csv"])
    if mode == "clinvar_by_gene":
        return json.dumps(cfg["clinvar_csv_by_gene"], sort_keys=True, default=str)
    if mode == "csv_lookup":
        return str(cfg["labels_csv"])
    if mode == "hotspot_range":
        lo, hi = cfg["hotspot_range"]
        return f"position_range={lo}-{hi}"
    return mode


def load_canonical_cohort(
    protein: str,
    *,
    require_files: bool = True,
) -> pd.DataFrame:
    """Return the labelled Driver/Passenger cohort recorded by the pipeline.

    The returned frame has one row per evaluated variant.  It deliberately does
    not derive labels from a fusion cache and does not parse mutation identifiers
    from artifact filenames.
    """
    if protein not in CANONICAL_PROTEINS:
        raise ValueError(f"Unsupported protein {protein!r}")

    cohort_cfg = get_protein_config(protein, ["output_prefix", "label_mode"])
    parent = str(cohort_cfg.get("cohort_parent", protein))
    subunit_token = cohort_cfg.get("cohort_subunit_token")
    row_range = cohort_cfg.get("plddt_row_range")

    # A subunit dataset has no summary of its own; it reads the modelled
    # complex's and keeps the artifacts belonging to its own subunit.
    path = (
        STRUCTURE_METRICS_ROOT / parent
        / f"{parent}_plddt_distribution_summary.csv"
    )
    frame = pd.read_csv(path)
    if subunit_token:
        token = f"_{str(subunit_token).upper()}_"
        keep = frame["source_file"].astype(str).map(
            lambda value: token in Path(value).name.upper()
        )
        if not keep.any():
            raise ValueError(
                f"{protein}: no rows in {parent}'s summary carry {token!r}"
            )
        frame = frame[keep].copy()
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise KeyError(f"{protein}: distribution summary lacks {missing}")

    frame = frame[
        frame["mutation_type"].isin(["Driver", "Passenger"])
        & frame["status"].astype(str).str.lower().eq("ok")
    ].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["label", "source_file", "wt_source_file"])
    frame["label"] = frame["label"].astype(int)

    if not STRICT_MANIFEST.is_file():
        raise FileNotFoundError(
            f"strict canonical manifest is missing: {STRICT_MANIFEST}; run "
            "analyses/cohort/build_strict_canonical_manifest.py"
        )
    strict = pd.read_csv(STRICT_MANIFEST)
    required_strict = {"protein", "mutation_code", "mutation_type", "label"}
    missing_strict = sorted(required_strict - set(strict.columns))
    if missing_strict:
        raise KeyError(f"strict canonical manifest lacks {missing_strict}")
    strict = strict[strict["protein"].eq(protein)].copy()
    if strict.duplicated("mutation_code").any():
        raise ValueError(f"{protein}: duplicate mutation codes in strict manifest")
    frame = frame.merge(
        strict[["mutation_code", "mutation_type", "label"]],
        on="mutation_code", how="inner", validate="one_to_one",
        suffixes=("_recorded", ""),
    )
    if not frame["label_recorded"].astype(int).eq(frame["label"].astype(int)).all():
        raise ValueError(f"{protein}: strict and recorded numeric labels disagree")
    if not frame["mutation_type_recorded"].eq(frame["mutation_type"]).all():
        raise ValueError(f"{protein}: strict and recorded text labels disagree")

    expected = frame["mutation_type"].eq("Driver").astype(int)
    if not np.array_equal(frame["label"].to_numpy(), expected.to_numpy()):
        bad = frame.loc[frame["label"].ne(expected), "mutation_code"].tolist()
        raise ValueError(f"{protein}: numeric/text label disagreement for {bad[:10]}")

    duplicates = frame[frame.duplicated("mutation_code", keep=False)]
    if not duplicates.empty:
        details = duplicates[["mutation_code", "source_file"]].to_dict("records")
        raise ValueError(f"{protein}: duplicate labelled mutation codes: {details[:10]}")

    frame["source_file"] = [
        str(_existing_repo_path(value, protein=parent))
        for value in frame["source_file"]
    ]
    frame["wt_source_file"] = [
        str(_existing_repo_path(value, protein=parent))
        for value in frame["wt_source_file"]
    ]
    frame["source_family"] = [
        source_family(protein, value) for value in frame["source_file"]
    ]

    cfg = cohort_cfg
    frame["protein"] = protein
    frame["parent_dataset"] = parent
    frame["logo_fold_group"] = logo_fold_group(protein)
    if row_range is not None:
        start, stop = int(row_range[0]), int(row_range[1])
        frame["plddt_row_range_start"] = start
        frame["plddt_row_range_stop"] = stop
        # The descriptor sees only this block, so the recorded residue counts
        # from the parent's summary would otherwise overstate it.
        frame["n_wt_residues"] = stop - start
        frame["n_mutant_residues"] = stop - start
    else:
        frame["plddt_row_range_start"] = pd.NA
        frame["plddt_row_range_stop"] = pd.NA
    frame["label_provenance"] = cfg["label_mode"]
    frame["label_source"] = _label_source(cfg)

    frame["source_exists"] = frame["source_file"].map(lambda value: Path(value).is_file())
    frame["wt_exists"] = frame["wt_source_file"].map(lambda value: Path(value).is_file())
    if require_files and not bool(frame[["source_exists", "wt_exists"]].all().all()):
        bad = frame.loc[
            ~(frame["source_exists"] & frame["wt_exists"]),
            ["mutation_code", "source_file", "wt_source_file"],
        ]
        raise FileNotFoundError(f"{protein}: missing recorded pLDDT files: {bad.head(10)}")

    columns = [
        "protein", "parent_dataset", "logo_fold_group",
        "mutation_code", "mutation_type", "label",
        "label_provenance", "label_source", "source_family", "source_file", "wt_source_file",
        "n_wt_structures", "n_mutant_structures", "n_wt_residues",
        "n_mutant_residues", "plddt_row_range_start", "plddt_row_range_stop",
        "source_exists", "wt_exists",
    ]
    return frame[columns].sort_values("mutation_code").reset_index(drop=True)


def load_all_canonical_cohorts(*, require_files: bool = True) -> pd.DataFrame:
    """Concatenate every canonical cohort in the manuscript's fixed order."""
    return pd.concat(
        [load_canonical_cohort(p, require_files=require_files) for p in CANONICAL_PROTEINS],
        ignore_index=True,
    )
