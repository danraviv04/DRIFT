"""Schema-tolerant reading of ClinVar exports.

ClinVar's download UI has shipped two different column layouts, and this repo holds
files in both. Rather than teach every consumer about the difference (or maintain a
hand-curated copy of each export, which silently goes stale), every ClinVar read goes
through :func:`read_clinvar_csv`, which normalizes to the OLDER column names because
that is what the existing loaders, tests and `Ground_Truths/scripts/csv_clinvar.py`
already speak.

    older export (7 files here)      newer export (ClinVar UI as of 2026-08)
    ---------------------------      ---------------------------------------
    Name                             Variation
    Gene(s)                          Genes
    Germline classification          Classification      (values prefixed "G: ")
    Germline review status           Review status
    VariationID                      Variation ID
    Accession                        VCV accession
    Protein change                   Protein change      (unchanged)
    stars                            -                   (derived here)

Two content differences matter as much as the renames:

* **The "G: " prefix.** Newer exports write ``"G: Pathogenic"`` where the older wrote
  ``"Pathogenic"``. Since the downstream label functions test with substring matches
  (``"pathogenic" in value.lower()``), a prefix left in place would still match — which
  is fine for ``G:`` (germline) but would be WRONG for a somatic (``S:``) or
  oncogenicity (``O:``) classification, silently turning a somatic call into a germline
  Driver. So ``G:`` is stripped and any other prefix blanks the value instead, leaving
  the row unclassified rather than mislabelled.
* **``stars``** (0-4 review confidence) exists only in the older files, because
  `csv_clinvar.py` adds it after download. It is derived here when absent so both
  layouts expose it. Nothing currently filters on it, but it is the natural handle for
  a review-quality cutoff and shouldn't silently vanish when an export is refreshed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# canonical (older) name -> every spelling seen in the wild, best first
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "Name": ("Name", "Variation"),
    "Gene(s)": ("Gene(s)", "Genes"),
    "Germline classification": ("Germline classification", "Classification"),
    "Germline review status": ("Germline review status", "Review status"),
    "VariationID": ("VariationID", "Variation ID"),
    "Accession": ("Accession", "VCV accession"),
}

# "G: Pathogenic" -> "Pathogenic". Only germline; see module docstring.
_GERMLINE_PREFIX_RE = re.compile(r"^\s*G:\s*")
_ANY_PREFIX_RE = re.compile(r"^\s*([A-Za-z]+):\s")

_REVIEW_STATUS_TO_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "no assertion criteria provided": 0,
}


def _strip_germline_prefix(value):
    """Drop a leading ``G:``; blank anything carrying a different prefix.

    A non-germline prefix (``S:`` somatic, ``O:`` oncogenicity) means this row has no
    germline classification. Returning NA keeps it unclassified; leaving the text would
    let a substring match read it as a germline Pathogenic/Benign call.
    """
    if pd.isna(value):
        return value
    text = str(value)
    if _GERMLINE_PREFIX_RE.match(text):
        return _GERMLINE_PREFIX_RE.sub("", text).strip()
    other = _ANY_PREFIX_RE.match(text)
    if other:
        return pd.NA
    return text


def normalize_clinvar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with canonical (older-export) column names and values.

    Idempotent: a frame already in the older layout passes through with only the
    derived ``stars`` column possibly added.
    """
    out = df.copy()

    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out = out.rename(columns={alias: canonical})
                break

    if "Germline classification" in out.columns:
        out["Germline classification"] = out["Germline classification"].map(_strip_germline_prefix)

    if "stars" not in out.columns and "Germline review status" in out.columns:
        out["stars"] = (
            out["Germline review status"].astype(str).str.strip().str.lower()
            .map(_REVIEW_STATUS_TO_STARS)
        )

    return out


def read_clinvar_csv(path: Path | str, **read_csv_kwargs) -> pd.DataFrame:
    """Read a ClinVar export in either layout, normalized to the canonical columns."""
    read_csv_kwargs.setdefault("low_memory", False)
    if "sep" not in read_csv_kwargs and "delimiter" not in read_csv_kwargs:
        read_csv_kwargs["sep"] = "\t" if Path(path).suffix.lower() == ".tsv" else ","
    # ClinVar TSV rows can carry a trailing empty field absent from the header.
    # Prevent pandas from consuming Name as an implicit index and shifting labels.
    read_csv_kwargs.setdefault("index_col", False)
    return normalize_clinvar_frame(pd.read_csv(path, **read_csv_kwargs))


def clinvar_schema(path: Path | str) -> str:
    """``"old"`` / ``"new"`` / ``"unknown"`` — for diagnostics and audit output."""
    sep = "\t" if Path(path).suffix.lower() == ".tsv" else ","
    cols = set(pd.read_csv(path, sep=sep, nrows=0).columns)
    if {"Name", "Germline classification"} <= cols:
        return "old"
    if {"Variation", "Classification"} <= cols:
        return "new"
    return "unknown"


__all__ = ["read_clinvar_csv", "normalize_clinvar_frame", "clinvar_schema"]
