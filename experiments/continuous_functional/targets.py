"""Continuous assay targets for the four functional genes.

Functional analyses are continuous only (2026-10-13). No assay is thresholded
into driver/passenger anywhere in the project; TP53 and KRAS have no binary label
at all, and PTEN/MSH2 keep their ClinVar label for the (annotation) binary
analyses while their assay is used here.

TP53 : Phenotypic_Selection_Model (Giacomelli 2018). Strictly positive and
       heavy-tailed (0.007 - 39.7), so the target is log10 of it; Spearman is
       invariant to that choice.
KRAS : HA1E_WT_LFC from the GOF DMS screen (Kwon 2024), KRAS4B numbering,
       used as-is (already a log scale).
PTEN : Cum_score (Mighell 2018, Table S2), yeast lipid-phosphatase fitness;
       LOWER = more impaired, so the target is -Cum_score.
MSH2 : LOF score (Jia 2021, TableS5_AllVariants); HIGHER = more impaired.

TP53 and KRAS are oriented so LARGER = stronger cancer-relevant effect (selection,
activation); PTEN and MSH2 so LARGER = more impaired. Cohort rules live in
dataset.cohort().
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

TP53_PATH = REPO / "Ground_Truths" / "Supplementary table 3.xlsx"
TP53_SCORE_COL = "Phenotypic_Selection_Model"

KRAS_PATH = REPO / "Ground_Truths" / "KRAS" / "Extended Data Table 1.xlsx"
KRAS_SHEET = "KRAS GOF DMS Screen"
KRAS_SCORE_COL = "HA1E_WT_LFC"

CONTINUOUS_GENES = ("TP53", "KRAS", "PTEN", "MSH2")
FUNCTIONAL_SCORES = {
    gene: REPO / "Ground_Truths" / gene / "functional_scores.csv"
    for gene in ("PTEN", "MSH2")
}
KRAS_STRATA = (
    REPO / "experiments" / "kras4b_coverage" / "results" / "batch_strata.csv"
)


def _norm_mutation(s):
    return s.astype(str).str.strip().str.upper()


def tp53_targets():
    """(gene, mutation, y_raw, y) for every scored TP53 variant."""
    df = pd.read_excel(TP53_PATH, sheet_name=0)
    mutation = (
        df["AA_wt"].astype(str).str.strip()
        + df["Position"].astype(str).str.strip()
        + df["AA_variant"].astype(str).str.strip()
    )
    out = pd.DataFrame({
        "gene": "TP53",
        "mutation": _norm_mutation(mutation),
        "y_raw": pd.to_numeric(df[TP53_SCORE_COL], errors="coerce"),
    }).dropna(subset=["y_raw"])
    # Duplicate (gene, mutation) rows: keep the strongest reading.
    out = out.groupby(["gene", "mutation"], as_index=False)["y_raw"].max()
    out["y"] = np.log10(out["y_raw"])
    return out


def kras_targets():
    """(gene, mutation, y_raw, y) for every KRAS variant in the DMS screen."""
    df = pd.read_excel(KRAS_PATH, sheet_name=KRAS_SHEET)
    out = pd.DataFrame({
        "gene": "KRAS",
        "mutation": _norm_mutation(df["VAR"]),
        "y_raw": pd.to_numeric(df[KRAS_SCORE_COL], errors="coerce"),
    }).dropna(subset=["y_raw"])
    out = out.groupby(["gene", "mutation"], as_index=False)["y_raw"].max()
    out["y"] = out["y_raw"]
    return out


def _assay_csv_targets(gene, sign):
    """(gene, mutation, y_raw, y, flags) for one extracted assay CSV."""
    df = pd.read_csv(FUNCTIONAL_SCORES[gene])
    out = pd.DataFrame({
        "gene": gene,
        "mutation": _norm_mutation(df["mutation_code"]),
        "y_raw": pd.to_numeric(df["functional_score"], errors="raise"),
    })
    for flag in ("high_conf", "known_splice_disruptive"):
        if flag in df:
            out[flag] = df[flag].astype(bool)
    if out.duplicated(["gene", "mutation"]).any():
        raise ValueError(f"duplicate {gene} assay rows")
    out["y"] = sign * out["y_raw"]
    return out


def pten_targets():
    """PTEN impairment = -Cum_score (lower fitness = more impaired)."""
    return _assay_csv_targets("PTEN", -1.0)


def msh2_targets():
    """MSH2 LOF score, already oriented so higher = more impaired."""
    return _assay_csv_targets("MSH2", 1.0)


def kras_blind():
    """KRAS4B variants drawn blind to the assay (blind + blind+hotspot strata).

    The 44 hotspot_only variants were generated because of their assay values,
    so they are not a sample of the screen and are left out.
    """
    strata = pd.read_csv(KRAS_STRATA)
    return set(strata.loc[strata["stratum"].isin(["blind", "blind+hotspot"]), "VAR"].str.upper())


def continuous_targets():
    return pd.concat(
        [tp53_targets(), kras_targets(), pten_targets(), msh2_targets()],
        ignore_index=True,
    )
