#!/usr/bin/env python
"""Freeze the transcript-matched structural cohort used by canonical analyses.

    python experiments/build_strict_canonical_manifest.py                 # split layout
    python experiments/build_strict_canonical_manifest.py --layout merged # historical

SUBUNIT SPLIT (2026-09-07). MutS_alpha is emitted as two datasets, MSH2 and
MSH6, because the merged descriptor averaged over all 2,294 residues of the
heterodimer regardless of which subunit carried the mutation. Each split row
records `plddt_row_range_start/stop`, the half-open row block the descriptor is
restricted to, and `parent_dataset`, the modelled complex it came from.

`parent_dataset` is load-bearing, not documentation: MSH2 and MSH6 share one
wild-type ensemble and one set of AlphaFold jobs, so they are two datasets for
reporting but must stay ONE group for leave-one-gene-out evaluation. Grouping
on `parent_dataset` is what keeps that honest.

The split itself repartitioned MutS_alpha's 147 rows into 109 + 38 (total
3,643). `--layout merged` reproduced the historical single-dataset layout under
a suffixed filename; it is retired, see below.

MSH6 G:T RERUN (2026-10-12). MSH6 now comes from its own parent,
MutS_alpha_MSH6_GT: all 147 MSH6 ClinVar missense refolded with a WT on one
21/21-nt G:T duplex (af3_jobs/configs/MutS_alpha_MSH6_GT.json). The earlier 42
MSH6 variants in MutS_alpha (136/137-nt duplex, measured against MSH2's 21/20-nt
WT) are SUPERSEDED: dropped from the live inventory, never re-admitted from the
frozen v1 rows. MSH6 grows from 38 to 143 labelled (66 D / 77 P), the total from
3,643 to 3,748. LOGO still holds MSH2 and MSH6 out together
(canonical_cohort.LOGO_FOLD_GROUP). There is no merged layout any more: MSH2 and
MSH6 no longer share one parent or one WT.

PIK3CA (2026-10-12) joins as a ClinVar dataset under the same transcript rule
as PTEN and VHL: 91 variants (84 D / 7 P, transcript-matched germline labels),
all of which pass. Total 3,839.

NO BINARY FUNCTIONAL LABELS (2026-10-13). TP53 and KRAS leave the manifest.
Their labels came from thresholding a continuous assay (TP53 phenotypic-selection
score, KRAS HA1E LFC), which the project has abandoned: the assays are continuous
targets only, in experiments/continuous_functional, which builds its own cohort
(KRAS is now the KRAS4B batch). This manifest is the binary (annotation + FOXA1
positional control) cohort: 3,839 - 1,174 (TP53) - 45 (KRAS) = 2,620.

POLE P356S left on 2026-10-20 (no coordinate structures): 2,619. FOXA1, a positional
control no paper result uses, left the release on 2026-11-03: 1,798, the six annotation
genes. The frozen v1 inventory still holds its 821 rows (historical provenance).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metric_analysis.pipeline.clinvar import read_clinvar_csv
from metric_analysis.pipeline.config.protein_config import get_protein_config
from metric_analysis.pipeline.labels import classify_allele_or_artifact
from metric_analysis.pipeline.library.canonical_cohort import _label_source, source_family


OUT = REPO / "metric_analysis" / "manifests"
BASE_MANIFEST = (
    REPO / "metric_analysis" / "manifests" / "structural_variant_inventory_v1.csv"
)
CURRENT_INVENTORY = (
    REPO / "metric_analysis" / "manifests" / "structural_variant_inventory_v2.csv"
)
STRUCTURE_METRICS = REPO / "metric_analysis" / "structure_metrics"
PROTEINS = ("POLE", "PTEN", "VHL", "MutS_alpha", "MutS_alpha_MSH6_GT", "PIK3CA")
# Modelled complexes emitted as one dataset per subunit. Values are the
# half-open pLDDT matrix row block belonging to each subunit; they must match
# CHAIN_LAYOUT in metric_analysis/pipeline/library/combination_axes.py and
# plddt_row_range in both protein_configs.json files.
SUBUNIT_SPLIT = {
    "MutS_alpha": {"MSH2": (0, 934), "MSH6": (934, 2294)},
    "MutS_alpha_MSH6_GT": {"MSH6": (934, 2294)},
}
# Parent rows replaced by a rerun elsewhere: (parent, subunit token) -> never live.
SUPERSEDED = {("MutS_alpha", "MSH6")}
# Labelled variants left out of the cohort by decision, with the reason recorded in
# the exclusions file. POLE P356S (a passenger) has no coordinate structures; on
# 2026-10-20 it was dropped from every analysis so all descriptors share one cohort
# (POLE 1,074 = 38 / 1,036; annotation 1,798 = 612 / 1,186).
EXCLUDED_BY_DECISION = {
    ("POLE", "P356S"): "user_decision_2026-10-20_no_coordinate_structures_one_cohort_for_all_descriptors",
}


def subunit_of(protein: str, source_file: object) -> str | None:
    """The subunit a modelled complex's row belongs to, from its filename token."""
    name = Path(str(source_file)).name.upper()
    for subunit in SUBUNIT_SPLIT.get(protein, {}):
        if f"_{subunit}_" in name:
            return subunit
    return None


def cohort_key(protein: str, source_file: object, code: str) -> tuple[str, str]:
    """(dataset, code): a subunit's variants are one dataset whichever parent they come from."""
    return (subunit_of(protein, source_file) or protein, code)
CLINVAR = {
    "PTEN": ("PTEN", REPO / "Ground_Truths/ClinVar/clinvar_result_pten.csv"),
    "VHL": ("VHL", REPO / "Ground_Truths/ClinVar/clinvar_result_vhl.csv"),
    "MSH2": ("MSH2", REPO / "Ground_Truths/ClinVar/clinvar_result_msh2.csv"),
    "MSH6": ("MSH6", REPO / "Ground_Truths/ClinVar/clinvar_result_msh6.csv"),
    "PIK3CA": ("PIK3CA", REPO / "Ground_Truths/ClinVar/clinvar_result_pik3ca.tsv"),
}
AA = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def transcript_matched_codes(gene: str, path: Path) -> set[str]:
    frame = read_clinvar_csv(path)
    pattern = re.compile(
        rf"^NM_\d+\.\d+\({gene}\):.*?\(p\.([A-Z][a-z]{{2}})(\d+)([A-Z][a-z]{{2}})\)"
    )
    codes = set()
    for name in frame["Name"].astype(str):
        match = pattern.search(name)
        if not match or match.group(1) not in AA or match.group(3) not in AA:
            continue
        position = int(match.group(2))
        if position == 1:  # start-loss, not retained as a strict missense substitution
            continue
        codes.add(f"{AA[match.group(1)]}{position}{AA[match.group(3)]}")
    return codes


def source_record_audit(frame: pd.DataFrame, code: str) -> dict[str, str]:
    """Return the ClinVar rows that caused the broad loader to admit ``code``."""
    tokens = frame["Protein change"].fillna("").astype(str).str.split(",")
    matched = frame.loc[tokens.map(lambda values: code in {v.strip().upper() for v in values})]

    def joined(column: str) -> str:
        if column not in matched:
            return ""
        values = []
        for value in matched[column].dropna():
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
        return " | ".join(values)

    transcripts = []
    for name in matched.get("Name", pd.Series(dtype=str)).dropna().astype(str):
        match = re.match(r"^(NM_\d+\.\d+)", name)
        if match and match.group(1) not in transcripts:
            transcripts.append(match.group(1))
    return {
        "source_record_name": joined("Name"),
        "source_record_protein_change": joined("Protein change"),
        "source_record_classification": joined("Germline classification"),
        "source_record_accession": joined("Accession"),
        "source_record_variation_id": joined("VariationID"),
        "source_record_transcript": " | ".join(transcripts),
    }


def build_current_inventory() -> pd.DataFrame:
    """Apply the live label configuration to every successfully scored variant.

    The frozen v1 inventory is retained as historical provenance and supplies the
    15 formerly admitted ClinVar rows needed by the exclusion audit. It cannot be
    the live cohort source: changing a threshold can make a previously unlabelled
    structural variant eligible, as occurred for KRAS L19F and N116H.
    """
    if not BASE_MANIFEST.exists():
        raise FileNotFoundError(f"frozen broad-cohort manifest not found: {BASE_MANIFEST}")
    historical = pd.read_csv(BASE_MANIFEST)
    if len(historical) != 3656 or historical[["protein", "mutation_code"]].duplicated().any():
        raise ValueError(
            f"unexpected frozen broad-cohort manifest: {BASE_MANIFEST} "
            f"({len(historical)} rows)"
        )

    inventory_columns = list(historical.columns)
    rows = []
    live_keys = set()
    for protein in PROTEINS:
        summary_path = (
            STRUCTURE_METRICS / protein
            / f"{protein}_plddt_distribution_summary.csv"
        )
        summary = pd.read_csv(summary_path)
        required = {
            "mutation_code", "source_file", "wt_source_file", "status",
            "n_wt_structures", "n_mutant_structures", "n_wt_residues",
            "n_mutant_residues",
        }
        missing = sorted(required - set(summary.columns))
        if missing:
            raise KeyError(f"{summary_path} lacks {missing}")
        summary = summary[summary["status"].astype(str).str.lower().eq("ok")]

        cfg = get_protein_config(protein, ["output_prefix", "label_mode"])
        frozen = historical[historical["protein"].eq(protein)]
        # A parent added after the v1 freeze has no frozen row to copy from.
        template = (frozen.iloc[0] if len(frozen) else
                    {"label_provenance": cfg["label_mode"], "label_source": _label_source(cfg)})
        for row in summary.to_dict("records"):
            if (protein, subunit_of(protein, row["source_file"])) in SUPERSEDED:
                continue
            label_class = classify_allele_or_artifact(
                Path(str(row["source_file"])).stem, cfg
            )
            if label_class not in {"Driver", "Passenger"}:
                continue
            source_path = Path(str(row["source_file"]))
            wt_path = Path(str(row["wt_source_file"]))
            record = {
                "protein": protein,
                "mutation_code": str(row["mutation_code"]),
                "mutation_type": label_class,
                "label": 1 if label_class == "Driver" else 0,
                "label_provenance": template["label_provenance"],
                "label_source": template["label_source"],
                "source_family": source_family(protein, source_path),
                "source_file": str(source_path),
                "wt_source_file": str(wt_path),
                "n_wt_structures": int(row["n_wt_structures"]),
                "n_mutant_structures": int(row["n_mutant_structures"]),
                "n_wt_residues": int(row["n_wt_residues"]),
                "n_mutant_residues": int(row["n_mutant_residues"]),
                "source_exists": source_path.is_file(),
                "wt_exists": wt_path.is_file(),
            }
            rows.append(record)
            live_keys.add(cohort_key(protein, source_path, record["mutation_code"]))

    # Preserve only the historical ClinVar-labelled rows that the strict live
    # loader now rejects. They are required to keep the exclusion audit explicit.
    clinvar_proteins = {"PTEN", "VHL", "MutS_alpha"}
    for row in historical.to_dict("records"):
        key = cohort_key(str(row["protein"]), row["source_file"], str(row["mutation_code"]))
        if key not in live_keys and row["protein"] in clinvar_proteins:
            rows.append({column: row[column] for column in inventory_columns})

    inventory = pd.DataFrame(rows)[inventory_columns]
    inventory = inventory.sort_values(["protein", "mutation_code"]).reset_index(drop=True)
    duplicates = inventory[inventory.duplicated(["protein", "mutation_code"], keep=False)]
    if not duplicates.empty:
        raise ValueError(
            "current structural inventory contains duplicate keys: "
            f"{duplicates[['protein', 'mutation_code']].head(10).to_dict('records')}"
        )
    return inventory


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", choices=("split", "merged"), default="split",
                    help="split: MutS_alpha becomes MSH2 + MSH6 (default). "
                         "merged: historical single-dataset layout, written to a "
                         "suffixed filename so the canonical manifest is untouched.")
    args = ap.parse_args(argv)
    split = args.layout == "split"
    if not split:
        raise SystemExit("--layout merged is retired: since 2026-10-12 MSH2 and MSH6 come from "
                         "different parents with different WTs, so there is no single MutS_alpha "
                         "dataset to reproduce. The last merged output is in git history.")

    clinvar_frames = {name: read_clinvar_csv(path) for name, (_, path) in CLINVAR.items()}
    exact = {name: transcript_matched_codes(gene, path) for name, (gene, path) in CLINVAR.items()}
    base = build_current_inventory()
    # 3,658 before the G:T rerun; MutS_alpha's 42 superseded MSH6 rows leave
    # (its 4 frozen M1x rows return as exclusion-audit rows) and 143 arrive;
    # then PIK3CA's 91; TP53 and KRAS left on 2026-10-13 (no binary label): 2,635;
    # FOXA1's 821 left on 2026-11-03: 1,814.
    if len(base) != 1814:
        raise ValueError(f"unexpected current broad-cohort inventory ({len(base)} rows)")
    included, excluded = [], []
    for protein in PROTEINS:
        frame = base.loc[base["protein"].eq(protein)].copy()
        for row in frame.to_dict("records"):
            source = str(row["source_file"]).upper()
            gene = None
            if protein in ("PTEN", "VHL", "PIK3CA"):
                gene = protein
            elif protein in SUBUNIT_SPLIT:
                gene = subunit_of(protein, row["source_file"])
            decided = EXCLUDED_BY_DECISION.get((protein, row["mutation_code"]))
            keep = (gene is None or row["mutation_code"] in exact[gene]) and decided is None
            is_start_loss = gene is not None and int(re.search(r"\d+", row["mutation_code"]).group()) == 1
            # A subunit of a modelled complex becomes its own dataset, restricted
            # to its own rows. parent_dataset keeps the shared wild-type ensemble
            # and the shared AlphaFold jobs recoverable, which is what LOGO must
            # group on.
            block = SUBUNIT_SPLIT.get(protein, {}).get(gene) if split else None
            dataset = gene if block is not None else protein
            record = dict(row)
            record.update({
                "label": int(row["label"]),
                "protein": dataset,
                "parent_dataset": protein,
                "plddt_row_range_start": block[0] if block else "",
                "plddt_row_range_stop": block[1] if block else "",
                "label_gene": gene or protein,
                "cohort_status": "included" if keep else "excluded",
                "cohort_rule": (
                    "non_clinvar_label_route_unchanged"
                    if gene is None else "target_gene_nm_transcript_protein_substitution"
                ),
                "exclusion_reason": (
                    ""
                    if keep else
                    decided
                    if decided else
                    "initiator_codon_variant_not_representable_as_full_length_missense"
                    if is_start_loss else
                    "no_target_gene_nm_transcript_protein_substitution"
                ),
            })
            if block is not None:
                # The descriptor now sees only this subunit's rows.
                record["n_wt_residues"] = block[1] - block[0]
                record["n_mutant_residues"] = block[1] - block[0]
            if not keep and gene is not None:
                record.update(source_record_audit(clinvar_frames[gene], row["mutation_code"]))
            (included if keep else excluded).append(record)

    manifest = pd.DataFrame(included).sort_values(["protein", "mutation_code"])
    audit = pd.DataFrame(excluded).sort_values(["protein", "mutation_code"])
    expected = {"POLE": 1074, "PTEN": 257, "VHL": 124, "PIK3CA": 91}
    # MSH6: 38 of the superseded batch -> 143 of the G:T rerun (2026-10-12).
    expected.update({"MSH2": 109, "MSH6": 143})
    observed = manifest.groupby("protein").size().to_dict()
    if observed != expected or len(manifest) != 1798 or len(audit) != 16:
        raise ValueError(f"strict cohort mismatch: counts={observed}, excluded={len(audit)}")
    parents = manifest.groupby("parent_dataset").size().to_dict()
    if parents.get("MutS_alpha") != 109 or parents.get("MutS_alpha_MSH6_GT") != 143:
        raise ValueError(f"MutS parents mismatch: {parents}")
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if split else "_mutsalpha_merged"
    base.to_csv(CURRENT_INVENTORY, index=False)
    manifest.to_csv(OUT / f"strict_canonical_manifest{suffix}.csv", index=False)
    audit.to_csv(OUT / f"strict_canonical_exclusions{suffix}.csv", index=False)
    print(f"layout: {args.layout}")
    print(manifest.groupby("protein").agg(n=("label", "size"), positive=("label", "sum")))
    print("\nExcluded:\n", audit.to_string(index=False))


if __name__ == "__main__":
    main()
