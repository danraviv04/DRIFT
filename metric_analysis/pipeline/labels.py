"""Single source of truth for driver/passenger label assignment across all 5 proteins.

Moved out of run_ensemble_sweep_parallel.py so both that script and the
per-protein pipeline (run_structure_metrics.py) classify mutations identically.
Before this, each notebook reimplemented its own label logic by hand — MutS_alpha's
notebook only ever loaded the MSH2 ClinVar file (ignoring MSH6 despite the config
listing both), and KRAS's notebook computed ClinVar labels live and then overwrote
them with a hardcoded frozenset snapshot. classify_allele_or_artifact here is the
one correct implementation (dispatches by cfg["label_mode"]) that both bugs are
fixed by simply routing through.
"""

import re

import pandas as pd

from metric_analysis.pipeline.clinvar import read_clinvar_csv


def is_wt_artifact(value):
    return bool(re.search(r"(^|[_\-/])WT($|[_\-/])", str(value).upper()))


_ARCHIVE_EXT = re.compile(r"\.(tar\.gz|tgz|tar|zip)$", re.IGNORECASE)


def artifact_key(value):
    """The name two sources must agree on to mean the same ensemble.

    The same AF3 job reaches us as ``P53_WT_with_ZN_and_DNA.tar.gz`` (data pipeline
    run on the cluster), ``P53_WT_with_ZN_and_DNA_data.tar.gz`` (inference from an
    MSA-ready ``*_data.json``: the runner names its output after the input file), or
    either without the extension inside a cache. Configs, caches and archives were
    built at different times and disagree on the ``_data``, which twice made a
    configured WT match nothing and every WT-relative metric silently disappear
    (PTEN until 10-14, TP53 after its merge). So: drop the archive extension and one
    trailing ``_data``, case-insensitively, and nothing else. Unlike
    ``canonical_artifact_stem`` it keeps everything after ``_with``, so
    ``P53_WT_with_ZN_and_DNA`` and ``..._again`` stay different ensembles.
    """
    name = str(value).rstrip("/").rsplit("/", 1)[-1]
    name = _ARCHIVE_EXT.sub("", name)
    name = re.sub(r"_data$", "", name, flags=re.IGNORECASE)
    return name.upper()


def extract_mutation_code(value):
    # Token boundaries keep gene names that embed the pattern from matching
    # first: without them "PIK3CA_H1047R" parses as "K3C".
    m = re.search(r"(?<![A-Z0-9])([A-Z]\d+[A-Z])(?![A-Z0-9])", str(value).upper())
    return m.group(1) if m else None


def artifact_to_allele(value):
    if is_wt_artifact(value):
        return "WT"
    code = extract_mutation_code(value)
    return code if code else str(value)


_ALLELE_RE = re.compile(r"^[A-Z]\d+[A-Z]$")


_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def load_clinvar_driver_passenger(csv_path, target_gene=None):
    """Driver/passenger alleles from a raw ClinVar export.

    When ``target_gene`` is supplied, require a target-gene ``NM_`` transcript
    protein substitution in ``Name`` and exclude residue-1 start-loss records.
    This is the strict route used for PTEN, VHL, MSH2, and MSH6.  Without a
    target gene, retain the historical schema-tolerant route that reads clean
    allele codes such as ``G12D`` from ``Protein change``.

    'Protein change' holds **comma-separated alleles** — one row is often several
    transcript annotations of the same variant, e.g. "M67L, M1L" or "R106K, R40K".
    Each must be split out. Grouping on the raw string instead silently matched almost
    nothing: 111 of MSH2's 120 rows are comma-separated, so MSH2 produced *zero* driver
    labels for 109 structural variants, and the pre-transcript-audit MutS_alpha pool
    ended up with 15/151 labelled instead of 151/151. Splitting recovers 71 MSH2
    drivers, exactly matching what the notebook's Name-parenthetical approach found;
    the later strict cohort separately excludes four initiator-methionine variants.

    Classification is a per-allele consensus across every submission mentioning that
    allele (case-insensitive, so "Likely pathogenic"/"Likely benign" count).
    """
    # Schema-tolerant: ClinVar's UI ships two column layouts and this repo has both.
    df = read_clinvar_csv(csv_path)
    df = df[df["Protein change"].notna()]

    pairs = []
    names = df["Name"] if "Name" in df else pd.Series("", index=df.index)
    strict_names = bool(target_gene) and "Name" in df
    for name, protein_change, classification in zip(
        names, df["Protein change"], df["Germline classification"]
    ):
        if strict_names:
            match = re.search(
                rf"^NM_\d+\.\d+\({re.escape(target_gene)}\):.*?"
                r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)",
                str(name),
            )
            if not match or int(match.group(2)) == 1:
                continue
            ref, alt = _AA3.get(match.group(1).upper()), _AA3.get(match.group(3).upper())
            if ref and alt:
                pairs.append((f"{ref}{match.group(2)}{alt}", classification))
            continue
        for allele in str(protein_change).split(","):
            allele = allele.strip().upper()
            if _ALLELE_RE.match(allele):  # skips nonsense/frameshift forms like "G71*"
                pairs.append((allele, classification))

    def _consensus(classes):
        cs = [str(c).lower() for c in classes]
        # ClinVar's aggregate verdict "Conflicting classifications of
        # pathogenicity" is NOT a pathogenic call -- it is what ClinVar returns
        # when submitters disagree, i.e. uncertain. The votes below are decided
        # by substring, and that string contains "pathogenic" (inside
        # "pathogenicity") while containing no "benign", so without this filter
        # it reads as a confident driver label.
        #
        # This study excludes uncertain alleles rather than assigning them: an
        # allele left with no usable record returns "Unknown" and lands in
        # neither the driver nor the passenger set, exactly as "Uncertain
        # significance" already does.
        #
        # These records are dropped rather than made to veto the allele, so any
        # other submission still counts. No record in the current exports
        # carries this value, so the filter changes no present label; it is a
        # guard for the next ClinVar refresh. See
        # experiments/clinvar_conflict_audit/.
        cs = [c for c in cs if "conflicting" not in c]
        if any("pathogenic" in c and "benign" not in c for c in cs):
            return "Pathogenic"
        if any("benign" in c for c in cs):
            return "Benign"
        return "Unknown"

    if not pairs:
        return set(), set()

    cv_map = (
        pd.DataFrame(pairs, columns=["allele", "classification"])
        .groupby("allele")["classification"]
        .apply(_consensus)
    )
    driver = set(cv_map[cv_map == "Pathogenic"].index)
    passenger = set(cv_map[cv_map == "Benign"].index)
    return driver, passenger


def _get_clinvar_sets(cfg, gene=None):
    """Lazily load + cache the driver/passenger sets on the cfg dict (once per process).

    `gene` selects which per-gene ClinVar csv to load (config's "clinvar_csv_by_gene");
    omit it for proteins with a single flat "clinvar_csv".
    """
    cache_key = "_clinvar_sets" if gene is None else f"_clinvar_sets_{gene}"
    if cache_key not in cfg:
        csv_path = cfg["clinvar_csv_by_gene"][gene] if gene is not None else cfg["clinvar_csv"]
        target_gene = gene if gene is not None else cfg.get("clinvar_gene")
        cfg[cache_key] = load_clinvar_driver_passenger(csv_path, target_gene=target_gene)
    return cfg[cache_key]


def extract_gene(value, genes):
    """Finds which of `genes` (e.g. ["MSH2", "MSH6"]) appears in the artifact name."""
    upper = str(value).upper()
    for gene in genes:
        if gene in upper:
            return gene
    return None


def load_csv_label_lookup(csv_path, variant_col="Variant", label_col="Label"):
    """POLE's Driver(1)/Passenger(0) labels are precomputed, not derived from raw ClinVar
    (see Ground_Truths/POLE/combined_pole_labels.csv). Keyed by uppercased allele string.
    """
    df = pd.read_csv(csv_path)
    return {str(v).upper(): int(l) for v, l in zip(df[variant_col], df[label_col])}


def _get_csv_label_lookup(cfg):
    if "_csv_label_lookup" not in cfg:
        cfg["_csv_label_lookup"] = load_csv_label_lookup(cfg["labels_csv"])
    return cfg["_csv_label_lookup"]


def classify_allele_or_artifact(value, cfg):
    """Dispatches to whichever labeling scheme this protein's config uses:
    - hotspot_range (FOXA1): Driver/Passenger purely from mutated-residue index.
    - clinvar (single flat list, KRAS): Driver/Passenger from ClinVar pathogenic/benign lists.
    - clinvar_by_gene (MutS_alpha): same as clinvar, but the ClinVar list is selected
      per-gene since MSH2/MSH6 mutation codes use independent numbering.
    - csv_lookup (POLE): precomputed Variant->Label(1/0) csv.
    - continuous_assay (TP53, KRAS): NO binary label. Their assays are continuous
      targets only (analyses/continuous/continuous_functional); the former percentile
      (TP53) and LFC-cutoff (KRAS) rules were deleted on 2026-10-13, so every
      variant is UNK here.
    """
    if is_wt_artifact(value):
        return "WT"
    allele = str(artifact_to_allele(value)).upper()

    if cfg["label_mode"] == "hotspot_range":
        m = re.match(r"[A-Z](\d+)[A-Z]$", allele)
        if not m:
            return "UNK"
        idx = int(m.group(1))
        lo, hi = cfg["hotspot_range"]
        return "Driver" if lo <= idx <= hi else "Passenger"

    if cfg["label_mode"] == "clinvar":
        driver_set, passenger_set = _get_clinvar_sets(cfg)
        if allele in driver_set:
            return "Driver"
        if allele in passenger_set:
            return "Passenger"
        return "UNK"

    if cfg["label_mode"] == "clinvar_by_gene":
        gene = extract_gene(value, list(cfg["clinvar_csv_by_gene"]))
        if gene is None:
            return "UNK"
        driver_set, passenger_set = _get_clinvar_sets(cfg, gene=gene)
        if allele in driver_set:
            return "Driver"
        if allele in passenger_set:
            return "Passenger"
        return "UNK"

    if cfg["label_mode"] == "csv_lookup":
        label = _get_csv_label_lookup(cfg).get(allele)
        if label is None:
            return "UNK"
        return "Driver" if label == 1 else "Passenger"

    if cfg["label_mode"] == "continuous_assay":
        return "UNK"

    raise ValueError(f"unknown label_mode: {cfg['label_mode']!r}")
