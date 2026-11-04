#!/usr/bin/env python
"""Shared ground for new SI / thesis figures.

Everything a new figure script needs so it does not re-derive plumbing, and so
every panel written here lands on the same cohort, the same style and the same
provenance record as the other figure scripts.

Usage
-----
    from figlib import *              # style is applied on import

    df = labeled()                    # strict-cohort per-variant table, N=90
    fig, ax = panel()
    ...
    save(fig, "S12a_my_panel")
    record("S12a", "median_auc", 0.707, "six evaluation genes")
    write_values()

Run any figure script with the project env:

    python figures/si/lib/<script>.py

Traps this module exists to absorb
----------------------------------
1. **Gene keys.** Since the 2026-09-07 subunit split every v5 source says MSH2 and
   MSH6. `to_cache_key` / `to_dir_key` remain only for pre-split outputs, which
   spell the merged complex ``MutSalpha`` (caches) or ``MutS_alpha`` (pipeline).
2. **Strict cohort.** `labeled()` reads the fusion cache of the CURRENT protein
   config (fingerprint-pinned, v5: e1596fb1503b) and applies the strict manifest
   (1,798 = the six annotation genes; the FOXA1 control left on 2026-11-03). Never plot
   a raw cache frame, and run `check_cohort()` on anything that filters or merges:
   it raises on a mismatch.
3. **Type 3 fonts.** Several journals bounce Type 3 outright. Set to 42 here.
4. **Import cost.** `drift.run` initialises MPS and loads the
   AlphaMissense table on import (~seconds). It is imported lazily, only when
   `labeled()` is actually called.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FUSION = REPO / "drift"
FUSION_CACHE = FUSION / "cache"
STRUCTURE_METRICS = REPO / "metric_analysis" / "structure_metrics"
MANIFESTS = REPO / "metric_analysis" / "manifests"
STRICT_MANIFEST = MANIFESTS / "strict_canonical_manifest.csv"
PLDDT_ROOT = REPO / "PLDDTs"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metric_analysis.pipeline.paths import RUN_TAG, tagged  # noqa: E402


# ── genes ─────────────────────────────────────────────────────────────────────

# The v5 cohorts (RUN_TAG v5_kras4b). Since the 2026-09-07 subunit split the
# caches, manifests and pipeline all spell MSH2/MSH6 the same way.
#: driver/passenger labels (SI Table S1a). The SI's binary analyses use these only.
ANNOTATION_GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
#: positional controls (FOXA1 until 2026-11-03; none in the release)
CONTROL_GENES = []
#: continuous assays, never thresholded (SI Table S1b)
CONTINUOUS_GENES = ["TP53", "KRAS", "PTEN", "MSH2"]
#: default gene set for the SI figures
GENES = ANNOTATION_GENES
EVALUATION_GENES = ANNOTATION_GENES
CANONICAL_PROTEINS = ANNOTATION_GENES + CONTROL_GENES

#: label route per dataset
LABEL_SOURCE = {g: "annotation" for g in ANNOTATION_GENES}

#: (variants, drivers) per gene, from SI Table S1a; total 1,798 / 612
#: (POLE P356S left the cohort on 2026-10-20: 1,075 -> 1,074)
EXPECTED_COHORT = {
    "POLE": (1074, 38), "PTEN": (257, 248), "VHL": (124, 105),
    "MSH2": (109, 71), "MSH6": (143, 66), "PIK3CA": (91, 84),
}
#: the positional controls, (variants, positives)
EXPECTED_CONTROL = {}
#: variants with a continuous measurement, from SI Table S1b
EXPECTED_CONTINUOUS = {"TP53": 1174, "KRAS": 935, "PTEN": 253, "MSH2": 106}


def to_cache_key(gene: str) -> str:
    """Pipeline/manifest spelling -> fusion-cache spelling."""
    return "MutSalpha" if gene == "MutS_alpha" else gene


def to_dir_key(gene: str) -> str:
    """Fusion-cache spelling -> pipeline/manifest spelling."""
    return "MutS_alpha" if gene == "MutSalpha" else gene


def pretty(gene: str) -> str:
    """Display label for axes and legends."""
    return {"MutSalpha": "MutSα", "MutS_alpha": "MutSα"}.get(gene, gene)


# ── style ─────────────────────────────────────────────────────────────────────

# Palette carried over from the earlier article panels so new panels sit beside
# the existing ones without a colour discontinuity.
DRV = "#d1495b"   # driver / positive
PAS = "#2a78d6"   # passenger / negative (= main Fig. 1f)
WTC = "#8d8d8d"   # WT-vs-WT background (= main Fig. 1f)
AMC = "#2e86ab"   # AlphaMissense
ACC = "#6a4c93"   # accent (fusion, combined models)
GRY = "#8d8d8d"   # null, reference, de-emphasised

# Figure 2's gene colours (figures/figure2/make_figure2.py GENE_C), so a gene
# looks the same in the main figures and the SI.
GENE_COLOR = {
    "POLE": "#0f7b6c", "MSH2": "#2a78d6", "MSH6": "#c2255c",
    "VHL": "#b58900", "PTEN": "#6a4c93", "PIK3CA": "#e8590c",
    "FOXA1": "#8d8d8d", "TP53": "#2e86ab", "KRAS": "#00798c",
}

matplotlib.rcParams.update({
    # Type 42 (TrueType), not the Type 3 default -- the most common figure bounce
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})

# Nature single column is 89 mm, double 180 mm. Sized in inches here.
MM = 1 / 25.4
W1 = 89 * MM      # single column
W15 = 120 * MM    # 1.5 column
W2 = 180 * MM     # double column


def panel(w=W1, h=None, **kw):
    """One axes at a journal-legal width. Default aspect is 4:3."""
    h = h or w * 0.75
    return plt.subplots(figsize=(w, h), **kw)


def grid(nrows=1, ncols=2, w=W2, h=None, **kw):
    """A row/grid of axes at a journal-legal width."""
    h = h or w * 0.4 * nrows
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def save(fig, name, out_dir=None):
    """Write PDF (for the manuscript) and PNG (for the HTML gallery).

    Nothing should ever cite the PNG.
    """
    out_dir = Path(out_dir or RESULTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{name}.pdf", out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)          # 300 dpi: Extended Data minimum
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(REPO)}  (+ png)")
    return pdf


def class_violin(ax, groups, *, p95=None, p95_label="WT–WT 95th pct.", seed=0,
                 title=None, ylabel=None):
    """Main Fig. 1f's class plot: violin + box + jittered points per group.

    groups: [(values, label, colour), ...], drawn left to right; the tick label
    is "<label>\n(n=<count>)". p95 draws the dashed background line with its
    label. Style is copied from analyses/msh6_case_study/wt_backgrounds_200 draw_panel, which
    makes Fig. 1f, scaled to the SI's smaller panels.
    """
    rng = np.random.default_rng(seed)
    data = [np.asarray(v, dtype=float) for v, _, _ in groups]
    parts = ax.violinplot(data, positions=range(len(data)), widths=.8, showextrema=False)
    for body, (_, _, colour) in zip(parts["bodies"], groups):
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_alpha(.2)
    for x, (vals, (_, _, colour)) in enumerate(zip(data, groups)):
        ax.boxplot([vals], positions=[x], widths=.22, showfliers=False, patch_artist=True,
                   medianprops=dict(color="black", lw=1.0),
                   boxprops=dict(facecolor="white", edgecolor=colour, lw=0.8),
                   whiskerprops=dict(color="#4a4a4a", lw=0.7), capprops=dict(color="#4a4a4a", lw=0.7))
        ax.scatter(x + rng.uniform(-.2, .2, len(vals)), vals, s=7 if len(vals) < 30 else 3,
                   color=colour, alpha=.55, edgecolor="none", zorder=3)
    if p95 is not None:
        ax.axhline(p95, ls=(0, (3, 2)), lw=.7, color=GRY, zorder=1)
        if p95_label:  # pass p95_label=None in a grid and say it once in a figure legend
            ax.text(len(data) - .55, p95, p95_label, fontsize=5.5, color=GRY, va="top", ha="right")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{lab}\n(n={len(v):,})" for v, (_, lab, _) in zip(data, groups)])
    ax.set_xlim(-.6, len(data) - .4)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


# ── quotable numbers ──────────────────────────────────────────────────────────

_VALUES: list[dict] = []


def record(panel_name, quantity, value, note=""):
    """Log a number so a caption can quote it without re-deriving it by hand."""
    _VALUES.append({"panel": panel_name, "quantity": quantity,
                    "value": value, "note": note})
    return value


def write_values(name="si_figure_values.csv", out_dir=None):
    if not _VALUES:
        return None
    out = Path(out_dir or RESULTS) / name
    pd.DataFrame(_VALUES).to_csv(out, index=False)
    print(f"  wrote {out.relative_to(REPO)}  ({len(_VALUES)} values)")
    return out


def provenance(out_dir=None, **extra):
    """Stamp what produced these figures.

    Every SI number needs script + source + date + which side of the strict
    cohort cutover it sits on. This writes that once per run.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = "unknown", None
    stamp = {
        "run_date": date.today().isoformat(),
        "script": str(Path(sys.argv[0]).resolve().relative_to(REPO))
                  if sys.argv and sys.argv[0] else "interactive",
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "python": sys.version.split()[0],
        "cohort_manifest": str(STRICT_MANIFEST.relative_to(REPO)),
        "cohort_n": int(len(pd.read_csv(STRICT_MANIFEST))),
        "strict_cutover": "2026-08-31 14:04 -- results older than this are PRE",
        **extra,
    }
    out = Path(out_dir or RESULTS) / "run_metadata.json"
    out.write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"  wrote {out.relative_to(REPO)}")
    return stamp


# ── data ──────────────────────────────────────────────────────────────────────

def labeled(size=90, seed=42, genes=None):
    """Strict-cohort per-variant table at one ensemble size and seed.

    Columns include: gene, mutation, label, afm_score,
    plddt_spread_ratio, plddt_wasserstein, plddt_ks_residue, plddt_ks_centred.

    The cache is the CURRENT protein config's (fingerprint-pinned), default
    sampling policy, newest descriptor version. Picking the newest file by mtime
    instead could load an older config's cache. ``genes`` filters the rows
    (default: every gene in the cache); pass ``ANNOTATION_GENES`` for the
    1,798-variant annotation cohort.
    """
    from drift.protein_config import protein_config_fingerprint
    from drift.run import _apply_strict_canonical_cohort  # lazy import
    fp = f"proteins{protein_config_fingerprint()}"
    hits = [p for p in FUSION_CACHE.glob(f"labeled_size{size}_seed{seed}_*{fp}_esfeat*.csv")
            if "nested-" not in p.name and "top_confidence" not in p.name]
    if not hits:
        raise SystemExit(
            f"no default-policy cache for size={size} seed={seed} {fp} under {FUSION_CACHE}")
    esfeat = lambda p: int(p.stem.rsplit("esfeat", 1)[1])
    frame = _apply_strict_canonical_cohort(pd.read_csv(max(hits, key=esfeat)))
    if genes is not None:
        frame = frame[frame["gene"].isin(genes)].reset_index(drop=True)
    return frame


def cohort(protein=None, require_files=True):
    """Canonical cohort with EXACT mutant and WT pLDDT paths, labels, provenance.

    This is the authoritative route. Never reconstruct mutation keys from
    filenames -- that is what dropped all eight labelled ``POLE_268-471_*``
    positives.

    Gene keys are the pipeline spelling (``MutS_alpha``).
    """
    from metric_analysis.pipeline.library.canonical_cohort import (
        load_canonical_cohort, load_all_canonical_cohorts)
    if protein is None:
        return load_all_canonical_cohorts(require_files=require_files)
    return load_canonical_cohort(to_dir_key(protein), require_files=require_files)


def plddt_matrix(path) -> np.ndarray:
    """Per-residue pLDDT for one ensemble: (n_residues, n_structures).

    Columns are ``seed-<s>_sample-<i>``; the index is residue position.
    """
    return pd.read_csv(path, index_col=0).to_numpy(dtype=float)


def plddt_frame(path) -> pd.DataFrame:
    """Same, keeping residue index and structure column names."""
    return pd.read_csv(path, index_col=0)


#: USalign-frame caches of the subunits lived under their modelled complex
_USALIGN_DIR = {"MSH2": "MutS_alpha", "MSH6": "MutS_alpha_MSH6_GT"}


def coord_cache_path(protein, frame="core_fit"):
    """Path of a gene's C-alpha coordinate cache.

    frame="core_fit" (default since 2026-10-16): every structure re-fitted by
    Kabsch onto one WT-defined rigid core per gene
    (metric_analysis/pipeline/cache_builders/subunit_core_fit.py; the v6
    coordinate results, analyses/coordinates/coord_core_fit/REPORT.md). MSH2 and MSH6
    each have their own cache, fitted on their own chain.
    frame="usalign": the old caches, USalign-superposed on the first chain (MSH6
    then sits in MSH2's frame). Kept as the frame control only.
    """
    gene = to_dir_key(protein)
    if frame == "core_fit":
        return (STRUCTURE_METRICS / gene / "ensemble_aware" / "cache_corefit"
                / f"{gene}_rmsf_aligned_ca_coordinate_cache.parquet")
    if frame == "usalign":
        d = _USALIGN_DIR.get(gene, gene)
        return (STRUCTURE_METRICS / d / "ensemble_aware" / "cache"
                / f"{d}_rmsf_aligned_ca_coordinate_cache.parquet")
    raise ValueError(f"frame must be 'core_fit' or 'usalign', not {frame!r}")


def coord_fit_report(protein):
    """The core-fit report (WT artifact, chain, pLDDT rows, core) of a gene."""
    gene = to_dir_key(protein)
    p = (STRUCTURE_METRICS / gene / "ensemble_aware" / "cache_corefit"
         / f"{gene}_core_fit_report.json")
    return json.loads(p.read_text())


def coords(protein, artifact=None, columns=None, frame="core_fit"):
    """C-alpha coordinates of a gene, in the WT rigid-core frame by default.

    The full table is tens of millions of rows, so pass ``artifact`` (e.g.
    ``"PTEN_WT_data"``) to push the filter into the parquet reader. Core-fit
    columns: artifact, structure_index, sample, chain, resid, icode, resname,
    x_aligned, y_aligned, z_aligned, core. See coord_cache_path for the frames.
    """
    p = coord_cache_path(protein, frame)
    if not p.is_file():
        raise FileNotFoundError(f"no {frame} coordinate cache for {protein}: {p}")
    filters = [("artifact", "==", artifact)] if artifact else None
    return pd.read_parquet(p, filters=filters, columns=columns)


class CohortMismatch(AssertionError):
    """A figure's frame does not hold the documented v5 cohort."""


def check_cohort(df=None, gene_col="gene", cohort="annotation", verbose=True):
    """Assert a frame holds exactly the documented v5 cohort; raise if not.

    cohort="annotation": the six annotation genes, variants and drivers
    (Table S1a, 1,798 / 612); rows of any other gene are an error, so filter the
    FOXA1 control out first (``labeled(genes=ANNOTATION_GENES)``).
    cohort="continuous": variants per assay (Table S1b); no labels.
    Call it in every figure that filters or merges, so a silent cohort change
    fails instead of shipping as a wrong number.
    """
    if cohort not in ("annotation", "continuous"):
        raise ValueError(f"cohort must be 'annotation' or 'continuous', not {cohort!r}")
    if df is None:
        if cohort != "annotation":
            raise ValueError("pass the frame to check for the continuous cohort")
        df = labeled(genes=ANNOTATION_GENES)
    expected = (EXPECTED_COHORT if cohort == "annotation"
                else {g: (n, None) for g, n in EXPECTED_CONTINUOUS.items()})
    genes = df[gene_col].astype(str)
    rows = []
    for gene, (exp_n, exp_pos) in expected.items():
        block = df[genes.eq(gene)]
        n = len(block)
        pos = int(block["label"].sum()) if exp_pos is not None and "label" in block else None
        good = n == exp_n and (exp_pos is None or pos == exp_pos)
        rows.append({"gene": gene, "n": n, "expected_n": exp_n,
                     "positives": pos, "expected_positives": exp_pos, "ok": good})
    extra = sorted(set(genes) - set(expected))
    table = pd.DataFrame(rows)
    ok = bool(table["ok"].all()) and not extra
    if verbose or not ok:
        print(table.to_string(index=False))
        exp_total = sum(n for n, _ in expected.values())
        print(f"  {cohort}: total {len(df)} (expected {exp_total})"
              + (f"; unexpected genes {extra}" if extra else "")
              + f"  ->  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise CohortMismatch(f"{cohort} cohort mismatch"
                             + (f"; unexpected genes {extra}" if extra else ""))
    return table


__all__ = [
    "np", "pd", "plt", "matplotlib",
    "REPO", "HERE", "RESULTS", "FUSION", "FUSION_CACHE",
    "STRUCTURE_METRICS", "MANIFESTS", "STRICT_MANIFEST", "PLDDT_ROOT",
    "RUN_TAG", "tagged",
    "GENES", "EVALUATION_GENES", "CANONICAL_PROTEINS", "LABEL_SOURCE",
    "ANNOTATION_GENES", "CONTROL_GENES", "CONTINUOUS_GENES",
    "EXPECTED_COHORT", "EXPECTED_CONTROL", "EXPECTED_CONTINUOUS", "CohortMismatch",
    "to_cache_key", "to_dir_key", "pretty",
    "DRV", "PAS", "WTC", "AMC", "ACC", "GRY", "GENE_COLOR", "class_violin",
    "MM", "W1", "W15", "W2", "panel", "grid", "save",
    "record", "write_values", "provenance",
    "labeled", "cohort", "plddt_matrix", "plddt_frame",
    "coords", "coord_cache_path", "coord_fit_report", "check_cohort",
]
