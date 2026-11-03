"""Count the AF3 predictions behind the paper and measure their archives.

For the Data availability sentence "Because of their size (N structures from
M AlphaFold 3 ensembles, X TB), the predicted structures are available on
request". Read-only on $DRIFT_ARCHIVE_ROOT.

Cohort = what the paper analyses:
  * 1,798 labelled variants (canonical annotation manifest minus FOXA1 and
    POLE P356S): POLE, PTEN, VHL, MSH2, MSH6, PIK3CA
  * TP53 (1,174) and KRAS4B (935) from the continuous-functional scores
    (the PTEN 253 and MSH2 106 functional variants are a subset of the
    labelled ones; checked below)
  * one WT ensemble per system, plus the PTEN and MSH6 WT-control extensions

Every analysed ensemble has 125 predictions (25 seeds x 5 samples; SI
Table S4). On disk, POLE is plain folders (prediction dirs counted and sizes
walked); everything else is one .tar.gz per run (sizes from stat; prediction
count taken from the seeds in the name, 125 otherwise).

Writes runs.csv (one row per archive) and summary.csv next to this script.
"""

import csv
import glob
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DRIVE = Path(os.environ.get("DRIFT_ARCHIVE_ROOT", "$DRIFT_ARCHIVE_ROOT"))

MANIFEST = REPO / "drift/results/afm_fusion/canonical_annotation_v6_ca/canonical_afm_manifest.csv"
FUNCTIONAL = REPO / "experiments/continuous_functional/four_gene_ks/variant_scores.csv"

# (folder, filename patterns tried in order) per gene; {m} is the mutation
# code. POLE folders carry two naming schemes (160 POLE_268-471_*, 1,068
# POLE_*); only M294V exists under both, and the first pattern wins.
VARIANT_GLOBS = {
    "POLE": ("POLE_folders", ["POLE_268-471_{m}", "POLE_{m}"]),
    "PTEN": ("PTEN", ["PTEN_{m}_data.tar.gz"]),
    "VHL": ("VHL", ["VHL_{m}_data.tar.gz"]),
    "MSH2": ("MutSalpha", ["MutS_alpha_MSH2_{m}_data.tar.gz"]),
    "MSH6": ("MSH6_GT", ["MutS_alpha_MSH6_GT_{m}_data.tar.gz"]),
    "PIK3CA": ("PIK3CA", ["PIK3CA_{m}_data.tar.gz"]),
    "TP53": ("TP53", ["P53_{m}_with_ZN_and_DNA*.tar.gz"]),
    "KRAS": ("KRAS4B_Mutations", ["KRAS_{m}_data.tar.gz"]),
}

# WT ensembles the paper uses (protein_configs.json wt_artifact pins), with
# the number of predictions each holds.
WT_RUNS = [
    ("POLE", "POLE_folders/POLE_268-471_WT", "wt", None),
    ("PTEN", "PTEN/PTEN_WT_data.tar.gz", "wt", 125),
    ("PTEN", "PTEN/PTEN_WT_seeds26-200_data.tar.gz", "wt_control_extension", 175 * 5),
    ("VHL", "VHL/VHL_WT_data.tar.gz", "wt", 125),
    ("MSH2", "EXTRA_WT/MutS_alpha_MSH2_WT_vartmpl_data.tar.gz", "wt", 125),
    ("MSH6", "MSH6_GT/MutS_alpha_MSH6_GT_WT_data.tar.gz", "wt", 125),
    ("MSH6", "MSH6_GT/MutS_alpha_MSH6_GT_WT_seeds_26-75_data.tar.gz", "wt_control_extension", 50 * 5),
    ("PIK3CA", "EXTRA_WT/PIK3CA_WT_vartmpl_data.tar.gz", "wt", 125),
    ("TP53", "EXTRA_WT/P53_WT_vartmpl_with_ZN_and_DNA_data.tar.gz", "wt", 125),
    ("KRAS", "KRAS4B_Mutations/KRAS_WT_data.tar.gz", "wt", 125),
]

SEED_DIR = re.compile(r"^seed-\d+_sample-\d+$")


def cohort():
    rows = []
    with open(MANIFEST) as fh:
        for r in csv.DictReader(fh):
            if r["protein"] == "FOXA1":
                continue
            if r["protein"] == "POLE" and r["mutation_code"] == "P356S":
                continue  # dropped from the cohort 10-20
            rows.append((r["protein"], r["mutation_code"], "labelled"))
    labelled = {(g, m) for g, m, _ in rows}
    with open(FUNCTIONAL) as fh:
        for r in csv.DictReader(fh):
            g, m = r["gene"], r["mutation"]
            if g in ("TP53", "KRAS"):
                rows.append((g, m, "functional"))
            elif (g, m) not in labelled:
                sys.exit(f"{g} {m}: functional variant missing from the labelled cohort")
    return rows


def folder_stats(path):
    """Bytes under a POLE folder and the number of prediction dirs in it."""
    total, n_pred = 0, 0
    for root, dirs, files in os.walk(path):
        n_pred += sum(1 for d in dirs if SEED_DIR.match(d))
        for f in files:
            total += os.lstat(os.path.join(root, f)).st_size
    return total, n_pred


def measure(gene, path, role, n_pred):
    if path.is_dir():
        size, n_pred = folder_stats(path)
    else:
        size = path.stat().st_size
    return {"gene": gene, "role": role, "path": str(path), "bytes": size, "predictions_on_disk": n_pred}


def main():
    variants = cohort()
    print(f"{len(variants)} variants", flush=True)
    out, missing = [], []
    for i, (gene, mut, role) in enumerate(variants):
        folder, patterns = VARIANT_GLOBS[gene]
        for pattern in patterns:
            hits = sorted(glob.glob(str(DRIVE / folder / pattern.format(m=mut))))
            if hits:
                break
        if len(hits) != 1:
            missing.append((gene, mut, hits))
            continue
        rec = measure(gene, Path(hits[0]), role, 125)
        rec["mutation"] = mut
        rec["predictions_used"] = 125
        out.append(rec)
        if i % 250 == 0:
            print(f"  {i}/{len(variants)} {gene} {mut}", flush=True)
    for gene, rel, role, n_pred in WT_RUNS:
        rec = measure(gene, DRIVE / rel, role, n_pred)
        rec["mutation"] = "WT"
        rec["predictions_used"] = rec["predictions_on_disk"] if role != "wt" else 125
        out.append(rec)

    cols = ["gene", "mutation", "role", "predictions_used", "predictions_on_disk", "bytes", "path"]
    with open(OUT / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    summary = {}
    for r in out:
        s = summary.setdefault(r["gene"], {"gene": r["gene"], "ensembles": 0, "predictions_used": 0, "bytes": 0})
        s["ensembles"] += 1
        s["predictions_used"] += r["predictions_used"]
        s["bytes"] += r["bytes"]
    tot = {"gene": "TOTAL", "ensembles": 0, "predictions_used": 0, "bytes": 0}
    for s in summary.values():
        for k in ("ensembles", "predictions_used", "bytes"):
            tot[k] += s[k]
    rows = list(summary.values()) + [tot]
    for s in rows:
        s["GB"] = round(s["bytes"] / 1e9, 1)
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene", "ensembles", "predictions_used", "bytes", "GB"])
        w.writeheader()
        w.writerows(rows)
    for s in rows:
        print(f"{s['gene']:8s} {s['ensembles']:6d} {s['predictions_used']:9d} {s['GB']:9.1f} GB")
    print(f"missing/ambiguous: {len(missing)}")
    for m in missing[:30]:
        print("  ", m)


if __name__ == "__main__":
    main()
