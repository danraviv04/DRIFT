"""Q15: position-clustered bootstrap for the binary (annotation) intervals.

Supervisor question Q15 asks whether the binary ROC-AUC and AlphaMissense-
combination intervals should resample whole residue positions, as the
functional-correlation intervals already do. This recomputes the intervals the
main text quotes with exactly the manuscript's estimators, changing ONLY the
resampling unit from a variant to a residue position (all substitutions at a
position kept together). No new predictions; the inputs are the corrected
POLE-WT caches the publication tables were built from (fingerprint
ebbbda31758d, experiments/pole_wt_repair/cache).

Two estimators, both from Methods ("Statistical analysis and uncertainty"):

1. KS-score AUC per gene (hierarchical; `ks_auc_ci/ks_auc_ci.py`):
   class-stratified variant resample, the same resample applied to every
   descriptor seed, the ten seeds resampled, per-seed AUCs averaged.
   Clustered analogue: positions are stratified by their label content
   (driver-only, passenger-only, mixed) and resampled within each stratum, so
   class balance is held as closely as clustering allows and mixed-label
   positions are kept whole. An unstratified-position version is a sensitivity.

2. LOGO combination minus AM (paired; `simple_fusion_benchmark/annotation_tables.bootstrap`):
   per gene and draw, one of the ten seeds is picked and variants are resampled
   without class stratification; AM and every combination are scored on the same
   draw; standardisation and operating points are held fixed; one-class draws
   are dropped. Clustered analogue: positions resampled instead of variants.

Each clustered function, given singleton clusters, consumes the RNG exactly as
the published code does, so the "variant" rows here must reproduce the
published intervals bit for bit; the script checks this before writing the
clustered results.

    python -I experiments/position_cluster_bootstrap/run.py [--reps 2000 10000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# PCB_* redirect a rerun on other caches (experiments/annotation_stages); unset = as published.
# PCB_REPAIR is a folder laid out like pole_wt_repair: publication/annotation_tables/,
# and cache/ + cache_migration.csv unless PCB_CACHE names the cache folder.
REPAIR = Path(os.environ.get("PCB_REPAIR", ROOT / "experiments" / "pole_wt_repair"))
CACHE = Path(os.environ.get("PCB_CACHE", REPAIR / "cache"))
FP = os.environ.get("PCB_FP", "ebbbda31758d")
os.environ.setdefault("DRIFT_RUN_TAG", "v6_pole_wt_20261018")
os.environ["DRIFT_DEVICE"] = "cpu"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.path[:0] = [str(ROOT)]

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GENES = ("POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA")
SEEDS = tuple(range(42, 52))
SIZE = 90
KS_RNG_SEED = 20261015          # ks_auc_ci_v6.py
COMBO_RNG_SEED = 0              # annotation_tables.build default
PUBLISHED_KS = Path(os.environ.get("PCB_PUBLISHED_KS", ROOT / "experiments/ks_auc_ci/results/ks_auc_ci_v6.csv"))
PUBLISHED_COMBO = REPAIR / "publication/annotation_tables/bootstrap_intervals.csv"
MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")
STATUS = Path(os.environ.get("PCB_STATUS", HERE / "STATUS.log"))


def log(message):
    line = f"{datetime.now():%H:%M:%S} {message}"
    print(line, flush=True)
    with STATUS.open("a") as handle:
        handle.write(line + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def position_of(mutation):
    match = MUTATION.match(mutation)
    if match is None:
        raise ValueError(f"cannot parse a residue position from {mutation!r}")
    return int(match.group(2))


def clusters_of(mutations, singleton=False):
    """Member index arrays, one per residue position, in first-seen order."""
    if singleton:
        return [np.array([i]) for i in range(len(mutations))]
    positions = [position_of(m) for m in mutations]
    order = {}
    for i, p in enumerate(positions):
        order.setdefault(p, []).append(i)
    return [np.array(v) for v in order.values()]


# ---------------------------------------------------------------- KS-score AUC
def ks_matrices():
    """Variant x seed KS matrices, built exactly as ks_auc_ci_v6.py builds them."""
    frames = [pd.read_csv(CACHE / f"labeled_size{SIZE}_seed{s}_ratio_of_means_proteins{FP}_esfeat6.csv")
              .assign(seed=s) for s in SEEDS]
    d = pd.concat(frames)
    d = d[d.gene.isin(GENES)]
    out = {}
    for g in GENES:
        w = d[d.gene == g].pivot_table(index=["mutation", "label"], columns="seed",
                                       values="plddt_ks_residue").dropna()
        out[g] = (w.index.get_level_values("mutation").to_numpy(),
                  w.index.get_level_values("label").to_numpy().astype(int),
                  w.to_numpy())
    return out


def ks_strata(y, clusters, stratified):
    """Cluster groups to resample independently: driver-only, passenger-only, mixed."""
    if not stratified:
        return [list(range(len(clusters)))]
    kinds = {"pos": [], "neg": [], "mixed": []}
    for c, members in enumerate(clusters):
        labels = set(y[members])
        kinds["pos" if labels == {1} else "neg" if labels == {0} else "mixed"].append(c)
    return [kinds[k] for k in ("pos", "neg", "mixed") if kinds[k]]


def ks_bootstrap(matrices, reps, rng, *, singleton, stratified):
    rows, draws = [], {}
    for g in GENES:
        mutations, y, X = matrices[g]
        clusters = clusters_of(mutations, singleton=singleton)
        strata = [np.array(s) for s in ks_strata(y, clusters, stratified)]
        obs = np.mean([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
        boot = np.full(reps, np.nan)
        for b in range(reps):
            # rng.choice(array, n) == array[rng.integers(0, len(array), n)], so with
            # singleton clusters this is ks_auc_ci_v6's r_[choice(pos), choice(neg)].
            picked = np.concatenate([rng.choice(s, s.size) for s in strata])
            idx = np.concatenate([clusters[c] for c in picked])
            cols = rng.integers(0, X.shape[1], X.shape[1])
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            boot[b] = np.mean([roc_auc_score(yy, X[idx, j]) for j in cols])
        ok = boot[np.isfinite(boot)]
        lo, hi = np.percentile(ok, [2.5, 97.5])
        n_mixed = sum(len(set(y[c])) > 1 for c in clusters_of(mutations))
        rows.append(dict(gene=g, n=len(y), pos=int(y.sum()), neg=int((y == 0).sum()),
                         positions=len(clusters_of(mutations)), mixed_positions=n_mixed,
                         auc=obs, lo=lo, hi=hi, width=hi - lo,
                         valid_reps=int(ok.size), reps=reps))
        draws[g] = boot
    return pd.DataFrame(rows), draws


# -------------------------------------------------- combination minus AM (paired)
def combo_inputs():
    from experiments.simple_fusion_benchmark import annotation_tables, size_sweep
    replays, order = {}, None
    for seed in SEEDS:
        _, replay = annotation_tables.per_seed_metrics(SIZE, seed, "annotation")
        frame, path = size_sweep.load_cell(SIZE, seed, "annotation")
        assert path.parent == CACHE and FP in path.name, path
        mutations = {g: frame.mutation[frame.gene == g].to_numpy() for g in frame.gene.unique()}
        if order is None:
            order = mutations
        assert all((order[g] == mutations[g]).all() for g in order), "row order differs across seeds"
        replays[seed] = replay
        log(f"  combination inputs: seed {seed}")
    return replays, order


def combo_bootstrap(replays, seeds, columns, draws, rng, mutations, *, singleton):
    """annotation_tables.bootstrap with the variant index replaced by a position draw.

    Identical control flow and RNG calls; with singleton clusters
    `rng.integers(0, k, k)` is the published `rng.integers(0, n, n)`.
    """
    from experiments.simple_fusion_benchmark.annotation_tables import METHODS, REFERENCE, _metrics_at
    genes = sorted({gene for (_, gene) in replays[seeds[0]]})
    out = {}
    macro = {model: {c: np.zeros(draws) for c in columns} for model in METHODS}
    for gene in genes:
        clusters = clusters_of(mutations[gene], singleton=singleton)
        k = len(clusters)
        per_model = {model: {c: np.zeros(draws) for c in columns} for model in METHODS}
        for draw in range(draws):
            seed = seeds[rng.integers(len(seeds))]
            picks = rng.integers(0, k, k)
            index = np.concatenate([clusters[p] for p in picks])
            reference = replays[seed][(REFERENCE, gene)]
            if len(set(reference["y"][index])) < 2:
                for model in METHODS:
                    for c in columns:
                        per_model[model][c][draw] = np.nan
                continue
            base = _metrics_at(reference["y"][index], reference["score"][index],
                               reference["cut"], reference["is_probability"])
            for model in METHODS:
                cell = replays[seed][(model, gene)]
                got = _metrics_at(cell["y"][index], cell["score"][index],
                                  cell["cut"], cell["is_probability"])
                for c in columns:
                    per_model[model][c][draw] = got.get(c, np.nan) - base.get(c, np.nan)
        for model in METHODS:
            for c in columns:
                out[(model, gene, c)] = per_model[model][c]
                macro[model][c] += per_model[model][c] / len(genes)
        log(f"    {gene}: {k} clusters, {draws} draws")
    for model in METHODS:
        for c in columns:
            out[(model, "MACRO", c)] = macro[model][c]
    return out


# ---------------------------------------------------------------------- checks
def check_ks_reproduction(variant):
    published = pd.read_csv(PUBLISHED_KS).set_index("gene")
    rows = []
    for r in variant.itertuples():
        p = published.loc[r.gene]
        rows.append(dict(gene=r.gene, auc=r.auc, published_auc=p.auc, lo=r.lo, published_lo=p.lo,
                         hi=r.hi, published_hi=p.hi,
                         max_abs_diff=max(abs(r.auc - p.auc), abs(r.lo - p.lo), abs(r.hi - p.hi))))
    table = pd.DataFrame(rows)
    # The published file read the pooled-POLE caches (d61d29328362); every other
    # gene's inputs are bit-identical in the corrected caches, and POLE consumes the
    # RNG identically (same class counts), so non-POLE genes must match exactly.
    others = table[table.gene != "POLE"]
    assert (others.max_abs_diff < 1e-12).all(), f"KS reproduction failed:\n{others}"
    return table


def check_combo_reproduction(variant):
    published = pd.read_csv(PUBLISHED_COMBO)
    merged = variant.merge(published, on=["model", "gene", "metric"], suffixes=("", "_published"))
    assert len(merged) == len(published) == len(variant), (len(merged), len(published), len(variant))
    diff = np.nanmax(np.abs(np.r_[merged.lo - merged.lo_published, merged.hi - merged.hi_published]))
    both_nan = (merged.lo.isna() == merged.lo_published.isna()).all()
    assert diff < 1e-12 and both_nan, f"combination reproduction failed: max diff {diff}"
    return float(diff)


# ---------------------------------------------------------------------- report
MAIN_TEXT = [  # (label, kind, key) for every binary interval quoted in the main text
    ("MSH6 KS-score AUC", "ks", "MSH6"),
    ("PIK3CA KS-score AUC", "ks", "PIK3CA"),
    ("Macro combination - AM, ROC-AUC", "combo", ("gz_mean", "MACRO", "roc_auc")),
    ("Macro combination - AM, AP", "combo", ("gz_mean", "MACRO", "average_precision")),
    ("Macro combination - AM, prec@R", "combo", ("gz_mean", "MACRO", "precision_at_R")),
    ("POLE combination - AM, ROC-AUC", "combo", ("gz_mean", "POLE", "roc_auc")),
]


def main_text_table(ks, combo, combo_point):
    rows = []
    for label, kind, key in MAIN_TEXT:
        for (scheme, reps), table in (ks if kind == "ks" else combo).items():
            if kind == "ks":
                r = table.set_index("gene").loc[key]
                point = r.auc
            else:
                r = table.set_index(["model", "gene", "metric"]).loc[key]
                point = combo_point[key]
            rows.append(dict(quantity=label, scheme=scheme, reps=reps, estimate=point,
                             lo=r.lo, hi=r.hi, width=r.hi - r.lo,
                             excludes_zero=(kind == "combo") and bool(r.lo > 0 or r.hi < 0)))
    out = pd.DataFrame(rows)
    ref = out[out.scheme == "variant"].set_index(["quantity", "reps"]).width
    out["width_vs_variant"] = [w / ref.get((q, n), np.nan) for q, n, w in
                               zip(out.quantity, out.reps, out.width)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, nargs="+", default=[2000, 10000],
                    help="2000 is the manuscript's count; larger counts check Monte Carlo error")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log(f"start: reps={args.reps} out={args.out}")

    from drift.protein_config import protein_config_fingerprint
    assert protein_config_fingerprint() == FP, "scientific config changed; revalidate first"
    if (REPAIR / "cache_migration.csv").exists():   # a patched cache: check it is the recorded one
        migration = pd.read_csv(REPAIR / "cache_migration.csv")
        for row in migration.itertuples():
            assert Path(row.new_path).parent == CACHE and sha(row.new_path) == row.new_sha256, row.new_path
    from drift import run
    from experiments.simple_fusion_benchmark import size_sweep, annotation_tables
    run.CACHE_DIR = CACHE
    size_sweep.CACHE = CACHE

    # --- KS-score AUC
    matrices = ks_matrices()
    ks = {}
    for reps in args.reps:
        for scheme, singleton, stratified in (("variant", True, True),
                                              ("position_stratified", False, True),
                                              ("position_unstratified", False, False)):
            t0 = time.time()
            table, draws = ks_bootstrap(matrices, reps, np.random.default_rng(KS_RNG_SEED),
                                        singleton=singleton, stratified=stratified)
            table.insert(0, "scheme", scheme)
            ks[(scheme, reps)] = table
            np.savez_compressed(args.out / f"ks_draws_{scheme}_{reps}.npz", **draws)
            log(f"KS {scheme} reps={reps}: {time.time() - t0:.0f}s")
            if scheme == "variant" and reps == 2000:
                check = check_ks_reproduction(table)
                check.to_csv(args.out / "check_ks_reproduction.csv", index=False)
                log("KS variant scheme reproduces ks_auc_ci_v6 for every non-POLE gene")
    pd.concat(ks.values(), ignore_index=True).to_csv(args.out / "ks_auc_intervals.csv", index=False)

    # --- combination minus AM
    replays, mutations = combo_inputs()
    columns = [c for c, _ in annotation_tables.COLUMNS]
    combo, combo_point = {}, {}
    mean_published = pd.read_csv(REPAIR / "publication/annotation_tables/tables_long.csv")
    for r in mean_published.itertuples():
        combo_point[(r.model, r.gene, r.metric)] = r.delta_vs_am
    for reps in args.reps:
        for scheme, singleton in (("variant", True), ("position", False)):
            t0 = time.time()
            draws = combo_bootstrap(replays, list(SEEDS), columns, reps,
                                    np.random.default_rng(COMBO_RNG_SEED), mutations,
                                    singleton=singleton)
            table = annotation_tables.summarise_bootstrap(draws)
            table.insert(0, "scheme", scheme)
            table.insert(1, "reps", reps)
            combo[(scheme, reps)] = table
            keep = {f"{m}|{g}|{c}": v for (m, g, c), v in draws.items()
                    if c in ("roc_auc", "average_precision", "precision_at_R")}
            np.savez_compressed(args.out / f"combo_draws_{scheme}_{reps}.npz", **keep)
            log(f"combination {scheme} reps={reps}: {time.time() - t0:.0f}s")
            if scheme == "variant" and reps == 2000:
                diff = check_combo_reproduction(table.drop(columns=["scheme", "reps"]))
                log(f"combination variant scheme reproduces the published intervals (max diff {diff:.1e})")
    pd.concat(combo.values(), ignore_index=True).to_csv(args.out / "combination_intervals.csv", index=False)

    # --- descriptive: how clustered are the labels?
    rows = []
    for g in GENES:
        mutations_g, y, _ = matrices[g]
        clusters = clusters_of(mutations_g)
        sizes = np.array([len(c) for c in clusters])
        rows.append(dict(gene=g, variants=len(y), positions=len(clusters),
                         variants_per_position_mean=sizes.mean(), max_per_position=sizes.max(),
                         driver_positions=sum(set(y[c]) == {1} for c in clusters),
                         passenger_positions=sum(set(y[c]) == {0} for c in clusters),
                         mixed_positions=sum(len(set(y[c])) > 1 for c in clusters),
                         drivers=int(y.sum()), passengers=int((y == 0).sum())))
    pd.DataFrame(rows).to_csv(args.out / "position_structure.csv", index=False)

    summary = main_text_table(ks, combo_published_keyed(combo), combo_point)
    summary.to_csv(args.out / "main_text_intervals.csv", index=False)
    log("main-text intervals:\n" + summary.round(4).to_string(index=False))

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    (args.out / "provenance.json").write_text(json.dumps({
        "fingerprint": FP, "cache_directory": str(CACHE),
        "cache_manifest_sha256": sha(REPAIR / "cache_migration.csv"),
        "script_sha256": sha(__file__), "git_head": head,
        "reps": args.reps, "ks_rng_seed": KS_RNG_SEED, "combination_rng_seed": COMBO_RNG_SEED,
        "published_ks": str(PUBLISHED_KS), "published_combination": str(PUBLISHED_COMBO),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(time.time() - started),
    }, indent=2) + "\n")
    log(f"done in {time.time() - started:.0f}s")


def combo_published_keyed(combo):
    return {key: table.drop(columns=["scheme", "reps"]) for key, table in combo.items()}


if __name__ == "__main__":
    main()
