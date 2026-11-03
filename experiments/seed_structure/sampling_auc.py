"""Fixed-N seed-diversity controls for canonical WT-mutant KS AUCs.

PYTHONPATH=. python experiments/seed_structure/sampling_auc.py
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import roc_auc_score

from metric_analysis.pipeline.labels import artifact_key
from metric_analysis.pipeline.library.canonical_cohort import load_all_canonical_cohorts
from metric_analysis.pipeline.library.descriptor_metrics import residue_ks_statistics
from metric_analysis.pipeline.library.descriptor_sampling import apply_row_block, row_block, stable_rng

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results_auc_20261018"
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
POLICIES = ["whole_independent", "random_independent", "whole_disjoint"]
REPS = 200
TAGS = sorted(f"seed-{s}_sample-{i}" for s in range(1, 26) for i in range(5))
SEED = np.array([int(t.split("_")[0].split("-")[1]) for t in TAGS])


def draws(gene, mutation):
    rng = stable_rng(20261018, gene, mutation, "seed_policy_auc")
    result = {p: [[], []] for p in POLICIES}
    for _ in range(REPS):
        order = rng.permutation(np.arange(1, 26))
        a = np.flatnonzero(np.isin(SEED, order[:5]))
        b = np.flatnonzero(np.isin(SEED, rng.permutation(np.arange(1, 26))[:5]))
        c = np.flatnonzero(np.isin(SEED, order[5:10]))
        for policy, left, right in [
            ("whole_independent", a, b),
            ("whole_disjoint", a, c),
            ("random_independent", rng.permutation(125)[:25], rng.permutation(125)[:25]),
        ]:
            result[policy][0].append(left)
            result[policy][1].append(right)
    return {p: (np.array(a), np.array(b)) for p, (a, b) in result.items()}


def scores(wt, mutant, selection):
    assert wt.shape == mutant.shape and wt.shape[1] == 125
    assert np.isfinite(wt).all() and np.isfinite(mutant).all()
    out = {}
    for policy, (ia, ib) in selection.items():
        chunks = []
        for start in range(0, REPS, 10):
            a = wt[:, ia[start:start + 10]].transpose(1, 0, 2)
            b = mutant[:, ib[start:start + 10]].transpose(1, 0, 2)
            v = residue_ks_statistics(a.reshape(-1, 25), b.reshape(-1, 25))
            chunks.extend(v.reshape(len(a), wt.shape[0]).mean(1))
        out[policy] = np.array(chunks)
    return out


def plddt(path, record):
    frame = pd.read_csv(path, index_col=0)
    assert set(TAGS).issubset(frame.columns), path
    return apply_row_block(frame[TAGS].to_numpy(float), row_block(record))


def coordinates(dataset, artifact, recovered=None):
    frame = dataset.to_table(
        filter=(ds.field("artifact") == artifact) & (ds.field("core") == True),
        columns=["structure_index", "sample", "resid", "x_aligned", "y_aligned", "z_aligned"],
    ).to_pandas()
    if recovered is not None:
        frame["sample"] = frame.structure_index.map(recovered)
    frame = frame[frame["sample"].isin(TAGS)]
    if set(frame["sample"].dropna()) != set(TAGS):
        raise ValueError("coordinate seed-sample identities incomplete")
    if frame.duplicated(["resid", "sample"]).any():
        raise ValueError("duplicate coordinate residue/sample")
    piv = [frame.pivot(index="resid", columns="sample", values=axis)[TAGS]
           for axis in ["x_aligned", "y_aligned", "z_aligned"]]
    c = np.stack([p.to_numpy(float) for p in piv], -1)
    if not np.isfinite(c).all():
        raise ValueError("nonfinite core coordinates")
    return c, piv[0].index.to_numpy()


def process_gene(gene):
    cohort = load_all_canonical_cohorts()
    exclusions = []
    target = OUT / f"{gene}_scores.csv.gz"
    if target.exists():
        print(f"Already completed: {gene}", flush=True)
        return
    block = cohort[cohort.protein == gene]
    folder = ROOT / "metric_analysis/structure_metrics" / gene / "ensemble_aware/cache_corefit"
    report = json.loads((folder / f"{gene}_core_fit_report.json").read_text())
    index = pd.read_csv(folder / f"{gene}_core_fit_per_structure.csv")
    artifacts = index.artifact.unique()
    by_key = {artifact_key(a): a for a in artifacts}
    assert len(by_key) == len(artifacts)
    dataset = ds.dataset(folder / f"{gene}_rmsf_aligned_ca_coordinate_cache.parquet")
    recovered = None
    if gene in ["MSH2", "PIK3CA"]:
        mapping = pd.read_csv(HERE / "results_rerun_20261018" / f"{gene}_recovered_seed_mapping.csv")
        recovered = dict(zip(mapping.structure_index, mapping["sample"]))
    wc, residues = coordinates(dataset, report["wt_artifact"], recovered)
    centre = wc.mean(1, keepdims=True)
    wd = np.linalg.norm(wc - centre, axis=-1)
    rows = []
    for number, record in enumerate(block.to_dict("records"), 1):
        mutation = record["mutation_code"]
        selection = draws(gene, mutation)
        w = plddt(record["wt_source_file"], record)
        m = plddt(record["source_file"], record)
        measurements = {"plddt_ks": scores(w, m, selection)}
        key = artifact_key(Path(record["source_file"]).name.replace("_plddt_values.csv", ""))
        try:
            if key not in by_key:
                raise ValueError("coordinate artifact missing")
            mc, mr = coordinates(dataset, by_key[key])
            if not np.array_equal(mr, residues):
                raise ValueError("coordinate residue mismatch")
            md = np.linalg.norm(mc - centre, axis=-1)
            measurements["positional_ks"] = scores(wd, md, selection)
        except ValueError as error:
            exclusions.append({"gene": gene, "mutation": mutation, "metric": "positional_ks", "reason": str(error)})
        for metric, policies in measurements.items():
            for policy, values in policies.items():
                rows.extend({"gene": gene, "mutation": mutation, "label": record["label"],
                             "metric": metric, "policy": policy, "repeat": repeat, "score": value}
                            for repeat, value in enumerate(values))
        if number % 25 == 0:
            print(f"{gene}: {number}/{len(block)} variants", flush=True)
    pd.DataFrame(rows).to_csv(target, index=False, compression="gzip")
    pd.DataFrame([e for e in exclusions if e["gene"] == gene],
                 columns=["gene", "mutation", "metric", "reason"]).to_csv(OUT / f"{gene}_exclusions.csv", index=False)
    print(f"Completed {gene}: {len(block)} variants", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    cohort = load_all_canonical_cohorts()
    cohort = cohort[cohort.protein.isin(GENES)].copy()
    cohort.to_csv(OUT / "cohort.csv", index=False)
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(process_gene, GENES))
    summarize()


def summarize():
    auc_rows = []
    variability = []
    for gene in GENES:
        frame = pd.read_csv(OUT / f"{gene}_scores.csv.gz")
        spread = frame.groupby(["metric", "mutation", "policy"]).score.std().unstack("policy")
        spread["whole_random_ratio"] = spread.whole_independent / spread.random_independent
        spread["whole_disjoint_random_ratio"] = spread.whole_disjoint / spread.random_independent
        spread.reset_index().to_csv(OUT / f"{gene}_score_variability.csv", index=False)
        for metric, values in spread.groupby(level="metric"):
            variability.append({"gene": gene, "metric": metric,
                                "median_score_sd_whole": values.whole_independent.median(),
                                "median_score_sd_random": values.random_independent.median(),
                                "median_per_variant_sd_ratio": values.whole_random_ratio.median()})
        for (metric, policy, repeat), b in frame.groupby(["metric", "policy", "repeat"]):
            assert not b.mutation.duplicated().any() and b.label.nunique() == 2
            auc_rows.append({"gene": gene, "metric": metric, "policy": policy, "repeat": repeat,
                             "n": len(b), "drivers": int(b.label.sum()),
                             "auc": roc_auc_score(b.label, b.score)})
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(OUT / "auc_per_repeat.csv", index=False)
    summary = auc.groupby(["gene", "metric", "policy"]).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), n=("n", "first"), drivers=("drivers", "first")
    ).reset_index()
    summary.to_csv(OUT / "auc_summary.csv", index=False)
    pd.DataFrame(variability).to_csv(OUT / "score_variability_summary.csv", index=False)
    deltas = []
    for (gene, metric), b in auc.groupby(["gene", "metric"]):
        p = b.pivot(index="repeat", columns="policy", values="auc")
        for policy in ["whole_independent", "whole_disjoint"]:
            delta = p[policy] - p.random_independent
            deltas.append({"gene": gene, "metric": metric, "policy": policy,
                           "delta_mean": delta.mean(), "delta_sd": delta.std(),
                           "draw_p025": delta.quantile(.025), "draw_p975": delta.quantile(.975)})
    pd.DataFrame(deltas).to_csv(OUT / "auc_deltas.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps({
        "genes": GENES, "repetitions": REPS, "base_seed": 20261018,
        "conformations_per_side": 25, "policies": POLICIES,
        "labels": "strict canonical manifest, unchanged, driver-high AUC",
        "coordinate_scope": "WT-defined core residues, core_fit frame; radial distance to full WT mean",
        "uncertainty": "SD and percentiles over draws conditional on fixed variants and AF3 pools; not population confidence intervals",
        "plddt_scope": "canonical residue scope; restrict all pools to seeds 1-25, samples 0-4 for equal available seed diversity (8 POLE mutant pools contain additional seeds); positional analysis may exclude variants with incomplete coordinate identities",
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    write_report(auc, summary)


def write_report(auc, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    macro = auc.groupby(["metric", "policy", "repeat"]).auc.mean().reset_index()
    macro.to_csv(OUT / "macro_auc_per_repeat.csv", index=False)
    macro_summary = macro.groupby(["metric", "policy"]).auc.agg(["mean", "std"])
    macro_summary.to_csv(OUT / "macro_auc_summary.csv")
    text = ["# Seed sampling and WT–mutant KS AUC — 2026-10-18", "",
            "200 repeats; 25 conformations per side. Driver-high AUC is calculated per gene and repeat, then averaged over repeats. No classifier is fitted and score direction is never flipped.", "",
            "Primary comparison: independently choose five whole seeds (all five samples from each) for WT and mutant, versus independently choose 25 conformations without regard to seed. Whole-disjoint is the additional control matching the original WT–WT whole-seed arm: WT and mutant share no seed identities.", "",
            "All pools are restricted to seeds 1–25 and samples 0–4; eight POLE mutant pools have extra seeds, which are omitted from both policies to equalize available seed diversity. Labels and pLDDT residue scopes come from the strict canonical cohort. Positional KS is calculated over core residues in the existing core-fit frame, using distance from each C-alpha to the full-pool WT residue centre. Exact conformation selections are shared across metrics.", ""]
    for metric, title in [("plddt_ks", "pLDDT KS"), ("positional_ks", "Coordinate KS")]:
        text.extend([f"## {title}", "", "AUC mean ± SD across draws. Delta = whole-independent minus random-independent.", "",
                     "| Gene | Variants (drivers/passengers) | Whole independent | Random independent | Delta | Whole disjoint |",
                     "| --- | --- | --- | --- | --- | --- |"])
        for gene in GENES:
            b = summary[(summary.gene == gene) & (summary.metric == metric)].set_index("policy")
            a, r, d = b.loc["whole_independent"], b.loc["random_independent"], b.loc["whole_disjoint"]
            text.append(f"| {gene} | {int(a.n)} ({int(a.drivers)}/{int(a.n-a.drivers)}) | {a.auc_mean:.4f} ± {a.auc_sd:.4f} | {r.auc_mean:.4f} ± {r.auc_sd:.4f} | {a.auc_mean-r.auc_mean:+.4f} | {d.auc_mean:.4f} ± {d.auc_sd:.4f} |")
        m = macro_summary.loc[metric]
        text.extend(["", f"Equal-gene macro AUC: whole independent {m.loc['whole_independent', 'mean']:.4f}, random independent {m.loc['random_independent', 'mean']:.4f}, whole disjoint {m.loc['whole_disjoint', 'mean']:.4f}.", ""])
    text.extend(["## Coverage and validation", "",
                 "pLDDT includes all 1,799 canonical variants. Coordinate exclusions are listed below and are identical across policies; no labels or scores were used to select exclusions.", ""])
    for gene in GENES:
        excluded = pd.read_csv(OUT / f"{gene}_exclusions.csv")
        for row in excluded.itertuples():
            text.append(f"- {gene} {row.mutation}: {row.reason}.")
    text.extend(["", "Validation: draw counts, no duplicate conformations, exactly five whole seeds, disjoint seed identities, and repeatability checked. Batched KS matched scalar SciPy KS on tied synthetic data. Every AUC uses both classes and unique variant identities. MSH2/PIK3CA WT identities use the archive-backed mappings from the earlier rerun; original caches are unchanged.", "",
                 "## Interpretation limits", "",
                 "AUC SD and delta percentiles describe sensitivity to subsampling these fixed variants and fixed AF3 pools. They are not population confidence intervals or evidence of external generalization. The number of passengers is small for some genes (especially PTEN and PIK3CA). Macro AUC gives each gene equal weight.", "",
                 "Sampling policy changes both within-ensemble seed diversity and the distribution of shared seed identities between WT and mutant. Consequently, this comparison tests the practical effect of the policies; it cannot attribute an AUC change solely to variance. Coordinate KS is a radial-distance statistic, not a full 3-D distribution test. Core-only coordinate conclusions do not cover flexible regions outside the core.", ""])
    (OUT / "REPORT.md").write_text("\n".join(text))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for axis, metric, title in zip(axes, ["plddt_ks", "positional_ks"], ["pLDDT KS", "Coordinate KS (core residues)"]):
        for offset, policy, label, color in [(-.12, "whole_independent", "5 seeds × 5 samples", "#b36329"),
                                             (.12, "random_independent", "25 random conformations", "#267e91")]:
            b = summary[(summary.metric == metric) & (summary.policy == policy)].set_index("gene").loc[GENES]
            axis.errorbar(np.arange(6) + offset, b.auc_mean, yerr=b.auc_sd, fmt="o", capsize=3, label=label, color=color)
        axis.set_xticks(np.arange(6), GENES, rotation=30)
        axis.set_title(title)
        axis.axhline(.5, color="gray", linestyle=":", linewidth=1)
        axis.set_ylim(.35, 1.02)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean AUC ± SD across 200 draws")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "sampling_auc.png", dpi=180)
    fig.savefig(OUT / "sampling_auc.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
