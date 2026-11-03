"""Position-cluster uncertainty, score precision and tie-aware rank/pair stability."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from plateau_run import OUT, ROOT, GENES, POLICIES, METRICS, SIZES, REPS

BOOT = 2000
CONSISTENCY = .95


def rank_auc(values, labels):
    ranks = rankdata(values, axis=0, method="average")
    nd = int(labels.sum())
    auc = (ranks[labels == 1].sum(0) - nd * (nd + 1) / 2) / (nd * (len(labels) - nd))
    return ranks, auc


def rank_agreement(ranks, reference):
    centred = ranks - ranks.mean(0)
    norm = np.sqrt((centred ** 2).sum(0))
    valid = norm > 0
    z = centred[:, valid] / norm[valid]
    corr = z.T @ z
    pairs = corr[np.triu_indices(corr.shape[0], 1)]
    ref = rankdata(reference)
    ref -= ref.mean()
    refnorm = np.linalg.norm(ref)
    agreement = z.T @ (ref / refnorm) if refnorm > 0 else np.array([])
    return {"rank_rho_between_draws": float(np.mean(pairs)) if len(pairs) else np.nan,
            "rank_rho_between_draws_p05": float(np.quantile(pairs, .05)) if len(pairs) else np.nan,
            "rank_rho_to_full": float(np.mean(agreement)) if len(agreement) else np.nan,
            "valid_rank_draws": int(valid.sum())}


def pair_probabilities(values, labels):
    a, b = values[labels == 1], values[labels == 0]
    correct = np.zeros((len(a), len(b)))
    ties = np.zeros_like(correct)
    for repeat in range(values.shape[1]):
        difference = a[:, repeat, None] - b[None, :, repeat]
        correct += difference > 0
        ties += difference == 0
    correct /= values.shape[1]
    ties /= values.shape[1]
    return correct, ties


def cluster_setup(mutations, labels, rng):
    positions = pd.Series(mutations).str.extract(r"(\d+)")[0].astype(int).to_numpy()
    unique, inverse = np.unique(positions, return_inverse=True)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=BOOT)
    driver_counts = np.bincount(inverse[labels == 1], minlength=len(unique))
    passenger_counts = np.bincount(inverse[labels == 0], minlength=len(unique))
    denominator = (weights @ driver_counts) * (weights @ passenger_counts)
    return inverse, weights, denominator


def cluster_bootstrap(probability, labels, setup):
    inverse, weights, denominator = setup
    matrix = np.zeros((weights.shape[1], weights.shape[1]))
    np.add.at(matrix, (inverse[labels == 1, None], inverse[None, labels == 0]), probability)
    numerator = np.einsum("bi,ij,bj->b", weights, matrix, weights, optimize=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator
    values[denominator == 0] = np.nan
    return values


def summarize_tensor(gene, mutations, labels, tensor, policies, sizes, destination):
    """tensor shape variants x metrics x policies x sizes x draws."""
    rows, auc_draws, pair_arrays, variant_rows = [], [], {}, []
    setup = cluster_setup(mutations, labels, np.random.default_rng(20261018))
    cohort = pd.read_csv(OUT / "cohort.csv")
    residue_counts = [int(cohort[cohort.protein == gene].iloc[0].n_wt_residues),
                      json.loads((ROOT / f"metric_analysis/structure_metrics/{gene}/ensemble_aware/cache_corefit/{gene}_core_fit_report.json").read_text())["core_residues"]]
    for metric_index, metric in enumerate(METRICS):
        valid = np.isfinite(tensor[:, metric_index]).all((1, 2, 3))
        y, names = labels[valid], np.asarray(mutations, dtype=str)[valid]
        if not valid.any() or np.unique(y).size != 2:
            continue
        if not valid.all():
            current_setup = cluster_setup(names, y, np.random.default_rng(20261018))
        else:
            current_setup = setup
        full = tensor[valid, metric_index, 0, sizes.index(125), 0]
        # Equal-N mean residue KS is an integer multiple of 1/(N*R).
        # Restore exact ties at the scalar-backend full-pool endpoint;
        # resampled scores already use integer ECDF counts before division.
        exact_full = np.rint(full * 125 * residue_counts[metric_index]) / (125 * residue_counts[metric_index])
        np.testing.assert_allclose(full, exact_full, atol=1e-14, rtol=0)
        full = exact_full
        cf, tf = pair_probabilities(full[:, None], y)
        full_boot = cluster_bootstrap(cf + .5 * tf, y, current_setup)
        for policy_index, policy in enumerate(policies):
            for j, size in enumerate(sizes):
                values = tensor[valid, metric_index, policy_index, j]
                if size == 125:
                    values = np.broadcast_to(full[:, None], values.shape)
                ranks, auc = rank_auc(values, y)
                correct, ties = pair_probabilities(values, y)
                probability = correct + .5 * ties
                np.testing.assert_allclose(probability.mean(), auc.mean(), atol=1e-14)
                boot = cluster_bootstrap(probability, y, current_setup)
                delta = full_boot - boot
                finite = np.isfinite(delta)
                sd = values.std(1, ddof=1)
                means = values.mean(1)
                rank_means = ranks.mean(1)
                rank_sd = ranks.std(1,ddof=1)
                variant_rows.extend({"gene":gene,"metric":metric,"policy":policy,"size":size,
                                     "mutation":name,"label":int(y[k]),"score_mean":float(means[k]),
                                     "score_sd":float(sd[k]),"mean_variant_rank":float(rank_means[k]),
                                     "variant_rank_sd":float(rank_sd[k]),"full_pool_score":float(full[k])}
                                    for k,name in enumerate(names))
                row = {"gene": gene, "metric": metric, "policy": policy, "size": size,
                       "n": len(y), "drivers": int(y.sum()), "positions": current_setup[1].shape[1],
                       "score_sd_median": float(np.median(sd)), "score_sd_q25": float(np.quantile(sd, .25)),
                       "score_sd_q75": float(np.quantile(sd, .75)),
                       "score_sd_driver_median": float(np.median(sd[y == 1])),
                       "score_sd_passenger_median": float(np.median(sd[y == 0])),
                       "auc_mean": float(auc.mean()), "auc_sd": float(auc.std(ddof=1)),
                       "gain_to_full_mean": float((cf + .5 * tf).mean() - auc.mean()),
                       "gain_to_full_cluster_low": float(np.nanquantile(delta, .025)),
                       "gain_to_full_cluster_high": float(np.nanquantile(delta, .975)),
                       "invalid_bootstrap_draws": int((~finite).sum()),
                       "bootstrap_low_first1000": float(np.nanquantile(delta[:1000], .025)),
                       "bootstrap_high_first1000": float(np.nanquantile(delta[:1000], .975)),
                       "fraction_pairs_consistently_correct": float((correct >= CONSISTENCY).mean()),
                       "fraction_pairs_consistently_reversed": float(((1-correct-ties) >= CONSISTENCY).mean()),
                       "pair_tie_probability_mean": float(ties.mean()),
                       **rank_agreement(ranks, full)}
                rows.append(row)
                auc_draws.extend({"gene": gene, "metric": metric, "policy": policy, "size": size,
                                  "repeat": repeat, "auc": value} for repeat, value in enumerate(auc))
                token = f"{metric}__{policy}__N{size}"
                pair_arrays[token + "__correct"] = correct.astype(np.float32)
                pair_arrays[token + "__ties"] = ties.astype(np.float32)
                pair_arrays[token + "__bootstrap_auc"] = boot
            pair_arrays[f"{metric}__drivers"] = names[y == 1]
            pair_arrays[f"{metric}__passengers"] = names[y == 0]
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / f"{gene}_summary.csv", index=False)
    pd.DataFrame(auc_draws).to_csv(destination / f"{gene}_auc_draws.csv.gz", index=False)
    pd.DataFrame(variant_rows).to_csv(destination / f"{gene}_variant_precision.csv.gz",index=False)
    np.savez_compressed(destination / f"{gene}_pair_ordering.npz", **pair_arrays)
    return rows


def main():
    summary_dir = OUT / "summary"
    summary_dir.mkdir(exist_ok=True)
    all_rows = []
    for gene in GENES:
        print(f"Summarizing {gene}", flush=True)
        cohort = pd.read_csv(OUT / "cohort.csv")
        cohort = cohort[cohort.protein == gene].sort_values("mutation_code")
        arrays, inventory = [], []
        for record in cohort.itertuples():
            with np.load(OUT / gene / f"{record.mutation_code}.npz") as saved:
                arrays.append(saved["scores"])
                inventory.append(json.loads(str(saved["metadata"])))
        all_rows.extend(summarize_tensor(gene, cohort.mutation_code.to_numpy(), cohort.label.to_numpy(),
                                        np.stack(arrays), POLICIES, SIZES, summary_dir))
        pd.DataFrame(inventory).to_csv(summary_dir / f"{gene}_source_inventory.csv", index=False)
        print(f"Summary {gene} complete", flush=True)
    pd.DataFrame(all_rows).to_csv(summary_dir / "all_summary.csv", index=False)
    (summary_dir / "method.json").write_text(json.dumps({
        "position_cluster_bootstrap": BOOT, "draw_repetitions": REPS,
        "consistency_probability_threshold": CONSISTENCY,
        "practical_auc_tolerance": .005,
        "interval_target": "mean per-draw AUC difference conditional on stored AF3 pools",
        "bootstrap_rule": "same whole-position multiplicities at every size/policy; mixed-label positions preserved",
        "invalid_replicates": "single-class position samples omitted, counts in summary",
        "exact_ties": "integer ECDF counts; scalar N125 scores restored to mathematical 1/(N*R) grid within 1e-14; raw checkpoints preserved",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
