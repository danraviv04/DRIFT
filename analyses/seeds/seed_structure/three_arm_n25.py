"""Three seed-allocation arms at N=25 per side, from the plateau run's saved per-draw AUCs.

5 seeds x 5 samples, 25 random conformations (seed-blind, the production sampler) and
25 seeds x 1 sample, each with WT and mutant drawn independently. No KS is recomputed:
AUCs come from plateau_20261018/summary/GENE_auc_draws.csv.gz.

contrasts.csv holds the paired differences between seed policies with 95% position-cluster
intervals, at N = 10, 25, 40 and 90 (Note S7: "excluded zero in four of the six genes" at
N = 25; the equal-gene mean of `delta` at N = 90). The plateau run saves every policy's
AUC on each position-cluster bootstrap draw (GENE_pair_ordering.npz); an interval is the
2.5-97.5% quantile of the per-draw difference.

python analyses/seeds/seed_structure/three_arm_n25.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "plateau_20261018" / "summary"
OUT = HERE / "three_arm_n25"
GENES = ["POLE", "PTEN", "VHL", "MSH2", "MSH6", "PIK3CA"]
SIZE = 25
METRICS = [("plddt_ks", "pLDDT KS"), ("positional_ks", "Coordinate KS (core residues)")]
# Ordered by number of distinct seeds per side: 5, ~17 on average, 25.
ARMS = [("concentrated_independent", "5 seeds × 5 samples", "#eb6834", "s"),
        ("uniform_independent", "25 random conformations (~17 seeds)", "#2a78d6", "o"),
        ("spread_independent", "25 seeds × 1 sample", "#1baf7a", "^")]
SEED_CONTRASTS = [
    ("spread-minus-concentrated-independent", "spread_independent", "concentrated_independent"),
    ("spread-minus-concentrated-matched", "spread_matched", "concentrated_matched"),
    ("spread-minus-uniform-independent", "spread_independent", "uniform_independent"),
]
CONTRAST_SIZES = [10, 25, 40, 90]


def paired_contrasts():
    summary = pd.read_csv(SRC / "all_summary.csv")
    rows = []
    for gene in GENES:
        with np.load(SRC / f"{gene}_pair_ordering.npz") as draws:
            for metric, _ in METRICS:
                for n in CONTRAST_SIZES:
                    auc = summary[(summary.gene == gene) & (summary.metric == metric)
                                  & (summary["size"] == n)].set_index("policy").auc_mean
                    for name, a, b in SEED_CONTRASTS:
                        key_a, key_b = (f"{metric}__{p}__N{n}__bootstrap_auc" for p in (a, b))
                        if key_a not in draws or key_b not in draws:
                            continue
                        diff = draws[key_a] - draws[key_b]
                        rows.append({"gene": gene, "metric": metric, "size": n, "contrast": name,
                                     "auc_A": auc[a], "auc_B": auc[b], "delta": auc[a] - auc[b],
                                     "cluster_low": np.nanquantile(diff, .025),
                                     "cluster_high": np.nanquantile(diff, .975),
                                     "invalid_bootstrap_draws": int((~np.isfinite(diff)).sum())})
    return pd.DataFrame(rows)


def draw_panel(axis, values, metric, title):
    x = np.arange(len(GENES))
    for offset, (policy, label, color, marker) in zip([-.22, 0, .22], ARMS):
        v = values[(values.metric == metric) & (values.policy == policy)].set_index("gene").loc[GENES]
        axis.errorbar(x + offset, v.auc_mean, yerr=v.auc_sd, fmt=marker, ms=9, capsize=4,
                      elinewidth=1.8, color=color, label=label)
    axis.set_title(title, fontsize=15)
    axis.set_ylabel("Mean AUC ± SD across 200 draws")
    axis.set_xticks(x, GENES, rotation=30)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e6e5e0", linewidth=.6)
    axis.set_axisbelow(True)


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=180)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    draws = pd.concat(pd.read_csv(SRC / f"{g}_auc_draws.csv.gz") for g in GENES)
    draws = draws[(draws["size"] == SIZE) & draws.policy.isin([a[0] for a in ARMS])]
    macro = draws.groupby(["metric", "policy", "repeat"]).auc.mean().reset_index().assign(gene="Macro")
    values = (pd.concat([draws, macro]).groupby(["metric", "policy", "gene"]).auc
              .agg(auc_mean="mean", auc_sd="std", draws="size").reset_index())
    contrasts = paired_contrasts()
    values.to_csv(OUT / "auc_by_arm.csv", index=False)
    contrasts.to_csv(OUT / "contrasts.csv", index=False)

    # Paired contrasts are saved to contrasts.csv but not plotted.
    plt.rcParams.update({"font.size": 13})
    suptitle = f"Seed allocation at N={SIZE} conformations per side (WT and mutant drawn independently)"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for axis, (metric, title) in zip(axes, METRICS):
        draw_panel(axis, values, metric, title)
    axes[1].legend(frameon=False, fontsize=12, loc="upper right")
    fig.suptitle(suptitle, fontsize=15, y=.995)
    save(fig, "three_arm_n25")

    fig, axis = plt.subplots(figsize=(8.5, 6.4))
    draw_panel(axis, values, *METRICS[0])
    axis.legend(frameon=False, fontsize=12, loc="lower left")
    axis.set_title(f"pLDDT KS, N={SIZE} conformations per side", fontsize=15)
    save(fig, "three_arm_n25_plddt")

    table = values[values.gene == "Macro"].pivot(index="metric", columns="policy", values="auc_mean")
    print(table[[a[0] for a in ARMS]].round(4).to_string())
    main_contrast = contrasts[(contrasts.metric == "plddt_ks")
                              & (contrasts.contrast == "spread-minus-concentrated-independent")]
    print("25x1 minus 5x5, pLDDT KS, equal-gene mean of delta by N:",
          main_contrast.groupby("size").delta.mean().round(4).to_dict())


if __name__ == "__main__":
    main()
