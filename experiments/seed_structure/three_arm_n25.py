"""Three seed-allocation arms at N=25 per side, from the plateau run's saved per-draw AUCs.

5 seeds x 5 samples, 25 random conformations (seed-blind, the production sampler) and
25 seeds x 1 sample, each with WT and mutant drawn independently. No KS is recomputed:
AUCs come from plateau_20261018/summary/GENE_auc_draws.csv.gz and the paired contrasts
with position-cluster intervals from plateau_20261018/summary/seed_policy_contrasts.csv.

python experiments/seed_structure/three_arm_n25.py
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
CONTRASTS = [("spread-minus-concentrated-independent", "25×1 minus 5×5", "#eb6834", "s"),
             ("spread-minus-uniform-independent", "25×1 minus random 25", "#2a78d6", "o")]


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
    contrasts = pd.read_csv(SRC / "seed_policy_contrasts.csv")
    contrasts = contrasts[(contrasts["size"] == SIZE) & contrasts.contrast.isin([c[0] for c in CONTRASTS])]
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


if __name__ == "__main__":
    main()
