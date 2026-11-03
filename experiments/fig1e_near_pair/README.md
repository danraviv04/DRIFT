# Fig. 1e with a driver and passenger at neighbouring sites (2026-10-22)

Fig. 1e showed the driver R772L against the passenger E221D. They are 551 residues apart, so the passenger's own site was never in the window. This experiment picks pairs whose two mutated residues fit in one 16-residue window. It also changes the panel's y axis to $\mathrm{pLDDT}_i - \langle \mathrm{pLDDT}_i^{\mathrm{WT}} \rangle$ and the KS strip's to $\mathrm{KS}_i$.

The paper repo was not touched.

## Pair choice

Candidates were MSH6 driver/passenger pairs at most 14 residues apart, under the canonical cell (`DRIFT_RUN_TAG=v6_ca`, N = 90, seed 20260829). Each pair needed a driver at or above the 60th percentile of the 66 drivers' KS-scores, and a passenger between the 20th and 80th percentiles of the 77 passengers. The aim was a typical passenger, not the cleanest one.

| Driver (KS, class pct) | Passenger (KS, class pct) | Gap | Panel |
|---|---|---|---|
| A1055P (0.227, 71st) | I1054F (0.147, 69th) | 1 | clean; all residues WT pLDDT 95–98 |
| R482P (0.277, 86th) | K479R (0.147, 70th) | 3 | left half flexible (WT SD up to 1.0) |
| H367R (0.286, 88th) | T363I (0.146, 66th) | 4 | window runs into disorder (WT pLDDT 33–46); y axis to −60 |
| R772L (0.243, 77th) | E760D (0.140, 48th) | 12 | current driver; the two sites sit at opposite edges |

Panels: `../si_figures/results_fig1_MSH6_near_<driver>_<passenger>/MSH6_residue_distributions.pdf`. Logs: `../si_figures/logs_near_pairs/`. Each panel run asserts canonical KS parity for both variants.

## Per-residue calibration (the numbers the text quotes)

`residue_null_near_pair.py` reruns `residue_null/residue_null.py` unchanged. It uses 2,000 splits, the matched and random arms, and the 75-seed WT pool. Both calibration arms equal the published `wt_background_2000` run exactly, and this is asserted. R772L reproduces its published row exactly.

| Variant | Residues differing, mean [range] over 10 draws | In window | Variant-level P |
|---|---|---|---|
| A1055P | 20.1% [18.2–22.8%] | 16/16 every draw | 0.019 |
| I1054F | 0.8% [0–1.3%] | 0–3 | 0.24 |
| R772L | 25.3% [22.6–31.2%] | 16/16 | 0.003 |
| E760D | 0% | 0 | 0.52 |

In the saved draw, I1054F differs at its own residue 1054 (raw P ≈ 1e-28), at 1048 and 1057, and at 14 residues outside the window.

## Gene-wide view (`gene_wide.py`)

`results/gene_wide/MSH6_gene_wide_A1055P_vs_I1054F.pdf` shows the same pair at all 1,360 residues on one residue axis. It has four tracks:

- WT mean pLDDT.
- KS_i, against each variant's own canonical WT draw. The mean of each trace equals the KS-score, and this is asserted.
- The calibrated residue calls, drawn with opacity equal to the share of the ten draws that called each residue. `residue_null.py` saves only draw 0, so the script replays its RNG streams. Draw 0 and the mean share equal the saved run exactly, and this is asserted.
- The mean shift ⟨pLDDT_i^MUT⟩ − ⟨pLDDT_i^WT⟩, drawn against the WT ±σ band.

| | Residues called in ≥ 5 of 10 draws | Outside the Fig. 1e window | In low-confidence residues (WT pLDDT < 70, 318 of them) |
|---|---|---|---|
| A1055P | 276 | 260 | 2 |
| I1054F | 16 | 13 | 0 |

The driver's effect is not local. Most of its called residues lie in blocks across 520–1200, and its own window is one of them. The disordered N-terminal region shows nothing for either variant. The passenger's few calls outside its own site fall in the 800–900 stretch where the driver's effect is largest.

## Previews

`preview_<pair>/Figure1_resKS_preview.pdf` was assembled by the paper repo's `tools/make_figures.py Figure1_resKS` on a copy of `figures/`, with only `panels/MSH6_residue_distributions.pdf` swapped. The copied inputs were deleted afterwards.
