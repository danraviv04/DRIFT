# SI Note S2 figures (2026-10-29)

Supplementary Figs. S3 and S4, regrouped by argument and drawn whole at the SI
text width (168 mm) with `figlib`'s style. Before, the WT-background figure carried
two arguments (the KS-score background and the raw per-residue test); the raw
test now sits with the per-residue figure.

| SI number | File (paper repo `figures/`) | Content |
|---|---|---|
| S3 | `FigureS_per_residue_test.pdf` | a, raw Mann–Whitney test between two WT halves, six genes (seed-disjoint vs random); b–e, the MSH6 pair of Fig. 1e at all 1,360 residues |
| S4 | `FigureS_wt_backgrounds.pdf` | KS-score of passengers and drivers against the WT–WT background, six genes |

    python experiments/si_note_s2_figures/rebuild.py

## Inputs (`inputs/`)

- `wt_background_*.csv`, `wt_backgrounds_values.csv` — `experiments/annotation_stages/side/wt_split_12v12/`
  (the POLE 25-seed run; the older `wt_split_12v12/results_12v12_2000` still has 1,037 POLE passengers).
- `MSH6_gene_wide_*.csv` — `experiments/fig1e_near_pair/results/gene_wide/`.

`rebuild.py` asserts before drawing that the inputs give the Note S2 table (95th
percentiles and the counts above them), the 12–24% and 0.5–3.7% raw-test rates,
and the MSH6 KS-scores and call counts (276/260 and 16/13).
