# SI Fig. S5: KS-score against AlphaMissense (2026-10-29)

`FigureS_ks_vs_alphamissense.pdf` (paper repo `figures/`), drawn whole at the SI
text width with `figlib`'s style. It replaces the stacked `FigureS3.pdf`.
a, KS-score at N = 90 against AlphaMissense per annotation gene; b, Spearman ρ per
gene for all variants and within each class, with the macro, beside a.

    python experiments/si_ks_vs_am_figure/rebuild.py

## Inputs (`inputs/`)

- `ks_vs_am_variants_25seed.csv` — per-variant KS-score (mean of the ten N = 90 draws,
  seeds 42–51) and AlphaMissense score, averaged exactly as `si_figures/ks_vs_am.py`
  does from the POLE 25-seed score caches
  (`experiments/annotation_stages/cache/labeled_size90_seed*_…49028c16cbb3_esfeat6.csv`).
  1,798 variants.
- `ks_vs_am_correlations.csv`, `ks_vs_am_macro.csv` — that run's published statistics
  (`…/annotation_stages/publication/si_figures/`), position-clustered intervals.

The main tree's `si_figures/results/ks_vs_am_*` are from 10-16, before the POLE
rerun (1,075 POLE variants), and are not used.

`rebuild.py` asserts that the per-variant table gives back all 18 published ρ, the
0.47–0.76 range quoted in the main text, the macro 0.64 (0.59–0.68) and POLE
passengers 0.61.
