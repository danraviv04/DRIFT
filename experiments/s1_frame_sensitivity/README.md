# Note S1 superposition sensitivity on the matched seed 1–25 draws (2026-10-21)

This reruns the Note S1 paragraph "Sensitivity to the superposition" on the predictions of Supplementary Table S2. The earlier version used the coordinate-sweep draws, in which eight POLE drivers included AF3 seeds 26–32. Dan approved the rerun as option (b).

- **Cohort.** 1,797 variants: POLE on seeds 1–25, P356S out, MSH2 G162R excluded.
- **Draws.** The ten production N=90 draws (seeds 42–51).
- **Frames.** The WT rigid-core fit (`cache_corefit`, the published S1 frame) against US-align of the scored chain on a WT prediction (`cache_usalign`). For MSH6 the US-align frame is `cache_usalign_scored_chain`, so MSH6 is fitted on its own chain, not in MSH2's frame.
- **Method.** Both frames run `s1_matched_draws.calculate` unchanged, except for the coordinate cache (`COORD_FRAME`).
- **Inputs and run time.** Local caches only; 7.5 minutes.

## Checks

- The core-fit run reproduces the published S1 scores of the 2026-10-20 rerun exactly (max |diff| 0 for RMSD, ΔRMSF, positional KS, RMSF and KS).
- Its AUCs equal Supplementary Table S2. MSH6 positional KS is 0.762.
- pLDDT KS is identical in both frames.

## Results

AUC is the mean of ten per-seed AUCs. Δ is core-fit minus US-align, with a paired 95% interval over residue positions (stratified driver-only / passenger-only / mixed, 2,000 replicates). ρ is the per-variant Spearman correlation between frames of the ten-draw mean scores.

| Gene | RMSD core / US | ΔRMSF core / US | Positional KS core / US | Positional KS Δ [95% CI] | ρ RMSD, ΔRMSF, posKS |
|---|---|---|---|---|---|
| POLE | 0.735 / 0.735 | 0.718 / 0.718 | 0.762 / 0.762 | 0.000 | 1.000, 1.000, 1.000 |
| PTEN | 0.595 / 0.596 | 0.471 / 0.469 | 0.897 / 0.882 | +0.015 [−0.057, +0.078] | 1.000, 1.000, 0.898 |
| VHL | 0.585 / 0.587 | 0.498 / 0.493 | 0.815 / 0.812 | +0.003 [−0.057, +0.066] | 0.999, 0.991, 0.883 |
| MSH2 | 0.658 / 0.659 | 0.469 / 0.466 | 0.683 / 0.703 | −0.021 [−0.041, −0.001] | 1.000, 0.998, 0.970 |
| MSH6 | 0.552 / 0.553 | 0.468 / 0.467 | **0.762 / 0.588** | **+0.173 [+0.109, +0.236]** | 1.000, 0.995, 0.581 |
| PIK3CA | 0.766 / 0.765 | 0.543 / 0.544 | 0.749 / 0.766 | −0.017 [−0.037, +0.004] | 1.000, 1.000, 0.992 |
| Macro | 0.649 / 0.649 | 0.528 / 0.526 | 0.778 / 0.752 | +0.026 [+0.006, +0.045] | |

Summary quantities, in `results/summary.json`:

- **Per-variant ρ between frames:** ≥ 0.991 for RMSD and ΔRMSF in every gene. The minimum is VHL ΔRMSF.
- **Largest |ΔAUC|, RMSD and ΔRMSF:** 0.005 (VHL ΔRMSF).
- **Positional KS, five non-MSH6 genes:** largest |ΔAUC| 0.021 (MSH2); ρ ≥ 0.88.
- **MSH6 positional KS:** 0.762 (core fit) vs 0.588 (US-align), ρ = 0.58.
- **POLE:** identical in both frames. Its rigid core spans the whole construct, so the two fits coincide.

## Against the current SI paragraph (earlier sweep draws)

| Claim in the paragraph | Matched draws |
|---|---|
| ρ ≥ 0.99, RMSD/ΔRMSF, every gene | holds (min 0.991) |
| ROC-AUC changes ≤ 0.01, RMSD/ΔRMSF | holds (max 0.005) |
| Fig. 1c POLE ensembles agree within 0.001 Å | holds (0.0002 Å; `annotation_stages/side/s1a`) |
| Positional KS changed by at most 0.02 in five genes | 0.021 (MSH2) |
| MSH6 positional KS fell from 0.75 to 0.60 | 0.76 → 0.59 |

The 0.76 is Table S2's 0.762.

## Suggested paragraph (for review; not applied)

> Both fits were applied to the same predictions as Supplementary Table S2. Ensemble-mean RMSD and ΔRMSF were almost unchanged (per-variant Spearman ρ ≥ 0.99 between the fits in every gene; ROC-AUC changes ≤ 0.005), and for the POLE ensembles of Fig. 1c the two fits agreed to within 0.001 Å. Positional KS changed by at most 0.02 in ROC-AUC in five genes. In MSH6, whose chain contains blocks that move relative to each other (Supplementary Fig. S1d), it fell from 0.76 to 0.59 (difference 0.17, 95% CI 0.11 to 0.24): there, positional KS depends on fitting the rigid core.

```
PYTHONPATH="." python experiments/s1_frame_sensitivity/frame_sensitivity.py
```
