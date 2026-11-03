# SI coordinate figures (2026-10-29)

The old Supplementary Fig. S1 was four panel PDFs from three scripts, stacked by
the paper repo's `tools/make_figures.py` (`FigureS1` layout). Nothing shared a
size, the fonts came out at about 4 pt, and its caption ran off the page. It is
now two figures, drawn whole at the SI text width (168 mm) with `figlib`'s style:

| SI number | File (paper repo `figures/`) | Content |
|---|---|---|
| S1 | `FigureS_coordinate_descriptors.pdf` | a, shift vs ΔRMSF in POLE P286R; b, AUC of the four descriptors with 95% intervals; c, RMSD and ΔRMSF by class per gene |
| S2 | `FigureS_rigid_core.pdf` | WT RMSF against WT pLDDT, rigid-core residues coloured (the old panel d) |

The figures that followed moved up one: gene-wide MSH6 is S3, WT backgrounds
(`FigureS2_v2.pdf`) S4, KS vs AlphaMissense (`FigureS3.pdf`) S5.

    python experiments/si_coordinate_figures/rebuild.py

## Inputs (`inputs/`, gitignored copies)

Nothing is recomputed from coordinates. Every value comes from the outputs that
made the old panels:

- `auc_matched_draws_25seed.csv` — `experiments/annotation_stages/side/s1_matched_draws/`
- `variant_seed_scores_25seed.csv.gz` — `experiments/annotation_stages/side/s1_matched_draws/`.
  The file of the same name in that worktree's `s1_matched_draws/results/` is the
  10-19 run, before POLE moved to seeds 1–25. It gives POLE RMSD 0.7357 against
  0.7354 and fails the check below.
- `shift_vs_rmsf_P286R_25seed.csv`, `..._values_25seed.csv` — `experiments/annotation_stages/side/s1a/`
- `wt_rmsf_vs_plddt_residues.csv` — `experiments/si_figures/results/`

`rebuild.py` asserts before drawing that:
- the AUC table reproduces Supplementary Table S2 as printed;
- the per-variant scores give back every RMSD and ΔRMSF AUC as a mean of per-seed AUCs (to 1e-9);
- panel a reproduces r = 0.427 and ρ = 0.371;
- panel d reproduces the printed medians and ρ, and the core sizes of Table S1.

The paper's current panels (`figures/si_panels/`) are byte-identical to the
25-seed outputs. Its `MANIFEST.txt` still points at the 10-19 paths.
