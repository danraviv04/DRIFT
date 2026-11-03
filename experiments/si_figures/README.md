# experiments/si_figures/

New figures for the SI master (and, later, the thesis). One script per figure or
per tight figure family; all of them import `figlib`.

## Run

```
DRIFT_RUN_TAG=v6_ca PYTHONPATH="." python experiments/si_figures/<script>.py
```

`DRIFT_RUN_TAG=v6_ca` matters (2026-10-16): the fusion caches are picked by the
protein-config fingerprint (v6: d61d29328362) whatever the tag, but run-tagged
inputs (`tagged(...)`: the size sweep, `ks_meanfree/results/canonical_*`) fall
back to v5_kras4b without it. `logs_v6/run_rest.sh` is the v6 rebuild.

Outputs land in `results/` as **PDF + PNG**, alongside `si_figure_values.csv`
(every quotable number) and `run_metadata.json` (date, git commit, cohort size,
dirty-tree flag).

## What `figlib` gives you

| Call | Returns |
| --- | --- |
| `labeled(size=90, seed=42)` | Strict-cohort per-variant table: gene, mutation, label, afm_score, d1_score, plddt_ks_residue, plddt_wasserstein, plddt_spread_ratio, axis_* |
| `cohort(protein=None)` | Canonical cohort with **exact** mutant and WT pLDDT paths, labels, provenance |
| `plddt_matrix(path)` | Per-residue pLDDT, `(n_residues, n_structures)` |
| `size_scores()` | Audited per-variant × seed × requested-size D1 table |
| `coords(protein, artifact=..., frame="core_fit")` | CA coordinates in the WT rigid-core frame (v6; `frame="usalign"` = old control) |
| `sweep("dirname")` | Any `sweep_runs.csv` under `drift/results/` |
| `check_cohort(df, cohort=...)` | Raises unless the frame is exactly the v5 annotation (1,799) or continuous (2,468) cohort |

Plus `panel()` / `grid()` at journal widths, `save()` (PDF + 300 dpi PNG),
`record()` / `write_values()`, `provenance()`, and the `article_figures`
palette so new panels sit beside the existing ones.

## Conventions

1. **Start from the docstring.** Every script states what the figure
   *establishes* before it states how. A panel that cannot answer that in one
   sentence is not ready to be drawn.
2. **`check_cohort()` in anything that filters or merges.** A silent cohort
   change should fail, not ship.
3. **`record()` every number a caption will quote.** Captions cite
   `si_figure_values.csv`; nothing is retyped by hand.
4. **PDF is the deliverable, PNG is for the gallery.** Never cite the PNG.
5. **Say PRE or POST.** Results predating the strict-canonical cutover
   (`strict_canonical_manifest.csv`, 2026-08-31 14:04) are not quotable —
   note it in the docstring if a source is PRE.
6. **Names carry no figure number** (since 2026-10-15). A panel stem is a slug
   (`descriptors_auc_vs_size`), a script is `<figure id>.py`, and the values file is
   `<figure id>_values.csv`. The figure number and panel letter come from the order
   of `drift/si_panels.SI_FIGURE_LIST`, so a figure is dropped by deleting
   its entry. Keep a stem stable once a caption cites it.
7. **SI text** drafts (captions, note paragraphs) go in `SI_TEXT_DRAFTS.md`,
   with numbers taken from the values files.

## Traps `figlib` absorbs

- Fusion caches spell it `MutSalpha`; manifests and pipeline dirs spell it
  `MutS_alpha`. Use `to_cache_key` / `to_dir_key`.
- Fusion caches predate the transcript-matched audit — their membership and
  labels are not authoritative. `labeled()` always re-applies the strict
  manifest.
- Type 3 fonts are a common journal bounce; `figlib` sets Type 42.
- `drift.run` initialises MPS and loads AlphaMissense on import, so it is
  imported lazily — only `labeled()` pays that cost.
- The coordinate cache is tens of millions of rows; pass `artifact=` so the
  filter is pushed into the parquet reader.

## KS estimator policy (2026-09-16)

Current paper panels use **residue KS**: compute a two-sample KS at each
residue, then average over residues. Its cache field is `plddt_ks_residue`;
its canonical-analysis field is `ks_raw`. The `plddt_ks` field is historical
profile KS and must not supply current paper panels.

`combo_roc.py` defaults to the current tagged canonical residue-KS table.
Profile-KS reproduction requires both `--ks-source profileKS` and
`--historical-profile-ks`. `figure2_panelC_subset.py` is wholly historical
(merged MutSalpha, profile KS) and requires `--historical-profile-ks`.
Use `experiments/figure2/make_figure2.py` for the current residue-KS Figure 2.
Existing profileKS-named outputs in `results/` are historical comparisons.

Article ROC/violin panels use one cache seed (default N=90, seed=42), retaining
matched D1 and residue-KS sampling. Depth curves summarize the canonical
subsampling seeds. Their N=90 AUCs need not coincide; panel provenance records
this distinction. Manuscript layout and final SI selection remain separate.
