# drift

The DRIFT package: from per-residue pLDDT matrices to the per-variant scores every
analysis reads.

| module | role |
|---|---|
| `structure_draws.py` | which predictions each (variant, ensemble size N, seed) draw uses (`SamplingPolicy`, `select_structure_columns`, `source_draw_key`); where the matrices are found (`discover_wt_csv_sources`, `discover_mut_csv_sources`, POLE's 25-seed files) and how they are read; `build_d1_table` assembles the labelled dataset |
| `ensemble_shift.py` | the ensemble descriptors on the same draws, the residue KS-score (`plddt_ks_residue`) among them |
| `core.py` | dataset assembly: labels, AlphaMissense, descriptors |
| `run.py` | `python -m drift.run --sweep`: builds the score caches, one CSV per (N, seed) in `cache/`, named by the config fingerprint |
| `protein_config.py`, `protein_configs.json` | per-dataset inputs, labels and pLDDT locations; the fingerprint is a hash of every non-underscore field |
| `article_figures.py` | Figure 1h (`panel_nsweep_metrics`) |
| `metrics.py`, `convergence.py`, `sampling_plan.py` | shared helpers |

`run_all.sh` builds the caches with

```bash
python -m drift.run --sweep --sizes 1,5,10,20,40,60,90 --seeds 42-42 --gt-group annotation \
    --feature-variants base --out drift/results/_cache_shard_seed42      # one per seed, 42-51
```

and every later stage reads `cache/labeled_size{N}_seed{S}_ratio_of_means_proteins{fingerprint}_esfeat6.csv`.

**Draws do not depend on where the checkout lives.** Each draw key names its source file
relative to the repo root (`structure_draws.CHECKOUT_ROOT`), with that root replaced by a
fixed string (`LEGACY_KEY_ROOT`, the folder the published caches were built in). Do not
change that string: it fixes which predictions every draw picks.

**Also here but not used by the paper:**
- the learned-fusion models and benchmark tracks of `run.py` and `benchmark.py`;
- W&B logging (`wandb_progress.py`);
- the D1 descriptor, computed alongside KS.

They will be removed in a later revision.
