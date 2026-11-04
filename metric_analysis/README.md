# metric_analysis

The per-variant structural data and the cohort that every analysis reads:

- pLDDT extraction from the AlphaFold 3 outputs;
- C-alpha coordinate caches;
- driver/passenger labels;
- the strict canonical cohort.

The KS-score itself, the score caches and the evaluation are in `drift/`.

## What `run_all.sh` runs from here

| stage | module | what it writes |
|---|---|---|
| `X1_plddt` (extract) | `pipeline.orchestration.run_structure_metrics --stage plddt` | per-residue pLDDT matrices, one CSV per variant and WT, in `PLDDTs/<dataset>/` |
| `X2_coords_*` (extract) | `pipeline.cache_builders.repair_rmsf_cache_chunk` (4 chunks, then `--merge`) | C-alpha coordinates superposed on the WT with US-align, in `structure_metrics/<dataset>/ensemble_aware/cache/` |
| `L0_clinvar_sha` | `manifests/verify_clinvar_manifest.py` | checks the ClinVar exports against their SHA-256 in `manifests/clinvar_source_manifest.csv` |
| `A1_structure_metrics` | `pipeline.orchestration.run_structure_metrics --stage plddt_distribution` | the per-variant cohort summaries, `<dataset>_plddt_distribution_summary.csv`, which the manifest builder reads |
| `K1_core_fit_caches` | `pipeline.cache_builders.subunit_core_fit` | the coordinate caches re-fitted on each WT's rigid core, `ensemble_aware/cache_corefit/` |
| `K2_usalign_frame_caches` | `pipeline.cache_builders.subunit_core_fit --frame usalign` | the same structures in the original US-align frame, the comparison control (`cache_usalign/`) |
| `K4_coordinate_sweeps` | `pipeline.orchestration.run_structure_metrics --stage ensemble_aware` | RMSF, RMSD and positional KS by ensemble size in the core-fit frame, `ensemble_aware/ensemble_sweeps/` (Figs. S1, S2) |

The two extract stages need the raw archives (`DRIFT_ARCHIVE_ROOT`). The rest read the
processed data only. `pipeline.cache_builders.stream_align` builds the same US-align cache
as `repair_rmsf_cache_chunk` in one sequential pass, which is faster on a slow external drive.

## Layout

```
metric_analysis/
  protein_configs.json        per dataset: archive batches, chains, WT files, label source, cache folders
  manifests/                  ClinVar checksums; the structural inventory and strict canonical manifest (data)
  pipeline/
    paths.py                  the repository root, against which protein_configs.json paths resolve
    labels.py, clinvar.py     driver/passenger labels and ClinVar parsing
    config/protein_config.py  reads protein_configs.json
    orchestration/            run_structure_metrics (pLDDT extraction, cohort summaries, coordinate sweeps);
                              run_ensemble_sweep_parallel (coordinate descriptors by ensemble size)
    cache_builders/           repair_rmsf_cache_chunk, stream_align, subunit_core_fit
    library/                  the shared definitions, below
  tests/
```

`pipeline/library/`:

| module | what |
|---|---|
| `canonical_cohort.py` | the strict canonical cohort (1,798 annotation variants) and each row's file provenance |
| `descriptor_metrics.py` | descriptor definitions, including the residue KS (the KS-score) |
| `descriptor_sampling.py` | descriptor values from a variant's pLDDT matrix (the subunit row blocks for MSH2 and MSH6) |
| `plddt_distribution.py` | per-residue WT-vs-variant pLDDT distribution metrics |
| `positional_ks.py` | the KS-score on C-alpha positions instead of pLDDT |
| `af3_seeds.py` | AF3 seed numbers of the predictions, and the per-gene seed cap |
| `fast_plddt.py` | pLDDT extraction with output identical to the `plddt` stage |
| `position_bootstrap.py` | bootstrap intervals that resample residue positions |
| `bootstrap_uncertainty.py` | bootstrap intervals for per-residue class means |
| `ensemble_aware_pipeline.py`, `structure_metrics.py` | coordinate caches and coordinate descriptors (RMSF, RMSD) |

## Running a module on its own

From the repository root, after `pip install -e .`:

```bash
run-structure-metrics --protein POLE --stage plddt_distribution --force
python -m metric_analysis.pipeline.cache_builders.subunit_core_fit --dataset POLE
```

Add `--help` to any module for its options.

## Tests

```bash
PYTHONPATH=. pytest metric_analysis/tests
```

Most tests need the processed data in place.
