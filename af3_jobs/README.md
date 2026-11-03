# af3_jobs — AlphaFold 3 input JSONs for mutation ensembles

A complex is described once in `af3_jobs/configs/<NAME>.json`, and the CLI turns that
description into one AF3 job JSON per variant.

```bash
python -m af3_jobs list                 # what configs exist
python -m af3_jobs show PIK3CA          # chain lengths, variant count, skips — writes nothing
python -m af3_jobs generate PIK3CA      # muts_jsons/PIK3CA_Mutations.zip + manifest
```

Running AF3 on the jobs: `slurm/` (Slurm arrays on an HPC cluster, as for the paper) or
`local_runner/` (a machine without Slurm). POLE and the first VHL and TP53 batches predate
the configs and were written by `legacy_generate_af3_ensembles.py`.

## Adding a protein

```bash
python -m af3_jobs scaffold BRAF --chain P15056
python -m af3_jobs scaffold PIK3CA --chain P42336 --chain P27986:322-600
```

The first `--chain` is the mutated one. Then edit the generated config and run
`show` before `generate`.

## Config reference

| key | meaning |
|---|---|
| `name` | config name; also the zip stem (`<name>_Mutations<tag>.zip`) |
| `job_prefix` | prefix of the AF3 job name. Defaults to `name` |
| `file_prefix` | prefix of the written filenames. Defaults to `job_prefix` |
| `model_seeds` | an int `n` means seeds `1..n`; a list is used verbatim. AF3 returns 5 structures per seed |
| `version`, `dialect` | AF3 dialect version. Existing batches use 2 or 4 — keep a re-run on whatever its original batch used |
| `chains[]` | one entry per AF3 sequence element, in order |
| `mutations` | where the variant list comes from |
| `include_wt` | whether to emit a `<prefix>_WT` job |
| `output.dir`, `output.tag` | where the archive lands, and a suffix that keeps a re-run from overwriting an earlier batch |

### Chains

```json
{"id": "A", "type": "protein", "gene": "PIK3CA", "uniprot": "P42336", "mutate": true}
{"id": "B", "type": "protein", "uniprot": "P27986", "range": "322-600"}
{"id": "C", "type": "dna", "sequence": "GGATCGTTTTTTTTTGTCGAC"}
{"id": "D", "type": "ligand", "ccd_codes": ["ATP"]}
```

- `type` is `protein` (default), `dna`, `rna` or `ligand`.
- A chain is either `uniprot` (fetched once, cached in `seq_cache/`) or a literal
  `sequence`. Isoform accessions work, and for a multi-isoform gene name the
  isoform explicitly: bare `P01116` resolves to UniProt's canonical KRAS
  **4A** (189 aa); 4B is `P01116-2` (188 aa).
- `range` is 1-based inclusive in the source numbering, and it also sets
  `first_residue`. That keeps variant names in UniProt coordinates even for a
  fragment chain: on `322-600`, `N564D` means UniProt 564, not offset 564.
- Exactly one chain may carry `"mutate": true`. Two mutated subunits means two
  batches — that is how MSH2 and MSH6 were generated, and why MSH6 variants sit
  in chain B.

### Mutation sources

```json
{"source": "clinvar", "csv": "Ground_Truths/ClinVar/clinvar_result_pten.csv", "gene": "PTEN"}
{"source": "list",    "include": ["H1047R", "E545K"]}
{"source": "file",    "path": "experiments/foo/variants.txt"}
{"source": "enumerate", "max_nt_subs": 1}
{"source": "none"}
```

`include` is always added on top; `exclude` and `done` are always removed
(`done` is for variants an earlier batch already produced). `--variants` and
`--wt-only` override the source from the command line for a one-off.

The ClinVar `gene` filter reads the transcript-level annotation, so a
`KLLN|PTEN` locus export does not leak KLLN rows into a PTEN batch.

## What the generator guarantees

- **WT-residue check.** A variant whose stated WT residue disagrees with the
  chain sequence is skipped and reported, never silently applied. This is the
  check that catches an isoform or numbering mismatch.
- **Range check.** A position outside the chain is skipped, which matters for
  fragment chains.
- **Manifest.** Every run writes `<stem>_manifest.json`: per-chain length,
  residue range and SHA-256, the seed list, the variant list and every skip. It
  answers "which construct produced these ensembles?" without opening the zip.

## Tests

```bash
PYTHONPATH=. pytest af3_jobs/tests      # or, without pytest: python af3_jobs/tests/test_af3_jobs.py
```

The reproduction tests are the important ones: the `PTEN` and `MutS_alpha_MSH2`
configs regenerate the paper's job JSONs (`muts_jsons/`, in the data) byte for byte.

## Known trap in the existing archives

`muts_jsons/KRAS4A_Mutations.zip` is **not** 4A: it holds 188-aa 4B sequences at
dialect version 2 (the 4A config, superseded by 4B, is not in this release).
The current 4B batch is `configs/KRAS4B.json` -> `KRAS4B_Mutations.zip`
(188 aa, version 4); its jobs are named `KRAS_*` too, so keep its outputs in
a separate directory from the 4A archives.

## Homo-oligomers and name suffixes

- `"id": ["A", "B", "C", "D"]` declares one entity with four identical copies, which AF3 accepts directly. A `"mutate": true` entity is then mutated in **every** copy. That is how the TP53 tetramer batch was made, and `configs/TP53_hotspots.json` reproduces its A119D job exactly: name, seeds, sequences, ids, DNA and Zn. Its hotspot list was never folded; the config is kept as the TP53 construct `wt_from_variant` checks the A119D source against (the TP53 WT regeneration, `muts_jsons/P53_WT_vartmpl_manifest.json`).
- `"job_suffix"` adds text after the variant: `P53_R175H_with_ZN_and_DNA`.

## Seed extensions (MSA-ready)

```bash
python -m af3_jobs.seed_extension <job>_data.json --seeds 26-75 --chunk 5 --out muts_jsons/<name>
```

This writes `<job>_seeds_<a>-<b>_data.json` files, plus a zip and a manifest. Each is the finished job's own `_data.json` (alignments included) with only `modelSeeds` changed, so it runs inference only. The output archives come back as `<job>_seeds_<a>-<b>_data.tar.gz`, the TP53-extension layout.

`merge_seed_extensions.py` pairs ONE extension per job. For several chunks, either extract the chunk archives as their own pLDDT CSVs and pool them (as `pten_wt_null_panel --wt-csv` does), or merge them one at a time.

First use: `muts_jsons/MutS_alpha_MSH6_GT_WT_seed_ext_26-75.zip`, 10 × 5 seeds of the MSH6 G:T WT, for the seed-matched WT-vs-WT background.

## WT jobs with the variants' templates

```bash
python -m af3_jobs.wt_from_variant <variant>_data.json --config <config> --out muts_jsons/<prefix>_WT_vartmpl
```

AF3's template search drops any hit that exactly matches the query over more than 95% of its length. So when the PDB holds the exact WT construct, the WT job loses that template while every variant keeps it. This happened to MSH2 (2O8B) and PIK3CA (4L1B, 5DXH); see `experiments/wt_variant_offset/`. This tool takes a finished variant job's `_data.json` (MSAs and templates included) and builds an inference-only WT job from it. Only three things change:

- the target chain's sequence,
- the query row of that chain's `unpairedMsa` and `pairedMsa`,
- `name` and `modelSeeds`.

The templates are copied unchanged. Before writing anything, it checks every protein chain against the config's WT, the target chain against the named variant, and the MSA query rows against that variant's sequence. Any mismatch is refused. The default seeds are 1-25, the variants' own, so the jobs stay seed-paired. The job is named `<prefix>_WT_vartmpl`, so the archive comes back as `<prefix>_WT_vartmpl_data.tar.gz`, separate from the existing WT.

The MSH2 and PIK3CA WT jobs, from the raw archives:

```bash
SRC=wt_vartmpl_src && mkdir -p $SRC
tar -xzf $DRIFT_ARCHIVE_ROOT/MutSalpha/MutS_alpha_MSH2_Q629R_data.tar.gz -C $SRC \
    MutS_alpha_MSH2_Q629R_data/MutS_alpha_MSH2_Q629R/MutS_alpha_MSH2_Q629R_data.json   # 179 MB
tar -xzf $DRIFT_ARCHIVE_ROOT/PIK3CA/PIK3CA_K532R_data.tar.gz -C $SRC \
    PIK3CA_K532R_data/PIK3CA_K532R/PIK3CA_K532R_data.json                               # 62 MB
python -m af3_jobs.wt_from_variant $SRC/MutS_alpha_MSH2_Q629R_data/MutS_alpha_MSH2_Q629R/MutS_alpha_MSH2_Q629R_data.json \
    --config MutS_alpha_MSH2 --out muts_jsons/MutS_alpha_MSH2_WT_vartmpl
python -m af3_jobs.wt_from_variant $SRC/PIK3CA_K532R_data/PIK3CA_K532R/PIK3CA_K532R_data.json \
    --config PIK3CA --out muts_jsons/PIK3CA_WT_vartmpl
```

Upload the zips and run them with `--run_data_pipeline=false`. MutS-alpha took about 5 h per 25-seed job on the HPC cluster, so `--chunk` can split a job if the queue needs it.

To use a regenerated WT, point the dataset's `wt_hint` / `wt_artifact` in `metric_analysis/protein_configs.json` at its archive.
