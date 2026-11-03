# slurm: AF3 on an HPC cluster with Slurm (as run for the paper)

Most of the study's structures were generated with these scripts on the Technion HPC
cluster, using the AlphaFold 3 container. AF3 runs in two stages: an MSA stage on CPU
nodes, then inference on GPUs from the MSA-ready `*_data.json`. Each stage has a
scheduler, which submits a Slurm array, and a worker, which runs one job per array task.

| file | role |
|---|---|
| `slurm.env` | site settings: shared filesystem, lab project space, AF3 install (container, models, databases) |
| `msa_scheduler.sh` | picks job JSONs not yet done or locked, writes a manifest, submits `msa_worker.sbatch` as an array |
| `msa_worker.sbatch` | one job's data pipeline (`--run_data_pipeline=true --run_inference=false`) |
| `gen_scheduler.sh` | the same for `*_data.json`; skips jobs whose archive or `_SUCCESS` exists |
| `gen_worker.sbatch` | inference only (`--run_data_pipeline=false`), then tars the output and deletes the folder |

```bash
# job JSONs from the configs (python -m af3_jobs generate <config>), copied under $PROJECT_ROOT
./msa_scheduler.sh $PROJECT_ROOT/$USER/mut_jsons/<batch> <QOS> <MAX_NEW> <MAX_PARALLEL> 1
GEN_OUT_ROOT=$PROJECT_ROOT/$USER/output_gen/<batch> \
  ./gen_scheduler.sh $PROJECT_ROOT/$USER/output_msa_only <QOS> <MAX_NEW> <MAX_PARALLEL> 1
```

`DRYRUN=1` prints the `sbatch` command without submitting. A scheduler creates one lock
per job it submits and the worker removes it on exit. A killed array can leave locks
behind; `GEN_TREAT_LOCKS_AS_STALE=1` clears them, but only once nothing is queued.

Edit `slurm.env` and the `#SBATCH` headers (partition, QOS, GPU type, log folder) for
another cluster. The scripts need bash 4 or later.

- `gen_worker.sbatch` is the version that ran. By default (`JAX_MEMORY_DIAGNOSTIC=1`) it
  limits JAX to 25% of GPU memory and logs GPU memory once a second.
- Only the site paths changed from the cluster copies; they now come from `slurm.env`.
- `../local_runner/` runs the same two stages on a machine without Slurm and
  writes the same output layout.
