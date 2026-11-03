# local_runner: AF3 in two stages on a machine without Slurm

These scripts port the Slurm scripts `msa_scheduler.sh` / `msa_worker.sbatch` and
`gen_scheduler.sh` / `gen_worker.sbatch` (in `../slurm/`) to a plain Linux box such as golgi,
with Apptainer and the AF3 `.sif`. Each stage runs as a pool of background
workers, which survive logout.

| file | role |
|---|---|
| `af3_local.env` | per-server settings: work root, container, weights, databases, GPUs, worker counts |
| `msa_local.sh` | stage 1, the data pipeline (CPU only): `inputs/<batch>/*.json` → `msa/<batch>/<stem>/<job>/<job>_data.json` |
| `gen_local.sh` | stage 2, inference (one worker per GPU): `*_data.json` → `gen/<batch>/<job>_data.tar.gz` |
| `af3_local_lib.sh` | shared worker loop, claims, start/stop/status |
| `merge_seed_extensions.py` | merges seed-extension archives into their original runs |

The output layouts are the same as on the Slurm cluster: `output_msa_only` for MSAs, and
per-variant `.tar.gz` holding `<job>_data/<job>/...` for structures. Downstream
tooling therefore needs no changes.

## Setup (once per machine)

```bash
rsync -av af3_jobs/local_runner/ <user>@<server>:<install-dir>/
```

On the server:
Edit `af3_local.env`: set `AF3_WORK_ROOT`, `AF3_SIF`, `AF3_MODEL_DIR`, `AF3_DB_DIR`
and `GEN_GPUS` for the machine.

## A batch

```bash
# laptop: build the inputs and send them over
python -m af3_jobs generate MutS_alpha_MSH6_GT
scp muts_jsons/<zip> <user>@<server>:
# server: unpack into inputs/<batch>/
unzip <zip> -d $AF3_WORK_ROOT/inputs/MSH6_GT

./msa_local.sh start MSH6_GT 4        # 4 CPU workers
./gen_local.sh start MSH6_GT 0,1      # one worker per listed GPU
./msa_local.sh status MSH6_GT
./gen_local.sh status MSH6_GT         # counts, running variants, failures, rough ETA
```

Both stages can start together. While the MSA stage of the same batch is
running, gen workers poll for newly finished MSAs every `POLL_SECS` instead of
exiting. To feed gen from MSAs made elsewhere (for example copied from the Slurm cluster's
`output_msa_only`), set `GEN_IN=/path ./gen_local.sh start <batch>`.

## Dry runs

Prefix any command with `DRYRUN=1` to see what it would do. Nothing is
launched, claimed, created or deleted, and no locks are written (the Slurm cluster's
`DRYRUN=1` did write locks).

```bash
DRYRUN=1 ./msa_local.sh start MSH6_GT 4
DRYRUN=1 ./gen_local.sh start MSH6_GT 0,1
DRYRUN=1 ./gen_local.sh one MSH6_GT muts_alpha_msh6_gt_wt_data
DRYRUN=1 ./msa_local.sh stop MSH6_GT --now
DRYRUN=1 ./msa_local.sh reset MSH6_GT
```

A dry `start` reports:
- every preflight check as `ok` or `MISSING` (container, databases or
  weights, writable work root), and exits 1 if any is missing
- the queue counts (done, running, pending, retrying, gave up)
- the workers it would launch; for gen, each GPU's current use and owner, and
  whether the worker would start now or wait for memory
- the claim order
- the full AF3 command for the first variant, one option per line, followed
  by what happens after AF3 returns (the expected CIF count, the archive path)

`status` is read-only with or without `DRYRUN`.

Other commands are `stop <batch>` (workers finish their current variant, then
exit), `stop <batch> --now` (kills running variants too, without charging them
an attempt), `reset <batch>` (given-up variants get retried) and
`one <batch> <variant>`, which runs a single variant in the foreground. Use
`one` to test a new construct before launching the batch.

## How it differs from the Slurm version

- **No stale locks.** A variant is claimed with `flock` on
  `state/<batch>/<stage>/locks/<variant>.lock`, and the kernel releases that
  lock when the holder dies. A killed run, a crash or a reboot never leaves a
  lock behind, so `GEN_TREAT_LOCKS_AS_STALE` and its duplicate-job trap are
  gone. A worker also re-checks "done" after claiming, so a second `start`
  only adds workers and never duplicates work.
- **Retries are bounded.** A failure counts as one attempt. After
  `MAX_ATTEMPTS` (default 2) the variant is left alone and status lists it as
  `GAVE UP` with its log. A worker that fails `MAX_CONSEC_FAIL` (default 3)
  variants in a row stops, so a bad GPU or a missing database cannot burn
  through the queue.
- **No decoy `_data.json`.** Before a retry, the variant's partial output
  directory is cleared. Without that, AF3 writes a timestamped sibling
  directory next to it, which is how the null-filled decoys in the Slurm-cluster
  archives arose. Gen also skips any `*_data.json` whose first byte is not `{`.
- **Gen checks completeness.** `_SUCCESS` is written only when the run holds one
  CIF per seed × sample. Archiving happens only after that, and the tarball is
  verified before the directory is deleted.
- **Shared-machine manners.** MSA jobs run under `nice` / `ionice`. Before
  preallocating, a gen worker waits for its GPU to have
  `GEN_XLA_MEM_FRACTION` free, instead of crashing into someone else's job.
  `TMPDIR` points under `AF3_WORK_ROOT`, keeping hmmer's temporary files off
  the root disk. A worker will not start a variant when its output disk has
  less than `MIN_FREE_GB` free.


## Where things are

```
$AF3_WORK_ROOT/
  inputs/<batch>/            AF3 job JSONs (you put them here)
  msa/<batch>/<stem>/        MSA output + _SUCCESS / _FAILED
  gen/<batch>/<job>_data.tar.gz
  state/<batch>/<stage>/
    logs/worker<k>.log       one line per start/finish
    logs/<variant>.try<n>.log  full AF3 output of one attempt
    runs.tsv                 time, worker, slot, variant, ok/FAIL, seconds, exit code
    locks/ attempts/ pids/ running/ STOP
  tmp/
```
