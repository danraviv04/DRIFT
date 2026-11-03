#!/usr/bin/env bash
###############################################################################
# msa_local.sh - AF3 stage 1 (data pipeline / MSAs only) on a machine without
# Slurm. The counterpart of msa_scheduler.sh + msa_worker.sbatch (../slurm/).
#
#   ./msa_local.sh start  <batch> [N_WORKERS]
#   ./msa_local.sh status <batch>
#   ./msa_local.sh stop   <batch> [--now]
#   ./msa_local.sh reset  <batch>
#   ./msa_local.sh one    <batch> <variant>
#   DRYRUN=1 ./msa_local.sh <any of the above>     (changes nothing)
#
# Inputs : $AF3_WORK_ROOT/inputs/<batch>/**/*.json   (override: MSA_IN=...)
# Outputs: $AF3_WORK_ROOT/msa/<batch>/<stem>/<job>/<job>_data.json + _SUCCESS
#          (override: MSA_OUT=...). Same layout as the Slurm cluster's output_msa_only,
#          so gen_local.sh (or slurm/gen_scheduler.sh) reads it directly.
###############################################################################
set -uo pipefail

STAGE=msa
STAGE_SLOT_HELP="N_WORKERS, default \$MSA_WORKERS"
STAGE_LAYOUT_HELP="Inputs:  \$AF3_WORK_ROOT/inputs/<batch>/**/*.json
Outputs: \$AF3_WORK_ROOT/msa/<batch>/<stem>/   (_SUCCESS when done)"

source "$(dirname "$(realpath "$0")")/af3_local_lib.sh"

# Files the protein search needs. RNA databases are only needed for RNA chains.
MSA_REQUIRED_DB=(
  uniref90_2022_05.fa
  mgy_clusters_2022_05.fa
  bfd-first_non_consensus_sequences.fasta
  uniprot_all_2021_04.fa
  pdb_seqres_2022_09_28.fasta
  mmcif_files
)
MSA_RNA_DB=(
  nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
  rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
  rnacentral_active_seq_id_90_cov_80_linclust.fasta
)

stage_setup() {
  IN_ROOT="${MSA_IN:-$AF3_WORK_ROOT/inputs/$BATCH}"
  OUT_ROOT="${MSA_OUT:-$AF3_WORK_ROOT/msa/$BATCH}"
}

has_inputs() { [[ -n "$(stage_list_inputs | head -n 1)" ]]; }

stage_preflight() {
  require "input dir $IN_ROOT" test -d "$IN_ROOT"
  require "at least one job *.json under it" has_inputs
  require "container $AF3_SIF" test -f "$AF3_SIF"
  require "apptainer/singularity ($APPTAINER_BIN)" have_cmd "$APPTAINER_BIN"
  require "$APPTAINER_BIN exec supports --env" runtime_has_env_flag
  require_af3_version
  local f
  for f in "${MSA_REQUIRED_DB[@]}"; do
    require "database $AF3_DB_DIR/$f" test -e "$AF3_DB_DIR/$f"
  done
  for f in "${MSA_RNA_DB[@]}"; do
    [[ -e "$AF3_DB_DIR/$f" ]] || echo "   note     $f absent (only RNA chains need it)"
  done
}

stage_slots() {
  local n="${1:-$MSA_WORKERS}"
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "N_WORKERS must be a positive integer, got '$n'"
  seq 1 "$n"
}

stage_list_inputs() {
  find -L "$IN_ROOT" -type f -name '*.json' ! -name '*_manifest.json' ! -name '*_data.json' \
    2>/dev/null | sort
}

stage_name_of() { local b="${1##*/}"; echo "${b%.json}"; }
stage_is_done() { [[ -f "$OUT_ROOT/$1/_SUCCESS" ]]; }
stage_ready() { return 0; }
stage_should_wait() { return 1; }
stage_status_extra() {
  echo "   machine : $(nproc) cores, load $(cut -d' ' -f1 /proc/loadavg); each worker runs ~4 hmmer searches x MSA_N_CPU=$MSA_N_CPU threads"
}

stage_plan_slot() {
  echo "       CPU worker: hmmer searches at $MSA_N_CPU threads each, under nice $MSA_NICE"
}

stage_plan_after() {  # <json> <name>
  echo "   then    : require one real *_data.json in $OUT_ROOT/$2, write _SUCCESS"
}

stage_cmd() {  # <json> <name> <slot> <tmp>
  local json="$1" name="$2" tmp="$4"
  build_binds "$IN_ROOT" "$OUT_ROOT" "$AF3_DB_DIR" "$tmp"
  AF3_CMD=(nice -n "$MSA_NICE")
  command -v ionice >/dev/null && AF3_CMD+=(ionice -c2 -n7)
  AF3_CMD+=(
    "$APPTAINER_BIN" exec --cleanenv
    --env "TMPDIR=$tmp"
    --env PYTHONUNBUFFERED=1
    --env PYTHONNOUSERSITE=1
    "${BINDS[@]}"
    "$AF3_SIF"
    python "$AF3_PY"
      --json_path="$(realpath -m "$json")"
      --output_dir="$OUT_ROOT/$name"
      --db_dir="$(realpath -m "$AF3_DB_DIR")"
      --run_data_pipeline=true
      --run_inference=false
      --jackhmmer_n_cpu="$MSA_N_CPU"
      --nhmmer_n_cpu="$MSA_N_CPU"
  )
}

stage_run_one() {  # <json> <name> <slot> <tmp>
  local json="$1" name="$2" tmp="$4"
  local out="$OUT_ROOT/$name"

  # Anything here is a failed or interrupted earlier attempt (we hold the lock
  # and _SUCCESS is absent). Clear it: AF3 would otherwise write a timestamped
  # sibling directory, and leave a second, partial *_data.json beside the real one.
  [[ -e "$out" ]] && safe_rmdir "$out"
  mkdir -p "$out"

  stage_cmd "$json" "$name" "$3" "$tmp"
  echo "== $(printf '%q ' "${AF3_CMD[@]}")"
  "${AF3_CMD[@]}"

  # Success means exactly one real *_data.json (a zero-byte or null-filled
  # decoy does not count: see the AF3 archive-layout notes).
  local -a data
  mapfile -t data < <(find "$out" -name '*_data.json' -type f)
  (( ${#data[@]} == 1 )) || { echo "expected one *_data.json in $out, found ${#data[@]}"; return 1; }
  [[ "$(head -c 1 "${data[0]}" | tr -d '\0')" == "{" ]] || { echo "not JSON: ${data[0]}"; return 1; }
  echo "== data json: ${data[0]} ($(du -h "${data[0]}" | cut -f1))"
  write_provenance "$out"
  touch "$out/_SUCCESS"
  rm -f "$out/_FAILED"
}

main "$@"
