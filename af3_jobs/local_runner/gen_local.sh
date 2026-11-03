#!/usr/bin/env bash
###############################################################################
# gen_local.sh - AF3 stage 2 (inference / structure generation) on a machine
# without Slurm. The counterpart of gen_scheduler.sh + gen_worker.sbatch.
#
#   ./gen_local.sh start  <batch> [GPUS]      e.g. start MSH6_GT 0,1
#   ./gen_local.sh status <batch>
#   ./gen_local.sh stop   <batch> [--now]
#   ./gen_local.sh reset  <batch>
#   ./gen_local.sh one    <batch> <variant> [GPU]
#   DRYRUN=1 ./gen_local.sh <any of the above>     (changes nothing)
#
# Inputs : *_data.json under $AF3_WORK_ROOT/msa/<batch>/ (override: GEN_IN=...),
#          taken only once their MSA run wrote _SUCCESS. If the MSA stage of the
#          same batch is still running, workers wait for more instead of exiting,
#          so both stages can be started together.
# Outputs: $AF3_WORK_ROOT/gen/<batch>/<job>_data.tar.gz (override: GEN_OUT=...),
#          holding <job>_data/<job>/..., the same layout as the Slurm-cluster archives.
###############################################################################
set -uo pipefail

STAGE=gen
STAGE_SLOT_HELP="GPUS, e.g. 0,1; default \$GEN_GPUS"
STAGE_LAYOUT_HELP="Inputs:  \$AF3_WORK_ROOT/msa/<batch>/<stem>/<job>/<job>_data.json
Outputs: \$AF3_WORK_ROOT/gen/<batch>/<job>_data.tar.gz"

source "$(dirname "$(realpath "$0")")/af3_local_lib.sh"

: "${GEN_XLA_FLAGS:=}"             # empty keeps the container's XLA_FLAGS
: "${GEN_DIFFUSION_SAMPLES:=5}"    # AF3 default; structures per seed

stage_setup() {
  IN_ROOT="${GEN_IN:-$AF3_WORK_ROOT/msa/$BATCH}"
  OUT_ROOT="${GEN_OUT:-$AF3_WORK_ROOT/gen/$BATCH}"
}

msa_stage_alive() {
  local f
  for f in "$AF3_WORK_ROOT/state/$BATCH/msa/pids/"worker*.pid; do
    [[ -f "$f" ]] && worker_alive "$f" "$HERE/msa_local.sh" && return 0
  done
  return 1
}

inputs_or_msa_running() { [[ -d "$IN_ROOT" ]] || msa_stage_alive; }
has_weights() { compgen -G "$AF3_MODEL_DIR/af3.bin*"; }

stage_preflight() {
  require "input dir $IN_ROOT (or the MSA stage of $BATCH running)" inputs_or_msa_running
  require "container $AF3_SIF" test -f "$AF3_SIF"
  require "apptainer/singularity ($APPTAINER_BIN)" have_cmd "$APPTAINER_BIN"
  require "$APPTAINER_BIN exec supports --env" runtime_has_env_flag
  require_af3_version
  require "nvidia-smi" have_cmd nvidia-smi
  require "model weights $AF3_MODEL_DIR/af3.bin*" has_weights
}

stage_slots() {
  local spec="${1:-$GEN_GPUS}" g present
  # One query for all indices. (`nvidia-smi -i N -L` is not a valid
  # combination: the real nvidia-smi rejects it whatever N is.)
  present=" $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')"
  for g in ${spec//,/ }; do
    [[ "$g" =~ ^[0-9]+$ ]] || die "GPU list must be nvidia-smi indices, got '$g'"
    [[ "$present" == *" $g "* ]] || die "no GPU with index $g (present:${present:- none})"
    echo "$g"
  done
}

# Real *_data.json files whose MSA run finished. A zero-byte or null-filled
# decoy (first byte not '{') is skipped, and one name is taken once.
#  - In the MSA stage's layout (<stem>/<job>/<job>_data.json) a file counts
#    only once its run wrote <stem>/_SUCCESS, because it may still be written.
#  - Anywhere else (e.g. finished MSAs copied in flat from the Slurm cluster) the MSA
#    stage never writes, so the file is taken as it is.
stage_list_inputs() {
  local f rel
  find -L "$IN_ROOT" -type f -name '*_data.json' 2>/dev/null | sort | while IFS= read -r f; do
    rel="${f#"$IN_ROOT"/}"
    if [[ "$GEN_REQUIRE_MSA_SUCCESS" == 1 && "$rel" == */*/* && "$rel" != */*/*/* ]]; then
      [[ -f "$(dirname "$(dirname "$f")")/_SUCCESS" ]] || continue
    fi
    [[ "$(head -c 1 "$f" | tr -d '\0')" == "{" ]] || continue
    printf '%s\n' "$f"
  done | awk -F/ '!seen[$NF]++'
}

stage_name_of() { local b="${1##*/}"; echo "${b%.json}"; }

stage_is_done() {
  if [[ "$GEN_ARCHIVE" == 1 ]]; then
    [[ -f "$OUT_ROOT/$1.tar.gz" ]]
  else
    [[ -f "$OUT_ROOT/$1/_SUCCESS" ]]
  fi
}

stage_should_wait() { msa_stage_alive; }

# With preallocation JAX grabs GEN_XLA_MEM_FRACTION of the card at start-up and
# dies if that much is not free. Wait for it rather than burn an attempt.
gpu_mem() {  # <gpu>: sets GPU_TOTAL, GPU_USED, GPU_NEED (MiB)
  read -r GPU_TOTAL GPU_USED < <(nvidia-smi -i "$1" --query-gpu=memory.total,memory.used \
                                   --format=csv,noheader,nounits | tr -d ',')
  GPU_NEED="$(awk -v t="$GPU_TOTAL" -v f="$GEN_XLA_MEM_FRACTION" 'BEGIN { printf "%d", t * f }')"
}

stage_ready() {
  local gpu="$1" said=0
  [[ "$GEN_XLA_PREALLOCATE" == true ]] || return 0
  while :; do
    gpu_mem "$gpu"
    (( GPU_TOTAL - GPU_USED >= GPU_NEED )) && return 0
    [[ -f "$STOP_FILE" ]] && return 1
    (( said )) || log "GPU $gpu has $(( GPU_TOTAL - GPU_USED )) MiB free, needs $GPU_NEED; waiting"
    said=1
    sleep 120
  done
}

stage_status_extra() {
  local all ready
  all="$(find -L "$IN_ROOT" -type f -name '*_data.json' 2>/dev/null | wc -l)"
  ready="$(stage_list_inputs | wc -l)"
  (( all > ready )) && echo "   ($(( all - ready )) more *_data.json not usable yet: MSA unfinished, or a decoy/duplicate)"
  msa_stage_alive && echo "   MSA stage for $BATCH is running; workers will wait for new MSAs"
}

n_seeds_of() {  # <data json>: length of its modelSeeds list (whitespace-insensitive)
  local seed_list
  seed_list="$(tr -d ' \n\t\r' < "$1" | grep -o '"modelSeeds":\[[0-9,]*\]' || true)"
  [[ -n "$seed_list" ]] || return 1
  echo $(( $(tr -cd ',' <<<"$seed_list" | wc -c) + 1 ))
}

stage_plan_slot() {
  local gpu="$1" name
  name="$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader 2>/dev/null)"
  gpu_mem "$gpu"
  printf '       GPU %s: %s, %s of %s MiB in use' "$gpu" "$name" "$GPU_USED" "$GPU_TOTAL"
  if [[ "$GEN_XLA_PREALLOCATE" != true ]]; then
    echo "; no preallocation, so it would start now"
  elif (( GPU_TOTAL - GPU_USED >= GPU_NEED )); then
    echo "; would preallocate $GPU_NEED MiB and start now"
  else
    echo "; needs $GPU_NEED MiB free, so it would WAIT"
  fi
  local who
  who="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
         | while IFS=', ' read -r pid mem _; do printf '%s(%s) ' "$(ps -o user= -p "$pid" 2>/dev/null)" "$mem"; done)"
  [[ -n "$who" ]] && echo "       already on it: $who"
  return 0
}

stage_plan_after() {  # <json> <name>
  local n
  if n="$(n_seeds_of "$1")"; then
    echo "   then    : require $n seeds x $GEN_DIFFUSION_SAMPLES samples = $(( n * GEN_DIFFUSION_SAMPLES )) CIFs, write _SUCCESS"
  else
    echo "   WARNING : no modelSeeds list found in the input, so the completeness check would fail"
  fi
  if [[ "$GEN_ARCHIVE" == 1 ]]; then
    echo "             archive to $OUT_ROOT/$2.tar.gz, verify it, delete the directory"
  fi
}

stage_cmd() {  # <json> <name> <gpu> <tmp>
  local json="$1" name="$2" gpu="$3" tmp="$4"
  build_binds "$IN_ROOT" "$(dirname "$(realpath -m "$json")")" "$OUT_ROOT" "$AF3_MODEL_DIR" "$tmp"
  # --cleanenv keeps the host's conda/PYTHONPATH/XLA_* out of the container;
  # in particular a host XLA_PYTHON_CLIENT_MEM_FRACTION, which this JAX
  # rejects alongside XLA_CLIENT_MEM_FRACTION.
  AF3_CMD=(
    "$APPTAINER_BIN" exec --nv --cleanenv
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID
    --env "CUDA_VISIBLE_DEVICES=$gpu"
    --env "XLA_PYTHON_CLIENT_PREALLOCATE=$GEN_XLA_PREALLOCATE"
    --env "XLA_CLIENT_MEM_FRACTION=$GEN_XLA_MEM_FRACTION"
    --env "TMPDIR=$tmp"
    --env PYTHONUNBUFFERED=1
    --env PYTHONNOUSERSITE=1
  )
  [[ -n "$GEN_XLA_FLAGS" ]] && AF3_CMD+=(--env "XLA_FLAGS=$GEN_XLA_FLAGS")
  AF3_CMD+=(
    "${BINDS[@]}"
    "$AF3_SIF"
    python "$AF3_PY"
      --json_path="$(realpath -m "$json")"
      --output_dir="$OUT_ROOT/$name"
      --model_dir="$(realpath -m "$AF3_MODEL_DIR")"
      --run_data_pipeline=false
      --run_inference=true
  )
  [[ -n "$GEN_FLASH_ATTN" ]] && AF3_CMD+=(--flash_attention_implementation="$GEN_FLASH_ATTN")
  [[ "$GEN_DIFFUSION_SAMPLES" != 5 ]] && AF3_CMD+=(--num_diffusion_samples="$GEN_DIFFUSION_SAMPLES")
  return 0
}

stage_run_one() {  # <json> <name> <gpu> <tmp>
  local json="$1" name="$2" gpu="$3" tmp="$4"
  local out="$OUT_ROOT/$name"

  if [[ -f "$out/_SUCCESS" ]]; then
    echo "== inference finished on an earlier attempt; only archiving"
  else
    # A failed or interrupted earlier attempt. Clear it so AF3 does not write
    # a timestamped sibling directory next to the partial one.
    [[ -e "$out" ]] && safe_rmdir "$out"
    mkdir -p "$out"

    stage_cmd "$json" "$name" "$gpu" "$tmp"
    echo "== $(printf '%q ' "${AF3_CMD[@]}")"
    nvidia-smi -i "$gpu" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    "${AF3_CMD[@]}"

    # Complete means one CIF per seed x sample.
    local n_seeds expected got
    n_seeds="$(n_seeds_of "$json")" || { echo "no modelSeeds list in $json"; return 1; }
    expected=$(( n_seeds * GEN_DIFFUSION_SAMPLES ))
    got="$(find "$out" -path '*seed-*_sample-*' -name '*model.cif' | wc -l)"
    echo "== structures: $got (expected $expected)"
    (( got == expected )) || { echo "incomplete output"; return 1; }
    write_provenance "$out" "$gpu"
    touch "$out/_SUCCESS"
    rm -f "$out/_FAILED"
  fi

  if [[ "$GEN_ARCHIVE" == 1 ]]; then
    local arch="$OUT_ROOT/$name.tar.gz" part="$OUT_ROOT/$name.tar.gz.partial"
    rm -f "$part"
    if command -v pigz >/dev/null; then
      tar -C "$OUT_ROOT" -cf - "$name" | pigz -p 4 > "$part"
    else
      tar -C "$OUT_ROOT" -czf "$part" "$name"
    fi
    tar -tzf "$part" > /dev/null
    mv "$part" "$arch"
    echo "== archived: $arch ($(du -h "$arch" | cut -f1))"
    safe_rmdir "$out"
  fi
}

main "$@"
