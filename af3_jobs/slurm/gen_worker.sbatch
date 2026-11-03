#!/bin/bash
###############################################################################
# gen_worker.sbatch - AF3 generation-only Slurm array worker
# Fixed diagnostic version:
#   - Logs GPU memory during generation
#   - Forces JAX/XLA memory env into Apptainer with --env
#   - Prints host + container env before AF3
###############################################################################

# The cluster's partition, QOS, GPU type and log folder; adjust for your cluster.
#SBATCH --job-name=af3_gen_array
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --partition=a100-public
#SBATCH --qos=12h_4g
#SBATCH --output=/home/%u/tmp/logs/gen_logs/gen_%x.%N.%A_%a.out
#SBATCH --error=/home/%u/tmp/logs/gen_logs/gen_%x.%N.%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

echo "=============================================================="
echo "AF3 generation worker starting"
echo "Host              : $(hostname)"
echo "Job ID            : ${SLURM_JOB_ID:-unset}"
echo "Array Task ID     : ${SLURM_ARRAY_TASK_ID:-unset}"
echo "Manifest          : ${SLURM_MANIFEST:-unset}"
echo "PWD               : $(pwd)"
echo "User              : $(whoami)"
echo "=============================================================="

# Site paths (slurm.env), exported by gen_scheduler.sh
AF3_CONTAINER="${AF3_CONTAINER:?AF3_CONTAINER is not set (slurm.env)}"
SHARED_ROOT="${SHARED_ROOT:?SHARED_ROOT is not set (slurm.env)}"
AF3_ROOT="${AF3_ROOT:?AF3_ROOT is not set (slurm.env)}"
AF3_PYTHON="${AF3_PYTHON:-/app/alphafold/run_alphafold.py}"
MODEL_DIR="${MODEL_DIR:-/notroot/models}"
GEN_OUT_ROOT="${SLURM_GEN_OUT_ROOT:-}"
GEN_ARCHIVE_EXT="${GEN_ARCHIVE_EXT:-tar.gz}"
ARCHIVE_AFTER_SUCCESS="${ARCHIVE_AFTER_SUCCESS:-1}"
DELETE_AFTER_ARCHIVE="${DELETE_AFTER_ARCHIVE:-1}"

if [[ -z "${GEN_OUT_ROOT}" ]]; then
  echo "ERROR: SLURM_GEN_OUT_ROOT is not set." >&2
  exit 20
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -z "${SLURM_MANIFEST:-}" ]]; then
  echo "ERROR: SLURM_MANIFEST is not set." >&2
  exit 10
fi

# Defensive cleanup: strip accidental trailing backslashes from sbatch --export values.
SLURM_MANIFEST="${SLURM_MANIFEST%\\}"
GEN_OUT_ROOT="${GEN_OUT_ROOT%\\}"

SLURM_GEN_INFLIGHT_DIR="${SLURM_GEN_INFLIGHT_DIR:-}"
SLURM_GEN_INFLIGHT_DIR="${SLURM_GEN_INFLIGHT_DIR%\\}"

GEN_ARCHIVE_EXT="${GEN_ARCHIVE_EXT%\\}"
AF3_CONTAINER="${AF3_CONTAINER%\\}"
AF3_PYTHON="${AF3_PYTHON%\\}"
MODEL_DIR="${MODEL_DIR%\\}"

echo "Sanitized manifest : ${SLURM_MANIFEST}"
echo "Sanitized inflight : ${SLURM_GEN_INFLIGHT_DIR}"

if [[ ! -f "${SLURM_MANIFEST}" ]]; then
  echo "ERROR: Manifest file does not exist: ${SLURM_MANIFEST}" >&2
  exit 11
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
  exit 12
fi

JSON_PATH="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SLURM_MANIFEST}")"

if [[ -z "${JSON_PATH}" ]]; then
  echo "ERROR: No manifest entry for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 13
fi

if [[ ! -f "${JSON_PATH}" ]]; then
  echo "ERROR: JSON file does not exist: ${JSON_PATH}" >&2
  exit 14
fi

BASENAME="$(basename "${JSON_PATH}" .json)"
JSON_DIR="$(dirname "${JSON_PATH}")"
OUT_DIR="${GEN_OUT_ROOT}/${BASENAME}"
ARCHIVE_PATH="${OUT_DIR}.${GEN_ARCHIVE_EXT}"
ARCHIVE_TMP="${ARCHIVE_PATH}.tmp.${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"

LOCK_FILE=""
if [[ -n "${SLURM_GEN_INFLIGHT_DIR:-}" ]]; then
  LOCK_FILE="${SLURM_GEN_INFLIGHT_DIR}/${BASENAME}.lock"
fi

mkdir -p "${GEN_OUT_ROOT}"
mkdir -p "${OUT_DIR}"
mkdir -p "/home/${USER}/tmp/logs/gen_logs" || true

echo "JSON path          : ${JSON_PATH}"
echo "JSON dir           : ${JSON_DIR}"
echo "Basename           : ${BASENAME}"
echo "SLURM_GEN_OUT_ROOT : ${SLURM_GEN_OUT_ROOT}"
echo "GEN_OUT_ROOT       : ${GEN_OUT_ROOT}"
echo "Generation out dir : ${OUT_DIR}"
echo "Archive path       : ${ARCHIVE_PATH}"
echo "Archive on success : ${ARCHIVE_AFTER_SUCCESS}"
echo "Delete after tar   : ${DELETE_AFTER_ARCHIVE}"
echo "Container          : ${AF3_CONTAINER}"
echo "AF3 script         : ${AF3_PYTHON}"
echo "MODEL_DIR          : ${MODEL_DIR}"
echo "OMP threads        : ${OMP_NUM_THREADS}"
echo "APPTAINER_BINDPATH : ${APPTAINER_BINDPATH:-}"
echo "LOCK_FILE          : ${LOCK_FILE}"

APPTAINER_BIN="${APPTAINER_BIN:-/usr/bin/apptainer}"

if [[ ! -x "${APPTAINER_BIN}" ]]; then
  echo "ERROR: Apptainer executable not found or not executable: ${APPTAINER_BIN}" >&2
  echo "PATH=${PATH}" >&2
  exit 18
fi

if [[ ! -f "${AF3_CONTAINER}" ]]; then
  echo "ERROR: Container not found: ${AF3_CONTAINER}" >&2
  exit 16
fi

echo "Host-side preflight passed."
echo "MODEL_DIR will be checked implicitly inside container via bind: ${MODEL_DIR}"

# ============================================================
# JAX/XLA memory diagnostic controls
# ============================================================
# Default is diagnostic mode because this worker is intended to measure AF3
# generation VRAM behavior. For production throughput, submit with:
#   JAX_MEMORY_DIAGNOSTIC=0
# or override individual XLA_* variables explicitly.
#
# IMPORTANT for this container/JAX version:
# Use XLA_CLIENT_MEM_FRACTION only. Do NOT also set
# XLA_PYTHON_CLIENT_MEM_FRACTION, because this JAX build rejects both being set.
JAX_MEMORY_DIAGNOSTIC="${JAX_MEMORY_DIAGNOSTIC:-1}"

configure_jax_memory_env() {
  if [[ "${JAX_MEMORY_DIAGNOSTIC}" == "1" ]]; then
    # Diagnostic mode: try to avoid JAX grabbing ~95% of the GPU up front.
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

    # This is the memory-fraction variable accepted by the AF3/JAX container.
    export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.25}"

    # Make absolutely sure the deprecated/conflicting variable is absent.
    unset XLA_PYTHON_CLIENT_MEM_FRACTION
    unset APPTAINERENV_XLA_PYTHON_CLIENT_MEM_FRACTION
  fi

  # Keep APPTAINERENV_* too, but do not rely on it exclusively.
  export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE="${APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE:-${XLA_PYTHON_CLIENT_PREALLOCATE:-}}"
  export APPTAINERENV_XLA_PYTHON_CLIENT_ALLOCATOR="${APPTAINERENV_XLA_PYTHON_CLIENT_ALLOCATOR:-${XLA_PYTHON_CLIENT_ALLOCATOR:-}}"
  export APPTAINERENV_XLA_CLIENT_MEM_FRACTION="${APPTAINERENV_XLA_CLIENT_MEM_FRACTION:-${XLA_CLIENT_MEM_FRACTION:-}}"

  # Ensure the conflicting variable is not forwarded by Apptainer.
  unset APPTAINERENV_XLA_PYTHON_CLIENT_MEM_FRACTION
}

print_jax_memory_env() {
  echo "===== JAX/XLA MEMORY DIAGNOSTIC SETTINGS ====="
  echo "JAX_MEMORY_DIAGNOSTIC=${JAX_MEMORY_DIAGNOSTIC}"

  echo "----- host env -----"
  env | grep -E '^(XLA|JAX|APPTAINERENV_XLA|APPTAINERENV_JAX)' | sort || true

  echo "----- container env after explicit --env override -----"
  srun --overlap "${APPTAINER_BIN}" exec --nv \
    --env "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
    --env "XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}" \
    --env "XLA_CLIENT_MEM_FRACTION=${XLA_CLIENT_MEM_FRACTION:-0.25}" \
    --env "XLA_PYTHON_CLIENT_MEM_FRACTION=" \
    -B "${SHARED_ROOT}:${SHARED_ROOT}" \
    -B /home/$USER:/home/$USER \
    -B "${AF3_ROOT}:/notroot" \
    "${AF3_CONTAINER}" \
    bash -lc 'unset XLA_PYTHON_CLIENT_MEM_FRACTION; env | grep -E "^(XLA|JAX)" | sort || true'

  echo "==============================================="
}

# ============================================================
# GPU memory logger for generation-stage VRAM measurement
# ============================================================
GPU_LOGGER_PID=""
GPU_MEMLOG_TOTAL=""
GPU_MEMLOG_APPS=""

start_gpu_logger() {
  local log_root
  log_root="${GEN_OUT_ROOT}/_gpu_mem_logs"
  mkdir -p "${log_root}"

  local task_id job_id
  task_id="${SLURM_ARRAY_TASK_ID:-0}"
  job_id="${SLURM_JOB_ID:-manual}"

  GPU_MEMLOG_TOTAL="${log_root}/gpu_total_${BASENAME}_${job_id}_${task_id}.csv"
  GPU_MEMLOG_APPS="${log_root}/gpu_compute_apps_${BASENAME}_${job_id}_${task_id}.csv"

  echo "timestamp,index,name,memory_used_mib,memory_total_mib,utilization_gpu" > "${GPU_MEMLOG_TOTAL}"
  echo "timestamp,pid,process_name,used_memory_mib" > "${GPU_MEMLOG_APPS}"

  (
    while true; do
      ts="$(date +%s)"

      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -v ts="${ts}" '{print ts "," $0}' >> "${GPU_MEMLOG_TOTAL}"

      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -v ts="${ts}" '{print ts "," $0}' >> "${GPU_MEMLOG_APPS}"

      sleep 1
    done
  ) &
  GPU_LOGGER_PID=$!

  echo "[GPU LOGGER] Total-GPU log    : ${GPU_MEMLOG_TOTAL}"
  echo "[GPU LOGGER] Compute-apps log : ${GPU_MEMLOG_APPS}"
}

stop_gpu_logger() {
  if [[ -n "${GPU_LOGGER_PID:-}" ]]; then
    kill "${GPU_LOGGER_PID}" 2>/dev/null || true
    wait "${GPU_LOGGER_PID}" 2>/dev/null || true
    GPU_LOGGER_PID=""
  fi
}

print_gpu_memory_summary() {
  echo
  echo "===== GENERATION GPU MEMORY SUMMARY ====="

  if [[ -n "${GPU_MEMLOG_TOTAL:-}" && -s "${GPU_MEMLOG_TOTAL}" ]]; then
    awk -F, '
    NR>1 {
      gsub(/^[ \t]+|[ \t]+$/, "", $4)
      if ($4+0 > max) max=$4+0
    }
    END {
      if (max > 0) printf "Peak total GPU memory during generation: %.2f GB\n", max/1024;
      else print "Peak total GPU memory during generation: not captured";
    }' "${GPU_MEMLOG_TOTAL}"
    echo "Total-GPU memory log: ${GPU_MEMLOG_TOTAL}"
  else
    echo "Total-GPU memory log: not found or empty"
  fi

  if [[ -n "${GPU_MEMLOG_APPS:-}" && -s "${GPU_MEMLOG_APPS}" ]]; then
    awk -F, '
    NR>1 {
      gsub(/^[ \t]+|[ \t]+$/, "", $4)
      if ($4+0 > max) max=$4+0
    }
    END {
      if (max > 0) printf "Peak compute-app process memory during generation: %.2f GB\n", max/1024;
      else print "Peak compute-app process memory during generation: not captured";
    }' "${GPU_MEMLOG_APPS}"
    echo "Compute-apps memory log: ${GPU_MEMLOG_APPS}"
  else
    echo "Compute-apps memory log: not found or empty"
  fi
}

archive_successful_output() {
  if [[ "${ARCHIVE_AFTER_SUCCESS}" != "1" ]]; then
    echo "ARCHIVE_AFTER_SUCCESS!=1, keeping output directory unarchived."
    return 0
  fi

  if [[ ! -d "${OUT_DIR}" ]]; then
    echo "ERROR: Cannot archive missing output directory: ${OUT_DIR}" >&2
    return 30
  fi

  if [[ ! -f "${OUT_DIR}/_SUCCESS" ]]; then
    echo "ERROR: Refusing to archive because _SUCCESS is missing in ${OUT_DIR}" >&2
    return 31
  fi

  if [[ -f "${ARCHIVE_PATH}" ]]; then
    echo "ERROR: Archive already exists, refusing to overwrite: ${ARCHIVE_PATH}" >&2
    return 32
  fi

  echo "Archiving successful output: ${OUT_DIR} -> ${ARCHIVE_PATH}"
  rm -f "${ARCHIVE_TMP}"
  tar -C "$(dirname "${OUT_DIR}")" -czf "${ARCHIVE_TMP}" "$(basename "${OUT_DIR}")"

  echo "Validating archive listing: ${ARCHIVE_TMP}"
  tar -tzf "${ARCHIVE_TMP}" >/dev/null

  mv "${ARCHIVE_TMP}" "${ARCHIVE_PATH}"
  echo "Archive finalized: ${ARCHIVE_PATH}"

  if [[ "${DELETE_AFTER_ARCHIVE}" == "1" ]]; then
    echo "Deleting archived output directory: ${OUT_DIR}"
    rm -rf -- "${OUT_DIR}"
  fi
}

cleanup() {
  local exit_code=$?
  echo "=== Generation worker exiting with code ${exit_code} ==="

  stop_gpu_logger
  print_gpu_memory_summary

  if [[ ${exit_code} -eq 0 ]]; then
    echo "Generation successful. Writing _SUCCESS flag."
    touch "${OUT_DIR}/_SUCCESS"
    rm -f "${OUT_DIR}/_FAILED"

    if ! archive_successful_output; then
      exit_code=40
      echo "Archiving failed. Marking task as failed." >&2
      mkdir -p "${OUT_DIR}"
      touch "${OUT_DIR}/_FAILED"
      rm -f "${OUT_DIR}/_SUCCESS"
      rm -f "${ARCHIVE_TMP}" || true
    fi
  else
    echo "Generation failed. Writing _FAILED flag."
    if [[ -d "${OUT_DIR}" ]]; then
      touch "${OUT_DIR}/_FAILED"
      rm -f "${OUT_DIR}/_SUCCESS"
    fi
    rm -f "${ARCHIVE_TMP}" || true
  fi

  if [[ -n "${LOCK_FILE}" && -f "${LOCK_FILE}" ]]; then
    echo "Removing lock file: ${LOCK_FILE}"
    rm -f "${LOCK_FILE}"
  fi

  exit ${exit_code}
}
trap cleanup EXIT HUP INT TERM

echo "=============================================================="
echo "Launching AF3 generation-only run"
echo "=============================================================="

configure_jax_memory_env
[[ "${GEN_ENV_DIAGNOSTIC:-0}" == "1" ]] && print_jax_memory_env
start_gpu_logger

set -x
srun --overlap "${APPTAINER_BIN}" exec --nv \
  --env "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  --env "XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}" \
  --env "XLA_CLIENT_MEM_FRACTION=${XLA_CLIENT_MEM_FRACTION:-0.25}" \
  --env "XLA_PYTHON_CLIENT_MEM_FRACTION=" \
  -B "${SHARED_ROOT}:${SHARED_ROOT}" \
  -B /home/$USER:/home/$USER \
  -B "${AF3_ROOT}:/notroot" \
  "${AF3_CONTAINER}" \
  bash -lc 'unset XLA_PYTHON_CLIENT_MEM_FRACTION; exec python "$@"' bash "${AF3_PYTHON}" \
    --json_path "${JSON_PATH}" \
    --output_dir "${OUT_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --run_data_pipeline=false \
    --run_inference=true
set +x

echo "=============================================================="
echo "AF3 generation worker finished successfully"
echo "JSON : ${JSON_PATH}"
echo "OUT  : ${OUT_DIR}"
echo "=============================================================="

