#!/bin/bash
###############################################################################
# msa_worker.sbatch
# AlphaFold 3 MSA-only Slurm array worker, containerized
###############################################################################
# The cluster's partition and log folder; adjust for your cluster. The container is set by
# msa_scheduler.sh (--container=$AF3_CONTAINER from slurm.env).
#SBATCH --job-name=af3_msa_array
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --partition=a100-public
#SBATCH --output=/home/%u/tmp/logs/msa_logs/msa_%x.%N.%j_%a.out
#SBATCH --error=/home/%u/tmp/logs/msa_logs/msa_%x.%N.%j_%a.err

set -euo pipefail

echo "=== AF3 MSA-only (array task) ==="
echo "Host           : $(hostname)"
echo "Job            : ${SLURM_JOB_ID:-NA}"
echo "Task           : ${SLURM_ARRAY_TASK_ID:-NA}"
echo "PWD            : $(pwd)"
echo "User           : $(whoami)"
echo "CPUs           : ${SLURM_CPUS_PER_TASK:-NA}"
echo "QOS            : ${SLURM_JOB_QOS:-NA}"

###############################################################################
# Resolve manifest and selected JSON
###############################################################################
if [[ -z "${SLURM_MANIFEST:-}" ]]; then
  echo "ERROR: SLURM_MANIFEST is not set." >&2
  exit 1
fi

echo "Manifest       : ${SLURM_MANIFEST}"

if [[ ! -f "${SLURM_MANIFEST}" ]]; then
  echo "ERROR: Manifest file not found: ${SLURM_MANIFEST}" >&2
  exit 2
fi

mapfile -t JSON_LIST < "${SLURM_MANIFEST}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Invalid task id: ${TASK_ID}" >&2
  exit 3
fi

if (( TASK_ID < 0 || TASK_ID >= ${#JSON_LIST[@]} )); then
  echo "ERROR: Task index ${TASK_ID} is out of range for manifest with ${#JSON_LIST[@]} entries." >&2
  exit 4
fi

echo "--- Manifest head ---"
head -n 5 "${SLURM_MANIFEST}" || true
echo "---------------------"

JSON_IN="${JSON_LIST[${TASK_ID}]}"
if [[ -z "${JSON_IN}" ]]; then
  echo "ERROR: Empty JSON path selected from manifest." >&2
  exit 5
fi

if [[ ! -f "${JSON_IN}" ]]; then
  echo "ERROR: JSON file does not exist: ${JSON_IN}" >&2
  exit 6
fi

echo "Selected JSON  : ${JSON_IN}"

###############################################################################
# Path setup
###############################################################################
DB_DIR="${DB_DIR:-/notroot/databases}"
MODEL_DIR="${MODEL_DIR:-/notroot/models}"
OUT_ROOT="${SLURM_OUT_ROOT:-}"

if [[ -z "${OUT_ROOT}" ]]; then
  echo "ERROR: SLURM_OUT_ROOT is not set." >&2
  exit 25
fi

mkdir -p "${OUT_ROOT}"

if [[ ! -d "${OUT_ROOT}" ]]; then
  echo "ERROR: OUT_ROOT does not exist and could not be created: ${OUT_ROOT}" >&2
  exit 22
fi

if [[ ! -w "${OUT_ROOT}" ]]; then
  echo "ERROR: OUT_ROOT is not writable: ${OUT_ROOT}" >&2
  exit 23
fi

JSON_BASENAME="$(basename "${JSON_IN}")"
JSON_STEM="${JSON_BASENAME%.json}"
OUT_DIR="${OUT_ROOT}/${JSON_STEM}"

LOCK_FILE=""
if [[ -n "${SLURM_INFLIGHT_DIR:-}" ]]; then
  LOCK_FILE="${SLURM_INFLIGHT_DIR}/${JSON_STEM}.lock"
fi

mkdir -p "${OUT_DIR}"

echo "DB_DIR         : ${DB_DIR}"
echo "MODEL_DIR      : ${MODEL_DIR}"
echo "SLURM_OUT_ROOT : ${SLURM_OUT_ROOT}"
echo "OUT_ROOT       : ${OUT_ROOT}"
echo "OUT_DIR        : ${OUT_DIR}"
echo "SLURM_INFLIGHT : ${SLURM_INFLIGHT_DIR:-}"
echo "LOCK_FILE      : ${LOCK_FILE}"

ls -ld "${OUT_ROOT}" || true
ls -ld "${OUT_DIR}" || true

###############################################################################
# Cleanup and state markers
###############################################################################
cleanup() {
    local exit_code=$?
    echo "=== Worker exiting with code ${exit_code} ==="

    if [[ ${exit_code} -eq 0 ]]; then
        echo "Pipeline successful. Writing _SUCCESS flag."
        touch "${OUT_DIR}/_SUCCESS"
        rm -f "${OUT_DIR}/_FAILED"
    else
        echo "Pipeline failed. Writing _FAILED flag."
        if [[ -d "${OUT_DIR}" ]]; then
            touch "${OUT_DIR}/_FAILED"
            rm -f "${OUT_DIR}/_SUCCESS"
        fi
    fi

    if [[ -n "${LOCK_FILE}" && -f "${LOCK_FILE}" ]]; then
        echo "Removing lock file: ${LOCK_FILE}"
        rm -f "${LOCK_FILE}"
    fi
}
trap cleanup EXIT HUP INT TERM

###############################################################################
# Preflight checks
###############################################################################
if [[ ! -d "${DB_DIR}" ]]; then
  echo "ERROR: Database directory not found: ${DB_DIR}" >&2
  exit 21
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: Model directory not found: ${MODEL_DIR}" >&2
  exit 24
fi

echo "=== Running AlphaFold3 data pipeline (MSA-only) ==="
echo "Output Directory: ${OUT_DIR}"

###############################################################################
# Main run
###############################################################################
set -x
python3 /app/alphafold/run_alphafold.py \
  --json_path="${JSON_IN}" \
  --db_dir="${DB_DIR}" \
  --model_dir="${MODEL_DIR}" \
  --output_dir="${OUT_DIR}" \
  --run_data_pipeline=true \
  --run_inference=false \
  --jackhmmer_n_cpu=2 \
  --nhmmer_n_cpu=1
set +x

echo "Done."
echo "Outputs in     : ${OUT_DIR}"
