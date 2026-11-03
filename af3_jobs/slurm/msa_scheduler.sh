#!/usr/bin/env bash
###############################################################################
# msa_scheduler.sh
# Chunked scheduler for AlphaFold 3 MSA-only array jobs
#
# Usage:
#   ./msa_scheduler.sh [--force] <INPUT_DIR> <QOS> <MAX_NEW> <MAX_PARALLEL> [RECURSIVE=0/1]
#
# Example (site paths come from slurm.env):
#   OUTPUT_ROOT=$PROJECT_ROOT/$USER/output_msa_only/MSH6_GT \
#   ./msa_scheduler.sh $PROJECT_ROOT/$USER/mut_jsons/MSH6_GT 4h_0g 20 10 1
###############################################################################

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "${HERE}/slurm.env"; set +a     # site settings, exported to the workers

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  shift
fi

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 [--force] <INPUT_DIR> <QOS> <MAX_NEW> <MAX_PARALLEL> [RECURSIVE=0/1]" >&2
  exit 1
fi

INPUT_DIR="$1"
QOS="$2"
MAX_NEW="$3"
MAX_PARALLEL="$4"
RECURSIVE="${5:-1}"
DRYRUN="${DRYRUN:-0}"

USER_NAME="${USER:-$(whoami)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/${USER_NAME}/output_msa_only}"
STATE_ROOT="${STATE_ROOT:-${PROJECT_ROOT}/${USER_NAME}/tmp/msa_scheduler}"
WORKER_SBAT="${WORKER_SBAT:-${HERE}/msa_worker.sbatch}"

MANIFEST_DIR="${STATE_ROOT}/manifests"
INFLIGHT_DIR="${STATE_ROOT}/inflight"

###############################################################################
# Helpers
###############################################################################
die() {
  echo "ERROR: $*" >&2
  exit 2
}

qos_to_wall() {
  local qos="$1"
  case "${qos}" in
    2h_2g)   echo "02:00:00" ;;
    4h_0g)   echo "03:59:00" ;;
    12h_4g)  echo "12:00:00" ;;
    24h_1g)  echo "1-00:00:00" ;;
    24h_4g)  echo "1-00:00:00" ;;
    72h_8g)  echo "3-00:00:00" ;;
    4d_1g)   echo "4-00:00:00" ;;
    *)       echo "" ;;
  esac
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -ge 1 ]]
}

is_done() {
  local variant_dir="$1"
  [[ -f "${variant_dir}/_SUCCESS" ]]
}

has_failed_marker() {
  local variant_dir="$1"
  [[ -f "${variant_dir}/_FAILED" ]]
}

has_lock() {
  local lock_file="$1"
  [[ -f "${lock_file}" ]]
}

needs_msa() {
  local json="$1"
  local variant_name variant_dir lock_file

  variant_name="$(basename "${json}" .json)"
  variant_dir="${OUTPUT_ROOT}/${variant_name}"
  lock_file="${INFLIGHT_DIR}/${variant_name}.lock"

  if [[ "${FORCE}" == "1" ]]; then
    rm -f "${lock_file}" || true
    return 0
  fi

  if has_lock "${lock_file}"; then
    return 1
  fi

  if has_failed_marker "${variant_dir}"; then
    return 0
  fi

  if is_done "${variant_dir}"; then
    return 1
  fi

  return 0
}

###############################################################################
# Preflight
###############################################################################
[[ -d "${INPUT_DIR}" ]] || die "Input directory not found: ${INPUT_DIR}"
[[ -f "${WORKER_SBAT}" ]] || die "Worker sbatch script not found: ${WORKER_SBAT}"
is_positive_int "${MAX_NEW}" || die "MAX_NEW must be a positive integer, got: ${MAX_NEW}"
is_positive_int "${MAX_PARALLEL}" || die "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}"

[[ "${OUTPUT_ROOT}" == "${SHARED_ROOT}"/* ]] || die "OUTPUT_ROOT must be under ${SHARED_ROOT}/*, got: ${OUTPUT_ROOT}"
[[ "${STATE_ROOT}" == "${SHARED_ROOT}"/* ]] || die "STATE_ROOT must be under ${SHARED_ROOT}/*, got: ${STATE_ROOT}"

mkdir -p "${OUTPUT_ROOT}" "${MANIFEST_DIR}" "${INFLIGHT_DIR}"

echo "======================================================="
echo "[QOS=${QOS}] MSA scheduler"
echo "  INPUT_DIR      : ${INPUT_DIR}"
echo "  MAX_NEW        : ${MAX_NEW}"
echo "  MAX_PARALLEL   : ${MAX_PARALLEL}"
echo "  RECURSIVE      : ${RECURSIVE}   (0=no, 1=yes)"
echo "  FORCE          : ${FORCE}       (0=no, 1=yes)"
echo "  DRYRUN         : ${DRYRUN}"
echo "  WORKER_SBAT    : ${WORKER_SBAT}"
echo "  OUTPUT_ROOT    : ${OUTPUT_ROOT}"
echo "  STATE_ROOT     : ${STATE_ROOT}"
echo "======================================================="

###############################################################################
# Discover inputs
###############################################################################
if [[ "${RECURSIVE}" == "1" ]]; then
  mapfile -d '' -t ALL_JSONS < <(find "${INPUT_DIR}" -type f -name '*.json' -print0 | sort -z)
else
  mapfile -d '' -t ALL_JSONS < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)
fi

if (( ${#ALL_JSONS[@]} == 0 )); then
  echo "[${QOS}] No .json files found in ${INPUT_DIR}"
  exit 0
fi

echo "[${QOS}] Found ${#ALL_JSONS[@]} JSON files in ${INPUT_DIR}"

###############################################################################
# Select eligible inputs
###############################################################################
declare -a TO_SUBMIT=()
declare -i skipped_done=0
declare -i skipped_locked=0
declare -i skipped_not_rg=0

for path in "${ALL_JSONS[@]}"; do
  abs_path="$(realpath "${path}")"

  if [[ "${abs_path}" != "${SHARED_ROOT}"/* ]]; then
    echo "[${QOS}] WARN: Skipping JSON not under ${SHARED_ROOT}: ${abs_path}" >&2
    ((skipped_not_rg+=1))
    continue
  fi

  variant_name="$(basename "${abs_path}" .json)"
  lock_file="${INFLIGHT_DIR}/${variant_name}.lock"

  if [[ "${FORCE}" != "1" && -f "${lock_file}" ]]; then
    ((skipped_locked+=1))
    continue
  fi

  if ! needs_msa "${abs_path}"; then
    ((skipped_done+=1))
    continue
  fi

  TO_SUBMIT+=( "${abs_path}" )

  if (( ${#TO_SUBMIT[@]} >= MAX_NEW )); then
    break
  fi
done

if (( ${#TO_SUBMIT[@]} == 0 )); then
  echo "[${QOS}] No eligible JSONs remain to submit."
  echo "[${QOS}] Skipped done      : ${skipped_done}"
  echo "[${QOS}] Skipped locked    : ${skipped_locked}"
  echo "[${QOS}] Skipped outside shared : ${skipped_not_rg}"
  echo "[${QOS}] To clear stale locks manually:"
  echo "  rm -f ${INFLIGHT_DIR}/*.lock"
  exit 0
fi

###############################################################################
# Build manifest
###############################################################################
N_NEW=${#TO_SUBMIT[@]}
ARRAY_RANGE="0-$((N_NEW - 1))%${MAX_PARALLEL}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${MANIFEST_DIR}/msa_manifest_${QOS}_${TIMESTAMP}.txt"

printf '%s\n' "${TO_SUBMIT[@]}" > "${MANIFEST}"

echo "[${QOS}] Eligible in this chunk : ${N_NEW}"
echo "[${QOS}] Skipped done           : ${skipped_done}"
echo "[${QOS}] Skipped locked         : ${skipped_locked}"
echo "[${QOS}] Skipped outside shared      : ${skipped_not_rg}"
echo "[${QOS}] Manifest               : ${MANIFEST}"
echo "[${QOS}] Array spec             : --array=${ARRAY_RANGE}"

###############################################################################
# Walltime
###############################################################################
WALLTIME="$(qos_to_wall "${QOS}")"
SBATCH_TIME_ARGS=()

if [[ -n "${WALLTIME}" ]]; then
  echo "[${QOS}] Using walltime: ${WALLTIME}"
  SBATCH_TIME_ARGS=( -t "${WALLTIME}" )
else
  echo "[${QOS}] No walltime mapping for ${QOS}; using worker default."
fi

###############################################################################
# Dry run
###############################################################################
if [[ "${DRYRUN}" == "1" ]]; then
  echo "[${QOS}] DRY-RUN mode enabled."
  echo "sbatch --qos=\"${QOS}\" --array=\"${ARRAY_RANGE}\" --container=\"${AF3_CONTAINER}\" --export=ALL,SLURM_MANIFEST=\"${MANIFEST}\",SLURM_OUT_ROOT=\"${OUTPUT_ROOT}\",SLURM_INFLIGHT_DIR=\"${INFLIGHT_DIR}\" ${SBATCH_TIME_ARGS[*]} \"${WORKER_SBAT}\""
  exit 0
fi

###############################################################################
# Create locks before submission
###############################################################################
for p in "${TO_SUBMIT[@]}"; do
  v="$(basename "${p}" .json)"
  : > "${INFLIGHT_DIR}/${v}.lock"
done

###############################################################################
# Submit
###############################################################################
set +e
JOB_OUTPUT=$(
  sbatch \
    --qos="${QOS}" \
    --array="${ARRAY_RANGE}" \
    --container="${AF3_CONTAINER}" \
    --export=ALL,SLURM_MANIFEST="${MANIFEST}",SLURM_OUT_ROOT="${OUTPUT_ROOT}",SLURM_INFLIGHT_DIR="${INFLIGHT_DIR}" \
    "${SBATCH_TIME_ARGS[@]}" \
    "${WORKER_SBAT}"
)
RC=$?
set -e

if [[ ${RC} -ne 0 ]]; then
  echo "[${QOS}] ERROR: sbatch failed (rc=${RC}). Cleaning newly created locks." >&2
  while read -r p; do
    [[ -z "${p}" ]] && continue
    v="$(basename "${p}" .json)"
    rm -f "${INFLIGHT_DIR}/${v}.lock" || true
  done < "${MANIFEST}"
  exit 6
fi

echo "[${QOS}] Submission result: ${JOB_OUTPUT}"
echo "[${QOS}] Submitted ${N_NEW} tasks total, max ${MAX_PARALLEL} concurrent."
echo "[${QOS}] Done."
