#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "${HERE}/slurm.env"; set +a     # site settings, exported to the workers

usage() {
  cat >&2 <<USAGE
Usage:
  $0 [--force] <JSON_DIR> <QOS> <MAX_NEW> <MAX_PARALLEL> [RECURSIVE=0/1]

Environment (site paths come from slurm.env):
  GEN_OUT_ROOT              Required. Generation output root.
  GEN_STATE_ROOT            Default: \$PROJECT_ROOT/\$USER/tmp/gen_scheduler
  GEN_TREAT_LOCKS_AS_STALE  Default: 0
  GEN_ARCHIVE_EXT           Default: tar.gz
  AF3_CONTAINER             Default: from slurm.env
  AF3_PYTHON                Default: /app/alphafold/run_alphafold.py
  MODEL_DIR                 Default: /notroot/models
  GEN_WORKER                Default: gen_worker.sbatch next to this script
  DRYRUN                    Default: 0
USAGE
}

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  shift
fi

if [[ $# -lt 4 ]]; then
  usage
  exit 2
fi

JSON_DIR="$1"
QOS="$2"
MAX_NEW="$3"
MAX_PARALLEL="$4"
RECURSIVE="${5:-0}"

if [[ ! -d "$JSON_DIR" ]]; then
  echo "ERROR: JSON dir does not exist: $JSON_DIR" >&2
  exit 3
fi

if [[ -z "${GEN_OUT_ROOT:-}" ]]; then
  echo "ERROR: GEN_OUT_ROOT is not set." >&2
  exit 4
fi

USER_NAME="${USER:-$(whoami)}"

GEN_STATE_ROOT="${GEN_STATE_ROOT:-${PROJECT_ROOT}/${USER_NAME}/tmp/gen_scheduler}"
GEN_INFLIGHT_DIR="${GEN_STATE_ROOT}/inflight"
MANIFEST_DIR="${GEN_STATE_ROOT}/manifests"

GEN_ARCHIVE_EXT="${GEN_ARCHIVE_EXT:-tar.gz}"
GEN_TREAT_LOCKS_AS_STALE="${GEN_TREAT_LOCKS_AS_STALE:-0}"
DRYRUN="${DRYRUN:-0}"

GEN_WORKER="${GEN_WORKER:-${HERE}/gen_worker.sbatch}"
AF3_CONTAINER="${AF3_CONTAINER:?set AF3_CONTAINER in slurm.env}"
AF3_PYTHON="${AF3_PYTHON:-/app/alphafold/run_alphafold.py}"
MODEL_DIR="${MODEL_DIR:-/notroot/models}"

APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-${SHARED_ROOT}:${SHARED_ROOT},/home/${USER_NAME}:/home/${USER_NAME},${AF3_ROOT}:/notroot}"

mkdir -p "$GEN_INFLIGHT_DIR" "$MANIFEST_DIR" "$GEN_OUT_ROOT"

if [[ ! -f "$GEN_WORKER" ]]; then
  echo "ERROR: Worker not found: $GEN_WORKER" >&2
  exit 5
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
safe_qos="${QOS//[^A-Za-z0-9_.-]/_}"
MANIFEST="${MANIFEST_DIR}/gen_manifest_${safe_qos}__${timestamp}.txt"
: > "$MANIFEST"

TOTAL_JSONS=0
ELIGIBLE_JSONS=0
SKIPPED_DONE=0
SKIPPED_LOCKED=0
STALE_LOCKS_REMOVED=0
FAILED_RERUN_CANDIDATES=0
SKIPPED_NON_RG=0

if [[ "$RECURSIVE" == "1" ]]; then
  mapfile -t JSONS < <(find "$JSON_DIR" -type f -name "*_data.json" | sort)
else
  mapfile -t JSONS < <(find "$JSON_DIR" -maxdepth 1 -type f -name "*_data.json" | sort)
fi

TOTAL_JSONS="${#JSONS[@]}"

for json in "${JSONS[@]}"; do
  if [[ "$json" != "${SHARED_ROOT}"/* ]]; then
    SKIPPED_NON_RG=$((SKIPPED_NON_RG + 1))
    continue
  fi

  base="$(basename "$json" .json)"
  out_dir="${GEN_OUT_ROOT%/}/${base}"
  archive_path="${out_dir}.${GEN_ARCHIVE_EXT}"
  lock_file="${GEN_INFLIGHT_DIR}/${base}.lock"

  done=0
  if [[ -f "$archive_path" ]]; then
    done=1
  elif [[ -f "${out_dir}/_SUCCESS" ]]; then
    done=1
  fi

  failed=0
  if [[ -f "${out_dir}/_FAILED" ]]; then
    failed=1
  fi

  if [[ "$FORCE" != "1" && "$done" == "1" ]]; then
    SKIPPED_DONE=$((SKIPPED_DONE + 1))
    continue
  fi

  if [[ "$failed" == "1" ]]; then
    FAILED_RERUN_CANDIDATES=$((FAILED_RERUN_CANDIDATES + 1))
  fi

  if [[ -f "$lock_file" ]]; then
    if [[ "$GEN_TREAT_LOCKS_AS_STALE" == "1" ]]; then
      rm -f "$lock_file"
      STALE_LOCKS_REMOVED=$((STALE_LOCKS_REMOVED + 1))
    else
      SKIPPED_LOCKED=$((SKIPPED_LOCKED + 1))
      continue
    fi
  fi

  echo "$json" >> "$MANIFEST"
  touch "$lock_file"
  ELIGIBLE_JSONS=$((ELIGIBLE_JSONS + 1))

  if [[ "$ELIGIBLE_JSONS" -ge "$MAX_NEW" ]]; then
    break
  fi
done

if [[ "$ELIGIBLE_JSONS" -eq 0 ]]; then
  echo "No eligible JSON files remain to submit."
  echo "Skipped done            : $SKIPPED_DONE"
  echo "Skipped locked          : $SKIPPED_LOCKED"
  echo "Stale locks removed     : $STALE_LOCKS_REMOVED"
  echo "Failed rerun candidates : $FAILED_RERUN_CANDIDATES"
  echo "Skipped outside shared  : $SKIPPED_NON_RG"
  echo "To clear stale locks manually:"
  echo "  rm -f ${GEN_INFLIGHT_DIR}/*.lock"
  echo "Or intentionally redo incomplete locked variants with:"
  echo "  GEN_TREAT_LOCKS_AS_STALE=1 ... $0 $JSON_DIR $QOS $MAX_NEW $MAX_PARALLEL $RECURSIVE"
  exit 0
fi

ARRAY_RANGE="0-$((ELIGIBLE_JSONS - 1))%${MAX_PARALLEL}"

echo "=============================================================="
echo "AF3 generation scheduler"
echo "JSON dir                 : $JSON_DIR"
echo "QoS                      : $QOS"
echo "Max new (this batch)     : $MAX_NEW"
echo "Max parallel             : $MAX_PARALLEL"
echo "Recursive                : $RECURSIVE"
echo "Force                    : $FORCE"
echo "Treat locks as stale     : $GEN_TREAT_LOCKS_AS_STALE"
echo "Worker                   : $GEN_WORKER"
echo "Manifest                 : $MANIFEST"
echo "Total JSONs              : $TOTAL_JSONS"
echo "Eligible JSONs           : $ELIGIBLE_JSONS"
echo "Skipped done             : $SKIPPED_DONE"
echo "Skipped locked           : $SKIPPED_LOCKED"
echo "Stale locks removed      : $STALE_LOCKS_REMOVED"
echo "Failed rerun candidates  : $FAILED_RERUN_CANDIDATES"
echo "Skipped outside shared   : $SKIPPED_NON_RG"
echo "Array range              : $ARRAY_RANGE"
echo "GEN_OUT_ROOT             : $GEN_OUT_ROOT"
echo "GEN_STATE_ROOT           : $GEN_STATE_ROOT"
echo "GEN_ARCHIVE_EXT          : $GEN_ARCHIVE_EXT"
echo "AF3_CONTAINER            : $AF3_CONTAINER"
echo "AF3_PYTHON               : $AF3_PYTHON"
echo "MODEL_DIR                : $MODEL_DIR"
echo "APPTAINER_BINDPATH       : $APPTAINER_BINDPATH"
echo "=============================================================="

SBATCH_EXPORT="ALL,SLURM_MANIFEST=${MANIFEST},SLURM_GEN_OUT_ROOT=${GEN_OUT_ROOT%/}/,SLURM_GEN_INFLIGHT_DIR=${GEN_INFLIGHT_DIR},AF3_CONTAINER=${AF3_CONTAINER},AF3_PYTHON=${AF3_PYTHON},MODEL_DIR=${MODEL_DIR},GEN_ARCHIVE_EXT=${GEN_ARCHIVE_EXT},APPTAINER_BINDPATH=${APPTAINER_BINDPATH}"

CMD=(
  sbatch
  --qos="$QOS"
  --array="$ARRAY_RANGE"
  --export="$SBATCH_EXPORT"
  --time="${GEN_TIME:-12:00:00}"
  "$GEN_WORKER"
)

echo "Submit command:"
printf "  "
printf "%q " "${CMD[@]}"
echo

if [[ "$DRYRUN" == "1" ]]; then
  echo "DRYRUN=1, not submitting."
  exit 0
fi

"${CMD[@]}"
