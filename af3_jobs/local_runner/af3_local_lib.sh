# af3_local_lib.sh - shared machinery for msa_local.sh and gen_local.sh.
# Sourced, not run.
#
# This replaces Slurm with a pool of background worker loops. Each worker walks
# the batch's input list and claims one variant at a time with flock(1) on
# state/<batch>/<stage>/locks/<variant>.lock, then runs it.
#
# An flock is held by an open file descriptor, so the kernel drops it when the
# holder dies. A killed or crashed run therefore never leaves a stale lock, and
# nothing like GEN_TREAT_LOCKS_AS_STALE is needed. Workers also re-check "done"
# after taking the lock, so two workers (or two `start`s) cannot run the same
# variant.
#
# DRYRUN=1 works with every command. It prints what would happen, including the
# exact AF3 command line, and launches, claims, creates and deletes nothing.
# `status` is always read-only.
#
# A stage script defines these hooks and then calls `main "$@"`:
#   STAGE                         msa | gen
#   stage_setup                   set IN_ROOT / OUT_ROOT for $BATCH
#   stage_preflight               `require` everything the stage needs
#   stage_slots [arg]             one token per worker (gpu index, or 1..N)
#   stage_list_inputs             candidate JSON paths, one per line
#   stage_name_of <json>          variant name, unique within the batch
#   stage_is_done <name>          0 when the variant is finished
#   stage_ready <slot>            block until the slot can take work (GPU free)
#   stage_cmd <json> <name> <slot> <tmpdir>   fill AF3_CMD with the AF3 command
#   stage_run_one <json> <name> <slot> <tmpdir>   run it; runs under set -e
#   stage_should_wait             0 if an empty queue may still grow (gen waits on msa)
#   stage_status_extra            extra status lines
#   stage_plan_slot <slot>        dry run: describe a worker slot
#   stage_plan_after <json> <name>  dry run: what happens after AF3 returns

SELF="$(realpath "$0")"
HERE="$(dirname "$SELF")"
# shellcheck source=af3_local.env
source "${AF3_LOCAL_ENV:-$HERE/af3_local.env}"
DRYRUN="${DRYRUN:-0}"

log() { printf '%s [%s %s] %s\n' "$(date '+%F %T')" "$STAGE" "${BATCH:-}" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
dry() { [[ "$DRYRUN" == 1 ]]; }

setup_batch() {  # sets paths only; creates nothing
  BATCH="${1:-}"
  [[ -n "$BATCH" ]] || die "missing <batch> name"
  [[ "$BATCH" =~ ^[A-Za-z0-9._-]+$ ]] || die "batch name may only use letters, digits, . _ -: '$BATCH'"
  STATE_DIR="$AF3_WORK_ROOT/state/$BATCH/$STAGE"
  LOCK_DIR="$STATE_DIR/locks"
  ATTEMPT_DIR="$STATE_DIR/attempts"
  LOG_DIR="$STATE_DIR/logs"
  PID_DIR="$STATE_DIR/pids"
  RUNNING_DIR="$STATE_DIR/running"
  RUNS_TSV="$STATE_DIR/runs.tsv"
  STOP_FILE="$STATE_DIR/STOP"
  TMP_ROOT="$AF3_WORK_ROOT/tmp"
  stage_setup
}

make_dirs() {
  dry && return 0
  mkdir -p "$LOCK_DIR" "$ATTEMPT_DIR" "$LOG_DIR" "$PID_DIR" "$RUNNING_DIR" "$TMP_ROOT" "$OUT_ROOT" \
    || die "cannot create directories under AF3_WORK_ROOT=$AF3_WORK_ROOT (set it in af3_local.env)"
}

# require <description> <test command...>: die if the test fails. Under DRYRUN
# it reports every check and carries on, so one dry run lists all problems.
PREFLIGHT_FAILED=0
require() {
  local what="$1"; shift
  if "$@" >/dev/null 2>&1; then
    dry && echo "   ok       $what"
    return 0
  fi
  if dry; then echo "   MISSING  $what"; PREFLIGHT_FAILED=1; return 0; fi
  die "preflight: missing $what"
}
have_cmd() { command -v "$1"; }
# The alphafold3 version inside AF3_SIF must be AF3_EXPECT_VERSION. Exported,
# so workers launched by start can record it in their logs and outputs.
require_af3_version() {
  [[ -n "${AF3_EXPECT_VERSION:-}" ]] || return 0
  # PYTHONNOUSERSITE as in the real runs, so ~/.local packages cannot answer instead.
  AF3_VERSION_SEEN="$("$APPTAINER_BIN" exec --cleanenv --env PYTHONNOUSERSITE=1 "$AF3_SIF" python -c \
    'import importlib.metadata as m; print(m.version("alphafold3"))' 2>/dev/null | tail -n 1)"
  export AF3_VERSION_SEEN
  require "alphafold3 $AF3_EXPECT_VERSION in $AF3_SIF (it reports: ${AF3_VERSION_SEEN:-nothing})" \
    test "$AF3_VERSION_SEEN" = "$AF3_EXPECT_VERSION"
}
# Written next to _SUCCESS, so every result says which build and machine made it.
write_provenance() {  # <dir> [gpu]
  {
    echo "alphafold3_version=${AF3_VERSION_SEEN:-unknown}"
    echo "container=$AF3_SIF"
    echo "container_bytes=$(stat -c %s "$AF3_SIF" 2>/dev/null)"
    echo "container_mtime=$(stat -c %y "$AF3_SIF" 2>/dev/null)"
    echo "host=$(hostname)"
    [[ -n "${2:-}" ]] && echo "gpu=$(nvidia-smi -i "$2" --query-gpu=name --format=csv,noheader 2>/dev/null)"
    echo "finished=$(date -Is)"
  } > "$1/_PROVENANCE.txt"
}
# --env needs Singularity >= 3.6 or any Apptainer.
runtime_has_env_flag() { "$APPTAINER_BIN" exec --help 2>&1 | grep -q -- '--env'; }
writable_path() {  # the nearest existing ancestor is writable
  local d="$1"
  while [[ ! -e "$d" ]]; do d="$(dirname "$d")"; done
  [[ -d "$d" && -w "$d" ]]
}

# ---------------------------------------------------------------------------
# Apptainer binds: every host path the run touches, mounted at the same path.
# $HOME is bound by Apptainer already, and a path under one already bound is
# skipped.
# ---------------------------------------------------------------------------
build_binds() {
  BINDS=()
  local p q bound=() covered
  for p in "$AF3_WORK_ROOT" "$@" $AF3_EXTRA_BINDS; do
    [[ -n "$p" ]] || continue
    p="$(realpath -e "$p" 2>/dev/null)" || continue
    [[ "$p" == "$HOME" || "$p" == "$HOME"/* ]] && continue
    covered=0
    for q in "${bound[@]}"; do [[ "$p" == "$q" || "$p" == "$q"/* ]] && covered=1; done
    (( covered )) && continue
    bound+=("$p")
    BINDS+=(-B "$p")
  done
}

# ---------------------------------------------------------------------------
# Claims, attempts, space, workers
# ---------------------------------------------------------------------------
try_claim() {
  exec {CLAIM_FD}>>"$LOCK_DIR/$1.lock" || return 1
  if flock -n "$CLAIM_FD"; then return 0; fi
  exec {CLAIM_FD}>&-
  return 1
}
release_claim() { exec {CLAIM_FD}>&-; }

attempts_of() { cat "$ATTEMPT_DIR/$1" 2>/dev/null || echo 0; }
bump_attempts() { echo $(( $(attempts_of "$1") + 1 )) > "$ATTEMPT_DIR/$1"; }

free_gb() {  # free space of the filesystem that holds (or would hold) $1
  local d="$1"
  while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
  df -P -BG "$d" | awk 'NR==2 { sub(/G$/, "", $4); print $4 }'
}

# The flocks held on this machine right now, as /proc/locks prints them
# ("MAJ:MIN:INODE", major/minor in hex). Status reads them instead of probing
# with flock, so it never takes a lock itself and never races a worker. (The
# pid column is useless here: it is the pid of flock(1), which exits at once
# while the worker's open descriptor keeps the lock.)
held_lock_keys() { awk '$2 == "FLOCK" { print $6 }' /proc/locks 2>/dev/null | sort -u; }
lock_key() {  # <file>: the same MAJ:MIN:INODE key, from stat's st_dev + inode
  local d i
  read -r d i < <(stat -c '%d %i' "$1" 2>/dev/null) || return 1
  printf '%02x:%02x:%d\n' \
    $(( ((d >> 8) & 0xfff) | ((d >> 32) & ~0xfff) )) \
    $(( (d & 0xff) | ((d >> 12) & ~0xff) )) "$i"
}
lock_is_held() {  # <name> <held key list>
  local key
  key="$(lock_key "$LOCK_DIR/$1.lock")" || return 1
  grep -qxF "$key" <<<"$2"
}

# A pid file names a live worker only if that pid is still running this
# script's worker for this batch, so a recycled pid is never signalled.
worker_alive() {  # <pid file> <stage script path>
  local pid
  read -r pid _ < "$1" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -qF -- "$2 _worker $BATCH "
}
live_workers() {  # prints "k slot pid" for each live worker of this stage+batch
  local f k slot pid
  for f in "$PID_DIR"/worker*.pid; do
    [[ -f "$f" ]] && worker_alive "$f" "$SELF" || continue
    read -r pid _ slot _ < "$f"
    k="${f##*/worker}"; k="${k%.pid}"
    echo "$k $slot $pid"
  done
}

# Path-safe rm -rf: only a direct child of OUT_ROOT or TMP_ROOT.
safe_rmdir() {
  local d="$1"
  [[ -n "$d" && ( "$d" == "$OUT_ROOT"/?* || "$d" == "$TMP_ROOT"/?* ) && "$d" != *"/.."* ]] \
    || { echo "refusing to delete '$d'" >&2; return 1; }
  rm -rf -- "$d"
}

tmp_dir_for() { echo "$TMP_ROOT/${STAGE}_${BATCH}_$1"; }

# Sorts the batch's inputs into done / running / pending / given up.
classify_inputs() {
  C_INPUTS=(); C_PENDING=(); C_RUNNING=(); C_GIVEN=(); C_DONE=0; C_RETRY=0
  local held json name a
  mapfile -t C_INPUTS < <(stage_list_inputs)
  held="$(held_lock_keys)"
  for json in "${C_INPUTS[@]}"; do
    name="$(stage_name_of "$json")"
    if stage_is_done "$name"; then C_DONE=$(( C_DONE + 1 )); continue; fi
    if lock_is_held "$name" "$held"; then C_RUNNING+=("$name"); continue; fi
    a="$(attempts_of "$name")"
    if (( a >= MAX_ATTEMPTS )); then C_GIVEN+=("$name"); continue; fi
    (( a > 0 )) && C_RETRY=$(( C_RETRY + 1 ))
    C_PENDING+=("$json")
  done
}

print_summary() {
  echo "   inputs  : $IN_ROOT"
  echo "   outputs : $OUT_ROOT  ($(free_gb "$OUT_ROOT")G free)"
  echo "   state   : $STATE_DIR"
  printf '   total %d | done %d | running %d | pending %d (%d retrying) | gave up %d\n' \
    "${#C_INPUTS[@]}" "$C_DONE" "${#C_RUNNING[@]}" "${#C_PENDING[@]}" "$C_RETRY" "${#C_GIVEN[@]}"
  stage_status_extra 2>/dev/null || true
}

print_cmd() {  # AF3_CMD, readably: launcher line, one -B/--env per line, one --flag=value per line
  local -a lines=()
  local i=0 n=${#AF3_CMD[@]} cur="" a
  while (( i < n )); do
    a="${AF3_CMD[i]}"
    case "$a" in
      -B|--env)
        [[ -n "$cur" ]] && lines+=("$cur"); cur=""
        lines+=("$(printf '%q %q' "$a" "${AF3_CMD[i+1]}")")
        i=$(( i + 2 )); continue ;;
      --*=*|*.sif)
        [[ -n "$cur" ]] && lines+=("$cur"); cur=""
        lines+=("$(printf '%q' "$a")") ;;
      *) cur+="${cur:+ }$(printf '%q' "$a")" ;;
    esac
    i=$(( i + 1 ))
  done
  [[ -n "$cur" ]] && lines+=("$cur")
  for (( i = 0; i < ${#lines[@]}; i++ )); do
    if (( i + 1 < ${#lines[@]} )); then echo "     ${lines[i]} \\"; else echo "     ${lines[i]}"; fi
  done
}

plan_variant() {  # <json> <slot>: dry-run description of one variant
  local json="$1" slot="$2" name a
  name="$(stage_name_of "$json")"
  a="$(attempts_of "$name")"
  echo "   variant : $name"
  echo "   input   : $json"
  (( a > 0 )) && echo "   history : failed $a time(s) before (MAX_ATTEMPTS=$MAX_ATTEMPTS)"
  [[ -e "$OUT_ROOT/$name" ]] && echo "   first   : clear the partial output $OUT_ROOT/$name"
  echo "   command :"
  stage_cmd "$json" "$name" "$slot" "$(tmp_dir_for "$name")"
  print_cmd
  stage_plan_after "$json" "$name"
  echo "   log     : $LOG_DIR/$name.try$(( a + 1 )).log"
}

# ---------------------------------------------------------------------------
# One claimed variant
# ---------------------------------------------------------------------------
run_claimed() {  # <json> <name> <slot> <k>
  local json="$1" name="$2" slot="$3" k="$4"
  local attempt=$(( $(attempts_of "$name") + 1 ))
  local vlog="$LOG_DIR/$name.try$attempt.log"
  local tmp; tmp="$(tmp_dir_for "$name")"
  local t0=$SECONDS rc

  echo "$k $slot $$ $(date +%s)" > "$RUNNING_DIR/$name"
  log "start $name (worker $k, slot $slot, attempt $attempt) log: $vlog"
  safe_rmdir "$tmp" && mkdir -p "$tmp"
  {
    echo "== $STAGE $name  attempt $attempt  worker $k  slot $slot  host $(hostname)  $(date -Is)"
    echo "== input: $json"
    echo "== alphafold3 ${AF3_VERSION_SEEN:-unknown} from $AF3_SIF"
  } > "$vlog"
  # set -e applies inside the subshell; stdin is closed so nothing the worker
  # reads can be swallowed by AF3.
  ( set -euo pipefail; stage_run_one "$json" "$name" "$slot" "$tmp" ) < /dev/null >> "$vlog" 2>&1
  rc=$?
  safe_rmdir "$tmp"
  rm -f "$RUNNING_DIR/$name"

  local dt=$(( SECONDS - t0 )) status=ok
  if (( rc != 0 )); then
    status=FAIL
    bump_attempts "$name"
    [[ -d "$OUT_ROOT/$name" ]] && touch "$OUT_ROOT/$name/_FAILED"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$k" "$slot" "$name" "$status" "$dt" "$rc" >> "$RUNS_TSV"
  log "$status $name rc=$rc in $(( dt / 60 ))m$(( dt % 60 ))s"
  return "$rc"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_worker() {  # <batch> <slot> <k>   (internal; launched by start)
  DRYRUN=0
  setup_batch "$1"
  make_dirs
  local slot="$2" k="$3" consec=0 progressed final=0 json name rc
  local -a inputs
  echo "$$ $(ps -o pgid= -p $$ | tr -d ' ') $slot $(date +%s)" > "$PID_DIR/worker$k.pid"
  trap 'rm -f "$PID_DIR/worker'"$k"'.pid"' EXIT
  log "worker $k up: slot $slot, pid $$"

  while :; do
    progressed=0
    mapfile -t inputs < <(stage_list_inputs)
    for json in "${inputs[@]}"; do
      [[ -f "$STOP_FILE" ]] && { log "worker $k: STOP file found, exiting"; return 0; }
      name="$(stage_name_of "$json")"
      stage_is_done "$name" && continue
      (( $(attempts_of "$name") >= MAX_ATTEMPTS )) && continue
      stage_ready "$slot" || return 0
      try_claim "$name" || continue
      # Re-check under the lock: another worker may have just finished it.
      if stage_is_done "$name" || (( $(attempts_of "$name") >= MAX_ATTEMPTS )); then
        release_claim; continue
      fi
      if (( $(free_gb "$OUT_ROOT") < MIN_FREE_GB )); then
        release_claim
        log "worker $k: under ${MIN_FREE_GB}G free on $OUT_ROOT, exiting"
        return 3
      fi
      run_claimed "$json" "$name" "$slot" "$k"
      rc=$?
      release_claim
      progressed=1
      if (( rc == 0 )); then consec=0; else consec=$(( consec + 1 )); fi
      if (( consec >= MAX_CONSEC_FAIL )); then
        log "worker $k: $consec failures in a row, stopping. Read the logs, then 'reset' and 'start'."
        return 2
      fi
    done
    if (( progressed )); then final=0; continue; fi
    if stage_should_wait; then
      [[ -f "$STOP_FILE" ]] && { log "worker $k: STOP file found, exiting"; return 0; }
      final=0
      sleep "$POLL_SECS"
      continue
    fi
    # One more pass after the queue looks empty: an MSA that finished between
    # our scan and the MSA stage exiting would otherwise be left behind.
    (( final )) && break
    final=1
  done
  log "worker $k: nothing left to claim, exiting"
}

next_worker_index() {
  local f k next=1
  for f in "$PID_DIR"/worker*.pid; do
    [[ -f "$f" ]] || continue
    k="${f##*/worker}"; k="${k%.pid}"
    (( k >= next )) && next=$(( k + 1 ))
  done
  echo "$next"
}

cmd_start() {  # <batch> [slots]
  setup_batch "$1"
  dry && { plan_start "${2:-}"; return; }
  stage_preflight
  make_dirs
  rm -f "$STOP_FILE"

  local -a slots
  local slot_list
  slot_list="$(stage_slots "${2:-}")" || exit 1
  mapfile -t slots <<<"$slot_list"
  [[ -n "$slot_list" ]] || die "no worker slots"

  local f live k slot
  for f in "$PID_DIR"/worker*.pid; do
    [[ -f "$f" ]] && ! worker_alive "$f" "$SELF" && rm -f "$f"
  done
  live="$(live_workers)"
  if [[ -n "$live" ]]; then
    log "workers already running for this batch (new ones join them; claims prevent overlap):"
    sed 's/^/    worker /' <<<"$live"
  fi
  k="$(next_worker_index)"
  for slot in "${slots[@]}"; do
    # setsid + nohup: the worker survives logout and gets its own process
    # group, which is what `stop --now` kills.
    setsid nohup bash "$SELF" _worker "$BATCH" "$slot" "$k" \
      >> "$LOG_DIR/worker$k.log" 2>&1 < /dev/null &
    log "launched worker $k on slot $slot (log: $LOG_DIR/worker$k.log)"
    k=$(( k + 1 ))
  done
  sleep 2
  cmd_status "$BATCH"
}

plan_start() {  # [slots]
  echo "=================================================================="
  echo " DRY RUN  $STAGE start $BATCH ${1:-}"
  echo " Nothing is launched, claimed, created or deleted."
  echo "------------------------------------------------------------------ preflight"
  stage_preflight
  require "writable work root $AF3_WORK_ROOT" writable_path "$AF3_WORK_ROOT"
  local slot_list; slot_list="$(stage_slots "${1:-}")" || exit 1
  local -a slots; mapfile -t slots <<<"$slot_list"

  echo "------------------------------------------------------------------ queue"
  classify_inputs
  print_summary

  echo "------------------------------------------------------------------ workers"
  local live k slot
  live="$(live_workers)"
  [[ -n "$live" ]] && while read -r k slot _; do
    echo "   already running: worker $k on slot $slot (new workers would join it)"
  done <<<"$live"
  [[ -f "$STOP_FILE" ]] && echo "   would remove the STOP file"
  k="$(next_worker_index)"
  for slot in "${slots[@]}"; do
    echo "   would launch worker $k on slot $slot  (log: $LOG_DIR/worker$k.log)"
    stage_plan_slot "$slot"
    k=$(( k + 1 ))
  done
  local free; free="$(free_gb "$OUT_ROOT")"
  (( free < MIN_FREE_GB )) && echo "   WARNING: ${free}G free < MIN_FREE_GB=$MIN_FREE_GB, so workers would exit at once"

  local n=${#C_PENDING[@]} i shown=15 name a
  if (( n > 0 )); then
    echo "------------------------------------------------------------------ claim order"
    for (( i = 0; i < n && i < shown; i++ )); do
      name="$(stage_name_of "${C_PENDING[i]}")"
      a="$(attempts_of "$name")"
      printf '   %3d  %s%s\n' $(( i + 1 )) "$name" "$( (( a > 0 )) && echo "   (retry; failed $a time(s))")"
    done
    (( n > shown )) && echo "   ... and $(( n - shown )) more"
  fi
  (( ${#C_RUNNING[@]} )) && printf '   left to their worker (running now): %s\n' "${C_RUNNING[*]}"
  (( ${#C_GIVEN[@]} )) && printf '   skipped, gave up after %d attempts ("reset" retries): %s\n' "$MAX_ATTEMPTS" "${C_GIVEN[*]}"

  if (( n > 0 )); then
    echo "------------------------------------------------------------------ first variant, on slot ${slots[0]}"
    plan_variant "${C_PENDING[0]}" "${slots[0]}"
  else
    echo "   nothing pending: workers would start, find no work and exit"
    stage_should_wait && echo "   (gen workers would instead wait for the running MSA stage)"
  fi
  echo "=================================================================="
  (( PREFLIGHT_FAILED )) && { echo "start would stop at preflight: fix the MISSING items above."; return 1; }
  return 0
}

cmd_stop() {  # <batch> [--now]
  setup_batch "$1"
  local now=0; [[ "${2:-}" == "--now" ]] && now=1
  local live; live="$(live_workers)"
  if dry; then
    echo "DRY RUN  $STAGE stop $BATCH ${2:-}"
    echo "   would write $STOP_FILE"
    if [[ -z "$live" ]]; then echo "   no live workers"; return 0; fi
    local k slot pid
    while read -r k slot pid; do
      if (( now )); then echo "   would kill worker $k (pid $pid, slot $slot) and its AF3 process"
      else echo "   worker $k (slot $slot) would finish its current variant, then exit"; fi
    done <<<"$live"
    classify_inputs
    (( now && ${#C_RUNNING[@]} )) && echo "   interrupted (restart later, no attempt charged): ${C_RUNNING[*]}"
    return 0
  fi

  make_dirs
  touch "$STOP_FILE"
  if (( ! now )); then
    log "STOP file written. Workers finish their current variant, then exit."
    log "To kill running variants as well: $(basename "$SELF") stop $BATCH --now"
    return 0
  fi
  local f pid pgid
  for f in "$PID_DIR"/worker*.pid; do
    [[ -f "$f" ]] && worker_alive "$f" "$SELF" || continue
    read -r pid pgid _ < "$f"
    log "killing worker pid $pid (process group $pgid)"
    kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid"
  done
  sleep 10
  for f in "$PID_DIR"/worker*.pid; do
    [[ -f "$f" ]] || continue
    if worker_alive "$f" "$SELF"; then
      read -r pid pgid _ < "$f"
      kill -KILL -- "-$pgid" 2>/dev/null
    fi
    rm -f "$f"
  done
  rm -f "$RUNNING_DIR"/*
  log "stopped. Interrupted variants restart from scratch next time (no attempt is charged)."
}

cmd_reset() {  # <batch>: forget failure counts of unfinished variants
  setup_batch "$1"
  local f name n=0
  dry && echo "DRY RUN  $STAGE reset $BATCH"
  for f in "$ATTEMPT_DIR"/*; do
    [[ -f "$f" ]] || continue
    name="${f##*/}"
    stage_is_done "$name" && continue
    if dry; then echo "   would clear $name (failed $(attempts_of "$name") time(s))"
    else rm -f "$f"; fi
    n=$(( n + 1 ))
  done
  if dry; then
    echo "   $n failure count(s) would be cleared"
    [[ -f "$STOP_FILE" ]] && echo "   would remove the STOP file"
    return 0
  fi
  rm -f "$STOP_FILE"
  log "cleared failure counts of $n unfinished variant(s)"
}

cmd_one() {  # <batch> <name|json> [slot]: run one variant in the foreground
  setup_batch "$1"
  local want="${2:-}" slot="${3:-}" json="" j
  [[ -n "$want" ]] || die "usage: one <batch> <variant name or json path> [slot]"
  # Take the whole list, then its first line (piping into head could SIGPIPE
  # stage_slots while it still checks later GPUs).
  local slot_list; slot_list="$(stage_slots "$slot")" || exit 1
  slot="${slot_list%%$'\n'*}"
  while IFS= read -r j; do
    if [[ "$j" == "$want" || "$(stage_name_of "$j")" == "$want" ]]; then json="$j"; break; fi
  done < <(stage_list_inputs)
  local name; name="$(stage_name_of "${json:-$want}")"

  if dry; then
    echo "=================================================================="
    echo " DRY RUN  $STAGE one $BATCH $want (slot $slot)"
    echo "------------------------------------------------------------------ preflight"
    stage_preflight
    require "writable work root $AF3_WORK_ROOT" writable_path "$AF3_WORK_ROOT"
    echo "------------------------------------------------------------------ variant"
    if [[ -z "$json" ]]; then echo "   '$want' is not among the inputs of $BATCH"; return 1; fi
    if stage_is_done "$name"; then echo "   $name is already done: nothing would run"; return 0; fi
    lock_is_held "$name" "$(held_lock_keys)" && { echo "   $name is being run by a worker: one would refuse"; return 0; }
    plan_variant "$json" "$slot"
    echo "=================================================================="
    (( PREFLIGHT_FAILED )) && return 1
    return 0
  fi

  stage_preflight
  make_dirs
  [[ -n "$json" ]] || die "'$want' is not among the inputs of $BATCH (see: status $BATCH)"
  stage_is_done "$name" && { log "$name is already done"; return 0; }
  try_claim "$name" || die "$name is being run by a worker right now"
  run_claimed "$json" "$name" "$slot" 0
  local rc=$?
  release_claim
  log "log: $LOG_DIR/$name.try*.log"
  return $rc
}

cmd_status() {  # <batch>  (read-only, with or without DRYRUN)
  setup_batch "$1"
  classify_inputs
  echo "=================================================================="
  echo " $STAGE / $BATCH"
  print_summary

  local live; live="$(live_workers)"
  local n_live=0; [[ -n "$live" ]] && n_live="$(wc -l <<<"$live")"
  echo "   workers alive: $n_live"
  [[ -n "$live" ]] && while read -r k slot pid; do echo "     worker $k  slot $slot  pid $pid"; done <<<"$live"
  [[ -f "$STOP_FILE" ]] && echo "   STOP file present: workers exit after their current variant"

  local now f k slot pid t0 name
  now="$(date +%s)"
  for name in "${C_RUNNING[@]}"; do
    f="$RUNNING_DIR/$name"
    if [[ -f "$f" ]]; then
      read -r k slot pid t0 < "$f"
      printf '     running %-45s worker %s slot %s  %dm\n' "$name" "$k" "$slot" $(( (now - t0) / 60 ))
    else
      printf '     running %s\n' "$name"
    fi
  done
  for name in "${C_GIVEN[@]}"; do
    printf '     GAVE UP %-45s %s\n' "$name" "$(ls -t "$LOG_DIR/$name".try*.log 2>/dev/null | head -n 1)"
  done

  if [[ -s "$RUNS_TSV" ]]; then
    local med todo=$(( ${#C_PENDING[@]} + ${#C_RUNNING[@]} ))
    med="$(awk -F'\t' '$5 == "ok" { print $6 }' "$RUNS_TSV" | sort -n \
           | awk '{ v[NR] = $1 } END { if (NR) print v[int((NR + 1) / 2)] }')"
    if [[ -n "$med" ]]; then
      printf '   median per variant: %dm' $(( med / 60 ))
      if (( n_live > 0 && todo > 0 )); then
        printf '   |  rough ETA: %.1f h' "$(awk -v p="$todo" -v m="$med" -v w="$n_live" \
          'BEGIN { print p * m / w / 3600 }')"
      fi
      echo
    fi
    echo "   last runs:"
    tail -n 5 "$RUNS_TSV" | awk -F'\t' '{ printf "     %s  %-4s %-45s %4dm  (worker %s, slot %s)\n", substr($1,1,16), $5, $4, $6/60, $2, $3 }'
  fi
  echo "=================================================================="
}

usage() {
  local me; me="$(basename "$SELF")"
  cat <<USAGE
Usage:
  $me start  <batch> [$STAGE_SLOT_HELP]   launch background workers (survive logout)
  $me status <batch>                     progress, running variants, failures, ETA (read-only)
  $me stop   <batch> [--now]             finish current variants and exit (--now: kill them)
  $me reset  <batch>                     forget failure counts so given-up variants retry
  $me one    <batch> <variant> [slot]    run a single variant in the foreground (testing)

  DRYRUN=1 $me <any of the above>       show what would happen; changes nothing

$STAGE_LAYOUT_HELP
Settings: $HERE/af3_local.env
USAGE
}

main() {
  local cmd="${1:-}"
  shift || true
  case "$cmd" in
    start)   cmd_start "$@" ;;
    status)  cmd_status "$@" ;;
    stop)    cmd_stop "$@" ;;
    reset)   cmd_reset "$@" ;;
    one)     cmd_one "$@" ;;
    _worker) cmd_worker "$@" ;;
    *)       usage; exit 2 ;;
  esac
}
