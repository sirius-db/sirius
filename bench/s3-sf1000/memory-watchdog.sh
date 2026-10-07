#!/usr/bin/env bash
# Host-memory watchdog: kill the benchmark, not the machine.
#
# The pinned host tier is not reclaimable, so on a box configured to give most of
# its RAM to Sirius anything else that grows -- a CPU operator, a pin, a burst of
# spill encodes -- runs straight into the kernel OOM killer, which once took a
# whole box down mid-suite (a DuckDB CPU replay of q18 at SF1000). run.sh disables
# that replay; this is the backstop for every other path.
#
# When MemAvailable drops below the floor it SIGTERMs the harness process (so
# run-each.sh records a status and moves on), SIGKILLs it if that does not land,
# then waits for memory to recover before re-arming. Started by run-each.sh;
# safe to run standalone:
#
#   bench/s3-sf1000/memory-watchdog.sh [available_MiB_floor] [poll_s]
#
# Default floor 6144 MiB of MemAvailable (which excludes pinned pages).
set -uo pipefail

FLOOR_MIB="${1:-6144}"
POLL_S="${2:-2}"
LOG="${WATCHDOG_LOG:-/tmp/sirius-memory-watchdog.log}"
# Anchored on the interpreter so it cannot match a shell whose command line
# merely mentions the script.
HARNESS='^[^ ]*python[0-9.]* test/tpch_performance/performance_test.py'

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG" >&2; }
avail_mib() { echo $(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1024 )); }

log "watchdog armed: floor=${FLOOR_MIB} MiB available, poll=${POLL_S}s"

while true; do
  if [ "$(avail_mib)" -lt "$FLOOR_MIB" ]; then
    log "TRIPPED: MemAvailable $(avail_mib) MiB < ${FLOOR_MIB} MiB floor"
    log "  top RSS consumers:"
    ps -eo pid,rss,comm --sort=-rss | head -6 | while read -r l; do log "    $l"; done

    pids=$(pgrep -f "$HARNESS" || true)
    if [ -n "$pids" ]; then
      log "  SIGTERM harness: $pids"
      kill -TERM $pids 2>/dev/null
      for _ in $(seq 1 10); do
        sleep 1
        pgrep -f "$HARNESS" >/dev/null || break
      done
    fi
    if pgrep -f "$HARNESS" >/dev/null; then
      log "  still alive after 10s; SIGKILL"
      pkill -9 -f "$HARNESS" 2>/dev/null
    fi

    log "  killed; waiting for memory to recover before re-arming"
    for _ in $(seq 1 60); do
      sleep "$POLL_S"
      [ "$(avail_mib)" -ge $(( FLOOR_MIB * 2 )) ] && break
    done
    log "  re-armed (available now $(avail_mib) MiB)"
  fi
  sleep "$POLL_S"
done
