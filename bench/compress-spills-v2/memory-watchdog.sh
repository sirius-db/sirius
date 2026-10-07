#!/usr/bin/env bash
# Host-memory watchdog for the SF1000 benchmark arms.
#
# Why this exists: a GPU OOM used to fall back to DuckDB CPU inside the same
# transaction, and a CPU TPC-H q18 at SF1000 does not fit in the ~22 GB left
# after the host tier pins 40 GB of the box's 62 GB. The kernel OOM killer then
# took the whole machine down mid-run, losing the suite. run-*.sh now sets
# enable_duckdb_fallback=false, which removes that specific path -- this
# watchdog is the backstop for every other way the box can run out of memory
# (a large pin, a spill-encode burst, an unexpected CPU operator).
#
# It kills the benchmark, NOT the machine: SIGTERM to the harness first so the
# per-query driver records a status and moves on, SIGKILL if that does not land.
# The point is to lose one query rather than the host.
#
# Usage (backgrounded by run-suite.sh; safe to run standalone):
#   bench/compress-spills-v2/memory-watchdog.sh [available_MiB_floor] [poll_s]
#
# Default floor 6144 MiB of *available* memory (MemAvailable, which excludes
# pinned pages and counts reclaimable cache). Tuned for this box: steady state
# during a run sits near 19-22 GB available, so 6 GB is well clear of normal
# operation while still leaving the kernel room to act.
set -uo pipefail

FLOOR_MIB="${1:-6144}"
POLL_S="${2:-2}"
LOG="${WATCHDOG_LOG:-/tmp/sirius-memory-watchdog.log}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG" >&2; }

log "watchdog armed: floor=${FLOOR_MIB} MiB available, poll=${POLL_S}s"

while true; do
  avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  avail_mib=$(( avail_kb / 1024 ))

  if [ "$avail_mib" -lt "$FLOOR_MIB" ]; then
    log "TRIPPED: MemAvailable ${avail_mib} MiB < ${FLOOR_MIB} MiB floor"
    log "  top RSS consumers:"
    ps -eo pid,rss,comm --sort=-rss | head -6 | while read -r l; do log "    $l"; done

    # The harness first: killing it lets run-suite.sh record a status for the
    # query and continue with the rest of the suite.
    pids=$(pgrep -f "performance_test.py" || true)
    if [ -n "$pids" ]; then
      log "  SIGTERM performance_test.py: $pids"
      kill -TERM $pids 2>/dev/null
      for _ in $(seq 1 10); do
        sleep 1
        pgrep -f "performance_test.py" >/dev/null || break
      done
    fi
    if pgrep -f "performance_test.py" >/dev/null; then
      log "  still alive after 10s; SIGKILL"
      pkill -9 -f "performance_test.py" 2>/dev/null
    fi
    # The engine process can outlive the harness (Sirius's shutdown path wedges:
    # all threads in do_exit, one blocked writing the crash-handler pipe).
    pkill -9 -f "release/duckdb" 2>/dev/null

    log "  killed; waiting for memory to recover before re-arming"
    for _ in $(seq 1 60); do
      sleep "$POLL_S"
      a=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1024 ))
      [ "$a" -ge $(( FLOOR_MIB * 2 )) ] && break
    done
    log "  re-armed (available now $(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1024 )) MiB)"
  fi

  sleep "$POLL_S"
done
