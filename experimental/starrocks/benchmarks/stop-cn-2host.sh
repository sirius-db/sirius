#!/usr/bin/env bash
# Stop this host's Sirius CN(s) and, if present, the StarRocks FE.
#
#   ./benchmarks/stop-cn-2host.sh          # this host only — run on BOTH gcn-09 and gcn-18
#
# Resolves victims via /proc/<pid>/exe (and cmdline for the FE's java). Never pkill -f:
# that matches cn-2host.sh, this script, and any shell that mentioned the binary.
set -euo pipefail

say() { printf 'stop-cn-2host: %s\n' "$*"; }

# Collect PIDs that actually *are* the engine, not processes that mention it.
collect() {
  local pid exe base cmd
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    [ "$pid" = "$$" ] && continue
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null) || continue
    [ -n "$exe" ] || continue
    base=${exe##*/}; base=${base% (deleted)}
    case $base in
      sirius-starrocks-cn) printf '%s\n' "$pid" ;;
      java)
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
        case $cmd in *StarRocksFE*) printf '%s\n' "$pid" ;; esac
        ;;
      bash|sh)
        # start_fe.sh and cn-2host.sh keep the cluster alive via wait/trap.
        # Do not match stop-cn-2host.sh: the substring "cn-2host.sh" is inside this
        # script's own name, so a naive glob SIGTERMs the stopper and any relaunch.
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
        case $cmd in
          *stop-cn-2host.sh*) continue ;;
        esac
        case $cmd in
          *start_fe.sh*|*gb200-8gpu/launch.sh*|*benchmarks/cn-2host.sh*) printf '%s\n' "$pid" ;;
        esac
        ;;
    esac
  done
}

uniq_pids() { sort -u | grep -E '^[0-9]+$' || true; }

pids=$(collect | uniq_pids)
if [ -z "$pids" ]; then
  say "nothing running on $(hostname)"
else
  say "SIGTERM $(hostname): $(tr '\n' ' ' <<< "$pids")"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  deadline=$((SECONDS + 15))
  while [ "$SECONDS" -lt "$deadline" ]; do
    left=$(collect | uniq_pids)
    [ -z "$left" ] && break
    sleep 1
  done
  left=$(collect | uniq_pids)
  if [ -n "$left" ]; then
    say "SIGKILL leftovers: $(tr '\n' ' ' <<< "$left")"
    # shellcheck disable=SC2086
    kill -KILL $left 2>/dev/null || true
    sleep 1
  fi
fi

still=$(collect | uniq_pids)
if [ -n "$still" ]; then
  say "FAILED — still alive: $(tr '\n' ' ' <<< "$still")" >&2
  exit 1
fi

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)
if [ -n "$apps" ]; then
  say "engine processes gone; GPU compute-apps still held by: $(tr '\n' ' ' <<< "$apps")"
  say "idle memory.used is ~30 MiB; compute-apps must be empty before the next launch"
  exit 1
fi

say "stopped on $(hostname); nvidia-smi compute-apps empty"
