#!/usr/bin/env bash
# fs_watchdog.sh <probe file> <report file> <pid>...: stop a GDS job when the shared filesystem
# stops answering.
#
# Every 2 s it runs, with a 5 s limit each, a stat and a 1 MiB direct read of <probe file> (a
# file on the filesystem the job reads from). After two failures in a row it kills the given
# pids, cancels the Slurm job if there is one, and appends what it saw to <report file>.
#
# Run it from a node-local copy with a node-local or NFS report file: it must keep working when
# the filesystem under test does not. Everything it needs after start-up is already in memory
# or on the local disk.
set -u
probe=${1:?usage: fs_watchdog.sh <probe file> <report file> <pid>...}
report=${2:?}
shift 2
fails=0
say() { echo "$(date -u +%FT%TZ) $(hostname -s) $*" >> "$report"; }
say "watchdog started: probe=$probe pids=$*"
while :; do
  if timeout 5 stat "$probe" > /dev/null 2>&1 &&
    timeout 5 dd if="$probe" of=/dev/null bs=1M count=1 iflag=direct 2> /dev/null; then
    fails=0
  else
    fails=$((fails + 1))
    say "probe failed ($fails)"
  fi
  if [ "$fails" -ge 2 ]; then
    say "filesystem not answering: killing $* and cancelling job ${SLURM_JOB_ID:-none}"
    kill -9 "$@" 2> /dev/null
    grep -E '^(Reads|Mmap|Active)' /proc/driver/nvidia-fs/stats >> "$report" 2>&1
    [ -n "${SLURM_JOB_ID:-}" ] && timeout 20 scancel "$SLURM_JOB_ID"
    exit 1
  fi
  alive=0
  for pid in "$@"; do kill -0 "$pid" 2> /dev/null && alive=1; done
  [ "$alive" = 1 ] || { say "watched processes are gone; watchdog done"; exit 0; }
  sleep 2
done
