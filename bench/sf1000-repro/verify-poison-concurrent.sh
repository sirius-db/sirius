#!/usr/bin/env bash
# CONCURRENT poison gate: SIRIUS_POISON_FREES=1 on a mode=both run with a
# reduced stream count (default 3). A sequential poison gate (MODE=power)
# passes on binaries that still strike under concurrency — the freed-while-read
# class needs concurrent queries + refresh churn. Poison turns any read of
# freed device memory into deterministic 0xEE-derived addresses, so the FIRST
# actor that touches freed memory faults, and the on-exception coredump
# captures it red-handed (instead of a downstream victim minutes later).
#
# SIRIUS_POISON_FREES / SIRIUS_QUARANTINE_FREES are env-gated engine debug
# modes that ship separately (memory-hardening PR); on an engine without them
# they are inert and this script degrades to a plain concurrent stress gate
# with coredump-on-exception capture.
#
# SCRATCH must point at a disposable copy of the SF1000 native .duckdb (the
# run mutates it; ROLLBACK=1 restores it by WAL discard afterwards).
#
# Usage: SCRATCH=/path/to/sf1000_scratch.duckdb \
#        pixi run bash bench/sf1000-repro/verify-poison-concurrent.sh [tag] [streams]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TAG="${1:-gate-cpoison}"
NSTREAMS="${2:-3}"
SCRATCH="${SCRATCH:?set SCRATCH=/path/to/disposable sf1000 .duckdb copy}"
QUERYDIR="${QUERY_DIR:-$REPO/test_datasets/tpch_queries_sf1000}"
REFRESHDIR="${REFRESH:-$REPO/test_datasets/tpch_refresh_sf1000}"
DUMPDIR="${DUMPDIR:-$HERE/coredumps}"
LEDGER="${LEDGER:-$HERE/VERIFY-LEDGER.txt}"
# Serialize gates against other GPU work on a shared box (flock on this file).
LEASE="${GPU_LEASE:-$HERE/gpu.lease}"
mkdir -p "$DUMPDIR"

# Never contend with foreign GPU work — a shared-box gate must abort, not queue
# behind an unknown workload.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "$TAG FOREIGN-GPU-WORK abort $(date +%F_%T)" >> "$LEDGER"; exit 9
fi
before_dumps=$(ls "$DUMPDIR"/*.nvcudmp 2>/dev/null | wc -l)
(
  flock -w 7200 9 || { echo "$TAG LEASE-TIMEOUT $(date +%F_%T)" >> "$LEDGER"; exit 8; }
  cd "$REPO"
  SIRIUS_POISON_FREES=1 \
  CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
  CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
  CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 \
  CUDA_COREDUMP_FILE="$DUMPDIR/core_${TAG}_%h_%p.nvcudmp" \
  CUDA_COREDUMP_PIPE="$DUMPDIR/corepipe_${TAG}_%h_%p" \
  CFG="${CFG:-$HERE/sirius-sf1000-gate.yaml}" \
  ROLLBACK=1 MODE=both REFRESH="$REFRESHDIR" \
  timeout 5400 bash "$HERE/run-power.sh" \
    --scratch-db "$SCRATCH" \
    --vary-predicates --query-dir "$QUERYDIR" \
    --streams "$NSTREAMS"
) 9>>"$LEASE"
rc=$?
[ -e "$SCRATCH.wal" ] && rm -f "$SCRATCH.wal"
after_dumps=$(ls "$DUMPDIR"/*.nvcudmp 2>/dev/null | wc -l)
rundir=$(ls -td "$REPO"/test/tpch_performance/output/tpch_power_*_sf1000_s${NSTREAMS} 2>/dev/null | head -1)
strikes=0
if [ -n "$rundir" ]; then
  strikes=$(cat "$rundir"/log_dir/sirius_*.log 2>/dev/null | grep -cE 'signal handler|Invalid unicode|illegal address|cudaErrorIllegal' || true)
fi
echo "$TAG mode=both poison=1 streams=$NSTREAMS rc=$rc new_dumps=$((after_dumps - before_dumps)) strike_lines=${strikes} dir=${rundir##*/} $(date +%F_%T)" >> "$LEDGER"
[ $rc -eq 0 ] && [ $((after_dumps - before_dumps)) -eq 0 ] && [ "${strikes}" = "0" ]
