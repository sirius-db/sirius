#!/usr/bin/env bash
# Track-2 corruption gate — CONCURRENT poison run: SIRIUS_POISON_FREES=1 on a
# mode=both SF1000 run with a reduced stream count (default 3). A sequential
# poison gate (MODE=power) passes on binaries that still strike under
# concurrency — the freed-while-read class needs concurrent queries + refresh
# churn. Poison turns any read of freed device memory into deterministic
# 0xEE-derived addresses, so the FIRST actor that touches freed memory faults
# and the on-exception coredump captures it red-handed (instead of a
# downstream victim minutes later).
#
# REQUIRES the power/throughput harness (bench/sf1000-repro/run-power.sh,
# PR #1554) and a cucascade build with poison-on-free (cucascade 015dad9).
#
# Usage:
#   SCRATCH=/path/sf1000_pristine.duckdb REFRESHDIR=/path/tpch_refresh_sf1000 \
#   QUERYDIR=/path/tpch_queries_sf1000 \
#     pixi run bash bench/sf1000-repro/verify-poison-concurrent.sh [tag] [streams]
# Optional env:
#   GPU_LEASE=/path/to/lease   flock this file for the GPU (shared-box etiquette)
#   CFG=...                    config override (default: sirius-sf1000-gate.yaml,
#                              0.80 cap = higher-than-production downgrade churn)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TAG="${1:-gate-cpoison}"
NSTREAMS="${2:-3}"
SCRATCH="${SCRATCH:?set SCRATCH=/path/to/sf1000_pristine.duckdb}"
QUERYDIR="${QUERYDIR:?set QUERYDIR=/path/to/tpch_queries_sf1000}"
REFRESHDIR="${REFRESHDIR:?set REFRESHDIR=/path/to/tpch_refresh_sf1000}"
DUMPDIR="${DUMPDIR:-$HERE/coredumps}"
LEDGER="${LEDGER:-$HERE/VERIFY-LEDGER.txt}"
mkdir -p "$DUMPDIR"

[ -f "$HERE/run-power.sh" ] || {
  echo "ERROR: bench/sf1000-repro/run-power.sh not found — this gate needs the"
  echo "power/throughput harness (PR #1554)."; exit 2; }

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "$TAG FOREIGN-GPU-WORK abort $(date +%F_%T)" >> "$LEDGER"; exit 9
fi
before_dumps=$(ls "$DUMPDIR"/*.nvcudmp 2>/dev/null | wc -l)
run_gate() {
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
    --staged-refresh \
    --vary-predicates --query-dir "$QUERYDIR" \
    --streams "$NSTREAMS"
}
if [ -n "${GPU_LEASE:-}" ]; then
  ( flock -w 7200 9 || { echo "$TAG LEASE-TIMEOUT $(date +%F_%T)" >> "$LEDGER"; exit 8; }
    run_gate ) 9>>"$GPU_LEASE"
else
  run_gate
fi
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
