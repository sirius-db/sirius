#!/usr/bin/env bash
# Track-1 corruption gate: compute-sanitizer sweep on the SMALL-SCALE
# concurrent repro. SF1 data + refresh sets, mode=both, N streams, staged
# refresh, and the SF1 pressure config (tiny GPU cap -> the SF1000
# downgrade/eviction churn regime at toy sizes). Under memcheck every invalid
# device access self-identifies with device PC + host allocation/free
# backtraces — the whole surviving freed-while-in-use mouth population in one
# pass, in minutes instead of hours.
#
# REQUIRES the power/throughput harness (bench/sf1000-repro/run-power.sh,
# PR #1554). Until that merges, run this from a checkout that includes it.
#
# Usage:
#   SF1DB=/path/tpch_sf1.duckdb REFRESHDIR=/path/tpch_refresh_sf1 \
#     pixi run bash bench/sf1000-repro/verify-memcheck-sf1.sh [tool] [streams]
# Optional env:
#   GPU_LEASE=/path/to/lease   flock this file for the GPU (shared-box etiquette)
#   SCRATCH=/path/scratch.duckdb  scratch DB location (default: alongside SF1DB)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TOOL="${1:-memcheck}"
NSTREAMS="${2:-3}"
SF1DB="${SF1DB:?set SF1DB=/path/to/tpch_sf1.duckdb}"
REFRESHDIR="${REFRESHDIR:?set REFRESHDIR=/path/to/tpch_refresh_sf1}"
SCRATCH="${SCRATCH:-${SF1DB%.duckdb}_scratch.duckdb}"
LEDGER="${LEDGER:-$HERE/VERIFY-LEDGER.txt}"

[ -f "$HERE/run-power.sh" ] || {
  echo "ERROR: bench/sf1000-repro/run-power.sh not found — this gate needs the"
  echo "power/throughput harness (PR #1554)."; exit 2; }

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "memcheck-sf1 FOREIGN-GPU-WORK abort $(date +%F_%T)" >> "$LEDGER"; exit 9
fi
# Fresh content-pristine scratch each invocation (a ~267 MB copy, cheap at SF1).
cp -f "$SF1DB" "$SCRATCH"
rm -f "$SCRATCH.wal"

SANLOG="$HERE/sanitizer_${TOOL}_$(date +%Y%m%d_%H%M%S).log"
run_gate() {
  cd "$REPO"
  SF=1 DB="$SF1DB" REFRESH="$REFRESHDIR" \
  CFG="$HERE/sirius-sf1-memcheck.yaml" \
  SANITIZER="$TOOL" SANITIZER_LOG="$SANLOG" \
  ROLLBACK=1 MODE=both \
  timeout 10800 bash "$HERE/run-power.sh" \
    --scratch-db "$SCRATCH" \
    --staged-refresh \
    --streams "$NSTREAMS"
}
if [ -n "${GPU_LEASE:-}" ]; then
  ( flock -w 7200 9 || { echo "memcheck-sf1 LEASE-TIMEOUT $(date +%F_%T)" >> "$LEDGER"; exit 8; }
    run_gate ) 9>>"$GPU_LEASE"
else
  run_gate
fi
rc=$?
rm -f "$SCRATCH.wal"
errors=$(grep -cE '^========= (Invalid|Program hit)' "$SANLOG" 2>/dev/null | head -1); errors=${errors:-0}
echo "memcheck-sf1 tool=$TOOL streams=$NSTREAMS rc=$rc sanitizer_errors=${errors} log=${SANLOG##*/} $(date +%F_%T)" >> "$LEDGER"
[ "$rc" -eq 0 ] && [ "$errors" = "0" ]
