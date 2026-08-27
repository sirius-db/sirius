#!/usr/bin/env bash
# Verification gate: compute-sanitizer sweep on the SMALL-SCALE concurrent
# repro. SF1 data + refresh sets, mode=both, 3 streams, and the SF1 pressure
# config (tiny GPU cap -> the SF1000 downgrade/eviction churn regime at toy
# sizes). Under memcheck every invalid device access self-identifies with a
# device PC + host allocation/free backtraces — the whole surviving invalid-
# access population in one seconds-scale pass, where the 10-40x sanitizer
# overhead is irrelevant.
#
# Prerequisites: SF1 native .duckdb (generate_tpch_data.sh 1 --format duckdb)
# and SF1 refresh sets (generate_tpch_refresh.sh 1 4), plus the run-power.sh
# prerequisites (patched libcudf).
#
# Usage: pixi run bash bench/sf1000-repro/verify-memcheck-sf1.sh [tool] [streams]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TOOL="${1:-memcheck}"
NSTREAMS="${2:-3}"
SF1DB="${DB:-$REPO/test_datasets/tpch_sf1.duckdb}"
REFRESHDIR="${REFRESH:-$REPO/test_datasets/tpch_refresh_sf1}"
SCRATCH="${SCRATCH:-$REPO/test_datasets/sf1_scratch.duckdb}"
LEDGER="${LEDGER:-$HERE/VERIFY-LEDGER.txt}"
# Serialize gates against other GPU work on a shared box (flock on this file).
LEASE="${GPU_LEASE:-$HERE/gpu.lease}"
mkdir -p "$(dirname "$SCRATCH")"

# Never contend with foreign GPU work — a shared-box gate must abort, not queue
# behind an unknown workload.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "memcheck-sf1 FOREIGN-GPU-WORK abort $(date +%F_%T)" >> "$LEDGER"; exit 9
fi
# Fresh content-pristine scratch each invocation (a ~267 MB copy, cheap at SF1).
cp -f "$SF1DB" "$SCRATCH"
rm -f "$SCRATCH.wal"

SANLOG="$HERE/sanitizer_${TOOL}_$(date +%Y%m%d_%H%M%S).log"
(
  flock -w 7200 9 || { echo "memcheck-sf1 LEASE-TIMEOUT $(date +%F_%T)" >> "$LEDGER"; exit 8; }
  cd "$REPO"
  SF=1 DB="$SF1DB" REFRESH="$REFRESHDIR" \
  CFG="$HERE/sirius-sf1-memcheck.yaml" \
  SANITIZER="$TOOL" SANITIZER_LOG="$SANLOG" \
  ROLLBACK=1 MODE=both \
  timeout 10800 bash "$HERE/run-power.sh" \
    --scratch-db "$SCRATCH" \
    --streams "$NSTREAMS"
) 9>>"$LEASE"
rc=$?
rm -f "$SCRATCH.wal"
errors=$(grep -cE '^========= (Invalid|Program hit)' "$SANLOG" 2>/dev/null | head -1); errors=${errors:-0}
echo "memcheck-sf1 tool=$TOOL streams=$NSTREAMS rc=$rc sanitizer_errors=${errors} log=${SANLOG##*/} $(date +%F_%T)" >> "$LEDGER"
[ "$rc" -eq 0 ] && [ "$errors" = "0" ]
