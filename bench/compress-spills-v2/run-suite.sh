#!/usr/bin/env bash
# Drive one arm query-by-query, each in its own harness process.
#
# Why not just hand --queries 1-22 to performance_test.py: in grouped mode a
# failed pin raises out of _execute_multi and takes the whole run down. At SF1000
# on a 96 GB card several queries (q9 first) have a pinned column set larger than
# the GPU tier, and pinned data is not evictable, so the pin hard-fails. Running
# one query per process contains that to the query it affects and still lets the
# rest of the suite produce numbers.
#
#   bash bench/compress-spills-v2/run-suite.sh baseline
#   bash bench/compress-spills-v2/run-suite.sh spill
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

ARM="${1:-baseline}"
case "$ARM" in
  baseline) SCRIPT="$HERE/run-baseline.sh" ;;
  spill)    SCRIPT="$HERE/run-spill-compress.sh" ;;
  *) echo "usage: $0 [baseline|spill]"; exit 1 ;;
esac

ITERS="${ITERS:-3}"
QLIST="${QLIST:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY="$REPO/test/tpch_performance/output/suite_${ARM}_${STAMP}.tsv"
export PIN="${PIN:-gpu}"
export PIN_HOST_TABLES="${PIN_HOST_TABLES:-}"
# Append when resuming a partial arm (QLIST set), so a run killed part-way can be
# finished without losing the queries already measured.
if [ ! -s "$SUMMARY" ]; then
  printf 'query\tstatus\tbest_s\tdir\n' > "$SUMMARY"
fi

# Host-memory watchdog: kill the benchmark, not the machine. An OOM here took the
# whole box down mid-suite once already. Dies with this script.
WATCHDOG_LOG="$REPO/test/tpch_performance/output/watchdog_${ARM}_${STAMP}.log" \
  bash "$HERE/memory-watchdog.sh" "${WATCHDOG_FLOOR_MIB:-6144}" &
WATCHDOG_PID=$!
cleanup() {
  kill "$WATCHDOG_PID" 2>/dev/null
  pkill -9 -f "release/duckdb" 2>/dev/null
}
trap cleanup EXIT INT TERM

for q in $QLIST; do
  # A wedged shutdown from the previous query would otherwise hold the whole
  # device pool and make the next engine start look like an OOM.
  pkill -9 -f "release/duckdb" 2>/dev/null
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 500 ] && break
    sleep 2
  done

  log="$REPO/test/tpch_performance/output/${ARM}_${STAMP}_q${q}.log"
  echo "=== $ARM q$q ==="
  QUERIES="$q" ITERS="$ITERS" NAME="${ARM}_${STAMP}_q${q}" \
    timeout -s KILL 3600 bash "$SCRIPT" > "$log" 2>&1
  rc=$?

  dir=$(ls -dt "$REPO"/test/tpch_performance/output/tpch_*_"${ARM}_${STAMP}_q${q}" 2>/dev/null | head -1)
  best=$(awk -F, 'NR>1 && $4!="" {if (b=="" || $4+0<b) b=$4+0} END {if (b!="") printf "%.4f", b}' \
           "$dir/csv/runtimes.csv" 2>/dev/null)

  if [ -n "$best" ]; then
    status=ok
  elif grep -q "not enough capacity to allocate memory" "$log"; then
    status=pin_oom          # pinned column set larger than the GPU tier
  elif grep -q "exceeded maximum retry limit\|Sirius GPU execution failed" "$log"; then
    status=gpu_oom          # GPU ran out mid-query; no CPU fallback (by design)
  elif grep -q "TRIPPED" "$REPO/test/tpch_performance/output/watchdog_${ARM}_${STAMP}.log" 2>/dev/null \
       && [ $rc -ne 0 ]; then
    status=watchdog_killed  # host memory floor hit; see the watchdog log
  elif [ $rc -eq 137 ]; then
    status=timeout
  else
    status=failed
  fi
  printf '%s\t%s\t%s\t%s\n' "q$q" "$status" "${best:--}" "${dir:--}" >> "$SUMMARY"
  echo "  -> $status ${best:-}"
done

pkill -9 -f "release/duckdb" 2>/dev/null
echo
echo "summary: $SUMMARY"
column -t "$SUMMARY"
exit 0
