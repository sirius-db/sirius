#!/usr/bin/env bash
# Run each query in its own harness process, so one query failing does not
# abort the rest: the harness stops a multi-query run at the first error, and at
# these scales a GPU OOM at the retry limit (no CPU fallback for S3 inputs), a
# pinned set larger than its tier, or exhausted pinned staging is routine.
#
#   SF=3000 NAME=baseline bash bench/s3-sf1000/run-each.sh
#   SF=3000 NAME=spillcomp SPILL_COMPRESSION=1 QUERIES="3 9 21" bash bench/s3-sf1000/run-each.sh
#
# Writes test/tpch_performance/output/each_sf<SF>_<NAME>.tsv (query, status,
# best_s, dir), appending when it exists so a killed arm can be resumed with the
# remaining QUERIES. Compare two arms with ab-report.py. Every other knob passes
# through to run.sh / render-config.sh.
#
#   QUERY_TIMEOUT_S=1200     per-query wall-clock limit (SF3000 q3, the longest
#                            query that completes, takes 5-10 min; q18 thrashes)
#   WATCHDOG_FLOOR_MIB=6144  host MemAvailable floor for memory-watchdog.sh (0: off)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SF="${SF:-1000}"
NAME="${NAME:-each}"
QUERIES="${QUERIES:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22}"
QUERY_TIMEOUT_S="${QUERY_TIMEOUT_S:-1200}"
OUTDIR="$REPO/test/tpch_performance/output"
OUT="$OUTDIR/each_sf${SF}_${NAME}.tsv"
LOGDIR="$OUTDIR/each_sf${SF}_${NAME}_logs"
WDLOG="$LOGDIR/watchdog.log"
mkdir -p "$LOGDIR"
[ -s "$OUT" ] || printf 'query\tstatus\tbest_s\tdir\n' > "$OUT"

WATCHDOG_PID=""
if [ "${WATCHDOG_FLOOR_MIB:-6144}" != 0 ]; then
  WATCHDOG_LOG="$WDLOG" bash "$HERE/memory-watchdog.sh" "${WATCHDOG_FLOOR_MIB:-6144}" \
    2>/dev/null &
  WATCHDOG_PID=$!
fi
trap '[ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null' EXIT

trips() { { grep -c TRIPPED "$WDLOG" 2>/dev/null || true; } | head -1; }

for q in $QUERIES; do
  # A previous query's engine still releasing its pool would make this one's
  # startup look like an OOM; wait for the device to drain first.
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${used:-0}" -lt 500 ] && break
    sleep 2
  done

  log="$LOGDIR/q$q.log"
  before=$(trips); before=${before:-0}
  SF="$SF" QUERIES="$q" ITERS="${ITERS:-1}" NAME="${NAME}_q$q" \
    timeout -s KILL "$QUERY_TIMEOUT_S" bash "$HERE/run.sh" > "$log" 2>&1
  rc=$?
  after=$(trips); after=${after:-0}

  best=$(grep -oE "\] q$q iter[0-9]+: [0-9.]+s" "$log" | grep -oE '[0-9.]+s$' | tr -d s |
         sort -g | head -1)
  dir=$(ls -dt "$OUTDIR"/tpch_*_"${NAME}_q$q" 2>/dev/null | head -1)
  if [ $rc -eq 0 ] && [ -n "$best" ]; then
    status=ok
  elif [ "$after" -gt "$before" ]; then
    status=watchdog_killed      # host memory floor hit during this query
  elif [ $rc -eq 137 ]; then
    status=timeout
  elif grep -q 'pinned staging exhausted' "$log" "$dir"/log_dir/*.log 2>/dev/null; then
    status=staging_exhausted    # pinned host pool full: S3 reads could not stage
  elif grep -q 'exceeded maximum retry limit' "$log"; then
    status=gpu_oom              # GPU ran out mid-query; no CPU fallback by design
  elif [ "${PIN:-none}" != none ] && grep -q 'not enough capacity to allocate memory' "$log"; then
    status=pin_oom              # pinned column set larger than its tier
  else
    status=failed
  fi
  printf 'q%s\t%s\t%s\t%s\n' "$q" "$status" "${best:--}" "${dir:--}" | tee -a "$OUT"
done
echo "summary: $OUT"
