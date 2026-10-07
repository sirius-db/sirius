#!/usr/bin/env bash
# Run each query in its own process, so one query failing (e.g. a GPU OOM at
# the retry limit, which has no CPU fallback for S3 inputs) does not abort the
# rest -- the harness stops a multi-query run at the first error.
#
#   SF=3000 NAME=baseline QUERIES="1 2 3" bash bench/s3-sf1000/run-each.sh
#
# Every other knob passes through to run.sh / render-config.sh. Writes a
# one-line-per-query summary to test/tpch_performance/output/each_sf<SF>_<NAME>.tsv.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SF="${SF:-1000}"
NAME="${NAME:-each}"
QUERIES="${QUERIES:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22}"
OUT="$REPO/test/tpch_performance/output/each_sf${SF}_${NAME}.tsv"
LOGDIR="$REPO/test/tpch_performance/output/each_sf${SF}_${NAME}_logs"
mkdir -p "$LOGDIR"
printf 'query\tstatus\tseconds\n' > "$OUT"

for q in $QUERIES; do
  log="$LOGDIR/q$q.log"
  SF="$SF" QUERIES="$q" ITERS="${ITERS:-1}" NAME="${NAME}_q$q" bash "$HERE/run.sh" > "$log" 2>&1
  rc=$?
  t=$(grep -oE "q$q iter0: [0-9.]+s" "$log" | grep -oE '[0-9.]+' | tail -1)
  if [ $rc -eq 0 ] && [ -n "$t" ]; then
    status=ok
  else
    status="failed: $(grep -oE 'Underlying GPU error: .{0,90}|Error: .{0,90}' "$log" | tail -1)"
    t=""
  fi
  printf 'q%s\t%s\t%s\n' "$q" "$status" "$t" | tee -a "$OUT"
done
echo "summary: $OUT"
