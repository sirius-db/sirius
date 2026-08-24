#!/usr/bin/env bash
# A/B one TPC-H query at SF100 with spill compression off vs on.
#
# Standalone (does not use performance_test.py, which requires passwordless sudo
# for drop_os_cache). Reports wall-clock per arm plus the spill-compression
# counters from each arm's Sirius log.
set -uo pipefail

SF_DIR=test_datasets/tpch_parquet_sf100
QNUM=${1:-21}
ITERS=${2:-2}
OUT=./spill_ab
mkdir -p "$OUT"

QSQL=$(tr '\n' ' ' < "test/tpch_performance/tpch_queries/orig/q${QNUM}.sql" | sed "s/'/''/g")

views() {
  for t in lineitem orders customer part partsupp supplier nation region; do
    echo "CREATE VIEW ${t} AS SELECT * FROM read_parquet('${SF_DIR}/${t}/*.parquet');"
  done
}

for ARM in off on; do
  LOGDIR="$OUT/${ARM}_q${QNUM}_logs"
  rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
  SQL="SET sirius_log_backend='spdlog'; SET sirius_log_dir='${LOGDIR}'; SET sirius_log_level='debug'; $(views)"
  for i in $(seq 1 "$ITERS"); do
    SQL="${SQL} FROM gpu_execution('${QSQL}');"
  done

  echo "=== arm=${ARM} q${QNUM} iters=${ITERS}"
  START=$(date +%s.%N)
  SIRIUS_CONFIG_FILE="$(pwd)/test/tpch_performance/spill_compression_${ARM}.yaml" \
    timeout 2400 ./build/release/duckdb -c "$SQL" > "$OUT/${ARM}_q${QNUM}.out" 2>&1
  RC=$?
  END=$(date +%s.%N)
  echo "  exit=${RC} wall=$(echo "$END - $START" | bc)s"

  S=$(ls "$LOGDIR"/sirius_*.log 2>/dev/null | head -1)
  if [[ -n "${S}" ]]; then
    printf "  %-30s %s\n" "explored spill plans"      "$(grep -c 'explored spill plans' "$S")"
    printf "  %-30s %s\n" "spilled->compressed host"  "$(grep -c 'compressed host' "$S")"
    printf "  %-30s %s\n" "below threshold"           "$(grep -c 'below threshold' "$S")"
    printf "  %-30s %s\n" "spill declined (fallback)" "$(grep -c 'compressed spill declined' "$S")"
  fi
  printf "  %-30s %s\n" "simpatico encode OOM" "$(grep -c 'cpp encode: exception' "$OUT/${ARM}_q${QNUM}.out")"
done
