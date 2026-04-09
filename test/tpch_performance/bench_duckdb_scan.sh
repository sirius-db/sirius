#!/usr/bin/env bash
# Benchmark duckdb_scan estimation accuracy and performance.
# Usage: ./bench_duckdb_scan.sh [SF1|SF10] [DB_PATH]
#
# Requires: sirius built (pixi run -e cuda12 make release)
# Output: logs to /tmp/sirius_scan_bench/

set -uo pipefail

SF="${1:-SF1}"
DB_PATH="${2:-}"
BENCH_DIR="/tmp/sirius_scan_bench"
LOG_DIR="$BENCH_DIR/logs"
DUCKDB="build/release/duckdb"

mkdir -p "$LOG_DIR"

# Resolve DB path
if [ -z "$DB_PATH" ]; then
  case "$SF" in
    SF1)  DB_PATH="test_datasets/tpch_sf1.duckdb" ;;
    SF10) DB_PATH="test_datasets/tpch_sf10.duckdb" ;;
    *)    echo "Unknown scale factor: $SF (use SF1 or SF10)"; exit 1 ;;
  esac
fi

if [ ! -f "$DB_PATH" ]; then
  echo "Database not found: $DB_PATH"
  echo "Generate with: build/release/duckdb $DB_PATH -c 'INSTALL tpch; LOAD tpch; CALL dbgen(sf=N);'"
  exit 1
fi

echo "=== DuckDB Scan Benchmark ($SF) ==="
echo "Database: $DB_PATH"
echo ""

# Queries to benchmark (table scan through gpu_execution)
QUERIES=(
  "SELECT * FROM lineitem"
  "SELECT l_comment FROM lineitem"
  "SELECT l_shipmode, l_shipinstruct, l_comment FROM lineitem"
  "SELECT * FROM orders"
  "SELECT o_comment FROM orders"
  "SELECT * FROM customer"
  "SELECT * FROM part"
)

LABELS=(
  "lineitem_all_cols"
  "lineitem_comment_only"
  "lineitem_3_varchars"
  "orders_all_cols"
  "orders_comment_only"
  "customer_all_cols"
  "part_all_cols"
)

echo "query,tasks,rows,estimated_bytes_per_row,alloc_mb,varchar_used_mb,varchar_alloc_mb,utilization_pct,overflows,wall_ms" \
  > "$BENCH_DIR/results_$SF.csv"

for i in "${!QUERIES[@]}"; do
  Q="${QUERIES[$i]}"
  LABEL="${LABELS[$i]}"
  LOGFILE="$LOG_DIR/${LABEL}_${SF}.log"

  echo -n "Running $LABEL... "

  # Clear log and run query
  > "$LOGFILE"
  export SIRIUS_LOG_DIR="$LOG_DIR"
  export SIRIUS_LOG_LEVEL=info
  START_MS=$(date +%s%3N)
  echo "CALL gpu_execution('$Q');" | "$DUCKDB" "$DB_PATH" -unsigned > /dev/null 2>&1 || true
  END_MS=$(date +%s%3N)
  WALL_MS=$((END_MS - START_MS))

  # Find today's log (spdlog names by date)
  TODAY=$(date +%Y-%m-%d)
  SLOG="$LOG_DIR/sirius_${TODAY}.log"
  if [ ! -f "$SLOG" ]; then
    echo "SKIP (no log)"
    continue
  fi

  # Parse results
  TASKS=$(grep "duckdb_scan.*task=" "$SLOG" | grep -v "estimation\|metadata\|overflow" | wc -l)
  ROWS=$(grep "duckdb_scan.*task=" "$SLOG" | grep -v "estimation\|metadata\|overflow" \
    | grep -oP 'rows=\K\d+' | paste -sd+ | bc 2>/dev/null || echo 0)
  EST=$(grep "estimated_row_bytes=" "$SLOG" | head -1 | grep -oP 'estimated_row_bytes=\K\d+' || echo 0)
  OVERFLOWS=$(grep "varchar overflow" "$SLOG" | wc -l)

  # Sum varchar usage across all tasks
  VUSED=$(grep "duckdb_scan.*task=" "$SLOG" | grep -v "estimation\|metadata\|overflow" \
    | grep -oP 'col\d+=\K\d+(?=/)' | paste -sd+ | bc 2>/dev/null || echo 0)
  VALLOC=$(grep "duckdb_scan.*task=" "$SLOG" | grep -v "estimation\|metadata\|overflow" \
    | grep -oP 'col\d+=\d+/\K\d+' | paste -sd+ | bc 2>/dev/null || echo 0)

  ALLOC_MB=$(grep "duckdb_scan.*task=" "$SLOG" | grep -v "estimation\|metadata\|overflow" \
    | grep -oP 'alloc=\d+/\K\d+' | paste -sd+ | bc 2>/dev/null || echo 0)
  ALLOC_MB=$((ALLOC_MB / 1048576))

  UTIL=0
  if [ "$VALLOC" -gt 0 ] 2>/dev/null; then
    UTIL=$(echo "scale=1; $VUSED * 100 / $VALLOC" | bc 2>/dev/null || echo 0)
  fi

  VUSED_MB=$(echo "scale=1; $VUSED / 1048576" | bc 2>/dev/null || echo 0)
  VALLOC_MB=$(echo "scale=1; $VALLOC / 1048576" | bc 2>/dev/null || echo 0)

  echo "$LABEL,$TASKS,$ROWS,$EST,$ALLOC_MB,$VUSED_MB,$VALLOC_MB,$UTIL,$OVERFLOWS,$WALL_MS" \
    >> "$BENCH_DIR/results_$SF.csv"

  echo "OK (${TASKS} tasks, ${ROWS} rows, ${WALL_MS}ms, ${UTIL}% varchar util, ${OVERFLOWS} overflows)"

  # Clear log for next query
  > "$SLOG"
done

echo ""
echo "=== Results ==="
column -t -s',' "$BENCH_DIR/results_$SF.csv"
echo ""
echo "CSV saved to: $BENCH_DIR/results_$SF.csv"
echo "Logs saved to: $LOG_DIR/"
