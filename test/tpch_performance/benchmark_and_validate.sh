#!/bin/bash
# compare_tpch_engines.sh
#
# Runs all 22 TPC-H queries for both sirius and duckdb, compares results,
# and writes two CSVs:
#   comparison_sf<SF>.csv  - per-query match/error status
#   timings_sf<SF>.csv     - long-format iteration runtimes (engine,query,iteration,runtime_s)
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./test/tpch_performance/compare_tpch_engines.sh <scale_factor> <iterations>
#
# Example:
#   ./test/tpch_performance/compare_tpch_engines.sh 1 3

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_tpch_parquet.sh"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <scale_factor> <iterations>"
    echo "Example: $0 1 3"
    exit 1
fi

SF="$1"
ITERATIONS="$2"
QUERIES=($(seq 1 22))

COMPARISON_CSV="$PROJECT_DIR/comparison_sf${SF}.csv"
TIMINGS_CSV="$PROJECT_DIR/timings_sf${SF}.csv"

echo "Scale factor: SF${SF}   Iterations: ${ITERATIONS}"
echo "=========================================="

echo ""
echo "=== Running sirius ==="
"$RUN_SCRIPT" sirius "$SF" "$ITERATIONS" "${QUERIES[@]}" || true

echo ""
echo "=== Running duckdb ==="
"$RUN_SCRIPT" duckdb "$SF" "$ITERATIONS" "${QUERIES[@]}" || true

echo ""
echo "=== Comparing results ==="
echo "=========================================="

# Returns 0 (true) if the result file contains a DuckDB error message.
has_error() {
    local file="$1"
    [[ ! -f "$file" ]] && return 0
    grep -qE "^(Error|Binder Error|Parser Error|Runtime Error|Catalog Error|Fatal Error|Invalid Error):" \
        "$file" 2>/dev/null
}

printf 'query,status\n' | tee "$COMPARISON_CSV"

ok=0; validate=0; errors=0

for q in "${QUERIES[@]}"; do
    SIRIUS_FILE="$PROJECT_DIR/result_sirius_sf${SF}_q${q}.txt"
    DUCKDB_FILE="$PROJECT_DIR/result_duckdb_sf${SF}_q${q}.txt"

    if has_error "$SIRIUS_FILE" || has_error "$DUCKDB_FILE"; then
        status="error"
        (( errors++ ))
    elif diff -q "$SIRIUS_FILE" "$DUCKDB_FILE" >/dev/null 2>&1; then
        status="success"
        (( ok++ ))
    else
        status="validation"
        (( validate++ ))
    fi

    printf 'Q%s,%s\n' "$q" "$status" | tee -a "$COMPARISON_CSV"
done

echo ""
echo "=========================================="
printf 'Summary: %d/22 success   %d validate   %d error\n' "$ok" "$validate" "$errors"
echo "Comparison CSV saved to $COMPARISON_CSV"

# Build combined timings CSV in long format.
# Source files: timings_${ENGINE}_sf${SF}_q${q}.csv
#   step,runtime_s
#   views,0.12       <- skip (view creation, not a query iteration)
#   iter_1,4.56
#   iter_2,1.23
# Output: engine,query,iteration,runtime_s
echo ""
echo "=== Building combined timings CSV ==="

printf 'engine,query,iteration,runtime_s\n' > "$TIMINGS_CSV"

for engine in sirius duckdb; do
    for q in "${QUERIES[@]}"; do
        TIMING_FILE="$PROJECT_DIR/timings_${engine}_sf${SF}_q${q}.csv"
        [[ ! -f "$TIMING_FILE" ]] && continue

        # Skip the header line and the 'views' row; extract iter_N rows.
        awk -F',' -v engine="$engine" -v query="Q${q}" '
            NR == 1 { next }                       # skip CSV header
            $1 ~ /^iter_/ {
                iter = substr($1, 6)               # strip "iter_" prefix
                printf "%s,%s,%s,%s\n", engine, query, iter, $2
            }
        ' "$TIMING_FILE" >> "$TIMINGS_CSV"
    done
done

echo "Timings CSV saved to $TIMINGS_CSV"
