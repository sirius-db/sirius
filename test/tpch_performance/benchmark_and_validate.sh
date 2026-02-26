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
#   ./test/tpch_performance/benchmark_and_validate.sh <scale_factor> <iterations>
#
# Example:
#   ./test/tpch_performance/benchmark_and_validate.sh 1 3

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

# ---------- Print comparison table ----------
echo ""
echo "============================================================"
printf "  Results Summary  (SF%s)\n" "$SF"
echo "============================================================"
echo ""

declare -A DC DW SC SW

for q in "${QUERIES[@]}"; do
    DUCKDB_TIMING="$PROJECT_DIR/timings_duckdb_sf${SF}_q${q}.csv"
    SIRIUS_TIMING="$PROJECT_DIR/timings_sirius_sf${SF}_q${q}.csv"

    if [ -f "$DUCKDB_TIMING" ]; then
        DC[$q]=$(awk -F',' '$1=="iter_1"{print $2}' "$DUCKDB_TIMING")
        DW[$q]=$(awk -F',' '$1~/^iter_/ && $1!="iter_1"{v=$2+0; if(min==""||v<min)min=v}END{if(min!="")print min}' "$DUCKDB_TIMING")
    fi
    if [ -f "$SIRIUS_TIMING" ]; then
        SC[$q]=$(awk -F',' '$1=="iter_1"{print $2}' "$SIRIUS_TIMING")
        SW[$q]=$(awk -F',' '$1~/^iter_/ && $1!="iter_1"{v=$2+0; if(min==""||v<min)min=v}END{if(min!="")print min}' "$SIRIUS_TIMING")
    fi
done

printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
    "Query" "DuckDB Cold" "DuckDB Warm" "Sirius Cold" "Sirius Warm" "Speedup (warm)"
printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s\n" \
    "-------" "-------------" "-------------" "-------------" "-------------" "--------------"

TOTAL_DC=0; TOTAL_DW=0; TOTAL_SC=0; TOTAL_SW=0

for q in "${QUERIES[@]}"; do
    dc="${DC[$q]:-N/A}"; dw="${DW[$q]:-N/A}"
    sc="${SC[$q]:-N/A}"; sw="${SW[$q]:-N/A}"

    speedup="N/A"
    if [ "$dw" != "N/A" ] && [ "$sw" != "N/A" ]; then
        speedup=$(echo "scale=2; $dw / $sw" | bc 2>/dev/null || echo "N/A")
        [ "$speedup" != "N/A" ] && speedup="${speedup}x"
    fi

    fmt_dc="N/A"; fmt_dw="N/A"; fmt_sc="N/A"; fmt_sw="N/A"
    [ "$dc" != "N/A" ] && fmt_dc=$(printf "%.2fs" "$dc")
    [ "$dw" != "N/A" ] && fmt_dw=$(printf "%.2fs" "$dw")
    [ "$sc" != "N/A" ] && fmt_sc=$(printf "%.2fs" "$sc")
    [ "$sw" != "N/A" ] && fmt_sw=$(printf "%.2fs" "$sw")

    printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
        "Q${q}" "$fmt_dc" "$fmt_dw" "$fmt_sc" "$fmt_sw" "$speedup"

    [ "$dc" != "N/A" ] && TOTAL_DC=$(echo "$TOTAL_DC + $dc" | bc)
    [ "$dw" != "N/A" ] && TOTAL_DW=$(echo "$TOTAL_DW + $dw" | bc)
    [ "$sc" != "N/A" ] && TOTAL_SC=$(echo "$TOTAL_SC + $sc" | bc)
    [ "$sw" != "N/A" ] && TOTAL_SW=$(echo "$TOTAL_SW + $sw" | bc)
done

total_speedup="N/A"
if [ "$(echo "$TOTAL_SW > 0" | bc)" -eq 1 ]; then
    total_speedup=$(echo "scale=2; $TOTAL_DW / $TOTAL_SW" | bc 2>/dev/null || echo "N/A")
    [ "$total_speedup" != "N/A" ] && total_speedup="${total_speedup}x"
fi

printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s\n" \
    "-------" "-------------" "-------------" "-------------" "-------------" "--------------"
printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
    "TOTAL" "$(printf '%.2fs' "$TOTAL_DC")" "$(printf '%.2fs' "$TOTAL_DW")" \
    "$(printf '%.2fs' "$TOTAL_SC")" "$(printf '%.2fs' "$TOTAL_SW")" "$total_speedup"
echo ""
echo "============================================================"
