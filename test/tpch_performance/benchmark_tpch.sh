#!/bin/bash
# =============================================================================
# TPC-H Benchmark: DuckDB vs Sirius
#
# For each query, runs it twice within the SAME DuckDB session (cold then
# warm) so the warm run benefits from internal caches.  Does this for both
# plain DuckDB and Sirius, then prints a comparison table.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=/path/to/integration.cfg
#   ./test/tpch_performance/benchmark_tpch.sh [scale_factor]
#
# Default scale factor is 100.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/gpu"

SF="${1:-100}"
QUERIES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22)

PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"

if [ ! -d "$PARQUET_DIR" ]; then
    echo "ERROR: Parquet directory not found: $PARQUET_DIR"
    exit 1
fi

# Output directory
OUTPUT_DIR="$PROJECT_DIR/benchmark_results_sf${SF}"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  TPC-H Benchmark: DuckDB vs Sirius  (SF${SF})"
echo "============================================================"
echo "Queries: ${QUERIES[*]}"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Build CREATE VIEW statements
TPCH_TABLES=(customer lineitem nation orders part partsupp region supplier)
VIEW_SQL=""
for TABLE_NAME in "${TPCH_TABLES[@]}"; do
    FILES=()
    for f in "$PARQUET_DIR/${TABLE_NAME}.parquet" "$PARQUET_DIR/${TABLE_NAME}_"*.parquet; do
        [ -f "$f" ] && FILES+=("'$f'")
    done
    FILE_LIST=$(IFS=,; echo "${FILES[*]}")
    VIEW_SQL+="CREATE VIEW ${TABLE_NAME} AS SELECT * FROM read_parquet([${FILE_LIST}]);"$'\n'
done

# Initialize timing CSVs
for name in duckdb_cold duckdb_warm sirius_cold sirius_warm; do
    echo "query,seconds" > "$OUTPUT_DIR/${name}.csv"
done

# parse_timer_output <output_text>
# Extracts real (wall clock) times from DuckDB .timer on output.
# Format: "Run Time (s): real 5.419 user 20.134 sys 0.512"
# Prints one time per line (first = cold, second = warm).
parse_timer_output() {
    echo "$1" | grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+'
}

# run_duckdb_query <query_num>
# Runs the query twice in a single DuckDB session with .timer on.
# Prints two lines: cold_seconds warm_seconds
run_duckdb_query() {
    local q=$1
    local QUERY_FILE="$QUERY_DIR/q${q}.sql"
    local INNER_SQL
    INNER_SQL=$(sed -n "s/call gpu_execution('//; s/');//; p" "$QUERY_FILE" | sed "s/''/'/g")

    local TEMP_SQL
    TEMP_SQL=$(mktemp /tmp/tpch_bench_duckdb_q${q}_XXXXXX.sql)
    {
        printf '%s\n' "$VIEW_SQL"
        echo ".timer on"
        printf '%s;\n' "$INNER_SQL"
        printf '%s;\n' "$INNER_SQL"
    } > "$TEMP_SQL"

    local OUTPUT
    OUTPUT=$("$DUCKDB" -f "$TEMP_SQL" 2>&1)
    rm -f "$TEMP_SQL"

    parse_timer_output "$OUTPUT"
}

# run_sirius_query <query_num>
# Runs the gpu_execution query twice in a single DuckDB session with .timer on.
# Prints two lines: cold_seconds warm_seconds
run_sirius_query() {
    local q=$1
    local QUERY_FILE="$QUERY_DIR/q${q}.sql"
    local GPU_SQL
    GPU_SQL=$(cat "$QUERY_FILE")

    local TEMP_SQL
    TEMP_SQL=$(mktemp /tmp/tpch_bench_sirius_q${q}_XXXXXX.sql)
    {
        printf '%s\n' "$VIEW_SQL"
        echo ".timer on"
        printf '%s\n' "$GPU_SQL"
        printf '%s\n' "$GPU_SQL"
    } > "$TEMP_SQL"

    local OUTPUT
    OUTPUT=$("$DUCKDB" -f "$TEMP_SQL" 2>&1)
    rm -f "$TEMP_SQL"

    parse_timer_output "$OUTPUT"
}

# Declare timing arrays
declare -A DC DW SC SW

for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    if [ ! -f "$QUERY_FILE" ]; then
        echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
        continue
    fi

    echo "========== Q${q} =========="

    # DuckDB: cold + warm in same session
    readarray -t DUCKDB_TIMES < <(run_duckdb_query "$q")
    dc="${DUCKDB_TIMES[0]:-N/A}"
    dw="${DUCKDB_TIMES[1]:-N/A}"
    DC[$q]="$dc"
    DW[$q]="$dw"
    echo "${q},${dc}" >> "$OUTPUT_DIR/duckdb_cold.csv"
    echo "${q},${dw}" >> "$OUTPUT_DIR/duckdb_warm.csv"
    echo "  DuckDB cold: ${dc}s  warm: ${dw}s"

    # Sirius: cold + warm in same session
    readarray -t SIRIUS_TIMES < <(run_sirius_query "$q")
    sc="${SIRIUS_TIMES[0]:-N/A}"
    sw="${SIRIUS_TIMES[1]:-N/A}"
    SC[$q]="$sc"
    SW[$q]="$sw"
    echo "${q},${sc}" >> "$OUTPUT_DIR/sirius_cold.csv"
    echo "${q},${sw}" >> "$OUTPUT_DIR/sirius_warm.csv"
    echo "  Sirius cold: ${sc}s  warm: ${sw}s"

    echo ""
done

# ---------- Print comparison table ----------
echo "============================================================"
echo "  Results Summary  (SF${SF})"
echo "============================================================"
echo ""

# Print header
printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
    "Query" "DuckDB Cold" "DuckDB Warm" "Sirius Cold" "Sirius Warm" "Speedup (warm)"
printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s\n" \
    "-------" "-------------" "-------------" "-------------" "-------------" "--------------"

TOTAL_DC=0
TOTAL_DW=0
TOTAL_SC=0
TOTAL_SW=0

for q in "${QUERIES[@]}"; do
    dc="${DC[$q]:-N/A}"
    dw="${DW[$q]:-N/A}"
    sc="${SC[$q]:-N/A}"
    sw="${SW[$q]:-N/A}"

    # Compute speedup (DuckDB warm / Sirius warm)
    speedup="N/A"
    if [ "$dw" != "N/A" ] && [ "$sw" != "N/A" ]; then
        speedup=$(echo "scale=2; $dw / $sw" | bc 2>/dev/null || echo "N/A")
        if [ "$speedup" != "N/A" ]; then
            speedup="${speedup}x"
        fi
    fi

    # Format seconds
    fmt_dc="N/A"
    fmt_dw="N/A"
    fmt_sc="N/A"
    fmt_sw="N/A"
    [ "$dc" != "N/A" ] && fmt_dc=$(printf "%.2fs" "$dc")
    [ "$dw" != "N/A" ] && fmt_dw=$(printf "%.2fs" "$dw")
    [ "$sc" != "N/A" ] && fmt_sc=$(printf "%.2fs" "$sc")
    [ "$sw" != "N/A" ] && fmt_sw=$(printf "%.2fs" "$sw")

    printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
        "Q${q}" "$fmt_dc" "$fmt_dw" "$fmt_sc" "$fmt_sw" "$speedup"

    # Accumulate totals
    [ "$dc" != "N/A" ] && TOTAL_DC=$(echo "$TOTAL_DC + $dc" | bc)
    [ "$dw" != "N/A" ] && TOTAL_DW=$(echo "$TOTAL_DW + $dw" | bc)
    [ "$sc" != "N/A" ] && TOTAL_SC=$(echo "$TOTAL_SC + $sc" | bc)
    [ "$sw" != "N/A" ] && TOTAL_SW=$(echo "$TOTAL_SW + $sw" | bc)
done

# Total row
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
echo "Timing CSVs saved to: $OUTPUT_DIR/"
echo "============================================================"
