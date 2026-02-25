#!/bin/bash
# Run TPC-H queries against Parquet files using plain DuckDB (no Sirius)
# Used as a baseline for validating Sirius results.
#
# Usage:
#   ./test/tpch_performance/run_tpch_parquet_duckdb.sh <scale_factor> <query_numbers...>
#
# Environment variables:
#   TIMING_CSV  - path to write per-query timing CSV (optional)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/gpu"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <scale_factor> <query_numbers...>"
    exit 1
fi

SF="$1"
shift
QUERIES=("$@")

PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"

if [ ! -d "$PARQUET_DIR" ]; then
    echo "ERROR: Parquet directory not found: $PARQUET_DIR"
    exit 1
fi

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

# Initialize timing CSV if requested
if [ -n "${TIMING_CSV:-}" ]; then
    echo "query,seconds" > "$TIMING_CSV"
fi

echo "Running TPC-H DuckDB baseline queries against SF${SF} parquet data"
echo "=========================================="

for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    RESULT_FILE="$PROJECT_DIR/result_duckdb_sf${SF}_q${q}.txt"

    if [ ! -f "$QUERY_FILE" ]; then
        echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
        continue
    fi

    echo ""
    echo "========== Q${q} =========="

    # Extract SQL from inside gpu_execution('...')
    INNER_SQL=$(sed -n "s/call gpu_execution('//; s/');//; p" "$QUERY_FILE" | sed "s/''/'/g")

    TEMP_SQL=$(mktemp /tmp/tpch_duckdb_q${q}_XXXXXX.sql)
    printf '%s\n' "$VIEW_SQL" > "$TEMP_SQL"
    printf '%s\n' "$INNER_SQL" >> "$TEMP_SQL"

    START_TIME=$(date +%s.%N)
    "$DUCKDB" -f "$TEMP_SQL" 2>&1 | tee "$RESULT_FILE"
    END_TIME=$(date +%s.%N)

    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    echo "  Time: ${ELAPSED}s"

    if [ -n "${TIMING_CSV:-}" ]; then
        echo "${q},${ELAPSED}" >> "$TIMING_CSV"
    fi

    rm -f "$TEMP_SQL"
done

echo ""
echo "=========================================="
echo "All DuckDB baseline queries complete."
