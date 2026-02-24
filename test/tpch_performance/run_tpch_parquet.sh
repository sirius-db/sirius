#!/bin/bash
# Run TPC-H GPU queries against Parquet files
#
# Creates views from parquet files found in the dataset directory,
# then runs the standard GPU query files unchanged.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=/home/felipe/sirius/test/cpp/integration/integration.cfg
#   ./test/tpch_performance/run_tpch_parquet.sh <scale_factor> <query_numbers...>
#
# Example:
#   ./test/tpch_performance/run_tpch_parquet.sh 100 1 3 4 5 6 7 8 9 10 12 13 14 18 19

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/gpu"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <scale_factor> <query_numbers...>"
    echo "Example: $0 100 1 3 4 5 6 7 8 9 10 12 13 14 18 19"
    exit 1
fi

SF="$1"
shift
QUERIES=("$@")

PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"

if [ ! -d "$PARQUET_DIR" ]; then
    echo "ERROR: Parquet directory not found: $PARQUET_DIR"
    echo "Generate it first with:"
    echo "  ./build/release/duckdb -c \"INSTALL tpch; LOAD tpch; CALL dbgen(sf=${SF}); EXPORT DATABASE '${PARQUET_DIR}' (FORMAT PARQUET);\""
    exit 1
fi

# Build CREATE VIEW statements for the TPC-H tables.
# Match both single files (table.parquet) and partitioned files (table_0.parquet, table_1.parquet, ...).
# A plain glob like part*.parquet would also match partsupp.parquet, so we collect
# matching files with bash globs and pass an explicit list to read_parquet().
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

echo "Running TPC-H queries against SF${SF} parquet data"
echo "Parquet dir: $PARQUET_DIR"
echo "Views: ${VIEW_SQL}"
echo "Queries: ${QUERIES[*]}"
echo "=========================================="

for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    RESULT_FILE="$PROJECT_DIR/result_sirius_sf${SF}_q${q}.txt"

    if [ ! -f "$QUERY_FILE" ]; then
        echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
        continue
    fi

    echo ""
    echo "========== Q${q} =========="

    TEMP_SQL=$(mktemp /tmp/tpch_q${q}_XXXXXX.sql)
    printf '%s\n' "$VIEW_SQL" > "$TEMP_SQL"
    cat "$QUERY_FILE" >> "$TEMP_SQL"

    "$DUCKDB" -f "$TEMP_SQL" 2>&1 | tee "$RESULT_FILE"

    rm -f "$TEMP_SQL"
done

echo ""
echo "=========================================="
echo "All queries complete. Results saved as result_sirius_sf${SF}_q*.txt"
