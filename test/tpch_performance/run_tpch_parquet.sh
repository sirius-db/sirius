#!/bin/bash
# Run TPC-H GPU queries against Parquet files
#
# Replaces table names inside gpu_execution() with read_parquet() calls.
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

echo "Running TPC-H queries against SF${SF} parquet data"
echo "Parquet dir: $PARQUET_DIR"
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

    # Use python to replace table names with read_parquet() calls
    TEMP_SQL=$(mktemp /tmp/tpch_q${q}_XXXXXX.sql)
    python3 -c "
import re, sys

sql = open('$QUERY_FILE').read()
parquet_dir = '$PARQUET_DIR'

# Replace table names with read_parquet() - longest names first to avoid partial matches
tables = ['partsupp', 'lineitem', 'customer', 'supplier', 'orders', 'nation', 'region', 'part']

for table in tables:
    # Word-boundary replacement, but skip 'as orders' pattern (Q13 derived table alias)
    sql = re.sub(r'(?<!as )(?<!\w)' + table + r'(?!\w)', f'read_parquet(\"{parquet_dir}/{table}.parquet\")', sql)

sys.stdout.write(sql)
" > "$TEMP_SQL"

    "$DUCKDB" -f "$TEMP_SQL" 2>&1 | tee "$RESULT_FILE"

    rm -f "$TEMP_SQL"
done

echo ""
echo "=========================================="
echo "All queries complete. Results saved as result_sirius_sf${SF}_q*.txt"
