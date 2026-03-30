#!/usr/bin/env bash
# Compare TPC-H query performance: parquet vs iceberg (no deletes) vs iceberg (with equality deletes)
#
# Runs all 22 TPC-H queries through gpu_execution for each data source.
# Reports cold-run wall time for each query (each query in its own DuckDB process).
#
# Usage (from sirius-dev root):
#   export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg
#   bash test/tpch_performance/run_iceberg_benchmark.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"

PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf10"
ICEBERG_DIR="$PROJECT_DIR/test_datasets/tpch_iceberg_sf10"
ICEBERG_EQ_DIR="$PROJECT_DIR/test_datasets/tpch_iceberg_sf10_eq5"
QUERY_DIR="$SCRIPT_DIR/tpch_queries/orig"

# Extension is statically linked; just need CUDA libs on LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="$PROJECT_DIR/.pixi/envs/cuda12/lib:${LD_LIBRARY_PATH:-}"

TABLES=(nation region part supplier partsupp customer orders lineitem)

if [ ! -f "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found at $DUCKDB"
    exit 1
fi

run_queries() {
    local label="$1"
    local setup_sql="$2"
    shift 2

    echo ""
    echo "=== $label ==="
    printf "%-6s %10s %s\n" "Query" "Time(s)" "Status"
    echo "-----------------------------"

    for q in $(seq 1 22); do
        query_file="$QUERY_DIR/q${q}.sql"
        if [ ! -f "$query_file" ]; then
            printf "%-6s %10s %s\n" "Q${q}" "-" "SKIP (no file)"
            continue
        fi

        query_sql=$(cat "$query_file")
        # Escape single quotes for gpu_execution wrapper
        escaped=$(echo "$query_sql" | sed "s/'/''/g" | tr '\n' ' ')

        # Write SQL to a temp file — DuckDB CLI with -f handles dot-commands
        # but we avoid dot-commands entirely; use external timing instead.
        tmp_sql=$(mktemp /tmp/iceberg_q${q}_XXXXXX.sql)
        cat > "$tmp_sql" <<EOSQL
${setup_sql}
CALL gpu_execution('${escaped}');
EOSQL

        # Run with timeout, measure wall time externally
        t_start=$(date +%s.%N)
        result=$( { timeout 120 "$DUCKDB" -unsigned -noheader -f "$tmp_sql"; } 2>&1 )
        exit_code=$?
        t_end=$(date +%s.%N)
        rm -f "$tmp_sql"

        if [ $exit_code -ne 0 ]; then
            printf "%-6s %10s %s\n" "Q${q}" "-" "FAIL (exit=$exit_code)"
            continue
        fi

        timing=$(echo "$t_end - $t_start" | bc)
        status="OK"

        printf "%-6s %10s %s\n" "Q${q}" "${timing}" "$status"
    done
}

# Setup SQL for parquet path — no LOAD needed, extension is statically linked
PARQUET_SETUP="LOAD iceberg;"
for t in "${TABLES[@]}"; do
    files=""
    for f in "$PARQUET_DIR/${t}.parquet" "$PARQUET_DIR/${t}/"*.parquet "$PARQUET_DIR/${t}_"*.parquet; do
        [ -f "$f" ] && files="${files:+${files},}'${f}'"
    done
    [ -n "$files" ] && PARQUET_SETUP+="
CREATE VIEW ${t} AS SELECT * FROM read_parquet([${files}]);"
done

# Setup SQL for iceberg (no deletes)
ICEBERG_SETUP="LOAD iceberg;"
for t in "${TABLES[@]}"; do
    tpath="$ICEBERG_DIR/$t"
    [ -d "$tpath/metadata" ] && ICEBERG_SETUP+="
CREATE VIEW ${t} AS SELECT * FROM iceberg_scan('${tpath}');"
done

# Setup SQL for iceberg with equality deletes
ICEBERG_EQ_SETUP="LOAD iceberg;"
for t in "${TABLES[@]}"; do
    tpath="$ICEBERG_EQ_DIR/$t"
    [ -d "$tpath/metadata" ] && ICEBERG_EQ_SETUP+="
CREATE VIEW ${t} AS SELECT * FROM iceberg_scan('${tpath}');"
done

echo "TPC-H SF=10 Iceberg Benchmark"
echo "=============================="
echo "Parquet:       $PARQUET_DIR"
echo "Iceberg:       $ICEBERG_DIR"
echo "Iceberg+EqDel: $ICEBERG_EQ_DIR"

run_queries "PARQUET (baseline)" "$PARQUET_SETUP"
run_queries "ICEBERG (no deletes)" "$ICEBERG_SETUP"
run_queries "ICEBERG (5% equality deletes on lineitem)" "$ICEBERG_EQ_SETUP"

echo ""
echo "Done."
