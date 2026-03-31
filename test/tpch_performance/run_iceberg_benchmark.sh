#!/usr/bin/env bash
# Compare TPC-H query performance: parquet vs iceberg vs iceberg + equality deletes
#
# Each query runs in its own DuckDB process (cold-run only) to measure the
# full startup + scan cost of each data source.  This is intentionally
# different from run_tpch_parquet.sh (single session, cold + warm) because
# the goal is to isolate the Iceberg metadata + delete overhead.
#
# Output: per-query result and timing files.
# When OUTPUT_DIR is set, results go to:
#   $OUTPUT_DIR/<source>/q<N>/{result.txt, timing.txt}
# Otherwise printed to stdout.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./test/tpch_performance/run_iceberg_benchmark.sh [options] <scale_factor> [query_numbers...]
#
# Options:
#   --parquet-dir <path>    Source parquet directory (default: test_datasets/tpch_parquet_sf<SF>)
#   --delete-pct <N>        Percentage of rows to equality-delete (default: 5)
#   --timeout <seconds>     Per-query timeout (default: 120)
#
# Examples:
#   ./test/tpch_performance/run_iceberg_benchmark.sh 10
#   ./test/tpch_performance/run_iceberg_benchmark.sh 10 1 3 6
#   ./test/tpch_performance/run_iceberg_benchmark.sh --parquet-dir /data/tpch 10 $(seq 1 22)
#
# Environment variables:
#   SIRIUS_CONFIG_FILE - path to Sirius config (required)
#   OUTPUT_DIR         - directory to save per-query results (optional)
#   TIMING_CSV         - path to write combined timing CSV (optional)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
QUERY_DIR="$SCRIPT_DIR/tpch_queries/orig"

# Extension is statically linked; just need CUDA libs on LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="$PROJECT_DIR/.pixi/envs/cuda12/lib:${LD_LIBRARY_PATH:-}"

PARQUET_DIR=""
DELETE_PCT=5
QUERY_TIMEOUT=120
while [ "${1:-}" = "--parquet-dir" ] || [ "${1:-}" = "--delete-pct" ] || [ "${1:-}" = "--timeout" ]; do
    if [ "$1" = "--parquet-dir" ]; then
        PARQUET_DIR="$2"
        shift 2
    elif [ "$1" = "--delete-pct" ]; then
        DELETE_PCT="$2"
        shift 2
    elif [ "$1" = "--timeout" ]; then
        QUERY_TIMEOUT="$2"
        shift 2
    fi
done

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--parquet-dir <path>] [--delete-pct <N>] [--timeout <seconds>] <scale_factor> [query_numbers...]"
    echo "Example: $0 10 \$(seq 1 22)"
    exit 1
fi

SF="$1"
shift
if [ $# -gt 0 ]; then
    QUERIES=("$@")
else
    QUERIES=($(seq 1 22))
fi

if [ -z "$PARQUET_DIR" ]; then
    PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"
fi
ICEBERG_DIR="$PROJECT_DIR/test_datasets/tpch_iceberg_sf${SF}"
ICEBERG_EQ_DIR="$PROJECT_DIR/test_datasets/tpch_iceberg_sf${SF}_eq${DELETE_PCT}"

TABLES=(nation region part supplier partsupp customer orders lineitem)

if [ ! -f "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found at $DUCKDB"
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-generate missing datasets
# ---------------------------------------------------------------------------
if [ ! -d "$PARQUET_DIR" ]; then
    echo "Parquet directory not found: $PARQUET_DIR"
    echo "Generating TPC-H SF${SF} dataset using tpchgen-rs..."
    (cd "$SCRIPT_DIR" && pixi run bash generate_tpch_data.sh "$SF" "$PARQUET_DIR")
fi

if [ ! -d "$ICEBERG_DIR" ]; then
    echo "Iceberg directory not found: $ICEBERG_DIR"
    echo "Wrapping parquet as Iceberg V2 (no deletes)..."
    python3 "$SCRIPT_DIR/wrap_parquet_as_iceberg.py" "$PARQUET_DIR" "$ICEBERG_DIR"
fi

if [ ! -d "$ICEBERG_EQ_DIR" ]; then
    echo "Iceberg+deletes directory not found: $ICEBERG_EQ_DIR"
    echo "Wrapping parquet as Iceberg V2 (${DELETE_PCT}% equality deletes on lineitem)..."
    python3 "$SCRIPT_DIR/wrap_parquet_as_iceberg.py" "$PARQUET_DIR" "$ICEBERG_EQ_DIR" \
        --delete-pct "$DELETE_PCT" --delete-tables lineitem
fi

# ---------------------------------------------------------------------------
# Build setup SQL for each data source
# ---------------------------------------------------------------------------
build_parquet_setup() {
    local sql="LOAD iceberg;"
    for t in "${TABLES[@]}"; do
        local files=""
        for f in "$PARQUET_DIR/${t}.parquet" "$PARQUET_DIR/${t}/"*.parquet "$PARQUET_DIR/${t}_"*.parquet; do
            [ -f "$f" ] && files="${files:+${files},}'${f}'"
        done
        [ -n "$files" ] && sql+=$'\n'"CREATE VIEW ${t} AS SELECT * FROM read_parquet([${files}]);"
    done
    echo "$sql"
}

build_iceberg_setup() {
    local dir="$1"
    local sql="LOAD iceberg;"
    for t in "${TABLES[@]}"; do
        local tpath="${dir}/${t}"
        [ -d "$tpath/metadata" ] && sql+=$'\n'"CREATE VIEW ${t} AS SELECT * FROM iceberg_scan('${tpath}');"
    done
    echo "$sql"
}

# ---------------------------------------------------------------------------
# Run queries for one data source
# ---------------------------------------------------------------------------
run_queries() {
    local label="$1"
    local source_key="$2"
    local setup_sql="$3"

    echo ""
    echo "=== $label ==="
    printf "%-6s %10s %s\n" "Query" "Time(s)" "Status"
    echo "-----------------------------"

    for q in "${QUERIES[@]}"; do
        local query_file="$QUERY_DIR/q${q}.sql"
        if [ ! -f "$query_file" ]; then
            printf "%-6s %10s %s\n" "Q${q}" "-" "SKIP (no file)"
            continue
        fi

        local query_sql
        query_sql=$(cat "$query_file")
        local escaped
        escaped=$(echo "$query_sql" | sed "s/'/''/g" | tr '\n' ' ')

        local tmp_sql
        tmp_sql=$(mktemp /tmp/iceberg_q${q}_XXXXXX.sql)
        cat > "$tmp_sql" <<EOSQL
${setup_sql}
CALL gpu_execution('${escaped}');
EOSQL

        local t_start t_end result exit_code timing
        t_start=$(date +%s.%N)
        result=$( { timeout "$QUERY_TIMEOUT" "$DUCKDB" -unsigned -noheader -f "$tmp_sql"; } 2>&1 )
        exit_code=$?
        t_end=$(date +%s.%N)
        rm -f "$tmp_sql"

        if [ $exit_code -ne 0 ]; then
            printf "%-6s %10s %s\n" "Q${q}" "-" "FAIL (exit=$exit_code)"
            if [ -n "${OUTPUT_DIR:-}" ]; then
                local q_dir="$OUTPUT_DIR/${source_key}/q${q}"
                mkdir -p "$q_dir"
                echo "FAIL (exit=$exit_code)" > "$q_dir/result.txt"
                echo "N/A" > "$q_dir/timing.txt"
            fi
            continue
        fi

        timing=$(echo "$t_end - $t_start" | bc)
        printf "%-6s %10s %s\n" "Q${q}" "${timing}" "OK"

        if [ -n "${OUTPUT_DIR:-}" ]; then
            local q_dir="$OUTPUT_DIR/${source_key}/q${q}"
            mkdir -p "$q_dir"
            echo "$result" > "$q_dir/result.txt"
            echo "$timing" > "$q_dir/timing.txt"
        fi

        if [ -n "${TIMING_CSV:-}" ]; then
            echo "${source_key},${q},${timing}" >> "$TIMING_CSV"
        fi
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "TPC-H SF=${SF} Iceberg Benchmark (cold runs)"
echo "=============================================="
echo "Parquet:       $PARQUET_DIR"
echo "Iceberg:       $ICEBERG_DIR"
echo "Iceberg+EqDel: $ICEBERG_EQ_DIR"
echo "Queries:       ${QUERIES[*]}"
echo "Timeout:       ${QUERY_TIMEOUT}s per query"
echo "=============================================="

if [ -n "${TIMING_CSV:-}" ]; then
    echo "source,query,seconds" > "$TIMING_CSV"
fi

run_queries "PARQUET (baseline)" "parquet" "$(build_parquet_setup)"
run_queries "ICEBERG (no deletes)" "iceberg" "$(build_iceberg_setup "$ICEBERG_DIR")"
run_queries "ICEBERG (${DELETE_PCT}% equality deletes on lineitem)" "iceberg_eq${DELETE_PCT}" "$(build_iceberg_setup "$ICEBERG_EQ_DIR")"

echo ""
echo "=============================================="
echo "All queries complete."
if [ -n "${OUTPUT_DIR:-}" ]; then
    echo "Results saved under $OUTPUT_DIR"
fi
if [ -n "${TIMING_CSV:-}" ]; then
    echo "Timings written to $TIMING_CSV"
fi
