#!/usr/bin/env bash
# Profile TPC-H GPU queries with NVIDIA Nsight Systems (nsys) against DuckDB native tables
#
# Similar to profile_tpch_nsys.sh but scans from a DuckDB database file instead of
# parquet files. This exercises the DuckDB table scan path (duckdb_scan_task) rather
# than the parquet scan path (parquet_scan_task).
#
# Runs each query in its own DuckDB process wrapped by nsys, producing
# per-query .nsys-rep and .sqlite files for analysis. Each query is
# executed multiple times (cold + hot) within the same process so both runs
# share a single nsys capture.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=/path/to/config.cfg
#   ./test/tpch_performance/profile_tpch_nsys_duckdb_native.sh <db_path> [OPTIONS] [query_numbers...]
#
# Examples:
#   ./test/tpch_performance/profile_tpch_nsys_duckdb_native.sh tpch_10.duckdb
#   ./test/tpch_performance/profile_tpch_nsys_duckdb_native.sh tpch_10.duckdb 1 6 9
#   ITERATIONS=4 ./test/tpch_performance/profile_tpch_nsys_duckdb_native.sh tpch_10.duckdb 1 3
#
# Output:
#   nsys_profiles/duckdb_native_<dbname>/q<N>.nsys-rep   - Nsight Systems report per query
#   nsys_profiles/duckdb_native_<dbname>/q<N>.sqlite     - SQLite export for analysis
#   nsys_profiles/duckdb_native_<dbname>/q<N>_result.txt - Query output + nsys messages
#   nsys_profiles/duckdb_native_<dbname>/q<N>_timings.csv - Per-iteration wall-clock timings
#   nsys_profiles/duckdb_native_<dbname>/summary.txt     - Pass/fail summary
#
# After profiling, analyze with:
#   ./test/tpch_performance/nsys_analyze.sh nsys_profiles/duckdb_native_<dbname>/

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DUCKDB="${DUCKDB:-$PROJECT_DIR/build/release/duckdb}"
ITERATIONS=${ITERATIONS:-2}
QUERY_TIMEOUT=${QUERY_TIMEOUT:-120}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <db_path> [query_numbers...]"
    echo "  db_path: path to DuckDB database file (e.g., tpch_10.duckdb)"
    echo "  query_numbers: optional list (default: 1-22)"
    echo ""
    echo "Environment variables:"
    echo "  SIRIUS_CONFIG_FILE - path to Sirius config (required)"
    echo "  DUCKDB             - path to DuckDB binary (default: build/release/duckdb)"
    echo "  QUERY_DIR          - path to GPU query SQL files (default: test/tpch_performance/tpch_queries/gpu)"
    echo "  OUTPUT_DIR         - output directory for profiles (default: nsys_profiles/duckdb_native_<dbname>)"
    echo "  QUERY_TIMEOUT      - per-query timeout in seconds (default: 120)"
    echo "  ITERATIONS         - number of query iterations (default: 2 for cold+hot)"
    exit 1
fi

DB_PATH="$1"; shift
if [ $# -gt 0 ]; then
    QUERIES=("$@")
else
    QUERIES=($(seq 1 22))
fi

if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database file not found: $DB_PATH"
    exit 1
fi

DB_NAME=$(basename "$DB_PATH" .duckdb)
QUERY_DIR="${QUERY_DIR:-$SCRIPT_DIR/tpch_queries/gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/nsys_profiles/duckdb_native_${DB_NAME}}"

if [ ! -f "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found: $DUCKDB"
    echo "  Build with: pixi run make -j12"
    exit 1
fi

if ! command -v nsys &>/dev/null; then
    echo "ERROR: nsys not found in PATH"
    echo "  Install NVIDIA Nsight Systems or add to PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Nsight Systems TPC-H Profiling"
echo "  (DuckDB Native Table Scan)"
echo "============================================"
echo "Database     : $DB_PATH"
echo "DB size      : $(du -h "$DB_PATH" | cut -f1)"
echo "Iterations   : $ITERATIONS (1 cold + $((ITERATIONS - 1)) hot)"
echo "Timeout      : ${QUERY_TIMEOUT}s per query"
echo "Queries      : ${QUERIES[*]}"
echo "Output dir   : $OUTPUT_DIR"
echo "Config       : ${SIRIUS_CONFIG_FILE:-<not set>}"
echo "nsys version : $(nsys --version 2>&1 | head -1)"
echo "============================================"
echo ""

SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
if [ "$ITERATIONS" -le 2 ]; then
    printf "%-6s  %-10s  %-10s  %-8s\n" "Query" "Cold(s)" "Hot(s)" "Status" > "$SUMMARY_FILE"
    printf "%-6s  %-10s  %-10s  %-8s\n" "-----" "--------" "--------" "------" >> "$SUMMARY_FILE"
else
    HDR=$(printf "%-6s  %-10s" "Query" "Cold(s)")
    SEP=$(printf "%-6s  %-10s" "-----" "--------")
    for ((hi = 2; hi <= ITERATIONS; hi++)); do
        HDR+=$(printf "  %-10s" "Hot${hi}(s)")
        SEP+=$(printf "  %-10s" "--------")
    done
    HDR+=$(printf "  %-10s  %-8s" "Best(s)" "Status")
    SEP+=$(printf "  %-10s  %-8s" "--------" "------")
    echo "$HDR" > "$SUMMARY_FILE"
    echo "$SEP" >> "$SUMMARY_FILE"
fi

PASSED=0
FAILED=0
SKIPPED=0

write_summary_line() {
    local query="$1" status="$2"
    if [ "$ITERATIONS" -le 2 ]; then
        printf "%-6s  %-10s  %-10s  %-8s\n" "$query" "-" "-" "$status" >> "$SUMMARY_FILE"
    else
        local line
        line=$(printf "%-6s  %-10s" "$query" "-")
        for ((hi = 2; hi <= ITERATIONS; hi++)); do
            line+=$(printf "  %-10s" "-")
        done
        line+=$(printf "  %-10s  %-8s" "-" "$status")
        echo "$line" >> "$SUMMARY_FILE"
    fi
}

for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    NSYS_OUTPUT="$OUTPUT_DIR/q${q}"
    RESULT_FILE="$OUTPUT_DIR/q${q}_result.txt"
    TIMING_FILE="$OUTPUT_DIR/q${q}_timings.csv"

    if [ ! -f "$QUERY_FILE" ]; then
        echo "[Q${q}] SKIP - query file not found: $QUERY_FILE"
        write_summary_line "Q${q}" "SKIP"
        ((SKIPPED++))
        continue
    fi

    echo "---------- Q${q} ----------"

    # Build temp SQL: timing table + N iterations with timestamps
    # No views needed — tables are native in the DuckDB database
    TEMP_SQL=$(mktemp /tmp/tpch_nsys_native_q${q}_XXXXXX.sql)

    printf 'CREATE TEMP TABLE _timings (seq INTEGER, step VARCHAR, ts TIMESTAMP);\n' > "$TEMP_SQL"
    printf "INSERT INTO _timings VALUES (0, 'start', current_timestamp);\n" >> "$TEMP_SQL"
    printf "INSERT INTO _timings VALUES (1, 'views', current_timestamp);\n" >> "$TEMP_SQL"

    for ((i = 1; i <= ITERATIONS; i++)); do
        # Start nsys capture before hot iterations (skip cold run)
        if [ "$i" -eq 2 ]; then
            printf "CALL profiler_start();\n" >> "$TEMP_SQL"
        fi
        cat "$QUERY_FILE" >> "$TEMP_SQL"
        printf "\nINSERT INTO _timings VALUES (%d, 'iter_%d', current_timestamp);\n" \
            $((i + 1)) "$i" >> "$TEMP_SQL"
    done
    # Stop nsys capture after all iterations
    if [ "$ITERATIONS" -ge 2 ]; then
        printf "CALL profiler_stop();\n" >> "$TEMP_SQL"
    fi

    # Extract per-step timings
    cat >> "$TEMP_SQL" <<EOF
COPY (
    SELECT step, runtime_s FROM (
        SELECT
            seq,
            step,
            extract(epoch FROM (ts - LAG(ts) OVER (ORDER BY seq))) AS runtime_s
        FROM _timings
    )
    WHERE seq > 0
    ORDER BY seq
) TO '${TIMING_FILE}' (FORMAT CSV, HEADER);
EOF

    # Run under nsys with low-overhead settings.
    # Open database in readonly mode to avoid WAL contention.
    START_TIME=$(date +%s.%N)
    if [ "$ITERATIONS" -ge 2 ]; then
        CAPTURE_ARGS="--capture-range=cudaProfilerApi --capture-range-end=stop"
    else
        CAPTURE_ARGS=""
    fi
    timeout "$QUERY_TIMEOUT" \
    nsys profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cudabacktrace=none \
        $CAPTURE_ARGS \
        --output="$NSYS_OUTPUT" \
        --force-overwrite=true \
        --stats=false \
        --export=sqlite \
        "$DUCKDB" "$DB_PATH" -readonly -f "$TEMP_SQL" \
        > "$RESULT_FILE" 2>&1
    EXIT_CODE=$?
    END_TIME=$(date +%s.%N)

    WALL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    rm -f "$TEMP_SQL"

    if [ $EXIT_CODE -eq 124 ]; then
        echo "[Q${q}] TIMEOUT after ${QUERY_TIMEOUT}s"
        write_summary_line "Q${q}" "TIMEOUT"
        ((FAILED++))
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "[Q${q}] FAILED (exit code $EXIT_CODE) - wall time: ${WALL_TIME}s"
        echo "  See: $RESULT_FILE"
        tail -5 "$RESULT_FILE" 2>/dev/null | sed 's/^/  > /'
        write_summary_line "Q${q}" "FAIL"
        ((FAILED++))
    else
        # Parse iteration times from the timing CSV
        COLD_TIME="-"
        BEST_HOT="-"
        HOT_TIMES=()
        if [ -f "$TIMING_FILE" ]; then
            COLD_TIME=$(awk -F, 'NR==3 {printf "%.2f", $2}' "$TIMING_FILE")
            for ((hi = 2; hi <= ITERATIONS; hi++)); do
                ROW=$((hi + 2))
                HT=$(awk -F, -v r="$ROW" 'NR==r {printf "%.2f", $2}' "$TIMING_FILE")
                HOT_TIMES+=("${HT:-"-"}")
            done
            BEST_HOT=$(printf '%s\n' "${HOT_TIMES[@]}" | grep -v '^-$' | sort -n | head -1)
            BEST_HOT="${BEST_HOT:-"-"}"
        fi

        if [ "$ITERATIONS" -le 2 ]; then
            echo "[Q${q}] OK - cold: ${COLD_TIME}s, hot: ${HOT_TIMES[0]:-"-"}s, wall: ${WALL_TIME}s"
            echo "  Profile: ${NSYS_OUTPUT}.nsys-rep"
            printf "%-6s  %-10s  %-10s  %-8s\n" "Q${q}" "$COLD_TIME" "${HOT_TIMES[0]:-"-"}" "OK" >> "$SUMMARY_FILE"
        else
            TIMES_STR=$(IFS=', '; echo "${HOT_TIMES[*]}")
            echo "[Q${q}] OK - cold: ${COLD_TIME}s, hot: [${TIMES_STR}]s, best: ${BEST_HOT}s, wall: ${WALL_TIME}s"
            echo "  Profile: ${NSYS_OUTPUT}.nsys-rep"
            LINE=$(printf "%-6s  %-10s" "Q${q}" "$COLD_TIME")
            for ht in "${HOT_TIMES[@]}"; do
                LINE+=$(printf "  %-10s" "$ht")
            done
            LINE+=$(printf "  %-10s  %-8s" "$BEST_HOT" "OK")
            echo "$LINE" >> "$SUMMARY_FILE"
        fi
        ((PASSED++))
    fi
    echo ""
done

echo "============================================"
echo "  Summary: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "============================================"
cat "$SUMMARY_FILE"
echo ""
echo "Profiles saved to: $OUTPUT_DIR/"
echo "Analyze with:      ./test/tpch_performance/nsys_analyze.sh $OUTPUT_DIR/"
