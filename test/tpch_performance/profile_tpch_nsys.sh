#!/usr/bin/env bash
# Profile TPC-H GPU queries with NVIDIA Nsight Systems (nsys)
#
# Runs each query in its own DuckDB process wrapped by nsys, producing
# per-query .nsys-rep and .sqlite files for analysis. Each query is
# executed twice (cold + hot) within the same process so both runs
# share a single nsys capture.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=/path/to/config.cfg
#   ./test/tpch_performance/profile_tpch_nsys.sh <scale_factor> [query_numbers...]
#
# If no query numbers are given, queries 1-22 are attempted.
#
# Examples:
#   ./test/tpch_performance/profile_tpch_nsys.sh 300_rg2m
#   ./test/tpch_performance/profile_tpch_nsys.sh 300_rg2m 1 3 6 9
#   QUERY_TIMEOUT=120 ./test/tpch_performance/profile_tpch_nsys.sh 100
#
# Output:
#   nsys_profiles/sf<SF>/q<N>.nsys-rep   - Nsight Systems report per query
#   nsys_profiles/sf<SF>/q<N>.sqlite     - SQLite export for programmatic analysis
#   nsys_profiles/sf<SF>/q<N>_result.txt - Query output + nsys messages
#   nsys_profiles/sf<SF>/q<N>_timings.csv - Per-iteration wall-clock timings
#   nsys_profiles/sf<SF>/summary.txt     - Pass/fail summary across all queries
#
# After profiling, analyze with:
#   ./test/tpch_performance/nsys_analyze.sh nsys_profiles/sf<SF>/

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# All paths configurable via environment variables
DUCKDB="${DUCKDB:-$PROJECT_DIR/build/release/duckdb}"
ITERATIONS=${ITERATIONS:-2}
# Per-query timeout in seconds (covers both iterations + nsys overhead).
QUERY_TIMEOUT=${QUERY_TIMEOUT:-90}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scale_factor> [query_numbers...]"
    echo "  scale_factor: e.g. 300_rg2m, 100, 100_rg2m (used to derive default paths)"
    echo "  query_numbers: optional list (default: 1-22)"
    echo ""
    echo "Environment variables:"
    echo "  SIRIUS_CONFIG_FILE - path to Sirius config (required)"
    echo "  DUCKDB             - path to DuckDB binary (default: build/release/duckdb)"
    echo "  PARQUET_DIR        - path to parquet data directory (default: test_datasets/tpch_parquet_sf<SF>)"
    echo "  QUERY_DIR          - path to GPU query SQL files (default: test/tpch_performance/tpch_queries/gpu)"
    echo "  OUTPUT_DIR         - output directory for profiles (default: nsys_profiles/sf<SF>)"
    echo "  QUERY_TIMEOUT      - per-query timeout in seconds (default: 90)"
    echo "  ITERATIONS         - number of query iterations (default: 2 for cold+hot)"
    exit 1
fi

SF="$1"; shift
if [ $# -gt 0 ]; then
    QUERIES=("$@")
else
    QUERIES=($(seq 1 22))
fi

PARQUET_DIR="${PARQUET_DIR:-$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}}"
QUERY_DIR="${QUERY_DIR:-$SCRIPT_DIR/tpch_queries/gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/nsys_profiles/sf${SF}}"

if [ ! -f "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found: $DUCKDB"
    echo "  Build with: pixi run make -j12"
    exit 1
fi

if [ ! -d "$PARQUET_DIR" ]; then
    echo "ERROR: Parquet directory not found: $PARQUET_DIR"
    exit 1
fi

if ! command -v nsys &>/dev/null; then
    echo "ERROR: nsys not found in PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Build CREATE VIEW statements
TPCH_TABLES=(customer lineitem nation orders part partsupp region supplier)
VIEW_SQL=""
for TABLE_NAME in "${TPCH_TABLES[@]}"; do
    FILES=()
    for f in "$PARQUET_DIR/${TABLE_NAME}.parquet" \
             "$PARQUET_DIR/${TABLE_NAME}_"*.parquet \
             "$PARQUET_DIR/${TABLE_NAME}/"*.parquet; do
        [ -f "$f" ] && FILES+=("'$f'")
    done
    FILE_LIST=$(IFS=,; echo "${FILES[*]}")
    VIEW_SQL+="CREATE VIEW ${TABLE_NAME} AS SELECT * FROM read_parquet([${FILE_LIST}]);"$'\n'
done

echo "============================================"
echo "  Nsight Systems TPC-H Profiling"
echo "============================================"
echo "Scale factor : $SF"
echo "Parquet dir  : $PARQUET_DIR"
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
    # Build header with columns for each hot iteration
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

# Write a summary line with dashes for all timing columns
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

    # Build temp SQL: timing table + views + N iterations with timestamps
    TEMP_SQL=$(mktemp /tmp/tpch_nsys_q${q}_XXXXXX.sql)

    printf 'CREATE TEMP TABLE _timings (seq INTEGER, step VARCHAR, ts TIMESTAMP);\n' > "$TEMP_SQL"
    printf "INSERT INTO _timings VALUES (0, 'start', current_timestamp);\n" >> "$TEMP_SQL"

    printf '%s\n' "$VIEW_SQL" >> "$TEMP_SQL"
    printf "INSERT INTO _timings VALUES (1, 'views', current_timestamp);\n" >> "$TEMP_SQL"

    for ((i = 1; i <= ITERATIONS; i++)); do
        cat "$QUERY_FILE" >> "$TEMP_SQL"
        printf "\nINSERT INTO _timings VALUES (%d, 'iter_%d', current_timestamp);\n" \
            $((i + 1)) "$i" >> "$TEMP_SQL"
    done

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

    # Run under nsys with low-overhead settings:
    #   --trace=cuda,nvtx    : capture CUDA API + NVTX markers
    #   --sample=none        : no CPU sampling (less overhead)
    #   --cudabacktrace=none : no CUDA backtraces (less overhead)
    #   --export=sqlite      : also export .sqlite for easy analysis
    START_TIME=$(date +%s.%N)
    timeout "$QUERY_TIMEOUT" \
    nsys profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cudabacktrace=none \
        --output="$NSYS_OUTPUT" \
        --force-overwrite=true \
        --stats=false \
        --export=sqlite \
        "$DUCKDB" -f "$TEMP_SQL" \
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
        # CSV format: step,runtime_s with rows: views, iter_1, iter_2, ..., iter_N
        COLD_TIME="-"
        BEST_HOT="-"
        HOT_TIMES=()
        if [ -f "$TIMING_FILE" ]; then
            # Row 3 = iter_1 (cold), rows 4+ = iter_2..N (hot)
            COLD_TIME=$(awk -F, 'NR==3 {printf "%.2f", $2}' "$TIMING_FILE")
            for ((hi = 2; hi <= ITERATIONS; hi++)); do
                ROW=$((hi + 2))  # CSV row: header=1, views=2, iter_1=3, iter_2=4, ...
                HT=$(awk -F, -v r="$ROW" 'NR==r {printf "%.2f", $2}' "$TIMING_FILE")
                HOT_TIMES+=("${HT:-"-"}")
            done
            # Best hot = minimum across all hot iterations
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

# Print summary
echo "============================================"
echo "  Summary: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "============================================"
cat "$SUMMARY_FILE"
echo ""
echo "Profiles saved to: $OUTPUT_DIR/"
echo "Analyze with:      ./test/tpch_performance/nsys_analyze.sh $OUTPUT_DIR/"
echo "Open in nsys-ui:   nsys-ui $OUTPUT_DIR/q<N>.nsys-rep"
echo "Query SQLite:      sqlite3 $OUTPUT_DIR/q<N>.sqlite"
