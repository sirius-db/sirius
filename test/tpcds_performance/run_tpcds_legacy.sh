#!/usr/bin/env bash
# =============================================================================
# Run TPC-DS benchmark on legacy Sirius (gpu_processing).
#
# Requires data to be pre-generated via generate_tpcds_data.sh.
# Each query runs in its own DuckDB process. Each query is run twice
# (cold + warm).
#
# Usage:
#   ./run_tpcds_legacy.sh <gpu_caching_size> <gpu_processing_size> [options]
#
# Examples:
#   ./run_tpcds_legacy.sh "1 GB" "2 GB"
#   ./run_tpcds_legacy.sh "1 GB" "2 GB" --sf 1 --queries 1 2 3
#   ./run_tpcds_legacy.sh "10 GB" "20 GB" --sf 10 --queries $(seq 1 99)
#   ./run_tpcds_legacy.sh "1 GB" "2 GB" --db /data/tpcds_sf1.duckdb
#
# Options:
#   --sf <N>              Scale factor, used to locate database (default: 1)
#   --db <path>           Override path to DuckDB database file
#   --queries <N...>      Specific query numbers to run (default: all 1-99)
#   --output-dir <path>   Directory for results
#   --pin-cache           Use pinned host memory for caching instead of GPU memory
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
EXTENSION="$PROJECT_DIR/build/release/extension/sirius/sirius.duckdb_extension"
QUERY_DIR="$SCRIPT_DIR/queries"

# --- Parse arguments ---
if [ $# -lt 2 ]; then
    echo "Usage: $0 <gpu_caching_size> <gpu_processing_size> [--sf N] [--db path] [--queries N...] [--output-dir path] [--pin-cache]"
    echo "Example: $0 '1 GB' '2 GB' --sf 1 --queries 1 2 3"
    echo "         $0 '1 GB' '2 GB' --pin-cache"
    exit 1
fi

GPU_CACHING_SIZE="$1"
shift
GPU_PROCESSING_SIZE="$1"
shift

SF=1
DB_PATH=""
OUTPUT_DIR=""
PIN_CACHE=false
QUERIES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --sf)
            SF="$2"
            shift 2
            ;;
        --db)
            DB_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pin-cache)
            PIN_CACHE=true
            shift
            ;;
        --queries)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                QUERIES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Legacy Sirius uses gpu_buffer_init + gpu_processing; config file interferes
unset SIRIUS_CONFIG_FILE

# Defaults
if [ -z "$DB_PATH" ]; then
    DB_PATH="$PROJECT_DIR/test_datasets/tpcds_sf${SF}.duckdb"
fi

if [ ${#QUERIES[@]} -eq 0 ]; then
    QUERIES=($(seq 1 99))
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_DIR/tpcds_results_sf${SF}"
fi
mkdir -p "$OUTPUT_DIR"

# --- Validate prerequisites ---
if [ ! -x "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found at $DUCKDB"
    echo "Build first: CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) make"
    exit 1
fi

if [ ! -f "$EXTENSION" ]; then
    echo "ERROR: Sirius extension not found at $EXTENSION"
    echo "Build first: CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) make"
    exit 1
fi

if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database file not found: $DB_PATH"
    echo "Generate data first: bash $SCRIPT_DIR/generate_tpcds_data.sh $SF"
    exit 1
fi

if [ ! -d "$QUERY_DIR" ]; then
    echo "ERROR: Query directory not found: $QUERY_DIR"
    echo "Generate queries first: bash $SCRIPT_DIR/generate_tpcds_data.sh $SF"
    exit 1
fi

# --- Initialize output ---
TIMING_FILE="$OUTPUT_DIR/timings.csv"
echo "query,run1_time,run1_status,run2_time,run2_status" > "$TIMING_FILE"

echo "=========================================="
echo "TPC-DS Benchmark — Legacy Sirius (gpu_processing)"
echo "=========================================="
echo "Scale Factor:     $SF"
echo "Database:         $DB_PATH"
echo "GPU Caching:      $GPU_CACHING_SIZE"
echo "GPU Processing:   $GPU_PROCESSING_SIZE"
echo "Cache Mode:       $([ "$PIN_CACHE" = true ] && echo "pinned host memory" || echo "GPU memory")"
echo "Queries:          ${QUERIES[*]}"
echo "Output:           $OUTPUT_DIR"
echo "=========================================="
echo ""

# parse_timer_output <output_text>
# Extracts real (wall clock) times from DuckDB .timer output.
parse_timer_output() {
    echo "$1" | grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+'
}

# --- Run each query ---
for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    if [ ! -f "$QUERY_FILE" ]; then
        echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
        echo "${q},-1,SKIP,-1,SKIP" >> "$TIMING_FILE"
        continue
    fi

    QUERY_SQL=$(cat "$QUERY_FILE")
    if [ -z "$QUERY_SQL" ]; then
        echo "WARNING: Query ${q} is empty, skipping"
        echo "${q},-1,EMPTY,-1,EMPTY" >> "$TIMING_FILE"
        continue
    fi

    echo "--- Query ${q} ---"

    # Strip trailing semicolons from the query
    CLEANED_SQL=$(echo "$QUERY_SQL" | sed 's/;[[:space:]]*$//')

    # Escape inner double quotes, then wrap with gpu_processing("...")
    ESCAPED_SQL=$(echo "$CLEANED_SQL" | sed 's/"/\\"/g')

    # Write SQL file using printf to avoid bash heredoc expansion issues
    TEMP_SQL="$OUTPUT_DIR/tmp_q${q}.sql"
    {
        if [ "$PIN_CACHE" = true ]; then
            printf "SET use_pin_memory_for_caching = true;\n"
        fi
        printf "CALL gpu_buffer_init('%s', '%s');\n" "$GPU_CACHING_SIZE" "$GPU_PROCESSING_SIZE"
        printf ".timer on\n"
        printf 'CALL gpu_processing("%s");\n' "$ESCAPED_SQL"
        printf 'CALL gpu_processing("%s");\n' "$ESCAPED_SQL"
    } > "$TEMP_SQL"

    Q_RESULT_FILE="$OUTPUT_DIR/result_q${q}.txt"
    Q_LOG="$OUTPUT_DIR/log_q${q}.txt"

    # Run in a fresh DuckDB process against the pre-generated database
    OUTPUT=$("$DUCKDB" "$DB_PATH" < "$TEMP_SQL" 2>&1) || true
    EXIT_CODE=${PIPESTATUS[0]:-$?}

    echo "$OUTPUT" > "$Q_LOG"
    rm -f "$TEMP_SQL"

    # Parse timer output — expect two "Run Time" lines
    readarray -t TIMES < <(parse_timer_output "$OUTPUT")

    RUN1_TIME="${TIMES[0]:--1}"
    RUN2_TIME="${TIMES[1]:--1}"

    # Check for errors in the output
    HAS_ERROR=$(echo "$OUTPUT" | grep -ci "error" || true)

    if [ "$HAS_ERROR" -gt 0 ] && [ "$RUN1_TIME" = "-1" ]; then
        ERROR_MSG=$(echo "$OUTPUT" | grep -i "error" | head -1)
        echo "  Run 1: FAILED — $ERROR_MSG"
        echo "  Run 2: FAILED"
        echo "${q},${RUN1_TIME},FAILED,${RUN2_TIME},FAILED" >> "$TIMING_FILE"
    else
        RUN1_STATUS="OK"
        RUN2_STATUS="OK"

        if [ "$RUN1_TIME" = "-1" ]; then
            RUN1_STATUS="NO_TIMER"
        fi
        if [ "$RUN2_TIME" = "-1" ]; then
            RUN2_STATUS="NO_TIMER"
        fi

        echo "  Run 1 (cold): ${RUN1_TIME}s  [${RUN1_STATUS}]"
        echo "  Run 2 (warm): ${RUN2_TIME}s  [${RUN2_STATUS}]"
        echo "${q},${RUN1_TIME},${RUN1_STATUS},${RUN2_TIME},${RUN2_STATUS}" >> "$TIMING_FILE"
    fi

    # Save query result (output minus timer lines)
    echo "$OUTPUT" | grep -v "Run Time (s):" > "$Q_RESULT_FILE"
done

# --- Summary ---
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
printf "%-8s %12s %12s %10s\n" "Query" "Cold (s)" "Warm (s)" "Status"
printf "%s\n" "----------------------------------------------"

PASSED=0
FAILED=0
while IFS=, read -r q t1 s1 t2 s2; do
    [ "$q" = "query" ] && continue  # skip header
    if [ "$s1" = "OK" ] && [ "$s2" = "OK" ]; then
        STATUS="OK"
        ((PASSED++))
    else
        STATUS="FAILED"
        ((FAILED++))
    fi
    printf "%-8s %12s %12s %10s\n" "$q" "$t1" "$t2" "$STATUS"
done < "$TIMING_FILE"

printf "%s\n" "----------------------------------------------"
echo "Passed: $PASSED / $((PASSED + FAILED))"
[ $FAILED -gt 0 ] && echo "Failed: $FAILED / $((PASSED + FAILED))"
echo ""
echo "Timings:  $TIMING_FILE"
echo "Results:  $OUTPUT_DIR/result_q*.txt"
echo "Logs:     $OUTPUT_DIR/log_q*.txt"
