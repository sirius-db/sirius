#!/usr/bin/env bash
# =============================================================================
# Run TPC-DS benchmark on plain DuckDB (CPU baseline, no GPU).
#
# Runs raw SQL queries directly on DuckDB without any Sirius GPU wrapping.
# Supports both parquet-based and DuckDB database-based data sources.
# Each query runs twice (cold + warm).
#
# Usage:
#   ./run_tpcds_duckdb.sh --parquet-dir <path> [options]
#   ./run_tpcds_duckdb.sh --db <path> [options]
#
# Examples:
#   ./run_tpcds_duckdb.sh --parquet-dir /data/tpcds_parquet_sf1
#   ./run_tpcds_duckdb.sh --db /data/tpcds_sf1.duckdb --queries 3 7 17 25
#   ./run_tpcds_duckdb.sh --parquet-dir /data/tpcds_parquet_sf10 --output-dir /results/duckdb_sf10
#
# Options:
#   --parquet-dir <path>  Directory containing TPC-DS parquet files
#   --db <path>           Path to DuckDB database file
#   --queries <N...>      Specific query numbers to run (default: all 1-99)
#   --output-dir <path>   Directory for results (default: tpcds_duckdb_results/)
# =============================================================================

set -uo pipefail

# Prevent Sirius config from interfering — this is a pure CPU baseline
unset SIRIUS_CONFIG_FILE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
QUERY_DIR="$SCRIPT_DIR/queries"

# --- Parse arguments ---
PARQUET_DIR=""
DB_PATH=""
OUTPUT_DIR=""
QUERIES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --parquet-dir)
            PARQUET_DIR="$2"
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
        --queries)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                QUERIES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --parquet-dir <path> | --db <path> [--queries N...] [--output-dir path]"
            exit 1
            ;;
    esac
done

# Validate: exactly one data source required
if [ -z "$PARQUET_DIR" ] && [ -z "$DB_PATH" ]; then
    echo "ERROR: Must specify either --parquet-dir or --db"
    echo "Usage: $0 --parquet-dir <path> | --db <path> [--queries N...] [--output-dir path]"
    exit 1
fi

if [ -n "$PARQUET_DIR" ] && [ -n "$DB_PATH" ]; then
    echo "ERROR: Specify only one of --parquet-dir or --db, not both"
    exit 1
fi

# Defaults
if [ ${#QUERIES[@]} -eq 0 ]; then
    QUERIES=($(seq 1 99))
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_DIR/tpcds_duckdb_results"
fi
mkdir -p "$OUTPUT_DIR"

# --- Validate prerequisites ---
if [ ! -x "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found at $DUCKDB"
    echo "Build first: CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) make"
    exit 1
fi

if [ -n "$PARQUET_DIR" ] && [ ! -d "$PARQUET_DIR" ]; then
    echo "ERROR: Parquet directory not found: $PARQUET_DIR"
    echo "Generate data first: bash $SCRIPT_DIR/generate_tpcds_data.sh <SF> --format parquet"
    exit 1
fi

if [ -n "$DB_PATH" ] && [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database file not found: $DB_PATH"
    echo "Generate data first: bash $SCRIPT_DIR/generate_tpcds_data.sh <SF>"
    exit 1
fi

if [ ! -d "$QUERY_DIR" ]; then
    echo "ERROR: Query directory not found: $QUERY_DIR"
    echo "Generate queries first: bash $SCRIPT_DIR/generate_tpcds_data.sh <SF>"
    exit 1
fi

# --- Build CREATE VIEW statements (parquet mode only) ---
VIEW_SQL=""
if [ -n "$PARQUET_DIR" ]; then
    TPCDS_TABLES=(
        call_center catalog_page catalog_returns catalog_sales
        customer customer_address customer_demographics date_dim
        household_demographics income_band inventory item
        promotion reason ship_mode store
        store_returns store_sales time_dim warehouse
        web_page web_returns web_sales web_site
    )

    for TABLE_NAME in "${TPCDS_TABLES[@]}"; do
        FILES=()
        for f in "$PARQUET_DIR/${TABLE_NAME}.parquet" \
                 "$PARQUET_DIR/${TABLE_NAME}_"*.parquet \
                 "$PARQUET_DIR/${TABLE_NAME}/"*.parquet; do
            [ -f "$f" ] && FILES+=("'$f'")
        done
        if [ ${#FILES[@]} -eq 0 ]; then
            echo "WARNING: No parquet files found for table $TABLE_NAME"
            continue
        fi
        FILE_LIST=$(IFS=,; echo "${FILES[*]}")
        VIEW_SQL+="CREATE VIEW ${TABLE_NAME} AS SELECT * FROM read_parquet([${FILE_LIST}]);"$'\n'
    done
fi

# --- Initialize output ---
TIMING_FILE="$OUTPUT_DIR/timings.csv"
echo "query,run1_time,run1_status,run2_time,run2_status" > "$TIMING_FILE"

DATA_SOURCE=""
if [ -n "$PARQUET_DIR" ]; then
    DATA_SOURCE="$PARQUET_DIR (parquet)"
else
    DATA_SOURCE="$DB_PATH (duckdb)"
fi

echo "=========================================="
echo "TPC-DS Benchmark — DuckDB CPU Baseline"
echo "=========================================="
echo "Data Source:       $DATA_SOURCE"
echo "Queries:          ${QUERIES[*]}"
echo "Output:           $OUTPUT_DIR"
echo "=========================================="
echo ""

# parse_timer_output <output_text>
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

    # Write SQL file: optional views, timer, then run raw SQL twice
    TEMP_SQL="$OUTPUT_DIR/tmp_q${q}.sql"
    {
        if [ -n "$VIEW_SQL" ]; then
            printf '%s\n' "$VIEW_SQL"
        fi
        printf ".timer on\n"
        printf '%s\n' "$QUERY_SQL"
        printf '%s\n' "$QUERY_SQL"
    } > "$TEMP_SQL"

    Q_RESULT_FILE="$OUTPUT_DIR/result_q${q}.txt"
    Q_LOG="$OUTPUT_DIR/log_q${q}.txt"

    # Run in a fresh DuckDB process
    if [ -n "$DB_PATH" ]; then
        OUTPUT=$("$DUCKDB" "$DB_PATH" < "$TEMP_SQL" 2>&1) || true
    else
        OUTPUT=$("$DUCKDB" < "$TEMP_SQL" 2>&1) || true
    fi

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
