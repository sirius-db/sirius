#!/usr/bin/env bash
# Run TPC-H GPU queries against Parquet files
#
# Creates views from parquet files found in the dataset directory,
# then runs the specified queries on the specified engine with the specified number of iterations.
# All iterations run in a single DuckDB session so that scan caches warm on the first run.
# Timings are measured using DuckDB's .timer command and parsed from its output.
# Output results are saved as result_<engine>_sf<scale_factor>_q<query_number>.txt
# and timing results as timings_<engine>_sf<scale_factor>_q<query_number>.csv.
# If OUTPUT_DIR is set (this is done by benchmark_and_validate.sh),
# results are saved in the specified directory (in a subdirectory per query),
# otherwise they are saved in the project directory.
# All iterations run in a single DuckDB session so that scan caches warm on the first run.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=/home/felipe/sirius/test/cpp/integration/integration.cfg
#   ./test/tpch_performance/run_tpch_parquet.sh [--parquet-dir <path>] <engine> <scale_factor> <iterations> <query_numbers...>
# with engine = [sirius/duckdb]
#
# Example:
#   ./test/tpch_performance/run_tpch_parquet.sh sirius 100 3 `seq 1 22`
#   ./test/tpch_performance/run_tpch_parquet.sh --parquet-dir /data/tpch sirius 100 3 `seq 1 22`
#
# Environment variables:
#   SIRIUS_CONFIG_FILE - path to Sirius config file (required)
#   TIMING_CSV         - path to write per-query timing CSV (optional)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"

PARQUET_DIR=""
if [ "${1:-}" = "--parquet-dir" ]; then
    PARQUET_DIR="$2"
    shift 2
fi

if [ $# -lt 4 ]; then
    echo "Usage: $0 [--parquet-dir <path>] <engine> <scale_factor> <iterations> <query_numbers...>"
    echo "Example: $0 sirius 100 3 \`seq 1 22\`"
    exit 1
fi

ENGINE="$1"
shift
SF="$1"
shift
ITERATIONS="$1"
shift
QUERIES=("$@")

if ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: iterations must be a positive integer, got: $ITERATIONS"
    exit 1
fi

if [ -z "$PARQUET_DIR" ]; then
    PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"
fi

if [ "$ENGINE" != "sirius" ] && [ "$ENGINE" != "duckdb" ]; then
    echo "Unknown engine, please use sirius or duckdb"
    exit 1
fi
if [ "$ENGINE" == "sirius" ]; then
    QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/gpu"
else
    QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/orig"
fi

if [ ! -d "$PARQUET_DIR" ]; then
    echo "Parquet directory not found: $PARQUET_DIR"
    echo "Generating TPC-H SF${SF} dataset using tpchgen-rs..."
    (cd "$SCRIPT_DIR" && pixi run bash generate_tpch_data.sh "$SF" "$PARQUET_DIR")
fi

# Build CREATE VIEW statements for the TPC-H tables.
# Match both single files (table.parquet) and partitioned files (table_0.parquet, table_1.parquet, ...).
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

# Initialize timing CSV if requested
if [ -n "${TIMING_CSV:-}" ]; then
    echo "query,seconds" > "$TIMING_CSV"
fi

echo "Running TPC-H queries against SF${SF} parquet data"
echo "Engine: $ENGINE"
echo "Parquet dir: $PARQUET_DIR"
echo "Iterations: $ITERATIONS"
echo "Views: ${VIEW_SQL}"
echo "Queries: ${QUERIES[*]}"
echo "=========================================="

# parse_timer_output <output_text>
# Extracts real (wall clock) times from DuckDB .timer on output.
# Format: "Run Time (s): real 5.419 user 20.134 sys 0.512"
parse_timer_output() {
    echo "$1" | grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+'
}

for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    if [ -n "${OUTPUT_DIR:-}" ]; then
        Q_DIR="$OUTPUT_DIR/q${q}"
        mkdir -p "$Q_DIR"
        RESULT_FILE="$Q_DIR/result.txt"
        TIMING_FILE="$Q_DIR/timings.csv"
    else
        RESULT_FILE="$PROJECT_DIR/result_${ENGINE}_sf${SF}_q${q}.txt"
        TIMING_FILE="$PROJECT_DIR/timings_${ENGINE}_sf${SF}_q${q}.csv"
    fi

    if [ ! -f "$QUERY_FILE" ]; then
        echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
        continue
    fi

    echo ""
    echo "========== Q${q} =========="

    if [ -n "${OUTPUT_DIR:-}" ]; then
        TEMP_SQL="$Q_DIR/query.sql"
    else
        TEMP_SQL=$(mktemp /tmp/tpch_q${q}_XXXXXX.sql)
    fi

    # Build the SQL file: views, then .timer on, then N iterations of the query.
    printf '%s\n' "$VIEW_SQL" > "$TEMP_SQL"
    echo ".timer on" >> "$TEMP_SQL"
    for ((i = 1; i <= ITERATIONS; i++)); do
        cat "$QUERY_FILE" >> "$TEMP_SQL"
        printf '\n' >> "$TEMP_SQL"
    done

    # Run DuckDB and capture full output
    local_output=""
    START_TIME=$(date +%s.%N)
    if [ -n "${OUTPUT_DIR:-}" ]; then
        local_output=$(SIRIUS_LOG_DIR="$Q_DIR" "$DUCKDB" -f "$TEMP_SQL" 2>&1)
    else
        local_output=$("$DUCKDB" -f "$TEMP_SQL" 2>&1)
    fi
    END_TIME=$(date +%s.%N)

    # Write result (query output) to file
    echo "$local_output" | tee "$RESULT_FILE"

    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    echo "  Time: ${ELAPSED}s"

    # Parse .timer output into per-iteration timings CSV
    readarray -t TIMES < <(parse_timer_output "$local_output")
    {
        echo "step,runtime_s"
        for ((i = 0; i < ${#TIMES[@]}; i++)); do
            echo "iter_$((i + 1)),${TIMES[$i]}"
        done
    } > "$TIMING_FILE"

    if [ -n "${TIMING_CSV:-}" ]; then
        echo "${q},${ELAPSED}" >> "$TIMING_CSV"
    fi

    if [ -z "${OUTPUT_DIR:-}" ]; then
        rm -f "$TEMP_SQL"
    fi
    echo "Timings written to $TIMING_FILE"
done

echo ""
echo "=========================================="
echo "All queries complete."
echo "Results saved as result_${ENGINE}_sf${SF}_q*.txt"
echo "Timings saved as timings_${ENGINE}_sf${SF}_q*.csv"
