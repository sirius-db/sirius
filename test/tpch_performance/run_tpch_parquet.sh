#!/bin/bash
# Run TPC-H GPU queries against Parquet files
#
# Creates views from parquet files found in the dataset directory,
# then runs the specified queries on the specified engine with the specified number of iterations.
# All iterations run in a single DuckDB session so that scan caches warm on the first run.
# Timings are recorded using current_timestamp between steps and written to a CSV file.
# Output results are saved as result_<engine>_sf<scale_factor>_q<query_number>.txt
# and timing results as timings_<engine>_sf<scale_factor>_q<query_number>.csv.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=/home/felipe/sirius/test/cpp/integration/integration.cfg
#   ./test/tpch_performance/run_tpch_parquet.sh <engine> <scale_factor> <iterations> <query_numbers...>
# with engine = [sirius/duckdb]
#
# All iterations run in a single DuckDB session so that scan caches warm on the first run.
# Timings are recorded using current_timestamp between steps and written to a CSV file.
#
# Example:
#   ./test/tpch_performance/run_tpch_parquet.sh sirius 100 3 `seq 1 22`
#
# Environment variables:
#   SIRIUS_CONFIG_FILE - path to Sirius config file (required)
#   TIMING_CSV         - path to write per-query timing CSV (optional)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"

if [ $# -lt 4 ]; then
    echo "Usage: $0 <engine> <scale_factor> <iterations> <query_numbers...>"
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

PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"

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
    echo "Generating TPC-H SF${SF} dataset with tpchgen-cli..."

    VENV_DIR="$PROJECT_DIR/.venv"
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet tpchgen-cli
    tpchgen-cli -s "$SF" --format=parquet --parts 1 --output-dir "$PARQUET_DIR"
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

for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    RESULT_FILE="$PROJECT_DIR/result_${ENGINE}_sf${SF}_q${q}.txt"
    TIMING_FILE="$PROJECT_DIR/timings_${ENGINE}_sf${SF}_q${q}.csv"

    if [ ! -f "$QUERY_FILE" ]; then
        echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
        continue
    fi

    echo ""
    echo "========== Q${q} =========="

    TEMP_SQL=$(mktemp /tmp/tpch_q${q}_XXXXXX.sql)

    # Timing table: one row per checkpoint, ordered by seq
    printf 'CREATE TEMP TABLE _timings (seq INTEGER, step VARCHAR, ts TIMESTAMP);\n' > "$TEMP_SQL"
    printf "INSERT INTO _timings VALUES (0, 'start', current_timestamp);\n" >> "$TEMP_SQL"

    # View creation
    printf '%s\n' "$VIEW_SQL" >> "$TEMP_SQL"
    printf "INSERT INTO _timings VALUES (1, 'views', current_timestamp);\n" >> "$TEMP_SQL"

    # Append the query N times, recording a timestamp after each run
    for ((i = 1; i <= ITERATIONS; i++)); do
        cat "$QUERY_FILE" >> "$TEMP_SQL"
        printf "\nINSERT INTO _timings VALUES (%d, 'iter_%d', current_timestamp);\n" \
            $((i + 1)) "$i" >> "$TEMP_SQL"
    done

    # Write per-step runtimes (seconds) to CSV using LAG over the checkpoints.
    # The 'start' row is excluded from output after the window function runs,
    # so that LAG for 'views' (seq=1) can still see seq=0 as its predecessor.
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

    START_TIME=$(date +%s.%N)
    "$DUCKDB" -f "$TEMP_SQL" 2>&1 | tee "$RESULT_FILE"
    END_TIME=$(date +%s.%N)

    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    echo "  Time: ${ELAPSED}s"

    if [ -n "${TIMING_CSV:-}" ]; then
        echo "${q},${ELAPSED}" >> "$TIMING_CSV"
    fi

    rm -f "$TEMP_SQL"
    echo "Timings written to $TIMING_FILE"
done

echo ""
echo "=========================================="
echo "All queries complete."
echo "Results saved as result_${ENGINE}_sf${SF}_q*.txt"
echo "Timings saved as timings_${ENGINE}_sf${SF}_q*.csv"
