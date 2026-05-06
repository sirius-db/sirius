#!/usr/bin/env bash
# Run TPC-H queries against Parquet files and emit Quent telemetry.
#
# All queries run in a single Sirius/DuckDB process so the run produces one
# engine/worker pair plus one query per iteration. Each iteration is tagged
# with a per-query label via `sirius_set_query_label` so events can be
# attributed back to a specific (query, iteration) pair.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./run_tpch_parquet_and_generate_telemetry.sh [options] <scale_factor> [query_numbers...]
#
# If no query numbers are given, all 22 TPC-H queries are run.
#
# Options:
#   --parquet-dir <path>   Directory containing TPC-H parquet files
#                          (default: $PROJECT_DIR/test_datasets/tpch_parquet_sf<SF>)
#   --iterations <N>       Iterations per query (default: 2)
#   --output-dir <path>    Where to write Quent ndjson (default: telemetry_data/)
#   --note <text>          Free-form annotation prefixed onto each query label
#
# Example:
#   ./run_tpch_parquet_and_generate_telemetry.sh 1
#   ./run_tpch_parquet_and_generate_telemetry.sh --note before_change 100 1 6 9

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/orig"

PARQUET_DIR=""
NUM_ITERATIONS=2
OUTPUT_DIR="telemetry_data"
RUN_NOTE=""
while [ $# -gt 0 ]; do
    case "${1:-}" in
        --parquet-dir)  PARQUET_DIR="$2";    shift 2 ;;
        --iterations)   NUM_ITERATIONS="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";     shift 2 ;;
        --note)         RUN_NOTE="$2";       shift 2 ;;
        *) break ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--parquet-dir <path>] [--iterations <N>] [--output-dir <path>] [--note <text>] <scale_factor> [query_numbers...]"
    exit 1
fi

SF="$1"; shift
QUERIES=("$@")
if [ ${#QUERIES[@]} -eq 0 ]; then
    QUERIES=($(seq 1 22))
fi

if [ -z "$PARQUET_DIR" ]; then
    PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"
fi

if [ ! -x "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found at $DUCKDB"
    exit 1
fi
if [ ! -d "$PARQUET_DIR" ]; then
    echo "ERROR: Parquet directory not found: $PARQUET_DIR"
    exit 1
fi

# Build CREATE VIEW statements for all TPC-H tables.
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

# Single quotes in the user-supplied note must be SQL-escaped (doubled).
RUN_NOTE_ESCAPED="${RUN_NOTE//\'/\'\'}"
LABEL_PREFIX="tpch_"
[ -n "$RUN_NOTE_ESCAPED" ] && LABEL_PREFIX="${RUN_NOTE_ESCAPED}_${LABEL_PREFIX}"

mkdir -p "$OUTPUT_DIR"

# Build a single SQL file: views, telemetry config, then per-iteration label
# CALLs followed by the query.
TEMP_SQL=$(mktemp /tmp/tpch_telemetry_XXXXXX.sql)
{
    printf '%s\n' "$VIEW_SQL"
    printf "SET enable_quent = true;\n"
    printf "SET quent_output_directory = '%s';\n" "$OUTPUT_DIR"
    # `.timer on` prints a `Run Time` line after every statement, including the
    # label CALLs interleaved with each query.
    printf ".timer on\n"
    for q in "${QUERIES[@]}"; do
        QUERY_FILE="$QUERY_DIR/q${q}.sql"
        if [ ! -f "$QUERY_FILE" ]; then
            echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}" >&2
            continue
        fi
        for ((iter = 1; iter <= NUM_ITERATIONS; iter++)); do
            printf ".print =============== Q%d iter %d ===============\n" "$q" "$iter"
            printf "CALL sirius_set_query_label('%sq%d_iter%d');\n" \
                "$LABEL_PREFIX" "$q" "$iter"
            cat "$QUERY_FILE"
            printf '\n'
        done
    done
} > "$TEMP_SQL"

echo "Running TPC-H SF${SF} queries with telemetry export"
echo "  Parquet dir:  $PARQUET_DIR"
echo "  Queries:      ${QUERIES[*]}"
echo "  Iterations:   $NUM_ITERATIONS"
echo "  Output dir:   $OUTPUT_DIR"
[ -n "$RUN_NOTE" ] && echo "  Note:         $RUN_NOTE"
echo "=========================================="

"$DUCKDB" -f "$TEMP_SQL"
EXIT=$?
rm -f "$TEMP_SQL"

echo ""
echo "Telemetry data written under $OUTPUT_DIR"
exit "$EXIT"
