#!/usr/bin/env bash
# Run TPC-H GPU queries against Parquet files
#
# All specified queries run in a single DuckDB session, wrapped with the
# `timeout` command so that the entire session is killed if it exceeds
# the timeout.  Each query is executed twice back-to-back (cold + warm)
# with nothing in between so that the Sirius scan cache remains valid
# for the warm run.
# Per-query results and timings are extracted from the combined output
# using delimiter markers (.print).
#
# Output: per-query result and timing files.
# When OUTPUT_DIR is set (by benchmark_and_validate.sh), results go to
#   $OUTPUT_DIR/q<N>/{result.txt, timings.csv, query.sql}
# Otherwise:
#   result_<engine>_sf<SF>_q<N>.txt  and  timings_<engine>_sf<SF>_q<N>.csv
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./test/tpch_performance/run_tpch_parquet.sh [--parquet-dir <path>] [--timeout <seconds>] <engine> <scale_factor> <query_numbers...>
# with engine = [sirius/duckdb]
#
# Example:
#   ./test/tpch_performance/run_tpch_parquet.sh sirius 100 `seq 1 22`
#   ./test/tpch_performance/run_tpch_parquet.sh --parquet-dir /data/tpch --timeout 1200 sirius 100 `seq 1 22`
#
# Environment variables:
#   SIRIUS_CONFIG_FILE - path to Sirius config file (required for sirius engine)
#   OUTPUT_DIR         - directory to save per-query results (optional)
#   TIMING_CSV         - path to write per-query timing CSV (optional)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIRIUS_DUCKDB="$PROJECT_DIR/build/release/duckdb"

PARQUET_DIR=""
NUM_ITERATIONS=2
SESSION_TIMEOUT=1200
while [ "${1:-}" = "--parquet-dir" ] || [ "${1:-}" = "--iterations" ] || [ "${1:-}" = "--timeout" ]; do
    if [ "$1" = "--parquet-dir" ]; then
        PARQUET_DIR="$2"
        shift 2
    elif [ "$1" = "--iterations" ]; then
        NUM_ITERATIONS="$2"
        shift 2
    elif [ "$1" = "--timeout" ]; then
        SESSION_TIMEOUT="$2"
        shift 2
    fi
done

if [ $# -lt 3 ]; then
    echo "Usage: $0 [--parquet-dir <path>] [--timeout <seconds>] <engine> <scale_factor> <query_numbers...>"
    echo "Example: $0 sirius 100 \`seq 1 22\`"
    echo "  --timeout N   Kill the entire DuckDB session after N seconds (default: 1200, 0 = no timeout)"
    exit 1
fi

ENGINE="$1"
shift
SF="$1"
shift
QUERIES=("$@")

if [ -z "$PARQUET_DIR" ]; then
    PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"
fi

if [ "$ENGINE" != "sirius" ] && [ "$ENGINE" != "duckdb" ]; then
    echo "Unknown engine, please use sirius or duckdb"
    exit 1
fi
if [ "$ENGINE" == "sirius" ]; then
    DUCKDB="$SIRIUS_DUCKDB"
    QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/gpu"
else
    # Use the same binary but without Sirius config so the extension doesn't initialize.
    DUCKDB="$SIRIUS_DUCKDB"
    unset SIRIUS_CONFIG_FILE 2>/dev/null || true
    export SIRIUS_CONFIG_FILE=
    QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/orig"
fi

if [ ! -d "$PARQUET_DIR" ]; then
    echo "Parquet directory not found: $PARQUET_DIR"
    echo "Generating TPC-H SF${SF} dataset using tpchgen-rs..."
    (cd "$SCRIPT_DIR" && pixi run bash generate_tpch_data.sh "$SF" "$PARQUET_DIR")
fi

# Build CREATE VIEW statements.
# Match single files (table.parquet), partitioned (table_0.parquet, ...),
# and subdirectory layouts (table/*.parquet).
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

if [ -n "${TIMING_CSV:-}" ]; then
    echo "query,seconds" > "$TIMING_CSV"
fi

echo "Running TPC-H queries against SF${SF} parquet data"
echo "Engine: $ENGINE"
echo "Parquet dir: $PARQUET_DIR"
echo "Iterations: $NUM_ITERATIONS (1 cold + $((NUM_ITERATIONS - 1)) warm)"
echo "Queries: ${QUERIES[*]}"
if [ "$SESSION_TIMEOUT" -gt 0 ] 2>/dev/null; then
    echo "Session timeout: ${SESSION_TIMEOUT}s"
else
    echo "Session timeout: disabled"
fi
echo "=========================================="

# ---------------------------------------------------------------------------
# Build a single SQL file: views, then 2 back-to-back iterations per query.
# Delimiter markers (.print) separate query sections in the output;
# they are dot-commands, not SQL, so they won't invalidate the scan cache.
# ---------------------------------------------------------------------------
MARKER_PREFIX="__TPCH_MARKER__"
END_MARKER="__TPCH_END__"

TEMP_SQL=$(mktemp /tmp/tpch_all_XXXXXX.sql)
printf '%s\n' "$VIEW_SQL" > "$TEMP_SQL"
echo ".timer on" >> "$TEMP_SQL"

VALID_QUERIES=()
# Queries where scan results must be cached in host instead of GPU (too large to fit in GPU memory).
# Only needed at SF1000+; at smaller scale factors everything fits in GPU memory.
if [ "$SF" -ge 1000 ] 2>/dev/null; then
    HOST_CACHE_QUERIES="1 7 9 10 17 18 19 21"
else
    HOST_CACHE_QUERIES=""
fi
for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    if [ ! -f "$QUERY_FILE" ]; then
        echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
        continue
    fi
    VALID_QUERIES+=("$q")
    # Set per-query scan cache level.  Bracket the SET with .timer off/on
    # so it doesn't produce a spurious "Run Time" line in the output.
    if [ "$ENGINE" = "sirius" ]; then
        if echo " $HOST_CACHE_QUERIES " | grep -q " $q "; then
            printf ".timer off\nSET scan_cache_level = 'table_host';\n.timer on\n" >> "$TEMP_SQL"
        else
            printf ".timer off\nSET scan_cache_level = 'table_gpu';\n.timer on\n" >> "$TEMP_SQL"
        fi
    fi
    echo ".print ${MARKER_PREFIX} ${q}" >> "$TEMP_SQL"
    # N iterations back-to-back — nothing between them.
    for ((iter = 0; iter < NUM_ITERATIONS; iter++)); do
        cat "$QUERY_FILE" >> "$TEMP_SQL"
        printf '\n' >> "$TEMP_SQL"
    done
done
echo ".print ${END_MARKER}" >> "$TEMP_SQL"

if [ -n "${OUTPUT_DIR:-}" ]; then
    mkdir -p "$OUTPUT_DIR"
    cp "$TEMP_SQL" "$OUTPUT_DIR/all_queries.sql"
fi

# ---------------------------------------------------------------------------
# Run DuckDB once for all queries, with optional session timeout
# ---------------------------------------------------------------------------
echo ""
echo "Running all queries in a single DuckDB session..."
START_TIME=$(date +%s.%N)
if [ "$SESSION_TIMEOUT" -gt 0 ] 2>/dev/null; then
    if [ -n "${OUTPUT_DIR:-}" ]; then
        FULL_OUTPUT=$(timeout "$SESSION_TIMEOUT" env SIRIUS_LOG_DIR="$OUTPUT_DIR" "$DUCKDB" -f "$TEMP_SQL" 2>&1)
    else
        FULL_OUTPUT=$(timeout "$SESSION_TIMEOUT" "$DUCKDB" -f "$TEMP_SQL" 2>&1)
    fi
else
    if [ -n "${OUTPUT_DIR:-}" ]; then
        FULL_OUTPUT=$(SIRIUS_LOG_DIR="$OUTPUT_DIR" "$DUCKDB" -f "$TEMP_SQL" 2>&1)
    else
        FULL_OUTPUT=$("$DUCKDB" -f "$TEMP_SQL" 2>&1)
    fi
fi
SESSION_EXIT=$?
END_TIME=$(date +%s.%N)

TOTAL_ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
echo "Total wall-clock time: ${TOTAL_ELAPSED}s"

if [ "$SESSION_EXIT" -eq 124 ]; then
    echo "SESSION TIMEOUT: DuckDB was killed after ${SESSION_TIMEOUT}s"
elif [ "$SESSION_EXIT" -ne 0 ]; then
    echo "SESSION FAILED: DuckDB exited with code $SESSION_EXIT"
fi

rm -f "$TEMP_SQL"

# ---------------------------------------------------------------------------
# Parse output: split by markers, extract per-query results and timings.
#
# Each query section (between its marker and the next) contains:
#   <iter1 result>
#   Run Time (s): real X.XXX user Y.YYY sys Z.ZZZ
#   <iter2 result>
#   Run Time (s): real X.XXX user Y.YYY sys Z.ZZZ
#
# We extract:
#   result.txt  — warm-run output only (between the two "Run Time" lines)
#   timings.csv — cold and warm real times
# ---------------------------------------------------------------------------
TEMP_OUTPUT=$(mktemp /tmp/tpch_output_XXXXXX.txt)
echo "$FULL_OUTPUT" > "$TEMP_OUTPUT"

for q in "${VALID_QUERIES[@]}"; do
    if [ -n "${OUTPUT_DIR:-}" ]; then
        Q_DIR="$OUTPUT_DIR/q${q}"
        mkdir -p "$Q_DIR"
        RESULT_FILE="$Q_DIR/result.txt"
        TIMING_FILE="$Q_DIR/timings.csv"
        cp "$QUERY_DIR/q${q}.sql" "$Q_DIR/query.sql"
    else
        RESULT_FILE="$PROJECT_DIR/result_${ENGINE}_sf${SF}_q${q}.txt"
        TIMING_FILE="$PROJECT_DIR/timings_${ENGINE}_sf${SF}_q${q}.csv"
    fi

    echo ""
    echo "========== Q${q} =========="

    # Extract the section between this query's marker and the next marker.
    SECTION=$(awk -v start="${MARKER_PREFIX} ${q}" \
                  -v prefix="${MARKER_PREFIX}" \
                  -v end="${END_MARKER}" '
        $0 == start                                   { cap = 1; next }
        cap && ($0 == end || index($0, prefix) == 1)  { exit }
        cap                                           { print }
    ' "$TEMP_OUTPUT")

    if [ -z "$SECTION" ]; then
        # No output for this query — session likely timed out or crashed before reaching it.
        echo "  NO OUTPUT (session may have timed out or crashed before this query)"
        echo "no output" > "$RESULT_FILE"
        {
            echo "step,runtime_s"
            for ((i = 0; i < NUM_ITERATIONS; i++)); do
                echo "iter_$((i + 1)),N/A"
            done
        } > "$TIMING_FILE"
        echo "  Timings written to $TIMING_FILE"
        continue
    fi

    # Save last-iteration result only (lines between the 2nd-to-last and last "Run Time" lines).
    awk -v n="$NUM_ITERATIONS" '
        /Run Time \(s\):/ { tc++; next }
        tc == (n - 1)     { print }
    ' <<< "$SECTION" > "$RESULT_FILE"

    # Extract per-iteration timings.
    readarray -t TIMES < <(grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+' <<< "$SECTION")

    {
        echo "step,runtime_s"
        for ((i = 0; i < ${#TIMES[@]}; i++)); do
            echo "iter_$((i + 1)),${TIMES[$i]}"
        done
    } > "$TIMING_FILE"

    cold="${TIMES[0]:-N/A}"
    warm="${TIMES[${#TIMES[@]}-1]:-N/A}"
    echo "  Cold: ${cold}s   Warm: ${warm}s   (${#TIMES[@]} iterations)"

    if [ -n "${TIMING_CSV:-}" ] && [ "$cold" != "N/A" ]; then
        echo "${q},${cold}" >> "$TIMING_CSV"
    fi

    echo "  Timings written to $TIMING_FILE"
done

rm -f "$TEMP_OUTPUT"

# ---------------------------------------------------------------------------
# Split the Sirius log into per-query segments.
#
# The log contains "QueryBegin: call gpu_execution(...)" lines for each
# iteration.  We group every 2 consecutive gpu_execution QueryBegin entries
# (cold + warm) into one query segment and copy it to Q_DIR/sirius.log.
# The combined log is kept in OUTPUT_DIR.
# ---------------------------------------------------------------------------
if [ "$ENGINE" = "sirius" ] && [ -n "${OUTPUT_DIR:-}" ] && [ ${#VALID_QUERIES[@]} -gt 0 ]; then
    # spdlog daily sink names files sirius_YYYY-MM-DD.log; find the most recent one.
    LOG_FILE=""
    for f in "$OUTPUT_DIR"/sirius*.log; do
        [ -f "$f" ] && LOG_FILE="$f"
    done
    if [ -n "$LOG_FILE" ]; then
        echo ""
        echo "Splitting Sirius log per query..."
        readarray -t QB_LINES < <(grep -n 'QueryBegin: call' "$LOG_FILE" | cut -d: -f1)
        TOTAL_LOG_LINES=$(wc -l < "$LOG_FILE")

        for ((i = 0; i < ${#VALID_QUERIES[@]}; i++)); do
            q="${VALID_QUERIES[$i]}"
            start_idx=$((i * 2))
            next_idx=$(((i + 1) * 2))

            [ "$start_idx" -ge "${#QB_LINES[@]}" ] && continue
            start_line="${QB_LINES[$start_idx]}"

            if [ "$next_idx" -lt "${#QB_LINES[@]}" ]; then
                end_line=$((QB_LINES[$next_idx] - 1))
            else
                end_line="$TOTAL_LOG_LINES"
            fi

            sed -n "${start_line},${end_line}p" "$LOG_FILE" > "$OUTPUT_DIR/q${q}/sirius.log"
            echo "  Q${q}: lines ${start_line}-${end_line} -> q${q}/sirius.log"
        done
    fi
fi

echo ""
echo "=========================================="
echo "All queries complete."
if [ -n "${OUTPUT_DIR:-}" ]; then
    echo "Results saved under $OUTPUT_DIR"
else
    echo "Results saved as result_${ENGINE}_sf${SF}_q*.txt"
    echo "Timings saved as timings_${ENGINE}_sf${SF}_q*.csv"
fi
