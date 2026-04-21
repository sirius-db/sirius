#!/usr/bin/env bash
# Run TPC-H GPU queries against Parquet files
#
# By default, all specified queries run in a single DuckDB session (single-
# session mode).  This keeps the Sirius scan cache valid across queries.
#
# Use --multi-session to run each query in its own fresh DuckDB process.
# This is useful for DuckDB CPU baselines where you want independent runs.
#
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
#   ./test/tpch_performance/run_tpch_parquet.sh [options] <engine> <scale_factor> <query_numbers...>
# with engine = [sirius/duckdb]
#
# Options:
#   --parquet-dir <path>  Directory containing TPC-H parquet files
#   --iterations <N>      Number of iterations per query (default: 2)
#   --timeout <seconds>   Kill DuckDB session after N seconds (default: 1200)
#   --multi-session       Run each query in its own DuckDB process
#
# Example:
#   ./test/tpch_performance/run_tpch_parquet.sh sirius 100 `seq 1 22`
#   ./test/tpch_performance/run_tpch_parquet.sh --multi-session duckdb 100 `seq 1 22`
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
SCAN_CACHE_LEVEL=""
MULTI_SESSION=false
while [ "${1:-}" = "--parquet-dir" ] || [ "${1:-}" = "--iterations" ] || [ "${1:-}" = "--timeout" ] || [ "${1:-}" = "--cache-level" ] || [ "${1:-}" = "--multi-session" ]; do
    if [ "$1" = "--parquet-dir" ]; then
        PARQUET_DIR="$2"
        shift 2
    elif [ "$1" = "--iterations" ]; then
        NUM_ITERATIONS="$2"
        shift 2
    elif [ "$1" = "--timeout" ]; then
        SESSION_TIMEOUT="$2"
        shift 2
    elif [ "$1" = "--cache-level" ]; then
        SCAN_CACHE_LEVEL="$2"
        shift 2
    elif [ "$1" = "--multi-session" ]; then
        MULTI_SESSION=true
        shift
    fi
done

if [ $# -lt 3 ]; then
    echo "Usage: $0 [--parquet-dir <path>] [--iterations <N>] [--timeout <seconds>] [--multi-session] <engine> <scale_factor> <query_numbers...>"
    echo "Example: $0 sirius 100 \`seq 1 22\`"
    echo "  --iterations N    Number of iterations per query (default: 2, 1 cold + N-1 warm)"
    echo "  --timeout N       Kill the DuckDB session after N seconds (default: 1200, 0 = no timeout)"
    echo "  --multi-session   Run each query in its own DuckDB process (fresh state per query)"
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
DUCKDB="$SIRIUS_DUCKDB"
# Both engines use the same plain SQL queries — transparent execution
# routes queries through GPU when SiriusContext is initialized.
QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/orig"
if [ "$ENGINE" != "sirius" ]; then
    # Disable Sirius so the extension doesn't initialize (CPU-only).
    export SIRIUS_DISABLE=1
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

# Load per-query scan cache level overrides from config file.
# Format: <query_number> <cache_level> (one per line, # comments ignored).
# Queries not listed use the default (table_gpu).
CACHE_CONFIG="$SCRIPT_DIR/scan_cache_levels.conf"
declare -A QUERY_CACHE_LEVEL
if [ "$SF" -ge 1000 ] 2>/dev/null && [ -f "$CACHE_CONFIG" ]; then
    while IFS=' ' read -r qnum level; do
        [[ -z "$qnum" || "$qnum" == \#* ]] && continue
        QUERY_CACHE_LEVEL[$qnum]="$level"
    done < "$CACHE_CONFIG"
fi

# Build list of valid queries (those with existing SQL files).
VALID_QUERIES=()
for q in "${QUERIES[@]}"; do
    QUERY_FILE="$QUERY_DIR/q${q}.sql"
    if [ ! -f "$QUERY_FILE" ]; then
        echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
        continue
    fi
    VALID_QUERIES+=("$q")
done

SESSION_MODE="single (all queries in one process)"
if [ "$MULTI_SESSION" = true ]; then
    SESSION_MODE="multi (fresh process per query)"
fi

echo "Running TPC-H queries against SF${SF} parquet data"
echo "Engine: $ENGINE"
echo "Parquet dir: $PARQUET_DIR"
echo "Session: $SESSION_MODE"
echo "Iterations: $NUM_ITERATIONS (1 cold + $((NUM_ITERATIONS - 1)) warm)"
echo "Queries: ${QUERIES[*]}"
if [ "$SESSION_TIMEOUT" -gt 0 ] 2>/dev/null; then
    echo "Session timeout: ${SESSION_TIMEOUT}s"
else
    echo "Session timeout: disabled"
fi
echo "=========================================="

# =============================================================================
# Single-session mode: all queries in one DuckDB process
# =============================================================================
run_single_session() {
    # Build a single SQL file: views, then N back-to-back iterations per query.
    # Delimiter markers (.print) separate query sections in the output;
    # they are dot-commands, not SQL, so they won't invalidate the scan cache.
    local MARKER_PREFIX="__TPCH_MARKER__"
    local END_MARKER="__TPCH_END__"

    local TEMP_SQL
    TEMP_SQL=$(mktemp /tmp/tpch_all_XXXXXX.sql)
    printf '%s\n' "$VIEW_SQL" > "$TEMP_SQL"
    echo ".timer on" >> "$TEMP_SQL"

    for q in "${VALID_QUERIES[@]}"; do
        local QUERY_FILE="$QUERY_DIR/q${q}.sql"
        # Set per-query scan cache level.  Bracket the SET with .timer off/on
        # so it doesn't produce a spurious "Run Time" line in the output.
        if [ "$ENGINE" = "sirius" ]; then
            local qlevel="${SCAN_CACHE_LEVEL:-${QUERY_CACHE_LEVEL[$q]:-table_gpu}}"
            printf ".timer off\nSET scan_cache_level = '%s';\n.timer on\n" "$qlevel" >> "$TEMP_SQL"
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

    # Run DuckDB once for all queries, with optional session timeout.
    echo ""
    echo "Running all queries in a single DuckDB session..."
    local START_TIME END_TIME FULL_OUTPUT SESSION_EXIT TOTAL_ELAPSED
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

    # Parse output: split by markers, extract per-query results and timings.
    local TEMP_OUTPUT
    TEMP_OUTPUT=$(mktemp /tmp/tpch_output_XXXXXX.txt)
    echo "$FULL_OUTPUT" > "$TEMP_OUTPUT"

    for q in "${VALID_QUERIES[@]}"; do
        local RESULT_FILE TIMING_FILE
        if [ -n "${OUTPUT_DIR:-}" ]; then
            local Q_DIR="$OUTPUT_DIR/q${q}"
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
        local SECTION
        SECTION=$(awk -v start="${MARKER_PREFIX} ${q}" \
                      -v prefix="${MARKER_PREFIX}" \
                      -v end="${END_MARKER}" '
            $0 == start                                   { cap = 1; next }
            cap && ($0 == end || index($0, prefix) == 1)  { exit }
            cap                                           { print }
        ' "$TEMP_OUTPUT")

        if [ -z "$SECTION" ]; then
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
        local TIMES
        readarray -t TIMES < <(grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+' <<< "$SECTION")

        {
            echo "step,runtime_s"
            for ((i = 0; i < ${#TIMES[@]}; i++)); do
                echo "iter_$((i + 1)),${TIMES[$i]}"
            done
        } > "$TIMING_FILE"

        local cold="${TIMES[0]:-N/A}"
        local warm="N/A"
        for ((i = 1; i < ${#TIMES[@]}; i++)); do
            if [ "$warm" = "N/A" ] || (( $(echo "${TIMES[$i]} < $warm" | bc -l) )); then
                warm="${TIMES[$i]}"
            fi
        done
        echo "  Cold: ${cold}s   Warm(best): ${warm}s   (${#TIMES[@]} iterations)"

        if [ -n "${TIMING_CSV:-}" ] && [ "$cold" != "N/A" ]; then
            echo "${q},${cold}" >> "$TIMING_CSV"
        fi

        echo "  Timings written to $TIMING_FILE"
    done

    rm -f "$TEMP_OUTPUT"
}

# =============================================================================
# Multi-session mode: each query in its own fresh DuckDB process (duckdb only)
# =============================================================================
run_multi_session() {
    for q in "${VALID_QUERIES[@]}"; do
        local QUERY_FILE="$QUERY_DIR/q${q}.sql"

        local RESULT_FILE TIMING_FILE
        if [ -n "${OUTPUT_DIR:-}" ]; then
            local Q_DIR="$OUTPUT_DIR/q${q}"
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

        # Build per-query SQL: views + scan cache level (sirius) + timer + N iterations.
        local TEMP_SQL
        TEMP_SQL=$(mktemp /tmp/tpch_q${q}_XXXXXX.sql)
        {
            printf '%s\n' "$VIEW_SQL"
            if [ "$ENGINE" = "sirius" ]; then
                local qlevel="${SCAN_CACHE_LEVEL:-${QUERY_CACHE_LEVEL[$q]:-table_gpu}}"
                printf "SET scan_cache_level = '%s';\n" "$qlevel"
            fi
            printf ".timer on\n"
            for ((iter = 0; iter < NUM_ITERATIONS; iter++)); do
                cat "$QUERY_FILE"
                printf '\n'
            done
        } > "$TEMP_SQL"

        # Run in a fresh DuckDB process.
        # For sirius, set SIRIUS_LOG_DIR to the per-query directory so logs are isolated.
        local OUTPUT=""
        local Q_EXIT=0
        local RUN_ENV=("$DUCKDB" -f "$TEMP_SQL")
        if [ "$ENGINE" = "sirius" ] && [ -n "${Q_DIR:-}" ]; then
            RUN_ENV=(env SIRIUS_LOG_DIR="$Q_DIR" "${RUN_ENV[@]}")
        fi
        if [ "$SESSION_TIMEOUT" -gt 0 ] 2>/dev/null; then
            OUTPUT=$(timeout "$SESSION_TIMEOUT" "${RUN_ENV[@]}" 2>&1) || Q_EXIT=$?
        else
            OUTPUT=$("${RUN_ENV[@]}" 2>&1) || Q_EXIT=$?
        fi

        rm -f "$TEMP_SQL"

        if [ "$Q_EXIT" -eq 124 ]; then
            echo "  TIMEOUT: killed after ${SESSION_TIMEOUT}s"
        elif [ "$Q_EXIT" -ne 0 ]; then
            echo "  FAILED: DuckDB exited with code $Q_EXIT"
        fi

        # Check for errors in output.
        local HAS_ERROR
        HAS_ERROR=$(echo "$OUTPUT" | grep -ci "error" || true)

        if [ "$HAS_ERROR" -gt 0 ] && [ "$Q_EXIT" -ne 0 ]; then
            local ERROR_MSG
            ERROR_MSG=$(echo "$OUTPUT" | grep -i "error" | head -1)
            echo "  Error: $ERROR_MSG"
            echo "error: $ERROR_MSG" > "$RESULT_FILE"
            {
                echo "step,runtime_s"
                for ((i = 0; i < NUM_ITERATIONS; i++)); do
                    echo "iter_$((i + 1)),N/A"
                done
            } > "$TIMING_FILE"
            echo "  Timings written to $TIMING_FILE"
            continue
        fi

        # Save last-iteration result (lines between the 2nd-to-last and last "Run Time" lines).
        awk -v n="$NUM_ITERATIONS" '
            /Run Time \(s\):/ { tc++; next }
            tc == (n - 1)     { print }
        ' <<< "$OUTPUT" > "$RESULT_FILE"

        # Extract per-iteration timings.
        local TIMES
        readarray -t TIMES < <(grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+' <<< "$OUTPUT")

        {
            echo "step,runtime_s"
            for ((i = 0; i < ${#TIMES[@]}; i++)); do
                echo "iter_$((i + 1)),${TIMES[$i]}"
            done
        } > "$TIMING_FILE"

        local cold="${TIMES[0]:-N/A}"
        local warm="N/A"
        for ((i = 1; i < ${#TIMES[@]}; i++)); do
            if [ "$warm" = "N/A" ] || (( $(echo "${TIMES[$i]} < $warm" | bc -l) )); then
                warm="${TIMES[$i]}"
            fi
        done
        echo "  Cold: ${cold}s   Warm(best): ${warm}s   (${#TIMES[@]} iterations)"

        if [ -n "${TIMING_CSV:-}" ] && [ "$cold" != "N/A" ]; then
            echo "${q},${cold}" >> "$TIMING_CSV"
        fi

        echo "  Timings written to $TIMING_FILE"
    done
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
if [ "$MULTI_SESSION" = true ]; then
    run_multi_session
else
    run_single_session
fi

# ---------------------------------------------------------------------------
# Split the Sirius log into per-query segments.
#
# The log contains "QueryBegin: call gpu_execution(...)" lines for each
# iteration.  We group every NUM_ITERATIONS consecutive QueryBegin entries
# (one per iteration) into one query segment and copy it to Q_DIR/sirius.log.
# The combined log is kept in OUTPUT_DIR.
# ---------------------------------------------------------------------------
if [ "$ENGINE" = "sirius" ] && [ "$MULTI_SESSION" = false ] && [ -n "${OUTPUT_DIR:-}" ] && [ ${#VALID_QUERIES[@]} -gt 0 ]; then
    # spdlog daily sink names files sirius_YYYY-MM-DD.log; find the most recent one.
    LOG_FILE=""
    for f in "$OUTPUT_DIR"/sirius*.log; do
        [ -f "$f" ] && LOG_FILE="$f"
    done
    if [ -n "$LOG_FILE" ]; then
        echo ""
        echo "Splitting Sirius log per query (${NUM_ITERATIONS} iterations per query)..."
        readarray -t QB_LINES < <(grep -n 'QueryBegin: call' "$LOG_FILE" | cut -d: -f1)
        TOTAL_LOG_LINES=$(wc -l < "$LOG_FILE")

        for ((i = 0; i < ${#VALID_QUERIES[@]}; i++)); do
            q="${VALID_QUERIES[$i]}"
            start_idx=$((i * NUM_ITERATIONS))
            next_idx=$(((i + 1) * NUM_ITERATIONS))

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
