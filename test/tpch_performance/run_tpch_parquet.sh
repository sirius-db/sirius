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
DROP_OS_CACHE=false
MULTI_SESSION=false
PINNING_MODE="none"
PIN_AFTER_ITERATION=0
while [ "${1:-}" = "--parquet-dir" ] || [ "${1:-}" = "--iterations" ] || [ "${1:-}" = "--timeout" ] || [ "${1:-}" = "--drop-os-cache" ] || [ "${1:-}" = "--multi-session" ] || [ "${1:-}" = "--pinning-mode" ] || [ "${1:-}" = "--pin-after-iteration" ]; do
    if [ "$1" = "--parquet-dir" ]; then
        PARQUET_DIR="$2"
        shift 2
    elif [ "$1" = "--iterations" ]; then
        NUM_ITERATIONS="$2"
        shift 2
    elif [ "$1" = "--timeout" ]; then
        SESSION_TIMEOUT="$2"
        shift 2
    elif [ "$1" = "--drop-os-cache" ]; then
        DROP_OS_CACHE=true
        shift
    elif [ "$1" = "--multi-session" ]; then
        MULTI_SESSION=true
        shift
    elif [ "$1" = "--pinning-mode" ]; then
        PINNING_MODE="$2"
        shift 2
    elif [ "$1" = "--pin-after-iteration" ]; then
        PIN_AFTER_ITERATION="$2"
        shift 2
    fi
done

case "$PINNING_MODE" in
    none|per-query|pinned-hot) ;;
    *)
        echo "ERROR: --pinning-mode must be 'none', 'per-query', or 'pinned-hot' (got: $PINNING_MODE)"
        exit 1
        ;;
esac

if [ $# -lt 3 ]; then
    echo "Usage: $0 [--parquet-dir <path>] [--iterations <N>] [--timeout <seconds>] [--multi-session] [--drop-os-cache] [--pinning-mode none|per-query|pinned-hot] [--pin-after-iteration <N>] <engine> <scale_factor> <query_numbers...>"
    echo "Example: $0 sirius 100 \`seq 1 22\`"
    echo "  --iterations N      Number of iterations per query (default: 2, 1 cold + N-1 warm)"
    echo "  --timeout N         Kill the DuckDB session after N seconds (default: 1200, 0 = no timeout)"
    echo "  --multi-session     Run each query in its own DuckDB process (fresh state per query)"
    echo "  --drop-os-cache     Drop OS filesystem cache before each query (requires --multi-session and sudo)"
    echo "  --pinning-mode MODE 'per-query' calls pin_table for each query's columns, runs its remaining"
    echo "                      iterations, then unpin_table. 'pinned-hot' pins the union of"
    echo "                      referenced columns once for the whole single-session run."
    echo "                      Sirius engine only. Default: 'none'."
    echo "  --pin-after-iteration N   With --pinning-mode per-query: run the first N of each query's"
    echo "                      iterations unpinned (e.g. cold + warm), then pin for the rest"
    echo "                      (e.g. hot). Default: 0 (pin from the first iteration). Ignored"
    echo "                      with --pinning-mode pinned-hot (there is no per-query split)."
    exit 1
fi

if [ "$DROP_OS_CACHE" = true ] && [ "$MULTI_SESSION" = false ]; then
    echo "ERROR: --drop-os-cache requires --multi-session (each query must run in its own process)"
    exit 1
fi

if [ "$DROP_OS_CACHE" = true ]; then
    if ! sudo -n -l /usr/bin/tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
        echo "ERROR: --drop-os-cache requires passwordless sudo for /usr/bin/tee."
        echo "Configure it with:"
        echo "  echo '\$(whoami) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches' | sudo tee /etc/sudoers.d/drop_caches"
        exit 1
    fi
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

if [ "$PINNING_MODE" = "pinned-hot" ] && [ "$MULTI_SESSION" = true ] && [ "$ENGINE" = "sirius" ]; then
    echo "ERROR: --pinning-mode pinned-hot requires single-session mode; remove --multi-session."
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

has_parquet_data() {
    local parquet_dir="$1"
    local parquet_file
    for parquet_file in "$parquet_dir"/*.parquet "$parquet_dir"/*/*.parquet; do
        [ -f "$parquet_file" ] && return 0
    done
    return 1
}

ensure_parquet_data() {
    if [ -d "$PARQUET_DIR" ] && has_parquet_data "$PARQUET_DIR"; then
        return 0
    fi

    echo "Parquet directory not found: $PARQUET_DIR"
    echo "Generating TPC-H SF${SF} dataset..."
    (
        cd "$SCRIPT_DIR" &&
            pixi run bash generate_tpch_data.sh "$SF" "$PARQUET_DIR"
    )
    local status=$?
    if [ "$status" -ne 0 ]; then
        echo "ERROR: failed to generate parquet data for SF${SF} (exit code ${status})."
        return "$status"
    fi

    if ! [ -d "$PARQUET_DIR" ] || ! has_parquet_data "$PARQUET_DIR"; then
        echo "ERROR: parquet data is still unavailable after generation: $PARQUET_DIR"
        return 1
    fi
}

ensure_parquet_data
status=$?
if [ "$status" -ne 0 ]; then
    exit "$status"
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
if [ "$DROP_OS_CACHE" = true ]; then
    echo "Drop OS cache: enabled"
fi
echo "Iterations: $NUM_ITERATIONS (1 cold + $((NUM_ITERATIONS - 1)) warm)"
if [ "$PINNING_MODE" != "none" ]; then
    if [ "$ENGINE" = "sirius" ]; then
        if [ "$PINNING_MODE" = "per-query" ]; then
            echo "Pinning mode: $PINNING_MODE (pin_table per query, tier=${SIRIUS_PIN_TIER:-gpu}, pin after iteration ${PIN_AFTER_ITERATION})"
        else
            echo "Pinning mode: $PINNING_MODE (pin_table once before timed queries, tier=${SIRIUS_PIN_TIER:-gpu})"
        fi
    else
        echo "Pinning mode: $PINNING_MODE (ignored — Sirius-only feature)"
    fi
fi
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

    local PIN_PER_QUERY=false
    local PINNED_HOT=false
    if [ "$PINNING_MODE" = "per-query" ] && [ "$ENGINE" = "sirius" ]; then
        PIN_PER_QUERY=true
    elif [ "$PINNING_MODE" = "pinned-hot" ] && [ "$ENGINE" = "sirius" ]; then
        PINNED_HOT=true
    fi

    local TEMP_SQL
    TEMP_SQL=$(mktemp /tmp/tpch_all_XXXXXX.sql)
    printf '%s\n' "$VIEW_SQL" > "$TEMP_SQL"
    echo ".timer on" >> "$TEMP_SQL"

    if [ "$PINNED_HOT" = true ]; then
        echo ".print __TPCH_PIN_BEGIN__ all" >> "$TEMP_SQL"
        python3 "$SCRIPT_DIR/tpch_pin_columns.py" pin-all "$PARQUET_DIR" >> "$TEMP_SQL"
        echo ".print __TPCH_PIN_END__ all" >> "$TEMP_SQL"
    fi

    for q in "${VALID_QUERIES[@]}"; do
        local QUERY_FILE="$QUERY_DIR/q${q}.sql"
        echo ".print ${MARKER_PREFIX} ${q}" >> "$TEMP_SQL"

        # --pin-after-iteration splits this query's N iterations into an
        # unpinned prefix (e.g. cold + warm) and a pinned suffix (e.g. hot).
        # Pin/unpin brackets live INSIDE the __TPCH_MARKER__ section now (mid-query
        # when PIN_POINT > 0), so the result-extraction awk below skips their
        # Run Time output explicitly instead of relying on section boundaries.
        local PIN_POINT=0
        if [ "$PIN_PER_QUERY" = true ]; then
            PIN_POINT="$PIN_AFTER_ITERATION"
            [ "$PIN_POINT" -gt "$NUM_ITERATIONS" ] && PIN_POINT="$NUM_ITERATIONS"
        fi

        # Unpinned iterations first (0 of them when PIN_POINT is 0 — the old
        # "pin from the start" behavior).
        for ((iter = 0; iter < PIN_POINT; iter++)); do
            cat "$QUERY_FILE" >> "$TEMP_SQL"
            printf '\n' >> "$TEMP_SQL"
        done

        local PINNED_THIS_QUERY=false
        if [ "$PIN_PER_QUERY" = true ] && [ "$PIN_POINT" -lt "$NUM_ITERATIONS" ]; then
            PINNED_THIS_QUERY=true
            echo ".print __TPCH_PIN_BEGIN__ ${q}" >> "$TEMP_SQL"
            python3 "$SCRIPT_DIR/tpch_pin_columns.py" pin "$q" "$PARQUET_DIR" >> "$TEMP_SQL"
            echo ".print __TPCH_PIN_END__ ${q}" >> "$TEMP_SQL"
        fi

        # Remaining iterations (pinned, if pinning is on and any are left).
        for ((iter = PIN_POINT; iter < NUM_ITERATIONS; iter++)); do
            cat "$QUERY_FILE" >> "$TEMP_SQL"
            printf '\n' >> "$TEMP_SQL"
        done

        if [ "$PINNED_THIS_QUERY" = true ]; then
            echo ".print __TPCH_UNPIN_BEGIN__ ${q}" >> "$TEMP_SQL"
            python3 "$SCRIPT_DIR/tpch_pin_columns.py" unpin "$q" >> "$TEMP_SQL"
            echo ".print __TPCH_UNPIN_END__ ${q}" >> "$TEMP_SQL"
        fi
    done

    if [ "$PINNED_HOT" = true ]; then
        echo ".print __TPCH_UNPIN_BEGIN__ all" >> "$TEMP_SQL"
        python3 "$SCRIPT_DIR/tpch_pin_columns.py" unpin-all >> "$TEMP_SQL"
        echo ".print __TPCH_UNPIN_END__ all" >> "$TEMP_SQL"
    fi

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

    local SESSION_OUTPUT_FILE
    if [ -n "${OUTPUT_DIR:-}" ]; then
        SESSION_OUTPUT_FILE="$OUTPUT_DIR/session_output.txt"
    else
        SESSION_OUTPUT_FILE="$PROJECT_DIR/session_output_${ENGINE}_sf${SF}.txt"
    fi
    printf '%s\n' "$FULL_OUTPUT" > "$SESSION_OUTPUT_FILE"

    local RUN_STATUS=0
    if [ "$SESSION_EXIT" -eq 124 ]; then
        echo "SESSION TIMEOUT: DuckDB was killed after ${SESSION_TIMEOUT}s"
        RUN_STATUS=124
    elif [ "$SESSION_EXIT" -ne 0 ]; then
        echo "SESSION FAILED: DuckDB exited with code $SESSION_EXIT"
        echo "DuckDB output saved to $SESSION_OUTPUT_FILE"
        echo "DuckDB error excerpt:"
        local ERROR_EXCERPT
        ERROR_EXCERPT=$(printf '%s\n' "$FULL_OUTPUT" \
            | grep -iE '(^Error:|Invalid Error|IO Error|Catalog Error|Parser Error|Binder Error|Out of Memory|std::bad_alloc|CUDA|RMM|Exception)' \
            | head -20)
        if [ -n "$ERROR_EXCERPT" ]; then
            printf '%s\n' "$ERROR_EXCERPT"
        else
            printf '%s\n' "$FULL_OUTPUT" | tail -40
        fi
        RUN_STATUS=$SESSION_EXIT
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

        # Extract this query's section: starts at its marker, ends at the next
        # query's marker (or the final end marker). --pin-after-iteration can put
        # a __TPCH_PIN_*/__TPCH_UNPIN_* bracket in the MIDDLE of this section (once
        # pinning kicks in partway through the iterations), so drop everything
        # between a *_BEGIN__ and its matching *_END__ line rather than treating
        # any '__TPCH_' line as the end of the section.
        local SECTION
        SECTION=$(awk -v start="${MARKER_PREFIX} ${q}" -v marker_prefix="${MARKER_PREFIX}" -v end_marker="${END_MARKER}" '
            {
                if (cap == 0) { if ($0 == start) cap = 1; next }
                if (skip == 1) {
                    if (index($0, "__TPCH_PIN_END__") == 1 || index($0, "__TPCH_UNPIN_END__") == 1) skip = 0
                    next
                }
                if (index($0, "__TPCH_PIN_BEGIN__") == 1 || index($0, "__TPCH_UNPIN_BEGIN__") == 1) { skip = 1; next }
                if (index($0, marker_prefix) == 1 || $0 == end_marker) exit
                print
            }
        ' "$TEMP_OUTPUT")

        if [ -z "$SECTION" ]; then
            echo "  NO OUTPUT (session may have timed out or crashed before this query)"
            {
                echo "error: no output (session may have timed out or crashed before this query)"
                echo "session_output: $SESSION_OUTPUT_FILE"
            } > "$RESULT_FILE"
            {
                echo "step,runtime_s"
                for ((i = 0; i < NUM_ITERATIONS; i++)); do
                    echo "iter_$((i + 1)),N/A"
                done
            } > "$TIMING_FILE"
            [ "$RUN_STATUS" -eq 0 ] && RUN_STATUS=1
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
    return "$RUN_STATUS"
}

# =============================================================================
# Multi-session mode: each query in its own fresh DuckDB process (duckdb only)
# =============================================================================
run_multi_session() {
    local RUN_STATUS=0
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

        # Drop OS filesystem cache for true cold-run benchmarking.
        if [ "$DROP_OS_CACHE" = true ]; then
            echo "  Dropping OS filesystem cache..."
            sync
            if echo 3 | sudo -n /usr/bin/tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
                echo "  OS cache dropped."
            else
                echo "  ERROR: Failed to drop OS cache. Configure passwordless sudo:"
                echo "    echo '\$(whoami) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches' | sudo tee /etc/sudoers.d/drop_caches"
                exit 1
            fi
        fi

        # Build per-query SQL: views + timer + N iterations.
        # In --pinning-mode per-query, pin before iterations and unpin after — the unpin
        # is mandatory even though the process is about to exit, to release host-pinned
        # memory cleanly back to the allocator before the next process starts.
        local PIN_ENABLED=false
        if [ "$PINNING_MODE" = "per-query" ] && [ "$ENGINE" = "sirius" ]; then
            PIN_ENABLED=true
        fi
        local TEMP_SQL
        TEMP_SQL=$(mktemp /tmp/tpch_q${q}_XXXXXX.sql)
        {
            printf '%s\n' "$VIEW_SQL"
            printf ".timer on\n"
            if [ "$PIN_ENABLED" = true ]; then
                printf ".print __TPCH_PIN_BEGIN__ %s\n" "$q"
                python3 "$SCRIPT_DIR/tpch_pin_columns.py" pin "$q" "$PARQUET_DIR"
                printf ".print __TPCH_PIN_END__ %s\n" "$q"
            fi
            for ((iter = 0; iter < NUM_ITERATIONS; iter++)); do
                cat "$QUERY_FILE"
                printf '\n'
            done
            if [ "$PIN_ENABLED" = true ]; then
                printf ".print __TPCH_UNPIN_BEGIN__ %s\n" "$q"
                python3 "$SCRIPT_DIR/tpch_pin_columns.py" unpin "$q"
                printf ".print __TPCH_UNPIN_END__ %s\n" "$q"
            fi
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
            RUN_STATUS=124
        elif [ "$Q_EXIT" -ne 0 ]; then
            echo "  FAILED: DuckDB exited with code $Q_EXIT"
            [ "$RUN_STATUS" -eq 0 ] && RUN_STATUS=$Q_EXIT
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

        # When pinning is on, the session output contains pin/unpin Run Time lines
        # bracketing the iterations. Extract just the iteration window so the
        # downstream awk and grep see only query iterations.
        local PARSE_INPUT
        if [ "$PIN_ENABLED" = true ]; then
            PARSE_INPUT=$(awk '
                index($0, "__TPCH_PIN_END__")    == 1 { cap = 1; next }
                index($0, "__TPCH_UNPIN_BEGIN__") == 1 { exit }
                cap { print }
            ' <<< "$OUTPUT")
        else
            PARSE_INPUT="$OUTPUT"
        fi

        # Save last-iteration result (lines between the 2nd-to-last and last "Run Time" lines).
        awk -v n="$NUM_ITERATIONS" '
            /Run Time \(s\):/ { tc++; next }
            tc == (n - 1)     { print }
        ' <<< "$PARSE_INPUT" > "$RESULT_FILE"

        # Extract per-iteration timings.
        local TIMES
        readarray -t TIMES < <(grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+' <<< "$PARSE_INPUT")

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
    return "$RUN_STATUS"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
RUN_STATUS=0
if [ "$MULTI_SESSION" = true ]; then
    run_multi_session || RUN_STATUS=$?
else
    run_single_session || RUN_STATUS=$?
fi

# ---------------------------------------------------------------------------
# Split the Sirius log into per-query segments.
#
# Under Super Sirius transparent execution, each query iteration is logged as
# "QueryBegin: SQL: <raw SQL>" — there is no `call gpu_execution(...)` wrapper.
# Skip the session prologue (CREATE VIEW for view setup) and any pinning-mode
# CALLs (pin_table / unpin_table) so the remaining QueryBegin lines correspond
# 1:1 to user query iterations. We group every NUM_ITERATIONS consecutive
# entries into one query segment and copy it to Q_DIR/sirius.log. The combined
# log is kept in OUTPUT_DIR.
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
        readarray -t QB_LINES < <(
            grep -nE 'QueryBegin:' "$LOG_FILE" \
                | grep -ivE 'QueryBegin:[[:space:]]*(SQL:[[:space:]]*)?(CREATE VIEW|CALL[[:space:]]+(pin_table|unpin_table))' \
                | cut -d: -f1
        )
        TOTAL_LOG_LINES=$(wc -l < "$LOG_FILE")

        if [ "${#QB_LINES[@]}" -ne $((${#VALID_QUERIES[@]} * NUM_ITERATIONS)) ]; then
            echo "  WARNING: expected $((${#VALID_QUERIES[@]} * NUM_ITERATIONS)) QueryBegin lines (queries × iterations) but found ${#QB_LINES[@]} — split may be misaligned."
        fi

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
exit "$RUN_STATUS"
