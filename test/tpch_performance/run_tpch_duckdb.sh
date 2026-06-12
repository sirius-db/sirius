#!/usr/bin/env bash
# Run TPC-H GPU queries against TPC-H tables stored in a DuckDB database file
# (default: performance_test.duckdb in the project root).
#
# Same session model as run_tpch_parquet.sh: single DuckDB process by default,
# optional --multi-session for CPU baselines.
#
# Output layout matches run_tpch_parquet.sh (OUTPUT_DIR, TIMING_CSV, markers).
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./test/tpch_performance/run_tpch_duckdb.sh [options] <engine> <scale_factor> <query_numbers...>
# with engine = [sirius/duckdb]
#
# Options:
#   --duckdb-file <path>  Path to DuckDB file (default: <project>/performance_test.duckdb)
#   --iterations <N>      Number of iterations per query (default: 2)
#   --timeout <seconds>   Kill DuckDB session after N seconds (default: 1200)
#   --cache-level <lvl>   Default Sirius scan_cache_level (sirius only)
#   --multi-session       Run each query in its own DuckDB process
#   --gpu-native-scan     No-op (sirius only). The GPU-native DuckDB scan is the only
#                         seq_scan path; the plain `sirius` engine already uses it. Kept
#                         as a backward-compatibility alias.
#
# Example:
#   ./test/tpch_performance/run_tpch_duckdb.sh sirius 1 `seq 1 22`   # uses GPU-native scan by default
#   ./test/tpch_performance/run_tpch_duckdb.sh --duckdb-file ./data/perf.duckdb duckdb 1 1 6
#
# Set SIRIUS_NATIVE_SCAN_VERIFY=1 to also emit `SET sirius_log_level='debug'`, exposing the
# GPU-native-scan log markers (off by default to keep query timings clean).
#
# The scale_factor argument is kept for compatibility with benchmark drivers
# (output filenames, logs). It does not select data inside performance_test.duckdb —
# that file must already contain the TPC-H tables (customer, lineitem, ...).
#
# Environment variables:
#   SIRIUS_CONFIG_FILE - path to Sirius config file (required for sirius engine)
#   OUTPUT_DIR         - directory to save per-query results (optional)
#   TIMING_CSV         - path to write per-query timing CSV (optional)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIRIUS_DUCKDB="$PROJECT_DIR/build/release/duckdb"

DUCKDB_FILE=""
NUM_ITERATIONS=2
SESSION_TIMEOUT=1200
DROP_OS_CACHE=false
MULTI_SESSION=false
GPU_NATIVE_SCAN=false
while [ "${1:-}" = "--duckdb-file" ] || [ "${1:-}" = "--iterations" ] || [ "${1:-}" = "--timeout" ] || [ "${1:-}" = "--drop-os-cache" ] || [ "${1:-}" = "--multi-session" ] || [ "${1:-}" = "--gpu-native-scan" ]; do
    if [ "$1" = "--duckdb-file" ]; then
        DUCKDB_FILE="$2"
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
    elif [ "$1" = "--gpu-native-scan" ]; then
        GPU_NATIVE_SCAN=true
        shift
    fi
done

if [ $# -lt 3 ]; then
    echo "Usage: $0 [--duckdb-file <path>] [--iterations <N>] [--timeout <seconds>] [--multi-session] [--drop-os-cache] [--gpu-native-scan] <engine> <scale_factor> <query_numbers...>"
    echo "Example: $0 sirius 1 \`seq 1 22\`"
    echo "  Default database: \$PROJECT_DIR/performance_test.duckdb"
    echo "  --iterations N    Number of iterations per query (default: 2, 1 cold + N-1 warm)"
    echo "  --timeout N       Kill the DuckDB session after N seconds (default: 1200, 0 = no timeout)"
    echo "  --multi-session   Run each query in its own DuckDB process (fresh state per query)"
    echo "  --drop-os-cache   Drop OS filesystem cache before each query (requires --multi-session and sudo)"
    echo "  --gpu-native-scan No-op alias (GPU-native scan is always the sirius seq_scan path)"
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

if [ -z "$DUCKDB_FILE" ]; then
    DUCKDB_FILE="$PROJECT_DIR/performance_test.duckdb"
fi

# Resolve to absolute path (stable if cwd changes).
DUCKDB_FILE="$(cd "$(dirname "$DUCKDB_FILE")" && pwd)/$(basename "$DUCKDB_FILE")"

if [ "$ENGINE" != "sirius" ] && [ "$ENGINE" != "duckdb" ]; then
    echo "Unknown engine, please use sirius or duckdb"
    exit 1
fi
DUCKDB="$SIRIUS_DUCKDB"
QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/orig"
if [ "$ENGINE" != "sirius" ]; then
    export SIRIUS_DISABLE=1
fi

if [ ! -f "$DUCKDB_FILE" ]; then
    echo "DuckDB database not found: $DUCKDB_FILE"
    echo "Create it with TPC-H tables (customer, lineitem, nation, orders, part, partsupp, region, supplier),"
    echo "or pass --duckdb-file <path> to an existing file."
    exit 1
fi

if [ -n "${TIMING_CSV:-}" ]; then
    echo "query,seconds" > "$TIMING_CSV"
fi

# Load per-query scan cache level overrides from YAML config.
# Used in multi-session mode only — in single-session, the Sirius config YAML controls cache level.
CACHE_CONFIG="$SCRIPT_DIR/scan_cache_levels.yaml"
declare -A QUERY_CACHE_LEVEL
if [ -f "$CACHE_CONFIG" ]; then
    while IFS=': ' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        value="${value## }"
        [[ -z "$value" ]] && continue
        QUERY_CACHE_LEVEL[$key]="$value"
    done < "$CACHE_CONFIG"
fi

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

echo "Running TPC-H queries from DuckDB file (sf label: ${SF})"
echo "Engine: $ENGINE"
echo "Database: $DUCKDB_FILE"
echo "Session: $SESSION_MODE"
if [ "$DROP_OS_CACHE" = true ]; then
    echo "Drop OS cache: enabled"
fi
echo "Iterations: $NUM_ITERATIONS (1 cold + $((NUM_ITERATIONS - 1)) warm)"
echo "Queries: ${QUERIES[*]}"
if [ "$SESSION_TIMEOUT" -gt 0 ] 2>/dev/null; then
    echo "Session timeout: ${SESSION_TIMEOUT}s"
else
    echo "Session timeout: disabled"
fi
echo "=========================================="

run_single_session() {
    local MARKER_PREFIX="__TPCH_MARKER__"
    local END_MARKER="__TPCH_END__"

    local TEMP_SQL
    TEMP_SQL=$(mktemp /tmp/tpch_all_XXXXXX.sql)
    echo ".timer on" >> "$TEMP_SQL"

    # The sirius engine always routes seq_scan to the GPU-native scan; no SET needed.
    # SIRIUS_NATIVE_SCAN_VERIFY=1 raises the log level so the native-scan markers appear,
    # emitted once before the first query marker so its Run Time stays outside every
    # timing window.
    if [ "$ENGINE" = "sirius" ] && [ "${SIRIUS_NATIVE_SCAN_VERIFY:-0}" = 1 ]; then
        echo "SET sirius_log_level = 'debug';" >> "$TEMP_SQL"
    fi

    for q in "${VALID_QUERIES[@]}"; do
        local QUERY_FILE="$QUERY_DIR/q${q}.sql"
        echo ".print ${MARKER_PREFIX} ${q}" >> "$TEMP_SQL"
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

    echo ""
    echo "Running all queries in a single DuckDB session..."
    local START_TIME END_TIME FULL_OUTPUT SESSION_EXIT TOTAL_ELAPSED
    START_TIME=$(date +%s.%N)
    if [ "$SESSION_TIMEOUT" -gt 0 ] 2>/dev/null; then
        if [ -n "${OUTPUT_DIR:-}" ]; then
            FULL_OUTPUT=$(timeout "$SESSION_TIMEOUT" env SIRIUS_LOG_DIR="$OUTPUT_DIR" "$DUCKDB" "$DUCKDB_FILE" -f "$TEMP_SQL" 2>&1)
        else
            FULL_OUTPUT=$(timeout "$SESSION_TIMEOUT" "$DUCKDB" "$DUCKDB_FILE" -f "$TEMP_SQL" 2>&1)
        fi
    else
        if [ -n "${OUTPUT_DIR:-}" ]; then
            FULL_OUTPUT=$(SIRIUS_LOG_DIR="$OUTPUT_DIR" "$DUCKDB" "$DUCKDB_FILE" -f "$TEMP_SQL" 2>&1)
        else
            FULL_OUTPUT=$("$DUCKDB" "$DUCKDB_FILE" -f "$TEMP_SQL" 2>&1)
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

        awk -v n="$NUM_ITERATIONS" '
            /Run Time \(s\):/ { tc++; next }
            tc == (n - 1)     { print }
        ' <<< "$SECTION" > "$RESULT_FILE"

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

        local TEMP_SQL
        TEMP_SQL=$(mktemp /tmp/tpch_q${q}_XXXXXX.sql)
        {
            if [ "$ENGINE" = "sirius" ] && [ -n "${QUERY_CACHE_LEVEL[$q]:-}" ]; then
                printf "SET scan_cache_level = '%s';\n" "${QUERY_CACHE_LEVEL[$q]}"
            fi
            if [ "$ENGINE" = "sirius" ] && [ "${SIRIUS_NATIVE_SCAN_VERIFY:-0}" = 1 ]; then
                printf "SET sirius_log_level = 'debug';\n"
            fi
            printf ".timer on\n"
            for ((iter = 0; iter < NUM_ITERATIONS; iter++)); do
                cat "$QUERY_FILE"
                printf '\n'
            done
        } > "$TEMP_SQL"

        local OUTPUT=""
        local Q_EXIT=0
        local RUN_ENV=("$DUCKDB" "$DUCKDB_FILE" -f "$TEMP_SQL")
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

        awk -v n="$NUM_ITERATIONS" '
            /Run Time \(s\):/ { tc++; next }
            tc == (n - 1)     { print }
        ' <<< "$OUTPUT" > "$RESULT_FILE"

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

if [ "$MULTI_SESSION" = true ]; then
    run_multi_session
else
    run_single_session
fi

if [ "$ENGINE" = "sirius" ] && [ "$MULTI_SESSION" = false ] && [ -n "${OUTPUT_DIR:-}" ] && [ ${#VALID_QUERIES[@]} -gt 0 ]; then
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
