#!/usr/bin/env bash
# Run TPC-H queries scanning from native DuckDB tables (not parquet).
#
# Creates a DuckDB database from parquet files, then runs queries using both
# Sirius gpu_execution and plain DuckDB CPU to compare correctness and timing.
#
# This exercises the DuckDB table scan path (duckdb_scan_task) rather than
# the parquet scan path (parquet_scan_task).
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./test/tpch_performance/run_tpch_duckdb_native.sh [OPTIONS] <scale_factor> [query_numbers...]
#
# Options:
#   --parquet-dir <path>   Path to TPC-H parquet files (default: test_datasets/tpch_parquet_sf<SF>)
#   --iterations <N>       Number of iterations per query (default: 2, 1 cold + N-1 warm)
#   --timeout <seconds>    Kill session after N seconds (default: 600, 0 = no timeout)
#   --db-path <path>       Path to DuckDB database file (default: /tmp/tpch_sf<SF>.duckdb)
#   --skip-load            Skip database creation (reuse existing --db-path)
#
# Examples:
#   ./test/tpch_performance/run_tpch_duckdb_native.sh 10
#   ./test/tpch_performance/run_tpch_duckdb_native.sh --iterations 3 10 1 3 6
#   ./test/tpch_performance/run_tpch_duckdb_native.sh --skip-load --db-path /tmp/tpch_sf10.duckdb 10

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIRIUS_DUCKDB="$PROJECT_DIR/build/release/duckdb"
GPU_QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/gpu"
ORIG_QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/orig"

TPCH_TABLES=(customer lineitem nation orders part partsupp region supplier)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PARQUET_DIR=""
NUM_ITERATIONS=2
SESSION_TIMEOUT=600
DB_PATH=""
SKIP_LOAD=false
SCAN_CACHE_LEVEL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --parquet-dir)  PARQUET_DIR="$2"; shift 2 ;;
        --iterations)   NUM_ITERATIONS="$2"; shift 2 ;;
        --timeout)      SESSION_TIMEOUT="$2"; shift 2 ;;
        --db-path)      DB_PATH="$2"; shift 2 ;;
        --skip-load)    SKIP_LOAD=true; shift ;;
        --cache-level)  SCAN_CACHE_LEVEL="$2"; shift 2 ;;
        -*)             echo "Unknown option: $1"; exit 1 ;;
        *)              break ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "Usage: $0 [OPTIONS] <scale_factor> [query_numbers...]"
    echo "  If no query numbers given, runs all 22 TPC-H queries."
    exit 1
fi

SF="$1"; shift
if [ $# -gt 0 ]; then
    QUERIES=("$@")
else
    QUERIES=($(seq 1 22))
fi

[ -z "$PARQUET_DIR" ] && PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"

# Auto-detect existing DuckDB files in project root
if [ -z "$DB_PATH" ]; then
    for candidate in \
        "$PROJECT_DIR/tpch_${SF}.duckdb" \
        "$PROJECT_DIR/tpch_sf${SF}.duckdb" \
        "$PROJECT_DIR/performance_test_sf${SF}.duckdb"; do
        if [ -f "$candidate" ]; then
            DB_PATH="$candidate"
            SKIP_LOAD=true
            break
        fi
    done
    [ -z "$DB_PATH" ] && DB_PATH="/tmp/claude-1002/tpch_sf${SF}.duckdb"
fi

if [ ! -f "$SIRIUS_DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found: $SIRIUS_DUCKDB"
    echo "Build first: CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) make"
    exit 1
fi

# ---------------------------------------------------------------------------
# Create output directory
# ---------------------------------------------------------------------------
RUN_DIR="$PROJECT_DIR/runs/duckdb_native_$(date +%Y-%m-%d_%H-%M-%S)_sf${SF}"
mkdir -p "$RUN_DIR/sirius" "$RUN_DIR/duckdb"

echo "=========================================="
echo "TPC-H DuckDB Native Table Scan Benchmark"
echo "=========================================="
echo "Scale factor:  SF${SF}"
echo "Database:      $DB_PATH"
echo "Iterations:    $NUM_ITERATIONS (1 cold + $((NUM_ITERATIONS - 1)) warm)"
echo "Queries:       ${QUERIES[*]}"
if [ -n "$SCAN_CACHE_LEVEL" ]; then
    echo "Cache level:   $SCAN_CACHE_LEVEL"
fi
echo "Run directory: $RUN_DIR"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 1: Create DuckDB database from parquet (unless --skip-load)
# ---------------------------------------------------------------------------
if [ "$SKIP_LOAD" = true ]; then
    if [ ! -f "$DB_PATH" ]; then
        echo "ERROR: --skip-load specified but database not found: $DB_PATH"
        exit 1
    fi
    echo ""
    echo "Reusing existing database: $DB_PATH ($(du -h "$DB_PATH" | cut -f1))"
else
    if [ ! -d "$PARQUET_DIR" ]; then
        echo "ERROR: Parquet directory not found: $PARQUET_DIR"
        exit 1
    fi

    echo ""
    echo "=== Loading parquet data into DuckDB database ==="
    echo "Source: $PARQUET_DIR"

    # Remove old database if it exists
    rm -f "$DB_PATH" "${DB_PATH}.wal"

    LOAD_SQL=""
    for TABLE_NAME in "${TPCH_TABLES[@]}"; do
        FILES=()
        for f in "$PARQUET_DIR/${TABLE_NAME}.parquet" \
                 "$PARQUET_DIR/${TABLE_NAME}_"*.parquet \
                 "$PARQUET_DIR/${TABLE_NAME}/"*.parquet; do
            [ -f "$f" ] && FILES+=("'$f'")
        done
        FILE_LIST=$(IFS=,; echo "${FILES[*]}")
        LOAD_SQL+="CREATE TABLE ${TABLE_NAME} AS SELECT * FROM read_parquet([${FILE_LIST}]);"$'\n'
    done
    LOAD_SQL+="CHECKPOINT;"$'\n'

    LOAD_TEMP=$(mktemp /tmp/claude-1002/tpch_load_XXXXXX.sql)
    echo "$LOAD_SQL" > "$LOAD_TEMP"

    LOAD_START=$(date +%s.%N)
    "$SIRIUS_DUCKDB" "$DB_PATH" -f "$LOAD_TEMP" 2>&1
    LOAD_EXIT=$?
    LOAD_END=$(date +%s.%N)
    rm -f "$LOAD_TEMP"

    if [ "$LOAD_EXIT" -ne 0 ]; then
        echo "ERROR: Failed to load data into DuckDB (exit code $LOAD_EXIT)"
        exit 1
    fi

    LOAD_TIME=$(echo "$LOAD_END - $LOAD_START" | bc)
    DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
    echo "Database created: $DB_PATH ($DB_SIZE) in ${LOAD_TIME}s"
fi

echo ""
echo "Database path: $DB_PATH"

# ---------------------------------------------------------------------------
# Helper: run one engine
# ---------------------------------------------------------------------------
MARKER_PREFIX="__TPCH_MARKER__"
END_MARKER="__TPCH_END__"

run_engine() {
    local ENGINE="$1"
    local ENGINE_DIR="$RUN_DIR/$ENGINE"

    echo ""
    echo "=== Running $ENGINE ==="

    # Build SQL file
    local TEMP_SQL
    TEMP_SQL=$(mktemp /tmp/claude-1002/tpch_native_${ENGINE}_XXXXXX.sql)

    if [ "$ENGINE" = "sirius" ]; then
        # For Sirius: the extension auto-loads, tables are already in the database
        echo ".timer on" > "$TEMP_SQL"
        # Set scan cache level if specified (must be before queries, timer off so SET isn't timed)
        if [ -n "$SCAN_CACHE_LEVEL" ]; then
            printf ".timer off\nSET scan_cache_level = '%s';\n.timer on\n" "$SCAN_CACHE_LEVEL" >> "$TEMP_SQL"
        fi
    else
        # For DuckDB: unset Sirius config so extension doesn't initialize
        echo ".timer on" > "$TEMP_SQL"
    fi

    local VALID_QUERIES=()
    for q in "${QUERIES[@]}"; do
        local QUERY_FILE
        if [ "$ENGINE" = "sirius" ]; then
            QUERY_FILE="$GPU_QUERY_DIR/q${q}.sql"
        else
            QUERY_FILE="$ORIG_QUERY_DIR/q${q}.sql"
        fi

        if [ ! -f "$QUERY_FILE" ]; then
            echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}"
            continue
        fi
        VALID_QUERIES+=("$q")

        echo ".print ${MARKER_PREFIX} ${q}" >> "$TEMP_SQL"
        for ((iter = 0; iter < NUM_ITERATIONS; iter++)); do
            cat "$QUERY_FILE" >> "$TEMP_SQL"
            printf '\n' >> "$TEMP_SQL"
        done
    done
    echo ".print ${END_MARKER}" >> "$TEMP_SQL"

    cp "$TEMP_SQL" "$ENGINE_DIR/all_queries.sql"

    # Run
    local START_TIME END_TIME FULL_OUTPUT SESSION_EXIT
    START_TIME=$(date +%s.%N)

    if [ "$ENGINE" = "sirius" ]; then
        if [ "$SESSION_TIMEOUT" -gt 0 ] 2>/dev/null; then
            FULL_OUTPUT=$(timeout "$SESSION_TIMEOUT" env SIRIUS_LOG_DIR="$ENGINE_DIR" \
                "$SIRIUS_DUCKDB" "$DB_PATH" -readonly -f "$TEMP_SQL" 2>&1)
        else
            FULL_OUTPUT=$(SIRIUS_LOG_DIR="$ENGINE_DIR" \
                "$SIRIUS_DUCKDB" "$DB_PATH" -readonly -f "$TEMP_SQL" 2>&1)
        fi
    else
        # Run without Sirius config
        if [ "$SESSION_TIMEOUT" -gt 0 ] 2>/dev/null; then
            FULL_OUTPUT=$(timeout "$SESSION_TIMEOUT" env SIRIUS_CONFIG_FILE= \
                "$SIRIUS_DUCKDB" "$DB_PATH" -readonly -f "$TEMP_SQL" 2>&1)
        else
            FULL_OUTPUT=$(SIRIUS_CONFIG_FILE= \
                "$SIRIUS_DUCKDB" "$DB_PATH" -readonly -f "$TEMP_SQL" 2>&1)
        fi
    fi
    SESSION_EXIT=$?
    END_TIME=$(date +%s.%N)

    local TOTAL_ELAPSED
    TOTAL_ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    echo "  Total wall-clock: ${TOTAL_ELAPSED}s (exit=$SESSION_EXIT)"

    if [ "$SESSION_EXIT" -eq 124 ]; then
        echo "  SESSION TIMEOUT after ${SESSION_TIMEOUT}s"
    elif [ "$SESSION_EXIT" -ne 0 ]; then
        echo "  SESSION FAILED (exit code $SESSION_EXIT)"
        if echo "$FULL_OUTPUT" | grep -q "Requested number of GPUs exceeds available"; then
            echo "  ERROR: No GPU available. Sirius requires a GPU to run."
            echo "  Skipping $ENGINE results."
            echo "$FULL_OUTPUT" > "$ENGINE_DIR/full_output.txt"
            rm -f "$TEMP_SQL"
            return
        fi
    fi

    rm -f "$TEMP_SQL"

    # Save full output for debugging
    echo "$FULL_OUTPUT" > "$ENGINE_DIR/full_output.txt"

    # Parse per-query results and timings
    local TEMP_OUTPUT
    TEMP_OUTPUT=$(mktemp /tmp/claude-1002/tpch_output_${ENGINE}_XXXXXX.txt)
    echo "$FULL_OUTPUT" > "$TEMP_OUTPUT"

    for q in "${VALID_QUERIES[@]}"; do
        local Q_DIR="$ENGINE_DIR/q${q}"
        mkdir -p "$Q_DIR"

        local SECTION
        SECTION=$(awk -v start="${MARKER_PREFIX} ${q}" \
                      -v prefix="${MARKER_PREFIX}" \
                      -v end="${END_MARKER}" '
            $0 == start                                   { cap = 1; next }
            cap && ($0 == end || index($0, prefix) == 1)  { exit }
            cap                                           { print }
        ' "$TEMP_OUTPUT")

        if [ -z "$SECTION" ]; then
            echo "  Q${q}: NO OUTPUT"
            echo "no output" > "$Q_DIR/result.txt"
            {
                echo "step,runtime_s"
                for ((i = 0; i < NUM_ITERATIONS; i++)); do
                    echo "iter_$((i + 1)),N/A"
                done
            } > "$Q_DIR/timings.csv"
            continue
        fi

        # Save last-iteration result
        awk -v n="$NUM_ITERATIONS" '
            /Run Time \(s\):/ { tc++; next }
            tc == (n - 1)     { print }
        ' <<< "$SECTION" > "$Q_DIR/result.txt"

        # Extract timings
        readarray -t TIMES < <(grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+' <<< "$SECTION")

        {
            echo "step,runtime_s"
            for ((i = 0; i < ${#TIMES[@]}; i++)); do
                echo "iter_$((i + 1)),${TIMES[$i]}"
            done
        } > "$Q_DIR/timings.csv"

        local cold="${TIMES[0]:-N/A}"
        local warm="N/A"
        for ((i = 1; i < ${#TIMES[@]}; i++)); do
            if [ "$warm" = "N/A" ] || (( $(echo "${TIMES[$i]} < $warm" | bc -l) )); then
                warm="${TIMES[$i]}"
            fi
        done
        echo "  Q${q}: Cold=${cold}s  Warm(best)=${warm}s  (${#TIMES[@]} iters)"
    done

    rm -f "$TEMP_OUTPUT"
}

# ---------------------------------------------------------------------------
# Step 2: Run both engines
# ---------------------------------------------------------------------------
run_engine "duckdb"
run_engine "sirius"

# ---------------------------------------------------------------------------
# Step 3: Validate results
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "=== Result Validation ==="
echo "=========================================="

VALIDATION_CSV="$RUN_DIR/validation.csv"
printf 'query,status\n' > "$VALIDATION_CSV"

ok=0; mismatch=0; errors=0

has_error() {
    local file="$1"
    [[ ! -f "$file" ]] && return 0
    [[ ! -s "$file" ]] && return 0
    grep -qE "(Error|Segmentation fault|no output)" "$file" 2>/dev/null
}

for q in "${QUERIES[@]}"; do
    SIRIUS_FILE="$RUN_DIR/sirius/q${q}/result.txt"
    DUCKDB_FILE="$RUN_DIR/duckdb/q${q}/result.txt"

    if has_error "$SIRIUS_FILE"; then
        status="error"
        (( errors++ ))
    elif diff -q "$SIRIUS_FILE" "$DUCKDB_FILE" >/dev/null 2>&1; then
        status="match"
        (( ok++ ))
    else
        status="mismatch"
        (( mismatch++ ))
    fi

    printf 'Q%-2s  %s\n' "$q" "$status"
    printf 'Q%s,%s\n' "$q" "$status" >> "$VALIDATION_CSV"
done

echo ""
printf 'Summary: %d/%d match   %d mismatch   %d error\n' \
    "$ok" "${#QUERIES[@]}" "$mismatch" "$errors"

# ---------------------------------------------------------------------------
# Step 4: Timing comparison table
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
printf "  Timing Comparison  (SF%s, DuckDB Native Table Scan)\n" "$SF"
echo "============================================================"
echo ""

declare -A DC DW SC SW

for q in "${QUERIES[@]}"; do
    DUCKDB_TIMING="$RUN_DIR/duckdb/q${q}/timings.csv"
    SIRIUS_TIMING="$RUN_DIR/sirius/q${q}/timings.csv"

    if [ -f "$DUCKDB_TIMING" ]; then
        DC[$q]=$(awk -F',' '$1=="iter_1"{print $2}' "$DUCKDB_TIMING")
        DW[$q]=$(awk -F',' '$1~/^iter_/ && $1!="iter_1"{v=$2+0; if(min==""||v<min)min=v}END{if(min!="")print min}' "$DUCKDB_TIMING")
    fi
    if [ -f "$SIRIUS_TIMING" ]; then
        SC[$q]=$(awk -F',' '$1=="iter_1"{print $2}' "$SIRIUS_TIMING")
        SW[$q]=$(awk -F',' '$1~/^iter_/ && $1!="iter_1"{v=$2+0; if(min==""||v<min)min=v}END{if(min!="")print min}' "$SIRIUS_TIMING")
    fi
done

printf "%-7s | %13s | %13s | %13s | %13s | %14s | %s\n" \
    "Query" "DuckDB Cold" "DuckDB Warm" "Sirius Cold" "Sirius Warm" "Speedup (warm)" "Valid"
printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s-+-%s\n" \
    "-------" "-------------" "-------------" "-------------" "-------------" "--------------" "--------"

TOTAL_DC=0; TOTAL_DW=0; TOTAL_SC=0; TOTAL_SW=0

for q in "${QUERIES[@]}"; do
    dc="${DC[$q]:-N/A}"; dw="${DW[$q]:-N/A}"
    sc="${SC[$q]:-N/A}"; sw="${SW[$q]:-N/A}"

    speedup="N/A"
    if [ "$dw" != "N/A" ] && [ "$sw" != "N/A" ]; then
        speedup=$(echo "scale=2; $dw / $sw" | bc 2>/dev/null || echo "N/A")
        [ "$speedup" != "N/A" ] && speedup="${speedup}x"
    fi

    fmt_dc="N/A"; fmt_dw="N/A"; fmt_sc="N/A"; fmt_sw="N/A"
    [ "$dc" != "N/A" ] && fmt_dc=$(printf "%.3fs" "$dc")
    [ "$dw" != "N/A" ] && fmt_dw=$(printf "%.3fs" "$dw")
    [ "$sc" != "N/A" ] && fmt_sc=$(printf "%.3fs" "$sc")
    [ "$sw" != "N/A" ] && fmt_sw=$(printf "%.3fs" "$sw")

    # Validation status
    vstatus=$(awk -F',' -v q="Q${q}" '$1==q{print $2}' "$VALIDATION_CSV")

    printf "%-7s | %13s | %13s | %13s | %13s | %14s | %s\n" \
        "Q${q}" "$fmt_dc" "$fmt_dw" "$fmt_sc" "$fmt_sw" "$speedup" "$vstatus"

    [ "$dc" != "N/A" ] && TOTAL_DC=$(echo "$TOTAL_DC + $dc" | bc)
    [ "$dw" != "N/A" ] && TOTAL_DW=$(echo "$TOTAL_DW + $dw" | bc)
    [ "$sc" != "N/A" ] && TOTAL_SC=$(echo "$TOTAL_SC + $sc" | bc)
    [ "$sw" != "N/A" ] && TOTAL_SW=$(echo "$TOTAL_SW + $sw" | bc)
done

total_speedup="N/A"
if [ "$(echo "$TOTAL_SW > 0" | bc)" -eq 1 ]; then
    total_speedup=$(echo "scale=2; $TOTAL_DW / $TOTAL_SW" | bc 2>/dev/null || echo "N/A")
    [ "$total_speedup" != "N/A" ] && total_speedup="${total_speedup}x"
fi

printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s-+-%s\n" \
    "-------" "-------------" "-------------" "-------------" "-------------" "--------------" "--------"
printf "%-7s | %13s | %13s | %13s | %13s | %14s |\n" \
    "TOTAL" "$(printf '%.3fs' "$TOTAL_DC")" "$(printf '%.3fs' "$TOTAL_DW")" \
    "$(printf '%.3fs' "$TOTAL_SC")" "$(printf '%.3fs' "$TOTAL_SW")" "$total_speedup"

echo ""
echo "============================================================"
echo "All output saved to $RUN_DIR"
echo "Database: $DB_PATH"
echo "============================================================"
