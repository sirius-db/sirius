#!/usr/bin/env bash
# benchmark_and_validate.sh
#
# Runs TPC-DS queries for both sirius (Super Sirius) and duckdb, compares
# results, and writes:
#   validation.csv  - per-query match/error status
#   comparison.txt  - results summary table (cold/warm timings, speedup)
#   timings.csv     - long-format runtimes (engine,query,iteration,runtime_s)
#
# Each run gets its own timestamped directory under runs/:
#   runs/tpcds_<timestamp>_sf<SF>/
#     run_info.txt     - git, hardware, build info
#     run_info.patch   - git diff when tree is dirty
#     sirius_config.cfg
#     sirius/          - result_q<N>.txt, log_q<N>.txt, timings.csv
#     duckdb/          - result_q<N>.txt, log_q<N>.txt, timings.csv
#     validation.csv
#     comparison.txt
#     timings.csv
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./test/tpcds_performance/benchmark_and_validate.sh <scale_factor>
#   ./test/tpcds_performance/benchmark_and_validate.sh --gpu-only <scale_factor>
#   ./test/tpcds_performance/benchmark_and_validate.sh --queries 3 7 42 -- <scale_factor>
#   ./test/tpcds_performance/benchmark_and_validate.sh --report <run_dir>
#   ./test/tpcds_performance/benchmark_and_validate.sh --duckdb-results <run_dir> <scale_factor>

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUPER_SCRIPT="$SCRIPT_DIR/run_tpcds_super.sh"
DUCKDB_SCRIPT="$SCRIPT_DIR/run_tpcds_duckdb.sh"

# TPC-DS queries confirmed to execute completely on Super Sirius with correct results.
# This list grows as more queries gain full GPU support.
GPU_QUERIES=(3 7 22 26 32 37 42 52 55 62 82 85 92 93 97)

# ---------------------------------------------------------------------------
# Report generation from an existing run directory.
# ---------------------------------------------------------------------------
generate_report() {
    local RUN_DIR="$1"
    local QUERIES=()

    # Discover which queries are present by scanning sirius/ and duckdb/ result files.
    for f in "$RUN_DIR"/sirius/result_q*.txt "$RUN_DIR"/duckdb/result_q*.txt; do
        [ -f "$f" ] || continue
        local base
        base=$(basename "$f")
        # Extract query number from result_q<N>.txt
        local qnum="${base#result_q}"
        qnum="${qnum%.txt}"
        QUERIES+=("$qnum")
    done
    # Deduplicate and sort numerically.
    readarray -t QUERIES < <(printf '%s\n' "${QUERIES[@]}" | sort -un)

    if [ ${#QUERIES[@]} -eq 0 ]; then
        echo "ERROR: no query result files found in $RUN_DIR"
        return 1
    fi

    # Try to extract SF from the directory name (e.g. tpcds_..._sf10).
    local SF="?"
    local dir_base
    dir_base=$(basename "$RUN_DIR")
    if [[ "$dir_base" =~ _sf([0-9]+) ]]; then
        SF="${BASH_REMATCH[1]}"
    fi

    local VALIDATION_CSV="$RUN_DIR/validation.csv"
    local TIMINGS_CSV="$RUN_DIR/timings.csv"

    # ---------- Validation ----------
    echo ""
    echo "=== Comparing results ==="
    echo "=========================================="

    has_error() {
        local file="$1"
        [[ ! -f "$file" ]] && return 0
        [[ ! -s "$file" ]] && return 0
        grep -qE "(Error|Segmentation fault)" "$file" 2>/dev/null
    }

    printf 'query,status\n' | tee "$VALIDATION_CSV"

    local ok=0 validate=0 errors=0

    for q in "${QUERIES[@]}"; do
        local SIRIUS_FILE="$RUN_DIR/sirius/result_q${q}.txt"
        local DUCKDB_FILE="$RUN_DIR/duckdb/result_q${q}.txt"
        local status
        if has_error "$SIRIUS_FILE"; then
            status="error"
            (( errors++ ))
        elif [ ! -f "$DUCKDB_FILE" ]; then
            status="error"
            (( errors++ ))
        elif diff -q "$SIRIUS_FILE" "$DUCKDB_FILE" >/dev/null 2>&1; then
            status="success"
            (( ok++ ))
        else
            status="validation"
            (( validate++ ))
        fi

        printf 'Q%s,%s\n' "$q" "$status" | tee -a "$VALIDATION_CSV"
    done

    echo ""
    echo "=========================================="
    printf 'Summary: %d/%d success   %d validation   %d error\n' \
        "$ok" "${#QUERIES[@]}" "$validate" "$errors"
    echo "Validation CSV saved to $VALIDATION_CSV"

    # ---------- Combined timings CSV ----------
    echo ""
    echo "=== Building combined timings CSV ==="

    printf 'engine,query,iteration,runtime_s\n' > "$TIMINGS_CSV"

    for engine in sirius duckdb; do
        local ENGINE_TIMINGS="$RUN_DIR/$engine/timings.csv"
        [[ ! -f "$ENGINE_TIMINGS" ]] && continue

        # TPC-DS timings.csv format: query,run1_time,run1_status,run2_time,run2_status
        awk -F',' -v engine="$engine" '
            NR == 1 { next }
            {
                query = "Q" $1
                if ($2 != "-1" && $3 != "SKIP" && $3 != "EMPTY") {
                    printf "%s,%s,1,%s\n", engine, query, $2
                }
                if ($4 != "-1" && $5 != "SKIP" && $5 != "EMPTY") {
                    printf "%s,%s,2,%s\n", engine, query, $4
                }
            }
        ' "$ENGINE_TIMINGS" >> "$TIMINGS_CSV"
    done

    echo "Timings CSV saved to $TIMINGS_CSV"

    # ---------- Comparison table ----------
    {
    echo ""
    echo "============================================================"
    printf "  TPC-DS Results Summary  (SF%s)\n" "$SF"
    echo "============================================================"
    echo ""

    declare -A DC DW SC SW

    for q in "${QUERIES[@]}"; do
        local ENGINE_TIMINGS

        # Parse DuckDB timings (wide format: query,run1_time,run1_status,run2_time,run2_status)
        ENGINE_TIMINGS="$RUN_DIR/duckdb/timings.csv"
        if [ -f "$ENGINE_TIMINGS" ]; then
            DC[$q]=$(awk -F',' -v q="$q" '$1==q && $3=="OK"{print $2}' "$ENGINE_TIMINGS")
            DW[$q]=$(awk -F',' -v q="$q" '$1==q && $5=="OK"{print $4}' "$ENGINE_TIMINGS")
        fi

        # Parse Sirius timings
        ENGINE_TIMINGS="$RUN_DIR/sirius/timings.csv"
        if [ -f "$ENGINE_TIMINGS" ]; then
            SC[$q]=$(awk -F',' -v q="$q" '$1==q && $3=="OK"{print $2}' "$ENGINE_TIMINGS")
            SW[$q]=$(awk -F',' -v q="$q" '$1==q && $5=="OK"{print $4}' "$ENGINE_TIMINGS")
        fi
    done

    printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
        "Query" "DuckDB Cold" "DuckDB Warm" "Sirius Cold" "Sirius Warm" "Speedup (warm)"
    printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s\n" \
        "-------" "-------------" "-------------" "-------------" "-------------" "--------------"

    local TOTAL_DC=0 TOTAL_DW=0 TOTAL_SC=0 TOTAL_SW=0

    for q in "${QUERIES[@]}"; do
        local dc="${DC[$q]:-N/A}" dw="${DW[$q]:-N/A}"
        local sc="${SC[$q]:-N/A}" sw="${SW[$q]:-N/A}"

        local speedup="N/A"
        if [ "$dw" != "N/A" ] && [ -n "$dw" ] && [ "$sw" != "N/A" ] && [ -n "$sw" ]; then
            speedup=$(echo "scale=2; $dw / $sw" | bc 2>/dev/null || echo "N/A")
            [ "$speedup" != "N/A" ] && speedup="${speedup}x"
        fi

        local fmt_dc="N/A" fmt_dw="N/A" fmt_sc="N/A" fmt_sw="N/A"
        [ "$dc" != "N/A" ] && [ -n "$dc" ] && fmt_dc=$(printf "%.2fs" "$dc")
        [ "$dw" != "N/A" ] && [ -n "$dw" ] && fmt_dw=$(printf "%.2fs" "$dw")
        [ "$sc" != "N/A" ] && [ -n "$sc" ] && fmt_sc=$(printf "%.2fs" "$sc")
        [ "$sw" != "N/A" ] && [ -n "$sw" ] && fmt_sw=$(printf "%.2fs" "$sw")

        printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
            "Q${q}" "$fmt_dc" "$fmt_dw" "$fmt_sc" "$fmt_sw" "$speedup"

        [ "$dc" != "N/A" ] && [ -n "$dc" ] && TOTAL_DC=$(echo "$TOTAL_DC + $dc" | bc)
        [ "$dw" != "N/A" ] && [ -n "$dw" ] && TOTAL_DW=$(echo "$TOTAL_DW + $dw" | bc)
        [ "$sc" != "N/A" ] && [ -n "$sc" ] && TOTAL_SC=$(echo "$TOTAL_SC + $sc" | bc)
        [ "$sw" != "N/A" ] && [ -n "$sw" ] && TOTAL_SW=$(echo "$TOTAL_SW + $sw" | bc)
    done

    local total_speedup="N/A"
    if [ "$(echo "$TOTAL_SW > 0" | bc)" -eq 1 ]; then
        total_speedup=$(echo "scale=2; $TOTAL_DW / $TOTAL_SW" | bc 2>/dev/null || echo "N/A")
        [ "$total_speedup" != "N/A" ] && total_speedup="${total_speedup}x"
    fi

    printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s\n" \
        "-------" "-------------" "-------------" "-------------" "-------------" "--------------"
    printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
        "TOTAL" "$(printf '%.2fs' "$TOTAL_DC")" "$(printf '%.2fs' "$TOTAL_DW")" \
        "$(printf '%.2fs' "$TOTAL_SC")" "$(printf '%.2fs' "$TOTAL_SW")" "$total_speedup"
    echo ""
    echo "============================================================"
    echo "All output saved to $RUN_DIR"
    } | tee "$RUN_DIR/comparison.txt"
}

# ---------------------------------------------------------------------------
# --report mode: regenerate comparison/timings from an existing run directory
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--report" ]; then
    if [ $# -ne 2 ]; then
        echo "Usage: $0 --report <run_dir>"
        exit 1
    fi
    RUN_DIR="$2"
    if [ ! -d "$RUN_DIR" ] && [ -d "$PROJECT_DIR/runs/$RUN_DIR" ]; then
        RUN_DIR="$PROJECT_DIR/runs/$RUN_DIR"
    fi
    if [ ! -d "$RUN_DIR" ]; then
        echo "ERROR: run directory not found: $RUN_DIR"
        exit 1
    fi
    generate_report "$RUN_DIR"
    exit 0
fi

# ---------------------------------------------------------------------------
# Normal benchmark mode
# ---------------------------------------------------------------------------

# Parse optional flags
DUCKDB_RESULTS_DIR=""
PARQUET_DIR=""
ENGINES=""
USER_QUERIES=()
GPU_ONLY=false

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            export SIRIUS_CONFIG_FILE="$2"
            shift 2
            ;;
        --parquet-dir)
            PARQUET_DIR="$2"
            shift 2
            ;;
        --engines)
            ENGINES="$2"
            shift 2
            ;;
        --gpu-only)
            GPU_ONLY=true
            shift
            ;;
        --queries)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]] && [[ "$1" != [^0-9]* || "$1" =~ ^[0-9]+$ ]]; do
                USER_QUERIES+=("$1")
                shift
            done
            ;;
        --duckdb-results)
            DUCKDB_RESULTS_DIR="$2"
            shift 2
            ;;
        --note)
            export RUN_NOTE="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [ $# -ne 1 ]; then
    echo "Usage: $0 [options] <scale_factor>"
    echo "       $0 --report <run_dir>"
    echo ""
    echo "Options:"
    echo "  --config <file>           Override Sirius config file"
    echo "  --parquet-dir <path>      Custom parquet data location"
    echo "  --engines 'sirius duckdb' Space-separated engine list (default: 'sirius duckdb')"
    echo "  --gpu-only                Run only the GPU-compatible query subset"
    echo "  --queries <N...> --       Specific query numbers (use -- before scale_factor)"
    echo "  --duckdb-results <dir>    Reuse previously stored DuckDB results"
    echo "  --note <text>             Free-form annotation prepended to telemetry labels"
    echo ""
    echo "Examples:"
    echo "  $0 1"
    echo "  $0 --gpu-only 10"
    echo "  $0 --queries 3 7 42 -- 1"
    echo "  $0 --duckdb-results runs/tpcds_2026-04-05_sf1 10"
    exit 1
fi

SF="$1"

# Determine query set
if [ ${#USER_QUERIES[@]} -gt 0 ]; then
    QUERIES=("${USER_QUERIES[@]}")
elif [ "$GPU_ONLY" = true ]; then
    QUERIES=("${GPU_QUERIES[@]}")
else
    QUERIES=($(seq 1 99))
fi

ENGINES="${ENGINES:-sirius duckdb}"

RUN_DIR="$PROJECT_DIR/runs/tpcds_$(date +%Y-%m-%d_%H-%M-%S)_sf${SF}"
mkdir -p "$RUN_DIR"

# Resolve config — only required for sirius engine.
if [[ " $ENGINES " == *" sirius "* ]]; then
    if [ -z "${SIRIUS_CONFIG_FILE:-}" ]; then
        export SIRIUS_CONFIG_FILE="$HOME/.sirius/sirius.cfg"
    fi
    if [ ! -f "$SIRIUS_CONFIG_FILE" ]; then
        echo "ERROR: config file not found: $SIRIUS_CONFIG_FILE"
        exit 1
    fi
    echo "Config file: $SIRIUS_CONFIG_FILE"
    cp "$SIRIUS_CONFIG_FILE" "$RUN_DIR/sirius_config.cfg"
fi

# Resolve parquet directory
PARQUET_DIR="${PARQUET_DIR:-$PROJECT_DIR/test_datasets/tpcds_parquet_sf${SF}}"

# ---------------------------------------------------------------------------
# --duckdb-results: reuse previously stored DuckDB results for validation.
# ---------------------------------------------------------------------------
if [ -n "$DUCKDB_RESULTS_DIR" ]; then
    if [ ! -d "$DUCKDB_RESULTS_DIR" ] && [ -d "$PROJECT_DIR/runs/$DUCKDB_RESULTS_DIR" ]; then
        DUCKDB_RESULTS_DIR="$PROJECT_DIR/runs/$DUCKDB_RESULTS_DIR"
    fi
    if [ -d "$DUCKDB_RESULTS_DIR/duckdb" ]; then
        DUCKDB_RESULTS_DIR="$DUCKDB_RESULTS_DIR/duckdb"
    fi
    if [ ! -d "$DUCKDB_RESULTS_DIR" ]; then
        echo "ERROR: DuckDB results directory not found: $DUCKDB_RESULTS_DIR"
        exit 1
    fi

    DUCKDB_RESULT_COUNT=0
    for f in "$DUCKDB_RESULTS_DIR"/result_q*.txt; do
        [ -f "$f" ] && (( DUCKDB_RESULT_COUNT++ ))
    done
    if [ "$DUCKDB_RESULT_COUNT" -eq 0 ]; then
        echo "ERROR: no query results (result_q*.txt) found in $DUCKDB_RESULTS_DIR"
        exit 1
    fi

    echo "Using stored DuckDB results from: $DUCKDB_RESULTS_DIR ($DUCKDB_RESULT_COUNT queries)"
    cp -a "$DUCKDB_RESULTS_DIR" "$RUN_DIR/duckdb"

    ENGINES=$(echo "$ENGINES" | sed 's/\bduckdb\b//g' | xargs)
fi

RUN_INFO_FILE="$RUN_DIR/run_info.txt"

QUERY_MODE="all 99"
if [ "$GPU_ONLY" = true ]; then
    QUERY_MODE="GPU-compatible (${#GPU_QUERIES[@]})"
elif [ ${#USER_QUERIES[@]} -gt 0 ]; then
    QUERY_MODE="custom (${#USER_QUERIES[@]})"
fi

echo "=========================================="
echo "TPC-DS Benchmark"
echo "=========================================="
echo "Scale factor: SF${SF}"
echo "Parquet dir:  $PARQUET_DIR"
echo "Queries:      $QUERY_MODE — ${QUERIES[*]}"
echo "Engines:      $ENGINES"
echo "Run dir:      $RUN_DIR"
if [ -n "${DUCKDB_RESULTS_DIR:-}" ]; then
    echo "DuckDB:       reusing from $DUCKDB_RESULTS_DIR"
fi
echo "=========================================="
echo ""

# ---------- Run info and environment ----------
echo "=== Collecting run info ==="
{
    echo "Run info — $(date -Iseconds)"
    echo "================================"
    echo ""

    echo "--- Benchmark ---"
    echo "benchmark: tpcds"
    echo "suite: TPC-DS"
    echo "scale_factor: $SF"
    echo "query_mode: $QUERY_MODE"
    echo "queries: ${QUERIES[*]}"
    echo "note: ${RUN_NOTE:-}"
    echo ""

    if [ -n "${DUCKDB_RESULTS_DIR:-}" ]; then
        echo "--- DuckDB results ---"
        echo "source: $DUCKDB_RESULTS_DIR (copied, not re-run)"
        echo ""
    fi

    echo "--- Git ---"
    if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
        echo "branch: $(git -C "$PROJECT_DIR" branch --show-current)"
        echo "revision: $(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
        if git -C "$PROJECT_DIR" diff --quiet 2>/dev/null && git -C "$PROJECT_DIR" diff --cached --quiet 2>/dev/null; then
            echo "tree: clean"
        else
            echo "tree: dirty (uncommitted changes, see run_info.patch)"
            {
                echo "=== git diff ==="
                git -C "$PROJECT_DIR" diff
                echo ""
                echo "=== git diff --cached ==="
                git -C "$PROJECT_DIR" diff --cached
            } > "$RUN_DIR/run_info.patch"
        fi
    else
        echo "not a git repository"
    fi
    echo ""

    echo "--- Build ---"
    DUCKDB_BIN="$PROJECT_DIR/build/release/duckdb"
    if [ -f "$DUCKDB_BIN" ]; then
        echo "duckdb binary: $DUCKDB_BIN"
        echo "duckdb mtime:  $(stat -c %y "$DUCKDB_BIN" 2>/dev/null || stat -f '%Sm' "$DUCKDB_BIN" 2>/dev/null)"
    else
        echo "duckdb binary: not found ($DUCKDB_BIN)"
    fi
    echo ""

    echo "--- Hardware ---"
    echo "hostname: $(hostname)"
    echo "memory:"
    sed -n 's/^MemTotal:/  MemTotal: /p; s/^MemAvailable:/  MemAvailable: /p' /proc/meminfo 2>/dev/null || true
    echo "num_cpus: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo '?')"
    echo ""
    echo "GPUs:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null | while read -r line; do echo "  $line"; done || nvidia-smi
    else
        echo "  nvidia-smi not available"
    fi
} | tee "$RUN_INFO_FILE"

echo ""
echo "Run info saved to $RUN_INFO_FILE"
echo "=========================================="

# ---------- Run benchmarks ----------
for engine in $ENGINES; do
    ENGINE_DIR="$RUN_DIR/$engine"
    mkdir -p "$ENGINE_DIR"

    echo ""
    echo "=== Running $engine ==="

    if [ "$engine" = "sirius" ]; then
        bash "$SUPER_SCRIPT" "$PARQUET_DIR" \
            --queries "${QUERIES[@]}" \
            --output-dir "$ENGINE_DIR" \
            2>&1 | tee "$ENGINE_DIR/run.log" || true
    elif [ "$engine" = "duckdb" ]; then
        bash "$DUCKDB_SCRIPT" --parquet-dir "$PARQUET_DIR" \
            --queries "${QUERIES[@]}" \
            --output-dir "$ENGINE_DIR" \
            2>&1 | tee "$ENGINE_DIR/run.log" || true
    fi
done

generate_report "$RUN_DIR"
