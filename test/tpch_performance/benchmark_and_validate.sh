#!/usr/bin/env bash
# benchmark_and_validate.sh
#
# Runs all 22 TPC-H queries for both sirius and duckdb, compares results,
# and writes two CSVs:
#   validation.csv  - per-query match/error status
#   comparison.txt  - results summary table (cold/warm timings, speedup)
#   timings.csv     - long-format iteration runtimes (engine,query,iteration,runtime_s)
#
# Each run gets its own timestamped directory under runs/:
#   runs/<timestamp>_sf<SF>_<N>iter/
#     run_info.txt    - git branch/revision, tree clean/dirty, build freshness,
#                       hostname, memory, CPUs, load, GPUs/free memory, fs read benchmark
#     run_info.patch  - when tree is dirty, full git diff and diff --cached
#     sirius_config.yaml - copy of SIRIUS_CONFIG_FILE
#     sirius/run.log  sirius/q<N>/result.txt  sirius/q<N>/timings.csv
#     duckdb/run.log  duckdb/q<N>/result.txt  duckdb/q<N>/timings.csv
#     validation.csv
#     comparison.txt
#     timings.csv
#
# Before running benchmarks, a tiny read-only filesystem benchmark is run on the
# input location (parquet directory or DuckDB database file) and recorded in run_info.txt.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./test/tpch_performance/benchmark_and_validate.sh <scale_factor>
#   ./test/tpch_performance/benchmark_and_validate.sh --pinning-mode pinned-hot <scale_factor>
#   ./test/tpch_performance/benchmark_and_validate.sh --data-source duckdb <scale_factor>
#   ./test/tpch_performance/benchmark_and_validate.sh --data-source duckdb-native <scale_factor>
#   ./test/tpch_performance/benchmark_and_validate.sh --report <run_dir>
#   ./test/tpch_performance/benchmark_and_validate.sh --duckdb-results <run_dir> <scale_factor>
#
# Example:
#   ./test/tpch_performance/benchmark_and_validate.sh 1
#   ./test/tpch_performance/benchmark_and_validate.sh --pinning-mode pinned-hot --iterations 5 1
#   ./test/tpch_performance/benchmark_and_validate.sh --data-source duckdb --duckdb-file ./performance_test.duckdb 1
#   ./test/tpch_performance/benchmark_and_validate.sh --data-source duckdb-native --duckdb-file ./test_datasets/tpch_sf1.duckdb 1
#   ./test/tpch_performance/benchmark_and_validate.sh --report runs/2026-03-10_12-00-00_sf1_2iter
#   ./test/tpch_performance/benchmark_and_validate.sh --duckdb-results runs/2026-03-10_12-00-00_sf1_2iter 1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Set by --data-source (default: parquet → run_tpch_parquet.sh).
DATA_SOURCE="parquet"
RUN_SCRIPT="$SCRIPT_DIR/run_tpch_parquet.sh"

# True for any data source backed by a .duckdb file; duckdb and duckdb-native share the
# same file, runner, and file-resolution machinery.
is_duckdb_source() { [ "$DATA_SOURCE" = "duckdb" ] || [ "$DATA_SOURCE" = "duckdb-native" ]; }

# ---------------------------------------------------------------------------
# Report generation from an existing run directory.
# Shared by both --report mode and the end of a normal benchmark run.
# ---------------------------------------------------------------------------
generate_report() {
    local RUN_DIR="$1"
    local FLOAT_TOL="${2:-1e-10}"
    local QUERIES=()

    # Discover which queries are present by scanning sirius/ and duckdb/ subdirs.
    for d in "$RUN_DIR"/sirius/q* "$RUN_DIR"/duckdb/q*; do
        [ -d "$d" ] || continue
        local qnum="${d##*/q}"
        QUERIES+=("$qnum")
    done

    if [ ${#QUERIES[@]} -eq 0 ]; then
        echo "ERROR: no query directories found in $RUN_DIR"
        return 1
    fi

    # Deduplicate and sort numerically.
    readarray -t QUERIES < <(printf '%s\n' "${QUERIES[@]}" | sort -un)

    # Try to extract SF from the directory name (e.g. ..._sf10_2iter).
    local SF="?"
    local dir_base
    dir_base=$(basename "$RUN_DIR")
    if [[ "$dir_base" =~ _sf([^_]+)_ ]]; then
        SF="${BASH_REMATCH[1]}"
    fi

    local VALIDATION_CSV="$RUN_DIR/validation.csv"
    local TIMINGS_CSV="$RUN_DIR/timings.csv"

    # ---------- Validation ----------
    echo ""
    echo "=== Comparing results (float tolerance: $FLOAT_TOL) ==="
    echo "=========================================="

    has_error() {
        local file="$1"
        [[ ! -f "$file" ]] && return 0
        [[ ! -s "$file" ]] && return 0
        grep -qiE "(^error:|^no output|Invalid Error|IO Error|Catalog Error|Parser Error|Binder Error|Segmentation fault|^Error:)" "$file" 2>/dev/null
    }

    has_valid_timings() {
        local file="$1"
        [[ -f "$file" ]] || return 1
        awk -F',' '
            NR == 1 { next }
            $2 != "" && $2 != "N/A" { found = 1 }
            END { exit(found ? 0 : 1) }
        ' "$file"
    }

    printf 'query,status\n' | tee "$VALIDATION_CSV"

    local ok=0 validate=0 errors=0

    for q in "${QUERIES[@]}"; do
        local SIRIUS_FILE="$RUN_DIR/sirius/q${q}/result.txt"
        local DUCKDB_FILE="$RUN_DIR/duckdb/q${q}/result.txt"
        local SIRIUS_TIMING="$RUN_DIR/sirius/q${q}/timings.csv"
        local DUCKDB_TIMING="$RUN_DIR/duckdb/q${q}/timings.csv"
        local status
        if has_error "$SIRIUS_FILE" || has_error "$DUCKDB_FILE" || \
            ! has_valid_timings "$SIRIUS_TIMING" || ! has_valid_timings "$DUCKDB_TIMING"; then
            status="error"
            (( errors++ ))
        elif diff -q "$SIRIUS_FILE" "$DUCKDB_FILE" >/dev/null 2>&1; then
            status="success"
            (( ok++ ))
        else
            # Files differ byte-exact — fall back to tolerance-aware comparator
            # so float/double columns within $FLOAT_TOL absolute diff still pass.
            local cmp_msg
            if cmp_msg=$(python3 "$SCRIPT_DIR/compare_results.py" \
                    --float-tolerance "$FLOAT_TOL" \
                    "$DUCKDB_FILE" "$SIRIUS_FILE" 2>&1); then
                status="success"
                (( ok++ ))
                printf '  Q%s: success (within tolerance %s)\n' "$q" "$FLOAT_TOL"
            else
                status="validation"
                (( validate++ ))
                if [ -n "$cmp_msg" ]; then
                    printf '  Q%s: validation — %s\n' "$q" "$cmp_msg"
                fi
            fi
        fi

        printf 'Q%s,%s\n' "$q" "$status" | tee -a "$VALIDATION_CSV"
    done

    echo ""
    echo "=========================================="
    printf 'Summary: %d/%d success   %d validate   %d error\n' \
        "$ok" "${#QUERIES[@]}" "$validate" "$errors"
    echo "Validation CSV saved to $VALIDATION_CSV"

    # ---------- Combined timings CSV ----------
    echo ""
    echo "=== Building combined timings CSV ==="

    printf 'engine,query,iteration,runtime_s\n' > "$TIMINGS_CSV"

    for engine in sirius duckdb; do
        for q in "${QUERIES[@]}"; do
            local TIMING_FILE="$RUN_DIR/$engine/q${q}/timings.csv"
            [[ ! -f "$TIMING_FILE" ]] && continue

            awk -F',' -v engine="$engine" -v query="Q${q}" '
                NR == 1 { next }
                $1 ~ /^iter_/ {
                    iter = substr($1, 6)
                    printf "%s,%s,%s,%s\n", engine, query, iter, $2
                }
            ' "$TIMING_FILE" >> "$TIMINGS_CSV"
        done
    done

    echo "Timings CSV saved to $TIMINGS_CSV"

    # ---------- Comparison table ----------
    {
    echo ""
    echo "============================================================"
    printf "  Results Summary  (SF%s)\n" "$SF"
    echo "============================================================"
    echo ""

        declare -A DC DW SC SW

        for q in "${QUERIES[@]}"; do
            local DUCKDB_TIMING="$RUN_DIR/duckdb/q${q}/timings.csv"
            local SIRIUS_TIMING="$RUN_DIR/sirius/q${q}/timings.csv"

            if [ -f "$DUCKDB_TIMING" ]; then
                DC[$q]=$(awk -F',' '$1=="iter_1" && $2 != "N/A"{print $2; exit}' "$DUCKDB_TIMING")
                DW[$q]=$(awk -F',' '$1~/^iter_/ && $1!="iter_1" && $2 != "N/A"{v=$2+0; if(min==""||v<min)min=v}END{if(min!="")print min}' "$DUCKDB_TIMING")
            fi
            if [ -f "$SIRIUS_TIMING" ]; then
                SC[$q]=$(awk -F',' '$1=="iter_1" && $2 != "N/A"{print $2; exit}' "$SIRIUS_TIMING")
                SW[$q]=$(awk -F',' '$1~/^iter_/ && $1!="iter_1" && $2 != "N/A"{v=$2+0; if(min==""||v<min)min=v}END{if(min!="")print min}' "$SIRIUS_TIMING")
            fi
        done

    printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
        "Query" "DuckDB Cold" "DuckDB Warm" "Sirius Cold" "Sirius Warm" "Speedup (warm)"
    printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s\n" \
        "-------" "-------------" "-------------" "-------------" "-------------" "--------------"

        local TOTAL_DC=0 TOTAL_DW=0 TOTAL_SC=0 TOTAL_SW=0
        local HAVE_DC=0 HAVE_DW=0 HAVE_SC=0 HAVE_SW=0

    for q in "${QUERIES[@]}"; do
        local dc="${DC[$q]:-N/A}" dw="${DW[$q]:-N/A}"
        local sc="${SC[$q]:-N/A}" sw="${SW[$q]:-N/A}"

        local speedup="N/A"
        if [ "$dw" != "N/A" ] && [ "$sw" != "N/A" ]; then
            speedup=$(echo "scale=2; $dw / $sw" | bc 2>/dev/null || echo "N/A")
            [ "$speedup" != "N/A" ] && speedup="${speedup}x"
        fi

        local fmt_dc="N/A" fmt_dw="N/A" fmt_sc="N/A" fmt_sw="N/A"
        [ "$dc" != "N/A" ] && fmt_dc=$(printf "%.2fs" "$dc")
        [ "$dw" != "N/A" ] && fmt_dw=$(printf "%.2fs" "$dw")
        [ "$sc" != "N/A" ] && fmt_sc=$(printf "%.2fs" "$sc")
        [ "$sw" != "N/A" ] && fmt_sw=$(printf "%.2fs" "$sw")

        printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
            "Q${q}" "$fmt_dc" "$fmt_dw" "$fmt_sc" "$fmt_sw" "$speedup"

            if [ "$dc" != "N/A" ]; then
                TOTAL_DC=$(echo "$TOTAL_DC + $dc" | bc)
                HAVE_DC=1
            fi
            if [ "$dw" != "N/A" ]; then
                TOTAL_DW=$(echo "$TOTAL_DW + $dw" | bc)
                HAVE_DW=1
            fi
            if [ "$sc" != "N/A" ]; then
                TOTAL_SC=$(echo "$TOTAL_SC + $sc" | bc)
                HAVE_SC=1
            fi
            if [ "$sw" != "N/A" ]; then
                TOTAL_SW=$(echo "$TOTAL_SW + $sw" | bc)
                HAVE_SW=1
            fi
        done

        local total_speedup="N/A"
        if [ "$HAVE_DW" -eq 1 ] && [ "$HAVE_SW" -eq 1 ] && [ "$(echo "$TOTAL_SW > 0" | bc)" -eq 1 ]; then
            total_speedup=$(echo "scale=2; $TOTAL_DW / $TOTAL_SW" | bc 2>/dev/null || echo "N/A")
            [ "$total_speedup" != "N/A" ] && total_speedup="${total_speedup}x"
        fi

        local fmt_total_dc="N/A" fmt_total_dw="N/A" fmt_total_sc="N/A" fmt_total_sw="N/A"
        [ "$HAVE_DC" -eq 1 ] && fmt_total_dc=$(printf '%.2fs' "$TOTAL_DC")
        [ "$HAVE_DW" -eq 1 ] && fmt_total_dw=$(printf '%.2fs' "$TOTAL_DW")
        [ "$HAVE_SC" -eq 1 ] && fmt_total_sc=$(printf '%.2fs' "$TOTAL_SC")
        [ "$HAVE_SW" -eq 1 ] && fmt_total_sw=$(printf '%.2fs' "$TOTAL_SW")

        printf "%-7s-+-%13s-+-%13s-+-%13s-+-%13s-+-%14s\n" \
            "-------" "-------------" "-------------" "-------------" "-------------" "--------------"
        printf "%-7s | %13s | %13s | %13s | %13s | %14s\n" \
            "TOTAL" "$fmt_total_dc" "$fmt_total_dw" "$fmt_total_sc" "$fmt_total_sw" "$total_speedup"
        echo ""
        echo "============================================================"
        echo "All output saved to $RUN_DIR"
    } | tee "$RUN_DIR/comparison.txt"
}

# ---------------------------------------------------------------------------
# --report mode: regenerate comparison/timings from an existing run directory
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--report" ]; then
    shift
    REPORT_FLOAT_TOL="1e-10"
    while [ $# -gt 1 ]; do
        case "$1" in
            --float-tolerance)
                REPORT_FLOAT_TOL="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    if [ $# -ne 1 ]; then
        echo "Usage: $0 --report [--float-tolerance <value>] <run_dir>"
        exit 1
    fi
    RUN_DIR="$1"
    # Resolve relative paths against PROJECT_DIR/runs/ as a convenience.
    if [ ! -d "$RUN_DIR" ] && [ -d "$PROJECT_DIR/runs/$RUN_DIR" ]; then
        RUN_DIR="$PROJECT_DIR/runs/$RUN_DIR"
    fi
    if [ ! -d "$RUN_DIR" ]; then
        echo "ERROR: run directory not found: $RUN_DIR"
        exit 1
    fi
    generate_report "$RUN_DIR" "$REPORT_FLOAT_TOL"
    exit 0
fi

# ---------------------------------------------------------------------------
# Normal benchmark mode
# ---------------------------------------------------------------------------

# Parse optional flags
DUCKDB_RESULTS_DIR=""
DUCKDB_FILE=""
MULTI_SESSION=false
DROP_OS_CACHE=false
PINNING_MODE="none"
FLOAT_TOL="1e-10"
while [ $# -gt 1 ]; do
    case "$1" in
        --config)
            export SIRIUS_CONFIG_FILE="$2"
            shift 2
            ;;
        --float-tolerance)
            FLOAT_TOL="$2"
            shift 2
            ;;
        --pinning-mode)
            PINNING_MODE="$2"
            shift 2
            ;;
        --data-source)
            case "$2" in
                parquet | duckdb | duckdb-native)
                    DATA_SOURCE="$2"
                    ;;
                *)
                    echo "ERROR: --data-source must be 'parquet', 'duckdb', or 'duckdb-native' (got: $2)"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --duckdb-file)
            DUCKDB_FILE="$2"
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
        --iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --timeout)
            QUERY_TIMEOUT="$2"
            shift 2
            ;;
        --duckdb-results)
            DUCKDB_RESULTS_DIR="$2"
            shift 2
            ;;
        --multi-session)
            MULTI_SESSION=true
            shift
            ;;
        --drop-os-cache)
            DROP_OS_CACHE=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-1200}"

case "$PINNING_MODE" in
    none|per-query|pinned-hot) ;;
    *)
        echo "ERROR: --pinning-mode must be 'none', 'per-query', or 'pinned-hot' (got: $PINNING_MODE)"
        exit 1
        ;;
esac

if [ $# -ne 1 ]; then
    echo "Usage: $0 [--config <config_file>] [--data-source parquet|duckdb|duckdb-native] [--parquet-dir <path>] [--duckdb-file <path>]"
    echo "          [--engines 'sirius duckdb'] [--iterations N] [--timeout <seconds>] [--duckdb-results <run_dir>]"
    echo "          [--multi-session] [--drop-os-cache] [--pinning-mode none|per-query|pinned-hot]"
    echo "          [--float-tolerance <value>] <scale_factor>"
    echo "       $0 --report [--float-tolerance <value>] <run_dir>"
    echo "  --data-source parquet       (default) → run_tpch_parquet.sh + test_datasets/tpch_parquet_sf<SF> or --parquet-dir"
    echo "  --data-source duckdb                  → run_tpch_duckdb.sh + performance_test.duckdb or --duckdb-file (GPU-native scan — the default for sirius)"
    echo "  --data-source duckdb-native           → alias of 'duckdb' (GPU-native scan is the default now; kept for compatibility)"
    echo "Example: $0 --config ~/.sirius/sirius.yaml --engines sirius --iterations 3 --timeout 120 1000"
    echo "         $0 --duckdb-results runs/2026-03-10_sf1_2iter 1   # reuse stored DuckDB results for validation"
    echo "         $0 --multi-session --engines duckdb 100            # run DuckDB with fresh process per query"
    echo "         $0 --data-source duckdb --duckdb-file ./performance_test.duckdb 1"
    echo "         $0 --data-source duckdb-native --duckdb-file ./test_datasets/tpch_sf1.duckdb 1"
    echo "         $0 --multi-session --drop-os-cache --engines sirius 1000  # cold-run with OS cache drops"
    exit 1
fi

if [ "$PINNING_MODE" != "none" ] && is_duckdb_source; then
    echo "ERROR: --pinning-mode is parquet-only; do not combine it with --data-source $DATA_SOURCE."
    exit 1
fi

if [ "$PINNING_MODE" = "pinned-hot" ] && [ "$MULTI_SESSION" = true ]; then
    echo "ERROR: --pinning-mode pinned-hot requires single-session mode; remove --multi-session."
    exit 1
fi

SF="$1"
QUERIES=($(seq 1 22))

RUN_DIR="$PROJECT_DIR/runs/$(date +%Y-%m-%d_%H-%M-%S)_sf${SF}_${NUM_ITERATIONS}iter"
mkdir -p "$RUN_DIR"

# Resolve config: explicit --config / env var / default ~/.sirius/sirius.yaml
# Only required when running the sirius engine.
ENGINES="${ENGINES:-sirius duckdb}"
if [[ " $ENGINES " == *" sirius "* ]]; then
    if [ -z "${SIRIUS_CONFIG_FILE:-}" ]; then
        export SIRIUS_CONFIG_FILE="$HOME/.sirius/sirius.yaml"
    fi
    if [ ! -f "$SIRIUS_CONFIG_FILE" ]; then
        echo "ERROR: config file not found: $SIRIUS_CONFIG_FILE"
        exit 1
    fi
    echo "Config file: $SIRIUS_CONFIG_FILE"
    cp "$SIRIUS_CONFIG_FILE" "$RUN_DIR/sirius_config.yaml"
fi

# ---------------------------------------------------------------------------
# --duckdb-results: reuse previously stored DuckDB results for validation.
# Resolves the path, copies results into the new run dir, and removes duckdb
# from the engines list so it is not re-run.
# ---------------------------------------------------------------------------
if [ -n "$DUCKDB_RESULTS_DIR" ]; then
    # Resolve relative paths against PROJECT_DIR/runs/ as a convenience.
    if [ ! -d "$DUCKDB_RESULTS_DIR" ] && [ -d "$PROJECT_DIR/runs/$DUCKDB_RESULTS_DIR" ]; then
        DUCKDB_RESULTS_DIR="$PROJECT_DIR/runs/$DUCKDB_RESULTS_DIR"
    fi
    # Accept either a run directory (with duckdb/ inside) or the duckdb/ directory itself.
    if [ -d "$DUCKDB_RESULTS_DIR/duckdb" ]; then
        DUCKDB_RESULTS_DIR="$DUCKDB_RESULTS_DIR/duckdb"
    fi
    if [ ! -d "$DUCKDB_RESULTS_DIR" ]; then
        echo "ERROR: DuckDB results directory not found: $DUCKDB_RESULTS_DIR"
        exit 1
    fi
    # Verify it contains at least one query result.
    DUCKDB_RESULT_COUNT=0
    for d in "$DUCKDB_RESULTS_DIR"/q*; do
        [ -d "$d" ] && [ -f "$d/result.txt" ] && (( DUCKDB_RESULT_COUNT++ ))
    done
    if [ "$DUCKDB_RESULT_COUNT" -eq 0 ]; then
        echo "ERROR: no query results (q*/result.txt) found in $DUCKDB_RESULTS_DIR"
        exit 1
    fi

    echo "Using stored DuckDB results from: $DUCKDB_RESULTS_DIR ($DUCKDB_RESULT_COUNT queries)"
    cp -a "$DUCKDB_RESULTS_DIR" "$RUN_DIR/duckdb"

    # Remove duckdb from the engines list since we already have its results.
    ENGINES=$(echo "$ENGINES" | sed 's/\bduckdb\b//g' | xargs)
fi

RUN_INFO_FILE="$RUN_DIR/run_info.txt"

if is_duckdb_source; then
    RUN_SCRIPT="$SCRIPT_DIR/run_tpch_duckdb.sh"
    if [ ! -f "$RUN_SCRIPT" ]; then
        echo "ERROR: DuckDB run script not found: $RUN_SCRIPT"
        exit 1
    fi
    DUCKDB_FILE="${DUCKDB_FILE:-$PROJECT_DIR/performance_test.duckdb}"
    DUCKDB_FILE="$(cd "$(dirname "$DUCKDB_FILE")" && pwd)/$(basename "$DUCKDB_FILE")"
    if [ ! -f "$DUCKDB_FILE" ]; then
        echo "ERROR: DuckDB database not found: $DUCKDB_FILE"
        exit 1
    fi
else
    RUN_SCRIPT="$SCRIPT_DIR/run_tpch_parquet.sh"
    if [ ! -f "$RUN_SCRIPT" ]; then
        echo "ERROR: Parquet run script not found: $RUN_SCRIPT"
        exit 1
    fi
    PARQUET_DIR="${PARQUET_DIR:-$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}}"
fi

echo "Scale factor: SF${SF}   Iterations: ${NUM_ITERATIONS} (1 cold + $((NUM_ITERATIONS - 1)) warm)"
echo "Data source: $DATA_SOURCE   Run script: $(basename "$RUN_SCRIPT")"
if is_duckdb_source; then
    echo "DuckDB file: $DUCKDB_FILE"
else
    echo "Parquet dir: $PARQUET_DIR"
fi
echo "Run directory: $RUN_DIR"
if [ -n "${DUCKDB_RESULTS_DIR:-}" ]; then
    echo "DuckDB results: reusing from $DUCKDB_RESULTS_DIR"
fi
echo "=========================================="
echo ""
read -r -p "Optional note about this run (press Enter to skip): " RUN_NOTE
echo ""

# ---------- Run info and environment ----------
echo "=== Collecting run info and filesystem benchmark ==="
{
    echo "Run info — $(date -Iseconds)"
    echo "================================"
    echo ""

    echo "--- Run note ---"
    if [ -n "${RUN_NOTE:-}" ]; then
        echo "$RUN_NOTE"
    else
        echo "(none)"
    fi
    echo ""

    if [ -n "${DUCKDB_RESULTS_DIR:-}" ]; then
        echo "--- DuckDB results ---"
        echo "source: $DUCKDB_RESULTS_DIR (copied, not re-run)"
        echo ""
    fi

    echo "--- Benchmark settings ---"
    echo "multi_session: $MULTI_SESSION"
    echo "drop_os_cache: $DROP_OS_CACHE"
    echo "pinning_mode: $PINNING_MODE"
    echo ""

    echo "--- Benchmark input ---"
    echo "data_source: $DATA_SOURCE"
    if is_duckdb_source; then
        echo "duckdb_file: $DUCKDB_FILE"
    else
        echo "parquet_dir: $PARQUET_DIR"
    fi
    echo ""

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
        # Compare binary to the most recently modified source file (src/ and cucascade/)
        SRC_REF=""
        for dir in "$PROJECT_DIR/src" "$PROJECT_DIR/cucascade"; do
            [ ! -d "$dir" ] && continue
            while IFS= read -r -d '' f; do
                [ -f "$f" ] || continue
                if [ -z "$SRC_REF" ] || [ "$f" -nt "$SRC_REF" ]; then
                    SRC_REF="$f"
                fi
            done < <(find "$dir" -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.c' -o -name '*.h' \) -print0 2>/dev/null)
        done
        if [ -n "$SRC_REF" ]; then
            echo "newest_src: $SRC_REF"
            echo "newest_src_mtime: $(stat -c %y "$SRC_REF" 2>/dev/null || stat -f '%Sm' "$SRC_REF" 2>/dev/null)"
            if [ "$DUCKDB_BIN" -nt "$SRC_REF" ]; then
                echo "build: binary newer than newest source (likely compiled after last source change)"
            else
                echo "build: binary older than newest source (source may have changed since build)"
            fi
        fi
    else
        echo "duckdb binary: not found ($DUCKDB_BIN)"
    fi
    echo ""

    echo "--- Hardware ---"
    echo "hostname: $(hostname)"
    echo "memory:"
    sed -n 's/^MemTotal:/  MemTotal: /p; s/^MemAvailable:/  MemAvailable: /p' /proc/meminfo 2>/dev/null || true
    echo "num_cpus: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo '?')"
    echo "load: $(cat /proc/loadavg 2>/dev/null || (uptime 2>/dev/null | sed 's/.*load average: //') || echo 'N/A')"
    echo ""
    echo "GPUs:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null | while read -r line; do echo "  $line"; done || nvidia-smi
    else
        echo "  nvidia-smi not available"
    fi
    echo ""

    echo "--- Filesystem benchmark (read-only, input location) ---"
    if is_duckdb_source; then
        if [ -f "$DUCKDB_FILE" ]; then
            SIZE_BYTES=$(stat -c %s "$DUCKDB_FILE" 2>/dev/null || stat -f %z "$DUCKDB_FILE" 2>/dev/null)
            SIZE_MB=$((SIZE_BYTES / 1048576))
            READ_MB=$((SIZE_MB < 100 ? SIZE_MB : 100))
            echo "file: $DUCKDB_FILE"
            echo "read_size_mb: $READ_MB"
            START=$(date +%s.%N)
            dd if="$DUCKDB_FILE" of=/dev/null bs=1M count="$READ_MB" 2>/dev/null
            END=$(date +%s.%N)
            ELAPSED=$(echo "$END - $START" | bc 2>/dev/null || echo "?")
            if [ "$ELAPSED" != "?" ] && [ "$(echo "$ELAPSED > 0" | bc 2>/dev/null)" -eq 1 ]; then
                THROUGHPUT=$(echo "scale=2; $READ_MB / $ELAPSED" | bc 2>/dev/null)
                echo "elapsed_s: $ELAPSED"
                echo "throughput_mb_s: $THROUGHPUT"
            else
                echo "elapsed_s: $ELAPSED (could not compute throughput)"
            fi
        else
            echo "duckdb file not found: $DUCKDB_FILE (benchmark skipped)"
        fi
    elif [ -d "$PARQUET_DIR" ]; then
        FIRST_PARQUET=""
        for f in "$PARQUET_DIR"/lineitem.parquet \
                 "$PARQUET_DIR"/lineitem_*.parquet \
                 "$PARQUET_DIR"/lineitem/*.parquet \
                 "$PARQUET_DIR"/*.parquet; do
            [ -f "$f" ] && { FIRST_PARQUET="$f"; break; }
        done
        if [ -n "$FIRST_PARQUET" ]; then
            SIZE_BYTES=$(stat -c %s "$FIRST_PARQUET" 2>/dev/null || stat -f %z "$FIRST_PARQUET" 2>/dev/null)
            SIZE_MB=$((SIZE_BYTES / 1048576))
            # Read 100 MB or the whole file if smaller
            READ_MB=$((SIZE_MB < 100 ? SIZE_MB : 100))
            echo "file: $FIRST_PARQUET"
            echo "read_size_mb: $READ_MB"
            START=$(date +%s.%N)
            dd if="$FIRST_PARQUET" of=/dev/null bs=1M count="$READ_MB" 2>/dev/null
            END=$(date +%s.%N)
            ELAPSED=$(echo "$END - $START" | bc 2>/dev/null || echo "?")
            if [ "$ELAPSED" != "?" ] && [ "$(echo "$ELAPSED > 0" | bc 2>/dev/null)" -eq 1 ]; then
                THROUGHPUT=$(echo "scale=2; $READ_MB / $ELAPSED" | bc 2>/dev/null)
                echo "elapsed_s: $ELAPSED"
                echo "throughput_mb_s: $THROUGHPUT"
            else
                echo "elapsed_s: $ELAPSED (could not compute throughput)"
            fi
        else
            echo "no parquet file found in $PARQUET_DIR (benchmark skipped)"
        fi
    else
        echo "input dir not present: $PARQUET_DIR (benchmark skipped)"
    fi
} | tee "$RUN_INFO_FILE"

echo "Run info saved to $RUN_INFO_FILE"
echo "=========================================="

OVERALL_STATUS=0
for engine in $ENGINES; do
    ENGINE_DIR="$RUN_DIR/$engine"
    mkdir -p "$ENGINE_DIR"

    echo ""
    echo "=== Running $engine ==="
    EXTRA_ARGS=()
    if is_duckdb_source; then
        # Both duckdb and duckdb-native route the sirius engine's seq_scan to the
        # GPU-native scan (the only seq_scan path); the duckdb engine stays an
        # unaffected CPU baseline for validation.
        EXTRA_ARGS+=(--duckdb-file "$DUCKDB_FILE")
    elif [ -n "${PARQUET_DIR:-}" ]; then
        EXTRA_ARGS+=(--parquet-dir "$PARQUET_DIR")
    fi
    EXTRA_ARGS+=(--iterations "$NUM_ITERATIONS")
    EXTRA_ARGS+=(--timeout "$QUERY_TIMEOUT")
    if [ "$MULTI_SESSION" = true ]; then
        EXTRA_ARGS+=(--multi-session)
    fi
    if [ "$DROP_OS_CACHE" = true ]; then
        EXTRA_ARGS+=(--drop-os-cache)
    fi
    # Pinning is a Sirius-only feature; the runner ignores it for the duckdb engine,
    # but we still pass the flag so DuckDB-engine logs reflect the requested mode.
    if [ "$PINNING_MODE" != "none" ]; then
        EXTRA_ARGS+=(--pinning-mode "$PINNING_MODE")
    fi
    OUTPUT_DIR="$ENGINE_DIR" "$RUN_SCRIPT" "${EXTRA_ARGS[@]}" "$engine" "$SF" "${QUERIES[@]}" \
        2>&1 | tee "$ENGINE_DIR/run.log"
    status=$?
    if [ "$status" -ne 0 ]; then
        OVERALL_STATUS=1
        echo "ERROR: $engine benchmark run failed with exit code $status" | tee -a "$ENGINE_DIR/run.log"
    fi
done

generate_report "$RUN_DIR" "$FLOAT_TOL" || OVERALL_STATUS=1
exit "$OVERALL_STATUS"
