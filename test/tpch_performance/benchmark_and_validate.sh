#!/bin/bash
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
#     sirius_config.cfg - copy of SIRIUS_CONFIG_FILE
#     sirius/run.log  sirius/q<N>/result.txt  sirius/q<N>/timings.csv
#     duckdb/run.log  duckdb/q<N>/result.txt  duckdb/q<N>/timings.csv
#     validation.csv
#     comparison.txt
#     timings.csv
#
# Before running benchmarks, a tiny read-only filesystem benchmark is run on the
# input parquet directory and results are recorded in run_info.txt.
#
# Usage:
#   export SIRIUS_CONFIG_FILE=...
#   ./test/tpch_performance/benchmark_and_validate.sh <scale_factor>
#   ./test/tpch_performance/benchmark_and_validate.sh --report <run_dir>
#
# Example:
#   ./test/tpch_performance/benchmark_and_validate.sh 1
#   ./test/tpch_performance/benchmark_and_validate.sh --report runs/2026-03-10_12-00-00_sf1_2iter

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_tpch_parquet.sh"

# ---------------------------------------------------------------------------
# Report generation from an existing run directory.
# Shared by both --report mode and the end of a normal benchmark run.
# ---------------------------------------------------------------------------
generate_report() {
    local RUN_DIR="$1"
    local QUERIES=()

    # Discover which queries are present by scanning sirius/ and duckdb/ subdirs.
    for d in "$RUN_DIR"/sirius/q* "$RUN_DIR"/duckdb/q*; do
        [ -d "$d" ] || continue
        local qnum="${d##*/q}"
        QUERIES+=("$qnum")
    done
    # Deduplicate and sort numerically.
    readarray -t QUERIES < <(printf '%s\n' "${QUERIES[@]}" | sort -un)

    if [ ${#QUERIES[@]} -eq 0 ]; then
        echo "ERROR: no query directories found in $RUN_DIR"
        return 1
    fi

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
        local SIRIUS_FILE="$RUN_DIR/sirius/q${q}/result.txt"
        local DUCKDB_FILE="$RUN_DIR/duckdb/q${q}/result.txt"
        local status
        if has_error "$SIRIUS_FILE"; then
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
            DC[$q]=$(awk -F',' '$1=="iter_1"{print $2}' "$DUCKDB_TIMING")
            DW[$q]=$(awk -F',' '$1~/^iter_/ && $1!="iter_1"{v=$2+0; if(min==""||v<min)min=v}END{if(min!="")print min}' "$DUCKDB_TIMING")
        fi
        if [ -f "$SIRIUS_TIMING" ]; then
            SC[$q]=$(awk -F',' '$1=="iter_1"{print $2}' "$SIRIUS_TIMING")
            SW[$q]=$(awk -F',' '$1~/^iter_/ && $1!="iter_1"{v=$2+0; if(min==""||v<min)min=v}END{if(min!="")print min}' "$SIRIUS_TIMING")
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

        [ "$dc" != "N/A" ] && TOTAL_DC=$(echo "$TOTAL_DC + $dc" | bc)
        [ "$dw" != "N/A" ] && TOTAL_DW=$(echo "$TOTAL_DW + $dw" | bc)
        [ "$sc" != "N/A" ] && TOTAL_SC=$(echo "$TOTAL_SC + $sc" | bc)
        [ "$sw" != "N/A" ] && TOTAL_SW=$(echo "$TOTAL_SW + $sw" | bc)
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
    # Resolve relative paths against PROJECT_DIR/runs/ as a convenience.
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
if [ $# -ne 1 ]; then
    echo "Usage: $0 <scale_factor>"
    echo "       $0 --report <run_dir>"
    echo "Example: $0 1"
    exit 1
fi

SF="$1"
QUERIES=($(seq 1 22))

RUN_DIR="$PROJECT_DIR/runs/$(date +%Y-%m-%d_%H-%M-%S)_sf${SF}_2iter"
mkdir -p "$RUN_DIR"

if [ -n "${SIRIUS_CONFIG_FILE:-}" ] && [ -f "${SIRIUS_CONFIG_FILE}" ]; then
    cp "$SIRIUS_CONFIG_FILE" "$RUN_DIR/sirius_config.cfg"
else
    cp "$HOME/.sirius/sirius.cfg" "$RUN_DIR/"
fi

RUN_INFO_FILE="$RUN_DIR/run_info.txt"
PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"

echo "Scale factor: SF${SF}   Iterations: 2 (cold + warm)"
echo "Run directory: $RUN_DIR"
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
    if [ -d "$PARQUET_DIR" ]; then
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

for engine in sirius duckdb; do
    ENGINE_DIR="$RUN_DIR/$engine"
    mkdir -p "$ENGINE_DIR"

    echo ""
    echo "=== Running $engine ==="
    OUTPUT_DIR="$ENGINE_DIR" "$RUN_SCRIPT" "$engine" "$SF" "${QUERIES[@]}" \
        2>&1 | tee "$ENGINE_DIR/run.log" || true
done

generate_report "$RUN_DIR"
