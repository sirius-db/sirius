#!/bin/bash
# ClickBench benchmark for Sirius on RTX 6000 (24 GB)
# Based on: scripts/clickbench_runner/run_official.sh (L40S/GH200 branches)
#
# Usage: bash benchmark_results/run_benchmark.sh [database]
#
# GPU memory: caching + processing <= ~85% of 24GB VRAM

TRIES=3
GPU_CACHING_SIZE="${GPU_CACHING_SIZE:-10 GB}"
GPU_PROCESSING_SIZE="${GPU_PROCESSING_SIZE:-10 GB}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DUCKDB="$REPO_DIR/build/release/duckdb"
QUERIES_FILE="$REPO_DIR/scripts/clickbench_runner/queries.sql"
HITS_DB="${1:-$REPO_DIR/clickbench_10m.duckdb}"

export LD_LIBRARY_PATH="$REPO_DIR/.pixi/envs/default/lib:$LD_LIBRARY_PATH"

if [[ ! -x "$DUCKDB" ]]; then
    echo "ERROR: Sirius binary not found at $DUCKDB. Run: bash build.sh"
    exit 1
fi

if [[ ! -f "$HITS_DB" ]]; then
    echo "ERROR: Database not found at $HITS_DB"
    exit 1
fi

echo "=== Sirius ClickBench (RTX 6000, $(basename "$HITS_DB")) ==="
echo "GPU cache: $GPU_CACHING_SIZE | GPU processing: $GPU_PROCESSING_SIZE | Tries: $TRIES"
echo ""

QUERY_NUM=0
cat "$QUERIES_FILE" | while read -r query; do
    printf "Q%02d: %.80s\n" "$QUERY_NUM" "$query"

    cli_params=()
    cli_params+=("-c" ".timer on")
    cli_params+=("-c" "call gpu_buffer_init('${GPU_CACHING_SIZE}', '${GPU_PROCESSING_SIZE}');")
    for i in $(seq 1 $TRIES); do
        cli_params+=("-c" "call gpu_processing(\"${query}\");")
    done

    "$DUCKDB" "$HITS_DB" "${cli_params[@]}" 2>&1

    QUERY_NUM=$((QUERY_NUM + 1))
done
