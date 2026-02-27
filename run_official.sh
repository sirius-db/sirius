#!/bin/bash
# Official ClickBench run script for Sirius (clickbench_with_topk branch)
# Based on: https://github.com/ClickHouse/ClickBench/tree/main/sirius/run.sh
#
# GPU memory sizes below are tuned for GH200 (96 GB HBM3).
# Adjust GPU_CACHING_SIZE + GPU_PROCESSING_SIZE to fit your GPU.
# Rule of thumb: caching + processing <= 85% of GPU VRAM.

TRIES=3
GPU_CACHING_SIZE='80 GB'
GPU_PROCESSING_SIZE='40 GB'
CPU_PROCESSING_SIZE="100 GB"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUCKDB="$SCRIPT_DIR/build/release/duckdb"
QUERIES_FILE="$SCRIPT_DIR/scripts/clickbench_runner/queries.sql"
HITS_DB="$SCRIPT_DIR/hits.duckdb"

if [[ ! -x "$DUCKDB" ]]; then
    echo "ERROR: Sirius binary not found. Run: bash build.sh"
    exit 1
fi

# Create hits.duckdb from hits.parquet if needed
if [[ ! -f "$HITS_DB" ]]; then
    if [[ ! -f "$SCRIPT_DIR/hits.parquet" ]]; then
        echo "ERROR: hits.parquet not found. Download from ClickBench:"
        echo "  wget https://datasets.clickhouse.com/hits_compatible/hits.parquet"
        exit 1
    fi
    echo "Creating hits.duckdb from hits.parquet (one-time, ~2 min)..."
    "$DUCKDB" "$HITS_DB" -c "CREATE TABLE hits AS SELECT * FROM read_parquet('$SCRIPT_DIR/hits.parquet');"
    echo "hits.duckdb created."
fi

cat "$QUERIES_FILE" | while read -r query; do
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

    echo "$query";
    cli_params=()
    cli_params+=("-c")
    cli_params+=(".timer on")
    cli_params+=("-c")
    cli_params+=("call gpu_buffer_init(\"${GPU_CACHING_SIZE}\", \"${GPU_PROCESSING_SIZE}\", pinned_memory_size = \"${CPU_PROCESSING_SIZE}\");")
    for i in $(seq 1 $TRIES); do
      cli_params+=("-c")
      cli_params+=("call gpu_processing(\"${query}\");")
    done;
    "$DUCKDB" "$HITS_DB" "${cli_params[@]}"
done
