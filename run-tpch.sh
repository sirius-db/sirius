#!/bin/bash
TRIES=3
GPU_CACHING_SIZE='100 GB'
GPU_PROCESSING_SIZE='80 GB'
CPU_PROCESSING_SIZE="100 GB"

cat tpch-queries-run.sql | while read -r query; do
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
    echo "${cli_params[@]}"
    ./build/release/duckdb tpch.duckdb "${cli_params[@]}"
done;