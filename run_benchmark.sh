#!/bin/bash
# ClickBench benchmark for Sirius on L40S (48GB VRAM)
# GPU memory: 20GB caching + 23GB processing = 43GB total
set -e

TRIES=3
GPU_CACHING_SIZE='20 GB'
GPU_PROCESSING_SIZE='20 GB'
CPU_PROCESSING_SIZE="40 GB"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUCKDB="$SCRIPT_DIR/build/release/duckdb"
QUERIES_FILE="$SCRIPT_DIR/scripts/clickbench_runner/queries.sql"
HITS_DB="$SCRIPT_DIR/hits.duckdb"
RESULT_FILE="$SCRIPT_DIR/benchmark_results/l40s_$(date +%Y-%m-%d_%H%M%S).txt"

mkdir -p "$SCRIPT_DIR/benchmark_results"

if [[ ! -x "$DUCKDB" ]]; then
    echo "ERROR: Sirius binary not found. Run: bash build.sh"
    exit 1
fi

if [[ ! -f "$HITS_DB" ]]; then
    echo "ERROR: hits.duckdb not found."
    exit 1
fi

echo "=== Sirius ClickBench Benchmark ===" | tee "$RESULT_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)" | tee -a "$RESULT_FILE"
echo "Caching: $GPU_CACHING_SIZE, Processing: $GPU_PROCESSING_SIZE" | tee -a "$RESULT_FILE"
echo "Tries: $TRIES" | tee -a "$RESULT_FILE"
echo "Date: $(date)" | tee -a "$RESULT_FILE"
echo "===================================" | tee -a "$RESULT_FILE"

query_idx=0
cat "$QUERIES_FILE" | while read -r query; do
    echo "" | tee -a "$RESULT_FILE"
    echo "--- Q${query_idx} ---" | tee -a "$RESULT_FILE"
    echo "$query" | tee -a "$RESULT_FILE"

    cli_params=()
    cli_params+=("-c")
    cli_params+=(".timer on")
    cli_params+=("-c")
    cli_params+=("call gpu_buffer_init('${GPU_CACHING_SIZE}', '${GPU_PROCESSING_SIZE}', pinned_memory_size = '${CPU_PROCESSING_SIZE}');")
    for i in $(seq 1 $TRIES); do
      cli_params+=("-c")
      cli_params+=("call gpu_processing(\"${query}\");")
    done

    output=$("$DUCKDB" "$HITS_DB" "${cli_params[@]}" 2>&1) || true
    echo "$output" | tee -a "$RESULT_FILE"

    # Extract run times
    times=$(echo "$output" | grep "Run Time" | awk '{print $4}' | tr '\n' ' ')
    echo "TIMES: $times" | tee -a "$RESULT_FILE"

    # Check for GPU fallback
    if echo "$output" | grep -q "Error in GPUExecuteQuery"; then
        echo "STATUS: CPU_FALLBACK" | tee -a "$RESULT_FILE"
    else
        echo "STATUS: GPU" | tee -a "$RESULT_FILE"
    fi

    query_idx=$((query_idx + 1))
done

echo "" | tee -a "$RESULT_FILE"
echo "=== Benchmark Complete ===" | tee -a "$RESULT_FILE"
echo "Results saved to: $RESULT_FILE"
