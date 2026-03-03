#!/bin/bash
# Run a single ClickBench query by number (e.g., 27 for Q27)
# Usage: bash benchmark_results/run_query.sh <query_number> [database]
#
# Examples:
#   bash benchmark_results/run_query.sh 27              # Run Q27
#   bash benchmark_results/run_query.sh 27 hits.duckdb  # Run Q27 on a different DB

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <query_number> [database]"
    echo "  e.g.: $0 27"
    exit 1
fi

QUERY_NUM=$1
GPU_CACHING_SIZE="${GPU_CACHING_SIZE:-10 GB}"
GPU_PROCESSING_SIZE="${GPU_PROCESSING_SIZE:-10 GB}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DUCKDB="$REPO_DIR/build/release/duckdb"
QUERIES_FILE="$REPO_DIR/scripts/clickbench_runner/queries.sql"
HITS_DB="${2:-$REPO_DIR/clickbench_10m.duckdb}"

export LD_LIBRARY_PATH="$REPO_DIR/.pixi/envs/default/lib:$LD_LIBRARY_PATH"

# Extract the query (1-indexed line number = query_number + 1)
LINE=$((QUERY_NUM + 1))
query=$(sed -n "${LINE}p" "$QUERIES_FILE")

if [[ -z "$query" ]]; then
    echo "ERROR: Query Q$(printf '%02d' "$QUERY_NUM") not found (line $LINE of $QUERIES_FILE)"
    exit 1
fi

printf "Q%02d: %s\n\n" "$QUERY_NUM" "$query"

"$DUCKDB" "$HITS_DB" \
    -c ".timer on" \
    -c "call gpu_buffer_init('${GPU_CACHING_SIZE}', '${GPU_PROCESSING_SIZE}');" \
    -c "call gpu_processing(\"${query}\");" \
    -c "call gpu_processing(\"${query}\");" \
    -c "call gpu_processing(\"${query}\");" \
    2>&1
