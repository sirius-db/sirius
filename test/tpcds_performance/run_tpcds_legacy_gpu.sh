#!/usr/bin/env bash
# =============================================================================
# Run the 22 TPC-DS queries that execute on GPU via legacy Sirius (gpu_processing).
#
# Usage:
#   ./run_tpcds_legacy_gpu.sh <gpu_caching_size> <gpu_processing_size> [options]
#
# Examples:
#   ./run_tpcds_legacy_gpu.sh "1 GB" "2 GB"
#   ./run_tpcds_legacy_gpu.sh "10 GB" "20 GB" --sf 10
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# TPC-DS queries confirmed to run on GPU (no fallback to DuckDB)
GPU_QUERIES=(3 7 17 25 26 29 37 42 43 46 50 52 55 62 68 69 79 82 85 91 92 96)

exec bash "$SCRIPT_DIR/run_tpcds_legacy.sh" "$@" --queries "${GPU_QUERIES[@]}"
