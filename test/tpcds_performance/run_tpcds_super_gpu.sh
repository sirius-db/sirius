#!/usr/bin/env bash
# =============================================================================
# Run the 15 TPC-DS queries that execute cleanly on GPU via new Sirius (gpu_execution).
#
# Usage:
#   ./run_tpcds_super_gpu.sh <parquet_dir> [options]
#
# Examples:
#   ./run_tpcds_super_gpu.sh /data/tpcds_parquet_sf1
#   ./run_tpcds_super_gpu.sh /data/tpcds_parquet_sf10 --output-dir /results/sf10
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# TPC-DS queries confirmed to run cleanly on GPU with new Sirius (no errors)
GPU_QUERIES=(3 7 22 26 32 37 42 52 55 62 82 85 92 93 97)

exec bash "$SCRIPT_DIR/run_tpcds_super.sh" "$@" --queries "${GPU_QUERIES[@]}"
