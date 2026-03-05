#!/usr/bin/env bash
# Generate TPC-H parquet datasets using sirius-db/tpchgen-rs.
#
# Clones and builds tpchgen-cli from source (with native CPU optimizations),
# then runs scripts/generate_tpch.py to produce partitioned parquet files
# with optimized row groups, encodings, and compression.
#
# Usage:
#   ./generate_tpch_data.sh <scale_factor> [output_dir] [jobs]
#
# Arguments:
#   scale_factor  - TPC-H scale factor (e.g. 1, 10, 100)
#   output_dir    - Output directory (default: test_datasets/tpch_parquet_sf<SF>)
#   jobs          - Number of parallel jobs (default: nproc)
#
# This script is intended to be run via pixi from test/tpch_performance/:
#   cd test/tpch_performance
#   pixi run bash generate_tpch_data.sh 100
#
# Or it can be called standalone if rust, python, and pyarrow are in PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scale_factor> [output_dir] [jobs]"
    echo "Example: $0 100"
    exit 1
fi

SF="$1"
OUTPUT_DIR="${2:-$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}}"
JOBS="${3:-$(nproc)}"
TPCHGEN_DIR="$PROJECT_DIR/test_datasets/tpchgen-rs"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory already exists: $OUTPUT_DIR"
    echo "Skipping generation. Remove the directory to regenerate."
    exit 0
fi

# Step 1: Clone tpchgen-rs if not present
if [ ! -d "$TPCHGEN_DIR" ]; then
    echo "Cloning sirius-db/tpchgen-rs..."
    git clone https://github.com/sirius-db/tpchgen-rs.git "$TPCHGEN_DIR"
else
    echo "tpchgen-rs already cloned at $TPCHGEN_DIR"
fi

# Step 2: Build tpchgen-cli from source (skip if already built)
TPCHGEN_CLI="$TPCHGEN_DIR/target/release/tpchgen-cli"
if [ ! -f "$TPCHGEN_CLI" ]; then
    echo "Building tpchgen-cli with native CPU optimizations..."
    (cd "$TPCHGEN_DIR" && RUSTFLAGS="-C target-cpu=native" cargo build --release -p tpchgen-cli)
else
    echo "tpchgen-cli already built at $TPCHGEN_CLI"
fi

# Step 3: Generate parquet data
echo "Generating TPC-H SF${SF} parquet data with ${JOBS} parallel jobs..."
echo "Output: $OUTPUT_DIR"
python "$TPCHGEN_DIR/scripts/generate_tpch.py" \
    -s "$SF" \
    -f parquet \
    -j "$JOBS" \
    -o "$OUTPUT_DIR"

echo ""
echo "TPC-H SF${SF} data generation complete."
echo "Output: $OUTPUT_DIR"
