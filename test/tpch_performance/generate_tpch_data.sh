#!/usr/bin/env bash
# Generate TPC-H datasets using tpchgen-rs (parquet) or DuckDB's dbgen (duckdb).
#
# Usage:
#   ./generate_tpch_data.sh <scale_factor> [--format duckdb|parquet] [--output <path>] [--jobs N]
#   ./generate_tpch_data.sh <scale_factor> [output_dir] [jobs]     # backward-compatible
#
# Arguments:
#   scale_factor  - TPC-H scale factor (e.g. 1, 10, 100)
#   --format      - Output format: 'parquet' (default) or 'duckdb'
#   --output      - Output path (default depends on format)
#   --jobs        - Number of parallel jobs for parquet generation (default: nproc)
#
# Parquet format uses tpchgen-rs for optimized row groups and compression.
# DuckDB format uses DuckDB's built-in dbgen() extension.
#
# Examples:
#   cd test/tpch_performance
#   pixi run bash generate_tpch_data.sh 100
#   pixi run bash generate_tpch_data.sh 100 --format duckdb
#   pixi run bash generate_tpch_data.sh 100 --format parquet --output /data/tpch_sf100
#
# Or it can be called standalone if rust, python, and pyarrow are in PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scale_factor> [--format duckdb|parquet] [--output <path>] [--jobs N]"
    echo "Examples:"
    echo "  $0 100                        # SF100, parquet (default)"
    echo "  $0 100 --format duckdb        # SF100, duckdb database"
    echo "  $0 100 --format parquet --output /data/tpch_sf100"
    exit 1
fi

SF="$1"
shift

FORMAT="parquet"
OUTPUT=""
JOBS="$(nproc)"

while [ $# -gt 0 ]; do
    case "$1" in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            # Backward compatibility: positional [output_dir] [jobs]
            if [ -z "$OUTPUT" ]; then
                OUTPUT="$1"
            else
                JOBS="$1"
            fi
            shift
            ;;
    esac
done

if [ "$FORMAT" != "duckdb" ] && [ "$FORMAT" != "parquet" ]; then
    echo "ERROR: format must be 'duckdb' or 'parquet', got: $FORMAT"
    exit 1
fi

# --- Set default output path ---
if [ -z "$OUTPUT" ]; then
    if [ "$FORMAT" = "duckdb" ]; then
        OUTPUT="$PROJECT_DIR/test_datasets/tpch_sf${SF}.duckdb"
    else
        OUTPUT="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"
    fi
fi

# --- Generate data ---
if [ "$FORMAT" = "parquet" ]; then
    # Use tpchgen-rs for optimized parquet output
    if [ -d "$OUTPUT" ]; then
        echo "Output directory already exists: $OUTPUT"
        echo "Skipping generation. Remove the directory to regenerate."
        exit 0
    fi

    TPCHGEN_DIR="$PROJECT_DIR/test_datasets/tpchgen-rs"

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
    echo "Output: $OUTPUT"
    python "$TPCHGEN_DIR/scripts/generate_tpch.py" \
        -s "$SF" \
        -f parquet \
        -j "$JOBS" \
        -o "$OUTPUT"

elif [ "$FORMAT" = "duckdb" ]; then
    # Use DuckDB's built-in dbgen() for DuckDB format
    if [ ! -x "$DUCKDB" ]; then
        echo "ERROR: DuckDB binary not found at $DUCKDB"
        echo "Build first: CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) make"
        exit 1
    fi

    if [ -f "$OUTPUT" ]; then
        echo "WARNING: Database file already exists, will overwrite tables"
    fi

    echo "Generating TPC-H SF${SF} data into $OUTPUT ..."
    "$DUCKDB" "$OUTPUT" -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=${SF});"
fi

echo ""
echo "TPC-H SF${SF} data generation complete."
echo "  Format: $FORMAT"
echo "  Output: $OUTPUT"
