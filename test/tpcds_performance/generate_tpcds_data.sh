#!/usr/bin/env bash
# =============================================================================
# Generate TPC-DS data using DuckDB's dsdgen() extension.
#
# Usage:
#   ./generate_tpcds_data.sh <scale_factor> [--format duckdb|parquet] [--output <path>]
#
# Examples:
#   ./generate_tpcds_data.sh 1
#   ./generate_tpcds_data.sh 10 --format parquet
#   ./generate_tpcds_data.sh 100 --format duckdb --output /data/tpcds_sf100.duckdb
#   ./generate_tpcds_data.sh 100 --format parquet --output /data/tpcds_parquet_sf100
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <scale_factor> [--format duckdb|parquet] [--output <path>]"
    exit 1
fi

SF="$1"
shift

FORMAT="duckdb"
OUTPUT=""

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
        *)
            echo "Unknown option: $1"
            exit 1
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
        OUTPUT="$PROJECT_DIR/test_datasets/tpcds_sf${SF}.duckdb"
    else
        OUTPUT="$PROJECT_DIR/test_datasets/tpcds_parquet_sf${SF}"
    fi
fi

# --- Validate prerequisites ---
if [ ! -x "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found at $DUCKDB"
    echo "Build first: CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) make"
    exit 1
fi

# --- Extract query texts (skip if already present) ---
QUERY_DIR="$SCRIPT_DIR/queries"
if [ -d "$QUERY_DIR" ] && [ -f "$QUERY_DIR/q99.sql" ]; then
    echo "Query files already exist in $QUERY_DIR, skipping extraction."
else
    mkdir -p "$QUERY_DIR"

    echo "Extracting TPC-DS query texts to $QUERY_DIR..."
    EXTRACT_SQL="INSTALL tpcds; LOAD tpcds;"$'\n'
    for q in $(seq 1 99); do
        EXTRACT_SQL+="COPY (SELECT query FROM tpcds_queries() WHERE query_nr = ${q}) TO '${QUERY_DIR}/q${q}.sql' (HEADER false, QUOTE '', DELIMITER '');"$'\n'
    done

    # Use a temporary database for query extraction (dsdgen not needed for queries)
    TEMP_DB="$PROJECT_DIR/tpcds_extract_tmp.duckdb"
    rm -f "$TEMP_DB"
    echo "$EXTRACT_SQL" | "$DUCKDB" "$TEMP_DB" 2>&1
    rm -f "$TEMP_DB"
    echo "Extracted 99 query files to $QUERY_DIR"

    # Convert double-quoted identifiers to single quotes in all queries
    # e.g. "order count" -> 'order count'
    # This allows uniform gpu_processing("...") wrapping without escaping conflicts
    echo "Converting double-quoted identifiers to single quotes..."
    for f in "$QUERY_DIR"/q*.sql; do
        sed -i 's/"/'"'"'/g' "$f"
    done
fi

# --- Generate TPC-DS data ---
if [ "$FORMAT" = "duckdb" ]; then
    echo ""
    echo "Generating TPC-DS data at SF${SF} into $OUTPUT ..."
    if [ -f "$OUTPUT" ]; then
        echo "WARNING: Database file already exists, will overwrite tables"
    fi

    "$DUCKDB" "$OUTPUT" -c "INSTALL tpcds; LOAD tpcds; CALL dsdgen(sf=${SF}, overwrite=true);"
    echo "Done. Database saved to: $OUTPUT"

elif [ "$FORMAT" = "parquet" ]; then
    echo ""
    echo "Generating TPC-DS data at SF${SF} (parquet export to $OUTPUT) ..."
    mkdir -p "$OUTPUT"

    TEMP_DB="$PROJECT_DIR/tpcds_gen_tmp.duckdb"
    rm -f "$TEMP_DB"
    "$DUCKDB" "$TEMP_DB" <<EOF
INSTALL tpcds;
LOAD tpcds;
CALL dsdgen(sf=${SF});
EXPORT DATABASE '${OUTPUT}' (FORMAT PARQUET);
EOF
    rm -f "$TEMP_DB"
    echo "Done. Parquet files saved to: $OUTPUT"
fi

echo ""
echo "Summary:"
echo "  Scale factor: $SF"
echo "  Format:       $FORMAT"
echo "  Output:       $OUTPUT"
echo "  Queries:      $QUERY_DIR/q{1..99}.sql"
