#!/usr/bin/env bash
# Generate TPC-H datasets using tpchgen-rs (parquet) or DuckDB's dbgen (duckdb).
#
# Usage:
#   ./generate_tpch_data.sh <scale_factor> [--format duckdb|parquet] [--output <path>] [--jobs N]
#   ./generate_tpch_data.sh <scale_factor> --format duckdb --cluster [--cluster-keys <spec>]
#   ./generate_tpch_data.sh <scale_factor> [output_dir] [jobs]     # backward-compatible
#
# Arguments:
#   scale_factor    - TPC-H scale factor (e.g. 1, 10, 100)
#   --format        - Output format: 'parquet' (default) or 'duckdb'
#   --output        - Output path (default depends on format)
#   --jobs          - Number of parallel jobs for parquet generation (default: nproc)
#   --cluster       - (duckdb only) physically sort tables at load so on-disk zone
#                     maps (per-row-group min/max) become selective. This is what
#                     makes Sirius's native-scan row-group pruning effective on
#                     date-filtered queries. Default sort keys cluster the fact and
#                     order tables on their date columns.
#   --cluster-keys  - Override the sort keys, as "table:col,table:col,...".
#                     Default: "lineitem:l_shipdate,orders:o_orderdate". Implies --cluster.
#
# Parquet format uses tpchgen-rs for optimized row groups and compression.
# DuckDB format uses DuckDB's built-in dbgen() extension.
#
# --cluster is duckdb-only on purpose: tpchgen-rs writes Arrow Decimal128 as
# FIXED_LEN_BYTE_ARRAY, which trips Sirius's `skip_pushdown_due_to_flba` and
# disables row-group pruning for the whole parquet file (so sorting wouldn't help).
# The duckdb path stores decimals as INT64 and prunes correctly.
#
# Examples:
#   cd test/tpch_performance
#   pixi run bash generate_tpch_data.sh 100
#   pixi run bash generate_tpch_data.sh 100 --format duckdb
#   pixi run bash generate_tpch_data.sh 100 --format duckdb --cluster
#   pixi run bash generate_tpch_data.sh 10 --format duckdb --cluster-keys "lineitem:l_shipdate,orders:o_orderdate"
#   pixi run bash generate_tpch_data.sh 100 --format parquet --output /data/tpch_sf100
#
# Or it can be called standalone if rust, python, and pyarrow are in PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scale_factor> [--format duckdb|parquet] [--output <path>] [--jobs N] [--cluster] [--cluster-keys <spec>]"
    echo "Examples:"
    echo "  $0 100                          # SF100, parquet (default)"
    echo "  $0 100 --format duckdb          # SF100, duckdb database"
    echo "  $0 100 --format duckdb --cluster  # SF100, duckdb database sorted on date columns"
    echo "  $0 100 --format parquet --output /data/tpch_sf100"
    exit 1
fi

SF="$1"
shift

FORMAT="parquet"
OUTPUT=""
JOBS="$(nproc)"
CLUSTER="false"
# Default clustering keys: sort the fact/order tables on their date columns so
# date-range predicates prune row groups. Override with --cluster-keys.
CLUSTER_KEYS="lineitem:l_shipdate,orders:o_orderdate"
# Fixed TPC-H table set (used by the clustered duckdb build).
TPCH_TABLES="nation region part supplier partsupp customer orders lineitem"

# shellcheck source=dbgen_bootstrap.sh
. "$SCRIPT_DIR/dbgen_bootstrap.sh"

# table:dbgen-flag, largest table first. Case matters: -T l is nation/region
# while -T L is lineitem. Each of these flags writes exactly one .tbl.
TPCH_TABLE_FLAGS="lineitem:L orders:O partsupp:S customer:c part:P supplier:s nation:n region:r"

# Build a .duckdb from the classic dbgen, the same generator that produces the
# refresh sets and query streams. dbgen writes into DSS_PATH but resolves
# dists.dss relative to its own directory, hence the subshell.
#
# Tables are generated one at a time and each .tbl is deleted once loaded, so
# peak disk is the largest single table plus the database rather than the whole
# raw dataset alongside it. Largest first matters: staging lineitem while the
# database is still empty sets a lower high-water mark than doing it last, when
# the other seven tables are already there. At SF1000 that is the difference
# between roughly 890 GiB and 970 GiB.
#
# The cost is that -T O and -T L each walk the order generator, so orders and
# lineitem take two passes instead of one. Disk is the scarcer resource at the
# scale factors where this matters.
build_duckdb_from_dbgen() {
    local out="$1" sf="$2"
    local dbgen_dir="$PROJECT_DIR/test_datasets/tpch-dbgen"
    local stage="${out%.duckdb}.tbl_stage"

    ensure_tpch_tools "$dbgen_dir" "$PROJECT_DIR" dbgen
    mkdir -p "$stage"
    stage="$(cd "$stage" && pwd)"
    # shellcheck disable=SC2064
    trap "rm -rf '$stage'" EXIT

    rm -f "$out" "$out".wal
    # Loading needs no GPU, and a stale Sirius config would otherwise block it.
    SIRIUS_DISABLE=1 "$DUCKDB" "$out" -c "$(cat "$SCRIPT_DIR/tpch_schema.sql")"

    local spec table flag
    for spec in $TPCH_TABLE_FLAGS; do
        table="${spec%%:*}"
        flag="${spec##*:}"
        echo "  ${table}: dbgen -T ${flag} ..."
        (cd "$dbgen_dir" && DSS_PATH="$stage" ./dbgen -f -q -s "$sf" -T "$flag")
        echo "  ${table}: loading $(du -h "$stage/$table.tbl" | cut -f1) ..."
        SIRIUS_DISABLE=1 "$DUCKDB" "$out" \
            -c "COPY $table FROM '$stage/$table.tbl' (HEADER false, DELIMITER '|');"
        rm -f "$stage/$table.tbl"
    done

    rm -rf "$stage"
    trap - EXIT
}

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
        --cluster)
            CLUSTER="true"
            shift
            ;;
        --cluster-keys)
            CLUSTER_KEYS="$2"
            CLUSTER="true"
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

if [ "$CLUSTER" = "true" ] && [ "$FORMAT" != "duckdb" ]; then
    echo "ERROR: --cluster is only supported with --format duckdb."
    echo "       Parquet clustering is intentionally not implemented: tpchgen-rs writes Arrow"
    echo "       Decimal128 as FIXED_LEN_BYTE_ARRAY, which disables Sirius row-group pruning for"
    echo "       the whole file (skip_pushdown_due_to_flba). Use --format duckdb."
    exit 1
fi

# --- Set default output path ---
if [ -z "$OUTPUT" ]; then
    if [ "$FORMAT" = "duckdb" ]; then
        if [ "$CLUSTER" = "true" ]; then
            # Distinct name so the sorted dataset can coexist with the unsorted one.
            OUTPUT="$PROJECT_DIR/test_datasets/tpch_sf${SF}_sorted.duckdb"
        else
            OUTPUT="$PROJECT_DIR/test_datasets/tpch_sf${SF}.duckdb"
        fi
    else
        OUTPUT="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"
    fi
fi

# The duckdb build deletes its target before writing, and the backward-compatible
# positional [output] [jobs] arguments accept any string, so a misplaced path
# lands here as the destination. Refuse to remove anything that is not already a
# DuckDB database; those start with "DUCK" at byte 8.
if [ "$FORMAT" = "duckdb" ] && [ -e "$OUTPUT" ]; then
    if [ -d "$OUTPUT" ]; then
        echo "ERROR: --format duckdb writes a file, but '$OUTPUT' is a directory."
        exit 1
    fi
    if [ "$(od -An -c -j8 -N4 "$OUTPUT" 2>/dev/null | tr -d ' \n')" != "DUCK" ]; then
        echo "ERROR: refusing to overwrite '$OUTPUT' - it is not a DuckDB database."
        echo "       The positional [output] [jobs] arguments take any path, so this is"
        echo "       usually a stray or mistyped argument. To overwrite it anyway, delete"
        echo "       it first, or pass an explicit --output <file>.duckdb."
        exit 1
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

    if [ "$CLUSTER" = "true" ]; then
        # Clustered build. dbgen into a staging database, then write a *fresh*,
        # COMPACT database with the requested tables physically sorted (ORDER BY at
        # load). A fresh copy — rather than an in-place CREATE+DROP — avoids leaving
        # dead blocks behind, which would roughly double the file size and inflate
        # cold-scan reads of un-pruned data.
        STAGING="${OUTPUT%.duckdb}.staging.duckdb"
        rm -f "$OUTPUT" "$OUTPUT".wal "$STAGING" "$STAGING".wal
        trap 'rm -f "$STAGING" "$STAGING".wal' EXIT

        echo "Generating TPC-H SF${SF} (staging) ..."
        build_duckdb_from_dbgen "$STAGING" "$SF"
        SIRIUS_DISABLE=1 "$DUCKDB" "$STAGING" -c "CHECKPOINT;"

        # Parse --cluster-keys "table:col,table:col" into a lookup.
        declare -A SORT_COL
        IFS=',' read -ra _pairs <<< "$CLUSTER_KEYS" || true
        for _p in "${_pairs[@]}"; do
            _t="${_p%%:*}"
            _c="${_p##*:}"
            [ -z "$_t" ] && continue
            case " $TPCH_TABLES " in
                *" $_t "*) SORT_COL["$_t"]="$_c" ;;
                *) echo "WARNING: --cluster-keys references unknown table '$_t' (ignored)" ;;
            esac
        done

        # Build the fresh, sorted database from the staging copy.
        BUILD_SQL="ATTACH '${STAGING}' AS s (READ_ONLY);"
        echo "Clustering tables:"
        for _t in $TPCH_TABLES; do
            if [ -n "${SORT_COL[$_t]:-}" ]; then
                BUILD_SQL+=" CREATE TABLE ${_t} AS FROM s.${_t} ORDER BY ${SORT_COL[$_t]};"
                echo "  ${_t} ORDER BY ${SORT_COL[$_t]}"
            else
                BUILD_SQL+=" CREATE TABLE ${_t} AS FROM s.${_t};"
            fi
        done
        BUILD_SQL+=" CHECKPOINT;"

        echo "Writing sorted database into $OUTPUT ..."
        SIRIUS_DISABLE=1 "$DUCKDB" "$OUTPUT" -c "$BUILD_SQL"

        rm -f "$STAGING" "$STAGING".wal
        trap - EXIT
    else
        if [ -f "$OUTPUT" ]; then
            echo "WARNING: Database file already exists, will overwrite tables"
        fi

        echo "Generating TPC-H SF${SF} data into $OUTPUT ..."
        build_duckdb_from_dbgen "$OUTPUT" "$SF"
    fi
fi

echo ""
echo "TPC-H SF${SF} data generation complete."
echo "  Format:  $FORMAT"
echo "  Output:  $OUTPUT"
if [ "$CLUSTER" = "true" ]; then
    echo "  Cluster: $CLUSTER_KEYS"
fi
