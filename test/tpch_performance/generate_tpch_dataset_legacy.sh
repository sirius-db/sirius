#!/usr/bin/env bash
# =============================================================================
# Generate TPC-H data in TBL format and load it into a DuckDB database.
#
# Clones and builds tpchgen-cli from sirius-db/tpchgen-rs (bwy/sirius branch),
# generates pipe-delimited .tbl files, then loads them into a DuckDB database
# that can be used by run_tpch_legacy.sh.
#
# Usage:
#   ./generate_tpch_dataset_legacy.sh <scale_factor> [output_db] [jobs]
#
# Arguments:
#   scale_factor  - TPC-H scale factor (e.g. 1, 10, 100)
#   output_db     - Output DuckDB file (default: test_datasets/tpch_sf<SF>.duckdb)
#   jobs          - Number of parallel threads for tpchgen-cli (default: nproc)
#
# Examples:
#   ./generate_tpch_dataset_legacy.sh 1
#   ./generate_tpch_dataset_legacy.sh 10 /data/tpch_sf10.duckdb
#   ./generate_tpch_dataset_legacy.sh 100 /data/tpch_sf100.duckdb 16
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scale_factor> [output_db] [jobs]"
    echo "Example: $0 1"
    exit 1
fi

SF="$1"
OUTPUT_DB="${2:-$PROJECT_DIR/test_datasets/tpch_sf${SF}.duckdb}"
JOBS="${3:-$(nproc)}"
TPCHGEN_DIR="$PROJECT_DIR/test_datasets/tpchgen-rs"
TBL_DIR="$PROJECT_DIR/test_datasets/tpch_tbl_sf${SF}"

if [ -f "$OUTPUT_DB" ]; then
    echo "Database already exists: $OUTPUT_DB"
    echo "Skipping generation. Remove the file to regenerate."
    exit 0
fi

# --- Validate prerequisites ---
if [ ! -x "$DUCKDB" ]; then
    echo "ERROR: DuckDB binary not found at $DUCKDB"
    echo "Build first: CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) make"
    exit 1
fi

# --- Step 1: Clone tpchgen-rs if not present ---
if [ ! -d "$TPCHGEN_DIR" ]; then
    echo "Cloning sirius-db/tpchgen-rs (bwy/sirius branch)..."
    git clone --branch bwy/sirius https://github.com/sirius-db/tpchgen-rs.git "$TPCHGEN_DIR"
else
    echo "tpchgen-rs already cloned at $TPCHGEN_DIR"
fi

# --- Step 2: Build tpchgen-cli from source ---
TPCHGEN_CLI="$TPCHGEN_DIR/target/release/tpchgen-cli"
if [ ! -f "$TPCHGEN_CLI" ]; then
    echo "Building tpchgen-cli with native CPU optimizations..."
    (cd "$TPCHGEN_DIR" && RUSTFLAGS="-C target-cpu=native" cargo build --release -p tpchgen-cli)
else
    echo "tpchgen-cli already built at $TPCHGEN_CLI"
fi

# --- Step 3: Generate TBL data ---
if [ -d "$TBL_DIR" ]; then
    echo "TBL directory already exists: $TBL_DIR"
    echo "Reusing existing TBL files."
else
    echo "Generating TPC-H SF${SF} TBL data with ${JOBS} threads..."
    mkdir -p "$TBL_DIR"
    "$TPCHGEN_CLI" -s "$SF" --format tbl --output-dir "$TBL_DIR" --num-threads "$JOBS" -v
    echo "TBL files generated in $TBL_DIR"
fi

# --- Step 4: Load TBL files into DuckDB ---
echo ""
echo "Loading TBL files into DuckDB database: $OUTPUT_DB"

"$DUCKDB" "$OUTPUT_DB" <<EOF
DROP TABLE IF EXISTS nation;
DROP TABLE IF EXISTS region;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS partsupp;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS lineitem;

CREATE TABLE nation  ( n_nationkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                       n_name       CHAR(25) NOT NULL,
                       n_regionkey  INTEGER NOT NULL,
                       n_comment    VARCHAR(152));

CREATE TABLE region  ( r_regionkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                       r_name       CHAR(25) NOT NULL,
                       r_comment    VARCHAR(152));

CREATE TABLE part  ( p_partkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                     p_name        VARCHAR(55) NOT NULL,
                     p_mfgr        CHAR(25) NOT NULL,
                     p_brand       CHAR(10) NOT NULL,
                     p_type        VARCHAR(25) NOT NULL,
                     p_size        INTEGER NOT NULL,
                     p_container   CHAR(10) NOT NULL,
                     p_retailprice DECIMAL(15,2) NOT NULL,
                     p_comment     VARCHAR(23) NOT NULL );

CREATE TABLE supplier ( s_suppkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        s_name        CHAR(25) NOT NULL,
                        s_address     VARCHAR(40) NOT NULL,
                        s_nationkey   INTEGER NOT NULL,
                        s_phone       CHAR(15) NOT NULL,
                        s_acctbal     DECIMAL(15,2) NOT NULL,
                        s_comment     VARCHAR(101) NOT NULL);

CREATE TABLE partsupp ( ps_partkey     INTEGER NOT NULL,
                        ps_suppkey     INTEGER NOT NULL,
                        ps_availqty    INTEGER NOT NULL,
                        ps_supplycost  DECIMAL(15,2)  NOT NULL,
                        ps_comment     VARCHAR(199) NOT NULL,
                        CONSTRAINT PS_PARTSUPPKEY UNIQUE(PS_PARTKEY, PS_SUPPKEY) );

CREATE TABLE customer ( c_custkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        c_name        VARCHAR(25) NOT NULL,
                        c_address     VARCHAR(40) NOT NULL,
                        c_nationkey   INTEGER NOT NULL,
                        c_phone       CHAR(15) NOT NULL,
                        c_acctbal     DECIMAL(15,2)   NOT NULL,
                        c_mktsegment  CHAR(10) NOT NULL,
                        c_comment     VARCHAR(117) NOT NULL);

CREATE TABLE orders  ( o_orderkey       BIGINT NOT NULL UNIQUE PRIMARY KEY,
                       o_custkey        INTEGER NOT NULL,
                       o_orderstatus    CHAR(1) NOT NULL,
                       o_totalprice     DECIMAL(15,2) NOT NULL,
                       o_orderdate      DATE NOT NULL,
                       o_orderpriority  CHAR(15) NOT NULL,
                       o_clerk          CHAR(15) NOT NULL,
                       o_shippriority   INTEGER NOT NULL,
                       o_comment        VARCHAR(79) NOT NULL);

CREATE TABLE lineitem ( l_orderkey    BIGINT NOT NULL,
                        l_partkey     INTEGER NOT NULL,
                        l_suppkey     INTEGER NOT NULL,
                        l_linenumber  INTEGER NOT NULL,
                        l_quantity    DECIMAL(15,2) NOT NULL,
                        l_extendedprice  DECIMAL(15,2) NOT NULL,
                        l_discount    DECIMAL(15,2) NOT NULL,
                        l_tax         DECIMAL(15,2) NOT NULL,
                        l_returnflag  CHAR(1) NOT NULL,
                        l_linestatus  CHAR(1) NOT NULL,
                        l_shipdate    DATE NOT NULL,
                        l_commitdate  DATE NOT NULL,
                        l_receiptdate DATE NOT NULL,
                        l_shipinstruct CHAR(25) NOT NULL,
                        l_shipmode     CHAR(10) NOT NULL,
                        l_comment      VARCHAR(44) NOT NULL);

COPY nation FROM '${TBL_DIR}/nation.tbl' WITH (HEADER false, DELIMITER '|');
COPY region FROM '${TBL_DIR}/region.tbl' WITH (HEADER false, DELIMITER '|');
COPY part FROM '${TBL_DIR}/part.tbl' WITH (HEADER false, DELIMITER '|');
COPY supplier FROM '${TBL_DIR}/supplier.tbl' WITH (HEADER false, DELIMITER '|');
COPY partsupp FROM '${TBL_DIR}/partsupp.tbl' WITH (HEADER false, DELIMITER '|');
COPY customer FROM '${TBL_DIR}/customer.tbl' WITH (HEADER false, DELIMITER '|');
COPY orders FROM '${TBL_DIR}/orders.tbl' WITH (HEADER false, DELIMITER '|');
COPY lineitem FROM '${TBL_DIR}/lineitem.tbl' WITH (HEADER false, DELIMITER '|');

SELECT 'region' as tbl, count(*) as rows FROM region
UNION ALL SELECT 'nation', count(*) FROM nation
UNION ALL SELECT 'part', count(*) FROM part
UNION ALL SELECT 'supplier', count(*) FROM supplier
UNION ALL SELECT 'partsupp', count(*) FROM partsupp
UNION ALL SELECT 'customer', count(*) FROM customer
UNION ALL SELECT 'orders', count(*) FROM orders
UNION ALL SELECT 'lineitem', count(*) FROM lineitem
ORDER BY tbl;
EOF

echo ""
echo "=========================================="
echo "TPC-H SF${SF} dataset generation complete."
echo "Database: $OUTPUT_DB"
echo "TBL files: $TBL_DIR"
echo "=========================================="
