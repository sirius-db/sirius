#!/usr/bin/env bash
# Loads the TPC-H parquet dataset into Doris internal (OLAP) tables for the benchmark's
# optional reference C (plan-doc experiments/sf10-bench §3: "Doris on its home turf"):
# the official DDL of doris/tools/tpch-tools/ddl (duplicate-key tables, hash buckets, colocated
# lineitem/orders and part/partsupp, one replica), INSERT INTO ... SELECT from the parquet
# views of sql/tpch-views.sql, then ANALYZE ... WITH SYNC as tools/tpch-tools/bin/load-tpch-data.sh
# does. The tables live in database tpch_olap (run-tpch.sh --db tpch_olap); revenue0 (Q15) is
# a view over tpch_olap.lineitem.
#
#   scripts/olap-load.sh --data DIR [--sf N] [--db tpch_olap]
#
#   --data DIR   parquet dataset root (the views are re-pointed at it)
#   --sf N       picks the DDL: the sf1 file (fewer buckets) below 100, sf100 from 100 on
#                (default: from the dataset name, else 1)
#
# Needs the FE, and the native BE alive (scripts/be-native.sh start); the Sirius backend must
# not be alive (it cannot take tablets). Idempotent: drops the database (FORCE) and recreates
# the tables.
set -euo pipefail

cd "$(dirname "$0")/.."
MYSQL=("${MYSQL_BIN:-$(command -v mysql || echo "$(pwd)/.pixi/envs/fe/bin/mysql")}" -h 127.0.0.1 -P "${FE_QUERY_PORT:-9030}" -u root)
DATA=""
SF=""
DB="tpch_olap"
while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --sf) SF="$2"; shift 2 ;;
        --db) DB="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "${DATA}" ] || { echo "error: --data DIR is required" >&2; exit 2; }
DATA="$(cd "${DATA}" && pwd)"
[ -n "${SF}" ] || SF=$(basename "${DATA}" | grep -o 'sf[0-9]*' | head -1 | tr -d 'sf' || true)
[ -n "${SF}" ] || SF=1
if [ "${SF}" -ge 100 ]; then ddl=doris/tools/tpch-tools/ddl/create-tpch-tables-sf100.sql; else ddl=doris/tools/tpch-tools/ddl/create-tpch-tables-sf1.sql; fi
[ -f "${ddl}" ] || { echo "error: ${ddl} missing (git submodule update --init doris)" >&2; exit 1; }

sql() { "${MYSQL[@]}" -e "$1"; }
# A registered Sirius backend (even a dead one) pins the FE's auto parallelism to 1 instance
# (see scripts/bench.sh drop_sirius_be); it holds no tablets, so dropping it is free and it
# registers itself again on its next start.
if "${MYSQL[@]}" -N -e "SHOW BACKENDS" | awk -F'\t' '$3 == 9050 { found = 1 } END { exit !found }'; then
    echo "==> dropping the Sirius backend (127.0.0.1:9050) from the FE for the load"
    bash scripts/be.sh stop >/dev/null 2>&1 || true
    sql "ALTER SYSTEM DROPP BACKEND '127.0.0.1:9050'"
fi
echo "==> views over ${DATA}, tables from ${ddl} into ${DB}"
sed "s|@@TPCH_DIR@@|${DATA}|g" sql/tpch-views.sql | "${MYSQL[@]}"
# FORCE: the DDL's own DROP TABLE IF EXISTS leaves the old tables in the FE's recycle bin,
# where they keep their colocate groups alive — and a group's bucket count is fixed, so
# loading the sf100 DDL (96 buckets) after the sf1 one (32) fails with "Colocate tables
# must have same bucket num".
sql "DROP DATABASE IF EXISTS ${DB} FORCE"
sql "CREATE DATABASE ${DB}"
"${MYSQL[@]}" -D "${DB}" < "${ddl}"
"${MYSQL[@]}" < sql/session-native.sql

start=$(date +%s)
# Column lists in the DDL's order (the parquet order is the TPC-H one; the DDL leads with the
# duplicate key columns).
load() { # table select-list
    local t0
    t0=$(date +%s)
    echo "  loading ${DB}.$1 ..."
    sql "INSERT INTO ${DB}.$1 SELECT $2 FROM tpch.$1"
    echo "  ${DB}.$1: $(sql "SELECT count(*) FROM ${DB}.$1" | tail -1) rows in $(( $(date +%s) - t0 )) s"
}
load nation   "n_nationkey, n_name, n_regionkey, n_comment"
load region   "r_regionkey, r_name, r_comment"
load supplier "s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment"
load customer "c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment"
load part     "p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment"
load partsupp "ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment"
load orders   "o_orderkey, o_orderdate, o_custkey, o_orderstatus, o_totalprice, o_orderpriority, o_clerk, o_shippriority, o_comment"
load lineitem "l_shipdate, l_orderkey, l_linenumber, l_partkey, l_suppkey, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment"
echo "==> loaded in $(( $(date +%s) - start )) s"

sql "CREATE OR REPLACE VIEW ${DB}.revenue0 (supplier_no, total_revenue) AS
SELECT l_suppkey, sum(l_extendedprice * (1 - l_discount))
FROM ${DB}.lineitem
WHERE l_shipdate >= date '1996-01-01' AND l_shipdate < date '1996-01-01' + interval '3' month
GROUP BY l_suppkey"

start=$(date +%s)
echo "==> ANALYZE DATABASE ${DB} WITH FULL WITH SYNC"
sql "ANALYZE DATABASE ${DB} WITH FULL WITH SYNC"
echo "==> analyzed in $(( $(date +%s) - start )) s"
sql "SHOW DATA FROM ${DB}.lineitem" | tail -3
