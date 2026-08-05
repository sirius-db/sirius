#!/usr/bin/env bash
# Pin-once variant of run.sh: pin all tables ONCE per session instead of per query.
# Measured 2026-08-03 on the same binary as run.sh's 6.918 s: 7.866 s (+13.7%),
# 22/22 byte-identical. The premium is C2C traffic for host-resident columns
# (q2 +127%, q9 +29%); note the harness metric EXCLUDES pin time, so per-query
# mode also pays ~25 s of wall-clock re-pinning that this mode pays once —
# pin-once is a latency-vs-throughput trade, not a slower engine.
#
# Layout (GPU pool is 237 GiB; the full union does NOT fit GPU-resident —
# pinned memory is not evictable and q9/q13/q18 then OOM to CPU, 61.4 s):
#   GPU  'lineitem'        14 cols compressed        133.8 GiB
#   GPU  'orders'           7 cols compressed, no o_comment
#   HOST 'orders_comment'  {o_orderkey,o_custkey,o_comment} uncompressed (serves q13)
#   HOST part/partsupp/supplier/customer/nation/region uncompressed
# Two 'orders' entries are the only way to tier columns of one table today:
# _pinned_entries is keyed by pin name and try_match_cached_entry takes the
# first column-superset entry. q7/q22 are subsets of both orders entries and
# land on either nondeterministically (~50 ms / ~16 ms if host).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:-$HOME/tpch_parquet_sf1000}"
CUDF_SO="${CUDF_SO:-$HOME/cudf-src/cpp/build/libcudf.so}"
PLANS="${PLANS:-$HERE/plans}"
CFG="${CFG:-$HERE/sirius-sf1000.yaml}"
NAME="${NAME:-sf1000_pinonce}"

[ -d "$DATA" ]    || { echo "ERROR: no SF1000 parquet at $DATA (set DATA=)"; exit 1; }
[ -f "$CUDF_SO" ] || { echo "ERROR: no patched libcudf at $CUDF_SO -- run build-libcudf.sh first"; exit 1; }

export LD_PRELOAD="$CUDF_SO"
export SIRIUS_EXP_FUSED_SCAN_FILTER=1

# --pin none disables the harness's own pinning; PRE_SQL below is the whole pin
# sequence and runs once per connection (sequential mode = one connection).
# 'orders_comment' deliberately matches no plan-file stem, so it pins uncompressed.
export SIRIUS_PRE_SQL="\
SET pin_table_compression = true; \
SET pin_table_input_compression_plan_dir = '$PLANS'; \
SET expression_evaluator_strategy = 'ast_jit'; \
CALL pin_table('$DATA/lineitem/*.parquet', tier='gpu', name='lineitem', cols=['l_commitdate','l_discount','l_extendedprice','l_linestatus','l_orderkey','l_partkey','l_quantity','l_receiptdate','l_returnflag','l_shipdate','l_shipinstruct','l_shipmode','l_suppkey','l_tax']); \
CALL pin_table('$DATA/orders/*.parquet', tier='gpu', name='orders', cols=['o_custkey','o_orderdate','o_orderkey','o_orderpriority','o_orderstatus','o_shippriority','o_totalprice']); \
CALL pin_table('$DATA/orders/*.parquet', tier='host', name='orders_comment', cols=['o_comment','o_custkey','o_orderkey']); \
CALL pin_table('$DATA/part/*.parquet', tier='host', name='part', cols=['p_brand','p_container','p_mfgr','p_name','p_partkey','p_size','p_type']); \
CALL pin_table('$DATA/supplier/*.parquet', tier='host', name='supplier', cols=['s_acctbal','s_address','s_comment','s_name','s_nationkey','s_phone','s_suppkey']); \
CALL pin_table('$DATA/partsupp/*.parquet', tier='host', name='partsupp', cols=['ps_availqty','ps_partkey','ps_suppkey','ps_supplycost']); \
CALL pin_table('$DATA/nation/*.parquet', tier='host', name='nation', cols=['n_name','n_nationkey','n_regionkey']); \
CALL pin_table('$DATA/region/*.parquet', tier='host', name='region', cols=['r_name','r_regionkey']); \
CALL pin_table('$DATA/customer/*.parquet', tier='host', name='customer', cols=['c_acctbal','c_address','c_comment','c_custkey','c_mktsegment','c_name','c_nationkey','c_phone'])"

echo "data      : $DATA"
echo "libcudf   : $CUDF_SO"
echo "plans     : $PLANS"
echo "config    : $CFG"
echo

cd "$REPO"
python3 test/tpch_performance/performance_test.py \
  --input "$DATA" \
  --mode sequential --iterations 3 --engine gpu --pin none \
  --queries 1-22 --config "$CFG" --name "$NAME"
