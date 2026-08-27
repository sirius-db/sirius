#!/usr/bin/env bash
# Smoke-test the 8-CN Sirius cluster against TPC-H SF500 parquet.
#
# Covers four tiers in order of complexity:
#   T1 — cluster health (no query)
#   T2 — single-fragment scan/agg (one CN, no exchange)
#   T3 — multi-CN exchange (partial agg → hash fanout → merge)
#   T4 — join + exchange (hash join + GROUP BY)
#
# Usage:
#   ./benchmarks/nixl-nvlink/smoke-test.sh                        # SF500 default
#   DATA=/home/ubuntu/tpch_parquet_sf1000 ./benchmarks/nixl-nvlink/smoke-test.sh
#
# Requires: cluster running (./benchmarks/nixl-nvlink/script-box.sh in another terminal)
set -euo pipefail

DATA=${DATA:-/home/ubuntu/tpch_parquet_sf500}
MYSQL="mysql --host 127.0.0.1 --port 9030 --user root --connect-timeout 10"
PASS=0; FAIL=0; SKIP=0

# ── helpers ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

run_query() {
    local label="$1" sql="$2"
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  $label"
    echo "────────────────────────────────────────────────────────────"
    local t0; t0=$(date +%s%3N)
    local out; out=$(cd /home/ubuntu/sirius/experimental/starrocks && \
        pixi run $MYSQL -e "$sql" 2>&1) && local rc=0 || local rc=$?
    local t1; t1=$(date +%s%3N)
    local ms=$(( t1 - t0 ))

    if [ $rc -eq 0 ]; then
        echo "$out"
        printf "${GREEN}  PASS${RESET}  %d ms\n" "$ms"
        PASS=$(( PASS + 1 ))
    else
        echo "$out" >&2
        printf "${RED}  FAIL${RESET}  rc=%d  %d ms\n" "$rc" "$ms"
        FAIL=$(( FAIL + 1 ))
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Sirius Engine-A Smoke Test  ·  DATA=$(basename $DATA)      "
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  FE: 127.0.0.1:9030"

# ── T1: cluster health ────────────────────────────────────────────────────────

echo ""
echo "━━━  T1: Cluster health  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NODES=$(cd /home/ubuntu/sirius/experimental/starrocks && \
    pixi run $MYSQL -e "SHOW COMPUTE NODES;" 2>&1)
ALIVE=$(echo "$NODES" | grep -c "true" || true)
TOTAL=$(echo "$NODES" | tail -n +2 | grep -c "127.0.0.1" || true)

if [ "$ALIVE" -eq "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
    printf "${GREEN}  PASS${RESET}  %d/%d CNs alive\n" "$ALIVE" "$TOTAL"
    PASS=$(( PASS + 1 ))
else
    printf "${RED}  FAIL${RESET}  only %d/%d CNs alive\n" "$ALIVE" "$TOTAL"
    echo "$NODES"
    FAIL=$(( FAIL + 1 ))
fi

# ── T2: single-fragment queries ───────────────────────────────────────────────

echo ""
echo "━━━  T2: Single-fragment scan / agg (no cross-CN exchange)  ━━"

# T2a: tiny table — 5-row region table, proves FILES() + GPU scan path
run_query "T2a · region COUNT (5 rows expected)" \
"WITH r AS (SELECT * FROM FILES(
  'path'='file://${DATA}/region/*.parquet',
  'format'='parquet'))
SELECT count(*) AS n_regions FROM r;"

# T2b: supplier filter + count — ~500 K rows at SF500, single agg
run_query "T2b · supplier filter+count" \
"WITH s AS (SELECT * FROM FILES(
  'path'='file://${DATA}/supplier/*.parquet',
  'format'='parquet'))
SELECT count(*) AS n_us_suppliers
FROM s
WHERE s_comment NOT LIKE '%Customer%Complaints%';"

# T2c: lineitem raw count — proves full scan distributes across CNs
run_query "T2c · lineitem COUNT(*) — full SF500 scan" \
"WITH l AS (SELECT * FROM FILES(
  'path'='file://${DATA}/lineitem/*.parquet',
  'format'='parquet'))
SELECT count(*) AS n_rows FROM l;"

# ── T3: multi-CN exchange ─────────────────────────────────────────────────────

echo ""
echo "━━━  T3: Multi-CN exchange (partial agg → hash fanout → merge)  ━━"

# T3a: Q06 — filter + sum, single-fragment output but exercises all CNs for scan
run_query "T3a · Q06 shape — revenue filter sum" \
"WITH l AS (SELECT * FROM FILES(
  'path'='file://${DATA}/lineitem/*.parquet',
  'format'='parquet'))
SELECT sum(l_extendedprice * l_discount) AS revenue
FROM l
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate <  DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;"

# T3b: Q01 — the canonical multi-CN exchange query:
#   partial agg per CN → hash-partition by (returnflag, linestatus) → merge agg
run_query "T3b · Q01 — GROUP BY returnflag/linestatus (4 rows expected)" \
"WITH l AS (SELECT * FROM FILES(
  'path'='file://${DATA}/lineitem/*.parquet',
  'format'='parquet'))
SELECT
  l_returnflag,
  l_linestatus,
  sum(l_quantity)                                        AS sum_qty,
  sum(l_extendedprice)                                   AS sum_base_price,
  sum(l_extendedprice * (1 - l_discount))                AS sum_disc_price,
  sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
  avg(l_quantity)                                        AS avg_qty,
  avg(l_extendedprice)                                   AS avg_price,
  avg(l_discount)                                        AS avg_disc,
  count(*)                                               AS count_order
FROM l
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;"

# T3c: Q12 shape — date filter + GROUP BY shipmode (2 groups if dates hit)
run_query "T3c · Q12 shape — late arrivals by shipmode" \
"WITH l AS (SELECT * FROM FILES(
  'path'='file://${DATA}/lineitem/*.parquet',
  'format'='parquet')),
orders AS (SELECT * FROM FILES(
  'path'='file://${DATA}/orders/*.parquet',
  'format'='parquet'))
SELECT
  l_shipmode,
  sum(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
  sum(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
FROM orders, l
WHERE o_orderkey = l_orderkey
  AND l_shipmode IN ('MAIL', 'SHIP')
  AND l_commitdate < l_receiptdate
  AND l_shipdate   < l_commitdate
  AND l_receiptdate >= DATE '1994-01-01'
  AND l_receiptdate <  DATE '1995-01-01'
GROUP BY l_shipmode
ORDER BY l_shipmode;"

# ── T4: join + exchange ───────────────────────────────────────────────────────

echo ""
echo "━━━  T4: Join + exchange  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# T4a: Q04 — EXISTS semijoin: orders with at least one late lineitem
run_query "T4a · Q04 shape — order priority with late lineitems" \
"WITH orders AS (SELECT * FROM FILES(
  'path'='file://${DATA}/orders/*.parquet',
  'format'='parquet')),
l AS (SELECT * FROM FILES(
  'path'='file://${DATA}/lineitem/*.parquet',
  'format'='parquet'))
SELECT
  o_orderpriority,
  count(*) AS order_count
FROM orders
WHERE o_orderdate >= DATE '1993-07-01'
  AND o_orderdate <  DATE '1993-10-01'
  AND EXISTS (
    SELECT 1 FROM l
    WHERE l_orderkey = o_orderkey
      AND l_commitdate < l_receiptdate)
GROUP BY o_orderpriority
ORDER BY o_orderpriority;"

# T4b: Q16 — part/supplier anti-semijoin, GROUP BY brand + type + size
#   (small result set, exercises multi-table hash exchange)
run_query "T4b · Q16 shape — supplier count by part brand/type/size" \
"WITH p AS (SELECT * FROM FILES(
  'path'='file://${DATA}/part/*.parquet',
  'format'='parquet')),
ps AS (SELECT * FROM FILES(
  'path'='file://${DATA}/partsupp/*.parquet',
  'format'='parquet')),
s AS (SELECT * FROM FILES(
  'path'='file://${DATA}/supplier/*.parquet',
  'format'='parquet'))
SELECT
  p_brand,
  p_type,
  p_size,
  count(DISTINCT ps_suppkey) AS supplier_cnt
FROM ps, p
WHERE p_partkey = ps_partkey
  AND p_brand <> 'Brand#45'
  AND p_type NOT LIKE 'MEDIUM POLISHED%'
  AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
  AND ps_suppkey NOT IN (
    SELECT s_suppkey FROM s
    WHERE s_comment LIKE '%Customer%Complaints%')
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size
LIMIT 10;"

# ── summary ───────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
printf "║  Results:  ${GREEN}%d PASS${RESET}  /  ${RED}%d FAIL${RESET}  /  ${YELLOW}%d SKIP${RESET}                    ║\n" \
    "$PASS" "$FAIL" "$SKIP"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

[ "$FAIL" -eq 0 ]
