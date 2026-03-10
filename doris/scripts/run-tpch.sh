#!/usr/bin/env bash
# Run TPC-H SF1 queries (Q1-Q22) against Sirius BEs via Doris FE.
#
# Each bare table name is replaced with a local() TVF pointing to partitioned
# parquet files (p16). For multi-BE mode, shared_storage=true distributes
# partitions across backends.
#
# Prerequisites: FE and BE(s) must already be running (see BUILD_DEPLOY_TEST_GUIDE.md).
#
# Usage:
#   ./doris/scripts/run-tpch.sh [--skip-build] [--bes 1|2] [--queries 1,2,6]
#   ./doris/scripts/run-tpch.sh --data-dir /data/tpch/sf10/p16/snappy --queries 1,6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
SKIP_BUILD=false
NUM_BES=1
DATA_DIR="/data/tpch/sf1/p16/snappy"
FE_HOST="127.0.0.1"
FE_HTTP_PORT=8030
QUERY_FILTER=""  # empty = all

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --bes)        NUM_BES="$2"; shift 2 ;;
        --queries)    QUERY_FILTER="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --fe-host)    FE_HOST="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

FE_HTTP="${FE_HOST}:${FE_HTTP_PORT}"

# Ensure python3 is available (pixi env or system)
if ! command -v python3 &>/dev/null; then
    PIXI_PYTHON="$PROJECT_ROOT/.pixi/envs/default/bin/python3"
    if [[ -x "$PIXI_PYTHON" ]]; then
        export PATH="$(dirname "$PIXI_PYTHON"):$PATH"
    else
        echo "ERROR: python3 not found. Run via pixi or install python3."
        exit 1
    fi
fi

# shared_storage=true even for 1 BE — FE requires backend_id when false
SHARED_STORAGE="true"

# Run a SQL statement via FE HTTP API, return JSON response
fe_query() {
    local sql="$1"
    local escaped
    escaped=$(echo "$sql" | sed 's/\\/\\\\/g; s/"/\\"/g')
    curl -sf "http://${FE_HTTP}/api/query/default_cluster/information_schema" \
        -u root: -H "Content-Type: application/json" \
        -d "{\"stmt\":\"$escaped\"}" 2>/dev/null
}

# ── Table name → local() TVF rewrite (Python) ──────────────────────────────
rewrite_query() {
    local sql="$1"
    python3 -c "
import re, sys

sql = sys.argv[1]
data_dir = sys.argv[2]
shared = sys.argv[3]

tables = {
    'partsupp': data_dir + '/partsupp/*.parquet',
    'lineitem': data_dir + '/lineitem/*.parquet',
    'customer': data_dir + '/customer/*.parquet',
    'supplier': data_dir + '/supplier/*.parquet',
    'nation':   data_dir + '/nation/*.parquet',
    'orders':   data_dir + '/orders/*.parquet',
    'region':   data_dir + '/region/*.parquet',
    'part':     data_dir + '/part/*.parquet',
}

# Replace longer names first to avoid partial matches (partsupp before part)
for name in sorted(tables.keys(), key=len, reverse=True):
    path = tables[name]
    tvf = f'local(\"file_path\"=\"{path}\", \"format\"=\"parquet\", \"shared_storage\"=\"{shared}\")'
    # Match table name preceded by FROM/JOIN keyword or comma (with whitespace)
    sql = re.sub(
        r'((?:\bfrom|\bjoin)\s+|,\s*)' + name + r'\b',
        r'\g<1>' + tvf.replace('\\\\', '\\\\\\\\'),
        sql,
        flags=re.IGNORECASE
    )

print(sql)
" "$sql" "$DATA_DIR" "$SHARED_STORAGE"
}

# ── TPC-H Queries ──────────────────────────────────────────────────────────
declare -a TPCH_QUERIES
TPCH_QUERIES[1]="select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem where l_shipdate <= date '1998-12-01' - interval '90' day group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus"
TPCH_QUERIES[2]="select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment from part, supplier, partsupp, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = 15 and p_type like '%BRASS' and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'EUROPE' and ps_supplycost = ( select min(ps_supplycost) from partsupp, supplier, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'EUROPE' ) order by s_acctbal desc, n_name, s_name, p_partkey limit 100"
TPCH_QUERIES[3]="select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority from customer, orders, lineitem where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15' group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate limit 10"
TPCH_QUERIES[4]="select o_orderpriority, count(*) as order_count from orders where o_orderdate >= date '1993-07-01' and o_orderdate < date '1993-07-01' + interval '3' month and exists ( select * from lineitem where l_orderkey = o_orderkey and l_commitdate < l_receiptdate ) group by o_orderpriority order by o_orderpriority"
TPCH_QUERIES[5]="select n_name, sum(l_extendedprice * (1 - l_discount)) as revenue from customer, orders, lineitem, supplier, nation, region where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey and c_nationkey = s_nationkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'ASIA' and o_orderdate >= date '1994-01-01' and o_orderdate < date '1994-01-01' + interval '1' year group by n_name order by revenue desc"
TPCH_QUERIES[6]="select sum(l_extendedprice * l_discount) as revenue from lineitem where l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year and l_discount between .06 - 0.01 and .06 + 0.01 and l_quantity < 24"
TPCH_QUERIES[7]="select supp_nation, cust_nation, l_year, sum(volume) as revenue from ( select n1.n_name as supp_nation, n2.n_name as cust_nation, extract(year from l_shipdate) as l_year, l_extendedprice * (1 - l_discount) as volume from supplier, lineitem, orders, customer, nation n1, nation n2 where s_suppkey = l_suppkey and o_orderkey = l_orderkey and c_custkey = o_custkey and s_nationkey = n1.n_nationkey and c_nationkey = n2.n_nationkey and ( (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE') ) and l_shipdate between date '1995-01-01' and date '1996-12-31' ) as shipping group by supp_nation, cust_nation, l_year order by supp_nation, cust_nation, l_year"
TPCH_QUERIES[8]="select o_year, sum(case when nation = 'BRAZIL' then volume else 0 end) / sum(volume) as mkt_share from ( select extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) as volume, n2.n_name as nation from part, supplier, lineitem, orders, customer, nation n1, nation n2, region where p_partkey = l_partkey and s_suppkey = l_suppkey and l_orderkey = o_orderkey and o_custkey = c_custkey and c_nationkey = n1.n_nationkey and n1.n_regionkey = r_regionkey and r_name = 'AMERICA' and s_nationkey = n2.n_nationkey and o_orderdate between date '1995-01-01' and date '1996-12-31' and p_type = 'ECONOMY ANODIZED STEEL' ) as all_nations group by o_year order by o_year"
TPCH_QUERIES[9]="select nation, o_year, sum(amount) as sum_profit from ( select n_name as nation, extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount from part, supplier, lineitem, partsupp, orders, nation where s_suppkey = l_suppkey and ps_suppkey = l_suppkey and ps_partkey = l_partkey and p_partkey = l_partkey and o_orderkey = l_orderkey and s_nationkey = n_nationkey and p_name like '%green%' ) as profit group by nation, o_year order by nation, o_year desc"
TPCH_QUERIES[10]="select c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) as revenue, c_acctbal, n_name, c_address, c_phone, c_comment from customer, orders, lineitem, nation where c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate >= date '1993-10-01' and o_orderdate < date '1993-10-01' + interval '3' month and l_returnflag = 'R' and c_nationkey = n_nationkey group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment order by revenue desc limit 20"
TPCH_QUERIES[11]="select ps_partkey, sum(ps_supplycost * ps_availqty) as value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.000002 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY' ) order by value desc"
TPCH_QUERIES[12]="select l_shipmode, sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count, sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count from orders, lineitem where o_orderkey = l_orderkey and l_shipmode in ('MAIL', 'SHIP') and l_commitdate < l_receiptdate and l_shipdate < l_commitdate and l_receiptdate >= date '1994-01-01' and l_receiptdate < date '1994-01-01' + interval '1' year group by l_shipmode order by l_shipmode"
TPCH_QUERIES[13]="select c_count, count(*) as custdist from ( select c_custkey, count(o_orderkey) as c_count from customer left outer join orders on c_custkey = o_custkey and o_comment not like '%special%requests%' group by c_custkey ) as c_orders group by c_count order by custdist desc, c_count desc"
TPCH_QUERIES[14]="select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice * (1 - l_discount) else 0 end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue from lineitem, part where l_partkey = p_partkey and l_shipdate >= date '1995-09-01' and l_shipdate < date '1995-09-01' + interval '1' month"
TPCH_QUERIES[15]="with revenue0 as ( select l_suppkey supplier_no, sum(l_extendedprice * (1 - l_discount)) total_revenue from lineitem where l_shipdate >= date '1996-01-01' and l_shipdate < date '1996-01-01' + interval '3' month group by l_suppkey ) select s_suppkey, s_name, s_address, s_phone, total_revenue from supplier, revenue0 where s_suppkey = supplier_no and total_revenue = ( select max(total_revenue) from revenue0 ) order by s_suppkey"
TPCH_QUERIES[16]="select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt from partsupp, part where p_partkey = ps_partkey and p_brand <> 'Brand#45' and p_type not like 'MEDIUM POLISHED%' and p_size in (49, 14, 23, 45, 19, 3, 36, 9) and ps_suppkey not in ( select s_suppkey from supplier where s_comment like '%Customer%Complaints%' ) group by p_brand, p_type, p_size order by supplier_cnt desc, p_brand, p_type, p_size"
TPCH_QUERIES[17]="select sum(l_extendedprice) / 7.0 as avg_yearly from lineitem, part where p_partkey = l_partkey and p_brand = 'Brand#23' and p_container = 'MED BOX' and l_quantity < ( select 0.2 * avg(l_quantity) from lineitem where l_partkey = p_partkey )"
TPCH_QUERIES[18]="select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) from customer, orders, lineitem where o_orderkey in ( select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300 ) and c_custkey = o_custkey and o_orderkey = l_orderkey group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice order by o_totalprice desc, o_orderdate limit 100"
TPCH_QUERIES[19]="select sum(l_extendedprice* (1 - l_discount)) as revenue from lineitem, part where ( p_partkey = l_partkey and p_brand = 'Brand#12' and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') and l_quantity >= 1 and l_quantity <= 1 + 10 and p_size between 1 and 5 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' ) or ( p_partkey = l_partkey and p_brand = 'Brand#23' and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') and l_quantity >= 10 and l_quantity <= 10 + 10 and p_size between 1 and 10 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' ) or ( p_partkey = l_partkey and p_brand = 'Brand#34' and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') and l_quantity >= 20 and l_quantity <= 20 + 10 and p_size between 1 and 15 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' )"
TPCH_QUERIES[20]="select s_name, s_address from supplier, nation where s_suppkey in ( select ps_suppkey from partsupp where ps_partkey in ( select p_partkey from part where p_name like 'forest%' ) and ps_availqty > ( select 0.5 * sum(l_quantity) from lineitem where l_partkey = ps_partkey and l_suppkey = ps_suppkey and l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year ) ) and s_nationkey = n_nationkey and n_name = 'CANADA' order by s_name"
TPCH_QUERIES[21]="select s_name, count(*) as numwait from supplier, lineitem l1, orders, nation where s_suppkey = l1.l_suppkey and o_orderkey = l1.l_orderkey and o_orderstatus = 'F' and l1.l_receiptdate > l1.l_commitdate and exists ( select * from lineitem l2 where l2.l_orderkey = l1.l_orderkey and l2.l_suppkey <> l1.l_suppkey ) and not exists ( select * from lineitem l3 where l3.l_orderkey = l1.l_orderkey and l3.l_suppkey <> l1.l_suppkey and l3.l_receiptdate > l3.l_commitdate ) and s_nationkey = n_nationkey and n_name = 'SAUDI ARABIA' group by s_name order by numwait desc, s_name limit 100"
TPCH_QUERIES[22]="select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal from ( select substring(c_phone, 1, 2) as cntrycode, c_acctbal from customer where substring(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17') and c_acctbal > ( select avg(c_acctbal) from customer where c_acctbal > 0.00 and substring(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17') ) and not exists ( select * from orders where o_custkey = c_custkey ) ) as custsale group by cntrycode order by cntrycode"

# ── Build ────────────────────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
    echo "==> Building Rust BE..."
    pixi run -e doris doris-build
fi

# ── Verify cluster ──────────────────────────────────────────────────────────
echo "==> Checking FE health..."
for i in $(seq 1 30); do
    if curl -sf "http://${FE_HTTP}/api/health" 2>/dev/null | grep -q success; then
        echo "    FE healthy"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "ERROR: FE not reachable at ${FE_HTTP}. Start it first:"
        echo "  pixi run -e doris-fe doris-fe"
        exit 1
    fi
    sleep 1
done

echo "==> Checking for ${NUM_BES} alive BE(s)..."
for i in $(seq 1 30); do
    ALIVE_COUNT=$(fe_query "SHOW BACKENDS" 2>/dev/null \
        | tr -d '\n' | grep -o '"true"' | wc -l || echo 0)
    ALIVE_COUNT=$(echo "$ALIVE_COUNT" | tr -d '[:space:]')
    if [[ "$ALIVE_COUNT" -ge "$NUM_BES" ]]; then
        echo "    $ALIVE_COUNT BE(s) alive"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "ERROR: Only $ALIVE_COUNT BEs alive (expected $NUM_BES). Start BE(s) first:"
        echo "  pixi run -e doris sirius-be"
        exit 1
    fi
    sleep 1
done

# ── Query runner ─────────────────────────────────────────────────────────────
PASS_COUNT=0
FAIL_COUNT=0
TIMEOUT_COUNT=0

run_tpch_query() {
    local qnum="$1"
    local sql="$2"
    local timeout="${3:-120}"

    # Rewrite table references to local() TVF
    local rewritten
    rewritten=$(rewrite_query "$sql")

    printf "\n── Q%-2d ──────────────────────────────────────────────────────────\n" "$qnum"

    # Escape for JSON (double quotes → \", backslashes → \\)
    local escaped
    escaped=$(echo "$rewritten" | sed 's/\\/\\\\/g; s/"/\\"/g')

    local start_ts
    start_ts=$(date +%s%3N)

    local result
    result=$(timeout "$timeout" curl -sf \
        "http://${FE_HTTP}/api/query/default_cluster/information_schema" \
        -u root: -H "Content-Type: application/json" \
        -d "{\"stmt\":\"$escaped\"}" 2>/dev/null) || {
        local exit_code=$?
        local end_ts
        end_ts=$(date +%s%3N)
        local elapsed=$(( end_ts - start_ts ))
        if [[ $exit_code -eq 124 ]]; then
            printf "  TIMEOUT (%ds limit, %d.%03ds elapsed)\n" "$timeout" $((elapsed/1000)) $((elapsed%1000))
            TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
        else
            printf "  ERROR   (exit=%d, %d.%03ds)\n" "$exit_code" $((elapsed/1000)) $((elapsed%1000))
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        wait_for_be
        return
    }

    local end_ts
    end_ts=$(date +%s%3N)
    local elapsed=$(( end_ts - start_ts ))

    # Parse response with python
    local status
    status=$(echo "$result" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    if d.get('msg') == 'success' and 'data' in d and isinstance(d['data'], dict) and 'data' in d['data']:
        rows = d['data']['data']
        print(f'OK:{len(rows)} rows')
    elif d.get('msg') == 'success':
        print('OK:0 rows')
    else:
        msg = d.get('msg', 'unknown')
        detail = str(d.get('data', ''))[:200]
        print(f'ERR:{msg}: {detail}')
except Exception as e:
    print(f'ERR:parse error: {e}')
" 2>/dev/null)

    if [[ "$status" == OK:* ]]; then
        local row_info="${status#OK:}"
        printf "  PASS    %s  (%d.%03ds)\n" "$row_info" $((elapsed/1000)) $((elapsed%1000))

        # Print first few rows for verification
        echo "$result" | python3 -c "
import json, sys
d = json.load(sys.stdin)
rows = d.get('data',{}).get('data',[])
meta = d.get('data',{}).get('meta',[])
cols = [m['name'] for m in meta] if meta else []
if cols:
    print('  Columns:', ', '.join(cols[:8]), '...' if len(cols) > 8 else '')
for i, r in enumerate(rows[:3]):
    vals = [str(v)[:30] for v in r[:6]]
    print(f'  [{i}] {vals}')
if len(rows) > 3:
    print(f'  ... ({len(rows)} total rows)')
" 2>/dev/null
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        local err_info="${status#ERR:}"
        printf "  FAIL    %s  (%d.%03ds)\n" "$err_info" $((elapsed/1000)) $((elapsed%1000))
        FAIL_COUNT=$((FAIL_COUNT + 1))
        wait_for_be
    fi
}

count_alive_bes() {
    local alive
    alive=$(fe_query "SHOW BACKENDS" 2>/dev/null \
        | tr -d '\n' | grep -o '"true"' | wc -l || echo 0)
    echo "$alive" | tr -d '[:space:]'
}

wait_for_be_alive() {
    local required="${1:-1}"
    local max_wait="${2:-90}"
    for i in $(seq 1 "$max_wait"); do
        if [[ "$(count_alive_bes)" -ge "$required" ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# Called after a query failure — wait for BE to recover (it may have crashed).
wait_for_be() {
    printf "  (waiting for BE to recover...)\n"
    if wait_for_be_alive 1 60; then
        printf "  (BE alive)\n"
        sleep 2
    else
        printf "  WARNING: BE not recovered after 60s\n"
    fi
}

# Called before each query. Verifies at least 1 BE is alive.
ensure_be_running() {
    if [[ "$(count_alive_bes)" -ge 1 ]]; then
        return
    fi
    printf "  (waiting for BE registration...)\n"
    if wait_for_be_alive 1 30; then
        sleep 2
        return
    fi
    printf "  WARNING: no alive BEs detected\n"
}

# ── Determine which queries to run ──────────────────────────────────────────
get_query_list() {
    if [[ -n "$QUERY_FILTER" ]]; then
        echo "$QUERY_FILTER" | tr ',' ' '
    else
        seq 1 22
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
# Increase query timeout for complex queries
fe_query "SET GLOBAL query_timeout = 300" >/dev/null 2>&1 || true

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " TPC-H SF1 (p16 partitioned) — ${NUM_BES} BE(s)"
echo "════════════════════════════════════════════════════════════════"

for q in $(get_query_list); do
    if [[ -z "${TPCH_QUERIES[$q]+x}" ]]; then
        echo "  Q${q}: unknown query, skipping"
        continue
    fi
    ensure_be_running
    run_tpch_query "$q" "${TPCH_QUERIES[$q]}" 120
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Summary: PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  TIMEOUT=$TIMEOUT_COUNT"
echo "════════════════════════════════════════════════════════════════"
