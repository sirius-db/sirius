#!/usr/bin/env bash
# Pin lineitem + orders on every alive CN via ADMIN EXECUTE, with retries: right
# after a cluster restart the FE's brpc client can fail with 'Unable to validate
# object' before the channel is usable, which would silently leave a CN unpinned.
#
# The column lists are the TPC-H-used subsets that fit the memory budget:
# l_shipmode (compresses only 1.7x, ~49 GB at SF1000) and the two comment columns
# are left out — q12/q13/q19 then miss the pin BY DESIGN. On a box with more
# memory per GPU (e.g. GB200), add them back and switch PIN_TIER=gpu.
#
# Env: TPCH_DATA (required), PIN_TIER (host|gpu, default host)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SR_DIR="$(cd "$HERE/../.." && pwd)"
D=${TPCH_DATA:?set TPCH_DATA to the parquet root (<table>/*.parquet)}
TIER=${PIN_TIER:-host}
LCOLS=l_orderkey,l_partkey,l_suppkey,l_quantity,l_extendedprice,l_discount,l_tax,l_returnflag,l_linestatus,l_shipdate,l_commitdate,l_receiptdate,l_shipinstruct
OCOLS=o_orderkey,o_custkey,o_orderstatus,o_totalprice,o_orderdate,o_orderpriority,o_shippriority

cd "$SR_DIR"
pixi run -e client bash -c "
ids=\$(mysql -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;' | awk -F'\t' '\$9==\"true\" {print \$1}')
[ -n \"\$ids\" ] || { echo 'pin-all: no alive CNs' >&2; exit 1; }
fail=0
for id in \$ids; do
  ( t0=\$(date +%s); ok=0
    for try in 1 2 3 4 5; do
      out=\$(mysql -h127.0.0.1 -P9030 -uroot -N -e \"ADMIN EXECUTE ON \$id 'pin_table path=$D/lineitem/*.parquet tier=$TIER name=lineitem cols=$LCOLS
pin_table path=$D/orders/*.parquet tier=$TIER name=orders cols=$OCOLS';\" 2>&1) && { ok=1; break; }
      echo \"PIN CN\$id try \$try failed: \$out\" >&2; sleep 15
    done
    echo \"\$out\"
    echo \"PIN CN\$id: \$(( \$(date +%s) - t0 ))s ok=\$ok\"
    [ \$ok = 1 ] ) &
done
wait || fail=1
exit \$fail"
