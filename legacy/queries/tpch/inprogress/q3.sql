select
  l_orderkey,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  o_orderdate,
  o_shippriority
from
  customer,
  orders,
  lineitem
where
  c_mktsegment = 1
  and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < 19950315
  and l_shipdate > 19950315
group by
  l_orderkey,
  o_orderdate,
  o_shippriority
order by
  revenue desc,
  o_orderdate,
  l_orderkey;