select
  c_custkey,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  n_nationkey,
from
  customer,
  orders,
  lineitem,
  nation
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate >= 19931001
  and o_orderdate <= 19931231
  and l_returnflag = 0
  and c_nationkey = n_nationkey
group by
  c_custkey,
  n_nationkey
order by
  c_custkey;