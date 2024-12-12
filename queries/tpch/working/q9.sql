select
  nation,
  o_year,
  sum(amount) as sum_profit
from(
  select
    n_nationkey as nation,
    o_orderdate//10000 as o_year,
    l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
  from
    part,
    supplier,
    lineitem,
    partsupp,
    orders,
    nation
  where
    s_suppkey = l_suppkey
    and ps_suppkey = l_suppkey
    and ps_partkey = l_partkey
    and p_partkey = l_partkey
    and o_orderkey = l_orderkey
    and s_nationkey = n_nationkey
    and p_size = 15
  ) as profit
group by
  nation,
  o_year;