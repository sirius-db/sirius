select
  c_custkey,
  c_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  c_acctbal,
  n_name,
  c_address,
  c_phone,
  c_comment
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
  c_name,
  c_acctbal,
  c_phone,
  n_name,
  c_address,
  c_comment;


select
  c_custkey,
  c_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  c_acctbal,
  n_name,
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
  c_name,
  c_acctbal,
  n_name;


select
  c_custkey,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  c_acctbal,
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
  c_acctbal,
  n_nationkey;