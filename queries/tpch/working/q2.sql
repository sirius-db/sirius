select
  s_acctbal,
  n_nationkey,
  p_partkey,
  p_mfgr,
  s_suppkey
from
  part,
  supplier,
  partsupp,
  nation,
  region
where
  p_partkey = ps_partkey
  and s_suppkey = ps_suppkey
  and p_size = 15
  and (p_type + 3) % 5 = 0
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_regionkey = 3
  and ps_supplycost = (
    select
      min(ps_supplycost)
    from
      partsupp,
      supplier,
      nation,
      region
    where
      p_partkey = ps_partkey
      and s_suppkey = ps_suppkey
      and s_nationkey = n_nationkey
      and n_regionkey = r_regionkey
      and r_regionkey = 3
    );