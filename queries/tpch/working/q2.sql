select
  s_acctbal,
  s_name,
  n_name,
  p_partkey,
  p_mfgr,
  s_address,
  s_phone,
  s_comment
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
    ) limit 100;

select
  sum(s_acctbal),
  s_name,
  n_name,
  p_partkey,
  p_mfgr,
  s_address,
  s_phone,
  s_comment
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
    )
group by
  p_partkey,
  s_name,
  n_name,
  p_mfgr,
  s_address,
  s_phone,
  s_comment
order by
  p_partkey,
  s_name,
  n_name,
  p_mfgr,
  s_address,
  s_phone,
  s_comment