select
  p_brand,
  p_type,
  count(distinct ps_suppkey) as supplier_cnt,
  p_size
from
  partsupp,
  part
where
  p_partkey = ps_partkey
  and p_brand <> 45
  and (p_type < 65 or p_type >= 70)
  and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
  and ps_suppkey not in (
    select
      s_suppkey
    from
      supplier
    where
      s_acctbal < 2000
  )
group by
  p_brand,
  p_type,
  p_size;