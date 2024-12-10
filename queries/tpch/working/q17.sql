select
  sum(l_extendedprice) / 7.0 as avg_yearly
from
  lineitem,
  part
where
  p_partkey = l_partkey
  and p_brand = 23
  and p_container = 17
  and l_quantity < (
    select
      avg(l_quantity) * 0.2
    from
      lineitem
    where
      l_partkey = p_partkey
  );
