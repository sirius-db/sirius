select
    sum(case
    when (p_type >= 125 and p_type < 150)
    then l_extendedprice * (1 - l_discount)
    else 0.0
    end) * 100.0 / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
  lineitem,
  part
where
  l_partkey = p_partkey
  and l_shipdate >= 19950901
  and l_shipdate <= 19950931;