select
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  lineitem,
  part
where
  p_partkey = l_partkey
  and (
    (
      p_brand = 12
      and p_container in (0, 1, 4, 5)
      and l_quantity >= 1 and l_quantity <= 11
      and p_size between 1 and 5
      and l_shipmode in (0, 1)
      and l_shipinstruct = 0
    )
    or
    (
      p_brand = 23
      and p_container in (17, 18, 20, 21)
      and l_quantity >= 10 and l_quantity <= 20
      and p_size between 1 and 10
      and l_shipmode in (0, 1)
      and l_shipinstruct = 0
    )
    or
    (
      p_brand = 34
      and p_container in (8, 9, 12, 13)
      and l_quantity >= 20 and l_quantity <= 30
      and p_size between 1 and 15
      and l_shipmode in (0, 1)
      and l_shipinstruct = 0
    )
  );