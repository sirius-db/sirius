select
  o_orderpriority,
  count(*) as order_count
from
  orders
where
  o_orderdate >= 19930701
  and o_orderdate <= 19930931
  and exists (
    select
      *
    from
      lineitem
    where
      l_orderkey = o_orderkey
      and l_commitdate < l_receiptdate
    )
group by
  o_orderpriority;