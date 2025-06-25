select
  l_shipmode,
  sum(case
    when o_orderpriority = 0
      or o_orderpriority = 1
    then 1
    else 0
  end) as high_line_count,
  sum(case
    when o_orderpriority <> 0
      and o_orderpriority <> 1
    then 1
    else 0
  end) as low_line_count
from
  orders,
  lineitem
where
  o_orderkey = l_orderkey
  and l_shipmode in (4, 6)
  and l_commitdate < l_receiptdate
  and l_shipdate < l_commitdate
  and l_receiptdate >= 19940101
  and l_receiptdate <= 19941231
group by
  l_shipmode
order by
  l_shipmode;