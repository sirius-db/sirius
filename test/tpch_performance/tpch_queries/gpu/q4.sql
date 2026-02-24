call gpu_execution('select
  o.o_orderpriority,
  count(*) as order_count
from
  orders o
where
  o.o_orderdate >= date ''1996-10-01''
  and o.o_orderdate < date ''1997-01-01''
  and
  exists (
    select
      *
    from
      lineitem l
    where
      l.l_orderkey = o.o_orderkey
      and l.l_commitdate < l.l_receiptdate
  )
group by
  o.o_orderpriority
order by
  o.o_orderpriority');
