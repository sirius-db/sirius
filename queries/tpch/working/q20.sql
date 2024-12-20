select
  s_suppkey,
  s_acctbal
from
  supplier, nation
where
  s_suppkey in (
    select
      ps_suppkey
    from
      partsupp
    where
      ps_partkey in (
        select
          p_partkey
        from
          part
        where
          p_size = 1
        )
      and ps_availqty > (
        select
          sum(l_quantity) * 0.5
        from
          lineitem
        where
          l_partkey = ps_partkey
          and l_suppkey = ps_suppkey
          and l_shipdate >= 19940101
          and l_shipdate <= 19941231
        )
    )
  and s_nationkey = n_nationkey
  and n_nationkey = 3;

select
  s_suppkey,
  sum(s_acctbal)
from
  supplier, nation
where
  s_suppkey in (
    select
      ps_suppkey
    from
      partsupp
    where
      ps_partkey in (
        select
          p_partkey
        from
          part
        where
          p_name like '%forest%'
        )
      and ps_availqty > (
        select
          sum(l_quantity) * 0.5
        from
          lineitem
        where
          l_partkey = ps_partkey
          and l_suppkey = ps_suppkey
          and l_shipdate >= 19940101
          and l_shipdate <= 19941231
        )
    )
  and s_nationkey = n_nationkey
  and n_nationkey = 3
group by
  s_suppkey,
order by
  s_suppkey;