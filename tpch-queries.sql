-- Copyright 2025 Sirius Contributors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

call gpu_processing("select
  l_returnflag,
  l_linestatus,
  sum(l_quantity) as sum_qty,
  sum(l_extendedprice) as sum_base_price,
  sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
  sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
  avg(l_quantity) as avg_qty,
  avg(l_extendedprice) as avg_price,
  avg(l_discount) as avg_disc,
  count(*) as count_order
from
  lineitem
where
  l_shipdate <= 19940902
group by
  l_returnflag,
  l_linestatus
order by
  l_returnflag,
  l_linestatus;");

call gpu_processing("select
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
  and r_name = 'EUROPE'
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
      and r_name = 'EUROPE'
    )
order by
  s_acctbal desc,
  n_name,
  s_name,
  p_partkey");
    
call gpu_processing("select
  l_orderkey,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  o_orderdate,
  o_shippriority
from
  customer,
  orders,
  lineitem
where
  c_mktsegment = 1
  and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < 19950315
  and l_shipdate > 19950315
group by
  l_orderkey,
  o_orderdate,
  o_shippriority
order by
  revenue desc,
  o_orderdate");

call gpu_processing("select
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
  o_orderpriority
order by
  o_orderpriority;");

call gpu_processing("select
  n_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  customer,
  orders,
  lineitem,
  supplier,
  nation,
  region
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'ASIA'
  and o_orderdate >= 19940101
  and o_orderdate <= 19941231
group by
  n_name
order by
  revenue desc;");

call gpu_processing("select
  sum(l_extendedprice * l_discount) as revenue
from
  lineitem
where
  l_shipdate >= 19940101
  and l_shipdate <= 19941231
  and l_discount between 0.05 and 0.07
  and l_quantity < 24;");

call gpu_processing("select
  supp_nation,
  cust_nation,
  l_year,
  sum(volume) as revenue
from (
  select
    n1.n_name as supp_nation,
    n2.n_name as cust_nation,
    l_shipdate//10000 as l_year,
    l_extendedprice * (1 - l_discount) as volume
  from
    supplier,
    lineitem,
    orders,
    customer,
    nation n1,
    nation n2
  where
    s_suppkey = l_suppkey
    and o_orderkey = l_orderkey
    and c_custkey = o_custkey
    and s_nationkey = n1.n_nationkey
    and c_nationkey = n2.n_nationkey
    and (
      (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
      or (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
    )
    and l_shipdate between 19950101 and 19961231
  ) as shipping
group by
  supp_nation,
  cust_nation,
  l_year
order by
  supp_nation,
  cust_nation,
  l_year;");

call gpu_processing("select
  o_year,
  sum(case
    when nation = 1
    then volume
    else 0
  end) / sum(volume) as mkt_share
from (
  select
    o_orderdate//10000 as o_year,
    l_extendedprice * (1 - l_discount) as volume,
    n2.n_nationkey as nation
  from
    part,
    supplier,
    lineitem,
    orders,
    customer,
    nation n1,
    nation n2,
    region
  where
    p_partkey = l_partkey
    and s_suppkey = l_suppkey
    and l_orderkey = o_orderkey
    and o_custkey = c_custkey
    and c_nationkey = n1.n_nationkey
    and n1.n_regionkey = r_regionkey
    and r_regionkey = 1
    and s_nationkey = n2.n_nationkey
    and o_orderdate between 19950101 and 19961231
    and p_type = 103
  ) as all_nations
group by
  o_year
order by
  o_year;");

call gpu_processing("select
  nation,
  o_year,
  sum(amount) as sum_profit
from(
  select
    n_name as nation,
    o_orderdate//10000 as o_year,
    l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
  from
    part,
    supplier,
    lineitem,
    partsupp,
    orders,
    nation
  where
    s_suppkey = l_suppkey
    and ps_suppkey = l_suppkey
    and ps_partkey = l_partkey
    and p_partkey = l_partkey
    and o_orderkey = l_orderkey
    and s_nationkey = n_nationkey
    and p_name like '%green%'
  ) as profit
group by
  nation,
  o_year
order by
  nation,
  o_year desc;");

call gpu_processing("select
  c_custkey,
  c_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  c_acctbal,
  n_name,
  c_address,
  c_phone,
  c_comment
from
  customer,
  orders,
  lineitem,
  nation
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate >= 19931001
  and o_orderdate <= 19931231
  and l_returnflag = 0
  and c_nationkey = n_nationkey
group by
  c_custkey,
  c_name,
  c_acctbal,
  c_phone,
  n_name,
  c_address,
  c_comment 
order by
  revenue desc;");

call gpu_processing("select
  *
from (
  select
    ps_partkey,
    sum(ps_supplycost * ps_availqty) as value
  from
    partsupp,
    supplier,
    nation
  where
    ps_suppkey = s_suppkey
    and s_nationkey = n_nationkey
    and n_name = 'GERMANY'
  group by
    ps_partkey
) as inner_query
where
  value > (
    select
      sum(ps_supplycost * ps_availqty) * 0.0000000333
    from
      partsupp,
      supplier,
      nation
    where
      ps_suppkey = s_suppkey
      and s_nationkey = n_nationkey
      and n_name = 'GERMANY'
  )
order by
  value desc,
  ps_partkey;");

call gpu_processing("select
  l_shipmode,
  sum(case
    when o_orderpriority = 0
      or o_orderpriority = 1
    then CAST(1 AS DOUBLE)
    else CAST(0 AS DOUBLE)
  end) as high_line_count,
  sum(case
    when o_orderpriority <> 0
      and o_orderpriority <> 1
    then CAST(1 AS DOUBLE)
    else CAST(0 AS DOUBLE)
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
  l_shipmode;");

call gpu_processing("select
  c_count,
  count(*) as custdist
from (
  select
    c_custkey,
    count(o_orderkey) as c_count
  from
    customer left outer join orders on (
      c_custkey = o_custkey
      and o_comment not like '%special%requests%'
    )
  group by
    c_custkey
  ) as c_orders
group by
  c_count
order by
  custdist desc,
  c_count desc;");

call gpu_processing("select
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
  and l_shipdate <= 19950931;");

call gpu_processing("with revenue_view as (
  select
    l_suppkey as supplier_no,
    sum(l_extendedprice * (1 - l_discount)) as total_revenue
  from
    lineitem
  where
    l_shipdate >= 19960101
    and l_shipdate <= 19960331
  group by
    l_suppkey
)

select
  s_suppkey,
  total_revenue
from
  supplier,
  revenue_view
where
  s_suppkey = supplier_no
  and total_revenue = (
    select
      max(total_revenue)
    from
      revenue_view
    )
order by
  s_suppkey;");

call gpu_processing("select
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
      s_comment like '%Customer%Complaints%'
  )
group by
  p_brand,
  p_type,
  p_size
order by
  supplier_cnt desc,
  p_brand,
  p_type,
  p_size;");

call gpu_processing("select
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
  );");

call gpu_processing("select
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice,
  sum(l_quantity)
from
  customer,
  orders,
  lineitem
where
  o_orderkey in (
    select
      l_orderkey
    from
      lineitem
    group by
      l_orderkey
    having
      sum(l_quantity) > 300
    )
  and c_custkey = o_custkey
  and o_orderkey = l_orderkey
group by
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice
order by
  o_totalprice desc,
  o_orderdate,
  o_orderkey;");

call gpu_processing("select
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
  );");

call gpu_processing("select
  s_name,
  s_address
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
          p_name like 'forest%'
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
  and n_name = 'CANADA'
  order by
    s_name;");

call gpu_processing("select
  s_name,
  count(*) as numwait
from
  supplier,
  lineitem l1,
  orders,
  nation
where
  s_suppkey = l1.l_suppkey
  and o_orderkey = l1.l_orderkey
  and o_orderstatus = 1
  and l1.l_receiptdate > l1.l_commitdate
  and exists (
    select
      *
    from
      lineitem l2
    where
      l2.l_orderkey = l1.l_orderkey
      and l2.l_suppkey <> l1.l_suppkey
  )
  and not exists (
    select
      *
    from
      lineitem l3
    where
      l3.l_orderkey = l1.l_orderkey
      and l3.l_suppkey <> l1.l_suppkey
      and l3.l_receiptdate > l3.l_commitdate
  )
  and s_nationkey = n_nationkey
  and n_name = 'SAUDI ARABIA'
group by
  s_name
order by
  numwait desc,
  s_name;");

call gpu_processing("select
  cntrycode,
  count(*) as numcust,
  sum(c_acctbal) as totacctbal
from (
  select
    substr(c_phone, 1, 2) as cntrycode,
    c_acctbal
  from
    customer
  where
    substr(c_phone, 1, 2) in ('13', '31', '23')
    and c_acctbal > (
      select
        avg(c_acctbal)
      from
        customer
      where
        c_acctbal > 0.00
        and substr(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17')
      )
    and not exists (
      select
        *
      from
        orders
      where
        o_custkey = c_custkey
    )
  ) as custsale
group by
  cntrycode
order by
  cntrycode;");