# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""TPC-H query definitions (base SQL without gpu_processing wrapper).

QUERY_TEMPLATES holds the 22 queries with {PLACEHOLDER} substitution
parameters. QUERIES renders them with DEFAULT_PARAMS, one fixed draw shared
by every stream; tpch_substitutions.stream_params draws per-stream values.
"""

QUERY_TEMPLATES = {
    "q1": """select
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
    l_shipdate <= date '{DATE}'
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus""",
    "q2": """select
  s.s_acctbal,
  s.s_name,
  n.n_name,
  p.p_partkey,
  p.p_mfgr,
  s.s_address,
  s.s_phone,
  s.s_comment
from
  part p,
  supplier s,
  partsupp ps,
  nation n,
  region r
where
  p.p_partkey = ps.ps_partkey
  and s.s_suppkey = ps.ps_suppkey
  and p.p_size = {SIZE}
  and p.p_type like '%{TYPE}'
  and s.s_nationkey = n.n_nationkey
  and n.n_regionkey = r.r_regionkey
  and r.r_name = '{REGION}'
  and ps.ps_supplycost = (
    select
      min(ps.ps_supplycost)
    from
      partsupp ps,
      supplier s,
      nation n,
      region r
    where
      p.p_partkey = ps.ps_partkey
      and s.s_suppkey = ps.ps_suppkey
      and s.s_nationkey = n.n_nationkey
      and n.n_regionkey = r.r_regionkey
      and r.r_name = '{REGION}'
  )
order by
  s.s_acctbal desc,
  n.n_name,
  s.s_name,
  p.p_partkey
limit 100""",
    "q3": """select
  l.l_orderkey,
  sum(l.l_extendedprice * (1 - l.l_discount)) as revenue,
  o.o_orderdate,
  o.o_shippriority
from
  customer c,
  orders o,
  lineitem l
where
  c.c_mktsegment = '{SEGMENT}'
  and c.c_custkey = o.o_custkey
  and l.l_orderkey = o.o_orderkey
  and o.o_orderdate < date '{DATE}'
  and l.l_shipdate > date '{DATE}'
group by
  l.l_orderkey,
  o.o_orderdate,
  o.o_shippriority
order by
  revenue desc,
  o.o_orderdate
limit 10""",
    "q4": """select
  o.o_orderpriority,
  count(*) as order_count
from
  orders o
where
  o.o_orderdate >= date '{DATE}'
  and o.o_orderdate < date '{DATE_END}'
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
  o.o_orderpriority""",
    "q5": """select
  n.n_name,
  sum(l.l_extendedprice * (1 - l.l_discount)) as revenue
from
  orders o,
  lineitem l,
  supplier s,
  nation n,
  region r,
  customer c
where
  c.c_custkey = o.o_custkey
  and l.l_orderkey = o.o_orderkey
  and l.l_suppkey = s.s_suppkey
  and c.c_nationkey = s.s_nationkey
  and s.s_nationkey = n.n_nationkey
  and n.n_regionkey = r.r_regionkey
  and r.r_name = '{REGION}'
  and o.o_orderdate >= date '{DATE}'
  and o.o_orderdate < date '{DATE_END}'
group by
  n.n_name
order by
  revenue desc""",
    "q6": """select
  sum(l_extendedprice * l_discount) as revenue
from
  lineitem
where
  l_shipdate >= date '{DATE}'
  and l_shipdate < date '{DATE_END}'
  and l_discount between {DISCOUNT} - 0.01 and {DISCOUNT} + 0.01
  and l_quantity < {QUANTITY}""",
    "q7": """select
  supp_nation,
  cust_nation,
  l_year,
  sum(volume) as revenue
from
  (
    select
      n1.n_name as supp_nation,
      n2.n_name as cust_nation,
      extract(year from l.l_shipdate) as l_year,
      l.l_extendedprice * (1 - l.l_discount) as volume
    from
      supplier s,
      lineitem l,
      orders o,
      customer c,
      nation n1,
      nation n2
    where
      s.s_suppkey = l.l_suppkey
      and o.o_orderkey = l.l_orderkey
      and c.c_custkey = o.o_custkey
      and s.s_nationkey = n1.n_nationkey
      and c.c_nationkey = n2.n_nationkey
      and (
        (n1.n_name = '{NATION1}' and n2.n_name = '{NATION2}')
        or (n1.n_name = '{NATION2}' and n2.n_name = '{NATION1}')
      )
      and l.l_shipdate between date '1995-01-01' and date '1996-12-31'
  ) as shipping
group by
  supp_nation,
  cust_nation,
  l_year
order by
  supp_nation,
  cust_nation,
  l_year""",
    "q8": """select
  o_year,
  sum(case
    when nation = '{NATION}' then volume
    else 0
  end) / sum(volume) as mkt_share
from
  (
    select
      extract(year from o.o_orderdate) as o_year,
      l.l_extendedprice * (1 - l.l_discount) as volume,
      n2.n_name as nation
    from
      lineitem l,
      part p,
      supplier s,
      orders o,
      customer c,
      nation n1,
      nation n2,
      region r
    where
      p.p_partkey = l.l_partkey
      and s.s_suppkey = l.l_suppkey
      and l.l_orderkey = o.o_orderkey
      and o.o_custkey = c.c_custkey
      and c.c_nationkey = n1.n_nationkey
      and n1.n_regionkey = r.r_regionkey
      and r.r_name = '{REGION}'
      and s.s_nationkey = n2.n_nationkey
      and o.o_orderdate between date '1995-01-01' and date '1996-12-31'
      and p.p_type = '{TYPE}'
  ) as all_nations
group by
  o_year
order by
  o_year""",
    "q9": """select
  nation,
  o_year,
  sum(amount) as sum_profit
from
  (
    select
      n.n_name as nation,
      extract(year from o.o_orderdate) as o_year,
      l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity as amount
    from
      part p,
      supplier s,
      lineitem l,
      partsupp ps,
      orders o,
      nation n
    where
      s.s_suppkey = l.l_suppkey
      and ps.ps_suppkey = l.l_suppkey
      and ps.ps_partkey = l.l_partkey
      and p.p_partkey = l.l_partkey
      and o.o_orderkey = l.l_orderkey
      and s.s_nationkey = n.n_nationkey
      and p.p_name like '%{COLOR}%'
  ) as profit
group by
  nation,
  o_year
order by
  nation,
  o_year desc""",
    "q10": """select
  c.c_custkey,
  c.c_name,
  sum(l.l_extendedprice * (1 - l.l_discount)) as revenue,
  c.c_acctbal,
  n.n_name,
  c.c_address,
  c.c_phone,
  c.c_comment
from
  customer c,
  orders o,
  lineitem l,
  nation n
where
  c.c_custkey = o.o_custkey
  and l.l_orderkey = o.o_orderkey
  and o.o_orderdate >= date '{DATE}'
  and o.o_orderdate < date '{DATE_END}'
  and l.l_returnflag = 'R'
  and c.c_nationkey = n.n_nationkey
group by
  c.c_custkey,
  c.c_name,
  c.c_acctbal,
  c.c_phone,
  n.n_name,
  c.c_address,
  c.c_comment
order by
  revenue desc
limit 20""",
    "q11": """select
  ps.ps_partkey,
  sum(ps.ps_supplycost * ps.ps_availqty) as value
from
  partsupp ps,
  supplier s,
  nation n
where
  ps.ps_suppkey = s.s_suppkey
  and s.s_nationkey = n.n_nationkey
  and n.n_name = '{NATION}'
group by
  ps.ps_partkey having
    sum(ps.ps_supplycost * ps.ps_availqty) > (
      select
        sum(ps.ps_supplycost * ps.ps_availqty) * {FRACTION}
      from
        partsupp ps,
        supplier s,
        nation n
      where
        ps.ps_suppkey = s.s_suppkey
        and s.s_nationkey = n.n_nationkey
        and n.n_name = '{NATION}'
    )
order by
  value desc""",
    "q12": """select
  l.l_shipmode,
  sum(case
    when o.o_orderpriority = '1-URGENT'
      or o.o_orderpriority = '2-HIGH'
      then 1
    else 0
  end) as high_line_count,
  sum(case
    when o.o_orderpriority <> '1-URGENT'
      and o.o_orderpriority <> '2-HIGH'
      then 1
    else 0
  end) as low_line_count
from
  orders o,
  lineitem l
where
  o.o_orderkey = l.l_orderkey
  and l.l_shipmode in ('{SHIPMODE1}', '{SHIPMODE2}')
  and l.l_commitdate < l.l_receiptdate
  and l.l_shipdate < l.l_commitdate
  and l.l_receiptdate >= date '{DATE}'
  and l.l_receiptdate < date '{DATE_END}'
group by
  l.l_shipmode
order by
  l.l_shipmode""",
    "q13": """select
  c_count,
  count(*) as custdist
from
  (
    select
      c.c_custkey,
      count(o.o_orderkey)
    from
      customer c
      left outer join orders o
        on c.c_custkey = o.o_custkey
        and o.o_comment not like '%{WORD1}%{WORD2}%'
    group by
      c.c_custkey
  ) as orders (c_custkey, c_count)
group by
  c_count
order by
  custdist desc,
  c_count desc""",
    "q14": """select
  100.00 * sum(case
    when p.p_type like 'PROMO%'
      then l.l_extendedprice * (1 - l.l_discount)
    else 0
  end) / sum(l.l_extendedprice * (1 - l.l_discount)) as promo_revenue
from
  lineitem l,
  part p
where
  l.l_partkey = p.p_partkey
  and l.l_shipdate >= date '{DATE}'
  and l.l_shipdate < date '{DATE_END}'""",
    "q15": """with revenue_view as (
  select
    l_suppkey as supplier_no,
    sum(l_extendedprice * (1 - l_discount)) as total_revenue
  from
    lineitem
  where
    l_shipdate >= date '{DATE}'
    and l_shipdate < date '{DATE_END}'
  group by
    l_suppkey
)

select
  s.s_suppkey,
  s.s_name,
  s.s_address,
  s.s_phone,
  r.total_revenue
from
  supplier s,
  revenue_view r
where
  s.s_suppkey = r.supplier_no
  and r.total_revenue = (
    select
      max(total_revenue)
    from
      revenue_view
  )
order by
  s.s_suppkey""",
    "q16": """select
  p.p_brand,
  p.p_type,
  p.p_size,
  count(distinct ps.ps_suppkey) as supplier_cnt
from
  partsupp ps,
  part p
where
  p.p_partkey = ps.ps_partkey
  and p.p_brand <> 'Brand#{BRAND}'
  and p.p_type not like '{TYPE}%'
  and p.p_size in ({SIZES})
  and ps.ps_suppkey not in (
    select
      s.s_suppkey
    from
      supplier s
    where
      s.s_comment like '%Customer%Complaints%'
  )
group by
  p.p_brand,
  p.p_type,
  p.p_size
order by
  supplier_cnt desc,
  p.p_brand,
  p.p_type,
  p.p_size""",
    "q17": """select
  sum(l.l_extendedprice) / 7.0 as avg_yearly
from
  lineitem l,
  part p
where
  p.p_partkey = l.l_partkey
  and p.p_brand = 'Brand#{BRAND}'
  and p.p_container = '{CONTAINER}'
  and l.l_quantity < (
    select
      0.2 * avg(l2.l_quantity)
    from
      lineitem l2
    where
      l2.l_partkey = p.p_partkey
  )""",
    "q18": """select
  c.c_name,
  c.c_custkey,
  o.o_orderkey,
  o.o_orderdate,
  o.o_totalprice,
  sum(l.l_quantity)
from
  customer c,
  orders o,
  lineitem l
where
  o.o_orderkey in (
    select
      l_orderkey
    from
      lineitem
    group by
      l_orderkey having
        sum(l_quantity) > {QUANTITY}
  )
  and c.c_custkey = o.o_custkey
  and o.o_orderkey = l.l_orderkey
group by
  c.c_name,
  c.c_custkey,
  o.o_orderkey,
  o.o_orderdate,
  o.o_totalprice
order by
  o.o_totalprice desc,
  o.o_orderdate
limit 100""",
    "q19": """select
  sum(l.l_extendedprice* (1 - l.l_discount)) as revenue
from
  lineitem l,
  part p
where
  (
    p.p_partkey = l.l_partkey
    and p.p_brand = 'Brand#{BRAND1}'
    and p.p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
    and l.l_quantity >= {QTY1} and l.l_quantity <= {QTY1} + 10
    and p.p_size between 1 and 5
    and l.l_shipmode in ('AIR', 'AIR REG')
    and l.l_shipinstruct = 'DELIVER IN PERSON'
  )
  or
  (
    p.p_partkey = l.l_partkey
    and p.p_brand = 'Brand#{BRAND2}'
    and p.p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
    and l.l_quantity >= {QTY2} and l.l_quantity <= {QTY2} + 10
    and p.p_size between 1 and 10
    and l.l_shipmode in ('AIR', 'AIR REG')
    and l.l_shipinstruct = 'DELIVER IN PERSON'
  )
  or
  (
    p.p_partkey = l.l_partkey
    and p.p_brand = 'Brand#{BRAND3}'
    and p.p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
    and l.l_quantity >= {QTY3} and l.l_quantity <= {QTY3} + 10
    and p.p_size between 1 and 15
    and l.l_shipmode in ('AIR', 'AIR REG')
    and l.l_shipinstruct = 'DELIVER IN PERSON'
  )""",
    "q20": """select
  s.s_name,
  s.s_address
from
  supplier s,
  nation n
where
  s.s_suppkey in (
    select
      ps.ps_suppkey
    from
      partsupp ps
    where
      ps. ps_partkey in (
        select
          p.p_partkey
        from
          part p
        where
          p.p_name like '{COLOR}%'
      )
      and ps.ps_availqty > (
        select
          0.5 * sum(l.l_quantity)
        from
          lineitem l
        where
          l.l_partkey = ps.ps_partkey
          and l.l_suppkey = ps.ps_suppkey
          and l.l_shipdate >= date '{DATE}'
          and l.l_shipdate < date '{DATE_END}'
      )
  )
  and s.s_nationkey = n.n_nationkey
  and n.n_name = '{NATION}'
order by
  s.s_name""",
    "q21": """select
  s.s_name,
  count(*) as numwait
from
  supplier s,
  lineitem l1,
  orders o,
  nation n
where
  s.s_suppkey = l1.l_suppkey
  and o.o_orderkey = l1.l_orderkey
  and o.o_orderstatus = 'F'
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
  and s.s_nationkey = n.n_nationkey
  and n.n_name = '{NATION}'
group by
  s.s_name
order by
  numwait desc,
  s.s_name
limit 100""",
    "q22": """select
  cntrycode,
  count(*) as numcust,
  sum(c_acctbal) as totacctbal
from
  (
    select
      substring(c_phone from 1 for 2) as cntrycode,
      c_acctbal
    from
      customer c
    where
      substring(c_phone from 1 for 2) in
        ({CODES})
      and c_acctbal > (
        select
          avg(c_acctbal)
        from
          customer
        where
          c_acctbal > 0.00
          and substring(c_phone from 1 for 2) in
            ({CODES})
      )
      and not exists (
        select
          *
        from
          orders o
        where
          o.o_custkey = c.c_custkey
      )
  ) as custsale
group by
  cntrycode
order by
  cntrycode""",
}

DEFAULT_PARAMS = {
    "q1": {"DATE": "1995-08-19"},
    "q2": {"SIZE": 41, "TYPE": "NICKEL", "REGION": "EUROPE"},
    "q3": {"SEGMENT": "HOUSEHOLD", "DATE": "1995-03-25"},
    "q4": {"DATE": "1996-10-01", "DATE_END": "1997-01-01"},
    "q5": {"REGION": "EUROPE", "DATE": "1997-01-01", "DATE_END": "1998-01-01"},
    "q6": {
        "DATE": "1997-01-01",
        "DATE_END": "1998-01-01",
        "DISCOUNT": "0.03",
        "QUANTITY": 24,
    },
    "q7": {"NATION1": "EGYPT", "NATION2": "UNITED STATES"},
    "q8": {"NATION": "EGYPT", "REGION": "MIDDLE EAST", "TYPE": "PROMO BRUSHED COPPER"},
    "q9": {"COLOR": "yellow"},
    "q10": {"DATE": "1994-03-01", "DATE_END": "1994-06-01"},
    "q11": {"NATION": "GERMANY", "FRACTION": "0.0001000000"},
    "q12": {
        "SHIPMODE1": "TRUCK",
        "SHIPMODE2": "REG AIR",
        "DATE": "1994-01-01",
        "DATE_END": "1995-01-01",
    },
    "q13": {"WORD1": "special", "WORD2": "requests"},
    "q14": {"DATE": "1994-08-01", "DATE_END": "1994-09-01"},
    "q15": {"DATE": "1993-05-01", "DATE_END": "1993-08-01"},
    "q16": {
        "BRAND": "21",
        "TYPE": "MEDIUM PLATED",
        "SIZES": "38, 2, 8, 31, 44, 5, 14, 24",
    },
    "q17": {"BRAND": "13", "CONTAINER": "JUMBO CAN"},
    "q18": {"QUANTITY": 300},
    "q19": {
        "BRAND1": "41",
        "QTY1": 2,
        "BRAND2": "13",
        "QTY2": 14,
        "BRAND3": "55",
        "QTY3": 23,
    },
    "q20": {
        "COLOR": "antique",
        "DATE": "1993-01-01",
        "DATE_END": "1994-01-01",
        "NATION": "KENYA",
    },
    "q21": {"NATION": "BRAZIL"},
    "q22": {"CODES": "'24', '31', '11', '16', '21', '20', '34'"},
}


def render(name, params):
    """Fill one query template with substitution parameter values."""
    return QUERY_TEMPLATES[name].format(**params)


# The fixed default rendering, shared by every stream unless a runner opts
# into per-stream parameters (tpch_substitutions.stream_params).
QUERIES = {name: render(name, DEFAULT_PARAMS[name]) for name in QUERY_TEMPLATES}
