import os
import duckdb
c=duckdb.connect()
for t,col in (('lineitem','l_shipdate'),('orders','o_orderdate')):
    print('===',t)
    r=c.execute(f"""select file_name, min(stats_min_value) mn, max(stats_max_value) mx, count(*) rgs
        from parquet_metadata('/datasets/tpch_sf100_sorted/{t}/*.parquet')
        where path_in_schema='{col}' group by 1 order by 1""").fetchall()
    for x in r: print('  ',x[0].split('/')[-1], x[1], x[2], 'rgs=',x[3])
    # rowgroup-level span distribution
    s=c.execute(f"""select count(*), sum(case when stats_min_value=stats_max_value then 1 else 0 end),
        avg(date_diff('day', stats_min_value::date, stats_max_value::date))
        from parquet_metadata('/datasets/tpch_sf100_sorted/{t}/*.parquet') where path_in_schema='{col}'""").fetchone()
    print('   rowgroups:',s[0],' single-day:',s[1],' avg span days: %.1f'%s[2])
    s2=c.execute(f"""select count(*), avg(date_diff('day', stats_min_value::date, stats_max_value::date))
        from parquet_metadata('/datasets/tpch_sf100/{t}/*.parquet') where path_in_schema='{col}'""").fetchone()
    print('   ORIGINAL rowgroups:',s2[0],' avg span days: %.1f'%s2[1])
