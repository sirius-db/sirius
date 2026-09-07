import os
import duckdb, numpy as np, time
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
c=duckdb.connect()
c.execute("create view lineitem as select * from read_parquet('/datasets/tpch_sf10/lineitem/*.parquet')")
t0=time.time()
cols="l_orderkey,l_partkey,l_suppkey,(l_shipdate - DATE '1970-01-01') as l_shipdate,(l_commitdate - DATE '1970-01-01') as l_commitdate,(l_receiptdate - DATE '1970-01-01') as l_receiptdate,(l_quantity*100)::bigint as l_quantity,(l_discount*100)::bigint as l_discount,l_returnflag,l_shipmode,l_shipinstruct"
c.execute(f"copy (select {cols} from lineitem) to '{SP}/li10.parquet' (format parquet, compression zstd)")
print('dump %.1fs'%(time.time()-t0))
print(c.execute(f"select count(*) from '{SP}/li10.parquet'").fetchone())
