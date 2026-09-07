import os
import duckdb, os, time
base='/datasets/tpch_sf1000'
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
c=duckdb.connect()
parts=[]
for t in sorted(os.listdir(base)):
    p=os.path.join(base,t)
    if not os.path.isdir(p): continue
    t0=time.time()
    c.execute(f"""create or replace temp view v as select '{t}' as tbl, file_name, row_group_id,
        row_group_num_rows, path_in_schema as col, stats_min_value as mn, stats_max_value as mx,
        stats_null_count as nulls, total_compressed_size as csize, total_uncompressed_size as usize
        from parquet_metadata('{p}/*.parquet')""")
    c.execute(f"copy (select * from v) to '{SP}/stats_{t}.parquet' (format parquet)")
    print(t, '%.1fs'%(time.time()-t0), c.execute("select count(*) from v").fetchone()[0], flush=True)
