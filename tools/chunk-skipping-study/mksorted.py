import os
import duckdb, os, time, shutil
SRC='/datasets/tpch_sf100'; DST='/datasets/tpch_sf100_sorted'; TMP='/datasets/.sorttmp'
os.makedirs(TMP, exist_ok=True)
c=duckdb.connect()
c.execute("SET threads=48"); c.execute("SET memory_limit='320GB'")
c.execute("SET temp_directory='/datasets/.duckdb_tmp'")
c.execute("SET preserve_insertion_order=true")

SPEC={'lineitem':('l_shipdate',1048576,6), 'orders':('o_orderdate',1310720,2)}
for t,(key,rg,nfiles) in SPEC.items():
    out=os.path.join(DST,t)
    for f in os.listdir(out): os.remove(os.path.join(out,f))
    one=os.path.join(TMP,f'{t}.parquet')
    t0=time.time()
    c.execute(f"""COPY (SELECT * FROM read_parquet('{SRC}/{t}/*.parquet') ORDER BY {key})
                  TO '{one}' (FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE {rg})""")
    n=c.execute(f"select count(*) from '{one}'").fetchone()[0]
    print(f'{t}: sorted {n} rows into one file in {time.time()-t0:.0f}s', flush=True)
    per=(n+nfiles-1)//nfiles
    for k in range(nfiles):
        c.execute(f"""COPY (SELECT * FROM read_parquet('{one}') LIMIT {per} OFFSET {k*per})
                      TO '{out}/part.{k}.parquet' (FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE {rg})""")
    print(f'{t}: split into {nfiles} files in {time.time()-t0:.0f}s total', flush=True)
    os.remove(one)
os.rmdir(TMP)
print('done')
