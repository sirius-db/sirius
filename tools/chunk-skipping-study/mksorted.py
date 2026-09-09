import os
import duckdb, os, time, shutil
SRC='/datasets/tpch_sf100'; DST='/datasets/tpch_sf100_sorted'; TMP='/datasets/.sorttmp'
os.makedirs(TMP, exist_ok=True)
c=duckdb.connect()
c.execute("SET threads=48"); c.execute("SET memory_limit='320GB'")
c.execute("SET temp_directory='/datasets/.duckdb_tmp'")
c.execute("SET preserve_insertion_order=true")

# SORT=0 writes the SAME tables through the SAME DuckDB writer with NO ORDER BY, producing a
# writer-matched control so a sorted-vs-unsorted comparison isolates row order from encoding.
SORT=os.environ.get('SORT','1')=='1'
DST=DST if SORT else os.environ.get('CTL_DST','/datasets/tpch_sf100_duckdb_natural')
SPEC={'lineitem':('l_shipdate',1048576,6), 'orders':('o_orderdate',1310720,2)}
for t,(key,rg,nfiles) in SPEC.items():
    out=os.path.join(DST,t); os.makedirs(out, exist_ok=True)
    for f in os.listdir(out): os.remove(os.path.join(out,f))
    one=os.path.join(TMP,f'{t}.parquet')
    t0=time.time()
    c.execute(f"""COPY (SELECT * FROM read_parquet('{SRC}/{t}/*.parquet') {'ORDER BY '+key if SORT else ''})
                  TO '{one}' (FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE {rg})""")
    n=c.execute(f"select count(*) from '{one}'").fetchone()[0]
    print(f'{t}: {"sorted" if SORT else "copied"} {n} rows into one file in {time.time()-t0:.0f}s', flush=True)
    per=(n+nfiles-1)//nfiles
    for k in range(nfiles):
        c.execute(f"""COPY (SELECT * FROM read_parquet('{one}') LIMIT {per} OFFSET {k*per})
                      TO '{out}/part.{k}.parquet' (FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE {rg})""")
    print(f'{t}: split into {nfiles} files in {time.time()-t0:.0f}s total', flush=True)
    os.remove(one)
for t in sorted(os.listdir(SRC)):
    pp=os.path.join(SRC,t)
    if not os.path.isdir(pp) or t in SPEC: continue
    o=os.path.join(DST,t); os.makedirs(o, exist_ok=True)
    for f in os.listdir(pp):
        d=os.path.join(o,f)
        if not os.path.exists(d): os.symlink(os.path.join(pp,f), d)
os.rmdir(TMP)
print('done')
