import duckdb, time
c=duckdb.connect(); c.execute("SET threads=72"); c.execute("SET memory_limit='350GB'")
c.execute("SET temp_directory='/datasets/.duckdb_tmp'")
t0=time.time()
c.execute("CREATE TABLE li AS SELECT * FROM read_parquet('/datasets/tpch_sf100/lineitem/*.parquet')")
n=c.execute("select count(*) from li").fetchone()[0]
print(f'load {n} rows: {time.time()-t0:.1f}s', flush=True)
for k in ('l_shipdate',):
    t0=time.time()
    c.execute(f"CREATE OR REPLACE TABLE s AS SELECT * FROM li ORDER BY {k}")
    dt=time.time()-t0
    print(f'CPU global sort by {k}: {dt:.1f}s  -> {n/dt/1e6:.0f} Mrows/s', flush=True)
