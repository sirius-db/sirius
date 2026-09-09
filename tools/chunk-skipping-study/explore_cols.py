import os
import duckdb, os, time
SP='/tmp/sirius-chunk-skipping'; OUT=SP+'/explore_cols'; os.makedirs(OUT, exist_ok=True)
c=duckdb.connect(); c.execute("SET threads=32"); c.execute("SET preserve_insertion_order=true")
COLS=['l_shipdate','l_commitdate','l_receiptdate','l_quantity','l_discount','l_orderkey','l_returnflag']
SRC={'unsorted':'/datasets/tpch_sf100/lineitem/part.0.parquet',
     'sorted'  :'/datasets/tpch_sf100_sorted/lineitem/part.0.parquet'}
for tag,src in SRC.items():
    for col in COLS:
        p=f'{OUT}/{tag}.{col}.parquet'
        if os.path.exists(p): continue
        c.execute(f"COPY (SELECT {col} FROM read_parquet('{src}')) TO '{p}' (FORMAT parquet, COMPRESSION uncompressed)")
    print(tag,'done', flush=True)
print(c.execute(f"select count(*) from '{OUT}/sorted.l_shipdate.parquet'").fetchone())
