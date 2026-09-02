#!/usr/bin/env python3
"""DuckDB CPU oracle for the TPC-H sweep.

The bench harness times queries and counts rows; it never checks answers. This runs the
SAME SQL through DuckDB over the same parquet and writes one TSV per query, formatted to
match the mysql --batch output so the two can be diffed directly.

Usage: oracle.py <queries_dir> <tpch_data> <out_dir> [qNN ...]
"""
import decimal
import glob
import os
import re
import sys
import time

import duckdb

qdir, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
only = sys.argv[4:]
os.makedirs(out, exist_ok=True)

# FILES("path"="file:///d/tbl/*.parquet","format"="parquet") -> read_parquet('/d/tbl/*.parquet')
FILES = re.compile(
    r'FILES\(\s*"path"\s*=\s*"file://([^"]+)"\s*,\s*"format"\s*=\s*"parquet"\s*\)',
    re.I,
)


def fmt(v):
    """Match mysql --batch rendering closely enough to diff."""
    if v is None:
        return "NULL"
    if isinstance(v, decimal.Decimal):
        # strip trailing zeros but keep an integer looking like an integer
        s = format(v.normalize(), "f")
        return s
    if isinstance(v, float):
        return repr(v)
    return str(v)


names = sorted(
    os.path.basename(p)[:-4] for p in glob.glob(os.path.join(qdir, "q*.sql"))
)
if only:
    names = [n for n in names if n in only]

con = duckdb.connect()
con.execute("PRAGMA threads=%d" % int(os.environ.get("ORACLE_THREADS", "48")))
con.execute("SET preserve_insertion_order=false")
# At SF500 several queries spill. The root volume has no room, so temp must go on the nvme.
con.execute("SET memory_limit='%s'" % os.environ.get("ORACLE_MEM", "380GB"))
_tmp = os.environ.get("ORACLE_TMP", "/opt/dlami/nvme/duckdb-tmp")
os.makedirs(_tmp, exist_ok=True)
con.execute("SET temp_directory='%s'" % _tmp)

for n in names:
    sql = open(os.path.join(qdir, f"{n}.sql")).read().replace("__TPCH_DATA__", data)
    sql = FILES.sub(lambda m: f"read_parquet('{m.group(1)}')", sql).strip().rstrip(";")
    t0 = time.time()
    try:
        rel = con.sql(sql)
        cols = rel.columns
        rows = rel.fetchall()
        with open(os.path.join(out, f"{n}.tsv"), "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(fmt(v) for v in r) + "\n")
        print(f"{n} ok rows={len(rows)} {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        with open(os.path.join(out, f"{n}.err"), "w") as f:
            f.write(str(e))
        print(f"{n} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
