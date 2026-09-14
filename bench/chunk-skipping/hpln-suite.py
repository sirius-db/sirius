#!/usr/bin/env python3
"""Run the 22 TPC-H queries over a `.hpln` directory and over parquet, and compare.

Deliberately simple and single-process: this is the gap-finder that has to run before a big
generation, not the timing harness. `performance_test.py` is the harness -- it knows about
pinning, iterations and cache dropping -- but it has no `.hpln` data source, so this stands in
while the format is being shaken out.

  pixi run python bench/chunk-skipping/hpln-suite.py \
      --hpln /datasets/tpch_sf1_hpln --parquet /datasets/tpch_sf1

Reports per query: hpln time, parquet time, and whether the two results agree.
"""
import argparse
import glob
import os
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTENSION_PATH = os.path.join(REPO, "build/release/extension/sirius/sirius.duckdb_extension")
QUERY_DIR = os.path.join(REPO, "test/tpch_performance/tpch_queries/orig")
TABLES = ["lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region"]


def parquet_views(d):
    out = {}
    for t in TABLES:
        files = []
        for pattern in (f"{t}.parquet", f"{t}_*.parquet", os.path.join(t, "*.parquet")):
            files.extend(glob.glob(os.path.join(d, pattern)))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"no parquet for {t} in {d}")
        out[t] = "read_parquet([" + ",".join(f"'{f}'" for f in files) + "])"
    return out


def hpln_views(d):
    out = {}
    for t in TABLES:
        path = os.path.join(d, f"{t}.hpln")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no .hpln for {t} in {d}")
        out[t] = f"read_simpatico('{path}')"
    return out


def all_files(d, hpln):
    """Every file of a dataset, for cache eviction."""
    out = []
    for t in TABLES:
        if hpln:
            out.append(os.path.join(d, f"{t}.hpln"))
        else:
            for pattern in (f"{t}.parquet", f"{t}_*.parquet", os.path.join(t, "*.parquet")):
                out.extend(glob.glob(os.path.join(d, pattern)))
    return sorted(set(p for p in out if os.path.exists(p)))


def drop_cache(paths):
    """Evict `paths`, so the next query reads from storage rather than from the last query."""
    os.sync()
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def io_read_bytes():
    """Bytes this process has pulled from block devices, as the kernel counts them."""
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("read_bytes:"):
                return int(line.split(":")[1])
    return 0


def register(con, views):
    for table, source in views.items():
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM {source};")


def run(con, sql):
    b0 = io_read_bytes()
    t0 = time.time()
    rows = con.execute(sql).fetchall()
    return time.time() - t0, io_read_bytes() - b0, rows


def normalize(rows):
    """Stringify with a fixed float precision, so a 1-ulp difference is not a mismatch."""
    def cell(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return [tuple(cell(v) for v in r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpln", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--queries", default=",".join(str(i) for i in range(1, 23)))
    ap.add_argument("--config", default=os.environ.get("SIRIUS_CONFIG_FILE"))
    ap.add_argument(
        "--drop-cache",
        action="store_true",
        help="evict each arm's files before EVERY query, so a cold run measures storage rather "
        "than the previous query's reads",
    )
    args = ap.parse_args()
    if args.config:
        os.environ["SIRIUS_CONFIG_FILE"] = args.config
    import duckdb

    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXTENSION_PATH}'")
    con.execute("SET gpu_execution = true")
    con.execute("SET enable_duckdb_fallback = false")

    qs = [int(q) for q in args.queries.split(",")]
    hv, pv = hpln_views(args.hpln), parquet_views(args.parquet)
    fails, results = [], []
    for q in qs:
        sql = open(os.path.join(QUERY_DIR, f"q{q}.sql")).read()
        line = f"q{q:<3}"
        try:
            register(con, pv)
            if args.drop_cache:
                drop_cache(all_files(args.parquet, hpln=False))
            pt, pbytes, prows = run(con, sql)
            line += f" parquet {pt:7.3f}s {pbytes/1e9:6.2f}GB"
        except Exception as e:
            prows, pt, pbytes = None, float("nan"), 0
            line += f" parquet FAILED: {str(e).splitlines()[0][:90]}"
        try:
            register(con, hv)
            if args.drop_cache:
                drop_cache(all_files(args.hpln, hpln=True))
            ht, hbytes, hrows = run(con, sql)
            line += f" | hpln {ht:7.3f}s {hbytes/1e9:6.2f}GB"
        except Exception as e:
            hrows, ht, hbytes = None, float("nan"), 0
            line += f" | hpln FAILED: {str(e).splitlines()[0][:110]}"
            fails.append(q)
        if prows is not None and hrows is not None:
            np_, nh = normalize(prows), normalize(hrows)
            same = np_ == nh
            # A query whose ORDER BY has ties emits them in whichever order the engine produced,
            # and the two sources produce different ones. Sorted equality is the real check;
            # report the weaker agreement rather than calling it a mismatch.
            if not same and sorted(np_) == sorted(nh):
                same = True
                line += "  MATCH(unordered)"
            else:
                line += "  MATCH" if same else f"  *** MISMATCH ({len(prows)} vs {len(hrows)} rows)"
            if not same:
                fails.append(q)
            results.append((q, pt, ht, pbytes, hbytes))
        print(line, flush=True)

    if results:
        tp = sum(r[1] for r in results)
        th = sum(r[2] for r in results)
        bp = sum(r[3] for r in results)
        bh = sum(r[4] for r in results)
        print(f"\nTOTAL over {len(results)} comparable queries: "
              f"parquet {tp:.3f}s, hpln {th:.3f}s ({(th/tp - 1) * 100:+.1f}%)")
        print(f"  bytes read: parquet {bp/1e9:.1f} GB, hpln {bh/1e9:.1f} GB "
              f"({(bh/bp - 1) * 100:+.1f}%)" if bp else "")
    print(f"FAILED/MISMATCHED: {sorted(set(fails)) if fails else 'none'}")


if __name__ == "__main__":
    main()
