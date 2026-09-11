#!/usr/bin/env python3
"""Host-tier pin comparison: a parquet pin against a `.hpln` pin, over the 22 TPC-H queries.

Four arms, which together separate "the file format" from "the row order":

  parquet            pin the parquet dataset, no clustering -- the plain baseline
  parquet-clustered  same, with cluster_by at pin time (the -8.57% of plan 5.4)
  hpln-sorted        pin a .hpln written from a globally ORDER BY'd query
  hpln-unsorted      pin a .hpln written in the dataset's natural order

A parquet pin decodes and re-compresses every batch; a `.hpln` pin is an I/O copy of bytes that
are already in the pinned representation. Pin wall time is therefore a headline number here, not
setup cost -- it is reported per table alongside the query suite.

The two pins are NOT byte-matched: the parquet arm pins the union of columns the 22 queries
reference (`cols=[...]`), the .hpln arm pins the whole file, because a .hpln pin has no column
subset. That asymmetry costs the .hpln arm pin time and host memory and does NOT cost it query
time -- a projected serve already fetches only the selected columns' payload buffers
(compression_converters.cpp). So it understates the .hpln pin; the footprint is reported so the
size of the handicap is visible rather than assumed.

  SIRIUS_CONFIG_FILE=bench/chunk-skipping/sirius-sf100-hpln.yaml \
    bench-lock.sh pixi run python bench/chunk-skipping/hpln-pin-bench.py \
      --arms parquet,parquet-clustered,hpln-sorted,hpln-unsorted \
      --parquet /datasets/tpch_sf100 \
      --hpln-sorted /datasets/tpch_sf100_hpln_sorted \
      --hpln-unsorted /datasets/tpch_sf100_hpln_unsorted
"""
import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "test/tpch_performance"))
EXTENSION_PATH = os.path.join(REPO, "build/release/extension/sirius/sirius.duckdb_extension")
QUERY_DIR = os.path.join(REPO, "test/tpch_performance/tpch_queries/orig")
TABLES = ["lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region"]

# Which tables the .hpln arms sort, mirroring mk-hpln.py, so the clustered parquet arm asks for
# the same order at pin time and the two are comparable.
CLUSTER_KEYS = {"lineitem": "l_shipdate", "orders": "o_orderdate"}


def parquet_source(d, table):
    files = []
    for pattern in (f"{table}.parquet", f"{table}_*.parquet", os.path.join(table, "*.parquet")):
        files.extend(glob.glob(os.path.join(d, pattern)))
    if not files:
        raise FileNotFoundError(f"no parquet for {table} in {d}")
    return "read_parquet([" + ",".join(f"'{f}'" for f in sorted(set(files))) + "])"


def hpln_source(d, table):
    path = os.path.join(d, f"{table}.hpln")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no .hpln for {table} in {d}")
    return f"read_simpatico('{path}')"


def arm_config(arm, args):
    """(views, pin_statements) for `arm`, as {table: sql} and [(table, sql)]."""
    from tpch_pin_columns import detect_pin_glob, union_columns_by_table

    if arm.startswith("hpln"):
        d = args.hpln_sorted if arm == "hpln-sorted" else args.hpln_unsorted
        views = {t: hpln_source(d, t) for t in TABLES}
        pins = [
            (t, f"CALL pin_table('{os.path.join(d, f'{t}.hpln')}', format => 'simpatico', "
                f"tier => 'host', name => '{t}');")
            for t in TABLES
        ]
        return views, pins

    views = {t: parquet_source(args.parquet, t) for t in TABLES}
    cols_by_table = union_columns_by_table()
    pins = []
    for table, cols in cols_by_table.items():
        col_literals = ",".join(f"'{c}'" for c in cols)
        cluster = ""
        if arm == "parquet-clustered":
            key = CLUSTER_KEYS.get(table)
            if key and key in cols:
                cluster = f", cluster_by=['{key}']"
        path = detect_pin_glob(args.parquet, table)
        pins.append((table, f"CALL pin_table('{path}', tier => 'host', name => '{table}', "
                            f"cols=[{col_literals}]{cluster});"))
    return views, pins


def drop_cache(paths):
    """Evict `paths` from the OS page cache, so a pin is measured against storage.

    A pin is a one-time cold operation and SF100 fits in this box's page cache several times
    over, so a warm pin time is a measurement of the cache, not of the path. posix_fadvise
    touches only these files and needs no root.
    """
    os.sync()
    for path in paths:
        try:
            with open(path, "rb") as fh:
                os.posix_fadvise(fh.fileno(), 0, os.fstat(fh.fileno()).st_size,
                                 os.POSIX_FADV_DONTNEED)
        except OSError as e:
            print(f"  WARNING: fadvise({path}): {e}", flush=True)


def arm_files(arm, args):
    """Every input file the arm reads, for cache eviction."""
    if arm.startswith("hpln"):
        d = args.hpln_sorted if arm == "hpln-sorted" else args.hpln_unsorted
        return [os.path.join(d, f"{t}.hpln") for t in TABLES]
    files = []
    for t in TABLES:
        for pattern in (f"{t}.parquet", f"{t}_*.parquet", os.path.join(t, "*.parquet")):
            files.extend(glob.glob(os.path.join(args.parquet, pattern)))
    return sorted(set(files))


def normalize(rows):
    def cell(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return [tuple(cell(v) for v in r) for r in rows]


def run_arm(arm, args, reference):
    import duckdb

    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXTENSION_PATH}'")
    con.execute("SET gpu_execution = true")
    if not arm.startswith("hpln"):
        # The parquet arm has to compress at pin time to be the same representation the .hpln
        # already is; without this it pins raw and the comparison is about compression, not I/O.
        con.execute("SET pin_table_compression = true")
        con.execute(f"SET pin_table_input_compression_plan_dir = '{args.plan_dir}'")

    views, pins = arm_config(arm, args)
    for table, source in views.items():
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM {source};")

    if args.drop_cache:
        drop_cache(arm_files(arm, args))

    pin_times = {}
    t_pin0 = time.time()
    for table, sql in pins:
        t0 = time.time()
        con.execute(sql)
        pin_times[table] = time.time() - t0
    pin_total = time.time() - t_pin0
    print(f"  pin: {pin_total:7.2f}s total  " +
          "  ".join(f"{t}={pin_times[t]:.1f}" for t, _ in pins if pin_times[t] >= 0.05),
          flush=True)

    best, results = {}, {}
    for q in args.queries:
        sql = open(os.path.join(QUERY_DIR, f"q{q}.sql")).read()
        times = []
        for _ in range(args.iterations):
            t0 = time.time()
            rows = con.execute(sql).fetchall()
            times.append(time.time() - t0)
        best[q] = min(times)
        results[q] = normalize(rows)

    mismatched = []
    if reference is not None:
        for q in args.queries:
            if results[q] != reference[q] and sorted(results[q]) != sorted(reference[q]):
                mismatched.append(q)

    for table, _ in pins:
        con.execute(f"CALL unpin_table('{table}');")
    con.close()
    return dict(pin_total=pin_total, pin_times=pin_times, best=best,
                suite=sum(best.values()), mismatched=mismatched), results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--hpln-sorted")
    ap.add_argument("--hpln-unsorted")
    ap.add_argument("--arms", default="parquet,parquet-clustered,hpln-sorted,hpln-unsorted")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--queries", default=",".join(str(i) for i in range(1, 23)))
    ap.add_argument("--plan-dir", default=os.path.join(REPO, "bench/chunk-skipping/plans-hpln"))
    ap.add_argument("--drop-cache", action="store_true",
                    help="evict the arm's input files from the page cache before pinning")
    ap.add_argument("--out", default=os.path.join(REPO, "bench/chunk-skipping/results-hpln-pin"))
    args = ap.parse_args()
    args.queries = [int(q) for q in args.queries.split(",")]

    os.makedirs(args.out, exist_ok=True)
    report, reference = {}, None
    for arm in args.arms.split(","):
        arm = arm.strip()
        print(f"######## {arm}", flush=True)
        r, results = run_arm(arm, args, reference)
        if reference is None:
            reference = results
        report[arm] = r
        print(f"  suite: {r['suite']:7.3f}s (sum of per-query bests over {args.iterations} iters)"
              + (f"  *** MISMATCHED {r['mismatched']}" if r["mismatched"] else "  results agree"),
              flush=True)

    base = report.get("parquet")
    print("\narm                 pin (s)   suite (s)   vs parquet")
    for arm, r in report.items():
        delta = f"{(r['suite'] / base['suite'] - 1) * 100:+6.2f}%" if base else "     --"
        print(f"{arm:<18} {r['pin_total']:8.2f}  {r['suite']:9.3f}   {delta}")

    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
