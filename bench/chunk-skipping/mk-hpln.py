#!/usr/bin/env python3
"""Write a TPC-H parquet dataset out as one `.hpln` file per table.

`COPY (SELECT ... ) TO 't.hpln' (FORMAT simpatico)` is the only writer, so this is a thin
driver: resolve each table's parquet files in a deterministic order, register a view, and
copy it. A table is sorted when `SORT_KEYS` names a key for it -- the COPY is single
threaded, so the query's row order is the file's row order, and a global ORDER BY is what
makes whole chunks prunable (a per-chunk `cluster_by` leaves every chunk spanning the key
range, cf. CHUNK_SKIPPING_PLAN.md 7.9).

  pixi run python bench/chunk-skipping/mk-hpln.py \
      --input /datasets/tpch_sf100_sorted --output /datasets/tpch_sf100_hpln

Reports per table: rows, wall time, .hpln bytes vs parquet bytes.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTENSION_PATH = os.path.join(REPO, "build/release/extension/sirius/sirius.duckdb_extension")

TABLES = [
    "lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region",
]

# Sort keys mirror the clustered dataset the rest of the study uses: the date column each
# table's selective predicates hit. A table absent here is written in natural order.
SORT_KEYS = {
    "lineitem": "l_shipdate",
    "orders": "o_orderdate",
}


def resolve_files(parquet_dir, table):
    """Parquet files for `table`, in an order that preserves a global sort.

    Lexicographic order puts `part.10` before `part.2`, which shuffles a sorted dataset
    across file boundaries; sort on the numeric part instead.
    """
    candidates = []
    for pattern in (
        os.path.join(parquet_dir, f"{table}.parquet"),
        os.path.join(parquet_dir, f"{table}_*.parquet"),
        os.path.join(parquet_dir, table, "*.parquet"),
    ):
        candidates.extend(glob.glob(pattern))

    def key(path):
        name = os.path.basename(path)
        digits = [int(p) for p in "".join(c if c.isdigit() else " " for c in name).split()]
        return (digits, name)

    return sorted(set(candidates), key=key)


def parse_bytes(text):
    units = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "K": 2**10, "M": 2**20, "G": 2**30}
    t = text.strip().upper()
    for suffix, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if t.endswith(suffix):
            return int(float(t[: -len(suffix)]) * mult)
    return int(t)


def decoded_bytes_per_row(con, source):
    """Average cuDF footprint of one row of `source`.

    Decoded, not compressed, because that is what the batch budget counts: a scan's coalescer
    caps a batch at `approximate_batch_size` of "total decoded bytes"
    (duckdb_native_batch_coalescer.hpp). A chunk is what a pinned .hpln serves as one batch, so
    sizing chunks in decoded bytes is what makes a .hpln pin's batches the same size as a parquet
    pin's at the same scan_task_batch_size.

    Fixed-width columns are exact. A VARCHAR is its average byte length -- octet_length over a
    BLOB cast, since DuckDB's length() counts characters -- plus the 4-byte offset cuDF carries
    per row, sampled rather than scanned: the estimate only has to be right to within a factor,
    and a full pass over lineitem to size a chunk would cost more than it saves.
    """
    fixed = {
        "BOOLEAN": 1, "TINYINT": 1, "SMALLINT": 2, "INTEGER": 4, "BIGINT": 8,
        "UTINYINT": 1, "USMALLINT": 2, "UINTEGER": 4, "UBIGINT": 8,
        "FLOAT": 4, "DOUBLE": 8, "DATE": 4, "TIME": 8,
        "TIMESTAMP": 8, "TIMESTAMP WITH TIME ZONE": 8,
    }
    schema = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    total, varchars = 0, []
    for name, dtype, *_ in schema:
        base = dtype.split("(")[0].strip()
        if base == "DECIMAL":
            precision = int(dtype[dtype.index("(") + 1: dtype.index(",")])
            total += 4 if precision <= 9 else (8 if precision <= 18 else 16)
        elif base in ("VARCHAR", "BLOB"):
            total += 4  # cuDF's per-row offset
            varchars.append(name)
        elif base in fixed:
            total += fixed[base]
        else:
            total += 8  # unknown: assume a 64-bit carrier rather than refuse
    if varchars:
        avg = ", ".join(f"avg(octet_length(CAST({c} AS BLOB)))" for c in varchars)
        row = con.execute(
            f"SELECT {avg} FROM {source} USING SAMPLE 100000 ROWS"
        ).fetchone()
        total += sum(float(v or 0) for v in row)
    return max(1.0, total)


def chunk_rows_for(bytes_per_row, chunk_bytes):
    """Rows whose decoded footprint is about `chunk_bytes`."""
    return max(1, int(chunk_bytes / bytes_per_row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TPC-H parquet directory")
    ap.add_argument("--output", required=True, help="directory to write <table>.hpln into")
    ap.add_argument("--tables", default=",".join(TABLES))
    ap.add_argument("--chunk-rows", type=int, default=1 << 20)
    ap.add_argument(
        "--chunk-bytes",
        type=str,
        default=None,
        help="target DECODED bytes per chunk (e.g. 2GB); overrides --chunk-rows per table. "
        "A chunk is what a pinned entry serves as one batch, so this is the knob that makes a "
        "pinned .hpln batch like a parquet pin's scan_task_batch_size rather than 573 times "
        "smaller",
    )
    ap.add_argument("--group-rows", type=int, default=8192)
    ap.add_argument(
        "--plan-dir",
        default=os.path.join(REPO, "src/compression/simpatico_codegen/plans/tpch_sf1000"),
        help="compression plans; a table with no <table>.txt is stored uncompressed",
    )
    ap.add_argument(
        "--sort",
        choices=["global", "none"],
        default="global",
        help="global: ORDER BY the table's sort key in the COPY query",
    )
    ap.add_argument("--config", default=os.environ.get("SIRIUS_CONFIG_FILE"))
    args = ap.parse_args()

    if args.config:
        os.environ["SIRIUS_CONFIG_FILE"] = args.config
    import duckdb

    os.makedirs(args.output, exist_ok=True)
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXTENSION_PATH}'")
    con.execute(f"SET pin_table_input_compression_plan_dir = '{args.plan_dir}'")

    report = []
    for table in args.tables.split(","):
        table = table.strip()
        if not table:
            continue
        files = resolve_files(args.input, table)
        if not files:
            print(f"!! no parquet files for {table} in {args.input}", flush=True)
            continue
        parquet_bytes = sum(os.path.getsize(f) for f in files)
        file_list = ",".join(f"'{f}'" for f in files)
        out = os.path.join(args.output, f"{table}.hpln")

        chunk_rows = args.chunk_rows
        bytes_per_row = None
        if args.chunk_bytes:
            bytes_per_row = decoded_bytes_per_row(con, f"read_parquet([{file_list}])")
            chunk_rows = chunk_rows_for(bytes_per_row, parse_bytes(args.chunk_bytes))

        order = ""
        key = SORT_KEYS.get(table)
        if args.sort == "global" and key:
            order = f" ORDER BY {key}"
        select = f"SELECT * FROM read_parquet([{file_list}]){order}"
        opts = (
            f"FORMAT simpatico, chunk_rows {chunk_rows}, "
            f"group_rows {args.group_rows}, plan_table '{table}'"
        )
        has_plan = os.path.exists(os.path.join(args.plan_dir, f"{table}.txt"))
        if not has_plan:
            opts = f"FORMAT simpatico, chunk_rows {chunk_rows}, group_rows {args.group_rows}"

        print(
            f"== {table}: {len(files)} parquet file(s), {parquet_bytes/1e9:.2f} GB"
            f"{' ORDER BY ' + key if order else ''}"
            f"{'' if has_plan else ' [no plan: stored uncompressed]'}",
            flush=True,
        )
        t0 = time.time()
        con.execute(f"COPY ({select}) TO '{out}' ({opts});")
        elapsed = time.time() - t0
        hpln_bytes = os.path.getsize(out)
        rows = con.execute(f"SELECT count(*) FROM read_simpatico('{out}')").fetchone()[0]
        print(
            f"   -> {hpln_bytes/1e9:.2f} GB ({parquet_bytes/hpln_bytes:.2f}x parquet), "
            f"{rows} rows, {elapsed:.1f} s, "
            f"{max(1, -(-rows // chunk_rows))} chunk(s) of {chunk_rows} rows"
            + (f" (~{bytes_per_row * chunk_rows / 1e9:.2f} GB decoded)" if bytes_per_row else ""),
            flush=True,
        )
        report.append(
            dict(table=table, rows=rows, chunk_rows=chunk_rows, seconds=round(elapsed, 2),
                 parquet_bytes=parquet_bytes, hpln_bytes=hpln_bytes,
                 sort_key=key if order else None, plan=has_plan)
        )

    with open(os.path.join(args.output, "mk-hpln.json"), "w") as fh:
        json.dump(dict(input=args.input, chunk_bytes=args.chunk_bytes,
                       group_rows=args.group_rows, sort=args.sort, tables=report), fh, indent=2)
    total_p = sum(r["parquet_bytes"] for r in report)
    total_h = sum(r["hpln_bytes"] for r in report)
    print(f"\nTOTAL: parquet {total_p/1e9:.2f} GB -> hpln {total_h/1e9:.2f} GB "
          f"({total_p/max(total_h,1):.2f}x), {sum(r['seconds'] for r in report):.1f} s")


if __name__ == "__main__":
    main()
