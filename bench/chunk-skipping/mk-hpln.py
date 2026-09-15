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
import threading
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTENSION_PATH = os.path.join(
    REPO, "build/release/extension/sirius/sirius.duckdb_extension"
)

TABLES = [
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
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
        digits = [
            int(p) for p in "".join(c if c.isdigit() else " " for c in name).split()
        ]
        return (digits, name)

    return sorted(set(candidates), key=key)


def parse_bytes(text):
    units = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "K": 2**10, "M": 2**20, "G": 2**30}
    t = text.strip().upper()
    for suffix, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if t.endswith(suffix):
            return int(float(t[: -len(suffix)]) * mult)
    return int(t)


def drop_from_page_cache(paths):
    """Ask the kernel to forget `paths`, whichever process cached them.

    posix_fadvise is per-inode, so this works on a file another library is writing: it drops the
    CLEAN pages and leaves dirty ones to writeback. That matters at this scale -- a table's COPY
    reads ~223 GB of parquet and writes ~220 GB of .hpln, and every byte of both lands in page
    cache. The cache is reclaimable, so nothing is actually short of memory, but `free` reads as
    almost none and anything watching that number concludes the machine is about to die. It killed
    this generation 30 minutes in, one chunk from the end.
    """
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        finally:
            os.close(fd)


class cache_trimmer:
    """Background thread trimming the page cache while a long COPY runs, and logging memory.

    The logging is not decoration: if a run is killed again, the trace says whether the process's
    own RSS grew (DuckDB buffering ahead of a single-threaded sink) or only the cache did.
    """

    def __init__(self, paths, interval=20.0):
        self._paths = list(paths)
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=self._interval + 5)
        drop_from_page_cache(self._paths)

    def _run(self):
        while not self._stop.wait(self._interval):
            drop_from_page_cache(self._paths)
            try:
                rss_kb = int(
                    subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", str(os.getpid())]
                    )
                )
                with open("/proc/meminfo") as fh:
                    info = {
                        k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in fh)
                    }
                print(
                    f"   [mem] rss {rss_kb/1e6:.1f} GB  free {info['MemFree']/1e6:.0f} GB  "
                    f"available {info['MemAvailable']/1e6:.0f} GB  "
                    f"cached {info['Cached']/1e6:.0f} GB",
                    flush=True,
                )
            except Exception:
                pass


def hpln_is_complete(path):
    """True when `path` is a .hpln a reader can open — i.e. a COPY finished it.

    An interrupted write leaves payload bytes and no trailer, which is exactly what makes this
    checkable: the file is large and unreadable rather than short and plausible.
    """
    import struct

    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size < 16:
                return False
            fh.seek(-16, os.SEEK_END)
            trailer = fh.read(16)
    except OSError:
        return False
    if trailer[12:] != b"HPLN":
        return False
    # The magic alone is not proof: check that the postscript it points at actually ends where the
    # file does. A truncated write that happened to end in those four bytes would pass otherwise.
    ps_offset, ps_len, _version = struct.unpack("<QHH", trailer[:12])
    return ps_offset + ps_len + 16 == size


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
        "BOOLEAN": 1,
        "TINYINT": 1,
        "SMALLINT": 2,
        "INTEGER": 4,
        "BIGINT": 8,
        "UTINYINT": 1,
        "USMALLINT": 2,
        "UINTEGER": 4,
        "UBIGINT": 8,
        "FLOAT": 4,
        "DOUBLE": 8,
        "DATE": 4,
        "TIME": 8,
        "TIMESTAMP": 8,
        "TIMESTAMP WITH TIME ZONE": 8,
    }
    schema = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    total, varchars = 0, []
    for name, dtype, *_ in schema:
        base = dtype.split("(")[0].strip()
        if base == "DECIMAL":
            precision = int(dtype[dtype.index("(") + 1 : dtype.index(",")])
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
    ap.add_argument(
        "--output", required=True, help="directory to write <table>.hpln into"
    )
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
        default=os.path.join(
            REPO, "src/compression/simpatico_codegen/plans/tpch_sf1000"
        ),
        help="compression plans; a table with no <table>.txt is stored uncompressed",
    )
    ap.add_argument(
        "--sort",
        choices=["global", "cluster", "none"],
        default="global",
        help="global: ORDER BY the table's sort key in the COPY query (whole chunks prune). "
        "cluster: sort each chunk as it is written, the analogue of pin_table's cluster_by "
        "(chunks still span the key range; the group index does the pruning). none: file order",
    )
    ap.add_argument("--config", default=os.environ.get("SIRIUS_CONFIG_FILE"))
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip a table whose .hpln already carries a trailer, so a long generation that was "
        "interrupted does not restart from the first table",
    )
    ap.add_argument(
        "--no-trim-cache",
        action="store_true",
        help="do not drop the inputs and output from the page cache as the COPY runs",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the post-write row count, which re-reads the whole file — at SF1000 that is a "
        "second 220 GB pass per table, and hpln-suite.py is the check that actually matters",
    )
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
        # `cluster_by` sorts each chunk as it is written, the way pin_table's cluster_by sorts each
        # pin chunk: every chunk still spans the whole key range, so nothing prunes at chunk level,
        # but the 8192-row groups inside it narrow and the group index does the work. Strictly
        # cheaper than --sort global (no whole-table sort) and strictly weaker.
        cluster = ""
        if args.sort == "cluster" and key:
            cluster = f", cluster_by '{key}'"
        opts = (
            f"FORMAT simpatico, chunk_rows {chunk_rows}, "
            f"group_rows {args.group_rows}, plan_table '{table}'{cluster}"
        )
        has_plan = os.path.exists(os.path.join(args.plan_dir, f"{table}.txt"))
        if not has_plan:
            opts = (
                f"FORMAT simpatico, chunk_rows {chunk_rows}, "
                f"group_rows {args.group_rows}{cluster}"
            )

        if args.resume and hpln_is_complete(out):
            print(
                f"== {table}: already complete at {out}, skipping (--resume)",
                flush=True,
            )
            continue

        print(
            f"== {table}: {len(files)} parquet file(s), {parquet_bytes/1e9:.2f} GB"
            f"{' ORDER BY ' + key if order else ''}"
            f"{'' if has_plan else ' [no plan: stored uncompressed]'}",
            flush=True,
        )
        t0 = time.time()
        if args.no_trim_cache:
            con.execute(f"COPY ({select}) TO '{out}' ({opts});")
        else:
            with cache_trimmer(files + [out]):
                con.execute(f"COPY ({select}) TO '{out}' ({opts});")
        elapsed = time.time() - t0
        hpln_bytes = os.path.getsize(out)
        rows = -1
        if not args.no_verify:
            rows = con.execute(
                f"SELECT count(*) FROM read_simpatico('{out}')"
            ).fetchone()[0]
            drop_from_page_cache([out])
        print(
            f"   -> {hpln_bytes/1e9:.2f} GB ({parquet_bytes/hpln_bytes:.2f}x parquet), "
            f"{rows} rows, {elapsed:.1f} s, "
            f"{max(1, -(-rows // chunk_rows))} chunk(s) of {chunk_rows} rows"
            + (
                f" (~{bytes_per_row * chunk_rows / 1e9:.2f} GB decoded)"
                if bytes_per_row
                else ""
            ),
            flush=True,
        )
        report.append(
            dict(
                table=table,
                rows=rows,
                chunk_rows=chunk_rows,
                seconds=round(elapsed, 2),
                parquet_bytes=parquet_bytes,
                hpln_bytes=hpln_bytes,
                sort_key=key if order else None,
                plan=has_plan,
            )
        )

    with open(os.path.join(args.output, "mk-hpln.json"), "w") as fh:
        json.dump(
            dict(
                input=args.input,
                chunk_bytes=args.chunk_bytes,
                group_rows=args.group_rows,
                sort=args.sort,
                tables=report,
            ),
            fh,
            indent=2,
        )
    total_p = sum(r["parquet_bytes"] for r in report)
    total_h = sum(r["hpln_bytes"] for r in report)
    print(
        f"\nTOTAL: parquet {total_p/1e9:.2f} GB -> hpln {total_h/1e9:.2f} GB "
        f"({total_p/max(total_h,1):.2f}x), {sum(r['seconds'] for r in report):.1f} s"
    )


if __name__ == "__main__":
    main()
