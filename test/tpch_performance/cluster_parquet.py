# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
"""Physically sort TPC-H parquet tables so their row-group statistics prune.

Called by ``generate_tpch_data.sh --format parquet --cluster``. Each table named in the
cluster-key spec is rewritten in sort-key order, so every row group covers a narrow range
of that key and a range predicate can skip most of them; the remaining tables are copied
through unchanged.

DuckDB performs the ORDER BY because it sorts larger-than-memory inputs, and pyarrow writes
the result because it stores DECIMAL as FIXED_LEN_BYTE_ARRAY -- the same physical encoding
tpchgen-rs produces. Writing through DuckDB instead would silently re-encode decimals as
INT32/INT64 and produce a dataset that no longer matches the unclustered one it is meant to
be compared against.
"""

import argparse
import shutil
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

TPCH_TABLES = (
    "nation",
    "region",
    "part",
    "supplier",
    "partsupp",
    "customer",
    "orders",
    "lineitem",
)


def find_table_files(input_dir: Path, table: str) -> list[Path]:
    """Return the parquet files holding one table, across the layouts tpchgen-rs emits.

    A table is written either as a single file, as a numbered set of files, or as a
    directory of parts, depending on the partition count requested at generation time.
    """
    candidates: list[Path] = []
    single = input_dir / f"{table}.parquet"
    if single.is_file():
        candidates.append(single)
    candidates.extend(sorted(input_dir.glob(f"{table}.*.parquet")))
    candidates.extend(sorted(input_dir.glob(f"{table}_*.parquet")))
    table_dir = input_dir / table
    if table_dir.is_dir():
        candidates.extend(sorted(table_dir.glob("*.parquet")))
    # A single file also matches the numbered globs on some layouts; keep first occurrences.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def parse_keys(spec: str) -> dict[str, str]:
    """Parse a "table:col,table:col" cluster-key spec, rejecting unknown table names."""
    keys: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            sys.exit(f"ERROR: malformed --keys entry '{pair}' (expected table:column)")
        table, column = pair.split(":", 1)
        table, column = table.strip(), column.strip()
        if table not in TPCH_TABLES:
            print(f"WARNING: --keys references unknown table '{table}' (ignored)")
            continue
        keys[table] = column
    return keys


def sort_table(
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    column: str,
    destination: Path,
    row_group_size: int,
    compression: str,
) -> int:
    """Stream one table through DuckDB's ORDER BY into a single sorted parquet file.

    Batches are accumulated to ``row_group_size`` rows before each write so the row-group
    granularity -- which is what determines how selective the resulting zone maps are --
    does not depend on DuckDB's internal batch size.
    """
    sources = [str(path) for path in files]
    query = f'SELECT * FROM read_parquet(?) ORDER BY "{column}"'
    result = con.execute(query, [sources])
    # to_arrow_reader is the current spelling; fetch_record_batch is its pre-1.4 name.
    to_reader = getattr(result, "to_arrow_reader", None) or result.fetch_record_batch
    reader = to_reader(131_072)

    writer: pq.ParquetWriter | None = None
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    total_rows = 0

    def flush(*, final: bool = False) -> None:
        nonlocal pending, pending_rows
        if not pending:
            return
        table = pa.Table.from_batches(pending)
        # Emit whole row groups only and carry the remainder forward: writing the raw
        # accumulation would split every ~row_group_size-plus-epsilon batch into one full
        # group and one small fragment, and fragmented groups defeat the selectivity this
        # tool exists to create. Only the final flush may write a short group.
        whole = (
            table.num_rows
            if final
            else (table.num_rows // row_group_size) * row_group_size
        )
        if whole:
            writer.write_table(table.slice(0, whole), row_group_size=row_group_size)
        remainder = table.slice(whole)
        pending = remainder.to_batches()
        pending_rows = remainder.num_rows

    try:
        for batch in reader:
            if writer is None:
                # store_decimal_as_integer=False is the default, but it is the whole
                # reason this writer is pyarrow rather than DuckDB, so it is stated
                # rather than assumed.
                writer = pq.ParquetWriter(
                    destination,
                    batch.schema,
                    compression=compression,
                    store_decimal_as_integer=False,
                )
            pending.append(batch)
            pending_rows += batch.num_rows
            total_rows += batch.num_rows
            if pending_rows >= row_group_size:
                flush()
        flush(final=True)
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="unsorted parquet directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="destination directory"
    )
    parser.add_argument(
        "--keys",
        required=True,
        help='sort keys as "table:col,table:col" (tables not listed are copied unchanged)',
    )
    parser.add_argument("--row-group-size", type=int, default=1_000_000)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument(
        "--threads", type=int, default=0, help="DuckDB threads (0 = default)"
    )
    parser.add_argument(
        "--memory-limit", default="", help="DuckDB memory limit, e.g. '32GB'"
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        sys.exit(f"ERROR: input directory not found: {args.input}")
    sort_columns = parse_keys(args.keys)
    args.output.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    if args.threads:
        con.execute(f"SET threads={args.threads}")
    if args.memory_limit:
        con.execute(f"SET memory_limit='{args.memory_limit}'")
    # Sorting a table far larger than memory is the expected case at high scale factors.
    con.execute("SET preserve_insertion_order=false")

    for table in TPCH_TABLES:
        files = find_table_files(args.input, table)
        if not files:
            print(f"  {table}: no parquet files found, skipping")
            continue
        column = sort_columns.get(table)
        if column is None:
            # Preserve each file's path relative to the input root: the directory layout
            # (<table>/part.N.parquet) uses generic part names that would collide across
            # tables if flattened into the output root.
            for path in files:
                dest = args.output / path.relative_to(args.input)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            print(f"  {table}: copied {len(files)} file(s) unsorted")
            continue
        destination = args.output / f"{table}.parquet"
        rows = sort_table(
            con,
            files,
            column,
            destination,
            args.row_group_size,
            args.compression,
        )
        print(f"  {table}: sorted {rows} rows by {column} -> {destination.name}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
