#!/usr/bin/env python3
"""
Rewrite TPC-H parquet files with GPU-optimized settings using cudf.

Reads each table from the source directory and writes it back with:
  - Snappy compression
  - Parquet V2 page headers
  - Configurable row group size (default 10M rows)
  - 8 MiB max page size
  - Statistics at ROWGROUP level
  - Dictionary encoding enabled

Large tables are processed by reading row groups in batches via cudf,
then writing via pyarrow ParquetWriter (single output file with append).
The original parquet schema is preserved (dates stay as dates, etc.).

Usage:
  pixi run python rewrite_parquet.py <source_dir> <dest_dir> [row_group_rows]

Example:
  pixi run python rewrite_parquet.py \
      ../../test_datasets/tpch_parquet_sf100 \
      ../../test_datasets/tpch_parquet_sf100_optimized \
      10000000
"""

import os
import sys
import time
import glob

import cudf
import pyarrow as pa
import pyarrow.parquet as pq

TPCH_TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]

MAX_PAGE_SIZE_BYTES = 8 * 1024 * 1024  # 8 MiB

# Tables with fewer rows than this are read in one shot with cudf
SMALL_TABLE_THRESHOLD = 50_000_000


def cudf_write_kwargs(row_group_size_rows):
    return dict(
        compression="snappy",
        header_version="2.0",
        row_group_size_rows=row_group_size_rows,
        max_page_size_bytes=MAX_PAGE_SIZE_BYTES,
        statistics="ROWGROUP",
        use_dictionary=True,
        index=False,
    )


def cast_to_schema(arrow_table, target_schema):
    """Cast an arrow table to match the target schema types."""
    arrays = []
    for i, field in enumerate(target_schema):
        col = arrow_table.column(i)
        if col.type != field.type:
            col = col.cast(field.type)
        arrays.append(col)
    return pa.table(arrays, schema=target_schema)


def rewrite_table(table_name, source_dir, dest_dir, row_group_size_rows):
    """Read a table's parquet file(s) with cudf and write back optimized."""
    single = os.path.join(source_dir, f"{table_name}.parquet")
    partitioned = sorted(glob.glob(os.path.join(source_dir, f"{table_name}_*.parquet")))

    source_files = []
    if os.path.isfile(single):
        source_files.append(single)
    source_files.extend(partitioned)

    if not source_files:
        print(f"  WARNING: No parquet files found for {table_name}, skipping")
        return

    # Get metadata and original schema
    orig_schema = pq.read_schema(source_files[0])
    total_rows = 0
    for f in source_files:
        total_rows += pq.read_metadata(f).num_rows
    src_size = sum(os.path.getsize(f) for f in source_files)
    print(f"  {table_name}: {total_rows:,} rows, {src_size / 1e9:.2f} GB on disk")

    dest_path = os.path.join(dest_dir, f"{table_name}.parquet")
    t0 = time.time()

    if total_rows <= SMALL_TABLE_THRESHOLD:
        # Small enough — read with cudf, cast back to original schema, write with pyarrow
        df = cudf.read_parquet(source_files)
        arrow_table = cast_to_schema(df.to_arrow(), orig_schema)
        del df
        pq.write_table(
            arrow_table,
            dest_path,
            compression="snappy",
            version="2.6",
            data_page_version="2.0",
            write_statistics=True,
            use_dictionary=True,
            data_page_size=MAX_PAGE_SIZE_BYTES,
            row_group_size=row_group_size_rows,
        )
        total_written = len(arrow_table)
        del arrow_table
    else:
        # Large table — read row groups in batches with cudf,
        # cast to original schema, write with pyarrow ParquetWriter
        meta = pq.read_metadata(source_files[0])
        num_rgs = meta.num_row_groups
        rows_per_src_rg = meta.row_group(0).num_rows

        rgs_per_batch = max(1, row_group_size_rows // rows_per_src_rg)
        print(f"    {num_rgs} source row groups of {rows_per_src_rg:,} rows each")
        print(
            f"    Reading {rgs_per_batch} source RGs per batch -> ~{rgs_per_batch * rows_per_src_rg:,} rows/batch"
        )

        writer = pq.ParquetWriter(
            dest_path,
            orig_schema,
            compression="snappy",
            version="2.6",
            data_page_version="2.0",
            write_statistics=True,
            use_dictionary=True,
            data_page_size=MAX_PAGE_SIZE_BYTES,
        )
        total_written = 0

        for batch_start in range(0, num_rgs, rgs_per_batch):
            batch_end = min(batch_start + rgs_per_batch, num_rgs)
            rg_indices = list(range(batch_start, batch_end))

            df = cudf.read_parquet(source_files[0], row_groups=[rg_indices])
            arrow_table = cast_to_schema(df.to_arrow(), orig_schema)
            del df

            writer.write_table(arrow_table)
            total_written += len(arrow_table)
            del arrow_table

            pct = total_written * 100 // total_rows
            print(
                f"    Wrote row group: {total_written:,} / {total_rows:,} rows ({pct}%)"
            )

        writer.close()

    elapsed = time.time() - t0
    dst_size = os.path.getsize(dest_path)
    print(
        f"    Wrote {total_written:,} rows in {elapsed:.1f}s  "
        f"({src_size / 1e9:.2f} GB -> {dst_size / 1e9:.2f} GB, "
        f"{dst_size / src_size:.1%})"
    )


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <dest_dir> [row_group_rows]")
        sys.exit(1)

    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    row_group_size_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000

    if not os.path.isdir(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Rewriting TPC-H parquet: {source_dir} -> {dest_dir}")
    print(f"  Row group size: {row_group_size_rows:,} rows")
    print(f"  Max page size: {MAX_PAGE_SIZE_BYTES // (1024*1024)} MiB")
    print(f"  Compression: snappy")
    print(f"  Header version: 2.0 (V2 pages)")
    print()

    t_total = time.time()
    for table in TPCH_TABLES:
        rewrite_table(table, source_dir, dest_dir, row_group_size_rows)
        print()

    elapsed = time.time() - t_total
    print(f"Done. Total time: {elapsed:.1f}s")
    print(f"Output: {dest_dir}")


if __name__ == "__main__":
    main()
