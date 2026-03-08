#!/usr/bin/env python3
"""
Rewrite TPC-H parquet files with GPU-optimized settings.

Uses cudf (GPU) if available, otherwise falls back to pyarrow (CPU-only).
Reads each table from the source directory and writes it back with:
  - Snappy compression
  - Parquet V2 page headers
  - Configurable row group size (default 10M rows)
  - 8 MiB max page size
  - Statistics at ROWGROUP level
  - Dictionary encoding enabled
  - Max file size limit (default 20 GB) — large tables are split into
    multiple numbered files (e.g., lineitem_0000.parquet, lineitem_0001.parquet)

Large tables are processed by reading row groups in batches,
then writing via pyarrow ParquetWriter.
The original parquet schema is preserved (dates stay as dates, etc.).

Usage:
  pixi run python rewrite_parquet.py <source_dir> <dest_dir> [row_group_rows] [max_file_gb]

Example:
  pixi run python rewrite_parquet.py \
      ../../test_datasets/tpch_parquet_sf100 \
      ../../test_datasets/tpch_parquet_sf100_optimized \
      10000000 20
"""

import os
import sys
import time
import glob

try:
    import cudf

    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

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
DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB

# Tables with fewer rows than this are read in one shot with cudf
SMALL_TABLE_THRESHOLD = 50_000_000

# Columns to cast to int32. l_orderkey and o_orderkey stay int64 because they
# exceed the int32 range at large scale factors.
INT32_COLUMNS = {
    "customer": {"c_custkey", "c_nationkey"},
    "lineitem": {"l_partkey", "l_suppkey", "l_linenumber"},
    "nation": {"n_nationkey", "n_regionkey"},
    "orders": {"o_custkey"},
    "part": {"p_partkey"},
    "partsupp": {"ps_partkey", "ps_suppkey"},
    "region": {"r_regionkey"},
    "supplier": {"s_suppkey", "s_nationkey"},
}


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


def apply_int32_overrides(schema, table_name):
    """Return a new schema with key columns downcast to int32 where specified."""
    overrides = INT32_COLUMNS.get(table_name, set())
    if not overrides:
        return schema
    new_fields = []
    for field in schema:
        if field.name in overrides and pa.types.is_integer(field.type):
            new_fields.append(field.with_type(pa.int32()))
        else:
            new_fields.append(field)
    return pa.schema(new_fields)


def cast_to_schema(arrow_table, target_schema):
    """Cast an arrow table to match the target schema types."""
    arrays = []
    for i, field in enumerate(target_schema):
        col = arrow_table.column(i)
        if col.type != field.type:
            col = col.cast(field.type)
        arrays.append(col)
    return pa.table(arrays, schema=target_schema)


def _make_writer(path, schema):
    """Create a ParquetWriter with standard settings."""
    return pq.ParquetWriter(
        path,
        schema,
        compression="snappy",
        version="2.6",
        data_page_version="2.0",
        write_statistics=True,
        use_dictionary=True,
        data_page_size=MAX_PAGE_SIZE_BYTES,
    )


def _is_oom(e):
    """Check if an exception is a GPU out-of-memory error."""
    name = type(e).__name__
    msg = str(e).lower()
    return "MemoryError" in name or "bad_alloc" in msg or "out of memory" in msg


def _read_small_table(source_files, target_schema, use_gpu):
    """Read a small table in one shot. Returns None on GPU OOM."""
    if use_gpu:
        try:
            df = cudf.read_parquet(source_files)
            table = cast_to_schema(df.to_arrow(), target_schema)
            del df
            return table
        except Exception as e:
            if _is_oom(e):
                return None
            raise
    return cast_to_schema(pq.read_table(source_files), target_schema)


def _read_batch(source_file, rg_indices, target_schema, use_gpu):
    """Read a batch of row groups. Returns (table, used_gpu)."""
    if use_gpu:
        try:
            df = cudf.read_parquet(source_file, row_groups=[rg_indices])
            table = cast_to_schema(df.to_arrow(), target_schema)
            del df
            return table, True
        except Exception as e:
            if _is_oom(e):
                print(f"    GPU OOM on batch, falling back to pyarrow")
            else:
                raise
    pf = pq.ParquetFile(source_file)
    tables = [pf.read_row_group(i) for i in rg_indices]
    table = cast_to_schema(pa.concat_tables(tables), target_schema)
    del tables
    return table, False


def _dest_path(dest_dir, table_name, file_idx, multi_file):
    """Return the output path, using numbered suffix only when splitting."""
    if multi_file:
        return os.path.join(dest_dir, f"{table_name}_{file_idx:04d}.parquet")
    return os.path.join(dest_dir, f"{table_name}.parquet")


def rewrite_table(
    table_name, source_dir, dest_dir, row_group_size_rows, max_file_bytes
):
    """Read a table's parquet file(s) with cudf and write back optimized."""
    single = os.path.join(source_dir, f"{table_name}.parquet")
    partitioned = sorted(glob.glob(os.path.join(source_dir, f"{table_name}_*.parquet")))
    # tpchgen-cli produces <table>/<table>.<part>.parquet subdirectory layout
    subdir = sorted(
        glob.glob(os.path.join(source_dir, table_name, f"{table_name}.*.parquet"))
    )

    source_files = []
    if os.path.isfile(single):
        source_files.append(single)
    source_files.extend(partitioned)
    source_files.extend(subdir)

    if not source_files:
        print(f"  WARNING: No parquet files found for {table_name}, skipping")
        return

    # Get metadata and original schema
    orig_schema = pq.read_schema(source_files[0])
    target_schema = apply_int32_overrides(orig_schema, table_name)
    total_rows = 0
    for f in source_files:
        total_rows += pq.read_metadata(f).num_rows
    src_size = sum(os.path.getsize(f) for f in source_files)
    print(f"  {table_name}: {total_rows:,} rows, {src_size / 1e9:.2f} GB on disk")

    t0 = time.time()
    output_files = []

    use_gpu = HAS_CUDF

    if total_rows <= SMALL_TABLE_THRESHOLD:
        # Small enough — read in one shot, cast to target schema, write with pyarrow
        arrow_table = _read_small_table(source_files, target_schema, use_gpu)
        if arrow_table is None:
            # cudf OOM — retry with pyarrow
            print(f"    GPU OOM on small table, falling back to pyarrow")
            use_gpu = False
            arrow_table = _read_small_table(source_files, target_schema, False)
        dest_path = _dest_path(dest_dir, table_name, 0, False)
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
        output_files.append(dest_path)
        del arrow_table
    else:
        # Large table — read row groups in batches, cast to target schema,
        # write with pyarrow ParquetWriter.
        # Split into multiple output files if they exceed max_file_bytes.
        meta = pq.read_metadata(source_files[0])
        num_rgs = meta.num_row_groups
        rows_per_src_rg = meta.row_group(0).num_rows

        rgs_per_batch = max(1, row_group_size_rows // rows_per_src_rg)
        backend = "cudf" if use_gpu else "pyarrow"
        print(f"    Backend: {backend}")
        print(f"    {num_rgs} source row groups of {rows_per_src_rg:,} rows each")
        print(
            f"    Reading {rgs_per_batch} source RGs per batch -> ~{rgs_per_batch * rows_per_src_rg:,} rows/batch"
        )
        print(f"    Max file size: {max_file_bytes / (1024**3):.0f} GiB")

        # Estimate whether we'll need multiple files based on source size ratio
        est_output = src_size  # conservative: output ~ input size
        will_split = est_output > max_file_bytes

        file_idx = 0
        dest_path = _dest_path(dest_dir, table_name, file_idx, will_split)
        writer = _make_writer(dest_path, target_schema)
        output_files.append(dest_path)
        total_written = 0

        for batch_start in range(0, num_rgs, rgs_per_batch):
            # Check if current file exceeds limit and roll to a new one
            if total_written > 0 and os.path.getsize(dest_path) >= max_file_bytes:
                writer.close()
                print(
                    f"    File {dest_path}: {os.path.getsize(dest_path) / 1e9:.2f} GB (limit reached)"
                )
                file_idx += 1
                if not will_split:
                    # We didn't predict splitting — rename first file to numbered format
                    old_path = _dest_path(dest_dir, table_name, 0, False)
                    new_path = _dest_path(dest_dir, table_name, 0, True)
                    os.rename(old_path, new_path)
                    output_files[0] = new_path
                    will_split = True
                dest_path = _dest_path(dest_dir, table_name, file_idx, True)
                writer = _make_writer(dest_path, target_schema)
                output_files.append(dest_path)

            batch_end = min(batch_start + rgs_per_batch, num_rgs)
            rg_indices = list(range(batch_start, batch_end))

            arrow_table, used_gpu = _read_batch(
                source_files[0], rg_indices, target_schema, use_gpu
            )
            if not used_gpu:
                use_gpu = False  # stay on pyarrow for remaining batches

            writer.write_table(arrow_table)
            total_written += len(arrow_table)
            del arrow_table

            pct = total_written * 100 // total_rows
            print(
                f"    Wrote row group: {total_written:,} / {total_rows:,} rows ({pct}%)"
            )

        writer.close()

    elapsed = time.time() - t0
    dst_size = sum(os.path.getsize(f) for f in output_files)
    files_str = f" across {len(output_files)} files" if len(output_files) > 1 else ""
    print(
        f"    Wrote {total_written:,} rows in {elapsed:.1f}s{files_str}  "
        f"({src_size / 1e9:.2f} GB -> {dst_size / 1e9:.2f} GB, "
        f"{dst_size / src_size:.1%})"
    )


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <source_dir> <dest_dir> [row_group_rows] [max_file_gb]"
        )
        sys.exit(1)

    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    row_group_size_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000
    max_file_gb = float(sys.argv[4]) if len(sys.argv) > 4 else 20
    max_file_bytes = int(max_file_gb * 1024 * 1024 * 1024)

    if not os.path.isdir(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    os.makedirs(dest_dir, exist_ok=True)
    backend = "cudf (GPU)" if HAS_CUDF else "pyarrow (CPU)"
    print(f"Rewriting TPC-H parquet: {source_dir} -> {dest_dir}")
    print(f"  Backend: {backend}")
    print(f"  Row group size: {row_group_size_rows:,} rows")
    print(f"  Max file size: {max_file_gb:.0f} GiB")
    print(f"  Max page size: {MAX_PAGE_SIZE_BYTES // (1024*1024)} MiB")
    print(f"  Compression: snappy")
    print(f"  Header version: 2.0 (V2 pages)")
    print()

    t_total = time.time()
    for table in TPCH_TABLES:
        rewrite_table(table, source_dir, dest_dir, row_group_size_rows, max_file_bytes)
        print()

    elapsed = time.time() - t_total
    print(f"Done. Total time: {elapsed:.1f}s")
    print(f"Output: {dest_dir}")


if __name__ == "__main__":
    main()
