---
name: dataset-manager
description: Manage TPC-H parquet datasets — generate data at any scale factor, consolidate small parquet files into fewer larger files, inspect dataset layout, and optimize row group sizes. Auto-selects cudf (GPU) or pyarrow (CPU) based on available GPU memory.
---

# TPC-H Dataset Manager

You are managing TPC-H parquet datasets for Sirius, a GPU-accelerated SQL query engine. Your job is to generate, inspect, consolidate, and optimize parquet files used for benchmarking and testing.

## Backend Selection

Before any data manipulation, detect available GPU memory:

```bash
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```

This returns total GPU memory in MiB. Convert to GB and apply the rule:
- **>= 80 GB GPU memory**: Use **cudf** (GPU-accelerated, much faster for large datasets)
- **< 80 GB GPU memory**: Use **pyarrow** (CPU-only, works everywhere)

Tell the user which backend you selected and why.

## TPC-H Tables

All 8 TPC-H tables: `customer`, `lineitem`, `nation`, `orders`, `part`, `partsupp`, `region`, `supplier`.

## Working Directory

All commands run from the **project root** (the repository root).
Scripts are in `test/tpch_performance/`.
Datasets live under `test_datasets/`.
The pixi environment for Python scripts: `cd test/tpch_performance && pixi run python ...`

## Workflows

### Workflow A: Generate TPC-H Data

Use `generate_tpch_data.sh` which clones and builds [sirius-db/tpchgen-rs](https://github.com/sirius-db/tpchgen-rs) from source, then generates parquet files with optimized row groups, encodings, and compression.

```bash
cd test/tpch_performance
pixi run bash generate_tpch_data.sh <scale_factor> [output_dir] [jobs]
```

Arguments:
- `scale_factor` — TPC-H scale factor (e.g. 1, 10, 100)
- `output_dir` — Output directory (default: `test_datasets/tpch_parquet_sf<SF>`)
- `jobs` — Number of parallel jobs (default: `nproc`)

Examples:
```bash
# Generate SF100 with default settings
cd test/tpch_performance
pixi run bash generate_tpch_data.sh 100

# Generate SF10 to a custom location with 8 jobs
cd test/tpch_performance
pixi run bash generate_tpch_data.sh 10 ../../test_datasets/tpch_sf10_custom 8
```

The script:
1. Clones `sirius-db/tpchgen-rs` to `test_datasets/tpchgen-rs/` (if not already present)
2. Builds `tpchgen-cli` with native CPU optimizations (`RUSTFLAGS="-C target-cpu=native"`)
3. Runs `scripts/generate_tpch.py` to produce parquet files

If the output directory already exists, the script skips generation. Remove the directory to regenerate.

### Workflow B: Consolidate / Optimize Parquet Files

This is the most common operation. Takes many small parquet files and consolidates them into fewer, larger files with GPU-optimized settings.

#### With cudf (>= 80 GB GPU memory)

The existing `rewrite_parquet.py` script handles this:

```bash
cd test/tpch_performance
pixi run python rewrite_parquet.py <source_dir> <dest_dir> [row_group_rows]
```

Default row group size: 10,000,000 rows. Settings applied:
- Snappy compression
- Parquet V2 page headers
- 8 MiB max page size
- Dictionary encoding enabled
- ROWGROUP-level statistics

Example:
```bash
cd test/tpch_performance
pixi run python rewrite_parquet.py \
    ../../test_datasets/tpch_parquet_sf100 \
    ../../test_datasets/tpch_parquet_sf100_optimized \
    10000000
```

#### With pyarrow (< 80 GB GPU memory)

When cudf is not available (insufficient GPU memory), write a Python script that uses pyarrow only. The script should:

1. Read all parquet files for each table using `pyarrow.parquet.ParquetFile` or `pyarrow.parquet.read_table`
2. For large tables, read in batches using row group iteration to avoid OOM
3. Write consolidated output using `pyarrow.parquet.ParquetWriter` with these settings:
   ```python
   pq.ParquetWriter(
       dest_path,
       schema,
       compression="snappy",
       version="2.6",
       data_page_version="2.0",
       write_statistics=True,
       use_dictionary=True,
       data_page_size=8 * 1024 * 1024,  # 8 MiB
   )
   ```
4. Use `row_group_size=<target_rows>` when writing (default 10M)
5. Apply int32 downcasts for key columns (same as rewrite_parquet.py):
   - customer: c_custkey, c_nationkey
   - lineitem: l_partkey, l_suppkey, l_linenumber
   - nation: n_nationkey, n_regionkey
   - orders: o_custkey
   - part: p_partkey
   - partsupp: ps_partkey, ps_suppkey
   - region: r_regionkey
   - supplier: s_suppkey, s_nationkey

For the pyarrow path, process large tables (> 50M rows) in batches:
```python
pf = pq.ParquetFile(source_path)
writer = pq.ParquetWriter(dest_path, schema, ...)
for batch in pf.iter_batches(batch_size=row_group_size):
    table = pa.Table.from_batches([batch])
    writer.write_table(table)
writer.close()
```

### Workflow C: Inspect Dataset

Show the user what's in a dataset directory:

```bash
# List files and sizes
ls -lhS <dataset_dir>/*.parquet 2>/dev/null
ls -lhS <dataset_dir>/**/*.parquet 2>/dev/null
```

Then use Python to inspect parquet metadata:

```python
import pyarrow.parquet as pq

for f in parquet_files:
    meta = pq.read_metadata(f)
    print(f"{f}: {meta.num_rows:,} rows, {meta.num_row_groups} row groups, {meta.num_columns} cols")
    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        print(f"  RG {i}: {rg.num_rows:,} rows")
```

Report: table name, total rows, number of files, number of row groups, row group sizes, file sizes on disk, compression.

### Workflow D: Merge Specific Tables

Sometimes the user wants to merge only specific tables (e.g., just lineitem). Follow the same consolidation logic from Workflow B but for the requested tables only.

### Workflow E: Split Large Files

If a user wants to split a large parquet file into multiple smaller ones:

```python
pf = pq.ParquetFile(source_path)
file_idx = 0
writer = None
rows_in_current = 0
target_rows_per_file = <user_specified>

for batch in pf.iter_batches(batch_size=1_000_000):
    if writer is None or rows_in_current >= target_rows_per_file:
        if writer:
            writer.close()
        file_idx += 1
        writer = pq.ParquetWriter(f"{dest_dir}/{table}_{file_idx:04d}.parquet", schema, ...)
        rows_in_current = 0
    writer.write_table(pa.Table.from_batches([batch]))
    rows_in_current += len(batch)

if writer:
    writer.close()
```

## Key Considerations

- **Memory safety**: For large datasets (SF100+), always process in batches. Never load an entire large table into memory at once.
- **Schema preservation**: Preserve original date32 and decimal types. cudf internally promotes date32 to timestamp; cast back before writing.
- **Int32 downcasts**: Apply the INT32_COLUMNS mapping from rewrite_parquet.py to reduce memory footprint.
- **File discovery**: Handle three parquet layouts:
  1. Single file: `<dir>/<table>.parquet`
  2. Partitioned by suffix: `<dir>/<table>_*.parquet`
  3. Subdirectory (tpchgen-rs): `<dir>/<table>/<table>.*.parquet`
- **Row group sizing**: Default 10M rows. For small tables (nation, region), use the full table as one row group. For large tables (lineitem at SF100 = ~600M rows), 10M row groups give ~60 row groups.
- **Recommended sizes**: 2M-10M rows per row group for GPU workloads.

## Before Running

- **Ask the user** for source/destination paths if not clear from context.
- Confirm the backend choice (cudf vs pyarrow) with the user.
- For cudf operations, use the pixi environment: `cd test/tpch_performance && pixi run python ...`
- For pyarrow-only operations, pyarrow should be available in the pixi env or system Python.

## Output

Always report:
- Backend used (cudf or pyarrow) and why
- Per-table: rows processed, source size, destination size, compression ratio
- Total time elapsed
- Output directory location
