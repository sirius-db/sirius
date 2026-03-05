---
name: dataset-manager
description: Manage TPC-H parquet datasets — generate data at any scale factor, consolidate small parquet files into fewer larger files, inspect dataset layout, and optimize row group sizes. Uses rewrite_parquet.py which auto-selects cudf (GPU) or pyarrow (CPU) with OOM fallback.
---

# TPC-H Dataset Manager

You are managing TPC-H parquet datasets for Sirius, a GPU-accelerated SQL query engine. Your job is to generate, inspect, consolidate, and optimize parquet files used for benchmarking and testing.

## Backend Selection

The `rewrite_parquet.py` script handles backend selection automatically:
- If **cudf** is installed, it uses GPU-accelerated reads (much faster for large datasets)
- If cudf is **not installed** or hits a **GPU out-of-memory** error, it falls back to **pyarrow** (CPU-only)
- The fallback is per-table and per-batch — if a large table OOMs on the GPU mid-way, remaining batches continue with pyarrow

No manual GPU memory check is needed. Just run the script.

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

This is the most common operation. Takes parquet files and rewrites them with optimized row groups, compression, and file size limits. Automatically uses cudf if available, falls back to pyarrow on import failure or GPU OOM.

```bash
cd test/tpch_performance
pixi run python rewrite_parquet.py <source_dir> <dest_dir> [row_group_rows] [max_file_gb]
```

Parameters:
- `source_dir` — Directory containing source parquet files
- `dest_dir` — Output directory for rewritten files
- `row_group_rows` — Rows per row group (default: 10,000,000)
- `max_file_gb` — Maximum output file size in GB (default: 20). Tables exceeding this are split into numbered files (e.g., `lineitem_0000.parquet`, `lineitem_0001.parquet`)

Settings applied:
- Snappy compression
- Parquet V2 page headers
- 8 MiB max page size
- Dictionary encoding enabled
- ROWGROUP-level statistics
- Int32 downcasts for key columns to reduce memory footprint

Examples:
```bash
cd test/tpch_performance

# Default: 10M-row row groups, 20 GB max file size
pixi run python rewrite_parquet.py \
    ../../test_datasets/tpch_parquet_sf100 \
    ../../test_datasets/tpch_parquet_sf100_optimized

# Custom: 2M-row row groups, 10 GB max file size
pixi run python rewrite_parquet.py \
    ../../test_datasets/tpch_parquet_sf100 \
    ../../test_datasets/tpch_parquet_sf100_rg2m \
    2000000 10
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

Sometimes the user wants to merge only specific tables (e.g., just lineitem). Edit the `TPCH_TABLES` list in `rewrite_parquet.py` or call `rewrite_table()` directly for the requested tables.

## Key Considerations

- **Memory safety**: For large datasets (SF100+), the script processes in batches automatically. If cudf OOMs, it falls back to pyarrow for remaining batches.
- **Schema preservation**: Preserve original date32 and decimal types. cudf internally promotes date32 to timestamp; the script casts back before writing.
- **File splitting**: Output files exceeding `max_file_gb` (default 20 GB) are automatically split into numbered files. Small tables that fit under the limit stay as single files (e.g., `customer.parquet`).
- **File discovery**: The script handles three parquet layouts:
  1. Single file: `<dir>/<table>.parquet`
  2. Partitioned by suffix: `<dir>/<table>_*.parquet`
  3. Subdirectory (tpchgen-rs): `<dir>/<table>/<table>.*.parquet`
- **Row group sizing**: Default 10M rows. For small tables (nation, region), the full table is one row group. For large tables (lineitem at SF100 = ~600M rows), 10M row groups give ~60 row groups.
- **Recommended sizes**: 2M-10M rows per row group for GPU workloads.

## Before Running

- **Ask the user** for source/destination paths if not clear from context.
- Use the pixi environment: `cd test/tpch_performance && pixi run python ...`

## Output

Always report:
- Backend used (cudf or pyarrow) — the script prints this automatically
- Per-table: rows processed, source size, destination size, compression ratio, number of output files
- Total time elapsed
- Output directory location
