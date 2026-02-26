# TPC-H Performance Testing

This directory contains benchmarking and performance testing tools for comparing DuckDB (CPU) vs Sirius (GPU) on TPC-H queries at various scale factors.

## Prerequisites

- Sirius must be built: `pixi run make -j12` (from project root)
- Binary: `build/release/duckdb` with Sirius extension at `build/release/extension/sirius/sirius.duckdb_extension`
- Sirius config: `test/cpp/integration/integration.cfg` (set `SIRIUS_CONFIG_FILE` env var)
- Parquet data must exist in `test_datasets/tpch_parquet_sf<N>/`

## Generating Test Data

### From DuckDB's built-in TPC-H generator (simplest)

```bash
# From project root - generates parquet files with DuckDB's default row groups (122K rows)
./build/release/duckdb -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=100); EXPORT DATABASE 'test_datasets/tpch_parquet_sf100' (FORMAT PARQUET);"
```

### Rewriting parquet with GPU-optimized settings

The `rewrite_parquet.py` script reads existing parquet files via cudf and rewrites them with larger row groups, snappy compression, and V2 page headers. Requires the pixi environment in this directory (`pixi install`).

```bash
cd test/tpch_performance

# Rewrite with 10M-row row groups (recommended for GPU workloads)
pixi run python rewrite_parquet.py ../../test_datasets/tpch_parquet_sf100 ../../test_datasets/tpch_parquet_sf100_optimized 10000000

# Rewrite with 2M-row row groups
pixi run python rewrite_parquet.py ../../test_datasets/tpch_parquet_sf100 ../../test_datasets/tpch_parquet_sf100_rg2m 2000000
```

The rewriter reads with cudf (GPU), casts back to the original parquet schema (preserving date32 and decimal types), and writes single-file output via pyarrow ParquetWriter. Large tables are processed in batches by row group to stay within GPU memory.

### From tpchgen-rs (alternative, supports partitioned output)

```bash
cd test/tpch_performance
pixi run python generate_test_data_tpchgen-rs.py <SF> <partitions> <format>
```

## Running Benchmarks

All commands run from the **project root** directory.

### Full DuckDB vs Sirius benchmark (cold + warm per query)

Runs each TPC-H query (1-20, 22, skipping 21) twice in the same DuckDB session for both DuckDB and Sirius. Uses DuckDB's `.timer on` for timing. Produces a comparison table with speedup ratios.

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg

# Against original parquet (122K row groups)
bash test/tpch_performance/benchmark_tpch.sh 100

# Against optimized parquet (10M row groups)
bash test/tpch_performance/benchmark_tpch.sh 100_optimized

# Against 2M row group parquet
bash test/tpch_performance/benchmark_tpch.sh 100_rg2m
```

The scale factor argument maps to the directory name: `test_datasets/tpch_parquet_sf<arg>/`. Results are saved to `benchmark_results_sf<arg>/` as CSV files.

### Sirius-only individual queries

Run a single query cold+warm to quickly test:

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg

# Create a temp SQL with views + query (run twice for cold+warm)
PARQUET_DIR="test_datasets/tpch_parquet_sf100_optimized"
# ... create views, .timer on, query, query in temp file ...
./build/release/duckdb -f /tmp/test.sql
```

### DuckDB-only baseline

```bash
bash test/tpch_performance/run_tpch_parquet_duckdb.sh <scale_factor> <query_numbers...>
# Example: bash test/tpch_performance/run_tpch_parquet_duckdb.sh 100 1 3 6 9
```

### Sirius-only run

```bash
export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg
bash test/tpch_performance/run_tpch_parquet.sh <scale_factor> <query_numbers...>
```

Both scripts accept `TIMING_CSV` env var to write per-query timing to a CSV file.

### Thread configuration sweep

Runs Sirius-only across multiple thread configurations (pipeline, scan, task_creator threads) to find optimal settings. Modifies `integration.cfg` during the run and restores baseline when done.

```bash
bash test/tpch_performance/sweep_threads.sh
```

Results are saved to `benchmark_results_thread_sweep/` as CSV files per configuration.

### Python-based performance test (in-memory database)

Loads data into a DuckDB database, runs all 22 queries with both CPU and GPU, verifies results match:

```bash
pixi run python test/tpch_performance/performance_test.py <scale_factor>
```

## Query Files

- `tpch_queries/orig/q*.sql` - Plain SQL queries
- `tpch_queries/gpu/q*.sql` - Queries wrapped in `call gpu_execution('...');` for Sirius

## Key Files

| File | Purpose |
|------|---------|
| `benchmark_tpch.sh` | Full DuckDB vs Sirius benchmark with cold/warm timing |
| `sweep_threads.sh` | Thread configuration sweep (Sirius-only) |
| `run_tpch_parquet.sh` | Run Sirius queries on parquet files |
| `run_tpch_parquet_duckdb.sh` | Run DuckDB queries on parquet files |
| `rewrite_parquet.py` | Rewrite parquet with GPU-optimized row groups |
| `performance_test.py` | Python-based benchmark with result verification |
| `queries.py` | TPC-H query definitions (base SQL) |
| `generate_test_data.py` | Generate test data via dbgen |
| `generate_test_data_tpchgen-rs.py` | Generate test data via tpchgen-rs + query files |
| `pixi.toml` | Python environment with cudf for parquet rewriting |

## Sirius Configuration

The Sirius config file (`test/cpp/integration/integration.cfg`) controls:
- **GPU memory**: `usage_limit_fraction`, `reservation_limit_fraction`
- **Host memory**: `capacity_bytes`, `initial_number_pools`, `pool_size`, `block_size`
  - Initial allocation = `initial_number_pools * pool_size * block_size`
- **Thread pools**: `pipeline`, `duckdb_scan`, `task_creator`, `downgrade` thread counts
- **Scan cache**: `duckdb_scan.cache = true` enables caching

## Parquet Format Notes

- DuckDB's default export creates 122,880-row row groups (its internal vector size)
- For GPU workloads, 2M-10M row groups perform significantly better
- The `rewrite_parquet.py` script preserves the original schema (date32, decimal128) to avoid type mismatch issues with Sirius
- cudf internally promotes date32 to timestamp; the rewriter casts back before writing
