---
name: dataset-manager
description: >
  Use this skill to generate benchmark datasets (TPC-H, TPC-DS, etc.). Trigger when the user
  needs test data at a specific scale factor for benchmarking or testing. Supports parquet and
  duckdb output formats.
argument-hint: "[benchmark] [scale_factor] [--format duckdb|parquet] [--cluster] [--output <path>]"
---

# Dataset Generator

Generate benchmark datasets by running the corresponding shell script under `test/<benchmark>_performance/`.

## Gather Parameters

Parse `$ARGUMENTS` for:
- **Benchmark**: name from the registry below (first positional arg)
- **Scale factor**: integer (second positional arg)
- **Format**: `--format duckdb|parquet` (optional — each benchmark has a default)
- **Cluster** (TPC-H duckdb only): `--cluster` / `--cluster-keys <spec>` (optional — physically sorts tables so row-group pruning works; errors with `--format parquet`)
- **Output path**: `--output <path>` (optional — scripts have sensible defaults)

If any required parameter (benchmark, scale factor) is missing, ask the user.

## Benchmark Registry

Each entry follows the same structure: script location, command template, supported formats, defaults, and prerequisites.

<!-- To add a new benchmark: copy an existing entry, fill in the fields, and update the description frontmatter. -->

### TPC-H

| Field | Value |
|-------|-------|
| Script | `test/tpch_performance/generate_tpch_data.sh` |
| Default format | `parquet` |
| Formats | `parquet` (tpchgen-rs), `duckdb` (DuckDB `dbgen()`) |
| Default output (parquet) | `test_datasets/tpch_parquet_sf<SF>` |
| Default output (duckdb) | `test_datasets/tpch_sf<SF>.duckdb` |
| Default output (duckdb + `--cluster`) | `test_datasets/tpch_sf<SF>_sorted.duckdb` |
| Prerequisites | Parquet: pixi env (rust, python, pyarrow). DuckDB: `build/release/duckdb` |

```bash
cd test/tpch_performance && pixi run bash generate_tpch_data.sh <SF> --format <FORMAT> [--cluster] [--cluster-keys <spec>] [--output <path>]
```

Notes:
- If the parquet output directory already exists, the script skips generation
- `--cluster` (duckdb only) physically sorts tables at load so each row group's per-column min/max becomes selective — this is what makes Sirius's native-scan row-group pruning skip work on date-filtered queries. It writes a distinct `tpch_sf<SF>_sorted.duckdb` so it can coexist with the unsorted dataset.
- Default cluster keys: `lineitem:l_shipdate,orders:o_orderdate`. Override with `--cluster-keys "table:col,table:col,..."` (implies `--cluster`).
- `--cluster` errors if combined with `--format parquet` (tpchgen-rs writes decimals as FIXED_LEN_BYTE_ARRAY, which disables Sirius row-group pruning for the whole file).

### TPC-DS

| Field | Value |
|-------|-------|
| Script | `test/tpcds_performance/generate_tpcds_data.sh` |
| Default format | `duckdb` |
| Formats | `duckdb`, `parquet` |
| Default output (duckdb) | `test_datasets/tpcds_sf<SF>.duckdb` |
| Default output (parquet) | `test_datasets/tpcds_parquet_sf<SF>` |
| Prerequisites | `build/release/duckdb` |

```bash
cd test/tpcds_performance && bash generate_tpcds_data.sh <SF> --format <FORMAT> [--output <path>]
```

Notes:
- Also extracts TPC-DS query files to `test/tpcds_performance/queries/q{1..99}.sql`

## Prerequisites

For any benchmark that requires the DuckDB binary, check before running:
```bash
test -x build/release/duckdb
```
If missing, tell the user to build first: `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make`

## Report Results

- Benchmark name
- Output path
- Format used
- Whether generation was skipped (output already existed) or completed
