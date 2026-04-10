# Sirius-Doris GPU BE: Testing Findings

## Date: 2026-02-26

## Overview

End-to-end testing of the Sirius-Doris GPU Backend with Apache Doris FE,
executing TPC-H queries via `local()` TVF over Parquet files.

## Setup

- **Docker**: Rootless Docker with CDI GPU passthrough
- **FE**: Apache Doris 4.0.3 (doris-fe container)
- **BE**: sirius-doris-be (Rust) with DuckDB + Sirius GPU extension
- **Data**: TPC-H SF1 (partitioned 16 files), SF10 (single file), SF100 (single file)
- **Execution mode**: `--force-cpu` (all queries via DuckDB `from_substrait`, no GPU pipeline)

## GPU Pipeline Status

### SiriusContext Initialization (FIXED)
- **Problem**: Without a config file, `SiriusContext` was not created (NULL pointer).
- **Root cause**: `read_config_file_if_exists()` returned early when no config file existed.
- **Fix**: Sirius now uses YAML configuration (`sirius.yaml`) and calls `apply_defaults()`
  when no config file is found. Set `SIRIUS_DISABLE=1` to skip initialization entirely.
  Modified `read_config_file_if_exists()` to always create `SiriusContext` with defaults.

### QueryBegin / ClientContext (FIXED)
- **Problem**: SIGSEGV in `task_creator::prepare_for_query()` — `_client_context` raw pointer
  was uninitialized because `QueryBegin` callback hadn't fired.
- **Root cause**: `SiriusContext` was registered on the outer connection during the
  `GPUExecutionSubstraitBind` phase, AFTER DuckDB's `QueryBegin` had already fired for
  that query.
- **Fix**: Modified `SiriusContext::create_query()` to accept `ClientContext&` and explicitly
  call `task_creator_->reset()` + `set_client_context()` before `prepare_for_query()`.

### GPU Physical Plan Generation
- **Status**: PARTIALLY WORKING
- `SiriusGeneratePhysicalPlan` fails with "Attempted to access index -2 within vector of
  size 16" for plans with PARQUET_SCAN. Works for some plan shapes but not all.
- The -2 index is likely a `COLUMN_IDENTIFIER_ROW_ID` sentinel used by DuckDB that the
  Sirius plan generator doesn't handle.

### GPU Execution (CUDA Errors)
- **Status**: NOT WORKING in Docker CDI environment
- When GPU plan generation succeeds, the pipeline starts reading Parquet data and
  allocating GPU buffers. H2D copy via `cudaMemcpyBatchAsync` (CUDA 13.0+) fails with
  `cudaErrorInvalidValue`.
- This causes SIGSEGV in background GPU pipeline threads, crashing the process.
- **Workaround**: `--force-cpu` flag routes all queries through DuckDB's `from_substrait`
  (CPU execution), completely bypassing the GPU pipeline.

### DuckDB Fallback (FIXED)
- **Problem**: When GPU plan failed, the C++ fallback executed a SQL comment
  (`"-- substrait plan (no SQL fallback available)"`) producing empty results.
- **Fix**: Store the Substrait blob in `SiriusTableFunctionData` and use
  `from_substrait(blob)` as the fallback path. For runtime GPU errors, propagate the
  error to Rust for safe fallback on a separate connection.

### View/Table Name Collisions (FIXED)
- **Problem**: `CREATE OR REPLACE TABLE` fails when an object with the same name exists
  as a VIEW (or vice versa) from a previous query.
- **Fix**: Added `DROP VIEW IF EXISTS` before `register_file_table()` and
  `DROP TABLE IF EXISTS` before `register_parquet_view()`.

## Query Results

### TPC-H SF1 (16 partitioned parquet files, 224MB total)

| Query | Status | Notes |
|-------|--------|-------|
| count(*) lineitem | PASS | 6,001,215 rows |
| count(*) orders | PASS | 1,500,000 rows |
| count(*) customer | PASS | 150,000 rows |
| count(*) part | PASS | 200,000 rows |
| count(*) supplier | PASS | 10,000 rows |
| Q1 (Pricing Summary) | PASS | All 4 groups correct |
| Q3 (Shipping Priority) | DATA OK, COLUMNS WRONG | Column names mismatched (known Substrait Root.names ordering bug) |
| Q4 (Order Priority) | PASS | EXISTS subquery works |
| Q5 (Local Supplier) | PASS | 3-way join (lineitem x supplier x nation) |
| Q6 (Revenue Forecast) | PASS | 123,141,078.23 |
| Q10 (Returned Items) | DATA OK, COLUMNS WRONG | 4-way join works but column ordering bug |
| Q12 (Shipping Modes) | PASS | 2-way join with CASE WHEN |
| Q13 (Customer Dist.) | DATA OK, SORT WRONG | LEFT OUTER JOIN works, ORDER BY not preserved |
| Q14 (Promotion Effect) | PASS | JOIN + CASE WHEN + aggregation |
| 8-way UNION ALL | FAIL | DuckDB from_substrait doesn't support SetRel with >2 inputs |
| GROUP BY + ORDER BY | PASS | Basic group by with sort |
| Q7 (Volume Shipping) | HANG | Self-join on nation table → from_substrait hangs |
| Q9 (Product Type Profit) | PASS | 6-way join works, ORDER BY not preserved |
| Q11 (Important Stock) | HANG | HAVING subquery with 3-way join → from_substrait hangs |
| Q16 (Parts/Supplier) | DATA WRONG | count(DISTINCT) returns 1 for all rows |
| Q18 (Large Volume Cust.) | FAIL | Column ordering bug: c_name not found in sort_limit |
| Q19 (Discounted Revenue) | PASS | 2-way join with complex OR conditions |
| Q22 (Global Sales) | HANG | Customer self-join + NOT EXISTS → from_substrait hangs |

### TPC-H SF10 (single parquet files, ~2.4GB lineitem)

| Query | Status | Notes |
|-------|--------|-------|
| count(*) lineitem | PASS | 59,986,052 rows |
| count(*) orders | PASS | 15,000,000 rows |
| Q1 (Pricing Summary) | PASS | All 4 groups correct, LocalFiles path |
| Q4 (Order Priority) | PASS | EXISTS subquery (values ≈ 10x of SF1) |
| Q6 (Revenue Forecast) | PASS | 1,230,113,636.01 |
| Q12 (Shipping Modes) | PASS | 2-way join at SF10, LocalFiles path |

### TPC-H SF1 (16 partitioned files, LocalFiles path — no table materialization)

| Query | Status | Notes |
|-------|--------|-------|
| count(*) lineitem | PASS | 6,001,215 rows, 16 files via LocalFiles |
| Q1 (Pricing Summary) | PASS | All 4 groups correct |
| Q4 (Order Priority) | PASS | EXISTS subquery, 2 tables × 16 partitions |
| Q5 (3-way join) | PASS | lineitem × supplier (partitioned) × nation (single) |
| Q6 (Revenue Forecast) | PASS | 123,141,078.23 |
| Q12 (Shipping Modes) | PASS | 2-way join, both tables partitioned 16 files |
| Q14 (Promotion Effect) | PASS | JOIN + CASE + AGG, mixed partitioned/single |
| Q2 (Min Cost Supplier) | PASS | 5-way join + correlated subquery, mixed partitioned/single |
| Q3 (Shipping Priority) | DATA OK, COLUMNS WRONG | Same column ordering bug as single-file |
| Q13 (Customer Dist.) | DATA OK, SORT WRONG | LEFT OUTER JOIN, ORDER BY not preserved |

### TPC-H SF100 (single parquet files, ~26GB lineitem, LocalFiles path)

| Query | Status | Notes |
|-------|--------|-------|
| count(*) lineitem | PASS | 600,037,902 rows from 26GB |
| count(*) part | PASS | 20,000,000 rows from 658MB |
| count(*) supplier | PASS | 1,000,000 rows from 86MB |
| Q1 (Pricing Summary) | PASS | All 4 groups correct (sum_qty = 3.775B) |
| Q6 (Revenue Forecast) | PASS | 12,330,426,888.46 |
| Q4 (Order Priority) | PASS | EXISTS subquery on 33GB |
| Q12 (Shipping Modes) | PASS | 33GB join (26GB lineitem + 7GB orders) |
| Q2 (Min Cost Supplier) | PASS | 5-way join + correlated subquery on ~34GB |

### TPC-H SF1000 (single parquet files, ~247GB lineitem, LocalFiles path)

| Query | Status | Notes |
|-------|--------|-------|
| count(*) lineitem | PASS | 5,999,989,709 rows from 247GB |
| count(*) supplier | PASS | 10,000,000 rows |
| Q1 (Pricing Summary) | PASS | All 4 groups correct (sum_qty = 37.7B) |
| Q6 (Revenue Forecast) | PASS | 123,313,653,126.11 |

**Note**: SF1000 queries require `SET GLOBAL query_timeout = 3600` (FE default is 900s).
Transient gRPC keepalive failures may occur — retrying usually succeeds.

## Known Limitations

### 1. Column Ordering Bug (Substrait Root.names)
Complex joins produce correct data but with column names mismatched. The DuckDB optimizer
reorders join output columns, but Substrait Root.names are applied positionally. This
affects queries with JOIN + GROUP BY + ORDER BY on multiple tables (Q3, Q10).

### 2. ORDER BY Not Preserved (from_substrait)
DuckDB's `from_substrait()` doesn't always preserve sort order from SortRel/FetchRel.
The Rust code has `sort_limit_sql` wrapper for some cases but it doesn't cover all
query shapes. Affects Q13 and other queries with ORDER BY on derived columns.

### 3. Multi-input UNION ALL
DuckDB's Substrait consumer doesn't support SetRel with more than 2 inputs.
Workaround: use separate queries or the SQL path.

### 4. from_substrait Hangs on Certain Patterns
Certain query patterns cause DuckDB's `from_substrait` to hang indefinitely:
- Q7: Self-join on nation table (two references to same .parquet file)
- Q11: HAVING subquery with 3-way join
- Q22: Customer self-join + NOT EXISTS subquery
Common factor: correlated subqueries with self-references. These plans translate
successfully to Substrait but DuckDB's execution never completes. Likely DuckDB's
Substrait consumer creates an O(N^2) or worse plan.

### 5. count(DISTINCT) Aggregate
Q16 returns supplier_cnt=1 for all rows, suggesting count(DISTINCT ps_suppkey) is
not handled correctly through from_substrait. The Substrait invocation=DISTINCT (2)
may not be properly consumed by DuckDB's Substrait extension.

### 5. GPU Pipeline CUDA Errors
`cudaMemcpyBatchAsync` returns `cudaErrorInvalidValue` in Docker CDI environment.
Possibly related to host memory not being pinned, or CUDA runtime version mismatch.
The `--force-cpu` flag is the current workaround.

## Architecture Decisions

### `--force-cpu` Flag
Added to bypass the Sirius GPU pipeline entirely when the GPU runtime is not functional.
Converts `ExecPlan::Substrait` → `SubstraitCpuOnly` and `ExecPlan::Sql` → `SqlCpuOnly`
at the `execute_plan` level. All execution goes through DuckDB's `from_substrait` or
`execute_sql` on CPU.

### Error Propagation Strategy
GPU runtime errors (CUDA failures) are propagated to the Rust caller rather than
attempting fallback on the same DuckDB connection. This prevents SIGSEGV from
background GPU threads interfering with fallback execution. The Rust side has its
own `from_substrait` fallback using a clean connection.

### LocalFiles for Multi-File Parquet (force-cpu)
When `--force-cpu` is active and the format is parquet, the BE skips table/view
materialization and generates Substrait `ReadRel::LocalFiles` with one `FileOrFiles`
entry per file. DuckDB's `from_substrait` converts this to `parquet_scan([file_list])`.

Benefits:
- No `CREATE TABLE AS SELECT *` or `CREATE VIEW AS SELECT *` overhead
- Schema resolved via `DESCRIBE SELECT * FROM read_parquet('first_file')` (fast)
- When GPU is functional: `parquet_scan` → `sirius_physical_parquet_scan` (native GPU reader)
- Works with `shared_storage=true` where FE distributes N partition files per BE

The GPU path still uses VIEW + NamedTable because `sirius_physical_parquet_scan`
crashes on complex queries (only count(*) works).

## Files Modified (This Session)

| File | Change |
|------|--------|
| `sirius_config.hpp` | Added `ensure_default_memory_configs()` |
| `sirius_config.cpp` | Implemented `ensure_default_memory_configs()` |
| `sirius_context.hpp` | `create_query()` now takes `ClientContext&` |
| `sirius_context.cpp` | `create_query()` calls `reset()` + `set_client_context()` |
| `sirius_engine.cpp` | Passes context to `create_query()`, added logging |
| `sirius_extension.cpp` | `from_substrait` fallback, Substrait blob storage, error propagation |
| `config.rs` | Added `--force-cpu` CLI flag |
| `grpc_service.rs` | `force_cpu` threading, empty IPC guard, `execute_plan` force-cpu mode |
| `sirius-ffi/lib.rs` | DROP VIEW/TABLE IF EXISTS before register, `get_parquet_columns()` |
| `docker-compose.yml` | Added `--force-cpu` to BE command |
| `grpc_service.rs` | LocalFiles path for force-cpu parquet (no table materialization) |
| `scan_translator.rs` | `ReadRel::LocalFiles` generation for multi-file parquet |
| `node_translator.rs` | Thread `file_scan_map` parameter through plan translation |
| `plan-translator/lib.rs` | `FileScanInfo`/`FileScanFile` types, `translate_fragment()` signature |
