# `gpu_execution`

`gpu_execution` is the recommended execution path — out-of-core execution with tiered memory management (GPU/host/disk), automatic data partitioning, and spilling. It works with both **Parquet** and **DuckDB-native** storage.

## Building

Follow the [source quick start](README.md#quick-start) to build Sirius and the
DuckDB extension using Pixi.

## Configuration

`gpu_execution` reads an optional YAML config file; without one, built-in defaults apply. See the [Configuration documentation](super-sirius/configuration.md) for the full reference, including [config file resolution order](super-sirius/configuration.md#config-file-resolution), all available options, and [value formats](super-sirius/configuration.md#value-formats). An example config file is provided at [`test/cpp/integration/integration.yaml`](../test/cpp/integration/integration.yaml).

## Running

```bash
export SIRIUS_CONFIG_FILE=/path/to/sirius.yaml  # optional
pixi run sirius-duckdb/build/release/duckdb -unsigned -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';"
```

From the DuckDB shell, create views pointing to your Parquet files and run queries with `gpu_execution`:

```sql
-- Create views for parquet data
CREATE VIEW lineitem AS SELECT * FROM read_parquet('/data/lineitem/*.parquet');
CREATE VIEW orders AS SELECT * FROM read_parquet('/data/orders/*.parquet');
CREATE VIEW customer AS SELECT * FROM read_parquet('/data/customer/*.parquet');

-- Run a query on GPU
CALL gpu_execution('SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price
FROM lineitem
WHERE l_shipdate <= date ''1998-09-02''
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus');
```

`gpu_execution` takes exactly one SQL statement; an empty string or several statements raise an
error. Named parameters:

- `enable_optimizer` (default `true`; `NULL` means the default): plan the query with DuckDB's
  optimizer before Sirius takes it. `false` hands Sirius the unoptimized plan, for example to
  test operators the optimizer would rewrite away. The session's own optimizer setting is left
  unchanged.
- `query_label`: names this query in telemetry, as `sirius_set_query_label` does.

### Transparent Execution

When `gpu_execution` is enabled (the default after loading the extension), all DuckDB queries are automatically intercepted by the optimizer hook and run on GPU — no `CALL gpu_execution('...')` wrapper needed:

```sql
-- Plain SQL, runs on GPU automatically
SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price
FROM lineitem
WHERE l_shipdate <= date '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
```

Queries with unsupported operators fall back silently to DuckDB CPU execution, unless `enable_duckdb_fallback` is `false`. `gpu_execution` is a session-scoped setting: an unqualified `SET` changes it for the current connection only. To change it for every connection of the database instance, including connections that other extensions open for their own SQL, use `SET GLOBAL`; a connection with its own `SET SESSION` value keeps that value. `RESET gpu_execution` removes the connection's own value, so the connection follows the global value again; `RESET GLOBAL gpu_execution` removes the global value, so connections without their own value return to the default, `true`.

```sql
SET gpu_execution = false;          -- this connection only
SET GLOBAL gpu_execution = false;   -- every connection of this database instance without a session value
```

Sirius does not intercept the SQL that the DuckLake extension runs on the connections it opens for its own catalog. DuckLake marks the metadata connections it creates through `DuckLakeTransaction::GetConnection()` in DuckDB's per-connection state under the key `ducklake_internal_connection`; Sirius reads the marker at the connection's first statement and leaves the connection alone for its lifetime. Connections without the marker (every connection of a DuckLake build older than the marker, such as the qualified `d8a1881e`) are recognised by the combination they set before their first statement, a hidden attached database as the default catalog and `catalog_error_max_schemas` at 0. A statement that reads a table of a hidden attached database is not intercepted either. Both hold whatever `gpu_execution` says; other extensions that open connections for themselves are intercepted like user connections unless they carry the same marker or fingerprint.

To re-enable:

```sql
SET gpu_execution = true;
```

Sirius plans a correlated subquery only in the form DuckDB's default `delim_join_as_cte = true` gives it, a CTE. With `false`, which DuckDB has deprecated, a correlated subquery that DuckDB cannot flatten stays a delim join, and Sirius has no operators for one. A query whose plan has a delim join is left to the CPU like any other plan Sirius cannot run, on the transparent path and in `gpu_execution(...)`: Sirius logs a warning that names the setting, and with `enable_duckdb_fallback = false` the query fails with that message. Every other query runs on the GPU under either value.

**How it works:** Sirius registers two optimizer hooks at extension load time, and disables the DuckDB optimizers it cannot plan after (such as `IN_CLAUSE`, `COMPRESSED_MATERIALIZATION`, and `LATE_MATERIALIZATION`). The pre-optimizer hook decides whether the statement stays on DuckDB and, for a query headed to the GPU, rewrites correlated subqueries before DuckDB orders the joins (see [Optimizations](super-sirius/optimizations.md#subquery-keys-carried-across-an-equality)). The post-optimizer hook captures the optimized logical plan, and GPU plan generation runs on it via `sirius_physical_plan_generator`. If plan generation succeeds, a `PhysicalSiriusExecution` node replaces the DuckDB physical plan and the query runs on GPU; if it throws, DuckDB's CPU plan runs, or the error is raised when `enable_duckdb_fallback` is `false`. When the pre-optimizer hook had rewritten the query, DuckDB first plans the statement again without the rewrite, so the CPU runs the plan DuckDB makes alone.

## Generating Test Datasets

For TPC-H benchmarking, use the provided data generation script:

```bash
cd test/tpch_performance
pixi run bash generate_tpch_data.sh 100   # generates SF100 parquet data
```

This produces partitioned Parquet files under `test_datasets/tpch_parquet_sf100/`. Then create views from the DuckDB shell:

```sql
CREATE VIEW lineitem AS SELECT * FROM read_parquet('test_datasets/tpch_parquet_sf100/lineitem/*.parquet');
-- repeat for other tables...
```

For your own data, point `read_parquet()` at any Parquet file or glob:

```sql
CREATE VIEW my_table AS SELECT * FROM read_parquet('/path/to/my_data/*.parquet');
```

## Testing

`gpu_execution` uses C++ unit tests built with [Catch2](https://github.com/catchorg/Catch2). Test files are in `test/cpp/`.

Run all unit tests:
```
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/test/cpp/sirius_unittest
```

Run tests associated with a specific tag or a specific test:
```
build/release/test/cpp/sirius_unittest "[path_utils]"
build/release/test/cpp/sirius_unittest "strip_file_scheme folds dot, dot-dot and empty segments"
```

Test logs are saved in:
```
build/release/test/cpp/log
```

## Developer Documentation

For in-depth documentation on the `gpu_execution` engine, see the [Super Sirius Documentation](super-sirius/README.md).
