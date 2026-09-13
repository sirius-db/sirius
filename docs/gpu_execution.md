# `gpu_execution`

`gpu_execution` is the recommended execution path — out-of-core execution with tiered memory management (GPU/host/disk), automatic data partitioning, and spilling. It currently works with **Parquet** data format.

## Building

Clone the Sirius repository:
```
git clone --recurse-submodules https://github.com/sirius-db/sirius.git
cd sirius
```

Set up the environment with [Pixi](https://pixi.sh/) and build:
```
pixi shell
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
```

Note that if building consumes too much memory, try reducing the `CMAKE_BUILD_PARALLEL_LEVEL` value.

## Configuration

`gpu_execution` requires a config file in YAML format. See the [Configuration documentation](super-sirius/configuration.md) for the full reference, including [config file resolution order](super-sirius/configuration.md#config-file-resolution), all available options, and [byte suffixes](super-sirius/configuration.md#byte-suffixes). An example config file is provided at [`test/cpp/integration/integration.yaml`](../test/cpp/integration/integration.yaml).

## Running

```bash
export SIRIUS_CONFIG_FILE=/path/to/sirius.yaml
./build/release/duckdb
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

Queries with unsupported operators fall back silently to DuckDB CPU execution. `gpu_execution` is a session-scoped setting: an unqualified `SET` changes it for the current connection only. To change it for every connection of the database instance, including connections that other extensions open for their own SQL, use `SET GLOBAL`; a connection with its own `SET SESSION` value keeps that value.

```sql
SET gpu_execution = false;          -- this connection only
SET GLOBAL gpu_execution = false;   -- every connection of this database instance without a session value
```

Sirius does not intercept the SQL that the DuckLake extension runs on the connections it opens for its own catalog: such a connection has a hidden attached database as its default catalog and `catalog_error_max_schemas` set to 0 before its first statement, and Sirius recognises that combination (qualified against DuckLake `d8a1881e`) and leaves the connection alone for its lifetime. A statement that reads a table of a hidden attached database is not intercepted either. Both hold whatever `gpu_execution` says; other extensions that open connections for themselves are intercepted like user connections unless they carry the same fingerprint.

To re-enable:

```sql
SET gpu_execution = true;
```

**How it works:** Two optimizer extensions are registered at extension load time. A pre-optimizer hook disables DuckDB optimizers incompatible with Sirius (such as `IN_CLAUSE`, `COMPRESSED_MATERIALIZATION`, and `LATE_MATERIALIZATION`). A post-optimizer hook captures the optimized logical plan and attempts GPU plan generation via `sirius_physical_plan_generator`. If plan generation succeeds, a `PhysicalSiriusExecution` node replaces the DuckDB physical plan and the query runs on GPU; if plan generation throws, the original DuckDB CPU plan runs unchanged.

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
build/release/extension/sirius/test/cpp/sirius_unittest
```

Run tests associated with a specific tag or a specific test:
```
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"
```

Test logs are saved in:
```
build/release/extension/sirius/test/cpp/log
```

## Developer Documentation

For in-depth documentation on the `gpu_execution` engine, see the [Super Sirius Documentation](super-sirius/README.md).
