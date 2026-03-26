# `gpu_execution`

`gpu_execution` is the recommended execution path — out-of-core execution with tiered memory management (GPU/host/disk), automatic data partitioning, and spilling. It currently works with **Parquet** data format.

## Configuration

`gpu_execution` requires a config file in [libconfig++](https://hyperreckoning.com/libconfig/) format. Point to it via the `SIRIUS_CONFIG_FILE` environment variable, or place it at the default path `~/.sirius/sirius.cfg`.

An example config file is provided at [`test/cpp/integration/integration.cfg`](../test/cpp/integration/integration.cfg). See the [Configuration documentation](super-sirius/configuration.md) for a full reference of all options.

## Running

```bash
export SIRIUS_CONFIG_FILE=/path/to/sirius.cfg
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
