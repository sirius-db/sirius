# `gpu_processing`

`gpu_processing` is the in-memory execution path. It works with DuckDB's native storage format and requires the dataset to fit in GPU memory.

## Building

Clone the Sirius repository:
```
git clone --recurse-submodules https://github.com/sirius-db/sirius.git
cd sirius
```

By default, only the `gpu_execution` code path is compiled. To build `gpu_processing`, enable the `ENABLE_LEGACY_SIRIUS` CMake option:
```bash
pixi shell
cmake -S duckdb --preset release -DENABLE_LEGACY_SIRIUS=ON
cmake --build build/release
```

## Running

Start the DuckDB shell with `./build/release/duckdb {DATABASE_NAME}.duckdb`.
From the DuckDB shell, initialize the Sirius buffer manager with `call gpu_buffer_init`. This API accepts 2 parameters, the GPU caching region size and the GPU processing region size. The GPU caching region is a memory region where the raw data is stored in GPUs, whereas the GPU processing region is where intermediate results are stored in GPUs (hash tables, join results .etc).
For example, to set the caching region as 1 GB and the processing region as 2 GB, we can run the following command:
```
call gpu_buffer_init("1 GB", "2 GB");
```

By default, Sirius also allocates pinned memory based on the above two arguments. To explicility specify the amount of pinned memory to allocate during initialization run:
```
call gpu_buffer_init("1 GB", "2 GB", pinned_memory_size = "4 GB");
```

After setting up Sirius, we can execute SQL queries using the `call gpu_processing`:
```
call gpu_processing("select
  l.l_orderkey,
  sum(l.l_extendedprice * (1 - l.l_discount)) as revenue,
  o.o_orderdate,
  o.o_shippriority
from
  customer c,
  orders o,
  lineitem l
where
  c.c_mktsegment = 'HOUSEHOLD'
  and c.c_custkey = o.o_custkey
  and l.l_orderkey = o.o_orderkey
  and o.o_orderdate < date '1995-03-25'
  and l.l_shipdate > date '1995-03-25'
group by
  l.l_orderkey,
  o.o_orderdate,
  o.o_shippriority
order by
  revenue desc,
  o.o_orderdate
limit 10;");
```
**The cold run in Sirius would be significantly slower due to data loading from storage and conversion from DuckDB format to Sirius native format. Subsequent runs would be faster since it benefits from caching on GPU memory.**

All 22 TPC-H queries are saved in tpch-queries.sql. To run all queries:
```
.read tpch-queries.sql
```

## Generating and Loading Test Datasets

A setup script generates both TPC-H (SF1) and ClickBench datasets in one step:
```bash
./setup_test_datasets.sh
```

This creates the necessary data files under `test_datasets/`. The script is idempotent — it skips datasets that already exist.

To load the TPC-H dataset into DuckDB:
```
./build/release/duckdb {DATABASE_NAME}.duckdb
.read scripts/tpch_load.sql
```

To load the ClickBench dataset:
```
./build/release/duckdb {DATABASE_NAME}.duckdb
.read scripts/clickbench_load_duckdb.sql
```

## Testing

`gpu_processing` uses SQLLogic tests that compare Sirius results against DuckDB for correctness. These are end-to-end tests that run SQL queries and compare against expected results.

Generate the datasets using the method described [above](#generating-and-loading-test-datasets), then run:
```
pixi run sql-test
```

To run a specific test:
```
pixi run sql-test test/sql/tpch-sirius.test
```
