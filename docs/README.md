<!-- ![Sirius](sirius-full.png) -->
<p align="center">
  <img src="sirius-full.png" alt="Diagram" width="500"/>
</p>

Sirius is a GPU-native SQL engine. It plugs into existing databases such as DuckDB via the standard Substrait query format, requiring no query rewrites or major system changes. Sirius currently supports DuckDB and Doris (coming soon), other systems marked with * are on our roadmap. Built on NVIDIA CUDA-X libraries including cuDF and RAPIDS Memory Manager (RMM), Sirius delivers high-performance GPU-accelerated analytics.

<!-- ![Architecture](sirius-architecture.png) -->
<p align="center">
  <img src="sirius-arch.png" alt="Diagram" width="900"/>
</p>

## Performance
Running TPC-H on SF=100, Sirius achieves ~8x speedup over existing CPU query engines at the same hardware rental cost, making it well-suited for interactive analytics, financial workloads, and ETL jobs.

Experiment Setup:
- GPU instance: GH200@LambdaLabs ($1.5/hour)
- CPU instance: c8i.8xlarge@AWS ($1.5/hour)

![Performance](sirius-perf.png)

## Requirements

- Linux (x86_64 or aarch64)
- NVIDIA Volta™ or higher with compute capability 7.0+
- CUDA >= 13.0 (driver only, requires NVIDIA driver >= 570 — the toolkit is provided by Pixi)
- Git
- [Pixi](https://pixi.sh/latest/installation/) (manages all build dependencies: CMake, clang, cuDF, RMM, spdlog, etc.)

## Building Sirius

To clone the Sirius repository:
```
git clone --recurse-submodules https://github.com/sirius-db/sirius.git
cd sirius
```
The `--recurse-submodules` will ensure DuckDB is pulled which is required to build the extension.

Pixi manages all build dependencies (CUDA toolkit, cuDF, CMake, clang, etc.). Start a shell in the environment with:
```
pixi shell
```

To build Sirius (release by default):
```
pixi run build
```

Other build presets are available:
```
pixi run build debug              # Debug build
pixi run build relwithdebinfo     # Release with debug symbols (for profiling)
pixi run build clang-debug        # Clang debug build (for sanitizers)
```

If building consumes too much memory, reduce ninja parallelism:
```
CMAKE_BUILD_PARALLEL_LEVEL=8 pixi run build
```

Optionally, to use the Python API in Sirius, we also need to build the duckdb-python package with the following commands:
```
pushd duckdb-python
pip install .
popd
```
Common issues: If `pip install .` only works inside an environment, then do the following from the Sirius home directory before the installation:
```
python3 -m venv --prompt duckdb .venv
source .venv/bin/activate
```

## Generating test datasets

A setup script generates both TPC-H (SF1) and ClickBench datasets in one step:
```
./setup_test_datasets.sh
```

This creates the necessary data files under `test_datasets/`. The script is idempotent — it skips datasets that already exist.

### Loading datasets into DuckDB

To load the TPC-H dataset:
```
./build/release/duckdb {DATABASE_NAME}.duckdb
.read scripts/tpch_load.sql
```

To load the ClickBench dataset:
```
./build/release/duckdb {DATABASE_NAME}.duckdb
.read scripts/clickbench_load_duckdb.sql
```

## Running Sirius: CLI
To run Sirius CLI, start the DuckDB shell with Sirius preloaded (rebuilds if needed):
```
pixi run duckdb
```
From the duckdb shell, initialize the Sirius buffer manager with `call gpu_buffer_init`. This API accepts 2 parameters, the GPU caching region size and the GPU processing region size. The GPU caching region is a memory region where the raw data is stored in GPUs, whereas the GPU processing region is where intermediate results are stored in GPUs (hash tables, join results .etc).
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

## Running Sirius: Python API
Make sure to build the duckdb-python package before using the Python API with the method described [here](https://github.com/sirius-db/sirius?tab=readme-ov-file#building-sirius). To use the Sirius Python API, add the following code to the beginning of your Python program:
```
import duckdb
con = duckdb.connect('{DATABASE_NAME}.duckdb', config={"allow_unsigned_extensions": "true"})
con.execute("load '{SIRIUS_HOME_PATH}/build/release/extension/sirius/sirius.duckdb_extension'")
con.execute("call gpu_buffer_init('{GPU_CACHE_SIZE}', '{GPU_PROCESSING_SIZE}')")
```
To execute query in Python:
```
con.execute('''
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
            ''').fetchall()
```

## Correctness Testing

### SQLLogic Tests

Sirius provides SQL logic tests that compare Sirius against DuckDB for correctness. Generate the datasets using the method described [here](#generating-test-datasets), then run:
```
pixi run sql-test
```

To run a specific test file:
```
pixi run sql-test test/sql/tpch-sirius.test
```

### C++ Tests

Sirius also implements C++ unit tests for individual classes and functions. You can find examples in `test/cpp`. To run all unit tests:
```
pixi run unittest
```

To run tests by tag or name:
```
pixi run unittest "[cpu_cache]"
pixi run unittest "test_cpu_cache_basic_string_single_col"
```

All test tasks automatically rebuild if sources have changed. Test logs are saved in `build/<preset>/extension/sirius/test/cpp/log`.

We use [Catch2](https://github.com/catchorg/Catch2) as the testing framework.

## Logging
Sirius uses [spdlog](https://github.com/gabime/spdlog) for logging messages during query execution. Default log directory is `log` (relative to the current working directory) and default log level is `info`.

Log directory and level can be initialized via environment variables before loading the extension:
```bash
export SIRIUS_LOG_DIR=/path/to/logs
export SIRIUS_LOG_LEVEL=debug
```

Both can also be configured at runtime via DuckDB's `SET` command:
```sql
SET sirius_log_dir = '/path/to/logs';
SET sirius_log_level = 'debug';
SET sirius_log_flush_seconds = 1;
```

## Limitations
Sirius is under active development, and several features are still in progress. Notable current limitations include:
- **Data Size Limitations:** Sirius currently only works when the dataset fits in the GPU memory capacity. To be more specific, it would return an error if the input data size is larger than the GPU caching region or if the intermediate results is larger than the GPU processing region. We are actively addressing this issue by adding support for partitioning and batch execution (issues [#12](https://github.com/sirius-db/sirius/issues/12) and [#19](https://github.com/sirius-db/sirius/issues/19)), multi-GPUs execution (issue [#18](https://github.com/sirius-db/sirius/issues/18)), spilling to disk/host memory (issue [#19](https://github.com/sirius-db/sirius/issues/19)), and distributed query execution (issue [#18](https://github.com/sirius-db/sirius/issues/18)).
- **Row Count Limitations:** Sirius uses libcudf to implement `FILTER`, `PROJECTION`, `JOIN`, `GROUP-BY`, `ORDER-BY`, `AGGREGATION`. However, since libcudf uses `int32_t` for row IDs, this imposes limits on the maximum row count that Sirius can currently handle (~2B rows). See libcudf issue [#13159](https://github.com/rapidsai/cudf/issues/13159) for more details. We are actively addressing this by adding support for partitioning and batch execution. See Sirius issue [#12](https://github.com/sirius-db/sirius/issues/12) for more details.
- **Data Type Coverage:** Sirius currently supports commonly used data types including `INTEGER`, `BIGINT`, `FLOAT`, `DOUBLE`, `VARCHAR`, `DATE`, `TIMESTAMP`, and `DECIMAL`. We are actively working on supporting additional data types—such as nested types. See issue [#20](https://github.com/sirius-db/sirius/issues/20) for more details.
- **Operator Coverage:** At present, Sirius only supports a range of operators including `FILTER`, `PROJECTION`, `JOIN`, `GROUP-BY`, `ORDER-BY`, `AGGREGATION`, `TOP-N`, `LIMIT`, and `CTE`. We are working on adding more advanced operators such as `WINDOW` functions and `ASOF JOIN`, etc. See issue [#21](https://github.com/sirius-db/sirius/issues/21) for more details.

For a full list of current limitations and ongoing work, please refer to our [GitHub issues page](https://github.com/sirius-db/sirius/issues). **If these issues are encountered when running Sirius, Sirius will gracefully fallback to DuckDB query execution on CPUs.**

## Future Roadmap
Sirius is still under major development and we are working on adding more features to Sirius, such as [storage/disk support](https://github.com/sirius-db/sirius/issues/19), [multi-GPUs](https://github.com/sirius-db/sirius/issues/18), [multi-node](https://github.com/sirius-db/sirius/issues/18), more [operators](https://github.com/sirius-db/sirius/issues/21), [data types](https://github.com/sirius-db/sirius/issues/20), accelerating more engines, and many more.

Sirius always welcomes new contributors! If you are interested, check our [website](https://www.sirius-db.com/), reach out to our [email](siriusdb@cs.wisc.edu), or join our [slack channel](https://join.slack.com/t/sirius-db/shared_invite/zt-33tuwt1sk-aa2dk0EU_dNjklSjIGW3vg).

**Let's kickstart the GPU eras for Data Analytics!**
