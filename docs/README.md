<!-- ![Sirius](sirius-full.png) -->
<p align="center">
  <img src="sirius-full.png" alt="Diagram" width="500"/>
</p>

# Sirius
Sirius is a GPU acceleration layer for SQL analytics. It plugs into existing engines such as DuckDB via the standard Substrait query format, requiring no query rewrites or major system changes. Currently supports DuckDB and Doris (coming soon), other systems marked with * are on our roadmap.

<!-- ![Architecture](sirius-architecture.png) -->
<p align="center">
  <img src="sirius-architecture.png" alt="Diagram" width="900"/>
</p>

## Supported OS/GPU/CUDA/CMake
- Ubuntu >= 20.04
- NVIDIA Volta™ or higher with compute capability 7.0+
- CUDA >= 11.2
- CMake >= 3.30.4 (follow this [instruction](https://medium.com/@yulin_li/how-to-update-cmake-on-ubuntu-9602521deecb) to upgrade CMake)

## Installing dependencies

### Install duckdb dependencies
```
sudo apt-get update && sudo apt-get install -y git g++ cmake ninja-build libssl-dev
```

### Install CUDA
If CUDA is not installed, download [here](https://developer.nvidia.com/cuda-downloads). Follow the instructions for the deb(local) installer and complete the [post-installation steps](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#mandatory-actions).

Verify installation:
```
nvcc --version
nvidia-smi
```

### Install libcudf dependencies
libcudf will be installed via conda/miniconda. Miniconda can be downloaded [here](https://www.anaconda.com/docs/getting-started/miniconda/install). After downloading miniconda, install libcudf by running these commands:
```
conda create --name libcudf-env
conda activate libcudf-env
conda install -c rapidsai -c conda-forge -c nvidia rapidsai::libcudf
```
Set the environment variables `USE_CUDF` to 1 and `LIBCUDF_ENV_PREFIX` to the conda environment's path. For example, if we installed miniconda in `~/miniconda3` and installed libcudf in the conda environment `libcudf-env`, then we would set the `LIBCUDF_ENV_PREFIX` to `~/miniconda3/envs/libcudf-env`.
```
export USE_CUDF=1
export LIBCUDF_ENV_PREFIX = {PATH to libcudf-env}
```

### Clone the Sirius repository
```
git clone --recurse-submodules https://github.com/bwyogatama/sirius.git
cd sirius
export SIRIUS_HOME_PATH=`pwd`
cd duckdb
mkdir -p extension_external && cd extension_external
git clone https://github.com/duckdb/substrait.git
cd substrait
git reset --hard ec9f8725df7aa22bae7217ece2f221ac37563da4 #go to the right commit hash for duckdb substrait extension
cd $SIRIUS_HOME_PATH
```
The `--recurse-submodules` will ensure DuckDB is pulled which is required to build the extension.

## Building Sirius
To build Sirius:
```
make -j {nproc}
```

## Generating TPC-H dataset
Unzip `dbgen.zip` and run `./dbgen -s {SF}`.
To load the TPC-H dataset to duckdb:
```
./build/release/duckdb {DATABASE_NAME}.duckdb
.read tpch_load_duckdb.sql
```

## Running Sirius
To run Sirius, simply start the shell with `./build/release/duckdb {DATABASE_NAME}.duckdb`. 
From the duckdb shell, initialize the Sirius buffer manager with `call gpu_buffer_init`. This API accepts 2 parameters, the GPU caching region size and the GPU processing region size. The GPU caching region is a memory region where the raw data is stored in GPUs, whereas the GPU processing region is where intermediate results are stored in GPUs (hash tables, join results .etc).
For example, to set the caching region as 1 GB and the processing region as 2 GB, we can run the following command:
```
call gpu_buffer_init("1 GB", "2 GB")
```

After setting up Sirius, we can execute SQL queries using the `call gpu_processing`:
```
call gpu_processing("select
  l_orderkey,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  o_orderdate,
  o_shippriority
from
  customer,
  orders,
  lineitem
where
  c_mktsegment = 1
  and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < 19950315
  and l_shipdate > 19950315
group by
  l_orderkey,
  o_orderdate,
  o_shippriority
order by
  revenue desc,
  o_orderdate");
```
The cold run in Sirius would be significantly slower due to data loading from storage and conversion from DuckDB format to Sirius native format. Subsequent runs would be faster since it benefits from caching on GPU memory.

All 22 TPC-H queries are saved in tpch-queries.sql. To run all queries:
```
.read tpch-queries.sql
```

## Testing
Sirius provides a unit test that compares Sirius against DuckDB for correctness across all 22 TPC-H queries. To run the unittest, generate SF=1 TPC-H dataset using method described [here](https://github.com/sirius-db/sirius?tab=readme-ov-file#generating-tpc-h-dataset) and run the unittest using the following command:
```
make test
```

## Logging
Sirius uses [spdlog](https://github.com/gabime/spdlog) for logging messages during query execution. Default log directory is `${CMAKE_BINARY_DIR}/log` and default log level is `info`, which can be configured by environment variables `SIRIUS_LOG_DIR` and `SIRIUS_LOG_LEVEL`. For example:
```
export SIRIUS_LOG_DIR={PATH for logging}
export SIRIUS_LOG_LEVEL=debug
```

## Performance
Running TPC-H on SF=100, Sirius achieves ~10x speedup over existing CPU query engines at the same hardware rental cost, making it well-suited for interactive analytics, financial workloads, and ETL jobs.

![Performance](sirius-performance.png)

## Limitations
Sirius is under active development, and several features are still in progress. Notable current limitations include:
- **Working Set Size Limitations:** Sirius recently switches to libcudf to implement `FILTER`, `PROJECTION`, `JOIN`, `GROUP-BY`, `ORDER-BY`, `AGGREGATION`. However, since libcudf uses `int32_t` for row IDs and string offsets, this imposes limits on the maximum working set size that Sirius can currently handle. For string columns this imposes a ~2 GB limit, for `int32_t` columns this imposes a ~8 GB limit. See libcudf issue [#13159](https://github.com/rapidsai/cudf/issues/13159) for more details. We are actively addressing this by adding support for partitioning and chunked pipeline execution. See Sirius issue [#12](https://github.com/sirius-db/sirius/issues/12) for more details.
- **Limited Data Type Support:** Sirius currently only supports `INTEGER`, `BIGINT`, `FLOAT`, `DOUBLE`, and `VARCHAR` data types. We are actively working on supporting additional data types—such as `DECIMAL`, `DATE/TIME`, and nested types. See issue [#20](https://github.com/sirius-db/sirius/issues/20) for more details.
- **Operator Coverage:** At present, Sirius only supports a range of operators including `FILTER`, `PROJECTION`, `JOIN`, `GROUP-BY`, `ORDER-BY`, `AGGREGATION`, `TOP-N`, `LIMIT`, and `CTE`. We are working on adding more advanced operators such as `WINDOW` functions and `ASOF JOIN`, etc. See issue [#21](https://github.com/sirius-db/sirius/issues/21) for more details.
- **No Support for Partially NULL Columns:** Sirius currently does not support columns where only some values are `NULL`. This limitation is being tracked and will be addressed in a future update. See issue [#27](https://github.com/sirius-db/sirius/issues/27) for more details.

For a full list of current limitations and ongoing work, please refer to our [GitHub issues page](https://github.com/sirius-db/sirius/issues). **If these issues are encountered when running Sirius, Sirius will gracefully fallback to DuckDB query execution on CPUs.**

## Future Roadmap
Sirius is still under major development and we are working on adding more features to Sirius, such as [storage/disk support](https://github.com/sirius-db/sirius/issues/19), [multi-GPUs](https://github.com/sirius-db/sirius/issues/18), [multi-node](https://github.com/sirius-db/sirius/issues/18), more [operators](https://github.com/sirius-db/sirius/issues/21), [data types](https://github.com/sirius-db/sirius/issues/20), accelerating more engines, and many more.

Sirius always welcomes new contributors! If you are interested, check our [website](https://www.sirius-db.com/), reach out to our [email](siriusdb@cs.wisc.edu), or join our [slack channel](https://join.slack.com/t/sirius-db/shared_invite/zt-33tuwt1sk-aa2dk0EU_dNjklSjIGW3vg).

**Let's kickstart the GPU eras for Data Analytics!**