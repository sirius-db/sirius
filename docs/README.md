<!-- ![Sirius](sirius-full.png) -->
<p align="center">
  <img src="sirius-full.png" alt="Diagram" width="500"/>
</p>

Sirius is a GPU-native SQL engine. It plugs into existing databases such as DuckDB via the standard Substrait query format, requiring no query rewrites or major system changes. Sirius currently supports DuckDB and Doris (coming soon), other systems marked with * are on our roadmap.

<!-- ![Architecture](sirius-architecture.png) -->
<p align="center">
  <img src="sirius-architecture.png" alt="Diagram" width="900"/>
</p>

## Performance
Running TPC-H on SF=100, Sirius achieves ~10x speedup over existing CPU query engines at the same hardware rental cost, making it well-suited for interactive analytics, financial workloads, and ETL jobs.

Experiment Setup:
- GPU instance: GH200@LambdaLabs ($3.2/hour)
- CPU instance: m7i.16xlarge@AWS ($3.2/hour)

![Performance](sirius-performance.png)

## Supported OS/GPU/CUDA/CMake
- Ubuntu >= 20.04
- NVIDIA Volta™ or higher with compute capability 7.0+
- CUDA >= 11.2
- CMake >= 3.30.4 (follow this [instruction](https://medium.com/@yulin_li/how-to-update-cmake-on-ubuntu-9602521deecb) to upgrade CMake)
- We recommend building Sirius with at least **16 vCPUs** to ensure faster compilation.

## Dependencies (Option 1): Use AWS Image
For users who have access to AWS and want to launch AWS EC2 instances to run Sirius, the following images are prepared with dependencies fully installed.

<table border="1" cellpadding="6" cellspacing="0">
  <tr>
    <td><b>AMI Name</b></td>
    <td><b>AWS Region</b></td>
    <td><b>AMI ID</b></td>
  </tr>
  <tr>
    <td rowspan="3">Sirius Dependencies AMI (Ubuntu 24.04) 20250611</td>
    <td>us-east-1</td>
    <td>ami-06020f2b2161f5d62</td>
  </tr>
  <tr>
    <td>us-east-2</td>
    <td>ami-016b589f441fecc5d</td>
  </tr>
  <tr>
    <td>us-west-2</td>
    <td>ami-060043bae3f9b5eb4</td>
  </tr>
</table>

Supported EC2 instances: G4dn, G5, G6, Gr6, G6e, P4, P5, P6.

## Dependencies (Option 2): Use Docker Image
To use the docker image with dependencies fully installed:

For x86 machine:
```
sudo docker run --gpus all -it siriusdb/sirius_dependencies_x86_64:stable bash
```
For aarch64 machine:
```
sudo docker run --gpus all -it siriusdb/sirius_dependencies_aarch64:stable bash
```

If encounting errors like the following when running the docker image as above:
```
docker: Error response from daemon: could not select device driver “” with capabilities: [[gpu]].
```
This means `nvidia-driver` or `nvidia-container-toolkit` is not installed.

To install `nvidia-driver`:
```
sudo apt install nvidia-driver-535
```

To install `nvidia-container-toolkit`, please follow the [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Finally restart docker by
```
sudo systemctl restart docker
```

## Dependencies (Option 3): Install Manually

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

This approach requires cloning the repo following the instructions specifed in [Building Sirius](#building-sirius). 

libcudf will be installed via conda/miniconda. Miniconda can be downloaded [here](https://www.anaconda.com/docs/getting-started/miniconda/install). After downloading miniconda, install libcudf by running these commands:
```
conda create --name libcudf-env
conda activate libcudf-env
conda install -c rapidsai -c conda-forge -c nvidia rapidsai::libcudf
```
Set the environment variables `LIBCUDF_ENV_PREFIX` to the conda environment's path. For example, if we installed miniconda in `~/miniconda3` and installed libcudf in the conda environment `libcudf-env`, then we would set the `LIBCUDF_ENV_PREFIX` to `~/miniconda3/envs/libcudf-env`.
```
export LIBCUDF_ENV_PREFIX={PATH to libcudf-env}
```
It is recommended to add the environment variables to your `bashrc` to avoid repetition. 

## Building Sirius
To clone the Sirius repository:
```
git clone --recurse-submodules https://github.com/sirius-db/sirius.git
cd sirius
source setup_sirius.sh
```
The `--recurse-submodules` will ensure DuckDB is pulled which is required to build the extension.

To build Sirius:
```
CMAKE_BUILD_PARALLEL_LEVEL={nproc} make
```

Common issues: 
If you encounter an error such as:
```
/usr/bin/ld: /home/ubuntu/miniconda3/envs/libcudf-env/lib/libcudf.so: undefined reference to `std::ios_base_library_init()@GLIBCXX_3.4.32'
/usr/bin/ld: /home/ubuntu/miniconda3/envs/libcudf-env/lib/libcudf.so: undefined reference to `__cxa_call_terminate@CXXABI_1.3.15'
```
Solve this issue by running the following command, then delete the build directory and install Sirius once again:
```
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib $LDFLAGS"
rm -rf build
CMAKE_BUILD_PARALLEL_LEVEL={nproc} make
```

Note that if building the extension consumes too much memory, try reducing the `CMAKE_BUILD_PARALLEL_LEVEL` value used when invoking `make`.

Optionally, to use the Python API in Sirius, we also need to build the duckdb-python package with the following commands:
```
cd duckdb/tools/pythonpkg/
pip install .
cd $SIRIUS_HOME_PATH
```
Common issues: If `pip install .` only works inside an environment, then do the following from the Sirius home directory before the installation:
```
python3 -m venv --prompt duckdb .venv
source .venv/bin/activate
```

## Generating and Loading test datasets

### TPC-H Dataset 

To generate the TPC-H dataset
```
cd test_datasets
unzip tpch-dbgen.zip
cd tpch-dbgen
./dbgen -s 1 && mkdir s1 && mv *.tbl s1  # this generates dataset of SF1
cd ../../
```

To load the TPC-H dataset to duckdb:
```
./build/release/duckdb {DATABASE_NAME}.duckdb
.read tpch_load.sql
```

### ClickBench Dataset

To download the dataset run:
```
cd test_datasets
wget https://pages.cs.wisc.edu/~yxy/sirius-datasets/test_hits.tsv.gz
gzip -d test_hits.tsv.gz
cd ..
```

To load the dataset to duckdb:
```
./build/release/duckdb {DATABASE_NAME}.duckdb
.read clickbench_load_duckdb.sql
```

## Running Sirius: CLI
To run Sirius CLI, simply start the shell with `./build/release/duckdb {DATABASE_NAME}.duckdb`. 
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

Sirius provides a unit test that compares Sirius against DuckDB for correctness across many test queries. Note that these tests are meant to be end to end tests as they run SQL queries using Sirius and compare that against the expected result. To run the unittest, generate the datasets using the method described [here](#generating-and-loading-test-datasets) and run the unittest using the following command:
```
make test
```

To run a specific test run the command from the root directory:
```
CMAKE_BUILD_PARALLEL_LEVEL={nproc} make
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

### C++ Tests

Sirius also implements C++ tests for all of the APIs it implements. These tests are meant to be individual unit tests for each of the classes/functions used to run Sirius. You can find examples on how to implement these unit tests in `test/cpp`. You can run all of the unit tests using:
```
CMAKE_BUILD_PARALLEL_LEVEL={nproc} make
build/release/extension/sirius/test/cpp/sirius_unittest
```

To run tests associated with specific tag or to run a specific test you can execute the the test script like this:
```
CMAKE_BUILD_PARALLEL_LEVEL={nproc} make
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"
```

Any logs produced during test execution are saved in: 
```
build/release/extension/sirius/test/cpp/log
```

Just like duckdb, we are using [Catch2](https://github.com/catchorg/Catch2) as our testing framework so more details about writing and running tests can be found there.  

## Performance Testing
Make sure to build the duckdb-python package before running this test using the method described [here](https://github.com/sirius-db/sirius?tab=readme-ov-file#building-sirius). To test Sirius performance against DuckDB across all 22 TPC-H queries, run the following command (replace {SF} with the desired scale factor):
```
python3 test/tpch_performance/generate_test_data.py {SF}
python3 test/tpch_performance/performance_test.py {SF}
```

## Logging
Sirius uses [spdlog](https://github.com/gabime/spdlog) for logging messages during query execution. Default log directory is `${CMAKE_BINARY_DIR}/log` and default log level is `info`, which can be configured by environment variables `SIRIUS_LOG_DIR` and `SIRIUS_LOG_LEVEL`. For example:
```
export SIRIUS_LOG_DIR={PATH for logging}
export SIRIUS_LOG_LEVEL=debug
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
