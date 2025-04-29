# Sirius
This repository contains Sirius, a GPU-accelerated DuckDB extension

## Getting started
Install duckdb dependencies
```
sudo apt-get update && sudo apt-get install -y git g++ cmake ninja-build libssl-dev
```

If CUDA is not installed in your machine, install cuda from https://developer.nvidia.com/cuda-downloads. Note: Use the deb(local) installer.
After that, perform the post-installation from https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#mandatory-actions.

Check if CUDA is available by running
```
nvcc --version
nvidia-smi
```

Clone the Sirius repository using 
```
git clone --recurse-submodules https://github.com/bwyogatama/sirius.git
cd sirius
export SIRIUS_HOME_PATH=`pwd`
```
Note that `--recurse-submodules` will ensure DuckDB is pulled which is required to build the extension.

## Building
```
cd duckdb
mkdir -p extension_external
cd extension_external
git clone https://github.com/duckdb/substrait.git
cd substrait
git reset --hard ec9f8725df7aa22bae7217ece2f221ac37563da4 #go to the right commit hash for duckdb substrait extension
cd $SIRIUS_HOME_PATH
make -j {nproc} #build extension
```
Currently, we are using duckdb v1.0.0. Since we develop it as an extension and no modification is made to the duckdb source code, it should not be too difficult to bump it to the latest duckdb and substrait version.

## Running the extension
To run the extension code, simply start the shell with `./build/release/duckdb`. This shell will have the extension pre-loaded. 

<!-- To cache data in GPUs (e.g. caching l_orderkey from lineitem)
```
D call gpu_caching("lineitem.l_orderkey")
``` -->

<!-- We also provided a script (load.txt) to cache all the TPC-H columns in GPUs.
```
D .read load.txt
``` -->

To execute query on GPUs
```
D call gpu_processing("select
  l_orderkey,
  sum(l_extendedprice) as revenue,
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
  o_shippriority;")
```

The cold run would be slow as Sirius would need to read the data from storage via DuckDB and does a data format conversion from the DuckDB format to Sirius native format. The hot run of the queries would be significantly faster as the data would alreadby be cached on the device memory.

## Generating TPC-H dataset
Unzip `dbgen.zip` and run `./dbgen -s {SF}`.
To load the dataset to duckdb, use the SQL command in `$SIRIUS_HOME_PATH\tpch_load_duckdb_simple.sql`.

## Changing the caching and processing region (optional)
The GPU caching region is a memory region where the raw data is stored in GPUs. The GPUs/CPUs processing region is a memory region where intermediate results are stored in GPUs/CPUs (hash tables, .etc). The default region sizes are 10GB, 11GB, and 16GB for the GPU caching size, the GPU processing size, and the CPU processing size, respectively. The users can also modify these parameters by setting it in [SiriusExtension::GPUCachingBind](https://github.com/sirius-db/sirius/blob/058ee7291c5321727f566a2a72dda267c294f624/src/sirius_extension.cpp#L89).

## Running the queries
The TPC-H queries is in the `queries` folder. 
Queries in the `queries/working` folder should work in Sirius (These queries does not include string and order by operations).
Queries in the `queries/inprogress` folder is still under development.

## Using libcudf with Sirius (optional)
If users want to integrate with libcudf, it is recommended to install libcudf via conda/miniconda. Miniconda can be downloaded [here](https://www.anaconda.com/docs/getting-started/miniconda/install). After downloading miniconda, user can install libcudf via these commands:
```
conda create --name libcudf-env
conda activate libcudf-env
conda install -c rapidsai -c conda-forge -c nvidia rapidsai::libcudf
```
User would also need to make sure that the environment variable `LIBCUDF_ENV_PREFIX` is set to the path to the conda environment's directory. For example, if you installed miniconda to the path `~/miniconda3` and you installed libcudf in the conda environment `libcudf-env` then you would set the `LIBCUDF_ENV_PREFIX` to `~/miniconda3/envs/libcudf-env`

libcudf might requires a later cmake version, as of April 2025, it would require cmake version > 3.30.4. User can follow the instruction in this [link](https://medium.com/@yulin_li/how-to-update-cmake-on-ubuntu-9602521deecb) to download the specific cmake version.

To use libcudf, set the environment variable USE_CUDF, and rebuild sirius.
```
export USE_CUDF=1
rm -r build/*
make -j {nproc}
```
