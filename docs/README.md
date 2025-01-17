# Sirius
This repository contains Sirius, a GPU-accelerated DuckDB extension

## Getting started
Install duckdb dependencies
```
$ sudo apt-get update && sudo apt-get install -y git g++ cmake ninja-build libssl-dev
```

If CUDA is not installed in your machine, install cuda from https://developer.nvidia.com/cuda-downloads. Note: Use the deb(local) installer.
After that, perform the post-installation from https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#mandatory-actions.

Check if CUDA is available by running
```
$ nvcc --version
$ nvidia-smi
```

Clone the Sirius repository using 
```
$ git clone --recurse-submodules https://github.com/bwyogatama/sirius.git
```
Note that `--recurse-submodules` will ensure DuckDB is pulled which is required to build the extension.

## Building
```
$ cd duckdb
$ git reset --hard 1f98600c2cf8722a6d2f2d805bb4af5e701319fc #go to the commit hash of duckdb v1.0.0
$ mkdir -p extension_external
$ cd extension_external
$ git clone https://github.com/duckdb/substrait.git
$ cd substrait 
$ git reset --hard b6f56643cb11d52de0e32c24a01dfd5947df62be #go to the right commit hash for duckdb substrait extension
$ cd {SIRIUS_HOME_PATH}
$ make -j 8 #build extension
```
Currently, we are using duckdb v1.0.0. Since we develop it as an extension and no modification is made to the duckdb source code, it should not be too difficult to bump it to the latest duckdb and substrait version.

## Running the extension
To run the extension code, simply start the shell with `./build/release/duckdb`. This shell will have the extension pre-loaded. 

To cache data in GPUs (e.g. caching l_orderkey from lineitem)
```
D call gpu_caching("lineitem.l_orderkey")
```

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

## Generating TPC-H dataset
Unzip `dbgen.zip` and run `./dbgen -s {SF}`.
To load the dataset to duckdb, use the SQL command in `{SIRIUS_HOME_PATH}\tpch_load_duckdb_simple.sql`.

## Running the queries
The TPC-H queries is in the `queries` folder. 
Queries in the `queries/working` folder should work in Sirius (These queries does not include string and order by operations).
Queries in the `queries/inprogress` folder is still under development.

## Devesh Notes
We have provided a helper docker container that you can easily use to install all the depedencies:
```
$ cd ..
$ export CURRENT_DIR=`pwd`
$ cd sirius
$ docker build -t sirius:latest docker/.
$ docker kill sirius
$ docker rm sirius
$ docker run --gpus all -d -v $CURRENT_DIR:/working_dir/ --name=sirius --cap-add=SYS_ADMIN sirius:latest sleep infinity
$ docker exec -it sirius bash
$ cd sirius
```

Build the code:
```
$ make -j$(nproc)
```

Start duckdb using: `./build/release/duckdb tpch_s10.duckdb`. 

Group By Query:
```
$ call gpu_caching("customer.c_comment");
$ call gpu_caching("customer.c_nationkey");
$ call gpu_caching("nation.n_nationkey");
$ call gpu_caching("nation.n_comment");
$ call gpu_processing("SELECT n_comment, c_comment, COUNT(*) FROM customer, nation WHERE customer.c_nationkey = nation.n_nationkey GROUP BY n_comment, c_comment;");
```