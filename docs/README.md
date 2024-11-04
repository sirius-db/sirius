# Sirius
This repository contains Sirius, a GPU-accelerated DuckDB extension

## Getting started
Clone your new repository using 
```sh
git clone --recurse-submodules https://github.com/bwyogatama/sirius.git
```
Note that `--recurse-submodules` will ensure DuckDB is pulled which is required to build the extension.

## Building
```
cd duckdb
git reset --hard 1f98600c2cf8722a6d2f2d805bb4af5e701319fc #go to the commit hash of duckdb v1.0.0
mkdir -p extension_external
cd extension_external
git clone https://github.com/duckdb/substrait.git
cd substrait 
git reset --hard b6f56643cb11d52de0e32c24a01dfd5947df62be #go to the right commit hash for duckdb substrait extension
cd {SIRIUS_HOME_PATH}
make -j 8 #build extension
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
  o_shippriority;");
```

## Generating TPC-H dataset
Checkout to the `bobbi/tpch` branch in the `new-crystal` repo and generate the tpch dataset from the `tpch_dataset_generator` directory. To load the dataset to duckdb, use the SQL command in `{SIRIUS_HOME_PATH}\tpch_load_duckdb_simple.sql`.


## Docker container
We have provided a helper docker container that you can easily use to install all the depedencies:
```
$ export CURRENT_DIR=`pwd`
$ docker build -t sirius:latest docker/.
$ docker kill sirius
$ docker rm sirius
$ docker run --gpus all -d -v $CURRENT_DIR:/working_dir/ --name=sirius --cap-add=SYS_ADMIN sirius:latest sleep infinity
$ docker exec -it sirius bash
```