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

Start duckdb using: `./build/release/duckdb tpch_s1.duckdb`. 

Example table creator:
```
$ CREATE TABLE example_strs(record VARCHAR);
$ INSERT INTO example_strs (record) VALUES ('hello world');
$ INSERT INTO example_strs (record) VALUES ('lorem ipsum');
$ INSERT INTO example_strs (record) VALUES ('running example values');
```

String Matching queries:
```
$ SELECT record FROM example_strs;
$ call gpu_caching("example_strs.record");
$ call gpu_processing("select record from example_strs;");
$ call gpu_processing("select record FROM example_strs WHERE record LIKE '%hello%';");
$ call gpu_processing("select record FROM example_strs WHERE record LIKE '%lorem%ipsum%';");
```

Substring Queries:
```
$ select o_orderkey, o_comment from orders where o_comment like '%special%requests%';
$ call gpu_caching("orders.o_comment");
$ call gpu_caching("orders.o_orderkey");
$ call gpu_processing("select o_orderkey, o_comment from orders where o_comment like '%special%requests%';");
```

Duck DB query:
```
$ SELECT p_partkey, p_comment FROM part WHERE p_comment LIKE '%wake%' ORDER BY p_partkey DESC LIMIT 50;
```
