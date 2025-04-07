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
```
Note that `--recurse-submodules` will ensure DuckDB is pulled which is required to build the extension.

## Building
```
cd duckdb
mkdir -p extension_external
cd extension_external
git clone https://github.com/duckdb/substrait.git
cd substrait
git reset --hard 611d92b9980c3b673ba3755bc10dfdb6f94e7384 #go to the right commit hash for duckdb substrait extension
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

We also provided a script (load.txt) to cache all the TPC-H columns in GPUs.
```
D .read load.txt
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

## Changing the caching and processing region (optional)
The GPU caching region is a memory region where the raw data is stored in GPUs. The GPUs/CPUs processing region is a memory region where intermediate results are stored in GPUs/CPUs (hash tables, .etc). The default region sizes are 10GB, 11GB, and 16GB for the GPU caching size, the GPU processing size, and the CPU processing size, respectively. The users can also modify these parameters by setting it in [SiriusExtension::GPUCachingBind](https://github.com/sirius-db/sirius/blob/058ee7291c5321727f566a2a72dda267c294f624/src/sirius_extension.cpp#L89).

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
$ call gpu_caching("orders.o_comment");
$ call gpu_caching("orders.o_orderkey");
$ call gpu_processing("select o_orderkey, o_comment from orders where o_comment like '%special%requests%';");
```

Substring Queries:
```
$ call gpu_caching("customer.c_phone");
$ call gpu_processing("select substr(c_phone, 1, 2) as countrycode from customer where substr(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17')")
```

Prefix query:
```
$ call gpu_caching("part.p_name");
$ select p_name from part where p_name like 'forest%';
$ call gpu_processing("select p_name from part where p_name like 'forest%';");
```
