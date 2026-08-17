#!/usr/bin/env bash

cd test_datasets

if [ ! -f tpch-dbgen/s1/customer.tbl ]; then
    unzip -n tpch-dbgen.zip
    cd tpch-dbgen
    make dbgen
    ./dbgen -f -s 1 && mkdir -p s1 && mv *.tbl s1
    cd ..
fi

if [ ! -f hits_0.parquet ]; then
    wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet
fi

cd ..
