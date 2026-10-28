#!/usr/bin/env bash

cd test_datasets

if [ ! -f tpch-dbgen/s10/customer.tbl ]; then
    unzip -n tpch-dbgen.zip
    cd tpch-dbgen
    make -f Makefile dbgen
    ./dbgen -f -s 10 && mkdir -p s10 && mv *.tbl s10
    cd ..
fi

if [ ! -f hits_0.parquet ]; then
    wget https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet
fi

cd ..
