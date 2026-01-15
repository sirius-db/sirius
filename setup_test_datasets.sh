#!/bin/bash

cd test_datasets

if [ ! -f tpch-dbgen/s1/customer.tbl ]; then
    unzip -n tpch-dbgen.zip
    cd tpch-dbgen
    ./dbgen -f -s 1 && mkdir -p s1 && mv *.tbl s1
    cd ..
fi

if [ ! -f test_hits.tsv ]; then
    wget https://pages.cs.wisc.edu/~yxy/sirius-datasets/test_hits.tsv.gz
    gzip -d test_hits.tsv.gz
fi

cd ..
