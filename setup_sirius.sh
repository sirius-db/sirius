#!/bin/bash

git submodule update --init --recursive
export SIRIUS_HOME_PATH=`pwd`
cd duckdb
mkdir -p extension_external && cd extension_external
git clone https://github.com/duckdb/substrait.git
cd substrait
git reset --hard ec9f8725df7aa22bae7217ece2f221ac37563da4 # go to the right commit hash for duckdb substrait extension
cd $SIRIUS_HOME_PATH
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib $LDFLAGS"
export CUDF_VERSION=$(conda list cudf | awk '!/^#/ && $1 ~ /^(cudf|libcudf|cudf-python)$/ {print $2; exit}')