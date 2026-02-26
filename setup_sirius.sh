#!/usr/bin/env bash

git submodule update --init --recursive
export SIRIUS_HOME_PATH=`pwd`
cd $SIRIUS_HOME_PATH
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib $LDFLAGS"
