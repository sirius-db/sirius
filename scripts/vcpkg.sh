#!/usr/bin/env sh -e

export VCPKG_DISABLE_METRICS=1

if [ ! -f "./sirius-duckdb/vcpkg/vcpkg" ]; then
  ./sirius-duckdb/vcpkg/bootstrap-vcpkg.sh
fi
