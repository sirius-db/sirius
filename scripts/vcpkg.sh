#!/usr/bin/env sh -e

if [ ! -f "./vcpkg/vcpkg" ]; then
  ./vcpkg/bootstrap-vcpkg.sh
fi
