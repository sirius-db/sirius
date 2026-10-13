#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  exit 0
fi

if [[ -z "${LIBCLANG_PATH:-}" ]]; then
  export LIBCLANG_PATH="$CONDA_PREFIX/lib"
fi

clang_cpp="$CONDA_PREFIX/bin/clang-cpp"
clang_pp="$CONDA_PREFIX/bin/clang++"

if [[ -x "$clang_cpp" ]]; then
  ln -sf "$clang_cpp" "$clang_pp"
fi

mkdir -p build
