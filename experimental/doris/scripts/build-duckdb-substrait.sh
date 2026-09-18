#!/usr/bin/env bash
# Builds the DuckDB substrait consumer Sirius uses (the repo's `substrait/` submodule,
# `SubstraitToDuckDB` in src/from_substrait.cpp — the same reader `src/sirius_ffi.cpp`
# compiles into libsirius) as a loadable DuckDB extension, plus a DuckDB shell with it
# linked in. This is what the CPU differential (scripts/validate_tpch_results.py consume)
# feeds the translator's plans to on a machine without a GPU.
#
#   scripts/build-duckdb-substrait.sh            # configure + build (idempotent)
#   scripts/build-duckdb-substrait.sh --clean    # wipe the build directory first
#
# Output (under .duckdb-substrait/, ignored by git):
#   build/duckdb                                          shell with the extension linked in
#   build/extension/substrait/substrait.duckdb_extension  loadable extension (unsigned)
#
# DuckDB itself is a shallow clone of the upstream tag the extension is pinned to
# (`substrait/.gitmodules` duckdb → v1.5.5). Keep DUCKDB_TAG in sync with `python-duckdb` in
# pixi.toml's `check` feature: a loadable extension only loads into the exact same version.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"
DUCKDB_TAG="${DUCKDB_TAG:-v1.5.5}"
WORK="${ROOT}/.duckdb-substrait"
SRC="${WORK}/duckdb"
BUILD="${WORK}/build"
EXT_SRC="${REPO_ROOT}/substrait"

if [ "${1:-}" = "--clean" ]; then
    rm -rf "${BUILD}"
fi

if [ ! -f "${EXT_SRC}/src/from_substrait.cpp" ]; then
    echo "error: ${EXT_SRC} is empty; run 'git submodule update --init --depth=1 substrait' at the repo root" >&2
    exit 1
fi

mkdir -p "${WORK}"
if [ ! -d "${SRC}/.git" ]; then
    echo "==> cloning duckdb ${DUCKDB_TAG} into ${SRC}"
    git clone -q --depth=1 --branch "${DUCKDB_TAG}" https://github.com/duckdb/duckdb.git "${SRC}"
fi
actual_tag="$(git -C "${SRC}" describe --tags --exact-match 2>/dev/null || git -C "${SRC}" rev-parse --short HEAD)"
echo "==> duckdb source: ${SRC} (${actual_tag})"

# core_functions and parquet come from DuckDB's base extension config; substrait is ours.
cat > "${WORK}/extension_config.cmake" <<EOF
duckdb_extension_load(substrait
    SOURCE_DIR ${EXT_SRC}
    INCLUDE_DIR ${EXT_SRC}/src/include)
EOF

cmake_args=(
    -S "${SRC}" -B "${BUILD}" -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DDUCKDB_EXTENSION_CONFIGS="${WORK}/extension_config.cmake"
    -DBUILD_UNITTESTS=OFF
    -DBUILD_BENCHMARKS=OFF
    -DBUILD_SHELL=ON
    -DENABLE_EXTENSION_AUTOLOADING=OFF
    -DENABLE_EXTENSION_AUTOINSTALL=OFF
)
if command -v ccache >/dev/null 2>&1; then
    cmake_args+=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

# The extension's test/c subdirectory needs the unittest tree; it is not part of this build.
export SKIP_SUBSTRAIT_C_TESTS=1
echo "==> configuring ${BUILD}"
cmake "${cmake_args[@]}"
echo "==> building duckdb shell + substrait loadable extension"
cmake --build "${BUILD}" --target duckdb substrait_loadable_extension

echo "==> done:"
ls -la "${BUILD}/duckdb" "${BUILD}/extension/substrait/substrait.duckdb_extension"
"${BUILD}/duckdb" -c "SELECT version() AS duckdb, extension_name, loaded FROM duckdb_extensions() WHERE extension_name = 'substrait'"
