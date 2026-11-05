#!/usr/bin/env bash
# Builds the pinned pre-v2 Substrait reader as a loadable DuckDB extension and
# a DuckDB shell with it linked in. The CPU differential feeds translator plans
# to this consumer without a GPU; Sirius embeds the v2 reader separately.
#
#   scripts/build-duckdb-substrait.sh            # configure + build (idempotent)
#   scripts/build-duckdb-substrait.sh --clean    # wipe the build directory first
#
# Output (under .duckdb-substrait/, ignored by git):
#   build/duckdb                                          shell with the extension linked in
#   build/extension/substrait/substrait.duckdb_extension  loadable extension (unsigned)
#
# Keep DUCKDB_TAG in sync with python-duckdb in pixi.toml's check feature:
# a loadable extension only loads into the exact same version. SUBSTRAIT_COMMIT
# must also support that DuckDB version.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DUCKDB_TAG="${DUCKDB_TAG:-v1.5.5}"
WORK="${ROOT}/.duckdb-substrait"
SRC="${WORK}/duckdb"
BUILD="${WORK}/build"
EXT_SRC="${WORK}/substrait"
SUBSTRAIT_COMMIT=a7e045befd5479569f0a84b241868253cdfde0b3

if [ "${1:-}" = "--clean" ]; then
    rm -rf "${BUILD}"
fi

mkdir -p "${WORK}"
if [ ! -d "${EXT_SRC}/.git" ]; then
    git init -q "${EXT_SRC}"
    git -C "${EXT_SRC}" remote add origin https://github.com/sirius-db/duckdb-substrait-extension.git
fi
if [ "$(git -C "${EXT_SRC}" rev-parse HEAD 2>/dev/null || true)" != "${SUBSTRAIT_COMMIT}" ]; then
    git -C "${EXT_SRC}" fetch -q --depth=1 origin "${SUBSTRAIT_COMMIT}"
    git -C "${EXT_SRC}" checkout -q --detach "${SUBSTRAIT_COMMIT}"
fi
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
