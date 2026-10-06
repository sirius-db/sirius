#!/usr/bin/env bash
# Run Cargo with the same Conda C/C++ toolchain and runtime libraries as libsirius.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "error: run through the Doris pixi default environment" >&2
    exit 2
fi

target="$(uname -m)-conda-linux-gnu"
export CC="${CC:-${CONDA_PREFIX}/bin/${target}-cc}"
export CXX="${CXX:-${CONDA_PREFIX}/bin/${target}-c++}"
build_dir="${SIRIUS_BUILD_DIR:-$(cd ../.. && pwd)/build/release}"
export LD_LIBRARY_PATH="${build_dir}/extension/sirius:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec cargo "$@"
