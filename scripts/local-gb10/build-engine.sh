#!/usr/bin/env bash
# Native NVIDIA GB10 build; the generated device code targets this machine's SM121 GPU.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec pixi run --frozen bash -c '
    export CUDAARCHS=121-real
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
    exec make "$@"
' bash "$@"
