#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root/experimental/starrocks"
exec pixi run --frozen bash -c '
    # Link against the host glibc and the transitive engine libraries.
    export PATH="/usr/bin:$PATH"
    export LD_LIBRARY_PATH="$PIXI_PROJECT_ROOT/../../.pixi/envs/default/lib:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    exec "$CONDA_PREFIX/bin/cargo" rustc --release -p sirius-starrocks-cn \
        --bin sirius-starrocks-cn -j "${CARGO_BUILD_JOBS:-4}" -- -C linker=/usr/bin/g++
'
