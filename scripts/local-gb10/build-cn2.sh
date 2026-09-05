#!/usr/bin/env bash
# Build the engine-linked CN with real NIXL; run build-engine.sh and build-transport.sh first.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export TOOLS_DIR="${TRANSPORT_ROOT:-$repo_root/build/local-gb10/transport}"
cd "$repo_root/experimental/starrocks"
exec pixi run --frozen bash -c '
    set -euo pipefail
    source scripts/cn-env.sh
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TOOLS_DIR/toolenv/.pixi/envs/default/lib:/usr/lib/aarch64-linux-gnu"
    "$CONDA_PREFIX/bin/cargo" rustc --release -p sirius-starrocks-cn \
        --bin sirius-starrocks-cn -j "${CARGO_BUILD_JOBS:-4}" -- -C linker=/usr/bin/g++
    readelf -d target/release/sirius-starrocks-cn | grep -E "NEEDED.*(nixl|sirius)"
    readelf -d target/release/sirius-starrocks-cn | grep -q "NEEDED.*libnixl"
    link_check="$(ldd target/release/sirius-starrocks-cn 2>&1)"
    echo "$link_check"
    if [[ "$link_check" == *"not found"* ]]; then
        echo "Unresolved CN runtime dependency" >&2
        exit 1
    fi
'
