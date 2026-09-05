#!/usr/bin/env bash
# Native ARM64 UCX CUDA transport and NIXL for the two-CN StarRocks stack.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
transport_root="${TRANSPORT_ROOT:-$repo_root/build/local-gb10/transport}"
export TRANSPORT_ROOT="$transport_root"
mkdir -p "$transport_root/toolenv"
if [[ ! -f "$transport_root/toolenv/pixi.toml" ]]; then
    cat > "$transport_root/toolenv/pixi.toml" <<'EOF'
[workspace]
name = "sirius-gb10-transport-tools"
channels = ["conda-forge"]
platforms = ["linux-aarch64"]

[dependencies]
meson = "*"
ninja = "*"
pkg-config = "*"
libhwloc = "*"
numactl = "*"
pybind11 = "*"
python = "3.13.*"
EOF
fi
if [[ "${1:-}" != --activated ]]; then
    pixi install --manifest-path "$transport_root/toolenv/pixi.toml"
    exec pixi run --frozen --manifest-path "$transport_root/toolenv/pixi.toml" \
        bash "$repo_root/scripts/local-gb10/build-transport.sh" --activated
fi

tool_env="$transport_root/toolenv/.pixi/envs/default"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export CUDACXX="$CUDA_HOME/bin/nvcc"
# This small tools environment has no compiler/linker; put its Python first for Meson.
export PATH="$tool_env/bin:/usr/bin:$CUDA_HOME/bin:$PATH"
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
export PKG_CONFIG_PATH="$transport_root/ucx-install/lib/pkgconfig:$tool_env/lib/pkgconfig"
export LD_LIBRARY_PATH="$transport_root/ucx-install/lib:$tool_env/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
jobs="${TRANSPORT_BUILD_JOBS:-4}"
cd "$transport_root"

if [[ ! -f ucx-install/lib/ucx/libuct_cuda.so ]]; then
    if [[ ! -f ucx-1.21.0.tar.gz ]]; then
        curl -L --fail --retry 3 -o ucx-1.21.0.tar.gz \
            https://github.com/openucx/ucx/releases/download/v1.21.0/ucx-1.21.0.tar.gz
    fi
    echo '2374d2fcf3186fbfd5e27633ab153aabaeb6b4f503a88563d2aca67cf51ed2c1  ucx-1.21.0.tar.gz' | sha256sum --check
    [[ -d ucx-1.21.0 ]] || tar xf ucx-1.21.0.tar.gz
    (
        cd ucx-1.21.0
        ./configure --prefix="$transport_root/ucx-install" --with-cuda="$CUDA_HOME" \
            --enable-mt
        make -j "$jobs"
        make install
    )
fi

if [[ ! -f nvda_nixl/lib/aarch64-linux-gnu/plugins/libplugin_UCX.so ]]; then
    if [[ ! -d nixl-src ]]; then
        git clone --depth 1 --branch v1.3.2 https://github.com/ai-dynamo/nixl.git nixl-src
    fi
    [[ "$(git -C nixl-src rev-parse HEAD)" == de8115ca97d3f8fb63a4988e9b4d4a038b2e0f72 ]]
    # Meson's declared mirror avoids SourceForge download redirects that stall here.
    if [[ ! -f nixl-src/subprojects/packagecache/asio-1.30.2.tar.gz ]]; then
        mkdir -p nixl-src/subprojects/packagecache
        curl -L --fail --retry 3 --max-time 120 \
            -o nixl-src/subprojects/packagecache/asio-1.30.2.tar.gz \
            https://github.com/mesonbuild/wrapdb/releases/download/asio_1.30.2-2/asio-1.30.2.tar.gz
    fi
    echo '12e7bb4dada8bc1191de9d550a59ee658ce4e645ffc97c911c099ab4e8699d55  nixl-src/subprojects/packagecache/asio-1.30.2.tar.gz' | sha256sum --check
    (
        cd nixl-src
        meson setup build --prefix="$transport_root/nvda_nixl" \
            --libdir=lib/aarch64-linux-gnu \
            -Ducx_path="$transport_root/ucx-install" \
            -Dcudapath_inc="$CUDA_HOME/include" \
            -Dcudapath_lib="$CUDA_HOME/lib64" \
            -Dcudapath_stub="$CUDA_HOME/lib64/stubs" \
            -Dnixl_cuda_arch_list=121 \
            -Denable_plugins=UCX -Dbuild_tests=false -Dbuild_examples=false -Dbuild_docs=false
        meson compile -C build -j "$jobs"
        meson install -C build
    )
fi

export LD_LIBRARY_PATH="$transport_root/nvda_nixl/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
"$transport_root/ucx-install/bin/ucx_info" -v
for transport_library in \
    "$transport_root/ucx-install/lib/ucx/libuct_cuda.so" \
    "$transport_root/nvda_nixl/lib/aarch64-linux-gnu/plugins/libplugin_UCX.so"; do
    link_check="$(ldd -r "$transport_library" 2>&1)"
    echo "$link_check"
    if [[ "$link_check" == *'not found'* || "$link_check" == *'undefined symbol:'* ]]; then
        echo "Unresolved native transport dependency: $transport_library" >&2
        exit 1
    fi
done
echo "Native transport installed at $transport_root"
