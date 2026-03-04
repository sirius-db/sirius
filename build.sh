#!/bin/bash
# Build Sirius for ClickBench on GH200 (or other CUDA GPU)
# Usage: bash build.sh [clean]
set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXI_ENV="$REPO/.pixi/envs/default"

if [[ ! -d "$PIXI_ENV" ]]; then
    echo "ERROR: pixi env not installed. Run: ~/.pixi/bin/pixi install"
    exit 1
fi

# For cudf 26.02 only: patch CUB header to fix data() ambiguity
CUDF_VER=$(cat "$PIXI_ENV/conda-meta/libcudf-"*.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['version'])" 2>/dev/null || echo "unknown")
echo "cudf version: $CUDF_VER"

CUB_HDR="$PIXI_ENV/include/rapids/cub/block/block_load_to_shared.cuh"
if [[ "$CUDF_VER" == 26.02.* ]] && grep -q "= data(smem_dst)" "$CUB_HDR" 2>/dev/null; then
    echo "Patching cudf 26.02 CUB header (data() ambiguity fix)..."
    python3 -c "
path = '$CUB_HDR'
with open(path) as f: c = f.read()
c = c.replace('    const auto dst_ptr  = data(smem_dst);', '    const auto dst_ptr  = smem_dst.data();')
c = c.replace('    const auto src_ptr  = ::cuda::ptr_rebind<char>(data(gmem_src));', '    const auto src_ptr  = ::cuda::ptr_rebind<char>(gmem_src.data());')
c = c.replace('      return {::cuda::ptr_rebind<T>(data(smem_dst)), size(gmem_src)};', '      return {::cuda::ptr_rebind<T>(smem_dst.data()), size(gmem_src)};')
with open(path, 'w') as f: f.write(c)
print('CUB patch applied.')
"
fi

if [[ "${1}" == "clean" ]]; then
    rm -rf "$REPO/build"
fi
mkdir -p "$REPO/build/release"

export PATH="$PIXI_ENV/bin:$PATH"
export LIBCUDF_ENV_PREFIX="$PIXI_ENV"

# Detect GPU compute capability dynamically (e.g. "7.5" -> "75")
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "Detected GPU compute capability: $CUDA_ARCH"

"$PIXI_ENV/bin/cmake" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$PIXI_ENV/bin/ninja" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$PIXI_ENV" \
    -DCMAKE_C_COMPILER="$PIXI_ENV/bin/gcc" \
    -DCMAKE_CXX_COMPILER="$PIXI_ENV/bin/g++" \
    -DCMAKE_CUDA_COMPILER="$PIXI_ENV/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$PIXI_ENV/bin/g++" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}-real" \
    -DEXTENSION_STATIC_BUILD=1 \
    -DDUCKDB_EXTENSION_CONFIGS="$REPO/extension_config.cmake" \
    -DDUCKDB_EXPLICIT_PLATFORM=linux_$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/') \
    -S "$REPO/duckdb/" \
    -B "$REPO/build/release"

"$PIXI_ENV/bin/cmake" --build "$REPO/build/release" -j$(nproc)
echo "Build complete: $REPO/build/release/duckdb"
