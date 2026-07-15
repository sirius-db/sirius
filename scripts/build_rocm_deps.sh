#!/usr/bin/env bash
# =============================================================================
# build_rocm_deps.sh — Build and install hipDF + hipMM from source.
#
# These are the ROCm-DS drop-in equivalents of cuDF and RMM. They export
# the same CMake targets (cudf::cudf, rmm::rmm) and the same C++ namespaces
# (cudf::, rmm::) so Sirius's source and CMakeLists.txt work unchanged.
#
# Requirements:
#   - ROCm 7.2.1+ (hip-clang, hipCUB, rocThrust, rocPRIM)
#   - CMake 3.30+
#   - Ubuntu 24.04+ (or equivalent ROCm Linux)
#   - At least 32 GB free disk space for the build
#
# Usage:
#   ./scripts/build_rocm_deps.sh [--prefix /path/to/install]
#
# Default install prefix: /opt/rocm (system-wide, requires root).
# Use --prefix to install to a user-writable location.
#
# After building, configure Sirius with:
#   cmake -B build/rocm -S . \
#     -DENABLE_ROCM=ON \
#     -DCMAKE_PREFIX_PATH=/path/to/install
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build/rocm_deps"
INSTALL_PREFIX="/opt/rocm"
JOBS=$(nproc 2>/dev/null || echo 8)

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) INSTALL_PREFIX="$2"; shift 2;;
    --jobs)   JOBS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "=== ROCm Dependencies Build ==="
echo "Install prefix: $INSTALL_PREFIX"
echo "Build dir:      $BUILD_DIR"
echo "Jobs:           $JOBS"
echo ""

# Verify ROCm is installed
if [ ! -d "/opt/rocm" ]; then
  echo "ERROR: /opt/rocm not found. Install ROCm first."
  exit 1
fi

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-${ROCM_PATH}/lib/cmake}"

# -----------------------------------------------------------------------------
# Critical: force git to use HTTP/1.1. ROCm DSW pods and some CI environments
# have flaky HTTP/2 connections to GitHub — git clone fails with
# "GnuTLS recv error (-110): The TLS connection was non-properly terminated"
# or "HTTP/2 stream 1 was not closed cleanly". HTTP/1.1 is more robust.
# -----------------------------------------------------------------------------
git config --global http.version HTTP/1.1 2>/dev/null || true
echo "Set git http.version to HTTP/1.1 (fixes flaky GitHub clones)"

# -----------------------------------------------------------------------------
# Critical: use hipcc as BOTH the C and CXX compiler. hipMM's .cpp sources
# are compiled with -x hip --offload-arch=gfx942 flags that only hip-clang
# understands. Plain g++ fails with "unrecognized command-line option
# '--offload-arch=gfx942'". hipcc is a wrapper around clang++ that sets
# the right HIP target.
# -----------------------------------------------------------------------------
HIPCC=$(command -v hipcc || echo "/usr/bin/hipcc")
if [ ! -x "$HIPCC" ]; then
  # Fall back to the ROCm install path
  HIPCC="${ROCM_PATH}/lib/llvm/bin/clang++"
fi
echo "Using compiler: $HIPCC"

# -----------------------------------------------------------------------------
# Critical: set ROCM_AMDGPU_TARGETS to match CMAKE_HIP_ARCHITECTURES.
# rapids_cmake's rapids_hip_set_architectures checks that these match;
# a mismatch is a fatal error ("mismatch between CMAKE_HIP_ARCH_ARCHITECTURES
# and AMDGPU_TARGETS").
# -----------------------------------------------------------------------------
export ROCM_AMDGPU_TARGETS="gfx942"
export GPU_TARGETS="gfx942"
HIP_ARCH="gfx942"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# =============================================================================
# Step 0: Pre-clone all CPM dependencies (avoids flaky network during configure)
# =============================================================================
echo "=== Step 0: Pre-cloning CPM dependencies ==="

DEPS_CACHE="$BUILD_DIR/deps_cache"
mkdir -p "$DEPS_CACHE"

clone_dep() {
  local name="$1" url="$2" branch="$3"
  local dir="$DEPS_CACHE/${name}-src"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "  $name: already cached"
    return 0
  fi
  echo -n "  $name: cloning $branch... "
  for attempt in 1 2 3 4 5; do
    if timeout 120 git clone --depth 1 --branch "$branch" "$url" "$dir" 2>/dev/null; then
      echo "OK"
      return 0
    fi
    echo -n "retry$attempt... "
    rm -rf "$dir"
    sleep 3
  done
  echo "FAILED (will try during configure)"
  return 1
}

# These are the exact branch/tag names from rapids-cmake versions.json.
# Using wrong branch names (e.g. "rocm-0.2.0" instead of "release/rocmds-26.03")
# causes git clone to fail with "remote branch not found".
clone_dep "rapids-cmake" "https://github.com/rapidsai/rapids-cmake.git" "branch-25.10"
clone_dep "cccl" "https://github.com/NVIDIA/cccl.git" "v3.0.3"
clone_dep "libhipcxx" "https://github.com/ROCm/libhipcxx.git" "release/rocmds-26.03"
clone_dep "jitify" "https://github.com/ROCm/jitify.git" "release/rocmds-26.03"
clone_dep "hipcomp" "https://github.com/ROCm/hipcomp-core.git" "release/rocmds-26.03"
clone_dep "spdlog" "https://github.com/gabime/spdlog.git" "v1.14.1"
clone_dep "fmt" "https://github.com/fmtlib/fmt.git" "11.0.2"
clone_dep "rapids-logger" "https://github.com/rapidsai/rapids-logger.git" "branch-0.1.0"
clone_dep "flatbuffers" "https://github.com/google/flatbuffers.git" "v24.3.25"
clone_dep "roaring" "https://github.com/RoaringBitmap/CRoaring.git" "v4.3.11"
clone_dep "dlpack" "https://github.com/dmlc/dlpack.git" "v1.0"
clone_dep "nanoarrow" "https://github.com/apache/arrow-nanoarrow.git" "apache-arrow-nanoarrow-0.6.0"
clone_dep "thread-pool" "https://github.com/bshoshany/thread-pool.git" "v4.1.0"
clone_dep "zstd" "https://github.com/facebook/zstd.git" "v1.5.6"
clone_dep "kvikio" "https://github.com/ROCm/kvikio.git" "branch-25.10"
clone_dep "nvtx" "https://github.com/NVIDIA/NVTX.git" "v3.2.0"
clone_dep "arrow" "https://github.com/apache/arrow.git" "apache-arrow-18.0.0"

# Build FETCHCONTENT_SOURCE_DIR overrides for cmake — these tell CPM/FetchContent
# to use the pre-cloned source dirs instead of fetching from GitHub.
FETCH_ARGS=""
for dir in "$DEPS_CACHE"/*-src; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir" | sed 's/-src$//')
  # CMake FETCHCONTENT_SOURCE_DIR_<name> uses uppercase with underscores
  upper=$(echo "$name" | tr 'a-z-' 'A-Z_')
  FETCH_ARGS="$FETCH_ARGS -DFETCHCONTENT_SOURCE_DIR_${upper}=${dir}"
done
echo "  Pre-cloned deps: $(ls -d "$DEPS_CACHE"/*-src 2>/dev/null | wc -l)"
echo ""

# =============================================================================
# Step 1: Build hipMM (RMM port for HIP)
# =============================================================================
echo "=== Step 1: Building hipMM ==="

HIPMM_DIR="${BUILD_DIR}/hipMM"
HIPMM_BRANCH="release/rocmds-26.03"

if [ ! -d "$HIPMM_DIR" ]; then
  git clone --depth 1 --branch "$HIPMM_BRANCH" \
    https://github.com/ROCm-DS/hipMM.git "$HIPMM_DIR"
fi

cd "$HIPMM_DIR"
mkdir -p cpp/build
cd cpp/build

# Key fixes applied here (discovered on real gfx942/ROCm 7.2.1):
# 1. CMAKE_CXX_COMPILER=hipcc — hipMM's .cpp files need -x hip flags
# 2. ROCM_AMDGPU_TARGETS env — avoids arch mismatch fatal error
# 3. CPM_SOURCE_CACHE + FETCHCONTENT overrides — uses pre-cloned deps
# 4. No separate AMDGPU_TARGETS/GPU_TARGETS cmake vars — only env vars
env ROCM_AMDGPU_TARGETS="$ROCM_AMDGPU_TARGETS" GPU_TARGETS="$GPU_TARGETS" \
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH="$ROCM_PATH" \
  -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
  -DCMAKE_CXX_COMPILER="$HIPCC" \
  -DCMAKE_C_COMPILER="$HIPCC" \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DCPM_SOURCE_CACHE="$DEPS_CACHE" \
  $FETCH_ARGS

env ROCM_AMDGPU_TARGETS="$ROCM_AMDGPU_TARGETS" GPU_TARGETS="$GPU_TARGETS" \
cmake --build . -j"$JOBS"
cmake --install .

echo "=== hipMM installed to $INSTALL_PREFIX ==="
cd "$BUILD_DIR"

# =============================================================================
# Step 2: Build hipDF (cuDF port for HIP)
# =============================================================================
# VERSION GAP: hipDF release/rocmds-26.03 is derived from cuDF 25.10.
# Sirius uses libcudf 26.06 with APIs absent from this branch:
#   - cudf::io::parquet::fetch_footer_to_host (absent)
#   - hybrid_scan_reader::all_column_chunks_byte_ranges (absent — 26.03 has
#     3 separate *_byte_ranges methods instead)
# These APIs must be ported from upstream RAPIDS cuDF branch-26.06 into hipDF
# before Sirius can compile end-to-end. See issue sirius-db/sirius#1158.
echo "=== Step 2: Building hipDF ==="

HIPDF_DIR="${BUILD_DIR}/hipDF"
HIPDF_BRANCH="release/rocmds-26.03"

if [ ! -d "$HIPDF_DIR" ]; then
  git clone --depth 1 --branch "$HIPDF_BRANCH" \
    https://github.com/ROCm-DS/hipDF.git "$HIPDF_DIR"
fi

# Apply cuDF 26.06 API patches (fetch_footer_to_host, all_column_chunks_byte_ranges)
# AND the get_rmm.cmake patch (use find_package instead of CPM re-fetch).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/hipdf_26.06_api_patch.sh" ]; then
  bash "$SCRIPT_DIR/hipdf_26.06_api_patch.sh" "$HIPDF_DIR"
else
  echo "WARNING: hipdf_26.06_api_patch.sh not found — Sirius will fail to compile"
  echo "  without fetch_footer_to_host and all_column_chunks_byte_ranges."
fi

cd "$HIPDF_DIR"
mkdir -p cpp/build
cd cpp/build

# Key fixes applied here (discovered on real gfx942/ROCm 7.2.1):
# 1. CMAKE_CXX_COMPILER=hipcc — hipDF requires hipcc as C/CXX (its CMakeLists
#    checks that the compiler matches .*hipcc$ or .*clang\+\+$)
# 2. ROCM_AMDGPU_TARGETS env — same arch mismatch fix as hipMM
# 3. CPM_USE_LOCAL_PACKAGES=ON — prefer system-installed rocThrust/hipCUB/etc
# 4. CPM_SOURCE_CACHE + FETCHCONTENT overrides — uses pre-cloned deps
# 5. get_rmm.cmake patched to find_package(rmm) — avoids rapids_logger
#    "Unknown CMake command rapids_make_logger" error when CPM re-fetches rmm
env ROCM_AMDGPU_TARGETS="$ROCM_AMDGPU_TARGETS" GPU_TARGETS="$GPU_TARGETS" \
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH="$ROCM_PATH" \
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX/lib/cmake;${ROCM_PATH}/lib/cmake" \
  -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
  -DCMAKE_CXX_COMPILER="$HIPCC" \
  -DCMAKE_C_COMPILER="$HIPCC" \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DCUDF_EXPORT_NVCOMP=OFF \
  -DCPM_SOURCE_CACHE="$DEPS_CACHE" \
  -DCPM_USE_LOCAL_PACKAGES=ON \
  $FETCH_ARGS

env ROCM_AMDGPU_TARGETS="$ROCM_AMDGPU_TARGETS" GPU_TARGETS="$GPU_TARGETS" \
cmake --build . -j"$JOBS"
cmake --install .

echo "=== hipDF installed to $INSTALL_PREFIX ==="
echo ""
echo "=== Done. Now configure Sirius: ==="
echo "  cmake -B build/rocm -S . \\"
echo "    -DENABLE_ROCM=ON \\"
echo "    -DSIRIUS_ENABLE_CUCO=OFF \\"
echo "    -DSIRIUS_ENABLE_CUCASCADE=OFF \\"
echo "    -DSIRIUS_BUILD_S3_TESTS=OFF \\"
echo "    -DSIRIUS_BUILD_TELEMETRY=OFF \\"
echo "    -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX \\"
echo "    -DCMAKE_HIP_ARCHITECTURES=gfx942"
