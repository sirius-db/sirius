#!/usr/bin/env bash
# =============================================================================
# build_rocm_deps.sh — Build and install hipDF + hipMM from source.
#
# These are the ROCm-DS drop-in equivalents of cuDF and RMM. They export
# the same CMake targets (cudf::cudf, rmm::rmm) and the same C++ namespaces
# (cudf::, rmm::) so Sirius's source and CMakeLists.txt work unchanged.
#
# Requirements:
#   - ROCm 7.2.3+ (hip-clang, hipCUB, rocThrust, rocPRIM)
#   - GCC 14+
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

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

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

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH="$ROCM_PATH" \
  -DCMAKE_HIP_ARCHITECTURES="gfx942" \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF

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
# These APIs are absent from hipDF 25.10 but Sirius needs them.
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

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH="$ROCM_PATH" \
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX/lib/cmake;${ROCM_PATH}/lib/cmake" \
  -DCMAKE_HIP_ARCHITECTURES="gfx942" \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DCUDF_EXPORT_NVCOMP=OFF

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
