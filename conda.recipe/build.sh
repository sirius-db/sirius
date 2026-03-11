#!/bin/bash
set -euxo pipefail

# Select CUDA architectures based on CUDA version
if [[ "${cuda_version}" == "13"* ]]; then
  # Turing through Blackwell
  CUDA_ARCHS="75-real;80-real;86-real;90a-real;100f-real;120a-real;120"
else
  # Turing through Hopper (CUDA 12)
  CUDA_ARCHS="75-real;80-real;86-real;90a-real"
fi

# Create the CMakePresets.json symlink (normally done by pixi_activate.sh)
ln -sf ../cmake/CMakePresets.json duckdb/CMakePresets.json
rm -f duckdb/CMakeUserPresets.json

# Configure — drive cmake directly from the duckdb/ subdirectory
cmake -G Ninja \
  -S duckdb \
  -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXTENSION_STATIC_BUILD=ON \
  -DDUCKDB_EXTENSION_CONFIGS="$SRC_DIR/extension_config.cmake" \
  -DEXPORT_DYNAMIC_SYMBOLS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"

# Build only the loadable extension target
cmake --build build/release --target sirius_loadable_extension -j${CPU_COUNT}

# Install the extension
mkdir -p $PREFIX/lib
cp build/release/extension/sirius/sirius.duckdb_extension $PREFIX/lib/

# Install activation scripts
mkdir -p $PREFIX/etc/conda/activate.d
mkdir -p $PREFIX/etc/conda/deactivate.d

cat > $PREFIX/etc/conda/activate.d/sirius-duckdb.sh << 'ACTIVATE'
export SIRIUS_EXTENSION="${CONDA_PREFIX}/lib/sirius.duckdb_extension"
ACTIVATE

cat > $PREFIX/etc/conda/deactivate.d/sirius-duckdb.sh << 'DEACTIVATE'
unset SIRIUS_EXTENSION
DEACTIVATE
