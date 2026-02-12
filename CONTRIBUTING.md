# Contributing to SiriusDB

## Building in debug mode

```bash
pixi shell

mkdir build && pushd build

cmake ../duckdb \
  -DCMAKE_BUILD_TYPE=Debug \
  -DDUCKDB_EXTENSION_CONFIGS=../extension_config.cmake \
  -DCMAKE_C_COMPILER=${CONDA_PREFIX}/bin/clang \
  -DCMAKE_CXX_COMPILER=${CONDA_PREFIX}/bin/clang++ \
  -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
  -DCMAKE_CXX_FLAGS_DEBUG="-g -O0" \
  -DCMAKE_CUDA_FLAGS_DEBUG="-g -G -O0" \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache \
  -DBUILD_TESTS=ON

cmake --build . --parallel 12 --verbose

popd
```

