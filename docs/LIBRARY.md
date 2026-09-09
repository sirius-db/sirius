# Building libsirius

`libsirius.so` exposes the existing C++ embedding API in `sirius_ffi.hpp`.
DuckDB remains an internal dependency for planning and catalog support.
Consumers need compatible C++ tooling and the library's runtime dependencies,
but do not need DuckDB or CUDA headers to include the public header.

Build just the shared library using the normal Pixi environment:

```bash
pixi run make release MAIN_BUILD_TARGETS=sirius_shared TEST_BUILD_TARGET=
pixi run cmake --install build/release --prefix "$PWD/build/install" --component SiriusLibrary
```

The normal `make` build also includes `sirius_shared`. To disable it, configure
with `SIRIUS_BUILD_SHARED_LIBRARY=OFF` and omit `sirius_shared` from
`MAIN_BUILD_TARGETS` when invoking Make.

The install component provides the versioned shared library, public header,
and CMake package. Downstream CMake projects can use:

```cmake
find_package(Sirius CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE Sirius::sirius)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix. The installed library uses
`$ORIGIN` as its runtime search path; external dependencies must be available
alongside it or through the runtime loader's search paths.

Validate the installed package with the independent consumer (no GPU execution):

```bash
pixi run cmake -S test/library-consumer -B build/library-consumer -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
pixi run cmake --build build/library-consumer
pixi run ctest --test-dir build/library-consumer --output-on-failure
```
