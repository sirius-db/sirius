# Building Sirius

## Separate DuckDB extension

[`sirius-duckdb/`](../sirius-duckdb/README.md) contains the independent extension
setup, with its own DuckDB checkout and Makefile. It consumes
an installed shared or static Sirius CMake package and runs GPU SQL tests
through the normal DuckDB extension test target. The root CMake project builds
the Sirius libraries and C++ tests.

For local development, `pixi run make` builds Sirius and its C++ tests, installs
Sirius under `build/release/install`, and builds the shared DuckDB extension in
`sirius-duckdb/`. Use `pixi run make release` for just Sirius and its C++ tests,
or invoke standalone CMake directly:

```bash
pixi run cmake --preset release
pixi run cmake --build --preset release --target sirius_library sirius_unittest
```

## Shared implementation objects

The internal `sirius_objects` CMake target compiles the common C++ and CUDA
implementation once per build configuration. Its objects form `sirius_core`, an
internal archive, and feed the shared and static Sirius libraries. CUDA device
linking takes place on concrete library targets, not on the object target.

Compile options, dependency headers, PIC, and visibility belong to the object
target. Final library targets also declare their link dependencies: consuming
`$<TARGET_OBJECTS:sirius_objects>` alone does not propagate usage requirements.
Objects are shared only within compatible compiler and dependency configurations.

DuckDB entrypoints are separate from the engine's registration API. NVTX setup is
part of the shared implementation objects. At runtime the ELF loader's link map
identifies whether the code is embedded in the main executable (including PIE)
or a shared library. Libraries publish their own image path; executables use the
private injection sentinel and exported initializer. No filename convention or
output-specific compilation is required.

## NVTX linkage tests

These tests need a C++ compiler but no GPU, CUDA toolkit, or DuckDB build. They
check environment precedence, explicit injector configuration, discovery in PIE
and non-PIE executables and shared libraries, and forwarding to the embedded
initializer.

```bash
pixi run cmake -S test/cmake/nvtx_injection -B build/nvtx-test -G Ninja
pixi run cmake --build build/nvtx-test
pixi run ctest --test-dir build/nvtx-test --output-on-failure
```

## DuckDB dependency

`cmake/sirius-duckdb-provider.cmake` owns the temporary source dependency. Its
`sirius::duckdb_dependency` target carries DuckDB headers, compile definitions,
and the core and Parquet libraries. `SIRIUS_DUCKDB_SOURCE_DIR` selects the source
tree; use the revision pinned by this repository. This is a build-only contract,
not an installed Sirius target or a stable DuckDB ABI.

The intended replacement is `find_package(duckdb CONFIG REQUIRED)` using a conda
package with the required headers and libraries. Decoupling the library's and
extension's DuckDB versions requires a separate API/ABI change.

## Installed CMake package

Install the `sirius_library` component and consume `sirius::sirius` with
`find_package(sirius CONFIG REQUIRED)`. Its public headers do not require CUDA or
DuckDB headers. The shared library records its runtime dependencies.

```bash
pixi run cmake --install build/release --prefix "$PWD/build/stage" --component sirius_library
mv build/stage build/relocated
pixi run cmake -S test/cmake/installed_consumer -B build/consumer \
  -DCMAKE_PREFIX_PATH="$PWD/build/relocated"
pixi run cmake --build build/consumer
```

Consumers are responsible for DuckDB and C++ runtime ABI compatibility.

## Static library

`SIRIUS_BUILD_STATIC=ON` installs a normal static Sirius library and its
source-built support libraries. Third-party dependencies remain separate packages;
CMake's exported targets carry their link requirements. No archives are merged.

The wrapper owns the vcpkg dependency setup. Build and install the local static
library without building the extension:

```bash
git submodule update --init duckdb substrait cucascade sirius-duckdb/vcpkg
VCPKG_CUDA_ARCHITECTURES=100 pixi run -e vcpkg sirius-duckdb/vcpkg/vcpkg install \
  --x-manifest-root=sirius-duckdb --x-feature=static \
  --x-install-root=build/static-install
```

The installed package and dependencies are in `build/static-install/<triplet>`.
The Sirius overlay port builds this checkout, including uncommitted changes.
Replace `100` with the GPU architecture list you need;
`VCPKG_CUDA_ARCHITECTURES` applies to Sirius and its dependencies.

Consumers use `find_package(sirius CONFIG REQUIRED COMPONENTS static)` and link
`sirius::sirius_static`, with the dependency packages available through the vcpkg
toolchain. The final consumer chooses its compiler-runtime linkage.

For extension development, use the shared library with Conda/Pixi dependencies.
For distribution, vcpkg builds the local static package and the final extension
link bundles its dependencies. See the
[wrapper instructions](../sirius-duckdb/README.md) for both paths.

Static consumers select their own compiler runtime linkage. The distribution
extension uses `-static-libgcc -static-libstdc++` to bundle those runtimes.

`make test` builds the shared wrapper and sets `SIRIUS_EXTENSION_PATH` for
extension-loading checks. Dynamic scan checks run in the DuckDB-only
`sirius_extension_host` executable; engine tests use static Sirius registration.
When running `scripts/run_unit_tests.py` or the test
binary directly, set this variable to the wrapper's absolute path. CI supplies it.
