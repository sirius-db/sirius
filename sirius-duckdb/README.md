# sirius-duckdb

DuckDB extension wrapper for an installed Sirius CMake package. The wrapper stays
here until it can move to `sirius-db/sirius-duckdb`. Its Makefile uses DuckDB's
normal extension build infrastructure.

Use matching DuckDB revisions and compatible C++ toolchains for Sirius and the
wrapper. The engine and wrapper gitlinks record the supported revision.

## Shared development build

Shared linkage is the default. From the repository root, use the existing
Conda/Pixi environment to build Sirius and install it into a local prefix:

```sh
git submodule update --init duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
pixi run cmake --preset release -DSIRIUS_BUILD_TESTS=OFF
pixi run cmake --build build/release --target sirius_library
pixi run cmake --install build/release --component sirius_library --prefix "$PWD/build/dev-install"
pixi run make -C sirius-duckdb release EXT_FLAGS="-DCMAKE_PREFIX_PATH=$PWD/build/dev-install"
```

Run DuckDB with the environment's shared libraries on its runtime search path:

```sh
pixi run bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"; exec "$@"' -- \
  sirius-duckdb/build/release/duckdb -unsigned \
  -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';"
```

After changing Sirius,
rebuild and reinstall the shared library, then restart DuckDB. Rebuild the wrapper
when its code or the library interface changes. No Conda package build is needed.

## Static distribution build

The wrapper's vcpkg manifest has a `static` feature that installs Sirius and its
static dependencies. The final extension link bundles those libraries; the
post-link check rejects unexpected runtime dependencies. NVIDIA driver libraries
and standard Linux libraries remain external.

```sh
git submodule update --init sirius-duckdb/vcpkg sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
pixi run -e vcpkg make -C sirius-duckdb release \
  SIRIUS_DUCKDB_LINKAGE=static \
  VCPKG_TOOLCHAIN_PATH="$PWD/sirius-duckdb/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

Use `-e vcpkg-cuda12` for CUDA 12; `vcpkg` selects CUDA 13. The overlay triplets
support amd64 and arm64. Set `VCPKG_CUDA_ARCHITECTURES` before building to limit
the engine's GPU architectures; it participates in the binary cache key.
The Makefile exports this from Pixi's `CUDAARCHS`. Direct vcpkg users must set
`VCPKG_CUDA_ARCHITECTURES` explicitly.
Use separate build directories when switching toolchains or linkage modes.

The Sirius port pins its source and vendored dependency revisions. Update these
pins together with the engine's submodules. The `configure_ci` hook updates the
Sirius source revision to the current commit when building in this repository
on GitHub Actions. Only Sirius's binary cache key changes;
dependency caches remain reusable. The wrapper owns the vcpkg manifest, ports, and triplets.

Both builds produce
`build/release/extension/sirius/sirius.duckdb_extension` under this directory.
Run the wrapper's SQL tests from the root with the same runtime search path:

```sh
pixi run bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"; make -C sirius-duckdb test_release'
```
