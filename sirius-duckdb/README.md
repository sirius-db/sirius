# sirius-duckdb

DuckDB extension wrapper for an installed Sirius CMake package. The wrapper stays
here until it can move to `sirius-db/sirius-duckdb`. Its Makefile uses DuckDB's
normal extension build infrastructure.

Use matching DuckDB revisions and compatible C++ toolchains for Sirius and the
wrapper. The engine and wrapper gitlinks record the supported revision.

## Shared development build

Shared linkage is the default. From the repository root, use the existing
Conda/Pixi environment. The root Makefile builds Sirius and its tests, installs
Sirius into `build/release/install`, then builds the wrapper against it:

```sh
git submodule update --init duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
pixi run make
```

Run DuckDB with the environment's shared libraries on its runtime search path:

```sh
pixi run bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"; exec "$@"' -- \
  sirius-duckdb/build/release/duckdb -unsigned \
  -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';"
```

After changing Sirius, rerun `pixi run make`, then restart DuckDB. Use
`pixi run make release` to build only Sirius and its tests. No Conda package build
is needed.

## Static distribution build

The Makefile builds and installs Sirius from the local checkout before building
the extension. Its vcpkg manifest supplies only third-party static dependencies.
The final extension link bundles those libraries; the
post-link check rejects unexpected runtime dependencies. NVIDIA driver libraries
and standard Linux libraries remain external.

```sh
git submodule update --init duckdb substrait cucascade sirius-duckdb/vcpkg sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools
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

Both linkage modes use the local engine sources, including uncommitted changes.
The wrapper Makefile installs its engine build under
`build/sirius-<linkage>/<configuration>/install`. Rerunning Make rebuilds changed
sources; vcpkg's third-party dependency caches remain reusable. Distribution CI
uses the same flow, with `configure_ci` initializing the engine's submodules.

To consume an existing Sirius installation instead of building the local engine,
set `SIRIUS_INSTALL_DIR` to its absolute prefix. The root Makefile uses this to
reuse its `build/release/install` package. The wrapper CMake project itself only
uses `find_package(sirius)` and never builds the engine.

Both builds produce
`build/release/extension/sirius/sirius.duckdb_extension` under this directory.
Run the wrapper's SQL tests from the root with the same runtime search path:

```sh
pixi run bash -c 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"; make -C sirius-duckdb test_release'
```
