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

Start DuckDB with Sirius loaded:

```sh
pixi run duckdb
```

This task loads the unsigned local extension. CMake and the Pixi toolchain record
the local library search paths in the binaries. The equivalent direct invocation is:

```sh
pixi run sirius-duckdb/build/release/duckdb -unsigned \
  -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension';"
```

After changing Sirius, rerun `pixi run make`, then restart DuckDB. Use
`pixi run make release` to build only Sirius and its tests. No Conda package build
is needed.

## Static distribution build

The local vcpkg overlay port builds and installs Sirius from this checkout,
alongside its static dependencies. The extension uses that installed package.
The final extension link bundles those libraries; the
post-link check rejects unexpected runtime dependencies. NVIDIA driver libraries
and standard Linux libraries remain external.

Hosts that statically link their C++ runtime must export its exception-handling
symbols. The supplied DuckDB build enables this with `EXPORT_DYNAMIC_SYMBOLS=ON`.

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
The static port hashes source contents and filenames, including the engine's
submodules and untracked, nonignored source files, so source edits invalidate
Sirius's package without invalidating third-party dependency caches.
Distribution CI uses the same port, with
`configure_ci` initializing the engine's submodules.

For shared builds, first build/install Sirius with the root Makefile or CMake,
then pass its prefix as `SIRIUS_INSTALL_DIR` (or use CMake's normal package search
path). The root Makefile does this automatically. The wrapper CMake project only
uses `find_package(sirius)` and never builds the engine itself.

Both builds produce
`build/release/extension/sirius/sirius.duckdb_extension` under this directory.
Run the wrapper's SQL tests from the root:

```sh
pixi run make -C sirius-duckdb test_release
```
