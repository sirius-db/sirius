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

## CUDA-selecting core extensions

The core-extension layout has three artifacts:

- `sirius`: a CUDA-free loader built from `loader/`, with no Sirius package or CUDA toolkit dependency.
- `sirius_cuda12`: the backend built with CUDA 12.
- `sirius_cuda13`: the backend built with CUDA 13.

`INSTALL sirius; LOAD sirius;` opens `libcuda.so.1` and queries `cuDriverGetVersion`.
Driver support below CUDA 12, a missing driver, or an unsuccessful version query
produces an error. Versions 12000–12999 select `sirius_cuda12`; 13000 and newer
select `sirius_cuda13`. The loader uses DuckDB's `AutoLoadExtension`, including
its configured core repository and auto-install policy. Backend load failures
retain their message inside DuckDB's autoload error; the loader does not retry
with the other CUDA version.

These are Linux glibc amd64/arm64 builds. The existing Makefile and distribution
workflows still build the direct `sirius` backend for development and existing
consumers. Use a separate build directory for each core configuration:

| DuckDB extension config | Artifacts |
| --- | --- |
| `config/loader.cmake` | CUDA-free `sirius` only |
| `config/cuda12.cmake` | `sirius_cuda12` |
| `config/cuda13.cmake` | CUDA-free `sirius` and `sirius_cuda13` |

Pass the absolute config path through `DUCKDB_EXTENSION_CONFIGS`. Backend configs
use an installed static Sirius package by default; build that package with the
matching CUDA toolkit and configure its vcpkg dependencies as described above.
Use `-DEXPORT_DYNAMIC_SYMBOLS=ON` for backend scan verification.
The backend selects BFD locally; the host and other extensions keep DuckDB's
linker settings.
For shared development testing, override `-DSIRIUS_DUCKDB_LINKAGE=shared` and
pass `-Dsirius_DIR=/path/to/install/lib/cmake/sirius`.
The loader-only config needs no engine submodules. Upstream DuckDB's extension
metadata and CI dependency provisioning remain separate integration work.

The loader tests build fake drivers and backend extensions, so they need no GPU:

```sh
pixi run cmake -S sirius-duckdb/duckdb -B build/loader -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_SHELL=OFF \
  -DEXTENSION_STATIC_BUILD=OFF -DDUCKDB_LINKER=bfd \
  -DDUCKDB_EXTENSION_CONFIGS="$PWD/sirius-duckdb/config/loader.cmake" \
  -DSIRIUS_LOADER_TESTS=ON
pixi run cmake --build build/loader \
  --target sirius_loadable_extension sirius_loader_test_fixtures
pixi run python sirius-duckdb/loader/test/test_loader.py \
  --duckdb sirius-duckdb/build/release/duckdb --build-dir build/loader
```

The test host must use the matching DuckDB revision and export its C++ API;
`pixi run make` produces that host. Tests cover version boundaries, driver and
backend failures, repeated loading, and automatic installation from a local
repository. Production loaders use DuckDB's normal extension linkage defaults.
