# NIXL packaging

This port pins NIXL 1.5.0 and links its UCX backend into the static SDK. Sirius's
CMake development build shares its patches and `nixl-targets.cmake`, using
environment-provided shared UCX, Abseil, and CUDA runtime libraries. CMake's
external project downloads, patches, and builds the pinned source during the
build step. This port uses the static UCX overlay port.

The static port builds Release only for Sirius distribution packages. Consumers
use the Release archives in every build configuration.

The temporary patches address this pinned version:

- `native-cpp-only.patch` omits language bindings, device helpers, test utilities,
  external telemetry plugins, and ETCD discovery. Core BUFFER/NOP telemetry
  remains available. Existing Meson options disable tests, examples, and tracing.
- `system-tomlplusplus.patch` uses the installed toml++ headers without requiring
  its optional shared library. It requires the `native_only` option from
  `native-cpp-only.patch`; apply the patches in the listed order.
- `static-sdk.patch` honors static linkage for common utilities and installs
  the UCX archive.

Remove each backport when the pinned version provides the corresponding fix.
Use a fresh CMake build directory after changing the pinned source or patches.
The system CUDA toolkit remains a build prerequisite, as for the other Sirius
GPU ports; CUDA driver libraries are runtime dependencies.

Installed Sirius consumers should use the standalone shared package or the vcpkg
static package described in [Building Sirius](../../docs/building.md). The legacy
installed DuckDB export set does not discover Sirius's dependencies and is not
supported for NIXL-enabled consumers.

## Isolated validation

From the Sirius checkout:

```sh
pixi run -e vcpkg vcpkg/vcpkg install nixl:x64-linux --classic \
  --overlay-ports=vcpkg_ports --overlay-triplets=vcpkg_triplets \
  --x-install-root=build/exchange-vcpkg-installed
pixi run -e vcpkg cmake -S test/cmake/nixl -B build/nixl-static-probe \
  -DCMAKE_PREFIX_PATH="$PWD/build/exchange-vcpkg-installed/x64-linux"
pixi run -e vcpkg cmake --build build/nixl-static-probe
pixi run -e vcpkg ctest --test-dir build/nixl-static-probe --output-on-failure
```

The probe exercises the installed `nixl::nixl` target, GPU memory registration,
a 1 MiB GPU transfer, and a transfer completion notification between two agents.
The two agents run in one process.
CTest checks TCP and shared memory enabled transport with CUDA copy, audits
dynamic dependencies, and builds a shared consumer with the BFD linker, hidden
archive symbols, and unresolved symbols forbidden. The shared memory case also
enables TCP because NIXL requires a control transport with peer failure handling.
