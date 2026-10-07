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
supported for installed consumers.
