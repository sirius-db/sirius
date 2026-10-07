# Static UCX

This Linux port builds UCX 1.20.1 with TCP, self, POSIX/System V shared
memory, CMA, CUDA copy and CUDA IPC transports. InfiniBand/RoCE, RDMA CM,
GDRCopy and other optional transports are disabled explicitly.

The port builds Release only for Sirius distribution packages. Consumers use
the Release archives in every build configuration.

`find_package(ucx CONFIG REQUIRED)` exposes `ucx::ucx`. Its link interface
retains transport registration constructors and includes the complete static
UCX dependency closure. CUDA's runtime is static; the NVIDIA CUDA driver and
NVML libraries remain host driver dependencies. No UCX shared libraries or
loadable modules are needed.

The temporary patches address this pinned version:

- `static-pic.patch` leaves `UCX_SHARED_LIB` undefined for static archives built
  with `--with-pic`, so UCX uses linked transport registrations instead of loading
  modules that are not installed.
- `static-cudart.patch` selects the static CUDA runtime for this package.
- `library-only.patch` omits tools, bindings, examples, and test applications.

Remove each backport when the pinned version provides the corresponding fix;
the CUDA linkage and library-only build remain packaging choices.

The CUDA toolkit must provide `nvcc`, headers, `libcudart_static.a`, and CUDA
driver/NVML link stubs. Both a standard CUDA toolkit layout and Conda's
`targets/<architecture>-linux` layout are supported. Build tools come from
the Pixi `vcpkg` environment (`autoconf`, `automake`, and `libtool`).

CMake discovers the consumer's CUDA toolkit through `FindCUDAToolkit`.
The pkg-config file records CUDA library names without build-machine paths;
pkg-config consumers must provide their current toolkit's library and driver
stub directories when linking.

## Check the installed package

For an isolated dependency installation:

```bash
pixi run -e vcpkg cmake -S test/cmake/ucx -B build/ucx-static-probe \
  -Ducx_DIR="$PWD/build/exchange-vcpkg-installed/x64-linux/share/ucx" \
  -DCMAKE_BUILD_TYPE=Release
pixi run -e vcpkg cmake --build build/ucx-static-probe
pixi run -e vcpkg ctest --test-dir build/ucx-static-probe --output-on-failure
```

Use the `arm64-linux` triplet path on ARM. The checks reject shared UCX/CUDA
runtime dependencies and build-machine library paths in pkg-config metadata,
and transfer 1 MiB between GPU buffers using TCP and shared memory with CUDA
copy. The two UCX contexts are in one process; these checks do not establish
CUDA IPC behavior between processes.
