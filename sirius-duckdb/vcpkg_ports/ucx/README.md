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
NIXL's Meson build uses upstream pkg-config metadata to discover UCX headers
and core libraries. Sirius's final static link uses `ucx::ucx` to include the
CUDA transports and their registration constructors.
