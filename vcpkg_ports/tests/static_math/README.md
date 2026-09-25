# Static CUDA math link probe

The cuBLAS, cuSOLVER, cuSPARSE, and culibos overlay ports install NVIDIA static
archives and expose explicit CMake targets. RAFT and cuVS patches select these
targets, with `CUDA_STATIC_MATH_LIBRARIES=ON` also selecting static cuRAND from
the CUDA toolkit. A matching CUDA toolkit is still required to compile Sirius.

`VCPKG_CUDA_VERSION=12` selects the CUDA 12.9.1 redistributables; `13` selects
CUDA 13.3.0. Each port's `redistrib.json` pins component versions and archive
SHA512 hashes for Linux x64 and arm64 (SBSA). The port version denotes the CUDA
13 component, following the existing nvjitlink port convention. CUDA 12 obtains
culibos from the cudart archive; CUDA 13 has a separate culibos archive.

When updating pins, verify archives against NVIDIA's release manifests at
`https://developer.download.nvidia.com/compute/cuda/redist/redistrib_<release>.json`,
then calculate SHA512 hashes for vcpkg. Keep the component versions aligned with
the selected toolkit and the existing nvjitlink port.

Configure against an installed Sirius vcpkg prefix and CUDA toolkit:

```sh
cmake -S vcpkg_ports/tests/static_math -B build/static-math-probe \
  -DCMAKE_PREFIX_PATH=/path/to/vcpkg_installed/x64-linux-release \
  -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build build/static-math-probe
ctest --test-dir build/static-math-probe --output-on-failure
```

This links real cuBLAS, cuSOLVER, cuSPARSE, and nvJitLink entry points into a
shared library with unresolved symbols forbidden, then checks its ELF dynamic
dependencies. It does not execute GPU calls and needs no GPU or driver.
Run it against both CUDA 12 and CUDA 13 prefixes. Full Sirius loading and GPU
execution still require integration tests on a GPU runner.
