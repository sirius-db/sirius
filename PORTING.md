# Sirius ROCm/HIP Port — Engineering Reference

## 1. Architecture Overview

Sirius is a GPU-native SQL engine that runs as a DuckDB extension, routing
SQL operators to the GPU and falling back to DuckDB's CPU execution for
unsupported operations. The upstream codebase targets NVIDIA CUDA exclusively,
built on cuDF (DataFrames), RMM (memory manager), cuCollections (GPU hash
maps/Bloom filters), and cuCascade (out-of-core memory reservation/repository).

This port adds an opt-in AMD ROCm/HIP backend (`ENABLE_ROCM=ON`) that uses
the ROCm-DS drop-in equivalents hipDF and hipMM, plus compatibility shims
that eliminate per-file HIPIFY. The design goal is **zero source-file edits**
for the 57 files containing `cuda*` runtime calls — achieved via a
`cuda_runtime.h` compatibility shim that macro-aliases `cuda*` to `hip*`.

### Component map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CMakeLists.txt                               │
│  ENABLE_ROCM → CXX HIP, gfx90a/942/950, .cu→HIP ext mapping         │
│  SIRIUS_ENABLE_CUCO (OFF on ROCm)    SIRIUS_ENABLE_CUCASCADE (OFF)  │
│  SIRIUS_BUILD_TELEMETRY (gated)      ROCTX find_library             │
├─────────────────────────────────────────────────────────────────────┤
│  cuda2rocm shims (via FetchContent)  ← BEFORE on include path, shadows NVIDIA hdrs  │
│  ├── cuda_runtime.h      cuda*→hip* macro aliases (68 macros + 6 type aliases) │
│  ├── cuda_runtime_api.h   redirect                                   │
│  ├── cuda.h               redirect                                   │
│  ├── cub/cub.cuh          #include hipcub + namespace cub=hipcub     │
│  │   └── cub::detail::warp_threads = 64 (AMD wavefront)              │
│  ├── cub/{config,util_arch,util_ptx,warp_scan,thread_store,...}.cuh  │
│  ├── cucascade/           32-header stub (compile-only, throws RT)   │
│  │   ├── error.hpp        CUCASCADE_CUDA_TRY macro                   │
│  │   ├── memory/          Tier, memory_space, reservation, OOM, ...  │
│  │   ├── data/            data_batch, repository, idata_representation│
│  │   ├── cudf/            gpu/host_data_representation               │
│  │   └── cuda/            cuda_event stub                            │
│  └── nvtx3/nvtx3.hpp      scoped_range → roctxRangeStartA/Stop       │
├─────────────────────────────────────────────────────────────────────┤
│  Source fixes:                                                      │
│  src/include/cuda/scan/detail/fsst.cuh:132                          │
│    __shfl_xor_sync mask cast to unsigned long long (HIP=64-bit)      │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Backend Selection

**Entry point:** `ENABLE_ROCM` CMake option (default OFF).

| Property | NVIDIA CUDA | AMD ROCm |
|---|---|---|
| Language | `CXX CUDA` | `CXX HIP` |
| Source extension | `.cu` → CUDA (built-in) | `.cu` → HIP (via `CMAKE_HIP_SOURCE_FILE_EXTENSIONS`) |
| Architecture | `CMAKE_CUDA_ARCHITECTURES` (native) | `CMAKE_HIP_ARCHITECTURES` (gfx90a;gfx942;gfx950) |
| GPU standard | `CUDA_STANDARD 20` | `HIP_STANDARD 20` |
| Device symbols | `CUDA_RESOLVE_DEVICE_SYMBOLS ON` | `HIP_RESOLVE_DEVICE_SYMBOLS ON` |
| Extended lambda | `--expt-extended-lambda` (nvcc) | not needed (hip-clang default) |

`CMAKE_HIP_SOURCE_FILE_EXTENSIONS` is set after `project()` but before the
first HIP source is attached to a target. CMake re-reads the variable at
source-add time (not project-time), so `.cu` files resolve to HIP.

## 3. Dependency Wiring

### 3.1 cuDF → hipDF

hipDF preserves `namespace cudf`, exports `cudf::cudf` target, and uses
`project(CUDF LANGUAGES C CXX HIP)`. `find_package(cudf REQUIRED CONFIG)`
and the `cudf::cudf` link line are identical across backends. No source
file using `cudf::` needs editing.

### 3.2 RMM → hipMM

hipMM preserves `namespace rmm`, `rmm/` includes, exports `rmm::rmm` target.
The `rmm::rmm` link line is identical across backends.

### 3.3 thrust/cub → rocThrust/hipCUB

rocThrust provides `thrust/` natively. hipCUB provides `hipcub/` — NOT `cub/`.
The `cub/cub.cuh` shim redirects to `<hipcub/hipcub.hpp>` and creates
`namespace cub = hipcub;` so all `cub::` usage resolves.

### 3.4 cuco → gated OFF (no ROCm port)

`SIRIUS_ENABLE_CUCO` defaults OFF on ROCm. The 2 `.cu` files using
`cuco::bloom_filter` / `cuco::static_set` are excluded from `CUDA_SOURCES`.
The PIMPL'd header (`sirius_dynamic_filter.hpp`) still compiles.

### 3.5 cuCascade → stub (cuda2rocm cucascade-rocm/, via FetchContent)

`SIRIUS_ENABLE_CUCASCADE` defaults OFF on ROCm. Instead of the real cuCascade
submodule, 32 stub headers provide ~70 types so 63 `.cpp` files compile.
Stub methods throw `std::runtime_error("cuCascade stub")` at runtime →
graceful degradation to DuckDB CPU execution.

## 4. Compatibility Shim Design

### 4.1 cuda_runtime.h

`#include <hip/hip_runtime.h>` then `#define cuda* hip*` for 68 macros + 6 type aliases
Sirius and its dependencies (RMM, cuDF, cuCascade) use. Key mappings:

| Category | CUDA | HIP | Notes |
|---|---|---|---|
| Types | `cudaError_t` | `hipError_t` | `using` alias |
| | `cudaDeviceProp` | `hipDeviceProp_t` | name mismatch, `using` alias |
| | `cudaEvent_t` | `hipEvent_t` | direct |
| | `cudaStream_t` | `hipStream_t` | direct |
| Runtime | `cudaMalloc` | `hipMalloc` | `#define` |
| | `cudaMemcpy` | `hipMemcpy` | `#define` |
| | `cudaDeviceSynchronize` | `hipDeviceSynchronize` | `#define` |
| Enums | `cudaSuccess` | `hipSuccess` | `#define` |
| | `cudaDevAttrL2CacheSize` | `hipDeviceAttributeL2CacheSize` | prefix mismatch |
| Version | `CUDART_VERSION` | `0` | routes `#if >= 12080` to serial fallback |
| Profiler | `cudaProfilerStart` | `hipProfilerStart` | `#define` |

`CUDART_VERSION = 0` is critical: it routes the `#if CUDART_VERSION >= 12080`
guards (which gate `cudaMemcpyBatchAsync`) to the serial `cudaMemcpyAsync`
fallback. HIP 7.2.1 has `hipMemcpyBatchAsync` but the struct layout differs;
the serial fallback is simpler and correct.

### 4.2 cub/ Redirect Headers

hipCUB uses `hipcub::` namespace (not `cub::`). The `cub/cub.cuh` shim:
1. `#include <hipcub/hipcub.hpp>` — pulls in all hipCUB headers
2. `namespace cub = hipcub;` — aliases the namespace
3. Defines `cub::detail::warp_threads = 64` — CUB internal constant
   (= 32 on NVIDIA, not exposed by hipCUB). AMD wavefront = 64 on CDNA.

Remaining 6 headers (`config.cuh`, `util_arch.cuh`, `util_ptx.cuh`,
`warp/warp_scan.cuh`, `thread/thread_store.cuh`, `device/device_for.cuh`)
redirect to `cub/cub.cuh`.

**Note:** hipCUB's `ShuffleUp`/`ShuffleIndex`/`ShuffleDown` ignore the
`member_mask` parameter (rocPRIM doesn't support masked shuffles). Sirius's
`FULL_MASK` is passed but silently ignored — functionally correct for
unmasked warps.

### 4.3 cuCascade Stub (32 headers)

Provides ~70 types across 32 headers in `cuda2rocm/cucascade-rocm/include/cucascade/` (via FetchContent).
Design principles:

- **5 real virtual base classes** (Sirius subclasses them): `memory_reservation_manager`,
  `oom_handling_policy`, `idata_batch_probe`, `idata_representation`,
  `reservation_aware_resource_adaptor`. These have real vtables so Sirius's
  subclasses link cleanly.
- **Template methods**: inline, throw `std::runtime_error` at runtime.
- **`cucascade_out_of_memory`**: derives from `rmm::out_of_memory` (hipMM
  provides this), with `error_kind`/`requested_bytes`/`global_usage`/`pool_handle`
  members. Sirius catches `rmm::out_of_memory&` then `dynamic_cast` to this type.
- **`CUCASCADE_CUDA_TRY` macro**: wraps `cudaError_t` (from cuda_runtime.h shim).
- **`data_repository`/`data_repository_manager`**: fully inline (nearly verbatim
  from real cuCascade — they only manipulate `std::vector<shared_ptr<data_batch>>`).
- **`register_builtin_converters`**: inline no-op.

### 4.4 nvtx3 → roctx Shim

`nvtx3::scoped_range` (the only nvtx3 type Sirius uses, 47 call sites) maps
to `roctxRangeStartA`/`roctxRangeStop`. The shim is at
`cuda2rocm/cuda-compat-shims/include/nvtx3/nvtx3.hpp` (via FetchContent). ROCm's roctx header is at
`/opt/rocm/include/roctracer/roctx.h` (discovered via `find_path`).

### 4.5 Source Fix: __shfl_xor_sync

`fsst.cuh:132` calls `__shfl_xor_sync(FULL_MASK, ...)` where `FULL_MASK` is
`unsigned int` (32-bit). HIP's `__shfl_xor_sync` requires a 64-bit mask
(`static_assert: The mask must be a 64-bit integer`). CUDA accepts both.

Fix: cast `FULL_MASK` to `unsigned long long` at the call site. The
`cub::ShuffleUp`/`ShuffleIndex` calls in the same file take `unsigned int`
and are unaffected (hipCUB ignores the mask).

## 5. Build Target Topology

| Target | Built when | Shims | ROCTX | cuCascade | Telemetry |
|---|---|---|---|---|---|
| `sirius_extension` | always | BEFORE PRIVATE | if `ENABLE_ROCM` | if `SIRIUS_ENABLE_CUCASCADE` | if `SIRIUS_BUILD_TELEMETRY` |
| `sirius_loadable_extension` | always | BEFORE PRIVATE | if `ENABLE_ROCM` | if `SIRIUS_ENABLE_CUCASCADE` | if `SIRIUS_BUILD_TELEMETRY` |
| `sirius_unittest` | always | BEFORE PRIVATE | if `ENABLE_ROCM` | via `sirius_extension` | via `sirius_extension` |
| `parquet_benchmark` | always | BEFORE PRIVATE | if `ENABLE_ROCM` | if `SIRIUS_ENABLE_CUCASCADE` | — |
| `telemetry_bridge` | if `SIRIUS_BUILD_TELEMETRY` | — | — | — | (is the telemetry) |

The shim include path uses `BEFORE PRIVATE` so it precedes system includes —
load-bearing: without `BEFORE`, the compiler may find a stray system
`cuda_runtime.h` (e.g. from Triton's NVIDIA backend) ahead of the shim.

## 6. Build Configuration

### NVIDIA CUDA (default — unchanged)

```bash
pixi run make
```

### AMD ROCm

```bash
cmake -B build/rocm -S . \
  -DENABLE_ROCM=ON \
  -DSIRIUS_ENABLE_CUCO=OFF \
  -DSIRIUS_ENABLE_CUCASCADE=OFF \
  -DSIRIUS_BUILD_S3_TESTS=OFF \
  -DSIRIUS_BUILD_TELEMETRY=OFF \
  -DCMAKE_HIP_ARCHITECTURES="gfx942"
cmake --build build/rocm
```

## 7. Remaining Work

1. **hipDF/hipMM installation**: `scripts/build_rocm_deps.sh` builds and
   installs both from source. See §8 for the real-build fixes discovered on
   gfx942/ROCm 7.2.1 hardware.
2. **Runtime validation**: The stub makes code compile; runtime behavior needs
   testing on a host with hipDF installed. Stub methods throw → CPU fallback.
3. **cuco port or replacement**: Needed for Bloom/in-list dynamic filters.
4. **NVML → rocm-smi/hwloc**: For multi-GPU topology.

## 8. Real-Build Fixes (discovered on gfx942 / ROCm 7.2.1)

These fixes were discovered by actually building hipDF + hipMM on a real
AMD MI300 (gfx942, ROCm 7.2.1, hipcc 7.2.53211). They are incorporated into
`scripts/build_rocm_deps.sh` and `scripts/hipdf_26.06_api_patch.sh`.

### 8.1 Compiler: hipcc as C/CXX (not g++)

hipMM's `.cpp` sources (e.g. `rmm/src/cuda_device.cpp`) are compiled with
`-x hip --offload-arch=gfx942` flags. Plain `g++` fails with:
```
c++: error: unrecognized command-line option '--offload-arch=gfx942'
```
**Fix:** Set `CMAKE_CXX_COMPILER=hipcc` and `CMAKE_C_COMPILER=hipcc`.
hipcc is a wrapper around clang++ that understands HIP flags.

### 8.2 Architecture: ROCM_AMDGPU_TARGETS env var

`rapids_cmake`'s `rapids_hip_set_architectures` checks that
`CMAKE_HIP_ARCHITECTURES` matches `AMDGPU_TARGETS`. A mismatch is fatal:
```
mismatch between CMAKE_HIP_ARCHITECTURES='gfx908;gfx90a;gfx942;gfx950',
AMDGPU_TARGETS='gfx942'
```
**Fix:** Set `ROCM_AMDGPU_TARGETS=gfx942` and `GPU_TARGETS=gfx942` as
environment variables (not cmake -D vars). The ROCm toolchain defaults
`CMAKE_HIP_ARCHITECTURES` to `gfx908;gfx90a;gfx942;gfx950`; the env var
overrides this to a single target.

### 8.3 Network: git HTTP/1.1 (fixes flaky GitHub clones)

DSW pods and some CI environments have flaky HTTP/2 connections to GitHub.
git clone fails with:
```
GnuTLS recv error (-110): The TLS connection was non-properly terminated
HTTP/2 stream 1 was not closed cleanly before end of the underlying stream
```
**Fix:** `git config --global http.version HTTP/1.1`

### 8.4 Pre-clone all CPM dependencies

hipDF has 16+ transitive dependencies fetched via CPM/FetchContent. Each
fetch is a git clone from GitHub — any failure aborts the entire configure.
**Fix:** `build_rocm_deps.sh` Step 0 pre-clones all deps into a cache
directory and passes them as `FETCHCONTENT_SOURCE_DIR_<name>` overrides.
Correct branch/tag names (from rapids-cmake `versions.json`):

| Dependency | Branch/Tag |
|------------|-----------|
| rapids-cmake | branch-25.10 |
| CCCL | v3.0.3 |
| libhipcxx | release/rocmds-26.03 |
| jitify | release/rocmds-26.03 |
| hipcomp | release/rocmds-26.03 |
| spdlog | v1.14.1 |
| fmt | 11.0.2 |
| rapids-logger | branch-0.1.0 |
| flatbuffers | v24.3.25 |
| CRoaring | v4.3.11 |
| dlpack | v1.0 |
| nanoarrow | apache-arrow-nanoarrow-0.6.0 |
| thread-pool | v4.1.0 |
| zstd | v1.5.6 |
| kvikio | branch-25.10 |
| NVTX | v3.2.0 |
| arrow | apache-arrow-18.0.0 |

### 8.5 get_rmm.cmake: find_package instead of CPM re-fetch

hipDF's `get_rmm.cmake` calls `rapids_cpm_rmm` which in HIP mode calls
`rapids_cpm_hipmm` — this re-fetches rmm from source via CPM, causing:
```
CMake Error: Unknown CMake command "rapids_make_logger"
```
The CPM-fetched rmm doesn't have the `rapids_logger` module that the
system-installed hipMM has.
**Fix:** `hipdf_26.06_api_patch.sh` patches `get_rmm.cmake` to use
`find_package(rmm REQUIRED CONFIG)` instead.

### 8.6 hipMM build verified

hipMM was successfully built and installed on real gfx942/ROCm 7.2.1:
- `librmm.so` → `/opt/rocm/lib/`
- Headers → `/opt/rocm/include/rmm/`
- CMake config → `/opt/rocm/lib/cmake/rmm/rmm-config.cmake`
- `find_package(rmm CONFIG)` finds version 4.0.0

## Appendix A: Compile-Test Success Logs (gfx942, ROCm 7.2.1)

The following compile-only tests were run on a real AMD MI300 (gfx942) host
with ROCm 7.2.1, hipcc 7.2.53211. Tests were run in `/dev/shm` (RAM-backed
tmpfs) to avoid interfering with a concurrent training job. No GPU execution.

### A.1 Basic HIP compile

```
$ cat test_basic.cu
#include <hip/hip_runtime.h>
__global__ void kernel(int* x) { *x = 42; }
int main() { return 0; }

$ hipcc -c test_basic.cu -o test_basic.o
RESULT: BASIC COMPILE: OK
```

### A.2 cuda_runtime.h — NOT found on stock ROCm

```
$ cat test_cuda_rt.cu
#include <cuda_runtime.h>
__global__ void k(int* x) { *x = 1; }
int main() { return 0; }

$ hipcc -c test_cuda_rt.cu -o test_cuda_rt.o
test_cuda_rt.cu:1:10: fatal error: 'cuda_runtime.h' file not found
RESULT: FAILED (expected — no cuda_runtime.h in ROCm; shim provides it)
```

### A.3 cuda* runtime API — NOT aliased on stock ROCm

```
$ cat test_compat.cu
#include <hip/hip_runtime.h>
int main() {
    int* d;
    cudaError_t err = cudaMalloc(&d, 4);
    cudaMemcpy(d, &d, 4, cudaMemcpyHostToDevice);
    cudaFree(d);
    return (int)err;
}

$ hipcc -c test_compat.cu -o test_compat.o
t_cuda.cu:4:2: error: unknown type name 'cudaError_t'
t_cuda.cu:4:20: error: use of undeclared identifier 'cudaMalloc'
t_cuda.cu:5:23: error: use of undeclared identifier 'cudaMemcpyHostToDevice'; did you mean 'hipMemcpyHostToDevice'?
RESULT: FAILED (expected — shim provides the aliases)
```

### A.4 cub/cub.cuh — NOT found on stock ROCm

```
$ cat test_cub.cu
#include <cub/cub.cuh>
__global__ void k() {}
int main() { return 0; }

$ hipcc -c test_cub.cu -o test_cub.o
test_cub.cu:1:10: fatal error: 'cub/cub.cuh' file not found
RESULT: FAILED (expected — hipCUB provides hipcub/, not cub/; shim redirects)
```

### A.5 thrust — OK

```
$ cat test_thrust.cu
#include <thrust/device_vector.h>
int main() { thrust::device_vector<int> v(10); return 0; }

$ hipcc -c test_thrust.cu -o test_thrust.o
RESULT: OK
```

### A.6 __shfl_xor_sync with 32-bit mask — FAILS

```
$ cat test_shfl.cu
#include <hip/hip_runtime.h>
__global__ void k() {
    unsigned m=0xFFFFFFFFu; int v=1;
    v+=__shfl_xor_sync(m,v,1);
    __syncwarp();
}
int main(){return 0;}

$ hipcc -c test_shfl.cu -o t_shfl.o
/opt/rocm-7.2.1/.../amd_warp_sync_functions.h:307:62: error: static assertion
failed due to requirement 'sizeof(unsigned int) == 8': The mask must be a
64-bit integer.
RESULT: FAILED (fixed: fsst.cuh:132 now casts to unsigned long long)
```

### A.7 __shfl_xor_sync with 64-bit mask — OK

```
$ cat test_shfl64.cu
#include <hip/hip_runtime.h>
__global__ void k() {
    unsigned long long m=0xFFFFFFFFFFFFFFFFull; int v=1;
    v+=__shfl_xor_sync(m,v,1);
}
int main(){return 0;}

$ hipcc -c test_shfl64.cu -o test_shfl64.o
RESULT: 64BIT MASK OK
```

### A.8 roctx shim — OK

```
$ cat test_roctx.cpp
#include <roctracer/roctx.h>
#include <string>
class scoped_range {
    roctx_range_id_t id_;
public:
    explicit scoped_range(const char* n) noexcept : id_(roctxRangeStartA(n?n:"")) {}
    explicit scoped_range(std::string const& n) noexcept : id_(roctxRangeStartA(n.c_str())) {}
    ~scoped_range() noexcept { roctxRangeStop(id_); }
};
int main() { scoped_range r{"test"}; return 0; }

$ hipcc -I/opt/rocm-7.2.1/include -c test_roctx.cpp -o test_roctx.o
RESULT: roctx shim OK
```

### A.9 cooperative_groups via hip — OK

```
$ cat test_cg.cu
#include <hip/hip_cooperative_groups.h>
__global__ void k() { auto b = cooperative_groups::this_thread_block(); }
int main(){return 0;}

$ hipcc -c test_cg.cu -o test_cg.o
RESULT: hip cg OK
```

### A.10 __syncwarp() — OK

```
$ cat test_syncwarp.cu
#include <hip/hip_runtime.h>
__global__ void k() { __syncwarp(); }
int main(){return 0;}

$ hipcc -c test_syncwarp.cu -o test_syncwarp.o
RESULT: OK
```

### A.11 hipify-clang available

```
$ which hipify-clang
/usr/bin/hipify-clang
```

### A.12 ROCm environment summary

```
GPU: gfx942 (AMD Instinct MI300)
ROCm: 7.2.1
hipcc: 7.2.53211-e1a6bc5663 (AMD clang 22.0.0git)
cmake: 3.31.10
hipCUB: /opt/rocm/include/hipcub/
rocThrust: /opt/rocm/include/thrust/
roctx: /opt/rocm-7.2.1/include/roctracer/roctx.h
hipify-clang: /usr/bin/hipify-clang
```

### A.13 What was NOT on the host

```
cuda_runtime.h: NOT in /opt/rocm (only in Triton's NVIDIA backend package)
cub/cub.cuh: NOT in /opt/rocm (hipCUB provides hipcub/)
libcudf/librmm: NOT installed (ML training box, not data-science box)
libroctx64.so: NOT installed (header exists, library missing)
docker/podman: NOT installed
conda/mamba: NOT installed
```
