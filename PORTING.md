# Sirius → AMD ROCm/HIP Port

This branch (`feat/rocm-port`) adds an opt-in AMD ROCm/HIP backend to Sirius
alongside the original NVIDIA CUDA path. It is a **build-system and
compatibility foundation** — not a working ROCm engine. The build does **not**
succeed end-to-end yet. The reasons, scope, and remaining work are documented
honestly below, including a **fatal flaw** discovered during line-by-line
review (§3).

## What this branch changes

| File | Change |
|---|---|
| `CMakeLists.txt` | `ENABLE_ROCM` option; HIP language + `gfx90a;gfx942;gfx950` archs + `.cu`→HIP extension mapping; gates for `cuco`/`cuCascade`/NVML; per-backend GPU target properties; conditional `install()`/export; roctx discovery via `find_library` |
| `cmake/rocm_compat/nvtx3/nvtx3.hpp` | Shim mapping `nvtx3::scoped_range` → `roctxRangeStartA`/`roctxRangeStop` |
| `PORTING.md` | This document |

**Zero `.cu`/`.cpp`/`.hpp` source files are edited** (the shim is a new file).
This is deliberate but has a critical consequence explained in §3.

---

## 1. The dependency map (verified against upstream)

| Sirius dependency | ROCm-DS equivalent | Status | Notes |
|---|---|---|---|
| **cuDF** | **hipDF** | ✅ Drop-in | Keeps `namespace cudf`, exports `cudf::cudf` target, `project(CUDF LANGUAGES ... HIP)`. `find_package(cudf)` and `cudf::cudf` link line unchanged. |
| **RMM** | **hipMM** | ✅ Drop-in | Keeps `namespace rmm`, `rmm/` includes, exports `rmm::rmm` target. |
| **cuCollections (cuco)** | **None** | ❌ Hard blocker | No ROCm port. Sirius uses `cuco::bloom_filter`, `cuco::static_set` + policies in 4 files. Gated behind `SIRIUS_ENABLE_CUCO` (OFF on ROCm). |
| **cuCascade** | **None** | ❌ Hard blocker | No ROCm port. **59 EXTENSION_SOURCES `.cpp` files + 3 `.cu` files** use `cucascade::`. See §3. |
| **thrust / cub** | rocThrust / hipCUB | ✅ Drop-in | Same include paths + namespaces, provided transitively by hipDF. |
| **nvtx3** | roctx | ✅ Shimmed | `cmake/rocm_compat/nvtx3/nvtx3.hpp` maps `nvtx3::scoped_range` (the only nvtx3 type used, 47 call sites) to roctx. |
| **NVML** | rocm-smi / hwloc | ⚠️ Gated off | `CUDA::nvml_static` block skipped under `ENABLE_ROCM`. Topology rewrite is future work. |

### Key finding: hipDF does NOT HIPIFY its own source

hipDF's `cpp/src/utilities/cuda_memcpy.cu` and `cuda.cpp` keep `cuda*` runtime
calls verbatim. hipDF relies on the ROCm SDK's CUDA compatibility layer: when
compiled as HIP, `cuda*` symbols resolve to `hip*` equivalents. This means
Sirius's files with `cuda*` calls need **zero source edits** for the same
coverage — IF the compatibility layer is complete (see §4 for gaps).

---

## 2. What the CMake changes do (correctly)

- `ENABLE_ROCM=ON` → `project(sirius LANGUAGES CXX HIP)`, sets `gfx90a;gfx942;gfx950`
- `CMAKE_HIP_SOURCE_FILE_EXTENSIONS` extended to include `.cu` so hip-clang compiles the kernel files
- `SIRIUS_ENABLE_CUCO` / `SIRIUS_ENABLE_CUCASCADE` options (OFF on ROCm, ON on CUDA)
- `cuco::cuco` link only inside `if(SIRIUS_ENABLE_CUCO)`; `cuCascade::` links only inside `if(SIRIUS_ENABLE_CUCASCADE)`
- 3 `.cu` kernels excluded from `CUDA_SOURCES` when their deps are gated off:
  - `dynamic_filter_replica_transfer.cu` → gated on `SIRIUS_ENABLE_CUCASCADE` (uses `cucascade::`, no `cuco::`)
  - `sirius_dynamic_bloom_filter.cu` → gated on `SIRIUS_ENABLE_CUCO AND SIRIUS_ENABLE_CUCASCADE` (uses both)
  - `sirius_dynamic_in_list_filter.cu` → gated on `SIRIUS_ENABLE_CUCO AND SIRIUS_ENABLE_CUCASCADE` (uses both — verified: includes `<cucascade/memory/memory_space.hpp>`)
- NVML `CUDA::nvml_static` block gated with `if(NOT ENABLE_ROCM AND TARGET CUDA::nvml_static)`
- `install()` branches on `SIRIUS_ENABLE_CUCASCADE` (cuCascade targets only in ON branch)
- `parquet_benchmark` excluded from build when `SIRIUS_ENABLE_CUCASCADE=OFF` (its source unconditionally includes cucascade headers)
- roctx `find_library`/`find_path` with `ROCTX_PATH` hint; `ROCTX_INCLUDE_DIR` + `ROCTX_LIBRARY` added to all 4 targets (extension ×2, unittest, parquet_benchmark)
- `HIP_RESOLVE_DEVICE_SYMBOLS ON` (mirrors CUDA path)
- nvtx3 shim on include path for all targets under `ENABLE_ROCM`

---

## 3. ⚠️ FATAL FLAW: 59 `.cpp` files have unguarded cuCascade includes

**This is the single most critical finding and it blocks the ROCm build.**

The `EXTENSION_SOURCES` list in `CMakeLists.txt` (lines ~293–446) is a flat,
unconditional `set()`. **No `.cpp` entry is conditionally excluded based on
`SIRIUS_ENABLE_CUCASCADE`.** And **no `.cpp` file contains internal
`#ifdef SIRIUS_ENABLE_CUCASCADE` guards** (verified: `grep -rn
SIRIUS_ENABLE_CUCASCADE src/ --include="*.cpp"` returns zero matches).

Yet **59 of those `.cpp` files** hard-include `<cucascade/...>` headers and use
`cucascade::` types. Examples (verified by reading the files):

| File | cuCascade usage |
|---|---|
| `src/sirius_extension.cpp:31-35` | `#include <cucascade/cudf/gpu_data_representation.hpp>`, `<cucascade/memory/common.hpp>`, `<cucascade/memory/memory_space.hpp>`; uses `cucascade::memory::Tier::GPU` |
| `src/memory/defragmenter_oom_policy.cpp:24` | `#include <cucascade/memory/error.hpp>`; uses `cucascade::memory::cucascade_out_of_memory` |
| `src/op/scan/duckdb_native_decoder.cpp:653-654` | `cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation` |
| `src/op/dynamic_filter_publisher.cpp:28` | `#include <cucascade/memory/memory_space.hpp>` |

When `SIRIUS_ENABLE_CUCASCADE=OFF` (the ROCm default):
- The cuCascade submodule is NOT built → headers not on any include path
- Every one of these 59 files fails at `#include <cucascade/...>` with
  "No such file or directory"

**The `.cu` gating works (§2). The `.cpp` gating does not exist.** The ROCm
build cannot reach link time. This is a design-level gap, not a typo.

### Why this is hard to fix

Unlike the 3 `.cu` kernels (easily excluded from a list), these 59 `.cpp`
files are the **core of the live engine** — the operators, pipeline executor,
memory manager, context, and extension entry point. They cannot be simply
excluded without gutting the engine. The real fix is one of:
1. Port cuCascade to HIP (months of library work)
2. Add `#ifdef SIRIUS_ENABLE_CUCASCADE` guards to 59 files (requires compile
   feedback to validate; ~infeasible blind)
3. Build a ROCm-native reservation/repository subsystem to replace cuCascade

This branch does **not** attempt any of these. It documents the gap honestly.

---

## 4. At-risk CUDA APIs (beyond the compat layer)

Of ~67 distinct `cuda*` symbols across 69 live files, most are covered by
HIP's compat layer. These are NOT covered or are uncertain:

| Symbol | File(s) | Status |
|---|---|---|
| `cudaMemcpyBatchAsync` + types | `duckdb_native_decoder.cpp:664`, `dynamic_filter_replica_transfer.cu:131` | Guarded by `#if CUDART_VERSION >= 12080`. If HIP compat defines `CUDART_VERSION`, the batch path is taken — HIP may lack this API. If undefined, the serial fallback is used. **Uncertain.** Both files also fail per §3. |
| `cudaMemPool*` | `defragmenter_oom_policy.cpp:40,47,50,91` | HIP has `hipMemPool*` but enum mappings must match. File fails per §3 regardless. |
| `cudaDevAttrL2CacheSize` | `dynamic_filter_publisher.cpp:52,61` | HIP names it `hipDeviceAttributeL2CacheSize`. File fails per §3 regardless. |
| `cudaProfilerStart/Stop` | `sirius_extension.cpp:38-39,1282,1294` | `extern "C"` decls linked via libcudart. Under ROCm, symbol may not resolve. File fails per §3 regardless. |
| `__shfl_xor_sync`, `__syncwarp()` | `src/include/cuda/scan/strings/fsst.cuh:132,227,229,237,239` | Raw CUDA warp intrinsics. HIP compat may provide these, but semantics differ. In a header used by 3 portable `.cu` kernels. |
| `<cooperative_groups.h>` | `src/include/cuda/scan/detail/shared_staging.cuh:25-26` | HIP provides via `<hip/cooperative_groups.h>`; include path must be remapped. |
| `<cuda/__memory/aligned_size.h>` | `src/include/cuda/scan/detail/shared_staging.cuh:28` | CCCL private internal header; version-dependent availability in hipDF's CCCL. |

---

## 5. Build configuration

### NVIDIA CUDA (default — unchanged)

```bash
pixi run make    # ENABLE_ROCM defaults OFF
```

### AMD ROCm (does NOT build yet — see §3)

```bash
# Prerequisites: ROCm 7.2.3+, hipDF, hipMM, roctx64, hip-clang
cmake -B build/rocm -S . \
  -DENABLE_ROCM=ON \
  -DSIRIUS_ENABLE_CUCO=OFF \
  -DSIRIUS_ENABLE_CUCASCADE=OFF \
  -DSIRIUS_BUILD_S3_TESTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a"
# This will FAIL at compile time: 59 .cpp files include <cucascade/...> (§3).
```

---

## 6. Remaining work (roadmap)

1. **Resolve §3** — the fatal flaw. Port cuCascade, guard 59 files, or build a
   ROCm-native replacement. This is the gating item for any working ROCm build.
2. **Validate CMake configure** on Linux + AMD (gfx90a) with hipDF/hipMM.
3. **Verify §4 at-risk APIs** resolve under HIP's compat layer on a real toolchain.
4. **Legacy `gpu_processing` path**: 2 legacy files use cucascade:: — investigate
   whether those can be guarded to get a minimal in-memory path compiling.
5. **cuco port or replacement** for Bloom/in-list dynamic filters.
6. **NVML → rocm-smi/hwloc** for multi-GPU topology.

---

## 7. Verification status (honest)

| Claim | How verified |
|---|---|
| hipDF keeps `namespace cudf` + `cudf::cudf` target | Fetched header + CMakeLists from GitHub |
| hipMM keeps `rmm::` + `rmm::rmm` | `project(RMM ...)`; derived from RMM |
| hipDF does not HIPIFY source | `cuda_memcpy.cu` / `cuda.cpp` contain verbatim `cuda*` calls |
| cuco / cuCascade have no ROCm port | ROCm-DS org has 10 repos; none is a port |
| roctx API signatures | `ROCm/roctracer/inc/roctx.h` |
| 9 portable `.cu` kernels are cuco/cucascade-free | Read every line of all 9 files |
| 3 gated `.cu` kernels correctly gated | Read every line; `in_list_filter.cu` DOES use cucascade:: |
| nvtx3 shim matches all call sites | 47 `nvtx3::scoped_range` uses — all pass string literal or `.c_str()` |
| **59 `.cpp` files have unguarded cuCascade includes** | **Read CMakeLists.txt + grep-verified; the build CANNOT succeed** |
| CMake if/endif balanced | 40/40 (after fixes) |
| **Not compiled** | No AMD GPU or x86_64 ROCm toolchain available |

---

## 8. Why not a container on macOS?

Developed on Apple Silicon (arm64). A Linux VM (Colima) was started, but ROCm's
gfx90a toolchain is not packaged for arm64, and there is no AMD GPU to pass
through. A container here can run HIPIFY/static analysis but cannot compile
against hipDF/hipMM or execute HIP kernels. That requires a real Linux + AMD
Instinct box.
