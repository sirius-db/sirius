# Sirius → AMD ROCm/HIP Port

This branch (`feat/rocm-port`) adds an opt-in AMD ROCm/HIP backend to Sirius
alongside the original NVIDIA CUDA path. It is a **build-system and
compatibility foundation**: it makes the codebase backend-aware and wires the
ROCm-DS drop-in equivalents (hipDF, hipMM), but does **not** yet produce a
fully building-and-running ROCm engine. The reasons, scope, and remaining work
are documented honestly below.

## TL;DR — what builds on AMD today

**Nothing end-to-end yet.** This branch establishes the CMake plumbing and a
compatibility shim. A full build is blocked on a library port that does not
exist yet (cuCascade — see §2). The *subset* that is architecturally portable
is the **legacy `gpu_processing` in-memory path** (cudf + rmm + thrust/cub
only), which is gated behind `ENABLE_LEGACY_SIRIUS`.

## What this branch actually changes

| File | Change |
|---|---|
| `CMakeLists.txt` | `ENABLE_ROCM` option; HIP language + `gfx90a;gfx942;gfx950` archs; gates for `cuco`/`cuCascade`/NVML; per-backend GPU target properties; conditional `install()`/export; roctx discovery via `find_library` |
| `cmake/rocm_compat/nvtx3/nvtx3.hpp` | Shim mapping `nvtx3::scoped_range` → `roctxRangeStartA`/`roctxRangeStop` |

**Zero source files are edited.** This is deliberate and verified (see §3).

---

## 1. The dependency map (verified against upstream)

| Sirius dependency | ROCm-DS equivalent | Status | Notes |
|---|---|---|---|
| **cuDF** | **hipDF** | ✅ Drop-in | Keeps `namespace cudf`, exports `cudf::cudf` target, `project(CUDF LANGUAGES ... HIP)`. `find_package(cudf)` and the `cudf::cudf` link line are unchanged. |
| **RMM** | **hipMM** | ✅ Drop-in | Keeps `namespace rmm`, `rmm/` includes, exports `rmm::rmm` target. `find_package(rmm)` and `rmm::rmm` link line unchanged. |
| **cuCollections (cuco)** | **None** | ❌ Hard blocker | No ROCm port exists. Sirius uses `cuco::bloom_filter`, `cuco::static_set`, and policies in 4 files. Gated behind `SIRIUS_ENABLE_CUCO` (OFF on ROCm). |
| **cuCascade** | **None** | ❌ Hard blocker | No ROCm port exists. 171 live files use `cucascade::` (repository manager, GPU/host/disk data representations, topology discovery). Gated behind `SIRIUS_ENABLE_CUCASCADE` (OFF on ROCm). |
| **thrust / cub** | rocThrust / hipCUB | ✅ Drop-in | Same include paths (`<thrust/...>`, `<cub/...>`) and namespaces. Provided transitively by hipDF. |
| **nvtx3** | roctx | ✅ Shimmed | `cmake/rocm_compat/nvtx3/nvtx3.hpp` maps `nvtx3::scoped_range` (the only nvtx3 type Sirius uses, 47 call sites) to `roctxRangeStartA`/`roctxRangeStop`. |
| **NVML** | rocm-smi / hwloc | ⚠️ Gated off | `CUDA::nvml_static` block skipped under `ENABLE_ROCM`. Topology rewrite is future work. |

### Key finding: hipDF does NOT HIPIFY its own source

Inspection of hipDF's `cpp/src/utilities/cuda_memcpy.cu` and `cuda.cpp` shows
**`cuda*` runtime calls kept verbatim** (`cudaMemcpyAsync`, `cudaGetDevice`,
`cudaDeviceGetAttribute`). hipDF relies on the ROCm SDK's CUDA compatibility
layer: when compiled as HIP, `cuda*` symbols resolve to `hip*` equivalents, and
a `cuda_runtime.h` wrapper is installed. This means **Sirius's 69 files with
`cuda*` calls need zero source edits** — the same compatibility layer covers
them.

---

## 2. The hard blocker: cuCascade

cuCascade is NVIDIA's GPU memory-reservation and out-of-core repository system.
Sirius's **live** engine ("Super Sirius", the `gpu_execution` path, on by
default) is built directly on it:

- `cucascade::shared_data_repository_manager` — the central data repository
- `cucascade::gpu_table_representation` / `host_data_representation` /
  `disk_data_representation` — tiered memory representations
- `cucascade::topology_discovery` — multi-GPU topology
- `cucascade::memory::memory_space` — stream + allocator + device context

**171 live (non-legacy) source files** reference `cucascade::`. 58 of the
`EXTENSION_SOURCES` `.cpp` files are cuCascade-coupled. Without a ROCm port of
cuCascade, these cannot compile.

This is not a shim problem — it's a **library port** estimated at months of
effort. This branch gates cuCascade off cleanly so the rest of the porting
work can proceed without it.

### Consequence for queries on ROCm

With `SIRIUS_ENABLE_CUCASCADE=OFF`, the live `gpu_execution` operators are not
compiled. Queries routed at them hit Sirius's existing **graceful degradation
path** — they fall back to DuckDB's native CPU engine. This is the same
behavior Sirius uses for any unsupported operator.

---

## 3. The at-risk CUDA APIs (4 symbols)

Of the 67 distinct `cuda*` symbols across 69 live files, 63 are covered by
HIP's compatibility layer (same mechanism hipDF relies on). Four are at risk:

| Symbol | Location | Status |
|---|---|---|
| `cudaMemcpyBatchAsync` + `cudaMemcpyAttributes` | `duckdb_native_decoder.cpp`, `dynamic_filter_replica_transfer.cu` | ✅ Already guarded by `#if CUDART_VERSION >= 12080` with serial `cudaMemcpyAsync` fallback. Under HIP, `CUDART_VERSION` is undefined → takes the fallback. No edit needed. |
| `cudaMemPool*` (`GetAttribute`, `TrimTo`, `AttrUsedMemCurrent`) | `defragmenter_oom_policy.cpp` | ⚠️ In cuCascade-gated code path. Not compiled on ROCm until cuCascade is ported. |
| `cudaDevAttrL2CacheSize` | `dynamic_filter_publisher.cpp` | ⚠️ In cuCascade-gated code path. HIP names it `hipDeviceAttributeL2CacheSize`. Needs mapping when that file is un-gated. |
| `cudaProfilerStart`/`Stop` | `sirius_extension.cpp` | ⚠️ `extern "C"` declarations linked via libcudart. Under ROCm, use `rocprofiler` API or no-op. File is always compiled — **needs a guard** (next step). |

---

## 4. Build configuration

### NVIDIA CUDA (default — unchanged)

```bash
pixi run make    # original build, ENABLE_ROCM defaults OFF
```

### AMD ROCm (experimental, incomplete)

```bash
# Prerequisites: ROCm 7.2.3+, hipDF, hipMM, roctx64, hip-clang on PATH
# Target GPUs: gfx90a / gfx942 / gfx950 (AMD Instinct MI series)

cmake -B build/rocm -S . \
  -DENABLE_ROCM=ON \
  -DSIRIUS_ENABLE_CUCO=OFF \
  -DSIRIUS_ENABLE_CUCASCADE=OFF \
  -DENABLE_LEGACY_SIRIUS=ON \
  -DSIRIUS_BUILD_S3_TESTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a" \
  [DuckDB extension build flags]

cmake --build build/rocm
```

> ⚠️ This configure step has **not been executed** in this branch's development
> (no AMD GPU or x86_64 ROCm toolchain was available). It represents the
> intended invocation. Validate on a real Linux + AMD Instinct host.

---

## 5. Remaining work (roadmap)

1. **Validate CMake configure** on Linux + AMD (gfx90a) with hipDF/hipMM
   installed. Fix any target-property or flag deltas the real toolchain
   reveals.
2. **Guard `cudaProfilerStart/Stop`** in `sirius_extension.cpp` behind
   `#ifndef SIRIUS_ROCM` (or map to rocprofiler).
3. **Legacy `gpu_processing` path**: get it compiling + a smoke test running
   on AMD. This is the first milestone that produces a working (in-memory)
   ROCm Sirius.
4. **cuCascade port** (the large item): either port cuCascade to HIP, or
   design a ROCm-native reservation/repository subsystem. This unblocks the
   live `gpu_execution` engine.
5. **cuco port or replacement**: needed for Bloom/in-list dynamic filters.
6. **NVML → rocm-smi/hwloc**: for multi-GPU topology discovery.
7. **`cudaDevAttrL2CacheSize` → `hipDeviceAttributeL2CacheSize`** mapping when
   `dynamic_filter_publisher.cpp` is un-gated.

---

## 6. Verification status (honest)

| Claim | How verified |
|---|---|
| hipDF keeps `namespace cudf` + `cudf::cudf` target | Fetched `cpp/include/cudf/table/table_view.hpp` — shows `namespace cudf`; CMakeLists shows `add_library(cudf::cudf ALIAS cudf)` |
| hipMM keeps `rmm::` namespace + `rmm::rmm` target | `project(RMM ...)`; hipMM is derived from RMM with matching structure |
| hipDF does not HIPIFY source | `cuda_memcpy.cu` / `cuda.cpp` contain verbatim `cuda*` calls |
| hipDF enables HIP language natively | `project(CUDF ... LANGUAGES C CXX HIP)`; `ConfigureHIP.cmake` sets `HIP_STANDARD 20`, links `hip::host` |
| cuco has no ROCm port | ROCm-DS org has 10 repos; none is a cuco/cuCollections port |
| cuCascade has no ROCm port | Same; no repository/port exists |
| roctx API signatures | `ROCm/roctracer/inc/roctx.h`: `roctx_range_id_t roctxRangeStartA(const char*)`, `void roctxRangeStop(roctx_range_id_t)`, `typedef uint64_t roctx_range_id_t` |
| CMake if/endif balanced | 38 if / 38 endif, 5 foreach / 5 endforeach, 1 function / 1 endfunction |
| **Not compiled** | No AMD GPU or x86_64 ROCm toolchain was available. This is static, by-inspection work. |

---

## 7. Why not a container on macOS?

This branch was developed on Apple Silicon (arm64). A Linux VM (Colima) was
started, but ROCm's gfx90a cross-compiler toolchain is not packaged for arm64,
and there is no AMD GPU to pass through. A container on this host can run
HIPIFY and static analysis but **cannot compile against hipDF/hipMM or execute
HIP kernels**. That requires a real Linux + AMD Instinct box.
