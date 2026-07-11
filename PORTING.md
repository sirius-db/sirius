# Sirius ROCm/HIP Port — Component Architecture

This document describes how the ROCm/HIP backend is implemented: the
components, how they are wired, and the constraints that shaped each decision.
It is a reference for anyone continuing the port.

## 1. Backend Selection Component

**Entry point:** `ENABLE_ROCM` CMake option (`CMakeLists.txt:42`, default OFF).

The build selects a GPU backend at configure time. Exactly one of two paths
activates:

| | NVIDIA CUDA (default) | AMD ROCm (opt-in) |
|---|---|---|
| Language enabled | `CXX CUDA` | `CXX HIP` |
| Architecture var | `CMAKE_CUDA_ARCHITECTURES` (native) | `CMAKE_HIP_ARCHITECTURES` (gfx90a;gfx942;gfx950) |
| Source-file ext | `.cu` → CUDA (built-in) | `.cu` → HIP (via `CMAKE_HIP_SOURCE_FILE_EXTENSIONS`) |
| GPU standard prop | `CUDA_STANDARD 20` | `HIP_STANDARD 20` |
| Device-symbol resolve | `CUDA_RESOLVE_DEVICE_SYMBOLS ON` | `HIP_RESOLVE_DEVICE_SYMBOLS ON` |
| Separable compilation | `CUDA_SEPARABLE_COMPILATION ON` | `HIP_SEPARABLE_COMPILATION ON` |
| Extended lambda | `--expt-extended-lambda` (nvcc) | not needed (hip-clang default) |

**Why `.cu` extension mapping is needed:** CMake's HIP language recognizes `.hip`
by default. Sirius's kernel files are `.cu`. Under `ENABLE_ROCM`, the CUDA
language is not enabled, so `.cu` files have no language assignment unless
`CMAKE_HIP_SOURCE_FILE_EXTENSIONS` is extended. This is set at
`CMakeLists.txt:66`:

```cmake
if(ENABLE_ROCM)
  list(APPEND CMAKE_HIP_SOURCE_FILE_EXTENSIONS cu)
endif()
```

hipDF itself uses the same convention (its source files are also `.cu` compiled
as HIP).

## 2. Dependency Wiring

### 2.1 cuDF → hipDF (drop-in)

hipDF is AMD's port of cuDF. It preserves three things that make it a drop-in:

- **Package name:** `find_package(cudf REQUIRED CONFIG)` resolves to hipDF's
  `cudf-config.cmake` (hipDF's `project(CUDF ...)` keeps the name).
- **CMake target:** `cudf::cudf` (hipDF defines `add_library(cudf::cudf ALIAS cudf)`).
- **C++ namespace:** `namespace cudf` (verified in
  `cpp/include/cudf/table/table_view.hpp`).

The `target_link_libraries(... cudf::cudf ...)` line in Sirius's CMakeLists is
identical across backends. No source file using `cudf::` needs editing.

### 2.2 RMM → hipMM (drop-in)

hipMM preserves the same three things: package name (`rmm`), target
(`rmm::rmm`), namespace (`rmm::`), and include paths (`<rmm/...>`). The
`rmm::rmm` link line is identical across backends.

### 2.3 thrust/cub → rocThrust/hipCUB (transitive)

hipDF depends on rocThrust and hipCUB, which expose the same `thrust::` and
`cub::` namespaces with the same include paths (`<thrust/...>`, `<cub/...>`).
No source edits needed.

### 2.4 The CUDA Runtime Compatibility Layer

**This is the mechanism that eliminates per-file HIPIFY.**

hipDF's own source files (`cpp/src/utilities/cuda_memcpy.cu`, `cuda.cpp`) keep
`cuda*` runtime calls verbatim — `cudaMemcpyAsync`, `cudaGetDevice`,
`cudaDeviceGetAttribute`, etc. They compile as HIP because the ROCm SDK
installs a compatibility layer: when hip-clang compiles a `.cu` file, `cuda*`
symbols are macro-aliased to `hip*` equivalents, and a `cuda_runtime.h`
wrapper is provided.

Sirius has ~67 distinct `cuda*` symbols across 69 live files. Because hipDF
relies on this same layer, Sirius's `cuda*` calls need zero source edits —
the layer covers them. The at-risk symbols (where the layer may be incomplete)
are catalogued in §5.

## 3. Gating Architecture

Two NVIDIA-only dependencies have no ROCm port. Each is controlled by a CMake
option that defaults OFF under `ENABLE_ROCM` and ON otherwise.

### 3.1 SIRIUS_ENABLE_CUCO (cuCollections)

**What cuco is:** NVIDIA's GPU hash-map and Bloom-filter library
(cuCollections). Sirius uses a narrow slice of its API:

- `cuco::bloom_filter` (in `sirius_dynamic_bloom_filter.cu`)
- `cuco::static_set` (in `sirius_dynamic_in_list_filter.cu`)
- Supporting types: `cuco::extent`, `cuco::empty_key`,
  `cuco::arrow_filter_policy`, `cuco::default_filter_policy`,
  `cuco::xxhash_64`, `cuco::double_hashing`, `cuco::default_hash_function`,
  `cuco::contains`

**How it's wired:**
- CMake option `SIRIUS_ENABLE_CUCO` (`CMakeLists.txt:85`)
- When ON: `find_package(cuco CONFIG REQUIRED)` or FetchContent (header-only),
  creates `cuco::cuco` target, links it to extension targets, defines
  `SIRIUS_ENABLE_CUCO=1`
- When OFF: `cuco::cuco` target is never created; the two `.cu` files that use
  it are excluded from `CUDA_SOURCES`

**What breaks when OFF:** Bloom-filter and IN-list dynamic filters are not
compiled. `sirius_dynamic_bloom_filter.cpp` / `sirius_dynamic_in_list_filter.cu`
are dropped. The `sirius_dynamic_filter.hpp` header still compiles (it
forward-declares the classes via PIMPL). Dynamic-filter *publication* code in
`.cpp` files that references these classes will still compile but the filters
won't be constructed at runtime.

### 3.2 SIRIUS_ENABLE_CUCASCADE (cuCascade)

**What cuCascade is:** NVIDIA's GPU memory-reservation and out-of-core data
repository system. It provides:

- `cucascade::shared_data_repository_manager` — central data repository
- `cucascade::gpu_table_representation` / `host_data_representation` /
  `disk_data_representation` — tiered memory representations
- `cucascade::topology_discovery` — multi-GPU topology
- `cucascade::memory::memory_space` — stream + allocator + device context
- `cucascade::memory::memory_reservation_manager` — reservation subsystem
- `cucascade::memory::fixed_size_host_memory_resource` — pinned host allocator

**How it's wired:**
- CMake option `SIRIUS_ENABLE_CUCASCADE` (`CMakeLists.txt:86`)
- When ON: `add_subdirectory(cucascade)` builds the submodule; `cuCascade::cucascade`
  and `cuCascade::cucascade_cudf` targets are linked to extension targets;
  `SIRIUS_ENABLE_CUCASCADE=1` is defined
- When OFF: submodule not built; targets not linked; one `.cu` file
  (`dynamic_filter_replica_transfer.cu`) excluded from `CUDA_SOURCES`

**The critical coupling (see §6):** 59 `.cpp` files in `EXTENSION_SOURCES`
hard-include `<cucascade/...>` headers. These are NOT excluded from the build
when `SIRIUS_ENABLE_CUCASCADE=OFF`. This is the port's blocking issue.

### 3.3 NVML

`CUDA::nvml_static` (NVIDIA Management Library) is pulled in transitively by
cuCascade. The CMake block that redirects it from the static stub to the
shared library (`CMakeLists.txt:663`) is gated:

```cmake
if(NOT ENABLE_ROCM AND TARGET CUDA::nvml_static)
```

Under ROCm, `CUDA::nvml_static` is never a target, so the block is skipped.
ROCm's topology source is `rocm-smi` / hwloc — not wired yet.

## 4. nvtx3 → roctx Shim

**Component:** `cmake/rocm_compat/nvtx3/nvtx3.hpp`

Sirius uses exactly one nvtx3 type — `nvtx3::scoped_range` — as an RAII
profiling range (47 call sites across 31 files). Every call site passes either
a string literal (`{"sirius::query"}`) or a `.c_str()` result (`const char*`).

ROCm's equivalent is the roctx API:
- `roctx_range_id_t roctxRangeStartA(const char* message)` — starts a range
- `void roctxRangeStop(roctx_range_id_t id)` — stops a range
- `roctx_range_id_t` is `typedef uint64_t`

The shim defines `nvtx3::scoped_range` with two constructors (`char const*` and
`std::string const&`) and a destructor that calls `roctxRangeStop`. It is
non-copyable and non-movable, matching nvtx3 semantics.

**Wiring:** The `cmake/rocm_compat/` directory is added to the include path of
all four build targets (extension ×2, unittest, parquet_benchmark) only when
`ENABLE_ROCM=ON`. This shadows the NVIDIA `<nvtx3/nvtx3.hpp>` header (which
doesn't exist on an AMD host). The roctx include dir (`ROCTX_INCLUDE_DIR`,
discovered via `find_path`) and library (`ROCTX_LIBRARY`, discovered via
`find_library`) are also added to all four targets.

**Transitive reach:** `<nvtx3/nvtx3.hpp>` is included by public headers
(`src/include/pipeline/sirius_pipeline.hpp:28`,
`src/include/helper/helper.hpp:19`), so every TU that includes those headers
reaches the shim. This is why all four targets need the roctx include path.

## 5. CUDA Runtime API Coverage

The compatibility layer (§2.4) covers the majority of Sirius's `cuda*` calls.
The following symbols are at risk — the layer may not map them, or semantics
may differ:

| Symbol | Where used | Why at risk |
|---|---|---|
| `__shfl_xor_sync` | `src/include/cuda/scan/strings/fsst.cuh:132` | Raw warp intrinsic. HIP compat may provide it but semantics differ. In a header used by 2 portable `.cu` kernels (`dict_fsst.cu`, `fsst.cu`). |
| `__syncwarp()` | `src/include/cuda/scan/strings/fsst.cuh:227,229,237,239` | Raw warp-sync intrinsic. HIP provides it but behavior may differ. |
| `<cooperative_groups.h>` | `src/include/cuda/scan/detail/shared_staging.cuh:25-26` | HIP provides via `<hip/cooperative_groups.h>`; include path must be remapped by compat layer. |
| `<cuda/__memory/aligned_size.h>` | `src/include/cuda/scan/detail/shared_staging.cuh:28` | CCCL private internal header; version-dependent in hipDF's CCCL. |
| `cudaMemcpyBatchAsync` + types | `src/op/scan/duckdb_native_decoder.cpp:664`, `src/cuda/dynamic_filter_replica_transfer.cu:131` | CUDA 12.8+ API. Guarded by `#if CUDART_VERSION >= 12080`. If HIP compat defines `CUDART_VERSION`, the batch path is taken but HIP may lack the API. |
| `cudaMemPool*` | `src/memory/defragmenter_oom_policy.cpp:40,47,50,91` | HIP has `hipMemPool*` but enum value mappings must match. |
| `cudaDevAttrL2CacheSize` | `src/op/dynamic_filter_publisher.cpp:52,61` | HIP names it `hipDeviceAttributeL2CacheSize`. |
| `cudaProfilerStart/Stop` | `src/sirius_extension.cpp:38-39,1282,1294` | `extern "C"` decls linked via libcudart. Under ROCm the symbol may not resolve. |
| `cub::ThreadLoad<cub::LOAD_LDG>` | `src/cuda/scan/gpu_decode_rle.cu:392,399,409` | hipCUB provides `LOAD_LDG` but maps to a regular load on AMD (no texture cache). Performance difference, not compile error. |

## 6. The cuCascade Coupling (Blocking Issue)

### 6.1 The `.cu` layer (gated, works)

Three `.cu` files use cuCascade and/or cuco. They are correctly excluded from
`CUDA_SOURCES` when their gates are OFF:

| File | Gate | Uses cuco? | Uses cuCascade? |
|---|---|---|---|
| `dynamic_filter_replica_transfer.cu` | `SIRIUS_ENABLE_CUCASCADE` | No | Yes |
| `sirius_dynamic_bloom_filter.cu` | `SIRIUS_ENABLE_CUCO AND SIRIUS_ENABLE_CUCASCADE` | Yes | Yes |
| `sirius_dynamic_in_list_filter.cu` | `SIRIUS_ENABLE_CUCO AND SIRIUS_ENABLE_CUCASCADE` | Yes | Yes (includes `<cucascade/memory/memory_space.hpp>`, passes `memory_space` to `enqueue_replica_copy`) |

### 6.2 The `.cpp` layer (NOT gated, blocks the build)

`EXTENSION_SOURCES` (`CMakeLists.txt:~293-446`) is a flat, unconditional
`set()`. No `.cpp` entry is conditionally excluded. No `.cpp` file contains
`#ifdef SIRIUS_ENABLE_CUCASCADE` guards (verified: zero matches).

**63 of these `.cpp` files** hard-include `<cucascade/...>` headers and use
`cucascade::` types (205 `#include <cucascade/...>` directives total). They
include the engine's core: `sirius_extension.cpp` (entry point),
`sirius_context.cpp`, operators (`sirius_physical_hash_join.cpp`, etc.),
pipeline executor (`gpu_pipeline_executor.cpp`, `gpu_pipeline_task.cpp`),
memory manager (`sirius_memory_reservation_manager.cpp`,
`defragmenter_oom_policy.cpp`), and the downgrade executor.

When `SIRIUS_ENABLE_CUCASCADE=OFF`, the cuCascade submodule is not built, its
headers are not on any include path, and every one of these 63 files fails at
`#include <cucascade/...>`.

### 6.3 Why this can't be fixed by exclusion or guards

The cuCascade coupling exists at three architectural levels, not just
implementation:

1. **Inheritance** — `sirius_memory_reservation_manager` inherits from
   `cucascade::memory::memory_reservation_manager` (IS-A relationship).
   Guarding this requires redesigning the class hierarchy, not adding `#ifdef`.
2. **Public API signatures** — `gpu_pipeline_executor`'s constructor takes
   `cucascade::memory::memory_space*`; `sirius_context`'s constructor takes
   `std::vector<cucascade::memory::memory_space_config>`. Removing cuCascade
   changes the public API of core classes.
3. **Implementation** — exception types (`cucascade::memory::cucascade_out_of_memory`),
   memory resources, stream pools. Some could be guarded, but the guards would
   need to disable not just includes but all `cucascade::` type usage, which is
   woven through the operator/pipeline/memory architecture.

The options are:
1. **Port cuCascade to HIP** — a full library port (repository manager,
   tiered representations, topology, reservation). Months of work.
2. **Build a ROCm-native reservation/repository subsystem** — design and
   implement a replacement providing equivalent functionality on AMD.
3. `#ifdef` guards alone are **not feasible** for levels 1 and 2 above.

None of these are attempted in this branch.

## 7. SHOWSTOPPER #2: No ROCm Package Ecosystem

**Independent of §6. Either alone prevents an end-to-end ROCm build.**

`pixi.toml` and `vcpkg.json` are 100% NVIDIA:
- `pixi.toml` channels: `rapidsai`, `conda-forge` (no `rocm` channel)
- GPU packages: `cuda-nvcc`, `cuda-nvml-dev`, `librmm`, `libcudf 26.06`,
  `libcurand-dev`, `libnvjitlink-dev`, `cuda-nvrtc-dev`
- Features: `cuda12`, `cuda13` only — no `rocm` feature/environment
- `vcpkg.json` dependencies: `cudf`, `cuco` (NVIDIA ports; no hipDF overlay)
- `.gitmodules`: `cucascade` → `github.com/NVIDIA/cuCascade.git`

**The name-collision problem:** hipDF reuses the `cudf` package name (by
design, for drop-in compatibility). `find_package(cudf REQUIRED CONFIG)` at
`CMakeLists.txt:150` has no `ENABLE_ROCM`-specific `HINTS`/`PATHS` (unlike the
roctx block which honors `ROCTX_PATH`/`/opt/rocm`). With pixi's NVIDIA cudf
installed, `find_package(cudf)` resolves to **NVIDIA cudf**, not hipDF. The
`.cu` kernels compile as HIP but link against NVIDIA libcudf → CUDA/HIP runtime
ABI conflict.

**What's needed:** a ROCm pixi environment (or vcpkg overlay ports for
hipDF/hipMM), and `ENABLE_ROCM`-specific `find_package` hints that point at the
hipDF/hipMM install, away from any NVIDIA cudf on the path.

## 8. Legacy Path (`gpu_processing`)

The legacy in-memory path (`src/legacy/`) is gated behind
`ENABLE_LEGACY_SIRIUS` (OFF by default, controlled by
`src/legacy/CMakeLists.txt`). Its dependency surface is smaller:

- 0 cuco references
- 2 files use `cucascade::` (vs 63 in the live path)

It depends on cudf, rmm, thrust, cub only — all of which have drop-in ROCm
equivalents. This makes it the most architecturally portable subset, but the
2 cuCascade-coupled legacy files still need investigation.

## 8. Build Target Map

| Target | Always built? | Gets rocm_compat include? | Gets ROCTX include+lib? | cuCascade linked? |
|---|---|---|---|---|
| `sirius_extension` | Yes | If `ENABLE_ROCM` | If `ENABLE_ROCM` | If `SIRIUS_ENABLE_CUCASCADE` |
| `sirius_loadable_extension` | Yes | If `ENABLE_ROCM` | If `ENABLE_ROCM` | If `SIRIUS_ENABLE_CUCASCADE` |
| `sirius_unittest` | Yes | If `ENABLE_ROCM` | If `ENABLE_ROCM` | Via `sirius_extension` |
| `parquet_benchmark` | Excluded if `SIRIUS_ENABLE_CUCASCADE=OFF` | If `ENABLE_ROCM` | If `ENABLE_ROCM` | If `SIRIUS_ENABLE_CUCASCADE` |

`parquet_benchmark` is excluded from the build when cuCascade is off because
`test/io/parquet_benchmark.cpp` unconditionally `#include`s `<cucascade/...>`
headers — there is no source-level guard.

## 9. Latent Risks

These do not block the build today (they are masked by §6/§7 showstoppers)
but would surface once those are resolved:

- **`CMAKE_HIP_FLAGS` CCCL strip missing** (`CMakeLists.txt:~104-130`): the
  VCPKG_BUILD block strips `-I.../include/cccl` from `CMAKE_CXX_FLAGS` and
  `CMAKE_CUDA_FLAGS` but not `CMAKE_HIP_FLAGS`. If a pixi/conda env injects
  CCCL into HIP flags under `VCPKG_BUILD + ENABLE_ROCM`, hip-clang sees a
  conflicting CCCL version. Dormant only because §7 makes that combo unreachable.
- **`telemetry_bridge` is an ungated Rust-toolchain landmine**
  (`CMakeLists.txt:~286`): `add_subdirectory(rust/crates/telemetry/bridge)` is
  unconditional and needs a Rust toolchain. Contrast: `SIRIUS_BUILD_S3_TESTS`
  (which needs Go) is gated. A Go/Rust-less host hits this before even reaching
  the cuCascade `.cpp` failures.
- **roctx wiring is manually replicated** across 4 targets rather than wrapped
  in a single `INTERFACE` library. Any new executable that transitively
  includes a Sirius header pulling `nvtx3` will silently break under
  `ENABLE_ROCM` with `roctracer/roctx.h: file not found`.

## 10. Bug Dependency Graph

The failures are not independent — they form a causal chain. Understanding the
order in which they manifest is essential for prioritizing fixes:

```
SHOWSTOPPER #1 (§6: 63 .cpp files, 0 guards, unguarded cucascade includes)
  ├── BLOCKS the entire build at #include resolution
  ├── MASKS cudaProfilerStart/Stop link error (never reached)
  ├── MASKS cudaMemPool* issue in defragmenter_oom_policy.cpp (never reached)
  └── MASKS all at-risk symbols in .cpp files (cudaDevAttrL2CacheSize, etc.)

SHOWSTOPPER #2 (§7: pixi/vcpkg 100% NVIDIA, no ROCm package env)
  ├── find_package(cudf) resolves to NVIDIA cudf (name collision)
  └── INDEPENDENT of #1 — both must be fixed for an end-to-end build

IF both showstoppers resolved, then (in order):
  ├── fsst.cuh __shfl_xor_sync/__syncwarp     → 2 .cu kernels fail (dict_fsst, fsst)
  ├── shared_staging.cuh cooperative_groups   → 2 .cu kernels fail (alp, bitpacking)
  ├── cudaProfilerStart/Stop extern "C"        → link error in sirius_extension
  └── 5 truly clean kernels: rle, strings, native_decode, dictionary, uncompressed
```

The `#ifdef`-guard approach is infeasible for cuCascade (§6.3: inheritance +
public API signatures), so showstopper #1 requires either a cuCascade HIP port
or a ROCm-native replacement subsystem.

## 11. Verification State

This port was developed on Apple Silicon (arm64) with no AMD GPU and no
x86_64 ROCm toolchain. A Linux VM (Colima) was started but ROCm's gfx90a
toolchain is not packaged for arm64. The work is static, verified by
inspection and by two independent line-by-line review agents (each read every
line of all relevant files). It has **not** been compiled or run.
